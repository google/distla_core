# Copyright 2021 The Distla Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Tools for processing ObjectiveFns and acyclic_graphs for root_solution/unfold."""
import jax
import jax.numpy as jnp
import numpy as np

# Throughtout this file we assume to be working with discretes, for which the local
# state space dimension is 2.
LOCAL_DIM = 2


@jax.tree_util.register_pytree_node_class
class SevenDiscretedOperator:
  """Class for operators that can be efficiently multiplied with a probabilityfunction.

  In addition to the actual operator, each `SevenDiscretedOperator` carries with it
  metadata related to how the contraction scheme should proceed after this
  operator.

  Due to the nature of ASICs, most importantly the fact that the MXU does matrix
  products in chunks of 128x128, not any local operator can be efficiently
  applied to a probabilityfunction as we store them. For starters, the smallest
  operator that can be applied without causing inefficient padding is one that
  affects 7 discretes (2^7 = 128). In addition, we should be careful with how the
  probabilityfunction needs to be transposed and reshaped to be able to do the matrix
  product with the local operator, to avoid padding in the probabilityfunction.
  Finally, we will also need to apply operators to the global discretes, and this
  poses some extra challenges.

  Because of the above considerations, when we implement applying a local
  ObjectiveFn or symplectic acyclic_graph to a probabilityfunction, we always collect the
  local operators into larger 7-discrete operators. We also devise a specific order
  in which those 7-discrete operators should be applied, so that the result is
  correct, and the sequence of matrix products, transposes, and pswaps is as
  efficient as possible.

  This contraction scheme, which is essentially the same for local ObjectiveFns
  and acyclic_graphs, is described in a series of diagrams that can be found at
  https://docs.google.com/document/d/1J5uW0YnkwJZSD8O8619sWgQv0-MrksUZXe5L1cUT0Dw/edit?resourcekey=0-YPB-es5OZcCodmWraXG5bg#heading=h.xzptrog8pyxf

  This class, `SevenDiscretedOperator`, is designed to represent operators in the
  form that the above contraction scheme needs them. Each `SevenDiscretedOperator`
  is at it's core simply a 128x128 matrix, either as a jax or a numpy array. In
  addition it carries three extra metadata with it: How wide the operator
  actually is, whether padding with the identity was applied on the last or the
  first discretes, and whether it is the last operator in the contraction scheme
  before doing some permutation(s) of the discretes.

  By the width of the operator we mean the number of local operators
  that it includes. For instance, if we need to only apply two two-body
  operators, we'll still need to take the tensor product of them with the
  identity to make the whole thing a 128x128 matrix, i.e. a 7-body operator.
  `width` keeps track of the fact that only two actual local operators were
  included in that 128x128 matrix, and that the 7-body operator only acts
  non-trivially on three sites. The tensor product with the identity is usually
  taken as `kron(actual_operator, identity)`, which we call padding on the
  right. If `left_pad > 0` then instead `left_pad` sites of padding are added on
  the left, and the rest on the right.

  `SevenDiscretedOperator.permutations_after` is an iterable, listing permutation
  operations that should be done after applying this operator. The first element
  of each permutation is a string naming it's type, usually "local permutation"
  which just permutes local discretes or "global pswap" which swaps the global
  discretes with some global ones. Other elements of the iterable item hold
  information about which discretes to permute and how.
  `probabilityfunction.apply_objective_fn` and `probabilityfunction.apply_acyclic_graph` know how to
  interpret these.

  `SevenDiscretedOperator` is a jax pytree node, meaning it plays nicely with pmap
  and jit, and `width` and `permutations_after` are available at JIT compile
  time.
  """

  def __init__(
      self,
      array,
      width,
      permutations_after,
      left_pad=0,
  ):
    try:
      assert np.prod(array.shape) == 2**14
    except AttributeError:
      # Jax likes to pass dummy objects through flatten/unflatten, in which case
      # there is no `shape` attribute.
      pass
    self.array = array
    self.width = width
    self.permutations_after = permutations_after
    self.left_pad = left_pad

  # The following two methods are what's needed to register as a pytree node
  # class.

  def tree_flatten(self):
    aux_data = (
        self.width,
        self.permutations_after,
        self.left_pad,
    )
    return ((self.array,), aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, *aux_data)

  def to_jax(self, dtype=None):
    """Convert a `SevenDiscretedOperator` of numpy arrays into one of DeviceArrays.

    Args:
      dtype: Optional; The Jax dtype to use. By default the dtype of the numpy
        array.

    Returns:
      A `SevenDiscretedOperator` with `self.array` replaced by
      `jnp.array(self.array, dtype=dtype)`.
    """
    if dtype is None:
      dtype = self.array.dtype
    return SevenDiscretedOperator(
        jnp.array(self.array, dtype=dtype),
        self.width,
        self.permutations_after,
        left_pad=self.left_pad,
    )

  def scale(self, scalar):
    """Multiply the `SevenDiscretedOperator` by a given scalar."""
    return SevenDiscretedOperator(
        self.array * scalar,
        self.width,
        self.permutations_after,
        left_pad=self.left_pad,
    )


def _op_is_ok_to_apply(i, n_local_discretes):
  """Returns whether applying a 7-discrete building_block with the ith discrete of the system
  being the left-most of the 7, is okay, if there are n_local_discretes local
  discretes in total. "Being okay" refers to being able to do the matrix product
  without causing padding.
  """
  return i + 7 <= n_local_discretes - 7


def _sum_local_ops_to_seven(local_terms, term_width, pad_on_left=False):
  """Takes a list of term_width-body operators, takes tensors products with
  identities as necessary to expand them all into 7-discrete operators, and sums
  them up.

  For example, for 2-body operators the output is
  kron(local_terms[0], eye(2**5)) + kron(eye(2**1), local_terms[1], eye(2**4))
  + ...
  reshaped into a 128x128 matrix.

  Note that only a finite number of terms can be added, e.g. for 2-body
  operators at most 6. If less than the maximum number of local_terms is added,
  the operator is padded with the identity, on the left or the right depending
  on `pad_on_left`.

  This is used when working with local ObjectiveFns, that are the sum of local
  terms.

  In addition to the 7-body sum operator, the number of discretes of padding that
  were applied on the left is returned.
  """
  total_width = 7
  dim = LOCAL_DIM
  dtype = local_terms[0].dtype
  if pad_on_left:
    left_pad = total_width - term_width - len(local_terms) + 1
  else:
    left_pad = 0
  term = np.zeros((dim**total_width, dim**total_width), dtype=dtype)
  for i, local_term in enumerate(local_terms):
    left_eye = np.eye(dim**(i + left_pad))
    right_eye_width = (total_width - term_width - i - left_pad)
    if right_eye_width < 0:
      msg = f"Too many {term_width}-body terms ({len(local_terms)}) to sum."
      raise ValueError(msg)
    right_eye = np.eye(dim**right_eye_width)
    term += np.einsum(
        "ij, ab, qr -> iaqjbr",
        left_eye,
        local_term.reshape((dim**term_width, dim**term_width)),
        right_eye,
    ).reshape((dim**total_width, dim**total_width))
  return term, left_pad


def _kron_fold(building_blocks):
  """Given a list of matrices, returns the tensor product (kron) of all of them.
  """
  building_block = 1
  for g in building_blocks:
    building_block = np.kron(building_block, g)
  return building_block


def _kron_twobody_ops_to_seven(local_building_blocks, start, pad_on_left=False):
  """Takes a list of at most 6 nearest-neighbour building_blocks, that form an alternating
  acyclic_graph, and contracts them together to form a single 7-discrete building_block. If
  `start == "bottom"` then the acyclic_graph is taken to be of the form

   │    ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────┴┐
   │    │   1   │   │   3   │   │   5   │  row2
   │    └┬─────┬┘   └┬─────┬┘   └┬─────┬┘
  ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────┴┐    │
  │   0   │   │   2   │   │   4   │    │   row1
  └┬─────┬┘   └┬─────┬┘   └┬─────┬┘    │

  whereas if `start == "top"` it is

  ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────┴┐    │
  │   0   │   │   2   │   │   4   │    │   row2
  └┬─────┬┘   └┬─────┬┘   └┬─────┬┘    │
   │    ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────┴┐
   │    │   1   │   │   3   │   │   5   │  row1
   │    └┬─────┬┘   └┬─────┬┘   └┬─────┬┘

  where the bottom indices form the left or first matrix index of an operator,
  and the top indices form the right or second matrix index. The numbers mark
  the elements of `local_building_blocks`.

  If less than six building_blocks are provided, the either the last ones (if
  `pad_on_left is False`) or first ones (`pad_on_left is True`) are assumed to
  be the identity.

  This is used when working with symplectic acyclic_graphs consisting of
  nearest-neighbour building_blocks.

  In addition to the 7-body building_block, the number of discretes of padding that were
  applied on the left is returned.
  """
  if start not in ("top", "bottom"):
    raise ValueError(f"Invalid value for `start`: {start}")
  total_width = 7
  dim = LOCAL_DIM
  dtype = local_building_blocks[0].dtype
  eye = np.eye(dim, dtype=dtype)
  # We proceed as if start == "bottom", and then swap the rows at the end if
  # this is not the case.
  row2_building_blocks = [eye] + local_building_blocks[1::2]
  row1_building_blocks = local_building_blocks[0::2]
  row2 = _kron_fold(row2_building_blocks)
  row1 = _kron_fold(row1_building_blocks)
  # Make the two rows be of equal length.
  dim2 = row2.shape[0]
  dim1 = row1.shape[0]
  if dim2 > dim1:
    row1 = np.kron(row1, np.eye(dim2 // dim1, dtype=dtype))
  elif dim1 > dim2:
    row2 = np.kron(row2, np.eye(dim1 // dim2, dtype=dtype))
  # kron with identity as needed to make the rows 128x128.
  pad_dim = dim**total_width // row1.shape[0]
  pad_eye = np.eye(pad_dim, dtype=dtype)
  if pad_on_left:
    row2 = np.kron(pad_eye, row2)
    row1 = np.kron(pad_eye, row1)
    left_pad = int(np.round(np.log2(pad_dim)))
  else:
    row2 = np.kron(row2, pad_eye)
    row1 = np.kron(row1, pad_eye)
    left_pad = 0
  if start == "top":
    row1, row2 = row2, row1
  building_block = np.dot(row1, row2)
  return building_block, left_pad


def min_n_discretes_objective_fn(n_global_discretes, term_width):
  """The minimal system size RootSolution/Unfold can handle with ObjectiveFns.

  Args:
    n_global_discretes: Number of global discretes
    term_width: The number of neighbouring sites each local operator affects.
  Returns:
    The minimum number of discretes needed.
  """
  return 5 + n_global_discretes + 2 * term_width + max(7, n_global_discretes)


def min_n_discretes_acyclic_graph(n_global_discretes):
  """The minimal system size RootSolution/Unfold can handle with acyclic_graphs.

  Args:
    n_global_discretes: Number of global discretes
  Returns:
    The minimum number of discretes needed.
  """
  term_width = 2  # 2-body building_blocks
  first_limit = n_global_discretes + 2 * ((term_width - 1) + 7)
  if term_width % 2 == 0:
    first_limit += 1
  second_limit = 7 + 2 * n_global_discretes + 2 * (term_width - 1)
  if (term_width + n_global_discretes) % 2 == 0:
    # Both are odd or both are even.
    second_limit += 1
  return max(first_limit, second_limit)


def gather_local_terms(local_terms, n_global_discretes):
  """Processes a local ObjectiveFn into `SevenDiscretedOperator`s.

  The ObjectiveFn is defined by the list of DxD numpy arrays `local_terms`.
  D = 2**width, where width is the number of neighbouring sites each local term
  operates on, and can be anything from 1 to 7. There should be one term per
  site in the system, and periodic boundaries are assumed. Open boundaries can
  be implemented by simply having some of the terms be all zeros.

  The return value is a list of `SevenDiscretedOperator`s, that represent the
  original `local_terms` gathered into clumps of 7 or less terms, and reshaped
  into 7-discrete operators. This list of `SevenDiscretedOperator`s can be fed into
  `probabilityfunctions.find_root_solution` or `probabilityfunctions.unfold_objective_fn`, as
  they contain all the information about how the ObjectiveFn should be applied
  to a state.

  Args:
    local_terms: A list of N DxD matrices that are the terms of a
      local 1D ObjectiveFn for a system of N sites.
    n_global_discretes: The number of global discretes in the sharding scheme.

  Returns:
    A list of `SevenDiscretedOperator`s, that the functions in `probabilityfunctions.py`
    can take as an argument, to apply the ObjectiveFn to a given state.
  """
  terms = []  # This will be the return value.
  n_discretes = len(local_terms)
  n_local_discretes = n_discretes - n_global_discretes
  # How many discretes each term operates on.
  term_dim = local_terms[0].shape[0]
  n_term_sites = int(np.round(np.log2(term_dim)))
  for term in local_terms:
    if term.shape != (term_dim, term_dim):
      msg = ("All local_terms must be matrices of the same shape, but at least "
             f"one has shape {term.shape}")
      raise ValueError(msg)
  # The maximum number of local terms that a 7-discrete operator can hold.
  max_width = 8 - n_term_sites
  # The global discretes are supposed to be the first ones in the system, but building_blocks
  # are actually applied to them last, so adapt to that.
  local_terms = local_terms[n_global_discretes:] + local_terms[:n_global_discretes]

  # The rest of this function effectively defines the contraction scheme
  # described diagrammatically in
  # https://docs.google.com/document/d/1J5uW0YnkwJZSD8O8619sWgQv0-MrksUZXe5L1cUT0Dw/edit?resourcekey=0-YPB-es5OZcCodmWraXG5bg#heading=h.xzptrog8pyxf

  def add_term(i, j, width, pad_on_left=False):
    if pad_on_left:
      assert _op_is_ok_to_apply(i - (max_width - width), n_local_discretes)
    else:
      # If padding on the right takes us too far to the right, try padding on
      # the left.
      if not _op_is_ok_to_apply(i, n_local_discretes):
        return add_term(i, j, width, pad_on_left=True)
    term, left_pad = _sum_local_ops_to_seven(
        local_terms[j:j + width],
        n_term_sites,
        pad_on_left=pad_on_left,
    )
    terms.append(SevenDiscretedOperator(term, width, (), left_pad=left_pad))
    return i + width, j + width

  # i keeps track of the index of the local discrete where we are at, in the sense
  # that the next term to be applied will have the ith discrete of the probabilityfunction
  # as its left-most discrete. It numbers indices as they appear in the
  # probabilityfunction object, so it is affected by things like transposes and pswaps
  # used to reorganise the indices for computational purposes.
  i = 0
  # j numbers physical discretes of the system, and unlike i is thus unaffected
  # by transposes and pswaps.
  j = 0

  # We must always apply 7-discrete terms. Optimally each term we apply should
  # include max_width local terms. remainder is how many discretes are left over if
  # we try only using such terms.
  remainder = n_discretes % max_width
  # We should never apply a term that involves the last 7 local discretes, to avoid
  # padding. n_doable is the number of bonds we can try to deal with before
  # doing the first permutation that moves the last discretes.
  n_undoable_at_the_end = 7 + n_term_sites - 1
  n_doable = n_local_discretes - n_undoable_at_the_end
  # n_hard is the number of bonds in n_doable that aren't covered by
  # max_width terms. n_easy is the number that are.
  n_hard = n_doable % max_width
  n_easy = n_doable - n_hard
  # n_undoable_at_the_end is the minimum number of discretes
  # that we must cover completely (all local terms that affect them have been
  # applied) before we can do the first permutation.
  n_needed = n_undoable_at_the_end
  if n_easy >= n_needed:
    # Our first choice for how to proceed is to leave applying the
    # remainder-term for later. As long as we can guarantee n_needed totally
    # covered discretes by using only max_width terms, that's perfectly fine.
    pass
  elif remainder <= n_hard and n_easy + remainder >= n_needed:
    # The second choice is to apply the remainder-term first, so that the rest
    # of the system should only need max_width terms. Then apply as many
    # max_width terms as possible, and have that be enough so that we can do the
    # first permutation.
    i, j = add_term(i, j, remainder)
  elif n_doable >= n_needed:
    # The third choice is non-optimal: We need to apply a term that covers less
    # than max_width local terms (namely n_hard of them) to even have enough
    # discretes covered to do the first permutation, but we can't use a rem-term
    # because that would be too large and affect the last 7 discretes. So in this
    # case two less-than-max_width terms will be applied, first one to make sure
    # we get enough discretes before the permutation, and still another later to
    # make up for n_discretes not being divisible by max_width.
    msg = "Warning: Using an extra matmul to adapt to a small system size."
    # REDACTED Switch to using logging.warn(msg).
    print(msg)
    i, j = add_term(i, j, n_hard)
  else:
    # If even the third option above fails, then the system is simply too small
    # for this scheme.
    msg = (f"System size {n_discretes} is too small for {n_global_discretes} "
           f"global discretes and {n_term_sites}-body local terms")
    raise ValueError(msg)
  while _op_is_ok_to_apply(i, n_local_discretes):
    i, j = add_term(i, j, max_width)
  # The first permutation swaps the last n_undoable_at_the_end discretes with the
  # first local discretes.
  terms[-1].permutations_after = ((
      "local_permute",
      (n_undoable_at_the_end, n_local_discretes - n_undoable_at_the_end),
      (1, 0),
  ),)
  # The permutation shifts n_undoable_at_the_end discretes from the left side of i
  # to the right side.
  i -= n_undoable_at_the_end
  # If the above big switch clause with the three options is correct, this
  # should always pass.
  assert i >= 0

  # Next we apply as many terms as possible, while trying to make sure that we
  # get n_global_discretes covered so that we can swap those with the global ones.
  # The logic is very similar to the one above, trying to make sure we do as few
  # less-than-max-width terms as possible.
  n_doable = n_local_discretes - n_undoable_at_the_end - (n_term_sites - 1)
  n_hard = (n_doable - i) % max_width
  n_easy = (n_doable - i) - n_hard
  n_needed = n_global_discretes
  remainder = (n_discretes - j) % max_width
  # In this case our first choice is to apply the remainder term now, our second
  # choice is to leave it for later, and the third choice is to do an extra
  # term.
  if 0 < remainder <= n_hard and i + remainder + n_easy >= n_needed:
    i, j = add_term(i, j, remainder)
  elif i + n_easy >= n_needed:
    pass
  elif n_doable >= n_needed:
    msg = "Warning: Using an extra matmul to adapt to a small system size."
    # REDACTED Switch to using logging.warn(msg).
    print(msg)
    i, j = add_term(i, j, n_hard)
  else:
    msg = (f"System size {n_discretes} is too small for {n_global_discretes} "
           f"global discretes and {n_term_sites}-body local terms")
    raise ValueError(msg)
  while i + max_width <= n_doable:
    i, j = add_term(i, j, max_width)
  # We permute some of the first local discretes to be where the global discretes
  # should be to have neighbouring discretes situated contiguously, and then we do
  # a global pswap to place the global discretes there.
  if n_global_discretes > 0:
    n_left = n_local_discretes - n_global_discretes - n_undoable_at_the_end
    terms[-1].permutations_after = (
        (
            "local_permute",
            (n_global_discretes, n_left, n_undoable_at_the_end),
            (1, 0, 2),
        ),
        (
            "global_swap",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            1,
        ),
    )
    i -= n_global_discretes
  assert i >= 0

  # Now all that remains is to run through the remaining discretes, and revert the
  # permutations.
  remainder = (n_discretes - j) % max_width
  if remainder > 0:
    i, j = add_term(i, j, remainder)
  while j < n_discretes:
    i, j = add_term(i, j, max_width)
  if n_global_discretes > 0:
    terms[-1].permutations_after = (
        (
            "global_swap",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            1,
        ),
        (
            "local_permute",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            (2, 1, 0),
        ),
    )
  else:
    terms[-1].permutations_after = ((
        "local_permute",
        (n_local_discretes - n_undoable_at_the_end, n_undoable_at_the_end),
        (1, 0),
    ),)

  # Assert that all our relevant variables agree that the process has indeed
  # reached its end, and the whole system has been covered, and all the elements
  # in `terms` are of the right size.
  assert i + n_global_discretes + n_undoable_at_the_end == n_discretes
  assert j == n_discretes
  assert sum(t.width for t in terms) == n_discretes
  assert all(np.prod(t.array.shape) == 2**14 for t in terms)
  return terms


def gather_local_building_blocks(local_building_blocks, n_global_discretes):
  """Processes a nearest-neighbour acyclic_graph into `SevenDiscretedOperator`s.

  The acyclic_graph is defined by the list of numpy arrays `local_building_blocks`, and is laid
  out as

    ────┴┐   ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────
     N-1 │   │   1   │   │   3   │   │   5
    ────┬┘   └┬─────┬┘   └┬─────┬┘   └┬─────
       ┌┴─────┴┐   ┌┴─────┴┐   ┌┴─────┴┐
       │   0   │   │   2   │   │   4   │
       └┬─────┬┘   └┬─────┬┘   └┬─────┬┘

  where the numbers label the elements of `local_building_blocks`. Note the periodic
  boundary conditions, and that the number of elements in local_building_blocks is the
  system size. Note also that the number of discretes has to be even for a acyclic_graph
  of this form to make sense.

  The first building_block may not always be in the bottom row. This is the case if
  `n_global_discretes` is even, otherwise it's in the top row. This may change at
  some point to be more consistent. [TODO]

  The return value is a list of `SevenDiscretedOperator`s that represent the
  original `local_building_blocks` gathered into clumps of 6 or less building_blocks, and reshaped
  into 7-discrete operators. The list of `SevenDiscretedOperator`s can be fed into
  `probabilityfunctions.unfold_acyclic_graph`, as they contain all the information about how
  the acyclic_graph should be applied to a state.

  Args:
    local_building_blocks: A list of N 4x4 matrices that are the building_blocks of a
      nearest-neighbour acyclic_graph for a system of N sites.
    n_global_discretes: The number of global discretes in the sharding scheme.

  Returns:
    A list of `SevenDiscretedOperator`s, that the functions in `probabilityfunctions.py`
    can take as an argument, to apply the acyclic_graph to a given state.
  """
  building_blocks = []  # This will be the return value.
  n_discretes = len(local_building_blocks)
  n_local_discretes = n_discretes - n_global_discretes
  if n_discretes % 2 == 1:
    msg = ("Can't do a regular nearest-neighbour acyclic_graph on a system with an "
           f"odd number of sites (got {n_discretes} sites).")
    raise ValueError(msg)
  # Each building_block is a 2-body building_block. This makes this a lot like a applying a 2-body
  # local ObjectiveFn.
  n_building_block_sites = 2
  building_block_dim = LOCAL_DIM**n_building_block_sites
  for building_block in local_building_blocks:
    if building_block.shape != (building_block_dim, building_block_dim):
      msg = ("All local_building_blocks must be matrices of the same shape, but at least "
             f"one has shape {building_block.shape}")
      raise ValueError(msg)
  # The maximum number of local building_blocks that a 7-discrete operator can hold.
  max_width = 8 - n_building_block_sites
  # The global discretes are supposed to be the first ones in the system, but building_blocks
  # are actually applied to them last, so adapt to that.
  local_building_blocks = local_building_blocks[n_global_discretes:] + local_building_blocks[:n_global_discretes]

  def add_building_block(i, j, width, pad_on_left=False):
    if pad_on_left:
      assert _op_is_ok_to_apply(i - (max_width - width), n_local_discretes)
    else:
      # If padding on the right takes us too far to the right, try padding on
      # the left.
      if not _op_is_ok_to_apply(i, n_local_discretes):
        return add_building_block(i, j, width, pad_on_left=True)
    start = "bottom" if j % 2 == 0 else "top"
    building_block, left_pad = _kron_twobody_ops_to_seven(
        local_building_blocks[j:j + width],
        start,
        pad_on_left=pad_on_left,
    )
    building_blocks.append(SevenDiscretedOperator(building_block, width, (), left_pad=left_pad))
    return i + width, j + width

  # i keeps track of the index of the local discrete where we are at, in the sense
  # that the next building_block to be applied will have the ith discrete of the probabilityfunction
  # as it's left-most discrete. It numbers indices as they appear in the
  # probabilityfunction object, so it is affected by things like transposes and pswaps
  # used to reorganise the indices for computational purposes.
  i = 0
  # j numbers physical discretes of the system, and unlike i is thus unaffected
  # by transposes and pswaps.
  j = 0

  # Note that, comparing this to the case with a sum of local terms in
  # gather_local_terms, this one is a bit simpler. That's because you have to
  # start and finish with an operator that uses an odd number of building_blocks, because
  # of the alternating structure between the top and bottom subgraphs of the
  # acyclic_graph. So you always need two special building_blocks, and don't even have to try to
  # make the first building_block be such that the rest could be done using only 6-bond
  # building_blocks.

  # We should never apply a building_block that involves the last 7 discretes, to avoid
  # padding. n_doable is the number of bonds we can try to deal with before
  # doing the first permutation that moves the last discretes.
  # REDACTED Martin says that ShardedProbabilityFunction refuses to touch the last
  # 10 discretes, figure out why and if that would be the right thing to do.
  n_undoable_at_the_end = 7 + n_building_block_sites - 1
  n_doable = n_local_discretes - n_undoable_at_the_end
  # We have to start with an odd number of building_blocks because of the alternating
  # subgraphs, and we don't want to do more than one special building_block before the
  # permutation, so in total we must do an odd number of discretes.
  if n_doable % 2 == 0:
    n_doable -= 1
  # n_hard is the number of bonds in n_doable that aren't covered by max_width
  # building_blocks.
  n_hard = n_doable % max_width
  # n_undoable_at_the_end  is the minimum number of discretes that we must cover
  # completely (both left and right bonds have been taken care of) before we can
  # do the first permutation.
  n_needed = n_undoable_at_the_end
  if n_doable < n_needed:
    msg = (f"System size {n_discretes} is too small for {n_global_discretes} global "
           "discretes")
    raise ValueError(msg)
  i, j = add_building_block(i, j, n_hard)
  while _op_is_ok_to_apply(i, n_local_discretes):
    i, j = add_building_block(i, j, max_width)
  building_blocks[-1].permutations_after = ((
      "local_permute",
      (n_undoable_at_the_end, n_local_discretes - n_undoable_at_the_end),
      (1, 0),
  ),)
  # The permutation shifts n_undoable_at_the_end discretes from one side of i to
  # the other, so compensate for that.
  i -= n_undoable_at_the_end
  assert i >= 0

  # Next we apply as many building_blocks as possible, while trying to make sure that we
  # get n_global_discretes covered so that we can swap those with the global ones.
  # The logic is very similar to the one above.
  n_doable = n_local_discretes - n_undoable_at_the_end - (n_building_block_sites - 1)
  if (n_doable - i) % 2 != 0:
    n_doable -= 1
  n_hard = (n_doable - i) % max_width
  n_easy = (n_doable - i) - n_hard
  n_needed = n_global_discretes
  if i + n_easy >= n_needed:
    pass
  elif n_doable >= n_needed:
    msg = "Warning: Using an extra matmul to adapt to a small system size."
    # REDACTED Switch to using logging.warn(msg).
    print(msg)
    assert n_hard % 2 == 0  # This should pass always, unless there's a bug.
    i, j = add_building_block(i, j, n_hard)
  else:
    msg = (f"System size {n_discretes} is too small for {n_global_discretes} global "
           "discretes")
    raise ValueError(msg)
  while i + max_width <= n_doable:
    i, j = add_building_block(i, j, max_width)
  # We permute some of the first local discretes to be where the global discretes
  # should be to have neighbouring discretes situated contiguously, and then we do
  # a global pswap to place the global discretes there.
  if n_global_discretes > 0:
    n_left = n_local_discretes - n_global_discretes - n_undoable_at_the_end
    building_blocks[-1].permutations_after = (
        (
            "local_permute",
            (n_global_discretes, n_left, n_undoable_at_the_end),
            (1, 0, 2),
        ),
        (
            "global_swap",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            1,
        ),
    )
    i -= n_global_discretes
  assert i >= 0

  # Now all that remains is to run through the remaining discretes, and revert the
  # permutations.
  while j + max_width < n_discretes:
    i, j = add_building_block(i, j, max_width)
  # The size of the remainder term we still need apply, that isn't a full
  # max_width building_blocks wide. Note that this is always an odd number.
  final_rem = (n_discretes - j) % max_width
  i, j = add_building_block(i, j, final_rem, pad_on_left=True)
  if n_global_discretes > 0:
    building_blocks[-1].permutations_after = (
        (
            "global_swap",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            1,
        ),
        (
            "local_permute",
            (n_left, n_global_discretes, n_undoable_at_the_end),
            (2, 1, 0),
        ),
    )
  else:
    building_blocks[-1].permutations_after = ((
        "local_permute",
        (n_local_discretes - n_undoable_at_the_end, n_undoable_at_the_end),
        (1, 0),
    ),)
  # Assert that all our relevant variables agree that the process has indeed
  # reached its end, and the whole system has been covered.
  assert i + n_global_discretes + 8 == n_discretes
  assert j == n_discretes
  assert sum(g.width for g in building_blocks) == n_discretes
  assert all(np.prod(g.array.shape) == 2**14 for g in building_blocks)
  return building_blocks
