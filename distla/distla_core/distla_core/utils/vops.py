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
"""Functions to manipulate vectors and thin matrices.

They must be distinguished from matrices that are closer to square in the
distributed setting, since they demand a different distribution pattern.

Like normal distributed matrices, DistlaCore thin matrices are represented by
ShardedDeviceArrays with three indices, the first being the sharded index, and
the other two specifying the dimensions of the local panels. In addition, they
are wrapped in a ReplicatedThinMatrix object, which, in addition to the actual
array, has as its attribute a boolean is_column_replicated.

Thus we distinguish between two different kinds of thin matrices.
"Column-replicated" means is_column_replicated=True, and repeat data as one
traverses over pcols. "Row-replicated" means is_column_replicated=False, and
repeat data as one traverses over prows.

This is clarified in the following depiction, which shows the local elements of
a vector v, first undistributed, then column-replicated on a 4 x 2 processor
grid, then row-replicated on the same:

v (undistributed) = [1, 2, 3, 4, 5, 6, 7, 8]
v (column-replicated)  = [[1, 2], [1, 2]
                          [3, 4], [3, 4]
                          [5, 6], [5, 6]
                          [7, 8], [7, 8]]

v (row-replicated)  = [[1, 2, 3, 4], [5, 6, 7, 8]
                       [1, 2, 3, 4], [5, 6, 7, 8]
                       [1, 2, 3, 4], [5, 6, 7, 8]
                       [1, 2, 3, 4], [5, 6, 7, 8]]

Notice the local size and thus the memory footprint differs between the
two strategies in the case of a rectangular processor grid. Note also we've
used a plain vector above for illustration, but in reality this would be stored
as a (8, 1) thin matrix.

The motivation here is to simplify matrix-vector multiplication. For example,
consider the product of a block 2x2 matrix A with a block-2 vector v. We have

[A00 A01] @ [v0] = [A00@v0 + A01@v1]
[A10 A11]   [v1]   [A10@v0 + A11@v1]

Given that the matrix A is distributed in the checkerboard fashion used by
DistlaCore, on a 2x2 grid the necessary local components of A and v each reside on
the same processor only if v is row-replicated. The matrix-vector product can
then be achieved by first having each process perform A @ v locally, and then
gathering terms by a parallel sum reduction over processor columns. The result
is a *column*-replicated vector A @ v.

If, on the other hand, the product A.T @ v is desired, the same holds only if v
is column-replicated. Each process performs A.T @ v locally, and then gathers
terms by a parallel sum reduction over processor rows, resulting in a
*row*-replicated vector A.T @ v.

Due to the MXU units on a ASIC operating on 128x128 blocks, it makes sense to do
a matrix-vector product like the above for e.g. 128 vectors. For this purpose
you can replace the vector v in the above example with a matrix of shape (D,
128). We call these matrices thin because typically D >> 128, but this does not
have to be the case.

For quick reference the shape convention is summarized here:

"Column-replicated": - shape (NPROC, n_rows / n_processor_rows, n_columns)
                     - each pcol has a copy.
                     - adapted to A.T @ x.
                     - the result of A.T @ x is row-replicated.
"Row-replicated": - shape (NPROC, n_rows / n_processors_cols, n_columns)
                  - each prow has a copy.
                  - adapted to A @ x.
                  - the result of A @ x is column-replicated.

"""
import operator
import time

import jax
from jax import lax
import jax.numpy as jnp

from distla_core.utils import misc
from distla_core.utils import pops


def _generate_binary_deferer(op_func):
  """
  Given a binary operator, generate a method that applies that operator
  element-wise to a self and an other. See
  ReplicatedThinMatrices._defer_binary_elementwise for more.
  """

  def deferer(self, other, *args, **kwargs):
    return type(self)._defer_binary_elementwise(
        self,
        other,
        op_func,
        *args,
        **kwargs,
    )

  return deferer


def _generate_unary_deferer(op_func):
  """
  Given a unary operator, generate a method that applies that operator
  element-wise on a self and an other. See
  ReplicatedThinMatrix._defer_unary_elementwise for more.
  """

  def deferer(self, *args, **kwargs):
    return type(self)._defer_unary_elementwise(self, op_func, *args, **kwargs)

  return deferer


def _swap_first_args(op):
  """
  Given a binary operator function, return a function that applies it
  but with argument order swapped for the first two arguments.
  """

  def op_swapped(a, b, *args, **kwargs):
    return op(b, a, *args, **kwargs)

  return op_swapped


@jax.tree_util.register_pytree_node_class
class ReplicatedThinMatrix:
  """
  A class for column and row-replicated thin matrices.

  This class is registered as a jax pytree node class.

  Attributes:
    array: DeviceArray or ShardedDeviceArray holding the data
    is_column_replicated: Boolean for whether the matrix is column or
      row-replicated.
    dtype: dtype of array
    shape: shape of array
  """

  def __init__(self, array, is_column_replicated):
    self.array = array
    self.is_column_replicated = is_column_replicated

  # The following two methods are what's needed to register as a pytree node
  # class.

  def tree_flatten(self):
    return ((self.array,), self.is_column_replicated)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children, aux_data)

  # We want all kinds of basic arithmetic operations, comparison operators, and
  # functions like abs and sqrt to be able to operate element-wise on
  # ReplicatedThinMatrices. For that purpose we define two methods that simply
  # pass these operations to the .array attribute, one for binary operators
  # and one for unary operators, and use those to generate all the particular
  # operations we need.
  #
  # This code follows design of
  # https://github.com/mhauru/abeliantensors
  # but that library is MIT licensed and the author of this code (mhauru@) owns
  # the copyright, so that shouldn't be an issue.

  def _defer_binary_elementwise(self, other, op_func, *args, **kwargs):
    """
    Element-wise binary operators like + and / as methods for
    ReplicatedThinMatrix.

    If both self and other are ReplicatedThinMatrices, then their arrays are
    operated on pair-wise with op_func(_, _, *args, **kwargs), but only after
    making sure other is replicated in the same way as self.  If other is not a
    ReplicatedThinMatrix then self.array is operated on with
    op_func(_, other, *args, **kwargs).

    The operation is never in-place, and returns a a new ReplicatedThinMatrix,
    with the same replication pattern (row/column) as self.

    This method can be used to create element-wise binary operations, such as
    basic arithmetic and comparisons.
    """
    if not isinstance(other, ReplicatedThinMatrix):
      arr = op_func(self.array, other, *args, **kwargs)
    else:
      if self.is_column_replicated:
        other = to_column_replicated(other)
      else:
        other = to_row_replicated(other)
      array_l = self.array
      array_r = other.array
      if array_l.ndim == 1:
        array_l = array_l.reshape((array_l.size, 1))
      if array_r.ndim == 1:
        array_r = array_r.reshape((array_r.size, 1))
      arr = op_func(self.array, other.array, *args, **kwargs)
    return ReplicatedThinMatrix(arr, self.is_column_replicated)

  def _defer_unary_elementwise(self, op_func, *args, **kwargs):
    """
    Produces a new ReplicatedThinMatrix that is like self, but with
    array = op_func(self.array, *args, **kwargs).

    This method can be used to create basic element-wise unary operations, such
    as negation and element-wise absolute value.
    """
    return ReplicatedThinMatrix(
        op_func(self.array, *args, **kwargs),
        self.is_column_replicated,
    )

  __add__ = _generate_binary_deferer(operator.add)
  __sub__ = _generate_binary_deferer(operator.sub)
  __mul__ = _generate_binary_deferer(operator.mul)
  __truediv__ = _generate_binary_deferer(operator.truediv)
  __floordiv__ = _generate_binary_deferer(operator.floordiv)
  __mod__ = _generate_binary_deferer(operator.mod)
  __pow__ = _generate_binary_deferer(pow)

  __radd__ = _generate_binary_deferer(_swap_first_args(operator.add))
  __rsub__ = _generate_binary_deferer(_swap_first_args(operator.sub))
  __rmul__ = _generate_binary_deferer(_swap_first_args(operator.mul))
  __rtruediv__ = _generate_binary_deferer(_swap_first_args(operator.truediv))
  __rfloordiv__ = _generate_binary_deferer(_swap_first_args(operator.floordiv))
  __rmod__ = _generate_binary_deferer(_swap_first_args(operator.mod))
  __rpow__ = _generate_binary_deferer(_swap_first_args(pow))

  __eq__ = _generate_binary_deferer(operator.eq)
  __ne__ = _generate_binary_deferer(operator.ne)
  __lt__ = _generate_binary_deferer(operator.lt)
  __le__ = _generate_binary_deferer(operator.le)
  __gt__ = _generate_binary_deferer(operator.gt)
  __ge__ = _generate_binary_deferer(operator.ge)

  __neg__ = _generate_unary_deferer(operator.neg)
  __pos__ = _generate_unary_deferer(operator.pos)
  __abs__ = _generate_unary_deferer(abs)

  conj = _generate_unary_deferer(jnp.conj)
  sqrt = _generate_unary_deferer(jnp.sqrt)
  sign = _generate_unary_deferer(jnp.sign)
  log = _generate_unary_deferer(jnp.log)
  exp = _generate_unary_deferer(jnp.exp)
  imag = _generate_unary_deferer(jnp.imag)
  real = _generate_unary_deferer(jnp.real)
  abs = __abs__
  conjubuilding_block = conj

  zeros_like = _generate_unary_deferer(jnp.zeros_like)
  ones_like = _generate_unary_deferer(jnp.ones_like)
  full_like = _generate_unary_deferer(jnp.full_like)

  # These two don't fit the pattern because they return scalars, not arrays.
  def min(self):
    """Returns the smallest element of the matrix.

    See jax.numpy.min for full documentation.
    """
    return jnp.min(self.array)

  def max(self):
    """Returns the largest element of the matrix.

    See jax.numpy.max for full documentation.
    """
    return jnp.max(self.array)

  def all(self):
    """Returns whether all elements of the matrix are True.

    See jax.numpy.all for full documentation.
    """
    return jnp.all(self.array)

  def any(self):
    """Returns whether any elements of the matrix are True.

    See jax.numpy.any for full documentation.
    """
    return jnp.any(self.array)

  def allclose(self, other, *args, **kwargs):
    """Returns whether all elements of the two arrays are close to each toher.

    See jax.numpy.allclose for full documentation.
    """
    return self._defer_binary_elementwise(
        other,
        jnp.allclose,
        *args,
        **kwargs,
    ).array

  @property
  def dtype(self):
    return self.array.dtype

  @property
  def shape(self):
    return self.array.shape

  def __repr__(self):
    return f"ReplicatedThinMatrix({self.array}, {self.is_column_replicated}"


###############################################################################
# UTILITIES
###############################################################################
def _local_rows(global_shape, column_replicated, grid="full"):
  """
  Return the number of rows each block of a matrix of shape `global_shape`
  would have when distributed over the grid specified by
  `grid` with the replication pattern specified by `column_replicated`.
  """
  global_rows, _ = global_shape
  if grid == "full":
    grid_rows, grid_cols = pops.GRID
  elif grid == "host":
    grid_rows, grid_cols = pops.HGRID
  elif grid == "device":
    grid_rows, grid_cols = pops.DGRID
  else:
    raise ValueError(f"Invalid grid {grid}.")

  if column_replicated:
    if global_rows % grid_rows != 0:
      raise ValueError(f"Can not distribute {global_rows} entries over"
                       f"{grid_rows} rows.")
    local_rows = global_rows // grid_rows
  else:
    if global_rows % grid_cols != 0:
      raise ValueError(f"Can not distribute {global_rows} entries over"
                       f"{grid_cols} cols.")
    local_rows = global_rows // grid_cols
  return local_rows


def _align(align_me, align_to):
  """ Returns a ReplicatedThinMatrix representing the same data as `align_me`
  with the same replication patter as `align_to`.
  """
  if align_to.is_column_replicated:
    if not(align_me.is_column_replicated):
      align_me = to_column_replicated(align_me)
  else:
    if align_me.is_column_replicated:
      align_me = to_row_replicated(align_me)
  return align_me


def _psum(vec):
  """ Sums over the unreplicated data in vec. No local sum is performed.
  """
  if vec.is_column_replicated:
    groups = pops._axis_index_pcols()
  else:
    groups = pops._axis_index_prows()
  summed_array = lax.psum(
    vec.array, axis_name=pops.AXIS_NAME,
    axis_index_groups=groups)
  return ReplicatedThinMatrix(summed_array, vec.is_column_replicated)


def frobnorm(vec):
  """ Computes the Frobenius norm of the ReplicatedThinMatrix `vec`.
  """
  vec = ReplicatedThinMatrix(jnp.abs(vec.array)**2, vec.is_column_replicated)
  psummed = _psum(vec)
  summed = jnp.sum(psummed.array)
  return jnp.sqrt(summed)


def _indices_vec(vec):
  """ Returns local vectors storing the row and column indices of vec.
  This is hidden since calling it from outside can be unstable, as
  the results change depending on whether vec is row or column replicated.
  """
  m, n = vec.shape
  if vec.is_column_replicated:
    i = pops.my_prow()
  else:
    i = pops.my_pcol()
  rows_vector = jnp.arange(m) + i * m
  cols_vector = jnp.arange(n)
  return rows_vector, cols_vector


def add_to_diagonal(vec, val, k=0):
  """ Adds the scalar `val` to the diagonal entries of `vec`.
  """
  rows_vector, cols_vector = _indices_vec(vec)
  on_kth_diagonal = rows_vector[:, None] == cols_vector - k
  data = vec.array
  data = jnp.where(on_kth_diagonal, x=data + val, y=data)
  return ReplicatedThinMatrix(data, vec.is_column_replicated)


###############################################################################
# INITIALIZATION
###############################################################################
def random(global_shape, column_replicated=True, key_seed=None):
  """
  Generates a normal distributed random ReplicatedThinMatrix.

  Args:
    global_shape: The shape of the full, logical matrix.
    column_replicated: Boolean for whether the matrix should be column or
      row-replicated.
    key_seed: Seed for jax.random.PRNGKey

  Returns:
    A: A ReplicatedThinMatrix.
  """
  if key_seed is None:
    key_seed = int(time.time())
  key_seed += jax.host_id()
  _, global_cols = global_shape
  if column_replicated:
    unique_procs = pops.DGRID[0]
    replicated_procs = pops.DGRID[1]
  else:
    unique_procs = pops.DGRID[1]
    replicated_procs = pops.DGRID[0]

  # We need to generate one unique_keys per unreplicated process, and then
  # broadcast these appropriately over the full process grid. Each key
  # individually has two elements, which is why "2" appears in the row
  # replicated case.
  unique_keys = jax.random.split(jax.random.PRNGKey(key_seed), unique_procs)
  key_list = [unique_keys, ] * replicated_procs
  if not column_replicated:
    all_keys = jnp.vstack(key_list)
  else:
    all_keys = jnp.hstack(key_list).reshape((pops.NDPROCS, 2))

  local_rows = _local_rows(global_shape, column_replicated)
  array = jax.pmap(
    lambda key: jax.random.normal(key, (local_rows, global_cols)))(all_keys)
  return ReplicatedThinMatrix(array, column_replicated)


def zeros(global_shape, dtype=jnp.float32, column_replicated=False):
  _, global_cols = global_shape
  local_rows = _local_rows(global_shape, column_replicated)
  array = jnp.zeros((local_rows, global_cols), dtype=dtype)
  return ReplicatedThinMatrix(array, column_replicated)


def ones(global_shape, dtype=jnp.float32, column_replicated=False):
  _, global_cols = global_shape
  local_rows = _local_rows(global_shape, column_replicated)
  array = jnp.ones((local_rows, global_cols), dtype=dtype)
  return ReplicatedThinMatrix(array, column_replicated)


def full(global_shape, fill_value, dtype=jnp.float32, column_replicated=False):
  _, global_cols = global_shape
  local_rows = _local_rows(global_shape, column_replicated)
  array = jnp.full((local_rows, global_cols), fill_value, dtype=dtype)
  return ReplicatedThinMatrix(array, column_replicated)


def distribute(v, pmap=True, column_replicated=True,
               host_replicated_input=True):
  """
  Converts the matrix `v` on the `host` to either a `column-replicated` or
  `row-replicated` matrix depending on the value of `column_replicated`.

  This function distributes across devices, but does have some limited support
  for multi-host settings. With `host_replicated_input=False`,
  the vector is assumed to already be appropriately distributed across hosts,
  and each host will act as if in the single-host case.
  With `host_replicated_input=True`, a copy of the full vector is assumed
  to be present upon each host. Each host will first apply an appropriate mask,
  and then act otherwise as if in the single-host case.

  Args:
    v: Matrix to distribute.
    pmap: Whether or not to pmap.
    column_replicated: Whether to column or row replicate the matrix.
    host_replicated_input: If False, `distribute` will assume the vector is
                           already distributed with respect to the hosts;
                           if True, it will assume a copy of the full vector
                           to be present on each.
  Returns:
    v: The distributed matrix.
  """
  device_rows, device_cols = pops.DGRID
  if host_replicated_input:
    if column_replicated:
      v = _divvy_prows(v, host_only=True)
    else:
      v = _divvy_pcols(v, host_only=True)

  _, global_cols = v.shape
  local_rows = _local_rows(v.shape, column_replicated, grid="device")

  if column_replicated:
    v = v.reshape((device_rows, local_rows, global_cols))
    v = jnp.stack([v] * device_cols, axis=0)
  else:
    v = v.reshape((device_cols, local_rows, global_cols))
    v = jnp.stack([v] * device_rows, axis=1)
  v = v.reshape((device_rows * device_cols, local_rows, global_cols))

  if pmap:
    v = pops.pmap(lambda x: x)(v)
  return ReplicatedThinMatrix(v, column_replicated)


def undistribute(v, replicate_over_hosts=False):
  """
  Collects the unique elements of v back to the host.
  Args:
    v: A column or row-replicated matrix.
  Returns:
    v: A host matrix.
  """
  if replicate_over_hosts:
    raise NotImplementedError("replicate_over_hosts not implemented.")
  # TODO: multi-host
  if not isinstance(v, ReplicatedThinMatrix):
    msg = "The matrix to undistribute is not of type ReplicatedThinMatrix."
    raise ValueError(msg)

  arr = v.array
  grid_rows, grid_cols = pops.GRID
  _, local_rows, global_cols = arr.shape
  if v.is_column_replicated:
    global_rows = local_rows * grid_rows
  else:
    global_rows = local_rows * grid_cols
  arr = jax.device_put(arr)
  arr = arr.reshape((grid_cols, grid_rows, local_rows, global_cols))

  if v.is_column_replicated:
    arr = arr[0, :, :, :].reshape(global_rows, global_cols)
  else:
    arr = arr[:, 0, :, :].reshape(global_rows, global_cols)
  return arr


def big_to_thin(A, trim_columns_to=None):
  """ Replicates the columns of the checkerboard distributed matrix A across
  processor rows, returning a row-replicated ReplicatedThinMatrix
  representing the same data. This should be pmapped with the second
  argument static.

  Args:
    A: Checkerboard distributed matrix to replicate.
    trim_columns_to: If not None, A[:, :trim_columns_to] will be
      replicated.
  Returns:
    A column-replicated ReplicatedThinMatrix representing the same data as A.
  """
  A = pops.gather_rows(A)
  if trim_columns_to is not None:
    A = A[:, :trim_columns_to]
  return ReplicatedThinMatrix(A, is_column_replicated=True)


def set_columns(vecs, new_columns, col_start):
  """ Logically performs
  `vecs[:, col_start:col_start + new_columns.shape[1]] = new_columns`,
  where `new_columns` is a ReplicatedThinMatrix, and `vecs` is either
  a ReplicatedThinMatrix or a checkerboard-distributed matrix.

  Args:
    vecs: A ReplicatedThinMatrix or checkerboard-distributed matrix into
      which new_columns is to be inserted.
    new_columns: The columns to insert.

    col_start: The first column to begin the insertion at.
  Returns:
    A copy of vecs with the columns inserted.
  """
  if isinstance(vecs, ReplicatedThinMatrix):
    new_columns = _align(new_columns, vecs)
    new_vecs = lax.dynamic_update_slice(
      vecs.array, new_columns.array, (0, col_start))
    return ReplicatedThinMatrix(new_vecs, vecs.is_column_replicated)
  else:
    new_columns = to_column_replicated(new_columns)
    if vecs.shape[0] != new_columns.shape[0]:
      raise TypeError(f"vecs {vecs.shape} and new_columns {new_columns}.shape"
                      "had incompatible shapes.")
    n_rows = new_columns.shape[0]
    n_new_cols = new_columns.shape[1]
    cols_per_pcol = vecs.shape[1]
    if n_new_cols > cols_per_pcol:
      # This branch covers the case that more new columns are to be added
      # than would fit in a single local block of vecs.
      # We then loop over these columns and insert them local panel by local
      # panel. This requires that we first pad new_columns (with columns
      # from vecs) to be evenly divisible by vecs' local number of columns.
      pad_size = misc.distance_to_next_divisor(n_new_cols, cols_per_pcol)
      padded_size = n_new_cols + pad_size
      pad_cols = get_columns(vecs, n_new_cols, pad_size).array
      padded = jnp.zeros((n_rows, padded_size), dtype=new_columns.dtype)
      padded = lax.dynamic_update_slice(padded, new_columns.array, (0, 0))
      padded = lax.dynamic_update_slice(padded, pad_cols, (0, n_new_cols))

      n_insertions = padded_size // cols_per_pcol
      insertion_start = 0
      insertion_col = col_start
      for _ in range(n_insertions):
        insertion = lax.dynamic_slice(
          padded, (0, insertion_start), (n_rows, cols_per_pcol))
        vecs = _set_columns_mat(vecs, insertion, insertion_col)
        insertion_start += cols_per_pcol
        insertion_col += cols_per_pcol
    else:
      vecs = _set_columns_mat(vecs, new_columns.array, col_start)
    return vecs


def _set_columns_mat_single_pcol(args):
  """ This branch covers the simplest case that all columns are to
  be added to a single block of mat.
  """
  mat, new_columns, col_start = args
  _, n_l = mat.shape
  pcol = col_start // n_l
  bcol = col_start - n_l * pcol
  inserted = lax.dynamic_update_slice(mat, new_columns, (0, bcol))
  return jnp.where(pops.my_pcol() == pcol, x=inserted, y=mat)


def _set_columns_mat_multiple_pcols(args):
  """ This branch covers the case that the desired insertion spans
  multiple blocks of mat (but all columns are to
  be added to a single block of mat.
  """
  mat, new_columns, col_start = args
  # First we pad new_columns to the local size of mat.
  _, mat_cols = pops.indices(mat.shape)
  n_new_cols = new_columns.shape[1]
  new_columns = lax.dynamic_update_slice(
    jnp.zeros_like(mat), new_columns, (0, 0))

  # We need to sort the columns of new_columns so that those to
  # be inserted are aligned with the points of mat at which to insert
  # them.
  col_idxs = jnp.arange(0, new_columns.shape[1]) + col_start
  col_end = n_new_cols + col_start

  # If this is the leftmost pcol containing an insertion,
  # aligning the columns requires us to bring those not being inserted to the
  # left of the block. Otherwise they must be brought to the right.
  # We thus replace the col_idxs outside the insertion with -1 or
  # max(col_idx) + 1 respectively. Sorting the columns then achieves this.

  # True iff col_start is in this block of mat_cols.
  leftmost_pcol = jnp.isin(jnp.full(1, col_start), mat_cols)
  leftmost_pcol = jnp.full_like(col_idxs, leftmost_pcol)
  mask_value = jnp.where(leftmost_pcol,
                         x=jnp.full_like(col_idxs, -1),
                         y=jnp.full_like(col_idxs, col_end + 1))

  first_pcol_idx = mat_cols[0, 0]
  last_pcol_idx = mat_cols[0, -1]
  to_be_inserted_here = jnp.logical_and(
    col_idxs >= first_pcol_idx, col_idxs <= last_pcol_idx)
  masked_col_idxs = jnp.where(to_be_inserted_here, x=col_idxs, y=mask_value)
  sort_idxs = jnp.argsort(masked_col_idxs)
  new_columns = new_columns[:, sort_idxs]
  in_range = jnp.logical_and(mat_cols >= col_start, mat_cols < col_end)
  return jnp.where(in_range, x=new_columns, y=mat)


def _set_columns_mat(mat, new_columns, col_start):
  """ mat is a checkerboard distributed matrix, and new_columns
  is the data from a ReplicatedThinMatrix, to be inserted into
  the columns of mat starting from col_start. It is assumed
  that there are fewer new_columns than locally held by mat.
  """
  multiple_pcols = (new_columns.shape[1] + col_start) > mat.shape[1]
  return lax.cond(multiple_pcols,
                  _set_columns_mat_multiple_pcols,
                  _set_columns_mat_single_pcol,
                  (mat, new_columns, col_start))


def get_columns(vecs, col_start, n_cols):
  """ Returns the ReplicatedThinMatrix logically specified as
  `vecs[:, col_start:col_start + n_cols]`. `n_cols` must be static.

  Args:
    vecs: A ReplicatedThinMatrix or checkerboard distributed matrix from
      which to extract columns. In the latter case the desired extraction
      must not straddle a pcol. The code does not test for this since
      doing so would require col_start to be concretized.
    col_start: First column to extract.
    n_cols: Number of columns to extract.
  Returns:
    The extracted columns, a ReplicatedThinMatrix. If vecs was a
      ReplicatedThinMatrix, these will have the same replication pattern
      as it did. Otherwise they will be column replicated.
  """
  if isinstance(vecs, ReplicatedThinMatrix):
    columns = lax.dynamic_slice(
      vecs.array, (0, col_start), (vecs.shape[0], n_cols))
    columns = ReplicatedThinMatrix(columns, vecs.is_column_replicated)
  else:
    _, n_l = vecs.shape
    pcol = col_start // n_l
    bcol = col_start - n_l * pcol
    columns = lax.dynamic_slice(vecs, (0, bcol), (vecs.shape[0], n_cols))
    columns = pops.broadcast_pcol(columns, pcol)
    columns = ReplicatedThinMatrix(columns, True)
  return columns


##############################################################################
# GATHERS
##############################################################################
def gather_prows(v):
  """
  Does an all-gather over processor rows. The result is column-replicated.

  Before (grid=(2, 2)):
  v = [[1, 2], [3, 4]]
      [[5, 6], [7, 8]]
  After:
  v = [[1, 2, 3, 4], [1, 2, 3, 4]]
      [[5, 6, 7, 8], [5, 6, 7, 8]]
  """
  arr = v.array
  _, global_cols = arr.shape
  groups = pops._axis_index_prows()
  arr = jax.lax.all_gather(arr, axis_index_groups=groups, axis_name="i")
  return ReplicatedThinMatrix(arr.reshape(-1, global_cols), True)


def gather_pcols(v):
  """
  Does an all-gather over processor cols. The result is row-replicated.

  Before (grid=(2, 2)):
  v = [[1, 2], [3, 4]]
      [[5, 6], [7, 8]]
  After:
  v = [[1, 2, 5, 6], [3, 4, 7, 8]]
      [[1, 2, 5, 6], [3, 4, 7, 8]]
  """
  arr = v.array
  _, global_cols = arr.shape
  groups = pops._axis_index_pcols()
  arr = jax.lax.all_gather(arr, axis_index_groups=groups, axis_name="i")
  return ReplicatedThinMatrix(arr.reshape(-1, global_cols), False)


def hstack_pair(vec_1: ReplicatedThinMatrix, vec_2: ReplicatedThinMatrix):
  """ Returns hstack([vec_1, vec_2]) as a ReplicatedThinMatrix with the same
  replication pattern as vec_1. Can be pmapped or not.

  Args:
    vec_1, vec_2: ReplicatedThinMatrices to concatenate.
  Returns:
    The hstacked result, another ReplicatedThinMatrix with the same pattern
  as vec_1.
  """
  if vec_1.array.ndim == 3:
    return pops.pmap(_hstack_pair)(vec_1, vec_2)
  return _hstack_pair(vec_1, vec_2)


def _hstack_pair(vec_1, vec_2):
  """ Does the work for hstack_pair. This is assumed to be pmapped.
  """
  vec_2 = _align(vec_2, vec_1)
  if vec_1.shape[0] != vec_2.shape[0]:
    raise TypeError("Vectors to hstack had misaligned shapes "
                    f"{vec_1.shape}, {vec_2.shape}.")
  return_dat = jnp.hstack([vec_1.array, vec_2.array])
  return ReplicatedThinMatrix(return_dat, vec_1.is_column_replicated)


###############################################################################
# REDISTRIBUTION
###############################################################################
def to_column_replicated(v):
  """
  Converts the row-replicated matrix `v` into its column-replicated equivalent.
  If `v` is already column-replicated, this function does nothing.

  Unless `v` is already column-replicated, an error is emitted if
  (local_size * grid[1]) % grid[0] != 0.

  Input (grid=(4, 2)):
  v = [[1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]]

  Output:
  v = [[1, 2], [1, 2]
       [3, 4], [3, 4]
       [5, 6], [5, 6]
       [7, 8], [7, 8]]

  Args:
    v: A distributed matrix.
  Returns:
    v: The column-replicated equivalent of v.
  """
  if not isinstance(v, ReplicatedThinMatrix):
    msg = "v is not a ReplicatedThinMatrix. Did you forget to distribute v?"
    raise ValueError(msg)
  if v.is_column_replicated:
    return v
  if len(v.shape) > 2:
    msg = "v has more than 2 dimensions. Did you forget to pmap v?"
    raise ValueError(msg)

  v = gather_prows(v)
  arr = v.array
  arr = _divvy_prows(arr)
  return ReplicatedThinMatrix(arr, True)


def to_row_replicated(v):
  """
  Converts the column-replicated matrix `v` into its row-replicated equivalent.
  If `v` is already row-replicated, this function does nothing.

  Unless `v` is already row-replicated, an error is emitted if
  (local_size * grid[0]) % grid[1] != 0.

  Input (grid=(4, 2)):
  v = [[1, 2], [1, 2]
       [3, 4], [3, 4]
       [5, 6], [5, 6]
       [7, 8], [7, 8]]

  Output:
  v = [[1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]
       [1, 2, 3, 4], [5, 6, 7, 8]]


  Args:
    v: A distributed matrix.
    grid: Processor grid.
  Returns:
    v: The row-replicated equivalent of v.
  """
  if not isinstance(v, ReplicatedThinMatrix):
    msg = "v is not a ReplicatedThinMatrix. Did you forget to distribute v?"
    raise ValueError(msg)
  if not v.is_column_replicated:
    return v
  if len(v.shape) > 2:
    msg = "v has more than 2 dimensions. Did you forget to pmap v?"
    raise ValueError(msg)

  v = gather_pcols(v)
  arr = v.array
  arr = _divvy_pcols(arr)
  return ReplicatedThinMatrix(arr, False)


def _divvy_prows(v, host_only=False):
  """
  Divides v into one even contiguous block per prow. Each processor extracts
  the prow'th such block.

  With `host_only`, this function operates only between the hosts e.g. data is
  not divvied between devices.

  This function is more easily depicted than described. On a 2 x 2 grid:

  Before (grid=(2, 2)):
  v = [[1, 2, 3, 4], [9, 10, 11, 12]]
      [[5, 6, 7, 8], [13, 14, 15, 16]]

  After:
  v = [[1, 2], [9, 10]]
      [[7, 8], [15, 16]]

  Typically this function would be called on an input whose local entries were
  all identical. In this case the result is a column-replicated matrix.
  """
  if host_only:
    prow = jax.host_id() % pops.NHCOLS
    grid = "host"
  else:
    prow = pops.my_prow()
    grid = "full"
  panel_size = _local_rows(v.shape, True, grid=grid)
  _, global_cols = v.shape
  start = prow * panel_size
  zero = jnp.zeros_like(start)  # Make sure the types of zero and start match
  return jax.lax.dynamic_slice(v, (start, zero), (panel_size, global_cols))


def _divvy_pcols(v, host_only=False):
  """
  Divides v into one even contiguous block per pcol. Each processor extracts
  the pcol'th such block.

  With `host_only`, this function operates only between the hosts e.g. data is
  not divvied between devices.

  This function is more easily depicted than described. On a 2 x 2 grid:


  Before (grid=(2, 2)):
  v = [[1, 2, 3, 4], [9, 10, 11, 12]]
      [[5, 6, 7, 8], [13, 14, 15, 16]]

  After:
  v = [[1, 2], [11, 12]]
      [[5, 6], [15, 16]]

  Typically this function would be called on an input whose local entries were
  all identical. In this case the result is a row-replicated matrix.
  """
  if host_only:
    pcol = jax.host_id() // pops.NHROWS
    grid = "host"
  else:
    pcol = pops.my_pcol()
    grid = "full"
  _, global_cols = v.shape
  panel_size = _local_rows(v.shape, False, grid=grid)
  start = pcol * panel_size
  zero = jnp.zeros_like(start)  # Make sure the types of zero and start match
  return jax.lax.dynamic_slice(v, (start, zero), (panel_size, global_cols))


###############################################################################
# MULTIPLICATIONS
###############################################################################
def matvec(A, x, transpose_A=False, precision=lax.Precision.HIGHEST):
  """
  Computes either A @ x or A.T @ x, with A a checkerboard-distributed matrix
  and x a row or column-replicated matrix. For A @ x (transpose_A = False), x
  will, if necessary, first be brought into row-replicated form, and the result
  will be column-replicated. For A.T @ x (transpose_A = True), x will, if
  necessary, first be brought into column-replicated form, and the result will
  be row-replicated.

  This function is to be called within a pmap.

  Args:
    A: A checkerboard-distributed matrix.
    x: A column or row-replicated matrix.
    transpose_A: Bool indicating whether A @ x or A.T @ x will be performed.
  Returns:
    x: If transpose_A, the column-replicated A.T @ x.
       If not, the row-replicated A @ x.
  """
  if transpose_A:
    x = to_column_replicated(x)
    local_terms = jnp.dot(A.T, x.array, precision=precision)
    result = pops.sum_over_pcols(local_terms)
    return ReplicatedThinMatrix(result, False)

  x = to_row_replicated(x)
  local_terms = jnp.dot(A, x.array, precision=precision)
  result = pops.sum_over_prows(local_terms)
  return ReplicatedThinMatrix(result, True)


def _diagmult_right(matrix, vector):
  """
  Does `matrix @ diag(vector)`.
  """
  vector = to_row_replicated(vector)
  return matrix * jnp.ravel(vector.array)


def _diagmult_left(matrix, vector):
  """
  Does `diag(vector) @ matrix`.
  """
  vector = to_column_replicated(vector)
  return jnp.ravel(vector.array)[:, None] * matrix


def diagmult(matrix, vector, vector_on_right=True):
  """
  Performs either `matrix @ diag(vector)` (`vector_on_right==True`) or
  `diag(vector) @ matrix` (`vector_on_right==False`) using broadcasted
  elementwise multiplication.

  Args:
    matrix: A checkerboard-distributed matrix.
    vector: A `ReplicatedThinMatrix` representing a vector.
    vector_side: `vector_on_right==True` (`False`) does `matrix @ diag(vector)`
                 (`diag(vector) @ matrix`).
  Returns:
    The checkerboard-distributed result.
  """
  if jnp.squeeze(vector.array).shape == vector.shape:
    raise TypeError(f"`vector` must be a vector; had shape {vector.shape}.")

  if vector_on_right:
    diagmult_f = _diagmult_right
  else:
    diagmult_f = _diagmult_left

  if matrix.ndim == 3:
    diagmult_f = pops.pmap(diagmult_f)
  elif matrix.ndim != 2:
    raise TypeError(f"matrix.ndim=={matrix.ndim} unsupported.")
  return diagmult_f(matrix, vector)


def vec_t_mat(vec, mat, precision=lax.Precision.HIGHEST):
  """ Computes vec^T @ mat = vec_out, where vec^T and vec_out are
  ReplicatedThinMatrices and mat is a checkerboard-distributed matrix.
  Logically a row vector, the result is returned as a column-replicated
  "column" vector which may be viewed as its transpose; that is, the
  result needs to be transposed after passing through vops.undistribute.
  Note this function does not conjubuilding_block vec.
  Args:
    vec: vec in vec^T @ mat, a ReplicatedThinMatrix (m, b).
    mat: mat in the same, a checkerboard distributed matrix (m, n).
    precision: ASIC matmul precision.
  Returns:
    vec_out: A row-replicated ReplicatedThinMatrix (k, b). b is the
      replicated dimension; this may be viewed as the transpose of
      the mathematical row-vector result.
  """
  vec = to_column_replicated(vec)
  summands = jnp.dot(vec.array.T, mat, precision=precision)
  result = pops.sum_over_pcols(summands)
  return ReplicatedThinMatrix(result.T, is_column_replicated=False)


def vecvec(vec_l, vec_r, precision=lax.Precision.HIGHEST):
  """ Computes `vec_l^H @ vec_r`, where both `vec_l` and `vec_r` are
  ReplicatedThinMatrices of respective logical shapes
  `(N, k_l)` and `(N, k_r)`. The `(k_l, k_r)` result is fully replicated
  across processors.
  """
  vec_l = to_column_replicated(vec_l)
  vec_r = _align(vec_r, vec_l)
  summands = jnp.dot(vec_l.array.conj().T, vec_r.array, precision=precision)
  return pops.sum_over_pcols(summands)


def outer(vec_L, vec_R, precision=lax.Precision.HIGHEST):
  """ Computes 'vec_L (m, b) otimes vec_R (n, b) = vec_L @ vec_R^T = mat',
  where vec_L and vec_R are ReplicatedThinMatrices and mat is a
  checkerboard-distributed matrix. This is a literal outer product for b = 1
  and a "rank-b update" otherwise.
  Note this function does not conjubuilding_block vec_R. This is essentially
  a single iteration of SUMMA.
  Args:
    vec_L: LHS in product (m, b).
    vec_R: RHS in product (n, b).
    precision: ASIC matmul precision.
  Returns:
    mat: Checkerboard-distributed result (m, n).
  """
  vec_L = to_column_replicated(vec_L)
  vec_R = to_row_replicated(vec_R)
  return jnp.dot(vec_L.array, vec_R.array.T, precision=precision)


def vecsmall(vec, small, precision=lax.Precision.HIGHEST):
  """ Computes `vec @ small`, where `vec` is a ReplicatedThinMatrix and
  `small` is fully replicated across processors.

  Args:
    vec: ReplicatedThinMatrix, the LHS operand.
    small: Replicated across processors, the RHS operand.
    precision: ASIC matmul precision.
  Returns:
    The result `vec @ small`, a ReplicatedThinMatrix
  """
  result = jnp.dot(vec.array, small, precision=precision)
  if result.ndim == 1:
    result = result.reshape((result.size, 1))
  return ReplicatedThinMatrix(result, vec.is_column_replicated)
