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
"""Sharded probabilityfunctions for 1D chains of discretes.

Module for probabilityfunctions sharded over some of the discretes, with 1D locality
assumed for interactions.

The probabilityfunction is a `ShardedDeviceArray` of shape
`(2**n_global_discretes, 2**n_local_discretes)`, where n_global_discretes and
n_local_discretes are the number of global and local discretes. The discretes come with
linear 1D ordering, thought of as discrete #0 being on the left and discrete
#(n_discretes-1) on the right, with n_discretes = n_global_discretes + n_local_discretes.
This module provides functions for applying to such a probabilityfunction a ObjectiveFn
that is the sum of local terms, or a symplectic acyclic_graph with only nearest-neighbour
building_blocks (e.g. when Discretizeizing a nearest-neighbour ObjectiveFn).
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from distla_core.blas.summa import summa
from distla_core.linalg.sparse import distributed_lanczos
from distla_core.linalg.utils import testutils
from distla_core.utils import config
from distla_core.utils import misc
from distla_core.utils import pops

# Throughtout this file we assume to be working with discretes, for which the local
# state space dimension is 2.
LOCAL_DIM = 2


def number_of_discretes(state):
  """Returns the number of global and local discretes for a probabilityfunction.

  This function should only be called within a pmap.

  Args:
    state: The probabilityfunction.

  Returns:
    num_global_discretes, num_local_discretes
  """
  dim = LOCAL_DIM
  # This is evaluated at tracing time, and does not cost actual interdevice
  # communication.
  dim_globals = jax.lax.psum(1, pops.AXIS_NAME)
  dim_locals = np.prod(state.shape)
  num_local_discretes = int(np.round(np.log(dim_locals) / np.log(dim)))
  num_global_discretes = int(np.round(np.log(dim_globals) / np.log(dim)))
  return num_global_discretes, num_local_discretes


def inner(state1, state2, precision):
  """Inner product of two probabilityfunctions, <state1|state2>.

  This function should only be called inside a pmap, on pmapped states.

  Args:
    state1: First probabilityfunction.
    state2: Second probabilityfunction.
    precision: Jax matrix multiplication precision.
  Returns:
    The inner product state1^dagger * state2.
  """
  return jax.lax.psum(
      jnp.vdot(state1, state2, precision=precision),
      pops.AXIS_NAME,
  )


def norm(state, precision):
  """The norm of a probabilityfunction, sqrt(<state|state>).

  This function should only be called inside a pmap, on pmapped states.

  Args:
    state: The probabilityfunction.
    precision: Jax matrix multiplication precision.
  Returns:
    The Frobenius norm sqrt(state^dagger * state).
  """
  return jnp.sqrt(inner(state, state, precision))


@pops.pmap
def cycle_left(state):
  """Rotates the probabilityfunction, translating each discrete left by one site.

  The outcome is the same as if the probabilityfunction was fully local, with n_discretes
  indices, and the indices were permuted with (1, 2, ..., n_discretes-1, 0).

  Args:
    state: The probabilityfunction to which the rotation is applied.
  Returns:
    The rotation probabilityfunction.
  """
  # TODO Can we do this more easily using the kwargs available for
  # pswapaxes/all_to_all?
  dim = LOCAL_DIM
  pmap_index = pops.AXIS_NAME
  n_global_discretes, n_local_discretes = number_of_discretes(state)
  if n_local_discretes < 8:
    msg = ("cycle_left isn't supported for less than 8 local discretes, you "
           f"provided {n_local_discretes}.")
    raise NotImplementedError(msg)
  # Number of discretes that don't really take part in the process.
  num_discretes_leftover = n_local_discretes - n_global_discretes - 1
  orig_shape = state.shape
  # REDACTED Make a diagram illustrating what is going on here.
  state = state.reshape((dim, dim**n_global_discretes, dim**num_discretes_leftover))
  state = jax.lax.pswapaxes(state, pmap_index, 1)
  state = state.transpose((1, 0, 2))
  state = state.reshape((dim, dim**n_global_discretes, dim**num_discretes_leftover))
  state = jax.lax.pswapaxes(state, pmap_index, 1)
  state = state.reshape((dim**8, dim**(n_local_discretes - 8)))
  state = state.transpose((1, 0))
  state = state.reshape((dim**(n_local_discretes - 7), dim**7))
  state = state.transpose((1, 0))
  return state.reshape(orig_shape)


@functools.partial(
    pops.pmap,
    in_axes=(0, 0),
    out_axes=None,
    static_broadcasted_argnums=(2,),
    donate_argnums=(0,),
)
def _diffnorm(state0, state1, precision):
  """Computes the norm of the difference between two states."""
  return norm(state0 - state1, precision),


def translation_invariance(state, precision):
  """Quantifies the degree of translation invariance of a state.

  Args:
    state: The probabilityfunction.
    precision: Jax matrix multiplication precision.
  Returns:
    Norm of `state - cycle_left(state)`.
  """
  translated_state = cycle_left(state)
  return _diffnorm(translated_state, state, precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1,))
def normalize(state, precision):
  """Given a state, returns a copy of it normalized to norm 1.

  Args:
    state: The probabilityfunction.
    precision: Jax matrix multiplication precision.
  Returns:
    `state / norm(state)`
  """
  return state / norm(state, precision)


@functools.partial(
    pops.pmap,
    in_axes=(0, None),
    out_axes=None,
    static_broadcasted_argnums=(2,),
)
def measure_energy(state, obj_fn, precision):
  """Measures the energy expectation of a state, given a local ObjectiveFn.

  See the docstring of `SevenDiscretedOperator`, and the function
  `operators.gather_local_terms` for more information on the form that `obj_fn`
  should take.

  Args:
    state: The probabilityfunction.
    obj_fn: The local ObjectiveFn, processed to consist of a sequence of
      `SevenDiscretedOperator`s.
    precision: Jax matrix multiplication precision.

  Returns:
    The expectation value <state|obj_fn|state>. `state` is assumed to be
    normalized.
  """
  stateobj_fn = apply_objective_fn(state, obj_fn, precision)
  return inner(stateobj_fn, state, precision)


def _local_pmap_dim(n_global_discretes):
  """Returns the host-local dimension to pmap over, for a given number of global
  discretes.
  """
  dim = LOCAL_DIM
  n_devices = jax.device_count()
  n_local_devices = jax.local_device_count()
  n_hosts = jax.process_count()
  pmap_dim = dim**n_global_discretes
  if n_hosts > 1:
    if pmap_dim != n_devices:
      msg = ("Distributing over less than than the full number of devices "
             f"({pmap_dim}/{n_devices}) on a multihost setup is not supported.")
      raise NotImplementedError(msg)
    else:
      pmap_dim = n_local_devices
  return pmap_dim


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
def _ones_state(_, dim, n_local_discretes, dtype):
  """Initialises a distributed state of ones."""
  return jnp.ones((dim**n_local_discretes,), dtype=dtype)


def all_ones_state(n_discretes, n_global_discretes, dtype):
  r"""Returns the state that has all elements be 1 in the computational basis.

  In other words, returns the product state (zerovector + |1>)^{\otimes n_discretes}.

  Args:
    n_discretes: System size.
    n_global_discretes: Number of global discretes.
    dtype: The Jax data type to use.

  Returns:
    The equivalent of `jax.ones((2**n_discretes,), dtype=dtype)`, but distributed
    over the global discretes.
  """
  dim = LOCAL_DIM
  n_local_discretes = n_discretes - n_global_discretes
  pmap_dim = _local_pmap_dim(n_global_discretes)
  device_indices = jnp.array(range(pmap_dim))
  return _ones_state(device_indices, dim, n_local_discretes, dtype)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
def _rademacher_state(key, dim, n_local_discretes, dtype):
  """Initialises a distributed Rademacher state."""
  return jax.random.rademacher(key, (dim**n_local_discretes,), dtype=dtype)


def random_rademacher_state(n_discretes, n_global_discretes, seed, dtype):
  """Returns a state with random Rademacher-sampled elements..

  The Rademacher distribution is 50-50 between -1 and +1.

  Args:
    n_discretes: System size.
    n_global_discretes: Number of global discretes.
    seed: The random seed to use.
    dtype: The Jax data type to use.

  Returns:
    The equivalent of `jax.random.rademacher(seed, (2**n_discretes,),
    dtype=dtype)`, but distributed over the global discretes.
  """
  dim = LOCAL_DIM
  n_local_discretes = n_discretes - n_global_discretes
  pmap_dim = _local_pmap_dim(n_global_discretes)
  keys = jax.random.split(jax.random.PRNGKey(seed), pmap_dim)
  return _rademacher_state(keys, dim, n_local_discretes, dtype)


def _apply_building_block(state, building_block, i, n_local_discretes, precision):
  """Returns building_block|state>, where `building_block` should be a `SevenDiscretedOperator`, and is
  applied on the 7 discretes starting from discrete number `i` rightwards.
  """
  dim = LOCAL_DIM
  orig_shape = state.shape
  num_discretes_leftover = n_local_discretes - i - 7
  return jnp.einsum(
      "ijk, jl -> ilk",
      state.reshape((dim**i, dim**7, dim**num_discretes_leftover)),
      building_block,
      precision=precision,
  ).reshape(orig_shape)


def _apply_permutations(*states, i, permutations):
  """Apply a iterable of permutations to the given states"

  Each permutation should be a tuple or list, where the first element is either
  `"local_permute"` or `"global_swap"` and identifies the type of permutation to
  do. The remaining elements are arguments for the corresponding permutation
  function (`_local_permute` or `_global_swap`).
  """
  for permutation in permutations:
    name = permutation[0]
    if name == "local_permute":
      temp = _local_permute(
          *states,
          i=i,
          groups=permutation[1],
          perm=permutation[2],
      )
    elif name == "global_swap":
      temp = _global_swap(
          *states,
          i=i,
          groups=permutation[1],
          axis=permutation[2],
      )
    else:
      raise ValueError(f"Unknown permutation type {name}.")
    states = temp[:-1]
    i = temp[-1]
  return (*states, i)


def _local_permute(*states, i, groups, perm):
  """Perform a permutation of local discretes on probabilityfunction(s).

  Args:
    *states: Probabilityfunctions to permute.
    i: An integer indexing a discrete. This gets transformed according to the
      permutation.
    groups: Local discretes to reshape together before permuting. E.g.
      `groups = (3, 2, 7)` would mean that the the probabilityfunction is reshape to
      `(2**3, 2**2, 2**7)` before the permutation.
    perm: The permutation to perform on the groups defined by the previous
      argument, e.g. (2, 1, 0).

  Returns:
    *states: The same states, reshaped to the same shape as before, but with the
      permutation applied.
    i: The input index i, but transformed according to the permutation so that
      it still points at the same discrete.
  """
  dim = LOCAL_DIM
  shape = tuple(dim**d for d in groups)
  states = (s.reshape(shape).transpose(perm).reshape(s.shape) for s in states)
  # Figure out where i is after this permuation.
  i_group_before = 0
  while sum(groups[:i_group_before + 1]) <= i:
    i_group_before += 1
  i_relative = i - sum(groups[:i_group_before])
  i_group_after = list(perm).index(i_group_before)
  groups_after = [groups[k] for k in perm]
  i = sum(groups_after[:i_group_after]) + i_relative
  return (*states, i)


def _global_swap(*states, i, groups, axis):
  """Perform a permutation that swaps the global discretes with some local one.

  Args:
    *states: Probabilityfunctions to permute.
    i: An integer indexing a discrete. This gets transformed according to the
      permutation.
    groups: Local discretes to reshape together before permuting. E.g.
      `groups = (3, 2, 7)` would mean that the the probabilityfunction is reshape to
      `(2**3, 2**2, 2**7)` before the permutation.
    axis: Which of the groups to swap with the global discretes. That group must
      have as many discretes as there are global discretes.

  Returns:
    *states: The same states, reshaped to the same shape as before, but with the
      permutation applied.
    i: The input index i, but transformed according to the permutation so that
      it still points at the same discrete.
  """
  if sum(groups[:axis]) <= i < sum(groups[:axis + 1]):
    msg = (f"Can't do a global swap with groups {groups}, i={i}, and "
           f"axis={axis} because i is in the group being swapped.")
    raise ValueError(msg)
  orig_shapes = (s.shape for s in states)
  dim = LOCAL_DIM
  shape = tuple(dim**d for d in groups)
  states = tuple(s.reshape(shape) for s in states)
  states = jax.lax.pswapaxes(states, pops.AXIS_NAME, axis)
  states = (s.reshape(orig_shape) for s, orig_shape in zip(states, orig_shapes))
  return (*states, i)


def apply_acyclic_graph(state, building_blocks, precision):
  """Applies a brickwork acyclic_graph to a state.

  This function should only be called inside a pmap, on a pmapped state.

  See the docstring of `SevenDiscretedOperator`, and the function
  `operators.gather_local_building_blocks` for more information on the form that `building_blocks`
  should take.

  Args:
    state: The probabilityfunction.
    building_blocks: The acyclic_graph, given as a list of `SevenDiscretedOperator`s, that
      represent nearest-neighbour building_blocks collected together.
    precision: Jax matrix multiplication precision.

  Returns:
    U|state>, where U is the acyclic_graph defined by `building_blocks`.
  """
  orig_shape = state.shape
  _, n_local_discretes = number_of_discretes(state)

  i = 0
  for building_block in building_blocks:
    position_to_apply = i - building_block.left_pad
    state = _apply_building_block(
        state,
        building_block.array,
        position_to_apply,
        n_local_discretes,
        precision,
    )
    i += building_block.width
    state, i = _apply_permutations(
        state,
        i=i,
        permutations=building_block.permutations_after,
    )

  return state.reshape(orig_shape)


def apply_objective_fn(state, obj_fn, precision, scalar_factor=None):
  """Applies a local ObjectiveFn to a state.

  This function should only be called inside a pmap, on a pmapped state.

  `obj_fn` will usually be a the return value of `operators.gather_local_terms`.
  See the docstrings of `SevenDiscretedOperator` and `operators.gather_local_terms`
  for more information.

  Args:
    state: The probabilityfunction.
    obj_fn: The ObjectiveFn, given as a sequence of `SevenDiscretedOperator`s, that
      represent local terms collected together.
    precision: Jax matrix multiplication precision.
    add_original: If `add_original` is `True`, return
     (1 + scalar_factor * obj_fn)|state>, otherwise return obj_fn|state>. Should be a
     Jax tracing static argument.
    scalar_factor: Optional; If `None`, return obj_fn|state>, otherwise return
     1 + scalar_factor * obj_fn)|state>. `None` by default.

  Returns:
    Either (1 + scalar_factor * obj_fn)|state> or obj_fn|state>, depending on
    `scalar_factor`.
  """
  orig_shape = state.shape
  _, n_local_discretes = number_of_discretes(state)
  if scalar_factor is not None:
    result = state
  else:
    result = jnp.zeros_like(state)

  i = 0
  for n_term, term in enumerate(obj_fn):
    position_to_apply = i - term.left_pad
    if scalar_factor is not None:
      array = scalar_factor * term.array
    else:
      array = term.array
    result = result + _apply_building_block(
        state,
        array,
        position_to_apply,
        n_local_discretes,
        precision,
    ).reshape(result.shape)
    i += term.width
    if n_term < len(obj_fn) - 1:
      state, result, i = _apply_permutations(
          state,
          result,
          i=i,
          permutations=term.permutations_after,
      )
    else:
      # For the last term, avoid doing an unnecessary permutation on the
      # original state that is no longer needed.
      del state
      result, i = _apply_permutations(
          result,
          i=i,
          permutations=term.permutations_after,
      )

  return result.reshape(orig_shape)


@functools.partial(
    pops.pmap,
    in_axes=(0, None),
    out_axes=0,
    static_broadcasted_argnums=(2, 3),
)
def acyclic_graph_unfold(state, building_blocks, n_steps, precision):
  """Unfolds a state by `n_steps` applications of a given acyclic_graph.

  `building_blocks` will usually be a the return value of `operators.gather_local_building_blocks`.
  See the docstrings of `SevenDiscretedOperator` and `operators.gather_local_building_blocks`
  for more information.

  Args:
    state: The probabilityfunction.
    building_blocks: The acyclic_graph, given as a sequence of `SevenDiscretedOperator`s, that
      represent nearest-neighbour building_blocks collected together.
    n_steps: The number of times the acyclic_graph should be applied.
    precision: Jax matrix multiplication precision.

  Returns:
    U^n_steps |state>, where U is the acyclic_graph defined by `building_blocks`.
  """

  def body(_, state):
    return apply_acyclic_graph(state, building_blocks, precision)

  return jax.lax.fori_loop(0, n_steps, body, state)


@functools.partial(
    pops.pmap,
    in_axes=(0, None),
    out_axes=0,
    static_broadcasted_argnums=(2, 3, 4),
)
def objective_fn_unfold(state, obj_fn, total_time, time_step, precision):
  """Unfolds a state by a local ObjectiveFn, for a given time.

  The unfolding is done by applying `m` copies of the operator
  exp(-i * obj_fn * time_step), where `m = total_time / time_step`. The exponential
  is expanded to 6th order, so the error is O((E * time_step)^7), where E is
  the energy of the state. The action of the 6th-order approximation to the
  exponential is achieved by 6 successive applications of
  |state> -> (1 - i * a_i * time_step * obj_fn) |state>, with a_0, ..., a_5 chosen
  so that all the terms below order-7 come out correctly for the Taylor series
  of the exponential.

  `obj_fn` will usually be a the return value of `operators.gather_local_terms`.
  See the docstrings of `SevenDiscretedOperator` and `operators.gather_local_terms`
  for more information.

  Args:
    state: The probabilityfunction.
    obj_fn: The ObjectiveFn, given as a sequence of `SevenDiscretedOperator`s, that
      represent local terms collected together.
    total_time: The time to unfold for. Should be a multiple of `time_step`.
    time_step: Size of each individual time step. The error is O(time_step^7).
    precision: Jax matrix multiplication precision.

  Returns:
    Approximation to exp(-i * total_time * obj_fn) |state>.
  """
  num_steps_float = total_time / time_step
  num_steps = int(np.round(num_steps_float))
  rounding_eps = np.finfo(type(num_steps_float)).eps
  if abs(num_steps - num_steps_float) > rounding_eps:
    msg = "In objective_fn_unfold, total_time is not a multiple of time_step."
    raise ValueError(msg)

  # Sixth order unfolding formula. Ask guifre@ where the numbers come from.
  a_list = [
      0.37602583 - 0.13347447j,
      0.37602583 + 0.13347447j,
      -0.05612287 - 0.25824122j,
      -0.05612287 + 0.25824122j,
      0.18009704 + 0.30409897j,
      0.18009704 - 0.30409897j,
  ]
  scalar_factors = [-1j * time_step * a for a in a_list]

  def body(_, state):
    for s in scalar_factors:
      state = apply_objective_fn(state, obj_fn, precision, scalar_factor=s)
    return state

  return jax.lax.fori_loop(0, num_steps, body, state)


@functools.partial(
    pops.pmap,
    in_axes=(0, None),
    out_axes=(None, 0),
    static_broadcasted_argnums=(2, 3, 4, 5),
)
def _lanczos(state, obj_fn, num_krylov_vecs, maxiter, gs_iterations, precision):
  """Applies Lanczos to the ObjectiveFn `obj_fn`.

  This function is essentially a wrapper around
  `distla_core.distributed_lanczos.lanczos_iterated_GS`

  Args:
    state: Initial guess.
    obj_fn: The ObjectiveFn. See `apply_objective_fn`.
    num_krylov_vecs: Number of Krylov vectors.
    maxiter: Number of restarts. Contrary to the name, this is *not* the maximum
      number, but this number of restarts is always applied regardless of
      convergence.
    gs_iterations: Number of Gram-Schmidt iterations applied to maintain
      orthonormality. 1 or 2 is sufficient unless the problem is ill
      conditioned.
    precision: Jax matrix multiplication precision.

  Returns:
    Lanczos approximation to the dominant eigenvalue and eigenvector of `obj_fn`.
  """
  energy, state = distributed_lanczos.lanczos_iterated_GS(
      lambda x: apply_objective_fn(x, obj_fn, precision),
      lambda x, y: inner(x, y, precision),
      [],
      state,
      num_krylov_vecs,
      maxiter,
      gs_iterations,
  )
  return energy, state


def find_root_solution(
    obj_fn_np,
    n_discretes,
    precision,
    dynamic_dtype=False,
    n_krylov=100,
    n_iters=2,
    n_ortho=2,
    n_global_discretes=None,
    initial_state=None,
    seed=0,
):
  """Finds the dominant eigenstate of a given local ObjectiveFn.

  This function uses the Lanczos algorithm. It finds the dominant eigenpair, so
  presuming that the user wants the root_solution, they should shift the spectrum
  of the ObjectiveFn downwards (subtract a contribution proportional to the
  identity) if necessary, to make sure that the root_solution energy is the
  largest energy eigenvalue by magnitude. See the `apply_shift` argument of
  `operators.gather_local_terms`.

  `obj_fn` will usually be a the return value of `operators.gather_local_terms`.
  See the docstrings of `SevenDiscretedOperator` and `operators.gather_local_terms`
  for more information.

  Args:
    obj_fn: The ObjectiveFn, given as a sequence of `SevenDiscretedOperator`s, that
      represent local terms collected together. The operators can be
      e.g. Numpy arrays of type float64, and `find_root_solution` will convert
      them to Jax arrays of suitable dtype as necessary.
    n_discretes: System size.
    precision: Jax matrix multiplication precision.
    dynamic_dtype: Optional; If `True`, then the algorithm is first run in
      bfloat16, before using that as the initial guess for the final round
      done in float32. May speed up the search, but usually not by much. `False`
      by default.
    n_krylov: Optional; Number of Krylov vectors to use in Lanczos. 100 by
      default.
    n_iters: Optional; Number of restarts to use in Lanczos. 2 by default.
    n_ortho: Optional; Number of Gram-Schmidt iterations to use in Lanczos, to
      maintain orthonormality. 2 by default.
    n_global_discretes: Optional; Number of global discretes to use. Must be set if
      `initial_state` is `None`. `None` dy default.
    initial_state: Optional; Initial state to start from. `None` by default.
    seed: Optional; Random seed for the initial state. Only used if
      `initial_state is None`. 0 by default.

  Returns:
    energy: An approximation to the dominant eigenvalue of `obj_fn`.
    state: An approximation to the dominant eigenvector of `obj_fn`.
  """
  if dynamic_dtype:
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32
  obj_fn = [term.to_jax(dtype=dtype) for term in obj_fn_np]
  if initial_state is None:
    if n_global_discretes is None:
      msg = "If initial_state is not provided, n_global_discretes must be set."
      raise ValueError(msg)
    state = random_rademacher_state(n_discretes, n_global_discretes, seed, dtype)
  else:
    state = initial_state
    del initial_state
  state = normalize(state, precision)
  energy, state = _lanczos(state, obj_fn, n_krylov, n_iters, n_ortho, precision)
  if dtype == jnp.bfloat16:
    # REDACTED Use logging.info
    print("Done with bfloat16, upgrading to float32.")
    dtype = jnp.float32
    state = state.astype(dtype)
    obj_fn = [term.to_jax(dtype=dtype) for term in obj_fn_np]
    energy, state = _lanczos(state, obj_fn, n_krylov, n_iters, n_ortho, precision)
  return energy, state


def _probabilityfunction_to_matrix(state, left_discretes):
  """Reshapes a state into a DistlaCore matrix, making left_discretes number of discretes
  into a left index, and the remaining into a right index.

  The indices that are reshaped to the left are the ceil(n_global_discretes/2)
  first global indices, plus as many of the last local indices as needed.

  See reduced_density_matrix for more.
  """
  distla_core_grid = config.get_processor_grid()
  n_global_discretes, n_local_discretes = number_of_discretes(state)
  n_discretes = n_global_discretes + n_local_discretes
  right_discretes = n_discretes - left_discretes
  if 2**n_global_discretes != distla_core_grid.size:
    msg = (f"Can't convert a state with {n_global_discretes} global discretes "
           "into a density matrix, when the DistlaCore grid has "
           f"{distla_core_grid.size} elements.")
    raise ValueError(msg)
  left_global_discretes = int(np.round(np.log2(distla_core_grid.shape[0])))
  right_global_discretes = int(np.round(np.log2(distla_core_grid.shape[1])))
  if left_discretes < left_global_discretes:
    msg = (f"Can't create a density matrix with {left_discretes} discretes on the "
           f"left when the DistlaCore grid has {left_global_discretes} global "
           "discretes on the left.")
    raise ValueError(msg)
  if right_discretes < right_global_discretes:
    msg = (f"Can't create a density matrix with {right_discretes} discretes on the "
           f"right when the DistlaCore grid has {right_global_discretes} global "
           "discretes on the right.")
    raise ValueError(msg)
  left_local_discretes = left_discretes - left_global_discretes
  right_local_discretes = right_discretes - right_global_discretes

  permutation = misc.inverse_permutation(distla_core_grid.flatten())
  matrix = jax.lax.pshuffle(state, config.AXIS_NAME, permutation)
  matrix = matrix.reshape((2**right_local_discretes,
                           2**left_local_discretes)).transpose()
  return matrix


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
def reduced_density_matrix(state, traced_discretes, p_sz, precision):
  """Computes a reduced density matrix of a state, as a DistlaCore matrix.

  Only the number of discretes to trace over can be provided, the choice of which
  discretes are traced is fixed: The first ceil(N_G/2) global discretes, where N_G is
  the total number of global discretes, and `traced_discretes - ceil(N_G/2)` of the
  last discretes. With periodic boundaries, this set is contiguous.

  The returned matrix will be distributed in the DistlaCore fashion, meaning that
  one can e.g. use SUMMA to take matmuls with it.

  This function has been pmapped, and should not be further jitted/pmapped. If
  you need an unpmapped version, try working with _probabilityfunction_to_matrix.

  Args:
    state: The probabilityfunction.
    traced_discretes: Number of discretes to trace over.
    p_sz: SUMMA panel size.
    precision: ASIC matmul precision.
  Raises:
    ValueError if `state` isn't utilmaxcut all Jax devices as global discretes.
    ValueError if there are too few or many discretes to trace over, to allow
      returning the result as a DistlaCore matrix.
  Returns:
    The reduced density matrix.
  """
  # REDACTED Add the option of choosing exactly which discretes to trace over.
  matrix = _probabilityfunction_to_matrix(state, traced_discretes)
  matrix = summa.summa(matrix, matrix.conj(), p_sz, True, False, precision)
  return matrix


@functools.partial(pops.pmap, out_axes=None, static_broadcasted_argnums=(1, 2))
def renyi_entropy(matrix, alpha, precision):
  """Computes the alpha Renyi entropy of a reduced density matrix.

  By alpha Renyi entropy we mean 1/(1- alpha) * log2 Tr[matrix^alpha].

  This function has been pmapped, and should not be further jitted/pmapped.

  Args:
    matrix: A reduced density matrix, as a DistlaCore-distributed matrix.
    alpha: The index of Renyi entropy.
    precision: ASIC matmul precision.
  Raises:
    ValueError for invalid choice of alpha.
    NotImplementedError if the choice of alpha is valid, but unimplemented.
  Returns:
    The Renyi entropy.
  """
  if alpha < 0 or alpha == 1:
    raise ValueError(f"Invalid choice of alpha: {alpha}")
  elif alpha != 2:
    msg = f"alpha != 2 not implemented (got alpha = {alpha})"
    raise NotImplementedError(msg)
  else:
    trace = inner(matrix, matrix, precision)
  return jnp.log2(trace) / (1 - alpha)
