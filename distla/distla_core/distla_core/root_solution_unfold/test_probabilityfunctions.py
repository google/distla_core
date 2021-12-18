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
"""Unit tests for probabilityfunctions.py."""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import pytest

from distla_core.root_solution_unfold import operators
from distla_core.root_solution_unfold import test_utils
from distla_core.root_solution_unfold import probabilityfunctions
from distla_core.linalg.utils import testutils as la_testutils
from distla_core.utils import config
from distla_core.utils import pops

# The parameters used by most tests.
DTYPES = (jnp.float32,)
PRECISIONS = (jax.lax.Precision.HIGHEST,)
SEEDS = (0,)
NUMS_GLOBAL_DISCRETEDS = (0, 1, 2, 3)
SYSTEM_SIZES = tuple(range(12, 18))
BOOLS = (True, False)


def _complex_dtype(dtype):
  """Get the complex version of a real dtype, e.g. float32 -> complex64.
  """
  if dtype == jnp.float32:
    complex_dtype = jnp.complex64
  elif dtype == jnp.float64:
    complex_dtype = jnp.complex128
  else:
    msg = f"Don't know what the complex version of {dtype} is."
    raise ValueError(msg)
  return complex_dtype


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_all_ones_state(n_discretes, n_global_discretes, dtype):
  """Creates an all-ones state, compares to numpy."""
  expected = np.ones((2**n_discretes,), dtype=dtype)
  state = probabilityfunctions.all_ones_state(n_discretes, n_global_discretes, dtype)
  state_np = np.array(state).reshape((2**n_discretes,))
  np.testing.assert_allclose(state_np, expected)


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_random_rademacher_state(n_discretes, n_global_discretes, seed, dtype):
  """Creates a random Rademacher-distributed state, checks that it really is all
  +/-1 and of the right size.
  """
  state = probabilityfunctions.random_rademacher_state(
      n_discretes,
      n_global_discretes,
      seed,
      dtype,
  )
  state_np = np.array(state)
  assert np.prod(state_np.shape) == 2**n_discretes
  np.testing.assert_allclose(np.abs(state_np), 1.0)


norm_pmapped = pops.pmap(
    probabilityfunctions.norm,
    out_axes=None,
    static_broadcasted_argnums=(1,),
)
inner_pmapped = pops.pmap(
    probabilityfunctions.inner,
    out_axes=None,
    static_broadcasted_argnums=(2,),
)


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_norm_and_inner(n_discretes, n_global_discretes, seed, dtype, precision):
  """Creates a random Rademacher state, checks that its norm is correct, that
  `inner` with itself is the norm squared, and that `normalize` normalizes it.
  """
  state = probabilityfunctions.random_rademacher_state(
      n_discretes,
      n_global_discretes,
      seed,
      dtype,
  )
  state_norm = norm_pmapped(state, precision)
  np.testing.assert_allclose(state_norm, 2**(n_discretes / 2))
  np.testing.assert_allclose(
      state_norm,
      np.sqrt(inner_pmapped(state, state, precision)),
  )

  state_normalized = probabilityfunctions.normalize(state, precision)
  np.testing.assert_allclose(norm_pmapped(state_normalized, precision), 1.0)


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_cycle_left_on_ones(n_discretes, n_global_discretes, dtype, precision):
  """Creates an all-ones state and checks that it's invariant under translation.
  """
  state = probabilityfunctions.all_ones_state(n_discretes, n_global_discretes, dtype)
  state_translated = probabilityfunctions.cycle_left(state)
  np.testing.assert_allclose(state, state_translated)
  transinvar = probabilityfunctions.translation_invariance(state, precision)
  np.testing.assert_allclose(transinvar, 0.0)


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_cycle_left_full_circle(
    n_discretes,
    n_global_discretes,
    seed,
    dtype,
    precision,
):
  """Creates a linear combination of all `n_discretes` translations of the the same
  random state, and checks that it's translation invariant.
  """
  state = probabilityfunctions.random_rademacher_state(
      n_discretes,
      n_global_discretes,
      seed,
      dtype,
  )
  state_symmetrised = state
  for _ in range(n_discretes - 1):
    state = probabilityfunctions.cycle_left(state)
    state_symmetrised = state_symmetrised + state
  transinvar = probabilityfunctions.translation_invariance(
      state_symmetrised,
      precision,
  )
  np.testing.assert_allclose(transinvar, 0.0)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_measure_energy(n_global_discretes, seed, dtype, precision):
  """Checks the energy of the MaxCut ObjectiveFn on an all-ones state."""
  np.random.seed(seed)
  n_discretes = operators.min_n_discretes_objective_fn(n_global_discretes, 2)
  neigh_coupling = np.random.rand(n_discretes)
  onsite_coupling = np.random.rand(n_discretes)
  local_terms, shift = test_utils.global_maxcut_objective_fn(
      neigh_coupling,
      onsite_coupling,
      n_discretes,
  )
  assert shift == 0
  obj_fn = operators.gather_local_terms(local_terms, n_global_discretes)
  state = probabilityfunctions.normalize(
      probabilityfunctions.all_ones_state(n_discretes, n_global_discretes, dtype),
      precision,
  )
  energy = probabilityfunctions.measure_energy(state, obj_fn, precision)
  energy += shift * n_discretes
  energy_expected = np.sum(neigh_coupling)
  rtol = 100 * la_testutils.eps(precision, dtype)
  np.testing.assert_allclose(energy, energy_expected, rtol=rtol)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("term_width", (2, 3))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_find_root_solution(n_global_discretes, term_width, seed, dtype, precision):
  """Finds the ground state of the ObjectiveFn sum_i -Z_i, checks that the
  energy and the state are correct.
  """
  n_discretes = operators.min_n_discretes_objective_fn(n_global_discretes, term_width)
  neigh_coupling = 0.0
  onsite_coupling = -1.0
  local_terms, shift = test_utils.global_maxcut_objective_fn(
      neigh_coupling,
      onsite_coupling,
      n_discretes,
      boundary="open",
      apply_shift=True,
  )
  # Pad the terms with identity to make them of the desired width.
  while local_terms[0].shape[0] < 2**term_width:
    eye = np.eye(2, dtype=local_terms[0].dtype)
    local_terms = [np.kron(term, eye) for term in local_terms]
  obj_fn = operators.gather_local_terms(local_terms, n_global_discretes)
  expected_energy = onsite_coupling * n_discretes
  expected_state = np.zeros((2**n_discretes,), dtype=dtype)
  expected_state[0] = 1
  # Since in the test we can only afford a few iterations, we start from a state
  # that is already close to the right one.
  initial_state = probabilityfunctions.random_rademacher_state(
      n_discretes,
      n_global_discretes,
      seed,
      dtype,
  )
  initial_state = pops.pmap(lambda x, y: x + 1e-2 * y)(
      expected_state.reshape(initial_state.shape),
      initial_state,
  )
  # REDACTED The reason for only doing dynamic_dtype=False is that device
  # spoofing doesn't support bf16 AllReduce. If that ever gets fixed, start
  # testing both True and False.
  energy, state = probabilityfunctions.find_root_solution(
      obj_fn,
      n_discretes,
      precision,
      dynamic_dtype=False,
      n_krylov=10,
      initial_state=initial_state,
  )
  energy += shift * n_discretes
  energy_measured = probabilityfunctions.measure_energy(state, obj_fn, precision)
  energy_measured += shift * n_discretes
  rtol = 20 * la_testutils.eps(precision, dtype)
  np.testing.assert_allclose(energy, energy_measured, rtol=rtol)
  state = probabilityfunctions.normalize(state, precision)
  np.testing.assert_allclose(energy, expected_energy, rtol=rtol)
  fidelity = abs(np.vdot(expected_state, np.array(state)))
  np.testing.assert_allclose(fidelity, 1.0, rtol=rtol)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("term_width", (2, 3))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_objective_fn_unfold(
    n_global_discretes,
    term_width,
    seed,
    dtype,
    precision,
):
  """Unfolds a random state by a random ObjectiveFn for a short time. Checks
  that the norm remains 1 and that the state hasn't changed much.
  """
  # REDACTED Would be useful to have some time unfolding where we can
  # unfold by very little time, and know the exact solution to compare to. Here
  # we just check that the code runs and the state isn't changed in any crazy
  # way, but not whether the unfolding is actually correct.
  n_discretes = operators.min_n_discretes_objective_fn(n_global_discretes, term_width)
  np.random.seed(seed)
  time_step = 1e-4
  n_steps = 2
  local_terms = []
  dim = 2**term_width
  for _ in range(n_discretes):
    term = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    term = term + term.T.conj()
    local_terms.append(term)
  obj_fn = operators.gather_local_terms(local_terms, n_global_discretes)
  obj_fn = [term.to_jax(dtype=dtype) for term in obj_fn]
  state = probabilityfunctions.normalize(
      probabilityfunctions.random_rademacher_state(
          n_discretes,
          n_global_discretes,
          seed,
          _complex_dtype(dtype),
      ),
      precision,
  )
  state_unfoldd = probabilityfunctions.objective_fn_unfold(
      state,
      obj_fn,
      n_steps * time_step,
      time_step,
      precision,
  )
  rtol = 10 * la_testutils.eps(precision, dtype)
  norm_unfoldd = norm_pmapped(state_unfoldd, precision)
  np.testing.assert_allclose(norm_unfoldd, 1.0, rtol=rtol)
  fidelity = inner_pmapped(state, state_unfoldd, precision)
  rtol = 10 * time_step * n_steps
  np.testing.assert_allclose(fidelity, 1.0, rtol=rtol)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_acyclic_graph_unfold(n_global_discretes, dtype, precision):
  """Unfolds the all-ones state, that is the state |+>^n_discretes, by a acyclic_graph
  that applies the Hadamard building_block to each site once. Checks that the outcome is
  the state zerovector^n_discretes.
  """
  n_discretes = operators.min_n_discretes_acyclic_graph(n_global_discretes)
  # Brickwork acyclic_graphs only work with even system size
  n_discretes += (n_discretes % 2)
  hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2)
  eye = np.eye(2)
  local_building_blocks = [np.kron(hadamard, eye)] * n_discretes
  acyclic_graph = operators.gather_local_building_blocks(local_building_blocks, n_global_discretes)
  acyclic_graph = [building_block.to_jax(dtype) for building_block in acyclic_graph]
  state = probabilityfunctions.normalize(
      probabilityfunctions.all_ones_state(
          n_discretes,
          n_global_discretes,
          _complex_dtype(dtype),
      ),
      precision,
  )
  state_unfoldd = probabilityfunctions.acyclic_graph_unfold(
      state,
      acyclic_graph,
      1,
      precision,
  )
  rtol = 10 * la_testutils.eps(precision, dtype)
  expected_state = np.zeros((2**n_discretes,), dtype=dtype)
  expected_state[0] = 1
  fidelity = abs(np.vdot(expected_state, np.array(state_unfoldd)))
  np.testing.assert_allclose(fidelity, 1.0, rtol=rtol)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_acyclic_graph_unfold_translation(n_global_discretes, seed, dtype, precision):
  """Tests that acyclic_graph unfold respects 2-translation invariance when the
  initial state is translation invariant and all of the building_blocks in the acyclic_graph
  are identical.
  """
  n_discretes = operators.min_n_discretes_acyclic_graph(n_global_discretes)
  # Brickwork acyclic_graphs only work with even system size
  n_discretes += (n_discretes % 2)
  # Initial state is translation invariant
  state = probabilityfunctions.normalize(
      probabilityfunctions.all_ones_state(n_discretes, n_global_discretes, dtype),
      precision,
  )
  # Random real two-discrete building_block
  np.random.seed(seed)
  U = scipy.stats.special_ortho_group.rvs(4)
  # The acyclic_graph is 2-translation invariant
  local_building_blocks = [U] * n_discretes
  building_blocks = operators.gather_local_building_blocks(local_building_blocks, n_global_discretes)
  building_blocks = [building_block.to_jax(dtype) for building_block in building_blocks]
  # Apply the acyclic_graph
  state = probabilityfunctions.acyclic_graph_unfold(state, building_blocks, 1, precision)
  # Get the 2-translated state
  trans_state = probabilityfunctions.cycle_left(probabilityfunctions.cycle_left(state))
  # Overlap should be 1
  ovlp = pops.pmap(
      probabilityfunctions.inner,
      out_axes=None,
      static_broadcasted_argnums=(2,),
  )(state, trans_state, precision)
  # This high tolerance works around an observed degradation of accuracy when
  # spoofing multiple devices on CPU
  rtol = 1e-2
  np.testing.assert_allclose(ovlp, 1., rtol=rtol)


@pytest.mark.parametrize("n_global_discretes", NUMS_GLOBAL_DISCRETEDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_acyclic_graph_unfold_basis_order(n_global_discretes, seed, dtype, precision):
  """Tests that the basis order is what we expect it to be after applying a
  acyclic_graph (global discretes come first, before local discretes).
  """
  n_discretes = operators.min_n_discretes_acyclic_graph(n_global_discretes)
  # Brickwork acyclic_graphs only work with even system size
  n_discretes += (n_discretes % 2)
  # Product X = +1 state initially
  state = probabilityfunctions.normalize(
      probabilityfunctions.all_ones_state(n_discretes, n_global_discretes, dtype),
      precision,
  )
  # Generate random real one-discrete building_blocks
  np.random.seed(seed)
  building_blocks_1q = [scipy.stats.special_ortho_group.rvs(2) for i in range(n_discretes)]
  # Construct the brickwork acyclic_graph
  eye = np.eye(2)
  building_blocks_2q = [np.kron(u, eye) for u in building_blocks_1q]
  building_blocks = operators.gather_local_building_blocks(building_blocks_2q, n_global_discretes)
  building_blocks = [building_block.to_jax(dtype) for building_block in building_blocks]
  # Apply the acyclic_graph
  state = probabilityfunctions.acyclic_graph_unfold(state, building_blocks, 1, precision)
  # Covert to numpy array for later comparison
  state = np.array(state).reshape(2**n_discretes)

  # Compute the state in a different way for comparison
  state_1q = np.ones(2) / np.sqrt(2)
  states_1q = [np.einsum("i,ij->j", state_1q, u) for u in building_blocks_1q]
  kron_state = operators._kron_fold(states_1q)

  # Compare the states obtained in different ways
  ovlp = np.vdot(state, kron_state)
  rtol = 10 * la_testutils.eps(precision, dtype)
  np.testing.assert_allclose(ovlp, 1, rtol=rtol)


@pytest.mark.parametrize("n_discretes", SYSTEM_SIZES)
@pytest.mark.parametrize("less_than_half_traced", (0, 1, 3))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_reduced_density_matrix(n_discretes, less_than_half_traced, seed, dtype):
  """Tests that reduced_density_matrix agrees with the Numpy result."""
  p_sz = 256
  precision = jax.lax.Precision.HIGHEST
  np.random.seed(seed)
  n_traced = n_discretes // 2 - less_than_half_traced
  n_untraced = n_discretes - n_traced
  # The conversion to DistlaCore matrix doesn't work if all devices aren't
  # utilised.
  n_global_discretes = int(np.round(np.log2(config.NPROCS)))
  n_local_discretes = n_discretes - n_global_discretes
  state_shape = 2**n_global_discretes, 2**n_local_discretes
  state_np = np.random.randn(*state_shape)
  if dtype in (jnp.complex64, jnp.complex128):
    state_np = state_np + 1j * np.random.randn(*state_shape)
  state_np /= np.linalg.norm(state_np)
  matrix = probabilityfunctions.reduced_density_matrix(
      jnp.array(state_np, dtype=dtype),
      n_traced,
      p_sz,
      precision,
  )
  # Choose the discretes to trace over to match the choice in
  # reduced_density_matrix.
  traced_global_discretes = int(np.ceil(n_global_discretes / 2))
  traced_local_discretes = n_traced - traced_global_discretes
  untraced_discretes = n_discretes - traced_local_discretes - traced_global_discretes
  state_np = state_np.reshape((2**traced_global_discretes, 2**untraced_discretes,
                               2**traced_local_discretes))
  matrix_np = np.tensordot(state_np, state_np.conj(), axes=((0, 2), (0, 2)))
  matrix = pops.undistribute(matrix, collect_to_host=True)
  assert matrix.shape[0] == matrix.shape[1] == 2**n_untraced
  tol = 10 * la_testutils.eps(precision, dtype)
  np.testing.assert_allclose(matrix, matrix.T.conj(), rtol=tol, atol=tol)
  np.testing.assert_allclose(matrix, matrix_np, rtol=tol, atol=tol)


@pytest.mark.parametrize("dim", (16, 128))
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("alpha", (2,))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_renyi_entropy(dim, seed, alpha, dtype):
  """Tests that renyi_entropy agrees with the Numpy result."""
  precision = jax.lax.Precision.HIGHEST
  np.random.seed(seed)
  if alpha != 2:
    raise NotImplementedError("alpha != 2 not implemented.")

  matrix_np = np.random.randn(dim, dim)
  if dtype in (jnp.complex64, jnp.complex128):
    matrix_np = matrix_np + 1j * np.random.randn(dim, dim)
  matrix_np = np.dot(matrix_np, matrix_np.conjubuilding_block().transpose())
  matrix_np /= np.trace(matrix_np)

  expected = np.log2(np.trace(np.dot(matrix_np, matrix_np))) / (1 - alpha)
  matrix = pops.distribute(matrix_np)
  result = probabilityfunctions.renyi_entropy(matrix, alpha, precision)
  tol = 10 * la_testutils.eps(precision, dtype)
  np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)
