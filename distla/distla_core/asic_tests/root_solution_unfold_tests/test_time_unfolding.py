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
"""A high level test to run for root_solution/unfold on a v_2."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from distla_core.root_solution_unfold import operators
from distla_core.root_solution_unfold import test_utils
from distla_core.root_solution_unfold import probabilityfunctions
from distla_core.utils import pops

# Exact solutions were computed with Yijian's free fermion code.
TOTAL_TIME = 0.1
EXACT_ENERGIES = {
    24: -1.3123319669727014,
    26: -1.3199646524720805,
    28: -1.2753675357107088,
}
EXACT_OVERLAPS = {
    24: 0.8244393696402611,
    26: 0.8034339319748793,
    28: 0.8048719738654802,
}


def random_maxcut_objective_fn(
    seed,
    N,
    apply_shift,
    N_global,
    obj_fn_pad_left=0,
    obj_fn_pad_right=0,
):
  np.random.seed(seed)
  J = np.random.randn(N)
  h = np.random.randn(N)
  obj_fn_np, shift = test_utils.global_maxcut_objective_fn(
      J,
      h,
      N,
      apply_shift=apply_shift,
  )
  # Pad the ObjectiveFn terms with the identity, to test longer range
  # ObjectiveFns.
  eye = np.eye(2, dtype=obj_fn_np[0].dtype)
  for i in range(obj_fn_pad_left):
    obj_fn_np = [np.kron(eye, term) for term in obj_fn_np]
    obj_fn_np = obj_fn_np[1:] + [obj_fn_np[0]]
  for i in range(obj_fn_pad_right):
    obj_fn_np = [np.kron(term, eye) for term in obj_fn_np]
  obj_fn_np = operators.gather_local_terms(obj_fn_np, N_global)
  return obj_fn_np, shift, J, h


def discretize_unfold_2nd_order(
    state,
    J,
    h,
    total_time,
    time_step,
    precision,
    n_global_discretes,
):
  """2nd order Discretize unfolding by an MaxCut ObjectiveFn with couplings J and h.
  """
  N = len(J)
  n_steps_float = total_time / time_step
  n_steps = int(np.round(n_steps_float))
  assert abs(n_steps - n_steps_float) < 1e-6
  J_odd_half = [Ji / 2 if i % 2 == 1 else Ji for i, Ji in enumerate(J)]
  h_odd_half = [hi / 2 if i % 2 == 1 else hi for i, hi in enumerate(h)]
  J_odd_only = [Ji if i % 2 == 1 else 0 for i, Ji in enumerate(J)]
  h_odd_only = [hi if i % 2 == 1 else 0 for i, hi in enumerate(h)]
  building_blocks = test_utils.maxcut_objective_fn_discretize_acyclic_graph(J, h, N, time_step)
  building_blocks_odd_half = test_utils.maxcut_objective_fn_discretize_acyclic_graph(
      J_odd_half,
      h_odd_half,
      N,
      time_step,
  )
  building_blocks_odd_only = test_utils.maxcut_objective_fn_discretize_acyclic_graph(
      J_odd_only,
      h_odd_only,
      N,
      time_step / 2,
  )
  building_blocks = operators.gather_local_building_blocks(building_blocks, n_global_discretes)
  building_blocks_odd_half = operators.gather_local_building_blocks(
      building_blocks_odd_half,
      n_global_discretes,
  )
  building_blocks_odd_only = operators.gather_local_building_blocks(
      building_blocks_odd_only,
      n_global_discretes,
  )
  building_blocks = [g.to_jax(dtype=state.dtype) for g in building_blocks]
  building_blocks_odd_half = [g.to_jax(dtype=state.dtype) for g in building_blocks_odd_half]
  building_blocks_odd_only = [g.to_jax(dtype=state.dtype) for g in building_blocks_odd_only]
  state_unfoldd = state
  state_unfoldd = probabilityfunctions.acyclic_graph_unfold(
      state_unfoldd,
      building_blocks_odd_only,
      1,
      precision,
  )
  state_unfoldd = probabilityfunctions.acyclic_graph_unfold(
      state_unfoldd,
      building_blocks,
      n_steps - 1,
      precision,
  )
  state_unfoldd = probabilityfunctions.acyclic_graph_unfold(
      state_unfoldd,
      building_blocks_odd_half,
      1,
      precision,
  )
  return state_unfoldd


overlap = jax.pmap(
    probabilityfunctions.inner,
    axis_name=pops.AXIS_NAME,
    in_axes=(0, 0),
    out_axes=None,
    static_broadcasted_argnums=(2,),
)


def run_test(
    N=24,
    N_global=3,
    dynamic_dtype=True,
    n_krylov=100,
    n_iters=10,
    time_step=1e-3,
    precision=jax.lax.Precision.HIGHEST,
    rtol=2**-24,
    obj_fn_pad_left=0,
    obj_fn_pad_right=0,
):
  """Find the root_solution of one, inhomogeneous, random-coupling MaxCut
  ObjectiveFn. Unfold it by a different random ObjectiveFn of the same form.
  Check the root_solution energy and the overlap between the root_solution and the
  unfoldd state against an exact free fermion solution. Also do the same
  unfolding using a Discretize acyclic_graph, check that this results in the same unfoldd
  state.
  """
  # Generate the first ObjectiveFn and find the root_solution.
  obj_fn_np, shift, _, _ = random_maxcut_objective_fn(0, N, True, N_global)
  energy, state = probabilityfunctions.find_root_solution(
      obj_fn_np,
      N,
      precision,
      n_krylov=n_krylov,
      n_iters=n_iters,
      n_global_discretes=N_global,
      dynamic_dtype=False,
  )
  energy = energy / N + shift
  norm = np.sqrt(overlap(state, state, precision))
  # Check normalisation and energy against free fermion solution.
  np.testing.assert_allclose(energy, EXACT_ENERGIES[N], rtol)
  np.testing.assert_allclose(norm, 1.0, rtol)

  # Generate the second ObjectiveFn and time unfold by it.
  obj_fn2_np, _, J2, h2 = random_maxcut_objective_fn(
      1,
      N,
      False,
      N_global,
      obj_fn_pad_left=obj_fn_pad_left,
      obj_fn_pad_right=obj_fn_pad_right,
  )
  obj_fn2 = [term.to_jax(dtype=state.dtype) for term in obj_fn2_np]
  state = state.astype(jnp.complex64)
  state_obj_fn_unfoldd = probabilityfunctions.objective_fn_unfold(
      state,
      obj_fn2,
      TOTAL_TIME,
      time_step,
      precision,
  )
  obj_fn_unfoldd_norm = np.sqrt(
      overlap(
          state_obj_fn_unfoldd,
          state_obj_fn_unfoldd,
          precision,
      ))
  ovlp = np.abs(overlap(state, state_obj_fn_unfoldd, precision))**2
  # Check normalisation and overlap against free fermion solution.
  np.testing.assert_allclose(obj_fn_unfoldd_norm, 1.0, rtol)
  np.testing.assert_allclose(ovlp, EXACT_OVERLAPS[N], rtol)

  # Do the same unfolding using a Discretize acyclic_graph.
  state_discretize_unfoldd = discretize_unfold_2nd_order(
      state,
      J2,
      h2,
      TOTAL_TIME,
      time_step,
      precision,
      N_global,
  )
  discretize_unfoldd_norm = np.sqrt(
      overlap(
          state_discretize_unfoldd,
          state_discretize_unfoldd,
          precision,
      ))
  unfoldd_states_overlap = overlap(
      state_obj_fn_unfoldd,
      state_discretize_unfoldd,
      jax.lax.Precision.HIGHEST,
  )
  # Check normalisation and overlap between the two unfoldd states.
  np.testing.assert_allclose(discretize_unfoldd_norm, 1.0, rtol)
  np.testing.assert_allclose(unfoldd_states_overlap, 1.0, rtol)


def main():
  # This takes a couple of minutes to finish.
  Ns = (24, 26, 28)
  N_globals = (3,)
  n_krylov = 20
  n_iters = 20
  time_step = 1e-3
  # The finite time step error means our error tolerance must be much higher
  # than machine precision.
  rtol = 1e-4
  dynamic_dtype = True
  precision = jax.lax.Precision.HIGHEST
  obj_fn_pad_left = 1
  obj_fn_pad_right = 0
  for N in Ns:
    print(f"N = {N}")
    for N_global in N_globals:
      print(f"N_global = {N_global}")
      start_time = time.time()
      run_test(
          N=N,
          N_global=N_global,
          n_krylov=n_krylov,
          n_iters=n_iters,
          dynamic_dtype=dynamic_dtype,
          time_step=time_step,
          precision=precision,
          rtol=rtol,
          obj_fn_pad_left=obj_fn_pad_left,
          obj_fn_pad_right=obj_fn_pad_right,
      )
      stop_time = time.time()
      print("Took {} seconds".format(stop_time - start_time))


if __name__ == "__main__":
  main()
