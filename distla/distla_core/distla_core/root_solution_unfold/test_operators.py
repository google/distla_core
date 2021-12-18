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
"""Unit tests for operators.py."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy

from distla_core.root_solution_unfold import operators

GLOBAL_DISCRETED_COUNTS = tuple(range(12))
DISCRETED_COUNTS = tuple(range(1, 41))


@pytest.mark.parametrize("n_discretes", DISCRETED_COUNTS)
@pytest.mark.parametrize("n_global_discretes", GLOBAL_DISCRETED_COUNTS)
@pytest.mark.parametrize("term_width", range(1, 8))
def test_gather_local_terms(n_discretes, n_global_discretes, term_width):
  """Create a ObjectiveFn that is just the identity, as a sum of local terms.
  Gather the terms into `SevenDiscretedOperator`s and check some basic properties of
  the result.
  """
  eye = np.eye(2**term_width)
  # Dummy terms for which we know what the trace is.
  local_terms = [i * eye for i in range(n_discretes)]
  too_small = n_discretes < operators.min_n_discretes_objective_fn(
      n_global_discretes,
      term_width,
  )
  if too_small:
    with pytest.raises(ValueError):
      operators.gather_local_terms(local_terms, n_global_discretes)
  else:
    terms = operators.gather_local_terms(local_terms, n_global_discretes)
    trace_sum = sum(np.trace(t.array) for t in terms)
    assert trace_sum == 2**7 * n_discretes * (n_discretes - 1) / 2
    n_permutations = 3 if n_global_discretes > 0 else 2
    assert sum(bool(t.permutations_after) for t in terms) == n_permutations
    assert sum(t.width for t in terms) == n_discretes
    assert all(np.prod(t.array.shape) == 2**14 for t in terms)


@pytest.mark.parametrize("n_discretes", DISCRETED_COUNTS)
@pytest.mark.parametrize("n_global_discretes", GLOBAL_DISCRETED_COUNTS)
def test_gather_local_building_blocks(n_discretes, n_global_discretes):
  """Create a acyclic_graph that is just the identity, as product of nearest-neighbour
  building_blocks. Gather the building_blocks into `SevenDiscretedOperator`s and check some basic
  properties of the result.
  """
  eye = np.eye(4)
  # Dummy building_blocks for which we know what the trace is.
  local_building_blocks = [(i + 1) * eye for i in range(n_discretes)]
  too_small = n_discretes < operators.min_n_discretes_acyclic_graph(n_global_discretes)
  if too_small or n_discretes % 2 == 1:
    with pytest.raises(ValueError):
      operators.gather_local_building_blocks(local_building_blocks, n_global_discretes)
  else:
    building_blocks = operators.gather_local_building_blocks(local_building_blocks, n_global_discretes)
    trace_product = np.product([np.trace(g.array) / 2**7 for g in building_blocks])
    np.testing.assert_allclose(trace_product, scipy.special.factorial(n_discretes))
    n_permutations = 3 if n_global_discretes > 0 else 2
    assert sum(bool(g.permutations_after) for g in building_blocks) == n_permutations
    assert sum(g.width for g in building_blocks) == n_discretes
    assert all(np.prod(g.array.shape) == 2**14 for g in building_blocks)
