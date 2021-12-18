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
"""Test for ptensordot.py."""
import jax
import numpy as np
import pytest

from distla_core.linalg.tensor import ptensordot
from distla_core.linalg.tensor import utils
from distla_core.linalg.utils import testutils
from distla_core.utils import misc


def get_grid(shape):
  """
  Compute a random processor grid for a tensor with shape
  `shape`.
  """
  primes = misc.prime_factors(jax.device_count())
  tmpshape = list(shape)
  grid = np.ones(len(shape), dtype=np.int32)
  while len(primes) > 0:
    divisible = [
        p for p in range(len(tmpshape)) if tmpshape[p] % primes[0] == 0
    ]
    if len(divisible) == 0:
      raise ValueError(f"could not place any primes {primes} on a "
                       f"tensor leg on tensor a . a.shape = {shape} and "
                       f"jax.device_count() = {jax.device_count()} are "
                       f"incompatible.")

    n = int(np.random.choice(divisible, size=1)[0])
    sa = tmpshape[n]
    if sa % primes[0] == 0:
      prime = primes.pop()
      tmpshape[n] //= prime
      grid[n] *= prime
  return list(grid)


def get_contractable_tensors(
    ndima,
    ndimb,
    num_common,
    dims=(32,),
    dtype=np.float32,
):
  """
  get two contractable tensors and the meta data necessary to contract
  their sharded versions.
  """
  assert num_common <= ndima
  assert num_common <= ndimb
  shape_a = np.random.choice(list(set(dims)), size=ndima, replace=True)
  axes_a = np.random.choice(np.arange(ndima), size=num_common, replace=False)
  axes_b = np.random.choice(np.arange(ndimb), size=num_common, replace=False)
  shape_b = np.random.choice(list(set(dims)), size=ndimb, replace=True)
  shape_b[axes_b] = shape_a[axes_a]
  free_axes_a = sorted(set(range(ndima)) - set(axes_a))
  free_axes_b = sorted(set(range(ndimb)) - set(axes_b))
  shape_c = tuple(shape_a[free_axes_a]) + tuple(shape_b[free_axes_b])
  a = np.random.randn(*shape_a).astype(dtype)
  b = np.random.randn(*shape_b).astype(dtype)
  grid_a = get_grid(shape_a)
  grid_b = get_grid(shape_b)
  grid_c = get_grid(shape_c)
  return a, b, grid_a, grid_b, grid_c, (axes_a, axes_b)


ptensordot_p = jax.pmap(ptensordot.ptensordot,
                        in_axes=(0, 0),
                        static_broadcasted_argnums=(2, 3, 4, 5, 6, 7, 8),
                        axis_name='i')


@pytest.mark.parametrize('ndima, ndimb, axes_len', [(2, 2, 1), (3, 2, 1),
                                                    (3, 3, 1), (3, 3, 2),
                                                    (4, 2, 1), (4, 3, 1),
                                                    (4, 3, 2), (4, 4, 2),
                                                    (4, 4, 3)])
@pytest.mark.parametrize('seed', [10])
def test_ptensordot(ndima, ndimb, axes_len, seed):
  np.random.seed(seed)
  a, b, agrid, bgrid, cgrid, axes = get_contractable_tensors(
      ndima, ndimb, axes_len)
  A = utils.distribute(a, agrid)
  B = utils.distribute(b, bgrid)
  p_sz = 1
  actual = ptensordot_p(A, B, agrid, bgrid, cgrid, axes, p_sz, a.shape, b.shape)
  expected = np.tensordot(a, b, axes)
  actual = utils.undistribute(actual, cgrid, expected.shape)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, np.float32)
  np.testing.assert_allclose(expected, actual, atol=100 * eps, rtol=1.0)
