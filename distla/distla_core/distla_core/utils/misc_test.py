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
"""Test for misc.py."""
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import pytest

from distla_core.linalg.utils import testutils
from distla_core.utils import misc

Ns = [8, 16, 32]
dtypes = [jnp.float32]
precisions = [lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST]
seeds = tuple(range(5))


def test_distance_to_next_divisor():
  assert misc.distance_to_next_divisor(5, 5) == 0
  assert misc.distance_to_next_divisor(6, 5) == 4
  assert misc.distance_to_next_divisor(5, 6) == 1


@pytest.mark.parametrize('N', Ns)
@pytest.mark.parametrize('dtype', dtypes)
@pytest.mark.parametrize('precision', precisions)
def test_similarity_transform(N, dtype, precision):
  np.random.seed(1)
  A = jnp.array(np.random.randn(N, N), dtype=dtype)
  B = jnp.array(np.random.randn(N, N), dtype=dtype)
  normA = jnp.linalg.norm(A)
  normB = jnp.linalg.norm(B)
  eps = testutils.eps(precision, dtype=dtype)
  eps = eps * normA * normB * normB
  expected = jnp.dot(A, B, precision=precision)
  expected = jnp.dot(B.T.conj(), expected, precision=precision)
  result = misc.similarity_transform(A, B, precision)
  testutils.assert_allclose(expected, result, eps)


def test_primes():
  factors = [2, 2, 3, 5, 5, 7, 11]
  np.testing.assert_allclose(misc.prime_factors(np.prod(factors)), factors)


@pytest.mark.parametrize('seed', np.random.randint(1, 100000, 10))
@pytest.mark.parametrize('ndim', np.arange(2, 10))
def test_inv_perm(seed, ndim):
  np.random.seed(seed)
  perm = np.arange(ndim)
  np.random.shuffle(perm)
  inv_perm = misc.inverse_permutation(perm)
  np.testing.assert_allclose(perm[inv_perm], np.arange(ndim))


def test_flatten():
  l = [[1, 2], [3, 4]]
  np.testing.assert_allclose(misc.flatten(l), [1, 2, 3, 4])


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("dtype", dtypes)
def test_gershgorin(N, dtype):
  np.random.seed(10)
  A = jnp.array(np.random.rand(N, N)).astype(dtype)
  A = 0.5 * (A + A.conj().T)
  result_min, result_max = misc.gershgorin(A)
  evs = np.linalg.eigvalsh(A)
  true_min = evs[0]
  true_max = evs[-1]
  assert result_min <= true_min
  assert result_max >= true_max


def test_is_power_of_two():
  powers = [2**i for i in range(8)]
  for i in range(min(powers) - 1, max(powers) + 2):
    result = misc.is_power_of_two(i)
    expected = i in powers
    assert result == expected
