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
"""Tests for pops.py."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.utils import initializers as init
from distla_core.linalg.utils import testutils
from distla_core.utils import pops
from distla_core.utils import config

DTYPE = jnp.float32
AXIS_NAME = pops.AXIS_NAME
NROW = config.NROWS
NCOL = config.NCOLS
NPROCS = config.NPROCS
GRID = config.GRID

matrix_shapes = [(16, 16), (32, 16), (16, 32)]
Ns = [8, 16, 32]
dtypes = [jnp.float32]


def _local_shape(matrix_shape):
  m = matrix_shape[0] // GRID[0]
  n = matrix_shape[1] // GRID[1]
  return m, n


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_zeros(matrix_shape):
  dtype = np.float32
  actual = init.zeros(matrix_shape, dtype)
  np.testing.assert_allclose(actual, 0.0)
  actualnp = pops.undistribute_global(actual)
  assert actualnp.shape == matrix_shape


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_ones(matrix_shape):
  dtype = np.float32
  actual = init.ones(matrix_shape, dtype)
  np.testing.assert_allclose(actual, 1.0)
  actualnp = pops.undistribute_global(actual)
  assert actualnp.shape == matrix_shape


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("mu", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("sig", [0.5, 1.0, 1.5])
def test_normal(matrix_shape, mu, sig):
  dtype = np.float32
  seed = 0
  sig = dtype(sig)
  mu = dtype(mu)
  actual = init.normal(matrix_shape, dtype=dtype, mu=mu, sigma=sig, seed=seed)
  local_shape = _local_shape(matrix_shape)
  keys = jax.random.split(jax.random.PRNGKey(seed), jax.local_device_count())
  for n in range(jax.local_device_count()):
    expected = jax.random.normal(keys[n], local_shape, dtype=dtype) * sig + mu
    tol = testutils.eps(jax.lax.Precision.HIGHEST, dtype)
    np.testing.assert_allclose(
        actual[n], expected, atol=10 * tol, rtol=10 * tol)

  actualnp = pops.undistribute_global(actual)
  assert actualnp.shape == matrix_shape


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("minval, maxval", [(0.0, 1.0), (-1, 1), (1, 2)])
def test_uniform(matrix_shape, minval, maxval):
  dtype = np.float32
  seed = 0
  actual = init.uniform(
      matrix_shape, dtype=dtype, minval=minval, maxval=maxval, seed=seed)
  local_shape = _local_shape(matrix_shape)
  keys = jax.random.split(jax.random.PRNGKey(seed), jax.local_device_count())
  for n in range(jax.local_device_count()):
    expected = jax.random.uniform(
        keys[n], local_shape, dtype=dtype, minval=minval, maxval=maxval)
    tol = testutils.eps(jax.lax.Precision.HIGHEST, dtype)
    np.testing.assert_allclose(
        actual[n], expected, atol=10 * tol, rtol=10 * tol)

  actualnp = pops.undistribute_global(actual)
  assert actualnp.shape == matrix_shape
