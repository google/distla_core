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
"""Test for polar.py."""
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp

from distla_core.linalg.backends import serial_backend
from distla_core.linalg.polar import polar_utils
from distla_core.linalg.polar.serial import polar
from distla_core.linalg.polar.serial import qdwh
from distla_core.linalg.utils import testutils

shapes = [(8, 8), (12, 8), (16, 16)]


@pytest.mark.parametrize("shape", shapes)
def test_magnify_spectrum(shape):
  m = shape[0]
  n = shape[1]
  s_thresh = 0.1
  matrix = jnp.array(np.random.randn(m, n).astype(np.float32))
  matrix = matrix / jnp.linalg.norm(matrix)
  backend = serial_backend.SerialBackend()
  result, _ = polar_utils._magnify_spectrum(matrix, 200,
                                            jnp.finfo(matrix.dtype).eps,
                                            s_thresh, backend)
  _, svals, _ = np.linalg.svd(np.array(result, dtype=np.float64))
  too_small = svals[svals < s_thresh]
  assert too_small.size == 0


@pytest.mark.parametrize("shape", shapes)
def test_polar(shape):
  m = shape[0]
  n = shape[1]
  matrix = jnp.array(np.random.randn(m, n).astype(np.float32))
  unitary, posdef, _, _, _ = polar.polar(matrix)
  tol = jnp.linalg.norm(matrix) * jnp.finfo(matrix.dtype).eps

  if m >= n:
    unitary2 = jnp.matmul(
        unitary.conj().T, unitary, precision=lax.Precision.HIGHEST)
  else:
    unitary2 = jnp.matmul(
        unitary, unitary.conj().T, precision=lax.Precision.HIGHEST)

  eye_mat = jnp.eye(unitary2.shape[0], dtype=unitary2.dtype)
  testutils.assert_allclose(eye_mat, unitary2, atol=tol)
  testutils.assert_allclose(posdef, posdef.conj().T, atol=10 * tol)

  ev, _ = jnp.linalg.eigh(posdef)
  ev = ev[jnp.abs(ev) > tol]
  negative_ev = jnp.sum(ev < 0.)
  assert negative_ev == 0.

  recon = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
  testutils.assert_allclose(matrix, recon, atol=10 * tol)

  unitary_sp, _ = sp.linalg.polar(np.array(matrix, dtype=np.float64))
  testutils.assert_allclose(unitary_sp, unitary, atol=10 * tol)


@pytest.mark.parametrize("shape", shapes)
def test_polar_qdwh(shape):
  m = shape[0]
  n = shape[1]
  matrix = jnp.array(np.random.randn(m, n).astype(np.float32))
  unitary, posdef, jq, jc = qdwh.polar(matrix)
  tol = jnp.linalg.norm(matrix) * jnp.finfo(matrix.dtype).eps

  if m >= n:
    unitary2 = jnp.matmul(
        unitary.conj().T, unitary, precision=lax.Precision.HIGHEST)
  else:
    unitary2 = jnp.matmul(
        unitary, unitary.conj().T, precision=lax.Precision.HIGHEST)

  eye_mat = jnp.eye(unitary2.shape[0], dtype=unitary2.dtype)
  testutils.assert_allclose(eye_mat, unitary2, atol=tol)

  testutils.assert_allclose(posdef, posdef.conj().T, atol=10 * tol)

  ev, _ = jnp.linalg.eigh(posdef)
  ev = ev[jnp.abs(ev) > tol]
  negative_ev = jnp.sum(ev < 0.)
  assert negative_ev == 0.

  recon = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
  testutils.assert_allclose(matrix, recon, atol=10 * tol)

  unitary_sp, _ = sp.linalg.polar(np.array(matrix, dtype=np.float64))
  testutils.assert_allclose(unitary_sp, unitary, atol=10 * tol)
