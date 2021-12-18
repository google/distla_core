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
"""Tests for invsqrt.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.backends import serial_backend
from distla_core.linalg.invsqrt import invsqrt_utils
from distla_core.linalg.invsqrt.serial import invsqrt

dims = (8, 16, 128)
dtypes = (np.float32, np.complex64)


def random_posdef(D, delta, dtype):
  """
  Returns, as a numpy array, a random `D x D` positive definite matrix of
  `dtype`, that has the ratio of smallest eigenvalue/frobenius norm be
  approximately `delta`. `delta` is assumed to be small compared to 1.
  """
  A = np.random.randn(D, D).astype(dtype)
  if issubclass(dtype, np.complexfloating):
    A = A + 1j * np.random.randn(D, D).astype(dtype)
  A = np.dot(A.T.conj(), A)
  A = A / np.linalg.norm(A)
  Emin = np.min(np.linalg.eigh(A)[0])
  A = A + (delta - Emin) * np.eye(D, dtype=dtype)
  A *= 3 * dtype(np.random.rand())
  return A


def relative_diff_norm(A, B):
  """
  Returns `frobnorm(A - B) / avg(frobnorm(A), frobnorm(B))`.
  """
  A_norm = np.linalg.norm(A)
  B_norm = np.linalg.norm(B)
  diff_norm = np.linalg.norm(A - B)
  return 2 * diff_norm / (A_norm + B_norm)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_magnify_spectrum(dim, dtype):
  """
  Generates a random positive definite `dim x dim` matrix `A` of `dtype`, and
  checks that `invsqrt_backend._magnify_spectrum` correctly inflates the
  spectrum of `[[0, A], [I, 0]]`.
  """
  s_thresh = 0.1
  delta = 1e-3  # Threshold of positive-definiteness
  maxiter = 200
  np.random.seed(0)
  A = random_posdef(dim, delta, dtype)
  A = A / np.linalg.norm(A)
  eye = np.eye(dim, dtype=dtype)
  backend = serial_backend.SerialBackend()

  Y, Z, j = invsqrt_utils._magnify_spectrum(
      A,
      eye,
      maxiter,
      jnp.finfo(A.dtype).eps,
      s_thresh,
      backend,
  )

  assert 0 < j < maxiter
  Y, Z = (np.array(x) for x in (Y, Z))
  zeros = np.zeros((dim, dim))
  B = np.block([[zeros, Y], [Z, zeros]])
  E = np.abs(np.linalg.eigvals(B))
  assert np.all(E > s_thresh)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_invsqrt(dim, dtype):
  """
  Generates a random positive definite `dim x dim` matrix of `dtype`, and checks
  that `invsqrt.invsqrt` returns its square root and inverse square root.
  """
  s_thresh = 0.1
  delta = 1e-3  # Threshold of positive-definiteness
  maxiter = 200
  np.random.seed(0)
  A = random_posdef(dim, delta, dtype)
  Y, Z, jr, jt = invsqrt.invsqrt(A, maxiter=maxiter)
  assert 0 < jr < jt <= maxiter
  YZ = np.dot(Y, Z)
  ZY = np.dot(Z, Y)
  YY = np.dot(Y, Y)
  eye = np.eye(dim, dtype=dtype)
  tol = 100 * jnp.finfo(A.dtype).eps
  np.testing.assert_allclose(YZ, eye, atol=tol)
  np.testing.assert_allclose(ZY, eye, atol=tol)
  np.testing.assert_allclose(YY, A, atol=tol)
