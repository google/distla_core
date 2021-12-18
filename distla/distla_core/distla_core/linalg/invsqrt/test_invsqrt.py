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
import functools

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.invsqrt import invsqrt
from distla_core.linalg.invsqrt import invsqrt_utils
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

dims = (8, 64)
dtypes = (np.float32, np.complex64)
# The precision with DEFAULT is so bad that almost no matrix is well-conditioned
# enough, so skip that.
precisions = (lax.Precision.HIGH, lax.Precision.HIGHEST)


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
@pytest.mark.parametrize("precision", precisions)
def test_magnify_spectrum(dim, dtype, precision):
  """
  Generates a random positive definite `dim x dim` matrix `A` of `dtype`, and
  checks that `invsqrt_utils._magnify_spectrum` correctly inflates the
  spectrum of `[[0, A], [I, 0]]`.
  """
  s_thresh = 0.1
  eps = testutils.eps(precision, dtype=dtype)
  delta = 10 * np.sqrt(eps)
  maxiter = 200
  np.random.seed(0)
  A = random_posdef(dim, delta, dtype)
  A = A / np.linalg.norm(A)
  A = pops.distribute(A)
  eye = np.eye(dim, dtype=dtype)
  eye = pops.distribute(eye)
  backend = distributed_backend.DistributedBackend(dim, precision=precision)

  @functools.partial(jax.pmap, out_axes=(0, 0, None), axis_name=pops.AXIS_NAME)
  def magnify_dist(Y, Z):
    return invsqrt_utils._magnify_spectrum(
        Y,
        Z,
        maxiter,
        jnp.finfo(A.dtype).eps,
        s_thresh,
        backend,
    )

  Y, Z, j = magnify_dist(A, eye)
  assert 0 < j < maxiter
  Y = np.array(pops.undistribute(Y))
  Z = np.array(pops.undistribute(Z))
  zeros = np.zeros((dim, dim))
  B = np.block([[zeros, Y], [Z, zeros]])
  E = np.abs(np.linalg.eigvals(B))
  assert np.all(E > s_thresh)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("precision", precisions)
def test_invsqrt(dim, dtype, precision):
  """
  Generates a random positive definite `dim x dim` matrix of `dtype`, and checks
  that `invsqrt.invsqrt` returns its square root and inverse square root.
  """
  eps = testutils.eps(precision, dtype=dtype)
  delta = 10 * np.sqrt(eps)
  maxiter = 200
  np.random.seed(0)
  A_numpy = random_posdef(dim, delta, dtype)
  A = pops.distribute(A_numpy)
  invsqrt_dist = functools.partial(
      invsqrt.invsqrt,
      maxiter=maxiter,
      p_sz=dim,
      precision=precision,
  )
  Y, Z, jr, jt = jax.pmap(
      invsqrt_dist,
      out_axes=(0, 0, None, None),
      axis_name=pops.AXIS_NAME,
  )(A)
  assert 0 < jr < jt <= maxiter
  Y = np.array(pops.undistribute(Y))
  Z = np.array(pops.undistribute(Z))
  YZ = np.dot(Y, Z)
  ZY = np.dot(Z, Y)
  YY = np.dot(Y, Y)
  eye = np.eye(dim, dtype=dtype)
  atol = 10 * eps
  Ynorm = np.linalg.norm(Y)
  np.testing.assert_allclose(YZ, eye, atol=atol)
  np.testing.assert_allclose(ZY, eye, atol=atol)
  np.testing.assert_allclose(YY, A_numpy, atol=Ynorm * Ynorm * atol)
