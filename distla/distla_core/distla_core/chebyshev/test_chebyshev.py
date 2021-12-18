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
"""Tests for chebyshev.py."""
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.chebyshev import chebyshev
from distla_core.linalg.utils import testutils
from distla_core.utils import pops
from distla_core.utils import vops

precisions = [lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST]

# REDACTED I think pytest has some nice way of doing this properly.
np.random.seed(1)


# REDACTED This function is duplicated in at least two places within
# distla_core. We need something like utils for tests.
def random_posdef(D, delta, dtype):
  """
  Returns, as a numpy array, a random D x D positive definite matrix of
  dtype, that has the ratio of smallest eigenvalue/frobenius norm be
  approximately delta. delta is assumed to be small compared to 1.
  """
  A = np.random.randn(D, D).astype(dtype)
  if issubclass(dtype, np.complexfloating):
    A = A + 1j * np.random.randn(D, D).astype(dtype)
  A = np.dot(A.T.conj(), A)
  A = A / np.linalg.norm(A)
  Emin = np.min(np.linalg.eigh(A)[0])
  A = A + (delta - Emin) * np.eye(D, dtype=dtype)
  A *= dtype(3) * dtype(np.random.rand())
  return A


def random_self_adjoint(D, dtype):
  """Returns, as a numpy array, a random D x D Hermitian matrix of dtype."""
  A = np.random.randn(D, D).astype(dtype)
  if issubclass(dtype, np.complexfloating):
    A = A + 1j * np.random.randn(D, D).astype(dtype)
  A = (A + A.T.conj()) / dtype(2)
  A *= dtype(3) * dtype(np.random.rand())
  return A


def test_return_which():
  """Test the usage of the return_which keyword argument."""

  def f(x):
    return x**2

  interval = (-1, 1)
  D = 16
  n = 1000
  p_sz = 2
  atol = 1e-7
  matmat, scalar1, pmatmat, scalar2, _ = chebyshev.chebyshize(
      f,
      interval,
      p_sz=p_sz,
      n=n,
      return_which=("matmat", "scalar", "pmatmat", "scalar"),
  )
  M = random_self_adjoint(D, np.float32)
  M = M / np.linalg.norm(M)
  np.testing.assert_allclose(scalar1(M), M**2, atol=atol)
  np.testing.assert_allclose(scalar2(M), M**2, atol=atol)
  M_sq_exact = np.dot(M, M)
  M_sq = matmat(M)
  np.testing.assert_allclose(M_sq, M_sq_exact, atol=atol)
  M_dist = pops.distribute(M)
  M_sq_dist = jax.pmap(pmatmat, axis_name=pops.AXIS_NAME)(M_dist)
  M_sq_collected = pops.undistribute(M_sq_dist)
  np.testing.assert_allclose(M_sq_collected, M_sq_exact, atol=atol)


def chebyshev_test(
    f,
    interval,
    M,
    v,
    n_cheb,
    is_vectorized,
    atol,
    rtol,
    test_samples,
    test_margin,
    p_sz,
    precision,
):
  """A utility function for running tests of Chebyshation.

  Chebyshizes the function f, and checks the accuracy of the approximation for
  scalars, the matrix M, M distributed over several devices, and for
  matrix-vector product f(M) @ v for both undistributed and distributed.

  Args:
    f: The function to test on
    interval: The interval to test in
    M: The matrix to test on
    v: The vector to test on
    n_cheb: Order of Chebyshev expansion
    is_vectorized: Whether f is already capabable of handling vector arguments.
    atol: Absolute tolerance for accuracy.
    rtol: Relative tolerance for accuracy.
    test_samples: How many scalar points to sample for testing within the
      interval.
    test_margin: How many points from the ends of the interval to discard.
      Chebyshev expansions are inaccurate near the ends.
    p_sz: Panel size for SUMMA.
    precision: Matmul precision.
  Raises:
    AssertionError if any of the tests fail.
  """
  scalar, matmat, matvec, pmatmat, pmatvec, _ = chebyshev.chebyshize(
      f,
      interval,
      n=n_cheb,
      is_vectorized=is_vectorized,
      p_sz=p_sz,
      precision=precision,
  )
  if is_vectorized:
    f_vec = f
  else:
    f_vec = np.vectorize(f)

  # Skip the first and last few points of the interval, because accuracy there
  # is bad.
  xs = np.linspace(
      interval[0],
      interval[1],
      test_samples,
  )[test_margin:-test_margin - 1]
  ys_exact = f_vec(xs)
  ys_cheb = scalar(xs)
  np.testing.assert_allclose(ys_cheb, ys_exact, rtol=rtol, atol=atol)

  # Apply f exactly to M using an eigenvalue decomposition.
  E, U = jnp.linalg.eigh(M)
  fE = f_vec(E)
  fM_exact = pops.dot(U * fE, U.T.conj())
  fM_cheb = jax.jit(matmat)(M)
  np.testing.assert_allclose(fM_cheb, fM_exact, rtol=rtol, atol=atol)

  M_dist = pops.distribute(M)
  fM_dist = jax.pmap(pmatmat, axis_name=pops.AXIS_NAME)(M_dist)
  fM_collected = pops.undistribute(fM_dist)
  np.testing.assert_allclose(fM_collected, fM_exact, rtol=rtol, atol=atol)

  fMv_exact = pops.dot(fM_exact, v)
  fMv_cheb = jax.jit(matvec)(M, v)
  np.testing.assert_allclose(fMv_cheb, fMv_exact, rtol=rtol, atol=atol)

  v_dist = vops.distribute(v, column_replicated=True)
  fMv_dist = jax.pmap(pmatvec, axis_name=pops.AXIS_NAME)(M_dist, v_dist)
  fMv_collected = vops.undistribute(fMv_dist)
  np.testing.assert_allclose(fMv_collected, fMv_exact, rtol=rtol, atol=atol)


@pytest.mark.parametrize("precision", precisions)
def test_xlogx(precision):
  """Creates a Chebyshev approximation of x log(x) within the interval
  (1e-6, 1), and tests its accuracy for scalars, matrices, and distributed
  matrices.
  """

  def f(x):
    return x * np.log(x)

  is_vectorized = True
  interval = (1e-6, 1.0)
  n_cheb = 200
  # The first one comes from Chebyshev error, the latter from numerical.
  rtol = max(5e-6, 10 * testutils.eps(precision))
  atol = max(5e-6, 10 * testutils.eps(precision))
  test_samples = 1000
  test_margin = 1
  p_sz = 32
  D = 128
  dtype = np.float32
  delta = 1e-4
  M = random_posdef(D, delta, dtype)
  # Ensure the spectrum of M is within the interval.
  M = M / jnp.linalg.norm(M)
  v = np.random.randn(D, 8).astype(dtype)
  chebyshev_test(
      f,
      interval,
      M,
      v,
      n_cheb,
      is_vectorized,
      atol,
      rtol,
      test_samples,
      test_margin,
      p_sz,
      precision=precision,
  )


@pytest.mark.parametrize("precision", precisions)
def test_piecewise_quartic(precision):
  """Creates a Chebyshev approximation of a piecewise quartic function within
  the interval (-2, 2), and tests its accuracy for scalars, matrices, and
  distributed matrices.
  """

  def f(x):
    if x < 0:
      return x**4
    else:
      return -x**4

  is_vectorized = False
  interval = (-2, 2)
  n_cheb = 60
  # The first one comes from Chebyshev error, the latter from numerical.
  rtol = max(5e-5, 10 * testutils.eps(precision))
  atol = max(5e-5, 10 * testutils.eps(precision))
  test_samples = 1000
  test_margin = 0
  p_sz = 8
  D = 128
  dtype = np.float32
  M = random_self_adjoint(D, dtype)
  # Make sure the spectrum of M is within the interval.
  interval_range = max(abs(i) for i in interval)
  M = M / (jnp.linalg.norm(M) / interval_range)
  v = np.random.randn(D, 128).astype(dtype)
  chebyshev_test(
      f,
      interval,
      M,
      v,
      n_cheb,
      is_vectorized,
      atol,
      rtol,
      test_samples,
      test_margin,
      p_sz,
      precision=precision,
  )


@pytest.mark.parametrize("precision", precisions)
def test_piecewise_fermidirac(precision):
  """Creates a Chebyshev approximation of the Fermi-Dirac distribution within
  the interval (-3, 3), and tests its accuracy for scalars, matrices, and
  distributed matrices.
  """
  mu = 0.0
  beta = 10.0

  def f(x):
    return 1 / (np.exp(beta * (x - mu)) + 1)

  is_vectorized = True
  interval = (-3, 3)
  n_cheb = 200
  # The first one comes from Chebyshev error, the latter from numerical.
  rtol = max(5e-6, 10 * testutils.eps(precision))
  atol = max(5e-6, 10 * testutils.eps(precision))
  test_samples = 1000
  test_margin = 0
  p_sz = 16
  D = 128
  dtype = np.float32
  M = random_self_adjoint(D, dtype)
  # Make sure the spectrum of M is within the interval.
  interval_range = max(abs(i) for i in interval)
  M = M / (jnp.linalg.norm(M) / interval_range)
  v = np.random.randn(D, 1).astype(dtype)
  chebyshev_test(
      f,
      interval,
      M,
      v,
      n_cheb,
      is_vectorized,
      atol,
      rtol,
      test_samples,
      test_margin,
      p_sz,
      precision=precision,
  )


def test_trace_estimation():
  """
  Tests trace estimation of a projector using Chebyshev approximation to the
  step function.

  We generate a random Hermitian matrix M, and try to estimate the rank of P(M),
  where P(M) is the projector onto the space of positive eigenvalues of M. This
  is done by computing n_samples copies of v^T @ P(M) @ v, where v is a random
  vector with norm 1. Each of these elements vPv is an approximation to the
  rank, and by taking the average of the n_samples elements we get a more
  accurate estimate.

  Chebyshev polynomials enter as a way of approximating P(M) @ v by only using
  matrix-vector products. The function P(M) is essentially a step function on
  the eigenvalues of M, which is hard to approximate with Chebyshev polynomials
  due to its discontinuity, so instead we replace P(M) with a Fermi-Dirac
  distribution of M, with a very low temperature.
  """
  D = 512
  beta = 100
  n_cheb = 1000
  interval = (-1, 1)
  p_sz = 8
  n_samples = 128
  tol = 1e-2
  dtype = np.float32

  # We approximate the step function by a Fermi-Dirac distribution.
  def f(x):
    return 1 / (np.exp(beta * x) + 1)

  matvec, pmatvec, _ = chebyshev.chebyshize(
      f,
      interval,
      n=n_cheb,
      p_sz=p_sz,
      return_which=("matvec", "pmatvec"),
  )

  # A random Hermitian matrix with spectrum within the given interval.
  M = random_self_adjoint(D, dtype)
  interval_range = max(abs(i) for i in interval)
  M = M / (jnp.linalg.norm(M) / interval_range)
  # A random thin matrix with normalised columns.
  v = np.random.randn(D, n_samples).astype(dtype)
  v = v / np.linalg.norm(v, axis=0)

  E, _ = np.linalg.eigh(M)
  k_exact = np.count_nonzero(E > 0) / D

  Pv = jax.jit(matvec)(M, v)
  k_estimate = jnp.vdot(v, Pv, precision=jax.lax.Precision.HIGHEST) / n_samples
  assert abs(k_exact - k_estimate) < tol

  M_dist = pops.distribute(M)
  v_dist = vops.distribute(v, column_replicated=True)
  Pv_dist = jax.pmap(pmatvec, axis_name=pops.AXIS_NAME)(M_dist, v_dist)
  Pv_collected = vops.undistribute(Pv_dist)
  k_estimate = jnp.vdot(
      v, Pv_collected, precision=jax.lax.Precision.HIGHEST) / n_samples
  assert abs(k_exact - k_estimate) < tol
