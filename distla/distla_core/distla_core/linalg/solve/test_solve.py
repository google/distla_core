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
"""Test for solve.py."""
import functools

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest
import scipy as sp

from distla_core.linalg.solve import solve
from distla_core.linalg.utils import testutils
from distla_core.utils import pops
from distla_core.utils import vops

AXIS_NAME = pops.AXIS_NAME

NCOL = pops.NCOLS
NROW = pops.NROWS
NPROC = pops.NPROCS
dims = [4, 16]
seeds = [1, 2]
precisions = [lax.Precision.HIGHEST, ]
dtypes = [np.float32, ]


##############################################################################
# HELPERS
##############################################################################
def _dephase_r(R):
  """ Maps the R factor from an arbitrary QR decomposition to the unique one
  with non-negative diagonal entries.
  """
  phases_data = np.sign(np.diagonal(R))
  phases = np.ones((max(R.shape)))
  phases[:phases_data.size] = phases_data
  R = phases.conj()[:, None] * R
  return R


def _arnoldi_assert_one(A, V, H, j, tol):
  lhs = np.dot(A, V[:, :j])
  rhs = np.dot(V[:, :j + 1], H[:j + 1, :j])
  testutils.assert_allclose(lhs, rhs, atol=tol)


def _arnoldi_assert_two(A, V, H, j, tol):
  lhs = H[:j, :j]
  rhs = np.dot(V[:, :j].conj().T, A)
  rhs = np.dot(rhs, V[:, :j])
  testutils.assert_allclose(lhs, rhs, atol=tol)


def _gmres_update_np(X, V, R, beta):
  Y = sp.linalg.solve_triangular(R[:-1, :], beta[:-1])
  dX = np.dot(V[:, :-1], Y).reshape(X.shape)
  return X + dX


##############################################################################
# GIVENS QR
##############################################################################
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("dtype", dtypes)
def test_givens(seed, dtype):
  """ Tests that when the Givens factors produced by
  `solve._compute_givens_rotation` are applied to a two-vector
  `v = (a, b)` via `solve._apply_ith_rotation`, the result is
  `h = (||v||, 0)` up to a sign.
  """
  np.random.seed(seed)
  v = np.random.randn(2).astype(dtype)
  v = jnp.array(v)
  cs, sn = solve._compute_givens_rotation(v[0], v[1])
  cs = jnp.full(1, cs)
  sn = jnp.full(1, sn)
  r = np.sqrt(v[0] ** 2 + v[1] ** 2)
  h, _, _ = solve._apply_ith_rotation(0, (v, cs, sn))
  expected = np.array([r, 0.])
  eps = jnp.finfo(dtype).eps * r
  testutils.assert_allclose(np.abs(h), expected, atol=eps)


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_hessenberg_qr(seed, dim, dtype):
  """ Tests that solve._update_hessenberg_qr correctly performs the
  QR factorization of an upper Hessenberg matrix.
  """
  np.random.seed(seed)
  H = np.random.randn(dim + 1, dim).astype(dtype)
  H = jnp.triu(H, k=-1)
  H_r = jnp.zeros_like(H)
  cs = jnp.zeros(dim, dtype=dtype)
  sn = jnp.zeros(dim, dtype=dtype)
  tol = jnp.finfo(dtype).eps * np.linalg.cond(H)
  for j in range(dim):
    H_r, cs, sn = solve._update_hessenberg_qr(H_r, H, cs, sn, j)

    this_H_r = np.array(H_r)[:j+2, :j+1]
    this_H = np.array(H)[:j+2, :j+1]

    # H_r is upper triangular.
    testutils.assert_allclose(this_H_r, np.triu(this_H_r), atol=tol)

    # Agreement with NumPy up to a phase.
    _, expected_R = np.linalg.qr(this_H)
    expected_R = _dephase_r(expected_R)
    result_R = _dephase_r(this_H_r[:-1, :])
    testutils.assert_allclose(expected_R, result_R, tol)


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_cgs(dim, seed, dtype):
  j = dim // 2
  np.random.seed(seed)
  Vs = np.random.randn(dim, dim).astype(dtype)
  Vs, _ = np.linalg.qr(Vs)
  Vs[:, j:] = 0.
  V_v = vops.distribute(Vs)
  new_v = np.random.randn(dim, 1).astype(dtype)
  new_v = vops.distribute(new_v)

  @functools.partial(pops.pmap, out_axes=(0, None, None))
  def cgs_f(new_v, V_v):
    return solve.cgs(new_v, V_v)

  orth_v, _, _ = cgs_f(new_v, V_v)
  orth_v = vops.undistribute(orth_v)
  Vs[:, j] = orth_v.ravel()

  testutils.test_unitarity(
    Vs[:, :j + 1], eps_coef=np.linalg.cond(Vs[:, :j + 1]) * 10)


# Arnoldi iteration
def test_arnoldi_cond():
  """ Checks that the Arnoldi loop condition obeys the correct logic.
  """
  maxiter = 2
  tol = 1.0

  err_1 = tol * 2
  err_2 = tol // 2
  err_3 = tol
  errs = [err_1, err_2, err_3]
  err_gt_tol = [True, False, False]

  j_1 = maxiter * 2
  j_2 = maxiter // 2
  j_3 = maxiter
  js = [j_1, j_2, j_3]
  j_lt_maxiter = [False, True, False]

  for err, err_bool in zip(errs, err_gt_tol):
    for j, j_bool in zip(js, j_lt_maxiter):
      err = jnp.full(1, err)
      j = jnp.full(1, j)
      expected = jnp.logical_and(err_bool, j_bool)
      args = (0, 0, 0, 0, 0, 0, err, j)
      result = solve._arnoldi_cond(maxiter, tol, args)
      assert expected == result


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("dim", [4, ])
@pytest.mark.parametrize("N_k", [3, ])
@pytest.mark.parametrize("j", [1, ])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("precision", precisions)
def test_update_arnoldi(dim, N_k, seed, j, dtype, precision):
  """ Tests that solve._update_arnoldi puts vectors in the correct places.
  """
  np.random.seed(seed)
  V = np.random.randn(dim, N_k).astype(dtype)
  V[:, j + 1:] = 0.
  H_s = np.random.randn(N_k + 1, N_k).astype(dtype)
  H_s_data = np.random.randn(j + 1, j).astype(dtype)
  H_s_data = np.triu(H_s_data, k=-1)
  H_s = np.zeros((N_k + 1, N_k), dtype=dtype)
  H_s[:j + 1, :j] = H_s_data

  orth = np.random.randn(dim, 1).astype(dtype)
  overlaps = np.random.randn(N_k + 1, 1).astype(dtype)
  overlaps[j + 1:, :] = 0.
  norm_v = np.abs(np.random.randn(1, 1).astype(dtype))
  V_expected = np.copy(V)
  V_expected[:, j + 1] = orth[:, 0]
  H_expected = np.copy(H_s)
  H_expected[:, j] = overlaps[:, 0]
  H_expected[j + 1, j] = norm_v[0, 0]

  V_v = vops.distribute(V)
  orth_v = vops.distribute(orth)

  @functools.partial(
      pops.pmap, in_axes=(0, None, None, 0, None, None), out_axes=(0, None))
  def test_f(V_v, H_s, j, orth_v, overlaps, norm_v):
    return solve._update_arnoldi(V_v, H_s, j, orth_v, overlaps, norm_v)

  V_result, H_result = test_f(V_v, H_s, j, orth_v, overlaps, norm_v)
  V_result = vops.undistribute(V_result)
  tol = testutils.eps(precision, dtype=dtype)
  testutils.assert_allclose(V_expected, V_result, atol=tol * np.linalg.norm(V))
  testutils.assert_allclose(
    H_expected, H_result, atol=tol * np.linalg.norm(H_expected))


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("check_residual", [True, ])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("precision", precisions)
def test_arnoldi_qr(dim, seed, check_residual, dtype, precision):
  """ Initializes GMRES and runs the iterated Arnoldi QR to
  completion. At each iteration, tests that:
    1. The Arnoldi matrices satisfy the relations given in
       test_arnoldi_step.
    2. The computed R factor is indeed that of the Arnoldi
       upper Hessenberg matrix.
    3. The error vector beta has been appropriately rotated.
  If check_residual is True, also performs the GMRES linear solve,
  and makes sure the updated error agrees with the actual residual norm.
  """
  dtype = dtype
  np.random.seed(seed)
  A = np.random.randn(dim, dim).astype(dtype)
  cond = np.linalg.cond(A)
  expected = np.random.randn(dim, 1).astype(dtype)
  B = np.dot(A, expected)
  b_norm = np.linalg.norm(B)
  tol = testutils.eps(precision, dtype=dtype)
  tol *= cond * np.linalg.norm(B)
  A_d = pops.distribute(A)
  B_d = vops.distribute(B)

  @functools.partial(
      pops.pmap, out_axes=(0, 0, None, None, None, None, None, None, None))
  def init_f(A, B):
    X_v, args, _, _, _ = solve._gmres_init(A, B, dim, None, precision, None)
    V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j = args
    return X_v, V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j

  X_v, V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j = init_f(A_d, B_d)
  beta_0 = jnp.array(beta_s)
  itertol = testutils.eps(precision, A.dtype) * b_norm

  @functools.partial(
      pops.pmap,
      in_axes=(0, 0, None, None, None, None, None, None, None),
      out_axes=(0, None, None, None, None, None, None, None))
  def test_f(A, V, H, R, beta, cos, sin, err, j):
    args = (V, H, R, beta, cos, sin, err, jnp.full(1, j, dtype=jnp.int32))
    V, H, R, beta, cos, sin, err, j = solve._arnoldi_qr(
      A, args, precision=precision)
    return V, H, R, beta, cos, sin, err, j

  @functools.partial(
      pops.pmap,
      in_axes=(0, None, None, None, None, None, None, None),
      out_axes=None)
  def test_cond_f(V, H, R, beta, cos, sin, err, j):
    args = (V, H, R, beta, cos, sin, err, j)
    return solve._arnoldi_cond(dim, itertol, args)

  if check_residual:

    @functools.partial(
        pops.pmap,
        in_axes=(0, 0, None, None, None),
        static_broadcasted_argnums=(4,))
    def _update_f(X, V, R, beta, arnoldi_maxiter):
      return solve._gmres_update_solution(X, V, R, beta, None, arnoldi_maxiter)

  while test_cond_f(V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j):
    out = test_f(A_d, V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j)
    V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j = out
    V_test = vops.undistribute(V_v)

    _arnoldi_assert_one(A, V_test, H_s, j[0], tol)
    _arnoldi_assert_two(A, V_test, H_s, j[0], tol)

    j = int(j)
    _, R_expected = np.linalg.qr(H_s[:j + 1, :j], mode="complete")
    R_expected = _dephase_r(R_expected)
    R_result = _dephase_r(R_s)
    testutils.assert_allclose(R_expected, R_result[:j + 1, :j], atol=tol)

    beta_expected = jnp.array(beta_0)
    for i in range(j):
      beta_expected, _, _ = solve._apply_ith_rotation(
        i, (beta_expected, cos_s, sin_s))
    testutils.assert_allclose(beta_expected, beta_s, atol=tol)

    if check_residual:
      this_X_v = _update_f(X_v, V_v, R_s, beta_s, j)
      this_X = vops.undistribute(this_X_v)
      residual = np.linalg.norm(B - np.dot(A, this_X))
      testutils.assert_allclose(np.array([residual]), err, atol=tol)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("precision", precisions)
def test_gmres_update_solution(dim, seed, dtype, precision):
  np.random.seed(seed)
  X = np.zeros((dim, 1), dtype=dtype)
  X_v = vops.distribute(X)
  V = np.random.rand(dim, dim + 1).astype(dtype)
  V_o, _ = np.linalg.qr(V[:, :-1])
  V[:, :-1] = V_o
  V_v = vops.distribute(V)

  Y = np.random.randn(dim,).astype(dtype)
  R = np.triu(np.random.randn(dim + 1, dim).astype(dtype))
  beta = np.zeros(dim + 1, dtype=dtype)
  beta[:-1] = np.dot(R[:-1, :], Y)
  R_s = jnp.array(R)
  beta_s = jnp.array(beta)

  @functools.partial(
      pops.pmap,
      in_axes=(0, 0, None, None),
  )
  def update_f(X, V, R, beta):
    return solve._gmres_update_solution(X, V, R, beta, None, dim)

  expected = _gmres_update_np(X, V, R, beta)
  result = update_f(X_v, V_v, R_s, beta_s)
  result = vops.undistribute(result)

  cond = np.linalg.cond(R[:-1, :])
  tol = testutils.eps(precision, dtype=dtype) * np.linalg.norm(expected) * cond
  testutils.assert_allclose(expected, result, atol=tol)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("precision", precisions)
def test_arnoldi_step(dim, dtype, precision):
  """ Iterates through solve._arnoldi_step and ensures the relations
    `A @ V_v[:, :j] = V_v[:, :j+1] @ H_s[:j+1, :j]`
  along with
    `H_s[:j+1, :j] = V_v[:, :j]^H @ A @ V_v[:, :j]`
  are satisfied.
  """
  np.random.seed(1)
  A = np.random.randn(dim, dim).astype(dtype)
  cond = np.linalg.cond(A)
  A_d = pops.distribute(A)
  x0 = np.random.randn(dim).astype(dtype)
  x0_norm = np.linalg.norm(x0)
  x0 /= x0_norm

  V_v = np.zeros((dim, dim + 1), dtype=A.dtype)
  V_v[:, 0] = x0
  V_v = vops.distribute(V_v)
  H_s = jnp.eye(dim + 1, dim, dtype=A.dtype)

  tol = testutils.eps(precision, dtype=dtype) * x0_norm * cond

  @functools.partial(
      pops.pmap, in_axes=(None, 0, None, 0, None), out_axes=(0, None))
  def _arnoldi_f(j, V_v, H_s, A_d, A_inv):
    return solve._arnoldi_step(j, V_v, H_s, A_d, A_inv, precision)

  for j in range(dim):
    V_v, H_s = _arnoldi_f(j, V_v, H_s, A_d, None)
    V = vops.undistribute(V_v)
    _arnoldi_assert_one(A, V, H_s, j + 1, tol)
    _arnoldi_assert_two(A, V, H_s, j + 1, tol)


# Linear solve
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_gmres_bare(dim, dtype):
  """ Tests that solve.gmres produces a correct solution when run with
  arnold_maxiter = dim.
  """
  np.random.seed(1)
  A = np.random.randn(dim, dim).astype(dtype)
  cond = np.linalg.cond(A)
  expected = np.random.randn(dim, 1).astype(dtype)
  B = np.dot(A, expected)
  tol = testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  tol *= cond * np.linalg.norm(B)
  A_d = pops.distribute(A)
  B_d = vops.distribute(B)

  @functools.partial(pops.pmap, out_axes=(0, None, None))
  def solve_f(A, B):
    return solve.gmres(A, B, arnoldi_maxiter=dim)

  result, _, _ = solve_f(A_d, B_d)
  result = vops.undistribute(result)
  testutils.assert_allclose(expected, result, atol=tol)


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("dtype", dtypes)
def test_gmres_solve(dim, dtype):
  """ Tests that solve.solve produces a correct solution.
  """
  A = np.random.randn(dim, dim).astype(dtype)
  cond = np.linalg.cond(A)
  expected = np.random.randn(dim, 1).astype(dtype)
  B = np.dot(A, expected)
  tol = testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  tol *= cond * np.linalg.norm(B)
  A_d = pops.distribute(A)
  B_d = vops.distribute(B)

  @functools.partial(pops.pmap, out_axes=(0, None, None, 0))
  def solve_f(A, B):
    return solve.solve(A, B)
  result, _, _, _ = solve_f(A_d, B_d)
  result = vops.undistribute(result)
  testutils.assert_allclose(expected, result, atol=tol)