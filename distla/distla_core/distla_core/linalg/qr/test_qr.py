"""Tests for qr.py."""
from jax import lax
import jax.numpy as jnp

import functools
import numpy as np
import pytest

from distla_core.linalg.utils import testutils
from distla_core.linalg.qr import qr
from distla_core.utils import pops
from distla_core.utils import vops

DTYPE = jnp.float32

seeds = [0, 1]
flags = [True, False]


@pops.pmap
def _rows_cols_f(M):
  rows, cols = pops.indices(M.shape)
  return rows, cols


def _dephase_qr(Q, R):
  """ Maps the Q and R factor from an arbitrary QR decomposition to the unique
  with non-negative diagonal entries.
  """
  phases_data = np.sign(np.diagonal(R))
  m, n = R.shape
  if m > n:
    phases = np.ones(m)
    phases[:n] = phases_data
  else:
    phases = phases_data
  R = phases.conj()[:, None] * R
  Q = Q * phases
  return Q, R


def _form_q_from_wy(Q_wy, offset=0):
  M = Q_wy[0].shape[0]
  result = np.eye(M, M, dtype=Q_wy[0].dtype)
  result[:offset, :] = 0.
  result[:, :offset] = 0.
  W = np.array(Q_wy[0])
  Y = np.array(Q_wy[1])
  result = result - np.dot(W, Y.conj().T)
  return result


@pytest.mark.parametrize("matrix_shape", [[16, 4], [32, 8]])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("pad", [0, 1, 2])
@pytest.mark.parametrize("column_replicate", flags)
def test_yamamoto(matrix_shape, seed, pad, column_replicate):
  """ Computes the Yamamoto WY representation of a random panel.
  Forms the corresponding Q factor. Confirms that Q^H @ panel = R.
  """
  np.random.seed(seed)
  panel = np.random.randn(*matrix_shape).astype(DTYPE)

  panel[:pad, :] = 0.
  Q_reduced, _ = np.linalg.qr(panel, mode="reduced")
  Q_d = vops.distribute(Q_reduced, column_replicated=column_replicate)

  Q_full, R_full = np.linalg.qr(panel[pad:, :], mode="complete")
  Q_full, R_full = _dephase_qr(Q_full, R_full)

  @functools.partial(pops.pmap, in_axes=(0, None))
  def _yamamoto_f(Q, pad):
    return qr._yamamoto(Q, pad)

  W, Y = _yamamoto_f(Q_d, pad)
  _check_replication(W)
  _check_replication(Y)
  W = vops.undistribute(W)
  Y = vops.undistribute(Y)

  n = matrix_shape[1]
  np.testing.assert_array_almost_equal(W[:pad, :], np.zeros_like(W[:pad, :]))
  np.testing.assert_array_almost_equal(
    W[pad:pad + n, :], Q_reduced[pad:pad + n, :] - np.eye(n))
  np.testing.assert_array_almost_equal(W[pad + n:, :], Q_reduced[pad + n:, :])
  np.testing.assert_array_almost_equal(Y[:pad, :], np.zeros_like(Y[:pad, :]))
  np.testing.assert_array_almost_equal(Y[pad:pad + n, :], -np.eye(n))

  Y2 = Y[pad + n:, :]
  W1 = W[pad:pad + n, :]
  W2 = W[pad + n:, :]
  W1Y2 = np.dot(-W1.conj().T, Y2.conj().T).T
  eps = testutils.eps(lax.Precision.HIGHEST, dtype=panel.dtype)
  np.testing.assert_allclose(W1Y2, W2, atol=eps * np.linalg.norm(W2))
  Q_formed = _form_q_from_wy((W, Y), offset=pad)
  Q_full, R_full = np.linalg.qr(panel[pad:, :], mode="complete")
  R_formed = np.dot(np.array(Q_formed).conj().T, panel)
  testutils.assert_allclose(
    R_formed[:pad, :], np.zeros_like(R_formed[:pad, :]),
    eps * np.linalg.norm(R_formed))
  testutils.assert_allclose(
    np.abs(R_full), np.abs(R_formed[pad:, :]), eps * np.linalg.norm(R_formed))


@pytest.mark.parametrize("M", [8, 32])
@pytest.mark.parametrize("N", [4, 12])
@pytest.mark.parametrize("panel_size", [2, 4])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("column_replicate_w", flags)
@pytest.mark.parametrize("column_replicate_y", flags)
@pytest.mark.parametrize("left", flags)
def test_apply_yamamoto(
  M, N, panel_size, seed, column_replicate_w,
    column_replicate_y, left):
  """ Tests that the apply_wy routines perform the correct multiplication
  given randomly generated W and Y.
  """
  np.random.seed(seed)
  W = np.random.randn(M, panel_size).astype(DTYPE)
  Y = np.random.randn(M, panel_size).astype(DTYPE)
  Y[:panel_size, :] = -np.eye(panel_size, dtype=DTYPE)
  Q = _form_q_from_wy((W, Y))

  W_d = vops.distribute(W, column_replicated=column_replicate_w)
  Y_d = vops.distribute(Y, column_replicated=column_replicate_y)
  if left:
    @pops.pmap
    def test_f(W, Y, A):
      return qr._apply_wy_left(W, Y, A)
    A = np.random.randn(M, N).astype(DTYPE)
    A_d = pops.distribute(A)
    expected = np.copy(A)
    expected = np.dot(Q.conj().T, A)
  else:
    @pops.pmap
    def test_f(W, Y, A):
      return qr._apply_wy_right(W, Y, A)
    A = np.random.randn(N, M).astype(DTYPE)
    A_d = pops.distribute(A)
    expected = np.dot(A, Q)
  result = test_f(W_d, Y_d, A_d)
  result = pops.undistribute(result)
  eps = testutils.eps(lax.Precision.HIGHEST, dtype=DTYPE) * M * N
  testutils.assert_allclose(result, expected, atol=10 * eps)


def _check_replication(vec):
  if vec.is_column_replicated:
    same_idx = [(0, 4), (1, 5), (2, 6), (3, 7)]
  else:
    same_idx = [(0, 1, 2, 3), (4, 5, 6, 7)]
  for tup in same_idx:
    block_0 = vec.array[tup[0], :, :]
    for other in tup[1:]:
      block_i = vec.array[other, :, :]
      np.testing.assert_array_equal(block_0, block_i)


@pytest.mark.parametrize("matrix_shape", [[16, 8], [32, 16]])
@pytest.mark.parametrize("panel_size", [1, 2, 4])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("step", [0, 1])
def test_factor_panel(matrix_shape, panel_size, seed, step):
  """ Tests that qr._factor_panel performs the correct updates
  on the given panel.
  """
  # Unpack.
  dtype = DTYPE
  precision = lax.Precision.HIGHEST
  np.random.seed(seed)
  n_rows, n_cols = matrix_shape
  ci = panel_size * step
  cf = ci + panel_size

  # Factor a random matrix to the current step.
  A = np.random.randn(n_rows, n_cols).astype(dtype)
  panel = A[ci:, ci:cf]
  Q_panel, R_panel = np.linalg.qr(panel, mode="reduced")
  Q_panel, R_panel = _dephase_qr(Q_panel, R_panel)

  A_d = pops.distribute(A)

  # Run the function.
  @functools.partial(
    pops.pmap, in_axes=(0, None, None), static_broadcasted_argnums=(2,),
    out_axes=(0, 0))
  def _test_f(A, column_idx, panel_size):
    return qr._factor_panel(A, column_idx, panel_size)
  Q, R = _test_f(A_d, ci, panel_size)
  _check_replication(Q)
  Q = vops.undistribute(Q)
  R = R[0]
  Q = Q[ci:, :]
  Q, R = _dephase_qr(Q, R)
  eps = testutils.eps(precision, dtype=DTYPE) * np.linalg.norm(A)
  np.testing.assert_allclose(Q, Q_panel, atol=eps)
  np.testing.assert_allclose(R, R_panel, atol=eps)


@pytest.mark.parametrize("matrix_shape", [[32, 32], [32, 16], [16, 32]])
@pytest.mark.parametrize("panel_size", [1, 2, 4])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("step", [0, 1, 2, 3])
def test_qr_step(matrix_shape, panel_size, seed, step):
  """ Tests that a single qr step operates correctly upon a previously
  factored matrix.
  """
  # Unpack.
  dtype = DTYPE
  precision = lax.Precision.HIGHEST
  np.random.seed(seed)
  n_rows, n_cols = matrix_shape
  ci = panel_size * step
  cf = ci + panel_size

  # Factor a random matrix to the current step.
  A = np.random.randn(n_rows, n_cols).astype(dtype)
  expected_Q, true_R = np.linalg.qr(A, mode="complete")
  eps = testutils.eps(precision, dtype=DTYPE) * n_rows * n_cols * 10
  previous_panel = A[:, :ci]
  input_Q, _ = np.linalg.qr(previous_panel, mode="complete")
  input_A = np.dot(input_Q.conj().T, A)
  input_A[:, :ci] = np.triu(input_A[:, :ci])
  test_Q = pops.distribute(input_Q)
  test_A = pops.distribute(input_A)

  # Factor this step.
  panel = input_A[ci:, ci:cf]
  Q_panel, _ = np.linalg.qr(panel, mode="complete")
  update_A = np.dot(Q_panel.conj().T, input_A[ci:, ci:])
  update_A = np.triu(update_A)
  expected_A = np.copy(input_A)
  expected_A[ci:, ci:] = update_A

  # Run the function.
  @functools.partial(
    pops.pmap, in_axes=(0, 0, None, None, None),
    static_broadcasted_argnums=(3,), out_axes=(0, 0, None, None))
  def _test_f(Q, R, column_idx, panel_size, first_failure):
    return qr.qr_step(Q, R, column_idx, panel_size, first_failure)

  Q, R, column_idx, first_failure = _test_f(test_Q, test_A, ci, panel_size, -1)
  Q = pops.undistribute(Q)
  R = pops.undistribute(R)
  assert int(first_failure) == -1

  @functools.partial(pops.pmap, out_axes=None)
  def _get_rows(x):
    return pops.get_rows(x, ci, panel_size)
  R_result = np.copy(R)
  R_result[cf:, cf:] = 0.
  R_compare = np.copy(true_R)
  R_compare[cf:, cf:] = 0.

  testutils.assert_allclose(np.abs(R_result), np.abs(R_compare), atol=eps)
  testutils.assert_allclose(
    np.abs(Q[:, :cf]), np.abs(expected_Q[:, :cf]), atol=eps)


@pytest.mark.parametrize("matrix_shape", [[32, 32], [32, 16], [16, 32]])
@pytest.mark.parametrize("panel_size", [1, 2, 4])
@pytest.mark.parametrize("seed", seeds)
def test_full_qr(matrix_shape, seed, panel_size):
  """ Tests that qr produces a correct qr decomposition.
  """
  np.random.seed(seed)
  dtype = DTYPE
  m, n = matrix_shape
  A = np.random.randn(m, n).astype(dtype)
  Q_expected, R_expected = np.linalg.qr(A, mode="complete")
  Q_expected, R_expected = _dephase_qr(Q_expected, R_expected)

  A_d = pops.distribute(A)

  @functools.partial(pops.pmap, out_axes=(0, 0, None))
  def _test_f(A):
    return qr.qr(A, panel_size=panel_size)

  Q_result, R_result, first_failure = _test_f(A_d)
  assert int(first_failure) == -1
  Q_result = pops.undistribute(Q_result)
  R_result = pops.undistribute(R_result)
  Q_result, R_result = _dephase_qr(Q_result, R_result)
  eps = 10 * m * n * testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  recon = np.dot(Q_result, R_result)
  testutils.assert_allclose(Q_expected[:, :n], Q_result[:, :n], atol=eps)
  testutils.assert_allclose(A, recon, atol=eps)
  testutils.assert_allclose(R_expected, R_result, atol=eps)
