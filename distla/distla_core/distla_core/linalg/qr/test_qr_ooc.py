"""Tests for qr.py."""
from jax import lax
import jax.numpy as jnp

import numpy as np
import pytest
import tempfile

from distla_core.linalg.utils import testutils
from distla_core.linalg.qr import qr_ooc
from distla_core.utils import pops

DTYPE = jnp.float32

seeds = [0, 1]
flags = [True, False]


def _dephase_qr(R, Q=None):
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
  if Q is not None:
    Q = Q * phases
  return Q, R


@pytest.mark.parametrize("N", [8, 32, 128])
@pytest.mark.parametrize("aspect_ratio", [1, 2, 10])
@pytest.mark.parametrize("panel_size", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_qr_ooc(N, aspect_ratio, panel_size, seed):
  dtype = np.float32
  M = N * aspect_ratio
  np.random.seed(seed)
  A = np.random.randn(M, N).astype(dtype)
  _, expected = np.linalg.qr(A)
  _, expected = _dephase_qr(expected)
  with tempfile.NamedTemporaryFile(delete=False) as f:
    np.save(f, A)
    f.close()  # Explicit close needed to open again as a memmap.
               # The file is still deleted when the context goes out of scope.
    result = qr_ooc.qr_ooc(f.name, caqr_panel_size=panel_size)
  result = pops.undistribute(result)
  _, result = _dephase_qr(result)

  atol = testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  atol *= np.linalg.norm(A) ** 2

  testutils.assert_allclose(result, expected, atol=atol)


@pytest.mark.parametrize("N", [8, 32, 128])
@pytest.mark.parametrize("aspect_ratio", [1, 2, 10])
@pytest.mark.parametrize("panel_size", [1, 2])
@pytest.mark.parametrize("seed", [0, 1])
def test_fake_cholesky(N, aspect_ratio, panel_size, seed):
  fname = "fake_cholesky_test_matrix"
  dtype = np.float32
  M = N * aspect_ratio
  np.random.seed(seed)
  A = np.random.randn(M, N).astype(dtype)
  cond = np.linalg.cond(A)
  expected_gram = np.dot(A.T, A)
  expected_chol = np.linalg.cholesky(expected_gram).T
  _, expected_chol = _dephase_qr(expected_chol)

  np.save(fname, A)
  fread = fname + ".npy"

  chol_fname = "cholesky_transpose"
  gram_fname = "gram_matrix"
  qr_ooc.fake_cholesky(fread, caqr_panel_size=panel_size,
                       chol_fname=chol_fname, gram_fname=gram_fname)
  result_gram = np.load(gram_fname + ".npy")
  result_chol = np.load(chol_fname + ".npy")
  _, result_chol = _dephase_qr(result_chol)

  atol = testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  atol *= cond * np.linalg.norm(expected_gram) ** 2
  testutils.assert_allclose(result_chol, expected_chol, atol=10 * atol)
  testutils.assert_allclose(result_gram, expected_gram, atol=atol)
