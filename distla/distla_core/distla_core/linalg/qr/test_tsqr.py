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
"""Tests for the tsqr code in qr.py."""
from jax import lax
import jax.numpy as jnp

import numpy as np
import pytest

from distla_core.linalg.utils import testutils
from distla_core.linalg.qr import qr
from distla_core.utils import pops
from distla_core.utils import vops

DTYPE = jnp.float32

ncols = [4, 8, 16]
rows_per_pcol_per_prow = [1, 2, 4]
seeds = [0, 1]
flags = [True, False]


def _dephase_qr(Q, R):
  """ Maps the Q and R factor from an arbitrary QR decomposition to the unique
  with non-negative diagonal entries.
  """
  phases_data = np.sign(np.diagonal(R))
  phases = np.ones((max(R.shape)))
  phases[:phases_data.size] = phases_data
  R = phases.conj()[:, None] * R
  Q = Q * phases
  return Q, R


@pytest.mark.parametrize("ncols", ncols)
@pytest.mark.parametrize("aspect_ratio", rows_per_pcol_per_prow)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("column_replicate", flags)
def test_tsqr(ncols, aspect_ratio, seed, column_replicate):
  nrows = pops.NROWS * ncols * aspect_ratio
  np.random.seed(seed)
  A = np.random.randn(nrows, ncols).astype(DTYPE)
  expected_q, expected_r = np.linalg.qr(A)
  A_v = vops.distribute(A, column_replicated=column_replicate)

  @pops.pmap
  def test_f(vecs):
    return qr.tsqr(vecs, compute_q=True)

  Q, R = test_f(A_v)
  for i in range(1, R.shape[0]):
    assert np.all(R[i, :, :] == R[0, :, :])
  R = R[0, :, :]
  Q = vops.undistribute(Q)
  testutils.test_unitarity(Q, eps_coef=10.)
  recon = np.dot(Q, R)
  eps = 30 * testutils.eps(lax.Precision.HIGHEST, dtype=A.dtype)
  testutils.assert_allclose(recon, A, atol=np.linalg.norm(A))

  Q, R = _dephase_qr(Q, R)
  expected_q, expected_r = _dephase_qr(expected_q, expected_r)
  testutils.assert_allclose(R, expected_r, atol=eps)
  testutils.assert_allclose(Q, expected_q, atol=eps)