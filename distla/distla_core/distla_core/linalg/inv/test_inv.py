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
""" Tests Newton-Schulz matrix inversion.
"""
import functools

import jax
from jax import lax
import numpy as np
import pytest
import warnings

from distla_core.linalg.inv import inv
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

Ns = [8, 128]
m_ratios = [1, 0.5, 2]
lefts = [True, False]
precisions = [lax.Precision.HIGHEST, ]
dtypes = [np.complex64, np.float32]
seeds = [1, ]
unpadded_dims = [4, None]


@pytest.mark.parametrize("linear_size", Ns)
@pytest.mark.parametrize("m_ratio", m_ratios)
@pytest.mark.parametrize("left", lefts)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_inv(linear_size, m_ratio, left, precision, seed, dtype, unpadded_dim):
  """  Tests that either A^-1 @ A == I (left == True) or A @ A^-1 == I
  (left == False), where A^-1 is the result of distla_core.inv.inv(A) and
  A is a linear_size x linear_size matrix of random elements. A tolerance
  of 10 * the error of numpy inv is allowed.
  """
  np.random.seed(seed)
  n_rows = int(linear_size * m_ratio)
  mat = np.random.randn(n_rows, linear_size).astype(dtype)
  if dtype == np.complex64 or dtype == np.complex128:
    mat += 1.0j * np.random.randn(n_rows, linear_size).astype(dtype)
  if unpadded_dim is not None:
    mat[:, unpadded_dim:] = 0.
    mat[unpadded_dim:, :] = 0.

  mat_d = pops.distribute(mat)
  inv_f = pops.pmap(functools.partial(inv.inv, left=left, precision=precision))

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result, err, i = inv_f(mat_d)
  err = err[0]
  i = i[0]
  result = pops.undistribute(result)
  cond = np.linalg.cond(mat[:unpadded_dim, :unpadded_dim])
  tol = 10 * cond * testutils.eps(precision, dtype=dtype)
  if m_ratio > 1:
    left = True
  if m_ratio < 1:
    left = False

  if left:
    error_mat = np.dot(result, mat)
  else:
    error_mat = np.dot(mat, result)
  identity = np.eye(*error_mat.shape, dtype=dtype)

  if unpadded_dim is not None:
    identity[:, unpadded_dim:] = 0.
    identity[unpadded_dim:, :] = 0.
  testutils.assert_allclose(error_mat, identity, atol=tol)
