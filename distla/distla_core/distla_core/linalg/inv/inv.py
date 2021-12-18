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
""" Newton-Schulz matrix inversion.
"""

from jax import lax
import jax.numpy as jnp
import warnings

from distla_core.blas.summa import summa
from distla_core.linalg.utils import testutils
from distla_core.utils import pops


def inv(mat, left=True, precision=lax.Precision.HIGHEST, maxiter=200,
        p_sz=1024, tol=None, unpadded_dim=None):
  """ Computes an approximate inverse of the matrix `mat` using Newton-Schulz
  iterations. Two matmuls are performed per iteration.

  The polynomial applied is:
    X' = 2 * X - X @ A_t @ X

  where the fixed matrix A_t is either mat or mat^H for left=True or False
  respectively. X is initialized as either c * mat^T or c * mat for
  left = True or False respectively, c = 1 / ||mat^H mat||_F.

  If mat is rectangular, this function instead yields an approximation of the
  appropriate Moore-Penrose pseudo-inverse. For thin (fat) matrices, left
  should be set to False (True) respectively. If the wrong 'left' is chosen,
  it will be toggled with a warning.

  Args:
    mat: Matrix to invert.
    left: If `True` the result satisfies `result @ mat = I` more accurately
      than `mat @ result = I`, and vice versa if `False`.
    precision: ASIC matmul precision.
    maxiter: Iteration count to terminate at even if convergence is not
      reached.
    p_sz: SUMMA panel size.
    tol: Convergence threshold. Machine epsilon if None.
    unpadded_dim: If specified, the matrix is assumed to be padded with zeroes
      outside its top-left `unpadded_dim x unpadded_dim` block.

  Returns:
    result: The approximate inverse.
    err: Relative Frobenius difference from the previous iterate at the final
      iteration.
    i: Number of iterations.
  """
  n_rows_l, n_cols_l = mat.shape
  n_rows = n_rows_l * pops.NROWS
  n_cols = n_cols_l * pops.NCOLS
  if n_rows < n_cols and left:
    warnings.warn(
      "Newton-Schulz inversion fails for fat matrices (yours was "
      f"({n_rows}, {n_cols}) with left=True. left=False will be used instead."
    )
    left = False
  if n_rows > n_cols and not(left):
    warnings.warn(
      "Newton-Schulz inversion fails for thin matrices (yours was "
      f"({n_rows}, {n_cols}) with left=False. left=True will be used instead."
    )
    left = True

  mat2 = summa.summa(mat.conj(), mat, p_sz, True, False, precision=precision)
  coef = 1 / pops.frobnorm(mat2)
  iter_0 = coef * pops.transpose(mat.conj())

  if tol is None:
    tol = testutils.eps(precision, mat.dtype)
  err = 2 * tol
  args = (iter_0, err, coef * tol, 0)

  # Though equivalent in infinite precision, the subsequent two iterations
  # produce differently accurate left/right inverses in floating point.
  if left:
    def ns_polynomial(X):
      """ (2I - XA) X
      """
      result = summa.summa(X, mat, p_sz, False, False, precision=precision)
      result = -pops.add_to_diagonal(result, -2.0, unpadded_dim=unpadded_dim)
      return summa.summa(result, X, p_sz, False, False, precision=precision)
  else:
    def ns_polynomial(X):
      """ X (2I - AX)
      """
      result = summa.summa(mat, X, p_sz, False, False, precision=precision)
      result = -pops.add_to_diagonal(result, -2.0, unpadded_dim=unpadded_dim)
      return summa.summa(X, result, p_sz, False, False, precision=precision)

  def not_done(args):
    _, err, tol, i = args
    return jnp.logical_and(i < maxiter, err > tol)

  def ns_iter(args):
    iter_i, _, _, i = args
    iter_f = ns_polynomial(iter_i)
    err = pops.frobnorm(iter_f - iter_i)
    return iter_f, err.real, 2 * tol, i + 1

  result, err, _, i = lax.while_loop(not_done, ns_iter, args)
  return result, err, i
