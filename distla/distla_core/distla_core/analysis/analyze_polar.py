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
from distla_core.analysis import initializers
from distla_core.analysis import pmaps
from distla_core.linalg.polar.serial import polar as polar_serial


def _initialize_polar(out_rows, out_cols, dtype, p_sz, precision, seed,
                      serial, eps, maxiter, s_min_est, s_thresh, s_min,
                      s_max, n_zero_svs, sv_distribution, compute_rho,
                      return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `polar`.

  Args:
    out_rows: rows of the input matrix
    out_cols: columns of the input matrix (=out_rows if None)
    dtype: dtype of the matrices
    p_sz: panel size of the SUMMA multiplications
    precisions: ASIC matmul precision
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    eps: Convergence threshold.
    maxiter: When to terminate if convergence stagnates.
    s_min_est: Estimated lowest singular value; None means machine epsilon;
      -1 means the true value.
    s_thresh: When to switch to Newton-Schulz from `rogue` iterations.
    s_min: Smallest nonzero singular value of the input matrix.
    s_max: Largest singular value of the input matrix.
    n_zero_svs: The number of zero singular values in the input matrix.
    sv_distribution: `linear` or `geometric` distribution of singular values in
      the input matrix.
    compute_rho:  Whether to compute the positive-semidefinite factor in
      addition to the unitary one.
    return_answer: If True, the correct result computed from the SVD is
      returned along with the input matrix.
  Returns:
    matrix: Input matrix for the polar decomposition.
    p_sz, precision, dist
    p_sz, transpose_a, transpose_b, precision: Same as input.
  """
  if out_cols is None:
    out_cols = out_rows
  min_dim = min(out_rows, out_cols)

  svs = initializers.sv_spectrum(
    min_dim - n_zero_svs, sv_min=s_min, sv_max=s_max,
    distribution=sv_distribution, n_zeros=n_zero_svs, dtype=dtype)
  initialized = initializers.random_from_singular_spectrum(
    (out_rows, out_cols), svs, dtype=dtype, seed=seed, serial=serial, p_sz=p_sz,
    precision=precision, return_factors=return_answer)

  if return_answer:
    matrix, U, V = initialized
    result = pmaps.matmul(U, V, transpose_b=True, conj_b=True)
    initialized = (matrix, result)

  out = (initialized, p_sz, precision, serial, eps, maxiter, s_min_est, s_thresh,
         compute_rho)
  return out


def _rogue_iterations(out, in_args, params):
  return int(out[-3]), "Rogue Iterations"


def _total_iterations(out, in_args, params):
  return int(out[-2]), "Total Iterations"


def _run_polar(matrix, p_sz, precision, serial, eps, maxiter, s_min_est,
               s_thresh, compute_rho):
  """ Helper to call polar.
  """
  if len(matrix) == 2: # handles return_answer
    matrix, _ = matrix

  if serial:
    if compute_rho:
      out = polar_serial.polar(matrix, eps=eps, maxiter=maxiter,
                               s_min=s_min_est, s_thresh=s_thresh,
                               precision=precision)
    else:
      out = polar_serial.polarU(matrix, eps=eps, maxiter=maxiter,
                                s_min=s_min_est, s_thresh=s_thresh,
                                precision=precision)
  else:
    if compute_rho:
      out = pmaps.polar(matrix, eps=eps, maxiter=maxiter, s_min=s_min_est,
                        s_thresh=s_thresh, p_sz=p_sz, precision=precision)
    else:
      out = pmaps.polarU(matrix, eps=eps, maxiter=maxiter, s_min=s_min_est,
                         s_thresh=s_thresh, p_sz=p_sz, precision=precision)
  out = (out[0].block_until_ready(), *(out[1:]))
  return out
