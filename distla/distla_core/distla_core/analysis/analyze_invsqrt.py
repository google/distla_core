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
from distla_core.linalg.invsqrt.serial import invsqrt as invsqrt_serial


def _initialize_invsqrt(out_rows, dtype, p_sz, precision, seed,
                        serial, eps, maxiter, s_min_est, s_thresh, s_min,
                        s_max, ev_distribution, return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `invsqrt`.

  Args:
    out_rows: rows of the input matrix
    dtype: dtype of the matrices
    p_sz: panel size of the SUMMA multiplications
    precisions: ASIC matmul precision
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    eps: Convergence threshold.
    maxiter: When to terminate if convergence stagnates.
    s_min_est: Estimated smallest eigenvalue; None means machine epsilon;
      -1 means the true value.
    s_thresh: When to switch to Newton-Schulz from `rogue` iterations.
    s_min: Smallest nonzero eigenvalue of the input matrix.
    s_max: Largest eigenvalue of the input matrix.
    ev_distribution: `linear` or `geometric` distribution of eigenvalues in
      the input matrix.
    return_answer: If True, the correct result computed from the eigenvalue
      distribution is returned along with the input matrix.
  Returns:
    matrix: Input matrix for invsqrt.
    p_sz, precision, dist
    p_sz, transpose_a, transpose_b, precision: Same as input.
  """
  e_vals = initializers.sv_spectrum(
    out_rows, sv_min=s_min, sv_max=s_max, distribution=ev_distribution,
    dtype=dtype) # sv spectrum is used since the eigenvalues must be positive
  initialized = initializers.random_from_eigenspectrum(
    e_vals, dtype=dtype, seed=seed, serial=serial, p_sz=p_sz,
    precision=precision, return_factors=return_answer)

  if return_answer:
    matrix, e_vecs = initialized  # otherwise matrix = initialized
    sqrt_matrix = initializers.combine_spectrum(e_vals ** (1 / 2), e_vecs,
                                                p_sz=p_sz, precision=precision)
    invsqrt_matrix = initializers.combine_spectrum(e_vals ** (-1 / 2), e_vecs,
                                                   p_sz=p_sz,
                                                   precision=precision)
    initialized = (matrix, sqrt_matrix, invsqrt_matrix)

  return (initialized, p_sz, precision, eps, maxiter, s_min_est, s_thresh)


def _rogue_iterations(out, in_args, params):
  return int(out[-2]), "Rogue Iterations"


def _total_iterations(out, in_args, params):
  return int(out[-1]), "Total Iterations"


def _run_invsqrt(initialized, p_sz, precision, eps, maxiter, s_min_est,
                 s_thresh):
  """ Helper to call invsqrt.
  """
  if len(initialized) == 3: # handles return_answer
    matrix, _, _ = initialized
  else:
    matrix = initialized

  if matrix.ndim == 2:
    out = invsqrt_serial.invsqrt(matrix, eps=eps, maxiter=maxiter,
                                 s_min=s_min_est, s_thresh=s_thresh,
                                 precision=precision)
  elif matrix.ndim == 3:
    out = pmaps.invsqrt(matrix, eps=eps, maxiter=maxiter, s_min=s_min_est,
                        s_thresh=s_thresh, p_sz=p_sz, precision=precision)
  else:
    raise ValueError(f"matrix had invalid ndim {matrix.ndim}.")
  return out[0].block_until_ready(), *(out[1:]) # Y, Z, jr, jt
