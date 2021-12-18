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
import numpy as np

from distla_core.analysis import initializers
from distla_core.linalg.eigh import purify
from distla_core.linalg.eigh.serial import purify as purify_serial
from distla_core.utils import pops


def _initialize_purify(
  rows, dtype, p_sz, precision, seed, serial, tol, maxiter,
  include_overlaps, method, eig_min, eig_max, k_factor, gap_start, gap_factor,
  ev_distribution, return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `purify`. TODO: refactor the arguments.

  Args:
    rows: rows of the input matrix
    dtype: dtype of the matrices
    p_sz: panel size of the SUMMA multiplications
    precision: ASIC matmul precision
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    tol: Convergence threshold.
    maxiter: When to terminate if convergence stagnates.
    include_overlaps: If True, dummy "overlap matrices" are included in the
      runtime.
    method: Purification method.
    eig_min: Smallest (most negative) eigenvalue of the input.
    eig_max: Largest (most positive) eigenvalue of the input.
    k_factor: we seek the `rows // k_factor`'th subspace.
    gap_start: Value of `spectrum[k_target - 1]`.
    gap_factor: the gap is of size `(eig_max - eig_min) / gap_factor`
    ev_distribution: `linear` or `geometric` distribution of eigenvalues in
      the input matrix.
    return_answer: If True, the correct result computed from the eigenvalue
      distribution is returned along with the input matrix.
  Returns:
    initialized: Output of the initialization, a dict.
      initialized["matrix"]: Input matrix to `purify`.
      initialized["projector"]: If not `return_answer`, `None`; otherwise the
       projector as computed from the eigenspectrum.
      initialized["overlap"]: If not `include_overlaps`, `None`; otherwise the
       computed S^-1/2.
      initialized["k_target"]: `out_rows // k_factor`.
      initialized["mu"]: Chemical potential corresponding to `k_target`.
      initialized["tol"], "maxiter", "precision", "method",
                  "p_sz": Same as input.
  """
  k_target = rows // k_factor
  gap_size = (eig_max - eig_min) / gap_factor
  e_vals = initializers.gapped_real_eigenspectrum(
    rows, eig_min, eig_max, gap_position=k_target, gap_start=gap_start,
    gap_size=gap_size, distribution=ev_distribution, dtype=dtype)
  matrix = initializers.random_from_eigenspectrum(
    e_vals, dtype=dtype, seed=seed, serial=serial, p_sz=p_sz,
    return_factors=return_answer)
  if return_answer:
    matrix, e_vecs = matrix

  mu = gap_start + gap_size / 2

  overlap = None
  if include_overlaps:
    overlap = initializers.random_fixed_norm(
      (rows, rows), dtype, seed=seed, serial=serial)

  projector = None
  if return_answer:
    e_vals_sign = np.zeros_like(e_vals)
    e_vals_sign[:k_target] = 1.
    projector = initializers.combine_spectrum(e_vals_sign, e_vecs, p_sz=p_sz,
                                              precision=precision)

  initialized = {
    "matrix": matrix, "projector": projector, "overlap_invsqrt": overlap,
    "mu": mu, "k_target": k_target, "tol": tol, "maxiter": maxiter,
    "precision": precision, "method": method, "p_sz": p_sz}
  return [initialized, ]


def _total_iterations(out, in_args, params):
  dt, result = out
  return result["n_iter"], "Total Iterations"


def _run_purify_fixed_potential(initialized):
  matrix = initialized["matrix"]
  split_point = initialized["mu"]
  precision = initialized["precision"]
  if matrix.ndim == 2:
    result = purify_serial.grand_canonically_purify(
      matrix, split_point, precision)
  elif matrix.ndim == 3:
    rows = matrix.shape[1] * pops.NROWS
    result = purify.grand_canonically_purify(
      matrix, rows, split_point, initialized["p_sz"], precision)
  projector, j_rogue, j_total, errs = result
  out = {"projector": projector.block_until_ready(),
         "n_iter": int(j_total),
         "errs": errs[:(j_total - j_rogue)]}
  print("Errs: ", out["errs"])
  return [out, ]


def _run_purify_canonical(initialized):
  matrix = initialized["matrix"]
  k_target = initialized["k_target"]
  keys = ["tol", "maxiter", "overlap_invsqrt", "precision", "method"]
  kwargs = {key: initialized[key] for key in keys}

  #TODO: improve/generalize the error catching system
  try:
    if matrix.ndim == 2:
      result = purify_serial.canonically_purify(matrix, k_target, **kwargs)
    elif matrix.ndim == 3:
      result = purify.canonically_purify(matrix, k_target,
                                         p_sz=initialized["p_sz"], **kwargs)
    else:
      raise ValueError(f"matrix had invalid ndim {matrix.ndim}.")
    out = {"projector": result[0].block_until_ready(),
           "n_iter": int(result[1]),
           "errs": result[2]}
  except RuntimeError:
    print("FAILURE")
    out = {"projector": initialized["matrix"],
           "n_iter": -1,
           "errs": -1}
  print("Errs: ", out["errs"])
  return [out, ]


def _run_purify_dac(initialized):
  matrix = initialized["matrix"]
  k_target = initialized["k_target"]
  precision = initialized["precision"]
  p_sz = initialized["p_sz"]
  if matrix.ndim == 2:
    result = purify_serial.newton_schulz_purify(
      matrix, k_target, precision=precision)
    projector, n_recursions = result
  elif matrix.ndim == 3:
    result = purify.divide_and_conquer_purify(matrix, k_target, p_sz=p_sz,
                                              precision=precision)
    projector, n_recursions, _ = result
  out = {"projector": projector.block_until_ready(),
         "n_iter": int(n_recursions),
         "errs": np.zeros(1)}
  return [out, ]


def _run_purify(initialized):
  """ Helper to call purify.
  """
  method = initialized["method"]
  if method == "fixed potential":
    return _run_purify_fixed_potential(initialized)
  elif method == "dac":
    return _run_purify_dac(initialized)
  else:
    return _run_purify_canonical(initialized)
