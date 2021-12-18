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
"""
Performs errors of invsqrt. The errors are performed for each
combination of various parameters, specified within `main()` in the case that
this file is the main program. Output is returned as a `DataFrame` and saved as
a `.csv` file to `output_dir_path/output_name`.csv.
In the default case of `output_dir_path=None`, it is taken to be the current
working directory.
"""

import functools
import itertools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.analysis import analysis
from distla_core.analysis import analyze_invsqrt
from distla_core.analysis import pmaps
from distla_core.analysis.errors import error_utils


def _invsqrt_inversion(out, in_args, params, relative=True):
  """ How well S^-1/2 inverts S^1/2.
  """
  sqrt_matrix, invsqrt_matrix, _, _ = out
  eye = pmaps.eye(sqrt_matrix.shape, dtype=sqrt_matrix.dtype)
  should_be_eye = pmaps.matmul(sqrt_matrix, invsqrt_matrix)
  return error_utils.comparison_error(eye, should_be_eye, relative,
                                      "I", "S^1/2 S^-1/2")


def _invsqrt_comparison(out, in_args, params, relative=True, plus=True):
  """ Compares either S^1/2 or S^-1/2 with the SVD-computed answer.
  """
  _, sqrt_expected, invsqrt_expected = in_args[0]
  sqrt_matrix, invsqrt_matrix, _, _ = out
  if plus:
    expected = sqrt_expected
    result = sqrt_matrix
    expected_name = "expected^1/2"
    result_name = "result^1/2"
  else:
    expected = invsqrt_expected
    result = invsqrt_matrix
    expected_name = "expected^-1/2"
    result_name = "result^-1/2"
  return error_utils.comparison_error(expected, result, relative, expected_name,
                                      result_name)


def _invsqrt_reconstruction(out, in_args, params, relative=True):
  """ How well S^1/2 S^1/2 recovers the input.
  """
  sqrt_matrix = out[0]
  sqrt_squared = pmaps.matmul(sqrt_matrix, sqrt_matrix)
  matrix = in_args[0][0]
  return error_utils.comparison_error(matrix, sqrt_squared, relative,
                                      "matrix", "S^1/2 S^1/2")


def error_invsqrt(out_rows, dtypes, p_szs, precisions, seeds, serial, eps,
                  maxiter, s_min_est, s_thresh, s_min, s_max,
                  ev_distribution, batch_size=5, reduction_mode="median",
                  output_dir_path=None, output_name="error_invsqrt"):
  """
  Runs errors of invsqrt. One error per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    out_rows: rows of the input matrix
    dtypes: dtype of the matrices
    p_szs: panel size of the SUMMA multiplications
    precisions: ASIC matmul precision
    seeds: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    eps: Convergence threshold.
    maxiter: When to terminate if convergence stagnates.
    s_min_est: Estimated lowest eigenvalue; None means machine epsilon;
      -1 means the true value.
    s_thresh: When to switch to Newton-Schulz from `rogue` iterations.
    s_min: Smallest nonzero eigenvalue of the input matrix.
    s_max: Largest eigenvalue of the input matrix.
    ev_distribution: `linear` or `geometric` distribution of eigenvalues in
      the input matrix.
    batch_size: Number of runs to do per error.
    reduction_mode: how to assemblage batches.
    output_dir_path: directory of output; CWD if None
    output_name: output saved to OUTPUT_NAME.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  params = list(itertools.product(
    out_rows, dtypes, p_szs, precisions, seeds, serial, eps, maxiter,
    s_min_est, s_thresh, s_min, s_max, ev_distribution))

  param_headers = ["N_rows", "dtype", "p_sz", "precision", "seed",
                   "serial", "eps", "maxiter", "s_min_est", "s_thresh",
                   "s_min", "s_max", "ev_distribution"]
  init_f = functools.partial(
    analyze_invsqrt._initialize_invsqrt, return_answer=True
  )
  target_f = analyze_invsqrt._run_invsqrt

  error_functions = [analyze_invsqrt._rogue_iterations,
                     analyze_invsqrt._total_iterations]
  for rel in [True, False]:
    for plus in [True, False]:
      error_functions.append(functools.partial(_invsqrt_reconstruction,
                                               relative=rel))
      error_functions.append(functools.partial(_invsqrt_inversion,
                                               relative=rel))
      error_functions.append(functools.partial(_invsqrt_comparison,
                                               relative=rel, plus=plus))
  return analysis.measure_error(init_f, target_f, params, param_headers,
                                reduction_mode=reduction_mode,
                                batch_size=batch_size,
                                error_functions=error_functions,
                                output_dir_path=output_dir_path,
                                output_name=output_name)


def main():
  OUT_ROWS = (1024,)  # rows of the input matrix
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False, True) # Whether to run in serial or distributed mode.
  EPS = (None,) # Convergence threshold.
  MAXITER = (200,) # When to terminate if convergence stagnates.
  S_MIN_EST = (None,) # Estimated lowest eigenvalue;
                      # None means machine epsilon; -1 means the true value.
  S_THRESH = (0.1,) # When to switch to Newton-Schulz from `rogue` iterations.
  S_MIN = (0.1,) # Smallest nonzero eigenvalue of the input matrix.
  S_MAX = (1.0,) # Largest eigenvalue of the input matrix.
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "error_invsqrt" # output saved to OUTPUT_NAME.csv
  _ = error_invsqrt(
    OUT_ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, eps=EPS, maxiter=MAXITER, s_min_est=S_MIN_EST,
    s_thresh=S_THRESH, s_min=S_MIN, s_max=S_MAX,
    ev_distribution=EV_DISTRIBUTION, batch_size=BATCH_SIZE,
    reduction_mode=REDUCTION_MODE, output_dir_path=OUTPUT_DIR_PATH,
    output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
