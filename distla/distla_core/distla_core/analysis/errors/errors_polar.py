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
Performs error analysis of polar. The analyses are performed for each combination
of various parameters, specified as the global (module-scope) variables below
in the case that this file is the main program. Output is returned as a
`DataFrame` and saved as a `.csv` file to `output_dir_path/output_name`.csv.
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
from distla_core.analysis import analyze_polar
from distla_core.analysis.errors import error_utils


# TODO: -modify initialize to emit the correct answer; possibly move it to
#        a separate file.
#       -write error functions.
#       -tests for error utils?


# The initialize and run functions are the same as for benchmark_poalr,
# except that for error analysis we always want to compute rho.
def _init_f(
  out_rows, out_cols, dtype, p_sz, precision, seed, serial, eps, maxiter,
  s_min_est, s_thresh, s_min, s_max, n_zero_svs, sv_distribution):
  return analyze_polar._initialize_polar(
    out_rows, out_cols, dtype, p_sz, precision, seed, serial, eps, maxiter,
    s_min_est, s_thresh, s_min, s_max, n_zero_svs, sv_distribution, True, True)


def _u_unitarity_fs():
  def _u_unitary_f(out, in_args, params, relative=False, dagger_left=False):
    u_polar = out[0]
    return error_utils.isometry_error(u_polar, relative=relative,
                                      dagger_left=dagger_left, name="U")
  bools = [False, True]
  fs = []
  for rel in bools:
    for d_left in bools:
      fs.append(functools.partial(
        _u_unitary_f, relative=rel, dagger_left=d_left
      ))
  return fs


def _comparison_fs():
  def _comparison_f(out, in_args, params, relative=False):
    u_polar = out[0]
    _, expected = in_args[0]
    return error_utils.comparison_error(
      u_polar, expected, relative, "result", "expected")
  return [functools.partial(_comparison_f, relative=r) for r in [True, False]]


def _reconstruction_fs():
  def _recon_f(out, in_args, params, relative=False):
    matrix, _ = in_args[0]
    factors = (out[0], out[1])
    return error_utils.reconstruction_error(
      matrix, factors, relative=relative, name="M", recon_name="UH")
  return [functools.partial(_recon_f, relative=r) for r in [True, False]]


def errors_polar(out_rows, out_cols, dtypes, p_szs, precisions, seeds,
                 serial, eps, maxiter, s_min_est, s_thresh, s_min,
                 s_max, n_zero_svs, sv_distribution, compute_rho,
                 batch_size=5, reduction_mode="median", output_dir_path=None,
                 output_name="errors_polar"):
  """
  Runs error analysis of polar. `batch_size` analyses per combination of
  arguments are run. Output is saved to `output_dir_path/output_name`.csv, and
  also returned as a `DataFrame`. `output_dir_path` is the current working
  directory if unspecified.

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
    batch_size: Number of runs to do per benchmark.
    reduction_mode: how to assemblage batches.
    output_dir_path: directory of output; CWD if None
    output_name: output saved to OUTPUT_NAME.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  params = list(itertools.product(
    out_rows, out_cols, dtypes, p_szs, precisions, seeds, serial, eps, maxiter,
    s_min_est, s_thresh, s_min, s_max, n_zero_svs, sv_distribution))

  param_headers = ["N_rows", "N_cols", "dtype", "p_sz", "precision", "seed",
                   "serial", "eps", "maxiter", "s_min_est", "s_thresh",
                   "s_min", "s_max", "n_zero_svs", "sv_distribution"]

  error_functions = [analyze_polar._rogue_iterations, ]
  error_functions += [analyze_polar._total_iterations, ]
  error_functions += _u_unitarity_fs()
  error_functions += _comparison_fs()
  error_functions += _reconstruction_fs()

  return analysis.measure_error(_init_f, analyze_polar._run_polar, params,
                                param_headers, reduction_mode=reduction_mode,
                                error_functions=error_functions,
                                batch_size=batch_size,
                                output_dir_path=output_dir_path,
                                output_name=output_name)


def main():
  OUT_ROWS = (1024,)  # rows of the input matrix
  OUT_COLS = (None,)  # columns of the input matrix (=out_rows if None)
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False, True) # Whether to run in serial or distributed mode.
  EPS = (None,) # Convergence threshold.
  MAXITER = (200,) # When to terminate if convergence stagnates.
  S_MIN_EST = (None,) # Estimated lowest singular value;
                      # None means machine epsilon; -1 means the true value.
  S_THRESH = (0.1,) # When to switch to Newton-Schulz from `rogue` iterations.
  S_MIN = (0.1,) # Smallest nonzero singular value of the input matrix.
  S_MAX = (1.0,) # Largest singular value of the input matrix.
  N_ZERO_SVS = (0,) # The number of zero singular values in the input matrix.
  SV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # singular values in the input matrix.
  COMPUTE_RHO = (False,) # Whether to compute the positive-semidefinite factor
                         # in addition to the unitary one.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_polar" # output saved to OUTPUT_NAME.csv
  _ = errors_polar(
    OUT_ROWS, out_cols=OUT_COLS, dtypes=DTYPES, p_szs=P_SZS,
    precisions=PRECISIONS, seeds=SEEDS, serial=SERIAL, eps=EPS, maxiter=MAXITER,
    s_min_est=S_MIN_EST, s_thresh=S_THRESH, s_min=S_MIN, s_max=S_MAX,
    n_zero_svs=N_ZERO_SVS, sv_distribution=SV_DISTRIBUTION,
    compute_rho=COMPUTE_RHO, batch_size=BATCH_SIZE,
    reduction_mode=REDUCTION_MODE, output_dir_path=OUTPUT_DIR_PATH,
    output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
