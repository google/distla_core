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
Performs benchmarks of purify. The benchmarks are performed for each combination
of various parameters defined in `main()` in the case that this file is the
main program. Output is returned as a `DataFrame` and saved as a `.csv` file to
`output_dir_path/output_name`.csv.
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
from distla_core.analysis import analyze_purify
from distla_core.analysis.benchmarks import benchmark_utils


def _mmult_tflops(out, in_args, params):
  rows = params[0]
  dt = out[0]
  return benchmark_utils.mmult_tflops(rows, rows, dt), "Matmul TFLOPS/s"


def benchmark_purify(
  out_rows, dtypes, p_szs, precisions, seeds, serial, tol, maxiter,
  include_overlaps, method, eig_min, eig_max, k_factor, gap_start, gap_factor,
  ev_distribution,  batch_size=5, reduction_mode="median", output_dir_path=None,
  output_name="benchmark_purify"):
  """ Runs benchmarks of purify. One benchmark per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    out_rows : rows of the input matrix
    dtypes : dtype of the matrices
    p_szs : panel size of the summa multiplications
    precisions: asic matmul precision
    seeds : random seed to initialize input; system clock if none.
    serial: whether to run in serial or distributed mode.
    tol: convergence threshold.
    maxiter: when to terminate if convergence stagnates.
    include_overlaps: whether to also perform basis transform.
    method: string specifying purification method.
    eig_min: most negative eigenvalue in the spectrum.
    eig_max: most positive eigenvalue in the spectrum.
    k_factor: we seek the `out_rows // k_factor`'th subspace.
    gap_start: eigenvalue[out_rows // k_factor]
    gap_factor: the gap is of size `(eig_max - eig_min) / gap_factor`
    ev_distribution: `linear` or `geometric` distribution of
      eigenvalues in the input matrix.
    batch_size: how many runs to assemblage
    reduction_mode: how to assemblage results
    output_dir_path: directory of output; cwd if none
    output_name: output saved to output_name.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  params = list(itertools.product(
    out_rows, dtypes, p_szs, precisions, seeds, serial, tol, maxiter,
    include_overlaps, method, eig_min, eig_max, k_factor, gap_start, gap_factor,
    ev_distribution))

  param_headers = [
    "rows", "dtype", "p_sz", "precision", "seed", "serial", "tol", "maxiter",
    "include overlaps", "method", "eig min", "eig max", "occupancy",
    "gap start", "gap factor", "ev distribution"]
  init_f = functools.partial(
    analyze_purify._initialize_purify, return_answer=False
  )
  target_f = analyze_purify._run_purify

  speed_functions = [analyze_purify._total_iterations, _mmult_tflops]
  return analysis.benchmark(init_f, target_f, params, param_headers,
                            reduction_mode=reduction_mode,
                            batch_size=batch_size,
                            speed_functions=speed_functions,
                            output_dir_path=output_dir_path,
                            output_name=output_name)


def main():
  OUT_ROWS = (124,)  # rows of the input matrix
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False, True) # Whether to run in serial or distributed mode.
  TOL = (None,) # Convergence threshold.
  MAXITER = (200,) # When to terminate if convergence stagnates.
  INCLUDE_OVERLAPS = (False, True) # Whether to also perform basis transform.
  METHOD = ("hole-particle", "fixed potential") # purification method.
  EIG_MIN = (-10.,) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (10.,) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (10, 2) # We seek the OUT_ROWS // K_FACTOR'th subspace.
  GAP_START = (0.,) # eigenvalue[OUT_ROWS // K_FACTOR]
  GAP_FACTOR = (10,) # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 5 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "benchmark_purify" # output saved to OUTPUT_NAME.csv
  _ = benchmark_purify(
    OUT_ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, tol=TOL, maxiter=MAXITER, include_overlaps=INCLUDE_OVERLAPS,
    method=METHOD, eig_min=EIG_MIN, eig_max=EIG_MAX, k_factor=K_FACTOR,
    gap_start=GAP_START, gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
