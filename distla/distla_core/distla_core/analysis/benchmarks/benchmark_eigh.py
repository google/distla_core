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
Performs benchmarks of eigh. The benchmarks are performed for each combination
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
from distla_core.analysis import analyze_eigh
from distla_core.analysis.benchmarks import benchmark_utils


def _eigh_skip(out, in_args, params):
  result = in_args[0]["skip"]
  header = "SKIP"
  return result, header


def _eigh_tflops(out, in_args, params):
  """ `speed_function` for `benchmark` estimating the throughput of a eigh
  decomposition in TFLOP/s. The TFLOP/s are scaled by the same estimate used
  for a polar decomposition of the input, for ease of comparison:
  `((2 or 8) * 2 * rows * out_cols ** 2 + rows * out_cols) / dt`, i.e.
  the cost of a Newton-Schulz iteration.
  """
  rows, cols, dtype = params[:3]
  if cols is None:
    cols = rows
  dt = out[0]

  if dtype in set([np.complex64, np.complex128, jnp.complex64, jnp.complex128]):
    coef = 8
  else:
    coef = 2
  flops_mmult = 2 * coef * rows * (cols ** 2)
  flops = flops_mmult + rows * cols
  result = benchmark_utils.tflops_per_second(flops, dt)
  header = "TFLOPS/s"
  return result, header


def _eigh_gbps(out, in_args, params):
  """ `speed_function` for `benchmark` estimating the effective bandwidth
  of eigh in GB/s. The number of elements is estimated as
  the size of the input.
  """
  rows, cols, dtype = params[:3]
  if cols is None:
    cols = rows
  dt = out[0]
  n_elements = 2 * rows * cols
  result = benchmark_utils.gbps(n_elements, dtype, dt)
  header = "GB/s"
  return result, header


def benchmark_eigh(
  rows, cols, dtypes, p_szs, precisions, seeds, serial, canonical, svd,
  eig_min, eig_max, k_factor, gap_start, gap_factor, ev_distribution,
  batch_size=5, reduction_mode="median", output_dir_path=None,
  output_name="benchmark_eigh"):
  """ Runs benchmarks of eigh. One benchmark per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    rows : rows of the input matrix
    cols: cols of the input matrix. If `svd == False` and `cols != rows` or
      None the run is skipped. If `cols >= rows` the run is skipped.
    dtypes : dtype of the matrices
    p_szs : panel size of the summa multiplications
    precisions: asic matmul precision
    seeds : random seed to initialize input; system clock if none.
    serial: whether to run in serial or distributed mode.
    canonical: Whether to run in canonical or grand canonical mode.
    svd: Whether to compute the `SVD` or the symmetric eigendecomposition.
      With `SVD == True`, the parameters related to the eigenspectrum are
      used to construct the input matrix's positive-semidefinite factor.
      `eig_min, eig_max, gap_start` must therefore be non-negative in this case.
    eig_min: most negative eigenvalue in the spectrum.
    eig_max: most positive eigenvalue in the spectrum.
    k_factor: a gap of fixed size may optionally be added at the
      `rows // k_factor`'th eigenvalue.
    gap_start: Value of `spectrum[k_target - 1]`; must be non-negative if
      `SVD == True` (or else the run is skipped).
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
    rows, cols, dtypes, p_szs, precisions, seeds, serial, canonical, svd,
    eig_min, eig_max, k_factor, gap_start, gap_factor, ev_distribution))

  param_headers = [
    "N_rows", "N_cols", "dtype", "p_sz", "precision", "seed", "serial",
    "canonical", "svd", "eig min", "eig max", "occupancy", "gap start",
    "gap factor", "ev distribution"]
  init_f = functools.partial(analyze_eigh._initialize_eigh, return_answer=False)
  target_f = analyze_eigh._run_eigh

  speed_functions = [_eigh_tflops, _eigh_gbps, _eigh_skip]

  return analysis.benchmark(init_f, target_f, params, param_headers,
                            reduction_mode=reduction_mode,
                            batch_size=batch_size,
                            speed_functions=speed_functions,
                            output_dir_path=output_dir_path,
                            output_name=output_name)


def main():
  ROWS = (512,)  # rows of the input matrix
  COLS = (None, 64) # cols of the input matrix; == rows if None.
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False, True) # Whether to run in serial or distributed mode.
  CANONICAL = (False, True) # Whether to run in canonical or grand canonical
                            # mode.
  SVD = (False, True) # Whether to compute the SVD.
  EIG_MIN = (-10., 1.) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (10.,) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (10, 2) # Location of gap.
  GAP_START = (2.,) # eigenvalue[COLS // K_FACTOR]
  GAP_FACTOR = (10,) # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 5 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "benchmark_eigh" # output saved to OUTPUT_NAME.csv
  _ = benchmark_eigh(
    ROWS, COLS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, canonical=CANONICAL, svd=SVD, eig_min=EIG_MIN,
    eig_max=EIG_MAX, k_factor=K_FACTOR, gap_start=GAP_START,
    gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
