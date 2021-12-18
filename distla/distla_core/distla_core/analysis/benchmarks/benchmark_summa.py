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
Performs benchmarks of SUMMA. The benchmarks are performed for each combination
of various parameters, specified as the global (module-scope) variables below
in the case that this file is the main program. Output is returned as a
`DataFrame` and saved as a `.csv` file to `output_dir_path/output_name`.csv.
In the default case of `output_dir_path=None`, it is taken to be the current
working directory.
"""

from typing import Sequence
import itertools
import time

from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.analysis import analysis
from distla_core.analysis import pmaps
from distla_core.analysis.benchmarks import benchmark_utils
from distla_core.linalg.tensor import utils
from distla_core.utils import pops

###############################################################################
# CONFIGURATIONS WHEN THIS IS THE MAIN PROGRAM
###############################################################################
OUT_ROWS = (1024, 2048, 4096)  # rows of the output matrix
OUT_COLS = (None,)  # columns of the output matrix (=out_rows if None)
SHARED_DIM = (None,)  # shared dim of the multiply (=out_rows if None)
DTYPES = (np.float32,)  # dtype of the matrices
P_SZS = (128,)  # panel size of the multiplication
TRANSPOSE_AS = (False,) # whether to do C = A.T @ B
TRANSPOSE_BS = (False,) # whether to do C = A @ B.T
PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
SEED_AS = (None,)  # Random seed to initialize A; system clock if None.
SEED_BS = (None,)  # Same to initialize B; system clock + 1 if None.
BATCH_SIZE = 1     # Number of runs to assemblage.
REDUCTION_MODE = "median"  # how to assemblage results
OUTPUT_DIR_PATH = None  # directory of output; CWD if None
OUTPUT_NAME = "benchmark_summa" # output saved to OUTPUT_NAME.csv


def _initialize_summa(out_rows, out_cols, shared_dim, dtype,
                      p_sz, transpose_a, transpose_b, precision,
                      seed_a, seed_b):
  """ An `inif_f` functional argument to `benchmark` to initialize SUMMA.

  Args:
    out_rows: Rows of the output matrix (called `m` in BLAS).
    out_cols: Columns of the output matrix (called `n` in BLAS).
    shared_dim: Shared dimension of the multiplication (called `k` in BLAS).
    p_sz: Panel size of the multiplication.
    transpose_a: Specifies whether to take `matrix_a = matrix_a.T`.
    transpose_b: Specifies whether to take `matrix_b = matrix_b.T`.
    precision: ASIC matmul precision.
    seed_a: Random seed to initialize `A`. System clock if None.
    seed_b: Random seed to initialize `B`. System clock + 1 if None.
  Returns:
    matrix_a: LHS matrix of Gaussian random elements.
    matrix_b: RHS matrix of Gaussian random elements.
    p_sz, transpose_a, transpose_b, precision: Same as input.
  """
  if out_cols is None:
    out_cols = out_rows

  if shared_dim is None:
    shared_dim = out_rows

  if transpose_a:
    shape_a = (shared_dim, out_rows)
  else:
    shape_a = (out_rows, shared_dim)

  if transpose_b:
    shape_b = (out_cols, shared_dim)
  else:
    shape_b = (shared_dim, out_cols)

  if seed_a is None:
    seed_a = int(time.time())
  if seed_b is None:
    seed_b = int(time.time()) + 1

  if p_sz == -1:
    p_sz = min([out_rows, out_cols, shared_dim])

  matrix_a = utils.normal(shape_a, pops.GRID, dtype=dtype,
                          seed=seed_a).block_until_ready()
  matrix_b = utils.normal(shape_b, pops.GRID, dtype=dtype,
                          seed=seed_b).block_until_ready()
  return matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision


def run_summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision):
  """ Helper to call SUMMA.
  """
  out = pmaps.summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b,
                    precision=precision)
  out = out.block_until_ready()
  return out


def _summa_tflops(out, in_args, params):
  """ `speed_function` for `benchmark` estimating the throughput of a matrix
  multiplication in TFLOP/s.
  For a matrix multiplication of dimensions `m, n, k` that
  took `dt` seconds, we define `TFLOP/s := coef * m * n * k / (1E12 * dt)`,
  where `coef = 2` for real input and `coef = 8` for complex input.
  """
  dt = out[0]
  out_rows, out_cols, shared_dim, dtype = params[:4]
  if out_cols is None:
    out_cols = out_rows
  if shared_dim is None:
    shared_dim = out_rows
  if dtype in set([np.complex64, np.complex128, jnp.complex64, jnp.complex128]):
    coef = 8
  else:
    coef = 2
  flops = coef * out_rows * out_cols
  result = benchmark_utils.tflops_per_second(flops, dt)
  return result, "TFLOPS/s"


def _summa_gbps(out, in_args, params):
  """ `speed_function` for `benchmark` estimating the effective bandwidth
  of a matrix multiplication in GB/s.
  For a matrix multiplication of dimensions `m, n, k` that
  took `dt` seconds, we define `GB/s := (GB of input + GB of output) / (1E12 * dt)`,
  where `coef = 2` for real input and `coef = 8` for complex input.
  """
  dt = out[0]
  out_rows, out_cols, shared_dim, dtype = params[:4]
  if out_cols is None:
    out_cols = out_rows
  if shared_dim is None:
    shared_dim = out_rows
  elements_a = out_rows * shared_dim
  elements_b = shared_dim * out_cols
  elements_c = out_rows * out_cols
  n_elements = elements_a + elements_b + elements_c
  result = benchmark_utils.gbps(n_elements, dtype, dt)
  return result, "GB/s"


def benchmark_summa(out_rows: Sequence, out_cols: Sequence = (None,),
                    shared_dim: Sequence = (None,),
                    dtypes=(jnp.float32,), p_szs=(128,),
                    transpose_as=(False,), transpose_bs=(False,),
                    precisions=(lax.Precision.HIGHEST,),
                    seed_as=(None,), seed_bs=(None,), batch_size=1,
                    reduction_mode="median", output_dir_path=None,
                    output_name="benchmark_summa"):
  """
  Runs benchmarks of SUMMA. One benchmark per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    out_rows: rows of the output matrix
    out_cols: columns of the output matrix (=out_rows if None)
    shared_dim: shared dim of the multiply (=out_rows if None)
    dtypes: dtype of the matrices
    p_szs:  panel size of the multiplication
    transpose_as:  whether to do C: A.T @ B
    transpose_bs:  whether to do C: A @ B.T
    precisions: ASIC matmul precision
    seed_as: Random seed to initialize A; system clock if None.
    seed_bs: Same to initialize B; system clock + 1 if None.
    reduction_mode: how to assemblage results
    output_dir_path: directory of output; CWD if None
    output_name: output saved to OUTPUT_NAME.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  param_lists = [x for x in itertools.product(out_rows, out_cols, shared_dim,
                                              dtypes, p_szs, transpose_as,
                                              transpose_bs, precisions, seed_as,
                                              seed_bs)]
  param_headers = ["N", "M", "K", "dtype", "p_sz", "transpose_a", "transpose_b",
                   "precision", "seed_a", "seed_b"]
  init_f = _initialize_summa
  target_f = run_summa
  speed_functions = [_summa_tflops, _summa_gbps]
  return analysis.benchmark(init_f, target_f, param_lists, param_headers,
                            reduction_mode=reduction_mode,
                            speed_functions=speed_functions,
                            batch_size=batch_size,
                            output_dir_path=output_dir_path,
                            output_name=output_name)


if __name__ == "__main__":
  _ = benchmark_summa(OUT_ROWS, out_cols=OUT_COLS, shared_dim=SHARED_DIM,
                      dtypes=DTYPES, p_szs=P_SZS, transpose_as=TRANSPOSE_AS,
                      transpose_bs=TRANSPOSE_BS, precisions=PRECISIONS,
                      seed_as=SEED_AS, seed_bs=SEED_BS, batch_size=BATCH_SIZE,
                      reduction_mode=REDUCTION_MODE,
                      output_dir_path=OUTPUT_DIR_PATH,
                      output_name=OUTPUT_NAME)
