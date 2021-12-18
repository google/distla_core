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

import functools
import itertools
from typing import Sequence

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.analysis import analysis
from distla_core.blas.summa import summa
from distla_core.linalg.tensor import utils
from distla_core.utils import pops


def _unpack_shape(out_rows, out_cols, shared_dim):
  if out_cols is None:
    out_cols = out_rows

  if shared_dim is None:
    shared_dim = out_rows
  return out_rows, out_cols, shared_dim


def _initialize_summa_constant(out_rows, out_cols, shared_dim, dtype,
                               p_sz, transpose_a, transpose_b, precision,
                               val_a, val_b):
  """ An `inif_f` functional argument to `benchmark` to initialize SUMMA
  using constant matrices of fixed Frobenius norm.

  Args:
    out_rows: Rows of the output matrix (called `m` in BLAS).
    out_cols: Columns of the output matrix (called `n` in BLAS).
    shared_dim: Shared dimension of the multiplication (called `k` in BLAS).
    p_sz: Panel size of the multiplication.
    transpose_a: Specifies whether to take `matrix_a = matrix_a.T`.
    transpose_b: Specifies whether to take `matrix_b = matrix_b.T`.
    precision: ASIC matmul precision.
    val_a: Elementwise value of `matrix_a`.
    val_b: Elementwise value of `matrix_b`.
  Returns:
    matrix_a: LHS matrix.
    matrix_b: RHS matrix.
    p_sz, transpose_a, transpose_b, precision: Same as input.
    val_c: Value of the (constant) elements of the result.
    norm_c: Frobenius norm of the result.
  """
  out_rows, out_cols, shared_dim = _unpack_shape(out_rows, out_cols, shared_dim)

  if transpose_a:
    shape_a = (shared_dim, out_rows)
  else:
    shape_a = (out_rows, shared_dim)

  if transpose_b:
    shape_b = (out_cols, shared_dim)
  else:
    shape_b = (shared_dim, out_cols)

  if val_b is None:
    val_b = val_a

  val_c = val_a * val_b * shared_dim
  norm_c = val_c * jnp.sqrt(out_rows * out_cols)

  matrix_a = utils.uniform(shape_a, pops.GRID, dtype=dtype, minval=val_a,
                           maxval=val_a)
  matrix_b = utils.uniform(shape_b, pops.GRID, dtype=dtype, minval=val_b,
                           maxval=val_b)
  out = (matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision, val_c,
         norm_c)
  return out


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4, 5),
                   in_axes=(0, 0, None, None, None, None, None, None))
def run_summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision,
              val_c, norm_c):
  """ Helper to call SUMMA.
  """
  return summa.summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b,
                     precision=precision)


@functools.partial(pops.pmap, in_axes=(0, None, None, None),
                   static_broadcasted_argnums=(3,))
def _compute_error(matrix_c, val_c, norm_c, relative):
  error = pops.frobnorm(matrix_c - val_c)
  if relative:
    error /= norm_c
  return error


def _summa_constant_error_f(result, in_args, params, relative):
  """ `error_function` for `measure_error` estimating the error in a
  multiplication between two constant matrices.
  This is ||result - expected||_F / ||denom||_F, where denom=1 for
  relative=False and ||expected||_F for relative=True.
  """
  val_c, norm_c = in_args[-2:]
  error = _compute_error(result, val_c, norm_c, relative)[0]
  header = "||C'-C||_F"
  if relative:
    header += "/||C||_F"
  return error, header


def _summa_constant_error_absolute(result, in_args, params):
  return _summa_constant_error_f(result, in_args, params, False)


def _summa_constant_error_relative(result, in_args, params):
  return _summa_constant_error_f(result, in_args, params, True)


def _summa_constant_norm_a(result, in_args, params):
  out_rows, out_cols, shared_dim = _unpack_shape(*(params[:3]))
  val_a = params[-2]
  size_a = shared_dim * out_rows
  norm_a = jnp.sqrt(size_a) * val_a
  return norm_a, "||A||_F"


def _summa_constant_norm_b(result, in_args, params):
  out_rows, out_cols, shared_dim = _unpack_shape(*(params[:3]))

  val_b = params[-1]
  if val_b is None:
    val_b = params[-2]
  size_b = shared_dim * out_cols
  norm_b = jnp.sqrt(size_b) * val_b
  return norm_b, "||B||_F"


def errors_summa_constant(out_rows: Sequence, val_as: Sequence,
                          val_bs: Sequence = (None,),
                          out_cols: Sequence = (None,),
                          shared_dim: Sequence = (None,),
                          dtypes=(jnp.float32,), p_szs=(128,),
                          transpose_as=(False,), transpose_bs=(False,),
                          precisions=(lax.Precision.HIGHEST,),
                          output_dir_path=None,
                          output_name="errors_summa_constant"):
  """
  Computes estimated errors from SUMMA, using constant matrices as input.
  One benchmark per combination of arguments is run. Output is saved to
  `output_dir_path/output_name`.csv, and also returned as a `DataFrame`.
  `output_dir_path` is the current working directory if unspecified.

  Args:
    out_rows: rows of the output matrix
    val_as: Elementwise value of the LHS matrix A.
    val_bs: Elementwise value of the RHS matrix B. None means same as A.
    out_cols: columns of the output matrix (=out_rows if None)
    shared_dim: shared dim of the multiply (=out_rows if None)
    dtypes: dtype of the matrices
    p_szs:  panel size of the multiplication
    transpose_as:  whether to do C: A.T @ B
    transpose_bs:  whether to do C: A @ B.T
    precisions: ASIC matmul precision
    reduction_mode: how to assemblage results
    output_dir_path: directory of output; CWD if None
    output_name: output saved to OUTPUT_NAME.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  param_lists = list(itertools.product(out_rows, out_cols, shared_dim,
                                       dtypes, p_szs, transpose_as,
                                       transpose_bs, precisions, val_as,
                                       val_bs))
  param_headers = ["N", "M", "K", "dtype", "p_sz", "transpose_a", "transpose_b",
                   "precision", "val_a", "val_b"]
  init_f = _initialize_summa_constant
  target_f = run_summa
  error_functions = [_summa_constant_norm_a,
                     _summa_constant_norm_b,
                     _summa_constant_error_absolute,
                     _summa_constant_error_relative]
  return analysis.measure_error(init_f, target_f, param_lists, param_headers,
                                error_functions=error_functions,
                                output_dir_path=output_dir_path,
                                output_name=output_name,
                                batch_size=1)


def main():
  OUT_ROWS = (1024, 2048, 4096)  # rows of the output matrix
  OUT_COLS = (None,)  # columns of the output matrix (=out_rows if None)
  SHARED_DIM = (None,)  # shared dim of the multiply (=out_rows if None)
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the multiplication
  TRANSPOSE_AS = (False,) # whether to do C = A.T @ B
  TRANSPOSE_BS = (False,) # whether to do C = A @ B.T
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  VAL_A = (0.25, 1.0, 2.0, 10.0) # Elementwise value of A.
  VAL_B = (0.25, 1.0, 2.0, 10.0) # Elementwise value of B. None means same as A.
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_summa" # output saved to OUTPUT_NAME.csv
  _ = errors_summa_constant(OUT_ROWS, VAL_A, VAL_B, out_cols=OUT_COLS,
                            shared_dim=SHARED_DIM, dtypes=DTYPES, p_szs=P_SZS,
                            transpose_as=TRANSPOSE_AS,
                            transpose_bs=TRANSPOSE_BS, precisions=PRECISIONS,
                            output_dir_path=OUTPUT_DIR_PATH,
                            output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
