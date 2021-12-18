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
Contains workhorse functions to compute the polar decomposition, interfaced
to by distla_core/linalg/polar.py and distla_core/linalg/serial/polar.py. This
avoids code duplication between these versions.
"""
import functools

import jax
from jax import lax
import jax.numpy as jnp

from distla_core.linalg.utils import testutils
from distla_core.utils import misc


def _eps_if_none(q, precision, dtype):
  # TODO: put this somewhere else, and base it on precision instead of dtype.
  if q is None:
    q = testutils.eps(precision, dtype=dtype)
  return q


@jax.jit
def _f3_scalar(x, a, b):
  """
  Scalar version of the polynomial iterations.
  """
  return a * x + b * x**3


@functools.partial(jax.jit, static_argnums=(3,))
def _f3_matrix(matrix, a, b, backend):
  """
  Computes a * X + b * X @ X^H @ X.
  """
  matrix3, _ = _f3_matrix_multiplication(matrix, backend)
  return a * matrix + b * matrix3


@functools.partial(jax.jit, static_argnums=(1,))
def _f3_matrix_multiplication(matrix, backend):
  """
  Computes X @ X^H @ X and X^H @ X.
  """
  matrix_dag_matrix = backend.matmul(
      matrix.conj(), matrix, transpose_A=True, transpose_B=False)
  matrix3 = backend.matmul(
      matrix, matrix_dag_matrix, transpose_A=False, transpose_B=False)
  return matrix3, matrix_dag_matrix


@functools.partial(jax.jit, static_argnums=(2, 5))
def _polarU(matrix, eps, maxiter, s_min, s_thresh, backend):
  """
  Computes the unitary factor in the polar decomposition. See the interface
  function docstrings in either version of polar.py for details.
  """
  global_rows, global_cols = backend.shape(matrix)
  if global_cols > global_rows:
    raise NotImplementedError("cols > rows case unimplemented.")
  coef = 1.0
  if eps is None:
    if backend.precision == lax.Precision.DEFAULT:
      coef = 0.5 * jnp.sqrt(global_rows)
    else:
      coef = 0.5 * global_rows
  eps = coef * _eps_if_none(eps, backend.precision, dtype=matrix.dtype)
  matrix = matrix / backend.frobnorm(matrix)
  s_min = _eps_if_none(s_min, lax.Precision.HIGHEST, dtype=matrix.dtype)
  matrix, rogue_itercount = _magnify_spectrum(matrix, maxiter, s_min, s_thresh,
                                              backend)
  matrix, total_itercount, errs = _newton_schultz(matrix, maxiter,
                                                  rogue_itercount, eps, backend)
  return matrix, rogue_itercount, total_itercount, errs


@functools.partial(jax.jit, static_argnums=(1, 4))
def _newton_schultz(matrix, maxiter, initial_itercount, eps, backend):
  """
  Repeatedly applies the Newton-Schulz polynomial
    X' = (3 / 2) * X - (1 / 2) * X @ X^H @ X
  to X. This procedure drives the singular values of X to 1 while preserving
  the singular vectors, and thus yields the unitary factor in the polar
  decomposition.
  """
  a_u = jnp.array(1.5, dtype=matrix.dtype)
  b_u = jnp.array(-0.5, dtype=matrix.dtype)
  _, global_cols = backend.shape(matrix)
  errs = jnp.zeros(maxiter, dtype=matrix.real.dtype)

  def making(args):
    _, err, j, _ = args
    return jnp.logical_and(j < maxiter, err > eps)

  def make(args):
    matrix, _, j, errs = args
    matrix3, matrix_dag_matrix = _f3_matrix_multiplication(matrix, backend)
    matrix = a_u * matrix + b_u * matrix3
    diff_norm = backend.frobnorm(matrix - matrix3)
    err = diff_norm / jnp.sqrt(global_cols)
    errs = errs.at[j - initial_itercount].set(err)
    return matrix, err, j + 1, errs

  out_0 = make((matrix, eps, initial_itercount, errs))
  out = jax.lax.while_loop(making, make, out_0)
  out = make(out)
  matrix, _, final_itercount, errs = out
  return matrix, final_itercount, errs


@functools.partial(jax.jit, static_argnums=(4,))
def _magnify_spectrum(matrix, maxiter, s_min, s_thresh, backend):
  """
  Repeatedly applies the 'rogue' polynomial
    X' = a_m * X - 4 * (a_m/3)**3 * X @ X^H @ X
  where a_m = (3 / 2) * sqrt(3) - s_thresh
  and `s_thresh` is a supplied lower bound. This
  polynomial drives the singular values of X to within the range
  [s_thresh, 1] while preserving the singular vectors, and serves as a
  preparatory step to accelerate the convergence of the later
  Newton-Schultz iterations.
  """
  j = 0
  s_thresh = jnp.array(s_thresh, dtype=matrix.dtype)
  s_min = jnp.array(s_min, dtype=matrix.dtype)
  a_m = jnp.array(1.5 * jnp.sqrt(3) - s_thresh, dtype=matrix.dtype)
  b_m = 4 * (a_m / 3)**3

  def magnifying(args):
    _, s_min, j = args
    return jnp.logical_and(j < maxiter, s_min < s_thresh)

  def magnify(args):
    matrix, s_min, j = args
    matrix = _f3_matrix(matrix, a_m, -b_m, backend)
    s_min = _f3_scalar(s_min, a_m, -b_m)
    return matrix, s_min, j + 1

  matrix, _, j = jax.lax.while_loop(magnifying, magnify, (matrix, s_min, j))
  return matrix, j
