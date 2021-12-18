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
Functions to compute the polar decomposition of the m x n matrix matrix = U @ H
where U is unitary (an m x n isometry in the m > n case) and H is n x n and
positive
semidefinite (or positive definite if matrix is nonsingular). The method
is described in the docstring to `polarU`. This file covers the serial
case. These functions are the interface, with the work functions defined
in distla_core.linalg.polar_backend.
"""
import jax.numpy as jnp
from jax import lax

from distla_core.linalg.backends import serial_backend
from distla_core.linalg.polar import polar_utils


def polar(matrix,
          eps=None,
          maxiter=200,
          s_min=None,
          s_thresh=0.1,
          precision=lax.Precision.HIGHEST):
  """
  Computes the polar decomposition of the m x n matrix = U @ H where U is
  unitary (an m x n isometry in the m > n case) and H is n x n and positive
  semidefinite (or positive definite if matrix is nonsingular). The method
  is described in the docstring to `polarU`.

  matrixrgs:
    matrix: The m x n input matrix. Currently n > m is unsupported.
    eps: The final result will satisfy |n - trace(U^H @ U)| <= eps.
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
    s_min: An underestimate of the smallest singular value of matrix. Machine
           epsilon is used if unspecified.
    s_thresh: The iteration switches from the `rogue` polynomial to the
              Newton-Schultz iterations (see the polarU docstring) after
              s_min is estimated to have reached this value.
    precision: The precision of the matrix multiplications.
  Returns:
    unitary: The unitary factor (m x n).
    posdef: The positive-semidefinite factor (n x n).
    j_rogue: The number of `rogue` iterations.
    j_total: The total number of iterations.
    errs: Convergence history.
  """
  unitary, j_rogue, j_total, errs = polarU(
      matrix,
      eps=eps,
      maxiter=maxiter,
      s_min=s_min,
      s_thresh=s_thresh,
      precision=precision)
  posdef = jnp.dot(unitary.conj().T, matrix, precision=precision)
  posdef = 0.5 * (posdef + posdef.T.conj())
  return unitary, posdef, j_rogue, j_total, errs


def polarU(matrix,
           eps=None,
           maxiter=200,
           s_min=None,
           s_thresh=0.1,
           precision=lax.Precision.HIGHEST):
  """
  Computes the unitary factor in the polar decomposition of matrix. An iterative
  method is used: first, the singular values of matrix are bounded to within
  the range [s_thresh, 1] by repeated application of the so-called `rogue`
  polynomial
    X' = a_m * X - 4 * (a_m/3)**3 * X @ X^H @ X
  where a_m = (3 / 2) * sqrt(3) - s_thresh
  and `s_thresh` is a supplied lower bound.
  'rogue' polynomial method.

  matrixrgs:
    matrix: The m x n input matrix. Currently n > m is unsupported.
    eps: The final result will satisfy |n - trace(U^H @ U)| <= eps.
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
    s_min: An underestimate of the smallest singular value of matrix. Machine
           epsilon is used if unspecified.
    s_thresh: The iteration switches from the `rogue` polynomial to the
              Newton-Schultz iterations after
              s_min is estimated to have reached this value.
    precision: The precision of the matrix multiplications.
  Returns:
    unitary: The unitary factor (m x n).
    j_rogue: The number of `rogue` iterations.
    j_total: The total number of iterations.
    errs: Convergence history.
  """
  backend = serial_backend.SerialBackend(precision=precision)
  unitary, j_rogue, j_total, errs = polar_utils._polarU(
    matrix, eps, maxiter, s_min, s_thresh, backend)
  return unitary, j_rogue, j_total, errs
