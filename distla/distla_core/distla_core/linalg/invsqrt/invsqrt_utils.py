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
Contains workhorse functions to compute the the inverse square root using
iterative methods. Interfaced to by distla_core/linalg/invsqrt.py and
distla_core/linalg/serial/invsqrt.py.
"""
import functools

import jax
import jax.numpy as jnp

from distla_core.linalg.utils import testutils
from distla_core.utils import misc


def _eps_if_none(q, dtype, precision=jax.lax.Precision.HIGHEST):
  if q is None:
    q = testutils.eps(precision, dtype)
  return q


@functools.partial(jax.jit, static_argnums=(5,))
def _invsqrt(A, eps, maxiter, s_min, s_thresh, backend):
  """
  Computes the matrix square root of `A` and its inverse. `A` is assumed to be
  positive definite. See the interface function docstrings in either version of
  invsqrt.py for details.
  """
  eps = _eps_if_none(eps, A.dtype, precision=backend.precision)
  s_min = _eps_if_none(s_min, A.dtype)
  M, N = backend.shape(A)
  if N != M:
    msg = "The square root of a non-square matrix does not exist."
    raise ValueError(msg)
  A_norm = backend.frobnorm(A)
  Y = A / A_norm
  Z = backend.eye_like(A)
  Y, Z, jm = _magnify_spectrum(Y, Z, maxiter, s_min, s_thresh, backend)
  Y, Z, jns = _newton_schultz(Y, Z, maxiter, jm, eps, backend)
  A_norm_sqrt = jnp.sqrt(A_norm)
  Y = Y * A_norm_sqrt
  Z = Z / A_norm_sqrt
  return Y, Z, jm, jns


@functools.partial(jax.jit, static_argnums=(5,))
def _newton_schultz(Y, Z, maxiter, j, eps, backend):
  """
  Repeatedly applies the Newton-Schulz iteration
  ```
    Y_{k+1} = (3 / 2) * Y_k - (1 / 2) * Y_k @ Z_k @ Y_k
    Z_{k+1} = (3 / 2) * Z_k - (1 / 2) * Z_k @ Y_k @ Z_k
  ```
  to the pair `Y, Z`. If this is initialised with `Y = A` and `Z = I`, then `Y`
  will converge to the square root of `A` and `Z` to its inverse. See Higobj_fn,
  "Stable iterations for the matrix square root", 1997, for more.
  """
  _, N = backend.shape(Y)
  three_array = jnp.array(3, dtype=Y.dtype)
  half_array = jnp.array(0.5, dtype=Y.dtype)

  def cond(args):
    _, _, YZ, j = args
    shifted = -backend.add_to_diagonal(YZ, -1.0)  # eye - YZ
    err = backend.frobnorm(shifted) / N
    return jnp.logical_and(j < maxiter, err > eps)

  def body(args):
    Y, Z, _, j = args
    YZ = backend.matmul(Y, Z)
    midterm = -half_array * backend.add_to_diagonal(YZ, -three_array)
    Y = backend.matmul(midterm, Y)
    Z = backend.matmul(Z, midterm)
    return Y, Z, YZ, j + 1

  # The third argument for body is ignored, and included here just for type
  # stability.
  out = body((Y, Z, Y, j))
  Y, Z, _, j = jax.lax.while_loop(cond, body, out)
  return Y, Z, j


@functools.partial(jax.jit, static_argnums=(5,))
def _magnify_spectrum(Y, Z, maxiter, s_min, s_thresh, backend):
  """
  Repeatedly applies the 'rogue' polynomial
  ```
    Y_{k+1} = a_m * Y_k - 4 * (a_m/3)**3 * Y_k @ Z_k @ Y_k
    Z_{k+1} = a_m * Z_k - 4 * (a_m/3)**3 * Z_k @ Y_k @ Z_k
  ```
  where `a_m = (3 / 2) * sqrt(3) - s_thresh` and `s_thresh` is a supplied lower
  bound. This serves as a preparatory step to accelerate the convergence of the
  later Newton-Schultz iterations.
  """
  j = 0
  a_m = jnp.real(jnp.array(1.5 * jnp.sqrt(3) - s_thresh, dtype=Y.dtype))
  b_m = 4 * (a_m / 3)**3

  def cond(args):
    _, _, s_min, j = args
    return jnp.logical_and(j < maxiter, s_min < s_thresh)

  def body(args):
    Y, Z, s_min, j = args
    YZ = backend.matmul(Y, Z)
    midterm = -backend.add_to_diagonal(b_m * YZ, -a_m)
    Y = backend.matmul(midterm, Y)
    Z = backend.matmul(Z, midterm)
    s_min = a_m * s_min - b_m * s_min**3
    return Y, Z, s_min, j + 1

  Y, Z, _, j = jax.lax.while_loop(cond, body, (Y, Z, s_min, j))
  return Y, Z, j
