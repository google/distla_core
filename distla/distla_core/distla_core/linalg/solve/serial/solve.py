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
"""Contains linear solvers, finding x in A @ x = b given A and b.
"""
import jax
import jax.numpy as jnp

from distla_core.linalg.polar.serial import polar


def solve_positive_definite(A, b, maxiter=100, eps=None):
  """
  Solves `A @ x = b` for the vector `x` where `A` is a positive definite
  matrix and `b` is a vector.

  The system is solved using the matrix exponential algorithm described in
  DOI: 10.1109/CDC.2011.6160593.  The algorithm works by viewing `A @ x = b`
  as a discretization of the continuous-time dynamical system
  `y'(t) = -A y(t) + b`, with y(t->inf) = x after discretization.

  The solution of this system at time `t = k h` is
     `y((k+1)h) = exp(-Ah)y(kh) + int_0^h exp(-A(h - tau))b d tau`.
  For large `h` we then have `y ~ x`, and since A is positive-definite,
  `exp(-Ah)y(kh) ~ 0`. Again since A is positive definite we can solve
  the integral to find:
    `x ~ (-A)^-1 @ (exp(-A h) - I) @ b`.

  If we now form the matrix
    X = | A  b |
        | 0  0 |
  we find
    exp(Xh) ~ | exp(-Ah) y(kh)  x |
              | 0               0 |
  for large h. The algorithm forms this matrix and approximates
  exp(Xh) by Taylor expansion, which amounts to repeatedly squaring
  `Y = I + X * h / 2 ^ s` where `s` is a constant.

  Args:
    A: A positive definite matrix (N x N).
    b: A length-N vector.
    maxiter: Limits the number of iterations.
    eps: Accuracy threshold for convergence:
         ||Y[:N, :N]|| / ||Y[:N, N]|| <= x is the convergence criterion
         where ||.|| is the Frobenius norm.
  Returns:
    x: The solution.
    j: The number of iterations.
  """
  if eps is None:
    eps = jnp.finfo(A.dtype).eps

  lambda_max = jnp.linalg.norm(A)
  N = A.shape[0]
  if N != A.shape[1]:
    raise TypeError(f"Input A must be square but had shape {A.shape}.")
  Y = jnp.eye(N + 1, dtype=A.dtype)
  Y = Y.at[:N, :N].add(-A / lambda_max)
  Y = Y.at[:N, N].set(b / lambda_max)

  def _check_if_done(args):
    j, Y = args
    norm11 = jnp.linalg.norm(Y[:N, :N])
    norm12 = jnp.linalg.norm(Y[:N, N])
    not_converged = (norm11 / norm12) > eps
    not_out_of_time = j < maxiter
    return jnp.logical_and(not_converged, not_out_of_time)

  def _square(args):
    j, Y = args
    return j + 1, jnp.dot(Y, Y, precision=jax.lax.Precision.HIGHEST)

  j, Y = jax.lax.while_loop(_check_if_done, _square, (0, Y))
  return Y[:N, N], j


def solve(A, b, maxiter=100, eps=None):
  """
  Solves `A @ x = b` for the vector `x` where `A` is a nonsingular square
  matrix and `b` is a vector.

  Args:
    A: A nonsingular square matrix (N x N).
    b: A length-N vector.
    maxiter: Limits the number of iterations.
    eps: Accuracy threshold for convergence; defaults to machine eps.
  Returns:
    x: The solution.
    j: The number of iterations.
  """
  U, H, _, _, _ = polar.polar(A, eps=eps, maxiter=maxiter)
  Ub = jnp.dot(U.conj().T, b, precision=jax.lax.Precision.HIGHEST)
  return solve_positive_definite(H, Ub, maxiter=maxiter, eps=eps)
