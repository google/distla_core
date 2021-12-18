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
Functions to compute the polar decomposition of the m x n matrix A, A = U @ H
where U is unitary (an m x n isometry in the m > n case) and H is n x n and
positive semidefinite (or positive definite if A is nonsingular). The method
is described in the docstring to `polarU`. This file covers the serial
case.
"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _dot(A, B):
  return jnp.dot(A, B, precision=jax.lax.Precision.HIGHEST)


def polar(A, eps=None, maxiter=6):
  """
  Computes the polar decomposition of the m x n matrix A, A = U @ H where U is
  unitary (an m x n isometry in the m > n case) and H is n x n and positive
  semidefinite (or positive definite if A is nonsingular) using the
  QDWH method.

  Args:
    A: The m x n input matrix. Currently n > m is unsupported.
    eps: The final result will satisfy |X_k - X_k-1| < |X_k| * (4*eps)**(1/3) .
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
  Returns:
    U: The unitary factor (m x n).
    H: The positive-semidefinite factor (n x n).
    jq: Number of QR iterations.
    jc: Number of Cholesky iterations.
  """
  U, jq, jc = qdwh(A, eps=eps, maxiter=maxiter)
  H = _dot(U.conj().T, A)
  H = 0.5 * (H + H.T.conj())
  return U, H, jq, jc


def qdwh(A, eps=None, maxiter=6):
  """
  Computes the unitary factor in the polar decomposition of A using
  the QDWH method. QDWH implements a 3rd order Pade approximation to the
  matrix sign function,

  X' = X * (aI + b X^H X)(I + c X^H X)^-1, X0 = A / ||A||_2.          (1)

  The coefficients a, b, and c are chosen dynamically based on an evolving
  estimate of the matrix condition number. Specifically,

  a = h(l), b = g(a), c = a + b - 1, h(x) = x g(x^2), g(x) = a + bx / (1 + cx)

  where l is initially a lower bound on the smallest singular value of X0,
  and subsequently unfolds according to l' = l (a + bl^2) / (1 + c l^2).

  For poorly conditioned matrices
  (c > 100) the iteration (1) is rewritten in QR form,

  X' = (b / c) X + (1 / c)(a - b/c) Q1 Q2^H,   [Q1] R = [sqrt(c) X]   (2)
                                               [Q2]     [I        ].

  For well-conditioned matrices it is instead formulated using cheaper
  Cholesky iterations,

  X' = (b / c) X + (a - b/c) (X W^-1) W^-H,   W = chol(I + c X^H X).  (3)

  The QR iterations rapidly improve the condition number, and typically
  only 1 or 2 are required. A maximum of 6 iterations total are required
  for backwards stability to double precision.

  Args:
    A: The m x n input matrix.
    eps: The final result will satisfy |X_k - X_k-1| < |X_k| * (4*eps)**(1/3) .
    maxiter: Iterations will terminate after this many steps even if the
             above is unsatisfied.
  Returns:
    U: The unitary factor (m x n).
    jq: The number of QR iterations (1).
    jc: The number of Cholesky iterations (2).
  """
  if eps is None:
    eps = jnp.finfo(A.dtype).eps
  eps = (4 * eps)**(1. / 3)
  X, Q, lk = _initialize_qdwh(A)
  coefs = _qdwh_coefs(lk)
  X, jq, coefs, err = _qdwh_qr(X, coefs, 2 * eps, eps, maxiter)
  X, jc, _, _ = _qdwh_cholesky(X, coefs, err, eps, maxiter - jq)
  X = _dot(Q, X)
  return X, jq, jc


@jax.jit
def _initialize_qdwh(A):
  """
  Does preparatory computations for QDWH:
    1. Computes an initial QR factorization of the input A. The iterations
       will be on the triangular factor R, whose condition is more easily
       estimated, and which is square even when A is rectangular.
    2. Computes R -> R / ||R||_F. Now 1 is used to upper-bound ||R||_2.
    3. Computes R^-1 by solving R R^-1 = I.
    4. Uses sqrt(N) * ||R^-1||_1 as a lower bound for ||R^-2||.
  1 / sqrt(N) * ||R^-1||_1 is then used as the initial l_0. It should be clear
  there is room for improvement here.

  Returns:
    X = R / ||R||_F;
    Q from A -> Q @ R;
    l0, the initial estimate for the QDWH coefficients.
  """
  m, n = A.shape
  if n > m:
    raise TypeError(f"m {m} < n {n} case unsupported.")
  Q, R = jnp.linalg.qr(A, mode="reduced")
  alpha = jnp.linalg.norm(R)
  R /= alpha
  Rinv = jsp.linalg.solve_triangular(R, jnp.eye(R.shape[0]))
  Rinvnorm = jnp.linalg.norm(Rinv, ord=1)
  l0 = 1 / (jnp.sqrt(n) * Rinvnorm)
  l0 = jnp.array(l0, dtype=R.real.dtype)
  return R, Q, l0


@jax.jit
def _qdwh_coefs(lk):
  """
  Computes a, b, c, l for the QDWH iterations.
  """
  d = (4. * (1. - lk**2) / (lk**4))**(1 / 3)
  f = 8. * (2. - lk**2) / (lk**2 * (1. + d)**(1 / 2))
  a = (1. + d)**(1 / 2) + 0.5 * (8. - 4. * d + f)**0.5
  b = (a - 1.)**2 / 4
  c = a + b - 1.
  lk = lk * (a + b * lk**2) / (1 + c * lk**2)
  return a, b, c, lk


@jax.jit
def _qdwh_qr(X, coefs, err0, eps, maxiter):
  """
  Applies the QDWH iteration formulated as

  X' = (b / c) X + (1 / c)(a - b/c) Q1 Q2^H,   [Q1] R = [sqrt(c) X]
                                               [Q2]     [I        ]

  to X until either c < 100, ||X' - X|| < eps||X'||,
  or the iteration count exceeds maxiter.
  """
  m, n = X.shape
  eye = jnp.eye(n, dtype=X.dtype)

  def _do_qr(args):
    X, j, coefs, err = args
    c = coefs[2]
    ill_conditioned = c >= 100.
    unconverged = err > (eps * jnp.linalg.norm(X))
    iterating = j < maxiter
    keep_going = jnp.logical_and(ill_conditioned, unconverged)
    return jnp.logical_and(keep_going, iterating)[0]

  def _qr_work(args):
    X, j, coefs, err0 = args
    a, b, c, lk = coefs
    csqrt = jnp.sqrt(c)
    XI = jnp.vstack((csqrt * X, eye))
    Q, _ = jnp.linalg.qr(XI, mode="reduced")
    Q1 = Q[:m, :]
    Q2 = Q[m:, :]
    coef = (1 / csqrt) * (a - (b / c))
    X *= (b / c)
    X += coef * _dot(Q1, Q2.T.conj())

    err = jnp.linalg.norm(X - XI[:m, :] / csqrt).astype(err0.dtype)
    coefs = _qdwh_coefs(lk)
    return X, j + 1, coefs, err

  j = jnp.zeros(1, dtype=jnp.float32)
  return jax.lax.while_loop(_do_qr, _qr_work, (X, j, coefs, err0))


@jax.jit
def _qdwh_cholesky(X, coefs, err0, eps, maxiter):
  """
  Applies the QDWH iteration formulated as

  X' = (b / c) X + (a - b/c) B,  B = (X W^-1) W^-H,  W = chol(I + c X^H X).

  to X until either ||X' - X|| < eps * ||X'||,
  or the iteration count exceeds maxiter.
  """
  m, n = X.shape
  eye = jnp.eye(n, dtype=X.dtype)

  def _do_cholesky(args):
    X, j, coefs, err = args
    unconverged = err > (eps * jnp.linalg.norm(X))
    iterating = j < maxiter
    return jnp.logical_and(unconverged, iterating)[0]

  def _cholesky_work(args):
    X, j, coefs, err0 = args
    X0 = X
    a, b, c, lk = coefs
    Z = eye + c * _dot(X.T.conj(), X)
    W = jsp.linalg.cholesky(Z)
    B = jsp.linalg.solve_triangular(W.T, X.T, lower=True).conj()
    B = jsp.linalg.solve_triangular(W, B).conj().T
    X = (b / c) * X + (a - b / c) * B
    err = jnp.linalg.norm(X - X0).astype(err0.dtype)
    coefs = _qdwh_coefs(lk)
    return X, j + 1, coefs, err

  j = jnp.zeros(1, dtype=jnp.float32)
  return jax.lax.while_loop(_do_cholesky, _cholesky_work, (X, j, coefs, err0))
