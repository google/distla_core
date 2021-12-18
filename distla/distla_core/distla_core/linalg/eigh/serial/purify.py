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
Functions to compute the projection matrix into specified eigenspaces
of a Hermitian matrix.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from distla_core.linalg.backends import serial_backend
from distla_core.linalg.eigh import purify_utils
from distla_core.linalg.polar.serial import polar
from distla_core.linalg.utils import testutils  # TODO: put `eps` somewhere else.
from distla_core.utils import misc


def canonically_purify(
  self_adjoint, k_target, tol=None, maxiter=200, overlap_invsqrt=None,
  precision=lax.Precision.HIGHEST, method="hole-particle",
    eigenvalue_bounds=None):
  """
  Computes a projection matrix into the eigenspace sharing the smallest
  `k_target` eigenpairs as `H`.

  Args:
    self_adjoint:  The Hermitian matrix to be purified.
    k_target: The rank of the projector to be computed (number of electrons,
              in the DFT interpretation).
    tol: Convergence is declared if the idempotency error drops beneath this
         value.
    maxiter: Maximum number of iterations allowed.
    overlap_invsqrt: If specified, a matrix `S^-1/2` such that
      `S^-1/2 @ H @ S^-1/2` is orthonormal. Both `H` and the output will in
      this case undergo this transformation.
    precision: lax matmul precision.
    method: The purification method to use. Currently only PM is supported.
    eigenvalue_bounds: Optional guess (eig_min, eig_max) such that eig_min
      (eig_max) is a close lower (upper) bound on the most negative and most
      positive eigenvalue of the unpadded `self_adjoint` respectively.
  Returns:
    projector: The projector.
    j: The number of iterations run.
    errs: Convergence history.
  """
  backend = serial_backend.SerialBackend(precision=precision)
  if overlap_invsqrt is not None:
    self_adjoint = backend.similarity_transform(self_adjoint, overlap_invsqrt)

  out = purify_utils.canonically_purify(
    self_adjoint, k_target, backend, tol, maxiter, method,
    eigenvalue_bounds=eigenvalue_bounds)
  projector, j, errs = out
  if overlap_invsqrt is not None:
    projector = backend.similarity_transform(projector, overlap_invsqrt)
  return projector, j, errs


###############################################################################
# Newton-Schulz purification.
###############################################################################
@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def subspace(P, k, precision, mode):
  """
  Computes isometries `V1` and `V2` respectively mapping into the column and
  null space of a rank-`k` projection matrix `P`.

  Args:
    P: The input matrix, expected to be a projector as described in the
       docstring.
    k: The rank of `P`.
    precision: The matmul precision.
  Returns:
    V1: A `(P.shape[0], k)` isometry into the column space of `P`.
    V2: A `(P.shape[0], P.shape[1] - k)` isometry into the null space of `P`.
  """
  norms_of_columns = jnp.linalg.norm(P, axis=0)
  sort_idxs = jnp.argsort(norms_of_columns)[::-1]
  sort_idxs = sort_idxs[:k]
  X = P[:, sort_idxs]
  V1, _ = jnp.linalg.qr(X, mode="reduced")

  X = jnp.dot(P, V1, precision=precision)
  V, _ = jnp.linalg.qr(X, mode=mode)

  if mode == "complete":
    V1 = V[:, :k]
    V2 = V[:, k:]
    return V1, V2

  return V, None


@functools.partial(jax.jit, static_argnums=(1, 3))
def _purify_step(H, precision, polar_tol, polar_maxiter):
  mu = jnp.median(jnp.diag(H))
  P, _, _, _ = grand_canonically_purify(
    H, mu, precision, polar_tol=polar_tol, polar_maxiter=polar_maxiter)
  k = jnp.round(jnp.trace(P)).astype(jnp.int32)
  return P, k


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def _subspace_step(P, k, k_target, precision, V_old):
  subspace_mode = "reduced"
  if k < k_target:
    subspace_mode = "complete"
  Vk_minus, Vk_plus = subspace(P, k, precision, subspace_mode)
  V_minus = jnp.dot(V_old, Vk_minus, precision=precision)
  return Vk_minus, Vk_plus, V_minus


def _ns_purify_work(polar_tol, polar_maxiter, max_recursions, precision, H,
                    k_target, j, V_old, V_list):
  """
  Recursive function implementing `newton_schulz_purify`.

  Args:
    polar_tol: Convergence threshold of the internal polar decompositions.
    polar_maxiter: Iteration cap for the internal polar decompositions.
    max_recursions: The algorithm will terminate with an error if more
                    recursions than this are performed.
    precision: Jax matmul precision.
    H: The current submatrix to be purified.
    k_target: The number of remaining eigenvalues of `H` to be found.
    j: The current number of recursions.
    V_old: Stores the isometries mapping from the original `H` to the current
           one.
    V_list: A list of isometries from which the projector will eventually
            be formed.

  Returns:
    P: The projector into the subspace sharing the original `H`'s `k`
       smallest eigenvalues.
    j: The number of recursions which were performed.
  """
  P, k = _purify_step(H, precision, polar_tol, polar_maxiter)
  k = int(k)
  if k == k_target and j == 0:
    return P, j

  j += 1
  if j >= max_recursions:
    raise RuntimeError(f"User-specified max recursion depth {max_recursions}"
                       " exceeded.")
  Vk_minus, Vk_plus, V_minus = _subspace_step(P, k, k_target, precision, V_old)

  if k == k_target:
    V_list += [V_minus, ]
    V_full = jnp.hstack(V_list)
    P = jnp.dot(V_full, V_full.conj().T, precision=precision)
    return P, j

  if k > k_target:
    V_old = V_minus
    H = misc.similarity_transform(H, Vk_minus, precision)
  else:
    V_list += [V_minus, ]
    V_old = jnp.dot(V_old, Vk_plus, precision=precision)
    H = misc.similarity_transform(H, Vk_plus, precision)
    k_target -= k
  return _ns_purify_work(polar_tol, polar_maxiter, max_recursions, precision,
                         H, k_target, j, V_old, V_list)


def newton_schulz_purify(H, k_target, precision=lax.Precision.HIGHEST,
                         tol=None, polar_maxiter=200, max_recursions=200):
  """
  Computes the projection matrix into the eigenspace sharing `H`'s
  lowest-`k` eigenvalues. When `H` can be interpreted as the ObjectiveFn
  sourced by `k` particles as in DFT, the result is the ground state
  correlation matrix.

  Args:
    H: The matrix to be purified.
    k: Number of positive eigenvalues in the purified result.
    precision: Jax matmul precision.
    tol: Convergence threshold of the internal polar decompositions.
    max_recursions: The algorithm will terminate with an error if more
                    recursions than this are performed.

  Returns:
    P: The projector into the subspace sharing the original `H`'s `k`
       smallest eigenvalues.
    j: The number of recursions which were performed.
  """
  if len(H.shape) != 2 or H.shape[0] != H.shape[1]:
    raise TypeError(f"H must be a square matrix, but had shape = {H.shape}.")

  if tol is None:
    tol = testutils.eps(precision, H.dtype) * H.shape[1] / 2

  V_old = jnp.eye(H.shape[0], dtype=H.dtype)
  V_list = []
  j = jnp.zeros(1, dtype=jnp.int32)
  return _ns_purify_work(tol, polar_maxiter, max_recursions, precision, H,
                         k_target, j, V_old, V_list)


def grand_canonically_purify(H, mu, precision, polar_tol=None,
                             polar_maxiter=200):
  """ Computes an orthogonal projector into the invariant subspace corresponding
  to the eigenvalues of `H` beneath the specified `mu`. Returns, that is,
  `P = -0.5 * (U - I)` where `U = polar(H - sigma * I)`.

  Args:
    H: The matrix to be projected.
    mu: Bounds the subspace to be projected into.
    precision: The matmul precision.
    polar_tol: Accuracy threshold for an internal polar decomposition.
    polar_maxiter: Termination threshold for an internal polar decomposition.
  Returns:
    P: The projector.
    k: Rank of the projector.
  """
  return _grand_canonically_purify_work(
    H, mu, precision, polar_tol, polar_maxiter)


@functools.partial(jax.jit, static_argnums=(2, 4))
def _grand_canonically_purify_work(H, mu, precision, polar_tol, polar_maxiter):
  Id = jnp.eye(H.shape[0], dtype=H.dtype)
  H_shifted = H - mu * jnp.abs(Id)
  Up, j_rogue, j_total, errs = polar.polarU(
      H_shifted, eps=polar_tol, maxiter=polar_maxiter, precision=precision)
  return -0.5 * (Up - Id), j_rogue, j_total, errs
