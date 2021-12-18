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
"""Serial algorithm for eigh."""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from distla_core.linalg.eigh.serial import purify
from distla_core.linalg.polar.serial import polar
from distla_core.utils import misc


@functools.partial(jax.jit, static_argnums=(2,))
def positive_projector(H, sigma, precision, polar_eps=None, polar_maxiter=200):
  """ Computes an orthogonal projector into the invariant subspace corresponding
  to the eigenvalues of H above the specified sigma. Returns, that is,
  P = 0.5 * (U + I) where U = polar(H - sigma * I).

  Args:
    H: The matrix to be projected.
    sigma: Bounds the subspace to be projected into.
    precision: The matmul precision.
    polar_eps: Accuracy threshold for an internal polar decomposition.
    polar_maxiter: Termination threshold for an internal polar decomposition.
  Returns:
    P: The projector.
    k: Rank of the projector.
  """
  Id = jnp.eye(H.shape[0], dtype=H.dtype)
  H_shifted = H - sigma * jnp.abs(Id)
  Up, _, _, _ = polar.polarU(
      H_shifted, eps=polar_eps, maxiter=polar_maxiter, precision=precision)
  P = 0.5 * (Up + Id)
  return P


@functools.partial(jax.jit, static_argnums=(3, 4))
def _split_spectrum_work(P, H, V, k, precision):
  """
  This helper function performs the bulk of operations for _split_spectrum.
  It is separated from the latter so that the rank of the projector k can
  concretized before being passed into Jitted code.

  Args:
    P: Projection matrix.
    H: Matrix to be projected.
    V: Accumulates the isometries into the projected subspaces.
    k: Rank of P.
    precision: The matmul precision.
  Returns:
    H1, V1: Projection of H into the column space of P, and the accumulated
            isometry performing that projection.
    H2, V2: Projection of H into the null space of P, and the accumulated
            isometry performing that projection.
  """
  Vk1, Vk2 = purify.subspace(P, k, precision, "complete")
  H1 = misc.similarity_transform(H, Vk1, precision)
  H2 = misc.similarity_transform(H, Vk2, precision)
  V1 = jnp.dot(V, Vk1, precision=precision)
  V2 = jnp.dot(V, Vk2, precision=precision)
  return H1, V1, H2, V2


def split_spectrum(P, H, V, precision=lax.Precision.HIGHEST):
  """
  Computes projections of the matrix H into the column and null spaces
  of the projector P.
  Returns the projected matrices along with copies of V, updated to now
  include the isometries effecting the projections.

  Args:
    P: The projector.
    H: The matrix to project.
    V: Matrix of isometries to be updated
    precision: The matmul precision.
  Returns:
    H1: The projection of H into the column space of P.
    V1: V times the isometry mapping H to H1.
    H2: The projection of H into the null space of P.
    V2: V times the isometry mapping H to H2.
  """
  k = jnp.round(jnp.trace(P)).astype(jnp.int32)
  k = int(k)
  return _split_spectrum_work(P, H, V, k, precision)


def _combine_eigenblocks(H1, V1, H2, V2):
  """
  Concatenates H1 with H2, and V1 with V2.
  """
  H = jnp.hstack((H1, H2))
  V = jnp.hstack((V1, V2))
  return H, V


def _eigh_work(H, V, median_ev_func, precision=lax.Precision.HIGHEST):
  """
  The main work loop performing the symmetric eigendecomposition of H.
  Each step recursively computes a projector into the space of eigenvalues
  above jnp.mean(jnp.diag(H)). The result of the projections into and out of
  that space, along with the isometries accomplishing these, are then computed.
  This is performed recursively until the projections have size 1, and thus
  store an eigenvalue of the original input; the corresponding isometry is
  the related eigenvector. The results are then composed.

  This function cannot be Jitted because the internal split_spectrum cannot
  be.

  Args:
    H: The Hermitian input.
    V: Stores the isometries projecting H into its subspaces.
    median_ev_func: A function of one matrix-valued argument that will be
       used to estimate that argument's median eigenvalue. Currently only
       the default jnp.mean(jnp.diag(X)) is tested.
    precision: The matmul precision.

  Returns:
    H, V: The result of the projection.
  """
  if H.shape[0] <= 128:
    H, Vk = jnp.linalg.eigh(H)
    V = jnp.dot(V, Vk, precision=precision)
    return H, V

  sigma = median_ev_func(H)
  P = positive_projector(H, sigma, precision)
  H1, V1, H2, V2 = split_spectrum(P, H, V, precision=precision)
  H1, V1 = _eigh_work(H1, V1, median_ev_func, precision=precision)
  H2, V2 = _eigh_work(H2, V2, median_ev_func, precision=precision)
  H, V = _combine_eigenblocks(H1, V1, H2, V2)
  return H, V


def _initial_step(H, median_ev_func, precision=lax.Precision.HIGHEST):
  """
  Does the first step of _eigh_work. Separating this code avoids initialization
  and multiplication by the matrix V, which at this stage is simply the
  identity.
  """
  sigma = median_ev_func(H)
  P = positive_projector(H, sigma, precision)
  k = jnp.round(jnp.trace(P)).astype(jnp.int32)
  k = int(k)
  V1, V2 = purify.subspace(P, k, precision, "complete")
  H1 = misc.similarity_transform(H, V1, precision)
  H2 = misc.similarity_transform(H, V2, precision)
  return H1, V1, H2, V2


def eigh(H, median_ev_func=None, precision=lax.Precision.HIGHEST):
  """
  Computes the eigendecomposition of the symmetric/Hermitian matrix H.

  Args:
    H: The Hermitian input. Hermiticity is not enforced.
    median_ev_func: A function of one matrix-valued argument that will be
       used to estimate that argument's median eigenvalue. Currently only
       the default jnp.mean(jnp.diag(X)) is tested.
    precision: The matmul precision.
  Returns:
    evals, eVecs: The sorted eigenvalues and eigenvectors.
  """
  N, M = H.shape
  if N != M:
    raise TypeError(f"Input H of shape {H.shape} must be square.")

  if N <= 128:
    return jnp.linalg.eigh(H)

  if median_ev_func is None:

    def median_ev_func(H):
      return jnp.mean(jnp.diag(H))
  else:
    raise NotImplementedError("median_ev_func is not implemented.")

  H1, V1, H2, V2 = _initial_step(H, median_ev_func, precision=precision)
  ev1, eV1 = _eigh_work(H1, V1, median_ev_func, precision=precision)
  ev2, eV2 = _eigh_work(H2, V2, median_ev_func, precision=precision)
  ev, eV = _combine_eigenblocks(ev1, eV1, ev2, eV2)
  sort_idxs = jnp.argsort(ev)
  ev = ev[sort_idxs]
  eV = eV[:, sort_idxs]
  return ev, eV


def svd(A, precision=lax.Precision.HIGHEST):
  """
  Computes the SVD of the input matrix A.

  Args:
    A: The input matrix.
    precision: The matmul precision.
  Returns:
    U, S, V: Such that A = (U * S) @ V.conj().T
  """
  Up, H, _, _, _ = polar.polar(A, precision=precision)
  S, V = eigh(H, precision=precision)
  U = jnp.dot(Up, V, precision=precision)
  return U, S, V
