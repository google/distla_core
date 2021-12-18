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
"""Serial algorithm for eigh, employing canonical (fixed trace) purification."""
import functools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.linalg.eigh.serial import purify
from distla_core.linalg.polar.serial import polar
from distla_core.utils import misc


@functools.partial(jax.jit, static_argnums=(3, 4))
def split_spectrum(P, H, V, k, precision):
  """
  Computes projections of the matrix H into the column and null spaces
  of the projector P.
  Returns the projected matrices along with copies of V, updated to now
  include the isometries effecting the projections.

  Args:
    P: The `N x N` Hermitian projector into the subspace of `H`'s `k` smallest
       eigenvalues.
    H: The `N x N` Hermitian matrix to project.
    k: The number of eigenvalues to be placed in the subspace of small
       eigenvalues.
    V: Matrix of isometries to be updated
    precision: The matmul precision.
  Returns:
    H_minus: The `k x k` matrix sharing `H`'s `k` smallest eigenvalues.
    V_minus: `V` times the isometry mapping `H` to `H_minus`.
    H_plus: The `N-k x N-k` matrix sharing `H`'s `N-k` largest eigenvalues.
    V_plus: `V` times the isometry mapping `H` to `H_plus`.
  """
  V_minus, V_plus = purify.subspace(P, k, precision, "complete")
  H_minus = misc.similarity_transform(H, V_minus, precision)
  H_plus = misc.similarity_transform(H, V_plus, precision)

  if V is not None:
    V_minus = jnp.dot(V, V_minus, precision=precision)
    V_plus = jnp.dot(V, V_plus, precision=precision)
  return H_minus, V_minus, H_plus, V_plus


def _combine_eigenblocks(out_minus, out_plus):
  """
  Concatenates H_minus with H_plus, and V_minus with V_plus.
  """
  H_minus, V_minus = out_minus
  H_plus, V_plus = out_plus
  H = np.hstack((H_minus, H_plus))
  V = np.hstack((V_minus, V_plus))
  return H, V


def _eigh_work(H, V, precision):
  """
  The main work loop performing the symmetric eigendecomposition of an
  `N x N` Hermitian matrix `H`.
  Each step recursively computes a projector into the space of eigenvalues
  above and beneath the `N // 2`'th eigenvalue.
  The result of the projections into and out of
  that space, along with the isometries accomplishing these, are then computed.
  This is performed recursively until the projections have size 128, at
  which point a standard eigensolver is used. The results are then composed.

  A future implementation will use the Jax rather than the NumPy version of
  `eigh`, once the fast ASIC version of the former is added.

  Args:
    H: The Hermitian input.
    V: Stores the isometries projecting H into its subspaces.
    precision: The matmul precision.

  Returns:
    H, V: The result of the projection.
  """
  N = H.shape[0]
  if N <= 128:
    H, Vk = np.linalg.eigh(H)  # TODO: replace with jnp.linalg.eigh
    if V is not None:
      Vk = jnp.dot(V, Vk, precision=precision)
    return H, Vk

  k = N // 2
  P, _, errs = purify.canonically_purify(H, k, precision=precision)
  H_minus, V_minus, H_plus, V_plus = split_spectrum(P, H, V, k, precision)
  out_minus = _eigh_work(H_minus, V_minus, precision)
  out_plus = _eigh_work(H_plus, V_plus, precision)
  return _combine_eigenblocks(out_minus, out_plus)


def eigh(H, precision=lax.Precision.HIGHEST):
  """
  Computes the eigendecomposition of the symmetric/Hermitian matrix H.

  Args:
    H: The Hermitian input. Hermiticity is not enforced.
    precision: The matmul precision.
  Returns:
    evals, eVecs: The *unsorted* eigenvalues and eigenvectors.
  """
  N, M = H.shape
  if N != M:
    raise TypeError(f"Input H of shape {H.shape} must be square.")
  if N <= 128:
    return np.linalg.eigh(H)

  ev, eV = _eigh_work(H, None, precision)
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
