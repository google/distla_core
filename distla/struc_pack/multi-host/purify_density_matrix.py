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
import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

from distla_core.blas.summa import summa
from distla_core.linalg.eigh import purify
from distla_core.linalg.invsqrt import invsqrt
from distla_core.linalg.polar import polar
from distla_core.utils import config
from distla_core.utils import misc
from distla_core.utils import pops

# # # UTILTIES # # #


def _pad_for_distribution(matrix, global_shape):
  """Pads a matrix so that it fits the distla_core distribution pattern."""
  g0, g1 = global_shape
  d0, d1 = matrix.shape
  largest_dimension = max(pops.GRID)
  pad0 = misc.distance_to_next_divisor(g0, largest_dimension)
  pad1 = misc.distance_to_next_divisor(g1, largest_dimension)
  b0 = (g0 + pad0) // pops.HGRID[0]
  b1 = (g1 + pad1) // pops.HGRID[1]
  result = np.zeros((b0, b1), dtype=matrix.dtype)
  result[:d0, :d1] += matrix
  return result


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4))
def similarity_transform(A, V, transpose, p_sz, precision):
  """Similarity transforms A by V.

  Args:
    A: The matrix to transform.
    V: The transformation.
    transpose: If `transpose is False`, return V^H @ A V, if `True`, V @ A V^H.
    p_sz: Summa panel size.
    precision: Matmul precision.

  Returns:
    The transformed matrix
  """
  if transpose:
    AV = summa.summa(A, V.conj(), p_sz, False, True, precision=precision)
    return summa.summa(V, AV, p_sz, False, False, precision=precision)
  else:
    AV = summa.summa(A, V, p_sz, False, False, precision=precision)
    return summa.summa(V.conj(), AV, p_sz, True, False, precision=precision)


# # # COMPUTING THE TRUNCATING ISOMETRY # # #


@functools.partial(
    pops.pmap, out_axes=(0, None), static_broadcasted_argnums=(1, 2))
def _condition_projector(
    overlap_matrix,
    overlap_threshold,
    condition_polar_kwargs,
):
  """Computes the projector onto the span of eigenvectors of `overlap_matrix`
  for which the eigenvalue is above `overlap_threshold`. Returns the projector
  and its rank.
  """
  # The use here of an eye that extends into the padded region is intentional:
  # We want to project out the padding as well, if possible.
  eye = pops.eye(overlap_matrix.shape, overlap_matrix.dtype)
  overlap_matrix = overlap_matrix - overlap_threshold * eye
  U, _, _, _ = polar.polarU(overlap_matrix, **condition_polar_kwargs)
  P = (U + eye) / 2
  k = pops.trace(P)
  return P, k


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3, 4, 5, 6))
def _subspace_iter(P, k, k_loc, n_iter, p_sz, precision, subspace_polar_kwargs):
  """Computes the isometry V such that P = V @ V^H. If P is of size D x D and
  rank k, then V will be of size D x k_loc where k_loc = k + k_pad, where k_pad
  makes sure that V fits the distla_core distribution pattern. The last k_pad
  columns of V will be zero.
  """
  V = pops.eye((P.shape[0], k_loc), P.dtype)
  V = pops.apply_pad(V, k)

  for i in range(n_iter):
    PV = summa.summa(P, V, p_sz, False, False, precision=precision)
    V, _, _, _ = polar.polarU(PV, **subspace_polar_kwargs)
  return V


def _condition_isometry(
    overlap_matrix,
    overlap_threshold,
    p_sz,
    precision,
    condition_polar_kwargs,
    subspace_n_iter,
    subspace_polar_kwargs,
):
  """Computes the isometry that projects onto the span of eigenvectors of
  `overlap_matrix` for which the eigenvalue is above `overlap_threshold`.
  Returns the isometry and the dimension of the space that it projects onto.
  """
  P, k = _condition_projector(
      overlap_matrix,
      overlap_threshold,
      condition_polar_kwargs,
  )
  k = int(np.round(k))
  largest_dimension = max(pops.GRID)
  k_pad = misc.distance_to_next_divisor(k, largest_dimension)
  k_loc = (k + k_pad) // pops.NCOLS
  V = _subspace_iter(
      P,
      k,
      k_loc,
      subspace_n_iter,
      p_sz,
      precision,
      subspace_polar_kwargs,
  )
  return V, k


# # # COMPUTING THE TRUNCATED INVERSE SQUARE ROOT # # #


def _set_padded_diagonal(M, k):
  """For a D x D matrix `M`, that is assumed to only be nonzero in `M[:k, :k]`
  due to padding, sets the diagonal of `M[k:, k:]` to be ones.

  This is needed when inverting a padded matrix, since the padding would create
  zero eigenvalues that would make the the inverse blow up.
  """
  eye = pops.eye(M.shape, M.dtype)
  rows, cols = pops.indices(M.shape)
  left_of_k = rows < k
  above_k = cols < k
  return jnp.where(jnp.logical_or(left_of_k, above_k), x=M, y=eye)


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4, 5))
def _overlap_matrix_invsqrt_part2(
    overlap_matrix,
    V,
    k,
    p_sz,
    precision,
    invsqrt_kwargs,
):
  """_overlap_matrix_invsqrt needs to be pmapped in two parts, this is the
  second part.
  """
  overlap_matrix = _set_padded_diagonal(overlap_matrix, k)
  _, om_invsqrt, _, _ = invsqrt.invsqrt(overlap_matrix, **dict(invsqrt_kwargs))
  om_invsqrt = pops.apply_pad(om_invsqrt, k)
  if V is not None:
    om_invsqrt = summa.summa(V, om_invsqrt, p_sz, False, False, precision)
  return om_invsqrt


def overlap_matrix_invsqrt(
    overlap_matrix,
    unpadded_dim,
    overlap_threshold=-1,
    p_sz=None,
    precision=jax.lax.Precision.HIGHEST,
    condition_polar_kwargs={},
    subspace_n_iter=2,
    subspace_polar_kwargs={},
    invsqrt_kwargs={},
):
  """Compute the inverse square root of an overlap matrix.

  The inverse is regularised by truncating away small eigenvalues. Hence the
  resulting inverse square root matrix may not be square, but of size D x k_loc,
  where D is the dimension of the original matrix, and k_loc = k + k_pad, with
  k_pad making sure that the matrix conforms to the distla_core distribution
  pattern.

  Args:
    overlap_matrix: The overlap matrix, as a numpy array.
    overlap_threshold: Eigenvalues of the overlap matrix below this number will
      be discarded.
    p_sz: Optional; SUMMA panel size. Maximum by default.
    precision: Optional; Jax matrix multiplication precision.
      `jax.lax.Precision.HIGHEST` by default
    condition_polar_kwargs: Optional; A dictionary of keyword arguments to be
      passed to `distla_core.linalg.polar.polarU` when computing the projector that
      truncates the overlap matrix. `{}` by default.
    subspace_n_iter: Optional; Number of subspace iterations when finding the
      isometry that truncates the overlap matrix.
    subspace_polar_kwargs: Optional; A dictionary of keyword arguments to be
      passed to `distla_core.linalg.polar.polarU` when computing the isometry that
      truncates the overlap matrix. `{}` by default.
    invsqrt_kwargs: Optional; A dictionary of keyword arguments to be
      passed to `distla_core.linalg.invsqrt.invsqrt` when computing the inverse
      square root of the overlap matrix. `{}` by default.

  Returns:
    om_invsqrt: Inverse square root of `overlap_matrix`.
    k: The unpadded dimension of `om_invsqrt`.
  """
  if p_sz is None:
    # In practice this is going to get cut down, this choice is essentially
    # equivalent to MAXINT.
    p_sz = max(overlap_matrix.shape)
  if "p_sz" not in condition_polar_kwargs:
    condition_polar_kwargs["p_sz"] = p_sz
  if "p_sz" not in subspace_polar_kwargs:
    subspace_polar_kwargs["p_sz"] = p_sz
  if "p_sz" not in invsqrt_kwargs:
    invsqrt_kwargs["p_sz"] = p_sz

  logging.info("Computing invsqrt(S)")
  if overlap_threshold > 0:
    V, k = _condition_isometry(
        overlap_matrix,
        overlap_threshold,
        p_sz,
        precision,
        condition_polar_kwargs,
        subspace_n_iter,
        subspace_polar_kwargs,
    )
    overlap_matrix = similarity_transform(
        overlap_matrix,
        V,
        False,
        p_sz,
        precision,
    )
  else:
    V = None
    k = unpadded_dim
  om_invsqrt = _overlap_matrix_invsqrt_part2(
      overlap_matrix,
      V,
      k,
      p_sz,
      precision,
      tuple(invsqrt_kwargs.items()),
  )
  return om_invsqrt, k


# # # MISCELLANEOUS # # #


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(1, 3),
    out_axes=(None, None),
    in_axes=(0, None, None, None))
def error_checks(rho, p_sz, num_occupied, precision):
  """Returns idempotency error (|rho^2 - rho|_F) and trace error
  (|Tr(rho) - num_occupied|) for the transformed density matrix, rho.
  """
  rho_squared = summa.summa(rho, rho, p_sz, False, False, precision)
  rho_norm = pops.frobnorm(rho)
  idempotency_error = pops.frobnorm(rho - rho_squared) / rho_norm
  rho_trace = pops.trace(rho)
  trace_error = jnp.abs(rho_trace - num_occupied) / num_occupied
  return idempotency_error, trace_error


def compute_energy_weighted_density_matrix(
    objective_fn,
    density_matrix,
    p_sz=None,
    precision=jax.lax.Precision.HIGHEST,
):
  """Given a (converged) ObjectiveFn and density matrix, computes the
  energy-weighted density matrix (EDM).

  Simply, Q = D @ H @ D, where Q is the EDM, D is the density matrix, and
  H is the ObjectiveFn. The EDM is used to calculate Pulay forces.

  Args:
    objective_fn: The ObjectiveFn, as a numpy array.
    density_matrix: The density matrix, as a numpy array.
    p_sz: Optional; SUMMA panel size. 128 by default.

  Returns:
    en_weighted_density_matrix: The energy-weighted density matrix,
      as a numpy array.
  """
  if p_sz is None:
    # In practice this is going to get cut down, this choice is essentially
    # equivalent to MAXINT.
    p_sz = max(objective_fn.shape)
  logging.info("Computing EDM")
  en_weighted_density_matrix = similarity_transform(
      objective_fn,
      density_matrix,
      False,
      p_sz,
      precision,
  )
  del objective_fn
  del density_matrix
  en_weighted_density_matrix.block_until_ready()  # For benchmarking
  return en_weighted_density_matrix


# # # PURIFICATION # # #


# REDACTED Should this sum be accumulated in efloat57?
@functools.partial(pops.pmap, out_axes=None)
def compute_ebs(objective_fn, density_matrix):
  local_sum = jnp.sum(objective_fn.conj() * density_matrix)
  return jax.lax.psum(local_sum, axis_name=pops.AXIS_NAME)


def purify_density_matrix(
    objective_fn,
    om_invsqrt,
    k,
    num_occupied,
    p_sz=None,
    precision=jax.lax.Precision.HIGHEST,
    canonically_purify_kwargs={},
):
  """Computes the DFT density matrix.

  By the density matrix we mean the projector onto the `num_occupied`
  eigenvectors with smallest eigenvalues of the generalised eigenvalue problem
  H D = e S D, where H and S are the ObjectiveFn and the overlap matrix,
  respectively.

  Args:
    objective_fn: The ObjectiveFn, as a numpy array.
    om_invsqrt: The inverse square root of the overlap_matrix, as a distributed
      ShardedDeviceArray.
    k: The unpadded dimension of `om_invsqrt`.
    num_occupied: Number of occupied modes in the density matrix.
    p_sz: Optional; SUMMA panel size. Maximum by default.
    precision: Optional; Jax matrix multiplication precision.
      `jax.lax.Precision.HIGHEST` by default
    canonically_purify_kwargs: Optional; A dictionary of keyword arguments to be
      passed to `distla_core.linalg.eigh.purify.canonically_purify`. `{}` by
      default.

  Returns:
    density_matrix: Approximation to the density matrix, as a numpy array.
    ebs: Electronic band structure energy (Tr[objective_fn @ density_matrix]).
  """
  if p_sz is None:
    # In practice this is going to get cut down, this choice is essentially
    # equivalent to MAXINT.
    p_sz = max(objective_fn.shape)
  if "p_sz" not in canonically_purify_kwargs:
    canonically_purify_kwargs["p_sz"] = p_sz

  logging.info("Type casting invsqrt(S)")
  om_invsqrt = pops.pmap(lambda x: x.astype(objective_fn.dtype))(om_invsqrt)
  om_invsqrt.block_until_ready()

  logging.info("Similarity transforming H")
  objective_fn = similarity_transform(
      objective_fn,
      om_invsqrt,
      False,
      p_sz,
      precision,
  )
  objective_fn.block_until_ready()  # For benchmarking
  logging.info("Running canonically purify")
  # TODO What to do about the fact that since objective_fn is padded,
  # it has fake 0 eigenvalues, which might get excited? Shift the unpadded part
  # to make it negative definite?
  out = purify.canonically_purify(
      objective_fn,
      num_occupied,
      unpadded_dim=k,
      **canonically_purify_kwargs,
  )
  density_matrix, purify_iters, purify_errs = out
  del out
  idempotency_error, trace_error = error_checks(
      density_matrix,
      p_sz,
      num_occupied,
      precision,
  )
  density_matrix.block_until_ready()  # For benchmarking
  logging.info(f'idempotency_error = {idempotency_error}')
  logging.info(f'trace_error = {trace_error}')
  logging.info(f'purify_iters = {purify_iters}')
  logging.info(f'purify_errs = {purify_errs}')

  logging.info("Computing EBS")
  ebs = compute_ebs(objective_fn, density_matrix)
  del objective_fn
  ebs.block_until_ready()  # For benchmarking
  logging.info("Similarity transforming the DM")
  density_matrix = similarity_transform(
      density_matrix,
      om_invsqrt,
      False,
      p_sz,
      precision,
  )
  density_matrix.block_until_ready()  # For benchmarking
  return density_matrix, ebs
