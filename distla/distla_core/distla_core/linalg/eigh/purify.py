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
from jax import lax
import jax.numpy as jnp
import warnings
from distla_core.blas.summa import summa
from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.eigh import purify_utils
from distla_core.linalg.polar import polar
from distla_core.utils import pops


###############################################################################
# PMAPS
###############################################################################
@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4, 5))
def _summa_pmap(matrix_l, matrix_r, p_sz, transpose_A, transpose_B, precision):
  return summa.summa(
      matrix_l, matrix_r, p_sz, transpose_A, transpose_B, precision=precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3))
def _add_matrix_square(summand, to_square, p_sz, precision):
  return summand + summa.summa(
      to_square, to_square.conj(), p_sz, False, True, precision=precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3))
def _similarity_transform_pmap(matrix_inner, matrix_outer, p_sz, precision):
  return summa.similarity_transform(
    matrix_inner, matrix_outer, p_sz, precision)


@functools.partial(pops.pmap, out_axes=None)
def _trace_pmap(matrix):
  return pops.trace(matrix)


def subspace(
    projector,
    projector_rank: int,
    local_ncols,
    p_sz: int,
    precision=lax.Precision.HIGHEST,
    subspace_iterations=2,
    polar_maxiter=50,
):
  """
  Finds an orthonormal basis for the column space of the Hermitian projector
  P whose rank is k_tup[0].

  This function is assumed to be pmapped, with arguments 1, 2, 3, 4 all
  static.

  Args:
    projector: The Hermitian projector.
    projector_rank: Rank of `projector`, and thus the number of columns of
      `isometry` without padding.
    local_ncols: The number of columns per-processor of the output
      isometry, including padding.
    p_sz: SUMMA panel size.
    precision: ASIC matmul precision.
    subspace_iterations: The number of subspace iterations to execute.
    polar_maxiter: The maximum number of polar iterations allowed per subspace
                   iteration.
  Returns:
    isometry:  An isometry into the column space of `projector`. It has local
      shape `(local rows of `projector`, `local_output_cols`)`. Its last
      `local_output_cols * NCOLS - rank(projector)` columns are all zero.
    info = 1 tuple (j_rogue, j_total, errs) per `subspace_iterations`:
      j_rogue: The number of 'rogue' iterations.
      j_total: The total number of polar iterations including rogue iterations.
      errs: errs[:j_total] is the convergence history of this polar
            decomposition.
  """
  # TODO: Support mixed precision.
  # TODO: Investibuilding_block effect of zero-eigenvalues in V.
  isometry = pops.eye((projector.shape[0], local_ncols), projector.dtype)
  isometry = apply_pad(isometry, projector_rank)
  info = []

  for i in range(subspace_iterations):
    product = summa.summa(
        projector, isometry, p_sz, False, False, precision=precision)
    isometry, j_rogue, j_total, errs = polar.polarU(
        product, p_sz=p_sz, maxiter=polar_maxiter, precision=precision)
    info.append((j_rogue, j_total, errs))
  return isometry, info


def apply_pad(matrix, unpadded_dim):
  """
  Zero-pads all entries of `matrix` outside its top-left unpadded_dim
  by unpadded_dim block (of the full distributed matrix).

  Args:
    matrix: A checkerboard-distributed matrix.
    unpadded_dim: Size of the block to leave unpadded.
  Returns:
    matrix: With the pad applied.
  """
  rows, cols = pops.indices(matrix.shape)
  left_of_k = rows < unpadded_dim
  above_k = cols < unpadded_dim
  return jnp.where(
      jnp.logical_and(left_of_k, above_k), x=matrix, y=jnp.zeros_like(matrix))


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(1, 3, 4),
    in_axes=(0, None, None, None, None),
    out_axes=(0, None, None, None))
def grand_canonically_purify(self_adjoint, unpadded_dim: int, split_point: float,
                             p_sz: int, precision):
  """ Computes an orthogonal projector into the invariant subspace corresponding
  to the eigenvalues of H *beneath* the specified `sigma`. Returns, that is,
  `P = -0.5 * (U - I)` where `U = polar(H - sigma * I)`.

  This function should not be pmapped (since it already is).

  Args:
    self_adjoint: The matrix to be projected.
    unpadded_dim: The size of the unpadded diagonal of `H`.
    split_point: Bounds the smallest / largest eigenvalue in the subspace.
    p_sz: SUMMA panel size.
    precision: SUMMA matmul precision.
  Returns:
    projector: The projector.
    j_rogue: Number of rogue iterations.
    j_total: Total number of iterations.
    errs: Convergence history.
  """
  shifted = pops.add_to_diagonal(
      self_adjoint, -split_point, unpadded_dim=unpadded_dim)
  unitary, j_rogue, j_total, errs = polar.polarU(
      shifted, p_sz=p_sz, precision=precision)
  projector = -0.5 * pops.add_to_diagonal(
      unitary, -1.0, unpadded_dim=unpadded_dim)
  return projector, j_rogue, j_total, errs


def canonically_purify(
  self_adjoint, target_rank, tol=None, maxiter=200, overlap_invsqrt=None,
  precision=lax.Precision.HIGHEST, method="hole-particle", p_sz=256,
    eigenvalue_bounds=None, unpadded_dim=None):
  """
  Computes a projection matrix into the eigenspace sharing the smallest
  `target_rank` eigenpairs as `H`.

  Args:
    self_adjoint:  The Hermitian matrix to be purified.
    target_rank: The rank of the projector to be computed (number of electrons,
              in the DFT interpretation).
    tol: Convergence is declared if the idempotency error drops beneath this
         value.
    maxiter: Maximum number of iterations allowed.
    overlap_invsqrt: If specified, a matrix `S^-1/2` such that
      `S^-1/2 @ H @ S^-1/2` is orthonormal. Both `H` and the output will in
      this case undergo this transformation.
    precision: lax matmul precision.
    method: The purification method to use. Currently only PM is supported.
    p_sz: Panel size for the SUMMA multiplications.
    eigenvalue_bounds: Optional guess (eig_min, eig_max) such that eig_min
      (eig_max) is a close lower (upper) bound on the most negative and most
      positive eigenvalue of the unpadded `self_adjoint` respectively.
    unpadded_dim: Optional handling of padded matrices; the number of
      columns of `self_adjoint` which are not part of the pad.
  Returns:
    projector: The projector.
    j: The number of iterations run, not including the NS steps.
    errs: Convergence history, including the NS steps.
  """
  backend = distributed_backend.DistributedBackend(
      precision=precision, p_sz=p_sz)
  if overlap_invsqrt is not None:
    self_adjoint = pops.pmap(backend.similarity_transform)(self_adjoint,
                                                        overlap_invsqrt)

  out = purify_utils.canonically_purify(
    self_adjoint, target_rank, backend, tol, maxiter, method,
    eigenvalue_bounds=eigenvalue_bounds, unpadded_dim=unpadded_dim)
  projector, j, errs = out

  if overlap_invsqrt is not None:
    projector = pops.pmap(backend.similarity_transform)(projector,
                                                        overlap_invsqrt)
  return projector, j, errs


###############################################################################
# Newton-Schulz purification.
###############################################################################
def _padded_local_ncols(logical_ncols):
  warnings.warn("purify._padded_local_ncols is deprecated in favour "
                "of pops.padded_local_ncols", DeprecationWarning)
  return pops.padded_local_ncols(logical_ncols)


@functools.partial(
    pops.pmap, static_broadcasted_argnums=(1, 2, 3, 4, 5), out_axes=(0, None))
def _subspace_step(projector, projector_rank, p_sz, precision, complement,
                   unpadded_dim):
  if complement:
    id_mat = pops.eye(projector.shape, projector.dtype)
    id_mat = apply_pad(id_mat, unpadded_dim)
    projector = id_mat - projector
    projector_rank = unpadded_dim - projector_rank
  isometry_ncols = _padded_local_ncols(projector_rank)
  isometry_k, subspace_info = subspace(projector, projector_rank,
                                       isometry_ncols, p_sz, precision)
  return isometry_k, subspace_info


@functools.partial(pops.pmap, in_axes=(0, None, None), out_axes=None)
def _mean_of_diag(matrix, unpadded_dim, target_rank):
  return pops.trace(matrix) / unpadded_dim


def _purify_step(self_adjoint, unpadded_dim, target_rank, split_point,
                 split_point_estimator, p_sz, precision):
  """ Computes a projector into the subspace of eigenvalues <= split_point.
  """
  if split_point is None:
    split_point = split_point_estimator(self_adjoint, unpadded_dim, target_rank)
  projector, j_rogue, j_total, errs = grand_canonically_purify(
      self_adjoint, unpadded_dim, split_point, p_sz, precision)
  errs = errs[:j_total - j_rogue]
  convergence_info = (j_rogue, j_total, errs)
  projector_rank = int(jnp.round(_trace_pmap(projector)))
  return projector, projector_rank, convergence_info


def _dac_purify_recursion(self_adjoint,
                          unpadded_dim,
                          target_rank,
                          n_recursions,
                          isometry_previous,
                          isometry_list,
                          p_sz,
                          precision,
                          max_recursions,
                          split_point_estimator,
                          split_point=None):
  """
  Recursive function doing the main work of `divide_and_conquer_purify`.
  """
  projector, projector_rank, convergence_info = _purify_step(
      self_adjoint, unpadded_dim, target_rank, split_point, split_point_estimator,
      p_sz, precision)

  if projector_rank < 1:
    raise ValueError(f"Projector had invalid rank {projector_rank} at"
                     f" iteration {n_recursions}.")
  if projector_rank == target_rank and n_recursions == 0:
    return projector, n_recursions, convergence_info

  isometry_kminus, subspace_minus_info = _subspace_step(
      projector, projector_rank, p_sz, precision, False, unpadded_dim)
  if isometry_previous is not None:
    isometry_minus = _summa_pmap(isometry_previous, isometry_kminus, p_sz,
                                 False, False, precision)
  else:
    isometry_minus = isometry_kminus

  if projector_rank == target_rank:
    projector = _summa_pmap(isometry_minus, isometry_minus.conj(), p_sz, False,
                            True, precision)
    for isometry in isometry_list:
      projector = _add_matrix_square(projector, isometry, p_sz, precision)
    return projector, n_recursions, convergence_info

  if projector_rank > target_rank:
    isometry_previous = isometry_minus
    self_adjoint = _similarity_transform_pmap(self_adjoint, isometry_kminus, p_sz,
                                           precision)
    unpadded_dim = projector_rank
  else:
    isometry_list += [
        isometry_minus,
    ]
    isometry_kplus, subspace_plus_info = _subspace_step(
        projector, projector_rank, p_sz, precision, True, unpadded_dim)
    isometry_previous = _summa_pmap(isometry_previous, isometry_kplus, p_sz,
                                    False, False, precision)
    self_adjoint = _similarity_transform_pmap(self_adjoint, isometry_kplus, p_sz,
                                           precision)
    unpadded_dim -= projector_rank
    target_rank -= projector_rank

  n_recursions += 1
  if n_recursions >= max_recursions:
    raise RuntimeError(f"User-specified max recursion depth {max_recursions}"
                       " exceeded.")
  return _dac_purify_recursion(
      self_adjoint, unpadded_dim, target_rank, n_recursions, isometry_previous,
      isometry_list, p_sz, precision, max_recursions, split_point_estimator)


def _distribute_if_needed(matrix):
  if matrix.ndim == 2:
    matrix = pops.distribute(matrix)
  elif matrix.ndim != 3:
    raise TypeError(f"matrix.ndim: {matrix.ndim} must be 2 or 3.")
  return matrix


def divide_and_conquer_purify(self_adjoint,
                              target_rank,
                              p_sz=None,
                              precision=lax.Precision.HIGHEST,
                              unpadded_dim=None,
                              max_recursions=200,
                              overlap_invsqrt=None,
                              split_point_estimator=None,
                              split_point_guess=None):
  """
  Computes the projection matrix into the eigenspace sharing `self_adjoint`'s
  lowest-`target_rank` eigenvalues. When `self_adjoint` can be interpreted as the
  ObjectiveFn sourced by `target_rank` particles as in DFT, the result is the
  ground state correlation matrix.

  Args:
    self_adjoint: The matrix to be purified. May be distributed or undistributed;
      the result will be distributed in either case.
    target_rank: Desired rank of the projector (density matrix).
    p_sz: SUMMA panel size. The largest possible size is used if None.
    precision: Jax matmul precision.
    unpadded_dim: Optional handling of padded matrices; the number of
      columns of `self_adjoint` which are not part of the pad.
    max_recursions: For debugging; the algorithm will raise RuntimeError if
      more recursions than this are performed.
    overlap_invsqrt: Optional handling of non-orthogonal DFT bases.
      If not None, we operate upon
      `overlap_invsqrt @ self_adjoint @ overlap_invsqrt^H` and return
      `overlap_invsqrt @ projector @ overlap_invsqrt^H`.
    split_point_estimator: A function with signature
      `f(matrix, unpadded_dim, target_rank)` estimating an intermediate
      eigenvalue of `matrix`. If unspecified, the (unweighted) mean of the
      trace is used.
    split_point_guess: An optional guess value used in place of the result
      of `split_point_estimator` at the first recursion.

  Returns:
    projector: The projector into the subspace sharing the original
      `self_adjoint`'s `target_rank` smallest eigenvalues.
    n_recursions: The number of recursions which were performed.
    info: In a future version this will store information about the convergence
          history.
  """
  self_adjoint = _distribute_if_needed(self_adjoint)
  nrows = self_adjoint.shape[1] * pops.NROWS
  ncols = self_adjoint.shape[2] * pops.NCOLS
  if nrows != ncols:
    raise TypeError("self_adjoint must be a square matrix, but it had shape "
                    f"({nrows}, {ncols}).")
  if p_sz is None:
    p_sz = ncols
  if unpadded_dim is None:
    unpadded_dim = ncols
  if split_point_estimator is None:
    split_point_estimator = _mean_of_diag

  if overlap_invsqrt is not None:
    overlap_invsqrt = _distribute_if_needed(overlap_invsqrt)
    self_adjoint = _similarity_transform_pmap(self_adjoint, overlap_invsqrt, p_sz,
                                           precision)

  projector, n_recursions, info = _dac_purify_recursion(
      self_adjoint,
      unpadded_dim,
      target_rank,
      0,
      None, [],
      p_sz,
      precision,
      max_recursions,
      split_point_estimator,
      split_point=split_point_guess)

  if overlap_invsqrt is not None:
    projector = _similarity_transform_pmap(projector, overlap_invsqrt, p_sz,
                                           precision)
  return projector, n_recursions, info
