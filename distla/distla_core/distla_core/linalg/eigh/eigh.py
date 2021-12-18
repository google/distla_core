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
""" Distributed spectral divide and conquer eigensolver.

This module is concerned with the computation of a) eigendecompositions of
Hermitian matrices and b) SVDs of general matrices, using the `spectral
divide and conquer' (spectral D&C) method of Nagatsukasa and Higobj_fn,
https://www.semanticscholar.org/paper/Stable-and-Efficient-Spectral-Divide-and
-Conquer-Nakatsukasa-Higobj_fn/66de2f7018d073a5d9bce003bdb63c7b2905ccbb.

The SVD algorithm works as follows. We seek to write the input matrix `A`
as `A = U @ S @ V`, with the singular vector matrices `U` and `V^H`
unitary and the singular values matrix `S` nonnegative diagonal. To do this
we:
  a) Compute `A -> Up @ H` via polar factorization, with `Up` unitary and
     `H` Hermitian positive-definite. We have `Up = U @ V^H`.
  b) Compute the eigendecomposition `H = V @ S @ V^H`. `V` and `S` are the
     same matrices appearing in the SVD.
  c) Compute `U = Up @ V`.
  d) Return `U`, `S`, `V`. Note that `V` rather than `V^H` is returned.

This is implemented by the function `svd` contained within.

Most of the complexity here is contained in step b), the eigendecomposition of
a Hermitian matrix `H`. This is performed as follows:
  a) Hermitian projectors into the eigenspaces above and below a certain
     eigenvalue are computes. The details of this step differ depending
     on whether or not `canonical` mode is used:
      i) In `canonical` mode, purification techniques arising from "linear
         scaling" electronic structure theory are used to directly compute
         projectors into subspaces of size N // 2 and N - N // 2
         respectively.
      ii) Otherwise :
        ii.a) An estimate `sigma` of `H`'s *median* eigenvalue is made.
        ii.c) The unitary polar factor `Up` of `H' = H - sigma * I` is
          computed.
        ii.d) Hermitian projectors `P1 = -0.5 * (Up - Id)` (into the subspace
          beneath sigma) and `P2 = 0.5 * (Up + Id)` (into the subspace above it)
          are formed. The traces of these matrices are thus equal to their ranks
          and to the sizes of these subspaces.
  b) The projectors are factored into isometries `V` by subspace iteration.
  c) H1 = V1^H @ H @ V1 and H2 = V2^H @ H @ V2 now have spectra also within
     these subspaces. These matrices are k1 x k1 and k2 x k2.
  d) The entire process is repeated recursively upon H1 and H2. The isometries
     V1, V2 from previous iterations are composed (multiplied by) the new
     ones found at each stage.
  e) Eventually H1 and H2 reach some minimum size, at which point their
     eigendecompositions are computed by standard methods.
  f) The resulting eigenvalues and composed eigenvectors are those of the
     original matrix. They are combined and returned.


A distributed-ASIC implementation of this procedure encounters a few
complications:
  1) Whereas k1 and k2 may take essentially arbitrary values, the local array
     sizes on each core must be divisible by both `NROWS` and `NCOLS` in
     order to take transposes. Consequently we must manually pad `k1` and `k2`
     so that e.g. `k1_padded % NROWS == 0`.

     1.1) Apart from the obvious performance implications, this padding must be
          kept track of in order to be eventually removed, so that spurious
          zero-eigenvalues are not added to the spectrum.
  2) The 'minimum size' of step h), at which point the matrices are, needs to
     be around 128 for good local performance. Consequently the size of the
     *gathered* matrices presently increases with the number of cores. A future
     implementation will send reduced subproblems to larger subsets of the full
     processor grid (e.g. invidual asic_nodes) for further reduction before the
     single-core stage.
  3) The subspace iteration stage e) is a bit tricky since we lack a good
     QR factorization routine. Currently the orthonormalization step is handled
     by polar decomposition. Unfortunately this is expensive.
  4) Combining the factored "small" matrix into a distributed result is a
     bit tricky. This is handled by the various subroutines of `finalize`.
  5) Sorting the final result by eigenvalue is nontrivial and is not yet
     implemented.
"""

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxlib.xla_client import PrecisionConfig
import numpy as np
import scipy.optimize

from distla_core.blas.summa import summa
from distla_core.linalg.eigh import purify
from distla_core.linalg.eigh.serial import eigh as serial_eigh
from distla_core.linalg.polar import polar
from distla_core.linalg.tensor import utils
from distla_core.utils import misc
from distla_core.utils import pops
from distla_core.utils import vops

DistlaCoreMatrix = jnp.array


###############################################################################
# PMAPS - Wrappers for parallel functions needed by non-pmapped code.
###############################################################################
@functools.partial(misc.log_time, header="trace_f", shape_argnums=(0,))
@functools.partial(pops.pmap, out_axes=None)
def trace_f(matrix):
  return pops.trace(matrix)


@functools.partial(
    misc.log_time, header="polar_f", shape_argnums=(0,), block_argnums=(0,))
@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2))
def polar_f(A, p_sz, precision):
  Up, H, _, _, _ = polar.polar(A, p_sz=p_sz, precision=precision)
  return Up, H


@functools.partial(
    misc.log_time, header="summa_f", shape_argnums=(0,), block_argnums=(0,))
def summa_f(A, B, p_sz, precision, transpose_A=False, transpose_B=False):
  return _summa_f(A, B, p_sz, precision, transpose_A, transpose_B)


@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4, 5))
def _summa_f(A, B, p_sz, precision, transpose_A, transpose_B):
  return summa.summa(A, B, p_sz, transpose_A, transpose_B, precision)


@functools.partial(pops.pmap, in_axes=(0, 0, None))
def _set_columns_f(vecs, columns, col_start):
  return vops.set_columns(vecs, columns, col_start)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1,))
def _big_to_thin_f(A, trim_columns_to):
  return vops.big_to_thin(A, trim_columns_to=trim_columns_to)


@functools.partial(
    pops.pmap, static_broadcasted_argnums=(2,), in_axes=(0, None, None))
def _vecsmall_f(vec, small, precision):
    return vops.vecsmall(vec, small, precision=precision)


###############################################################################
# UTILITIES
###############################################################################
def _similarity_transform(
    A: DistlaCoreMatrix,
    V: DistlaCoreMatrix,
    p_sz: int,
    precision=lax.Precision.HIGHEST,
) -> DistlaCoreMatrix:
  """ Computes V.T.conj() @ A @ V.

  This function must be called within a pmap.

  Args:
    A, V: The checkerboard-distributed matrices of the transform.
    p_sz: Summa panel size.
    precision: Matmul precision.
  Returns:
    The result.
  """
  AV = summa.summa(A, V, p_sz, False, False, precision=precision)
  return summa.summa(V.conj(), AV, p_sz, True, False, precision=precision)


def _pad_to(unpadded_array, padded_dim):
  """ Zero-pads each dimension of `unpadded_array` to `padded_dim`.
  """
  padded_shape = tuple([padded_dim, ] * unpadded_array.ndim)
  if unpadded_array.shape == padded_shape:
    return unpadded_array

  start_idx = tuple([0, ] * unpadded_array.ndim)
  padded_array = jnp.zeros(padded_shape, dtype=unpadded_array.dtype)
  return lax.dynamic_update_slice(padded_array, unpadded_array, start_idx)


@functools.partial(misc.log_time, header="_padded_ranks")
def _padded_ranks(unpadded_dim: int, proj_rank: int):
  """ Computes (on the host) the number of columns to allocate to the next
  round of isometries so that transposition remains possible on a rectangular
  processor grid.

  Args:
    unpadded_dim: Unpadded linear size of the current `H`.
    proj_rank: Rank of the current negative projector.
  Returns:
    proj_rank: As an integer on the host.
    proj_local_dim: linear size of the padded rank `proj_rank` projector.
    comp_rank: `unpadded_dim - proj_rank`.
    comp_local_dim: linear size of the padded rank `comp_rank` projector.
  """
  proj_rank = int(proj_rank)
  proj_local_dim = pops.padded_local_ncols(proj_rank)
  comp_rank = unpadded_dim - proj_rank
  comp_local_dim = pops.padded_local_ncols(comp_rank)
  if comp_rank <= 0:
    raise ValueError(f"Failure: comp_rank={comp_rank} <= 0.")
  return proj_rank, proj_local_dim, comp_rank, comp_local_dim


###############################################################################
# SPLIT SPECTRUM
###############################################################################
@functools.partial(
    misc.log_time, header="_purify", shape_argnums=(0,), block_argnums=(1,))
def _purify(self_adjoint, unpadded_dim, split_point, p_sz, precision, canonical):
  if canonical:
    projector, _, _ = purify.canonically_purify(
      self_adjoint, split_point, p_sz=p_sz, precision=precision,
      unpadded_dim=unpadded_dim)
    proj_rank = split_point
  else:
    projector, _, _, _ = purify.grand_canonically_purify(
      self_adjoint, unpadded_dim, split_point, p_sz, precision)
    proj_rank = jnp.round(trace_f(projector)).astype(jnp.int32)
  return proj_rank, projector


@functools.partial(
    misc.log_time,
    header="Split Spectrum",
    shape_argnums=(0,),
    block_argnums=(0, 0))
def split_spectrum(self_adjoint: DistlaCoreMatrix,
                   unpadded_dim: int,
                   split_point,
                   canonical=False,
                   prior_isometry=None,
                   p_sz=1024,
                   precision=lax.Precision.HIGHEST):
  """Computes projections of the matrix H into the eigenspaces sharing its
  smallest `k` and greatest `N - k` eigenvalues respectively. Returns the
  projected matrices along with the isometries that accomplished the
  projections.

  This function should *not* be pmapped.
  Args:
    self_adjoint: The matrix whose spectrum to split.
    unpadded_dim: The total (global) number of columns in H excluding padding.
    split_point: If canonical=False, the split will be made around the value
      `split_point`. If canonical=True, the split will be made around the
      `split_point`'th eigenvalue.
    canonical: Flags whether canonical or grand canonical purification is
      used.
    prior_isometry: None, or the isometry from the original `self_adjoint`
      to this one.
    p_sz: SUMMA panel size.
    precision: Matmul precision.
  Returns:
    split_m, split_p: Tuples containing data relevant to the splits
      beneath and above `split_point` respectively:
        self_adjoint_i: The projection of `H` into the subspace.
        isometry_i: `prior_isometry` times the isometry from `self_adjoint`
          to `self_adjoint_i`.
        rank_i: Rank of `isometry_i`.
        info_i: Stores convergence information.
  """
  proj_rank, projector = _purify(self_adjoint, unpadded_dim, split_point, p_sz,
                                 precision, canonical)

  proj_rank, proj_local_dim, comp_rank, comp_local_dim = _padded_ranks(
    unpadded_dim, proj_rank)

  project_out = _project_H(
    projector, self_adjoint, proj_rank, proj_local_dim, comp_rank, comp_local_dim,
    prior_isometry, p_sz, precision)
  herm_m, iso_m, info_m, herm_p, iso_p, info_p = project_out
  split_m = (herm_m, iso_m, proj_rank, info_m)
  split_p = (herm_p, iso_p, comp_rank, info_p)
  return split_m, split_p


###############################################################################
# MAIN WORK FUNCTIONS
###############################################################################
@functools.partial(
    misc.log_time, header="_project_H", shape_argnums=(0,), block_argnums=(0,))
@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(2, 3, 4, 5, 7, 8),
    out_axes=(0, 0, None, 0, 0, None))
def _project_H(projector: DistlaCoreMatrix, self_adjoint: DistlaCoreMatrix,
               proj_rank: int, proj_local_dim: int, comp_rank: int,
               comp_local_dim: int, prior_isometry: DistlaCoreMatrix, p_sz: int,
               precision):
  """ Projects the Hermitian matrix `self_adjoint` into the columns and null
  spaces of the Hermitian projector `projector`.

  This function is already pmapped.

  Args:
    projector: The Hermitian projector.
    self_adjoint: Matrix to be projected.
    proj_rank: Rank of `projector` (its trace, rounded down).
    proj_local_dim: The number of columns, including padding, which
      will be allocated to the isometry into `projector`'s column space.
    comp_rank: Rank of `projector`'s orthogonal complement
      (its logical dimension minus proj_rank).
    comp_local_dim: The number of columns, including padding, which
      will be allocated to the isometry into `projector`'s null space.
    prior_isometry: An optional isometry. If provided, it will be multiplied by
      those mapping `self_adjoint` into the `projector` and `complement` subspaces
      , thus yielding isometries from some other space (e.g. that of the
      original matrix being diagonalized) into these.
    p_sz: panel size of the SUMMA multiplications.
    precision: Matmul precision.
  Returns:
    self_adjoint_c: Projection of `self_adjoint` into `projector`'s column space.
    isometry_c: `prior_isometry` times the isometry mapping `self_adjoint` into
      `projector`'s column space.
    info_c: Diagnostic info from the subspace iterations for `isometry_c`.
    self_adjoint_n: Projection of `self_adjoint` into `projector`'s null space.
    isometry_n: `prior_isometry` times the isometry mapping `self_adjoint` into
      `projector`'s null space.
    info_n: Diagnostic info from the subspace iterations for `isometry_n`.
  """
  proj_bar = -pops.add_to_diagonal(projector, -1.0)  # P_bar = 1 - P
  isometry_c, info_c = purify.subspace(
    projector, proj_rank, proj_local_dim, p_sz, precision=precision)
  isometry_n, info_n = purify.subspace(
    proj_bar, comp_rank, comp_local_dim, p_sz, precision=precision)
  self_adjoint_c = _similarity_transform(
    self_adjoint, isometry_c, p_sz, precision=precision)
  self_adjoint_n = _similarity_transform(
    self_adjoint, isometry_n, p_sz, precision=precision)
  if prior_isometry is not None:
    isometry_c = summa.summa(
      prior_isometry, isometry_c, p_sz, False, False, precision=precision)
    isometry_n = summa.summa(
      prior_isometry, isometry_n, p_sz, False, False, precision=precision)
  return self_adjoint_c, isometry_c, info_c, self_adjoint_n, isometry_n, info_n


@functools.partial(misc.log_time, header="_eigh_work", shape_argnums=(0,))
def _eigh_work(H: DistlaCoreMatrix, V: DistlaCoreMatrix, unpadded_dim: int, info,
               p_sz, minimum_rank, precision, verbose, canonical):
  """ Internal function continuing the D&C recursion from the second step
  until the point at which each eigenblock has rank <= `p_sz`. This function
  requires the additional specification of a `split_spectrum_function`,
  mapping a Hermitian matrix `H` and an optional isometry `V` to a pair
  of Hermitian matrices and isometries collectively spanning the original
  eigenspace.

  This function should *not* be pmapped.
  Args:
    H: The Hermitian input.
    V: None, or an isometry from the original `H` to the current one.
    unpadded_dim: Linear size of `H` excluding padding.
    info: None, or concatenated convergence data from previous iterations.
    p_sz: Panel size for the SUMMA calls.
    minimum_rank: The recursion terminates once `unpadded_dim <= minimum_rank`.
    precision: Working precision.
    verbose: Various debugging information is printed to terminal if True.

  This function's return type differs depending on the stage of the recursion.

  At the intermediate stages, returns:
    H: A logically `k[0] x k[0]` or, with padding included,
       `(k[1] * NROWS) x (k[1] * NCOLS)` matrix containing the `k` largest
       eigenvalues of the input `H`.
    V: A logically `N x k[0]` or, with padding included,
       `N x (k[1] * NCOLS)` isometry from the original `H` to the new `H`.
    unpadded_dim: Size of the unpadded top-left block of `H`.

  At the final stages, returns:
    Hlist: A list of `(p_sz, p_sz)` Hermitian matrices representing separate
           eigenspaces of `H`.
    Vlist: A list of `(N, p_sz)` isometries mapping from H into those spaces.
    klist: A list of integers specifying the global unpadded dimensions of the
           entries to `Hlist` and `Vlist`.
  """
  if unpadded_dim <= minimum_rank:
    Hlist = [H, ]
    Vlist = [V, ]
    unpadded_dim_list = [unpadded_dim, ]
    info_list = [info, ]
    split = [Hlist, Vlist, unpadded_dim_list, info_list]
  else:
    if canonical:
      split_point = unpadded_dim // 2
    else:
      split_point = trace_f(H) / unpadded_dim
    split_m, split_p = split_spectrum(
      H, unpadded_dim, split_point, canonical=canonical, prior_isometry=V,
      p_sz=p_sz, precision=precision)
    if verbose:
      _print_debug_info(*split_m, *split_p)

    split_m = _eigh_work(
      *split_m, p_sz, minimum_rank, precision, verbose, canonical)
    # Note: not lists until the final recursion
    split_p = _eigh_work(
      *split_p, p_sz, minimum_rank, precision, verbose, canonical)

    # It is assumed here that the recursion has first reached the other
    # branch of the conditional, so that the input data are all lists.
    if not isinstance(split_m[0], list):
      raise TypeError("split_m[0] was not a list, indicating the"
                      " recursion ended earlier than expected.")
    split = [m + p for m, p in zip(split_m, split_p)]
  return split


###############################################################################
# FINAL STEPS
###############################################################################
@functools.partial(
    misc.log_time,
    header="_gather_blocks",
    shape_argnums=(0, 0),
    block_argnums=(0, -1))
def _gather_blocks(Hlist, unpadded_dim_list: int) -> jnp.array:
  """ All_gathers each Hi and removes the padding.
  """
  Hlist = [_gather_block(Hi, ki) for Hi, ki in zip(Hlist, unpadded_dim_list)]
  return Hlist


@functools.partial(pops.pmap, static_broadcasted_argnums=(1,), out_axes=None)
def _gather_block(Hi, ki):
  Hi = jax.lax.all_gather(Hi, axis_name=pops.AXIS_NAME)
  Hi = pops.undistribute(Hi, host_local=False)
  Hi = Hi[:ki, :ki]
  return Hi


@functools.partial(misc.log_time, header="_compute_final_eigenblocks")
def _compute_final_eigenblocks(Hlist, unpadded_dim_list, precision):
  """ Gather each Hi to all processors. Compute the final
  eigendecompositions.
  """
  Hlist = _gather_blocks(Hlist, unpadded_dim_list)
  evlist = []
  Vlist_block = []
  for Hi in Hlist:
    ev, eV_block = serial_eigh.eigh(Hi, precision=precision)
    evlist.append(ev)
    Vlist_block.append(eV_block)
  return evlist, Vlist_block


@functools.partial(
    misc.log_time, header="_compute_final_eigenvectors", block_argnums=(0,))
def _compute_final_eigenvectors(Vlist, Vlist_block, precision, padded_dim):
  """ Multiply each Vlist[i] by Vlist_block[i] to get a block of
  eigenvectors.
  """
  eV = utils.zeros(
    (padded_dim, padded_dim), pops.GRID, dtype=Vlist[0].dtype)
  j = 0
  for Vi, Vi_block in zip(Vlist, Vlist_block):
    n_cols = Vi_block.shape[1]
    Vi = _big_to_thin_f(Vi, n_cols)
    Vi = _vecsmall_f(Vi, Vi_block, precision)
    eV = _set_columns_f(eV, Vi, j)
    j += n_cols
  return eV


@functools.partial(misc.log_time, header="_sort_blocks")
def _sort_blocks(evlist, Vlist_block, Vlist):
  """ Sorts the blocks by ascending evlist. The former is
  returned as a concatenated Jax array.
  """
  first_evs = [ev[0] for ev in evlist]
  _, evlist, Vlist_block, Vlist = zip(*sorted(
    zip(first_evs, evlist, Vlist_block, Vlist), key=lambda x: x[0]))
  return jnp.hstack(evlist), Vlist_block, Vlist


@functools.partial(misc.log_time, header="_finalize")
def _finalize(Hlist: list, Vlist: list, unpadded_dim_list: list,
              padded_dim: int, precision) -> Tuple[jnp.array, DistlaCoreMatrix]:
  """ Performs the final steps of the D&C recursion, computing the eigenpairs
  of the original input matrix from the lists of reduced blocks
  computed by _eigh_list.

  To achieve this:
    i) Each entry of Hlist is all_gathered to each core, and the padding is
       removed (`_gather_block`).
    ii) Each core computes the eigendecomposition of each block, using first
        the serial implementation of spectral D&C, and finally the Jax eigh
        implementation.

        Presently the numpy eigh is used, and thus each computation
        involves a host-device round trip. The need for this will be eliminated
        by an upcoming change to Jax.
    iii) The rows of each entry of Vlist are gathered and the padding removed.
         Each entry is multiplied by the corresponding eigenvectors computed in
         step ii. This is repeated until each core stores a copy of all rows
         (the columns remain distributed). A future implementation will
         hopefully avoid gathering the rows.
    iv) The columns are scattered.

  Args:
    Hlist: A list of (p_sz, p_sz) Hermitian matrices representing separate
           eigenspaces of H.
    Vlist: A list of (N, p_sz) isometries mapping from H into those spaces.
    unpadded_dim_list : A list of integers specifying the global unpadded
      dimensions (ranks) of the entries to Hlist and Vlist.
    padded_dim : Global linear size of the original input matrix, including
      any padding apart from that introduced by the eigensolver itself.
    precision: Working floating point precision.
  Returns:
    evs: The N unsorted eigenvalues of the original input, a
         ReplicatedThinMatrix.
    V  : The N x N matrix of eigenvectors corresponding to evs, distributed
         across cores in the same fashion as was the original input.
         V[:, i] (referring to the distributed matrix) is the eigenvector
         corresponding to evs[i] (referring to the i'th entry of each local
         copy).
  """

  if Vlist[0] is None:
    # Covers the minimum_rank > unpadded_dim case.
    H = _gather_block(Hlist[0], unpadded_dim_list[0])
    ev, eV = serial_eigh.eigh(H, precision=precision)
    ev = _pad_to(ev, padded_dim)
    eV = _pad_to(eV, padded_dim)
    eV = pops.distribute_global(eV)
    return ev, eV

  evlist, Vlist_block = _compute_final_eigenblocks(
    Hlist, unpadded_dim_list, precision)
  ev, Vlist_block, Vlist = _sort_blocks(evlist, Vlist_block, Vlist)
  eV = _compute_final_eigenvectors(Vlist, Vlist_block, precision, padded_dim)
  ev = _pad_to(ev, padded_dim)
  return ev, eV


##############################################################################
# USER-FACING FUNCTIONS
##############################################################################
def _print_debug_info(H1, V1, unpadded_dim1, info1, H2, V2, unpadded_dim2,
                      info2):
  H1_np = pops.undistribute(H1, collect_to_host=True)
  npad_1 = H1_np.shape[0] - unpadded_dim1
  H1_np = H1_np[:unpadded_dim1, :unpadded_dim1]
  H2_np = pops.undistribute(H2, collect_to_host=True)
  npad_2 = H2_np.shape[0] - unpadded_dim2
  H2_np = H2_np[:unpadded_dim2, :unpadded_dim2]

  V1_np = pops.undistribute(V1, collect_to_host=True)[:, :unpadded_dim1]
  V1_iso = np.dot(V1_np.conj().T, V1_np)
  V1_iso_err = np.linalg.norm(V1_iso - np.eye(*(V1_iso.shape)))
  V2_np = pops.undistribute(V2, collect_to_host=True)[:, :unpadded_dim2]
  V2_iso = np.dot(V2_np.conj().T, V2_np)
  V2_iso_err = np.linalg.norm(V2_iso - np.eye(*(V2_iso.shape)))

  np_evs_1 = np.linalg.eigvalsh(H1_np)
  H1_herm = np.linalg.norm(H1_np - H1_np.conj().T)
  H2_herm = np.linalg.norm(H2_np - H2_np.conj().T)
  np_evs_2 = np.linalg.eigvalsh(H2_np)
  print("*****")
  print("unpadded_dim1 : ", unpadded_dim1)
  print("npad_1: ", npad_1)
  print("H1 herm:", H1_herm)
  print("V1_iso_err: ", V1_iso_err)
  print("H1 evs:", np_evs_1)
  print("*")
  print("unpadded_dim2 : ", unpadded_dim2)
  print("npad_2: ", npad_2)
  print("H2 herm:", H2_herm)
  print("V2_iso_err: ", V2_iso_err)
  print("H2 evs:", np_evs_2)


@functools.partial(
    misc.log_time, header="eigh", shape_argnums=(0,), block_argnums=(1,))
def eigh(H,
         p_sz=1024,
         minimum_rank=128,
         precision=lax.Precision.HIGHEST,
         canonical=False,
         verbose=False,
         unpadded_dim=None) -> Tuple[jnp.array, DistlaCoreMatrix]:
  """
  Computes the eigendecomposition of the symmetric/Hermitian matrix H.

  This function should *not* be pmapped.

  Args:
    H: The Hermitian input. Hermiticity is not enforced.
    p_sz: Panel size of the SUMMA multiplications. The local dimensions of H
          must be divisible by this number.
    minimum_rank: The recursion terminates once `unpadded_dim <= minimum_rank`.
    precision: ASIC matmul precision.
    canonical: Flags whether canonical or grand canonical purification is
      used.
    verbose: Triggers (expensive) diagnostic output if True.
    unpadded_dim: If specified the matrix is assumed to be stored within
      H[:unpadded_dim, :unpadded_dim.]
  Returns:
    evals: The N unsorted eigenvalues of the original input, replicated across
      processors.
    evecs: The N x N matrix of eigenvectors corresponding to evs, distributed
         across cores in the same fashion as was the original input.
         V[:, i] (referring to the distributed matrix) is the eigenvector
         corresponding to evs[i] (referring to the i'th entry of each
         local copy).
  """
  padded_dim = H.shape[-1] * pops.NCOLS
  if unpadded_dim is None:
    unpadded_dim = padded_dim
    n_rows = H.shape[1] * pops.NROWS
    if n_rows != padded_dim:
      raise TypeError(f"Input of shape {H.shape}; {n_rows, padded_dim} must be "
                      "square.")

  Hlist, Vlist, unpadded_dim_list, info_list = _eigh_work(
    H, None, unpadded_dim, None, p_sz, minimum_rank, precision, verbose,
    canonical)

  evals, evecs = _finalize(
    Hlist, Vlist, unpadded_dim_list, padded_dim, precision)
  return evals, evecs


@functools.partial(
    misc.log_time, header="svd", shape_argnums=(0,), block_argnums=(0,))
def svd(A: DistlaCoreMatrix,
        p_sz=1024,
        minimum_rank=128,
        precision=lax.Precision.HIGHEST,
        canonical=False,
        unpadded_dim=None) -> Tuple[DistlaCoreMatrix, jnp.array, DistlaCoreMatrix]:
  """ Computes the SVD of the input matrix A.

  This function should not be pmapped.

  Args:
    A: The input matrix.
    p_sz: Panel size of the SUMMA multiplications. The local dimensions of H
          must be divisible by this number.
    precision: Matmul precision.
    minimum_rank: The recursion terminates once `unpadded_dim <= minimum_rank`.
    unpadded_dim: If specified the matrix is assumed to be stored within
  Returns:
    U, S, V: Such that A = (U * S) @ V.conj().T
      U and V are DistlaCoreMatrices distributed in the same manner as A.
  """
  if len(A.shape) != 3:
    raise TypeError(f"A.shape={A.shape}; did you forget to distribute first?")
  global_rows = A.shape[1] * pops.NROWS
  global_cols = A.shape[2] * pops.NCOLS
  if global_cols > global_rows:
    raise NotImplementedError("Fat case not yet implemented. Your matrix had "
                              f"global shape = ({global_rows, global_cols}).")
  Up, H = polar_f(A, p_sz, precision)
  S, V = eigh(
    H, p_sz, precision=precision, canonical=canonical,
    minimum_rank=minimum_rank, unpadded_dim=unpadded_dim)
  U = summa_f(Up, V, p_sz, precision)
  return U, S, V


@functools.partial(
    misc.log_time,
    header="matrix_function",
    shape_argnums=(1,),
    block_argnums=(0,))
def matrix_function(function,
                    H,
                    *function_args,
                    p_sz=1024,
                    minimum_rank=128,
                    precision=lax.Precision.HIGHEST,
                    canonical=False,
                    unpadded_dim=None,
                    verbose=False,
                    **function_kwargs) -> DistlaCoreMatrix:
  """ Applies the scalar function `function(x)` to the Hermitian matrix `H`;
  returns, that is, `V @ diag(function(s_i) ... ) @ V^H`  where `V[:, i]` is
  the eigenvector of `H` with eigenvalue `s_i`. Don't pmap this.

  Args:
    function: A function
      `function(evals[:unpadded_dim], *function_args, **function_kwargs)`
      to be applied to the eigenvalues (the Jax array) of `H`.
    H: The Hermitian input. Hermiticity is not enforced.
    p_sz: Panel size of the SUMMA multiplications. The local dimensions of H
          must be divisible by this number.
    minimum_rank: The recursion terminates once `unpadded_dim <= minimum_rank`.
    precision: ASIC matmul precision.
    canonical: Flags whether canonical or grand canonical purification is
      used.
    unpadded_dim: If specified, the matrix specified by `H` is assumed to lie
      logically within H[:unpadded_dim, :unpadded_dim].
    verbose: Triggers (expensive) diagnostic output if True.
  Returns:
    result: The matrix `V @ diag(function(s_i) ... ) @ V^H`.
    mapped_evals: The eigenvalues of that matrix.
    other_output: None, or any additional output of `function`.
  """
  evals, evecs = eigh(
    H, p_sz=p_sz, minimum_rank=minimum_rank,
    precision=precision, canonical=canonical, verbose=verbose,
    unpadded_dim=unpadded_dim)

  try:
    out = function(evals[:unpadded_dim], *function_args,
                   **function_kwargs)
    if isinstance(out, tuple):
      mapped_evals = out[0]
      other_output = out[1:]
    else:
      mapped_evals = out
      other_output = None
  except TypeError:
    print("TypeError was raised during the evaluation of your matrix function."
          "\nThis may indicate that you attempted to pass invalid positional "
          "or keyword arguments to eigh.matrix_function (which were in turn "
          "passed to the function).")
    raise

  mapped_evals_padded = _pad_to(
    mapped_evals.ravel(), evals.size).reshape((evals.size, 1))
  mapped_evals_padded = vops.distribute(mapped_evals_padded)
  result = pops.pmap(vops.diagmult)(evecs, mapped_evals_padded)
  result = summa_f(
    result, evecs.conj(), p_sz, precision=precision, transpose_B=True)
  return result, mapped_evals, other_output


def _fermi_broadening(evals, fermi_level, width):
  """ 1 / (1 + exp[(e_l - e_F) / width])
  """
  arg = (evals - fermi_level) / (2. * width)
  return 0.5 * (1.0 + np.tanh(-arg))


def _canonical_broadening(
    evals, n_occupied, occupancy_function, *args, **kwargs):
  """ Solves
  `sum(occupancy_function(evals, fermi_level, *args, **kwargs)) = n_occupied`
  for `fermi_level`. Returns fermi_level and the application of the
  corresponding broadening function to the given eigenvales.
  """
  evals = np.array(evals)
  min_eval = np.amin(evals)
  max_eval = np.amax(evals)
  bracket = [min_eval, max_eval]

  def _occupancy_errf(fermi_level):
    mapped = occupancy_function(evals, fermi_level, *args, **kwargs)
    return np.sum(mapped) - n_occupied

  root_result = scipy.optimize.root_scalar(
    _occupancy_errf, bracket=bracket)
  if not root_result.converged:
    raise ValueError(root_result.flag)
  fermi_level = root_result.root
  broadened_evals = occupancy_function(evals, fermi_level, *args, **kwargs)
  return fermi_level, broadened_evals


@functools.partial(
    misc.log_time,
    header="fermi_broadened_density",
    shape_argnums=(0,),
    block_argnums=(0,))
def fermi_broadened_density(H,
                            n_occupied,
                            width,
                            p_sz=1024,
                            minimum_rank=128,
                            precision=lax.Precision.HIGHEST,
                            canonical=False,
                            unpadded_dim=None,
                            verbose=False):
  """ Returns the Fermi-broadened density matrix P corresponding to the
  Hermitian H, P = V @ f(S) @ V^H where V are the eigenvectors of H,
  S the eigenvalues, and f(S) = 1 / (1 + exp[(e_l - e_F) / width]).
  e_f is chosen such that sum(f(S)) = n_occupied. Don't pmap this.

  Args:
    H: The Hermitian input. Hermiticity is not enforced.
    n_occupied, width: Parameters to the Fermi broadening.
    p_sz: Panel size of the SUMMA multiplications. The local dimensions of H
          must be divisible by this number.
    minimum_rank: The recursion terminates once `unpadded_dim <= minimum_rank`.
    precision: ASIC matmul precision.
    canonical: Flags whether canonical or grand canonical purification is
      used.
    unpadded_dim: If specified, the matrix specified by `H` is assumed to lie
      logically within H[:unpadded_dim, :unpadded_dim].
    verbose: Triggers (expensive) diagnostic output if True.
  Returns:
    density: The Fermi-broadened density matrix.
    mapped_evals: The eigenvalues of that matrix.
    fermi_level: The Fermi level.
  """

  def fermi_broaden(evals):
    fermi_level, broadened = _canonical_broadening(
      evals, n_occupied, _fermi_broadening, width)
    return broadened, fermi_level

  density, mapped_evals, fermi_level = matrix_function(
    fermi_broaden, H, p_sz=p_sz, minimum_rank=minimum_rank,
    precision=precision, canonical=canonical, unpadded_dim=unpadded_dim,
    verbose=verbose)
  return density, mapped_evals, fermi_level
