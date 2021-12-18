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
"""Contains functions implementing the SUMMA distributed matrix multiplication
algorithm across ASIC slices.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from distla_core.utils import misc
from distla_core.utils import pops


def _adjust_psz(p_sz, local_dims):
  """
  Replaces a value of `p_sz` incompatible with the given distribution pattern
  with one that is. `local_dims` stores the per-processor sizes of the
  shared dimensions in the multiplication. `p_sz` is modified
  according to the following rules:
    -If `p_sz > min(local_dims)`, `p_sz = min(local_dims)`.
    -If `p_sz % min(local_dims) != 0`, it is replaced with the next smallest
     integer that does.

  Args:
    p_sz: The user-specified value of `p_sz`.
    local_dims: A tuple (local_rows, local_cols) storing the per-processor
                sizes of the row and column dimensions respectively.
  Returns:
    The adjusted `p_sz`.
  """
  smallest_local_dim = min(local_dims)
  if p_sz > smallest_local_dim:
    p_sz = smallest_local_dim
  else:
    p_sz += misc.distance_to_next_divisor(p_sz, smallest_local_dim)
  return p_sz


def _panel_indices(p_idx: int, p_sz: int, row_size: int, col_size: int):
  """
  Finds the starting index of the current row and column panel for SUMMA.

  Args:
    p_idx: Loop iteration number.
    p_sz: Size of the panels.
    row_size: Local dimension from which row panels are taken.
    col_size: Local dimension from which column panels are taken.
  Returns:
    p_row: Processor row index of the next row panel.
    b_row: Row index of the next row panel within local memory.
    p_col: Processor col index of the next col panel.
    b_col: Row index of the next col panel within local memory.
  """
  s = p_idx * p_sz
  p_row = s // row_size
  b_row = s % row_size
  p_col = s // col_size
  b_col = s % col_size
  return p_row, b_row, p_col, b_col


def _broadcast_row_panel(M, p_sz, p_row, b_row):
  """
  Takes a slice of p_sz rows starting from b_row within the p_row'th
  processor row of processor grid `grid`, and broadcasts it to all other
  processor rows.
  """
  panel = jax.lax.dynamic_slice(M, (b_row, 0), (p_sz, M.shape[1]))
  return pops.broadcast_prow(panel, p_row)


def _broadcast_col_panel(M, p_sz, p_col, b_col):
  """
  Takes a slice of p_sz cols starting from b_col within the p_col'th
  processor col of processor grid `grid`, and broadcasts it to all other
  processor cols.
  """
  panel = jax.lax.dynamic_slice(M, (0, b_col), (M.shape[0], p_sz))
  return pops.broadcast_pcol(panel, p_col)


def _updateC(C, summed_updates, mask_cond, panel_idx):
  """
  A panel of `C` shaped as `summed_updates` starting from `panel_idx` is
  considered. If `mask_cond` is True (False) that panel will be replaced with
  `summed_updates` (will be left as is). Returns a copy of `C` with the
  appropriate panel either replaced or not as specified.
  """
  C_panel = jax.lax.dynamic_slice(C, panel_idx, summed_updates.shape)
  keep = jnp.zeros_like(C_panel, dtype=np.bool) + mask_cond
  updates = jnp.where(keep, x=summed_updates, y=C_panel)
  return jax.lax.dynamic_update_slice(C, updates, panel_idx)


def _summaNT(A, B, p_sz: int, precision, ef57_paneling):
  """
  Performs the SUMMA matrix multiplication algorithm for `C = A @ B.T`.
  """
  r, c = pops.GRID
  mA, kA = A.shape
  mB, kB = B.shape
  mB_global = mB * r
  nC = mB_global // c
  p_sz = _adjust_psz(p_sz, (mB, nC))
  if kA != kB:
    raise TypeError(f"A ({A.shape}) and B({B.shape}) had bad shared dimension.")

  if (mB * r) % c != 0:
    raise ValueError(
        f"mB={mB} was not evenly transposed over grid={pops.GRID}.")

  C = jnp.zeros((mA, nC), dtype=A.dtype)

  def _summa_work(p_idx, C):
    p_row, b_row, p_col, b_col = _panel_indices(p_idx, p_sz, mB, nC)
    B_panel = _broadcast_row_panel(B, p_sz, p_row, b_row)
    local_updates = pops.dot(
        A,
        B_panel.T,
        precision=precision,
        ef57_paneling=ef57_paneling,
    )
    summed_updates = pops.sum_over_prows(local_updates)
    mask_cond = pops.in_this_pcol(p_col)
    return _updateC(C, summed_updates, mask_cond, (0, b_col))

  N_blocks = mB_global // p_sz
  return jax.lax.fori_loop(0, N_blocks, _summa_work, C)


def _summaTN(A, B, p_sz: int, precision, ef57_paneling):
  """
  Performs the SUMMA matrix multiplication algorithm for `C = A.T @ B`.
  """
  r, c, = pops.GRID

  kA, nA = A.shape
  kB, nB = B.shape
  nA_global = nA * c
  mC = nA_global // r
  p_sz = _adjust_psz(p_sz, (mC, nA))

  if kA != kB:
    raise TypeError(f"A ({A.shape}) and B({B.shape}) had bad shared dimension.")

  if (nA * c) % r != 0:
    raise ValueError(
        f"nA={nA} was not evenly transposed over grid={pops.GRID}.")

  C = jnp.zeros((mC, nB), dtype=A.dtype)

  def _summa_work(p_idx, C):
    p_row, b_row, p_col, b_col = _panel_indices(p_idx, p_sz, mC, nA)
    A_panel = _broadcast_col_panel(A, p_sz, p_col, b_col)
    local_updates = pops.dot(
        A_panel.T,
        B,
        precision=precision,
        ef57_paneling=ef57_paneling,
    )
    summed_updates = pops.sum_over_pcols(local_updates)
    mask_cond = pops.in_this_prow(p_row)
    return _updateC(C, summed_updates, mask_cond, (b_row, 0))

  N_blocks = nA_global // p_sz
  return jax.lax.fori_loop(0, N_blocks, _summa_work, C)


def _summaNN(A, B, p_sz: int, precision, ef57_paneling):
  """
  Performs the SUMMA matrix multiplication algorithm for `C = A @ B`.
  """
  r, c = pops.GRID
  mA, kA = A.shape
  kB, nB = B.shape
  k_matrix = kA * c
  kB_matrix = kB * r
  if k_matrix != kB_matrix:
    raise TypeError(f"A {A.shape} and B {B.shape} had bad shared dimension.")

  p_sz = _adjust_psz(p_sz, (kA, kB))
  C = jnp.zeros((mA, nB), dtype=A.dtype)

  def _summa_work(p_idx, C):
    p_row, b_row, p_col, b_col = _panel_indices(p_idx, p_sz, kB, kA)
    A_panel = _broadcast_col_panel(A, p_sz, p_col, b_col)
    B_panel = _broadcast_row_panel(B, p_sz, p_row, b_row)
    return C + pops.dot(
        A_panel,
        B_panel,
        precision=precision,
        ef57_paneling=ef57_paneling,
    )

  number_of_panels = k_matrix // p_sz
  return jax.lax.fori_loop(0, number_of_panels, _summa_work, C)


def summa(
    A,
    B,
    p_sz: int,
    transpose_A: bool,
    transpose_B: bool,
    precision=lax.Precision.HIGHEST,
    ef57_paneling=True,
):
  """
  summa multiplies the matrices A and B, which are contiguously distributed
  across ASIC cores. Depending on the choice of transpose_A and transpose_B,
  either `C = A @ B`, `C = A.T @ B`, or `C = A @ B.T` can be computed.

  Args:
    A: ShardedDeviceArray representing the matrix `A`.
    B: ShardedDeviceArray representing the matrix `B`.
    p_sz: Each device computes local matrix multiplies of shape
          `A[:, i:i+p_sz] @ B[i:i+p_sz, :]`. Increasing `p_sz` thus makes for
          fewer, larger multiplies, but also increases the memory overhead of
          each step, since the given panels need to be communicated.

          The panel size actually used in the code must evenly divide both the
          shared dimension of the matrix multiplication (e.g.
          `A.shape[1] == B.shape[0]` for `transpose_A == transpose_B == False`)
          and the per-device sizes of those dimensions. If `p_sz` does not
          fulfill these requirements, it is replaced with the next-largest
          value which does (or next-smallest, in the case that `p_sz` is
          larger than one of these per-device sizes).

          These shared dimensions are:
            -for `transpose_A = transpose_B = False`:
              `k1 = A.shape[1]`
              `k2 = B.shape[0]`
            -for `transpose_A = True`, `transpose_B = False`:
              `k1 = A.shape[1]`
              `k2 = B.shape[1]`
            -for `transpose_A = False`, `transpose_B = True`:
              `k1 = A.shape[0]`
              `k2 = B.shape[1]`

          This argument should be listed as static within a pmap.


    transpose_A, transpose_B: Setting one of these True will compute
          `A.T @ B` or `A @ B.T` instead of `A @ B`. `A.T @ B.T` is not
          supported.
    precision: Matmul precision. `HIGHEST` by default.
    ef57_paneling: Whether to use paneling for large ef57 matmuls. True by
      default. See `pops.dot` for more.

  Returns:
    C: ShardedDeviceArray representing the matrix C.
  """
  if transpose_A and transpose_B:
    raise NotImplementedError("A.T @ B.T is not supported.")
  if transpose_A:
    return _summaTN(A, B, p_sz, precision, ef57_paneling)
  if transpose_B:
    return _summaNT(A, B, p_sz, precision, ef57_paneling)
  return _summaNN(A, B, p_sz, precision, ef57_paneling)


def similarity_transform(
    inner_matrix,
    outer_matrix,
    p_sz: int,
    precision=lax.Precision.HIGHEST,
    ef57_paneling=True,
):
  """
  Computes `outer_matrix.T.conj() @ inner_matrix @ outer_matrix`.

  This function must be called within a pmap.

  Args:
    inner_matrix, outer_matrix: The checkerboard-distributed matrices of the
      transform.
    p_sz: Summa panel size.
    precision: Matmul precision. `HIGHEST` by default.
    ef57_paneling: Whether to use paneling for large ef57 matmuls. True by
      default. See `pops.dot` for more.
  Returns:
    The result.
  """
  inner_outer = summa(
      inner_matrix,
      outer_matrix,
      p_sz,
      False,
      False,
      precision=precision,
      ef57_paneling=ef57_paneling,
  )
  return summa(
      outer_matrix.conj(),
      inner_outer,
      p_sz,
      True,
      False,
      precision=precision,
      ef57_paneling=ef57_paneling,
  )
