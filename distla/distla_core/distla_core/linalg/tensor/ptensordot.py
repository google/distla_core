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
from typing import Optional, Tuple

import jax
from jax.interpreters import pxla
import numpy as np

from distla_core.blas.summa import summa
from distla_core.linalg.tensor import preshape
from distla_core.linalg.tensor import ptranspose
from distla_core.utils import config
from distla_core.utils import misc


def ptensordot(A: pxla.ShardedDeviceArray,
               B: pxla.ShardedDeviceArray,
               grid_shape_A: Tuple[int],
               grid_shape_B: Tuple[int],
               grid_shape_C: Tuple[int],
               axes: Tuple[Tuple[int]],
               p_sz: Optional[int] = None,
               shape_A: Optional[Tuple] = None,
               shape_B: Optional[Tuple] = None):
  """
  Parallel tensordot function. Contracts two tensors `A`
  and `B` along `axes`. The function currently only covers
  the case where the result of the contraction is a tensor
  of at least order 2 (i.e. at least a matrix). The shapes of
  the input tensors need to be "large enough".

  Args:
    A, B: Two distributed tensors.
    grid_shape_A: Distribution pattern for tensor `A`
    grid_shape_B: Distribution pattern for tensor `B`
    grid_shape_C: Distribution pattern for resulting
      tensor `C`.
    axes: The axes to be contracted.
    p_sz: Panel size parameter for SUMMA. Following is the
       summa docstring for p_sz:
          "Each device computes local matrix multiplies of shape
           A[:, i:i+p_sz] @ B[i:i+p_sz, :]. Increasing p_sz thus makes for
           fewer, larger multiplies, but also increases the memory overhead of
           each step, since the given panels need to be communicated.

           `p_sz` must be chosen to evenly divide the serially iterated
           dimensions of the matrix multiplication, k1 and k2.
           These shared dimensions are:
             -for transpose_A = transpose_B = False:
               k1 = A.shape[1]
               k2 = B.shape[0]
             -for transpose_A = True, transpose_B = False:
               k1 = A.shape[1]
               k2 = B.shape[1]
             -for transpose_A = False, transpose_B = True:
               k1 = A.shape[0]
               k2 = B.shape[1]"
    shape_A, shape_B: An optional global shape for `A` and `B`.
      This is necessary because some parallel operations reshape tensors
      to avoid/minimize zero-padding on ASIC. As a result, the global shape
      can then no longer be inferred from the local shape and the processor
      grid shape.

  Returns:
    ShardedDeviceArray: The result `C` of the contraction.
    Tuple[int]: The resulting processor grid shape of `C`
      (identical to `grid_shape_C`).
  """
  # TODO (mganahl): We need to cover cases where `C`
  # cannot be distributed to a processor grid, for example
  # a contraction to a scalar.

  # TODO (mganahl): Improve performance for corner cases
  # where some parallel operations could be avoided or
  # improved. For example, in certain cases some ptranspose
  # calls can be avoided by swapping the argument order to
  # summa.

  grid_shape_A = np.asarray(grid_shape_A)
  grid_shape_B = np.asarray(grid_shape_B)
  grid_shape_C = np.asarray(grid_shape_C)

  if shape_A is None:
    shape_A = misc.global_shape(A.shape, grid_shape_A)
  if shape_B is None:
    shape_B = misc.global_shape(B.shape, grid_shape_B)

  A_axes, B_axes = [list(a) for a in axes]
  A_ndim = len(grid_shape_A)
  B_ndim = len(grid_shape_B)

  # check if grid_shape_C is cpompatible with C.ndim
  C_ndim = A_ndim + B_ndim - len(A_axes) - len(B_axes)
  if C_ndim != len(grid_shape_C):
    raise ValueError(f"C.ndim = {C_ndim} is different from "
                     f"len(grid_shape_C) = {len(grid_shape_C)}")

  free_axes_A = set(range(A_ndim)) - set(A_axes)
  free_axes_B = set(range(B_ndim)) - set(B_axes)
  perm_A = sorted(free_axes_A) + A_axes
  perm_B = B_axes + sorted(free_axes_B)
  if tuple(perm_A) != tuple(range(len(perm_A))):
    A_perm = ptranspose.ptranspose(A, perm_A, grid_shape_A)
  else:
    A_perm = A
  if tuple(perm_B) != tuple(range(len(perm_B))):
    B_perm = ptranspose.ptranspose(B, perm_B, grid_shape_B)
  else:
    B_perm = B

  perm_pgrid_A = grid_shape_A[perm_A]
  perm_pgrid_B = grid_shape_B[perm_B]

  shape_A_perm = [shape_A[p] for p in perm_A]
  shape_B_perm = [shape_B[p] for p in perm_B]

  mat_shape_A = (np.prod(shape_A_perm[:len(free_axes_A)]),
                 np.prod(shape_A_perm[len(free_axes_A):]))
  mat_shape_B = (np.prod(shape_B_perm[:len(B_axes)]),
                 np.prod(shape_B_perm[len(B_axes):]))
  final_shape = shape_A_perm[:len(free_axes_A)] + shape_B_perm[len(B_axes):]
  if len(final_shape) < 2:
    raise ValueError(f"contraction resulted in a tensor of "
                     f"order {len(final_shape)}."
                     f"The current implementation requires "
                     f"the resulting tensor C"
                     f"to be of order >= 2.")
  # check if grid_shape_C is compatible with the shape of
  # the resulting tensor.
  if not np.all([s % p == 0 for s, p in zip(final_shape, grid_shape_C)]):
    raise ValueError(f"shape for the resulting tensor C.shape = {final_shape}"
                     f"is not evenly divisible by grid_shape_C = "
                     f"{grid_shape_C}")

  SUMMA_gridshape = config.GRID
  if tuple(SUMMA_gridshape) != tuple(perm_pgrid_A) or len(A_perm.shape) != 2:
    mat_A = preshape.preshape(A_perm, mat_shape_A, SUMMA_gridshape,
                              perm_pgrid_A)
  else:
    mat_A = A_perm
  if tuple(SUMMA_gridshape) != tuple(perm_pgrid_B) or len(B_perm.shape) != 2:
    mat_B = preshape.preshape(B_perm, mat_shape_B, SUMMA_gridshape,
                              perm_pgrid_B)
  else:
    mat_B = B_perm

  if p_sz is None:
    primes_A = list(misc.prime_factors(mat_A.shape[1]))
    primes_B = list(misc.prime_factors(mat_B.shape[0]))
    primes = misc.find_common(primes_A, primes_B)
    p_sz = 1
    primes_128 = list([2] * 7)
    while len(primes_128) > 0 and len(primes) > 0 and p_sz < 128:
      p = primes_128.pop()
      ind = np.argmin(np.abs(np.array(primes) - p))
      prime = primes.pop(ind)
      p_sz *= prime

  # TODO (mganahl): these three pshuffles need to go ...

  summa_perm = config.get_processor_grid().ravel()
  inv_summa_perm = misc.inverse_permutation(summa_perm)
  mat_A = jax.lax.pshuffle(mat_A, 'i', inv_summa_perm)
  mat_B = jax.lax.pshuffle(mat_B, 'i', inv_summa_perm)
  result = summa.summa(
      A=mat_A, B=mat_B, p_sz=p_sz, transpose_A=False, transpose_B=False)
  result = jax.lax.pshuffle(result, 'i', summa_perm)
  if tuple(grid_shape_C) == tuple(SUMMA_gridshape) and len(final_shape) == 2:
    return result
  return preshape.preshape(result, final_shape, grid_shape_C, SUMMA_gridshape)
