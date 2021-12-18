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
pops contains utility functions for manipulating matrices distributed across
ASIC slices.

Functions `distribute` and `undistribute` map data stored in host memory
to a ShardedDeviceArray representing the same data distributed across all
connected ASIC cores. The result is a ShardedDeviceArray of shape
(# of processors, rows per processor, columns per processor). pmapping that
array over its first axis then assigns each core a contiguous matrix block.

A `column-major` processor distribution is assumed, with cores
treated as rows. That is, for

grid = (Ncore, Nrow, Ncol) = (2, 2, 2),
processors are arranged as follows:

0  4
1  5
2  6
3  7
"""
import collections
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

AXIS_NAME = "i"
Grid_t = Tuple[int, int, int]


################################################################################
# INITIALIZATION
################################################################################
def random(matrix_shape, grid, dtype, key_seed=0):
  Ncore, Nrow, Ncol = grid
  p = grid[0] * grid[1] * grid[2]
  keys = jax.random.split(jax.random.PRNGKey(key_seed), p)
  m_l = matrix_shape[0] // (Ncore * Nrow)
  n_l = matrix_shape[1] // Ncol
  return jax.pmap(lambda key: jax.random.normal(key, (m_l, n_l)))(keys)


def distribute(A, grid: Grid_t, pmap=True):
  """
  Takes the local array A and creates a ShardedDeviceArray corresponding to A's
  block-contiguous distribution across the processor grid `grid`. It is assumed
  that the dimensions of `A` are evenly divided by `grid`.

  If `pmap` is False, A is reshaped into the same dimensions as its distributed
  counterpart but is not actually distributed, which is useful for testing.

  This function currently assumes a single-host setup. In a multi-host setup,
  it will cause each host individually to distribute (scatter) the data
  in its host process to each connected ASIC core, and thus an additional
  initial step of distributing between hosts is required. `grid` in that case
  should refer to the processor cores connected to a single host rather than
  the full slice dimensions.

  Args:
    A: An array.
    grid: The processor grid connected to a single host.
    pmap: If False, A is reshaped but not actually distributed.
  Raises:
    TypeError: If A's dimensions are not evenly divided by `grid`.
  Returns:
    Ap: The distributed ShardedDeviceArray.
  """
  Ncore, Nrow, Ncol = grid
  p = Ncore * Nrow * Ncol
  M, N = A.shape
  if M % Nrow != 0 or N % Ncol != 0:
    raise TypeError(f"A.shape={A.shape} was not evenly divided by grid={grid}.")
  mA = M // (Nrow * Ncore)
  nA = N // Ncol
  A = A.reshape([Ncore * Nrow, mA, Ncol, nA]).transpose([2, 0, 1, 3])
  A = A.reshape([p, mA, nA])
  if pmap:
    return jax.pmap(lambda x: x)(A)
  return A


def undistribute(A, grid: Grid_t):
  """
  Reshapes `A` from its block-contiguously-distributed shape as defined by
  `grid` to its mathematical shape, and collects the result onto the host
  process. The function also works if `A` is already on the host process.
  """
  Ncore, Nrow, Ncol = grid
  p, mA, nA = A.shape
  M = mA * Nrow * Ncore
  N = nA * Ncol
  A = jax.device_put(A)
  A = A.reshape([Ncol, Nrow * Ncore, mA, nA]).transpose([1, 2, 0, 3])
  return A.reshape([M, N])


################################################################################
# PROCESSOR ADDRESSING
################################################################################
def my_name():
  """
  The pmap axis of this processor.
  Returns:
    i: The axis.
  """
  return jax.lax.axis_index(axis_name=AXIS_NAME)


def your_pcol(p, grid: Grid_t):
  """
  Returns the pcol inhabited by processor p.
  Args:
    p: The processor number.
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    pcol: The pcol of processor p.
  """
  Ncore, Nrow, Ncol = grid
  return p // (Nrow * Ncore)


def your_prow(p, grid: Grid_t):
  """
  Returns the prow inhabited by processor p.
  Args:
    p: The processor number.
    grid: Tuple of processor grid dimensions (pcols, prows).
  Returns:
    prow: The prow of processor p.
  """
  Ncore, Nrow, Ncol = grid
  return (p // Ncore) % Nrow


def your_pcore(p, grid: Grid_t):
  """
  Returns the pcore inhabited by processor p.
  Args:
    p: The processor number.
    grid: Tuple of processor grid dimensions (pcols, prows).
  Returns:
    prow: The prow of processor p.
  """
  Ncore, Nrow, Ncol = grid
  return p % Ncore


def my_pcol(grid: Grid_t):
  """
  Returns the pcol inhabited by this processor.
  Args:
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    pcol: The pcol of this processor.
  """
  return your_pcol(my_name(), grid)


def my_prow(grid: Grid_t):
  """
  Returns the prow inhabited by this processor.
  Args:
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    prow: The prow of this processor.
  """
  return your_prow(my_name(), grid)


def my_pcore(grid: Grid_t):
  """
  Returns the pcore inhabited by this processor.
  Args:
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    prow: The prow of this processor.
  """
  return your_pcore(my_name(), grid)


################################################################################
# MESSAGE TUPLES
################################################################################
def _make_shuffle_pcores(n, grid):
  """
  Generates a list of integers instructing `pshuffle` to have each processor
  communicate to nth closes core within its chip.
  Returns e.g.
    (1, 0, 3, 2, 5, 4, 7, 6)
  For n=1, grid=(2, 2, 2).
  """
  Ncore, Nrow, Ncol = grid

  shuffle_core = []
  for j in range(Ncol * Nrow):
    these_cores = range(j * Ncore, (j + 1) * Ncore)
    chip = collections.deque(these_cores)
    chip.rotate(n)
    shuffle_core += chip
  return tuple(shuffle_core)


def _make_shuffle_prows(n, grid):
  """
  Generates a list of integers instructing `pshuffle` to have each processor
  communicate n steps to the same core of its rightwards neighbour.

  Returns e.g.
    (2, 3, 0, 1, 6, 7, 4, 5)
  For n=1, grid=(2, 2, 2)
  """
  Ncore, Nrow, Ncol = grid
  out = []
  for i in range(Nrow):
    row = collections.deque(range(i * Ncol * Ncore, (i + 1) * Ncol * Ncore))
    row.rotate(n * Ncore)
    out += row
  return tuple(out)


def _make_shuffle_pcols(n, grid):
  """
  Generates a list of integers instructing `pshuffle` to have each processor
  n units downward.

  Returns e.g.
    (4, 5, 6, 7, 0, 1, 2, 3)
  For n=1, grid=(2, 2, 2), "shift hor"
  """
  Ncore, Nrow, Ncol = grid
  chips_per_pcol = Ncore * Nrow
  out = collections.deque()
  for i in range(Ncol):
    col = tuple(list(range(i * chips_per_pcol, (i + 1) * chips_per_pcol)))
    out.append(col)
  out.rotate(n)
  out = list(out)
  to_return = []
  for tup in out:
    to_return += list(tup)
  return tuple(to_return)


def _make_skew_prows(grid):
  """
  0 1 2 3 6 7 4 5, "right init"
  """
  Ncore, Nrow, Ncol = grid
  chips_per_pcol = Ncore * Nrow
  out = []
  for i in range(Ncol):
    col = collections.deque(range(i * chips_per_pcol, (i + 1) * chips_per_pcol))
    col.rotate(Ncore * i)
    out += tuple(col)
  return tuple(out)


def _make_skew_pcols(grid):
  """
  0 1 6 7 4 5 2 3, "left_init"
  """
  Ncore, Nrow, Ncol = grid
  out = []
  out.append(tuple(range(Ncore)))
  to_flip = []
  for i in range(1, Nrow * Ncol):
    chip = list(range(i * Ncore, (i + 1) * Ncore))
    to_flip.append(tuple(chip))
  out += to_flip[::-1]
  to_return = []
  for tup in out:
    to_return += list(tup)
  return tuple(to_return)


################################################################################
# COMMUNICATION
################################################################################
def cycle_pcols(A, grid, n):
  shuffle = _make_shuffle_pcols(n, grid)
  return jax.lax.pshuffle(A, AXIS_NAME, shuffle)


def cycle_pcores(A, grid, n):
  shuffle = _make_shuffle_pcores(n, grid)
  return jax.lax.pshuffle(A, AXIS_NAME, shuffle)


def cycle_prows(A, grid, n):
  shuffle = _make_shuffle_prows(n, grid)
  return jax.lax.pshuffle(A, AXIS_NAME, shuffle)


def skew_prows(A, grid):
  return jax.lax.pshuffle(A, AXIS_NAME, _make_skew_prows(grid))


def skew_pcols(A, grid):
  return jax.lax.pshuffle(A, AXIS_NAME, _make_skew_pcols(grid))


# PRECISION TYPES
# bfloat16: jax.lax.Precision.DEFAULT
# float24ish: jax.lax.Precision.HIGH
# float32: jax.lax.Precision.HIGHEST
@functools.partial(jax.jit, static_argnums=(2,))
def two_core_multiply(A, B, grid, precision=jax.lax.Precision.HIGHEST):
  k_mid = A.shape[1] // 2

  def _concat(x):
    return jnp.hstack([x[:, k_mid:], x[:, :k_mid]])

  A = jnp.where(my_pcore(grid), x=_concat(A), y=A)
  A_L = A[:, :k_mid]
  A_R = A[:, k_mid:]
  C = jnp.matmul(A_L, B, precision=precision)
  B = cycle_pcores(B, grid, 1)
  return C + jnp.matmul(A_R, B, precision=precision)


def chip_distribute(A, grid, pmap=True):
  """
  Distributes A across chips and copies it between cores.
  """
  Nprows, Npcols, Npcores = grid
  M, N = A.shape
  if M % Nprows != 0 or N % Npcols != 0:
    raise TypeError(f"A.shape={A.shape} was not evenly divided by grid={grid}.")
  A = np.stack([
      A,
  ] * Npcores)
  mA = M // Nprows
  nA = N // Npcols
  A = A.reshape([Npcores, Nprows, mA, Npcols, nA]).transpose([1, 0, 3, 2, 4])
  # prows, pcores, pcols, mA, nA
  if pmap:
    return jax.pmap(lambda x: x)(A)
  return A


def chip_undistribute(A):
  A = jax.device_put(A[:, 0, :, :, :])
  Nprows, Npcols, mA, nA = A.shape
  M = mA * Nprows
  N = nA * Npcols
  A = A.reshape([Nprows, Npcols, mA, nA]).transpose([0, 2, 1, 3])
  return A.reshape([M, N])


@functools.partial(jax.jit, static_argnums=(2,))
def cannons_NN(A, B, grid):
  Ncore, Nrow, Ncol = grid
  if Nrow != Ncol:
    raise TypeError("Only square grids supported.")
  if Ncore != 2:
    raise TypeError(f"Ncore={Ncore} must be 2.")
  A = skew_pcols(A, grid)
  B = skew_prows(B, grid)
  C = two_core_multiply(A, B, grid)
  for i in range(Nrow - 1):
    A = cycle_pcols(A, grid, -1)
    B = cycle_prows(B, grid, -1)
    C += two_core_multiply(A, B, grid)
  return C


def two_point_five(A, B):
  """
  Cannon's algorithm, with the main loop parallelized across ASIC cores.
  Args:
    A: first JAX array to be multiplied.
    B: second JAX array.

  Returns:
    A new JAX array.
  """
  skew_cols = [0, 5, 6, 3, 4, 1, 2, 7]
  skew_rows = [0, 3, 2, 1, 6, 5, 4, 7]
  A = jax.lax.pshuffle(A, perm=skew_rows)
  B = jax.lax.pshuffle(B, perm=skew_cols)
  C = jnp.matmul(A, B, precision=jax.lax.Precision.HIGHEST)
  return jax.lax.psum(C, axis_index_groups=[[0, 1], [2, 3], [4, 5], [6, 7]])
