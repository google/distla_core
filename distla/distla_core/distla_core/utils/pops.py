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
"""Utility functions for distributed matrices across ASIC slices.

`pops` contains utility functions for manipulating matrices distributed across
ASIC slices.

Functions `distribute` and `undistribute` map data stored in host memory
to a ShardedDeviceArray representing the same data distributed across all
connected ASIC cores. The result is a ShardedDeviceArray of shape
(# of processors, rows per processor, columns per processor). pmapping that
array over its first axis then assigns each core a contiguous matrix block.

The processor distribution for NHROWS = NHCOLS = NDROW = NDCOLS = 2 is as
follows:

0 2  8 10
1 3  9 11

4 6  12 14
5 7  13 15
"""
import functools
from typing import Tuple, Sequence, Union
import warnings

import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numbers
import numpy as np

from distla_core.utils import misc
from distla_core.utils import config


################################################################################
# CONFIGURATION
################################################################################
NHROWS = config.NHROWS
NHCOLS = config.NHCOLS
NDROWS = config.NDROWS
NDCOLS = config.NDCOLS
AXIS_NAME = config.get_axis_name()

NROWS = config.NROWS
NCOLS = config.NCOLS
NDPROCS = config.NDPROCS
NPROCS = config.NPROCS
GRID = config.GRID
DGRID = config.DGRID
HGRID = config.HGRID

EF57_DTYPES = (jnp.float64, jnp.complex128)


################################################################################
# UTILITIES
################################################################################
def pmap(f, *args, **kwargs):
  return jax.pmap(f, *args, axis_name=AXIS_NAME, **kwargs)


def padded_local_ncols(logical_ncols):
  if logical_ncols < 1:
    raise ValueError(f"Invalid logical_ncols={logical_ncols}.")
  largest_proc_dim = max(GRID)
  pad_size = misc.distance_to_next_divisor(logical_ncols, largest_proc_dim)
  return (logical_ncols + pad_size) // NCOLS


def _get_all_to_all_axis_index_groups(grid_shape, sharded_axes):
  grid = np.arange(int(np.prod(grid_shape))).reshape(grid_shape, order='C')
  reduced_shape = [
      grid_shape[s] for s in range(len(grid_shape)) if s not in sharded_axes
  ]
  axis_index_groups = []
  for i in np.ndindex(*reduced_shape):
    slices = list(i)
    for sharded_axis in sharded_axes:
      slices.insert(sharded_axis, slice(0, grid_shape[sharded_axis]))
    axis_index_groups.append(list(np.ravel(grid[tuple(slices)])))
  return axis_index_groups


def _to_tuple(val):
  if isinstance(val, numbers.Number):
    return (val,)
  return tuple(val)


def all_to_all(
    array: pxla.ShardedDeviceArray,
    sharded_axes: Union[int, Sequence[int]],
    split_axes: Union[int, Sequence[int]],
    concat_axes: Union[int, Sequence[int]],
    grid_shape: Tuple[int],
):
  """
    Swap pmapped axes `sharded_axes` with local axes `split_axes`, and place
    them at the local positions `concat_axes`.
    The global part of `array` is considered to be of shape `grid_shape`, and
    the pmap-axis placement of each shard of the global array is in 'C' order,
    i.e. shard `i` of the array (the `i`-th element of the pmapped axis) is
    placed on position `np.ravel(grid)[i]` on the grid, with
    `grid = np.arange(jax.device_count()).reshape(grid_shape, order='C')`.
    `sharded_axis`, `split_axes` and `concat_axes` have be either ints, or
    sequences of ints of identical length.

    Note: Due to an XLA bug (https://github.com/google/jax/issues/5861) this
    function currently only works properly in ASICs.

    Args:
      array: A sharded array.
      sharded_axes: The sharded axes to be swapped with local axes.
      split_axes: The local axes to be pmapped.
      concat_axes: local axes positions where `sharded_axes`
        should be placed after localizing them.
      grid_shape: the processor grid shape.

    Returns:
      ShardedDeviceArray: The result of the operation.
    """

  def ind_sort(sequence, inds):
    return tuple([sequence[i] for i in inds])

  sharded_axes = _to_tuple(sharded_axes)
  split_axes = _to_tuple(split_axes)
  concat_axes = _to_tuple(concat_axes)

  if len(split_axes) != len(concat_axes):
    raise ValueError(f"split_axes and concat_axes are of unequal "
                     f"length {len(split_axes)} and {len(concat_axes)}.")

  if len(split_axes) != len(sharded_axes):
    raise ValueError(f"split_axes and sharded_axes are of unequal "
                     f"length {len(split_axes)} and {len(sharded_axes)}.")

  sharded_dims = np.asarray([grid_shape[a] for a in sharded_axes])
  local_dims = np.asarray([array.shape[a] for a in split_axes])
  if not np.all(sharded_dims == local_dims):
    raise ValueError(f"dimensions {sharded_dims} of global axes "
                     f"do not match dimensions {local_dims} of "
                     f"the local axes")

  # we first sort sharded_axes
  inds = np.argsort(sharded_axes)
  sharded_axes = ind_sort(sharded_axes, inds)
  split_axes = ind_sort(split_axes, inds)
  concat_axes = ind_sort(concat_axes, inds)

  axis_index_groups = _get_all_to_all_axis_index_groups(grid_shape,
                                                        sharded_axes)

  if len(split_axes) == 1:
    # this case is already covered within jax
    return jax.lax.all_to_all(
        array,
        axis_name=AXIS_NAME,
        split_axis=split_axes[0],
        concat_axis=concat_axes[0],
        axis_index_groups=axis_index_groups,
    )

  # we move all split_axes to the left side of the array
  # and combine them into a single dimension

  # transpose
  n_split = len(split_axes)
  permarray = jnp.moveaxis(array, split_axes, tuple(range(n_split)))
  # now reshape
  permshape = permarray.shape
  comb_permshape = (int(np.prod(permshape[:n_split])),) + permshape[n_split:]
  permarray = permarray.reshape(comb_permshape)

  #now we swap the local index 0 with `sharded_axes`
  result = jax.lax.all_to_all(
      permarray,
      axis_name=AXIS_NAME,
      split_axis=0,
      concat_axis=0,
      axis_index_groups=axis_index_groups,
  )

  # finally we split the swapped axes back into their original shapes
  # and move them to their final positions.
  final_shape = tuple([grid_shape[a] for a in sharded_axes
                       ]) + comb_permshape[1:]
  return jnp.moveaxis(
      result.reshape(final_shape), tuple(range(len(sharded_axes))), concat_axes)


def distribution_type(array):
  """Returns the distribution pattern of a matrix.

  The possible values are `"undistributed"` if it is a Jax or numpy array with
  two indices; `"distributed"` if it is a `ShardedDeviceArray` with three
  indices; and `"traced"` if it is a `DynamicJaxprTracer` with two indices.

  Args:
    array: The matrix.
  Raises:
    TypeError: If the distribution pattern is not one of the above.
  Returns:
    The distribution pattern of `array`.
  """
  ndim = array.ndim
  RegularArray = (jnp.DeviceArray, np.ndarray)
  ShardedDeviceArray = jax.interpreters.pxla.ShardedDeviceArray
  DynamicTracer = jax.interpreters.partial_eval.DynamicJaxprTracer
  if isinstance(array, ShardedDeviceArray) and ndim == 3:
    return "distributed"
  if isinstance(array, DynamicTracer) and ndim == 2:
    return "traced"
  if isinstance(array, RegularArray) and ndim == 2:
    return "undistributed"
  msg = ("Don't know how to interpret the distribution pattern of a matrix of "
         f"type {type(array)} with {ndim} indices.")
  raise TypeError(msg)


###############################################################################
# SLICING
###############################################################################
def _in_range(rows, row_start, n_rows):
  return jnp.logical_and(rows >= row_start, rows < row_start + n_rows)


def get_rows(matrix, row_start, n_rows, rows=None):
  """ Extracts matrix[row_start:row_start + n_rows, :] from each processor
  column. The result is replicated across processor columns.
  Presently, n_rows must be less than the local
  number of rows per panel, but this could probably be relaxed with
  some effort.

  Args:
    matrix: ShardedDeviceArray to take from.
    row_start: First row to take.
    n_rows: Number of rows to take.
    rows: Optionally previously-computed row indices.
  Returns:
    The n_rows x matrix.shape[1]: panel, replicated across processor columns.
  """
  m_l, _ = matrix.shape
  if rows is None:
    rows, _ = indices(matrix.shape)
  if n_rows > m_l:
    raise TypeError(
      f"Cannot extract more rows {n_rows} than local rows {m_l}.")
  row_start = jnp.full(1, row_start)
  prow_start = row_start // m_l
  prow_finish = (row_start + n_rows - 1) // m_l
  two_prows = jnp.all(prow_start != prow_finish)
  return jax.lax.cond(two_prows,
                      lambda x: _get_rows_two_prows(n_rows, x),
                      lambda x: _get_rows_single_prow(n_rows, x),
                      (matrix, row_start, rows))


def _get_rows_single_prow(n_rows, args):
  """ Handles the simple case that the row panel lies within a prow.
  We take the needed slice (whose size is known in advance)
  and broadcast it.
  """
  matrix, row_start, _ = args
  m_l, n = matrix.shape
  prow, brow = divmod(row_start, m_l)
  panel = jax.lax.dynamic_slice(matrix, (brow[0], 0), (n_rows, n))
  return broadcast_prow(panel, prow)


def _get_rows_two_prows(n_rows, args):
  """ Handles the trickier case that the row panel straddles two prows.
  The size of the local slices to extract is not known at compile time in
  this case.

  All of the elements are either in the "top", matrix[:n_rows, :],
  or the "bottom", matrix[-n_rows:, :], which do have known sizes.
  We extract these panels, mask all elements not in the desired extraction,
  and then cyclically permute the elements upwards so that the extracted
  rows are lined up with their location in the extraction. We then sum
  the masked data, both locally and within pcols.
  """
  matrix, row_start, rows = args
  m_l, n = matrix.shape
  top_offset = row_start % m_l  # Local index of first row

  top_blocks = matrix[:n_rows, :]
  top_rows = rows[:n_rows, :]
  good_top_rows = _in_range(top_rows, row_start, n_rows)
  good_top_blocks = mask(top_blocks, good_top_rows)
  good_top_blocks = jnp.roll(good_top_blocks, -top_offset, axis=0)

  bottom_blocks = matrix[-n_rows:, :]
  bottom_rows = rows[-n_rows:, :]
  good_bottom_rows = _in_range(bottom_rows, row_start, n_rows)
  good_bottom_rows = jnp.logical_and(
    good_bottom_rows, bottom_rows != top_rows)
  good_bottom_blocks = mask(bottom_blocks, good_bottom_rows)
  good_bottom_blocks = jnp.roll(good_bottom_blocks, -top_offset, axis=0)
  panel = good_top_blocks + good_bottom_blocks
  return sum_over_pcols(panel)


def set_rows(matrix, panel, row_start, rows=None):
  """ Inserts the (n_rows, n_l) panel to the locally
  (m_l, n_l) matrix, logically as
  matrix[row_start:row_start + n_rows, :]. It is assumed the panel data
  is replicated over processor columns, so that n_rows is the mathematical
  size of the panel.
  Presently, we must have n_rows <= m_l.

  Args:
    matrix: ShardedDeviceArray to take from.
    panel: Panel to insert.
    row_start: Slice offset.
    rows: Optionally previously-computed row indices of matrix.
  Returns:
    The matrix with the panel inserted.
  """
  m_l, n_l = matrix.shape
  n_rows, n_l_p = panel.shape
  if n_l_p != n_l:
    raise TypeError("Incompatible shapes {matrix.shape}, {panel.shape}.")
  if n_rows > m_l:
    raise TypeError(
      f"Cannot insert more rows {n_rows} than local rows {m_l}.")
  if rows is None:
    rows, _ = indices(matrix.shape)
  prow_start = row_start // m_l
  prow_finish = (row_start + n_rows - 1) // m_l
  two_prows = jnp.all(prow_start != prow_finish)
  result = jax.lax.cond(two_prows,
                        _set_rows_two_prows,
                        _set_rows_single_prow,
                        (matrix, panel, row_start, rows))
  return result


def _set_rows_single_prow(args):
  matrix, panel, row_start, rows = args
  n_rows = panel.shape[0]
  panel_masked = mask(
    matrix, jnp.logical_not(_in_range(rows, row_start, n_rows)))
  brow = row_start % matrix.shape[0]
  update = jax.lax.dynamic_update_slice(matrix, panel, (brow, 0))
  update = mask(update, _in_range(rows, row_start, n_rows))
  return panel_masked + update


def _set_rows_two_prows(args):
  matrix, panel, row_start, rows = args
  # First we pad the panel to the local size of matrix.
  n_rows = panel.shape[0]
  panel = jax.lax.dynamic_update_slice(
    jnp.zeros_like(matrix), panel, (0, 0))

  # We need to sort the rows of panel so that those to
  # be inserted are correctly aligned with matrix.
  row_idxs = jnp.arange(0, panel.shape[0]) + row_start
  row_end = n_rows + row_start

  # If this is the topmost prow containing an insertion,
  # aligning the rows requires us to bring those not being inserted to the
  # top of the block. Otherwise they must be brought to the bottom.
  # We thus replace the row_idxs outside the insertion with -1 or
  # max(row_idx) + 1 respectively. Sorting the rows then achieves this.

  # True iff row_start is in this block of rows.
  upper_prow = jnp.isin(jnp.full(1, row_start), rows)
  upper_prow = jnp.full_like(row_idxs, upper_prow)
  mask_value = jnp.where(upper_prow,
                         x=jnp.full_like(row_idxs, -1),
                         y=jnp.full_like(row_idxs, row_end + 1))

  first_prow_idx = rows[0, 0]
  last_prow_idx = rows[-1, 0]
  to_be_inserted_here = jnp.logical_and(
    row_idxs >= first_prow_idx, row_idxs <= last_prow_idx)
  masked_row_idxs = jnp.where(to_be_inserted_here, x=row_idxs, y=mask_value)
  sort_idxs = jnp.argsort(masked_row_idxs)
  panel = panel[sort_idxs, :]
  in_range = jnp.logical_and(rows >= row_start, rows < row_end)
  return jnp.where(in_range, x=panel, y=matrix)


################################################################################
# DOT
################################################################################
def _paneled_dot(A, B, precision, panel_size_threshold):
  """Compute A @ B, breaking the computation into panels if necessary for memory
  use.

  By panels we mean computing C = A @ B as e.g. C[:ps0, :ps2] = A[:ps0, :] @
  B[:, :ps2], etc. The breaking into panels is done if the temporary arrays that
  the call to jnp.dot(A, B) would result in would cause roughly more than
  `panel_size_threshold` gigabytes of memory allocations. Enough panels are used
  such that the allocations would stay (roughly) below that threshold.

  This should rarely if ever be necessary for float32 matmuls, but with ef57
  data types jnp.dot allocates a lot of temporaries (roughly 12x the inputs + 8x
  the output).

  The paneling, while saving memory, obviously causes a time overhead, that
  varies with matrix size.

  Args:
    A, B: The matrices to multiply.
    precision: Jax matmul precision.
    panel_size_threshold: Rough maximum number of gigabytes to allow for
      temporary arrays.
  Returns:
    jnp.dot(A, B, precision=precision)
  """
  if A.dtype != B.dtype:
    msg = (f"Can't compute a paneled matmul of matrices with mixed dtypes "
           f"({A.dtype} and {B.dtype}.")
    raise TypeError(msg)
  dim0, dim1 = A.shape
  _, dim2 = B.shape
  # Sizes in gigabytes
  bytes_per_element = misc.byte_count(A.dtype)
  giga = 2**30
  A_size = dim0 * dim1 * bytes_per_element / giga
  B_size = dim1 * dim2 * bytes_per_element / giga
  C_size = dim0 * dim2 * bytes_per_element / giga
  # Compute how many panels each matrix needs to be divided to, at the least.
  is_ef57 = A.dtype in EF57_DTYPES
  # The fact that these prefactors are large for ef57 is the reason for the
  # existence of this function. See REDACTED
  input_prefactor = 6 if is_ef57 else 1
  output_prefactor = 4 if is_ef57 else 1
  A_num_panels = int(np.ceil(input_prefactor * A_size / panel_size_threshold))
  B_num_panels = int(np.ceil(input_prefactor * B_size / panel_size_threshold))
  C_num_panels = int(np.ceil(output_prefactor * C_size / panel_size_threshold))
  if A_num_panels == 1 and B_num_panels == 1 and C_num_panels == 1:
    return jnp.dot(A, B, precision=precision)
  # C will get A_num_panels * B_num_panels panels. If that's too small, increase
  # the paneling of A and/or B as necessary. This will only trigger in cases
  # where the summed over index is smaller than the free indices, a somewhat
  # rare occasion.
  while A_num_panels * B_num_panels < C_num_panels:
    if A_num_panels == 1:
      B_num_panels = C_num_panels
    elif B_num_panels == 1:
      A_num_panels = C_num_panels
    else:
      A_num_panels *= 2
      B_num_panels *= 2
  # One may wonder if we should make sure that the dimensions are divisible by
  # the panel sizes. Turns out we don't, because of how dynamic_slice and
  # dynamic_update_slice handle overruns. What happens in that case is that dim0
  # % A_num_panels rows (dim2 % B_num_panels columns) are computed twice, a
  # negligible cost since the panel numbers will always be much smaller than the
  # dimension.
  # ps for panel size. These are rounded up, to rather recompute a few
  # rows/columns than leave a few uncomputed.
  ps0 = int(np.ceil(dim0 / A_num_panels))
  ps2 = int(np.ceil(dim2 / B_num_panels))
  C = jnp.empty((dim0, dim2), dtype=A.dtype)

  def body(i, args):
    A, B, C = args
    i0, i2 = divmod(i, B_num_panels)
    start0 = i0 * ps0
    start2 = i2 * ps2
    A_panel = jax.lax.dynamic_slice(A, (start0, 0), (ps0, dim1))
    B_panel = jax.lax.dynamic_slice(B, (0, start2), (dim1, ps2))
    C_panel = jnp.dot(A_panel, B_panel, precision=precision)
    C = jax.lax.dynamic_update_slice(C, C_panel, (start0, start2))
    return A, B, C

  _, _, C = jax.lax.fori_loop(0, A_num_panels * B_num_panels, body, (A, B, C))
  return C


def dot(
    A,
    B,
    precision=jax.lax.Precision.HIGHEST,
    ef57_paneling=True,
    panel_size_threshold=1,
):
  """Compute `jnp.dot(A, B, precision=precision)`, with bells and whistles.

  The bells: The default precision is `HIGHEST` (for `jnp.dot` it's `DEFAULT`).
  The whistles: If `ef57_paneling` is `True`, then large ef57 matrices are
  broken into panels and the matmul is done in parts, to work around the large
  memory overhead of the ef57 `jnp.dot`.

  Args:
    A, B: The matrices to multiply.
    precision: Jax matmul precision. `jax.lax.Precision.HIGHEST` by default.
    ef57_paneling: Boolean for whether to use paneling for large ef57 matrices.
      `True` by default.
    panel_size_threshold: The rough maximum amount of memory we should allow
      the temporary arrays in `jnp.dot` to use, in gigabytes. Only relevant if
      `ef57_paneling is True`. 1 by default. Changing this should rarely be
      necessary.
  Returns:
    `jnp.dot(A, B, precision=precision)`
  """
  if not ef57_paneling or (A.dtype not in EF57_DTYPES and
                           B.dtype not in EF57_DTYPES):
    return jnp.dot(A, B, precision=precision)
  else:
    return _paneled_dot(A, B, precision, panel_size_threshold)


################################################################################
# INITIALIZATION
################################################################################
def distribute_global(matrix: np.ndarray):
  """
  Distribute a 2D array onto the globally available Jax devices.

  In a single-host setting this is equivalent to `distribute(matrix)`. In a
  multi-host setting, each host should have a copy of the same matrix, and this
  matrix should have dimensions divisible by `config.GRID`. The matrix is then
  distributed over the global device grid according to
  `config.get_processor_grid`.

  WARNING: No check is performed that all host-matrices have identical values.

  Args:
    matrix: A two-dimensional array to be distributed.

  Returns:
    ShardedDeviceArray: The distributed matrix.

  Raises:
    ValueError: If `matrix.shape` is not divisible by `config.GRID`.

  """
  nrows, ncols = matrix.shape
  if nrows % NROWS != 0:
    raise ValueError(f"matrix.shape[0] = {nrows} not "
                     f"evenly divisible by NHROWS = {NHROWS}")
  if ncols % NCOLS != 0:
    raise ValueError(f"matrix.shape[1] = {ncols} not "
                     f"evenly divisible by NHCOLS = {NHCOLS}")
  d0 = nrows // NROWS
  d1 = ncols // NCOLS
  local_devices = jax.local_devices()
  host_id = jax.host_id()
  panels = []
  for i, dev in enumerate(local_devices):
    axis_index = host_id * NDPROCS + i
    pcol = your_pcol(axis_index)
    prow = your_prow(axis_index)
    row_slice = slice(prow * d0, (prow + 1) * d0, None)
    col_slice = slice(pcol * d1, (pcol + 1) * d1, None)
    panels.append(jax.device_put(matrix[row_slice, col_slice], dev))
  return jax.device_put_sharded(panels, local_devices)


def distribute(matrix: np.ndarray,
               pmap: bool = True) -> pxla.ShardedDeviceArray:
  """
  Distribute a 2D array onto the local Jax devices.

  In a multi-host setting, each host should hold one piece of a larger global
  matrix, that is to be distributed over the devices of that host. In a
  single-host setting this function is equivalent to `distribute_global`.
  The matrix local to each host should have dimensions divisible by
  `config.DGRID`.

  Args:
    matrix: A two-dimensional array to be distributed.

  Returns:
    ShardedDeviceArray: The distributed matrix.

  Raises:
    ValueError: If `matrix.shape` is not divisible by `config.DGRID`.

  """
  if not np.all([s % p == 0 for s, p in zip(matrix.shape, DGRID)]):
    raise ValueError(f"matrix.shape = {matrix.shape} not evenly divisible "
                     f"by DGRID = {DGRID}.")

  ndim = matrix.ndim
  if ndim != 2:
    raise ValueError(f"matrix.ndim = {ndim} must be 2 in this version.")

  pshape = np.asarray(DGRID)

  shape = misc.flatten(
      [p, s] for s, p in zip(np.array(matrix.shape) // pshape, pshape))
  perm = list(range(2 * ndim - 2, -1, -2)) + list(range(1, 2 * ndim, 2))
  reshaped = matrix.reshape(shape).transpose(perm)
  final_shape = (np.prod(reshaped.shape[:ndim]), *reshaped.shape[ndim:])
  A = reshaped.reshape(final_shape)
  if not pmap:
    return A
  return jax.pmap(lambda x: x, devices=jax.local_devices())(A)


def distribute_sparse(A):
  msg = "distribute_sparse has been renamed to distribute_sparse_global"
  warnings.warn(msg, DeprecationWarning)
  return distribute_sparse_global(A)


def distribute_sparse_global(A):
  """Distributes a sparse matrix as a dense ShardedDeviceArray, globally.

  This is the equivalent of `distribute_global`, but for sparse matrices.

  The function works by building each dense block on a host in turn, sending it
  to a device, and discarding it from host memory. Thus it is capable of
  distributing matrices that wouldn't fit in host memory as dense matrices, as
  long as the individual device-blocks fit in host and device memory. In a
  multi-host setup, the sparse matrix A should be the same on all hosts.

  Args:
    A: A scipy.sparse sparse matrix.
  Raises:
    TypeError: If A's dimensions are not divisible by `config.GRID`.
    ValueError: If the number of local devices is different from the number of
      grid points assigned to this host.
  Returns:
    Ap: The distributed ShardedDeviceArray.
  """
  local_devices = jax.local_devices()
  n_ldevices = len(local_devices)
  if n_ldevices != NDPROCS:
    msg = ("Number of local devices ({}) is different from number of local "
           "grid points ({}).".format(n_ldevices, NDPROCS))
    raise ValueError(msg)
  n_rows, n_cols = A.shape
  if n_rows % NROWS != 0:
    msg = ("The first dimension of A ({}) isn't divisible by the number of "
           "grid rows ({})".format(n_rows, NROWS))
    raise ValueError(msg)
  if n_cols % NCOLS != 0:
    msg = ("The second dimension of A ({}) isn't divisible by the number of "
           "grid columns ({})".format(n_cols, NCOLS))
    raise ValueError(msg)
  d0 = n_rows // NROWS
  d1 = n_cols // NCOLS
  host_id = jax.host_id()
  A = A.tocsr()
  shards = []
  for i, dev in enumerate(local_devices):
    name = host_id * NDPROCS + i
    pcol = your_pcol(name)
    prow = your_prow(name)
    row_slice = slice(prow * d0, (prow + 1) * d0, None)
    col_slice = slice(pcol * d1, (pcol + 1) * d1, None)
    block = A[row_slice, :].tocsc()[:, col_slice]
    shards.append(jax.device_put(block.todense(), dev))

  Ap = jax.device_put_sharded(shards, local_devices)
  return Ap


@functools.partial(pmap, out_axes=None)
def undistribute_global(matrix):
  """Collect a globally distributed matrix into a 2D array.

  This is the reverse operation of `distribute_global`: It collects the globally
  distributed matrix onto each ASIC as a single array. In a single-host setting
  this is equivalent to `undistribute(matrix)`.

  Args:
    matrix: Array to be undistributed.

  Returns:
    DeviceArray: The undistributed 2D array.
  """
  d0, d1 = matrix.shape
  result = jnp.zeros((NROWS * d0, NCOLS * d1), dtype=matrix.dtype)
  prow = my_prow()
  pcol = my_pcol()
  result = jax.lax.dynamic_update_slice(result, matrix, (prow * d0, pcol * d1))
  result = jax.lax.psum(result, axis_name=AXIS_NAME)
  return result


def undistribute(
    matrix: pxla.ShardedDeviceArray,
    host_local: bool = True,
    collect_to_host: bool = False,
) -> np.ndarray:
  """Collect a distributed matrix into a 2D array.

  This is the reverse operation of `distribute`. In a multi-host setting, each
  host gets an array corresponding to the block of the global matrix that
  resided on the devices local to that host. In contrast to
  `undistribute_global`, this block isn't embedded in a global matrix padded
  with zeros to the right size, but just returned by itself.

  Args:
    matrix: A distributed array to be undistributed into a local array.
    host_local: If True, it is assumed that each host contains a different
      matrix so that the resulting undistributed matrix will be different on
      each host. If False, it is instead assumed that each host contains a
      copy of the same matrix which was initially distributed across the full
      grid (e.g. a multi-host-distributed matrix after an all-gather).
    collect_to_host: By default, the return value is a `DeviceArray` on device
      #0. If `collect_to_host is True`, then it is instead a Numpy array on the
      host.

  Returns:
    numpy.ndarray: The undistributed 2D array.
  """
  if host_local:
    grid_shape = DGRID
  else:
    grid_shape = GRID

  local_shape = matrix.shape[1:]

  shape = tuple(grid_shape[::-1]) + local_shape
  perm = misc.flatten([[len(grid_shape) - 1 - n, n + len(grid_shape)]
                       for n in range(len(grid_shape))])

  final_shape = misc.global_shape(local_shape, grid_shape)
  if collect_to_host:
    matrix = np.array(matrix)
  return matrix.reshape(shape).transpose(perm).reshape(final_shape)


@functools.partial(jax.jit, static_argnums=(0, 1))
def eye(local_shape, dtype, k=0, unpadded_dim=None):
  """
  Returns a matrix with ones on its `k'th`, diagonal and zeroes elsewhere,
  with local shape `local_shape`, dtype `dtype`,
  distributed across the grid `grid`.

  Args:
    local_shape: The shape of the matrix block on each core.
    dtype: dtype of the matrix.
    k: The diagonal to fill with ones.
    unpadded_dim: If specified, only the top left `unpadded_dim x unpadded_dim`
                  block will be nonzero.
  Returns:
    The distributed rectangular identity.
  """
  identity_matrix = jnp.zeros(local_shape, dtype=dtype)
  identity_matrix = fill_diagonal(identity_matrix, 1, k=k)
  if unpadded_dim is not None:
    identity_matrix = apply_pad(identity_matrix, unpadded_dim)
  return identity_matrix


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


def your_pcol(p):
  """
  Returns the pcol inhabited by processor p.
  Args:
    p: The processor number.
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    pcol: The pcol of processor p.
  """
  host_idx = p // NDPROCS
  device_idx = p - NDPROCS * host_idx
  host_col = host_idx // NHROWS
  device_col = device_idx // NDROWS
  return host_col * NDCOLS + device_col


def your_prow(p):
  """
  Returns the prow inhabited by processor p.
  Args:
    p: The processor number.
  Returns:
    prow: The prow of processor p.
  """
  host_idx = p // NDPROCS
  device_idx = p - NDPROCS * host_idx
  host_row = host_idx % NHROWS
  device_row = device_idx % NDROWS
  return host_row * NDROWS + device_row


def my_pcol():
  """
  Returns the pcol inhabited by this processor.
  Args:
    grid: Tuple of processor grid dimensions (prows, pcols).
  Returns:
    pcol: The pcol of this processor.
  """
  return your_pcol(my_name())


def my_prow():
  """
  Returns the prow inhabited by this processor.
  Returns:
    prow: The prow of this processor.
  """
  return your_prow(my_name())


def in_this_pcol(pcol):
  """
  Returns a bool describing whether this processor inhabits `pcol`.
  Args:
    pcol: The pcol.
  Returns:
    The bool.
  """
  return pcol == my_pcol()


def in_this_prow(prow):
  """
  Returns a bool describing whether this processor inhabits `prow`.
  Args:
    prow: The prow.
  Returns:
    The bool.
  """
  return prow == my_prow()


################################################################################
# AXIS_INDEX_GROUPS
################################################################################
def _totuple(a):
  """
  Converts a numpy array into nested tuples, so that each row of the array
  is a different tuple.
  E.g.
  a = [ [0 1 2 3]
        [4 5 6 7] ]
  out = ((0, 1, 2, 3), (4, 5, 6, 7))
  """
  try:
    return tuple(_totuple(i) for i in a)
  except TypeError:
    return a


def _axis_index_prows():
  """
  Returns axis_index_groups such that the relevant operation is performed
  over prows. Thus for grid=(4, 2), returns
  ( (0, 4), (1, 5), (2, 6), (3, 7) ); that is, one nest tuple for each row
  of processors.
  """
  return _totuple(config.get_processor_grid())


def _axis_index_pcols():
  """
  Returns axis_index_groups such that the relevant operation is performed
  over pcols. Thus for grid=(4, 2), returns
  ( (0, 1, 2, 3), (4, 5, 6, 7) ); that is, one nested tuple for each column
  of processors.
  """
  return _totuple(config.get_processor_grid().T)


def _axis_index_prow_pairs(start_from_zero=True):
  """ Returns axis_index_groups such that the relevant operation is performed
  over alternating prows. Thus for grid=(4, 2), returns
  ((0, 1), (2, 3), (4, 5), (6, 7)) (start_from_zero = True) or
  ((0, 3), (1, 2), (4, 7), (5, 6) (False)); that is, one nested tuple for each
  pair of cores.
  """
  ps = config.get_processor_grid().T
  if not start_from_zero:
    ps = np.roll(ps, 1, axis=1)
  ps = ps.flatten()
  n_pairs = ps.size // 2
  tuples = _totuple(ps.reshape((n_pairs, 2)))
  return tuple(tuple(sorted(t)) for t in tuples)


def _paxis_groups_to_linear(groups, rows=True):
  """ `groups` is a nested sequence of prows to group together. Returns the
  corresponding nested sequence of pmap axis indices, one per pcol per group.

  E.g. on (4, 2) with rows = True:
    groups = ((0, 1), (2, 3))
    returns ((0, 1), (2, 3), (4, 5), (6, 7))
  On (4, 2) with rows = False:
    groups = ((1, 0))
    returns ((4, 0), (5, 1), (6, 2), (7, 3))
  """
  pgrid = config.get_processor_grid()
  if not rows:
    pgrid = pgrid.T
  grouped = pgrid[groups, :]
  n_groups, per_group, n_other = grouped.shape
  reshaped = grouped.transpose((1, 0, 2)).reshape(
    (per_group, n_groups * n_other))
  return _totuple(reshaped.T)


################################################################################
# COMMUNICATION
################################################################################
def _psum_masked(masked, axis_name, axis_index_groups=None):
  """Computes `psum` for a mask matrix in a broadcast.

  The result is the same as that of
  `jax.lax.psum(masked, axis_name, axis_index_groups)`.
  However, for ef57 matrices an extra optimisation is used to save memory and
  communication time, using the assumption that `masked` is only non-zero on
  one of the cores in each `axis_index_groups`. This often arises when using
  `psum` for broadcasting.
  """
  dtype = masked.dtype
  if dtype in (jnp.float64, jnp.complex128):
    lower_dtype = jnp.float32 if dtype == jnp.float64 else jnp.complex64
    hi = masked.astype(lower_dtype)
    lo = (masked - hi.astype(dtype)).astype(lower_dtype)
    hi, lo = jax.lax.psum(
        (hi, lo),
        axis_name,
        axis_index_groups=axis_index_groups,
    )
    return hi.astype(dtype) + lo.astype(dtype)
  else:
    return jax.lax.psum(masked, axis_name, axis_index_groups=axis_index_groups)


def broadcast_prow(A, prow):
  """
  Returns the result of a broadcast of the portion of `A` in `prow` to all
  other prows.
  E.g. with grid=(2, 2), prow=1:
  A = [ [a], [b] ]
      [ [c], [d] ]
  Out = [ [c], [d] ]
        [ [c], [d] ]

  Args:
    A: The array.
    prow: The prow to broadcast.
  Returns:
    The broadcasted array.
  """
  groups = _axis_index_pcols()
  masked_A = mask_except_prow(A, prow)
  return _psum_masked(masked_A, AXIS_NAME, axis_index_groups=groups)


def _paxis_groups_error_checks(bcast_indices, groups, axis_size):
  """ Error checks for _axis_index_prow_groups and _axis_index_pcol_groups.
  """
  if not len(groups):
    raise TypeError("Must specify at least one group.")
  ngroups = len(bcast_indices)
  if ngroups != len(groups):
    raise TypeError(f"len(bcast_indices)={ngroups} must equal "
                    f"len(groups)={len(groups)}")
  group_sizes = np.array([len(group) for group in groups])
  group_size = group_sizes[0]
  if ngroups > 1:
    if not np.all(group_sizes[1:] == group_size):
      raise ValueError("Groups must be of equal size.")
  all_pidxs = np.hstack([g for g in groups])
  if set(all_pidxs) != set(range(axis_size)):
    raise ValueError(f"groups={groups} must contain each paxis exactly once.")
  too_small = np.array(bcast_indices) < 0
  too_big = np.array(bcast_indices) >= group_size
  if np.any(np.logical_or(too_small, too_big)):
    raise ValueError(f"Invalid group indices {bcast_indices}.")


def broadcast_paxis_to_groups(A, bcast_indices, groups, rows=True):
  """ Broadcasts A's data in each prow (rows=True) or pcol (False)
  groups[i][bcast_indices[i]] to the other prows/pcols in groups[i].

  E.g. Rows True, grid=(4, 2), bcast_indices=(1, 0), groups=((0, 3), (1, 2))
  A = [ [a], [b] ]
      [ [c], [d] ]
      [ [e], [f] ]
      [ [g], [h] ]
  Out = [ [g], [h] ]  x prows (0, 3) grouped, groups[0][1] = 3 broadcast
        [ [c], [d] ]  < prows (1, 2) grouped, groups[1][0] = 1 broadcast
        [ [c], [d] ]  <
        [ [g], [h] ]  x

  Rows False, grid=(4, 4), bcast_indices=(1, 0), groups=((0, 3), (1, 2))
  A = [ [a], [b], [c], [d] ]
      [ [e], [f], [g], [h] ]
      [ [i], [j], [k], [l] ]
      [ [m], [n], [o], [p] ]

  Out = [ [d], [b], [b], [d] ]
        [ [h], [f], [f], [h] ]
        [ [l], [j], [j], [l] ]
        [ [p], [n], [n], [p] ]
  Args:
    A: Matrix to broadcast.
    bcast_indices: Concrete sequence of the same length as `groups`,
      specifying which entry in the corresponding group to broadcast.
    groups: Concrete nested sequence of equally-sized concrete integer
      sequences, specifying which prows will be grouped together. Each
      integer in range(NROWS or COLS) must appear exactly once.
  """
  if rows:
    axis_size = NROWS
    my_pidx = my_prow()
  else:
    axis_size = NCOLS
    my_pidx = my_pcol()
  _paxis_groups_error_checks(bcast_indices, groups, axis_size)
  # the linear indices of the groups
  linear_groups = _paxis_groups_to_linear(groups, rows=rows)
  to_broadcast = [g[idx] for g, idx in zip(groups, bcast_indices)]
  do_not_mask = jnp.isin(my_pidx, np.array(to_broadcast))
  masked_A = mask(A, do_not_mask)
  return _psum_masked(masked_A, AXIS_NAME, axis_index_groups=linear_groups)


def broadcast_pcol(A, pcol):
  """
  Returns the result of a broadcast of the portion of `A` in `pcol` to all
  other pcols.
  E.g. with grid=(2, 2), pcol=1:
  A = [ [a], [b] ]
      [ [c], [d] ]
  Out = [ [b], [b] ]
        [ [d], [d] ]

  Args:
    A: The array.
    pcol: The pcol to broadcast.
  Returns:
    The broadcasted array.
  """
  groups = _axis_index_prows()
  masked_A = mask_except_pcol(A, pcol)
  return _psum_masked(masked_A, AXIS_NAME, axis_index_groups=groups)


def gather_columns(A):
  """
  Concatenates (vstacks) the checkerboard-distributed matrix A within each
  pcol, so that each prow now stores a copy of the same data.

  If A had local shape (m_l, n_l), the result has local shape
  (grid[0] * m_l, n_l).
  """
  groups = _axis_index_pcols()
  A = jax.lax.all_gather(A, axis_index_groups=groups, axis_name=AXIS_NAME)
  return A.reshape((A.shape[0] * A.shape[1], A.shape[2]))


def scatter_columns(A):
  """
  Performs the reverse operation as gather_columns. Thus, each prow receives
  the prow'th row-slice of A.

  If A had local shape(grid[0] * m_l, n_l), thre result has local shape
  (m_l, n_l). If the number of local rows in A is not an even multiple
  of grid[0] an error is thrown.
  """
  m_l, n_l = A.shape
  if m_l % NROWS != 0:
    raise TypeError(f"Rows of A: {A.shape} can't be scattered over"
                    f"{NROWS} prows.")
  panel_size = m_l // NROWS
  start = my_prow() * panel_size
  return jax.lax.dynamic_slice(A, (start, jnp.zeros_like(start)),
                               (panel_size, n_l))


def gather_prow_pairs(A, start_from_zero=True):
  """ Each prow vstacks its data with its immediate downstairs neighbour.
  Depending on the value of `start_from_zero`, prows are paired either like
  (e.g.) (0, 1), (2, 3) (start_from_zero=True),  or (1, 2), (3, 0)
  (start_from_zero=False).
  """
  groups = _axis_index_prow_pairs(start_from_zero)
  A = jax.lax.all_gather(A, axis_index_groups=groups, axis_name=AXIS_NAME)
  return A.reshape((A.shape[0] * A.shape[1], A.shape[2]))


@functools.partial(jax.jit, static_argnums=(1, 2))
def roll_paxis(A, shift, rows):
  """ Cyclically permutes the data in A by `shift` prows.

  Args:
    A: Matrix to permute.
    shift: Number of prows to permute by.
    rows: Rolls prows if True, pcols if False.
  Returns:
    The permuted matrix.
  """
  if rows:
    paxis = np.arange(NROWS)
  else:
    paxis = np.arange(NCOLS)
  rolled = np.roll(paxis, -shift)
  stacked = np.vstack((paxis, rolled)).T
  tups = _totuple(stacked)
  groups = _paxis_groups_to_linear(tups, rows=rows)
  return jax.lax.ppermute(A, AXIS_NAME, groups)


def vstack_equal_shape(A, B):
  """ Returns C = [A, B]^T, where  A and B are checkerboard-distributed
  matrices of the same shape. This should be pmapped.

  Args:
    A: An M x N checkerboard distributed matrix.
    B: An M x N checkerboard distributed matrix.
  Returns:
    C: The 2M x N checkerboard distributed result.
  """
  if NROWS % 2 != 0:
    raise ValueError("vstack_equal_shape assumes an even number of prows.")
  if A.shape != B.shape:
    raise TypeError(f"A.shape = {A.shape} must equal B.shape = {B.shape}.")
  A = gather_prow_pairs(A)  # Now 2M x N
  B = gather_prow_pairs(B)  # Now 2M x N
  prow_half = NROWS // 2 - 1
  A = roll_paxis(A, -prow_half, True)
  B = roll_paxis(B, prow_half, True)
  C = jnp.where(my_prow() <= prow_half, x=A, y=B)
  return C


def gather_rows(A):
  """
  Concatenates (hstacks) the checkerboard-distributed matrix A within each
  pcol, so that each pcol now stores a copy of the same data.

  If A had local shape (m_l, n_l), the result has local shape
  (m_l, grid[1] * n_l).
  """
  groups = _axis_index_prows()
  A = jax.lax.all_gather(A, axis_index_groups=groups, axis_name=AXIS_NAME)
  A = A.transpose((1, 0, 2))
  return A.reshape((A.shape[0], A.shape[1] * A.shape[2]))


def scatter_rows(A):
  """
  Performs the reverse operation as gather_columns. Thus, each pcol receives
  the pcol'th column-slice of A.

  If A had local shape(m_l, grid[1] * n_l), the result has local shape
  (m_l, n_l). If the number of local columns in A is not an even multiple
  of grid[1] an error is thrown.
  """
  m_l, n_l = A.shape
  if n_l % NCOLS != 0:
    raise TypeError(f"Columns of A: {A.shape} can't be scattered over"
                    f"{NCOLS} pcols.")
  panel_size = n_l // NCOLS
  start = my_pcol() * panel_size
  return jax.lax.dynamic_slice(A, (jnp.zeros_like(start), start),
                               (m_l, panel_size))


def _asic_cluster_to_asic_node_axis_index_groups(grid):
  """Creates the axis_index_groups that group together all the device-blocks
  that would be part of the same block if the matrix was distributed over a
  single asic_node.

  In the device `grid` these groups correspond to submatrices
  `grid[:block_rows, :block_cols], grid[block_rows:2*block_rows, :block_cols]`,
  etc.
  """
  asic_node_rows, asic_node_cols = NDROWS, NDCOLS
  block_rows = NROWS // asic_node_rows
  block_cols = NCOLS // asic_node_cols
  axis_index_groups = grid.reshape(
      (asic_node_rows, block_rows,
       asic_node_cols, block_cols)).transpose((0, 2, 1, 3)).reshape(
           asic_node_rows * asic_node_cols, block_rows * block_cols)
  return list(axis_index_groups)


def gather_to_asic_nodes(A):
  """
  Gathers a globally distributed matrix to individual asic_nodes.

  `gather_to_asic_nodes` takes a matrix that is distributed over the global device
  grid, possibly over multiple hosts, and gathers its blocks so that each asic_node
  attached to a single host gets a copy of the matrix, distributed in the same
  manner as `distribute` would use if running on a single asic_node. Note that when
  running on a single asic_node this is a no-op.

  This function should be called within a pmap, on a pmapped argument. The cost
  is that of one `all_gather` and one `pshuffle`.

  Args:
    A: The distributed matrix to gather.
  Returns:
    The gathered matrix.
  """
  # NOTE(mhauru): So far this code has been tested on a v_3 and a v_2.
  ndim = A.ndim
  if ndim != 2:
    raise ValueError(f"A.ndim should be 2 in gather_to_asic_nodes, but was {ndim}.")
  local_rows, local_cols = A.shape
  grid = config.get_processor_grid()
  asic_node_rows, asic_node_cols = NDROWS, NDCOLS
  # Let us call the blocks of the matrix that reside on individual devices when
  # the distribution is over the whole global grid asic_cluster-blocks, and the blocks
  # that reside on individual devices when the distribution is over a single
  # asic_node asic_node-blocks. Each asic_cluster-block corresponds to one element in `grid`.
  #
  # The heart of this function is an all_gather with axis_index_groups and a
  # pshuffle. The axis_index_groups are such that those asic_cluster-blocks which
  # should form a single asic_node-block go in one group.
  axis_index_groups = _asic_cluster_to_asic_node_axis_index_groups(grid)
  A = jax.lax.all_gather(
      A,
      axis_name=AXIS_NAME,
      axis_index_groups=axis_index_groups,
  )
  # Each core now has block_rows * block_cols asic_cluster-blocks, that we reshape to
  # form a single asic_node-block.
  block_rows = NROWS // asic_node_rows
  block_cols = NCOLS // asic_node_cols
  A = A.reshape((block_rows, block_cols, local_rows, local_cols)).transpose(
      (0, 2, 1, 3)).reshape((block_rows * local_rows, block_cols * local_cols))
  # Finally, we need to permute the distribution of the blocks, so that each
  # asic_node has a full copy of the matrix distributed in the usual (4, 2) asic_node
  # distribution pattern. For reasons hard to explain without getting graphical,
  # this corresponds to a permutation where both the row and the column space of
  # `grid` is partitioned into a tensor product of two indices, one for the
  # asic_cluster-blocks internal to a asic_node-block, and one for the asic_node-blocks, and
  # these two indices are swapped.
  grid_perm = grid.reshape((asic_node_rows, block_rows, asic_node_cols,
                            block_cols)).transpose((1, 0, 3, 2)).reshape(
                                (NROWS, NCOLS))
  # grid_perm gives the device ordering we should permute to on the grid. Now we
  # just read the elements of grid_perm in the order given by grid, to find out
  # the pshuffle permutation that implements this new ordering.
  perm = grid_perm.ravel()[misc.inverse_permutation(grid.ravel())]
  A = jax.lax.pshuffle(A, axis_name=AXIS_NAME, perm=perm)
  return A


################################################################################
# PROCESSOR MASKS
################################################################################
def mask(A, cond):
  """
  Returns an array, copied from A, with entries zeroed out where
  `cond` is False. `cond` may be a single boolean value (in which case all
  of A is either zeroed out or not) or an array of these with the same shape
  as `A` (in which case the zeroing-out is done elementwise).

  Args:
    A: The array to be masked.
    cond: False where the mask is to be applied.
  Returns:
    A: The masked array.
  """
  do_not_mask = jnp.zeros_like(A, dtype=np.bool) + cond
  return jnp.where(do_not_mask, x=A, y=jnp.zeros_like(A))


def mask_except_pcol(A, pcol):
  """
  Return an array, copied from A but with all entries zeroed out save those
  which inhabit `pcol`.

  E.g. with grid=(2, 2), pcol=1:
  A = [ [a], [b] ]
      [ [c], [d] ]
  Out = [ [0], [b] ]
        [ [0], [d] ]

  Args:
    A: The array to be masked.
    pcol: The pcol to preserve.
  Returns:
    The masked array.
  """
  do_not_mask = in_this_pcol(pcol)
  return mask(A, do_not_mask)


def mask_except_prow(A, prow):
  """
  Return an array, copied from A but with all entries zeroed out save those
  which inhabit `prow`.

  E.g. with grid=(2, 2), prow=1:
  A = [ [a], [b] ]
      [ [c], [d] ]
  Out = [ [a], [b] ]
        [ [0], [0] ]

  Args:
    A: The array to be masked.
    prow: The prow to preserve.
  Returns:
    The masked array.
  """
  do_not_mask = in_this_prow(prow)
  return mask(A, do_not_mask)


def mask_off_diagonal(matrix, k=0, unpadded_dim=None):
  """ Returns a copy of `matrix` with all values 0 except those on the `k`'th
  diagonal.

  Args:
    matrix: The matrix to mask.
    k: The diagonal to leave intact.
    unpadded_dim: If specified, only the top left `unpadded_dim x unpadded_dim`
                  block will be unmasked.
  Returns:
    A copy of `matrix` with all values 0 except those on the `k`th diagonal.
  """
  leave_unmasked = on_kth_diagonal(matrix.shape, k=k, unpadded_dim=unpadded_dim)
  return mask(matrix, leave_unmasked)


def apply_pad(matrix, unpadded_dim):
  """Zero-pads all entries of `matrix` outside its top-left unpadded_dim
  by unpadded_dim block (of the full distributed matrix).

  Args:
    matrix: A checkerboard-distributed matrix.
    unpadded_dim: Size of the block to leave unpadded.
  Returns:
    matrix: With the pad applied.
  """
  leave_unmasked = within_unpadded_block(matrix.shape, unpadded_dim)
  return mask(matrix, leave_unmasked)


################################################################################
# REDUCTIONS
################################################################################
def safe_psum(A, axis_name, **kwargs):
  """Calls `tree_psum` on efloat57 arrays, `jax.lax.psum` for other arrays."""
  if A.dtype in (jnp.float64, jnp.complex128):
    return tree_psum(A, axis_name, **kwargs)
  else:
    return jax.lax.psum(A, axis_name, **kwargs)


def tree_psum(A, axis_name, axis_index_groups=None):
  """Compute a `psum` as a tree-reduction, to save memory with ef57.

  The return value is the same as that of
  `jax.lax.psum(masked, axis_name, axis_index_groups)`.
  This function is only needed as a workaround for psum for ef57 using a lot of
  memory, see b/188162871.

  tree_psum computes the psum as a tree of pair-wise reductions. It makes
  log2(N) calls to jax.lax.psum, where N is the range of the sharded index, but
  each call only sums over pairs of cores. It only works if all the groups in
  `axis_index_groups` are of a length that is a power of two.

  Args:
    Same as for jax.lax.psum.
  Raises:
    Same as jax.lax.psum, plus
    ValueError if the length of an `axis_index_group` is not a power of two.
  Returns:
    Same as for jax.lax.psum.
  """
  if axis_index_groups is None:
    l = jax.lax.psum(1, axis_name)
    axis_index_groups = [list(range(l))]
  l = len(axis_index_groups[0])
  if not misc.is_power_of_two(l):
    msg = (f"tree_psum got an axis_index_group of length {l}. "
           "It can only handle group lengths that are powers of two.")
    raise ValueError(msg)
  for group in axis_index_groups:
    if len(group) != l:
      msg = ("All axis_index_groups must be of equal size, "
             f"got two of sizes {l} and {len(group)}.")
      raise ValueError(msg)
  # The algorithm works like this: Break the cores in each group into pairs,
  # psum over those pairs. Keep the first of each pair in the group, and move
  # the second one to "trash", i.e. since it's already been absorbed into the
  # first one we don't care about it anymore. Repeat the process on the groups
  # that now have half as many cores in them, until each group only has one
  # core. Those cores hold the result of the psum, that is finally broadcasted
  # to the rest of the cores. The reason for keeping track of the "trash" cores
  # is that every psum call must involve all cores. So even though we don't care
  # about the values in the trash cores, we still need to psum them with
  # something.
  groups = list(map(list, axis_index_groups))  # Convert to lists, make a copy.
  trash = []
  while l > 1:
    # It doesn't actually matter how we pair up the cores in trash, just pick
    # some way.
    # REDACTED We could try to optimise the pairing of the cores, to make
    # communication distances as short as possible. I suspect they are usually
    # close to optimal anyway, so the benefit might be small.
    pairs = list(zip(trash[0::2], trash[1::2]))
    for i, g in enumerate(groups):
      evens = g[0::2]
      odds = g[1::2]
      pairs += list(zip(evens, odds))
      groups[i] = evens
      trash += odds
    A = jax.lax.psum(A, axis_name, axis_index_groups=pairs)
    l = l // 2
  # By this point all groups are of length 1. gather_cores are the cores to
  # which we have gathered the result of psumming each axis_index_group, and
  # from which we will broadcast that result.
  gather_cores = sum(groups, [])
  core_id = my_name()
  is_gather_core = sum(core_id == c for c in gather_cores)
  A = mask(A, is_gather_core)
  A = _psum_masked(A, axis_name, axis_index_groups=axis_index_groups)
  return A


def sum_over_prows(A):
  """
  Return an array whose elements are the sum of A's corresponding elements on
  its prow.

  E.g with grid=(2, 2):
    A = [ [1, 2], [3, 4] ]
        [ [5, 6], [7, 8] ]
    Out = [ [4, 6], [4, 6] ]
          [ [12, 14], [12, 14] ]
  """
  groups = _axis_index_prows()
  return safe_psum(A, AXIS_NAME, axis_index_groups=groups)


def sum_over_pcols(A):
  """
  Return an array whose elements are the sum of A's corresponding elements on
  its pcol.

  E.g with grid=(2, 2):
    A = [ [1, 2], [3, 4] ]
        [ [5, 6], [7, 8] ]
    Out = [ [6, 8], [10, 12] ]
          [ [6, 8], [10, 12] ]
  """
  groups = _axis_index_pcols()
  return safe_psum(A, AXIS_NAME, axis_index_groups=groups)


def sum_prow_pairs(A):
  """
  Return an array which differs from A in that each pair of prows has been
  summed. On a ASIC asic_cluster slice, this corresponds to summing the data within
  each chip.

  E.g with grid=(4, 2):
    A = [ [1, 2], [3, 4] ]
        [ [5, 6], [7, 8] ]
        [ [9, 10], [11, 12] ]
        [ [13, 14], [15, 16] ]

    Out = [ [6, 8], [10, 12] ]
          [ [6, 8], [10, 12] ]
          [ [21, 24], [26, 28] ]
          [ [21, 24], [25, 28] ]
  """
  groups = _axis_index_prow_pairs()
  return jax.lax.psum(A, axis_name=AXIS_NAME, axis_index_groups=groups)


def trace(A):
  """
  Returns the trace of A.

  Args:
    A: The matrix.
  Returns:
    The trace (a ShardedDeviceArray with one element per core).
  """
  only_diagonal = mask_off_diagonal(A)
  local_trace = jnp.sum(only_diagonal)
  return jax.lax.psum(local_trace, axis_name=AXIS_NAME)


def frobnorm(A):
  """
  Computes the Frobenius norm of A.

  Args:
    A: The matrix.
  Returns:
    The norm (a ShardedDeviceArray with one element per core).
  """
  squared = jnp.abs(A)**2
  local_sum = jnp.sum(squared)
  global_sum = jax.lax.psum(local_sum, axis_name=AXIS_NAME)
  return jnp.sqrt(global_sum)


def gershgorin(H):
  """
  Computes estimates of the smallest and largest eigenvalues of a Hermitian
  `H` using the "Gershgorin" method. The estimates are guaranteed to bound the
  spectrum, but can be quite loose in many cases.

  Args:
    H: The Hermitian matrix whose spectrum is to be bounded.
  Returns:
    min_est: A lower bound on `H`'s smallest eigenvalue.
    max_est: An upper bound on `H`'s largest eigenvalue.
  """

  def _sum_cols(M):
    M = jnp.sum(M, axis=0)
    M = sum_over_pcols(M)
    M = M.reshape((1, M.shape[0]))
    M = gather_rows(M)
    return jnp.ravel(M)

  def _sum_rows(M):
    M = jnp.sum(M, axis=1)
    M = sum_over_prows(M)
    M = M.reshape((M.shape[0], 1))
    M = gather_columns(M)
    return jnp.ravel(M)

  H_diag = mask_off_diagonal(H)
  H_diag = _sum_cols(H_diag)

  abs_H_diag0 = jnp.abs(fill_diagonal(H, 0.))
  col_sums = _sum_cols(abs_H_diag0)
  row_sums = _sum_rows(abs_H_diag0)

  row_min = jnp.min(H_diag - row_sums)
  col_min = jnp.min(H_diag - col_sums)
  min_est = jnp.max(jnp.array([row_min, col_min]))

  row_max = jnp.max(H_diag + row_sums)
  col_max = jnp.max(H_diag + col_sums)
  max_est = jnp.min(jnp.array([row_max, col_max]))
  return min_est, max_est


################################################################################
# INDEXING AND MAIN DIAGONAL
################################################################################
@functools.partial(jax.jit, static_argnums=(0,))
def indices_vec(local_shape: Tuple):
  """ Given `local_shape = (rows per processor, cols per processor)`,
  returns vectors `row_vec` and `col_vec` respectively indexing these rows
  and cols within the full checkerboard-distributed matrix.
  Args:
    local_shape: Shape of the matrix block on each processor.
  Returns:
    row_vec: Whose `i`th entry indexes the `i`th local row within the full
             distributed matrix.
    col_vec: Whose `j`th entry indexes the `j`th local row within the full
             distributed matrix.
  """
  m, n = local_shape
  i = my_prow()
  j = my_pcol()
  rows_vector = jnp.arange(m) + i * m
  cols_vector = jnp.arange(n) + j * n
  return rows_vector, cols_vector


@functools.partial(jax.jit, static_argnums=(0,))
def indices(local_shape: Tuple):
  """ Returns arrays of shape local_shape storing the respctive column
  and row indices of each local matrix element within the mathematical
  matrix.
  Args:
    local_shape: Shape of the matrix block on each processor.
  Returns:
    rows, cols: The mathematical row and column indices of each local matrix
                element.
  """
  rows_vector, cols_vector = indices_vec(local_shape)
  cols, rows = jnp.meshgrid(cols_vector, rows_vector)
  return rows, cols


def within_unpadded_block(local_shape, unpadded_dim):
  """ Returns a boolean matrix of local shape `local_shape` whose entries
  are `True` within the top-left `unpadded_dim x unpadded_dim` block of the
  corresponding checkerboard-distributed matrix and `False` elsewhere.
  Args:
    local_shape: Shape of the matrix block on each processor.
    unpadded_dim: Size of the top-left block. May be None, in which case a
      matrix of True is returned.
  Returns:
    A matrix of `True` within the top-left `unpadded_dim x unpadded_dim` block
  and `False` elsewhere.
  """
  if unpadded_dim is None:
    return jnp.ones(local_shape, dtype=np.bool)

  rows_vector, cols_vector = indices_vec(local_shape)
  left_panel = rows_vector < unpadded_dim
  top_panel = cols_vector < unpadded_dim
  return jnp.logical_and(left_panel[:, None], top_panel)


def on_kth_diagonal(local_shape: Tuple, k=0, unpadded_dim=None):
  """ Returns a boolean matrix of local shape `local_shape` whose entries
  are `True` upon the `k`th diagonal of the corresponding
  checkerboard-distributed matrix and `False` elsewhere.
  Args:
    local_shape: Shape of the matrix block on each processor.
    k: The diagonal to be selected. k=0 is the main diagonal.
    unpadded_dim: If specified, only entries in the top-left
      `unpadded_dim x unpadded_dim` block of the global matrix will be
      potentially `True`.
  Returns:
    on_kth_diagonal: Of the given local shape; `True` on the `k`th diagonal
      of the global matrix and `False` elsewhere.
  """
  rows_vector, cols_vector = indices_vec(local_shape)
  cols_vector = cols_vector - k
  result = rows_vector[:, None] == cols_vector
  if unpadded_dim is not None:
    unmasked = within_unpadded_block(local_shape, unpadded_dim)
    result = jnp.logical_and(result, unmasked)
  return result


def fill_diagonal(matrix, value, k=0, unpadded_dim=None):
  """
  Returns a matrix identical to `matrix` except that the `k'th` diagonal has
  been overwritten with the value `value`.
  Args:
    matrix: Matrix whose diagonal to fill.
    value: The value to fill the diagonal with.
    k: The diagonal to fill.
    unpadded_dim: If specified, only the `unpadded_dim x unpadded_dim` top left
                  block will be filled.
  Returns:
    A copy of `matrix`, with the `k'th` diagonal replaced by `value`.
  """
  replace_here = on_kth_diagonal(matrix.shape, k=k, unpadded_dim=unpadded_dim)
  replace_with = jnp.full(replace_here.shape[1], value)
  return jnp.where(replace_here, x=replace_with, y=matrix)


def add_to_diagonal(matrix, value, k=0, unpadded_dim=None):
  """
  Returns a matrix identical to `matrix` except that the `k'th` diagonal has
  been summed with the value `value`; for `k=0` this performs
  `matrix = matrix + value * eye`.
  Args:
    matrix: Matrix whose diagonal to add to.
    value: The value to add to the diagonal.
    k: The diagonal to increment.
    unpadded_dim: If specified, only the `unpadded_dim x unpadded_dim` top left
                  block will be incremented.
  Returns:
    A copy of `matrix`, with the `k'th` diagonal incremented by `value`.
  """
  add_here = on_kth_diagonal(matrix.shape, k=k, unpadded_dim=unpadded_dim)
  return jnp.where(add_here, x=matrix + value, y=matrix)


################################################################################
# MATRIX OPERATIONS
################################################################################
def _transpose_preprocess():
  """Do matrix-independent preprocessing in preparation of transposing matrices.

  Many things about how distributed matrices are transposed depend only on the
  processor grid, and not on the matrix. We compute those things here, so that
  they can be computed once at load-time, rather than every time `transpose` is
  JIT compiled.

  Transposes are only supported for processor grids with proportions 2:1 or 1:2,
  which informs the design of this function.

  Args:
    N/A
  Returns:
    horizontal_devices: Whether device blocks are more wide than tall, or vice
      versa.
    perm0: Permutation for the first pshuffle.
    perm1: Permutation for the second pshuffle.
    pre_reversed_devices: Which device numbers need to reverse their blocks
      before the pshuffles.
    pre_reversed_devices: Which device numbers need to reverse their blocks
      after the pshuffles.
  """
  # The square matrix is split into equal sized square blocks. Each device holds
  # two of these blocks, if horizontal_devices is True then as
  # [[block0, block1]], otherwise as
  # [[block0],
  #  [block1]].
  if NROWS == NCOLS * 2:
    horizontal_devices = True
  elif NCOLS == NROWS * 2:
    horizontal_devices = False
  else:
    msg = ("WARNING: transpose is only supported for device grids with "
           "proportions 2:1 or 1:2.")
    # TODO use logging.warn
    print(msg)
    return None

  # There are max_grid_dim * max_grid_dim blocks in total. Each block is given a
  # numerical label, and these labels are organised in a block_grid. The name
  # _pre refers to the situation before the transpose, _post to after.
  max_grid_dim = max(NCOLS, NROWS)
  block_grid_pre = np.arange(max_grid_dim**2).reshape((max_grid_dim,
                                                       max_grid_dim))
  block_grid_post = block_grid_pre.T
  device_grid = config.get_processor_grid()

  # Here we figure out which device has which blocks, both before and after the
  # transpose.
  blocks_by_device_pre = np.zeros((np.size(device_grid), 2), dtype=np.int)
  blocks_by_device_post = np.zeros((np.size(device_grid), 2), dtype=np.int)
  for index, device_number in np.ndenumerate(device_grid):
    i, j = index
    if horizontal_devices:
      row_slc = slice(i, i + 1)
      col_slc = slice(2 * j, 2 * j + 2)
    else:
      row_slc = slice(2 * i, 2 * i + 2)
      col_slc = slice(j, j + 1)
    blocks_by_device_pre[device_number, :] = block_grid_pre[row_slc,
                                                            col_slc].ravel()
    blocks_by_device_post[device_number, :] = block_grid_post[row_slc,
                                                              col_slc].ravel()

  # The sending of blocks between devices can be done with two pshuffles.
  # However, one has to be careful when deciding which blocks are moved in the
  # first pshuffle and which in the second, so that each device only sends and
  # receives at most one block in each pshuffle. One way to guarantee this is to
  # checkerboard color the block_grid (pre or post, doesn't matter), and say
  # that "white" blocks go in the first pshuffle, "black" blocks in the second.
  # Here we create a list indexed by block numbers, and having values 0 or 1 for
  # whether this block should be in the first or second pshuffle.
  perm_by_block = [None] * (max_grid_dim**2)
  for index, block in np.ndenumerate(block_grid_pre):
    perm_by_block[block] = np.sum(index) % 2

  # Next we create the permutations for the two pshuffles, and also figure out
  # which devices need to swap the order of their blocks before and/or after
  # the pshuffles.
  perm0 = [None] * np.size(device_grid)
  perm1 = [None] * np.size(device_grid)
  pre_reversed_devices = []
  post_reversed_devices = []
  # device_pre refers to the number of the device that is sending blocks out
  # (pre-transpose), device_post to the device that is receiving them
  # (post-transpose).
  for device_pre in range(blocks_by_device_pre.shape[0]):
    blocks = blocks_by_device_pre[device_pre, :]
    # If the first block of this block-sending device ("first" meaning left or
    # top depending on horizontal_devices) goes in the second pshuffle, then
    # this device should swap the order of its blocks.
    if perm_by_block[blocks[0]] == 1:
      pre_reversed_devices.append(device_pre)
    for block in blocks:
      # Find out which device has this block post-transpose, and whether for
      # that device this block will be its second or first block.
      device_post, post_which_block = np.where(blocks_by_device_post == block)
      device_post = int(device_post)
      post_which_block = int(post_which_block)
      perm = perm0 if perm_by_block[block] == 0 else perm1
      perm[device_post] = device_pre
      # If the block that gets sent to device_post in the first pshuffle is its
      # second block, then this device should swap the order of its blocks after
      # the pshuffles.
      if perm_by_block[block] == 0 and post_which_block == 1:
        post_reversed_devices.append(device_post)

  return (
      horizontal_devices,
      perm0,
      perm1,
      pre_reversed_devices,
      post_reversed_devices,
  )


def symmetrize(matrix):
  """ Returns 0.5 * (matrix + matrix^H). Should be pmapped.

  Args:
    matrix: Matrix to symmetrize.
  Returns:
    The symmetrized matrix.
  """
  matrix_t = transpose(matrix).conj()
  return 0.5 * (matrix + matrix_t)


_TRANSPOSE_PREPROCESSED = _transpose_preprocess()


def transpose(matrix):
  """Transposes a distributed matrix.

  `transpose` is only implemented for processor grids with either 2:1 or 1:2
  proportions. It also requires that the matrix be of a shape that could be
  distributed on either one of (NROWS, NCOLS) and (NCOLS, NROWS) processor
  grids. Square matrices always fulfill this requirement.

  `transpose` should be called within a pmap. It costs two pshuffles, and some
  local O(D^2) operations.

  Args:
    Matrix to transpose
  Returns:
    The transposed matrix.
  """
  preprocessed = _TRANSPOSE_PREPROCESSED
  if preprocessed is None:
    msg = ("transpose is only supported for device grids with "
           "proportions 2:1 or 1:2.")
    raise NotImplementedError(msg)
  # See _transpose_preprocess for the meaning of these variables.
  (
      horizontal_devices,
      perm0,
      perm1,
      pre_reversed_devices,
      post_reversed_devices,
  ) = preprocessed

  this_device_number = my_name()
  # The next 6 lines just implement
  # pre_reverse = this_device_number in pre_reversed_devices
  # and
  # post_reverse = this_device_number in post_reversed_devices
  # TODO Is there a nicer way to do this even when this_device_number is only
  # known at runtime?
  pre_reverse = jnp.array(False)
  for dev in pre_reversed_devices:
    pre_reverse = jnp.logical_or(pre_reverse, this_device_number == dev)
  post_reverse = jnp.array(False)
  for dev in post_reversed_devices:
    post_reverse = jnp.logical_or(post_reverse, this_device_number == dev)

  d0, d1 = matrix.shape
  if horizontal_devices:
    if d1 % 2 != 0:
      msg = "The number of local columns should be divisible by two."
      raise NotImplementedError(msg)
    block0 = matrix[:, :d1 // 2]
    block1 = matrix[:, d1 // 2:]
  else:
    if d0 % 2 != 0:
      msg = "The number of local rows should be divisible by two."
    block0 = matrix[:d0 // 2, :]
    block1 = matrix[d0 // 2:, :]
  # Effectively `if pre_reverse: block0, block1 = block1, block0`.
  block0_new = jnp.where(pre_reverse, block1, block0)
  block1 = jnp.where(pre_reverse, block0, block1)
  block0 = block0_new
  block0 = jax.lax.pshuffle(block0, AXIS_NAME, perm0).T
  block1 = jax.lax.pshuffle(block1, AXIS_NAME, perm1).T
  # Effectively `if post_reverse: block0, block1 = block1, block0`.
  block0_new = jnp.where(post_reverse, block1, block0)
  block1 = jnp.where(post_reverse, block0, block1)
  block0 = block0_new
  array = jnp.concatenate(
      (block0, block1),
      axis=(1 if horizontal_devices else 0),
  )
  return array
