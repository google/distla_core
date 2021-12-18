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
"""Module for writing and reading DistlaCore matrices to/from sparse formats.

When reading from/writing to a file in the CSC matrix format, this module uses
a 1D block distribution over the column space. This makes the I/O much simpler
in a multihost setup. This 1D distribution is such that each host 0, 1, ..., N
holds a block of full columns like
┌───┬───┬─────┬───┐
│   │   │     │   │
│ 0 │ 1 │ ... │ N │
│   │   │     │   │
└───┴───┴─────┴───┘
These blocks live in host memory, not ASIC memory. The matrix is converted
between this distribution and the DistlaCore 2D distribution as needed, so for
writing, the process is 2D DistlaCore distribution -> 1D distribution -> write to
disk, and for reading it's the reverse. The change of distribution pattern is
done on the ASICs.
"""
import functools
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from distla_core.utils import misc
from distla_core.utils import pops

N_LOCAL_DEVICES = jax.local_device_count()
# STRUC_PACK's file format constants.
STRUC_PACK_HEADER_SIZE = 16
# TODO These next two values depend on the STRUC_PACK file version. Make that
# dependence dynamic.
STRUC_PACK_HEADER_DTYPE = np.int64
STRUC_PACK_COLUMN_POINTER_DTYPE = np.int64
STRUC_PACK_ROW_INDEX_DTYPE = np.int32
STRUC_PACK_VALUE_DTYPE = np.float64
STRUC_PACK_HEADER_STRUC_PACKZE = np.dtype(STRUC_PACK_HEADER_DTYPE).itemsize
STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE = np.dtype(STRUC_PACK_COLUMN_POINTER_DTYPE).itemsize
STRUC_PACK_ROW_INDEX_STRUC_PACKZE = np.dtype(STRUC_PACK_ROW_INDEX_DTYPE).itemsize
STRUC_PACK_VALUE_STRUC_PACKZE = np.dtype(STRUC_PACK_VALUE_DTYPE).itemsize
# Some constant values STRUC_PACK uses in its header format.
STRUC_PACK_FILE_VERSION = STRUC_PACK_HEADER_DTYPE(170915)
STRUC_PACK_UNSET = STRUC_PACK_HEADER_DTYPE(-910910)
STRUC_PACK_REAL_DATA = STRUC_PACK_HEADER_DTYPE(0)
# The STRUC_PACK header. The zeros are where n_basis, n_electrons, and nnz go.
# The last 8 STRUC_PACK_UNSET is what STRUC_PACK calls "user header".
STRUC_PACK_HEADER_TEMPLATE = np.array(
    (STRUC_PACK_FILE_VERSION, STRUC_PACK_UNSET, STRUC_PACK_REAL_DATA, 0, 0, 0, STRUC_PACK_UNSET,
     STRUC_PACK_UNSET) + (STRUC_PACK_UNSET,) * 8,
    dtype=STRUC_PACK_HEADER_DTYPE,
)


def _read_elements(f, dtype, length, offset, buffer_bytes=2**30):
  """From the file `f` read `length` elements of `dtype`, starting at byte
  `offset`. `dtype` should be a Numpy dtype.
  """
  element_bytes = np.dtype(dtype).itemsize
  buffer_elements = buffer_bytes // element_bytes
  if buffer_elements <= 0:
    raise ValueError(f"buffer_bytes = {buffer_bytes} is too small for {dtype}")
  array = np.empty((length,), dtype=dtype)
  os.lseek(f, offset, 0)
  read_elements = 0
  # Buffered reads are needed, if for nothing else, then because os.read goes
  # wrong if the number of bytes to read is more than max int32.
  while read_elements < length:
    read_size = min(buffer_elements, length - read_elements)
    data = np.frombuffer(os.read(f, element_bytes * read_size), dtype=dtype)
    # In case something like int overflow caused reading less than expected.
    read_size = data.size
    array[read_elements:read_elements + read_size] = data
    if read_size <= 0:
      raise ValueError("Failed to read all elements.")
    read_elements += read_size
  return array


def _write_elements(f, array, offset, buffer_bytes=2**30):
  """To the file `f` write the elements of `array` starting at byte `offset`.
  """
  dtype = array.dtype
  element_bytes = np.dtype(dtype).itemsize
  buffer_elements = buffer_bytes // element_bytes
  if buffer_elements <= 0:
    raise ValueError(f"buffer_bytes = {buffer_bytes} is too small for {dtype}")
  os.lseek(f, offset, 0)
  written_elements = 0
  while written_elements < array.size:
    wrote_bytes = os.write(
        f, array.data[written_elements:written_elements + buffer_elements])
    wrote_elements = wrote_bytes // element_bytes
    if wrote_elements <= 0:
      raise ValueError("Failed to write all elements.")
    written_elements += wrote_elements
  assert written_elements == array.size
  return None


@functools.partial(jax.pmap, axis_name="i")
def _barrier_pmap(x):
  return jax.lax.psum(x, axis_name="i")


def _barrier():
  """Force all ASIC hosts to block until every host has reached the same call."""
  if jax.process_count() > 1:
    xs = jnp.arange(jax.local_device_count())
    _barrier_pmap(xs).block_until_ready()


class StructPackCscFile():
  """Handles for STRUC_PACK CSC files."""

  def __init__(
      self,
      path,
      dim=None,
      nnz=None,
      n_electrons=None,
      process_index=jax.process_index(),
      barrier_fn=_barrier,
  ):
    """Creates a new file handle for the file at `path`.

    These file handles can be used for both reading and writing.

    Args:
      path: path to the file, which may or may not already exist.
      dim, nnz, n_electrons: Attributes of the matrix that this file stores/will
        store. If these are `None` (the default), they are read from the header
        of the file.
    """
    self.path = pathlib.Path(path)
    self.dim = dim
    self.nnz = nnz
    self.n_electrons = n_electrons
    already_exists = self.path.exists()
    if process_index == 0:
      os.umask(0)
      self.f = os.open(path, os.O_RDWR | os.O_SYNC | os.O_CREAT, mode=0o666)
    # Let other hosts catch up to the fact that host #0 created the file.
    barrier_fn()
    if process_index != 0:
      # REDACTED Set a time out
      while not self.path.exists():
        time.sleep(1)
      self.f = os.open(path, os.O_RDWR | os.O_SYNC)
    barrier_fn()
    header_not_given = dim is None or nnz is None or n_electrons is None
    if header_not_given:
      if already_exists:
        self.read_header(overwrite=False)
      else:
        msg = ("When creating a StructPackCscFile for a file that does not exist "
               f"({self.path}), dim, nnz, and n_electrons must be provided.")
        raise ValueError(msg)

    self.size = STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE + self.dim * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE + self.nnz * STRUC_PACK_ROW_INDEX_STRUC_PACKZE + self.nnz * STRUC_PACK_VALUE_STRUC_PACKZE

    if process_index == 0:
      current_size = os.lseek(self.f, 0, 2)
      if current_size < self.size:
        # The file probably just got created, let's set it to the right size.
        os.lseek(self.f, self.size - 1, 0)
        os.write(self.f, b"\0")
    barrier_fn()

  def __del__(self):
    """Close the underlying file object on delete."""
    # REDACTED Should there be a _barrier here as well? Might result in
    # hangs, if e.g. file opening goes wrong in one process.
    try:
      os.close(self.f)
    except AttributeError:
      # If things go wrong in __init__, e.g. when opening the file, self.f may
      # not exist.
      pass

  def read_header(self, overwrite=True):
    """Reads and returns the header of the file.

    This both returns the whole header and updates the `dim`, `n_electrons`, and
    `nnz` attributes of this file handle. If overwrite=False the attributes are
    only updated if they are unset (`None`).
    """
    header = _read_elements(self.f, STRUC_PACK_HEADER_DTYPE, STRUC_PACK_HEADER_SIZE, 0)
    if self.dim is None or overwrite:
      self.dim = header[3]
    if self.n_electrons is None or overwrite:
      self.n_electrons = header[4]
    if self.nnz is None or overwrite:
      self.nnz = header[5]
    return header

  def read_column_pointers(self, start_col, end_col):
    """Reads and returns the column pointers from `start_col` to `end_col`."""
    if start_col < 0:
      raise ValueError(f"start_col = {start_col} < 0")
    if end_col > self.dim:
      raise ValueError(f"end_col = {end_col} > {self.dim} = dim")
    length = end_col - start_col
    offset = (STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
              start_col * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE)
    return _read_elements(self.f, STRUC_PACK_COLUMN_POINTER_DTYPE, length, offset)

  def read_row_indices(self, start_index, end_index):
    """Reads and returns the row indices from `start_index` to `end_index`."""
    if start_index < 0:
      raise ValueError(f"start_index = {start_index} < 0")
    if end_index > self.nnz:
      raise ValueError(f"end_index = {end_index} > {self.nnz} = nnz")
    length = end_index - start_index
    offset = (STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
              self.dim * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE +
              start_index * STRUC_PACK_ROW_INDEX_STRUC_PACKZE)
    return _read_elements(self.f, STRUC_PACK_ROW_INDEX_DTYPE, length, offset)

  def read_values(self, start_index, end_index):
    """Reads and returns the values from `start_index` to `end_index`."""
    if start_index < 0:
      raise ValueError(f"start_index = {start_index} < 0")
    if end_index > self.nnz:
      raise ValueError(f"end_index = {end_index} > {self.nnz} = nnz")
    length = end_index - start_index
    offset = (
        STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
        self.dim * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE + self.nnz * STRUC_PACK_ROW_INDEX_STRUC_PACKZE
        + start_index * STRUC_PACK_VALUE_STRUC_PACKZE)
    return _read_elements(self.f, STRUC_PACK_VALUE_DTYPE, length, offset)

  def write_header(self):
    """Write the header of the file.

    Assumes that the file exists already.

    The non-trivial elements -- `dim`, `n_electrons`, and `nnz` -- are taken
    from the attributes of the file handle.
    """
    header = STRUC_PACK_HEADER_TEMPLATE.copy().astype(STRUC_PACK_HEADER_DTYPE)
    header[3] = self.dim
    header[4] = self.n_electrons
    header[5] = self.nnz
    return _write_elements(self.f, header, 0)

  def write_column_pointers(self, array, start_col):
    """Write `array` as the column pointers of this file, from `start_col`
    onwards.

    Assumes that the file exists already.
    """
    length = len(array)
    if start_col < 0:
      raise ValueError(f"start_col = {start_col} < 0")
    if start_col + length > self.dim:
      msg = (f"Can't write {length} columns to a matrix of size {self.dim}, "
             f"starting from column {start_col}.")
      raise ValueError(msg)
    offset = (STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
              start_col * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE)
    array = array.astype(STRUC_PACK_COLUMN_POINTER_DTYPE)
    return _write_elements(self.f, array, offset)

  def write_row_indices(self, array, start_index):
    """Write `array` as the row indices of this file, from `start_index`
    onwards.

    Assumes that the file exists already.
    """
    length = len(array)
    if start_index < 0:
      raise ValueError(f"start_index = {start_index} < 0")
    if start_index + length > self.nnz:
      msg = (f"Can't write {length} columns to a matrix of size {self.nnz}, "
             f"starting from column {start_index}.")
      raise ValueError(msg)
    offset = (STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
              self.dim * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE +
              start_index * STRUC_PACK_ROW_INDEX_STRUC_PACKZE)
    array = array.astype(STRUC_PACK_ROW_INDEX_DTYPE)
    return _write_elements(self.f, array, offset)

  def write_values(self, array, start_index):
    """Write `array` as the values of this file, from `start_index` onwards.

    Assumes that the file exists already.
    """
    length = len(array)
    if start_index < 0:
      raise ValueError(f"start_index = {start_index} < 0")
    if start_index + length > self.nnz:
      msg = (f"Can't write {length} columns to a matrix of size {self.nnz}, "
             f"starting from column {start_index}.")
      raise ValueError(msg)
    offset = (
        STRUC_PACK_HEADER_SIZE * STRUC_PACK_HEADER_STRUC_PACKZE +
        self.dim * STRUC_PACK_COLUMN_POINTER_STRUC_PACKZE + self.nnz * STRUC_PACK_ROW_INDEX_STRUC_PACKZE
        + start_index * STRUC_PACK_VALUE_STRUC_PACKZE)
    array = array.astype(STRUC_PACK_VALUE_DTYPE)
    return _write_elements(self.f, array, offset)

  def read_struc_pack_csc_host_block(
      self,
      process_index,
      process_count,
      max_n_blocks,
  ):
    """Reads a block of columns from an STRUC_PACK CSC file.

    The full CSC matrix is divided into N blocks in the column space, where N is
    the number of hosts, and the ith block will be read and converted into a CSC
    submatrix, where i is the Jax process index of this process. The matrix is
    also padded as required by max_n_blocks.

    Args:
      process_index: Index of this host/Jax process.
      max_n_blocks: Maximum number of blocks the row/column space of the matrix
        will need to be divisible by. This determines padding.
    Returns:
      matrix: A sparse CSC matrix representing the part of the original full CSC
        matrix that this host needs.
      padded_dim: Dimension of the complete matrix after padding.
    """
    unpadded_dim = self.dim
    padding = misc.distance_to_next_divisor(unpadded_dim, max_n_blocks)
    padded_dim = unpadded_dim + padding
    host_n_cols = padded_dim // process_count
    start_col = host_n_cols * process_index
    end_col = min(start_col + host_n_cols, unpadded_dim - 1)
    column_pointers = self.read_column_pointers(start_col, end_col + 1)
    # Change from 1-based indexing to 0-based.
    column_pointers -= 1
    if len(column_pointers) <= host_n_cols:
      # This triggers on the last host because of padding, and because
      # column_pointers are missing the customary nnz as the last element.
      column_pad = host_n_cols + 1 - len(column_pointers)
      column_pointers = np.pad(
          column_pointers,
          (0, column_pad),
          constant_values=self.nnz,
      )
    start_index = column_pointers[0]
    end_index = column_pointers[-1]
    values = self.read_values(start_index, end_index)
    row_indices = self.read_row_indices(start_index, end_index)
    # Change from 1-based indexing to 0-based.
    row_indices -= 1
    column_pointers -= column_pointers[0]
    matrix = sparse.csc_matrix(
        (values, row_indices, column_pointers),
        shape=(padded_dim, host_n_cols),
    )
    return matrix, padded_dim

  def write_struc_pack_csc_host_block(
      self,
      matrix,
      process_index,
      process_count,
      padded_dim,
      all_nnzs,
  ):
    """Writes a block of columns to the memmaps of an STRUC_PACK CSC file.

    The full CSC matrix is divided into N blocks in the column space, where N is
    the number of hosts, and the ith block will be written by the ith Jax
    process. Each process should have a matrix that represents its block. This
    is the reverse operation of _read_struc_pack_csc_host_block.

    Args:
      matrix: The block of this host.
      process_index: Index of this host/Jax process.
      padded_dim: The padded dimension of the full matrix.
      all_nnzs: The output of gather_nnzs(matrix).
    Returns:
      None
    """
    matrix = sparse.csc_matrix(matrix)
    if matrix.nnz != all_nnzs[process_index]:
      msg = (f"matrix.nnz = {matrix.nnz} != "
             f"all_nnzs[{process_index}] = {all_nnzs[process_index]}")
      raise ValueError(msg)
    nnz_preceding = sum(all_nnzs[:process_index])
    total_nnz = sum(all_nnzs)
    # Only one host needs to write the header.
    if process_index == 0:
      self.write_header()
    unpadded_dim = self.dim
    host_n_cols = padded_dim // process_count
    start_col = host_n_cols * process_index
    end_col = min(start_col + host_n_cols, unpadded_dim)
    local_column_pointers = matrix.indptr.astype(np.int64) + nnz_preceding
    # Change from 0-based indexing to 1-based.
    self.write_column_pointers(
        local_column_pointers[:(end_col - start_col)] + 1,
        start_col,
    )
    start_index = local_column_pointers[0]
    end_index = local_column_pointers[-1]
    self.write_values(matrix.data, start_index)
    # Change from 0-based indexing to 1-based.
    self.write_row_indices(matrix.indices + 1, start_index)


def gather_nnzs(matrix):
  """Returns an array of numbers of non-zero elements in the host blocks of
  `matrix`.

  Each host/Jax process has its own matrix, on the host or on its local asic_node.
  This function computes the number of non-zero elements of each of them,
  all-gathers the results, and returns them as an array with as many elements as
  there are hosts.

  Args:
    matrix: The matrix for which to count non-zero elements.
  Returns:
    all_nnzs: An array, the ith element of which is the number of non-zero
      in the matrix on host i.
  """
  try:
    local_nnz = matrix.nnz
  except AttributeError:
    # If the matrix isn't sparse
    local_nnz = np.count_nonzero(matrix)
  all_nnzs = np.zeros(N_LOCAL_DEVICES, dtype=np.int64)
  all_nnzs[0] = local_nnz  # Put local_nnz on local ASIC #0.
  all_nnzs = jax.pmap(
      lambda x: jax.lax.all_gather(x, axis_name=pops.AXIS_NAME),
      axis_name=pops.AXIS_NAME,
      out_axes=None,
  )(all_nnzs)
  # Pick the elements from local ASIC #0 of each host.
  all_nnzs = np.array(all_nnzs)[0::N_LOCAL_DEVICES]
  return all_nnzs


def _host_grid_change_groups(host_grid_dim):
  "Axis index groups for changing the distribution over hosts." ""
  # Yes, this is opaque. Even to mhauru@ who wrote this. Sorry.
  groups = np.arange(host_grid_dim**2 * N_LOCAL_DEVICES).reshape(
      host_grid_dim, N_LOCAL_DEVICES, host_grid_dim).transpose(1, 0, 2).reshape(
          host_grid_dim, host_grid_dim * N_LOCAL_DEVICES).T
  groups = list(map(list, groups))
  return groups


def _local_grid_change_groups(host_grid_dim):
  "Axis index groups for changing the distribution within asic_nodes." ""
  # Yes, this is opaque. Even to mhauru@ who wrote this. Sorry.
  return [
      [2 * i, 2 * i + 1] for i in range(N_LOCAL_DEVICES // 2 * host_grid_dim**2)
  ]


@functools.partial(pops.pmap, in_axes=1, out_axes=0)
def _distribute_1d_to_distla_core_inner(matrix):
  """The pmapped part of distribute_1d_to_distla_core."""
  host_grid_dim, dim0, dim1 = matrix.shape
  device_labels = np.arange(host_grid_dim**2 * N_LOCAL_DEVICES)
  # Change the distribution over hosts to a square grid.
  axis_index_groups = _host_grid_change_groups(host_grid_dim)
  matrix = jax.lax.all_to_all(
      matrix,
      pops.AXIS_NAME,
      0,
      1,
      axis_index_groups=axis_index_groups,
  )
  matrix = matrix.reshape(dim0, dim1 * host_grid_dim)
  # Change distribution within a asic_node to be a (4, 2) grid.
  matrix = matrix.reshape(dim0, 2, dim1 * host_grid_dim // 2)
  axis_index_groups = _local_grid_change_groups(host_grid_dim)
  matrix = jax.lax.all_to_all(
      matrix,
      pops.AXIS_NAME,
      1,
      0,
      axis_index_groups=axis_index_groups,
  )
  matrix = matrix.reshape(dim0 * 2, dim1 * host_grid_dim // 2)
  # Switch between row-major and column-major grids on the asic_nodes.
  permutation = (0, 2, 4, 6, 1, 3, 5, 7)
  permutation = sum([[j + N_LOCAL_DEVICES * i
                      for j in permutation]
                     for i in range(host_grid_dim**2)], [])
  matrix = jax.lax.pshuffle(matrix, pops.AXIS_NAME, permutation)
  return matrix


def distribute_1d_to_distla_core(matrix):
  """Transform a 1D distributed matrix to the DistlaCore 2D distribution.

  The dimensions of the matrix are expected to be padded so that they are
  divisible by all distribution patterns involved.

  The 1D distribution is natural to use when reading/writing CSC files.

  Args:
    The 1D distributed matrix, as an array on the host.
  Returns:
    The DistlaCore-distributed matrix as a ShardedDeviceArray.
  """
  dim0, dim1 = matrix.shape
  host_grid_dim = pops.NHROWS
  if pops.NHCOLS != host_grid_dim:
    msg = ("distribute_1d_to_distla_core only works with square host grids, "
           f"got {pops.HGRID}")
    raise ValueError(msg)
  matrix = matrix.reshape((host_grid_dim, 8, dim0 // (8 * host_grid_dim), dim1))
  return _distribute_1d_to_distla_core_inner(matrix)


@functools.partial(pops.pmap, in_axes=0, out_axes=1)
def _distribute_distla_core_to_1d_inner(matrix):
  """The pmapped part of distribute_distla_core_to_1d."""
  dim0, dim1 = matrix.shape
  sharded_dim = pops.NPROCS
  host_grid_dim = pops.NHROWS
  device_labels = np.arange(sharded_dim)
  # Switch between row-major and column-major grids on the asic_nodes.
  permutation = (0, 4, 1, 5, 2, 6, 3, 7)
  permutation = sum(
      [[j + 8 * i for j in permutation] for i in range(host_grid_dim**2)], [])
  matrix = jax.lax.pshuffle(matrix, pops.AXIS_NAME, permutation)
  # Change distribution within a asic_node to be 1D row-blocked.
  matrix = matrix.reshape(2, dim0 // 2, dim1)
  axis_index_groups = _local_grid_change_groups(host_grid_dim)
  matrix = jax.lax.all_to_all(
      matrix,
      pops.AXIS_NAME,
      0,
      1,
      axis_index_groups=axis_index_groups,
  )
  matrix = matrix.reshape(dim0 // 2, dim1 * 2)
  # Change the distribution over hosts to be 1D column-blocked.
  matrix = matrix.reshape(dim0 // 2, host_grid_dim, dim1 * 2 // host_grid_dim)
  axis_index_groups = _host_grid_change_groups(host_grid_dim)
  matrix = jax.lax.all_to_all(
      matrix,
      pops.AXIS_NAME,
      1,
      0,
      axis_index_groups=axis_index_groups,
  )
  matrix = matrix.reshape(host_grid_dim, dim0 // (2 * host_grid_dim), dim1 * 2)
  return matrix


def distribute_distla_core_to_1d(matrix):
  """Transform a DistlaCore 2D distributed matrix to a 1D distribution.

  The dimensions of the matrix are expected to be padded so that they are
  divisible by all distribution patterns involved.

  The 1D distribution is natural to use when reading/writing CSC files.

  Args:
    The DistlaCore-distributed matrix as a ShardedDeviceArray.
  Returns:
    The 1D distributed matrix, as an array on the host.
  """
  _, dim0, dim1 = matrix.shape
  host_grid_dim = pops.NHROWS
  if pops.NHCOLS != host_grid_dim:
    msg = ("distribute_distla_core_to_1d only works with square host grids, "
           f"got {pops.HGRID}")
    raise ValueError(msg)
  matrix = _distribute_distla_core_to_1d_inner(matrix)
  matrix = np.array(matrix).reshape(
    dim0 * 4 * host_grid_dim,
    dim1 * 2 // host_grid_dim,
  )
  return matrix


def read_struc_pack_csc(path):
  """Reads an STRUC_PACK CSC file and distributes it over the ASICs.

  In a multihost configuration each host needs access to (a copy of) the full
  CSC file.

  Args:
    path: The path to the STRUC_PACK CSC file.
  Returns:
    matrix: The dense, DistlaCore distributed matrix on the ASICs.
    unpadded_dim: The original dimension of the matrix. Note that all STRUC_PACK CSC
      matrices are square. The matrix that is returned may have extra padding
      with zeros beyond this dimension.
    n_electrons: The number of electrons, which STRUC_PACK writes to the same file.
  """
  host_grid_dim = pops.NHROWS
  if pops.NHCOLS != host_grid_dim:
    msg = "distribute_global_struc_pack_csc only implemented for square host grids."
    raise NotImplementedError(msg)
  csc_file = StructPackCscFile(path)
  # This sets the maximum number of blocks the row and column space of the
  # matrix must be divisible into. The reason this is an LCM of various numbers
  # is that it needs to account for various intermediate reshapes that the I/O
  # functions need to make. Without those, it would just be lcm(*pops.GRID).
  # TODO I'm not sure if this is optimal. It should be on the safe side.
  max_n_blocks = np.lcm.reduce(
      [host_grid_dim**2,
       16 * host_grid_dim,
       2 * max(pops.GRID) * host_grid_dim,
       *pops.GRID,
       ],
  )
  process_index = jax.process_index()
  process_count = jax.process_count()
  matrix_undistributed, _ = csc_file.read_struc_pack_csc_host_block(
      process_index,
      process_count,
      max_n_blocks,
  )
  matrix_undistributed = matrix_undistributed.toarray()
  matrix = distribute_1d_to_distla_core(matrix_undistributed)
  return matrix, csc_file.dim, csc_file.n_electrons


def write_struc_pack_csc(path, matrix, n_electrons=0, unpadded_dim=None):
  """Writes a DistlaCore distributed matrix to disk in the STRUC_PACK CSC format.

  In a multihost configuration with N hosts, each host writes to the file only
  1/N'th of the columns. If each host has access to the same file (e.g. a
  network drive) the file will in the end hold the full matrix. Otherwise each
  host will write its own file with only part of the matrix in it.

  Args:
    path: The path to write to.
    matrix: The matrix.
    n_electrons: Number of electrons, which STRUC_PACK writes to the same file. 0 by
      default.
    unpadded_dim: The dimension of the non-zero part of matrix, excluding
      padding. By default assumed to be the full dimension of matrix.
  Returns:
    None
  """
  _, dim0, dim1 = matrix.shape
  padded_dim = dim0 * pops.NROWS
  if dim1 * pops.NCOLS != padded_dim:
    msg = ("Can't write a non-square matrix "
           f"({(padded_dim, dim1 * pops.NCOLS)}) in the STRUC_PACK CSC format.")
    raise ValueError(msg)
  if unpadded_dim is None:
    unpadded_dim = padded_dim
  matrix_undistributed = distribute_distla_core_to_1d(matrix)
  all_nnzs = gather_nnzs(matrix_undistributed)
  total_nnz = sum(all_nnzs)
  csc_file = StructPackCscFile(
      path,
      dim=unpadded_dim,
      nnz=total_nnz,
      n_electrons=n_electrons,
  )
  process_index = jax.process_index()
  process_count = jax.process_count()
  csc_file.write_struc_pack_csc_host_block(
      matrix_undistributed,
      process_index,
      process_count,
      padded_dim,
      all_nnzs,
  )
