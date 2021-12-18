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
"""Tests for distla_core.io.sparse."""
import os
import multiprocessing
import pathlib
import tempfile

import jax
from jax import numpy as jnp
import numpy as np
import pytest
import scipy as sp

from distla_core.blas.summa import summa
from distla_core.io import sparse
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

# These matrices were written by STRUC_PACK's own I/O module.
THIS_DIR = pathlib.Path(__file__).parent
TEST_MATRIX_PATH = THIS_DIR / "test_matrices/test_matrix01.csc"
TEST_MATRIX_SQUARED_PATH = THIS_DIR / "test_matrices/test_matrix01_squared.csc"

DIMS = (32, 256)
SEEDS = (0,)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("length", (2, 100, 1000))
@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.float64))
@pytest.mark.parametrize("offset", (0, 500, 1200))
@pytest.mark.parametrize("buffer_bytes", (8, 100, 2**30))
def test_read_write_elements(seed, length, dtype, offset, buffer_bytes):
  element_bytes = np.dtype(dtype).itemsize
  rbg = np.random.PCG64(seed)  # Random bit generator
  length_bytes = element_bytes * length
  # Division by 8 because random_raw generates uint64.
  array = np.frombuffer(rbg.random_raw(length_bytes // 8), dtype=dtype)
  assert array.size == length
  with tempfile.TemporaryDirectory() as tempdir:
    path = pathlib.Path(f"{tempdir}/tmpfile")
    f = os.open(path, os.O_RDWR | os.O_SYNC | os.O_CREAT)
    try:
      # Set the full file size.
      os.lseek(f, offset + length_bytes - 1, 0)
      os.write(f, b"\0")
      sparse._write_elements(f, array, offset, buffer_bytes=buffer_bytes)
      array_read = sparse._read_elements(
          f,
          dtype,
          length,
          offset,
          buffer_bytes=buffer_bytes,
      )
    finally:
      os.close(f)
  np.testing.assert_array_equal(array_read, array)


def _host_block_write(args):
  (
      matrix,
      host_index,
      path,
      all_nnzs,
      num_hosts,
      padded_dim,
      unpadded_dim,
      n_electrons,
  ) = args
  total_nnz = sum(all_nnzs)
  csc_file = sparse.StructPackCscFile(
      path,
      dim=unpadded_dim,
      nnz=total_nnz,
      n_electrons=n_electrons,
  )
  csc_file.write_struc_pack_csc_host_block(
      matrix,
      host_index,
      num_hosts,
      padded_dim,
      all_nnzs,
  )
  return None


def _host_block_read(args):
  (
      matrix,
      host_index,
      path,
      all_nnzs,
      num_hosts,
      padded_dim,
      unpadded_dim,
      n_electrons,
  ) = args
  total_nnz = sum(all_nnzs)
  csc_file = sparse.StructPackCscFile(
      path,
      dim=unpadded_dim,
      nnz=total_nnz,
      n_electrons=n_electrons,
  )
  matrix_undistributed, _ = csc_file.read_struc_pack_csc_host_block(
      host_index,
      num_hosts,
      num_hosts,
  )
  matrix_undistributed = matrix_undistributed.toarray()
  return matrix_undistributed


@pytest.mark.parametrize("num_hosts", (1, 4, 8))
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("density", (0.1, 0.95))
@pytest.mark.parametrize("n_electrons", (10,))
@pytest.mark.parametrize("pad", (0, 3))
@pytest.mark.parametrize("seed", SEEDS)
def test_struc_pack_write_read_host_block(
    num_hosts,
    dim,
    density,
    n_electrons,
    pad,
    seed,
):
  """Test reading and writing STRUC_PACK CSC files emulating a multihost setup.

  A multihost setup is emulated using multiprocessing.

  Unlike test_struc_pack_write_read, this one does not include the part about changing
  matrix distributions on the ASICs, since that is hard to emulate without an
  actual asic_cluster slice. Instead, this just writes and reads a matrix between host
  memory and the disk.
  """
  np.random.seed(seed)
  unpadded_dim = dim - pad
  matrix = sp.sparse.random(
      unpadded_dim,
      unpadded_dim,
      density=density,
      dtype=np.float64,
  ).toarray()
  matrix_padded = np.zeros((dim, dim), dtype=np.float64)
  matrix_padded[:unpadded_dim, :unpadded_dim] = matrix
  host_indices = tuple(range(num_hosts))
  host_block_width = dim // num_hosts
  blocks = [
      matrix_padded[:, i * host_block_width:(i + 1) * host_block_width]
      for i in host_indices
  ]
  all_nnzs = [np.count_nonzero(b) for b in blocks]
  with tempfile.TemporaryDirectory() as tempdir:
    path = pathlib.Path(f"{tempdir}/tmp_test_matrix.csc")
    # Arguments to be passed to the multiprocessing calls.
    args = [[b, i, path, all_nnzs, num_hosts, dim, unpadded_dim, n_electrons]
            for b, i in zip(blocks, host_indices)]
    with multiprocessing.Pool(processes=num_hosts) as pool:
      # Wait for writing to be done before reading.
      # TODO The _barriers in sparse.py don't work when using emulated hosts,
      # which may cause this test to fail intermittently. Figure a way around
      # this.
      pool.map_async(_host_block_write, args).wait()
      blocks = pool.map_async(_host_block_read, args)
      blocks.wait()
  blocks = blocks.get()
  matrix_reconstructed = np.hstack(blocks)
  matrix_reconstructed = matrix_reconstructed[:unpadded_dim, :unpadded_dim]
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype=jnp.float32)
  np.testing.assert_allclose(matrix_reconstructed, matrix, rtol=10 * eps)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("density", (0.1, 0.95))
@pytest.mark.parametrize("n_electrons", (10,))
@pytest.mark.parametrize("unpadded_dim", (11, 32))
@pytest.mark.parametrize("seed", SEEDS)
def test_struc_pack_write_read_full(dim, density, n_electrons, unpadded_dim, seed):
  """Create a random matrix, write it in STRUC_PACK CSC format, and read it back."""
  np.random.seed(seed)
  matrix = sp.sparse.random(
      unpadded_dim,
      unpadded_dim,
      density=density,
      dtype=np.float64,
  ).toarray()
  matrix_padded = np.zeros((dim, dim), dtype=np.float64)
  matrix_padded[:unpadded_dim, :unpadded_dim] = matrix
  matrix_distributed = pops.distribute_global(matrix_padded)
  with tempfile.TemporaryDirectory() as tempdir:
    path = pathlib.Path(f"{tempdir}/tmp_test_matrix.csc")
    sparse.write_struc_pack_csc(
        path,
        matrix_distributed,
        n_electrons=n_electrons,
        unpadded_dim=unpadded_dim,
    )
    matrix_read, unpadded_dim_read, n_electrons_read = sparse.read_struc_pack_csc(
        path)
  assert n_electrons_read == n_electrons
  assert unpadded_dim_read == unpadded_dim
  matrix_undistributed = pops.undistribute_global(matrix_read)
  matrix_undistributed = matrix_undistributed[:unpadded_dim, :unpadded_dim]
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype=jnp.float32)
  np.testing.assert_allclose(matrix_undistributed, matrix, rtol=10 * eps)


def test_struc_pack_read_and_square():
  """Read a fixed STRUC_PACK CSC matrix and its square from disk. Check that computing
  the square comes out correct.
  """
  matrix = sparse.read_struc_pack_csc(TEST_MATRIX_PATH)[0]
  matrix_squared_expected = sparse.read_struc_pack_csc(TEST_MATRIX_SQUARED_PATH)[0]
  matrix_squared = pops.pmap(lambda x, y: summa.summa(
      x,
      y,
      p_sz=128,
      transpose_A=False,
      transpose_B=False,
      precision=jax.lax.Precision.HIGHEST,
  ))(matrix, matrix)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype=jnp.float32)
  # These test matrices are 77 x 77, so handling them with numpy is safe.
  norm = np.linalg.norm(matrix_squared_expected)
  np.testing.assert_allclose(
      matrix_squared,
      matrix_squared_expected,
      atol=10 * norm * eps,
  )
