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
"""Tests for pops.py."""
import functools
import jax
import jax.numpy as jnp

import numpy as np
import pytest
from scipy import sparse

from distla_core.linalg.utils import testutils
import distla_core.testing.markers as markers
from distla_core.utils import config
from distla_core.utils import misc
from distla_core.utils import pops

DTYPE = jnp.float32
AXIS_NAME = pops.AXIS_NAME
NROW = pops.NROWS
NCOL = pops.NCOLS
NPROC = NROW * NCOL
GRID = (NROW, NCOL)
PRECISIONS = (
    jax.lax.Precision.DEFAULT,
    jax.lax.Precision.HIGH,
    jax.lax.Precision.HIGHEST,
)

matrix_shapes = [(16, 16), (32, 16), (16, 32)]
seeds = [0, 1]
axis_index_groups_list = [
    None,
    # On a asic_node, the following is [[0, 1], [2, 3], [4, 5], [6, 7]]
    list(map(list,
             np.arange(NPROC).reshape(NPROC // 2, 2))),
    # On a asic_node, the following is [[0, 1, 2, 3], [4, 5, 6, 7]]
    list(map(list,
             np.arange(NPROC).reshape(2, NPROC // 2))),
]
Ns = [8, 16, 32]
ks = [-2, 0, 2]
unpadded_dims = [None, 5]
dtypes = [jnp.float32]
flags = [True, False]


def _local_shape(matrix_shape, device_grid=False):
  grid = pops.DGRID if device_grid else GRID
  m = matrix_shape[0] // grid[0]
  n = matrix_shape[1] // grid[1]
  return m, n


###############################################################################
# DOT
###############################################################################
@pytest.mark.parametrize("A_shape, B_shape", (
    ((16, 16), (16, 16)),
    ((16, 64), (64, 64)),
    ((128, 16), (16, 16)),
    ((128, 16), (16, 128)),
    ((128, 128), (128, 128)),
))
@pytest.mark.parametrize(
    "dtype",
    (jnp.float32, jnp.float64, jnp.complex64),
)
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("seed", (0,))
@pytest.mark.parametrize("ef57_paneling", (True, False))
@testutils.with_x64
def test_dot(A_shape, B_shape, dtype, precision, seed, ef57_paneling):
  """Test that pops.dot matches jnp.dot."""
  # The panel_size_threshold here is stupidly small, but chosen such that many
  # but not all of the combinations of dimensions would trigger it.
  panel_size_threshold = 2**-14
  np.random.seed(seed)
  A = jnp.array(np.random.randn(*A_shape), dtype=dtype)
  B = jnp.array(np.random.randn(*B_shape), dtype=dtype)

  @jax.jit
  def jit_dot(A, B):
    return pops.dot(
        A,
        B,
        precision=precision,
        ef57_paneling=ef57_paneling,
        panel_size_threshold=panel_size_threshold,
    )

  C = jit_dot(A, B)
  C_expected = jnp.dot(A, B, precision=precision)
  tol = 0 if dtype in (jnp.float32, jnp.complex64) else 1e-13
  np.testing.assert_allclose(C, C_expected, rtol=tol, atol=tol)


###############################################################################
# MATRIX DISTRIBUTION
###############################################################################
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_distribute_global_undistribute_global(matrix_shape):
  """
  Checks that undistribute inverts distribute.
  """
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  Ap = pops.distribute_global(A)
  Aup = pops.undistribute_global(Ap)
  np.testing.assert_array_equal(A, Aup)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_distribute_undistribute(matrix_shape):
  """
  Checks that undistribute inverts distribute.
  """
  np.random.seed(0)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  Ap = pops.distribute(A)
  Aup = pops.undistribute(Ap)
  np.testing.assert_equal(A, Aup)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_distribute_sparse_global_undistribute_global(matrix_shape):
  """
  Checks that undistribute inverts distribute sparse.
  """
  np.random.seed(0)
  d0, d1 = matrix_shape
  n_elements = min(d0, d1)
  A_rowinds = np.random.randint(0, d0, (n_elements,))
  A_colinds = np.random.randint(0, d1, (n_elements,))
  A_vals = np.random.randn(n_elements).astype(DTYPE)
  A_sparse = sparse.coo_matrix(
      (A_vals, (A_rowinds, A_colinds)),
      shape=matrix_shape,
  )
  A_dense = A_sparse.todense()
  A_dist = pops.distribute_sparse_global(A_sparse)
  A_collected = pops.undistribute_global(A_dist)
  np.testing.assert_equal(A_dense, A_collected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_undistribute(matrix_shape):
  """
  Checks that undistribute correctly reshapes a manually contructed block
  contiguous matrix.
  """
  mA, nA = _local_shape(matrix_shape, device_grid=True)
  local_shape = (mA, nA)
  A = np.zeros((pops.NDCOLS, pops.NDROWS, *local_shape), dtype=np.int32)
  expected = np.arange(np.prod(matrix_shape)).reshape(*matrix_shape)
  s = 0
  for i in range(pops.NDROWS):
    for a in range(mA):
      for j in range(pops.NDCOLS):
        for b in range(nA):
          A[j, i, a, b] = s
          s = s + 1
  A = A.reshape((pops.NDPROCS, *local_shape))
  A = pops.undistribute(A)
  np.testing.assert_equal(A, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("k_diag", ks)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_eye(matrix_shape, k_diag, unpadded_dim):
  A = jnp.eye(*matrix_shape, dtype=DTYPE, k=k_diag)
  A = misc.apply_pad_serial(A, unpadded_dim)
  local_shape = _local_shape(matrix_shape)
  eye_f = functools.partial(
      pops.eye, local_shape, DTYPE, k=k_diag, unpadded_dim=unpadded_dim)
  ps = jnp.arange(pops.NDPROCS)
  Id = pops.pmap(lambda p: eye_f())(ps)
  Id = pops.undistribute_global(Id)
  np.testing.assert_allclose(A, Id)


###############################################################################
# PROCESSOR ADDRESSING
###############################################################################
def test_myname():
  p = jnp.arange(pops.NDPROCS) + jax.process_index() * pops.NDPROCS
  names = pops.pmap(lambda x: pops.my_name())(p)
  np.testing.assert_allclose(names, p, atol=0)


def test_myprow():
  p = jnp.arange(pops.NDPROCS) + jax.process_index() * pops.NDPROCS
  names = pops.pmap(lambda x: pops.my_prow())(p)
  expected = [pops.your_prow(x) for x in p]
  np.testing.assert_allclose(names, expected, atol=0)


def test_mypcol():
  p = jnp.arange(pops.NDPROCS) + jax.process_index() * pops.NDPROCS
  names = pops.pmap(lambda x: pops.my_pcol())(p)
  expected = [pops.your_pcol(x) for x in p]
  np.testing.assert_allclose(names, expected, atol=0)


def test_in_this_prow():
  p = jnp.arange(pops.NDPROCS) + jax.process_index() * pops.NDPROCS
  prows = np.array([pops.your_prow(x) for x in p])
  for pi in range(NROW):
    names = pops.pmap(lambda x: pops.in_this_prow(prow=pi))(p)
    expected = prows == pi
    np.testing.assert_allclose(names, expected, atol=0)


def test_in_this_pcol():
  p = jnp.arange(pops.NDPROCS) + jax.process_index() * pops.NDPROCS
  cores = np.array([pops.your_pcol(x) for x in p])
  for pi in range(NCOL):
    names = pops.pmap(lambda x: pops.in_this_pcol(pcol=pi))(p)
    expected = cores == pi
    np.testing.assert_allclose(names, expected, atol=0)


def test_your_prow():
  host_id = jax.process_index()
  ps = jnp.arange(pops.NDPROCS) + host_id * pops.NDPROCS
  cores = [pops.your_prow(p) for p in ps]
  expected = ps % pops.NDROWS + pops.NDROWS * (host_id % pops.NHROWS)
  np.testing.assert_allclose(cores, expected, atol=0)


def test_your_pcol():
  host_id = jax.process_index()
  ps = jnp.arange(pops.NDPROCS) + host_id * pops.NDPROCS
  cores = [pops.your_pcol(p) for p in ps]
  expected = (ps % pops.NDPROCS) // pops.NDROWS + pops.NDCOLS * (
      host_id // pops.NHROWS)
  np.testing.assert_allclose(cores, expected, atol=0)


###############################################################################
# PROCESSOR MASKS
###############################################################################
def _initialize_proc_test(matrix_shape, host_id=jax.process_index()):
  n_elements = np.prod(matrix_shape)
  mat = np.arange(n_elements).reshape(*matrix_shape).astype(DTYPE)
  mat += host_id * n_elements
  mat_p = pops.distribute(mat)
  mat_t = pops.distribute(mat, pmap=False)
  local_shape = mat_t.shape[1:]
  mat_t = mat_t.reshape((pops.NDCOLS, pops.NDROWS, *local_shape))
  return mat_p, mat_t, local_shape


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("prow", [0, 1])
def test_mask_except_prow(matrix_shape, prow):
  A_p, expected, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.mask_except_prow, prow=prow)
  out = pops.pmap(f)(A_p)
  host_row = jax.process_index() % pops.NHROWS
  for i in range(pops.NDROWS):
    if i + pops.NDROWS * host_row != prow:
      expected[:, i, ...] = np.zeros_like(expected[:, i, ...])
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("pcol", [0, 1])
def test_mask_except_pcol(matrix_shape, pcol):
  A_p, expected, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.mask_except_pcol, pcol=pcol)
  out = pops.pmap(f)(A_p)
  host_col = jax.process_index() // pops.NHROWS
  for i in range(pops.NDCOLS):
    if i + pops.NDCOLS * host_col != pcol:
      expected[i, ...] = np.zeros_like(expected[i, ...])
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("k", ks)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_mask_off_diagonal(matrix_shape, k, unpadded_dim):
  np.random.seed(10)
  A = np.random.rand(*matrix_shape)
  rows = np.arange(A.shape[0])
  cols = np.arange(A.shape[1])
  mask_condition = rows[:, None] == cols - k
  expected = np.where(mask_condition, A, np.zeros_like(A))
  expected = misc.apply_pad_serial(jnp.array(expected), unpadded_dim)
  Ap = pops.distribute_global(A)
  fill_f = functools.partial(
      pops.mask_off_diagonal, k=k, unpadded_dim=unpadded_dim)
  result = pops.pmap(fill_f)(Ap)
  result = pops.undistribute_global(result)
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_apply_pad(matrix_shape, unpadded_dim):
  np.random.seed(10)
  matrix = jnp.array(np.random.rand(*matrix_shape).astype(np.float32))
  expected = misc.apply_pad_serial(matrix, unpadded_dim)
  matrix = pops.distribute_global(matrix)
  pad_f = functools.partial(pops.apply_pad, unpadded_dim=unpadded_dim)
  result = pops.pmap(pad_f)(matrix)
  result = pops.undistribute_global(result)
  np.testing.assert_array_equal(result, expected)


###############################################################################
# SLICING
###############################################################################
@pytest.mark.parametrize("matrix_shape", [[8, 2], [8, 8], [16, 16], [64, 64]])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("offset", [0, 1, 4])
@pytest.mark.parametrize("n_rows", [1, 2, 4])
def test_get_rows(matrix_shape, seed, offset, n_rows):
  """ Tests that get_rows correctly extracts the specified block.
  """
  np.random.seed(seed)
  m, n = matrix_shape
  A = np.random.randn(m, n).astype(DTYPE)
  expected = A[offset:offset + n_rows, :]
  A_d = pops.distribute(A)

  @functools.partial(
    pops.pmap, in_axes=(0, None, None), static_broadcasted_argnums=(2,))
  def _test_f(A, offset, n_rows):
    return pops.get_rows(A, offset, n_rows)

  if n_rows > A_d.shape[1]:
    with pytest.raises(TypeError):
      result = _test_f(A_d, offset, n_rows)
  else:
    result = _test_f(A_d, offset, n_rows)
    result = pops.undistribute(result)
    result_top = result[:n_rows, :]
    for row_idx in range(n_rows, n_rows * pops.NROWS, n_rows):
      this_stripe = result[row_idx:row_idx + n_rows, :]
      np.testing.assert_array_equal(result_top, this_stripe)
    np.testing.assert_array_equal(result_top, expected)


@pytest.mark.parametrize("matrix_shape", [[8, 2], [8, 8], [16, 16], [64, 64]])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("offset", [0, 1, 4])
@pytest.mark.parametrize("n_rows", [1, 2, 4])
def test_set_rows(matrix_shape, seed, offset, n_rows):
  """ Tests that get_rows correctly sets the specified block.
  """
  np.random.seed(seed)
  m, n = matrix_shape
  A = np.random.randn(m, n).astype(DTYPE)
  rows = np.random.randn(n_rows, n).astype(DTYPE)
  row_update = np.zeros((n_rows * pops.NROWS, n), dtype=DTYPE)
  for row_i in range(0, n_rows * pops.NROWS, n_rows):
    row_update[row_i: row_i + n_rows, :] = rows

  expected = np.copy(A)
  expected[offset: offset + n_rows, :] = rows

  A_d = pops.distribute(A)
  row_update = pops.distribute(row_update)

  @functools.partial(pops.pmap, in_axes=(0, 0, None))
  def _test_f(A, panel, offset):
    return pops.set_rows(A, panel, offset)

  if n_rows > A_d.shape[1]:
    with pytest.raises(TypeError):
      result = _test_f(A_d, row_update, offset)
  else:
    result = _test_f(A_d, row_update, offset)
    result = pops.undistribute(result)
    np.testing.assert_array_equal(result, expected)


###############################################################################
# MESSAGE TUPLES
###############################################################################
@markers.effective_asic_node_only
def test_axis_index_pcols():
  out = pops._axis_index_pcols()
  expected = ((0, 1, 2, 3), (4, 5, 6, 7))
  assert out == expected


@markers.effective_asic_node_only
def test_axis_index_prows():
  out = pops._axis_index_prows()
  expected = ((0, 4), (1, 5), (2, 6), (3, 7))
  assert out == expected


@markers.effective_asic_node_only
@pytest.mark.parametrize("start_from_zero", flags)
def test_axis_index_prow_pairs(start_from_zero):
  out = pops._axis_index_prow_pairs(start_from_zero)
  if start_from_zero:
    expected = ((0, 1), (2, 3), (4, 5), (6, 7))
  else:
    expected = ((0, 3), (1, 2), (4, 7), (5, 6))
  assert out == expected


row_in_8 = ((0, 1), (2, 3))
row_out_8 = ((0, 1), (2, 3), (4, 5), (6, 7))
col_in_8 = ((1, 0),)
col_out_8 = ((4, 0), (5, 1), (6, 2), (7, 3))
params_8 = [8, (row_in_8, True, row_out_8), (col_in_8, False, col_out_8)]

row_in_32 = ((0, 1), (2, 3), (4, 5), (6, 7))
row_out_32 = ((0, 1), (2, 3), (4, 5), (6, 7))
col_in_32 = ((2, 0), (1, 3))
col_out_32 = ((4, 0), (5, 1), (6, 2), (7, 3))
col_out_32 = ((16, 0), (17, 1), (18, 2), (19, 3), (24, 8), (25, 9), (26, 10),
              (27, 11), (4, 20), (5, 21), (6, 22), (7, 23), (12, 28), (13, 29),
              (14, 30), (15, 31))
params_32 = [32, (row_in_32, True, row_out_32), (col_in_32, False, col_out_32)]

@pytest.mark.parametrize("params", [params_8, params_32])
def test_axis_index_paxis_groups(params):
  """ Forms pairs ((0, 1), (1, 2), ...) and tests
  _axis_index_paxis_groups handles them correctly.
  """
  cores, row_tup, col_tup = params
  if cores != pops.NPROCS:
    pytest.skip()
  for tup in [row_tup, col_tup]:
    groups, flag, expected = tup
    result = pops._paxis_groups_to_linear(groups, rows=flag)
    assert set(result) == set(expected)


###############################################################################
# COMMUNICATION
###############################################################################
@markers.effective_asic_node_only
@pytest.mark.parametrize("prow", [0, 1])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_broadcast_prow(matrix_shape, prow):
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.broadcast_prow, prow=prow)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for i in range(NROW):
    expected[:, i, ...] = mat_t[:, prow, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("groups", [((0, 1), (2, 3)), ((0, 1, 2, 3),)])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("seed", [0, 1])
def test_broadcast_prows_to_groups(matrix_shape, groups, seed):
  np.random.seed(seed)
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  n_groups = len(groups)
  group_size = len(groups[0])
  idxs = np.random.randint(0, high=group_size, size=n_groups)
  f = functools.partial(
      pops.broadcast_paxis_to_groups,
      bcast_indices=idxs,
      groups=groups,
      rows=True)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for idx, group in zip(idxs, groups):
    head = group[idx]
    for prow in group:
      expected[:, prow, ...] = mat_t[:, head, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("groups", [((0, 1),), ((1, 0),)])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("seed", [0, 1])
def test_broadcast_pcols_to_groups(matrix_shape, groups, seed):
  np.random.seed(seed)
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  n_groups = len(groups)
  group_size = len(groups[0])
  idxs = np.random.randint(0, high=group_size, size=n_groups)
  f = functools.partial(
      pops.broadcast_paxis_to_groups,
      bcast_indices=idxs,
      groups=groups,
      rows=False)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for idx, group in zip(idxs, groups):
    head = group[idx]
    for pcol in group:
      expected[pcol, ...] = mat_t[head, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("pcol", [0, 1])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_broadcast_pcol(matrix_shape, pcol):
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.broadcast_pcol, pcol=pcol)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for i in range(NCOL):
    expected[i, ...] = mat_t[pcol, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_gather_to_asic_nodes(matrix_shape):
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  Ap = pops.distribute_global(A)
  Ag = pops.pmap(pops.gather_to_asic_nodes)(Ap)
  if jax.process_count() == 1:
    # On a single host gather_to_asic_nodes should be a no-op.
    np.testing.assert_array_equal(Ag, Ap)
  # undistribute assumes asic_node-distribution, so after gather_to_asic_nodes it
  # should return the full original matrix.
  Aup = pops.undistribute(Ag)
  np.testing.assert_array_equal(Aup, A)


###############################################################################
# REDUCTION
###############################################################################
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("axis_index_groups", axis_index_groups_list)
def test_tree_psum(matrix_shape, dtype, axis_index_groups):
  mat_p, _, _ = _initialize_proc_test(matrix_shape)
  expected = pops.pmap(lambda x: jax.lax.psum(
      x,
      AXIS_NAME,
      axis_index_groups=axis_index_groups,
  ))(mat_p)
  result = pops.pmap(
      pops.tree_psum,
      static_broadcasted_argnums=(1, 2),
  )(mat_p, AXIS_NAME, axis_index_groups)
  np.testing.assert_array_equal(result, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_sum_over_prows(matrix_shape):
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.sum_over_prows)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for i in range(NCOL):
    for j in range(NCOL):
      expected[i, ...] += mat_t[j, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_sum_over_pcols(matrix_shape):
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.sum_over_pcols)
  out = pops.pmap(f)(mat_p)
  expected = np.zeros_like(mat_t)
  for i in range(NROW):
    for j in range(NROW):
      expected[:, i, ...] += mat_t[:, j, ...]
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  np.testing.assert_allclose(out, expected)


@markers.effective_asic_node_only
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_sum_prow_pairs(matrix_shape):
  mat_p, mat_t, local_shape = _initialize_proc_test(matrix_shape)
  f = functools.partial(pops.sum_prow_pairs)
  out = pops.pmap(f)(mat_p)
  expected = np.copy(mat_t)
  expected = expected.reshape((pops.NDPROCS, *local_shape))
  for i in range(0, NPROC, 2):
    core0 = np.copy(expected[i, ...])
    core1 = np.copy(expected[i + 1, ...])
    expected[i, ...] += core1
    expected[i + 1, ...] += core0
  expected = expected.reshape((NPROC, *local_shape))
  np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_frobnorm(matrix_shape):
  np.random.seed(10)
  A = np.random.rand(*matrix_shape)
  expected = np.linalg.norm(A)
  expected = np.zeros(pops.NDPROCS, dtype=A.dtype) + expected
  A_p = pops.distribute_global(A)
  result = pops.pmap(pops.frobnorm)(A_p)
  tol = jnp.finfo(DTYPE).eps * A.size
  np.testing.assert_allclose(result, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("dtype", dtypes)
def test_gershgorin(N, dtype):
  np.random.seed(10)
  A = jnp.array(np.random.rand(N, N)).astype(dtype)
  A = 0.5 * (A + A.conj().T)
  A_p = pops.distribute_global(A)
  result_min, result_max = pops.pmap(pops.gershgorin)(A_p)
  expected_min, expected_max = misc.gershgorin(A)
  tol = jnp.finfo(dtype).eps
  np.testing.assert_allclose(result_min[0], expected_min, atol=tol, rtol=tol)
  np.testing.assert_allclose(result_max[0], expected_max, atol=tol, rtol=tol)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_trace(matrix_shape):
  np.random.seed(10)
  A = np.random.rand(*matrix_shape)
  expected = np.trace(A)
  expected = np.zeros(pops.NDPROCS, dtype=A.dtype) + expected
  A_p = pops.distribute_global(A)
  trace_f = functools.partial(pops.trace)
  result = pops.pmap(trace_f)(A_p)
  tol = jnp.finfo(A.dtype).eps * A.size
  np.testing.assert_allclose(result, expected, atol=tol)


@markers.effective_asic_node_only
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("start_from_zero", flags)
def test_gather_prow_pairs(matrix_shape, start_from_zero):
  np.random.seed(1)
  A = np.random.randn(*matrix_shape).astype(np.float32)
  A_d = pops.distribute(A)
  N = matrix_shape[1]
  m_l, n_l = A_d.shape[1:]

  A_d_split = np.array(A_d).reshape((pops.NDCOLS, pops.NDROWS, m_l, n_l))
  A_d_split = A_d_split.transpose((1, 2, 0, 3)).reshape((pops.NDROWS, m_l, N))
  expected = np.zeros((pops.NDROWS, 2 * m_l, pops.NDCOLS * n_l))
  if start_from_zero:
    pairs = ((0, 1), (2, 3))
  else:
    pairs = ((1, 2), (0, 3))
  for pair in pairs:
    piece_0 = A_d_split[pair[0], :, :].reshape((m_l, N))
    piece_1 = A_d_split[pair[1], :, :].reshape((m_l, N))
    paired = np.vstack((piece_0, piece_1))
    expected[pair[0], :, :] = paired
    expected[pair[1], :, :] = paired
  expected = expected.reshape((pops.NDROWS * 2 * m_l, pops.NDCOLS * n_l))

  @pops.pmap
  def test_f(x):
    return pops.gather_prow_pairs(x, start_from_zero=start_from_zero)

  result = test_f(A_d)
  result = pops.undistribute(result)
  expected = expected.astype(result.dtype)
  testutils.assert_allclose(expected, result, atol=jnp.finfo(A.dtype).eps)


@markers.effective_asic_node_only
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rows", [True, False])
@pytest.mark.parametrize("shift", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_roll_paxis(matrix_shape, shift, rows, seed):
  """ Compares the result of roll_paxis against manually rolled data.
  """
  dtype = np.float32
  np.random.seed(seed)
  A = np.random.randn(*matrix_shape).astype(dtype)
  A_d = pops.distribute(A)
  m_b, n_b = A_d.shape[1:]
  A_copy = A_d.reshape((pops.NDCOLS, pops.NDROWS, m_b, n_b))
  axis = 0
  if rows:
    axis = 1
  expected = jnp.roll(A_copy, shift, axis).reshape((pops.NDPROCS, m_b, n_b))
  expected = pops.undistribute(expected)

  @pops.pmap
  def test_f(x):
    return pops.roll_paxis(x, shift, rows)
  result = test_f(A_d)
  result = pops.undistribute(result)
  atol = jnp.finfo(A.dtype).eps
  testutils.assert_allclose(expected, result, atol=atol)


@markers.effective_asic_node_only
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_vstack_equal_shape(matrix_shape, seed):
  """ Compares the result of vstack_equal_shape with a manually constructed
  equivalent.
  """
  dtype = np.float32
  np.random.seed(seed)
  A = np.random.randn(*matrix_shape).astype(dtype)
  B = np.random.randn(*matrix_shape).astype(dtype)
  expected = np.vstack((A, B))
  A_d = pops.distribute(A)
  B_d = pops.distribute(B)

  @pops.pmap
  def test_f(A, B):
    return pops.vstack_equal_shape(A, B)
  result = test_f(A_d, B_d)
  result = pops.undistribute(result)
  atol = jnp.finfo(A.dtype).eps
  testutils.assert_allclose(expected, result, atol=atol)

###############################################################################
# INDEXING AND MAIN DIAGONAL
###############################################################################
@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_within_unpadded_block(matrix_shape, unpadded_dim):
  expected = jnp.ones(matrix_shape, dtype=np.bool)
  expected = misc.apply_pad_serial(expected, unpadded_dim)
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(
      pops.pmap, static_broadcasted_argnums=(1,), in_axes=(0, None, None))
  def my_within_unpadded_block(ps, local_shape, unpadded_dim):
    return pops.within_unpadded_block(local_shape, unpadded_dim)

  local_shape = _local_shape(matrix_shape)
  result = my_within_unpadded_block(ps, local_shape, unpadded_dim)
  result = pops.undistribute_global(result)
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("k_diag", ks)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_on_kth_diagonal(matrix_shape, k_diag, unpadded_dim):
  expected = jnp.eye(*matrix_shape, dtype=np.bool, k=k_diag)
  expected = misc.apply_pad_serial(expected, unpadded_dim)
  local_shape = _local_shape(matrix_shape)
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(
      pops.pmap,
      static_broadcasted_argnums=(1,), in_axes=(0, None, None, None))
  def my_on_kth_diagonal(ps, local_shape, k, unpadded_dim):
    return pops.on_kth_diagonal(local_shape, k=k, unpadded_dim=unpadded_dim)

  result = my_on_kth_diagonal(ps, local_shape, k_diag, unpadded_dim)
  result = pops.undistribute_global(result)
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("k_diag", ks)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_fill_diagonal(matrix_shape, k_diag, unpadded_dim):
  value = -4.0
  expected = jnp.eye(*matrix_shape, dtype=DTYPE, k=k_diag) * value
  expected = misc.apply_pad_serial(expected, unpadded_dim)

  A = np.zeros(matrix_shape, dtype=DTYPE)
  Ap = pops.distribute_global(A)
  fill_f = functools.partial(
      pops.fill_diagonal, value=value, k=k_diag, unpadded_dim=unpadded_dim)
  A = pops.pmap(fill_f)(Ap)
  A = pops.undistribute_global(A)
  np.testing.assert_allclose(A, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("k_diag", ks)
@pytest.mark.parametrize("unpadded_dim", unpadded_dims)
def test_add_to_diagonal(matrix_shape, k_diag, unpadded_dim):
  value = -4.0
  expected = jnp.eye(*matrix_shape, dtype=DTYPE, k=k_diag) * value
  expected = misc.apply_pad_serial(expected, unpadded_dim)

  A = np.zeros(matrix_shape, dtype=DTYPE)
  Ap = pops.distribute_global(A)
  fill_f = functools.partial(
      pops.add_to_diagonal, value=value, k=k_diag, unpadded_dim=unpadded_dim)
  A = pops.pmap(fill_f)(Ap)
  A = pops.undistribute_global(A)
  np.testing.assert_allclose(A, expected)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_indices_vec(matrix_shape):
  local_shape = _local_shape(matrix_shape)
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(pops.pmap, static_broadcasted_argnums=(1,))
  def indices_vec(p, local_shape):
    return pops.indices_vec(local_shape)

  result_row_vec, result_col_vec = indices_vec(ps, local_shape)
  processor_grid = config.get_processor_grid()
  host_id = jax.process_index()
  host_row = host_id % pops.NHROWS
  host_col = host_id // pops.NHROWS
  for i in range(host_row * pops.NDROWS, (host_row + 1) * pops.NDROWS):
    rows_expected = np.arange(local_shape[0]) + i * local_shape[0]
    for j in range(host_col * pops.NDCOLS, (host_col + 1) * pops.NDCOLS):
      cols_expected = np.arange(local_shape[1]) + j * local_shape[1]
      this_proc = processor_grid[i, j]
      result_index = this_proc - host_id * pops.NDPROCS
      np.testing.assert_array_equal(
          result_row_vec[result_index, :],
          rows_expected,
      )
      np.testing.assert_array_equal(
          result_col_vec[result_index, :],
          cols_expected,
      )


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
def test_indices(matrix_shape):
  local_shape = _local_shape(matrix_shape)
  indices_f = functools.partial(pops.indices, local_shape)
  ps = jnp.arange(pops.NDPROCS)
  rows_t, cols_t = pops.pmap(lambda p: indices_f())(ps)
  rows_t = pops.undistribute_global(rows_t)
  cols_t = pops.undistribute_global(cols_t)
  rows_v = jnp.arange(matrix_shape[0])
  cols_v = jnp.arange(matrix_shape[1])
  cols, rows = jnp.meshgrid(cols_v, rows_v)
  np.testing.assert_allclose(cols, cols_t, atol=0.)
  np.testing.assert_allclose(rows, rows_t, atol=0.)


@markers.asic_node_only  # The spoofed all-to-all raises scary warnings.
@pytest.mark.parametrize("N_local", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("seed", np.arange(10))
def test_all_to_all(N_local, seed):
  np.random.seed(seed)
  N_global = int(np.log2(jax.local_device_count()))
  N = N_global + N_local
  shape = (2,) * N
  grid = (2,) * N_global
  dist_shape = (jax.local_device_count(),) + (2,) * N_local
  nparray = np.random.rand(*shape)
  array = jax.pmap(
      lambda x: x, devices=jax.local_devices())(nparray.reshape(dist_shape))

  sharded_axis = np.random.randint(0, N_global)
  split_axis = np.random.randint(0, N_local)
  concat_axis = split_axis
  result = pops.pmap(
      pops.all_to_all, static_broadcasted_argnums=(1, 2, 3, 4))(
          array, sharded_axis, split_axis, concat_axis, grid)

  actual = np.array(result).reshape(shape)
  perm = np.arange(N)
  t = perm[sharded_axis]
  perm[sharded_axis] = perm[N_global + split_axis]
  perm[N_global + split_axis] = t

  exp = np.squeeze(nparray.transpose(perm))
  np.testing.assert_allclose(actual, exp)


@markers.asic_node_only  # the spoofed all-to-all raises scary warnings.
@pytest.mark.parametrize("N_local", [5, 6, 7])
@pytest.mark.parametrize("grid, num_swapped", [((2, 2, 2), 1), ((2, 2, 2), 2),
                                               ((2, 2, 2), 3), ((4, 2), 1),
                                               ((4, 2), 2), ((2, 4), 1),
                                               ((2, 4), 2), ((8,), 1)])
@pytest.mark.parametrize("seed", np.arange(10))
def test_all_to_all_general(N_local, grid, num_swapped, seed):
  np.random.seed(seed)
  N_global = len(grid)
  sharded_axes = np.sort(
      np.random.choice(np.arange(N_global), size=num_swapped, replace=False))
  split_axes = np.sort(
      np.random.choice(np.arange(N_local), size=num_swapped, replace=False))
  concat_axes = np.random.choice(
      np.arange(N_local),
      size=num_swapped,
      replace=False,
  )
  shape = np.array(grid + (2,) * N_local)
  shape[split_axes + len(grid)] = np.asarray(grid)[sharded_axes]
  shape = tuple(shape)
  dist_shape = (jax.local_device_count(),) + shape[N_global:]
  nparray = np.random.rand(*shape)
  array = jax.pmap(
      lambda x: x, devices=jax.local_devices())(nparray.reshape(dist_shape))

  result = pops.pmap(
      pops.all_to_all, static_broadcasted_argnums=(1, 2, 3, 4))(
          array, sharded_axes, split_axes, concat_axes, grid)
  sharded_axes = tuple(sharded_axes)
  split_axes = tuple(split_axes)
  concat_axes = tuple(concat_axes)

  source = sharded_axes + tuple([s + len(grid) for s in split_axes])
  dest = tuple([s + len(grid) for s in split_axes]) + sharded_axes
  source2 = tuple([s + len(grid) for s in split_axes])
  dest2 = tuple([s + len(grid) for s in concat_axes])
  exp = np.moveaxis(np.moveaxis(nparray, source, dest), source2, dest2)
  actual = np.array(result).reshape(exp.shape)
  np.testing.assert_allclose(actual, exp)


###############################################################################
# MATRIX OPERATIONS
###############################################################################
@pytest.mark.parametrize("N", Ns)
def test_symmetrize(N):
  matrix = np.arange(N**2).reshape((N, N))
  expected = 0.5 * (matrix + matrix.conj().T)
  result = pops.pmap(pops.symmetrize)(pops.distribute_global(matrix))
  np.testing.assert_allclose(expected, pops.undistribute_global(result))


@pytest.mark.parametrize("N", Ns)
def test_transpose(N):
  matrix = np.arange(N**2).reshape((N, N))
  expected = matrix.T
  matrix = pops.distribute_global(matrix)
  matrix_T = pops.pmap(pops.transpose)(matrix)
  result = pops.undistribute_global(matrix_T)
  np.testing.assert_array_equal(result, expected)
