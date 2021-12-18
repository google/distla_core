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
"""Test for vops.py."""
import functools
import operator

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.utils import testutils
from distla_core.utils import pops
from distla_core.utils import vops

DTYPE = jnp.float32
dtypes = [jnp.float32]
AXIS_NAME = pops.AXIS_NAME
precisions = [lax.Precision.HIGHEST]

shapes = [(8, 1), (16, 16), (8, 128), (128, 8)]
flags = [True, False]
seeds = [0, 1]


###############################################################################
# REPLICATEDTHINMATRIX
###############################################################################
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
def test_replicatedthinmatrix_unary(shape, column_replicate, seed):
  """
  Test methods of ReplicatedThinMatrix that only operate on a single matrix.
  """
  A_jnp = jax.random.normal(jax.random.PRNGKey(seed), shape)
  A = vops.distribute(A_jnp, column_replicate)
  np.testing.assert_allclose(A_jnp + 2, vops.undistribute(A + 2))
  np.testing.assert_allclose(2 + A_jnp, vops.undistribute(2 + A))
  np.testing.assert_allclose(A_jnp - 2, vops.undistribute(A - 2))
  np.testing.assert_allclose(2 - A_jnp, vops.undistribute(2 - A))
  np.testing.assert_allclose(A_jnp * 2, vops.undistribute(A * 2))
  np.testing.assert_allclose(2 * A_jnp, vops.undistribute(2 * A))
  np.testing.assert_allclose(A_jnp / 2, vops.undistribute(A / 2))
  np.testing.assert_allclose(2 / A_jnp, vops.undistribute(2 / A))
  np.testing.assert_allclose(A_jnp // 2, vops.undistribute(A // 2))
  np.testing.assert_allclose(2 // A_jnp, vops.undistribute(2 // A))
  np.testing.assert_allclose(A_jnp % 2, vops.undistribute(A % 2))
  np.testing.assert_allclose(2 % A_jnp, vops.undistribute(2 % A))
  np.testing.assert_allclose(A_jnp**2, vops.undistribute(A**2))
  np.testing.assert_allclose(2**A_jnp, vops.undistribute(2**A))

  np.testing.assert_allclose(A_jnp == 0, vops.undistribute(A == 0))
  np.testing.assert_allclose(0 == A_jnp, vops.undistribute(0 == A))
  np.testing.assert_allclose(A_jnp != 0, vops.undistribute(A != 0))
  np.testing.assert_allclose(0 != A_jnp, vops.undistribute(0 != A))
  np.testing.assert_allclose(A_jnp < 0, vops.undistribute(A < 0))
  np.testing.assert_allclose(0 < A_jnp, vops.undistribute(0 < A))
  np.testing.assert_allclose(A_jnp > 0, vops.undistribute(A > 0))
  np.testing.assert_allclose(0 > A_jnp, vops.undistribute(0 > A))
  np.testing.assert_allclose(A_jnp <= 0, vops.undistribute(A <= 0))
  np.testing.assert_allclose(0 <= A_jnp, vops.undistribute(0 <= A))
  np.testing.assert_allclose(A_jnp >= 0, vops.undistribute(A >= 0))
  np.testing.assert_allclose(0 >= A_jnp, vops.undistribute(0 >= A))

  np.testing.assert_allclose(
      operator.neg(A_jnp),
      vops.undistribute(operator.neg(A)),
  )
  np.testing.assert_allclose(
      operator.pos(A_jnp),
      vops.undistribute(operator.pos(A)),
  )
  np.testing.assert_allclose(abs(A_jnp), vops.undistribute(abs(A)))
  np.testing.assert_allclose(jnp.conj(A_jnp), vops.undistribute(A.conj()))
  np.testing.assert_allclose(jnp.sqrt(A_jnp), vops.undistribute(A.sqrt()))
  np.testing.assert_allclose(jnp.sign(A_jnp), vops.undistribute(A.sign()))
  np.testing.assert_allclose(jnp.log(A_jnp), vops.undistribute(A.log()))
  np.testing.assert_allclose(jnp.exp(A_jnp), vops.undistribute(A.exp()))
  np.testing.assert_allclose(jnp.imag(A_jnp), vops.undistribute(A.imag()))
  np.testing.assert_allclose(jnp.real(A_jnp), vops.undistribute(A.real()))
  np.testing.assert_allclose(jnp.min(A_jnp), A.min())
  np.testing.assert_allclose(jnp.max(A_jnp), A.max())

  np.testing.assert_allclose(
      jnp.zeros(shape),
      vops.undistribute(A.zeros_like()),
  )
  np.testing.assert_allclose(
      jnp.ones(shape),
      vops.undistribute(A.ones_like()),
  )
  np.testing.assert_allclose(
      jnp.full(shape, 3),
      vops.undistribute(A.full_like(3)),
  )

  np.testing.assert_allclose(jnp.all(A_jnp > 0), (A > 0).all())
  np.testing.assert_allclose(jnp.any(A_jnp > 0), (A > 0).any())


@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate_A", flags)
@pytest.mark.parametrize("column_replicate_B", flags)
def test_replicatedthinmatrix_binary(
    seed,
    shape,
    column_replicate_A,
    column_replicate_B,
):
  """
  Test methods of ReplicatedThinMatrix that operate on two matrices.
  """
  A_jnp = jax.random.normal(jax.random.PRNGKey(seed), shape)
  A = vops.distribute(A_jnp, column_replicate_A)
  B_jnp = jax.random.normal(jax.random.PRNGKey(seed + 1), shape)
  B = vops.distribute(B_jnp, column_replicate_B)
  np.testing.assert_allclose(A_jnp + B_jnp, vops.undistribute(A + B))
  np.testing.assert_allclose(A_jnp - B_jnp, vops.undistribute(A - B))
  np.testing.assert_allclose(A_jnp * B_jnp, vops.undistribute(A * B))
  np.testing.assert_allclose(A_jnp / B_jnp, vops.undistribute(A / B))
  np.testing.assert_allclose(A_jnp // B_jnp, vops.undistribute(A // B))
  np.testing.assert_allclose(A_jnp % B_jnp, vops.undistribute(A % B))
  np.testing.assert_allclose(A_jnp**B_jnp, vops.undistribute(A**B))

  np.testing.assert_allclose(A_jnp == B_jnp, vops.undistribute(A == B))
  np.testing.assert_allclose(A_jnp != B_jnp, vops.undistribute(A != B))
  np.testing.assert_allclose(A_jnp < B_jnp, vops.undistribute(A < B))
  np.testing.assert_allclose(A_jnp > B_jnp, vops.undistribute(A > B))
  np.testing.assert_allclose(A_jnp <= B_jnp, vops.undistribute(A <= B))
  np.testing.assert_allclose(A_jnp >= B_jnp, vops.undistribute(A >= B))
  np.testing.assert_allclose(jnp.allclose(A_jnp, B_jnp), A.allclose(B))


###############################################################################
# INITIALIZATION
###############################################################################
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
def test_random(seed, shape, column_replicate):
  random = vops.random(
      shape,
      column_replicated=column_replicate,
      key_seed=seed,
  )
  data = random.array.reshape((*pops.DGRID, *random.array.shape[1:]))
  eps = jnp.finfo(data.dtype).eps
  if column_replicate:
    for i in range(pops.DGRID[1]):
      testutils.assert_allclose(data[:, i, :, :], data[:, 0, :, :], atol=eps)
  else:
    for i in range(pops.DGRID[0]):
      testutils.assert_allclose(data[i, :, :, :], data[0, :, :, :], atol=eps)

  assert random.is_column_replicated == column_replicate


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
def test_zeros(shape, column_replicate):
  dtype = jnp.float32
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
  def test_f(ps, shape, dtype, column_rep):
    return vops.zeros(shape, dtype, column_rep)
  result = test_f(ps, shape, dtype, column_replicate)
  assert result.is_column_replicated == column_replicate
  result = vops.undistribute(result)
  expected = jnp.zeros(shape, dtype=dtype)
  testutils.assert_allclose(
    expected, result, atol=jnp.finfo(result.dtype).eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
def test_ones(shape, column_replicate):
  dtype = jnp.float32
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
  def test_f(ps, shape, dtype, column_rep):
    return vops.ones(shape, dtype, column_rep)
  result = test_f(ps, shape, dtype, column_replicate)
  assert result.is_column_replicated == column_replicate
  result = vops.undistribute(result)
  expected = jnp.ones(shape, dtype=dtype)
  testutils.assert_allclose(
    expected, result, atol=jnp.finfo(result.dtype).eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
def test_full(shape, column_replicate):
  val = 3.0
  dtype = jnp.float32
  ps = jnp.arange(pops.NDPROCS)

  @functools.partial(
      pops.pmap,
      static_broadcasted_argnums=(1, 3, 4),
      in_axes=(0, None, None, None, None))
  def test_f(ps, shape, val, dtype, column_rep):
    return vops.full(shape, val, dtype, column_rep)

  result = test_f(ps, shape, val, dtype, column_replicate)
  assert result.is_column_replicated == column_replicate
  result = vops.undistribute(result)
  expected = jnp.full(shape, val, dtype=dtype)
  testutils.assert_allclose(
    expected, result, atol=jnp.finfo(result.dtype).eps)


@pytest.mark.parametrize("shape", ([16, 16], [128, 128], [16, 128], [128, 16]))
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("trim_columns_to", [None, 3])
def test_big_to_thin(shape, dtype, seed, trim_columns_to):
  np.random.seed(seed)
  expected = np.random.randn(*shape).astype(dtype)
  A_d = pops.distribute(expected)
  if trim_columns_to is not None:
    expected = expected[:, :trim_columns_to]

  @pops.pmap
  def _big_to_thin_f(A):
    return vops.big_to_thin(A, trim_columns_to=trim_columns_to)

  A_v = _big_to_thin_f(A_d)
  result = vops.undistribute(A_v)
  testutils.assert_allclose(expected, result, atol=0.)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
def test_frobnorm(shape, column_replicate, dtype, seed):
  np.random.seed(seed)
  matrix = np.random.randn(*shape).astype(dtype)
  expected = np.linalg.norm(matrix)
  vec_d = vops.distribute(matrix, column_replicated=column_replicate)

  @functools.partial(pops.pmap, out_axes=None)
  def test_f(vec):
    return vops.frobnorm(vec)

  result = test_f(vec_d)
  assert (expected - result) / expected < jnp.finfo(dtype).eps


@pytest.mark.parametrize("shape", [[4, 8]])
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("n_cols", [1, 3])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("big", flags)
def test_get_columns(
    shape, column_replicate, n_cols, offset, dtype, seed, big):
  np.random.seed(seed)
  matrix = np.random.randn(*shape).astype(dtype)
  expected = matrix[:, offset:offset + n_cols]
  if big:
    matrix_d = pops.distribute(matrix)
  else:
    matrix_d = vops.distribute(matrix, column_replicated=column_replicate)

  @functools.partial(
      pops.pmap, static_broadcasted_argnums=(2,), in_axes=(0, None, None))
  def test_f(matrix, offset, n_cols):
    return vops.get_columns(matrix, offset, n_cols)

  result = test_f(matrix_d, offset, n_cols)
  result = vops.undistribute(result)
  testutils.assert_allclose(expected, result, atol=np.finfo(dtype).eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate_l", flags)
@pytest.mark.parametrize("column_replicate_r", flags)
@pytest.mark.parametrize("n_cols", [1, 3])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
def test_set_columns_vec(
   shape, column_replicate_l, column_replicate_r, n_cols, offset, dtype, seed):

  shape = [shape[0], shape[1] + offset + n_cols]
  np.random.seed(seed)
  matrix = np.random.randn(*shape).astype(dtype)
  new_cols = np.random.randn(shape[0], n_cols).astype(dtype)
  matrix_d = vops.distribute(matrix, column_replicated=column_replicate_l)
  new_cols_d = vops.distribute(new_cols, column_replicated=column_replicate_r)

  @functools.partial(pops.pmap, in_axes=(0, 0, None))
  def test_f(matrix, new_vecs, offset):
    return vops.set_columns(matrix, new_vecs, offset)

  result = test_f(matrix_d, new_cols_d, offset)
  result = vops.undistribute(result)
  matrix[:, offset:offset + n_cols] = new_cols
  testutils.assert_allclose(matrix, result, atol=np.finfo(dtype).eps)


@pytest.mark.parametrize("shape", [[4, 6]])
@pytest.mark.parametrize("column_replicate_r", flags)
@pytest.mark.parametrize("n_cols", [1, 2, 3, 4])
@pytest.mark.parametrize("offset", [0, 2])
@pytest.mark.parametrize("dtype", [
    np.float32,
])
@pytest.mark.parametrize("seed", seeds)
def test_set_columns_mat(
   shape, column_replicate_r, n_cols, offset, dtype, seed):
  np.random.seed(seed)
  matrix = np.random.randn(*shape).astype(dtype)
  new_cols = np.random.randn(shape[0], n_cols).astype(dtype)
  matrix_d = pops.distribute(matrix)
  new_cols_d = vops.distribute(new_cols, column_replicated=column_replicate_r)

  @functools.partial(pops.pmap, in_axes=(0, 0, None))
  def test_f(matrix, new_vecs, offset):
    return vops.set_columns(matrix, new_vecs, offset)

  result = test_f(matrix_d, new_cols_d, offset)
  result = pops.undistribute(result)
  matrix[:, offset:offset + n_cols] = new_cols
  testutils.assert_allclose(matrix, result, atol=np.finfo(dtype).eps)


@pytest.mark.parametrize("shapeA", [[2, 3], [4, 8], [8, 12]])
@pytest.mark.parametrize("shapeB", [[2, 3], [8, 12]])
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("pmap", flags)
@pytest.mark.parametrize("col_rep_A", flags)
@pytest.mark.parametrize("col_rep_B", flags)
def test_hstack_pair(shapeA, shapeB, dtype, seed, pmap, col_rep_A, col_rep_B):
  np.random.seed(seed)
  shapeA = (shapeA[0] * pops.NROWS, shapeA[1])
  shapeB = (shapeB[0] * pops.NROWS, shapeB[1])
  vec_l = np.random.randn(*shapeA).astype(dtype)
  vec_r = np.random.randn(*shapeB).astype(dtype)
  vec_ld = vops.distribute(vec_l, column_replicated=col_rep_A)
  vec_rd = vops.distribute(vec_r, column_replicated=col_rep_B)

  if pmap:
    test_f = pops.pmap(vops.hstack_pair)
  else:
    test_f = vops.hstack_pair

  if shapeA[0] != shapeB[0]:
    with pytest.raises(TypeError):
      result = test_f(vec_ld, vec_rd)
    return
  result = vops.undistribute(test_f(vec_ld, vec_rd))
  expected = np.hstack([vec_l, vec_r])
  testutils.assert_allclose(result, expected, atol=0.)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
def test_indices_vec(shape, column_replicate, dtype, seed):
  np.random.seed(seed)
  vec = np.random.randn(*shape).astype(dtype)
  vec_d = vops.distribute(vec, column_replicated=column_replicate)

  @pops.pmap
  def test_f(vec):
    rows, cols = vops._indices_vec(vec)
    rows = vops.ReplicatedThinMatrix(rows, vec.is_column_replicated)
    cols = vops.ReplicatedThinMatrix(cols, vec.is_column_replicated)
    prow = pops.my_prow()
    pcol = pops.my_pcol()
    pname = pops.my_name()
    return rows, cols, prow, pcol, pname

  all_rows = np.arange(shape[0])
  expected_cols = np.arange(shape[1])
  local_rows = vec_d.shape[1]
  rows, cols, prows, pcols, pnames = test_f(vec_d)
  for p in pnames:
    these_rows = rows.array[p, :]
    these_cols = cols.array[p, :]
    if vec_d.is_column_replicated:
      pidx = prows[p]
    else:
      pidx = pcols[p]
    expected_rows = all_rows[pidx * local_rows:(pidx + 1) * local_rows]
    np.testing.assert_array_equal(these_rows, expected_rows)
    np.testing.assert_array_equal(these_cols, expected_cols)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("val", [1., -35.4])
@pytest.mark.parametrize("k", [-2, 0, 1])
def test_add_to_diagonal(shape, column_replicate, dtype, seed, val, k):
  np.random.seed(seed)
  vec = np.random.randn(*shape).astype(dtype)
  expected = np.copy(vec)
  id = np.eye(*shape, k=k, dtype=dtype)
  expected = vec + val * id
  vec_d = vops.distribute(vec, column_replicated=column_replicate)

  @pops.pmap
  def test_f(vec):
    return vops.add_to_diagonal(vec, val, k=k)
  result = test_f(vec_d)
  result = vops.undistribute(result)
  np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate_l", flags)
@pytest.mark.parametrize("column_replicate_r", flags)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
def test_vecvec(shape, column_replicate_l, column_replicate_r, dtype, seed):
  np.random.seed(seed)
  matrix_l = np.random.randn(*shape).astype(dtype)
  matrix_r = np.random.randn(*shape).astype(dtype)
  expected = np.dot(matrix_l.conj().T, matrix_r)

  matrix_ld = vops.distribute(matrix_l, column_replicated=column_replicate_l)
  matrix_rd = vops.distribute(matrix_r, column_replicated=column_replicate_r)

  @functools.partial(pops.pmap, out_axes=None)
  def test_f(matrix_ld, matrix_rd):
    return vops.vecvec(matrix_ld, matrix_rd)

  result = test_f(matrix_ld, matrix_rd)
  tol = np.finfo(dtype).eps * np.linalg.norm(expected) ** 2
  testutils.assert_allclose(expected, result, atol=tol)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("seed", seeds)
def test_vecsmall(shape, column_replicate, dtype, seed):
  np.random.seed(seed)
  matrix_l = np.random.randn(*shape).astype(dtype)
  matrix_r = np.random.randn(*shape[::-1]).astype(dtype)
  expected = np.dot(matrix_l, matrix_r)

  matrix_ld = vops.distribute(matrix_l, column_replicated=column_replicate)
  matrix_r = jnp.array(matrix_r)

  @functools.partial(pops.pmap, in_axes=(0, None))
  def test_f(matrix_ld, matrix_r):
    return vops.vecsmall(matrix_ld, matrix_r)

  result = test_f(matrix_ld, matrix_r)
  result = vops.undistribute(result)
  tol = np.finfo(dtype).eps * np.linalg.norm(expected) ** 2
  testutils.assert_allclose(expected, result, atol=tol)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("pmap", flags)
@pytest.mark.parametrize("seed", seeds)
def test_distribute_column_undistribute(shape, pmap, seed):
  np.random.seed(seed)
  v = np.random.rand(*shape)
  vp = vops.distribute(v, pmap=pmap, column_replicated=True,
                       host_replicated_input=True)
  vo = vops.undistribute(vp)
  np.testing.assert_allclose(v, vo)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("pmap", flags)
@pytest.mark.parametrize("seed", seeds)
def test_distribute_row_undistribute(shape, pmap, seed):
  np.random.seed(seed)
  v = np.random.rand(*shape)
  vp = vops.distribute(v, pmap=pmap, column_replicated=False,
                       host_replicated_input=True)
  vo = vops.undistribute(vp)
  np.testing.assert_allclose(v, vo)


###############################################################################
# REDISTRIBUTION
###############################################################################
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("seed", seeds)
def test_column_replicated(shape, column_replicate, seed):
  np.random.seed(seed)
  v = np.random.rand(*shape)
  vp = vops.distribute(v, column_replicated=column_replicate)
  expected = vops.distribute(v, column_replicated=True)

  @pops.pmap
  def test_f(v):
    return vops.to_column_replicated(v)

  result = test_f(vp)
  np.testing.assert_allclose(expected.array, result.array)
  assert result.is_column_replicated == expected.is_column_replicated


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("row_replicate", flags)
@pytest.mark.parametrize("seed", seeds)
def test_row_replicated(shape, row_replicate, seed):
  np.random.seed(seed)
  v = np.random.rand(*shape)
  vp = vops.distribute(v, column_replicated=(not row_replicate))
  expected = vops.distribute(v, column_replicated=False)

  @pops.pmap
  def test_f(v):
    return vops.to_row_replicated(v)

  result = test_f(vp)
  np.testing.assert_allclose(expected.array, result.array)
  assert result.is_column_replicated == expected.is_column_replicated


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("transpose", flags)
@pytest.mark.parametrize("seed", seeds)
def test_matvec(shape, column_replicate, transpose, seed):
  np.random.seed(seed)
  A = np.random.randn(shape[0], shape[0]).astype(DTYPE)
  Ap = pops.distribute(A)
  x = np.random.randn(*shape).astype(DTYPE)
  if transpose:
    expected = np.dot(A.T, x)
  else:
    expected = np.dot(A, x)
  xp = vops.distribute(x, column_replicated=column_replicate)

  @pops.pmap
  def _matvec_f(A, x):
    return vops.matvec(A, x, transpose_A=transpose)

  xp = _matvec_f(Ap, xp)
  result = vops.undistribute(xp)
  eps = 10 * np.finfo(DTYPE).eps * np.linalg.norm(x)
  np.testing.assert_allclose(expected, result, atol=eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("precision", precisions)
def test_vec_t_mat(shape, column_replicate, seed, precision):
  np.random.seed(seed)
  A = np.random.randn(shape[0], shape[0]).astype(DTYPE)
  vec = np.random.randn(shape[0], shape[1]).astype(DTYPE)
  A_d = pops.distribute(A)
  vec_d = vops.distribute(vec, column_replicated=column_replicate)
  expected = jnp.dot(vec.T, A, precision=precision)

  @pops.pmap
  def _vectmat_f(vec, mat):
    return vops.vec_t_mat(vec, mat, precision=precision)
  result = _vectmat_f(vec_d, A_d)
  result = vops.undistribute(result).T
  eps = 10 * np.finfo(DTYPE).eps * np.linalg.norm(vec) * np.linalg.norm(A)
  testutils.assert_allclose(result, expected, atol=eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate_l", flags)
@pytest.mark.parametrize("column_replicate_r", flags)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("precision", precisions)
def test_outer(
    shape, column_replicate_l, column_replicate_r, seed, precision):
  np.random.seed(seed)
  vec_l = np.random.randn(shape[0], shape[1]).astype(DTYPE)
  vec_r = np.random.randn(shape[0], shape[1]).astype(DTYPE)
  vec_ld = vops.distribute(vec_l, column_replicated=column_replicate_l)
  vec_rd = vops.distribute(vec_r, column_replicated=column_replicate_r)
  expected = jnp.dot(vec_l, vec_r.T, precision=precision)

  @pops.pmap
  def _outer_f(vec_l, vec_r):
    return vops.outer(vec_l, vec_r, precision=precision)
  result = _outer_f(vec_ld, vec_rd)
  result = pops.undistribute(result)
  eps = 10 * np.finfo(DTYPE).eps * np.linalg.norm(vec_l)
  eps *= np.linalg.norm(vec_r)
  testutils.assert_allclose(result, expected, atol=eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate", flags)
@pytest.mark.parametrize("right", flags)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
def test_diagmult(shape, column_replicate, right, dtype, seed):
  np.random.seed(seed)
  A = np.random.randn(shape[0], shape[0]).astype(dtype)
  Ap = pops.distribute(A)
  x = np.random.randn(shape[0], 1).astype(dtype)
  if right:
    expected = np.dot(A, np.diag(x.ravel()))
  else:
    expected = np.dot(np.diag(x.ravel()), A)
  xp = vops.distribute(x, column_replicated=column_replicate)

  @pops.pmap
  def _diagmult_f(A, x):
    return vops.diagmult(A, x, vector_on_right=right)

  result_1 = _diagmult_f(Ap, xp)
  result_1 = pops.undistribute(result_1)
  result_2 = vops.diagmult(Ap, xp, vector_on_right=right)
  result_2 = pops.undistribute(result_2)
  eps = 10 * np.linalg.norm(x) * np.linalg.norm(A)
  eps *= testutils.eps(lax.Precision.HIGHEST, dtype)
  testutils.assert_allclose(result_1, result_2, eps)
  testutils.assert_allclose(expected, result_1, eps)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("column_replicate_l", flags)
@pytest.mark.parametrize("column_replicate_r", flags)
@pytest.mark.parametrize("seed", seeds)
def test_align(shape, column_replicate_l, column_replicate_r, seed):
  np.random.seed(seed)
  vec1 = np.zeros(shape, dtype=DTYPE)
  vec2 = np.zeros(shape, dtype=DTYPE)
  vec1 = vops.distribute(vec1, column_replicate_l)
  vec2 = vops.distribute(vec2, column_replicate_r)
  vec2 = vops._align(vec2, vec1)
  assert vec2.is_column_replicated == vec1.is_column_replicated
