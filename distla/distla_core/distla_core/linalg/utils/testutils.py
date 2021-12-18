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
"""Utilities for tests in distla_core."""
import copy
import functools

import jax
from jax import lax
import jax.numpy as jnp
import jaxlib
import numpy as np

from distla_core.utils import misc
from distla_core.utils import pops


def matmul(matrix_a, matrix_b):
  matrix_a = np.array(matrix_a).astype(np.float64)
  matrix_b = np.array(matrix_b).astype(np.float64)
  return np.dot(matrix_a, matrix_b)


def assert_allclose(A, B, atol):
  if A.dtype == jnp.bfloat16:
    A = A.astype(np.float32)
  if B.dtype == jnp.bfloat16:
    B = B.astype(np.float32)
  if hasattr(atol, "dtype") and atol.dtype == jnp.bfloat16:
    atol = atol.astype(np.float32)
  np.testing.assert_allclose(A, B, atol=atol)


def test_unitarity(matrix, precision=lax.Precision.HIGHEST, eps_coef=1.):
  dtype = matrix.dtype
  my_eps = eps_coef * eps(precision, dtype)
  if matrix.ndim == 3:
    matrix = pops.undistribute(matrix, collect_to_host=True)
  matrix = np.array(matrix)

  n_rows, n_cols = matrix.shape

  matrix_t_matrix = np.dot(matrix.conj().T, matrix)
  expected_mtm = jnp.eye(n_cols, dtype=dtype)
  assert_allclose(matrix_t_matrix, expected_mtm, my_eps)

  if n_rows == n_cols:
    matrix_matrix_t = np.dot(matrix, matrix.conj().T)
    expected_mmt = jnp.eye(n_rows, dtype=dtype)
    assert_allclose(matrix_matrix_t, expected_mmt, my_eps)


def test_hermiticity(matrix, eps_coef=1.):
  dtype = matrix.dtype
  my_eps = eps_coef * eps(lax.Precision.HIGHEST, dtype)
  if matrix.ndim == 3:
    matrix = pops.undistribute(matrix)
  n_rows, n_cols = matrix.shape
  assert n_rows == n_cols
  matrix_t = matrix.conj().T
  assert_allclose(matrix_t, matrix, my_eps)


def _mantissa_eps(mantissa_bits):
  return 0.5 * (2**(1 - mantissa_bits))


def eps(precision, dtype=jnp.float32):
  return _eps(precision, dtype)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _eps(precision, dtype):
  if dtype in (jnp.float64, jnp.complex128, np.float64, np.complex128):
    dtype_eps = _mantissa_eps(49)
  elif dtype in (jnp.float32, jnp.complex64, np.float32, np.complex64):
    if precision == lax.Precision.DEFAULT:
      dtype_eps = jnp.finfo(jnp.bfloat16).eps
    elif precision == lax.Precision.HIGH:
      dtype_eps = _mantissa_eps(15)
    elif precision == lax.Precision.HIGHEST:
      dtype_eps = jnp.finfo(jnp.float32).eps
    else:
      raise ValueError(f"Invalid precision {precision}.")
  else:
    dtype_eps = jnp.finfo(dtype).eps
  dtype_eps = jnp.full(1, dtype_eps, dtype=dtype).real
  return dtype_eps[0]


def unique_permutations(elements):
  unique = list(set(elements))
  stack = []
  counts = [elements.count(u) for u in unique]
  return perm_helper(unique, counts, stack, [0] * len(elements),
                     len(elements) - 1)


def perm_helper(unique, counts, stack, perm, pos):
  if pos < 0:
    stack.append(copy.copy(perm))
    return stack

  for j, u in enumerate(unique):
    if counts[j] > 0:
      perm[pos] = u
      counts[j] -= 1
      stack = perm_helper(unique, counts, stack, perm, pos - 1)
      counts[j] += 1

  return stack


def get_shapes(ndim):
  """
  produce a bunch of tensor shapes of order `ndim`.

  Args:
    ndim: The tensor order.

  Returns:
    list[tuple[int]]: A list of shapes.
  """
  if ndim == 3:
    shapes = unique_permutations((8, 64, 128))
    some_combs = sum((list(zip(shapes, unique_permutations(pshape)))
                      for pshape in ((1, 2, 4), (2, 2, 2), (1, 1, 8))), [])
    return some_combs

  if ndim == 4:
    shapes = unique_permutations((8, 8, 64, 128))
    some_combs = sum((list(zip(shapes, unique_permutations(pshape)))
                      for pshape in ((1, 1, 2, 4), (1, 2, 2, 2), (1, 1, 1, 8))),
                     [])
    return some_combs
  raise ValueError(f"ndim={ndim} not implemented")


def get_reshape_test_shapes(ndim):
  """
  produce a bunch of tensor shapes of order `ndim`
  for testing `preshape`.

  Args:
    ndim: The tensor order.

  Returns:
    list[tuple[int]]: A list of shapes.
  """
  if ndim == 3:
    shapes = unique_permutations((8, 32, 32))
    all_combs = [[(s, p) for s in shapes] for p in unique_permutations((1, 2, 4))] +\
      [[(s, p) for s in shapes] for p in unique_permutations((1, 1, 8))] +\
      [[(s, p) for s in shapes] for p in unique_permutations((2, 2, 2))]
    return misc.flatten(all_combs)

  if ndim == 4:
    shapes = unique_permutations((8, 8, 32, 32))
    all_combs = [[(s, p) for s in shapes] for p in unique_permutations((1, 1, 2, 4))] + \
      [[(s, p) for s in shapes] for p in unique_permutations((1, 2, 2, 2))] + \
      [[(s, p) for s in shapes] for p in unique_permutations((1, 1, 1, 8))]
    return misc.flatten(all_combs)
  raise ValueError(f"ndim={ndim} not implemented")


def with_x64(f):
  """A decorator to run a function within `with jax.experimental.enable_x64()`.
  """

  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    with jax.experimental.enable_x64():
      return f(*args, **kwargs)

  return wrapped


def on_cpu_devices():
  """Returns whether Jax is running on CPU devices."""
  return isinstance(jax.devices()[0], jaxlib.xla_extension.CpuDevice)
