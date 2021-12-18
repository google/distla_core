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
"""Test for complex_workaround.py"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.utils import complex_workaround as cw
from distla_core.linalg.utils import testutils


def test_is_jittable():

  @jax.jit
  def f(a):
    return cw.ComplexDeviceArray(a.real + 1, a.imag + 2)

  a = cw.ComplexDeviceArray(1, 2)
  b = f(a)
  assert isinstance(b, cw.ComplexDeviceArray)
  assert b.real == 2
  assert b.imag == 4


def test_is_fori_loopable():

  def f(i, a):
    return cw.ComplexDeviceArray(a.real + i, a.imag - i)

  a = cw.ComplexDeviceArray(0, 0)
  b = jax.lax.fori_loop(0, 5, f, a)
  assert isinstance(b, cw.ComplexDeviceArray)
  assert b.real == sum(range(5))
  assert b.imag == -sum(range(5))


def test_basic_add():
  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5))
  ai = np.random.normal(size=((2,) * 5))
  b = np.random.normal(size=((2,) * 5))
  bi = np.random.normal(size=((2,) * 5))
  C = cw.ComplexDeviceArray(a, ai) + cw.ComplexDeviceArray(b, bi)
  np.testing.assert_allclose(a + b, C.real)
  np.testing.assert_allclose(ai + bi, C.imag)


def test_basic_sub():
  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5))
  ai = np.random.normal(size=((2,) * 5))
  b = np.random.normal(size=((2,) * 5))
  bi = np.random.normal(size=((2,) * 5))
  C = cw.ComplexDeviceArray(a, ai) - cw.ComplexDeviceArray(b, bi)
  np.testing.assert_allclose(a - b, C.real)
  np.testing.assert_allclose(ai - bi, C.imag)


def test_basic_mul():
  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  b = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  B = cw.ComplexDeviceArray(b.real, b.imag)
  for v in [5, -2.2, -2.5j, 2 + 2.5j]:
    np.testing.assert_allclose((a * v).real, (A * v).real)
    np.testing.assert_allclose((a * v).imag, (A * v).imag)
    np.testing.assert_allclose((A * v).real, (v * A).real)
    np.testing.assert_allclose((A * v).imag, (v * A).imag)

  AB = A * B
  ab = a * b
  np.testing.assert_allclose(AB.real, ab.real)
  np.testing.assert_allclose(AB.imag, ab.imag)


def test_jit_mul():

  @jax.jit
  def fn(ar, v):
    return ar * v

  @functools.partial(jax.jit, static_argnums=1)
  def fn_static(ar, v):
    return ar * v

  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  for v in [5, -2.2, -2.5j, 2 + 2.5j]:
    np.testing.assert_allclose((a * v).real, fn(A, v).real, rtol=1e-5)
    np.testing.assert_allclose((a * v).imag, fn(A, v).imag, rtol=1e-5)
    np.testing.assert_allclose((a * v).real, fn_static(A, v).real, rtol=1e-5)
    np.testing.assert_allclose((a * v).imag, fn_static(A, v).imag, rtol=1e-5)


def test_basic_truediv():
  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  b = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  B = cw.ComplexDeviceArray(b.real, b.imag)
  for v in [5, -2.2, -2.5j, 2 + 2.5j]:
    np.testing.assert_allclose((a / v).real, (A / v).real, atol=1E-7, rtol=1e-6)
    np.testing.assert_allclose((a / v).imag, (A / v).imag, atol=1E-7, rtol=1e-6)

  AB = A / B
  ab = a / b
  np.testing.assert_allclose(AB.real, ab.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(AB.imag, ab.imag, atol=1E-7, rtol=1e-6)


def test_jit_truediv():

  @jax.jit
  def fn(ar, v):
    return ar / v

  @functools.partial(jax.jit, static_argnums=1)
  def fn_static(ar, v):
    return ar / v

  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  for v in [5, -2.2, -2.5j, 2 + 2.5j]:
    np.testing.assert_allclose((a / v).real, fn(A, v).real, rtol=1e-5)
    np.testing.assert_allclose((a / v).imag, fn(A, v).imag, rtol=1e-5)
    np.testing.assert_allclose((a / v).real, fn_static(A, v).real, rtol=1e-5)
    np.testing.assert_allclose((a / v).imag, fn_static(A, v).imag, rtol=1e-5)


def test_dtype():
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  a.astype(np.complex64)
  A = cw.ComplexDeviceArray(a.real, a.imag)
  assert a.real.dtype == A.dtype


def test_conj():
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  np.testing.assert_allclose(a.conj().real, A.conj().real)
  np.testing.assert_allclose(a.conj().imag, A.conj().imag)

  np.testing.assert_allclose(a.conj().real, cw.conj(A).real)
  np.testing.assert_allclose(a.conj().imag, cw.conj(A).imag)

  np.testing.assert_allclose(a.conj().real, cw.conjubuilding_block(A).real)
  np.testing.assert_allclose(a.conj().imag, cw.conjubuilding_block(A).imag)


def test_getitem():
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  Aslice = A[1:3]
  aslice = a[1:3]
  np.testing.assert_allclose(aslice.real, Aslice.real)
  np.testing.assert_allclose(aslice.imag, Aslice.imag)
  Aslice = A[-3:-1]
  aslice = a[-3:-1]
  np.testing.assert_allclose(aslice.real, Aslice.real)
  np.testing.assert_allclose(aslice.imag, Aslice.imag)


def test_ndim():
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  assert a.ndim == A.ndim


def test_tensordot():
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  b = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  B = cw.ComplexDeviceArray(b.real, b.imag)
  AB = cw.tensordot(A, B, ([1, 2], [3, 4]), precision=jax.lax.Precision.HIGHEST)
  ab = cw.tensordot(a, b, ([1, 2], [3, 4]), precision=jax.lax.Precision.HIGHEST)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, AB.real.dtype)
  np.testing.assert_allclose(AB.real, ab.real, atol=10 * eps, rtol=1e-6)
  np.testing.assert_allclose(AB.imag, ab.imag, atol=10 * eps, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32])
def test_zeros(dtype):
  shape = (2,) * 5
  a = np.zeros(shape, dtype)
  A = cw.zeros(shape, dtype, float)
  np.testing.assert_allclose(A.real, a.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(A.imag, a.imag, atol=1E-7, rtol=1e-6)
  assert a.dtype == A.dtype


@pytest.mark.parametrize("dtype", [np.float32])
def test_array(dtype):
  np.random.seed(0)
  shape = (2,) * 5
  a = np.random.random_sample(shape).astype(dtype) + \
      1j *np.random.random_sample(shape).astype(dtype)
  A = cw.array(a, dtype, float)
  np.testing.assert_allclose(A.real, a.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(A.imag, 0.0, atol=1E-7, rtol=1e-6)

  A = cw.array(a, dtype, complex)
  np.testing.assert_allclose(A.real, a.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(A.imag, a.imag, atol=1E-7, rtol=1e-6)

  assert a.real.dtype == A.dtype


def test_einsum():
  np.random.seed(0)
  a = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  b = np.random.normal(size=((2,) * 5)) + np.random.normal(size=((2,) * 5)) * 1j
  A = cw.ComplexDeviceArray(a.real, a.imag)
  B = cw.ComplexDeviceArray(b.real, b.imag)
  AB = cw.einsum(
      'abcde,aBcDe ->bBdD',
      A,
      B,
      precision=jax.lax.Precision.HIGHEST,
  )
  ab = cw.einsum(
      'abcde,aBcDe ->bBdD',
      a,
      b,
      precision=jax.lax.Precision.HIGHEST,
  )
  np.testing.assert_allclose(AB.real, ab.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(AB.imag, ab.imag, atol=1E-7, rtol=1e-6)


def test_sqrt():
  np.random.seed(0)
  shape = (2,) * 5
  a = (np.random.random_sample(shape) +
       1j * np.random.random_sample(shape)).astype(np.float32)
  A = cw.ComplexDeviceArray(a.real, a.imag)
  sqrtA = cw.sqrt(A)
  sqrta = cw.sqrt(a)
  np.testing.assert_allclose(sqrtA.real, sqrta.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(sqrtA.imag, sqrta.imag, atol=1E-7, rtol=1e-6)


def test_index_update():
  np.random.seed(0)
  a = jnp.array(
      np.random.normal(size=10) + np.random.normal(size=10) * 1j,
      dtype=np.complex64)
  b = jnp.array(
      np.random.normal(size=4) + np.random.normal(size=4) * 1j,
      dtype=np.complex64)
  A = cw.ComplexDeviceArray(a.real, a.imag)
  B = cw.ComplexDeviceArray(b.real, b.imag)

  ab = a.at[jnp.array([0, 2, 3, 5])].set(b)
  AB = A.at[jnp.array([0, 2, 3, 5])].set(B)
  np.testing.assert_allclose(AB.real, ab.real, atol=1E-7, rtol=1e-6)
  np.testing.assert_allclose(AB.imag, ab.imag, atol=1E-7, rtol=1e-6)
