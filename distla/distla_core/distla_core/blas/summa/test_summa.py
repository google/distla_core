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
# Lint as: python3
"""Contains tests of the functions in summa.py.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from distla_core.utils import pops
from distla_core.blas.summa import summa

DTYPE = jnp.float32
AXIS_NAME = pops.AXIS_NAME

NROW = pops.NROWS
NCOL = pops.NCOLS
matrix_shapes = [(16, 16), (32, 16), (16, 32), (128, 128)]
p_szs = [3, 4, 8, 16]
precisions = [lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST]


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_summa_TT(matrix_shape, p_sz, precision):
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  B = np.random.randn(*matrix_shape).astype(DTYPE)
  Ap = pops.distribute(A)
  Bp = pops.distribute(B)
  summa_f = functools.partial(
      summa.summa,
      p_sz=p_sz,
      transpose_A=True,
      transpose_B=True,
      precision=precision)

  with pytest.raises(NotImplementedError):
    _ = jax.pmap(summa_f, axis_name=AXIS_NAME)(Ap, Bp)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_summa_TN(matrix_shape, p_sz, precision):
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  B = np.random.randn(*matrix_shape).astype(DTYPE)
  C = pops.dot(A.T, B, precision=precision)
  Ap = pops.distribute(A)
  Bp = pops.distribute(B)
  summa_f = functools.partial(
      summa.summa,
      p_sz=p_sz,
      transpose_A=True,
      transpose_B=False,
      precision=precision)

  Cp = jax.pmap(summa_f, axis_name=AXIS_NAME)(Ap, Bp)
  Cp = pops.undistribute(Cp)
  atol = jnp.finfo(DTYPE).eps * jnp.linalg.norm(C)
  np.testing.assert_allclose(C, Cp, atol=atol)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_summa_NT(matrix_shape, p_sz, precision):
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  B = np.random.randn(*matrix_shape).astype(DTYPE)
  C = pops.dot(A, B.T, precision=precision)
  Ap = pops.distribute(A)
  Bp = pops.distribute(B)
  summa_f = functools.partial(
      summa.summa,
      p_sz=p_sz,
      transpose_A=False,
      transpose_B=True,
      precision=precision)

  Cp = jax.pmap(summa_f, axis_name=AXIS_NAME)(Ap, Bp)
  Cp = pops.undistribute(Cp)
  atol = jnp.finfo(DTYPE).eps * jnp.linalg.norm(C)
  np.testing.assert_allclose(C, Cp, atol=atol)


@pytest.mark.parametrize("matrix_shape", matrix_shapes)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_summa_NN(matrix_shape, p_sz, precision):
  np.random.seed(10)
  A = np.random.randn(*matrix_shape).astype(DTYPE)
  B = np.random.randn(*matrix_shape).astype(DTYPE).T
  C = pops.dot(A, B, precision=precision)
  Ap = pops.distribute(A)
  Bp = pops.distribute(B)
  summa_f = functools.partial(
      summa.summa,
      p_sz=p_sz,
      transpose_A=False,
      transpose_B=False,
      precision=precision)
  Cp = jax.pmap(summa_f, axis_name=AXIS_NAME)(Ap, Bp)
  Cp = pops.undistribute(Cp)
  atol = jnp.finfo(DTYPE).eps * jnp.linalg.norm(C)
  np.testing.assert_allclose(C, Cp, atol=atol)


def test_summa_TN_bad_shape():
  matrix_shape = (4, 8)
  A = np.ones(matrix_shape, dtype=DTYPE)
  Ap = pops.distribute(A)
  Bp = pops.distribute(A.T)
  summa_f = functools.partial(
      summa.summa, p_sz=1, transpose_A=True, transpose_B=False)
  summa_f = jax.pmap(summa_f, axis_name=AXIS_NAME)
  with pytest.raises(TypeError):
    _ = summa_f(Ap, Bp)


def test_summa_NT_bad_shape():
  matrix_shape = (4, 8)
  A = np.ones(matrix_shape, dtype=DTYPE)
  Ap = pops.distribute(A)
  Bp = pops.distribute(A.T)
  summa_f = functools.partial(
      summa.summa, p_sz=1, transpose_A=False, transpose_B=True)

  summa_f = jax.pmap(summa_f, axis_name=AXIS_NAME)
  with pytest.raises(TypeError):
    _ = summa_f(Ap, Bp)


def test_summa_NN_bad_shape():
  matrix_shape = (4, 8)
  A = np.ones(matrix_shape, dtype=DTYPE)
  Ap = pops.distribute(A)
  Bp = pops.distribute(A)
  summa_f = functools.partial(
      summa.summa, p_sz=1, transpose_A=False, transpose_B=False)
  summa_f = jax.pmap(summa_f, axis_name=AXIS_NAME)
  with pytest.raises(TypeError):
    _ = summa_f(Ap, Bp)
