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
import functools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.blas.summa import summa as summa_mod
from distla_core.linalg.eigh import eigh
from distla_core.linalg.invsqrt import invsqrt as invsqrt_mod
from distla_core.linalg.polar import polar as polar_mod
from distla_core.linalg.tensor import ptranspose
from distla_core.utils import pops


def serial_or_distributed(ndim, serial_f, dist_f):
  if ndim == 2:
    return serial_f
  if ndim == 3:
    return dist_f
  raise ValueError(f"ndim={ndim} must be 2 (serial) or 3 (distributed).")


# eye
@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3))
def _eye(x, local_shape, dtype, k):
  return pops.eye(local_shape, dtype, k=k)


def eye_dist(shape, dtype, k):
  ps = np.arange(pops.NDPROCS)
  return _eye(ps, shape[1:], dtype, k)


def eye(shape, dtype, k=0):

  def serial_f(shape, dtype, k):
    return jnp.eye(*shape, dtype=dtype, k=k)

  out_f = serial_or_distributed(len(shape), serial_f, eye_dist)
  return out_f(shape, dtype=dtype, k=k)


# frobnorm
@functools.partial(pops.pmap, out_axes=None)
def _frobnorm(matrix):
  return pops.frobnorm(matrix)


def frobnorm(matrix):
  out_f = serial_or_distributed(matrix.ndim, jnp.linalg.norm, _frobnorm)
  return out_f(matrix)


# frobdiff
@functools.partial(pops.pmap, out_axes=None)
def _frobdiff(matrix_a, matrix_b):
  return pops.frobnorm(matrix_a - matrix_b)


def frobdiff(matrix_a, matrix_b):
  if matrix_a.ndim != matrix_b.ndim:
    raise TypeError(f"matrix_a.ndim={matrix_a.ndim} != "
                    f"matrix_b.ndim={matrix_b.ndim}.")

  def serial_f(matrix_a, matrix_b):
    return jnp.linalg.norm(matrix_a - matrix_b)

  def dist_f(matrix_a, matrix_b):
    return _frobdiff(matrix_a, matrix_b)

  out_f = serial_or_distributed(matrix_a.ndim, serial_f, dist_f)
  return out_f(matrix_a, matrix_b)


# matmul
@functools.partial(pops.pmap, static_broadcasted_argnums=(2, 3, 4, 5))
def _summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision):
  return summa_mod.summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b,
                         precision=precision)


def summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b,
          precision=lax.Precision.HIGHEST):
  return _summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b, precision)


def matmul_serial(matrix_a, matrix_b, transpose_a, transpose_b,
                  conj_a, conj_b, precision):
  if transpose_a:
    matrix_a = matrix_a.T
  if conj_a:
    matrix_a = matrix_a.conj()
  if transpose_b:
    matrix_b = matrix_b.T
  if conj_b:
    matrix_b = matrix_b.conj()
  return jnp.dot(matrix_a, matrix_b, precision=precision)


def matmul(matrix_a, matrix_b, transpose_a=False, transpose_b=False,
           conj_a=False, conj_b=False, p_sz=128,
           precision=lax.Precision.HIGHEST):

  def serial_f(matrix_a, matrix_b):
    return matmul_serial(matrix_a, matrix_b, transpose_a, transpose_b,
                         conj_a, conj_b, precision)

  def dist_f(matrix_a, matrix_b):
    return summa(matrix_a, matrix_b, p_sz, transpose_a, transpose_b,
                 precision=precision)
  out_f = serial_or_distributed(matrix_a.ndim, serial_f, dist_f)
  return out_f(matrix_a, matrix_b)


def similarity_transform(operand, operator, p_sz=128,
                         precision=lax.Precision.HIGHEST):
  first = matmul(operand, operator, p_sz=p_sz, precision=precision)
  return matmul(operator, first, transpose_a=True, conj_a=True, p_sz=p_sz,
                precision=precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1,))
def _transpose(matrix, conjubuilding_block):
  if conjubuilding_block:
    matrix = matrix.conj()
  return ptranspose.ptranspose(matrix, (1, 0), pops.GRID)


# transpose
def transpose(matrix, conjubuilding_block=True):
  def serial_f(matrix):
    if conjubuilding_block:
      matrix = matrix.conj()
    return matrix.T

  def dist_f(matrix):
    return _transpose(matrix, conjubuilding_block)

  out_f = serial_or_distributed(matrix.ndim, serial_f, dist_f)
  return out_f(matrix)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2))
def _eigh_canonical(matrix, p_sz, precision):
  return eigh.eigh(matrix, p_sz, precision, canonical=True)


def eigh_canonical(matrix, p_sz=128, precision=lax.Precision.HIGHEST):
  return _eigh_canonical(matrix, p_sz, precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2))
def _svd_canonical(matrix, p_sz, precision):
  return eigh.svd(matrix, p_sz, precision, canonical=True)


def svd_canonical(matrix, p_sz=128, precision=lax.Precision.HIGHEST):
  return _svd_canonical(matrix, p_sz, precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2))
def _eigh_grand_canonical(matrix, p_sz, precision):
  return eigh.eigh(matrix, p_sz, precision, canonical=False)


def eigh_grand_canonical(matrix, p_sz=128, precision=lax.Precision.HIGHEST):
  return _eigh_grand_canonical(matrix, p_sz, precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2))
def _svd_grand_canonical(matrix, p_sz, precision):
  return eigh.svd(matrix, p_sz, precision, canonical=False)


def svd_grand_canonical(matrix, p_sz=128, precision=lax.Precision.HIGHEST):
  return _svd_grand_canonical(matrix, p_sz, precision)


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(2, 5, 6),
    in_axes=(0, None, None, None, None, None, None),
    out_axes=(0, 0, None, None))
def _invsqrt(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision):
  return invsqrt_mod.invsqrt(matrix, eps, maxiter, s_min, s_thresh, p_sz,
                             precision)


def invsqrt(matrix, eps=None, maxiter=200, s_min=None, s_thresh=0.1, p_sz=128,
            precision=lax.Precision.HIGHEST):
  return _invsqrt(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision)


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(2, 5, 6),
    in_axes=(0, None, None, None, None, None, None),
    out_axes=(0, 0, None, None, None))
def _polar(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision):
  return polar_mod.polar(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision)


def polar(matrix, eps=None, maxiter=200, s_min=None, s_thresh=0.1, p_sz=128,
          precision=lax.Precision.HIGHEST):
  return _polar(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision)


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(2, 5, 6),
    in_axes=(0, None, None, None, None, None, None),
    out_axes=(0, None, None, None))
def _polarU(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision):
  return polar_mod.polarU(matrix, eps, maxiter, s_min, s_thresh, p_sz,
                          precision)


def polarU(matrix, eps=None, maxiter=200, s_min=None, s_thresh=0.1, p_sz=128,
           precision=lax.Precision.HIGHEST):
  return _polarU(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision)


@functools.partial(pops.pmap, static_broadcasted_argnums=(1, 2, 3, 4, 5))
def subspace(matrix, k_tup, p_sz, precision, maxiter, polar_iter):
  return eigh_grand_canonical_mod._subspace(
      matrix,
      k_tup,
      p_sz,
      maxiter=maxiter,
      polar_iter=polar_iter,
      precision=precision)
