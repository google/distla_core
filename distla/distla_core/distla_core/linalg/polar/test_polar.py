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
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest
import scipy as sp

from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.polar import polar
from distla_core.linalg.polar import polar_utils
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

shapes = [(8, 8), (16, 8), (128, 128)]
precisions = [lax.Precision.DEFAULT, lax.Precision.HIGH, lax.Precision.HIGHEST]
dtypes = [jnp.float32, ]


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(3,),
    in_axes=(0, None, None, None),
    out_axes=(0, None),
)
def magnify_f(matrix, s_min, s_thresh, backend):
  return polar_utils._magnify_spectrum(matrix, 200, s_min, s_thresh, backend)


@functools.partial(
    pops.pmap,
    static_broadcasted_argnums=(2, 5, 6),
    in_axes=(0, None, None, None, None, None, None),
    out_axes=(0, 0, None, None, None),
)
def polar_f(matrix, eps, maxiter, s_min, s_thresh, p_sz, precision):
  return polar.polar(matrix, eps=eps, maxiter=maxiter, s_min=s_min,
                     s_thresh=s_thresh, p_sz=p_sz, precision=precision)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("dtype", dtypes)
def test_magnify_spectrum(shape, precision, dtype):
  m = shape[0]
  n = shape[1]
  s_thresh = 0.1
  input_matrix = np.random.randn(m, n).astype(dtype)
  input_matrix /= np.linalg.norm(input_matrix)
  backend = distributed_backend.DistributedBackend(128, precision=precision)
  s_min = testutils.eps(lax.Precision.HIGHEST, dtype=dtype)
  input_matrix_distributed = pops.distribute(input_matrix)
  result, j = magnify_f(input_matrix_distributed, s_min, s_thresh, backend)
  result = pops.undistribute(result)
  svals = np.linalg.svd(np.array(result, dtype=np.float64), compute_uv=False)
  too_small = svals[svals < s_thresh]
  assert too_small.size == 0


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("dtype", dtypes)
def test_polar(shape, precision, dtype):
  input_matrix = np.random.randn(*shape).astype(dtype)
  input_matrix_distributed = pops.distribute(input_matrix)
  eps = None
  s_min = None
  s_thresh = 0.1

  unitary, posdef, j_rogue, j_total, err = polar_f(
    input_matrix_distributed, eps, 50, s_min, s_thresh, 128, precision)
  unitary = pops.undistribute(unitary)
  posdef = pops.undistribute(posdef)
  tol = shape[1] * testutils.eps(precision, dtype=dtype)

  if shape[0] >= shape[1]:
    unitary2 = testutils.matmul(unitary.conj().T, unitary)
  else:
    unitary2 = testutils.matmul(unitary, unitary.conj().T)

  id_mat = jnp.eye(unitary2.shape[0], dtype=dtype)
  testutils.assert_allclose(id_mat, unitary2, atol=tol)

  testutils.assert_allclose(posdef, posdef.conj().T, atol=20 * tol)
  ev, _ = jnp.linalg.eigh(posdef)
  ev = ev[jnp.abs(ev) > tol]
  negative_ev = jnp.sum(ev < 0.)
  assert negative_ev == 0.

  reconstructed = jnp.matmul(unitary, posdef, precision=lax.Precision.HIGHEST)
  testutils.assert_allclose(input_matrix, reconstructed, atol=20 * tol)

  unitary_sp, _ = sp.linalg.polar(input_matrix)
  testutils.assert_allclose(unitary_sp, unitary, atol=tol)
