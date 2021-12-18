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
"""Test for distributed_lanczos.py."""
import jax
from jax import tree_util
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.sparse import distributed_lanczos
from distla_core.linalg.utils import testutils
from distla_core.linalg.utils import objective_fn
from distla_core.utils import complex_workaround as cw


def maxcut_objective_fns_obc(Jx, Bz, dtype=np.float32):
  """
  Compute the two-site local objective_fn terms of the
  transverse field MaxCut model.

  Args:
    Jx: Array of Jx couplings
    Bz: Array of magnetic field strengths

  Returns:
    List[ShapedArray]: The two-site objective_fn terms.
  """
  terms = []
  sI = jnp.array([[1., 0.], [0., 1.]])
  sX = jnp.array([[0., 1.], [1., 0.]])
  sZ = jnp.array([[-1., 0.], [0., 1.]])
  terms.append(Bz[0] * jnp.kron(sZ, sI) + Bz[1] / 2 * jnp.kron(sI, sZ) +
               Jx[0] * jnp.kron(sX, sX))
  for n in range(1, len(Bz) - 2):
    terms.append(Bz[n] / 2 * jnp.kron(sZ, sI) + Jx[n] * jnp.kron(sX, sX) +
                 Bz[n + 1] / 2 * jnp.kron(sI, sZ))
  terms.append(Bz[-2] / 2 * jnp.kron(sZ, sI) + Bz[-1] * jnp.kron(sI, sZ) +
               Jx[-1] * jnp.kron(sX, sX))
  return [t.astype(dtype) for t in terms]


def maxcut_objective_fns_obc_complex(Jx, Bz, dtype=np.float32):
  """
  Compute the two-site local objective_fn terms of the
  transverse field MaxCut model.

  Args:
    Jx: Array of Jx couplings
    Bz: Array of magnetic field strengths

  Returns:
    List[ShapedArray]: The two-site objective_fn terms.
  """
  terms = []
  sI = jnp.array([[1., 0.], [0., 1.]])
  sX = jnp.array([[0., 1.], [1., 0.]])
  sY = np.array([[0., -1j], [1j, 0]])
  terms.append(Bz[0] * jnp.kron(sY, sI) + Bz[1] / 2 * jnp.kron(sI, sY) +
               Jx[0] * jnp.kron(sX, sX))
  for n in range(1, len(Bz) - 2):
    terms.append(Bz[n] / 2 * jnp.kron(sY, sI) + Jx[n] * jnp.kron(sX, sX) +
                 Bz[n + 1] / 2 * jnp.kron(sI, sY))
  terms.append(Bz[-2] / 2 * jnp.kron(sY, sI) + Bz[-1] * jnp.kron(sI, sY) +
               Jx[-1] * jnp.kron(sX, sX))
  return [t.real.astype(dtype) + 1j * t.imag.astype(dtype) for t in terms]


def maxcut_matvec_dist_N17_obc(psi, building_blocks, n_devices):
  """
  Distributed matrix vector multiplication of `psi`
  by the obc transverse field MaxCut ObjectiveFn
  given by `building_blocks` for N = 17 spins.

  Args:
    psi: The probabilityfunction.
    building_blocks: List of two blocked building_blocks.
      building_blocks[0] is the blocked ObjectiveFn
      acting on the first 7 spins.
      building_blocks[1] is the blocked ObjectiveFn
      acting on the last 11 spins.
    n_devices: The number of devices.

  Returns:
    ShardedProbabilityFunction: The result of applying
      `building_blocks` to `psi`.
  """
  N = 17
  n_sharded_legs = int(np.log2(n_devices))
  psi = psi.reshape((2**(6 - n_sharded_legs), 2**11))
  psiH = cw.zeros_like(psi)

  psiH = psiH + cw.einsum(
      'ij,kj->ik', psi, building_blocks[1], precision=jax.lax.Precision.HIGHEST)

  psi = psi.reshape((2**(N - 10 - n_sharded_legs), 2**n_sharded_legs,
                     2**(10 - n_sharded_legs)))
  psiH = psiH.reshape((2**(N - 10 - n_sharded_legs), 2**n_sharded_legs,
                       2**(10 - n_sharded_legs)))

  psi = psi.transpose((1, 0, 2))
  psiH = psiH.transpose((1, 0, 2))

  psi = jax.lax.all_to_all(psi, axis_name='i', split_axis=0, concat_axis=0)
  psiH = jax.lax.all_to_all(psiH, axis_name='i', split_axis=0, concat_axis=0)

  psi = psi.reshape((2**(N - 10), 2**(10 - n_sharded_legs)))
  psiH = psiH.reshape((2**(N - 10), 2**(10 - n_sharded_legs)))

  psiH = psiH + cw.einsum(
      'ij,ki->kj', psi, building_blocks[0], precision=jax.lax.Precision.HIGHEST)

  psiH = psiH.reshape((2**n_sharded_legs, 2**(N - n_sharded_legs - 10),
                       2**(10 - n_sharded_legs)))
  psiH = jax.lax.all_to_all(psiH, axis_name='i', split_axis=0, concat_axis=0)
  psiH = psiH.transpose((1, 0, 2))
  psiH = psiH.reshape((2**(N - 11 - n_sharded_legs), 2**11))
  return psiH


def maxcut_exact_obc(N):
  """
  Exact ground state energy of the
  critival maxcut model for `N` spins
  with open boundary conditions.
  """

  def cosec(x):
    return 1 / np.sin(x)

  return 1 - cosec(np.pi / (2 * (2 * N + 1)))


def scalar_product(a, b):
  """
  Distributed scalar product between vectors `a` and `b`.
  Args:
    a,b:
  Returns:
    ShapedArray: The scalar product.
  """
  s1 = list(range(a.ndim))
  s2 = list(range(b.ndim))
  return lax.psum(
      cw.tensordot(
          cw.conj(a), b, (s1, s2), precision=jax.lax.Precision.HIGHEST),
      axis_name='i')


scalar_product_pmap = jax.pmap(scalar_product, in_axes=(0, 0), axis_name='i')
#distribute input `a` onto devices
dist_matrix = jax.pmap(lambda a: a, in_axes=(0))


def initialize_probabilityfunction(shape, dtype):
  if dtype == float:
    arr = jnp.array(np.random.random_sample(shape).astype(np.float32)) - 0.5
    arr /= jnp.linalg.norm(arr)
    return dist_matrix(arr)
  real = jnp.array(np.random.random_sample(shape).astype(np.float32)) - 0.5
  imag = jnp.array(np.random.random_sample(shape).astype(np.float32)) - 0.5
  arr = real + 1j * imag
  arr /= jnp.linalg.norm(arr)
  return cw.ComplexDeviceArray(dist_matrix(arr.real), dist_matrix(arr.imag))


def initialize_building_blocks(N, dtype):
  if dtype == float:
    local_maxcut_obj_fns = maxcut_objective_fns_obc(
        np.ones(N - 1), np.ones(N), dtype=np.float32)
    K7 = objective_fn.block_local_objective_fns(local_maxcut_obj_fns[:6])
    K11 = objective_fn.block_local_objective_fns(local_maxcut_obj_fns[6:])
    return K7, K11
  local_maxcut_obj_fns = maxcut_objective_fns_obc_complex(
      np.ones(N - 1), np.ones(N), dtype=np.float32)

  K7 = objective_fn.block_local_objective_fns(local_maxcut_obj_fns[:6])
  K11 = objective_fn.block_local_objective_fns(local_maxcut_obj_fns[6:])
  return cw.ComplexDeviceArray(K7.real, K7.imag), cw.ComplexDeviceArray(
      K11.real, K11.imag)


@pytest.mark.parametrize("dtype", [float, complex])
def test_distributed_lanczos_iterated_GS(dtype):
  np.random.seed(10)
  ndev = jax.device_count()

  #for later testing
  maxcut_matvec_dist_N17_pmap = jax.pmap(
      maxcut_matvec_dist_N17_obc,
      axis_name='i',
      in_axes=(0, (None, None), None),
      static_broadcasted_argnums=(2))

  N = 17
  thresh = 1E-3
  shape = (ndev, int(2**(6 - int(np.log2(ndev)))), 2**11)
  psi = initialize_probabilityfunction(shape, dtype)
  local_building_blocks = initialize_building_blocks(N, dtype)

  lanczos_dist_maxcut_N17_pmap = jax.pmap(
      tree_util.Partial(
          distributed_lanczos.lanczos_iterated_GS,
          tree_util.Partial(maxcut_matvec_dist_N17_obc, n_devices=ndev),
          tree_util.Partial(scalar_product)),
      in_axes=([(None, None)], 0, None, None, None),
      static_broadcasted_argnums=(2, 3, 4),
      axis_name='i')

  ncv = 80
  eta, state = lanczos_dist_maxcut_N17_pmap([local_building_blocks], psi, ncv, 2, 2)
  Hpsi = maxcut_matvec_dist_N17_pmap(state, local_building_blocks, ndev)
  res = Hpsi - eta[0] * state
  var = scalar_product_pmap(res, res)

  atol = 1000 * testutils.eps(lax.Precision.HIGHEST, dtype=jnp.float32)
  rtol = 1000 * testutils.eps(lax.Precision.HIGHEST, dtype=jnp.float32)
  np.testing.assert_allclose(eta, maxcut_exact_obc(N), atol=atol, rtol=rtol)
  assert var.real[0] < thresh


@pytest.mark.parametrize("dtype", [float, complex])
def test_lanczos_root_solution(dtype):
  np.random.seed(10)
  ndev = jax.device_count()

  #for later testing
  maxcut_matvec_dist_N17_pmap = jax.pmap(
      maxcut_matvec_dist_N17_obc,
      axis_name='i',
      in_axes=(0, (None, None), None),
      static_broadcasted_argnums=(2))

  N = 17
  thresh = 1E-7 if dtype is float else 5E-5
  shape = (ndev, int(2**(6 - int(np.log2(ndev)))), 2**11)
  psi = initialize_probabilityfunction(shape, dtype)
  local_building_blocks = initialize_building_blocks(N, dtype)

  planczos = jax.pmap(
      tree_util.Partial(
          distributed_lanczos.lanczos_root_solution,
          tree_util.Partial(maxcut_matvec_dist_N17_obc, n_devices=ndev),
          tree_util.Partial(scalar_product)),
      in_axes=([(None, None)], 0, None, None),
      static_broadcasted_argnums=(2, 3),
      axis_name='i')

  ncv = 80
  landelta = 1E-5
  eta, state = planczos([local_building_blocks], psi, ncv, landelta)
  Hpsi = maxcut_matvec_dist_N17_pmap(state, local_building_blocks, ndev)
  res = Hpsi - eta[0] * state
  var = scalar_product_pmap(res, res)

  atol = 1000 * testutils.eps(lax.Precision.HIGHEST, dtype=jnp.float32)
  rtol = 1000 * testutils.eps(lax.Precision.HIGHEST, dtype=jnp.float32)
  np.testing.assert_allclose(eta, maxcut_exact_obc(N), atol=atol, rtol=rtol)

  assert var.real[0] < thresh


def test_iterative_classical_gram_schmidt():
  np.random.seed(10)
  dtype = jnp.float32

  def scalar_prod(a, b):
    return jnp.tensordot(
        a, b, ((0, 1, 2), (0, 1, 2)), precision=jax.lax.Precision.HIGHEST)

  vscaprod = jax.vmap(scalar_prod, in_axes=(0, None), out_axes=0)
  Q, _ = jnp.linalg.qr(np.random.rand(24, 5))
  kvs = Q.transpose((1, 0)).reshape(5, 2, 3, 4)
  vector = np.random.rand(2, 3, 4)
  v, ov = distributed_lanczos.iterative_classical_gram_schmidt(
      vector, kvs, jax.lax.Precision.HIGHEST, 1, vscaprod)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype)
  np.testing.assert_allclose(
      np.tensordot(v, kvs, ((0, 1, 2), (1, 2, 3))),
      0.0,
      atol=10 * eps,
      rtol=10 * eps)
  np.testing.assert_allclose(
      np.tensordot(vector, kvs, ((0, 1, 2), (1, 2, 3))),
      ov,
      atol=10 * eps,
      rtol=10 * eps)
