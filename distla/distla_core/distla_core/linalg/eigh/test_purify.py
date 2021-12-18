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
"""Test for purify.py."""
import itertools
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from distla_core.linalg.eigh.serial import test_purify
from distla_core.linalg.eigh import purify
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

Ns = [64, 16]
k_fractions = [10, 4]
precisions = [lax.Precision.HIGHEST]
dtypes = [jnp.float32]
seeds = [0, 1]
p_szs = [4, ]

purify_test_argstring = "N,k_fraction,precision,dtype,seed,p_sz"
purify_test_args = list(
  itertools.product(Ns, k_fractions, precisions, dtypes, seeds, p_szs))

methods = ["hole-particle", ]


# TODO: Pad size
@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
@pytest.mark.parametrize("pad_size", [0, ])
def test_grand_canonically_purify(
    N, k_fraction, p_sz, precision, dtype, pad_size, seed):
  """
  Tests that eigh.grand_canonically_purify correctly implements the grand
  canonical purification of Hp into the projector into the subspace bounded by
  sigma.
  """
  init = test_purify.purify_init(
    N, k_fraction, precision, dtype, pad_size, seed, False, distribute=True)
  self_adjoint, k_target, expected, eps, _, overlap_sqrt = init

  ev = np.linalg.eigvalsh(pops.undistribute(self_adjoint))
  mu = np.mean([ev[k_target - 1], ev[k_target]])
  result, _, _, _ = purify.grand_canonically_purify(
    self_adjoint, N, mu, p_sz, precision)
  result = pops.undistribute(result)[:N, :N]
  test_purify.purify_asserts(result, expected, eps)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
@pytest.mark.parametrize("overlaps", [True, False])
@pytest.mark.parametrize("pad_size", [0, 4])
@pytest.mark.parametrize("method", methods)
def test_canonically_purify(N, k_fraction, precision, dtype, p_sz, method,
                            pad_size, overlaps, seed):
  """
  Tests the PM purification method.
  """
  np.random.seed(seed)
  init = test_purify.purify_init(
    N, k_fraction, precision, dtype, pad_size, seed, overlaps, distribute=True)
  self_adjoint, k_target, expected, eps, overlap_inv_sqrt, overlap_sqrt = init
  result, j, errs = purify.canonically_purify(
    self_adjoint, k_target, method=method, p_sz=p_sz,
    overlap_invsqrt=overlap_inv_sqrt, unpadded_dim=N)
  result = pops.undistribute(result)
  result = result[:N, :N]
  test_purify.purify_asserts(result, expected, eps, overlap_sqrt=overlap_sqrt)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
@pytest.mark.parametrize("pad_size", [0, 4])
@pytest.mark.parametrize("overlaps", [False, True])
@pytest.mark.parametrize("start_distributed", [False, True])
def test_dac_purify(N, k_fraction, precision, dtype, p_sz, pad_size, seed,
                    start_distributed, overlaps):
  init = test_purify.purify_init(
    N, k_fraction, precision, dtype, pad_size, seed, overlaps,
    distribute=start_distributed)
  self_adjoint, k_target, expected, eps, overlap_inv_sqrt, overlap_sqrt = init
  result, _, _ = purify.divide_and_conquer_purify(
    self_adjoint, k_target, precision=precision, unpadded_dim=N,
    overlap_invsqrt=overlap_inv_sqrt)
  result = pops.undistribute(result)
  result = result[:N, :N]
  test_purify.purify_asserts(result, expected, 100 * eps,
                             overlap_sqrt=overlap_sqrt)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_subspace(N, p_sz, precision):
  """
  Tests that eigh._subspace correctly produces an isometry into the full-rank
  subspace of a randomly-generated projection matrix.
  """
  np.random.seed(1)
  H = np.random.randn(N, N).astype(np.float32)
  H = 0.5 * (H + H.conj().T)

  Us, _, Vhs = np.linalg.svd(H)
  Up = np.dot(Us, Vhs)
  Id = jnp.eye(N, dtype=H.dtype)
  P = 0.5 * (Up + Id)

  k = jnp.round(jnp.trace(P)).astype(jnp.int32)
  Pd = pops.distribute(P)

  @pops.pmap
  def test_f(P):
    return purify.subspace(P, k, N // pops.NCOLS, p_sz, precision=precision)

  Vkd, _ = test_f(Pd)
  Vk = pops.undistribute(Vkd)
  Vk = Vk[:, :int(k)]
  PVk = np.dot(P, Vk)
  eps = testutils.eps(precision)
  np.testing.assert_allclose(PVk, Vk, atol=10 * eps * jnp.linalg.norm(P))
  orth_delta = np.dot(Vk.conj().T, Vk)
  np.testing.assert_allclose(np.eye(orth_delta.shape[0],
                                    dtype=orth_delta.dtype),
                             orth_delta,
                             atol=10 * eps * jnp.linalg.norm(P))
