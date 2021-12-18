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
"""Tests for refine_polar.py."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.mixed_precision import refine_polar
from distla_core.linalg.mixed_precision import utils
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

DIMS = (16, 32, 64)
SEEDS = (0, 1)
BOOLS = (True, False)
PANEL_SIZES = (128,)
# The pairs for (orig_dtype, target_dtype) for which refinement makes sense,
# i.e. target_dtype has at least as much precision as orig_dtype.
PROMOTION_PAIRS = tuple(
    (orig_dtype, target_dtype)
    for (orig_dtype,
         target_dtype) in itertools.product(utils.valid_dtypes, repeat=2)
    if (utils.mantissa_bits(orig_dtype, jax.lax.Precision.HIGHEST) <=
        utils.mantissa_bits(target_dtype, jax.lax.Precision.HIGHEST)))


def polar_decompose(K):
  """Exact polar decomposition using numpy."""
  dtype = K.dtype
  K = np.array(K)
  U, S, V = np.linalg.svd(K, full_matrices=False)
  U = U @ V
  P = (V.T.conj() * S) @ V
  U = jnp.array(U, dtype=dtype)
  P = jnp.array(P, dtype=dtype)
  return U, P


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("distributed", BOOLS)
@pytest.mark.parametrize("orig_dtype, target_dtype", PROMOTION_PAIRS)
@pytest.mark.parametrize("p_sz", PANEL_SIZES)
@pytest.mark.parametrize("lyapunov_method", ("CG", "Newton-Schultz"))
@testutils.with_x64
def test_refine_polar_hermiticity(
    dim,
    seed,
    distributed,
    orig_dtype,
    target_dtype,
    p_sz,
    lyapunov_method,
):
  if distributed and testutils.on_cpu_devices():
    # bf16 isn't supported on CPUs. Even if orig_dtype and target_dtype are
    # jnp.float32 or jnp.float64, bf16s are still used in the intermediate
    # steps.
    pytest.skip()
  np.random.seed(seed)
  U = np.linalg.svd(np.random.randn(dim, dim), full_matrices=False)[0]
  P = np.random.randn(dim, dim)
  V, S, _ = np.linalg.svd(P)
  P = (V * S) @ V.T.conj()  # Makes P pos. def.
  U_jax = jnp.array(U, dtype=target_dtype)
  P_jax = jnp.array(P, dtype=orig_dtype).astype(target_dtype)
  # Make sure the last bits of P_jax aren't just zeros.
  noise_level = 2**(
      -utils.mantissa_bits(orig_dtype, jax.lax.Precision.HIGHEST) - 1)
  P_noise = np.random.rand(dim, dim) * np.abs(P) * noise_level
  P_jax = P_jax + jnp.array(P_noise, dtype=target_dtype)
  M = U @ np.array(P_jax)
  U_norm = np.linalg.norm(U)
  P_norm = np.linalg.norm(P)
  M_norm = np.linalg.norm(M)

  if distributed:
    U_jax = pops.distribute_global(U_jax)
    P_jax = pops.distribute_global(P_jax)
  U_refined, P_refined = refine_polar.refine_polar_hermiticity(
      U_jax,
      P_jax,
      orig_dtype,
      p_sz=p_sz,
      lyapunov_method=lyapunov_method,
  )
  if distributed:
    U_refined = pops.undistribute_global(U_refined)
    P_refined = pops.undistribute_global(P_refined)

  assert U_refined.dtype == target_dtype
  assert P_refined.dtype == target_dtype
  testutils.test_unitarity(U_refined, eps_coef=10 * U_norm)
  testutils.test_hermiticity(P_refined, eps_coef=10 * P_norm)
  UP_refined = jnp.dot(
      U_refined,
      P_refined,
      precision=jax.lax.Precision.HIGHEST,
  )
  eps = testutils.eps(jax.lax.Precision.HIGHEST, target_dtype)
  testutils.assert_allclose(UP_refined, M, 10 * M_norm * eps)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("distributed", BOOLS)
@pytest.mark.parametrize("orig_dtype, target_dtype", PROMOTION_PAIRS)
@pytest.mark.parametrize("p_sz", PANEL_SIZES)
@testutils.with_x64
def test_refine_polar(dim, seed, distributed, orig_dtype, target_dtype, p_sz):
  if distributed and testutils.on_cpu_devices():
    # bf16 isn't supported on CPUs. Even if orig_dtype and target_dtype are
    # jnp.float32 or jnp.float64, bf16s are still used in the intermediate
    # steps.
    pytest.skip()
  np.random.seed(seed)
  K_np = np.random.randn(dim, dim)
  U_np, P_np = polar_decompose(K_np)
  K_norm = np.linalg.norm(K_np)
  U_norm = np.linalg.norm(U_np)
  P_norm = np.linalg.norm(P_np)
  K_target = jnp.array(K_np, dtype=target_dtype)
  # Polar decompose at lower precision
  U_orig, _ = polar_decompose(K_target.astype(orig_dtype))
  assert U_orig.dtype == orig_dtype

  if distributed:
    U_orig = pops.distribute_global(U_orig)
    K_target = pops.distribute_global(K_target)
  U_refined, P_refined = refine_polar.refine_polar(U_orig, K_target, p_sz)
  if distributed:
    U_refined = pops.undistribute_global(U_refined)
    P_refined = pops.undistribute_global(P_refined)

  assert U_refined.dtype == target_dtype
  assert P_refined.dtype == target_dtype
  K_reco = jnp.dot(U_refined, P_refined, precision=jax.lax.Precision.HIGHEST)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, target_dtype)
  testutils.test_unitarity(U_refined, eps_coef=10 * U_norm)
  testutils.test_hermiticity(P_refined, eps_coef=10 * P_norm)
  testutils.assert_allclose(K_reco, K_np, 10 * K_norm * eps)
  testutils.assert_allclose(U_refined, U_np, 10 * U_norm * eps)
  testutils.assert_allclose(P_refined, P_np, 10 * P_norm * eps)
