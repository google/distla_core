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
"""Tests for refine_unitary.py."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.mixed_precision import refine_unitarity
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


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("distributed", BOOLS)
@pytest.mark.parametrize("orig_dtype, target_dtype", PROMOTION_PAIRS)
@pytest.mark.parametrize("p_sz", PANEL_SIZES)
@testutils.with_x64
def test_refine_unitarity(
    dim,
    seed,
    distributed,
    orig_dtype,
    target_dtype,
    p_sz,
):
  if distributed and testutils.on_cpu_devices():
    # bf16 isn't supported on CPUs. Even if orig_dtype and target_dtype are
    # jnp.float32 or jnp.float64, bf16s are still used in the intermediate
    # steps.
    pytest.skip()
  np.random.seed(seed)
  U = np.linalg.svd(np.random.randn(dim, dim))[0]
  U_jax = jnp.array(U, dtype=orig_dtype)
  U_norm = np.linalg.norm(U)
  if distributed:
    U_jax = pops.distribute_global(U_jax)
  assert U_jax.dtype == orig_dtype
  U_refined = refine_unitarity.refine_unitarity(U_jax, target_dtype)
  assert U_refined.dtype == target_dtype
  if distributed:
    U_refined = pops.undistribute_global(U_refined)
  testutils.test_unitarity(U_refined, eps_coef=U_norm)
