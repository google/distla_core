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
"""Tests for basic mixed precision operations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.utils import testutils

DS = (128, 256)
SEEDS = (0, 1)


@pytest.mark.parametrize("D", DS)
@pytest.mark.parametrize("seed", SEEDS)
@testutils.with_x64
def test_promote_32to64(D, seed):
  """Check that Jax matmuls with type-promoted matrices work when promoting from
  float32 to float64.
  """
  np.random.seed(seed)
  A = np.random.randn(D, D).astype(np.float32).astype(np.float64)
  B = np.random.randn(D, D).astype(np.float32).astype(np.float64)
  A_jax = jnp.array(A, dtype=jnp.float64)
  B_jax = jnp.array(B, dtype=jnp.float64)
  AB = np.dot(A, B)
  AB_jax = jnp.dot(A_jax, B_jax, precision=jax.lax.Precision.HIGHEST)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype=jnp.float64)
  testutils.assert_allclose(AB, AB_jax, atol=np.linalg.norm(AB) * eps)


@pytest.mark.parametrize("D", DS)
@pytest.mark.parametrize("seed", SEEDS)
@testutils.with_x64
def test_promote_16to32(D, seed):
  """Check that Jax matmuls with type-promoted matrices work when promoting from
  bfloat16 to float32.
  """
  np.random.seed(seed)
  A = np.array(
      jnp.array(np.random.randn(D, D), dtype=jnp.bfloat16).astype(jnp.float32))
  B = np.array(
      jnp.array(np.random.randn(D, D), dtype=jnp.bfloat16).astype(jnp.float32))
  A_jax = jnp.array(A, dtype=jnp.float32)
  B_jax = jnp.array(B, dtype=jnp.float32)
  AB = np.dot(A, B)
  AB_jax = jnp.dot(A_jax, B_jax, precision=jax.lax.Precision.HIGHEST)
  eps = testutils.eps(jax.lax.Precision.HIGHEST, dtype=jnp.float32)
  testutils.assert_allclose(AB, AB_jax, atol=np.linalg.norm(AB) * eps)
