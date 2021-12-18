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
"""Test for solve.py"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.solve.serial import solve

Ns = [4, 8, 16, 128]
DTYPE = np.float32


@pytest.mark.parametrize("N", Ns)
def test_solve_positive_definite(N):
  """
  Checks that solve_positive_definite produces a solution which meets the
  HPLinpack accuracy standard.
  """
  A = np.random.randn(N, N).astype(DTYPE)
  A = 0.5 * (A + A.conj().T)
  evalsA, eVecsA = np.linalg.eigh(A)
  evalsA = np.abs(evalsA)
  A = jnp.dot((evalsA * eVecsA),
              eVecsA.conj().T,
              precision=jax.lax.Precision.HIGHEST)
  x = np.random.randn(N).astype(DTYPE)
  b = jnp.dot(A, x, precision=jax.lax.Precision.HIGHEST)

  result, j = solve.solve_positive_definite(A, b)
  eps = jnp.finfo(A.dtype).eps
  num = jnp.dot(A, result, precision=jax.lax.Precision.HIGHEST)
  num = jnp.linalg.norm(num - b)
  den = jnp.linalg.norm(A) * jnp.linalg.norm(result) * N * eps
  err = num / den
  print("Error was: ", err)
  print("Ran for : ", j, "iterations.")
  assert err <= 1.0


@pytest.mark.parametrize("N", Ns)
def test_solve(N):
  """
  Checks that solve produces a solution which meets the
  HPLinpack accuracy standard.
  """
  A = np.random.randn(N, N).astype(DTYPE)
  x = np.random.randn(N).astype(DTYPE)
  b = jnp.dot(A, x, precision=jax.lax.Precision.HIGHEST)

  result, j = solve.solve(A, b)
  eps = jnp.finfo(A.dtype).eps
  num = jnp.dot(A, result, precision=jax.lax.Precision.HIGHEST)
  num = jnp.linalg.norm(num - b)
  den = jnp.linalg.norm(A) * jnp.linalg.norm(result) * N * eps
  err = num / den
  print("Error was: ", err)
  print("Ran for : ", j, "iterations.")
  assert err <= 1.0
