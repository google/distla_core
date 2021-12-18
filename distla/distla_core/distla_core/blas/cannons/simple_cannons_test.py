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
import numpy as np
import pytest

from distla_core.blas.cannons import simple_cannons

Ns = [4]
NCORE = 2
NROW = 2
NCOL = 2

GRID = (NCORE, NROW, NCOL)
DTYPE = np.float32


@pytest.mark.parametrize("N", Ns)
def test_cannons(N):
  np.random.seed(1)
  A_l = np.arange((N * N)).reshape((N, N))
  B_l = np.arange((N * N)).reshape((N, N))
  A = simple_cannons.distribute(A_l, GRID)
  B = simple_cannons.distribute(B_l, GRID)

  expected = jnp.dot(A_l, B_l, precision=jax.lax.Precision.HIGHEST)

  @functools.partial(jax.pmap, axis_name=simple_cannons.AXIS_NAME)
  def cannons_f(A, B):
    return simple_cannons.cannons_NN(A, B, GRID)

  result = cannons_f(A, B)
  result = simple_cannons.undistribute(result, GRID)
  np.testing.assert_allclose(expected, result)
