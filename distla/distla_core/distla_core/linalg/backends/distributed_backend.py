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
"""
Backend class for bundling operations needed for distributed implementations of
various algorithms.
"""
from jax import lax
import jax.numpy as jnp

from distla_core.blas.summa import summa
from distla_core.utils import config
from distla_core.utils import pops


class DistributedBackend:
  """
  Bundles helper functions for distributed computations.
  """

  def __init__(self, p_sz: int, precision=lax.Precision.HIGHEST):
    self.p_sz = p_sz
    self.grid = config.GRID
    self.name = "DistributedBackend"
    self.precision = precision

  def add_to_diagonal(self, matrix, value, k=0, unpadded_dim=None):
    return pops.add_to_diagonal(matrix, value, k=k, unpadded_dim=unpadded_dim)

  def matmul(
      self,
      A,
      B,
      transpose_A=False,
      transpose_B=False,
      precision=None,
  ):
    if precision is None:
      precision = self.precision
    return summa.summa(
        A,
        B,
        self.p_sz,
        transpose_A,
        transpose_B,
        precision=precision,
    )

  def similarity_transform(self, A, B):
    AB = self.matmul(A, B)
    return self.matmul(B.conj(), AB, transpose_A=True)

  def trace(self, A):
    return pops.trace(A)

  def frobnorm(self, A):
    return pops.frobnorm(A)

  def vdot(self, A, B, precision=None):
    if precision is None:
      precision = self.precision
    return lax.psum(
        jnp.vdot(A, B, precision=precision),
        axis_name=pops.AXIS_NAME,
    )

  def gershgorin(self, A):
    return pops.gershgorin(A)

  def sum(self, A):
    return lax.psum(jnp.sum(A), axis_name=pops.AXIS_NAME)

  def shape(self, A):
    if len(A.shape) == 2:
      M = A.shape[0] * self.grid[0]
      N = A.shape[1] * self.grid[1]
    elif len(A.shape) == 3:
      M = A.shape[1] * self.grid[0]
      N = A.shape[2] * self.grid[1]
    else:
      raise TypeError(f"A had unexpected non-matrix shape {A.shape}.")
    return M, N

  def eye_like(self, A, unpadded_dim=None):
    return pops.eye(A.shape, A.dtype, unpadded_dim=unpadded_dim)

  def _can_transpose(self, A):
    # See documentation for pops.transpose to understand these restrictions.
    if pops.NCOLS == pops.NROWS * 2:
      return A.shape[0] % 2 == 0
    elif pops.NROWS == pops.NCOLS * 2:
      return A.shape[1] % 2 == 0
    else:
      return False

  def transpose(self, A):
    if self._can_transpose(A):
      return pops.transpose(A)
    else:
      msg = ("WARNING: Can't efficiently transpose a distributed matrix of "
             f"shape {A.shape} on a device grid {pops.GRID}. "
             "Falling back on matmul-based transpose.")
      # TODO use logging.warn
      print(msg)
      eye = self.eye_like(A)
      # Note that Precision.HIGHEST is used instead of self.precision.
      return self.matmul(
          A,
          eye,
          transpose_A=True,
          precision=lax.Precision.HIGHEST,
      )

  def __hash__(self):
    return hash((self.p_sz, self.grid, self.precision))

  def __eq__(self, other):
    return (self.p_sz == other.p_sz and self.grid == other.grid and
            self.precision == other.precision)
