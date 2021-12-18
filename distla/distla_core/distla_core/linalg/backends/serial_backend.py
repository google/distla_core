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
Backend class for bundling operations needed for serial implementations of
various algorithms
"""
import jax.numpy as jnp
from jax import lax

from distla_core.utils import misc
from distla_core.utils import pops


class SerialBackend:
  """
  Bundles helper functions for the serial computations.
  """

  def __init__(self, precision=lax.Precision.HIGHEST):
    self.name = "SerialBackend"
    self.precision = precision

  def add_to_diagonal(self, matrix, value, k=0, unpadded_dim=None):
    shifted = matrix + value * jnp.eye(*(matrix.shape), dtype=matrix.dtype, k=k)
    if unpadded_dim is not None:
      shifted = misc.apply_pad_serial(shifted, unpadded_dim)
    return shifted

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
    if transpose_A and transpose_B:
      return pops.dot(A.T, B.T, precision=precision)
    if transpose_A:
      return pops.dot(A.T, B, precision=precision)
    if transpose_B:
      return pops.dot(A, B.T, precision=precision)
    return pops.dot(A, B, precision=precision)

  def similarity_transform(self, A, B):
    AB = self.matmul(A, B)
    return self.matmul(B.conj(), AB, transpose_A=True)

  def trace(self, A):
    return jnp.trace(A)

  def frobnorm(self, A):
    return jnp.linalg.norm(A)

  def vdot(self, A, B, precision=None):
    if precision is None:
      precision = self.precision
    return jnp.vdot(A, B, precision=precision)

  def sum(self, A):
    return jnp.sum(A)

  def shape(self, A):
    return A.shape

  def eye_like(self, A, unpadded_dim=None):
    result = jnp.eye(*A.shape, dtype=A.dtype)
    if unpadded_dim is not None:
      result = misc.apply_pad_serial(result, unpadded_dim)
    return result

  def gershgorin(self, A):
    return misc.gershgorin(A)

  def transpose(self, A):
    return A.T

  def __hash__(self):
    return hash(self.precision)

  def __eq__(self, other):
    return self.name == other.name
