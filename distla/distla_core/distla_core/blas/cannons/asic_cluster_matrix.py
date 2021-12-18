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
"""DistlaCore ASICNode Matrix.

Easy distributed linear algebra on ASICs.

The goal of this module is to allow users of JAX or numpy to easily add support
for distributed linear algebra on ASICs to their code. The API focusses on
decorators for pure functions operating on JAX or numpy arrays that seamlessly
parallelize and accelerate linear algebra operations.

  Usage example:

  a = np.random.normal(size=(8, 8))
  b = np.random.normal(size=(8, 8))

  @distla_core.distribute_to_asic_node
  def op(a, b):
    a @ b

  op(a, b)  # This is distributed and accellerated on ASICs!
"""
from collections import namedtuple
import functools

import jax
import jax.numpy as jnp
import numpy as np

NUM_CORES_IN_DONUT = 8
PMAP_AXIS = "i"


class ASICNodeMatrix(namedtuple('Matrix', ['tensor', 'core_loc'])):
  """Wrapper class for JAX arrays.

  Attributes:
    tensor: Underlying jax tensor.
    core_loc: core_loc value used to locate slices on ASIC cores
  """

  def __matmul__(self, other):
    """Performs the distrubuted matrix multiplication."""
    result = cannons_alg(self.tensor, other.tensor, self.core_loc)
    return ASICNodeMatrix(result, self.core_loc)

  def __add__(self, other):
    return ASICNodeMatrix(self.tensor + other.tensor, self.core_loc)

  def __sub__(self, other):
    return ASICNodeMatrix(self.tensor - other.tensor, self.core_loc)

  def __truediv__(self, other):
    return ASICNodeMatrix(self.tensor / other, self.core_loc)

  def __mul__(self, other):
    if isinstance(other, ASICNodeMatrix):
      return ASICNodeMatrix(self.tensor * other.tensor, self.core_loc)
    return ASICNodeMatrix(self.tensor * other, self.core_loc)

  def __rmul__(self, other):
    if isinstance(other, ASICNodeMatrix):
      return ASICNodeMatrix(other.tensor * self.tensor, self.core_loc)
    return ASICNodeMatrix(other * self.tensor, self.core_loc)


class ASICNodeListOfMatrices:
  """Wrapper class for lists of JAX arrays.

  This class is used to easily concatenate lists of uniform arrays into a single
  object. Concatenation is done on the 0 axis and all tensors must be of the
  same size.

  Attributes:
    tensor: A single tensor storing multi-tensor concatenation.
    core_loc: core_loc value used to locate slices on ASIC cores
  """

  def __init__(self):
    self.tensor = None
    self.core_loc = None

  def append(self, tensor):
    """Append a new ASICNodeMatrix."""
    if not isinstance(tensor, ASICNodeMatrix):
      raise ValueError(
          f"Can only append with a ASICNodeMatrix, got {type(tensor)}")
    if self.tensor is None:
      self.tensor = tensor.tensor[None,]
      self.core_loc = tensor.core_loc
    else:
      self.tensor = jnp.concatenate((self.tensor, tensor.tensor[None,]))

  def __getitem__(self, key):
    return ASICNodeMatrix(self.tensor[key], self.core_loc)


@jax.jit
def prep_to_distribute_jax(tensor):
  shape = tensor.shape
  new_shape = [4, shape[0] // 4, 2, shape[1] // 2]
  tmp = jnp.reshape(tensor, new_shape)
  tmp = jnp.transpose(tmp, [2, 0, 1, 3])
  tmp = jnp.reshape(tmp, [8, shape[0] // 4, shape[1] // 2])
  return tmp


@jax.jit
def localize_jax(tensor):
  shape = tensor.shape
  tensor = jnp.array(tensor)
  tmp = jnp.reshape(tensor, [2, 4, shape[1], shape[2]])
  tmp = jnp.transpose(tmp, [1, 2, 0, 3])
  return jnp.reshape(tmp, [shape[1] * 4, shape[2] * 2])


def prep_to_distribute(tensor):
  shape = tensor.shape
  new_shape = [4, shape[0] // 4, 2, shape[1] // 2]
  tmp = np.reshape(tensor, new_shape)
  tmp = np.transpose(tmp, [2, 0, 1, 3])
  tmp = np.reshape(tmp, [8, shape[0] // 4, shape[1] // 2])
  return tmp


def localize(tensor):
  shape = tensor.shape
  tensor = np.array(tensor)
  tmp = np.reshape(tensor, [2, 4, shape[1], shape[2]])
  tmp = np.transpose(tmp, [1, 2, 0, 3])
  return np.reshape(tmp, [shape[1] * 4, shape[2] * 2])


def distribute_to_asic_node(fun):
  """Wraps a function operating on numpy arrays.

  Arguments are wrapped in a ASICNodeMatrix. ASICNodeMatrix returns are unwrapped.

  Args:
    fun: A pure function accepting numpy arrays as arguments and returning numpy
      arrays to be wrapped for distribution.

  Returns:
    A wrapped function whose inputs will be wrapped in a ASICNodeMatrix.
  """

  def pmap_body(*args, core_loc):
    new_args = [ASICNodeMatrix(tensor, core_loc) for tensor in args]
    result = fun(*new_args)
    if result is None:
      raise AssertionError("Function returned None value.")
    if isinstance(result, ASICNodeMatrix):
      return result.tensor
    elif isinstance(result, tuple):
      return tuple(x.tensor for x in result)

  core_locs = np.array([i % 2 for i in range(NUM_CORES_IN_DONUT)])
  pfun = functools.partial(jax.pmap(pmap_body, PMAP_AXIS), core_loc=core_locs)

  def wrapper(*args):
    new_args = [prep_to_distribute(tensor) for tensor in args]
    result = pfun(*new_args)
    if isinstance(result, tuple):
      return tuple(map(localize, result))
    return localize(result)

  return wrapper


def distribute_to_asic_node_jittable(fun):
  """Wraps a function operating on JAX arrays.

  This is a jittable version of distribute_to_asic_node.
  Arguments are wrapped in a ASICNodeMatrix. ASICNodeMatrix returns are unwrapped.

  Args:
    fun: A pure function accepting JAX arrays as arguments and returning JAX
      arrays to be wrapped for distribution.

  Returns:
    A wrapped function whose inputs will be wrapped in a ASICNodeMatrix.
  """

  def pmap_body(*args, core_loc):
    new_args = [ASICNodeMatrix(tensor, core_loc) for tensor in args]
    result = fun(*new_args)
    if result is None:
      raise AssertionError("Function returned None value.")
    if isinstance(result, ASICNodeMatrix):
      return result.tensor
    elif isinstance(result, tuple):
      return tuple(x.tensor for x in result)

  core_locs = jnp.array([i % 2 for i in range(8)])
  pfun = functools.partial(jax.pmap(pmap_body, PMAP_AXIS), core_loc=core_locs)

  def wrapper(*args):
    new_args = [prep_to_distribute_jax(tensor) for tensor in args]
    result = pfun(*new_args)
    if isinstance(result, tuple):
      return tuple(map(localize_jax, result))
    return localize_jax(result)

  return wrapper


@jax.jit
def two_core_prep(a, core_loc):

  def _concat(x):
    return jnp.concatenate([x[:, x.shape[1] // 2:], x[:, :x.shape[1] // 2]],
                           axis=1)

  return jax.lax.cond(core_loc, _concat, lambda x: x, a)


# PRECISION TYPES
# bfloat16: jax.lax.Precision.DEFAULT
# float24ish: jax.lax.Precision.HIGH
# float32: jax.lax.Precision.HIGHEST
@jax.jit
def two_core_multiply(a, b, core_loc, precision=jax.lax.Precision.HIGHEST):
  a = two_core_prep(a, core_loc)
  tmp = jnp.matmul(a[:, :a.shape[1] // 2], b, precision=precision)
  b = jax.lax.pshuffle(b, PMAP_AXIS, [1, 0, 3, 2, 5, 4, 7, 6])
  tmp += jnp.matmul(a[:, a.shape[1] // 2:], b, precision=precision)
  return tmp


@jax.jit
def cannons_alg(a, b, core_loc):
  """An implementation of Cannon's for distributed matrix multiplication.

  Args:
    a: first JAX array to be multiplied.
    b: second JAX array.
    core_loc: core_loc value

  Returns:
    A new JAX array.
  """

  right_init = [0, 1, 2, 3, 6, 7, 4, 5]
  left_init = [0, 1, 6, 7, 4, 5, 2, 3]
  shift_hor = [4, 5, 6, 7, 0, 1, 2, 3]
  shift_vert = [2, 3, 0, 1, 6, 7, 4, 5]

  loops = 2
  a = jax.lax.pshuffle(a, PMAP_AXIS, left_init)
  b = jax.lax.pshuffle(b, PMAP_AXIS, right_init)
  tmp = two_core_multiply(a, b, core_loc)
  for _ in range(loops - 1):
    a = jax.lax.pshuffle(a, PMAP_AXIS, shift_hor)
    b = jax.lax.pshuffle(b, PMAP_AXIS, shift_vert)
    tmp += two_core_multiply(a, b, core_loc)
  return tmp


def zeros_like(asic_node_tensor):
  return ASICNodeMatrix(jnp.zeros_like(asic_node_tensor.tensor), asic_node_tensor.core_loc)
