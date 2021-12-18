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
"""Utilities for distributed tensors."""
from typing import Sequence, Optional

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import numpy as np

from distla_core.utils import config
from distla_core.utils import misc

AXIS_NAME = config.get_axis_name()


def distribute(tensor: np.ndarray,
               grid_shape: Sequence[int],
               pmap: bool = True) -> pxla.ShardedDeviceArray:
  """
  Convert a numpy array into a ShardedDeviceArray (distributed according to
  `grid_shape`). It is assumed that the dimensions of `tensor`
  are evenly divided by `grid`.

  Args:
    tensor: A distributed array to be converted into a local
      numpy tensor.
    grid_shape: The shape of the processor grid
      according to which `tensor` is distributed.

  Returns:
    ShardedDeviceArray: The distributed tensor

  Raises:
    ValueError: If `tensor.shape` is not evenly divisible by `grid_shape`

  """
  if not np.all([s % p == 0 for s, p in zip(tensor.shape, grid_shape)]):
    raise ValueError(f"tensor.shape = {tensor.shape} not evenly divisible "
                     f"by grid_shape = {grid_shape}.")

  ndim = tensor.ndim
  pshape = np.asarray(grid_shape)

  shape = misc.flatten(
      [p, s] for s, p in zip(np.array(tensor.shape) // pshape, pshape))
  perm = list(range(0, 2 * ndim, 2)) + list(range(1, 2 * ndim, 2))
  reshaped = tensor.reshape(shape).transpose(perm)
  final_shape = (np.prod(reshaped.shape[:ndim]), *reshaped.shape[ndim:])
  A = reshaped.reshape(final_shape)
  if not pmap:
    return A
  return jax.pmap(lambda x: x, devices=jax.local_devices())(A)


def undistribute(tensor: pxla.ShardedDeviceArray,
                 grid_shape: Sequence[int],
                 shape: Optional[Sequence[int]] = None) -> np.ndarray:
  """
  Convert a ShardedDeviceArray (distributed according to `grid_shape`)
  into a numpy tensor.

  Args:
    tensor: A distributed array to be converted into a local numpy tensor.
    grid_shape: The shape of the processor grid according to which `tensor`
      is distributed.
    shape: An optional shape of the resulting numpy array. This can be necessary
      since certain routines like preshape and _pravel reshape local tensors
       into local shapes that avoid/minimize zero padding on ASICs.

  Returns:
    numpy.ndarray: The global tensor in host memory.
  """
  if shape is None:
    local_shape = tensor.shape[1:]
  else:
    local_shape = misc.local_shape(shape, grid_shape)

  shape = tuple(grid_shape) + local_shape
  perm = misc.flatten([[n, n + len(grid_shape)] for n in range(len(grid_shape))
                       ])

  final_shape = misc.global_shape(local_shape, grid_shape)
  return jax.device_put(tensor).reshape(shape).transpose(perm).reshape(
      final_shape)


def _process_random_input(shape, grid_shape, seed=0):
  """
  Helper function for initializing tensors.
  """
  shape = np.asarray(shape)
  grid_shape = np.asarray(grid_shape)
  num_devices = jax.device_count()
  if np.prod(grid_shape) != num_devices:
    raise ValueError(f"number of devices = {num_devices} is different from "
                     f"np.prod(grid_shape) = "
                     f"{np.prod(grid_shape)}")
  key = jax.random.PRNGKey(seed + jax.host_id())
  if np.prod(grid_shape) != num_devices:
    raise ValueError(f"grid_shape {grid_shape} is incompatible with "
                     f"num_devices = {num_devices}")
  if not np.all(shape % grid_shape == 0):
    raise ValueError(f"shape = {shape} is not integer divisible "
                     f"by grid_shape = {grid_shape}")
  local_shape = tuple(shape // grid_shape)
  keys = jax.random.split(key, jax.local_device_count())
  return keys, local_shape


@jax.pmap
def _combine_factors_f64(bulk, remainder):
  eps = jnp.finfo.eps(jnp.float32)
  bulk = bulk.astype(jnp.float64)
  remainder = remainder.astype(jnp.float64)
  return bulk + eps * remainder


def uniform(shape,
            grid_shape,
            dtype=jnp.float32,
            minval=0.0,
            maxval=1.0,
            seed=0):
  """
  Initialize a distributed tensor of shape `shape` with values
  drawn from a random-uniform distribution between [minval, maxval].

  Args:
    shape: The global shape of the tensor.
    grid_shape: The shape of the processor grid according to which
      `tensor` is distributed.
    dtype: The desired dtype of the output.
    minval, maxval: lower and upper boundary of the uniform
      distribution.
    seed: Seed for random initialization.

  Returns:
    ShardedDeviceArray: The random tensor.
  """
  if dtype in (np.float64, jnp.float64):
    bulk = uniform(shape, grid_shape, dtype=jnp.float32, minval=minval,
                   maxval=maxval, seed=seed).astype(jnp.float64)
    remainder = uniform(shape, grid_shape, dtype=jnp.float32, minval=minval,
                        maxval=maxval, seed=seed + 1).astype(jnp.float64)
    return _combine_factors_f64(bulk, remainder)

  keys, local_shape = _process_random_input(shape, grid_shape, seed=seed)
  return jax.pmap(jax.random.uniform,
                  static_broadcasted_argnums=(1, 2, 3, 4))(keys, local_shape,
                                                           dtype, minval,
                                                           maxval)


def normal(shape, grid_shape, dtype=jnp.float32, mu=0.0, sigma=1.0, seed=0):
  """
  Initialize a distributed tensor of shape `shape` with values drawn from a
  normal distribution.

  Args:
    shape: The global shape of the tensor.
    grid_shape: The shape of the processor grid according to which `tensor` is
      distributed.
    mu: The mean of the distribution.
    sigma: The standard deviation of the distribution.
    dtype: The desired dtype of the output.
    seed: Seed for random initialization.

  Returns:
    ShardedDeviceArray: The random tensor.

  """
  if dtype in (np.float64, jnp.float64):
    bulk = normal(shape, grid_shape, dtype=jnp.float32, mu=mu,
                  sigma=sigma, seed=seed).astype(jnp.float64)
    remainder = normal(shape, grid_shape, dtype=jnp.float32, mu=mu,
                       sigma=sigma, seed=seed + 1).astype(jnp.float64)
    return _combine_factors_f64(bulk, remainder)
  keys, local_shape = _process_random_input(shape, grid_shape, seed=seed)
  return jax.pmap(lambda key, shape, dtype: jax.random.normal(
      key, shape, dtype) * sigma + mu,
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)


def ones(shape, grid_shape, dtype=jnp.float32):
  """
  Initialize a distributed tensor of shape `shape` with ones.

  Args:
    shape: The global shape of the tensor.
    grid_shape: The shape of the processor grid according to which `tensor` is
      distributed.
    dtype: The desired dtype of the output.

  Returns:
    ShardedDeviceArray: The tensor.
  """

  keys, local_shape = _process_random_input(shape, grid_shape, seed=0)
  return jax.pmap(lambda key, shape, dtype: jax.numpy.ones(shape, dtype),
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)


def zeros(shape, grid_shape, dtype=jnp.float32):
  """
  Initialize a distributed tensor of shape `shape with zeros.

  Args:
    shape: The global shape of the tensor.
    grid_shape: The shape of the processor grid according to which `tensor` is
      distributed.
    dtype: The desired dtype of the output.

  Returns:
    ShardedDeviceArray: The tensor.
  """

  keys, local_shape = _process_random_input(shape, grid_shape, seed=0)
  return jax.pmap(lambda key, shape, dtype: jax.numpy.zeros(shape, dtype),
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)
