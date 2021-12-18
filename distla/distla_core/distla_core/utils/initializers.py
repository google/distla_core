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
"""Initialization routines for distributed matrices
"""
from typing import Tuple, Type
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numpy as np

from distla_core.utils import misc
from distla_core.linalg.tensor import utils
from distla_core.utils import config

GRID = config.GRID


################################################################################
# INITIALIZERS
################################################################################
def uniform(shape: Tuple[int],
            dtype: Type[np.number] = jnp.float32,
            minval: float = 0.0,
            maxval: float = 1.0,
            seed: int = 0):
  """
  Initialize a distributed matrix of shape `shape` with values drawn from a
  random-uniform distribution between [minval, maxval].

  Args:
    shape: The global shape of the matrix.
    dtype: The desired dtype of the output.
    minval, maxval: lower and upper boundary of the uniform distribution.
    seed: Seed for random initialization.

  Returns:
    ShardedDeviceArray: The random matrix.
  """
  if len(shape) != 2:
    raise ValueError(f"Can only initialize matrices." f"Got shape {shape}")
  keys, local_shape = utils._process_random_input(shape, GRID, seed=seed)
  return jax.pmap(jax.random.uniform,
                  static_broadcasted_argnums=(1, 2, 3, 4))(keys, local_shape,
                                                           dtype, minval,
                                                           maxval)


def normal(shape: Tuple[int],
           dtype: Type[np.number] = jnp.float32,
           mu: float = 0.0,
           sigma: float = 1.0,
           seed: int = 0):
  """
  Initialize a distributed matrix of shape `shape` with values drawn from a
   normal distribution with mean `mu` and standard deviation `sigma`.

  Args:
    shape: The global shape of the matrix.
    mu: The mean of the distribution.
    sigma: The standard deviation of the distribution.
    dtype: The desired dtype of the output.
    seed: Seed for random initialization.

  Returns:
    ShardedDeviceArray: The random matrix.

  """
  if len(shape) != 2:
    raise ValueError(f"Can only initialize matrices." f"Got shape {shape}")

  keys, local_shape = utils._process_random_input(shape, GRID, seed=seed)
  return jax.pmap(lambda k, s, d: jax.random.normal(k, s, d) * sigma + mu,
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)


def ones(shape, dtype=jnp.float32):
  """
  Initialize a distributed matrix of shape `shape` with ones.

  Args:
    shape: The global shape of the matrix.
    dtype: The desired dtype of the output.

  Returns:
    ShardedDeviceArray: The matrix.
  """
  if len(shape) != 2:
    raise ValueError(f"Can only initialize matrices." f"Got shape {shape}")

  keys, local_shape = utils._process_random_input(shape, GRID, seed=0)
  return jax.pmap(lambda a, b, c: jax.numpy.ones(b, c),
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)


def zeros(shape, dtype=jnp.float32):
  """
  Initialize a distributed matrix of shape `shape` with zeros.

  Args:
    shape: The global shape of the matrix.
    dtype: The desired dtype of the output.

  Returns:
    ShardedDeviceArray: The matrix.
  """
  if len(shape) != 2:
    raise ValueError(f"Can only initialize matrices." f"Got shape {shape}")
  keys, local_shape = utils._process_random_input(shape, GRID, seed=0)
  return jax.pmap(lambda a, b, c: jax.numpy.zeros(b, c),
                  static_broadcasted_argnums=(1, 2))(keys, local_shape, dtype)
