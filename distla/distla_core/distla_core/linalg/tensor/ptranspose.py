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
from typing import Sequence

import jax
from jax.interpreters import pxla
import numpy as np

from distla_core.utils import config

AXIS_NAME = config.get_axis_name()


def ptranspose(tensor: pxla.ShardedDeviceArray, perm: Sequence[int],
               processor_grid_shape: Sequence[int]) -> pxla.ShardedDeviceArray:
  """
  Parallel tranpose `tensor` with `perm`.

  Args:
    tensor: A distributed array to be converted into a local
      numpy tensor.
    perm: A permutation.
    processor_grid: The processor grid according to which
      `tensor` is distributed.

  Returns:
    ShardedDeviceArray: The permuted tensor.
  """
  ndev = np.prod(processor_grid_shape)
  grid = np.arange(ndev).reshape(processor_grid_shape)
  pperm = grid.transpose(perm).ravel()
  pshuffled = jax.lax.pshuffle(tensor, axis_name=AXIS_NAME, perm=pperm)
  return pshuffled.transpose(perm)
