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
"""Configuration utility for distla_core."""
# TODO (mganahl): move this config file one folder up
#                 and rename to global_config
import jax
import numpy as np

from distla_core.utils import misc

AXIS_NAME = 'i'


def get_axis_name() -> str:
  """
  Get the name of the pmapped axis.
  This is currently hard coded to 'i'.
  """
  return AXIS_NAME


n_hosts_primes = misc.prime_factors(jax.process_count())
n_local_device_primes = misc.prime_factors(jax.local_device_count())
NHCOLS = int(np.prod(n_hosts_primes[::2]))
NHROWS = int(np.prod(n_hosts_primes[1::2]))
NDROWS = int(np.prod(n_local_device_primes[0::2]))
NDCOLS = int(np.prod(n_local_device_primes[1::2]))

NROWS = NHROWS * NDROWS
NCOLS = NHCOLS * NDCOLS
NDPROCS = NDCOLS * NDROWS
NPROCS = NHROWS * NHCOLS * NDPROCS
GRID = (NROWS, NCOLS)
DGRID = (NDROWS, NDCOLS)
HGRID = (NHROWS, NHCOLS)


def get_processor_grid():
  """ Returns an array of shape GRID whose (i, j)'th entry is the pmap index
  of the processor at the (i, j)'th prow/pcol.
  """
  n_local_devices = jax.local_device_count()
  asic_node_grid = np.arange(n_local_devices).reshape(NDROWS, NDCOLS, order='F')
  hrows = np.concatenate(
      [asic_node_grid + n_local_devices * n for n in range(NHROWS)], axis=0).astype(
          np.int32)
  numel = np.max(hrows.ravel()) + 1
  return np.concatenate([hrows + numel * n for n in range(NHCOLS)], axis=1)
