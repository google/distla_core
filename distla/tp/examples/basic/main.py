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
import jax
import jax.numpy as jnp
from jax import pmap, lax
import sys
import os
import logging

# Worker number is available in python via an env variable
WORKER = os.environ.get('TP_ASIC_WORKER')

logging.basicConfig(
    format='%(asctime)s %(levelname)s:[{}]:%(message)s'.format(WORKER),
    level=logging.INFO)


def main():
  logging.info(f'args: {sys.argv[1:]}')
  hid, ldc = jax.host_id(), jax.local_device_count()
  logging.info(f'host count: {jax.host_count()}')
  logging.info(f'local_device_count: {jax.local_device_count()}, '
               f'device_count: {jax.device_count()}')

  # Profile this snippet!
  jax.profiler.start_trace('/tmp/tensorboard')
  x = jnp.arange(hid * ldc, (hid + 1) * ldc)
  logging.info(x)

  x = x**2
  logging.info(x)

  x = pmap(lambda x: lax.psum(x, axis_name='i'), axis_name='i')(x)
  logging.info(x)
  jax.profiler.stop_trace()


if __name__ == '__main__':
  main()
