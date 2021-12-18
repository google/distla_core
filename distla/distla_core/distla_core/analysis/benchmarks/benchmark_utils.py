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
import jax.numpy as jnp
import numpy as np


def mmult_tflops(rows, cols, dt, dtype=jnp.float32):
  """ The effective TFLOPS/s of squaring the input matrix.
  """
  coef = 2
  if dtype in (np.complex64, np.complex128, jnp.complex64, jnp.complex128):
    coef = 8
  tflops = coef * int(rows) / float(dt)
  tflops *= (int(cols) / 1E12)
  tflops *= int(cols)
  return tflops


def tflops_per_second(flops, dt):
  """ Computes an effective processing rate in TFLOPS per second.
    TFLOP/S = flops * / (dt * 1E12)
  Args:
    flops: Estimated FLOPS in the computation.
    dt: Elapsed time in seconds.
  Returns:
    The estimate.
  """
  return flops / (1E12 * dt)


def gbps(n_elements, dtype, dt):
  """ Computes an effective bandwidth in GB per second.
    GBPS = n_elements * (wordsize in bits) / (dt * 8E9)
  Args:
    n_elements: Number of input and output elements in the computation.
    dtype: Dtype of the elements.
    dt: Elapsed time in seconds.
  Returns:
    The estimate.
  """
  return n_elements * jnp.finfo(dtype).bits / (dt * 8E9)


def per_iter(per_iter, n_iter, result, header):
  """ Optionally modifies speed functions to produce per-iteration results.

  Args:
    per_iter: Whether or not to do the modification.
    n_iter: The number of iterations.
    result: The speed estimate.
    header: The unmodified header.
  Returns:
    result: result / n_iter
    header: With "per iteration" appended.
  """
  if n_iter == 0:
    n_iter = 1

  if per_iter:
    result /= n_iter
    header = header + " per iteration"
  return result, header
