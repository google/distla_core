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
"""Utilities for mixed precision."""
import jax
import jax.numpy as jnp
import numpy as np

valid_dtypes = (jnp.bfloat16, jnp.float32, jnp.float64)


def mantissa_bits(dtype, precision):
  """The number of bits in the mantissa for a given dtype and precision.

  Args:
    dtype: A Jax dtype.
    precision: One of `jax.lax.Precision`.
  Returns:
    The number of mantissa bits.
  """
  if dtype == jnp.bfloat16:
    return 7
  elif dtype == jnp.float32 and precision == jax.lax.Precision.DEFAULT:
    return 7
  elif dtype == jnp.float32 and precision == jax.lax.Precision.HIGH:
    return 15
  elif dtype == jnp.float32 and precision == jax.lax.Precision.HIGHEST:
    return 23
  elif dtype == jnp.float64:
    return 49
  else:
    msg = f"Unknown dtype ({dtype}) or precision ({precision})."
    raise ValueError(msg)


def minimal_precision(n_bits):
  """Returns the minimal (dtype, precision) combination with n_bits of accuracy.

  Args:
    n_bits: Number of mantissa bits
  Returns:
    dtype: A Jax dtype.
    precision: One of `jax.lax.Precision`.
  """
  precision_combinations = (
      (jnp.bfloat16, jax.lax.Precision.DEFAULT),
      (jnp.float32, jax.lax.Precision.DEFAULT),
      (jnp.float32, jax.lax.Precision.HIGH),
      (jnp.float32, jax.lax.Precision.HIGHEST),
      (jnp.float64, jax.lax.Precision.HIGHEST),
  )
  for dtype, precision in precision_combinations:
    m_bits = mantissa_bits(dtype, precision)
    if m_bits >= n_bits:
      return dtype, precision
