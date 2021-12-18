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
"""Module for refining the precision of unitaries."""
import jax
import jax.numpy as jnp
import numpy as np

from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.backends import serial_backend
from distla_core.linalg.mixed_precision import utils
from distla_core.utils import pops


def _refine_unitarity_32to64(U_32, order, backend):
  """Takes a float32 unitary and refines it to a float64 unitary.

  Args:
    U_32: A float32 unitary matrix.
    order: Order of the polynomial to use. Options are 3 and 5.
    backend: `distla_core`.linalg.backends `distributed_backend` or `serial_backend`
  Returns:
    The same matrix refined to a float64-unitary.
  """
  U_64 = U_32.astype(np.float64)
  eye_64 = backend.eye_like(U_64)
  B_64 = backend.matmul(
      U_64.conj(),
      U_64,
      transpose_A=True,
      precision=jax.lax.Precision.HIGHEST,
  ) - eye_64
  if order == 3:
    U_refined = U_64 - 0.5 * backend.matmul(
        U_64,
        B_64,
        precision=jax.lax.Precision.HIGHEST,
    )
  elif order == 5:
    B_16 = B_64.astype(jnp.bfloat16)
    term1 = backend.matmul(U_64, B_64, precision=jax.lax.Precision.HIGHEST)
    term2 = backend.matmul(
        term1.astype(jnp.bfloat16),
        B_16,
        precision=jax.lax.Precision.DEFAULT,
    )
    U_refined = U_64 - 0.5 * term1 + 0.375 * term2
  else:
    msg = f"Valid values for `order` are 3 and 5, got {order}."
    ValueError(msg)
  return U_refined


def _refine_unitarity_16to32(U_16, order, backend):
  """Takes a bfloat16 unitary and refines it to a float32 unitary.

  Args:
    U_16: A bfloat16 unitary matrix.
    order: Order of the polynomial to use. Options are 3, 5, and 7.
    backend: `distla_core`.linalg.backends `distributed_backend` or `serial_backend`
  Returns:
    The same matrix refined to a float32-unitary.
  """
  # REDACTED Once the ef57 division bug is fixed, change the decimal factors
  # back to fractions. 0.3125 is 5/16, 0.375 is 3/8.
  U_32 = U_16.astype(jnp.float32)
  eye_32 = backend.eye_like(U_32)
  B_32 = backend.matmul(
      U_32.conj(),
      U_32,
      transpose_A=True,
      precision=jax.lax.Precision.DEFAULT,
  ) - eye_32
  if order == 3:
    U_refined = U_32 - 0.5 * backend.matmul(
        U_32,
        B_32,
        precision=jax.lax.Precision.HIGH,
    )
  elif order == 5:
    B_16 = B_32.astype(jnp.bfloat16)
    term1 = backend.matmul(U_32, B_32, precision=jax.lax.Precision.HIGH)
    term2 = backend.matmul(
        term1.astype(jnp.bfloat16),
        B_16,
        precision=jax.lax.Precision.DEFAULT,
    )
    U_refined = U_32 - 0.5 * term1 + 0.375 * term2
  elif order == 7:
    B_16 = B_32.astype(jnp.bfloat16)
    term1 = backend.matmul(U_32, B_32, precision=jax.lax.Precision.HIGH)
    term2 = backend.matmul(
        term1.astype(jnp.bfloat16),
        B_16,
        precision=jax.lax.Precision.DEFAULT,
    )
    term3 = backend.matmul(
        term2,
        B_16,
        precision=jax.lax.Precision.DEFAULT,
    )
    U_refined = U_32 - 0.5 * term1 + 0.375 * term2 - 0.3125 * term3
  else:
    msg = f"Valid values for `order` are 3, 5, and 7, got {order}."
    ValueError(msg)
  return U_refined


def _refine_unitarity_bare(
    U,
    target_dtype,
    backend,
    order_16to32,
    order_32to64,
):
  """The unjitted, unpmapped refinement function."""
  orig_dtype = U.dtype
  assert target_dtype in utils.valid_dtypes
  if target_dtype == orig_dtype:
    return U
  if orig_dtype == jnp.bfloat16:
    U = _refine_unitarity_16to32(U, order_16to32, backend)
  if target_dtype == U.dtype:
    return U
  U = _refine_unitarity_32to64(U, order_32to64, backend)
  return U


_refine_unitarity_jit = jax.jit(
    _refine_unitarity_bare,
    static_argnums=(1, 2, 3, 4),
)
_refine_unitarity_pmap = pops.pmap(
    _refine_unitarity_bare,
    static_broadcasted_argnums=(1, 2, 3, 4),
)


def refine_unitarity(U, target_dtype, p_sz=128, order_16to32=7, order_32to64=5):
  """Refine a low-precision unitary to a higher precision.

  Note that such a refinement is not unique: There are e.g. many float64
  unitaries which, when truncated to float32, are unitary to the precision
  allowed by float32. We choose here the one that preserves the singular vectors
  as well as possible, and only modifies the singular values.

  `refine_unitarity` can be applied to single-core matrices or to distributed
  matrices, and called inside or outside a pmap. It should not be jitted, it
  does that internally.

  Args:
    U: A matrix that is unitary to the precision its dtype allows.
    target_dtype: The dtype we would like to refine U to.
    p_sz: Optional; SUMMA panel size. Only used if `U` is a distributed. 128 by
      default.
    order_16to32: Optional; The order of the polynomial to use when refining
      from bfloat16 to float32. Options are 3, 5, and 7. 7 is safe for all
      matrices, but in many cases 3 or 5 may be sufficient. 7 by default.
    order_32to64: Optional; The order of the polynomial to use when refining
      from float32 to bfloat16. Options are 3 and 5. 5 is safe for all matrices,
      but in many cases 3 may be sufficient. 5 by default.
  Returns:
    `U` promoted to `target_dtype`, and refined so that it is unitary to the
    precision allowed by `target_dtype`.
  """
  # Set precision to None, to make sure each call to backend.matmul has to
  # specify a precision.
  s_backend = serial_backend.SerialBackend(precision=None)
  d_backend = distributed_backend.DistributedBackend(p_sz, precision=None)
  distribution_type = pops.distribution_type(U)
  if distribution_type == "distributed":
    return _refine_unitarity_pmap(
        U,
        target_dtype,
        d_backend,
        order_16to32,
        order_32to64,
    )
  if distribution_type == "traced":
    return _refine_unitarity_bare(
        U,
        target_dtype,
        d_backend,
        order_16to32,
        order_32to64,
    )
  if distribution_type == "undistributed":
    return _refine_unitarity_jit(
        U,
        target_dtype,
        s_backend,
        order_16to32,
        order_32to64,
    )
