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
"""Temporary work around since cloud ASICs do not currently support complex64.

Wrapper functions and class for jax.DeviceArrays.
"""
import jax
from jax import tree_util
import jax.numpy as jnp


class _IndexUpdateRef:

  def __init__(self, real_ref, imag_ref):
    self.real_ref = real_ref
    self.imag_ref = imag_ref

  def set(self, value):
    return ComplexDeviceArray(
        self.real_ref.set(value.real), self.imag_ref.set(value.imag))


class _IndexUpdateHelper:

  def __init__(self, real, imag):
    self.real = real
    self.imag = imag

  def __getitem__(self, args):
    return _IndexUpdateRef(self.real.at[args], self.imag.at[args])


class ComplexDeviceArray:
  """Complex numbers work around since CloudASIC do not support complex values.

  See b/162373657.
  """

  def __init__(self, real, imag):
    self.real = real
    self.imag = imag

  def reshape(self, new_shape):
    return ComplexDeviceArray(
        self.real.reshape(new_shape), self.imag.reshape(new_shape))

  def transpose(self, perm):
    return ComplexDeviceArray(
        self.real.transpose(perm), self.imag.transpose(perm))

  def __add__(self, other):
    if not isinstance(other, ComplexDeviceArray):
      raise TypeError('ComplexDeviceArray cannot be added to object of '
                      f'type {type(other)}')
    return ComplexDeviceArray(self.real + other.real, self.imag + other.imag)

  def __sub__(self, other):
    if not isinstance(other, ComplexDeviceArray):
      raise TypeError('ComplexDeviceArray cannot be added to object of '
                      f'type {type(other)}')
    return ComplexDeviceArray(self.real - other.real, self.imag - other.imag)

  def __mul__(self, other):
    return ComplexDeviceArray(self.real * other.real - self.imag * other.imag,
                              self.real * other.imag + self.imag * other.real)

  def __truediv__(self, other):
    Z = jnp.power(other.real, 2) + jnp.power(other.imag, 2)
    real_part = (self.real * other.real + self.imag * other.imag) / Z
    imag_part = (self.imag * other.real - self.real * other.imag) / Z
    return ComplexDeviceArray(real_part, imag_part)

  def __matmul__(self, other):
    return dot(self, other)

  @property
  def dtype(self):
    return self.real.dtype

  def __rmul__(self, other):
    return self * other

  @property
  def shape(self):
    return self.real.shape

  @property
  def at(self):
    return _IndexUpdateHelper(self.real, self.imag)

  def conj(self):
    #NOTE (mganahl): is it save to pass by reference?
    return ComplexDeviceArray(self.real, -self.imag)

  def __getitem__(self, args):
    return ComplexDeviceArray(self.real[args], self.imag[args])

  def __repr__(self):
    return f"real: {self.real.__repr__()}, imag:{self.imag.__repr__()}"

  @property
  def ndim(self):
    return self.real.ndim


# Registers ComplexDeviceArray with JAX
tree_util.register_pytree_node(ComplexDeviceArray, lambda val: (
    [val.real, val.imag], None), lambda _, vals: ComplexDeviceArray(*vals))


def moveaxis(tensor, src, dest):
  if isinstance(tensor, ComplexDeviceArray):
    real = jnp.moveaxis(tensor.real, src, dest)
    imag = jnp.moveaxis(tensor.imag, src, dest)
    return ComplexDeviceArray(real, imag)
  return jnp.moveaxis(tensor, src, dest)


def reshape(tensor, new_shape):
  if isinstance(tensor, ComplexDeviceArray):
    real = jnp.reshape(tensor.real, new_shape)
    imag = jnp.reshape(tensor.imag, new_shape)
    return ComplexDeviceArray(real, imag)
  return jnp.reshape(tensor, new_shape)


def transpose(tensor, permutation):
  if isinstance(tensor, ComplexDeviceArray):
    real = jnp.transpose(tensor.real, permutation)
    imag = jnp.transpose(tensor.imag, permutation)
    return ComplexDeviceArray(real, imag)
  return jnp.transpose(tensor, permutation)


def all_to_all(tensor, *args, **kwargs):
  if isinstance(tensor, ComplexDeviceArray):
    real = jax.lax.all_to_all(tensor.real, *args, **kwargs)
    imag = jax.lax.all_to_all(tensor.imag, *args, **kwargs)
    return ComplexDeviceArray(real, imag)
  return jax.lax.all_to_all(tensor, *args, **kwargs)


def pswapaxes(tensor, name, axis):
  if isinstance(tensor, ComplexDeviceArray):
    real = jax.lax.pswapaxes(tensor.real, name, axis)
    imag = jax.lax.pswapaxes(tensor.imag, name, axis)
    return ComplexDeviceArray(real, imag)
  return jax.lax.pswapaxes(tensor, name, axis)


def tensordot(a, b, axes, precision=jax.lax.Precision.DEFAULT):

  ##################################
  # FIXME (mganahl): FIX WORKNIGAREA code and remove this part
  if isinstance(a, tuple):
    a = ComplexDeviceArray(a[0], a[1])
    assert isinstance(b, ComplexDeviceArray)
  ##################################
  if isinstance(a, ComplexDeviceArray) or isinstance(b, ComplexDeviceArray):
    real = jnp.tensordot(
        a.real, b.real, axes, precision=precision) - jnp.tensordot(
            a.imag, b.imag, axes, precision=precision)
    imag = jnp.tensordot(
        a.imag, b.real, axes, precision=precision) + jnp.tensordot(
            a.real, b.imag, axes, precision=precision)
    return ComplexDeviceArray(real, imag)

  return jnp.tensordot(a, b, axes, precision=precision)


def zeros_like(tensor):
  if isinstance(tensor, ComplexDeviceArray):
    real = jnp.zeros(shape=tensor.shape, dtype=tensor.dtype)
    imag = jnp.zeros(shape=tensor.shape, dtype=tensor.dtype)
    return ComplexDeviceArray(real, imag)
  return jnp.zeros_like(tensor)


def zeros(shape, dtype, arithmetic_dtype):
  if arithmetic_dtype is complex:
    return ComplexDeviceArray(
        jnp.zeros(shape=shape, dtype=dtype), jnp.zeros(
            shape=shape, dtype=dtype))
  return jnp.zeros(shape=shape, dtype=dtype)


def array(value, dtype, arithmetic_dtype):
  if arithmetic_dtype is complex:
    return ComplexDeviceArray(
        jnp.array(value).real.astype(dtype),
        jnp.array(value).imag.astype(dtype))
  return jnp.array(value).real.astype(dtype)


def einsum(subscripts, *operands, precision=jax.lax.Precision.DEFAULT):
  if len(operands) != 2:
    raise ValueError(f"only two operands allowed in einsum, "
                     f"got {len(operands)} instead")
  if isinstance(operands[0], ComplexDeviceArray) and isinstance(
      operands[1], ComplexDeviceArray):
    real = jnp.einsum(subscripts, operands[0].real, operands[1].real,
                      precision=precision) -\
           jnp.einsum(subscripts, operands[0].imag, operands[1].imag,
                      precision=precision)
    imag = jnp.einsum(subscripts, operands[0].real, operands[1].imag,
                      precision=precision) +\
           jnp.einsum(subscripts, operands[0].imag, operands[1].real,
                      precision=precision)
    return ComplexDeviceArray(real, imag)
  return jnp.einsum(subscripts, *operands, precision=precision)


def sqrt(tensor):
  """
  Compute the elementwise principal square root of `tensor`.
  """
  if isinstance(tensor, ComplexDeviceArray):
    rad = jnp.sqrt(
        jnp.abs(jnp.power(tensor.real, 2) + jnp.power(tensor.imag, 2)))
    real = jnp.sqrt((rad + tensor.real) / 2)
    sign = jnp.sign(tensor.imag)
    imag = jnp.sqrt(jnp.abs(rad - tensor.real) / 2)
    return ComplexDeviceArray(real, sign * imag)
  return jnp.sqrt(tensor)


def conj(tensor):
  return tensor.conj()


def conjubuilding_block(tensor):
  return tensor.conj()


def dot(a, b, precision=jax.lax.Precision.HIGHEST):
  """
  Wrapper for jnp.dot for ComplexDeviceArray
  """
  if isinstance(a, ComplexDeviceArray) or isinstance(b, ComplexDeviceArray):
    real = jnp.dot(
        a.real, b.real, precision=precision) - jnp.dot(
            a.imag, b.imag, precision=precision)
    imag = jnp.dot(
        a.imag, b.real, precision=precision) + jnp.dot(
            a.real, b.imag, precision=precision)
    return ComplexDeviceArray(real, imag)

  return jnp.dot(a, b, precision=precision)
