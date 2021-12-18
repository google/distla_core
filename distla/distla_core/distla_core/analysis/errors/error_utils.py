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
"""
Functions to compute error metrics.
"""

import functools
import jax
from jax import lax
import numpy as np

from distla_core.analysis import pmaps
from distla_core.utils import pops


def comparison_header(name_a, name_b, relative):
  """
  Generates an appropriate header string based on `name_a` of `matrix_a` and
  `name_b` of `matrix_b`.
  """
  header = f"||{name_a} - {name_b}||_F"
  if relative:
    header += f"/||{name_a}||_F"
  return header


def comparison_error(matrix_a, matrix_b, relative, name_a, name_b):
  """ Measures how nearly `matrix_a == matrix_b`, `||matrix_a - matrix_b||_F`,
  transparently handling both the distributed and undistributed cases.
  Also divides the error by `||matrix_a||_F` if `relative` is True.


  Args:
    matrix_a: First matrix to be compared.
    matrix_b: Second matrix to be compared.
    relative: Flags whether the relative or absolute error is computed.
  Returns:
    err: ||matrix_a - matrix-b||_F, divided by ||matrix_a||_F if relative.
    header: A string "||name_a - name_b||_F", with "/||name_a||_F" appended
      if relative.
  """
  err = pmaps.frobdiff(matrix_a, matrix_b)
  if relative:
    err /= pmaps.frobnorm(matrix_a)
  header = comparison_header(name_a, name_b, relative)
  return err, header


def isometry_error(matrix, p_sz=128, precision=lax.Precision.HIGHEST,
                   relative=False, dagger_left=True, name="matrix"):
  """ Measures how nearly `matrix` is a left isometry
  (`||matrix^H matrix - I||`, if `dagger_left` is `True`) or a right isometry
  (`||matrix matrix^H - I||`, if `dagger_left` is `False`). The serial and
  distributed cases are handled transparently.

  Args:
    matrix: The result to be tested.
    p_sz: p_sz for the SUMMA multiplications. This is only used in the
      distributed case.
    precision: ASIC matmul precision.
    relative: If `True`, errors are divided by `||I||_F`, and the
      header string is correspondingly modified.
    name: String to be used in place of `matrix` in the header strings.
  Returns:
    err: The error.
    header: The header.
  """
  should_name = f"{name} {name}^H"
  if dagger_left:
    should_name = f"{name}^H {name}"
  header = comparison_header("I", should_name, relative)
  if matrix is None:
    return -1, header

  if dagger_left:
    should_be_eye = pmaps.matmul(matrix, matrix, transpose_a=True,
                                 conj_a=True, p_sz=p_sz, precision=precision)
  else:
    should_be_eye = pmaps.matmul(matrix, matrix, transpose_b=True,
                                 conj_b=True, p_sz=p_sz, precision=precision)
    should_name = f"{name} {name}^H"
  eye = pmaps.eye(should_be_eye.shape, should_be_eye.dtype)
  return comparison_error(eye, should_be_eye, relative, "I", should_name)


def hermiticity_error(matrix, relative=False, name="matrix"):
  """ Measures how nearly `matrix` is Hermitian, `||matrix - matrix^H||_F`.
  The serial and distributed cases are handled transparently.

  Args:
    matrix: The result to be tested.
    relative: If `True`, errors are divided by `||matrix||_F`, and the
      header string is correspondingly modified.
    name: String to be used in place of `matrix` in the header strings.
  Returns:
    err: The error.
    header: The header.
  """
  header = comparison_header(name, name + "^H", relative)
  if matrix is None:
    return -1, header
  matrix_t = pmaps.transpose(matrix, conjubuilding_block=True)
  return comparison_error(matrix, matrix_t, relative, name, name + "^H")


def reconstruction_error(matrix, factors, p_sz=128, relative=False,
                         name="matrix", recon_name="prod(factors)",
                         precision=lax.Precision.HIGHEST):
  """ Measures how nearly `factors = prod(factors)`;
  `||factors - prod(factors)||_F`.
  The serial and distributed cases are handled transparently.
  Args:
    matrix: The result to be tested.
    p_sz: p_sz for the SUMMA multiplications. This is only used in the
      distributed case.
    relative: If `True`, errors are divided by `||matrix||_F`, and the
      header string is correspondingly modified.
    name: String to be used in place of `matrix` in the header string.
    recon_name: String to be used in place of `prod(factors)` in the header
      string.
    precision: ASIC matmul precision.
  Returns:
    err: The error.
    header: The header.
  """
  header = comparison_header(name, recon_name, relative)
  if matrix is None or factors is None:
    return -1, header
  mult_f = functools.partial(pmaps.matmul, p_sz=p_sz, precision=precision)
  reconstructed = functools.reduce(mult_f, factors)
  return comparison_error(matrix, reconstructed, relative, name, recon_name)


def idempotency_error(matrix, p_sz=128, relative=False, name="matrix",
                      precision=lax.Precision.HIGHEST):
  """ Measures how nearly `matrix = matrix^2`;
  `||matrix - matrix^2||_F`.
  The serial and distributed cases are handled transparently.
  Args:
    matrix: The result to be tested.
    p_sz: p_sz for the SUMMA multiplications. This is only used in the
      distributed case.
    relative: If `True`, errors are divided by `||matrix||_F`, and the
      header string is correspondingly modified.
    name: String to be used in place of `matrix` in the header string.
    precision: ASIC matmul precision.
  Returns:
    err: The error.
    header: The header.
  """
  matrix_2_name = name + "^2"
  header = comparison_header(name, matrix_2_name, relative)
  if matrix is None:
    return -1, header
  matrix_2 = pmaps.matmul(matrix, matrix, p_sz=p_sz, precision=precision)
  return comparison_error(matrix, matrix_2, relative, name, matrix_2_name)


def subspace_header(name_a, name_b):
  return f"max_angle({name_a}, {name_b})"


def subspace_angle(subspace_a, subspace_b, p_sz=128, name_a="result",
                   name_b="expected", precision=lax.Precision.HIGHEST):
  """ Measures how nearly `subspace_a` and `subspace_b` span the same subspace.
  At present, this function only works for small matrices, since the bulk of
  the computation must be done on the host.
  """
  max_size = 3600 ** 2
  header = subspace_header(name_a, name_b)
  if subspace_a is None or subspace_b is None:
    return -1, header
  if subspace_a.size > max_size or subspace_b.size > max_size:
    return -1, header

  product = pmaps.matmul(subspace_a, subspace_b, p_sz=p_sz, precision=precision,
                         conj_a=True, transpose_a=True)
  if product.ndim == 3:
    product = pops.undistribute(product)
  product, _ = np.linalg.qr(product)
  svs = np.linalg.svd(product, compute_uv=False)
  result = np.arccos(svs[-1])
  return result, header
