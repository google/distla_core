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
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest
from distla_core.analysis.errors import error_utils
from distla_core.utils import pops
from distla_core.linalg.utils import testutils


serial = [True, False]
sizes = [4, 8, 12]
dtypes = [jnp.float32]
flags = [True, False]


@pytest.mark.parametrize("serial", serial)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("relative", flags)
def test_comparison_error(n_rows, serial, dtype, relative):
  np.random.seed(1)
  matrix_a = np.random.randn(n_rows, n_rows).astype(dtype)
  matrix_b = np.random.randn(n_rows, n_rows).astype(dtype)
  name_a = "A"
  name_b = "B"
  expected_err = np.linalg.norm(matrix_a - matrix_b)
  expected_header = "||A - B||_F"
  if relative:
    expected_err /= np.linalg.norm(matrix_a)
    expected_header += "/||A||_F"
  if not serial:
    matrix_a = pops.distribute(matrix_a)
    matrix_b = pops.distribute(matrix_b)
  err, header = error_utils.comparison_error(
    matrix_a, matrix_b, relative, name_a, name_b)
  eps = 100 * testutils.eps(lax.Precision.HIGHEST, dtype)
  assert np.abs(err - expected_err) < eps
  assert header == expected_header


@pytest.mark.parametrize("serial", serial)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("relative", flags)
@pytest.mark.parametrize("dagger_left", flags)
def test_isometry_error(n_rows, serial, dtype, relative, dagger_left):
  np.random.seed(1)
  matrix = np.random.randn(n_rows, n_rows).astype(dtype)
  name = "U"
  if dagger_left:
    should = (matrix.conj().T) @ matrix
    expected_header = f"||I - {name}^H {name}||_F"
  else:
    should = matrix @ (matrix.conj().T)
    expected_header = f"||I - {name} {name}^H||_F"
  eye = np.eye(*(should.shape), dtype=should.dtype)
  expected_err = np.linalg.norm(eye - should)

  if relative:
    expected_err /= np.linalg.norm(eye)
    expected_header += "/||I||_F"
  if not serial:
    matrix = pops.distribute(matrix)
  err, header = error_utils.isometry_error(
    matrix, relative=relative, name=name, dagger_left=dagger_left)
  eps = 100 * testutils.eps(lax.Precision.HIGHEST, dtype)
  assert np.abs(err - expected_err) < eps
  assert header == expected_header


@pytest.mark.skip("Transpose doesn't work yet.")
@pytest.mark.parametrize("serial", serial)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("relative", flags)
def test_hermiticity_error(n_rows, serial, dtype, relative):
  np.random.seed(1)
  matrix = np.random.randn(n_rows, n_rows).astype(dtype)
  name = "H"
  expected_err = np.linalg.norm(matrix - matrix.conj().T)
  expected_header = f"||{name} - {name}^H||_F"

  if relative:
    expected_err /= np.linalg.norm(matrix)
    expected_header += f"/||{name}||_F"
  if not serial:
    matrix = pops.distribute(matrix)
  err, header = error_utils.hermiticity_error(
    matrix, relative=relative, name=name)
  eps = 50 * testutils.eps(lax.Precision.HIGHEST, dtype)
  assert np.abs(err - expected_err) < eps
  assert header == expected_header


@pytest.mark.parametrize("serial", serial)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("relative", flags)
def test_reconstruction_error(n_rows, serial, dtype, relative):
  np.random.seed(1)
  matrix = np.random.randn(n_rows, n_rows).astype(dtype)
  matrix_b = np.random.randn(n_rows, n_rows).astype(dtype)
  result = matrix @ matrix_b
  name = "M"
  recon_name = "UH"
  expected_err = np.linalg.norm(result - matrix @ matrix_b)
  expected_header = f"||{name} - {recon_name}||_F"
  if relative:
    expected_err /= np.linalg.norm(matrix)
    expected_header += f"/||{name}||_F"
  if not serial:
    matrix = pops.distribute(matrix)
    matrix_b = pops.distribute(matrix_b)
    result = pops.distribute(result)
  err, header = error_utils.reconstruction_error(
    result, (matrix, matrix_b), relative=relative, name=name,
    recon_name=recon_name)
  eps = 50 * testutils.eps(lax.Precision.HIGHEST, dtype)
  assert np.abs(err - expected_err) < eps
  assert header == expected_header


@pytest.mark.parametrize("serial", serial)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("relative", flags)
def test_idempotency_error(n_rows, serial, dtype, relative):
  np.random.seed(1)
  matrix = np.random.randn(n_rows, n_rows).astype(dtype)
  result = matrix @ matrix
  name = "M"
  recon_name = "M^2"
  expected_err = np.linalg.norm(matrix - result)
  expected_header = f"||{name} - {recon_name}||_F"
  if relative:
    expected_err /= np.linalg.norm(matrix)
    expected_header += f"/||{name}||_F"
  if not serial:
    matrix = pops.distribute(matrix)
  err, header = error_utils.idempotency_error(
    matrix, relative=relative, name=name)
  eps = 50 * testutils.eps(lax.Precision.HIGHEST, dtype)
  assert np.abs(err - expected_err) < eps
  assert header == expected_header
