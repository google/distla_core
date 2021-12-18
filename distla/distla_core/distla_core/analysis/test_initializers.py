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
import pytest
import numpy as np

from jax import lax
import jax.numpy as jnp

from distla_core.analysis import initializers
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

precisions = [lax.Precision.HIGHEST]
dtypes = [jnp.float32]
seeds = [1, 10]
sizes = [4, 8]
bools = [False, True]
distributions = ["linear", "geometric"]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("n_cols", sizes)
@pytest.mark.parametrize("serial", bools)
def test_random_fixed_norm(n_rows, n_cols, dtype, seed, serial):
  """
  Tests that random_fixed norm produces a matrix of given Frobenius norm.
  """
  norm = 100.
  matrix = initializers.random_fixed_norm((n_rows, n_cols),
                                          dtype,
                                          seed=seed,
                                          serial=serial,
                                          norm=norm)
  if not serial:
    matrix = pops.undistribute(matrix)
  assert matrix.shape == (n_rows, n_cols)
  eps = testutils.eps(lax.Precision.HIGHEST, dtype)
  np.testing.assert_allclose([norm], [np.linalg.norm(matrix)], norm * eps)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("n_cols", sizes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("serial", bools)
def test_random_isometry(n_rows, n_cols, dtype, seed, precision, serial):
  """
  Tests that random_isometry produces an isometry, and when square, a unitary.
  """
  if serial and precision != lax.Precision.HIGHEST:
    pytest.skip("Precision not used in the serial case.")

  if n_cols > n_rows:
    with pytest.raises(ValueError):
      matrix = initializers.random_isometry((n_rows, n_cols),
                                            dtype,
                                            seed=seed,
                                            serial=serial)
    return
  matrix = initializers.random_isometry((n_rows, n_cols),
                                        dtype,
                                        seed=seed,
                                        serial=serial)
  if not serial:
    matrix = pops.undistribute(matrix)
  assert matrix.shape == (n_rows, n_cols)
  testutils.test_unitarity(
      matrix, precision=precision, eps_coef=2 * max(n_rows, n_cols))


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("n_cols", sizes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("serial", bools)
@pytest.mark.parametrize("return_factors", bools)
def test_from_singular_spectrum(n_rows, n_cols, dtype, seed, precision, serial,
                                return_factors):
  """
  Tests that `random_from_singular_spectrum` actually produces a matrix with
  the given spectrum of singular values.
  """
  if serial and precision != lax.Precision.HIGHEST:
    pytest.skip("Precision not used in the serial case.")
  if not serial and n_cols > n_rows:
    pytest.skip("Polar does not yet support n_cols > n_rows.")
  return_factors = True

  np.random.seed(seed)
  spectrum = np.random.randn(min(n_rows, n_cols)).astype(dtype)
  spectrum = np.sort(np.abs(spectrum))[::-1]
  matrix = initializers.random_from_singular_spectrum(
      (n_rows, n_cols),
      spectrum,
      dtype=dtype,
      seed=seed,
      serial=serial,
      precision=precision,
      return_factors=return_factors)

  if return_factors:
    matrix, U, V = matrix
  if not serial:
    matrix = pops.undistribute(matrix)
    if return_factors:
      U = pops.undistribute(U)
      V = pops.undistribute(V)

  eps = testutils.eps(precision, dtype=dtype)
  if return_factors:
    recon = jnp.dot((U * spectrum), V.conj().T, precision=lax.Precision.HIGHEST)
    testutils.assert_allclose(matrix, recon, max(n_rows, n_cols) * eps)

  assert matrix.shape == (n_rows, n_cols)
  spectrum_out = np.linalg.svd(matrix, compute_uv=False)
  testutils.assert_allclose(spectrum, spectrum_out, max(n_rows, n_cols) * eps)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("serial", bools)
@pytest.mark.parametrize("return_factors", bools)
def test_from_eigenspectrum(n_rows, dtype, seed, precision, serial,
                            return_factors):
  """
  Tests that `random_from_eigenspectrum` actually produces a matrix with
  the given eigenspectrum.

  TODO: Add complex support.
  """
  if serial and precision != lax.Precision.HIGHEST:
    pytest.skip("Precision not used in the serial case.")
  return_factors = True

  np.random.seed(seed)
  spectrum = np.random.randn(n_rows).astype(dtype)
  spectrum = np.sort(spectrum)
  matrix = initializers.random_from_eigenspectrum(
      spectrum,
      dtype=dtype,
      seed=seed,
      serial=serial,
      precision=precision,
      return_factors=return_factors)
  if return_factors:
    matrix, V = matrix
  if not serial:
    matrix = pops.undistribute(matrix)
    if return_factors:
      V = pops.undistribute(V)

  eps = testutils.eps(precision, dtype=dtype)
  if return_factors:
    recon = jnp.dot((V * spectrum), V.conj().T, precision=lax.Precision.HIGHEST)
    testutils.assert_allclose(matrix, recon, n_rows * eps)
  assert matrix.shape == (n_rows, n_rows)
  spectrum_out = np.sort(np.linalg.eigvals(matrix))
  testutils.assert_allclose(spectrum, spectrum_out, n_rows * eps)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("n_rows", sizes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("serial", bools)
@pytest.mark.parametrize("right_vectors", bools)
def test_combine_spectrum(n_rows, dtype, precision, serial, right_vectors):
  np.random.seed(10)
  spectrum = np.random.randn(n_rows).astype(dtype)
  left_vectors = np.random.randn(n_rows, n_rows).astype(dtype)
  left_vectors, _ = np.linalg.qr(left_vectors)
  if right_vectors:
    right_vectors = np.random.randn(n_rows, n_rows).astype(dtype)
    right_vectors, _ = np.linalg.qr(right_vectors)
    expected = (left_vectors * spectrum) @ (right_vectors.conj().T)
    if not serial:
      right_vectors = pops.distribute(right_vectors)
  else:
    expected = (left_vectors * spectrum) @ (left_vectors.conj().T)
    right_vectors = None

  if not serial:
    left_vectors = pops.distribute(left_vectors)
  result = initializers.combine_spectrum(
      spectrum, left_vectors, right_vectors=right_vectors, precision=precision)
  if not serial:
    result = pops.undistribute(result)
  eps = testutils.eps(precision, dtype)
  testutils.assert_allclose(result, expected, 50 * eps)


def _test_distribution(distribution, array):
  if distribution == "geometric":
    array = np.log(array + np.abs(array[0]) + 1.)
  diff = np.diff(array)
  diff_0 = diff - diff[0]
  testutils.assert_allclose(diff_0, np.zeros_like(diff_0), 0.1)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("eig_bounds", [(-10, 100), (1, 1), (0, 0)])
@pytest.mark.parametrize("n_eigs", sizes)
@pytest.mark.parametrize("gap_size", [0.01, 0.5])
@pytest.mark.parametrize("gap_position", [0, 3, None])
@pytest.mark.parametrize("distribution", distributions)
def test_gapped_real_eigenspectrum(n_eigs, eig_bounds, gap_position, gap_size,
                                   distribution, dtype):
  eig_min, eig_max = eig_bounds
  error = eig_min > eig_max or n_eigs < 3
  gap_start = np.median([eig_min, eig_max])
  if gap_position is not None:
    error = error or gap_start <= eig_min or gap_size + gap_start > eig_max
    error = error or eig_min == eig_max or gap_position < 1
    error = error or gap_position >= n_eigs - 2 or gap_size < 0
  if error:
    with pytest.raises(ValueError):
      spectrum = initializers.gapped_real_eigenspectrum(
          n_eigs,
          eig_min,
          eig_max,
          gap_position=gap_position,
          gap_start=gap_start,
          gap_size=gap_size,
          distribution=distribution,
          dtype=dtype)
    return

  spectrum = initializers.gapped_real_eigenspectrum(
      n_eigs,
      eig_min,
      eig_max,
      gap_position=gap_position,
      gap_start=gap_start,
      gap_size=gap_size,
      distribution=distribution,
      dtype=dtype)
  assert np.abs(np.abs(spectrum[0]) - np.abs(eig_min)) < 1E-4
  assert np.abs(np.abs(spectrum[-1]) - np.abs(eig_max)) < 1E-4
  testutils.assert_allclose(np.sort(spectrum), spectrum, 0.)
  assert spectrum.dtype == dtype
  assert spectrum.size == n_eigs

  if gap_position is not None:
    before_gap = spectrum[:gap_position]
    after_gap = spectrum[gap_position:]
    assert np.abs(after_gap[0] - before_gap[-1] - gap_size) < 1E-4
    _test_distribution(distribution, before_gap)
    _test_distribution(distribution, after_gap)
  else:
    _test_distribution(distribution, spectrum)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("sv_bounds", [(0.2, 100.), (1., 1.)])
@pytest.mark.parametrize("n_nonzero", sizes)
@pytest.mark.parametrize("n_zeros", sizes)
@pytest.mark.parametrize("distribution", distributions)
def test_sv_spectrum(n_nonzero, sv_bounds, n_zeros, distribution, dtype):
  sv_min, sv_max = sv_bounds
  spectrum = initializers.sv_spectrum(
      n_nonzero,
      sv_min,
      sv_max,
      distribution=distribution,
      n_zeros=n_zeros,
      dtype=dtype)
  testutils.assert_allclose(np.sort(spectrum)[::-1], spectrum, 0.)
  spectrum = np.array(spectrum)
  spectrum_zeros = spectrum[::-1][:n_zeros]
  np.testing.assert_equal(spectrum_zeros, np.zeros(n_zeros, dtype=dtype))
  np.testing.assert_allclose([spectrum[n_nonzero - 1]], [sv_min], atol=1E-4)
  assert np.abs(np.abs(spectrum[0]) - np.abs(sv_max)) < 1E-4
  assert spectrum.dtype == dtype
  assert spectrum.size == n_nonzero + n_zeros

  _test_distribution(distribution, spectrum[:n_nonzero][::-1])
