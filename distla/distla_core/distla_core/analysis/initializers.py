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
import functools
import time

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.analysis import pmaps
from distla_core.linalg.tensor import utils
from distla_core.utils import pops
from distla_core.utils import vops


def random_fixed_norm(shape, dtype, seed=None, serial=False, norm=None):
  """
  Generates a Gaussian random matrix of shape `shape` and dtype `dtype`,
  seeded by seed `seed` (system clock if None), and Frobenius-normalized
  by norm `norm`.

  Args:
    shape: Shape of the matrix.
    dtype: dtype of the matrix.
    seed: The random seed; system clock if None.
    serial: If False (True), the matrix is distributed (not distributed).
      Default False.
    norm: Frobenius norm of the output; unfixed if None.
  Returns:
    The matrix.
  """
  if serial:
    return _random_fixed_norm_serial(shape, dtype, seed, norm)
  else:
    return _random_fixed_norm_distributed(shape, dtype, seed, norm)


def _random_fixed_norm_serial(shape, dtype, seed, norm):
  """ Handles `random_fixed_norm` for `serial=False`.
  """
  if seed is None:
    seed = int(time.time())
  key = jax.random.PRNGKey(seed)
  matrix = jax.random.normal(key, shape=shape, dtype=dtype)
  if norm is not None:
    matrix *= norm / jnp.linalg.norm(matrix)
  return matrix


@functools.partial(pops.pmap, in_axes=(0, None))
def _normalize(matrix, norm):
  matrix *= norm / pops.frobnorm(matrix)
  return matrix


def _random_fixed_norm_distributed(shape, dtype, seed, norm):
  """ Handles `random_fixed_norm` for `serial=True`.
  """
  if seed is None:
    seed = int(time.time())
  matrix = utils.normal(shape, pops.GRID, dtype=dtype, seed=seed)
  if norm is not None:
    matrix = _normalize(matrix, norm)
  return matrix


def random_isometry(shape, dtype, seed=None, serial=False, p_sz=128,
                    precision=lax.Precision.HIGHEST):
  """
  Generates a random unitary/isometric matrix (a matrix of random elements
  seeded by seed `seed` which is subsequently orthogonalized to precision
  `precision`) of shape `shape` and dtype `dtype`.

  Args:
    shape: Shape of the matrix.
    dtype: dtype of the matrix.
    seed: The random seed; system clock if None.
    serial: If False (True), the matrix is distributed (not distributed).
            Default False.
    p_sz: With `serial=False`, the SUMMA p_sz for the orthogonalization.
    precision: With `serial=False`, the SUMMA precision for the
               orthogonalization.
  Returns:
    The matrix.
  """
  if shape[0] < shape[1]:
    raise ValueError(f"shape={shape} must not have shape[0] < shape[1].")
  if serial:
    return _random_isometry_serial(shape, dtype, seed)
  else:
    return _random_isometry_distributed(shape, dtype, seed, p_sz, precision)


def _random_isometry_serial(shape, dtype, seed):
  """
  Handles `random_isometry` when `serial=True`. See that docstring for details.
  The orthogonalization is handled with a QR decomposition in this case.
  """
  if seed is None:
    seed = int(time.time())
  key = jax.random.PRNGKey(seed)
  matrix = jax.random.normal(key, shape=shape, dtype=dtype)
  if dtype == jnp.bfloat16:
    matrix, _ = jnp.linalg.qr(matrix.astype(jnp.float32))
    matrix = matrix.astype(jnp.bfloat16)
  else:
    matrix, _ = jnp.linalg.qr(matrix)
  return matrix


def _random_isometry_distributed(shape, dtype, seed, p_sz, precision):
  """
  Handles `random_isometry` when `serial=False`. See that docstring for details.
  The orthogonalization is handled with a Newton-Schulz polar decomposition
  in this case.
  """
  if seed is None:
    seed = int(time.time())
  matrix = utils.normal(shape, pops.GRID, dtype=dtype, seed=seed)
  matrix, _, _, _ = pmaps.polarU(matrix, p_sz=p_sz, precision=precision)
  return matrix


def random_from_singular_spectrum(shape, spectrum, dtype=jnp.float32, seed=None,
                                  serial=False, p_sz=128,
                                  precision=lax.Precision.HIGHEST,
                                  return_factors=False):
  """
  Generates a random matrix of shape `shape` with a given spectrum of singular
  values `spectrum`.

  Args:
    shape: Shape of the matrix.
    spectrum: Singular values of the matrix. This must be a vector of length
              min(shape).
    dtype: dtype of the matrix.
    seed: The random seed; system clock if None.
    serial: If False (True), the matrix is distributed (not distributed).
            Default False.
    p_sz: Panel size for the distributed case.
    precision: Precision of internal matrix multiplications.
    return_factors: If `True`, the singular vector matrices `U` and `V` are
      returned along with the matrix, `matrix = U @ spectrum @ V^H`.
  Returns:
    If `return_factors` is False: `matrix`.
    If `return_factors` is True: `matrix, U, V`.
  """
  if seed is None:
    seed = int(time.time())
  n_rows, n_cols = shape
  min_dim = min(n_rows, n_cols)
  if spectrum.size != min(shape):
    raise TypeError(f"spectrum.size={spectrum.size}, but this must equal "
                    f"min(shape={shape})={min(shape)}.")
  if (jnp.count_nonzero(spectrum < 0.) != 0 or
     jnp.count_nonzero(spectrum.imag) != 0):
    raise ValueError("spectrum must be real and non-negative.")
  left_vectors = random_isometry((n_rows, min_dim), dtype, seed=seed,
                                 serial=serial)
  right_vectors = random_isometry((n_cols, min_dim), dtype, seed=seed + 1,
                                  serial=serial)
  result = combine_spectrum(spectrum, left_vectors, right_vectors=right_vectors,
                            p_sz=p_sz, precision=precision)
  if return_factors:
    result = (result, left_vectors, right_vectors)
  return result


def random_from_eigenspectrum(spectrum, dtype=jnp.float32, seed=None,
                              precision=lax.Precision.HIGHEST,
                              serial=False, p_sz=128, return_factors=False):
  """
  Generates a random normal (and thus square) matrix with a given eigenspectrum
  `spectrum`.

  Args:
    spectrum: Eigenvalues of the matrix. The matrix will be square with
              `spectrum.size` rows.
    dtype: dtype of the matrix.
    seed: The random seed; system clock if None.
    precision: Precision of internal matrix multiplications.
    serial: If False (True), the matrix is distributed (not distributed).
            Default False.
    p_sz: Panel size for the distributed case.
    return_factors: If True, the eigenvectors are returned along with the
      matrix.
  Returns:
    The matrix, and the eigenvectors if `return_factors` is True.
  """
  n_rows = spectrum.size
  e_vecs = random_isometry((n_rows, n_rows), dtype, seed=seed, serial=serial)
  result = combine_spectrum(spectrum, e_vecs, p_sz=p_sz, precision=precision)
  if return_factors:
    result = (result, e_vecs)
  return result


def combine_spectrum(spectrum, left_vectors, right_vectors=None,
                     p_sz=128, precision=lax.Precision.HIGHEST):
  """ Forms the product `left_vectors @ diag(spectrum) @ right_vectors^H`.
  The distributed and serial cases are handled transparently. We take
  `right_vectors = left_vectors` if the former is unspecified.

  Args:
    spectrum: Either a np/jnp array or a `ReplicatedThinMatrix` of diagonal
      entries (typically eigen or singular values). If we are handling the
      distributed case but `spectrum` is a np/jnp array, it will be distributed,
      but the distributed entries will then be discarded; therefore it is more
      efficient to distribute first if `spectrum` is to be reused.
    left_vectors: Either a np/jnp array or a `ShardedDeviceArray` representing
      the LHS matrix (typically eigenvectors or left singular vectors).
    right_vectors: Either a np/jnp array or a `ShardedDeviceArray` representing
      the Hermitian conjubuilding_block of the RHS matrix. If None this is taken equal
      to `left_vectors` (typically right singular vectors otherwise).
    p_sz: Panel size for the SUMMA multiplications. Ignored in the serial case.
    precision: ASIC matmul precision.
  Returns:
    matrix: The product `left_vectors @ diag(spectrum) @ right_vectors^H`.
  """
  if right_vectors is None:
    right_vectors = left_vectors

  if left_vectors.ndim == 2:
    left_vectors = left_vectors * spectrum.ravel()
  else:
    if not isinstance(spectrum, vops.ReplicatedThinMatrix):
      spectrum = vops.distribute(spectrum.reshape(spectrum.size, 1),
                                 column_replicated=False,
                                 host_replicated_input=True)
    left_vectors = vops.diagmult(left_vectors, spectrum, vector_on_right=True)
    if right_vectors.ndim == 2:
      right_vectors = pops.distribute(right_vectors)

  return pmaps.matmul(left_vectors, right_vectors, transpose_b=True,
                      conj_b=True, precision=precision, p_sz=p_sz)


def gapped_real_eigenspectrum(n_eigs, eig_min, eig_max,
                              gap_position=None, gap_start=0., gap_size=0.1,
                              distribution="linear", dtype=jnp.float32):
  """
  Generates a spectrum of real eigenvalues with desired properties and an
  optional targetted gap.

  Args:
    n_eigs: Size of the spectrum.
    eig_min: The smallest (most negative) entry in the spectrum.
    eig_max: The largest entry in the spectrum. Must be >= `eig_min`.
    gap_position: Optional index at which the gap will begin. Must be at least
                  1 and no larger than n_eigs - 3. Default None, which means
                  no gap, and that `gap_start` and `gap_size` will be
                  ignored.
    gap_start: Value of `spectrum[gap_position]`. Default 0.
    gap_size: Value of `spectrum[gap_position + 1] - spectrum[gap_position]`.
              Default 0.1.
    distribution: `"linear"` or `"geometric"`; determines whether the
                  values are spaced evenly on a linear or logarithmic scale
                  respectively. Default `"linear"`.
    dtype: Dtype of the spectrum. Default `jnp.float32`.
  Returns:
    The spectrum, a length- n_eigs vector of dtype `dtype`
    containing the ascendingly-sorted eigenvalues.
  """
  if n_eigs < 3:
    raise ValueError(f"Must have n_eigs={n_eigs} at least 3.")
  if eig_min > eig_max:
    raise ValueError(f"Must have eig_min={eig_min} <= eig_max={eig_max}.")

  spectrum_f = _get_spectrum_f(distribution, dtype)
  shift = 0.
  if distribution == "geometric" and eig_min < 1:
    shift = np.abs(eig_min) + 1

  if gap_position is None:
    if eig_min == eig_max:
      spectrum = np.full(n_eigs, eig_min, dtype=dtype)
    else:
      spectrum = spectrum_f(eig_min + shift, eig_max + shift, num=n_eigs)
      spectrum -= shift
  else:
    gap_finish = gap_start + gap_size
    if gap_start <= eig_min:
      raise ValueError(
          f"gap_start={gap_start} was less than eig_min={eig_min}.")
    if eig_min == eig_max:
      raise ValueError("Cannot specify gap_position when eig_min==eig_max.")
    if gap_position >= (n_eigs - 3) or gap_position < 1:
      raise ValueError(f"If specified, gap_position={gap_position} must be "
                       f"in [1, n_eigs - 3 = {n_eigs - 3}).")
    if gap_size < 0:
      raise ValueError(f"gap_size={gap_size} must be non-negative.")
    if gap_finish > eig_max:
      raise ValueError(f"End of gap {gap_finish} exceeded eig_max={eig_max}.")
    spectrum_left = spectrum_f(eig_min + shift, gap_start + shift,
                               num=gap_position) - shift
    spectrum_right = spectrum_f(gap_finish + shift, eig_max + shift,
                                num=n_eigs - gap_position) - shift
    spectrum = np.hstack([spectrum_left, spectrum_right])
  return spectrum


def _get_spectrum_f(distribution, dtype):
  if distribution == "linear":
    spectrum_f = functools.partial(np.linspace, dtype=dtype)
  elif distribution == "geometric":
    spectrum_f = functools.partial(np.geomspace, dtype=dtype)
  else:
    raise ValueError(f"Invalid distribution {distribution}.")
  return spectrum_f


def sv_spectrum(n_nonzero, sv_min=0.1, sv_max=1., distribution="linear",
                n_zeros=0, dtype=jnp.float32):
  """
  Generates a spectrum of singular values with desired properties.

  Args:
    n_nonzero: The number of nonzero entries in the spectrum.
    sv_min: The smallest nonzero entry in the spectrum. Must be > 0. Default
            0.1.
    sv_max: The largest nonzero entry in the spectrum. Must be > 0 and >=
            `sv_min`. Default 1.
    distribution: `"linear"` or `"geometric"`; determines whether the singular
                  values are spaced evenly on a linear or logarithmic scale
                  respectively. Default `"linear"`.
    n_zeros: The number of zero entries in the spectrum. Default 0.
    dtype: Dtype of the spectrum. Default `jnp.float32`.
  Returns:
    The spectrum, a length- n_nonzero + n_zeros vector of dtype `dtype`
    containing the descendingly-sorted singular values.
  """
  if sv_max <= 0.:
    raise ValueError(f"Cannot have sv_max={sv_max} <= 0.")
  if sv_min <= 0.:
    raise ValueError(f"Cannot have sv_min={sv_min} <= 0.")
  if sv_max < sv_min:
    raise ValueError(f"Cannot have sv_max={sv_max} < sv_min={sv_min}.")
  if n_nonzero < 0:
    raise ValueError(f"Cannot have n_nonzero={n_nonzero} < 0")
  if n_zeros < 0:
    raise ValueError(f"Cannot have n_zeros={n_zeros} < 0")

  shift = 0.
  if distribution == "geometric" and sv_min < 1:
    shift = np.abs(sv_min) + 1

  sv_zero = jnp.zeros(n_zeros, dtype=dtype)
  spectrum_f = _get_spectrum_f(distribution, dtype)
  sv_nonzero = spectrum_f(sv_min + shift, sv_max + shift, num=n_nonzero) - shift
  return jnp.hstack([sv_nonzero[::-1], sv_zero])
