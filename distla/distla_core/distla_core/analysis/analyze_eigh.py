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
import gc
import jax.numpy as jnp

from distla_core.analysis import initializers
from distla_core.analysis import pmaps
from distla_core.linalg.eigh import eigh
from distla_core.linalg.eigh.serial import eigh as eigh_serial
from distla_core.linalg.eigh.serial import eigh_canonical as eigh_serial_canonical
from distla_core.linalg.eigh.serial import purify as purify_serial

from distla_core.utils import pops


def _initialize_eigh(
  rows, cols, dtype, p_sz, precision, seed, serial, canonical, svd, eig_min,
  eig_max, k_factor, gap_start, gap_factor, ev_distribution,
  return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `eigh`.

  Args:
    rows: rows of the input matrix.
    cols: cols of the input matrix. If `svd == False` and `cols != rows` or
      None the run is skipped. If `cols >= rows` the run is skipped.
    dtype: dtype of the matrices.
    p_sz: panel size of the SUMMA multiplications; ignored if `serial=True`.
    precision: ASIC matmul precision.
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    canonical: Whether to run in canonical or grand canonical mode.
    svd: Whether to compute the `SVD` or the symmetric eigendecomposition.
      With `SVD == True`, the parameters related to the eigenspectrum are
      used to construct the input matrix's positive-semidefinite factor.
      `eig_min, eig_max, gap_start` must therefore be non-negative in this case.
    eig_min: Smallest (most negative) eigenvalue of the input, or of the
      input's positive-semidefinite factor with `SVD == True`. If `eig_min` is
      negative in that case, the run is skipped.
    eig_max: Largest (most positive) eigenvalue of the input, or of the input's,
      positive-semidefinite factor with `SVD == True`, in which case this must
      be non-negative (or else the run is skipped).
    k_factor: a gap of fixed size may optionally be added at the
      `rows // k_factor`'th eigenvalue.
    gap_start: Value of `spectrum[k_target - 2]`; must be non-negative if
      `SVD == True` (or else the run is skipped).
    gap_factor: the gap is of size `(eig_max - eig_min) / gap_factor`
    ev_distribution: `linear` or `geometric` distribution of eigenvalues in
      the input matrix.
    return_answer: If True, a correct result is returned along with the input
      matrix.
  Returns:
    initialized: Output of the initialization, a dict.
      initialized["matrix"]: Input matrix to `eigh`.
      initialized["skip"]: Whether to skip this run.
      initialized["e_vecs"]: If not `return_answer`, `None`; otherwise the
       computed eigenvectors (the right singular vectors if SVD=True).
      initialized["e_vals"]: If not `return_answer`, `None`; otherwise the
       computed eigenvalues (the singular values if SVD=True).
      initialized["left_vecs"]: Unless `return_answer` and `svd`, `None`;
       otherwise the computed left singular vectors.
    precision, canonical, svd, p_sz: Same as input
  """
  skip = False
  if svd and (eig_min < 0 or eig_max < 0 or gap_start < 0):
    skip = True
  if cols is None:
    cols = rows
  if cols > rows or (not svd and cols != rows):
    skip = True
  if cols < p_sz:
    p_sz = cols // 2
  k_target = cols // k_factor
  gap_size = (eig_max - eig_min) / gap_factor
  e_vals = initializers.gapped_real_eigenspectrum(
    cols, eig_min, eig_max, gap_position=k_target, gap_start=gap_start,
    gap_size=gap_size, distribution=ev_distribution, dtype=dtype)
  init_tup = initializers.random_from_eigenspectrum(
    e_vals, dtype=dtype, seed=seed, serial=serial, p_sz=p_sz,
    precision=precision, return_factors=return_answer)

  e_vecs = None
  left_vectors = None
  if return_answer:
    matrix, e_vecs = init_tup
  else:
    matrix = init_tup
    e_vals = None

  if svd:
    polar_factor = initializers.random_isometry(
      (rows, cols), dtype, seed=seed, serial=serial, p_sz=p_sz,
      precision=precision)
    matrix = pmaps.matmul(polar_factor, matrix, p_sz=p_sz, precision=precision)
    if return_answer:
      left_vectors = pmaps.matmul(polar_factor, e_vecs, p_sz=p_sz,
                                  precision=precision)

  initialized = {"matrix": matrix, "e_vecs": e_vecs, "e_vals": e_vals,
                 "skip": skip, "left_vectors": left_vectors}
  return initialized, precision, canonical, svd, p_sz


def _run_eigh(initialized, precision, canonical, svd, p_sz):
  """ Helper to call eigh.
  """

  matrix = initialized["matrix"]
  if initialized["skip"]:
    return None, None

  if matrix.ndim == 2:
    kwargs = {"precision": precision}
    if canonical:
      module = eigh_serial_canonical
    else:
      module = eigh_serial
  elif matrix.ndim == 3:
    kwargs = {"precision": precision, "p_sz": p_sz, "canonical": canonical}
    module = eigh
  else:
    raise ValueError(f"matrix had invalid ndim {matrix.ndim}.")

  if svd:
    out = module.svd(matrix, **kwargs)
  else:
    out = module.eigh(matrix, **kwargs)

  try:
    out0 = out[0].block_until_ready()
  except AttributeError:
    out0 = jnp.array(out[0]).block_until_ready()
  return out0, *(out[1:])


def _initialize_split_spectrum(
  rows, dtype, p_sz, precision, seed, serial, canonical, eig_min,
  eig_max, k_factor, gap_start, gap_factor, ev_distribution,
  return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `split_spectrum`.

  Args:
    rows: rows of the input matrix.
    dtype: dtype of the matrices.
    p_sz: panel size of the SUMMA multiplications; ignored if `serial=True`.
    precision: ASIC matmul precision.
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    canonical: Whether to run in canonical or grand canonical mode.
    eig_min: Smallest (most negative) eigenvalue of the input.
    eig_max: Largest (most positive) eigenvalue of the input.
    k_factor: The split will be at `k_rank = rows // k_factor`'th eigenvalue.
    gap_start: Value of `spectrum[k_rank - 2]`.
    gap_factor: the gap is of size `(eig_max - eig_min) / gap_factor`
    ev_distribution: `linear` or `geometric` distribution of eigenvalues in
      the input matrix.
    return_answer: If True, a correct result is returned along with the input
      matrix.
  Returns:
    initialized: Output of the initialization, a dict.
      initialized["matrix"]: Input matrix to `eigh`.
      initialized["skip"]: Whether to skip this run.
      initialized["isometry_l"]: If not `return_answer`, `None`; otherwise the
        normalized eigenvectors associated with `eigenvalues[:k_rank]`.
      initialized["matrix_l"]: If not `return_answer`, `None`; otherwise the
        projection of `matrix` into the subspace spanned by `isometry_l`.
      initialized["isometry_r"]: If not `return_answer`, `None`; otherwise the
        normalized eigenvectors associated with `eigenvalues[k_rank:]`.
      initialized["matrix_r"]: If not `return_answer`, `None`; otherwise the
        projection of `matrix` into the subspace spanned by `isometry_r`.
      initialized["split_point"]: `rows // k_factor`
      initialized["mu"]: Midpoint between `eigenvalue[split_point - 2]` and
        `eigenvalue[split_point - 1]`.
    canonical, rows, p_sz: Same as input
  """
  skip = False
  k_target = rows // k_factor
  gap_size = (eig_max - eig_min) / gap_factor
  e_vals = initializers.gapped_real_eigenspectrum(
    rows, eig_min, eig_max, gap_position=k_target, gap_start=gap_start,
    gap_size=gap_size, distribution=ev_distribution, dtype=dtype)
  init_tup = initializers.random_from_eigenspectrum(
    e_vals, dtype=dtype, seed=seed, serial=serial, p_sz=p_sz,
    return_factors=return_answer)

  if return_answer:
    if serial:
      shape_l = (rows, k_target)
      shape_r = (rows, rows - k_target)
    else:
      shape_l = (pops.NDPROCS, rows // pops.NROWS, k_target // pops.NCOLS)
      shape_r = (pops.NDPROCS, rows // pops.NROWS,
                 (rows - k_target) // pops.NCOLS)

    matrix, e_vecs = init_tup
    split_l = pmaps.eye(shape_l, dtype=matrix.dtype)
    isometry_l = pmaps.matmul(e_vecs, split_l, p_sz=p_sz)
    matrix_l = pmaps.similarity_transform(matrix, isometry_l, p_sz=p_sz)
    del split_l
    gc.collect()
    split_r = pmaps.eye(shape_r, dtype=matrix.dtype, k=-k_target)
    isometry_r = pmaps.matmul(e_vecs, split_r, p_sz=p_sz)
    matrix_r = pmaps.similarity_transform(matrix, isometry_r, p_sz=p_sz)
    del split_r
    gc.collect()

  else:
    isometry_l = None
    matrix_l = None
    isometry_r = None
    matrix_r = None
    matrix = init_tup
  mu = gap_start + 0.5 * gap_size

  initialized = {
    "matrix": matrix, "isometry_l": isometry_l, "matrix_l": matrix_l,
    "isometry_r": isometry_r, "matrix_r": matrix_r, "skip": skip,
    "k_target": k_target, "mu": mu}
  return initialized, rows, precision, canonical, p_sz


def _run_split_spectrum(initialized, rows, precision, canonical, p_sz):
  """ Helper to call split spectrum.
  """

  if initialized["skip"]:
    return None
  matrix = initialized["matrix"]

  if matrix.ndim == 2:
    if canonical:
      out = _split_serial_canonical(matrix, initialized["k_target"], precision)
    else:
      out = _split_serial_grand_canonical(matrix, initialized["mu"], precision)
  elif matrix.ndim == 3:
    if canonical:
      out = canonical_eigh.split_spectrum(matrix, rows, initialized["k_target"],
                                          None, precision=precision, p_sz=p_sz)
    else:
      out = eigh.split_spectrum(matrix, rows, initialized["mu"], None,
                                precision=precision, p_sz=p_sz)
    out = (out[0], out[1], out[3], out[4])
  else:
    raise ValueError(f"matrix had invalid ndim {matrix.ndim}.")

  return out[0].block_until_ready(), *(out[1:])


def _split_serial_canonical(matrix, k_target, precision):
  P, _, _ = purify_serial.canonically_purify(matrix, k_target,
                                             precision=precision)
  return eigh_serial_canonical.split_spectrum(P, matrix, None, k_target,
                                              precision)


def _split_serial_grand_canonical(matrix, mu, precision):
  def median_ev_func(matrix):
    return mu

  return eigh_serial._initial_step(matrix, median_ev_func, precision=precision)


def _initialize_subspace(
  rows, dtype, p_sz, precision, seed, serial, k_factor, maxiter, polar_iter,
  return_answer=False):
  """ An `init_f` functional argument to `_analysis_decorator` to initialize
  `subspace`.

  Args:
    rows: rows of the input matrix.
    dtype: dtype of the matrices.
    p_sz: panel size of the SUMMA multiplications; ignored if `serial=True`.
    precision: ASIC matmul precision.
    seed: Random seed to initialize input; system clock if None.
    serial: Whether to run in serial or distributed mode.
    k_factor: The projector projects into a subspace of rank
      `k_rank = rows // k_factor`.
    maxiter: Maximum (actual, in some implementations) number of iterations
      of subspace iteration.
    polar_iter: Maximum number of iterations of polar iteration.
    return_answer: If True, a correct result is returned along with the input
      matrix.
  Returns:
    initialized: Output of the initialization, a dict.
      initialized["matrix"]: The projection matrix of rank `k_rank`.
      initialized["skip"]: Whether to skip this run.
      initialized["isometry"]: If not `return_answer`, `None`; otherwise an
        isometry such that `isometry isometry^H = projector` (i.e. one spanning
        the same subspace as the output).
    precision, p_sz: Same as input
  """
  skip = False
  initialized = {}
  k_target = rows // k_factor
  isometry = initializers.random_isometry((rows, k_target), dtype, seed=seed,
                                          serial=serial, p_sz=p_sz,
                                          precision=precision)
  projector = pmaps.matmul(isometry, isometry, transpose_b=True, conj_b=True,
                           p_sz=p_sz, precision=precision)
  initialized["matrix"] = projector

  initialized["isometry"] = None
  initialized["k_target"] = k_target
  initialized["k_tup"] = (k_target, k_target // pops.NCOLS)
  if return_answer:
    initialized["isometry"] = isometry
  initialized["skip"] = skip
  return initialized, p_sz, maxiter, polar_iter, precision


def _run_subspace(initialized, p_sz, maxiter, polar_iter, precision):
  """ Helper to call subspace.
  """

  if initialized["skip"]:
    return None
  matrix = initialized["matrix"]

  if matrix.ndim == 2:
    out = purify_serial.subspace(matrix, initialized["k_target"], precision,
                                 "reduced")[0]
  elif matrix.ndim == 3:
    out = pmaps.subspace(matrix, initialized["k_tup"], p_sz, precision,
                         maxiter, polar_iter)
  else:
    raise ValueError(f"matrix had invalid ndim {matrix.ndim}.")

  return out.block_until_ready()
