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
Performs errors of eigh. The errors are performed for each combination
of various parameters defined in `main()` in the case that this file is the
main program. Output is returned as a `DataFrame` and saved as a `.csv` file to
`output_dir_path/output_name`.csv.
In the default case of `output_dir_path=None`, it is taken to be the current
working directory.
"""

import functools
import itertools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.analysis import analysis
from distla_core.analysis import analyze_eigh
from distla_core.analysis import initializers
from distla_core.analysis.errors import error_utils


def _eigh_skip(out, in_args, params):
  result = in_args[0]["skip"]
  header = "SKIP"
  return result, header


def _unpack(out, in_args, params):
  if in_args[0]["skip"]:
    return None, None, None, None, None, None
  svd = params[8]
  if not svd:
    vals = out[0]
    vecs = out[1]
    vals_expected = in_args[0]["e_vals"]
    vecs_expected = in_args[0]["e_vecs"]
    return vecs, vals, vecs, vecs_expected, vals_expected, vecs_expected
  else:
    left = out[0]
    vals = out[1]
    right = out[2]
    left_expected = in_args[0]["left_vectors"]
    vals_expected = in_args[0]["e_vals"]
    right_expected = in_args[0]["e_vecs"]
    return left, vals, right, left_expected, vals_expected, right_expected


def _vec_angle(out, in_args, params, left_vec=True):
  data = _unpack(out, in_args, params)
  left, vals, right, left_expected, vals_expected, right_expected = data
  if left_vec:
    return error_utils.subspace_angle(left, left_expected, name_a="U_result",
                                      name_b="U_expected")
  else:
    return error_utils.subspace_angle(right, right_expected, name_a="V_result",
                                      name_b="V_expected")


def _vec_isometry(out, in_args, params, left_vec=True, isometry_left=True,
                  relative=False):
  data = _unpack(out, in_args, params)
  left, vals, right, left_expected, vals_expected, right_expected = data
  if left_vec:
    return error_utils.isometry_error(left, name="U", relative=relative,
                                      dagger_left=isometry_left)
  else:
    return error_utils.isometry_error(right, name="V", relative=relative,
                                      dagger_left=isometry_left)


def _ev_recon_error(out, in_args, params, relative=False):
  skip = in_args[0]["skip"]
  if skip:
    return -1, error_utils.comparison_header("input", "USV^H", relative)
  matrix = in_args[0]["matrix"]
  data = _unpack(out, in_args, params)
  left, vals, right, left_expected, vals_expected, right_expected = data
  reconstructed = initializers.combine_spectrum(vals, left, right_vectors=right)
  return error_utils.comparison_error(matrix, reconstructed, relative, "input",
                                      "USV^H")


def _ev_eqn_error(out, in_args, params, relative=False):
  # TODO: Support this
  data = _unpack(out, in_args, params)
  left, vals, right, left_expected, vals_expected, right_expected = data
  return -1, "||matrix * vecs - vals * vecs||_F"


def _evs_error(out, in_args, params, relative=False):
  header = "||vals - vals_expected||_F"
  if in_args[0]["skip"]:
    return -1, header
  data = _unpack(out, in_args, params)
  left, vals, right, left_expected, vals_expected, right_expected = data
  err = np.linalg.norm(np.sort(vals) - np.sort(vals_expected))
  return err, header


def error_eigh(
  rows, cols, dtypes, p_szs, precisions, seeds, serial, canonical, svd,
  eig_min, eig_max, k_factor, gap_start, gap_factor, ev_distribution,
  batch_size=5, reduction_mode="median", output_dir_path=None,
  output_name="error_eigh"):
  """ Runs errors of eigh. One error per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    rows : rows of the input matrix
    cols: cols of the input matrix. If `svd == False` and `cols != rows` or
      None the run is skipped. If `cols >= rows` the run is skipped.
    dtypes : dtype of the matrices
    p_szs : panel size of the summa multiplications
    precisions: asic matmul precision
    seeds : random seed to initialize input; system clock if none.
    serial: whether to run in serial or distributed mode.
    canonical: Whether to run in canonical or grand canonical mode.
    svd: Whether to compute the `SVD` or the symmetric eigendecomposition.
      With `SVD == True`, the parameters related to the eigenspectrum are
      used to construct the input matrix's positive-semidefinite factor.
      `eig_min, eig_max, gap_start` must therefore be non-negative in this case.
    eig_min: most negative eigenvalue in the spectrum.
    eig_max: most positive eigenvalue in the spectrum.
    k_factor: a gap of fixed size may optionally be added at the
      `rows // k_factor`'th eigenvalue.
    gap_start: Value of `spectrum[k_target - 1]`; must be non-negative if
      `SVD == True` (or else the run is skipped).
    gap_factor: the gap is of size `(eig_max - eig_min) / gap_factor`
    ev_distribution: `linear` or `geometric` distribution of
      eigenvalues in the input matrix.
    batch_size: how many runs to assemblage
    reduction_mode: how to assemblage results
    output_dir_path: directory of output; cwd if none
    output_name: output saved to output_name.csv
  Returns:
    The results as a DataFrame.
  """
  dtypes = tuple([jnp.dtype(d) for d in dtypes])
  params = list(itertools.product(
    rows, cols, dtypes, p_szs, precisions, seeds, serial, canonical, svd,
    eig_min, eig_max, k_factor, gap_start, gap_factor, ev_distribution))

  param_headers = [
    "N_rows", "N_cols", "dtype", "p_sz", "precision", "seed", "serial",
    "canonical", "svd", "eig min", "eig max", "occupancy", "gap start",
    "gap factor", "ev distribution"]
  init_f = functools.partial(analyze_eigh._initialize_eigh, return_answer=True)
  target_f = analyze_eigh._run_eigh
  error_functions = [_eigh_skip, ]
  for left in [False, True]:
    error_functions.append(functools.partial(_vec_angle, left_vec=left))
  for rel in [False, True]:
    error_functions.append(functools.partial(_ev_recon_error, relative=rel))

  for rel in [False, True]:
    error_functions.append(functools.partial(_ev_eqn_error, relative=rel))
    error_functions.append(functools.partial(_evs_error, relative=rel))
    for left in [False, True]:
      for isometry_left in [False, True]:
        error_functions.append(functools.partial(_vec_isometry, relative=rel,
                                           left_vec=left,
                                           isometry_left=isometry_left))

  return analysis.measure_error(init_f, target_f, params, param_headers,
                                reduction_mode=reduction_mode,
                                batch_size=batch_size,
                                error_functions=error_functions,
                                output_dir_path=output_dir_path,
                                output_name=output_name)


def main():
  ROWS = (512,)  # rows of the input matrix
  COLS = (None,) # cols of the input matrix; == rows if None.
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (64,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False, True) # Whether to run in serial or distributed mode.
  CANONICAL = (False, True) # Whether to run in canonical or grand canonical
                            # mode.
  SVD = (False, True) # Whether to compute the SVD.
  EIG_MIN = (-10., 1.) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (10.,) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (2, 4) # Location of gap.
  GAP_START = (2.,) # eigenvalue[COLS // K_FACTOR]
  GAP_FACTOR = (10,) # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 5 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "error_eigh" # output saved to OUTPUT_NAME.csv
  _ = error_eigh(
    ROWS, COLS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, canonical=CANONICAL, svd=SVD, eig_min=EIG_MIN,
    eig_max=EIG_MAX, k_factor=K_FACTOR, gap_start=GAP_START,
    gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
