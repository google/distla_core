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
Performs errors of split_spectrum. The errors are performed for each combination
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
from distla_core.analysis.errors import error_utils


def _split_spectrum_skip(out, in_args, params):
  result = in_args[0]["skip"]
  header = "SKIP"
  return result, header


def _isometry_comparison(out, in_args, params, left=True):
  if left:
    result = out[1]
    expected = in_args[0]["isometry_l"]
    append = "_l"
  else:
    result = out[3]
    expected = in_args[0]["isometry_r"]
    append = "_r"
  name_a = "isometry_result" + append
  name_b = "isometry_expected" + append
  return error_utils.subspace_angle(result, expected, name_a=name_a,
                                    name_b=name_b)


def _isometry_isometric(out, in_args, params, left=True, relative=False):
  if left:
    result = out[1]
    append = "_l"
  else:
    result = out[3]
    append = "_r"
  name = "isometry" + append
  return error_utils.isometry_error(result, relative=relative, name=name)


def _matrix_comparison(out, in_args, params, left=True, relative=False):
  if left:
    result = out[0]
    expected = in_args[0]["matrix_l"]
    append = "_l"
  else:
    result = out[2]
    expected = in_args[0]["matrix_r"]
    append = "_r"
  name_a = "projected_result" + append
  name_b = "projected_expected" + append
  return error_utils.subspace_angle(result, expected, name_a=name_a,
                                    name_b=name_b)


def error_split_spectrum(
  rows, dtypes, p_szs, precisions, seeds, serial, canonical,  eig_min, eig_max,
  k_factor, gap_start, gap_factor, ev_distribution, batch_size=5,
  reduction_mode="median", output_dir_path=None,
  output_name="error_split_spectrum"):
  """ Runs errors of split_spectrum. One error per combination of arguments
  is run. Output is saved to `output_dir_path/output_name`.csv, and also
  returned as a `DataFrame`. `output_dir_path` is the current working directory
  if unspecified.

  Args:
    rows : rows of the input matrix
    dtypes : dtype of the matrices
    p_szs : panel size of the summa multiplications
    precisions: asic matmul precision
    seeds : random seed to initialize input; system clock if none.
    serial: whether to run in serial or distributed mode.
    canonical: Whether to run in canonical or grand canonical mode.
    eig_min: most negative eigenvalue in the spectrum.
    eig_max: most positive eigenvalue in the spectrum.
    k_factor: a gap of fixed size may optionally be added at the
      `rows // k_factor`'th eigenvalue.
    gap_start: Value of `spectrum[k_target - 1]`.
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
    rows, dtypes, p_szs, precisions, seeds, serial, canonical, eig_min, eig_max,
    k_factor, gap_start, gap_factor, ev_distribution))

  param_headers = [
    "N_rows", "dtype", "p_sz", "precision", "seed", "serial", "canonical",
    "eig min", "eig max", "occupancy", "gap start", "gap factor",
    "ev distribution"]
  init_f = functools.partial(analyze_eigh._initialize_split_spectrum,
                             return_answer=True)
  target_f = analyze_eigh._run_split_spectrum

  fs = []
  for left in [True, False]:
    fs.append(functools.partial(_isometry_comparison, left=left))

    for relative in [True, False]:
      fs.append(functools.partial(_isometry_isometric, left=left,
                                  relative=relative))
      fs.append(functools.partial(_matrix_comparison, left=left,
                                  relative=relative))

  return analysis.measure_error(init_f, target_f, params, param_headers,
                                reduction_mode=reduction_mode,
                                batch_size=batch_size,
                                error_functions=fs,
                                output_dir_path=output_dir_path,
                                output_name=output_name)


def main():
  ROWS = (256,)  # rows of the input matrix
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (128,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False,) # Whether to run in serial or distributed mode.
  CANONICAL = (False,) # Whether to run in canonical or grand canonical
                            # mode.
  EIG_MIN = (1.,) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (10.,) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (2,) # Location of gap.
  GAP_START = (2.,) # eigenvalue[COLS // K_FACTOR]
  GAP_FACTOR = (10,) # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "median"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "error_split_spectrum" # output saved to OUTPUT_NAME.csv
  _ = error_split_spectrum(
    ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, canonical=CANONICAL, eig_min=EIG_MIN,
    eig_max=EIG_MAX, k_factor=K_FACTOR, gap_start=GAP_START,
    gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)


if __name__ == "__main__":
  main()
