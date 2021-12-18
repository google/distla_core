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
import numpy as np

from distla_core.analysis.errors import errors_purify
from distla_core.utils import pops


if __name__ == "__main__":
  local_rows = np.array([256, 512, 1024, 2048])
  ROWS = tuple(pops.NROWS * local_rows)
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (8192,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.DEFAULT, lax.Precision.HIGH,
                lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (1, )  # Random seed to initialize input; system clock if None.
  SERIAL = (False,) # Whether to run in serial or distributed mode.
  TOL = (None,) # Convergence threshold.
  MAXITER = (200,) # When to terminate if convergence stagnates.
  INCLUDE_OVERLAPS = (False,) # Whether to also perform basis transform.
  METHOD = ("hole-particle",) # String specifying purification method.
  EIG_MIN = (1.,) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (50., 100.) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (2, 6, 10) # We seek the OUT_ROWS // K_FACTOR'th subspace.
  GAP_START = (50.,) # eigenvalue[OUT_ROWS // K_FACTOR]
  GAP_FACTOR = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "min"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_purify" # output saved to OUTPUT_NAME.csv
  _ = errors_purify.errors_purify(
    ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, tol=TOL, maxiter=MAXITER, include_overlaps=INCLUDE_OVERLAPS,
    method=METHOD, eig_min=EIG_MIN, eig_max=EIG_MAX, k_factor=K_FACTOR,
    gap_start=GAP_START, gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)
