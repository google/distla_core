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

from distla_core.analysis.errors import errors_split_spectrum
from distla_core.utils import pops


if __name__ == "__main__":
  local_rows = np.array([128, 256, 512, 1024, 2048])
  ROWS = tuple(pops.NROWS * local_rows)
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (256,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.DEFAULT, lax.Precision.HIGH,
                lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False,) # Whether to run in serial or distributed mode.
  CANONICAL = (False,) # Whether to run in canonical or grand canonical
                            # mode.
  EIG_MIN = (-10.,) # Most negative eigenvalue in the spectrum.
  EIG_MAX = (10.,) # Most positive eigenvalue in the spectrum.
  K_FACTOR = (2, 8) # We seek the OUT_ROWS // K_FACTOR'th subspace.
  GAP_START = (0.,) # eigenvalue[OUT_ROWS // K_FACTOR]
  GAP_FACTOR = (2, 4, 6, 8, 10, 50, 100, 1000)
    # the next eigenvalue is (EIG_MAX - EIG_MIN) / GAP_FACTOR
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # eigenvalues in the input matrix.
  BATCH_SIZE = 2 # How many runs to assemblage
  REDUCTION_MODE = "min"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_split_spectrum" # output saved to OUTPUT_NAME.csv
  _ = errors_split_spectrum.error_split_spectrum(
    ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, canonical=CANONICAL, eig_min=EIG_MIN,
    eig_max=EIG_MAX, k_factor=K_FACTOR, gap_start=GAP_START,
    gap_factor=GAP_FACTOR, ev_distribution=EV_DISTRIBUTION,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)
