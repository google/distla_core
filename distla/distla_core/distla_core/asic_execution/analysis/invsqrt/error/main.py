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
# =============================================================================from jax import lax
import numpy as np

from distla_core.analysis.errors import errors_invsqrt
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
  EPS = (None, ) # Convergence threshold.
  MAXITER = (50,) # When to terminate if convergence stagnates.
  S_MIN_EST = (None, -1) # Estimated lowest singular value;
                         # None means machine epsilon; -1 means the true value.
  S_THRESH = (0., 0.1) # When to switch to Newton-Schulz from `rogue` iteration.
  S_MIN = (1E-5, 1E-4, 1E-3, 1E-2, 0.1,) # Smallest nonzero singular value.
  S_MAX = (1.0,) # Largest singular value of the input matrix.
  EV_DISTRIBUTION = ("linear",) # `linear` or `geometric` distribution of
                                # singular values in the input matrix.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "min"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_invsqrt" # output saved to OUTPUT_NAME.csv
  _ = errors_invsqrt.errors_invsqrt(
    ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, eps=EPS, maxiter=MAXITER, s_min_est=S_MIN_EST,
    s_thresh=S_THRESH, s_min=S_MIN, s_max=S_MAX,
    ev_distribution=EV_DISTRIBUTION, batch_size=BATCH_SIZE,
    reduction_mode=REDUCTION_MODE, output_dir_path=OUTPUT_DIR_PATH,
    output_name=OUTPUT_NAME)
