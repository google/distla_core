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

from distla_core.analysis.errors import errors_subspace
from distla_core.utils import pops


if __name__ == "__main__":
  local_rows = np.array([128, 256, 512, 1024, 2048, 4096])
  ROWS = tuple(pops.NROWS * local_rows)
  COLS = (None, )
  DTYPES = (np.float32,)  # dtype of the matrices
  P_SZS = (256,)  # panel size of the SUMMA multiplications
  PRECISIONS = (lax.Precision.DEFAULT, lax.Precision.HIGH,
                lax.Precision.HIGHEST,) # ASIC matmul precision
  SEEDS = (None,)  # Random seed to initialize input; system clock if None.
  SERIAL = (False,) # Whether to run in serial or distributed mode.
  K_FACTOR = (2, 8) # Location of gap.
  MAXITER = (2,) # Maximum number of subspace iterations.
  POLAR_ITER = (26,) # Maximum num of polar iterations per subspace iteration.
  BATCH_SIZE = 1 # How many runs to assemblage
  REDUCTION_MODE = "min"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "errors_subspace" # output saved to OUTPUT_NAME.csv
  _ = errors_subspace.errors_subspace(
    ROWS, dtypes=DTYPES, p_szs=P_SZS, precisions=PRECISIONS, seeds=SEEDS,
    serial=SERIAL, k_factor=K_FACTOR, maxiter=MAXITER, polar_iter=POLAR_ITER,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)
