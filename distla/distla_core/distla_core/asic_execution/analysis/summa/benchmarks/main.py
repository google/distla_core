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

from distla_core.analysis.benchmarks import benchmark_summa
from distla_core.utils import pops


if __name__ == "__main__":
  local_rows = np.array([128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192,
                         16384])
  OUT_ROWS = tuple(pops.NROWS * local_rows)
  OUT_COLS = (None,)  # columns of the output matrix (=out_rows if None)
  SHARED_DIM = (None,)  # shared dim of the multiply (=out_rows if None)
  DTYPES = (jnp.float32,)  # dtype of the matrices
  P_SZS = (128, 256, -1)  # panel size of the multiplication
  TRANSPOSE_AS = (False, True) # whether to do C = A.T @ B
  TRANSPOSE_BS = (False,) # whether to do C = A @ B.T
  PRECISIONS = (lax.Precision.DEFAULT, lax.Precision.HIGH,
                lax.Precision.HIGHEST,) # ASIC matmul precision
  SEED_AS = (None,)  # Random seed to initialize A; system clock if None.
  SEED_BS = (None,)  # Same to initialize B; system clock + 1 if None.
  BATCH_SIZE = 3
  REDUCTION_MODE = "min"  # how to assemblage results
  OUTPUT_DIR_PATH = None  # directory of output; CWD if None
  OUTPUT_NAME = "benchmark_summa" # output saved to OUTPUT_NAME.csv
  _ = benchmark_summa.benchmark_summa(
    OUT_ROWS, out_cols=OUT_COLS, shared_dim=SHARED_DIM, dtypes=DTYPES,
    p_szs=P_SZS, transpose_as=TRANSPOSE_AS, transpose_bs=TRANSPOSE_BS,
    precisions=PRECISIONS, seed_as=SEED_AS, seed_bs=SEED_BS,
    batch_size=BATCH_SIZE, reduction_mode=REDUCTION_MODE,
    output_dir_path=OUTPUT_DIR_PATH, output_name=OUTPUT_NAME)
