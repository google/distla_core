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
import logging

from jax import lax
import jax.numpy as jnp
import numpy as np
import time

from distla_core.analysis import initializers
from distla_core.linalg.eigh import eigh
from distla_core.utils import misc
from distla_core.utils import pops


def _initialize_eigh(N, ev_min, gap, dtype, seed):
  evals = np.zeros(N, dtype=dtype).real
  evals[0] = ev_min
  evals[1:] = gap
  evals = jnp.cumsum(evals)
  matrix, evecs = initializers.random_from_eigenspectrum(
    evals, dtype=dtype, seed=seed, serial=False, p_sz=20000,
    precision=lax.Precision.HIGHEST, return_factors=True)
  matrix = matrix.astype(dtype)
  evecs = evecs.astype(dtype)
  evals = evals.astype(dtype)
  return matrix, evals, evecs


def _compute_errs(matrix, evals, evecs, result_evals, result_evecs):
  N = evals.size
  matrix = pops.undistribute(matrix)
  evecs = pops.undistribute(evecs)
  result_evals = result_evals[:N]
  result_evecs = pops.undistribute(result_evecs)[:N, :N]
  sort_indices = np.argsort(result_evals)
  result_evals = result_evals[sort_indices]
  result_evecs = result_evecs[:, sort_indices]
  evals_norm = np.linalg.norm(evals)

  evals_errs = np.abs(evals - result_evals)
  max_evals_err = np.amax(evals_errs)
  evals_err = np.linalg.norm(evals_errs)
  evals_err_rel = evals_err / evals_norm

  ev_eqn_left = np.dot(matrix, result_evecs)
  ev_eqn_right = result_evals * result_evecs
  ev_eqn_errs = np.abs(ev_eqn_left - ev_eqn_right)
  max_ev_eqn_err = np.amax(ev_eqn_errs)
  ev_eqn_err = np.linalg.norm(ev_eqn_errs)
  ev_eqn_err_rel = ev_eqn_err / evals_norm

  if N <= 2048:
    subspace_prod = np.dot(evecs.conj().T, result_evecs)
    subspace_prod, _ = np.linalg.qr(subspace_prod)
    svs = np.linalg.svd(subspace_prod, compute_uv=False)
    subspace_err = np.arccos(svs[-1])
  else:
    subspace_err = -1.
  return (evals_err, max_evals_err, evals_err_rel, ev_eqn_err, max_ev_eqn_err,
          ev_eqn_err_rel, subspace_err)


def _run_eigh(N, ev_min, gap, canonical, dtype=np.float32, seed=None,
              p_sz=1024, minimum_rank=128,
              precision=lax.Precision.HIGHEST):
  """ Helper to call eigh.
  """
  if seed is None:
    seed = int(time.time())
  matrix, evals, evecs = _initialize_eigh(N, ev_min, gap, dtype, seed)
  result_evals, result_evecs = eigh.eigh(
    matrix, p_sz=p_sz, minimum_rank=minimum_rank,
    precision=precision, canonical=canonical)
  if N <= 4096:
    errs = _compute_errs(matrix, evals, evecs, result_evals, result_evecs)
  else:
    errs = None
  return errs


def main():
  misc.initialize_distla_core_timing_log()
  logger = logging.getLogger("distla_core_timing")
  logger.debug("Running with %s cores", pops.NPROCS)
  local_rows = np.array([128, 256, 512, 1024])
  global_ns = pops.NROWS * local_rows
  gaps = [1E-1, 1E-5]
  canonicals = [False, ]
  ev_min_factors = [1., ]
  seed = 1
  for local_row, global_n in zip(local_rows, global_ns):
    for gap in gaps:
      for ev_min_factor in ev_min_factors:
        ev_min = -local_row * ev_min_factor
        for canonical in canonicals:
          for run_idx in range(2):
            logger.debug("****************************************************")
            logger.debug("****************************************************")
            logger.debug("****************************************************")
            logger.debug("local_rows = %s", local_row)
            logger.debug("global_n = %s", global_n)
            logger.debug("GAP = %s", gap)
            logger.debug("EV_MIN = %s", ev_min)
            logger.debug("SEED = %s", seed)
            logger.debug("CANONICAL = %s", canonical)
            logger.debug("RUN_IDX = %s", run_idx)
            try:
              errs = _run_eigh(
                global_n, ev_min, gap, canonical, seed=seed,
                minimum_rank=512)
              if errs is not None:
                (evals_err, max_evals_err, evals_err_rel, ev_eqn_err,
                max_ev_eqn_err, ev_eqn_err_rel, subspace_err) = errs
                logger.debug("EVALS_ERR = %s", evals_err)
                logger.debug("MAX_EVALS_ERR = %s", max_evals_err)
                logger.debug("EVALS_ERR_REL = %s", evals_err_rel)
                logger.debug("EV_EQN_ERR = %s", ev_eqn_err)
                logger.debug("MAX_EV_EQN_ERR = %s", max_ev_eqn_err)
                logger.debug("EV_EQN_ERR_REL = %s", ev_eqn_err_rel)
                logger.debug("SUBSPACE_ERR = %s", subspace_err)
                logger.debug("Done!")
            except RuntimeError:
              logger.debug("Runtime error encountered; skipping")


if __name__ == "__main__":
  main()
