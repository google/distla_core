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
"""A solver for STRUC_PACK that uses np.linalg.eigh on CPUs.

The point is that it uses the same reading/writing interface for STRUC_PACK as the
distla_core solver, and can thus be used for testing said interface.
"""
from ctypes import ArgumentError
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname).1s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import sys
import os
import numpy as np
from struc_pack_wrapper import StructPack

STRUC_PACK_LIB_ENV = 'DISTLA_STRUC_PACK_LIB_PATH'
SPIN_DEGEN = 2 # spin degeneracy (hard-coded for now)

class EnvValueError(Exception):
  pass

class IllegalArgumentError(ValueError):
  pass

def invsqrt(M):
  E, U = np.linalg.eigh(M)
  # M should be posdef anyway, but in case there are numerical errors near zero.
  E = np.abs(E)
  return (U / np.sqrt(E)) @ U.T.conj()


def purify(obj_fn, ovlp, num_states):
  num_states = int(np.round(num_states))
  ovlp_invsqrt = invsqrt(ovlp)
  # ovlp_invsqrt is Hermitian, but just for numerical errors sake we take the
  # conjubuilding_block.
  ovlp_invsqrt_dg = ovlp_invsqrt.T.conj()
  obj_fn_orth = ovlp_invsqrt_dg @ obj_fn @ ovlp_invsqrt
  E, U = np.linalg.eigh(obj_fn_orth)
  U = U[:, :num_states]
  density_matrix_orth = U @ U.T.conj()
  density_matrix = ovlp_invsqrt @ density_matrix_orth @ ovlp_invsqrt_dg
  ebs = np.sum(E[:num_states])

  return density_matrix, ebs


def get_dm(struc_pack, obj_fn_path, ovlp_path, dm_path, ebs_path):
  # Purify ObjectiveFn.

  obj_fn, n_elec = struc_pack.read_matrix(obj_fn_path)
  logging.info(f'Loaded a ObjectiveFn, shape = {obj_fn.shape}')
  logging.info(f'n_elec = {n_elec}')

  ovlp, _ = struc_pack.read_matrix(ovlp_path)
  logging.info(f'Loaded an overlap matrix, shape = {ovlp.shape}')

  logging.info("Purifying")
  dm, ebs = purify(obj_fn, ovlp, n_elec / SPIN_DEGEN)
  return dm, ebs
  

def get_edm(struc_pack, obj_fn_path, dm_path, edm_path):
  # Compute energy-weighted density matrix (edm) for Forces.

  obj_fn, n_elec = struc_pack.read_matrix(obj_fn_path)
  logging.info(f'Loaded a ObjectiveFn, shape = {obj_fn.shape}')
  logging.info(f'n_elec = {n_elec}')

  dm, _ = struc_pack.read_matrix(dm_path)
  logging.info(f'Loaded a density matrix, shape = {dm.shape}')  

  logging.info("Computing energy-weighted density matrix")
  edm =  dm @ obj_fn @ dm
  return edm
    

def main():
  logging.info("Entered launch_distla_core_numpy.py")
  args = sys.argv
  obj_fn_path = args[1]
  ovlp_path = args[2]
  dm_path = args[3]

  lib_path = os.getenv(STRUC_PACK_LIB_ENV)
  if not lib_path:
    raise EnvValueError(
        f'The path to the STRUC_PACK lib must be specified as {STRUC_PACK_LIB_ENV}')
  struc_pack = StructPack(lib_path)

  if args[4] == 'edm.tmp':
    edm_path = args[4]
    edm = get_edm(struc_pack, obj_fn_path, dm_path, edm_path)

    logging.info("Writing energy-weighted density matrix to disk")
    struc_pack.write_matrix(edm_path, (1/SPIN_DEGEN)*edm)
  elif args[4] == 'ebs.tmp':
    ebs_path = args[4]
    dm, ebs = get_dm(struc_pack, obj_fn_path, ovlp_path, dm_path, ebs_path)

    logging.info("Writing density matrix to disk")
    struc_pack.write_matrix(dm_path, SPIN_DEGEN * dm)

    with open(ebs_path, "w") as f:
      f.write(str(ebs))
  else:
    raise IllegalArgumentError(
        f'4th command line argument can only be ebs.tmp or edm.tmp')

  logging.info("Leaving launch_distla_core_numpy.py")


if __name__ == "__main__":
  main()
  sys.exit(0)
