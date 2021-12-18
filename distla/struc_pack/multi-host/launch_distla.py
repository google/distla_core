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
"""Module for running the distla_core solver directly on the ASIC host."""
import functools
import hashlib
import logging
import os
import pathlib
import sys
import yaml

import jax
from jax.experimental.compilation_cache import compilation_cache
import numpy as np

from distla_core.utils import pops
from distla_core.io import sparse
import purify_density_matrix

STRUC_PACK_LIB_ENV = "DISTLA_STRUC_PACK_LIB_PATH"
HASH_BUFFER = 65536
SPIN_DEGEN = 2  # spin degeneracy (hard-coded for now)

jax.config.update("jax_enable_x64", True)


class EnvValueError(Exception):
  pass


class IllegalArgumentError(ValueError):
  pass


def setup_logging():
  # Importing jax seems to result to a call to logging.basicConfig, so we can't
  # use that here anymore and have to do this more manually.
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  # Notice that the header part of the log string includes jax_id.
  jax_id = jax.process_index()
  for h in logger.handlers:
    h.setFormatter(
        logging.Formatter(
            f"%(asctime)s %(levelname).1s{jax_id}: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))


def hash_file(path):
  """Computes the MD5 hash of a file at a given path."""
  md5 = hashlib.md5()
  with open(path, "rb") as f:
    data = f.read(HASH_BUFFER)
    while data:
      md5.update(data)
      data = f.read(HASH_BUFFER)
  return md5.hexdigest()


distributed_astype = pops.pmap(
    lambda x, y: x.astype(y), static_broadcasted_argnums=(1,))


@functools.partial(pops.pmap, static_broadcasted_argnums=(1,))
def _pmapped_multiply(x, a):
  return x * a


@functools.partial(pops.pmap, out_axes=None)
def _pmapped_psum(x):
  return jax.lax.psum(x, axis_name=pops.AXIS_NAME)


def get_distla_core_dtype(distla_core_conf, aims_json_path):
  """Figure out which dtype DistlaCore should use. Each host tries to read the
  relevant YAML file, but only one of them will have that file, so in the end we
  vote, such that if any host thinks we should use double we use double.
  """
  dtype = get_distla_core_dtype_from_yaml(distla_core_conf, aims_json_path)
  n_local_devices = jax.local_device_count()
  vote_accumulator = np.zeros((n_local_devices,), dtype=np.int32)
  if dtype == np.float64:
    vote_accumulator[0] = 1
  vote = _pmapped_psum(vote_accumulator)
  return np.float64 if vote > 0 else np.float32


def get_distla_core_dtype_from_yaml(distla_core_conf, aims_json_path):
  """This is the function that on a single host is sufficient to figure out
  which dtype DistlaCore should use. It reads the yaml file at aims_json_path and
  goes from there.
  """
  aims_log = read_yaml_file(aims_json_path)
  if not aims_log:
    logging.warning(f"{aims_json_path} empty or non-existent")
  dtype_density_threshold = float(
      distla_core_conf["dtype_relative_density_change_threshold"])
  dtype_energy_threshold = float(
      distla_core_conf["dtype_relative_energy_change_threshold"])
  dtype_max_iter = int(distla_core_conf["dtype_max_iter"])
  # Find the last scf_iteration entry, get the values from that.
  relative_change_charge_density = None
  relative_change_energy = None
  iteration = None
  found_data = False
  for entry in reversed(aims_log):
    if entry["record_type"] == "scf_iteration":
      relative_change_charge_density = abs(
          entry["change_charge_density"] / entry["n_electrons"])
      relative_change_energy = abs(
          entry["change_total_energy"] / entry["total_energy"])
      iteration = entry["iteration_this_cycle"]
      found_data = True
      break
  if not found_data:
    logging.warning(f"Could not find a convergence data in {aims_json_path}")
    # Only use double if max_iter is zero or less, which we interpret as "please
    # _always_ use double".
    use_double = dtype_max_iter <= 0
  else:
    logging.info(
        f"relative_change_charge_density = {relative_change_charge_density}")
    logging.info(f"relative_change_energy = {relative_change_energy}")
    logging.info(f"iteration = {iteration}")
    use_double = (relative_change_charge_density < dtype_density_threshold and
                  relative_change_energy < dtype_energy_threshold
                 ) or iteration > dtype_max_iter
  return np.float64 if use_double else np.float32


def read_yaml_file(path):
  """Returns the contents of a YAML file at `path` as a dict, or an empty dict
  if the file doesn't exist.
  """
  if path.exists():
    with open(path) as f:
      s = f.read()
    try:
      d = yaml.load(s, Loader=yaml.SafeLoader)
    except yaml.parser.ParserError:
      # FHI-AIMS has a bad habit of writing unfinished JSON files that are
      # missing the closing bracket.
      d = yaml.load(s + ",\n]", Loader=yaml.SafeLoader)
  else:
    d = {}
  return d


def check_recyclability(ovlp_hash, distla_core_conf, old_ovlp_info_path):
  """Checks whether the previously computed invsqrt(S) can be used again.

  The previously computed invsqrt(S) can be recycled if:
  1) S hasn't changed since last iteration. This is checked by hashing `S.csc`.
  2) The dtype DistlaCore is using hasn't changed.
  3) The overlap_threshold DistlaCore is using hasn't changed.

  Args:
    ovlp_hash: Hash of current S
    distla_core_conf: Current DistlaCore configuration, as a dict
    old_ovlp_info_path: Path to a YAML file that holds the hash of S and DistlaCore
      configuration used from previous iteration.

  Returns:
    recycle: A boolean.
  """
  old_ovlp_info = read_yaml_file(old_ovlp_info_path)
  old_ovlp_hash = old_ovlp_info.get("ovlp_hash", None)
  old_threshold = old_ovlp_info.get("overlap_threshold", None)
  recycle = (old_ovlp_hash is not None and ovlp_hash == old_ovlp_hash and
             old_threshold is not None and
             distla_core_conf["overlap_threshold"] == old_threshold)
  return recycle


def update_ovlp_info(ovlp_hash, distla_core_conf, old_ovlp_info_path):
  """Updates the `old_ovlp_info.yaml` file with the new hash and DistlaCore
  configuration.
  """
  ovlp_info = {}
  ovlp_info["ovlp_hash"] = ovlp_hash
  ovlp_info["overlap_threshold"] = distla_core_conf["overlap_threshold"]
  with open(old_ovlp_info_path, "w") as f:
    yaml.dump(ovlp_info, f)


def get_ovlp_invsqrt(
    struc_pack_ovlp_path,
    old_ovlp_info_path,
    old_ovlp_invsqrt_path,
    old_k_path,
    distla_core_conf,
    distla_core_dtype,
):
  """Returns the inverse square root of the STRUC_PACK overlap matrix.

  We first check if the overlap matrix, S, or the DistlaCore configuration, has
  changed from the previous iteration. If not, we deem that we can reuse
  invsqrt(S), and load it as well as its unpadded size from disk. If
  recycling can not be done, S itself is instead read from disk, invsqrt(S) is
  computed, and the result is written to disk for future reuse.

  This function also casts the overlap matrix into the Numpy dtype used by
  DistlaCore. The matrix stored on disk, however, is always in double precision.

  Args:
    struc_pack_ovlp_path: Path to which STRUC_PACK has written the overlap matrix.
    old_ovlp_info_path: Path to a YAML file that holds information about S
      from the previous iteration.
    old_ovlp_invsqrt_path: Path to where invsqrt(S) from previous iteration is
      written.
    old_k_path: Path to where the unpadded dimension of invsqrt(S) from previous
      iteration is written.
    distla_core_conf: DistlaCore configuration, as a dict.
    distla_core_dtype: dtype to cast the result to.

  Returns:
    ovlp_invsqrt: invsqrt(S)
    k: Unpadded dimension of invsqrt(S)
  """
  logging.info("Figuring out whether to recycle.")
  ovlp_hash = hash_file(struc_pack_ovlp_path)
  recycle = check_recyclability(
      ovlp_hash,
      distla_core_conf,
      old_ovlp_info_path,
  )
  if recycle and (not os.path.exists(old_ovlp_invsqrt_path) or
                  not os.path.exists(old_k_path)):
    msg = ("Was asked to recycle invsqrt(S), but it's not on disk, will "
           "recompute")
    logging.info(msg)
    recycle = False

  if recycle:
    logging.info("S the same as before, reading and distributing its invsqrt")
    ovlp_invsqrt = np.load(old_ovlp_invsqrt_path).astype(distla_core_dtype)
    ovlp_invsqrt = jax.pmap(lambda x: x.astype(distla_core_dtype))(ovlp_invsqrt)
    ovlp_invsqrt.block_until_ready()
    k = int(np.load(old_k_path))
  else:
    logging.info("Reading S")
    ovlp, unpadded_dim, _ = sparse.read_struc_pack_csc(struc_pack_ovlp_path)
    # ovlp here has dtype np.float64
    overlap_threshold = float(distla_core_conf["overlap_threshold"])
    ovlp_invsqrt, k = purify_density_matrix.overlap_matrix_invsqrt(
        ovlp,
        unpadded_dim,
        overlap_threshold=overlap_threshold,
    )
    ovlp_invsqrt.block_until_ready()
    logging.info("Writing invsqrt(S)")
    np.save(old_ovlp_invsqrt_path, ovlp_invsqrt)
    np.save(old_k_path, k)
    update_ovlp_info(ovlp_hash, distla_core_conf, old_ovlp_info_path)
    ovlp_invsqrt = distributed_astype(ovlp_invsqrt, distla_core_dtype)

  return ovlp_invsqrt, k


def get_edm(obj_fn_path, dm_path):
  # Compute energy-weighted density matrix (edm) for Forces.
  base_path = pathlib.Path("./")
  distla_core_config_path = base_path / "./distla_core_config.yaml"
  aims_json_path = base_path / "./aims.json"

  distla_core_conf = read_yaml_file(distla_core_config_path)
  distla_core_dtype = get_distla_core_dtype(distla_core_conf, aims_json_path)

  logging.info("Reading H")
  obj_fn, unpadded_dim, n_elec = sparse.read_struc_pack_csc(obj_fn_path)
  obj_fn = distributed_astype(obj_fn, distla_core_dtype)
  logging.info(f"unpadded_dim = {unpadded_dim}")
  logging.info(f"n_elec = {n_elec}")
  logging.info(f"distla_core_dtype = {distla_core_dtype}")

  logging.info("Reading DM")
  dm, unpadded_dim, n_elec = sparse.read_struc_pack_csc(dm_path)
  dm = distributed_astype(dm, distla_core_dtype)
  logging.info(f"unpadded_dim = {unpadded_dim}")
  logging.info(f"n_elec = {n_elec}")
  logging.info(f"distla_core_dtype = {distla_core_dtype}")
  logging.info(f"local shape obj_fn = {obj_fn.shape}")

  edm = purify_density_matrix.compute_energy_weighted_density_matrix(
      obj_fn,
      dm,
  )
  edm = _pmapped_multiply(edm, 1 / SPIN_DEGEN)
  return edm, unpadded_dim


def get_dm(obj_fn_path, ovlp_path):
  # Purify ObjectiveFn.
  base_path = pathlib.Path("./")
  distla_core_config_path = base_path / "./distla_core_config.yaml"
  aims_json_path = base_path / "./aims.json"
  data_folder = pathlib.Path("/struc_pack/data")
  old_ovlp_info_path = data_folder / pathlib.Path("./old_ovlp_info.yaml")
  old_ovlp_invsqrt_path = data_folder / pathlib.Path("old_ovlp_insqrt.npy")
  old_k_path = data_folder / pathlib.Path("old_k.npy")

  data_folder.mkdir(exist_ok=True)
  distla_core_conf = read_yaml_file(distla_core_config_path)
  distla_core_dtype = get_distla_core_dtype(distla_core_conf, aims_json_path)

  ovlp_invsqrt, k = get_ovlp_invsqrt(
      ovlp_path,
      old_ovlp_info_path,
      old_ovlp_invsqrt_path,
      old_k_path,
      distla_core_conf,
      distla_core_dtype,
  )

  logging.info("Reading H")
  obj_fn, unpadded_dim, n_elec = sparse.read_struc_pack_csc(obj_fn_path)
  obj_fn = distributed_astype(obj_fn, distla_core_dtype)
  logging.info(f"unpadded_dim = {unpadded_dim}")
  logging.info(f"n_elec = {n_elec}")
  logging.info(f"distla_core_dtype = {distla_core_dtype}")
  logging.info(f"local shape obj_fn = {obj_fn.shape}")
  logging.info(f"local shape ovlp = {ovlp_invsqrt.shape}")
  logging.info(f"ovlp k = {k}")

  logging.info("Calling purify_density_matrix")
  n_occupied = n_elec / SPIN_DEGEN
  dm, ebs = purify_density_matrix.purify_density_matrix(
      obj_fn,
      ovlp_invsqrt,
      k,
      n_occupied,
  )
  dm = _pmapped_multiply(dm, SPIN_DEGEN)

  return dm, unpadded_dim, ebs


def main():
  setup_logging()
  logging.info("Entered launch_distla_core.py")
  logging.info("Setting up in launch_distla_core.py")
  if jax.device_count() < 8:
    msg = "Jax has less than 8 devices. Are ASICs not being recognised?"
    raise RuntimeError(msg)

  args = sys.argv
  struc_pack_obj_fn_path = args[1]
  struc_pack_ovlp_path = args[2]
  struc_pack_dm_path = args[3]

  cache_path = pathlib.Path.home() / ".jax_compilation_cache"
  compilation_cache.initialize_cache(cache_path)

  if args[4][-7:] == 'edm.tmp':
    edm_path = args[4]
    edm, unpadded_dim = get_edm(struc_pack_obj_fn_path, struc_pack_dm_path)

    logging.info("Writing energy-weighted density matrix to disk")
    sparse.write_struc_pack_csc(edm_path, edm, unpadded_dim=unpadded_dim)
  elif args[4][-7:] == 'ebs.tmp':
    ebs_path = args[4]
    dm, unpadded_dim, ebs = get_dm(struc_pack_obj_fn_path, struc_pack_ovlp_path)

    logging.info("Writing density matrix to disk")
    sparse.write_struc_pack_csc(
        struc_pack_dm_path,
        dm,
        unpadded_dim=unpadded_dim,
    )
    if jax.process_index() == 0:
      # Only one host needs to do this.
      with open(ebs_path, "w") as f:
        f.write(str(ebs))
  else:
    raise IllegalArgumentError(
        f'4th command line argument can only be ebs.tmp or edm.tmp')

  logging.info("Leaving launch_distla_core.py")


if __name__ == "__main__":
  main()
  sys.exit(0)
