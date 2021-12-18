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
""" The DistlaCore wrapper used in struc_pack_test_distla_core.f90.
That test should be called with DISTLA_PYTHON_PATH
pointing to this file.
"""

import sys
import os
from struc_pack_wrapper import StructPack


STRUC_PACK_LIB_ENV = 'DISTLA_STRUC_PACK_LIB_PATH'


class EnvValueError(Exception):
  pass


def main():
  print('- in python -')
  args = sys.argv
  obj_fn_path = args[1]
  ovlp_path = args[2]
  dm_path = args[3]
  ebs_path = args[4]

  lib_path = os.getenv(STRUC_PACK_LIB_ENV)

  if not lib_path:
    raise EnvValueError(f'The path to the STRUC_PACK lib must be specified as {STRUC_PACK_LIB_ENV}')

  struc_pack = StructPack(lib_path)
  obj_fn, n_elec = struc_pack.read_matrix(obj_fn_path)
  print('loaded obj_fn - size: ', obj_fn.shape)
  print('n_elec: ', n_elec)
  ovlp, n_elec = struc_pack.read_matrix(ovlp_path)
  print('loaded ovlp - size: ', ovlp.shape)

  # Process data here
  dm = 2 * obj_fn

  struc_pack.write_matrix(dm_path, dm)

  with open(ebs_path, 'w') as f:
    f.write(str(1.2345E6))

  print('- leaving python -')

if __name__ == "__main__":
  main()
  sys.exit(0)