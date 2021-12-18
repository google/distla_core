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
""" make_execution_directories.py
Prepares ASIC execution directories.

One directory for each given slice size is created. The contents of the target
and template directories respectively along with an appropriate asic.yaml
are copied into these targets. The arguments to asic.yaml can be configured
as command line arguments to this script.

TODO: Add exclude list (e.g. to avoid copying readme)
"""

import argparse
import os
import shutil


parser = argparse.ArgumentParser(
  description="Prepares ASIC execution directories.")
parser.add_argument(
  '-s', '--slice_sizes', nargs='+', default=(8, 32, 128, 512, 2048),
  help="Desired ASIC slice sizes (default: 8 32 128 512 2048).")
parser.add_argument(
  '-v', '--asic_version', type=int, default=3,
  help="ASIC version (default: 3).")
parser.add_argument(
  '-n', '--asic_name', type=str, default="spicy-asics",
  help="ASIC name stem (default: spicy-asics).")


def _write_preflight(fname):
  out_string = (
    "sudo apt install python3.8 -y\n"
    "python3.8 -m pip install pip\n"
    "sudo update-alternatives --install /usr/bin/python3 python3 "
    "/usr/bin/python3.8 1\n"
    "pip install setuptools\n"
    "pip install requests\n"
    "pip install -e ./dist/distla_core")
  with open(fname, "w") as f:
    f.write(out_string)


def _write_submit(fname):
  out_string = (
    "#!/bin/bash\n"
    "# Creates a ASIC slice, runs, and syncs the file named after the first\n"
    "# command line argument\n\n"
    "if [ \"$1\" == \"-h\" ] || [ $# -eq 0 ]; then\n"
    "  echo \"Usage: ./submit.sh FILE_TO_COPY_FROM_ASIC_VM\"\n"
    "  exit 0;\n"
    "fi\n\n"
    "tp create\n"
    "tp run\n"
    "tp sync 0:$1 .\n"
    "tp delete")
  with open(fname, "w") as f:
    f.write(out_string)
  mode = os.stat(fname).st_mode
  mode |= (mode & 292) >> 2
  os.chmod(fname, mode)  # equivalent to chmod +x {fname}


def write_asic_yaml(
  number_of_asic_cores, distla_core_path, asic_name="spicy-asics", zone="us-east1-d",
  runtime_version="v_1", asic_version="v3", asic_preemptible="false",
  preflight="./preflight.sh", entry_point="./main.py",
  rsync_ignore="./rsyncignore", outfile_name="asic.yaml"):

  accelerator_type = f"{asic_version}-{number_of_asic_cores}"

  out_string = (f"asic_name: {asic_name}-{accelerator_type}\n"
                f"zone: {zone}\n"
                f"accelerator_type: {accelerator_type}\n\n"
                f"runtime_version: {runtime_version}\n\n"
                f"asic_preemptible: {asic_preemptible}\n"
                "# determines if allocated from preemptible quota\n\n"
                f"dist_dir: {distla_core_path}\n"
                f"# dist_dir contains user code to move to asic\n\n"
                f"preflight: {preflight}\n"
                "# preflight must be a shell script inside of dist_dir\n"
                "# This property is optional\n\n"
                f"entry_point: {entry_point}\n"
                "# entry_point must be a python file inside of dist dir")

  with open(f"{outfile_name}", "w") as f:
    f.write(out_string)


def make_dirs(slice_sizes, target_path, distla_core_path, asic_version, asic_name):
  target_contents = [f for f in os.listdir(target_path) if os.path.isfile(f)]
  target_srcs = [os.path.join(target_path, fname) for fname in target_contents]
  dirnames = []
  print("Copying files: ")
  for slice_size in slice_sizes:
    dirname = os.path.join(target_path, f"{asic_version}-{slice_size}")
    dirnames.append(dirname)
    copy_dests = [os.path.join(dirname, fname) for fname in target_contents]
    if not os.path.exists(dirname):
      os.mkdir(dirname)

    for src, dest in zip(target_srcs, copy_dests):
      print(dest)
      shutil.copy(src, dest)

    zone = "europe-west4-a"
    if slice_size > 128:
      zone = "us-east1-d"

    asic_yaml_name = os.path.join(dirname, "asic.yaml")
    write_asic_yaml(slice_size, distla_core_path, asic_name=asic_name, zone=zone,
                   asic_version=asic_version, outfile_name=asic_yaml_name)
    submit_sh_name = os.path.join(dirname, "submit.sh")
    _write_submit(submit_sh_name)
    preflight_name = os.path.join(dirname, "preflight.sh")
    _write_preflight(preflight_name)


def main(args):
  target_path = os.getcwd()
  distla_core_path = os.path.abspath(__file__)  # path to this file
  for _ in range(5):
    distla_core_path = os.path.dirname(distla_core_path)  # each time does a ../
  slice_sizes = args.slice_sizes
  asic_version = "v" + str(args.asic_version)
  asic_name = args.asic_name
  _ = make_dirs(slice_sizes, target_path, distla_core_path, asic_version, asic_name)


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)
