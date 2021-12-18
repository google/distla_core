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
""" Traverses directories named v3-{number}, loads {filename}.csv files if they
exist within them, and joins them together into a single {filename}.csv file
within this directory, in which {number} now serves as a new field.
"""

import argparse
import os

import pandas as pd


parser = argparse.ArgumentParser(
  description="Traverses ASIC execution directories and merges csv files.")
parser.add_argument(
  '-f', '--filename', type=str,
  help="{filename} will be loaded.")
parser.add_argument(
  '-v', '--version', type=str, default="v3",
  help="ASIC version name.")


def merge_csvs(filename, version):
  here = os.path.realpath(os.getcwd())
  vlen = len(version)
  directories = [f for f in os.listdir(here) if os.path.isdir(f)]
  version_match = [f for f in directories if f[:vlen] == version]
  number_match = [f for f in version_match if f[vlen + 1:].isnumeric()]
  all_numbers = [int(f[vlen + 1:]) for f in number_match]
  paths = [os.path.join(here, d, filename) for d in number_match]
  paths = [x for _, x in sorted(zip(all_numbers, paths))]
  all_numbers = sorted(all_numbers)
  data_frames = []
  numbers = []
  for n, p in zip(all_numbers, paths):
    try:
      data_frames.append(pd.read_csv(p))
      numbers.append(n)
      print(f"{p} loaded.")
    except FileNotFoundError:
      print(f"{p} not found.")
  joined = pd.concat(data_frames, keys=numbers)
  joined.to_csv(os.path.join(here, filename),
                index_label=("Slice Size", "Index"))


def main(args):
  filename = args.filename
  version = args.version
  merge_csvs(filename, version)


if __name__ == "__main__":
  main(parser.parse_args())
