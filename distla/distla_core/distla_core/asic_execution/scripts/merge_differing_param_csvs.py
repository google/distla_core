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
"""
Loads one or more specified .csv files as DataFrames, combines these using
pandas.concat, and saves the result as another .csv.

The concatenation rule is the Pandas default: the DataFrames are stacked
vertically such that repeated headers are concatenated, with any headers present
in only some DataFrames treated as NaN entries in the others. No checking
for duplicate entries is done.
"""

import argparse
import pandas as pd


parser = argparse.ArgumentParser(
  description="Concatenates specified .csv files.")
parser.add_argument(
  '-i', '--inputs', type=str, nargs='+', help="Paths to load.")
parser.add_argument('-o', '--output', type=str, help="Path to save.")


def merge_csvs(inputs, output):
  data_frames = [pd.read_csv(p) for p in inputs]
  joined = pd.concat(data_frames)
  joined.to_csv(path_or_buf=output, index=False, index_label=False)
  print(f"Saved to {output}.")


def main(args):
  merge_csvs(args.inputs, args.output)


if __name__ == "__main__":
  main(parser.parse_args())
