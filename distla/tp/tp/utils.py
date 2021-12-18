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
import sys
import os
import yaml


class ApiException(Exception):
  pass


class ArgList(list):
  """Helper class for creating arglists to be passed to subprocess."""
  def append_split(self, element):
    self.extend(element.split())

  def extend_split(self, iterable):
    for element in iterable:
      self.extend(element.split())

  def extend_chain(self, iterable):
    if isinstance(iterable, str):
      return self
    for item in iterable:
      if isinstance(item, str):
        self.append(item)
      else:
        self.extend(item)

  @classmethod
  def from_command(cls, cmd):
    return cls(cmd.split())

class Spinner():
  """Helper class for a spinner in console output."""
  _frames = [
          "⠋",
          "⠙",
          "⠚",
          "⠞",
          "⠖",
          "⠦",
          "⠴",
          "⠲",
          "⠳",
          "⠓"
      ]
  _n_frames = len(_frames)
  def __init__(self):
    self._frame = 0

  def write_frame(self, text):
    """Update the spinner."""
    sys.stdout.write("\33[2K") # Clear line
    sys.stdout.write("\r")
    sys.stdout.write(f'{text}')
    sys.stdout.write(self._frames[self._frame])
    sys.stdout.flush()
    self._frame = (self._frame + 1) % self._n_frames

  def write_finally(self, text):
    """Clear spinner and flush output."""
    sys.stdout.write("\33[2K") # Clear line
    sys.stdout.write("\r")
    sys.stdout.write(f'{text}\n')
    sys.stdout.flush()



def load_conf(conf_path):
  """Load a yaml encoded conf file."""
  try:
    with open(conf_path) as f:
      y = yaml.load(f, Loader=yaml.CLoader)
    return y or {}
  except Exception as e:
    raise ValueError('Invalid or nonexistent config filename')


def dir_contains_file(dir_path, file_path):
  """Return a bool indicating if the file_path is contained within the
  dir path.
  """
  dir_path = os.path.abspath(os.path.expanduser(dir_path))
  file_path = os.path.abspath(os.path.expanduser(file_path))
  return os.path.commonprefix([dir_path, file_path]) == dir_path


def build_remote_path(local_dir, remote_dir, file_path):
  """Translate local_dir/.../file_path -> remote_dir/.../file_path."""
  rel_path = os.path.relpath(file_path, local_dir)
  remote_path = os.path.join(remote_dir, rel_path)
  return remote_path
