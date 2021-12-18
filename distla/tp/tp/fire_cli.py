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
import os
import sys
import fire
from tp.tp_lib import TP
from tp.utils import load_conf

DEFAULT_CONFIG = './asic.yaml'
VALID_KWARGS = {
    'name': None,
    'asic_name': 'name',
    'zone': None,
    'accelerator_type': None,
    'asic_preemptible': 'preemptible',
    'preemptible': None,
    'dist_dir': None,
    'entry_point': None,
    'setup': None,
    'preflight': None,
    'rsync_ignore': None,
    'run_env': None,
}


def process_arg_dicts(*dicts, required_keys=[]):
  """Flatten all items in dicts into a single dict with subsequent dict items
  overwriting prior ones. Rename kwargs basid on the VALID_KWARGS mapping and
  reject non-valid args.
  """

  # Set default values for non-required keys
  conf = {'accelerator_type': None, 'dist_dir': None, 'entry_point': None}
  for key in required_keys:
    del conf[key]

  for d in dicts:
    for k, v in d.items():
      if k not in VALID_KWARGS.keys():
        print(f'Invalid argument: "{k}"')
        sys.exit()
      new_k = VALID_KWARGS[k] or k
      conf[new_k] = v
  return conf


class TPFire(object):
  """
  A utility for running code on Cloud ASIC VMs.

  gcloud is used to get metadata about the requested asic.
  The alpha asic component must be installed. All connections will use
  the currently active gcloud user account and the ~/.ssh/google_compute_engine
  key for authentication.

  tp will check for a tp.yaml configuration file in the current directory.

  For help run:
  $ tp
  $ tp CMD -- --help

  For each command the following global flags can be supplied:
    name: Name of the ASIC VM.
    zone: Cloud zone of the ASIC VM.
    accelerator_type: Accelerator type string.
    dist_dir: User code directory to distribute to ASIC worker VM's.
    entry_point: User code python entry point. Must be contained within the
      dist_dir.
    preemptible: Bool whether the ASIC VM should be created as preemptible.
    version: Runtime driver version.
    setup: User setup script to run on all workers on creation.
    preflight: User preflight script to run on all workers before 'run' call.
    run_env: Dict of environment variables to set for `run` call.
    rysnc_ignore: Specifies a path to an ignore file. Similar to .gitignore
      files and directories matching rsync exclude patterns in this file will
      not be sent during file transfers.

  Args:
    f: Path to a asic configuration yaml. The values in this file will be used
    as cmd arguments. Arguments you specify will overwrite.
  """

  def __init__(self, f=None):
    if f:
      self._conf = load_conf(f)
    elif f is None and os.path.exists(DEFAULT_CONFIG):
      self._conf = load_conf(DEFAULT_CONFIG)
    else:
      self._conf = {}

  def create(self, use_async=False, no_setup=False, **kwargs):
    conf = process_arg_dicts(self._conf,
                             kwargs,
                             required_keys=['dist_dir', 'accelerator_type'])
    client = TP(**conf)
    client.create(use_async, no_setup)

  def setup(self, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs, required_keys=['dist_dir'])
    client = TP(**conf)
    client.setup()

  def delete(self, use_async=False, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.delete(use_async)

  def info(self, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.info()

  def list(self, zone=None):
    TP.list(zone)

  def ssh(self, worker=0, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.ssh(worker)

  def sync(
      self,
      src,
      dst,
      scatter=False,
      gather=False,
      rsync_args='-azP',
      **kwargs,
  ):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.sync(src, dst, scatter, gather, rsync_args.split())

  def mirror(self, destination='./mirrored', **kwargs):
    conf = process_arg_dicts(self._conf, kwargs, required_keys=['dist_dir'])
    client = TP(**conf)
    client.mirror(destination=destination)

  def exec(self, cmd, worker=None, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.exec(cmd, worker)

  def run(self,
          *run_args,
          no_update=False,
          no_preflight=False,
          use_nohup=False,
          stream_workers=0,
          **kwargs):
    conf = process_arg_dicts(self._conf,
                             kwargs,
                             required_keys=['dist_dir', 'entry_point'])
    client = TP(**conf)
    client.run([str(arg) for arg in run_args], no_update, no_preflight,
               use_nohup, stream_workers)

  def profile(self, path, local_port=8080, remote_port=6006, **kwargs):
    conf = process_arg_dicts(self._conf, kwargs)
    client = TP(**conf)
    client.profile(path)

  @property
  def config(self):
    """Current configuration."""
    return self._conf


# Configure Doc strings for CLI Help
for a in [
    'create', 'setup', 'delete', 'info', 'ssh', 'sync', 'exec', 'run', 'profile'
]:
  getattr(TPFire, a).__doc__ = getattr(TP, a).__doc__


def main():
  fire.Fire(TPFire)


if __name__ == '__main__':
  main()
