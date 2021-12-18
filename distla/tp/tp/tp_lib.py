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
import subprocess
import json
import time
import functools
import shutil
import tempfile
from typing import Iterable
import yaml
from tabulate import tabulate

from tp import rsync
from tp import ssh
from tp.utils import (ApiException, ArgList, load_conf, dir_contains_file,
                      build_remote_path, Spinner)

BASE_CMD = 'gcloud container node-pools asic-vm'
TP_PATH = os.path.dirname(os.path.abspath(__file__))
RUNTIME_VERSION = 'v_1'
REMOTE_USER_DIR = 'dist'
DEFAULT_ZONES = ['us-central1-a', 'us-east1-d', 'europe-west4-a']
SPIN_DELAY = 0.1


@functools.lru_cache(maxsize=None)
def get_asic_instance_ips(name, zone):
  """Return list of worker IP's for specified instance."""
  cmd = f'gcloud container node-pools asic-vm describe {name} --zone {zone}'
  p = subprocess.run(ArgList.from_command(cmd), capture_output=True, text=True)
  if p.returncode:
    raise ApiException(p.stderr)
  y = yaml.load(p.stdout, Loader=yaml.CLoader)
  return [
      worker['accessConfig']['externalIp'] for worker in y['networkEndpoints']
  ]


@functools.lru_cache(maxsize=None)
def get_active_user():
  """Return the gcloud active user string."""
  cmd = 'gcloud auth list --filter=status:ACTIVE --format=yaml'
  p = subprocess.run(ArgList.from_command(cmd), capture_output=True, text=True)
  if p.returncode:
    raise ApiException(p.stderr)
  y = yaml.load(p.stdout, Loader=yaml.CLoader)
  return y['account']


def check_full_ready(name, zone):
  """Check if the ASIC VM Instance is HEALTHY"""
  args = ArgList.from_command(BASE_CMD)
  args.append_split(f'describe {name} --zone {zone}')
  args.append('--format=json(state, health)')
  p = subprocess.run(args, capture_output=True)
  if p.returncode:
    raise ApiException(p.stderr)
  d = json.loads(p.stdout)
  if d.get('state', None) == 'READY' and d.get('health', None) == 'HEALTHY':
    return True
  return False


class TP(object):
  """
  A utility for running code on Cloud ASIC VMs.

  gcloud is used to get metadata about the requested asic.
  The alpha asic component must be installed. All connections will use
  the currently active gcloud user account and the ~/.ssh/google_compute_engine
  key for authentication.

  The TP object can be instantiated from arguments or from a YAML configuration
  file with the `from_file` method.
  """

  def __init__(self,
               name,
               zone,
               accelerator_type,
               dist_dir,
               entry_point,
               preemptible=False,
               version=RUNTIME_VERSION,
               setup=None,
               preflight=None,
               run_env={},
               rsync_ignore=None):
    """Initialize a TP object.

    This object is initialized with configuration parameters and can then be
    used to dispatch commands to the ASIC VM it represents.

    Args:
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
    """

    self.name = name
    self.zone = zone
    self.accelerator_type = accelerator_type
    self.dist_dir = dist_dir
    self.entry_point = entry_point
    self.preemptible = preemptible
    self.version = version
    self.user_setup = setup
    self.user_preflight = preflight
    self.run_env = run_env
    self.rsync_ignore = rsync_ignore

  @classmethod
  def from_file(cls, file):
    config = load_conf(file)
    return cls(**config)

  def create(self, use_async=False, no_setup=False):
    """
    Create a new asic instance.

    Args:
      use_async: Specifies whether the command is run in async mode. In async
      mode the command will return immidiately. If not in async mode an
      additional health check will be performed.
      no_setup: Skip running system and user setup scripts. Use async
      automatically skips.
    """
    _name = self.name
    _zone = self.zone
    _accelerator_type = self.accelerator_type
    _version = self.version
    _preemptible = self.preemptible

    print(f'Creating asic: {_name} in zone {_zone} of type {_accelerator_type}')
    print(f'Runtime version: {_version}')
    print(f'Preemptible [{"X" if _preemptible else " "}]')

    args = ArgList.from_command(BASE_CMD)
    args.extend_split([
        'create', f'{_name}', f'--zone {_zone}',
        f'--accelerator-type {_accelerator_type}', f'--version {_version}',
        f'--metadata enable-oslogin=FALSE',
        f'{"--preemptible" if _preemptible else ""}'
    ])
    if use_async:
      if not no_setup:
        print('WARNING - setup will not run when use_async is True')
      args.append('--async')

    c = subprocess.run(args)
    if c.returncode:
      raise ApiException(f'ASIC VM creation failed with '
                         f'return code {c.returncode}.')

    if not use_async:
      spinner = Spinner()
      while not check_full_ready(_name, _zone):
        spinner.write_frame('Waiting for asics to be healthy...')
        time.sleep(SPIN_DELAY)
      spinner.write_finally('Waiting for asics to be healthy...done')
      if not no_setup:
        self.setup()

  def setup(self):
    """
    Run the system and user setup scripts.
    """

    _user_dist_dir = self.dist_dir
    _user_setup = self.user_setup
    _rsync_ignore = self.rsync_ignore

    rsync_args = ['-azP', '--delete']
    if _rsync_ignore:
      rsync_args.append(f'--exclude-from={_rsync_ignore}')

    src_dir = os.path.join(TP_PATH, 'worker_bin/')
    dst_dir = 'bin'
    self.sync(src_dir, dst_dir, scatter=True, rsync_args=rsync_args)

    src_dir = os.path.join(os.path.expanduser(self.dist_dir), '')
    dst_dir = REMOTE_USER_DIR
    self.sync(src_dir, dst_dir, scatter=True, rsync_args=rsync_args)

    print('Running system setup...')
    self.exec('sh bin/setup.sh')

    if _user_setup:
      print('Running user setup...')
      remote_path = build_remote_path(_user_dist_dir, REMOTE_USER_DIR,
                                      _user_setup)
      self.exec(f'sh {remote_path}')

  def delete(self, use_async=False):
    """
    Delete a asic instance.

    Args:
      use_async: Specifies whether the command is run in async mode. In async
      mode the command will return immidiately.
    """
    _name = self.name
    _zone = self.zone

    args = ArgList.from_command(BASE_CMD)
    args.extend_split(['delete', f'{_name}', f'--zone {_zone}'])
    if use_async:
      args.append('--async')
    c = subprocess.run(args)

  def info(self):
    """
    Get info about a zone and asic instance.

    If no zone has been configured the default asic zones will be checked.
    """
    _name = self.name
    _zone = self.zone

    print(f'--{_name} Info--')
    cmd = f'gcloud container node-pools asic-vm describe {_name} --zone {_zone}'
    subprocess.run(ArgList.from_command(cmd))

  @staticmethod
  def list(zone=None):
    """
    List all ASIC VM instances.
    """
    if zone:
      cmd = f'gcloud container node-pools asic-vm list --zone {zone}'
      subprocess.run(ArgList.from_command(cmd))
    else:
      nodes = []
      for z in DEFAULT_ZONES:
        cmd = f'gcloud container node-pools asic-vm list --zone {z}'
        args = ArgList.from_command(cmd)
        args.append(
            '--format=json( name.basename(), name.segment(-3):label=ZONE, '
            'acceleratorType.basename():label=ACCELERATOR_TYPE, '
            'networkConfig.network.basename():label=NETWORK, '
            'cidrBlock:label=RANGE, state:label=STATUS)')
        p = subprocess.run(args, capture_output=True, text=True)
        if not p.returncode:
          nodes.extend([[
              v['name'], z, v['acceleratorType'], v['networkConfig']['network'],
              v['cidrBlock'], v.get('state', 'Unknown')
          ] for v in json.loads(p.stdout)])
        else:
          print(p.stderr)
      print(
          tabulate(nodes,
                   headers=[
                       'NAME', 'ZONE', 'ACCELERATOR_TYPE', 'NETWORK', 'RANGE',
                       'STATUS'
                   ],
                   tablefmt='plain'))

  def ssh(self, worker=0):
    """
    SSH into a asic worker.

    Args:
      worker: The worker number to connect to.
    """
    _name = self.name
    _zone = self.zone

    print(f'Attempting to ssh to {_name} - worker {worker}')

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    cmd = ssh.gen_ssh_cmd(active_user, ips[worker])
    args = ArgList.from_command(cmd)

    c = subprocess.run(args)

  def sync(self, src, dst, scatter=False, gather=False, rsync_args=['-azP']):
    """
    Sync files to or from any number of workers using rsync.

    This command attempts to mimic rsync as closely as possible while providing
    a convenient and intuitive syntax for syncing with multiple remote locations
    at the same time. There are two general syntaxes that can be used:

    Single host syntax - similar to rsync remote paths but a worker number is
    used instead of a user@host string.

    for example:
      tp sync local_file 0:remote_file
        # send a file to worker 0
      tp sync 2:remote_dir/ local_dir
        # get all files in remote_dir on worker 2 and place in local_dir

    Scatter/gather syntax - in this syntax no remote indicators are contained in
    the path. Instead the scatter or gather args are used to determine the
    direction of the sync as well as which workers to involve. This is
    straightforward in the scatter case, the rsync command is executed against
    each host.

    In the gather case this becomes more complex. Files from multiple hosts are
    being gathered to one local location. This function attempts to handle this
    as intuitively as possible, by creating a directory w{worker_num} for each
    worker involved in the dst location and place the src indicated files in
    these folders.

    for example:
      tp sync local_dir/ remote_dir --scatter
        # copies contents of local_dir into remote_dir on each worker
      tp sync remote_dir/ local_dir --gather
        # copies contents of remote_dir on each worker to local_dir/w{num}
      tp sync remote_file local_file --gather='[2,3]'
        # copies remote_file on workers 2, 3 to w{num}/local_file locally

    NOTE: When passing rsync_args all args must be passed in a string literal.
    This literal will replace, not extend the default args. To add args such
    as an rsync exclude the defaults must be also passed:
      tp sync folder 0:remote_dir --rsync_args='-azP --rsync_ignore=./ignore'

    Args:
      src: sync src path, this corresponds to rsync src.
      dst: sync dst path, this corresponds to rsync dst.
      scatter: This specifies that the sync is outgoing. If the value is a bool
        it specifies all workers. If a list it specifies which workers.
      gather: This specifies that the sync is incoming. Same semantics as
        scatter.
      rsync_args: List of strings to pass to the underlying rsync command as
        args. The default args are '-azP'.
    """
    _name = self.name
    _zone = self.zone

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    rsync.validate_sync_args(src, dst, scatter, gather)

    if rsync.is_sync_path(dst):
      w, dst = dst.split(':')
      ip = ips[int(w)]
      rsync.sync_to_ips(src, dst, active_user, [ip], rsync_args=rsync_args)

    elif rsync.is_sync_path(src):
      w, src = src.split(':')
      ip = ips[int(w)]
      rsync.sync_from_ips(src, dst, active_user, [ip], rsync_args=rsync_args)

    elif scatter:
      if isinstance(scatter, list):
        ips = [ips[w] for w in sorted(scatter)]
      rsync.sync_to_ips(src, dst, active_user, ips, rsync_args=rsync_args)

    elif gather:
      if isinstance(gather, list):
        ips = [ips[w] for w in sorted(gather)]
        ids = [f'w{w}' for w in sorted(gather)]
      else:
        ids = [f'w{w}' for w in range(len(ips))]
      rsync.sync_from_ips(src,
                          dst,
                          active_user,
                          ips,
                          ids,
                          rsync_args=rsync_args)

  def mirror(self, destination='./mirrored'):
    """
    Mirror the configured dist_dir to all ASIC VM workers using lsyncd.

    Args:
      destination: The remote destination on the ASIC VMs to mirror the dist_dir.
        This cannot be the home directory.
    """
    _name = self.name
    _zone = self.zone

    if destination in ['.', './', '~', '~/']:
      raise ValueError('Cannot mirror to remote home directory.')

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    lsyncd_path = shutil.which('lsyncd')
    if not lsyncd_path:
      raise EnvironmentError(
          'lsyncd does not appear to be installed. Ensure that lsyncd is '
          'installed and available in PATH.')

    src_path = os.path.join(os.path.expanduser(self.dist_dir), '')
    print(f'Mirroring to {len(ips)} worker(s)...')
    p_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
      for i, ip in enumerate(ips):
        config = rsync._gen_lsyncd_config_string(src_path, destination,
                                                 active_user, ip,
                                                 self.rsync_ignore)
        config_path = os.path.join(tmpdir, f'tp_config{i}')
        with open(config_path, mode='wt') as f:
          f.write(config)
        p_list.append(rsync._create_lsyncd_process(config_path))
      try:
        for p in p_list:
          p.wait()
      except KeyboardInterrupt:
        print('\nKeyboard interrupt, exiting...')

  def exec(self, cmd, worker=None):
    """
    Execute a command over SSH on a asic instance or single worker.

    Args:
      cmd: A command string to execute.
      worker: Which worker to run the command on. If None will all workers
        will execute.
    """
    _name = self.name
    _zone = self.zone

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    if worker is not None:
      ips = [ips[worker]]

    ssh.exec_cmd_on_ips(active_user, ips, _name, cmd)

  def run(self,
          run_args=[],
          no_update=False,
          no_preflight=False,
          use_nohup=False,
          stream_workers=0):
    """
    Transfer and execute code on a asic instance.

    During the process dist_dir is transfered to each ASIC VM, preflight scripts
    are fired, then entry_point is executed with python with the configured
    run_env.

    Args:
      run_args: Argument strings to pass to python call.
      no_update: Flag specifying whether file transfer should be skipped.
      no_preflight: Flag specifying whether preflight should be skipped.
      use_nohup: Run the command with nohup and save output to tp.out. With this
        option execution will not be stopped by closing the ssh connection.
      stream_workers: Int or list of ints representing worker number(s) to
        stream output back from. None will stream from all ips.
        Default is worker 0 only.
    """
    _name = self.name
    _zone = self.zone
    _user_dist_dir = self.dist_dir
    _user_preflight = self.user_preflight
    _entry_point = self.entry_point
    _run_env = self.run_env
    _rsync_ignore = self.rsync_ignore

    if isinstance(stream_workers, int):
      _stream_workers = (stream_workers,)
    elif isinstance(stream_workers, Iterable):
      _stream_workers = set(stream_workers)
    else:
      _stream_workers = None

    if not dir_contains_file(_user_dist_dir, _entry_point):
      raise TypeError('entry_point must be contained within dist_dir')

    if not all([isinstance(arg, str) for arg in run_args]):
      raise TypeError('All run_args elements must be strings')

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    if not no_update:
      rsync_args = ['-azP', '--delete']
      if _rsync_ignore:
        rsync_args.append(f'--exclude-from={_rsync_ignore}')

      # Transfer User Dist Dir
      src_dir = os.path.join(os.path.expanduser(_user_dist_dir), '')
      dst_dir = REMOTE_USER_DIR
      rsync.sync_to_ips(src_dir,
                        dst_dir,
                        active_user,
                        ips,
                        rsync_args=rsync_args)

    if not no_preflight and _user_preflight:
      print('Running user preflight...')
      remote_path = build_remote_path(_user_dist_dir, REMOTE_USER_DIR,
                                      _user_preflight)
      ssh.exec_cmd_on_ips(active_user,
                          ips,
                          _name,
                          f'sh {remote_path}',
                          stream_ips=_stream_workers)

    # Run user code
    print('Running user code...')
    remote_path = build_remote_path(_user_dist_dir, REMOTE_USER_DIR,
                                    _entry_point)

    arg_string = ' '.join(["'" + arg + "'" for arg in run_args])
    if use_nohup:
      print('Running with nohup...')
      cmd = (f'nohup python3 -u {remote_path} {arg_string} &> tp.out & '
             'tail -f tp.out')
    else:
      cmd = f'python3 {remote_path} {arg_string}'

    try:
      ssh.exec_cmd_on_ips(active_user,
                          ips,
                          _name,
                          cmd,
                          env=_run_env,
                          stream_ips=_stream_workers)
    except KeyboardInterrupt:
      print('\nKeyboard interrupt, exiting...')
      if use_nohup:
        print('use_nohup is enabled, '
              'execution will continue to run on the ASIC VM Instance...')

  def profile(self, path, local_port=8080, remote_port=6006):
    """
    Start a tensorboard server instance.

    Tensorboard will be started on the asic VM with the tensorboard remote_port
    forwarded to the local_port on the calling host. Tensorboard will serve
    from the logdir specified at path.

    Args:
      path: logdir path.
      local_port: local port to forward to ASIC VM.
      remote_port: remote port to start tensorboard on.
    """
    _name = self.name
    _zone = self.zone

    if self.accelerator_type != 'v_2':
      return ValueError('Only v_2 profiling is currently supported')

    ips = get_asic_instance_ips(_name, _zone)
    active_user = get_active_user().split('@')[0]

    try:
      ssh.exec_cmd_on_ips(
          active_user,
          ips,
          _name,
          f'tensorboard serve --logdir {path} --port {remote_port}',
          env={},
          port_map=((local_port, remote_port),))
    except KeyboardInterrupt:
      print('\nClosing tensorboard...')
      self.exec('pkill -9 tensorboard')
