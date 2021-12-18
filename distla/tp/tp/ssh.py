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
"""Utilities for tp ssh."""

import subprocess
from tp.utils import (ArgList)

EXEC_ENV = {'ASIC_MIN_LOG_LEVEL': '0'}
# ASIC_MIN_LOG_LEVEL sets the lib logging level - 0: info, 3: fatal

KEY_PATH = '~/.ssh/google_compute_engine'
# This is the default gcloud ssh key file path

USER_KNOWN_HOSTS_PATH = '~/.ssh/google_compute_known_hosts'
# This is the default gcloud known hosts file path


def gen_ssh_cmd(user=None,
                ip=None,
                key_path=KEY_PATH,
                user_known_hosts_path='/dev/null',
                port_map=None):
  """Generate and return an ssh shell string."""

  if (user is None) != (ip is None):
    raise ValueError('Either both user and ip must be specified or neither.')

  host_str = f'{user}@{ip} ' if user is not None else ''

  ssh_cmd = (f'ssh -i {key_path} '
             + host_str +
             f'-o UserKnownHostsFile={user_known_hosts_path} '
             '-o StrictHostKeyChecking=no '
             '-o ConnectionAttempts=3 '
             '-o LogLevel=Error ')
  if port_map is not None:
    ssh_cmd += f'-L {port_map[0]}:localhost:{port_map[1]} '
  return ssh_cmd


def _create_ssh_exec_process(user,
                             ip,
                             cmd,
                             env=None,
                             stdout=None,
                             port_map=None):
  """
  Create and return a new ssh subprocess.

  Args:
    user: Username string.
    ip: IP string.
    cmd: Command string to execute via ssh.
    env: Dict of environment variables to execute with.
    stdout: Popen stdout redirect object.
    port_map: tuple of two ints (local_port, remote_port) to forward.
  """
  ssh_cmd = ArgList.from_command(gen_ssh_cmd(user, ip, port_map=port_map))
  if env:
    env_str = ' '.join([f'{key}={val}' for key, val in env.items()])
    ssh_cmd.append(env_str + ' ' + cmd)
  else:
    ssh_cmd.append(cmd)
  return subprocess.Popen(ssh_cmd, stderr=subprocess.STDOUT, stdout=stdout)


def exec_cmd_on_ips(user,
                    ips,
                    asic_name,
                    cmd,
                    env={},
                    stream_ips=None,
                    port_map=None):
  """Run the given cmd on all listed ips.

  Args:
    user: User to authenticate with ASIC VMs.
    ips: list of ips to attempt remote execution.
    asic_name: TP_ASIC_NAME to supply in env.
    cmd: cmd string to execute.
    env: dict of env vars to add before execution.
    stream_ips: list of indices of 'ips' to stream output back from.
      None will stream from all ips.
    port_map: tuple of port_map tuples for each ip
      ((local_port, remote_port),...).
  """
  _name = asic_name
  p_list = []
  _env = EXEC_ENV.copy()
  _env.update(env)

  if port_map is not None:
    assert len(port_map) == len(ips)
  else:
    port_map = [None for _ in range(len(ips))]
  if stream_ips is not None:
    _stream_ips = set(stream_ips)
  else:
    _stream_ips = set(range(len(ips)))

  for i, ip in enumerate(ips):
    _local_env = _env.copy()
    _local_env.update({'TP_ASIC_NAME': _name, 'TP_ASIC_WORKER': i})
    stdout = None if i in _stream_ips else subprocess.DEVNULL
    p_list.append(
        _create_ssh_exec_process(user, ip, cmd, env=_local_env, stdout=stdout, port_map=port_map[i]))
  for p in p_list:
    p.wait()
