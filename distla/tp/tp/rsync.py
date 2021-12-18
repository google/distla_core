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
"""Utilities for tp rsync."""
import os
import string
import subprocess
from os import path
from tp import ssh
from tp.utils import (ArgList)


def _gen_lsyncd_config_string(src, dst, user, host_ip, rsync_ignore, delay=1):
  """Generate an lsyncd config file string."""
  ssh_cmd = ssh.gen_ssh_cmd()
  config_t = string.Template("""
  settings {
    nodaemon = true,
  }

  sync {
    default.rsync,
    source="$src",
    target="$user@$host_ip:$dst",
    delay=$delay,
    rsync = {
      archive = true,
      compress = true,
      whole_file = false,
      rsh = "$ssh_cmd",
      verbose = true,
      $_extra
    },
  }
  """)

  if rsync_ignore is not None:
    _extra = f'_extra = {{"--exclude-from={rsync_ignore}"}}'
  else:
    _extra = ''

  config = config_t.substitute(
      src=src,
      dst=dst,
      user=user,
      host_ip=host_ip,
      ssh_cmd=ssh_cmd,
      _extra=_extra,
      delay=delay,
  )
  return config


def _create_lsyncd_process(config_path):
  """Create and return an new lsyncd subprocess."""
  cmd = ArgList.from_command('lsyncd')
  cmd.append(config_path)
  return subprocess.Popen(cmd)


def _create_rsync_process(src, dst, user, ip, rsync_args):
  """Create and return a new rsync subprocess."""
  ssh_cmd = ssh.gen_ssh_cmd(user, ip)
  cmd = ArgList.from_command('rsync')
  cmd.extend(rsync_args)
  cmd.extend([f'{src}', '-e', ssh_cmd, dst])
  return subprocess.Popen(cmd)


def validate_sync_args(src, dst, scatter, gather):
  num_remote = sum([is_sync_path(s) for s in (src, dst)])
  if sum([bool(scatter), bool(gather), num_remote]) != 1:
    raise ValueError(
        "The scatter and gather options must be exclusively "
        "present or the ':' remote indicator must be present in exactly one of "
        "src and dst")


def is_sync_path(path):
  return len(path.split(':')) == 2


def prepare_worker_sync_paths(dst, ids):
  """Create new w{num} directories and return a list of paths that represent dst
  with a worker subdirectory.
  """
  if path.exists(dst):
    # file . -> ./w0/file
    paths = [path.join(dst, id) for id in ids]
    for p in paths:
      if not path.exists(p):
        os.mkdir(p)
    return paths
  else:
    # file new_file -> ./w0/new_file
    base = path.dirname(dst)
    leaf = path.basename(dst)
    new_dst = []
    for id in ids:
      tmp = path.join(base, id)
      p = path.join(tmp, leaf)
      new_dst.append(p)
      if not path.exists(tmp):
        os.makedirs(tmp)
  return new_dst


def sync_from_ips(src, dst, user, ips, ids=[], rsync_args=[]):
  """Sync remote src to local dst, creating a w{num} subdirectory for each
  worker.
  """
  if len(ips) > 1 and len(ips) != len(ids):
    raise ValueError(
        'ips and ids must be the same length if more than one ip is present')
  if len(ips) > 1:
    dst_list = prepare_worker_sync_paths(dst, ids)
  else:
    dst_list = [dst]
  p_list = [
      _create_rsync_process(f':{src}', dst, user, ip, rsync_args)
      for ip, dst in zip(ips, dst_list)
  ]
  for p in p_list:
    p.wait()


def sync_to_ips(src, dst, user, ips, rsync_args=[]):
  """Sync local src to dst on all remote ips."""
  p_list = [
      _create_rsync_process(src, f':{dst}', user, ip, rsync_args) for ip in ips
  ]
  for p in p_list:
    p.wait()
