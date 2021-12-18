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
import ctypes
from ctypes import byref
import numpy as np


class StructPack:
  """ Wrapper class exposing STRUC_PACK read/write matrix functions.
  Initialize as `foo = StructPack(PATH/TO/libstruc_pack.so)`.
  """
  def __init__(self, lib_path):
    self._libstruc_pack = ctypes.CDLL(lib_path)
    self.rh = ctypes.c_void_p()

  def read_matrix(self, path, format_sparse=True):
    """ Reads the matrix stored at `path`.
    Args:
      path: Location of the matrix.
      format_sparse: The matrix is assumed to be in sparse format if True.
    Returns:
      matrix: The matrix.
      n_electrons: Number of electrons in the system.
    """
    r_task = ctypes.c_int(0)
    para_mode = ctypes.c_int(0)
    n_basis = ctypes.c_int(0)
    n_elec = ctypes.c_double(0.0)
    n_lrow = ctypes.c_int(0)
    n_lcol = ctypes.c_int(0)
    path_buffer = path.encode('ASCII')

    self._libstruc_pack.c_struc_pack_init_rw(
      byref(self.rh), r_task, para_mode, n_basis, n_elec)
    self._libstruc_pack.c_struc_pack_read_mat_dim(
      self.rh, path_buffer, byref(n_elec), byref(n_basis), byref(n_lrow),
      byref(n_lcol))
    matrix = np.zeros([n_basis.value, n_basis.value], dtype='double')

    pointer_to_matrix = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if format_sparse:
      self._libstruc_pack.c_struc_pack_read_mat_real(self.rh, path_buffer,
                                         pointer_to_matrix)
    else:
      self._libstruc_pack.c_struc_pack_read_dense_matrix_real(
        self.rh, path_buffer, pointer_to_matrix)

    self._libstruc_pack.c_struc_pack_finalize_rw(self.rh)
    return matrix, n_elec.value

  def write_matrix(self, path, matrix, format_sparse=True):
    """ Writes `matrix` to `path`.
    Args:
      path: Location of the matrix.
      format_sparse: The matrix is written in sparse format if True.
    Returns:
      None
    """
    w_task = ctypes.c_int(1)
    para_mode = ctypes.c_int(0)
    n_basis = ctypes.c_int(matrix.shape[0])
    n_elec = ctypes.c_double(0.0)
    path_buffer = path.encode('ASCII')


    self._libstruc_pack.c_struc_pack_init_rw(
      byref(self.rh), w_task, para_mode, n_basis, n_elec)

    pointer_to_matrix = matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    if format_sparse:
      self._libstruc_pack.c_struc_pack_write_mat_real(
        self.rh, path_buffer, pointer_to_matrix)
    else:
      self._libstruc_pack.c_struc_pack_write_dense_matrix_real(
        self.rh, path_buffer, pointer_to_matrix)

    self._libstruc_pack.c_struc_pack_finalize_rw(self.rh)
