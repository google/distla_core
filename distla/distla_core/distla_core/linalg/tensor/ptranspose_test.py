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
"""Test for ptranspose.py."""
import jax
import numpy as np
import pytest

from distla_core.linalg.tensor import ptranspose
from distla_core.linalg.tensor import utils
from distla_core.linalg.utils import testutils as tu
from distla_core.utils import config

AXIS_NAME = config.get_axis_name()


@pytest.mark.parametrize('shape, pshape', tu.get_shapes(3) + tu.get_shapes(4))
def test_ptranspose(shape, pshape):
  pshape = np.asarray(pshape)
  perm = np.arange(len(shape))
  np.random.shuffle(perm)
  array = np.random.rand(*shape)
  sharded_array = utils.distribute(array, pshape)
  pmapped_ptranspose = jax.pmap(
      ptranspose.ptranspose,
      axis_name=AXIS_NAME,
      static_broadcasted_argnums=(1, 2),
      in_axes=(0, None, None))
  actual = utils.undistribute(
      pmapped_ptranspose(sharded_array, perm, pshape),
      [pshape[p] for p in perm], [shape[p] for p in perm])
  expected = array.transpose(perm)
  np.testing.assert_allclose(actual, expected)
