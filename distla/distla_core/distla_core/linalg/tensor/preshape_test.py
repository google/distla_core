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
import jax
import numpy as np
import pytest

from distla_core.linalg.tensor import preshape
from distla_core.linalg.tensor import utils
from distla_core.linalg.utils import testutils as tu
from distla_core.utils import config

AXIS_NAME = config.get_axis_name()


@pytest.mark.parametrize('shape, pshape', tu.get_shapes(3) + tu.get_shapes(4))
def test_pravel(shape, pshape):
  pshape = np.asarray(pshape)
  array = np.random.rand(*shape)
  sharded_array = utils.distribute(array, pshape)
  expected = jax.pmap(
      preshape.pravel,
      in_axes=(0, None),
      static_broadcasted_argnums=(1),
      axis_name=AXIS_NAME)(sharded_array, pshape)
  np.testing.assert_allclose(array.ravel(), expected.ravel())


@pytest.mark.parametrize('shape, pshape', tu.get_shapes(3) + tu.get_shapes(4))
def test_preshape(shape, pshape):
  pshape = np.asarray(pshape)
  newshape = (np.prod(shape[:2]),) + (np.prod(shape[2:]),)
  newpshape = (np.prod(pshape[:2]),) + (np.prod(pshape[2:]),)

  array = np.random.rand(*shape)
  sharded_array = utils.distribute(array, pshape)
  expected = jax.pmap(
      preshape.preshape,
      in_axes=(0, None, None, None),
      static_broadcasted_argnums=(1, 2, 3),
      axis_name=AXIS_NAME)(sharded_array, newshape, newpshape, pshape)
  np.testing.assert_allclose(
      array.reshape(newshape), utils.undistribute(expected, newpshape,
                                                  newshape))
