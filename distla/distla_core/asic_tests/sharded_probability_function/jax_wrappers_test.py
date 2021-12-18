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
"""Tests for pops.py."""
import jax
import jax.numpy as jnp

import numpy as np
import pytest
from scipy import sparse
from asic_la.sharded_probability_function import jax_wrappers

backend = jax.lib.xla_bridge.get_backend()
asic_only = pytest.mark.skipif(backend.platform != "asic", reason="only on ASICs")


@asic_only
@pytest.mark.parametrize("N_local", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("seed", np.arange(10))
def test_all_to_all(N_local, seed):
    np.random.seed(seed)
    N_global = int(np.log2(jax.local_device_count()))
    N = N_global + N_local
    shape = (2,) * N
    grid = (2,) * N_global
    dist_shape = (jax.local_device_count(),) + (2,) * N_local
    nparray = np.random.rand(*shape)
    array = jax.pmap(lambda x: x, devices=jax.local_devices())(
        nparray.reshape(dist_shape)
    )

    sharded_axis = np.random.randint(0, N_global)
    split_axis = np.random.randint(0, N_local)
    concat_axis = split_axis
    result = jax.pmap(
        jax_wrappers.all_to_all,
        static_broadcasted_argnums=(1, 2, 3, 4),
        axis_name="i",
    )(array, sharded_axis, split_axis, concat_axis, grid)

    actual = np.array(result).reshape(shape)
    perm = np.arange(N)
    t = perm[sharded_axis]
    perm[sharded_axis] = perm[N_global + split_axis]
    perm[N_global + split_axis] = t

    exp = np.squeeze(nparray.transpose(perm))
    np.testing.assert_allclose(actual, exp)


@asic_only
@pytest.mark.parametrize("N_local", [5, 6, 7])
@pytest.mark.parametrize("grid", [(2, 2, 2), (4, 2), (2, 4), (8,)])
@pytest.mark.parametrize("seed", np.arange(10))
def test_all_to_all_general(N_local, grid, seed):
    np.random.seed(seed)
    N_global = len(grid)
    N = N_global + N_local

    sharded_axes = np.sort(
        np.random.choice(np.arange(N_global), size=len(grid), replace=False)
    )
    split_axes = np.sort(
        np.random.choice(np.arange(N_local), size=len(grid), replace=False)
    )
    concat_axes = np.random.choice(
        np.arange(N_local), size=len(grid), replace=False
    )
    shape = np.full(N, fill_value=2)
    shape[sharded_axes] = np.asarray(grid)[sharded_axes]
    shape[split_axes + len(grid)] = np.asarray(grid)[sharded_axes]
    shape = tuple(shape)
    dist_shape = (jax.local_device_count(),) + shape[N_global:]
    nparray = np.random.rand(*shape)
    array = jax.pmap(lambda x: x, devices=jax.local_devices())(
        nparray.reshape(dist_shape)
    )

    result = jax.pmap(
        jax_wrappers.all_to_all,
        static_broadcasted_argnums=(1, 2, 3, 4),
        axis_name="i",
    )(array, sharded_axes, split_axes, concat_axes, grid)
    sharded_axes = tuple(sharded_axes)
    split_axes = tuple(split_axes)
    concat_axes = tuple(concat_axes)

    source = sharded_axes + tuple([s + len(grid) for s in split_axes])
    dest = tuple([s + len(grid) for s in split_axes]) + sharded_axes
    source2 = tuple([s + len(grid) for s in split_axes])
    dest2 = tuple([s + len(grid) for s in concat_axes])
    exp = np.moveaxis(np.moveaxis(nparray, source, dest), source2, dest2)
    actual = np.array(result).reshape(exp.shape)
    np.testing.assert_allclose(actual, exp)
