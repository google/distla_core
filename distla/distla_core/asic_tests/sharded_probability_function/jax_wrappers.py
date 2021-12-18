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
Wrapper functions for jax
"""
from typing import Union, Sequence, Tuple
import jax
from jax.interpreters import pxla
import jax.numpy as jnp
import numbers
import numpy as np

AXIS_NAME = "i"


def _get_all_to_all_axis_index_groups(grid_shape, sharded_axes):
    """
    Helper function for `all_to_all`. Computes the `axis_index_groups`
    argument for a given opertion.
    """
    grid = np.arange(int(np.prod(grid_shape))).reshape(grid_shape, order="C")
    reduced_shape = [
        grid_shape[s] for s in range(len(grid_shape)) if s not in sharded_axes
    ]
    axis_index_groups = []
    for i in np.ndindex(*reduced_shape):
        slices = list(i)
        for sharded_axis in sharded_axes:
            slices.insert(sharded_axis, slice(0, grid_shape[sharded_axis]))
        axis_index_groups.append(list(np.ravel(grid[tuple(slices)])))
    return axis_index_groups


def _to_tuple(val):
    if isinstance(val, numbers.Number):
        return (val,)
    return tuple(val)


def all_to_all(
    array: pxla.ShardedDeviceArray,
    sharded_axes: Union[int, Sequence[int]],
    split_axes: Union[int, Sequence[int]],
    concat_axes: Union[int, Sequence[int]],
    grid_shape: Tuple[int],
):
    """
    Swap pmapped axes `sharded_axes` with local axes `split_axes`, and place
    them at the local positions `concat_axes`.
    The global part of `array` is assumed to be of shape `grid_shape`, and
    the device placement of each shard of the global array is in 'C' order,
    i.e. shard `i` of the array (the `i`-th element of the pmapped axis) is
    placed on device number `np.ravel(grid)[i]`, with
    `grid = np.arange(jax.device_count()).reshape(grid_shape, order='C')`.
    `sharded_axis`, `split_axes` and `concat_axes` have be either ints, or
    sequences of ints of identical length.

    Args:
      array: A sharded array.
      sharded_axes: The sharded axes to be swapped with local axes.
      split_axes: The local axes to be pmapped.
      concat_axes: local axes positions where `sharded_axes`
        should be placed after localizing them.
      grid_shape: the processor grid shape.

    Returns:
      ShardedDeviceArray: The result of the operation.
    """

    def ind_sort(sequence, inds):
        return tuple([sequence[i] for i in inds])

    sharded_axes = _to_tuple(sharded_axes)
    split_axes = _to_tuple(split_axes)
    concat_axes = _to_tuple(concat_axes)

    if len(split_axes) != len(concat_axes):
        raise ValueError("split_axes and concat_axes are of unequal length")

    if len(split_axes) != len(sharded_axes):
        raise ValueError("split_axes and sharded_axes are of unequal length")

    sharded_dims = np.asarray([grid_shape[a] for a in sharded_axes])
    local_dims = np.asarray([array.shape[a] for a in split_axes])
    if not np.all(sharded_dims == local_dims):
        raise ValueError(
            f"dimensions {sharded_dims} of global axes "
            f"do not match dimensions {local_dims} of "
            f"the local axes"
        )

    # we first sort sharded_axes
    inds = np.argsort(sharded_axes)
    sharded_axes = ind_sort(sharded_axes, inds)
    split_axes = ind_sort(split_axes, inds)
    concat_axes = ind_sort(concat_axes, inds)

    axis_index_groups = _get_all_to_all_axis_index_groups(
        grid_shape, sharded_axes
    )

    if len(split_axes) == 1:
        # this case is already covered within jax
        return jax.lax.all_to_all(
            array,
            axis_name=AXIS_NAME,
            split_axis=split_axes[0],
            concat_axis=concat_axes[0],
            axis_index_groups=axis_index_groups,
        )

    # we move all split_axes to the left side of the array
    # and combine them into a single dimension

    # transpose
    n_split = len(split_axes)
    permarray = jnp.moveaxis(array, split_axes, tuple(range(n_split)))
    # now reshape
    permshape = permarray.shape
    comb_permshape = (int(np.prod(permshape[:n_split])),) + permshape[n_split:]
    permarray = permarray.reshape(comb_permshape)

    # now we swap the local index 0 with `sharded_axes`
    result = jax.lax.all_to_all(
        permarray,
        axis_name=AXIS_NAME,
        split_axis=0,
        concat_axis=0,
        axis_index_groups=axis_index_groups,
    )

    # finally we split the swapped axes back into their original shapes
    # and move them to their final positions.
    final_shape = (
        tuple([grid_shape[a] for a in sharded_axes]) + comb_permshape[1:]
    )
    return jnp.moveaxis(
        result.reshape(final_shape),
        tuple(range(len(sharded_axes))),
        concat_axes,
    )
