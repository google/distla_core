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
A minimal array class for debugging distributed AbstractShardedProbabilityFunction
operations. The array class can be used instead of jax-arrays to debug
operations on the static attributes of AbstractShardedProbabilityFunction for
cases where using actual jax-arrays would be infeasible due to memory
constraints.
"""
from typing import Tuple
import numpy as np
import jax


def flatten(list_of_list):
    return [l for sublist in list_of_list for l in sublist]


class DebugArray:
    """
    Helper array class for debugging AbstractShardedProbabilityFunction.
    """

    def __init__(self, array, shape: Tuple[Tuple[int]]):
        """
        Initialize a `DebugArray`.

        Args:
          array: A jax array of arbitrary size. The shape of the
            passed array can be different from `shape`. The array
            is used only to enable natice jitting support for the class.
          shape: The hypothetical shape of the array. This should be a
            tuple[tuple[int]]. Operations on DebugArray change the
            DebugArray.shape attribute. For example, transposing the
            array transposes the `shape` in the given way.
            `DebugArray.reshape` collects combined axes into a single
            tuple, such that `shape` is a tuple of tuples of length>=1.

        """
        self.shape = tuple(shape)
        self.array = array

    def reshape(self, new_shape):
        _shape = np.cumsum([0] + [int(np.log2(s)) for s in new_shape])
        flatshape = flatten(self.shape)
        shape = []
        for n in range(len(_shape) - 1):
            shape.append(tuple(flatshape[_shape[n] : _shape[n + 1]]))
        self.shape = tuple(shape)
        return self

    def transpose(self, perm):
        self.shape = tuple([self.shape[p] for p in perm])
        return self

    def moveaxis(self, src, dst):
        shape = list(self.shape)
        shape.insert(dst, self.shape[src])
        self.shape = tuple(shape)
        return self


def flatten_DebugArray(array):
    return (array.array,), (array.shape,)


def unflatten_DebugArray(static_data, children):
    return DebugArray(*children, *static_data)


jax.tree_util.register_pytree_node(
    DebugArray, flatten_DebugArray, unflatten_DebugArray
)
