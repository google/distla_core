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
sharded probability functions class for simulating symplectic acyclic_graphs.
"""

import jax
import jax.numpy as jnp
import math
import numpy as np

from asic_la.sharded_probability_function import utils
from asic_la.sharded_probability_function import complex_workaround as cw
import asic_la.sharded_probability_function.abstract_sharded_probability_function as aswf


@jax.tree_util.register_pytree_node_class
class ShardedProbabilityFunction(aswf.AbstractShardedProbabilityFunction):
    """
    Sharded probability function that is optimized for 8x128 ASIC memory lanes.

    Attributes:
      concrete_tensor: A possibly distributed jax-array representing the
        probability function of a discrete system.
      perm: List of ints; the elements in `perm` map the axes of the
        distributed concrete tensor to the corresponding positions if the tensor
        was a numpy array, in the following referred to as the "virtual tensor".
        For example: consider a distributed array
        with a single global and three local discretes. `perm=[2,3,1,0]`
        would mean that:

        global (distributed) axis maps to axis 2 of the virtual tensor
        axes 0 of `concrete_tensor` maps to axis 3 of the virtual tensor
        axes 1 of `concrete_tensor` maps to axis 1 of the virtual tensor
        axes 2 of `concrete_tensor` maps to axis 0 of the virtual tensor

        Note that the elements in `perm` do *not* correspond to any discrete labels.
        The main purpose of having `perm` is to establish a persistent
        mapping from `concrete_tensor` to a virtual tensor.
        The function `ShardedProbabilityFunction.dot` performs "numpy-style"
        contractions of building_blocks with `ShardedProbabilityFunction`, i.e. emulates numpy
        behaviour of the form

        ```
        def dot(self, matrix, axes):
          n = len(axes)
          return np.tensordot(self, matrix.reshape((2,) * (n * 2), [axes, range(n)])
        ```
        Internally, `dot` uses a ASIC friendly complicated mechanism to perform
        this operation. During the operation, the `concrete_tensor` needs to be
        permuted, globally swapped, and so on. `perm` gets updated during all
        these operations such that at any time the distributed array can be
        mapped to its virtual counterpart. After the `dot` operation, the
        elements in `perm` again map global array axes and local array axes
        (i.e. axes of `concrete_tensor`) to the position where they would
        be if the `dot` function had been performed in numpy.
      distributed: Whether the state is distributed or not.
      num_global_discretes: Number of global discretes. Equal to log_2(num_devices).
      num_discretes: Number of total discretes.
      num_local_discretes: Number of discretes that are represented by the local shard.
    """

    def __init__(self, concrete_tensor, perm, num_global_discretes=None):
        """Initialize ShardedProbabilityFunction.

        Args:
          concrete_tensor: The ShardedDeviceArray that stores the amplitudes.
          perm: List of ints of the permutation from the "virtual tensor" to
            concrete_tensor.
          num_global_discretes: Number of global discretes. If `None`, then
            `num_global_discretes = int(np.log2(jax.device_count()))`.
        """
        super().__init__(
            concrete_tensor=concrete_tensor,
            perm=perm,
            num_global_discretes=num_global_discretes,
        )

    def discrete_dot(self, matrix, axes, target_ndiscretes=7):
        raise NotImplementedError(
            "method `discrete_dot` is not implemented " "in ShardedProbabilityFunction"
        )

    def dot(self, matrix, axes, target_ndiscretes=7):
        """
        Dot a matrix on the given virtual axes.

        Equivalent code for the "virtual tensor":

        ```
        def dot(self, matrix, axes):
          n = len(axes)
          return np.tensordot(self, matrix.reshape((2,) * (n * 2), [axes, range(n, 2*n)])
        ```
        Args:
          matrix: A tuple of A (2**n, 2**n) float32 matrices where n is len(axes).
          axes: Which virtual axes to dot with on this ShardedProbabilityFunction.
          target_ndiscretes: An optional integer; `matrix` will be extended to a
            (2**target_ndiscretes, 2**target_ndiscretes) shaped matrix to increase
            performance of tensor contractions on ASIC.

        Returns:
          The new ShardedProbabilityFunction.
        """
        new_tensor, perm = self._dot_helper(matrix, axes, target_ndiscretes)
        new_perm = utils.remove_and_reduce(perm, axes)
        new_perm = (
            new_perm[: self.num_global_discretes]
            + tuple(range(self.num_discretes - len(axes), self.num_discretes))
            + new_perm[self.num_global_discretes :]
        )
        return ShardedProbabilityFunction(new_tensor, new_perm, self.num_global_discretes)

    def transpose(self, perm):
        """Transpose the virtual tensor according to the given permutation.

        Note: This method only updates the perm attribute, as calls no JAX code.

        Args:
          perm: The given permutation.

        Returns:
          The permutated ShardedProbabilityFunction.
        """
        # Proof of how this works
        # perm is a permutation on the axes of the virtual tensor,
        # i.e. the array obtained by mapping the distributed array
        # to e.g. a numpy array using `self.perm`.

        # `self.perm` maps the axes of the `concrete_tensor`
        #  to their respective virtual positions.
        # `tmp` is the permutation that brings `self.perm` into
        # this virtual (linearly increasing) order, i.e.
        # self.perm[tmp] = [0,1,2,3,...]
        tmp = utils.invert_permutation(self.perm)

        # once we are in virtual order we can apply `perm` to `tmp`.
        tmp = utils.permute(tmp, perm)

        # Finally, we invert transform the virtual permutation `tmp`
        # back to `self.perm`.
        new_perm = utils.invert_permutation(tmp)
        # Set tmp to be the new perm and don't touch the concrete_tensor.
        return self.__class__(
            self.concrete_tensor, new_perm, self.num_global_discretes
        )
