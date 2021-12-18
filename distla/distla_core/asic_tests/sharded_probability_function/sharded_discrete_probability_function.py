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
import asic_la.sharded_probability_function.abstract_sharded_probability_function as aswf
from asic_la.sharded_probability_function import complex_workaround as cw


@jax.tree_util.register_pytree_node_class
class ShardedDiscretedProbabilityFunction(aswf.AbstractShardedProbabilityFunction):
    """
    A sharded probability function for representating of the probability function of a
    symplectic computer, i.e. a complex array of shape (2, ) * num_discretes.
    This class can be used to contract symplectic building_blocks of shape (128, 128)
    with a probability function to simulate a symplectic computation. Contraction
    is performed with the `.discrete_dot` method, with the signature
    ```

      result = state.discrete_dot(matrix, axes)

    ```
    `matrix` is a (128, 128) matrix representing a (collection of) symplectic
    building_block(s), and `axes` is a list of integers denoting on which discretes the
    symplectic building_block acts. I.e., `axes` is a sequence of integer labels which
    uniquely identify any discrete throughout the whole simulation.
    As such, the `.discrete_dot` method of this class deviates from the standard numpy
    format, where `axes` correspond to actual tensor-axes which may or may not
    coincide with the discrete label of the axes.

    Attributes:
      concrete_tensor: A possibly distribuetd jax-array representing the
        probability function of a discrete system.
      perm: A list of ints representing discrete labels that are associated
        with the axes of `concrete_tensor`. For example: consider a
        distributed array with a single global and three local discretes.
        `perm=[2,3,1,0]` would mean that:
        global (distributed) axis corresponds to discrete number 2
        axes 0 of `concrete_tensor` corresponds to discrete number 3
        axes 1 of `concrete_tensor` corresponds to discrete number 1
        axes 2 of `concrete_tensor` corresponds to discrete number 0
      distributed: Whether the state is distributed or not.
      num_global_discretes: Number of global discretes. Equal to log_2(num_devices).
      num_discretes: Number of total discretes.
      num_local_discretes: Number of discretes that are represented by the local shard.
    """

    def __init__(self, concrete_tensor, perm, num_global_discretes=None):
        """
        Initialize ShardedDiscretedProbabilityFunction.

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

    def dot(self, matrix, axes, target_ndiscretes):
        raise NotImplementedError(
            "method `dot` is not implemented " "in ShardedDiscretedProbabilityFunction"
        )

    def discrete_dot(self, matrix, axes, target_ndiscretes=7):
        """
        Dot a matrix on the given `axes`. `axes` here is interpreted as a sequence
        of integer labels, each uniquely identifying a discrete in the computation.

        The `self.perm` attribute of `ShardedDiscretedProbabilityFunction` maps each axis of
        `concrete_tensor` to a unique discrete in the computation. This method uses
        the `perm` attribute of `ShardedDiscretedProbabilityFunction` to identify the axes
        with which `matrix` should be contracted.

        Args:
          matrix: A ComplexDeviceArray of shape (2**n, 2**n) where n is len(axes).
          axes: The discrete labels of the axes that should be dotted with `matrix`.
          target_ndiscretes: An optional integer; `matrix` will be extended to a
            (2**target_ndiscretes, 2**target_ndiscretes) shaped matrix to increase
            performance of tensor contractions on ASIC.

        Returns:
          The new ShardedDiscretedProbabilityFunction.
        """
        tensor, perm = self._dot_helper(matrix, axes, target_ndiscretes)
        return ShardedDiscretedProbabilityFunction(tensor, perm, self.num_global_discretes)
