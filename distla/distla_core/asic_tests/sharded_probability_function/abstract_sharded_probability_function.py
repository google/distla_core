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
Base class for sharded probability functions
"""
import jax
import jax.numpy as jnp
import math
import numpy as np
import os

from asic_la.sharded_probability_function import complex_workaround as cw
from asic_la.sharded_probability_function import debug_array
from asic_la.sharded_probability_function import distributed_sampling
from asic_la.sharded_probability_function import jax_wrappers
from asic_la.sharded_probability_function import utils

AXIS_NAME = jax_wrappers.AXIS_NAME
JAX_BACKEND = jax.lib.xla_bridge.get_backend().platform
MIN_LOCAL_DISCRETEDS = 18
MAX_GLOBAL_DISCRETEDS = 11
# if ENABLE_ALL_TO_ALL is set we use jax.lax.all_to_all with `axis_index_groups`
# to swap individual global with local discrete-axes. The default version always
# swaps all global discretes simultaneously. The experimental version is more
# efficient if one needs to only swap a fraction of the global discretes.
# Unfortunately there is an XLA bug on CPU in jax.lax.all_to_all if
# axis_index_groups is different from `None`, so only enable this on ASICs.
ENABLE_ALL_TO_ALL = False
if "PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL" in os.environ:
    if int(os.environ["PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL"]) > 0:
        ENABLE_ALL_TO_ALL = True


@jax.tree_util.register_pytree_node_class
class AbstractShardedProbabilityFunction:
    """
    Abstract base class for sharded probability functions. A sharded probability function
    is the representation of the probability function of a symplectic computer, i.e.
    a complex array of shape (2, ) * num_discretes. This abstract class implements
    infrastructure needed to contract tensors into a distributed probability function
    on a ASIC device.
    The implementation is optimized for 8x128 ASIC memory lanes. Concrete classes
    used for actual computations are derived from this base class. Currently, two
    derived classes are used:

      * ShardedProbabilityFunction
      * ShardedDiscretedProbabilityFunction

    The two classes differ in how they contract symplectic building_blocks into the wave
    function, see each class for details.

    Attributes:
      concrete_tensor: A possibly distributed jax-array representing the
        probability function of a discrete system. To avoid padding effects on ASIC, the
        last 10 indices of the array are kept in a shape (8, 128).
      perm: A list of integer values mapping the virtual probability function to the concrete
        tensor. The precise meaning of `perm` depends on how subclasses use it.
        For details, see the docstring of the corresponding subclass.
      distributed: Whether the state is distributed or not.
      num_global_discretes: Number of global discretes. Equal to log_2(num_devices).
      num_discretes: Number of total discretes.
      num_local_discretes: Number of discretes that are represented by the local shard.
    """

    def __init__(self, concrete_tensor, perm, num_global_discretes=None):
        """Initialize an AbstractShardedProbabilityFunction.

        Args:
          concrete_tensor: The ShardedDeviceArray that stores the amplitudes.
          perm: List of ints of the permutation from the "virtual tensor" to
            concrete_tensor.
          num_global_discretes: Number of global discretes. If `None`, then
            `num_global_discretes = int(np.log2(jax.device_count()))`.
        """
        assert set(perm) == set(
            range(len(perm))
        ), f"{perm} is not a valid permutation"

        self.perm = tuple(perm)
        if num_global_discretes is None:
            num_global_discretes = int(math.log2(jax.device_count()))
        self.num_global_discretes = num_global_discretes

        if (
            self.distributed
            and len(perm) < MIN_LOCAL_DISCRETEDS + self.num_global_discretes
        ):
            raise ValueError(
                f"Number of discretes = {len(perm)} is too small"
                f", need at least "
                f"{MIN_LOCAL_DISCRETEDS + self.num_global_discretes} "
                f"discretes"
            )
        if num_global_discretes < 0:
            raise ValueError("number of global discretes cannot be negative.")

        self.num_discretes = len(perm)
        self.num_local_discretes = self.num_discretes - self.num_global_discretes

        if not isinstance(
            concrete_tensor, (cw.ComplexDeviceArray, debug_array.DebugArray)
        ):
            concrete_tensor = cw.ComplexDeviceArray(
                concrete_tensor.real, concrete_tensor.imag
            )

        self.concrete_tensor = concrete_tensor

    @property
    def distributed(self):
        return self.num_global_discretes > 0

    @property
    def _testing_correctly_permuted_tensor(self):
        """
        Permute the `concrete_tensor` such that `self.perm` becomes the
        identity permutation, and return the resulting tensor.
        Only used for testing, only returns the real part of the tensor
        and only for non-distributed probabilityfunctions.
        """
        assert not self.distributed
        tensor = cw.reshape(self.concrete_tensor, (2,) * self.num_discretes)
        tensor = cw.transpose(tensor, utils.invert_permutation(self.perm))
        return tensor.real

    @property
    def major_lane_num_discretes(self):
        return 7

    @property
    def major_lane_size(self):
        return 2 ** self.major_lane_num_discretes

    @property
    def minor_lane_num_discretes(self):
        return 3

    @property
    def non_free_num_discretes(self):
        return self.major_lane_num_discretes + self.minor_lane_num_discretes

    @property
    def free_num_discretes(self):
        return self.num_local_discretes - self.non_free_num_discretes

    @property
    def global_axes(self):
        """Virtual axes for the global discretes."""
        return self.perm[: self.num_global_discretes]

    @property
    def local_axes(self):
        """Virtual axes for the local discretes."""
        return self.perm[self.num_global_discretes :]

    @property
    def local_perm(self):
        """Permutation of just the local shard, ignoring the global discretes."""
        perm = utils.remove_and_reduce(self.perm, self.global_axes)
        return utils.invert_permutation(perm)

    @property
    def free_axes(self):
        """Virtual axes that can be permuted freely without memory padding."""
        return self.perm[self.num_global_discretes : -self.non_free_num_discretes]

    @property
    def minor_axes(self):
        """Virtual axes for the minor axes of the memory lane (sized 8)."""
        return self.perm[
            -self.non_free_num_discretes : -self.major_lane_num_discretes
        ]

    @property
    def major_axes(self):
        """Virtual axes for the major axes of the memory lane (sized 128)."""
        return self.perm[-self.major_lane_num_discretes :]

    @property
    def non_free_axes(self):
        """Virtual axes that are in either the major or minor lane."""
        return self.perm[-self.non_free_num_discretes :]

    @classmethod
    def zeros(cls, num_discretes, perm=None, num_global_discretes=None):
        """
        Create a state vector filled with 0.

        Args:
          num_discretes: Number of discretes for the probabilityfunction.
          perm: A list of integer values mapping the virtual probability function to the concrete
            tensor. The precise meaning of `perm` depends on how subclasses use it.
            For details, see the docstring of the corresponding subclass.
          num_global_discretes: Number of global discretes. If `None`, then
            `num_global_discretes = int(np.log2(jax.device_count()))`.

        Returns:
          AbstractShardedProbabilityFunction: A probabilityfunction whose amplitudes are all zero.
        """
        if num_global_discretes is None:
            num_global_discretes = int(np.log2(jax.device_count()))
        if num_global_discretes < 0:
            raise ValueError("number of global discretes cannot be negative")

        num_local_discretes = num_discretes - num_global_discretes
        assert (
            num_local_discretes >= MIN_LOCAL_DISCRETEDS
        ), f"{num_local_discretes} is less than the minimum support on ASIC: {MIN_LOCAL_DISCRETEDS}"

        real = jnp.zeros(2 ** num_local_discretes, dtype=jnp.float32)
        imag = jnp.zeros(2 ** num_local_discretes, dtype=jnp.float32)
        imag = jax.lax.tie_in(real, imag)
        state = cw.ComplexDeviceArray(real, imag)
        concrete_tensor = cw.reshape(
            state,
            # Minus 10 is need to account for the (8, 128), a.k.a. (2**3, 2**7)
            ((2,) * (num_local_discretes - 10) + (8, 128)),
        )
        if perm is None:
            perm = tuple(range(num_discretes))
        return cls(concrete_tensor, perm, num_global_discretes)

    @classmethod
    def zero_state(cls, num_discretes, num_global_discretes=None):
        """
        Create a zerostate |0000...> probabilityfunction. If ravelled, the
        `concrete_tensor` would have the form [1,0,0,...,0] in the
        tensor-product basis of the two-discrete states q1 x q2 x ... qN
        (with tensor products carried out from right to left, i.e.
        np.kron(q1, np.kron(q2, np.kron(q3,...)))

        Args:
          num_discretes: Number of discretes for the probabilityfunction.
          num_global_discretes: Number of global discretes. If `None`, then
            `num_global_discretes = int(np.log2(jax.device_count()))`.

        Returns:
          ShardedDiscretedProbabilityFunction: The initialized probability function.
        """

        if num_global_discretes is None:
            num_global_discretes = int(np.log2(jax.device_count()))
        if num_global_discretes < 0:
            raise ValueError("number of global discretes cannot be negative")
        if num_global_discretes > 0:
            idx = jax.lax.axis_index("i")
        else:
            idx = 0

        num_local_discretes = num_discretes - num_global_discretes
        assert (
            num_local_discretes >= MIN_LOCAL_DISCRETEDS
        ), f"{num_local_discretes} is less than the minimum support on ASIC: {MIN_LOCAL_DISCRETEDS}"

        real = jnp.zeros(2 ** num_local_discretes, dtype=jnp.float32)
        real = jax.lax.cond(
            idx == 0,
            lambda x: jax.ops.index_update(x, 0, 1.0),
            lambda x: x,
            real,
        )
        imag = jnp.zeros(2 ** num_local_discretes, dtype=jnp.float32)
        imag = jax.lax.tie_in(real, imag)
        state = cw.ComplexDeviceArray(real, imag)
        concrete_tensor = cw.reshape(
            state,
            # Minus 10 is need to account for the (8, 128), a.k.a. (2**3, 2**7)
            ((2,) * (num_local_discretes - 10) + (8, 128)),
        )
        perm = tuple(range(num_discretes))
        return cls(concrete_tensor, perm, num_global_discretes)

    def _add_sub_checks(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot add type {type(self)} with type {type(other)}"
            )

        if self.num_discretes != other.num_discretes:
            raise ValueError("Wave functions must have equal num_discretes")

    def __add__(self, other):
        """
        Add self and other. The result has the same `perm` attribute as `other.perm`.
        """
        self._add_sub_checks(other)
        if self.perm != other.perm:
            state = self.align_axes(other.perm)
        else:
            state = self
        return self.__class__(
            state.concrete_tensor + other.concrete_tensor,
            state.perm,
            state.num_global_discretes,
        )

    def __sub__(self, other):
        """
        Subtract other from self. The result has the same `perm` attribute as `other.perm`.
        """
        self._add_sub_checks(other)
        if self.perm != other.perm:
            state = self.align_axes(other.perm)
        else:
            state = self
        return self.__class__(
            state.concrete_tensor - other.concrete_tensor,
            state.perm,
            state.num_global_discretes,
        )

    def __mul__(self, other):
        return self.__class__(
            self.concrete_tensor * other, self.perm, self.num_global_discretes
        )

    def __rmul__(self, other):
        return self * other

    def move_to_left(self, axes):
        """Move the given free axes to the left of all free axes while preserving
        their relative order.

        Args:
          axes: List of ints of the virtual axes.

        Returns:
          A new ShardedProbabilityFunction.
        """
        assert set(axes).issubset(
            set(self.free_axes)
        ), f"{axes} not a subset of {self.free_axes}"
        target = utils.send_to_left_side(axes, self.free_axes)
        perm = utils.relative_permutation(self.free_axes, target)
        new_tensor = cw.transpose(
            self.concrete_tensor, perm + tuple([len(perm), len(perm) + 1])
        )
        new_perm = self.global_axes + target + self.non_free_axes
        return self.__class__(new_tensor, new_perm, self.num_global_discretes)

    def move_to_right(self, axes):
        """Move the given virtual axes to the inner right side of the free axes
        while preserving their relative order.

        Args:
          axes: List of ints of the virtual axes.

        Returns:
          A new ShardedProbabilityFunction.
        """
        assert set(axes).issubset(
            set(self.free_axes)
        ), f"{axes} not a subset of {self.free_axes}"
        target = utils.send_to_right_side(axes, self.free_axes)
        perm = utils.relative_permutation(self.free_axes, target)
        new_tensor = cw.transpose(
            self.concrete_tensor, perm + tuple([len(perm), len(perm) + 1])
        )
        new_perm = self.global_axes + target + self.non_free_axes
        return self.__class__(new_tensor, new_perm, self.num_global_discretes)

    def swap_major(self):
        """
        Swap major lane discretes with the seven left-most free local discretes"""
        tmp = cw.reshape(
            self.concrete_tensor,
            # New shape of (2, 2, 2, 2, ...., 128, 128)
            ((128,) + (2,) * (self.num_local_discretes - 17) + (8, 128)),
        )
        perm = list(range(len(tmp.shape)))
        perm[0] = len(perm) - 1
        perm[-1] = 0
        tmp = cw.transpose(tmp, perm)
        tmp = cw.reshape(tmp, (2,) * (self.num_local_discretes - 10) + (8, 128))
        new_perm = (
            self.global_axes
            + self.major_axes
            + self.free_axes[7:]
            + self.minor_axes
            + self.free_axes[:7]
        )
        obj = self.__new__(type(self))
        obj.__init__(tmp, new_perm, self.num_global_discretes)
        return obj

    def swap_minor(self):
        """
        Swap minor lane discretes with the three left-most free local discretes"""
        tmp = cw.reshape(
            self.concrete_tensor,
            # New shape of (2, 2, 2, 2, ...., 128, 128)
            ((8,) + (2,) * (self.num_local_discretes - 13) + (8, 128)),
        )
        perm = list(range(len(tmp.shape)))
        perm[0] = len(perm) - 2
        perm[-2] = 0
        tmp = cw.transpose(tmp, perm)
        tmp = cw.reshape(tmp, (2,) * (self.num_local_discretes - 10) + (8, 128))
        new_perm = (
            self.global_axes
            + self.minor_axes
            + self.free_axes[3:]
            + self.free_axes[:3]
            + self.major_axes
        )
        obj = self.__new__(type(self))
        obj.__init__(tmp, new_perm, self.num_global_discretes)
        return obj

    def left_cycle_perm(self):
        """Cyclicly shift all axes by 7 discretes to the left, sending first 7 discretes
        into the major lane"""
        tmp = cw.reshape(
            self.concrete_tensor,
            # New shape of (2, 2, 2, 2, ...., 128, 128)
            ((128,) + (2,) * (self.num_local_discretes - 17) + (8, 128)),
        )
        tmp = cw.moveaxis(tmp, 0, -1)
        tmp = cw.reshape(tmp, (2,) * (self.num_local_discretes - 10) + (8, 128))
        new_perm = (
            self.global_axes
            + self.free_axes[7:]
            + self.non_free_axes
            + self.free_axes[:7]
        )
        return self.__class__(tmp, new_perm, self.num_global_discretes)

    def minor_left_cycle_perm(self):
        """Shift free axes and minor axes by three discretes to the left, sending first
        three discretes into the minor lane."""
        tmp = cw.reshape(
            self.concrete_tensor,
            ((8,) + (2,) * (self.num_local_discretes - 13) + (8, 128)),
        )
        tmp = cw.moveaxis(tmp, 0, -2)
        tmp = cw.reshape(tmp, (2,) * (self.num_local_discretes - 10) + (8, 128))
        new_perm = (
            self.global_axes
            + self.free_axes[3:]
            + self.minor_axes
            + self.free_axes[:3]
            + self.major_axes
        )
        return self.__class__(tmp, new_perm, self.num_global_discretes)

    def dot(self, matrix, axes, target_ndiscretes):
        raise NotImplementedError(
            "method `dot` is not implemented " "in AbstractShardedProbabilityFunction"
        )

    def discrete_dot(self, matrix, axes, target_ndiscretes):
        raise NotImplementedError(
            "method `qdot` is not implemented " "in AbstractShardedProbabilityFunction"
        )

    def move_to_non_free(self, collectables):
        """
        Move `collectables` into the non-free axes, in the given order.
        The method requires free_num_discretes > 7.

        Args:
          collectables: A tuple of virtual axes values to be moved into
            the non-free concrete axes. All values in `collectables`
            have to be in `self.local_axes`.

        Returns:
          AbstractShardedProbabilityFunction: The result of the operation.

        Raises:
          ValueError: If not all `collectables` are contained in
            `AbstractShardedProbabilityFunction.local_axes`.
          ValueError: If `minor_lane_num_discretes` >= `free_num_discretes`
          ValueError: If `len(collectables)` != non_free_num_discretes.

        """
        if self.minor_lane_num_discretes >= self.free_num_discretes:
            raise ValueError(
                f"Number of free discretes = {self.free_num_discretes} "
                f"is smaller or equal to the number of non-free "
                f"discretes = {self.non_free_num_discretes}"
            )

        if not set(self.local_axes) & set(collectables) == set(collectables):
            raise ValueError(
                f"not all values in `collectables` are in "
                f"self.local_axes, got collectables = {collectables}"
                f" and self.local_axes = {self.local_axes}"
            )
        if len(collectables) != self.non_free_num_discretes:
            raise ValueError(
                f"len(collectables) = {len(collectables)} is different"
                f" from number of non-free discretes"
                f" = {self.non_free_num_discretes}"
            )

        if len(self.local_axes) < 21:
            # In this case we need to do it in two steps.
            # For larger number of discretes, this is slower
            # than the code below
            minor_collectables = collectables[:3]
            major_collectables = collectables[3:]
            state = self
            state = state.move_to_minor(minor_collectables)
            return state.move_to_major(major_collectables)

        minor_collectables = collectables[
            -self.non_free_num_discretes : -self.non_free_num_discretes
            + self.minor_lane_num_discretes
        ]
        major_collectables = collectables[-self.major_lane_num_discretes :]
        collectables = major_collectables + minor_collectables
        state = self
        while any_collectables_in_non_free(state, collectables):
            state = collect_move_right(state, collectables)
            if any_collectables_in_major(state, collectables):
                state = state.left_cycle_perm()
            if any_collectables_in_minor(state, collectables):
                state = state.minor_left_cycle_perm()
            if not any_collectables_in_non_free(state, collectables):
                break

            state = collect_move_left(state, collectables)
            if any_collectables_in_major(state, collectables):
                state = state.left_cycle_perm()
            if any_collectables_in_minor(state, collectables):
                state = state.minor_left_cycle_perm()

        state = state.move_to_left(collectables)
        state = state.left_cycle_perm()
        state = state.minor_left_cycle_perm()
        return state

    def move_to_major(self, collectables):
        """
        Move `collectables` into the major axes, in the given order.
        This method requires free_num_discretes > 7.  If the minor axes
        are disjoint from `collectables`, then they remain unchanged.

        Args:
          collectables: A tuple of virtual axes values to be moved into
            the non-free concrete axes. All values in `collectables`
            have to be in `self.local_axes`.

        Returns:
          AbstractShardedProbabilityFunction: The result of the operation.

        Raises:
          ValueError: If not all `collectables` are contained in
            `AbstractShardedProbabilityFunction.local_axes`.
          ValueError: If `free_num_discretes` <= 7
          ValueError: If len(collectables) != 7
          AssertionError: If the routine fails to move collectables to major.
        """
        if len(collectables) != 7:
            raise ValueError(
                f"len(collectables) = {len(collectables)} "
                "is different from 7"
            )
        if self.free_num_discretes <= 7:
            raise ValueError(
                f"Number of free discretes = {self.free_num_discretes} "
                f"is smaller or equal to 7 "
            )

        if not set(collectables).issubset(set(self.local_axes)):
            raise ValueError(
                f"`collectables` is not a subset of "
                f"self.local_axes, got collectables = {collectables}"
                f" and self.local_axes = {self.local_axes}"
            )
        state = self
        collectables_set = set(collectables)
        if collectables_set == set(state.major_axes):
            if collectables != state.major_axes:
                state = state.swap_major()
                state = state.move_to_left(collectables)
                state = state.swap_major()
            return state

        if not set(state.minor_axes).isdisjoint(collectables_set):
            if len(set(state.free_axes) - collectables_set) < 3:
                # in this case we first need to move as many collectables
                # into major as possible
                state = collect_move_left(state, collectables)
                state = state.swap_major()
            if len(set(state.free_axes) - collectables_set) < 3:
                raise ValueError(
                    f"cannot move `collectables` = {collectables} "
                    f"into major axes for perm = {state.perm}."
                )
            state = collect_move_right(state, collectables)
            state = state.minor_left_cycle_perm()
        # now all collectables are in either major or free axes
        if num_collectables_in_major(state, collectables) <= 3:
            # more than half of the collectables are in the free axes.
            # in this case
            state = collect_move_left(state, collectables)
            state = state.swap_major()

        while not all_collectables_in_major(state, collectables):
            state = collect_move_right(state, collectables)
            state = state.swap_major()
            state = collect_move_left(state, collectables)
            state = state.swap_major()

        assert state.major_axes == collectables, "move_to_major failed!"
        return state

    def move_to_minor(self, collectables):
        """
        Move `collectables` into the minor axes, in the given order.
        This method requires free_num_discretes > 7. If the major axes
        are disjoint from `collectibles`, then they remain unchanged.

        Args:
          collectables: A tuple of virtual axes values to be moved into
            the non-free concrete axes. All values in `collectables`
            have to be in `self.local_axes`.

        Returns:
          AbstractShardedProbabilityFunction: The result of the operation.

        Raises:
          ValueError: If not all `collectables` are contained in
            `AbstractShardedProbabilityFunction.local_axes`.
          ValueError: If `free_num_discretes` <= 7
          ValueError: if len(collectables) != 3
          AssertionError: If the routine fails to move collectables to minor.
        """
        if len(collectables) != 3:
            raise ValueError(
                f"len(collectables) = {len(collectables)} "
                "is different from 3"
            )
        if self.free_num_discretes <= 7:
            raise ValueError(
                f"Number of free discretes = {self.free_num_discretes} "
                f"is smaller or equal to 7."
            )

        if not set(collectables).issubset(set(self.local_axes)):
            raise ValueError(
                f"`collectables` is not a subset of "
                f"self.local_axes, got collectables = {collectables}"
                f" and self.local_axes = {self.local_axes}"
            )
        state = self
        if state.minor_axes == tuple(collectables):
            return state

        if num_collectables_in_major(state, collectables):
            if (
                num_collectables_in_free(state, collectables)
                > state.free_num_discretes - state.major_lane_num_discretes
            ):
                state = collect_move_left(state, collectables)
                state = state.swap_minor()
            state = collect_move_right(state, collectables)
            state = state.left_cycle_perm()
        if num_collectables_in_minor(state, collectables):
            state = collect_move_right(state, collectables)
            state = state.swap_minor()
        state = collect_move_left(state, collectables)
        state = state.swap_minor()

        assert state.minor_axes == collectables, "move_to_minor failed!"
        return state

    def _align_global_axes_experimental_all_to_all(self, perm=None):
        """
        Align global concrete axes to virtual axes and return a new
        AbstractShardedProbabilityFunction `state` with
        state.global_axes == [0, 1, ..., num_global_discretes-1]. This routine
        uses an experimental version of jax.lax.all_to_all which currently
        is not supported on CPU hardware. This routine has a potentially smaller
        memory footprint and less restrictions on the number of free axes than
        the default implementation of align_axes().

        To enable this feature, set the environment variable
        PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL=1.

        Args:
          perm: An optional permutation to align the concrete axes to.

        Returns:
          AbstractShardedProbabilityFunction - with identity perm value for the global
            discretes.

        Raises:
          ValueError: len(perm) != state.num_global_discretes
          AssertionError: If the routine fails to bring global axes into `perm`.
        """
        state = self
        if perm is None:
            global_axes = tuple(range(state.num_global_discretes))
        else:
            if len(perm) != state.num_global_discretes:
                raise ValueError(
                    f"len(perm) = {len(perm)} is different from"
                    f" number of global discretes = {state.num_global_discretes}."
                )
            global_axes = tuple(perm)

        if not global_equals_collectables(state, global_axes):
            # repeat until all virtual global axes are in the concrete global
            # axes
            while not all_collectables_in_global(state, global_axes):
                # move as many virtual global axes into the concrete free axes
                # as possible by moving as many other axes into the non-free axes
                local_collectables = tuple(
                    a for a in state.local_axes if a in global_axes
                )
                local_other = tuple(
                    a for a in state.local_axes if a not in global_axes
                )
                to_major = local_other[: state.major_lane_num_discretes]
                local_other = local_other[state.major_lane_num_discretes :]
                # first move to major
                if len(to_major) < state.major_lane_num_discretes:
                    diff = state.major_lane_num_discretes - len(to_major)
                    to_major = to_major + local_collectables[:diff]
                    local_collectables = local_collectables[diff:]
                state = state.move_to_major(to_major)

                # if there are still some left, move to minor
                to_minor = local_other[: state.minor_lane_num_discretes]
                if len(to_minor) > 0:
                    if len(to_minor) < state.minor_lane_num_discretes:
                        diff = state.minor_lane_num_discretes - len(to_minor)
                        to_minor = to_minor + local_collectables[:diff]
                    state = state.move_to_minor(to_minor)

                # now move all virtual global axes in the concrete free axes
                # to the concrete global axes
                target_axes = utils.to_index(global_axes, state.free_axes)
                global_other = [
                    a for a in state.global_axes if a not in global_axes
                ]
                source_axes = utils.to_index(
                    global_other[: len(target_axes)], state.global_axes
                )
                state = state.swap_global_axes(source_axes, target_axes)

            assert all_collectables_in_global(state, global_axes)
            # finally, we only need to reorder the global axes and we're done
            if not global_equals_collectables(state, global_axes):
                perm = tuple([state.global_axes.index(l) for l in global_axes])
                state = state.pshuffle(perm)
                assert state.global_axes == global_axes
                return state
        assert state.global_axes == global_axes
        return state

    def _align_global_axes(self, perm=None):
        """
        Align global axes to virtual axes and return a new
        AbstractShardedProbabilityFunction `state` with
        state.global_axes == [0, 1, ..., num_global_discretes-1].
        This method uses standard jax.lax.pswapaxes calls to swap all
        global discretes with local ones. The method requires
        num_global_discretes < free_num_discretes and num_global_discretes <= 11.

        Args:
          perm: An optional permutation to align the concrete axes to.

        Returns:
          AbstractShardedProbabilityFunction - with identity perm value for the global
            discretes.

        Raises:
          ValueError: If `free_num_discretes` <= `num_global_discretes`.
          ValueError: If `num_global_discretes` > 11.
          ValueError: if len(perm) != num_global_discretes.
        """
        if self.free_num_discretes <= self.num_global_discretes:
            raise ValueError(
                f"Number of free discretes = {self.free_num_discretes} "
                f"is smaller of equal to the number of global "
                f"discretes = {self.num_global_discretes}"
            )
        if self.num_global_discretes > MAX_GLOBAL_DISCRETEDS:
            raise ValueError(
                f"Number of global discretes = {self.num_global_discretes}"
                f"is larger than {MAX_GLOBAL_DISCRETEDS}. This is currently"
                f" not supported."
            )
        state = self
        if perm is None:
            global_axes = tuple(range(state.num_global_discretes))
        else:
            if len(perm) != state.num_global_discretes:
                raise ValueError(
                    f"len(perm) = {len(perm)} is different from"
                    f" number of global discretes = {state.num_global_discretes}."
                )
            global_axes = tuple(perm)

        if not global_equals_collectables(state, global_axes):
            if any_collectables_in_global(
                state, global_axes
            ) and not all_collectables_in_global(state, global_axes):

                if (
                    len(set(state.free_axes) - set(global_axes))
                    >= state.num_global_discretes
                ):
                    # in this case we can localize all virtual global axes with a single pswap
                    free_collectables = tuple(
                        a for a in state.free_axes if a in global_axes
                    )
                    state = collect_move_right(state, free_collectables)
                    state = state.swap_global_axes()
                else:
                    # For up to (including) 11 global discretes, the number of virtual global axes
                    # in the local concrete axes at this point can at most be 10.
                    # We now move all these virtual global axes into the minor and major axes
                    # and then swap the global ones
                    local_collectables = tuple(
                        a for a in global_axes if a in state.local_axes
                    )
                    local_other = tuple(
                        a for a in state.local_axes if a not in global_axes
                    )
                    to_non_free = local_collectables[
                        : state.non_free_num_discretes
                    ]
                    if len(to_non_free) < state.non_free_num_discretes:
                        diff = state.non_free_num_discretes - len(to_non_free)
                        to_non_free = to_non_free + local_other[:diff]
                    state = state.move_to_non_free(to_non_free)
                    # for up to (including) 11 global discretes, the free_axes
                    # at this point do not contain any virtual global axes.
                    # Hence we can localize all virtual global axes with a single
                    # swap.
                    state = state.swap_global_axes()

            # at this point the global concrete axes contain all global virtual axes
            # in correct order or the global concrete axes contain no global
            # virtual axes at all.
            assert global_equals_collectables(
                state, global_axes
            ) or not any_collectables_in_global(state, global_axes)
            # fill non-free axes with axes values different from the virtual
            # global axes
            collectables = tuple(
                [a for a in state.local_axes if a not in global_axes]
            )[: state.non_free_num_discretes]
            state = state.move_to_non_free(collectables)

            # finally collect all virtual global axes and move them to concrete
            # global axes
            state = state.move_to_left(global_axes)
            state = state.swap_global_axes()
        assert state.global_axes == global_axes
        return state

    def align_global_axes(self, perm=None):
        """
        Align global axes to virtual axes and return a new
        AbstractShardedProbabilityFunction `state` with
        state.global_axes == [0, 1, ..., num_global_discretes-1]

        If the environment variable PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL
        is not set, the method uses standard jax.lax.pswapaxes calls to swap
        all global discretes with local ones. It is only guaranteed to work if
        num_global_discretes < free_num_discretes and num_global_discretes <= 11.

        If the environment variable PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL=1
        the method uses jax.lax.all_to_all with the `axis_index_groups`
        argument to swap individual global with local discrete-axes as needed.
        This has a potentially smaller memory footprint and less restrictions
        on the number of free axes than the default implementation.

        Args:
          perm: An optional permutation to align the concrete axes to.

        Returns:
          AbstractShardedProbabilityFunction - with identity perm value for the global
            discretes.

        Raises:
          ValueError: If `free_num_discretes` <= `num_global_discretes`.
          ValueError: If `num_global_discretes` > 11.
          ValueError: if len(perm) != num_global_discretes.
        """

        if ENABLE_ALL_TO_ALL:
            return self._align_global_axes_experimental_all_to_all(perm)
        return self._align_global_axes(perm)

    def align_local_axes(self, perm=None):
        """
        Align local axes to virtual axes/labels and return a new
        AbstractShardedProbabilityFunction with
        state.local_axes == [num_global_discretes, num_global_discretes+1, ..., num_discretes - 1]
        This method requires free_num_discretes > 7.

        Args:
          perm: An optional permutation to align the concrete axes to.

        Returns:
          AbstractShardedProbabilityFunction - with identity perm value for the global
            discretes.

        Raises:
            ValueError: If `free_num_discretes` <= 7.
            ValueError: If `perm` is not a valid permutation of the local axes.
        """
        if perm is None:
            local_axes = tuple(range(self.num_global_discretes, self.num_discretes))
        else:
            local_axes = tuple(perm)
        if set(self.local_axes) != set(local_axes):
            raise ValueError(
                f"perm = {perm} is not a valid permutation"
                f" of the local axes {self.local_axes}"
            )
        state = self
        free_axes = local_axes[: -state.non_free_num_discretes]
        non_free_axes = local_axes[-state.non_free_num_discretes :]

        state = state.move_to_non_free(non_free_axes)
        state = state.move_to_left(free_axes)
        return state

    def align_axes(self, perm=None):
        """
        Align concrete axes to virtual axes/labels and return a new
        AbstractShardedProbabilityFunction where perm == [0, 1, 2, 3, ...].
        This method requires free_num_discretes > max(num_global_discretes, 7)

        Args:
          perm: An optional permutation to align the concrete axes to.

        Returns:
          AbstractShardedProbabilityFunction: The result of aligning the axes
            to `perm`.

        Raises:
          ValueError: If `free_num_discretes` <= 7.
          ValueError: If `free_num_discretes` <= num_global_discretes.
          ValueError: If `num_global_discretes` > 11.
          ValueError: If `perm` is not a valid permutation of the array-axes.

        """
        if perm is None:
            perm = tuple(range(self.num_discretes))
        else:
            perm = tuple(perm)
        global_perm = perm[: self.num_global_discretes]
        local_perm = perm[self.num_global_discretes :]
        state = self.align_global_axes(global_perm)
        state = state.align_local_axes(local_perm)
        assert state.perm == perm
        return state

    def pshuffle(self, perm):
        """
        Perform a global pshuffle operation to bring
        the concrete global discrete axes into a new order
        given by `perm`.

        Args:
          perm: Tuple of ints denoting the new permutation
            of the global discretes.

        Returns:
          AbstractShardedProbabilityFunction: The result of the operation.

        Raises:
          ValueError: If len(perm) != num_global_discretes
        """
        if len(perm) != self.num_global_discretes:
            raise ValueError(
                f"Invalid argument perm = {perm} for "
                f"state with {self.num_global_discretes} discretes."
            )
        grid = np.arange(2 ** self.num_global_discretes).reshape(
            (2,) * self.num_global_discretes
        )
        linear_perm = grid.transpose(perm).ravel()
        if not isinstance(self.concrete_tensor, debug_array.DebugArray):
            shuffled_real = jax.lax.pshuffle(
                self.concrete_tensor.real, axis_name=AXIS_NAME, perm=linear_perm
            )
            shuffled_imag = jax.lax.pshuffle(
                self.concrete_tensor.imag, axis_name=AXIS_NAME, perm=linear_perm
            )
            shuffled = cw.ComplexDeviceArray(shuffled_real, shuffled_imag)
        else:
            shuffled = self.concrete_tensor

        global_axes = tuple(self.global_axes[p] for p in perm)
        new_perm = global_axes + self.local_axes
        return self.__class__(shuffled, new_perm, self.num_global_discretes)

    def swap_global_axes(self, source_axes=None, target_axes=None):
        """
        Swap the global axes `source_axes` with the local axes `target_axes`.
        Counting of axes always starts at 0 for the given axes type, i.e.
        `target_axes = (1, 2)` targets the local axes 1 and 2 of the local concrete
        array.

        `source_axes` and `target_axes` have to either both be `None`, or a sequence
        of integers of identical length. If they are `None`, the method swaps the
        global axes with the first `self.num_global_discretes` local axes of the
        `concrete_tensor`.

        Args:
          source_axes: Optional sequence of integers denoting
            the global (distributed) axes that should be swapped
            with the local axes `target_axes`.
          target_axes: The local axes that should become global
            axes, in the given order.

        Returns:
          AbstractShardedProbabilityFunction: The new probability function after swapping
            axes.

        Raises:
          ValueError: if source_axes or target_axes are passed without setting
            the environment variable PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL=1
          ValueError: If source_axes or target_axes is not None,
            PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL=1, but the operation is run on
            a CPU device.
          ValueError:L If len(source_axes) != len(target_axes)
        """

        # NOTE: For `source_axes=None`, `target_axes=None` the method
        # reduces to swapping the global axes with the first
        # num_global_discretes local axes. In this case the method works
        # correctly on CPU and ASIC.
        # If `source_axes` and `target_axes` are sequences of int, then
        # for certain values of these sequences there is an XLA bug on
        # CPU which causes the result to be wrong,
        # see https://github.com/google/jax/issues/5861.
        grid = (2,) * self.num_global_discretes
        if source_axes is not None or target_axes is not None:
            if not ENABLE_ALL_TO_ALL:
                raise ValueError(
                    "source_axes or target_axes are different from "
                    "None, but distla_enable_experimental_all_to_all "
                    "is not set. Set environment variable "
                    "distla_enable_experimental_all_to_all=1 to enable"
                    " this feature."
                )
            if not isinstance(self.concrete_tensor, debug_array.DebugArray):
                if JAX_BACKEND != "asic":
                    raise NotImplementedError(
                        "source_axes or target_axes different from "
                        "None are currently only supported on ASIC due"
                        " to an XLA bug on CPU "
                        "(see https://github.com/google/jax/issues/5861.)."
                    )

        if source_axes is None:
            source_axes = tuple(range(self.num_global_discretes))

        if target_axes is None:
            target_axes = tuple(range(self.num_global_discretes))
        if len(source_axes) != len(target_axes):
            raise ValueError(
                "source_axes and target_axes must have the same length"
            )

        # NOTE : we handle real and imag part separatly here instead of in
        # complex_workaround.py because the interface of jax_wrappers.all_to_all
        # differs from jax.lax.all_to_all. complex_workaround.py should precisely
        # copy jax's API.
        if not isinstance(self.concrete_tensor, debug_array.DebugArray):
            swapped_real = jax_wrappers.all_to_all(
                self.concrete_tensor.real,
                sharded_axes=source_axes,
                split_axes=target_axes,
                concat_axes=target_axes,
                grid_shape=grid,
            )

            swapped_imag = jax_wrappers.all_to_all(
                self.concrete_tensor.imag,
                sharded_axes=source_axes,
                split_axes=target_axes,
                concat_axes=target_axes,
                grid_shape=grid,
            )

            swapped = cw.ComplexDeviceArray(swapped_real, swapped_imag)
        else:
            swapped = self.concrete_tensor
        global_axes = list(self.global_axes)
        free_axes = list(self.free_axes)
        for s, t in zip(source_axes, target_axes):
            global_axes[s] = self.free_axes[t]
            free_axes[t] = self.global_axes[s]
        new_perm = tuple(global_axes) + tuple(free_axes) + self.non_free_axes
        return self.__class__(swapped, new_perm, self.num_global_discretes)

    def _dot_helper(self, matrix, axes, target_ndiscretes=7):
        """
        Helper function for performing a matrix dot on the given virtual axes.

        Args:
          matrix: A tuple of A (2**n, 2**n) float32 matrices where n is len(axes).
          axes: Which virtual axes to dot with on this ShardedProbabilityFunction.
          target_ndiscretes: An optional integer; `matrix` will be extended to a
            (2**target_ndiscretes, 2**target_ndiscretes) shaped matrix to increase
            performance of tensor contractions on ASIC.
        Returns:
          ComplexDeviceArray: The result of dotting `matrix` on
            self.concrete_tensor.
          Tuple[int]: The `perm` attribute immediately before contracting
            `matrix` with `self`.
        """
        shape = matrix.shape
        assert len(axes) <= target_ndiscretes
        assert len(shape) == 2
        assert shape[0] == 2 ** len(axes)
        assert shape[1] == 2 ** len(axes)
        state = self
        if not set(axes).isdisjoint(set(state.global_axes)):
            if ENABLE_ALL_TO_ALL:
                source_axes = utils.to_index(
                    targets=axes, labels=state.global_axes
                )
                unconctracted_free_axes = [
                    a for a in state.free_axes if a not in axes
                ]
                assert len(source_axes) <= len(unconctracted_free_axes), (
                    f"Number of uncontracted free axes ({len(unconctracted_free_axes)})"
                    f" is less than the number "
                    f"required to localize {len(source_axes)} contracted global axes."
                )
                # we swap the first len(source_axes) local uncontracted axes
                # with source_axes
                uncontracted_free_axes = []
                for a in state.free_axes:
                    if a not in axes:
                        uncontracted_free_axes.append(a)
                    if len(uncontracted_free_axes) >= len(source_axes):
                        break

                if len(uncontracted_free_axes) < len(source_axes):
                    raise ValueError(
                        "number of moveable free axes is smaller than"
                        " source_axes. Can't perform axes swap."
                    )
                target_axes = utils.to_index(
                    targets=uncontracted_free_axes, labels=state.free_axes
                )
                state = state.swap_global_axes(source_axes, target_axes)
            else:
                moveable_axes = [x for x in axes if x in state.free_axes]
                num_swapable = len(state.free_axes) - len(moveable_axes)
                assert num_swapable >= self.num_global_discretes, (
                    f"Number of non-contracting free axes less than number of global "
                    f"axes.\nNum non-contracting free: {num_swapable}\n"
                    f"Number of global: {self.num_global_discretes}"
                )
                state = state.move_to_right(moveable_axes)
                state = state.swap_global_axes()

        # Sometimes we need to do this twice if a contracting axis falls in the far
        # right 3 axes of the non_free_axes.
        if not set(axes).isdisjoint(set(state.minor_axes)):
            new_minor = tuple(
                [a for a in state.local_axes if a not in axes][:3]
            )
            state = state.move_to_minor(new_minor)
        if not set(axes).isdisjoint(set(state.major_axes)):
            new_major = tuple(
                [
                    a
                    for a in state.free_axes + state.major_axes
                    if a not in axes
                ][:7]
            )

            state = state.move_to_major(new_major)

        state = state.move_to_left(axes)
        # we extend `matrix` to a (2**target_ndiscrete,2**target_ndiscrete) shaped
        # matrix to increase performance.
        diff_axes = target_ndiscretes - len(axes)
        if not isinstance(self.concrete_tensor, debug_array.DebugArray):
            matrix = cw.kron(matrix, cw.eye(2 ** diff_axes, dtype=matrix.dtype))
            new_tensor = state.concrete_tensor.reshape(
                (2 ** target_ndiscretes,)
                + (2,) * (state.num_local_discretes - 10 - target_ndiscretes)
                + (8, 128)
            )
            new_tensor = cw.tensordot(
                matrix,
                new_tensor,
                [[1], [0]],
                precision=jax.lax.Precision.HIGHEST,
            )
            new_tensor = new_tensor.reshape(
                (2,) * (self.num_local_discretes - 10) + (8, 128)
            )
        else:
            new_tensor = state.concrete_tensor
        return new_tensor, state.perm

    def transpose(self, perm):
        raise NotImplementedError(
            "method `transpose` is not implemented "
            "in AbstractShardedProbabilityFunction"
        )

    def block_until_ready(self):
        self.concrete_tensor.block_until_ready()
        return self

    def _distributed_sampling(self, repetitions, prng_key):
        """
        Compute `repetitions` samples from the probability distribution
        given by the absolute square of the probability function, for a distributed
        probability function. This functions assumes that it is called inside a pmap.

        Args:
          repetitions: Desired number of samples.
          prng_key: A jax.random.PRNGKey for randomization. Note that all
            processes require the same key.

        Returns:
          Array of int: The global part of the samples, encoded into uint32.
          Array of int: The local part of the samples, encoded into uint32.
        """
        probability = (
            self.concrete_tensor.real ** 2 + self.concrete_tensor.imag ** 2
        )
        return distributed_sampling.sample(prng_key, probability, repetitions)

    def _single_core_sampling(self, repetitions, prng_key):
        """
        Compute `repetitions` samples from the probability distribution
        given by the absolute square of the probability function, for an UNDISTRIBUTED
        probability function.

        Args:
          repetitions: Desired number of samples.
          prng_key: A jax.random.PRNGKey for randomization

        Returns:
          Array of int: The global part of the samples, encoded into uint32.
            Since this routine assumes that the probabilityfunction is not distributed,
            this is just an array of zeros.
          Array of int: The local part of the samples, encoded into uint32.
        """
        probability = (
            self.concrete_tensor.real ** 2 + self.concrete_tensor.imag ** 2
        )
        return distributed_sampling.single_core_sample(
            prng_key, probability, repetitions
        )

    def sample(self, repetitions, prng_key):
        """
        Compute `repetitions` samples from the probability distribution
        given by the absolute square of the probability function. If the wave
        function is distributed, then this function should be called inside
        a `jax.pmap`.

        Args:
          repetitions: Desired number of samples.
          seed: A seed for randomization.

        Returns:
          Array of int: The global part of the samples, encoded into uint32.
            If the probabilityfunction is not distributed, this is just an array of zeros.
          Array of int: The local part of the samples, encoded into uint32.
        """

        if self.distributed:
            return self._distributed_sampling(repetitions, prng_key)
        return self._single_core_sampling(repetitions, prng_key)

    def tree_flatten(self):
        return (self.concrete_tensor,), (self.perm, self.num_global_discretes)

    @classmethod
    def tree_unflatten(cls, static_data, children):
        return cls(*children, *static_data)


####################   helper functions   #############################
def all_collectables_in_global(state, collectables):
    """Return a bool describing if concrete global axes contain all
    `collectables`.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool

    """
    return set(state.global_axes) == set(collectables)


def any_collectables_in_global(state, collectables):
    """Return a bool describing if concrete global axes contain any
    `collectables`.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return not set(state.global_axes).isdisjoint(set(collectables))


def any_collectables_in_non_free(state, collectables):
    """Return a bool describing if non-free axes contain
    any `collectables`.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return not set(state.non_free_axes).isdisjoint(set(collectables))


def any_collectables_in_minor(state, collectables):
    """Return a bool describing if minor axes contain
    any `collectables`.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return not set(state.minor_axes).isdisjoint(set(collectables))


def any_collectables_in_major(state, collectables):
    """Return a bool describing if major axes contain
    any `collectables`.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return not set(state.major_axes).isdisjoint(set(collectables))


def all_collectables_in_major(state, collectables):
    """Return a bool describing if all collectables are in
    the major axes.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return set(state.major_axes) == set(collectables)


def all_collectables_in_minor(state, collectables):
    """Return a bool describing if all collectables are in
    the minor axes.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return set(state.minor_axes) == set(collectables)


def global_equals_collectables(state, collectables):
    """Return a bool describing if concrete global axes contain all
    `collectables` in the given order.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return tuple(state.global_axes) == tuple(collectables)


def num_collectables_in_major(state, collectables):
    """Return a bool describing if the state contains the virtual axes values
    in collectables in the concrete major axes.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return len(set(collectables) & set(state.major_axes))


def num_collectables_in_minor(state, collectables):
    """Return a bool describing if the state contains the virtual axes values
    in collectables in the concrete minor axes.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return len(set(collectables) & set(state.minor_axes))


def num_collectables_in_free(state, collectables):
    """Return a bool describing if the state contains the virtual axes values
    in collectables in the free axes.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      bool
    """
    return len(set(collectables) & set(state.free_axes))


def collect_move_right(state, collectables):
    """Collect virtual axes values from collectables that are in the concrete
    free axes. Move these virtual axes values to the right of the concrete
    free axes and return a new state.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      A new AbstractShardedProbabilityFunction.
    """
    movable_axes = [x for x in collectables if x in state.free_axes]
    return state.move_to_right(movable_axes)


def collect_move_left(state, collectables):
    """Collect virtual axes values from collectables that are in the concrete
    free axes. Move these virtual axes values to the right of the concrete
    free axes and return a new state.

    Args:
      state: AbstractShardedProbabilityFunction.
      collectables: List of virtual axes values.

    Returns:
      A new AbstractShardedProbabilityFunction.
    """
    movable_axes = [x for x in collectables if x in state.free_axes]
    return state.move_to_left(movable_axes)
