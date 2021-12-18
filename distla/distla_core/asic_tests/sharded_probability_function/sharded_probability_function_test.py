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
"""Tests forShardedProbabilityFunction"""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import random
import unittest

from asic_la.sharded_probability_function import complex_workaround as cw
from asic_la.sharded_probability_function import sharded_probability_function


class ShardedProbabilityFunctionTest(unittest.TestCase):
    def test_dot_free_only(self):
        np.random.seed(3)
        val = np.random.uniform(size=(2,) * 10 + (8, 128)) + 1.0
        building_block = np.random.uniform(size=(128, 128)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        axes = [2, 4, 1, 0, 5, 3, 6]
        new_state = state.dot(building_block, axes)
        expected = np.tensordot(
            val,
            building_block.reshape((2,) * 14),
            [[2, 4, 1, 0, 5, 3, 6], list(range(len(axes), len(axes) * 2))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dot_single_left_perm(self):
        np.random.seed(10)
        val = np.random.uniform(size=(2,) * 10 + (8, 128)) + 1.0
        building_block = np.random.uniform(size=(2 ** 2, 2 ** 2)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        axes = [2, 15]
        new_state = state.dot(building_block, axes)
        expected = np.tensordot(
            val.reshape((2,) * 20),
            building_block.reshape((2,) * 4),
            [[2, 15], list(range(len(axes), 2 * len(axes)))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dot_double_left_perm(self):
        np.random.seed(11)
        val = np.random.uniform(size=(2,) * 10 + (8, 128)) + 1.0
        building_block = np.random.uniform(size=(2 ** 3, 2 ** 3)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        axes = [2, 15, 19]
        new_state = state.dot(building_block, axes)
        expected = np.tensordot(
            val.reshape((2,) * 20),
            building_block.reshape((2,) * 6),
            [[2, 15, 19], list(range(len(axes), 2 * len(axes)))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_dot_double_left_perm_as_tuple(self):
        np.random.seed(11)
        val = np.random.uniform(size=(2,) * 10 + (8, 128)) + 1.0
        building_block = np.random.uniform(size=(2 ** 3, 2 ** 3)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        axes = [2, 15, 19]
        new_state = state.dot(cw.ComplexDeviceArray(building_block.real, building_block.imag), axes)
        expected = np.tensordot(
            val.reshape((2,) * 20),
            building_block.reshape((2,) * 6),
            [[2, 15, 19], list(range(len(axes), 2 * len(axes)))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_move_contractable_axes(self):
        np.random.seed(11)
        val = np.random.uniform(size=(2,) * 10 + (8, 128)) + 1.0
        building_block = np.random.uniform(size=(2 ** 7, 2 ** 7)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        axes = [0, 1, 2, 3, 4, 5, 19]
        new_state = state.dot(cw.ComplexDeviceArray(building_block.real, building_block.imag), axes)
        expected = np.tensordot(
            val.reshape((2,) * 20),
            building_block.reshape((2,) * 14),
            [axes, list(range(len(axes), 2 * len(axes)))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_scrambled_double_left_perm(self):
        np.random.seed(0)
        perm = [
            11,
            0,
            8,
            5,
            9,
            4,
            6,
            17,
            18,
            2,
            14,
            19,
            7,
            1,
            13,
            16,
            12,
            3,
            15,
            10,
        ]
        val = np.random.uniform(size=(2,) * 20) + 1.0
        tmp_val = np.transpose(val, perm).reshape((2,) * 10 + (8, 128))
        building_block = np.random.uniform(size=(2 ** 3, 2 ** 3)) + 1.0
        state = sharded_probability_function.ShardedProbabilityFunction(
            tmp_val, perm=perm, num_global_discretes=0
        )
        axes = [2, 15, 19]
        new_state = state.dot(building_block, axes)
        expected = np.tensordot(
            val.reshape((2,) * 20),
            building_block.reshape((2,) * 6),
            [axes, list(range(len(axes), 2 * len(axes)))],
        )
        np.testing.assert_allclose(
            expected.reshape((2,) * 20),
            new_state._testing_correctly_permuted_tensor,
            atol=1e-6,
            rtol=1e-6,
        )


def test_pytree_registry():
    l = 12
    a = np.random.normal(size=(2,) * l).reshape((2,) * (l - 10) + (8, 128))
    A = sharded_probability_function.ShardedProbabilityFunction(a, list(range(l)), False)

    @jax.jit
    def add(U, V):
        return U + V

    B = add(A, A)
    np.testing.assert_allclose(
        B.concrete_tensor.real, 2 * A.concrete_tensor.real
    )
