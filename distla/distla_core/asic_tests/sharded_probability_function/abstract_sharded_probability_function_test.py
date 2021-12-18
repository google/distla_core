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
"""Tests for AbstractShardedProbabilityFunction"""
import functools
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import os
import operator
import pytest
import random
import unittest
from asic_la.sharded_probability_function import jax_wrappers

asic_only = pytest.mark.skipif(
    jax.config.FLAGS.jax_xla_backend != "asic_driver", reason="only on ASICs"
)
AXIS_NAME = jax_wrappers.AXIS_NAME
from asic_la.sharded_probability_function import (
    abstract_sharded_probability_function as aswf,
)
from asic_la.sharded_probability_function import debug_array


class AbstractShardedProbabilityFunctionTest(unittest.TestCase):
    def test_zerostate_sanity(self):
        state = aswf.AbstractShardedProbabilityFunction.zero_state(23, False)
        self.assertEqual(state.concrete_tensor.shape, (2,) * 13 + (8, 128))
        tmp = state.concrete_tensor.reshape((-1,))
        self.assertEqual(tmp.real[0], 1.0)
        self.assertTrue((tmp.real[1:] == jnp.zeros_like(tmp.real[1:])).all())
        self.assertTrue((tmp.imag == jnp.zeros_like(tmp.imag)).all())

    def test_axes_types(self):
        state = aswf.AbstractShardedProbabilityFunction(
            jnp.ones((2,) * 3 + (8, 128)),
            perm=[2, 9, 6, 11, 7, 5, 10, 12, 8, 0, 3, 1, 4],
            num_global_discretes=0,
        )
        self.assertEqual(state.global_axes, ())
        self.assertEqual(state.free_axes, (2, 9, 6))
        self.assertEqual(state.minor_axes, (11, 7, 5))
        self.assertEqual(state.major_axes, (10, 12, 8, 0, 3, 1, 4))
        self.assertEqual(state.non_free_axes, (11, 7, 5, 10, 12, 8, 0, 3, 1, 4))

    def test_move_to_left(self):
        np.random.seed(0)
        val = np.random.normal(size=(2,) * 3 + (8, 128))
        state = aswf.AbstractShardedProbabilityFunction(
            val,
            perm=[2, 9, 6, 11, 7, 5, 10, 12, 8, 0, 3, 1, 4],
            num_global_discretes=0,
        )
        state = state.move_to_left(
            [
                6,
            ]
        )
        self.assertEqual(state.perm, (6, 2, 9, 11, 7, 5, 10, 12, 8, 0, 3, 1, 4))
        expected = jnp.moveaxis(val, 2, 0)
        self.assertTrue((state.concrete_tensor.real == expected).all())

    def test_move_to_right(self):
        np.random.seed(0)
        val = np.random.normal(size=(2,) * 3 + (8, 128))
        state = aswf.AbstractShardedProbabilityFunction(
            val,
            perm=[2, 9, 6, 11, 7, 5, 10, 12, 8, 0, 3, 1, 4],
            num_global_discretes=0,
        )
        state = state.move_to_right(
            [
                9,
            ]
        )
        self.assertEqual(state.perm, (2, 6, 9, 11, 7, 5, 10, 12, 8, 0, 3, 1, 4))
        expected = jnp.moveaxis(val, 1, 2)
        self.assertTrue((state.concrete_tensor.real == expected).all())

    def test_left_cycle_perm(self):
        np.random.seed(0)
        val = np.random.normal(size=(2,) * 10 + (8, 128))
        state = aswf.AbstractShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        state = state.left_cycle_perm()
        self.assertEqual(state.perm, tuple(range(7, 20)) + tuple(range(7)))
        val = jnp.reshape(val, (128,) + (2, 2, 2, 8, 128))
        expected = jnp.moveaxis(val, 0, -1).reshape((2,) * 10 + (8, 128))
        self.assertTrue((state.concrete_tensor.real == expected).all())

    def test_minor_left_cycle_perm(self):
        np.random.seed(0)
        val = np.random.normal(size=(2,) * 10 + (8, 128))
        state = aswf.AbstractShardedProbabilityFunction(
            val, perm=list(range(20)), num_global_discretes=0
        )
        state = state.minor_left_cycle_perm()
        self.assertEqual(
            state.perm,
            tuple(range(3, 13)) + tuple(range(3)) + tuple(range(13, 20)),
        )
        val = jnp.reshape(val, (8,) + (2, 2, 2, 2, 2, 2, 2, 8, 128))
        expected = jnp.moveaxis(val, 0, -2).reshape((2,) * 10 + (8, 128))
        self.assertTrue((state.concrete_tensor.real == expected).all())

    @pytest.mark.skipif(
        jax.device_count() != 8, reason="This test requires a asic_node"
    )
    def test_swap_global(self):
        np.random.seed(0)
        assert jax.device_count() == 8
        l = 21
        perm = list(range(l))
        perm = perm[3:6] + perm[:3] + perm[6:]
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape(
            (8,) + (2,) * (l - 13) + (8, 128)
        )

        @functools.partial(jax.pmap, axis_name="i")
        def f(val):
            state = aswf.AbstractShardedProbabilityFunction(val, perm=perm)
            state = state.swap_global_axes()
            return state.concrete_tensor.real

        out = f(tmp_val)
        np.testing.assert_allclose(val, out.reshape((2,) * l))

    def test_align_axes(self):
        random.seed(0)
        np.random.seed(0)
        l = 23
        perm = random.sample(range(l), l)
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape((2,) * (l - 10) + (8, 128))
        state = aswf.AbstractShardedProbabilityFunction(
            tmp_val, perm=perm, num_global_discretes=0
        )
        state = state.align_axes()
        self.assertEqual(state.perm, tuple(range(l)))
        np.testing.assert_allclose(
            val, state._testing_correctly_permuted_tensor
        )
        t = jnp.reshape(state.concrete_tensor.real, ((2,) * l))
        np.testing.assert_allclose(val, t)

    def test_align_axes_with_perm(self):
        random.seed(0)
        np.random.seed(0)
        l = 23
        perm = tuple(random.sample(range(l), l))
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape((2,) * (l - 10) + (8, 128))
        val = val.reshape((2,) * (l - 10) + (8, 128))

        state = aswf.AbstractShardedProbabilityFunction(
            val, perm=tuple(range(l)), num_global_discretes=0
        )
        state = state.align_axes(perm)
        self.assertEqual(state.perm, perm)
        t = jnp.reshape(state.concrete_tensor.real, ((2,) * l))
        np.testing.assert_allclose(tmp_val.reshape((2,) * l), t)

    @pytest.mark.skipif(
        jax.device_count() != 8, reason="This test requires a asic_node"
    )
    def test_distributed_align_axes(self):
        np.random.seed(0)
        assert jax.device_count() == 8
        l = 24
        perm = list(range(l))
        perm = perm[:3] + list(reversed(perm[3:]))
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape(
            (8,) + (2,) * (l - 13) + (8, 128)
        )

        @functools.partial(jax.pmap, axis_name="i")
        def f(val):
            state = aswf.AbstractShardedProbabilityFunction(val, perm=perm)
            state = state.align_axes()
            return state.concrete_tensor.real

        out = f(tmp_val)
        self.assertTrue(jnp.allclose(out, val.reshape(out.shape)))

    @pytest.mark.skipif(
        jax.device_count() != 8, reason="This test requires a asic_node"
    )
    def test_distributed_align_axes_with_perm(self):
        np.random.seed(0)
        assert jax.device_count() == 8
        l = 24
        perm = tuple(random.sample(range(l), l))
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape(
            (8,) + (2,) * (l - 13) + (8, 128)
        )
        val = val.reshape((8,) + (2,) * (l - 13) + (8, 128))

        @functools.partial(jax.pmap, axis_name="i")
        def f(val):
            state = aswf.AbstractShardedProbabilityFunction(val, perm=tuple(range(l)))
            state = state.align_axes(perm)
            return state.concrete_tensor.real

        out = f(val)
        self.assertTrue(
            jnp.allclose(out.reshape((2,) * l), tmp_val.reshape((2,) * l))
        )

    @pytest.mark.skipif(
        jax.device_count() != 8, reason="This test requires a asic_node"
    )
    def test_global_align_axes(self):
        np.random.seed(0)
        assert jax.device_count() == 8
        l = 24
        perm = list(reversed(range(l)))
        perm = perm[-1:] + perm[:-1]
        val = np.random.normal(size=(2,) * l)
        tmp_val = np.transpose(val, perm).reshape(
            (8,) + (2,) * (l - 13) + (8, 128)
        )

        @functools.partial(jax.pmap, axis_name="i")
        def f(val):
            state = aswf.AbstractShardedProbabilityFunction(val, perm=perm)
            state = state.align_axes()
            return state.concrete_tensor.real

        out = f(tmp_val)
        self.assertTrue(jnp.allclose(out, val.reshape(out.shape)))

    def test_basic_mul(self):
        np.random.seed(0)
        l = 20
        a = np.random.normal(size=(2,) * l).reshape((2,) * (l - 10) + (8, 128))
        SWF = aswf.AbstractShardedProbabilityFunction(a, list(range(l)), False)
        np.testing.assert_allclose(
            (SWF * 5).concrete_tensor.real, (SWF.concrete_tensor.real) * 5
        )
        np.testing.assert_allclose(
            (SWF * 5).concrete_tensor.imag, (SWF.concrete_tensor.imag) * 5
        )

        np.testing.assert_allclose(
            (5 * SWF).concrete_tensor.real, 5 * SWF.concrete_tensor.real
        )
        np.testing.assert_allclose(
            (5 * SWF).concrete_tensor.imag, 5 * SWF.concrete_tensor.imag
        )

    def test_jit_mul(self):
        np.random.seed(0)
        l = 15
        a = (
            np.random.normal(size=(2,) * l)
            + np.random.normal(size=(2,) * l) * 1j
        )
        inp = a.reshape((2,) * (l - 10) + (8, 128))

        @jax.jit
        def fn(ar, v):
            swf = aswf.AbstractShardedProbabilityFunction(ar, list(range(l)), False)
            swf = swf * v
            return swf.concrete_tensor

        for v in [2, 2.5, -2.5j, 2 + 2.5j]:
            b = fn(inp, v).reshape((2,) * l)
            assert a.shape == b.shape
            np.testing.assert_allclose((a * v).real, b.real, atol=1e-5)
            np.testing.assert_allclose((a * v).imag, b.imag, atol=1e-5)


def test_pytree_registry():
    l = 12
    a = np.random.normal(size=(2,) * l).reshape((2,) * (l - 10) + (8, 128))
    A = aswf.AbstractShardedProbabilityFunction(a, list(range(l)), False)

    @jax.jit
    def add(U, V):
        return U + V

    B = add(A, A)
    np.testing.assert_allclose(
        B.concrete_tensor.real, 2 * A.concrete_tensor.real
    )


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_basic_add_sub(op):
    np.random.seed(0)
    l = 18
    a = np.random.normal(size=(2,) * l).astype(np.float32)
    b = np.random.normal(size=(2,) * l).astype(np.float32)
    perm_a = tuple(np.random.choice(np.arange(l), l, replace=False))
    perm_b = tuple(np.random.choice(np.arange(l), l, replace=False))

    perm = tuple(perm_a.index(b) for b in perm_b)
    A = aswf.AbstractShardedProbabilityFunction(
        a.reshape((2,) * (l - 10) + (8, 128)), perm_a, num_global_discretes=0
    )
    B = aswf.AbstractShardedProbabilityFunction(
        b.reshape((2,) * (l - 10) + (8, 128)), perm_b, num_global_discretes=0
    )
    C = op(A, B)
    assert C.perm == B.perm
    np.testing.assert_allclose(
        op(np.transpose(a, perm), b), C.concrete_tensor.real.reshape((2,) * l)
    )


MAX_NUM_GLOBAL_DISCRETEDS = int(np.log2(jax.device_count()))


def flatten(ll):
    return [a for l in ll for a in l]


num_global_num_free = flatten(
    [
        list(
            itertools.product(
                [num_global],
                list(range(max(num_global, 7) + 1, max(num_global, 7) + 11)),
            )
        )
        for num_global in range(12)
    ]
)


@pytest.mark.parametrize("num_global_num_free", num_global_num_free)
@pytest.mark.parametrize("seed", range(10))
def test_align_global_axes_large(num_global_num_free, seed):
    random.seed(seed)
    num_global_discretes, num_free_discretes = num_global_num_free
    N = num_global_discretes + 10 + num_free_discretes
    shape = (2,) * num_global_discretes + (2,) * num_free_discretes + (8, 128)
    shape = tuple((s,) for s in shape)
    perm = tuple(random.sample(range(N), N))
    state = aswf.AbstractShardedProbabilityFunction(
        debug_array.DebugArray(jnp.zeros(1), shape),
        perm,
        num_global_discretes=num_global_discretes,
    )
    state = state.align_global_axes()
    assert state.global_axes == tuple(range(num_global_discretes))


@pytest.mark.parametrize("num_global_num_free", num_global_num_free)
@pytest.mark.parametrize("seed", range(10))
def test_align_axes_large(num_global_num_free, seed):
    random.seed(seed)
    num_global_discretes, num_free_discretes = num_global_num_free
    N = num_global_discretes + 10 + num_free_discretes
    shape = (2,) * num_global_discretes + (2,) * num_free_discretes + (8, 128)
    shape = tuple((s,) for s in shape)
    perm = tuple(random.sample(range(N), N))
    state = aswf.AbstractShardedProbabilityFunction(
        debug_array.DebugArray(jnp.zeros(1), shape),
        perm,
        num_global_discretes=num_global_discretes,
    )
    state = state.align_axes()
    assert state.perm == tuple(range(N))


@pytest.mark.skipif(
    jax.device_count() != 8, reason="This test requires a asic_node"
)
@pytest.mark.parametrize(
    "num_free_discretes", list(range(MAX_NUM_GLOBAL_DISCRETEDS + 1, 7))
)
def test_align_axes_local_raises(num_free_discretes):
    N = int(np.log2(jax.device_count())) + 10 + num_free_discretes
    num_global_discretes = int(np.log2(jax.device_count()))
    shape = (2,) * num_global_discretes + (2,) * num_free_discretes + (8, 128)
    shape = tuple((s,) for s in shape)
    perm = tuple(random.sample(range(N), N))
    with pytest.raises(
        ValueError, match=f"Number of discretes = {N} is too small"
    ):
        _ = jax.pmap(
            lambda x: aswf.AbstractShardedProbabilityFunction(
                debug_array.DebugArray(jnp.zeros(1), shape), perm
            ).align_axes(),
            axis_name="i",
        )(jnp.ones(jax.local_device_count()))


@pytest.mark.skipif(
    jax.device_count() != 8, reason="This test requires a asic_node"
)
@pytest.mark.parametrize("num_free_discretes", (8, 9, 10))
@pytest.mark.parametrize(
    "perm",
    list(
        itertools.permutations(tuple(range(int(np.log2(jax.device_count())))))
    ),
)
def test_pshuffle(num_free_discretes, perm):
    random.seed(0)
    N = int(np.log2(jax.device_count())) + num_free_discretes + 10
    shape = (jax.device_count(),) + (2,) * num_free_discretes + (8, 128)
    array = np.random.rand(*shape)
    state = jax.pmap(
        lambda x: aswf.AbstractShardedProbabilityFunction(x, tuple(range(N))).pshuffle(
            perm
        ),
        axis_name="i",
    )(array)
    assert state.global_axes == perm
    global_shape = (2, 2, 2) + (2,) * num_free_discretes + (2,) * 10
    actual = state.concrete_tensor.real.reshape(global_shape)
    expected = array.reshape(global_shape).transpose(perm + tuple(range(3, N)))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("num_global_num_free", num_global_num_free)
@pytest.mark.parametrize("seed", range(10))
def test_align_axes_given_perm_large(num_global_num_free, seed):
    num_global_discretes, num_free_discretes = num_global_num_free
    random.seed(seed)
    N = num_global_discretes + 10 + num_free_discretes
    shape = (2,) * num_global_discretes + (2,) * num_free_discretes + (8, 128)
    shape = tuple((s,) for s in shape)
    perm = tuple(random.sample(range(N), N))

    desired_perm = tuple(random.sample(range(N), N))
    state = aswf.AbstractShardedProbabilityFunction(
        debug_array.DebugArray(jnp.zeros(1), shape),
        perm,
        num_global_discretes=num_global_discretes,
    )
    state = state.align_axes(desired_perm)
    assert state.perm == desired_perm


@pytest.mark.parametrize("num_discretes", [21, 22, 23])
@pytest.mark.parametrize("seed", range(4))
def test_move_to_non_free(seed, num_discretes):
    np.random.seed(seed)
    val = np.random.normal(size=(2,) * (num_discretes - 10) + (8, 128))
    perm = np.arange(num_discretes)
    np.random.shuffle(perm)
    perm = tuple(perm)
    state = aswf.AbstractShardedProbabilityFunction(
        val, perm=perm, num_global_discretes=0
    )
    to_non_free = tuple(
        np.random.choice(np.arange(num_discretes), size=(10,), replace=False)
    )
    state = state.move_to_non_free(to_non_free)
    _perm = tuple(perm.index(a) for a in state.perm)
    assert state.non_free_axes == to_non_free
    expected = np.transpose(val.reshape((2,) * num_discretes), _perm)
    actual = state.concrete_tensor.real
    expected = np.reshape(expected, (2,) * (num_discretes - 10) + (8, 128))
    np.testing.assert_allclose(np.array(actual), expected)


@pytest.mark.parametrize("num_discretes", [21, 22, 23])
@pytest.mark.parametrize("seed", range(4))
def test_move_to_minor(seed, num_discretes):
    np.random.seed(seed)
    val = np.random.normal(size=(2,) * (num_discretes - 10) + (8, 128))
    perm = np.arange(num_discretes)
    np.random.shuffle(perm)
    perm = tuple(perm)
    state = aswf.AbstractShardedProbabilityFunction(
        val, perm=perm, num_global_discretes=0
    )
    to_minor = tuple(
        np.random.choice(np.arange(num_discretes), size=(3,), replace=False)
    )
    state = state.move_to_minor(to_minor)
    _perm = tuple(perm.index(a) for a in state.perm)
    state.minor_axes == to_minor
    expected = np.transpose(val.reshape((2,) * num_discretes), _perm)
    actual = state.concrete_tensor.real
    expected = np.reshape(expected, (2,) * (num_discretes - 10) + (8, 128))
    np.testing.assert_allclose(np.array(actual), expected)


@pytest.mark.parametrize("num_discretes", [21, 22, 23])
@pytest.mark.parametrize("seed", range(4))
def test_move_to_major(seed, num_discretes):
    np.random.seed(seed)
    val = np.random.normal(size=(2,) * (num_discretes - 10) + (8, 128))
    perm = np.arange(num_discretes)
    np.random.shuffle(perm)
    perm = tuple(perm)
    state = aswf.AbstractShardedProbabilityFunction(
        val, perm=perm, num_global_discretes=0
    )
    to_major = tuple(
        np.random.choice(np.arange(num_discretes), size=(7,), replace=False)
    )
    state = state.move_to_major(to_major)
    _perm = tuple(perm.index(a) for a in state.perm)
    state.major_axes == to_major
    expected = np.transpose(val.reshape((2,) * num_discretes), _perm)
    actual = state.concrete_tensor.real
    expected = np.reshape(expected, (2,) * (num_discretes - 10) + (8, 128))
    np.testing.assert_allclose(np.array(actual), expected)


@asic_only
@pytest.mark.parametrize("num_discretes", [21, 22, 23])
@pytest.mark.parametrize("seed", range(4))
def test_align_global_axes_experimental_all_to_all(num_discretes, seed):
    assert "PARALLELACCEL_ENABLE_EXPERIMENTAL_ALL_TO_ALL" in os.environ
    random.seed(seed)
    num_global_discretes = int(np.log2(jax.device_count()))
    val = np.random.normal(
        size=(jax.local_device_count(),)
        + (2,) * (num_discretes - num_global_discretes - 10)
        + (8, 128)
    )
    perm = tuple(random.sample(range(num_discretes), num_discretes))
    desired_global_perm = tuple(
        random.sample(range(num_discretes), num_global_discretes)
    )
    state = jax.pmap(
        lambda x: aswf.AbstractShardedProbabilityFunction(
            x, perm
        )._align_global_axes_experimental_all_to_all(desired_global_perm),
        axis_name=AXIS_NAME,
    )(val)
    assert state.global_axes == desired_global_perm
    _perm = tuple(perm.index(a) for a in state.perm)
    expected = np.transpose(val.reshape((2,) * num_discretes), _perm)
    actual = state.concrete_tensor.real
    expected = np.reshape(expected, (2,) * (num_discretes - 10) + (8, 128))
    np.testing.assert_allclose(
        np.array(actual).reshape(expected.shape), expected
    )
