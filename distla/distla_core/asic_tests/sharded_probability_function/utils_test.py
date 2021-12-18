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
"""Tests of helper functions"""
import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import random
import unittest

from asic_la.sharded_probability_function import utils


def test_permute():
    result = utils.permute(["a", "b", "c", "d", "e"], [2, 4, 3, 1, 0])
    assert result == ("c", "e", "d", "b", "a")


def test_perm_calculator():
    result = utils.relative_permutation(
        ["a", "b", "c", "d", "e"], ["c", "e", "d", "b", "a"]
    )
    assert result == (2, 4, 3, 1, 0)


def test_send_to_left():
    result = utils.send_to_left_side([4, 2, 7], [1, 2, 3, 4, 5, 6, 7])
    assert result == (4, 2, 7, 1, 3, 5, 6)


def test_send_to_right():
    result = utils.send_to_right_side([4, 2, 7], [1, 2, 3, 4, 5, 6, 7])
    assert result == (1, 3, 5, 6, 4, 2, 7)


def test_remove_and_reduce():
    perm = (3, 4, 5, 8, 2, 1, 6, 0, 7)
    remove = (6, 1, 3)
    result = utils.remove_and_reduce(perm, remove)
    assert result == (2, 3, 5, 1, 0, 4)


def test_invert_permutation():
    perm = (3, 4, 5, 8, 2, 1, 6, 0, 7)
    invperm = utils.invert_permutation(perm)
    assert tuple([perm[i] for i in invperm]) == tuple(range(len(perm)))


def test_to_index():
    targets = ("a", "c", "e")
    labels = ("a", "b", "c", "d", "e", "f")
    inds = utils.to_index(targets, labels)
    assert inds == (0, 2, 4)
