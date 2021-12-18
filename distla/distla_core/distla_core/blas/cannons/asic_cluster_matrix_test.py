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
"""Tests for asic_node_matrix.py"""
import jax
import numpy as np
import unittest

from distla_core.blas.cannons import asic_node_matrix


class ASICNodeMatrixTest(unittest.TestCase):

  ## distribute_to_asic_node tests
  def test_basic_mul(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    b = np.random.normal(size=(8, 8))
    out = np.matmul(a, b)

    @asic_node_matrix.distribute_to_asic_node
    def op(a, b):
      return a @ b

    test_val = op(a, b)
    np.testing.assert_allclose(out, test_val, rtol=1e-04)
    self.assertIsInstance(test_val, np.ndarray)

  def test_basic2_mul(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    b = np.random.normal(size=(8, 8))
    out = np.matmul(a, b)
    out = np.matmul(out, a)
    out = np.matmul(out, b)

    @asic_node_matrix.distribute_to_asic_node
    def op(a, b):
      val = a @ b
      val = val @ a
      val = val @ b
      return val

    test_val = op(a, b)
    np.testing.assert_allclose(out, test_val, rtol=1e-04)
    self.assertIsInstance(test_val, np.ndarray)

  def test_tuple_mul(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    b = np.random.normal(size=(8, 8))
    out = (np.matmul(a, a), np.matmul(b, b))

    @asic_node_matrix.distribute_to_asic_node
    def op(a, b):
      return a @ a, b @ b

    vals = op(a, b)
    self.assertIsInstance(vals, tuple)
    for v in vals:
      self.assertIsInstance(v, np.ndarray)
    np.testing.assert_allclose(out, vals, rtol=1e-04)

  ## distribute_to_asic_node_jittable tests
  def test_basic_mul_jittable(self):
    np.random.seed(10)
    a = jax.numpy.array(np.random.normal(size=(8, 8)))
    b = jax.numpy.array(np.random.normal(size=(8, 8)))
    out = np.matmul(a, b)

    @asic_node_matrix.distribute_to_asic_node_jittable
    def op(a, b):
      return a @ b

    test_val = op(a, b)
    np.testing.assert_allclose(out, test_val, rtol=1e-04)
    self.assertIsInstance(test_val, jax.interpreters.xla.DeviceArray)

  def test_jitted_basic_mul_jittable(self):
    np.random.seed(10)
    a = jax.numpy.array(np.random.normal(size=(8, 8)))
    b = jax.numpy.array(np.random.normal(size=(8, 8)))
    out = np.matmul(a, b)

    @jax.jit
    def do_op(a, b):

      @asic_node_matrix.distribute_to_asic_node_jittable
      def op(a, b):
        return a @ b

      return op(a, b)

    test_val = do_op(a, b)
    np.testing.assert_allclose(out, test_val, rtol=1e-04)
    self.assertIsInstance(test_val, jax.interpreters.xla.DeviceArray)

  def test_basic2_mul_jittable(self):
    np.random.seed(10)
    a = jax.numpy.array(np.random.normal(size=(8, 8)))
    b = jax.numpy.array(np.random.normal(size=(8, 8)))
    out = np.matmul(a, b)
    out = np.matmul(out, a)
    out = np.matmul(out, b)

    @asic_node_matrix.distribute_to_asic_node_jittable
    def op(a, b):
      val = a @ b
      val = val @ a
      val = val @ b
      return val

    test_val = op(a, b)
    np.testing.assert_allclose(out, test_val, rtol=1e-04)
    self.assertIsInstance(test_val, jax.interpreters.xla.DeviceArray)

  def test_tuple_mul_jittable(self):
    np.random.seed(10)
    a = jax.numpy.array(np.random.normal(size=(8, 8)))
    b = jax.numpy.array(np.random.normal(size=(8, 8)))
    out = (np.matmul(a, a), np.matmul(b, b))

    @asic_node_matrix.distribute_to_asic_node_jittable
    def op(a, b):
      return a @ a, b @ b

    vals = op(a, b)
    self.assertIsInstance(vals, tuple)
    for v in vals:
      self.assertIsInstance(v, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(out, vals, rtol=1e-04)

  def test_basic_fori_loop(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    val = a
    for _ in range(0, 3):
      val = val @ val

    @asic_node_matrix.distribute_to_asic_node
    def do_op(a):

      def op(i, a):  # pylint: disable=unused-argument
        return a @ a

      return jax.lax.fori_loop(0, 3, op, a)  # pylint: disable=no-value-for-parameter

    out = do_op(a)

    np.testing.assert_allclose(out, val, rtol=0.1)
    self.assertIsInstance(out, np.ndarray)

  def test_tuple_fori_loop(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    b = np.random.normal(size=(8, 8))

    vals = (a, b)
    for _ in range(0, 3):
      vals = (vals[0] @ vals[0], vals[1] @ vals[1])

    @asic_node_matrix.distribute_to_asic_node
    def do_op(a, b):

      def op(i, vals):  # pylint: disable=unused-argument
        a, b = vals
        return a @ a, b @ b

      return jax.lax.fori_loop(0, 3, op, (a, b))

    out = do_op(a, b)

    np.testing.assert_allclose(out, vals, rtol=0.1)
    self.assertIsInstance(out, tuple)

  def test_jitted_fori_loop(self):
    np.random.seed(10)
    a = np.random.normal(size=(8, 8))
    val = a
    for _ in range(0, 3):
      val = val @ val

    @jax.jit
    def run(a):

      @asic_node_matrix.distribute_to_asic_node_jittable
      def do_op(a):

        def op(i, a):  # pylint: disable=unused-argument
          return a @ a

        return jax.lax.fori_loop(0, 3, op, a)  # pylint: disable=no-value-for-parameter

      return do_op(a)

    out = run(a)

    np.testing.assert_allclose(out, val, rtol=0.1)
    self.assertIsInstance(out, jax.interpreters.xla.DeviceArray)


if __name__ == '__main__':
  unittest.main()
