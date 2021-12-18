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
"""Module for Chebyshev polynomial approximations of matrix functions.

See e.g. Chapter 3 of Numerical Methods for Special Functions by Amparo Gil,
Javier Segura, and Nico Temme for the theory.
"""
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import scipy

from distla_core.blas.summa import summa
from distla_core.utils import pops
from distla_core.utils import vops

# REDACTED In the future I would like there to be much more automation for
# choosing the order of the Chebyshev polynomial. Ideally something like a
# maximum order provided by the user, and an error threshold, so that if the
# error in the approximation falls below the threshold a lower order polynomial
# can be used.


def _interval_mid_radius(interval, dtype):
  """Gets the midpoint and radius of interval, as elements of dtype."""
  mid = dtype((interval[0] + interval[1]) / 2)
  radius = dtype((interval[1] - interval[0]) / 2)
  return mid, radius


def chebyshev_coeffs(
    f,
    interval,
    n=200,
    is_vectorized=True,
):
  """Computes the Chebyshev expansion coefficients of f within the interval.

  Args:
    f: The function for which to compute the Chebyshev coefficients. f should
      be able to take a single scalar and return a scalar. If is_vectorized is
      True, f is also assumed to operate element-wise on Numpy arrays.
    interval: A two-element tuple defining the lower and upper bounds of the
      interval within which the Chebyshev expansion should be valid.
    n: Optional; The order of the Chebyshev polynomial in the approximation.
      200 by default.
    is_vectorized: Optional; A boolean for whether f can be called on vectors,
      or only on individual scalars. True by default.
  Returns:
    cs: The Chebyshev expansion coefficients, as a Numpy vector of length n+1.
  """
  if not is_vectorized:
    f = np.vectorize(f)
  interval_mid, interval_radius = _interval_mid_radius(interval, np.float64)
  thetas = (np.arange(0, n + 1, 1) + 0.5) * np.pi / (n + 1)
  xs = np.cos(thetas) * interval_radius + interval_mid
  fs = f(xs)
  # DCT is Discrete Cosine Transform
  cs = scipy.fft.dct(fs, type=2) / (n + 1)
  cs[0] = cs[0] / 2
  return cs


def clenshaw_evaluate_scalar(x, cs, interval):
  """Evaluates the Chebyshev series defined by coefficients cs and the interval,
  at point x.

  Args:
    x: A scalar or numpy array of scalars.
    cs: Chebyshev expansion coefficients, as a Numpy array.
    interval: Tuple defining the lower and upper bounds of the interval within
      which the Chebyshev expansion was done.
  Returns:
    The Chebyshev expansion approximation to f(x), where f is the function for
    which the expansion was done.
  """
  dtype = x.dtype.type
  cs = cs.astype(dtype)
  interval_mid, interval_radius = _interval_mid_radius(interval, dtype)
  x = (x - interval_mid) / interval_radius
  z = dtype(0)
  y = cs[-1]
  for c in cs[-2:0:-1]:
    f = dtype(2) * x * y - z + c
    z = y
    y = f
  return x * y - z + cs[0]


def clenshaw_evaluate_matmat(x, cs, interval, precision=lax.Precision.HIGHEST):
  """Evaluates the Chebyshev series defined by coefficients cs and the interval,
  for a matrix x.

  Args:
    x: A matrix, as a jax DeviceArray.
    cs: Chebyshev expansion coefficients, as jax DeviceArray.
    interval: Tuple defining the lower and upper bounds of the interval within
      which the Chebyshev expansion was done.
    precision: Precision for matmuls.
  Returns:
    The Chebyshev expansion approximation to f(x), where f is the matrix version
    of the function for which the expansion was done.
  """
  dtype = x.dtype.type
  cs = cs.astype(dtype)
  interval_mid, interval_radius = _interval_mid_radius(interval, dtype)
  eye = jnp.eye(x.shape[0], dtype=dtype)
  x = (x - eye * interval_mid) / interval_radius
  z = jnp.zeros(x.shape, dtype=dtype)
  y = eye * cs[-1]
  c_index = cs.shape[0] - 2

  def cond(args):
    c_index = args[-1]
    return c_index > 0

  def body(args):
    y, z, c_index = args
    c = cs[c_index]
    # REDACTED Can we subtract contributions proportional to the identity
    # faster and with less memory use?
    f = dtype(2) * pops.dot(x, y, precision=precision) - z + eye * c
    z = y
    y = f
    return y, z, c_index - 1

  y, z, c_index = jax.lax.while_loop(cond, body, (y, z, c_index))
  return pops.dot(x, y, precision=precision) - z + eye * cs[0]


def clenshaw_evaluate_matvec(
    x,
    v,
    cs,
    interval,
    precision=lax.Precision.HIGHEST,
):
  """Evaluates the Chebyshev series defined by coefficients cs and the interval
  for a matrix x, multiplied from the right by matrix v.

  If x is D x D and v is D x k, only O(D^2 k) operations are used (so no O(D^3)
  matrix products).

  Args:
    x: A matrix, as a jax DeviceArray.
    v: A vector or a "thin" matrix, as a jax DeviceArray.
    cs: Chebyshev expansion coefficients, as jax DeviceArray.
    interval: Tuple defining the lower and upper bounds of the interval within
      which the Chebyshev expansion was done.
    precision: Precision for matmuls.
  Returns:
    The Chebyshev expansion approximation to f(x) @ v, where f is the matrix
    version of the function for which the expansion was done.
  """
  dtype = x.dtype.type
  cs = cs.astype(dtype)
  interval_mid, interval_radius = _interval_mid_radius(interval, dtype)
  eye = jnp.eye(x.shape[0], dtype=dtype)
  x = (x - eye * interval_mid) / interval_radius
  zv = jnp.zeros(v.shape, dtype=dtype)
  yv = v * cs[-1]
  c_index = cs.shape[0] - 2

  def cond(args):
    c_index = args[-1]
    return c_index > 0

  def body(args):
    yv, zv, c_index = args
    c = cs[c_index]
    fv = dtype(2) * pops.dot(x, yv, precision=precision) - zv + v * c
    zv = yv
    yv = fv
    return yv, zv, c_index - 1

  yv, zv, c_index = jax.lax.while_loop(cond, body, (yv, zv, c_index))
  return pops.dot(x, yv, precision=precision) - zv + v * cs[0]


def clenshaw_evaluate_pmatmat(
    x,
    cs,
    interval,
    p_sz,
    precision=lax.Precision.HIGHEST,
):
  """Evaluates the Chebyshev series defined by coefficients cs and the interval,
  for a distributed matrix x.

  This function is desiged to be pmapped over.

  Args:
    x: A matrix, as a jax ShardedDeviceArray.
    cs: Chebyshev expansion coefficients, as jax DeviceArray.
    interval: Tuple defining the lower and upper bounds of the interval within
      which the Chebyshev expansion was done.
    p_sz: Panel size for SUMMA.
    precision: Precision for matmuls.
  Returns:
    The Chebyshev expansion approximation to f(x), where f is the matrix version
    of the function for which the expansion was done.
  """
  dtype = x.dtype.type
  cs = cs.astype(dtype)
  interval_mid, interval_radius = _interval_mid_radius(interval, dtype)
  eye = pops.eye(x.shape, dtype)
  x = (x - eye * interval_mid) / interval_radius
  z = jnp.zeros(x.shape, dtype=dtype)
  y = eye * cs[-1]
  c_index = cs.shape[0] - 2

  def summa_f(x, y):
    return summa.summa(x, y, p_sz, False, False, precision=precision)

  def cond(args):
    c_index = args[-1]
    return c_index > 0

  def body(args):
    y, z, c_index = args
    c = cs[c_index]
    # REDACTED Can we subtract contributions proportional to the identity
    # faster and with less memory use?
    f = dtype(2) * summa_f(x, y) - z + eye * c
    z = y
    y = f
    return y, z, c_index - 1

  y, z, c_index = jax.lax.while_loop(cond, body, (y, z, c_index))
  return summa_f(x, y) - z + eye * cs[0]


def clenshaw_evaluate_pmatvec(
    x,
    v,
    cs,
    interval,
    precision=lax.Precision.HIGHEST,
):
  """Evaluates the Chebyshev series defined by coefficients cs and the interval
  for a distributed matrix x, multiplied from the right by a distributed vector
  or thin matrix v.

  If x is D x D and v is D x k, only O(D^2 k) operations are used (so no O(D^3)
  matrix products).

  Args:
    x: A matrix, as a jax ShardedDeviceArray.
    v: A vector or a "thin" matrix, as a jax vops.ReplicatedThinMatrix.
    cs: Chebyshev expansion coefficients, as jax DeviceArray.
    interval: Tuple defining the lower and upper bounds of the interval within
      which the Chebyshev expansion was done.
    precision: Precision for matmuls.
  Returns:
    The Chebyshev expansion approximation to f(x) @ v, where f is the matrix
    version of the function for which the expansion was done.
  """
  dtype = x.dtype.type
  cs = cs.astype(dtype)
  interval_mid, interval_radius = _interval_mid_radius(interval, dtype)
  eye = pops.eye(x.shape, dtype)
  x = (x - eye * interval_mid) / interval_radius
  zv = v.zeros_like()
  yv = v * cs[-1]
  c_index = cs.shape[0] - 2

  def cond(args):
    c_index = args[-1]
    return c_index > 0

  def body(args):
    yv, zv, c_index = args
    c = cs[c_index]
    fv = dtype(2) * vops.matvec(x, yv, precision=precision) - zv + v * c
    zv = yv
    yv = fv
    return yv, zv, c_index - 1

  yv, zv, c_index = jax.lax.while_loop(cond, body, (yv, zv, c_index))
  return vops.matvec(x, yv, precision=precision) - zv + v * cs[0]


def chebyshize(
    f,
    interval,
    n=200,
    is_vectorized=True,
    return_which=("scalar", "matmat", "matvec", "pmatmat", "pmatvec"),
    p_sz=128,
    precision=lax.Precision.HIGHEST,
):
  """Constructs a Chebyshev polynomial approximation to f within the interval.

  For a given real Numpy scalar function f, computes its order-n Chebyshev
  expansion within the given interval. The return values can include various
  versions of the Chebyshev approximation to f, that can be applied to scalars,
  matrices, or distributed matrices, or that can directly evaluate the
  matrix-vector product f(x) @ v. See the argument return_which and the
  description of return values for the details. The Chebyshev series is
  evaluated using Clenshaw's method, which allows the matrix versions to be
  evaluated using only matrix products.

  In most use cases f is the scalar version of some function for which we need
  the matrix equivalent, built by combining basic Numpy functions. For instance,
  if you would want to evaluate the matrix exponential, then f = np.exp. The
  returned approximation to the scalar version of f can then be used to check
  how accurate the approximation is, and perhaps adjust the order of the
  polynomial accordingly. The Chebyshev coefficients are also returned, and can
  be used to estimate the error. Then either the matrix or the distributed
  matrix version can be used as needed, providing an approximation to the
  desired matrix function, but only using matrix products to evaluate it.

  Note that the approximation is only valid for scalars within the given
  interval. Outside it, it typically diverges very rapidly. For matrices this
  means that the approximation is valid for Hermitian matrices with all
  eigenvalues within the given interval.

  For the Chebyshev series to converge (for infinite n), f should be
  sufficiently smooth. Differentiability in the interval is sufficient.

  See e.g. Chapter 3 of Numerical Methods for Special Functions by Amparo Gil,
  Javier Segura, and Nico Temme for more about Chebyshev expansions.

  Args:
    f: The real function to approximate. f should be able to take a single
      scalar and return a scalar. If is_vectorized is True, f is also assumed to
      operate element-wise on Numpy arrays.
    interval: A two-element tuple defining the lower and upper bounds of the
      interval within which the Chebyshev approximation should be accurate.
    n: Optional; The order of the Chebyshev polynomial in the approximation.
      The higher the order, the more accurate the approximation. The matrix
      functions returned take n matmuls during each evaluation. 200 by default.
    is_vectorized: Optional; A boolean for whether f can be called on vectors,
      or only on individual scalars. True by default.
    return_which: Optional; A Tuple of strings, that specifies which functions
      should be returned. The possible elements in the string are "scalar",
      "matmat", "matvec", "pmatmat", and "pvecmat", and by default they are all
      returned, in that order. See the section on return values for more.
    p_sz: Optional; The panel size for SUMMA, used in the distributed version of
      the approximation. See SUMMA's documentation for more.
    precision: Precision of the matrix multiplications.
  Returns:
    The return value depends on the argument return_which. For instance, if
    return_which=("scalar", "pmatmat"), then the return value would be scalar,
    pmatmat, cs. The different possibilities are explained below.

    scalar: Function that evaluates the Chebyshev polynomial approximation to
      f for a scalar or vector of scalars. Takes a single argument x and returns
      an approximation to f(x).
    matmat: Function that evaluates the Chebyshev polynomial approximation to
      the matrix function corresponding to f. Takes a single jax matrix and
      returns a single jax matrix. The approximation should be valid for
      Hermitian matrices with eigenvalues in the range of the given interval.
      Can be jitted.
    matvec: Function that evaluates the Chebyshev polynomial approximation to
      f(x) @ v, where x is a matrix and v is a vector or a matrix. Takes as
      arguments x and v that should be jax arrays, and returns a single jax
      matrix. The approximation should be valid for Hermitian matrices with
      eigenvalues in the range of the given interval.  Can be jitted.
    pmatmat: Like matmat, but can be pmapped over and applied to distributed
      matrices. Uses summa for the matrix products.
    pmatvec: Like matvec, but can be pmapped over and applied to distributed
      matrices. Takes as argument a distributed matrix x and a
      pops.ReplicatedThinMatrix v.
    cs: The Chebyshev coefficients used. This is always returned as the last
      return value, regardless of return_which.
  """
  cs = chebyshev_coeffs(f, interval, n, is_vectorized)
  cs_jnp = jax.device_put(cs)

  retval = []
  for s in return_which:
    if s == "scalar":
      retval.append(lambda x: clenshaw_evaluate_scalar(x, cs, interval))
    elif s == "matmat":
      retval.append(lambda x: clenshaw_evaluate_matmat(
          x,
          cs_jnp,
          interval,
          precision=precision,
      ))
    elif s == "matvec":
      retval.append(lambda x, v: clenshaw_evaluate_matvec(
          x,
          v,
          cs_jnp,
          interval,
          precision=precision,
      ))
    elif s == "pmatmat":
      retval.append(lambda x: clenshaw_evaluate_pmatmat(
          x,
          cs_jnp,
          interval,
          p_sz,
          precision=precision,
      ))
    elif s == "pmatvec":
      retval.append(lambda x, v: clenshaw_evaluate_pmatvec(
          x,
          v,
          cs_jnp,
          interval,
          precision=precision,
      ))
    else:
      msg = "Unknown value in return_which: {}".format(s)
      raise ValueError(msg)
  retval.append(cs)
  return tuple(retval)
