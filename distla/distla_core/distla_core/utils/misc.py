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
"""Misc. functions."""
from typing import Any, List
import functools
import logging
import re
import time

import numpy as np
import jax.numpy as jnp


def initialize_distla_core_timing_log(
    fname="distla_core_timing.log", mode="w", timestamp=False):
  """ A logger named `distla_core_timing` is initialized, set to DEBUG
  level, and configured to write to a file `fname` opened in mode
  `mode`. This is intended for use with the `log_time` decorator
  below, and should be called at the beginning of the main program.
  """
  logger = logging.getLogger("distla_core_timing")
  logger.setLevel(logging.DEBUG)
  handler = logging.FileHandler(fname, mode=mode)
  if timestamp:
    formatter = logging.Formatter('%(asctime)s - %(message)s')
  else:
    formatter = logging.Formatter('%(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)


def log_time(f, header="", block_argnums=None, shape_argnums=None):
  """  If a logger named "distla_core_timing" has been initialized
  with level DEBUG or lower, the decorated function is timed,
  and the timing along with the supplied header is logged. Otherwise
  this decorator has no effect. The decorated function may be
  Jitted, but the decorator itself should not be (i.e.
  an @logged function cannot be used within a Jit).

  Args:
    f: Function to time.
    header: Optional string to output before the timing.
    block_argnums (tuple): Unless None,
      out[block_argnum[0]][block_argnum[1]][...].block_until_ready()
      will be called within the timing.
    shape_argnums (tuple):
      args[shape_argnum[0]][shape_argnum[1]][...], where args stores the
      positional argnuments to f, will have its .shape attribute logged.
  Returns:
    The decorated function.
  """
  def _unpack(tup, argnums):
    for argnum in argnums:
      tup = tup[argnum]
    return tup

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    logger = logging.getLogger("distla_core_timing")
    log_flag = logger.getEffectiveLevel() == logging.DEBUG
    if log_flag:
      t0 = time.time()
    out = f(*args, **kwargs)
    if log_flag:
      if block_argnums is not None:
        try:
          _unpack(out, block_argnums).block_until_ready()
        except AttributeError:
          print(f"Output {block_argnums} from {header} "
                "had no block_until_ready method.")
      dt = time.time() - t0
      if shape_argnums is None:
        logger.debug("%s, dt=%s", header, dt)
      else:
        try:
          shape = _unpack(args, shape_argnums).shape
        except AttributeError:
          print(f"Input {shape_argnums} to {header} had no shape.")
          raise

        logger.debug("%s, shape= %s, dt=%s", header, (shape,), dt)
    return out
  return wrapper


def apply_pad_serial(matrix, unpadded_dim):
  """ Sets all but the top left unpadded_dim x unpadded_dim block of matrix
  to 0.
  Args:
    matrix: The matrix to pad.
    unpadded_dim: Size of the block to leave unpadded.
      If None this is a null op.
  Returns:
    The padded matrix.
  """
  if unpadded_dim is None:
    return matrix
  m, n = matrix.shape
  rows_vector = jnp.arange(m)
  cols_vector = jnp.arange(n)
  left_panel = rows_vector < unpadded_dim
  top_panel = cols_vector < unpadded_dim
  leave_unmasked = jnp.logical_and(left_panel[:, None], top_panel)
  return jnp.where(leave_unmasked, x=matrix, y=jnp.zeros_like(matrix))


def distance_to_next_divisor(num: int, den: int) -> int:
  """
  Returns `delta`, the smallest non-negative integer which must be added to
  `num` such that `(num + delta) % den == 0`.

  Example:
    `num = 5, den = 5: delta = 0`.
    `num = 6, den = 5: delta = 4`.
    `num = 5, den = 6: delta = 1`.
  """
  return (den - num % den) * ((num % den) != 0)


def similarity_transform(A, V, precision):
  """
  Computes `V.T.conj() @ A @ V`.
  TODO: Move this.
  """
  AV = jnp.dot(A, V, precision=precision)
  return jnp.dot(V.T.conj(), AV, precision=precision)


def gershgorin(H):
  """
  Computes estimates of the smallest and largest eigenvalues of a Hermitian
  `H` using the "Gershgorin" method. The estimates are guaranteed to bound the
  spectrum, but can be quite loose in many cases.

  Args:
    H: The Hermitian matrix whose spectrum is to be bounded.
  Returns:
    min_est: A lower bound on `H`'s smallest eigenvalue.
    max_est: An upper bound on `H`'s largest eigenvalue.
  """
  H_diag = jnp.diag(H)
  diag_elements = jnp.diag_indices_from(H)
  abs_H_diag0 = jnp.abs(H.at[diag_elements].set(0.))
  col_sums = jnp.sum(abs_H_diag0, axis=0)
  row_sums = jnp.sum(abs_H_diag0, axis=1)

  row_min = jnp.min(H_diag - row_sums)
  row_max = jnp.max(H_diag + row_sums)
  col_min = jnp.min(H_diag - col_sums)
  col_max = jnp.max(H_diag + col_sums)

  min_est = jnp.max(jnp.array([row_min, col_min]))
  max_est = jnp.min(jnp.array([row_max, col_max]))
  return min_est, max_est


def global_shape(localshape, grid_shape):
  return tuple(np.array(localshape) * np.array(grid_shape))


def local_shape(globalshape, grid_shape):
  return tuple(np.asarray(globalshape) // np.asarray(grid_shape))


def flatten(list_of_list: List[List[Any]]):
  """
  Flatten a list of lists.

  Args:
    list_of_list: A list of lists.

  Returns:
    List: the flattened `list_of_list`.
  """
  return [a for p in list_of_list for a in p]


def inverse_permutation(perm) -> list:
  """
  Compute inverse permutation `invperm` of `perm`, i.e.
  ```python
  assert np.all(perm[invperm] == np.arange(len(perm)))

  Args:
    perm: A permutation.

  Returns:
    The inverse permutation of `perm`.
  ```
  """
  return list(np.unique(perm, return_index=True)[1])


def prime_factors(N: int) -> List[int]:
  """
  Compute prime factors of `N`.

  Args:
    An integer.

  Return:
    list[int]: The prime factors of `N`, excluding 1.j
  """
  if N == 1:
    return [1]

  prime_facs = []
  divisor = 2
  while divisor**2 <= N:
    while N % divisor == 0:
      N //= divisor
      prime_facs.append(divisor)
    divisor += 1
  prime_facs.append(N)
  result = [p for p in prime_facs if p > 1]
  return result


def maybe_ravel_shape(shape):
  """
  If possible, reshape `shape` into a 2d shape that avoids extensive padding;
  otherwise ravel `shape`.
  """
  primes = np.sort(flatten([prime_factors(s) for s in shape]))[::-1]
  inds = np.nonzero(np.cumprod(primes) >= 128)[0]
  if len(inds) > 0:
    ind = inds[0]
    newshape = np.prod(primes[ind + 1:]), np.prod(primes[:ind + 1])
    return newshape
  return int(np.prod(shape))


def find_common(p1, p2):
  """
  find common values between prime factors p1 and p2.

  Args:
    p1, p2: A list of numbers

  Returns:
    The common values with corresponding multiplicity.
  """
  p1 = list(p1)
  p2 = list(p2)
  common = []
  while len(p1) > 0:
    p = p1.pop()
    if p in p2:
      common.append(p)
      ind = p2.index(p)
      p2.pop(ind)
  return common


def math_def(_func=None, **kwargs):
  """Generates additional docstring for a function for math definitions.

  This wrapper finds all non-standard variable names inside a function.
  For those variable names, it attaches their definitions in the function
  docstring. **DISCLAIMER** the function should disable `invalid-name`
  pylinter rule:

  >>> def a_function_to_be_decorated():
  >>>   # pylint:disable=invalid-name
  >>>   pass

  Here are several examples.

  >>> @math_def(A="tensor",
  >>>           B="another tensor",
  >>>           T="Temperature")
  >>> def f(A, B):
  >>>   # pylint:disable=invalid-name
  >>>   a = 1
  >>>   b = 2
  >>>   snake_case = 3
  >>>   T = 4
  >>>   return (a * A + b * B) * snake_case / T
  >>> help(f)
  Help on function f in module __main__:

  f(A, B)
      This function has been decorated with @distla_core.utils.misc.math_defs.
      Please refer to the following definitions on the math variables
      when reading this function code.

        Def1 : A = tensor
        Def2 : B = another tensor
        Def3 : T = Temperature



  If you don't specify all of the non-standard math name definitions,
  it will complain like:

  >>> @math_def
  >>> def f(A, B):
  >>>   # pylint:disable=invalid-name
  >>>   a = 1
  >>>   b = 2
  >>>   snake_case = 3
  >>>   T = 4
  >>>   return (a * A + b * B) * snake_case / T
  ValueError: math name style `A` is not defined.
  Pleae add a description of the variable in a Python string.

  For example:  `@math_def(A='mathematical definition of `A`')


  Args:
    _func: the function with non-standard math name variables.

  Returns:
    A decorated wrapper function of `_func`.
  """

  def outer_wrapper(func):
    varnames = func.__code__.co_varnames
    non_snake_case_varnames = [
        name for name in varnames if not re.search("^[a-z][a-z0-9_]*$", name)
    ]

    docstring_sub_header = ("\n\nThis function has been decorated with "
                            "@distla_core.utils.misc.math_def.\nPlease refer to "
                            "the following definitions on the math variables\n"
                            "when reading this function code.\n")
    definitions = ""
    for i, name in enumerate(non_snake_case_varnames):
      if name in kwargs:
        definitions += f"\n  Def{i+1} : {name} = {kwargs[name]}"
      else:
        raise ValueError(f"math name style `{name}` is not defined.\n"
                         "Pleae add a description of the variable in a Python "
                         f"string.\n\nFor example:  `@math_def({name}="
                         f"'mathematical definition of `{name}`')")

    appendix = docstring_sub_header + definitions
    if func.__doc__:
      func.__doc__ += appendix
    else:
      func.__doc__ = appendix

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      ret = func(*args, **kwargs)
      return ret

    return wrapper

  if _func:
    return outer_wrapper(_func)
  return outer_wrapper


def is_power_of_two(a):
  """Return whether the argument, cast to int, is a power of 2."""
  a = int(a)
  # Bit manipulations. A power of 2 has a bit represetenation like 0...010...0.
  # For such a number subtracting 1 from it turns it into 0...001...1, so ANDing
  # a-1 and a should yield 0.
  return a > 0 and ((a - 1) & a) == 0


def byte_count(dtype):
  """Return the number of bytes taken by an element of a given dtype."""
  if dtype == jnp.bfloat16:
    return 2
  elif dtype == jnp.float32:
    return 4
  elif dtype == jnp.float64 or dtype == jnp.complex64:
    return 8
  elif dtype == jnp.complex128:
    return 16
  else:
    msg = f"Don't know the byte_count of dtype {dtype}"
    raise NotImplementedError(msg)
