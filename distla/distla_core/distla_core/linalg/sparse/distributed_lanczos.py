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
"""Implementation of pmap-able Lanczos algorithm."""
from typing import Callable, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp

from distla_core.utils import complex_workaround as cw

ArrayLike = Any


def iterative_classical_gram_schmidt(vector, krylov_vectors, precision,
                                     iterations, matvec):
  """
  Orthogonalize `vector`  to all elements in `krylov_vectors`.
  `krylov_vectors` is a (num_krylov_vecs, ...) shaped DeviceArray
  holding the Krylov vectors along its first dimension (other dimensions
  can be arbitrarily shaped).

  Args:
    vector: Initial vector.
    krylov_vectors: Array of krylov vectors.
    iterations: Number of iterations.
    precision: Arithmetic precision used in jax operations.
    matvec: Callable for computing the matrix vector product between
      `krylov_vectors` and `vector`. Expected signature is
      `overlaps = matvec(krylov_vectors.conj(), vector)`.

  Returns:
    jax.ShapedArray: The orthogonalized vector.
    jax.ShapedArray: The overlaps of `vector` with `krylov_vectors`.
  """

  def body(i, vals):
    vec, overlaps = vals
    ov = matvec(krylov_vectors.conj(), vec)
    vec = vec - jax.numpy.tensordot(
        ov, krylov_vectors, ([0], [0]), precision=precision)
    return vec, overlaps + ov

  return jax.lax.fori_loop(0, iterations, body,
                           (vector, jnp.zeros(krylov_vectors.shape[0])))


def tridiagonalize(alphas, betas):
  offdiag = betas.real[:-1]  #offdiags are always real in Lanczos applications
  diag = alphas.real  #diag is always real in Lanczos applications
  tridiag = jnp.diag(diag) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
  eta, U = jnp.linalg.eigh(tridiag)
  return [eta, U]


def normalize(
    scalar_product: Callable,
    a: Union[cw.ComplexDeviceArray, jax.ShapedArray]) -> jax.ShapedArray:
  """
  Normalize the ShapedArray `a`.

  Args:
    scalar_product: Function to compute the scalar product
      <a,a>.
    a: An input array.

  Returns:
    ShapedArray: The normalized vector.
  """
  return a / cw.sqrt(scalar_product(a, a))


def lanczos_three_term_recurrence_iterated_GS(
    matvec: Callable, scalar_product: Callable, args: Sequence[jax.ShapedArray],
    x1: Union[cw.ComplexDeviceArray, jax.ShapedArray], num_krylov_vecs: int,
    gs_iterations: int) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
  """
  Compute a Lanczos tridiagonalization of a Hermitian
  linear operator `matvec`. This variant uses an iterated
  Gram-Schmidt orthogonalization to improve orthogonality
  between the three Lanczos vectors.

  Args:
    matvec: A Hermitian linear operator. The function signature
      for `matvec` is `y = matvec(x, *args)`.
      `x` can be a jax-array of arbitrary shape. `y` has to be
      a jax-array of identical shape and dtype as `x`. `args`
      is a sequence of additional arguments to `matvec`.
    args: Sequence of additional arguments to `matvec`.
          Signature of `matvec`: `matvec(x1, *args)`.
    scalar_product: The scalar product between two vectors.
    x1: A NORMALIZED initial state for the tridiagonalization procedure
    num_krylov_vecs: Number of Krylov vectors.
    gs_iterations: Number of Gram-Schmidt iterations for reorthogonaliztion.

  Returns:
    ShapedArray: The diagonal elements
    ShapedArray: The off diagonal elements.
    ShapedArray: The overlap coefficients <x1,x2> between
      vectors x2 and x1 at every step of the iterated GS method.
      The array can be used for reconstructing the GS in a second
      Lanczos round.
    ShapedArray: The overlap coefficients <x0,x2> between
      vectors x2 and x0 at every step of the iterated GS method.
      The array can be used for reconstructing the GS in a second
      Lanczos round.
    ShapedArray: Array of overlaps between <x2, x1>, <x2, x0>
      and <x1, x0> at each Lanczos iteration.
  """
  arithmetic_dtype = complex if isinstance(x1, cw.ComplexDeviceArray) else float

  def body(i, vals):
    x0, x1, alphas, betas, alpha_coeffs, gamma_coeffs = vals
    x2 = matvec(x1, *args)

    # orthogonalize
    # "three is enough" for single precision
    def body_gs(j, vals):
      vector, alpha_coeffs, gamma_coeffs, alpha = vals
      a = scalar_product(x1, vector)
      alpha = alpha + a
      alpha_coeffs = alpha_coeffs.at[i, j].set(a)
      vector = vector - a * x1
      gamma = scalar_product(x0, vector)
      gamma_coeffs = gamma_coeffs.at[i, j].set(gamma)
      vector = vector - gamma * x0
      return vector, alpha_coeffs, gamma_coeffs, alpha

    x2, alpha_coeffs, gamma_coeffs, alpha = jax.lax.fori_loop(
        0, gs_iterations, body_gs,
        (x2, alpha_coeffs, gamma_coeffs,
         cw.array(0, dtype=x1.dtype, arithmetic_dtype=arithmetic_dtype)))

    #use complex sqrt
    beta = cw.sqrt(scalar_product(x2, x2))
    x2 /= beta
    alphas = alphas.at[i].set(alpha.real)
    betas = betas.at[i].set(beta.real)
    return x1, x2, alphas, betas, alpha_coeffs, gamma_coeffs

  # Note (mganahl): alphas and betas should always be real for
  # Hermitian matrices. Use of plain jax arrays would be OK here.
  alphas = cw.zeros(num_krylov_vecs, x1.dtype, arithmetic_dtype)
  betas = cw.zeros(num_krylov_vecs, x1.dtype, arithmetic_dtype)
  alpha_coeffs = cw.zeros((num_krylov_vecs, gs_iterations), x1.dtype,
                          arithmetic_dtype)
  gamma_coeffs = cw.zeros((num_krylov_vecs, gs_iterations), x1.dtype,
                          arithmetic_dtype)

  _, _, alphas, betas, alpha_coeffs, gamma_coeffs = jax.lax.fori_loop(
    0, num_krylov_vecs, body, (cw.zeros_like(x1), x1,
                               alphas, betas, alpha_coeffs, gamma_coeffs))
  return alphas, betas, alpha_coeffs, gamma_coeffs


def lanczos_root_solution_three_term_recurrence_iterated_GS(
    matvec: Callable, scalar_product: Callable,
    args: Sequence[Union[cw.ComplexDeviceArray, jax.ShapedArray]],
    x1: Union[cw.ComplexDeviceArray, jax.ShapedArray],
    U: Union[cw.ComplexDeviceArray, jax.ShapedArray],
    betas: Union[cw.ComplexDeviceArray, jax.ShapedArray],
    alpha_coeffs: Union[cw.ComplexDeviceArray, jax.ShapedArray],
    gamma_coeffs: Union[cw.ComplexDeviceArray, jax.ShapedArray]
) -> Union[cw.ComplexDeviceArray, jax.ShapedArray]:
  """
  Compute the ground-state from the eigenvectors `U`
  of the tridiagonalized operator `matvec`, using the
  data obtained from `lanczos_three_term_recurrence_iterated_GS`.

  Args:
    matvec: A Hermitian linear operator. The function signature
      for `matvec` is `y = matvec(x, *args)`.
      `x` can be a jax-array of arbitrary shape. `y` has to be
      a jax-array of identical shape and dtype as `x`. `args`
      is a sequence of additional arguments to `matvec`.
    args: Sequence of additional arguments to `matvec`.
          Signature of `matvec`: `matvec(x1, *args)`.
    scalar_product: The scalar product between two vectors.
    x1: A NORMALIZED initial state of the Lanczos procedure.
    U: Matrix of eigenvectors of the tridiagonal operator.
    betas: diagonal and off diagonal elements of the
      tridiagonal operator.
    alpha_coeffs, gamma_coeffs: Overlaps from the iterated
      Gram Schmidt method.

  Returns:
    ShapedArray: The normalized ground state.
  """

  def body(i, vals):
    x0, x1, x2, gs = vals
    gs = gs + U[i, 0] * x1
    x2 = matvec(x1, *args)

    # orthogonalize (iterated modified gram schmid)
    def body_gs(j, vector):
      vector = vector - alpha_coeffs[i, j] * x1
      vector = vector - gamma_coeffs[i, j] * x0
      return vector

    x2 = jax.lax.fori_loop(0, alpha_coeffs.shape[1], body_gs, x2)
    x2 = x2 / betas[i]
    x0 = x1
    x1 = x2
    return x0, x1, x2, gs

  ground_state = jax.lax.fori_loop(
      0, betas.shape[0], body,
      (cw.zeros_like(x1), x1, cw.zeros_like(x1), cw.zeros_like(x1)))[3]

  return normalize(scalar_product, ground_state)


def lanczos_root_solution(
    matvec: Callable, scalar_product: Callable, args: Sequence[ArrayLike],
    x1: ArrayLike, num_krylov_vecs: int,
    landelta: float) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
  """
  Compute a vanilla Lanczos tridiagonalization of a Hermitian
  linear operator `matvec` and return the dominant eigenvalue-eigenvector
  pair of it. The routine terminates early if it encounters an invariant
  subspace. This routine explicitly stores stored `num_krylov_vecs` Krylov
  vectors in memory.

  Args:
    matvec: A Hermitian linear operator. The function signature for `matvec`
      is `y = matvec(x, *args)`. `x` and `y` have to have identical shapes
      and dtypes. `args` is a sequence of additional arguments to `matvec`.
    args: Sequence of additional arguments to `matvec`.
      Signature of `matvec`: `matvec(x1, *args)`. `args` should be a sequence
      of objects that can be passed to jax.jit as non-static arguments.
    scalar_product: The scalar product between two vectors.
    x1: A NORMALIZED initial state for the tridiagonalization procedure.
      This can be any array-like object which supports the basic operations
      of linear vector-spaces (i.e. addition, subtraction, multiplication by
      scalars).
    num_krylov_vecs: Number of Krylov vectors.
    landelta: Convergence parameter of Lanczos iteration. If the norm of the
      current Lanczos vector falls below `landelta` the iteration is stopped.

  Returns:
    float: The eigenvalue.
    DeviceArray: The eigenvector.
  """
  arithmetic_dtype = complex if isinstance(x1, cw.ComplexDeviceArray) else float

  def body(vals):
    krylov_vectors, alphas, betas, i = vals
    previous_vector = krylov_vectors[i]
    #use complex sqrt
    beta = cw.sqrt(scalar_product(previous_vector, previous_vector))
    normalized_vector = previous_vector / beta
    Av = matvec(normalized_vector, *args)
    alpha = scalar_product(normalized_vector, Av)
    alphas = alphas.at[i - 1].set(alpha.real)
    betas = betas.at[i].set(beta.real)
    # Lanczos update
    next_vector = Av - normalized_vector * alpha - krylov_vectors[i - 1] * beta
    krylov_vectors = krylov_vectors.at[i].set(normalized_vector)
    krylov_vectors = krylov_vectors.at[i + 1].set(next_vector)
    return krylov_vectors, alphas, betas, i + 1

  def cond(vals):
    betas, i = vals[-2], vals[-1]
    norm = betas[i - 1].real
    return jnp.logical_and(i <= num_krylov_vecs, norm > landelta)

  # Note (mganahl): alphas and betas should always be real for
  # Hermitian matrices. Use of plain jax arrays would be OK here.
  alphas = cw.zeros(num_krylov_vecs, x1.dtype, arithmetic_dtype)
  betas = cw.zeros(num_krylov_vecs + 1, x1.dtype, arithmetic_dtype)
  betas = betas.at[0].set(1.0)
  krylov_vecs = cw.zeros((num_krylov_vecs + 2,) + x1.shape, x1.dtype,
                         arithmetic_dtype)
  # NOTE (mganahl): initial vector is normalized inside the loop
  krylov_vecs = krylov_vecs.at[1].set(x1)
  initvals = (krylov_vecs, alphas, betas, 1)
  krylov_vecs, alphas, betas, _ = jax.lax.while_loop(cond, body, initvals)
  offdiag = betas.real[2:]  #offdiags are always real in Lanczos applications
  tridiag = jnp.diag(alphas.real) + jnp.diag(offdiag, 1) + jnp.diag(offdiag, -1)
  eta, U = jnp.linalg.eigh(tridiag)

  def body_state(vals):
    krylov_vecs, gs, betas, i = vals
    gs = gs + U[i, 0] * krylov_vecs[i + 1]
    return krylov_vecs, gs, betas, i + 1

  def cond_state(vals):
    _, _, betas, i = vals
    norm = betas[i + 1].real
    return jnp.logical_and(i < eta.shape[0], norm > landelta)

  ground_state = cw.zeros_like(x1)
  _, ground_state, _, _ = jax.lax.while_loop(
      cond_state, body_state, (krylov_vecs, ground_state, betas, 0))
  return eta[0], normalize(scalar_product, ground_state)


def lanczos_iterated_GS(
    matvec: Callable, scalar_product: Callable,
    args: Sequence[Union[cw.ComplexDeviceArray, jax.ShapedArray]],
    x1: Union[cw.ComplexDeviceArray, jax.ShapedArray], num_krylov_vecs: int,
    maxiter: int,
    gs_iterations: int) -> Tuple[jax.ShapedArray, jax.ShapedArray]:
  """
  Explicitly num_krylov_vecsed Lanczos tridiagonalization of a
  Hermitian linear operator `matvec`. This variant
  uses a Lanczos tridiagonalization with iterated
  Gram Schmidt orthogonalization. This function performs
  all operations in fixed precision using jax operations.
  Args:
    matvec: A Hermitian linear operator. The function signature
      for `matvec` is `y = matvec(x, *args)`.
      `x` can be a jax-array of arbitrary shape. `y` has to be
      a jax-array of identical shape and dtype as `x`. `args`
      is a sequence of additional arguments to `matvec`.
    args: Sequence of additional arguments to `matvec`.
          Signature of `matvec`: `matvec(x1, *args)`.
    x1: The initial state of the Lanczos procedure.
    num_krylov_vecs: The number of Krylov vectors.
    maxiter: Maximum number of explicit restarts.
    gs_iterations: Number of Gram-Schmidt iterations.

  Returns:
    ShapedArray: The lowest eigenvalue.
    ShapedArray: The lowest eigenvector.
  """

  def body(i, vals):  # pylint: disable=unused-argument
    _, x1 = vals
    variables = lanczos_three_term_recurrence_iterated_GS(
        matvec, scalar_product, args, x1, num_krylov_vecs, gs_iterations)
    alphas, betas, alpha_coeffs, gamma_coeffs = variables
    eta, U = tridiagonalize(alphas, betas)
    x1 = lanczos_root_solution_three_term_recurrence_iterated_GS(
        matvec, scalar_product, args, x1, U, betas, alpha_coeffs, gamma_coeffs)
    return eta[0], x1

  eta, ground_state = jax.lax.fori_loop(
      0, maxiter, body, (x1.dtype.type(0.0), normalize(scalar_product, x1)))
  return eta, ground_state
