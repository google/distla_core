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
"""Module for refining the precision of a polar decomposition."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from distla_core.blas.summa import summa
from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.backends import serial_backend
from distla_core.linalg.mixed_precision import refine_unitarity
from distla_core.linalg.mixed_precision import utils
from distla_core.utils import misc
from distla_core.utils import pops


def _rotate_decomp_preprocess(dim, A_norm, input_dtype, max_order):
  """Preprocessing for `_rotate_decomp`.

  This function finds out which order we need to go to in the Taylor expansion
  of exp(A), and at what precision the various matmuls need to be done, so that
  the action of exp(A) is unitary up to the precision afforded by `input_dtype`.

  The point of separating this preprocessing from `_rotate_decomp` is that this
  part can be run on the host without doing any matmuls, and once that is done
  `_rotate_decomp` can know all the dtypes used at various stages and thus
  compile the whole operation into one pmapped/jitted call.

  Args:
    dim: The dimension of the matrix `A`.
    A_norm: Frobenius norm of `A`.
    input_dtype: The precision at which we need to preserve unitarity.
    max_order: The maximum order to use in the Taylor expansion.

  Returns:
    iterations: A tuple of tuples, where each element corresponds to one order
      in the Taylor expansion (starting with the first, not the zeroth), and holds
      the following values, in order :
      factorial: The factorial order!
      sign: (-1)^order
      dtype: The dtype that should be used in the matmuls at this order.
      precision: The precision that should be used in the matmuls at this order.
  """
  # This is the mean-square of the absolute values elements in A. We use it as
  # our estimate of the "typical" element in A.
  mean_element = float(A_norm) / dim
  # error_size keeps track of the typical element size in the dominant order
  # that we've so far ignored.
  error_size = mean_element
  order = 0
  factorial = 1
  final_epsilon = np.float64(
      2**(-utils.mantissa_bits(input_dtype, jax.lax.Precision.HIGHEST)))
  iterations = []
  # We keep adding more terms to the expansion of exp(A) until the largest term
  # we are neglecting is so small that it's outside our numerical precision.
  while error_size > final_epsilon and order < max_order:
    order += 1
    # The number of bits that must be correct in the new term we are adding.
    remaining_bits = np.log(error_size / final_epsilon) / np.log(2)
    # The minimal dtype and precision that allow that number of bits of
    # precision.
    dtype, precision = utils.minimal_precision(remaining_bits)
    factorial *= order
    sign = 1 if order % 2 == 0 else -1
    # At this point the we would actually do the matmuls. However, that is a job
    # for _rotate_decomp, not the preprocessing.
    # Multiplying the current term by A changes the size of its typical elements
    # roughly as x -> x * mean_element * dim. This is a somewhat safe estimate,
    # often cancellations will make this only grow as sqrt(dim), but we play it
    # safe. Dividing by (order + 1) keeps track of the factorial term in the
    # Taylor expansion of exp.
    error_size *= mean_element * dim / (order + 1)
    iterations.append((factorial, sign, dtype, precision))
  return tuple(iterations)


def _rotate_decomp_bare(U, P, A, iterations, backend):
  """Applies a small unitary rotation to a polar decomposition.

  A is an anti-Hermitian matrix that defines the unitary V = exp(-A). The polar
  decomposition (U, P) is then rotated to be (U @ V, V^dg @ P).

  This is done by expanding the Taylor series of exp(-A), assuming A is small.
  The order of the expansion and precision for the matmuls used at each order is
  encoded in `iterations`, which should be the output of
  `_rotate_decomp_preprocess`.

  This, the _bare version of the function, should be either jitted or pmapped
  with `static_argnums=(3, 4)`.

  Args:
    U, P: The polar decomposition to rotate.
    A: The anti-Hermitian matrix defining the rotation.
    iterations: The output of `_rotate_decomp_preprocess`, holding information
      about the orders in the expansion.
    backend: A `distla_core.linalg.backends.distributed_backend` or `serial_backend`

  Returns:
    U_rotated, P_rotated: The rotated polar decomposition.
  """
  # The cumulants will hold the current term in the expansion.
  U_cumulant = U
  P_cumulant = P
  input_dtype = U.dtype
  for (factor, sign, dtype, precision) in iterations:
    # dtypes at successive iterations only go down in precision, so we can
    # safely cast A and the cumulants.
    if A.dtype != dtype:
      A = A.astype(dtype)
    if U_cumulant.dtype != dtype:
      U_cumulant = U_cumulant.astype(dtype)
    if P_cumulant.dtype != dtype:
      P_cumulant = P_cumulant.astype(dtype)
    # The terms to add are U @ (-A)**order / order! and A**order @ P / order!
    U_cumulant = backend.matmul(U_cumulant, A, precision=precision)
    P_cumulant = backend.matmul(A, P_cumulant, precision=precision)
    U_term = dtype(sign / factor) * U_cumulant
    P_term = dtype(1 / factor) * P_cumulant
    U = U + U_term.astype(input_dtype)
    P = P + P_term.astype(input_dtype)
  return U, P


_rotate_decomp_jit = jax.jit(
    _rotate_decomp_bare,
    static_argnums=(3, 4),
)
_rotate_decomp_pmap = pops.pmap(
    _rotate_decomp_bare,
    static_broadcasted_argnums=(3, 4),
)


def _lyapunov_iteration(a, b, X, Y, j, maxiter, eps, precision, backend):
  """Newton-Schultz style iteration for solving a Lyapunov equation."""
  N = backend.shape(X)[0]
  a = jnp.array(a, dtype=X.dtype)
  b = jnp.array(b, dtype=X.dtype)
  eye = backend.eye_like(X)
  N_sqrt = jnp.sqrt(N).astype(X.dtype)

  def cond(args):
    X, Y, j, err, prev_err, err_change = args
    return jnp.logical_and(j < maxiter,
                           jnp.logical_and(err > eps, err_change < 0))

  def body(args):
    X, Y, j, err, prev_err, err_change = args
    X2 = backend.matmul(X.conj(), X, transpose_A=True, precision=precision)
    X3 = backend.matmul(X, X2, precision=precision)
    XY = backend.matmul(X, Y, precision=precision)
    Q = backend.matmul(
        X,
        XY + backend.transpose(XY.conj()) / 2,
        precision=precision,
    )
    commterm = Q - backend.transpose(Q.conj())
    X = a * X + b * X3
    Y = a * Y + b * commterm
    j += 1
    prev_err = err
    err = backend.frobnorm(eye - X) / N_sqrt
    err_change = (err - prev_err) / err
    return X, Y, j, err, prev_err, err_change

  err = backend.frobnorm(eye - X) / N_sqrt
  prev_err = jnp.array(jnp.inf, dtype=X.dtype)
  err_change = jnp.array(-jnp.inf, dtype=X.dtype)
  init_val = (X, Y, j, err, prev_err, err_change)
  X, Y, j, err, prev_err, err_change = jax.lax.while_loop(cond, body, init_val)
  return X, Y, j, err


def _newton_schultz(X, Y, j, maxiter, eps, precision, backend):
  return _lyapunov_iteration(
      1.5,
      -0.5,
      X,
      Y,
      j,
      maxiter,
      eps,
      precision,
      backend,
  )


def _solve_lyapunov_ns_bare(P, dtype, precision, maxiter, backend):
  """Solves the Lyapunov equation
  Y + AX + X^dagger A = 0,
  where Y = P^dagger - P and X = (P^dagger + P) / 2, using Newton-Schultz
  iteration.

  X and Y are cast to the given dtype before running the Newton-Schultz
  iteration to find the solution, and that is also the dtype of the solution A.

  This, the _bare version of the function, should be either jitted or pmapped
  with `static_argnums=(1, 2, 3, 4)`.

  Args:
    P: The matrix that defines the equation.
    dtype: The dtype to use. `Y = P^H - P` is computed in full precision of `P`,
      but then cast to `dtype` for the Newton-Schultz iterations.
    precision: The Jax matrix multiplication precision to use.
    maxiter: Maximum number of Newton-Schultz iterations.
    backend: A `distla_core.linalg.backends.distributed_backend` or `serial_backend`

  Returns:
    A: The solution matrix.
    itercount: Number of Newton-Schultz iterations used.
    err: A convergence measure for Newton-Schultz.
  """
  N = P.shape[0]
  PH = backend.transpose(P.conj())
  X = ((PH + P) / 2).astype(dtype)
  Y = (PH - P).astype(dtype)
  # Normalise to make sure Newton-Schultz doesn't diverge.
  X_norm = backend.frobnorm(X)
  X = X / X_norm
  Y = Y / X_norm
  eps = 2**(-utils.mantissa_bits(dtype, precision))
  _, Y, itercount, err = _newton_schultz(
      X,
      Y,
      0,
      maxiter,
      eps,
      precision,
      backend,
  )
  A = Y / 2
  return A, itercount, err


_solve_lyapunov_ns_jit = jax.jit(
    _solve_lyapunov_ns_bare,
    static_argnums=(1, 2, 3, 4),
)
_solve_lyapunov_ns_pmap = pops.pmap(
    _solve_lyapunov_ns_bare,
    out_axes=(0, None, None),
    static_broadcasted_argnums=(1, 2, 3, 4),
)


def _solve_lyapunov_cg_bare(P, dtype, precision, maxiter, backend):
  """Solves the Lyapunov equation
  Y + AX + X^dagger A = 0,
  where Y = P^dagger - P and X = (P^dagger + P) / 2, using conjubuilding_block gradient.

  X and Y are cast to the given dtype before running the solver, and that is
  also the dtype of the solution A.

  This, the _bare version of the function, should be either jitted or pmapped
  with `static_argnums=(1, 2, 3, 4)`.

  Args:
    P: The matrix that defines the equation.
    dtype: The dtype to use. `Y = P^H - P` is computed in full precision of `P`,
      but then cast to `dtype` for the Newton-Schultz iterations.
    precision: The Jax matrix multiplication precision to use.
    maxiter: Maximum number of conjubuilding_block gradient iterations.
    backend: A `distla_core.linalg.backends.distributed_backend` or `serial_backend`

  Returns:
    A: The solution matrix.
    itercount: Number of iterations used.
    err: Conjubuilding_block gradient residual error.
  """
  N = P.shape[0]
  PH = backend.transpose(P.conj())
  X = ((PH + P) / 2).astype(dtype)
  Y = (PH - P).astype(dtype)
  Y_norm = backend.frobnorm(Y)
  eps = 2**(-utils.mantissa_bits(dtype, precision))

  def cond(args):
    _, _, _, residual_norm_sq, counter = args
    err = jnp.sqrt(residual_norm_sq) / Y_norm
    return jnp.logical_and(counter < maxiter, err > eps)

  def body(args):
    A, conjvec, residual, residual_norm_sq, counter = args
    conjvec_X = backend.matmul(conjvec, X, precision=precision)
    # convjec_mapped = conjvec @ X + X^H @ conjvec, but we make use of the fact
    # that conjvec is exactly anti-Hermitian and X is exactly Hermitian.
    conjvec_mapped = conjvec_X - backend.transpose(conjvec_X.conj())
    alpha = residual_norm_sq / backend.vdot(
        conjvec,
        conjvec_mapped,
        precision=precision,
    )
    A = A + alpha * conjvec
    residual = residual - alpha * conjvec_mapped
    new_residual_norm_sq = backend.frobnorm(residual)**2
    beta = new_residual_norm_sq / residual_norm_sq
    conjvec = residual + beta * conjvec
    return A, conjvec, residual, new_residual_norm_sq, counter + 1

  A = Y
  AX = backend.matmul(A, X, precision=precision)
  residual = Y - (AX - backend.transpose(AX.conj()))
  residual_normsq = backend.frobnorm(residual)**2
  initval = (A, residual, residual, residual_normsq, 0)
  A, _, _, residual_norm_sq, itercount = jax.lax.while_loop(cond, body, initval)
  err = jnp.sqrt(residual_norm_sq) / Y_norm
  return A, itercount, err


_solve_lyapunov_cg_jit = jax.jit(
    _solve_lyapunov_cg_bare,
    static_argnums=(1, 2, 3, 4),
)
_solve_lyapunov_cg_pmap = pops.pmap(
    _solve_lyapunov_cg_bare,
    out_axes=(0, None, None),
    static_broadcasted_argnums=(1, 2, 3, 4),
)


def _hermiticity_bare(P, P_norm, backend):
  """Computes ||P - P^H||_frob / P_norm. Should be pmapped/jitted."""
  diff_norm = backend.frobnorm(P - backend.transpose(P.conj()))
  return diff_norm / P_norm


_hermiticity_jit = jax.jit(_hermiticity_bare, static_argnums=(2,))
_hermiticity_pmap = pops.pmap(
    _hermiticity_bare,
    in_axes=(0, None),
    out_axes=None,
    static_broadcasted_argnums=(2,),
)


def refine_polar_hermiticity(
    U,
    P,
    orig_dtype,
    p_sz=128,
    maxiter=10,
    lyapunov_method="CG",
    lyapunov_maxiter=200,
    max_rotation_order=5,
):
  """Refine a polar decomposition that is only Hermitian to low precision.

  The input should be a polar decomposition, i.e. matrices `U` and `P`, where
  `U` is unitary at the precision allowed by its dtype, and the decomposition is
  accurate (`U @ P` is the desired matrix) at that precision, but `P` is only
  Hermitian to a lower precision. This function will apply a small (almost
  identity) unitary `V` as `U -> U @ V^H`, `P -> V @ P`, to make `P` Hermitian
  at that precision.

  `refine_polar_hermiticity` can be applied to single-core matrices or to
  distributed matrices, but should not be called inside a pmap/jit.

  Args:
    U: The unitary factor of a polar decomposition.
    P: The polar factor, that is only Hermitian to a low precision.
    orig_dtype: `P` should be Hermitian to the precision allowed by this
      dtype.
    p_sz: Optional; SUMMA panel size. Only used if the matrices are distributed.
      128 by default.
    maxiter: Optional; The maximum number of rounds of refining to use. 10 by
      default.
    lyapunov_method: Optional; Method for solving the Lyapunov equation. Options
      are "Newton-Schultz" and "CG" for conjubuilding_block gradient. "CG by default.
    lyapunov_maxiter: Optional; Maximum number of iterations in the Lyapunov
      solver. 200 by default.
    max_rotation_order: Optional; Maximum order when Taylor expanding exp(A) to
      apply a unitary rotation. 5 by default.
  Returns:
    U, P: Polar decomposition where `U` is as unitary as before, and `U @ P`
      hasn't changed, but `P` is now Hermitian to the precision allowed by the
      dtype.
  """
  if type(U) != type(P):
    msg = f"Got mixed types for polar decomposition: {type(U)} and {type(P)}."
    raise TypeError(msg)
  if U.ndim != P.ndim:
    msg = ("Got different numbers of indices for polar decomposition matrices: "
           f"{U.ndim} and {P.ndim}.")
    raise TypeError(msg)
  if orig_dtype not in utils.valid_dtypes:
    msg = f"Invalid dtype {orig_dtype}."
    raise TypeError(msg)
  target_dtype = U.dtype
  if target_dtype == orig_dtype:
    return U, P

  distribution_type = pops.distribution_type(U)
  if distribution_type == "distributed":
    # Set precision to None, to make sure each call to backend.matmul has to
    # specify a precision.
    backend = distributed_backend.DistributedBackend(p_sz, precision=None)
    if lyapunov_method == "CG":
      solve_lyapunov = _solve_lyapunov_cg_pmap
    elif lyapunov_method == "Newton-Schultz":
      solve_lyapunov = _solve_lyapunov_ns_pmap
    else:
      raise ValueError(f"Unknown lyapunov_method: {lyapunov_method}")
    rotate_decomp = _rotate_decomp_pmap
    hermiticity = _hermiticity_pmap
    frobnorm = pops.pmap(backend.frobnorm, out_axes=None)
  elif distribution_type == "undistributed":
    # Set precision to None, to make sure each call to backend.matmul has to
    # specify a precision.
    backend = serial_backend.SerialBackend(precision=None)
    if lyapunov_method == "CG":
      solve_lyapunov = _solve_lyapunov_cg_jit
    elif lyapunov_method == "Newton-Schultz":
      solve_lyapunov = _solve_lyapunov_ns_jit
    else:
      raise ValueError(f"Unknown lyapunov_method: {lyapunov_method}")
    rotate_decomp = _rotate_decomp_jit
    hermiticity = _hermiticity_jit
    frobnorm = backend.frobnorm
  else:
    msg = "refine_polar_hermiticity should not be called inside a pmap/jit."
    raise RuntimeError(msg)

  dim = backend.shape(P)[0]
  P_norm = frobnorm(P)
  # Factor 4 just to give a bit of leeway.
  eps = 4 * 2**(-utils.mantissa_bits(target_dtype, jax.lax.Precision.HIGHEST))
  itercount = 0
  P_hermiticity = hermiticity(P, P_norm, backend)
  P_hermiticity_change = jnp.inf
  while (itercount < maxiter and P_hermiticity > eps and
         P_hermiticity_change > eps):
    A, _, _ = solve_lyapunov(
        P,
        jnp.float32,
        jax.lax.Precision.DEFAULT,
        lyapunov_maxiter,
        backend,
    )
    # The norm of A is transferred to the host machine, because it is used in
    # _rotate_decomp_preprocess to compute the necessary iterations and
    # precisions.
    A_norm = float(frobnorm(A))
    rotation_iterations = _rotate_decomp_preprocess(
        dim,
        A_norm,
        U.dtype,
        max_rotation_order,
    )
    U, P = rotate_decomp(U, P, A, rotation_iterations, backend)
    prev_P_hermiticity = P_hermiticity
    P_hermiticity = hermiticity(P, P_norm, backend)
    P_hermiticity_change = np.abs(P_hermiticity -
                                  prev_P_hermiticity) / P_hermiticity
    itercount += 1
  if P_hermiticity > eps:
    if itercount >= maxiter:
      msg = f"refine_polar_hermiticity hit maximum iterations ({maxiter})."
    elif P_hermiticity_change <= eps:
      msg = "refine_polar_hermiticity stalled before full precision."
    # TODO Use logging.warn instead.
    print(msg)
  return U, P


def _cast_undistributed(X, dtype):
  """Cast an undistributed matrix to a given dtype."""
  return X.astype(dtype)


_cast_distributed = pops.pmap(
    _cast_undistributed,
    static_broadcasted_argnums=(1,),
)


def _compute_P_undistributed(U, A):
  """Compute U^H @ A, for undistributed matrices, using full precision."""
  return jnp.dot(
      U.T.conj(),
      A,
      precision=jax.lax.Precision.HIGHEST,
  )


@functools.partial(pops.pmap, static_broadcasted_argnums=(2,))
def _compute_P_distributed(U, A, p_sz):
  """Compute U^H @ A, for distributed matrices, using full precision."""
  return summa.summa(
      U.conj(),
      A,
      p_sz,
      True,
      False,
      precision=jax.lax.Precision.HIGHEST,
  )


def refine_polar(
    U,
    A,
    p_sz=128,
    refine_hermiticity_kwargs={},
    refine_unitarity_kwargs={},
):
  """Refine the precision of a polar decomposition.

  Note that the arguments are the a low precision unitary factor and a high
  precision original matrix. The low precision polar factor is not needed.

  `refine_polar` can be applied to single-core matrices, or to distributed
  matrices. It should not be called inside a pmap/jit.

  Args:
    U: The unitary factor of a polar decomposition of `A`, in a low precision
      dtype.
    A: The matrix for which we want to compute the polar decomposition, in a
      high precision dtype.
    p_sz: Optional; SUMMA panel size. Only used if the matrices are distributed.
      128 by default.
    refine_hermiticity_kwargs: Optional; Keyword arguments to be passed to
      `refine_polar_hermiticity`. `{}` by default.
    refine_unitarity_kwargs: Optional; Keyword arguments to be passed to
      `refine_unitarity`. `{}` by default.
  Returns:
    U, P: Polar decomposition of `A` at the precision allowed by the dtype of
      `A`.
  """
  target_dtype = A.dtype
  orig_dtype = U.dtype
  if type(U) != type(A):
    msg = f"Got mixed matrix types in refine_polar: {type(U)} and {type(A)}."
    raise TypeError(msg)
  if U.ndim != A.ndim:
    msg = ("Got different numbers of indices for inputs of refine_polar: "
           f"{U.ndim} and {A.ndim}.")
    raise TypeError(msg)
  distribution_type = pops.distribution_type(U)
  if distribution_type == "undistributed":
    cast = _cast_undistributed
    compute_P = _compute_P_undistributed
  elif distribution_type == "distributed":
    cast = _cast_distributed
    compute_P = lambda U, A: _compute_P_distributed(U, A, p_sz)
  else:
    msg = "refine_polar should not be called inside a pmap/jit."
    raise RuntimeError(msg)

  if orig_dtype == jnp.bfloat16 and target_dtype == jnp.float64:
    # Go to float32 first as an intermediate step. This can save us some f64
    # matmuls.
    A_32 = cast(A, jnp.float32)
    U, _ = refine_polar(
        U,
        A_32,
        p_sz=p_sz,
        refine_hermiticity_kwargs=refine_hermiticity_kwargs,
        refine_unitarity_kwargs=refine_unitarity_kwargs,
    )
    del A_32
    orig_dtype = U.dtype
  U = refine_unitarity.refine_unitarity(
      U,
      target_dtype,
      p_sz=p_sz,
      **refine_unitarity_kwargs,
  )
  P = compute_P(U, A)
  U, P = refine_polar_hermiticity(
      U,
      P,
      orig_dtype,
      p_sz=p_sz,
      **refine_hermiticity_kwargs,
  )
  return U, P
