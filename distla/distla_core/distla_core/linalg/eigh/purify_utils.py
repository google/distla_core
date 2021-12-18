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
""" Functions to compute the projection matrix into specified eigenspaces
of a self_adjoint matrix.
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from distla_core.linalg.utils import testutils  # TODO: put `eps` somewhere else.
from distla_core.utils import misc
from distla_core.utils import pops


###############################################################################
# Initialization.
###############################################################################
def _to_high(entry):
  """ Brings the scalar `entry` to the host and converts it to fp64, unless
  it already is fp64, in which case it is converted to longdouble.
  """
  entry = np.full(1, entry)  # catches Python scalars
  if entry.dtype == np.float64:
    return np.full(1, entry, dtype=np.longdouble)
  return np.full(1, entry, dtype=np.double)


def _from_high(entry, dtype):
  """ Sends the numpy size-1 scalar to device and converts it back to dtype.
  """
  if entry.dtype == np.longdouble:
    entry = entry.astype(np.float64)
  entry = jnp.array(entry, dtype=dtype)
  return entry


def _canonical_initial_guess(self_adjoint, rank, backend, unpadded_dim,
                             eigenvalue_bounds, pmap):
  """ Constructs the initial value of `P` for the PM and hole-particle
  canonical purification.
  The guess is chosen to have eigenvalues in `[0, 1]` and trace = `rank`.
  Furthermore, the "mixing" method incorporating information from guess
  values for the projectors into both the negative and positive eigenspaces
  specified by `self_adjoint` and `rank` is used
  See eqns 2, 11, and 14 in https://aip.scitation.org/doi/10.1063/1.4943213.
  """
  trace_mean_f, to_projector_f, mix_projectors_f = _initial_guess_functions(
      pmap)
  mu = trace_mean_f(self_adjoint, unpadded_dim, backend)
  coefs = _initial_guess_coefs_on_host(mu, rank, unpadded_dim,
                                       eigenvalue_bounds, self_adjoint.dtype)
  P0, P0_bar, trace_P2, trace_PPbar, trace_Pbar2 = to_projector_f(
      self_adjoint, coefs, unpadded_dim, backend)
  alpha = _find_alpha_on_host(trace_P2, trace_PPbar, trace_Pbar2, rank,
                              unpadded_dim)
  projector = mix_projectors_f(P0, P0_bar, alpha, self_adjoint.dtype, rank,
                               unpadded_dim, backend)
  return projector


def _initial_guess_functions(pmap):
  """ Returns three functions needed to form the initial guess for
  hole-particle purification, pmapping or jitting depending on the
  flag pmap. The functions are:
    trace_mean_f: Computes the mean of the the input's trace.
    to_projector_f: Performs various matrix computations mapping
      the self_adjoint input to a projector and its orthogonal complement
      with the appropriate ranks.
    mix_projectors_f: Computes alpha * P0 + alpha_b * P0_bar,
      where alpha and alpha_b are mixing coefficients of the projector
      P0 and its orthogonal complement P0_bar. Adds
      (rank - trace(result)) / unpadded_dim to the diagonal of the
      result as a safeguard.
  """
  if pmap:
    trace_mean_f = pops.pmap(
        _initial_guess_trace_mean,
        static_broadcasted_argnums=(2,),
        in_axes=(0, None, None),
        out_axes=None)
    to_projector_f = pops.pmap(
        _initial_guess_projector,
        static_broadcasted_argnums=(3,),
        in_axes=(0, None, None, None),
        out_axes=(0, 0, None, None, None))
    mix_projectors_f = pops.pmap(
        _initial_guess_mix_projectors,
        static_broadcasted_argnums=(3, 6),
        in_axes=(0, 0, None, None, None, None, None))
  else:
    trace_mean_f = jax.jit(_initial_guess_trace_mean, static_argnums=(2,))
    to_projector_f = jax.jit(_initial_guess_projector, static_argnums=(3,))
    mix_projectors_f = jax.jit(
        _initial_guess_mix_projectors, static_argnums=(3, 6))
  return trace_mean_f, to_projector_f, mix_projectors_f


def _initial_guess_trace_mean(self_adjoint, unpadded_dim, backend):
  return backend.trace(self_adjoint) / unpadded_dim


def _initial_guess_projector(self_adjoint, coefs, unpadded_dim, backend):
  p_scale, p_shift, pbar_scale, pbar_shift = coefs
  P0 = -backend.add_to_diagonal(
      p_scale * self_adjoint, p_shift, unpadded_dim=unpadded_dim)
  P0_bar = -backend.add_to_diagonal(
      pbar_scale * self_adjoint, pbar_shift, unpadded_dim=unpadded_dim)
  trace_P2 = backend.sum(P0.conj() * P0)  # trace(A @ B) = sum(A.T * B)
  trace_PPbar = backend.sum(P0.conj() * P0_bar)
  trace_Pbar2 = backend.sum(P0_bar.conj() * P0_bar)
  return P0, P0_bar, trace_P2, trace_PPbar, trace_Pbar2


def _initial_guess_mix_projectors(P0, P0_bar, alpha, dtype, rank, unpadded_dim,
                                  backend):
  result = (alpha * P0 + (1 - alpha) * P0_bar).astype(dtype)
  tr_result = backend.trace(result)
  perturbation = (rank - tr_result) / unpadded_dim
  projector = backend.add_to_diagonal(result, perturbation)
  projector = projector.astype(dtype)
  return projector


def _initial_guess_coefs_on_host(mu, rank, unpadded_dim, eigenvalue_bounds,
                                 dtype):
  rank = int(rank)
  unpadded_dim = int(unpadded_dim)
  lambda_min = _to_high(eigenvalue_bounds[0])
  lambda_max = _to_high(eigenvalue_bounds[1])
  mu = _to_high(mu)
  theta = _to_high(int(rank) / int(unpadded_dim))
  beta = theta / (lambda_max - mu)
  beta_bar = (1 - theta) / (mu - lambda_min)
  if beta <= beta_bar:
    p_scale = beta
    pbar_scale = beta_bar
  else:
    p_scale = beta_bar
    pbar_scale = beta
  p_shift = -(theta + p_scale * mu)
  pbar_shift = -(theta + pbar_scale * mu)
  p_scale = _from_high(p_scale, dtype)
  p_shift = _from_high(p_shift, dtype)
  pbar_scale = _from_high(pbar_scale, dtype)
  pbar_shift = _from_high(pbar_shift, dtype)
  return p_scale, p_shift, pbar_scale, pbar_shift


def _find_alpha_on_host(trace_P2, trace_PPbar, trace_Pbar2, rank, unpadded_dim):
  """ Finds alpha in eqn 14 of
  https://aip.scitation.org/doi/10.1063/1.4943213.
  Given `P0 = D0` and `P0_bar = I - bar(D0)` in that equation, `alpha`
  is the solution to `Tr[(alpha * P0 + (1 - alpha) * P0_bar)^2] = k`
  where `k = rank - (2 / 3) * rank if N / rank <= 1 / 3`,
        `k = rank - (2 / 3) * (N - rank)` otherwise,
  under the constraint that `0 <= alpha <= 1`. This function forms the
  relevant quadratic formula and returns `alpha` with the constraint enforced.
  If the constraint cannot be enforced, this function returns `alpha = 1`,
  which implicitly reverts the initial guess to that given by eqns
  2 and 11 in https://aip.scitation.org/doi/10.1063/1.4943213.
  """
  rank = int(rank)
  unpadded_dim = int(unpadded_dim)
  dtype = trace_P2.dtype
  trace_P2 = _to_high(trace_P2)
  trace_PPbar = _to_high(trace_PPbar)
  trace_Pbar2 = _to_high(trace_Pbar2)
  theta = rank / unpadded_dim
  delta = 2 / 3
  if theta < (1 - delta):
    k = rank * (1 - delta)
  else:
    k = rank * (1 + delta) - unpadded_dim * delta

  a_coef = trace_P2 + trace_Pbar2
  b_coef = -2 * (trace_PPbar + trace_Pbar2)
  c_coef = trace_PPbar + trace_Pbar2 - k
  determinant = np.sqrt(b_coef**2 - 4 * a_coef * c_coef)
  root_1 = -(b_coef + np.sign(b_coef) * determinant) / (2 * a_coef)
  root_2 = c_coef / (a_coef * root_1)
  root_1_good = (root_1 >= 0 and root_1 <= 1)
  root_2_good = (root_2 >= 0 and root_2 <= 1)
  both_bad = not (root_1_good or root_2_good)

  if both_bad:
    alpha = 0.5
  elif root_1_good:
    alpha = root_1
  else:
    alpha = root_2
  return jnp.full(1, alpha, dtype=dtype)


###############################################################################
# Hole-particle purification.
###############################################################################
def _hole_particle_polynomial(projector, backend, unpadded_dim):
  """
  Performs one application of the hole-particle purification function to `P`.

  `P_bar = I - P`.
  `P -> P + 2 * (P^2 @ P_bar - coef * P @ P_bar)`
  `coef = Tr(P^2 @ P_bar) / Tr(P @ P_bar)`

  Theoretically this requires the storage of 3 matrices of the same
  size of P (including P itself), plus the internal overhead of the
  matrix multiply.
  """
  padded_eye = backend.eye_like(projector, unpadded_dim=unpadded_dim)
  proj_2bar = backend.matmul(projector, padded_eye - projector)
  proj_3bar = backend.matmul(projector, proj_2bar)
  tr_proj_2bar = jnp.abs(backend.trace(proj_2bar))
  err = tr_proj_2bar
  eps = jnp.finfo(projector.dtype).eps
  tr_proj_2bar = jnp.where(eps > tr_proj_2bar, x=eps, y=tr_proj_2bar)
  tr_proj_3bar = backend.trace(proj_3bar)
  coef = tr_proj_3bar / tr_proj_2bar
  projector += 2 * (proj_3bar - coef * proj_2bar)
  return projector, err


def _hole_particle_work(self_adjoint, tol, maxiter, backend, unpadded_dim):
  return _purify_work(_hole_particle_polynomial, self_adjoint, tol, maxiter,
                      backend, unpadded_dim)


###############################################################################
# Canonical purification interface.
###############################################################################
def _purify_work(purify_function, projector, tol, maxiter, backend,
                 unpadded_dim):
  errs = jnp.zeros(maxiter, dtype=projector.real.dtype)

  def _keep_purifying(carry):
    _, j, err, _, failure = carry
    out_of_time = j >= maxiter
    converged = err < tol
    stop_iterating = jnp.logical_or(jnp.logical_or(converged, out_of_time), failure)
    return jnp.logical_not(stop_iterating)[0]

  def _purify(carry):
    projector, j, _, errs, _ = carry
    projector, err = purify_function(projector, backend, unpadded_dim)
    errs = errs.at[j].set(err)
    failure = jnp.isnan(errs[j])
    return projector, j + 1, err, errs, failure

  j = jnp.zeros(1, dtype=jnp.int32)
  out = _purify((projector, j, errs[0], errs, False))
  projector, j, _, errs, failure = lax.while_loop(_keep_purifying, _purify, out)
  j = j[0]
  return projector, j, errs, failure


def canonically_purify(self_adjoint,
                       k_target,
                       backend,
                       tol,
                       maxiter,
                       method,
                       eigenvalue_bounds=None,
                       unpadded_dim=None):
  """
  Computes a projection matrix into the eigenspace sharing the smallest
  `k_target` eigenpairs as `self_adjoint`.

  Args:
    self_adjoint:  The self_adjoint matrix to be purified.
    k_target: The rank of the projector to be computed (number of electrons,
              in the DFT interpretation).
    backend: Backend object determining whether the serial or distributed
             implementation is used.
    tol: Convergence is declared if the idempotency error drops beneath this
         value.
    maxiter: Maximum number of iterations allowed.
    method: The purification method to use. Currently only "hole-particle" is
            supported.
    eigenvalue_bounds: Optional guess (eig_min, eig_max) such that eig_min
      (eig_max) is a close lower (upper) bound on the most negative and most
      positive eigenvalue of the unpadded `self_adjoint` respectively.
    unpadded_dim: Optional handling of padded matrices; the number of
      columns of `self_adjoint` which are not part of the pad.
  Returns:
    P: The projector.
    j: The number of iterations run, not including the NS steps.
    errs: Convergence history, including the NS steps.
  """
  self_adjoint_shape = backend.shape(self_adjoint)
  if unpadded_dim is None:
    unpadded_dim = self_adjoint_shape[1]

  if backend.name == "SerialBackend":
    decorator = functools.partial(jax.jit, static_argnums=(2, 3, 4))
    if eigenvalue_bounds is None:
      eigenvalue_bounds = jax.jit(backend.gershgorin)(self_adjoint)
    pmap = False

  elif backend.name == "DistributedBackend":
    decorator = functools.partial(
        pops.pmap,
        static_broadcasted_argnums=(2, 3, 4),
        in_axes=(0, None, None, None, None),
        out_axes=(0, None, None, None))
    if eigenvalue_bounds is None:
      eigenvalue_bounds = functools.partial(
          pops.pmap, out_axes=None)(backend.gershgorin)(self_adjoint)
    pmap = True

  else:
    raise ValueError(f"Invalid backend {backend.name}.")

  if tol is None:
    if backend.precision == lax.Precision.DEFAULT:
      coef = jnp.sqrt(unpadded_dim) * 0.5
    else:
      coef = unpadded_dim * 0.5
    tol = testutils.eps(backend.precision, dtype=self_adjoint.dtype) * coef

  if method != "hole-particle":
    raise NotImplementedError("Only method `hole-particle` is supported.")
  projector = _canonical_initial_guess(self_adjoint, k_target, backend,
                                       unpadded_dim, eigenvalue_bounds, pmap)
  work_f = _hole_particle_work
  out = decorator(work_f)(projector, tol, maxiter, backend, unpadded_dim)
  projector, j, errs, failure = out
  if failure:
    raise RuntimeError(f"canonically_purify returned NaN at iteration {j}.")

  errs = errs[:j]
  return projector, j, errs
