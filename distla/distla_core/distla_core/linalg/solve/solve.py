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
""" DistlaCore linear solvers.
"""

from jax import lax
import jax.numpy as jnp
import jax.scipy as jsp

from distla_core.linalg.inv import inv
from distla_core.linalg.utils import testutils
from distla_core.utils import pops
from distla_core.utils import vops

#  Variables are named according to the following convention:
#   Capitalized and neither of below: Checkerboard (big x big) matrix.
#   Lowercase and neither of below: Not a matrix or vector.
#   _v: ReplicatedThin (big x small) matrix or vector; the small dimension
#       is replicated over processors.
#   _s: Small (small x small) matrix; the entire matrix is replicated over
#       processors.


##############################################################################
# Givens rotation code. All of this is serial.
##############################################################################
def _compute_givens_rotation(a, b):
  """ a and b are the components of a two-vector. This function computes
  the cosine and sine factors of a "Givens" rotation matrix that zeroes
  out b.
  """
  b_zero = abs(b) == 0
  a_lt_b = abs(a) < abs(b)
  t = -jnp.where(a_lt_b, a, b) / jnp.where(a_lt_b, b, a)
  r = lax.rsqrt(1 + abs(t) ** 2)
  cs = jnp.where(b_zero, 1, jnp.where(a_lt_b, r * t, r))
  sn = jnp.where(b_zero, 0, jnp.where(a_lt_b, r, r * t))
  return cs, sn


def _apply_ith_rotation(i, args):
  """ Applies the two-dimensional ("Givens") rotation matrix specified by
  cs[i] and sn[i] (the cosine and sine factors of the rotation) to the
  two-vector specified by the ith and i+1th components of h. Returns
  the rotated vector along with the unmodified cs and sn, which is
  necessary for Jax typing reasons.
  """
  h, cs, sn = args
  x1 = h[i]
  y1 = h[i + 1]
  x2 = cs[i].conj() * x1 - sn[i].conj() * y1
  y2 = sn[i] * x1 + cs[i] * y1
  h = h.at[i].set(x2)
  h = h.at[i + 1].set(y2)
  return h, cs, sn


def _update_hessenberg_qr(H_r, H, cs, sn, j):
  """ H[:j, :j] is an upper Hessenberg matrix, H_r[:j, :j] the R
  factor in a QR decomposition of it, and (cs[:j-1], sn[:j-1])
  the Givens rotations used to compute that QR decomposition (thus an implicit
  representation of the Q factor). This function updates H_r, cs, and
  sn by a column, so that each j in the above gets incremented by 1.
  That is, the returned (H_r[:j+1, :j+1], cs, sn) are the R factor and
  Givens rotation components specifying a QR decomposition of H[:j+1, :j+1].
  """
  h = H[:, j]
  h, _, _ = lax.fori_loop(0, j, _apply_ith_rotation, (h, cs, sn))
  cs_j, sn_j = _compute_givens_rotation(h[j], h[j + 1])
  cs = cs.at[j].set(cs_j)
  sn = sn.at[j].set(sn_j)
  h, _, _ = _apply_ith_rotation(j, (h, cs, sn))
  H_r = H_r.at[:, j].set(h)
  return H_r, cs, sn


##############################################################################
# Arnoldi process.
##############################################################################
def _arnoldi_cond(maxiter, tol, arnoldi_args):
  _, _, _, _, _, _, err, j = arnoldi_args
  return jnp.logical_and(err > tol, j < maxiter)[0]


def cgs(new_v, V_v, precision=lax.Precision.HIGHEST, n_iter=2):
  """ Iterated classical Gram-Schmidt orthogonalization.
  new_v is rotated and scaled into orth_v such that hstack(V_v, orth_v)
  has orthonormal columns (provided the input V_v did).
  Returns orth_v and a vector of its overlaps with the other vectors in V_v.

  Though numerically unstable, CGS is preferred to MGS on ASICs due to its
  capacity for vectorization. The instability is ameliorated by iterating
  the latter a fixed n_iter number of times.

  Args:
    new_v: ReplicatedThinMatrix (N, 1) representing the vector to be
      orthonormalized against V_v. A future implementation may allow for
      second dimensions other than 1.
    V_v: ReplicatedThinMatrix (N, N_k) representing the (possibly zero-padded)
      bundle of vectors against which new_v is to be orthonormalized.
      V_v is assumed to have either orthonormal columns or columns of zeros.
      This is not checked.
    precision: ASIC matmul precision.
    n_iter: Number of iterations of CGS to perform. Typically n_iter=2 should
      be necessary and sufficient, but more (less) may be required if the
      nonzero block of V_v is very poorly (well) conditioned.
  Returns:
    orth_v: ReplicatedThinMatrix (N, 1) representing the orthonormalized new_v.
    overlaps: A length (N_k) vector whose j'th component is
      vdot(V_v[j], new_v).
    norm_v: The Frobenius norm of new_v.
  """
  overlaps = vops.vecvec(V_v.conj(), new_v, precision=precision)
  new_v -= vops.vecsmall(V_v, overlaps, precision=precision)
  for _ in range(n_iter - 1):
    d_overlaps = vops.vecvec(V_v.conj(), new_v, precision=precision)
    new_v -= vops.vecsmall(V_v, d_overlaps, precision=precision)
    overlaps += d_overlaps
  norm_v = vops.frobnorm(new_v)
  orth_v = new_v / norm_v
  return orth_v, overlaps, norm_v


def _update_arnoldi(V_v, H_s, j, orth_v, overlaps, norm_v):
  """ Updates V_v so that V_v[:, j + 1] = orth_v, and H_s so that
  H_s[:, j] = (overlaps, norm_v). If norm_v < eps, zeros are added
  instead.
  """
  not_zero = jnp.sqrt(norm_v) > jnp.finfo(orth_v.dtype).eps
  orth_v_dat = jnp.where(
    not_zero, x=orth_v.array, y=jnp.zeros_like(orth_v.array))
  orth_v = vops.ReplicatedThinMatrix(orth_v_dat, orth_v.is_column_replicated)
  norm_v = jnp.where(not_zero, x=norm_v, y=jnp.zeros_like(norm_v))
  overlaps = lax.dynamic_update_slice(
    overlaps, norm_v.reshape((1, 1)), (j + 1, 0))
  H_s = lax.dynamic_update_slice(H_s, overlaps, (0, j))
  V_v = vops.set_columns(V_v, orth_v, j + 1)
  return V_v, H_s


def _arnoldi_step(j, V_v, H_s, A, A_inv, precision):
  """ Performs an iteration of the Arnoldi process:
    - A new Krylov vector new_v = (A @ A_inv) @ V_v[:, j] is computed.
    - new_v is orthogonalized against the columns of V_v, yielding
      the orthogonalized vector orth_v along with new overlaps.
    - orth_v is stored in V_v[:, j+1] and the overlaps in H_s[:, j].
  We should have
    `A @ V_v[:, :j] = V_v[:, :j+1] @ H_s[:j+1, :j]`
  along with
    `H_s[:j+1, :j] = V_v[:, :j]^H @ A @ V_v[:, :j]`.
  """
  new_v = vops.get_columns(V_v, j, 1)
  if A_inv is not None:
    new_v = vops.matvec(A_inv, new_v, precision=lax.Precision.DEFAULT)
  new_v = vops.matvec(A, new_v, precision=precision)
  orth_v, overlaps, v_norm = cgs(new_v, V_v, precision=precision)
  V_v, H_s = _update_arnoldi(V_v, H_s, j, orth_v, overlaps, v_norm)
  return V_v, H_s


def _arnoldi_qr(
    A, arnoldi_args, precision=lax.Precision.HIGHEST, A_inv=None):
  """ Performs a single step of the Arnoldi process, with specializations
  to update a QR decomposition of the overlap matrix at each iteration. The
  decomposition is used to continually update the GMRES residual
  norm at each iteration, allowing for early termination in the common event
  that GMRES converges before the Arnoldi process does.

  Args:
    A:  The matrix from which to build the Krylov space.
    arnoldi_args = (V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j) s.t.
      V_v: Unitary basis of the Krylov space. At iteration 0 its first
        logical column stores the vector used to seed the space.
      H_s: Upper Hessenberg matrix of Krylov overlaps.
      R_s: R factor in a QR decompositon of H_s.
      beta_s: At iteration 0, a vector [residual_norm, 0, 0, 0,...];
        subsequent iterations later rotate this vector by the same Givens
        transformations used to produce R_s from H_s. As a
        far-from-obvious consequence, at the end of each iteration j the
        j + 1'th entry of this vector stores the current residual norm.
      cos_s: Stores the cosine Givens factors used to produce R_s
        from H_s. Together with sin_s this forms an implicit representation
        of the H_s' Q factor corresponding to R_s.
      sin_s: Stores the sine Givens factors used to produce R_s from
        H_s.
      err: The GMRES residual norm at each iteration / b_norm.
      j_arr: Iteration count; the subscript arr refers to this having been
        explicitly cast as a Jax array.
    A_inv: Optional estimate for the inverse of A.
    precision: ASIC matmul precision. Defaults to HIGHEST.
  Returns:
    The updated arnoldi_args.
  """
  V_v, H_s, R_s, beta_s, cos_s, sin_s, _, j_arr = arnoldi_args
  j = j_arr[0]
  V_v, H_s = _arnoldi_step(j, V_v, H_s, A, A_inv, precision)
  R_s, cos_s, sin_s = _update_hessenberg_qr(
    R_s, H_s, cos_s, sin_s, j)
  beta_s, _, _ = _apply_ith_rotation(j, (beta_s, cos_s, sin_s))
  err = jnp.full(1, jnp.abs(beta_s[j + 1]), dtype=H_s.real.dtype)
  return V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j_arr + 1


##############################################################################
# GMRES.
##############################################################################
def _gmres_init(A, B_v, arnoldi_maxiter, tol, precision, A_inv):
  n_rows = A.shape[0] * pops.GRID[0]
  if A_inv is None:
    X_v = B_v.zeros_like()
  else:
    X_v = vops.matvec(A_inv, B_v, precision=lax.Precision.DEFAULT)
  R_v = B_v - vops.matvec(A, X_v, precision=precision)
  b_norm = vops.frobnorm(B_v)
  r_norm = vops.frobnorm(R_v)
  if tol is None:
    tol = testutils.eps(precision, A.dtype) * b_norm
  err = jnp.full(1, r_norm, dtype=A.real.dtype)
  beta_s = jnp.zeros(arnoldi_maxiter + 1, dtype=A.dtype)
  beta_s = beta_s.at[0].set(r_norm)

  V_v = vops.zeros(
    (n_rows, arnoldi_maxiter + 1), dtype=A.dtype)  # Krylov basis
  V_v = vops.set_columns(V_v, R_v / r_norm, 0)
  H_s = jnp.eye(
    arnoldi_maxiter + 1, arnoldi_maxiter, dtype=A.dtype)  # Overlaps
  R_s = H_s  # R factor in Harn = Qarn @ Rarn
  sin_s = jnp.zeros(arnoldi_maxiter, dtype=A.dtype)  # Implicit Qarn
  cos_s = jnp.zeros_like(sin_s)

  def cond_f(args):
    return _arnoldi_cond(arnoldi_maxiter, tol, args)

  def arnoldi_f(args):
    return _arnoldi_qr(A, args, A_inv=A_inv, precision=precision)

  j = jnp.zeros(1, dtype=jnp.int32)
  arnoldi_args = (
    V_v, H_s, R_s, beta_s, cos_s, sin_s, err, j)
  return X_v, arnoldi_args, cond_f, arnoldi_f, b_norm


def _gmres_update_solution(X_v, V_v, R_s, beta_s, A_inv, arnoldi_maxiter):
  """ Given the current solution estimate X_v and the Krylov subspace seeeded
  from that estimate's residual specified by V_v, R_s, beta_s, and
  arnoldi_maxiter, solves the GMRES least squares problem. Returns the closest
  approximation to the true solution available within the given Krylov space.
  """
  Y_s = jsp.linalg.solve_triangular(
    R_s[:arnoldi_maxiter, :arnoldi_maxiter], beta_s[:arnoldi_maxiter])
  Vt_v = vops.get_columns(V_v, 0, arnoldi_maxiter)
  dX_v = vops.vecsmall(Vt_v, Y_s)
  if A_inv is not None:
    dX_v = vops.matvec(A_inv, dX_v, precision=lax.Precision.DEFAULT)
  X_v = X_v + dX_v
  return X_v


def gmres(
  A, B_v, arnoldi_maxiter=24, tol=None, precision=lax.Precision.HIGHEST,
    A_inv=None):
  """ GMRES returns X in A @ X = B given A and B. This function should
  be pmapped with arnoldi_maxiter and precision declared static.

  Args:
    A: The N x N checkerboard matrix `A`.
    B_v: The N x k ThinMatrix `B`. Presently `k` must be 1.
    arnoldi_maxiter: Maximum number of Arnoldi iterations allowed to the
      unrestarted GMRES solver. Heuristically, when Newton-Schulz
      preconditioning is used only around 5 should be required to reach
      single-precision accuracy in practice, but this has not been tested
      thoroughly.
    tol: The unrestarted GMRES solver will terminate when
      `norm(residual) <= tol * norm(b)`. Machine epsilon is used if this is
      unspecified.
    precision: ASIC matmul precision.
    A_inv: An optional approximation to A^(-1) used to accelerate
      the unrestarted GMRES solver. `right` preconditioning is used, so
      that the solver implicitly solves
      `A @ A_inv @ u = B`, `x = A_inv @ u`. If `None`, no preconditioner is
      used.
  Returns:
    X_v: The N x k ThinMatrix solution `X`.
    j: The number of GMRES iterations.
    err: norm(residual) / b_norm.
  """
  if B_v.shape[1] != 1:
    raise NotImplementedError("Multiple right hand sides not yet supported.")

  X_v, arnoldi_args, arnoldi_cond_f, arnoldi_body_f, b_norm = _gmres_init(
    A, B_v, arnoldi_maxiter, tol, precision, A_inv
  )
  arnoldi_args = lax.while_loop(arnoldi_cond_f, arnoldi_body_f, arnoldi_args)
  V_v, _, R_s, beta_s, _, _, err, j = arnoldi_args
  X_v = _gmres_update_solution(X_v, V_v, R_s, beta_s, A_inv, arnoldi_maxiter)
  return X_v, j, err / b_norm


##############################################################################
# Solve interface.
##############################################################################
def solve(A, B, arnoldi_maxiter=24, tol=None, precision=lax.Precision.HIGHEST,
          A_inv=None, p_sz=1024):
  """ solve returns X in A @ X = B given A and B. This function should be
  pmapped with arnoldi_maxiter, precision, and p_sz declared static.

  Args:
    A: The N x N checkerboard matrix `A`.
    B: The N x k ThinMatrix `B`. Presently `k` must be 1.
    arnoldi_maxiter: Maximum number of Arnoldi iterations allowed to the
      unrestarted GMRES solver. Heuristically, when Newton-Schulz
      preconditioning is used only around 5 should be required to reach
      single-precision accuracy in practice, but this has not been tested
      thoroughly.
    tol: The unrestarted GMRES solver will terminate when
      `norm(residual) <= tol * norm(b)`. Machine epsilon is used if this is
      unspecified.
    precision: ASIC matmul precision.
    A_inv: An optional approximation to A^(-1) used to accelerate
      the unrestarted GMRES solver. `right` preconditioning is used, so
      that the solver implicitly solves
      `A @ A_inv @ u = B`, `x = A_inv @ u`. If `None`,
      a preconditioner is computed using low precision Newton-Schulz inversion.
      The preconditioner will be returned in either case, so that it can be
      recycled should later solves with the same `A` be required.
    p_sz: SUMMA matmul panel size.
  Returns:
    X: The N x k ThinMatrix solution `X`.
    j: The number of GMRES iterations.
    err: norm(residual) / norm(b).
    A_inv: The preconditioner.
  """
  if A_inv is None:
    A_inv, _, _ = inv.inv(
      A, left=False, precision=lax.Precision.DEFAULT, p_sz=p_sz)
  X, j, err = gmres(
    A, B, arnoldi_maxiter=arnoldi_maxiter, tol=tol, precision=precision,
    A_inv=A_inv)
  return X, j, err, A_inv
