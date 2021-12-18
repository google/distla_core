import functools
import jax
from jax import lax
import jax.numpy as jnp
import math

from distla_core.utils import misc
from distla_core.utils import pops
from distla_core.utils import vops


##############################################################################
# TSQR
##############################################################################
def _assemblage_q(
    Qold, Qnew, prow_group, compute_q, precision=lax.Precision.HIGHEST):
  """ Collects the total Q factor as tsqr proceeds. This requires that the
  appropriate half of the new Q factor Qnew be masked. This is achieved by
  the variable prow_group, tracking which processors need the top and bottom
  piece respectively. While this could be computed on the fly as
  (prow // (2 ** recursion_step)) % 2,  we instead track the former factor
  iteratively in order to avoid forming a possibly large denominator
  should recursion_step ever grow large.
  """
  if not compute_q:
    return None, None
  if prow_group is None:
    prow_group = pops.my_prow()
  else:
    prow_group = prow_group // 2
  N = Qnew.shape[1]  # Qnew.shape[0] is 2N
  take_lower_half = prow_group % 2
  Qnew = jnp.where(take_lower_half, x=Qnew[N:, :], y=Qnew[:N, :])
  Qnew = jnp.dot(Qold, Qnew, precision=precision)
  return Qnew, prow_group


def tsqr(A: vops.ReplicatedThinMatrix, compute_q=False):
  """ TSQR computes the QR decomposition of the ReplicatedThinMatrix A.
  It is assumed the matrix is in fact "thin"; i.e. that it has many more
  mathematical rows than columns.

  Args:
    A: (M, N) ReplicatedThinMatrix whose QR decomposition is to be computed.
    explicit_q (bool): Determines whether the Q factor will be returned
      implicitly as a list of elements of a  tree representation (if False)
      or explicitly as a ReplicatedThinMatrix (if True).
  Returns:
    Q: None (if compute_q is False), or a ReplicatedThinMatrix (True)
      representing an (M, N) "reduced" Q factor of A.
    R: (N, N) "small" matrix (replicated on all cores) representing the
      corresponding R factor.
  """
  # As a starting point we have the matrix `A` divided into panels, one
  # per prow: A = [A0, A1, A2, A3]^T. `A` is `M, N`, and each `Ai`
  # `m_l = M // nrows, N`; it is assumed that m_l <= N. Note that `N`
  # is also the local column size since `A` is a ReplicatedThinMatrix.
  #
  # The first stage computes reduced QR factorizations of each Ai. Each
  # processor now has `m_l, N` `Qi` and `N, N` `Ri`:
  # Q = [Q0, Q1, Q2, Q3]^T, R = [R0, R1, R2, R3]^T. The Q factor is
  # discarded if compute_q is False.
  #
  # Next, each Ri is gathered and stacked between pairs of prows,
  # initially starting from 0 (e.g. prow 0 is the "top" of its pair
  # with prow 1): R = [R01, R01, R23, R23]^T,
  # where each (2N, N) Rij = vstack([R0, R1]). Another local QR is
  # performed on each R factor, resulting in another (N, N) R factor:
  # R = [R00, R00, R22, R22]^T.
  #
  # If compute_q is True, the "top" ("bottom") processor in its pair
  # extracts the corresponding (N, N) block of its (2N, N) Q factor,
  # and multiplies its previous Q factor by the result:
  # Q = [Q0@=Q00[:N, :], Q1@=Q00[N:, :], Q2@=Q22[:N, :], Q2@=Q22[N:, :]]^T.
  #
  # The process is now repeated, choosing pairs of prows which this time
  # start from 1 (e.g. prow 1 is now the "top" of its pair with
  # prow 2): R = [R2200, R1122, R1122, R2200]^T. The steps are repeated
  # until each prow contains a copy of the same R factor, which is that of the
  # original A. If compute_q was True, each prow also now contains one
  # panel of the corresponding Q factor, stored as a column-replicated
  # ReplicatedThinMatrix of the same dimensions as A.

  if not misc.is_power_of_two(pops.NROWS):
    raise TypeError(f"{pops.NROWS} must be a power of two.")
  n_steps = int(round(math.log2(pops.NROWS)))

  R2 = vops.to_column_replicated(A).array  # m_l, N
  m_l, N = R2.shape
  if N > m_l:
    raise TypeError(f"A must have fewer columns ({N}) than local rows "
                    f"({m_l}).")
  # Each tuple contains the prows which will hold the same data at the end
  # of the next step.
  # If Jit overhead becomes a problem, note it should be possible to build
  # groups separately as static data, and write the rest as a fori.
  # It hopefully should be ok since n_steps grows only
  # logarithmically with NROWS, and is still only 6 on a full asic_cluster.
  groups = tuple([(g,) for g in range(pops.NROWS)])
  Q, R = jnp.linalg.qr(R2, mode="reduced")
  prow_group = None
  for i in range(n_steps):
    idx_in_new_group = groups[0][-1]
    groups = [gl + gr for gl, gr in zip(groups[::2], groups[1::2])]
    # Merge each pair of groups. E.g. with 8 rows as i proceeds:
    # i = 0: groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    #   We gather e.g. 0 with 1 via gather_prow_pairs, start_from_zero True.
    #   No broadcast is needed at this stage.

    # i = 1: groups = ((0, 1, 2, 3), (4, 5, 6, 7))
    #   We gather e.g. 1 with 2 via gather_prow_pairs, start_from_zero False.
    #   However, this also performs gathers we do not need. To erase them,
    #   we also broadcast 1 and 5 (the final prow in the first of each newly
    #   merged group) to the rest of their new group.

    # i = 2: groups = ((0, 1, 2, 3, 4, 5, 6, 7))
    #   gather_prow_pairs, start_from_zero False. Then broadcast 3.
    R2 = pops.gather_prow_pairs(R, start_from_zero=i == 0)  # 2n, n
    if i > 0:
      bcast_indices = [idx_in_new_group for _ in range(len(groups))]
      R2 = pops.broadcast_paxis_to_groups(R2, bcast_indices, groups, rows=True)
    Qi, R = jnp.linalg.qr(R2, mode="reduced")  # n, n
    # Q is None if compute_q is False
    Q, prow_group = _assemblage_q(Q, Qi, prow_group, compute_q)

  if compute_q:
    Q = vops.ReplicatedThinMatrix(Q, is_column_replicated=True)
  return Q, R


##############################################################################
# WY Representation
##############################################################################
def _yamamoto(Q_reduced, pad):
  """ Converts from the reduced (pad + m, n) Q factor, Q_reduced ,in a QR
  decomposition of a thin (pad + m, n) matrix to the W and Y factors
  formed from Yamamoto's basis-kernel representation.

  The first `pad` rows of Q_reduced are assumed to be zero. Undefined behaviour
  will obtain otherwise. This is not checked.

  The basis-kernel representation is:
  Q (m x m) S (m x m) = I (m x m) - W_s (m x n) @ S @ T (n x n) @ B^H (n x m),
  where S is a diagonal sign matrix.

  With Q_reduced partitioned into A = Q_reduced @ R = [Q1] @ R, Q1 and R
                                                      [Q2]
  (n x n) and Q2 (m - n, n), we have
  W = Q_reduced - S
  T^-H = S - Q1 = W[:n, :]
  S is a diagonal sign matrix, with diagonal entries +1 by default, or -1 if
  this would give T^-H a zero on its main diagonal. This catches certain
  cases (e.g. Q1 = I) where T^-H would otherwise be singular.

  NOTE: The computation of S is not yet implemented. Therefore, it is possible
  in rare cases for this routine to compute singular T^-H, which will manifest
  as NaN entries in the returned Y. Code involving this function should
  include an error trap for such entries at the earliest un-Jitted point.

  Y is then computed as -T @ W, so that Q = I - W Y^H.
  W and Y are returned as (pad + m, n) ReplicatedThinMatrices, with the
  first pad rows zero. Note Y[pad:pad+m, :] = -I.
  """
  _, n = Q_reduced.shape
  # TODO: account for S
  W = vops.add_to_diagonal(Q_reduced, -1, k=-pad)
  Tinv = -pops.get_rows(vops.to_column_replicated(W).array, pad, n).conj().T

  # TODO: In a future implementation we should consider handling the -I block
  # implicitly rather than actually storing the -1 entries.
  Y = W.zeros_like()
  Y = vops.add_to_diagonal(Y, -1, k=-pad)
  Y_rows, _ = vops._indices_vec(Y)

  # TODO: A future implementation could operate only on W2. In the distributed
  # case it is not trivial doing this in a way that actually saves FLOPs,
  # however, and this appears not to be an important expense.
  Y2 = jnp.linalg.solve(Tinv, W.array.T).T
  Y_dat = jnp.where(Y_rows[:, None] >= pad + n, x=Y2, y=Y.array)
  Y = vops.ReplicatedThinMatrix(Y_dat, W.is_column_replicated)
  return W, Y


def _apply_wy_left(W, Y, A):
  """ Given Q = 1 - W @ Y^H with W = Q_wy[0] and Y = Q_wy[1],
  returns Q^H @ A = A - Y @ W^H @ A.
  """
  X = vops.vec_t_mat(W.conj(), A)
  X = vops.outer(Y, X)
  return A - X


def _apply_wy_right(W, Y, A):
  """ Given Q = 1 - W @ Y^H with W = Q_wy[0] and Y = Q_wy[1],
  returns A @ Q = A - A @ W @ Y^H.
  """
  X = vops.matvec(A, W)
  X = vops.outer(X, Y.conj())
  return A - X


##############################################################################
# Panel factorization
##############################################################################
@functools.partial(jax.jit, static_argnums=(2,))
def _factor_panel(A, column_idx, panel_size):
  """ Computes a reduced QR factorization of
  A[column_idx:, column_idx:column_idx + panel_size], by
  extracting panel = A[:, column_idx:column_idx + panel_size],
  masking panel[:column_idx, :] with zeros, and factoring the result.
  """
  panel = vops.get_columns(A, column_idx, panel_size)
  panel_rows, _ = vops._indices_vec(panel)
  masked_dat = jnp.where(panel_rows[:, None] >= column_idx,
                         x=panel.array, y=panel.zeros_like().array)
  masked_panel = vops.ReplicatedThinMatrix(
    masked_dat, panel.is_column_replicated)
  Q_panel, R_panel = tsqr(masked_panel, compute_q=True)
  return Q_panel, R_panel


@functools.partial(jax.jit, static_argnums=(2,))
def _triu_panel(A, column_idx, panel_size):
  """ Zeroes out the entries of A's coumn_idx'th column panel
  beneath its main diagonal.
  """
  rows, cols = pops.indices(A.shape)
  under_diagonal = rows > cols
  in_this_panel = jnp.logical_and(
    cols >= column_idx, cols < column_idx + panel_size)
  A = jnp.where(jnp.logical_and(under_diagonal, in_this_panel),
                x=jnp.zeros_like(A), y=A)
  return A


@functools.partial(jax.jit, static_argnums=(3,))
def qr_step(Q, A, column_idx, panel_size, first_failure):
  """ An iteration of `qr` over the column_idx'th column panel.
  The return value `first_failure` is -1 as long as the run has been
  successful, and the count of the first iteration at which a failure
  is detected otherwise.
  """
  Q_panel, _ = _factor_panel(A, column_idx, panel_size)
  W, Y = _yamamoto(Q_panel, column_idx)
  failed = jnp.any(jnp.isnan(Y.array))
  failed = lax.pmax(failed, axis_name=pops.AXIS_NAME)
  first_failure = jnp.where(
    jnp.logical_and(first_failure == -1, failed == 1),
    x=column_idx, y=first_failure)
  A = _apply_wy_left(W, Y, A)
  A = _triu_panel(A, column_idx, panel_size)
  if Q is not None:
    Q = _apply_wy_right(W, Y, Q)
  column_idx += panel_size
  return Q, A, column_idx, first_failure


##############################################################################
# Full QR
##############################################################################
@functools.partial(jax.jit, static_argnums=(1, 2))
def _qr(A, panel_size, compute_q):
  """ Work function for qr.
  """
  m_l, n_l = A.shape

  if n_l % panel_size != 0:
    raise ValueError(f"panel_size = {panel_size} must evenly divide "
                     f"local cols = {n_l}.")
    # TODO: handle the final panel.
  n_panels = n_l * pops.NCOLS // panel_size

  if panel_size > m_l:
    raise ValueError(f"A.shape = {A.shape} must have local rows >= "
                     f"panel_size = {panel_size}.")

  if compute_q:
    Q = pops.eye((m_l, m_l * pops.NROWS // pops.NCOLS), dtype=A.dtype)
  else:
    Q = None
  R = A

  def _qr_f(step_idx, args):
    Q, A, column_idx, first_failure = args
    return qr_step(Q, A, column_idx, panel_size, first_failure)

  args_0 = (Q, R, 0, jnp.full(1, -1, dtype=jnp.int32))
  Q, R, _, first_failure = lax.fori_loop(0, n_panels, _qr_f, args_0)
  return Q, R, first_failure


def qr(A, panel_size=128, compute_q=True):
  """ Computes a QR factorization of the checkerboard distributed matrix A.

  Args:
    A:  (M, N) matrix whose QR factorization is to be computed.
    panel_size: Specifies the size of the thin QR panel factorizations
      from which the full QR is formed. We must have N % panel_size = 0
      and (M // NROWS) <= panel_size.

      panel_size ends up as one of the dimensions in the internal matmuls,
      and thus should be a multiple of 128 (128 seems to work well) for
      optimal performance.
    compute_q: The Q factor is computed only if True.

    This function should be pmapped with panel_size and compute_q static.
  Returns:
    Q:  If compute_q, a Q factor, another checkerboard-distributed matrix.
        Otherwise None.
    R:  The corresponding R factor, also checkerboard-distributed.
    first_failure:  -1 if the factorization was successful, or the iteration
      at which a failure was first detected otherwise. Presently, failure is
      most likely due to lack of support in _yamamoto for computation of a
      certain matrix S, which can cause NaN output in some rare cases.
      Support for this will be added in a future implementation, but more
      quickly so if you run into this problem, so please ping Adam if so.
  """
  return _qr(A, panel_size, compute_q)
