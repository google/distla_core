import functools
import jax.numpy as jnp
import numpy as np

from distla_core.blas.summa import summa
from distla_core.linalg.qr import qr
from distla_core.utils import pops


@functools.partial(pops.pmap, out_axes=None)
def _my_name_f(ps):
  return pops.my_name()


def _host_idx():
  ps = jnp.arange(pops.NDPROCS)
  p = _my_name_f(ps)
  return p // pops.NDPROCS


def _eye(global_rows, global_cols, dtype):
  ps = jnp.arange(pops.NDPROCS)
  m_l = global_rows // pops.NROWS
  n_l = global_cols // pops.NCOLS

  @pops.pmap
  def _eye_f(ps):
    return pops.eye((m_l, n_l), dtype)
  return _eye_f(ps)


@pops.pmap
def _vstack_f(A, B):
  return pops.vstack_equal_shape(A, B)


@pops.pmap
def _square(A):
  return summa.summa(A, A, 512, True, False)


@pops.pmap
def _gather_to_asic_nodes(A):
  return pops.gather_to_asic_nodes(A)


def qr_ooc(
    fname, caqr_panel_size=128, compute_q=False, dtype=np.float32):
  """ Computes an R factor of a QR decomposition of the matrix stored in fname.
  The matrix should have global shape M, N. M may be arbitrarily large, but N
  must be small enough to fit in the given slice. The file should be copied in
  full to all hosts.
  """
  matrix = np.load(fname, mmap_mode="r", allow_pickle=True)
  M, N = matrix.shape
  if M % N != 0:
    raise TypeError(f"N = {N} must evenly divide M = {M} for now.")
  if N % pops.NHROWS != 0:
    raise TypeError(f"N = {N} must be evenly divided by NHROWS={pops.NHROWS}.")
  if compute_q:
    raise NotImplementedError()
  n_blocks = M // N
  rows_per_host = N // pops.NHROWS
  cols_per_host = N // pops.NHCOLS
  host_idx = _host_idx()
  host_row, host_col = divmod(host_idx, pops.NDPROCS)
  host_row_offset = rows_per_host * host_row
  host_col_offset = cols_per_host * host_col

  trim_rows_matrix = _eye(N, 2 * N, dtype)

  @pops.pmap
  def _qr_f(A):
    _, R, _ = qr.qr(A, caqr_panel_size, compute_q=compute_q)
    return R

  @pops.pmap
  def _trim_rows_f(A, trim_rows_matrix):
    out = summa.summa(trim_rows_matrix, A, caqr_panel_size, False, False)
    return out

  def _load_block(block_index):
    ri = block_index * N + host_row_offset
    rf = ri + rows_per_host
    ci = host_col_offset
    cf = host_col_offset + cols_per_host
    block = matrix[ri:rf, ci:cf]
    block = pops.distribute(block)
    return _qr_f(block)

  R_top = _load_block(0)
  for i in range(1, n_blocks):
    R_bot = _load_block(i)
    R_bot = _vstack_f(R_top, R_bot)
    R_top = _qr_f(R_bot)
    R_top = _trim_rows_f(R_top, trim_rows_matrix)
  return R_top


def fake_cholesky(
    fname, caqr_panel_size=128, gram_fname="gram_matrix",
    chol_fname="chol_transpose"):
    """ Given an M x N matrix A stored on all hosts within the file named
    fname, computes the 'chol transpose' chol(A).T and 'gram matrix' A^T @ A.
    These are saved to all hosts in files named gram_fname + ".npy" and
    chol_fname + ".npy" respectively.

    Args:
      fname: Name of the file storing A.
      M, N: Shape of A.
      caqr_panel_size: Panel size for the QR factorizations.
    """

    chol_transpose = qr_ooc(
      fname, caqr_panel_size=caqr_panel_size, compute_q=False)

    gram_matrix = _square(chol_transpose)
    chol_transpose = _gather_to_asic_nodes(chol_transpose)
    chol_transpose = pops.undistribute(
      chol_transpose, host_local=False, collect_to_host=True)
    gram_matrix = _gather_to_asic_nodes(gram_matrix)
    gram_matrix = pops.undistribute(
      gram_matrix, host_local=False, collect_to_host=True)
    np.save(gram_fname, gram_matrix)
    np.save(chol_fname, chol_transpose)
