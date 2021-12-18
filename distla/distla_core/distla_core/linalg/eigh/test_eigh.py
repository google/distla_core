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
"""
Tests for the distributed eigh solver.
"""
import pytest
import random

from jax import lax
import jax.numpy as jnp
import numpy as np

from distla_core.linalg.eigh import eigh
from distla_core.utils import misc
from distla_core.utils import pops
from distla_core.linalg.utils import testutils

Ns = [24, 48]
# TODO: Test fat SVD
m_factors = [1, 2]
p_szs = [256, ]
seeds = [1, ]
precisions = [lax.Precision.HIGHEST, ]
bools = [False, True]
padding = [None, 4]
minimum_ranks = [4, 48]


def _subspace_angle(subspace_1, subspace_2, orth1=True, orth2=True):
  if orth1:
    subspace_1, _ = np.linalg.qr(subspace_1)
  if orth2:
    subspace_2, _ = np.linalg.qr(subspace_2)

  product = np.dot(subspace_1.conj().T, subspace_2)
  return np.arccos(np.linalg.norm(product, ord=-2))


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("m_factor", m_factors)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_similarity_transform(N, m_factor, p_sz, precision):
  """
  Tests that eigh._similarity_transform indeed computes B^H @ A @ B, by
  comparing up against jnp.dot at the given precision up to
  eps * |B|_F^2 |A|_F.
  Here A is (N * m_factor, N * m_factor) and B is (N * m_factor, N).
  """
  np.random.seed(10)
  A = np.random.randn(N * m_factor, N * m_factor).astype(np.float32)
  B = np.random.randn(N * m_factor, N).astype(np.float32)
  normA = np.linalg.norm(A)
  normB = np.linalg.norm(B)

  C = jnp.dot(A, B, precision=precision)
  C = jnp.dot(B.conj().T, C, precision=precision)

  A = pops.distribute(A)
  B = pops.distribute(B)

  @pops.pmap
  def test_f(A, V):
    return eigh._similarity_transform(A, V, p_sz, precision=precision)

  result = test_f(A, B)
  result = pops.undistribute(result)
  atol = normB * normA * normB * testutils.eps(precision)
  np.testing.assert_allclose(C, result, atol=atol)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("canonical", bools)
@pytest.mark.parametrize("seed", seeds)
def test_split_spectrum(N, p_sz, precision, canonical, seed):
  """ Tests that split spectrum correctly divides a Hermitian matrix into blocks.
  """
  np.random.seed(seed)
  H = np.random.randn(N, N).astype(np.float32)
  H = jnp.array(0.5 * (H + H.conj().T))

  ev_exp, eV_exp = jnp.linalg.eigh(H)
  Hd = pops.distribute(H)

  @pops.pmap
  def median_ev_func(H):
    return pops.trace(H) / N

  if canonical:
    sigma = N // 2
  else:
    sigma = median_ev_func(Hd)[0]

  split_1, split_2 = eigh.split_spectrum(
    Hd, N, sigma, prior_isometry=None, p_sz=p_sz, precision=precision,
    canonical=canonical)
  H1d, V1d, k1, _ = split_1
  H2d, V2d, k2, _ = split_2

  H1 = pops.undistribute(H1d)
  V1 = pops.undistribute(V1d)
  H2 = pops.undistribute(H2d)
  V2 = pops.undistribute(V2d)
  H1 = H1[:k1, :k1]
  H2 = H2[:k2, :k2]
  V1 = V1[:, :k1]
  V2 = V2[:, :k2]

  ev_Hm, _ = np.linalg.eigh(H1)
  ev_Hp, _ = np.linalg.eigh(H2)
  proj_1 = np.dot(np.dot(V1.conj().T, H), V1)
  ev_p1, _ = np.linalg.eigh(proj_1)
  proj_2 = np.dot(np.dot(V2.conj().T, H), V2)
  ev_p2, _ = np.linalg.eigh(proj_2)
  eps = testutils.eps(precision)
  np.testing.assert_allclose(jnp.sort(ev_p1),
                             jnp.sort(ev_Hm),
                             atol=10 * eps * jnp.linalg.norm(ev_Hm))
  np.testing.assert_allclose(jnp.sort(ev_p2),
                             jnp.sort(ev_Hp),
                             atol=10 * eps * jnp.linalg.norm(ev_Hp))
  np.testing.assert_allclose(ev_exp[:k1], ev_Hm,
                             atol=10 * eps * jnp.linalg.norm(H1))
  np.testing.assert_allclose(ev_exp[k1:], ev_Hp,
                             atol=10 * eps * jnp.linalg.norm(H2))


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
def test_project_H(N, p_sz, precision, seed):
  np.random.seed(seed)
  H = np.random.randn(N, N).astype(np.float32)
  H = 0.5 * (H + H.conj().T)
  ev, eV = np.linalg.eigh(H)
  k_target = N // 2
  P = np.dot(eV[:, :k_target], eV[:, :k_target].conj().T)
  P_bar = np.dot(eV[:, k_target:], eV[:, k_target:].conj().T)
  rank_1, dim_1, rank_2, dim_2 = eigh._padded_ranks(N, k_target)
  out = eigh._project_H(
    pops.distribute(P), pops.distribute(H), rank_1, dim_1, rank_2, dim_2, None,
    p_sz, precision)
  H1, Vk1, info1, H2, Vk2, info2 = out
  Vk1 = pops.undistribute(Vk1)
  Vk2 = pops.undistribute(Vk2)
  ev1 = np.linalg.eigvalsh(pops.undistribute(H1))
  ev2 = np.linalg.eigvalsh(pops.undistribute(H2))
  eps = testutils.eps(precision)

  P_recon = np.dot(Vk1, Vk1.conj().T)
  Pbar_recon = np.dot(Vk2, Vk2.conj().T)
  Pscale = 10 * eps * jnp.linalg.norm(P)
  Hscale = 10 * eps * jnp.linalg.norm(H)

  np.testing.assert_allclose(P_recon, P, atol=Pscale)
  np.testing.assert_allclose(Pbar_recon, P_bar, atol=Pscale)

  angle_1 = _subspace_angle(eV[:, :k_target], Vk1)
  angle_2 = _subspace_angle(eV[:, k_target:], Vk2)
  assert angle_1 < 1E-3
  assert angle_2 < 1E-3

  np.testing.assert_allclose(ev1, ev[:k_target], atol=Hscale)
  np.testing.assert_allclose(ev2, ev[k_target:], atol=Hscale)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("canonical", bools)
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("minimum_rank", minimum_ranks)
@pytest.mark.parametrize("dtype", [np.float32, ])
def test_eigh_full(
    N, p_sz, precision, seed, canonical, padding, minimum_rank, dtype):
  """ Tests that the results of eigh satisfy the eigenvalue equation.
  """
  if padding is not None:
    unpadded_dim = N - padding
  else:
    unpadded_dim = N

  np.random.seed(seed)
  H = np.random.randn(N, N).astype(dtype)
  H[unpadded_dim:, :] = 0.
  H[:, unpadded_dim:] = 0.
  H = jnp.array(0.5 * (H + H.conj().T))
  ev_exp, eV_exp = jnp.linalg.eigh(H)

  Hp = pops.distribute(H)
  evs, V = eigh.eigh(
    Hp, p_sz=p_sz, precision=precision, canonical=canonical,
    unpadded_dim=unpadded_dim, minimum_rank=minimum_rank)
  V = pops.undistribute(V)

  testutils.test_unitarity(
    V[:unpadded_dim, :unpadded_dim], eps_coef=jnp.linalg.norm(H) * 10)

  HV = jnp.dot(H, V, precision=lax.Precision.HIGHEST)
  vV = evs * V

  angle = _subspace_angle(eV_exp, V)
  assert angle < 1E-3
  eps = testutils.eps(precision)
  atol = jnp.linalg.norm(H) * 10 * eps
  np.testing.assert_allclose(ev_exp, jnp.sort(evs), atol=10 * atol)
  np.testing.assert_allclose(HV, vV, atol=10 * atol)


# TODO: Support padding in SVD.
@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("m_factor", m_factors)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("canonical", bools)
@pytest.mark.parametrize("minimum_rank", minimum_ranks)
def test_svd(N, m_factor, p_sz, precision, seed, canonical,
             minimum_rank):
  """
  Tests that svd produces a valid SVD.
  """
  np.random.seed(seed)
  matrix = np.random.randn(N * m_factor, N).astype(np.float32)

  Unp, S_svd, Vnp = np.linalg.svd(matrix)
  Vnp = Vnp.conj().T

  eps = 100 * testutils.eps(precision, dtype=matrix.dtype)

  U, S, V = eigh.svd(
    pops.distribute(matrix), p_sz=p_sz, precision=precision,
    canonical=canonical, minimum_rank=minimum_rank)

  U = pops.undistribute(U)
  V = pops.undistribute(V)

  angle_U = _subspace_angle(U, Unp)
  angle_V = _subspace_angle(V, Vnp)
  assert angle_U < 1E-3  # Note: I pulled this number out of a hat.
  assert angle_V < 1E-3

  # The singular values agree with the numpy results.
  S_sorted = np.sort(S)[::-1]
  np.testing.assert_allclose(
    S_svd, S_sorted, atol=eps * np.sum(np.abs(S_sorted)))

  # Vectors are unitary.
  testutils.test_unitarity(U, eps_coef=10 * jnp.linalg.norm(U)**2)
  testutils.test_unitarity(V, eps_coef=10 * jnp.linalg.norm(V)**2)

  # U @ S @ V^H recovers the result.
  recon = np.dot((U * S), V.conj().T)
  np.testing.assert_allclose(
    matrix, recon, atol=2 * eps * jnp.linalg.norm(matrix))


def _initialize_finalize(N, p_sz):
  """
  Initializes dummy results from eigh._eigh_list for eigh.finalize to act upon.
  """
  k_counter = 0
  klist = []
  while k_counter < (N - p_sz):
    k = random.randrange(p_sz) + 1
    k_counter += k
    klist.append(k)
  klist.append(N - k_counter)

  Hlist = []
  Vlist = []
  evs = []
  eVs = np.zeros((N, N), dtype=np.float32)
  k_counter = 0
  for i, k in enumerate(klist):
    H = np.random.randn(k, k).astype(np.float32)
    H = 0.5 * (H + H.conj().T)
    evs_i, eVs_i = np.linalg.eigh(H)
    evs.extend(evs_i)

    Hbig = np.zeros((p_sz, p_sz), dtype=np.float32)
    Hbig[:k, :k] = H
    Hbig = pops.distribute(Hbig)

    V = np.random.randn(N, k).astype(np.float32)
    V, _ = np.linalg.qr(V, mode="reduced")
    eVs_i = np.dot(V, eVs_i)
    eVs[:, k_counter:k_counter + k] = eVs_i
    k_counter += k

    Vbig = np.zeros((N, p_sz), dtype=np.float32)
    Vbig[:N, :k] = V
    Vbig = pops.distribute(Vbig)
    Hlist.append(Hbig)
    Vlist.append(Vbig)

  return Hlist, Vlist, klist, evs, eVs


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
def test_finalize(N, p_sz, precision):
  """
  Tests eigh.finalize (which converts the output from eigh._eigh_list into the
  final eigh results).
  """
  largest_dimension = max(pops.NROWS, pops.NCOLS)
  if p_sz % largest_dimension != 0:
    p_sz += misc.distance_to_next_divisor(p_sz, largest_dimension)
  Hlist, Vlist, klist, evs, Vs = _initialize_finalize(N, p_sz)
  evs_out, Vs_out = eigh._finalize(Hlist, Vlist, klist, N, precision)
  evs_out = np.array(evs_out)
  Vs_out = pops.undistribute(Vs_out)
  eps = testutils.eps(precision)
  atol = 10 * N * eps
  np.testing.assert_allclose(evs, evs_out, atol=atol)
  np.testing.assert_allclose(np.abs(Vs), np.abs(Vs_out), atol=atol)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("canonical", bools)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("minimum_rank", minimum_ranks)
def test_matrix_function(
  N, p_sz, precision, seed, canonical, dtype, padding, minimum_rank
):
  """ Tests that matrix_function properly compules a * A ** 2.
  """
  a = 3.0
  np.random.seed(seed)
  if padding is None:
    unpadded_dim = N
  else:
    unpadded_dim = N - padding
  H = np.random.randn(N, N).astype(np.float32)
  H[unpadded_dim:, :] = 0.
  H[:, unpadded_dim:] = 0.
  H = 0.5 * (H + H.conj().T)
  expected = a * jnp.dot(H, H, precision=precision)

  H_d = pops.distribute(H)

  def function(x, a, exponent=3):
    return a * x ** exponent

  result, _, _ = eigh.matrix_function(
    function, H_d, a, p_sz=p_sz, precision=precision, canonical=canonical,
    exponent=2, unpadded_dim=unpadded_dim,
    minimum_rank=minimum_rank)
  result = pops.undistribute(result)

  atol = 10 * testutils.eps(precision, dtype=dtype)
  testutils.assert_allclose(
    expected, result, atol=atol * jnp.linalg.norm(expected) ** 2)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("p_sz", p_szs)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("seed", seeds)
@pytest.mark.parametrize("canonical", bools)
@pytest.mark.parametrize("dtype", [np.float32, ])
@pytest.mark.parametrize("padding", padding)
@pytest.mark.parametrize("minimum_rank", minimum_ranks)
@pytest.mark.parametrize("n_occupied", [4, 12])
def test_fermi_broadened_density(
  N, p_sz, precision, seed, canonical, dtype, padding, minimum_rank,
  n_occupied
):
  """ Tests that fermi_broadened_density produces a density matrix with the
  correct spectrum.
  """
  width = 0.003
  np.random.seed(seed)
  if padding is None:
    unpadded_dim = N
  else:
    unpadded_dim = N - padding
  H_orig = np.random.randn(N, N).astype(dtype)
  H_orig = 0.5 * (H_orig + H_orig.conj().T)
  H = np.copy(H_orig)
  H[unpadded_dim:, :] = 0.
  H[:, unpadded_dim:] = 0.

  H_d = pops.distribute(H)
  result, mapped, fermi_level = eigh.fermi_broadened_density(
    H_d, n_occupied, width, p_sz=p_sz, precision=precision,
    canonical=canonical, unpadded_dim=unpadded_dim,
    minimum_rank=minimum_rank)

  mapped = np.sort(mapped)
  result = pops.undistribute(result)
  evals = np.linalg.eigvalsh(result[:unpadded_dim, :unpadded_dim])
  eps = 10 * testutils.eps(precision, dtype=dtype) * np.linalg.norm(evals) ** 2
  testutils.assert_allclose(mapped, evals, atol=eps)

  assert int(round(np.sum(evals))) == n_occupied

  evals, evecs = np.linalg.eigh(H[:unpadded_dim, :unpadded_dim])
  broadened = eigh._fermi_broadening(evals, fermi_level, width)
  H_broadened = np.zeros_like(H)
  H_broadened[:unpadded_dim, :unpadded_dim] = np.dot(
    evecs * broadened, evecs.conj().T)
  testutils.assert_allclose(H_broadened, result, atol=eps)
