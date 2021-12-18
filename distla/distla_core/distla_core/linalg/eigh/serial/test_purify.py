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
"""Test for purify.py."""
import itertools
from jax import lax
import jax.numpy as jnp
import numpy as np
import pytest

from distla_core.linalg.eigh.serial import purify
from distla_core.linalg.utils import testutils
from distla_core.utils import misc
from distla_core.utils import pops


Ns = [8, 16]
k_fractions = [4, ]
precisions = [lax.Precision.HIGHEST]
dtypes = [jnp.bfloat16, jnp.float32]
seeds = [1, 2]

purify_test_argstring = "N,k_fraction,precision,dtype,seed"
purify_test_args = list(
  itertools.product(Ns, k_fractions, precisions, dtypes, seeds))


def _generate_overlaps(N, dtype, pad_size=0):
  S = np.random.randn(N, N)
  S /= np.linalg.norm(S)
  evS, eVS = np.linalg.eigh(S)
  evS = np.abs(evS) + 0.2
  S_inv_sqrt = (eVS / np.sqrt(evS)) @ eVS.conj().T
  S_sqrt = (eVS * np.sqrt(evS)) @ eVS.conj().T
  S_inv_sqrt_padded = np.zeros((N + pad_size, N + pad_size))
  S_inv_sqrt_padded[:N, :N] = S_inv_sqrt
  S_inv_sqrt_padded = jnp.array(S_inv_sqrt_padded).astype(dtype)
  return S_inv_sqrt_padded, S_sqrt


def _correct_projector(H, k_target):
  """
  Returns the projector into the rank-`k_target` negative subspace of `H`
  computed using eigh. Also computes relevant chemical potential.
  """
  ev, eV = np.linalg.eigh(H)
  mu = np.mean([ev[k_target - 1], ev[k_target]])
  return np.dot(eV[:, :k_target], eV[:, :k_target].T.conj()), mu


def _random_self_adjoint(N, dtype, pad_size=0):
  np.random.seed(1)
  V = np.random.randn(N, N).astype(dtype)
  V, _ = np.linalg.qr(V)
  S = np.linspace(-10 * N, 10 * N, num=N)
  H = (V * S) @ V.conj().T
  H_padded = np.zeros((N + pad_size, N + pad_size), dtype=dtype)
  H_padded[:N, :N] = H
  return H_padded


def purify_init(rows, k_fraction, precision, dtype, pad_size, seed, overlaps,
                distribute=False):
  """
  Generates a Hermitian `rows x rows` matrix. Constructs a projector into the
  `rows // k_fraction`'th negative subspace. If overlaps is True, also makes
  dummy S^-1/2 and S^1/2 matrices. The returned Hermitian matrix is then
  H~ = S^-1/2 @ (original Hermitian matrix) @ S^-1/2. The returned projector
  is S^-1/2 @ P~ @ S^-1/2, where P~ is into the negative subspace of H~.

  If overlaps is False the S^-1/2 and S^1/2 matrices are both None.

  If distribute is True, both S^-1/2 and self_adjoint will be distributed.

  Returns:
    self_adjoint, k_target, expected, eps, overlap_inv_sqrt, overlap_sqrt
  """
  np.random.seed(seed)
  self_adjoint = _random_self_adjoint(rows, dtype, pad_size)
  k_target = rows // k_fraction
  if overlaps:
    overlap_inv_sqrt, overlap_sqrt = _generate_overlaps(rows, dtype, pad_size)
    norm_inv_sqrt = jnp.linalg.norm(overlap_inv_sqrt)
    eps = testutils.eps(precision, dtype=dtype) * (norm_inv_sqrt)**2
    self_adjoint_mapped = misc.similarity_transform(
      self_adjoint, overlap_inv_sqrt, precision=precision)
    expected, _ = _correct_projector(
      self_adjoint_mapped[:rows, :rows], k_target)
    expected = misc.similarity_transform(
      expected, overlap_inv_sqrt[:rows, :rows], precision=precision)
    if distribute:
      overlap_inv_sqrt = pops.distribute(overlap_inv_sqrt)
  else:
    overlap_inv_sqrt = None
    overlap_sqrt = None
    expected, _ = _correct_projector(self_adjoint[:rows, :rows], k_target)
    eps = testutils.eps(precision, dtype=dtype) * rows
  if distribute:
    self_adjoint = pops.distribute(self_adjoint)
  return self_adjoint, k_target, expected, eps, overlap_inv_sqrt, overlap_sqrt


def purify_asserts(result, expected, eps, overlap_sqrt=None):
  """ Checks that result ~= expected and result^2 ~= result, in the latter
  case after a similarity by overlap_sqrt when overlap_sqrt is not None.
  """
  if overlap_sqrt is not None:
    idempotent = misc.similarity_transform(
      result, overlap_sqrt, precision=lax.Precision.HIGHEST)
  else:
    idempotent = result
  idempotent_2 = jnp.dot(
    idempotent, idempotent, precision=lax.Precision.HIGHEST)
  testutils.assert_allclose(result, expected, atol=eps)
  testutils.assert_allclose(idempotent, idempotent_2, atol=10 * eps)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
def test_subspaces(N, k_fraction, precision, dtype, seed):
  init = purify_init(N, k_fraction, precision, dtype, 0, seed, False)
  self_adjoint, k_target, projector, eps, _, _ = init
  isometry_1, isometry_2 = purify.subspace(
    projector, k_target, precision, "complete")
  proj_v1 = jnp.dot(projector, isometry_1, precision=precision)
  proj_v2 = jnp.dot(projector, isometry_2, precision=precision)

  eps = testutils.eps(precision)
  atol = 10 * N * eps
  testutils.assert_allclose(proj_v1, isometry_1, atol=atol)
  testutils.assert_allclose(proj_v2, np.zeros_like(isometry_2), atol=atol)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
@pytest.mark.parametrize("overlaps", [True, False])
@pytest.mark.parametrize("method", ["hole-particle", ])
def test_canonically_purify(N, k_fraction, precision, dtype, method, seed,
                            overlaps):
  """
  Tests the canonical purification methods.
  """
  init = purify_init(N, k_fraction, precision, dtype, 0, seed, overlaps)
  self_adjoint, k_target, expected, eps, overlap_invsqrt, overlap_sqrt = init
  result, j, errs = purify.canonically_purify(
    self_adjoint, k_target, method=method, overlap_invsqrt=overlap_invsqrt)
  purify_asserts(result, expected, eps, overlap_sqrt=overlap_sqrt)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
def test_newton_schulz_purify(N, k_fraction, precision, dtype, seed):
  """
  Tests the Newton-Schulz purification method.
  """
  init = purify_init(N, k_fraction, precision, dtype, 0, seed, False)
  self_adjoint, k_target, expected, eps, _, _ = init
  result, _ = purify.newton_schulz_purify(
    self_adjoint, k_target, precision=precision)
  purify_asserts(result, expected, eps)


@pytest.mark.parametrize(purify_test_argstring, purify_test_args)
def test_grand_canonically_purify(N, precision, dtype, k_fraction, seed):
  init = purify_init(N, k_fraction, precision, dtype, 0, seed, False)
  self_adjoint, k_target, expected, eps, _, _ = init
  ev = np.linalg.eigvalsh(self_adjoint)
  mu = np.mean([ev[k_target - 1], ev[k_target]])
  result, _, _, _ = purify.grand_canonically_purify(self_adjoint, mu, precision)
  purify_asserts(result, expected, eps)
