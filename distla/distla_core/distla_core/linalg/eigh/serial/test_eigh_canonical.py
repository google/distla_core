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
"""Test for eigh.py."""
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from distla_core.linalg.eigh.serial import eigh_canonical as eigh
from distla_core.linalg.eigh.serial import purify
from distla_core.linalg.utils import testutils
from distla_core.utils import pops

# TODO: add dtype tests

Ns = [4, 8, 16]
precisions = [lax.Precision.HIGHEST]


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("precision", precisions)
def test_split_spectrum(N, precision):
  np.random.seed(15)
  H = np.random.randn(N, N).astype(np.float32)
  H = jnp.array(0.5 * (H + H.conj().T))
  ev_exp, eV_exp = jnp.linalg.eigh(H)

  V = jnp.eye(N, dtype=np.float32)
  k = N // 2
  P, _, _ = purify.canonically_purify(H, k, precision=precision)

  Hm, Vm, Hp, Vp = eigh.split_spectrum(P, H, V, k, precision)
  ev_Hm, _ = jnp.linalg.eigh(Hm)
  ev_Hp, _ = jnp.linalg.eigh(Hp)
  proj_1 = jnp.dot(
      jnp.dot(Vm.conj().T, H, precision=lax.Precision.HIGHEST),
      Vm,
      precision=lax.Precision.HIGHEST)
  ev_p1, _ = jnp.linalg.eigh(proj_1)
  proj_2 = jnp.dot(
      jnp.dot(Vp.conj().T, H, precision=lax.Precision.HIGHEST),
      Vp,
      precision=lax.Precision.HIGHEST)
  ev_p2, _ = jnp.linalg.eigh(proj_2)
  ev_concat = jnp.sort(jnp.hstack([ev_Hm, ev_Hp]))
  eps = testutils.eps(precision)
  np.testing.assert_allclose(
      ev_exp, ev_concat, atol=10 * eps * jnp.linalg.norm(ev_exp))
  np.testing.assert_allclose(
      jnp.sort(ev_p1), jnp.sort(ev_Hm), atol=10 * eps * jnp.linalg.norm(ev_Hm))
  np.testing.assert_allclose(
      jnp.sort(ev_p2), jnp.sort(ev_Hp), atol=10 * eps * jnp.linalg.norm(ev_Hp))


@pytest.mark.parametrize("N", Ns)
def test_combine_eigenblocks(N):
  np.random.seed(10)
  H1 = np.random.randn(N).astype(np.float32)
  H2 = np.random.randn(N // 2).astype(np.float32)
  V1 = np.random.randn(3, N).astype(np.float32)
  V2 = np.random.randn(3, N // 2).astype(np.float32)
  H, V = eigh._combine_eigenblocks((H1, V1), (H2, V2))
  np.testing.assert_allclose(H, jnp.hstack((H1, H2)))
  np.testing.assert_allclose(V, jnp.hstack((V1, V2)))


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("precision", precisions)
def test_serial_eigh(N, precision):
  np.random.seed(10)
  H = np.random.randn(N, N).astype(np.float32)
  H = jnp.array(0.5 * (H + H.conj().T))
  ev_exp, eV_exp = jnp.linalg.eigh(H)
  evs, V = eigh.eigh(H)
  HV = pops.dot(H, V, precision=precision)
  vV = evs * V
  eps = testutils.eps(precision)
  atol = jnp.linalg.norm(H) * eps
  np.testing.assert_allclose(ev_exp, jnp.sort(evs), atol=10 * atol)
  np.testing.assert_allclose(HV, vV, atol=10 * atol)

  # The eigenvalues agree with the numpy results.
  evs_sorted = np.sort(evs)
  np.testing.assert_allclose(evs, evs_sorted, atol=10 * atol)

  # V is unitary.
  v_unitary_delta = np.dot(V.conj().T, V)
  v_eye = np.eye(v_unitary_delta.shape[0])
  np.testing.assert_allclose(v_unitary_delta, v_eye, atol=10 * atol)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("precision", precisions)
def test_svd(N, precision):
  np.random.seed(10)
  H = np.random.randn(N, N).astype(np.float32)
  S_expected = np.linalg.svd(H, compute_uv=False)
  U, S, V = eigh.svd(H, precision=precision)
  recon = pops.dot((U * S), V.conj().T, precision=precision)
  eps = testutils.eps(precision)
  eps = eps * jnp.linalg.norm(H) * 10
  np.testing.assert_allclose(np.sort(S), np.sort(S_expected), atol=eps)
  np.testing.assert_allclose(H, recon, atol=eps)

  # U is unitary.
  u_unitary_delta = jnp.dot(U.conj().T, U, precision=lax.Precision.HIGHEST)
  u_eye = np.eye(u_unitary_delta.shape[0])
  np.testing.assert_allclose(u_unitary_delta, u_eye, atol=eps)

  # V is unitary.
  v_unitary_delta = jnp.dot(V.conj().T, V, precision=lax.Precision.HIGHEST)
  v_eye = np.eye(v_unitary_delta.shape[0])
  np.testing.assert_allclose(v_unitary_delta, v_eye, atol=eps)
