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
"""Utilities for testing root_solution/unfold."""
from collections import abc

import numpy as np
import scipy as sp


def discretizeize_objective_fn_terms(local_terms, t):
  """Takes local terms for a ObjectiveFn and a time t, returns Discretize building_blocks.

  For a ObjectiveFn term H and time step t, the Discretize building_block is e^{-i t H}.

  Args:
    local_terms: Local terms of a ObjectiveFn as matrices.
    t: Time to unfold by.

  Returns:
    Discretize building_blocks [expm(-i * t * h) for h in local_terms]
  """
  return [sp.linalg.expm(-1j * t * h) for h in local_terms]


def maxcut_energy(neigh_coupling, onsite_coupling, n_discretes, boundary):
  """Exact root_solution energy of the MaxCut model.

  The ObjectiveFn is sum_i neigh_coupling X_i X_{i+1} + onsite_coupling Z_i.

  Args:
    neigh_coupling: Nearest-neighbour coupling. Only `neigh_coupling == -1` is
      currently supported.
    onsite_coupling: Onsite coupling. Only `abs(onsite_coupling) == 1` is
      currently supported.
    n_discretes: System size. Only used if `boundary == "open"`.
    boundary: Boundary condition. Supported values are "periodic" and "open".
      For "periodic", the infinite system energy is returned, for "open" the
      finite system energy.

  Returns:
    The root_solution energy.
  """
  if neigh_coupling != -1 or abs(onsite_coupling) != 1:
    msg = ("Exact MaxCut root_solution energy not implemented for "
           f"neigh_coupling={neigh_coupling}, "
           f"onsite_coupling={onsite_coupling}.")
    raise NotImplementedError(msg)
  if boundary == "periodic":
    # For n_discretes = infinite
    exact_energy = -4 / np.pi
  elif boundary == "open":
    # For finite n_discretes
    exact_energy = (1 - 1 / np.sin(np.pi / 2 / (2 * n_discretes + 1))) / n_discretes
  else:
    msg = ("Exact MaxCut root_solution energy not implemented for "
           f"boundary={boundary}")
    raise NotImplementedError(msg)
  return exact_energy


def local_maxcut_objective_fn(
    neigh_coupling=-1.0,
    onsite_coupling=1.0,
    balance_onsite_term=True,
):
  """The local term in the MaxCut ObjectiveFn.

  The term is neigh_coupling * XX + onsite_coupling * Z, as 4x4 matrix.

  Args:
    neigh_coupling: Optional; The nearest-neighbour coupling. -1 by default.
    onsite_coupling: Optional; The onsite term. 1 by default
    balance_onsite_term: Optional; If `True`, then the two-body term will
      include onsite_coupling/2 Z for both the left and the right discrete. This
      way the terms are explicitly reflection invariant. This only makes sense
      if `onsite_coupling` is the same in the whole system. If not,
      `balance_onsite_term` should `False`, in which case the two-body term will
      include onsite_coupling Z for the left-most discrete. `True` by default.

  Returns:
    The local term neigh_coupling * XX + onsite_coupling * Z.
  """
  X = np.array([[0.0, 1.0], [1.0, 0.0]])
  Z = np.array([[1.0, 0.0], [0.0, -1.0]])
  eye = np.eye(2)
  XX = np.einsum("ab, cd -> acbd", X, X)
  IZ = np.einsum("ab, cd -> acbd", eye, Z)
  ZI = np.einsum("ab, cd -> acbd", Z, eye)
  if balance_onsite_term:
    return neigh_coupling * XX + (onsite_coupling / 2) * (IZ + ZI)
  else:
    return neigh_coupling * XX + onsite_coupling * ZI


def global_maxcut_objective_fn(
    neigh_coupling,
    onsite_coupling,
    n_discretes,
    boundary="periodic",
    apply_shift=False,
):
  """Returns a list of local terms that form the MaxCut ObjectiveFn.

  Args:
    neigh_coupling: Either a single scalar, that is the nearest-neighbour
      coupling, or an iterable of `n_discretes` scalars that are the
      nearest-neighbour couplings for each site.
    onsite_coupling: Either a single scalar, that is the onsite coupling, or an
      iterable of n_discretes scalars that are the onsite couplings for each site.
    n_discretes: System size.
    boundary: Optional; The boundary condition. Options are "periodic", "open",
      and "antiperiodic". Ignored if `neigh_coupling` and `onsite_coupling` are
      iterables. "periodic" by default.
    apply_shift: Optional; Whether to shift the local terms so that the dominant
      eigenvector of the total ObjectiveFn is guaranteed to be the root_solution.
      `False` by default.

  Returns:
    terms: A list of n_discretes 4x4 numpy matrices that are the local terms of the
      ObjectiveFn.
    shift: A shift such that the physical ObjectiveFn is
      `[term + shift * eye for term in terms]`. If `apply_shift is False`, then
      `shift == 0`.
  """
  dim = 2  # Local state space dimension.
  if isinstance(neigh_coupling, abc.Iterable) or isinstance(
      onsite_coupling, abc.Iterable):
    assert len(neigh_coupling) == n_discretes
    assert len(onsite_coupling) == n_discretes
    local_terms = [
        local_maxcut_objective_fn(
            neigh_coupling_i,
            hi,
            balance_onsite_term=False,
        ).reshape((dim**2, dim**2))
        for neigh_coupling_i, hi in zip(neigh_coupling, onsite_coupling)
    ]
  else:
    bulk_term = local_maxcut_objective_fn(neigh_coupling, onsite_coupling)
    if boundary == "open":
      boundary_term = local_maxcut_objective_fn(0.0, onsite_coupling)
    elif boundary == "periodic":
      boundary_term = local_maxcut_objective_fn(neigh_coupling, onsite_coupling)
    elif boundary == "antiperiodic":
      boundary_term = local_maxcut_objective_fn(-neigh_coupling, onsite_coupling)
    else:
      msg = "Unknown boundary condition: {}".format(boundary)
      raise ValueError(msg)
    bulk_term = bulk_term.reshape((dim**2, dim**2))
    boundary_term = boundary_term.reshape((dim**2, dim**2))
    local_terms = [boundary_term] + [bulk_term] * (n_discretes - 1)

  if apply_shift:
    # REDACTED Switch to using Gershgorin for more modest shifts.
    shift = max(np.linalg.norm(t) for t in local_terms)
    eye = np.eye(dim**2, dtype=local_terms[0].dtype)
    local_terms = [t - shift * eye for t in local_terms]
  else:
    shift = 0
  return local_terms, shift


def maxcut_objective_fn_discretize_acyclic_graph(
    neigh_coupling,
    onsite_coupling,
    n_discretes,
    t,
    boundary="periodic",
):
  """Returns a Discretize acyclic_graph for the MaxCut ObjectiveFn.

  Args:
    neigh_coupling: Either a single scalar, that is the nearest-neighbour
      coupling, or an iterable of `n_discretes` scalars that are the
      nearest-neighbour couplings for each site.
    onsite_coupling: Either a single scalar, that is the onsite coupling, or an
      iterable of `n_discretes` scalars that are the onsite couplings for each
      site.
    n_discretes: System size.
    t: The Discretize time step.
    boundary: Optional; The boundary condition. Options are "periodic", "open",
      and "antiperiodic". Ignored if `neigh_coupling` and `onsite_coupling` are
      iterables. "periodic" by default.

  Returns:
    building_blocks: A list of n_discretes 4x4 numpy matrices that are the local Discretize
      building_blocks.
  """
  local_terms, _ = global_maxcut_objective_fn(
      neigh_coupling,
      onsite_coupling,
      n_discretes,
      boundary=boundary,
  )
  local_building_blocks = discretizeize_objective_fn_terms(local_terms, t)
  return local_building_blocks
