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
Function to compute the square root and its inverse for positive definite
matrix. This file covers the distributed case. This function is the interface,
with the work functions defined in `distla_core.linalg.invsqrt_utils`.
"""
from jax import lax

from distla_core.linalg.backends import distributed_backend
from distla_core.linalg.invsqrt import invsqrt_utils


def invsqrt(A,
            eps=None,
            maxiter=200,
            s_min=None,
            s_thresh=0.1,
            p_sz=128,
            precision=lax.Precision.HIGHEST):
  """
  Computes the square root and inverse square root of the positive definite
  matrix `A`.

  The method is an iterative one. As explained in Higobj_fn's "Stable iterations
  for the matrix square root", 1997, the matrix sign function of the block
  matrix `[[0, A], [I, 0]]` is `[[0, sqrt(A)], [inv(sqrt(A)), 0]]`, and hence
  the same Newton-Schultz iteration that is used for computing the matrix sign
  function (see `polar.py`) can be applied to simultaneously compute `sqrt(A)`,
  `inv(sqrt(A))`.

  The iteration proceeds in two stages. First we repeatedly apply the so called
  "rogue" polynomial
  ```
    Y_{k+1} = a_m * Y_k - 4 * (a_m/3)**3 * Y_k @ Z_k @ Y_k
    Z_{k+1} = a_m * Z_k - 4 * (a_m/3)**3 * Z_k @ Y_k @ Z_k
  ```
  where `a_m = (3 / 2) * sqrt(3) - s_thresh`, and `Y_0 = A` and `Z_0 = I`, to
  bring the eigenvalues of `[[0, Y], [Z, 0]]` to within the range `[s_thresh,
  1]`. Then we switch to the Newton-Schultz iteration
  ```
    Y_{k+1} = (3 / 2) * Y_k - (1 / 2) * Y_k @ Z_k @ Y_k
    Z_{k+1} = (3 / 2) * Z_k - (1 / 2) * Z_k @ Y_k @ Z_k
  ```
  until convergence.

  Args:
    `A`: The input matrix. Assumed to be positive definite.
    `eps`: The final result will satisfy `|I - Y @ Z| <= eps * |I|`, where `Y`
           and `Z` are the returned approximations to `sqrt(A)` and
           `inv(sqrt(A))` respectively. Machine epsilon by default.
    `maxiter`: Iterations will terminate after this many steps even if the
               above is unsatisfied. 200 by default.
    `s_min`: An under estimate of the smallest eigenvalue value of
             `[[0, A], [I, 0]]`. Machine epsilon by default.
    `s_thresh`: The iteration switches from the `rogue` polynomial to the
                Newton-Schultz iterations after `s_min` is estimated to have
                reached this value. 0.1 by default.
    `p_sz`: Panel size for the SUMMA matmuls. 128 by default.
    `precision`: Precision of the matrix multiplications.
  Returns:
    `Y`: approximation to `sqrt(A)`.
    `Z`: approximation to `inv(sqrt(A))`.
    `jr`: The number of 'rogue' iterations.
    `jt`: The total number of iterations.
  """
  # TODO The above description for `s_min` isn't very helpful. How do we
  # understand the connection between eigenvalues of the block matrix, and
  # eigenvalues of A?
  backend = distributed_backend.DistributedBackend(p_sz, precision=precision)
  return invsqrt_utils._invsqrt(A, eps, maxiter, s_min, s_thresh, backend)
