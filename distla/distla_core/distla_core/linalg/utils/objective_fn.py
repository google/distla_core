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
import jax.numpy as jnp
import numpy as np


def block_local_objective_fns(twosite_objective_fns):
  """
  Block a list of two-side objective_fn terms
  `twosite_objective_fns` into a single matrix.

  Args:
    twosite_objective_fns: A list of two-site
      objective_fn terms.
  Returns:
    ShapedArray: The blocked terms.
  """
  shapes = [t.shape for t in twosite_objective_fns]
  if not np.all([s == (4, 4) for s in shapes]):
    raise ValueError(f"all shapes of `twosite_objective_fns` have "
                     f"to be (4, 4), found {shapes}")
  dtype = twosite_objective_fns[0].dtype
  nqbits = len(twosite_objective_fns) + 1
  blocked = jnp.zeros((2**nqbits, 2**nqbits), dtype=dtype)
  for i, h in enumerate(twosite_objective_fns):
    eye1 = jnp.eye(2**i, dtype=dtype)
    eye2 = jnp.eye(2**(nqbits - 2 - i), dtype=dtype)
    blocked += jnp.kron(jnp.kron(eye1, h), eye2)
  return blocked
