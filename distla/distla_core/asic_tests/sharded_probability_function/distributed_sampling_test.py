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
import numpy as np
import jax
from jax import pmap
import jax.lax as lax
import jax.numpy as jnp
from asic_la.sharded_probability_function.distributed_sampling import sample
import pytest
import math

jax.config.update("jax_enable_x64", True)


def gaussian(x, mu=0, sigma=1):
    return (
        1
        / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    )


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("mu", [-0.5, 0.0, 0.5])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sample(sigma, mu, dtype):
    ndev = jax.device_count()
    N = 14
    L = 2 ** N
    x = np.linspace(-4 * sigma, 4 * sigma, L)
    dx = x[1] - x[0]
    gauss_prob = (gaussian(x, mu, sigma) * dx).astype(dtype)
    key = jax.random.PRNGKey(10)
    prob = gauss_prob.reshape(ndev, 2 ** (N - int(math.log2(ndev))))
    Nsamp = 10000
    global_samples, local_samples = jax.pmap(
        sample,
        in_axes=(None, 0, None),
        axis_name="i",
        static_broadcasted_argnums=(0, 2),
    )(key, prob, Nsamp)
    num_local_discretes = N - int(np.log2(jax.device_count()))
    actual_samples = (
        np.asarray(global_samples, np.int64) << num_local_discretes
    ) + np.asarray(local_samples, np.int64)

    actual_samples = (actual_samples - L / 2) * dx
    # the variance of the mean is sigma/np.sqrt(N)
    assert np.abs(np.mean(actual_samples) - mu) < 2.0 * sigma / np.sqrt(Nsamp)
    # the distribution of the unbiased sample variance S^2 is roughly chi-squared
    # i.e. z = (n-1) S^2/sigma^2 ~ chi-squared_(n-1). The variance of the
    # z is 2(n-1) -> var(z) = (n-1)**2/simg**4 var(S^2) = (n-1)**2/sigma**2 * 2 * (n-1)
    # i.e. var(S^2) = 2 sigma**4/(n-1)
    varS2 = 2 * sigma ** 4 / (Nsamp - 1)
    assert np.abs(np.var(actual_samples, ddof=1) - sigma ** 2) < 2 * np.sqrt(
        varS2
    )
