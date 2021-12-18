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
"""Helper functions for distributed sampling."""
import math
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

AXIS_NAME = "i"


@partial(jax.vmap)
def vchoice(key, probs):
    """A vectorized version of jax.random.choice.
    Useful for conditional sampling.

    Args:
      key: A vectorized set of PRNGKeys
      progs: A vectorized set of probability vectors.

    """
    return jax.random.choice(key, probs.shape[0], shape=(), p=probs)


def sample(prng_key, sharded_probs, repetitions):
    """Distributed bit-string sampling.

    Performs a observation across all discretes in the acyclic_graph using all
    accelerator cores in parallel.

    Args:
      prng_key: `jax.random` key used for sampling.
      sharded_probs: A single shard of the probability distribution.
      repetitions: Number of samples

    Returns:
      jax.DeviceArray[int], jax.DeviceArray[int]: The global and the local
        bistring samples, encoded into 32 bit integers.
    """
    # Due to a bug in jax (https://github.com/google/jax/issues/6935)
    # we can't have a repetitions being a prime number.
    reps = repetitions + repetitions % 2

    # Create all of the keys needed for the different samplings
    # First two are for the local discretes, last is for the global.
    key1, key2, key3 = jax.random.split(prng_key, 3)

    # Reshaped the given probs into a matrix. Each axes will be sampled
    # separately. This is needed because a float32 value only has 23 bits in
    # its mantissa, and cannot differentiate more than 2**23 inidividual states.
    # Our simulator is supposed to supprt >30 discretes, and this 2**30 possible states.
    # Sampling twice fixes this.
    size = np.prod(sharded_probs.shape)
    sharded_probs = sharded_probs.reshape((size // 1024, 1024))

    # Get the marginal probability of just the local discretes.
    p_local = jax.lax.psum(sharded_probs, AXIS_NAME)

    # Sample the first chunk.
    sample_locals_chunk1 = jax.random.choice(
        key1, p_local.shape[0], shape=(reps,), p=p_local.sum(1)
    )

    # Sample the second chunk conditional on the first chunk.
    # We don't have to normalize here because jax.random.choice
    # uses inverse sampling which does not care about normalization of
    # the probability
    sample_locals_chunk2 = vchoice(
        jax.random.split(key2, reps), p_local[sample_locals_chunk1]
    ).reshape((reps,))

    # Combine the samples.
    # 10 is from the arbitrary 1024 above (2**10 == 1024).
    sample_locals = (sample_locals_chunk1 << 10) + sample_locals_chunk2

    # Now to sample the global discretes.
    # First, grab the conditional probability for the given core.
    p_globals_given_locals = sharded_probs.ravel()[sample_locals]
    # we need to reshape into an array with 3 dimensions because
    # of a bug in jax which causes pswapaxes to hang otherwise
    # see https://github.com/google/jax/issues/6935
    p_globals_given_locals = jnp.reshape(p_globals_given_locals, (2, reps // 2))

    # Next, we replicate the value on each core to every other core.
    count_global = int(math.log2(jax.device_count()))
    p_globals = jnp.array([p_globals_given_locals] * 2 ** count_global)
    p_globals = jax.lax.pswapaxes(p_globals, AXIS_NAME, 0)

    # reshape back into a matrix (see comment above)
    p_globals = p_globals.reshape((jax.device_count(), reps))

    p_globals = p_globals.transpose((1, 0))

    # Finally, we sample the conditional probability.
    sample_globals = vchoice(jax.random.split(key3, reps), p_globals)

    # Combine and sort the samples.
    all_global_samples = jnp.asarray(sample_globals[:repetitions], jnp.uint32)
    all_local_samples = jnp.asarray(sample_locals[:repetitions], jnp.uint32)
    return all_global_samples, all_local_samples


def single_core_sample(prng_key, probs, repetitions):
    """Single core bit-string sampling.

    Performs a observation across all discretes in the acyclic_graph

    Args:
      prng_key: `jax.random` key used for sampling.
      probs: The probability distribution.
      repetitions: Number of samples

    Returns:
      jax.DeviceArray[int], jax.DeviceArray[int]: The global and the local
        bistring samples, encoded into 32 bit integers. Note that since
        `probs` has no global discretes, the global samples is an array of
        0.
    """
    # Create all of the keys needed for the different samplings
    key1, key2 = jax.random.split(prng_key, 2)

    # Reshaped the given probs into a matrix to avoid padding
    size = np.prod(probs.shape)
    probs = probs.reshape((size // 1024, 1024))

    # Sample the first chunk.
    sample_locals_chunk1 = jax.random.choice(
        key1, probs.shape[0], shape=(repetitions,), p=probs.sum(1)
    )

    # Sample the second chunk conditional on the first chunk.
    # We don't have to normalize here because jax.random.choice
    # uses inverse sampling which does not care about normalization of
    # the probability
    sample_locals_chunk2 = vchoice(
        jax.random.split(key2, repetitions), probs[sample_locals_chunk1]
    ).reshape((repetitions,))

    # Combine the samples.
    # 10 is from the arbitrary 1024 above (2**10 == 1024).
    all_samples = (sample_locals_chunk1 << 10) + sample_locals_chunk2
    all_samples = jnp.asarray(all_samples, jnp.uint32)
    return jnp.zeros_like(all_samples), all_samples
