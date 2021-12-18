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
This module collects pytest markers which can be used to add more granular
control over how tests are performed.
By decorating a test with a pytest marker one can modify the behaviour
of the test function.
"""
import pytest
import jax

backend = jax.lib.xla_bridge.get_backend()
multi_host_setting = jax.process_count() > 1
#only run the decorated test on gnodes
asic_node_only = pytest.mark.skipif(
    backend.platform != 'asic' or multi_host_setting,
    reason='Only on a node',
)

single_host_only = pytest.mark.skipif(
    multi_host_setting,
    reason='Only on single host',
)

effective_asic_node_only = pytest.mark.skipif(
    multi_host_setting or len(jax.devices()) != 8,
    reason="Only on a spoofed or actual node"
)

#only run the decorated test on ASICs
asic_only = pytest.mark.skipif(backend.platform != 'asic', reason='Only on ASICs')

#only run the decorated test on a ASIC slice with more than one host vm
multihost_only = pytest.mark.skipif(
    backend.platform != 'asic' or not multi_host_setting,
    reason='Only on a ASIC slice')
