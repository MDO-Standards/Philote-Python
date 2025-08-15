# Philote-Python
#
# Copyright 2022-2025 Christopher A. Lupp
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
#
#
# This work has been cleared for public release, distribution unlimited, case
# number: AFRL-2023-5713.
#
# The views expressed are those of the authors and do not reflect the
# official guidance or position of the United States Government, the
# Department of Defense or of the United States Air Force.
#
# Statement from DoD: The Appearance of external hyperlinks does not
# constitute endorsement by the United States Department of Defense (DoD) of
# the linked websites, of the information, products, or services contained
# therein. The DoD does not exercise any editorial, security, or other
# control over the information you may find at these locations.
import grpc
import numpy as np
import philote_mdo.general as pmdo


client = pmdo.ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

# transfer the stream options to the server
client.send_stream_options()

# run setup
client.run_setup()
client.get_variable_definitions()
client.get_partials_definitions()

# define some inputs
inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
outputs = {}

# run a function evaluation
outputs = client.run_compute(inputs)
print(outputs)

# run a gradient evaluation
partials = client.run_compute_partials(inputs)
print(partials)
