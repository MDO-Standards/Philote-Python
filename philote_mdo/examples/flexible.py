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
import numpy as np
import philote_mdo.general as pmdo


class FlexibleDiscipline(pmdo.ExplicitDiscipline):
    """
    Example explicit discipline with dynamic shapes.

    This discipline doubles every element of the input vector.  The input
    and output shapes are not fixed by the server — the client is
    expected to set them via ``SetVariableShapes`` before computation.
    """

    def setup(self):
        self.add_input("x", dynamic_shape=True, units="m")
        self.add_output("y", dynamic_shape=True, units="m")

    def setup_partials(self):
        self.declare_partials("y", "x")

    def compute(self, inputs, outputs):
        outputs["y"] = 2.0 * inputs["x"]

    def compute_partials(self, inputs, partials):
        n = inputs["x"].size
        partials["y", "x"] = 2.0 * np.eye(n)
