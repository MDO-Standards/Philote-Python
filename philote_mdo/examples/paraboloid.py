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
import philote_mdo.general as pmdo


class Paraboloid(pmdo.ExplicitDiscipline):
    """
    Basic two-dimensional paraboloid example (explicit) discipline.
    """

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")

        self.add_output("f_xy", shape=(1,), units="m**2")

    def setup_partials(self):
        self.declare_partials("f_xy", "x")
        self.declare_partials("f_xy", "y")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        y = inputs["y"]

        outputs["f_xy"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs["x"]
        y = inputs["y"]

        partials["f_xy", "x"] = 2.0 * x - 6.0 + y
        partials["f_xy", "y"] = 2.0 * y + 8.0 + x
