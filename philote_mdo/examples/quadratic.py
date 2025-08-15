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


class QuadradicImplicit(pmdo.ImplicitDiscipline):
    def setup(self):
        self.add_input("a", shape=(1,))
        self.add_input("b", shape=(1,))
        self.add_input("c", shape=(1,))

        self.add_output("x", shape=(1,))

    def setup_partials(self):
        self.declare_partials("x", "a")
        self.declare_partials("x", "b")
        self.declare_partials("x", "c")
        self.declare_partials("x", "x")

    def compute_residuals(self, inputs, outputs, residuals):
        a = inputs["a"]
        b = inputs["b"]
        c = inputs["c"]
        x = outputs["x"]

        residuals["x"] = a * x**2.0 + b * x + c

    def solve_residuals(self, inputs, outputs):
        a = inputs["a"]
        b = inputs["b"]
        c = inputs["c"]
        outputs["x"] = np.array((-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a))

    def residual_partials(self, inputs, outputs, partials):
        a = inputs["a"]
        b = inputs["b"]
        c = inputs["c"]
        x = outputs["x"]

        partials["x", "a"] = np.array([x**2])
        partials["x", "b"] = np.array([x])
        partials["x", "c"] = np.array([1.0])
        partials["x", "x"] = np.array([2 * a * x + b])

        self.inv_jac = 1.0 / (2 * a * x + b)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        a = inputs["a"]
        b = inputs["b"]
        c = inputs["c"]
        x = outputs["x"]
        if mode == "fwd":
            if "x" in d_residuals:
                if "x" in d_outputs:
                    d_residuals["x"] += (2 * a * x + b) * d_outputs["x"]
                if "a" in d_inputs:
                    d_residuals["x"] += x**2 * d_inputs["a"]
                if "b" in d_inputs:
                    d_residuals["x"] += x * d_inputs["b"]
                if "c" in d_inputs:
                    d_residuals["x"] += d_inputs["c"]
        elif mode == "rev":
            if "x" in d_residuals:
                if "x" in d_outputs:
                    d_outputs["x"] += (2 * a * x + b) * d_residuals["x"]
                if "a" in d_inputs:
                    d_inputs["a"] += x**2 * d_residuals["x"]
                if "b" in d_inputs:
                    d_inputs["b"] += x * d_residuals["x"]
                if "c" in d_inputs:
                    d_inputs["c"] += d_residuals["x"]
