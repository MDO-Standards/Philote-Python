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
import unittest
import numpy as np
import philote_mdo.utils as utils
from philote_mdo.examples import QuadradicImplicit
import philote_mdo.generated.data_pb2 as data


class TestQuadradicImplicit(unittest.TestCase):
    """
    Unit tests for the QuadradicImplicit discipline.
    """

    def test_setup(self):
        """
        Tests the setup function of the QuadradicImplicit server.
        """
        disc = QuadradicImplicit()
        disc.setup()

        self.assertEqual(disc._var_meta[0].name, "a")
        self.assertEqual(disc._var_meta[1].name, "b")
        self.assertEqual(disc._var_meta[2].name, "c")
        self.assertEqual(disc._var_meta[3].name, "x")

        self.assertEqual(disc._var_meta[0].type, data.kInput)
        self.assertEqual(disc._var_meta[1].type, data.kInput)
        self.assertEqual(disc._var_meta[2].type, data.kInput)
        self.assertEqual(disc._var_meta[3].type, data.kOutput)

        self.assertEqual(disc._var_meta[0].shape, [1])
        self.assertEqual(disc._var_meta[1].shape, [1])
        self.assertEqual(disc._var_meta[2].shape, [1])
        self.assertEqual(disc._var_meta[3].shape, [1])

    def test_setup_partials(self):
        """
        Tests the setup function of the QuadradicImplicit server.
        """
        disc = QuadradicImplicit()
        disc.setup_partials()

        self.assertEqual(disc._partials_meta[0].name, "x")
        self.assertEqual(disc._partials_meta[0].subname, "a")

        self.assertEqual(disc._partials_meta[1].name, "x")
        self.assertEqual(disc._partials_meta[1].subname, "b")

        self.assertEqual(disc._partials_meta[2].name, "x")
        self.assertEqual(disc._partials_meta[2].subname, "c")

        self.assertEqual(disc._partials_meta[3].name, "x")
        self.assertEqual(disc._partials_meta[3].subname, "x")

    def test_compute_residuals(self):
        """
        Tests the compute function of the QuadradicImplicit discipline.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        residuals = {"x": np.array([0.0])}
        disc = QuadradicImplicit()
        disc.compute_residuals(inputs, outputs, residuals)

        self.assertEqual(residuals["x"][0], 22.0)

    def test_solve_residuals(self):
        """
        Tests the solve_residuals function of the QuadradicImplicit discipline.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.zeros(1)}
        disc = QuadradicImplicit()
        disc.solve_residuals(inputs, outputs)

        self.assertAlmostEqual(outputs["x"][0], 0.73205081, places=8)

    def test_residual_partials(self):
        """
        Tests the compute function of the QuadradicImplicit discipline.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        jac = utils.PairDict()
        jac["f_xy", "x"] = 0.0
        jac["f_xy", "y"] = 0.0
        disc = QuadradicImplicit()
        disc.residual_partials(inputs, outputs, jac)

        self.assertEqual(jac[("x", "a")][0], 16.0)
        self.assertEqual(jac[("x", "b")][0], 4.0)
        self.assertEqual(jac[("x", "c")][0], 1.0)
        self.assertEqual(jac[("x", "x")][0], 10.0)

    def test_apply_linear_fwd_mode_all_inputs(self):
        """
        Tests the apply_linear function in forward mode with all inputs.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        d_inputs = {"a": np.array([0.1]), "b": np.array([0.2]), "c": np.array([0.3])}
        d_outputs = {"x": np.array([0.4])}
        d_residuals = {"x": np.array([0.0])}
        
        disc = QuadradicImplicit()
        disc.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "fwd")
        
        # Expected: (2*a*x + b)*d_x + x^2*d_a + x*d_b + d_c
        # = (2*1*4 + 2)*0.4 + 4^2*0.1 + 4*0.2 + 0.3 = 10*0.4 + 16*0.1 + 0.8 + 0.3 = 6.7
        self.assertAlmostEqual(d_residuals["x"][0], 6.7, places=8)

    def test_apply_linear_fwd_mode_partial_inputs(self):
        """
        Tests the apply_linear function in forward mode with partial inputs.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        d_inputs = {"a": np.array([0.1])}  # Only d_a provided
        d_outputs = {"x": np.array([0.4])}
        d_residuals = {"x": np.array([0.0])}
        
        disc = QuadradicImplicit()
        disc.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "fwd")
        
        # Expected: (2*a*x + b)*d_x + x^2*d_a = 10*0.4 + 16*0.1 = 5.6
        self.assertAlmostEqual(d_residuals["x"][0], 5.6, places=8)

    def test_apply_linear_rev_mode_all_inputs(self):
        """
        Tests the apply_linear function in reverse mode with all inputs.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        d_inputs = {"a": np.array([0.0]), "b": np.array([0.0]), "c": np.array([0.0])}
        d_outputs = {"x": np.array([0.0])}
        d_residuals = {"x": np.array([0.5])}
        
        disc = QuadradicImplicit()
        disc.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "rev")
        
        # Expected derivatives based on reverse mode
        self.assertAlmostEqual(d_outputs["x"][0], 5.0, places=8)  # (2*a*x + b) * d_residuals = 10 * 0.5
        self.assertAlmostEqual(d_inputs["a"][0], 8.0, places=8)   # x^2 * d_residuals = 16 * 0.5
        self.assertAlmostEqual(d_inputs["b"][0], 2.0, places=8)   # x * d_residuals = 4 * 0.5
        self.assertAlmostEqual(d_inputs["c"][0], 0.5, places=8)   # d_residuals = 0.5

    def test_apply_linear_rev_mode_partial_outputs(self):
        """
        Tests the apply_linear function in reverse mode with partial outputs.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        d_inputs = {"a": np.array([0.0])}  # Only d_a provided
        d_outputs = {}  # No d_outputs
        d_residuals = {"x": np.array([0.5])}
        
        disc = QuadradicImplicit()
        disc.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "rev")
        
        # Only d_a should be modified
        self.assertAlmostEqual(d_inputs["a"][0], 8.0, places=8)   # x^2 * d_residuals = 16 * 0.5

    def test_apply_linear_no_x_in_residuals(self):
        """
        Tests the apply_linear function when 'x' is not in d_residuals.
        """
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        d_inputs = {"a": np.array([0.1]), "b": np.array([0.2]), "c": np.array([0.3])}
        d_outputs = {"x": np.array([0.4])}
        d_residuals = {}  # No 'x' in residuals
        
        disc = QuadradicImplicit()
        disc.apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, "fwd")
        
        # Nothing should change since 'x' not in d_residuals
        self.assertEqual(len(d_residuals), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
