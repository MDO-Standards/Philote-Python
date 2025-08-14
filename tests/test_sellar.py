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
import openmdao.api as om
from philote_mdo.examples.sellar import SellarMDA, SellarGroup


class TestSellarMDA(unittest.TestCase):
    """
    Unit tests for the SellarMDA group.
    """

    def test_setup(self):
        """
        Tests the setup function of the SellarMDA group.
        """
        group = SellarMDA()
        prob = om.Problem(model=group)
        prob.setup()
        prob.run_model()
        
        # Test that all subsystems were added
        self.assertIn('cycle', group._subsystems_allprocs)
        self.assertIn('obj_cmp', group._subsystems_allprocs)
        self.assertIn('con_cmp1', group._subsystems_allprocs)
        self.assertIn('con_cmp2', group._subsystems_allprocs)
        
        # Test that cycle has the right subsystems
        cycle = group.cycle
        self.assertIn('d1', cycle._subsystems_allprocs)
        self.assertIn('d2', cycle._subsystems_allprocs)
        
        # Test that variables are properly set up (values may vary by OpenMDAO version)
        x_val = prob.get_val('x')
        self.assertIsInstance(x_val, np.ndarray)
        self.assertEqual(len(x_val), 1)
        
        z_val = prob.get_val('z')
        self.assertIsInstance(z_val, np.ndarray)
        self.assertEqual(len(z_val), 2)  # z should have 2 elements
        np.testing.assert_allclose(z_val, np.array([5.0, 2.0]))

    def test_run_model(self):
        """
        Tests running the SellarMDA model.
        """
        group = SellarMDA()
        prob = om.Problem(model=group)
        prob.setup()
        prob.run_model()
        
        # Check that outputs are reasonable
        y1 = prob.get_val('y1')
        y2 = prob.get_val('y2')
        obj = prob.get_val('obj')
        con1 = prob.get_val('con1')
        con2 = prob.get_val('con2')
        
        # These should be positive values for the default inputs
        self.assertGreater(y1[0], 0)
        self.assertGreater(y2[0], 0)
        self.assertGreater(obj[0], 0)
        
        # Constraints should be computed correctly
        self.assertAlmostEqual(con1[0], 3.16 - y1[0], places=8)
        self.assertAlmostEqual(con2[0], y2[0] - 24.0, places=8)

    def test_with_different_inputs(self):
        """
        Tests the SellarMDA with different input values.
        """
        group = SellarMDA()
        prob = om.Problem(model=group)
        prob.setup()
        
        # Set different input values
        prob.set_val('x', 2.0)
        prob.set_val('z', np.array([3.0, 4.0]))
        
        prob.run_model()
        
        # Verify the model converged with different inputs
        y1 = prob.get_val('y1')
        y2 = prob.get_val('y2')
        
        self.assertGreater(y1[0], 0)
        self.assertGreater(y2[0], 0)


class TestSellarGroup(unittest.TestCase):
    """
    Unit tests for the SellarGroup class.
    """

    def test_initialize(self):
        """
        Tests the initialize function of the SellarGroup.
        """
        sellar_group = SellarGroup()
        sellar_group.initialize()
        
        # Check that the problem and model were set up
        self.assertIsNotNone(sellar_group._prob)
        self.assertIsNotNone(sellar_group._model)
        self.assertIsInstance(sellar_group._model, SellarMDA)

    def test_set_options(self):
        """
        Tests the set_options function of the SellarGroup.
        """
        sellar_group = SellarGroup()
        
        # This should run without error even though it does nothing
        sellar_group.set_options({})
        sellar_group.set_options({'some_option': 'value'})
        
        # No assertions needed since the method just passes

    def test_full_workflow(self):
        """
        Tests the complete workflow with SellarGroup.
        """
        sellar_group = SellarGroup()
        sellar_group.initialize()
        
        # The group should be properly initialized
        self.assertIsInstance(sellar_group._model, SellarMDA)
        
        # Test that we can use the internal problem
        sellar_group._prob.setup()
        sellar_group._prob.run_model()
        
        # Basic checks that the model runs
        y1 = sellar_group._prob.get_val('y1')
        y2 = sellar_group._prob.get_val('y2')
        self.assertGreater(y1[0], 0)
        self.assertGreater(y2[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
