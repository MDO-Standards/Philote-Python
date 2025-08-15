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
from philote_mdo.openmdao.group import OpenMdaoSubProblem


class SimpleGroup(om.Group):
    """Simple test group for OpenMDAO testing."""
    
    def setup(self):
        self.add_subsystem('comp', om.ExecComp('y = 2*x + 1'), promotes=['*'])


class TestOpenMdaoSubProblem(unittest.TestCase):
    """
    Tests for the OpenMdaoSubProblem class.
    """

    def test_add_mapped_input(self):
        """
        Test adding mapped inputs to the sub-problem.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add a mapped input
        subprob.add_mapped_input('local_x', 'x', shape=(2,), units='m')
        
        # Check that the input was added to the map
        self.assertIn('local_x', subprob._input_map)
        self.assertEqual(subprob._input_map['local_x']['sub_prob_name'], 'x')
        self.assertEqual(subprob._input_map['local_x']['shape'], (2,))
        self.assertEqual(subprob._input_map['local_x']['units'], 'm')

    def test_add_mapped_output(self):
        """
        Test adding mapped outputs to the sub-problem.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add a mapped output
        subprob.add_mapped_output('local_y', 'y', shape=(3,), units='kg')
        
        # Check that the output was added to the map
        self.assertIn('local_y', subprob._output_map)
        self.assertEqual(subprob._output_map['local_y']['sub_prob_name'], 'y')
        self.assertEqual(subprob._output_map['local_y']['shape'], (3,))
        self.assertEqual(subprob._output_map['local_y']['units'], 'kg')

    def test_clear_mapped_variables(self):
        """
        Test clearing mapped variables.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add some mapped variables
        subprob.add_mapped_input('x1', 'sub_x1')
        subprob.add_mapped_output('y1', 'sub_y1')
        
        # Verify they exist
        self.assertEqual(len(subprob._input_map), 1)
        self.assertEqual(len(subprob._output_map), 1)
        
        # Clear the maps
        subprob.clear_mapped_variables()
        
        # Verify they are cleared
        self.assertEqual(len(subprob._input_map), 0)
        self.assertEqual(len(subprob._output_map), 0)

    def test_declare_subproblem_partial(self):
        """
        Test declaring partials for the sub-problem.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add mapped variables first
        subprob.add_mapped_input('local_x', 'x')
        subprob.add_mapped_output('local_y', 'y')
        
        # Declare partial
        subprob.declare_subproblem_partial('local_y', 'local_x')
        
        # Check that the partial was added to the map
        self.assertIn(('local_y', 'local_x'), subprob._partials_map)
        self.assertEqual(subprob._partials_map[('local_y', 'local_x')], ('y', 'x'))

    def test_initialize(self):
        """
        Test the initialize method (currently does nothing).
        """
        subprob = OpenMdaoSubProblem()
        
        # This should run without error
        result = subprob.initialize()
        self.assertIsNone(result)

    def test_full_workflow_setup_compute(self):
        """
        Test the complete workflow: setup and compute.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add a simple OpenMDAO group
        group = SimpleGroup()
        subprob.add_group(group)
        
        # Add mapped variables
        subprob.add_mapped_input('local_x', 'x')
        subprob.add_mapped_output('local_y', 'y')
        
        # Set up the subproblem
        subprob.setup()
        
        # Verify that the discipline has the expected inputs and outputs
        # Note: We can't easily test the internal state without accessing private members,
        # but we can verify the setup completed without error
        self.assertIsNotNone(subprob._prob)
        self.assertIsNotNone(subprob._model)

    def test_compute_with_simple_model(self):
        """
        Test the compute method with a simple model.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add a simple OpenMDAO group
        group = SimpleGroup()
        subprob.add_group(group)
        
        # Add mapped variables
        subprob.add_mapped_input('local_x', 'x')
        subprob.add_mapped_output('local_y', 'y')
        
        # Setup the sub-problem
        subprob.setup()
        
        # Create inputs and outputs
        inputs = {'local_x': np.array([3.0])}
        outputs = {'local_y': np.array([0.0])}
        
        # Run compute
        subprob.compute(inputs, outputs)
        
        # Check the result (y = 2*x + 1 = 2*3 + 1 = 7)
        expected = 2.0 * 3.0 + 1.0
        self.assertAlmostEqual(outputs['local_y'][0], expected, places=6)

    def test_compute_partials_with_simple_model(self):
        """
        Test the compute_partials method with a simple model.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add a simple OpenMDAO group
        group = SimpleGroup()
        subprob.add_group(group)
        
        # Add mapped variables
        subprob.add_mapped_input('local_x', 'x')
        subprob.add_mapped_output('local_y', 'y')
        
        # Declare partial derivative
        subprob.declare_subproblem_partial('local_y', 'local_x')
        
        # Setup the sub-problem
        subprob.setup()
        
        # Create inputs and partials
        inputs = {'local_x': np.array([3.0])}
        partials = {('local_y', 'local_x'): np.array([0.0])}
        
        # Run compute_partials
        subprob.compute_partials(inputs, partials)
        
        # Check the partial derivative (dy/dx = 2)
        self.assertAlmostEqual(partials[('local_y', 'local_x')][0], 2.0, places=6)

    def test_multiple_variables_workflow(self):
        """
        Test workflow with multiple mapped variables.
        """
        subprob = OpenMdaoSubProblem()
        
        # Create a group with multiple variables
        group = om.Group()
        group.add_subsystem('comp', om.ExecComp(['y1 = 2*x1 + 1', 'y2 = x1 * x2']), 
                           promotes=['*'])
        subprob.add_group(group)
        
        # Add multiple mapped variables
        subprob.add_mapped_input('local_x1', 'x1')
        subprob.add_mapped_input('local_x2', 'x2')
        subprob.add_mapped_output('local_y1', 'y1')
        subprob.add_mapped_output('local_y2', 'y2')
        
        # Declare partials
        subprob.declare_subproblem_partial('local_y1', 'local_x1')
        subprob.declare_subproblem_partial('local_y2', 'local_x1')
        subprob.declare_subproblem_partial('local_y2', 'local_x2')
        
        # Setup
        subprob.setup()
        
        # Test compute
        inputs = {'local_x1': np.array([3.0]), 'local_x2': np.array([4.0])}
        outputs = {'local_y1': np.array([0.0]), 'local_y2': np.array([0.0])}
        
        subprob.compute(inputs, outputs)
        
        # Check results
        self.assertAlmostEqual(outputs['local_y1'][0], 7.0, places=6)  # 2*3 + 1
        self.assertAlmostEqual(outputs['local_y2'][0], 12.0, places=6)  # 3*4

    def test_units_and_shapes(self):
        """
        Test that units and shapes are properly handled.
        """
        subprob = OpenMdaoSubProblem()
        
        # Add variables with specific units and shapes
        subprob.add_mapped_input('vector_in', 'x', shape=(3,), units='m')
        subprob.add_mapped_output('vector_out', 'y', shape=(3,), units='kg')
        
        # Verify the mappings include units and shapes
        self.assertEqual(subprob._input_map['vector_in']['shape'], (3,))
        self.assertEqual(subprob._input_map['vector_in']['units'], 'm')
        self.assertEqual(subprob._output_map['vector_out']['shape'], (3,))
        self.assertEqual(subprob._output_map['vector_out']['units'], 'kg')


if __name__ == "__main__":
    unittest.main(verbosity=2)
