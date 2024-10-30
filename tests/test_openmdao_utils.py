# Philote-Python
#
# Copyright 2022-2024 Christopher A. Lupp
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
from unittest.mock import Mock, MagicMock
import numpy as np

from philote_mdo.generated.data_pb2 import kInput, kOutput
import philote_mdo.openmdao.utils as utils


class TestOpenMdaoUtils(unittest.TestCase):
    """
    Tests the utility functions used by the OpenMDAO clients and servers.
    """

    def test_openmdao_client_setup(self):
        comp = Mock()
        var1 = Mock()
        var1.name = "var1"
        var1.units = "m"
        var1.type = kInput
        var1.shape = [2]

        var2 = Mock()
        var2.name = "var2"
        var2.units = None
        var2.type = kOutput
        var2.shape = [1]

        comp._client._var_meta = [var1, var2]

        utils.client_setup(comp)

        comp._client.run_setup.assert_called_once()
        comp._client.get_variable_definitions.assert_called_once()

        expected_calls = [
            ("add_input", ("var1",), {"shape": (2,), "units": "m"}),
            ("add_output", ("var2",), {"shape": (1,), "units": None})
        ]
        for call in expected_calls:
            getattr(comp, call[0]).assert_called_once_with(*call[1], **call[2])

    def test_openmdao_client_setup_partials(self):
        # mocking the component and its necessary attributes/methods
        comp_mock = MagicMock()

        par1 = Mock()
        par1.name = "partial1"
        par1.subname = "subpartial1"

        par2 = Mock()
        par2.name = "partial2"
        par2.subname = "subpartial2"

        comp_mock._client.get_partials_definitions.return_value = None
        comp_mock._client._partials_meta = [par1, par2]

        # call the function
        utils.client_setup_partials(comp_mock)

        # assert that the necessary methods are called and that
        # declare_partials is called for each partial
        comp_mock._client.get_partials_definitions.assert_called_once()
        comp_mock.declare_partials.assert_any_call('partial1', 'subpartial1')
        comp_mock.declare_partials.assert_any_call('partial2', 'subpartial2')

    def test_create_local_inputs(self):
        # define sample inputs and var_meta
        inputs = {'var1': 10, 'var2': 20, 'var3': 30}

        # case 1: 3 inputs
        # ----------------

        # create variable metadata
        var1 = Mock()
        var1.name = "var1"
        var1.type = kInput

        var2 = Mock()
        var2.name = "var2"
        var2.type = kInput

        var3 = Mock()
        var3.name = "var3"
        var3.type = kInput

        var_meta = [var1, var2, var3]

        # call the function
        local_inputs1 = utils.create_local_inputs(inputs, var_meta)

        # assert that only relative variable names are included in local_inputs
        self.assertIn('var1', local_inputs1)
        self.assertIn('var2', local_inputs1)
        self.assertIn('var3', local_inputs1)
        self.assertEqual(local_inputs1['var1'], 10)
        self.assertEqual(local_inputs1['var2'], 20)
        self.assertEqual(local_inputs1['var3'], 30)

        # case 2: 2 inputs, 1 output
        # --------------------------
        var2.type = kOutput

        local_inputs2 = utils.create_local_inputs(inputs, var_meta)

        # assert that only relative variable names are included in local_inputs
        self.assertIn('var1', local_inputs2)
        self.assertNotIn('var2', local_inputs2)
        self.assertIn('var3', local_inputs2)
        self.assertEqual(local_inputs2['var1'], 10)
        self.assertEqual(local_inputs2['var3'], 30)

        # case 2: 1 input, 2 outputs
        # --------------------------
        var3.type = kOutput

        local_inputs3 = utils.create_local_inputs(inputs, var_meta, kOutput)

        # assert that only relative variable names are included in local_inputs
        self.assertNotIn('var1', local_inputs3)
        self.assertIn('var2', local_inputs3)
        self.assertIn('var3', local_inputs3)
        self.assertEqual(local_inputs3['var2'], 20)
        self.assertEqual(local_inputs3['var3'], 30)

    def test_create_local_outputs(self):
        # Define sample out and outputs dictionaries
        out = {'output1': 10, 'output2': 20}
        outputs = {'output1': None, 'output2': None, 'output3': None}

        # call the function
        utils.assign_global_outputs(out, outputs)

        # assert that the values in outputs are updated correctly
        self.assertEqual(outputs['output1'], 10)
        self.assertEqual(outputs['output2'], 20)
        # ensure that other keys in outputs are unchanged
        self.assertEqual(outputs['output3'], None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
