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
from unittest.mock import Mock, MagicMock
from philote_mdo.general import DisciplineServer, DisciplineClient, ExplicitDiscipline
import philote_mdo.generated.data_pb2 as data


class TestDisciplineServerEdgeCases(unittest.TestCase):
    """
    Tests for edge cases in DisciplineServer.
    """

    def test_attach_discipline(self):
        """
        Test attaching a discipline to the server (line 62).
        """
        server = DisciplineServer()
        discipline = ExplicitDiscipline()
        
        # Test attach_discipline method
        server.attach_discipline(discipline)
        
        # Verify the discipline was attached
        self.assertEqual(server._discipline, discipline)

    def test_get_available_options_with_str_type(self):
        """
        Test GetAvailableOptions with str option type (covers line 101).
        """
        server = DisciplineServer()
        discipline = Mock()
        
        # Mock the options_list attribute to return a dict with str type
        discipline.options_list = {"str_option": "str"}
        
        server.attach_discipline(discipline)
        
        # Create a mock request and context
        request = Mock()
        context = Mock()
        
        # This should work and exercise the str type branch
        result = server.GetAvailableOptions(request, context)
        
        # The method should complete without error and return options
        self.assertIsNotNone(result)

    def test_get_available_options_with_invalid_type(self):
        """
        Test GetAvailableOptions with invalid option type (covers lines 100-103).
        """
        server = DisciplineServer()
        discipline = Mock()
        
        # Mock the options_list attribute to return a dict with invalid type
        discipline.options_list = {"invalid_option": "invalid_type"}
        
        server.attach_discipline(discipline)
        
        # Create a mock request and context
        request = Mock()
        context = Mock()
        
        # This should raise a ValueError
        with self.assertRaises(ValueError) as context_err:
            server.GetAvailableOptions(request, context)
        
        self.assertIn("Invalid value for discipline option", str(context_err.exception))
        self.assertIn("invalid_option", str(context_err.exception))

    def test_process_inputs_with_empty_continuous_data(self):
        """
        Test process_inputs with empty continuous data arrays (line 216).
        """
        server = DisciplineServer()
        discipline = Mock()
        
        # Set up discipline with continuous variables
        discipline._is_continuous = True
        discipline._var_meta = [Mock()]
        discipline._var_meta[0].name = "test_var"
        discipline._var_meta[0].shape = [2]
        discipline._var_meta[0].type = data.kInput
        
        server.attach_discipline(discipline)
        
        # Create a message with empty data
        message = Mock()
        message.name = "test_var"
        message.type = data.VariableType.kInput
        message.start = 0
        message.end = 1
        message.data = []  # Empty data array
        
        # Create mock for flat_inputs and flat_outputs
        flat_inputs = {"test_var": [0.0, 0.0]}
        flat_outputs = {}
        
        # This should raise a ValueError
        with self.assertRaises(ValueError) as context:
            server.process_inputs([message], flat_inputs, flat_outputs)
        
        self.assertIn("Expected continuous variables but arrays were empty", str(context.exception))


class TestDisciplineClientEdgeCases(unittest.TestCase):
    """
    Tests for edge cases in DisciplineClient.
    """

    def test_recover_outputs_with_empty_data(self):
        """
        Test _recover_outputs with empty data arrays (line 197).
        """
        # Create a mock channel
        channel = Mock()
        client = DisciplineClient(channel)
        
        # Set up outputs structure
        client._var_meta = [Mock()]
        client._var_meta[0].name = "test_output"
        client._var_meta[0].shape = [2]
        client._var_meta[0].type = data.kOutput
        
        # Create a response message with empty data
        message = Mock()
        message.name = "test_output"
        message.type = data.kOutput
        message.start = 0
        message.end = 1
        message.data = []  # Empty data array
        
        # This should raise a ValueError
        with self.assertRaises(ValueError) as context:
            client._recover_outputs([message])
        
        self.assertIn("Expected continuous variables, but array is empty", str(context.exception))

    # NOTE: Other client edge case tests are more complex to set up properly
    # and would require extensive mocking. The _recover_outputs test above
    # demonstrates the pattern for testing these error paths.


if __name__ == "__main__":
    unittest.main(verbosity=2)
