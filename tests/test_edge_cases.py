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

from philote_mdo.general import Discipline
from conftest import job_context, make_server, make_server_from_instance, bind_job
from unittest.mock import Mock, MagicMock

import grpc

from philote_mdo.general import DisciplineServer, DisciplineClient, ExplicitDiscipline
from philote_mdo.utils.validation import PhiloteValidationError
import philote_mdo.generated.data_pb2 as data


class TestDisciplineServerEdgeCases(unittest.TestCase):
    """
    Tests for edge cases in DisciplineServer.
    """

    def test_attach_discipline_factory(self):
        """
        Test attaching a discipline to the server (line 62).
        """
        discipline = ExplicitDiscipline()
        
        # Test attach_discipline method
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )
        
        # Verify the discipline was attached
        self.assertEqual(job.discipline, discipline)

    def test_get_available_options_with_str_type(self):
        """
        Test GetAvailableOptions with str option type (covers line 101).
        """
        discipline = Mock()
        
        # Mock the options_list attribute to return a dict with str type
        discipline.options_list = {"str_option": "str"}
        
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )
        
        # Create a mock request and context
        request = Mock()
        context = job_context(job=job)
        
        # This should work and exercise the str type branch
        result = server.GetAvailableOptions(request, context)
        
        # The method should complete without error and return options
        self.assertIsNotNone(result)

    def test_get_available_options_with_dict_type(self):
        """
        Test GetAvailableOptions with dict option type (covers kStruct mapping).
        """
        discipline = Mock()

        discipline.options_list = {"config": "dict"}

        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        request = Mock()
        context = job_context(job=job)

        result = server.GetAvailableOptions(request, context)

        self.assertIsNotNone(result)
        self.assertEqual(list(result.options), ["config"])
        self.assertEqual(list(result.type), [data.kStruct])

    def test_get_available_options_with_invalid_type(self):
        """
        Test GetAvailableOptions with invalid option type aborts with
        INVALID_ARGUMENT.
        """
        discipline = Mock()

        # Mock the options_list attribute to return a dict with invalid type
        discipline.options_list = {"invalid_option": "invalid_type"}

        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # Create a mock request and context
        request = Mock()
        context = job_context(job=job)

        server.GetAvailableOptions(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Invalid value for discipline option", args[0][1])

    def test_process_inputs_with_empty_continuous_data(self):
        """
        Test process_inputs with empty continuous data arrays.
        """
        discipline = Mock()

        # Set up discipline with continuous variables
        discipline._is_continuous = True
        discipline._var_meta = [Mock()]
        discipline._var_meta[0].name = "test_var"
        discipline._var_meta[0].shape = [2]
        discipline._var_meta[0].type = data.kInput

        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # Create a VariableMessage wrapping an Array with empty data
        message = data.VariableMessage(
            continuous=data.Array(
                name="test_var",
                type=data.VariableType.kInput,
                start=0,
                end=1,
                data=[],
            )
        )

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
        Test _recover_outputs with empty data arrays.
        """
        # Create a mock channel
        channel = Mock()
        client = bind_job(DisciplineClient(channel))

        # Set up outputs structure
        client._var_meta = [Mock()]
        client._var_meta[0].name = "test_output"
        client._var_meta[0].shape = [2]
        client._var_meta[0].type = data.kOutput

        # Create a VariableMessage wrapping an Array with empty data
        message = data.VariableMessage(
            continuous=data.Array(
                name="test_output",
                type=data.kOutput,
                start=0,
                end=1,
                data=[],
            )
        )

        # This should raise a ValueError
        with self.assertRaises(ValueError) as context:
            client._recover_outputs([message])

        self.assertIn("Expected continuous variables, but array is empty", str(context.exception))

    # NOTE: Other client edge case tests are more complex to set up properly
    # and would require extensive mocking. The _recover_outputs test above
    # demonstrates the pattern for testing these error paths.


if __name__ == "__main__":
    unittest.main(verbosity=2)
