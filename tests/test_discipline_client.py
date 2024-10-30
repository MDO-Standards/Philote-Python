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
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from google.protobuf.empty_pb2 import Empty
from google.protobuf.struct_pb2 import Struct
from philote_mdo.general import DisciplineClient
import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc
import philote_mdo.utils as utils


class TestDisciplineClient(unittest.TestCase):
    """
    Unit tests for the discipline client.
    """

    def test_init(self):
        # Create a mock grpc channel
        mock_channel = Mock()

        # Create an instance of YourClass with the mock channel
        instance = DisciplineClient(mock_channel)

        # Assert that the attributes are initialized correctly
        self.assertTrue(instance.verbose)
        self.assertEqual(instance.grpc_options, [])
        self.assertFalse(instance._is_continuous)
        self.assertFalse(instance._is_differentiable)
        self.assertFalse(instance._provides_gradients)
        self.assertIsInstance(instance._disc_stub, disc.DisciplineServiceStub)
        self.assertEqual(instance._stream_options.num_double, 1000)
        self.assertEqual(instance._var_meta, [])
        self.assertEqual(instance._partials_meta, [])
        self.assertEqual(instance.options_list, {})


    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_get_discipline_info(self, mock_discipline_stub):
        """
        Tests the get_discipline_info function of the Discipline Client.
        """
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value
        mock_stub.GetInfo.return_value = [
            data.DisciplineProperties(
                continuous=True, differentiable=True, provides_gradients=True
            )
        ]

        client = DisciplineClient(mock_channel)
        client.get_discipline_info()

        # check the values of the response
        self.assertTrue(client._is_continuous)
        self.assertTrue(client._is_differentiable)
        self.assertTrue(client._provides_gradients)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_send_stream_options(self, mock_discipline_stub):
        """
        Tests the set_stream_options function of the Discipline Client.
        """
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value

        client = DisciplineClient(mock_channel)
        expected_num_double = 10
        client._stream_options = expected_options = data.StreamOptions(
            num_double=expected_num_double,
        )
        client.send_stream_options()

        # check that the streaming options were set properly
        expected_options = data.StreamOptions(
            num_double=expected_num_double,
        )
        self.assertTrue(mock_stub.SetStreamOptions.called)
        mock_stub.SetStreamOptions.assert_called_with(expected_options)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_get_available_options(self, mock_discipline_stub):
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value

        instance = DisciplineClient(channel=mock_channel)

        # mock the _disc_stub.GetAvailableOptions method
        mock_options = MagicMock()
        mock_options.options = ['option1', 'option2', 'option3', 'option4']
        mock_options.type = [data.kBool, data.kDouble, data.kString, data.kInt]
        instance._disc_stub.GetAvailableOptions.return_value = mock_options

        # call the get_available_options method
        instance.get_available_options()

        # assert that options_list is populated correctly
        expected_options_list = {'option1': 'bool', 'option2': 'float', 'option3': 'str', 'option4': 'int'}
        self.assertEqual(instance.options_list, expected_options_list)

    def test_send_options(self):
        # mock gRPC stub and channel
        mock_stub = Mock()
        mock_channel = Mock()
        mock_channel.stub.return_value = mock_stub

        # create an instance of YourClass with the mocked channel
        client = DisciplineClient(channel=mock_channel)
        client._disc_stub = mock_stub

        # mock options dictionary
        options = {"key1": "value1", "key2": 42}

        # call the send_options method
        client.send_options(options)

        # assert that SetOptions was called with the expected ProtoBuf structure
        expected_proto_options = data.DisciplineOptions()
        expected_proto_options.options.update(options)
        mock_stub.SetOptions.assert_called_once_with(expected_proto_options)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_run_setup(self, mock_discipline_stub):
        """
        Tests the run_setup function of the Discipline Client.
        """
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value
        client = DisciplineClient(mock_channel)
        client.run_setup()

        # assert that the 'setup' and 'setup_partials' methods were called
        self.assertTrue(mock_stub.Setup.called)
        mock_stub.Setup.assert_called_with(Empty())

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_get_variable_definitions(self, mock_discipline_stub):
        """
        Tests the get_variable_definitions function of the Discipline Client.
        """
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value
        client = DisciplineClient(mock_channel)

        input_definition = data.VariableMetaData(
            name="x", shape=[2, 2], units="m", type=data.VariableType.kInput
        )
        output_definition = data.VariableMetaData(
            name="f", shape=[1], units="m**2", type=data.VariableType.kOutput
        )

        # configure the stub to return the mock input and output definitions
        mock_stub.GetVariableDefinitions.return_value = [
            input_definition,
            output_definition,
        ]

        # call the get_variables_definitions method
        client.get_variable_definitions()

        # assert that the gRPC call was made
        self.assertTrue(mock_stub.GetVariableDefinitions.called)

        # assert that the _var_meta attribute is updated with the expected input
        # and output definitions
        self.assertEqual(len(client._var_meta), 2)

        # check the response data for input definition
        input = client._var_meta[0]
        self.assertEqual(input.name, "x")
        self.assertEqual(input.shape, [2, 2])
        self.assertEqual(input.units, "m")
        self.assertEqual(input.type, data.kInput)

        # check the response data for output definition
        output = client._var_meta[1]
        self.assertEqual(output.name, "f")
        self.assertEqual(output.shape, [1])
        self.assertEqual(output.units, "m**2")
        self.assertEqual(output.type, data.kOutput)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.DisciplineServiceStub")
    def test_get_partial_definitions(self, mock_discipline_stub):
        """
        Tests the get_partial_definitions function of the Discipline Client.
        """
        mock_channel = Mock()
        mock_stub = mock_discipline_stub.return_value
        client = DisciplineClient(mock_channel)

        partials_metadata = [
            data.PartialsMetaData(name="input1", subname="output1"),
            data.PartialsMetaData(name="input2", subname="output2"),
        ]

        mock_stub.GetPartialDefinitions.return_value = partials_metadata

        client.get_partials_definitions()

        # check that the client function was called
        self.assertTrue(mock_stub.GetPartialDefinitions.called)

        # check that the number of entries is correct
        self.assertEqual(len(client._partials_meta), 2)

        # check the values
        self.assertEqual(client._partials_meta[0].name, "input1")
        self.assertEqual(client._partials_meta[0].subname, "output1")
        self.assertEqual(client._partials_meta[1].name, "input2")
        self.assertEqual(client._partials_meta[1].subname, "output2")

    def test_assemble_input_messages(self):
        """
        Tests the _assemble_input_messages function of the Discipline Client.
        """
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)
        client._stream_options.num_double = 2

        input_data = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }
        output_data = {
            "f": np.array([5.0, 6.0, 7.0]).reshape(1, 3),
        }

        expected_messages = [
            data.Array(
                name="x", start=0, end=1, type=data.VariableType.kInput, data=[1.0, 2.0]
            ),
            data.Array(
                name="x", start=2, end=3, type=data.VariableType.kInput, data=[3.0, 4.0]
            ),
            data.Array(
                name="f",
                start=0,
                end=1,
                type=data.VariableType.kOutput,
                data=[5.0, 6.0],
            ),
            data.Array(
                name="f", start=2, end=2, type=data.VariableType.kOutput, data=[7.0]
            ),
        ]

        messages = client._assemble_input_messages(input_data, output_data)

        # check that the resulting messages match the expected messages
        self.assertEqual(len(messages), len(expected_messages))
        for msg, expected_msg in zip(messages, expected_messages):
            self.assertEqual(msg, expected_msg)

    def test_recover_outputs(self):
        """
        Tests the _recover_outputs function of the Discipline Client.
        """
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)

        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
        ]

        response1 = data.Array(
            name="f", start=0, end=1, type=data.kOutput, data=[1.0, 2.0]
        )
        response2 = data.Array(
            name="f", start=2, end=3, type=data.kOutput, data=[3.0, 4.0]
        )
        response3 = data.Array(
            name="g", start=0, end=2, type=data.kOutput, data=[4.0, 5.0, 6.0]
        )
        mock_responses = [response1, response2, response3]

        expected_outputs = {
            "f": np.array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2)),
            "g": np.array([4.0, 5.0, 6.0]),
        }

        outputs = client._recover_outputs(mock_responses)

        # check that the resulting outputs match the expected output data
        self.assertEqual(len(outputs), len(expected_outputs))
        for output_name, expected_data in expected_outputs.items():
            self.assertTrue(output_name in outputs)
            np.testing.assert_array_equal(outputs[output_name], expected_data)

    def test_recover_residuals(self):
        """
        Tests the _recover_residuals function of the Discipline Client.
        """
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)

        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kResidual, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]

        response1 = data.Array(
            name="f", start=0, end=1, type=data.kResidual, data=[1.0, 2.0]
        )
        response2 = data.Array(
            name="f", start=2, end=3, type=data.kResidual, data=[3.0, 4.0]
        )
        response3 = data.Array(
            name="g", start=0, end=2, type=data.kResidual, data=[4.0, 5.0, 6.0]
        )
        mock_responses = [response1, response2, response3]

        expected_outputs = {
            "f": np.array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2)),
            "g": np.array([4.0, 5.0, 6.0]),
        }

        outputs = client._recover_residuals(mock_responses)

        # check that the resulting outputs match the expected output data
        self.assertEqual(len(outputs), len(expected_outputs))
        for output_name, expected_data in expected_outputs.items():
            self.assertTrue(output_name in outputs)
            np.testing.assert_array_equal(outputs[output_name], expected_data)

    def test_recover_partials(self):
        """
        Tests the _recover_partials function of the Discipline Client.
        """
        mock_channel = Mock()
        client = DisciplineClient(mock_channel)

        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(1,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="y", type=data.kInput, shape=(1,)),
        ]

        partial_metadata1 = data.PartialsMetaData(name="f", subname="x")
        partial_metadata2 = data.PartialsMetaData(name="g", subname="y")
        client._partials_meta = [partial_metadata1, partial_metadata2]

        # Define mock responses
        response1 = data.Array(
            name="f", subname="x", type=data.kPartial, start=0, end=1, data=[1.0, 2.0]
        )
        response2 = data.Array(
            name="f", subname="x", type=data.kPartial, start=2, end=3, data=[3.0, 4.0]
        )
        response3 = data.Array(
            name="g",
            subname="y",
            type=data.kPartial,
            start=0,
            end=2,
            data=[4.0, 5.0, 6.0],
        )
        mock_responses = [response1, response2, response3]

        # Define expected partial data
        expected_partials = utils.PairDict()
        expected_flat_p = utils.PairDict()

        expected_partials[("f", "x")] = np.array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
        expected_partials[("g", "y")] = np.array([4.0, 5.0, 6.0])

        expected_flat_p[("f", "x")] = expected_partials[("f", "x")].ravel()
        expected_flat_p[("g", "y")] = expected_partials[("g", "y")].ravel()

        # Call the _recover_partials method with mock responses
        partials = client._recover_partials(mock_responses)

        # Assert that the resulting partials match the expected partial data
        self.assertEqual(len(partials), len(expected_partials))
        for (name, subname), expected_data in expected_partials.items():
            self.assertTrue((name, subname) in partials)
            np.testing.assert_array_equal(partials[(name, subname)], expected_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
