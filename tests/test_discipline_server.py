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
from unittest.mock import Mock

import numpy as np

from google.protobuf.empty_pb2 import Empty

from philote_mdo.general import Discipline, DisciplineServer
import philote_mdo.generated.data_pb2 as data


class TestDisciplineServer(unittest.TestCase):
    """
    Unit tests for the discipline server.
    """

    def test_get_info(self):
        """
        Tests the GetInfo RPC of the Discipline Server.
        """
        server = DisciplineServer()
        server._discipline = Discipline()
        server._discipline._is_continuous = True
        server._discipline._is_differentiable = True
        server._discipline._provides_gradients = True

        # mock arguments
        context = Mock()
        request = Empty()

        response_generator = server.GetInfo(request, context)

        # Generate responses and collect them into a list
        responses = list(response_generator)

        # check that there is only one response
        self.assertEqual(len(responses), 1)

        # check the values of the response
        response = responses[0]
        self.assertTrue(response.continuous)
        self.assertTrue(response.differentiable)
        self.assertTrue(response.provides_gradients)

    def test_set_stream_options(self):
        """
        Tests the SetStreamOptions RPC of the Discipline Server.
        """
        server = DisciplineServer()

        # mock arguments
        context = Mock()
        request = data.StreamOptions(num_double=2)

        server.SetStreamOptions(request, context)

        # check that the streaming options were set properly
        self.assertEqual(server._stream_opts.num_double, 2)

    def test_get_available_options(self):
        server = DisciplineServer()

        # mock the request and context parameters (since they are not used in this function)
        request_mock = Mock()
        context_mock = None

        server._discipline = Discipline()

        # set the mock options_list to _discipline.options_list
        server._discipline.options_list = {'option1': 'bool',
                                           'option2': 'int',
                                           'option3': 'float'}

        # call the function
        results = server.GetAvailableOptions(request_mock, context_mock)

        # assert that the results are correct
        expected_options = ['option1', 'option2', 'option3']
        expected_types = [data.kBool, data.kInt, data.kDouble]

        self.assertEqual(results.options, expected_options)
        self.assertEqual(results.type, expected_types)

    def test_set_options(self):
        server = DisciplineServer()

        # mock the request and context parameters
        request_mock = Mock()
        context_mock = Mock()

        # set some mock options in the request
        request_mock.options = {"key1": "value1", "key2": 42}

        # create a mock for the _discipline attribute
        discipline_mock = Mock()
        server._discipline = discipline_mock

        # call the SetOptions function with the mock parameters
        server.SetOptions(request_mock, context_mock)

        # assert that the discipline's initialize method was called with the expected options
        server._discipline.set_options.assert_called_once_with(
            {"key1": "value1", "key2": 42}
        )

    def test_setup(self):
        """
        Tests the Setup RPC of the Discipline Server.
        """
        context = Mock()
        request = Empty()

        server = DisciplineServer()

        # mock the 'setup' and 'setup_partials' methods of 'self.discipline'
        server._discipline = Mock()
        server._discipline.setup.return_value = None
        server._discipline.setup_partials.return_value = None

        server.Setup(request, context)

        # assert that the 'setup' and 'setup_partials' methods were called
        server._discipline.setup.assert_called_once()
        server._discipline.setup_partials.assert_called_once()

    def test_get_variable_definitions(self):
        """
        Tests the GetVariableDefinitions RPC of the Discipline Server.
        """
        server = DisciplineServer()
        server._discipline = Discipline()

        # add an input and an output
        server._discipline.add_input("x", shape=(2, 2), units="m")
        server._discipline.add_output("f", shape=(1,), units="m**2")

        # mock arguments
        context = Mock()
        request = Empty()

        response_generator = server.GetVariableDefinitions(request, context)

        # Generate responses and collect them into a list
        responses = list(response_generator)

        # check that there are two responses (one input, one output)
        self.assertEqual(len(responses), 2)

        # check the response data
        input = responses[0]
        output = responses[1]

        self.assertEqual(input.name, "x")
        self.assertEqual(input.shape, [2, 2])
        self.assertEqual(input.units, "m")
        self.assertEqual(input.type, data.kInput)

        self.assertEqual(output.name, "f")
        self.assertEqual(
            output.shape,
            [
                1,
            ],
        )
        self.assertEqual(output.units, "m**2")
        self.assertEqual(output.type, data.kOutput)

    # def test_get_partials_definition(self):
    #     """
    #     Tests the GetPartialDefinitions RPC of Discipline Server.
    #     """
    #     pass

    def test_preallocate_inputs_explicit(self):
        """
        Tests the preallocation of inputs for the explicit discipline cas of the
        Discipline Server (outputs are not an input).
        """
        server = DisciplineServer()
        discipline = server._discipline = Discipline()
        discipline.add_input("x", shape=(2, 2), units="m")
        discipline.add_input("y", shape=(3, 3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        # create simulated inputs for the function
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        server.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)

        # check the number of inputs and outputs
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(flat_inputs), 2)

        for var, shape in zip(["x", "y"], [(2, 2), (3, 3, 3)]):
            # check variable existence
            self.assertTrue(var in inputs)
            self.assertTrue(var in flat_inputs)

            # check that variables are numpy arrays
            self.assertIsInstance(inputs[var], np.ndarray)
            self.assertIsInstance(flat_inputs[var], np.ndarray)

            # check that the variables have the right shape
            self.assertEqual(inputs[var].shape, shape)
            self.assertEqual(flat_inputs[var].size, np.prod(shape))

    def test_preallocate_inputs_implicit(self):
        """
        Tests the preallocation of inputs for the implicit discipline cas of the
        Discipline Server (outputs are an input).
        """
        server = DisciplineServer()
        discipline = server._discipline = Discipline()
        discipline.add_input("x", shape=(2, 2), units="m")
        discipline.add_input("y", shape=(3, 3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        # create simulated inputs for the function
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        server.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)

        # check the number of inputs and outputs
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(flat_inputs), 2)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(flat_outputs), 2)

        # check inputs
        for var, shape in zip(["x", "y"], [(2, 2), (3, 3, 3)]):
            # check variable existence
            self.assertTrue(var in inputs)
            self.assertTrue(var in flat_inputs)

            # check that variables are numpy arrays
            self.assertIsInstance(inputs[var], np.ndarray)
            self.assertIsInstance(flat_inputs[var], np.ndarray)

            # check that the variables have the right shape
            self.assertEqual(inputs[var].shape, shape)
            self.assertEqual(flat_inputs[var].size, np.prod(shape))

        # check outputs
        for out, shape in zip(["f1", "f2"], [(1,), (2, 3)]):
            # check variable existence
            self.assertTrue(out in outputs)
            self.assertTrue(out in flat_outputs)

            # check that variables are numpy arrays
            self.assertIsInstance(outputs[out], np.ndarray)
            self.assertIsInstance(flat_outputs[out], np.ndarray)

            # check that the variables have the right shape
            self.assertEqual(outputs[out].shape, shape)
            self.assertEqual(flat_outputs[out].size, np.prod(shape))

    def test_preallocate_partials(self):
        """
        Tests the preallocation of the partial derivatives of the Discipline Server.

        This test is designed to catch the edge cases where either f or x are
        scalar.
        """
        server = DisciplineServer()
        discipline = server._discipline = Discipline()
        discipline.add_input("x", shape=(1,), units="m")
        discipline.add_input("y", shape=(3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        discipline.declare_partials("f1", "x")
        discipline.declare_partials("f1", "y")
        discipline.declare_partials("f2", "x")
        discipline.declare_partials("f2", "y")

        jac = server.preallocate_partials()

        pairs = [("f1", "x"), ("f1", "y"), ("f2", "x"), ("f2", "y")]
        expected_shapes = [(1,), (3, 3), (2, 3), (2, 3, 3, 3)]

        for pair, shape in zip(pairs, expected_shapes):
            self.assertTrue(pair in jac)
            self.assertIsInstance(jac[pair], np.ndarray)
            self.assertEqual(jac[pair].shape, shape)

    def test_process_inputs(self):
        # create a mock request_iterator
        request_iterator = [
            data.Array(
                start=0,
                end=2,
                data=[1.0, 2.0, 3.0],
                type=data.VariableType.kInput,
                name="x",
            ),
            data.Array(
                start=3, end=4, data=[4.0, 5.0], type=data.VariableType.kInput, name="x"
            ),
            data.Array(
                start=0,
                end=1,
                data=[0.1, 0.2],
                type=data.VariableType.kOutput,
                name="f",
            ),
        ]

        server = DisciplineServer()

        # create mock flat_inputs and flat_outputs dictionaries
        flat_inputs = {"x": np.zeros(6)}
        flat_outputs = {"f": np.zeros(3)}

        server.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # check the results
        self.assertEqual(flat_inputs["x"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
        self.assertEqual(flat_outputs["f"].tolist(), [0.1, 0.2, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
