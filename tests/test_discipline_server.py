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

from conftest import job_context, make_server, make_server_from_instance
from unittest.mock import Mock

import grpc
import numpy as np

from google.protobuf.empty_pb2 import Empty

from philote_mdo.general import Discipline, DisciplineServer
from philote_mdo.utils.validation import PhiloteValidationError
import philote_mdo.generated.data_pb2 as data


class TestDisciplineServer(unittest.TestCase):
    """
    Unit tests for the discipline server.
    """

    def test_get_info(self):
        """
        Tests the GetInfo RPC of the Discipline Server.
        """
        discipline = Discipline()
        discipline._is_continuous = True
        discipline._is_differentiable = True
        discipline._provides_gradients = True
        discipline._name = "TestDiscipline"
        discipline._version = "1.2.3"

        # GetInfo answers from an instance of the server's own, so point the
        # factory at the one this test configured
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # mock arguments
        context = job_context(job=job)
        request = Empty()

        # GetInfo is a unary RPC, so it must return a message (not a generator)
        response = server.GetInfo(request, context)

        self.assertIsInstance(response, data.DisciplineProperties)

        # check the values of the response
        self.assertTrue(response.continuous)
        self.assertTrue(response.differentiable)
        self.assertTrue(response.provides_gradients)
        self.assertEqual(response.name, "TestDiscipline")
        self.assertEqual(response.version, "1.2.3")

    def test_set_stream_options(self):
        """
        Tests the SetStreamOptions RPC of the Discipline Server.
        """
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # mock arguments
        context = job_context(job=job)
        request = data.StreamOptions(num_double=2)

        server.SetStreamOptions(request, context)

        # check that the streaming options were set properly
        self.assertEqual(job.stream_opts.num_double, 2)

    def test_get_available_options(self):
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # mock the request and context parameters (since they are not used in this function)
        request_mock = Mock()
        context_mock = context

        # set the mock options_list to the discipline's options_list
        discipline.options_list = {
            "option1": "bool",
            "option2": "int",
            "option3": "float",
        }

        # call the function
        results = server.GetAvailableOptions(request_mock, context_mock)

        # assert that the results are correct
        expected_options = ["option1", "option2", "option3"]
        expected_types = [data.kBool, data.kInt, data.kDouble]

        self.assertEqual(results.options, expected_options)
        self.assertEqual(results.type, expected_types)

    def test_set_options(self):
        discipline = Mock()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # mock the request and context parameters
        request_mock = Mock()
        context_mock = job_context(job=job)

        # set some mock options in the request
        request_mock.options = {"key1": "value1", "key2": 42}

        # create a mock for the _discipline attribute
        discipline_mock = Mock()
        job.discipline = discipline_mock

        # call the SetOptions function with the mock parameters
        server.SetOptions(request_mock, context_mock)

        # assert that the discipline's initialize method was called with the expected options
        job.discipline.set_options.assert_called_once_with(
            {"key1": "value1", "key2": 42}
        )

    def test_setup(self):
        """
        Tests the Setup RPC of the Discipline Server.
        """
        request = Empty()

        server, job, context = make_server_from_instance(
            DisciplineServer, Mock()
        )

        # mock the 'setup' and 'setup_partials' methods of the discipline
        job.discipline.setup.return_value = None
        job.discipline.setup_partials.return_value = None

        server.Setup(request, context)

        # assert that the 'setup' and 'setup_partials' methods were called
        job.discipline.setup.assert_called_once()
        job.discipline.setup_partials.assert_called_once()

    def test_get_variable_definitions(self):
        """
        Tests the GetVariableDefinitions RPC of the Discipline Server.
        """
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )
        discipline = job.discipline

        # add an input and an output
        job.discipline.add_input("x", shape=(2, 2), units="m")
        job.discipline.add_output("f", shape=(1,), units="m**2")

        # mock arguments
        context = job_context(job=job)
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
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = job.discipline
        discipline.add_input("x", shape=(2, 2), units="m")
        discipline.add_input("y", shape=(3, 3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        # create simulated inputs for the function
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        server.preallocate_inputs(job, inputs, flat_inputs, outputs, flat_outputs)

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
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = job.discipline
        discipline.add_input("x", shape=(2, 2), units="m")
        discipline.add_input("y", shape=(3, 3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        # create simulated inputs for the function
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        server.preallocate_inputs(job, inputs, flat_inputs, outputs, flat_outputs)

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
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = job.discipline
        discipline.add_input("x", shape=(1,), units="m")
        discipline.add_input("y", shape=(3, 3), units="m**2")
        discipline.add_output("f1", shape=(1,), units="m**3")
        discipline.add_output("f2", shape=(2, 3), units="m**3")

        discipline.declare_partials("f1", "x")
        discipline.declare_partials("f1", "y")
        discipline.declare_partials("f2", "x")
        discipline.declare_partials("f2", "y")

        jac = server.preallocate_partials(job)

        pairs = [("f1", "x"), ("f1", "y"), ("f2", "x"), ("f2", "y")]
        expected_shapes = [(1,), (3, 3), (2, 3), (2, 3, 3, 3)]

        for pair, shape in zip(pairs, expected_shapes):
            self.assertTrue(pair in jac)
            self.assertIsInstance(jac[pair], np.ndarray)
            self.assertEqual(jac[pair].shape, shape)

    def test_preallocate_partials_implicit_uses_the_residual_shape(self):
        """
        For an implicit discipline the partial is taken of the residual, so
        the function shape must be resolved against the residual entry and
        not against the output that shares its name.
        """
        discipline = Discipline()
        discipline._is_implicit = True
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )
        discipline.add_input("x", shape=(3,), units="m")
        discipline.add_output("y", shape=(2,), units="m")

        # force the two entries apart so that the lookup is observable
        residual = discipline._var_meta[-1]
        self.assertEqual(residual.type, data.VariableType.kResidual)
        residual.shape[:] = []
        residual.shape.extend((4,))

        discipline.declare_partials("y", "x")
        discipline.declare_partials("y", "y")

        jac = server.preallocate_partials(job)

        # d(residual y)/dx uses the residual (4,) and the input (3,)
        self.assertEqual(jac[("y", "x")].shape, (4, 3))
        # d(residual y)/dy uses the residual (4,) and the output (2,)
        self.assertEqual(jac[("y", "y")].shape, (4, 2))

    def test_preallocate_partials_unknown_variable(self):
        """
        A partial declared against a variable that was never added reports a
        validation error rather than a bare KeyError.
        """
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )
        discipline.add_input("x", shape=(1,))
        discipline.add_output("f", shape=(1,))
        discipline.declare_partials("f", "missing")

        with self.assertRaises(PhiloteValidationError):
            server.preallocate_partials(job)

    def test_process_inputs(self):
        # create a mock request_iterator
        request_iterator = [
            data.VariableMessage(
                continuous=data.Array(
                    start=0,
                    end=2,
                    data=[1.0, 2.0, 3.0],
                    type=data.VariableType.kInput,
                    name="x",
                )
            ),
            data.VariableMessage(
                continuous=data.Array(
                    start=3, end=4, data=[4.0, 5.0], type=data.VariableType.kInput, name="x"
                )
            ),
            data.VariableMessage(
                continuous=data.Array(
                    start=0,
                    end=1,
                    data=[0.1, 0.2],
                    type=data.VariableType.kOutput,
                    name="f",
                )
            ),
        ]

        server, job, context = make_server(DisciplineServer, Discipline)

        # create mock flat_inputs and flat_outputs dictionaries
        flat_inputs = {"x": np.zeros(6)}
        flat_outputs = {"f": np.zeros(3)}

        server.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # check the results
        self.assertEqual(flat_inputs["x"].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
        self.assertEqual(flat_outputs["f"].tolist(), [0.1, 0.2, 0.0])

    def test_get_available_options_with_dict_type(self):
        """
        Tests that GetAvailableOptions correctly maps dict options to kStruct.
        """
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        request_mock = Mock()
        context_mock = context

        discipline.options_list = {
            "config": "dict",
            "flag": "bool",
        }

        results = server.GetAvailableOptions(request_mock, context_mock)

        expected_options = ["config", "flag"]
        expected_types = [data.kStruct, data.kBool]

        self.assertEqual(results.options, expected_options)
        self.assertEqual(results.type, expected_types)

    def test_set_options_with_nested_dict(self):
        """
        Tests that SetOptions correctly passes nested dict values through.
        """
        server, job, context = make_server(DisciplineServer, Discipline)

        request_mock = Mock()
        context_mock = job_context(job=job)

        request_mock.options = {
            "config": {"solver": "newton", "tol": 1e-6, "nested": {"a": 1}},
            "name": "test",
        }

        discipline_mock = Mock()
        job.discipline = discipline_mock

        server.SetOptions(request_mock, context_mock)

        job.discipline.set_options.assert_called_once_with(
            {
                "config": {"solver": "newton", "tol": 1e-6, "nested": {"a": 1}},
                "name": "test",
            }
        )

    def test_get_available_options_invalid_type_aborts(self):
        """
        Tests that GetAvailableOptions calls context.abort for invalid option
        types.
        """
        discipline = Discipline()
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        # Add option with invalid type (bypasses add_option validation by
        # writing directly to options_list)
        discipline.options_list["invalid_option"] = "unknown_type"

        request = Empty()
        context = job_context(job=job)

        server.GetAvailableOptions(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("Invalid value for discipline option 'invalid_option'", args[0][1])

    def test_process_inputs_empty_array_raises_error(self):
        """
        Tests that process_inputs raises PhiloteValidationError when array
        data is empty.
        """
        server, job, context = make_server(DisciplineServer, Discipline)

        # Create request with empty data array
        request_iterator = [
            data.VariableMessage(
                continuous=data.Array(
                    start=0,
                    end=2,
                    data=[],  # Empty data array
                    type=data.VariableType.kInput,
                    name="x",
                )
            ),
        ]

        flat_inputs = {"x": np.zeros(3)}
        flat_outputs = {}

        with self.assertRaises(PhiloteValidationError) as context:
            server.process_inputs(request_iterator, flat_inputs, flat_outputs)

        self.assertIn("Expected continuous variables but arrays were empty for variable x", str(context.exception))

    def test_get_available_options_general_exception_aborts(self):
        """
        Tests that GetAvailableOptions calls context.abort with INTERNAL
        for unexpected exceptions.
        """
        discipline = Mock()
        # options_list property raises an unexpected error
        type(discipline).options_list = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("unexpected"))
        )
        server, job, context = make_server_from_instance(
            DisciplineServer, discipline
        )

        request = Mock()
        context = job_context(job=job)

        server.GetAvailableOptions(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INTERNAL)
        self.assertIn("GetAvailableOptions failed", args[0][1])

    def test_set_options_validation_error_aborts(self):
        """
        Tests that SetOptions calls context.abort with INVALID_ARGUMENT
        for PhiloteValidationError.
        """
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = Mock()
        discipline.set_options.side_effect = PhiloteValidationError("bad option")
        job.discipline = discipline

        request = Mock()
        request.options = {}
        context = job_context(job=job)

        server.SetOptions(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("bad option", args[0][1])

    def test_set_options_general_exception_aborts(self):
        """
        Tests that SetOptions calls context.abort with INTERNAL for
        unexpected exceptions.
        """
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = Mock()
        discipline.set_options.side_effect = RuntimeError("boom")
        job.discipline = discipline

        request = Mock()
        request.options = {}
        context = job_context(job=job)

        server.SetOptions(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INTERNAL)
        self.assertIn("SetOptions failed", args[0][1])

    def test_setup_validation_error_aborts(self):
        """
        Tests that Setup calls context.abort with INVALID_ARGUMENT
        for PhiloteValidationError.
        """
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = Mock()
        discipline.setup.side_effect = PhiloteValidationError("bad setup")
        job.discipline = discipline

        request = Mock()
        context = job_context(job=job)

        server.Setup(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INVALID_ARGUMENT)
        self.assertIn("bad setup", args[0][1])

    def test_setup_general_exception_aborts(self):
        """
        Tests that Setup calls context.abort with INTERNAL for
        unexpected exceptions.
        """
        server, job, context = make_server(DisciplineServer, Discipline)
        discipline = Mock()
        discipline._clear_data.side_effect = RuntimeError("crash")
        job.discipline = discipline

        request = Mock()
        context = job_context(job=job)

        server.Setup(request, context)

        context.abort.assert_called_once()
        args = context.abort.call_args
        self.assertEqual(args[0][0], grpc.StatusCode.INTERNAL)
        self.assertIn("Setup failed", args[0][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
