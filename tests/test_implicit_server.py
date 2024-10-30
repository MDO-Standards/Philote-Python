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
import numpy.testing as np_testing
from google.protobuf.empty_pb2 import Empty
from philote_mdo.general import ImplicitDiscipline, ImplicitServer
import philote_mdo.generated.data_pb2 as data


class TestImplicitServer(unittest.TestCase):
    """
    Unit tests for the discipline server.
    """

    def test_compute_residuals(self):
        """
        Tests the ComputeResiduals RPC of the Implicit Server.

        This test does not mock out the utility function calls within the
        compute_residuals function, so that the entire solution process is
        tested (the actual residual function is mocked).
        """
        server = ImplicitServer()
        server._stream_opts.num_double = 1
        discipline = server._discipline = ImplicitDiscipline()
        discipline.add_input("x", shape=(2,), units="m")
        discipline.add_input("y", shape=(2,), units="m")
        discipline.add_output("f", shape=(2,), units="m")

        # mock request iterator
        mock_request_iterator = [
            data.Array(name="x", start=0, end=2, type=data.kInput, data=[1.0, 2.0]),
            data.Array(name="y", start=0, end=2, type=data.kInput, data=[3.0, 4.0]),
            data.Array(name="f", start=0, end=2, type=data.kOutput, data=[5.0, 6.0]),
        ]

        # mock inputs, outputs, and residuals
        mock_inputs = {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
        mock_flat_inputs = {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
        mock_outputs = {"f": np.array([5.0, 6.0])}
        mock_flat_outputs = {"f": np.array([5.0, 6.0])}

        # mock function call
        def compute_residuals(inputs, outputs, residuals):
            residuals["f"] = np.array([7.0, 8.0])

        server._discipline.compute_residuals = compute_residuals

        # call the ComputeResiduals method
        response_generator = server.ComputeResiduals(mock_request_iterator, None)
        result = list(response_generator)

        # assert that the expected residual messages were yielded
        expected_result = [
            data.Array(
                name="f",
                start=0,
                end=1,
                type=data.VariableType.kResidual,
                data=[7.0],
            ),
            data.Array(
                name="f",
                start=1,
                end=2,
                type=data.VariableType.kResidual,
                data=[8.0],
            ),
        ]
        self.assertEqual(result, expected_result)

    def test_solve_residuals(self):
        """
        Tests the SolveResiduals RPC of the Implicit Server.

        This test does not mock out the utility function calls within the
        solve_residuals function, so that the entire solution process is tested
        (the actual residual function is mocked).
        """
        server = ImplicitServer()
        server._stream_opts.num_double = 1
        discipline = server._discipline = ImplicitDiscipline()
        discipline.add_input("x", shape=(2,), units="m")
        discipline.add_input("y", shape=(2,), units="m")
        discipline.add_output("f", shape=(2,), units="m")

        # mock request iterator
        mock_request_iterator = [
            data.Array(name="x", start=0, end=2, type=data.kInput, data=[1.0, 2.0]),
            data.Array(name="y", start=0, end=2, type=data.kInput, data=[3.0, 4.0]),
            data.Array(name="f", start=0, end=2, type=data.kOutput, data=[5.0, 6.0]),
        ]

        # mock inputs, outputs, and residuals
        mock_inputs = {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
        mock_flat_inputs = {"x": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
        mock_outputs = {"f": np.array([5.0, 6.0])}
        mock_flat_outputs = {"f": np.array([5.0, 6.0])}

        # mock function call
        def solve_residuals(inputs, outputs):
            outputs["f"] = np.array([7.0, 8.0])

        server._discipline.solve_residuals = solve_residuals

        # call the ComputeResiduals method
        response_generator = server.SolveResiduals(mock_request_iterator, None)
        result = list(response_generator)

        # assert that the expected residual messages were yielded
        expected_result = [
            data.Array(
                name="f",
                start=0,
                end=1,
                type=data.VariableType.kOutput,
                data=[7.0],
            ),
            data.Array(
                name="f",
                start=1,
                end=2,
                type=data.VariableType.kOutput,
                data=[8.0],
            ),
        ]
        self.assertEqual(result, expected_result)

    def test_residual_gradients(self):
        """
        Tests the ComputeResiduals RPC of the Implicit Server.
        """
        server = ImplicitServer()
        discipline = server._discipline = ImplicitDiscipline()
        server._stream_opts.num_double = 3
        discipline.add_input("x", shape=(5,), units="")
        discipline.add_output("f", shape=(1,), units="")
        discipline.declare_partials("f", "x")

        context = Mock()
        request_iterator = [
            data.Array(
                start=0,
                end=2,
                data=[0.5, 1.5, 3.5],
                type=data.VariableType.kInput,
                name="x",
            ),
            data.Array(
                start=3,
                end=4,
                data=[4.5, 5.5],
                type=data.VariableType.kInput,
                name="x",
            ),
        ]

        # mock function call
        def residual_partials(inputs, residuals, jac):
            jac["f", "x"] = np.array([-251.0, -499.0, 11105.0, 25007.0, -2950.0])

        server._discipline.residual_partials = residual_partials

        # call the function
        response_generator = server.ComputeResidualGradients(request_iterator, context)
        responses = list(response_generator)

        # check that there is only one response
        self.assertEqual(len(responses), 2)

        # check the function value
        response = responses[0]
        self.assertEqual(response.name, "f")
        self.assertEqual(response.subname, "x")
        self.assertEqual(response.start, 0)
        self.assertEqual(response.end, 3)
        grad = np.array(response.data)

        response = responses[1]
        grad = np.append(grad, np.array(response.data))
        self.assertTrue(
            np.array_equal(grad, np.array([-251.0, -499.0, 11105.0, 25007.0, -2950.0]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
