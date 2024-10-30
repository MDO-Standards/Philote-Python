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
from unittest.mock import Mock, patch

import numpy as np

from philote_mdo.general import ImplicitClient
import philote_mdo.generated.data_pb2 as data
import philote_mdo.utils as utils


class TestImplicitClient(unittest.TestCase):
    """
    Unit tests for the implicit client.
    """

    @patch("philote_mdo.generated.disciplines_pb2_grpc.ImplicitServiceStub")
    def test_compute_residuals(self, mock_implicit_stub):
        """
        Tests the compute function of the Implicit Client.
        """
        mock_channel = Mock()
        mock_stub = mock_implicit_stub.return_value
        client = ImplicitClient(mock_channel)
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(3,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]

        input_data = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }

        output_data = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([7.0, 6.0, 5.0]),
        }

        response1 = data.Array(
            name="f", type=data.kResidual, start=0, end=2, data=[5.0, 6.0, 7.0]
        )
        response2 = data.Array(
            name="g", type=data.kResidual, start=0, end=2, data=[8.0, 9.0, 10.0]
        )
        mock_responses = [response1, response2]

        mock_stub.ComputeResiduals.return_value = mock_responses

        residuals = client.run_compute_residuals(input_data, output_data)

        # checks
        self.assertTrue(mock_stub.ComputeResiduals.called)

        expected_residuals = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([8.0, 9.0, 10.0]),
        }
        for output_name, expected_data in expected_residuals.items():
            self.assertTrue(output_name in residuals)
            np.testing.assert_array_equal(residuals[output_name], expected_data)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.ImplicitServiceStub")
    def test_solve_residuals(self, mock_implicit_stub):
        """
        Tests the compute function of the Implicit Client.
        """
        mock_channel = Mock()
        mock_stub = mock_implicit_stub.return_value
        client = ImplicitClient(mock_channel)
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(3,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2, 2)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]

        input_data = {
            "x": np.array([1.0, 2.0, 3.0, 4.0]).reshape(2, 2),
        }

        output_data = {
            "f": np.array([5.0, 6.0, 7.0]),
        }

        response1 = data.Array(
            name="f", type=data.kOutput, start=0, end=2, data=[5.0, 6.0, 7.0]
        )
        response2 = data.Array(
            name="g", type=data.kOutput, start=0, end=2, data=[8.0, 9.0, 10.0]
        )
        mock_responses = [response1, response2]

        mock_stub.SolveResiduals.return_value = mock_responses

        outputs = client.run_solve_residuals(input_data)

        # checks
        self.assertTrue(mock_stub.SolveResiduals.called)

        expected_outputs = {
            "f": np.array([5.0, 6.0, 7.0]),
            "g": np.array([8.0, 9.0, 10.0]),
        }
        for output_name, expected_data in expected_outputs.items():
            self.assertTrue(output_name in outputs)
            np.testing.assert_array_equal(outputs[output_name], expected_data)

    @patch("philote_mdo.generated.disciplines_pb2_grpc.ImplicitServiceStub")
    def test_residual_partials(self, mock_implicit_stub):
        """
        Tests the residual_partials function of the Implicit Client.
        """
        mock_channel = Mock()
        mock_stub = mock_implicit_stub.return_value
        client = ImplicitClient(mock_channel)
        client._var_meta = [
            data.VariableMetaData(name="f", type=data.kOutput, shape=(2,)),
            data.VariableMetaData(name="f", type=data.kResidual, shape=(2,)),
            data.VariableMetaData(name="x", type=data.kInput, shape=(2,)),
            data.VariableMetaData(name="g", type=data.kOutput, shape=(3,)),
            data.VariableMetaData(name="g", type=data.kResidual, shape=(3,)),
        ]
        client._partials_meta = [data.PartialsMetaData(name="f", subname="x")]

        input_data = {
            "x": np.array([1.0, 2.0]),
        }

        output_data = {
            "f": np.array([5.0, 6.0]),
        }

        response1 = data.Array(
            name="f",
            subname="x",
            type=data.kPartial,
            start=0,
            end=2,
            data=[5.0, 6.0, 7.0],
        )
        response2 = data.Array(
            name="f", subname="x", type=data.kPartial, start=3, end=3, data=[4.0]
        )
        mock_responses = [response1, response2]

        mock_stub.ComputeResidualGradients.return_value = mock_responses

        partials = client.run_residual_gradients(input_data, output_data)

        # checks
        self.assertTrue(mock_stub.ComputeResidualGradients.called)

        expected_partials = utils.PairDict()
        expected_partials[("f", "x")] = np.array([5.0, 6.0, 7.0, 4.0]).reshape((2, 2))

        for key, expected_data in expected_partials.items():
            self.assertTrue(key in partials)
            np.testing.assert_array_equal(partials[key], expected_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
