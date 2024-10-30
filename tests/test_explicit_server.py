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
from scipy.optimize import rosen, rosen_der

from google.protobuf.empty_pb2 import Empty

from philote_mdo.general import ExplicitDiscipline, ExplicitServer
import philote_mdo.generated.data_pb2 as data


class TestExplicitServer(unittest.TestCase):
    """
    Unit tests for the discipline server.
    """

    def test_compute_function(self):
        """
        Tests the ComputeFunction RPC of the Explicit Server.
        """
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
        server._stream_opts.num_double = 3
        discipline.add_input("x", shape=(5,), units="")
        discipline.add_output("f", shape=(2,), units="")

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
                start=3, end=4, data=[4.5, 5.5], type=data.VariableType.kInput, name="x"
            ),
        ]

        # mock function call
        def compute(inputs, outputs):
            outputs["f"] = np.array([rosen(inputs["x"]), rosen(inputs["x"].T - 2.0)])

        server._discipline.compute = compute

        # call the function
        response_generator = server.ComputeFunction(request_iterator, context)
        responses = list(response_generator)

        # check that there is only one response
        self.assertEqual(len(responses), 1)

        # check the function value
        response = responses[0]
        self.assertEqual(response.name, "f")
        self.assertEqual(response.start, 0)
        self.assertEqual(response.end, 1)
        self.assertEqual(response.data[0], 28094.0)
        self.assertEqual(response.data[1], 1686.0)

    def test_compute_gradient(self):
        """
        Tests the ComputeGradient RPC of the Explicit Server.
        """
        server = ExplicitServer()
        discipline = server._discipline = ExplicitDiscipline()
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
                start=3, end=4, data=[4.5, 5.5], type=data.VariableType.kInput, name="x"
            ),
        ]

        # mock function call
        def compute_partials(inputs, jac):
            jac["f", "x"] = rosen_der(inputs["x"])

        server._discipline.compute_partials = compute_partials

        # call the function
        response_generator = server.ComputeGradient(request_iterator, context)
        responses = list(response_generator)

        # check that there is only one response
        self.assertEqual(len(responses), 2)

        # check the function value
        response = responses[0]
        self.assertEqual(response.name, "f")
        self.assertEqual(response.subname, "x")
        self.assertEqual(response.start, 0)
        self.assertEqual(response.end, 2)
        grad = np.array(response.data)

        response = responses[1]
        grad = np.append(grad, np.array(response.data))
        self.assertTrue(
            np.array_equal(grad, np.array([-251.0, -499.0, 11105.0, 25007.0, -2950.0]))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
