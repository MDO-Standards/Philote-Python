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
from concurrent import futures
import os
import unittest
import grpc
import numpy as np
import philote_mdo.general as pmdo

try:
    from philote_mdo.wrappers.julia import JuliaWrapperDiscipline, JuliaImplicitWrapperDiscipline
    HAS_JULIACALL = True
except ImportError:
    HAS_JULIACALL = False


@unittest.skipIf(not HAS_JULIACALL, "juliacall not installed")
class JuliaIntegrationTests(unittest.TestCase):
    """
    Integration tests for Julia discipline wrappers via gRPC.
    """

    @classmethod
    def setUpClass(cls):
        """Set up paths to example Julia files."""
        # Get the project root directory
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tests_dir)
        examples_dir = os.path.join(project_root, "examples", "julia")

        cls.paraboloid_file = os.path.join(examples_dir, "paraboloid.jl")
        cls.quadratic_file = os.path.join(examples_dir, "quadratic.jl")

    def test_julia_paraboloid_compute(self):
        """
        Integration test for Julia Paraboloid compute function via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        explicit_server = pmdo.ExplicitServer(discipline=discipline)
        explicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}

        # run a function evaluation
        outputs = client.run_compute(inputs)

        self.assertEqual(outputs["f_xy"][0], 39.0)

        # stop the server
        server.stop(0)

    def test_julia_paraboloid_compute_partials(self):
        """
        Integration test for Julia Paraboloid compute_partials function via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        explicit_server = pmdo.ExplicitServer(discipline=discipline)
        explicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}

        # run compute_partials
        jac = client.run_compute_partials(inputs)

        self.assertEqual(jac["f_xy", "x"][0], -2.0)
        self.assertEqual(jac["f_xy", "y"][0], 13.0)

        # stop the server
        server.stop(0)

    def test_julia_paraboloid_with_options(self):
        """
        Integration test for Julia Paraboloid with custom options via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline",
            options={"scale_factor": 2.0, "offset": 10.0}
        )
        explicit_server = pmdo.ExplicitServer(discipline=discipline)
        explicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ExplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}

        # run a function evaluation
        outputs = client.run_compute(inputs)

        # With scale_factor=2.0 and offset=10.0: f = 2.0 * 39.0 + 10.0 = 88.0
        self.assertEqual(outputs["f_xy"][0], 88.0)

        # run compute_partials
        jac = client.run_compute_partials(inputs)

        # Partials are also scaled
        self.assertEqual(jac["f_xy", "x"][0], -4.0)
        self.assertEqual(jac["f_xy", "y"][0], 26.0)

        # stop the server
        server.stop(0)

    def test_julia_quadratic_compute_residuals(self):
        """
        Integration test for Julia Quadratic compute_residuals function via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        implicit_server = pmdo.ImplicitServer(discipline=discipline)
        implicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ImplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}

        # run compute_residuals
        residuals = client.run_compute_residuals(inputs, outputs)

        self.assertEqual(residuals["x"][0], 22.0)

        # stop the server
        server.stop(0)

    def test_julia_quadratic_solve_residuals(self):
        """
        Integration test for Julia Quadratic solve_residuals function via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        implicit_server = pmdo.ImplicitServer(discipline=discipline)
        implicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ImplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}

        # run solve_residuals
        outputs = client.run_solve_residuals(inputs)

        self.assertAlmostEqual(outputs["x"][0], 0.73205081, places=8)

        # stop the server
        server.stop(0)

    def test_julia_quadratic_residual_gradients(self):
        """
        Integration test for Julia Quadratic residual_gradients function via gRPC.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        implicit_server = pmdo.ImplicitServer(discipline=discipline)
        implicit_server.attach_to_server(server)

        server.add_insecure_port("[::]:50051")
        server.start()

        # client code
        client = pmdo.ImplicitClient(channel=grpc.insecure_channel("localhost:50051"))

        # transfer the stream options to the server
        client.send_stream_options()

        # run setup
        client.run_setup()
        client.get_variable_definitions()
        client.get_partials_definitions()

        # define some inputs
        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}

        # run residual_gradients
        jac = client.run_residual_gradients(inputs, outputs)

        self.assertEqual(jac[("x", "a")][0], 16.0)
        self.assertEqual(jac[("x", "b")][0], 4.0)
        self.assertEqual(jac[("x", "c")][0], 1.0)
        self.assertEqual(jac[("x", "x")][0], 10.0)

        # stop the server
        server.stop(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
