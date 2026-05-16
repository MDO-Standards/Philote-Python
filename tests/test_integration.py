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
import unittest
import grpc
import numpy as np
import philote_mdo.general as pmdo
from philote_mdo.examples import Paraboloid, QuadradicImplicit


class IntegrationTests(unittest.TestCase):
    """
    Integration tests for the paraboloid discipline.
    """

    def test_paraboloid_compute(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)

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

    def test_paraboloid_compute_partials(self):
        """
        Integration test for the Paraboloid compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)

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
        jac = client.run_compute_partials(inputs)

        self.assertEqual(jac["f_xy", "x"][0], -2.0)
        self.assertEqual(jac["f_xy", "y"][0], 13.0)

        # stop the server
        server.stop(0)

    def test_quadratic_compute_residuals(self):
        """
        Integration test for the QuadraticImplicit compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ImplicitServer(discipline=QuadradicImplicit())
        discipline.attach_to_server(server)

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

        # run a function evaluation
        residuals = client.run_compute_residuals(inputs, outputs)

        self.assertEqual(residuals["x"][0], 22.0)

        # stop the server
        server.stop(0)

    def test_quadratic_solve_residuals(self):
        """
        Integration test for the QuadraticImplicit compute function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ImplicitServer(discipline=QuadradicImplicit())
        discipline.attach_to_server(server)

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

        # run a function evaluation
        outputs = client.run_solve_residuals(inputs)

        self.assertAlmostEqual(outputs["x"][0], 0.73205081, places=8)

        # stop the server
        server.stop(0)

    def test_quadratic_residual_gradients(self):
        """
        Integration test for the QuadraticImplicit residual gradients function.
        """
        # server code
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        discipline = pmdo.ImplicitServer(discipline=QuadradicImplicit())
        discipline.attach_to_server(server)

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

        # run a function evaluation
        jac = client.run_residual_gradients(inputs, outputs)

        self.assertEqual(jac[("x", "a")][0], 16.0)
        self.assertEqual(jac[("x", "b")][0], 4.0)
        self.assertEqual(jac[("x", "c")][0], 1.0)
        self.assertEqual(jac[("x", "x")][0], 10.0)

        # stop the server
        server.stop(0)


class StructOptionDiscipline(pmdo.ExplicitDiscipline):
    """
    Minimal discipline that declares a dict option and uses it in compute.
    """

    def initialize(self):
        self.add_option("config", "dict")

    def set_options(self, options):
        self.config = dict(options["config"])

    def setup(self):
        self.add_input("x", shape=(1,), units="")
        self.add_output("f", shape=(1,), units="")

    def setup_partials(self):
        self.declare_partials("f", "x")

    def compute(self, inputs, outputs):
        scale = self.config.get("scale", 1.0)
        offset = self.config.get("offset", 0.0)
        outputs["f"] = scale * inputs["x"] + offset

    def compute_partials(self, inputs, partials):
        scale = self.config.get("scale", 1.0)
        partials["f", "x"] = np.array([scale])


class StructOptionIntegrationTests(unittest.TestCase):
    """
    Integration tests for struct (dict) options round-trip over gRPC.
    """

    def test_struct_option_round_trip(self):
        """
        Tests that a discipline with a dict option can be discovered,
        set with a nested dict, and used for compute over gRPC.
        """
        # server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        discipline = pmdo.ExplicitServer(discipline=StructOptionDiscipline())
        discipline.attach_to_server(server)
        server.add_insecure_port("[::]:50051")
        server.start()

        try:
            # client
            client = pmdo.ExplicitClient(
                channel=grpc.insecure_channel("localhost:50051")
            )

            # discover options and verify dict type
            client.get_available_options()
            self.assertEqual(client.options_list["config"], "dict")

            # send nested dict option
            client.send_options({"config": {"scale": 3.0, "offset": 5.0}})

            # standard setup
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()
            client.get_partials_definitions()

            # compute: f = 3.0 * 2.0 + 5.0 = 11.0
            inputs = {"x": np.array([2.0])}
            outputs = client.run_compute(inputs)
            self.assertAlmostEqual(outputs["f"][0], 11.0)

            # partials: df/dx = 3.0
            jac = client.run_compute_partials(inputs)
            self.assertAlmostEqual(jac["f", "x"][0], 3.0)
        finally:
            server.stop(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
