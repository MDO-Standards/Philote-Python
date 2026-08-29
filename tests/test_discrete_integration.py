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
"""
Integration tests for disciplines with discrete variables.

These tests spin up a real gRPC server and client and exercise the full
discrete variable round-trip: declaration, metadata discovery, message
serialization, discipline evaluation, and result recovery.
"""
from concurrent import futures
import unittest

import grpc
import numpy as np
import openmdao.api as om

import philote_mdo.general as pmdo
import philote_mdo.openmdao as pmdo_om


# ---------------------------------------------------------------------------
#  Example discipline with discrete inputs/outputs
# ---------------------------------------------------------------------------
class ScaledParaboloid(pmdo.ExplicitDiscipline):
    """
    Paraboloid whose output is scaled by a discrete mode flag.

    Continuous inputs:  x, y  (scalars)
    Discrete input:     mode  ("double" or "half")
    Continuous output:  f_xy  (scalar)
    Discrete output:    label (string describing the mode used)

    f_xy = scale * ((x - 3)^2 + x*y + (y + 4)^2 - 3)

    where scale = 2.0 for "double" and 0.5 for "half".
    """

    def setup(self):
        self.add_input("x", shape=(1,), units="m")
        self.add_input("y", shape=(1,), units="m")
        self.add_output("f_xy", shape=(1,), units="m**2")

        self.add_discrete_input("mode")
        self.add_discrete_output("label")

    def setup_partials(self):
        self.declare_partials("f_xy", "x")
        self.declare_partials("f_xy", "y")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs["x"]
        y = inputs["y"]
        base = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0

        mode = (discrete_inputs or {}).get("mode", None) or "double"
        scale = 2.0 if mode == "double" else 0.5

        outputs["f_xy"] = scale * base

        if discrete_outputs is not None:
            discrete_outputs["label"] = f"scaled_{mode}"

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x = inputs["x"]
        y = inputs["y"]

        mode = (discrete_inputs or {}).get("mode", None) or "double"
        scale = 2.0 if mode == "double" else 0.5

        partials["f_xy", "x"] = scale * (2.0 * x - 6.0 + y)
        partials["f_xy", "y"] = scale * (2.0 * y + 8.0 + x)


# ---------------------------------------------------------------------------
#  Integration tests
# ---------------------------------------------------------------------------
class TestDiscreteIntegration(unittest.TestCase):
    """
    End-to-end integration tests for disciplines with discrete variables.
    """

    def _start_server(self, discipline, port):
        """Helper to start a gRPC server with the given discipline."""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
        explicit_server = pmdo.ExplicitServer(discipline_factory=lambda: discipline)
        explicit_server.attach_to_server(server)
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        return server

    # ------------------------------------------------------------------
    #  Raw client tests (no OpenMDAO)
    # ------------------------------------------------------------------
    def test_client_compute_with_discrete_inputs(self):
        """
        Test that a raw ExplicitClient can send discrete inputs and
        receive both continuous and discrete outputs.
        """
        port = 50061
        server = self._start_server(ScaledParaboloid(), port)

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            client = pmdo.ExplicitClient(channel)

            # setup handshake
            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()
            client.get_partials_definitions()

            # verify discrete metadata was discovered
            self.assertTrue(len(client._discrete_var_meta) > 0)
            discrete_names = {v.name for v in client._discrete_var_meta}
            self.assertIn("mode", discrete_names)
            self.assertIn("label", discrete_names)

            # run compute with mode="double"
            inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
            result = client.run_compute(inputs, discrete_inputs={"mode": "double"})

            # Should return (outputs, discrete_outputs) tuple
            self.assertIsInstance(result, tuple)
            outputs, discrete_outputs = result

            # base paraboloid value at (1, 2) = (1-3)^2 + 1*2 + (2+4)^2 - 3 = 39
            # scaled by 2.0 => 78.0
            np.testing.assert_almost_equal(outputs["f_xy"][0], 78.0)
            self.assertEqual(discrete_outputs["label"], "scaled_double")

            # run compute with mode="half"
            result = client.run_compute(inputs, discrete_inputs={"mode": "half"})
            outputs, discrete_outputs = result

            # 39 * 0.5 = 19.5
            np.testing.assert_almost_equal(outputs["f_xy"][0], 19.5)
            self.assertEqual(discrete_outputs["label"], "scaled_half")

        finally:
            server.stop(0)

    def test_client_compute_partials_with_discrete_inputs(self):
        """
        Test that a raw ExplicitClient can send discrete inputs for
        gradient evaluation.
        """
        port = 50062
        server = self._start_server(ScaledParaboloid(), port)

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            client = pmdo.ExplicitClient(channel)

            client.send_stream_options()
            client.run_setup()
            client.get_variable_definitions()
            client.get_partials_definitions()

            inputs = {"x": np.array([1.0]), "y": np.array([2.0])}

            # mode="double", scale=2.0
            # df/dx = 2*(2*1 - 6 + 2) = 2*(-2) = -4.0
            # df/dy = 2*(2*2 + 8 + 1) = 2*(13) = 26.0
            partials = client.run_compute_partials(
                inputs, discrete_inputs={"mode": "double"}
            )

            np.testing.assert_almost_equal(partials[("f_xy", "x")][0], -4.0)
            np.testing.assert_almost_equal(partials[("f_xy", "y")][0], 26.0)

            # mode="half", scale=0.5
            # df/dx = 0.5*(-2) = -1.0
            # df/dy = 0.5*(13) = 6.5
            partials = client.run_compute_partials(
                inputs, discrete_inputs={"mode": "half"}
            )

            np.testing.assert_almost_equal(partials[("f_xy", "x")][0], -1.0)
            np.testing.assert_almost_equal(partials[("f_xy", "y")][0], 6.5)

        finally:
            server.stop(0)

    # ------------------------------------------------------------------
    #  OpenMDAO integration test
    # ------------------------------------------------------------------
    def test_openmdao_compute_with_discrete(self):
        """
        Test the full OpenMDAO integration: RemoteExplicitComponent
        auto-discovers discrete variables from the server and returns
        correct results.
        """
        port = 50063
        server = self._start_server(ScaledParaboloid(), port)

        try:
            channel = grpc.insecure_channel(f"localhost:{port}")

            prob = om.Problem()
            comp = pmdo_om.RemoteExplicitComponent(channel=channel)
            prob.model.add_subsystem("scaled", comp)

            prob.setup()

            # verify discrete variables were discovered
            discrete_names = {v.name for v in comp._client._discrete_var_meta}
            self.assertIn("mode", discrete_names)
            self.assertIn("label", discrete_names)

            # set continuous inputs
            prob.set_val("scaled.x", 1.0)
            prob.set_val("scaled.y", 2.0)

            # run model – mode defaults to "double" when not set (scale=2.0)
            prob.run_model()

            # base value at (1,2) = 39.0, scale = 2.0 => 78.0
            np.testing.assert_almost_equal(
                prob.get_val("scaled.f_xy")[0], 78.0
            )

        finally:
            server.stop(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
