# Philote-Python
#
# Copyright 2026 IRT Saint Exupery
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
# originally authored by Francois Gallard, IRT Saint Exupery.
from concurrent import futures
import unittest
import grpc
from numpy import array
import philote_mdo.general as pmdo
from philote_mdo.examples import Paraboloid
from philote_mdo.examples import Rosenbrock
from philote_mdo.gemseo import PhiloteDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


class DynamicShapeDiscipline(pmdo.ExplicitDiscipline):
    """A discipline whose input and output shapes are set by the client."""

    def setup(self):
        self.add_input("x", dynamic_shape=True)
        self.add_output("y", dynamic_shape=True)

    def compute(self, inputs, outputs):
        outputs["y"] = 2.0 * inputs["x"]


class DiscreteInputDiscipline(pmdo.ExplicitDiscipline):
    """A discipline with a discrete input next to a continuous one."""

    def setup(self):
        self.add_input("x", shape=(1,))
        self.add_discrete_input("mode", default="a")
        self.add_output("y", shape=(1,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["y"] = 2.0 * inputs["x"]


class PhiloteToGEMSEOTests(unittest.TestCase):
    """
    Integration tests for PhiloteDiscipline, which wraps a remote Philote
    explicit discipline server as a GEMSEO discipline.
    """

    def test_paraboloid_compute(self):
        """
        Integration test for the Paraboloid compute function.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        paraboloid_disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        out = paraboloid_disc.execute({"x": array([1.0]), "y": array([2.0])})

        server.stop(0)

        self.assertEqual(out["f_xy"][0], 39.0)

    def test_paraboloid_linearize(self):
        """
        Integration test for the Paraboloid linearization, checked against
        the analytic Jacobian at (x, y) = (1, 2).
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Paraboloid())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        paraboloid_disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        inputs = {"x": array([1.0]), "y": array([2.0])}
        paraboloid_disc.linearize(inputs, compute_all_jacobians=True)

        server.stop(0)

        self.assertEqual(paraboloid_disc.jac["f_xy"]["x"][0][0], -2.0)
        self.assertEqual(paraboloid_disc.jac["f_xy"]["y"][0][0], 13.0)

    def test_missing_channel_raises_value_error(self):
        """
        Constructing a PhiloteDiscipline without a channel raises ValueError.
        """
        with self.assertRaises(ValueError):
            PhiloteDiscipline(channel=None)

        with self.assertRaises(ValueError):
            PhiloteDiscipline(channel="")

    def test_rosenbrock_with_options(self):
        """
        Discipline options passed to the constructor are sent to the server
        and used to build a Rosenbrock discipline of the requested dimension.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        discipline = pmdo.ExplicitServer(discipline=Rosenbrock())
        discipline.attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()

        rosenbrock_disc = PhiloteDiscipline(
            channel=grpc.insecure_channel(CHANNEL), dimension=3
        )

        out = rosenbrock_disc.execute({"x": array([1.0, 1.0, 1.0])})

        server.stop(0)

        self.assertEqual(out["f"][0], 0.0)

    def _serve(self, discipline):
        """
        Serves a Philote discipline for the duration of the test.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pmdo.ExplicitServer(discipline=discipline).attach_to_server(server)
        server.add_insecure_port(PORT)
        server.start()
        self.addCleanup(server.stop, 0)

    def test_name_defaults_to_server_reported_name(self):
        """
        Without an explicit name, the discipline takes the name reported by
        the server.
        """
        paraboloid = Paraboloid()
        paraboloid._name = "paraboloid"
        self._serve(paraboloid)

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertEqual(disc.name, "paraboloid")

    def test_name_falls_back_to_class_name(self):
        """
        When the server reports no name, GEMSEO's class-name default is used.
        """
        self._serve(Paraboloid())

        disc = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertEqual(disc.name, "PhiloteDiscipline")

    def test_explicit_name_overrides_server_name(self):
        """
        A name passed to the constructor takes precedence over the server's.
        """
        paraboloid = Paraboloid()
        paraboloid._name = "paraboloid"
        self._serve(paraboloid)

        disc = PhiloteDiscipline(
            channel=grpc.insecure_channel(CHANNEL), name="remote_paraboloid"
        )

        self.assertEqual(disc.name, "remote_paraboloid")

    def test_dynamic_shape_variables_raise(self):
        """
        Dynamic-shape server variables are rejected at construction, naming
        the offending variables.
        """
        self._serve(DynamicShapeDiscipline())

        with self.assertRaises(NotImplementedError) as ctx:
            PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertIn("x (dynamic shape)", str(ctx.exception))
        self.assertIn("y (dynamic shape)", str(ctx.exception))

    def test_discrete_variables_raise(self):
        """
        Discrete server variables are rejected at construction rather than
        silently left at their server-side defaults.
        """
        self._serve(DiscreteInputDiscipline())

        with self.assertRaises(NotImplementedError) as ctx:
            PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        self.assertIn("mode (discrete)", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
