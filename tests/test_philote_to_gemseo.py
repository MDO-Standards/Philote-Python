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
from philote_mdo.gemseo import PhiloteDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
