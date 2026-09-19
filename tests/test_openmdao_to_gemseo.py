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
from numpy.testing import assert_allclose
import philote_mdo.general as pmdo
import philote_mdo.openmdao as pmdo_om
from philote_mdo.examples.sellar import SellarMDA
from philote_mdo.gemseo import PhiloteDiscipline

PORT = "[::]:50051"
CHANNEL = "localhost:50051"


def _serve_sellar_mda(server):
    """
    Wraps the OpenMDAO SellarMDA group as a Philote explicit discipline and
    attaches it to the given (not yet started) gRPC server.
    """
    subprob = pmdo_om.OpenMdaoSubProblem()
    subprob.add_group(SellarMDA())
    subprob.add_mapped_input("x", "x", shape=(1,), units="")
    subprob.add_mapped_input("z", "z", shape=(2,), units="")
    subprob.add_mapped_output("obj", "obj", shape=(1,), units="")
    subprob.add_mapped_output("con1", "con1", shape=(1,), units="")
    subprob.declare_subproblem_partial("obj", "x")
    subprob.declare_subproblem_partial("obj", "z")
    subprob.declare_subproblem_partial("con1", "x")
    subprob.declare_subproblem_partial("con1", "z")

    discipline = pmdo.ExplicitServer(discipline=subprob)
    discipline.attach_to_server(server)


class OpenMDAOToGEMSEOTests(unittest.TestCase):
    """
    Integration tests for the Sellar MDA (an OpenMDAO group wrapped through
    OpenMdaoSubProblem) served to a GEMSEO PhiloteDiscipline client.
    """

    def test_sellar_mda_compute(self):
        """
        Integration test for the Sellar MDA in GEMSEO.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        _serve_sellar_mda(server)
        server.add_insecure_port(PORT)
        server.start()

        sellar_mda = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        # canonical Sellar starting point
        out = sellar_mda.execute({"x": array([1.0]), "z": array([5.0, 2.0])})

        server.stop(0)

        # canonical Sellar result at that point
        assert_allclose(out["obj"], 28.58830817, rtol=1e-8)
        assert_allclose(out["con1"], -22.42830237, rtol=1e-8)

    def test_sellar_mda_linearize(self):
        """
        Integration test for the Sellar MDA Jacobian in GEMSEO.
        """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        _serve_sellar_mda(server)
        server.add_insecure_port(PORT)
        server.start()

        sellar_mda = PhiloteDiscipline(channel=grpc.insecure_channel(CHANNEL))

        inputs = {"x": array([1.0]), "z": array([2.0, 3.0])}
        sellar_mda.linearize(inputs, compute_all_jacobians=True)
        ok = sellar_mda.check_jacobian(inputs, step=1e-7, threshold=1e-4)

        server.stop(0)

        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
