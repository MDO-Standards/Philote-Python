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
import openmdao.api as om
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
import philote_mdo.general as pmdo
import philote_mdo.openmdao as pmdo_om
from philote_mdo.gemseo import GEMSEOtoPhiloteDiscipline

PORT_D1 = "[::]:50061"
CHANNEL_D1 = "localhost:50061"
PORT_D2 = "[::]:50062"
CHANNEL_D2 = "localhost:50062"
PORT_SYSTEM = "[::]:50063"
CHANNEL_SYSTEM = "localhost:50063"


def _serve(discipline, port):
    """
    Serves a GEMSEO discipline as a Philote explicit discipline.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    philote_discipline = GEMSEOtoPhiloteDiscipline(discipline)
    pmdo.ExplicitServer(discipline=philote_discipline).attach_to_server(server)
    server.add_insecure_port(port)
    server.start()
    return server


class SellarGEMSEOToOpenMDAOTests(unittest.TestCase):
    """
    Integration tests for the GEMSEO Sellar MDO problem (Sellar1, Sellar2,
    SellarSystem), served as three Philote explicit discipline servers and
    driven from an OpenMDAO model through RemoteExplicitComponent.

    The y_1/y_2 algebraic coupling between Sellar1 and Sellar2 is resolved
    by a Gauss-Seidel solver, and both the total derivatives and the
    gradient-based (SLSQP) optimum of the coupled model are checked.
    """

    def setUp(self):
        self._servers = [
            _serve(Sellar1(), PORT_D1),
            _serve(Sellar2(), PORT_D2),
            _serve(SellarSystem(), PORT_SYSTEM),
        ]

        prob = om.Problem()
        model = prob.model

        # Sellar1 and Sellar2 are coupled through y_1 and y_2: group them so
        # that this algebraic loop can be resolved by a dedicated solver.
        cycle = model.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem(
            "d1",
            pmdo_om.RemoteExplicitComponent(
                channel=grpc.insecure_channel(CHANNEL_D1)
            ),
            promotes=["*"],
        )
        cycle.add_subsystem(
            "d2",
            pmdo_om.RemoteExplicitComponent(
                channel=grpc.insecure_channel(CHANNEL_D2)
            ),
            promotes=["*"],
        )
        cycle.nonlinear_solver = om.NonlinearBlockGS(iprint=0)
        # Compute the exact total derivatives across the coupling loop.
        cycle.linear_solver = om.DirectSolver()

        model.add_subsystem(
            "system",
            pmdo_om.RemoteExplicitComponent(
                channel=grpc.insecure_channel(CHANNEL_SYSTEM)
            ),
            promotes=["*"],
        )

        model.add_design_var("x_1", lower=0.0, upper=10.0)
        model.add_design_var("x_2", lower=0.0, upper=10.0)
        model.add_design_var("x_shared", lower=[-10.0, 0.0], upper=[10.0, 10.0])
        model.add_objective("obj")
        model.add_constraint("c_1", upper=0.0)
        model.add_constraint("c_2", upper=0.0)

        prob.setup()

        # The Philote-MDO protocol only transfers variable names, shapes and
        # units, not GEMSEO's default values, so they must be set explicitly.
        prob.set_val("x_1", 1.0)
        prob.set_val("x_2", 1.0)
        prob.set_val("x_shared", [4.0, 3.0])
        prob.set_val("alpha", 3.16)
        prob.set_val("beta", 24.0)
        prob.set_val("gamma", 0.2)

        self._prob = prob

    def tearDown(self):
        for server in self._servers:
            server.stop(0)

    def test_sellar_derivatives(self):
        """
        The total derivatives through the remote GEMSEO disciplines are
        correct, checked against finite differences, across the y_1/y_2
        coupling loop and the three remote disciplines.
        """
        self._prob.run_model()

        check_data = self._prob.check_totals(
            of=["obj", "c_1", "c_2"],
            wrt=["x_1", "x_2", "x_shared"],
            out_stream=None,
        )

        for (of, wrt), error in check_data.items():
            rel_error = error["rel error"]
            value = rel_error.forward
            if value is None:
                value = rel_error.reverse
            self.assertLess(value, 1e-4, "d{}/d{}: relative error {} too large.".format(of, wrt, value))

    def test_sellar_optimum(self):
        """
        SLSQP finds the well-known optimum of the Sellar problem:
        obj = 3.18339, x_1 = 0, x_shared = [1.9776, 0], with c_1 active.

        Reference values from Sellar, R., Batill, S., and Renaud, J.
        (1996), "Response surface based, concurrent subspace optimization
        for multidisciplinary system design".
        """
        self._prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-8)

        self._prob.run_driver()

        self.assertAlmostEqual(self._prob.get_val("obj")[0], 3.18339, places=4)
        self.assertAlmostEqual(self._prob.get_val("x_1")[0], 0.0, places=3)
        self.assertAlmostEqual(self._prob.get_val("x_shared")[0], 1.9776, places=3)
        self.assertAlmostEqual(self._prob.get_val("x_shared")[1], 0.0, places=3)
        # The constraints shall be satisfied, c_1 being active at the optimum.
        self.assertAlmostEqual(self._prob.get_val("c_1")[0], 0.0, places=4)
        self.assertLess(self._prob.get_val("c_2")[0], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
