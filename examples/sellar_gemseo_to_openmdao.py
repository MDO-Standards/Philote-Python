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
# originally authored by Francois Gallard, IRT Saint Exupery
"""Optimize the GEMSEO Sellar problem from OpenMDAO through Philote-MDO.

This example serves each discipline of GEMSEO's customizable Sellar MDO
problem (``Sellar1``, ``Sellar2``, ``SellarSystem``) as an independent
Philote-MDO gRPC server, using ``philote_mdo.gemseo.GEMSEOtoPhiloteDiscipline``,
and builds an OpenMDAO model that calls them through
``philote_mdo.openmdao.RemoteExplicitComponent``. The algebraic coupling
between ``Sellar1`` and ``Sellar2`` (through ``y_1``, ``y_2``) is resolved
with a Gauss-Seidel solver, and OpenMDAO's SLSQP driver optimizes the
coupled system using the analytic gradients computed by the GEMSEO
disciplines and transferred over gRPC.

This demonstrates the "OpenMDAO calls a remote GEMSEO discipline" direction
of the ``philote_mdo.gemseo`` integration, complementing ``PhiloteDiscipline``
(used for the opposite direction: GEMSEO calling a remote discipline, e.g.
served from OpenMDAO -- see ``examples/paraboloid_gemseo.py``).

The optimal solution of this classic problem is well known:
``x_1 = 0``, ``x_shared = [1.9776, 0]``, ``obj = 3.1834``.

This requires GEMSEO to be installed (``pip install gemseo``).
"""

from __future__ import annotations

from concurrent import futures
from typing import TYPE_CHECKING

import grpc
import openmdao.api as om
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem

from philote_mdo.gemseo import GEMSEOtoPhiloteDiscipline
from philote_mdo.general import ExplicitServer
from philote_mdo.openmdao import RemoteExplicitComponent

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline

HOST = "localhost"
"""The host of the Philote discipline servers."""

PORT_D1 = 50061
"""The port serving the Sellar1 discipline (computes y_1)."""

PORT_D2 = 50062
"""The port serving the Sellar2 discipline (computes y_2)."""

PORT_SYSTEM = 50063
"""The port serving the SellarSystem discipline (computes obj, c_1, c_2)."""


def create_discipline_server(discipline: Discipline, port: int) -> grpc.Server:
    """Serve a GEMSEO discipline as a Philote-MDO explicit discipline.

    Args:
        discipline: The GEMSEO discipline to serve.
        port: The port to serve it on.

    Returns:
        The started gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    philote_discipline = GEMSEOtoPhiloteDiscipline(discipline)
    ExplicitServer(discipline=philote_discipline).attach_to_server(server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server


def create_remote_component(port: int) -> RemoteExplicitComponent:
    """Create an OpenMDAO component connected to a remote Philote discipline.

    Args:
        port: The port of the Philote discipline server to connect to.

    Returns:
        An OpenMDAO explicit component calling the remote discipline.
    """
    return RemoteExplicitComponent(channel=grpc.insecure_channel(f"{HOST}:{port}"))


if __name__ == "__main__":
    servers = [
        create_discipline_server(Sellar1(), PORT_D1),
        create_discipline_server(Sellar2(), PORT_D2),
        create_discipline_server(SellarSystem(), PORT_SYSTEM),
    ]

    try:
        prob = om.Problem()
        model = prob.model

        # Sellar1 and Sellar2 are coupled through y_1 and y_2: group them so
        # that this algebraic loop can be resolved by a dedicated solver.
        cycle = model.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem("d1", create_remote_component(PORT_D1), promotes=["*"])
        cycle.add_subsystem("d2", create_remote_component(PORT_D2), promotes=["*"])
        cycle.nonlinear_solver = om.NonlinearBlockGS()
        # Compute the exact total derivatives across the coupling loop.
        cycle.linear_solver = om.DirectSolver()

        model.add_subsystem(
            "system", create_remote_component(PORT_SYSTEM), promotes=["*"]
        )

        model.add_design_var("x_1", lower=0.0, upper=10.0)
        model.add_design_var("x_2", lower=0.0, upper=10.0)
        model.add_design_var("x_shared", lower=[-10.0, 0.0], upper=[10.0, 10.0])
        model.add_objective("obj")
        model.add_constraint("c_1", upper=0.0)
        model.add_constraint("c_2", upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-8)

        prob.setup()

        # The Philote-MDO protocol only transfers variable names, shapes and
        # units, not GEMSEO's default values, so the non-design parameters
        # and the design variables' starting point must be set explicitly
        # here, matching gemseo's SellarDesignSpace and get_initial_data.
        prob.set_val("x_1", 1.0)
        prob.set_val("x_2", 1.0)
        prob.set_val("x_shared", [4.0, 3.0])
        prob.set_val("alpha", 3.16)
        prob.set_val("beta", 24.0)
        prob.set_val("gamma", 0.2)

        prob.run_driver()

        print(
            "Optimal solution:",
            {
                name: prob.get_val(name)
                for name in ("x_1", "x_2", "x_shared", "obj", "c_1", "c_2")
            },
        )
    finally:
        # Stop the servers explicitly:
        # their worker threads would otherwise keep the process alive.
        for server in servers:
            server.stop(0)
