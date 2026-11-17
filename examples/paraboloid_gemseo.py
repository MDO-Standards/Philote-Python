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
"""Optimize a remote Philote discipline with GEMSEO.

This example starts a gRPC server exposing the
``philote_mdo.examples.Paraboloid`` explicit discipline through the
Philote-MDO protocol, connects a ``philote_mdo.gemseo.PhiloteDiscipline``
client to it, and uses this discipline like any other GEMSEO discipline in
a mono-disciplinary optimization scenario: minimize ``f_xy(x, y)`` under
the constraint ``0 <= x + y <= 10``.

This requires GEMSEO to be installed (``pip install gemseo``).
"""

from __future__ import annotations

from concurrent import futures

import grpc
from gemseo import create_design_space
from gemseo import create_scenario
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.settings.formulations import DisciplinaryOpt_Settings
from gemseo.settings.opt import COBYQA_Settings
from numpy import ndarray

from philote_mdo.examples import Paraboloid
from philote_mdo.gemseo import PhiloteDiscipline
from philote_mdo.general import ExplicitServer

HOST = "localhost"
"""The host of the Philote discipline server."""

PORT = 50051
"""The port of the Philote discipline server."""


def create_discipline(server: grpc.Server) -> PhiloteDiscipline:
    """Serve the Paraboloid discipline and connect a GEMSEO client to it.

    Args:
        server: The started gRPC server to attach the Paraboloid Philote
            discipline to.

    Returns:
        A GEMSEO discipline connected to the remote Paraboloid discipline.
    """
    discipline = ExplicitServer(discipline=Paraboloid())
    discipline.attach_to_server(server)
    return PhiloteDiscipline(channel=grpc.insecure_channel(f"{HOST}:{PORT}"))


def create_server() -> grpc.Server:
    """Create and start the gRPC server used to serve Philote disciplines.

    Returns:
        The started gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    return server


def compute_constraint(x: ndarray, y: ndarray) -> ndarray:
    """Compute the constraint function ``g``.

    Args:
        x: The first paraboloid variable.
        y: The second paraboloid variable.

    Returns:
        The constraint value ``g``, bounded between 0 and 10 in the scenario.
    """
    # AutoPyDiscipline infers the output name from this assignment, so the
    # variable must be named "g" and returned as-is (not e.g. "return x + y").
    g = x + y
    return g


if __name__ == "__main__":
    server = create_server()

    try:
        paraboloid_disc = create_discipline(server)
        design_space = create_design_space()
        design_space.add_variable("x", lower_bound=-50, upper_bound=50, value=3.0)
        design_space.add_variable("y", lower_bound=-50, upper_bound=50, value=-4.0)

        scenario = create_scenario(
            [paraboloid_disc, AutoPyDiscipline(py_func=compute_constraint)],
            objective_name="f_xy",
            design_space=design_space,
            formulation_settings_model=DisciplinaryOpt_Settings(),
        )

        # 0 <= g <= 10
        scenario.add_constraint("g", constraint_type="ineq", positive=True)
        scenario.add_constraint("g", constraint_type="ineq", value=10.0)
        scenario.execute(algo_settings_model=COBYQA_Settings(max_iter=100))

        print(
            "Optimal design found:", design_space.get_current_value(as_dict=True)
        )
    finally:
        # Stop the server explicitly:
        # its worker threads would otherwise keep the process alive.
        server.stop(0)
