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
"""Trim a remote OpenAeroStruct aerostructural discipline with GEMSEO.

This example starts a gRPC server exposing ``philote_examples``'
``OasAerostructDiscipline`` -- a coupled OpenAeroStruct aerostructural
analysis of a single wing (VLM aerodynamics + FEM structure, run through
OpenMDAO) wrapped as a Philote-MDO explicit discipline -- connects a
:class:`~philote_mdo.gemseo.philote_to_gemseo.PhiloteDiscipline` client to
it, and uses this discipline in a GEMSEO
:class:`~gemseo.scenarios.mdo_scenario.MDOScenario`: find the angle of
attack ``alpha`` that minimizes the drag coefficient ``CD`` while trimming
the wing at a target lift coefficient ``CL = 0.5``, using the
gradient-based SLSQP algorithm.

The remote discipline only exposes the partial derivatives of ``CL`` and
``CD`` with respect to ``alpha`` (see the two
``declare_subproblem_partial`` calls in
``OasAerostructDiscipline._build_discipline``), so ``alpha`` is the only
usable design variable, and ``CL``/``CD`` the only usable objective and
constraint, for a gradient-based scenario. The other flight-condition and
mission parameters are kept fixed at their default values.

This requires GEMSEO, OpenMDAO, OpenAeroStruct and ``philote-examples``
0.5.1 or later, the first version whose ``OasAerostructDiscipline``
declares these partial derivatives, to be installed
(``pip install gemseo openaerostruct "philote-examples>=0.5.1"``).
"""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import philote_mdo.general as pmdo
from gemseo import create_design_space
from gemseo import create_scenario
from gemseo import execute_post
from gemseo.settings.formulations import DisciplinaryOpt_Settings
from gemseo.settings.opt import SLSQP_Settings
from gemseo.settings.post import OptHistoryView_Settings
from numpy import array
from openaerostruct.utils.constants import grav_constant
from philote_examples import OasAerostructDiscipline

from philote_mdo.gemseo import PhiloteDiscipline

HOST = "localhost"
"""The host of the Philote discipline server."""

PORT = 50052
"""The port of the Philote discipline server."""

FIXED_INPUTS = {
    "v": array([248.136]),
    "Mach_number": array([0.84]),
    "re": array([1e6]),
    "rho": array([0.38]),
    "CT": array([grav_constant * 17.0e-6]),
    "R": array([11.165e6]),
    "W0": array([0.4 * 3e5]),
    "speed_of_sound": array([295.4]),
    "load_factor": array([1.0]),
    "empty_cg": np.zeros(3),
}
"""The flight-condition and mission inputs that are not design variables.

These are set as default input values of the discipline, so that the
scenario only has to provide a value for the "alpha" design variable.
"""


def create_discipline(server: grpc.Server) -> PhiloteDiscipline:
    """Serve the OAS aerostructural discipline and connect a GEMSEO client to it.

    Args:
        server: The started gRPC server to attach the OAS aerostructural
            Philote discipline to.

    Returns:
        A GEMSEO discipline connected to the remote OAS discipline, with
        its non-design inputs (flight conditions and mission parameters)
        set to their default values.
    """
    discipline = pmdo.ExplicitServer(discipline=OasAerostructDiscipline())
    discipline.attach_to_server(server)
    oas_discipline = PhiloteDiscipline(channel=grpc.insecure_channel(f"{HOST}:{PORT}"))
    oas_discipline.default_input_data.update(FIXED_INPUTS)
    return oas_discipline


def create_server() -> grpc.Server:
    """Create and start the gRPC server used to serve Philote disciplines.

    Returns:
        The started gRPC server.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    return server


if __name__ == "__main__":
    server = create_server()

    try:
        oas_disc = create_discipline(server)

        design_space = create_design_space()
        # alpha is in degrees, as declared by OasAerostructDiscipline.
        design_space.add_variable(
            "alpha", lower_bound=-10.0, upper_bound=15.0, value=5.0
        )

        scenario = create_scenario(
            [oas_disc],
            objective_name="CD",
            design_space=design_space,
            formulation_settings_model=DisciplinaryOpt_Settings(),
        )
        # Trim the wing: CL = 0.5
        scenario.add_constraint("CL", constraint_type="eq", value=0.5)
        scenario.execute(algo_settings_model=SLSQP_Settings(max_iter=20))

        out = oas_disc.local_data
        print(
            "Trimmed solution:",
            {name: out[name] for name in ("alpha", "CL", "CD", "failure")},
        )
        execute_post(scenario, OptHistoryView_Settings(save=True, show=False))

    finally:
        # Stop the server explicitly:
        # its worker threads would otherwise keep the process alive.
        server.stop(0)
