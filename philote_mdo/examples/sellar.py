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
import numpy as np
import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2
import philote_mdo.openmdao as pm


class SellarMDA(om.Group):
    """
    This is the Sellar MDO problem as implemented in the OpenMDAO codebase,
    licensed under the Apache 2 open source license:

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this software except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    It was included here so that the solver debug print statements could
    be turned off.
    """

    def setup(self):
        cycle = self.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem(
            "d1",
            SellarDis1(),
            promotes_inputs=["x", "z", "y2"],
            promotes_outputs=["y1"],
        )
        cycle.add_subsystem(
            "d2", SellarDis2(), promotes_inputs=["z", "y1"], promotes_outputs=["y2"]
        )

        cycle.set_input_defaults("x", 1.0)
        cycle.set_input_defaults("z", np.array([5.0, 2.0]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS(iprint=0)
        cycle.linear_solver = om.LinearBlockGS(iprint=0)

        self.add_subsystem(
            "obj_cmp",
            om.ExecComp(
                "obj = x**2 + z[1] + y1 + exp(-y2)", z=np.array([0.0, 0.0]), x=0.0
            ),
            promotes=["x", "z", "y1", "y2", "obj"],
        )

        self.add_subsystem(
            "con_cmp1", om.ExecComp("con1 = 3.16 - y1"), promotes=["con1", "y1"]
        )
        self.add_subsystem(
            "con_cmp2", om.ExecComp("con2 = y2 - 24.0"), promotes=["con2", "y2"]
        )


class SellarGroup(pm.OpenMdaoSubProblem):

    def initialize(self):
        self.add_group(SellarMDA())

    def set_options(self, options):
        pass

