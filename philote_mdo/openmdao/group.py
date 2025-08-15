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
import openmdao.api as om
import philote_mdo.general as pm


class OpenMdaoSubProblem(pm.ExplicitDiscipline):
    """
    Philote explicit discipline that calls an OpenMDAO group.

    While the Philote discipline is explicit, the underlying OpenMDAO
    group may have cycles that require a nonlinear solver.
    """

    def __init__(self):
        super().__init__()

        self._prob = None
        self._model = None

        self._input_map = {}
        self._output_map = {}
        self._partials_map = {}

    def add_group(self, group):
        """
        Adds an OpenMDAO group to the discipline.

        Warning: This will delete any previous problem settings and attached
        models.
        """
        self._prob = om.Problem(model=group)
        self._model = self._prob.model

    def add_mapped_input(self, local_var, subprob_var, shape=(1,), units=""):
        """
        Adds an input that is mapped from the discipline to the sub-problem.
        """
        self._input_map[local_var] = {
            "sub_prob_name": subprob_var,
            "shape": shape,
            "units": units,
        }

    def add_mapped_output(self, local_var, subprob_var, shape=(1,), units=""):
        """
        Adds an output that is mapped from the discipline to the sub-problem.
        """
        self._output_map[local_var] = {
            "sub_prob_name": subprob_var,
            "shape": shape,
            "units": units,
        }

    def clear_mapped_variables(self):
        """
        Clears the variable map and sets it to an empty dictionary.
        """
        self._input_map = {}
        self._output_map = {}

    def declare_subproblem_partial(self, local_func, local_var):
        """
        Declares the partials for this sub-problem.

        Parameters
        ----------
        local_func: str
            function name in the local name space
        local_var: str
            variable name in the local name space

        Returns
        -------
            None
        """
        self._partials_map[(local_func, local_var)] = (
            self._output_map[local_func]["sub_prob_name"],
            self._input_map[local_var]["sub_prob_name"],
        )

    def initialize(self):
        pass

    def setup(self):
        self._prob.setup()

        for local, var in self._input_map.items():
            self.add_input(local, shape=var["shape"], units=var["units"])

        for local, var in self._output_map.items():
            self.add_output(local, shape=var["shape"], units=var["units"])

        for pair in self._partials_map.keys():
            self.declare_partials(pair[0], pair[1])

    def compute(self, inputs, outputs):
        for local, var in self._input_map.items():
            sub = var["sub_prob_name"]
            self._prob[sub] = inputs[local]

        self._prob.run_model()

        for local, var in self._output_map.items():
            sub = var["sub_prob_name"]
            outputs[local] = self._prob[sub]

    def compute_partials(self, inputs, partials):
        for local, var in self._input_map.items():
            sub = var["sub_prob_name"]
            self._prob[sub] = inputs[local]

        self._prob.run_model()

        # get the list of functions and variables for the compute_totals call
        func = []
        var = []
        for val in self._partials_map.values():
            func += [val[0]]
            var += [val[1]]

        totals = self._prob.compute_totals(of=func, wrt=var)

        for local, sub in self._partials_map.items():
            partials[local] = totals[sub]
