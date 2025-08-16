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
    Philote explicit discipline that wraps an OpenMDAO group.

    This class allows you to wrap existing OpenMDAO groups as Philote disciplines,
    enabling integration of OpenMDAO models into Philote workflows. The wrapper
    creates a bridge between OpenMDAO's group-based architecture and Philote's
    discipline-based framework.

    While the Philote discipline interface is explicit, the underlying OpenMDAO
    group may contain cycles that require nonlinear solvers.

    Key Features:
        - Wraps any OpenMDAO group as a Philote discipline
        - Maps variables between Philote and OpenMDAO namespaces
        - Supports automatic partial derivative computation using OpenMDAO's compute_totals
        - Handles units and multi-dimensional arrays
        - Preserves OpenMDAO solver capabilities for cyclic groups

    Typical Usage:
        >>> import openmdao.api as om
        >>> from philote_mdo.openmdao.group import OpenMdaoSubProblem
        >>>
        >>> # Create wrapper and add OpenMDAO group
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_group(my_openmdao_group)
        >>>
        >>> # Map variables between namespaces
        >>> subprob.add_mapped_input('x_local', 'x', shape=(1,), units='m')
        >>> subprob.add_mapped_output('y_local', 'y', shape=(1,), units='kg')
        >>>
        >>> # Declare partials for derivative computation
        >>> subprob.declare_subproblem_partial('y_local', 'x_local')
        >>>
        >>> # Setup and use
        >>> subprob.setup()
        >>> subprob.compute(inputs, outputs)

    Attributes:
        _prob (om.Problem): Internal OpenMDAO problem object
        _model (om.Group): Reference to the OpenMDAO model
        _input_map (dict): Mapping from local to OpenMDAO input variable names
        _output_map (dict): Mapping from local to OpenMDAO output variable names
        _partials_map (dict): Mapping for partial derivative computation
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

        This method creates a new OpenMDAO Problem with the provided group as the model.
        Any previously configured problem will be replaced.

        Parameters
        ----------
        group : om.Group
            OpenMDAO Group object to wrap as a Philote discipline

        Warning
        -------
        This will delete any previous problem settings and attached models.

        Examples
        --------
        >>> import openmdao.api as om
        >>> from philote_mdo.openmdao.group import OpenMdaoSubProblem
        >>>
        >>> class MyGroup(om.Group):
        ...     def setup(self):
        ...         self.add_subsystem('comp', om.ExecComp('y = 2*x'))
        >>>
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_group(MyGroup())
        """
        self._prob = om.Problem(model=group)
        self._model = self._prob.model

    def add_mapped_input(self, local_var, subprob_var, shape=(1,), units=""):
        """
        Maps an input variable from the Philote discipline to the OpenMDAO group.

        This creates a mapping between a variable name in the Philote discipline
        namespace and a variable name in the OpenMDAO group namespace.

        Parameters
        ----------
        local_var : str
            Variable name in the Philote discipline namespace
        subprob_var : str
            Variable name in the OpenMDAO group namespace
        shape : tuple, optional
            Shape of the variable, by default (1,)
        units : str, optional
            Units for the variable, by default ""

        Examples
        --------
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_mapped_input('x_local', 'x', shape=(1,), units='m')
        >>> subprob.add_mapped_input('design_vars', 'z', shape=(2,), units='')
        """
        self._input_map[local_var] = {
            "sub_prob_name": subprob_var,
            "shape": shape,
            "units": units,
        }

    def add_mapped_output(self, local_var, subprob_var, shape=(1,), units=""):
        """
        Maps an output variable from the Philote discipline to the OpenMDAO group.

        This creates a mapping between a variable name in the Philote discipline
        namespace and a variable name in the OpenMDAO group namespace.

        Parameters
        ----------
        local_var : str
            Variable name in the Philote discipline namespace
        subprob_var : str
            Variable name in the OpenMDAO group namespace
        shape : tuple, optional
            Shape of the variable, by default (1,)
        units : str, optional
            Units for the variable, by default ""

        Examples
        --------
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_mapped_output('objective', 'obj', shape=(1,), units='')
        >>> subprob.add_mapped_output('constraint1', 'con1', shape=(1,), units='N')
        """
        self._output_map[local_var] = {
            "sub_prob_name": subprob_var,
            "shape": shape,
            "units": units,
        }

    def clear_mapped_variables(self):
        """
        Clears all variable mappings and resets to empty dictionaries.

        This removes all input and output variable mappings that were previously
        configured using add_mapped_input() and add_mapped_output().

        Examples
        --------
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_mapped_input('x', 'input_x')
        >>> subprob.add_mapped_output('y', 'output_y')
        >>> subprob.clear_mapped_variables()  # All mappings removed
        """
        self._input_map = {}
        self._output_map = {}

    def declare_subproblem_partial(self, local_func, local_var):
        """
        Declares partial derivatives for the sub-problem.

        This method sets up the mapping for computing partial derivatives of
        outputs with respect to inputs using OpenMDAO's compute_totals method.
        The partials will be computed automatically during compute_partials().

        Parameters
        ----------
        local_func : str
            Output variable name in the Philote discipline namespace
        local_var : str
            Input variable name in the Philote discipline namespace

        Notes
        -----
        Both local_func and local_var must have been previously mapped using
        add_mapped_output() and add_mapped_input() respectively.

        Examples
        --------
        >>> subprob = OpenMdaoSubProblem()
        >>> subprob.add_mapped_input('x_local', 'x')
        >>> subprob.add_mapped_output('y_local', 'y')
        >>> subprob.declare_subproblem_partial('y_local', 'x_local')
        """
        self._partials_map[(local_func, local_var)] = (
            self._output_map[local_func]["sub_prob_name"],
            self._input_map[local_var]["sub_prob_name"],
        )

    def initialize(self):
        """
        Initialize the discipline.

        This method is called during discipline setup and can be overridden
        by subclasses to perform custom initialization, such as adding groups
        and configuring variable mappings.

        Examples
        --------
        >>> class MySubProblem(OpenMdaoSubProblem):
        ...     def initialize(self):
        ...         self.add_group(MyOpenMDAOGroup())
        ...         self.add_mapped_input('x', 'input_x')
        ...         self.add_mapped_output('y', 'output_y')
        """
        pass

    def setup(self):
        """
        Set up the discipline by configuring the OpenMDAO problem and Philote variables.

        This method sets up the internal OpenMDAO problem, adds all mapped inputs
        and outputs to the Philote discipline, and declares any specified partial
        derivatives.

        Notes
        -----
        This method is automatically called by the Philote framework and should
        not be called directly by users.
        """
        self._prob.setup()

        for local, var in self._input_map.items():
            self.add_input(local, shape=var["shape"], units=var["units"])

        for local, var in self._output_map.items():
            self.add_output(local, shape=var["shape"], units=var["units"])

        for pair in self._partials_map.keys():
            self.declare_partials(pair[0], pair[1])

    def compute(self, inputs, outputs):
        """
        Execute the OpenMDAO group and transfer results.

        This method transfers input values from the Philote discipline to the
        OpenMDAO group, runs the OpenMDAO model, and transfers the output
        values back to the Philote discipline.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with Philote variable names as keys
        outputs : dict
            Dictionary to store output values with Philote variable names as keys

        Notes
        -----
        This method is automatically called by the Philote framework during
        discipline execution and should not be called directly by users.
        """
        for local, var in self._input_map.items():
            sub = var["sub_prob_name"]
            self._prob[sub] = inputs[local]

        self._prob.run_model()

        for local, var in self._output_map.items():
            sub = var["sub_prob_name"]
            outputs[local] = self._prob[sub]

    def compute_partials(self, inputs, partials):
        """
        Compute partial derivatives using OpenMDAO's compute_totals method.

        This method transfers input values to the OpenMDAO group, runs the model,
        and computes total derivatives for all declared partial derivative pairs
        using OpenMDAO's automatic differentiation capabilities.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with Philote variable names as keys
        partials : dict
            Dictionary to store partial derivative values with (output, input)
            tuples as keys

        Notes
        -----
        This method is automatically called by the Philote framework when
        partial derivatives are needed and should not be called directly by users.
        Only partials that were declared using declare_subproblem_partial()
        will be computed.
        """
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
