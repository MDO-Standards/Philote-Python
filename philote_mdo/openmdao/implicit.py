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
import philote_mdo.generated.data_pb2 as data
import philote_mdo.openmdao.utils as utils


class RemoteImplicitComponent(om.ImplicitComponent):
    """
    OpenMDAO implicit component that acts as a client to a remote Philote implicit analysis server.

    This component enables integration of Philote implicit disciplines into OpenMDAO workflows
    by connecting to a remote Philote server over gRPC. Unlike explicit components, implicit
    components solve residual equations of the form R(inputs, outputs) = 0, where the outputs
    are implicitly defined by the inputs.

    The component inherits from OpenMDAO's ImplicitComponent and implements the standard
    OpenMDAO implicit component interface while internally communicating with a Philote server.
    It supports both residual evaluation and solving, as well as partial derivative computation
    for gradient-based optimization.

    Key Features:
        - Automatic discovery of server variables and options
        - Residual evaluation via apply_nonlinear method
        - Implicit equation solving via solve_nonlinear method  
        - Jacobian computation for optimization via linearize method
        - Seamless integration with OpenMDAO's implicit solving capabilities
        - Support for coupled systems and nonlinear solvers

    Typical Usage:
        >>> import grpc
        >>> import openmdao.api as om
        >>> import philote_mdo.openmdao as pmom
        >>>
        >>> # Create gRPC channel to Philote server
        >>> channel = grpc.insecure_channel("localhost:50051")
        >>>
        >>> # Create OpenMDAO problem with remote implicit component
        >>> prob = om.Problem()
        >>> prob.model.add_subsystem('implicit_analysis', 
        ...     pmom.RemoteImplicitComponent(channel=channel),
        ...     promotes=['*'])
        >>>
        >>> # Setup and run with nonlinear solver if needed
        >>> prob.setup()
        >>> prob['a'] = 1.0
        >>> prob['b'] = -2.0
        >>> prob['c'] = 1.0
        >>> prob.run_model()
        >>> print(prob['x'])  # Solution to ax^2 + bx + c = 0

    Attributes:
        _client (pm.ImplicitClient): Internal Philote client for server communication

    Notes:
        - The server must be running and reachable during component initialization
        - Server options are automatically queried and declared as OpenMDAO options
        - Variable metadata (names, shapes, units) is automatically discovered from server
        - The component supports residual evaluation, solving, and linearization
        - Both inputs and outputs must be provided to the server for residual computation
    """

    def __init__(self, channel=None, num_par_fd=1, **kwargs):
        """
        Initialize the OpenMDAO implicit component and establish connection to Philote server.

        This constructor creates a Philote implicit client, establishes the gRPC connection,
        and configures the component with any provided options. The server must be reachable
        during initialization to query available options and variable metadata.

        Parameters
        ----------
        channel : grpc.Channel
            gRPC channel for connecting to the Philote server. This is required and
            should be created using grpc.insecure_channel() or grpc.secure_channel().
        num_par_fd : int, optional
            Number of parallel finite difference processes for partial derivatives,
            by default 1
        **kwargs : dict
            Additional keyword arguments passed as options to the Philote server.
            Available options are server-specific and queried automatically.

        Raises
        ------
        ValueError
            If no channel is provided or channel is None
        grpc.RpcError
            If unable to connect to the Philote server

        Examples
        --------
        >>> import grpc
        >>> channel = grpc.insecure_channel("localhost:50051")
        >>> component = RemoteImplicitComponent(
        ...     channel=channel,
        ...     tolerance=1e-8,
        ...     max_iterations=100
        ... )

        Notes
        -----
        The server connection is established immediately during initialization.
        If the server is not available, the constructor will raise an exception.
        """
        if not channel:
            raise ValueError(
                "No channel provided, the Philote client will not"
                "be able to connect."
            )

        # generic Philote client
        # The setting of OpenMDAO options requires the list of available
        # Philote discipline options to be known during initialize. That
        # means that the server must be reachable to query the
        # available options on this discipline.
        self._client = pm.ImplicitClient(channel=channel)

        # Initialize the parent OpenMDAO ImplicitComponent
        super().__init__(num_par_fd=num_par_fd, **kwargs)

        # Send initial options to the server
        # This must be done here and not in initialize, as the values of the
        # OpenMDAO options are only set after initialize has been called in the
        # parent init function. That is why the parent init function must be called
        # before sending the option values to the Philote server.
        self._client.send_options(kwargs)

    def initialize(self):
        """
        Define OpenMDAO component options based on server-available options.

        This method queries the Philote server for its available options and automatically
        declares them as OpenMDAO component options with appropriate types. This allows
        the server's configuration to be controlled through OpenMDAO's option system.

        The method is called automatically by OpenMDAO during component setup and should
        not be called directly by users.

        Notes
        -----
        - Server options are automatically discovered via gRPC
        - Option types (bool, int, float, str) are preserved from server metadata
        - This method requires an active connection to the Philote server
        """
        # get the available options from the philote discipline
        self._client.get_available_options()

        # add to the OpenMDAO component options
        utils.declare_options(self._client.options_list.items(), self.options)

    def setup(self):
        """
        Set up the OpenMDAO component by declaring inputs and outputs from server metadata.

        This method queries the Philote server for variable metadata and automatically
        declares the corresponding inputs and outputs in the OpenMDAO component. For
        implicit components, both inputs and outputs are declared, with outputs representing
        the unknowns to be solved for.

        The method is called automatically by OpenMDAO during problem setup and should
        not be called directly by users.

        Notes
        -----
        - Input and output variables are automatically discovered from server
        - Variable shapes and units are preserved from server metadata
        - This method uses utility functions to handle the variable declaration process
        - Outputs represent the unknowns in the implicit equations
        """
        utils.client_setup(self)

    def setup_partials(self):
        """
        Set up partial derivatives for the OpenMDAO implicit component.

        This method declares partial derivatives for all residual-input and residual-output
        pairs based on the server's partial derivative metadata. For implicit components,
        this includes both dR/dinputs and dR/doutputs terms needed for Newton-type solvers.

        The method is called automatically by OpenMDAO during component setup and should
        not be called directly by users.

        Notes
        -----
        - Partial derivative structure is automatically discovered from server
        - Both dR/dinputs and dR/doutputs partials are declared
        - Sparsity patterns are preserved when available from server metadata
        """
        utils.client_setup_partials(self)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        """
        Compute residual evaluation by calling the remote Philote server.

        This method transfers both input and output values to the server, requests a
        residual evaluation, and transfers the computed residuals back to the OpenMDAO
        component. The residuals represent R(inputs, outputs) where the goal is to
        find outputs such that R = 0.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict
            Dictionary of output values (current guess) with variable names as keys
        residuals : dict
            Dictionary to store computed residual values with variable names as keys
        discrete_inputs : dict, optional
            Dictionary of discrete input values (currently unused), by default None
        discrete_outputs : dict, optional
            Dictionary of discrete output values (currently unused), by default None

        Notes
        -----
        - Both inputs and outputs are sent to the server for residual computation
        - Residuals are computed as R(inputs, outputs) at the current point
        - The goal is to find outputs where residuals are zero
        - This method is called automatically by OpenMDAO during residual evaluation
        """
        local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
        local_outputs = utils.create_local_inputs(
            outputs, self._client._var_meta, data.kOutput
        )

        res = self._client.run_compute_residuals(local_inputs, local_outputs)
        utils.assign_global_outputs(res, residuals)

    def solve_nonlinear(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Solve the implicit equations by calling the remote Philote server.

        This method transfers input values to the server, requests the server to solve
        the implicit equations R(inputs, outputs) = 0 for the outputs, and transfers
        the solved output values back to the OpenMDAO component.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict
            Dictionary to store solved output values with variable names as keys
        discrete_inputs : dict, optional
            Dictionary of discrete input values (currently unused), by default None
        discrete_outputs : dict, optional
            Dictionary of discrete output values (currently unused), by default None

        Notes
        -----
        - The server performs the nonlinear solve internally
        - Server may use Newton's method, fixed-point iteration, or other solvers
        - Convergence criteria are controlled by server options
        - This method is called by OpenMDAO's nonlinear solvers
        - Output values are updated with the converged solution
        """
        local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
        out = self._client.run_solve_residuals(local_inputs)
        utils.assign_global_outputs(out, outputs)

    def linearize(self, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None):
        """
        Compute partial derivatives of residuals by calling the remote Philote server.

        This method transfers both input and output values to the server, requests
        computation of the residual Jacobian (dR/dinputs and dR/doutputs), and transfers
        the computed partial derivatives back to the OpenMDAO component. These derivatives
        are used by OpenMDAO's linear solvers and optimization algorithms.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict
            Dictionary of output values with variable names as keys
        partials : dict
            Dictionary to store computed partial derivatives with (residual, variable)
            tuples as keys
        discrete_inputs : dict, optional
            Dictionary of discrete input values (currently unused), by default None
        discrete_outputs : dict, optional
            Dictionary of discrete output values (currently unused), by default None

        Notes
        -----
        - Computes both dR/dinputs and dR/doutputs partial derivatives
        - Derivatives are computed at the current (inputs, outputs) point
        - Server determines whether to use analytic or finite difference derivatives
        - This method is called automatically by OpenMDAO when derivatives are needed
        - Results are used by linear solvers and optimization algorithms
        """
        local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
        local_outputs = utils.create_local_inputs(
            outputs, self._client._var_meta, data.kOutput
        )
        jac = self._client.run_residual_gradients(local_inputs, local_outputs)
        utils.assign_global_outputs(jac, partials)
