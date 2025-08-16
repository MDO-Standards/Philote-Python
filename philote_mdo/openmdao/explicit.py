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
import philote_mdo.openmdao.utils as utils


class RemoteExplicitComponent(om.ExplicitComponent):
    """
    OpenMDAO explicit component that acts as a client to a remote Philote analysis server.

    This component enables integration of Philote explicit disciplines into OpenMDAO workflows
    by connecting to a remote Philote server over gRPC. The component automatically discovers
    the server's input/output interface and handles data transfer, computation requests, and
    partial derivative computation.

    The component inherits from OpenMDAO's ExplicitComponent and implements the standard
    OpenMDAO component interface while internally communicating with a Philote server.

    Key Features:
        - Automatic discovery of server variables and options
        - Seamless integration with OpenMDAO's analysis and optimization capabilities
        - Support for partial derivatives via finite differencing or analytic computation
        - Handles unit conversions and data marshaling between OpenMDAO and Philote
        - Configurable through server-defined options

    Typical Usage:
        >>> import grpc
        >>> import openmdao.api as om
        >>> import philote_mdo.openmdao as pmom
        >>>
        >>> # Create gRPC channel to Philote server
        >>> channel = grpc.insecure_channel("localhost:50051")
        >>>
        >>> # Create OpenMDAO problem with remote component
        >>> prob = om.Problem()
        >>> prob.model.add_subsystem('analysis', 
        ...     pmom.RemoteExplicitComponent(channel=channel),
        ...     promotes=['*'])
        >>>
        >>> # Setup and run as normal OpenMDAO component
        >>> prob.setup()
        >>> prob['x'] = 2.0
        >>> prob.run_model()
        >>> print(prob['y'])

    Attributes:
        _client (pm.ExplicitClient): Internal Philote client for server communication

    Notes:
        - The server must be running and reachable during component initialization
        - Server options are automatically queried and declared as OpenMDAO options
        - Variable metadata (names, shapes, units) is automatically discovered from server
        - The component supports both function evaluation and partial derivative computation
    """

    def __init__(self, channel=None, num_par_fd=1, **kwargs):
        """
        Initialize the OpenMDAO component and establish connection to Philote server.

        This constructor creates a Philote explicit client, establishes the gRPC connection,
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
        >>> component = RemoteExplicitComponent(
        ...     channel=channel,
        ...     option1=True,
        ...     option2=42
        ... )

        Notes
        -----
        The server connection is established immediately during initialization.
        If the server is not available, the constructor will raise an exception.
        """
        if not channel:
            raise ValueError(
                "No channel provided, the Philote client will not be able to connect."
            )

        # generic Philote client
        # The setting of OpenMDAO options requires the list of available
        # Philote discipline options to be known during initialize. That
        # means that the server must be reachable to query the
        # available options on this discipline.
        self._client = pm.ExplicitClient(channel=channel)

        # Initialize the parent OpenMDAO ExplicitComponent
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
        declares the corresponding inputs and outputs in the OpenMDAO component. Variable
        properties including names, shapes, and units are preserved from the server.

        The method is called automatically by OpenMDAO during problem setup and should
        not be called directly by users.

        Notes
        -----
        - Input and output variables are automatically discovered from server
        - Variable shapes and units are preserved from server metadata
        - This method uses utility functions to handle the variable declaration process
        """
        utils.client_setup(self)

    def setup_partials(self):
        """
        Set up partial derivatives for the OpenMDAO component.

        This method declares partial derivatives for all output-input pairs based on
        the server's partial derivative metadata. The component can compute partials
        either analytically (if supported by the server) or via finite differencing.

        The method is called automatically by OpenMDAO during component setup and should
        not be called directly by users.

        Notes
        -----
        - Partial derivative structure is automatically discovered from server
        - Both analytic and finite difference methods are supported
        - Sparsity patterns are preserved when available from server metadata
        """
        utils.client_setup_partials(self)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute function evaluation by calling the remote Philote server.

        This method transfers input values to the server, requests a function evaluation,
        and transfers the computed outputs back to the OpenMDAO component. Data marshaling
        and unit conversions are handled automatically.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        outputs : dict  
            Dictionary to store computed output values with variable names as keys
        discrete_inputs : dict, optional
            Dictionary of discrete input values (currently unused), by default None
        discrete_outputs : dict, optional
            Dictionary of discrete output values (currently unused), by default None

        Notes
        -----
        - Input/output data is automatically converted between OpenMDAO and Philote formats
        - The method handles variable name mapping and unit conversions
        - Server communication is synchronous - the method blocks until computation completes
        - This method is called automatically by OpenMDAO during model execution
        """
        local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
        out = self._client.run_compute(local_inputs)
        utils.assign_global_outputs(out, outputs)

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        """
        Compute partial derivatives by calling the remote Philote server.

        This method transfers input values to the server, requests partial derivative
        computation, and transfers the computed Jacobian values back to the OpenMDAO
        component. The server may compute derivatives analytically or via finite differencing.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys
        partials : dict
            Dictionary to store computed partial derivatives with (output, input) 
            tuples as keys
        discrete_inputs : dict, optional
            Dictionary of discrete input values (currently unused), by default None
        discrete_outputs : dict, optional
            Dictionary of discrete output values (currently unused), by default None

        Notes
        -----
        - Partial derivatives are computed at the current input point
        - The method handles conversion between OpenMDAO and Philote Jacobian formats
        - Server determines whether to use analytic or finite difference derivatives
        - This method is called automatically by OpenMDAO when derivatives are needed
        - Sparsity patterns from the server are preserved in the OpenMDAO component
        """
        local_inputs = utils.create_local_inputs(inputs, self._client._var_meta)
        jac = self._client.run_compute_partials(local_inputs)
        utils.assign_global_outputs(jac, partials)
