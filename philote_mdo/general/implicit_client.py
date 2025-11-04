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
import grpc
from philote_mdo.general.discipline_client import DisciplineClient
import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc


class ImplicitClient(DisciplineClient):
    """
    Python client for connecting to and interacting with implicit Philote discipline servers.

    This class provides a Python interface for communicating with remote implicit disciplines
    over gRPC. It handles all the network communication, protocol buffer serialization,
    and array streaming automatically, allowing users to interact with remote implicit
    disciplines as if they were local Python objects.

    The client supports three main operations for implicit disciplines:
    - Residual computation: R(inputs, outputs) evaluation
    - Implicit solving: Find outputs such that R(inputs, outputs) = 0
    - Jacobian computation: Compute dR/d[inputs,outputs] for optimization

    Key Features:
        - Automatic gRPC connection management
        - Efficient streaming of large arrays
        - Variable metadata discovery from server
        - Option synchronization with server
        - Thread-safe operation
        - Automatic data type conversion

    Typical Usage:
        >>> import grpc
        >>> import philote_mdo.general as pmdo
        >>>
        >>> # Create gRPC channel to server
        >>> channel = grpc.insecure_channel('localhost:50051')
        >>> client = pmdo.ImplicitClient(channel)
        >>>
        >>> # Use the client for residual computation
        >>> inputs = {'a': [1.0], 'b': [-5.0], 'c': [6.0]}
        >>> outputs = {'x': [2.0]}  # Current guess
        >>> residuals = client.run_compute_residuals(inputs, outputs)
        >>> print(f"Residual: {residuals['x']}")  # Should be close to 0
        >>>
        >>> # Solve the implicit equations
        >>> solution = client.run_solve_residuals(inputs)
        >>> print(f"Solution: {solution['x']}")  # x = 2 or 3 for quadratic
        >>>
        >>> # Compute Jacobian for optimization
        >>> jacobian = client.run_residual_gradients(inputs, outputs)
        >>> print(f"dR/dx: {jacobian[('x', 'x')]}")  # Jacobian element

    Attributes:
        _impl_stub (ImplicitServiceStub): gRPC stub for implicit service calls

    Notes:
        - Inherits connection management and metadata handling from DisciplineClient
        - All methods are synchronous and block until server responds
        - Large arrays are automatically streamed for efficiency
        - The server must be running and reachable when creating the client
    """

    def __init__(self, channel):
        """
        Initialize the implicit discipline client.

        Creates a connection to the implicit discipline server and sets up the
        gRPC stub for making remote procedure calls.

        Parameters
        ----------
        channel : grpc.Channel
            gRPC channel for connecting to the implicit discipline server.
            Should be created with grpc.insecure_channel() or grpc.secure_channel().

        Examples
        --------
        >>> import grpc
        >>> channel = grpc.insecure_channel('localhost:50051')
        >>> client = ImplicitClient(channel)

        >>> # With secure channel
        >>> credentials = grpc.ssl_channel_credentials()
        >>> secure_channel = grpc.secure_channel('server:443', credentials)
        >>> client = ImplicitClient(secure_channel)

        Notes
        -----
        - The server must be running and reachable during initialization
        - Variable metadata and options are automatically discovered from server
        """
        super().__init__(channel=channel)
        self._impl_stub = disc.ImplicitServiceStub(channel)

    def run_compute_residuals(self, inputs, outputs):
        """
        Compute residuals R(inputs, outputs) by calling the remote server.

        This method sends both inputs and outputs to the server and receives the
        computed residual values. The residuals represent the constraint equations
        that should equal zero at the solution: R(inputs, outputs) = 0.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys and numpy arrays as values
        outputs : dict
            Dictionary of output values (current guess) with variable names as keys
            and numpy arrays as values

        Returns
        -------
        dict
            Dictionary of computed residual values with variable names as keys
            and numpy arrays as values

        Examples
        --------
        >>> # Quadratic equation: ax² + bx + c = 0
        >>> inputs = {'a': np.array([1.0]), 'b': np.array([-5.0]), 'c': np.array([6.0])}
        >>> outputs = {'x': np.array([2.0])}  # Guess x=2
        >>> residuals = client.run_compute_residuals(inputs, outputs)
        >>> print(residuals['x'])  # Should be [0.0] since 1*4 - 5*2 + 6 = 0

        >>> # Check a different guess
        >>> outputs = {'x': np.array([1.0])}  # Guess x=1  
        >>> residuals = client.run_compute_residuals(inputs, outputs)
        >>> print(residuals['x'])  # Should be [2.0] since 1*1 - 5*1 + 6 = 2

        >>> # Multi-dimensional case
        >>> inputs = {'params': np.array([1.0, 2.0, 3.0])}
        >>> outputs = {'states': np.array([0.5, 1.5])}
        >>> residuals = client.run_compute_residuals(inputs, outputs)

        Notes
        -----
        - Both inputs and outputs must be provided to evaluate residuals
        - Residual arrays have the same shape as their corresponding output arrays
        - Large arrays are automatically streamed for efficiency
        - This is typically used for residual evaluation during Newton iterations
        """
        # Assemble input messages and call server
        messages = self._assemble_input_messages(inputs, outputs)
        responses = self._impl_stub.ComputeResiduals(iter(messages))
        residuals = self._recover_residuals(responses)

        return residuals

    def run_solve_residuals(self, inputs):
        """
        Solve implicit equations R(inputs, outputs) = 0 by calling the remote server.

        This method sends inputs to the server and receives the solved output values
        that satisfy the residual equations. The server performs the nonlinear solving
        internally using its implemented solving strategy.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys and numpy arrays as values

        Returns
        -------
        dict
            Dictionary of solved output values with variable names as keys
            and numpy arrays as values

        Examples
        --------
        >>> # Solve quadratic equation: ax² + bx + c = 0
        >>> inputs = {'a': np.array([1.0]), 'b': np.array([-5.0]), 'c': np.array([6.0])}
        >>> solution = client.run_solve_residuals(inputs)
        >>> print(solution['x'])  # Should be [2.0] or [3.0] (roots of x²-5x+6=0)

        >>> # Solve system of equations
        >>> inputs = {
        ...     'param1': np.array([2.0]),
        ...     'param2': np.array([3.0])
        ... }
        >>> solution = client.run_solve_residuals(inputs)
        >>> print(solution['state1'], solution['state2'])

        >>> # Vector case
        >>> inputs = {'design': np.array([1.0, 2.0, 3.0])}
        >>> solution = client.run_solve_residuals(inputs)
        >>> print(solution['variables'])  # Multi-dimensional solution

        Raises
        ------
        grpc.RpcError
            If the server fails to solve the equations (non-convergence, etc.)
        
        Notes
        -----
        - Only inputs are required; initial guesses are handled by the server
        - The server determines convergence criteria and solving strategy
        - May raise exceptions for ill-conditioned or non-convergent problems
        - Solution quality depends on the server's implementation and input conditioning
        """
        # Assemble input messages and call server
        messages = self._assemble_input_messages(inputs)
        responses = self._impl_stub.SolveResiduals(iter(messages))
        outputs = self._recover_outputs(responses)
        return outputs

    def run_residual_gradients(self, inputs, outputs):
        """
        Compute Jacobian of residuals dR/d[inputs,outputs] by calling the remote server.

        This method sends both inputs and outputs to the server and receives the
        computed partial derivatives of residuals with respect to all inputs and outputs.
        These derivatives are essential for gradient-based optimization and sensitivity analysis.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values with variable names as keys and numpy arrays as values
        outputs : dict
            Dictionary of output values with variable names as keys and numpy arrays as values

        Returns
        -------
        dict
            Dictionary of partial derivatives with (residual_name, variable_name) tuples
            as keys and numpy arrays as values

        Examples
        --------
        >>> # Jacobian for quadratic equation R = ax² + bx + c
        >>> inputs = {'a': np.array([1.0]), 'b': np.array([-5.0]), 'c': np.array([6.0])}
        >>> outputs = {'x': np.array([2.0])}
        >>> jacobian = client.run_residual_gradients(inputs, outputs)
        >>>
        >>> # Access individual partial derivatives
        >>> print(jacobian[('x', 'a')])  # dR/da = x² = 4.0
        >>> print(jacobian[('x', 'b')])  # dR/db = x = 2.0
        >>> print(jacobian[('x', 'c')])  # dR/dc = 1.0
        >>> print(jacobian[('x', 'x')])  # dR/dx = 2ax + b = -1.0

        >>> # Multi-residual system
        >>> inputs = {'p1': np.array([1.0]), 'p2': np.array([2.0])}
        >>> outputs = {'x1': np.array([0.5]), 'x2': np.array([1.0])}
        >>> jacobian = client.run_residual_gradients(inputs, outputs)
        >>>
        >>> # Access cross-derivatives
        >>> print(jacobian[('x1', 'p1')])   # dR1/dp1
        >>> print(jacobian[('x1', 'x2')])   # dR1/dx2 (coupling)
        >>> print(jacobian[('x2', 'x1')])   # dR2/dx1 (coupling)

        >>> # Vector/matrix case
        >>> inputs = {'design': np.array([1.0, 2.0])}
        >>> outputs = {'states': np.array([0.5, 1.5, 2.5])}
        >>> jacobian = client.run_residual_gradients(inputs, outputs)
        >>> print(jacobian[('states', 'design')].shape)  # (3, 2) Jacobian block

        Notes
        -----
        - Both inputs and outputs must be provided for Jacobian evaluation
        - Returns both dR/dinputs and dR/doutputs partial derivatives
        - Jacobian elements are computed at the current (inputs, outputs) point
        - Used by optimization algorithms and sensitivity analysis tools
        - For large problems, consider matrix-free methods if available
        """
        # Assemble input messages and call server
        messages = self._assemble_input_messages(inputs, outputs)
        responses = self._impl_stub.ComputeResidualGradients(iter(messages))
        partials = self._recover_partials(responses)
        return partials
