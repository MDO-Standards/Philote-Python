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
import philote_mdo.generated.disciplines_pb2_grpc as disc
import philote_mdo.generated.data_pb2 as data
import philote_mdo.general as pmdo
from philote_mdo.utils import get_chunk_indices


class ImplicitServer(pmdo.DisciplineServer, disc.ImplicitServiceServicer):
    """
    gRPC server implementation for serving implicit Philote disciplines.

    This class creates a gRPC server that exposes implicit discipline functionality
    over the network. It handles client requests for residual computation, implicit
    solving, and Jacobian computation by delegating to the underlying implicit discipline.
    The server supports streaming of large arrays and handles all protocol buffer
    serialization/deserialization automatically.

    The server implements three main RPC methods:
    - ComputeResiduals: Evaluate residual equations R(inputs, outputs)
    - SolveResiduals: Solve implicit equations R(inputs, outputs) = 0
    - ComputeResidualGradients: Compute Jacobian dR/d[inputs,outputs]

    Key Features:
        - Automatic gRPC service implementation from discipline
        - Streaming support for large array transfers
        - Thread-safe concurrent client handling
        - Automatic variable metadata discovery and sharing
        - Option handling and configuration
        - Error handling and status reporting

    Typical Usage:
        >>> from concurrent import futures
        >>> import grpc
        >>> import philote_mdo.general as pmdo
        >>>
        >>> # Create your implicit discipline
        >>> discipline = MyImplicitDiscipline()
        >>>
        >>> # Create and configure server
        >>> server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        >>> impl_server = pmdo.ImplicitServer(discipline=discipline)
        >>> impl_server.attach_to_server(server)
        >>>
        >>> # Start server
        >>> server.add_insecure_port('[::]:50051')
        >>> server.start()
        >>> print('Server started on port 50051')
        >>> server.wait_for_termination()

    Attributes:
        _discipline (ImplicitDiscipline): The underlying implicit discipline being served

    Notes:
        - Inherits from both DisciplineServer and gRPC ImplicitServiceServicer
        - Automatically handles protocol buffer conversion and array streaming
        - Thread-safe for concurrent client connections
        - The underlying discipline must implement all required implicit methods
    """

    def __init__(self, discipline=None):
        """
        Initialize the implicit discipline server.

        Parameters
        ----------
        discipline : ImplicitDiscipline, optional
            The implicit discipline to serve. Must implement compute_residuals,
            solve_residuals, and residual_partials methods.

        Examples
        --------
        >>> discipline = MyImplicitDiscipline()
        >>> server = ImplicitServer(discipline=discipline)
        """
        super().__init__(discipline=discipline)

    def attach_to_server(self, server):
        """
        Attach this implicit service to a gRPC server.

        This method registers the implicit discipline service with the gRPC server,
        enabling clients to connect and call implicit discipline methods remotely.

        Parameters
        ----------
        server : grpc.Server
            The gRPC server instance to attach this service to

        Examples
        --------
        >>> server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        >>> impl_server = ImplicitServer(discipline=my_discipline)
        >>> impl_server.attach_to_server(server)
        >>> server.add_insecure_port('[::]:50051')
        >>> server.start()
        """
        super().attach_to_server(server)
        disc.add_ImplicitServiceServicer_to_server(self, server)

    def ComputeResiduals(self, request_iterator, context):
        """
        Handle client requests for residual computation.

        This gRPC method receives inputs and outputs from the client, calls the
        underlying discipline's compute_residuals method, and streams the computed
        residuals back to the client. The method handles protocol buffer conversion
        and array streaming automatically.

        Parameters
        ----------
        request_iterator : Iterator[data.Array]
            Stream of input and output arrays from the client
        context : grpc.ServicerContext
            gRPC context for the request

        Yields
        ------
        data.Array
            Stream of residual arrays back to the client

        Notes
        -----
        - Automatically handles large array streaming
        - Converts protocol buffers to numpy arrays for discipline computation
        - Streams results back in chunks for efficiency
        - This method is called automatically by the gRPC framework
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}
        residuals = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # Call the user-defined compute_residuals function
        self._discipline.compute_residuals(inputs, outputs, residuals)

        for res_name, value in residuals.items():
            for b, e in get_chunk_indices(value.size, self._stream_opts.num_double):
                yield data.Array(
                    name=res_name,
                    start=b,
                    end=e,
                    type=data.kResidual,
                    data=value.ravel()[b:e],
                )

    def SolveResiduals(self, request_iterator, context):
        """
        Handle client requests for implicit equation solving.

        This gRPC method receives inputs from the client, calls the underlying
        discipline's solve_residuals method to find outputs that satisfy
        R(inputs, outputs) = 0, and streams the solved outputs back to the client.

        Parameters
        ----------
        request_iterator : Iterator[data.Array]
            Stream of input arrays from the client
        context : grpc.ServicerContext
            gRPC context for the request

        Yields
        ------
        data.Array
            Stream of solved output arrays back to the client

        Notes
        -----
        - The discipline's solve_residuals method performs the actual solving
        - May raise exceptions if the solve fails to converge
        - Outputs are streamed back in chunks for large arrays
        - This method is called automatically by the gRPC framework
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # Call the user-defined solve function
        self._discipline.solve_residuals(inputs, outputs)

        for output_name, value in outputs.items():
            for b, e in get_chunk_indices(value.size, self._stream_opts.num_double):
                yield data.Array(
                    name=output_name,
                    start=b,
                    end=e,
                    type=data.kOutput,
                    data=value.ravel()[b:e],
                )

    def ComputeResidualGradients(self, request_iterator, context):
        """
        Handle client requests for residual Jacobian computation.

        This gRPC method receives inputs and outputs from the client, calls the
        underlying discipline's residual_partials method to compute the Jacobian
        dR/d[inputs,outputs], and streams the partial derivatives back to the client.

        Parameters
        ----------
        request_iterator : Iterator[data.Array]
            Stream of input and output arrays from the client
        context : grpc.ServicerContext
            gRPC context for the request

        Yields
        ------
        data.Array
            Stream of partial derivative arrays back to the client

        Notes
        -----
        - Computes both dR/dinputs and dR/doutputs partial derivatives
        - Results are streamed back with proper naming for Jacobian reconstruction
        - Used for gradient-based optimization and sensitivity analysis
        - This method is called automatically by the gRPC framework
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        jac = self.preallocate_partials()
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # Call the user-defined residual partials function
        self._discipline.residual_partials(inputs, outputs, jac)

        for jac, value in jac.items():
            for b, e in get_chunk_indices(value.size, self._stream_opts.num_double):
                yield data.Array(
                    name=jac[0],
                    subname=jac[1],
                    type=data.kPartial,
                    start=b,
                    end=e,
                    data=value.ravel()[b:e],
                )

    # def MatrixFreeGradients(self, request_iterator, context):
    #     """
    #     """
    #     pass
