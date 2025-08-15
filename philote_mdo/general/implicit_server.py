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
    Base class for creating an implicit discipline server.
    """

    def __init__(self, discipline=None):
        super().__init__(discipline=discipline)

    def attach_to_server(self, server):
        """
        Attaches this discipline server class to a gRPC server.
        """
        super().attach_to_server(server)
        disc.add_ImplicitServiceServicer_to_server(self, server)

    def ComputeResiduals(self, request_iterator, context):
        """
        Computes the residuals and sends the results to the client.
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}
        residuals = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # call the user-defined compute_residuals function
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
        Solves the implicit discipline so that the residuals are driven to zero.
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # call the user-defined solve function
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
        Computes the residual gradients and sends the results to the client.
        """
        # inputs and outputs
        inputs = {}
        flat_inputs = {}
        outputs = {}
        flat_outputs = {}

        self.preallocate_inputs(inputs, flat_inputs, outputs, flat_outputs)
        jac = self.preallocate_partials()
        self.process_inputs(request_iterator, flat_inputs, flat_outputs)

        # call the user-defined residual partials function
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
