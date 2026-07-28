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
from philote_mdo.general.discipline_server import DisciplineServer


class ExplicitServer(DisciplineServer, disc.ExplicitServiceServicer):
    """
    Base class for remote explicit components.
    """

    _supports_unary = True

    def __init__(self, discipline=None):
        super().__init__(discipline=discipline)

    def attach_to_server(self, server):
        """
        Attaches this discipline server class to a gRPC server.
        """
        super().attach_to_server(server)
        disc.add_ExplicitServiceServicer_to_server(self, server)

    def _compute_function(self, requests):
        """
        Runs the discipline compute function over a set of input messages.

        Shared by the streaming and unary transports, which differ only in how
        the results are serialized.

        :param requests: iterable of VariableMessage inputs
        :return: tuple of the continuous and discrete output dictionaries
        """
        inputs = {}
        flat_inputs = {}
        outputs = {}
        discrete_inputs = {}
        discrete_outputs = {}

        self.preallocate_inputs(inputs, flat_inputs)
        discrete_inputs, _ = self.process_inputs(
            requests, flat_inputs, discrete_inputs=discrete_inputs
        )

        # Call compute with discrete data when discrete variables are present
        if discrete_inputs or self._discipline._discrete_var_meta:
            self._discipline.compute(
                inputs, outputs, discrete_inputs, discrete_outputs
            )
        else:
            self._discipline.compute(inputs, outputs)

        return outputs, discrete_outputs

    def _compute_gradient(self, requests):
        """
        Runs the discipline compute_partials function over a set of input
        messages.

        :param requests: iterable of VariableMessage inputs
        :return: PairDict of Jacobian blocks
        """
        inputs = {}
        flat_inputs = {}
        discrete_inputs = {}

        self.preallocate_inputs(inputs, flat_inputs)
        jac = self.preallocate_partials()
        discrete_inputs, _ = self.process_inputs(
            requests, flat_inputs, discrete_inputs=discrete_inputs
        )

        if discrete_inputs or self._discipline._discrete_var_meta:
            self._discipline.compute_partials(inputs, jac, discrete_inputs)
        else:
            self._discipline.compute_partials(inputs, jac)

        return jac

    def ComputeFunction(self, request_iterator, context):
        """
        Computes the function evaluation and streams the result to the client.
        """
        with self._rpc_errors(context, "ComputeFunction"):
            outputs, discrete_outputs = self._compute_function(request_iterator)

            yield from self._continuous_messages(outputs, data.kOutput)
            yield from self._discrete_messages(discrete_outputs)

    def ComputeGradient(self, request_iterator, context):
        """
        Computes the gradient evaluation and streams the result to the client.
        """
        with self._rpc_errors(context, "ComputeGradient"):
            jac = self._compute_gradient(request_iterator)

            yield from self._partial_messages(jac)

    def ComputeFunctionUnary(self, request, context):
        """
        Computes the function evaluation and returns the result in a single
        message.
        """
        with self._rpc_errors(context, "ComputeFunctionUnary"):
            outputs, discrete_outputs = self._compute_function(request.variables)

            return data.VariableSet(
                variables=[
                    *self._continuous_messages(
                        outputs, data.kOutput, chunked=False
                    ),
                    *self._discrete_messages(discrete_outputs),
                ]
            )

    def ComputeGradientUnary(self, request, context):
        """
        Computes the gradient evaluation and returns the result in a single
        message.
        """
        with self._rpc_errors(context, "ComputeGradientUnary"):
            jac = self._compute_gradient(request.variables)

            return data.VariableSet(
                variables=list(
                    self._partial_messages(jac, chunked=False)
                )
            )
