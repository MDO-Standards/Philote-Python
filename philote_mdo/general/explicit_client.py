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
from philote_mdo.utils.validation import PhiloteServerError, validate_is_dict
import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc


class ExplicitClient(DisciplineClient):
    """
    Client for calling explicit analysis discipline servers.
    """

    def __init__(self, channel):
        super().__init__(channel)
        self._expl_stub = disc.ExplicitServiceStub(channel)

    def run_compute(self, inputs, discrete_inputs=None):
        """
        Requests and receives the function evaluation from the analysis server
        for a set of inputs (sent to the server).

        Parameters
        ----------
        inputs : dict
            Continuous input values.
        discrete_inputs : dict, optional
            Discrete input values.

        Returns
        -------
        dict or tuple(dict, dict)
            Continuous outputs, or (continuous outputs, discrete outputs) when
            the server returns discrete output data.
        """
        validate_is_dict(inputs, "run_compute (inputs)")
        try:
            responses = self._dispatch_compute(
                "ComputeFunction",
                self._expl_stub.ComputeFunctionUnary,
                self._expl_stub.ComputeFunction,
                inputs,
                discrete_inputs=discrete_inputs,
                request_types=(data.kInput,),
                response_types=(data.kOutput,),
            )
            return self._recover_outputs(responses)
        except grpc.RpcError as e:
            raise PhiloteServerError(
                f"Server error during run_compute: {e.details()}"
            ) from e

    def run_compute_partials(self, inputs, discrete_inputs=None):
        """
        Requests and receives the gradient evaluation from the analysis server
        for a set of inputs (sent to the server).

        Parameters
        ----------
        inputs : dict
            Continuous input values.
        discrete_inputs : dict, optional
            Discrete input values.
        """
        validate_is_dict(inputs, "run_compute_partials (inputs)")
        try:
            responses = self._dispatch_compute(
                "ComputeGradient",
                self._expl_stub.ComputeGradientUnary,
                self._expl_stub.ComputeGradient,
                inputs,
                discrete_inputs=discrete_inputs,
                request_types=(data.kInput,),
                response_types=(data.kPartial,),
            )
            partials = self._recover_partials(responses)

            return partials
        except grpc.RpcError as e:
            raise PhiloteServerError(
                f"Server error during run_compute_partials: {e.details()}"
            ) from e
