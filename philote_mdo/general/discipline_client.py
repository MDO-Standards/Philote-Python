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
import google.protobuf.empty_pb2 as empty
import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc
import philote_mdo.utils as utils
from philote_mdo.general.discipline_server import _python_to_value, _value_to_python
from philote_mdo.utils.validation import (
    PhiloteValidationError,
    validate_is_dict,
    validate_numpy_array,
)


class DisciplineClient:
    """
    Base class for analysis discipline clients.
    """

    def __init__(self, channel):
        # verbose outputs
        self.verbose = True

        # grpc options
        self.grpc_options = []

        # discipline properties
        self._is_continuous = False
        self._is_differentiable = False
        self._provides_gradients = False

        # discipline client stub
        self._disc_stub = disc.DisciplineServiceStub(channel)

        # streaming options
        self._stream_options = data.StreamOptions(num_double=1000)

        # variable and partials metadata
        self._var_meta = []
        self._discrete_var_meta = []
        self._partials_meta = []

        # list of available options
        self.options_list = {}

    def get_discipline_info(self):
        """
        Gets the discipline properties from the analysis server.
        """
        response = self._disc_stub.GetInfo(empty.Empty())
        self._is_continuous = response[0].continuous
        self._is_differentiable = response[0].differentiable
        self._provides_gradients = response[0].provides_gradients

    def send_stream_options(self):
        """
        Transmits the stream options for the remote analysis to the server.
        """
        self._disc_stub.SetStreamOptions(self._stream_options)

    def get_available_options(self):
        """
        Gets the available options for the analysis discipline.
        """
        opts = self._disc_stub.GetAvailableOptions(empty.Empty())

        for name, val in zip(opts.options, opts.type):
            type_str = None
            if val == data.kBool:
                type_str = "bool"
            if val == data.kInt:
                type_str = "int"
            if val == data.kDouble:
                type_str = "float"
            if val == data.kString:
                type_str = "str"
            if val == data.kStruct:
                type_str = "dict"
            self.options_list[name] = type_str

    def send_options(self, options):
        """
        Sends the discipline options to the analysis server.

        Parameters
        ----------
        options: dict
            Dictionary containing the discipline options

        Returns
        -------
            None
        """
        validate_is_dict(options, "send_options")
        proto_options = data.DisciplineOptions()
        proto_options.options.update(options)
        self._disc_stub.SetOptions(proto_options)

    def run_setup(self):
        """
        Runs the setup function on the analysis server.
        """
        self._disc_stub.Setup(empty.Empty())

    def get_variable_definitions(self):
        """
        Requests the input and output metadata from the server.

        Both continuous and discrete variable metadata are stored in their
        respective lists.
        """
        for message in self._disc_stub.GetVariableDefinitions(empty.Empty()):
            if message.type in (
                data.VariableType.kDiscreteInput,
                data.VariableType.kDiscreteOutput,
            ):
                self._discrete_var_meta += [message]
            else:
                self._var_meta += [message]

    def get_partials_definitions(self):
        """
        Requests metadata information on the partials from the analysis server.
        """
        for message in self._disc_stub.GetPartialDefinitions(empty.Empty()):
            if message.name not in self._partials_meta:
                self._partials_meta += [message]

    def _assemble_input_messages(
        self, inputs, outputs=None, discrete_inputs=None, discrete_outputs=None
    ):
        """
        Assembles the messages for transmitting the input variables to the
        server.

        Both continuous and discrete inputs are wrapped in ``VariableMessage``
        envelopes.
        """
        validate_is_dict(inputs, "_assemble_input_messages (inputs)")
        for input_name, value in inputs.items():
            validate_numpy_array(value, input_name)
        if outputs is not None:
            validate_is_dict(outputs, "_assemble_input_messages (outputs)")
            for output_name, value in outputs.items():
                validate_numpy_array(value, output_name)

        messages = []

        # Continuous inputs
        for input_name, value in inputs.items():
            for b, e in utils.get_chunk_indices(
                value.size, self._stream_options.num_double
            ):
                messages += [
                    data.VariableMessage(
                        continuous=data.Array(
                            name=input_name,
                            start=b,
                            end=e - 1,
                            type=data.VariableType.kInput,
                            data=value.ravel()[b:e],
                        )
                    )
                ]

        # Continuous outputs (for implicit disciplines)
        if outputs:
            for output_name, value in outputs.items():
                for b, e in utils.get_chunk_indices(
                    value.size, self._stream_options.num_double
                ):
                    messages += [
                        data.VariableMessage(
                            continuous=data.Array(
                                name=output_name,
                                start=b,
                                end=e - 1,
                                type=data.VariableType.kOutput,
                                data=value.ravel()[b:e],
                            )
                        )
                    ]

        # Discrete inputs
        if discrete_inputs:
            for name, value in discrete_inputs.items():
                messages += [
                    data.VariableMessage(
                        discrete=data.DiscreteVariable(
                            name=name,
                            type=data.VariableType.kDiscreteInput,
                            value=_python_to_value(value),
                        )
                    )
                ]

        # Discrete outputs (for implicit disciplines)
        if discrete_outputs:
            for name, value in discrete_outputs.items():
                messages += [
                    data.VariableMessage(
                        discrete=data.DiscreteVariable(
                            name=name,
                            type=data.VariableType.kDiscreteOutput,
                            value=_python_to_value(value),
                        )
                    )
                ]

        return messages

    def _recover_outputs(self, responses):
        """
        Recovers the outputs from the stream of responses.

        Returns both continuous outputs and discrete outputs.
        """
        outputs = {}
        flat_outputs = {}
        discrete_outputs = {}

        # preallocate continuous outputs
        for out in self._var_meta:
            if out.type == data.kOutput:
                name = out.name
                outputs[name] = np.zeros(out.shape)
                flat_outputs[name] = utils.get_flattened_view(outputs[name])

        for message in responses:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous
                if arr.type == data.kOutput:
                    b = arr.start
                    e = arr.end + 1
                    if len(arr.data) > 0:
                        flat_outputs[arr.name][b:e] = arr.data
                    else:
                        raise PhiloteValidationError(
                            "Expected continuous variables, but array is empty."
                        )

            elif variant == "discrete":
                dv = message.discrete
                if dv.type == data.VariableType.kDiscreteOutput:
                    discrete_outputs[dv.name] = _value_to_python(dv.value)

        if discrete_outputs:
            return outputs, discrete_outputs
        return outputs

    def _recover_residuals(self, responses):
        """
        Recovers the residuals from the stream of responses.
        """
        residuals = {}
        flat_residuals = {}

        # preallocate
        for res in self._var_meta:
            if res.type == data.kResidual:
                name = res.name
                residuals[name] = np.zeros(res.shape)
                flat_residuals[name] = utils.get_flattened_view(residuals[name])

        for message in responses:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous
                if arr.type == data.kResidual:
                    b = arr.start
                    e = arr.end + 1
                    if len(arr.data) > 0:
                        flat_residuals[arr.name][b:e] = arr.data
                    else:
                        raise PhiloteValidationError(
                            "Expected continuous variables, but array is empty."
                        )

        return residuals

    def _recover_partials(self, responses):
        """
        Recovers the partials from the stream of responses.
        """
        partials = utils.PairDict()
        flat_p = utils.PairDict()

        # preallocate
        for part in self._partials_meta:
            shapef = tuple([d.shape for d in self._var_meta if d.name == part.name][0])
            shapex = tuple(
                [d.shape for d in self._var_meta if d.name == part.subname][0]
            )

            if shapef == (1,):
                if shapex == (1,):
                    shape = (1,)
                else:
                    shape = shapex
            elif shapex == (1,):
                shape = shapef
            else:
                shape = shapef + shapex

            partials[(part.name, part.subname)] = np.zeros(shape)
            flat_p[(part.name, part.subname)] = utils.get_flattened_view(
                partials[(part.name, part.subname)]
            )

        for message in responses:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous
                b = arr.start
                e = arr.end + 1

                if arr.type == data.kPartial:
                    if len(arr.data) > 0:
                        flat_p[(arr.name, arr.subname)][b:e] = arr.data
                    else:
                        raise PhiloteValidationError(
                            "Expected continuous outputs for the "
                            "partials, but array was empty."
                        )

        return partials
