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

    # per-variable serialization overhead allowance, in bytes: field tags and
    # length prefixes for the VariableMessage and Array envelopes, plus the
    # start, end, and type fields
    _MESSAGE_OVERHEAD_BYTES = 32

    def __init__(self, channel):
        # verbose outputs
        self.verbose = True

        # transport selection: "auto", "unary", or "stream"
        self.transport = "auto"

        # largest unary request or response this client will attempt, in bytes.
        # The unary transport wins on small and medium payloads, where per-call
        # framing dominates. Past a few hundred kilobytes a stream pulls ahead,
        # because splitting the payload lets the server serialize one chunk
        # while the client reads the previous one. Measured with
        # utils/bench_transport.py, the crossover sits between 256 KiB and
        # 512 KiB. A unary payload must also fit in a single gRPC message,
        # whose default ceiling is 4 MiB, so 256 KiB leaves wide headroom.
        self.unary_max_bytes = 1 << 18

        # unary negotiation state. None means the server has not been asked and
        # no unary call has been attempted yet.
        self._unary_supported = None
        self._server_unary_max_bytes = 0

        # RPCs whose payload is known to be too large for the unary transport.
        # Continuous shapes are fixed after setup, so these verdicts are final.
        self._stream_only_rpcs = set()

        # discipline properties
        self._name = ""
        self._version = ""
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
        self._is_continuous = response.continuous
        self._is_differentiable = response.differentiable
        self._provides_gradients = response.provides_gradients
        self._name = response.name
        self._version = response.version

        # transport capabilities. A server that predates the unary RPCs leaves
        # these at the proto3 defaults, which correctly reads as "no unary".
        self._unary_supported = response.supports_unary
        self._server_unary_max_bytes = response.max_unary_bytes

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

    def get_dynamic_variables(self):
        """
        Returns a list of variable metadata entries that have
        ``dynamic_shape`` set to ``True``.
        """
        return [v for v in self._var_meta if v.dynamic_shape]

    def set_variable_shape(self, name, shape, var_type=data.VariableType.kInput):
        """
        Creates a ``VariableMetaData`` message for setting a dynamic
        variable's shape.

        Parameters
        ----------
        name : str
            the name of the variable
        shape : tuple
            the desired shape
        var_type : VariableType
            the variable type (kInput or kOutput)

        Returns
        -------
        VariableMetaData
            protobuf message ready for ``send_variable_shapes``
        """
        meta = data.VariableMetaData()
        meta.type = var_type
        meta.name = name
        meta.shape.extend(shape)
        return meta

    def send_variable_shapes(self, variable_metadata):
        """
        Sends shapes for variables flagged as ``dynamic_shape``.

        Call after ``get_variable_definitions()`` and before compute
        calls.

        Parameters
        ----------
        variable_metadata : list of VariableMetaData
            shapes for dynamic variables
        """
        self._disc_stub.SetVariableShapes(iter(variable_metadata))

        # update local metadata to reflect the new shapes
        for meta in variable_metadata:
            for var in self._var_meta:
                if var.name == meta.name and var.type == meta.type:
                    var.shape[:] = []
                    var.shape.extend(meta.shape)
                    break

            # for implicit outputs, also update the matching residual
            if meta.type == data.VariableType.kOutput:
                for var in self._var_meta:
                    if (
                        var.name == meta.name
                        and var.type == data.VariableType.kResidual
                    ):
                        var.shape[:] = []
                        var.shape.extend(meta.shape)
                        break

    def _estimated_bytes(self, var_types):
        """
        Estimates the serialized size of a payload carrying every continuous
        variable of the given types.

        The estimate is built from metadata gathered during setup, so it is
        available before any data is assembled. It is deliberately generous --
        this is a gate, not an accountant.

        :param var_types: tuple of VariableType values. A tuple containing
            data.kPartial selects the partials metadata instead.
        :return: estimated payload size in bytes
        """
        total = 0

        if data.kPartial in var_types:
            for part in self._partials_meta:
                shapef = tuple(
                    [d.shape for d in self._var_meta if d.name == part.name][0]
                )
                shapex = tuple(
                    [d.shape for d in self._var_meta if d.name == part.subname][0]
                )

                total += (
                    8 * int(np.prod(utils.get_partials_shape(shapef, shapex)))
                    + len(part.name)
                    + len(part.subname)
                    + self._MESSAGE_OVERHEAD_BYTES
                )

            return total

        for var in self._var_meta:
            if var.type in var_types:
                total += (
                    8 * int(np.prod(var.shape))
                    + len(var.name)
                    + self._MESSAGE_OVERHEAD_BYTES
                )

        return total

    def _unary_limit(self):
        """
        Returns the effective unary size limit, in bytes.

        The server advertises a capacity ceiling and the client holds a
        performance threshold, so the effective limit is the smaller of the
        two. A server that does not advertise leaves the field at zero.
        """
        if self._server_unary_max_bytes:
            return min(self.unary_max_bytes, self._server_unary_max_bytes)

        return self.unary_max_bytes

    def _unary_allowed(self, rpc_name, request_types, response_types):
        """
        Decides whether the unary transport can carry this call.

        Continuous variable shapes are fixed once setup completes, so this
        verdict is deterministic and is cached per RPC rather than being
        recomputed on every call. Running before the messages are assembled
        also means an oversized payload is never built only to be discarded.
        """
        if self.transport == "stream" or rpc_name in self._stream_only_rpcs:
            return False

        if self.transport == "unary":
            return True

        if self._unary_supported is False:
            return False

        limit = self._unary_limit()

        if (
            max(
                self._estimated_bytes(request_types),
                self._estimated_bytes(response_types),
            )
            > limit
        ):
            # the shapes cannot change, so neither can this verdict
            self._stream_only_rpcs.add(rpc_name)
            return False

        return True

    def _within_unary_limit(self, messages):
        """
        Guards against a payload whose size is not fixed at setup time, which
        in practice means an oversized discrete variable.

        Deliberately does not demote the RPC: a large discrete value on one
        call says nothing about the size of the next one.
        """
        limit = self._unary_limit()
        total = 0

        for message in messages:
            total += message.ByteSize()

            if total > limit:
                return False

        return True

    def _demote(self, rpc_name, code):
        """
        Records a failed unary attempt and reports whether the call should be
        retried over the streaming transport.

        Any status other than the two handled here is a real server error and
        must propagate rather than being silently retried.

        Note that a RESOURCE_EXHAUSTED raised while receiving the response
        means the server already ran the discipline, so the streaming retry
        runs it a second time. That is harmless for the pure-function contract
        Philote assumes, but a discipline that carries state between calls
        should pin transport="stream".
        """
        if code == grpc.StatusCode.UNIMPLEMENTED:
            # the server predates the unary RPCs
            self._unary_supported = False
            return True

        if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            if not self._discrete_var_meta:
                # nothing in this payload can vary in size, so it will recur
                self._stream_only_rpcs.add(rpc_name)
            return True

        return False

    def _dispatch_compute(
        self,
        rpc_name,
        unary_method,
        stream_method,
        inputs,
        outputs=None,
        discrete_inputs=None,
        discrete_outputs=None,
        request_types=(data.kInput,),
        response_types=(data.kOutput,),
    ):
        """
        Sends a compute request over the best available transport.

        Returns an iterable of response ``VariableMessage`` objects, suitable
        for any of the ``_recover_*`` methods regardless of which transport was
        used -- a ``VariableSet``'s ``variables`` field is iterable, just as a
        response stream is.
        """
        if self._unary_allowed(rpc_name, request_types, response_types):
            messages = self._assemble_input_messages(
                inputs,
                outputs,
                discrete_inputs,
                discrete_outputs,
                chunked=False,
            )

            # an explicit pin means "force unary", so the per-call guard is
            # skipped too. An oversized payload then fails with
            # RESOURCE_EXHAUSTED and is retried on the stream by _demote,
            # rather than being quietly rerouted before it is ever sent.
            if self.transport == "unary" or self._within_unary_limit(messages):
                try:
                    response = unary_method(
                        data.VariableSet(variables=messages)
                    )
                    return response.variables
                except grpc.RpcError as e:
                    if not self._demote(rpc_name, e.code()):
                        raise

        messages = self._assemble_input_messages(
            inputs, outputs, discrete_inputs, discrete_outputs
        )

        return stream_method(iter(messages))

    def _input_chunk_size(self, value, chunked):
        """
        Resolves the chunk size used to serialize one input array.

        The unary transport passes ``chunked=False`` so that each variable
        becomes exactly one message -- fragmenting at ``num_double`` would
        split a large array across several sub-messages of one request for no
        benefit.
        """
        if chunked:
            return self._stream_options.num_double

        # get_chunk_indices rejects a chunk size below one, which a zero-size
        # array would otherwise produce
        return max(int(value.size), 1)

    def _assemble_input_messages(
        self,
        inputs,
        outputs=None,
        discrete_inputs=None,
        discrete_outputs=None,
        chunked=True,
    ):
        """
        Assembles the messages for transmitting the input variables to the
        server.

        Both continuous and discrete inputs are wrapped in ``VariableMessage``
        envelopes. The same list serves both transports: the streaming path
        sends it as a stream, the unary path wraps it in a ``VariableSet``.
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
                value.size, self._input_chunk_size(value, chunked)
            ):
                message = data.VariableMessage(
                    continuous=data.Array(
                        name=input_name,
                        start=b,
                        end=e - 1,
                        type=data.VariableType.kInput,
                    )
                )
                utils.set_array_data(message.continuous, value.ravel()[b:e])

                messages += [message]

        # Continuous outputs (for implicit disciplines)
        if outputs:
            for output_name, value in outputs.items():
                for b, e in utils.get_chunk_indices(
                    value.size, self._input_chunk_size(value, chunked)
                ):
                    message = data.VariableMessage(
                        continuous=data.Array(
                            name=output_name,
                            start=b,
                            end=e - 1,
                            type=data.VariableType.kOutput,
                        )
                    )
                    utils.set_array_data(message.continuous, value.ravel()[b:e])

                    messages += [message]

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
                    if len(arr.data) > 0:
                        utils.read_array_into(arr, flat_outputs[arr.name])
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
                    if len(arr.data) > 0:
                        utils.read_array_into(arr, flat_residuals[arr.name])
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

            partials[(part.name, part.subname)] = np.zeros(
                utils.get_partials_shape(shapef, shapex)
            )
            flat_p[(part.name, part.subname)] = utils.get_flattened_view(
                partials[(part.name, part.subname)]
            )

        for message in responses:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous

                if arr.type == data.kPartial:
                    if len(arr.data) > 0:
                        utils.read_array_into(
                            arr, flat_p[(arr.name, arr.subname)]
                        )
                    else:
                        raise PhiloteValidationError(
                            "Expected continuous outputs for the "
                            "partials, but array was empty."
                        )

        return partials
