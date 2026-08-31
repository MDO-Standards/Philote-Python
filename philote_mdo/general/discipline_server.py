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

import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc
from google.protobuf.empty_pb2 import Empty
from google.protobuf import struct_pb2
from philote_mdo.utils import (
    PairDict,
    build_shape_index,
    get_flattened_view,
    get_function_shape,
    get_partials_shape,
    get_variable_shape,
    read_array_into,
)
from philote_mdo.utils.validation import PhiloteValidationError, validate_shape


class DisciplineServer(disc.DisciplineServiceServicer):
    """
    Base class for all server classes.
    """

    def __init__(self, discipline=None):
        self.verbose = False

        # user/developer supplied discipline
        self._discipline = discipline

        # discipline stream options
        # doubles per message. The cost of a stream is dominated by the number
        # of messages in it rather than by their size, so this wants to be as
        # large as the message ceiling safely allows: at 100k doubles a chunk
        # is about 780 KiB, or a fifth of gRPC's 4 MiB default, which leaves
        # room for metadata and for a peer that has lowered the limit.
        self._stream_opts = data.StreamOptions(num_double=100000)

    def attach_to_server(self, server):
        """
        Attaches this discipline server class to a gRPC server.
        """
        disc.add_DisciplineServiceServicer_to_server(self, server)

    def attach_discipline(self, impl):
        """
        Adds a discipline implementation to the server.
        """
        self._discipline = impl

    def GetInfo(self, request, context):
        """
        RPC that sends the discipline information/properties to the client.
        """
        return data.DisciplineProperties(
            continuous=self._discipline._is_continuous,
            differentiable=self._discipline._is_differentiable,
            provides_gradients=self._discipline._provides_gradients,
            name=self._discipline._name,
            version=self._discipline._version,
        )

    def SetStreamOptions(self, request, context):
        """
        Receives options from the client on how data will be transmitted to and
        received from the client. The options are stores locally for use in the
        compute routines.
        """
        self._stream_opts = request
        return Empty()

    def GetAvailableOptions(self, request, context):
        """
        RPC that gets the names and types of all available discipline options.
        """
        try:
            opts_dict = self._discipline.options_list
            opts = data.OptionsList()

            for name, val in opts_dict.items():
                opts.options.append(name)

                # assign the correct data type
                if val == "bool":
                    type = data.kBool
                elif val == "int":
                    type = data.kInt
                elif val == "float":
                    type = data.kDouble
                elif val == "str":
                    type = data.kString
                elif val == "dict":
                    type = data.kStruct
                else:
                    raise PhiloteValidationError(
                        "Invalid value for discipline option '{}'".format(name)
                    )

                opts.type.append(type)

            return opts
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"GetAvailableOptions failed: {e}"
            )

    def SetOptions(self, request, context):
        """
        RPC that sets the discipline options.
        """
        try:
            options = request.options
            self._discipline.set_options(options)
            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"SetOptions failed: {e}"
            )

    def Setup(self, request, context):
        """
        RPC that runs the setup function
        """
        try:
            self._discipline._clear_data()
            self._discipline.setup()
            self._discipline.setup_partials()
            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"Setup failed: {e}"
            )

    def GetVariableDefinitions(self, request, context):
        """
        Transmits variable metadata about the analysis discipline to the client.

        Both continuous and discrete variable metadata are streamed.
        """
        for var in self._discipline._var_meta:
            yield var

        for var in self._discipline._discrete_var_meta:
            yield var

    def GetPartialDefinitions(self, request, context):
        """
        Transmits partials metadata about the analysis discipline to the client.
        """
        for jac in self._discipline._partials_meta:
            yield jac

    def SetVariableShapes(self, request_iterator, context):
        """
        Receives client-defined shapes for variables flagged as
        dynamic_shape.

        The client must call this RPC after GetVariableDefinitions and
        before any compute RPCs for disciplines that contain variables
        with dynamic shapes.
        """
        try:
            # index by type and name once; searching the list per incoming
            # message is quadratic in the number of variables
            index = {}
            for var in self._discipline._var_meta:
                index.setdefault((var.type, var.name), var)

            for meta in request_iterator:
                validate_shape(tuple(meta.shape), "SetVariableShapes")

                var = index.get((meta.type, meta.name))

                if var is None:
                    raise PhiloteValidationError(
                        f"SetVariableShapes: variable '{meta.name}' "
                        f"not found."
                    )

                if not var.dynamic_shape:
                    raise PhiloteValidationError(
                        f"Variable '{meta.name}' does not allow "
                        f"dynamic shapes."
                    )

                var.shape[:] = []
                var.shape.extend(meta.shape)

                # if the variable is an output on an implicit discipline,
                # also update the matching residual entry
                if meta.type == data.VariableType.kOutput:
                    res = index.get((data.VariableType.kResidual, meta.name))

                    if res is not None and res.dynamic_shape:
                        res.shape[:] = []
                        res.shape.extend(meta.shape)

            return Empty()
        except PhiloteValidationError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            context.abort(
                grpc.StatusCode.INTERNAL, f"SetVariableShapes failed: {e}"
            )

    def preallocate_inputs(self, inputs, flat_inputs, outputs=None, flat_outputs=None):
        """
        Preallocates the inputs before receiving data from the client.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.
        """
        for var in self._discipline._var_meta:
            # validate that dynamic-shape variables have been resolved
            if var.dynamic_shape and len(var.shape) == 0:
                raise PhiloteValidationError(
                    f"Variable '{var.name}' has dynamic_shape=True but "
                    f"no shape has been set. Call SetVariableShapes "
                    f"before computing."
                )

            if var.type == data.kInput:
                inputs[var.name] = np.zeros(var.shape)
                flat_inputs[var.name] = get_flattened_view(inputs[var.name])

            if (
                var.type == data.kOutput
                and outputs is not None
                and flat_outputs is not None
            ):
                outputs[var.name] = np.zeros(var.shape)
                flat_outputs[var.name] = get_flattened_view(outputs[var.name])

    def preallocate_partials(self):
        """
        Preallocates the partials.

        Note: there are edge cases for this function, where either f or x, or
        both are scalar. In those cases the shapes of the partials must be
        treated differently.
        """
        jac = PairDict()

        # index the metadata by (type, name) once; scanning it per partial is
        # quadratic in the number of variables
        shapes = build_shape_index(self._discipline._var_meta)

        for pair in self._discipline._partials_meta:
            shape = get_partials_shape(
                get_function_shape(shapes, pair.name, "preallocate_partials"),
                get_variable_shape(shapes, pair.subname, "preallocate_partials"),
            )

            jac[(pair.name, pair.subname)] = np.zeros(shape)

        return jac

    def process_inputs(
        self,
        request_iterator,
        flat_inputs,
        flat_outputs=None,
        discrete_inputs=None,
        discrete_outputs=None,
    ):
        """
        Processes the message inputs from a gRPC stream.

        The stream consists of ``VariableMessage`` wrappers, each of which
        contains either a continuous ``Array`` or a ``DiscreteVariable``.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.
        """
        if discrete_inputs is None:
            discrete_inputs = {}
        if discrete_outputs is None:
            discrete_outputs = {}

        for message in request_iterator:
            variant = message.WhichOneof("payload")

            if variant == "continuous":
                arr = message.continuous

                if len(arr.data) > 0:
                    if arr.type == data.VariableType.kInput:
                        read_array_into(arr, flat_inputs[arr.name])
                    elif arr.type == data.VariableType.kOutput:
                        read_array_into(arr, flat_outputs[arr.name])
                else:
                    raise PhiloteValidationError(
                        "Expected continuous variables but arrays were"
                        " empty for variable %s." % (arr.name)
                    )

            elif variant == "discrete":
                dv = message.discrete
                # Convert protobuf Value to native Python type
                native_value = _value_to_python(dv.value)

                if dv.type == data.VariableType.kDiscreteInput:
                    discrete_inputs[dv.name] = native_value
                elif dv.type == data.VariableType.kDiscreteOutput:
                    discrete_outputs[dv.name] = native_value

        return discrete_inputs, discrete_outputs


def _value_to_python(value):
    """
    Converts a ``google.protobuf.Value`` to a native Python object.

    Parameters
    ----------
    value : google.protobuf.Value
        protobuf Value message

    Returns
    -------
    object
        Native Python equivalent (None, bool, int/float, str, list, or dict)
    """
    kind = value.WhichOneof("kind")

    if kind == "null_value":
        return None
    elif kind == "bool_value":
        return value.bool_value
    elif kind == "number_value":
        # protobuf stores all numbers as doubles; return int if lossless
        num = value.number_value
        if num == int(num):
            return int(num)
        return num
    elif kind == "string_value":
        return value.string_value
    elif kind == "list_value":
        return [_value_to_python(v) for v in value.list_value.values]
    elif kind == "struct_value":
        return {k: _value_to_python(v) for k, v in value.struct_value.fields.items()}
    else:  # pragma: no cover – all protobuf Value kinds are handled above
        return None


def _python_to_value(obj):
    """
    Converts a native Python object to a ``google.protobuf.Value``.

    Parameters
    ----------
    obj : object
        A Python scalar, list, or dict

    Returns
    -------
    google.protobuf.Value
        protobuf Value message
    """
    val = struct_pb2.Value()

    if obj is None:
        val.null_value = 0
    elif isinstance(obj, bool):
        val.bool_value = obj
    elif isinstance(obj, (int, float)):
        val.number_value = float(obj)
    elif isinstance(obj, str):
        val.string_value = obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            val.list_value.values.append(_python_to_value(item))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            val.struct_value.fields[str(k)].CopyFrom(_python_to_value(v))
    else:
        val.string_value = str(obj)

    return val
