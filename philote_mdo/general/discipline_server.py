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

import philote_mdo.generated.data_pb2 as data
import philote_mdo.generated.disciplines_pb2_grpc as disc
from google.protobuf.empty_pb2 import Empty
from philote_mdo.utils import PairDict, get_flattened_view


class DisciplineServer(disc.DisciplineService):
    """
    Base class for all server classes.
    """

    def __init__(self, discipline=None):
        self.verbose = False

        # user/developer supplied discipline
        self._discipline = discipline

        # discipline stream options
        self._stream_opts = data.StreamOptions(num_double=1000)

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
        yield data.DisciplineProperties(
            continuous=self._discipline._is_continuous,
            differentiable=self._discipline._is_differentiable,
            provides_gradients=self._discipline._provides_gradients,
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
            else:
                raise ValueError(
                    "Invalid value for discipline option '{}'".format(name)
                )

            opts.type.append(type)

        return opts

    def SetOptions(self, request, context):
        """
        RPC that sets the discipline options.
        """
        options = request.options
        self._discipline.set_options(options)
        return Empty()

    def Setup(self, request, context):
        """
        RPC that runs the setup function
        """
        self._discipline._clear_data()
        self._discipline.setup()
        self._discipline.setup_partials()
        return Empty()

    def GetVariableDefinitions(self, request, context):
        """
        Transmits variable metadata about the analysis discipline to the client.
        """
        for var in self._discipline._var_meta:
            yield var

    def GetPartialDefinitions(self, request, context):
        """
        Transmits partials metadata about the analysis discipline to the client.
        """
        for jac in self._discipline._partials_meta:
            yield jac

    def preallocate_inputs(self, inputs, flat_inputs, outputs=None, flat_outputs=None):
        """
        Preallocates the inputs before receiving data from the client.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.
        """
        for var in self._discipline._var_meta:
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

        for pair in self._discipline._partials_meta:
            shapef = tuple(
                [d.shape for d in self._discipline._var_meta if d.name == pair.name][0]
            )
            shapex = tuple(
                [d.shape for d in self._discipline._var_meta if d.name == pair.subname][
                    0
                ]
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

            jac[(pair.name, pair.subname)] = np.zeros(shape)

        return jac

    def process_inputs(self, request_iterator, flat_inputs, flat_outputs=None):
        """
        Processes the message inputs from a gRPC stream.

        Note, for implicit disciplines, the function values are considered
        inputs to evaluate the residuals and the partials of the residuals.
        """
        # process inputs
        for message in request_iterator:
            # start and end indices for the array chunk
            b = message.start
            e = message.end

            # assign either continuous or discrete data
            if len(message.data) > 0:
                if message.type == data.VariableType.kInput:
                    flat_inputs[message.name][b : e + 1] = message.data
                elif message.type == data.VariableType.kOutput:
                    flat_outputs[message.name][b : e + 1] = message.data
            else:
                raise ValueError(
                    "Expected continuous variables but arrays were"
                    " empty for variable %s." % (message.name)
                )
