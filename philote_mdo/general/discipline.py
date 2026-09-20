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
import philote_mdo.generated.data_pb2 as data
from philote_mdo.utils.validation import (
    validate_name,
    validate_shape,
    validate_units,
    validate_option_type,
    PhiloteValidationError,
)


class Discipline:
    """
    Base class for defining disciplines
    """

    def __init__(self):
        # discipline properties
        self._name = ""
        self._version = ""
        self._is_continuous = False
        self._is_differentiable = False
        self._provides_gradients = False

        # variable metadata
        self._var_meta = []

        # discrete variable metadata (name → default value)
        self._discrete_var_meta = []

        # partials metadata
        self._partials_meta = []

        # (type, name) of every declared variable, so that the duplicate check
        # on each add does not have to scan the lists above, which is
        # quadratic in the size of the discipline
        self._declared = set()

        # flag that indicates the discipline is implicit
        self._is_implicit = False

        # the job that owns this instance, assigned by the server when the
        # job is created. One discipline instance serves exactly one job, so
        # anything stored on self is private to that client.
        self.job = None

        # dictionary of available discipline options (with types)
        self.options_list = {}

        # create the available options and run any other initialization
        self.initialize()

    def add_option(self, name, type):
        """
        Adds an option definition to the discipline.

        Parameters
        ----------
        name : string
            the name of the option being added
        type : string
            the data type of the option. acceptable types are 'bool', 'int',
            'float', 'str', 'dict'
        """
        validate_name(name, "add_option")
        validate_option_type(type, name)
        if name in self.options_list:
            raise PhiloteValidationError(
                f"add_option: option '{name}' is already defined."
            )
        self.options_list[name] = type

    def add_input(self, name, shape=(1,), units="", dynamic_shape=False):
        """
        Define a continuous input.

        Parameters
        ----------
        name : string
            the name of the input variable
        shape : tuple
            the shape of the input variable (ignored when dynamic_shape
            is True)
        units : string
            the unit definition for the input variable
        dynamic_shape : bool
            when True, the client is allowed to set this variable's shape
        """
        validate_name(name, "add_input")
        if not dynamic_shape:
            validate_shape(shape, "add_input")
        validate_units(units, "add_input")
        key = (data.VariableType.kInput, name)
        if key in self._declared:
            raise PhiloteValidationError(
                f"add_input: input '{name}' is already defined."
            )
        meta = data.VariableMetaData()
        meta.type = data.VariableType.kInput
        meta.name = name
        if not dynamic_shape:
            meta.shape.extend(shape)
        meta.units = units
        meta.dynamic_shape = dynamic_shape
        self._var_meta += [meta]
        self._declared.add(key)

    def add_discrete_input(self, name, default=None):
        """
        Define a discrete input.

        Discrete inputs can hold any value that is representable as a
        ``google.protobuf.Value`` (scalars, lists, or nested dicts).

        Parameters
        ----------
        name : string
            the name of the discrete input variable
        default : object, optional
            the default value for the discrete input
        """
        validate_name(name, "add_discrete_input")
        key = (data.VariableType.kDiscreteInput, name)
        if key in self._declared:
            raise PhiloteValidationError(
                f"add_discrete_input: discrete input '{name}' is already defined."
            )
        meta = data.VariableMetaData()
        meta.type = data.VariableType.kDiscreteInput
        meta.name = name
        self._discrete_var_meta += [meta]
        self._declared.add(key)

    def add_discrete_output(self, name, default=None):
        """
        Define a discrete output.

        Discrete outputs can hold any value that is representable as a
        ``google.protobuf.Value`` (scalars, lists, or nested dicts).

        Parameters
        ----------
        name : string
            the name of the discrete output variable
        default : object, optional
            the default value for the discrete output
        """
        validate_name(name, "add_discrete_output")
        key = (data.VariableType.kDiscreteOutput, name)
        if key in self._declared:
            raise PhiloteValidationError(
                f"add_discrete_output: discrete output '{name}' is already defined."
            )
        meta = data.VariableMetaData()
        meta.type = data.VariableType.kDiscreteOutput
        meta.name = name
        self._discrete_var_meta += [meta]
        self._declared.add(key)

    def add_output(self, name, shape=(1,), units="", dynamic_shape=False):
        """
        Defines a continuous output.

        Parameters
        ----------
        name : string
            the name of the output variable
        shape : tuple
            the shape of the output variable (ignored when dynamic_shape
            is True)
        units : string
            the unit definition for the output variable
        dynamic_shape : bool
            when True, the client is allowed to set this variable's shape
        """
        validate_name(name, "add_output")
        if not dynamic_shape:
            validate_shape(shape, "add_output")
        validate_units(units, "add_output")
        key = (data.VariableType.kOutput, name)
        if key in self._declared:
            raise PhiloteValidationError(
                f"add_output: output '{name}' is already defined."
            )
        out_meta = data.VariableMetaData()
        out_meta.type = data.VariableType.kOutput
        out_meta.name = name
        if not dynamic_shape:
            out_meta.shape.extend(shape)
        out_meta.units = units
        out_meta.dynamic_shape = dynamic_shape
        self._var_meta += [out_meta]
        self._declared.add(key)

        if self._is_implicit:
            res_meta = data.VariableMetaData()
            res_meta.type = data.VariableType.kResidual
            res_meta.name = name
            if not dynamic_shape:
                res_meta.shape.extend(shape)
            res_meta.units = units
            res_meta.dynamic_shape = dynamic_shape
            self._var_meta += [res_meta]
            # the residual is a separate entry in the metadata list, so it
            # also has to be registered for the duplicate check to cover it
            self._declared.add((data.VariableType.kResidual, name))

    def declare_partials(self, func, var):
        """
        Defines partials that will be determined using the analysis server.
        """
        validate_name(func, "declare_partials (func)")
        validate_name(var, "declare_partials (var)")
        self._partials_meta += [data.PartialsMetaData(name=func, subname=var)]

    def initialize(self):
        """
        Sets up the available options.

        This function is called when the server is first started. It does not
        set options, but instead defines what option names (and types) are
        available. The set_options function is used to actually set the option
        values instead.
        """
        pass

    def set_options(self, options):
        """
        Sets the option values for the discipline.

        Parameters
        ----------
        options : DisciplineOptions
            options data structure (generated from the Philote MDO standard)
            that is used to set the discipline options. This data structure
            is received from the client and passed to this function.
        """
        pass

    def setup(self):
        """
        Sets up the discipline inputs and outputs.

        This function is called when the client invokes the Setup RPC. This
        function should be used to define inputs and outputs of the analysis
        discipline.
        """
        pass

    def setup_partials(self):
        """
        Sets up the discipline partials.

        This function is called when the client invokes the Setup RPC. This
        function should be used to define partial derivatives of the analysis
        discipline.
        """
        pass

    def configure(self):
        pass

    def teardown_job(self):
        """
        Releases whatever this instance holds, before its job is discarded.

        Called when the client ends the job and when the server evicts one
        that has gone idle, so a discipline that opened a file, started a
        subprocess, or built a solver should close it here. Overriding this
        is optional; the default does nothing.
        """
        pass

    def _clear_data(self):
        """
        Clears all metadata of the discipline.

        This function is invoked from the Setup function of the server.
        """
        self._var_meta = []
        self._discrete_var_meta = []
        self._partials_meta = []
        self._declared = set()
