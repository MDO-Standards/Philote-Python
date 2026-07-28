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
from philote_mdo.utils.validation import PhiloteValidationError


_TYPE_MAP = {
    "bool": bool,
    "int": int,
    "float": float,
    "double": float,
    "str": str,
    "string": str,
    "dict": dict,
}


def declare_options(opt_list, options):
    """
    Declares the options from the client options list.
    """
    for name, type_str in opt_list:
        opt_type = _TYPE_MAP.get(type_str)
        if opt_type is None:
            raise PhiloteValidationError(
                f"declare_options: unknown option type '{type_str}' "
                f"for option '{name}'."
            )

        options.declare(name, types=opt_type)


def client_setup(comp):
    """
    Sets up the OpenMDAO component with all required inputs and outputs.

    This function will call the required RPCs to obtain the variables
    from the remote discipline server.  Both continuous and discrete
    variables are declared.
    """
    # query the discipline properties first, so that the client learns whether
    # the server supports the unary transport before any compute call is made
    comp._client.get_discipline_info()

    # set up the remote discipline and get the variable definitions
    comp._client.run_setup()
    comp._client.get_variable_definitions()

    # define continuous inputs and outputs based on the discipline metadata
    for var in comp._client._var_meta:
        if not var.units:
            units = None
        else:
            units = var.units

        if var.dynamic_shape:
            # let OpenMDAO resolve the shape from connections
            if var.type == data.kInput:
                comp.add_input(var.name, shape_by_conn=True, units=units)

            if var.type == data.kOutput:
                comp.add_output(var.name, shape_by_conn=True, units=units)
        else:
            if var.type == data.kInput:
                comp.add_input(var.name, shape=tuple(var.shape), units=units)

            if var.type == data.kOutput:
                comp.add_output(var.name, shape=tuple(var.shape), units=units)

    # define discrete inputs and outputs
    for var in comp._client._discrete_var_meta:
        if var.type == data.VariableType.kDiscreteInput:
            comp.add_discrete_input(var.name, val=None)

        if var.type == data.VariableType.kDiscreteOutput:
            comp.add_discrete_output(var.name, val=None)


def send_resolved_shapes(comp):
    """
    Sends resolved shapes for dynamic-shape variables back to the server.

    After OpenMDAO resolves shapes (e.g. via ``shape_by_conn``), this
    function reads the resolved metadata from the component and transmits
    the shapes to the remote discipline server.
    """
    dynamic_shapes = []
    for var in comp._client._var_meta:
        if not var.dynamic_shape:
            continue

        resolved_meta = comp._var_rel2meta[var.name]
        dynamic_shapes.append(
            comp._client.set_variable_shape(
                var.name, resolved_meta["shape"], var.type
            )
        )

    if dynamic_shapes:
        comp._client.send_variable_shapes(dynamic_shapes)


def client_setup_partials(comp):
    """
    Sets up the partials for the OpenMDAO component.

    This function will call the required RPCs to obtain the partials
    from the remote discipline server.
    """
    # set up the remote discipline and get the variable definitions
    comp._client.get_partials_definitions()

    # declare partials based on the discipline meta data
    for partial in comp._client._partials_meta:
        comp.declare_partials(partial.name, partial.subname)


def create_local_discrete_inputs(discrete_inputs, discrete_var_meta, type=data.VariableType.kDiscreteInput):
    """
    Creates a Philote-Python local discrete inputs dictionary from OpenMDAO
    discrete inputs.
    """
    if discrete_inputs is None:
        return None
    local = {}
    for var in discrete_var_meta:
        if var.type == type:
            local[var.name] = discrete_inputs[var.name]
    return local if local else None


def create_local_inputs(inputs, var_meta, type=data.kInput):
    """
    Creates a Philote-Python local inputs dictionary from OpenMDAO inputs.

    We need to assign a local input dictionary, as the openmdao Vector class
    returns the absolute variable name (including all parent system). The
    remote client is unaware of any of the parent systems, so the relative
    name of all variables is required.
    """
    local = {}
    for var in var_meta:
        if var.type == type:
            local[var.name] = inputs[var.name]

    return local


def assign_global_outputs(out, outputs):
    """
    Assigns a OpenMDAO outputs from a Philote-Python output/residual dictionary.

    note: merely assigning the outputs from the compute (etc.) function will
    overwrite the OpenMDAO outputs variable reference and therefore not work.
    """
    for key, val in out.items():
        outputs[key] = val
