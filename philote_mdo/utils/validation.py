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


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------


class PhiloteError(Exception):
    """Base class for all Philote-specific errors."""

    pass


class PhiloteValidationError(PhiloteError, ValueError):
    """Raised when an input fails validation.

    Inherits from ``ValueError`` so that existing ``except ValueError``
    handlers in user code continue to work.
    """

    pass


class PhiloteServerError(PhiloteError, RuntimeError):
    """Raised on the client side when a gRPC server call fails.

    Wraps the gRPC error details into a framework-specific exception so
    that users do not need to catch ``grpc.RpcError`` directly.
    """

    pass


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

VALID_OPTION_TYPES = {"bool", "int", "float", "str", "dict"}


def validate_name(name, context):
    """Validate that *name* is a non-empty string.

    Parameters
    ----------
    name : object
        The name to validate.
    context : str
        Human-readable context for error messages (e.g. ``"add_input"``).

    Raises
    ------
    PhiloteValidationError
        If *name* is not a string or is empty.
    """
    if not isinstance(name, str):
        raise PhiloteValidationError(
            f"{context}: 'name' must be a string, got {type(name).__name__}."
        )
    if not name:
        raise PhiloteValidationError(f"{context}: 'name' must not be empty.")


def validate_shape(shape, context):
    """Validate that *shape* is a tuple of positive integers.

    Parameters
    ----------
    shape : object
        The shape to validate.
    context : str
        Human-readable context for error messages.

    Raises
    ------
    PhiloteValidationError
        If *shape* is not a tuple, or contains non-positive / non-integer
        elements.
    """
    if not isinstance(shape, tuple):
        raise PhiloteValidationError(
            f"{context}: 'shape' must be a tuple, got {type(shape).__name__}."
        )
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise PhiloteValidationError(
                f"{context}: all elements of 'shape' must be integers, "
                f"but element {i} is {type(dim).__name__}."
            )
        if dim <= 0:
            raise PhiloteValidationError(
                f"{context}: all elements of 'shape' must be positive, "
                f"but element {i} is {dim}."
            )


def validate_units(units, context):
    """Validate that *units* is a string.

    Parameters
    ----------
    units : object
        The units string to validate.
    context : str
        Human-readable context for error messages.

    Raises
    ------
    PhiloteValidationError
        If *units* is not a string.
    """
    if not isinstance(units, str):
        raise PhiloteValidationError(
            f"{context}: 'units' must be a string, got {type(units).__name__}."
        )


def validate_option_type(type_str, name):
    """Validate that *type_str* is one of the allowed option types.

    Parameters
    ----------
    type_str : object
        The option type string to validate.
    name : str
        The option name (for error messages).

    Raises
    ------
    PhiloteValidationError
        If *type_str* is not in the allowed set.
    """
    if type_str not in VALID_OPTION_TYPES:
        raise PhiloteValidationError(
            f"Invalid type '{type_str}' for option '{name}'. "
            f"Allowed types are: {sorted(VALID_OPTION_TYPES)}."
        )


def validate_is_dict(obj, context):
    """Validate that *obj* is a dictionary.

    Parameters
    ----------
    obj : object
        The object to validate.
    context : str
        Human-readable context for error messages.

    Raises
    ------
    PhiloteValidationError
        If *obj* is not a ``dict``.
    """
    if not isinstance(obj, dict):
        raise PhiloteValidationError(
            f"{context}: expected a dict, got {type(obj).__name__}."
        )


def validate_numpy_array(value, name):
    """Validate that *value* is a NumPy ndarray.

    Parameters
    ----------
    value : object
        The value to validate.
    name : str
        Variable name (for error messages).

    Raises
    ------
    PhiloteValidationError
        If *value* is not a ``numpy.ndarray``.
    """
    if not isinstance(value, np.ndarray):
        raise PhiloteValidationError(
            f"Variable '{name}' must be a numpy ndarray, "
            f"got {type(value).__name__}."
        )
