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
import unittest

import numpy as np

from philote_mdo.utils.validation import (
    PhiloteError,
    PhiloteValidationError,
    PhiloteServerError,
    validate_name,
    validate_shape,
    validate_units,
    validate_option_type,
    validate_is_dict,
    validate_numpy_array,
)


class TestExceptionHierarchy(unittest.TestCase):
    """Tests for the custom exception class hierarchy."""

    def test_philote_validation_error_is_value_error(self):
        with self.assertRaises(ValueError):
            raise PhiloteValidationError("test")

    def test_philote_validation_error_is_philote_error(self):
        with self.assertRaises(PhiloteError):
            raise PhiloteValidationError("test")

    def test_philote_server_error_is_runtime_error(self):
        with self.assertRaises(RuntimeError):
            raise PhiloteServerError("test")

    def test_philote_server_error_is_philote_error(self):
        with self.assertRaises(PhiloteError):
            raise PhiloteServerError("test")

    def test_philote_error_is_exception(self):
        with self.assertRaises(Exception):
            raise PhiloteError("test")


class TestValidateName(unittest.TestCase):
    """Tests for validate_name."""

    def test_valid_name(self):
        validate_name("x", "test")

    def test_non_string_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_name(123, "add_input")
        self.assertIn("must be a string", str(ctx.exception))
        self.assertIn("add_input", str(ctx.exception))

    def test_empty_string_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_name("", "add_output")
        self.assertIn("must not be empty", str(ctx.exception))

    def test_none_raises(self):
        with self.assertRaises(PhiloteValidationError):
            validate_name(None, "test")


class TestValidateShape(unittest.TestCase):
    """Tests for validate_shape."""

    def test_valid_shape_1d(self):
        validate_shape((3,), "test")

    def test_valid_shape_2d(self):
        validate_shape((2, 4), "test")

    def test_non_tuple_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_shape([2, 3], "add_input")
        self.assertIn("must be a tuple", str(ctx.exception))

    def test_int_raises(self):
        with self.assertRaises(PhiloteValidationError):
            validate_shape(3, "test")

    def test_non_integer_element_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_shape((2.0,), "test")
        self.assertIn("must be integers", str(ctx.exception))

    def test_zero_element_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_shape((0,), "test")
        self.assertIn("must be positive", str(ctx.exception))

    def test_negative_element_raises(self):
        with self.assertRaises(PhiloteValidationError):
            validate_shape((-1, 3), "test")


class TestValidateUnits(unittest.TestCase):
    """Tests for validate_units."""

    def test_valid_units(self):
        validate_units("m**2", "test")

    def test_empty_string_is_valid(self):
        validate_units("", "test")

    def test_non_string_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_units(42, "add_input")
        self.assertIn("must be a string", str(ctx.exception))


class TestValidateOptionType(unittest.TestCase):
    """Tests for validate_option_type."""

    def test_valid_types(self):
        for t in ("bool", "int", "float", "str", "dict"):
            validate_option_type(t, "opt")

    def test_invalid_type_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_option_type("unknown", "my_opt")
        self.assertIn("Invalid type", str(ctx.exception))
        self.assertIn("my_opt", str(ctx.exception))


class TestValidateIsDict(unittest.TestCase):
    """Tests for validate_is_dict."""

    def test_valid_dict(self):
        validate_is_dict({"a": 1}, "test")

    def test_non_dict_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_is_dict([1, 2], "send_options")
        self.assertIn("expected a dict", str(ctx.exception))


class TestValidateNumpyArray(unittest.TestCase):
    """Tests for validate_numpy_array."""

    def test_valid_array(self):
        validate_numpy_array(np.array([1.0, 2.0]), "x")

    def test_list_raises(self):
        with self.assertRaises(PhiloteValidationError) as ctx:
            validate_numpy_array([1.0, 2.0], "x")
        self.assertIn("must be a numpy ndarray", str(ctx.exception))

    def test_scalar_raises(self):
        with self.assertRaises(PhiloteValidationError):
            validate_numpy_array(1.0, "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
