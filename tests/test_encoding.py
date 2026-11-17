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
"""
Tests for the fast packed-double encoding helpers.

The central property, on which everything else rests, is that these helpers
produce and consume exactly the bytes the protobuf API would. If that holds,
the optimization is invisible to every peer.
"""
import unittest

import numpy as np

import philote_mdo.generated.data_pb2 as data
from philote_mdo.utils import (
    FAST_DECODE_MIN_ELEMENTS,
    get_array_data,
    read_array_into,
    set_array_data,
)
from philote_mdo.utils.encoding import _decode_varint, _encode_varint
from philote_mdo.utils.validation import PhiloteValidationError


def reference(values, **fields):
    """Builds an Array the ordinary way, through the protobuf API."""
    return data.Array(data=values, **fields)


def fast(values, **fields):
    """Builds the same Array through the fast path."""
    message = data.Array(**fields)
    set_array_data(message, values)
    return message


class TestVarint(unittest.TestCase):
    """
    Tests the base-128 varint codec used to frame the packed payload.
    """

    def test_round_trip(self):
        for value in (0, 1, 127, 128, 300, 16383, 16384, 1 << 20, 1 << 35):
            encoded = _encode_varint(value)
            decoded, index = _decode_varint(encoded, 0)
            self.assertEqual(decoded, value)
            self.assertEqual(index, len(encoded))

    def test_boundary_widths(self):
        self.assertEqual(len(_encode_varint(127)), 1)
        self.assertEqual(len(_encode_varint(128)), 2)
        self.assertEqual(len(_encode_varint(16383)), 2)
        self.assertEqual(len(_encode_varint(16384)), 3)


class TestByteIdentity(unittest.TestCase):
    """
    Tests that the fast encoder is indistinguishable on the wire.
    """

    def test_identical_bytes_across_sizes(self):
        """
        The whole approach rests on this: a packed repeated double and a raw
        little-endian buffer are the same bytes.
        """
        for size in (1, 2, 63, 64, 65, 1000, 100000):
            values = np.arange(size, dtype=np.float64) * 1.5 - 3.0
            slow = reference(values, name="v", start=0, end=size - 1)
            quick = fast(values, name="v", start=0, end=size - 1)

            with self.subTest(size=size):
                self.assertEqual(
                    quick.SerializeToString(), slow.SerializeToString()
                )

    def test_identical_bytes_with_all_metadata(self):
        values = np.array([1.25, -2.5, 3.75])
        fields = dict(
            name="f", subname="x", start=7, end=9, type=data.kPartial
        )

        self.assertEqual(
            fast(values, **fields).SerializeToString(),
            reference(values, **fields).SerializeToString(),
        )

    def test_fast_encoded_message_readable_by_protobuf_api(self):
        """
        A peer that knows nothing about this optimization must be able to read
        what we send.
        """
        values = np.linspace(-1.0, 1.0, 200)
        wire = fast(values, name="v", start=0, end=199).SerializeToString()

        parsed = data.Array.FromString(wire)

        self.assertTrue(np.array_equal(np.array(parsed.data), values))

    def test_protobuf_encoded_message_readable_by_fast_path(self):
        """
        The reverse: we must be able to read what an ordinary peer sends.
        """
        values = np.linspace(-1.0, 1.0, 200)
        wire = reference(values, name="v", start=0, end=199).SerializeToString()

        parsed = data.Array.FromString(wire)

        self.assertTrue(np.array_equal(get_array_data(parsed), values))


class TestDtypeCoercion(unittest.TestCase):
    """
    Tests the coercion that the protobuf API used to perform for us.

    Without it a float32 array would be written at half length, and an integer
    array would be written with its raw bit pattern reinterpreted as doubles.
    """

    def test_float32_is_widened(self):
        values = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        message = fast(values, name="v", start=0, end=2)

        self.assertEqual(len(message.data), 3)
        self.assertTrue(np.array_equal(np.array(message.data), [1.5, 2.5, 3.5]))

    def test_integers_are_converted_not_reinterpreted(self):
        values = np.array([1, 2, 3], dtype=np.int64)

        message = fast(values, name="v", start=0, end=2)

        self.assertTrue(np.array_equal(np.array(message.data), [1.0, 2.0, 3.0]))

    def test_matches_protobuf_coercion(self):
        for dtype in ("float32", "int32", "int64", "float64"):
            values = np.arange(5, dtype=dtype)

            with self.subTest(dtype=dtype):
                self.assertEqual(
                    fast(values, name="v", start=0, end=4).SerializeToString(),
                    reference(values, name="v", start=0, end=4).SerializeToString(),
                )

    def test_python_list_input(self):
        message = fast([1.0, 2.0, 3.0], name="v", start=0, end=2)

        self.assertTrue(np.array_equal(np.array(message.data), [1.0, 2.0, 3.0]))


class TestArrayLayout(unittest.TestCase):
    """
    Tests inputs whose memory layout is not a simple contiguous block.
    """

    def test_non_contiguous_slice(self):
        matrix = np.arange(12, dtype=np.float64).reshape(3, 4)
        strided = matrix[:, ::2]
        self.assertFalse(strided.flags["C_CONTIGUOUS"])

        message = fast(strided, name="v", start=0, end=5)

        self.assertTrue(
            np.array_equal(np.array(message.data), strided.ravel())
        )

    def test_multidimensional_input_is_flattened(self):
        matrix = np.arange(6, dtype=np.float64).reshape(2, 3)

        message = fast(matrix, name="v", start=0, end=5)

        self.assertTrue(np.array_equal(np.array(message.data), matrix.ravel()))

    def test_empty_array(self):
        message = fast(np.array([]), name="v", start=0, end=-1)

        self.assertEqual(len(message.data), 0)
        self.assertEqual(len(get_array_data(message)), 0)

    def test_get_array_data_on_message_without_data(self):
        message = data.Array(name="v", subname="s", start=3, end=9)

        self.assertEqual(len(get_array_data(message)), 0)


class TestReadArrayInto(unittest.TestCase):
    """
    Tests the thresholded scatter used by every decode site.
    """

    def _round_trip(self, size, start):
        values = np.arange(size, dtype=np.float64) + 0.5
        message = fast(values, name="v", start=start, end=start + size - 1)
        destination = np.zeros(start + size)

        read_array_into(message, destination)

        return values, destination

    def test_small_array_below_threshold(self):
        size = FAST_DECODE_MIN_ELEMENTS - 1
        values, destination = self._round_trip(size, 0)

        self.assertTrue(np.array_equal(destination, values))

    def test_large_array_above_threshold(self):
        size = FAST_DECODE_MIN_ELEMENTS + 1
        values, destination = self._round_trip(size, 0)

        self.assertTrue(np.array_equal(destination, values))

    def test_threshold_boundary_exactly(self):
        """
        Both branches must agree at the switchover point.
        """
        size = FAST_DECODE_MIN_ELEMENTS
        values, destination = self._round_trip(size, 0)

        self.assertTrue(np.array_equal(destination, values))

    def test_offset_chunk(self):
        """
        A chunk lands at its start index, not at the beginning.
        """
        for size in (4, FAST_DECODE_MIN_ELEMENTS + 4):
            with self.subTest(size=size):
                values, destination = self._round_trip(size, 10)

                self.assertTrue(np.array_equal(destination[10:], values))
                self.assertTrue(np.array_equal(destination[:10], np.zeros(10)))

    def test_chunks_reassemble_in_any_order(self):
        """
        Chunks carry their own indices, so arrival order must not matter.
        """
        total = 200
        values = np.arange(total, dtype=np.float64)
        bounds = [(0, 70), (70, 140), (140, total)]
        messages = [
            fast(values[b:e], name="v", start=b, end=e - 1) for b, e in bounds
        ]

        destination = np.zeros(total)
        for message in reversed(messages):
            read_array_into(message, destination)

        self.assertTrue(np.array_equal(destination, values))

    def test_reads_message_built_by_protobuf_api(self):
        for size in (4, FAST_DECODE_MIN_ELEMENTS + 4):
            values = np.arange(size, dtype=np.float64)
            message = reference(values, name="v", start=0, end=size - 1)
            destination = np.zeros(size)

            with self.subTest(size=size):
                read_array_into(message, destination)
                self.assertTrue(np.array_equal(destination, values))


class TestWireScanner(unittest.TestCase):
    """
    Tests the field scanner against wire shapes it must skip over.
    """

    def test_skips_preceding_fields_of_every_wire_type(self):
        """
        name and subname are length-delimited, start and end are varints, and
        type is a varint. The scanner must step over all of them.
        """
        values = np.arange(100, dtype=np.float64)
        message = fast(
            values,
            name="a_long_variable_name",
            subname="another_long_subname",
            start=1 << 20,
            end=(1 << 20) + 99,
            type=data.kPartial,
        )

        self.assertTrue(np.array_equal(get_array_data(message), values))

    def test_skips_fixed_width_fields(self):
        """
        Array has no fixed32 or fixed64 fields today, but the scanner must
        step over them so that it keeps working if the standard grows one.
        """

        class Stub:
            name = "v"

            def __init__(self, buffer):
                self._buffer = buffer

            def SerializeToString(self):
                return self._buffer

        values = np.arange(4, dtype=np.float64)
        payload = values.tobytes()
        buffer = (
            b"\x0d\x00\x00\x80\x3f"          # field 1, fixed32
            b"\x11" + b"\x00" * 8            # field 2, fixed64
            + b"\x32" + bytes([len(payload)]) + payload  # field 6, our data
        )

        self.assertTrue(np.array_equal(get_array_data(Stub(buffer)), values))

    def test_rejects_unknown_wire_type(self):
        """
        A malformed buffer must raise rather than silently return garbage.
        """

        class Stub:
            name = "v"

            def SerializeToString(self):
                # field 1 with the reserved wire type 3 (start-group)
                return b"\x0b\x00"

        with self.assertRaises(PhiloteValidationError):
            get_array_data(Stub())


if __name__ == "__main__":
    unittest.main()
