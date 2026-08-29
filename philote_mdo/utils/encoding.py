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
Fast paths for moving continuous array data in and out of ``Array`` messages.

The protobuf Python API converts every element of a ``repeated double`` field
to and from a boxed Python float, which for bulk arrays costs far more than the
transport itself -- around 36 ms per direction for 200k doubles, against about
2 ms for the gRPC round trip that carries them.

None of that work is required by the format. A packed ``repeated double`` is
encoded on the wire as a tag, a length, and a contiguous buffer of
little-endian doubles, which is byte for byte what NumPy already holds. The
helpers here read and write that buffer directly, producing output that is
identical to what the protobuf API produces. Nothing about the protocol
changes, and peers in any language are unaffected.
"""
import numpy as np

from philote_mdo.utils.validation import PhiloteValidationError


# Field number of Array.data in the Philote standard, and the protobuf tag byte
# that introduces it: (6 << 3) | 2, meaning field 6, length-delimited.
_DATA_FIELD = 6
_DATA_TAG = bytes([(_DATA_FIELD << 3) | 2])

# The representation the wire format specifies for continuous data. Casting
# through this pins the byte order, which matters on a big-endian host, and
# coerces integer or single-precision input, which the protobuf API would
# otherwise have converted for us. For the usual case of a float64 array on a
# little-endian host, the cast returns the original array untouched.
_DTYPE = np.dtype("<f8")

# Reading through the protobuf container is cheaper than re-serializing the
# message to reach its packed payload until an array is a few dozen elements
# long. The measured crossover is near 64, and the curve is flat on both sides,
# so the exact value is not sensitive.
FAST_DECODE_MIN_ELEMENTS = 64


def _encode_varint(value):
    """
    Encodes a non-negative integer as a protobuf base-128 varint.

    :param value: integer to encode
    :return: the encoded bytes
    """
    out = bytearray()

    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))

        if not value:
            return bytes(out)


def _decode_varint(buffer, index):
    """
    Decodes a protobuf base-128 varint.

    :param buffer: bytes to read from
    :param index: offset to start reading at
    :return: tuple of the decoded value and the offset just past it
    """
    result = 0
    shift = 0

    while True:
        byte = buffer[index]
        index += 1
        result |= (byte & 0x7F) << shift

        if not byte & 0x80:
            return result, index

        shift += 7


def set_array_data(message, values):
    """
    Fills an ``Array`` message's data field from a NumPy array.

    The result is byte for byte what ``Array(data=values)`` would produce, but
    without boxing each element as a Python float.

    :param message: the Array message to fill, which must not already have data
    :param values: array of values, of any shape or numeric dtype
    """
    raw = np.asarray(values, dtype=_DTYPE).tobytes()
    message.MergeFromString(_DATA_TAG + _encode_varint(len(raw)) + raw)


def get_array_data(message):
    """
    Returns an ``Array`` message's data field as a NumPy array.

    The message is re-serialized so that its packed payload can be read
    directly. That sounds wasteful but is not: protobuf holds packed doubles
    contiguously, so both the re-serialization and the read are memory copies.

    The returned array is a read-only view aliasing a temporary buffer, so
    callers must copy out of it rather than retain it.

    :param message: the Array message to read
    :return: the values, as a read-only numpy array
    """
    buffer = message.SerializeToString()
    index = 0
    size = len(buffer)

    while index < size:
        tag, index = _decode_varint(buffer, index)
        field = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 2:
            length, index = _decode_varint(buffer, index)

            if field == _DATA_FIELD:
                return np.frombuffer(
                    buffer, _DTYPE, count=length // 8, offset=index
                )

            index += length
        elif wire_type == 0:
            _, index = _decode_varint(buffer, index)
        elif wire_type == 1:
            index += 8
        elif wire_type == 5:
            index += 4
        else:
            raise PhiloteValidationError(
                "Unsupported protobuf wire type %d while decoding the data "
                "for variable '%s'." % (wire_type, message.name)
            )

    return np.empty(0, dtype=_DTYPE)


def read_array_into(message, destination):
    """
    Copies an ``Array`` message's data into a flat destination array.

    The message carries its own start and end indices, and ``end`` is
    inclusive, per the standard.

    Small arrays are read through the protobuf container, which costs less than
    re-serializing the message to reach the packed payload. See
    ``FAST_DECODE_MIN_ELEMENTS``.

    :param message: the Array message to read
    :param destination: flat numpy array to copy into
    """
    begin = message.start
    end = message.end + 1

    if end - begin >= FAST_DECODE_MIN_ELEMENTS:
        destination[begin:end] = get_array_data(message)
    else:
        destination[begin:end] = message.data
