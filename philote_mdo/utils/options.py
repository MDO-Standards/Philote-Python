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
from philote_mdo.utils.validation import PhiloteValidationError


# gRPC's own default maximum receive size, in bytes. A channel or server that
# is not given explicit options will reject anything larger than this.
DEFAULT_MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def _message_length_options(max_message_bytes):
    """
    Builds the send/receive message length option pairs shared by clients and
    servers.
    """
    if not isinstance(max_message_bytes, int) or isinstance(
        max_message_bytes, bool
    ) or max_message_bytes < 1:
        raise PhiloteValidationError(
            f"max_message_bytes must be a positive integer, "
            f"got {max_message_bytes!r}."
        )

    return [
        ("grpc.max_send_message_length", max_message_bytes),
        ("grpc.max_receive_message_length", max_message_bytes),
    ]


def channel_options(max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
    """
    Returns gRPC channel options sized for large Philote payloads.

    Pass the result to ``grpc.insecure_channel`` or ``grpc.secure_channel``.
    Raising a client's ``unary_max_bytes`` past
    ``DEFAULT_MAX_MESSAGE_BYTES`` has no effect unless the channel is built
    with these options and the server is built with :func:`server_options`.

    :param max_message_bytes: largest message the channel will send or receive
    :return: list of (option name, value) pairs
    """
    return _message_length_options(max_message_bytes)


def server_options(max_message_bytes=DEFAULT_MAX_MESSAGE_BYTES):
    """
    Returns gRPC server options sized for large Philote payloads.

    Pass the result to ``grpc.server`` as the ``options`` keyword argument.

    :param max_message_bytes: largest message the server will send or receive
    :return: list of (option name, value) pairs
    """
    return _message_length_options(max_message_bytes)
