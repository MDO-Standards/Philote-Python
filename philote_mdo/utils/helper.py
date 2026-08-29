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

from philote_mdo.utils.validation import PhiloteValidationError


def get_chunk_indices(num_values, chunk_size):
    if not isinstance(num_values, (int, np.integer)) or num_values < 0:
        raise PhiloteValidationError(
            f"get_chunk_indices: num_values must be a non-negative integer, "
            f"got {num_values!r}."
        )
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise PhiloteValidationError(
            f"get_chunk_indices: chunk_size must be a positive integer, "
            f"got {chunk_size!r}."
        )

    # the common case is a variable that fits in one chunk, where the answer
    # is known without building two arrays to derive it
    if 0 < num_values <= chunk_size:
        return iter([(0, num_values)])

    beg_i = np.arange(0, num_values, chunk_size)

    if beg_i.size == 1:
        end_i = [num_values]
    else:
        end_i = np.append(beg_i[1:], [num_values])

    return zip(beg_i, end_i)


def get_partials_shape(shapef, shapex):
    """
    Returns the shape of the Jacobian block of a function with respect to a
    variable.

    Note, there are edge cases for this function, where either f or x, or both
    are scalar. In those cases a (1,) dimension is dropped from the block
    shape instead of being carried through, so that the partial of a scalar
    with respect to a vector of length n is (n,) rather than (1, n).

    :param shapef: shape of the function, as a tuple
    :param shapex: shape of the variable, as a tuple
    :return: shape of the Jacobian block, as a tuple
    """
    if shapef == (1,):
        if shapex == (1,):
            return (1,)
        return shapex

    if shapex == (1,):
        return shapef

    return shapef + shapex


def get_flattened_view(arr):
    """
    Returns a flattened view of the input array. Used instead of reshape, ravel, flatten, etc. to guarante a copy is
    not made. If the input array does not support copy-free modification, AttributeError will be thrown
    :param arr: Array to get a flattened view
    :return: A view of the input array, guaranteed to not be a copy
    """
    if not isinstance(arr, np.ndarray):
        raise PhiloteValidationError(
            f"get_flattened_view: expected a numpy ndarray, "
            f"got {type(arr).__name__}."
        )
    flat_view = arr.view()
    flat_view.shape = -1
    return flat_view
