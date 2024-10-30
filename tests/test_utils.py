# Philote-Python
#
# Copyright 2022-2024 Christopher A. Lupp
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
from philote_mdo.utils import get_chunk_indices, get_flattened_view


class TestUtils(unittest.TestCase):
    """
    Tests the utility functions used by the individual servers and clients.
    """

    def test_get_chunk_indices(self):
        """
        Tests the chunking algorithm.
        """
        # test case 1: multiple chunks
        num_values = 10
        chunk_size = 3
        result = list(get_chunk_indices(num_values, chunk_size))
        self.assertEqual(result, [(0, 3), (3, 6), (6, 9), (9, 10)])

        # test case 2: single chunk
        num_values = 5
        chunk_size = 10
        result = list(get_chunk_indices(num_values, chunk_size))
        self.assertEqual(result, [(0, 5)])

    def test_get_flattened_view(self):
        """
        Tests the function that returns the flattened view of an array.
        """
        # Test case 1: 1D array
        input_array = np.array([1, 2, 3])
        result = get_flattened_view(input_array)
        self.assertIs(result.base, input_array)
        self.assertEqual(result.shape, (3,))

        # Test case 2: 2D array
        input_array_2d = np.array([[1, 2], [3, 4]])
        result_2d = get_flattened_view(input_array_2d)
        self.assertIs(result_2d.base, input_array_2d)
        self.assertEqual(result_2d.shape, (4,))

        # Test case 3: Empty array
        empty_array = np.array([])
        result_empty = get_flattened_view(empty_array)
        self.assertIs(result_empty.base, empty_array)
        self.assertEqual(result_empty.shape, (0,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
