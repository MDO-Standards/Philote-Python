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


class PairDict(dict):
    """
    Jacobian dictionary for storing values with respect to two keys.
    """

    @staticmethod
    def _validate_key(keys):
        if not isinstance(keys, tuple) or len(keys) != 2:
            raise PhiloteValidationError(
                f"PairDict keys must be a 2-tuple, got {keys!r}."
            )

    def __setitem__(self, keys, value):
        self._validate_key(keys)
        key1, key2 = keys
        super().__setitem__((key1, key2), value)

    def __getitem__(self, keys):
        self._validate_key(keys)
        key1, key2 = keys
        return super().__getitem__((key1, key2))
