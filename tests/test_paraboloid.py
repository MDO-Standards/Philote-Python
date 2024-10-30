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
import philote_mdo.utils as utils
from philote_mdo.examples import Paraboloid
import philote_mdo.generated.data_pb2 as data


class TestParaboloid(unittest.TestCase):
    """
    Unit tests for the paraboloid discipline.
    """

    def test_setup(self):
        """
        Tests the setup function of the Paraboloid discipline.
        """
        disc = Paraboloid()
        disc.setup()

        self.assertEqual(disc._var_meta[0].name, "x")
        self.assertEqual(disc._var_meta[1].name, "y")
        self.assertEqual(disc._var_meta[2].name, "f_xy")

        self.assertEqual(disc._var_meta[0].type, data.kInput)
        self.assertEqual(disc._var_meta[1].type, data.kInput)
        self.assertEqual(disc._var_meta[2].type, data.kOutput)

        self.assertEqual(disc._var_meta[0].shape, [1])
        self.assertEqual(disc._var_meta[1].shape, [1])
        self.assertEqual(disc._var_meta[2].shape, [1])

    def test_setup_partials(self):
        """
        Tests the setup function of the Paraboloid discipline.
        """
        disc = Paraboloid()
        disc.setup_partials()

        self.assertEqual(disc._partials_meta[0].name, "f_xy")
        self.assertEqual(disc._partials_meta[0].subname, "x")

        self.assertEqual(disc._partials_meta[1].name, "f_xy")
        self.assertEqual(disc._partials_meta[1].subname, "y")

    def test_compute(self):
        """
        Tests the compute function of the Paraboloid discipline.
        """
        inputs = {"x": 2.0, "y": 3.0}
        outputs = {"f_xy": 0.0}
        disc = Paraboloid()
        disc.compute(inputs, outputs)

        self.assertEqual(outputs["f_xy"], 53.0)

    def test_compute_partials(self):
        """
        Tests the compute function of the Paraboloid discipline.
        """
        inputs = {"x": 2.0, "y": 3.0}
        jac = utils.PairDict()
        jac["f_xy", "x"] = 0.0
        jac["f_xy", "y"] = 0.0
        disc = Paraboloid()
        disc.compute_partials(inputs, jac)

        self.assertEqual(jac["f_xy", "x"], 1.0)
        self.assertEqual(jac["f_xy", "y"], 16.0)
