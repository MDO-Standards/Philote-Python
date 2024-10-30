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
from scipy.optimize import rosen, rosen_der
import philote_mdo.utils as utils
from philote_mdo.examples import Rosenbrock
import philote_mdo.generated.data_pb2 as data


class TestRosenbrock(unittest.TestCase):
    """
    Unit tests for the Rosenbrock discipline.
    """

    def test_setup(self):
        """
        Tests the setup function of the Rosenbrock discipline.
        """
        disc = Rosenbrock()
        disc.set_options({"dimension": 2})
        disc.setup()

        self.assertEqual(disc._var_meta[0].name, "x")
        self.assertEqual(disc._var_meta[1].name, "f")

        self.assertEqual(disc._var_meta[0].type, data.kInput)
        self.assertEqual(disc._var_meta[1].type, data.kOutput)

        self.assertEqual(disc._var_meta[0].shape, [2])
        self.assertEqual(disc._var_meta[1].shape, [1])

    def test_setup_partials(self):
        """
        Tests the setup function of the Rosenbrock server.
        """
        disc = Rosenbrock()
        disc.setup_partials()

        self.assertEqual(disc._partials_meta[0].name, "f")
        self.assertEqual(disc._partials_meta[0].subname, "x")

    def test_compute(self):
        """
        Tests the compute function of the Rosenbrock server.
        """
        inputs = {"x": np.array([1.0, 2.0])}
        outputs = {"f": 0.0}
        disc = Rosenbrock()
        disc.compute(inputs, outputs)

        self.assertEqual(outputs["f"], rosen(inputs["x"]))

    def test_compute_partials(self):
        """
        Tests the compute_partials function of the Rosenbrock server.
        """
        inputs = {"x": np.array([1.0, 2.0])}
        jac = utils.PairDict()
        jac["f", "x"] = 0.0
        disc = Rosenbrock()
        disc.compute_partials(inputs, jac)

        np.testing.assert_array_equal(jac["f", "x"], rosen_der(inputs["x"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
