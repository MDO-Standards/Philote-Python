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
from philote_mdo.general import ExplicitDiscipline, ImplicitDiscipline


class TestAbstractMethods(unittest.TestCase):
    """
    Tests for abstract method implementations that should raise NotImplementedError.
    """

    def test_explicit_discipline_compute_not_implemented(self):
        """
        Test that ExplicitDiscipline.compute raises NotImplementedError.
        """
        discipline = ExplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.compute({}, {})
        
        self.assertEqual(str(context.exception), "compute not implemented")

    def test_explicit_discipline_compute_partials_not_implemented(self):
        """
        Test that ExplicitDiscipline.compute_partials raises NotImplementedError.
        """
        discipline = ExplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.compute_partials({}, {})
        
        self.assertEqual(str(context.exception), "compute_partials not implemented")

    def test_implicit_discipline_compute_residuals_not_implemented(self):
        """
        Test that ImplicitDiscipline.compute_residuals raises NotImplementedError.
        """
        discipline = ImplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.compute_residuals({}, {}, {})
        
        self.assertEqual(str(context.exception), "compute_residuals not implemented")

    def test_implicit_discipline_solve_residuals_not_implemented(self):
        """
        Test that ImplicitDiscipline.solve_residuals raises NotImplementedError.
        """
        discipline = ImplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.solve_residuals({}, {})
        
        self.assertEqual(str(context.exception), "solve_residuals not implemented")

    def test_implicit_discipline_residual_partials_not_implemented(self):
        """
        Test that ImplicitDiscipline.residual_partials raises NotImplementedError.
        """
        discipline = ImplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.residual_partials({}, {}, {})
        
        self.assertEqual(str(context.exception), "residual_partials not implemented")

    def test_implicit_discipline_apply_linear_not_implemented(self):
        """
        Test that ImplicitDiscipline.apply_linear raises NotImplementedError.
        """
        discipline = ImplicitDiscipline()
        
        with self.assertRaises(NotImplementedError) as context:
            discipline.apply_linear({}, {}, "fwd")
        
        self.assertEqual(str(context.exception), "apply_linear not implemented")


if __name__ == "__main__":
    unittest.main(verbosity=2)
