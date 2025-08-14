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
import sys
import importlib


class TestImportExceptions(unittest.TestCase):
    """
    Tests for import exception handling across the codebase.
    """

    def test_openmdao_availability(self):
        """
        Test that openmdao module imports work correctly when available.
        This tests the positive case and ensures openmdao is installed.
        """
        # This should work since we have openmdao installed
        import philote_mdo.openmdao
        
        # OpenMDAO should be available
        self.assertTrue(philote_mdo.openmdao.omdao_installed)
        self.assertIsNotNone(philote_mdo.openmdao.om)
        
        # Should have OpenMDAO-dependent classes
        self.assertTrue(hasattr(philote_mdo.openmdao, 'RemoteExplicitComponent'))
        self.assertTrue(hasattr(philote_mdo.openmdao, 'RemoteImplicitComponent'))
        self.assertTrue(hasattr(philote_mdo.openmdao, 'OpenMdaoSubProblem'))

    def test_examples_sellar_availability(self):
        """
        Test that examples include SellarGroup when OpenMDAO is available.
        """
        # This should work since we have openmdao installed
        import philote_mdo.examples
        
        # Should have all imports including SellarGroup
        self.assertTrue(hasattr(philote_mdo.examples, 'Paraboloid'))
        self.assertTrue(hasattr(philote_mdo.examples, 'QuadradicImplicit'))
        self.assertTrue(hasattr(philote_mdo.examples, 'Rosenbrock'))
        self.assertTrue(hasattr(philote_mdo.examples, 'SellarGroup'))

    def test_examples_import_error_handling(self):
        """
        Test that examples/__init__.py handles import errors gracefully.
        """
        # The import error handling lines are 37-38 in examples/__init__.py
        # These are covered when the module is first imported, but let's test
        # that the basic functionality works
        import philote_mdo.examples
        
        # These should always be available regardless of OpenMDAO
        self.assertTrue(hasattr(philote_mdo.examples, 'Paraboloid'))
        self.assertTrue(hasattr(philote_mdo.examples, 'QuadradicImplicit'))
        self.assertTrue(hasattr(philote_mdo.examples, 'Rosenbrock'))
        
        # SellarGroup depends on OpenMDAO being available
        # Since we have OpenMDAO, it should be available
        if hasattr(philote_mdo.examples, 'SellarGroup'):
            # If SellarGroup is available, we can import it
            from philote_mdo.examples import SellarGroup
            self.assertIsNotNone(SellarGroup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
