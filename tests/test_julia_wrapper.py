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
import os
import unittest
import numpy as np

try:
    from philote_mdo.wrappers.julia import JuliaWrapperDiscipline, JuliaImplicitWrapperDiscipline
    HAS_JULIACALL = True
except ImportError:
    HAS_JULIACALL = False


@unittest.skipIf(not HAS_JULIACALL, "juliacall not installed")
class JuliaWrapperTests(unittest.TestCase):
    """
    Unit tests for JuliaWrapperDiscipline (explicit disciplines).
    """

    @classmethod
    def setUpClass(cls):
        """Set up paths to example Julia files."""
        # Get the project root directory
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tests_dir)
        examples_dir = os.path.join(project_root, "examples", "julia")

        cls.paraboloid_file = os.path.join(examples_dir, "paraboloid.jl")
        cls.quadratic_file = os.path.join(examples_dir, "quadratic.jl")

    def test_initialization_successful(self):
        """
        Test successful initialization of JuliaWrapperDiscipline.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        self.assertIsNotNone(discipline)

    def test_initialization_with_options(self):
        """
        Test initialization with options.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline",
            options={"scale_factor": 2.0, "offset": 10.0}
        )
        self.assertIsNotNone(discipline)

    def test_initialization_file_not_found(self):
        """
        Test that FileNotFoundError is raised for missing Julia file.
        """
        with self.assertRaises(FileNotFoundError):
            JuliaWrapperDiscipline(
                julia_file="/nonexistent/file.jl",
                julia_type="SomeDiscipline"
            )

    def test_initialization_invalid_type(self):
        """
        Test that ValueError is raised for invalid Julia type.
        """
        with self.assertRaises(ValueError):
            JuliaWrapperDiscipline(
                julia_file=self.paraboloid_file,
                julia_type="NonExistentType"
            )

    def test_setup_metadata(self):
        """
        Test that setup correctly extracts metadata from Julia discipline.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        discipline.setup()

        # Check inputs
        self.assertIn("x", discipline._inputs)
        self.assertIn("y", discipline._inputs)

        # Check outputs
        self.assertIn("f_xy", discipline._outputs)

        # Check partials
        self.assertIn(("f_xy", "x"), discipline._partials)
        self.assertIn(("f_xy", "y"), discipline._partials)

    def test_compute_basic(self):
        """
        Test basic compute functionality.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        discipline.setup()

        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
        outputs = {"f_xy": np.zeros(1)}

        discipline.compute(inputs, outputs)

        self.assertEqual(outputs["f_xy"][0], 39.0)

    def test_compute_different_inputs(self):
        """
        Test compute with different input values.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        discipline.setup()

        inputs = {"x": np.array([2.0]), "y": np.array([3.0])}
        outputs = {"f_xy": np.zeros(1)}

        discipline.compute(inputs, outputs)

        self.assertEqual(outputs["f_xy"][0], 53.0)

    def test_compute_with_options(self):
        """
        Test compute with custom options.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline",
            options={"scale_factor": 2.0, "offset": 10.0}
        )
        discipline.setup()

        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
        outputs = {"f_xy": np.zeros(1)}

        discipline.compute(inputs, outputs)

        # f = 2.0 * 39.0 + 10.0 = 88.0
        self.assertEqual(outputs["f_xy"][0], 88.0)

    def test_compute_partials_basic(self):
        """
        Test basic compute_partials functionality.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        discipline.setup()

        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
        partials = {
            ("f_xy", "x"): np.zeros(1),
            ("f_xy", "y"): np.zeros(1)
        }

        discipline.compute_partials(inputs, partials)

        self.assertEqual(partials[("f_xy", "x")][0], -2.0)
        self.assertEqual(partials[("f_xy", "y")][0], 13.0)

    def test_compute_partials_different_inputs(self):
        """
        Test compute_partials with different input values.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline"
        )
        discipline.setup()

        inputs = {"x": np.array([2.0]), "y": np.array([3.0])}
        partials = {
            ("f_xy", "x"): np.zeros(1),
            ("f_xy", "y"): np.zeros(1)
        }

        discipline.compute_partials(inputs, partials)

        self.assertEqual(partials[("f_xy", "x")][0], 1.0)
        self.assertEqual(partials[("f_xy", "y")][0], 16.0)

    def test_compute_partials_with_options(self):
        """
        Test compute_partials with custom options.
        """
        discipline = JuliaWrapperDiscipline(
            julia_file=self.paraboloid_file,
            julia_type="ParaboloidDiscipline",
            options={"scale_factor": 2.0, "offset": 10.0}
        )
        discipline.setup()

        inputs = {"x": np.array([1.0]), "y": np.array([2.0])}
        partials = {
            ("f_xy", "x"): np.zeros(1),
            ("f_xy", "y"): np.zeros(1)
        }

        discipline.compute_partials(inputs, partials)

        # Partials are scaled by scale_factor
        self.assertEqual(partials[("f_xy", "x")][0], -4.0)
        self.assertEqual(partials[("f_xy", "y")][0], 26.0)


@unittest.skipIf(not HAS_JULIACALL, "juliacall not installed")
class JuliaImplicitWrapperTests(unittest.TestCase):
    """
    Unit tests for JuliaImplicitWrapperDiscipline (implicit disciplines).
    """

    @classmethod
    def setUpClass(cls):
        """Set up paths to example Julia files."""
        # Get the project root directory
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tests_dir)
        examples_dir = os.path.join(project_root, "examples", "julia")

        cls.quadratic_file = os.path.join(examples_dir, "quadratic.jl")

    def test_initialization_successful(self):
        """
        Test successful initialization of JuliaImplicitWrapperDiscipline.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        self.assertIsNotNone(discipline)

    def test_initialization_file_not_found(self):
        """
        Test that FileNotFoundError is raised for missing Julia file.
        """
        with self.assertRaises(FileNotFoundError):
            JuliaImplicitWrapperDiscipline(
                julia_file="/nonexistent/file.jl",
                julia_type="SomeDiscipline"
            )

    def test_initialization_invalid_type(self):
        """
        Test that ValueError is raised for invalid Julia type.
        """
        with self.assertRaises(ValueError):
            JuliaImplicitWrapperDiscipline(
                julia_file=self.quadratic_file,
                julia_type="NonExistentType"
            )

    def test_setup_metadata(self):
        """
        Test that setup correctly extracts metadata from Julia discipline.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        # Check inputs
        self.assertIn("a", discipline._inputs)
        self.assertIn("b", discipline._inputs)
        self.assertIn("c", discipline._inputs)

        # Check outputs
        self.assertIn("x", discipline._outputs)

        # Check residuals
        self.assertIn("x", discipline._residuals)

        # Check partials
        self.assertIn(("x", "a"), discipline._partials)
        self.assertIn(("x", "b"), discipline._partials)
        self.assertIn(("x", "c"), discipline._partials)
        self.assertIn(("x", "x"), discipline._partials)

    def test_compute_residuals_basic(self):
        """
        Test basic compute_residuals functionality.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        residuals = {"x": np.zeros(1)}

        discipline.compute_residuals(inputs, outputs, residuals)

        # r = a*x^2 + b*x + c = 1*16 + 2*4 + (-2) = 22
        self.assertEqual(residuals["x"][0], 22.0)

    def test_compute_residuals_zero(self):
        """
        Test compute_residuals at solution point (residual should be near zero).
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([0.73205081])}
        residuals = {"x": np.zeros(1)}

        discipline.compute_residuals(inputs, outputs, residuals)

        # Should be very close to zero at the solution
        self.assertAlmostEqual(residuals["x"][0], 0.0, places=6)

    def test_solve_residuals_basic(self):
        """
        Test basic solve_residuals functionality.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.zeros(1)}

        discipline.solve_residuals(inputs, outputs)

        # Solution: x = (-b + sqrt(b^2 - 4ac)) / 2a = (-2 + sqrt(4+8)) / 2 = 0.73205081
        self.assertAlmostEqual(outputs["x"][0], 0.73205081, places=8)

    def test_solve_residuals_different_inputs(self):
        """
        Test solve_residuals with different input values.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([1.0]), "b": np.array([0.0]), "c": np.array([-4.0])}
        outputs = {"x": np.zeros(1)}

        discipline.solve_residuals(inputs, outputs)

        # Solution: x = (-0 + sqrt(0+16)) / 2 = 2.0
        self.assertAlmostEqual(outputs["x"][0], 2.0, places=8)

    def test_residual_partials_basic(self):
        """
        Test basic residual_partials functionality.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([-2.0])}
        outputs = {"x": np.array([4.0])}
        partials = {
            ("x", "a"): np.zeros(1),
            ("x", "b"): np.zeros(1),
            ("x", "c"): np.zeros(1),
            ("x", "x"): np.zeros(1)
        }

        discipline.residual_partials(inputs, outputs, partials)

        # ∂r/∂a = x^2 = 16
        self.assertEqual(partials[("x", "a")][0], 16.0)
        # ∂r/∂b = x = 4
        self.assertEqual(partials[("x", "b")][0], 4.0)
        # ∂r/∂c = 1
        self.assertEqual(partials[("x", "c")][0], 1.0)
        # ∂r/∂x = 2*a*x + b = 2*1*4 + 2 = 10
        self.assertEqual(partials[("x", "x")][0], 10.0)

    def test_residual_partials_different_point(self):
        """
        Test residual_partials at different evaluation point.
        """
        discipline = JuliaImplicitWrapperDiscipline(
            julia_file=self.quadratic_file,
            julia_type="QuadraticDiscipline"
        )
        discipline.setup()

        inputs = {"a": np.array([2.0]), "b": np.array([1.0]), "c": np.array([-3.0])}
        outputs = {"x": np.array([3.0])}
        partials = {
            ("x", "a"): np.zeros(1),
            ("x", "b"): np.zeros(1),
            ("x", "c"): np.zeros(1),
            ("x", "x"): np.zeros(1)
        }

        discipline.residual_partials(inputs, outputs, partials)

        # ∂r/∂a = x^2 = 9
        self.assertEqual(partials[("x", "a")][0], 9.0)
        # ∂r/∂b = x = 3
        self.assertEqual(partials[("x", "b")][0], 3.0)
        # ∂r/∂c = 1
        self.assertEqual(partials[("x", "c")][0], 1.0)
        # ∂r/∂x = 2*a*x + b = 2*2*3 + 1 = 13
        self.assertEqual(partials[("x", "x")][0], 13.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
