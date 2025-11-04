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
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

try:
    from philote_mdo.wrappers.julia import cli
    HAS_JULIA_CLI = True
except ImportError:
    HAS_JULIA_CLI = False


@unittest.skipIf(not HAS_JULIA_CLI, "Julia CLI module not available")
class JuliaCLITests(unittest.TestCase):
    """
    Unit tests for the Julia CLI.
    """

    @classmethod
    def setUpClass(cls):
        """Set up paths to example config files."""
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tests_dir)
        configs_dir = os.path.join(project_root, "examples", "julia", "configs")

        cls.paraboloid_config = os.path.join(configs_dir, "paraboloid.yaml")
        cls.quadratic_config = os.path.join(configs_dir, "quadratic.yaml")

    def test_missing_config_file(self):
        """Test that missing config file exits with error."""
        with patch('sys.argv', ['philote-julia-serve', '/nonexistent/config.yaml']):
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
                self.assertEqual(cm.exception.code, 1)
                stderr_output = mock_stderr.getvalue()
                self.assertIn("Error:", stderr_output)
                self.assertIn("not found", stderr_output.lower())

    def test_invalid_yaml_structure(self):
        """Test that invalid YAML structure exits with error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("just a string, not a dict")
            temp_path = f.name

        try:
            with patch('sys.argv', ['philote-julia-serve', temp_path]):
                with patch('sys.stderr', new=StringIO()) as mock_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        cli.main()
                    self.assertEqual(cm.exception.code, 1)
                    stderr_output = mock_stderr.getvalue()
                    self.assertIn("error", stderr_output.lower())
        finally:
            os.unlink(temp_path)

    def test_missing_discipline_section(self):
        """Test that missing discipline section exits with error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("server:\n  address: '[::]:50051'\n")
            temp_path = f.name

        try:
            with patch('sys.argv', ['philote-julia-serve', temp_path]):
                with patch('sys.stderr', new=StringIO()) as mock_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        cli.main()
                    self.assertEqual(cm.exception.code, 1)
                    stderr_output = mock_stderr.getvalue()
                    self.assertIn("Configuration error", stderr_output)
                    self.assertIn("discipline", stderr_output.lower())
        finally:
            os.unlink(temp_path)

    @patch('philote_mdo.wrappers.julia.cli.serve_explicit_discipline')
    def test_route_to_explicit_server(self, mock_serve_explicit):
        """Test that explicit discipline routes to explicit server."""
        with patch('sys.argv', ['philote-julia-serve', self.paraboloid_config]):
            # Mock the serve function to avoid actually starting a server
            mock_serve_explicit.return_value = None

            cli.main()

            # Verify that serve_explicit_discipline was called
            mock_serve_explicit.assert_called_once()
            config = mock_serve_explicit.call_args[0][0]
            self.assertEqual(config.discipline.kind, "explicit")

    @patch('philote_mdo.wrappers.julia.cli.serve_implicit_discipline')
    def test_route_to_implicit_server(self, mock_serve_implicit):
        """Test that implicit discipline routes to implicit server."""
        with patch('sys.argv', ['philote-julia-serve', self.quadratic_config]):
            # Mock the serve function to avoid actually starting a server
            mock_serve_implicit.return_value = None

            cli.main()

            # Verify that serve_implicit_discipline was called
            mock_serve_implicit.assert_called_once()
            config = mock_serve_implicit.call_args[0][0]
            self.assertEqual(config.discipline.kind, "implicit")

    def test_version_flag(self):
        """Test that --version flag displays version and exits."""
        with patch('sys.argv', ['philote-julia-serve', '--version']):
            with patch('sys.stdout', new=StringIO()) as mock_stdout:
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
                # argparse exits with code 0 for --version
                self.assertEqual(cm.exception.code, 0)
                stdout_output = mock_stdout.getvalue()
                self.assertIn("0.1.0", stdout_output)

    def test_help_flag(self):
        """Test that --help flag displays help and exits."""
        with patch('sys.argv', ['philote-julia-serve', '--help']):
            with patch('sys.stdout', new=StringIO()) as mock_stdout:
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
                # argparse exits with code 0 for --help
                self.assertEqual(cm.exception.code, 0)
                stdout_output = mock_stdout.getvalue()
                self.assertIn("Serve Julia Philote disciplines", stdout_output)
                self.assertIn("Examples:", stdout_output)

    def test_no_arguments(self):
        """Test that running without arguments shows error."""
        with patch('sys.argv', ['philote-julia-serve']):
            with patch('sys.stderr', new=StringIO()) as mock_stderr:
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
                # argparse exits with code 2 for missing required arguments
                self.assertEqual(cm.exception.code, 2)
                stderr_output = mock_stderr.getvalue()
                self.assertIn("required", stderr_output.lower())

    @patch('philote_mdo.wrappers.julia.cli.serve_explicit_discipline')
    def test_config_loading_with_valid_file(self, mock_serve_explicit):
        """Test that valid config file is loaded correctly."""
        with patch('sys.argv', ['philote-julia-serve', self.paraboloid_config]):
            mock_serve_explicit.return_value = None

            cli.main()

            # Verify config was loaded correctly
            config = mock_serve_explicit.call_args[0][0]
            self.assertEqual(config.discipline.julia_type, "ParaboloidDiscipline")
            self.assertTrue(config.discipline.julia_file.endswith("paraboloid.jl"))

    def test_invalid_discipline_kind_in_config(self):
        """Test that invalid discipline kind exits with error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Julia file
            julia_file = os.path.join(tmpdir, "test.jl")
            with open(julia_file, 'w') as f:
                f.write("# Test file\n")

            # Create a config with invalid kind (will pass PhiloteConfig validation
            # but this tests the routing logic)
            config_file = os.path.join(tmpdir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write(
                    "discipline:\n"
                    "  kind: invalid_kind\n"
                    f"  julia_file: {julia_file}\n"
                    "  julia_type: TestDiscipline\n"
                )

            with patch('sys.argv', ['philote-julia-serve', config_file]):
                with patch('sys.stderr', new=StringIO()) as mock_stderr:
                    with self.assertRaises(SystemExit) as cm:
                        cli.main()
                    # Should exit during config validation
                    self.assertEqual(cm.exception.code, 1)
                    stderr_output = mock_stderr.getvalue()
                    self.assertIn("error", stderr_output.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
