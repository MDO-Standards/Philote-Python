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
import tempfile
import unittest

try:
    from philote_mdo.wrappers.julia.config import PhiloteConfig, DisciplineConfig, ServerConfig
    HAS_JULIA_CONFIG = True
except ImportError:
    HAS_JULIA_CONFIG = False


@unittest.skipIf(not HAS_JULIA_CONFIG, "Julia config module not available")
class DisciplineConfigTests(unittest.TestCase):
    """
    Unit tests for DisciplineConfig validation.
    """

    def test_valid_explicit_discipline(self):
        """Test creating a valid explicit discipline config."""
        config = DisciplineConfig(
            kind="explicit",
            julia_file="/path/to/file.jl",
            julia_type="MyDiscipline"
        )
        self.assertEqual(config.kind, "explicit")
        self.assertEqual(config.julia_file, "/path/to/file.jl")
        self.assertEqual(config.julia_type, "MyDiscipline")

    def test_valid_implicit_discipline(self):
        """Test creating a valid implicit discipline config."""
        config = DisciplineConfig(
            kind="implicit",
            julia_file="/path/to/file.jl",
            julia_type="MyDiscipline"
        )
        self.assertEqual(config.kind, "implicit")

    def test_invalid_kind(self):
        """Test that invalid kind raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            DisciplineConfig(
                kind="invalid",
                julia_file="/path/to/file.jl",
                julia_type="MyDiscipline"
            )
        self.assertIn("must be 'explicit' or 'implicit'", str(cm.exception))

    def test_missing_julia_file(self):
        """Test that empty julia_file raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            DisciplineConfig(
                kind="explicit",
                julia_file="",
                julia_type="MyDiscipline"
            )
        self.assertIn("julia_file is required", str(cm.exception))

    def test_missing_julia_type(self):
        """Test that empty julia_type raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            DisciplineConfig(
                kind="explicit",
                julia_file="/path/to/file.jl",
                julia_type=""
            )
        self.assertIn("julia_type is required", str(cm.exception))

    def test_with_options(self):
        """Test discipline config with options."""
        config = DisciplineConfig(
            kind="explicit",
            julia_file="/path/to/file.jl",
            julia_type="MyDiscipline",
            options={"scale_factor": 2.0, "offset": 10.0}
        )
        self.assertEqual(config.options["scale_factor"], 2.0)
        self.assertEqual(config.options["offset"], 10.0)


@unittest.skipIf(not HAS_JULIA_CONFIG, "Julia config module not available")
class ServerConfigTests(unittest.TestCase):
    """
    Unit tests for ServerConfig validation.
    """

    def test_default_server_config(self):
        """Test default server configuration."""
        config = ServerConfig()
        self.assertEqual(config.address, "[::]:50051")
        self.assertEqual(config.max_workers, 10)

    def test_custom_server_config(self):
        """Test custom server configuration."""
        config = ServerConfig(address="localhost:8080", max_workers=20)
        self.assertEqual(config.address, "localhost:8080")
        self.assertEqual(config.max_workers, 20)

    def test_invalid_max_workers_zero(self):
        """Test that max_workers=0 raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ServerConfig(max_workers=0)
        self.assertIn("max_workers must be >= 1", str(cm.exception))

    def test_invalid_max_workers_negative(self):
        """Test that negative max_workers raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            ServerConfig(max_workers=-5)
        self.assertIn("max_workers must be >= 1", str(cm.exception))


@unittest.skipIf(not HAS_JULIA_CONFIG, "Julia config module not available")
class PhiloteConfigTests(unittest.TestCase):
    """
    Unit tests for PhiloteConfig YAML loading and writing.
    """

    @classmethod
    def setUpClass(cls):
        """Set up paths to example config files."""
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(tests_dir)
        configs_dir = os.path.join(project_root, "examples", "julia", "configs")

        cls.paraboloid_config = os.path.join(configs_dir, "paraboloid.yaml")
        cls.quadratic_config = os.path.join(configs_dir, "quadratic.yaml")

    def test_load_explicit_config(self):
        """Test loading explicit discipline configuration from YAML."""
        config = PhiloteConfig.from_yaml(self.paraboloid_config)

        self.assertEqual(config.discipline.kind, "explicit")
        self.assertEqual(config.discipline.julia_type, "ParaboloidDiscipline")
        self.assertTrue(config.discipline.julia_file.endswith("paraboloid.jl"))
        self.assertTrue(os.path.exists(config.discipline.julia_file))

    def test_load_implicit_config(self):
        """Test loading implicit discipline configuration from YAML."""
        config = PhiloteConfig.from_yaml(self.quadratic_config)

        self.assertEqual(config.discipline.kind, "implicit")
        self.assertEqual(config.discipline.julia_type, "QuadraticDiscipline")
        self.assertTrue(config.discipline.julia_file.endswith("quadratic.jl"))
        self.assertTrue(os.path.exists(config.discipline.julia_file))

    def test_load_missing_file(self):
        """Test that loading non-existent config file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as cm:
            PhiloteConfig.from_yaml("/nonexistent/config.yaml")
        self.assertIn("Configuration file not found", str(cm.exception))

    def test_load_invalid_yaml_structure(self):
        """Test that invalid YAML structure raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("just a string, not a dict")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                PhiloteConfig.from_yaml(temp_path)
            self.assertIn("Invalid YAML", str(cm.exception))
        finally:
            os.unlink(temp_path)

    def test_load_missing_discipline_section(self):
        """Test that missing discipline section raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("server:\n  address: '[::]:50051'\n")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                PhiloteConfig.from_yaml(temp_path)
            self.assertIn("Missing required 'discipline' section", str(cm.exception))
        finally:
            os.unlink(temp_path)

    def test_load_missing_julia_file_in_yaml(self):
        """Test that missing julia_file field in YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("discipline:\n  kind: explicit\n  julia_type: MyDiscipline\n")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                PhiloteConfig.from_yaml(temp_path)
            self.assertIn("julia_file is required", str(cm.exception))
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_julia_file(self):
        """Test that non-existent julia_file raises FileNotFoundError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("discipline:\n  kind: explicit\n  julia_file: /nonexistent/file.jl\n  julia_type: MyDiscipline\n")
            temp_path = f.name

        try:
            with self.assertRaises(FileNotFoundError) as cm:
                PhiloteConfig.from_yaml(temp_path)
            self.assertIn("Julia file not found", str(cm.exception))
        finally:
            os.unlink(temp_path)

    def test_relative_path_resolution(self):
        """Test that relative julia_file paths are resolved correctly."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Julia file
            julia_file = os.path.join(tmpdir, "test.jl")
            with open(julia_file, 'w') as f:
                f.write("# Test file\n")

            # Create a config file with relative path
            config_file = os.path.join(tmpdir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write("discipline:\n  kind: explicit\n  julia_file: test.jl\n  julia_type: MyDiscipline\n")

            # Load config and verify path was resolved
            config = PhiloteConfig.from_yaml(config_file)
            self.assertEqual(config.discipline.julia_file, julia_file)
            self.assertTrue(os.path.isabs(config.discipline.julia_file))

    def test_round_trip_save_load(self):
        """Test saving and loading configuration (round trip)."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Julia file
            julia_file = os.path.join(tmpdir, "test.jl")
            with open(julia_file, 'w') as f:
                f.write("# Test file\n")

            # Create a config
            original_config = PhiloteConfig(
                discipline=DisciplineConfig(
                    kind="explicit",
                    julia_file=julia_file,
                    julia_type="TestDiscipline",
                    options={"param": 42}
                ),
                server=ServerConfig(
                    address="localhost:9999",
                    max_workers=5
                )
            )

            # Save to YAML
            config_file = os.path.join(tmpdir, "config.yaml")
            original_config.to_yaml(config_file)

            # Load it back
            loaded_config = PhiloteConfig.from_yaml(config_file)

            # Verify everything matches
            self.assertEqual(loaded_config.discipline.kind, "explicit")
            self.assertEqual(loaded_config.discipline.julia_type, "TestDiscipline")
            self.assertEqual(loaded_config.discipline.options["param"], 42)
            self.assertEqual(loaded_config.server.address, "localhost:9999")
            self.assertEqual(loaded_config.server.max_workers, 5)

    def test_config_with_options(self):
        """Test loading configuration with discipline options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Julia file
            julia_file = os.path.join(tmpdir, "test.jl")
            with open(julia_file, 'w') as f:
                f.write("# Test file\n")

            # Create a config file with options
            config_file = os.path.join(tmpdir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write(
                    "discipline:\n"
                    "  kind: explicit\n"
                    f"  julia_file: {julia_file}\n"
                    "  julia_type: TestDiscipline\n"
                    "  options:\n"
                    "    scale_factor: 2.0\n"
                    "    offset: 10.0\n"
                )

            # Load and verify
            config = PhiloteConfig.from_yaml(config_file)
            self.assertEqual(config.discipline.options["scale_factor"], 2.0)
            self.assertEqual(config.discipline.options["offset"], 10.0)

    def test_config_default_server(self):
        """Test that server config has defaults when not specified in YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Julia file
            julia_file = os.path.join(tmpdir, "test.jl")
            with open(julia_file, 'w') as f:
                f.write("# Test file\n")

            # Create a config file without server section
            config_file = os.path.join(tmpdir, "config.yaml")
            with open(config_file, 'w') as f:
                f.write(
                    "discipline:\n"
                    "  kind: explicit\n"
                    f"  julia_file: {julia_file}\n"
                    "  julia_type: TestDiscipline\n"
                )

            # Load and verify defaults
            config = PhiloteConfig.from_yaml(config_file)
            self.assertEqual(config.server.address, "[::]:50051")
            self.assertEqual(config.server.max_workers, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
