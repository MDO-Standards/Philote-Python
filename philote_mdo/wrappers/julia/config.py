"""
Configuration file loading and validation for Philote-Julia.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml


@dataclass
class DisciplineConfig:
    """Configuration for a Julia discipline."""

    kind: str  # "explicit" or "implicit"
    julia_file: str
    julia_type: str
    options: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.kind not in ["explicit", "implicit"]:
            raise ValueError(f"discipline.kind must be 'explicit' or 'implicit', got '{self.kind}'")

        if not self.julia_file:
            raise ValueError("discipline.julia_file is required")

        if not self.julia_type:
            raise ValueError("discipline.julia_type is required")


@dataclass
class ServerConfig:
    """Configuration for the gRPC server."""

    address: str = "[::]:50051"
    max_workers: int = 10

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_workers < 1:
            raise ValueError(f"server.max_workers must be >= 1, got {self.max_workers}")


@dataclass
class PhiloteConfig:
    """Complete configuration for Philote-Julia server."""

    discipline: DisciplineConfig
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PhiloteConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PhiloteConfig object

        Raises:
            FileNotFoundError: If yaml_path doesn't exist
            ValueError: If configuration is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML: expected dict, got {type(data)}")

        # Parse discipline config
        if 'discipline' not in data:
            raise ValueError("Missing required 'discipline' section in config")

        disc_data = data['discipline']
        discipline = DisciplineConfig(
            kind=disc_data.get('kind', 'explicit'),
            julia_file=disc_data.get('julia_file', ''),
            julia_type=disc_data.get('julia_type', ''),
            options=disc_data.get('options', {})
        )

        # Resolve relative path to julia_file from yaml file location
        if not os.path.isabs(discipline.julia_file):
            yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
            discipline.julia_file = os.path.join(yaml_dir, discipline.julia_file)

        # Verify julia_file exists
        if not os.path.exists(discipline.julia_file):
            raise FileNotFoundError(
                f"Julia file not found: {discipline.julia_file}\n"
                f"(specified in {yaml_path})"
            )

        # Parse server config (optional)
        server_data = data.get('server', {})
        server = ServerConfig(
            address=server_data.get('address', '[::]:50051'),
            max_workers=server_data.get('max_workers', 10)
        )

        return cls(discipline=discipline, server=server)

    def to_yaml(self, yaml_path: str):
        """
        Write configuration to a YAML file.

        Args:
            yaml_path: Path to write YAML configuration
        """
        data = {
            'discipline': {
                'kind': self.discipline.kind,
                'julia_file': self.discipline.julia_file,
                'julia_type': self.discipline.julia_type,
                'options': self.discipline.options
            },
            'server': {
                'address': self.server.address,
                'max_workers': self.server.max_workers
            }
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
