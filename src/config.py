"""Configuration management for GalaxyMorph."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Config:
    """Configuration loader for the project."""

    def __init__(self, config_file: str = None):
        """Initialize configuration from YAML file."""
        if config_file is None:
            config_file = PROJECT_ROOT / "configs" / "default.yaml"
        else:
            config_file = PROJECT_ROOT / config_file

        self.config_file = config_file
        self.config: Dict[str, Any] = {}

        if self.config_file.exists():
            self.load(str(self.config_file))

    def load(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f) or {}

    def get(self, key: str, default=None):
        """Get configuration value by key with dot notation support."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str):
        """Get configuration value like a dictionary."""
        return self.get(key)

    def __repr__(self):
        return f"Config({self.config})"


# Default global config instance
_config = None


def get_config(config_file: str = None) -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


def load_config(config_file: str):
    """Load a new configuration file."""
    global _config
    _config = Config(config_file)
    return _config
