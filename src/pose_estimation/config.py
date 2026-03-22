"""
Unified Configuration Manager for Pose Estimation.

Handles loading of config.yaml and provides configuration values as nested dictionaries.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Define base project directory (Project Root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default path to config.yaml in the project root
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

class ConfigManager:
    """Handles loading and retrieval of configuration values."""
    
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file or return empty dict if not found."""
        if not self.config_path.exists():
            print(f"[WARN] Configuration file not found at: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[ERROR] Failed to load configuration: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value from the nested dictionary."""
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def all(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return self._config


# Instantiate global configuration manager
cfg = ConfigManager()

# Pre-defined common configurations for easier access
DATASET_CFG = cfg.get("dataset", {})
MODEL_CFG = cfg.get("model", {})
TRAINING_CFG = cfg.get("training", {})
INFERENCE_CFG = cfg.get("inference", {})
