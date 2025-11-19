"""
Configuration loader for Microgrid Autopilot.

Loads system parameters from config.yaml with sensible defaults.
"""

import yaml
from pathlib import Path
from typing import Any


# Default configuration values
DEFAULTS = {
    'battery': {
        'capacity_mwh': 5.0,
        'max_power_mw': 2.0,
        'charge_efficiency': 0.95,
        'discharge_efficiency': 0.95,
        'soc_min': 0.20,
        'soc_max': 0.90,
        'soc_initial': 0.50,
        'cycle_cost_per_kwh': 0.02,
        'calendar_aging_per_day': 0.0001,
    },
    'optimization': {
        'carbon_weight': 0.001,
        'export_price_ratio': 0.5,
        'solver_timeout_seconds': 300,
        'horizon_hours': 48,
        'demand_charge': {
            'enabled': True,
            'rate_per_kw': 15.0,
            'billing_period_days': 30,
            'peak_window_start': 12,
            'peak_window_end': 20,
        }
    },
    'carbon': {
        'default_intensity': 410,
        'use_time_varying': True,
    },
    'paths': {
        'data_raw': 'data_raw',
        'data_processed': 'data_processed',
        'models': 'models',
        'results': 'results',
    }
}


class Config:
    """Configuration manager for Microgrid Autopilot."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config.yaml"

        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = {}

        # Merge with defaults
        self._config = self._merge_dicts(DEFAULTS, self._config)

    def _merge_dicts(self, default: dict, override: dict) -> dict:
        """Recursively merge override into default."""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, *keys, default=None) -> Any:
        """
        Get nested configuration value.

        Example: config.get('battery', 'capacity_mwh')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def battery(self) -> dict:
        return self._config.get('battery', {})

    @property
    def optimization(self) -> dict:
        return self._config.get('optimization', {})

    @property
    def demand_charge(self) -> dict:
        return self.optimization.get('demand_charge', {})

    @property
    def carbon(self) -> dict:
        return self._config.get('carbon', {})

    @property
    def paths(self) -> dict:
        return self._config.get('paths', {})

    def reload(self):
        """Reload configuration from file."""
        self._load_config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
