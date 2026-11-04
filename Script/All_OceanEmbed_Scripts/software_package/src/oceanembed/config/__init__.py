"""
OceanEmbed configuration module.

Responsible for loading and validating project configuration from
configs/oceanembed.yaml and environment-specific overrides.

Usage
-----
    from oceanembed.config import get_settings, resolve_data_root
    settings = get_settings()
    data_root = resolve_data_root()   # raises if not set
"""

from oceanembed.config.loader import get_settings, resolve_data_root, OceanEmbedSettings

__all__ = ["get_settings", "resolve_data_root", "OceanEmbedSettings"]
