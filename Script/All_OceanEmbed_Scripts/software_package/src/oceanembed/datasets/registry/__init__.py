"""
OceanEmbed dataset registry loader.

Provides functions for discovering, loading, and validating dataset registry
definitions from YAML files in configs/datasets/.

Responsibilities
----------------
- Locate registry YAML files by scanning a configured directory.
- Load YAML and validate each file against the DatasetDefinition schema.
- Expose a DatasetRegistry object that maps dataset_id → DatasetDefinition.
- Fail loudly on invalid configuration; never silently skip bad entries.

Not responsible for
-------------------
- Downloading data files.
- Making scientific decisions about dataset content.
- Resolving UNVERIFIED or UNRESOLVED fields.
- Applying any transformations to dataset values.

Usage example
-------------
    from oceanembed.datasets.registry import load_registry

    registry = load_registry(Path("configs/datasets"))
    oisst = registry.get("oisst")
    all_ids = registry.list_ids()
"""

from oceanembed.datasets.registry.loader import (
    DatasetRegistry,
    load_registry,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
    RegistryDuplicateError,
)

__all__ = [
    "DatasetRegistry",
    "load_registry",
    "RegistryError",
    "RegistryNotFoundError",
    "RegistryValidationError",
    "RegistryDuplicateError",
]
