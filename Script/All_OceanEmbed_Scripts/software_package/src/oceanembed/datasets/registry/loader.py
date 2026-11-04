"""
OceanEmbed Dataset Registry Loader.

Loads, validates, and indexes dataset registry YAML definitions.

Design principles
-----------------
- Fail loudly on any validation error.  A misconfigured registry entry is
  always a bug, never a recoverable state.
- Never silently skip or partially load a bad entry.
- Never hard-code dataset IDs, paths, or any source-specific value.
- Paths are provided by the caller, not resolved internally.
- No scientific decisions are made during loading.

See docs/datasets/DATASET_REGISTRY.md for the full contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import yaml
from pydantic import ValidationError

from oceanembed.datasets.schemas.registry import DatasetDefinition


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    """Base exception for all dataset registry errors."""


class RegistryNotFoundError(RegistryError):
    """Raised when a requested dataset_id does not exist in the registry."""

    def __init__(self, dataset_id: str, available: list[str]) -> None:
        self.dataset_id = dataset_id
        self.available = available
        super().__init__(
            f"Dataset '{dataset_id}' not found in registry. "
            f"Available datasets: {sorted(available)}"
        )


class RegistryValidationError(RegistryError):
    """Raised when a registry YAML file fails schema validation."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(
            f"Registry file failed validation: {path}\n{detail}"
        )


class RegistryDuplicateError(RegistryError):
    """Raised when two registry files declare the same dataset_id."""

    def __init__(self, dataset_id: str, paths: list[Path]) -> None:
        self.dataset_id = dataset_id
        self.paths = paths
        super().__init__(
            f"Duplicate dataset_id '{dataset_id}' found in: "
            + ", ".join(str(p) for p in paths)
        )


# ---------------------------------------------------------------------------
# DatasetRegistry
# ---------------------------------------------------------------------------


class DatasetRegistry:
    """
    Indexed collection of validated DatasetDefinition objects.

    Constructed by ``load_registry()``.  Do not instantiate directly.

    Methods
    -------
    get(dataset_id)
        Return the DatasetDefinition for the given ID.
        Raises RegistryNotFoundError if not found.

    list_ids()
        Return a sorted list of all registered dataset IDs.

    __len__()
        Return the number of registered datasets.

    __iter__()
        Iterate over (dataset_id, DatasetDefinition) pairs.
    """

    def __init__(self, entries: dict[str, DatasetDefinition]) -> None:
        self._entries: dict[str, DatasetDefinition] = entries

    def get(self, dataset_id: str) -> DatasetDefinition:
        """
        Return the DatasetDefinition for the given dataset_id.

        Parameters
        ----------
        dataset_id:
            The unique dataset identifier to look up.

        Raises
        ------
        RegistryNotFoundError
            If no entry with the given ID exists in the registry.
        """
        if dataset_id not in self._entries:
            raise RegistryNotFoundError(
                dataset_id=dataset_id,
                available=list(self._entries.keys()),
            )
        return self._entries[dataset_id]

    def list_ids(self) -> list[str]:
        """Return a sorted list of all registered dataset IDs."""
        return sorted(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[tuple[str, DatasetDefinition]]:
        yield from self._entries.items()

    def __repr__(self) -> str:
        ids = self.list_ids()
        return f"DatasetRegistry({len(self)} datasets: {ids})"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_registry(registry_dir: Path) -> DatasetRegistry:
    """
    Load and validate all dataset registry YAML files in *registry_dir*.

    Parameters
    ----------
    registry_dir:
        Path to a directory containing ``*.yaml`` registry files.
        Typically ``configs/datasets/`` relative to the repository root.
        Must exist and be a directory.

    Returns
    -------
    DatasetRegistry
        An indexed, validated collection of all dataset definitions found.

    Raises
    ------
    ValueError
        If *registry_dir* does not exist or is not a directory.
    RegistryValidationError
        If any YAML file fails to parse or fails Pydantic schema validation.
    RegistryDuplicateError
        If two YAML files declare the same ``dataset_id``.

    Notes
    -----
    - Files are processed in alphabetical order for reproducibility.
    - No entry is silently skipped.  Any error halts the load.
    - The loaded registry reflects only what is written in the YAML files.
      No defaults are injected for missing or UNVERIFIED fields.
    """
    registry_dir = Path(registry_dir)
    if not registry_dir.exists():
        raise ValueError(f"Registry directory does not exist: {registry_dir}")
    if not registry_dir.is_dir():
        raise ValueError(f"Registry path is not a directory: {registry_dir}")

    yaml_files = sorted(registry_dir.glob("*.yaml"))
    if not yaml_files:
        # Not an error — an empty registry is valid during early development.
        return DatasetRegistry({})

    entries: dict[str, DatasetDefinition] = {}
    # Track which file each ID came from to detect duplicates.
    id_to_path: dict[str, Path] = {}

    for yaml_path in yaml_files:
        definition = _load_single(yaml_path)

        dataset_id = definition.dataset_id

        # Enforce that dataset_id matches the filename stem.
        expected_id = yaml_path.stem
        if dataset_id != expected_id:
            raise RegistryValidationError(
                path=yaml_path,
                detail=(
                    f"dataset_id '{dataset_id}' does not match filename "
                    f"'{yaml_path.name}'. They must match."
                ),
            )

        # Detect duplicate IDs across multiple files.
        if dataset_id in id_to_path:
            raise RegistryDuplicateError(
                dataset_id=dataset_id,
                paths=[id_to_path[dataset_id], yaml_path],
            )

        entries[dataset_id] = definition
        id_to_path[dataset_id] = yaml_path

    return DatasetRegistry(entries)


def _load_single(yaml_path: Path) -> DatasetDefinition:
    """
    Load and validate a single registry YAML file.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file to load.

    Returns
    -------
    DatasetDefinition
        The validated dataset definition.

    Raises
    ------
    RegistryValidationError
        If the file cannot be read, parsed as YAML, or validated by Pydantic.
    """
    # --- Read file ---
    try:
        raw_text = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryValidationError(
            path=yaml_path,
            detail=f"Could not read file: {exc}",
        ) from exc

    # --- Parse YAML ---
    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise RegistryValidationError(
            path=yaml_path,
            detail=f"YAML parse error: {exc}",
        ) from exc

    if not isinstance(raw_data, dict):
        raise RegistryValidationError(
            path=yaml_path,
            detail=(
                f"Expected a YAML mapping at the top level, "
                f"got {type(raw_data).__name__}."
            ),
        )

    # --- Validate with Pydantic ---
    try:
        return DatasetDefinition.model_validate(raw_data)
    except ValidationError as exc:
        raise RegistryValidationError(
            path=yaml_path,
            detail=f"Schema validation failed:\n{exc}",
        ) from exc
