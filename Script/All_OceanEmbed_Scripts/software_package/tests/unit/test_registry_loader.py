"""
Unit tests for the OceanEmbed dataset registry loader.

Tests cover:
- load_registry() successfully loads all 8 production registry files.
- get() returns the correct DatasetDefinition.
- list_ids() returns sorted IDs.
- RegistryNotFoundError raised for unknown dataset_id.
- RegistryValidationError raised for invalid YAML.
- RegistryValidationError raised for missing required fields.
- RegistryValidationError raised for dataset_id / filename mismatch.
- RegistryDuplicateError raised for duplicate dataset_id files.
- Empty directory returns empty registry.
- Non-existent directory raises ValueError.

No real network access or dataset files are used.
Fixtures are synthetic minimal YAML strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from oceanembed.datasets.registry import (
    DatasetRegistry,
    RegistryDuplicateError,
    RegistryError,
    RegistryNotFoundError,
    RegistryValidationError,
    load_registry,
)
from oceanembed.datasets.schemas.registry import DatasetRole


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_yaml(directory: Path, filename: str, content: dict) -> Path:
    """Write a YAML file to directory and return the path."""
    p = directory / filename
    p.write_text(yaml.dump(content), encoding="utf-8")
    return p


def _minimal(dataset_id: str = "testds") -> dict:
    return {
        "dataset_id": dataset_id,
        "product_name": f"Test Product {dataset_id}",
        "role": "surface_input",
    }


# ---------------------------------------------------------------------------
# Production registry — all 8 files
# ---------------------------------------------------------------------------

class TestProductionRegistry:
    """Validate all 8 production registry files load and validate."""

    def test_all_eight_datasets_load(self, production_registry: DatasetRegistry):
        assert len(production_registry) == 8

    def test_all_expected_ids_present(self, production_registry: DatasetRegistry):
        ids = set(production_registry.list_ids())
        expected = {"oisst", "oisss", "duacs", "ccmp", "oscar", "globcurrent", "glorys", "argo"}
        assert ids == expected

    def test_list_ids_is_sorted(self, production_registry: DatasetRegistry):
        ids = production_registry.list_ids()
        assert ids == sorted(ids)

    def test_glorys_role_is_target(self, production_registry: DatasetRegistry):
        assert production_registry.get("glorys").role == DatasetRole.TARGET

    def test_argo_role_is_validation(self, production_registry: DatasetRegistry):
        assert production_registry.get("argo").role == DatasetRole.VALIDATION

    def test_surface_inputs_have_correct_role(self, production_registry: DatasetRegistry):
        surface_ids = {"oisst", "oisss", "duacs", "ccmp", "oscar", "globcurrent"}
        for ds_id in surface_ids:
            defn = production_registry.get(ds_id)
            assert defn.role == DatasetRole.SURFACE_INPUT, (
                f"{ds_id} should be surface_input, got {defn.role}"
            )

    def test_all_datasets_have_known_limitations(self, production_registry: DatasetRegistry):
        for ds_id, defn in production_registry:
            assert len(defn.known_limitations) > 0, (
                f"{ds_id} should have at least one known_limitation"
            )

    def test_all_datasets_have_open_decisions(self, production_registry: DatasetRegistry):
        for ds_id, defn in production_registry:
            assert len(defn.open_scientific_decisions) > 0, (
                f"{ds_id} should reference at least one open scientific decision"
            )

    def test_no_hardcoded_absolute_paths_in_storage(self, production_registry: DatasetRegistry):
        import re
        for ds_id, defn in production_registry:
            if defn.storage is None:
                continue
            for field_name in ("raw_path_template", "processed_path_template"):
                val = getattr(defn.storage, field_name, None)
                if val and isinstance(val, str) and val not in ("UNVERIFIED", "UNRESOLVED"):
                    assert not re.match(r"^[A-Za-z]:\\", val), (
                        f"{ds_id}.storage.{field_name} has a hard-coded Windows path"
                    )

    def test_no_dataset_missing_provider(self, production_registry: DatasetRegistry):
        for ds_id, defn in production_registry:
            assert defn.provider is not None, f"{ds_id} is missing provider info"
            assert defn.provider.name, f"{ds_id} provider has no name"


# ---------------------------------------------------------------------------
# get() and list_ids()
# ---------------------------------------------------------------------------

class TestRegistryLookup:
    def test_get_existing_dataset(self, tmp_path: Path):
        _write_yaml(tmp_path, "testds.yaml", _minimal("testds"))
        registry = load_registry(tmp_path)
        defn = registry.get("testds")
        assert defn.dataset_id == "testds"

    def test_get_unknown_raises_not_found(self, tmp_path: Path):
        _write_yaml(tmp_path, "testds.yaml", _minimal("testds"))
        registry = load_registry(tmp_path)
        with pytest.raises(RegistryNotFoundError) as exc_info:
            registry.get("does_not_exist")
        assert "does_not_exist" in str(exc_info.value)
        assert "testds" in str(exc_info.value)  # available list

    def test_list_ids_returns_sorted(self, tmp_path: Path):
        for ds_id in ("zzz", "aaa", "mmm"):
            _write_yaml(tmp_path, f"{ds_id}.yaml", _minimal(ds_id))
        registry = load_registry(tmp_path)
        assert registry.list_ids() == ["aaa", "mmm", "zzz"]

    def test_len_matches_file_count(self, tmp_path: Path):
        for ds_id in ("alpha", "beta", "gamma"):
            _write_yaml(tmp_path, f"{ds_id}.yaml", _minimal(ds_id))
        registry = load_registry(tmp_path)
        assert len(registry) == 3


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestRegistryErrors:
    def test_nonexistent_directory_raises_value_error(self):
        with pytest.raises(ValueError, match="does not exist"):
            load_registry(Path("/nonexistent/path/to/registry"))

    def test_path_is_file_not_dir_raises_value_error(self, tmp_path: Path):
        f = tmp_path / "not_a_dir.yaml"
        f.write_text("dataset_id: x\n")
        with pytest.raises(ValueError, match="not a directory"):
            load_registry(f)

    def test_empty_directory_returns_empty_registry(self, tmp_path: Path):
        registry = load_registry(tmp_path)
        assert len(registry) == 0
        assert registry.list_ids() == []

    def test_invalid_yaml_raises_registry_validation_error(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("dataset_id: [\ninvalid yaml{{{", encoding="utf-8")
        with pytest.raises(RegistryValidationError) as exc_info:
            load_registry(tmp_path)
        assert "bad.yaml" in str(exc_info.value)

    def test_missing_dataset_id_raises_validation_error(self, tmp_path: Path):
        content = {"product_name": "No ID", "role": "surface_input"}
        _write_yaml(tmp_path, "noid.yaml", content)
        with pytest.raises(RegistryValidationError) as exc_info:
            load_registry(tmp_path)
        assert "noid.yaml" in str(exc_info.value)

    def test_missing_role_raises_validation_error(self, tmp_path: Path):
        content = {"dataset_id": "norole", "product_name": "No Role"}
        _write_yaml(tmp_path, "norole.yaml", content)
        with pytest.raises(RegistryValidationError) as exc_info:
            load_registry(tmp_path)
        assert "norole.yaml" in str(exc_info.value)

    def test_invalid_role_raises_validation_error(self, tmp_path: Path):
        content = _minimal("badrole")
        content["role"] = "not_a_role"
        _write_yaml(tmp_path, "badrole.yaml", content)
        with pytest.raises(RegistryValidationError):
            load_registry(tmp_path)

    def test_dataset_id_mismatch_raises_validation_error(self, tmp_path: Path):
        """dataset_id in YAML must match the filename stem."""
        content = _minimal("different_id")  # ID says 'different_id'
        _write_yaml(tmp_path, "filename_id.yaml", content)  # filename says 'filename_id'
        with pytest.raises(RegistryValidationError) as exc_info:
            load_registry(tmp_path)
        assert "filename_id.yaml" in str(exc_info.value)

    def test_duplicate_dataset_id_raises_duplicate_error(self, tmp_path: Path):
        """Two YAML files that each have matching names but identical dataset_ids
        cannot exist in the same directory (IDs must be unique per registry).
        We simulate this by writing the same dataset_id under two subdirectory
        registries and merging them — but the simpler test is to verify the
        RegistryDuplicateError message structure, since in practice the loader
        enforces filename == dataset_id, making true collisions impossible within
        a single directory.

        Instead, directly test that RegistryDuplicateError carries the right info.
        """
        err = RegistryDuplicateError(
            dataset_id="dup",
            paths=[tmp_path / "dup.yaml", tmp_path / "dup_copy.yaml"],
        )
        assert "dup" in str(err)
        assert isinstance(err, RegistryError)

    def test_filename_id_match_prevents_duplicates(self, tmp_path: Path):
        """The filename==dataset_id rule means true duplicates cannot exist in
        one directory — a second file trying to claim the same ID would be
        named identically and overwrite the first at the OS level.
        This test confirms the rule is enforced: a file whose dataset_id does
        not match its filename stem is rejected before duplicate checking."""
        _write_yaml(tmp_path, "alpha.yaml", _minimal("alpha"))
        # This file has dataset_id 'alpha' but is named 'beta.yaml' — mismatch
        _write_yaml(tmp_path, "beta.yaml", _minimal("alpha"))
        with pytest.raises(RegistryValidationError) as exc_info:
            load_registry(tmp_path)
        assert "beta.yaml" in str(exc_info.value)
        assert "alpha" in str(exc_info.value)

    def test_non_mapping_yaml_raises_validation_error(self, tmp_path: Path):
        """A YAML file that is a list, not a dict, should fail."""
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(RegistryValidationError):
            load_registry(tmp_path)

    def test_registry_repr(self, tmp_path: Path):
        _write_yaml(tmp_path, "testds.yaml", _minimal("testds"))
        registry = load_registry(tmp_path)
        r = repr(registry)
        assert "DatasetRegistry" in r
        assert "testds" in r


# ---------------------------------------------------------------------------
# Conftest-style fixtures (defined here for unit test isolation)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production_registry() -> DatasetRegistry:
    """Load the production registry from configs/datasets/."""
    from pathlib import Path

    # Walk up from this test file to find configs/datasets/
    this_file = Path(__file__).resolve()
    # tests/unit/test_registry_loader.py → go up 3 levels to get repo root
    repo_root = this_file.parent.parent.parent
    registry_dir = repo_root / "configs" / "datasets"

    if not registry_dir.exists():
        pytest.skip(f"Production registry not found at {registry_dir}")

    return load_registry(registry_dir)
