"""
Unit tests for the OceanEmbed dataset registry schema.

Tests cover:
- Valid YAML definitions load and validate correctly.
- Invalid or missing required fields raise appropriate errors.
- Path template validation rejects hard-coded absolute paths.
- Sentinel string values (UNVERIFIED, UNRESOLVED) are accepted as-is.
- OSD ID format validation.
- DatasetRole and OceanEmbedDomain validation.

All test data is synthetic — no real source datasets are touched.
No network access is performed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from oceanembed.datasets.schemas.registry import (
    AcquisitionInfo,
    DatasetDefinition,
    DatasetRole,
    OceanEmbedDomain,
    OpenScientificDecision,
    StorageInfo,
    VariableInfo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_valid() -> dict:
    """Return a minimal valid DatasetDefinition payload."""
    return {
        "dataset_id": "testds",
        "product_name": "Test Product v1",
        "role": "surface_input",
    }


# ---------------------------------------------------------------------------
# DatasetDefinition — required fields
# ---------------------------------------------------------------------------

class TestDatasetDefinitionRequired:
    """Required fields must be present and well-formed."""

    def test_minimal_valid(self):
        defn = DatasetDefinition(**_minimal_valid())
        assert defn.dataset_id == "testds"
        assert defn.role == DatasetRole.SURFACE_INPUT

    def test_missing_dataset_id_raises(self):
        data = _minimal_valid()
        del data["dataset_id"]
        with pytest.raises(ValidationError) as exc_info:
            DatasetDefinition(**data)
        assert "dataset_id" in str(exc_info.value)

    def test_missing_product_name_raises(self):
        data = _minimal_valid()
        del data["product_name"]
        with pytest.raises(ValidationError) as exc_info:
            DatasetDefinition(**data)
        assert "product_name" in str(exc_info.value)

    def test_missing_role_raises(self):
        data = _minimal_valid()
        del data["role"]
        with pytest.raises(ValidationError) as exc_info:
            DatasetDefinition(**data)
        assert "role" in str(exc_info.value)

    def test_invalid_role_raises(self):
        data = _minimal_valid()
        data["role"] = "not_a_valid_role"
        with pytest.raises(ValidationError) as exc_info:
            DatasetDefinition(**data)
        assert "role" in str(exc_info.value)

    def test_all_three_roles_accepted(self):
        for role_value in ("surface_input", "target", "validation"):
            data = _minimal_valid()
            data["role"] = role_value
            defn = DatasetDefinition(**data)
            assert defn.role.value == role_value

    def test_uppercase_dataset_id_rejected(self):
        data = _minimal_valid()
        data["dataset_id"] = "TestDS"
        with pytest.raises(ValidationError) as exc_info:
            DatasetDefinition(**data)
        assert "dataset_id" in str(exc_info.value)

    def test_dataset_id_with_digits_accepted(self):
        data = _minimal_valid()
        data["dataset_id"] = "oisst2"
        defn = DatasetDefinition(**data)
        assert defn.dataset_id == "oisst2"

    def test_dataset_id_with_underscores_accepted(self):
        data = _minimal_valid()
        data["dataset_id"] = "glob_current"
        defn = DatasetDefinition(**data)
        assert defn.dataset_id == "glob_current"


# ---------------------------------------------------------------------------
# Optional fields and sentinel strings
# ---------------------------------------------------------------------------

class TestOptionalAndSentinels:
    """Optional fields and sentinel strings behave correctly."""

    def test_unverified_product_id_accepted(self):
        data = _minimal_valid()
        data["product_id"] = "UNVERIFIED"
        defn = DatasetDefinition(**data)
        assert defn.product_id == "UNVERIFIED"

    def test_unresolved_version_accepted(self):
        data = _minimal_valid()
        data["version"] = "UNRESOLVED"
        defn = DatasetDefinition(**data)
        assert defn.version == "UNRESOLVED"

    def test_known_limitations_list(self):
        data = _minimal_valid()
        data["known_limitations"] = ["limitation one", "limitation two"]
        defn = DatasetDefinition(**data)
        assert len(defn.known_limitations) == 2

    def test_empty_known_limitations_default(self):
        defn = DatasetDefinition(**_minimal_valid())
        assert defn.known_limitations == []

    def test_open_scientific_decisions_list(self):
        data = _minimal_valid()
        data["open_scientific_decisions"] = [
            {"id": "OSD-005", "description": "Grid specification"},
        ]
        defn = DatasetDefinition(**data)
        assert len(defn.open_scientific_decisions) == 1
        assert defn.open_scientific_decisions[0].id == "OSD-005"

    def test_extra_fields_ignored(self):
        """Unknown YAML keys must not cause a validation error."""
        data = _minimal_valid()
        data["some_future_field"] = "some value"
        data["another_field"] = {"nested": "dict"}
        defn = DatasetDefinition(**data)
        assert defn.dataset_id == "testds"

    def test_scientific_notes_accepted(self):
        data = _minimal_valid()
        data["scientific_notes"] = "Important scientific background."
        defn = DatasetDefinition(**data)
        assert defn.scientific_notes == "Important scientific background."


# ---------------------------------------------------------------------------
# OSD ID format
# ---------------------------------------------------------------------------

class TestOpenScientificDecision:
    def test_valid_osd_id(self):
        osd = OpenScientificDecision(id="OSD-001", description="test")
        assert osd.id == "OSD-001"

    def test_invalid_osd_id_missing_prefix(self):
        with pytest.raises(ValidationError):
            OpenScientificDecision(id="001", description="test")

    def test_invalid_osd_id_wrong_case(self):
        with pytest.raises(ValidationError):
            OpenScientificDecision(id="osd-001", description="test")

    def test_invalid_osd_id_no_digits(self):
        with pytest.raises(ValidationError):
            OpenScientificDecision(id="OSD-", description="test")

    def test_three_digit_osd_id(self):
        osd = OpenScientificDecision(id="OSD-017", description="test")
        assert osd.id == "OSD-017"


# ---------------------------------------------------------------------------
# OceanEmbedDomain
# ---------------------------------------------------------------------------

class TestOceanEmbedDomain:
    def test_valid_domain(self):
        domain = OceanEmbedDomain(lat_min=5.0, lat_max=30.0, lon_min=45.0, lon_max=105.0)
        assert domain.lat_min == 5.0

    def test_lat_min_must_be_less_than_lat_max(self):
        with pytest.raises(ValidationError):
            OceanEmbedDomain(lat_min=30.0, lat_max=5.0, lon_min=45.0, lon_max=105.0)

    def test_lat_min_equal_to_lat_max_rejected(self):
        with pytest.raises(ValidationError):
            OceanEmbedDomain(lat_min=10.0, lat_max=10.0, lon_min=45.0, lon_max=105.0)

    def test_lon_min_must_be_less_than_lon_max(self):
        with pytest.raises(ValidationError):
            OceanEmbedDomain(lat_min=5.0, lat_max=30.0, lon_min=105.0, lon_max=45.0)


# ---------------------------------------------------------------------------
# StorageInfo path validation
# ---------------------------------------------------------------------------

class TestStorageInfoPaths:
    def test_valid_template_with_placeholder(self):
        s = StorageInfo(raw_path_template="${data_root}/raw/oisst/")
        assert s.raw_path_template == "${data_root}/raw/oisst/"

    def test_windows_absolute_path_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            StorageInfo(raw_path_template=r"D:\OceanEmbed\data\raw\oisst")
        assert "hard-coded" in str(exc_info.value).lower() or "absolute" in str(exc_info.value).lower()

    def test_unix_absolute_path_without_placeholder_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            StorageInfo(raw_path_template="/home/user/data/raw/oisst")
        assert "absolute" in str(exc_info.value).lower() or "hard-coded" in str(exc_info.value).lower()

    def test_none_path_accepted(self):
        s = StorageInfo(raw_path_template=None)
        assert s.raw_path_template is None

    def test_unverified_sentinel_accepted(self):
        s = StorageInfo(raw_path_template="UNVERIFIED")
        assert s.raw_path_template == "UNVERIFIED"


# ---------------------------------------------------------------------------
# VariableInfo
# ---------------------------------------------------------------------------

class TestVariableInfo:
    def test_valid_variable(self):
        v = VariableInfo(
            name="sst",
            long_name="Sea surface temperature",
            oceanembed_standard_name="sst",
            native_units="degrees_C",
            oceanembed_units="UNRESOLVED",
        )
        assert v.name == "sst"
        assert v.oceanembed_units == "UNRESOLVED"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            VariableInfo(long_name="Sea surface temperature")

    def test_all_optional_except_name(self):
        v = VariableInfo(name="thetao")
        assert v.native_units is None
        assert v.oceanembed_units is None


# ---------------------------------------------------------------------------
# AcquisitionInfo
# ---------------------------------------------------------------------------

class TestAcquisitionInfo:
    def test_auth_required_true(self):
        a = AcquisitionInfo(authentication_required=True, authentication_type="CMEMS_account")
        assert a.authentication_required is True

    def test_auth_not_required_false(self):
        a = AcquisitionInfo(authentication_required=False)
        assert a.authentication_required is False

    def test_legacy_script_absent(self):
        a = AcquisitionInfo(legacy_script="ABSENT", legacy_script_audited=False)
        assert a.legacy_script == "ABSENT"
        assert a.legacy_script_audited is False
