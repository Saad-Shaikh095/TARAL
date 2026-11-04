"""
OceanEmbed Dataset Registry — Pydantic schema definitions.

This module defines the validated schema for OceanEmbed dataset registry entries.
Every YAML file in configs/datasets/ is validated against DatasetDefinition when
the registry loader processes it.

Design decisions
----------------
1.  Only ``dataset_id``, ``product_name``, and ``role`` are strictly required.
    Everything else is Optional because registry entries are built incrementally
    as datasets are audited.

2.  Fields that carry 'UNVERIFIED' or 'UNRESOLVED' as their value are typed as
    Optional[str].  These sentinel strings are preserved — they communicate
    meaningful engineering state (the field has been considered but cannot yet
    be confirmed or decided).

3.  Pydantic is configured with ``extra='allow'`` on all models.  Registry YAML
    files carry human-readable notes and dataset-specific fields that do not
    belong in the shared schema.  Silently ignoring unknown keys is safer than
    rejecting valid registry entries.

4.  No scientific defaults are encoded.  Fields like temporal resolution or
    regridding method are Optional[str]; their absence does not trigger any
    fallback behaviour.

5.  The ``open_scientific_decisions`` list is the machine-readable hook from
    a registry entry back to docs/architecture/OPEN_SCIENTIFIC_DECISIONS.md.

Scientific guardrail
--------------------
This schema must never be extended to include default values for fields that
represent scientific choices (regridding method, aggregation method, fill values,
etc.).  Any such addition requires an explicit scientific decision per
docs/development/SCIENTIFIC_DECISION_POLICY.md.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DatasetRole(str, Enum):
    """Role of a dataset in the OceanEmbed pipeline."""

    SURFACE_INPUT = "surface_input"
    TARGET = "target"
    VALIDATION = "validation"


class AcquisitionMechanism(str, Enum):
    """Known acquisition mechanism identifiers."""

    CMEMS_COPERNICUSMARINE = "CMEMS_copernicusmarine_client"
    NOAA_ERDDAP_OR_OPENDAP = "NOAA_ERDDAP_or_OPENDAP"
    NASA_PO_DAAC_OR_ERDDAP = "NASA_PO_DAAC_or_ERDDAP"
    NASA_PO_DAAC_HTTPS_OR_ERDDAP = "NASA_PO_DAAC_HTTPS_or_ERDDAP"
    RSS_DIRECT_DOWNLOAD = "RSS_direct_download"
    GDAC_FTP_OR_ERDDAP = "GDAC_FTP_or_ERDDAP"
    UNVERIFIED = "UNVERIFIED"


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class ProviderInfo(BaseModel):
    """Dataset source provider information."""

    model_config = {"extra": "allow"}

    name: str = Field(..., description="Provider organisation name")
    url: str | None = Field(None, description="Provider homepage URL")


class OceanEmbedDomain(BaseModel):
    """Geographic bounding box for OceanEmbed domain subsetting."""

    model_config = {"extra": "allow"}

    lat_min: float = Field(5.0, ge=-90.0, le=90.0)
    lat_max: float = Field(30.0, ge=-90.0, le=90.0)
    lon_min: float = Field(45.0, ge=-180.0, le=360.0)
    lon_max: float = Field(105.0, ge=-180.0, le=360.0)

    @model_validator(mode="after")
    def check_lat_order(self) -> "OceanEmbedDomain":
        if self.lat_min >= self.lat_max:
            raise ValueError(
                f"lat_min ({self.lat_min}) must be less than lat_max ({self.lat_max})"
            )
        return self

    @model_validator(mode="after")
    def check_lon_order(self) -> "OceanEmbedDomain":
        if self.lon_min >= self.lon_max:
            raise ValueError(
                f"lon_min ({self.lon_min}) must be less than lon_max ({self.lon_max})"
            )
        return self


class SpatialInfo(BaseModel):
    """Native spatial characteristics of a dataset."""

    model_config = {"extra": "allow"}

    # 'structure' distinguishes gridded datasets from point/profile datasets.
    # Gridded datasets (OISST, GLORYS, etc.) have resolution_degrees.
    # Profile datasets (Argo) have structure = 'point_profiles'.
    structure: str | None = Field(
        None,
        description=(
            "Spatial structure: 'regular_lat_lon' for gridded, "
            "'point_profiles' for sparse observational datasets."
        ),
    )
    grid_type: str | None = Field(None, description="Grid type identifier")
    resolution_degrees: float | None = Field(
        None,
        gt=0.0,
        description="Native spatial resolution in degrees (gridded datasets only)",
    )
    resolution_description: str | None = Field(
        None, description="Human-readable resolution description"
    )
    oceanembed_domain: OceanEmbedDomain | None = Field(
        None, description="OceanEmbed bounding box for domain subsetting"
    )


class TemporalInfo(BaseModel):
    """Native temporal characteristics of a dataset."""

    model_config = {"extra": "allow"}

    resolution: str | None = Field(
        None,
        description=(
            "Native temporal resolution (e.g. 'daily', 'monthly', '6-hourly'). "
            "May be 'UNVERIFIED' if not yet confirmed."
        ),
    )
    coverage_start: str | None = Field(
        None, description="Nominal start of data coverage (ISO 8601 date string)"
    )
    coverage_end: str | None = Field(
        None,
        description=(
            "Nominal end of data coverage (ISO 8601 date string, or descriptive "
            "string such as 'present' or 'UNVERIFIED')."
        ),
    )
    oceanembed_period_start: str | None = Field(
        None, description="Start of the OceanEmbed historical period"
    )
    oceanembed_period_end: str | None = Field(
        None, description="End of the OceanEmbed historical period"
    )


class VariableInfo(BaseModel):
    """A single variable within a dataset."""

    model_config = {"extra": "allow"}

    name: str = Field(..., description="Variable short name")
    long_name: str | None = Field(None, description="Human-readable variable name")
    native_variable_name: str | None = Field(
        None,
        description=(
            "Variable name as it appears in the source files. "
            "May be 'UNVERIFIED' if not yet confirmed from actual files."
        ),
    )
    oceanembed_standard_name: str | None = Field(
        None, description="OceanEmbed internal standard name for this variable"
    )
    native_units: str | None = Field(None, description="Units in the source files")
    oceanembed_units: str | None = Field(
        None,
        description=(
            "Units used in the OceanEmbed pipeline. May be 'UNRESOLVED' if "
            "the unit convention has not yet been scientifically approved."
        ),
    )


class VariablesSection(BaseModel):
    """Variables section of a registry entry."""

    model_config = {"extra": "allow"}

    primary: list[VariableInfo] = Field(
        default_factory=list,
        description="Primary variables used as OceanEmbed inputs or targets",
    )


class FileFormatInfo(BaseModel):
    """File format characteristics of a dataset."""

    model_config = {"extra": "allow"}

    type: str | None = Field(None, description="File format type (e.g. 'NetCDF4')")
    extension: str | None = Field(None, description="File extension (e.g. '.nc')")


class AcquisitionInfo(BaseModel):
    """Acquisition characteristics and access information."""

    model_config = {"extra": "allow"}

    mechanism: str | None = Field(
        None,
        description=(
            "Acquisition mechanism identifier. "
            "See AcquisitionMechanism enum for known values."
        ),
    )
    authentication_required: bool | None = Field(
        None,
        description="Whether authentication credentials are required for download",
    )
    authentication_type: str | None = Field(
        None,
        description="Authentication type (e.g. 'CMEMS_account', 'NASA_Earthdata_login')",
    )
    legacy_script: str | None = Field(
        None,
        description=(
            "Name of legacy acquisition script, 'ABSENT' if not recovered, "
            "or filename if staged in legacy/acquisition/."
        ),
    )
    legacy_script_audited: bool | None = Field(
        None, description="Whether the legacy script has been formally audited"
    )


class StorageInfo(BaseModel):
    """Expected local storage layout for a dataset."""

    model_config = {"extra": "allow"}

    raw_path_template: str | None = Field(
        None,
        description=(
            "Path template for raw downloaded files. Must use ${data_root} "
            "placeholder — never a hard-coded machine path."
        ),
    )
    processed_path_template: str | None = Field(
        None,
        description=(
            "Path template for processed files. Must use ${data_root} placeholder."
        ),
    )

    @field_validator("raw_path_template", "processed_path_template", mode="before")
    @classmethod
    def no_hardcoded_paths(cls, v: Any) -> Any:
        """Reject path templates that appear to contain hard-coded absolute paths."""
        if v is None or not isinstance(v, str):
            return v
        # Allow sentinel strings
        if v in ("UNVERIFIED", "UNRESOLVED"):
            return v
        # Reject Windows-style absolute paths (D:\..., C:\...)
        import re
        if re.match(r"^[A-Za-z]:\\", v):
            raise ValueError(
                f"Path template contains a hard-coded absolute path: {v!r}. "
                "Use ${data_root} placeholder instead. "
                "See docs/development/DATA_BOUNDARY.md."
            )
        # Reject Unix-style absolute paths that don't use the placeholder
        if v.startswith("/") and "${data_root}" not in v:
            raise ValueError(
                f"Path template contains a hard-coded absolute path: {v!r}. "
                "Use ${data_root} placeholder instead."
            )
        return v


class OpenScientificDecision(BaseModel):
    """A reference to an open scientific decision affecting this dataset."""

    model_config = {"extra": "allow"}

    id: str = Field(
        ...,
        pattern=r"^OSD-\d+$",
        description="Open Scientific Decision identifier (e.g. 'OSD-001')",
    )
    description: str = Field(
        ..., description="Short description of the decision and why it is open"
    )


# ---------------------------------------------------------------------------
# Top-level DatasetDefinition
# ---------------------------------------------------------------------------


class DatasetDefinition(BaseModel):
    """
    Machine-readable registry definition for a single OceanEmbed source dataset.

    Every YAML file in configs/datasets/ must validate against this model.

    Required fields
    ---------------
    - dataset_id  : unique lowercase identifier (e.g. 'oisst', 'glorys')
    - product_name: human-readable product name
    - role        : role in the OceanEmbed pipeline

    All other fields are Optional.  Their absence indicates that the information
    has not yet been verified or that it is not applicable for this dataset.

    Scientific guardrail
    --------------------
    This class must NOT define default values for scientifically meaningful
    parameters (temporal resolution, regridding method, unit conventions, etc.).
    Optional fields default to None, not to a specific scientific assumption.
    """

    model_config = {"extra": "allow"}

    # ------------------------------------------------------------------
    # Identity (required)
    # ------------------------------------------------------------------
    dataset_id: str = Field(
        ...,
        pattern=r"^[a-z][a-z0-9_]*$",
        description=(
            "Unique lowercase identifier for this dataset. "
            "Must match the registry YAML filename stem."
        ),
    )
    product_name: str = Field(
        ..., description="Human-readable product name from the source provider"
    )
    role: DatasetRole = Field(
        ...,
        description=(
            "Role in the OceanEmbed pipeline: "
            "'surface_input', 'target', or 'validation'."
        ),
    )

    # ------------------------------------------------------------------
    # Identity (optional)
    # ------------------------------------------------------------------
    product_id: str | None = Field(
        None,
        description=(
            "Source product identifier as used by the provider's catalogue. "
            "May be 'UNVERIFIED' if not yet confirmed."
        ),
    )
    version: str | None = Field(
        None,
        description=(
            "Dataset version string as reported by the source provider. "
            "May be 'UNVERIFIED'."
        ),
    )
    registry_version: str | None = Field(
        None,
        description="Version of this registry entry (semver string)",
    )
    role_description: str | None = Field(
        None, description="Free-text description of the dataset's role"
    )

    # ------------------------------------------------------------------
    # Source / provider
    # ------------------------------------------------------------------
    provider: ProviderInfo | None = Field(
        None, description="Source provider information"
    )

    # ------------------------------------------------------------------
    # Spatial / temporal
    # ------------------------------------------------------------------
    spatial: SpatialInfo | None = Field(
        None, description="Native spatial characteristics"
    )
    temporal: TemporalInfo | None = Field(
        None, description="Native temporal characteristics"
    )

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    variables: VariablesSection | None = Field(
        None, description="Variable definitions"
    )

    # ------------------------------------------------------------------
    # File format
    # ------------------------------------------------------------------
    file_format: FileFormatInfo | None = Field(
        None, description="Source file format characteristics"
    )

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    acquisition: AcquisitionInfo | None = Field(
        None, description="Acquisition mechanism and access characteristics"
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    storage: StorageInfo | None = Field(
        None, description="Expected local storage path templates"
    )

    # ------------------------------------------------------------------
    # Known limitations and open decisions
    # ------------------------------------------------------------------
    known_limitations: list[str] = Field(
        default_factory=list,
        description="List of known limitations documented for this dataset",
    )
    open_scientific_decisions: list[OpenScientificDecision] = Field(
        default_factory=list,
        description=(
            "Open scientific decisions that affect this dataset. "
            "References docs/architecture/OPEN_SCIENTIFIC_DECISIONS.md."
        ),
    )
    scientific_notes: str | None = Field(
        None,
        description="Free-text scientific background notes about this dataset",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_lowercase(cls, v: str) -> str:
        """dataset_id must be lowercase."""
        if v != v.lower():
            raise ValueError(
                f"dataset_id must be lowercase, got: {v!r}"
            )
        return v
