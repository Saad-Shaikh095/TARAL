"""
OceanEmbed dataset registry schemas.

Provides Pydantic models for validating dataset registry YAML definitions.

The schema validates structure and required identity fields.  Fields whose
values are genuinely unknown carry the sentinel strings 'UNVERIFIED' or
'UNRESOLVED'; these are preserved as-is rather than coerced into typed
values, because premature typing would imply false precision.

Design principles applied here:
- Required fields are those needed to identify a dataset unambiguously.
- All metadata fields are Optional to accommodate datasets at different
  stages of audit completeness.
- Scientific decision sentinels ('UNRESOLVED') are valid string values.
- The schema does NOT encode any scientific defaults.
- Extra/unknown YAML keys are ignored (extra='allow') to allow registry
  YAML files to carry richer documentation without breaking validation.

See docs/datasets/DATASET_REGISTRY.md for the full contract.
"""

from oceanembed.datasets.schemas.registry import (
    DatasetDefinition,
    DatasetRole,
    AcquisitionMechanism,
    ProviderInfo,
    SpatialInfo,
    TemporalInfo,
    VariableInfo,
    FileFormatInfo,
    AcquisitionInfo,
    StorageInfo,
    OpenScientificDecision,
)

__all__ = [
    "DatasetDefinition",
    "DatasetRole",
    "AcquisitionMechanism",
    "ProviderInfo",
    "SpatialInfo",
    "TemporalInfo",
    "VariableInfo",
    "FileFormatInfo",
    "AcquisitionInfo",
    "StorageInfo",
    "OpenScientificDecision",
]
