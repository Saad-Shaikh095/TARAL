"""
OceanEmbed dataset adapter contract.

This module defines the abstract base class that every dataset-specific
acquisition adapter must implement.

An adapter is responsible for SOURCE-SPECIFIC BEHAVIOR only:
- constructing download requests for a given dataset and date range
- describing what the adapter will retrieve (metadata, not data)
- declaring the source product, variables, and spatial region
- producing the information needed by the acquisition engine to execute
  a download job

An adapter is NOT responsible for:
- executing HTTP/FTP downloads  (→ acquisition engine)
- retrying failed requests      (→ acquisition engine)
- writing manifest records      (→ acquisition engine)
- regridding or temporal resampling  (→ processing layer)
- filling missing values             (→ processing layer, requires science decision)
- any scientific harmonization       (→ processing layer, requires science decision)

See docs/datasets/DATASET_REGISTRY.md §5 for the full adapter contract.
"""

from oceanembed.datasets.adapters.base import (
    DatasetAdapterBase,
    AcquisitionRequest,
    SourceDescription,
)

__all__ = [
    "DatasetAdapterBase",
    "AcquisitionRequest",
    "SourceDescription",
]
