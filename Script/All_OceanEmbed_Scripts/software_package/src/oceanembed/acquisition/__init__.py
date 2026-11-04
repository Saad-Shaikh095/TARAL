"""
OceanEmbed acquisition module.

Provides dataset downloading, manifest tracking, job management, and
retry logic for acquiring OceanEmbed source datasets.
"""

from oceanembed.acquisition.engine import (
    AcquisitionJob,
    JobManager,
    JobStatus,
    run_acquisition_job,
    download_file,
)
from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

__all__ = [
    "AcquisitionJob",
    "JobManager",
    "JobStatus",
    "run_acquisition_job",
    "download_file",
    "ManifestStore",
    "ManifestEntry",
    "ManifestStatus",
]
