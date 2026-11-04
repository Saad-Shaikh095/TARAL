"""
OceanEmbed API — Pydantic request/response models.

These are API-layer models only. They do NOT expose internal Python objects.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Dataset listing
# ---------------------------------------------------------------------------

class DatasetSummary(BaseModel):
    dataset_id: str
    product_name: str
    role: str
    provider_name: str | None
    native_temporal_resolution: str | None
    native_spatial_resolution_degrees: float | None
    primary_variables: list[str]
    temporal_coverage_start: str | None
    temporal_coverage_end: str | None
    engineering_status: str
    adapter_implemented: bool
    open_decisions_count: int


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]
    total: int


# ---------------------------------------------------------------------------
# Dataset detail
# ---------------------------------------------------------------------------

class DatasetDetailResponse(BaseModel):
    dataset_id: str
    product_name: str
    product_id: str | None
    version: str | None
    role: str
    role_description: str | None
    provider_name: str | None
    provider_url: str | None
    native_temporal_resolution: str | None
    native_spatial_resolution_degrees: float | None
    temporal_coverage_start: str | None
    temporal_coverage_end: str | None
    primary_variables: list[dict[str, Any]]
    file_format: str | None
    authentication_required: bool | None
    known_limitations: list[str]
    open_scientific_decisions: list[dict[str, str]]
    scientific_notes: str | None
    engineering_status: str
    adapter_implemented: bool


# ---------------------------------------------------------------------------
# Acquisition jobs
# ---------------------------------------------------------------------------

class AcquisitionJobRequest(BaseModel):
    dataset_id: str = Field(..., description="Dataset to acquire (e.g. 'oisst')")
    start_date: date = Field(..., description="First date to acquire (YYYY-MM-DD)")
    end_date: date = Field(..., description="Last date to acquire (YYYY-MM-DD)")

    model_config = {"json_schema_extra": {
        "example": {
            "dataset_id": "oisst",
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
        }
    }}


class AcquisitionJobResponse(BaseModel):
    job_id: str
    dataset_id: str
    start_date: date
    end_date: date
    status: str
    files_requested: int
    files_completed: int
    files_failed: int
    bytes_downloaded: int
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    elapsed_seconds: float | None = None


class JobLogEntry(BaseModel):
    ts: str
    level: str
    message: str
    extra: dict[str, Any] = Field(default_factory=dict)


class JobLogsResponse(BaseModel):
    job_id: str
    logs: list[JobLogEntry]
    total: int


class CancelJobResponse(BaseModel):
    job_id: str
    cancelled: bool
    message: str
