"""
OceanEmbed FastAPI application.

Provides the REST API for the OceanEmbed demo vertical slice.

Endpoints
---------
GET  /api/health
GET  /api/datasets
GET  /api/datasets/{dataset_id}
POST /api/acquisition/jobs
GET  /api/acquisition/jobs/{job_id}
GET  /api/acquisition/jobs/{job_id}/logs
POST /api/acquisition/jobs/{job_id}/cancel

Run (development)
-----------------
    uvicorn oceanembed.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env FIRST — before any import that triggers copernicusmarine,
# which reads its credential env vars (COPERNICUSMARINE_SERVICE_USERNAME /
# COPERNICUSMARINE_SERVICE_PASSWORD) at module-import time via os.getenv().
# Existing os.environ values are NEVER overridden — shell exports win.
# ---------------------------------------------------------------------------
def _bootstrap_dotenv() -> None:
    """
    Minimal .env loader: reads KEY=VALUE from <repo_root>/.env into os.environ.
    No external dependencies.  Runs once at app import time.
    """
    here = Path(__file__).resolve()
    repo_root: Path | None = None
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    if repo_root is None:
        return
    dotenv = repo_root / ".env"
    if not dotenv.exists():
        return
    with open(dotenv, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val

_bootstrap_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from oceanembed.__version__ import __version__
from oceanembed.acquisition.engine import JobManager, JobStatus
from oceanembed.api.models import (
    AcquisitionJobRequest,
    AcquisitionJobResponse,
    CancelJobResponse,
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetSummary,
    HealthResponse,
    JobLogEntry,
    JobLogsResponse,
)
from oceanembed.config.loader import get_settings
from oceanembed.datasets.registry import load_registry
from oceanembed.datasets.schemas.registry import DatasetDefinition

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OceanEmbed API",
    description=(
        "Dataset acquisition and management API for the OceanEmbed "
        "surface-to-subsurface ocean temperature reconstruction system."
    ),
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

def _get_allowed_origins() -> list[str]:
    """
    Resolve CORS allowed origins.

    LOCAL DEVELOPMENT:
        Always includes localhost:3000 / 127.0.0.1:3000 for the Next.js dev server.

    PRODUCTION (Vercel — same-origin deployment):
        The Next.js frontend and Python serverless function are served from the
        same Vercel domain. All /api/* requests are same-origin, so the CORS
        middleware is never triggered for browser requests from the frontend.
        OCEANEMBED_ALLOWED_ORIGINS is not required in this configuration.

    PRODUCTION (separate frontend/backend domains):
        Set OCEANEMBED_ALLOWED_ORIGINS to a comma-separated list of allowed origins:
            OCEANEMBED_ALLOWED_ORIGINS=https://oceanembed.vercel.app

    Do NOT set allow_origins=["*"] — that is never correct for this API.
    """
    defaults = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    extra = os.environ.get("OCEANEMBED_ALLOWED_ORIGINS", "")
    if extra:
        extras = [o.strip() for o in extra.split(",") if o.strip()]
        # Merge without duplicates, preserving order
        for origin in extras:
            if origin not in defaults:
                defaults.append(origin)
    return defaults


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Registry + adapter registry (loaded at startup)
# ---------------------------------------------------------------------------

_registry = None
_IMPLEMENTED_ADAPTERS: set[str] = {"oisst", "globcurrent", "duacs", "oscar", "oisss", "ccmp", "glorys", "argo"}


def get_registry():
    global _registry
    if _registry is None:
        settings = get_settings()
        _registry = load_registry(settings.registry_dir)
    return _registry


def get_adapter(dataset_id: str):
    """Return the adapter for a dataset, or None if not implemented."""
    if dataset_id == "oisst":
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        return OISSTAdapter()
    if dataset_id == "argo":
        from oceanembed.datasets.adapters.argo import ArgoAdapter
        from oceanembed.config.loader import _find_repo_root
        repo_root = _find_repo_root()
        index_candidates = [
            repo_root / "argo index" / "ar_index_global_prof.txt",
            repo_root.parent / "argo index" / "ar_index_global_prof.txt",
        ]
        index_path = next((p for p in index_candidates if p.exists()), None)
        if index_path is None:
            import logging
            logging.getLogger(__name__).warning(
                "ArgoAdapter: ar_index_global_prof.txt not found. "
                "Place it under '<repo>/argo index/'."
            )
            return None
        return ArgoAdapter(index_path=index_path)
    if dataset_id == "glorys":
        from oceanembed.datasets.adapters.glorys import GLORYSAdapter
        try:
            return GLORYSAdapter()
        except RuntimeError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "GLORYSAdapter init failed (credentials missing?): %s", exc
            )
            return None
    if dataset_id == "globcurrent":
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        return GlobCurrentAdapter()
    if dataset_id == "duacs":
        from oceanembed.datasets.adapters.duacs import DUACSAdapter
        return DUACSAdapter()
    if dataset_id == "oscar":
        from oceanembed.datasets.adapters.oscar import OSCARAdapter
        return OSCARAdapter()
    if dataset_id == "oisss":
        from oceanembed.datasets.adapters.oisss import OISSSAdapter
        return OISSSAdapter()
    if dataset_id == "ccmp":
        from oceanembed.datasets.adapters.ccmp import CCMPAdapter
        return CCMPAdapter()
    return None


def _engineering_status(dataset_id: str) -> str:
    if dataset_id == "argo":
        return "ACQUISITION_READY"
    if dataset_id == "oisst":
        return "DEMO_READY"
    if dataset_id == "glorys":
        return "ACQUISITION_READY"
    if dataset_id == "globcurrent":
        return "ACQUISITION_READY"
    if dataset_id == "duacs":
        return "ACQUISITION_READY"
    if dataset_id == "oscar":
        return "ACQUISITION_READY"
    if dataset_id == "oisss":
        return "ACQUISITION_READY"
    if dataset_id == "ccmp":
        return "ACQUISITION_READY"
    return "ADAPTER_PENDING"


def _defn_to_summary(defn: DatasetDefinition) -> DatasetSummary:
    prim_vars = []
    if defn.variables and defn.variables.primary:
        prim_vars = [v.name for v in defn.variables.primary]

    return DatasetSummary(
        dataset_id=defn.dataset_id,
        product_name=defn.product_name,
        role=defn.role.value,
        provider_name=defn.provider.name if defn.provider else None,
        native_temporal_resolution=(
            defn.temporal.resolution if defn.temporal else None
        ),
        native_spatial_resolution_degrees=(
            defn.spatial.resolution_degrees if defn.spatial else None
        ),
        primary_variables=prim_vars,
        temporal_coverage_start=(
            defn.temporal.coverage_start if defn.temporal else None
        ),
        temporal_coverage_end=(
            defn.temporal.coverage_end if defn.temporal else None
        ),
        engineering_status=_engineering_status(defn.dataset_id),
        adapter_implemented=defn.dataset_id in _IMPLEMENTED_ADAPTERS,
        open_decisions_count=len(defn.open_scientific_decisions),
    )


# ---------------------------------------------------------------------------
# Exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check backend logs."},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@app.get("/api/datasets", response_model=DatasetListResponse, tags=["Datasets"])
async def list_datasets():
    """List all registered datasets with engineering status."""
    registry = get_registry()
    summaries = [_defn_to_summary(defn) for _, defn in registry]
    return DatasetListResponse(datasets=summaries, total=len(summaries))


@app.get(
    "/api/datasets/{dataset_id}",
    response_model=DatasetDetailResponse,
    tags=["Datasets"],
)
async def get_dataset(dataset_id: str):
    """Get detailed information about a specific dataset."""
    registry = get_registry()
    try:
        defn = registry.get(dataset_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    prim_vars = []
    if defn.variables and defn.variables.primary:
        prim_vars = [
            {
                "name": v.name,
                "long_name": v.long_name,
                "native_units": v.native_units,
                "oceanembed_units": v.oceanembed_units,
            }
            for v in defn.variables.primary
        ]

    osds = [
        {"id": osd.id, "description": osd.description}
        for osd in defn.open_scientific_decisions
    ]

    return DatasetDetailResponse(
        dataset_id=defn.dataset_id,
        product_name=defn.product_name,
        product_id=defn.product_id,
        version=defn.version,
        role=defn.role.value,
        role_description=defn.role_description,
        provider_name=defn.provider.name if defn.provider else None,
        provider_url=defn.provider.url if defn.provider else None,
        native_temporal_resolution=(
            defn.temporal.resolution if defn.temporal else None
        ),
        native_spatial_resolution_degrees=(
            defn.spatial.resolution_degrees if defn.spatial else None
        ),
        temporal_coverage_start=(
            defn.temporal.coverage_start if defn.temporal else None
        ),
        temporal_coverage_end=(
            defn.temporal.coverage_end if defn.temporal else None
        ),
        primary_variables=prim_vars,
        file_format=(
            defn.file_format.type if defn.file_format else None
        ),
        authentication_required=(
            defn.acquisition.authentication_required
            if defn.acquisition
            else None
        ),
        known_limitations=defn.known_limitations,
        open_scientific_decisions=osds,
        scientific_notes=defn.scientific_notes,
        engineering_status=_engineering_status(defn.dataset_id),
        adapter_implemented=defn.dataset_id in _IMPLEMENTED_ADAPTERS,
    )


# ---------------------------------------------------------------------------
# Acquisition jobs
# ---------------------------------------------------------------------------

@app.post(
    "/api/acquisition/jobs",
    response_model=AcquisitionJobResponse,
    status_code=202,
    tags=["Acquisition"],
)
async def create_acquisition_job(body: AcquisitionJobRequest):
    """
    Create and start an acquisition job.

    Only OISST has a functional adapter. Other datasets return a clear status
    indicating the adapter is not yet implemented.
    """
    # Validate dataset exists
    registry = get_registry()
    try:
        registry.get(body.dataset_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{body.dataset_id}' not found in registry.",
        )

    # Validate dates
    if body.start_date > body.end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must be on or before end_date.",
        )

    # Check adapter availability first, then apply adapter-specific range limit
    adapter = get_adapter(body.dataset_id)
    if adapter is None:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Acquisition adapter for '{body.dataset_id}' is not yet implemented. "
                f"Status: ADAPTER_PENDING. "
                f"See docs/datasets/DATASET_ENGINEERING_STATUS.md."
            ),
        )

    # Date-range limit: OISST (daily files) is capped at 31 days per job to
    # keep demo acquisition manageable. GlobCurrent (yearly chunks) allows up
    # to 366 days (one full year) per job. The limit is adapter-aware.
    date_range = (body.end_date - body.start_date).days + 1
    max_days = getattr(adapter, "_max_job_days", 31)
    if date_range > max_days:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Date range ({date_range} days) exceeds the maximum allowed "
                f"for {body.dataset_id} ({max_days} days per job)."
            ),
        )

    # Resolve data_root
    settings = get_settings()
    if settings.data_root is None:
        # Fall back to a sibling directory of the repository root.
        # _find_repo_root() walks up to the directory containing pyproject.toml,
        # so this resolves to <workspace>/oceanembed-data regardless of how
        # many src/ subdirectories are between app.py and the project root.
        from oceanembed.config.loader import _find_repo_root
        repo_root = _find_repo_root()
        data_root = repo_root.parent / "oceanembed-data"
    else:
        data_root = settings.data_root

    # Create and start job
    manager = JobManager.get()
    job = manager.create_job(
        dataset_id=body.dataset_id,
        start_date=body.start_date,
        end_date=body.end_date,
        data_root=data_root,
    )
    manager.start_job(job, adapter)

    return _job_to_response(job)


@app.get(
    "/api/acquisition/jobs/{job_id}",
    response_model=AcquisitionJobResponse,
    tags=["Acquisition"],
)
async def get_job_status(job_id: str):
    """Get the status and progress of an acquisition job."""
    manager = JobManager.get()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_to_response(job)


@app.get(
    "/api/acquisition/jobs/{job_id}/logs",
    response_model=JobLogsResponse,
    tags=["Acquisition"],
)
async def get_job_logs(job_id: str, offset: int = 0):
    """Get log entries for an acquisition job."""
    manager = JobManager.get()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    log_entries = job.logs[offset:]
    parsed = []
    for entry in log_entries:
        extra = {k: v for k, v in entry.items() if k not in ("ts", "level", "message")}
        parsed.append(
            JobLogEntry(
                ts=entry.get("ts", ""),
                level=entry.get("level", "INFO"),
                message=entry.get("message", ""),
                extra=extra,
            )
        )

    return JobLogsResponse(
        job_id=job_id,
        logs=parsed,
        total=len(job.logs),
    )


@app.post(
    "/api/acquisition/jobs/{job_id}/cancel",
    response_model=CancelJobResponse,
    tags=["Acquisition"],
)
async def cancel_job(job_id: str):
    """Cancel a running or queued acquisition job."""
    manager = JobManager.get()
    job = manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    cancelled = manager.cancel_job(job_id)
    if cancelled:
        return CancelJobResponse(
            job_id=job_id,
            cancelled=True,
            message="Job cancellation requested.",
        )
    return CancelJobResponse(
        job_id=job_id,
        cancelled=False,
        message=f"Job cannot be cancelled in state: {job.status.value}",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _job_to_response(job) -> AcquisitionJobResponse:
    elapsed = None
    if job.started_at:
        end = job.completed_at or datetime.now(timezone.utc)
        elapsed = (end - job.started_at).total_seconds()

    return AcquisitionJobResponse(
        job_id=job.job_id,
        dataset_id=job.dataset_id,
        start_date=job.start_date,
        end_date=job.end_date,
        status=job.status.value,
        files_requested=job.files_requested,
        files_completed=job.files_completed,
        files_failed=job.files_failed,
        bytes_downloaded=job.bytes_downloaded,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        elapsed_seconds=elapsed,
    )
