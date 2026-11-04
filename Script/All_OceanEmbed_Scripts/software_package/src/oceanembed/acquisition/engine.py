"""
OceanEmbed Acquisition Engine.

Manages acquisition jobs, downloads source files via HTTPS or via the
copernicusmarine client or the NASA OPeNDAP (earthaccess) client,
computes checksums, and records results in the manifest.

Architecture
------------
JobManager          — singleton in-process job registry (no external queue)
AcquisitionJob      — dataclass tracking one job's state and logs
download_file       — async HTTPS downloader with retry, MD5, skip-if-valid
run_cmems_subset    — sync-in-thread CMEMS subset executor with MD5, skip-if-valid
run_oscar_subset    — sync-in-thread NASA OPeNDAP executor with MD5, skip-if-valid
run_acquisition_job — async task that drives adapter → downloader → manifest

Dispatch
--------
run_acquisition_job dispatches each AcquisitionRequest by inspecting url_or_query:
  - Starts with "cmems://"  → run_cmems_subset() (via asyncio.to_thread)
  - Starts with "oscar://"  → run_oscar_subset() (via asyncio.to_thread)
  - Otherwise               → download_file() (plain HTTPS, e.g. OISST)

Limitations (demo-grade)
------------------------
- Jobs are not persisted across backend restarts (in-memory dict).
- No external queue (Celery/Redis). Single-process asyncio tasks only.
- MD5 is computed for integrity; not a security guarantee.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from oceanembed.acquisition.manifest import ManifestEntry, ManifestStatus, ManifestStore
from oceanembed.datasets.adapters.base import AcquisitionRequest

# Import copernicusmarine at module level so that patch("copernicusmarine.subset")
# works reliably in tests regardless of import order.  If the package is not
# installed the engine still loads; CMEMS acquisition will fail gracefully when
# actually called (checked credentials before calling subset).
try:
    import copernicusmarine as _copernicusmarine  # noqa: F401
except ImportError:  # pragma: no cover
    _copernicusmarine = None  # type: ignore[assignment]

# Import earthaccess at module level so that patch("earthaccess.*") works
# reliably in tests. If not installed, OSCAR acquisition fails gracefully.
try:
    import earthaccess as _earthaccess  # noqa: F401
except ImportError:  # pragma: no cover
    _earthaccess = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ---------------------------------------------------------------------------
# AcquisitionJob
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionJob:
    job_id: str
    dataset_id: str
    start_date: date
    end_date: date
    data_root: Path
    status: JobStatus = JobStatus.QUEUED
    files_requested: int = 0
    files_completed: int = 0
    files_failed: int = 0
    bytes_downloaded: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def add_log(self, level: str, message: str, extra: dict | None = None) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if extra:
            entry.update(extra)
        self.logs.append(entry)
        log_fn = getattr(logger, level.lower(), logger.info)
        log_fn("[%s] %s", self.job_id, message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "status": self.status.value,
            "files_requested": self.files_requested,
            "files_completed": self.files_completed,
            "files_failed": self.files_failed,
            "bytes_downloaded": self.bytes_downloaded,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# JobManager
# ---------------------------------------------------------------------------

class JobManager:
    """In-process job registry. Jobs NOT persisted across backend restarts."""

    _instance: "JobManager | None" = None

    def __init__(self) -> None:
        self._jobs: dict[str, AcquisitionJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def get(cls) -> "JobManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_job(
        self,
        dataset_id: str,
        start_date: date,
        end_date: date,
        data_root: Path,
    ) -> AcquisitionJob:
        job_id = str(uuid.uuid4())[:8]
        job = AcquisitionJob(
            job_id=job_id,
            dataset_id=dataset_id,
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> AcquisitionJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[AcquisitionJob]:
        return list(self._jobs.values())

    def start_job(self, job: AcquisitionJob, adapter: Any) -> None:
        task = asyncio.create_task(
            run_acquisition_job(job, adapter),
            name=f"job-{job.job_id}",
        )
        self._tasks[job.job_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(job.job_id, None))

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False
        job._cancel_event.set()
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(timezone.utc)
        job.add_log("INFO", "Job cancelled by user.")
        return True


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

async def download_file(
    url: str,
    dest_path: Path,
    job: AcquisitionJob,
    timeout_seconds: int = 300,
    max_retries: int = 3,
) -> tuple[bool, int, str | None]:
    """
    Download *url* to *dest_path* with retry and MD5 checksum.

    Returns
    -------
    (success, bytes_written, md5_hex_or_None)

    Behaviour
    ---------
    - If the destination already exists and is non-empty, compute its MD5,
      log a SKIPPED message, and return immediately without downloading again.
      This makes repeated acquisition requests for the same date idempotent.
    - On download: write to a .tmp side-file, compute MD5 on-the-fly, then
      atomically rename to dest_path.  The .tmp file is always cleaned up on
      failure so a partial download is never treated as complete.
    - Retries: up to *max_retries* attempts with exponential back-off
      (2**attempt seconds).  HTTP 404 is treated as a permanent failure and
      not retried.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Skip — file already present and non-empty
    # ------------------------------------------------------------------
    if dest_path.exists() and dest_path.stat().st_size > 0:
        size = dest_path.stat().st_size
        # Compute checksum of the existing file so the manifest records it.
        md5 = _md5_of_file(dest_path)
        job.add_log(
            "INFO",
            f"File already exists, skipping download: {dest_path.name} "
            f"({size:,} bytes, md5={md5})",
            {"path": str(dest_path), "bytes": size, "md5": md5, "status": "skipped"},
        )
        return True, size, md5

    # ------------------------------------------------------------------
    # Download with retry
    # ------------------------------------------------------------------
    tmp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        if job._cancel_event.is_set():
            job.add_log("INFO", "Download cancelled before attempt.")
            return False, 0, None

        try:
            job.add_log(
                "DOWNLOAD",
                f"Downloading {dest_path.name} (attempt {attempt}/{max_retries})",
                {"url": url, "dest": str(dest_path)},
            )

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30, read=timeout_seconds, write=30, pool=30),
                follow_redirects=True,
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code == 404:
                        job.add_log(
                            "ERROR",
                            f"HTTP 404 — resource not found: {url}",
                            {"http_status": 404},
                        )
                        return False, 0, None

                    if response.status_code != 200:
                        job.add_log(
                            "ERROR",
                            f"HTTP {response.status_code}: {url}",
                            {"http_status": response.status_code},
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return False, 0, None

                    bytes_written = 0
                    hasher = hashlib.md5()

                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            if job._cancel_event.is_set():
                                tmp_path.unlink(missing_ok=True)
                                job.add_log("INFO", "Download cancelled mid-stream.")
                                return False, 0, None
                            f.write(chunk)
                            hasher.update(chunk)
                            bytes_written += len(chunk)

            if bytes_written == 0:
                tmp_path.unlink(missing_ok=True)
                job.add_log("ERROR", f"Downloaded zero bytes from {url}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False, 0, None

            # Atomic rename: only promote once the file is complete
            tmp_path.rename(dest_path)
            md5 = hasher.hexdigest()
            job.add_log(
                "SUCCESS",
                f"Downloaded {dest_path.name}: {bytes_written:,} bytes  md5={md5}",
                {"path": str(dest_path), "bytes": bytes_written, "md5": md5, "status": "ok"},
            )
            return True, bytes_written, md5

        except httpx.TimeoutException:
            tmp_path.unlink(missing_ok=True)
            job.add_log("ERROR", f"Timeout on attempt {attempt}: {url}")
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)

        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            job.add_log("ERROR", f"Error on attempt {attempt}: {exc}")
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            else:
                return False, 0, None

    return False, 0, None


def _md5_of_file(path: Path) -> str:
    """Compute MD5 hex digest of a file in streaming fashion."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# run_cmems_subset
# ---------------------------------------------------------------------------

def run_cmems_subset(
    sentinel_url: str,
    dest_path: Path,
    job: AcquisitionJob,
    max_retries: int = 3,
) -> tuple[bool, int, str | None]:
    """
    Execute a CMEMS copernicusmarine.subset() call and write the output to
    dest_path.  Designed to be called via asyncio.to_thread() from the async
    engine loop.

    Returns
    -------
    (success, bytes_written, md5_hex_or_None)

    Behaviour
    ---------
    - If dest_path already exists and is non-empty, compute its MD5, log
      SKIPPED, and return immediately — same idempotency guarantee as
      download_file().
    - Downloads to a .tmp side-file, then atomically renames to dest_path.
    - Retries up to max_retries times on transient failures (authentication
      errors are not retried).
    - Credential values are NEVER logged.
    """
    import time as _time

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Skip — file already present and non-empty
    # ------------------------------------------------------------------
    if dest_path.exists() and dest_path.stat().st_size > 0:
        size = dest_path.stat().st_size
        md5 = _md5_of_file(dest_path)
        job.add_log(
            "INFO",
            f"File already exists, skipping CMEMS download: {dest_path.name} "
            f"({size:,} bytes, md5={md5})",
            {"path": str(dest_path), "bytes": size, "md5": md5, "status": "skipped"},
        )
        return True, size, md5

    # ------------------------------------------------------------------
    # Parse sentinel URL
    # ------------------------------------------------------------------
    try:
        from oceanembed.datasets.adapters.globcurrent import parse_cmems_sentinel_url
        params = parse_cmems_sentinel_url(sentinel_url)
    except ValueError as exc:
        job.add_log("ERROR", f"Invalid CMEMS sentinel URL: {exc}")
        return False, 0, None

    # ------------------------------------------------------------------
    # Check credentials (presence only — never log values)
    # ------------------------------------------------------------------
    import os as _os
    username = _os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "").strip()
    password = _os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip()
    if not username or not password:
        job.add_log(
            "ERROR",
            "CMEMS credentials not found in environment. "
            "Set COPERNICUSMARINE_SERVICE_USERNAME and "
            "COPERNICUSMARINE_SERVICE_PASSWORD. "
            "See .env.example. No credential values will be logged.",
        )
        return False, 0, None

    tmp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        if job._cancel_event.is_set():
            job.add_log("INFO", "CMEMS download cancelled before attempt.")
            return False, 0, None

        try:
            job.add_log(
                "DOWNLOAD",
                f"CMEMS subset: {dest_path.name} "
                f"(attempt {attempt}/{max_retries}) "
                f"period={params['start'][:10]} → {params['end'][:10]}",
                {
                    "dataset": params["dataset_id"],
                    "period_start": params["start"],
                    "period_end": params["end"],
                    "bbox": (
                        f"lat {params['lat_min']}–{params['lat_max']}, "
                        f"lon {params['lon_min']}–{params['lon_max']}"
                    ),
                    "depth": params["depth"],
                    "variables": params["variables"],
                    "dest": str(dest_path),
                },
            )

            # Use the module-level import so patch("copernicusmarine.subset")
            # works reliably in tests regardless of test execution order.
            import copernicusmarine as copernicusmarine  # noqa: PLC0414 (re-bind for testability)

            # Build the subset kwargs. depth is optional — DUACS and other
            # no-depth datasets set depth=None in the sentinel URL.
            # GLORYS uses depth_min/depth_max for a range bracket.
            subset_kwargs: dict = dict(
                dataset_id=params["dataset_id"],
                variables=params["variables"],
                start_datetime=params["start"],
                end_datetime=params["end"],
                minimum_longitude=params["lon_min"],
                maximum_longitude=params["lon_max"],
                minimum_latitude=params["lat_min"],
                maximum_latitude=params["lat_max"],
                output_directory=str(tmp_path.parent),
                output_filename=tmp_path.stem,
            )
            # Pass dataset_version when specified (e.g. GLORYS pins "202311").
            # GlobCurrent and DUACS omit dataset_version; copernicusmarine
            # auto-selects latest for those. GLORYS requires pinning for
            # reproducibility across the 2012-2024 acquisition period.
            if params.get("dataset_version"):
                subset_kwargs["dataset_version"] = params["dataset_version"]
            # Three depth modes:
            #   depth_min + depth_max present → depth bracket (e.g. GLORYS 0-1100 m)
            #   depth single value present    → single level (e.g. GlobCurrent 0 m)
            #   neither                       → no depth arg (e.g. DUACS)
            if params.get("depth_min") is not None and params.get("depth_max") is not None:
                subset_kwargs["minimum_depth"] = params["depth_min"]
                subset_kwargs["maximum_depth"] = params["depth_max"]
            elif params.get("depth") is not None:
                subset_kwargs["minimum_depth"] = params["depth"]
                subset_kwargs["maximum_depth"] = params["depth"]

            # copernicusmarine.subset() writes the file to the given directory
            # with the given filename. We target the .tmp path, then rename.
            copernicusmarine.subset(**subset_kwargs)

            # copernicusmarine wrote the file as <stem>.nc  (it appended .nc)
            written_path = tmp_path.parent / (tmp_path.stem + ".nc")

            if not written_path.exists() or written_path.stat().st_size == 0:
                written_path.unlink(missing_ok=True)
                job.add_log(
                    "ERROR",
                    f"CMEMS subset completed but output file is missing or empty: "
                    f"{written_path.name}",
                )
                if attempt < max_retries:
                    _time.sleep(2 ** attempt)
                    continue
                return False, 0, None

            # Compute MD5 of the completed file before promoting
            md5 = _md5_of_file(written_path)
            size = written_path.stat().st_size

            # Atomic rename: <stem>.nc → dest_path (the final target name)
            written_path.rename(dest_path)

            job.add_log(
                "SUCCESS",
                f"CMEMS download complete: {dest_path.name} "
                f"({size:,} bytes, md5={md5})",
                {"path": str(dest_path), "bytes": size, "md5": md5, "status": "ok"},
            )
            return True, size, md5

        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            # Also clean up any .nc file copernicusmarine may have written
            (tmp_path.parent / (tmp_path.stem + ".nc")).unlink(missing_ok=True)
            exc_type = type(exc).__name__
            # (copernicusmarine auth errors sometimes echo the username in tracebacks).
            # Log the type and a safe summary only.
            safe_msg = str(exc)
            # Strip anything that looks like a password= or username= parameter from logs
            import re as _re
            safe_msg = _re.sub(
                r'(password|passwd|token|secret|credential)[^\s,;)]*',
                r'\1=<REDACTED>',
                safe_msg,
                flags=_re.IGNORECASE,
            )
            job.add_log(
                "ERROR",
                f"CMEMS error on attempt {attempt} ({exc_type}): {safe_msg}",
            )

            # Authentication errors are permanent — don't retry
            if "401" in safe_msg or "unauthorized" in safe_msg.lower() or "authentication" in safe_msg.lower():
                job.add_log(
                    "ERROR",
                    "CMEMS authentication error. Check credentials in environment. "
                    "Credential values will not be logged.",
                )
                return False, 0, None

            if attempt < max_retries:
                _time.sleep(2 ** attempt)
            else:
                return False, 0, None

    return False, 0, None


# ---------------------------------------------------------------------------
# run_oscar_subset
# ---------------------------------------------------------------------------

def run_oscar_subset(
    sentinel_url: str,
    dest_path: Path,
    job: AcquisitionJob,
    max_retries: int = 5,
) -> tuple[bool, int, str | None]:
    """
    Execute a NASA OPeNDAP download for one OSCAR daily granule.

    Uses earthaccess to create an authenticated requests session, then
    streams the DAP4 NetCDF response to dest_path. Designed to be called
    via asyncio.to_thread() from the async engine loop.

    Returns
    -------
    (success, bytes_written, md5_hex_or_None)

    Behaviour
    ---------
    - If dest_path exists and is non-empty, compute MD5, log SKIPPED, return.
    - Downloads to a .tmp side-file; atomically renames on success.
    - Retries up to max_retries on transient errors; auth failures abort.
    - Credential values are NEVER logged.
    """
    import time as _time
    import re as _re

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Skip — file already present and non-empty
    # ------------------------------------------------------------------
    if dest_path.exists() and dest_path.stat().st_size > 0:
        size = dest_path.stat().st_size
        md5 = _md5_of_file(dest_path)
        job.add_log(
            "INFO",
            f"File already exists, skipping OSCAR download: {dest_path.name} "
            f"({size:,} bytes, md5={md5})",
            {"path": str(dest_path), "bytes": size, "md5": md5, "status": "skipped"},
        )
        return True, size, md5

    # ------------------------------------------------------------------
    # Parse sentinel URL
    # ------------------------------------------------------------------
    try:
        from oceanembed.datasets.adapters.oscar import parse_oscar_sentinel_url
        params = parse_oscar_sentinel_url(sentinel_url)
    except ValueError as exc:
        job.add_log("ERROR", f"Invalid OSCAR sentinel URL: {exc}")
        return False, 0, None

    # ------------------------------------------------------------------
    # Check credentials — presence only, never log values
    # ------------------------------------------------------------------
    import os as _os
    username = _os.environ.get("EARTHDATA_USERNAME", "").strip()
    password = _os.environ.get("EARTHDATA_PASSWORD", "").strip()
    if not username or not password:
        job.add_log(
            "ERROR",
            "NASA Earthdata credentials not found in environment. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD. "
            "See .env.example. Credential values will not be logged.",
        )
        return False, 0, None

    # ------------------------------------------------------------------
    # Create authenticated session
    # ------------------------------------------------------------------
    try:
        import earthaccess as earthaccess  # noqa: PLC0414
        earthaccess.login(strategy="environment", persist=False)
        session = earthaccess.get_requests_https_session()
        if session is None:
            job.add_log("ERROR", "earthaccess.get_requests_https_session() returned None.")
            return False, 0, None
    except Exception as exc:
        safe_msg = _re.sub(
            r'(password|passwd|token|secret|credential)[^\s,;)]*',
            r'\1=<REDACTED>', str(exc), flags=_re.IGNORECASE,
        )
        job.add_log("ERROR", f"Failed to create Earthdata session: {safe_msg}")
        return False, 0, None

    opendap_url = params["opendap_url"]
    granule = params["granule"]
    tmp_path = dest_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        if job._cancel_event.is_set():
            job.add_log("INFO", "OSCAR download cancelled before attempt.")
            return False, 0, None

        try:
            job.add_log(
                "DOWNLOAD",
                f"OSCAR subset: {dest_path.name} (attempt {attempt}/{max_retries}) "
                f"granule={granule}",
                {
                    "granule": granule,
                    "date": params["request_date"].isoformat(),
                    "dest": str(dest_path),
                },
            )

            response = session.get(
                opendap_url,
                stream=True,
                timeout=(30, 300),
            )

            if response.status_code == 401:
                job.add_log(
                    "ERROR",
                    "NASA Earthdata authentication failed (HTTP 401). "
                    "Check credentials. Not retrying.",
                )
                tmp_path.unlink(missing_ok=True)
                return False, 0, None

            if response.status_code == 404:
                job.add_log(
                    "ERROR",
                    f"OSCAR granule not found (HTTP 404): {granule}. Not retrying.",
                )
                tmp_path.unlink(missing_ok=True)
                return False, 0, None

            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")

            content_type = response.headers.get("Content-Type", "")
            if "netcdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                raise RuntimeError(
                    f"Unexpected Content-Type: {content_type!r} — expected NetCDF"
                )

            # Stream to tmp file, compute MD5 on the fly
            hasher = hashlib.md5()
            total_bytes = 0
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    if job._cancel_event.is_set():
                        tmp_path.unlink(missing_ok=True)
                        job.add_log("INFO", "OSCAR download cancelled mid-stream.")
                        return False, 0, None
                    if chunk:
                        fh.write(chunk)
                        hasher.update(chunk)
                        total_bytes += len(chunk)

            response.close()

            if total_bytes == 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError("Downloaded file is empty.")

            md5 = hasher.hexdigest()

            # Atomic rename
            tmp_path.rename(dest_path)

            job.add_log(
                "SUCCESS",
                f"OSCAR download complete: {dest_path.name} "
                f"({total_bytes:,} bytes, md5={md5})",
                {"path": str(dest_path), "bytes": total_bytes, "md5": md5},
            )
            return True, total_bytes, md5

        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            exc_type = type(exc).__name__
            safe_msg = _re.sub(
                r'(password|passwd|token|secret|credential)[^\s,;)]*',
                r'\1=<REDACTED>', str(exc), flags=_re.IGNORECASE,
            )
            job.add_log(
                "ERROR",
                f"OSCAR error on attempt {attempt} ({exc_type}): {safe_msg}",
            )
            if "401" in safe_msg or "unauthorized" in safe_msg.lower() or "authentication" in safe_msg.lower():
                job.add_log("ERROR", "Authentication error — not retrying.")
                return False, 0, None
            if attempt < max_retries:
                _time.sleep(min(5 * (2 ** (attempt - 1)), 120))
            else:
                return False, 0, None

    return False, 0, None


# ---------------------------------------------------------------------------
# run_acquisition_job
# ---------------------------------------------------------------------------

async def run_acquisition_job(job: AcquisitionJob, adapter: Any) -> None:
    """
    Execute one acquisition job end-to-end.

    For each AcquisitionRequest produced by the adapter:
    1. Download (or skip if already present).
    2. Record a ManifestEntry with checksum, size, status.
    3. Update job counters.

    The checksum is always persisted — whether the file was freshly
    downloaded or already existed on disk.
    """
    manifest = ManifestStore(job.data_root)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)

    try:
        requests: list[AcquisitionRequest] = adapter.date_range_requests(
            job.start_date, job.end_date, str(job.data_root)
        )
        job.files_requested = len(requests)
        job.add_log(
            "INFO",
            f"Job started: {len(requests)} file(s) requested "
            f"({job.start_date} → {job.end_date})",
        )

        for req in requests:
            if job._cancel_event.is_set():
                break

            dest_abs = job.data_root / req.expected_output_path

            # ------------------------------------------------------------------
            # Dispatch: CMEMS sentinel URL → run_cmems_subset (via thread)
            #           OSCAR sentinel URL → run_oscar_subset (via thread)
            #           Plain URL          → download_file (async HTTPS)
            # ------------------------------------------------------------------
            if req.url_or_query.startswith("cmems://"):
                # copernicusmarine.subset() is synchronous — run in a thread
                # so the asyncio event loop is not blocked.
                success, nbytes, md5 = await asyncio.to_thread(
                    run_cmems_subset,
                    req.url_or_query,
                    dest_abs,
                    job,
                )
            elif req.url_or_query.startswith("oscar://"):
                # earthaccess session is synchronous — run in a thread.
                success, nbytes, md5 = await asyncio.to_thread(
                    run_oscar_subset,
                    req.url_or_query,
                    dest_abs,
                    job,
                )
            else:
                success, nbytes, md5 = await download_file(
                    url=req.url_or_query,
                    dest_path=dest_abs,
                    job=job,
                )

            status = ManifestStatus.SUCCESS if success else ManifestStatus.FAILED
            entry = ManifestEntry(
                dataset_id=req.dataset_id,
                request_date=req.request_date,
                source_url=req.url_or_query,
                local_path=str(dest_abs),
                product_id=req.source_description.product_id,
                file_size_bytes=nbytes if success else None,
                md5_checksum=md5,          # ← persisted (was always None before)
                status=status,
                error=None if success else "Download failed — see job logs",
            )
            manifest.record(entry)

            if success:
                job.files_completed += 1
                job.bytes_downloaded += nbytes
            else:
                job.files_failed += 1

        if job._cancel_event.is_set():
            job.status = JobStatus.CANCELLED
            job.add_log("INFO", "Job was cancelled.")
        elif job.files_failed == 0:
            job.status = JobStatus.COMPLETED
            job.add_log(
                "INFO",
                f"Job completed. {job.files_completed}/{job.files_requested} "
                f"file(s), {job.bytes_downloaded:,} bytes.",
            )
        else:
            job.status = JobStatus.FAILED
            job.add_log(
                "ERROR",
                f"Job finished with errors. "
                f"{job.files_completed} OK, {job.files_failed} failed.",
            )

    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.add_log("ERROR", f"Unexpected job error: {exc}")
        logger.exception("Unexpected error in job %s", job.job_id)

    finally:
        job.completed_at = datetime.now(timezone.utc)
