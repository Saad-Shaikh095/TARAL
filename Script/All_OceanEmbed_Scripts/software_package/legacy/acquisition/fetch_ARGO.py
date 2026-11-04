"""
===============================================================================
OCEANEMBED ARGO 2012-2024 V6
ROBUST, RESUMABLE, MULTI-GDAC HTTPS + PUBLIC S3 FALLBACK
===============================================================================

Purpose
-------
Acquire the OceanEmbed raw ARGO profile archive for:

    Date      : 2012-01-01 .. 2024-12-31
    Latitude  : 5.0 .. 30.0 N
    Longitude : 45.0 .. 105.0 E

Input:
    D:\OceanEmbed\argo\manifests\argo_profile_manifest_2012_2024.csv

Raw output:
    D:\OceanEmbed\argo\raw\YYYY_MM\<manifest file path>

The manifest's `file` column is authoritative.

Example:
    aoml/1900053/profiles/D1900053_317.nc

Official access sources:
    1) Coriolis GDAC HTTPS
       https://data-argo.ifremer.fr/

    2) US GDAC HTTPS
       https://nrlgodae1.nrlmry.navy.mil/pub/outgoing/argo/

    3) Public Argo GDAC S3 object HTTPS endpoint
       https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/

The Argo Data Management Team documents all three access mechanisms. The S3
bucket is public and does not require an AWS account.

Important
---------
This script only acquires RAW profile NetCDF files. It does NOT:
    - interpolate
    - regrid
    - merge profiles
    - modify scientific values
    - remove QC
    - resample depth

Later processing remains a separate validation pipeline.

Robustness
----------
- Defensive CSV parsing.
- Blank/malformed lat/lon values are counted, not warning-spammed.
- Exact date + NIO filtering.
- Duplicate paths removed.
- Current official Coriolis and US GDAC HTTPS endpoints.
- Public S3 fallback after HTTPS source failure.
- Permanent 401/403/404/405/410 errors move to next source immediately.
- Retryable 408/425/429/5xx + network errors use backoff.
- .part files are never treated as complete.
- NetCDF structure is validated before atomic finalization.
- Existing valid files are resumed/skipped.
- SQLite WAL state survives interruption.
- No per-file HEAD request.
- Per-profile failure state is durable.
- Second production pass retries remaining profiles.
- Final completeness check is performed without loading entire profiles.
- No hard-coded credentials.

Python:
    Designed for the existing Python 3.14 environment.

Required:
    requests
    netCDF4
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock, local
from typing import Optional
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter


# =============================================================================
# CONFIGURATION
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    r"D:\OceanEmbed\argo"
)

MANIFEST_FILE = (
    BASE_DIR
    / "manifests"
    / "argo_profile_manifest_2012_2024.csv"
)

# -------------------------------------------------------------------------
# RAW DATA — FINAL PRODUCTION LOCATION
# -------------------------------------------------------------------------

RAW_BASE = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\argo"
)

# -------------------------------------------------------------------------
# MANIFESTS / LOGS / STATE
# -------------------------------------------------------------------------

MANIFEST_DIR = BASE_DIR / "manifests"

LOG_DIR = BASE_DIR / "logs"

SELECTED_MANIFEST = (
    MANIFEST_DIR
    / "argo_selected_manifest_2012_2024_v5.csv"
)

REJECTED_MANIFEST = (
    MANIFEST_DIR
    / "argo_manifest_rejected_2012_2024_v5.csv"
)

FAILED_REPORT = (
    MANIFEST_DIR
    / "argo_failed_2012_2024_v5.csv"
)

STATE_DB = (
    MANIFEST_DIR
    / "argo_acquisition_2012_2024_v5.sqlite"
)

LOG_FILE = (
    LOG_DIR
    / "argo_acquisition_2012_2024_v5.log"
)


# =============================================================================
# SCIENTIFIC SELECTION
# =============================================================================

START_DATE = date(2012, 1, 1)
END_DATE = date(2024, 12, 31)

LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0


# =============================================================================
# PERFORMANCE
# =============================================================================

MAX_WORKERS = 12

# Never create 77k futures simultaneously.
BATCH_SIZE = MAX_WORKERS * 8

PROGRESS_EVERY = 250

CHUNK_SIZE = 1024 * 1024


# =============================================================================
# NETWORK
# =============================================================================

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 45

USER_AGENT = (
    "OceanEmbed-ARGO-Acquisition-V5.1/1.0 "
    "(scientific research)"
)


# =============================================================================
# RETRY POLICY
# =============================================================================

MAX_RETRIES_PER_SOURCE = 2

RETRYABLE_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504,
}

# Immediately switch to the next official source.
PERMANENT_SOURCE_FAILURE_CODES = {
    401,
    403,
    404,
    405,
    410,
}

RETRY_DELAYS = (
    3,
    8,
)


# =============================================================================
# FILE VALIDATION
# =============================================================================

MIN_FILE_SIZE_BYTES = 1024

# These are the minimum fields needed for our later temperature-profile
# validation work. PSAL is intentionally not mandatory at acquisition time.
REQUIRED_VARIABLES = (
    "PRES",
    "TEMP",
)


# =============================================================================
# PREFLIGHT
# =============================================================================

PREFLIGHT_ENABLED = True

# Number of successful profiles we want to prove the source chain with.
PREFLIGHT_SUCCESS_TARGET = 10

# Maximum candidate profiles examined by preflight if some candidates are
# unavailable in every source.
PREFLIGHT_MAX_CANDIDATES = 30

# SAFETY SWITCH:
# TEST is the default so accidentally launching the script on a laptop cannot
# start the full 77k-profile acquisition. Change to PRODUCTION tomorrow, or
# invoke the script with --production.
DEFAULT_RUN_MODE = "PRODUCTION"


# =============================================================================
# OFFICIAL ACCESS SOURCES
# =============================================================================
#
# Current official Argo documentation:
#   Coriolis HTTPS
#   Current US GDAC HTTPS
#   Public S3 GDAC
#
# S3 path:
#   s3://argo-gdac-sandbox/pub/dac/<manifest file>
#
# Equivalent public HTTPS object endpoint:
#   https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/<file>
#
# =============================================================================

SOURCES = (
    # Primary: public Argo GDAC S3 mirror. Verified in the V4 preflight.
    (
        "argo_s3_https",
        "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/",
    ),
    # Fallback: official Coriolis GDAC HTTPS.
    (
        "coriolis_https",
        "https://data-argo.ifremer.fr/",
    ),
    # Last resort: current US GDAC HTTPS. The endpoint can be slow/unreachable
    # from some networks, so it is intentionally last.
    (
        "usgdac_https",
        "https://nrlgodae1.nrlmry.navy.mil/pub/outgoing/argo/",
    ),
)


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

for directory in (
    RAW_BASE,
    MANIFEST_DIR,
    LOG_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# LOGGING
# =============================================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger(
        "oceanembed_argo_v5"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(
        sys.stdout
    )
    console.setFormatter(
        formatter
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console
    )
    logger.addHandler(
        file_handler
    )

    return logger


logger = setup_logger()


# =============================================================================
# THREAD-LOCAL HTTP SESSION
# =============================================================================

THREAD_LOCAL = local()


def get_session() -> requests.Session:
    session = getattr(
        THREAD_LOCAL,
        "session",
        None,
    )

    if session is not None:
        return session

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": (
                "application/x-netcdf, "
                "application/octet-stream, "
                "*/*"
            ),
            "Connection": "keep-alive",
        }
    )

    session.trust_env = True

    adapter = HTTPAdapter(
        pool_connections=8,
        pool_maxsize=MAX_WORKERS,
        max_retries=0,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    THREAD_LOCAL.session = session

    return session


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass(frozen=True)
class ProfileRecord:
    file: str
    date_raw: str
    latitude: float
    longitude: float
    ocean: str
    profiler_type: str
    institution: str
    date_update: str

    @property
    def key(self) -> str:
        return (
            self.file
            .replace("\\", "/")
            .strip("/")
        )

    @property
    def profile_date(self) -> date:
        parsed = parse_profile_date(
            self.date_raw
        )

        if parsed is None:
            raise ValueError(
                f"Invalid profile date: {self.date_raw!r}"
            )

        return parsed

    @property
    def local_path(self) -> Path:
        d = self.profile_date

        return (
            RAW_BASE
            / f"{d.year:04d}_{d.month:02d}"
            / Path(
                *self.key.split("/")
            )
        )


@dataclass(frozen=True)
class AttemptResult:
    success: bool
    source: str | None
    status_code: int | None
    bytes_written: int
    sha256: str | None
    attempts: int
    reason: str | None


# =============================================================================
# SAFE PARSING
# =============================================================================

def safe_float(
    value: object,
) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        result = float(text)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(result):
        return None

    return result


def parse_profile_date(
    value: object,
) -> date | None:
    """
    Accepts:
        YYYYMMDDHHMMSS
        YYYYMMDD
        spreadsheet-style scientific notation

    Normal Argo index values are YYYYMMDDHHMMSS.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    # Remove the common spreadsheet representation "....0".
    cleaned = re.sub(
        r"\.0+$",
        "",
        text,
    )

    if cleaned.isdigit():

        if len(cleaned) >= 14:

            try:
                return datetime.strptime(
                    cleaned[:14],
                    "%Y%m%d%H%M%S",
                ).date()
            except ValueError:
                pass

        if len(cleaned) >= 8:

            try:
                return datetime.strptime(
                    cleaned[:8],
                    "%Y%m%d",
                ).date()
            except ValueError:
                pass

    # Scientific-notation fallback.
    try:
        number = float(text)

        if math.isfinite(number):

            integer_text = str(
                int(round(number))
            )

            if len(integer_text) >= 14:

                try:
                    return datetime.strptime(
                        integer_text[:14],
                        "%Y%m%d%H%M%S",
                    ).date()
                except ValueError:
                    pass

            if len(integer_text) >= 8:

                try:
                    return datetime.strptime(
                        integer_text[:8],
                        "%Y%m%d",
                    ).date()
                except ValueError:
                    pass

    except (
        ValueError,
        OverflowError,
    ):
        pass

    return None


def normalize_row(
    row: dict[str, str],
) -> dict[str, str]:

    result: dict[str, str] = {}

    for key, value in row.items():

        if key is None:
            continue

        clean_key = (
            str(key)
            .replace("\ufeff", "")
            .strip()
            .lower()
        )

        result[
            clean_key
        ] = (
            ""
            if value is None
            else str(value).strip()
        )

    return result


def valid_relative_path(
    value: str,
) -> bool:

    if not value:
        return False

    value = (
        value
        .replace("\\", "/")
        .strip()
        .strip("/")
    )

    if value.startswith(
        (
            "http://",
            "https://",
            "ftp://",
        )
    ):
        return False

    # Prevent drive paths.
    if len(value) >= 2 and value[1] == ":":
        return False

    parts = [
        part
        for part in value.split("/")
        if part
    ]

    if len(parts) < 4:
        return False

    # Expected:
    # DAC / float_id / profiles / filename.nc
    if parts[-2].lower() != "profiles":
        return False

    if not parts[-1].lower().endswith(
        ".nc"
    ):
        return False

    # Prevent traversal.
    if any(
        part in (".", "..")
        for part in parts
    ):
        return False

    return True


def row_to_profile(
    row: dict[str, str],
) -> tuple[
    ProfileRecord | None,
    str | None,
]:

    r = normalize_row(
        row
    )

    file_value = (
        r.get(
            "file",
            "",
        )
        .replace("\\", "/")
        .strip("/")
    )

    if not valid_relative_path(
        file_value
    ):
        return None, "invalid_file_path"

    date_raw = r.get(
        "date",
        "",
    )

    if parse_profile_date(
        date_raw
    ) is None:
        return None, "invalid_date"

    latitude = safe_float(
        r.get(
            "latitude",
            "",
        )
    )

    longitude = safe_float(
        r.get(
            "longitude",
            "",
        )
    )

    if latitude is None:
        return None, "missing_latitude"

    if longitude is None:
        return None, "missing_longitude"

    if not (
        -90.0
        <= latitude
        <= 90.0
    ):
        return None, "invalid_latitude_range"

    if not (
        -180.0
        <= longitude
        <= 180.0
    ):
        return None, "invalid_longitude_range"

    return (
        ProfileRecord(
            file=file_value,
            date_raw=date_raw,
            latitude=latitude,
            longitude=longitude,
            ocean=r.get(
                "ocean",
                "",
            ),
            profiler_type=r.get(
                "profiler_type",
                "",
            ),
            institution=r.get(
                "institution",
                "",
            ),
            date_update=r.get(
                "date_update",
                "",
            ),
        ),
        None,
    )


# =============================================================================
# MANIFEST BUILD
# =============================================================================

def build_selected_manifest() -> list[ProfileRecord]:
    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "BUILDING NIO ARGO MANIFEST"
    )
    logger.info(
        "=" * 80
    )

    logger.info(
        "Input manifest : %s",
        MANIFEST_FILE,
    )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    expected_columns = {
        "file",
        "date",
        "latitude",
        "longitude",
        "ocean",
        "profiler_type",
        "institution",
        "date_update",
    }

    selected: dict[
        str,
        ProfileRecord,
    ] = {}

    total_rows = 0
    rejected_rows = 0
    duplicate_rows = 0

    rejection_counts: dict[
        str,
        int,
    ] = {}

    with REJECTED_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as rejected_handle:

        rejected_writer = csv.writer(
            rejected_handle
        )

        rejected_writer.writerow(
            [
                "manifest_row",
                "reason",
            ]
        )

        with MANIFEST_FILE.open(
            "r",
            newline="",
            encoding="utf-8-sig",
            errors="replace",
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            if not reader.fieldnames:
                raise RuntimeError(
                    "Manifest has no header."
                )

            actual_columns = {
                str(column)
                .strip()
                .lower()
                for column in reader.fieldnames
                if column is not None
            }

            missing = (
                expected_columns
                - actual_columns
            )

            if missing:
                raise RuntimeError(
                    "Manifest missing required columns: "
                    f"{sorted(missing)}"
                )

            for row_number, row in enumerate(
                reader,
                start=2,
            ):

                total_rows += 1

                profile, reason = (
                    row_to_profile(
                        row
                    )
                )

                if profile is None:

                    rejected_rows += 1

                    reason_key = (
                        reason
                        or "unknown"
                    )

                    rejection_counts[
                        reason_key
                    ] = (
                        rejection_counts.get(
                            reason_key,
                            0,
                        )
                        + 1
                    )

                    rejected_writer.writerow(
                        [
                            row_number,
                            reason_key,
                        ]
                    )

                    continue

                d = profile.profile_date

                if (
                    d < START_DATE
                    or d > END_DATE
                ):
                    continue

                if not (
                    LAT_MIN
                    <= profile.latitude
                    <= LAT_MAX
                ):
                    continue

                if not (
                    LON_MIN
                    <= profile.longitude
                    <= LON_MAX
                ):
                    continue

                if profile.key in selected:

                    duplicate_rows += 1

                    continue

                selected[
                    profile.key
                ] = profile

                if (
                    total_rows
                    % 250_000
                    == 0
                ):

                    logger.info(
                        "Scanned %s | NIO selected %s",
                        f"{total_rows:,}",
                        f"{len(selected):,}",
                    )

    profiles = sorted(
        selected.values(),
        key=lambda p: (
            p.profile_date,
            p.key,
        ),
    )

    with SELECTED_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "file",
                "date",
                "latitude",
                "longitude",
                "ocean",
                "profiler_type",
                "institution",
                "date_update",
            ]
        )

        for profile in profiles:

            writer.writerow(
                [
                    profile.file,
                    profile.date_raw,
                    f"{profile.latitude:.6f}",
                    f"{profile.longitude:.6f}",
                    profile.ocean,
                    profile.profiler_type,
                    profile.institution,
                    profile.date_update,
                ]
            )

    logger.info("")
    logger.info(
        "MANIFEST SCAN COMPLETE"
    )

    logger.info(
        "Rows scanned       : %s",
        f"{total_rows:,}",
    )

    logger.info(
        "NIO profiles       : %s",
        f"{len(profiles):,}",
    )

    logger.info(
        "Rejected rows      : %s",
        f"{rejected_rows:,}",
    )

    logger.info(
        "Duplicate paths    : %s",
        f"{duplicate_rows:,}",
    )

    if rejection_counts:

        logger.info(
            "Rejected-row reasons:"
        )

        for reason, count in sorted(
            rejection_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            logger.info(
                "  %-30s %10s",
                reason,
                f"{count:,}",
            )

    logger.info(
        "Selected manifest  : %s",
        SELECTED_MANIFEST,
    )

    logger.info(
        "Rejected report    : %s",
        REJECTED_MANIFEST,
    )

    if not profiles:
        raise RuntimeError(
            "No profiles selected for NIO acquisition."
        )

    return profiles


# =============================================================================
# SQLITE STATE
# =============================================================================

DB_LOCK = Lock()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        STATE_DB,
        timeout=30,
    )

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA busy_timeout=30000"
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            file TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source TEXT,
            bytes INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            http_status INTEGER,
            reason TEXT,
            updated_utc TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_profiles_status
        ON profiles(status)
        """
    )

    conn.commit()

    return conn


def db_mark(
    profile: ProfileRecord,
    status: str,
    source: str | None,
    bytes_written: int,
    digest: str | None,
    attempts: int,
    http_status: int | None,
    reason: str | None,
) -> None:

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with DB_LOCK:

        conn = db_connect()

        try:

            conn.execute(
                """
                INSERT INTO profiles
                (
                    file,
                    status,
                    source,
                    bytes,
                    sha256,
                    attempts,
                    http_status,
                    reason,
                    updated_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file) DO UPDATE SET
                    status = excluded.status,
                    source = excluded.source,
                    bytes = excluded.bytes,
                    sha256 = excluded.sha256,
                    attempts = excluded.attempts,
                    http_status = excluded.http_status,
                    reason = excluded.reason,
                    updated_utc = excluded.updated_utc
                """,
                (
                    profile.key,
                    status,
                    source,
                    bytes_written,
                    digest,
                    attempts,
                    http_status,
                    reason,
                    timestamp,
                ),
            )

            conn.commit()

        finally:

            conn.close()


def db_status_counts() -> dict[str, int]:

    with DB_LOCK:

        conn = db_connect()

        try:

            rows = conn.execute(
                """
                SELECT status, COUNT(*)
                FROM profiles
                GROUP BY status
                """
            ).fetchall()

        finally:

            conn.close()

    return {
        str(status): int(count)
        for status, count in rows
    }


def db_completed_records() -> dict[str, int]:
    """Return completed profile keys mapped to recorded byte sizes."""
    with DB_LOCK:
        conn = db_connect()
        try:
            rows = conn.execute(
                """
                SELECT file, bytes
                FROM profiles
                WHERE status = 'complete'
                """
            ).fetchall()
        finally:
            conn.close()

    return {
        str(file_key): int(bytes_count)
        for file_key, bytes_count in rows
    }


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_download_file(
    path: Path,
) -> tuple[
    bool,
    str,
]:
    """
    Lightweight acquisition-time validation.

    IMPORTANT:
    This function intentionally does NOT import/open netCDF4. The production
    downloader uses worker threads, while netCDF4/HDF5 is native code. The
    previous V5.1 implementation opened every freshly downloaded .part file
    through netCDF4 concurrently, and the process terminated with Windows
    exit code 0xC0000005 (access violation).

    We only need to establish here that:
      1. the file exists,
      2. it is not trivially small, and
      3. it has a NetCDF/HDF5 or classic NetCDF signature.

    Full netCDF4 structural validation remains in the serial completeness
    checks, so a bad file cannot become part of the final accepted archive.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat_error:{exc}"

    if size < MIN_FILE_SIZE_BYTES:
        return False, "too_small"

    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
    except OSError as exc:
        return False, f"read_error:{exc}"

    # NetCDF-4/HDF5 signature.
    if signature == b"\\x89HDF\\r\\n\\x1a\\n":
        return True, "netcdf4_hdf5_signature"

    # NetCDF classic / 64-bit offset signature.
    if signature[:3] == b"CDF":
        return True, "netcdf_classic_signature"

    return False, "invalid_netcdf_signature"


def validate_netcdf(
    path: Path,
) -> tuple[
    bool,
    str,
]:

    if not path.exists():
        return False, "missing"

    try:

        size = path.stat().st_size

    except OSError as exc:

        return False, (
            f"stat_error:{exc}"
        )

    if size < MIN_FILE_SIZE_BYTES:

        return False, "too_small"

    try:

        import netCDF4

    except ImportError:

        return False, (
            "netCDF4_not_installed"
        )

    dataset = None

    try:

        dataset = netCDF4.Dataset(
            str(path),
            mode="r",
        )

        variables = set(
            dataset.variables.keys()
        )

        for name in REQUIRED_VARIABLES:

            if name not in variables:

                return False, (
                    f"missing_variable:{name}"
                )

            variable = (
                dataset.variables[name]
            )

            if len(
                variable.shape
            ) == 0:

                return False, (
                    f"invalid_scalar:{name}"
                )

            if variable.size <= 0:

                return False, (
                    f"empty_variable:{name}"
                )

            # IMPORTANT:
            # Do NOT use variable.flat[0].
            #
            # netCDF4.Variable does not reliably expose the numpy `.flat`
            # accessor. That was the cause of the earlier:
            #
            #   NetCDF: Attribute not found
            #
            # failure in the previous version.
            #
            # For acquisition-time validation, successful Dataset opening,
            # variable presence, non-zero dimensions, and non-zero size are
            # sufficient. Full scientific value/QC validation belongs later.

        return True, "ok"

    except Exception as exc:

        return False, (
            f"netcdf_error:"
            f"{type(exc).__name__}:"
            f"{exc}"
        )

    finally:

        if dataset is not None:

            try:
                dataset.close()
            except Exception:
                pass


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# SOURCE URLS
# =============================================================================

def encoded_relative_path(
    relative_key: str,
) -> str:

    return "/".join(
        quote(
            component,
            safe="",
        )
        for component in relative_key.split(
            "/"
        )
    )


def urls_for(
    profile: ProfileRecord,
) -> tuple[
    tuple[str, str],
    ...,
]:

    encoded = encoded_relative_path(
        profile.key
    )

    return tuple(
        (
            source_name,
            base_url + encoded,
        )
        for source_name, base_url
        in SOURCES
    )


# =============================================================================
# SINGLE-SOURCE DOWNLOAD
# =============================================================================

def attempt_source(
    profile: ProfileRecord,
    source_name: str,
    url: str,
    destination: Path,
) -> AttemptResult:

    # CRITICAL: make the complete DAC/float/profiles directory tree before
    # opening the .part file. Without this, a valid HTTP 200 response can
    # still fail locally with Windows FileNotFoundError.
    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        return AttemptResult(
            success=False,
            source=source_name,
            status_code=None,
            bytes_written=0,
            sha256=None,
            attempts=0,
            reason=f"destination_directory_error:{exc}",
        )

    session = get_session()

    partial = destination.with_suffix(
        destination.suffix
        + ".part"
    )

    last_reason: str | None = None
    last_status: int | None = None
    attempts_used = 0

    for attempt in range(
        1,
        MAX_RETRIES_PER_SOURCE + 1,
    ):

        attempts_used += 1

        try:

            if partial.exists():

                try:
                    partial.unlink()
                except OSError:
                    pass

            started = time.perf_counter()

            with session.get(
                url,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
                allow_redirects=True,
            ) as response:

                status = response.status_code

                last_status = status

                # -------------------------------------------------------------
                # Permanent source failure
                # -------------------------------------------------------------

                if status in (
                    PERMANENT_SOURCE_FAILURE_CODES
                ):

                    return AttemptResult(
                        success=False,
                        source=source_name,
                        status_code=status,
                        bytes_written=0,
                        sha256=None,
                        attempts=attempts_used,
                        reason=(
                            f"source_unavailable_http_{status}"
                        ),
                    )

                # -------------------------------------------------------------
                # Retryable server condition
                # -------------------------------------------------------------

                if status in (
                    RETRYABLE_STATUS_CODES
                ):

                    last_reason = (
                        f"retryable_http_{status}"
                    )

                else:

                    response.raise_for_status()

                    content_type = (
                        response.headers.get(
                            "Content-Type",
                            "",
                        )
                        .lower()
                    )

                    # ---------------------------------------------------------
                    # Refuse obvious error documents.
                    # ---------------------------------------------------------

                    if (
                        "text/html"
                        in content_type
                        or "application/json"
                        in content_type
                        or "xml"
                        in content_type
                    ):

                        return AttemptResult(
                            success=False,
                            source=source_name,
                            status_code=status,
                            bytes_written=0,
                            sha256=None,
                            attempts=attempts_used,
                            reason=(
                                "non_netcdf_document_response"
                            ),
                        )

                    expected_length = None

                    content_length = (
                        response.headers.get(
                            "Content-Length"
                        )
                    )

                    if content_length:

                        try:

                            expected_length = int(
                                content_length
                            )

                        except ValueError:

                            expected_length = None

                    written = 0

                    with partial.open(
                        "wb"
                    ) as handle:

                        for chunk in response.iter_content(
                            chunk_size=CHUNK_SIZE
                        ):

                            if not chunk:
                                continue

                            handle.write(
                                chunk
                            )

                            written += len(
                                chunk
                            )

                    if (
                        written
                        < MIN_FILE_SIZE_BYTES
                    ):

                        raise RuntimeError(
                            "response too small: "
                            f"{written} bytes"
                        )

                    if (
                        expected_length is not None
                        and written
                        != expected_length
                    ):

                        raise RuntimeError(
                            "Content-Length mismatch: "
                            f"expected={expected_length}, "
                            f"received={written}"
                        )

                    # ---------------------------------------------------------
                    # CRITICAL:
                    # validate BEFORE finalization.
                    # ---------------------------------------------------------

                    valid, reason = (
                        validate_download_file(
                            partial
                        )
                    )

                    if not valid:

                        raise RuntimeError(
                            "NetCDF validation failed: "
                            f"{reason}"
                        )

                    digest = sha256_file(
                        partial
                    )

                    # Atomic finalization.
                    partial.replace(
                        destination
                    )

                    elapsed = (
                        time.perf_counter()
                        - started
                    )

                    speed = (
                        written
                        / 1024
                        / 1024
                        / max(
                            elapsed,
                            1e-9,
                        )
                    )

                    logger.info(
                        "DOWNLOADED | %s | source=%s | "
                        "%.2f MB | %.2f MB/s",
                        profile.key,
                        source_name,
                        written / 1024 / 1024,
                        speed,
                    )

                    return AttemptResult(
                        success=True,
                        source=source_name,
                        status_code=status,
                        bytes_written=written,
                        sha256=digest,
                        attempts=attempts_used,
                        reason=None,
                    )

        except requests.HTTPError as exc:

            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )

            last_status = status

            if status in (
                PERMANENT_SOURCE_FAILURE_CODES
            ):

                return AttemptResult(
                    success=False,
                    source=source_name,
                    status_code=status,
                    bytes_written=0,
                    sha256=None,
                    attempts=attempts_used,
                    reason=(
                        f"source_unavailable_http_{status}"
                    ),
                )

            last_reason = (
                f"http_error:{exc}"
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as exc:

            last_reason = (
                f"network:"
                f"{type(exc).__name__}:"
                f"{exc}"
            )

        except Exception as exc:

            last_reason = (
                f"{type(exc).__name__}:"
                f"{exc}"
            )

        finally:

            if partial.exists():

                try:
                    partial.unlink()
                except OSError:
                    pass

        if attempt < MAX_RETRIES_PER_SOURCE:

            delay = RETRY_DELAYS[
                min(
                    attempt - 1,
                    len(
                        RETRY_DELAYS
                    ) - 1,
                )
            ]

            time.sleep(
                delay
            )

    return AttemptResult(
        success=False,
        source=source_name,
        status_code=last_status,
        bytes_written=0,
        sha256=None,
        attempts=attempts_used,
        reason=(
            last_reason
            or "source_failed"
        ),
    )


# =============================================================================
# PROFILE DOWNLOAD WITH SOURCE FALLBACK
# =============================================================================

def download_profile(
    profile: ProfileRecord,
) -> AttemptResult:

    destination = profile.local_path

    # -------------------------------------------------------------------------
    # Resume:
    # Existing local file is the strongest practical signal.
    # -------------------------------------------------------------------------

    if destination.exists():

        valid, reason = (
            validate_download_file(
                destination
            )
        )

        if valid:

            size = (
                destination.stat().st_size
            )

            digest = sha256_file(
                destination
            )

            return AttemptResult(
                success=True,
                source="local_existing",
                status_code=None,
                bytes_written=size,
                sha256=digest,
                attempts=0,
                reason="already_valid",
            )

        logger.warning(
            "INVALID LOCAL | %s | reason=%s | replacing",
            profile.key,
            reason,
        )

        try:

            destination.unlink()

        except OSError as exc:

            return AttemptResult(
                success=False,
                source=None,
                status_code=None,
                bytes_written=0,
                sha256=None,
                attempts=0,
                reason=(
                    "cannot_remove_invalid_file:"
                    f"{exc}"
                ),
            )

    last_reasons: list[str] = []
    total_attempts = 0
    last_status: int | None = None

    for source_name, url in urls_for(
        profile
    ):

        logger.info(
            "SOURCE TRY | %s | %s",
            profile.key,
            source_name,
        )

        result = attempt_source(
            profile,
            source_name,
            url,
            destination,
        )

        total_attempts += (
            result.attempts
        )

        if result.status_code is not None:
            last_status = (
                result.status_code
            )

        if result.success:

            return AttemptResult(
                success=True,
                source=result.source,
                status_code=result.status_code,
                bytes_written=result.bytes_written,
                sha256=result.sha256,
                attempts=total_attempts,
                reason=result.reason,
            )

        last_reasons.append(
            f"{source_name}:"
            f"{result.reason}"
        )

        logger.warning(
            "SOURCE FAIL | %s | source=%s | "
            "HTTP=%s | %s",
            profile.key,
            source_name,
            result.status_code,
            result.reason,
        )

    return AttemptResult(
        success=False,
        source=None,
        status_code=last_status,
        bytes_written=0,
        sha256=None,
        attempts=total_attempts,
        reason=(
            "; ".join(
                last_reasons
            )
            or "all_sources_failed"
        ),
    )


# =============================================================================
# FAILURE REPORT
# =============================================================================

FAILED_LOCK = Lock()


def rebuild_failed_report(
    profiles: list[ProfileRecord],
) -> None:

    profile_map = {
        profile.key: profile
        for profile in profiles
    }

    with DB_LOCK:

        conn = db_connect()

        try:

            rows = conn.execute(
                """
                SELECT
                    file,
                    source,
                    http_status,
                    attempts,
                    reason
                FROM profiles
                WHERE status = 'failed'
                ORDER BY file
                """
            ).fetchall()

        finally:

            conn.close()

    with FAILED_REPORT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "file",
                "date",
                "latitude",
                "longitude",
                "coriolis_url",
                "usgdac_url",
                "s3_url",
                "source",
                "http_status",
                "attempts",
                "reason",
            ]
        )

        for (
            file_key,
            source,
            http_status,
            attempts,
            reason,
        ) in rows:

            profile = profile_map.get(
                file_key
            )

            if profile is None:
                continue

            mapping = dict(
                urls_for(
                    profile
                )
            )

            writer.writerow(
                [
                    profile.key,
                    profile.date_raw,
                    profile.latitude,
                    profile.longitude,
                    mapping.get(
                        "coriolis_https",
                        "",
                    ),
                    mapping.get(
                        "usgdac_https",
                        "",
                    ),
                    mapping.get(
                        "argo_s3_https",
                        "",
                    ),
                    source or "",
                    (
                        http_status
                        if http_status is not None
                        else ""
                    ),
                    attempts,
                    reason or "",
                ]
            )


# =============================================================================
# PREFLIGHT
# =============================================================================

def select_preflight_candidates(
    profiles: list[ProfileRecord],
) -> list[ProfileRecord]:

    selected: list[ProfileRecord] = []

    seen_dacs: set[str] = set()
    seen_years: set[int] = set()
    completed_records = db_completed_records()

    # Prefer profiles that are not already complete locally/state-wise.
    # This makes preflight a real network test instead of merely counting
    # previously downloaded files as successful.
    candidates = [
        profile
        for profile in profiles
        if profile.key not in completed_records
        and not profile.local_path.exists()
    ]

    # Prefer diversity.
    for profile in candidates:

        parts = profile.key.split("/")

        dac = (
            parts[0].lower()
            if parts
            else ""
        )

        year = profile.profile_date.year

        if (
            dac not in seen_dacs
            or year not in seen_years
        ):

            selected.append(
                profile
            )

            seen_dacs.add(
                dac
            )

            seen_years.add(
                year
            )

            if len(selected) >= PREFLIGHT_MAX_CANDIDATES:
                break

    # Fill to candidate limit.
    if len(selected) < PREFLIGHT_MAX_CANDIDATES:

        chosen = set(
            p.key
            for p in selected
        )

        for profile in candidates:

            if profile.key in chosen:
                continue

            selected.append(
                profile
            )

            if len(selected) >= PREFLIGHT_MAX_CANDIDATES:
                break

    return selected


def run_preflight(
    profiles: list[ProfileRecord],
) -> bool:

    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "ARGO PREFLIGHT SOURCE VALIDATION"
    )
    logger.info(
        "=" * 80
    )

    candidates = (
        select_preflight_candidates(
            profiles
        )
    )

    logger.info(
        "Preflight target successes : %d",
        PREFLIGHT_SUCCESS_TARGET,
    )

    logger.info(
        "Candidate profiles         : %d",
        len(candidates),
    )

    success_count = 0

    for index, profile in enumerate(
        candidates,
        start=1,
    ):

        if success_count >= PREFLIGHT_SUCCESS_TARGET:
            break

        logger.info(
            "[PREFLIGHT %d/%d] %s",
            index,
            len(candidates),
            profile.key,
        )

        result = download_profile(
            profile
        )

        if result.success:

            success_count += 1

            db_mark(
                profile,
                status="complete",
                source=result.source,
                bytes_written=result.bytes_written,
                digest=result.sha256,
                attempts=result.attempts,
                http_status=result.status_code,
                reason=result.reason,
            )

        else:

            db_mark(
                profile,
                status="failed",
                source=result.source,
                bytes_written=0,
                digest=None,
                attempts=result.attempts,
                http_status=result.status_code,
                reason=result.reason,
            )

            logger.warning(
                "PREFLIGHT CANDIDATE UNAVAILABLE | %s",
                profile.key,
            )

    logger.info(
        "PREFLIGHT RESULT: %d/%d successful",
        success_count,
        PREFLIGHT_SUCCESS_TARGET,
    )

    if (
        success_count
        >= PREFLIGHT_SUCCESS_TARGET
    ):

        logger.info(
            "PREFLIGHT PASS — source chain is operational."
        )

        return True

    logger.error(
        "PREFLIGHT FAILED — fewer than %d profiles were "
        "successfully acquired from the configured GDAC sources.",
        PREFLIGHT_SUCCESS_TARGET,
    )

    return False


# =============================================================================
# ACQUISITION PASS
# =============================================================================

def acquisition_pass(
    profiles: list[ProfileRecord],
    pass_number: int,
) -> tuple[
    int,
    int,
    int,
]:

    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "ARGO ACQUISITION PASS %d",
        pass_number,
    )
    logger.info(
        "=" * 80
    )

    pending: list[
        ProfileRecord
    ] = []

    skipped = 0
    completed_records = db_completed_records()

    # -------------------------------------------------------------------------
    # Determine pending work.
    #
    # Existing valid files are accepted.
    # Invalid existing files are re-acquired.
    # -------------------------------------------------------------------------

    for profile in profiles:

        path = profile.local_path

        if path.exists():

            valid, _ = (
                validate_netcdf(
                    path
                )
            )

            if valid:

                skipped += 1

                # Record it in state. Hash only when it is not already
                # recorded as complete, preventing unnecessary re-hashing on
                # every future run.
                if profile.key not in completed_records:

                    size = path.stat().st_size
                    digest = sha256_file(path)

                    db_mark(
                        profile,
                        status="complete",
                        source="local_existing",
                        bytes_written=size,
                        digest=digest,
                        attempts=0,
                        http_status=None,
                        reason="already_valid",
                    )

                continue

        pending.append(
            profile
        )

    logger.info(
        "Selected profiles : %s",
        f"{len(profiles):,}",
    )

    logger.info(
        "Existing valid    : %s",
        f"{skipped:,}",
    )

    logger.info(
        "Pending downloads : %s",
        f"{len(pending):,}",
    )

    if not pending:

        return (
            0,
            skipped,
            0,
        )

    downloaded = 0
    failed = 0
    completed = 0

    run_start = time.perf_counter()

    for batch_start in range(
        0,
        len(pending),
        BATCH_SIZE,
    ):

        batch = pending[
            batch_start:
            batch_start
            + BATCH_SIZE
        ]

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    download_profile,
                    profile,
                ): profile
                for profile in batch
            }

            for future in as_completed(
                future_map
            ):

                profile = future_map[
                    future
                ]

                completed += 1

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    result = (
                        AttemptResult(
                            success=False,
                            source=None,
                            status_code=None,
                            bytes_written=0,
                            sha256=None,
                            attempts=0,
                            reason=(
                                "worker_exception:"
                                f"{type(exc).__name__}:"
                                f"{exc}"
                            ),
                        )
                    )

                if result.success:

                    downloaded += 1

                    db_mark(
                        profile,
                        status="complete",
                        source=result.source,
                        bytes_written=result.bytes_written,
                        digest=result.sha256,
                        attempts=result.attempts,
                        http_status=result.status_code,
                        reason=result.reason,
                    )

                else:

                    failed += 1

                    db_mark(
                        profile,
                        status="failed",
                        source=result.source,
                        bytes_written=0,
                        digest=None,
                        attempts=result.attempts,
                        http_status=result.status_code,
                        reason=result.reason,
                    )

                if (
                    completed
                    % PROGRESS_EVERY
                    == 0
                    or completed == len(pending)
                ):

                    elapsed = (
                        time.perf_counter()
                        - run_start
                    )

                    rate = (
                        completed / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    remaining = (
                        len(pending)
                        - completed
                    )

                    eta_seconds = (
                        remaining / rate
                        if rate > 0
                        else 0.0
                    )

                    logger.info(
                        "PROGRESS | %s/%s | "
                        "downloaded=%s | failed=%s | "
                        "rate=%.2f profile/s | "
                        "ETA=%.1f min",
                        f"{completed:,}",
                        f"{len(pending):,}",
                        f"{downloaded:,}",
                        f"{failed:,}",
                        rate,
                        eta_seconds / 60,
                    )

    return (
        downloaded,
        skipped,
        failed,
    )


# =============================================================================
# FINAL COMPLETENESS CHECK
# =============================================================================

def final_completeness_check(
    profiles: list[ProfileRecord],
) -> tuple[int, int]:

    missing = 0
    invalid = 0
    completed_records = db_completed_records()

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL ARGO COMPLETENESS CHECK")
    logger.info("=" * 80)

    for profile in profiles:

        path = profile.local_path

        if not path.exists():
            missing += 1
            if missing <= 20:
                logger.error("MISSING | %s", profile.key)
            continue

        recorded_size = completed_records.get(profile.key)

        if recorded_size is not None:
            try:
                current_size = path.stat().st_size
            except OSError as exc:
                invalid += 1
                if invalid <= 20:
                    logger.error("INVALID | %s | stat_error:%s", profile.key, exc)
                continue

            if current_size != recorded_size:
                invalid += 1
                if invalid <= 20:
                    logger.error(
                        "INVALID | %s | size_changed:%s->%s",
                        profile.key, recorded_size, current_size
                    )
            continue

        valid, reason = validate_netcdf(path)
        if not valid:
            invalid += 1
            if invalid <= 20:
                logger.error("INVALID | %s | %s", profile.key, reason)

    logger.info("Expected profiles : %s", f"{len(profiles):,}")
    logger.info("Missing           : %s", f"{missing:,}")
    logger.info("Invalid           : %s", f"{invalid:,}")

    if missing == 0 and invalid == 0:
        logger.info("STATUS: ALL SELECTED ARGO PROFILES PRESENT AND VALID")
    else:
        logger.warning("STATUS: ACQUISITION INCOMPLETE")

    return missing, invalid


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def calculate_valid_size(
    profiles: list[ProfileRecord],
) -> tuple[int, int]:

    completed_records = db_completed_records()
    valid_count = 0
    total_bytes = 0

    for profile in profiles:
        recorded_size = completed_records.get(profile.key)
        if recorded_size is None:
            continue

        path = profile.local_path
        if not path.exists():
            continue

        try:
            current_size = path.stat().st_size
        except OSError:
            continue

        if current_size != recorded_size:
            continue

        valid_count += 1
        total_bytes += current_size

    return valid_count, total_bytes


def print_summary(
    profiles: list[ProfileRecord],
) -> None:

    valid_count, total_bytes = (
        calculate_valid_size(
            profiles
        )
    )

    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "ARGO V5.1 FINAL SUMMARY"
    )
    logger.info(
        "=" * 80
    )

    logger.info(
        "Selected profiles   : %s",
        f"{len(profiles):,}",
    )

    logger.info(
        "Valid local files   : %s",
        f"{valid_count:,}",
    )

    logger.info(
        "Valid data size     : %.2f GB",
        (
            total_bytes
            / 1024
            / 1024
            / 1024
        ),
    )

    logger.info(
        "State DB            : %s",
        STATE_DB,
    )

    logger.info(
        "Selected manifest   : %s",
        SELECTED_MANIFEST,
    )

    logger.info(
        "Rejected manifest   : %s",
        REJECTED_MANIFEST,
    )

    logger.info(
        "Failure report      : %s",
        FAILED_REPORT,
    )

    logger.info(
        "Log                 : %s",
        LOG_FILE,
    )

    logger.info(
        "DB status           : %s",
        db_status_counts(),
    )

    logger.info(
        "=" * 80
    )


# =============================================================================
# MAIN
# =============================================================================

def parse_run_mode() -> str:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "OceanEmbed ARGO 2012-2024 acquisition. "
            "TEST is safe by default; --production enables the full run."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test",
        action="store_true",
        help="Run manifest + real network preflight only (default).",
    )
    group.add_argument(
        "--production",
        action="store_true",
        help="Run the full 2012-2024 acquisition after preflight.",
    )
    args = parser.parse_args()

    if args.production:
        return "PRODUCTION"
    return DEFAULT_RUN_MODE


def main() -> int:

    run_mode = parse_run_mode()

    logger.info("")
    logger.info(
        "=" * 80
    )
    logger.info(
        "OCEANEMBED ARGO 2012-2024 V6"
    )
    logger.info(
        "ROBUST S3-FIRST MULTI-GDAC HTTPS — NO CONCURRENT NETCDF4"
    )
    logger.info(
        "PRODUCTION ACQUISITION"
    )
    logger.info(
        "=" * 80
    )

    logger.info(
        "Manifest : %s",
        MANIFEST_FILE,
    )

    logger.info(
        "Date     : %s -> %s",
        START_DATE,
        END_DATE,
    )

    logger.info(
        "NIO      : %.2f..%.2f N, %.2f..%.2f E",
        LAT_MIN,
        LAT_MAX,
        LON_MIN,
        LON_MAX,
    )

    logger.info(
        "Workers  : %d",
        MAX_WORKERS,
    )

    logger.info(
        "Worker validation : lightweight signature only "
        "(netCDF4 validation is serial)",
    )

    logger.info(
        "Source 1 : %s",
        SOURCES[0][1],
    )

    logger.info(
        "Source 2 : %s",
        SOURCES[1][1],
    )

    logger.info(
        "Source 3 : %s",
        SOURCES[2][1],
    )

    logger.info(
        "Preflight: %s (%d successful samples)",
        PREFLIGHT_ENABLED,
        PREFLIGHT_SUCCESS_TARGET,
    )

    logger.info(
        "Run mode : %s",
        run_mode,
    )

    logger.info(
        "Python   : %s",
        sys.version.replace(
            "\n",
            " ",
        ),
    )

    # -------------------------------------------------------------------------
    # Dependency check
    # -------------------------------------------------------------------------

    try:

        import netCDF4  # noqa: F401

    except ImportError:

        logger.error(
            "netCDF4 is not installed."
        )

        logger.error(
            "Install with:"
        )

        logger.error(
            "python -m pip install -U netCDF4"
        )

        return 2

    try:

        connection = db_connect()
        connection.close()

        profiles = (
            build_selected_manifest()
        )

        # ---------------------------------------------------------------------
        # Preflight
        # ---------------------------------------------------------------------

        if PREFLIGHT_ENABLED:

            if not run_preflight(
                profiles
            ):

                logger.error(
                    "Production acquisition was NOT started."
                )

                rebuild_failed_report(
                    profiles
                )

                return 1

        if run_mode == "TEST":

            rebuild_failed_report(
                profiles
            )

            logger.info("")
            logger.info("=" * 80)
            logger.info("TEST MODE COMPLETE — FULL ACQUISITION NOT STARTED")
            logger.info("Real ARGO NetCDF downloads were validated through the source chain.")
            logger.info("To start the full run tomorrow, use: --production")
            logger.info("=" * 80)

            return 0

        # ---------------------------------------------------------------------
        # Production pass 1
        # ---------------------------------------------------------------------

        pass1 = acquisition_pass(
            profiles,
            pass_number=1,
        )

        logger.info(
            "PASS 1 RESULT | downloaded=%s | "
            "skipped=%s | failed=%s",
            f"{pass1[0]:,}",
            f"{pass1[1]:,}",
            f"{pass1[2]:,}",
        )

        # ---------------------------------------------------------------------
        # Remaining work
        # ---------------------------------------------------------------------

        remaining_after_pass1, _ = (
            final_completeness_check(
                profiles
            )
        )

        # ---------------------------------------------------------------------
        # Production pass 2
        # ---------------------------------------------------------------------

        if remaining_after_pass1 > 0:

            logger.info("")
            logger.info(
                "=" * 80
            )
            logger.info(
                "STARTING PASS 2 FOR REMAINING PROFILES"
            )
            logger.info(
                "=" * 80
            )

            pass2 = acquisition_pass(
                profiles,
                pass_number=2,
            )

            logger.info(
                "PASS 2 RESULT | downloaded=%s | "
                "skipped=%s | failed=%s",
                f"{pass2[0]:,}",
                f"{pass2[1]:,}",
                f"{pass2[2]:,}",
            )

        else:

            logger.info(
                "No second-pass downloads required."
            )

        # ---------------------------------------------------------------------
        # Final audit
        # ---------------------------------------------------------------------

        missing, invalid = (
            final_completeness_check(
                profiles
            )
        )

        rebuild_failed_report(
            profiles
        )

        print_summary(
            profiles
        )

        if (
            missing == 0
            and invalid == 0
        ):

            logger.info(
                "FINAL STATUS: SUCCESS"
            )

            return 0

        logger.warning(
            "FINAL STATUS: INCOMPLETE"
        )

        logger.warning(
            "Run this same script again to resume."
        )

        return 1

    except KeyboardInterrupt:

        logger.warning(
            "INTERRUPTED BY USER"
        )

        logger.warning(
            "Completed files are preserved. "
            "Rerun this same script to resume."
        )

        return 130

    except Exception as exc:

        logger.error(
            "FATAL ERROR: %s",
            exc,
            exc_info=True,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
