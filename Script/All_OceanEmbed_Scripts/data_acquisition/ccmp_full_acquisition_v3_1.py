"""
================================================================================
OCEANEMBED CCMP V3.1
OFFICIAL PO.DAAC SUBSETTER + IMMEDIATE PROCESSING
================================================================================

Purpose
-------+
Acquire RSS CCMP V3.1 daily 6-hourly 10-m winds for:

    2012-01-01 .. 2024-12-31 inclusive

OceanEmbed output:
    North Indian Ocean: 45E..105E, 5N..30N
    0.25 degree target grid
    daily mean U/V from exactly 00/06/12/18 UTC

IMPORTANT DESIGN CHANGE FROM V3.0
---------------------------------
V3.0 manually called Harmony and failed because its hand-built request was not
accepted by the CCMP Harmony deployment.

V3.1 does NOT reverse-engineer Harmony.

It uses the official PO.DAAC `podaac-data-downloader --subset` interface.
PO.DAAC documents that `--subset` invokes NASA Harmony for server-side
granule subsetting, and the CCMP V3.1 collection explicitly advertises the
Harmony subset service.

The downloader receives:
    - exact CCMP granule name
    - OceanEmbed native support bounding box
    - --subset

Thus the global ~32 MB CCMP file is not intentionally downloaded for the
production path. The server returns the subsetted NetCDF.

AUTHENTICATION
--------------
Create an Earthdata Login netrc before running:

Windows:
    %USERPROFILE%\\_netrc

machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD

Install dependencies into the SAME Python used to run this script:

    python -m pip install -U podaac-data-subscriber requests xarray netCDF4 numpy

The PO.DAAC downloader command must be available:

    podaac-data-downloader --version

SAFETY
------
TEST_MODE is TRUE by default.

The test:
    1. downloads 2012-03-01 through official PO.DAAC downloader + --subset
    2. validates the subset
    3. downloads the official RSS global file for 2012-03-01
    4. processes both through the exact same OceanEmbed transform
    5. requires numerical agreement
    6. writes the processed test file only after all checks pass

Only after the test passes should TEST_MODE be changed to False.

RESUME / INTEGRITY
------------------
- Existing non-empty processed files are treated as complete.
- State is written atomically.
- State read/modify/write is locked as one transaction.
- Raw subset is deleted only after processed output is reopened and QC passes
  AND the completion state is committed.
- Failed days remain resumable.
- A downloaded source is reused if it is already present and structurally valid.
- Downloader checksum verification remains enabled by default; --force is NOT
  used.

SCIENTIFIC CONTRACT
-------------------
- Source: RSS CCMP 6-Hourly 10 Meter Surface Winds L4 V3.1
- PO.DAAC collection:
      CCMP_WINDS_10M6HR_L4_V3.1
- DOI:
      10.5067/CCMP-6HW10M-L4V31
- Native resolution:
      0.25 degree
- Native support requested:
      44.75E .. 105.25E
       4.75N ..  30.25N
- Exactly four analyses per day:
      00, 06, 12, 18 UTC
- U and V are interpolated separately to the OceanEmbed grid.
- Daily U and V are then averaged separately.
- Wind speed is not stored.
- No Harmony regridding/reprojection is requested.
- Final output contains only:
      uwnd, vwnd
- Final shape:
      time=1, latitude=101, longitude=241
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, local

import numpy as np
import requests
import xarray as xr


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\OceanEmbed\oceanembed-data")

RAW_ROOT = ROOT / "raw" / "ccmp"
TEMP_ROOT = RAW_ROOT / "_temp_v3_1"
PROCESSED_ROOT = RAW_ROOT / "processed"
STATE_ROOT = RAW_ROOT / "_state"
LOG_ROOT = RAW_ROOT / "_logs"

START_DATE = date(2012, 1, 1)
END_DATE = date(2024, 12, 31)

# SAFETY: keep TRUE for the first run.
TEST_MODE = True
TEST_DATE = date(2012, 3, 1)

# Official PO.DAAC collection.
COLLECTION_SHORTNAME = "CCMP_WINDS_10M6HR_L4_V3.1"
COLLECTION_ID = "C2916514952-POCLOUD"

# Official RSS source, used only for one-time numerical validation.
RSS_BASE = "https://data.remss.com/ccmp/v03.1"

# OceanEmbed target grid.
TARGET_LON_MIN = 45.0
TARGET_LON_MAX = 105.0
TARGET_LAT_MIN = 5.0
TARGET_LAT_MAX = 30.0
TARGET_RESOLUTION = 0.25

# Native support around the target grid.
# This is deliberately retained from the audited V2/V3 scientific contract.
NATIVE_BUFFER = 0.25

# Start conservatively. After a successful test, 6 is a good production
# starting point. Do not jump to dozens of workers.
MAX_WORKERS = 6

# Downloader process timeout. Increase only if the service is demonstrably
# taking longer than this for a single granule.
DOWNLOADER_TIMEOUT_SECONDS = 20 * 60

# Network timeout for the one-time RSS reference download.
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
NETCDF_COMPRESSION_LEVEL = 4

KEEP_RAW_AFTER_PROCESSING = False

VALIDATION_ATOL = 1e-6
VALIDATION_RTOL = 1e-6

STATE_FILE = STATE_ROOT / "ccmp_v3_1_state.json"
LOG_FILE = LOG_ROOT / "ccmp_v3_1.log"

USER_AGENT = "OceanEmbed-CCMP-V3.1/1.0"

STATE_LOCK = Lock()
THREAD_LOCAL = local()


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("oceanembed_ccmp_v3_1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# =============================================================================
# HELPERS
# =============================================================================

def human_size(value: int | float | None) -> str:
    if value is None:
        return "unknown"

    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_granule_name(target_date: date) -> str:
    return f"CCMP_Wind_Analysis_{target_date:%Y%m%d}_V03.1_L4.nc"


def build_rss_url(target_date: date) -> str:
    return (
        f"{RSS_BASE}/"
        f"Y{target_date.year:04d}/"
        f"M{target_date.month:02d}/"
        f"CCMP_Wind_Analysis_{target_date:%Y%m%d}_V03.1_L4.nc"
    )


def build_temp_day_dir(target_date: date) -> Path:
    day_dir = TEMP_ROOT / f"{target_date:%Y}" / f"{target_date:%m}" / f"{target_date:%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def build_processed_path(target_date: date) -> Path:
    day_dir = PROCESSED_ROOT / f"{target_date:%Y}" / f"{target_date:%m}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"CCMP_daily_NIO_{target_date:%Y%m%d}.nc"


def build_downloader_output_dir(target_date: date) -> Path:
    return build_temp_day_dir(target_date) / "downloader"


def build_expected_subset_path(target_date: date) -> Path:
    return build_temp_day_dir(target_date) / build_granule_name(target_date)


def build_rss_reference_path(target_date: date) -> Path:
    return build_temp_day_dir(target_date) / (
        f"CCMP_Wind_Analysis_{target_date:%Y%m%d}_RSS_GLOBAL.nc"
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
            }
        )
        session.trust_env = True
        THREAD_LOCAL.session = session

    return session


def is_complete(target_date: date) -> bool:
    path = build_processed_path(target_date)

    if not path.exists():
        return False

    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def mark_complete(target_date: date, processed_path: Path) -> None:
    """
    Full read-modify-write transaction under one lock.
    """
    with STATE_LOCK:
        if STATE_FILE.exists():
            try:
                with STATE_FILE.open("r", encoding="utf-8") as handle:
                    state = json.load(handle)
            except Exception as exc:
                logger.warning("State file could not be read during commit: %s", exc)
                state = {"completed": {}}
        else:
            state = {"completed": {}}

        if not isinstance(state, dict):
            state = {"completed": {}}

        state.setdefault("completed", {})
        state["completed"][target_date.isoformat()] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "processed_file": str(processed_path),
            "processed_size_bytes": processed_path.stat().st_size,
            "processed_sha256": file_sha256(processed_path),
            "pipeline_version": "CCMP_V3.1",
        }

        atomic_write_json(STATE_FILE, state)


# =============================================================================
# OFFICIAL PO.DAAC DOWNLOADER DISCOVERY
# =============================================================================

def find_downloader() -> str | None:
    """
    Find the official PO.DAAC CLI in a Windows-safe way.

    We first use PATH, then the Scripts directory belonging to the exact Python
    interpreter running this program.
    """
    candidates = [
        "podaac-data-downloader",
        "podaac-data-downloader.exe",
    ]

    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    scripts_dir = Path(sys.executable).resolve().parent / "Scripts"

    for name in candidates:
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)

    return None


def verify_downloader() -> str:
    executable = find_downloader()

    if not executable:
        raise RuntimeError(
            "\n"
            "Official PO.DAAC downloader was not found.\n\n"
            "Install it into THIS Python environment with:\n"
            f'  "{sys.executable}" -m pip install -U podaac-data-subscriber\n\n'
            "Then verify with:\n"
            "  podaac-data-downloader --version\n"
        )

    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not execute official PO.DAAC downloader: {executable}\n{exc}"
        ) from exc

    version_text = (result.stdout or result.stderr).strip()

    if result.returncode != 0:
        raise RuntimeError(
            "Official PO.DAAC downloader exists but --version failed.\n"
            f"Executable: {executable}\n"
            f"Return code: {result.returncode}\n"
            f"Output:\n{version_text}"
        )

    logger.info("PO.DAAC downloader: %s", executable)
    logger.info("PO.DAAC downloader version: %s", version_text)

    return executable


# =============================================================================
# OFFICIAL PO.DAAC SUBSET DOWNLOAD
# =============================================================================

def downloader_command(
    executable: str,
    target_date: date,
    output_dir: Path,
) -> list[str]:
    west = TARGET_LON_MIN - NATIVE_BUFFER
    south = TARGET_LAT_MIN - NATIVE_BUFFER
    east = TARGET_LON_MAX + NATIVE_BUFFER
    north = TARGET_LAT_MAX + NATIVE_BUFFER

    bbox = f"{west:.2f},{south:.2f},{east:.2f},{north:.2f}"

    return [
        executable,
        "-c",
        COLLECTION_SHORTNAME,
        "-d",
        str(output_dir),
        "-gr",
        build_granule_name(target_date),
        "-b",
        bbox,
        "--subset",
        "-e",
        ".nc",
    ]


def locate_downloaded_netcdf(
    target_date: date,
    output_dir: Path,
) -> Path | None:
    expected_name = build_granule_name(target_date)

    # Preferred: exact expected filename.
    direct = output_dir / expected_name
    if direct.exists() and direct.stat().st_size > 0:
        return direct

    # The downloader may create a nested directory structure.
    matches = list(output_dir.rglob(expected_name))
    matches = [
        path for path in matches
        if path.is_file() and path.stat().st_size > 0
    ]

    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    # Last-resort: identify an NC file containing the exact granule date.
    date_token = f"{target_date:%Y%m%d}"
    candidates = [
        path
        for path in output_dir.rglob("*.nc")
        if path.is_file()
        and path.stat().st_size > 0
        and date_token in path.name
    ]

    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    return None


def validate_netcdf_magic(path: Path) -> None:
    """
    Prevent an HTML/JSON/error response from being treated as NetCDF.
    """
    with path.open("rb") as handle:
        magic = handle.read(8)

    if magic[:3] != b"CDF":
        raise RuntimeError(
            f"Downloaded file is not classic NetCDF/HDF5-compatible: "
            f"{path} | first bytes={magic!r}"
        )


def download_subset(
    target_date: date,
    executable: str,
    destination: Path,
) -> bool:
    """
    Run the official PO.DAAC downloader for exactly one CCMP granule.

    The downloader itself handles Earthdata authentication, CMR discovery,
    checksum handling, and Harmony subsetting.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_dir = build_downloader_output_dir(target_date)

    # If a previous failed run left an output tree, keep it: the downloader
    # itself can verify/reuse matching products. We only remove an invalid
    # final destination if one exists.
    if destination.exists():
        try:
            validate_netcdf_magic(destination)
            if destination.stat().st_size > 0:
                return True
        except Exception:
            try:
                destination.unlink()
            except OSError:
                pass

    command = downloader_command(executable, target_date, output_dir)

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.perf_counter()

        logger.info(
            "[%s] PO.DAAC subset attempt %d/%d",
            target_date,
            attempt,
            MAX_RETRIES,
        )
        logger.info(
            "[%s] Native subset bbox: %.2f,%.2f,%.2f,%.2f",
            target_date,
            TARGET_LON_MIN - NATIVE_BUFFER,
            TARGET_LAT_MIN - NATIVE_BUFFER,
            TARGET_LON_MAX + NATIVE_BUFFER,
            TARGET_LAT_MAX + NATIVE_BUFFER,
        )

        env = os.environ.copy()
        env.setdefault("PODAAC_LOGLEVEL", "INFO")

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DOWNLOADER_TIMEOUT_SECONDS,
                check=False,
                env=env,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if stdout.strip():
                for line in stdout.splitlines()[-30:]:
                    logger.info("[%s] downloader: %s", target_date, line)

            if result.returncode != 0:
                logger.warning(
                    "[%s] downloader failed with return code %d",
                    target_date,
                    result.returncode,
                )
                if stderr.strip():
                    for line in stderr.splitlines()[-30:]:
                        logger.warning(
                            "[%s] downloader stderr: %s",
                            target_date,
                            line,
                        )
                raise RuntimeError(
                    f"PO.DAAC downloader returned {result.returncode}"
                )

            downloaded = locate_downloaded_netcdf(target_date, output_dir)

            if downloaded is None:
                raise RuntimeError(
                    "PO.DAAC downloader reported success but no matching "
                    f"NetCDF was found under {output_dir}"
                )

            validate_netcdf_magic(downloaded)

            # Move/copy to a deterministic per-day source path.
            # replace() is atomic on the same filesystem.
            temporary = destination.with_suffix(destination.suffix + ".part")

            if temporary.exists():
                temporary.unlink()

            shutil.copy2(downloaded, temporary)
            validate_netcdf_magic(temporary)
            temporary.replace(destination)

            elapsed = time.perf_counter() - started

            logger.info(
                "[%s] PO.DAAC subset ready: %s in %.1fs",
                target_date,
                human_size(destination.stat().st_size),
                elapsed,
            )

            return True

        except subprocess.TimeoutExpired:
            logger.warning(
                "[%s] downloader timed out after %ds",
                target_date,
                DOWNLOADER_TIMEOUT_SECONDS,
            )

        except Exception as exc:
            logger.warning(
                "[%s] subset attempt failed: %s",
                target_date,
                exc,
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error(
        "[%s] PO.DAAC subset FAILED after %d attempts",
        target_date,
        MAX_RETRIES,
    )
    return False


# =============================================================================
# ONE-TIME DIRECT RSS VALIDATION DOWNLOAD
# =============================================================================

def download_direct_rss(target_date: date, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    session = get_session()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if temporary.exists():
                temporary.unlink()

            started = time.perf_counter()

            response = session.get(
                build_rss_url(target_date),
                stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=True,
            )
            response.raise_for_status()

            expected = response.headers.get("Content-Length")
            expected_size = int(expected) if expected else None
            downloaded = 0

            with temporary.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE
                ):
                    if chunk:
                        handle.write(chunk)
                        downloaded += len(chunk)

            response.close()

            if downloaded <= 0:
                raise RuntimeError("RSS reference download returned zero bytes")

            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(
                    f"RSS size mismatch: expected {expected_size}, "
                    f"received {downloaded}"
                )

            validate_netcdf_magic(temporary)
            temporary.replace(destination)

            elapsed = time.perf_counter() - started

            logger.info(
                "[%s] RSS validation source downloaded: %s in %.1fs",
                target_date,
                human_size(downloaded),
                elapsed,
            )

            return True

        except Exception as exc:
            logger.warning(
                "[%s] RSS reference attempt %d/%d failed: %s",
                target_date,
                attempt,
                MAX_RETRIES,
                exc,
            )

            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return False


# =============================================================================
# SOURCE VALIDATION
# =============================================================================

def expected_target_grid() -> tuple[np.ndarray, np.ndarray]:
    target_lats = np.arange(
        TARGET_LAT_MIN,
        TARGET_LAT_MAX + TARGET_RESOLUTION / 2,
        TARGET_RESOLUTION,
        dtype=np.float32,
    )

    target_lons = np.arange(
        TARGET_LON_MIN,
        TARGET_LON_MAX + TARGET_RESOLUTION / 2,
        TARGET_RESOLUTION,
        dtype=np.float32,
    )

    return target_lats, target_lons


def normalize_source_coordinates(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" not in ds.coords and "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})

    if "longitude" not in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError(
            "Source is missing latitude/longitude coordinates"
        )

    if ds["latitude"].ndim != 1 or ds["longitude"].ndim != 1:
        raise RuntimeError(
            "Expected 1-D latitude/longitude coordinates"
        )

    return ds.sortby("latitude").sortby("longitude")


def validate_source_structure(
    ds: xr.Dataset,
    target_date: date,
    require_target_support: bool = True,
) -> None:
    required_coordinates = {"time", "latitude", "longitude"}
    missing = required_coordinates - set(ds.coords)

    if missing:
        raise RuntimeError(
            f"Missing coordinates: {sorted(missing)}"
        )

    required_variables = {"uwnd", "vwnd"}
    missing = required_variables - set(ds.data_vars)

    if missing:
        raise RuntimeError(
            f"Missing required variables: {sorted(missing)}"
        )

    if ds.sizes.get("time") != 4:
        raise RuntimeError(
            "Expected exactly 4 CCMP analyses; "
            f"found {ds.sizes.get('time')}"
        )

    time_values = ds["time"].values

    actual_hours = {
        int(
            np.datetime64(value, "h")
            .astype("datetime64[h]")
            .astype(int)
            % 24
        )
        for value in time_values
    }

    if actual_hours != {0, 6, 12, 18}:
        raise RuntimeError(
            f"Unexpected CCMP time hours: {sorted(actual_hours)}"
        )

    first_day = np.datetime64(time_values[0], "D")
    expected_day = np.datetime64(target_date.isoformat(), "D")

    if first_day != expected_day:
        raise RuntimeError(
            f"File date mismatch: expected {expected_day}, found {first_day}"
        )

    for variable in ("uwnd", "vwnd"):
        expected_dims = ("time", "latitude", "longitude")

        if ds[variable].dims != expected_dims:
            raise RuntimeError(
                f"{variable} dimensions are {ds[variable].dims}; "
                f"expected {expected_dims}"
            )

        values = ds[variable].values

        if not np.isfinite(values).any():
            raise RuntimeError(
                f"{variable} contains no finite values"
            )

        variable_min = float(np.nanmin(values))
        variable_max = float(np.nanmax(values))

        if variable_min < -100 or variable_max > 100:
            raise RuntimeError(
                f"{variable} has suspicious values: "
                f"{variable_min} to {variable_max} m/s"
            )

    if require_target_support:
        lat_values = ds["latitude"].values
        lon_values = ds["longitude"].values

        if lat_values.min() > TARGET_LAT_MIN:
            raise RuntimeError(
                "Subset lacks southern interpolation support"
            )

        if lat_values.max() < TARGET_LAT_MAX:
            raise RuntimeError(
                "Subset lacks northern interpolation support"
            )

        if lon_values.min() > TARGET_LON_MIN:
            raise RuntimeError(
                "Subset lacks western interpolation support"
            )

        if lon_values.max() < TARGET_LON_MAX:
            raise RuntimeError(
                "Subset lacks eastern interpolation support"
            )


# =============================================================================
# PROCESSING
# =============================================================================

def daily_oceanembed_uv(
    source_path: Path,
    target_date: date,
) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(
        source_path,
        engine="netcdf4",
        decode_times=True,
        mask_and_scale=True,
    ) as ds:
        ds = normalize_source_coordinates(ds)

        # For the RSS global reference, this validates the same four times but
        # must not require the NIO subset bounds.
        require_support = (
            ds["latitude"].min().item() <= TARGET_LAT_MIN
            and ds["latitude"].max().item() >= TARGET_LAT_MAX
            and ds["longitude"].min().item() <= TARGET_LON_MIN
            and ds["longitude"].max().item() >= TARGET_LON_MAX
        )

        validate_source_structure(
            ds,
            target_date,
            require_target_support=require_support,
        )

        target_lats, target_lons = expected_target_grid()

        regional = ds[["uwnd", "vwnd"]].sel(
            latitude=slice(
                TARGET_LAT_MIN - NATIVE_BUFFER,
                TARGET_LAT_MAX + NATIVE_BUFFER,
            ),
            longitude=slice(
                TARGET_LON_MIN - NATIVE_BUFFER,
                TARGET_LON_MAX + NATIVE_BUFFER,
            ),
        )

        if (
            regional.sizes.get("latitude", 0) < 2
            or regional.sizes.get("longitude", 0) < 2
        ):
            raise RuntimeError(
                "Source does not contain enough spatial cells for linear "
                "interpolation to the OceanEmbed target grid"
            )

        # Preserve the audited scientific operation:
        # native CCMP -> linear interpolation -> daily vector mean.
        regional = regional.interp(
            latitude=xr.DataArray(target_lats, dims="latitude"),
            longitude=xr.DataArray(target_lons, dims="longitude"),
            method="linear",
        )

        daily_u = (
            regional["uwnd"]
            .mean(dim="time", skipna=True)
            .astype("float32")
            .values
        )

        daily_v = (
            regional["vwnd"]
            .mean(dim="time", skipna=True)
            .astype("float32")
            .values
        )

        if not np.isfinite(daily_u).any():
            raise RuntimeError("Daily U field contains no finite values")

        if not np.isfinite(daily_v).any():
            raise RuntimeError("Daily V field contains no finite values")

        return daily_u, daily_v


def write_processed(
    target_date: date,
    daily_u: np.ndarray,
    daily_v: np.ndarray,
) -> Path:
    target_lats, target_lons = expected_target_grid()

    expected_shape = (
        len(target_lats),
        len(target_lons),
    )

    if daily_u.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected daily U shape: {daily_u.shape}; "
            f"expected {expected_shape}"
        )

    if daily_v.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected daily V shape: {daily_v.shape}; "
            f"expected {expected_shape}"
        )

    processed_path = build_processed_path(target_date)
    temporary = processed_path.with_suffix(processed_path.suffix + ".part")

    if temporary.exists():
        temporary.unlink()

    output = xr.Dataset(
        data_vars={
            "uwnd": (
                ("time", "latitude", "longitude"),
                daily_u[np.newaxis, :, :],
            ),
            "vwnd": (
                ("time", "latitude", "longitude"),
                daily_v[np.newaxis, :, :],
            ),
        },
        coords={
            "time": [np.datetime64(target_date.isoformat())],
            "latitude": target_lats,
            "longitude": target_lons,
        },
        attrs={
            "title": "OceanEmbed processed CCMP V3.1 daily surface winds",
            "source": "RSS CCMP 6-Hourly 10 Meter Surface Winds L4 V3.1",
            "source_doi": "10.5067/CCMP-6HW10M-L4V31",
            "source_url": RSS_BASE,
            "podaac_collection": COLLECTION_SHORTNAME,
            "podaac_concept_id": COLLECTION_ID,
            "processing": (
                "Official PO.DAAC downloader --subset / NASA Harmony "
                "native spatial subset, linear spatial interpolation to "
                "OceanEmbed 0.25 degree grid, daily vector mean of "
                "00/06/12/18 UTC"
            ),
            "region": "North Indian Ocean: 45E-105E, 5N-30N",
            "spatial_resolution": "0.25 degree",
            "temporal_resolution": "daily",
            "variables": "uwnd, vwnd",
            "wind_speed_note": (
                "Wind speed can be derived as sqrt(uwnd^2 + vwnd^2)."
            ),
            "pipeline_version": "OceanEmbed CCMP V3.1",
        },
    )

    output["uwnd"].attrs = {
        "long_name": "Daily mean eastward wind component at 10 meters",
        "standard_name": "eastward_wind",
        "units": "m s-1",
    }

    output["vwnd"].attrs = {
        "long_name": "Daily mean northward wind component at 10 meters",
        "standard_name": "northward_wind",
        "units": "m s-1",
    }

    output["latitude"].attrs = {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }

    output["longitude"].attrs = {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
    }

    output["time"].attrs = {"standard_name": "time"}

    encoding = {
        "uwnd": {
            "dtype": "float32",
            "zlib": True,
            "complevel": NETCDF_COMPRESSION_LEVEL,
            "shuffle": True,
        },
        "vwnd": {
            "dtype": "float32",
            "zlib": True,
            "complevel": NETCDF_COMPRESSION_LEVEL,
            "shuffle": True,
        },
        "latitude": {"dtype": "float32"},
        "longitude": {"dtype": "float32"},
    }

    try:
        output.to_netcdf(
            temporary,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding,
        )
        output.close()
        temporary.replace(processed_path)
    except Exception:
        try:
            output.close()
        except Exception:
            pass

        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass

        raise

    # Final reopen/QC.
    with xr.open_dataset(processed_path, engine="netcdf4") as final_ds:
        expected_final_shape = (
            1,
            len(target_lats),
            len(target_lons),
        )

        if final_ds["uwnd"].shape != expected_final_shape:
            raise RuntimeError(
                f"Unexpected final U shape: {final_ds['uwnd'].shape}; "
                f"expected {expected_final_shape}"
            )

        if final_ds["vwnd"].shape != expected_final_shape:
            raise RuntimeError(
                f"Unexpected final V shape: {final_ds['vwnd'].shape}; "
                f"expected {expected_final_shape}"
            )

        if set(final_ds.data_vars) != {"uwnd", "vwnd"}:
            raise RuntimeError(
                f"Unexpected final data variables: {set(final_ds.data_vars)}"
            )

        for variable in ("uwnd", "vwnd"):
            values = final_ds[variable].values

            if not np.isfinite(values).any():
                raise RuntimeError(
                    f"{variable} has no finite values in final output"
                )

    return processed_path


# =============================================================================
# SIDE-BY-SIDE VALIDATION
# =============================================================================

def compare_subset_against_rss(
    target_date: date,
    subset_path: Path,
    rss_path: Path,
) -> None:
    logger.info(
        "[%s] Running PO.DAAC subset vs official RSS comparison...",
        target_date,
    )

    subset_u, subset_v = daily_oceanembed_uv(
        subset_path,
        target_date,
    )

    rss_u, rss_v = daily_oceanembed_uv(
        rss_path,
        target_date,
    )

    for variable, actual, reference in (
        ("uwnd", subset_u, rss_u),
        ("vwnd", subset_v, rss_v),
    ):
        if actual.shape != reference.shape:
            raise RuntimeError(
                f"{variable} shape mismatch: "
                f"subset={actual.shape}, RSS={reference.shape}"
            )

        difference = np.abs(
            actual.astype(np.float64)
            - reference.astype(np.float64)
        )

        max_abs = float(np.nanmax(difference))
        mean_abs = float(np.nanmean(difference))

        if not np.allclose(
            actual,
            reference,
            rtol=VALIDATION_RTOL,
            atol=VALIDATION_ATOL,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"{variable} subset/RSS mismatch. "
                f"max_abs_difference={max_abs:.8g}, "
                f"mean_abs_difference={mean_abs:.8g}, "
                f"rtol={VALIDATION_RTOL}, atol={VALIDATION_ATOL}"
            )

        logger.info(
            "[%s] %s subset/RSS PASS | max_abs_diff=%.8g | mean_abs_diff=%.8g",
            target_date,
            variable,
            max_abs,
            mean_abs,
        )


# =============================================================================
# TEST MODE
# =============================================================================

def run_test(downloader: str) -> int:
    target_date = TEST_DATE

    subset_path = build_expected_subset_path(target_date)
    rss_path = build_rss_reference_path(target_date)

    logger.info("")
    logger.info("=" * 90)
    logger.info("CCMP V3.1 VALIDATION TEST")
    logger.info("=" * 90)
    logger.info("Test date: %s", target_date)
    logger.info("Collection: %s", COLLECTION_SHORTNAME)
    logger.info(
        "Native support: %.2f..%.2f E / %.2f..%.2f N",
        TARGET_LON_MIN - NATIVE_BUFFER,
        TARGET_LON_MAX + NATIVE_BUFFER,
        TARGET_LAT_MIN - NATIVE_BUFFER,
        TARGET_LAT_MAX + NATIVE_BUFFER,
    )
    logger.info("Official path: podaac-data-downloader --subset")

    if not download_subset(
        target_date,
        downloader,
        subset_path,
    ):
        logger.error("V3.1 TEST FAILED: PO.DAAC subset download")
        return 1

    try:
        with xr.open_dataset(
            subset_path,
            engine="netcdf4",
            decode_times=True,
            mask_and_scale=True,
        ) as ds:
            ds = normalize_source_coordinates(ds)
            validate_source_structure(
                ds,
                target_date,
                require_target_support=True,
            )

            logger.info(
                "Subset structure PASS | dims=%s | variables=%s",
                dict(ds.sizes),
                sorted(ds.data_vars),
            )

            logger.info(
                "Subset size: %s",
                human_size(subset_path.stat().st_size),
            )

        if not download_direct_rss(target_date, rss_path):
            logger.error("V3.1 TEST FAILED: direct RSS reference download")
            return 1

        compare_subset_against_rss(
            target_date,
            subset_path,
            rss_path,
        )

        daily_u, daily_v = daily_oceanembed_uv(
            subset_path,
            target_date,
        )

        processed_path = write_processed(
            target_date,
            daily_u,
            daily_v,
        )

        # Commit test completion only after the complete output passes QC.
        mark_complete(target_date, processed_path)

        logger.info(
            "[%s] V3.1 END-TO-END TEST PASS | %s | %s",
            target_date,
            processed_path,
            human_size(processed_path.stat().st_size),
        )

        logger.info("")
        logger.info("=" * 90)
        logger.info("CCMP V3.1 TEST PASSED")
        logger.info("=" * 90)
        logger.info(
            "You may now change TEST_MODE = False and run the full acquisition."
        )

        return 0

    except Exception as exc:
        logger.error(
            "V3.1 TEST FAILED: %s",
            exc,
            exc_info=True,
        )
        return 1

    finally:
        # Test-only sources are disposable. The processed output remains.
        for path in (subset_path, rss_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

        # The downloader may have created nested files. Remove only the test
        # downloader directory, never the production processed tree.
        test_downloader_dir = build_downloader_output_dir(target_date)
        if test_downloader_dir.exists():
            shutil.rmtree(test_downloader_dir, ignore_errors=True)


# =============================================================================
# FULL ONE-DAY TRANSACTION
# =============================================================================

def process_one_day(
    target_date: date,
    downloader: str,
) -> tuple[date, bool, str]:
    processed_path = build_processed_path(target_date)

    if is_complete(target_date):
        return target_date, True, "already complete"

    source_path = build_expected_subset_path(target_date)

    started = time.perf_counter()

    logger.info("[%s] START", target_date)

    try:
        if not source_path.exists():
            if not download_subset(
                target_date,
                downloader,
                source_path,
            ):
                return target_date, False, "PO.DAAC subset download failed"
        else:
            logger.info(
                "[%s] Existing subset source found: %s",
                target_date,
                human_size(source_path.stat().st_size),
            )

        # Never process an obviously invalid source.
        validate_netcdf_magic(source_path)

        daily_u, daily_v = daily_oceanembed_uv(
            source_path,
            target_date,
        )

        processed_path = write_processed(
            target_date,
            daily_u,
            daily_v,
        )

        # State commit happens before raw deletion.
        mark_complete(
            target_date,
            processed_path,
        )

    except Exception as exc:
        logger.error(
            "[%s] PROCESSING FAILED: %s",
            target_date,
            exc,
            exc_info=True,
        )
        return target_date, False, "processing failed"

    if not KEEP_RAW_AFTER_PROCESSING:
        try:
            source_size = source_path.stat().st_size
            source_path.unlink()

            logger.info(
                "[%s] Deleted raw subset: %s",
                target_date,
                human_size(source_size),
            )
        except Exception as exc:
            logger.warning(
                "[%s] Could not delete raw subset: %s",
                target_date,
                exc,
            )

        downloader_dir = build_downloader_output_dir(target_date)
        if downloader_dir.exists():
            shutil.rmtree(downloader_dir, ignore_errors=True)

    elapsed = time.perf_counter() - started

    logger.info(
        "[%s] COMPLETE in %.1fs",
        target_date,
        elapsed,
    )

    return target_date, True, "complete"


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    all_dates: list[date],
    results: list[tuple[date, bool, str]],
) -> None:
    successful = [item for item in results if item[1]]
    failed = [item for item in results if not item[1]]

    processed_files = 0
    total_processed_bytes = 0

    for target_date in all_dates:
        path = build_processed_path(target_date)

        if path.exists():
            try:
                size = path.stat().st_size
            except OSError:
                continue

            if size > 0:
                processed_files += 1
                total_processed_bytes += size

    logger.info("")
    logger.info("=" * 90)
    logger.info("CCMP V3.1 ACQUISITION SUMMARY")
    logger.info("=" * 90)
    logger.info("Requested dates : %d", len(all_dates))
    logger.info("Successful      : %d", len(successful))
    logger.info("Failed          : %d", len(failed))
    logger.info("Processed files : %d", processed_files)
    logger.info("Processed size  : %s", human_size(total_processed_bytes))
    logger.info("State file      : %s", STATE_FILE)
    logger.info("Log file        : %s", LOG_FILE)

    if failed:
        logger.error("")
        logger.error("FAILED DATES:")

        for target_date, _, reason in sorted(failed):
            logger.error(
                "  %s -> %s",
                target_date,
                reason,
            )

    logger.info("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    logger.info("")
    logger.info("=" * 90)
    logger.info("OCEANEMBED CCMP V3.1")
    logger.info("OFFICIAL PO.DAAC SUBSET + IMMEDIATE PROCESSING")
    logger.info("=" * 90)

    logger.info("Source: RSS CCMP V3.1 / PO.DAAC")
    logger.info("Collection: %s", COLLECTION_SHORTNAME)
    logger.info("Collection concept: %s", COLLECTION_ID)
    logger.info("DOI: 10.5067/CCMP-6HW10M-L4V31")

    logger.info(
        "Target region: %.2f..%.2f E, %.2f..%.2f N",
        TARGET_LON_MIN,
        TARGET_LON_MAX,
        TARGET_LAT_MIN,
        TARGET_LAT_MAX,
    )

    logger.info(
        "Native support bbox: %.2f,%.2f,%.2f,%.2f",
        TARGET_LON_MIN - NATIVE_BUFFER,
        TARGET_LAT_MIN - NATIVE_BUFFER,
        TARGET_LON_MAX + NATIVE_BUFFER,
        TARGET_LAT_MAX + NATIVE_BUFFER,
    )

    logger.info("Workers: %d", MAX_WORKERS)
    logger.info("TEST_MODE: %s", TEST_MODE)
    logger.info("Python: %s", sys.version.replace("\n", " "))

    ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        downloader = verify_downloader()
    except Exception as exc:
        logger.error("%s", exc)
        return 2

    if TEST_MODE:
        return run_test(downloader)

    dates = list(date_range(START_DATE, END_DATE))

    pending_dates = [
        target_date
        for target_date in dates
        if not is_complete(target_date)
    ]

    logger.info("Total calendar days: %d", len(dates))
    logger.info(
        "Already complete: %d",
        len(dates) - len(pending_dates),
    )
    logger.info("Pending: %d", len(pending_dates))

    if not pending_dates:
        print_summary(
            dates,
            [
                (target_date, True, "already complete")
                for target_date in dates
            ],
        )
        return 0

    results: list[tuple[date, bool, str]] = []

    logger.info(
        "Starting %d workers...",
        MAX_WORKERS,
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                process_one_day,
                target_date,
                downloader,
            ): target_date
            for target_date in pending_dates
        }

        for future in as_completed(future_map):
            target_date = future_map[future]

            try:
                result = future.result()
                results.append(result)

                _, success, message = result

                if success:
                    logger.info(
                        "FINISHED %s -> %s",
                        target_date,
                        message,
                    )
                else:
                    logger.error(
                        "FAILED %s -> %s",
                        target_date,
                        message,
                    )

            except Exception as exc:
                logger.error(
                    "UNEXPECTED ERROR for %s: %s",
                    target_date,
                    exc,
                    exc_info=True,
                )
                results.append(
                    (
                        target_date,
                        False,
                        f"unexpected error: {exc}",
                    )
                )

    print_summary(dates, results)

    return 1 if any(not result[1] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
