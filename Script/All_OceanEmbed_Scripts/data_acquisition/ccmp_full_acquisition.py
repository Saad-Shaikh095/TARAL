"""
===============================================================================
OCEANEMBED CCMP V3.1
FULL 2012-2024 ACQUISITION + IMMEDIATE PROCESSING PIPELINE
===============================================================================

Purpose
-------
Download CCMP V3.1 daily files for 2012-01-01 through 2024-12-31.

For every source file:

    1. Download official RSS CCMP daily NetCDF.
    2. Verify HTTP/download size.
    3. Open source NetCDF.
    4. Extract OceanEmbed NIO region.
    5. Regrid from native CCMP 0.25° cell centers to the
       OceanEmbed 0.25° target grid.
    6. Average 00/06/12/18 UTC vector winds into one daily U/V field.
    7. Save compressed float32 processed NetCDF.
    8. Delete the large global source file immediately.
    9. Mark the date complete.

Final processed variables
--------------------------
    uwnd : daily mean eastward wind component at 10 m
    vwnd : daily mean northward wind component at 10 m

NOT stored permanently
----------------------
    ws   : derived from U/V when needed
    nobs : retained only for source/QC checks
    global raw CCMP files : deleted after successful processing

Project target
--------------
    Longitude : 45.00E -> 105.00E
    Latitude  : 5.00N  -> 30.00N
    Resolution: 0.25°
    Time      : daily

Source
------
    RSS CCMP V3.1
    https://data.remss.com/ccmp/v03.1/

Official PO.DAAC collection
---------------------------
    CCMP_WINDS_10M6HR_L4_V3.1

DOI
---
    10.5067/CCMP-6HW10M-L4V31

IMPORTANT
---------
Run the January 2012 test first before changing TEST_MODE=False.

Dependencies
------------
    pip install xarray netCDF4 numpy

===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import xarray as xr


# =============================================================================
# CONFIGURATION
# =============================================================================

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------

ROOT = Path(r"D:\OceanEmbed\oceanembed-data")

RAW_ROOT = ROOT / "raw" / "ccmp"

# Temporary source downloads.
TEMP_ROOT = RAW_ROOT #/ "_temp"

# Small processed OceanEmbed files.
PROCESSED_ROOT = RAW_ROOT / "processed"

# Resume state.
STATE_ROOT = RAW_ROOT / "_state"

# Logs.
LOG_ROOT = RAW_ROOT / "_logs"


# ---------------------------------------------------------------------------
# Time range
# ---------------------------------------------------------------------------

START_DATE = date(2012, 1, 1)
END_DATE = date(2024, 12, 31)


# ---------------------------------------------------------------------------
# TEST MODE
# ---------------------------------------------------------------------------
#
# VERY IMPORTANT:
#
# Keep this True for the first run.
#
# It will process only:
#
#     2012-03-01
#
# because we already verified that source file.
#
# After the resulting processed file passes inspection:
#
#     TEST_MODE = False
#
# and the full 2012-2024 acquisition will run.
#
# ---------------------------------------------------------------------------

TEST_MODE = False

TEST_DATE = date(2012, 3, 1)


# ---------------------------------------------------------------------------
# Official RSS source
# ---------------------------------------------------------------------------

BASE_URL = "https://data.remss.com/ccmp/v03.1"


# ---------------------------------------------------------------------------
# OceanEmbed target grid
# ---------------------------------------------------------------------------

TARGET_LON_MIN = 45.0
TARGET_LON_MAX = 105.0

TARGET_LAT_MIN = 5.0
TARGET_LAT_MAX = 30.0

TARGET_RESOLUTION = 0.25


# ---------------------------------------------------------------------------
# Download configuration
# ---------------------------------------------------------------------------

# Start conservatively.
#
# 6 workers is intentionally chosen rather than 20+ because each worker can
# simultaneously download, decompress, read and write NetCDF.
#
# Increase only after observing CPU/network/disk behavior.
#
MAX_WORKERS = 6

CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5

DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB


# ---------------------------------------------------------------------------
# Processing configuration
# ---------------------------------------------------------------------------

# NetCDF compression level.
#
# 4 is a good balance between file size and CPU time.
#
# 1 = faster / larger
# 4 = balanced
# 6 = smaller / slower
# 9 = maximum compression / considerably slower
#
NETCDF_COMPRESSION_LEVEL = 4


# ---------------------------------------------------------------------------
# Raw source retention
# ---------------------------------------------------------------------------
#
# False = delete global source after successful processing.
#
# This is what we want for the 13-year run.
#
KEEP_RAW_AFTER_PROCESSING = False


# =============================================================================
# DIRECTORY SETUP
# =============================================================================

for directory in (
    RAW_ROOT,
    TEMP_ROOT,
    PROCESSED_ROOT,
    STATE_ROOT,
    LOG_ROOT,
):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE = LOG_ROOT / "ccmp_full_acquisition.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("ccmp_full")


# =============================================================================
# THREAD-SAFE STATE
# =============================================================================

STATE_LOCK = Lock()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def human_size(size_bytes: int | float) -> str:
    """Convert bytes to a readable binary size."""

    value = float(size_bytes)

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{size_bytes} B"


def sha256_file(path: Path) -> str:
    """Calculate SHA256 without loading the whole file into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(DOWNLOAD_CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def date_range(start: date, end: date):
    """Yield every date inclusively."""

    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


# =============================================================================
# URL / PATH CONSTRUCTION
# =============================================================================

def build_ccmp_url(target_date: date) -> str:
    """
    Build the deterministic official RSS CCMP V3.1 URL.

    Example:

    https://data.remss.com/ccmp/v03.1/Y2012/M03/
        CCMP_Wind_Analysis_20120301_V03.1_L4.nc
    """

    filename = (
        f"CCMP_Wind_Analysis_"
        f"{target_date.strftime('%Y%m%d')}"
        f"_V03.1_L4.nc"
    )

    return (
        f"{BASE_URL}/"
        f"Y{target_date.year}/"
        f"M{target_date.month:02d}/"
        f"{filename}"
    )


def build_temp_path(target_date: date) -> Path:
    """
    Temporary global source file.

    It is deliberately stored outside the final processed directory.
    """

    filename = (
        f"CCMP_Wind_Analysis_"
        f"{target_date.strftime('%Y%m%d')}"
        f"_V03.1_L4.nc"
    )

    directory = (
        TEMP_ROOT
        / f"{target_date.year:04d}"
        / f"{target_date.month:02d}"
    )

    directory.mkdir(parents=True, exist_ok=True)

    return directory / filename


def build_processed_path(target_date: date) -> Path:
    """
    Final small OceanEmbed daily file.
    """

    filename = (
        f"CCMP_UV_daily_"
        f"{target_date.strftime('%Y%m%d')}"
        f"_NIO_0p25deg.nc"
    )

    directory = (
        PROCESSED_ROOT
        / f"{target_date.year:04d}"
        / f"{target_date.month:02d}"
    )

    directory.mkdir(parents=True, exist_ok=True)

    return directory / filename


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

STATE_FILE = STATE_ROOT / "completed_dates.json"


def load_state() -> dict:
    """Load completed-date state."""

    if not STATE_FILE.exists():
        return {
            "completed": {},
        }

    try:
        with STATE_FILE.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if "completed" not in state:
            state["completed"] = {}

        return state

    except Exception as exc:
        logger.warning(
            "Could not read state file: %s",
            exc,
        )

        return {
            "completed": {},
        }


STATE = load_state()


def save_state() -> None:
    """Atomically save acquisition state."""

    temporary = STATE_FILE.with_suffix(".tmp")

    with STATE_LOCK:

        with temporary.open("w", encoding="utf-8") as file:
            json.dump(
                STATE,
                file,
                indent=2,
                sort_keys=True,
            )

        temporary.replace(STATE_FILE)


def mark_complete(
    target_date: date,
    processed_path: Path,
) -> None:
    """Mark a processed date as complete."""

    key = target_date.isoformat()

    with STATE_LOCK:

        STATE["completed"][key] = {
            "processed_file": str(processed_path),
            "completed_at_utc": datetime.now(datetime.UTC).isoformat(),
        }

    save_state()


def is_complete(target_date: date) -> bool:
    """
    Resume-safe completion check.

    State alone is not enough.
    The processed file must also exist and be non-empty.
    """

    key = target_date.isoformat()

    if key not in STATE.get("completed", {}):
        return False

    processed_path = build_processed_path(target_date)

    return (
        processed_path.exists()
        and processed_path.stat().st_size > 0
    )


# =============================================================================
# DOWNLOAD
# =============================================================================

def download_source(
    target_date: date,
    url: str,
    destination: Path,
) -> bool:
    """
    Download one CCMP daily source file.

    Uses:
        .part file
        streaming
        content-length verification
        retries
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    # Remove stale partial file.
    if temporary.exists():
        try:
            temporary.unlink()
        except Exception:
            pass

    request = Request(
        url,
        headers={
            "User-Agent": "OceanEmbed-CCMP-Acquisition/1.0",
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                "[%s] Download attempt %d/%d",
                target_date,
                attempt,
                MAX_RETRIES,
            )

            start_time = time.perf_counter()
            downloaded = 0

            with urlopen(
                request,
                timeout=READ_TIMEOUT_SECONDS,
            ) as response:

                expected_size_header = response.headers.get(
                    "Content-Length"
                )

                expected_size = (
                    int(expected_size_header)
                    if expected_size_header
                    else None
                )

                content_type = response.headers.get(
                    "Content-Type",
                    "",
                )

                logger.info(
                    "[%s] HTTP %s | remote=%s | type=%s",
                    target_date,
                    response.status,
                    human_size(expected_size)
                    if expected_size
                    else "unknown",
                    content_type,
                )

                with temporary.open("wb") as output:

                    while True:

                        chunk = response.read(
                            DOWNLOAD_CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        output.write(chunk)
                        downloaded += len(chunk)

            elapsed = time.perf_counter() - start_time

            if downloaded == 0:
                raise RuntimeError(
                    "Downloaded zero bytes."
                )

            if (
                expected_size is not None
                and downloaded != expected_size
            ):
                raise RuntimeError(
                    f"Size mismatch: "
                    f"expected {expected_size}, "
                    f"received {downloaded}"
                )

            temporary.replace(destination)

            speed = (
                downloaded
                / 1024
                / 1024
                / elapsed
                if elapsed > 0
                else 0
            )

            logger.info(
                "[%s] Downloaded %s in %.1fs (%.2f MB/s)",
                target_date,
                human_size(downloaded),
                elapsed,
                speed,
            )

            return True

        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as exc:

            logger.warning(
                "[%s] Download attempt failed: %s",
                target_date,
                exc,
            )

            if temporary.exists():

                try:
                    temporary.unlink()
                except Exception:
                    pass

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY_SECONDS * attempt
                )

            else:

                logger.error(
                    "[%s] Download FAILED after %d attempts.",
                    target_date,
                    MAX_RETRIES,
                )

                return False

    return False


# =============================================================================
# SOURCE VALIDATION
# =============================================================================

def validate_source_structure(
    ds: xr.Dataset,
    target_date: date,
) -> None:
    """
    Strict validation of the source CCMP file.

    Raises RuntimeError if the source is not what we expect.
    """

    required_coordinates = {
        "time",
        "latitude",
        "longitude",
    }

    missing_coordinates = (
        required_coordinates - set(ds.coords)
    )

    if missing_coordinates:
        raise RuntimeError(
            f"Missing coordinates: {sorted(missing_coordinates)}"
        )

    required_variables = {
        "uwnd",
        "vwnd",
        "nobs",
    }

    missing_variables = (
        required_variables - set(ds.data_vars)
    )

    if missing_variables:
        raise RuntimeError(
            f"Missing variables: {sorted(missing_variables)}"
        )

    # -------------------------------------------------------------------------
    # Expected temporal structure
    # -------------------------------------------------------------------------

    if ds.sizes.get("time") != 4:

        raise RuntimeError(
            "Expected exactly 4 CCMP analyses per daily file; "
            f"found {ds.sizes.get('time')}"
        )

    time_values = ds["time"].values

    expected_hours = {0, 6, 12, 18}

    actual_hours = {
        int(
            np.datetime64(value, "h")
            .astype("datetime64[h]")
            .astype(int)
            % 24
        )
        for value in time_values
    }

    if actual_hours != expected_hours:

        raise RuntimeError(
            f"Unexpected CCMP time hours: {sorted(actual_hours)}"
        )

    # -------------------------------------------------------------------------
    # Expected spatial dimensions
    # -------------------------------------------------------------------------

    if ds.sizes.get("latitude") != 720:

        raise RuntimeError(
            "Unexpected latitude dimension: "
            f"{ds.sizes.get('latitude')}"
        )

    if ds.sizes.get("longitude") != 1440:

        raise RuntimeError(
            "Unexpected longitude dimension: "
            f"{ds.sizes.get('longitude')}"
        )

    # -------------------------------------------------------------------------
    # Expected variable dimensions
    # -------------------------------------------------------------------------

    for variable in ("uwnd", "vwnd", "nobs"):

        expected_dims = (
            "time",
            "latitude",
            "longitude",
        )

        if ds[variable].dims != expected_dims:

            raise RuntimeError(
                f"{variable} dimensions are "
                f"{ds[variable].dims}; "
                f"expected {expected_dims}"
            )

    # -------------------------------------------------------------------------
    # Date check
    # -------------------------------------------------------------------------

    first_time = np.datetime64(
        time_values[0],
        "D",
    )

    expected_time = np.datetime64(
        target_date.isoformat(),
        "D",
    )

    if first_time != expected_time:

        raise RuntimeError(
            f"File date mismatch: "
            f"expected {expected_time}, "
            f"found {first_time}"
        )

    # -------------------------------------------------------------------------
    # Basic wind sanity
    # -------------------------------------------------------------------------

    for variable in ("uwnd", "vwnd"):

        values = ds[variable]

        finite_fraction = (
            np.isfinite(values.values).mean()
        )

        if finite_fraction <= 0:

            raise RuntimeError(
                f"{variable} contains no finite values."
            )

        # Extremely broad sanity bounds.
        # These are not scientific clipping limits.
        variable_min = float(
            values.min(skipna=True)
        )

        variable_max = float(
            values.max(skipna=True)
        )

        if variable_min < -100 or variable_max > 100:

            raise RuntimeError(
                f"{variable} has suspicious values: "
                f"{variable_min} to {variable_max} m/s"
            )


# =============================================================================
# PROCESS ONE SOURCE FILE
# =============================================================================

def process_ccmp_file(
    source_path: Path,
    processed_path: Path,
    target_date: date,
) -> bool:
    """
    Convert one global CCMP daily file into one small OceanEmbed daily file.

    Processing:
        global 0.25°
             ↓
        NIO subset
             ↓
        OceanEmbed grid
             ↓
        4 x 6-hourly → daily vector mean
             ↓
        U/V only
    """

    logger.info(
        "[%s] Processing source: %s",
        target_date,
        source_path.name,
    )

    try:

        # ---------------------------------------------------------------------
        # Open source
        # ---------------------------------------------------------------------

        with xr.open_dataset(
            source_path,
            engine="netcdf4",
            decode_times=True,
            mask_and_scale=True,
        ) as ds:

            validate_source_structure(
                ds,
                target_date,
            )

            # -------------------------------------------------------------
            # Build OceanEmbed target grid
            # -------------------------------------------------------------

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

            # -------------------------------------------------------------
            # NIO extraction
            #
            # The target grid is at:
            #
            # 45.00, 45.25, ...
            #
            # while CCMP native cell centers are:
            #
            # 44.875, 45.125, ...
            #
            # Therefore interpolation needs the neighboring native cells.
            #
            # xarray interp handles this cleanly.
            # -------------------------------------------------------------

            native_lat_min = (
                TARGET_LAT_MIN
                - TARGET_RESOLUTION
            )

            native_lat_max = (
                TARGET_LAT_MAX
                + TARGET_RESOLUTION
            )

            native_lon_min = (
                TARGET_LON_MIN
                - TARGET_RESOLUTION
            )

            native_lon_max = (
                TARGET_LON_MAX
                + TARGET_RESOLUTION
            )

            regional = ds[
                [
                    "uwnd",
                    "vwnd",
                    "nobs",
                ]
            ].sel(
                latitude=slice(
                    native_lat_min,
                    native_lat_max,
                ),
                longitude=slice(
                    native_lon_min,
                    native_lon_max,
                ),
            )

            # -------------------------------------------------------------
            # Regrid to OceanEmbed coordinates
            # -------------------------------------------------------------

            regional = regional.interp(
                latitude=xr.DataArray(
                    target_lats,
                    dims="latitude",
                ),
                longitude=xr.DataArray(
                    target_lons,
                    dims="longitude",
                ),
                method="linear",
            )

            # -------------------------------------------------------------
            # Daily vector mean
            #
            # Important:
            #
            # We average U and V separately.
            #
            # We intentionally DO NOT average ws.
            #
            # This preserves the vector nature of the wind.
            # -------------------------------------------------------------

            daily_uwnd = (
                regional["uwnd"]
                .mean(
                    dim="time",
                    skipna=True,
                )
                .astype("float32")
            )

            daily_vwnd = (
                regional["vwnd"]
                .mean(
                    dim="time",
                    skipna=True,
                )
                .astype("float32")
            )

            # -------------------------------------------------------------
            # Create final dataset
            # -------------------------------------------------------------

            output = xr.Dataset(
                data_vars={
                    "uwnd": (
                        ("time", "latitude", "longitude"),
                        daily_uwnd.values[
                            np.newaxis,
                            :,
                            :,
                        ],
                    ),
                    "vwnd": (
                        ("time", "latitude", "longitude"),
                        daily_vwnd.values[
                            np.newaxis,
                            :,
                            :,
                        ],
                    ),
                },
                coords={
                    "time": [
                        np.datetime64(
                            target_date.isoformat()
                        )
                    ],
                    "latitude": target_lats,
                    "longitude": target_lons,
                },
                attrs={
                    "title": (
                        "OceanEmbed processed CCMP V3.1 "
                        "daily surface winds"
                    ),
                    "source": (
                        "RSS CCMP 6-Hourly 10 Meter "
                        "Surface Winds L4 V3.1"
                    ),
                    "source_doi": (
                        "10.5067/CCMP-6HW10M-L4V31"
                    ),
                    "source_url": BASE_URL,
                    "processing": (
                        "NIO extraction, linear spatial "
                        "interpolation to OceanEmbed 0.25 degree "
                        "grid, daily vector mean of 00/06/12/18 UTC"
                    ),
                    "region": (
                        "North Indian Ocean: "
                        "45E-105E, 5N-30N"
                    ),
                    "spatial_resolution": "0.25 degree",
                    "temporal_resolution": "daily",
                    "variables": "uwnd, vwnd",
                    "wind_speed_note": (
                        "Wind speed can be derived as "
                        "sqrt(uwnd^2 + vwnd^2)."
                    ),
                },
            )

            # -------------------------------------------------------------
            # Variable metadata
            # -------------------------------------------------------------

            output["uwnd"].attrs = {
                "long_name": (
                    "Daily mean eastward wind component "
                    "at 10 meters"
                ),
                "standard_name": "eastward_wind",
                "units": "m s-1",
            }

            output["vwnd"].attrs = {
                "long_name": (
                    "Daily mean northward wind component "
                    "at 10 meters"
                ),
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

            output["time"].attrs = {
                "standard_name": "time",
            }

            # -------------------------------------------------------------
            # Compression
            # -------------------------------------------------------------

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
                "latitude": {
                    "dtype": "float32",
                },
                "longitude": {
                    "dtype": "float32",
                },
            }

            processed_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_output = processed_path.with_suffix(
                processed_path.suffix + ".part"
            )

            if temporary_output.exists():

                try:
                    temporary_output.unlink()
                except Exception:
                    pass

            # -------------------------------------------------------------
            # Write processed file
            # -------------------------------------------------------------

            output.to_netcdf(
                temporary_output,
                engine="netcdf4",
                format="NETCDF4",
                encoding=encoding,
            )

            output.close()

            # Atomic completion.
            temporary_output.replace(
                processed_path
            )

        # ---------------------------------------------------------------------
        # Final output audit
        # ---------------------------------------------------------------------

        if not processed_path.exists():

            raise RuntimeError(
                "Processed file was not created."
            )

        output_size = processed_path.stat().st_size

        if output_size <= 0:

            raise RuntimeError(
                "Processed file is empty."
            )

        # ---------------------------------------------------------------------
        # Reopen final file and verify
        # ---------------------------------------------------------------------

        with xr.open_dataset(
            processed_path,
            engine="netcdf4",
        ) as final_ds:

            expected_shape = (
                1,
                len(
                    np.arange(
                        TARGET_LAT_MIN,
                        TARGET_LAT_MAX
                        + TARGET_RESOLUTION / 2,
                        TARGET_RESOLUTION,
                    )
                ),
                len(
                    np.arange(
                        TARGET_LON_MIN,
                        TARGET_LON_MAX
                        + TARGET_RESOLUTION / 2,
                        TARGET_RESOLUTION,
                    )
                ),
            )

            actual_shape = final_ds["uwnd"].shape

            if actual_shape != expected_shape:

                raise RuntimeError(
                    f"Unexpected processed shape: "
                    f"{actual_shape}; "
                    f"expected {expected_shape}"
                )

            if final_ds["vwnd"].shape != expected_shape:

                raise RuntimeError(
                    "U/V shapes do not match."
                )

            if set(final_ds.data_vars) != {
                "uwnd",
                "vwnd",
            }:

                raise RuntimeError(
                    f"Unexpected final variables: "
                    f"{set(final_ds.data_vars)}"
                )

            # Basic finite-value check.
            for variable in ("uwnd", "vwnd"):

                finite_fraction = (
                    np.isfinite(
                        final_ds[variable].values
                    ).mean()
                )

                if finite_fraction <= 0:

                    raise RuntimeError(
                        f"{variable} has no finite values."
                    )

            logger.info(
                "[%s] Processed PASS | output=%s | size=%s",
                target_date,
                processed_path.name,
                human_size(output_size),
            )

        return True

    except Exception as exc:

        logger.error(
            "[%s] PROCESSING FAILED: %s",
            target_date,
            exc,
            exc_info=True,
        )

        # Do not leave a corrupt processed file.
        for path in (
            processed_path,
            processed_path.with_suffix(
                processed_path.suffix + ".part"
            ),
        ):

            if path.exists():

                try:
                    path.unlink()
                except Exception:
                    pass

        return False


# =============================================================================
# PROCESS ONE DAY
# =============================================================================

def process_one_day(
    target_date: date,
) -> tuple[date, bool, str]:
    """
    Complete transaction for one day:

        download
        process
        cleanup
        state
    """

    processed_path = build_processed_path(
        target_date
    )

    # -------------------------------------------------------------------------
    # Resume check
    # -------------------------------------------------------------------------

    if is_complete(target_date):

        return (
            target_date,
            True,
            "already complete",
        )

    source_path = build_temp_path(
        target_date
    )

    url = build_ccmp_url(
        target_date
    )

    start_time = time.perf_counter()

    logger.info(
        ""
    )

    logger.info(
        "=" * 80
    )

    logger.info(
        "[%s] START",
        target_date,
    )

    logger.info(
        "[%s] URL: %s",
        target_date,
        url,
    )

    # -------------------------------------------------------------------------
    # If an old source exists, use it.
    # -------------------------------------------------------------------------

    if not source_path.exists():

        success = download_source(
            target_date,
            url,
            source_path,
        )

        if not success:

            return (
                target_date,
                False,
                "download failed",
            )

    else:

        logger.info(
            "[%s] Existing temporary source found: %s",
            target_date,
            human_size(
                source_path.stat().st_size
            ),
        )

    # -------------------------------------------------------------------------
    # Process
    # -------------------------------------------------------------------------

    success = process_ccmp_file(
        source_path,
        processed_path,
        target_date,
    )

    if not success:

        return (
            target_date,
            False,
            "processing failed",
        )

    # -------------------------------------------------------------------------
    # Raw cleanup
    # -------------------------------------------------------------------------

    if not KEEP_RAW_AFTER_PROCESSING:

        try:

            source_size = source_path.stat().st_size

            source_path.unlink()

            logger.info(
                "[%s] Deleted global raw source: %s",
                target_date,
                human_size(source_size),
            )

        except Exception as exc:

            logger.warning(
                "[%s] Could not delete raw source: %s",
                target_date,
                exc,
            )

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    mark_complete(
        target_date,
        processed_path,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "[%s] COMPLETE in %.1fs",
        target_date,
        elapsed,
    )

    return (
        target_date,
        True,
        "complete",
    )


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    all_dates: list[date],
    results: list[tuple[date, bool, str]],
) -> None:

    successful = [
        item for item in results
        if item[1]
    ]

    failed = [
        item for item in results
        if not item[1]
    ]

    logger.info("")
    logger.info("=" * 80)
    logger.info("CCMP ACQUISITION SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Requested dates : %d",
        len(all_dates),
    )

    logger.info(
        "Successful      : %d",
        len(successful),
    )

    logger.info(
        "Failed          : %d",
        len(failed),
    )

    if failed:

        logger.error("")
        logger.error(
            "FAILED DATES:"
        )

        for target_date, _, reason in failed:

            logger.error(
                "  %s -> %s",
                target_date,
                reason,
            )

    # -------------------------------------------------------------------------
    # Processed storage
    # -------------------------------------------------------------------------

    total_processed_bytes = 0
    processed_files = 0

    for target_date in all_dates:

        path = build_processed_path(
            target_date
        )

        if path.exists():

            processed_files += 1
            total_processed_bytes += (
                path.stat().st_size
            )

    logger.info("")
    logger.info(
        "Processed files : %d",
        processed_files,
    )

    logger.info(
        "Processed size  : %s",
        human_size(total_processed_bytes),
    )

    logger.info(
        "State file      : %s",
        STATE_FILE,
    )

    logger.info(
        "Log file        : %s",
        LOG_FILE,
    )

    logger.info("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    logger.info("")
    logger.info("=" * 80)
    logger.info("OCEANEMBED CCMP V3.1")
    logger.info("FULL ACQUISITION + IMMEDIATE PROCESSING")
    logger.info("=" * 80)

    logger.info("")
    logger.info(
        "Source: %s",
        BASE_URL,
    )

    logger.info(
        "Collection: CCMP_WINDS_10M6HR_L4_V3.1",
    )

    logger.info(
        "DOI: 10.5067/CCMP-6HW10M-L4V31",
    )

    logger.info("")
    logger.info(
        "OceanEmbed region:"
    )

    logger.info(
        "  Longitude: %.2f -> %.2f",
        TARGET_LON_MIN,
        TARGET_LON_MAX,
    )

    logger.info(
        "  Latitude : %.2f -> %.2f",
        TARGET_LAT_MIN,
        TARGET_LAT_MAX,
    )

    logger.info(
        "  Resolution: %.2f degree",
        TARGET_RESOLUTION,
    )

    logger.info("")
    logger.info(
        "Requested period:"
    )

    logger.info(
        "  %s -> %s",
        START_DATE,
        END_DATE,
    )

    logger.info(
        "Max workers: %d",
        MAX_WORKERS,
    )

    logger.info(
        "Keep raw source: %s",
        KEEP_RAW_AFTER_PROCESSING,
    )

    # -------------------------------------------------------------------------
    # Build date list
    # -------------------------------------------------------------------------

    if TEST_MODE:

        dates = [TEST_DATE]

        logger.info("")
        logger.warning(
            "TEST MODE ENABLED"
        )

        logger.warning(
            "Only %s will be processed.",
            TEST_DATE,
        )

    else:

        dates = list(
            date_range(
                START_DATE,
                END_DATE,
            )
        )

        logger.info("")
        logger.info(
            "FULL MODE ENABLED"
        )

        logger.info(
            "Total calendar days: %d",
            len(dates),
        )

    # -------------------------------------------------------------------------
    # Resume filtering
    # -------------------------------------------------------------------------

    pending_dates = [
        target_date
        for target_date in dates
        if not is_complete(target_date)
    ]

    already_complete = (
        len(dates)
        - len(pending_dates)
    )

    logger.info("")
    logger.info(
        "Already complete: %d",
        already_complete,
    )

    logger.info(
        "Pending: %d",
        len(pending_dates),
    )

    if not pending_dates:

        logger.info(
            "Nothing to do. All requested dates are complete."
        )

        print_summary(
            dates,
            [
                (
                    target_date,
                    True,
                    "already complete",
                )
                for target_date in dates
            ],
        )

        return 0

    # -------------------------------------------------------------------------
    # Parallel acquisition + processing
    # -------------------------------------------------------------------------
    #
    # Each worker performs:
    #
    #     download -> process -> delete raw
    #
    # So we never need to download the whole dataset first.
    #
    # -------------------------------------------------------------------------

    results = []

    logger.info("")
    logger.info(
        "Starting workers..."
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                process_one_day,
                target_date,
            ): target_date
            for target_date in pending_dates
        }

        for future in as_completed(
            future_map
        ):

            target_date = future_map[
                future
            ]

            try:

                result = future.result()

                results.append(
                    result
                )

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

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_summary(
        dates,
        results,
    )

    failures = [
        result
        for result in results
        if not result[1]
    ]

    if failures:

        logger.error(
            "CCMP run finished with failures."
        )

        return 1

    logger.info(
        "CCMP run completed successfully."
    )

    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )