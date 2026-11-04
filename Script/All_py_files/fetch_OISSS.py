# Final code:
# ==============================================================================
# OISSS v3.0 — MULTI-YEAR DOWNLOAD + IMMEDIATE SPATIAL SUBSETTING
# ==============================================================================
#
# CURRENT RANGE:
#   2012 → 2024
#
# SOURCE:
#   https://oisss.esr.org/data/V3/Daily/{YEAR}/
#
# PROJECT BBOX:
#   [45°E, 5°N, 105°E, 30°N]
#
# RETAINED VARIABLES:
#   - sss
#   - sss_empirical_uncertainty
#   - sss_formal_uncertainty
#
# PIPELINE:
#
#   Download .part
#       ↓
#   Verify download
#       ↓
#   Open global NetCDF
#       ↓
#   Spatial subset
#       ↓
#   Verify subset
#       ↓
#   Atomically replace with subset
#       ↓
#   Delete global temporary file
#
# IMPORTANT:
#   The source values are NOT clipped, modified, or scientifically altered.
#
# ==============================================================================

import os
import sys
import time
import gc
import hashlib
import traceback
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import xarray as xr


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --------------------------------------------------------------------------
# Years
# --------------------------------------------------------------------------
# CHANGED ONLY HERE:
#
# Original:
#   YEARS_TO_RUN = [2012]
#
# Now:
#   2012 → 2024
#
YEARS_TO_RUN = list(range(2012, 2025))

# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE_DIR = Path(r"E:\Ocean_data\OISSS")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


# --------------------------------------------------------------------------
# OISSS source
# --------------------------------------------------------------------------

BASE_URL = "https://oisss.esr.org/data/V3/Daily"


# --------------------------------------------------------------------------
# Bounding box
#
# [lon_min, lat_min, lon_max, lat_max]
# --------------------------------------------------------------------------

LON_MIN = 45.0
LAT_MIN = 5.0
LON_MAX = 105.0
LAT_MAX = 30.0


# --------------------------------------------------------------------------
# Variables to retain
# --------------------------------------------------------------------------

KEEP_VARIABLES = [
    "sss",
    "sss_empirical_uncertainty",
    "sss_formal_uncertainty",
]


# --------------------------------------------------------------------------
# Parallelism
#
# Based on your January tests:
#
#   4 workers  → 240.56 sec
#   8 workers  → 248.92 sec
#   15 workers → 203.29 sec
#
# 5 download workers gives a reasonable balance between throughput,
# server friendliness and local processing.
# --------------------------------------------------------------------------

DOWNLOAD_WORKERS = 5
SUBSET_WORKERS = 5


# --------------------------------------------------------------------------
# Network settings
# --------------------------------------------------------------------------

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 180

CHUNK_SIZE = 1024 * 1024       # 1 MB

MAX_RETRIES = 4

RETRY_BACKOFF = 3              # seconds


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

MIN_EXPECTED_FILE_SIZE = 1024 * 1024     # 1 MB

EXPECTED_GLOBAL_LAT_COUNT = 720
EXPECTED_GLOBAL_LON_COUNT = 1440

EXPECTED_SUBSET_LAT_COUNT = 100
EXPECTED_SUBSET_LON_COUNT = 240


# ==============================================================================
# SESSION
# ==============================================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "OISSS-Downloader/1.0 "
        "(research data acquisition; Python requests)"
    )
})


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_bytes(num_bytes):
    """Human-readable byte formatting."""

    if num_bytes < 1024:
        return f"{num_bytes:.0f} B"

    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"

    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"

    return f"{num_bytes / (1024 ** 3):.3f} GB"


def get_date_range(year):
    """Return all dates belonging to a year."""

    start = date(year, 1, 1)
    end = date(year, 12, 31)

    current = start

    dates = []

    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def build_filename(current_date):
    """Construct official OISSS filename."""

    return (
        f"OISSS_L4_multimission_global_daily_v3.0_"
        f"{current_date:%Y-%m-%d}.nc"
    )


def build_url(current_date):
    """Construct official OISSS daily URL."""

    year = current_date.year
    filename = build_filename(current_date)

    return (
        f"{BASE_URL}/{year}/{filename}"
    )


def atomic_replace(source, destination):
    """
    Replace destination with source atomically where supported.

    This prevents leaving a half-written final .nc file if something fails.
    """

    os.replace(str(source), str(destination))


# ==============================================================================
# DOWNLOAD
# ==============================================================================

def download_file(current_date, output_dir):
    """
    Download one OISSS daily file.

    The file is first written as .part.
    It is NEVER treated as a completed global file until the download
    has finished and basic validation succeeds.
    """

    filename = build_filename(current_date)

    final_path = output_dir / filename
    part_path = output_dir / f"{filename}.part"

    # --------------------------------------------------------------------------
    # Already-subsetted file
    # --------------------------------------------------------------------------

    if final_path.exists():

        try:

            size = final_path.stat().st_size

            if size >= MIN_EXPECTED_FILE_SIZE:

                return {
                    "status": "already_subset",
                    "date": current_date,
                    "filename": filename,
                    "size": size,
                    "path": final_path,
                }

        except OSError:
            pass

    # --------------------------------------------------------------------------
    # URL
    # --------------------------------------------------------------------------

    url = build_url(current_date)

    last_error = None

    # --------------------------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            # Remove stale partial file before a fresh attempt.
            if part_path.exists():

                try:
                    part_path.unlink()
                except OSError:
                    pass

            start_time = time.perf_counter()

            with session.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as response:

                if response.status_code == 404:

                    return {
                        "status": "failed",
                        "date": current_date,
                        "filename": filename,
                        "error": "HTTP 404",
                        "path": final_path,
                    }

                response.raise_for_status()

                content_length = response.headers.get(
                    "Content-Length"
                )

                expected_size = (
                    int(content_length)
                    if content_length
                    else None
                )

                bytes_written = 0

                with open(part_path, "wb") as f:

                    for chunk in response.iter_content(
                        chunk_size=CHUNK_SIZE
                    ):

                        if not chunk:
                            continue

                        f.write(chunk)
                        bytes_written += len(chunk)

                # ------------------------------------------------------------------
                # Basic file validation
                # ------------------------------------------------------------------

                if bytes_written < MIN_EXPECTED_FILE_SIZE:

                    raise IOError(
                        f"Downloaded file is suspiciously small: "
                        f"{bytes_written} bytes"
                    )

                if (
                    expected_size is not None
                    and bytes_written != expected_size
                ):

                    raise IOError(
                        f"Incomplete download: "
                        f"{bytes_written} / {expected_size} bytes"
                    )

                # ------------------------------------------------------------------
                # Rename only after successful complete download
                # ------------------------------------------------------------------

                atomic_replace(part_path, final_path)

            elapsed = time.perf_counter() - start_time

            speed = (
                bytes_written / (1024 ** 2) / elapsed
                if elapsed > 0
                else 0
            )

            return {
                "status": "downloaded",
                "date": current_date,
                "filename": filename,
                "size": bytes_written,
                "elapsed": elapsed,
                "speed": speed,
                "path": final_path,
            }

        except Exception as exc:

            last_error = str(exc)

            # Remove incomplete part.
            try:

                if part_path.exists():
                    part_path.unlink()

            except OSError:
                pass

            if attempt < MAX_RETRIES:

                sleep_time = RETRY_BACKOFF * attempt

                time.sleep(sleep_time)

    return {
        "status": "failed",
        "date": current_date,
        "filename": filename,
        "error": last_error,
        "path": final_path,
    }


# ==============================================================================
# SUBSETTING
# ==============================================================================

def subset_file(download_result):
    """
    Spatially subset one downloaded OISSS file.

    IMPORTANT:
        The input file is the downloaded GLOBAL file.

    A temporary subset is created first.

    The original global file is deleted ONLY after:
        1. subset creation succeeds
        2. subset opens successfully
        3. coordinates are correct
        4. required variables exist
        5. dimensions are correct

    The final filename remains identical to the official filename.
    """

    current_date = download_result["date"]
    global_path = download_result["path"]

    filename = download_result["filename"]

    subset_temp_path = global_path.with_suffix(".subset.nc")

    try:

        # ----------------------------------------------------------------------
        # If final file already exists, don't touch it.
        # ----------------------------------------------------------------------

        if not global_path.exists():

            return {
                "status": "failed",
                "date": current_date,
                "filename": filename,
                "error": "Input file does not exist",
            }

        # ----------------------------------------------------------------------
        # Open source global dataset
        # ----------------------------------------------------------------------

        with xr.open_dataset(
            global_path,
            decode_cf=True,
            mask_and_scale=True,
        ) as ds:

            # ------------------------------------------------------------------
            # Required coordinate validation
            # ------------------------------------------------------------------

            if "latitude" not in ds.coords:

                raise ValueError(
                    "latitude coordinate missing"
                )

            if "longitude" not in ds.coords:

                raise ValueError(
                    "longitude coordinate missing"
                )

            # ------------------------------------------------------------------
            # Required variable validation
            # ------------------------------------------------------------------

            missing_variables = [
                variable
                for variable in KEEP_VARIABLES
                if variable not in ds.data_vars
            ]

            if missing_variables:

                raise ValueError(
                    "Missing variables: "
                    + ", ".join(missing_variables)
                )

            # ------------------------------------------------------------------
            # Coordinate orientation
            # ------------------------------------------------------------------

            lat_values = ds["latitude"].values
            lon_values = ds["longitude"].values

            lat_ascending = (
                len(lat_values) < 2
                or lat_values[1] > lat_values[0]
            )

            lon_ascending = (
                len(lon_values) < 2
                or lon_values[1] > lon_values[0]
            )

            # ------------------------------------------------------------------
            # Slice according to coordinate orientation
            # ------------------------------------------------------------------

            if lat_ascending:

                lat_slice = slice(
                    LAT_MIN,
                    LAT_MAX
                )

            else:

                lat_slice = slice(
                    LAT_MAX,
                    LAT_MIN
                )

            if lon_ascending:

                lon_slice = slice(
                    LON_MIN,
                    LON_MAX
                )

            else:

                lon_slice = slice(
                    LON_MAX,
                    LON_MIN
                )

            # ------------------------------------------------------------------
            # Subset ONLY required variables
            # ------------------------------------------------------------------

            subset = ds[
                KEEP_VARIABLES
            ].sel(
                latitude=lat_slice,
                longitude=lon_slice,
            )

            # ------------------------------------------------------------------
            # Force data into memory
            # ------------------------------------------------------------------

            subset = subset.load()

            # ------------------------------------------------------------------
            # Validate dimensions
            # ------------------------------------------------------------------

            if "latitude" not in subset.dims:

                raise ValueError(
                    "latitude dimension missing after subsetting"
                )

            if "longitude" not in subset.dims:

                raise ValueError(
                    "longitude dimension missing after subsetting"
                )

            lat_count = subset.sizes["latitude"]
            lon_count = subset.sizes["longitude"]

            if lat_count != EXPECTED_SUBSET_LAT_COUNT:

                raise ValueError(
                    f"Unexpected latitude count: "
                    f"{lat_count}; expected "
                    f"{EXPECTED_SUBSET_LAT_COUNT}"
                )

            if lon_count != EXPECTED_SUBSET_LON_COUNT:

                raise ValueError(
                    f"Unexpected longitude count: "
                    f"{lon_count}; expected "
                    f"{EXPECTED_SUBSET_LON_COUNT}"
                )

            # ------------------------------------------------------------------
            # Validate actual coordinate bounds
            # ------------------------------------------------------------------

            subset_lats = subset["latitude"].values
            subset_lons = subset["longitude"].values

            if (
                subset_lats.min() < LAT_MIN
                or subset_lats.max() > LAT_MAX
            ):

                raise ValueError(
                    "Latitude coordinates exceed requested BBOX"
                )

            if (
                subset_lons.min() < LON_MIN
                or subset_lons.max() > LON_MAX
            ):

                raise ValueError(
                    "Longitude coordinates exceed requested BBOX"
                )

            # ------------------------------------------------------------------
            # Verify time coordinate
            # ------------------------------------------------------------------

            if "time" in subset.coords:

                if subset.sizes.get("time", 0) != 1:

                    raise ValueError(
                        "Unexpected time dimension size"
                    )

            # ------------------------------------------------------------------
            # Preserve global metadata
            # ------------------------------------------------------------------

            subset.attrs = dict(ds.attrs)

            # ------------------------------------------------------------------
            # Add processing metadata
            # ------------------------------------------------------------------

            subset.attrs["subset_bbox"] = (
                f"{LON_MIN},{LAT_MIN},{LON_MAX},{LAT_MAX}"
            )

            subset.attrs["subset_source_file"] = filename

            subset.attrs["subset_processing"] = (
                "Spatial subset of original OISSS v3.0 global daily dataset"
            )

            # ------------------------------------------------------------------
            # Write temporary subset
            # ------------------------------------------------------------------

            if subset_temp_path.exists():

                subset_temp_path.unlink()

            subset.to_netcdf(
                subset_temp_path,
                format="NETCDF4",
            )

        # ----------------------------------------------------------------------
        # VERIFY TEMPORARY SUBSET
        # ----------------------------------------------------------------------

        with xr.open_dataset(
            subset_temp_path,
            decode_cf=True,
            mask_and_scale=True,
        ) as verification_ds:

            # Required variables
            for variable in KEEP_VARIABLES:

                if variable not in verification_ds.data_vars:

                    raise ValueError(
                        f"Verification failed: "
                        f"{variable} missing"
                    )

            # Coordinates
            if "latitude" not in verification_ds.coords:

                raise ValueError(
                    "Verification failed: latitude missing"
                )

            if "longitude" not in verification_ds.coords:

                raise ValueError(
                    "Verification failed: longitude missing"
                )

            # Dimensions
            if (
                verification_ds.sizes["latitude"]
                != EXPECTED_SUBSET_LAT_COUNT
            ):

                raise ValueError(
                    "Verification failed: incorrect latitude count"
                )

            if (
                verification_ds.sizes["longitude"]
                != EXPECTED_SUBSET_LON_COUNT
            ):

                raise ValueError(
                    "Verification failed: incorrect longitude count"
                )

            # Coordinate bounds
            verify_lats = verification_ds["latitude"].values
            verify_lons = verification_ds["longitude"].values

            if verify_lats.min() < LAT_MIN:
                raise ValueError(
                    "Verification failed: latitude below BBOX"
                )

            if verify_lats.max() > LAT_MAX:
                raise ValueError(
                    "Verification failed: latitude above BBOX"
                )

            if verify_lons.min() < LON_MIN:
                raise ValueError(
                    "Verification failed: longitude below BBOX"
                )

            if verify_lons.max() > LON_MAX:
                raise ValueError(
                    "Verification failed: longitude above BBOX"
                )

            # Read all required data
            verification_ds[KEEP_VARIABLES].load()

        # ----------------------------------------------------------------------
        # FINAL REPLACEMENT
        # ----------------------------------------------------------------------

        original_size = global_path.stat().st_size
        subset_size = subset_temp_path.stat().st_size

        atomic_replace(
            subset_temp_path,
            global_path,
        )

        # ----------------------------------------------------------------------
        # Final existence check
        # ----------------------------------------------------------------------

        if not global_path.exists():

            raise RuntimeError(
                "Final subset file disappeared after replacement"
            )

        final_size = global_path.stat().st_size

        if final_size != subset_size:

            raise RuntimeError(
                "Final file size mismatch after replacement"
            )

        gc.collect()

        return {
            "status": "subset_complete",
            "date": current_date,
            "filename": filename,
            "original_size": original_size,
            "final_size": final_size,
            "saved": original_size - final_size,
            "path": global_path,
        }

    except Exception as exc:

        # ----------------------------------------------------------------------
        # If subsetting fails, DO NOT delete the global file.
        # ----------------------------------------------------------------------

        try:

            if subset_temp_path.exists():
                subset_temp_path.unlink()

        except OSError:
            pass

        return {
            "status": "failed",
            "date": current_date,
            "filename": filename,
            "error": str(exc),
            "path": global_path,
        }


# ==============================================================================
# MAIN YEAR PROCESSOR
# ==============================================================================

def process_year(year):
    """Download and subset every day in one year."""

    print("\n")
    print("=" * 78)
    print(f"OISSS v3.0 — YEAR {year}")
    print("=" * 78)

    dates = get_date_range(year)

    expected_files = len(dates)

    print(f"Year                  : {year}")
    print(
        f"Date range            : "
        f"{dates[0]} → {dates[-1]}"
    )

    print(f"Expected files        : {expected_files}")

    print(f"Download workers      : {DOWNLOAD_WORKERS}")
    print(f"Subset workers        : {SUBSET_WORKERS}")

    print()
    print("Bounding box:")
    print(f"  Longitude            : {LON_MIN} → {LON_MAX}")
    print(f"  Latitude             : {LAT_MIN} → {LAT_MAX}")

    print()
    print("Variables retained:")

    for variable in KEEP_VARIABLES:
        print(f"  - {variable}")

    # --------------------------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------------------------

    year_dir = RAW_DIR / str(year)

    for month in range(1, 13):

        month_dir = year_dir / f"{month:02d}"

        month_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    print()
    print(f"Output                : {year_dir}")

    # --------------------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------------------

    stats = {
        "downloaded": 0,
        "already_subset": 0,
        "download_failed": 0,
        "subset_complete": 0,
        "subset_failed": 0,
        "original_bytes": 0,
        "final_bytes": 0,
    }

    failures = []

    year_start_time = time.perf_counter()

    # ==========================================================================
    # PHASE 1 — DOWNLOAD
    # ==========================================================================

    print("\n")
    print("-" * 78)
    print("PHASE 1 — DOWNLOADING GLOBAL FILES")
    print("-" * 78)

    download_results = []

    with ThreadPoolExecutor(
        max_workers=DOWNLOAD_WORKERS
    ) as executor:

        future_map = {}

        for current_date in dates:

            month_dir = (
                year_dir / f"{current_date.month:02d}"
            )

            future = executor.submit(
                download_file,
                current_date,
                month_dir,
            )

            future_map[future] = current_date

        for future in as_completed(future_map):

            current_date = future_map[future]

            try:

                result = future.result()

            except Exception as exc:

                result = {
                    "status": "failed",
                    "date": current_date,
                    "filename": build_filename(current_date),
                    "error": str(exc),
                }

            download_results.append(result)

            status = result["status"]

            if status == "downloaded":

                stats["downloaded"] += 1

                print(
                    f"[DOWNLOADED]       "
                    f"{current_date} | "
                    f"{format_bytes(result['size'])} | "
                    f"{result['speed']:.2f} MB/s"
                )

            elif status == "already_subset":

                stats["already_subset"] += 1

                print(
                    f"[ALREADY SUBSET]   "
                    f"{current_date} | "
                    f"{format_bytes(result['size'])}"
                )

            else:

                stats["download_failed"] += 1

                failures.append(result)

                print(
                    f"[DOWNLOAD FAILED]  "
                    f"{current_date} | "
                    f"{result.get('error', 'unknown error')}"
                )

    # ==========================================================================
    # PHASE 2 — SUBSETTING
    # ==========================================================================

    print("\n")
    print("-" * 78)
    print("PHASE 2 — SPATIAL SUBSETTING")
    print("-" * 78)

    to_subset = [
        result
        for result in download_results
        if result["status"] == "downloaded"
    ]

    # Also retry any existing files that are not yet subsetted.

    for current_date in dates:

        month_dir = (
            year_dir / f"{current_date.month:02d}"
        )

        final_path = (
            month_dir /
            build_filename(current_date)
        )

        if not final_path.exists():
            continue

        try:

            with xr.open_dataset(
                final_path,
                decode_cf=True,
                mask_and_scale=False,
            ) as test_ds:

                lat_count = test_ds.sizes.get(
                    "latitude",
                    -1
                )

                lon_count = test_ds.sizes.get(
                    "longitude",
                    -1
                )

                is_subset = (
                    lat_count == EXPECTED_SUBSET_LAT_COUNT
                    and
                    lon_count == EXPECTED_SUBSET_LON_COUNT
                )

            if not is_subset:

                already_scheduled = any(
                    r["date"] == current_date
                    for r in to_subset
                )

                if not already_scheduled:

                    to_subset.append({
                        "status": "downloaded",
                        "date": current_date,
                        "filename": build_filename(current_date),
                        "path": final_path,
                        "size": final_path.stat().st_size,
                    })

        except Exception as exc:

            failures.append({
                "status": "failed",
                "date": current_date,
                "filename": build_filename(current_date),
                "error": (
                    f"Could not inspect existing file: {exc}"
                ),
            })

    subset_results = []

    if to_subset:

        with ThreadPoolExecutor(
            max_workers=SUBSET_WORKERS
        ) as executor:

            future_map = {}

            for result in to_subset:

                future = executor.submit(
                    subset_file,
                    result,
                )

                future_map[future] = result["date"]

            for future in as_completed(future_map):

                current_date = future_map[future]

                try:

                    result = future.result()

                except Exception as exc:

                    result = {
                        "status": "failed",
                        "date": current_date,
                        "filename": build_filename(current_date),
                        "error": str(exc),
                    }

                subset_results.append(result)

                if result["status"] == "subset_complete":

                    stats["subset_complete"] += 1

                    stats["original_bytes"] += (
                        result["original_size"]
                    )

                    stats["final_bytes"] += (
                        result["final_size"]
                    )

                    saved_percent = (
                        result["saved"]
                        / result["original_size"]
                        * 100
                        if result["original_size"] > 0
                        else 0
                    )

                    print(
                        f"[SUBSET COMPLETE] "
                        f"{current_date} | "
                        f"{format_bytes(result['original_size'])}"
                        f" → "
                        f"{format_bytes(result['final_size'])}"
                        f" | "
                        f"saved {saved_percent:.1f}%"
                    )

                else:

                    stats["subset_failed"] += 1

                    failures.append(result)

                    print(
                        f"[SUBSET FAILED]   "
                        f"{current_date} | "
                        f"{result.get('error', 'unknown error')}"
                    )

    # ==========================================================================
    # PHASE 3 — FINAL FILE AUDIT
    # ==========================================================================

    print("\n")
    print("-" * 78)
    print("PHASE 3 — FINAL YEAR FILE AUDIT")
    print("-" * 78)

    final_count = 0
    part_count = 0
    subset_temp_count = 0

    missing_dates = []

    for current_date in dates:

        month_dir = (
            year_dir / f"{current_date.month:02d}"
        )

        filename = build_filename(current_date)

        final_path = month_dir / filename

        part_path = month_dir / f"{filename}.part"

        subset_temp_path = (
            month_dir /
            f"{filename[:-3]}.subset.nc"
        )

        if final_path.exists():

            final_count += 1

        else:

            missing_dates.append(current_date)

        if part_path.exists():
            part_count += 1

        if subset_temp_path.exists():
            subset_temp_count += 1

    print(f"Expected daily files      : {expected_files}")
    print(f"Final .nc files            : {final_count}")
    print(f".part files remaining      : {part_count}")
    print(f".subset files remaining    : {subset_temp_count}")

    if missing_dates:

        print("\nMissing dates:")

        for missing_date in missing_dates:

            print(
                f"  - {missing_date}"
            )

    # ==========================================================================
    # YEAR SUMMARY
    # ==========================================================================

    elapsed = time.perf_counter() - year_start_time

    print("\n")
    print("=" * 78)
    print(f"OISSS v3.0 — YEAR {year} COMPLETE")
    print("=" * 78)

    print()
    print("FILES")
    print(f"  Expected                : {expected_files}")
    print(f"  Final                   : {final_count}")
    print(f"  Missing                 : {len(missing_dates)}")

    print()
    print("DOWNLOAD")
    print(f"  Newly downloaded        : {stats['downloaded']}")
    print(f"  Already subset          : {stats['already_subset']}")
    print(f"  Download failures       : {stats['download_failed']}")

    print()
    print("SUBSETTING")
    print(f"  Successfully subset     : {stats['subset_complete']}")
    print(f"  Subset failures         : {stats['subset_failed']}")

    print()
    print("STORAGE")

    if stats["original_bytes"] > 0:

        print(
            f"  Original processed      : "
            f"{format_bytes(stats['original_bytes'])}"
        )

        print(
            f"  Final subset data       : "
            f"{format_bytes(stats['final_bytes'])}"
        )

        saved = (
            stats["original_bytes"]
            - stats["final_bytes"]
        )

        reduction = (
            saved
            / stats["original_bytes"]
            * 100
        )

        print(
            f"  Space saved             : "
            f"{format_bytes(saved)}"
        )

        print(
            f"  Reduction               : "
            f"{reduction:.2f}%"
        )

    else:

        print(
            "  No newly processed storage statistics."
        )

    print()
    print("BBOX")
    print(
        f"  [{LON_MIN}, {LAT_MIN}, "
        f"{LON_MAX}, {LAT_MAX}]"
    )

    print()
    print("RETAINED VARIABLES")

    for variable in KEEP_VARIABLES:

        print(f"  - {variable}")

    print()
    print(
        f"Elapsed time             : "
        f"{elapsed:.2f} seconds"
    )

    print()
    print(
        f"Files stored in:"
    )

    print(
        f"  {year_dir}"
    )

    # ==========================================================================
    # FINAL STATUS
    # ==========================================================================

    successful = (
        final_count == expected_files
        and
        len(missing_dates) == 0
        and
        part_count == 0
        and
        subset_temp_count == 0
        and
        stats["download_failed"] == 0
        and
        stats["subset_failed"] == 0
    )

    if successful:

        print("\n")
        print("=" * 78)
        print("SUCCESS")
        print("=" * 78)

        print()
        print(
            f"All {expected_files} daily files for "
            f"{year} are present and processed."
        )

    else:

        print("\n")
        print("=" * 78)
        print("COMPLETED WITH ISSUES")
        print("=" * 78)

        print()

        if missing_dates:

            print(
                f"Missing files : {len(missing_dates)}"
            )

        if stats["download_failed"]:

            print(
                f"Download failures : "
                f"{stats['download_failed']}"
            )

        if stats["subset_failed"]:

            print(
                f"Subset failures : "
                f"{stats['subset_failed']}"
            )

        if part_count:

            print(
                f"Remaining .part files : "
                f"{part_count}"
            )

        if subset_temp_count:

            print(
                f"Remaining .subset files : "
                f"{subset_temp_count}"
            )

    return {
        "year": year,
        "expected": expected_files,
        "final": final_count,
        "missing": len(missing_dates),
        "downloaded": stats["downloaded"],
        "already_subset": stats["already_subset"],
        "download_failures": stats["download_failed"],
        "subset_complete": stats["subset_complete"],
        "subset_failures": stats["subset_failed"],
        "elapsed": elapsed,
        "success": successful,
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print()
    print("=" * 78)
    print("OISSS v3.0 — MULTI-YEAR DOWNLOAD PIPELINE")
    print("=" * 78)

    print()
    print("Configured years:")

    print(
        f"  {YEARS_TO_RUN[0]} → "
        f"{YEARS_TO_RUN[-1]}"
    )

    print()
    print("IMPORTANT:")

    print(
        "Only the years listed in YEARS_TO_RUN will actually run."
    )

    print()
    print("Current configuration:")

    print(
        f"  Download workers : "
        f"{DOWNLOAD_WORKERS}"
    )

    print(
        f"  Subset workers   : "
        f"{SUBSET_WORKERS}"
    )

    print(
        f"  BBOX             : "
        f"[{LON_MIN}, {LAT_MIN}, "
        f"{LON_MAX}, {LAT_MAX}]"
    )

    print()

    overall_start = time.perf_counter()

    year_summaries = []

    for year in YEARS_TO_RUN:

        try:

            summary = process_year(year)

            year_summaries.append(
                summary
            )

        except KeyboardInterrupt:

            print("\n")
            print("=" * 78)
            print("INTERRUPTED BY USER")
            print("=" * 78)

            print(
                "\nThe pipeline stopped safely."
            )

            print(
                "Existing completed files were not deleted."
            )

            break

        except Exception as exc:

            print("\n")
            print("=" * 78)
            print(f"YEAR {year} FAILED")
            print("=" * 78)

            print()
            print(str(exc))

            traceback.print_exc()

            year_summaries.append({
                "year": year,
                "success": False,
                "error": str(exc),
            })

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    # ==========================================================================
    # OVERALL SUMMARY
    # ==========================================================================

    print("\n")
    print("=" * 78)
    print("OISSS v3.0 — OVERALL SUMMARY")
    print("=" * 78)

    print()

    for summary in year_summaries:

        year = summary["year"]

        if summary.get("success"):

            status = "SUCCESS"

        else:

            status = "ISSUES"

        print(
            f"{year} : {status} | "
            f"{summary.get('final', 0)} / "
            f"{summary.get('expected', 0)} files"
        )

    print()

    print(
        f"Total elapsed time : "
        f"{overall_elapsed:.2f} seconds"
    )

    print()
    print(
        "Pipeline finished."
    )


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == "__main__":

    main()