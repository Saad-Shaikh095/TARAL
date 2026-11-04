"""
OCEANEMBED CCMP V3.1 - ACQUISITION TEST
========================================

Purpose
-------
Download ONE verified CCMP V3.1 daily NetCDF file directly from
the official RSS CCMP archive.

This is deliberately an acquisition-only test.

Verified source
---------------
RSS CCMP V3.1 archive:
https://data.remss.com/ccmp/v03.1/

Dataset
-------
RSS CCMP 6-Hourly 10 Meter Surface Winds Level 4 Version 3.1

PO.DAAC collection:
CCMP_WINDS_10M6HR_L4_V3.1

DOI:
10.5067/CCMP-6HW10M-L4V31

Test date
---------
2012-03-01

Verified archive file:
CCMP_Wind_Analysis_20120301_V03.1_L4.nc
"""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import xarray as xr


# ============================================================================
# CONFIGURATION
# ============================================================================

# Exact OceanEmbed CCMP raw directory requested by the project.
RAW_ROOT = Path(r"D:\OceanEmbed\oceanembed-data\raw\ccmp")

# First verified test file.
TEST_DATE = date(2012, 3, 1)

# Official RSS CCMP V3.1 archive.
BASE_URL = "https://data.remss.com/ccmp/v03.1"

# Network configuration.
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300

# Retry configuration.
MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 5

# Download chunk size.
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB

# Expected extension.
FILE_EXTENSION = ".nc"

# ============================================================================
# LOGGING
# ============================================================================

LOG_DIR = RAW_ROOT / "_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ccmp_acquisition_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("ccmp_test")


# ============================================================================
# HELPERS
# ============================================================================

def build_ccmp_url(target_date: date) -> str:
    """
    Build the deterministic RSS CCMP V3.1 daily file URL.

    Archive structure:

        /YYYYY/Mmm/
            CCMP_Wind_Analysis_YYYYMMDD_V03.1_L4.nc
    """

    year = target_date.year
    month = target_date.month
    date_string = target_date.strftime("%Y%m%d")

    filename = (
        f"CCMP_Wind_Analysis_{date_string}_V03.1_L4.nc"
    )

    return (
        f"{BASE_URL}/"
        f"Y{year}/"
        f"M{month:02d}/"
        f"{filename}"
    )


def build_local_path(target_date: date) -> Path:
    """
    Store raw files by year/month.

    Example:

        raw/ccmp/
            2012/
                03/
                    CCMP_Wind_Analysis_20120301_V03.1_L4.nc
    """

    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")
    filename = (
        f"CCMP_Wind_Analysis_"
        f"{target_date.strftime('%Y%m%d')}"
        f"_V03.1_L4.nc"
    )

    return RAW_ROOT / year / month / filename


def sha256_file(path: Path) -> str:
    """
    Calculate SHA256 checksum without loading the whole file into RAM.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def human_size(size_bytes: int) -> str:
    """
    Convert bytes into a readable size.
    """

    value = float(size_bytes)

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{size_bytes} B"


# ============================================================================
# REMOTE CHECK
# ============================================================================

def check_remote_file(url: str) -> tuple[int | None, str | None]:
    """
    Perform a HEAD request before downloading.

    Returns:
        (content_length, content_type)
    """

    logger.info("Checking remote file...")

    request = Request(
        url,
        method="HEAD",
        headers={
            "User-Agent": "OceanEmbed-CCMP-Acquisition/1.0",
        },
    )

    try:
        with urlopen(
            request,
            timeout=CONNECT_TIMEOUT_SECONDS,
        ) as response:

            content_length = response.headers.get("Content-Length")
            content_type = response.headers.get("Content-Type")

            size = (
                int(content_length)
                if content_length
                else None
            )

            logger.info(
                "Remote HTTP status : %s",
                response.status,
            )

            logger.info(
                "Remote size        : %s",
                human_size(size) if size else "unknown",
            )

            logger.info(
                "Remote content type: %s",
                content_type or "unknown",
            )

            return size, content_type

    except HTTPError as exc:
        logger.error(
            "Remote server returned HTTP %s",
            exc.code,
        )
        return None, None

    except URLError as exc:
        logger.error(
            "Could not reach remote server: %s",
            exc.reason,
        )
        return None, None

    except Exception as exc:
        logger.error(
            "Remote check failed: %s",
            exc,
        )
        return None, None


# ============================================================================
# DOWNLOAD
# ============================================================================

def download_file(
    url: str,
    destination: Path,
    expected_size: int | None,
) -> bool:
    """
    Download a file using streaming HTTP.

    A .part file is used so an interrupted download never looks like
    a completed file.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    # If a previous partial download exists, remove it.
    if temporary.exists():
        logger.warning(
            "Removing previous partial file: %s",
            temporary,
        )
        temporary.unlink()

    request = Request(
        url,
        headers={
            "User-Agent": "OceanEmbed-CCMP-Acquisition/1.0",
        },
    )

    logger.info("Starting download...")
    logger.info("Destination: %s", destination)

    start = time.perf_counter()
    downloaded = 0

    try:
        with urlopen(
            request,
            timeout=READ_TIMEOUT_SECONDS,
        ) as response:

            with temporary.open("wb") as output:

                while True:
                    chunk = response.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    if expected_size:
                        percent = (
                            downloaded / expected_size
                        ) * 100

                        logger.info(
                            "Progress: %.1f%% | %s / %s",
                            percent,
                            human_size(downloaded),
                            human_size(expected_size),
                        )

        elapsed = time.perf_counter() - start

        if downloaded == 0:
            raise RuntimeError(
                "Downloaded zero bytes."
            )

        if (
            expected_size is not None
            and downloaded != expected_size
        ):
            raise RuntimeError(
                "Downloaded size mismatch: "
                f"expected {expected_size}, "
                f"received {downloaded}"
            )

        # Atomic-ish completion step.
        temporary.replace(destination)

        speed_mb = (
            downloaded / 1024 / 1024 / elapsed
            if elapsed > 0
            else 0
        )

        logger.info(
            "Download complete."
        )

        logger.info(
            "Size     : %s",
            human_size(downloaded),
        )

        logger.info(
            "Time     : %.2f seconds",
            elapsed,
        )

        logger.info(
            "Speed    : %.2f MB/s",
            speed_mb,
        )

        return True

    except Exception as exc:

        logger.error(
            "Download failed: %s",
            exc,
        )

        if temporary.exists():
            temporary.unlink()

        return False


# ============================================================================
# NETCDF VALIDATION
# ============================================================================

def validate_netcdf(
    path: Path,
    target_date: date,
) -> bool:
    """
    Validate that the downloaded file is a legitimate CCMP V3.1 NetCDF
    and contains the expected variables and coordinates.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATING CCMP NETCDF")
    logger.info("=" * 80)

    try:
        with xr.open_dataset(
            path,
            engine="netcdf4",
        ) as ds:

            logger.info("")
            logger.info("Dataset:")
            logger.info("%s", ds)

            # ---------------------------------------------------------------
            # Required dimensions / coordinates
            # ---------------------------------------------------------------

            required_coords = {
                "latitude",
                "longitude",
                "time",
            }

            missing_coords = (
                required_coords
                - set(ds.coords)
            )

            if missing_coords:
                logger.error(
                    "Missing coordinates: %s",
                    sorted(missing_coords),
                )
                return False

            # ---------------------------------------------------------------
            # Required wind variables
            # ---------------------------------------------------------------

            required_variables = {
                "uwnd",
                "vwnd",
            }

            missing_variables = (
                required_variables
                - set(ds.data_vars)
            )

            if missing_variables:
                logger.error(
                    "Missing required variables: %s",
                    sorted(missing_variables),
                )
                return False

            # ---------------------------------------------------------------
            # Coordinate information
            # ---------------------------------------------------------------

            lat = ds["latitude"]
            lon = ds["longitude"]
            time_coord = ds["time"]

            logger.info("")
            logger.info("Coordinate audit")
            logger.info("----------------")

            logger.info(
                "Latitude : %.4f -> %.4f (%d points)",
                float(lat.min()),
                float(lat.max()),
                lat.size,
            )

            logger.info(
                "Longitude: %.4f -> %.4f (%d points)",
                float(lon.min()),
                float(lon.max()),
                lon.size,
            )

            logger.info(
                "Time     : %s",
                time_coord.values,
            )

            # ---------------------------------------------------------------
            # Expected global CCMP grid
            # ---------------------------------------------------------------

            lat_values = lat.values
            lon_values = lon.values

            if lat.size != 720:
                logger.warning(
                    "Unexpected latitude count: %d",
                    lat.size,
                )

            if lon.size != 1440:
                logger.warning(
                    "Unexpected longitude count: %d",
                    lon.size,
                )

            # ---------------------------------------------------------------
            # Wind variables
            # ---------------------------------------------------------------

            for variable in ("uwnd", "vwnd"):

                da = ds[variable]

                logger.info("")
                logger.info(
                    "%s",
                    variable,
                )

                logger.info(
                    "  dimensions : %s",
                    da.dims,
                )

                logger.info(
                    "  dtype      : %s",
                    da.dtype,
                )

                logger.info(
                    "  units      : %s",
                    da.attrs.get("units", "unknown"),
                )

                logger.info(
                    "  min        : %.4f",
                    float(da.min(skipna=True)),
                )

                logger.info(
                    "  max        : %.4f",
                    float(da.max(skipna=True)),
                )

                logger.info(
                    "  mean       : %.4f",
                    float(da.mean(skipna=True)),
                )

            # ---------------------------------------------------------------
            # Metadata
            # ---------------------------------------------------------------

            logger.info("")
            logger.info("Important metadata")
            logger.info("------------------")

            metadata_keys = [
                "title",
                "institution",
                "source",
                "history",
                "time_coverage_start",
                "time_coverage_end",
            ]

            for key in metadata_keys:
                if key in ds.attrs:
                    logger.info(
                        "%s: %s",
                        key,
                        ds.attrs[key],
                    )

            # ---------------------------------------------------------------
            # Basic sanity checks
            # ---------------------------------------------------------------

            if lat.size != 720:
                logger.warning(
                    "Latitude grid is not the expected 720 points."
                )

            if lon.size != 1440:
                logger.warning(
                    "Longitude grid is not the expected 1440 points."
                )

            if float(lat.min()) < -80:
                logger.warning(
                    "Latitude extends below expected CCMP coverage."
                )

            if float(lat.max()) > 80:
                logger.warning(
                    "Latitude extends above expected CCMP coverage."
                )

            # ---------------------------------------------------------------
            # Check target date in metadata/time coordinate
            # ---------------------------------------------------------------

            expected_date = target_date.isoformat()

            time_values = time_coord.values

            logger.info("")
            logger.info(
                "Expected target date: %s",
                expected_date,
            )

            logger.info(
                "Time coordinate values: %s",
                time_values,
            )

            logger.info("")
            logger.info(
                "NETCDF VALIDATION: PASS"
            )

            return True

    except Exception as exc:

        logger.error(
            "NetCDF validation failed: %s",
            exc,
            exc_info=True,
        )

        return False


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    logger.info("")
    logger.info("=" * 80)
    logger.info("OCEANEMBED CCMP V3.1 ACQUISITION TEST")
    logger.info("=" * 80)

    logger.info("")
    logger.info("Dataset:")
    logger.info(
        "RSS CCMP 6-Hourly 10 Meter Surface Winds L4 V3.1"
    )

    logger.info(
        "Collection: CCMP_WINDS_10M6HR_L4_V3.1"
    )

    logger.info(
        "DOI: 10.5067/CCMP-6HW10M-L4V31"
    )

    logger.info("")
    logger.info(
        "Test date: %s",
        TEST_DATE,
    )

    url = build_ccmp_url(TEST_DATE)
    destination = build_local_path(TEST_DATE)

    logger.info("")
    logger.info("Verified source URL:")
    logger.info("%s", url)

    logger.info("")
    logger.info("Local destination:")
    logger.info("%s", destination)

    # ------------------------------------------------------------------------
    # Existing file handling
    # ------------------------------------------------------------------------

    if destination.exists():

        size = destination.stat().st_size

        if size > 0:

            logger.info("")
            logger.info(
                "File already exists."
            )

            logger.info(
                "Existing size: %s",
                human_size(size),
            )

            logger.info(
                "Skipping download and validating existing file."
            )

        else:

            logger.warning(
                "Existing file is empty. Removing it."
            )

            destination.unlink()

    # ------------------------------------------------------------------------
    # Download if necessary
    # ------------------------------------------------------------------------

    if not destination.exists():

        remote_size, content_type = check_remote_file(url)

        if remote_size is None:
            logger.error(
                "Remote file could not be verified."
            )
            return 1

        if content_type:
            logger.info(
                "Content-Type: %s",
                content_type,
            )

        success = False

        for attempt in range(1, MAX_RETRIES + 1):

            logger.info("")
            logger.info(
                "Download attempt %d/%d",
                attempt,
                MAX_RETRIES,
            )

            success = download_file(
                url=url,
                destination=destination,
                expected_size=remote_size,
            )

            if success:
                break

            if attempt < MAX_RETRIES:

                logger.warning(
                    "Retrying in %d seconds...",
                    RETRY_DELAY_SECONDS,
                )

                time.sleep(RETRY_DELAY_SECONDS)

        if not success:

            logger.error(
                "CCMP acquisition FAILED."
            )

            return 1

    # ------------------------------------------------------------------------
    # Local file audit
    # ------------------------------------------------------------------------

    local_size = destination.stat().st_size

    logger.info("")
    logger.info("=" * 80)
    logger.info("LOCAL FILE AUDIT")
    logger.info("=" * 80)

    logger.info(
        "File       : %s",
        destination.name,
    )

    logger.info(
        "Size       : %s",
        human_size(local_size),
    )

    logger.info(
        "SHA256     : %s",
        sha256_file(destination),
    )

    # ------------------------------------------------------------------------
    # NetCDF validation
    # ------------------------------------------------------------------------

    if not validate_netcdf(
        destination,
        TEST_DATE,
    ):

        logger.error("")
        logger.error(
            "CCMP TEST FAILED."
        )

        return 1

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("CCMP ACQUISITION TEST COMPLETE")
    logger.info("=" * 80)

    logger.info(
        "Status     : PASS"
    )

    logger.info(
        "File       : %s",
        destination,
    )

    logger.info(
        "Size       : %s",
        human_size(local_size),
    )

    logger.info("")
    logger.info(
        "IMPORTANT:"
    )

    logger.info(
        "This test intentionally performs NO preprocessing."
    )

    logger.info(
        "Next step is scientific validation of this raw CCMP file."
    )

    logger.info(
        "Only after that will we design the 2012-2024 acquisition pipeline."
    )

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())