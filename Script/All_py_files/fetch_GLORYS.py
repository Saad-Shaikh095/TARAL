from pathlib import Path
from calendar import monthrange
from datetime import datetime
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import xarray as xr
import copernicusmarine


# =============================================================================
# OCEANEMBED GLORYS ACQUISITION + REDUCTION
# =============================================================================
#
# Pipeline:
#
#   Copernicus GLORYS native subset
#          -> temporary NetCDF
#          -> native validation
#          -> 3x3 spatial reduction to 0.25°
#          -> vertical interpolation to 15 target depths
#          -> compressed final NetCDF
#          -> final validation + SHA256
#          -> delete native temporary file
#
# Credentials are NOT stored in this source file.
#
# Required Windows user environment variables:
#   COPERNICUSMARINE_SERVICE_USERNAME
#   COPERNICUSMARINE_SERVICE_PASSWORD
#
# The script passes those credentials explicitly to the Copernicus API.
# If they are missing, the script FAILS instead of opening an interactive
# username/password prompt during retries.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"D:\OceanEmbed\oceanembed-data")

RAW_DIR = BASE_DIR / "GLORYS"
TEMP_DIR = RAW_DIR / "_temporary"
FINAL_DIR = RAW_DIR / "processed"

LOG_DIR = BASE_DIR / "logs"
STATE_DIR = BASE_DIR / "manifests"

STATE_FILE = STATE_DIR / "glorys_acquisition_state.json"
LOG_FILE = LOG_DIR / "glorys_acquisition.log"

for directory in (
    RAW_DIR,
    TEMP_DIR,
    FINAL_DIR,
    LOG_DIR,
    STATE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COPERNICUS PRODUCT
# =============================================================================

PRODUCT_ID = "GLOBAL_MULTIYEAR_PHY_001_030"
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# The Copernicus response observed for this dataset is version 202311.
# Pinning the version makes the historical OceanEmbed acquisition reproducible.
DATASET_VERSION = "202311"

VARIABLES = ["thetao"]


# =============================================================================
# OCEANEMBED DATE RANGE
# =============================================================================

START_DATE = "2012-01-01"
END_DATE = "2024-12-31"


# =============================================================================
# OCEANEMBED SPATIAL DOMAIN
# =============================================================================

TARGET_MIN_LON = 45.0
TARGET_MAX_LON = 105.0

TARGET_MIN_LAT = 5.0
TARGET_MAX_LAT = 30.0

# Native GLORYS12V1 spacing.
NATIVE_STEP = 1.0 / 12.0

# One native point outside each boundary allows the centered 3x3 reduction
# to work at the final-domain edges.
REQUEST_MIN_LON = TARGET_MIN_LON - NATIVE_STEP
REQUEST_MAX_LON = TARGET_MAX_LON + NATIVE_STEP

REQUEST_MIN_LAT = TARGET_MIN_LAT - NATIVE_STEP
REQUEST_MAX_LAT = TARGET_MAX_LAT + NATIVE_STEP


# =============================================================================
# DEPTH
# =============================================================================

REQUEST_MIN_DEPTH = 0.0

# IMPORTANT:
# GLORYS first physical level is ~0.494 m.
# We request deeper than 1000 m so 1000 m can be interpolated between
# native levels instead of extrapolated.
REQUEST_MAX_DEPTH = 1100.0

TARGET_DEPTHS = np.array(
    [
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        500.0,
        700.0,
        1000.0,
    ],
    dtype=np.float32,
)


# =============================================================================
# TARGET HORIZONTAL GRID
# =============================================================================

TARGET_LONS = np.round(
    np.arange(
        TARGET_MIN_LON,
        TARGET_MAX_LON + 0.0001,
        0.25,
        dtype=np.float64,
    ),
    6,
)

TARGET_LATS = np.round(
    np.arange(
        TARGET_MIN_LAT,
        TARGET_MAX_LAT + 0.0001,
        0.25,
        dtype=np.float64,
    ),
    6,
)


# =============================================================================
# TEST / PRODUCTION MODE
# =============================================================================

# Keep TRUE until January 2012 is acquired and audited successfully.
# Then change to FALSE for the full 2012-2024 acquisition.
TEST_MODE = False

TEST_START_DATE = "2012-01-01"
TEST_END_DATE = "2012-01-31"


# =============================================================================
# RETRIES / VALIDATION
# =============================================================================

MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 15

# Copernicus/native coordinates can differ slightly from requested decimal
# values because the server returns values on the native GLORYS grid.
COORD_TOLERANCE = 0.01
DEPTH_TOLERANCE = 0.01

# The native GLORYS surface level is approximately 0.494 m.
MAX_ACCEPTABLE_SURFACE_LEVEL = 1.0


# =============================================================================
# NETCDF
# =============================================================================

NETCDF_COMPRESSION_LEVEL = 4


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def get_copernicus_credentials():
    """
    Read Copernicus credentials from environment variables.

    Deliberately fails instead of allowing an interactive prompt.
    """

    username = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
    password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

    missing = []

    if not username:
        missing.append("COPERNICUSMARINE_SERVICE_USERNAME")

    if not password:
        missing.append("COPERNICUSMARINE_SERVICE_PASSWORD")

    if missing:
        raise RuntimeError(
            "Copernicus credentials are not configured.\n\n"
            "Set these Windows user environment variables:\n"
            "  COPERNICUSMARINE_SERVICE_USERNAME\n"
            "  COPERNICUSMARINE_SERVICE_PASSWORD\n\n"
            "This script intentionally refuses to prompt for credentials."
        )

    return username, password


# =============================================================================
# STATE
# =============================================================================

def default_state():
    return {
        "product_id": PRODUCT_ID,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "completed": [],
        "failed": [],
        "records": {},
        "last_update": None,
    }


def load_state():
    if not STATE_FILE.exists():
        return default_state()

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

        state.setdefault("completed", [])
        state.setdefault("failed", [])
        state.setdefault("records", {})
        state.setdefault("last_update", None)

        return state

    except Exception as exc:
        logging.exception("Could not read state file.")
        print(f"WARNING: Could not read state file: {exc}")
        print("Starting with a fresh state.")
        return default_state()


def save_state(state):
    state["last_update"] = datetime.utcnow().isoformat() + "Z"

    temp_state = STATE_FILE.with_suffix(".tmp")

    with open(temp_state, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    temp_state.replace(STATE_FILE)


# =============================================================================
# DATE HELPERS
# =============================================================================

def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d")


def build_months(start_date=START_DATE, end_date=END_DATE):
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start > end:
        raise ValueError("START_DATE must not be after END_DATE.")

    months = []

    year = start.year
    month = start.month

    while True:
        first = datetime(year, month, 1)
        last = datetime(
            year,
            month,
            monthrange(year, month)[1],
        )

        chunk_start = max(first, start)
        chunk_end = min(last, end)

        months.append(
            (
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )

        if year == end.year and month == end.month:
            break

        month += 1

        if month == 13:
            month = 1
            year += 1

    return months


def expected_day_count(start_date, end_date):
    start = parse_date(start_date)
    end = parse_date(end_date)
    return (end - start).days + 1


# =============================================================================
# FILE NAMES
# =============================================================================

def final_filename(start_date, end_date):
    return (
        "GLORYS_thetao_"
        f"{start_date.replace('-', '')}_"
        f"{end_date.replace('-', '')}_"
        "NIO_0p25deg_15depth.nc"
    )


def temporary_filename(start_date, end_date):
    return (
        "GLORYS_native_"
        f"{start_date.replace('-', '')}_"
        f"{end_date.replace('-', '')}.nc"
    )


# =============================================================================
# FILE HELPERS
# =============================================================================

def validate_file_exists(path):
    return (
        path.exists()
        and path.is_file()
        and path.stat().st_size > 0
    )


def calculate_sha256(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def safe_unlink(path):
    if not path.exists():
        return

    try:
        path.unlink()
    except Exception as exc:
        logging.warning(
            "Could not delete %s: %s",
            path,
            exc,
        )


# =============================================================================
# COORDINATE NORMALIZATION
# =============================================================================

def normalize_coordinates(ds):
    rename = {}

    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"

    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"

    if rename:
        ds = ds.rename(rename)

    required = [
        "longitude",
        "latitude",
        "depth",
        "time",
    ]

    missing = [
        name
        for name in required
        if name not in ds.coords
    ]

    if missing:
        raise RuntimeError(
            "Missing required coordinates: "
            + ", ".join(missing)
        )

    return ds


# =============================================================================
# NATIVE VALIDATION
# =============================================================================

def validate_native_file(
    path,
    expected_start,
    expected_end,
):
    print("Validating native GLORYS file...")

    if not validate_file_exists(path):
        raise RuntimeError(
            f"Native file missing or empty:\n{path}"
        )

    try:
        with xr.open_dataset(
            path,
            decode_cf=True,
        ) as ds:
            ds = normalize_coordinates(ds)

            if "thetao" not in ds:
                raise RuntimeError(
                    "thetao is missing from native GLORYS file."
                )

            thetao = ds["thetao"]

            if thetao.size == 0:
                raise RuntimeError(
                    "thetao contains no data."
                )

            lon_min = float(ds.longitude.min().values)
            lon_max = float(ds.longitude.max().values)
            lat_min = float(ds.latitude.min().values)
            lat_max = float(ds.latitude.max().values)
            depth_min = float(ds.depth.min().values)
            depth_max = float(ds.depth.max().values)

            time_count = int(
                ds.sizes.get("time", 0)
            )

            print(f"  thetao shape : {thetao.shape}")
            print(
                f"  Longitude    : "
                f"{lon_min:.4f} -> {lon_max:.4f}"
            )
            print(
                f"  Latitude     : "
                f"{lat_min:.4f} -> {lat_max:.4f}"
            )
            print(
                f"  Depth        : "
                f"{depth_min:.4f} -> {depth_max:.4f}"
            )
            print(
                f"  Time points  : "
                f"{time_count}"
            )

            # -------------------------------------------------------------
            # Horizontal coverage.
            # -------------------------------------------------------------

            if (
                lon_min
                > REQUEST_MIN_LON + COORD_TOLERANCE
                or
                lon_max
                < REQUEST_MAX_LON - COORD_TOLERANCE
            ):
                raise RuntimeError(
                    "Native file does not cover "
                    "requested longitude."
                )

            if (
                lat_min
                > REQUEST_MIN_LAT + COORD_TOLERANCE
                or
                lat_max
                < REQUEST_MAX_LAT - COORD_TOLERANCE
            ):
                raise RuntimeError(
                    "Native file does not cover "
                    "requested latitude."
                )

            # -------------------------------------------------------------
            # Depth coverage.
            #
            # Literal 0 m is NOT required in the native file because the
            # first physical GLORYS level is approximately 0.494 m.
            # -------------------------------------------------------------

            if depth_min > MAX_ACCEPTABLE_SURFACE_LEVEL:
                raise RuntimeError(
                    "Native file starts too deep for "
                    f"the 0 m target: {depth_min:.4f} m"
                )

            if depth_max < 1000.0 - DEPTH_TOLERANCE:
                raise RuntimeError(
                    "Native GLORYS subset does not "
                    "reach the 1000 m target."
                )

            # Do not permit extrapolation at 1000 m.
            if depth_max <= 1000.0 + DEPTH_TOLERANCE:
                raise RuntimeError(
                    "Native GLORYS subset does not contain "
                    "a native level deeper than 1000 m. "
                    "Safe interpolation to 1000 m is "
                    "therefore impossible."
                )

            # -------------------------------------------------------------
            # Time coverage.
            # -------------------------------------------------------------

            expected_count = expected_day_count(
                expected_start,
                expected_end,
            )

            if time_count != expected_count:
                raise RuntimeError(
                    "Unexpected native time count: "
                    f"{time_count}, expected {expected_count}."
                )

            times = np.asarray(ds.time.values)

            first_time = np.datetime64(
                times[0],
                "D",
            )

            last_time = np.datetime64(
                times[-1],
                "D",
            )

            expected_first = np.datetime64(
                expected_start,
                "D",
            )

            expected_last = np.datetime64(
                expected_end,
                "D",
            )

            if (
                first_time != expected_first
                or
                last_time != expected_last
            ):
                raise RuntimeError(
                    "Native time coverage mismatch: "
                    f"{first_time} -> {last_time}, "
                    f"expected "
                    f"{expected_first} -> {expected_last}."
                )

            return {
                "shape": list(thetao.shape),
                "size_bytes": path.stat().st_size,
                "longitude": [
                    lon_min,
                    lon_max,
                ],
                "latitude": [
                    lat_min,
                    lat_max,
                ],
                "depth": [
                    depth_min,
                    depth_max,
                ],
                "time_count": time_count,
            }

    except RuntimeError:
        raise

    except Exception as exc:
        raise RuntimeError(
            "Native NetCDF validation failed: "
            f"{exc}"
        ) from exc


# =============================================================================
# SPATIAL REDUCTION
# =============================================================================

def reduce_spatial(thetao):
    print(
        "Reducing horizontal resolution "
        "to 0.25°..."
    )

    thetao = thetao.sortby("latitude")
    thetao = thetao.sortby("longitude")

    # Native = 1/12 degree.
    # Target = 1/4 degree = 3 * (1/12).
    #
    # Centered 3x3 averaging is used instead of simply discarding
    # two thirds of the native grid points.
    reduced = (
        thetao
        .rolling(
            longitude=3,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    reduced = (
        reduced
        .rolling(
            latitude=3,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    try:
        reduced = reduced.sel(
            longitude=TARGET_LONS,
            latitude=TARGET_LATS,
            method="nearest",
            tolerance=0.02,
        )

    except Exception as exc:
        raise RuntimeError(
            "Could not map the reduced native grid "
            "to the OceanEmbed 0.25° target grid."
        ) from exc

    reduced = reduced.assign_coords(
        longitude=TARGET_LONS,
        latitude=TARGET_LATS,
    )

    return reduced


# =============================================================================
# VERTICAL REDUCTION
# =============================================================================

def reduce_vertical(thetao):
    print(
        "Interpolating to OceanEmbed "
        "target depths..."
    )

    thetao = thetao.sortby("depth")

    depths = np.asarray(
        thetao.depth.values,
        dtype=np.float64,
    )

    if len(depths) < 2:
        raise RuntimeError(
            "Not enough native depth levels "
            "for interpolation."
        )

    max_target = float(
        TARGET_DEPTHS.max()
    )

    if depths[0] > MAX_ACCEPTABLE_SURFACE_LEVEL:
        raise RuntimeError(
            "Native data does not contain a "
            "sufficiently shallow level for "
            "the 0 m target."
        )

    if depths[-1] < (
        max_target - DEPTH_TOLERANCE
    ):
        raise RuntimeError(
            "Native GLORYS subset does not "
            "reach 1000 m."
        )

    if depths[-1] <= (
        max_target + DEPTH_TOLERANCE
    ):
        raise RuntimeError(
            "Native GLORYS data does not contain "
            "a depth level deeper than 1000 m. "
            "Refusing extrapolation."
        )

    # All positive target depths are inside the native range.
    positive_targets = (
        TARGET_DEPTHS[
            TARGET_DEPTHS > 0
        ]
        .astype(np.float64)
    )

    interpolated = thetao.interp(
        depth=positive_targets,
        method="linear",
    )

    # The first physical GLORYS level is ~0.494 m.
    # For 0 m, use the nearest physical level instead of extrapolation.
    surface = thetao.sel(
        depth=0.0,
        method="nearest",
    )

    surface = surface.expand_dims(
        depth=[0.0]
    )

    result = xr.concat(
        [
            surface,
            interpolated,
        ],
        dim="depth",
    )

    result = result.assign_coords(
        depth=TARGET_DEPTHS
    )

    return result


# =============================================================================
# FINAL VALIDATION
# =============================================================================

def validate_final_file(
    path,
    expected_start,
    expected_end,
):
    print(
        "Validating final OceanEmbed file..."
    )

    if not validate_file_exists(path):
        raise RuntimeError(
            f"Final file missing or empty:\n{path}"
        )

    with xr.open_dataset(
        path,
        decode_cf=True,
    ) as ds:
        ds = normalize_coordinates(ds)

        if "thetao" not in ds:
            raise RuntimeError(
                "Final file does not contain thetao."
            )

        thetao = ds["thetao"]

        expected_sizes = {
            "time": expected_day_count(
                expected_start,
                expected_end,
            ),
            "depth": len(TARGET_DEPTHS),
            "latitude": len(TARGET_LATS),
            "longitude": len(TARGET_LONS),
        }

        for dimension, expected_size in (
            expected_sizes.items()
        ):
            actual = int(
                ds.sizes.get(
                    dimension,
                    -1,
                )
            )

            if actual != expected_size:
                raise RuntimeError(
                    f"Dimension mismatch: "
                    f"{dimension}={actual}, "
                    f"expected {expected_size}."
                )

        if not np.allclose(
            ds.depth.values,
            TARGET_DEPTHS,
            atol=1e-4,
        ):
            raise RuntimeError(
                "Depth coordinates do not "
                "match OceanEmbed target."
            )

        if not np.allclose(
            ds.longitude.values,
            TARGET_LONS,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Longitude coordinates do not "
                "match OceanEmbed target."
            )

        if not np.allclose(
            ds.latitude.values,
            TARGET_LATS,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Latitude coordinates do not "
                "match OceanEmbed target."
            )

        times = np.asarray(
            ds.time.values
        )

        if len(times) == 0:
            raise RuntimeError(
                "Final file contains no time data."
            )

        first_day = np.datetime64(
            times[0],
            "D",
        )

        last_day = np.datetime64(
            times[-1],
            "D",
        )

        expected_first = np.datetime64(
            expected_start,
            "D",
        )

        expected_last = np.datetime64(
            expected_end,
            "D",
        )

        if (
            first_day != expected_first
            or
            last_day != expected_last
        ):
            raise RuntimeError(
                "Final time coverage mismatch: "
                f"{first_day} -> {last_day}, "
                f"expected "
                f"{expected_first} -> {expected_last}."
            )

        if thetao.dtype != np.float32:
            raise RuntimeError(
                f"thetao dtype is {thetao.dtype}, "
                "expected float32."
            )

        valid_count = int(
            thetao.count().values
        )

        total_count = int(
            thetao.size
        )

        valid_fraction = (
            valid_count / total_count
            if total_count
            else 0.0
        )

        if valid_fraction <= 0:
            raise RuntimeError(
                "Final thetao contains "
                "no valid values."
            )

        print(
            f"  Shape        : "
            f"{thetao.shape}"
        )

        print(
            f"  Time         : "
            f"{first_day} -> {last_day}"
        )

        print(
            f"  Depths       : "
            f"{len(ds.depth)}"
        )

        print(
            f"  Grid         : "
            f"{len(ds.latitude)} × "
            f"{len(ds.longitude)}"
        )

        print(
            f"  Valid thetao : "
            f"{valid_fraction * 100:.2f}%"
        )

        return {
            "shape": list(
                thetao.shape
            ),
            "valid_fraction": (
                valid_fraction
            ),
            "file_size_bytes": (
                path.stat().st_size
            ),
        }


# =============================================================================
# NATIVE -> FINAL
# =============================================================================

def process_native_file(
    native_path,
    final_path,
    expected_start,
    expected_end,
):
    print()
    print(
        "Reducing native GLORYS data..."
    )

    started = time.time()

    final_temp = (
        final_path.with_suffix(
            ".tmp.nc"
        )
    )

    safe_unlink(final_temp)

    try:
        with xr.open_dataset(
            native_path,
            decode_cf=True,
        ) as ds:
            ds = normalize_coordinates(ds)

            if "thetao" not in ds:
                raise RuntimeError(
                    "thetao missing from native file."
                )

            thetao = ds["thetao"]

            thetao = reduce_spatial(
                thetao
            )

            thetao = reduce_vertical(
                thetao
            )

            thetao = thetao.astype(
                np.float32
            )

            final_ds = thetao.to_dataset(
                name="thetao"
            )

            final_ds.attrs.update(
                {
                    "title": (
                        "OceanEmbed GLORYS "
                        "Daily Subsurface "
                        "Temperature"
                    ),
                    "source_product": PRODUCT_ID,
                    "source_dataset": DATASET_ID,
                    "source_dataset_version": (
                        DATASET_VERSION
                    ),
                    "source_variable": "thetao",
                    "processing": (
                        "Native GLORYS12V1 1/12 degree "
                        "subset reduced using centered "
                        "3x3 spatial mean to 0.25 degree "
                        "and vertically interpolated to "
                        "OceanEmbed target depths."
                    ),
                    "domain": (
                        "45E-105E, 5N-30N"
                    ),
                    "horizontal_resolution": (
                        "0.25 degree"
                    ),
                    "temporal_resolution": (
                        "Daily mean"
                    ),
                    "target_depths_m": (
                        ",".join(
                            str(int(depth))
                            for depth
                            in TARGET_DEPTHS
                        )
                    ),
                    "created_by": "OceanEmbed",
                }
            )

            encoding = {
                "thetao": {
                    "zlib": True,
                    "complevel": (
                        NETCDF_COMPRESSION_LEVEL
                    ),
                    "dtype": "float32",
                    "shuffle": True,
                }
            }

            print(
                "Writing compressed final NetCDF..."
            )

            final_ds.to_netcdf(
                final_temp,
                mode="w",
                format="NETCDF4",
                engine="netcdf4",
                encoding=encoding,
            )

            final_ds.close()

        validation = validate_final_file(
            final_temp,
            expected_start,
            expected_end,
        )

        final_temp.replace(
            final_path
        )

        elapsed = (
            time.time() - started
        )

        size_mb = (
            final_path.stat().st_size
            / 1024
            / 1024
        )

        print()
        print(
            "REDUCTION SUCCESS"
        )

        print(
            f"Final file : "
            f"{final_path.name}"
        )

        print(
            f"Final size : "
            f"{size_mb:,.2f} MB"
        )

        print(
            f"Shape      : "
            f"{validation['shape']}"
        )

        print(
            f"Valid data : "
            f"{validation['valid_fraction'] * 100:.2f}%"
        )

        print(
            f"Processing : "
            f"{elapsed / 60:.2f} minutes"
        )

        return validation

    except Exception:
        safe_unlink(final_temp)
        raise


# =============================================================================
# DOWNLOAD NATIVE
# =============================================================================

def download_native(
    start_date,
    end_date,
    native_path,
):
    username, password = (
        get_copernicus_credentials()
    )

    safe_unlink(native_path)

    print()
    print(
        "Requesting GLORYS from Copernicus..."
    )

    print(
        f"  Time       : "
        f"{start_date} -> {end_date}"
    )

    print(
        f"  Longitude  : "
        f"{REQUEST_MIN_LON:.6f} -> "
        f"{REQUEST_MAX_LON:.6f}"
    )

    print(
        f"  Latitude   : "
        f"{REQUEST_MIN_LAT:.6f} -> "
        f"{REQUEST_MAX_LAT:.6f}"
    )

    print(
        f"  Depth      : "
        f"{REQUEST_MIN_DEPTH} -> "
        f"{REQUEST_MAX_DEPTH} m"
    )

    print(
        "  Variable   : thetao"
    )

    print(
        "  Auth       : environment variables"
    )

    started = time.time()

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        username=username,
        password=password,
        variables=VARIABLES,
        minimum_longitude=REQUEST_MIN_LON,
        maximum_longitude=REQUEST_MAX_LON,
        minimum_latitude=REQUEST_MIN_LAT,
        maximum_latitude=REQUEST_MAX_LAT,
        minimum_depth=REQUEST_MIN_DEPTH,
        maximum_depth=REQUEST_MAX_DEPTH,
        start_datetime=(
            start_date + "T00:00:00"
        ),
        end_datetime=(
            end_date + "T23:59:59"
        ),
        coordinates_selection_method="inside",
        output_directory=str(
            TEMP_DIR
        ),
        output_filename=(
            native_path.name
        ),
        file_format="netcdf",
        netcdf_compression_level=4,
        overwrite=True,
    )

    elapsed = (
        time.time() - started
    )

    # Some Toolbox versions may normalize the filename.
    # If the exact requested filename is absent, recover the newest matching
    # .nc file from the temporary directory.
    if not validate_file_exists(
        native_path
    ):
        candidates = sorted(
            TEMP_DIR.glob(
                f"{native_path.stem}*.nc"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        candidates = [
            p
            for p in candidates
            if p != native_path
            and p.is_file()
        ]

        if candidates:
            candidates[0].replace(
                native_path
            )

    if not validate_file_exists(
        native_path
    ):
        raise RuntimeError(
            "Copernicus completed the request "
            "but the expected native output "
            "file was not found:\n"
            f"{native_path}"
        )

    size_mb = (
        native_path.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(
        "NATIVE DOWNLOAD COMPLETE"
    )

    print(
        f"Size    : "
        f"{size_mb:,.2f} MB"
    )

    print(
        f"Time    : "
        f"{elapsed / 60:.2f} minutes"
    )

    return native_path


# =============================================================================
# PROCESS ONE CHUNK
# =============================================================================

def process_chunk(
    start_date,
    end_date,
    state,
):
    chunk_id = (
        f"{start_date}_{end_date}"
    )

    final_path = (
        FINAL_DIR
        / final_filename(
            start_date,
            end_date,
        )
    )

    native_path = (
        TEMP_DIR
        / temporary_filename(
            start_date,
            end_date,
        )
    )

    # -------------------------------------------------------------------------
    # Already complete.
    # -------------------------------------------------------------------------

    if chunk_id in state["completed"]:
        if validate_file_exists(
            final_path
        ):
            try:
                validate_final_file(
                    final_path,
                    start_date,
                    end_date,
                )

                print(
                    f"[SKIP] {chunk_id} "
                    " - already complete and valid."
                )

                return True

            except Exception as exc:
                print(
                    "[WARNING] State says complete "
                    "but final file failed validation:"
                )

                print(
                    f"  {exc}"
                )

                state["completed"].remove(
                    chunk_id
                )

                state["records"].pop(
                    chunk_id,
                    None,
                )

                save_state(
                    state
                )

    # -------------------------------------------------------------------------
    # Existing final file not recorded in state.
    # -------------------------------------------------------------------------

    if validate_file_exists(
        final_path
    ):
        print(
            f"[CHECK] Existing final file: "
            f"{final_path.name}"
        )

        try:
            validation = validate_final_file(
                final_path,
                start_date,
                end_date,
            )

            if chunk_id not in state[
                "completed"
            ]:
                state[
                    "completed"
                ].append(
                    chunk_id
                )

            state["failed"] = [
                item
                for item in state["failed"]
                if item != chunk_id
            ]

            state["records"][chunk_id] = {
                "final_path": str(
                    final_path
                ),
                "final_size_bytes": (
                    final_path.stat().st_size
                ),
                "sha256": calculate_sha256(
                    final_path
                ),
                "validation": validation,
                "recovered": True,
            }

            save_state(
                state
            )

            print(
                "Existing final file is valid."
            )

            return True

        except Exception as exc:
            print(
                "Existing final file failed "
                f"validation: {exc}"
            )

            safe_unlink(
                final_path
            )

    # Start cleanly.
    safe_unlink(
        native_path
    )

    safe_unlink(
        final_path.with_suffix(
            ".tmp.nc"
        )
    )

    # Remove legacy files created by previous acquisition versions.
    legacy_names = [
        (
            "GLORYS_thetao_"
            f"{start_date.replace('-', '')}_"
            f"{end_date.replace('-', '')}.nc.part.nc"
        ),
        (
            "GLORYS_thetao_"
            f"{start_date.replace('-', '')}_"
            f"{end_date.replace('-', '')}.nc"
        ),
    ]

    for legacy_name in legacy_names:
        legacy = RAW_DIR / legacy_name

        if (
            legacy.exists()
            and legacy != final_path
        ):
            print(
                f"Removing legacy file: "
                f"{legacy.name}"
            )

            safe_unlink(
                legacy
            )

    # -------------------------------------------------------------------------
    # Retry complete chunk.
    # -------------------------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        print()
        print(
            "=" * 80
        )

        print(
            f"GLORYS CHUNK "
            f"{start_date} -> {end_date}"
        )

        print(
            f"Attempt "
            f"{attempt}/{MAX_RETRIES}"
        )

        print(
            "=" * 80
        )

        chunk_started = time.time()

        try:
            logging.info(
                "START chunk=%s attempt=%d",
                chunk_id,
                attempt,
            )

            # 1. Download native subset.
            download_native(
                start_date,
                end_date,
                native_path,
            )

            # 2. Validate native subset.
            native_info = (
                validate_native_file(
                    native_path,
                    start_date,
                    end_date,
                )
            )

            # 3. Reduce and validate final data.
            validation = (
                process_native_file(
                    native_path,
                    final_path,
                    start_date,
                    end_date,
                )
            )

            # 4. Checksum.
            print(
                "Calculating SHA256..."
            )

            checksum = (
                calculate_sha256(
                    final_path
                )
            )

            # 5. Record state BEFORE deleting the native file.
            state["completed"] = [
                item
                for item in state["completed"]
                if item != chunk_id
            ]

            state["completed"].append(
                chunk_id
            )

            state["failed"] = [
                item
                for item in state["failed"]
                if item != chunk_id
            ]

            state["records"][chunk_id] = {
                "start_date": start_date,
                "end_date": end_date,
                "native_size_bytes": (
                    native_info["size_bytes"]
                ),
                "final_size_bytes": (
                    final_path.stat().st_size
                ),
                "sha256": checksum,
                "validation": validation,
                "completed_at": (
                    datetime.utcnow()
                    .isoformat()
                    + "Z"
                ),
            }

            save_state(
                state
            )

            # 6. Delete native temporary file.
            print()
            print(
                "Deleting temporary "
                "native GLORYS file..."
            )

            safe_unlink(
                native_path
            )

            elapsed = (
                time.time()
                - chunk_started
            )

            final_size_mb = (
                final_path.stat().st_size
                / 1024
                / 1024
            )

            print()
            print(
                "=" * 80
            )

            print(
                "CHUNK COMPLETE"
            )

            print(
                "=" * 80
            )

            print(
                f"Period       : "
                f"{start_date} -> {end_date}"
            )

            print(
                f"Native size  : "
                f"{native_info['size_bytes'] / 1024 / 1024:,.2f} MB"
            )

            print(
                f"Final size   : "
                f"{final_size_mb:,.2f} MB"
            )

            print(
                f"Final shape  : "
                f"{validation['shape']}"
            )

            print(
                f"SHA256       : "
                f"{checksum}"
            )

            print(
                f"Elapsed      : "
                f"{elapsed / 60:.2f} minutes"
            )

            print(
                "=" * 80
            )

            logging.info(
                "SUCCESS chunk=%s native_bytes=%d "
                "final_bytes=%d sha256=%s",
                chunk_id,
                native_info[
                    "size_bytes"
                ],
                final_path.stat().st_size,
                checksum,
            )

            return True

        except Exception as exc:
            logging.exception(
                "FAILED chunk=%s attempt=%d",
                chunk_id,
                attempt,
            )

            print()
            print(
                "CHUNK FAILED"
            )

            print(
                f"Reason: {exc}"
            )

            # Never retain incomplete outputs.
            safe_unlink(
                final_path.with_suffix(
                    ".tmp.nc"
                )
            )

            safe_unlink(
                native_path
            )

            if attempt < MAX_RETRIES:
                delay = (
                    RETRY_DELAY_SECONDS
                    * attempt
                )

                print(
                    f"Retrying in "
                    f"{delay} seconds..."
                )

                time.sleep(
                    delay
                )

            else:
                state["failed"] = [
                    item
                    for item in state["failed"]
                    if item != chunk_id
                ]

                state["failed"].append(
                    chunk_id
                )

                state["records"][chunk_id] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "last_error": str(exc),
                    "failed_at": (
                        datetime.utcnow()
                        .isoformat()
                        + "Z"
                    ),
                }

                save_state(
                    state
                )

                print(
                    "Chunk permanently failed "
                    f"after {MAX_RETRIES} attempts."
                )

    return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Fail before any network request if credentials are absent.
    username, _ = (
        get_copernicus_credentials()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "OCEANEMBED GLORYS ACQUISITION V3"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"Product       : "
        f"{PRODUCT_ID}"
    )

    print(
        f"Dataset       : "
        f"{DATASET_ID}"
    )

    print(
        f"Dataset ver.  : "
        f"{DATASET_VERSION}"
    )

    print(
        "Variable      : thetao"
    )

    print(
        f"Copernicus user: "
        f"{username}"
    )

    print()
    print(
        "OceanEmbed domain:"
    )

    print(
        f"  Longitude   : "
        f"{TARGET_MIN_LON} -> "
        f"{TARGET_MAX_LON}"
    )

    print(
        f"  Latitude    : "
        f"{TARGET_MIN_LAT} -> "
        f"{TARGET_MAX_LAT}"
    )

    print()
    print(
        "Final grid:"
    )

    print(
        f"  Longitude   : "
        f"{len(TARGET_LONS)} points"
    )

    print(
        f"  Latitude    : "
        f"{len(TARGET_LATS)} points"
    )

    print(
        "  Resolution  : 0.25°"
    )

    print()
    print(
        "Native request buffer:"
    )

    print(
        f"  Longitude   : "
        f"{REQUEST_MIN_LON:.6f} -> "
        f"{REQUEST_MAX_LON:.6f}"
    )

    print(
        f"  Latitude    : "
        f"{REQUEST_MIN_LAT:.6f} -> "
        f"{REQUEST_MAX_LAT:.6f}"
    )

    print()
    print(
        "Target depths:"
    )

    print(
        "  "
        + ", ".join(
            f"{int(depth)}m"
            for depth
            in TARGET_DEPTHS
        )
    )

    print()
    print(
        "Native depth request:"
    )

    print(
        f"  {REQUEST_MIN_DEPTH} -> "
        f"{REQUEST_MAX_DEPTH} m"
    )

    print()
    print(
        "Storage strategy:"
    )

    print(
        "  Native GLORYS -> temporary"
    )

    print(
        "  Reduced data -> permanent"
    )

    print(
        "  Native file  -> DELETE"
    )

    if TEST_MODE:
        print()
        print(
            "MODE: TEST"
        )

        print(
            f"Test period: "
            f"{TEST_START_DATE} -> "
            f"{TEST_END_DATE}"
        )

        chunks = [
            (
                TEST_START_DATE,
                TEST_END_DATE,
            )
        ]

    else:
        print()
        print(
            "MODE: FULL"
        )

        print(
            f"Period: "
            f"{START_DATE} -> "
            f"{END_DATE}"
        )

        chunks = build_months()

        print(
            f"Monthly chunks: "
            f"{len(chunks):,}"
        )

    state = load_state()

    print()
    print(
        f"Completed chunks: "
        f"{len(state['completed']):,}"
    )

    print(
        f"Failed chunks   : "
        f"{len(state['failed']):,}"
    )

    total = len(chunks)
    successful = 0
    failed = 0

    started = time.time()

    for index, (
        start_date,
        end_date,
    ) in enumerate(
        chunks,
        start=1,
    ):
        print()
        print(
            f"[{index}/{total}] "
            f"{start_date} -> "
            f"{end_date}"
        )

        success = process_chunk(
            start_date,
            end_date,
            state,
        )

        if success:
            successful += 1
        else:
            failed += 1

    elapsed = (
        time.time()
        - started
    )

    print()
    print(
        "=" * 80
    )

    print(
        "GLORYS ACQUISITION FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"Chunks this run : "
        f"{total:,}"
    )

    print(
        f"Successful      : "
        f"{successful:,}"
    )

    print(
        f"Failed          : "
        f"{failed:,}"
    )

    print(
        f"Elapsed         : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "Final data:"
    )

    print(
        FINAL_DIR
    )

    print()
    print(
        "Temporary data:"
    )

    print(
        TEMP_DIR
    )

    print()
    print(
        "State:"
    )

    print(
        STATE_FILE
    )

    print()
    print(
        "Log:"
    )

    print(
        LOG_FILE
    )

    print(
        "=" * 80
    )

    if failed:
        print()
        print(
            "WARNING: Some chunks failed. "
            "Completed chunks are preserved; "
            "review the log and rerun the script."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "Acquisition interrupted by user."
        )

        print(
            "Completed chunks remain recorded "
            "in the state file."
        )

        sys.exit(130)

    except Exception as exc:
        logging.exception(
            "FATAL ERROR"
        )

        print()
        print(
            "=" * 80
        )

        print(
            "FATAL ERROR"
        )

        print(
            "=" * 80
        )

        print(
            str(exc)
        )

        print()

        sys.exit(1)
