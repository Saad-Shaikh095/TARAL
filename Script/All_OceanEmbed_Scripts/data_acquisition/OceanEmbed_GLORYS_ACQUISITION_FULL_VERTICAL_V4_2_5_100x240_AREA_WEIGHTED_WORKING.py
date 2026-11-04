from pathlib import Path
from calendar import monthrange
from datetime import datetime
import hashlib
import inspect
import json
import logging
import os
import sys
import time

import numpy as np
import xarray as xr
import copernicusmarine


# =============================================================================
# OCEANEMBED GLORYS ACQUISITION V4.2.5
# FULL NATIVE VERTICAL TARGET THROUGH 1062.44 m
# =============================================================================
#
# Purpose:
#   Acquire GLORYS thetao for OceanEmbed while PRESERVING ALL NATIVE
#   GLORYS VERTICAL LEVELS from the surface through the first native level
#   deeper than 1000 m (approximately 1062.44 m).
#
# Important change from the previous V3 pipeline:
#
#   V3:
#       Copernicus GLORYS native subset
#           -> temporary NetCDF
#           -> conservative area-weighted spatial aggregation to 0.25°
#           -> vertical interpolation to 15 SIH depths
#           -> final NetCDF
#
#   V4:
#       Copernicus GLORYS native subset
#           -> temporary NetCDF
#           -> conservative area-weighted spatial aggregation to 0.25°
#           -> KEEP ALL native depth levels through ~1062.44 m
#           -> final NetCDF
#
# The final product is intended to be the TRAINING TARGET.
# Do NOT reduce this file to 15 depths during acquisition.
#
# SIH 15-depth evaluation remains a downstream operation:
#   0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
#
# Credentials:
#   This script prefers the environment variables:
#       COPERNICUSMARINE_SERVICE_USERNAME
#       COPERNICUSMARINE_SERVICE_PASSWORD
#
#   If either env var is missing, it falls back to the hardcoded
#   defaults defined in DEFAULT_COPERNICUS_USERNAME / DEFAULT_COPERNICUS_PASSWORD.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(r"D:\OceanEmbed\oceanembed-data")

RAW_DIR = BASE_DIR / "GLORYS_FULL_VERTICAL"
TEMP_DIR = RAW_DIR / "_temporary"
FINAL_DIR = RAW_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
STATE_DIR = BASE_DIR / "manifests"

STATE_FILE = STATE_DIR / "glorys_full_vertical_acquisition_state.json"
LOG_FILE = LOG_DIR / "glorys_full_vertical_acquisition.log"

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

# Keep the pinned historical dataset version used by the existing
# OceanEmbed acquisition for reproducibility.
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

# Native GLORYS12V1 horizontal spacing.
NATIVE_STEP = 1.0 / 12.0

# OceanEmbed treats 45E-105E and 5N-30N as CELL-EDGE bounds.
# At 0.25 deg resolution this gives exactly 240 longitude cells and
# 100 latitude cells. Their centers are therefore:
#   longitude: 45.125 ... 104.875 (240)
#   latitude :  5.125 ...  29.875 (100)
#
# The installed Copernicus Toolbox returns the actual GLORYS native
# coordinates lying on the requested bounds. For this dataset, the exact
# OceanEmbed domain request therefore returns 301 x 721 native centers:
#   latitude  : 5.0 ... 30.0
#   longitude : 45.0 ... 105.0
#
# We intentionally KEEP those boundary centers. The target grid is defined
# by cell edges 5N-30N / 45E-105E, so the correct operation is conservative
# area-weighted aggregation from native GLORYS cells to the target cells.
# This avoids falsely forcing a 3x3 center grouping when the native grid
# origin is aligned to the domain edges.
REQUEST_MIN_LON = TARGET_MIN_LON
REQUEST_MAX_LON = TARGET_MAX_LON
REQUEST_MIN_LAT = TARGET_MIN_LAT
REQUEST_MAX_LAT = TARGET_MAX_LAT

EXPECTED_NATIVE_LAT_COUNT = 301
EXPECTED_NATIVE_LON_COUNT = 721


# =============================================================================
# DEPTH
# =============================================================================

# IMPORTANT:
# GLORYS has no coordinate at literal depth=0.0 m. The first physical
# GLORYS level is approximately 0.49402499 m.
#
# Copernicus `strict-inside` rejects a subset request when either bound is
# outside the dataset coordinate range. Therefore the minimum request depth
# must start at the actual first GLORYS coordinate, not literal 0.0.
#
# This preserves the first physical native level and avoids:
#   [0.0, 1100.0] exceeding [0.49402499198913574, 5727.9169921875]
REQUEST_MIN_DEPTH = 0.49402499198913574

# We request slightly deeper than the ~1062.44 m native level so the server
# includes the first native coordinate at/deeper than the SIH 1000 m target.
#
# The final product is explicitly trimmed to that native level.
REQUEST_MAX_DEPTH = 1100.0

PRESERVE_MAX_DEPTH = 1062.44

# The first physical GLORYS level is approximately 0.494 m.
MAX_ACCEPTABLE_SURFACE_LEVEL = 1.0

# Validation tolerance for rounded native coordinate values.
DEPTH_TOLERANCE = 1.0

# GLORYS12V1 has 50 fixed vertical levels. For OceanEmbed, preserve
# every native level from the first physical surface level through the
# first native level at or deeper than the SIH 1000 m target.
#
# With GLORYS12V1 this should normally be 36 levels, ending at about
# 1062.44 m. We validate the count and deepest level range, but do not
# hard-code every interior coordinate because the server's returned
# floating-point coordinate precision may vary slightly.
EXPECTED_NATIVE_DEPTH_COUNT = 36

SIH_DEEPEST_EVALUATION_DEPTH = 1000.0

# Expected physical GLORYS level immediately below the 1000 m target.
# These bounds are intentionally wider than floating-point rounding but
# narrow enough to detect an incorrect vertical subset.
EXPECTED_DEEPEST_NATIVE_MIN = 1050.0
EXPECTED_DEEPEST_NATIVE_MAX = 1075.0


# =============================================================================
# SIH 15 DEPTHS
# =============================================================================
#
# These are NOT used to reduce the training target.
# They are recorded in metadata for downstream validation/evaluation.
#
SIH_VALIDATION_DEPTHS = np.array(
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

# CELL-CENTERED 0.25 degree OceanEmbed grid over the 25 degree x 60 degree
# cell-edge domain. 100 latitude cells x 240 longitude cells.
TARGET_LONS = np.round(
    TARGET_MIN_LON + 0.125 + 0.25 * np.arange(240, dtype=np.float64),
    6,
)

TARGET_LATS = np.round(
    TARGET_MIN_LAT + 0.125 + 0.25 * np.arange(100, dtype=np.float64),
    6,
)


def validate_target_grid_definition():
    """Fail early if the OceanEmbed 100x240 grid definition is inconsistent."""
    if len(TARGET_LONS) != 240 or len(TARGET_LATS) != 100:
        raise RuntimeError(
            "OceanEmbed target grid must be exactly 100 x 240; got "
            f"{len(TARGET_LATS)} x {len(TARGET_LONS)}."
        )

    if not np.isclose(TARGET_LONS[0], 45.125, atol=1e-8):
        raise RuntimeError("Unexpected first longitude target center.")
    if not np.isclose(TARGET_LONS[-1], 104.875, atol=1e-8):
        raise RuntimeError("Unexpected last longitude target center.")
    if not np.isclose(TARGET_LATS[0], 5.125, atol=1e-8):
        raise RuntimeError("Unexpected first latitude target center.")
    if not np.isclose(TARGET_LATS[-1], 29.875, atol=1e-8):
        raise RuntimeError("Unexpected last latitude target center.")

    if not np.allclose(np.diff(TARGET_LONS), 0.25, atol=1e-8, rtol=0.0):
        raise RuntimeError("Target longitude spacing is not 0.25 degree.")
    if not np.allclose(np.diff(TARGET_LATS), 0.25, atol=1e-8, rtol=0.0):
        raise RuntimeError("Target latitude spacing is not 0.25 degree.")

    # The Copernicus request intentionally uses the exact OceanEmbed
    # cell-edge domain. The installed Toolbox returns the native coordinates
    # on those bounds, giving 301 x 721 source centers.
    if not np.isclose(REQUEST_MIN_LON, TARGET_MIN_LON, atol=1e-10):
        raise RuntimeError("Unexpected minimum native longitude request.")
    if not np.isclose(REQUEST_MAX_LON, TARGET_MAX_LON, atol=1e-10):
        raise RuntimeError("Unexpected maximum native longitude request.")
    if not np.isclose(REQUEST_MIN_LAT, TARGET_MIN_LAT, atol=1e-10):
        raise RuntimeError("Unexpected minimum native latitude request.")
    if not np.isclose(REQUEST_MAX_LAT, TARGET_MAX_LAT, atol=1e-10):
        raise RuntimeError("Unexpected maximum native latitude request.")


validate_target_grid_definition()


# =============================================================================
# TEST / PRODUCTION MODE
# =============================================================================

# Keep FALSE for the full acquisition.
#
# For a safe first test, change to TRUE:
#   TEST_MODE = True
#
# The default full run is 2012-01 through 2024-12.
TEST_MODE = False

TEST_START_DATE = "2012-01-01"
TEST_END_DATE = "2012-01-31"


# =============================================================================
# RETRIES / VALIDATION
# =============================================================================

MAX_RETRIES = 4
RETRY_DELAY_SECONDS = 30

# Copernicus documents dry_run=True as the pre-flight way to validate a
# subset request before the actual download. Use it when supported by the
# installed Toolbox version so request/credential/dataset problems fail fast.
ENABLE_SUBSET_PREFLIGHT = True

COORD_TOLERANCE = 0.01


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

# Hardcoded fallback credentials.
#
# The script first tries the standard Copernicus environment variables.
# If either env var is unset/empty, it falls back to these defaults so
# the pipeline can run without any environment configuration.
#
# NOTE: Keeping real credentials in source is convenient for local/offline
# use, but if this file is ever committed to version control, treat the
# values below as compromised and rotate them.
DEFAULT_COPERNICUS_USERNAME = "lhire"
DEFAULT_COPERNICUS_PASSWORD = "Hindusthan@11"


def get_copernicus_credentials():
    """
    Return Copernicus credentials.

    Resolution order:
      1. Environment variables:
             COPERNICUSMARINE_SERVICE_USERNAME
             COPERNICUSMARINE_SERVICE_PASSWORD
      2. Hardcoded fallback constants defined above.

    This function NEVER raises and NEVER prompts interactively.
    """
    env_username = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
    env_password = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")

    username = env_username.strip() if env_username else None
    password = env_password.strip() if env_password else None

    used_env_username = bool(username)
    used_env_password = bool(password)

    if not username:
        username = DEFAULT_COPERNICUS_USERNAME

    if not password:
        password = DEFAULT_COPERNICUS_PASSWORD

    source_bits = []
    source_bits.append(
        "env" if used_env_username else "default"
    )
    source_bits.append(
        "env" if used_env_password else "default"
    )

    print(
        "  Auth source  : "
        f"username={source_bits[0]}, "
        f"password={source_bits[1]}"
    )

    return username, password


# =============================================================================
# STATE
# =============================================================================

GRID_SIGNATURE = (
    "0.25deg_cell_edges_45E_105E_5N_30N_"
    "cell_centered_100x240"
)


def default_state():
    return {
        "product_id": PRODUCT_ID,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "grid_signature": GRID_SIGNATURE,
        "target_grid_shape": [len(TARGET_LATS), len(TARGET_LONS)],
        "target_longitude_centers": [float(v) for v in TARGET_LONS],
        "target_latitude_centers": [float(v) for v in TARGET_LATS],
        "preserve_max_depth_m": PRESERVE_MAX_DEPTH,
        "expected_native_depth_count": EXPECTED_NATIVE_DEPTH_COUNT,
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

        # Never reuse completion records from the previous 101x241 grid.
        # Preserve them under legacy_* for auditability, but force every
        # chunk through the new 100x240 validation/acquisition path.
        if state.get("grid_signature") != GRID_SIGNATURE:
            legacy_signature = state.get("grid_signature", "unknown")
            legacy_completed = list(state.get("completed", []))
            legacy_failed = list(state.get("failed", []))
            legacy_records = dict(state.get("records", {}))

            state = default_state()
            state["legacy_grid_signature"] = legacy_signature
            state["legacy_completed"] = legacy_completed
            state["legacy_failed"] = legacy_failed
            state["legacy_records"] = legacy_records

            save_state(state)

            print()
            print(
                "Detected acquisition state from a different horizontal grid. "
                "Legacy completion records were quarantined; no 101x241 file "
                "will be reused for the 100x240 run."
            )

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
        "NIO_0p25deg_FULLVERTICAL_to_1062m.nc"
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
# DEPTH PRESERVATION
# =============================================================================

def preserve_native_vertical(thetao):
    """
    Preserve every native GLORYS depth from the surface through the
    first native level at or deeper than the SIH 1000 m target.

    No vertical interpolation is performed.

    This is intentionally coordinate-driven:
      - find the first native GLORYS level >= 1000 m
      - keep every native level from the surface through that level
      - validate that the result is the expected 36-level, ~1062.44 m
        OceanEmbed training target

    We do NOT hard-code all 36 floating-point depth coordinates.
    """
    thetao = thetao.sortby("depth")

    depths = np.asarray(
        thetao.depth.values,
        dtype=np.float64,
    )

    if len(depths) < 2:
        raise RuntimeError(
            "Not enough native depth levels."
        )

    if not np.all(np.isfinite(depths)):
        raise RuntimeError(
            "Native depth coordinate contains non-finite values."
        )

    if not np.all(np.diff(depths) > 0):
        raise RuntimeError(
            "Native depth coordinate is not strictly increasing."
        )

    if depths[0] > MAX_ACCEPTABLE_SURFACE_LEVEL:
        raise RuntimeError(
            "Native data starts too deep for the surface: "
            f"{depths[0]:.6f} m"
        )

    # Locate the FIRST native level at or deeper than 1000 m.
    deeper_than_1000 = np.flatnonzero(
        depths >= SIH_DEEPEST_EVALUATION_DEPTH
    )

    if deeper_than_1000.size == 0:
        raise RuntimeError(
            "Native GLORYS subset does not contain a level at or "
            "deeper than the 1000 m SIH target."
        )

    end_index = int(deeper_than_1000[0])
    deepest_native_depth = float(
        depths[end_index]
    )

    if not (
        EXPECTED_DEEPEST_NATIVE_MIN
        <= deepest_native_depth
        <= EXPECTED_DEEPEST_NATIVE_MAX
    ):
        raise RuntimeError(
            "Unexpected native level around the 1000 m target: "
            f"{deepest_native_depth:.6f} m. "
            "Expected the GLORYS native level to be approximately "
            "1062.44 m."
        )

    # Preserve the complete native sequence through that level.
    preserved = thetao.isel(
        depth=slice(0, end_index + 1)
    )

    preserved_depths = np.asarray(
        preserved.depth.values,
        dtype=np.float64,
    )

    if len(preserved_depths) != EXPECTED_NATIVE_DEPTH_COUNT:
        raise RuntimeError(
            "Unexpected native depth count after preservation: "
            f"{len(preserved_depths)}; "
            f"expected {EXPECTED_NATIVE_DEPTH_COUNT}."
        )

    if float(preserved_depths[-1]) != deepest_native_depth:
        raise RuntimeError(
            "Internal depth-selection inconsistency."
        )

    return preserved


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

            depth_values = np.asarray(
                ds.depth.values,
                dtype=np.float64,
            )

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
                f"{depth_min:.4f} -> {depth_max:.4f} m"
            )
            print(
                f"  Native depth count : "
                f"{len(depth_values)}"
            )
            print(
                f"  Time points  : "
                f"{time_count}"
            )

            # -----------------------------------------------------------------
            # Horizontal coverage.
            # -----------------------------------------------------------------

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

            # -----------------------------------------------------------------
            # Depth coverage.
            # -----------------------------------------------------------------

            # The first returned native level must be the physical GLORYS
            # surface level, not a deeper layer caused by an incorrect subset.
            if abs(
                depth_min - REQUEST_MIN_DEPTH
            ) > 0.01:
                raise RuntimeError(
                    "Unexpected first native GLORYS depth: "
                    f"{depth_min:.8f} m. "
                    f"Expected approximately {REQUEST_MIN_DEPTH:.8f} m."
                )

            if depth_min > MAX_ACCEPTABLE_SURFACE_LEVEL:
                raise RuntimeError(
                    "Native file starts too deep for "
                    f"the surface: {depth_min:.4f} m"
                )

            if depth_max < (
                PRESERVE_MAX_DEPTH - DEPTH_TOLERANCE
            ):
                raise RuntimeError(
                    "Native GLORYS subset does not reach "
                    f"~{PRESERVE_MAX_DEPTH:.2f} m."
                )

            # The subset must include the complete native vertical sequence
            # through the first native level at/deeper than the 1000 m target.
            if len(depth_values) < EXPECTED_NATIVE_DEPTH_COUNT:
                raise RuntimeError(
                    "Native GLORYS subset returned too few depth levels: "
                    f"{len(depth_values)}; "
                    f"expected at least {EXPECTED_NATIVE_DEPTH_COUNT}."
                )

            deeper_than_1000 = np.flatnonzero(
                depth_values >= SIH_DEEPEST_EVALUATION_DEPTH
            )

            if deeper_than_1000.size == 0:
                raise RuntimeError(
                    "Native GLORYS subset has no level at or deeper "
                    "than the 1000 m SIH target."
                )

            first_deep_index = int(deeper_than_1000[0])
            first_deep_depth = float(
                depth_values[first_deep_index]
            )

            if not (
                EXPECTED_DEEPEST_NATIVE_MIN
                <= first_deep_depth
                <= EXPECTED_DEEPEST_NATIVE_MAX
            ):
                raise RuntimeError(
                    "Unexpected GLORYS native level around 1000 m: "
                    f"{first_deep_depth:.6f} m. "
                    "Expected approximately 1062.44 m."
                )

            if first_deep_index + 1 != EXPECTED_NATIVE_DEPTH_COUNT:
                raise RuntimeError(
                    "Unexpected number of native levels through the "
                    "first level at/deeper than 1000 m: "
                    f"{first_deep_index + 1}; "
                    f"expected {EXPECTED_NATIVE_DEPTH_COUNT}."
                )

            # -----------------------------------------------------------------
            # Time coverage.
            # -----------------------------------------------------------------

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
                "longitude": [lon_min, lon_max],
                "latitude": [lat_min, lat_max],
                "depth": [depth_min, depth_max],
                "native_depth_count": len(depth_values),
                "native_depths": depth_values.tolist(),
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
# CONSERVATIVE AREA-WEIGHTED REGRIDDING: GLORYS -> 0.25°
# =============================================================================

def _cell_boundaries_from_centers(centers):
    """Return cell boundaries implied by a 1-D center coordinate array."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 2:
        raise RuntimeError("Native coordinate must contain at least two centers.")
    if not np.all(np.diff(centers) > 0):
        raise RuntimeError("Native coordinate must be strictly increasing.")

    boundaries = np.empty(len(centers) + 1, dtype=np.float64)
    boundaries[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    boundaries[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    boundaries[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return boundaries


def _build_overlap_weights(source_centers, target_edges, latitude=False):
    """
    Build source-index and physical-area weights for every target cell.

    Latitude weights use sin(phi_hi)-sin(phi_lo), which is proportional to
    spherical surface area. Longitude weights use delta-lambda.
    """
    source_centers = np.asarray(source_centers, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    source_bounds = _cell_boundaries_from_centers(source_centers)

    index_rows = []
    weight_rows = []

    for i in range(len(target_edges) - 1):
        lo = float(target_edges[i])
        hi = float(target_edges[i + 1])

        overlap_lo = np.maximum(source_bounds[:-1], lo)
        overlap_hi = np.minimum(source_bounds[1:], hi)
        overlap = np.maximum(0.0, overlap_hi - overlap_lo)
        indices = np.flatnonzero(overlap > 1e-12)

        if len(indices) != 4:
            raise RuntimeError(
                "Unexpected native-cell overlap count: "
                f"target cell {i} intersects {len(indices)} native cells; "
                "expected exactly 4."
            )

        if latitude:
            # Exact spherical meridional area factor.
            weights = (
                np.sin(np.deg2rad(overlap_hi[indices]))
                - np.sin(np.deg2rad(overlap_lo[indices]))
            )
        else:
            weights = np.deg2rad(overlap[indices])

        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise RuntimeError("Invalid native-to-target area weights.")

        index_rows.append(indices.astype(np.int64))
        weight_rows.append(weights.astype(np.float64))

    return np.asarray(index_rows), np.asarray(weight_rows)


def validate_area_weighted_geometry(thetao):
    """Validate the exact source geometry required by area-weighted mapping."""
    if "latitude" not in thetao.dims or "longitude" not in thetao.dims:
        raise RuntimeError(
            "thetao must have latitude and longitude dimensions "
            "for area-weighted regridding."
        )

    n_lat = int(thetao.sizes["latitude"])
    n_lon = int(thetao.sizes["longitude"])

    if n_lat != EXPECTED_NATIVE_LAT_COUNT:
        raise RuntimeError(
            "Native latitude size mismatch: "
            f"{n_lat}; expected {EXPECTED_NATIVE_LAT_COUNT}."
        )

    if n_lon != EXPECTED_NATIVE_LON_COUNT:
        raise RuntimeError(
            "Native longitude size mismatch: "
            f"{n_lon}; expected {EXPECTED_NATIVE_LON_COUNT}."
        )

    source_lat = np.asarray(thetao.latitude.values, dtype=np.float64)
    source_lon = np.asarray(thetao.longitude.values, dtype=np.float64)

    if not np.allclose(np.diff(source_lat), NATIVE_STEP, atol=1e-5, rtol=0.0):
        raise RuntimeError("Native latitude spacing is not the expected 1/12 degree grid.")
    if not np.allclose(np.diff(source_lon), NATIVE_STEP, atol=1e-5, rtol=0.0):
        raise RuntimeError("Native longitude spacing is not the expected 1/12 degree grid.")

    if not np.isclose(source_lat[0], TARGET_MIN_LAT, atol=2e-4, rtol=0.0):
        raise RuntimeError(
            f"Unexpected native latitude start: {source_lat[0]:.8f}."
        )
    if not np.isclose(source_lat[-1], TARGET_MAX_LAT, atol=2e-4, rtol=0.0):
        raise RuntimeError(
            f"Unexpected native latitude end: {source_lat[-1]:.8f}."
        )
    if not np.isclose(source_lon[0], TARGET_MIN_LON, atol=2e-4, rtol=0.0):
        raise RuntimeError(
            f"Unexpected native longitude start: {source_lon[0]:.8f}."
        )
    if not np.isclose(source_lon[-1], TARGET_MAX_LON, atol=2e-4, rtol=0.0):
        raise RuntimeError(
            f"Unexpected native longitude end: {source_lon[-1]:.8f}."
        )

    lat_edges = np.linspace(
        TARGET_MIN_LAT,
        TARGET_MAX_LAT,
        len(TARGET_LATS) + 1,
        dtype=np.float64,
    )
    lon_edges = np.linspace(
        TARGET_MIN_LON,
        TARGET_MAX_LON,
        len(TARGET_LONS) + 1,
        dtype=np.float64,
    )

    lat_indices, lat_weights = _build_overlap_weights(
        source_lat,
        lat_edges,
        latitude=True,
    )
    lon_indices, lon_weights = _build_overlap_weights(
        source_lon,
        lon_edges,
        latitude=False,
    )

    if lat_indices.shape != (len(TARGET_LATS), 4):
        raise RuntimeError("Latitude overlap geometry is not 100 x 4.")
    if lon_indices.shape != (len(TARGET_LONS), 4):
        raise RuntimeError("Longitude overlap geometry is not 240 x 4.")

    # The target cell edges must be exactly represented by the overlap geometry.
    lat_weight_sums = lat_weights.sum(axis=1)
    lon_weight_sums = lon_weights.sum(axis=1)
    expected_lat_area = np.sin(np.deg2rad(lat_edges[1:])) - np.sin(np.deg2rad(lat_edges[:-1]))
    expected_lon_width = np.deg2rad(np.diff(lon_edges))

    if not np.allclose(lat_weight_sums, expected_lat_area, atol=1e-12, rtol=1e-10):
        raise RuntimeError("Latitude area weights do not exactly cover target cells.")
    if not np.allclose(lon_weight_sums, expected_lon_width, atol=1e-12, rtol=1e-10):
        raise RuntimeError("Longitude weights do not exactly cover target cells.")

    print("  Area-weighted geometry : VERIFIED")
    print(f"  Native grid            : {n_lat} x {n_lon}")
    print(f"  OceanEmbed grid        : {len(TARGET_LATS)} x {len(TARGET_LONS)}")
    print("  Mapping                : native cell-area overlaps -> 0.25° target cells")
    print("  Native cells / target : 4 latitude x 4 longitude")
    print("  Latitude weighting    : spherical area (sin latitude)")

    return lat_indices, lat_weights, lon_indices, lon_weights


def _area_weighted_block(data, lat_indices, lat_weights, lon_indices, lon_weights):
    """Area-weight one in-memory time/depth block without large rolling arrays."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise RuntimeError(
            f"Expected block with 4 dimensions (time, depth, lat, lon); got {data.shape}."
        )

    n_time, n_depth = data.shape[:2]
    n_lat = len(lat_indices)
    n_lon = len(lon_indices)

    numerator = np.zeros((n_time, n_depth, n_lat, n_lon), dtype=np.float64)
    denominator = np.zeros_like(numerator)

    for a in range(4):
        lat_values = np.take(data, lat_indices[:, a], axis=2)
        lat_weight = lat_weights[:, a][None, None, :, None]

        for b in range(4):
            values = np.take(lat_values, lon_indices[:, b], axis=3)
            weight = lat_weight * lon_weights[:, b][None, None, None, :]

            valid = np.isfinite(values)
            numerator += np.where(valid, values, 0.0) * weight
            denominator += valid * weight

    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = denominator > 0.0
    result[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return result


def reduce_spatial(thetao):
    """
    Regrid GLORYS native cell-centered values to the OceanEmbed 0.25°
    cell-centered grid using conservative area-weighted overlap.

    The native GLORYS grid is aligned to the OceanEmbed domain edges, so a
    target 0.25° cell intersects four native cells in each horizontal
    direction. The calculation therefore uses 4 x 4 native cells with
    fractional edge weights, rather than incorrectly forcing a 3 x 3 center
    grouping.

    Missing/land values are excluded from the weighted denominator. A target
    cell remains NaN only when every overlapping native value is NaN.
    """
    print()
    print(
        "Regridding GLORYS 1/12° -> OceanEmbed 0.25° "
        "using CONSERVATIVE AREA-WEIGHTED aggregation..."
    )

    thetao = thetao.sortby("latitude")
    thetao = thetao.sortby("longitude")

    lat_indices, lat_weights, lon_indices, lon_weights = (
        validate_area_weighted_geometry(thetao)
    )

    expected_shape = (
        thetao.sizes.get("time", 0),
        thetao.sizes.get("depth", 0),
        len(TARGET_LATS),
        len(TARGET_LONS),
    )

    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise RuntimeError("Native GLORYS data has invalid time/depth dimensions.")

    # Process modest time/depth blocks to keep peak RAM bounded. The native
    # January test is roughly 1 GB uncompressed, so loading the entire array
    # at once is deliberately avoided.
    time_block = 2
    depth_block = 4

    output = np.full(expected_shape, np.nan, dtype=np.float32)

    for t0 in range(0, expected_shape[0], time_block):
        t1 = min(t0 + time_block, expected_shape[0])

        for d0 in range(0, expected_shape[1], depth_block):
            d1 = min(d0 + depth_block, expected_shape[1])

            source_block = thetao.isel(
                time=slice(t0, t1),
                depth=slice(d0, d1),
            ).values

            output[t0:t1, d0:d1, :, :] = _area_weighted_block(
                source_block,
                lat_indices,
                lat_weights,
                lon_indices,
                lon_weights,
            )

    reduced = xr.DataArray(
        output,
        dims=("time", "depth", "latitude", "longitude"),
        coords={
            "time": thetao.time.values,
            "depth": thetao.depth.values,
            "latitude": TARGET_LATS,
            "longitude": TARGET_LONS,
        },
        name=thetao.name,
        attrs=dict(thetao.attrs),
    )

    if reduced.sizes.get("latitude", -1) != len(TARGET_LATS):
        raise RuntimeError("Area-weighted latitude size mismatch.")
    if reduced.sizes.get("longitude", -1) != len(TARGET_LONS):
        raise RuntimeError("Area-weighted longitude size mismatch.")

    print(
        "  Regridding complete   : "
        f"{len(TARGET_LATS)} × {len(TARGET_LONS)}"
    )
    print("  Value aggregation     : conservative 4 × 4 native-cell overlap")
    print("  Latitude weighting    : spherical cell area")
    print("  NaN handling          : weighted denominator excludes missing values")
    print("  Vertical levels       : UNCHANGED")

    return reduced


# =============================================================================
# FINAL VALIDATION
# =============================================================================

def validate_final_file(
    path,
    expected_start,
    expected_end,
):
    print(
        "Validating final OceanEmbed FULL-VERTICAL GLORYS file..."
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

        expected_time = expected_day_count(
            expected_start,
            expected_end,
        )

        expected_sizes = {
            "time": expected_time,
            "depth": EXPECTED_NATIVE_DEPTH_COUNT,
            "latitude": len(TARGET_LATS),
            "longitude": len(TARGET_LONS),
        }

        for dimension, expected_size in expected_sizes.items():
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

        depths = np.asarray(
            ds.depth.values,
            dtype=np.float64,
        )

        if depths[0] > MAX_ACCEPTABLE_SURFACE_LEVEL:
            raise RuntimeError(
                "Final file starts too deep for surface:"
                f" {depths[0]:.4f} m"
            )

        if abs(
            depths[-1] - PRESERVE_MAX_DEPTH
        ) > DEPTH_TOLERANCE:
            raise RuntimeError(
                "Final deepest native level is not near "
                f"{PRESERVE_MAX_DEPTH:.2f} m: "
                f"{depths[-1]:.4f} m"
            )

        # Ensure depths are strictly increasing.
        if not np.all(np.diff(depths) > 0):
            raise RuntimeError(
                "Final depth coordinate is not strictly increasing."
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

        thetao = ds["thetao"]

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
                "Final thetao contains no valid values."
            )

        print(
            f"  Shape        : {thetao.shape}"
        )
        print(
            f"  Time         : {first_day} -> {last_day}"
        )
        print(
            f"  Depths       : {len(depths)} native levels"
        )
        print(
            f"  Depth range  : "
            f"{depths[0]:.4f} -> {depths[-1]:.4f} m"
        )
        print(
            f"  Grid         : "
            f"{len(ds.latitude)} × {len(ds.longitude)}"
        )
        print(
            f"  Valid thetao : "
            f"{valid_fraction * 100:.2f}%"
        )

        print()
        print("  Preserved native depths:")
        print(
            "  "
            + ", ".join(
                f"{depth:.4f}"
                for depth in depths
            )
        )

        return {
            "shape": list(thetao.shape),
            "valid_fraction": valid_fraction,
            "depth_count": len(depths),
            "depths": depths.tolist(),
            "file_size_bytes": path.stat().st_size,
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
        "Processing native GLORYS data..."
    )

    started = time.time()

    final_temp = final_path.with_suffix(".tmp.nc")

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

            # IMPORTANT:
            # First reduce horizontally, then preserve ALL native vertical
            # levels through ~1062.44 m. No 15-depth interpolation occurs.
            thetao = reduce_spatial(thetao)
            thetao = preserve_native_vertical(thetao)

            thetao = thetao.astype(
                np.float32
            )

            final_ds = thetao.to_dataset(
                name="thetao"
            )

            final_ds.attrs.update(
                {
                    "title": (
                        "OceanEmbed GLORYS Daily "
                        "Subsurface Temperature - "
                        "Full Native Vertical Target "
                        "Through 1062.44 m"
                    ),
                    "source_product": PRODUCT_ID,
                    "source_dataset": DATASET_ID,
                    "source_dataset_version": (
                        DATASET_VERSION
                    ),
                    "source_variable": "thetao",
                    "processing": (
                        "Native GLORYS12V1 subset with "
                        "conservative area-weighted spatial aggregation to "
                        "0.25 degree; all native vertical "
                        "levels from the surface through "
                        "the native ~1062.44 m level are "
                        "preserved. No vertical interpolation "
                        "or 15-depth reduction is performed."
                    ),
                    "domain": (
                        "45E-105E, 5N-30N"
                    ),
                    "horizontal_resolution": (
                        "0.25 degree"
                    ),
                    "horizontal_grid_shape": (
                        "100 latitude cells x 240 longitude cells"
                    ),
                    "horizontal_coordinate_convention": (
                        "Cell-centered; domain edges 5N-30N and 45E-105E; "
                        "centers 5.125-29.875N and 45.125-104.875E"
                    ),
                    "source_native_horizontal_shape": (
                        "301 latitude centers x 721 longitude centers"
                    ),
                    "temporal_resolution": (
                        "Daily mean"
                    ),
                    "vertical_representation": (
                        "All native GLORYS levels from the physical "
                        "surface through the first native level at/"
                        "deeper than 1000 m (~1062.44 m)"
                    ),
                    "training_target": (
                        "Use all preserved native vertical levels"
                    ),
                    "evaluation_depths_m": (
                        ",".join(
                            str(int(depth))
                            for depth in SIH_VALIDATION_DEPTHS
                        )
                    ),
                    "created_by": "OceanEmbed",
                }
            )

            # Store the actual native vertical coordinate values in metadata
            # as a quick human-readable audit trail.
            actual_depths = np.asarray(
                thetao.depth.values,
                dtype=np.float64,
            )

            final_ds.attrs["native_depth_count"] = int(
                len(actual_depths)
            )

            final_ds.attrs["native_depth_coordinate_validation"] = (
                "validated native GLORYS depth sequence through first native level at/deeper than 1000 m"
            )

            final_ds.attrs["native_depths_m"] = ",".join(
                f"{depth:.6f}"
                for depth in actual_depths
            )

            encoding = {
                "thetao": {
                    "zlib": True,
                    "complevel": NETCDF_COMPRESSION_LEVEL,
                    "dtype": "float32",
                    "shuffle": True,
                }
            }

            print(
                "Writing compressed FULL-VERTICAL NetCDF..."
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
            "FULL-VERTICAL PROCESSING SUCCESS"
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
            f"Depths     : "
            f"{validation['depth_count']} native levels"
        )

        print(
            f"Depth range: "
            f"{validation['depths'][0]:.4f} -> "
            f"{validation['depths'][-1]:.4f} m"
        )

        print(
            f"Final shape: "
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

def _subset_supports_parameter(parameter_name):
    try:
        return parameter_name in inspect.signature(
            copernicusmarine.subset
        ).parameters
    except (TypeError, ValueError):
        return False


def _subset_kwargs(
    username,
    password,
    start_date,
    end_date,
    native_path,
):
    kwargs = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "username": username,
        "password": password,
        "variables": VARIABLES,
        "minimum_longitude": REQUEST_MIN_LON,
        "maximum_longitude": REQUEST_MAX_LON,
        "minimum_latitude": REQUEST_MIN_LAT,
        "maximum_latitude": REQUEST_MAX_LAT,
        "minimum_depth": REQUEST_MIN_DEPTH,
        "maximum_depth": REQUEST_MAX_DEPTH,
        "start_datetime": start_date + "T00:00:00",
        "end_datetime": end_date + "T23:59:59",
        "coordinates_selection_method": "strict-inside",
        "output_directory": str(TEMP_DIR),
        "output_filename": native_path.name,
        "file_format": "netcdf",
        "netcdf_compression_level": NETCDF_COMPRESSION_LEVEL,
        "overwrite": True,
    }

    if _subset_supports_parameter("disable_progress_bar"):
        kwargs["disable_progress_bar"] = True

    return kwargs


def preflight_subset_request(
    username,
    password,
    start_date,
    end_date,
    native_path,
):
    """Validate the exact subset request without downloading the data."""
    if not ENABLE_SUBSET_PREFLIGHT:
        return

    if not _subset_supports_parameter("dry_run"):
        print(
            "  Preflight          : SKIPPED (installed Toolbox has no dry_run parameter)"
        )
        return

    print(
        "  Preflight           : validating Copernicus subset request (dry_run)..."
    )

    kwargs = _subset_kwargs(
        username,
        password,
        start_date,
        end_date,
        native_path,
    )
    kwargs["dry_run"] = True

    try:
        copernicusmarine.subset(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Copernicus subset preflight failed: "
            f"{type(exc).__name__}: {repr(exc)}"
        ) from exc

    print(
        "  Preflight           : PASSED"
    )


def describe_exception(exc):
    """Return a useful error even when an exception has an empty str()."""
    message = str(exc).strip()
    representation = repr(exc)
    type_name = type(exc).__name__

    if message:
        return f"{type_name}: {message}"

    if representation and representation != f"{type_name}()":
        return f"{type_name}: {representation}"

    return f"{type_name}: <exception has no message>"


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
        f"  Time       : {start_date} -> {end_date}"
    )

    print(
        f"  Longitude  : {REQUEST_MIN_LON:.6f} -> {REQUEST_MAX_LON:.6f}"
    )

    print(
        f"  Latitude   : {REQUEST_MIN_LAT:.6f} -> {REQUEST_MAX_LAT:.6f}"
    )

    print(
        f"  Depth      : {REQUEST_MIN_DEPTH:.8f} -> {REQUEST_MAX_DEPTH} m"
    )

    print(
        f"  Permanent depth target: all native levels through ~{PRESERVE_MAX_DEPTH:.2f} m"
    )

    print(
        "  Variable   : thetao"
    )

    preflight_subset_request(
        username,
        password,
        start_date,
        end_date,
        native_path,
    )

    started = time.time()
    kwargs = _subset_kwargs(
        username,
        password,
        start_date,
        end_date,
        native_path,
    )

    try:
        copernicusmarine.subset(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Copernicus GLORYS subset download failed: "
            f"{describe_exception(exc)}"
        ) from exc

    elapsed = time.time() - started

    # Some Toolbox versions may normalize the filename.
    if not validate_file_exists(native_path):
        candidates = sorted(
            TEMP_DIR.glob(f"{native_path.stem}*.nc"),
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
            candidates[0].replace(native_path)

    if not validate_file_exists(native_path):
        raise RuntimeError(
            "Copernicus completed the request but the expected native output "
            "file was not found: "
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
        f"Size    : {size_mb:,.2f} MB"
    )

    print(
        f"Time    : {elapsed / 60:.2f} minutes"
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
            f"GLORYS FULL-VERTICAL CHUNK "
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

            # 1. Download native vertical subset.
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

            # 3. Reduce horizontal resolution but preserve all
            #    native vertical levels through ~1062.44 m.
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
                "native_depth_count": (
                    native_info["native_depth_count"]
                ),
                "native_depths": (
                    native_info["native_depths"]
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

            # 6. Delete temporary native file.
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
                f"Native depths: "
                f"{native_info['native_depth_count']}"
            )

            print(
                f"Final size   : "
                f"{final_size_mb:,.2f} MB"
            )

            print(
                f"Final depths : "
                f"{validation['depth_count']}"
            )

            print(
                f"Depth range  : "
                f"{validation['depths'][0]:.4f} -> "
                f"{validation['depths'][-1]:.4f} m"
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
                "final_bytes=%d depth_count=%d "
                "sha256=%s",
                chunk_id,
                native_info["size_bytes"],
                final_path.stat().st_size,
                validation["depth_count"],
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

            error_text = describe_exception(exc)

            print(
                f"Reason: {error_text}"
            )

            print(
                "  Full traceback: see the acquisition log file."
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
                    "last_error": describe_exception(exc),
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
    # Verify the canonical 100x240 cell-centered geometry before any network request.
    validate_target_grid_definition()

    # Resolve credentials early so any misconfiguration is visible
    # before any network request.
    username, _ = (
        get_copernicus_credentials()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "OCEANEMBED GLORYS ACQUISITION V4.2.3"
    )

    print(
        "FULL NATIVE VERTICAL TARGET THROUGH ~1062.44 m"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"Product       : {PRODUCT_ID}"
    )

    print(
        f"Dataset       : {DATASET_ID}"
    )

    print(
        f"Dataset ver.  : {DATASET_VERSION}"
    )

    print(
        "Variable      : thetao"
    )

    print(
        f"Copernicus user: {username}"
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
        f"  Longitude   : {len(TARGET_LONS)} centers "
        f"({TARGET_LONS[0]:.3f} -> {TARGET_LONS[-1]:.3f})"
    )

    print(
        f"  Latitude    : {len(TARGET_LATS)} centers "
        f"({TARGET_LATS[0]:.3f} -> {TARGET_LATS[-1]:.3f})"
    )

    print(
        "  Grid shape  : 100 x 240"
    )

    print(
        "  Resolution  : 0.25° cell-centered"
    )

    print(
        "  Cell edges  : 45.0E -> 105.0E, 5.0N -> 30.0N"
    )

    print()
    print(
        "Vertical strategy:"
    )

    print(
        f"  Native horizontal request: {EXPECTED_NATIVE_LAT_COUNT} x {EXPECTED_NATIVE_LON_COUNT} centers"
    )

    print(
        f"  Native request bounds   : lon {REQUEST_MIN_LON:.6f} -> {REQUEST_MAX_LON:.6f}; "
        f"lat {REQUEST_MIN_LAT:.6f} -> {REQUEST_MAX_LAT:.6f}"
    )

    print(
        f"  Download    : {REQUEST_MIN_DEPTH:.8f} -> {REQUEST_MAX_DEPTH:.1f} m"
    )

    print(
        "  Preserve    : ALL native GLORYS levels"
    )

    print(
        f"  Final max   : ~{PRESERVE_MAX_DEPTH:.2f} m"
    )

    print(
        f"  Expected    : "
        f"{EXPECTED_NATIVE_DEPTH_COUNT} native levels"
    )

    print(
        "  Interpolate : NO"
    )

    print(
        "  15-depth SIH reduction: NO"
    )

    print()
    print(
        "SIH validation depths:"
    )

    print(
        "  "
        + ", ".join(
            f"{int(depth)}m"
            for depth in SIH_VALIDATION_DEPTHS
        )
    )

    print()
    print(
        "Training target:"
    )

    print(
        "  Use ALL preserved native vertical levels."
    )

    print(
        "Evaluation:"
    )

    print(
        "  Evaluate/report only at the SIH 15 depths."
    )

    print()
    print(
        "Storage strategy:"
    )

    print(
        "  Native GLORYS -> temporary"
    )

    print(
        "  Full vertical target -> permanent"
    )

    print(
        "  Temporary native file -> DELETE after validation"
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
        "GLORYS FULL-VERTICAL ACQUISITION FINISHED"
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

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The permanent GLORYS files use the 100 x 240 OceanEmbed "
        "cell-centered horizontal grid."
    )

    print(
        "The permanent GLORYS files now retain "
        "all native vertical levels through "
        f"~{PRESERVE_MAX_DEPTH:.2f} m."
    )

    print(
        "Do not vertically resample these files "
        "to the 15 SIH depths before training."
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