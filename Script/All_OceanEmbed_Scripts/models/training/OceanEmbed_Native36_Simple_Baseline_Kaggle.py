from __future__ import annotations

# ============================================================
# OCEANEMBED NATIVE36 - SIMPLE REAL-DATA BASELINE
# ============================================================
#
# PURPOSE
# -------
# 1. Use the NEW 36-level GLORYS target directly.
# 2. Do NOT use Backbone / backbone_artifacts / TrainingFramework.
# 3. Load the same:
#       - 24 dynamic channels
#       - 6 static channels
#       - 4 spatial context channels
#       - 2 temporal context channels
# 4. Use a simple CNN producing all 36 native GLORYS depths.
# 5. Run a few real-data optimization steps.
#
# THIS IS A PIPELINE / INTEGRATION SANITY TEST.
# It is NOT a scientific benchmark.
#
# IMPORTANT:
# - No 15-depth reduction.
# - No Backbone.
# - No ARGO as input.
# - No Disaster Hazards as input.
# - GLORYS remains the target.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from calendar import monthrange
import math
import random
import re
import time

import numpy as np
import xarray as xr

import torch
from torch import nn


# ============================================================
# 1. CONFIG
# ============================================================

DATA_ROOT = Path(
    "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"
)

TARGET_ROOT = DATA_ROOT / "target" / "processed"

DUACS_FILE = DATA_ROOT / "Duacs_Oisst_2012_2024_FeaturesV1.nc"
OISSS_FILE = DATA_ROOT / "Oisss_GlobCurrent_2012_2024_FeaturesV1.nc"
CCMP_FILE = DATA_ROOT / "Ccmp_Oscar_2012_2024_100x240_FeaturesV1.nc"
GEBCO_FILE = DATA_ROOT / "GEBCO_OceanEmbed_Static_0p25deg_FINAL.nc"


# New GLORYS filename pattern.
TARGET_GLOB = (
    "GLORYS_thetao_*_NIO_0p25deg_FULLVERTICAL_to_1062m.nc"
)


# ------------------------------------------------------------
# Quick test settings
# ------------------------------------------------------------

QUICK_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
SEED = 42


# First target date.
#
# 14-day input window means this will use:
#
#   2012-01-19 ... 2012-02-01
#
# as the dynamic input window.
FIRST_TARGET_DATE = np.datetime64("2012-02-01", "D")


# ------------------------------------------------------------
# Expected OceanEmbed geometry
# ------------------------------------------------------------

H = 100
W = 240

WINDOW = 14

N_DYNAMIC = 24
N_STATIC = 6
N_SPATIAL = 4
N_TEMPORAL = 2

N_INPUT = (
    N_DYNAMIC
    + N_STATIC
    + N_SPATIAL
    + N_TEMPORAL
)


# SIH evaluation depths.
#
# They are NOT used to reduce the target in this test.
# The model still predicts all 36 native GLORYS depths.
SIH_DEPTHS = np.array(
    [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ],
    dtype=np.float32,
)


# ============================================================
# 2. CHANNEL CONTRACT
# ============================================================

DYNAMIC_VARS = [

    # --------------------------------------------------------
    # DUACS + OISST = 9
    # --------------------------------------------------------

    "duacs_sla",
    "duacs_adt",
    "duacs_ugos",
    "duacs_vgos",
    "oisst_sst",
    "oisst_anom",
    "oisst_err",
    "duacs_geospeed",
    "oisst_sst_gradient",

    # --------------------------------------------------------
    # OISSS + GlobCurrent = 8
    # --------------------------------------------------------

    "oisss_sss",
    "oisss_sss_empirical_uncertainty",
    "oisss_sss_formal_uncertainty",
    "globcurrent_uo",
    "globcurrent_vo",
    "globcurrent_speed",
    "oisss_sss_gradient",
    "globcurrent_vorticity",

    # --------------------------------------------------------
    # CCMP + OSCAR = 7
    # --------------------------------------------------------

    "ccmp_uwnd",
    "ccmp_vwnd",
    "oscar_u",
    "oscar_v",
    "ccmp_wind_speed",
    "oscar_current_speed",
    "oscar_vorticity",
]


STATIC_VARS = [
    "depth_mean_m",
    "depth_std_m",
    "ocean_fraction",
    "shallow_cell_fraction_0_200m",
    "bathymetric_slope_deg",
    "distance_to_coast_km",
]


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. BASIC FILE CHECKS
# ============================================================

for required_file in [
    DUACS_FILE,
    OISSS_FILE,
    CCMP_FILE,
    GEBCO_FILE,
]:

    if not required_file.is_file():

        raise FileNotFoundError(
            f"Required dataset not found:\n{required_file}"
        )


if not TARGET_ROOT.is_dir():

    raise FileNotFoundError(
        f"GLORYS target directory not found:\n{TARGET_ROOT}"
    )


GLORYS_FILES = sorted(
    TARGET_ROOT.glob(TARGET_GLOB)
)


if len(GLORYS_FILES) == 0:

    raise FileNotFoundError(
        "\n"
        "No GLORYS files matched.\n"
        f"Directory : {TARGET_ROOT}\n"
        f"Pattern   : {TARGET_GLOB}\n"
    )


# ============================================================
# 5. HEADER
# ============================================================

print("=" * 100)
print("OCEANEMBED NATIVE36 SIMPLE REAL-DATA BASELINE")
print("=" * 100)

print(f"DATA ROOT        : {DATA_ROOT}")
print(f"TARGET ROOT      : {TARGET_ROOT}")
print(f"Device           : {DEVICE}")

print()
print(f"Dynamic channels : {N_DYNAMIC}")
print(f"Static channels  : {N_STATIC}")
print(f"Spatial channels : {N_SPATIAL}")
print(f"Temporal channels: {N_TEMPORAL}")
print(f"Input channels   : {N_INPUT}")

print()
print(f"Input window     : {WINDOW} days")
print(f"Quick steps      : {QUICK_STEPS}")
print(f"Learning rate    : {LEARNING_RATE}")

print()
print("Target:")
print("  GLORYS native vertical levels")
print("  36 depths")
print("  ~0.494 m -> ~1062.44 m")

print()
print("ARGO             : independent validation, NOT input")
print("Disaster Hazards : NOT part of current input contract")

print()
print(f"GLORYS files discovered: {len(GLORYS_FILES)}")


# ============================================================
# 6. COORDINATE HELPERS
# ============================================================

def normalize_spatial(ds: xr.Dataset) -> xr.Dataset:

    rename = {}

    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"

    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"

    if rename:
        ds = ds.rename(rename)

    return ds


def as_float32(values) -> np.ndarray:

    return np.asarray(
        values,
        dtype=np.float32
    )


# ============================================================
# 7. OPEN INPUT DATASETS
# ============================================================

print()
print("=" * 100)
print("[OPENING INPUT DATASETS]")
print("=" * 100)


duacs = normalize_spatial(
    xr.open_dataset(DUACS_FILE)
)

oisss = normalize_spatial(
    xr.open_dataset(OISSS_FILE)
)

ccmp = normalize_spatial(
    xr.open_dataset(CCMP_FILE)
)

gebco = normalize_spatial(
    xr.open_dataset(GEBCO_FILE)
)


print(
    f"  {'DUACS/OISST':20s}: "
    f"{dict(duacs.sizes)}"
)

print(
    f"  {'OISSS/GlobCurrent':20s}: "
    f"{dict(oisss.sizes)}"
)

print(
    f"  {'CCMP/OSCAR':20s}: "
    f"{dict(ccmp.sizes)}"
)

print(
    f"  {'GEBCO':20s}: "
    f"{dict(gebco.sizes)}"
)


# ============================================================
# 8. VERIFY DYNAMIC VARIABLES
# ============================================================

print()
print("[VERIFYING DYNAMIC VARIABLES]")


DUACS_VARS = DYNAMIC_VARS[:9]
OISSS_VARS = DYNAMIC_VARS[9:17]
CCMP_VARS = DYNAMIC_VARS[17:24]


def verify_variables(
    dataset_name: str,
    ds: xr.Dataset,
    variables: list[str],
) -> None:

    missing = [
        variable
        for variable in variables
        if variable not in ds.data_vars
    ]

    if missing:

        raise KeyError(
            f"{dataset_name} missing variables:\n"
            f"{missing}"
        )

    print(
        f"  {dataset_name:20s}: "
        f"{len(variables)} variables OK"
    )


verify_variables(
    "DUACS/OISST",
    duacs,
    DUACS_VARS,
)

verify_variables(
    "OISSS/GlobCurrent",
    oisss,
    OISSS_VARS,
)

verify_variables(
    "CCMP/OSCAR",
    ccmp,
    CCMP_VARS,
)


missing_static = [
    variable
    for variable in STATIC_VARS
    if variable not in gebco.data_vars
]


if missing_static:

    raise KeyError(
        f"GEBCO missing static variables:\n"
        f"{missing_static}"
    )


print(
    f"  {'GEBCO static':20s}: "
    f"{len(STATIC_VARS)} variables OK"
)


# ============================================================
# 9. VERIFY COMMON GRID
# ============================================================

print()
print("[VERIFYING COMMON GRID]")


for dataset_name, ds in [
    ("DUACS/OISST", duacs),
    ("OISSS/GlobCurrent", oisss),
    ("CCMP/OSCAR", ccmp),
]:

    lat_size = ds.sizes.get("latitude")
    lon_size = ds.sizes.get("longitude")

    if lat_size != H or lon_size != W:

        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{H}x{W}, got "
            f"{lat_size}x{lon_size}"
        )

    print(
        f"  {dataset_name:20s}: "
        f"{lat_size} x {lon_size} OK"
    )


gebco_lat_size = gebco.sizes.get("latitude")
gebco_lon_size = gebco.sizes.get("longitude")


if (
    gebco_lat_size != H
    or gebco_lon_size != W
):

    raise RuntimeError(
        "GEBCO grid mismatch: "
        f"expected {H}x{W}, "
        f"got {gebco_lat_size}x{gebco_lon_size}"
    )


print(
    f"  {'GEBCO':20s}: "
    f"{gebco_lat_size} x {gebco_lon_size} OK"
)


# ============================================================
# 10. LOAD STATIC FEATURES
# ============================================================

print()
print("[LOADING STATIC FEATURES]")


static_arrays = []


for variable in STATIC_VARS:

    array = gebco[variable].load().values

    array = as_float32(array)

    if array.shape != (H, W):

        raise RuntimeError(
            f"Static variable '{variable}' "
            f"shape {array.shape} != {(H, W)}"
        )

    static_arrays.append(array)


static = np.stack(
    static_arrays,
    axis=0
)


# IMPORTANT:
# Save ocean fraction BEFORE nan_to_num.
ocean_fraction = static[2].copy()


ocean_mask = (
    ocean_fraction > 0.0
)


# Prevent static NaNs from poisoning the CNN.
static = np.nan_to_num(
    static,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
).astype(np.float32)


print(
    f"  Static shape : {static.shape}"
)

print(
    f"  Ocean cells  : "
    f"{int(ocean_mask.sum())}/{H * W}"
)


# ============================================================
# 11. SPATIAL CONTEXT
# ============================================================

print()
print("[BUILDING SPATIAL CONTEXT]")


lat = as_float32(
    gebco["latitude"].values
)

lon = as_float32(
    gebco["longitude"].values
)


if lat.size != H:

    raise RuntimeError(
        f"Latitude size {lat.size} != {H}"
    )


if lon.size != W:

    raise RuntimeError(
        f"Longitude size {lon.size} != {W}"
    )


lat_grid, lon_grid = np.meshgrid(
    lat,
    lon,
    indexing="ij",
)


spatial_context = np.stack(
    [
        np.sin(np.deg2rad(lat_grid)),
        np.cos(np.deg2rad(lat_grid)),
        np.sin(np.deg2rad(lon_grid)),
        np.cos(np.deg2rad(lon_grid)),
    ],
    axis=0,
).astype(np.float32)


if spatial_context.shape != (
    N_SPATIAL,
    H,
    W,
):

    raise RuntimeError(
        f"Spatial context shape "
        f"{spatial_context.shape}"
    )


print(
    f"  Spatial shape: "
    f"{spatial_context.shape}"
)


print(
    f"  Latitude     : "
    f"{lat[0]:.6f} -> {lat[-1]:.6f}"
)

print(
    f"  Longitude    : "
    f"{lon[0]:.6f} -> {lon[-1]:.6f}"
)


# ============================================================
# 12. ROBUST GLORYS FILENAME PARSER
# ============================================================
#
# IMPORTANT FIX
# --------------
# We do NOT depend on splitting the filename by "_".
#
# We explicitly extract:
#
#   GLORYS_thetao_YYYYMMDD_YYYYMMDD_...
#
# using regex.
#
# This avoids the previous month-index failure.
# ============================================================

print()
print("=" * 100)
print("[BUILDING ROBUST GLORYS MONTH INDEX]")
print("=" * 100)


GLORYS_FILENAME_RE = re.compile(
    r"GLORYS_thetao_"
    r"(?P<start>\d{8})_"
    r"(?P<end>\d{8})_"
)


def parse_glorys_filename(
    path: Path,
) -> tuple[np.datetime64, np.datetime64]:

    match = GLORYS_FILENAME_RE.search(
        path.name
    )

    if match is None:

        raise RuntimeError(
            "\n"
            "Could not parse GLORYS filename.\n"
            f"Filename:\n{path.name}\n\n"
            "Expected pattern similar to:\n"
            "GLORYS_thetao_20120101_20120131_..."
        )


    start_string = match.group("start")
    end_string = match.group("end")


    start = np.datetime64(
        f"{start_string[:4]}-"
        f"{start_string[4:6]}-"
        f"{start_string[6:8]}",
        "D",
    )


    end = np.datetime64(
        f"{end_string[:4]}-"
        f"{end_string[4:6]}-"
        f"{end_string[6:8]}",
        "D",
    )


    return start, end


# ------------------------------------------------------------
# Build month -> file mapping.
#
# Key:
#
#   "2012-01"
#   "2012-02"
#   ...
#   "2024-12"
# ------------------------------------------------------------

GLORYS_BY_MONTH: dict[str, Path] = {}

GLORYS_INDEX = []


for path in GLORYS_FILES:

    start, end = parse_glorys_filename(
        path
    )

    month_key = (
        np.datetime_as_string(
            start,
            unit="M",
        )
    )


    if month_key in GLORYS_BY_MONTH:

        previous = GLORYS_BY_MONTH[
            month_key
        ]

        raise RuntimeError(
            "\n"
            f"Duplicate GLORYS month detected: "
            f"{month_key}\n\n"
            f"File 1:\n{previous.name}\n\n"
            f"File 2:\n{path.name}"
        )


    GLORYS_BY_MONTH[
        month_key
    ] = path


    GLORYS_INDEX.append(
        (
            start,
            end,
            path,
        )
    )


GLORYS_INDEX.sort(
    key=lambda item: item[0]
)


# ============================================================
# 13. PRINT ACTUAL DISCOVERED GLORYS FILES
# ============================================================

print(
    f"  Files discovered : "
    f"{len(GLORYS_FILES)}"
)

print(
    f"  Unique months    : "
    f"{len(GLORYS_BY_MONTH)}"
)


if len(GLORYS_INDEX) > 0:

    first_start, first_end, first_path = (
        GLORYS_INDEX[0]
    )

    last_start, last_end, last_path = (
        GLORYS_INDEX[-1]
    )

    print()
    print(
        f"  First file       : "
        f"{first_path.name}"
    )

    print(
        f"  First period     : "
        f"{first_start} -> {first_end}"
    )

    print(
        f"  Last file        : "
        f"{last_path.name}"
    )

    print(
        f"  Last period      : "
        f"{last_start} -> {last_end}"
    )


# ============================================================
# 14. VERIFY EXPECTED 2012-01 -> 2024-12 MONTHS
# ============================================================

expected_months = []


for year in range(2012, 2025):

    for month in range(1, 13):

        expected_months.append(
            f"{year:04d}-{month:02d}"
        )


missing_months = [
    month
    for month in expected_months
    if month not in GLORYS_BY_MONTH
]


unexpected_months = [
    month
    for month in GLORYS_BY_MONTH
    if month not in expected_months
]


if missing_months:

    print()
    print("ERROR: Missing expected GLORYS months:")

    for month in missing_months:

        print(
            f"  {month}"
        )


    print()
    print(
        "Actual GLORYS filenames discovered:"
    )

    for path in GLORYS_FILES:

        print(
            f"  {path.name}"
        )


    raise RuntimeError(
        "\n"
        "GLORYS monthly coverage is incomplete.\n"
        f"Missing months: {missing_months}"
    )


if unexpected_months:

    print()
    print(
        "WARNING: GLORYS contains months "
        "outside the expected 2012-01 -> 2024-12 range:"
    )

    for month in sorted(
        unexpected_months
    ):

        print(
            f"  {month}"
        )


print()
print(
    "  Coverage         : "
    "2012-01 -> 2024-12 COMPLETE"
)

print(
    f"  Monthly files    : "
    f"{len(GLORYS_BY_MONTH)}"
)


# ============================================================
# 15. ROBUST GLORYS FILE LOOKUP
# ============================================================

def normalize_day(
    day: np.datetime64,
) -> np.datetime64:

    return np.datetime64(
        day,
        "D",
    )


def find_glorys_file(
    day: np.datetime64,
) -> Path:

    day = normalize_day(
        day
    )


    month_key = np.datetime_as_string(
        day,
        unit="M",
    )


    path = GLORYS_BY_MONTH.get(
        month_key
    )


    if path is None:

        raise KeyError(
            "\n"
            f"No GLORYS monthly file found "
            f"for {month_key}.\n"
            f"Requested day: {day}\n"
            f"Available range: "
            f"{min(GLORYS_BY_MONTH)} -> "
            f"{max(GLORYS_BY_MONTH)}"
        )


    return path


# ============================================================
# 16. TEST GLORYS LOOKUP BEFORE OPENING DATA
# ============================================================

print()
print("[TESTING GLORYS MONTH LOOKUP]")


test_dates = [
    np.datetime64("2012-01-01", "D"),
    np.datetime64("2012-01-15", "D"),
    np.datetime64("2012-02-01", "D"),
    np.datetime64("2012-06-15", "D"),
    np.datetime64("2020-01-01", "D"),
    np.datetime64("2024-12-01", "D"),
    np.datetime64("2024-12-31", "D"),
]


for test_day in test_dates:

    test_file = find_glorys_file(
        test_day
    )

    print(
        f"  {test_day} -> "
        f"{test_file.name}"
    )


# ============================================================
# 17. OPEN GLORYS FILE
# ============================================================

@lru_cache(maxsize=2)
def open_glorys(
    path_string: str,
) -> xr.Dataset:

    path = Path(
        path_string
    )


    ds = normalize_spatial(
        xr.open_dataset(path)
    )


    # --------------------------------------------------------
    # Required variables / coordinates
    # --------------------------------------------------------

    required = [
        "thetao",
        "time",
        "depth",
        "latitude",
        "longitude",
    ]


    missing = [
        variable
        for variable in required
        if (
            variable not in ds.variables
            and variable not in ds.coords
        )
    ]


    if missing:

        raise RuntimeError(
            f"GLORYS file {path.name} "
            f"missing: {missing}"
        )


    # --------------------------------------------------------
    # Horizontal shape
    # --------------------------------------------------------

    depth_size = ds.sizes.get(
        "depth"
    )

    latitude_size = ds.sizes.get(
        "latitude"
    )

    longitude_size = ds.sizes.get(
        "longitude"
    )


    if depth_size != 36:

        raise RuntimeError(
            f"GLORYS {path.name}: "
            f"expected 36 depths, "
            f"got {depth_size}"
        )


    if (
        latitude_size != H
        or longitude_size != W
    ):

        raise RuntimeError(
            f"GLORYS {path.name}: "
            f"expected {H}x{W}, "
            f"got "
            f"{latitude_size}x{longitude_size}"
        )


    # --------------------------------------------------------
    # Depth validation
    # --------------------------------------------------------

    depths = np.asarray(
        ds["depth"].values,
        dtype=np.float64,
    )


    if len(depths) != 36:

        raise RuntimeError(
            f"GLORYS {path.name}: "
            f"depth count is {len(depths)}, "
            "expected 36"
        )


    if not np.all(
        np.diff(depths) > 0
    ):

        raise RuntimeError(
            f"GLORYS {path.name}: "
            "depth coordinate is not increasing"
        )


    if not (
        1060.0
        < depths[-1]
        < 1065.0
    ):

        raise RuntimeError(
            f"GLORYS {path.name}: "
            f"unexpected deepest depth "
            f"{depths[-1]}"
        )


    # --------------------------------------------------------
    # Horizontal coordinate validation
    # --------------------------------------------------------

    glat = np.asarray(
        ds["latitude"].values,
        dtype=np.float64,
    )

    glon = np.asarray(
        ds["longitude"].values,
        dtype=np.float64,
    )


    if len(glat) != H:

        raise RuntimeError(
            f"{path.name}: "
            f"latitude count {len(glat)} != {H}"
        )


    if len(glon) != W:

        raise RuntimeError(
            f"{path.name}: "
            f"longitude count {len(glon)} != {W}"
        )


    # --------------------------------------------------------
    # Check target variable dimensions
    # --------------------------------------------------------

    thetao = ds["thetao"]


    if "time" not in thetao.dims:

        raise RuntimeError(
            f"{path.name}: "
            "thetao has no time dimension"
        )


    if "depth" not in thetao.dims:

        raise RuntimeError(
            f"{path.name}: "
            "thetao has no depth dimension"
        )


    return ds


# ============================================================
# 18. EXACT GLORYS DAILY DATE MATCH
# ============================================================

def load_target(
    day: np.datetime64,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    day = normalize_day(
        day
    )


    path = find_glorys_file(
        day
    )


    ds = open_glorys(
        str(path)
    )


    # --------------------------------------------------------
    # Normalize GLORYS time coordinate to calendar days.
    #
    # This avoids datetime64[ns] vs datetime64[D] precision
    # problems from xarray .sel().
    # --------------------------------------------------------

    raw_times = np.asarray(
        ds["time"].values
    )


    normalized_times = (
        raw_times.astype(
            "datetime64[D]"
        )
    )


    matches = np.flatnonzero(
        normalized_times == day
    )


    if len(matches) == 0:

        # Give useful diagnostic information.
        first_time = (
            normalized_times[0]
            if len(normalized_times) > 0
            else "EMPTY"
        )

        last_time = (
            normalized_times[-1]
            if len(normalized_times) > 0
            else "EMPTY"
        )


        raise RuntimeError(
            "\n"
            f"GLORYS file does not contain "
            f"requested day.\n"
            f"File       : {path.name}\n"
            f"Requested  : {day}\n"
            f"File range : "
            f"{first_time} -> {last_time}\n"
            f"Records    : {len(normalized_times)}"
        )


    if len(matches) > 1:

        raise RuntimeError(
            f"GLORYS {path.name}: "
            f"multiple records found for {day}"
        )


    time_index = int(
        matches[0]
    )


    # --------------------------------------------------------
    # Extract exact daily target
    # --------------------------------------------------------

    y = (
        ds["thetao"]
        .isel(time=time_index)
        .load()
        .values
    )


    y = as_float32(
        y
    )


    if y.shape != (
        36,
        H,
        W,
    ):

        raise RuntimeError(
            f"Target shape {y.shape} "
            f"!= {(36, H, W)}"
        )


    # --------------------------------------------------------
    # Validity mask
    # --------------------------------------------------------

    valid = (
        np.isfinite(y)
        & (
            ocean_fraction[
                None,
                :,
                :
            ] > 0.0
        )
    )


    depths = as_float32(
        ds["depth"].values
    )


    return (
        y,
        valid,
        depths,
    )


# ============================================================
# 19. DYNAMIC DATASET GROUPS
# ============================================================

SOURCE_DATASETS = [
    duacs,
    oisss,
    ccmp,
]


SOURCE_VAR_GROUPS = [
    DUACS_VARS,
    OISSS_VARS,
    CCMP_VARS,
]


# ============================================================
# 20. LOAD 14-DAY DYNAMIC WINDOW
# ============================================================

def load_dynamic_window(
    start_day: np.datetime64,
    end_day: np.datetime64,
) -> np.ndarray:

    start_day = np.datetime64(
        start_day,
        "D",
    )

    end_day = np.datetime64(
        end_day,
        "D",
    )


    pieces = []


    for ds, variables in zip(
        SOURCE_DATASETS,
        SOURCE_VAR_GROUPS,
    ):

        sub = ds.sel(
            time=slice(
                start_day,
                end_day,
            )
        )


        time_size = sub.sizes.get(
            "time"
        )


        if time_size != WINDOW:

            raise RuntimeError(
                "\n"
                "Dynamic window does not contain "
                f"{WINDOW} days.\n"
                f"Requested : "
                f"{start_day} -> {end_day}\n"
                f"Found     : {time_size}"
            )


        arrays = []


        for variable in variables:

            array = (
                sub[variable]
                .load()
                .values
            )


            if array.shape != (
                WINDOW,
                H,
                W,
            ):

                raise RuntimeError(
                    f"{variable}: "
                    f"shape {array.shape} "
                    f"!= {(WINDOW, H, W)}"
                )


            arrays.append(
                as_float32(array)
            )


        source_tensor = np.stack(
            arrays,
            axis=1,
        )


        pieces.append(
            source_tensor
        )


    x = np.concatenate(
        pieces,
        axis=1,
    )


    if x.shape != (
        WINDOW,
        N_DYNAMIC,
        H,
        W,
    ):

        raise RuntimeError(
            f"Dynamic shape {x.shape} "
            f"!= "
            f"{(WINDOW, N_DYNAMIC, H, W)}"
        )


    # For this simple sanity test only:
    # replace missing predictor values with zero.
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


    return x


# ============================================================
# 21. TEMPORAL CONTEXT
# ============================================================

def temporal_context(
    day: np.datetime64,
) -> np.ndarray:

    day = np.datetime64(
        day,
        "D",
    )


    date_string = np.datetime_as_string(
        day,
        unit="D",
    )


    year = int(
        date_string[:4]
    )

    month = int(
        date_string[5:7]
    )

    day_of_month = int(
        date_string[8:10]
    )


    day_of_year = (
        sum(
            monthrange(
                year,
                month_number,
            )[1]
            for month_number
            in range(1, month)
        )
        + day_of_month
    )


    is_leap_year = (
        year % 4 == 0
        and (
            year % 100 != 0
            or year % 400 == 0
        )
    )


    days_in_year = (
        366.0
        if is_leap_year
        else 365.0
    )


    phase = (
        2.0
        * math.pi
        * (day_of_year - 1)
        / days_in_year
    )


    return np.array(
        [
            math.sin(phase),
            math.cos(phase),
        ],
        dtype=np.float32,
    )


# ============================================================
# 22. REAL-DATA SAMPLE BUILDER
# ============================================================

def make_sample(
    day: np.datetime64,
):

    day = np.datetime64(
        day,
        "D",
    )


    # --------------------------------------------------------
    # 14-day dynamic window
    # --------------------------------------------------------

    start_day = (
        day
        - np.timedelta64(
            WINDOW - 1,
            "D",
        )
    )


    x_dynamic = load_dynamic_window(
        start_day,
        day,
    )


    # --------------------------------------------------------
    # 36-depth GLORYS target
    # --------------------------------------------------------

    y, target_valid, depths = load_target(
        day
    )


    # --------------------------------------------------------
    # Simple baseline:
    # 14-day mean of dynamic channels
    # --------------------------------------------------------

    dynamic_mean = np.mean(
        x_dynamic,
        axis=0,
    ).astype(np.float32)


    # --------------------------------------------------------
    # Temporal context
    # --------------------------------------------------------

    temporal = temporal_context(
        day
    )


    temporal_map = np.broadcast_to(
        temporal[
            :,
            None,
            None,
        ],
        (
            N_TEMPORAL,
            H,
            W,
        ),
    ).astype(np.float32)


    # --------------------------------------------------------
    # Full 36-channel input
    #
    # 24 dynamic
    # + 6 static
    # + 4 spatial
    # + 2 temporal
    # = 36
    # --------------------------------------------------------

    features = np.concatenate(
        [
            dynamic_mean,
            static,
            spatial_context,
            temporal_map,
        ],
        axis=0,
    )


    if features.shape != (
        N_INPUT,
        H,
        W,
    ):

        raise RuntimeError(
            f"Feature shape {features.shape} "
            f"!= {(N_INPUT, H, W)}"
        )


    y_clean = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


    x_tensor = torch.from_numpy(
        features[
            None,
            ...
        ]
    )


    y_tensor = torch.from_numpy(
        y_clean[
            None,
            ...
        ]
    )


    mask_tensor = torch.from_numpy(
        target_valid[
            None,
            ...
        ]
    )


    return (
        x_tensor,
        y_tensor,
        mask_tensor,
        depths,
    )


# ============================================================
# 23. REAL-DATA PREFLIGHT
# ============================================================

print()
print("=" * 100)
print("[REAL-DATA PREFLIGHT]")
print("=" * 100)


print(
    f"Target date : "
    f"{FIRST_TARGET_DATE}"
)


x0, y0, m0, depths0 = make_sample(
    FIRST_TARGET_DATE
)


print(
    f"  x shape       : "
    f"{tuple(x0.shape)}"
)

print(
    f"  y shape       : "
    f"{tuple(y0.shape)}"
)

print(
    f"  mask shape    : "
    f"{tuple(m0.shape)}"
)

print(
    f"  depths        : "
    f"{depths0.size}"
)

print(
    f"  depth range   : "
    f"{depths0[0]:.4f} -> "
    f"{depths0[-1]:.4f} m"
)

print(
    f"  valid target  : "
    f"{100.0 * m0.float().mean().item():.2f}%"
)

print(
    f"  x finite      : "
    f"{bool(torch.isfinite(x0).all())}"
)


if bool(
    m0.any()
):

    print(
        f"  y finite(mask): "
        f"{bool(torch.isfinite(y0[m0]).all())}"
    )

else:

    print(
        "  y finite(mask): "
        "NO VALID TARGET CELLS"
    )


# ============================================================
# 24. HARD PREFLIGHT CHECKS
# ============================================================

if x0.shape != (
    1,
    N_INPUT,
    H,
    W,
):

    raise RuntimeError(
        f"Preflight input shape failed: "
        f"{tuple(x0.shape)}"
    )


if y0.shape != (
    1,
    36,
    H,
    W,
):

    raise RuntimeError(
        f"Preflight target shape failed: "
        f"{tuple(y0.shape)}"
    )


if m0.shape != (
    1,
    36,
    H,
    W,
):

    raise RuntimeError(
        f"Preflight mask shape failed: "
        f"{tuple(m0.shape)}"
    )


if depths0.size != 36:

    raise RuntimeError(
        "Native36 preflight failed: "
        "target does not contain 36 depths"
    )


if not np.all(
    np.diff(depths0) > 0
):

    raise RuntimeError(
        "Native36 preflight failed: "
        "depths are not increasing"
    )


if not (
    1060.0
    < float(depths0[-1])
    < 1065.0
):

    raise RuntimeError(
        "Native36 preflight failed: "
        f"unexpected deepest depth {depths0[-1]}"
    )


# ============================================================
# 25. SIMPLE 36-DEPTH CNN
# ============================================================

class Native36SimpleCNN(
    nn.Module
):

    """
    Very small integration baseline.

    INPUT:
        24 dynamic mean channels
        + 6 static
        + 4 spatial
        + 2 temporal
        = 36 channels

    OUTPUT:
        36 native GLORYS depth levels
    """

    def __init__(self):

        super().__init__()


        self.net = nn.Sequential(

            nn.Conv2d(
                N_INPUT,
                48,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),


            nn.Conv2d(
                48,
                48,
                kernel_size=3,
                padding=1,
            ),

            nn.GELU(),


            nn.Conv2d(
                48,
                36,
                kernel_size=1,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


# ============================================================
# 26. MODEL
# ============================================================

model = Native36SimpleCNN().to(
    DEVICE
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print()
print("=" * 100)
print("[MODEL]")
print("=" * 100)

print(
    f"Parameters : "
    f"{parameter_count:,}"
)

print(
    "Input      : "
    "[B,36,100,240]"
)

print(
    "Output     : "
    "[B,36,100,240]"
)


# ============================================================
# 27. FEW REAL-DATA OPTIMIZATION STEPS
# ============================================================

print()
print("=" * 100)
print("[RUNNING REAL-DATA OPTIMIZATION]")
print("=" * 100)


model.train()


run_start = time.time()


for step in range(
    1,
    QUICK_STEPS + 1,
):

    # Consecutive target dates.
    day = (
        FIRST_TARGET_DATE
        + np.timedelta64(
            step - 1,
            "D",
        )
    )


    # --------------------------------------------------------
    # Load real sample
    # --------------------------------------------------------

    x, y, mask, depths = make_sample(
        day
    )


    x = x.to(
        DEVICE,
        non_blocking=True,
    )

    y = y.to(
        DEVICE,
        non_blocking=True,
    )

    mask = mask.to(
        DEVICE,
        non_blocking=True,
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    pred = model(
        x
    )


    if pred.shape != y.shape:

        raise RuntimeError(
            f"Prediction shape "
            f"{tuple(pred.shape)} "
            f"!= target "
            f"{tuple(y.shape)}"
        )


    if not bool(
        torch.isfinite(pred).all()
    ):

        raise FloatingPointError(
            "Model produced non-finite predictions"
        )


    # --------------------------------------------------------
    # Masked MSE
    # --------------------------------------------------------

    squared_error = (
        pred - y
    ).pow(2)


    valid_count = (
        mask.sum()
        .clamp_min(1)
    )


    loss = (
        squared_error[mask].sum()
        / valid_count
    )


    if not bool(
        torch.isfinite(loss)
    ):

        raise FloatingPointError(
            "Loss is non-finite"
        )


    # --------------------------------------------------------
    # Backprop
    # --------------------------------------------------------

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )


    optimizer.step()


    elapsed = (
        time.time()
        - run_start
    )


    valid_percentage = (
        100.0
        * mask.float().mean().item()
    )


    print(
        f"Step {step:02d}/{QUICK_STEPS}"
        f" | date="
        f"{np.datetime_as_string(day, unit='D')}"
        f" | loss={loss.item():.6f}"
        f" | valid={valid_percentage:.2f}%"
        f" | elapsed={elapsed:.1f}s",
        flush=True,
    )


# ============================================================
# 28. FINAL SANITY CHECK
# ============================================================

print()
print("=" * 100)
print("[FINAL SANITY CHECK]")
print("=" * 100)


model.eval()


with torch.no_grad():

    final_pred = model(
        x0.to(DEVICE)
    )


print(
    f"Final prediction shape : "
    f"{tuple(final_pred.shape)}"
)

print(
    f"Prediction finite      : "
    f"{bool(torch.isfinite(final_pred).all())}"
)


# ------------------------------------------------------------
# Check masked RMSE for sanity.
# ------------------------------------------------------------

final_target = y0.to(
    DEVICE
)

final_mask = m0.to(
    DEVICE
)


if bool(
    final_mask.any()
):

    final_mse = (
        (
            final_pred
            - final_target
        ).pow(2)[final_mask]
        .mean()
    )


    final_rmse = torch.sqrt(
        final_mse
    ).item()


    print(
        f"Final masked RMSE     : "
        f"{final_rmse:.6f} °C"
    )


# ============================================================
# 29. SUCCESS
# ============================================================

total_elapsed = (
    time.time()
    - run_start
)


print()
print("=" * 100)
print("NATIVE36 SIMPLE BASELINE: REAL-DATA FEW-STEP TEST PASSED")
print("=" * 100)

print()
print("Confirmed:")

print(
    "  [PASS] New GLORYS files discovered"
)

print(
    "  [PASS] Monthly GLORYS index built from actual filenames"
)

print(
    "  [PASS] 2012-01 -> 2024-12 monthly coverage"
)

print(
    "  [PASS] 24 dynamic channels loaded"
)

print(
    "  [PASS] 6 static channels loaded"
)

print(
    "  [PASS] 4 spatial context channels loaded"
)

print(
    "  [PASS] 2 temporal context channels loaded"
)

print(
    "  [PASS] 36 native GLORYS target depths loaded"
)

print(
    "  [PASS] 100 x 240 target grid"
)

print(
    "  [PASS] Real-data forward pass"
)

print(
    "  [PASS] Real-data loss calculation"
)

print(
    "  [PASS] Backpropagation"
)

print(
    "  [PASS] Optimizer step"
)

print(
    "  [PASS] Final prediction"
)

print()
print(
    f"Total optimization time: "
    f"{total_elapsed:.1f}s"
)

print()
print("IMPORTANT:")
print(
    "This is an integration/sanity test only."
)

print(
    "It is NOT a scientific 15-depth vs 36-depth comparison."
)

print(
    "No train/validation/test split was used."
)

print(
    "No train-only target normalization was used."
)

print(
    "The model is intentionally tiny."
)

print(
    "The purpose here is to prove that the NEW "
    "36-level GLORYS target can flow through the "
    "OceanEmbed data pipeline and train successfully."
)

print()
print("=" * 100)