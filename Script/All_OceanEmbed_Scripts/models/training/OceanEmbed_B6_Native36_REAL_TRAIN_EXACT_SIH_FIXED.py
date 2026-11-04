#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanEmbed B6-Native36 REAL RUN

TRAIN:
  * every eligible date in 2012-01-14 .. 2020-12-31
  * ALL 36 native GLORYS depths in the loss

VALIDATION / TEST:
  * ONLY the original 15 SIH depths are evaluated
  * validation/test do not score the remaining 21 native levels

Architecture:
  * real B6 mask-aware 3x3 projection 36->32
  * mask-aware 2x2 pooling -> 50x120
  * 2-layer spatial ConvLSTM, 32 hidden channels
  * 2x bilinear upsample
  * final-day high-resolution fusion
  * masked 3x3 decoder
  * masked 1x1 head 32->36

Target:
  * new GLORYS native 36-level collection
  * 0.494... -> 1062.44 m
  * 100x240 cell-centered grid
  * no depth interpolation during training
  * validation/test use exact SIH-depth metric mapping only
  * no Backbone artifacts
  * no target data enters the predictor tensor

Normalization:
  * unchanged 24 dynamic + 6 static train-only predictor statistics
  * new 36-depth train-only monthly GLORYS climatology
  * new 36-depth train-only anomaly standard deviation

Split:
  * train: 2012-01-01 .. 2020-12-31
  * val:   2021-01-01 .. 2022-12-31
  * test:  2023-01-01 .. 2024-12-31
  * input_within_split policy, 14-day window
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import time
from calendar import monthrange
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import torch
import torch.nn.functional as F
from torch import nn


# ============================================================================
# CONFIG
# ============================================================================

DATA_ROOT = Path("/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1")
TARGET_ROOT = DATA_ROOT / "target" / "processed"

DUACS_FILE = DATA_ROOT / "Duacs_Oisst_2012_2024_FeaturesV1.nc"
OISSS_FILE = DATA_ROOT / "Oisss_GlobCurrent_2012_2024_FeaturesV1.nc"
CCMP_FILE = DATA_ROOT / "Ccmp_Oscar_2012_2024_100x240_FeaturesV1.nc"
GEBCO_FILE = DATA_ROOT / "GEBCO_OceanEmbed_Static_0p25deg_FINAL.nc"

TARGET_GLOB = "GLORYS_thetao_*_NIO_0p25deg_FULLVERTICAL_to_1062m.nc"

GRID_H = 100
GRID_W = 240
WINDOW_DAYS = 14

DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CHANNELS = 4
TEMPORAL_CHANNELS = 2
TOTAL_INPUT_CHANNELS = 36

N_DEPTH = 36

SIH_DEPTHS_M = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=np.float64,
)

TRAIN_START = "2012-01-01"
TRAIN_END = "2020-12-31"
VAL_START = "2021-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42
AMP_ENABLED = True

LOG_EVERY = 25
DYNAMIC_DAY_CACHE_SIZE = 64
GLORYS_OPEN_CACHE_SIZE = 2
TARGET_DAY_CACHE_SIZE = 4

RUN_ROOT = Path("/kaggle/working/OceanEmbed_Runs")
EXPERIMENT_NAME = "CONVLSTM_MASKED_B6_NATIVE36"
PATTERN_VERSION = "B6_NATIVE36_REALRUN_V1"

# Set to a specific checkpoint path to resume THIS Native36 experiment.
RESUME_FROM: Optional[str] = None

# Stop cleanly after the current epoch if this wall-clock limit is reached.
SESSION_SOFT_LIMIT_HOURS = 11.5


# ============================================================================
# CHANNEL ORDER
# ============================================================================

DYNAMIC_VARS = [
    "duacs_sla",
    "duacs_adt",
    "duacs_ugos",
    "duacs_vgos",
    "oisst_sst",
    "oisst_anom",
    "oisst_err",
    "duacs_geospeed",
    "oisst_sst_gradient",
    "oisss_sss",
    "oisss_sss_empirical_uncertainty",
    "oisss_sss_formal_uncertainty",
    "globcurrent_uo",
    "globcurrent_vo",
    "globcurrent_speed",
    "oisss_sss_gradient",
    "globcurrent_vorticity",
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

DUACS_VARS = DYNAMIC_VARS[:9]
OISSS_VARS = DYNAMIC_VARS[9:17]
CCMP_VARS = DYNAMIC_VARS[17:24]


# ============================================================================
# VERIFIED TRAIN-ONLY PREDICTOR NORMALIZATION
# ============================================================================

DYNAMIC_MEAN = np.array([
    0.07942749845100085,
    0.8612582125187629,
    -8.937478315885423e-05,
    0.018409206760867835,
    28.282648715779374,
    0.32976628942980346,
    0.17667945853683328,
    0.2273019645241904,
    0.005145568274300638,
    34.70729842006135,
    0.32621312784875445,
    0.3253036174259157,
    0.031110467932021033,
    0.01117843785638541,
    0.2790384358904005,
    0.004380236260123362,
    -4.615104291789693e-08,
    0.6659558522620421,
    0.20397507602435688,
    0.011549108446757508,
    0.010916642200939719,
    3.8861716610752133,
    0.22831257440938915,
    -9.20918867687674e-09,
], dtype=np.float64)

DYNAMIC_STD = np.array([
    0.0963061571050102,
    0.1804334821390878,
    0.21356603068365193,
    0.21097738088989137,
    1.8017424762003613,
    0.7263709339889239,
    0.09584632268007205,
    0.19696019362117034,
    0.004978199584049074,
    2.006168766240221,
    0.1677503818030723,
    0.7407585508473763,
    0.2703060089487094,
    0.22481344403815276,
    0.21641809783736274,
    0.006085217124543345,
    5.171020761995382e-06,
    3.5499859417335604,
    3.335118041239444,
    0.2161439192905274,
    0.20047473568995072,
    3.017977035189871,
    0.18717434878095468,
    4.278129903998677e-06,
], dtype=np.float64)

STATIC_MEAN = np.array([
    2406.8568719219274,
    112.29810616909096,
    0.494109282395264,
    0.09219487268809946,
    1.3603298168288018,
    286.66007610369104,
], dtype=np.float64)

STATIC_STD = np.array([
    1615.7837800584325,
    159.93401578984256,
    0.49318098582328196,
    0.26962419739135096,
    1.7446405030520125,
    245.85379097475726,
], dtype=np.float64)


# ============================================================================
# SEED / DEVICE
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# ============================================================================
# HELPERS
# ============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def date_string(value: Any) -> str:
    return np.datetime_as_string(
        np.datetime64(value, "D"),
        unit="D",
    )


def month_key(value: Any) -> str:
    return np.datetime_as_string(
        np.datetime64(value, "D"),
        unit="M",
    )


def month_number(value: Any) -> int:
    return int(date_string(value)[5:7])


def normalize_spatial(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "lat" in ds.coords and "latitude" not in ds.coords:
        rename["lat"] = "latitude"
    if "lon" in ds.coords and "longitude" not in ds.coords:
        rename["lon"] = "longitude"
    if rename:
        ds = ds.rename(rename)
    return ds


def to_day_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    out = np.empty(len(arr), dtype="datetime64[D]")
    for i, value in enumerate(arr):
        try:
            out[i] = np.datetime64(value, "D")
        except Exception:
            out[i] = np.datetime64(str(value)[:10], "D")
    return out


def build_date_index(ds: xr.Dataset, name: str) -> Dict[str, int]:
    if "time" not in ds.coords:
        raise RuntimeError(f"{name}: missing time coordinate")
    days = to_day_array(ds["time"].values)
    result: Dict[str, int] = {}
    for idx, day in enumerate(days):
        key = date_string(day)
        if key in result:
            raise RuntimeError(f"{name}: duplicate calendar date {key}")
        result[key] = int(idx)
    return result


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def hash_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ============================================================================
# FILES
# ============================================================================

print_section("OCEANEMBED B6-NATIVE36 REAL RUN")

print(f"DATA ROOT       : {DATA_ROOT}")
print(f"TARGET ROOT     : {TARGET_ROOT}")
print(f"DEVICE          : {DEVICE}")
print(f"INPUT           : [B,{WINDOW_DAYS},{TOTAL_INPUT_CHANNELS},{GRID_H},{GRID_W}]")
print(f"TARGET          : [B,{N_DEPTH},{GRID_H},{GRID_W}]")
print("TRAIN LOSS      : ALL 36 native GLORYS depths")
print("VALIDATION      : ONLY 15 SIH depths")
print("TEST            : ONLY 15 SIH depths")

for p in [DUACS_FILE, OISSS_FILE, CCMP_FILE, GEBCO_FILE]:
    if not p.is_file():
        raise FileNotFoundError(f"Required dataset not found: {p}")

if not TARGET_ROOT.is_dir():
    raise FileNotFoundError(f"Target directory not found: {TARGET_ROOT}")

GLORYS_FILES = sorted(TARGET_ROOT.glob(TARGET_GLOB))

if len(GLORYS_FILES) != 156:
    raise RuntimeError(
        f"Expected exactly 156 new GLORYS monthly files, "
        f"found {len(GLORYS_FILES)}"
    )

print(f"GLORYS files    : {len(GLORYS_FILES)}")


# ============================================================================
# OPEN PREDICTORS
# ============================================================================

print_section("OPENING PREDICTOR DATASETS")

duacs = normalize_spatial(xr.open_dataset(DUACS_FILE))
oisss = normalize_spatial(xr.open_dataset(OISSS_FILE))
ccmp = normalize_spatial(xr.open_dataset(CCMP_FILE))
gebco = normalize_spatial(xr.open_dataset(GEBCO_FILE))

for name, ds in [
    ("DUACS/OISST", duacs),
    ("OISSS/GlobCurrent", oisss),
    ("CCMP/OSCAR", ccmp),
    ("GEBCO", gebco),
]:
    lat_n = ds.sizes.get("latitude")
    lon_n = ds.sizes.get("longitude")
    if lat_n != GRID_H or lon_n != GRID_W:
        raise RuntimeError(
            f"{name}: expected {GRID_H}x{GRID_W}, got {lat_n}x{lon_n}"
        )
    print(f"{name:20s}: {dict(ds.sizes)}")

for name, ds, variables in [
    ("DUACS/OISST", duacs, DUACS_VARS),
    ("OISSS/GlobCurrent", oisss, OISSS_VARS),
    ("CCMP/OSCAR", ccmp, CCMP_VARS),
]:
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"{name} missing variables: {missing}")
    print(f"{name:20s}: {len(variables)} required variables OK")

for variable in STATIC_VARS:
    if variable not in gebco.data_vars:
        raise KeyError(f"GEBCO missing static variable: {variable}")

if "depth_max_m" not in gebco.data_vars:
    raise KeyError("GEBCO missing required auxiliary: depth_max_m")


# ============================================================================
# EXACT DYNAMIC DATE INDEXES
# ============================================================================

print_section("BUILDING EXACT CALENDAR-DATE INDEXES")

SOURCE_DATASETS = [duacs, oisss, ccmp]
SOURCE_VARIABLE_GROUPS = [DUACS_VARS, OISSS_VARS, CCMP_VARS]

DUACS_DATE_INDEX = build_date_index(duacs, "DUACS/OISST")
OISSS_DATE_INDEX = build_date_index(oisss, "OISSS/GlobCurrent")
CCMP_DATE_INDEX = build_date_index(ccmp, "CCMP/OSCAR")

SOURCE_DATE_INDEXES = [
    DUACS_DATE_INDEX,
    OISSS_DATE_INDEX,
    CCMP_DATE_INDEX,
]

expected_all_dates = []
first = np.datetime64(TRAIN_START, "D")
last = np.datetime64(TEST_END, "D")
total_days = int(
    (last - first) / np.timedelta64(1, "D")
) + 1

for i in range(total_days):
    expected_all_dates.append(
        date_string(first + np.timedelta64(i, "D"))
    )

for name, index in [
    ("DUACS/OISST", DUACS_DATE_INDEX),
    ("OISSS/GlobCurrent", OISSS_DATE_INDEX),
    ("CCMP/OSCAR", CCMP_DATE_INDEX),
]:
    missing = [
        d for d in expected_all_dates
        if d not in index
    ]
    if missing:
        raise RuntimeError(
            f"{name} missing {len(missing)} expected dates. "
            f"First missing: {missing[:5]}"
        )
    print(f"{name:20s}: {len(index)} exact dates")


# ============================================================================
# STATIC FEATURES
# ============================================================================

print_section("STATIC FEATURES + STRUCTURAL OCEAN MASK")

static_raw = np.stack(
    [
        np.asarray(
            gebco[v].load().values,
            dtype=np.float32,
        )
        for v in STATIC_VARS
    ],
    axis=0,
)

if static_raw.shape != (
    STATIC_CHANNELS,
    GRID_H,
    GRID_W,
):
    raise RuntimeError(
        f"Static shape {static_raw.shape}"
    )

ocean_fraction_raw = static_raw[2].copy()

ocean_mask_np = (
    np.isfinite(ocean_fraction_raw)
    & (ocean_fraction_raw > 0.0)
)

depth_max_raw = np.asarray(
    gebco["depth_max_m"].load().values,
    dtype=np.float32,
)

if depth_max_raw.shape != (GRID_H, GRID_W):
    raise RuntimeError(
        f"depth_max_m shape {depth_max_raw.shape}"
    )

static_valid_np = np.isfinite(static_raw)

static_norm_np = (
    (
        static_raw.astype(np.float64)
        - STATIC_MEAN[:, None, None]
    )
    / STATIC_STD[:, None, None]
).astype(np.float32)

static_norm_np[~static_valid_np] = 0.0

lat = np.asarray(
    gebco["latitude"].values,
    dtype=np.float32,
)

lon = np.asarray(
    gebco["longitude"].values,
    dtype=np.float32,
)

if len(lat) != GRID_H or len(lon) != GRID_W:
    raise RuntimeError("GEBCO coordinate dimensions are not 100x240")

lat_grid, lon_grid = np.meshgrid(
    lat,
    lon,
    indexing="ij",
)

spatial_context_np = np.stack(
    [
        np.sin(np.deg2rad(lat_grid)),
        np.cos(np.deg2rad(lat_grid)),
        np.sin(np.deg2rad(lon_grid)),
        np.cos(np.deg2rad(lon_grid)),
    ],
    axis=0,
).astype(np.float32)

print(f"Static raw shape : {static_raw.shape}")
print(
    f"Ocean cells      : {int(ocean_mask_np.sum())}/"
    f"{GRID_H * GRID_W}"
)
print(
    f"Ocean fraction   : {float(ocean_mask_np.mean()):.4f}"
)
print(
    f"Latitude         : {lat[0]:.6f} -> {lat[-1]:.6f}"
)
print(
    f"Longitude        : {lon[0]:.6f} -> {lon[-1]:.6f}"
)


# ============================================================================
# GLORYS MONTH INDEX + NATIVE DEPTHS
# ============================================================================

print_section("NEW GLORYS INDEX + NATIVE DEPTH VALIDATION")

GLORYS_FILENAME_RE = re.compile(
    r"GLORYS_thetao_(?P<start>\d{8})_(?P<end>\d{8})_"
)

GLORYS_BY_MONTH: Dict[str, Path] = {}

for path in GLORYS_FILES:
    m = GLORYS_FILENAME_RE.search(path.name)
    if m is None:
        raise RuntimeError(f"Cannot parse GLORYS filename: {path.name}")

    start_text = m.group("start")
    start_day = np.datetime64(
        f"{start_text[:4]}-{start_text[4:6]}-{start_text[6:8]}",
        "D",
    )

    key = month_key(start_day)

    if key in GLORYS_BY_MONTH:
        raise RuntimeError(f"Duplicate GLORYS month: {key}")

    GLORYS_BY_MONTH[key] = path

expected_months = [
    f"{year:04d}-{month:02d}"
    for year in range(2012, 2025)
    for month in range(1, 13)
]

missing_months = [
    m for m in expected_months
    if m not in GLORYS_BY_MONTH
]

if missing_months:
    raise RuntimeError(
        f"Missing GLORYS months: {missing_months}"
    )

print("Coverage          : 2012-01 -> 2024-12 COMPLETE")


@lru_cache(maxsize=GLORYS_OPEN_CACHE_SIZE)
def open_glorys(path_string: str) -> xr.Dataset:
    path = Path(path_string)
    ds = normalize_spatial(
        xr.open_dataset(path)
    )

    for variable in [
        "thetao",
        "time",
        "depth",
        "latitude",
        "longitude",
    ]:
        if variable not in ds.variables and variable not in ds.coords:
            raise RuntimeError(
                f"{path.name} missing {variable}"
            )

    if (
        ds.sizes.get("depth") != N_DEPTH
        or ds.sizes.get("latitude") != GRID_H
        or ds.sizes.get("longitude") != GRID_W
    ):
        raise RuntimeError(
            f"{path.name}: invalid dimensions {dict(ds.sizes)}"
        )

    depth = np.asarray(
        ds["depth"].values,
        dtype=np.float64,
    )

    if depth.shape != (N_DEPTH,):
        raise RuntimeError(
            f"{path.name}: invalid depth vector shape {depth.shape}"
        )

    if not np.isfinite(depth).all():
        raise RuntimeError(
            f"{path.name}: non-finite depth vector"
        )

    if not np.all(np.diff(depth) > 0):
        raise RuntimeError(
            f"{path.name}: depth vector is not increasing"
        )

    if not 1060.0 < depth[-1] < 1065.0:
        raise RuntimeError(
            f"{path.name}: unexpected deepest depth {depth[-1]}"
        )

    return ds


reference_ds = open_glorys(
    str(
        GLORYS_BY_MONTH["2012-01"]
    )
)

NATIVE_DEPTHS_M = np.asarray(
    reference_ds["depth"].values,
    dtype=np.float64,
)

for path in GLORYS_FILES:
    ds = open_glorys(str(path))
    depth_here = np.asarray(
        ds["depth"].values,
        dtype=np.float64,
    )
    if not np.allclose(
        depth_here,
        NATIVE_DEPTHS_M,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"Native depth vector mismatch in {path.name}"
        )

print(f"Native depths     : {N_DEPTH}")
print(
    f"Depth range       : "
    f"{NATIVE_DEPTHS_M[0]:.6f} -> {NATIVE_DEPTHS_M[-1]:.6f} m"
)
print("All 156 GLORYS depth schemas: PASS")


# ============================================================================
# SIH 15 DEPTH MAPPING
# ============================================================================
#
# TRAINING:
#   The model predicts all 36 native GLORYS depths. No vertical interpolation
#   or depth reduction is used in the training target/loss.
#
# VALIDATION / TEST:
#   The historical OceanEmbed target uses the exact 15 SIH depths:
#       0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000 m
#
#   To make the new Native36 run comparable in metric space, predictions and
#   targets are mapped from the 36 native levels to those exact 15 SIH depths
#   using linear interpolation in PHYSICAL temperature space.
#
#   This interpolation is validation/test-only. It never enters the training
#   tensor, training loss, normalization fit, or target generation.
#
#   At 0 m, the nearest available native level is used because the first
#   native GLORYS level is ~0.494 m and linear interpolation cannot extrapolate.
# ============================================================================

sih_mapping: List[Dict[str, Any]] = []


for sih_depth in SIH_DEPTHS_M:

    depth_value = float(sih_depth)

    if depth_value <= float(NATIVE_DEPTHS_M[0]):

        lower_index = 0
        upper_index = 0
        weight = 0.0
        method = "clamped_first_native"

    elif depth_value >= float(NATIVE_DEPTHS_M[-1]):

        lower_index = N_DEPTH - 1
        upper_index = N_DEPTH - 1
        weight = 0.0
        method = "clamped_last_native"

    else:

        upper_index = int(
            np.searchsorted(
                NATIVE_DEPTHS_M,
                depth_value,
                side="right",
            )
        )

        lower_index = upper_index - 1

        denominator = (
            NATIVE_DEPTHS_M[upper_index]
            - NATIVE_DEPTHS_M[lower_index]
        )

        if denominator <= 0.0:
            raise RuntimeError(
                f"Invalid native depth bracket for SIH depth {depth_value}"
            )

        weight = float(
            (
                depth_value
                - NATIVE_DEPTHS_M[lower_index]
            )
            / denominator
        )

        method = "linear_native_bracket"

    nearest_index = int(
        np.argmin(
            np.abs(
                NATIVE_DEPTHS_M
                - depth_value
            )
        )
    )

    nearest_depth = float(
        NATIVE_DEPTHS_M[
            nearest_index
        ]
    )

    sih_mapping.append(
        {
            "sih_depth_m": depth_value,
            "lower_native_index": int(lower_index),
            "lower_native_depth_m": float(
                NATIVE_DEPTHS_M[
                    lower_index
                ]
            ),
            "upper_native_index": int(upper_index),
            "upper_native_depth_m": float(
                NATIVE_DEPTHS_M[
                    upper_index
                ]
            ),
            "weight_upper": float(weight),
            "nearest_native_index": nearest_index,
            "nearest_native_depth_m": nearest_depth,
            "nearest_absolute_difference_m": abs(
                nearest_depth
                - depth_value
            ),
            "method": method,
        }
    )


# Kept for easy reference / checkpoint metadata.
sih_native_indices = [
    int(
        row["nearest_native_index"]
    )
    for row in sih_mapping
]


print_section(
    "SIH 15 DEPTH -> NATIVE 36 METRIC MAPPING"
)


for row in sih_mapping:

    if row["lower_native_index"] == row["upper_native_index"]:

        print(
            f"SIH {row['sih_depth_m']:7.2f} m"
            f" -> native[{row['lower_native_index']:02d}]"
            f" {row['lower_native_depth_m']:9.4f} m"
            f" | method={row['method']}"
        )

    else:

        print(
            f"SIH {row['sih_depth_m']:7.2f} m"
            f" -> native[{row['lower_native_index']:02d}]"
            f" {row['lower_native_depth_m']:9.4f} m"
            f" + native[{row['upper_native_index']:02d}]"
            f" {row['upper_native_depth_m']:9.4f} m"
            f" | w_upper={row['weight_upper']:.6f}"
            f" | method={row['method']}"
        )


def interpolate_native_field_to_sih(
    field_native: np.ndarray,
    valid_native: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Map [36,H,W] native physical fields to exact [15,H,W] SIH depths.

    Only VALID bracket cells are interpolated.
    For the 0 m boundary, the first native level is used.
    """

    field_native = np.asarray(
        field_native,
        dtype=np.float64,
    )

    valid_native = np.asarray(
        valid_native,
        dtype=bool,
    )

    if field_native.shape != (
        N_DEPTH,
        GRID_H,
        GRID_W,
    ):

        raise RuntimeError(
            f"field_native shape {field_native.shape} "
            f"!= {(N_DEPTH, GRID_H, GRID_W)}"
        )

    if valid_native.shape != (
        N_DEPTH,
        GRID_H,
        GRID_W,
    ):

        raise RuntimeError(
            f"valid_native shape {valid_native.shape} "
            f"!= {(N_DEPTH, GRID_H, GRID_W)}"
        )

    output = np.zeros(
        (
            len(SIH_DEPTHS_M),
            GRID_H,
            GRID_W,
        ),
        dtype=np.float64,
    )

    output_mask = np.zeros(
        (
            len(SIH_DEPTHS_M),
            GRID_H,
            GRID_W,
        ),
        dtype=bool,
    )

    for sih_index, row in enumerate(
        sih_mapping
    ):

        lo = int(
            row["lower_native_index"]
        )

        hi = int(
            row["upper_native_index"]
        )

        weight_upper = float(
            row["weight_upper"]
        )

        lo_valid = valid_native[
            lo
        ]

        hi_valid = valid_native[
            hi
        ]

        valid = (
            lo_valid
            & hi_valid
            & np.isfinite(
                field_native[
                    lo
                ]
            )
            & np.isfinite(
                field_native[
                    hi
                ]
            )
        )

        if lo == hi:

            output[
                sih_index
            ][valid] = field_native[
                lo
            ][valid]

        else:

            output[
                sih_index
            ][valid] = (
                (
                    1.0
                    - weight_upper
                )
                * field_native[
                    lo
                ][valid]
                + weight_upper
                * field_native[
                    hi
                ][valid]
            )

        output_mask[
            sih_index
        ] = valid

    return output, output_mask

# PHYSICAL DEPTH MASK
# ============================================================================

physical_depth_mask_np = np.stack(
    [
        (
            ocean_mask_np
            & np.isfinite(depth_max_raw)
            & (depth_max_raw >= float(depth))
        )
        for depth in NATIVE_DEPTHS_M
    ],
    axis=0,
).astype(bool)

if physical_depth_mask_np.shape != (
    N_DEPTH,
    GRID_H,
    GRID_W,
):
    raise RuntimeError(
        f"Physical-depth mask shape "
        f"{physical_depth_mask_np.shape}"
    )


# ============================================================================
# TRAIN-ONLY 36-DEPTH TARGET NORMALIZATION
# ============================================================================
#
# Sum/sumsq/count are accumulated for each:
#   month x native depth x grid cell
#
# This gives:
#   monthly climatology = sum / count
#   anomaly variance = sumsq - sum^2/count
#
# over ALL target dates in 2012-2020, while respecting physical depth support.
# ============================================================================

print_section("FITTING TRAIN-ONLY 36-DEPTH TARGET NORMALIZATION")

CLIM_SHAPE = (
    12,
    N_DEPTH,
    GRID_H,
    GRID_W,
)

train_sum = np.zeros(
    CLIM_SHAPE,
    dtype=np.float64,
)

train_sumsq = np.zeros(
    CLIM_SHAPE,
    dtype=np.float64,
)

train_count = np.zeros(
    CLIM_SHAPE,
    dtype=np.uint32,
)

train_month_paths = []

for year in range(
    2012,
    2021,
):

    for month in range(
        1,
        13,
    ):

        key = f"{year:04d}-{month:02d}"

        train_month_paths.append(
            (
                key,
                GLORYS_BY_MONTH[key],
            )
        )


norm_timer = time.time()

for file_idx, (
    key,
    path,
) in enumerate(
    train_month_paths,
    start=1,
):

    ds = normalize_spatial(
        xr.open_dataset(path)
    )

    thetao = np.asarray(
        ds["thetao"].load().values,
        dtype=np.float64,
    )

    ds.close()

    if thetao.shape[1:] != (
        N_DEPTH,
        GRID_H,
        GRID_W,
    ):
        raise RuntimeError(
            f"{path.name}: thetao shape {thetao.shape}"
        )

    month_idx = int(key[5:7]) - 1

    for depth_idx in range(
        N_DEPTH
    ):

        physical = physical_depth_mask_np[
            depth_idx
        ]

        data = thetao[
            :,
            depth_idx,
            :,
            :,
        ]

        valid = (
            np.isfinite(data)
            & physical[None, :, :]
        )

        safe = np.where(
            valid,
            data,
            0.0,
        )

        train_count[
            month_idx,
            depth_idx,
        ] += valid.sum(
            axis=0
        ).astype(
            np.uint32
        )

        train_sum[
            month_idx,
            depth_idx,
        ] += safe.sum(
            axis=0,
            dtype=np.float64,
        )

        train_sumsq[
            month_idx,
            depth_idx,
        ] += np.square(
            safe
        ).sum(
            axis=0,
            dtype=np.float64,
        )

    if (
        file_idx == 1
        or file_idx % 12 == 0
        or file_idx == len(train_month_paths)
    ):
        print(
            f"  processed "
            f"{file_idx:3d}/108 train months"
            f" | elapsed={time.time() - norm_timer:.1f}s",
            flush=True,
        )


target_climatology_np = np.zeros(
    CLIM_SHAPE,
    dtype=np.float32,
)

positive = train_count > 0

target_climatology_np[
    positive
] = (
    train_sum[
        positive
    ]
    / train_count[
        positive
    ]
).astype(
    np.float32
)


# Residual variance around each monthly climatology.
residual_ss_by_depth = np.zeros(
    N_DEPTH,
    dtype=np.float64,
)

residual_count_by_depth = np.zeros(
    N_DEPTH,
    dtype=np.int64,
)


for month_idx in range(
    12
):

    for depth_idx in range(
        N_DEPTH
    ):

        count = train_count[
            month_idx,
            depth_idx,
        ].astype(
            np.float64
        )

        valid = count > 0

        if not np.any(
            valid
        ):
            continue

        sum_values = train_sum[
            month_idx,
            depth_idx,
        ][valid]

        sumsq_values = train_sumsq[
            month_idx,
            depth_idx,
        ][valid]

        n = count[
            valid
        ]


        # Sum (x - mean)^2 = sum(x^2) - sum(x)^2/n.
        ss = (
            sumsq_values
            - np.square(
                sum_values
            )
            / n
        )

        ss = np.maximum(
            ss,
            0.0,
        )

        residual_ss_by_depth[
            depth_idx
        ] += float(
            ss.sum(
                dtype=np.float64
            )
        )

        residual_count_by_depth[
            depth_idx
        ] += int(
            n.sum()
        )


target_std_np = np.ones(
    N_DEPTH,
    dtype=np.float32,
)


for depth_idx in range(
    N_DEPTH
):

    if (
        residual_count_by_depth[
            depth_idx
        ] > 0
    ):

        variance = (
            residual_ss_by_depth[
                depth_idx
            ]
            / residual_count_by_depth[
                depth_idx
            ]
        )

        if (
            np.isfinite(variance)
            and variance > 1e-12
        ):

            target_std_np[
                depth_idx
            ] = np.float32(
                math.sqrt(
                    variance
                )
            )


if not np.isfinite(
    target_climatology_np[
        physical_depth_mask_np[None, :, :, :]
        .repeat(12, axis=0)
    ]
).all():
    raise RuntimeError(
        "Non-finite target climatology at physically supported cells"
    )


print()
print("Target mode : anomaly")
print("Climatology : train-only monthly climatology")
print(
    f"Normalization dates: "
    f"{TRAIN_START} -> {TRAIN_END}"
)
print()
print("Per-depth target anomaly std:")

for idx, (
    depth,
    std,
) in enumerate(
    zip(
        NATIVE_DEPTHS_M,
        target_std_np,
    )
):

    print(
        f"  depth[{idx:02d}] "
        f"{depth:9.3f} m : "
        f"std={std:.6f} °C"
    )


# ============================================================================
# SPLITS
# ============================================================================

def canonical_date_list(
    start_date: str,
    end_date: str,
) -> List[str]:

    start = np.datetime64(
        start_date,
        "D",
    )

    end = np.datetime64(
        end_date,
        "D",
    )

    n = int(
        (
            end
            - start
        )
        / np.timedelta64(
            1,
            "D",
        )
    ) + 1

    return [
        date_string(
            start
            + np.timedelta64(
                i,
                "D",
            )
        )
        for i in range(
            n
        )
    ]


def usable_dates(
    start_date: str,
    end_date: str,
) -> List[str]:

    dates = canonical_date_list(
        start_date,
        end_date,
    )

    # 14-day window inclusive of target date:
    # t-13, ..., t
    return dates[
        WINDOW_DAYS - 1:
    ]


ALL_TRAIN_DATES = canonical_date_list(
    TRAIN_START,
    TRAIN_END,
)

ALL_VAL_DATES = canonical_date_list(
    VAL_START,
    VAL_END,
)

ALL_TEST_DATES = canonical_date_list(
    TEST_START,
    TEST_END,
)

TRAIN_DATES = usable_dates(
    TRAIN_START,
    TRAIN_END,
)

VAL_DATES = usable_dates(
    VAL_START,
    VAL_END,
)

TEST_DATES = usable_dates(
    TEST_START,
    TEST_END,
)


if len(TRAIN_DATES) != 3275:
    raise RuntimeError(
        f"Expected 3275 training samples, got {len(TRAIN_DATES)}"
    )

if len(VAL_DATES) != 717:
    raise RuntimeError(
        f"Expected 717 validation samples, got {len(VAL_DATES)}"
    )

if len(TEST_DATES) != 718:
    raise RuntimeError(
        f"Expected 718 test samples, got {len(TEST_DATES)}"
    )


print_section("TEMPORAL SPLIT")

print(
    f"Train normalization dates : {len(ALL_TRAIN_DATES)} "
    f"({ALL_TRAIN_DATES[0]} -> {ALL_TRAIN_DATES[-1]})"
)

print(
    f"Train model samples       : {len(TRAIN_DATES)} "
    f"({TRAIN_DATES[0]} -> {TRAIN_DATES[-1]})"
)

print(
    f"Validation samples        : {len(VAL_DATES)} "
    f"({VAL_DATES[0]} -> {VAL_DATES[-1]})"
)

print(
    f"Test samples              : {len(TEST_DATES)} "
    f"({TEST_DATES[0]} -> {TEST_DATES[-1]})"
)

print(
    "Boundary policy           : input_within_split"
)


# ============================================================================
# GLORYS TARGET ACCESS
# ============================================================================

GLORYS_TIME_INDEX_CACHE: Dict[
    str,
    Dict[str, int],
] = {}


def get_glorys_time_index(
    path: Path,
) -> Dict[str, int]:

    key = str(
        path
    )

    if key in GLORYS_TIME_INDEX_CACHE:
        return GLORYS_TIME_INDEX_CACHE[
            key
        ]

    ds = open_glorys(
        key
    )

    days = to_day_array(
        ds["time"].values
    )

    result = {}

    for idx, day in enumerate(
        days
    ):

        d = date_string(
            day
        )

        if d in result:
            raise RuntimeError(
                f"{path.name}: duplicate GLORYS date {d}"
            )

        result[d] = int(
            idx
        )

    GLORYS_TIME_INDEX_CACHE[
        key
    ] = result

    return result


def find_glorys_file(
    day: str,
) -> Path:

    path = GLORYS_BY_MONTH.get(
        month_key(
            day
        )
    )

    if path is None:
        raise RuntimeError(
            f"No GLORYS monthly file for {month_key(day)}"
        )

    return path


@lru_cache(
    maxsize=TARGET_DAY_CACHE_SIZE
)
def load_glorys_target_raw(
    day: str,
) -> np.ndarray:

    path = find_glorys_file(
        day
    )

    ds = open_glorys(
        str(path)
    )

    time_index = get_glorys_time_index(
        path
    )

    idx = time_index.get(
        day
    )

    if idx is None:
        raise RuntimeError(
            f"{path.name} does not contain exact day {day}"
        )

    target = np.asarray(
        ds["thetao"]
        .isel(
            time=idx
        )
        .load()
        .values,
        dtype=np.float32,
    )

    if target.shape != (
        N_DEPTH,
        GRID_H,
        GRID_W,
    ):
        raise RuntimeError(
            f"{path.name}: target shape {target.shape}"
        )

    return target


@lru_cache(
    maxsize=TARGET_DAY_CACHE_SIZE
)
def load_target_normalized(
    day: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    raw = load_glorys_target_raw(
        day
    )

    month_idx = (
        month_number(day)
        - 1
    )

    climatology = target_climatology_np[
        month_idx
    ]

    anomaly = (
        raw
        - climatology
    )

    normalized = (
        anomaly
        / target_std_np[
            :,
            None,
            None,
        ]
    ).astype(
        np.float32
    )

    target_valid = np.isfinite(
        raw
    )

    loss_mask = (
        target_valid
        & physical_depth_mask_np
    )

    normalized[
        ~loss_mask
    ] = 0.0

    if not np.isfinite(
        normalized
    ).all():
        raise RuntimeError(
            f"Non-finite normalized target for {day}"
        )

    return (
        normalized,
        loss_mask,
    )


# ============================================================================
# DYNAMIC ACCESS
# ============================================================================

@lru_cache(
    maxsize=DYNAMIC_DAY_CACHE_SIZE
)
def load_dynamic_day(
    day: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    blocks = []
    masks = []

    for (
        ds,
        variables,
        date_index,
    ) in zip(
        SOURCE_DATASETS,
        SOURCE_VARIABLE_GROUPS,
        SOURCE_DATE_INDEXES,
    ):

        idx = date_index.get(
            day
        )

        if idx is None:
            raise RuntimeError(
                f"Missing exact predictor date {day}"
            )

        source_ds = ds[variables].isel(time=idx).load()

        source = np.stack(
            [
                np.asarray(
                    source_ds[v].values,
                    dtype=np.float32,
                )
                for v in variables
            ],
            axis=0,
        )

        if source.shape != (
            len(variables),
            GRID_H,
            GRID_W,
        ):
            raise RuntimeError(
                f"{day}: source dynamic block shape {source.shape}"
            )

        blocks.append(
            source
        )

        masks.append(
            np.isfinite(
                source
            )
        )


    raw = np.concatenate(
        blocks,
        axis=0,
    )

    valid = np.concatenate(
        masks,
        axis=0,
    )


    if raw.shape != (
        DYNAMIC_CHANNELS,
        GRID_H,
        GRID_W,
    ):
        raise RuntimeError(
            f"{day}: dynamic shape {raw.shape}"
        )


    normalized = (
        (
            raw.astype(
                np.float64
            )
            - DYNAMIC_MEAN[
                :,
                None,
                None,
            ]
        )
        / DYNAMIC_STD[
            :,
            None,
            None,
        ]
    ).astype(
        np.float32
    )


    normalized[
        ~valid
    ] = 0.0


    valid = (
        valid
        & ocean_mask_np[
            None,
            :,
            :
        ]
    )


    return (
        normalized,
        valid,
    )


# ============================================================================
# TEMPORAL CONTEXT
# ============================================================================

def temporal_context(
    day: str,
) -> np.ndarray:

    d = date_string(
        day
    )

    year = int(
        d[:4]
    )

    month = int(
        d[5:7]
    )

    dom = int(
        d[8:10]
    )

    doy = (
        sum(
            monthrange(
                year,
                m,
            )[1]
            for m in range(
                1,
                month,
            )
        )
        + dom
    )

    leap = (
        year % 4 == 0
        and (
            year % 100 != 0
            or year % 400 == 0
        )
    )

    days_in_year = (
        366.0
        if leap
        else 365.0
    )

    phase = (
        2.0
        * math.pi
        * (
            doy
            - 1
        )
        / days_in_year
    )

    return np.array(
        [
            math.sin(phase),
            math.cos(phase),
        ],
        dtype=np.float32,
    )


# ============================================================================
# MODEL BATCH
# ============================================================================

def build_batch(
    target_day: str,
) -> Dict[str, torch.Tensor]:

    target_np = np.datetime64(
        target_day,
        "D",
    )

    start_np = (
        target_np
        - np.timedelta64(
            WINDOW_DAYS - 1,
            "D",
        )
    )

    input_days = [
        date_string(
            start_np
            + np.timedelta64(
                i,
                "D",
            )
        )
        for i in range(
            WINDOW_DAYS
        )
    ]

    dynamic = []
    dynamic_valid = []

    for day in input_days:

        x, m = load_dynamic_day(
            day
        )

        dynamic.append(
            x
        )

        dynamic_valid.append(
            m
        )

    x_dynamic = np.stack(
        dynamic,
        axis=0,
    ).astype(
        np.float32
    )

    x_dynamic_valid = np.stack(
        dynamic_valid,
        axis=0,
    ).astype(
        bool
    )

    y, loss_mask = load_target_normalized(
        target_day
    )

    temporal = np.stack(
        [
            temporal_context(
                day
            )
            for day in input_days
        ],
        axis=0,
    ).astype(
        np.float32
    )

    batch = {
        "x_dynamic": torch.from_numpy(
            x_dynamic[
                None,
                ...
            ]
        ),
        "x_dynamic_valid": torch.from_numpy(
            x_dynamic_valid[
                None,
                ...
            ]
        ),
        "x_static": torch.from_numpy(
            static_norm_np[
                None,
                ...
            ]
        ),
        "x_static_valid": torch.from_numpy(
            static_valid_np[
                None,
                ...
            ]
        ),
        "context_spatial": torch.from_numpy(
            spatial_context_np[
                None,
                ...
            ]
        ),
        "context_temporal": torch.from_numpy(
            temporal[
                None,
                ...
            ]
        ),
        "ocean_mask": torch.from_numpy(
            ocean_mask_np[
                None,
                None,
                ...
            ].astype(np.float32)
        ),
        "y": torch.from_numpy(
            y[
                None,
                ...
            ]
        ),
        "loss_mask": torch.from_numpy(
            loss_mask[
                None,
                ...
            ]
        ),
    }

    return batch


# ============================================================================
# B6 CORE MODEL SUPPORT
# ============================================================================
#
# The next four definitions are the real B6 model implementation:
#   * derive_ocean_mask_from_static
#   * prepare_model_inputs
#   * PartialConv2d
#   * MaskedAvgPool2d
#   * MaskedConvLSTMCell
#
# They are copied from the established B6 implementation rather than replaced
# by the earlier SimpleCNN sanity architecture.
# ============================================================================

# ===== Extracted B6 core implementation =====
STATIC_CHANNELS = 6
OCEAN_FRACTION_STATIC_INDEX = 2
EXPECTED_WINDOW_DAYS = 14
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2
TOTAL_INPUT_CHANNELS = 36
GRID_H = 100
GRID_W = 240
DYNAMIC_CHANNELS = 24
N_DEPTH = 36
MODEL_ID = 'CONVLSTM_MASKED_B6_NATIVE36'
MODEL_VERSION = '6.0.0-native36'
MODEL_CONTRACT = 'B6NATIVE36V1'
EXPECTED_LAND_OCEAN_FRACTION_Z = -1.001882263507042
OCEAN_FRACTION_Z_TOL = 1e-10
B6_HIGHRES_CHANNELS = 32
B6_HIDDEN_CHANNELS = 32
B6_RECURRENT_KERNEL = 3
B6_RECURRENT_PADDING = 1

class ContractError(RuntimeError): pass
class IntegrationTestError(RuntimeError): pass

def contract_hash(contract):
    return hash_json(contract)

def parameter_summary(model):
    return {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

def validate_native36_contract(contract):
    expected = {
        'dynamic': ['B', 'T', 24, 100, 240],
        'static': ['B', 6, 100, 240],
        'target': ['B', 36, 100, 240],
    }
    if not isinstance(contract, dict):
        raise ContractError('Model contract must be a dictionary')
    if contract.get('dynamic', {}).get('shape') != expected['dynamic']:
        raise ContractError(f'Invalid dynamic contract: {contract.get("dynamic")}')
    if contract.get('static', {}).get('shape') != expected['static']:
        raise ContractError(f'Invalid static contract: {contract.get("static")}')
    if contract.get('target', {}).get('shape') != expected['target']:
        raise ContractError(f'Invalid 36-depth target contract: {contract.get("target")}')

def _to_bool_mask(
    mask: torch.Tensor,
    shape: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """Validate an existing mask tensor and return it as boolean."""
    if not isinstance(mask, torch.Tensor):
        raise ContractError(
            f"{name} must be a torch.Tensor, got {type(mask).__name__}"
        )
    if tuple(mask.shape) != tuple(shape):
        raise ContractError(
            f"{name} shape {tuple(mask.shape)} != {tuple(shape)}"
        )
    return mask.bool()


def _normalize_explicit_ocean_mask(
    explicit: torch.Tensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if explicit.ndim == 2:
        explicit = explicit.unsqueeze(0).unsqueeze(0)
    elif explicit.ndim == 3:
        explicit = explicit.unsqueeze(1)
    elif explicit.ndim == 4:
        pass
    else:
        raise ContractError("ocean_mask must be [H,W], [B,H,W], or [B,1,H,W]")
    expected = (batch_size, 1, GRID_H, GRID_W)
    if tuple(explicit.shape) != expected:
        raise ContractError(f"ocean_mask shape {tuple(explicit.shape)} != {expected}")
    return (explicit > 0.5).to(device=device, dtype=dtype)


def derive_ocean_mask_from_static(
    x_static: torch.Tensor,
    batch: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if x_static.ndim != 4 or tuple(x_static.shape[1:]) != (
        STATIC_CHANNELS, GRID_H, GRID_W
    ):
        raise ContractError(f"x_static shape {tuple(x_static.shape)} is incompatible with B6")

    batch_size = x_static.shape[0]
    explicit = batch.get("ocean_mask") if batch is not None else None
    if explicit is not None:
        return _normalize_explicit_ocean_mask(
            explicit, batch_size, x_static.dtype, x_static.device
        )

    z = x_static[:, OCEAN_FRACTION_STATIC_INDEX:OCEAN_FRACTION_STATIC_INDEX + 1]
    if not torch.isfinite(z).all():
        raise ContractError("Normalized ocean_fraction contains non-finite values")

    expected_land_z = torch.as_tensor(
        EXPECTED_LAND_OCEAN_FRACTION_Z, dtype=z.dtype, device=z.device
    )
    land = torch.isclose(z, expected_land_z, rtol=0.0, atol=OCEAN_FRACTION_Z_TOL)
    land_counts = land.sum(dim=(-2, -1))
    if torch.any(land_counts <= 0):
        bad = [int(i) for i, v in enumerate(land_counts[:, 0].detach().cpu().tolist()) if v <= 0]
        raise ContractError(
            "Structural ocean-mask derivation failed: no frozen raw-land "
            f"ocean_fraction reference detected in samples {bad}."
        )

    min_z = z.amin(dim=(-2, -1), keepdim=True)
    if not torch.isclose(min_z, expected_land_z, rtol=0.0, atol=OCEAN_FRACTION_Z_TOL).all():
        raise ContractError(
            "Structural ocean-mask derivation failed: normalized ocean_fraction "
            "minimum is inconsistent with raw ocean_fraction=0 land reference"
        )
    return (~land).to(dtype=x_static.dtype)


def prepare_model_inputs(
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    required = (
        "x_dynamic", "x_static", "x_dynamic_valid", "x_static_valid",
        "context_spatial", "context_temporal",
    )
    missing = [k for k in required if k not in batch]
    if missing:
        raise ContractError(f"B6 batch missing required keys: {missing}")

    xd = batch["x_dynamic"]
    xs = batch["x_static"]
    xdv = batch["x_dynamic_valid"]
    xsv = batch["x_static_valid"]

    if xd.ndim != 5 or tuple(xd.shape[2:]) != (DYNAMIC_CHANNELS, GRID_H, GRID_W):
        raise ContractError(f"x_dynamic shape {tuple(xd.shape)} != [B,T,24,100,240]")
    bsz, t = xd.shape[:2]
    if t != EXPECTED_WINDOW_DAYS:
        raise ContractError(f"x_dynamic time dimension {t} != {EXPECTED_WINDOW_DAYS}")
    if xs.ndim != 4 or tuple(xs.shape[1:]) != (STATIC_CHANNELS, GRID_H, GRID_W):
        raise ContractError(f"x_static shape {tuple(xs.shape)} invalid")

    _to_bool_mask(xdv, (bsz, EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W), "x_dynamic_valid")
    _to_bool_mask(xsv, (bsz, STATIC_CHANNELS, GRID_H, GRID_W), "x_static_valid")

    spatial = batch["context_spatial"]
    if spatial.ndim == 3:
        if tuple(spatial.shape) != (SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W):
            raise ContractError(f"context_spatial shape {tuple(spatial.shape)} invalid")
        spatial = spatial.unsqueeze(0).expand(bsz, -1, -1, -1)
    elif spatial.ndim == 4:
        if tuple(spatial.shape) != (bsz, SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W):
            raise ContractError(f"context_spatial shape {tuple(spatial.shape)} invalid")
    else:
        raise ContractError("context_spatial must be [4,H,W] or [B,4,H,W]")

    temporal = batch["context_temporal"]
    if temporal.ndim != 3 or tuple(temporal.shape) != (
        bsz, EXPECTED_WINDOW_DAYS, TEMPORAL_CONTEXT_CHANNELS
    ):
        raise ContractError(f"context_temporal shape {tuple(temporal.shape)} invalid")

    ocean_mask = derive_ocean_mask_from_static(xs, batch).bool()
    dynamic_valid = xdv.bool()
    static_valid = xsv.bool()

    # Replicate static and spatial context over the 14-day sequence. Temporal
    # context remains day-specific.
    static_seq = xs.unsqueeze(1).expand(-1, EXPECTED_WINDOW_DAYS, -1, -1, -1)
    spatial_seq = spatial.unsqueeze(1).expand(-1, EXPECTED_WINDOW_DAYS, -1, -1, -1)
    temporal_map = temporal.unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, -1, GRID_H, GRID_W
    )

    model_input = torch.cat(
        [xd, static_seq, spatial_seq, temporal_map], dim=2
    )

    static_valid_seq = static_valid.unsqueeze(1).expand(
        -1, EXPECTED_WINDOW_DAYS, -1, -1, -1
    )
    deterministic_valid = torch.ones(
        bsz, EXPECTED_WINDOW_DAYS,
        SPATIAL_CONTEXT_CHANNELS + TEMPORAL_CONTEXT_CHANNELS,
        GRID_H, GRID_W,
        dtype=torch.bool,
        device=xd.device,
    )
    input_mask = torch.cat(
        [dynamic_valid, static_valid_seq, deterministic_valid], dim=2
    )
    input_mask = input_mask & ocean_mask.unsqueeze(1)

    if model_input.shape[2] != TOTAL_INPUT_CHANNELS:
        raise ContractError(
            f"Model input width {model_input.shape[2]} != {TOTAL_INPUT_CHANNELS}"
        )

    diagnostics = {
        "structural_ocean_cells": int(ocean_mask.sum().detach().cpu().item()),
        "structural_ocean_fraction": float(ocean_mask.float().mean().detach().cpu().item()),
        "dynamic_valid_fraction": float(dynamic_valid.float().mean().detach().cpu().item()),
        "static_valid_fraction": float(static_valid.float().mean().detach().cpu().item()),
    }

    if "loss_mask" in batch:
        loss_mask = batch["loss_mask"].bool()
        if tuple(loss_mask.shape) != (bsz, N_DEPTH, GRID_H, GRID_W):
            raise ContractError(f"loss_mask shape {tuple(loss_mask.shape)} invalid")
        surface_loss = loss_mask[:, 0:1]
        outside_ocean = surface_loss & ~ocean_mask
        if torch.any(outside_ocean):
            n = int(outside_ocean.sum().detach().cpu().item())
            raise IntegrationTestError(
                f"Surface loss cells outside structural ocean mask: {n}"
            )
        diagnostics["surface_loss_cells"] = int(surface_loss.sum().detach().cpu().item())
        diagnostics["loss_ocean_coverage_check"] = True

    return model_input, input_mask, ocean_mask.to(dtype=model_input.dtype), diagnostics

class PartialConv2d(nn.Module):
    """Channel-aware partial convolution with strict zero-support handling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = "replicate",
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size not in (1, 3):
            raise ValueError("B6 PartialConv2d supports kernel size 1 or 3")
        if padding_mode != "replicate":
            raise ValueError("B6 uses replicate padding only")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
            nn.init.uniform_(self.bias, -bound, bound)

        self.register_buffer(
            "_ones_kernel",
            torch.ones(1, in_channels, kernel_size, kernel_size),
            persistent=False,
        )
        self.register_buffer(
            "_kernel_elements", torch.tensor(float(in_channels * kernel_size * kernel_size)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"PartialConv2d expects [B,C,H,W], got {x.shape}")
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.expand(-1, self.in_channels, -1, -1)
        if tuple(mask.shape) != tuple(x.shape):
            raise ValueError(
                f"PartialConv2d mask shape {tuple(mask.shape)} != input {tuple(x.shape)}"
            )

        mask = mask.to(dtype=x.dtype, device=x.device)
        x_masked = x * mask
        if self.padding:
            pad = self.padding
            x_pad = F.pad(x_masked, (pad, pad, pad, pad), mode=self.padding_mode)
            mask_pad = F.pad(mask, (pad, pad, pad, pad), mode=self.padding_mode)
        else:
            x_pad = x_masked
            mask_pad = mask

        numerator = F.conv2d(x_pad, self.weight, bias=None, stride=1, padding=0)
        valid_count = F.conv2d(
            mask_pad,
            self._ones_kernel.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=0,
        )
        valid_fraction = valid_count / self._kernel_elements.to(dtype=x.dtype, device=x.device)
        safe_fraction = valid_fraction.clamp_min(1e-6)
        out = numerator / safe_fraction
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        support_mask = (valid_count > 0).to(dtype=x.dtype)
        return out * support_mask, support_mask


class MaskedAvgPool2d(nn.Module):
    """2x2 average pooling that averages only valid support."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        mask = mask.to(dtype=x.dtype, device=x.device)
        pooled_sum = F.avg_pool2d(
            x * mask,
            self.kernel_size,
            self.stride,
            count_include_pad=False,
        )
        pooled_fraction = F.avg_pool2d(
            mask,
            self.kernel_size,
            self.stride,
            count_include_pad=False,
        )
        out = pooled_sum / pooled_fraction.clamp_min(1e-6)
        support = (pooled_fraction > 0).to(dtype=x.dtype)
        return out * support, support

class MaskedConvLSTMCell(nn.Module):
    """ConvLSTM cell whose gate convolution is a partial convolution."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("B6 requires 3x3 ConvLSTM gate convolutions")
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.gate_conv = PartialConv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=True,
        )
        # Positive forget bias encourages temporal memory at initialization.
        with torch.no_grad():
            bias = self.gate_conv.bias
            if bias is not None:
                bias.view(4, hidden_channels).zero_()
                bias.view(4, hidden_channels)[1].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
        state_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(
                x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1],
                dtype=x.dtype, device=x.device,
            )
        if c is None:
            c = torch.zeros_like(h)
        if state_mask.ndim != 4 or state_mask.shape[1] != 1:
            raise ValueError("state_mask must be [B,1,H,W]")

        h = h * state_mask
        c = c * state_mask
        combined = torch.cat([x, h], dim=1)
        if x_mask.ndim != 4:
            raise ValueError("x_mask must be [B,C,H,W] or [B,1,H,W]")
        if x_mask.shape[1] == 1:
            x_mask = x_mask.expand(-1, self.input_channels, -1, -1)
        if tuple(x_mask.shape) != tuple(x.shape):
            raise ValueError(
                f"ConvLSTM input mask shape {tuple(x_mask.shape)} != input {tuple(x.shape)}"
            )
        hidden_mask = state_mask.expand(-1, self.hidden_channels, -1, -1)
        combined_mask = torch.cat([x_mask, hidden_mask], dim=1)

        gates, _ = self.gate_conv(combined, combined_mask)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = (f * c + i * g) * state_mask
        h_next = (o * torch.tanh(c_next)) * state_mask
        return h_next, c_next, state_mask

class Model(nn.Module):
    """OceanEmbed B6 mask-aware encoder + two-layer ConvLSTM + decoder."""

    model_id = MODEL_ID
    model_version = MODEL_VERSION
    model_contract = MODEL_CONTRACT

    def __init__(self, contract: Dict[str, Any]):
        super().__init__()
        validate_native36_contract(contract)
        self.contract = contract
        self.contract_hash = contract_hash(contract)
        self.window_days = EXPECTED_WINDOW_DAYS
        self.input_channels = TOTAL_INPUT_CHANNELS

        self.highres_encoder = PartialConv2d(
            TOTAL_INPUT_CHANNELS,
            B6_HIGHRES_CHANNELS,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=True,
        )
        self.encoder_activation = nn.GELU()
        self.pool = MaskedAvgPool2d(kernel_size=2, stride=2)

        self.convlstm1 = MaskedConvLSTMCell(
            B6_HIGHRES_CHANNELS,
            B6_HIDDEN_CHANNELS,
            kernel_size=B6_RECURRENT_KERNEL,
        )
        self.convlstm2 = MaskedConvLSTMCell(
            B6_HIDDEN_CHANNELS,
            B6_HIDDEN_CHANNELS,
            kernel_size=B6_RECURRENT_KERNEL,
        )

        self.decoder_projection = PartialConv2d(
            B6_HIDDEN_CHANNELS + B6_HIGHRES_CHANNELS,
            B6_HIGHRES_CHANNELS,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=True,
        )
        self.decoder_activation = nn.GELU()
        self.head = PartialConv2d(
            B6_HIGHRES_CHANNELS,
            N_DEPTH,
            kernel_size=1,
            padding=0,
            padding_mode="replicate",
            bias=True,
        )

    def encode_sequence(self, batch: Dict[str, Any], return_debug: bool = False):
        seq, seq_mask, ocean_mask_hr, _ = prepare_model_inputs(batch)
        bsz, timesteps, _, _, _ = seq.shape

        h1 = c1 = None
        h2 = c2 = None
        final_highres = None
        final_highres_mask = None
        lowres_ocean_mask = F.max_pool2d(
            ocean_mask_hr,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        lowres_ocean_mask = (lowres_ocean_mask > 0).to(dtype=seq.dtype)

        hidden1_states = []
        hidden2_states = []
        for t in range(timesteps):
            x_t = seq[:, t]
            mask_t = seq_mask[:, t].to(dtype=seq.dtype)
            highres, highres_mask = self.highres_encoder(x_t, mask_t)
            highres = self.encoder_activation(highres)
            highres = highres * highres_mask * ocean_mask_hr
            highres_mask = highres_mask * ocean_mask_hr

            pooled, pooled_mask = self.pool(highres, highres_mask)
            pooled = pooled * lowres_ocean_mask
            pooled_mask = pooled_mask * lowres_ocean_mask

            h1, c1, state_mask1 = self.convlstm1(
                pooled,
                pooled_mask,
                h1,
                c1,
                lowres_ocean_mask,
            )
            h2, c2, state_mask2 = self.convlstm2(
                h1,
                state_mask1,
                h2,
                c2,
                lowres_ocean_mask,
            )

            h1 = h1 * lowres_ocean_mask
            c1 = c1 * lowres_ocean_mask
            h2 = h2 * lowres_ocean_mask
            c2 = c2 * lowres_ocean_mask

            if return_debug:
                hidden1_states.append(h1)
                hidden2_states.append(h2)

            if t == timesteps - 1:
                final_highres = highres
                final_highres_mask = highres_mask

        if h2 is None or c2 is None or final_highres is None or final_highres_mask is None:
            raise IntegrationTestError("B6 temporal loop produced no final state")

        recurrent_up = F.interpolate(
            h2,
            size=(GRID_H, GRID_W),
            mode="bilinear",
            align_corners=False,
        )
        recurrent_mask_hr = F.interpolate(
            lowres_ocean_mask,
            size=(GRID_H, GRID_W),
            mode="nearest",
        ) * ocean_mask_hr
        recurrent_up = recurrent_up * recurrent_mask_hr

        fused = torch.cat([recurrent_up, final_highres], dim=1)
        fused_mask = torch.cat(
            [
                recurrent_mask_hr.expand(-1, B6_HIDDEN_CHANNELS, -1, -1),
                final_highres_mask.expand(-1, B6_HIGHRES_CHANNELS, -1, -1),
            ],
            dim=1,
        )

        if return_debug:
            return fused, fused_mask, ocean_mask_hr, {
                "final_hidden_layer1": h1,
                "final_cell_layer1": c1,
                "final_hidden_layer2": h2,
                "final_cell_layer2": c2,
                "lowres_ocean_mask": lowres_ocean_mask,
                "hidden1_states": hidden1_states,
                "hidden2_states": hidden2_states,
                "sequence_length": timesteps,
            }
        return fused, fused_mask, ocean_mask_hr

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        fused, fused_mask, ocean_mask_hr = self.encode_sequence(batch, return_debug=False)
        x, mask = self.decoder_projection(fused, fused_mask)
        x = self.decoder_activation(x)
        x = x * mask * ocean_mask_hr
        y, output_mask = self.head(x, mask)
        y = y * output_mask * ocean_mask_hr

        expected = (fused.shape[0], N_DEPTH, GRID_H, GRID_W)
        if tuple(y.shape) != expected:
            raise ContractError(f"B6 output shape {tuple(y.shape)} != {expected}")
        return y

    def model_summary(self) -> Dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_contract": MODEL_CONTRACT,
            "parameters": parameter_summary(self),
            "architecture": {
                "input_per_day_channels": TOTAL_INPUT_CHANNELS,
                "sequence_length": EXPECTED_WINDOW_DAYS,
                "highres_encoder": "masked 3x3 36->32 @ 100x240",
                "pooling": "mask-aware 2x2 stride 2 -> 50x120",
                "convlstm_layers": 2,
                "convlstm_hidden_channels": 32,
                "convlstm_kernel": [3, 3],
                "recurrent_hidden_shape": ["B", 32, 50, 120],
                "temporal_order": "t-13 ... t",
                "bidirectional": False,
                "attention": False,
                "decoder": "2x bilinear upsample + final-day high-resolution fusion + masked 3x3",
                "head": "masked 1x1 -> 36 native depths",
                "output_shape": ["B", N_DEPTH, GRID_H, GRID_W],
                "hidden_state_is_spatial": True,
            },
        }


def build_model(contract: Dict[str, Any]) -> Model:
    return Model(contract)


# ============================================================================
# LOSS / CONTRACT
# ============================================================================

def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    if pred.shape != target.shape or pred.shape != mask.shape:
        raise RuntimeError(
            f"Loss shape mismatch: "
            f"pred={tuple(pred.shape)}, "
            f"target={tuple(target.shape)}, "
            f"mask={tuple(mask.shape)}"
        )

    valid = mask.bool()

    if not torch.any(valid):
        raise RuntimeError("Masked MSE received zero valid target cells")

    return (
        (pred - target).pow(2)[valid].mean()
    )


def build_native36_contract() -> Dict[str, Any]:

    return {
        "dynamic": {
            "shape": [
                "B",
                "T",
                DYNAMIC_CHANNELS,
                GRID_H,
                GRID_W,
            ],
            "features": DYNAMIC_VARS,
        },
        "static": {
            "shape": [
                "B",
                STATIC_CHANNELS,
                GRID_H,
                GRID_W,
            ],
            "features": STATIC_VARS,
        },
        "target": {
            "shape": [
                "B",
                N_DEPTH,
                GRID_H,
                GRID_W,
            ],
            "depths_m": [
                float(x)
                for x in NATIVE_DEPTHS_M
            ],
            "mode": "anomaly",
            "train_loss": "all_36_native",
            "validation_metric": "SIH_15_only",
        },
        "context": {
            "spatial": [
                "B",
                SPATIAL_CHANNELS,
                GRID_H,
                GRID_W,
            ],
            "temporal": [
                "B",
                WINDOW_DAYS,
                TEMPORAL_CHANNELS,
            ],
        },
    }


# ============================================================================
# METRICS
# ============================================================================

class SIHMetricAccumulator:

    def __init__(self) -> None:

        self.records: List[Dict[str, float]] = []

        for _ in SIH_DEPTHS_M:

            self.records.append(
                {
                    "n": 0.0,
                    "abs_err": 0.0,
                    "sq_err": 0.0,
                    "err": 0.0,
                    "sum_p": 0.0,
                    "sum_t": 0.0,
                    "sum_p2": 0.0,
                    "sum_t2": 0.0,
                    "sum_pt": 0.0,
                    "base_sq": 0.0,
                    "base_abs": 0.0,
                    "acc_sum": 0.0,
                    "acc_days": 0.0,
                    "val_norm_sq": 0.0,
                }
            )


    @staticmethod
    def _pearson(
        rec: Dict[str, float],
    ) -> float:

        n = rec["n"]

        if n < 2:
            return float("nan")

        numerator = (
            n
            * rec["sum_pt"]
            - rec["sum_p"]
            * rec["sum_t"]
        )

        denominator_p = (
            n
            * rec["sum_p2"]
            - rec["sum_p"]
            * rec["sum_p"]
        )

        denominator_t = (
            n
            * rec["sum_t2"]
            - rec["sum_t"]
            * rec["sum_t"]
        )

        denominator = (
            math.sqrt(
                max(
                    denominator_p,
                    0.0,
                )
            )
            * math.sqrt(
                max(
                    denominator_t,
                    0.0,
                )
            )
        )

        if denominator <= 1e-15:
            return float("nan")

        return float(
            numerator
            / denominator
        )


    def update(
        self,
        day: str,
        pred_norm: np.ndarray,
        target_norm: np.ndarray,
        native_mask: np.ndarray,
    ) -> None:

        if pred_norm.shape != (
            N_DEPTH,
            GRID_H,
            GRID_W,
        ):

            raise RuntimeError(
                f"pred_norm shape {pred_norm.shape} "
                f"!= {(N_DEPTH, GRID_H, GRID_W)}"
            )

        if target_norm.shape != (
            N_DEPTH,
            GRID_H,
            GRID_W,
        ):

            raise RuntimeError(
                f"target_norm shape {target_norm.shape} "
                f"!= {(N_DEPTH, GRID_H, GRID_W)}"
            )

        if native_mask.shape != (
            N_DEPTH,
            GRID_H,
            GRID_W,
        ):

            raise RuntimeError(
                f"native_mask shape {native_mask.shape} "
                f"!= {(N_DEPTH, GRID_H, GRID_W)}"
            )


        month_idx = (
            month_number(day)
            - 1
        )


        # --------------------------------------------------------
        # Convert normalized Native36 outputs back to physical °C.
        # --------------------------------------------------------

        climatology_native = (
            target_climatology_np[
                month_idx
            ]
        )

        pred_physical_native = (
            pred_norm
            * target_std_np[
                :,
                None,
                None,
            ]
            + climatology_native
        )

        target_physical_native = (
            target_norm
            * target_std_np[
                :,
                None,
                None,
            ]
            + climatology_native
        )

        baseline_physical_native = (
            climatology_native
        )


        # --------------------------------------------------------
        # Exact SIH-depth mapping.
        # --------------------------------------------------------

        pred_sih, pred_sih_mask = (
            interpolate_native_field_to_sih(
                pred_physical_native,
                native_mask,
            )
        )

        target_sih, target_sih_mask = (
            interpolate_native_field_to_sih(
                target_physical_native,
                native_mask,
            )
        )

        baseline_sih, baseline_sih_mask = (
            interpolate_native_field_to_sih(
                baseline_physical_native,
                native_mask,
            )
        )


        # All metric fields must use the same target-valid support.
        metric_mask = (
            pred_sih_mask
            & target_sih_mask
            & baseline_sih_mask
        )


        # --------------------------------------------------------
        # Normalized metric-space comparison.
        #
        # We keep this in Native36 space for diagnostics because the
        # training loss itself is Native36. The reported physical metrics
        # below are always exact-SIH-depth metrics.
        # --------------------------------------------------------

        for sih_index in range(
            len(SIH_DEPTHS_M)
        ):

            m = metric_mask[
                sih_index
            ]

            if not np.any(m):
                continue


            p = pred_sih[
                sih_index
            ][m].astype(
                np.float64
            )

            t = target_sih[
                sih_index
            ][m].astype(
                np.float64
            )

            b = baseline_sih[
                sih_index
            ][m].astype(
                np.float64
            )


            error = (
                p - t
            )


            rec = self.records[
                sih_index
            ]


            n = float(
                p.size
            )


            rec["n"] += n

            rec["abs_err"] += float(
                np.abs(
                    error
                ).sum()
            )

            rec["sq_err"] += float(
                np.square(
                    error
                ).sum()
            )

            rec["err"] += float(
                error.sum()
            )

            rec["sum_p"] += float(
                p.sum()
            )

            rec["sum_t"] += float(
                t.sum()
            )

            rec["sum_p2"] += float(
                np.square(
                    p
                ).sum()
            )

            rec["sum_t2"] += float(
                np.square(
                    t
                ).sum()
            )

            rec["sum_pt"] += float(
                (
                    p * t
                ).sum()
            )


            baseline_error = (
                b - t
            )


            rec["base_sq"] += float(
                np.square(
                    baseline_error
                ).sum()
            )

            rec["base_abs"] += float(
                np.abs(
                    baseline_error
                ).sum()
            )


            # ----------------------------------------------------
            # Validation normalized MSE.
            #
            # Use physical SIH values mapped back into the local
            # SIH climatological anomaly units through the interpolated
            # anomaly and a representative interpolated std.
            #
            # For reporting, this is diagnostic only.
            # ----------------------------------------------------

            anomaly_prediction = (
                p - b
            )

            anomaly_target = (
                t - b
            )


            # Use a linearly interpolated target std between the same
            # native brackets. At 0m it is the first native std.
            row = sih_mapping[
                sih_index
            ]

            lo = int(
                row["lower_native_index"]
            )

            hi = int(
                row["upper_native_index"]
            )

            w = float(
                row["weight_upper"]
            )

            sih_std = (
                (1.0 - w)
                * float(
                    target_std_np[
                        lo
                    ]
                )
                + w
                * float(
                    target_std_np[
                        hi
                    ]
                )
            )

            sih_std = max(
                sih_std,
                1e-6,
            )

            normalized_error = (
                anomaly_prediction
                - anomaly_target
            ) / sih_std

            rec["val_norm_sq"] += float(
                np.square(
                    normalized_error
                ).sum()
            )


            # ----------------------------------------------------
            # OceanEmbed ACC definition:
            #
            # spatial Pearson correlation of prediction and target
            # anomalies for each day/depth, averaged equally over
            # valid days.
            # ----------------------------------------------------

            if int(
                m.sum()
            ) >= 2:

                pred_anomaly = (
                    anomaly_prediction
                    - anomaly_prediction.mean()
                )

                true_anomaly = (
                    anomaly_target
                    - anomaly_target.mean()
                )

                denominator = (
                    math.sqrt(
                        float(
                            np.dot(
                                pred_anomaly,
                                pred_anomaly,
                            )
                        )
                    )
                    * math.sqrt(
                        float(
                            np.dot(
                                true_anomaly,
                                true_anomaly,
                            )
                        )
                    )
                )

                if denominator > 1e-15:

                    correlation = float(
                        np.dot(
                            pred_anomaly,
                            true_anomaly,
                        )
                        / denominator
                    )

                    if math.isfinite(
                        correlation
                    ):

                        rec["acc_sum"] += correlation
                        rec["acc_days"] += 1.0


    def finalize(self) -> Dict[str, Any]:

        rows = []

        total_n = 0.0
        total_sq = 0.0
        total_abs = 0.0
        total_err = 0.0
        total_base_sq = 0.0

        acc_values = []


        for sih_index, row in enumerate(
            sih_mapping
        ):

            rec = self.records[
                sih_index
            ]

            n = int(
                rec["n"]
            )


            if n > 0:

                rmse = math.sqrt(
                    rec["sq_err"]
                    / rec["n"]
                )

                mae = (
                    rec["abs_err"]
                    / rec["n"]
                )

                bias = (
                    rec["err"]
                    / rec["n"]
                )

                baseline_rmse = math.sqrt(
                    rec["base_sq"]
                    / rec["n"]
                )

                skill = (
                    1.0
                    - rmse
                    / baseline_rmse
                    if baseline_rmse > 0.0
                    else float("nan")
                )

                normalized_mse = (
                    rec["val_norm_sq"]
                    / rec["n"]
                )

            else:

                rmse = float("nan")
                mae = float("nan")
                bias = float("nan")
                baseline_rmse = float("nan")
                skill = float("nan")
                normalized_mse = float("nan")


            pearson = self._pearson(
                rec
            )


            if rec["acc_days"] > 0:

                acc = (
                    rec["acc_sum"]
                    / rec["acc_days"]
                )

            else:

                acc = float("nan")


            if math.isfinite(
                acc
            ):

                acc_values.append(
                    acc
                )


            rows.append(
                {
                    "depth_m": float(
                        SIH_DEPTHS_M[
                            sih_index
                        ]
                    ),
                    "n_valid_cells": n,
                    "RMSE_C": rmse,
                    "MAE_C": mae,
                    "Bias_C": bias,
                    "Pearson_r": pearson,
                    "ACC_spatial_daily_mean": acc,
                    "ACC_days": int(
                        rec["acc_days"]
                    ),
                    "baseline_climatology_RMSE_C": baseline_rmse,
                    "RMSE_skill": skill,
                    "validation_normalized_MSE": normalized_mse,
                    "native_lower_index": int(
                        row["lower_native_index"]
                    ),
                    "native_lower_depth_m": float(
                        row["lower_native_depth_m"]
                    ),
                    "native_upper_index": int(
                        row["upper_native_index"]
                    ),
                    "native_upper_depth_m": float(
                        row["upper_native_depth_m"]
                    ),
                    "native_upper_weight": float(
                        row["weight_upper"]
                    ),
                    "mapping_method": row["method"],
                }
            )


            total_n += rec["n"]
            total_sq += rec["sq_err"]
            total_abs += rec["abs_err"]
            total_err += rec["err"]
            total_base_sq += rec["base_sq"]


        if total_n > 0:

            overall_rmse = math.sqrt(
                total_sq
                / total_n
            )

            overall_mae = (
                total_abs
                / total_n
            )

            overall_bias = (
                total_err
                / total_n
            )

            baseline_rmse = math.sqrt(
                total_base_sq
                / total_n
            )

            skill = (
                1.0
                - overall_rmse
                / baseline_rmse
                if baseline_rmse > 0.0
                else float("nan")
            )

        else:

            overall_rmse = float("nan")
            overall_mae = float("nan")
            overall_bias = float("nan")
            baseline_rmse = float("nan")
            skill = float("nan")


        if acc_values:

            mean_acc = float(
                np.mean(
                    acc_values
                )
            )

        else:

            mean_acc = float("nan")


        return {
            "overall": {
                "RMSE_weighted_C": overall_rmse,
                "MAE_weighted_C": overall_mae,
                "Bias_weighted_C": overall_bias,
                "baseline_climatology_RMSE_C": baseline_rmse,
                "RMSE_skill": skill,
                "ACC_mean_over_SIH_depths": mean_acc,
                "n_valid_cells": int(
                    total_n
                ),
            },
            "by_depth": rows,
        }


# EVALUATION
# ============================================================================

def evaluate_sih(
    model: nn.Module,
    dates: Sequence[str],
    split_name: str,
) -> Dict[str, Any]:

    model.eval()

    metrics = SIHMetricAccumulator()

    start = time.time()

    with torch.no_grad():

        for position, day in enumerate(
            dates,
            start=1,
        ):

            sample = build_batch(
                day
            )


            device_batch = {
                key: value.to(
                    DEVICE,
                    non_blocking=True,
                )
                for key, value in sample.items()
            }


            if (
                DEVICE.type == "cuda"
                and AMP_ENABLED
            ):

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    prediction = model(
                        device_batch
                    )

            else:

                prediction = model(
                    device_batch
                )


            prediction_np = (
                prediction
                .float()
                .cpu()
                .numpy()[0]
            )


            target_np = (
                sample[
                    "y"
                ]
                .numpy()[0]
            )


            mask_np = (
                sample[
                    "loss_mask"
                ]
                .numpy()[0]
            )


            metrics.update(
                day,
                prediction_np,
                target_np,
                mask_np,
            )


            if (
                position == 1
                or position % 100 == 0
                or position == len(dates)
            ):

                elapsed = (
                    time.time()
                    - start
                )

                print(
                    f"  {split_name:5s} "
                    f"{position:4d}/{len(dates)}"
                    f" | elapsed={elapsed/60.0:.1f}m",
                    flush=True,
                )


    result = metrics.finalize()

    result["split"] = split_name
    result["sample_count"] = len(dates)
    result["date_start"] = dates[0]
    result["date_end"] = dates[-1]

    result["evaluation_depth_policy"] = {
        "requested_depths_m": [
            float(x)
            for x in SIH_DEPTHS_M
        ],
        "native_depth_count": N_DEPTH,
        "mapping": sih_mapping,
        "training_uses_all_native_depths": True,
        "evaluation_interpolation": (
            "linear in physical temperature space between native levels; "
            "0 m is clamped to first native level because native grid starts "
            "at approximately 0.494 m"
        ),
    }

    return result


# CHECKPOINTS
# ============================================================================

def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    global_step: int,
    best_val_rmse: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:

    return {
        "schema": "OCEANEMBED_B6_NATIVE36_CHECKPOINT_V1",
        "experiment_name": EXPERIMENT_NAME,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "model_contract_hash": hash_json(
            build_native36_contract()
        ),
        "pattern_version": PATTERN_VERSION,
        "target_depth_count": N_DEPTH,
        "native_depths_m": [
            float(x)
            for x in NATIVE_DEPTHS_M
        ],
        "sih_depths_m": [
            float(x)
            for x in SIH_DEPTHS_M
        ],
        "sih_native_indices": [
            int(x)
            for x in sih_native_indices
        ],
        "normalization_hash": hash_json(
            normalization_payload
        ),
        "training_policy": {
            "train_loss": "all_36_native",
            "validation": "SIH_15_only",
            "test": "SIH_15_only",
            "boundary_policy": "input_within_split",
            "target_mode": "anomaly",
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_rmse": float(best_val_rmse),
        "best_epoch": int(best_epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "history": history,
        "saved_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================================
# SAVE METRICS
# ============================================================================

def save_result(
    result: Dict[str, Any],
    output_dir: Path,
) -> None:

    split = str(
        result["split"]
    ).lower()

    atomic_json(
        output_dir
        / f"{split}_metrics.json",
        result,
    )


    csv_path = (
        output_dir
        / f"{split}_depth_metrics.csv"
    )


    fields = [
        "depth_m",
        "n_valid_cells",
        "RMSE_C",
        "MAE_C",
        "Bias_C",
        "Pearson_r",
        "ACC_spatial_daily_mean",
        "ACC_days",
        "baseline_climatology_RMSE_C",
        "RMSE_skill",
        "validation_normalized_MSE",
        "native_lower_index",
        "native_lower_depth_m",
        "native_upper_index",
        "native_upper_depth_m",
        "native_upper_weight",
        "mapping_method",
    ]


    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            result["by_depth"]
        )


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main() -> None:

    run_start = time.time()


    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    output_dir = (
        RUN_ROOT
        / f"{EXPERIMENT_NAME}_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------------
    # Dataset / experiment manifest
    # ------------------------------------------------------------------------

    global normalization_payload

    normalization_payload = {
        "version": "DLNORM_NATIVE36_V1.0",
        "target_mode": "anomaly",
        "target_depths_m": [
            float(x)
            for x in NATIVE_DEPTHS_M
        ],
        "target_std": [
            float(x)
            for x in target_std_np
        ],
        "train_normalization_period": [
            TRAIN_START,
            TRAIN_END,
        ],
        "train_normalization_date_count": len(
            ALL_TRAIN_DATES
        ),
        "physical_depth_policy": {
            "mode": "max_reach",
            "ocean_fraction_threshold": 0.0,
        },
        "sih_mapping": sih_mapping,
        "predictor_normalization_version": "DLNORMV1.0",
        "pattern_version": PATTERN_VERSION,
    }


    atomic_json(
        output_dir
        / "experiment_spec.json",
        {
            "status": "STARTED",
            "experiment_name": EXPERIMENT_NAME,
            "pattern_version": PATTERN_VERSION,
            "device": str(DEVICE),
            "architecture": {
                "highres": "masked 3x3 36->32 @ 100x240",
                "pool": "mask-aware 2x2 -> 50x120",
                "convlstm_layers": 2,
                "hidden_channels": 32,
                "recurrent_kernel": "3x3",
                "upsample": "bilinear 2x",
                "fusion": "recurrent low-res hidden + final-day high-res features",
                "decoder": "masked 3x3 64->32",
                "head": "masked 1x1 32->36 native depths",
            },
            "input_contract": {
                "dynamic": DYNAMIC_VARS,
                "static": STATIC_VARS,
                "spatial_channels": [
                    "sin_lat",
                    "cos_lat",
                    "sin_lon",
                    "cos_lon",
                ],
                "temporal_channels": [
                    "sin_annual_phase",
                    "cos_annual_phase",
                ],
                "window_days": WINDOW_DAYS,
                "total_channels_per_day": TOTAL_INPUT_CHANNELS,
            },
            "target_contract": {
                "count": N_DEPTH,
                "native_depths_m": [
                    float(x)
                    for x in NATIVE_DEPTHS_M
                ],
                "train_loss": "ALL_36_NATIVE",
                "validation": "SIH_15_EXACT_DEPTHS_ONLY_METRIC_MAPPING",
                "test": "SIH_15_EXACT_DEPTHS_ONLY_METRIC_MAPPING",
                "sih_depths_m": [
                    float(x)
                    for x in SIH_DEPTHS_M
                ],
                "sih_native_indices": [
                    int(x)
                    for x in sih_native_indices
                ],
            },
            "split_contract": {
                "boundary_policy": "input_within_split",
                "train": [
                    TRAIN_START,
                    TRAIN_END,
                ],
                "validation": [
                    VAL_START,
                    VAL_END,
                ],
                "test": [
                    TEST_START,
                    TEST_END,
                ],
                "train_model_samples": len(
                    TRAIN_DATES
                ),
                "validation_samples": len(
                    VAL_DATES
                ),
                "test_samples": len(
                    TEST_DATES
                ),
            },
            "normalization": normalization_payload,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRAD_CLIP_NORM,
                "amp": AMP_ENABLED,
            },
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


    atomic_json(
        output_dir
        / "predictor_normalization_native36.json",
        {
            "version": "DLNORMV1.0",
            "train_only": True,
            "training_range": [
                TRAIN_START,
                TRAIN_END,
            ],
            "dynamic_feature_names": DYNAMIC_VARS,
            "dynamic_mean": [
                float(x)
                for x in DYNAMIC_MEAN
            ],
            "dynamic_std": [
                float(x)
                for x in DYNAMIC_STD
            ],
            "static_feature_names": STATIC_VARS,
            "static_mean": [
                float(x)
                for x in STATIC_MEAN
            ],
            "static_std": [
                float(x)
                for x in STATIC_STD
            ],
        },
    )


    np.savez_compressed(
        output_dir
        / "native36_target_normalization_arrays.npz",
        target_climatology=target_climatology_np,
        target_std=target_std_np,
        native_depths_m=NATIVE_DEPTHS_M,
        physical_depth_mask=physical_depth_mask_np,
    )


    np.savez(
        output_dir
        / "sih_depth_mapping.npz",
        sih_depths_m=SIH_DEPTHS_M,
        native_depths_m=NATIVE_DEPTHS_M,
        sih_native_indices=np.asarray(
            sih_native_indices,
            dtype=np.int64,
        ),
        lower_native_indices=np.asarray(
            [
                int(row["lower_native_index"])
                for row in sih_mapping
            ],
            dtype=np.int64,
        ),
        upper_native_indices=np.asarray(
            [
                int(row["upper_native_index"])
                for row in sih_mapping
            ],
            dtype=np.int64,
        ),
        upper_weights=np.asarray(
            [
                float(row["weight_upper"])
                for row in sih_mapping
            ],
            dtype=np.float64,
        ),
    )


    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    contract = build_native36_contract()

    model = Model(
        contract
    ).to(
        DEVICE
    )


    parameter_summary_value = parameter_summary(
        model
    )


    print_section(
        "REAL B6-NATIVE36 MODEL"
    )

    print(
        f"Model ID          : {MODEL_ID}"
    )

    print(
        f"Model version     : {MODEL_VERSION}"
    )

    print(
        f"Trainable params  : "
        f"{parameter_summary_value['trainable_parameters']:,}"
    )

    print(
        f"Total params      : "
        f"{parameter_summary_value['total_parameters']:,}"
    )


    # ------------------------------------------------------------------------
    # REAL-DATA FORWARD PREFLIGHT
    # ------------------------------------------------------------------------

    print_section(
        "REAL-DATA FORWARD PREFLIGHT"
    )


    preflight_day = TRAIN_DATES[0]

    preflight_batch = build_batch(
        preflight_day
    )


    print(
        f"Target date       : {preflight_day}"
    )

    print(
        f"Input             : "
        f"{tuple(preflight_batch['x_dynamic'].shape)}"
    )

    print(
        f"Target            : "
        f"{tuple(preflight_batch['y'].shape)}"
    )

    print(
        f"Loss mask         : "
        f"{tuple(preflight_batch['loss_mask'].shape)}"
    )

    print(
        f"Valid target      : "
        f"{100.0 * preflight_batch['loss_mask'].float().mean().item():.2f}%"
    )


    with torch.no_grad():

        device_preflight = {
            k: v.to(
                DEVICE
            )
            for k, v in preflight_batch.items()
        }


        if (
            DEVICE.type == "cuda"
            and AMP_ENABLED
        ):

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                preflight_prediction = model(
                    device_preflight
                )

        else:

            preflight_prediction = model(
                device_preflight
            )


    expected_output = (
        1,
        N_DEPTH,
        GRID_H,
        GRID_W,
    )


    if tuple(
        preflight_prediction.shape
    ) != expected_output:

        raise RuntimeError(
            f"Real B6 output shape "
            f"{tuple(preflight_prediction.shape)} "
            f"!= {expected_output}"
        )


    if not torch.isfinite(
        preflight_prediction
    ).all():

        raise FloatingPointError(
            "Real B6 preflight produced non-finite predictions"
        )


    print(
        f"Forward output     : "
        f"{tuple(preflight_prediction.shape)}"
    )

    print(
        "Real B6 forward    : PASS"
    )


    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
            and AMP_ENABLED
        ),
    )


    # ------------------------------------------------------------------------
    # Resume state
    # ------------------------------------------------------------------------

    start_epoch = 1

    global_step = 0

    best_val_rmse = float("inf")

    best_epoch = 0

    history: List[
        Dict[str, Any]
    ] = []


    if RESUME_FROM:

        resume_path = Path(
            RESUME_FROM
        )


        print_section(
            "RESUME"
        )

        print(
            f"Checkpoint: {resume_path}"
        )


        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
        )


        if checkpoint.get(
            "experiment_name"
        ) != EXPERIMENT_NAME:

            raise RuntimeError(
                "Refusing resume from another experiment"
            )


        if int(
            checkpoint.get(
                "target_depth_count",
                -1,
            )
        ) != N_DEPTH:

            raise RuntimeError(
                "Refusing resume: checkpoint is not Native36"
            )


        if [
            float(x)
            for x in checkpoint.get(
                "native_depths_m",
                [],
            )
        ] != [
            float(x)
            for x in NATIVE_DEPTHS_M
        ]:

            raise RuntimeError(
                "Refusing resume: Native36 depth vector mismatch"
            )


        if checkpoint.get(
            "normalization_hash"
        ) != hash_json(
            normalization_payload
        ):

            raise RuntimeError(
                "Refusing resume: target normalization mismatch"
            )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )


        if checkpoint.get(
            "scaler_state_dict"
        ) is not None:

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )


        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )

        best_val_rmse = float(
            checkpoint.get(
                "best_val_rmse",
                float("inf"),
            )
        )

        best_epoch = int(
            checkpoint.get(
                "best_epoch",
                0,
            )
        )

        history = list(
            checkpoint.get(
                "history",
                [],
            )
        )


        print(
            f"Resuming at epoch {start_epoch}"
        )


    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    print_section(
        "TRAINING — ALL 36 NATIVE TARGET DEPTHS"
    )


    print(
        "Every eligible training date is used once per epoch."
    )

    print(
        "Training target depths: ALL 36 native GLORYS levels."
    )

    print(
        "Validation target depths: ONLY 15 SIH levels."
    )


    train_start_time = time.time()


    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):

        epoch_start = time.time()

        model.train()


        # Deterministic shuffle, identical date set every epoch.
        shuffled_dates = list(
            TRAIN_DATES
        )


        random.Random(
            SEED + epoch
        ).shuffle(
            shuffled_dates
        )


        running_loss = 0.0


        for step_in_epoch, day in enumerate(
            shuffled_dates,
            start=1,
        ):

            sample = build_batch(
                day
            )


            batch = {
                k: v.to(
                    DEVICE,
                    non_blocking=True,
                )
                for k, v in sample.items()
            }


            optimizer.zero_grad(
                set_to_none=True
            )


            if (
                DEVICE.type == "cuda"
                and AMP_ENABLED
            ):

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    prediction = model(
                        batch
                    )

                    loss = masked_mse_loss(
                        prediction,
                        batch["y"],
                        batch["loss_mask"],
                    )


                if not torch.isfinite(
                    loss
                ):

                    raise FloatingPointError(
                        f"Non-finite loss at "
                        f"epoch={epoch}, day={day}"
                    )


                scaler.scale(
                    loss
                ).backward()


                scaler.unscale_(
                    optimizer
                )


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )


                scaler.step(
                    optimizer
                )

                scaler.update()


            else:

                prediction = model(
                    batch
                )

                loss = masked_mse_loss(
                    prediction,
                    batch["y"],
                    batch["loss_mask"],
                )


                if not torch.isfinite(
                    loss
                ):

                    raise FloatingPointError(
                        f"Non-finite loss at "
                        f"epoch={epoch}, day={day}"
                    )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP_NORM,
                )


                optimizer.step()


            loss_value = float(
                loss.detach().item()
            )

            running_loss += loss_value

            global_step += 1


            if (
                step_in_epoch == 1
                or step_in_epoch % LOG_EVERY == 0
                or step_in_epoch == len(
                    shuffled_dates
                )
            ):

                print(
                    f"  epoch={epoch:02d}"
                    f" | step={step_in_epoch:04d}/{len(shuffled_dates)}"
                    f" | date={day}"
                    f" | loss_all36={loss_value:.6f}"
                    f" | elapsed="
                    f"{(time.time()-epoch_start)/60.0:.1f}m",
                    flush=True,
                )


        train_loss = (
            running_loss
            / len(
                shuffled_dates
            )
        )


        # --------------------------------------------------------------------
        # VALIDATION — ONLY SIH 15 DEPTHS
        # --------------------------------------------------------------------

        validation = evaluate_sih(
            model,
            VAL_DATES,
            "VAL",
        )


        val_rmse = float(
            validation[
                "overall"
            ][
                "RMSE_weighted_C"
            ]
        )


        val_mae = float(
            validation[
                "overall"
            ][
                "MAE_weighted_C"
            ]
        )


        val_skill = float(
            validation[
                "overall"
            ][
                "RMSE_skill"
            ]
        )


        val_acc = float(
            validation[
                "overall"
            ][
                "ACC_mean_over_SIH_depths"
            ]
        )


        if not math.isfinite(
            val_rmse
        ):

            raise FloatingPointError(
                f"Validation RMSE is non-finite at epoch {epoch}"
            )


        is_best = (
            val_rmse
            < best_val_rmse
        )


        if is_best:

            best_val_rmse = val_rmse
            best_epoch = epoch


        epoch_seconds = (
            time.time()
            - epoch_start
        )


        row = {
            "epoch": int(epoch),
            "train_loss_all36_normalized_MSE": float(
                train_loss
            ),
            "val_RMSE_SIH15_C": float(
                val_rmse
            ),
            "val_MAE_SIH15_C": float(
                val_mae
            ),
            "val_RMSE_skill_SIH15": val_skill,
            "val_ACC_mean_SIH15": val_acc,
            "best_val_RMSE_SIH15_C": float(
                best_val_rmse
            ),
            "best_epoch": int(
                best_epoch
            ),
            "global_step": int(
                global_step
            ),
            "epoch_seconds": float(
                epoch_seconds
            ),
            "completed_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }


        history.append(
            row
        )


        # --------------------------------------------------------------------
        # Save latest every epoch
        # --------------------------------------------------------------------

        latest_payload = checkpoint_payload(
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            best_val_rmse,
            best_epoch,
            history,
        )


        atomic_torch_save(
            output_dir
            / "latest_model.pt",
            latest_payload,
        )


        save_result(
            validation,
            output_dir,
        )


        atomic_json(
            output_dir
            / "training_history.json",
            history,
        )


        # --------------------------------------------------------------------
        # Save best
        # --------------------------------------------------------------------

        if is_best:

            best_payload = checkpoint_payload(
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_rmse,
                best_epoch,
                history,
            )


            atomic_torch_save(
                output_dir
                / "best_model.pt",
                best_payload,
            )


            print()
            print(
                f"[BEST] epoch={epoch} "
                f"| SIH-15 VAL RMSE="
                f"{best_val_rmse:.6f} °C"
            )


        total_hours = (
            time.time()
            - train_start_time
        ) / 3600.0


        print()
        print(
            f"EPOCH {epoch} COMPLETE"
            f" | train_loss_all36={train_loss:.6f}"
            f" | val_RMSE_SIH15={val_rmse:.6f} °C"
            f" | skill={val_skill:.6f}"
            f" | ACC={val_acc:.6f}"
            f" | best_epoch={best_epoch}"
            f" | best_RMSE={best_val_rmse:.6f} °C"
            f" | epoch_time={epoch_seconds/60.0:.1f} min"
            f" | runtime={total_hours:.2f} h",
            flush=True,
        )


        if total_hours >= SESSION_SOFT_LIMIT_HOURS:

            print()
            print(
                "SOFT SESSION LIMIT REACHED. "
                "Stopping after completed epoch."
            )

            break


    # ------------------------------------------------------------------------
    # Restore best checkpoint
    # ------------------------------------------------------------------------

    best_path = (
        output_dir
        / "best_model.pt"
    )


    if best_path.is_file():

        best_payload = torch.load(
            best_path,
            map_location="cpu",
        )


        if int(
            best_payload.get(
                "target_depth_count",
                -1,
            )
        ) != N_DEPTH:

            raise RuntimeError(
                "Best checkpoint is not Native36"
            )


        model.load_state_dict(
            best_payload[
                "model_state_dict"
            ]
        )


        best_epoch = int(
            best_payload[
                "epoch"
            ]
        )


        best_val_rmse = float(
            best_payload[
                "best_val_rmse"
            ]
        )


        print()
        print(
            f"Restored BEST checkpoint: "
            f"epoch={best_epoch}, "
            f"SIH-15 val RMSE={best_val_rmse:.6f} °C"
        )


    # ------------------------------------------------------------------------
    # FINAL TEST — SIH 15 ONLY
    # ------------------------------------------------------------------------

    test_result = evaluate_sih(
        model,
        TEST_DATES,
        "TEST",
    )


    save_result(
        test_result,
        output_dir,
    )


    # ------------------------------------------------------------------------
    # Final artifacts
    # ------------------------------------------------------------------------

    total_runtime_seconds = (
        time.time()
        - run_start
    )


    final_report = {
        "status": "COMPLETED",
        "experiment_name": EXPERIMENT_NAME,
        "pattern_version": PATTERN_VERSION,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "parameters": parameter_summary_value,
        "native_depth_count": N_DEPTH,
        "native_depths_m": [
            float(x)
            for x in NATIVE_DEPTHS_M
        ],
        "sih_depths_m": [
            float(x)
            for x in SIH_DEPTHS_M
        ],
        "sih_native_indices": [
            int(x)
            for x in sih_native_indices
        ],
        "sih_mapping": sih_mapping,
        "train_loss_policy": "ALL_36_NATIVE",
        "validation_policy": "SIH_15_EXACT_DEPTHS_ONLY",
        "test_policy": "SIH_15_EXACT_DEPTHS_ONLY",
        "target_mode": "anomaly",
        "boundary_policy": "input_within_split",
        "train_normalization_period": [
            TRAIN_START,
            TRAIN_END,
        ],
        "train_normalization_days": len(
            ALL_TRAIN_DATES
        ),
        "train_model_samples": len(
            TRAIN_DATES
        ),
        "validation_samples": len(
            VAL_DATES
        ),
        "test_samples": len(
            TEST_DATES
        ),
        "epochs_requested": EPOCHS,
        "epochs_completed": len(
            history
        ),
        "best_epoch": best_epoch,
        "best_validation_RMSE_SIH15_C": float(
            best_val_rmse
        ),
        "training_history": history,
        "test_result": test_result,
        "output_dir": str(
            output_dir
        ),
        "runtime_seconds": float(
            total_runtime_seconds
        ),
        "completed_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


    atomic_json(
        output_dir
        / "final_report.json",
        final_report,
    )


    # CSV epoch history.
    if history:

        history_path = (
            output_dir
            / "training_history.csv"
        )


        history_fields = list(
            history[0].keys()
        )


        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=history_fields,
            )

            writer.writeheader()

            writer.writerows(
                history
            )


    # Update experiment spec.
    atomic_json(
        output_dir
        / "experiment_spec.json",
        {
            **json.loads(
                (
                    output_dir
                    / "experiment_spec.json"
                ).read_text(
                    encoding="utf-8"
                )
            ),
            "status": "COMPLETED",
            "best_epoch": int(
                best_epoch
            ),
            "best_validation_RMSE_SIH15_C": float(
                best_val_rmse
            ),
            "completed_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
    )


    # ------------------------------------------------------------------------
    # Final console
    # ------------------------------------------------------------------------

    print_section(
        "B6-NATIVE36 REAL RUN COMPLETE"
    )


    print(
        f"Run directory          : {output_dir}"
    )

    print(
        f"Best epoch             : {best_epoch}"
    )

    print(
        f"Best VAL RMSE (15 SIH) : "
        f"{best_val_rmse:.6f} °C"
    )

    print()
    print(
        "FINAL TEST — ONLY 15 SIH DEPTHS"
    )

    print(
        f"Test RMSE              : "
        f"{float(test_result['overall']['RMSE_weighted_C']):.6f} °C"
    )

    print(
        f"Test MAE               : "
        f"{float(test_result['overall']['MAE_weighted_C']):.6f} °C"
    )

    print(
        f"Test Bias              : "
        f"{float(test_result['overall']['Bias_weighted_C']):.6f} °C"
    )

    print(
        f"Test Climatology RMSE  : "
        f"{float(test_result['overall']['baseline_climatology_RMSE_C']):.6f} °C"
    )

    print(
        f"Test RMSE Skill         : "
        f"{float(test_result['overall']['RMSE_skill']):.6f}"
    )

    pearson_values = [
        float(r["Pearson_r"])
        for r in test_result["by_depth"]
        if math.isfinite(float(r["Pearson_r"]))
    ]

    test_mean_pearson = (
        float(np.mean(pearson_values))
        if pearson_values
        else float("nan")
    )

    print(
        f"Test Mean Pearson       : {test_mean_pearson:.6f}"
    )

    print(
        f"Test Mean ACC           : "
        f"{float(test_result['overall']['ACC_mean_over_SIH_depths']):.6f}"
    )

    print()
    print(
        "TRAINING USED ALL 36 NATIVE DEPTHS."
    )

    print(
        "VALIDATION/TEST SCORED ONLY THE 15 SIH DEPTHS."
    )

    print(
        "NO VERTICAL INTERPOLATION WAS USED IN TRAINING; "
        "validation/test map the 36 native outputs to the 15 SIH metric depths."
    )

    print(
        "NO GLORYS/ARGO TARGET INFORMATION ENTERED THE INPUT."
    )

    print(
        f"Total runtime           : "
        f"{total_runtime_seconds / 3600.0:.2f} h"
    )


if __name__ == "__main__":
    main()
