#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanEmbed TS-Cast-style Native36 — 9x9 local context / trailing 31-day window

Architecture adaptation (not a verbatim reproduction of the paper):
  * shared fully-convolutional equivalent of a sliding 9x9 local patch encoder
  * 4 residual 3D convolution blocks; temporal-only pooling 31 -> 16 -> 8 -> 4 -> 2 -> 1
  * train-only monthly GLORYS temperature climatology as a depth-profile prior
  * FiLM-conditioned depth U-Net over all 36 native GLORYS levels
  * per-depth log-variance head trained with masked Gaussian NLL
  * DUACS ADT seasonal anomaly context substitutes for the paper's dynamic-height field
  * density-consistency loss is omitted because current labels contain temperature only

TRAIN:
  * eligible 31-day windows contained wholly in 2012-01-01 .. 2020-12-31
  * all 36 native GLORYS depths in the training loss
  * randomized, date-sampled training budget suitable for a Kaggle session

VALIDATION / TEST:
  * validation: 2021-01-01 .. 2022-12-31, evaluated at 15 SIH depths
  * test: 2023-01-01 .. 2024-12-31, retained untouched and never evaluated here
  * SIH metrics use linear interpolation in physical-temperature space

INPUTS / NORMALIZATION:
  * existing 24 dynamic + 6 static predictor contract and train-only statistics
  * trailing 31-day input window; no future predictor values
  * train-only monthly GLORYS climatology and train-only per-depth anomaly scales
  * canonical 100x240 cell-centred grid; native depth range ~0.494..1062.44 m

SAFETY / RESUMABILITY:
  * SMOKE_TEST mode first; FULL mode is a separate run directory
  * 11-hour soft-stop with a 15-minute reserve and atomic in-epoch checkpoints
  * FP16 overflow backoff, exact resume cursor, quick/full validation schedule; no test use
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch import nn


# Start the wall-clock budget before notebook-side data bootstrap/preflight.
SCRIPT_START_MONOTONIC = time.monotonic()


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
WINDOW_DAYS = 31

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

EPOCHS = 20  # overridden by MAX_EPOCHS below for the TS-Cast run
BATCH_SIZE = 1  # per-GPU sample count; FULL-mode global date batch is BATCH_DAYS below
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42
AMP_ENABLED = True

LOG_EVERY = 25
DYNAMIC_DAY_CACHE_SIZE = 64
GLORYS_OPEN_CACHE_SIZE = 2
TARGET_DAY_CACHE_SIZE = 4

RUN_ROOT = Path("/kaggle/working/OceanEmbed_Runs")
EXPERIMENT_NAME = "OCEANEMBED_TSCAST_NATIVE36_9X9_31DAY"
PATTERN_VERSION = "TSCAST_ADAPTED_NATIVE36_LOCAL9X9_31DAY_V3"

# Run twice: first SMOKE_TEST, then change to FULL after the smoke test passes.
RUN_MODE = "FULL"  # fixed release: resume the existing TS-Cast checkpoint when available

# TS-Cast is a separate architecture and must never load a B6/TAFU checkpoint.
RESUME_CHECKPOINT: Optional[str] = None  # set to attached /kaggle/input/.../latest_model.pt if auto-discovery cannot find it
REQUIRE_RESUME_CHECKPOINT = True  # avoid silently restarting from random initialization after a Kaggle version commit
RESUME_FROM: Optional[str] = None  # legacy loader helper compatibility; intentionally unused
FALLBACK_RESUME_FROM: Optional[str] = None  # legacy loader helper compatibility; intentionally unused
SESSION_SOFT_LIMIT_HOURS = 11.0
SESSION_TIME_LIMIT_SECONDS = SESSION_SOFT_LIMIT_HOURS * 3600.0

class SessionTimeLimitReached(RuntimeError):
    """Raised when the notebook should stop before Kaggle's hard cutoff."""


def session_elapsed_seconds() -> float:
    return time.monotonic() - SCRIPT_START_MONOTONIC


def session_time_limit_reached() -> bool:
    return session_elapsed_seconds() >= SESSION_TIME_LIMIT_SECONDS


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
    torch.backends.cudnn.benchmark = False
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
    """Durable checkpoint write: temp file, flush/fsync, then atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def hash_json(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


MIN_EXPECTED_RESUME_EPOCH = 1

def resolve_resume_checkpoint() -> Optional[Path]:
    """Resolve explicit checkpoint or discover the most advanced Native36 checkpoint.

    Checks both /kaggle/working and mounted /kaggle/input datasets. This is
    important when the newest checkpoint is uploaded as a Kaggle dataset with a
    generated slug that differs from earlier runs. Candidates are ranked by
    epoch, global step, then completion state; the loader later validates the
    exact depth and normalization contract before restoring weights.
    """
    if RESUME_FROM:
        explicit = Path(RESUME_FROM)
        if not explicit.is_file():
            raise FileNotFoundError(f"RESUME_FROM checkpoint not found: {explicit}")
        print(f"RESUME: explicit checkpoint: {explicit}", flush=True)
        return explicit

    roots = []
    if RUN_ROOT.exists():
        roots.append(RUN_ROOT)
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        roots.append(kaggle_input)

    candidate_paths: set[Path] = set()
    for root in roots:
        try:
            for pattern in ('latest_model*.pt', 'best_model*.pt'):
                for candidate in root.rglob(pattern):
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        candidate_paths.add(candidate)
        except OSError as exc:
            print(f"[RESUME SCAN WARNING] Could not scan {root}: {exc}", flush=True)

    ranked = []
    for candidate_path in candidate_paths:
        try:
            candidate = torch.load(candidate_path, map_location='cpu')
            if not isinstance(candidate, dict):
                continue
            if candidate.get('experiment_name') != EXPERIMENT_NAME:
                continue
            model_config = candidate.get('model_config', {}) or {}
            depth_count = candidate.get('target_depth_count', model_config.get('native_depth_count', -1))
            if int(depth_count) != N_DEPTH:
                continue
            if candidate.get('pattern_version') != PATTERN_VERSION:
                continue
            depth_vec = [float(v) for v in candidate.get('native_depths_m', [])]
            if depth_vec and depth_vec != [float(v) for v in NATIVE_DEPTHS_M]:
                continue
            epoch = int(candidate.get('epoch', 0))
            if epoch < MIN_EXPECTED_RESUME_EPOCH:
                continue
            global_step = int(candidate.get('global_step', 0))
            epoch_complete = bool(candidate.get('epoch_complete', True))
            mtime = candidate_path.stat().st_mtime
            ranked.append(((epoch, global_step, int(epoch_complete), mtime), candidate_path))
        except Exception as exc:
            print(f"[RESUME SCAN] Skipping {candidate_path.name}: {exc}", flush=True)

    if ranked:
        _, chosen = max(ranked, key=lambda item: item[0])
        print(f"AUTO-RESUME: selected most advanced checkpoint: {chosen}", flush=True)
        return chosen

    fallback = Path(FALLBACK_RESUME_FROM) if FALLBACK_RESUME_FROM else None
    if fallback and fallback.is_file():
        try:
            fallback_payload = torch.load(fallback, map_location='cpu')
            fallback_epoch = int(fallback_payload.get('epoch', 0))
            if (
                fallback_payload.get('experiment_name') == EXPERIMENT_NAME
                and int(fallback_payload.get('target_depth_count', (fallback_payload.get('model_config', {}) or {}).get('native_depth_count', -1))) == N_DEPTH
                and fallback_epoch >= MIN_EXPECTED_RESUME_EPOCH
            ):
                print(f"AUTO-RESUME: using validated fallback: {fallback}", flush=True)
                return fallback
        except Exception as exc:
            print(f"[RESUME WARNING] Could not validate fallback {fallback}: {exc}", flush=True)

    raise FileNotFoundError(
        "No compatible TS-Cast Native36 checkpoint was found in "
        "/kaggle/input or /kaggle/working. Attach the recovered latest_model.pt "
        "as a Kaggle input dataset or explicitly set RESUME_CHECKPOINT. Refusing an accidental fresh restart."
    )


# ============================================================================
# FILES
# ============================================================================

print_section("OCEANEMBED TS-CAST NATIVE36 9x9 / 31-DAY RUN")

print(f"DATA ROOT       : {DATA_ROOT}")
print(f"TARGET ROOT     : {TARGET_ROOT}")
print(f"DEVICE          : {DEVICE}")
print(f"INPUT           : [B,{WINDOW_DAYS},{TOTAL_INPUT_CHANNELS},{GRID_H},{GRID_W}]")
print(f"TARGET          : [B,{N_DEPTH},{GRID_H},{GRID_W}]")
print("TRAIN LOSS      : ALL 36 native GLORYS depths")
print("VALIDATION      : ONLY 15 SIH depths")
print("TEST            : ONLY 15 SIH depths")
print(f"SESSION BUDGET   : {SESSION_SOFT_LIMIT_HOURS:.1f} h from script start")
print(f"RUN MODE         : {RUN_MODE}")
print("CHECKPOINT POLICY: atomic periodic saves + epoch-end saves + safe-stop saves")

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
# OPTIONAL REUSE OF THE VERIFIED TRAIN-ONLY TARGET NORMALIZATION
# ============================================================================
# When the previous run's native36_target_normalization_arrays.npz is attached
# as a Kaggle input, reuse it instead of re-reading 108 monthly GLORYS files.
# The checkpoint's normalization_hash is verified after resume below. If no
# valid cache is attached, the original train-only fitting path runs unchanged.

_target_norm_cache_loaded = False
_target_norm_cache_roots = [Path("/kaggle/input"), RUN_ROOT]
_target_norm_candidates = []
for _root in _target_norm_cache_roots:
    if _root.exists():
        try:
            _target_norm_candidates.extend(_root.rglob("native36_target_normalization_arrays*.npz"))
        except OSError as _exc:
            print(f"[TARGET NORM CACHE] Search warning for {_root}: {_exc}", flush=True)

for _cache_path in sorted(set(_target_norm_candidates), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
    try:
        with np.load(_cache_path, allow_pickle=False) as _cache:
            _required = {"target_climatology", "target_std", "native_depths_m", "physical_depth_mask"}
            if not _required.issubset(set(_cache.files)):
                continue
            _clim = np.asarray(_cache["target_climatology"], dtype=np.float32)
            _std = np.asarray(_cache["target_std"], dtype=np.float32)
            _depths = np.asarray(_cache["native_depths_m"], dtype=np.float64)
            _mask = np.asarray(_cache["physical_depth_mask"], dtype=bool)
        if _clim.shape != (12, N_DEPTH, GRID_H, GRID_W):
            continue
        if _std.shape != (N_DEPTH,) or not np.isfinite(_std).all() or np.any(_std <= 0):
            continue
        if _depths.shape != (N_DEPTH,) or not np.allclose(_depths, NATIVE_DEPTHS_M, rtol=0.0, atol=1e-6):
            continue
        if _mask.shape != physical_depth_mask_np.shape or not np.array_equal(_mask, physical_depth_mask_np):
            continue
        if not np.isfinite(_clim[np.broadcast_to(physical_depth_mask_np[None, ...], _clim.shape)]).all():
            continue
        target_climatology_np = _clim
        target_std_np = _std
        _target_norm_cache_loaded = True
        print_section("REUSING VERIFIED TARGET NORMALIZATION CACHE")
        print(f"Cache             : {_cache_path}")
        print(f"Climatology shape : {target_climatology_np.shape}")
        print(f"Target std shape  : {target_std_np.shape}")
        print("Physical depth mask and native depth vector: PASS")
        break
    except Exception as _exc:
        print(f"[TARGET NORM CACHE] Skipping {_cache_path}: {_exc}", flush=True)

if not _target_norm_cache_loaded:
    print("[TARGET NORM CACHE] No compatible cache attached; recomputing train-only normalization.", flush=True)

# ============================================================================
# TRAIN-ONLY 36-DEPTH TARGET NORMALIZATION
# ============================================================================
if not _target_norm_cache_loaded:
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


# Persist train-only target statistics so SMOKE_TEST followed by FULL does not
# have to scan/recompute all 108 training GLORYS months twice.
_target_norm_cache_output = RUN_ROOT / "native36_target_normalization_arrays_tscast.npz"
_target_norm_cache_output.parent.mkdir(parents=True, exist_ok=True)
_target_norm_tmp = Path(str(_target_norm_cache_output) + ".tmp")
with _target_norm_tmp.open("wb") as _cache_handle:
    np.savez_compressed(
        _cache_handle,
        target_climatology=target_climatology_np,
        target_std=target_std_np,
        native_depths_m=NATIVE_DEPTHS_M,
        physical_depth_mask=physical_depth_mask_np,
    )
    _cache_handle.flush()
    os.fsync(_cache_handle.fileno())
os.replace(_target_norm_tmp, _target_norm_cache_output)
print(f"[TARGET NORM CACHE] Atomically saved: {_target_norm_cache_output}", flush=True)


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

    # WINDOW_DAYS-day trailing window, inclusive of the target date:
    # t-30, ..., t
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


expected_train_dates = len(ALL_TRAIN_DATES) - WINDOW_DAYS + 1
expected_val_dates = len(ALL_VAL_DATES) - WINDOW_DAYS + 1
expected_test_dates = len(ALL_TEST_DATES) - WINDOW_DAYS + 1
if len(TRAIN_DATES) != expected_train_dates:
    raise RuntimeError(f"Expected {expected_train_dates} training samples, got {len(TRAIN_DATES)}")
if len(VAL_DATES) != expected_val_dates:
    raise RuntimeError(f"Expected {expected_val_dates} validation samples, got {len(VAL_DATES)}")
if len(TEST_DATES) != expected_test_dates:
    raise RuntimeError(f"Expected {expected_test_dates} test samples, got {len(TEST_DATES)}")


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


# ============================================================================
# TS-CAST ADAPTATION: full-field, shared 9x9 sliding-patch encoder
# ============================================================================

RUN_MODE = RUN_MODE  # configured near the top: SMOKE_TEST or FULL

# Model/training controls. Full-field batches provide one profile per grid cell.
TRAIN_STEPS_PER_EPOCH = 320  # shorter future epochs to fit the Kaggle wall-clock budget
CHECKPOINT_EVERY_DATES = 32  # frequent recovery points; atomic replacement
MAX_EPOCHS = 20
BATCH_DAYS = max(1, min(2, torch.cuda.device_count()))  # uses both T4s via DataParallel when available
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 1.0
AMP_ENABLED = (DEVICE.type == "cuda")
AMP_INITIAL_SCALE = 4096.0  # lower than PyTorch default; reduces early FP16 overflow risk
MAX_CONSECUTIVE_BAD_GRAD_STEPS = 6
QUICK_VALIDATION_DATES = 48
FULL_VALIDATION_INTERVAL = 5
VAL_EVERY_EPOCH = True
SESSION_SOFT_LIMIT_HOURS = 11.0
CHECKPOINT_RESERVE_MINUTES = 15.0
LOG_EVERY_STEPS = 25
SEED = 42
PATCH_SIZE = 9
PATCH_RADIUS = PATCH_SIZE // 2
MODEL_CHANNELS = (32, 48, 64, 96)
LATENT_CHANNELS = 512
FILM_CHANNELS = 128
PROFILE_CHANNELS = (16, 32, 64)
LOGVAR_MIN = -5.0
LOGVAR_MAX = 3.0
PRIOR_TEMP_OFFSET_C = 15.0
PRIOR_TEMP_SCALE_C = 10.0

if PATCH_SIZE != 9 or PATCH_RADIUS != 4:
    raise RuntimeError("This implementation is explicitly configured for 9x9 context patches.")

if not torch.cuda.is_available():
    print("[WARNING] CUDA is unavailable. Full-field TS-Cast is computationally heavy on CPU.", flush=True)

# Output isolation: smoke tests never overwrite full-run outputs/checkpoints.
RUN_ROOT = Path("/kaggle/working/OceanEmbed_Runs")
EXPERIMENT_NAME = "OCEANEMBED_TSCAST_NATIVE36_9X9_31DAY"
PATTERN_VERSION = "TSCAST_ADAPTED_NATIVE36_LOCAL9X9_31DAY_V3"
if RUN_MODE not in {"SMOKE_TEST", "FULL"}:
    raise ValueError(f"Invalid RUN_MODE={RUN_MODE!r}; expected SMOKE_TEST or FULL")
OUTPUT_DIR = RUN_ROOT / (EXPERIMENT_NAME + "_SMOKE_TEST" if RUN_MODE == "SMOKE_TEST" else EXPERIMENT_NAME)

# Avoid cuDNN searching for aggressive convolution workspaces on a long-running
# T4 session. This trades a little peak speed for more predictable memory use.
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def should_soft_stop() -> bool:
    reserve_s = CHECKPOINT_RESERVE_MINUTES * 60.0
    safe_deadline = SESSION_SOFT_LIMIT_HOURS * 3600.0 - reserve_s
    return (time.monotonic() - SCRIPT_START_MONOTONIC) >= safe_deadline


def _coords_xyz(context_spatial: torch.Tensor) -> torch.Tensor:
    # context order: sin(lat), cos(lat), sin(lon), cos(lon)
    sin_lat = context_spatial[:, 0:1]
    cos_lat = context_spatial[:, 1:2]
    sin_lon = context_spatial[:, 2:3]
    cos_lon = context_spatial[:, 3:4]
    return torch.cat(
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        dim=1,
    )


# Compute the paper's monthly sea-level seasonal context from the available
# DUACS ADT product. TS-Cast uses dynamic-height anomalies; our input collection
# has no matching steric dynamic-height field, so this is explicitly an ADT-
# relative-to-monthly-climatology analogue, not an exact reproduction.
def build_adt_monthly_climatology() -> np.ndarray:
    print_section("TRAIN-ONLY MONTHLY DUACS ADT CONTEXT")
    cache_path = RUN_ROOT / "adt_monthly_climatology_normalized_trainonly.npz"
    try:
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                cached = np.asarray(cache["adt_monthly_climatology_normalized"], dtype=np.float32)
                train_start_cached = str(cache["train_start"].item())
                train_end_cached = str(cache["train_end"].item())
                source_size_cached = int(cache["duacs_file_size"].item())
                source_mtime_cached = int(cache["duacs_file_mtime_ns"].item())
            source_stat = DUACS_FILE.stat()
            cache_ok = (
                cached.shape == (12, GRID_H, GRID_W)
                and np.isfinite(cached).all()
                and train_start_cached == TRAIN_START
                and train_end_cached == TRAIN_END
                and source_size_cached == int(source_stat.st_size)
                and source_mtime_cached == int(source_stat.st_mtime_ns)
            )
            if cache_ok:
                cached[:, ~ocean_mask_np] = 0.0
                print(f"[ADT CONTEXT CACHE] Reused: {cache_path}", flush=True)
                return cached
            print("[ADT CONTEXT CACHE] Stale/incompatible; recomputing.", flush=True)
    except Exception as cache_exc:
        print(f"[ADT CONTEXT CACHE] Could not use cache ({cache_exc}); recomputing.", flush=True)
    if "time" not in duacs["duacs_adt"].dims:
        raise RuntimeError("DUACS duacs_adt must have a time dimension")
    adt_train = duacs["duacs_adt"].sel(time=slice(TRAIN_START, TRAIN_END))
    if int(adt_train.sizes.get("time", 0)) < 3000:
        raise RuntimeError(
            f"Expected roughly 9 years of DUACS training ADT, found {adt_train.sizes.get('time', 0)} dates"
        )
    monthly = adt_train.groupby("time.month").mean("time", skipna=True).load()
    values = np.asarray(monthly.values, dtype=np.float32)
    if values.shape != (12, GRID_H, GRID_W):
        raise RuntimeError(f"ADT monthly climatology shape {values.shape}; expected {(12, GRID_H, GRID_W)}")
    norm = (values - float(DYNAMIC_MEAN[1])) / max(float(DYNAMIC_STD[1]), 1e-8)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm[:, ~ocean_mask_np] = 0.0
    print(f"ADT climatology shape : {norm.shape}")
    print("Climatology period    : training split only (2012-01-01 to 2020-12-31)")
    source_stat = DUACS_FILE.stat()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_tmp = Path(str(cache_path) + ".tmp")
    with cache_tmp.open("wb") as cache_handle:
        np.savez_compressed(
            cache_handle,
            adt_monthly_climatology_normalized=norm,
            train_start=np.array(TRAIN_START),
            train_end=np.array(TRAIN_END),
            duacs_file_size=np.array(int(source_stat.st_size), dtype=np.int64),
            duacs_file_mtime_ns=np.array(int(source_stat.st_mtime_ns), dtype=np.int64),
        )
        cache_handle.flush()
        os.fsync(cache_handle.fileno())
    os.replace(cache_tmp, cache_path)
    print(f"[ADT CONTEXT CACHE] Atomically saved: {cache_path}", flush=True)
    return norm


ADT_MONTHLY_CLIM_NORM_NP = build_adt_monthly_climatology()


def build_one_day_batch(target_day: str) -> Dict[str, torch.Tensor]:
    target_np = np.datetime64(target_day, "D")
    start_np = target_np - np.timedelta64(WINDOW_DAYS - 1, "D")
    input_days = [date_string(start_np + np.timedelta64(i, "D")) for i in range(WINDOW_DAYS)]

    dyn_days = []
    mask_days = []
    for day in input_days:
        x, mask = load_dynamic_day(day)
        dyn_days.append(x)
        mask_days.append(mask)

    x_dynamic = np.stack(dyn_days, axis=0).astype(np.float32)  # [T,24,H,W]
    x_dynamic_valid = np.stack(mask_days, axis=0).astype(bool)
    y, loss_mask = load_target_normalized(target_day)
    month_idx = month_number(target_day) - 1
    profile_prior = target_climatology_np[month_idx].astype(np.float32, copy=False)

    sample = {
        "x_dynamic": torch.from_numpy(x_dynamic[None, ...]),
        "x_dynamic_valid": torch.from_numpy(x_dynamic_valid[None, ...]),
        "x_static": torch.from_numpy(static_norm_np[None, ...].copy()),
        "x_static_valid": torch.from_numpy(static_valid_np[None, ...].copy()),
        "context_spatial": torch.from_numpy(spatial_context_np[None, ...].copy()),
        "context_temporal": torch.from_numpy(temporal_context(target_day)[None, ...].copy()),
        "ocean_mask": torch.from_numpy(ocean_mask_np[None, None, ...].astype(np.float32)),
        "y": torch.from_numpy(y[None, ...]),
        "loss_mask": torch.from_numpy(loss_mask[None, ...]),
        "profile_prior": torch.from_numpy(profile_prior[None, ...].copy()),
        "adt_monthly_clim": torch.from_numpy(ADT_MONTHLY_CLIM_NORM_NP[None, ...].copy()),
    }
    # Fail early on malformed dates/features before they reach a GPU kernel.
    expected_x = (1, WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W)
    if tuple(sample["x_dynamic"].shape) != expected_x:
        raise RuntimeError(f"{target_day}: x_dynamic shape {tuple(sample['x_dynamic'].shape)} != {expected_x}")
    if not torch.isfinite(sample["x_dynamic"]).all():
        raise FloatingPointError(f"{target_day}: non-finite normalized predictors")
    if not torch.isfinite(sample["y"]).all():
        raise FloatingPointError(f"{target_day}: non-finite normalized target")
    if not bool(sample["loss_mask"].any()):
        raise RuntimeError(f"{target_day}: no valid target cells")
    return sample


def collate_day_batches(samples: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty day batch")
    keys = samples[0].keys()
    return {key: torch.cat([sample[key] for sample in samples], dim=0) for key in keys}


class ChannelLayerNorm3d(nn.Module):
    """LayerNorm over channels independently at each t/z/y/x location."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels independently at each y/x location."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ResidualSatelliteBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = ChannelLayerNorm3d(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.act(self.norm(self.conv(x)) + residual)
        # Temporal-only pooling preserves spatial resolution and the 9x9 spatial RF.
        if y.shape[2] > 1:
            # Right-replicate odd sequences before pooling so the newest day is never dropped.
            if y.shape[2] % 2 == 1:
                y = F.pad(y, (0, 0, 0, 0, 0, 1), mode="replicate")
            y = F.avg_pool3d(y, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        return y


class ResidualAuxBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = ChannelLayerNorm2d(channels)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)) + x)


class FilmDepthBlock(nn.Module):
    """Depth-only convolution with location-specific FiLM conditioning."""
    def __init__(self, in_channels: int, out_channels: int, film_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=(3, 1, 1),
            padding=(1, 0, 0), bias=False,
        )
        self.norm = ChannelLayerNorm3d(out_channels)
        self.film = nn.Conv2d(film_channels, 2 * out_channels, kernel_size=1)
        self.act = nn.Mish()
        # Start FiLM as identity so the climatology prior is not violently modulated.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.conv(x))
        gamma_beta = self.film(condition)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = 1.0 + gamma
        y = y * gamma.unsqueeze(2) + beta.unsqueeze(2)
        return self.act(y + self.skip(x))


class TSCastOceanEmbed(nn.Module):
    """TS-Cast-style local satellite encoder + climatology-conditioned depth U-Net.

    A shared fully-convolutional encoder evaluates every 9x9 sliding patch at once;
    its 4 spatial 3x3 convolutions give each grid centre a 9x9 receptive field.
    Outputs are normalized temperature anomalies and per-depth log-variance.
    """
    def __init__(self, native_depths_m: np.ndarray):
        super().__init__()
        self.satellite_blocks = nn.ModuleList([
            ResidualSatelliteBlock3d(51, MODEL_CHANNELS[0]),
            ResidualSatelliteBlock3d(MODEL_CHANNELS[0], MODEL_CHANNELS[1]),
            ResidualSatelliteBlock3d(MODEL_CHANNELS[1], MODEL_CHANNELS[2]),
            ResidualSatelliteBlock3d(MODEL_CHANNELS[2], MODEL_CHANNELS[3]),
        ])
        satellite_out = MODEL_CHANNELS[-1]

        # [31,12] ADT-minus-each-month-climatology context, flattened per cell.
        self.aux_in = nn.Conv2d(WINDOW_DAYS * 12, 128, kernel_size=1)
        self.aux_norm = ChannelLayerNorm2d(128)
        self.aux_act = nn.Mish()
        self.aux_blocks = nn.Sequential(ResidualAuxBlock2d(128), ResidualAuxBlock2d(128))
        self.static_in = nn.Conv2d(STATIC_CHANNELS * 2 + 3 + TEMPORAL_CHANNELS, 32, kernel_size=1)
        self.static_norm = ChannelLayerNorm2d(32)
        self.static_act = nn.Mish()
        self.latent_in = nn.Conv2d(satellite_out + 128 + 32, LATENT_CHANNELS, kernel_size=1)
        self.latent_act = nn.Mish()
        self.condition_reduce = nn.Conv2d(LATENT_CHANNELS, FILM_CHANNELS, kernel_size=1)
        self.condition_act = nn.Mish()

        c0, c1, c2 = PROFILE_CHANNELS
        self.enc0 = FilmDepthBlock(2, c0, FILM_CHANNELS)
        self.enc1 = FilmDepthBlock(c0, c1, FILM_CHANNELS)
        self.enc2 = FilmDepthBlock(c1, c2, FILM_CHANNELS)
        self.dec1 = FilmDepthBlock(c2 + c1, c1, FILM_CHANNELS)
        self.dec0 = FilmDepthBlock(c1 + c0, c0, FILM_CHANNELS)
        self.output_head = nn.Conv3d(c0, 2, kernel_size=1)

        depths = torch.as_tensor(np.asarray(native_depths_m, dtype=np.float32))
        depth_coordinate = torch.log1p(depths) / float(np.log1p(max(float(native_depths_m[-1]), 1.0)))
        self.register_buffer("depth_coordinate", depth_coordinate, persistent=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # AMP is thread-local. Enable it inside the replica forward as well as at
        # the training call site, so secondary DataParallel GPU threads also use FP16.
        dynamic = batch["x_dynamic"]
        use_amp = bool(AMP_ENABLED and dynamic.device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return self._forward_impl(batch)

    def _forward_impl(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_dynamic = batch["x_dynamic"]  # [B,T,24,H,W], already train-normalized
        x_valid = batch["x_dynamic_valid"].to(dtype=x_dynamic.dtype)
        b, t, c, h, w = x_dynamic.shape
        if t != WINDOW_DAYS or c != DYNAMIC_CHANNELS:
            raise RuntimeError(f"TS-Cast input contract mismatch: got {tuple(x_dynamic.shape)}")

        xyz = _coords_xyz(batch["context_spatial"]).to(dtype=x_dynamic.dtype)
        dynamic_c = x_dynamic.permute(0, 2, 1, 3, 4).contiguous()
        valid_c = x_valid.permute(0, 2, 1, 3, 4).contiguous()
        xyz_t = xyz.unsqueeze(2).expand(-1, -1, t, -1, -1)
        satellite = torch.cat([dynamic_c, valid_c, xyz_t], dim=1)  # 24+24+3 = 51 channels

        # Replicate halo lets each output cell see its own 9x9 neighbourhood,
        # including stable boundary handling without constructing patch tensors.
        satellite = F.pad(
            satellite,
            (PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, PATCH_RADIUS, 0, 0),
            mode="replicate",
        )
        for block in self.satellite_blocks:
            satellite = block(satellite)
        # Four pairwise pools produce T=2 from 31 input days; average the final two
        # temporal bins. Odd-length pooling above uses replicate padding, so day t remains represented.
        if satellite.shape[2] > 1:
            satellite = satellite.mean(dim=2, keepdim=True)
        if satellite.shape[2] != 1:
            raise RuntimeError(f"Temporal encoder did not collapse 31 days to 1: {satellite.shape[2]}")
        satellite = satellite[:, :, 0, PATCH_RADIUS:-PATCH_RADIUS, PATCH_RADIUS:-PATCH_RADIUS]
        if tuple(satellite.shape[-2:]) != (h, w):
            raise RuntimeError(f"Local patch encoder spatial shape {tuple(satellite.shape[-2:])} != {(h,w)}")

        adt_now = x_dynamic[:, :, 1, :, :]  # normalized DUACS ADT channel
        adt_monthly = batch["adt_monthly_clim"]
        # [B,31,12,H,W]: current daily ADT relative to each training-only monthly mean.
        adt_context = adt_now[:, :, None, :, :] - adt_monthly[:, None, :, :, :]
        adt_context = adt_context.reshape(b, WINDOW_DAYS * 12, h, w)
        aux = self.aux_blocks(self.aux_act(self.aux_norm(self.aux_in(adt_context))))

        temporal = batch["context_temporal"].to(dtype=x_dynamic.dtype)
        if temporal.shape != (b, TEMPORAL_CHANNELS):
            raise RuntimeError(f"context_temporal shape {tuple(temporal.shape)} != {(b, TEMPORAL_CHANNELS)}")
        temporal_maps = temporal[:, :, None, None].expand(-1, -1, h, w)
        static = torch.cat([
            batch["x_static"],
            batch["x_static_valid"].to(dtype=x_dynamic.dtype),
            xyz,
            temporal_maps,
        ], dim=1)
        static_features = self.static_act(self.static_norm(self.static_in(static)))
        latent = self.latent_act(self.latent_in(torch.cat([satellite, aux, static_features], dim=1)))
        condition = self.condition_act(self.condition_reduce(latent))

        prior = batch["profile_prior"].to(dtype=x_dynamic.dtype)
        prior_scaled = (prior - PRIOR_TEMP_OFFSET_C) / PRIOR_TEMP_SCALE_C
        depth = self.depth_coordinate.to(dtype=x_dynamic.dtype).view(1, 1, -1, 1, 1).expand(b, 1, -1, h, w)
        profile_input = torch.stack([prior_scaled, depth[:, 0]], dim=1)

        e0 = self.enc0(profile_input, condition)  # [B,c0,36,H,W]
        d1_in = F.avg_pool3d(e0, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        e1 = self.enc1(d1_in, condition)          # [B,c1,18,H,W]
        d2_in = F.avg_pool3d(e1, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        e2 = self.enc2(d2_in, condition)          # [B,c2,9,H,W]

        u1 = F.interpolate(e2, size=(e1.shape[2], h, w), mode="trilinear", align_corners=False)
        u1 = self.dec1(torch.cat([u1, e1], dim=1), condition)
        u0 = F.interpolate(u1, size=(e0.shape[2], h, w), mode="trilinear", align_corners=False)
        u0 = self.dec0(torch.cat([u0, e0], dim=1), condition)

        output = self.output_head(u0).float()
        mean = output[:, 0]  # normalized anomaly units, [B,36,H,W]
        logvar = output[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return {"mean": mean, "logvar": logvar}


def masked_gaussian_nll(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != target.shape or logvar.shape != target.shape or mask.shape != target.shape:
        raise RuntimeError(
            f"NLL shape mismatch: mean={tuple(mean.shape)}, logvar={tuple(logvar.shape)}, "
            f"target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )
    valid = mask.bool()
    if not bool(valid.any()):
        raise RuntimeError("Masked Gaussian NLL received zero valid cells")
    mu = mean.float()[valid]
    lv = logvar.float().clamp(LOGVAR_MIN, LOGVAR_MAX)[valid]
    y = target.float()[valid]
    if not (torch.isfinite(mu).all() and torch.isfinite(lv).all() and torch.isfinite(y).all()):
        raise FloatingPointError("Non-finite value passed to masked Gaussian NLL")
    squared = (mu - y).square()
    return 0.5 * (lv + squared * torch.exp(-lv)).mean()


# The established SIH metric accumulator is reused unchanged, so the evaluation
# target mapping remains linear interpolation in physical temperature space.

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





def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _safe_autocast():
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ENABLED)


def _uncertainty_summary_init() -> Dict[str, np.ndarray]:
    return {
        "n": np.zeros(N_DEPTH, dtype=np.float64),
        "sq_error_norm": np.zeros(N_DEPTH, dtype=np.float64),
        "nll_sum": np.zeros(N_DEPTH, dtype=np.float64),
        "cover_1sigma": np.zeros(N_DEPTH, dtype=np.float64),
        "cover_95": np.zeros(N_DEPTH, dtype=np.float64),
        "sigma_norm_sum": np.zeros(N_DEPTH, dtype=np.float64),
    }


def _uncertainty_summary_update(
    stats: Dict[str, np.ndarray],
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    m = mask.bool()
    mu = mean.float()
    lv = logvar.float().clamp(LOGVAR_MIN, LOGVAR_MAX)
    y = target.float()
    err = mu - y
    sigma = torch.exp(0.5 * lv)
    nll = 0.5 * (lv + err.square() * torch.exp(-lv))
    dims = (0, 2, 3)
    stats["n"] += m.sum(dim=dims).detach().cpu().numpy().astype(np.float64)
    stats["sq_error_norm"] += (err.square() * m).sum(dim=dims).detach().cpu().numpy().astype(np.float64)
    stats["nll_sum"] += (nll * m).sum(dim=dims).detach().cpu().numpy().astype(np.float64)
    stats["cover_1sigma"] += ((err.abs() <= sigma) & m).sum(dim=dims).detach().cpu().numpy().astype(np.float64)
    stats["cover_95"] += ((err.abs() <= 1.96 * sigma) & m).sum(dim=dims).detach().cpu().numpy().astype(np.float64)
    stats["sigma_norm_sum"] += (sigma * m).sum(dim=dims).detach().cpu().numpy().astype(np.float64)


def uncertainty_summary_finalize(stats: Dict[str, np.ndarray]) -> Dict[str, Any]:
    rows = []
    total_n = float(stats["n"].sum())
    total_nll = float(stats["nll_sum"].sum())
    total_sq = float(stats["sq_error_norm"].sum())
    total_cover1 = float(stats["cover_1sigma"].sum())
    total_cover95 = float(stats["cover_95"].sum())
    for i, depth in enumerate(NATIVE_DEPTHS_M):
        n = float(stats["n"][i])
        if n <= 0:
            continue
        std_c = float(target_std_np[i])
        rows.append({
            "native_depth_m": float(depth),
            "n_valid_cells": int(n),
            "normalized_anomaly_RMSE": float(math.sqrt(stats["sq_error_norm"][i] / n)),
            "mean_NLL_no_constant": float(stats["nll_sum"][i] / n),
            "one_sigma_coverage": float(stats["cover_1sigma"][i] / n),
            "95pct_interval_coverage": float(stats["cover_95"][i] / n),
            "mean_predicted_sigma_C": float(stats["sigma_norm_sum"][i] / n * std_c),
        })
    return {
        "scale": "normalized monthly anomaly; sigma converted to Celsius by the train-only per-depth anomaly std",
        "overall_mean_NLL_no_constant": float(total_nll / total_n) if total_n else None,
        "overall_one_sigma_coverage": float(total_cover1 / total_n) if total_n else None,
        "overall_95pct_interval_coverage": float(total_cover95 / total_n) if total_n else None,
        "overall_normalized_anomaly_RMSE": float(math.sqrt(total_sq / total_n)) if total_n else None,
        "by_native_depth": rows,
        "interpretation_note": "Coverage is a diagnostic, not guaranteed calibration; assess on independent profiles before operational confidence claims.",
    }


def uses_full_validation(epoch: int) -> bool:
    """Run full 700-date validation on epoch 1, each interval, and final epoch."""
    return epoch == 1 or epoch == MAX_EPOCHS or epoch % FULL_VALIDATION_INTERVAL == 0


def validation_dates_for_epoch(epoch: int) -> Tuple[List[str], bool]:
    """Use a fixed, evenly spaced validation subset between full validations."""
    if uses_full_validation(epoch):
        return list(VAL_DATES), True
    count = min(QUICK_VALIDATION_DATES, len(VAL_DATES))
    if count <= 0:
        raise RuntimeError("Validation date list is empty")
    indices = np.linspace(0, len(VAL_DATES) - 1, num=count, dtype=np.int64)
    indices = np.unique(indices)
    return [VAL_DATES[int(i)] for i in indices], False


def evaluate_sih_tscast(
    model: nn.Module,
    dates: Sequence[str],
    split_name: str,
    *,
    check_deadline: bool = True,
) -> Dict[str, Any]:
    model.eval()
    metrics = SIHMetricAccumulator()
    uncertainty = _uncertainty_summary_init()
    nll_total = 0.0
    valid_total = 0
    start = time.time()
    with torch.inference_mode():
        for start_index in range(0, len(dates), BATCH_DAYS):
            if check_deadline and should_soft_stop():
                raise SessionTimeLimitReached(
                    f"safe deadline reached during {split_name} evaluation at {start_index}/{len(dates)} dates"
                )
            batch_days = list(dates[start_index:start_index + BATCH_DAYS])
            sample_cpu = collate_day_batches([build_one_day_batch(day) for day in batch_days])
            sample = _move_batch(sample_cpu, DEVICE)
            with _safe_autocast():
                output = model(sample)
            mean = output["mean"].float()
            logvar = output["logvar"].float()
            _uncertainty_summary_update(uncertainty, mean, logvar, sample["y"], sample["loss_mask"])
            valid = sample["loss_mask"].bool()
            if bool(valid.any()):
                nll = 0.5 * (logvar.clamp(LOGVAR_MIN, LOGVAR_MAX) + (mean - sample["y"].float()).square() * torch.exp(-logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)))
                nll_total += float(nll[valid].sum().detach().cpu())
                valid_total += int(valid.sum().detach().cpu())
            mean_cpu = mean.detach().cpu().numpy()
            target_cpu = sample_cpu["y"].numpy()
            mask_cpu = sample_cpu["loss_mask"].numpy()
            for i, day in enumerate(batch_days):
                metrics.update(day, mean_cpu[i], target_cpu[i], mask_cpu[i])
            if start_index == 0 or (start_index + BATCH_DAYS) % 100 == 0 or start_index + len(batch_days) >= len(dates):
                elapsed = time.time() - start
                print(
                    f"  {split_name:5s} {min(start_index + len(batch_days), len(dates)):4d}/{len(dates)}"
                    f" | elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )
    result = metrics.finalize()
    result["split"] = split_name
    result["sample_count"] = len(dates)
    result["date_start"] = dates[0] if dates else None
    result["date_end"] = dates[-1] if dates else None
    result["training_objective_validation_NLL_no_constant"] = float(nll_total / valid_total) if valid_total else None
    result["uncertainty_calibration"] = uncertainty_summary_finalize(uncertainty)
    result["evaluation_depth_policy"] = {
        "requested_depths_m": [float(x) for x in SIH_DEPTHS_M],
        "native_depth_count": N_DEPTH,
        "mapping": sih_mapping,
        "training_uses_all_native_depths": True,
        "evaluation_interpolation": "linear in physical temperature space between native levels; 0 m clamped to first native level (~0.494 m)",
    }
    return result


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def safe_atomic_json(path: Path, payload: Any) -> None:
    atomic_json(path, _json_sanitize(payload))


def save_depth_csv(result: Dict[str, Any], path: Path) -> None:
    rows = list(result.get("by_depth", []))
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def save_checkpoint(
    path: Path,
    core_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    epoch: int,
    epoch_complete: bool,
    validation_complete: bool,
    next_date_index: int,
    epoch_dates: Sequence[str],
    global_step: int,
    best_val_rmse: float,
    best_epoch: int,
    history: Sequence[Dict[str, Any]],
    running_loss_sum: float,
    run_mode: str = "FULL",
    validation_result: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "schema": "OCEANEMBED_TSCAST_NATIVE36_CHECKPOINT_V1",
        "experiment_name": EXPERIMENT_NAME,
        "pattern_version": PATTERN_VERSION,
        "target_depth_count": N_DEPTH,
        "native_depths_m": [float(v) for v in NATIVE_DEPTHS_M],
        "model_config": {
            "architecture": "TS-Cast-style 3D residual local encoder + ADT seasonal context + FiLM depth U-Net",
            "patch_size": PATCH_SIZE,
            "window_days": WINDOW_DAYS,
            "native_depth_count": N_DEPTH,
            "latent_channels": LATENT_CHANNELS,
            "profile_channels": list(PROFILE_CHANNELS),
            "input_channels": DYNAMIC_VARS,
            "input_boundary_policy": "replicate pad 4 cells; four 3x3 spatial convolutions; every pixel gets a shared 9x9 receptive field",
            "target_mode": "train-only monthly climatology residual, normalized by per-depth anomaly std",
            "uncertainty": "heteroscedastic Gaussian NLL, log-variance in normalized anomaly units",
        },
        "epoch": int(epoch),
        "epoch_complete": bool(epoch_complete),
        "validation_complete": bool(validation_complete),
        "next_date_index": int(next_date_index),
        "epoch_dates": list(epoch_dates),
        "global_step": int(global_step),
        "best_validation_rmse_C": float(best_val_rmse) if math.isfinite(best_val_rmse) else None,
        "best_epoch": int(best_epoch),
        "model_state_dict": core_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": list(history),
        "running_loss_sum": float(running_loss_sum),
        "normalization": {
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "target_climatology": "monthly, train-only GLORYS 2012-2020",
            "target_anomaly_std_C": [float(x) for x in target_std_np],
            "adt_climatology": "monthly, train-only DUACS ADT 2012-2020",
            "sih_depths_m": [float(x) for x in SIH_DEPTHS_M],
        },
        "validation_result": validation_result,
        "saved_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_torch_save(path, payload)


def load_checkpoint(path: Path, core_model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Any) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema") != "OCEANEMBED_TSCAST_NATIVE36_CHECKPOINT_V1":
        raise RuntimeError(f"Checkpoint schema mismatch: {path}")
    if checkpoint.get("pattern_version") != PATTERN_VERSION:
        raise RuntimeError("Checkpoint architecture/contract mismatch; refusing an unsafe resume")
    core_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def run_smoke_test(core_model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Any) -> None:
    print_section("TS-CAST SMOKE TEST — REAL DATA + FORWARD/BACKWARD")
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_day = TRAIN_DATES[0]
    sample_cpu = build_one_day_batch(train_day)
    sample = _move_batch(sample_cpu, DEVICE)
    core_model.train()
    optimizer.zero_grad(set_to_none=True)
    with _safe_autocast():
        out = core_model(sample)
    mean = out["mean"].float()
    logvar = out["logvar"].float()
    loss = masked_gaussian_nll(mean, logvar, sample["y"], sample["loss_mask"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke test produced non-finite Gaussian NLL")
    if mean.shape != sample["y"].shape or logvar.shape != sample["y"].shape:
        raise RuntimeError(f"Smoke output mismatch: mean={tuple(mean.shape)}, y={tuple(sample['y'].shape)}")
    if not torch.isfinite(mean).all() or not torch.isfinite(logvar).all():
        raise FloatingPointError("Smoke test produced non-finite model outputs")
    if scaler is not None and AMP_ENABLED:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), GRAD_CLIP_NORM)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("Smoke test produced non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), GRAD_CLIP_NORM)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("Smoke test produced non-finite gradient norm")
        optimizer.step()

    # Use the same SIH metric path on a real validation date; don't touch test data.
    core_model.eval()
    val_day = VAL_DATES[0]
    val_cpu = build_one_day_batch(val_day)
    val_batch = _move_batch(val_cpu, DEVICE)
    with torch.inference_mode():
        with _safe_autocast():
            val_out = core_model(val_batch)
    metrics = SIHMetricAccumulator()
    metrics.update(
        val_day,
        val_out["mean"].float().cpu().numpy()[0],
        val_cpu["y"].numpy()[0],
        val_cpu["loss_mask"].numpy()[0],
    )
    val_metrics = metrics.finalize()
    smoke = {
        "status": "SMOKE_TEST_PASSED",
        "experiment_name": EXPERIMENT_NAME,
        "architecture": "TS-Cast-style local 9x9 encoder, trailing 31-day input, climatology-conditioned depth U-Net, log-variance head",
        "train_day": train_day,
        "validation_day": val_day,
        "input_shape": list(sample_cpu["x_dynamic"].shape),
        "prediction_shape": list(mean.shape),
        "loss_mask_valid_cells": int(sample_cpu["loss_mask"].sum().item()),
        "nll": float(loss.detach().float().cpu().item()),
        "gradient_norm_before_clip": float(torch.as_tensor(grad_norm).detach().float().cpu().item()),
        "one_day_validation_RMSE_C": float(val_metrics["overall"]["RMSE_weighted_C"]),
        "one_day_validation_MAE_C": float(val_metrics["overall"]["MAE_weighted_C"]),
        "elapsed_seconds": round(time.time() - start, 2),
        "cuda_peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
        "cuda_peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
        "test_period_used": False,
        "source_checkpoint_modified": False,
        "note": "Smoke weights/optimizer state are not used to seed FULL mode; this validates the real data/model/optimizer path only.",
    }
    safe_atomic_json(OUTPUT_DIR / "smoke_test_report.json", smoke)
    safe_atomic_json(OUTPUT_DIR / "experiment_spec.json", build_experiment_spec("SMOKE_TEST", smoke_report=smoke))
    print(f"Input shape         : {tuple(sample_cpu['x_dynamic'].shape)}")
    print(f"Prediction shape    : {tuple(mean.shape)}")
    print(f"Masked Gaussian NLL : {smoke['nll']:.6f}")
    print(f"Gradient norm       : {smoke['gradient_norm_before_clip']:.6f}")
    print(f"One-day SIH VAL RMSE : {smoke['one_day_validation_RMSE_C']:.6f} °C")
    print("FORWARD / BACKWARD / OPTIMIZER STEP : PASS")
    print("SIH METRIC PATH                    : PASS")
    print(f"Report saved: {OUTPUT_DIR / 'smoke_test_report.json'}")


def build_experiment_spec(status: str, smoke_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    spec = {
        "status": status,
        "experiment_name": EXPERIMENT_NAME,
        "pattern_version": PATTERN_VERSION,
        "architecture": {
            "based_on": "TS-Cast (Chae et al., Ocean Science, 2026), adapted to OceanEmbed GLORYS Native36 gridded labels",
            "spatial_patch": "9x9; shared fully-convolutional sliding-window implementation, 4 x 3x3 spatial Conv3D layers, replicate halo 4",
            "temporal_reduction": "31 -> 16 -> 8 -> 4 -> 2 via replicate-padded pairwise pooling, then average final two bins; newest day is retained",
            "temporal_window_days": WINDOW_DAYS,
            "input_boundary": "trailing past 31 days including target date; no future predictors",
            "dynamic_channels": DYNAMIC_VARS,
            "dynamic_validity_masks": True,
            "coordinate_encoding": "Earth-centred XYZ derived from latitude/longitude sine-cosine features",
            "temporal_encoding": "target-day sine/cosine annual phase supplied to the location-wise conditioning branch",
            "static_features": STATIC_VARS,
            "auxiliary_context": "31-day normalized DUACS ADT relative to train-only 12-month ADT climatology; proxy for paper's dynamic-height seasonal context, not identical",
            "profile_prior": "monthly train-only GLORYS temperature climatology",
            "profile_decoder": "depth-only 1D U-Net represented as Conv3D with spatial kernel 1x1; FiLM conditioned on local satellite latent vector",
            "latent_channels": LATENT_CHANNELS,
            "uncertainty_head": "depth-wise log variance, trained by masked Gaussian NLL",
            "omitted_paper_component": "density-constraint loss omitted because current supervised targets contain temperature only, not paired GLORYS salinity profiles",
        },
        "run": {
            "mode": status,
            "session_soft_limit_hours": SESSION_SOFT_LIMIT_HOURS,
            "checkpoint_reserve_minutes": CHECKPOINT_RESERVE_MINUTES,
            "max_epochs": MAX_EPOCHS,
            "sampled_train_dates_per_epoch": TRAIN_STEPS_PER_EPOCH,
            "checkpoint_every_dates": CHECKPOINT_EVERY_DATES,
            "quick_validation_dates": QUICK_VALIDATION_DATES,
            "full_validation_epochs": "epoch 1, every 5 epochs, and final epoch",
            "numerical_stability": "FP16 scaler starts at 4096, non-finite gradient batches are skipped with scaler backoff; 6 consecutive skips stop after checkpoint",
            "day_batch_size": BATCH_DAYS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "amp_enabled": AMP_ENABLED,
            "gpu_count": torch.cuda.device_count(),
            "device": str(DEVICE),
        },
        "splits": {
            "train": [TRAIN_START, TRAIN_END],
            "validation": [VAL_START, VAL_END],
            "test": [TEST_START, TEST_END],
            "train_dates_usable": len(TRAIN_DATES),
            "validation_dates_usable": len(VAL_DATES),
            "test_dates_usable_not_evaluated": len(TEST_DATES),
            "boundary_policy": "input window must remain inside the date split",
        },
        "target": {
            "source": "GLORYS daily Native36 thetao",
            "native_depths_m": [float(z) for z in NATIVE_DEPTHS_M],
            "training_target": "all 36 native levels; train-only monthly climatology anomaly / train-only per-depth anomaly std",
            "validation_metric_depths_m": [float(z) for z in SIH_DEPTHS_M],
            "test_policy": "held out; not evaluated during this run",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if smoke_report is not None:
        spec["smoke_test_report"] = smoke_report
    return spec


def main() -> None:
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print_section("OCEANEMBED TS-CAST NATIVE36 — 9x9 / 31-DAY RUN")
    print(f"Mode              : {RUN_MODE}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print(f"Device            : {DEVICE}")
    print(f"Visible CUDA GPUs : {torch.cuda.device_count()}")
    print(f"Date batch size   : {BATCH_DAYS}")
    print(f"Input window      : {WINDOW_DAYS} trailing days")
    print(f"Patch context     : {PATCH_SIZE}x{PATCH_SIZE} (convolutional equivalent)")
    print(f"Train subset/epoch: {TRAIN_STEPS_PER_EPOCH} target dates")
    print(f"Soft stop         : {SESSION_SOFT_LIMIT_HOURS:.1f}h total; training stops with {CHECKPOINT_RESERVE_MINUTES:.0f}m reserve")
    if DEVICE.type != "cuda":
        raise RuntimeError(
            "This TS-Cast experiment is designed for a Kaggle CUDA GPU. Enable a GPU accelerator "
            "before running SMOKE_TEST or FULL; CPU fallback is intentionally disabled to avoid an impractical run."
        )

    core_model = TSCastOceanEmbed(NATIVE_DEPTHS_M).to(DEVICE)
    n_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
    print(f"Trainable params  : {n_params:,}")

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED, init_scale=AMP_INITIAL_SCALE, growth_interval=1000)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED, init_scale=AMP_INITIAL_SCALE, growth_interval=1000)
    optimizer = torch.optim.AdamW(core_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if RUN_MODE == "SMOKE_TEST":
        # Validate memory/shape/backward on one full 100x240 date on a single GPU.
        # This specifically tests the per-GPU batch size that FULL mode will use.
        run_smoke_test(core_model, optimizer, scaler)
        return

    parallel = torch.cuda.device_count() >= 2
    model: nn.Module = nn.DataParallel(core_model, device_ids=list(range(torch.cuda.device_count()))) if parallel else core_model
    print(f"DataParallel      : {'ON' if parallel else 'OFF'}")

    latest_path = OUTPUT_DIR / "latest_model.pt"
    best_path = OUTPUT_DIR / "best_model.pt"
    if RESUME_CHECKPOINT:
        resume_path = Path(RESUME_CHECKPOINT)
    elif latest_path.is_file():
        resume_path = latest_path
    else:
        try:
            resume_path = resolve_resume_checkpoint()
        except FileNotFoundError:
            resume_path = None
    start_epoch = 1
    global_step = 0
    best_val_rmse = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    resume_epoch_dates: List[str] = []
    resume_next_date_index = 0
    running_loss_sum = 0.0
    validation_pending_epoch: Optional[int] = None

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Configured resume checkpoint does not exist: {resume_path}")
        checkpoint = load_checkpoint(resume_path, core_model, optimizer, scaler)
        # Keep AdamW moments, but apply the safer release learning rate after restore.
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        if scaler is not None and AMP_ENABLED:
            scaler_state = scaler.state_dict()
            if float(scaler_state.get("scale", AMP_INITIAL_SCALE)) > AMP_INITIAL_SCALE:
                scaler_state["scale"] = AMP_INITIAL_SCALE
                if "_growth_tracker" in scaler_state:
                    scaler_state["_growth_tracker"] = 0
                scaler.load_state_dict(scaler_state)
        print(f"[STABILITY] effective_lr={LEARNING_RATE:.2e}; AMP initial/resume scale={scaler.get_scale() if scaler is not None and AMP_ENABLED else 'off'}", flush=True)
        global_step = int(checkpoint.get("global_step", 0))
        best_val_rmse = float(checkpoint.get("best_validation_rmse_C") or float("inf"))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        checkpoint_epoch = int(checkpoint["epoch"])
        if bool(checkpoint.get("epoch_complete", False)):
            if not bool(checkpoint.get("validation_complete", False)):
                validation_pending_epoch = checkpoint_epoch
                start_epoch = checkpoint_epoch
            else:
                start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = checkpoint_epoch
            resume_epoch_dates = list(checkpoint.get("epoch_dates", []))
            resume_next_date_index = int(checkpoint.get("next_date_index", 0))
            running_loss_sum = float(checkpoint.get("running_loss_sum", 0.0))
        print(
            f"[RESUME] {resume_path} | epoch={checkpoint_epoch} | global_step={global_step} "
            f"| next_date_index={checkpoint.get('next_date_index', 0)} "
            f"| epoch_complete={checkpoint.get('epoch_complete')} | val_complete={checkpoint.get('validation_complete')}",
            flush=True,
        )
    else:
        if REQUIRE_RESUME_CHECKPOINT:
            raise FileNotFoundError(
                "No resumable TS-Cast checkpoint was found. This release is configured to resume, "
                "not silently retrain from scratch. Attach the previous latest_model.pt as a Kaggle input "
                "dataset or set RESUME_CHECKPOINT to its /kaggle/input/... path."
            )
        print("[START] No TS-Cast checkpoint found; training from a fresh initialization.", flush=True)

    safe_atomic_json(OUTPUT_DIR / "experiment_spec.json", build_experiment_spec("STARTED"))
    safe_atomic_json(OUTPUT_DIR / "predictor_normalization_native36.json", {
        "train_only": True,
        "training_period": [TRAIN_START, TRAIN_END],
        "dynamic_features": DYNAMIC_VARS,
        "dynamic_mean": [float(x) for x in DYNAMIC_MEAN],
        "dynamic_std": [float(x) for x in DYNAMIC_STD],
        "static_features": STATIC_VARS,
        "static_mean": [float(x) for x in STATIC_MEAN],
        "static_std": [float(x) for x in STATIC_STD],
    })
    normalization_path = OUTPUT_DIR / "tscast_normalization.npz"
    normalization_tmp = OUTPUT_DIR / "tscast_normalization.npz.tmp"
    with normalization_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            target_climatology=target_climatology_np,
            target_anomaly_std_C=target_std_np,
            native_depths_m=NATIVE_DEPTHS_M,
            adt_monthly_climatology_normalized=ADT_MONTHLY_CLIM_NORM_NP,
            physical_depth_mask=physical_depth_mask_np,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(normalization_tmp, normalization_path)

    # If safe-stop occurred after training but during validation, redo validation
    # before starting the next epoch so best-checkpoint selection stays correct.
    while start_epoch <= MAX_EPOCHS:
        if should_soft_stop():
            if validation_pending_epoch is not None:
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=validation_pending_epoch, epoch_complete=True, validation_complete=False,
                    next_date_index=0, epoch_dates=[], global_step=global_step,
                    best_val_rmse=best_val_rmse, best_epoch=best_epoch, history=history, running_loss_sum=0.0,
                )
                pause_stage = "pending_validation"
            else:
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=max(1, start_epoch), epoch_complete=False, validation_complete=False,
                    next_date_index=resume_next_date_index, epoch_dates=resume_epoch_dates,
                    global_step=global_step, best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                    history=history, running_loss_sum=running_loss_sum,
                )
                pause_stage = "before_epoch_or_training_resume"
            safe_atomic_json(OUTPUT_DIR / "session_status.json", {
                "status": "PAUSED_AT_SAFE_DEADLINE", "stage": pause_stage,
                "epoch": validation_pending_epoch if validation_pending_epoch is not None else start_epoch,
                "global_step": global_step,
                "elapsed_hours": (time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0,
                "latest_checkpoint": str(latest_path),
            })
            return

        if validation_pending_epoch is not None:
            epoch = validation_pending_epoch
            validation_dates, full_validation = validation_dates_for_epoch(epoch)
            validation_scope = "VAL_FULL" if full_validation else "VAL_QUICK"
            print_section(f"RESUMING PENDING {validation_scope} — EPOCH {epoch} ({len(validation_dates)} dates)")
            try:
                val = evaluate_sih_tscast(model, validation_dates, validation_scope)
            except SessionTimeLimitReached as exc:
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=epoch, epoch_complete=True, validation_complete=False,
                    next_date_index=0, epoch_dates=[], global_step=global_step,
                    best_val_rmse=best_val_rmse, best_epoch=best_epoch, history=history,
                    running_loss_sum=0.0,
                )
                safe_atomic_json(OUTPUT_DIR / "session_status.json", {
                    "status": "PAUSED_AT_SAFE_DEADLINE", "stage": "validation",
                    "epoch": epoch, "global_step": global_step, "reason": str(exc),
                    "latest_checkpoint": str(latest_path),
                })
                return
            rmse = float(val["overall"]["RMSE_weighted_C"])
            record = {"epoch": epoch, "train_nll": None, "val_rmse_C": rmse,
                      "val_mae_C": float(val["overall"]["MAE_weighted_C"]),
                      "val_skill": float(val["overall"]["RMSE_skill"]),
                      "val_nll": val.get("training_objective_validation_NLL_no_constant"),
                      "validation_dates": len(validation_dates), "validation_scope": validation_scope,
                      "resumed_pending_validation": True,
                      "elapsed_hours": (time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0}
            history.append(record)
            if full_validation:
                safe_atomic_json(OUTPUT_DIR / "val_metrics.json", val)
                save_depth_csv(val, OUTPUT_DIR / "val_depth_metrics.csv")
                safe_atomic_json(OUTPUT_DIR / "val_uncertainty_metrics.json", val.get("uncertainty_calibration", {}))
                if math.isfinite(rmse) and rmse < best_val_rmse:
                    best_val_rmse, best_epoch = rmse, epoch
                    save_checkpoint(best_path, core_model, optimizer, scaler,
                        epoch=epoch, epoch_complete=True, validation_complete=True, next_date_index=0,
                        epoch_dates=[], global_step=global_step, best_val_rmse=best_val_rmse,
                        best_epoch=best_epoch, history=history, running_loss_sum=0.0, validation_result=val)
            else:
                safe_atomic_json(OUTPUT_DIR / f"val_quick_epoch_{epoch:02d}.json", val)
            save_checkpoint(latest_path, core_model, optimizer, scaler,
                epoch=epoch, epoch_complete=True, validation_complete=True, next_date_index=0,
                epoch_dates=[], global_step=global_step, best_val_rmse=best_val_rmse,
                best_epoch=best_epoch, history=history, running_loss_sum=0.0,
                validation_result=val if full_validation else None)
            safe_atomic_json(OUTPUT_DIR / "training_history.json", history)
            validation_pending_epoch = None
            start_epoch = epoch + 1
            continue

        epoch = start_epoch
        if resume_epoch_dates and epoch == start_epoch:
            epoch_dates = resume_epoch_dates
            cursor = resume_next_date_index
            running_sum = running_loss_sum
            resume_epoch_dates = []
            resume_next_date_index = 0
            running_loss_sum = 0.0
        else:
            rng = random.Random(SEED + epoch)
            if TRAIN_STEPS_PER_EPOCH >= len(TRAIN_DATES):
                epoch_dates = list(TRAIN_DATES)
            else:
                epoch_dates = sorted(rng.sample(TRAIN_DATES, TRAIN_STEPS_PER_EPOCH))
            cursor = 0
            running_sum = 0.0
        if not epoch_dates:
            raise RuntimeError(f"Epoch {epoch} has no training dates")
        print_section(f"TRAINING EPOCH {epoch}/{MAX_EPOCHS} — {len(epoch_dates)} SELECTED DATES")
        model.train()
        epoch_start_time = time.time()
        processed = cursor
        bad_gradient_streak = 0

        for date_start in range(cursor, len(epoch_dates), BATCH_DAYS):
            if should_soft_stop():
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=epoch, epoch_complete=False, validation_complete=False,
                    next_date_index=processed, epoch_dates=epoch_dates,
                    global_step=global_step, best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                    history=history, running_loss_sum=running_sum,
                )
                safe_atomic_json(OUTPUT_DIR / "session_status.json", {
                    "status": "PAUSED_AT_SAFE_DEADLINE", "stage": "training",
                    "epoch": epoch, "next_date_index": processed, "epoch_dates_count": len(epoch_dates),
                    "global_step": global_step,
                    "elapsed_hours": (time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0,
                    "latest_checkpoint": str(latest_path),
                })
                return

            day_chunk = epoch_dates[date_start:date_start + BATCH_DAYS]
            cpu_batch = collate_day_batches([build_one_day_batch(day) for day in day_chunk])
            batch = _move_batch(cpu_batch, DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with _safe_autocast():
                output = model(batch)
            loss = masked_gaussian_nll(output["mean"], output["logvar"], batch["y"], batch["loss_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite NLL at epoch={epoch}, date={day_chunk[0]}")
            gradient_skipped = False
            if scaler is not None and AMP_ENABLED:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), GRAD_CLIP_NORM)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    # Do not terminate on the first FP16 overflow. Skip this batch,
                    # lower GradScaler scale explicitly, preserve AdamW state, and continue.
                    old_scale = float(scaler.get_scale())
                    new_scale = max(old_scale * 0.5, 1.0)
                    scaler.update(new_scale=new_scale)
                    optimizer.zero_grad(set_to_none=True)
                    gradient_skipped = True
                    bad_gradient_streak += 1
                    print(
                        f"[AMP OVERFLOW SKIP] epoch={epoch} date={day_chunk[0]} "
                        f"grad_norm={float(torch.as_tensor(grad_norm).detach().cpu()):.4g} "
                        f"scale={old_scale:g}->{new_scale:g} streak={bad_gradient_streak}/{MAX_CONSECUTIVE_BAD_GRAD_STEPS}",
                        flush=True,
                    )
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    bad_gradient_streak = 0
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(core_model.parameters(), GRAD_CLIP_NORM)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"] = max(float(group["lr"]) * 0.5, 1e-7)
                    gradient_skipped = True
                    bad_gradient_streak += 1
                    print(f"[NON-AMP GRADIENT SKIP] epoch={epoch} date={day_chunk[0]} streak={bad_gradient_streak}", flush=True)
                else:
                    optimizer.step()
                    bad_gradient_streak = 0

            global_step += 1
            processed = date_start + len(day_chunk)
            running_sum += float(loss.detach().float().cpu().item()) * len(day_chunk)
            if gradient_skipped:
                # Persist the advanced cursor and reduced AMP scale before any possible stop.
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=epoch, epoch_complete=False, validation_complete=False,
                    next_date_index=processed, epoch_dates=epoch_dates, global_step=global_step,
                    best_val_rmse=best_val_rmse, best_epoch=best_epoch, history=history,
                    running_loss_sum=running_sum,
                )
                if bad_gradient_streak >= MAX_CONSECUTIVE_BAD_GRAD_STEPS:
                    safe_atomic_json(OUTPUT_DIR / "session_status.json", {
                        "status": "PAUSED_NUMERICAL_INSTABILITY", "epoch": epoch,
                        "next_date_index": processed, "bad_gradient_streak": bad_gradient_streak,
                        "global_step": global_step, "latest_checkpoint": str(latest_path),
                    })
                    raise FloatingPointError(
                        f"{bad_gradient_streak} consecutive non-finite gradient batches; checkpoint saved. "
                        "Inspect failure_report.json and AMP overflow logs before resuming."
                    )
            if global_step % LOG_EVERY_STEPS == 0 or processed >= len(epoch_dates):
                elapsed = time.time() - epoch_start_time
                print(
                    f"epoch={epoch:02d} dates={processed:4d}/{len(epoch_dates)} steps={global_step} "
                    f"nll={float(loss.detach().float().cpu().item()):.6f} "
                    f"mean_nll={running_sum / max(processed, 1):.6f} "
                    f"grad={float(torch.as_tensor(grad_norm).detach().float().cpu().item()):.4f} "
                    f"epoch_elapsed_min={elapsed / 60.0:.1f}",
                    flush=True,
                )

            if processed < len(epoch_dates) and processed % CHECKPOINT_EVERY_DATES == 0:
                save_checkpoint(
                    latest_path, core_model, optimizer, scaler,
                    epoch=epoch, epoch_complete=False, validation_complete=False,
                    next_date_index=processed, epoch_dates=epoch_dates,
                    global_step=global_step, best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                    history=history, running_loss_sum=running_sum,
                )
                print(
                    f"[RECOVERY CHECKPOINT] epoch={epoch} next_date_index={processed}/{len(epoch_dates)} "
                    f"path={latest_path}",
                    flush=True,
                )

        train_nll = running_sum / max(len(epoch_dates), 1)
        # Save an epoch-complete recovery point BEFORE validation. If the safe
        # deadline occurs during validation, it can be repeated on resume.
        save_checkpoint(
            latest_path, core_model, optimizer, scaler,
            epoch=epoch, epoch_complete=True, validation_complete=False,
            next_date_index=0, epoch_dates=[], global_step=global_step,
            best_val_rmse=best_val_rmse, best_epoch=best_epoch,
            history=history, running_loss_sum=0.0,
        )
        validation_dates, full_validation = validation_dates_for_epoch(epoch)
        validation_scope = "VAL_FULL" if full_validation else "VAL_QUICK"
        print(
            f"[EPOCH TRAIN COMPLETE] epoch={epoch} mean_nll={train_nll:.6f}; "
            f"beginning {validation_scope} on {len(validation_dates)} dates", flush=True
        )

        try:
            val = evaluate_sih_tscast(model, validation_dates, validation_scope)
        except SessionTimeLimitReached as exc:
            save_checkpoint(
                latest_path, core_model, optimizer, scaler,
                epoch=epoch, epoch_complete=True, validation_complete=False,
                next_date_index=0, epoch_dates=[], global_step=global_step,
                best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                history=history, running_loss_sum=0.0,
            )
            safe_atomic_json(OUTPUT_DIR / "session_status.json", {
                "status": "PAUSED_AT_SAFE_DEADLINE", "stage": "validation", "epoch": epoch,
                "global_step": global_step, "reason": str(exc), "latest_checkpoint": str(latest_path),
            })
            return

        rmse = float(val["overall"]["RMSE_weighted_C"])
        if not math.isfinite(rmse):
            raise FloatingPointError(f"Validation produced non-finite RMSE at epoch {epoch}")
        record = {
            "epoch": epoch,
            "train_nll": float(train_nll),
            "val_rmse_C": rmse,
            "val_mae_C": float(val["overall"]["MAE_weighted_C"]),
            "val_bias_C": float(val["overall"]["Bias_weighted_C"]),
            "val_skill": float(val["overall"]["RMSE_skill"]),
            "val_acc": float(val["overall"]["ACC_mean_over_SIH_depths"]),
            "val_nll": val.get("training_objective_validation_NLL_no_constant"),
            "val_one_sigma_coverage": val.get("uncertainty_calibration", {}).get("overall_one_sigma_coverage"),
            "val_95pct_coverage": val.get("uncertainty_calibration", {}).get("overall_95pct_interval_coverage"),
            "train_dates_used": len(epoch_dates),
            "validation_dates": len(validation_dates),
            "validation_scope": validation_scope,
            "epoch_elapsed_seconds": round(time.time() - epoch_start_time, 2),
            "elapsed_hours": round((time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0, 4),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        if full_validation:
            safe_atomic_json(OUTPUT_DIR / "val_metrics.json", val)
            save_depth_csv(val, OUTPUT_DIR / "val_depth_metrics.csv")
            safe_atomic_json(OUTPUT_DIR / "val_uncertainty_metrics.json", val.get("uncertainty_calibration", {}))
        else:
            safe_atomic_json(OUTPUT_DIR / f"val_quick_epoch_{epoch:02d}.json", val)
        safe_atomic_json(OUTPUT_DIR / "training_history.json", history)

        if full_validation and rmse < best_val_rmse:
            best_val_rmse, best_epoch = rmse, epoch
            save_checkpoint(
                best_path, core_model, optimizer, scaler,
                epoch=epoch, epoch_complete=True, validation_complete=True,
                next_date_index=0, epoch_dates=[], global_step=global_step,
                best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                history=history, running_loss_sum=0.0, validation_result=val,
            )
            print(f"[BEST FULL VAL] epoch={epoch} SIH RMSE={rmse:.6f} °C", flush=True)

        save_checkpoint(
            latest_path, core_model, optimizer, scaler,
            epoch=epoch, epoch_complete=True, validation_complete=True,
            next_date_index=0, epoch_dates=[], global_step=global_step,
            best_val_rmse=best_val_rmse, best_epoch=best_epoch,
            history=history, running_loss_sum=0.0,
            validation_result=val if full_validation else None,
        )
        safe_atomic_json(OUTPUT_DIR / "session_status.json", {
            "status": "RUNNING", "stage": "epoch_complete", "epoch": epoch,
            "global_step": global_step, "best_epoch": best_epoch,
            "best_validation_RMSE_C": best_val_rmse,
            "elapsed_hours": round((time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0, 4),
            "latest_checkpoint": str(latest_path), "best_checkpoint": str(best_path),
        })
        start_epoch = epoch + 1
        validation_pending_epoch = None

    # At configured epoch completion, preserve final status and report best result.
    if best_path.is_file():
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        safe_atomic_json(OUTPUT_DIR / "best_model_summary.json", {
            "status": "COMPLETED_EPOCH_BUDGET",
            "best_epoch": best_payload.get("best_epoch"),
            "best_validation_RMSE_C": best_payload.get("best_validation_rmse_C"),
            "best_checkpoint": str(best_path),
            "full_validation_metrics_file": str(OUTPUT_DIR / "val_metrics.json"),
            "test_period_used": False,
        })
    safe_atomic_json(OUTPUT_DIR / "session_status.json", {
        "status": "COMPLETED_EPOCH_BUDGET",
        "epoch": MAX_EPOCHS,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_validation_RMSE_C": best_val_rmse if math.isfinite(best_val_rmse) else None,
        "elapsed_hours": round((time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0, 4),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "test_period_used": False,
    })
    print(f"[COMPLETED] best epoch={best_epoch} validation RMSE={best_val_rmse:.6f} °C", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve a readable failure record, including CUDA OOMs, then re-raise.
        try:
            if "OUTPUT_DIR" in globals():
                safe_atomic_json(OUTPUT_DIR / "failure_report.json", {
                    "status": "FAILED", "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_hours": round((time.monotonic() - SCRIPT_START_MONOTONIC) / 3600.0, 4),
                    "cuda_peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
                    "cuda_peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
                })
        except Exception as report_exc:
            print(f"[WARNING] Could not write failure_report.json: {report_exc}", flush=True)
        raise
