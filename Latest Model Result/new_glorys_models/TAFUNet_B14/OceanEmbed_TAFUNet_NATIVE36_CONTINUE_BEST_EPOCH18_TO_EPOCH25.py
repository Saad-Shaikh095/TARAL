#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OceanEmbed TAFU-Net V2 Native36 — best epoch 18 fine-tune through epoch 25

TRAIN:
  * every eligible date in 2012-01-14 .. 2020-12-31
  * ALL 36 native GLORYS depths in the loss

VALIDATION / TEST:
  * ONLY the original 15 SIH depths are evaluated
  * validation/test do not score the remaining 21 native levels

Architecture:
  * shared per-day CNN encoder with predictor-validity masks appended
  * replicate-padded GroupNorm/GELU residual blocks with channel excitation
  * multi-scale 50x120 and 25x60 spatial representations
  * channel-wise, spatially varying temporal attention across the 14-day window
  * explicit 3-interval tendency and full-window trend at both encoder scales
  * U-Net decoder with final-day + tendency high-resolution skip connection
  * zero-initialized full-column 36->72->36 depth-profile residual MLP
  * 1x1 output head predicts all 36 native-depth anomalies

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


# Start the wall-clock budget before notebook-side data bootstrap/preflight.
SCRIPT_START_MONOTONIC = time.monotonic()


# ============================================================================
# CONFIG
# ============================================================================

# Resolve the original model-data dataset across common Kaggle mount layouts.
# This helps when the dataset is attached from a teammate's account with a different slug.
_EXPECTED_DATA_ROOT = Path("/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1")
_EXPECTED_PREDICTOR_FILE = "Duacs_Oisst_2012_2024_FeaturesV1.nc"

def _resolve_data_root() -> Path:
    candidates = [
        _EXPECTED_DATA_ROOT,
        Path("/kaggle/input/oceanembed-modeldata-v1"),
    ]
    for candidate in candidates:
        if (candidate / _EXPECTED_PREDICTOR_FILE).is_file():
            return candidate
    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = sorted(input_root.rglob(_EXPECTED_PREDICTOR_FILE))
        if matches:
            return matches[0].parent
    # Keep the familiar path so the downstream missing-file error is actionable.
    return _EXPECTED_DATA_ROOT

DATA_ROOT = _resolve_data_root()
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

EPOCHS = 25  # TOTAL epoch target; resume from completed best epoch 18 and train epochs 19-25.
# Effective batch remains one training date/sample per optimizer step; the current loop is not batched.
BATCH_SIZE = 1
LEARNING_RATE = 2e-4  # Original full-training peak LR; retained for provenance.
MIN_LEARNING_RATE = 1e-5  # Original full-training floor; retained for provenance.
# Low-LR continuation schedule, applied to epochs 19-25 only. This avoids
# recomputing the original 20-epoch cosine schedule over a 25-epoch horizon
# (which would incorrectly raise LR after restoring the epoch-18 checkpoint).
CONTINUATION_LR_START = 1e-5
CONTINUATION_LR_END = 3e-6
EXPECTED_RESUME_EPOCH = 18
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42
AMP_ENABLED = True

LOG_EVERY = 25
# Cache sizes are intentionally bounded. A target raw field is ~3.46 MB and a
# normalized target + boolean mask is ~4.32 MB, before allocator/cache overhead.
# Randomly shuffling individual dates destroys the 13/14-day sliding-window
# overlap. Deterministic block-shuffling below captures most locality instead
# of retaining several thousand full-grid tensors in RAM.
TRAIN_DATE_BLOCK_SIZE = 32
DYNAMIC_DAY_CACHE_SIZE = 96
GLORYS_OPEN_CACHE_SIZE = 16
TARGET_DAY_CACHE_SIZE = 48

RUN_ROOT = Path("/kaggle/working/OceanEmbed_Runs")
EXPERIMENT_NAME = "TAFUNET_CHANNELWISE_KINEMATIC_NATIVE36_V2"
PATTERN_VERSION = "TAFUNET_NATIVE36_KINEMATIC_V2"

# Fine-tune the verified BEST epoch-18 TAFU-Net Native36 checkpoint through total epoch 25.
# This is a continuation, not a fresh run. The source checkpoint is mounted read-only.
# The run writes new checkpoints/artifacts under /kaggle/working/OceanEmbed_Runs.
# Set RUN_MODE="SMOKE_TEST" only for a short real-data verification; FULL trains normally.
RUN_MODE = "FULL"

# Exact requested checkpoint: best model from completed epoch 18.
RESUME_FROM: Optional[str] = "/kaggle/input/datasets/soulshadow669/epoch-18-bestmodel/best_model.pt"
FALLBACK_RESUME_FROM: Optional[str] = None
MIN_EXPECTED_RESUME_EPOCH = EXPECTED_RESUME_EPOCH

# User requested NO periodic checkpoint writes during training.
# 0 disables periodic saves; the script still saves on the 11-hour safe stop
# and after every completed epoch. Logging every 25 steps does not save weights.
CHECKPOINT_EVERY_STEPS = 0

# Stop the Python process cleanly before Kaggle's ~12-hour GPU-version limit.
# The budget begins at Python script startup (including data preflight).
# At the deadline, checkpoint is written once and the script exits normally.
SESSION_SOFT_LIMIT_HOURS = 11.0
SESSION_TIME_LIMIT_SECONDS = SESSION_SOFT_LIMIT_HOURS * 3600.0

# Fail fast before opening large datasets if the run mode/explicit checkpoint is wrong.
if RUN_MODE not in {"SMOKE_TEST", "FULL"}:
    raise ValueError(f"Invalid RUN_MODE={RUN_MODE!r}; expected SMOKE_TEST or FULL")
if RESUME_FROM is not None and not Path(RESUME_FROM).is_file():
    raise FileNotFoundError(
        f"Explicit RESUME_FROM checkpoint is not mounted at {RESUME_FROM!r}. "
        "Set RESUME_FROM=None to enable auto-discovery, or correct the path."
    )


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


def resolve_resume_checkpoint() -> Optional[Path]:
    """Find the most advanced checkpoint for this exact TAFU-Net v2 experiment.

    Both /kaggle/working and mounted /kaggle/input are searched, but checkpoints
    with a different experiment name or model ID are never eligible. No match
    means a fresh initialization.
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
            if candidate.get('model_id') != MODEL_ID:
                continue
            if int(candidate.get('target_depth_count', -1)) != N_DEPTH:
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
                and fallback_payload.get('model_id') == MODEL_ID
                and int(fallback_payload.get('target_depth_count', -1)) == N_DEPTH
                and fallback_epoch >= MIN_EXPECTED_RESUME_EPOCH
            ):
                print(f"AUTO-RESUME: using validated fallback: {fallback}", flush=True)
                return fallback
        except Exception as exc:
            print(f"[RESUME WARNING] Could not validate fallback {fallback}: {exc}", flush=True)

    print(
        "[RESUME] No checkpoint for this TAFU-Net v2 experiment was found; "
        "starting a fresh v2 baseline run (not loading B6 or TAFU-Net v1 weights).",
        flush=True,
    )
    return None


# ============================================================================
# FILES
# ============================================================================

print_section("OCEANEMBED TAFU-NET V2 CHANNEL-WISE KINEMATIC NATIVE36 RUN")

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
print("CHECKPOINT POLICY: epoch end + safe stop only; source best checkpoint remains read-only")
print(f"CACHE POLICY     : dynamic={DYNAMIC_DAY_CACHE_SIZE}, target={TARGET_DAY_CACHE_SIZE}, open GLORYS={GLORYS_OPEN_CACHE_SIZE}")
print(f"DATE ORDER       : deterministic shuffled blocks of {TRAIN_DATE_BLOCK_SIZE} consecutive dates")
print(f"LR SCHEDULE      : fine-tune cosine {CONTINUATION_LR_START:.2e} -> {CONTINUATION_LR_END:.2e} over epochs {EXPECTED_RESUME_EPOCH + 1}-{EPOCHS}")

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


def block_shuffled_dates(dates: Sequence[str], block_size: int, seed: int) -> List[str]:
    """Shuffle contiguous date blocks, preserving chronological order within each block.

    Every date appears exactly once. This retains stochastic ordering between
    blocks while making adjacent 14-day windows overlap in cache, which avoids
    the I/O thrashing caused by a full random permutation of individual dates.
    The function is deterministic for a given seed, so mid-epoch resume recreates
    the same order and the saved step index remains valid.
    """
    if block_size < WINDOW_DAYS:
        raise ValueError(f"block_size must be >= WINDOW_DAYS ({WINDOW_DAYS})")
    blocks = [list(dates[i:i + block_size]) for i in range(0, len(dates), block_size)]
    random.Random(seed).shuffle(blocks)
    return [day for block in blocks for day in block]


# ============================================================================
# SHARED MASK-AWARE INPUT PREPARATION
# ============================================================================
# The Native36 data/mask contract is carried over from the reference script.
# TAFU-Net uses the same prepared inputs and loss mask, but its network blocks
# are new and do not instantiate the reference B6 ConvLSTM architecture.
# ============================================================================
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
MODEL_ID = 'TAFUNET_CHANNELWISE_KINEMATIC_NATIVE36_V2'
MODEL_VERSION = '2.0.0-native36-channelwise'
MODEL_CONTRACT = 'TAFUNETNATIVE36V2'
EXPECTED_LAND_OCEAN_FRACTION_Z = -1.001882263507042
OCEAN_FRACTION_Z_TOL = 1e-10
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
        raise ContractError(f"x_static shape {tuple(x_static.shape)} is incompatible with the Native36 input contract")

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


def _to_bool_mask(
    mask: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str,
) -> torch.Tensor:
    """Validate a validity mask and return it as a boolean tensor.

    Accepts bool masks or numeric 0/1 masks. Numeric masks must contain only
    finite values; any nonzero value is treated as valid after validation.
    """
    if not isinstance(mask, torch.Tensor):
        raise ContractError(f"{name} must be a torch.Tensor, got {type(mask).__name__}")
    if tuple(mask.shape) != tuple(expected_shape):
        raise ContractError(
            f"{name} shape {tuple(mask.shape)} != expected {tuple(expected_shape)}"
        )
    if mask.dtype != torch.bool:
        if mask.is_floating_point() and not torch.isfinite(mask).all():
            raise ContractError(f"{name} contains NaN or infinite values")
        if torch.any((mask != 0) & (mask != 1)):
            raise ContractError(f"{name} must contain only boolean/0/1 values")
    return mask.to(dtype=torch.bool)


def prepare_model_inputs(
    batch: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    required = (
        "x_dynamic", "x_static", "x_dynamic_valid", "x_static_valid",
        "context_spatial", "context_temporal",
    )
    missing = [k for k in required if k not in batch]
    if missing:
        raise ContractError(f"Model batch missing required keys: {missing}")

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

class ResidualConvBlock(nn.Module):
    """Residual convolution block with replicate padding and SE channel calibration."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        squeeze_channels = max(out_channels // 8, 8)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation,
            padding_mode="replicate", bias=False,
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=dilation, dilation=dilation,
            padding_mode="replicate", bias=False,
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, squeeze_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(squeeze_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        # 2*sigmoid initializes near a neutral multiplier of one.
        x = x * (2.0 * self.channel_excitation(x))
        return self.activation(x + residual)


class SpatialTemporalAttention(nn.Module):
    """Channel-wise, spatially varying attention over time."""

    def __init__(self, channels: int, window_days: int):
        super().__init__()
        mid = max(channels // 2, 16)
        self.score = nn.Sequential(
            nn.Conv2d(
                channels, mid, kernel_size=3, padding=1,
                padding_mode="replicate", bias=False,
            ),
            nn.GroupNorm(4, mid),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
        )
        # Soft recency prior; learned spatial/channel evidence can override it.
        self.time_bias = nn.Parameter(torch.linspace(-0.5, 0.5, window_days))

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sequence: [B,T,C,H,W]; attention weights are [B,T,C,H,W]
        if sequence.ndim != 5:
            raise ValueError(
                f"Temporal attention expects [B,T,C,H,W], got {tuple(sequence.shape)}"
            )
        batch_size, timesteps, channels, height, width = sequence.shape
        flat = sequence.reshape(batch_size * timesteps, channels, height, width)
        scores = self.score(flat).reshape(
            batch_size, timesteps, channels, height, width
        )
        scores = scores + self.time_bias[:timesteps].view(1, timesteps, 1, 1, 1)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=sequence.dtype)
        fused = torch.sum(sequence * weights, dim=1)
        return fused, weights


class Model(nn.Module):
    """TAFU-Net v2: channel-wise temporal fusion, temporal tendencies and profile MLP."""

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
        encoded_input_channels = TOTAL_INPUT_CHANNELS * 2  # values + validity indicators

        self.encoder_50 = nn.Sequential(
            nn.Conv2d(
                encoded_input_channels, 32, kernel_size=3, stride=2,
                padding=1, padding_mode="replicate", bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualConvBlock(32, 32),
        )
        self.encoder_25 = nn.Sequential(
            nn.Conv2d(
                32, 48, kernel_size=3, stride=2,
                padding=1, padding_mode="replicate", bias=False,
            ),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            ResidualConvBlock(48, 48),
        )
        self.low_context = nn.Sequential(
            ResidualConvBlock(48, 48, dilation=1),
            ResidualConvBlock(48, 48, dilation=2),
        )

        self.attention_50 = SpatialTemporalAttention(32, EXPECTED_WINDOW_DAYS)
        self.attention_25 = SpatialTemporalAttention(48, EXPECTED_WINDOW_DAYS)

        # Attention summary + final frame + short tendency + full-window trend.
        self.fuse_50 = nn.Sequential(
            nn.Conv2d(32 * 4, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualConvBlock(32, 32),
        )
        self.fuse_25 = nn.Sequential(
            nn.Conv2d(48 * 4, 48, kernel_size=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            ResidualConvBlock(48, 48),
        )

        # Final-day features plus 3-step feature change at full spatial resolution.
        self.highres_skip = nn.Sequential(
            nn.Conv2d(
                encoded_input_channels * 2, 24, kernel_size=3,
                padding=1, padding_mode="replicate", bias=False,
            ),
            nn.GroupNorm(8, 24),
            nn.GELU(),
            ResidualConvBlock(24, 24),
        )
        self.decoder_50 = ResidualConvBlock(48 + 32, 32)
        self.decoder_100 = ResidualConvBlock(32 + 24, 32)
        self.output_head = nn.Conv2d(32, N_DEPTH, kernel_size=1, bias=True)

        # Per-grid-cell full-column MLP: any native level can influence any other.
        # Zero-initialized last layer keeps this branch an exact residual identity
        # at initialization, avoiding a random perturbation to the initial profile.
        self.depth_profile_mixer = nn.Sequential(
            nn.Conv2d(N_DEPTH, N_DEPTH * 2, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(N_DEPTH * 2, N_DEPTH, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.depth_profile_mixer[-1].weight)
        nn.init.zeros_(self.depth_profile_mixer[-1].bias)

    @staticmethod
    def _masked_pool_ocean(ocean_mask: torch.Tensor, factor: int) -> torch.Tensor:
        pooled = F.max_pool2d(ocean_mask, kernel_size=factor, stride=factor)
        return (pooled > 0).to(dtype=ocean_mask.dtype)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        seq, seq_mask, ocean_mask_hr, _ = prepare_model_inputs(batch)
        batch_size, timesteps, channels, height, width = seq.shape
        if timesteps != self.window_days:
            raise ContractError(
                f"TAFU-Net expected {self.window_days} days, got {timesteps}"
            )
        if (height, width) != (GRID_H, GRID_W):
            raise ContractError(
                f"TAFU-Net expected {GRID_H}x{GRID_W}, got {height}x{width}"
            )

        valid = seq_mask.to(dtype=seq.dtype)
        encoded = torch.cat([seq * valid, valid], dim=2)
        encoded_flat = encoded.reshape(
            batch_size * timesteps, channels * 2, height, width
        )

        ocean_50 = self._masked_pool_ocean(ocean_mask_hr, factor=2)
        ocean_25 = self._masked_pool_ocean(ocean_mask_hr, factor=4)
        ocean_50_flat = ocean_50[:, None].expand(
            -1, timesteps, -1, -1, -1
        ).reshape(batch_size * timesteps, 1, GRID_H // 2, GRID_W // 2)
        ocean_25_flat = ocean_25[:, None].expand(
            -1, timesteps, -1, -1, -1
        ).reshape(batch_size * timesteps, 1, GRID_H // 4, GRID_W // 4)

        mid_flat = self.encoder_50(encoded_flat) * ocean_50_flat
        mid_sequence = mid_flat.reshape(
            batch_size, timesteps, 32, mid_flat.shape[-2], mid_flat.shape[-1]
        ) * ocean_50[:, None]

        low_flat = self.encoder_25(mid_flat) * ocean_25_flat
        low_flat = self.low_context(low_flat) * ocean_25_flat
        low_sequence = low_flat.reshape(
            batch_size, timesteps, 48, low_flat.shape[-2], low_flat.shape[-1]
        ) * ocean_25[:, None]

        mid_temporal, _ = self.attention_50(mid_sequence)
        low_temporal, _ = self.attention_25(low_sequence)

        # -1 versus -4 is a three-interval tendency; -1 versus 0 spans the window.
        mid_velocity_3 = mid_sequence[:, -1] - mid_sequence[:, -4]
        mid_trend = mid_sequence[:, -1] - mid_sequence[:, 0]
        mid_fused = self.fuse_50(torch.cat(
            [mid_temporal, mid_sequence[:, -1], mid_velocity_3, mid_trend], dim=1
        )) * ocean_50

        low_velocity_3 = low_sequence[:, -1] - low_sequence[:, -4]
        low_trend = low_sequence[:, -1] - low_sequence[:, 0]
        low_fused = self.fuse_25(torch.cat(
            [low_temporal, low_sequence[:, -1], low_velocity_3, low_trend], dim=1
        )) * ocean_25

        highres_input = torch.cat(
            [encoded[:, -1], encoded[:, -1] - encoded[:, -4]], dim=1
        )
        highres = self.highres_skip(highres_input) * ocean_mask_hr

        low_up = F.interpolate(
            low_fused, size=mid_fused.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded_50 = self.decoder_50(torch.cat([low_up, mid_fused], dim=1)) * ocean_50
        decoded_up = F.interpolate(
            decoded_50, size=(GRID_H, GRID_W), mode="bilinear", align_corners=False
        )
        decoded_100 = self.decoder_100(
            torch.cat([decoded_up, highres], dim=1)
        ) * ocean_mask_hr

        raw_profile = self.output_head(decoded_100)
        profile_correction = self.depth_profile_mixer(raw_profile)
        prediction = (raw_profile + profile_correction) * ocean_mask_hr

        expected = (batch_size, N_DEPTH, GRID_H, GRID_W)
        if tuple(prediction.shape) != expected:
            raise ContractError(
                f"TAFU-Net output shape {tuple(prediction.shape)} != {expected}"
            )
        return prediction

    def model_summary(self) -> Dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_contract": MODEL_CONTRACT,
            "parameters": parameter_summary(self),
            "architecture": {
                "input_per_day_channels": TOTAL_INPUT_CHANNELS,
                "validity_channels_appended": TOTAL_INPUT_CHANNELS,
                "sequence_length": EXPECTED_WINDOW_DAYS,
                "encoder_scales": ["50x120 / 32 channels", "25x60 / 48 channels"],
                "temporal_aggregation": "channel-wise spatial attention + 3-step tendency + window trend",
                "decoder": "replicate-padded multi-scale U-Net with final-day and tendency skip",
                "depth_head": "36-channel head + zero-initialized full-column 36->72->36 residual MLP",
                "residual_blocks": "GroupNorm/GELU/replicate padding + squeeze-excitation channel calibration",
                "output_shape": ["B", N_DEPTH, GRID_H, GRID_W],
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
            if session_time_limit_reached():
                raise SessionTimeLimitReached(
                    f"11-hour session limit reached during {split_name} at {position - 1}/{len(dates)} dates"
                )

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


    if session_time_limit_reached():
        raise SessionTimeLimitReached(
            f"11-hour session limit reached at the end of {split_name} evaluation"
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
    *,
    epoch_complete: bool = True,
    step_in_epoch: int = 0,
    running_loss: float = 0.0,
) -> Dict[str, Any]:

    return {
        "schema": "OCEANEMBED_TAFUNET_NATIVE36_CHECKPOINT_V2",
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
        "epoch_complete": bool(epoch_complete),
        "step_in_epoch": int(step_in_epoch),
        "running_loss": float(running_loss),
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


def record_session_stop(
    output_dir: Path,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    best_val_rmse: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step_in_epoch: int,
    running_loss: float,
    checkpoint_incomplete: bool = True,
    save_checkpoint: bool = True,
) -> None:
    """Persist a recovery point and a clear stop marker before clean exit."""
    if save_checkpoint:
        payload = checkpoint_payload(
            model, optimizer, scaler, epoch, global_step,
            best_val_rmse, best_epoch, history,
            epoch_complete=not checkpoint_incomplete,
            step_in_epoch=step_in_epoch,
            running_loss=running_loss,
        )
        atomic_torch_save(output_dir / 'latest_model.pt', payload)

    elapsed = session_elapsed_seconds()
    status = {
        'status': 'PAUSED_AT_SESSION_LIMIT',
        'reason': 'Configured 11-hour wall-clock budget reached; exiting before Kaggle hard cutoff.',
        'stage': stage,
        'epoch': int(epoch),
        'epoch_complete': not checkpoint_incomplete,
        'step_in_epoch': int(step_in_epoch),
        'global_step': int(global_step),
        'best_epoch': int(best_epoch),
        'best_validation_rmse_C': float(best_val_rmse) if math.isfinite(best_val_rmse) else None,
        'elapsed_hours': round(elapsed / 3600.0, 4),
        'resume_instruction': 'Attach this run output or publish latest_model.pt as a Kaggle dataset; this script resumes only matching TAFU-Net Native36 checkpoints.',
        'saved_utc': datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_dir / 'session_status.json', status)

    spec_path = output_dir / 'experiment_spec.json'
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding='utf-8'))
            spec['status'] = 'PAUSED_AT_SESSION_LIMIT'
            spec['session_status'] = status
            atomic_json(spec_path, spec)
        except Exception as exc:
            print(f'[SAVE WARNING] Could not update experiment_spec.json status: {exc}', flush=True)

    step_text = (
        f"step={step_in_epoch}/{len(TRAIN_DATES)}"
        if stage == 'training'
        else f"train_steps={step_in_epoch}/{len(TRAIN_DATES)}"
    )
    print(
        f"\n[SAFE STOP] 11-hour budget reached at stage={stage}, epoch={epoch}, "
        f"{step_text}, global_step={global_step}. "
        f"Checkpoint: {output_dir / 'latest_model.pt'}",
        flush=True,
    )
    print('[SAFE STOP] Exiting normally now so Kaggle can publish the saved output.', flush=True)


# ============================================================================
# SAVE METRICS
# ============================================================================

def save_result(
    result: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Persist metric JSON and per-depth CSV atomically.

    CSV columns are derived from the rows returned by SIHMetricAccumulator,
    so additions such as native_lower_index/mapping_method cannot break saving.
    """
    split = str(result.get("split", "unknown")).lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON is already atomic; keep it as a recoverable record even if CSV fails.
    atomic_json(output_dir / f"{split}_metrics.json", result)

    rows = list(result.get("by_depth", []))
    csv_path = output_dir / f"{split}_depth_metrics.csv"
    tmp_path = Path(str(csv_path) + ".tmp")

    if not rows:
        # Still create a valid empty CSV with a stable minimal schema.
        fieldnames = ["depth_m", "n_valid_cells", "RMSE_C", "MAE_C", "Bias_C"]
    else:
        # Preserve first-seen column order across all rows.
        fieldnames = list(dict.fromkeys(
            key for row in rows for key in row.keys()
        ))

    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
                restval="",
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main() -> None:

    run_start = SCRIPT_START_MONOTONIC

    resume_path = resolve_resume_checkpoint()

    # When resuming from /kaggle/working, continue writing into that same run
    # folder; this preserves previous checkpoints and validation JSON. Mounted
    # /kaggle/input checkpoints are read-only, so create a fresh working folder.
    if resume_path is not None:
        try:
            resume_path.resolve().relative_to(RUN_ROOT.resolve())
            resume_is_working = True
        except (ValueError, OSError):
            resume_is_working = False
    else:
        resume_is_working = False

    if resume_is_working:
        output_dir = resume_path.parent
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = RUN_ROOT / f"{EXPERIMENT_NAME}_{run_id}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT DIRECTORY: {output_dir}")

    # Recover export artifacts from any JSON that survived an earlier CSV error.
    # In the incident being repaired, val_metrics.json was written successfully
    # before the old CSV writer raised, so this reconstructs val_depth_metrics.csv.
    for existing_split in ("val", "test"):
        existing_json = output_dir / f"{existing_split}_metrics.json"
        if existing_json.is_file():
            try:
                existing_result = json.loads(existing_json.read_text(encoding="utf-8"))
                save_result(existing_result, output_dir)
                overall = existing_result.get("overall", {})
                print(f"[RECOVERY] Re-exported {existing_split} CSV from {existing_json.name}")
                if overall:
                    print(
                        f"[RECOVERY METRICS] {existing_split.upper()} "
                        f"RMSE={overall.get('RMSE_weighted_C', float('nan')):.6f} °C | "
                        f"MAE={overall.get('MAE_weighted_C', float('nan')):.6f} °C | "
                        f"skill={overall.get('RMSE_skill', float('nan')):.6f} | "
                        f"ACC={overall.get('ACC_mean_over_SIH_depths', float('nan')):.6f}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[RECOVERY WARNING] Could not re-export {existing_split} metrics: {exc}")


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
            "run_mode": RUN_MODE,
            "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
            "session_soft_limit_hours": SESSION_SOFT_LIMIT_HOURS,
            "cache_policy": {
                "dynamic_day_cache_entries": DYNAMIC_DAY_CACHE_SIZE,
                "target_day_cache_entries": TARGET_DAY_CACHE_SIZE,
                "open_glorys_files": GLORYS_OPEN_CACHE_SIZE,
                "train_date_block_size": TRAIN_DATE_BLOCK_SIZE,
            },
            "experiment_name": EXPERIMENT_NAME,
            "pattern_version": PATTERN_VERSION,
            "device": str(DEVICE),
            "architecture": {
                "family": "TAFU-Net v2 channel-wise temporal-attention multi-scale U-Net",
                "input": "36 predictors + 36 validity-indicator channels per day",
                "encoder_scales": ["50x120 / 32 channels", "25x60 / 48 channels"],
                "temporal_aggregation": "channel-wise spatial attention + 3-step tendency + full-window trend at 50x120 and 25x60",
                "decoder": "replicate-padded multi-scale skip decoder with final-day and tendency 100x240 features",
                "head": "36 native-depth output channels + zero-initialized full-column 36->72->36 residual MLP",
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
                "learning_rate_peak": CONTINUATION_LR_START,
                "learning_rate_floor": CONTINUATION_LR_END,
                "learning_rate_schedule": "cosine_per_step_finetune_epoch19_to_epoch25",
                "original_full_training_learning_rate_peak": LEARNING_RATE,
                "original_full_training_learning_rate_floor": MIN_LEARNING_RATE,
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
        "REAL TAFU-NET NATIVE36 MODEL"
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
            f"Real TAFU-Net output shape "
            f"{tuple(preflight_prediction.shape)} "
            f"!= {expected_output}"
        )


    if not torch.isfinite(
        preflight_prediction
    ).all():

        raise FloatingPointError(
            "Real TAFU-Net preflight produced non-finite predictions"
        )


    print(
        f"Forward output     : "
        f"{tuple(preflight_prediction.shape)}"
    )

    print(
        "Real TAFU-Net forward: PASS"
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
    resume_epoch: Optional[int] = None
    resume_step_in_epoch = 0
    resume_running_loss = 0.0
    checkpoint: Dict[str, Any] = {}

    global_step = 0
    best_val_rmse = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []


    if resume_path is not None:


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


        if checkpoint.get("experiment_name") != EXPERIMENT_NAME:
            raise RuntimeError("Refusing resume from another experiment")
        if checkpoint.get("model_id") != MODEL_ID:
            raise RuntimeError("Refusing resume from a different model architecture")
        if checkpoint.get("pattern_version") != PATTERN_VERSION:
            raise RuntimeError("Refusing resume: pattern version mismatch")
        if checkpoint.get("model_contract") != MODEL_CONTRACT:
            raise RuntimeError("Refusing resume: model contract mismatch")
        checkpoint_epoch_for_guard = int(checkpoint.get("epoch", -1))
        checkpoint_best_epoch_for_guard = int(checkpoint.get("best_epoch", -1))
        checkpoint_epoch_complete_for_guard = bool(checkpoint.get("epoch_complete", True))
        if checkpoint_epoch_for_guard != EXPECTED_RESUME_EPOCH:
            raise RuntimeError(
                f"Expected the completed epoch-{EXPECTED_RESUME_EPOCH} checkpoint, "
                f"but received epoch {checkpoint_epoch_for_guard}. "
                "Check RESUME_FROM before training."
            )
        if checkpoint_best_epoch_for_guard != EXPECTED_RESUME_EPOCH:
            raise RuntimeError(
                f"The provided checkpoint is epoch {checkpoint_epoch_for_guard}, "
                f"but its recorded best_epoch is {checkpoint_best_epoch_for_guard}; "
                "the requested source must be the epoch-18 best checkpoint."
            )
        if not checkpoint_epoch_complete_for_guard:
            raise RuntimeError("Refusing resume: epoch-18 source checkpoint is incomplete")


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


        checkpoint_epoch = int(checkpoint["epoch"])
        # Backward compatibility: old checkpoints were written only after a
        # completed epoch, so absence of this field means epoch-complete.
        checkpoint_epoch_complete = bool(checkpoint.get("epoch_complete", True))
        if checkpoint_epoch_complete:
            start_epoch = checkpoint_epoch + 1
            resume_epoch = None
            resume_step_in_epoch = 0
            resume_running_loss = 0.0
        else:
            start_epoch = checkpoint_epoch
            resume_epoch = checkpoint_epoch
            resume_step_in_epoch = int(checkpoint.get("step_in_epoch", 0))
            resume_running_loss = float(checkpoint.get("running_loss", 0.0))

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


        if resume_epoch is not None:
            print(
                f"Resuming IN epoch {resume_epoch} at step "
                f"{resume_step_in_epoch + 1}/"
                f"{len(TRAIN_DATES)}; accumulated loss "
                f"={resume_running_loss:.6f}"
            )
        else:
            print(f"Resuming at epoch {start_epoch}")

        # A previous version could crash while exporting the CSV after saving
        # latest_model.pt but before creating best_model.pt. Reconstruct best
        # from a checkpoint whose epoch matches best_epoch (never relabel latest
        # weights as "best" when its epoch differs).
        best_path = output_dir / "best_model.pt"
        if best_epoch > 0 and not best_path.is_file():
            best_candidates: List[Path] = [resume_path]
            fallback_best = Path(FALLBACK_RESUME_FROM) if FALLBACK_RESUME_FROM else None
            if fallback_best is not None and fallback_best.is_file():
                best_candidates.append(fallback_best)
            checkpoint_roots = [RUN_ROOT, Path('/kaggle/input')]
            for checkpoint_root in checkpoint_roots:
                if not checkpoint_root.exists():
                    continue
                for checkpoint_pattern in ('latest_model*.pt', 'best_model*.pt'):
                    for candidate_path in sorted(
                        checkpoint_root.rglob(checkpoint_pattern),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0,
                        reverse=True,
                    ):
                        if candidate_path not in best_candidates:
                            best_candidates.append(candidate_path)

            best_seeded = False
            for candidate_path in best_candidates:
                try:
                    candidate_payload = torch.load(candidate_path, map_location="cpu")
                    if (
                        candidate_payload.get("experiment_name") == EXPERIMENT_NAME
                        and int(candidate_payload.get("epoch", -1)) == best_epoch
                        and int(candidate_payload.get("target_depth_count", -1)) == N_DEPTH
                        and candidate_payload.get("normalization_hash") == hash_json(normalization_payload)
                    ):
                        atomic_torch_save(best_path, candidate_payload)
                        print(f"[RECOVERY] Reconstructed best_model.pt from {candidate_path}")
                        best_seeded = True
                        break
                except Exception as exc:
                    print(f"[RECOVERY WARNING] Could not inspect {candidate_path}: {exc}")

            if not best_seeded:
                print(
                    "[WARNING] Could not reconstruct the previous best checkpoint. "
                    "Training will continue, but preserve a copy of the checkpoint "
                    "for best_epoch before relying on final best-model restoration."
                )

        # The history is embedded in each checkpoint. Recreate the JSON now, not
        # only after the next epoch, so the prior epoch's record survives.
        try:
            atomic_json(output_dir / "training_history.json", history)
        except Exception as exc:
            print(f"[RECOVERY WARNING] Could not restore training_history.json: {exc}")


    # ------------------------------------------------------------------------
    # SMOKE TEST MODE: one real forward/backward + SIH-path validation
    # ------------------------------------------------------------------------
    if RUN_MODE not in {"SMOKE_TEST", "FULL"}:
        raise ValueError(f"Invalid RUN_MODE={RUN_MODE!r}; expected SMOKE_TEST or FULL")

    if RUN_MODE == "SMOKE_TEST":
        print_section("SMOKE TEST — REAL TRAIN STEP")
        print(f"Loaded checkpoint : {resume_path if resume_path is not None else 'none (fresh initialization test)'}")
        print(f"Checkpoint epoch  : {int(checkpoint.get('epoch', 0)) if resume_path is not None else 0}")
        print(f"Device            : {DEVICE}")
        print(f"Test date         : {preflight_day}")

        test_batch = {
            key: value.to(DEVICE, non_blocking=True)
            for key, value in preflight_batch.items()
        }
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if DEVICE.type == "cuda" and AMP_ENABLED:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                test_prediction = model(test_batch)
                test_loss = masked_mse_loss(
                    test_prediction, test_batch["y"], test_batch["loss_mask"]
                )
            if not torch.isfinite(test_loss):
                raise FloatingPointError("Smoke test produced non-finite loss")
            scaler.scale(test_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("Smoke test produced non-finite gradient norm")
            scaler.step(optimizer)
            scaler.update()
        else:
            test_prediction = model(test_batch)
            test_loss = masked_mse_loss(
                test_prediction, test_batch["y"], test_batch["loss_mask"]
            )
            if not torch.isfinite(test_loss):
                raise FloatingPointError("Smoke test produced non-finite loss")
            test_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError("Smoke test produced non-finite gradient norm")
            optimizer.step()

        expected_shape = (1, N_DEPTH, GRID_H, GRID_W)
        if tuple(test_prediction.shape) != expected_shape:
            raise RuntimeError(
                f"Smoke test output shape {tuple(test_prediction.shape)} != {expected_shape}"
            )
        if not torch.isfinite(test_prediction).all():
            raise FloatingPointError("Smoke test produced non-finite predictions")

        # Also run one real validation day through the SIH-depth interpolation
        # and metrics path; this is not a model-selection or test-set evaluation.
        smoke_val = evaluate_sih(model, VAL_DATES[:1], "SMOKE")
        smoke_val_rmse = float(smoke_val["overall"]["RMSE_weighted_C"])
        if not math.isfinite(smoke_val_rmse):
            raise FloatingPointError("Smoke validation returned non-finite SIH RMSE")

        smoke_report = {
            "status": "SMOKE_TEST_PASSED",
            "run_mode": RUN_MODE,
            "experiment_name": EXPERIMENT_NAME,
            "pattern_version": PATTERN_VERSION,
            "checkpoint_path": str(resume_path) if resume_path else None,
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)) if resume_path is not None else 0,
            "checkpoint_epoch_complete": bool(checkpoint.get("epoch_complete", True)) if resume_path is not None else None,
            "checkpoint_step_in_epoch": int(checkpoint.get("step_in_epoch", 0)) if resume_path is not None else None,
            "global_step_before_test": int(global_step),
            "test_date": preflight_day,
            "device": str(DEVICE),
            "prediction_shape": list(test_prediction.shape),
            "loss_all36_normalized_mse": float(test_loss.detach().float().cpu().item()),
            "gradient_norm_before_clip": float(torch.as_tensor(grad_norm).detach().float().cpu().item()),
            "forward_finite": bool(torch.isfinite(test_prediction).all().item()),
            "one_day_sih_validation_rmse_C": smoke_val_rmse,
            "one_day_sih_validation_mae_C": float(smoke_val["overall"]["MAE_weighted_C"]),
            "one_day_sih_validation_samples": int(smoke_val["sample_count"]),
            "backward_step": "PASS",
            "source_checkpoint_modified": False,
            "note": "The in-memory smoke-test optimizer step is not saved; any mounted source checkpoint remains unchanged.",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output_dir / "smoke_test_report.json", smoke_report)
        try:
            spec_path = output_dir / "experiment_spec.json"
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["status"] = "SMOKE_TEST_PASSED"
            spec["smoke_test_report"] = smoke_report
            atomic_json(spec_path, spec)
        except Exception as exc:
            print(f"[SMOKE TEST WARNING] Could not update experiment spec: {exc}", flush=True)
        print(f"Prediction shape   : {tuple(test_prediction.shape)}")
        print(f"Normalized loss    : {smoke_report['loss_all36_normalized_mse']:.6f}")
        print(f"Gradient norm      : {smoke_report['gradient_norm_before_clip']:.6f}")
        print(f"One-day SIH VAL RMSE: {smoke_val_rmse:.6f} °C")
        print("CHECKPOINT RESTORE : PASS" if resume_path is not None else "FRESH INITIALIZATION: PASS")
        print("REAL FORWARD       : PASS")
        print("BACKWARD + STEP    : PASS")
        print("SIH VALIDATION PATH: PASS")
        print("[SMOKE TEST PASSED] No checkpoint was overwritten. Set RUN_MODE='FULL' for the 11-hour run.", flush=True)
        return

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


        # Deterministic block shuffle: randomize block order but preserve
        # chronological order within each block for 14-day cache locality.
        shuffled_dates = block_shuffled_dates(
            TRAIN_DATES,
            block_size=TRAIN_DATE_BLOCK_SIZE,
            seed=SEED + epoch,
        )


        is_resuming_this_epoch = (resume_epoch == epoch)
        starting_step = resume_step_in_epoch if is_resuming_this_epoch else 0
        running_loss = resume_running_loss if is_resuming_this_epoch else 0.0

        if is_resuming_this_epoch and starting_step:
            print(
                f"  continuing epoch {epoch} after saved step "
                f"{starting_step}/{len(shuffled_dates)}",
                flush=True,
            )

        for step_in_epoch, day in enumerate(
            shuffled_dates,
            start=1,
        ):
            if step_in_epoch <= starting_step:
                continue

            if session_time_limit_reached():
                completed_steps = step_in_epoch - 1
                record_session_stop(
                    output_dir,
                    stage='training', epoch=epoch, global_step=global_step,
                    best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                    history=history, model=model, optimizer=optimizer, scaler=scaler,
                    step_in_epoch=completed_steps, running_loss=running_loss,
                    checkpoint_incomplete=True, save_checkpoint=True,
                )
                return

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


            # Fine-tuning cosine schedule only across epochs 19-25. The LR is
            # explicitly reset before every optimizer step, so restoring the
            # saved AdamW moments cannot accidentally restore the old LR.
            continuation_epochs = EPOCHS - EXPECTED_RESUME_EPOCH
            total_finetune_steps = max(continuation_epochs * len(TRAIN_DATES), 1)
            current_step_index = (epoch - (EXPECTED_RESUME_EPOCH + 1)) * len(TRAIN_DATES) + (step_in_epoch - 1)
            cosine_progress = min(
                max(current_step_index / max(total_finetune_steps - 1, 1), 0.0),
                1.0,
            )
            current_lr = CONTINUATION_LR_END + 0.5 * (CONTINUATION_LR_START - CONTINUATION_LR_END) * (
                1.0 + math.cos(math.pi * cosine_progress)
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            if step_in_epoch == starting_step + 1:
                print(
                    f"  [FINE-TUNE LR] epoch={epoch}/{EPOCHS}, "
                    f"step={step_in_epoch}/{len(shuffled_dates)}, lr={current_lr:.3e}",
                    flush=True,
                )

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

            # No per-step checkpoint I/O by default. Save at epoch boundaries;
            # the safe-stop handler below writes once if the deadline is reached.
            budget_hit = session_time_limit_reached()
            should_periodic_checkpoint = (
                CHECKPOINT_EVERY_STEPS > 0
                and step_in_epoch % CHECKPOINT_EVERY_STEPS == 0
            )
            if (not budget_hit) and (
                should_periodic_checkpoint or step_in_epoch == len(shuffled_dates)
            ):
                recovery_payload = checkpoint_payload(
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    best_val_rmse,
                    best_epoch,
                    history,
                    epoch_complete=False,
                    step_in_epoch=step_in_epoch,
                    running_loss=running_loss,
                )
                atomic_torch_save(
                    output_dir / "latest_model.pt",
                    recovery_payload,
                )

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
                    f" | lr={current_lr:.2e}"
                    f" | elapsed="
                    f"{(time.time()-epoch_start)/60.0:.1f}m",
                    flush=True,
                )

            if budget_hit:
                record_session_stop(
                    output_dir,
                    stage='training', epoch=epoch, global_step=global_step,
                    best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                    history=history, model=model, optimizer=optimizer, scaler=scaler,
                    step_in_epoch=step_in_epoch, running_loss=running_loss,
                    checkpoint_incomplete=True, save_checkpoint=True,
                )
                return


        train_loss = (
            running_loss
            / len(
                shuffled_dates
            )
        )


        # --------------------------------------------------------------------
        # VALIDATION — ONLY SIH 15 DEPTHS
        # --------------------------------------------------------------------

        try:
            validation = evaluate_sih(
                model,
                VAL_DATES,
                "VAL",
            )
        except SessionTimeLimitReached:
            record_session_stop(
                output_dir,
                stage='validation', epoch=epoch, global_step=global_step,
                best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                history=history, model=model, optimizer=optimizer, scaler=scaler,
                step_in_epoch=len(shuffled_dates), running_loss=running_loss,
                checkpoint_incomplete=True, save_checkpoint=True,
            )
            return


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
        # CHECKPOINT FIRST — metric/reporting failures must never lose weights.
        # --------------------------------------------------------------------

        latest_payload = checkpoint_payload(
            model, optimizer, scaler, epoch, global_step,
            best_val_rmse, best_epoch, history,
            epoch_complete=True,
            step_in_epoch=len(shuffled_dates),
            running_loss=running_loss,
        )
        atomic_torch_save(output_dir / "latest_model.pt", latest_payload)

        if is_best:
            best_payload = checkpoint_payload(
                model, optimizer, scaler, epoch, global_step,
                best_val_rmse, best_epoch, history,
                epoch_complete=True,
                step_in_epoch=len(shuffled_dates),
                running_loss=running_loss,
            )
            atomic_torch_save(output_dir / "best_model.pt", best_payload)
            print(
                f"\n[BEST] epoch={epoch} "
                f"| SIH-15 VAL RMSE={best_val_rmse:.6f} °C",
                flush=True,
            )

        # Keep training history even if a CSV export has a schema/disk problem.
        try:
            atomic_json(output_dir / "training_history.json", history)
        except Exception as exc:
            print(f"[SAVE WARNING] training_history.json failed: {exc}", flush=True)

        try:
            save_result(validation, output_dir)
        except Exception as exc:
            print(
                f"[SAVE WARNING] Validation export failed after checkpoints were saved: {exc}",
                flush=True,
            )

        # Clear the intra-epoch resume marker only after a completed epoch is saved.
        if resume_epoch == epoch:
            resume_epoch = None
            resume_step_in_epoch = 0
            resume_running_loss = 0.0

        total_hours = session_elapsed_seconds() / 3600.0


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


        if session_time_limit_reached():
            record_session_stop(
                output_dir,
                stage='post_epoch', epoch=epoch, global_step=global_step,
                best_val_rmse=best_val_rmse, best_epoch=best_epoch,
                history=history, model=model, optimizer=optimizer, scaler=scaler,
                step_in_epoch=len(shuffled_dates), running_loss=running_loss,
                checkpoint_incomplete=False, save_checkpoint=False,
            )
            return


    # ------------------------------------------------------------------------
    # Restore best checkpoint
    # ------------------------------------------------------------------------

    best_path = (
        output_dir
        / "best_model.pt"
    )

    if best_epoch > 0 and not best_path.is_file():
        raise RuntimeError(
            f"Missing best checkpoint {best_path}. Refusing to report final test "
            "metrics from a non-best model; recover/attach the checkpoint for "
            f"best_epoch={best_epoch} and resume."
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

    try:
        test_result = evaluate_sih(
            model,
            TEST_DATES,
            "TEST",
        )
    except SessionTimeLimitReached:
        last_epoch = int(history[-1]['epoch']) if history else max(0, start_epoch - 1)
        record_session_stop(
            output_dir,
            stage='test', epoch=last_epoch, global_step=global_step,
            best_val_rmse=best_val_rmse, best_epoch=best_epoch,
            history=history, model=model, optimizer=optimizer, scaler=scaler,
            step_in_epoch=len(TRAIN_DATES),
            running_loss=0.0,
            checkpoint_incomplete=False, save_checkpoint=False,
        )
        return


    try:
        save_result(test_result, output_dir)
    except Exception as exc:
        print(f"[SAVE WARNING] Test metric export failed; continuing to final report: {exc}", flush=True)
        try:
            atomic_json(output_dir / "test_metrics.json", test_result)
        except Exception as json_exc:
            print(f"[SAVE ERROR] Could not write test_metrics.json: {json_exc}", flush=True)


    # ------------------------------------------------------------------------
    # Final artifacts
    # ------------------------------------------------------------------------

    total_runtime_seconds = time.monotonic() - run_start


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
        "TAFU-NET NATIVE36 REAL RUN COMPLETE"
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
        "NO VERTICAL INTERPOLATION IN TRAINING; SIH interpolation is evaluation-only."
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
