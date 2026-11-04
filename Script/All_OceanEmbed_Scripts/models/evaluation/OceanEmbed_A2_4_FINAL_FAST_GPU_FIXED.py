# =============================================================================
# OceanEmbed A2.4
# Frozen B6 Epoch-10 Inference-Time Modality Ablation
#
# IMPORTANT:
#   - NO TRAINING
#   - NO BACKWARD
#   - NO OPTIMIZER STEP
#   - SAME EPOCH-10 CHECKPOINT FOR ALL ARMS
#   - SAME VALIDATION LOADER / SPLIT
#   - ONLY INPUT OBSERVABILITY IS CHANGED
# =============================================================================

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from datetime import datetime
import calendar

import numpy as np
import torch


# =============================================================================
# 1. PATHS
# =============================================================================

CHECKPOINT = Path(
    "/kaggle/input/datasets/lalithiray/b6-epoch-10/latest_model.pt"
)

OUTPUT_ROOT = Path("/kaggle/working/A2_4_results")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

assert CHECKPOINT.is_file(), f"Checkpoint not found: {CHECKPOINT}"


# =============================================================================
# 2. EXACT B6 TRAINER PATH — NO RECURSIVE /kaggle/input SCAN
# =============================================================================

B6_FILE = Path(
    "/kaggle/input/datasets/lalithiray/"
    "oceanembed-kaggletrainer-b6-convlstm/"
    "OceanEmbed_KaggleTrainer_B6_ConvLSTM_15Epoch_LatestOnlyResume_BestRestore.py"
)

if not B6_FILE.is_file():
    raise FileNotFoundError(
        f"Exact B6 trainer was not found:\n{B6_FILE}"
    )

print("=" * 80)
print("A2.4 — FROZEN B6 MODALITY ABLATION")
print("=" * 80)
print("B6 source :", B6_FILE)
print("Checkpoint:", CHECKPOINT)
print("Output    :", OUTPUT_ROOT)

spec = importlib.util.spec_from_file_location(
    "oceanembed_b6_a24_source",
    str(B6_FILE),
)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not import B6 source: {B6_FILE}")

b6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b6
spec.loader.exec_module(b6)

print("[PASS] B6 source imported without executing training.")

# =============================================================================
# 3. MATERIALIZE THE SAME EMBEDDED CANONICAL RUNTIME
# =============================================================================

runtime = b6._prepare_kaggle_runtime()

print("[PASS] Embedded canonical runtime materialized.")
print("  loader   :", runtime["loader_path"])
print("  backbone :", runtime["backbone_path"])
print("  framework:", runtime["framework_path"])


# =============================================================================
# 4. DISCOVER + BUILD THE SAME CANONICAL COMPATIBILITY TREE
# =============================================================================

# IMPORTANT:
# Search only the known OceanEmbed ModelData dataset.
# Do NOT recursively scan all of /kaggle/input.
MODEL_DATA_ROOT = Path(
    "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"
)

if not MODEL_DATA_ROOT.is_dir():
    raise FileNotFoundError(
        f"OceanEmbed ModelData root not found:\n{MODEL_DATA_ROOT}"
    )

# =============================================================================
# 4. DIRECT DATASET PATHS + CANONICAL COMPATIBILITY TREE
# NO RECURSIVE DISCOVERY / NO /kaggle/input SCAN
# =============================================================================
#
# These paths are fixed from the audited OceanEmbed ModelData layout.
# We validate exact files/directories and construct the canonical bundle
# expected by the frozen B6 Backbone. Large files are linked, not copied.
# =============================================================================

MODEL_DATA_ROOT = Path(
    "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"
)

if not MODEL_DATA_ROOT.is_dir():
    raise FileNotFoundError(
        f"OceanEmbed ModelData root not found:\n{MODEL_DATA_ROOT}"
    )

# -------------------------------------------------------------------------
# Exact predictor files
# -------------------------------------------------------------------------

KNOWN_PREDICTORS = {
    "Duacs_Oisst_2012_2024_FeaturesV1.nc":
        MODEL_DATA_ROOT / "Duacs_Oisst_2012_2024_FeaturesV1.nc",

    "Oisss_GlobCurrent_2012_2024_FeaturesV1.nc":
        MODEL_DATA_ROOT / "Oisss_GlobCurrent_2012_2024_FeaturesV1.nc",

    "Ccmp_Oscar_2012_2024_100x240_FeaturesV1.nc":
        MODEL_DATA_ROOT / "Ccmp_Oscar_2012_2024_100x240_FeaturesV1.nc",
}

# -------------------------------------------------------------------------
# Exact GEBCO static file
# -------------------------------------------------------------------------

STATIC_PATH = (
    MODEL_DATA_ROOT /
    "GEBCO_OceanEmbed_Static_0p25deg_FINAL.nc"
)

# -------------------------------------------------------------------------
# Exact GLORYS source directory
# -------------------------------------------------------------------------

GLORYS_SOURCE_DIR = (
    MODEL_DATA_ROOT /
    "target" /
    "target" /
    "glorys"
)

# -------------------------------------------------------------------------
# Exact frozen Backbone artifacts directory
# -------------------------------------------------------------------------

BACKBONE_ARTIFACTS_DIR = (
    MODEL_DATA_ROOT /
    "backbone_artifacts"
)

# -------------------------------------------------------------------------
# Fast existence checks — no directory-tree traversal
# -------------------------------------------------------------------------

missing_paths = [
    str(path)
    for path in (
        list(KNOWN_PREDICTORS.values())
        + [STATIC_PATH, GLORYS_SOURCE_DIR, BACKBONE_ARTIFACTS_DIR]
    )
    if not path.exists()
]

if missing_paths:
    raise FileNotFoundError(
        "Known OceanEmbed paths are missing:\n"
        + "\n".join(missing_paths)
    )

# -------------------------------------------------------------------------
# Build the exact 2012-01 through 2024-12 monthly GLORYS file list.
# This avoids scanning the GLORYS directory.
# -------------------------------------------------------------------------

glorys_files = []

for year in range(2012, 2025):
    for month in range(1, 13):
        last_day = calendar.monthrange(year, month)[1]

        fname = (
            f"GLORYS_thetao_"
            f"{year:04d}{month:02d}01_"
            f"{year:04d}{month:02d}{last_day:02d}_"
            f"NIO_0p25deg_15depth.nc"
        )

        path = GLORYS_SOURCE_DIR / fname

        if not path.is_file():
            raise FileNotFoundError(
                f"Expected GLORYS monthly file is missing:\n{path}"
            )

        glorys_files.append(path)

if len(glorys_files) != 156:
    raise RuntimeError(
        f"Expected exactly 156 GLORYS files, "
        f"constructed {len(glorys_files)}."
    )

# -------------------------------------------------------------------------
# Verify the exact frozen Backbone artifact set.
# -------------------------------------------------------------------------

missing_artifacts = [
    fname
    for fname in b6.BACKBONE_ARTIFACTS
    if not (BACKBONE_ARTIFACTS_DIR / fname).is_file()
]

if missing_artifacts:
    raise FileNotFoundError(
        "Missing required frozen Backbone artifacts:\n"
        + "\n".join(missing_artifacts)
    )

# -------------------------------------------------------------------------
# Construct the exact dictionary expected by the canonical compatibility
# builder. No discovery function is called.
# -------------------------------------------------------------------------

discovery = {
    "predictor_files": KNOWN_PREDICTORS,
    "static_file": STATIC_PATH,
    "glorys_files": glorys_files,
    "backbone_artifacts_dir": BACKBONE_ARTIFACTS_DIR,
    "target_zip": None,
    "backbone_zip": None,
}

print(
    "[PASS] Known dataset paths validated — "
    "NO recursive discovery used.",
    flush=True,
)
print("  predictors        :", len(KNOWN_PREDICTORS), flush=True)
print("  GLORYS files      :", len(glorys_files), flush=True)
print("  GEBCO             :", STATIC_PATH.name, flush=True)
print(
    "  backbone artifacts:",
    len(b6.BACKBONE_ARTIFACTS),
    flush=True,
)

# -------------------------------------------------------------------------
# Canonical B6 compatibility tree.
# The builder creates directories and symlinks only; it does not copy the
# multi-GB predictor/GLORYS NetCDF files.
# -------------------------------------------------------------------------

compat = b6.build_compatibility_tree(
    discovery,
    Path("/kaggle/working"),
)

BUNDLE_ROOT = Path(compat["bundle_root"])

if not BUNDLE_ROOT.is_dir():
    raise RuntimeError(
        f"Canonical compatibility bundle was not created:\n{BUNDLE_ROOT}"
    )

print(
    "[PASS] Canonical compatibility tree:",
    BUNDLE_ROOT,
    flush=True,
)


# =============================================================================
# 5. IMPORT THE SAME FRAMEWORK
# =============================================================================

framework = b6._import_module(
    Path(runtime["framework_path"]),
    "oceanembed_a24_framework",
)

b6.validate_framework_contract(framework)

print("[PASS] Framework contract:", framework.FRAMEWORK_CONTRACT)


# =============================================================================
# 6. BUILD THE FROZEN BACKBONE ADAPTER
#
# This loads the frozen Backbone state and validation loader.
# No training occurs here.
# =============================================================================

cfg = framework.TrainConfig(
    dataset_root=str(BUNDLE_ROOT),
    backbone_path=str(runtime["backbone_path"]),
    loader_path=str(runtime["loader_path"]),
    model_module=str(B6_FILE),
    model_class="Model",

    run_root=str(OUTPUT_ROOT),
    experiment_name="A2.4_B6_EPOCH10",

    epochs=1,
    batch_size=1,
    num_workers=4,

    learning_rate=1e-4,
    weight_decay=1e-4,
    grad_clip_norm=1.0,
    patience=1,

    device="auto",
    amp=True,
    seed=42,

    log_every=1000,
    checkpoint_every=1,
    checkpoint_every_steps=0,
    resume=None,
)

cfg.validate()

adapter = framework.BackboneAdapter(cfg)

b6.validate_backbone_runtime(adapter)
norm_report = b6.validate_runtime_normalization(adapter)

print("[PASS] Frozen Backbone runtime validated.")
print("  bundle fingerprint:", b6.EXPECTED_BUNDLE_FP)
print("  backbone version   :", b6.EXPECTED_BACKBONE_VERSION)
print("  backbone contract  :", b6.EXPECTED_BACKBONE_CONTRACT)
print("  framework contract :", b6.EXPECTED_FRAMEWORK_CONTRACT)


# =============================================================================
# 7. BUILD EXACT B6 MODEL
# =============================================================================

model = b6.Model(adapter.contract)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("[PASS] B6 model constructed.")
print("  model:", b6.MODEL_ID)
print("  parameters:", parameter_count)

if parameter_count != 177071:
    raise RuntimeError(
        f"Unexpected B6 parameter count: {parameter_count}"
    )


# =============================================================================
# 8. LOAD THE EPOCH-10 CHECKPOINT
# =============================================================================

checkpoint_report = b6._validate_resume_checkpoint_metadata(
    CHECKPOINT,
    model,
)

print("[PASS] Epoch-10 checkpoint metadata validated.")
print(json.dumps(checkpoint_report, indent=2))


payload = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

if "model_state_dict" not in payload:
    raise RuntimeError("Checkpoint has no model_state_dict.")

model.load_state_dict(
    payload["model_state_dict"],
    strict=True,
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "A2.4 requires the Kaggle GPU for this run, but "
        "torch.cuda.is_available() is False."
    )

device = framework.select_device("cuda")

if device.type != "cuda":
    raise RuntimeError(f"A2.4 expected CUDA device, got: {device}")

model = model.to(device)
model.eval()

print("[PASS] Epoch-10 B6 weights loaded.")
print("  device:", device)


# =============================================================================
# 9. FIXED A2.4 DOCUMENTATION
# =============================================================================

REFERENCE_NOTE = (
    "The reference checkpoint is epoch 10 (val_loss = 0.9249299274), "
    "not the historical best (val_loss = 0.9221295965, reached at or before "
    "the epoch-5 continuation point, checkpoint not retained). Absolute RMSE "
    "values are therefore referenced to a slightly sub-optimal B6 state. "
    "Comparisons between A2.4 arms and P1 are internally valid because all "
    "use the same epoch-10 reference. Comparisons against the originally "
    "reported B6 number must be treated as approximate."
)


# =============================================================================
# 10. LOCKED A2.4 ARMS
# =============================================================================

ARMS = {
    "Full": [],

    "Minus_Thermal": [4, 5, 6, 8],

    "Minus_Sea_level": [0, 1],

    "Minus_Salinity": [9, 10, 11, 15],

    # LOCKED OPERATIONAL GROUPING
    "Minus_Currents": [
        2, 3, 7,
        12, 13, 14, 16,
        19, 20, 22, 23,
    ],

    "Minus_Wind": [17, 18, 21],

    "Minus_Static": [24, 25, 26, 27, 28, 29],
}


# =============================================================================
# 11. HUMAN-READABLE CHANNEL MAP
# =============================================================================

print("\nLOCKED A2.4 ARMS")
print("-" * 80)

for name, idx in ARMS.items():
    if not idx:
        print(f"{name:22s}: none")
    else:
        names = []

        for i in idx:
            if i < 24:
                names.append(
                    f"{i}:{b6.DYNAMIC_FEATURE_NAMES[i]}"
                )
            else:
                sidx = i - 24
                names.append(
                    f"{i}:{b6.STATIC_FEATURE_NAMES[sidx]}"
                )

        print(f"{name:22s}: {', '.join(names)}")


# =============================================================================
# 12. REGIONAL DEFINITIONS
#
# Locked project geography:
#   Full domain: 5–30 N, 45–105 E
#   Arabian Sea: 5–30 N, 45–80 E
#   Bay of Bengal: 5–30 N, 80–105 E
#
# Current model grid is 100 x 240 at 0.25 degree spacing.
# =============================================================================

lat = (
    5.0
    + 0.25 * np.arange(b6.GRID_H)
)

lon = (
    45.0
    + 0.25 * np.arange(b6.GRID_W)
)

LAT, LON = np.meshgrid(
    lat,
    lon,
    indexing="ij",
)

REGION_MASKS = {
    "Arabian_Sea": (
        (LAT >= 5.0)
        & (LAT <= 30.0)
        & (LON >= 45.0)
        & (LON < 80.0)
    ),

    "Bay_of_Bengal": (
        (LAT >= 5.0)
        & (LAT <= 30.0)
        & (LON >= 80.0)
        & (LON <= 105.0)
    ),
}


# =============================================================================
# 13. SEASON DEFINITIONS
#
# Existing OceanEmbed seasonal convention:
#   Monsoon     = June–September
#   Non-monsoon = October–May
# =============================================================================

def season_name(date_string: str) -> str:
    month = datetime.fromisoformat(
        str(date_string)
    ).month

    if month in (6, 7, 8, 9):
        return "Monsoon"

    return "Non_monsoon"


# =============================================================================
# 14. SIMPLE RMSE ACCUMULATOR
# =============================================================================

class SimpleAccumulator:

    def __init__(self):
        self.sse = 0.0
        self.sae = 0.0
        self.n = 0

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ):
        p = pred[mask]
        t = target[mask]

        if p.size == 0:
            return

        err = p.astype(np.float64) - t.astype(np.float64)

        self.sse += float(np.square(err).sum())
        self.sae += float(np.abs(err).sum())
        self.n += int(err.size)

    def rmse(self):
        if self.n == 0:
            return float("nan")

        return math.sqrt(self.sse / self.n)

    def mae(self):
        if self.n == 0:
            return float("nan")

        return self.sae / self.n


# =============================================================================
# 15. CREATE METRIC STRUCTURES
# =============================================================================

def make_arm_metrics():

    return {
        "overall": framework.MetricAccumulator(
            b6.N_DEPTH
        ),

        "depth_bands": {
            "0_30m": framework.MetricAccumulator(
                b6.N_DEPTH
            ),
            "50_200m": framework.MetricAccumulator(
                b6.N_DEPTH
            ),
            "300_1000m": framework.MetricAccumulator(
                b6.N_DEPTH
            ),
        },

        "regions": {
            name: SimpleAccumulator()
            for name in REGION_MASKS
        },

        "seasons": {
            "Monsoon": SimpleAccumulator(),
            "Non_monsoon": SimpleAccumulator(),
        },
    }


METRICS = {
    arm: make_arm_metrics()
    for arm in ARMS
}


# =============================================================================
# 16. DAILY SSE STORAGE FOR TEMPORAL-BLOCK BOOTSTRAP
# =============================================================================

DAILY = {
    arm: {
        "dates": [],
        "sse": [],
        "n": [],
    }
    for arm in ARMS
}


# =============================================================================
# 17. DEPTH BAND MASKS
# =============================================================================

DEPTHS = np.asarray(
    b6.DEPTHS_M,
    dtype=float,
)

BAND_DEPTHS = {
    "0_30m": np.where(
        (DEPTHS >= 0)
        & (DEPTHS <= 30)
    )[0],

    "50_200m": np.where(
        (DEPTHS >= 50)
        & (DEPTHS <= 200)
    )[0],

    "300_1000m": np.where(
        (DEPTHS >= 300)
        & (DEPTHS <= 1000)
    )[0],
}


# =============================================================================
# 18. ARM FORWARD
# =============================================================================

@torch.inference_mode()
def run_arm(
    batch,
    arm_name,
    channel_indices,
):
    """
    Run exactly one frozen B6 forward pass.

    Dynamic ablations:
        x_dynamic = 0
        x_dynamic_valid = False

    Static ablation:
        preserve original structural ocean mask FIRST,
        then zero all six static channels and invalidate them.
    """

    if not channel_indices:

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            return framework.forward_prediction(
                model,
                batch,
            )

    # -------------------------------------------------------------------------
    # Dynamic ablations
    # -------------------------------------------------------------------------

    dynamic_indices = [
        i for i in channel_indices
        if i < 24
    ]

    static_indices = [
        i - 24
        for i in channel_indices
        if i >= 24
    ]

    if dynamic_indices:

        original_x = batch["x_dynamic"][
            :, :, dynamic_indices
        ].clone()

        original_valid = batch["x_dynamic_valid"][
            :, :, dynamic_indices
        ].clone()

        batch["x_dynamic"][
            :, :, dynamic_indices
        ] = 0

        batch["x_dynamic_valid"][
            :, :, dynamic_indices
        ] = False

        try:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                pred = framework.forward_prediction(
                    model,
                    batch,
                )
        finally:
            batch["x_dynamic"][
                :, :, dynamic_indices
            ] = original_x

            batch["x_dynamic_valid"][
                :, :, dynamic_indices
            ] = original_valid

        return pred

    # -------------------------------------------------------------------------
    # Static ablation
    # -------------------------------------------------------------------------

    # IMPORTANT:
    # Derive structural ocean mask from the ORIGINAL static channels
    # BEFORE zeroing them.
    original_ocean_mask = b6.derive_ocean_mask_from_static(
        batch["x_static"],
        batch,
    ).bool()

    old_explicit_mask = batch.get(
        "ocean_mask",
        None,
    )

    batch["ocean_mask"] = original_ocean_mask

    original_x = batch["x_static"][
        :, static_indices
    ].clone()

    original_valid = batch["x_static_valid"][
        :, static_indices
    ].clone()

    batch["x_static"][
        :, static_indices
    ] = 0

    batch["x_static_valid"][
        :, static_indices
    ] = False

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            pred = framework.forward_prediction(
                model,
                batch,
            )
    finally:
        batch["x_static"][
            :, static_indices
        ] = original_x

        batch["x_static_valid"][
            :, static_indices
        ] = original_valid

        if old_explicit_mask is None:
            batch.pop("ocean_mask", None)
        else:
            batch["ocean_mask"] = old_explicit_mask

    return pred


# =============================================================================
# 19. METRIC UPDATE
# =============================================================================

def update_arm_metrics(
    arm,
    pred_phys,
    target_phys,
    loss_mask,
    dates,
):

    arm_metrics = METRICS[arm]

    # -------------------------------------------------------------------------
    # Canonical overall B6 metrics
    # -------------------------------------------------------------------------

    baseline = np.zeros_like(
        pred_phys,
        dtype=np.float32,
    )

    clim = None

    if adapter.backbone.cfg.target_mode == "anomaly":

        clim = np.stack(
            [
                adapter.norm.climatology[
                    datetime.fromisoformat(
                        str(d)
                    ).month - 1
                ]
                for d in dates
            ],
            axis=0,
        ).astype(np.float32)

    arm_metrics["overall"].update(
        pred_phys,
        target_phys,
        loss_mask,
        baseline,
        anomaly_climatology=clim,
    )

    # -------------------------------------------------------------------------
    # Depth bands
    # -------------------------------------------------------------------------

    for band_name, depth_idx in BAND_DEPTHS.items():

        band_mask = np.zeros_like(
            loss_mask,
            dtype=bool,
        )

        band_mask[:, depth_idx, :, :] = (
            loss_mask[:, depth_idx, :, :]
        )

        arm_metrics["depth_bands"][
            band_name
        ].update(
            pred_phys,
            target_phys,
            band_mask,
            baseline,
        )

    # -------------------------------------------------------------------------
    # Regions
    # -------------------------------------------------------------------------

    for region_name, region_grid in REGION_MASKS.items():

        region_mask = (
            loss_mask
            & region_grid[None, None, :, :]
        )

        arm_metrics["regions"][
            region_name
        ].update(
            pred_phys,
            target_phys,
            region_mask,
        )

    # -------------------------------------------------------------------------
    # Seasons
    # -------------------------------------------------------------------------

    for date_index, date in enumerate(dates):

        season = season_name(str(date))

        season_mask = np.zeros_like(
            loss_mask,
            dtype=bool,
        )

        season_mask[
            date_index
        ] = loss_mask[
            date_index
        ]

        arm_metrics["seasons"][
            season
        ].update(
            pred_phys[date_index:date_index + 1],
            target_phys[date_index:date_index + 1],
            season_mask[date_index:date_index + 1],
        )

    # -------------------------------------------------------------------------
    # Daily SSE / N for temporal-block bootstrap
    # -------------------------------------------------------------------------

    for i, date in enumerate(dates):

        m = loss_mask[i]

        err = (
            pred_phys[i].astype(np.float64)
            - target_phys[i].astype(np.float64)
        )

        valid_err = err[m]

        DAILY[arm]["dates"].append(
            str(date)
        )

        DAILY[arm]["sse"].append(
            float(np.square(valid_err).sum())
        )

        DAILY[arm]["n"].append(
            int(valid_err.size)
        )


# =============================================================================
# 20. RUN ALL SEVEN ARMS
#
# One validation pass over the loader.
# Each validation batch gets seven frozen forward passes.
# =============================================================================

print("\n")
print("=" * 80)
print("STARTING A2.4 INFERENCE")
print("=" * 80)

# Freeze gradients explicitly before inference.
for p in model.parameters():
    p.requires_grad_(False)

model.eval()

print("\nNO TRAINING WILL OCCUR.")
print("model.training =", model.training)
print("requires_grad  =", any(
    p.requires_grad for p in model.parameters()
))

validation_loader = adapter.val_loader

n_batches = 0
start_time = time.perf_counter()

for step, raw_batch in enumerate(
    validation_loader,
    start=1,
):

    n_batches += 1

    # Keep raw batch untouched for masks/dates.
    dates = framework.batch_dates(raw_batch)

    # One device copy.
    batch = framework.move_batch_to_device(
        raw_batch,
        device,
    )

    # ---------------------------------------------------------
    # Calculate every arm using the SAME original batch.
    # ---------------------------------------------------------

    for arm_name, channels in ARMS.items():

        pred = run_arm(
            batch,
            arm_name,
            channels,
        )

        if not torch.isfinite(pred).all():
            raise RuntimeError(
                f"Non-finite prediction in arm {arm_name}, "
                f"validation batch {step}"
            )

        # Convert anomaly output back to physical temperature units
        # exactly as the canonical B6 evaluator does.
        pred_phys = framework.inverse_targets(
            adapter.norm,
            pred,
            dates,
        )

        target_phys = framework.inverse_targets(
            adapter.norm,
            batch["y"],
            dates,
        )

        loss_mask = (
            raw_batch["loss_mask"]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

        update_arm_metrics(
            arm_name,
            pred_phys,
            target_phys,
            loss_mask,
            dates,
        )

    if step == 1 or step % 25 == 0:

        elapsed = time.perf_counter() - start_time
        rate = step / max(elapsed, 1e-9)

        print(
            f"[A2.4] validation batch "
            f"{step} | "
            f"{rate:.3f} batches/s",
            flush=True,
        )


elapsed = time.perf_counter() - start_time

print("\n")
print("=" * 80)
print("A2.4 INFERENCE FINISHED")
print("=" * 80)
print("Validation batches:", n_batches)
print("Elapsed seconds   :", round(elapsed, 2))


# =============================================================================
# 21. EXTRACT OVERALL METRICS
# =============================================================================

SUMMARY = {}

for arm_name in ARMS:

    overall = METRICS[arm_name]["overall"].finalize(
        adapter.depths_m,
        adapter.backbone.cfg.target_mode,
    )

    SUMMARY[arm_name] = {
        "RMSE": overall["overall"]["RMSE_weighted"],
        "MAE": overall["overall"]["MAE_weighted"],
        "Bias": overall["overall"]["Bias_weighted"],
        "Pearson_r": overall["overall"]["Pearson_r_weighted"],
        "ACC": overall["overall"][
            "ACC_spatial_daily_mean_weighted"
        ],
        "by_depth": overall["by_depth"],
    }


# =============================================================================
# 22. FULL REFERENCE CHECK
# =============================================================================

FULL_RMSE = SUMMARY["Full"]["RMSE"]

print("\n")
print("=" * 80)
print("FULL EPOCH-10 REFERENCE")
print("=" * 80)

print(
    f"Full RMSE    : {FULL_RMSE:.9f}"
)

print(
    f"Full MAE     : {SUMMARY['Full']['MAE']:.9f}"
)

print(
    f"Full Pearson : {SUMMARY['Full']['Pearson_r']:.9f}"
)

print(
    f"Full ACC     : {SUMMARY['Full']['ACC']:.9f}"
)

if not math.isfinite(FULL_RMSE):
    raise RuntimeError(
        "Full reference RMSE is non-finite. "
        "STOPPING before interpreting ablations."
    )


# =============================================================================
# 23. DEPTH-BY-DEPTH RESULTS
# =============================================================================

depth_rows = []

for arm_name in ARMS:

    for row in SUMMARY[arm_name]["by_depth"]:

        depth_rows.append({
            "arm": arm_name,
            "depth_m": row["depth_m"],
            "n_cells": row["n_cells"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "Bias": row["Bias"],
            "Pearson_r": row["Pearson_r"],
            "ACC": row["ACC_spatial_daily_mean"],
        })


with open(
    OUTPUT_ROOT / "depthwise_results.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=depth_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(depth_rows)


# =============================================================================
# 24. DEPTH-BAND RESULTS
# =============================================================================

band_rows = []

for arm_name in ARMS:

    for band_name in BAND_DEPTHS:

        metric = METRICS[arm_name]["depth_bands"][
            band_name
        ].finalize(
            adapter.depths_m,
            adapter.backbone.cfg.target_mode,
        )

        band_rows.append({
            "arm": arm_name,
            "band": band_name,
            "RMSE": metric["overall"]["RMSE_weighted"],
            "MAE": metric["overall"]["MAE_weighted"],
            "n_cells": metric["overall"]["n_cells"],
        })


with open(
    OUTPUT_ROOT / "depth_band_results.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=band_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(band_rows)


# =============================================================================
# 25. REGIONAL RESULTS
# =============================================================================

regional_rows = []

for arm_name in ARMS:

    for region_name in REGION_MASKS:

        acc = METRICS[arm_name]["regions"][
            region_name
        ]

        regional_rows.append({
            "arm": arm_name,
            "region": region_name,
            "RMSE": acc.rmse(),
            "MAE": acc.mae(),
            "n_cells": acc.n,
        })


with open(
    OUTPUT_ROOT / "regional_results.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=regional_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(regional_rows)


# =============================================================================
# 26. SEASONAL RESULTS
# =============================================================================

season_rows = []

for arm_name in ARMS:

    for season in (
        "Monsoon",
        "Non_monsoon",
    ):

        acc = METRICS[arm_name]["seasons"][
            season
        ]

        season_rows.append({
            "arm": arm_name,
            "season": season,
            "RMSE": acc.rmse(),
            "MAE": acc.mae(),
            "n_cells": acc.n,
        })


with open(
    OUTPUT_ROOT / "seasonal_results.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=season_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(season_rows)


# =============================================================================
# 27. OVERALL A2.4 SUMMARY + ΔRMSE
# =============================================================================

summary_rows = []

for arm_name in ARMS:

    s = SUMMARY[arm_name]

    summary_rows.append({
        "arm": arm_name,
        "RMSE": s["RMSE"],
        "Delta_RMSE_vs_Full": (
            s["RMSE"] - FULL_RMSE
        ),
        "MAE": s["MAE"],
        "Bias": s["Bias"],
        "Pearson_r": s["Pearson_r"],
        "ACC": s["ACC"],
    })


with open(
    OUTPUT_ROOT / "A2_4_summary.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# =============================================================================
# 28. TEMPORAL-BLOCK BOOTSTRAP
#
# Paired bootstrap:
#   same resampled validation dates for Full and each ablation.
#
# Block length = 7 validation samples/days.
# 1000 bootstrap replicates.
# =============================================================================

BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 7
BOOTSTRAP_SEED = 42


def block_bootstrap_delta(
    full_sse,
    full_n,
    arm_sse,
    arm_n,
    reps=1000,
    block_length=7,
    seed=42,
):

    full_sse = np.asarray(
        full_sse,
        dtype=np.float64,
    )

    full_n = np.asarray(
        full_n,
        dtype=np.float64,
    )

    arm_sse = np.asarray(
        arm_sse,
        dtype=np.float64,
    )

    arm_n = np.asarray(
        arm_n,
        dtype=np.float64,
    )

    n = len(full_sse)

    if not (
        len(full_n)
        == len(arm_sse)
        == len(arm_n)
        == n
    ):
        raise ValueError(
            "Bootstrap arrays have inconsistent lengths."
        )

    if n < block_length:
        raise ValueError(
            f"Only {n} validation dates; "
            f"block length={block_length}."
        )

    rng = np.random.default_rng(seed)

    deltas = np.empty(
        reps,
        dtype=np.float64,
    )

    n_blocks = int(
        math.ceil(
            n / block_length
        )
    )

    starts_max = n

    for r in range(reps):

        starts = rng.integers(
            0,
            starts_max,
            size=n_blocks,
        )

        indices = np.concatenate(
            [
                (start + np.arange(block_length))
                % n
                for start in starts
            ]
        )[:n]

        f_rmse = math.sqrt(
            full_sse[indices].sum()
            / max(
                full_n[indices].sum(),
                1,
            )
        )

        a_rmse = math.sqrt(
            arm_sse[indices].sum()
            / max(
                arm_n[indices].sum(),
                1,
            )
        )

        deltas[r] = a_rmse - f_rmse

    return {
        "delta_rmse": float(
            np.mean(deltas)
        ),
        "ci95_low": float(
            np.percentile(
                deltas,
                2.5,
            )
        ),
        "ci95_high": float(
            np.percentile(
                deltas,
                97.5,
            )
        ),
        "bootstrap_replicates": reps,
        "block_length": block_length,
        "n_validation_samples": n,
    }


bootstrap_rows = []

for arm_name in ARMS:

    if arm_name == "Full":
        continue

    result = block_bootstrap_delta(
        DAILY["Full"]["sse"],
        DAILY["Full"]["n"],
        DAILY[arm_name]["sse"],
        DAILY[arm_name]["n"],
        reps=BOOTSTRAP_REPS,
        block_length=BLOCK_LENGTH,
        seed=BOOTSTRAP_SEED,
    )

    bootstrap_rows.append({
        "arm": arm_name,
        **result,
    })


with open(
    OUTPUT_ROOT / "bootstrap_results.csv",
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=bootstrap_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(bootstrap_rows)


# =============================================================================
# 29. FINAL CONSOLIDATED REPORT
# =============================================================================

final_report = {
    "status": "PASS",

    "experiment": "A2.4",

    "reference_checkpoint": {
        "path": str(CHECKPOINT),
        "epoch": 10,
        "val_loss": 0.9249299274,
        "historical_best_val_loss": 0.9221295965,
        "historical_best_checkpoint_retained": False,
    },

    "reference_note": REFERENCE_NOTE,

    "model": {
        "model_id": b6.MODEL_ID,
        "model_version": b6.MODEL_VERSION,
        "model_contract": b6.MODEL_CONTRACT,
        "parameters": parameter_count,
        "input_channels": b6.TOTAL_INPUT_CHANNELS,
        "window_days": b6.EXPECTED_WINDOW_DAYS,
        "depths_m": b6.DEPTHS_M,
        "bundle_fingerprint": b6.EXPECTED_BUNDLE_FP,
    },

    "validation": {
        "batches": n_batches,
        "batch_size": 1,
        "device": str(device),
        "elapsed_seconds": elapsed,
    },

    "arms": ARMS,

    "overall": summary_rows,

    "regional_definitions": {
        "Arabian_Sea": {
            "lat": [5.0, 30.0],
            "lon": [45.0, 80.0],
        },
        "Bay_of_Bengal": {
            "lat": [5.0, 30.0],
            "lon": [80.0, 105.0],
        },
    },

    "season_definitions": {
        "Monsoon": [6, 7, 8, 9],
        "Non_monsoon": [
            10, 11, 12, 1, 2, 3, 4, 5
        ],
    },

    "bootstrap": {
        "replicates": BOOTSTRAP_REPS,
        "block_length": BLOCK_LENGTH,
        "seed": BOOTSTRAP_SEED,
    },

    "files": {
        "summary": str(
            OUTPUT_ROOT / "A2_4_summary.csv"
        ),
        "depthwise": str(
            OUTPUT_ROOT / "depthwise_results.csv"
        ),
        "depth_bands": str(
            OUTPUT_ROOT / "depth_band_results.csv"
        ),
        "regional": str(
            OUTPUT_ROOT / "regional_results.csv"
        ),
        "seasonal": str(
            OUTPUT_ROOT / "seasonal_results.csv"
        ),
        "bootstrap": str(
            OUTPUT_ROOT / "bootstrap_results.csv"
        ),
    },
}

with open(
    OUTPUT_ROOT / "A2_4_final_report.json",
    "w",
) as f:
    json.dump(
        final_report,
        f,
        indent=2,
        default=str,
    )


# =============================================================================
# 30. PRINT THE RESULT WE ACTUALLY CARE ABOUT
# =============================================================================

print("\n")
print("=" * 100)
print("A2.4 RESULTS")
print("=" * 100)

print(
    f"{'ARM':25s} "
    f"{'RMSE':>14s} "
    f"{'ΔRMSE':>14s}"
)

print("-" * 100)

for row in summary_rows:

    print(
        f"{row['arm']:25s} "
        f"{row['RMSE']:14.6f} "
        f"{row['Delta_RMSE_vs_Full']:14.6f}"
    )


print("\n")
print("=" * 100)
print("BOOTSTRAP 95% CI FOR ΔRMSE")
print("=" * 100)

for row in bootstrap_rows:

    print(
        f"{row['arm']:25s} "
        f"ΔRMSE={row['delta_rmse']:+.6f} "
        f"95% CI=["
        f"{row['ci95_low']:+.6f}, "
        f"{row['ci95_high']:+.6f}]"
    )


print("\n")
print("=" * 100)
print("A2.4 COMPLETE")
print("=" * 100)

print(
    "Results written to:",
    OUTPUT_ROOT,
)

print("\nREFERENCE NOTE:")
print(REFERENCE_NOTE)