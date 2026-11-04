#!/usr/bin/env python3
"""
OceanEmbed Baseline 6 — Mask-Aware Convolutional LSTM
======================================================

Scientific purpose
------------------
B6 tests joint spatial + temporal recurrent learning for the gridded
OceanEmbed reconstruction problem.

Unlike B4 (spatial CNN -> vector LSTM), B6 keeps the recurrent hidden and
cell states spatially structured throughout the temporal recurrence.

Architecture
------------
14 daily 36-channel maps
    -> masked 3x3 spatial projection (36 -> 32) at 100x240
    -> mask-aware 2x2 spatial pooling to 50x120
    -> ConvLSTM layer 1 (32 -> 32, 3x3)
    -> ConvLSTM layer 2 (32 -> 32, 3x3)
    -> final hidden state [B,32,50,120]
    -> 2x upsample to 100x240
    -> fuse with final-day high-resolution encoder features
    -> masked 3x3 projection
    -> masked 1x1 depth head
    -> [B,15,100,240]

Masking
-------
Invalid predictor values are masked before spatial/gate convolutions.
Permanent land is represented by the frozen ocean_fraction-derived mask.
Hidden and cell states are suppressed outside permanent ocean structure.
The target/loss mask is never used as an input.

Notebook behavior
-----------------
Paste this complete file into ONE Kaggle notebook cell and run it.
The trainer is notebook-safe, validates the pinned data bundle, runs a
real-data integration test, trains with the canonical framework, packages
checkpoints/history/metrics, and explicitly terminates the Python process
with exit code 0 after all outputs are written.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import threading
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn




# ============================================================================
# Kaggle / standalone trainer configuration
# ============================================================================
# Only edit this section for normal Kaggle use.

RUN_MODE                     = "train"   # "train", "self-test", "test"
MODEL_NAME                   = "CONVLSTM_MASKED_B6"
EPOCHS                       = 15
BATCH_SIZE                   = 1
LEARNING_RATE                = 1e-4
WEIGHT_DECAY                = 1e-4
PATIENCE                     = 20
NUM_WORKERS                  = 4
GRAD_CLIP_NORM              = 1.0
SEED                        = 42
LOG_EVERY                   = 10
CHECKPOINT_EVERY            = 1
CHECKPOINT_EVERY_STEPS      = 500
AMP_ENABLED                 = True

# B6 starts from scratch by default. To resume, provide a B6 latest_model.pt.
# The checkpoint payload is sufficient; the optional .metadata.json sidecar is no
# longer required. Cross-baseline checkpoints are explicitly rejected.
RESUME_FROM                = None
AUTO_RESUME_COMPATIBLE      = True
FORCE_FRESH_START           = False

# Historical best checkpoint preservation for resumed runs.
# When the continuation checkpoint itself is the historical validation-best
# state (the original B6 Epoch-5 latest_model.pt case), the trainer recreates
# best_model.pt in the NEW run directory before training/test evaluation.
# For a later resume checkpoint, set HISTORICAL_BEST_CHECKPOINT explicitly.
HISTORICAL_BEST_CHECKPOINT  = None
PRESERVE_HISTORICAL_BEST    = True
BEST_MATCH_TOLERANCE        = 1e-8

# Runtime-only options; they do not alter the scientific model contract.
ENABLE_TF32                 = True
CUDNN_BENCHMARK             = True

# Long-run safety / checkpoint capture.
ENABLE_RUNTIME_MONITOR       = True
CHECKPOINT_POLL_SECONDS      = 2.0
HEARTBEAT_INTERVAL_SECONDS   = 60.0
HEARTBEAT_PRINT_INTERVAL_SEC = 300.0
STALL_DIAGNOSTIC_SECONDS     = 900.0
SESSION_SOFT_LIMIT_HOURS     = 11.5
SESSION_HARD_LIMIT_HOURS     = 12.0
SNAPSHOT_CHECKPOINTS         = True

# Required pre-training safety gate.
RUN_PRETRAIN_INTEGRATION_TEST = True

# Kaggle output root. The trainer creates OceanEmbed_Runs underneath this.
WORKING_DIR                 = "/kaggle/working"

# Set to None to auto-discover the attached OceanEmbed_ModelData_V1 dataset.
DATASET_ROOT                = None

# For explicit continuation runs, the checkpoint path can be changed above.
# latest_model.pt is validated against the frozen B6 experiment/model identity,
# model contract/hash, backbone fingerprint, framework contract, target depths,
# and checkpoint state before loading. The sidecar is optional.

# ============================================================================
# End Kaggle / standalone trainer configuration
# ============================================================================


# ============================================================================
# Frozen scientific contract
# ============================================================================

MODEL_VERSION = "6.0.0"
MODEL_CONTRACT = "CONVLSTMB6.0"
MODEL_ID = "CONVLSTM_MASKED_B6"
EXPERIMENT_ID = "B6"

EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)
EXPECTED_BACKBONE_VERSION = "1.0.0"
EXPECTED_BACKBONE_CONTRACT = "DLBACKBONEV1.0"
EXPECTED_FRAMEWORK_CONTRACT = "TRAINV1.1"
CHECKPOINT_IDENTITY_SCHEMA = "OCEANEMBED-CHECKPOINT-IDENTITY-V1"

GRID_H, GRID_W = 100, 240
EXPECTED_WINDOW_DAYS = 14
DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2
TOTAL_INPUT_CHANNELS = (
    DYNAMIC_CHANNELS
    + STATIC_CHANNELS
    + SPATIAL_CONTEXT_CHANNELS
    + TEMPORAL_CONTEXT_CHANNELS
)
N_DEPTH = 15

DEPTHS_M = [
    0.0, 5.0, 10.0, 20.0, 30.0,
    50.0, 75.0, 100.0, 125.0, 150.0,
    200.0, 300.0, 500.0, 700.0, 1000.0,
]

DYNAMIC_FEATURE_NAMES = [
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

STATIC_FEATURE_NAMES = [
    "depth_mean_m",
    "depth_std_m",
    "ocean_fraction",
    "shallow_cell_fraction_0_200m",
    "bathymetric_slope_deg",
    "distance_to_coast_km",
]

SPATIAL_CONTEXT_NAMES = [
    "sin_lat",
    "cos_lat",
    "sin_lon",
    "cos_lon",
]

TEMPORAL_CONTEXT_NAMES = [
    "sin_annual",
    "cos_annual",
]

SOURCE_SLICES = {
    "duacs_oisst": [0, 9],
    "oisss_globcurrent": [9, 17],
    "ccmp_oscar": [17, 24],
}

# Architecture
STEM_CHANNELS = 32
RESIDUAL_BLOCKS = 1
PROJECTION_CHANNELS = 16
SPATIAL_KERNEL = 3
SPATIAL_PADDING = 1
HEAD_KERNEL = 1

# Shared OceanEmbed baseline recipe defaults
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 15
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 42
DEFAULT_LOG_EVERY = 10
DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_CHECKPOINT_EVERY_STEPS = 500

# Frozen normalized static ocean_fraction representation.
# Production normalization maps:
#   z = (raw_ocean_fraction - mean) / std
# with raw land = 0. These values come from the frozen pinned predictor bundle
# normalization for the pinned ModelData_V1 bundle.
OCEAN_FRACTION_STATIC_INDEX = 2
OCEAN_FRACTION_TRAIN_MEAN = 0.494109282395264
OCEAN_FRACTION_TRAIN_STD = 0.49318098582328196
EXPECTED_LAND_OCEAN_FRACTION_Z = (
    (0.0 - OCEAN_FRACTION_TRAIN_MEAN) / OCEAN_FRACTION_TRAIN_STD
)
OCEAN_FRACTION_Z_TOL = 5e-5

# Partial-convolution numerical policy.
VALID_FRACTION_EPS = 1e-6


class B6Error(RuntimeError):
    """Base B6 error."""


class ContractError(B6Error):
    """Frozen contract mismatch."""


class IntegrationTestError(B6Error):
    """B6 integration/self-test failure."""


# ============================================================================
# Generic utilities
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return sanitize_json(value.item())
        return sanitize_json(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_json(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def contract_hash(contract: Dict[str, Any]) -> str:
    raw = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


# ============================================================================
# Frozen model-input contract validation
# ============================================================================

def validate_contract(contract: Dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise ContractError("Backbone model_input_contract() must be a dict")

    dynamic = contract.get("dynamic", {})
    static = contract.get("static", {})
    target = contract.get("target", {})
    context = contract.get("context", {})

    expected_dynamic_shape = ["B", "T", DYNAMIC_CHANNELS, GRID_H, GRID_W]
    if list(dynamic.get("shape", [])) != expected_dynamic_shape:
        raise ContractError(
            f"Dynamic shape mismatch: {dynamic.get('shape')!r} != {expected_dynamic_shape!r}"
        )
    if list(dynamic.get("features", [])) != DYNAMIC_FEATURE_NAMES:
        raise ContractError("Dynamic feature names/order do not match frozen OceanEmbed contract")
    if dynamic.get("source_slices") != SOURCE_SLICES:
        raise ContractError("Dynamic source slices do not match frozen contract")

    expected_static_shape = ["B", STATIC_CHANNELS, GRID_H, GRID_W]
    if list(static.get("shape", [])) != expected_static_shape:
        raise ContractError(
            f"Static shape mismatch: {static.get('shape')!r} != {expected_static_shape!r}"
        )
    if list(static.get("features", [])) != STATIC_FEATURE_NAMES:
        raise ContractError("Static feature names/order do not match frozen contract")

    expected_target_shape = ["B", N_DEPTH, GRID_H, GRID_W]
    if list(target.get("shape", [])) != expected_target_shape:
        raise ContractError(
            f"Target shape mismatch: {target.get('shape')!r} != {expected_target_shape!r}"
        )
    depths = [float(x) for x in target.get("depths_m", [])]
    if depths != DEPTHS_M:
        raise ContractError("Target depth vector does not match frozen 15-depth contract")
    if target.get("mode") != "anomaly":
        raise ContractError(f"B6 requires anomaly target mode, got {target.get('mode')!r}")

    expected_spatial = ["B", SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W]
    expected_temporal = ["B", "T", TEMPORAL_CONTEXT_CHANNELS]
    if context.get("spatial") != expected_spatial:
        raise ContractError("Spatial context contract mismatch")
    if context.get("temporal") != expected_temporal:
        raise ContractError("Temporal context contract mismatch")

    if contract.get("dynamic_validity") != expected_dynamic_shape:
        raise ContractError("Dynamic validity contract mismatch")
    if contract.get("static_validity") != expected_static_shape:
        raise ContractError("Static validity contract mismatch")

    if str(contract.get("backbone_version", "")) != EXPECTED_BACKBONE_VERSION:
        raise ContractError(
            f"Backbone version mismatch: expected {EXPECTED_BACKBONE_VERSION}, "
            f"got {contract.get('backbone_version')!r}"
        )


def scientific_spec() -> Dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "scientific_role": "joint_spatial_temporal_recurrent_baseline",
        "domain": {
            "lat_deg": [5.0, 30.0],
            "lon_deg": [45.0, 105.0],
            "resolution_deg": 0.25,
            "grid": [GRID_H, GRID_W],
        },
        "inputs": {
            "dynamic_channels": DYNAMIC_CHANNELS,
            "dynamic_feature_names": DYNAMIC_FEATURE_NAMES,
            "source_slices": SOURCE_SLICES,
            "static_channels": STATIC_CHANNELS,
            "static_feature_names": STATIC_FEATURE_NAMES,
            "spatial_context_channels": SPATIAL_CONTEXT_CHANNELS,
            "spatial_context_names": SPATIAL_CONTEXT_NAMES,
            "temporal_context_channels": TEMPORAL_CONTEXT_CHANNELS,
            "temporal_context_names": TEMPORAL_CONTEXT_NAMES,
            "total_per_day_channels": TOTAL_INPUT_CHANNELS,
            "window_days": EXPECTED_WINDOW_DAYS,
            "temporal_order": "t-13 ... t-1, t",
        },
        "target": {
            "variable": "GLORYS thetao",
            "depths_m": DEPTHS_M,
            "mode": "anomaly",
        },
        "architecture": {
            "encoder_highres_channels": B6_HIGHRES_CHANNELS,
            "recurrent_lowres_channels": B6_HIDDEN_CHANNELS,
            "recurrent_layers": 2,
            "recurrent_kernel": 3,
            "recurrent_grid": [GRID_H // 2, GRID_W // 2],
            "temporal_recurrence": True,
            "bidirectional": False,
            "attention": False,
            "decoder_upsample": "2x bilinear",
            "decoder_projection_kernel": 3,
            "head_kernel": 1,
            "spatial_hidden_state": True,
        },
        "spatial_validity_policy": {
            "method": "partial_convolution_and_masked_pooling",
            "dynamic_validity": True,
            "static_validity": True,
            "permanent_land_policy": "frozen_normalized_ocean_fraction_raw_zero_reference",
            "hidden_cell_land_suppression": True,
            "loss_mask_used_as_input_mask": False,
        },
        "scientific_controls": {
            "b4_distinction": "B4 encodes spatially then uses vector LSTM; B6 maintains spatial ConvLSTM hidden state through recurrence",
            "argo_validation_required_for_final_scientific_claim": True,
        },
    }


# ============================================================================
# Spatial validity preparation
# ============================================================================

def _to_bool_mask(mask: torch.Tensor, shape: Tuple[int, ...], name: str) -> torch.Tensor:
    if tuple(mask.shape) != shape:
        raise ContractError(f"{name} shape {tuple(mask.shape)} != {shape}")
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


# ============================================================================
# Masked / partial convolution and pooling
# ============================================================================

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


# ============================================================================
# ConvLSTM
# ============================================================================

B6_HIGHRES_CHANNELS = 32
B6_HIDDEN_CHANNELS = 32
B6_LSTM_LAYERS = 2
B6_RECURRENT_KERNEL = 3
B6_RECURRENT_PADDING = 1


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


# ============================================================================
# B6 model
# ============================================================================

class Model(nn.Module):
    """OceanEmbed B6 mask-aware encoder + two-layer ConvLSTM + decoder."""

    model_id = MODEL_ID
    model_version = MODEL_VERSION
    model_contract = MODEL_CONTRACT

    def __init__(self, contract: Dict[str, Any]):
        super().__init__()
        validate_contract(contract)
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
                "head": "masked 1x1 -> 15 depths",
                "output_shape": ["B", N_DEPTH, GRID_H, GRID_W],
                "hidden_state_is_spatial": True,
            },
        }


def build_model(contract: Dict[str, Any]) -> Model:
    return Model(contract)


def _import_module(path: Path, prefix: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = (
        f"{prefix}_"
        f"{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _search_sibling(filename: str) -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,
        here.parent / filename,
        here.parent / "framework" / filename,
        here.parent / "data" / filename,
        here.parent.parent / "src" / "framework" / filename,
        here.parent.parent / "src" / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_framework_path(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    found = _search_sibling("OceanEmbed_TrainingFramework_V1.py")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate OceanEmbed_TrainingFramework_V1.py; "
        "use --framework-path"
    )


def resolve_backbone_path(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    found = _search_sibling("OceanEmbed_DL_DataBackbone_V1.py")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate OceanEmbed_DL_DataBackbone_V1.py; "
        "use --backbone-path"
    )


def validate_framework_contract(framework: ModuleType) -> None:
    got = getattr(framework, "FRAMEWORK_CONTRACT", None)
    if got != EXPECTED_FRAMEWORK_CONTRACT:
        raise ContractError(
            f"Training Framework contract mismatch: "
            f"expected {EXPECTED_FRAMEWORK_CONTRACT}, got {got!r}"
        )

    required = (
        "TrainConfig",
        "BackboneAdapter",
        "Trainer",
        "forward_prediction",
        "masked_mse_loss",
        "move_batch_to_device",
    )
    missing = [name for name in required if not hasattr(framework, name)]
    if missing:
        raise IntegrationTestError(
            f"Training Framework missing required public APIs: {missing}"
        )


def validate_backbone_runtime(adapter: Any) -> None:
    """Validate the actual Backbone implementation and its frozen state.

    IMPORTANT: ``Backbone.mod`` is the dynamically imported synchronized
    loader module. It is NOT the Backbone implementation module, so
    BACKBONE_VERSION/BACKBONE_CONTRACT must be read from the class-defining
    module (or from the frozen backbone_state.json), never from
    ``adapter.backbone.mod``.
    """
    backbone_obj = getattr(adapter, "backbone", None)
    if backbone_obj is None:
        raise ContractError("BackboneAdapter has no backbone object")

    runtime_fp = getattr(backbone_obj, "fp", None)
    if runtime_fp != EXPECTED_BUNDLE_FP:
        raise ContractError(
            f"Bundle fingerprint mismatch: expected {EXPECTED_BUNDLE_FP}, "
            f"runtime {runtime_fp}"
        )

    # Resolve the module that actually defines the Backbone class.
    # The canonical implementation exports BACKBONE_VERSION and
    # BACKBONE_CONTRACT at module level; ``backbone_obj.mod`` does not.
    implementation_module = sys.modules.get(type(backbone_obj).__module__)
    if implementation_module is None:
        raise ContractError(
            "Could not resolve the imported Backbone implementation module "
            "from the runtime Backbone object's class"
        )

    implementation_version = getattr(
        implementation_module, "BACKBONE_VERSION", None
    )
    implementation_contract = getattr(
        implementation_module, "BACKBONE_CONTRACT", None
    )

    if implementation_version != EXPECTED_BACKBONE_VERSION:
        raise ContractError(
            f"Backbone implementation version mismatch: expected "
            f"{EXPECTED_BACKBONE_VERSION}, got {implementation_version!r}"
        )
    if implementation_contract != EXPECTED_BACKBONE_CONTRACT:
        raise ContractError(
            f"Backbone implementation contract mismatch: expected "
            f"{EXPECTED_BACKBONE_CONTRACT}, got {implementation_contract!r}"
        )

    # Cross-check the frozen on-disk Backbone state as an independent source
    # of truth. The canonical state uses ``backbone_contract_version``.
    bundle_obj = getattr(backbone_obj, "bundle", None)
    state_dir = getattr(bundle_obj, "state", None)
    if state_dir is None:
        raise ContractError(
            "Backbone runtime does not expose its canonical state directory"
        )

    state_path = Path(state_dir) / "backbone_state.json"
    if not state_path.is_file():
        raise ContractError(
            f"Canonical Backbone state file missing: {state_path}"
        )

    try:
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(
            f"Could not read canonical Backbone state: {state_path}: {exc}"
        ) from exc

    state_fp = state_data.get("bundle_fingerprint")
    if state_fp != EXPECTED_BUNDLE_FP:
        raise ContractError(
            f"Stored Backbone state fingerprint mismatch: expected "
            f"{EXPECTED_BUNDLE_FP}, got {state_fp!r}"
        )

    state_version = str(state_data.get("backbone_version", ""))
    if state_version != EXPECTED_BACKBONE_VERSION:
        raise ContractError(
            f"Stored Backbone state version mismatch: expected "
            f"{EXPECTED_BACKBONE_VERSION}, got {state_version!r}"
        )

    state_contract = str(state_data.get("backbone_contract_version", ""))
    if state_contract != EXPECTED_BACKBONE_CONTRACT:
        raise ContractError(
            f"Stored Backbone state contract mismatch: expected "
            f"{EXPECTED_BACKBONE_CONTRACT}, got {state_contract!r}"
        )

    # Finally validate the public adapter contract consumed by B3.
    validate_contract(adapter.contract)


def validate_runtime_normalization(adapter: Any) -> Dict[str, Any]:
    norm = getattr(adapter, "norm", None)
    if norm is None:
        raise ContractError("BackboneAdapter does not expose normalization state")

    static_mean = np.asarray(norm.static_mean, dtype=np.float64)
    static_std = np.asarray(norm.static_std, dtype=np.float64)

    if static_mean.shape != (STATIC_CHANNELS,):
        raise ContractError(
            f"Unexpected static_mean shape: {static_mean.shape}"
        )
    if static_std.shape != (STATIC_CHANNELS,):
        raise ContractError(
            f"Unexpected static_std shape: {static_std.shape}"
        )

    runtime_land_z = (
        -static_mean[OCEAN_FRACTION_STATIC_INDEX]
        / static_std[OCEAN_FRACTION_STATIC_INDEX]
    )

    if not math.isfinite(runtime_land_z):
        raise ContractError("Runtime land normalization reference is non-finite")

    if abs(runtime_land_z - EXPECTED_LAND_OCEAN_FRACTION_Z) > 1e-10:
        raise ContractError(
            "Frozen B3 land-mask normalization reference does not match "
            "runtime Backbone normalization: "
            f"expected {EXPECTED_LAND_OCEAN_FRACTION_Z:.15g}, "
            f"runtime {runtime_land_z:.15g}"
        )

    if not math.isclose(
        float(static_mean[OCEAN_FRACTION_STATIC_INDEX]),
        OCEAN_FRACTION_TRAIN_MEAN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError("Frozen ocean_fraction normalization mean mismatch")

    if not math.isclose(
        float(static_std[OCEAN_FRACTION_STATIC_INDEX]),
        OCEAN_FRACTION_TRAIN_STD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ContractError("Frozen ocean_fraction normalization std mismatch")

    bundle_fp = str(getattr(norm, "bundle_fp", ""))
    if bundle_fp != EXPECTED_BUNDLE_FP:
        raise ContractError(
            f"Normalization bundle fingerprint mismatch: "
            f"expected {EXPECTED_BUNDLE_FP}, got {bundle_fp}"
        )

    return {
        "ocean_fraction_static_index": OCEAN_FRACTION_STATIC_INDEX,
        "ocean_fraction_train_mean": OCEAN_FRACTION_TRAIN_MEAN,
        "ocean_fraction_train_std": OCEAN_FRACTION_TRAIN_STD,
        "normalized_land_reference": EXPECTED_LAND_OCEAN_FRACTION_Z,
        "runtime_land_reference": runtime_land_z,
        "bundle_fingerprint": bundle_fp,
    }


def build_framework_config(
    args: argparse.Namespace,
    framework: ModuleType,
) -> Any:
    cfg = framework.TrainConfig(
        dataset_root=str(Path(args.dataset_root).expanduser().resolve()),
        backbone_path=str(resolve_backbone_path(args.backbone_path)),
        loader_path=args.loader_path,
        model_module=str(Path(__file__).resolve()),
        model_class="Model",
        run_root=str(Path(args.run_root).expanduser().resolve()),
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume,
    )
    cfg.validate()
    return cfg


def runtime_spec(
    args: argparse.Namespace,
    framework_path: Path,
    backbone_path: Path,
    normalization_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "scientific_spec": scientific_spec(),
        "runtime_config": {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "run_root": str(Path(args.run_root).expanduser().resolve()),
            "experiment_name": args.experiment_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "patience": args.patience,
            "device": args.device,
            "amp": not args.no_amp,
            "seed": args.seed,
            "log_every": args.log_every,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_every_steps": args.checkpoint_every_steps,
        },
        "source_hashes": {
            "model": sha256_file(Path(__file__).resolve()),
            "framework": sha256_file(framework_path),
            "backbone": sha256_file(backbone_path),
        },
        "normalization_guard": normalization_report,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "runtime_monitor": {
            "enabled": bool(ENABLE_RUNTIME_MONITOR),
            "checkpoint_poll_seconds": float(CHECKPOINT_POLL_SECONDS),
            "heartbeat_interval_seconds": float(HEARTBEAT_INTERVAL_SECONDS),
            "heartbeat_print_interval_seconds": float(HEARTBEAT_PRINT_INTERVAL_SEC),
            "stall_diagnostic_seconds": float(STALL_DIAGNOSTIC_SECONDS),
            "session_soft_limit_hours": float(SESSION_SOFT_LIMIT_HOURS),
            "session_hard_limit_hours": float(SESSION_HARD_LIMIT_HOURS),
            "snapshot_checkpoints": bool(SNAPSHOT_CHECKPOINTS),
        },
        "created_utc": utc_now(),
    }


def write_start_spec(
    run_dir: Path,
    args: argparse.Namespace,
    framework_path: Path,
    backbone_path: Path,
    adapter: Any,
    trainer: Optional[Any] = None,
) -> Path:
    norm_report = validate_runtime_normalization(adapter)
    payload = {
        "status": "STARTED",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "model_parameters": parameter_summary(
            trainer.model if trainer is not None else Model(adapter.contract)
        ),
        "runtime_spec": runtime_spec(
            args,
            framework_path,
            backbone_path,
            norm_report,
        ),
        "started_utc": utc_now(),
    }
    path = run_dir / "b6_convlstm_experiment_spec.json"
    atomic_json(path, payload)
    return path


def update_end_spec(run_dir: Path, result: Dict[str, Any]) -> None:
    path = run_dir / "b6_convlstm_experiment_spec.json"
    payload = result
    payload["status"] = "COMPLETED"
    payload["completed_utc"] = utc_now()
    atomic_json(path, payload)


def _trainer_model_identity(model: nn.Module) -> str:
    return str(getattr(model, "model_id", ""))


def train_via_framework(args: argparse.Namespace) -> Dict[str, Any]:
    framework_path = resolve_framework_path(args.framework_path)
    backbone_path = resolve_backbone_path(args.backbone_path)

    framework = _import_module(
        framework_path,
        "oceanembed_training_framework_b6",
    )
    validate_framework_contract(framework)

    cfg = build_framework_config(args, framework)
    adapter = framework.BackboneAdapter(cfg)
    validate_backbone_runtime(adapter)
    norm_report = validate_runtime_normalization(adapter)

    trainer = framework.Trainer(cfg, adapter)

    if _trainer_model_identity(trainer.model) != MODEL_ID:
        raise IntegrationTestError(
            "Training Framework instantiated unexpected model identity: "
            f"{_trainer_model_identity(trainer.model)!r}"
        )

    run_dir = Path(trainer.run_dir)
    start_spec = {
        "status": "STARTED",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "model_parameters": parameter_summary(trainer.model),
        "normalization_guard": norm_report,
        "runtime_spec": runtime_spec(
            args,
            framework_path,
            backbone_path,
            norm_report,
        ),
        "model_summary": (
            trainer.model.model_summary()
            if hasattr(trainer.model, "model_summary")
            else None
        ),
        "started_utc": utc_now(),
    }
    atomic_json(run_dir / "b6_convlstm_experiment_spec.json", start_spec)

    result: Dict[str, Any]
    try:
        result = trainer.fit()
    except Exception as exc:
        failure = {
            **start_spec,
            "status": "FAILED",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "failed_utc": utc_now(),
        }
        atomic_json(run_dir / "b6_convlstm_experiment_spec.json", failure)
        raise

    final = {
        "status": "COMPLETED",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "parameters": parameter_summary(trainer.model),
        "model_summary": (
            trainer.model.model_summary()
            if hasattr(trainer.model, "model_summary")
            else None
        ),
        "normalization_guard": norm_report,
        "run_dir": str(run_dir),
        "result": result,
        "runtime_spec": runtime_spec(
            args,
            framework_path,
            backbone_path,
            norm_report,
        ),
        "completed_utc": utc_now(),
    }
    atomic_json(run_dir / "b6_convlstm_experiment_spec.json", final)
    return final


def _synthetic_contract() -> Dict[str, Any]:
    return {
        "backbone_version": EXPECTED_BACKBONE_VERSION,
        "dynamic": {
            "shape": ["B", "T", DYNAMIC_CHANNELS, GRID_H, GRID_W],
            "features": DYNAMIC_FEATURE_NAMES,
            "source_slices": SOURCE_SLICES,
        },
        "dynamic_validity": ["B", "T", DYNAMIC_CHANNELS, GRID_H, GRID_W],
        "static": {
            "shape": ["B", STATIC_CHANNELS, GRID_H, GRID_W],
            "features": STATIC_FEATURE_NAMES,
        },
        "static_validity": ["B", STATIC_CHANNELS, GRID_H, GRID_W],
        "target": {
            "shape": ["B", N_DEPTH, GRID_H, GRID_W],
            "depths_m": DEPTHS_M,
            "mode": "anomaly",
        },
        "context": {
            "spatial": ["B", SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W],
            "temporal": ["B", "T", TEMPORAL_CONTEXT_CHANNELS],
        },
    }


def _synthetic_batch(
    batch_size: int = 2,
    seed: int = 42,
    explicit_ocean_mask: bool = True,
) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    ocean_mask = torch.ones(batch_size, 1, GRID_H, GRID_W, dtype=torch.float32)
    ocean_mask[:, :, :20, :30] = 0.0

    static = torch.randn(batch_size, STATIC_CHANNELS, GRID_H, GRID_W, generator=g)
    if explicit_ocean_mask:
        static[:, OCEAN_FRACTION_STATIC_INDEX] = torch.where(
            ocean_mask[:, 0] > 0,
            torch.ones(GRID_H, GRID_W),
            torch.zeros(GRID_H, GRID_W),
        )
    else:
        land_z = torch.tensor(EXPECTED_LAND_OCEAN_FRACTION_Z, dtype=torch.float32)
        ocean_z = torch.tensor(0.25, dtype=torch.float32)
        static[:, OCEAN_FRACTION_STATIC_INDEX] = torch.where(
            ocean_mask[:, 0] > 0, ocean_z, land_z
        )

    x_dynamic = torch.randn(
        batch_size, EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W, generator=g
    )
    x_static_valid = torch.ones(
        batch_size, STATIC_CHANNELS, GRID_H, GRID_W, dtype=torch.bool
    )
    x_dynamic_valid = torch.ones(
        batch_size, EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W, dtype=torch.bool
    )

    return {
        "x_dynamic": x_dynamic,
        "x_static": static,
        "x_dynamic_valid": x_dynamic_valid,
        "x_static_valid": x_static_valid,
        "ocean_mask": ocean_mask,
        "context_spatial": torch.randn(
            SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W, generator=g
        ),
        "context_temporal": torch.randn(
            batch_size, EXPECTED_WINDOW_DAYS, TEMPORAL_CONTEXT_CHANNELS, generator=g
        ),
        "y": torch.randn(batch_size, N_DEPTH, GRID_H, GRID_W, generator=g),
        "loss_mask": ocean_mask.expand(-1, N_DEPTH, -1, -1).bool(),
    }


def _assert_tensor_equal(a: torch.Tensor, b: torch.Tensor, message: str) -> None:
    if not torch.equal(a, b):
        diff = float((a - b).abs().max().detach().cpu().item())
        raise IntegrationTestError(f"{message}; max_abs_diff={diff}")


def _set_b6_probe_weights(model: Model) -> None:
    """Deterministic non-zero weights for structural temporal tests."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PartialConv2d):
                module.weight.fill_(0.01)
                if module.bias is not None:
                    module.bias.zero_()
        # Preserve a usable forget gate while probing.
        for cell in (model.convlstm1, model.convlstm2):
            if cell.gate_conv.bias is not None:
                cell.gate_conv.bias.view(4, cell.hidden_channels).zero_()
                cell.gate_conv.bias.view(4, cell.hidden_channels)[1].fill_(1.0)


def _validate_temporal_path(model: Model, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Prove that the 14-day sequence affects the spatial recurrent state in order."""
    probe = Model(model.contract).cpu().eval()
    _set_b6_probe_weights(probe)
    probe_batch = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()
    }

    with torch.no_grad():
        _, _, _, original_debug = probe.encode_sequence(
            probe_batch, return_debug=True
        )
        original_out = probe(probe_batch)

    earlier = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in probe_batch.items()
    }
    earlier_day = EXPECTED_WINDOW_DAYS - 2  # day index 12 of 14
    earlier["x_dynamic"][:, earlier_day, 0, 50, 120] += 10.0

    with torch.no_grad():
        changed_fused, _, _, changed_debug = probe.encode_sequence(
            earlier, return_debug=True
        )
        changed_out = probe(earlier)

    # Directly compare the recurrent state immediately after the perturbed day.
    original_mid = original_debug["hidden2_states"][earlier_day]
    changed_mid = changed_debug["hidden2_states"][earlier_day]
    earlier_hidden_diff = float(
        (changed_mid - original_mid).abs().amax().cpu().item()
    )

    # Then verify that the perturbation survives into the final recurrent state.
    original_final_h2 = original_debug["final_hidden_layer2"]
    changed_final_h2 = changed_debug["final_hidden_layer2"]
    final_hidden_diff = float(
        (changed_final_h2 - original_final_h2).abs().amax().cpu().item()
    )

    # Finally verify that the decoder output can reflect the propagated change.
    earlier_output_diff = float(
        (changed_out - original_out).abs().amax().cpu().item()
    )

    if earlier_hidden_diff <= 1e-7:
        raise IntegrationTestError(
            "Earlier-day perturbation did not change the ConvLSTM hidden state "
            f"at the perturbed timestep: diff={earlier_hidden_diff:.6g}"
        )
    if final_hidden_diff <= 1e-8:
        raise IntegrationTestError(
            "Earlier-day perturbation did not propagate into the final ConvLSTM "
            f"hidden state: diff={final_hidden_diff:.6g}"
        )
    if earlier_output_diff <= 1e-10:
        raise IntegrationTestError(
            "Earlier-day perturbation did not affect the final B6 output: "
            f"diff={earlier_output_diff:.6g}"
        )

    reversed_batch = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in probe_batch.items()
    }
    reversed_batch["x_dynamic"] = reversed_batch["x_dynamic"].flip(1)
    reversed_batch["x_dynamic_valid"] = reversed_batch["x_dynamic_valid"].flip(1)
    reversed_batch["context_temporal"] = reversed_batch["context_temporal"].flip(1)
    with torch.no_grad():
        reversed_out = probe(reversed_batch)
    reverse_diff = float((reversed_out - original_out).abs().amax().cpu().item())
    if reverse_diff <= 1e-8:
        raise IntegrationTestError(
            "Reversing the 14-day sequence did not alter the ConvLSTM output"
        )

    final_h2 = original_debug["final_hidden_layer2"]
    if tuple(final_h2.shape) != (
        probe_batch["x_dynamic"].shape[0], 32, 50, 120
    ):
        raise IntegrationTestError(
            f"Final ConvLSTM hidden state has wrong shape: {tuple(final_h2.shape)}"
        )
    if len(original_debug["hidden2_states"]) != EXPECTED_WINDOW_DAYS:
        raise IntegrationTestError(
            "ConvLSTM did not expose exactly 14 recurrent hidden states"
        )
    for idx, state in enumerate(original_debug["hidden2_states"]):
        expected = (probe_batch["x_dynamic"].shape[0], 32, 50, 120)
        if tuple(state.shape) != expected:
            raise IntegrationTestError(
                f"ConvLSTM hidden state at timestep {idx} has shape "
                f"{tuple(state.shape)} != {expected}"
            )

    return {
        "earlier_day_index_tested": earlier_day,
        "earlier_day_hidden_change_at_timestep_max_abs": earlier_hidden_diff,
        "earlier_day_final_hidden_change_max_abs": final_hidden_diff,
        "earlier_day_output_change_max_abs": earlier_output_diff,
        "reversed_sequence_max_abs_change": reverse_diff,
        "recurrent_hidden_shape": list(final_h2.shape),
        "recurrent_steps": len(original_debug["hidden2_states"]),
    }


def _checkpoint_identity_sidecar(path: Path) -> Path:
    return Path(str(path) + ".metadata.json")


def _checkpoint_identity_payload(
    path: Path,
    epoch: int,
    checkpoint_kind: str,
) -> Dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "checkpoint_file": path.name,
        "epoch": int(epoch),
        "checkpoint_kind": str(checkpoint_kind),
        "created_utc": utc_now(),
    }


def _write_checkpoint_identity_sidecar(
    path: Path,
    epoch: int,
    checkpoint_kind: str,
) -> Path:
    sidecar = _checkpoint_identity_sidecar(path)
    atomic_json(
        sidecar,
        _checkpoint_identity_payload(path, epoch, checkpoint_kind),
    )
    return sidecar


def _read_checkpoint_identity_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    """Read the optional checkpoint identity sidecar if it exists.

    Resume no longer requires the sidecar: the checkpoint payload itself carries
    the B6 identity/contract information required for compatibility checks.
    When a sidecar is present, it is still validated as an additional consistency
    check. This lets a single ``latest_model.pt`` file be used for continuation.
    """
    sidecar = _checkpoint_identity_sidecar(path)
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(
            f"Refusing resume: checkpoint identity sidecar is unreadable: {sidecar}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ContractError("Refusing resume: checkpoint identity sidecar is not a JSON object")
    return data


def _validate_resume_checkpoint_metadata(path: Path, model: Model) -> Dict[str, Any]:
    """Validate a B6 ``latest_model.pt`` using payload metadata.

    The metadata sidecar is optional. The checkpoint payload must independently
    prove compatibility through the frozen B6 experiment name, model contract/hash,
    pinned Backbone fingerprint, checkpoint state payload, and epoch. If a sidecar
    exists, its identity fields are cross-checked as an extra integrity guard.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.name != "latest_model.pt":
        raise ContractError(
            f"Refusing resume: only latest_model.pt is accepted, got {path.name!r}"
        )

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ContractError("Resume checkpoint is not a dictionary payload")

    stored_fp = payload.get("backbone_fingerprint")
    if stored_fp != EXPECTED_BUNDLE_FP:
        raise ContractError(
            "Refusing resume: checkpoint Backbone fingerprint does not match the pinned bundle"
        )

    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ContractError("Refusing resume: checkpoint config metadata is missing")
    stored_experiment = str(cfg.get("experiment_name", ""))
    if stored_experiment != MODEL_NAME:
        raise ContractError(
            "Refusing cross-baseline resume: "
            f"checkpoint experiment_name={stored_experiment!r}, expected {MODEL_NAME!r}"
        )

    stored_contract = payload.get("model_contract")
    if isinstance(stored_contract, str):
        if stored_contract != MODEL_CONTRACT:
            raise ContractError(
                "Refusing resume: checkpoint model contract identifier does not match current B6 contract"
            )
    elif isinstance(stored_contract, dict):
        # Canonical Trainer checkpoints store the full model-input contract here.
        # The cryptographic contract hash below is the authoritative equality check.
        if not stored_contract:
            raise ContractError("Refusing resume: checkpoint model contract metadata is empty")
    else:
        raise ContractError("Refusing resume: checkpoint model contract metadata is missing or invalid")

    stored_hash = payload.get("model_contract_hash")
    if stored_hash != model.contract_hash:
        raise ContractError(
            "Refusing resume: checkpoint model contract hash does not match current B6 contract"
        )

    stored_framework_contract = payload.get("framework_contract")
    if stored_framework_contract != EXPECTED_FRAMEWORK_CONTRACT:
        raise ContractError(
            "Refusing resume: checkpoint framework contract does not match the pinned training framework"
        )

    stored_depths = payload.get("depths_m")
    if stored_depths is not None:
        try:
            if [float(x) for x in stored_depths] != DEPTHS_M:
                raise ContractError(
                    "Refusing resume: checkpoint target depth vector does not match frozen B6 depths"
                )
        except (TypeError, ValueError) as exc:
            raise ContractError("Refusing resume: checkpoint target depth vector is invalid") from exc

    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ContractError("Refusing resume: checkpoint lacks model/optimizer state")

    epoch = int(payload.get("epoch", 0))
    if epoch < 1:
        raise ContractError(f"Refusing resume: invalid checkpoint epoch {epoch}")

    kind = str(payload.get("checkpoint_kind", "epoch"))
    if kind != "epoch":
        raise ContractError(
            f"Refusing resume: latest_model.pt must be an epoch checkpoint, got {kind!r}"
        )

    identity = _read_checkpoint_identity_sidecar(path)
    if identity is None:
        identity_source = "checkpoint_payload"
        identity_schema = None
        identity_experiment = EXPERIMENT_ID
        identity_model_id = MODEL_ID
        identity_version = MODEL_VERSION
        identity_contract = MODEL_CONTRACT
        identity_fp = EXPECTED_BUNDLE_FP
        identity_epoch = epoch
        identity_kind = kind
    else:
        expected_identity = {
            "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_contract": MODEL_CONTRACT,
            "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        }
        for key, expected in expected_identity.items():
            got = identity.get(key)
            if got != expected:
                raise ContractError(
                    f"Refusing resume: checkpoint identity mismatch for {key}: "
                    f"got {got!r}, expected {expected!r}"
                )

        identity_epoch = int(identity.get("epoch", 0))
        if identity_epoch != epoch:
            raise ContractError(
                f"Refusing resume: checkpoint epoch mismatch between payload ({epoch}) "
                f"and identity sidecar ({identity_epoch})"
            )
        identity_kind = str(identity.get("checkpoint_kind", ""))
        if identity_kind != kind:
            raise ContractError(
                f"Refusing resume: checkpoint kind mismatch between payload ({kind}) "
                f"and identity sidecar ({identity_kind})"
            )
        identity_source = "checkpoint_payload+sidecar"
        identity_schema = identity["schema_version"]
        identity_experiment = identity["experiment_id"]
        identity_model_id = identity["model_id"]
        identity_version = identity["model_version"]
        identity_contract = identity["model_contract"]
        identity_fp = identity["bundle_fingerprint"]

    return {
        "status": "PASS",
        "checkpoint": str(path),
        "identity_source": identity_source,
        "identity_schema": identity_schema,
        "experiment_id": identity_experiment,
        "model_id": identity_model_id,
        "model_version": identity_version,
        "model_contract": identity_contract,
        "bundle_fingerprint": identity_fp,
        "experiment_name": stored_experiment,
        "model_contract_hash": stored_hash,
        "epoch": epoch,
        "checkpoint_kind": identity_kind,
        "backbone_fingerprint": stored_fp,
        "sidecar_present": identity is not None,
    }


def _validate_b6_historical_best_checkpoint(path: Path, model: Model) -> Dict[str, Any]:
    """Validate an epoch/best B6 checkpoint for restoration as best_model.pt.

    RESUME_FROM remains restricted to latest_model.pt. This helper accepts an
    immutable epoch or best_epoch checkpoint from an earlier B6 run and only
    restores it when the payload proves that it represented the validation-best
    state at that time.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ContractError("Historical best checkpoint is not a dictionary payload")

    if payload.get("backbone_fingerprint") != EXPECTED_BUNDLE_FP:
        raise ContractError(
            "Historical best checkpoint Backbone fingerprint does not match the pinned bundle"
        )

    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ContractError("Historical best checkpoint config metadata is missing")
    stored_experiment = str(cfg.get("experiment_name", ""))
    if stored_experiment != MODEL_NAME:
        raise ContractError(
            "Historical best checkpoint belongs to a different experiment: "
            f"{stored_experiment!r} != {MODEL_NAME!r}"
        )

    if payload.get("model_contract_hash") != model.contract_hash:
        raise ContractError(
            "Historical best checkpoint model contract hash does not match current B6 contract"
        )

    if payload.get("framework_contract") != EXPECTED_FRAMEWORK_CONTRACT:
        raise ContractError(
            "Historical best checkpoint framework contract does not match the pinned training framework"
        )

    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ContractError("Historical best checkpoint lacks model/optimizer state")

    epoch = int(payload.get("epoch", 0))
    if epoch < 1:
        raise ContractError(f"Historical best checkpoint has invalid epoch {epoch}")

    kind = str(payload.get("checkpoint_kind", "epoch"))
    if kind not in {"epoch", "best_epoch"}:
        raise ContractError(
            "Historical best checkpoint must be an epoch/best_epoch checkpoint, "
            f"got {kind!r}"
        )

    val_loss = payload.get("val_loss")
    best_val_loss = payload.get("best_val_loss")
    is_best_state = kind == "best_epoch"
    try:
        is_best_state = is_best_state or (
            val_loss is not None
            and best_val_loss is not None
            and math.isfinite(float(val_loss))
            and math.isfinite(float(best_val_loss))
            and abs(float(val_loss) - float(best_val_loss)) <= BEST_MATCH_TOLERANCE
        )
    except (TypeError, ValueError):
        pass

    if not is_best_state:
        raise ContractError(
            "Checkpoint is not identifiable as the historical validation-best state: "
            f"epoch={epoch}, val_loss={val_loss!r}, "
            f"best_val_loss={best_val_loss!r}, kind={kind!r}"
        )

    return {
        "status": "PASS",
        "checkpoint": str(path),
        "epoch": epoch,
        "checkpoint_kind": kind,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "experiment_name": stored_experiment,
        "model_contract_hash": payload.get("model_contract_hash"),
        "backbone_fingerprint": payload.get("backbone_fingerprint"),
    }


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy a completed checkpoint to a new destination."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _preserve_historical_best_checkpoint(
    run_dir: Path,
    model: Model,
    resume_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Restore a known historical best checkpoint as best_model.pt.

    Priority is an explicit HISTORICAL_BEST_CHECKPOINT, then the continuation
    checkpoint itself when that payload proves it was the validation-best state.
    A non-best final checkpoint is never relabeled as best_model.pt.
    """
    if not PRESERVE_HISTORICAL_BEST:
        return None

    destination = run_dir / "best_model.pt"
    if destination.is_file():
        print(f"[BEST] Existing best_model.pt preserved: {destination}", flush=True)
        return {"status": "EXISTS", "checkpoint": str(destination)}

    candidates: List[Tuple[str, Path]] = []
    if HISTORICAL_BEST_CHECKPOINT:
        candidates.append(
            (
                "explicit historical-best",
                Path(HISTORICAL_BEST_CHECKPOINT).expanduser().resolve(),
            )
        )
    if resume_path is not None:
        candidates.append(("resume checkpoint", resume_path))

    last_error: Optional[str] = None
    for label, candidate in candidates:
        try:
            report = _validate_b6_historical_best_checkpoint(candidate, model)
            _atomic_copy_file(candidate, destination)
            print(
                "[BEST] Restored historical best checkpoint before training/test:",
                flush=True,
            )
            print(
                f"       source={candidate} (epoch={report['epoch']}, "
                f"val_loss={report['val_loss']}, "
                f"best_val_loss={report['best_val_loss']})",
                flush=True,
            )
            print(f"       destination={destination}", flush=True)
            return {
                **report,
                "source_label": label,
                "destination": str(destination),
            }
        except Exception as exc:
            last_error = f"{label}: {type(exc).__name__}: {exc}"
            print(f"[BEST] Candidate rejected: {last_error}", flush=True)

    if last_error:
        print(
            "[BEST] No provable historical best checkpoint was restored. "
            "The framework will need to create best_model.pt during this run "
            "before evaluate_test() can succeed.",
            flush=True,
        )
    return None


def _find_compatible_b6_checkpoint(root: Path, model: Model) -> Optional[Path]:
    """Find the newest compatible B6 latest_model.pt without accepting another baseline."""
    if not root.is_dir():
        return None

    # Resume discovery intentionally accepts only the canonical latest checkpoint
    # file. best/step/interrupted/failure checkpoints remain available as artifacts
    # but are never auto-selected for continuation.
    candidate_names = {"latest_model.pt"}
    candidates: List[Path] = []
    try:
        for p in root.rglob("*.pt"):
            if p.is_file() and p.name in candidate_names:
                candidates.append(p)
    except Exception:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            report = _validate_resume_checkpoint_metadata(candidate, model)
            print(
                f"[RESUME] Auto-discovered compatible B6 checkpoint: "
                f"{candidate} (epoch={report['epoch']}, kind={report['checkpoint_kind']})",
                flush=True,
            )
            return candidate
        except Exception as exc:
            # Incompatible checkpoints are intentionally ignored during auto-discovery;
            # this prevents a B3/B4/B5 checkpoint from ever being selected.
            print(
                f"[RESUME] Ignoring candidate {candidate}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
    return None


def _checkpoint_round_trip_test(model: Model, contract: Dict[str, Any]) -> bool:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="oceanembed_b6_ckpt_") as td:
        path = Path(td) / "b6_test_checkpoint.pt"
        payload = {
            "framework_version": "self-test",
            "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
            "checkpoint_kind": "epoch",
            "epoch": 1,
            "epoch_step": 10,
            "global_step": 10,
            "backbone_fingerprint": EXPECTED_BUNDLE_FP,
            "model_contract": contract,
            "model_contract_hash": model.contract_hash,
            "config": {"experiment_name": MODEL_NAME},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        }
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu")
        clone = Model(contract).cpu()
        clone.load_state_dict(loaded["model_state_dict"])
        test_batch = _synthetic_batch(batch_size=1, seed=123, explicit_ocean_mask=True)
        model.eval(); clone.eval()
        with torch.no_grad():
            a = model(test_batch)
            b = clone(test_batch)
        _assert_tensor_equal(a, b, "checkpoint state-dict round-trip changed predictions")
    return True


def run_self_test() -> Dict[str, Any]:
    seed_everything(SEED)
    contract = _synthetic_contract()
    validate_contract(contract)

    model = Model(contract).cpu().train()
    params = parameter_summary(model)
    if params["trainable_parameters"] <= 0:
        raise IntegrationTestError("Model has no trainable parameters")

    batch = _synthetic_batch(batch_size=2, seed=42, explicit_ocean_mask=True)
    with torch.no_grad():
        prediction = model(batch)

    expected_shape = (2, N_DEPTH, GRID_H, GRID_W)
    if tuple(prediction.shape) != expected_shape:
        raise IntegrationTestError(
            f"Output shape {tuple(prediction.shape)} != {expected_shape}"
        )
    if not torch.isfinite(prediction).all():
        raise IntegrationTestError("Synthetic forward produced non-finite output")

    # Land cells must remain invalid/zero.
    land = batch["ocean_mask"] == 0
    if torch.any(prediction.abs()[land.expand(-1, N_DEPTH, -1, -1)] > 1e-7):
        raise IntegrationTestError("Synthetic output is non-zero on permanent land")

    # Deterministic repeat.
    with torch.no_grad():
        repeat = model(batch)
    _assert_tensor_equal(prediction, repeat, "Repeated evaluation is not deterministic")

    # Earlier-day and temporal-order structural tests.
    temporal_report = _validate_temporal_path(model, batch)

    # Invalid predictor invariance.
    transient = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    iy, ix = 50, 120
    transient["x_dynamic"][:, 5, 0, iy, ix] = 0.0
    transient["x_dynamic_valid"][:, 5, 0, iy, ix] = False
    transient_large = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in transient.items()}
    transient_large["x_dynamic"][:, 5, 0, iy, ix] = 1e6
    with torch.no_grad():
        a = model(transient)
        b = model(transient_large)
    _assert_tensor_equal(a, b, "Invalid predictor perturbation changed output")

    # Land-value invariance.
    land_perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    land_perturbed["x_dynamic"][:, :, :, :20, :30] += 999.0
    land_perturbed["x_static"][:, :, :20, :30] -= 777.0
    with torch.no_grad():
        land_a = model(batch)
        land_b = model(land_perturbed)
    _assert_tensor_equal(land_a, land_b, "Land-value perturbation changed output")

    # Masked loss, backward, finite gradients, optimizer step.
    train_model = Model(contract).cpu().train()
    train_batch = _synthetic_batch(batch_size=2, seed=777, explicit_ocean_mask=True)
    pred_initial = train_model(train_batch)
    loss = (
        pred_initial - train_batch["y"]
    ).pow(2)[train_batch["loss_mask"]].mean()
    if not torch.isfinite(loss):
        raise IntegrationTestError("Synthetic masked loss is non-finite")

    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    missing, nonfinite = [], []
    for name, p in train_model.named_parameters():
        if p.grad is None:
            missing.append(name)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(name)
    if missing or nonfinite:
        raise IntegrationTestError(
            f"Gradient failure: missing={missing}, nonfinite={nonfinite}"
        )
    optimizer.step()
    with torch.no_grad():
        pred_after = train_model(train_batch)
    if not torch.isfinite(pred_after).all():
        raise IntegrationTestError("Post-step synthetic prediction is non-finite")
    if torch.equal(pred_initial.detach(), pred_after.detach()):
        raise IntegrationTestError("Optimizer step did not change prediction")

    checkpoint_round_trip = _checkpoint_round_trip_test(train_model, contract)

    # Explicit incompatible-baseline guard test.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="oceanembed_b6_badckpt_") as td:
        bad_path = Path(td) / "b3_like.pt"
        bad_payload = {
            "backbone_fingerprint": EXPECTED_BUNDLE_FP,
            "model_contract_hash": model.contract_hash,
            "config": {"experiment_name": "SPATIAL_CNN_9RF_MASKED_V3"},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "epoch": 1,
            "checkpoint_kind": "epoch",
        }
        torch.save(bad_payload, bad_path)
        atomic_json(
            _checkpoint_identity_sidecar(bad_path),
            {
                "schema_version": CHECKPOINT_IDENTITY_SCHEMA,
                "experiment_id": "B3",
                "model_id": "SPATIAL_CNN_9RF_MASKED_V3",
                "model_version": "3.0.0",
                "model_contract": "SPATIALCNNV3.0",
                "bundle_fingerprint": EXPECTED_BUNDLE_FP,
                "epoch": 1,
                "checkpoint_kind": "epoch",
            },
        )
        try:
            _validate_resume_checkpoint_metadata(bad_path, model)
        except ContractError:
            incompatible_guard = True
        else:
            raise IntegrationTestError("Cross-baseline checkpoint guard did not reject B3-like checkpoint")

    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "parameters": params,
        "output_shape": list(expected_shape),
        "finite_forward": True,
        "masked_loss": True,
        "backward_finite_gradients": True,
        "optimizer_step": True,
        "checkpoint_round_trip": checkpoint_round_trip,
        "state_dict_round_trip": True,
        "model_identity": True,
        "land_output_suppressed": True,
        "land_perturbation_invariance": True,
        "invalid_predictor_invariance": True,
        "incompatible_checkpoint_guard": incompatible_guard,
        "temporal_test": temporal_report,
        "completed_utc": utc_now(),
    }


def _build_test_config(args: argparse.Namespace, framework: ModuleType, bundle_root: Path) -> Any:
    return framework.TrainConfig(
        dataset_root=str(bundle_root),
        backbone_path=str(_RUNTIME_BACKBONE_PATH),
        loader_path=str(_RUNTIME_LOADER_PATH),
        model_module=str(_MODEL_SHIM_PATH),
        model_class="Model",
        run_root=str(Path(args.run_root).expanduser().resolve()),
        experiment_name=args.experiment_name,
        epochs=max(1, int(args.epochs)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=None,
    )


def run_real_integration_test(args: argparse.Namespace) -> Dict[str, Any]:
    framework_path = resolve_framework_path(args.framework_path)
    backbone_path = resolve_backbone_path(args.backbone_path)
    framework = _import_module(framework_path, "oceanembed_training_framework_b6_test")
    validate_framework_contract(framework)

    cfg = _build_test_config(args, framework, Path(args.dataset_root))
    adapter = framework.BackboneAdapter(cfg)
    validate_backbone_runtime(adapter)
    norm_report = validate_runtime_normalization(adapter)
    trainer = framework.Trainer(cfg, adapter)
    model = trainer.model

    if _trainer_model_identity(model) != MODEL_ID:
        raise IntegrationTestError(
            f"Unexpected instantiated model identity: {_trainer_model_identity(model)!r}"
        )

    raw_batch = next(iter(adapter.train_loader))
    if int(raw_batch["x_dynamic"].shape[0]) != args.batch_size:
        raise IntegrationTestError(
            f"Real batch size {raw_batch['x_dynamic'].shape[0]} != configured {args.batch_size}"
        )

    _, _, _, diagnostics = prepare_model_inputs(raw_batch)
    device = trainer.device
    batch = framework.move_batch_to_device(raw_batch, device)
    model.train()

    autocast_ctx = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(getattr(trainer, "amp_enabled", False)),
        )
        if device.type == "cuda" else contextlib.nullcontext()
    )

    with autocast_ctx:
        prediction = framework.forward_prediction(model, batch)
        loss = framework.masked_mse_loss(prediction, batch["y"], batch["loss_mask"])

    expected = (args.batch_size, N_DEPTH, GRID_H, GRID_W)
    if tuple(prediction.shape) != expected:
        raise IntegrationTestError(f"Prediction shape {tuple(prediction.shape)} != {expected}")
    if not torch.isfinite(prediction).all():
        raise IntegrationTestError("Real forward produced non-finite prediction")
    if not torch.isfinite(loss):
        raise IntegrationTestError("Real masked loss is non-finite")

    # Required real-data temporal-consumption test: perturb an earlier valid day.
    temporal_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    _, _, ocean_hr, _ = prepare_model_inputs(batch)
    valid_cells = torch.nonzero(ocean_hr[0, 0] > 0.5, as_tuple=False)
    if valid_cells.numel() == 0:
        raise IntegrationTestError("Real batch contains no valid ocean cells for temporal perturbation test")
    py, px = [int(v) for v in valid_cells[len(valid_cells) // 2].tolist()]
    # Prefer an interior point to avoid boundary effects.
    py = max(2, min(GRID_H - 3, py))
    px = max(2, min(GRID_W - 3, px))
    temporal_batch["x_dynamic"][:, 0, 0, py, px] += 10.0
    with torch.no_grad():
        with autocast_ctx:
            temporal_prediction = framework.forward_prediction(model, temporal_batch)
    temporal_diff = float((temporal_prediction - prediction).abs().amax().float().cpu().item())
    if temporal_diff <= 1e-8:
        raise IntegrationTestError(
            "Earlier-day real-data perturbation did not affect B6 output"
        )

    # Reversed temporal order should generally change the ConvLSTM representation.
    reversed_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    reversed_batch["x_dynamic"] = reversed_batch["x_dynamic"].flip(1)
    reversed_batch["x_dynamic_valid"] = reversed_batch["x_dynamic_valid"].flip(1)
    reversed_batch["context_temporal"] = reversed_batch["context_temporal"].flip(1)
    with torch.no_grad():
        with autocast_ctx:
            reversed_prediction = framework.forward_prediction(model, reversed_batch)
    reverse_diff = float((reversed_prediction - prediction).abs().amax().float().cpu().item())
    if reverse_diff <= 1e-8:
        raise IntegrationTestError("Reversed 14-day real sequence did not change B6 output")

    trainer.optimizer.zero_grad(set_to_none=True)
    if getattr(trainer, "amp_enabled", False):
        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        trainer.optimizer.step()

    with torch.no_grad():
        with autocast_ctx:
            prediction_after = framework.forward_prediction(model, batch)
    if not torch.isfinite(prediction_after).all():
        raise IntegrationTestError("Post-step real prediction is non-finite")

    report = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_version": EXPECTED_BACKBONE_VERSION,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "device": str(device),
        "amp_enabled": bool(getattr(trainer, "amp_enabled", False)),
        "parameters": parameter_summary(model),
        "normalization_guard": norm_report,
        "model_summary": model.model_summary(),
        "real_batch_shapes": {
            key: list(batch[key].shape)
            for key in (
                "x_dynamic", "x_static", "x_dynamic_valid", "x_static_valid",
                "context_spatial", "context_temporal", "y", "loss_mask",
            )
        },
        "mask_diagnostics": diagnostics,
        "initial_loss": float(loss.detach().float().cpu()),
        "prediction_min": float(prediction.detach().float().min().cpu()),
        "prediction_max": float(prediction.detach().float().max().cpu()),
        "post_step_prediction_min": float(prediction_after.detach().float().min().cpu()),
        "post_step_prediction_max": float(prediction_after.detach().float().max().cpu()),
        "earlier_day_perturbation_max_abs_change": temporal_diff,
        "reversed_sequence_max_abs_change": reverse_diff,
        "runtime_spec": _b6_runtime_spec(
            args, _RUNTIME_FRAMEWORK_PATH, _RUNTIME_BACKBONE_PATH, norm_report
        ),
        "completed_utc": utc_now(),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "b6_model_integration_test_report.json", report)
    print("[PASS] B6 real-data forward + masked loss + optimizer-step + temporal-path test")
    return report


# ============================================================================
# Canonical OceanEmbed runtime sources embedded for Kaggle portability
# ============================================================================
#
# These are the same frozen loader/backbone/framework sources used by OceanEmbed baselines.
# They are decompressed at runtime into a temporary source tree and checked
# against their canonical SHA-256 digests before import.
#
# Trainer 1.0.2 fixes runtime Backbone provenance validation so the validator
# reads the actual Backbone implementation module + canonical backbone state,
# rather than incorrectly treating backbone_obj.mod (the loader module) as the
# Backbone implementation. Scientific model/data contracts are unchanged.

TRAINER_VERSION = "1.0.2"
GLORYS_FILE_COUNT = 156
SOURCE_SHA256 = {
    "loader": "5ecee73a5d0f9ebf5a6396d0462963217b032e4dd6fdae7a0a111f7a8d826b23",
    "backbone": "700d31e84c6c53caec90fe4426bf82152f0a35adcc6a7a5d651de81c69b70a69",
    "framework": "6a8d0c8dcdb7a4c600acfbb5118cdb207dc03226c2e0abb607b223429e48a789",
}

_OCEANEMBED_SYNCHRONIZED_DATA_LOADER_V1_2_B64 = (
    "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"
)

_OCEANEMBED_DL_DATABACKBONE_V1_B64 = (
    "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"
)

_OCEANEMBED_TRAININGFRAMEWORK_V1_B64 = (
    "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"
)


# ============================================================================
# Canonical Kaggle dataset layout
# ============================================================================
BUNDLE_NAME = "OceanEmbed_ModelData_V1"

PREDICTOR_FILES = [
    "Ccmp_Oscar_2012_2024_100x240_FeaturesV1.nc",
    "Duacs_Oisst_2012_2024_FeaturesV1.nc",
    "Oisss_GlobCurrent_2012_2024_FeaturesV1.nc",
]
STATIC_FILE = "GEBCO_OceanEmbed_Static_0p25deg_FINAL.nc"
RELEVANCE_MASK_FILE = "GEBCO_OceanEmbed_RelevanceMasks_0p25deg_V1.nc"
GLORYS_PATTERN = "GLORYS_thetao_*.nc"

BACKBONE_ARTIFACTS = [
    "backbone_state.json",
    "dataset_manifest.json",
    "predictor_normalization.json",
    "target_normalization.json",
    "target_climatology_monthly.npz",
    "split_manifest.json",
    "train_manifest.jsonl",
    "val_manifest.jsonl",
    "test_manifest.jsonl",
]
def _materialize_sources() -> Path:
    """Decompress and write all canonical source files into a temp runtime tree."""
    runtime_dir = Path(tempfile.mkdtemp(prefix="oceanembed_runtime_"))
    src_data      = runtime_dir / "src" / "data"
    src_framework = runtime_dir / "src" / "framework"
    src_models    = runtime_dir / "src" / "models"
    for d in (src_data, src_framework, src_models):
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")
    (runtime_dir / "__init__.py").write_text("")
    (runtime_dir / "src" / "__init__.py").write_text("")

    _KEY_MAP = {
        "_OCEANEMBED_SYNCHRONIZED_DATA_LOADER_V1_2_B64": "loader",
        "_OCEANEMBED_DL_DATABACKBONE_V1_B64": "backbone",
        "_OCEANEMBED_TRAININGFRAMEWORK_V1_B64": "framework",
    }

    def _write(varname: str, dest: Path) -> None:
        b64 = globals()[varname]
        raw = zlib.decompress(base64.b64decode(b64))
        dest.write_bytes(raw)
        # Verify SHA256 of decompressed bytes against embedded provenance
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        key = _KEY_MAP[varname]
        expected = SOURCE_SHA256[key]
        if actual_sha256 != expected:
            raise RuntimeError(
                f"Embedded source integrity check FAILED for {dest.name}\n"
                f"  Expected SHA256 : {expected}\n"
                f"  Actual SHA256   : {actual_sha256}\n"
                f"The trainer file may be corrupted."
            )

    _write("_OCEANEMBED_SYNCHRONIZED_DATA_LOADER_V1_2_B64", src_data / "OceanEmbed_Synchronized_Data_Loader_V1_2.py")
    _write("_OCEANEMBED_DL_DATABACKBONE_V1_B64",            src_data / "OceanEmbed_DL_DataBackbone_V1.py")
    _write("_OCEANEMBED_TRAININGFRAMEWORK_V1_B64",          src_framework / "OceanEmbed_TrainingFramework_V1.py")
    return runtime_dir

def _find_files_recursive(root: Path, filename: str) -> List[Path]:
    """Recursively find all files with the given name under root."""
    return list(root.rglob(filename))

def _find_glorys_recursive(root: Path) -> List[Path]:
    """Recursively find all GLORYS thetao NetCDF files."""
    import fnmatch
    results = []
    for p in root.rglob("*.nc"):
        if fnmatch.fnmatch(p.name, "GLORYS_thetao_*.nc"):
            results.append(p)
    return sorted(results)

def _find_backbone_artifacts_recursive(root: Path) -> Optional[Path]:
    """Find a directory containing all required backbone artifacts."""
    for p in root.rglob("backbone_state.json"):
        candidate = p.parent
        if all((candidate / a).is_file() for a in BACKBONE_ARTIFACTS):
            return candidate
    return None

def discover_dataset(kaggle_input: Path = Path("/kaggle/input")) -> Dict[str, Any]:
    """Single-pass discovery: one scandir walk collects everything the tree needs."""
    import fnmatch

    print(f"Scanning {kaggle_input} for OceanEmbed dataset components...")
    result: Dict[str, Any] = {
        "predictor_files": {}, "static_file": None, "glorys_files": [],
        "backbone_artifacts_dir": None, "target_zip": None, "backbone_zip": None,
        "kaggle_input": str(kaggle_input),
    }

    predictor_set = set(PREDICTOR_FILES)
    glorys_matches: List[Path] = []
    backbone_state_candidates: List[Path] = []
    target_zip: Optional[Path] = None
    backbone_zip: Optional[Path] = None

    stack: List[Path] = [kaggle_input]
    seen_dirs: set = set()
    while stack:
        d = stack.pop()
        try:
            real = str(d.resolve())
        except OSError:
            real = str(d)
        if real in seen_dirs:
            continue
        seen_dirs.add(real)
        try:
            entries = list(os.scandir(d))
        except (PermissionError, OSError):
            continue
        for entry in entries:
            try:
                name = entry.name
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if name in predictor_set:
                        result["predictor_files"].setdefault(name, Path(entry.path))
                    elif name == STATIC_FILE:
                        if result["static_file"] is None:
                            result["static_file"] = Path(entry.path)
                    elif fnmatch.fnmatch(name, "GLORYS_thetao_*.nc"):
                        glorys_matches.append(Path(entry.path))
                    elif name == "backbone_state.json":
                        backbone_state_candidates.append(Path(entry.path))
                    elif name == "target.zip" and target_zip is None:
                        target_zip = Path(entry.path)
                    elif name == "backbone_artifacts.zip" and backbone_zip is None:
                        backbone_zip = Path(entry.path)
            except OSError:
                continue

    result["glorys_files"] = sorted(glorys_matches)
    for cand in backbone_state_candidates:
        parent = cand.parent
        if all((parent / a).is_file() for a in BACKBONE_ARTIFACTS):
            result["backbone_artifacts_dir"] = parent
            break
    result["target_zip"] = target_zip
    result["backbone_zip"] = backbone_zip
    return result

def _symlink_or_fail(src: Path, dst: Path) -> None:
    """Create a symlink src->dst, failing clearly if symlinks are unsupported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError) as exc:
        raise RuntimeError(
            f"Symlinks are unavailable on this system: {exc}\n"
            f"Cannot link {src} -> {dst}\n"
            f"Refusing to silently copy multi-GB NetCDF files into /kaggle/working."
        ) from exc

def _extract_glorys_from_zip(target_zip: Path, glorys_dest: Path) -> List[Path]:
    """Extract GLORYS thetao files from target.zip into glorys_dest."""
    import fnmatch, zipfile as _zf
    glorys_dest.mkdir(parents=True, exist_ok=True)
    extracted = []
    with _zf.ZipFile(target_zip) as zf:
        members = zf.namelist()
        glorys_members = [m for m in members if fnmatch.fnmatch(Path(m).name, "GLORYS_thetao_*.nc")]
        if not glorys_members:
            raise RuntimeError(
                f"target.zip ({target_zip}) contains no GLORYS_thetao_*.nc files.\n"
                f"Archive contains: {members[:20]}"
            )
        for member in glorys_members:
            fname = Path(member).name
            dest_file = glorys_dest / fname
            if not dest_file.is_file():
                data = zf.read(member)
                dest_file.write_bytes(data)
            extracted.append(dest_file)
    return sorted(extracted)

def _extract_backbone_artifacts_from_zip(backbone_zip: Path, artifacts_dest: Path) -> None:
    """Extract backbone_artifacts.zip into artifacts_dest, stripping archive prefixes."""
    import zipfile as _zf
    artifacts_dest.mkdir(parents=True, exist_ok=True)
    with _zf.ZipFile(backbone_zip) as zf:
        members = zf.namelist()
        for member in members:
            # Strip known prefixes: keep only the filename part
            fname = Path(member).name
            if not fname:
                continue
            dest_file = artifacts_dest / fname
            if not dest_file.is_file():
                data = zf.read(member)
                dest_file.write_bytes(data)

def build_compatibility_tree(
    discovery: Dict[str, Any],
    working_dir: Path = Path("/kaggle/working"),
) -> Dict[str, Any]:
    """
    Build the canonical compatibility tree under working_dir/OceanEmbed_ModelData_V1.
    Uses symlinks for large NetCDF files. Extracts zips as needed.
    Returns paths dict for downstream use.
    """
    bundle_root = working_dir / BUNDLE_NAME
    predictors_dir = bundle_root / "predictors"
    static_dir     = bundle_root / "static"
    glorys_dir     = bundle_root / "target" / "GLORYS"
    manifests_dir  = bundle_root / "manifests" / "data_backbone_V1"

    for d in (predictors_dir, static_dir, glorys_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Predictor files (symlink) ---
    for fname in PREDICTOR_FILES:
        src = discovery["predictor_files"].get(fname)
        if src is None:
            raise RuntimeError(f"Predictor file not found in dataset: {fname}")
        _symlink_or_fail(src.resolve(), predictors_dir / fname)

    # --- Static GEBCO file (symlink) ---
    static_src = discovery["static_file"]
    if static_src is None:
        raise RuntimeError(f"Static file not found in dataset: {STATIC_FILE}")
    _symlink_or_fail(static_src.resolve(), static_dir / STATIC_FILE)

    # --- GLORYS files ---
    glorys_files: List[Path] = []
    if discovery["glorys_files"]:
        # Already extracted - symlink each file
        for gf in discovery["glorys_files"]:
            _symlink_or_fail(gf.resolve(), glorys_dir / gf.name)
            glorys_files.append(glorys_dir / gf.name)
    elif discovery["target_zip"]:
        print(f"Extracting GLORYS from {discovery['target_zip']}...")
        glorys_files = _extract_glorys_from_zip(discovery["target_zip"], glorys_dir)
    else:
        raise RuntimeError(
            "No GLORYS files found. Expected either:\n"
            "  - Extracted GLORYS_thetao_*.nc files anywhere under /kaggle/input\n"
            "  - A target.zip archive anywhere under /kaggle/input"
        )

    actual_count = len(sorted(glorys_dir.glob("GLORYS_thetao_*.nc")))
    if actual_count != GLORYS_FILE_COUNT:
        raise RuntimeError(
            f"Found {actual_count} GLORYS files in compatibility tree. "
            f"Expected exactly {GLORYS_FILE_COUNT}.\n"
            f"GLORYS directory: {glorys_dir}"
        )

    # --- Backbone artifacts ---
    if discovery["backbone_artifacts_dir"]:
        art_src = discovery["backbone_artifacts_dir"]
        for fname in BACKBONE_ARTIFACTS:
            src = art_src / fname
            dst = manifests_dir / fname
            if not dst.is_file():
                _symlink_or_fail(src.resolve(), dst)
    elif discovery["backbone_zip"]:
        print(f"Extracting backbone artifacts from {discovery['backbone_zip']}...")
        _extract_backbone_artifacts_from_zip(discovery["backbone_zip"], manifests_dir)
    else:
        raise RuntimeError(
            "No backbone artifacts found. Expected either:\n"
            "  - Extracted backbone_state.json + normalization files under /kaggle/input\n"
            "  - A backbone_artifacts.zip archive under /kaggle/input"
        )

    # Verify all required artifacts are present
    missing_arts = [a for a in BACKBONE_ARTIFACTS if not (manifests_dir / a).is_file()]
    if missing_arts:
        raise RuntimeError(f"Missing backbone artifacts after setup: {missing_arts}")

    return {
        "bundle_root": bundle_root,
        "predictors_dir": predictors_dir,
        "static_dir": static_dir,
        "glorys_dir": glorys_dir,
        "manifests_dir": manifests_dir,
        "glorys_file_count": actual_count,
    }

def compute_bundle_fingerprint(bundle_root: Path, backbone_mod) -> str:
    """Compute the runtime bundle fingerprint using the canonical Backbone code."""
    from pathlib import Path as _P
    bundle = backbone_mod.Bundle(_P(bundle_root))
    fp, _ = backbone_mod.bundle_fingerprint(bundle, hashes=False)
    return fp

def print_environment() -> None:
    import torch
    print("=" * 72)
    print("OceanEmbed Kaggle Trainer V1  |  Mask-Aware ConvLSTM B6  |  Environment")
    print("=" * 72)
    print(f"Trainer version : {TRAINER_VERSION}")
    print(f"Python version  : {sys.version}")
    print(f"PyTorch version : {torch.__version__}")
    try:
        import numpy as np
        print(f"NumPy version   : {np.__version__}")
    except ImportError:
        pass
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"CUDA version    : {torch.version.cuda}")
            print(f"GPU count       : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}         : {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"  (GPU info error: {e})")
    print(f"Platform        : {platform.platform()}")
    print("=" * 72)

def configure_cuda_backend() -> None:
    """Enable TF32 (Ampere+ only) and optional cuDNN benchmark."""
    import torch
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
        torch.backends.cudnn.allow_tf32 = ENABLE_TF32
        if CUDNN_BENCHMARK:
            torch.backends.cudnn.benchmark = True
        major = torch.cuda.get_device_capability(0)[0]
        note = "" if major >= 8 else " (TF32 requires sm_80+; no-op here)"
        print(f"[CUDA] TF32={ENABLE_TF32}{note}  cuDNN_benchmark={CUDNN_BENCHMARK}  (sm_{major}x)")
    except Exception as e:
        print(f"[CUDA] configuration skipped: {e}")

def package_results(run_dir: Path, working_dir: Path, run_id: str) -> Path:
    """Create result zip from checkpoints, metrics, logs, summary."""
    zip_path = working_dir / f"OceanEmbed_{run_id}_results.zip"
    important_patterns = [
        "best_model.pt", "latest_model.pt", "latest_step_model.pt",
        "training_history.csv", "final_metrics.json", "run_summary.json",
        "run_manifest.json", "metrics/", "logs/", "summary/",
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file() and f.stat().st_size < 200 * 1024 * 1024:  # skip files > 200MB
                arc_name = f.relative_to(run_dir.parent)
                zf.write(f, arc_name)
    return zip_path


# ============================================================================
# Kaggle compatibility layer
# ============================================================================

_RUNTIME_DIR: Optional[Path] = None
_RUNTIME_LOADER_PATH: Optional[Path] = None
_RUNTIME_BACKBONE_PATH: Optional[Path] = None
_RUNTIME_FRAMEWORK_PATH: Optional[Path] = None
_MODEL_SHIM_PATH: Optional[Path] = None


def _write_b6_model_shim(runtime_dir: Path) -> Path:
    """
    The canonical Training Framework imports a model from a .py path.
    In a Kaggle notebook cell the B6 Model class lives in __main__.
    This tiny shim lets the frozen framework import that exact class
    without copying/recompiling the B6 model.
    """
    shim = runtime_dir / "src" / "models" / "OceanEmbed_Model_ConvLSTM_B6.py"
    host_module = __name__
    shim.write_text(
        "\n".join(
            [
                "import importlib",
                f"_HOST = importlib.import_module({host_module!r})",
                "Model = _HOST.Model",
                "MODEL_ID = _HOST.MODEL_ID",
                "MODEL_VERSION = _HOST.MODEL_VERSION",
                "MODEL_CONTRACT = _HOST.MODEL_CONTRACT",
                "",
                "def build_model(contract):",
                "    return Model(contract)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return shim


def _prepare_kaggle_runtime() -> Dict[str, Path]:
    global _RUNTIME_DIR, _RUNTIME_LOADER_PATH, _RUNTIME_BACKBONE_PATH
    global _RUNTIME_FRAMEWORK_PATH, _MODEL_SHIM_PATH

    runtime_dir = _materialize_sources()
    _RUNTIME_DIR = runtime_dir

    _RUNTIME_LOADER_PATH = (
        runtime_dir / "src" / "data" / "OceanEmbed_Synchronized_Data_Loader_V1_2.py"
    )
    _RUNTIME_BACKBONE_PATH = (
        runtime_dir / "src" / "data" / "OceanEmbed_DL_DataBackbone_V1.py"
    )
    _RUNTIME_FRAMEWORK_PATH = (
        runtime_dir / "src" / "framework" / "OceanEmbed_TrainingFramework_V1.py"
    )
    _MODEL_SHIM_PATH = _write_b6_model_shim(runtime_dir)

    return {
        "runtime_dir": runtime_dir,
        "loader_path": _RUNTIME_LOADER_PATH,
        "backbone_path": _RUNTIME_BACKBONE_PATH,
        "framework_path": _RUNTIME_FRAMEWORK_PATH,
        "model_path": _MODEL_SHIM_PATH,
    }


def _resolve_dataset_input_root() -> Path:
    if DATASET_ROOT:
        root = Path(DATASET_ROOT).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Configured DATASET_ROOT does not exist: {root}")
        return root

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        return kaggle_input

    raise FileNotFoundError(
        "No Kaggle /kaggle/input directory found and DATASET_ROOT is not configured."
    )


def _build_args(
    dataset_root: Path,
    run_root: Path,
    output_dir: Path,
    *,
    mode: str,
    device: str = "auto",
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=str(dataset_root),
        run_root=str(run_root),
        framework_path=str(_RUNTIME_FRAMEWORK_PATH),
        backbone_path=str(_RUNTIME_BACKBONE_PATH),
        loader_path=str(_RUNTIME_LOADER_PATH),
        output_dir=str(output_dir),
        experiment_name=MODEL_NAME,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        patience=PATIENCE,
        device=device,
        seed=SEED,
        log_every=LOG_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_every_steps=CHECKPOINT_EVERY_STEPS,
        resume=RESUME_FROM,
        no_amp=not AMP_ENABLED,
        mode=mode,
    )


def _b6_runtime_spec(
    args: argparse.Namespace,
    framework_path: Path,
    backbone_path: Path,
    normalization_report: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    # When running as a real .py file, preserve a hash of the trainer source.
    # In a notebook cell there is no stable __file__, so record that honestly.
    if "__file__" in globals():
        trainer_path = Path(__file__).resolve()
        trainer_sha = sha256_file(trainer_path)
        trainer_source_mode = "python_file"
    else:
        trainer_path = None
        trainer_sha = None
        trainer_source_mode = "kaggle_notebook_cell"

    return {
        "scientific_spec": scientific_spec(),
        "runtime_config": {
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "run_root": str(Path(args.run_root).expanduser().resolve()),
            "experiment_name": args.experiment_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "patience": args.patience,
            "device": args.device,
            "amp": not args.no_amp,
            "seed": args.seed,
            "log_every": args.log_every,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_every_steps": args.checkpoint_every_steps,
        },
        "source_hashes": {
            "trainer": trainer_sha,
            "trainer_source_mode": trainer_source_mode,
            "framework": sha256_file(framework_path),
            "backbone": sha256_file(backbone_path),
        },
        "normalization_guard": normalization_report,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "created_utc": utc_now(),
    }


def _startup_validate_kaggle_bundle(
    bundle_root: Path,
    backbone_mod: ModuleType,
) -> Dict[str, Any]:
    print("")
    print("=" * 72)
    print("STARTUP VALIDATION — B6 Mask-Aware ConvLSTM")
    print("=" * 72)

    bundle = backbone_mod.Bundle(bundle_root)

    for fname in PREDICTOR_FILES:
        path = bundle.predictors / fname
        if not path.is_file():
            raise ContractError(f"Predictor file missing: {path}")
    print("[PASS] 1. All 3 predictor NetCDF files found")

    if not bundle.gebco.is_file():
        raise ContractError(f"Static GEBCO file missing: {bundle.gebco}")
    print("[PASS] 2. Static GEBCO file found")

    glorys_files = sorted(bundle.glorys.glob("GLORYS_thetao_*.nc"))
    if len(glorys_files) != GLORYS_FILE_COUNT:
        raise ContractError(
            f"Found {len(glorys_files)} GLORYS files; expected exactly {GLORYS_FILE_COUNT}"
        )
    print(f"[PASS] 3. GLORYS file count: {len(glorys_files)}")

    missing = [a for a in BACKBONE_ARTIFACTS if not (bundle.state / a).is_file()]
    if missing:
        raise ContractError(f"Missing backbone artifacts: {missing}")
    print("[PASS] 4. All backbone artifacts present")

    runtime_fp = compute_bundle_fingerprint(bundle_root, backbone_mod)
    print(f"       EXPECTED BUNDLE FINGERPRINT : {EXPECTED_BUNDLE_FP}")
    print(f"       RUNTIME  BUNDLE FINGERPRINT : {runtime_fp}")
    if runtime_fp != EXPECTED_BUNDLE_FP:
        raise ContractError(
            "Bundle fingerprint mismatch. Refusing to train against an unpinned bundle."
        )
    print(f"[PASS] 5. Bundle fingerprint MATCH: {runtime_fp[:16]}...")

    state_data = json.loads(
        (bundle.state / "backbone_state.json").read_text(encoding="utf-8")
    )
    if state_data.get("bundle_fingerprint") != EXPECTED_BUNDLE_FP:
        raise ContractError("Stored backbone fingerprint mismatch")
    if str(state_data.get("backbone_version", "")) != EXPECTED_BACKBONE_VERSION:
        raise ContractError(
            f"Backbone version mismatch: {state_data.get('backbone_version')!r}"
        )
    print(
        f"[PASS] 6. Backbone state loads "
        f"(version {state_data.get('backbone_version', '?')})"
    )

    pred_norm = json.loads(
        (bundle.state / "predictor_normalization.json").read_text(encoding="utf-8")
    )
    if len(pred_norm.get("dynamic_mean", [])) != DYNAMIC_CHANNELS:
        raise ContractError("Dynamic normalization shape mismatch")
    if len(pred_norm.get("static_mean", [])) != STATIC_CHANNELS:
        raise ContractError("Static normalization shape mismatch")
    print(
        f"[PASS] 7. Predictor normalization loads "
        f"({DYNAMIC_CHANNELS} dynamic + {STATIC_CHANNELS} static)"
    )

    tgt_norm = json.loads(
        (bundle.state / "target_normalization.json").read_text(encoding="utf-8")
    )
    if len(tgt_norm.get("target_std", [])) != N_DEPTH:
        raise ContractError("Target normalization depth mismatch")
    if tgt_norm.get("target_mode") != "anomaly":
        raise ContractError(
            f"Target mode mismatch: {tgt_norm.get('target_mode')!r}"
        )
    print(
        f"[PASS] 8. Target normalization loads "
        f"(mode={tgt_norm.get('target_mode', '?')})"
    )

    with np.load(bundle.state / "target_climatology_monthly.npz", allow_pickle=False) as clim:
        arr = clim["climatology"]
        if arr.shape != (12, N_DEPTH, GRID_H, GRID_W):
            raise ContractError(
                f"Monthly climatology shape mismatch: {arr.shape}"
            )
    print("[PASS] 9. Monthly climatology loads")

    split_counts = {}
    for split in ("train", "val", "test"):
        p = bundle.state / f"{split}_manifest.jsonl"
        rows = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        split_counts[split] = len(rows)
    print(
        f"[PASS] 10. Split manifests load: "
        f"train={split_counts['train']} | val={split_counts['val']} | "
        f"test={split_counts['test']}"
    )

    bb_cfg = backbone_mod.Config(dataset_root=str(bundle_root))
    bb = backbone_mod.Backbone(bb_cfg)
    bb.load_state()
    contract = bb.model_input_contract()
    validate_contract(contract)
    print("[PASS] 11. Frozen B6 model-input contract validated")

    sample = bb.dataset("train")[0]
    if sample["x_dynamic"].shape != (
        EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W
    ):
        raise ContractError(f"Unexpected real x_dynamic shape: {sample['x_dynamic'].shape}")
    if sample["x_static"].shape != (STATIC_CHANNELS, GRID_H, GRID_W):
        raise ContractError(f"Unexpected real x_static shape: {sample['x_static'].shape}")
    if sample["y"].shape != (N_DEPTH, GRID_H, GRID_W):
        raise ContractError(f"Unexpected real y shape: {sample['y'].shape}")
    if not torch.isfinite(sample["x_dynamic"]).all():
        raise ContractError("Real sample contains non-finite x_dynamic")
    if not torch.isfinite(sample["x_static"]).all():
        raise ContractError("Real sample contains non-finite x_static")
    if not torch.isfinite(sample["y"]).all():
        raise ContractError("Real sample contains non-finite y")
    print(
        f"[PASS] 12. Real training sample retrieved: "
        f"target_date={sample['target_date']}"
    )

    return {
        "bundle_root": str(bundle_root),
        "runtime_bundle_fingerprint": runtime_fp,
        "split_counts": split_counts,
        "real_sample_target_date": str(sample["target_date"]),
    }


def _build_b6_config(
    args: argparse.Namespace,
    framework: ModuleType,
    bundle_root: Path,
) -> Any:
    cfg = framework.TrainConfig(
        dataset_root=str(bundle_root),
        backbone_path=str(_RUNTIME_BACKBONE_PATH),
        loader_path=str(_RUNTIME_LOADER_PATH),
        model_module=str(_MODEL_SHIM_PATH),
        model_class="Model",
        run_root=str(Path(args.run_root).expanduser().resolve()),
        experiment_name=args.experiment_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_steps=args.checkpoint_every_steps,
        resume=args.resume,
    )
    cfg.validate()
    return cfg


def _run_b6_real_integration_test(
    bundle_root: Path,
    args: argparse.Namespace,
    framework: ModuleType,
    output_dir: Path,
) -> Dict[str, Any]:
    print("")
    print("=" * 72)
    print("B6 REAL-DATA INTEGRATION TEST")
    print("=" * 72)

    cfg = _build_b6_config(args, framework, bundle_root)
    adapter = framework.BackboneAdapter(cfg)
    validate_framework_contract(framework)
    validate_backbone_runtime(adapter)
    norm_report = validate_runtime_normalization(adapter)

    trainer = framework.Trainer(cfg, adapter)
    model = trainer.model
    if _trainer_model_identity(model) != MODEL_ID:
        raise IntegrationTestError(
            f"Unexpected instantiated model identity: {_trainer_model_identity(model)!r}"
        )

    raw_batch = next(iter(adapter.train_loader))
    if int(raw_batch["x_dynamic"].shape[0]) != args.batch_size:
        raise IntegrationTestError(
            f"Real batch size {raw_batch['x_dynamic'].shape[0]} != configured {args.batch_size}"
        )

    _, _, _, diagnostics = prepare_model_inputs(raw_batch)
    device = trainer.device
    batch = framework.move_batch_to_device(raw_batch, device)
    model.train()

    autocast_ctx = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(getattr(trainer, "amp_enabled", False)),
        )
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with autocast_ctx:
        prediction = framework.forward_prediction(model, batch)
        loss = framework.masked_mse_loss(
            prediction, batch["y"], batch["loss_mask"]
        )

    expected = (args.batch_size, N_DEPTH, GRID_H, GRID_W)
    if tuple(prediction.shape) != expected:
        raise IntegrationTestError(
            f"Prediction shape {tuple(prediction.shape)} != {expected}"
        )
    if not torch.isfinite(prediction).all():
        raise IntegrationTestError("Real forward produced non-finite prediction")
    if not torch.isfinite(loss):
        raise IntegrationTestError("Real masked loss is non-finite")

    trainer.optimizer.zero_grad(set_to_none=True)
    if getattr(trainer, "amp_enabled", False):
        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        trainer.optimizer.step()

    with torch.no_grad():
        with autocast_ctx:
            prediction_after = framework.forward_prediction(model, batch)

    if not torch.isfinite(prediction_after).all():
        raise IntegrationTestError("Post-step real prediction is non-finite")

    report = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_version": EXPECTED_BACKBONE_VERSION,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "device": str(device),
        "amp_enabled": bool(getattr(trainer, "amp_enabled", False)),
        "parameters": parameter_summary(model),
        "normalization_guard": norm_report,
        "model_summary": model.model_summary(),
        "real_batch_shapes": {
            key: list(batch[key].shape)
            for key in (
                "x_dynamic",
                "x_static",
                "x_dynamic_valid",
                "x_static_valid",
                "context_spatial",
                "context_temporal",
                "y",
                "loss_mask",
            )
        },
        "mask_diagnostics": diagnostics,
        "initial_loss": float(loss.detach().float().cpu()),
        "prediction_min": float(prediction.detach().float().min().cpu()),
        "prediction_max": float(prediction.detach().float().max().cpu()),
        "post_step_prediction_min": float(
            prediction_after.detach().float().min().cpu()
        ),
        "post_step_prediction_max": float(
            prediction_after.detach().float().max().cpu()
        ),
        "runtime_spec": _b6_runtime_spec(
            args,
            _RUNTIME_FRAMEWORK_PATH,
            _RUNTIME_BACKBONE_PATH,
            norm_report,
        ),
        "completed_utc": utc_now(),
    }
    atomic_json(output_dir / "b6_model_integration_test_report.json", report)
    print("[PASS] B6 real-data forward + masked loss + optimizer-step + temporal-path test")
    return report



# ============================================================================
# Long-run runtime monitor / checkpoint snapshotter
# ============================================================================

class _B6RuntimeMonitor:
    """
    Wrapper-level supervisor for long Kaggle runs.

    The canonical Training Framework remains untouched. The monitor only:
      * watches the framework's already-atomic checkpoint files;
      * copies completed checkpoints to immutable, epoch/step-named snapshots;
      * writes a tiny heartbeat JSON;
      * records a Python stack dump if progress appears stalled for a long time.

    It never writes into a checkpoint while the framework is writing it.
    ``atomic_torch_save()`` first writes a .tmp file and then os.replace()s it,
    so a visible checkpoint path is a complete checkpoint before this monitor
    copies it.
    """

    WATCH_FILES = (
        "latest_model.pt",
        "latest_step_model.pt",
        "best_model.pt",
        "interrupted_model.pt",
        "failure_model.pt",
    )

    def __init__(
        self,
        trainer: Any,
        run_dir: Path,
        *,
        poll_seconds: float = CHECKPOINT_POLL_SECONDS,
        heartbeat_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
        heartbeat_print_seconds: float = HEARTBEAT_PRINT_INTERVAL_SEC,
        stall_seconds: float = STALL_DIAGNOSTIC_SECONDS,
        soft_limit_hours: float = SESSION_SOFT_LIMIT_HOURS,
        snapshot_checkpoints: bool = SNAPSHOT_CHECKPOINTS,
    ) -> None:
        self.trainer = trainer
        self.run_dir = Path(run_dir)
        self.poll_seconds = max(float(poll_seconds), 0.5)
        self.heartbeat_seconds = max(float(heartbeat_seconds), 5.0)
        self.heartbeat_print_seconds = max(float(heartbeat_print_seconds), self.heartbeat_seconds)
        self.stall_seconds = max(float(stall_seconds), 60.0)
        self.soft_limit_seconds = max(float(soft_limit_hours), 0.0) * 3600.0
        self.snapshot_checkpoints = bool(snapshot_checkpoints)

        self.snapshot_root = self.run_dir / "checkpoint_snapshots"
        self.epoch_root = self.snapshot_root / "epochs"
        self.step_root = self.snapshot_root / "steps"
        self.event_root = self.snapshot_root / "events"
        for d in (self.epoch_root, self.step_root, self.event_root):
            d.mkdir(parents=True, exist_ok=True)

        self.index_path = self.snapshot_root / "checkpoint_index.json"
        self.heartbeat_path = self.run_dir / "runtime_heartbeat.json"
        self.stall_log_path = self.run_dir / "stall_diagnostics.log"

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = time.perf_counter()
        self._last_heartbeat = 0.0
        self._last_console_heartbeat = 0.0
        self._last_progress_signature: Optional[Tuple[int, int]] = None
        self._last_progress_monotonic = self._started
        self._last_stall_dump_monotonic = self._started
        self._soft_limit_warned = False
        self._seen_mtimes: Dict[str, Tuple[int, int]] = {}
        self._records: List[Dict[str, Any]] = self._load_index()

        try:
            self._expected_train_steps = int(len(trainer.backbone.train_loader))
        except Exception:
            self._expected_train_steps = None

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.index_path.is_file():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else []
        except Exception:
            return []

    def _write_index(self) -> None:
        atomic_json(self.index_path, self._records)

    @staticmethod
    def _atomic_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    @staticmethod
    def _checkpoint_metadata(path: Path) -> Optional[Dict[str, Any]]:
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                return None
            return {
                "epoch": int(payload.get("epoch", 0)),
                "epoch_step": int(payload.get("epoch_step", 0)),
                "global_step": int(payload.get("global_step", 0)),
                "checkpoint_kind": str(payload.get("checkpoint_kind", "unknown")),
                "val_loss": payload.get("val_loss"),
                "best_val_loss": payload.get("best_val_loss"),
                "saved_utc": payload.get("saved_utc"),
            }
        except Exception:
            return None

    def _snapshot_name(self, source_name: str, meta: Dict[str, Any]) -> Tuple[Path, str]:
        epoch = int(meta.get("epoch", 0))
        step = int(meta.get("epoch_step", 0))
        global_step = int(meta.get("global_step", 0))

        if source_name == "latest_model.pt":
            name = f"B6_ConvLSTM_epoch_{epoch:04d}.pt"
            return self.epoch_root / name, "epoch"
        if source_name == "best_model.pt":
            name = f"B6_ConvLSTM_best_epoch_{epoch:04d}.pt"
            return self.epoch_root / name, "best_epoch"
        if source_name == "latest_step_model.pt":
            name = (
                f"B6_ConvLSTM_epoch_{epoch:04d}"
                f"_step_{step:06d}_global_{global_step:08d}.pt"
            )
            return self.step_root / name, "step"
        kind = source_name.removesuffix(".pt")
        name = (
            f"B6_ConvLSTM_{kind}"
            f"_epoch_{epoch:04d}_step_{step:06d}_global_{global_step:08d}.pt"
        )
        return self.event_root / name, kind

    def _capture_one(self, source_name: str, force: bool = False) -> None:
        src = self.run_dir / source_name
        if not src.is_file():
            return

        try:
            stat = src.stat()
        except OSError:
            return

        marker = (int(stat.st_mtime_ns), int(stat.st_size))
        if not force and self._seen_mtimes.get(source_name) == marker:
            return

        meta = self._checkpoint_metadata(src)
        if meta is None:
            # The file may have just appeared; leave it for the next poll.
            return

        dst, kind = self._snapshot_name(source_name, meta)
        if dst.is_file():
            self._seen_mtimes[source_name] = marker
            return

        try:
            # The checkpoint payload has already been successfully parsed above,
            # so it is safe to publish the compatibility identity as a sidecar.
            _write_checkpoint_identity_sidecar(
                src, int(meta["epoch"]), str(meta["checkpoint_kind"])
            )
            self._atomic_copy(src, dst)
            _write_checkpoint_identity_sidecar(
                dst, int(meta["epoch"]), str(meta["checkpoint_kind"])
            )
            captured_utc = utc_now()
            self._records.append({
                "source": source_name,
                "snapshot": str(dst),
                "kind": kind,
                "epoch": int(meta["epoch"]),
                "epoch_step": int(meta["epoch_step"]),
                "global_step": int(meta["global_step"]),
                "checkpoint_kind": meta["checkpoint_kind"],
                "saved_utc": meta["saved_utc"],
                "captured_utc": captured_utc,
                "size_bytes": int(dst.stat().st_size),
            })
            self._write_index()
            self._seen_mtimes[source_name] = marker
            print(
                f"[CKPT] snapshot saved: {dst.name} "
                f"(epoch={meta['epoch']} step={meta['epoch_step']} global={meta['global_step']})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[CKPT] snapshot failed for {source_name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    def capture_now(self) -> None:
        if not self.snapshot_checkpoints:
            return
        for name in self.WATCH_FILES:
            self._capture_one(name, force=True)

    def _progress_state(self) -> Dict[str, Any]:
        try:
            epoch = int(getattr(self.trainer, "current_epoch", 0))
        except Exception:
            epoch = 0
        try:
            step = int(getattr(self.trainer, "current_epoch_step", 0))
        except Exception:
            step = 0
        try:
            global_step = int(getattr(self.trainer, "global_step", 0))
        except Exception:
            global_step = 0

        if self._expected_train_steps is not None:
            if step >= self._expected_train_steps and self._expected_train_steps > 0:
                phase = "post_train_or_validation"
            else:
                phase = "training"
        else:
            phase = "unknown"

        return {
            "current_epoch": epoch,
            "current_epoch_step": step,
            "global_step": global_step,
            "expected_train_steps": self._expected_train_steps,
            "phase_hint": phase,
        }

    def _write_heartbeat(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_heartbeat) < self.heartbeat_seconds:
            return

        progress = self._progress_state()
        elapsed = now - self._started
        latest_checkpoint = None
        latest_step = self.run_dir / "latest_step_model.pt"
        latest_epoch = self.run_dir / "latest_model.pt"
        for p in (latest_step, latest_epoch):
            if p.is_file():
                try:
                    st = p.stat()
                    latest_checkpoint = {
                        "path": str(p),
                        "mtime_utc": datetime.fromtimestamp(
                            st.st_mtime, tz=timezone.utc
                        ).isoformat(),
                        "age_seconds": max(time.time() - st.st_mtime, 0.0),
                    }
                    break
                except OSError:
                    pass

        payload = {
            "status": "RUNNING",
            "heartbeat_utc": utc_now(),
            "elapsed_wall_seconds": float(elapsed),
            "elapsed_wall_hours": float(elapsed / 3600.0),
            "soft_session_limit_hours": float(SESSION_SOFT_LIMIT_HOURS),
            "soft_limit_remaining_hours": (
                None
                if self.soft_limit_seconds <= 0
                else float(max(self.soft_limit_seconds - elapsed, 0.0) / 3600.0)
            ),
            "progress": progress,
            "latest_checkpoint": latest_checkpoint,
            "checkpoint_snapshot_count": len(self._records),
            "monitor_thread_alive": bool(
                self._thread is not None and self._thread.is_alive()
            ),
        }
        try:
            atomic_json(self.heartbeat_path, payload)
        except Exception as exc:
            print(
                f"[MONITOR] heartbeat write failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

        self._last_heartbeat = now
        if (
            force
            or (now - self._last_console_heartbeat) >= self.heartbeat_print_seconds
        ):
            print(
                "[HEARTBEAT] "
                f"elapsed={elapsed/3600.0:.2f}h "
                f"epoch={progress['current_epoch']} "
                f"step={progress['current_epoch_step']} "
                f"global={progress['global_step']} "
                f"snapshots={len(self._records)}",
                flush=True,
            )
            self._last_console_heartbeat = now

    def _diagnose_stall(self) -> None:
        progress = self._progress_state()
        signature = (
            int(progress["current_epoch"]),
            int(progress["global_step"]),
        )
        now = time.perf_counter()

        if self._last_progress_signature is None:
            self._last_progress_signature = signature
            self._last_progress_monotonic = now
            return

        if signature != self._last_progress_signature:
            self._last_progress_signature = signature
            self._last_progress_monotonic = now
            return

        stalled_for = now - self._last_progress_monotonic
        if stalled_for < self.stall_seconds:
            return

        if (now - self._last_stall_dump_monotonic) < self.stall_seconds:
            return

        # This is intentionally diagnostic-only: it NEVER interrupts training.
        try:
            self.stall_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.stall_log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"\n===== STALL DIAGNOSTIC {utc_now()} =====\n"
                    f"stalled_for_seconds={stalled_for:.1f}\n"
                    f"progress={json.dumps(progress, sort_keys=True)}\n"
                )
                faulthandler.dump_traceback(file=fh, all_threads=True)
                fh.write("===== END STALL DIAGNOSTIC =====\n")
            print(
                f"[MONITOR] No global-step progress for {stalled_for/60.0:.1f} min; "
                f"stack trace appended to {self.stall_log_path.name}. "
                f"Training was NOT interrupted.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[MONITOR] stall diagnostic failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

        self._last_stall_dump_monotonic = now

    def _soft_limit_warning(self) -> None:
        if self._soft_limit_warned or self.soft_limit_seconds <= 0:
            return
        elapsed = time.perf_counter() - self._started
        if elapsed >= self.soft_limit_seconds:
            self._soft_limit_warned = True
            remaining = max(SESSION_HARD_LIMIT_HOURS * 3600.0 - elapsed, 0.0)
            self.capture_now()
            print(
                "[SESSION] Soft runtime limit reached "
                f"({SESSION_SOFT_LIMIT_HOURS:.2f}h). "
                f"Last atomic checkpoints were captured. "
                f"Approximate time to 12h hard cap: {remaining/3600.0:.2f}h. "
                "The monitor will NOT attempt an unsafe concurrent checkpoint.",
                flush=True,
            )

    def _loop(self) -> None:
        self._write_heartbeat(force=True)
        while not self._stop.wait(self.poll_seconds):
            self.capture_now()
            self._write_heartbeat()
            self._diagnose_stall()
            self._soft_limit_warning()
        self.capture_now()
        self._write_heartbeat(force=True)

    def start(self) -> None:
        if not ENABLE_RUNTIME_MONITOR:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="b6-runtime-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.poll_seconds * 3.0))
        try:
            self.capture_now()
            self._write_heartbeat(force=True)
        except Exception:
            pass
        print("[MONITOR] runtime monitor stopped.", flush=True)


def _train_b6_kaggle(
    bundle_root: Path,
    args: argparse.Namespace,
    framework: ModuleType,
    working_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    cfg = _build_b6_config(args, framework, bundle_root)
    adapter = framework.BackboneAdapter(cfg)
    validate_backbone_runtime(adapter)
    norm_report = validate_runtime_normalization(adapter)

    # Validate/select resume metadata BEFORE the canonical Trainer attempts to load it.
    # The framework checkpoint payload does not encode MODEL_ID directly, so B6
    # enforces experiment_name + model contract + bundle fingerprint here.
    resume_probe = Model(adapter.contract)
    if FORCE_FRESH_START:
        args.resume = None
        cfg.resume = None
        print("[RESUME] FORCE_FRESH_START=True; starting B6 from scratch.", flush=True)
    elif args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resume_report = _validate_resume_checkpoint_metadata(resume_path, resume_probe)
        print("[RESUME] Explicit compatible B6 checkpoint verified:", resume_report, flush=True)
        args.resume = str(resume_path)
        cfg.resume = str(resume_path)
    elif AUTO_RESUME_COMPATIBLE:
        discovered = _find_compatible_b6_checkpoint(Path("/kaggle/input"), resume_probe)
        if discovered is not None:
            args.resume = str(discovered)
            cfg.resume = str(discovered)
        else:
            print("[RESUME] No compatible B6 checkpoint found; starting from epoch 1.", flush=True)
    else:
        print("[RESUME] Auto-discovery disabled; starting from epoch 1.", flush=True)

    cfg.validate()
    trainer = framework.Trainer(cfg, adapter)
    if _trainer_model_identity(trainer.model) != MODEL_ID:
        raise IntegrationTestError(
            "Training Framework instantiated unexpected model identity: "
            f"{_trainer_model_identity(trainer.model)!r}"
        )

    run_dir = Path(trainer.run_dir)

    # A latest-only resumed run restores model/optimizer state but starts with a
    # fresh run directory. Preserve the historical validation-best checkpoint
    # before fit/evaluate_test() so the canonical framework can load it later.
    historical_best_report = _preserve_historical_best_checkpoint(
        run_dir,
        trainer.model,
        Path(args.resume).expanduser().resolve() if args.resume else None,
    )

    start_spec = {
        "status": "STARTED",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "backbone_contract": EXPECTED_BACKBONE_CONTRACT,
        "framework_contract": EXPECTED_FRAMEWORK_CONTRACT,
        "model_parameters": parameter_summary(trainer.model),
        "normalization_guard": norm_report,
        "runtime_spec": _b6_runtime_spec(
            args, _RUNTIME_FRAMEWORK_PATH, _RUNTIME_BACKBONE_PATH, norm_report
        ),
        "model_summary": trainer.model.model_summary(),
        "historical_best_restore": historical_best_report,
        "started_utc": utc_now(),
    }
    atomic_json(run_dir / "b6_convlstm_experiment_spec.json", start_spec)

    monitor = _B6RuntimeMonitor(trainer, run_dir)

    try:
        print("")
        print("=" * 72)
        print("STARTING B6 TRAINING")
        print("=" * 72)
        print(
            f"Model={MODEL_NAME} | epochs={EPOCHS} | batch_size={BATCH_SIZE} | "
            f"workers={NUM_WORKERS} | lr={LEARNING_RATE} | wd={WEIGHT_DECAY} | "
            f"amp={trainer.amp_enabled}"
        )
        print(
            f"Checkpoint policy: every {args.checkpoint_every} epoch(s), "
            f"every {args.checkpoint_every_steps} global step(s)"
        )
        print(
            f"Immutable snapshots: {'ON' if SNAPSHOT_CHECKPOINTS else 'OFF'} | "
            f"heartbeat={HEARTBEAT_INTERVAL_SECONDS:.0f}s | "
            f"stall diagnostic={STALL_DIAGNOSTIC_SECONDS/60.0:.0f}min"
        )

        monitor.start()

        training_summary = trainer.fit()
        test_metrics = trainer.evaluate_test()

        final = {
            **start_spec,
            "status": "COMPLETED",
            "parameters": parameter_summary(trainer.model),
            "run_dir": str(run_dir),
            "training_summary": training_summary,
            "test_metrics": test_metrics,
            "completed_utc": utc_now(),
        }
        atomic_json(run_dir / "final_metrics.json", test_metrics)
        atomic_json(
            run_dir / "run_summary.json",
            {
                "trainer_version": TRAINER_VERSION,
                "model_name": MODEL_NAME,
                "experiment_id": EXPERIMENT_ID,
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "bundle_fingerprint": EXPECTED_BUNDLE_FP,
                "source_sha256": SOURCE_SHA256,
                "training_summary": training_summary,
                "test_metrics": test_metrics,
                "completed_utc": utc_now(),
                "runtime_config": {
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "epochs": EPOCHS,
                    "patience": PATIENCE,
                    "tf32_enabled": ENABLE_TF32,
                    "cudnn_benchmark": CUDNN_BENCHMARK,
                    "runtime_monitor_enabled": ENABLE_RUNTIME_MONITOR,
                    "checkpoint_snapshot_poll_seconds": CHECKPOINT_POLL_SECONDS,
                    "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                    "stall_diagnostic_seconds": STALL_DIAGNOSTIC_SECONDS,
                    "session_soft_limit_hours": SESSION_SOFT_LIMIT_HOURS,
                    "session_hard_limit_hours": SESSION_HARD_LIMIT_HOURS,
                },
            },
        )
        atomic_json(run_dir / "b6_convlstm_experiment_spec.json", final)

        # Capture the final atomic checkpoints before packaging.
        monitor.capture_now()

        zip_path = package_results(run_dir, working_dir, run_id)
        print("")
        print(f"RESULT ZIP: {zip_path.resolve()}", flush=True)
        print(f"RUN DIR  : {run_dir.resolve()}", flush=True)
        print(
            "[CHECKPOINT] Identity schema: "
            f"{CHECKPOINT_IDENTITY_SCHEMA} | "
            "resume compatibility requires experiment_id/model_id/model_version/"
            "model_contract/bundle_fingerprint metadata.",
            flush=True,
        )
        print("[FINAL] B6 trainer completed; all outputs have been written.", flush=True)
        return final

    except Exception as exc:
        # The canonical framework already writes its emergency/failure
        # checkpoint before re-raising. Snapshot it here before shutdown.
        try:
            monitor.capture_now()
        except Exception:
            pass
        failure = {
            **start_spec,
            "status": "FAILED",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "failed_utc": utc_now(),
        }
        atomic_json(run_dir / "b6_convlstm_experiment_spec.json", failure)
        raise
    finally:
        try:
            monitor.stop()
        except Exception:
            pass
        try:
            ld = getattr(adapter.backbone, "loader", None)
            if ld is not None:
                ld.close()
        except Exception:
            pass


def _in_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell", "TerminalInteractiveShell", "Shell")
    except NameError:
        return False


def _parse_kaggle_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OceanEmbed B6 Mask-Aware ConvLSTM Kaggle Trainer"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument(
        "--checkpoint-every-steps", type=int, default=CHECKPOINT_EVERY_STEPS
    )
    parser.add_argument("--resume", default=RESUME_FROM)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args([] if _in_notebook() else None)

    # In notebook mode, configuration is controlled by RUN_MODE above.
    if _in_notebook():
        if RUN_MODE not in {"train", "self-test", "test"}:
            raise ValueError(
                f"RUN_MODE must be 'train', 'test', or 'self-test'; got {RUN_MODE!r}"
            )
        args.self_test = RUN_MODE == "self-test"
        args.test = RUN_MODE == "test"
        args.train = RUN_MODE == "train"

    if not (args.self_test or args.test or args.train):
        # CLI script with no mode mirrors B2: normal execution means training.
        args.train = True

    if sum(bool(x) for x in (args.self_test, args.test, args.train)) != 1:
        raise ValueError("Choose exactly one of --self-test, --test, --train")

    return args


def main_kaggle() -> int:
    args = _parse_kaggle_args()

    if args.self_test:
        report = run_self_test()
        print("")
        print(json.dumps(sanitize_json(report), indent=2))
        return 0

    working_dir = Path(WORKING_DIR).expanduser().resolve()
    working_dir.mkdir(parents=True, exist_ok=True)
    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root
        else working_dir / "OceanEmbed_Runs"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else working_dir / "B6_IntegrationTest"
    )
    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    runtime = _prepare_kaggle_runtime()
    print_environment()
    configure_cuda_backend()

    print("")
    print("PROVENANCE")
    print("-" * 72)
    print(f"Trainer version     : {TRAINER_VERSION}")
    print(f"Experiment          : {EXPERIMENT_ID}")
    print(f"Model               : {MODEL_NAME}")
    print(f"Model contract      : {MODEL_CONTRACT}")
    print(f"Bundle fingerprint  : {EXPECTED_BUNDLE_FP}")
    print(f"Dynamic channels    : {DYNAMIC_CHANNELS}")
    print(f"Static channels     : {STATIC_CHANNELS}")
    print(f"Target depths       : {N_DEPTH}")
    print(f"Input window        : {EXPECTED_WINDOW_DAYS} days")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Num workers         : {NUM_WORKERS}")
    print("")
    print("Embedded canonical source SHA-256:")
    for key, value in SOURCE_SHA256.items():
        print(f"  {key:<10}: {value}")
    print("")

    try:
        dataset_input = (
            Path(args.dataset_root).expanduser().resolve()
            if args.dataset_root
            else _resolve_dataset_input_root()
        )

        print("Discovering dataset under:", dataset_input)
        discovery = discover_dataset(dataset_input)

        missing_preds = [
            f for f in PREDICTOR_FILES if f not in discovery["predictor_files"]
        ]
        if missing_preds:
            raise ContractError(f"Missing predictor files: {missing_preds}")
        if discovery["static_file"] is None:
            raise ContractError(f"Static file not found: {STATIC_FILE}")
        if not discovery["glorys_files"] and discovery["target_zip"] is None:
            raise ContractError("No GLORYS files and no target.zip found")

        print(
            f"Found predictor files={len(discovery['predictor_files'])}, "
            f"GLORYS files={len(discovery['glorys_files'])}"
        )
        if discovery["backbone_artifacts_dir"]:
            print(f"Backbone artifacts: {discovery['backbone_artifacts_dir']}")
        elif discovery["backbone_zip"]:
            print(f"Backbone zip: {discovery['backbone_zip']}")

        compat = build_compatibility_tree(discovery, working_dir)
        bundle_root = Path(compat["bundle_root"])
        print(f"Compatibility bundle: {bundle_root}")

        backbone_mod = _import_module(
            Path(_RUNTIME_BACKBONE_PATH), "oceanembed_b6_backbone"
        )
        framework_mod = _import_module(
            Path(_RUNTIME_FRAMEWORK_PATH), "oceanembed_b6_framework"
        )

        validate_framework_contract(framework_mod)
        startup = _startup_validate_kaggle_bundle(bundle_root, backbone_mod)

        args.dataset_root = str(bundle_root)
        args.run_root = str(run_root)
        args.output_dir = str(output_dir)
        args.framework_path = str(_RUNTIME_FRAMEWORK_PATH)
        args.backbone_path = str(_RUNTIME_BACKBONE_PATH)
        args.loader_path = str(_RUNTIME_LOADER_PATH)
        args.experiment_name = MODEL_NAME
        args.device = "auto"
        args.no_amp = not AMP_ENABLED
        args.mode = "train" if args.train else "test"

        # Overwrite any command/config values that came through argparse.
        args.epochs = int(args.epochs)
        args.batch_size = int(args.batch_size)
        args.num_workers = int(args.num_workers)
        args.learning_rate = float(args.learning_rate)
        args.weight_decay = float(args.weight_decay)
        args.grad_clip_norm = float(args.grad_clip_norm)
        args.patience = int(args.patience)
        args.seed = int(args.seed)
        args.log_every = int(args.log_every)
        args.checkpoint_every = int(args.checkpoint_every)
        args.checkpoint_every_steps = int(args.checkpoint_every_steps)

        if args.test:
            report = _run_b6_real_integration_test(
                bundle_root, args, framework_mod, output_dir
            )
            print("")
            print(json.dumps(sanitize_json(report), indent=2))
            return 0

        if RUN_PRETRAIN_INTEGRATION_TEST:
            _run_b6_real_integration_test(
                bundle_root,
                args,
                framework_mod,
                output_dir,
            )

        result = _train_b6_kaggle(
            bundle_root,
            args,
            framework_mod,
            working_dir,
            run_id,
        )

        print("")
        print("B6 TRAINING FINISHED SUCCESSFULLY")
        print(json.dumps(sanitize_json(result), indent=2))
        return 0

    finally:
        # Do not delete the runtime tree before imported framework/model objects
        # are fully done. At normal completion it can then be cleaned safely.
        try:
            if _RUNTIME_DIR is not None:
                shutil.rmtree(_RUNTIME_DIR, ignore_errors=True)
        except Exception:
            pass


# Kaggle/Jupyter-safe entry point.
# In a real .py process we return a normal OS exit code. In Kaggle/Jupyter,
# do NOT raise SystemExit(0): IPython renders that as a visible exception
# even when the computation succeeded. main_kaggle() already performs the
# full shutdown/cleanup path before returning.
if __name__ == "__main__":
    _KAGGLE_RC = main_kaggle()

    if _in_notebook():
        print(
            f"[FINAL] B6 completed successfully; notebook returning normally "
            f"(code={_KAGGLE_RC}).",
            flush=True,
        )
    else:
        raise SystemExit(0 if _KAGGLE_RC in (0, None) else int(_KAGGLE_RC))
