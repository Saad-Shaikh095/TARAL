"""
OceanEmbed H1 — PAM-ConvLSTM / B7
Physics-Aware Multi-Modal Masked ConvLSTM

Scientific experiment:
    H1 / B7
    PAM_CONVLSTM_B7
    "B7 — Physics-Aware Multi-Modal Masked ConvLSTM"
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================================
# H1 runtime configuration
# ============================================================================

RUN_MODE = "train"  # "train" or "self-test"

EXPERIMENT_ID = "H1"
MODEL_ID = "PAM_CONVLSTM_B7"
MODEL_VERSION = "7.0.0"
MODEL_CONTRACT = "PAMCONVLSTMB7.0"
MODEL_NAME = MODEL_ID

REFERENCE_B6_RMSE_WEIGHTED = 0.706064451
REFERENCE_B6_VAL_LOSS = 0.9249299274
REFERENCE_B6_EPOCH = 10
REFERENCE_B6_METRIC_NAME = "RMSE_weighted"
EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)

EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 4
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
SEED = 42
LOG_EVERY = 10
AMP_ENABLED = True

CALIBRATION_BATCHES = 10
CALIBRATION_EPS = 1e-6

IMPROVEMENT_THRESHOLD_C = 0.005

WORKING_DIR = "/kaggle/working"
RUN_ROOT = "/kaggle/working/OceanEmbed_Runs"

# ---------------------------------------------------------------------------
# OPTIONAL WARM START (H1 trainable weights only — NOT a full optimizer resume)
# ---------------------------------------------------------------------------
# Set this to the path of a previous H1 latest_model.pt / best_model.pt (e.g.
# attached as a Kaggle input dataset) to load the trainable modality
# encoders/fusion/adapter weights before training starts. Leave as "" to
# start fresh (current default behavior, unchanged).
#
# NOTE: this restores model weights only. Optimizer momentum, the
# ReduceLROnPlateau scheduler state, and the epoch/patience counters all
# restart fresh from this point. That is a deliberate, safe trade-off: the
# exact TrainConfig `resume` contract (resume_source/resume_policy) was not
# available to verify at the time this was written, so a full optimizer/
# scheduler resume is not attempted here to avoid guessing at an unverified
# API against a time-constrained run. Loading is best-effort and never
# raises — on any failure it prints a clear warning and falls back to a
# fresh model rather than aborting the run.
RESUME_FROM_H1_CHECKPOINT = ""  # e.g. "/kaggle/input/oceanembed-h1-ckpt/latest_model.pt"

# ---------------------------------------------------------------------------
# PATH OVERRIDES FOR THIS ENVIRONMENT
# ---------------------------------------------------------------------------
# Leave B6_TRAINER_PATH / B6_CHECKPOINT_PATH as None to auto-discover.
# DATASET_ROOT must point to the directory that CONTAINS `backbone_artifacts/`
# and `target/`.  If left as None the code auto-discovers it, but we set it
# explicitly here to make the environment deterministic.
B6_TRAINER_PATH: Optional[str] = None
B6_CHECKPOINT_PATH: Optional[str] = None
DATASET_ROOT: Optional[str] = (
    "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"
)

PREFLIGHT_BATCHES = 1

FINGERPRINT_KEYS = ("backbone_fingerprint",)
EXPERIMENT_NAME_VALUES = {"CONVLSTM_MASKED_B6"}


# ============================================================================
# Frozen H1 channel contract
# ============================================================================

GRID_H, GRID_W = 100, 240
WINDOW_DAYS = 14
DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2
TOTAL_CHANNELS = 36
N_DEPTH = 15

CHANNEL_GROUPS: Dict[str, Tuple[int, ...]] = {
    "sea_level": (0, 1),
    "thermal": (4, 5, 6, 8),
    "salinity": (9, 10, 11, 15),
    "currents": (2, 3, 7, 12, 13, 14, 16, 19, 20, 22, 23),
    "wind": (17, 18, 21),
    "static": (24, 25, 26, 27, 28, 29),
    "context": (30, 31, 32, 33, 34, 35),
}

BRANCH_WIDTHS = {
    "sea_level": 8,
    "thermal": 8,
    "salinity": 8,
    "currents": 12,
    "wind": 8,
    "static": 8,
    "context": 4,
}

FUSION_INPUT_CHANNELS = sum(BRANCH_WIDTHS.values())
FUSION_OUTPUT_CHANNELS = 32

assert FUSION_INPUT_CHANNELS == 56


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
    }


PREFLIGHT_PASS_MARKER = Path(RUN_ROOT) / "H1_B7_PREFLIGHT_PASS.json"


def _script_fingerprint() -> str:
    """Fingerprint this exact H1 source so a PASS cannot survive a code change."""
    source_path = Path(__file__).resolve()
    if not source_path.is_file():
        return "unknown"
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _preflight_marker_payload(checkpoint: Path, dataset_root: Path) -> Dict[str, Any]:
    return {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "script_sha256": _script_fingerprint(),
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "reference_epoch": REFERENCE_B6_EPOCH,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "created_by": "run_self_test",
    }


def _write_preflight_pass_marker(checkpoint: Path, dataset_root: Path) -> None:
    payload = _preflight_marker_payload(checkpoint, dataset_root)
    atomic_json(PREFLIGHT_PASS_MARKER, payload)
    print(f"[PREFLIGHT] PASS marker written: {PREFLIGHT_PASS_MARKER}", flush=True)


def _read_valid_preflight_pass_marker(
    checkpoint: Path,
    dataset_root: Path,
) -> Optional[Dict[str, Any]]:
    """Return prior PASS only when it matches this exact source + frozen contract."""
    if not PREFLIGHT_PASS_MARKER.is_file():
        return None
    try:
        payload = json.loads(PREFLIGHT_PASS_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return None

    required = {
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "script_sha256": _script_fingerprint(),
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "reference_epoch": REFERENCE_B6_EPOCH,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            return None

    # Kaggle session paths may differ; identity is pinned by the source,
    # bundle fingerprint, and exact B6 epoch rather than raw path strings.
    if Path(payload.get("checkpoint", "")).name != checkpoint.name:
        return None
    if Path(payload.get("dataset_root", "")).name != dataset_root.name:
        return None

    return payload


def import_module_from_path(path: Path, prefix: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{prefix}_{hashlib.sha1(str(path).encode()).hexdigest()[:12]}",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Dataset root resolution + view builder (env-specific fix)
# ============================================================================

def _discover_actual_dataset_root() -> Path:
    """Find the real OceanEmbed model-data root containing the frozen bundle."""
    search_roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("backbone_state.json"):
            parent = p.parent
            if parent.name == "backbone_artifacts":
                return parent.parent
    raise FileNotFoundError(
        "Could not locate backbone_state.json under /kaggle/input or "
        "/kaggle/working. Attach the OceanEmbed model-data dataset."
    )


def resolve_dataset_root(b6: ModuleType) -> Path:
    """Resolve the real model-data root and let the exact B6 compatibility
    builder create the canonical runtime bundle.

    This deliberately uses B6's own discover_dataset() +
    build_compatibility_tree() instead of constructing a parallel symlink view.
    A2.4 already validated this exact B6 path, and it preserves the frozen
    Backbone artifact layout expected by TrainingFramework V1.1.
    """
    if DATASET_ROOT:
        actual = Path(DATASET_ROOT).expanduser().resolve()
        if not actual.is_dir():
            raise FileNotFoundError(f"DATASET_ROOT does not exist: {actual}")
    elif hasattr(b6, "_resolve_dataset_input_root"):
        actual = Path(b6._resolve_dataset_input_root()).resolve()
    else:
        actual = _discover_actual_dataset_root()

    print(f"[DATA] actual dataset root: {actual}", flush=True)

    if not hasattr(b6, "discover_dataset"):
        raise RuntimeError(
            "B6 trainer is missing discover_dataset(); cannot construct the "
            "canonical frozen Backbone compatibility bundle."
        )
    if not hasattr(b6, "build_compatibility_tree"):
        raise RuntimeError(
            "B6 trainer is missing build_compatibility_tree(); cannot construct "
            "the canonical frozen Backbone compatibility bundle."
        )

    discovery = b6.discover_dataset(actual)

    missing_predictors = [
        name for name in getattr(b6, "PREDICTOR_FILES", [])
        if name not in discovery.get("predictor_files", {})
    ]
    if missing_predictors:
        raise RuntimeError(
            "Canonical B6 dataset discovery is missing predictor file(s): "
            + ", ".join(missing_predictors)
        )
    if discovery.get("static_file") is None:
        raise RuntimeError("Canonical B6 dataset discovery did not find the GEBCO static file.")
    if not discovery.get("glorys_files") and discovery.get("target_zip") is None:
        raise RuntimeError("Canonical B6 dataset discovery found no GLORYS files and no target.zip.")
    if discovery.get("backbone_artifacts_dir") is None:
        raise RuntimeError("Canonical B6 dataset discovery did not find the frozen Backbone artifacts.")

    compat = b6.build_compatibility_tree(
        discovery,
        Path(WORKING_DIR),
    )
    bundle_root = Path(compat["bundle_root"]).resolve()

    required = bundle_root / "manifests" / "data_backbone_V1" / "backbone_state.json"
    if not required.is_file():
        raise FileNotFoundError(
            "Canonical B6 compatibility bundle is missing backbone_state.json: "
            f"{required}"
        )

    print(f"[DATA] canonical B6 compatibility bundle: {bundle_root}", flush=True)
    print(f"[DATA] frozen Backbone state: {required}", flush=True)
    return bundle_root


# ============================================================================
# B6 runtime discovery
# ============================================================================

def find_b6_trainer() -> Path:
    if B6_TRAINER_PATH:
        p = Path(B6_TRAINER_PATH).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"B6_TRAINER_PATH does not exist: {p}")
        return p

    candidates: List[Path] = []
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    pattern = "OceanEmbed_KaggleTrainer_B6_ConvLSTM_15Epoch_LatestOnlyResume_BestRestore*.py"
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob(pattern))

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate the B6 trainer source. Attach the exact B6 trainer "
            "Python file as a Kaggle input dataset, or set B6_TRAINER_PATH."
        )
    return candidates[0]


def load_b6_runtime() -> Tuple[ModuleType, Dict[str, Path]]:
    b6_path = find_b6_trainer()
    print(f"[RUNTIME] B6 trainer: {b6_path}", flush=True)
    b6 = import_module_from_path(b6_path, "oceanembed_b6_host")

    required = [
        "Model",
        "prepare_model_inputs",
        "derive_ocean_mask_from_static",
        "_validate_resume_checkpoint_metadata",
    ]
    missing = [name for name in required if not hasattr(b6, name)]
    if missing:
        raise RuntimeError(
            f"B6 trainer source is incompatible; missing symbols: {missing}"
        )

    if hasattr(b6, "_prepare_kaggle_runtime"):
        runtime = b6._prepare_kaggle_runtime()
    elif hasattr(b6, "_materialize_sources"):
        runtime_dir = b6._materialize_sources()
        runtime = {
            "runtime_dir": runtime_dir,
            "loader_path": runtime_dir / "src" / "data" / "OceanEmbed_Synchronized_Data_Loader_V1_2.py",
            "backbone_path": runtime_dir / "src" / "data" / "OceanEmbed_DL_DataBackbone_V1.py",
            "framework_path": runtime_dir / "src" / "framework" / "OceanEmbed_TrainingFramework_V1.py",
        }
    else:
        raise RuntimeError(
            "B6 trainer does not expose its embedded runtime materialization."
        )

    return b6, {k: Path(v) for k, v in runtime.items()}


_CHECKPOINT_CACHE: Optional[Path] = None


def _load_checkpoint_payload(p: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_matches_b6_identity(payload: Dict[str, Any]) -> bool:
    fingerprint = None
    for key in FINGERPRINT_KEYS:
        if key in payload:
            fingerprint = payload[key]
            break
    if fingerprint != EXPECTED_BUNDLE_FP:
        return False

    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        return False
    experiment_name = str(cfg.get("experiment_name", ""))
    if experiment_name not in EXPERIMENT_NAME_VALUES:
        return False

    try:
        epoch = int(payload.get("epoch", -1))
    except (TypeError, ValueError):
        return False
    if epoch != REFERENCE_B6_EPOCH:
        return False

    return True


def _validate_exact_epoch10_checkpoint_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("B6 checkpoint payload is not a dictionary")

    if payload.get("backbone_fingerprint") != EXPECTED_BUNDLE_FP:
        raise RuntimeError("B6 checkpoint fingerprint does not match the pinned bundle")

    try:
        epoch = int(payload.get("epoch", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("B6 checkpoint epoch is invalid") from exc
    if epoch != REFERENCE_B6_EPOCH:
        raise RuntimeError(
            f"H1 requires B6 epoch-{REFERENCE_B6_EPOCH}; checkpoint contains epoch-{epoch}."
        )

    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("B6 checkpoint config metadata is missing")
    exp = str(cfg.get("experiment_name", ""))
    if exp not in EXPERIMENT_NAME_VALUES:
        raise RuntimeError(
            f"Checkpoint experiment_name={exp!r} is not an accepted B6 identity."
        )

    if "model_state_dict" not in payload:
        raise RuntimeError("B6 checkpoint lacks model_state_dict")


def discover_checkpoint(b6: ModuleType) -> Path:
    del b6
    global _CHECKPOINT_CACHE
    if _CHECKPOINT_CACHE is not None:
        return _CHECKPOINT_CACHE

    if B6_CHECKPOINT_PATH:
        p = Path(B6_CHECKPOINT_PATH).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"B6_CHECKPOINT_PATH does not exist: {p}")
        _CHECKPOINT_CACHE = p
        return p

    candidates: List[Path] = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            candidates.extend(root.rglob("latest_model.pt"))

    if not candidates:
        raise FileNotFoundError(
            "No latest_model.pt found under /kaggle/input or /kaggle/working. "
            "Attach the exact B6 epoch-10 checkpoint."
        )

    for p in sorted(set(candidates), key=lambda x: (len(str(x)), str(x))):
        payload = _load_checkpoint_payload(p)
        if payload is None:
            continue
        if _checkpoint_matches_b6_identity(payload):
            _CHECKPOINT_CACHE = p
            return p

    raise RuntimeError(
        "latest_model.pt files were found, but none passed the pinned B6 "
        "fingerprint + experiment identity check."
    )


# ============================================================================
# Mask-aware modality branch
# ============================================================================

class ModalityBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        partial_conv_cls: type,
    ):
        super().__init__()
        self.conv1 = partial_conv_cls(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=True,
        )
        self.conv2 = partial_conv_cls(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            bias=True,
        )
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool() & ocean_mask.bool()
        y, support = self.conv1(x, mask.to(dtype=x.dtype))
        y = self.act(y)
        y = y * support * ocean_mask
        support = support * ocean_mask

        y, support = self.conv2(y, support)
        y = self.act(y)
        y = y * support * ocean_mask
        support = support * ocean_mask
        return y, support


# ============================================================================
# Calibrated fusion normalization
# ============================================================================

class CalibratedAffine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.register_buffer("reference_mean", torch.zeros(channels), persistent=True)
        self.register_buffer("reference_std", torch.ones(channels), persistent=True)
        self.register_buffer("source_mean", torch.zeros(channels), persistent=True)
        self.register_buffer("source_std", torch.ones(channels), persistent=True)
        self.register_buffer("calibration_done", torch.tensor(False), persistent=True)

    @staticmethod
    def _masked_channel_stats(x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        mf = mask.float().expand_as(xf)
        count = mf.sum(dim=(0, 2, 3)).clamp_min(1.0)
        mean = (xf * mf).sum(dim=(0, 2, 3)) / count
        var = (((xf - mean.view(1, -1, 1, 1)) ** 2) * mf).sum(dim=(0, 2, 3)) / count
        std = var.clamp_min(CALIBRATION_EPS).sqrt()
        return mean, std

    @staticmethod
    def _init_stats(channels: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "source_sum": torch.zeros(channels, dtype=torch.float64, device=device),
            "source_sumsq": torch.zeros(channels, dtype=torch.float64, device=device),
            "reference_sum": torch.zeros(channels, dtype=torch.float64, device=device),
            "reference_sumsq": torch.zeros(channels, dtype=torch.float64, device=device),
            "count": torch.zeros(channels, dtype=torch.float64, device=device),
        }

    @staticmethod
    @torch.no_grad()
    def accumulate_pair_stats(
        stats: Dict[str, torch.Tensor],
        source: torch.Tensor,
        source_mask: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> None:
        common = source_mask.bool() & reference_mask.bool()
        if common.ndim != 4 or common.shape[1] != 1:
            raise ValueError("Calibration masks must be [B,1,H,W]")
        m = common.float().expand_as(source.float())
        s = source.float()
        r = reference.float()
        stats["source_sum"] += (s * m).sum(dim=(0, 2, 3), dtype=torch.float64)
        stats["source_sumsq"] += ((s * s) * m).sum(dim=(0, 2, 3), dtype=torch.float64)
        stats["reference_sum"] += (r * m).sum(dim=(0, 2, 3), dtype=torch.float64)
        stats["reference_sumsq"] += ((r * r) * m).sum(dim=(0, 2, 3), dtype=torch.float64)
        count = m.sum(dim=(0, 2, 3), dtype=torch.float64)
        stats["count"] += count

    @torch.no_grad()
    def calibrate_from_stats(
        self,
        stats: Dict[str, torch.Tensor],
        post_gelu_source: Optional[torch.Tensor] = None,
        post_gelu_reference: Optional[torch.Tensor] = None,
        post_gelu_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        count = stats["count"]
        if torch.any(count <= 0):
            bad = torch.nonzero(count <= 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Insufficient common calibration support in channels: {bad}")

        source_mean = stats["source_sum"] / count
        source_var = (stats["source_sumsq"] / count) - source_mean.square()
        source_std = source_var.clamp_min(CALIBRATION_EPS).sqrt()

        reference_mean = stats["reference_sum"] / count
        reference_var = (stats["reference_sumsq"] / count) - reference_mean.square()
        reference_std = reference_var.clamp_min(CALIBRATION_EPS).sqrt()

        gamma = reference_std / source_std.clamp_min(CALIBRATION_EPS)
        beta = reference_mean - gamma * source_mean
        if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all():
            raise RuntimeError("Calibration produced non-finite affine parameters")

        self.gamma.data.copy_(gamma.to(self.gamma.dtype))
        self.beta.data.copy_(beta.to(self.beta.dtype))
        self.source_mean.copy_(source_mean.to(self.source_mean.dtype))
        self.source_std.copy_(source_std.to(self.source_std.dtype))
        self.reference_mean.copy_(reference_mean.to(self.reference_mean.dtype))
        self.reference_std.copy_(reference_std.to(self.reference_std.dtype))
        self.calibration_done.fill_(True)

        pre_mean_gap = float((source_mean - reference_mean).abs().mean().cpu())
        pre_std_ratio = float((source_std / reference_std.clamp_min(CALIBRATION_EPS)).mean().cpu())

        calibrated_mean = reference_mean
        calibrated_std = reference_std
        post_mean_gap = float((calibrated_mean - reference_mean).abs().mean().cpu())
        post_std_ratio = float((calibrated_std / reference_std.clamp_min(CALIBRATION_EPS)).mean().cpu())

        report = {
            "samples_count_per_channel_min": int(count.min().cpu()),
            "samples_count_per_channel_max": int(count.max().cpu()),
            "pre_mean_abs_gap_pre_gelu": pre_mean_gap,
            "pre_std_ratio_mean_pre_gelu": pre_std_ratio,
            "post_mean_abs_gap_pre_gelu": post_mean_gap,
            "post_std_ratio_mean_pre_gelu": post_std_ratio,
            "gamma_mean": float(self.gamma.detach().float().mean().cpu()),
            "gamma_min": float(self.gamma.detach().float().min().cpu()),
            "gamma_max": float(self.gamma.detach().float().max().cpu()),
            "beta_mean_abs": float(self.beta.detach().float().abs().mean().cpu()),
        }

        if post_gelu_source is not None and post_gelu_reference is not None and post_gelu_mask is not None:
            src_post = post_gelu_source.float()
            ref_post = post_gelu_reference.float()
            mask_post = post_gelu_mask.float().expand_as(src_post)
            calibrated_post = F.gelu(self(src_post))
            src_mean2, src_std2 = self._masked_channel_stats(F.gelu(src_post), post_gelu_mask)
            cal_mean2, cal_std2 = self._masked_channel_stats(calibrated_post, post_gelu_mask)
            ref_mean2, ref_std2 = self._masked_channel_stats(ref_post, post_gelu_mask)
            report.update({
                "post_gelu_raw_mean_abs_gap": float((src_mean2 - ref_mean2).abs().mean().cpu()),
                "post_gelu_raw_std_ratio_mean": float((src_std2 / ref_std2.clamp_min(CALIBRATION_EPS)).mean().cpu()),
                "post_gelu_calibrated_mean_abs_gap": float((cal_mean2 - ref_mean2).abs().mean().cpu()),
                "post_gelu_calibrated_std_ratio_mean": float((cal_std2 / ref_std2.clamp_min(CALIBRATION_EPS)).mean().cpu()),
            })

        return report

    @torch.no_grad()
    def calibrate(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
        *,
        post_gelu_source: Optional[torch.Tensor] = None,
        post_gelu_reference: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        stats = self._init_stats(source.shape[1], source.device)
        self.accumulate_pair_stats(stats, source, source_mask, reference, reference_mask)
        common = source_mask.bool() & reference_mask.bool()
        return self.calibrate_from_stats(
            stats,
            post_gelu_source=post_gelu_source,
            post_gelu_reference=post_gelu_reference,
            post_gelu_mask=common,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)


# ============================================================================
# H1 / B7 model
# ============================================================================

class Model(nn.Module):
    model_id = MODEL_ID
    model_version = MODEL_VERSION
    model_contract = MODEL_CONTRACT

    def __init__(self, contract: Dict[str, Any]):
        super().__init__()

        self.contract = contract

        b6 = ensure_b6_host_module()
        self._b6_host = b6

        self.b6 = b6.Model(contract).cpu()

        checkpoint = discover_checkpoint(b6)
        print(f"[B6] validating checkpoint: {checkpoint}", flush=True)
        metadata = b6._validate_resume_checkpoint_metadata(
            checkpoint,
            self.b6,
        )
        print(
            "[B6] checkpoint metadata: "
            + json.dumps(metadata, indent=2, default=str),
            flush=True,
        )

        payload = _load_checkpoint_payload(checkpoint)
        if payload is None:
            raise RuntimeError(f"Could not load B6 checkpoint: {checkpoint}")
        _validate_exact_epoch10_checkpoint_payload(payload)
        self.b6.load_state_dict(payload["model_state_dict"], strict=True)
        self.b6.eval()

        for p in self.b6.parameters():
            p.requires_grad = False

        self.contract_hash = getattr(
            self.b6,
            "contract_hash",
            hashlib.sha256(
                json.dumps(contract, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )

        required_b6_attrs = (
            "highres_encoder",
            "encoder_activation",
            "pool",
            "convlstm1",
            "convlstm2",
            "decoder_projection",
            "decoder_activation",
            "head",
        )
        missing_b6_attrs = [
            name for name in required_b6_attrs if not hasattr(self.b6, name)
        ]
        if missing_b6_attrs:
            raise RuntimeError(
                "Loaded B6 model is missing expected attribute(s): "
                f"{missing_b6_attrs}."
            )

        partial_conv_cls = type(self.b6.highres_encoder)

        stem_out_channels = getattr(
            self.b6.highres_encoder, "out_channels", None
        )
        if (
            stem_out_channels is not None
            and int(stem_out_channels) != FUSION_OUTPUT_CHANNELS
        ):
            raise RuntimeError(
                "B6 highres_encoder.out_channels="
                f"{stem_out_channels} does not match H1's fusion output "
                f"width ({FUSION_OUTPUT_CHANNELS})."
            )

        self.branches = nn.ModuleDict({
            "sea_level": ModalityBranch(
                len(CHANNEL_GROUPS["sea_level"]),
                BRANCH_WIDTHS["sea_level"],
                partial_conv_cls,
            ),
            "thermal": ModalityBranch(
                len(CHANNEL_GROUPS["thermal"]),
                BRANCH_WIDTHS["thermal"],
                partial_conv_cls,
            ),
            "salinity": ModalityBranch(
                len(CHANNEL_GROUPS["salinity"]),
                BRANCH_WIDTHS["salinity"],
                partial_conv_cls,
            ),
            "currents": ModalityBranch(
                len(CHANNEL_GROUPS["currents"]),
                BRANCH_WIDTHS["currents"],
                partial_conv_cls,
            ),
            "wind": ModalityBranch(
                len(CHANNEL_GROUPS["wind"]),
                BRANCH_WIDTHS["wind"],
                partial_conv_cls,
            ),
            "static": ModalityBranch(
                len(CHANNEL_GROUPS["static"]),
                BRANCH_WIDTHS["static"],
                partial_conv_cls,
            ),
            "context": ModalityBranch(
                len(CHANNEL_GROUPS["context"]),
                BRANCH_WIDTHS["context"],
                partial_conv_cls,
            ),
        })

        self.fusion = partial_conv_cls(
            FUSION_INPUT_CHANNELS,
            FUSION_OUTPUT_CHANNELS,
            kernel_size=1,
            padding=0,
            padding_mode="replicate",
            bias=True,
        )

        self.fusion_norm = CalibratedAffine(FUSION_OUTPUT_CHANNELS)
        self.fusion_activation = nn.GELU()

        self._last_calibration_report: Optional[Dict[str, float]] = None

        for p in self.b6.parameters():
            p.requires_grad = False

    def _prepared_inputs(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        host = getattr(self, "_b6_host", None) or ensure_b6_host_module()
        seq, seq_mask, ocean_mask, _ = host.prepare_model_inputs(batch)
        return seq, seq_mask, ocean_mask

    def _branch_forward_sequence(
        self,
        seq: torch.Tensor,
        seq_mask: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, timesteps, _, h, w = seq.shape
        ocean_seq = ocean_mask.unsqueeze(1).expand(
            -1, timesteps, -1, -1, -1
        )

        outputs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        debug: Dict[str, torch.Tensor] = {}

        for name, indices in CHANNEL_GROUPS.items():
            x = seq[:, :, list(indices), :, :]
            m = seq_mask[:, :, list(indices), :, :]
            out_channels = BRANCH_WIDTHS[name]

            x_flat = x.reshape(bsz * timesteps, len(indices), h, w)
            m_flat = m.reshape(bsz * timesteps, len(indices), h, w)
            ocean_flat = ocean_seq.reshape(bsz * timesteps, 1, h, w)

            y, support = self.branches[name](
                x_flat,
                m_flat,
                ocean_flat,
            )

            y = y.reshape(bsz, timesteps, out_channels, h, w)
            support = support.reshape(bsz, timesteps, 1, h, w)

            outputs.append(y)
            masks.append(support)

            debug[name] = y
            debug[f"{name}_mask"] = support

        fused_input = torch.cat(outputs, dim=2)
        fused_mask = torch.cat(
            [
                m.expand(-1, -1, BRANCH_WIDTHS[name], -1, -1)
                for name, m in zip(CHANNEL_GROUPS.keys(), masks)
            ],
            dim=2,
        )

        any_support = (
            torch.stack(masks, dim=0).amax(dim=0)
            * ocean_seq
        )
        fused_mask = fused_mask * ocean_seq

        debug["fusion_input"] = fused_input
        debug["fusion_mask"] = fused_mask
        debug["fusion_any_support"] = any_support
        return fused_input, fused_mask, debug

    def _fusion(
        self,
        fusion_input: torch.Tensor,
        fusion_mask: torch.Tensor,
        ocean_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, timesteps, channels, h, w = fusion_input.shape

        x = fusion_input.reshape(bsz * timesteps, channels, h, w)
        m = fusion_mask.reshape(bsz * timesteps, channels, h, w)

        y, support = self.fusion(x, m)

        ocean_flat = ocean_seq.reshape(bsz * timesteps, 1, h, w)
        y = y * support * ocean_flat
        support = support * ocean_flat

        y = y.reshape(bsz, timesteps, FUSION_OUTPUT_CHANNELS, h, w)
        support = support.reshape(bsz, timesteps, 1, h, w)
        return y, support

    @torch.no_grad()
    def _calibrate_from_real_batch(
        self,
        seq: torch.Tensor,
        seq_mask: torch.Tensor,
        ocean_mask: torch.Tensor,
        fusion_input: torch.Tensor,
        fusion_mask: torch.Tensor,
    ) -> Dict[str, float]:
        bsz, timesteps, _, h, w = seq.shape
        fusion_flat = fusion_input.reshape(bsz * timesteps, FUSION_INPUT_CHANNELS, h, w)
        fusion_mask_flat = fusion_mask.reshape(bsz * timesteps, FUSION_INPUT_CHANNELS, h, w)
        x_flat = seq.reshape(bsz * timesteps, TOTAL_CHANNELS, h, w)
        mask_flat = seq_mask.reshape(bsz * timesteps, TOTAL_CHANNELS, h, w)
        ocean_flat = (
            ocean_mask.unsqueeze(1)
            .expand(-1, timesteps, -1, -1, -1)
            .reshape(bsz * timesteps, 1, h, w)
            .float()
        )

        ref_projection, ref_support = self.b6.highres_encoder(
            x_flat.float(), mask_flat.float()
        )
        ref_support = ref_support.float() * ocean_flat
        ref_post_gelu = self.b6.encoder_activation(ref_projection) * ref_support

        h1_fusion, h1_support = self.fusion(
            fusion_flat.float(), fusion_mask_flat.float()
        )
        h1_support = h1_support.float() * ocean_flat

        report = self.fusion_norm.calibrate(
            h1_fusion,
            h1_support,
            ref_projection.float(),
            ref_support,
            post_gelu_source=h1_fusion,
            post_gelu_reference=ref_post_gelu,
        )
        self._last_calibration_report = report
        print(
            "[CALIBRATION] single-batch pre-GELU: "
            f"mean_gap={report['pre_mean_abs_gap_pre_gelu']:.6g}, "
            f"std_ratio={report['pre_std_ratio_mean_pre_gelu']:.6g}",
            flush=True,
        )
        print(
            "[CALIBRATION] post-GELU calibrated gap: "
            f"mean={report.get('post_gelu_calibrated_mean_abs_gap', float('nan')):.6g}, "
            f"std_ratio={report.get('post_gelu_calibrated_std_ratio_mean', float('nan')):.6g}",
            flush=True,
        )
        return report

    @torch.no_grad()
    def calibrate_from_loader(
        self,
        loader,
        framework: ModuleType,
        device: torch.device,
        num_batches: int = CALIBRATION_BATCHES,
        warm_batches: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        was_training = self.training
        self.eval()

        stats = CalibratedAffine._init_stats(FUSION_OUTPUT_CHANNELS, device)
        collected = 0
        post_source_sum = torch.zeros(FUSION_OUTPUT_CHANNELS, dtype=torch.float64, device=device)
        post_source_sumsq = torch.zeros_like(post_source_sum)
        post_ref_sum = torch.zeros_like(post_source_sum)
        post_ref_sumsq = torch.zeros_like(post_source_sum)
        post_count = torch.zeros_like(post_source_sum)

        def consume(raw_batch: Dict[str, Any]) -> None:
            nonlocal collected
            seq, seq_mask, ocean_mask = self._prepared_inputs(raw_batch)
            fusion_input, fusion_mask, _ = self._branch_forward_sequence(seq, seq_mask, ocean_mask)
            bsz, timesteps, _, h, w = seq.shape

            fusion_flat = fusion_input.reshape(bsz * timesteps, FUSION_INPUT_CHANNELS, h, w)
            fusion_mask_flat = fusion_mask.reshape(bsz * timesteps, FUSION_INPUT_CHANNELS, h, w)
            x_flat = seq.reshape(bsz * timesteps, TOTAL_CHANNELS, h, w)
            mask_flat = seq_mask.reshape(bsz * timesteps, TOTAL_CHANNELS, h, w)
            ocean_flat = (
                ocean_mask.unsqueeze(1)
                .expand(-1, timesteps, -1, -1, -1)
                .reshape(bsz * timesteps, 1, h, w)
                .float()
            )

            ref_projection, ref_support = self.b6.highres_encoder(
                x_flat.float(), mask_flat.float()
            )
            ref_support = ref_support.float() * ocean_flat
            source_projection, source_support = self.fusion(
                fusion_flat.float(), fusion_mask_flat.float()
            )
            source_support = source_support.float() * ocean_flat

            CalibratedAffine.accumulate_pair_stats(
                stats,
                source_projection,
                source_support,
                ref_projection.float(),
                ref_support,
            )

            common = source_support.bool() & ref_support.bool()
            ref_post = self.b6.encoder_activation(ref_projection.float())
            calibrated_pre = self.fusion_norm(source_projection)
            calibrated_post = self.fusion_activation(calibrated_pre)
            ref_post = ref_post * ref_support
            m = common.float().expand_as(calibrated_post)
            cp = calibrated_post.float()
            rp = ref_post.float()
            post_source_sum.add_((cp * m).sum(dim=(0,2,3), dtype=torch.float64))
            post_source_sumsq.add_(((cp*cp) * m).sum(dim=(0,2,3), dtype=torch.float64))
            post_ref_sum.add_((rp * m).sum(dim=(0,2,3), dtype=torch.float64))
            post_ref_sumsq.add_(((rp*rp) * m).sum(dim=(0,2,3), dtype=torch.float64))
            post_count.add_(m.sum(dim=(0,2,3), dtype=torch.float64))
            collected += 1

        for raw_batch in list(warm_batches or [])[:num_batches]:
            consume(raw_batch)
        remaining = max(0, num_batches - collected)
        if remaining:
            it = iter(loader)
            for _ in range(remaining):
                try:
                    raw_batch = next(it)
                except StopIteration:
                    break
                consume(framework.move_batch_to_device(raw_batch, device))

        if collected == 0:
            raise RuntimeError("calibrate_from_loader: zero batches available")

        if torch.any(stats["count"] <= 0):
            bad = torch.nonzero(stats["count"] <= 0, as_tuple=False).flatten().tolist()
            raise RuntimeError(f"Calibration common support vanished in channels: {bad}")

        post_mean = post_source_sum / post_count.clamp_min(1.0)
        post_var = (post_source_sumsq / post_count.clamp_min(1.0)) - post_mean.square()
        post_std = post_var.clamp_min(CALIBRATION_EPS).sqrt()
        post_ref_mean = post_ref_sum / post_count.clamp_min(1.0)
        post_ref_var = (post_ref_sumsq / post_count.clamp_min(1.0)) - post_ref_mean.square()
        post_ref_std = post_ref_var.clamp_min(CALIBRATION_EPS).sqrt()

        report = self.fusion_norm.calibrate_from_stats(stats)
        report.update({
            "calibration_batches": int(collected),
            "post_gelu_raw_mean_abs_gap": float((post_mean - post_ref_mean).abs().mean().cpu()),
            "post_gelu_raw_std_ratio_mean": float((post_std / post_ref_std.clamp_min(CALIBRATION_EPS)).mean().cpu()),
            "post_gelu_reference_mean_abs": float(post_ref_mean.abs().mean().cpu()),
            "post_gelu_reference_std_mean": float(post_ref_std.mean().cpu()),
        })
        self._last_calibration_report = report

        print(
            f"[CALIBRATION] pooled over {collected} real TRAIN batch(es); "
            f"pre-GELU mean_gap={report['pre_mean_abs_gap_pre_gelu']:.6g}, "
            f"std_ratio={report['pre_std_ratio_mean_pre_gelu']:.6g}; "
            f"post-GELU (raw, pre-calibration) mean_gap={report['post_gelu_raw_mean_abs_gap']:.6g}, "
            f"std_ratio={report['post_gelu_raw_std_ratio_mean']:.6g}",
            flush=True,
        )
        self.train(was_training)
        return report

    def _forward_features(
        self,
        batch: Dict[str, Any],
        return_debug: bool = False,
    ):
        seq, seq_mask, ocean_mask = self._prepared_inputs(batch)

        bsz, timesteps, _, h, w = seq.shape

        ocean_seq = ocean_mask.unsqueeze(1).expand(
            -1, timesteps, -1, -1, -1
        )

        fusion_input, fusion_mask, branch_debug = self._branch_forward_sequence(
            seq,
            seq_mask,
            ocean_mask,
        )

        if not bool(self.fusion_norm.calibration_done.item()):
            raise RuntimeError(
                "H1 fusion affine is not calibrated. Run calibrate_from_loader() "
                "before any H1 training/evaluation forward pass."
            )

        fused_highres_seq, fused_highres_mask_seq = self._fusion(
            fusion_input,
            fusion_mask,
            ocean_seq,
        )

        fused_highres_seq = self.fusion_activation(
            self.fusion_norm(
                fused_highres_seq.reshape(
                    bsz * timesteps,
                    FUSION_OUTPUT_CHANNELS,
                    h,
                    w,
                )
            )
        ).reshape(
            bsz,
            timesteps,
            FUSION_OUTPUT_CHANNELS,
            h,
            w,
        )
        fused_highres_seq = (
            fused_highres_seq
            * fused_highres_mask_seq
            * ocean_seq
        )

        lowres_ocean_mask = F.max_pool2d(
            ocean_mask,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        lowres_ocean_mask = (lowres_ocean_mask > 0).to(
            dtype=fused_highres_seq.dtype
        )

        h1 = c1 = None
        h2 = c2 = None
        final_highres = None
        final_highres_mask = None

        for t in range(timesteps):
            x_t = fused_highres_seq[:, t]
            mask_t = fused_highres_mask_seq[:, t].to(
                dtype=fused_highres_seq.dtype
            )

            pooled, pooled_mask = self.b6.pool(
                x_t,
                mask_t,
            )
            pooled = pooled * lowres_ocean_mask
            pooled_mask = pooled_mask * lowres_ocean_mask

            h1, c1, state_mask1 = self.b6.convlstm1(
                pooled,
                pooled_mask,
                h1,
                c1,
                lowres_ocean_mask,
            )

            h2, c2, state_mask2 = self.b6.convlstm2(
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

            if t == timesteps - 1:
                final_highres = x_t
                final_highres_mask = mask_t

        if h2 is None or final_highres is None or final_highres_mask is None:
            raise RuntimeError("H1 recurrent sequence produced no final state.")

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
        ) * ocean_mask
        recurrent_up = recurrent_up * recurrent_mask_hr

        fused = torch.cat(
            [recurrent_up, final_highres],
            dim=1,
        )
        fused_mask = torch.cat(
            [
                recurrent_mask_hr.expand(
                    -1,
                    32,
                    -1,
                    -1,
                ),
                final_highres_mask.expand(
                    -1,
                    FUSION_OUTPUT_CHANNELS,
                    -1,
                    -1,
                ),
            ],
            dim=1,
        )

        if return_debug:
            branch_debug["fusion_output"] = fused_highres_seq
            branch_debug["fusion_output_mask"] = fused_highres_mask_seq
            branch_debug["ocean_mask"] = ocean_mask
            branch_debug["lowres_ocean_mask"] = lowres_ocean_mask
            branch_debug["recurrent_up"] = recurrent_up
            return fused, fused_mask, ocean_mask, branch_debug

        return fused, fused_mask, ocean_mask

    def train(self, mode: bool = True):
        super().train(mode)
        self.b6.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        fused, fused_mask, ocean_mask = self._forward_features(
            batch,
            return_debug=False,
        )

        x, mask = self.b6.decoder_projection(
            fused,
            fused_mask,
        )
        x = self.b6.decoder_activation(x)
        x = x * mask * ocean_mask

        y, output_mask = self.b6.head(
            x,
            mask,
        )
        y = y * output_mask * ocean_mask

        expected = (fused.shape[0], N_DEPTH, GRID_H, GRID_W)
        if tuple(y.shape) != expected:
            raise RuntimeError(
                f"H1 output shape {tuple(y.shape)} != {expected}"
            )
        if not torch.isfinite(y).all():
            raise FloatingPointError(
                "H1 produced non-finite predictions."
            )
        return y

    def model_summary(self) -> Dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_contract": MODEL_CONTRACT,
            "reference_model": "CONVLSTM_MASKED_B6",
            "reference_rmse_c_weighted": REFERENCE_B6_RMSE_WEIGHTED,
            "architecture": {
                "input_channels": 36,
                "sequence_days": 14,
                "modality_branches": {
                    name: {
                        "input_channels": len(CHANNEL_GROUPS[name]),
                        "output_channels": BRANCH_WIDTHS[name],
                        "channels": list(CHANNEL_GROUPS[name]),
                    }
                    for name in CHANNEL_GROUPS
                },
                "branch_total_channels": FUSION_INPUT_CHANNELS,
                "fusion": "masked 1x1 56->32",
                "fusion_scale_interface": (
                    "trainable affine calibrated from real-batch "
                    "H1-fusion vs frozen-B6-stem statistics"
                ),
                "attention": False,
                "gating": False,
                "imputation": False,
                "frozen_backbone": (
                    "B6 ConvLSTM1 + ConvLSTM2 + decoder projection + head"
                ),
                "output_shape": ["B", 15, 100, 240],
            },
            "parameters": parameter_summary(self),
            "calibration": self._last_calibration_report,
        }


B6_HOST_MODULE: Optional[ModuleType] = None


def ensure_b6_host_module() -> ModuleType:
    global B6_HOST_MODULE
    if B6_HOST_MODULE is not None:
        return B6_HOST_MODULE

    for name, module in list(sys.modules.items()):
        if name.startswith("oceanembed_b6_host_") and hasattr(module, "Model"):
            B6_HOST_MODULE = module
            return B6_HOST_MODULE

    B6_HOST_MODULE = import_module_from_path(
        find_b6_trainer(),
        "oceanembed_b6_host_model_import",
    )
    return B6_HOST_MODULE


# ============================================================================
# Preflight diagnostics
# ============================================================================

@torch.no_grad()
def run_preflight_probe(
    model: Model,
    batch: Dict[str, Any],
    framework: ModuleType,
    device: torch.device,
) -> Dict[str, Any]:
    batch = framework.move_batch_to_device(batch, device)
    model.eval()

    seq, seq_mask, ocean_mask = model._prepared_inputs(batch)
    fusion_input, fusion_mask, debug = model._branch_forward_sequence(
        seq,
        seq_mask,
        ocean_mask,
    )

    bsz, t, _, h, w = seq.shape
    x_flat = seq.reshape(bsz * t, TOTAL_CHANNELS, h, w)
    m_flat = seq_mask.reshape(bsz * t, TOTAL_CHANNELS, h, w)
    ocean_flat = (
        ocean_mask.unsqueeze(1)
        .expand(-1, t, -1, -1, -1)
        .reshape(bsz * t, 1, h, w)
    )

    ref_proj, ref_support = model.b6.highres_encoder(
        x_flat,
        m_flat.float(),
    )
    ref_stem = model.b6.encoder_activation(ref_proj)
    ref_stem = ref_stem * ref_support * ocean_flat

    fusion_flat = fusion_input.reshape(
        bsz * t,
        FUSION_INPUT_CHANNELS,
        h,
        w,
    )
    fusion_mask_flat = fusion_mask.reshape(
        bsz * t,
        FUSION_INPUT_CHANNELS,
        h,
        w,
    )
    raw_fusion, raw_support = model.fusion(
        fusion_flat,
        fusion_mask_flat.float(),
    )
    raw_fusion = raw_fusion * raw_support * ocean_flat

    src_mean, src_std = CalibratedAffine._masked_channel_stats(
        raw_fusion,
        raw_support * ocean_flat,
    )
    ref_mean, ref_std = CalibratedAffine._masked_channel_stats(
        ref_stem,
        ref_support * ocean_flat,
    )

    preflight = {
        "reference_stem_mean_abs": float(ref_mean.abs().mean().cpu()),
        "reference_stem_std_mean": float(ref_std.mean().cpu()),
        "raw_h1_fusion_mean_abs": float(src_mean.abs().mean().cpu()),
        "raw_h1_fusion_std_mean": float(src_std.mean().cpu()),
        "raw_mean_abs_gap": float(
            (src_mean - ref_mean).abs().mean().cpu()
        ),
        "raw_std_ratio_mean": float(
            (src_std / ref_std.clamp_min(CALIBRATION_EPS)).mean().cpu()
        ),
    }

    mixed_found = False
    mixed_info: Dict[str, Any] = {}

    ocean_bool = ocean_mask[0, 0].bool()
    candidate_cells_gpu = torch.nonzero(ocean_bool, as_tuple=False)
    scan_limit = min(5000, candidate_cells_gpu.shape[0])
    scan_cells = candidate_cells_gpu[:scan_limit]

    if scan_cells.numel() > 0:
        day = 0
        py_idx = scan_cells[:, 0]
        px_idx = scan_cells[:, 1]

        branch_names = list(CHANNEL_GROUPS.keys())
        per_branch_valid = torch.stack(
            [
                seq_mask[0, day, list(indices), :, :][:, py_idx, px_idx].any(
                    dim=0
                )
                for indices in CHANNEL_GROUPS.values()
            ],
            dim=0,
        )

        any_valid = per_branch_valid.any(dim=0)
        all_valid = per_branch_valid.all(dim=0)
        mixed_mask = any_valid & ~all_valid

        mixed_indices = torch.nonzero(mixed_mask, as_tuple=False)
        if mixed_indices.numel() > 0:
            first = int(mixed_indices[0].item())
            py, px = int(py_idx[first].item()), int(px_idx[first].item())
            branch_valid = {
                name: bool(per_branch_valid[i, first].item())
                for i, name in enumerate(branch_names)
            }
            mixed_found = True
            mixed_info = {
                "type": "real_mixed_validity",
                "day": day,
                "y": py,
                "x": px,
                "branch_valid": branch_valid,
            }

    if not mixed_found and candidate_cells_gpu.numel() > 0:
        candidate_cells_cpu = candidate_cells_gpu.cpu()

        margin = 5
        interior = (
            (candidate_cells_cpu[:, 0] >= margin)
            & (candidate_cells_cpu[:, 0] <= GRID_H - 1 - margin)
            & (candidate_cells_cpu[:, 1] >= margin)
            & (candidate_cells_cpu[:, 1] <= GRID_W - 1 - margin)
        )
        interior_cells = candidate_cells_cpu[interior]
        if interior_cells.numel() == 0:
            raise RuntimeError(
                "No interior ocean cell is available for the mixed-validity probe."
            )

        thermal_indices = list(CHANNEL_GROUPS["thermal"])
        iy = interior_cells[:, 0].to(device=seq_mask.device)
        ix = interior_cells[:, 1].to(device=seq_mask.device)
        thermal_ok = seq_mask[0, 0, thermal_indices, :, :][:, iy, ix].any(dim=0)
        thermal_ok_cpu = thermal_ok.detach().cpu()
        valid_interior = interior_cells[thermal_ok_cpu]
        if valid_interior.numel() == 0:
            raise RuntimeError(
                "No interior ocean cell with valid thermal observations was "
                "available for the controlled mixed-validity probe."
            )
        chosen = valid_interior[0]
        py, px = int(chosen[0].item()), int(chosen[1].item())
        day = 0

        probe_batch = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        if "x_dynamic_valid" not in probe_batch:
            raise RuntimeError(
                "Mixed-validity preflight probe requires 'x_dynamic_valid' "
                "in the batch to construct a controlled invalidation, but it "
                "is missing."
            )

        half = 5
        y0, y1 = max(0, py - half), min(GRID_H, py + half + 1)
        x0, x1 = max(0, px - half), min(GRID_W, px + half + 1)

        probe_batch["x_dynamic_valid"][
            0,
            day,
            list(CHANNEL_GROUPS["salinity"]),
            y0:y1,
            x0:x1,
        ] = False
        probe_batch["x_dynamic"][
            0,
            day,
            list(CHANNEL_GROUPS["salinity"]),
            y0:y1,
            x0:x1,
        ] = 0.0

        probe_seq, probe_mask, probe_ocean = model._prepared_inputs(
            probe_batch
        )
        _, _, probe_debug = model._branch_forward_sequence(
            probe_seq,
            probe_mask,
            probe_ocean,
        )

        salinity_pixel = probe_debug["salinity"][
            0,
            day,
            :,
            py,
            px,
        ]
        thermal_pixel = probe_debug["thermal"][
            0,
            day,
            :,
            py,
            px,
        ]

        salinity_zero = bool(
            torch.allclose(
                salinity_pixel,
                torch.zeros_like(salinity_pixel),
                atol=0.0,
                rtol=0.0,
            )
        )

        if not torch.isfinite(salinity_pixel).all():
            raise RuntimeError(
                "Mixed-validity probe produced non-finite salinity output."
            )

        mixed_found = True
        mixed_info = {
            "type": "controlled_real_pixel_mixed_validity",
            "day": day,
            "y": py,
            "x": px,
            "fully_invalid_branch": "salinity",
            "fully_invalid_branch_exact_zero": salinity_zero,
            "other_branch_finite": bool(
                torch.isfinite(thermal_pixel).all().item()
            ),
        }

        if not salinity_zero:
            raise RuntimeError(
                "Fully invalid salinity branch did not collapse to exact zero."
            )

    if not mixed_found:
        raise RuntimeError(
            "Could not establish a mixed-validity probe on the real batch."
        )

    land_cells = torch.nonzero(~ocean_bool, as_tuple=False)
    if land_cells.numel() > 0:
        ly, lx = [int(v) for v in land_cells[0].tolist()]
        land_nonzero = False
        for name in CHANNEL_GROUPS:
            val = debug[name][0, 0, :, ly, lx]
            if torch.any(val != 0):
                land_nonzero = True
                break
        if land_nonzero:
            raise RuntimeError(
                "Branch activation leaked into a structural land cell."
            )
        preflight["land_branch_activation_zero"] = True
    else:
        preflight["land_branch_activation_zero"] = None

    preflight["mixed_validity_probe"] = mixed_info
    preflight["branch_outputs_finite"] = all(
        bool(torch.isfinite(debug[name]).all().item())
        for name in CHANNEL_GROUPS
    )

    if not preflight["branch_outputs_finite"]:
        raise RuntimeError("One or more H1 branch outputs are non-finite.")

    if not bool(model.fusion_norm.calibration_done.item()):
        model._calibrate_from_real_batch(
            seq,
            seq_mask,
            ocean_mask,
            fusion_input,
            fusion_mask,
        )

    prediction = model(batch)
    preflight["prediction_shape"] = list(prediction.shape)
    preflight["prediction_finite"] = bool(
        torch.isfinite(prediction).all().item()
    )

    if tuple(prediction.shape) != (
        int(batch["y"].shape[0]),
        N_DEPTH,
        GRID_H,
        GRID_W,
    ):
        raise RuntimeError(
            f"H1 preflight prediction shape is {tuple(prediction.shape)}."
        )

    if not preflight["prediction_finite"]:
        raise RuntimeError("H1 preflight prediction contains non-finite values.")

    return preflight


# ============================================================================
# Gradient / optimization preflight
# ============================================================================


def run_gradient_sanity_probe(
    model: Model,
    batch: Dict[str, Any],
    framework: ModuleType,
) -> Dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    pred = framework.forward_prediction(model, batch)
    loss = framework.masked_mse_loss(pred, batch["y"], batch["loss_mask"])
    if not torch.isfinite(loss):
        raise RuntimeError("Gradient preflight loss is non-finite")
    loss.backward()

    trainable_params = []
    finite_grad_params = 0
    nonzero_grad_params = 0
    max_grad = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        trainable_params.append(name)
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if not torch.isfinite(g).all():
            raise RuntimeError(f"Non-finite H1 gradient in {name}")
        finite_grad_params += 1
        if bool(torch.any(g != 0).item()):
            nonzero_grad_params += 1
        max_grad = max(max_grad, float(g.abs().max().cpu()))

    frozen_with_grad = [
        name for name, p in model.b6.named_parameters()
        if p.grad is not None
    ]
    model.zero_grad(set_to_none=True)
    model.train()

    if not trainable_params:
        raise RuntimeError("H1 has no trainable parameters")
    if finite_grad_params == 0 or nonzero_grad_params == 0:
        raise RuntimeError("No finite/nonzero gradient reached the H1 trainable front end")
    if frozen_with_grad:
        raise RuntimeError(
            "Frozen B6 parameters unexpectedly received gradients: "
            + ", ".join(frozen_with_grad[:10])
        )

    return {
        "loss": float(loss.detach().float().cpu()),
        "trainable_parameter_count": len(trainable_params),
        "parameters_with_finite_grad": finite_grad_params,
        "parameters_with_nonzero_grad": nonzero_grad_params,
        "max_abs_grad": max_grad,
        "frozen_b6_parameters_with_grad": 0,
    }


# ============================================================================
# Framework integration
# ============================================================================

def build_framework_config(
    framework: ModuleType,
    runtime: Dict[str, Path],
    dataset_root: Path,
) -> Any:
    cfg = framework.TrainConfig(
        dataset_root=str(dataset_root.resolve()),
        backbone_path=str(runtime["backbone_path"]),
        loader_path=str(runtime["loader_path"]),
        model_module=str(Path(__file__).resolve()),
        model_class="Model",
        run_root=str(Path(RUN_ROOT).resolve()),
        experiment_name=MODEL_ID,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        patience=PATIENCE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=AMP_ENABLED,
        seed=SEED,
        log_every=LOG_EVERY,
        checkpoint_every=1,
        checkpoint_every_steps=500,
        resume=None,
    )
    cfg.validate()
    print(
        "[CONFIG] TrainingFramework V1.1 scheduler = ReduceLROnPlateau "
        "(factor=0.5, patience=2, min_lr=1e-7); "
        f"initial LR={LEARNING_RATE:.3g}",
        flush=True,
    )
    return cfg


def _read_best_validation_rmse(run_dir: Path) -> Tuple[float, Dict[str, Any]]:
    path = run_dir / "best_validation_metrics.json"
    if not path.is_file():
        raise RuntimeError(f"Framework did not write {path.name}; cannot determine best H1 RMSE.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        value = float(payload["overall"][REFERENCE_B6_METRIC_NAME])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"best_validation_metrics.json lacks numeric {REFERENCE_B6_METRIC_NAME}"
        ) from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Best H1 {REFERENCE_B6_METRIC_NAME} is non-finite")
    return value, payload


def _maybe_warm_start_h1_weights(model: nn.Module, path_str: str) -> None:
    """Best-effort load of a prior H1 model_state_dict into `model`.

    Weights only. Never raises: any failure is logged and the run continues
    with the freshly-initialized model, so a bad/missing/incompatible
    checkpoint path can never abort an otherwise-good run.
    """
    if not path_str:
        print("[WARM-START] RESUME_FROM_H1_CHECKPOINT not set; starting fresh.", flush=True)
        return

    p = Path(path_str)
    if not p.is_file():
        print(f"[WARM-START] WARNING: checkpoint not found at {p}; starting fresh.", flush=True)
        return

    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            print(f"[WARM-START] WARNING: no model_state_dict in {p}; starting fresh.", flush=True)
            return
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        prev_epoch = payload.get("epoch", "?")
        prev_best = payload.get("best_val_loss", "?")
        print(
            f"[WARM-START] Loaded H1 weights from {p} "
            f"(previous epoch={prev_epoch}, previous best_val_loss={prev_best}). "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}. "
            "Optimizer/scheduler/patience state NOT restored (fresh from this point).",
            flush=True,
        )
        if missing or unexpected:
            print(
                f"[WARM-START] NOTE: missing={list(missing)[:10]} "
                f"unexpected={list(unexpected)[:10]}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - best-effort by design
        print(f"[WARM-START] WARNING: failed to load {p} ({exc!r}); starting fresh.", flush=True)


def run_training(
    framework: ModuleType,
    runtime: Dict[str, Path],
    dataset_root: Path,
) -> Dict[str, Any]:
    reference_rmse = REFERENCE_B6_RMSE_WEIGHTED
    reference_convention = REFERENCE_B6_METRIC_NAME

    cfg = build_framework_config(framework, runtime, dataset_root)
    adapter = framework.BackboneAdapter(cfg)

    if hasattr(B6_HOST_MODULE, "validate_backbone_runtime"):
        B6_HOST_MODULE.validate_backbone_runtime(adapter)
    if hasattr(B6_HOST_MODULE, "validate_runtime_normalization"):
        norm_report = B6_HOST_MODULE.validate_runtime_normalization(adapter)
    else:
        norm_report = None

    # Training mode intentionally skips the diagnostic preflight suite.
    # The H1/B7 self-test was already completed before training. Do not gate
    # training on a /kaggle/working marker: Kaggle sessions are ephemeral and
    # a new session cannot be expected to retain artifacts from the test run.
    #
    # The exact B6 epoch-10 checkpoint/fingerprint is still validated during
    # normal runtime/model construction, so this removes only duplicate
    # diagnostics—not the scientific checkpoint contract.
    print(
        "[PREFLIGHT] SKIPPED — training mode uses the already-validated "
        "H1/B7 self-test contract; no duplicate diagnostic probes.",
        flush=True,
    )

    preflight = {
        "status": "SKIPPED_IN_TRAIN_MODE",
        "reason": "self-test was completed before training; diagnostic preflight is not rerun",
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "reference_epoch": REFERENCE_B6_EPOCH,
    }
    gradient_probe = {
        "status": "SKIPPED_IN_TRAIN_MODE",
        "reason": "gradient preflight is not rerun in training mode",
    }

    trainer = framework.Trainer(cfg, adapter)

    if getattr(trainer.model, "model_id", None) != MODEL_ID:
        raise RuntimeError(
            "Framework instantiated an unexpected H1 model identity: "
            f"{getattr(trainer.model, 'model_id', None)!r}"
        )

    trainable_b6 = [
        name for name, p in trainer.model.b6.named_parameters()
        if p.requires_grad
    ]
    if trainable_b6:
        raise RuntimeError(
            "B6 backbone is not frozen in the training model: "
            + ", ".join(trainable_b6[:10])
        )

    _maybe_warm_start_h1_weights(trainer.model, RESUME_FROM_H1_CHECKPOINT)

    trainer.model.calibrate_from_loader(
        adapter.train_loader,
        framework,
        trainer.device,
        num_batches=CALIBRATION_BATCHES,
        warm_batches=None,
    )
    if not bool(trainer.model.fusion_norm.calibration_done.item()):
        raise RuntimeError("Training H1 fusion affine is not calibrated")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = trainer.fit()
    result = dict(result) if isinstance(result, dict) else {"raw_result": result}

    run_dir = Path(trainer.run_dir)
    best_val_rmse, best_val_metrics = _read_best_validation_rmse(run_dir)

    improvement_c = float(reference_rmse - best_val_rmse)
    verdict = (
        "SUPPORTED"
        if improvement_c >= IMPROVEMENT_THRESHOLD_C
        else "NOT_SUPPORTED"
    )

    result.update({
        "h1_verdict": verdict,
        "h1_reference_metric": reference_convention,
        "h1_reference_rmse_c": float(reference_rmse),
        "h1_best_checkpoint_rmse_c": float(best_val_rmse),
        "h1_improvement_c": improvement_c,
        "h1_required_improvement_c": float(IMPROVEMENT_THRESHOLD_C),
        "h1_best_checkpoint": str(run_dir / "best_model.pt"),
    })

    if verdict == "SUPPORTED":
        print(
            f"[DECISION] H1 supported under pre-registered threshold: "
            f"{best_val_rmse:.9f} C vs {reference_rmse:.9f} C "
            f"(improvement {improvement_c:.6f} C).",
            flush=True,
        )
    else:
        print(
            f"[DECISION] H1 not supported under pre-registered threshold: "
            f"{best_val_rmse:.9f} C vs {reference_rmse:.9f} C "
            f"(improvement {improvement_c:.6f} C).",
            flush=True,
        )

    spec = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "scientific_role": "modality_specific_front_end_hypothesis_test",
        "reference": {
            "model_id": "CONVLSTM_MASKED_B6",
            "epoch": REFERENCE_B6_EPOCH,
            "val_loss": REFERENCE_B6_VAL_LOSS,
            "metric": REFERENCE_B6_METRIC_NAME,
            "rmse_c": reference_rmse,
            "comparison_policy": "exact same canonical weighted validation metric",
            "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        },
        "decision": {
            "best_checkpoint_rmse_c": best_val_rmse,
            "improvement_c": improvement_c,
            "threshold_c": IMPROVEMENT_THRESHOLD_C,
            "verdict": verdict,
        },
        "architecture": {
            "branch_groups": CHANNEL_GROUPS,
            "branch_widths": BRANCH_WIDTHS,
            "fusion_channels": [FUSION_INPUT_CHANNELS, FUSION_OUTPUT_CHANNELS],
            "calibration": (
                "trainable affine BEFORE H1 GELU, initialized from streamed "
                "common-support statistics matching H1 pre-GELU fusion to "
                "the frozen B6 pre-GELU high-resolution projection"
            ),
            "calibration_batches": CALIBRATION_BATCHES,
            "attention": False,
            "gating": False,
            "imputation": False,
            "frozen_backbone": [
                "B6 ConvLSTM layer 1",
                "B6 ConvLSTM layer 2",
                "B6 decoder projection",
                "B6 output head",
            ],
        },
        "training": {
            "epochs_max": EPOCHS,
            "batch_size": BATCH_SIZE,
            "initial_learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "scheduler": "ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7)",
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "amp": AMP_ENABLED,
            "seed": SEED,
        },
        "channel_contract": {
            "total": TOTAL_CHANNELS,
            "window_days": WINDOW_DAYS,
            "grid": [GRID_H, GRID_W],
            "depths": N_DEPTH,
        },
        "preflight": preflight,
        "gradient_preflight": gradient_probe,
        "calibration": trainer.model._last_calibration_report,
        "normalization_guard": norm_report,
        "parameters": parameter_summary(trainer.model),
        "result": result,
        "best_validation_metrics": best_val_metrics,
    }
    atomic_json(run_dir / "h1_pam_convlstm_b7_experiment_spec.json", spec)

    return {
        "run_dir": str(run_dir),
        "result": result,
        "preflight": preflight,
        "gradient_preflight": gradient_probe,
        "calibration": trainer.model._last_calibration_report,
        "parameters": parameter_summary(trainer.model),
    }


# ============================================================================
# Self-test
# ============================================================================

def run_self_test() -> Dict[str, Any]:
    set_seed(SEED)
    b6, runtime = load_b6_runtime()
    global B6_HOST_MODULE
    B6_HOST_MODULE = b6

    dataset_root = resolve_dataset_root(b6)
    framework = import_module_from_path(
        runtime["framework_path"],
        "oceanembed_training_framework_h1",
    )

    adapter = framework.BackboneAdapter(
        framework.TrainConfig(
            dataset_root=str(dataset_root),
            backbone_path=str(runtime["backbone_path"]),
            loader_path=str(runtime["loader_path"]),
            model_module=str(Path(__file__).resolve()),
            model_class="Model",
            run_root=str(Path(RUN_ROOT).resolve()),
            experiment_name=MODEL_ID,
            epochs=1,
            batch_size=1,
            num_workers=0,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            patience=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            amp=False,
            seed=SEED,
            log_every=1,
            checkpoint_every=1,
            checkpoint_every_steps=500,
            resume=None,
        )
    )

    model = Model(adapter.contract)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    batch = framework.move_batch_to_device(
        next(iter(adapter.train_loader)),
        device,
    )

    model.calibrate_from_loader(
        adapter.train_loader,
        framework,
        device,
        num_batches=2,
        warm_batches=[batch],
    )

    report = run_preflight_probe(
        model,
        batch,
        framework,
        device,
    )

    report["parameter_summary"] = parameter_summary(model)
    report["model_id"] = MODEL_ID
    report["status"] = "PASS"

    checkpoint = discover_checkpoint(b6)
    _write_preflight_pass_marker(checkpoint, dataset_root)

    return report


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    set_seed(SEED)

    print("=" * 80)
    print("OceanEmbed H1 — B7 PAM-ConvLSTM")
    print("Physics-Aware Multi-Modal Masked ConvLSTM")
    print("=" * 80)
    print(
        f"Reference B6 {REFERENCE_B6_METRIC_NAME}: "
        f"{REFERENCE_B6_RMSE_WEIGHTED:.9f} C (epoch {REFERENCE_B6_EPOCH})"
    )
    print(f"Reference B6 epoch: 10")
    print(f"Bundle fingerprint: {EXPECTED_BUNDLE_FP}")
    print("=" * 80)

    global B6_HOST_MODULE
    b6, runtime = load_b6_runtime()
    B6_HOST_MODULE = b6

    dataset_root = resolve_dataset_root(b6)
    print(f"[DATA] dataset root resolved for training: {dataset_root}", flush=True)

    checkpoint = discover_checkpoint(b6)
    print(f"[B6] selected exact epoch-{REFERENCE_B6_EPOCH} checkpoint: {checkpoint}", flush=True)
    payload = _load_checkpoint_payload(checkpoint)
    if payload is None:
        raise RuntimeError(f"Could not load selected B6 checkpoint: {checkpoint}")
    _validate_exact_epoch10_checkpoint_payload(payload)

    framework = import_module_from_path(
        runtime["framework_path"],
        "oceanembed_training_framework_h1",
    )

    if RUN_MODE == "self-test":
        report = run_self_test()
        print(json.dumps(report, indent=2, default=str))
        return 0

    if RUN_MODE != "train":
        raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}")

    result = run_training(
        framework,
        runtime,
        dataset_root,
    )

    print("=" * 80)
    print("H1 / B7 TRAINING COMPLETE")
    print(json.dumps(result, indent=2, default=str))
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())