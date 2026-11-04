"""
OceanEmbed H1 — PAM-ConvLSTM / B7
Physics-Aware Multi-Modal Masked ConvLSTM

Scientific experiment:
    H1 / B7
    PAM_CONVLSTM_B7
    "B7 — Physics-Aware Multi-Modal Masked ConvLSTM"

Purpose
-------
Test whether modality-specific masked encoders improve the frozen B6
ConvLSTM backbone under a deliberately restricted architectural change.

Frozen reference:
    B6 CONVLSTM_MASKED_B6
    epoch-10 latest_model.pt
    validation RMSE reference = 0.706064451 C
    bundle fingerprint =
      ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f

H1 architecture
---------------
36 input channels are split into:
    Sea-level : [0, 1]
    Thermal   : [4, 5, 6, 8]
    Salinity  : [9, 10, 11, 15]
    Currents  : [2, 3, 7, 12, 13, 14, 16, 19, 20, 22, 23]
    Wind      : [17, 18, 21]
    Static    : [24, 25, 26, 27, 28, 29]
    Context   : [30, 31, 32, 33, 34, 35]

Each branch has its own validity mask. Structural ocean masking is applied
inside every branch layer. There is no imputation.

Branch outputs:
    Sea-level  8
    Thermal    8
    Salinity   8
    Currents   12
    Wind       8
    Static     8
    Context    4
Total branch channels = 56

Fusion:
    masked 1x1, 56 -> 32

Scale interface:
    A trainable affine normalization is calibrated on a real batch so that
    the H1 fusion output matches the mean/std regime of the ORIGINAL frozen
    B6 high-resolution stem output immediately before B6 pooling.

Then:
    GELU -> B6 mask-aware 2x2 pooling
    -> frozen B6 ConvLSTM layer 1
    -> frozen B6 ConvLSTM layer 2
    -> frozen B6 decoder projection
    -> frozen B6 head
    -> 15 depths

No attention/gating is used in H1-min.

Runtime
-------
The script reuses the exact B6 trainer's embedded runtime sources. The B6
trainer .py must therefore be attached to the Kaggle notebook as an input
file/dataset, along with the B6 epoch-10 latest_model.pt.

The script searches recursively under /kaggle/input for:
    OceanEmbed_KaggleTrainer_B6_ConvLSTM_15Epoch_LatestOnlyResume_BestRestore*.py
and a compatible latest_model.pt.

If multiple compatible checkpoints exist, the script validates their B6
metadata and uses the first compatible candidate unless B6_CHECKPOINT_PATH
is explicitly set.

IMPORTANT:
    This script is intentionally an H1 experiment trainer, not the final
    proposed OceanEmbed model.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
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

# Exact B6 comparison reference. This is the A2.4 *weighted* RMSE — the name
# says so explicitly (Option B) so nobody accidentally treats it as pooled.
# If real per-day SSE/N arrays are available, set REFERENCE_B6_DAILY_SSE_PATH
# / REFERENCE_B6_DAILY_N_PATH (or REFERENCE_B6_DAILY_REPORT_PATH for a single
# JSON containing both) below and resolve_b6_reference() will prefer the
# pooled value computed from them (Option A) instead.
REFERENCE_B6_RMSE_WEIGHTED = 0.706064451
REFERENCE_B6_VAL_LOSS = 0.9249299274
EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)

# Set these if the A2.4 DAILY["Full"] per-day SSE/N arrays are available on
# disk (e.g. inside an A2_4_final_report.json, or two separate files). If
# left unset and auto-discovery below finds nothing, the run falls back to
# REFERENCE_B6_RMSE_WEIGHTED and says so loudly in the training result.
REFERENCE_B6_DAILY_REPORT_PATH: Optional[str] = None  # single JSON, see resolver
REFERENCE_B6_DAILY_SSE_PATH: Optional[str] = None      # JSON/CSV of per-day SSE
REFERENCE_B6_DAILY_N_PATH: Optional[str] = None        # JSON/CSV of per-day N

EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-3          # peak LR for cosine decay (fix 5a)
LEARNING_RATE_MIN = 1e-5      # trough LR for cosine decay (fix 5a)
LR_SCHEDULE = "cosine"        # applied if the framework's TrainConfig supports it
WEIGHT_DECAY = 1e-4
PATIENCE = 4
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
SEED = 42
LOG_EVERY = 10
AMP_ENABLED = True

# Pre-registered decision threshold (fix 5c): H1 is "SUPPORTED" only if it
# beats the frozen B6 reference by at least this many degrees C of pooled
# validation RMSE.
IMPROVEMENT_THRESHOLD_C = 0.005

WORKING_DIR = "/kaggle/working"
RUN_ROOT = "/kaggle/working/OceanEmbed_Runs"

# Set explicitly if automatic discovery finds the wrong file.
B6_TRAINER_PATH: Optional[str] = None
B6_CHECKPOINT_PATH: Optional[str] = None
DATASET_ROOT: Optional[str] = None

# Number of real batches used by the explicit preflight diagnostics.
PREFLIGHT_BATCHES = 1

# Number of real batches pooled together for the fusion-affine calibration
# (fix 6a/6b). A single batch_size=1 sample is not representative of the
# training distribution.
CALIBRATION_BATCHES = 10

# Number of days used by the first calibration pass.
# All 14 days are used when a real batch is available.
CALIBRATION_EPS = 1e-6

# Alternate keys some B6 checkpoint writers may have used for the same
# fields (fix 2c). Verify the true key against your actual checkpoint with
# `python -c "import torch; print(torch.load(p, map_location='cpu', weights_only=False).keys())"`
# before the first run — see the "Minimum fix list" in the review.
FINGERPRINT_KEYS = ("backbone_fingerprint", "bundle_fingerprint", "fingerprint")
EXPERIMENT_NAME_VALUES = {"CONVLSTM_MASKED_B6", "B6", "CONVLSTM_MASKED_B6_EPOCH10"}


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


def compute_pooled_b6_reference(
    daily_sse: Sequence[float],
    daily_n: Sequence[float],
) -> float:
    """Recompute the B6 reference RMSE under the *pooled* convention (fix
    5d): sqrt(sum(SSE) / sum(N)) across A2.4's DAILY["Full"] per-day arrays.

    REFERENCE_B6_RMSE_WEIGHTED (0.706064451) is the A2.4 *weighted* value. If
    the framework's MetricAccumulator.finalize() reports pooled RMSE,
    comparing H1's result directly against the weighted constant will
    misjudge the outcome. Call this with the real per-day SSE/N arrays before
    training and use its return value as the comparison reference instead.
    """
    total_sse = float(sum(daily_sse))
    total_n = float(sum(daily_n))
    if total_n <= 0:
        raise ValueError("compute_pooled_b6_reference: daily_n sums to zero.")
    return math.sqrt(total_sse / total_n)


def _load_daily_arrays_from_report(path: Path) -> Optional[Tuple[List[float], List[float]]]:
    """Best-effort extraction of per-day SSE/N arrays from an
    A2_4_final_report.json-shaped file. Tries a few plausible key layouts
    since the exact schema of that report isn't part of this script."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[REFERENCE] could not parse {path}: {exc}", flush=True)
        return None

    candidates = [
        payload,
        payload.get("DAILY", {}).get("Full") if isinstance(payload.get("DAILY"), dict) else None,
        payload.get("daily", {}).get("full") if isinstance(payload.get("daily"), dict) else None,
    ]
    for node in candidates:
        if not isinstance(node, dict):
            continue
        for sse_key, n_key in (
            ("sse", "n"),
            ("SSE", "N"),
            ("daily_sse", "daily_n"),
            ("sum_squared_error", "count"),
        ):
            if sse_key in node and n_key in node:
                sse, n = node[sse_key], node[n_key]
                if isinstance(sse, list) and isinstance(n, list) and sse and n:
                    return [float(v) for v in sse], [float(v) for v in n]
    return None


def _load_daily_array_file(path: Path) -> Optional[List[float]]:
    """Load a flat numeric array from a JSON list or single-column CSV."""
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("values", "data", "sse", "n"):
                    if key in data and isinstance(data[key], list):
                        return [float(v) for v in data[key]]
                return None
            return [float(v) for v in data]
        # CSV: one number per line/column.
        text = path.read_text(encoding="utf-8")
        values: List[float] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            values.append(float(line.split(",")[0]))
        return values
    except Exception as exc:
        print(f"[REFERENCE] could not parse {path}: {exc}", flush=True)
        return None


def resolve_b6_reference(dataset_root: Optional[Path] = None) -> Tuple[float, str]:
    """Decide which B6 reference RMSE to compare against.

    This is where Option A vs Option B is actually decided, once, so the
    training result and verdict never silently mix the two conventions:

    1. If REFERENCE_B6_DAILY_REPORT_PATH (or SSE_PATH + N_PATH) is set and
       loads successfully, use the pooled RMSE computed from real per-day
       SSE/N arrays (Option A).
    2. Otherwise, try a small set of conventional filenames under
       dataset_root and /kaggle/input, on the chance the A2.4 report is
       already attached as a Kaggle input.
    3. Otherwise, fall back to REFERENCE_B6_RMSE_WEIGHTED and say so
       explicitly (Option B) — the run is NOT blocked, but the artifact
       records exactly which convention was used.

    Returns (reference_rmse, convention) where convention is "pooled" or
    "weighted".
    """
    explicit_paths: List[Path] = []
    if REFERENCE_B6_DAILY_REPORT_PATH:
        explicit_paths.append(Path(REFERENCE_B6_DAILY_REPORT_PATH))
    if REFERENCE_B6_DAILY_SSE_PATH and REFERENCE_B6_DAILY_N_PATH:
        sse_path = Path(REFERENCE_B6_DAILY_SSE_PATH)
        n_path = Path(REFERENCE_B6_DAILY_N_PATH)
        if sse_path.is_file() and n_path.is_file():
            sse = _load_daily_array_file(sse_path)
            n = _load_daily_array_file(n_path)
            if sse and n:
                value = compute_pooled_b6_reference(sse, n)
                print(
                    f"[REFERENCE] pooled B6 RMSE = {value:.9f} C "
                    f"(from {sse_path.name} / {n_path.name})",
                    flush=True,
                )
                return value, "pooled"

    for p in explicit_paths:
        if p.is_file():
            arrays = _load_daily_arrays_from_report(p)
            if arrays:
                value = compute_pooled_b6_reference(*arrays)
                print(
                    f"[REFERENCE] pooled B6 RMSE = {value:.9f} C (from {p})",
                    flush=True,
                )
                return value, "pooled"

    # Best-effort auto-discovery: look for a conventionally-named A2.4 report
    # near the dataset, in case it's already attached as a Kaggle input.
    search_roots = [r for r in (dataset_root, Path("/kaggle/input")) if r]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("A2_4_final_report*.json", "*A2.4*final_report*.json"):
            for p in root.rglob(pattern):
                arrays = _load_daily_arrays_from_report(p)
                if arrays:
                    value = compute_pooled_b6_reference(*arrays)
                    print(
                        f"[REFERENCE] pooled B6 RMSE = {value:.9f} C "
                        f"(auto-discovered at {p})",
                        flush=True,
                    )
                    return value, "pooled"

    print(
        "[REFERENCE] no A2.4 daily SSE/N arrays found or configured; "
        f"falling back to REFERENCE_B6_RMSE_WEIGHTED = "
        f"{REFERENCE_B6_RMSE_WEIGHTED:.9f} C. Set "
        "REFERENCE_B6_DAILY_REPORT_PATH (or the SSE/N path pair) once the "
        "real arrays are available, so the verdict compares like-for-like.",
        flush=True,
    )
    return REFERENCE_B6_RMSE_WEIGHTED, "weighted"


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
    }


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

    # Prefer the exact trainer family used for A2.4/B6.
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


# Module-level memo so repeated Model() construction (preflight probe +
# framework Trainer) does not re-scan and re-load every candidate checkpoint
# on disk each time (fix 2e).
_CHECKPOINT_CACHE: Optional[Path] = None


def _load_checkpoint_payload(p: Path) -> Optional[Dict[str, Any]]:
    """Load a checkpoint, preferring the safer weights_only=True path
    (fix 2d) and falling back to weights_only=False only if that fails."""
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _checkpoint_matches_b6_identity(payload: Dict[str, Any]) -> bool:
    """Match the pinned B6 fingerprint + experiment identity, tolerating the
    alternate key/value spellings noted in the review (fix 2c)."""
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
    return experiment_name in EXPERIMENT_NAME_VALUES


def discover_checkpoint(b6: ModuleType) -> Path:
    """Locate and validate the pinned B6 epoch-10 checkpoint.

    `b6` is accepted (rather than dropped) so that, in the future, callers
    can validate a candidate's state_dict against `b6.Model`'s parameter
    names before accepting it; today it is unused beyond documenting that
    intent (fix 2a — no longer a silently-dead parameter).
    """
    del b6  # not currently needed for the identity check itself

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

    # Stop at the first candidate that passes the identity check instead of
    # loading every candidate on disk (fix 2b).
    for p in sorted(set(candidates), key=lambda x: (len(str(x)), str(x))):
        payload = _load_checkpoint_payload(p)
        if payload is None:
            continue
        if _checkpoint_matches_b6_identity(payload):
            _CHECKPOINT_CACHE = p
            return p

    raise RuntimeError(
        "latest_model.pt files were found, but none passed the pinned B6 "
        "fingerprint + experiment identity check. Verify the exact keys "
        f"({FINGERPRINT_KEYS}) and experiment_name values "
        f"({sorted(EXPERIMENT_NAME_VALUES)}) against the epoch-10 checkpoint "
        "(see the 'Minimum fix list' in the code review)."
    )


# ============================================================================
# Mask-aware modality branch
# ============================================================================

class ModalityBranch(nn.Module):
    """
    Two-layer PartialConv branch.

    Structural ocean mask is applied after every layer, so land cells cannot
    propagate branch activations. A fully unsupported location is forced to
    exact zero in both activation and returned support mask.
    """

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
        # x/mask: [N,C,H,W]
        # ocean_mask: [N,1,H,W]
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
    """
    Trainable affine scale/bias.

    At the first real training forward, gamma/beta are initialized so the
    H1 fusion output approximately matches the frozen B6 stem's mean/std.
    After that, gamma/beta remain trainable.

    This is deliberately NOT called "near-identity initialization":
    the inputs to fusion are newly learned branch representations and have no
    identity relationship to the original B6 36-channel stem input.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

        self.register_buffer(
            "reference_mean",
            torch.zeros(channels),
            persistent=True,
        )
        self.register_buffer(
            "reference_std",
            torch.ones(channels),
            persistent=True,
        )
        self.register_buffer(
            "source_mean",
            torch.zeros(channels),
            persistent=True,
        )
        self.register_buffer(
            "source_std",
            torch.ones(channels),
            persistent=True,
        )
        self.register_buffer(
            "calibration_done",
            torch.tensor(False),
            persistent=True,
        )

    @staticmethod
    def _masked_channel_stats(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        mf = mask.float().expand_as(xf)
        count = mf.sum(dim=(0, 2, 3)).clamp_min(1.0)
        mean = (xf * mf).sum(dim=(0, 2, 3)) / count
        var = (((xf - mean.view(1, -1, 1, 1)) ** 2) * mf).sum(
            dim=(0, 2, 3)
        ) / count
        std = var.clamp_min(CALIBRATION_EPS).sqrt()
        return mean, std

    @torch.no_grad()
    def calibrate(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> Dict[str, float]:
        source_mean, source_std = self._masked_channel_stats(
            source, source_mask
        )
        reference_mean, reference_std = self._masked_channel_stats(
            reference, reference_mask
        )

        gamma = reference_std / source_std.clamp_min(CALIBRATION_EPS)
        beta = reference_mean - gamma * source_mean

        self.gamma.data.copy_(gamma.to(self.gamma.dtype))
        self.beta.data.copy_(beta.to(self.beta.dtype))
        self.source_mean.copy_(source_mean)
        self.source_std.copy_(source_std)
        self.reference_mean.copy_(reference_mean)
        self.reference_std.copy_(reference_std)
        self.calibration_done.fill_(True)

        pre_mean_gap = float(
            (source_mean - reference_mean).abs().mean().cpu()
        )
        pre_std_ratio = float(
            (source_std / reference_std.clamp_min(CALIBRATION_EPS)).mean().cpu()
        )

        calibrated = source * self.gamma.view(1, -1, 1, 1) + self.beta.view(
            1, -1, 1, 1
        )
        calibrated_mean, calibrated_std = self._masked_channel_stats(
            calibrated, source_mask
        )
        post_mean_gap = float(
            (calibrated_mean - reference_mean).abs().mean().cpu()
        )
        post_std_ratio = float(
            (calibrated_std / reference_std.clamp_min(CALIBRATION_EPS))
            .mean()
            .cpu()
        )

        return {
            "pre_mean_abs_gap": pre_mean_gap,
            "pre_std_ratio_mean": pre_std_ratio,
            "post_mean_abs_gap": post_mean_gap,
            "post_std_ratio_mean": post_std_ratio,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(
            1, -1, 1, 1
        )


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

        # B6 classes are loaded lazily from the host trainer at construction
        # time so the exact frozen PartialConv/ConvLSTM implementation is reused.
        b6 = ensure_b6_host_module()
        # Stored on the instance (fix 4a) rather than relied on only via the
        # global, so this model does not depend on module-level init order.
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

        # Reuse the same safer-first loader used during discovery (fix: was
        # weights_only=False unconditionally here, inconsistent with
        # _load_checkpoint_payload's weights_only=True-first path).
        payload = _load_checkpoint_payload(checkpoint)
        if payload is None:
            raise RuntimeError(f"Could not load B6 checkpoint: {checkpoint}")
        self.b6.load_state_dict(payload["model_state_dict"], strict=True)
        self.b6.eval()

        for p in self.b6.parameters():
            p.requires_grad = False

        # Preserve the exact B6 contract hash for compatibility with the
        # canonical data contract. H1 changes architecture, not input semantics.
        self.contract_hash = getattr(
            self.b6,
            "contract_hash",
            hashlib.sha256(
                json.dumps(contract, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )

        # Fix 6e / 6d: verify the frozen B6 attributes and stem output width
        # this model depends on actually exist before using them, instead of
        # crashing deep inside calibration with an AttributeError or a
        # mismatched-shape error at the first forward pass.
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
                f"{missing_b6_attrs}. Verify these names against the actual "
                "B6 trainer source (see the 'Minimum fix list' in the code "
                "review)."
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
                f"width ({FUSION_OUTPUT_CHANNELS}); the frozen ConvLSTM "
                "input would mismatch on the first forward pass."
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

        # Explicitly freeze every B6 parameter again after module construction.
        for p in self.b6.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # H1 input decomposition
    # ------------------------------------------------------------------

    def _prepared_inputs(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Uses the instance's own host-module reference (fix 4a) instead of
        # the module-level global, so behavior does not depend on whatever
        # else has run first in this process.
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

        # A location remains structurally valid if at least one branch has
        # valid support. This is explicitly OR-like, not an AND across sources.
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

        # Structural mask is enforced again after fusion.
        ocean_flat = ocean_seq.reshape(bsz * timesteps, 1, h, w)
        y = y * support * ocean_flat
        support = support * ocean_flat

        y = y.reshape(bsz, timesteps, FUSION_OUTPUT_CHANNELS, h, w)
        support = support.reshape(bsz, timesteps, 1, h, w)
        return y, support

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

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

        # Use all 14 real input days from the calibration batch.
        fusion_flat = fusion_input.reshape(
            bsz * timesteps,
            FUSION_INPUT_CHANNELS,
            h,
            w,
        )
        fusion_mask_flat = fusion_mask.reshape(
            bsz * timesteps,
            FUSION_INPUT_CHANNELS,
            h,
            w,
        )

        # Original B6 stem, immediately before pooling:
        # PartialConv -> GELU -> support/ocean masking.
        x_flat = seq.reshape(
            bsz * timesteps,
            TOTAL_CHANNELS,
            h,
            w,
        )
        mask_flat = seq_mask.reshape(
            bsz * timesteps,
            TOTAL_CHANNELS,
            h,
            w,
        )

        device_type = x_flat.device.type
        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            ref_projection, ref_support = self.b6.highres_encoder(
                x_flat.float(),
                mask_flat.float(),
            )
            ref_post_stem = self.b6.encoder_activation(ref_projection)
            ocean_flat = ocean_mask.reshape(
                bsz,
                1,
                h,
                w,
            ).unsqueeze(1).expand(
                -1,
                timesteps,
                -1,
                -1,
                -1,
            ).reshape(
                bsz * timesteps,
                1,
                h,
                w,
            ).float()
            ref_post_stem = ref_post_stem * ref_support.float() * ocean_flat
            ref_support = ref_support.float() * ocean_flat

            # H1 fusion in float32 for stable calibration statistics.
            h1_fusion, h1_support = self.fusion(
                fusion_flat.float(),
                fusion_mask_flat.float(),
            )
            h1_fusion = h1_fusion * h1_support.float() * ocean_flat
            h1_support = h1_support.float() * ocean_flat

            report = self.fusion_norm.calibrate(
                h1_fusion,
                h1_support,
                ref_post_stem,
                ref_support,
            )

        print(
            "[CALIBRATION] original B6 stem vs raw H1 fusion: "
            f"mean_gap={report['pre_mean_abs_gap']:.6g}, "
            f"std_ratio={report['pre_std_ratio_mean']:.6g}",
            flush=True,
        )
        print(
            "[CALIBRATION] after affine initialization: "
            f"mean_gap={report['post_mean_abs_gap']:.6g}, "
            f"std_ratio={report['post_std_ratio_mean']:.6g}",
            flush=True,
        )
        self._last_calibration_report = report
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
        """Pool masked statistics over multiple real batches (fix 6a/6b).

        The original code estimated the fusion-affine gamma/beta from a
        single batch_size=1 sample, which may not represent the training
        distribution (e.g. one date/region). This pools `num_batches` real
        samples before calling the underlying single-shot calibration.

        `warm_batches`: already-fetched, already-device-moved batches (e.g.
        the one batch the preflight probe fetched via `next(iter(...))`) to
        count toward `num_batches` before pulling any further ones from
        `loader`. This avoids spinning up the DataLoader's worker pool twice
        in a row when startup cost (NetCDF opens, index building, etc.) is
        non-trivial.
        """
        was_training = self.training
        self.eval()

        seqs: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        oceans: List[torch.Tensor] = []
        fusion_inputs: List[torch.Tensor] = []
        fusion_masks: List[torch.Tensor] = []

        def _consume(raw_batch: Dict[str, Any]) -> None:
            seq, seq_mask, ocean_mask = self._prepared_inputs(raw_batch)
            fusion_input, fusion_mask, _ = self._branch_forward_sequence(
                seq, seq_mask, ocean_mask
            )
            seqs.append(seq)
            masks.append(seq_mask)
            oceans.append(ocean_mask)
            fusion_inputs.append(fusion_input)
            fusion_masks.append(fusion_mask)

        for raw_batch in (warm_batches or [])[:num_batches]:
            _consume(raw_batch)

        remaining = num_batches - len(seqs)
        if remaining > 0:
            it = iter(loader)
            for _ in range(remaining):
                try:
                    raw_batch = next(it)
                except StopIteration:
                    break
                raw_batch = framework.move_batch_to_device(raw_batch, device)
                _consume(raw_batch)

        if not seqs:
            raise RuntimeError(
                "calibrate_from_loader: the loader produced zero batches; "
                "cannot calibrate the fusion affine."
            )

        report = self._calibrate_from_real_batch(
            torch.cat(seqs, dim=0),
            torch.cat(masks, dim=0),
            torch.cat(oceans, dim=0),
            torch.cat(fusion_inputs, dim=0),
            torch.cat(fusion_masks, dim=0),
        )
        print(
            f"[CALIBRATION] pooled over {len(seqs)} real batch(es) "
            f"(requested {num_batches})",
            flush=True,
        )

        self.train(was_training)
        return report

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def _forward_features(
        self,
        batch: Dict[str, Any],
        calibrate: bool = True,
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

        if calibrate and not bool(self.fusion_norm.calibration_done.item()):
            self._calibrate_from_real_batch(
                seq,
                seq_mask,
                ocean_mask,
                fusion_input,
                fusion_mask,
            )

        fused_highres_seq, fused_highres_mask_seq = self._fusion(
            fusion_input,
            fusion_mask,
            ocean_seq,
        )

        # Match the B6 stem's post-projection activation ordering.
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

        # Mask-aware B6 pooling and recurrent path.
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

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        fused, fused_mask, ocean_mask = self._forward_features(
            batch,
            calibrate=self.training,
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


# Global host module cache. The canonical TrainingFramework imports this H1
# module again when it instantiates Model, so Model must also be able to
# self-resolve the B6 host module rather than relying only on main().
B6_HOST_MODULE: Optional[ModuleType] = None


def ensure_b6_host_module() -> ModuleType:
    """Resolve the B6 host module, reusing any already-loaded instance
    (fix 4b) instead of importing a second live copy under a different
    module-name prefix."""
    global B6_HOST_MODULE
    if B6_HOST_MODULE is not None:
        return B6_HOST_MODULE

    # A fresh execution context (e.g. the framework re-importing this H1
    # module) starts with B6_HOST_MODULE == None even if a B6 host module is
    # already sitting in sys.modules under load_b6_runtime's prefix. Reuse it
    # rather than materializing the runtime a second time.
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
    """
    Real-batch preflight.

    Checks:
      1. output shape and finite values
      2. gradients are not required from frozen B6 modules
      3. mixed-validity branch semantics
      4. fully-invalid branch output is exactly zero
      5. structural land suppression
      6. raw H1 fusion vs original B6 stem scale gap
    """
    batch = framework.move_batch_to_device(batch, device)
    model.eval()

    seq, seq_mask, ocean_mask = model._prepared_inputs(batch)
    fusion_input, fusion_mask, debug = model._branch_forward_sequence(
        seq,
        seq_mask,
        ocean_mask,
    )

    # Explicit raw-scale comparison BEFORE affine calibration.
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

    # Find an actual ocean pixel with at least one valid and one invalid
    # physical modality on a real day. If the first batch has none, create a
    # controlled mixed-validity probe at a real ocean pixel.
    #
    # Fix 3c: candidate_cells lives on the GPU; the original code called
    # .tolist()/.item() per-cell per-branch (up to 5000 * 7 = 35,000 syncs).
    # Instead, move the coordinates to CPU once and compute branch validity
    # as a single vectorized tensor op outside the Python loop.
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

        # [num_branches, num_cells] bool: does this branch have ANY valid
        # channel at this cell on `day`?
        branch_names = list(CHANNEL_GROUPS.keys())
        per_branch_valid = torch.stack(
            [
                seq_mask[0, day, list(indices), :, :][:, py_idx, px_idx].any(
                    dim=0
                )
                for indices in CHANNEL_GROUPS.values()
            ],
            dim=0,
        )  # [num_branches, num_cells]

        any_valid = per_branch_valid.any(dim=0)
        all_valid = per_branch_valid.all(dim=0)
        mixed_mask = any_valid & ~all_valid  # exactly the cells we want

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

        # Fix 3a: the original code always used candidate_cells[0], which can
        # sit on the domain boundary. With padding_mode="replicate", a
        # truncated invalidation window then lets the PartialConv "borrow"
        # replicated edge values, so the fully-invalid branch does not
        # collapse to exact zero and the fallback check below raises
        # spuriously. Prefer an interior cell (>= 5 px from every edge, wide
        # enough for an 11x11 invalidation window) so the window is never
        # truncated; fall back to the first available cell only if no
        # interior ocean cell exists in this batch.
        margin = 5
        interior = (
            (candidate_cells_cpu[:, 0] >= margin)
            & (candidate_cells_cpu[:, 0] <= GRID_H - 1 - margin)
            & (candidate_cells_cpu[:, 1] >= margin)
            & (candidate_cells_cpu[:, 1] <= GRID_W - 1 - margin)
        )
        interior_cells = candidate_cells_cpu[interior]
        chosen = (
            interior_cells[0] if interior_cells.numel() > 0 else candidate_cells_cpu[0]
        )
        py, px = int(chosen[0].item()), int(chosen[1].item())
        day = 0

        # Controlled source-observability perturbation at a real ocean pixel:
        # force the salinity source invalid while retaining thermal validity.
        probe_batch = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        if "x_dynamic_valid" not in probe_batch:
            # Fix 3b: silently skipping the invalidation left salinity_pixel
            # nonzero, which then raised the misleading "did not collapse to
            # exact zero" error below. Fail with an explicit, accurate cause
            # instead.
            raise RuntimeError(
                "Mixed-validity preflight probe requires 'x_dynamic_valid' "
                "in the batch to construct a controlled invalidation, but it "
                "is missing. Check the B6 loader/adapter contract."
            )

        # A center pixel can still receive valid support through a 3x3
        # partial-convolution neighborhood. Because the branch has TWO 3x3
        # layers, invalidate an 11x11 neighborhood (fix 3a) so the center has
        # no possible salinity support through either receptive field, even
        # accounting for replicate-padding leakage near an edge.
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

    # Land suppression probe.
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

    # Fix 3d: model.eval() makes forward()'s `calibrate=self.training` False,
    # so the shape/finiteness check below would otherwise run against the
    # un-calibrated affine (gamma=1, beta=0) even though calibration has
    # already been diagnosed above — the "preflight output" would silently
    # not be representative of what training actually sees. Explicitly
    # calibrate (if not already done) using the fusion tensors already
    # computed above, then leave the model in eval() for the shape check.
    if not bool(model.fusion_norm.calibration_done.item()):
        model._calibrate_from_real_batch(
            seq,
            seq_mask,
            ocean_mask,
            fusion_input,
            fusion_mask,
        )

    # Check model output shape before training.
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
# Framework integration
# ============================================================================

def build_framework_config(
    framework: ModuleType,
    runtime: Dict[str, Path],
    dataset_root: Path,
) -> Any:
    base_kwargs: Dict[str, Any] = dict(
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

    # Fix 5a: a flat 1e-4 LR for 15 epochs risks under-converging or
    # overfitting the ~10-12K trainable front-end parameters. Prefer a cosine
    # decay from LEARNING_RATE to LEARNING_RATE_MIN if the framework's
    # TrainConfig exposes fields for it; otherwise fall back to the flat LR
    # rather than crashing on an unexpected constructor kwarg.
    optional_schedule_kwargs = dict(
        lr_scheduler=LR_SCHEDULE,
        lr_schedule=LR_SCHEDULE,
        min_learning_rate=LEARNING_RATE_MIN,
        learning_rate_min=LEARNING_RATE_MIN,
        eta_min=LEARNING_RATE_MIN,
    )
    try:
        import inspect

        accepted = set(inspect.signature(framework.TrainConfig).parameters)
    except (TypeError, ValueError):
        accepted = set()

    applied_schedule_kwargs = {
        k: v for k, v in optional_schedule_kwargs.items() if k in accepted
    }
    if applied_schedule_kwargs:
        print(
            f"[CONFIG] applying LR schedule kwargs: {applied_schedule_kwargs}",
            flush=True,
        )
    else:
        print(
            "[CONFIG] framework.TrainConfig does not expose an LR-schedule "
            f"kwarg; training will use a flat learning_rate={LEARNING_RATE}. "
            "Add cosine decay in the framework's optimizer setup if desired "
            "(fix 5a).",
            flush=True,
        )

    cfg = framework.TrainConfig(**base_kwargs, **applied_schedule_kwargs)
    cfg.validate()
    return cfg


def resolve_dataset_root(b6: ModuleType) -> Path:
    if DATASET_ROOT:
        p = Path(DATASET_ROOT).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"DATASET_ROOT does not exist: {p}")
        return p

    if hasattr(b6, "_resolve_dataset_input_root"):
        return Path(b6._resolve_dataset_input_root()).resolve()

    candidates = []
    for p in Path("/kaggle/input").glob("**/oceanembed-modeldata-v1"):
        if p.is_dir():
            candidates.append(p)

    if candidates:
        return sorted(candidates, key=lambda x: len(str(x)))[0]

    raise FileNotFoundError(
        "Could not resolve OceanEmbed model-data root. Set DATASET_ROOT explicitly."
    )


def run_training(
    framework: ModuleType,
    runtime: Dict[str, Path],
    dataset_root: Path,
) -> Dict[str, Any]:
    # Resolve once, up front, which RMSE convention/value this run will be
    # judged against (Option A if real pooled data is available, Option B —
    # explicitly labeled — otherwise). Everything downstream (verdict logic,
    # printed banner, JSON spec) uses this single resolved value so the
    # weighted constant can never silently leak in as if it were pooled.
    reference_rmse, reference_convention = resolve_b6_reference(dataset_root)

    cfg = build_framework_config(
        framework,
        runtime,
        dataset_root,
    )

    # Build adapter first so the exact frozen data contract is available.
    adapter = framework.BackboneAdapter(cfg)

    # Contract/runtime guards from the B6 host are reused.
    if hasattr(B6_HOST_MODULE, "validate_backbone_runtime"):
        B6_HOST_MODULE.validate_backbone_runtime(adapter)
    if hasattr(B6_HOST_MODULE, "validate_runtime_normalization"):
        norm_report = B6_HOST_MODULE.validate_runtime_normalization(adapter)
    else:
        norm_report = None

    # Explicit B6 checkpoint contract validation happens inside Model().
    # A fresh model is instantiated by the canonical Trainer. We run the
    # preflight on a separate instance first so all diagnostics happen before
    # optimizer updates begin.
    probe_model = Model(adapter.contract)

    device = torch.device(cfg.device)
    probe_model.to(device)

    raw_batch = framework.move_batch_to_device(
        next(iter(adapter.train_loader)), device
    )
    preflight = run_preflight_probe(
        probe_model,
        raw_batch,
        framework,
        device,
    )

    print("[PREFLIGHT] PASS", flush=True)
    print(
        json.dumps(
            preflight,
            indent=2,
            default=str,
        ),
        flush=True,
    )

    # Explicitly verify frozen backbone gradients are disabled.
    frozen_grad_params = [
        name
        for name, p in probe_model.b6.named_parameters()
        if p.requires_grad
    ]
    if frozen_grad_params:
        raise RuntimeError(
            "Frozen B6 parameter(s) unexpectedly trainable: "
            + ", ".join(frozen_grad_params[:10])
        )

    print(
        "[PREFLIGHT] B6 frozen parameters: PASS",
        flush=True,
    )

    # Canonical framework Trainer performs the actual training/validation and
    # checkpoint management. H1's Model constructor internally loads the exact
    # B6 epoch-10 state, while the H1 branches/fusion remain trainable.
    trainer = framework.Trainer(cfg, adapter)

    # Fix 6a/6b: calibrate the actual training model's fusion affine from
    # several pooled real batches, explicitly and before any optimizer step,
    # rather than letting it happen implicitly (and from a single batch) on
    # the first training-mode forward call. Reuse the batch already fetched
    # above for the preflight probe (`raw_batch`) as the first calibration
    # sample, so the DataLoader's worker pool isn't spun up twice in a row.
    if not bool(trainer.model.fusion_norm.calibration_done.item()):
        trainer.model.calibrate_from_loader(
            adapter.train_loader,
            framework,
            device,
            CALIBRATION_BATCHES,
            warm_batches=[raw_batch],
        )

    if getattr(trainer.model, "model_id", None) != MODEL_ID:
        raise RuntimeError(
            "Framework instantiated an unexpected H1 model identity: "
            f"{getattr(trainer.model, 'model_id', None)!r}"
        )

    # Ensure the framework did not accidentally expose B6 parameters to training.
    trainable_b6 = [
        name
        for name, p in trainer.model.b6.named_parameters()
        if p.requires_grad
    ]
    if trainable_b6:
        raise RuntimeError(
            "B6 backbone is not frozen in the training model: "
            + ", ".join(trainable_b6[:10])
        )

    result = trainer.train()

    # Fix 5c/6f/(follow-up): automate the pre-registered decision using the
    # reference resolved once at the top of run_training, instead of the
    # unconditionally-weighted constant. If MetricAccumulator's convention is
    # unknown, we still don't guess it away: an explicit assumption is
    # recorded either way (Option A when pooled data was found, Option B —
    # named and dated — otherwise).
    result = dict(result) if isinstance(result, dict) else {"raw_result": result}
    best_val_rmse = result.get("best_val_rmse")
    if best_val_rmse is None:
        verdict = "UNKNOWN_NO_BEST_VAL_RMSE_REPORTED"
    elif best_val_rmse < reference_rmse - IMPROVEMENT_THRESHOLD_C:
        verdict = "SUPPORTED"
    elif best_val_rmse < reference_rmse:
        verdict = "INCONCLUSIVE_MARGINAL"
    else:
        verdict = "NOT_SUPPORTED"

    result["h1_verdict"] = verdict
    result["h1_verdict_reference_rmse_c"] = reference_rmse
    result["h1_verdict_reference_convention"] = reference_convention
    result["h1_verdict_threshold_c"] = IMPROVEMENT_THRESHOLD_C
    if reference_convention == "pooled":
        result["h1_verdict_reference_convention_note"] = (
            "reference_rmse_c was computed as a POOLED RMSE from real A2.4 "
            "per-day SSE/N arrays. Confirm best_val_rmse above is also "
            "pooled (not weighted) before trusting this verdict."
        )
    else:
        result["h1_verdict_reference_convention_warning"] = (
            "No A2.4 daily SSE/N arrays were found or configured, so "
            "reference_rmse_c falls back to REFERENCE_B6_RMSE_WEIGHTED "
            f"({REFERENCE_B6_RMSE_WEIGHTED:.9f} C). Confirm "
            "framework.MetricAccumulator.finalize() reports WEIGHTED RMSE "
            "(matching this constant) before trusting this verdict; if it "
            "reports pooled RMSE instead, set REFERENCE_B6_DAILY_REPORT_PATH "
            "(or the SSE/N path pair) and rerun so resolve_b6_reference() "
            "can compute the pooled value via compute_pooled_b6_reference()."
        )

    if verdict == "SUPPORTED":
        print(
            "[DECISION] H1-min improved on the pre-registered threshold -> "
            "proceed to the conditional H1-ablate run.",
            flush=True,
        )
    else:
        print(
            f"[DECISION] H1-min verdict = {verdict} -> do not proceed to "
            "H1-ablate under the pre-registered protocol.",
            flush=True,
        )

    run_dir = Path(trainer.run_dir)
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "scientific_role": "modality_specific_front_end_hypothesis_test",
        "reference": {
            "model_id": "CONVLSTM_MASKED_B6",
            "reference_rmse_c_used_for_verdict": reference_rmse,
            "reference_rmse_convention_used_for_verdict": reference_convention,
            "reference_rmse_c_weighted_constant": REFERENCE_B6_RMSE_WEIGHTED,
            "reference_epoch": 10,
            "reference_val_loss": REFERENCE_B6_VAL_LOSS,
            "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        },
        "architecture": {
            "branch_groups": CHANNEL_GROUPS,
            "branch_widths": BRANCH_WIDTHS,
            "fusion_channels": [FUSION_INPUT_CHANNELS, FUSION_OUTPUT_CHANNELS],
            "calibration": (
                "trainable affine initialized from real-batch statistics "
                "matching H1 fusion output to original B6 stem output "
                "immediately before pooling"
            ),
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
        "channel_contract": {
            "total": TOTAL_CHANNELS,
            "window_days": WINDOW_DAYS,
            "grid": [GRID_H, GRID_W],
            "depths": N_DEPTH,
        },
        "preflight": preflight,
        "normalization_guard": norm_report,
        "parameters": parameter_summary(trainer.model),
        "result": result,
    }

    atomic_json(
        run_dir / "h1_pam_convlstm_b7_experiment_spec.json",
        spec,
    )

    return {
        "run_dir": str(run_dir),
        "result": result,
        "preflight": preflight,
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

    # Exercise the multi-batch calibration path (fix: self-test only used to
    # cover the single-batch path inside run_preflight_probe). num_batches=2
    # keeps the self-test cheap while still catching loader-iteration bugs
    # in calibrate_from_loader before a real, expensive training run.
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
    print(f"Reference B6 RMSE (weighted, pinned): {REFERENCE_B6_RMSE_WEIGHTED:.9f} C")
    print(
        "  (the pooled convention, if resolvable, is chosen and logged "
        "inside run_training via resolve_b6_reference())"
    )
    print(f"Reference B6 epoch: 10")
    print(f"Bundle fingerprint: {EXPECTED_BUNDLE_FP}")
    print("=" * 80)

    global B6_HOST_MODULE
    b6, runtime = load_b6_runtime()
    B6_HOST_MODULE = b6

    dataset_root = resolve_dataset_root(b6)
    print(f"[DATA] dataset root: {dataset_root}", flush=True)

    checkpoint = discover_checkpoint(b6)
    print(f"[B6] selected checkpoint: {checkpoint}", flush=True)

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
