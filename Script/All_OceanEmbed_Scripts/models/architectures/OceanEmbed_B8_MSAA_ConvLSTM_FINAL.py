"""
OceanEmbed B8 — MSAA-ConvLSTM
Multi-Scale Spatial Attention + ConvLSTM + Depth-Aware Reconstruction

Purpose
-------
A single serious final-model candidate built to attack the limitations exposed
by B6/B7 without changing the frozen OceanEmbed data contract:

    mask-aware CNN stem
        -> multi-scale spatial CNN (local / medium / large receptive fields)
        -> masked window self-attention
        -> fully trainable mask-aware ConvLSTM x2
        -> high-resolution U-Net-style skip decoder
        -> depth-conditioned residual reconstruction head

B8 is warm-started from the exact B6 epoch-10 checkpoint for the matching
submodules. New modules are zero-initialized only where residual identity is
intentional, so the initial model is close to the established B6 reference
rather than an unrelated random network.

This is NOT a B7 continuation and does NOT use modality-specific branches.
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


# ==========================================================================
# B8 runtime configuration
# ==========================================================================

RUN_MODE = "train"  # "train" or "self-test"

EXPERIMENT_ID = "B8"
MODEL_ID = "MSAA_CONVLSTM_B8"
MODEL_VERSION = "8.0.0"
MODEL_CONTRACT = "MSAACONVLSTMB8.0"
MODEL_NAME = MODEL_ID

REFERENCE_B6_RMSE_WEIGHTED = 0.706064451
REFERENCE_B6_VAL_LOSS = 0.9249299274
REFERENCE_B6_EPOCH = 10
REFERENCE_B6_METRIC_NAME = "RMSE_weighted"
EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)

# Fresh B8 run. The B6 weights are used as a warm start inside matching
# submodules, but B8 has its own optimizer state and its own checkpoints.
EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
# Verified TrainingFramework behavior: PATIENCE=0 disables framework early
# stopping because the framework guards with `if patience > 0`.
PATIENCE = 0
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
SEED = 42
LOG_EVERY = 10
AMP_ENABLED = True

# Fast, no-duplicate diagnostic policy. RUN_MODE="train" goes straight to
# canonical training; exact B6 checkpoint/fingerprint and runtime contract
# checks still happen during model construction.
RUN_PREFLIGHT_IN_TRAIN = False
PREFLIGHT_BATCHES = 1

IMPROVEMENT_THRESHOLD_C = 0.005
WORKING_DIR = "/kaggle/working"
RUN_ROOT = "/kaggle/working/OceanEmbed_Runs"

# Exact frozen B6 checkpoint. Leave None for deterministic auto-discovery of
# a checkpoint that passes the pinned epoch-10 identity/fingerprint guard.
B6_TRAINER_PATH: Optional[str] = None
B6_CHECKPOINT_PATH: Optional[str] = None
DATASET_ROOT: Optional[str] = "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"

FINGERPRINT_KEYS = ("backbone_fingerprint",)
EXPERIMENT_NAME_VALUES = {"CONVLSTM_MASKED_B6"}


# ==========================================================================
# Frozen OceanEmbed channel / tensor contract
# ==========================================================================

GRID_H, GRID_W = 100, 240
WINDOW_DAYS = 14
DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2
TOTAL_CHANNELS = 36
N_DEPTH = 15
DEPTHS_M = [
    0.0, 5.0, 10.0, 20.0, 30.0,
    50.0, 75.0, 100.0, 125.0, 150.0,
    200.0, 300.0, 500.0, 700.0, 1000.0,
]

# ==========================================================================
# B8 architecture constants
# ==========================================================================

BASE_CHANNELS = 32
MULTISCALE_BRANCH_CHANNELS = 16
ATTN_DIM = 32
ATTN_HEADS = 4
ATTN_WINDOW = 5  # 50 x 120 is exactly divisible by a 5 x 5 window.
ATTN_MLP_RATIO = 2.0
DEPTH_FEATURES = 16
DEPTH_HEAD_HIDDEN = 24

assert GRID_H % 2 == 0 and GRID_W % 2 == 0
assert (GRID_H // 2) % ATTN_WINDOW == 0
assert (GRID_W // 2) % ATTN_WINDOW == 0
assert ATTN_DIM == BASE_CHANNELS

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
# B8 building blocks
# ============================================================================

class MaskedDilatedConv2d(nn.Module):
    """Partial-convolution equivalent supporting dilation 1/2/4."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dilation = int(dilation)
        self.kernel_size = 3
        self.padding = self.dilation
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 3, 3)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_channels * 9)
        nn.init.uniform_(self.bias, -bound, bound)
        self.register_buffer(
            "_ones_kernel",
            torch.ones(1, in_channels, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "_kernel_elements",
            torch.tensor(float(in_channels * 9)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.expand(-1, self.in_channels, -1, -1)
        if tuple(mask.shape) != tuple(x.shape):
            raise ValueError(
                f"MaskedDilatedConv2d mask {tuple(mask.shape)} != x {tuple(x.shape)}"
            )
        mask = mask.to(dtype=x.dtype, device=x.device)
        x_masked = x * mask
        p = self.padding
        x_pad = F.pad(x_masked, (p, p, p, p), mode="replicate")
        mask_pad = F.pad(mask, (p, p, p, p), mode="replicate")
        numerator = F.conv2d(
            x_pad,
            self.weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=self.dilation,
        )
        valid_count = F.conv2d(
            mask_pad,
            self._ones_kernel.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            padding=0,
            dilation=self.dilation,
        )
        valid_fraction = valid_count / self._kernel_elements.to(
            dtype=x.dtype, device=x.device
        )
        safe_fraction = valid_fraction.clamp_min(1e-6)
        out = numerator / safe_fraction + self.bias.view(1, -1, 1, 1)
        support = (valid_count > 0).to(dtype=x.dtype)
        return out * support, support



def _init_small_residual_module(module: nn.Module, std: float = 1e-3) -> None:
    """Initialize a residual projection near zero without blocking gradients."""
    for name, p in module.named_parameters():
        if p.ndim >= 2:
            nn.init.normal_(p, mean=0.0, std=std)
        else:
            nn.init.zeros_(p)


def _init_linear_small(linear: nn.Linear, std: float = 1e-3) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MultiScaleResidualBlock(nn.Module):
    """Mask-aware local/medium/large receptive-field residual CNN."""

    def __init__(
        self,
        channels: int,
        branch_channels: int,
        partial_conv_cls: type,
    ):
        super().__init__()
        self.channels = int(channels)
        self.branch_channels = int(branch_channels)

        self.local = MaskedDilatedConv2d(channels, branch_channels, dilation=1)
        self.medium = MaskedDilatedConv2d(channels, branch_channels, dilation=2)
        self.large = MaskedDilatedConv2d(channels, branch_channels, dilation=4)
        self.fuse = partial_conv_cls(
            branch_channels * 3, channels, kernel_size=1, padding=0,
            padding_mode="replicate", bias=True,
        )
        # Residual branch starts at zero, so warm-started B6 features are
        # preserved exactly at initialization while gradients can immediately
        # train the new multi-scale pathway.
        _init_small_residual_module(self.fuse)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool() & ocean_mask.bool()
        local, ml = self.local(x, mask)
        medium, mm = self.medium(x, mask)
        large, mg = self.large(x, mask)

        local = self.act(local) * ml * ocean_mask
        medium = self.act(medium) * mm * ocean_mask
        large = self.act(large) * mg * ocean_mask

        cat = torch.cat([local, medium, large], dim=1)
        cat_mask = torch.cat(
            [
                ml.expand(-1, self.branch_channels, -1, -1),
                mm.expand(-1, self.branch_channels, -1, -1),
                mg.expand(-1, self.branch_channels, -1, -1),
            ],
            dim=1,
        ) * ocean_mask

        delta, delta_mask = self.fuse(cat, cat_mask)
        delta = delta * delta_mask * ocean_mask

        out_mask = ((mask.float() + delta_mask.float()) > 0).to(x.dtype)
        out_mask = out_mask * ocean_mask
        out = (x * mask.to(x.dtype) + delta) * out_mask
        return out, out_mask


class MaskedWindowSelfAttention(nn.Module):
    """Windowed self-attention with explicit key masking and no padding gaps."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Attention dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        # Start the entire attention/MLP residual contribution at zero. This
        # makes B8 initialize as the warm-started B6 path while retaining all
        # attention parameters as trainable.
        _init_linear_small(self.proj)
        _init_linear_small(self.mlp[-1])

    @staticmethod
    def _window_partition(
        x: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        bsz, h, w, c = x.shape
        x = x.view(
            bsz, h // window, window,
            w // window, window, c,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(-1, window * window, c)

    @staticmethod
    def _window_reverse(
        windows: torch.Tensor,
        window: int,
        h: int,
        w: int,
        batch_size: int,
    ) -> torch.Tensor:
        c = windows.shape[-1]
        x = windows.view(
            batch_size, h // window, w // window,
            window, window, c,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size, h, w, c)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"Attention expects [B,{self.dim},H,W], got {tuple(x.shape)}")
        if mask.ndim == 4 and mask.shape[1] != 1:
            mask = mask[:, :1]
        mask = (mask > 0.5).to(dtype=x.dtype, device=x.device)
        bsz, _, h, w = x.shape
        if h % self.window_size or w % self.window_size:
            raise ValueError(
                f"Grid {(h,w)} not divisible by attention window {self.window_size}"
            )

        tokens = x.permute(0, 2, 3, 1).contiguous()
        tokens = self.norm1(tokens)
        token_windows = self._window_partition(tokens, self.window_size)

        mask_hw = mask[:, 0]
        mask_windows = self._window_partition(
            mask_hw.unsqueeze(-1), self.window_size
        ).squeeze(-1) > 0.5

        qkv = self.qkv(token_windows)
        qkv = qkv.view(
            qkv.shape[0], qkv.shape[1], 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention logits in float32 for AMP stability.
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale
        key_valid = mask_windows[:, None, None, :]
        scores = scores.masked_fill(~key_valid, -1e4)
        weights = torch.softmax(scores, dim=-1).to(dtype=v.dtype)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).contiguous().view(
            token_windows.shape[0], token_windows.shape[1], self.dim
        )
        attended = self.proj(attended)

        y = token_windows + attended
        y = y + self.mlp(self.norm2(y))
        y = y * mask_windows.unsqueeze(-1).to(y.dtype)

        y = self._window_reverse(
            y,
            self.window_size,
            h,
            w,
            bsz,
        )
        y = y.permute(0, 3, 1, 2).contiguous()
        y = y * mask
        return y, mask


class DepthConditionedResidualHead(nn.Module):
    """Depth-conditioned residual correction around a warm-started B6 head."""

    def __init__(
        self,
        feature_channels: int,
        partial_conv_cls: type,
        n_depth: int,
    ):
        super().__init__()
        self.n_depth = int(n_depth)
        self.feature_proj = nn.Conv2d(feature_channels, DEPTH_FEATURES, kernel_size=1)
        self.depth_mlp = nn.Sequential(
            nn.Linear(1, DEPTH_FEATURES),
            nn.GELU(),
            nn.Linear(DEPTH_FEATURES, DEPTH_FEATURES * 2),
        )
        self.residual_head = nn.Conv2d(DEPTH_FEATURES, 1, kernel_size=1)
        # Near-zero residual keeps the B6 warm start while allowing immediate
        # gradient flow into both the residual head and depth conditioner.
        nn.init.normal_(self.residual_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head.bias)

        depths = torch.tensor(DEPTHS_M, dtype=torch.float32).view(-1, 1)
        log_depth = torch.log1p(depths) / math.log1p(1000.0)
        self.register_buffer("normalized_log_depth", log_depth, persistent=True)

        self.alpha = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ocean_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, h, w = x.shape
        z = self.feature_proj(x)
        depth_params = self.depth_mlp(self.normalized_log_depth)
        scale, bias = depth_params.chunk(2, dim=-1)
        scale = scale.view(1, self.n_depth, DEPTH_FEATURES, 1, 1)
        bias = bias.view(1, self.n_depth, DEPTH_FEATURES, 1, 1)

        z = z.unsqueeze(1)
        conditioned = z * (1.0 + scale) + bias
        conditioned = conditioned.reshape(
            bsz * self.n_depth, DEPTH_FEATURES, h, w
        )
        residual = self.residual_head(conditioned).view(
            bsz, self.n_depth, h, w
        )
        valid = (mask > 0.5) & (ocean_mask > 0.5)
        residual = residual * valid.to(residual.dtype)
        return self.alpha * residual


# ============================================================================
# B8 model
# ============================================================================


class Model(nn.Module):
    """Warm-started multi-scale attention ConvLSTM with depth-aware residual head."""

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
        print(f"[B6] validating checkpoint for B8 warm start: {checkpoint}", flush=True)
        metadata = b6._validate_resume_checkpoint_metadata(checkpoint, self.b6)
        print(
            "[B6] warm-start checkpoint metadata: "
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

        partial_conv_cls = type(self.b6.highres_encoder)
        pool_cls = type(self.b6.pool)
        convlstm_cls = type(self.b6.convlstm1)

        # ------------------------------------------------------------------
        # Warm-started B6-compatible pathway components.
        # ------------------------------------------------------------------
        self.input_stem = partial_conv_cls(
            TOTAL_CHANNELS, BASE_CHANNELS,
            kernel_size=3, padding=1,
            padding_mode="replicate", bias=True,
        )
        self.input_stem.load_state_dict(self.b6.highres_encoder.state_dict(), strict=True)
        self.stem_activation = nn.GELU()
        self.pool = pool_cls(kernel_size=2, stride=2)

        self.convlstm1 = convlstm_cls(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3)
        self.convlstm2 = convlstm_cls(BASE_CHANNELS, BASE_CHANNELS, kernel_size=3)
        self.convlstm1.load_state_dict(self.b6.convlstm1.state_dict(), strict=True)
        self.convlstm2.load_state_dict(self.b6.convlstm2.state_dict(), strict=True)

        # ------------------------------------------------------------------
        # New B8 representation modules.
        # ------------------------------------------------------------------
        self.multiscale = MultiScaleResidualBlock(
            BASE_CHANNELS, MULTISCALE_BRANCH_CHANNELS, partial_conv_cls
        )
        self.attention = MaskedWindowSelfAttention(
            ATTN_DIM, ATTN_HEADS, ATTN_WINDOW, ATTN_MLP_RATIO
        )

        # ------------------------------------------------------------------
        # Warm-started U-Net-style high-resolution decoder projection.
        # ------------------------------------------------------------------
        self.decoder_projection = partial_conv_cls(
            BASE_CHANNELS + BASE_CHANNELS,
            BASE_CHANNELS,
            kernel_size=3, padding=1,
            padding_mode="replicate", bias=True,
        )
        self.decoder_projection.load_state_dict(
            self.b6.decoder_projection.state_dict(), strict=True
        )
        self.decoder_activation = nn.GELU()
        self.decoder_refine = MultiScaleResidualBlock(
            BASE_CHANNELS, MULTISCALE_BRANCH_CHANNELS, partial_conv_cls
        )

        self.base_head = partial_conv_cls(
            BASE_CHANNELS, N_DEPTH,
            kernel_size=1, padding=0,
            padding_mode="replicate", bias=True,
        )
        self.base_head.load_state_dict(self.b6.head.state_dict(), strict=True)

        self.depth_residual = DepthConditionedResidualHead(
            BASE_CHANNELS,
            partial_conv_cls,
            N_DEPTH,
        )

        self._warm_start_checkpoint = str(checkpoint)
        self._warm_start_metadata = metadata

    def _prepared_inputs(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        host = getattr(self, "_b6_host", None) or ensure_b6_host_module()
        seq, seq_mask, ocean_mask, _ = host.prepare_model_inputs(batch)
        return seq, seq_mask, ocean_mask

    def _forward_features(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        seq, seq_mask, ocean_mask_hr = self._prepared_inputs(batch)
        bsz, timesteps, _, h, w = seq.shape
        if (h, w) != (GRID_H, GRID_W):
            raise RuntimeError(f"Unexpected grid {(h,w)}; expected {(GRID_H,GRID_W)}")
        if timesteps != WINDOW_DAYS:
            raise RuntimeError(f"Unexpected sequence length {timesteps}; expected {WINDOW_DAYS}")

        ocean_mask_hr = ocean_mask_hr.to(dtype=seq.dtype)
        lowres_ocean_mask = F.max_pool2d(
            ocean_mask_hr, kernel_size=2, stride=2, padding=0
        )
        lowres_ocean_mask = (lowres_ocean_mask > 0).to(dtype=seq.dtype)

        h1 = c1 = None
        h2 = c2 = None
        final_highres = None
        final_highres_mask = None
        diagnostics: Dict[str, Any] = {"attention_window": ATTN_WINDOW}

        for t in range(timesteps):
            x_t = seq[:, t]
            mask_t = seq_mask[:, t].to(dtype=seq.dtype)

            high, high_mask = self.input_stem(x_t, mask_t)
            high = self.stem_activation(high)
            high = high * high_mask * ocean_mask_hr
            high_mask = high_mask * ocean_mask_hr

            high, high_mask = self.multiscale(
                high, high_mask, ocean_mask_hr
            )

            pooled, pooled_mask = self.pool(high, high_mask)
            pooled = pooled * lowres_ocean_mask
            pooled_mask = pooled_mask * lowres_ocean_mask

            pooled, pooled_mask = self.attention(
                pooled, pooled_mask
            )
            pooled = pooled * pooled_mask * lowres_ocean_mask

            h1, c1, state_mask1 = self.convlstm1(
                pooled, pooled_mask, h1, c1, lowres_ocean_mask
            )
            h2, c2, state_mask2 = self.convlstm2(
                h1, state_mask1, h2, c2, lowres_ocean_mask
            )

            h1 = h1 * lowres_ocean_mask
            c1 = c1 * lowres_ocean_mask
            h2 = h2 * lowres_ocean_mask
            c2 = c2 * lowres_ocean_mask

            if t == timesteps - 1:
                final_highres = high
                final_highres_mask = high_mask

        if h2 is None or final_highres is None or final_highres_mask is None:
            raise RuntimeError("B8 temporal loop produced no final state")

        recurrent_up = F.interpolate(
            h2, size=(GRID_H, GRID_W), mode="bilinear", align_corners=False
        )
        recurrent_mask_hr = F.interpolate(
            lowres_ocean_mask, size=(GRID_H, GRID_W), mode="nearest"
        ) * ocean_mask_hr
        recurrent_up = recurrent_up * recurrent_mask_hr

        # U-Net-style high-resolution skip: recurrent low-resolution context
        # is fused with the final-day high-resolution spatial representation.
        fused = torch.cat([recurrent_up, final_highres], dim=1)
        fused_mask = torch.cat(
            [
                recurrent_mask_hr.expand(-1, BASE_CHANNELS, -1, -1),
                final_highres_mask.expand(-1, BASE_CHANNELS, -1, -1),
            ],
            dim=1,
        )
        fused_mask = fused_mask * ocean_mask_hr
        fused = fused * fused_mask
        return fused, fused_mask, ocean_mask_hr, diagnostics

    def train(self, mode: bool = True):
        super().train(mode)
        # B8 is fully trainable; there is no frozen B6 forward path. `self.b6`
        # remains only as a frozen provenance/reference copy for checkpoint
        # lineage, so keep it in eval mode.
        self.b6.eval()
        return self

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        fused, fused_mask, ocean_mask_hr, diagnostics = self._forward_features(batch)

        x, mask = self.decoder_projection(fused, fused_mask)
        x = self.decoder_activation(x)
        x = x * mask * ocean_mask_hr

        x, mask = self.decoder_refine(x, mask, ocean_mask_hr)

        base_y, base_mask = self.base_head(x, mask)
        base_y = base_y * base_mask * ocean_mask_hr

        residual = self.depth_residual(x, mask, ocean_mask_hr)
        y = (base_y + residual) * ocean_mask_hr

        expected = (fused.shape[0], N_DEPTH, GRID_H, GRID_W)
        if tuple(y.shape) != expected:
            raise RuntimeError(f"B8 output shape {tuple(y.shape)} != {expected}")
        if not torch.isfinite(y).all():
            raise FloatingPointError("B8 produced non-finite predictions")
        return y

    def model_summary(self) -> Dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_contract": MODEL_CONTRACT,
            "reference_model": "CONVLSTM_MASKED_B6",
            "reference_epoch": REFERENCE_B6_EPOCH,
            "reference_rmse_c_weighted": REFERENCE_B6_RMSE_WEIGHTED,
            "warm_start": {
                "checkpoint": self._warm_start_checkpoint,
                "matching_submodules": [
                    "input_stem <- B6 highres_encoder",
                    "ConvLSTM1 <- B6 convlstm1",
                    "ConvLSTM2 <- B6 convlstm2",
                    "decoder_projection <- B6 decoder_projection",
                    "base_head <- B6 head",
                ],
                "new_residual_modules_zero_initialized": [
                    "multi-scale fusion projections",
                    "attention output projection",
                    "attention MLP output",
                    "depth residual head",
                ],
            },
            "architecture": {
                "input": [WINDOW_DAYS, TOTAL_CHANNELS, GRID_H, GRID_W],
                "stem": "masked 3x3 36->32 @ 100x240",
                "multiscale": [
                    "3x3 dilation=1, 32->16",
                    "3x3 dilation=2, 32->16",
                    "3x3 dilation=4, 32->16",
                    "masked 1x1 fusion 48->32 residual",
                ],
                "pool": "masked 2x2 -> 50x120",
                "attention": {
                    "type": "masked window self-attention",
                    "grid": [GRID_H // 2, GRID_W // 2],
                    "window": ATTN_WINDOW,
                    "heads": ATTN_HEADS,
                    "dim": ATTN_DIM,
                    "mlp_hidden": int(ATTN_DIM * ATTN_MLP_RATIO),
                },
                "recurrent": "2-layer mask-aware ConvLSTM, 32 hidden channels",
                "decoder": "2x bilinear upsample + final-day high-resolution skip + masked 3x3 + multiscale refine",
                "depth_head": "warm-started 1x1 B6 anomaly head + log-depth-conditioned residual correction",
                "depths_m": DEPTHS_M,
                "output_shape": ["B", N_DEPTH, GRID_H, GRID_W],
                "masking": "no imputation; validity masks + structural ocean mask throughout",
            },
            "parameters": parameter_summary(self),
            "diagnostics": diagnostics_placeholder(),
        }


def diagnostics_placeholder() -> Dict[str, Any]:
    return {
        "preflight": "disabled_in_train_mode",
        "attention": {
            "window": ATTN_WINDOW,
            "heads": ATTN_HEADS,
        },
    }


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
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
# B8 training / checkpointing
# ============================================================================


def _read_training_history(run_dir: Path) -> List[Dict[str, Any]]:
    candidates = [
        run_dir / "training_history.csv",
        run_dir / "history.csv",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return []

    import csv

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                row2 = dict(row)
                row2["epoch"] = int(float(row2.get("epoch", "0")))
                key = "val_RMSE_weighted"
                if key in row2:
                    row2[key] = float(row2[key])
                rows.append(row2)
            except (TypeError, ValueError):
                continue
    return rows


class B8CheckpointingTrainer:
    """Canonical TrainingFramework trainer + immutable B8 RMSE checkpoints."""

    def __init__(self, framework: ModuleType, cfg: Any, backbone: Any):
        base_cls = framework.Trainer

        class _Impl(base_cls):
            def __init__(self, inner_cfg, inner_backbone):
                self._b8_last_val_metrics: Optional[Dict[str, Any]] = None
                self.b8_best_rmse: float = float("inf")
                self.b8_best_epoch: Optional[int] = None
                self.b8_best_checkpoint: Optional[Path] = None
                super().__init__(inner_cfg, inner_backbone)

            def _evaluate(self, loader: Any, max_batches: Optional[int] = None):
                val_loss, val_metrics = super()._evaluate(
                    loader, max_batches=max_batches
                )
                self._b8_last_val_metrics = val_metrics
                return val_loss, val_metrics

            def _save_checkpoint(
                self,
                name: str,
                epoch: int,
                val_loss: float,
                *,
                epoch_step: Optional[int] = None,
                checkpoint_kind: str = "epoch",
            ) -> Path:
                path = super()._save_checkpoint(
                    name,
                    epoch,
                    val_loss,
                    epoch_step=epoch_step,
                    checkpoint_kind=checkpoint_kind,
                )

                if name == "latest_model.pt":
                    archive = path.parent / f"epoch_{int(epoch):03d}_latest_model.pt"
                    shutil.copy2(path, archive)

                    rmse = float("inf")
                    try:
                        rmse = float(
                            self._b8_last_val_metrics["overall"][
                                REFERENCE_B6_METRIC_NAME
                            ]
                        )
                    except (KeyError, TypeError, ValueError, AttributeError):
                        pass

                    if math.isfinite(rmse) and rmse < self.b8_best_rmse:
                        self.b8_best_rmse = rmse
                        self.b8_best_epoch = int(epoch)
                        self.b8_best_checkpoint = archive
                        shutil.copy2(archive, path.parent / "best_rmse_model.pt")
                        atomic_json(
                            path.parent / "best_rmse_selection.json",
                            {
                                "metric": REFERENCE_B6_METRIC_NAME,
                                "best_rmse_c": rmse,
                                "epoch": int(epoch),
                                "checkpoint": str(archive),
                                "source": "B8_checkpointing_trainer",
                                "reference_b6_rmse_c": REFERENCE_B6_RMSE_WEIGHTED,
                            },
                        )
                return path

        self.impl = _Impl(cfg, backbone)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.impl, name)


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
        "[CONFIG] B8 fresh start | "
        f"epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LEARNING_RATE:.3g} | "
        f"wd={WEIGHT_DECAY:.3g} | early stopping={'DISABLED' if PATIENCE == 0 else 'ENABLED'}",
        flush=True,
    )
    return cfg


def run_training(
    framework: ModuleType,
    runtime: Dict[str, Path],
    dataset_root: Path,
) -> Dict[str, Any]:
    cfg = build_framework_config(framework, runtime, dataset_root)
    adapter = framework.BackboneAdapter(cfg)

    if hasattr(B6_HOST_MODULE, "validate_backbone_runtime"):
        B6_HOST_MODULE.validate_backbone_runtime(adapter)
    if hasattr(B6_HOST_MODULE, "validate_runtime_normalization"):
        norm_report = B6_HOST_MODULE.validate_runtime_normalization(adapter)
    else:
        norm_report = None

    print(
        "[PREFLIGHT] SKIPPED — B8 train mode goes directly to canonical training; "
        "runtime/checkpoint contract validation remains active.",
        flush=True,
    )

    trainer = B8CheckpointingTrainer(framework, cfg, adapter)
    model = trainer.model

    if getattr(model, "model_id", None) != MODEL_ID:
        raise RuntimeError(
            f"Framework instantiated unexpected model identity: {getattr(model, 'model_id', None)!r}"
        )

    # Verify that the new path is actually trainable and the B6 provenance copy
    # is frozen.
    trainable_b6 = [
        name for name, p in model.b6.named_parameters() if p.requires_grad
    ]
    if trainable_b6:
        raise RuntimeError(
            "B8 B6 provenance copy is not frozen: " + ", ".join(trainable_b6[:10])
        )

    trainable = parameter_summary(model)
    print(
        f"[MODEL] B8 parameters total={trainable['total']:,} | "
        f"trainable={trainable['trainable']:,} | frozen reference={trainable['frozen']:,}",
        flush=True,
    )
    print(
        "[MODEL] Architecture = masked CNN + multi-scale dilated CNN + "
        "window attention + ConvLSTM×2 + U-Net skip decoder + depth-conditioned residual head",
        flush=True,
    )

    run_dir = Path(trainer.run_dir)
    start_spec = {
        "status": "STARTED",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "bundle_fingerprint": EXPECTED_BUNDLE_FP,
        "reference": {
            "model_id": "CONVLSTM_MASKED_B6",
            "epoch": REFERENCE_B6_EPOCH,
            "rmse_weighted_c": REFERENCE_B6_RMSE_WEIGHTED,
            "val_loss": REFERENCE_B6_VAL_LOSS,
        },
        "architecture": model.model_summary(),
        "parameters": trainable,
        "normalization_guard": norm_report,
        "preflight": "SKIPPED_IN_TRAIN_MODE",
        "warm_start_checkpoint": model._warm_start_checkpoint,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(run_dir / "b8_msaa_convlstm_experiment_spec.json", start_spec)

    print("")
    print("=" * 80)
    print("STARTING B8 — MSAA-ConvLSTM")
    print("=" * 80)
    print(
        f"Warm start: B6 epoch {REFERENCE_B6_EPOCH} | "
        f"Reference RMSE_weighted: {REFERENCE_B6_RMSE_WEIGHTED:.9f} C"
    )
    print(
        f"Epochs={EPOCHS} | batch={BATCH_SIZE} | LR={LEARNING_RATE:.3g} | "
        f"workers={NUM_WORKERS} | AMP={AMP_ENABLED}"
    )
    print("No B7 checkpoint resume. Fresh B8 optimizer/scheduler state.")
    print("=" * 80)

    training_summary = trainer.fit()
    training_summary = dict(training_summary) if isinstance(training_summary, dict) else {
        "raw_result": training_summary
    }

    history = _read_training_history(run_dir)
    finite_rows = []
    for row in history:
        try:
            rmse = float(row["val_RMSE_weighted"])
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(rmse):
            finite_rows.append((rmse, epoch))

    if not finite_rows:
        raise RuntimeError(
            f"No finite val_RMSE_weighted values found in {run_dir / 'training_history.csv'}"
        )

    best_rmse, best_epoch = min(finite_rows, key=lambda pair: pair[0])
    improvement_c = REFERENCE_B6_RMSE_WEIGHTED - best_rmse
    verdict = (
        "SUPPORTED"
        if improvement_c >= IMPROVEMENT_THRESHOLD_C
        else "NOT_SUPPORTED"
    )

    selection = {
        "metric": REFERENCE_B6_METRIC_NAME,
        "reference_b6_rmse_c": REFERENCE_B6_RMSE_WEIGHTED,
        "best_b8_rmse_c": float(best_rmse),
        "best_epoch": int(best_epoch),
        "improvement_c": float(improvement_c),
        "threshold_c": IMPROVEMENT_THRESHOLD_C,
        "verdict": verdict,
        "best_checkpoint": (
            None
            if trainer.b8_best_checkpoint is None
            else str(trainer.b8_best_checkpoint)
        ),
        "history": [
            {"epoch": e, "val_RMSE_weighted": r}
            for r, e in sorted(finite_rows, key=lambda pair: pair[1])
        ],
    }
    atomic_json(run_dir / "best_rmse_selection_overall.json", selection)

    final = {
        **start_spec,
        "status": "COMPLETED",
        "run_dir": str(run_dir),
        "training_summary": training_summary,
        "best_rmse_selection": selection,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(run_dir / "b8_final_summary.json", final)

    print("")
    print("=" * 80)
    print("B8 TRAINING COMPLETE")
    print("=" * 80)
    print(
        f"Best validation RMSE_weighted: {best_rmse:.9f} C "
        f"(epoch {best_epoch})"
    )
    print(
        f"B6 reference: {REFERENCE_B6_RMSE_WEIGHTED:.9f} C | "
        f"delta={improvement_c:+.6f} C"
    )
    print(f"Verdict by pre-registered threshold: {verdict}")
    if trainer.b8_best_checkpoint is not None:
        print(f"Best RMSE checkpoint: {trainer.b8_best_checkpoint}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)

    return final


# ============================================================================
# Optional one-batch self-test (explicitly invoked only with RUN_MODE=self-test)
# ============================================================================


def run_self_test() -> Dict[str, Any]:
    set_seed(SEED)
    b6, runtime = load_b6_runtime()
    global B6_HOST_MODULE
    B6_HOST_MODULE = b6
    dataset_root = resolve_dataset_root(b6)

    framework = import_module_from_path(
        runtime["framework_path"],
        "oceanembed_training_framework_b8",
    )
    cfg = framework.TrainConfig(
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
    cfg.validate()
    adapter = framework.BackboneAdapter(cfg)
    model = Model(adapter.contract)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = framework.move_batch_to_device(next(iter(adapter.train_loader)), device)

    model.eval()
    with torch.no_grad():
        pred = framework.forward_prediction(model, batch)
        loss = framework.masked_mse_loss(pred, batch["y"], batch["loss_mask"])

    expected = (1, N_DEPTH, GRID_H, GRID_W)
    if tuple(pred.shape) != expected:
        raise RuntimeError(f"Prediction shape {tuple(pred.shape)} != {expected}")
    if not torch.isfinite(pred).all() or not torch.isfinite(loss):
        raise RuntimeError("Self-test produced non-finite output/loss")

    # Gradient-flow sanity: only one tiny batch, no optimizer step.
    model.train()
    model.zero_grad(set_to_none=True)
    pred2 = framework.forward_prediction(model, batch)
    loss2 = framework.masked_mse_loss(pred2, batch["y"], batch["loss_mask"])
    loss2.backward()
    grad_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
    ]
    if not grad_names:
        raise RuntimeError("No finite gradients reached trainable B8 parameters")

    report = {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "reference_b6_rmse_c": REFERENCE_B6_RMSE_WEIGHTED,
        "checkpoint": model._warm_start_checkpoint,
        "prediction_shape": list(pred.shape),
        "prediction_finite": True,
        "loss_finite": True,
        "trainable_gradient_count": len(grad_names),
        "parameters": parameter_summary(model),
        "model_summary": model.model_summary(),
    }
    print(json.dumps(report, indent=2, default=str))
    return report


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    set_seed(SEED)
    print("=" * 80)
    print("OceanEmbed B8 — MSAA-ConvLSTM")
    print("Multi-Scale Spatial Attention + ConvLSTM + Depth-Aware Reconstruction")
    print("=" * 80)
    print(
        f"Reference B6 {REFERENCE_B6_METRIC_NAME}: "
        f"{REFERENCE_B6_RMSE_WEIGHTED:.9f} C (epoch {REFERENCE_B6_EPOCH})"
    )
    print(f"Bundle fingerprint: {EXPECTED_BUNDLE_FP}")
    print("=" * 80)

    global B6_HOST_MODULE
    b6, runtime = load_b6_runtime()
    B6_HOST_MODULE = b6

    dataset_root = resolve_dataset_root(b6)
    print(f"[DATA] dataset root resolved: {dataset_root}", flush=True)

    checkpoint = discover_checkpoint(b6)
    print(
        f"[B6] selected exact epoch-{REFERENCE_B6_EPOCH} warm-start checkpoint: {checkpoint}",
        flush=True,
    )
    payload = _load_checkpoint_payload(checkpoint)
    if payload is None:
        raise RuntimeError(f"Could not load selected B6 checkpoint: {checkpoint}")
    _validate_exact_epoch10_checkpoint_payload(payload)

    framework = import_module_from_path(
        runtime["framework_path"],
        "oceanembed_training_framework_b8",
    )

    if RUN_MODE == "self-test":
        return 0 if run_self_test() else 1
    if RUN_MODE != "train":
        raise ValueError(f"Unsupported RUN_MODE={RUN_MODE!r}")

    result = run_training(framework, runtime, dataset_root)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
