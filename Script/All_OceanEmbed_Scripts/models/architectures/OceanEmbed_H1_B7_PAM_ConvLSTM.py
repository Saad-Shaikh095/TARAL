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

# Exact B6 comparison reference.
REFERENCE_B6_RMSE = 0.706064451
REFERENCE_B6_VAL_LOSS = 0.9249299274
EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)

EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_WORKERS = 4
GRAD_CLIP_NORM = 1.0
SEED = 42
LOG_EVERY = 10
AMP_ENABLED = True

WORKING_DIR = "/kaggle/working"
RUN_ROOT = "/kaggle/working/OceanEmbed_Runs"

# Set explicitly if automatic discovery finds the wrong file.
B6_TRAINER_PATH: Optional[str] = None
B6_CHECKPOINT_PATH: Optional[str] = None
DATASET_ROOT: Optional[str] = None

# Number of real batches used by the explicit preflight diagnostics.
PREFLIGHT_BATCHES = 1

# Number of days used by the first calibration pass.
# All 14 days are used when a real batch is available.
CALIBRATION_EPS = 1e-6


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


def discover_checkpoint(b6: ModuleType) -> Path:
    if B6_CHECKPOINT_PATH:
        p = Path(B6_CHECKPOINT_PATH).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"B6_CHECKPOINT_PATH does not exist: {p}")
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

    # Validate candidates against a fresh B6 model before accepting one.
    # The B6 model constructor only validates the input contract; checkpoint
    # identity is validated separately below.
    dummy_contract = None
    if DATASET_ROOT:
        # Runtime adapter will provide the authoritative contract later.
        dummy_contract = None

    valid: List[Path] = []
    for p in sorted(set(candidates), key=lambda x: (len(str(x)), str(x))):
        try:
            payload = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("backbone_fingerprint") != EXPECTED_BUNDLE_FP:
            continue
        cfg = payload.get("config")
        if not isinstance(cfg, dict):
            continue
        if str(cfg.get("experiment_name", "")) != "CONVLSTM_MASKED_B6":
            continue
        valid.append(p)

    if not valid:
        raise RuntimeError(
            "latest_model.pt files were found, but none passed the pinned B6 "
            "fingerprint + experiment identity check."
        )

    return valid[0]


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

        self.b6 = b6.Model(contract).cpu()

        checkpoint = discover_checkpoint(b6)
        print(f"[B6] validating checkpoint: {checkpoint}", flush=True)
        metadata = b6._validate_resume_checkpoint_metadata(
            checkpoint,
            self.b6,
        )

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

        partial_conv_cls = type(self.b6.highres_encoder)

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
        seq, seq_mask, ocean_mask, _ = B6_HOST_MODULE.prepare_model_inputs(batch)
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
            "reference_rmse_c": REFERENCE_B6_RMSE,
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
    global B6_HOST_MODULE
    if B6_HOST_MODULE is None:
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
    mixed_found = False
    mixed_info: Dict[str, Any] = {}

    ocean_bool = ocean_mask[0, 0].bool()
    candidate_cells = torch.nonzero(ocean_bool, as_tuple=False)

    for cell in candidate_cells[:5000]:
        py, px = [int(v) for v in cell.tolist()]
        day = 0
        branch_valid = {}
        for name, indices in CHANNEL_GROUPS.items():
            valid = seq_mask[
                0,
                day,
                list(indices),
                py,
                px,
            ].any()
            branch_valid[name] = bool(valid.item())

        vals = list(branch_valid.values())
        if any(vals) and not all(vals):
            mixed_found = True
            mixed_info = {
                "type": "real_mixed_validity",
                "day": day,
                "y": py,
                "x": px,
                "branch_valid": branch_valid,
            }
            break

    if not mixed_found and candidate_cells.numel() > 0:
        py, px = [int(v) for v in candidate_cells[0].tolist()]
        day = 0

        # Controlled source-observability perturbation at a real ocean pixel:
        # force the salinity source invalid while retaining thermal validity.
        probe_batch = {
            k: (v.clone() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

        if "x_dynamic_valid" in probe_batch:
            # A center pixel can still receive valid support through a 3x3
            # partial-convolution neighborhood. Because the branch has TWO
            # 3x3 layers, invalidate a 7x7 neighborhood so the center has no
            # possible salinity support through either receptive field.
            y0, y1 = max(0, py - 3), min(GRID_H, py + 4)
            x0, x1 = max(0, px - 3), min(GRID_W, px + 4)

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

    raw_batch = next(iter(adapter.train_loader))
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

    run_dir = Path(trainer.run_dir)
    spec = {
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "scientific_role": "modality_specific_front_end_hypothesis_test",
        "reference": {
            "model_id": "CONVLSTM_MASKED_B6",
            "reference_rmse_c": REFERENCE_B6_RMSE,
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
    print(f"Reference B6 RMSE : {REFERENCE_B6_RMSE:.9f} C")
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
