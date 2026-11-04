#!/usr/bin/env python3
"""
OceanEmbed - Pointwise MLP Model V1.1

Scientific role
---------------
This is the first nonlinear learned model in the OceanEmbed experiment ladder.

Implementation revision 1.0.1 aligns runtime robustness with the validated
Pointwise Linear B1 implementation. The scientific MLP architecture and data
contract are unchanged.
It deliberately has NO spatial receptive field: every output pixel depends only
on the predictor values at the same grid cell on the target day, plus the
static/context values for that cell/day. The 14-day Backbone window is retained
for a common data contract, but this model intentionally consumes only its final
day so that the MLP and the exact linear reference differ primarily by
nonlinearity rather than temporal architecture.

The purpose is to isolate nonlinear pointwise mapping before introducing any
spatial architecture such as a U-Net. It is NOT a proof of a fundamental
"local-information limit" because it is one finite-capacity global MLP with
shared weights across the grid.

Inputs
------
* final day of the 14-day window: 24 dynamic channels
* 6 static GEBCO channels
* 4 spatial context channels (sin/cos latitude and longitude)
* 2 day-of-year context channels from the final input day

Total pointwise input width: 36 channels.
Output: [B, 15, 100, 240] normalized target anomalies/values, depending on the
frozen Backbone target mode.

Modes
-----
--self-test
    Synthetic model-only tests. Requires PyTorch only.

--test
    Real-data fail-fast integration test. It loads the frozen Backbone,
    constructs the REAL Training Framework Trainer, executes a capped real
    training epoch (5 train batches), capped validation (2 batches), capped
    test evaluation (2 batches), exercises AMP when CUDA is active, saves the
    real framework checkpoints/metrics, reloads the best checkpoint, verifies
    the checkpoint round-trip, and saves physical-unit sample artifacts.

--train
    Full training through the shared Training Framework.

Checkpoint policy
-----------------
The Training Framework owns the official best-checkpoint rule: lowest finite
validation loss. This wrapper adds resume protection by carrying forward the
previous best checkpoint before training continues. A run is never considered
successful if its best_model.pt is missing.

The model uses a zero-initialized final projection so the initial prediction is
exactly zero in normalized target space, i.e. the Backbone climatology/mean
reference. This makes the first real-batch loss directly interpretable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn


MODEL_VERSION = "1.0.1"
MODEL_CONTRACT = "POINTWISEMLPV1.0"
MODEL_ID = "POINTWISE_MLP_14DAY_V1"
N_DEPTH = 15
GRID_H = 100
GRID_W = 240
DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2
EXPECTED_WINDOW_DAYS = 14
DEPTHS_M = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0]
HIDDEN_1 = 64
HIDDEN_2 = 32

RETRYABLE_IO_ERRORS = (OSError,)


class PointwiseMLPError(RuntimeError):
    pass


class ContractError(PointwiseMLPError):
    pass


class IntegrationTestError(PointwiseMLPError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return sanitize_json(obj.item())
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return sanitize_json(obj.item())
        return sanitize_json(obj.detach().cpu().tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def atomic_json(path: Path, obj: Any, retries: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    clean = sanitize_json(obj)
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            tmp.write_text(
                json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, path)
            return
        except RETRYABLE_IO_ERRORS as exc:
            last_exc = exc
            if attempt + 1 == retries:
                break
            time.sleep(2**attempt)
    raise OSError(f"Atomic JSON write failed after {retries} attempts: {path}") from last_exc


def safe_copy(src: Path, dst: Path, retries: int = 4) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            return
        except RETRYABLE_IO_ERRORS as exc:
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt + 1 == retries:
                break
            time.sleep(2**attempt)
    raise OSError(f"Atomic copy failed after {retries} attempts: {src} -> {dst}") from last_exc


def contract_hash(contract: Dict[str, Any]) -> str:
    raw = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_contract(contract: Dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise ContractError("Backbone contract must be a dict")

    dynamic = contract.get("dynamic", {})
    static = contract.get("static", {})
    target = contract.get("target", {})
    context = contract.get("context", {})

    dshape = list(dynamic.get("shape", []))
    if len(dshape) != 5 or dshape[:2] != ["B", "T"]:
        raise ContractError(f"Dynamic shape must be ['B','T',24,100,240], got {dshape}")
    if dshape[2:] != [DYNAMIC_CHANNELS, GRID_H, GRID_W]:
        raise ContractError(f"Unexpected dynamic contract: {dshape}")
    if len(dynamic.get("features", [])) != DYNAMIC_CHANNELS:
        raise ContractError("Backbone dynamic feature list must contain exactly 24 features")

    if static.get("shape") != ["B", STATIC_CHANNELS, GRID_H, GRID_W]:
        raise ContractError(f"Unexpected static contract: {static.get('shape')}")
    if len(static.get("features", [])) != STATIC_CHANNELS:
        raise ContractError("Backbone static feature list must contain exactly 6 features")

    if target.get("shape") != ["B", N_DEPTH, GRID_H, GRID_W]:
        raise ContractError(f"Unexpected target contract: {target.get('shape')}")
    depths = np.asarray(target.get("depths_m", []), dtype=float)
    if depths.shape != (N_DEPTH,):
        raise ContractError(f"Backbone target must contain exactly 15 depths, got {depths.shape}")
    if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ContractError("Backbone target depths must be finite and strictly increasing")
    if not np.array_equal(depths, np.asarray(DEPTHS_M, dtype=float)):
        raise ContractError(
            f"Target depth vector does not match OceanEmbed 15-depth contract: {depths.tolist()}"
        )

    spatial = context.get("spatial")
    temporal = context.get("temporal")
    if spatial != ["B", SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W]:
        raise ContractError("This model requires 4 spatial context channels")
    if temporal != ["B", "T", TEMPORAL_CONTEXT_CHANNELS]:
        raise ContractError("This model requires 2 temporal context channels")


class Model(nn.Module):
    """Two-hidden-layer pointwise MLP implemented with 1x1 convolutions."""

    def __init__(self, contract: Dict[str, Any]):
        super().__init__()
        validate_contract(contract)
        self.contract = contract
        self.contract_hash = contract_hash(contract)
        self.window_days = EXPECTED_WINDOW_DAYS
        self.input_channels = (
            DYNAMIC_CHANNELS
            + STATIC_CHANNELS
            + SPATIAL_CONTEXT_CHANNELS
            + TEMPORAL_CONTEXT_CHANNELS
        )

        self.net = nn.Sequential(
            nn.Conv2d(self.input_channels, HIDDEN_1, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(HIDDEN_1, HIDDEN_2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(HIDDEN_2, N_DEPTH, kernel_size=1),
        )
        # Preserve the exact zero-anomaly reference at initialization while
        # keeping a nonlinear hidden representation for subsequent updates.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _assemble_inputs(self, batch: Dict[str, Any]) -> torch.Tensor:
        x_dynamic = batch.get("x_dynamic")
        x_static = batch.get("x_static")
        if not torch.is_tensor(x_dynamic) or x_dynamic.ndim != 5:
            raise ContractError("x_dynamic must be [B,14,24,H,W]")
        if not torch.is_tensor(x_static) or x_static.ndim != 4:
            raise ContractError("x_static must be [B,6,H,W]")

        b, t, c, h, w = x_dynamic.shape
        if (t, c, h, w) != (EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W):
            raise ContractError(
                f"x_dynamic must have [B,14,24,100,240], got {tuple(x_dynamic.shape)}"
            )
        if tuple(x_static.shape[1:]) != (STATIC_CHANNELS, GRID_H, GRID_W):
            raise ContractError(f"x_static shape invalid: {tuple(x_static.shape)}")

        pieces = [x_dynamic[:, -1], x_static]

        spatial = batch.get("context_spatial")
        if not torch.is_tensor(spatial):
            raise ContractError("context_spatial is required")
        if spatial.ndim == 3:
            if tuple(spatial.shape) != (SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W):
                raise ContractError(f"context_spatial shape invalid: {tuple(spatial.shape)}")
            spatial = spatial.unsqueeze(0).expand(b, -1, -1, -1)
        elif spatial.ndim == 4:
            if tuple(spatial.shape[1:]) != (SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W):
                raise ContractError(f"context_spatial shape invalid: {tuple(spatial.shape)}")
        else:
            raise ContractError(f"context_spatial must be [4,H,W] or [B,4,H,W], got {spatial.ndim}D")
        pieces.append(spatial)

        temporal = batch.get("context_temporal")
        if not torch.is_tensor(temporal) or temporal.ndim != 3:
            raise ContractError("context_temporal must be [B,14,2]")
        if tuple(temporal.shape) != (b, EXPECTED_WINDOW_DAYS, TEMPORAL_CONTEXT_CHANNELS):
            raise ContractError(f"context_temporal shape invalid: {tuple(temporal.shape)}")
        day_context = temporal[:, -1].unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, GRID_H, GRID_W
        )
        pieces.append(day_context)

        x = torch.cat(pieces, dim=1)
        if x.shape[1] != self.input_channels:
            raise ContractError(
                f"Assembled pointwise width {x.shape[1]} != {self.input_channels}"
            )
        return x

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        y = self.net(self._assemble_inputs(batch))
        expected = (batch["x_dynamic"].shape[0], N_DEPTH, GRID_H, GRID_W)
        if tuple(y.shape) != expected:
            raise ContractError(f"Model output shape {tuple(y.shape)} != {expected}")
        return y


def build_model(contract: Dict[str, Any]) -> nn.Module:
    return Model(contract)


def parameter_summary(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_contract() -> Dict[str, Any]:
    return {
        "backbone_version": "synthetic",
        "dynamic": {
            "shape": ["B", "T", DYNAMIC_CHANNELS, GRID_H, GRID_W],
            "features": [f"feature_{i:02d}" for i in range(DYNAMIC_CHANNELS)],
        },
        "static": {
            "shape": ["B", STATIC_CHANNELS, GRID_H, GRID_W],
            "features": [f"static_{i:02d}" for i in range(STATIC_CHANNELS)],
        },
        "target": {
            "shape": ["B", N_DEPTH, GRID_H, GRID_W],
            "depths_m": list(DEPTHS_M),
        },
        "context": {
            "spatial": ["B", SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W],
            "temporal": ["B", "T", TEMPORAL_CONTEXT_CHANNELS],
        },
    }


def synthetic_batch(batch_size: int = 2, seed: int = 42) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(seed)
    return {
        "x_dynamic": torch.randn(
            batch_size, EXPECTED_WINDOW_DAYS, DYNAMIC_CHANNELS, GRID_H, GRID_W, generator=g
        ),
        "x_static": torch.randn(
            batch_size, STATIC_CHANNELS, GRID_H, GRID_W, generator=g
        ),
        "context_spatial": torch.randn(SPATIAL_CONTEXT_CHANNELS, GRID_H, GRID_W, generator=g),
        "context_temporal": torch.randn(
            batch_size, EXPECTED_WINDOW_DAYS, TEMPORAL_CONTEXT_CHANNELS, generator=g
        ),
        "y": torch.randn(batch_size, N_DEPTH, GRID_H, GRID_W, generator=g),
        "loss_mask": torch.ones(
            batch_size, N_DEPTH, GRID_H, GRID_W, dtype=torch.bool
        ),
    }


def synthetic_self_test() -> Dict[str, Any]:
    seed_everything(42)
    model = build_model(synthetic_contract())
    batch = synthetic_batch()
    pred = model(batch)
    if tuple(pred.shape) != (2, N_DEPTH, GRID_H, GRID_W):
        raise IntegrationTestError(f"Synthetic output shape invalid: {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise IntegrationTestError("Synthetic prediction contains non-finite values")
    if not torch.equal(pred, torch.zeros_like(pred)):
        raise IntegrationTestError("Zero-initialized final projection did not produce zero output")

    expected_params = (
        (model.input_channels * HIDDEN_1 + HIDDEN_1)
        + (HIDDEN_1 * HIDDEN_2 + HIDDEN_2)
        + (HIDDEN_2 * N_DEPTH + N_DEPTH)
    )
    if parameter_summary(model)["total"] != expected_params:
        raise IntegrationTestError("Unexpected parameter count")

    target = batch["y"]
    mask = batch["loss_mask"].to(pred.dtype)
    loss = ((pred - target).square() * mask).sum() / mask.sum()
    if not torch.isfinite(loss):
        raise IntegrationTestError("Synthetic initial loss is non-finite")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise IntegrationTestError(f"Synthetic gradient is non-finite: {name}")
    opt.step()
    pred2 = model(batch)
    if not torch.isfinite(pred2).all():
        raise IntegrationTestError("Synthetic post-step prediction is non-finite")

    return {
        "status": "PASS",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "model_contract": MODEL_CONTRACT,
        "input_channels": model.input_channels,
        "parameters": parameter_summary(model),
        "output_shape": list(pred.shape),
        "initial_loss": float(loss.detach().cpu()),
    }


def _import_module(path: Path, prefix: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"{prefix}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _search_siblings(filename: str) -> Optional[Path]:
    """Search for a companion source file across supported repository layouts."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / filename,
        here.parent / "framework" / filename,
        here.parent / "data" / filename,
        here.parent.parent / "src" / "framework" / filename,
        here.parent.parent / "src" / "data" / filename,
    ]
    return next((c for c in candidates if c.is_file()), None)


def resolve_framework_path(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    found = _search_siblings("OceanEmbed_TrainingFramework_V1.py")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate OceanEmbed_TrainingFramework_V1.py. Use --framework-path."
    )


def resolve_backbone_path(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    found = _search_siblings("OceanEmbed_DL_DataBackbone_V1.py")
    if found:
        return found
    raise FileNotFoundError(
        "Could not locate OceanEmbed_DL_DataBackbone_V1.py. Use --backbone-path."
    )


def resolved_dataset_root(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_framework_config(args: argparse.Namespace, fw: ModuleType) -> Any:
    cfg = fw.TrainConfig(
        dataset_root=str(resolved_dataset_root(args.dataset_root)),
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
        resume=args.resume,
    )
    cfg.validate()
    return cfg


def validate_real_dependencies(
    args: argparse.Namespace, fw: ModuleType, backbone_path: Path
) -> None:
    required = ("BackboneAdapter", "Trainer", "forward_prediction", "masked_mse_loss")
    for name in required:
        if not hasattr(fw, name):
            raise IntegrationTestError(f"Training Framework missing required API: {name}")
    if not hasattr(fw, "FRAMEWORK_VERSION"):
        raise IntegrationTestError("Training Framework exposes no version")
    if getattr(fw, "FRAMEWORK_CONTRACT", None) != "TRAINV1.1":
        raise IntegrationTestError(
            f"This model requires Training Framework contract TRAINV1.1; got {getattr(fw, 'FRAMEWORK_CONTRACT', None)!r}"
        )
    if not hasattr(fw.Trainer, "fit"):
        raise IntegrationTestError("Training Framework Trainer exposes no fit() method")
    if not backbone_path.is_file():
        raise FileNotFoundError(backbone_path)
    root = resolved_dataset_root(args.dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")


def carry_forward_best_checkpoint(resume_path: Path, run_dir: Path) -> Optional[Path]:
    resume_path = resume_path.resolve()
    candidates = []
    if resume_path.name == "best_model.pt":
        candidates.append(resume_path)
    sibling_best = resume_path.parent / "best_model.pt"
    if sibling_best.is_file() and sibling_best.resolve() != resume_path:
        candidates.append(sibling_best)
    if not candidates:
        return None

    source = candidates[0]
    destination = run_dir / "best_model.pt"
    safe_copy(source, destination)

    old_metrics = source.parent / "best_validation_metrics.json"
    if old_metrics.is_file():
        safe_copy(old_metrics, run_dir / "best_validation_metrics.json")
    return destination


def select_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_numpy_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = np.asarray(arr)[np.asarray(mask, dtype=bool)]
    return v[np.isfinite(v)]


def assert_physical_range(
    name: str,
    arr: np.ndarray,
    mask: np.ndarray,
    lower_c: float = -2.0,
    upper_c: float = 40.0,
) -> Dict[str, Any]:
    values = masked_numpy_values(arr, mask)
    if values.size == 0:
        raise IntegrationTestError(f"{name}: no finite masked cells available for physical-range check")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < lower_c or maximum > upper_c:
        raise IntegrationTestError(
            f"{name}: masked physical temperatures outside [{lower_c}, {upper_c}] C: "
            f"min={minimum}, max={maximum}"
        )
    return {
        "min_C": minimum,
        "max_C": maximum,
        "lower_bound_C": lower_c,
        "upper_bound_C": upper_c,
        "finite_masked_cells": int(values.size),
    }


def verify_required_artifacts(run_dir: Path) -> Dict[str, str]:
    required = [
        "run_manifest.json",
        "training_history.csv",
        "training_summary.json",
        "best_model.pt",
        "latest_model.pt",
        "best_validation_metrics.json",
        "test_metrics.json",
        "metrics_by_depth.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise IntegrationTestError(
            f"Training Framework did not save required end-result artifacts: {missing}"
        )
    return {name: str(run_dir / name) for name in required}


def verify_checkpoint(
    fw: ModuleType,
    trainer: Any,
) -> Dict[str, Any]:
    path = trainer.run_dir / "best_model.pt"
    payload = torch.load(path, map_location=trainer.device)
    if payload.get("backbone_fingerprint") != trainer.backbone.backbone.fp:
        raise IntegrationTestError("Best checkpoint Backbone fingerprint mismatch")
    if [float(x) for x in payload.get("depths_m", [])] != [float(x) for x in trainer.backbone.depths_m]:
        raise IntegrationTestError("Best checkpoint depth vector mismatch")
    if "model_contract" in payload and payload.get("model_contract") != MODEL_CONTRACT:
        raise IntegrationTestError("Best checkpoint model contract mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise IntegrationTestError("Best checkpoint has no model_state_dict")

    fresh = build_model(trainer.backbone.contract).to(trainer.device)
    fresh.load_state_dict(state)
    for key, original in trainer.model.state_dict().items():
        restored = fresh.state_dict()[key]
        if not torch.equal(original.detach().cpu(), restored.detach().cpu()):
            raise IntegrationTestError(f"Checkpoint round-trip mismatch at parameter {key}")

    return {
        "path": str(path),
        "epoch": int(payload.get("epoch", -1)),
        "val_loss": float(payload.get("val_loss")),
        "backbone_fingerprint": payload.get("backbone_fingerprint"),
        "round_trip": "PASS",
    }


def physical_sample_from_test_batch(
    fw: ModuleType,
    trainer: Any,
) -> Dict[str, Any]:
    batch = next(iter(trainer.backbone.test_loader))
    moved = fw.move_batch_to_device(batch, trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        pred = fw.forward_prediction(trainer.model, moved)
    dates = fw.batch_dates(batch)
    pred_phys = fw.inverse_targets(trainer.backbone.norm, pred, dates)
    target_phys = fw.inverse_targets(trainer.backbone.norm, moved["y"], dates)
    mask = batch["loss_mask"].detach().cpu().numpy().astype(bool)

    pred_range = assert_physical_range("test_prediction", pred_phys, mask)
    target_range = assert_physical_range("test_target", target_phys, mask)

    output_dir = trainer.run_dir / "integration_physical_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prediction_physical_C.npy", pred_phys)
    np.save(output_dir / "target_physical_C.npy", target_phys)
    np.save(output_dir / "loss_mask.npy", mask)
    atomic_json(
        output_dir / "sample_metadata.json",
        {
            "target_date": dates,
            "input_dates": batch["input_dates"],
            "prediction_range": pred_range,
            "target_range": target_range,
            "saved_utc": utc_now(),
        },
    )
    return {
        "target_date": dates,
        "prediction_range": pred_range,
        "target_range": target_range,
        "prediction_npy": str(output_dir / "prediction_physical_C.npy"),
        "target_npy": str(output_dir / "target_physical_C.npy"),
        "mask_npy": str(output_dir / "loss_mask.npy"),
    }


def real_integration_test(args: argparse.Namespace) -> Dict[str, Any]:
    framework_path = resolve_framework_path(args.framework_path)
    backbone_path = resolve_backbone_path(args.backbone_path)
    fw = _import_module(framework_path, "oceanembed_framework")
    validate_real_dependencies(args, fw, backbone_path)

    seed_everything(args.seed)
    cfg = build_framework_config(args, fw)
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.epochs = 1
    cfg.validate()

    adapter = fw.BackboneAdapter(cfg)
    trainer = fw.Trainer(cfg, adapter)
    try:
        validate_contract(adapter.contract)
        if getattr(trainer, "amp_enabled", False) and trainer.device.type != "cuda":
            raise IntegrationTestError("Trainer reported AMP enabled without CUDA")

        initial_batch = next(iter(adapter.train_loader))
        initial_moved = fw.move_batch_to_device(initial_batch, trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            initial_pred = fw.forward_prediction(trainer.model, initial_moved)
            initial_loss = fw.masked_mse_loss(
                initial_pred,
                initial_moved["y"],
                initial_moved["loss_mask"],
            )
        if not torch.isfinite(initial_loss):
            raise IntegrationTestError("Initial real train-batch loss is non-finite")
        if not torch.equal(initial_pred, torch.zeros_like(initial_pred)):
            raise IntegrationTestError("Initial real prediction is not exact zero normalized reference")
        initial_loss_value = float(initial_loss.detach().cpu())
        if not 0.2 <= initial_loss_value <= 5.0:
            raise IntegrationTestError(
                f"Initial real normalized MSE {initial_loss_value:.6g} is outside expected smoke-test range [0.2, 5.0]"
            )

        # Exercise the actual Trainer epoch loop and actual result writers.
        training_summary = trainer.fit(max_train_batches=5, max_val_batches=2)
        test_metrics = trainer.evaluate_test(max_batches=2)

        artifacts = verify_required_artifacts(trainer.run_dir)
        checkpoint = verify_checkpoint(fw, trainer)
        physical_sample = physical_sample_from_test_batch(fw, trainer)

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "PASS",
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "framework_version": getattr(fw, "FRAMEWORK_VERSION", None),
            "framework_contract": getattr(fw, "FRAMEWORK_CONTRACT", None),
            "backbone_version": getattr(adapter.mod, "BACKBONE_VERSION", None),
            "backbone_fingerprint": adapter.backbone.fp,
            "dataset_root": str(dataset_root(args.dataset_root)),
            "framework_path": str(framework_path),
            "backbone_path": str(backbone_path),
            "device": str(trainer.device),
            "amp_enabled": bool(getattr(trainer, "amp_enabled", False)),
            "parameters": parameter_summary(trainer.model),
            "input_channels": int(trainer.model.input_channels),
            "model_input_contract": adapter.contract,
            "initial_train_batch": {
                "normalized_mse": initial_loss_value,
                "zero_prediction": True,
                "expected_range": [0.2, 5.0],
            },
            "trainer_smoke_limits": {
                "epochs": 1,
                "max_train_batches": 5,
                "max_val_batches": 2,
                "max_test_batches": 2,
            },
            "training_summary": training_summary,
            "test_metrics": test_metrics,
            "framework_artifacts": artifacts,
            "checkpoint": checkpoint,
            "physical_sample": physical_sample,
            "completed_utc": utc_now(),
        }
        atomic_json(output_dir / "model_integration_test_report.json", report)
        atomic_json(
            trainer.run_dir / "model_integration_test_summary.json",
            report,
        )
        print(f"Framework run directory: {trainer.run_dir}")
        return report
    finally:
        try:
            ld = getattr(adapter.backbone, "loader", None)
            if ld is not None:
                ld.close()
        except Exception:
            pass


def train_via_framework(args: argparse.Namespace) -> Dict[str, Any]:
    framework_path = resolve_framework_path(args.framework_path)
    fw = _import_module(framework_path, "oceanembed_framework")
    cfg = build_framework_config(args, fw)

    if cfg.resume:
        resume_path = Path(cfg.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        resume_path = None

    adapter = fw.BackboneAdapter(cfg)
    trainer = fw.Trainer(cfg, adapter)
    try:
        if resume_path is not None:
            carried_best = carry_forward_best_checkpoint(resume_path, trainer.run_dir)
            if carried_best is None:
                raise PointwiseMLPError(
                    "Resume was requested, but a previous best_model.pt could not be located "
                    "unambiguously. Refusing to start without a valid best-checkpoint carry-forward."
                )
            atomic_json(
                trainer.run_dir / "resume_checkpoint_policy.json",
                {
                    "status": "PASS",
                    "resume_source": str(resume_path),
                    "carried_forward_best": str(carried_best),
                    "created_utc": utc_now(),
                },
            )

        summary = trainer.fit()
        best_path = trainer.run_dir / "best_model.pt"
        if not best_path.is_file():
            raise PointwiseMLPError(
                "Training ended without best_model.pt. Final test evaluation is blocked."
            )
        metrics = trainer.evaluate_test()
        final = {
            "status": "PASS",
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "training_summary": summary,
            "test_metrics": metrics,
            "best_checkpoint": str(best_path),
            "completed_utc": utc_now(),
        }
        atomic_json(trainer.run_dir / "model_wrapper_final_summary.json", final)
        return final
    finally:
        try:
            ld = getattr(adapter.backbone, "loader", None)
            if ld is not None:
                ld.close()
        except Exception:
            pass


def classify_exception(exc: BaseException) -> Dict[str, Any]:
    msg = str(exc)
    lower = msg.lower()
    if isinstance(exc, KeyboardInterrupt):
        category = "INTERRUPTED"
        recovery = "Run stopped by user. Atomic checkpoints and reports already written remain intact."
    elif isinstance(exc, FileNotFoundError):
        category = "MISSING_FILE"
        recovery = "Verify dataset_root, prepared Backbone state, and sibling model/framework/backbone paths."
    elif isinstance(exc, PermissionError):
        category = "PERMISSION_ERROR"
        recovery = "Check filesystem permissions, file locks, and destination availability."
    elif isinstance(exc, OSError):
        category = "FILESYSTEM_OR_NETWORK_IO"
        recovery = (
            "A filesystem-level I/O failure occurred. If the dataset or run root is network-mounted, "
            "verify connectivity/mount state and retry. Atomic critical writes use bounded retries."
        )
    elif isinstance(exc, RuntimeError) and (
        "out of memory" in lower or ("cuda" in lower and "memory" in lower)
    ):
        category = "CUDA_MEMORY"
        recovery = "Reduce batch size/model memory or use a larger GPU. No automatic training retry is attempted."
    elif isinstance(exc, ImportError):
        category = "DEPENDENCY_IMPORT"
        recovery = "Verify the model, Training Framework, Backbone and Python dependencies."
    elif isinstance(exc, ContractError):
        category = "MODEL_CONTRACT"
        recovery = "Inspect the frozen Backbone contract and model input assumptions."
    elif hasattr(exc, "__class__") and exc.__class__.__name__ in {"ModelContractError", "RunStateError"}:
        category = "FRAMEWORK_CONTRACT_OR_STATE"
        recovery = "Inspect the framework error and verify the frozen Backbone state/checkpoint fingerprint."
    else:
        category = type(exc).__name__
        recovery = "Inspect the saved traceback and fix the reported invariant before restarting."

    return {
        "category": category,
        "exception_type": type(exc).__name__,
        "message": msg,
        "recovery": recovery,
        "created_utc": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OceanEmbed Pointwise MLP Model B2")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--train", action="store_true")

    p.add_argument(
        "--dataset-root",
        default=os.environ.get(
            "OCEANEMBED_DATASET_ROOT",
            str(Path(__file__).resolve().parents[2] / "datasets" / "OceanEmbed_ModelData_V1"),
        ),
    )
    p.add_argument("--backbone-path", default=None)
    p.add_argument("--framework-path", default=None)
    p.add_argument("--loader-path", default=None)
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "model_tests" / "PointwiseMLP_V1"),
    )
    p.add_argument(
        "--run-root",
        default=os.environ.get(
            "OCEANEMBED_RUN_ROOT",
            str(Path(__file__).resolve().parents[2] / "experiments"),
        ),
    )
    p.add_argument("--experiment-name", default="EXP002_PointwiseMLP_V1")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            result = synthetic_self_test()
            print("POINTWISE MLP MODEL SELF TEST: PASS")
            print(json.dumps(sanitize_json(result), indent=2))
            return 0
        if args.test:
            result = real_integration_test(args)
            print("POINTWISE MLP MODEL REAL INTEGRATION TEST: PASS")
            print(json.dumps(sanitize_json(result), indent=2))
            return 0
        if args.train:
            result = train_via_framework(args)
            print("POINTWISE MLP MODEL TRAINING: PASS")
            print(json.dumps(sanitize_json(result), indent=2))
            return 0
        raise ValueError("Select exactly one mode")
    except BaseException as exc:
        info = classify_exception(exc)
        output_dir = Path(args.output_dir).expanduser().resolve()
        if args.train and not args.test:
            output_dir = Path(args.run_root).expanduser().resolve() / args.experiment_name / "startup_or_wrapper_failure"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(output_dir / "failure_report.json", info)
            (output_dir / "failure_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        except Exception:
            pass
        print(f"FAIL [{info['category']}]: {info['message']}", file=sys.stderr, flush=True)
        print(f"Recovery: {info['recovery']}", file=sys.stderr, flush=True)
        print(f"Failure artifacts: {output_dir}", file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
