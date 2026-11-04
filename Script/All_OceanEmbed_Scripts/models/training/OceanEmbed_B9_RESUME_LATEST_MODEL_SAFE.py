# ============================================================
# OCEANEMBED B9 — PRODUCTION KAGGLE TRAINING CELL
# ============================================================
#
# B9 = Multi-Scale Spatiotemporal Profile Network
#
# Core hypothesis:
#   - preserve high-resolution surface SST information
#   - learn multi-scale spatial structure
#   - learn 14-day temporal dependence with temporal attention
#   - reconstruct the 15-depth vertical profile jointly
#
# TRAINING:
#   - from scratch
#   - canonical ModelData_V1
#   - canonical anomaly target
#   - no B6/B7/B8 checkpoint
#   - no physics loss yet
#   - no target regeneration
#
# SAFETY:
#   - NO recursive /kaggle/input scan
#   - bounded runtime discovery only
#   - exact dataset root
#   - real-data forward preflight
#   - real-data backward preflight
#   - only then expensive training
#
# ============================================================

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# SAFE IMPORTLIB STATE RESET
# ============================================================
#
# The previous B9 cell patched importlib.util globally.
# Re-running that cell wrapped the wrapper again and caused:
#
#   RecursionError: maximum recursion depth exceeded
#
# We do NOT monkey-patch importlib.
# Reloading importlib.util restores the standard-library helpers
# when this notebook cell is re-executed in the same kernel.
# ============================================================

import importlib
import importlib.util

importlib.reload(importlib.util)


# ============================================================
# SAFE PATHLIB STATE RESET
# ============================================================
#
# Earlier notebook cells monkey-patched pathlib.Path.mkdir
# ("safe_path_mkdir") to swallow Errno 30. Re-running those cells
# stacked the wrappers until one pointed at itself:
#
#   RecursionError: maximum recursion depth exceeded
#
# The real cause of the read-only errors is fixed in this script
# (writable manifests mirror), so the patches are no longer needed.
# If Path.mkdir is not the stdlib implementation, restore it.
# Runs once per cell execution and is a no-op on a clean kernel.
# ============================================================

import pathlib

if getattr(pathlib.Path.mkdir, "__module__", "pathlib") != "pathlib":

    def _stdlib_path_mkdir(
        self,
        mode=0o777,
        parents=False,
        exist_ok=False,
    ):
        # Same logic as CPython 3.12 pathlib.Path.mkdir
        try:
            os.mkdir(self, mode)
        except FileNotFoundError:
            if not parents or self.parent == self:
                raise
            self.parent.mkdir(parents=True, exist_ok=True)
            self.mkdir(mode, parents=False, exist_ok=exist_ok)
        except OSError:
            if not exist_ok or not self.is_dir():
                raise

    _stdlib_path_mkdir.__module__ = "pathlib"
    _stdlib_path_mkdir.__qualname__ = "Path.mkdir"

    pathlib.Path.mkdir = _stdlib_path_mkdir

    print(
        "[RESET] Restored pathlib.Path.mkdir to stdlib behavior "
        "(removed stacked safe_path_mkdir wrappers)",
        flush=True,
    )

# ============================================================
# 0. EXACT DATASET PATH
# ============================================================

DATA_ROOT = Path(
    "/kaggle/input/datasets/lalithiray/oceanembed-modeldata-v1"
)

if not DATA_ROOT.is_dir():
    raise FileNotFoundError(
        "\n[STOP] OceanEmbed ModelData_V1 not found.\n"
        f"Expected exact path:\n{DATA_ROOT}\n"
    )

print("=" * 110, flush=True)
print(
    "OCEANEMBED B9 — MULTI-SCALE SPATIOTEMPORAL PROFILE NETWORK",
    flush=True,
)
print("=" * 110, flush=True)

print(
    "\n[DATASET]",
    flush=True,
)

print(
    "DATA_ROOT (physical):",
    DATA_ROOT,
    flush=True,
)


# ============================================================
# 0b. CANONICAL DATASET VIEW
# ============================================================
#
# The physical Kaggle dataset is arranged as:
#
#   oceanembed-modeldata-v1/          (mount name; NOT canonical)
#       backbone_artifacts/
#       target/target/glorys/
#       Ccmp_Oscar_...
#       Duacs_Oisst_...
#       Oisss_GlobCurrent_...
#       GEBCO_...
#
# But the Backbone's Bundle enforces a stricter contract:
#
#   OceanEmbed_ModelData_V1/          (exact folder name)
#       predictors/
#       static/
#       target/GLORYS/
#       manifests/data_backbone_V1/
#
# AND preflight() checks:
#
#       bundle.root.name == "OceanEmbed_ModelData_V1"
#
# So we build a writable symlink view under /kaggle/working
# whose top-level directory is named EXACTLY
# "OceanEmbed_ModelData_V1".
#
# To maximise compatibility with whatever exact filenames the
# Bundle dataclass constructs, we expose each physical
# top-level entry at BOTH:
#
#   CANONICAL_ROOT/<original_name>
#   CANONICAL_ROOT/predictors/<original_name>     (predictor .nc)
#   CANONICAL_ROOT/static/<original_name>         (static .nc)
#
# plus a set of short-name aliases commonly used by the bundle:
#
#   predictors/duacs_oisst.nc
#   predictors/oisss_globcurrent.nc
#   predictors/ccmp_oscar.nc
#   static/gebco.nc
#
# /kaggle/input stays untouched.
#
# ============================================================

CANONICAL_NAME = "OceanEmbed_ModelData_V1"

CANONICAL_ROOT = Path(
    "/kaggle/working"
) / CANONICAL_NAME


# ------------------------------------------------------------
# Locate the physical directory that actually contains
# backbone_artifacts/, target/, etc.
# ------------------------------------------------------------

def locate_physical_model_root() -> Path:
    """
    Return the directory that physically holds the ModelData_V1
    payload (backbone_artifacts/, target/, predictor dirs, ...).
    """

    candidates = [DATA_ROOT]

    for child in sorted(
        DATA_ROOT.iterdir(),
        key=lambda p: str(p),
    ):

        if not child.is_dir():
            continue

        low = child.name.lower()

        if (
            "modeldata" in low
            or "oceanembed" in low
        ):
            candidates.append(child)

    for cand in candidates:

        has_marker = (
            (cand / "backbone_artifacts").is_dir()
            or (cand / "target").is_dir()
            or (cand / "predictors").is_dir()
        )

        if has_marker:
            return cand

    return DATA_ROOT


PHYSICAL_ROOT = locate_physical_model_root()

print(
    "DATA_ROOT (physical root):",
    PHYSICAL_ROOT,
    flush=True,
)


# ------------------------------------------------------------
# Detect whether the given directory is already canonical.
# ------------------------------------------------------------

def is_canonical_layout(
    root: Path,
) -> bool:
    """
    A directory is treated as already canonical when its name
    is EXACTLY OceanEmbed_ModelData_V1 and it exposes the
    Backbone-facing layout.
    """

    if root.name != CANONICAL_NAME:
        return False

    required = [
        root / "predictors",
        root / "static",
        root / "target" / "GLORYS",
        root / "manifests" / "data_backbone_V1",
    ]

    return all(p.is_dir() for p in required)


# ------------------------------------------------------------
# Keyword sets.
# ------------------------------------------------------------

PREDICTOR_KEYWORDS = (
    "ccmp",
    "oscar",
    "duacs",
    "oisst",
    "oisss",
    "globcurrent",
    "glob_current",
    "currents",
    "wind",
    "sla",
    "adt",
)

STATIC_KEYWORDS = (
    "gebco",
    "bathym",
    "static",
    "ocean_fraction",
    "land",
)

# Short-name alias map. Any physical file whose lowercased name
# contains the key will also be symlinked under the alias.
PREDICTOR_ALIASES = (
    ("duacs",       "duacs_oisst.nc"),
    ("oisst",       "duacs_oisst.nc"),
    ("oisss",       "oisss_globcurrent.nc"),
    ("globcurrent", "oisss_globcurrent.nc"),
    ("glob_current","oisss_globcurrent.nc"),
    ("ccmp",        "ccmp_oscar.nc"),
    ("oscar",       "ccmp_oscar.nc"),
)

STATIC_ALIASES = (
    ("gebco", "gebco.nc"),
)

# Directories in the canonical view that we create ourselves.
# Top-level physical entries with these names must NOT be
# symlinked at CANONICAL_ROOT top-level (would collide).
_CANONICAL_RESERVED = {
    "predictors",
    "static",
    "target",
    "manifests",
    "backbone_artifacts",
}


# ------------------------------------------------------------
# Symlink helpers.
# ------------------------------------------------------------

def _safe_symlink(
    src: Path,
    dst: Path,
):

    if dst.exists() or dst.is_symlink():

        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    os.symlink(
        src,
        dst,
    )


# ------------------------------------------------------------
# Writable mirror of a read-only directory.
#
# manifests/data_backbone_V1 is the Backbone's "state" dir. The
# Backbone WRITES into it (e.g. loader_reports/), but the physical
# source lives under read-only /kaggle/input. A plain symlink
# therefore fails with:
#   OSError: [Errno 30] Read-only file system
#
# Fix: real directories under /kaggle/working; small files are
# COPIED (so they are writable), files above the size limit are
# symlinked (read-only, avoids duplicating huge data).
# ------------------------------------------------------------

_MIRROR_COPY_LIMIT_BYTES = 256 * 1024 * 1024


def _mirror_writable_dir(
    src: Path,
    dst: Path,
    copy_limit: int = _MIRROR_COPY_LIMIT_BYTES,
) -> Tuple[int, int]:
    """
    Returns (files_copied, files_symlinked).
    """

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)

    dst.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0
    linked = 0

    for item in sorted(
        src.iterdir(),
        key=lambda p: p.name,
    ):

        target = dst / item.name

        if item.is_dir():

            c, l = _mirror_writable_dir(
                item,
                target,
                copy_limit,
            )

            copied += c
            linked += l

            continue

        try:
            size = item.stat().st_size
        except OSError:
            size = copy_limit + 1

        if size <= copy_limit:

            shutil.copy2(
                item,
                target,
            )

            try:
                os.chmod(
                    target,
                    0o644,
                )
            except OSError:
                pass

            copied += 1

        else:

            os.symlink(
                item,
                target,
            )

            linked += 1

    return copied, linked


# ------------------------------------------------------------
# Build the canonical symlink view.
# ------------------------------------------------------------

def build_canonical_data_root(
    physical_root: Path,
) -> Path:
    """
    Build a canonical symlinked view of ModelData_V1 under
    /kaggle/working whose folder name is exactly
    OceanEmbed_ModelData_V1.

    If physical_root is already canonical, it is returned as-is.
    """

    if is_canonical_layout(physical_root):

        print(
            "[CANONICAL] Dataset already canonical:",
            physical_root,
            flush=True,
        )

        return physical_root

    if CANONICAL_ROOT.exists():

        try:
            shutil.rmtree(CANONICAL_ROOT)
        except Exception:
            pass

    CANONICAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (CANONICAL_ROOT / "predictors").mkdir(
        parents=True, exist_ok=True,
    )

    (CANONICAL_ROOT / "static").mkdir(
        parents=True, exist_ok=True,
    )

    (CANONICAL_ROOT / "target").mkdir(
        parents=True, exist_ok=True,
    )

    (CANONICAL_ROOT / "manifests").mkdir(
        parents=True, exist_ok=True,
    )

    target_glorys_linked = False
    manifests_linked = False

    print(
        "\n[CANONICAL BUILD]",
        flush=True,
    )
    print(
        "  canonical root name:",
        CANONICAL_ROOT.name,
        flush=True,
    )

    for entry in sorted(
        physical_root.iterdir(),
        key=lambda p: p.name,
    ):

        name = entry.name
        low = name.lower()

        # --------------------------------------------------
        # target/target/glorys/  ->  target/GLORYS/
        # --------------------------------------------------
        if name == "target":

            glorys_candidates = [
                entry / "target" / "glorys",
                entry / "target" / "GLORYS",
                entry / "glorys",
                entry / "GLORYS",
            ]

            for gcand in glorys_candidates:

                if gcand.is_dir():

                    _safe_symlink(
                        gcand,
                        CANONICAL_ROOT
                        / "target"
                        / "GLORYS",
                    )

                    target_glorys_linked = True

                    print(
                        "  target/GLORYS <-",
                        gcand,
                        flush=True,
                    )

                    break

            continue

        # --------------------------------------------------
        # backbone_artifacts/  ->  manifests/data_backbone_V1/
        # --------------------------------------------------
        if (
            name == "backbone_artifacts"
            or "manifest" in low
            or "backbone" in low
        ):

            nested = entry / "data_backbone_V1"

            dst = (
                CANONICAL_ROOT
                / "manifests"
                / "data_backbone_V1"
            )

            mirror_src = (
                nested
                if nested.is_dir()
                else entry
            )

            n_copied, n_linked = _mirror_writable_dir(
                mirror_src,
                dst,
            )

            print(
                "  manifests/data_backbone_V1 (writable mirror) <-",
                mirror_src,
                f"[copied={n_copied}, symlinked={n_linked}]",
                flush=True,
            )

            manifests_linked = True

            continue

        # --------------------------------------------------
        # From here on, expose every physical top-level entry
        # at the canonical root as well (unless reserved).
        # --------------------------------------------------
        if name not in _CANONICAL_RESERVED:

            _safe_symlink(
                entry,
                CANONICAL_ROOT / name,
            )

            print(
                "  top-level/" + name + " <-",
                entry,
                flush=True,
            )

        # --------------------------------------------------
        # Static (.nc) file classification.
        # --------------------------------------------------
        if (
            entry.is_file()
            and any(k in low for k in STATIC_KEYWORDS)
        ):

            _safe_symlink(
                entry,
                CANONICAL_ROOT / "static" / name,
            )

            print(
                "  static/" + name + " <-",
                entry,
                flush=True,
            )

            for key, alias in STATIC_ALIASES:

                if key in low:

                    _safe_symlink(
                        entry,
                        CANONICAL_ROOT / "static" / alias,
                    )

                    print(
                        "  static/" + alias + " <-",
                        entry,
                        flush=True,
                    )

            continue

        # --------------------------------------------------
        # Predictor (.nc) file classification.
        # --------------------------------------------------
        if (
            entry.is_file()
            and any(k in low for k in PREDICTOR_KEYWORDS)
        ):

            _safe_symlink(
                entry,
                CANONICAL_ROOT / "predictors" / name,
            )

            print(
                "  predictors/" + name + " <-",
                entry,
                flush=True,
            )

            for key, alias in PREDICTOR_ALIASES:

                if key in low:

                    _safe_symlink(
                        entry,
                        CANONICAL_ROOT / "predictors" / alias,
                    )

                    print(
                        "  predictors/" + alias + " <-",
                        entry,
                        flush=True,
                    )

            continue

        # --------------------------------------------------
        # Any other file (e.g. docs): default into predictors/.
        # Any other directory: default into predictors/ as-is.
        # --------------------------------------------------
        _safe_symlink(
            entry,
            CANONICAL_ROOT / "predictors" / name,
        )

        print(
            "  predictors/" + name + " (fallback) <-",
            entry,
            flush=True,
        )

    if not target_glorys_linked:

        raise FileNotFoundError(
            "\n[STOP] Could not locate GLORYS target inside "
            f"physical root:\n{physical_root}\n"
            "Expected one of:\n"
            "  target/target/glorys/\n"
            "  target/glorys/\n"
        )

    if not manifests_linked:

        print(
            "[WARN] No backbone_artifacts / manifests dir found "
            "under physical root. Canonical view has empty "
            "manifests/data_backbone_V1/.",
            flush=True,
        )

    print(
        "\n[CANONICAL] Root:",
        CANONICAL_ROOT,
        flush=True,
    )

    return CANONICAL_ROOT


CANONICAL_DATA_ROOT = build_canonical_data_root(
    PHYSICAL_ROOT
)

print(
    "DATA_ROOT (canonical):",
    CANONICAL_DATA_ROOT,
    flush=True,
)


# ============================================================
# 1. RUNTIME SOURCE DISCOVERY
# ============================================================
#
# IMPORTANT:
# We deliberately DO NOT use rglob().
#
# We check only a bounded set of candidate directories.
# ============================================================

RUNTIME_FILENAMES = {
    "framework":
        "OceanEmbed_TrainingFramework_V1.py",

    "backbone":
        "OceanEmbed_DL_DataBackbone_V1.py",

    "loader":
        "OceanEmbed_Synchronized_Data_Loader_V1_2.py",
}


EXPLICIT_RUNTIME_DIRS = [
    Path(
        "/kaggle/input/datasets/lalithiray/"
        "oceanembed-input-python-files"
    ),
    Path(
        "/kaggle/input/oceanembed-input-python-files"
    ),
]


def exact_runtime_candidates():
    roots = []

    for ep in EXPLICIT_RUNTIME_DIRS:

        if ep.is_dir():

            roots.append(ep)

            roots.extend(
                [
                    ep / "src",
                    ep / "src" / "framework",
                    ep / "src" / "data",
                ]
            )

    kaggle_input = Path("/kaggle/input")

    if kaggle_input.exists():

        for dataset_dir in sorted(
            kaggle_input.iterdir(),
            key=lambda p: str(p),
        ):

            if not dataset_dir.is_dir():
                continue

            roots.extend(
                [
                    dataset_dir,
                    dataset_dir / "src",
                    dataset_dir / "src" / "framework",
                    dataset_dir / "src" / "data",
                ]
            )

            if dataset_dir.name == "datasets":

                for user_dir in sorted(
                    dataset_dir.iterdir(),
                    key=lambda p: str(p),
                ):

                    if not user_dir.is_dir():
                        continue

                    for ds_dir in sorted(
                        user_dir.iterdir(),
                        key=lambda p: str(p),
                    ):

                        if not ds_dir.is_dir():
                            continue

                        roots.extend(
                            [
                                ds_dir,
                                ds_dir / "src",
                                ds_dir / "src" / "framework",
                                ds_dir / "src" / "data",
                            ]
                        )

    kaggle_working = Path("/kaggle/working")

    if kaggle_working.exists():

        roots.extend(
            [
                kaggle_working,
                kaggle_working / "src",
                kaggle_working / "src" / "framework",
                kaggle_working / "src" / "data",
            ]
        )

    out = []
    seen = set()

    for r in roots:

        try:
            r = r.resolve()
        except Exception:
            continue

        if r in seen:
            continue

        seen.add(r)

        if r.exists() and r.is_dir():
            out.append(r)

    return out


def locate_runtime():
    roots = exact_runtime_candidates()

    framework_path = None
    backbone_path = None
    loader_path = None

    for root in roots:

        p = root / RUNTIME_FILENAMES["framework"]

        if framework_path is None and p.is_file():
            framework_path = p

        p = root / RUNTIME_FILENAMES["backbone"]

        if backbone_path is None and p.is_file():
            backbone_path = p

        p = root / RUNTIME_FILENAMES["loader"]

        if loader_path is None and p.is_file():
            loader_path = p

    return (
        framework_path,
        backbone_path,
        loader_path,
    )


FRAMEWORK_PATH, BACKBONE_PATH, LOADER_PATH = (
    locate_runtime()
)


print(
    "\n[RUNTIME CHECK]",
    flush=True,
)

print(
    "Framework:",
    FRAMEWORK_PATH
    if FRAMEWORK_PATH
    else "MISSING",
    flush=True,
)

print(
    "Backbone :",
    BACKBONE_PATH
    if BACKBONE_PATH
    else "MISSING",
    flush=True,
)

print(
    "Loader   :",
    LOADER_PATH
    if LOADER_PATH
    else "MISSING",
    flush=True,
)


missing_runtime = []

if FRAMEWORK_PATH is None:
    missing_runtime.append(
        "OceanEmbed_TrainingFramework_V1.py"
    )

if BACKBONE_PATH is None:
    missing_runtime.append(
        "OceanEmbed_DL_DataBackbone_V1.py"
    )

if LOADER_PATH is None:
    missing_runtime.append(
        "OceanEmbed_Synchronized_Data_Loader_V1_2.py"
    )


if missing_runtime:

    raise FileNotFoundError(
        "\n"
        + "=" * 90
        + "\n"
        "[STOP — RUNTIME DEPENDENCY MISSING]\n"
        + "=" * 90
        + "\n"
        "ModelData_V1 is present, but the existing OceanEmbed runtime "
        "source files are not mounted in this Kaggle session.\n\n"
        "Please attach the existing OceanEmbed runtime dataset/source "
        "containing:\n"
        + "\n".join(
            f"  - {x}"
            for x in missing_runtime
        )
        + "\n\n"
        "This check is bounded and finished immediately; "
        "the cell will NOT recursively scan /kaggle/input.\n"
    )


# ============================================================
# 2. IMPORT HELPERS
# ============================================================

def import_module_from_path(
    path: Path,
    module_name: str,
) -> ModuleType:

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not import module: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


framework = import_module_from_path(
    FRAMEWORK_PATH,
    "oceanembed_b9_framework",
)


# ============================================================
# 3. VERIFY FRAMEWORK CONTRACT
# ============================================================

EXPECTED_FRAMEWORK_CONTRACT = "TRAINV1.1"

actual_contract = getattr(
    framework,
    "FRAMEWORK_CONTRACT",
    None,
)

print(
    "\n[FRAMEWORK]",
    flush=True,
)

print(
    "Expected:",
    EXPECTED_FRAMEWORK_CONTRACT,
    flush=True,
)

print(
    "Found   :",
    actual_contract,
    flush=True,
)

if actual_contract != EXPECTED_FRAMEWORK_CONTRACT:

    raise RuntimeError(
        "OceanEmbed Training Framework contract mismatch:\n"
        f"Expected {EXPECTED_FRAMEWORK_CONTRACT}\n"
        f"Found    {actual_contract}"
    )


for required_api in [
    "TrainConfig",
    "BackboneAdapter",
    "Trainer",
    "forward_prediction",
    "masked_mse_loss",
    "move_batch_to_device",
]:

    if not hasattr(
        framework,
        required_api,
    ):

        raise RuntimeError(
            f"Training Framework missing required API: "
            f"{required_api}"
        )


# ============================================================
# 3b. FRAMEWORK BACKBONE-IMPORT FIX (sys.modules registration)
# ============================================================
#
# Problem:
#   BackboneAdapter._import_module() in the Training Framework loads the
#   DataBackbone file with module_from_spec() + exec_module() but never
#   registers the module in sys.modules. On Python 3.12, @dataclass
#   with string annotations then crashes:
#     AttributeError: 'NoneType' object has no attribute '__dict__'
#
# Fix (scoped, NOT a global importlib patch):
#   Replace only BackboneAdapter._import_module with a version that
#   registers the module before executing it. The replacement does not
#   call the old function, so re-running this cell never stacks wrappers
#   (no recursion risk). The framework module is re-imported fresh on
#   every run anyway.
# ============================================================

def _oceanembed_safe_import_module(path):

    path = Path(path)

    spec = importlib.util.spec_from_file_location(
        path.stem,
        str(path),
    )

    if spec is None or spec.loader is None:

        raise ImportError(
            f"Could not import Backbone: {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    previous = sys.modules.get(
        spec.name
    )

    sys.modules[spec.name] = module

    try:

        spec.loader.exec_module(
            module
        )

    except BaseException:

        if previous is not None:
            sys.modules[spec.name] = previous
        else:
            sys.modules.pop(
                spec.name,
                None,
            )

        raise

    return module


if not hasattr(
    framework.BackboneAdapter,
    "_import_module",
):

    raise RuntimeError(
        "Training Framework BackboneAdapter has no _import_module; "
        "framework layout changed, patch not applicable."
    )

framework.BackboneAdapter._import_module = staticmethod(
    _oceanembed_safe_import_module
)

print(
    "[PATCH] BackboneAdapter._import_module now registers "
    "modules in sys.modules",
    flush=True,
)


# ============================================================
# 4. B9 MODEL SOURCE
# ============================================================

MODEL_PATH = Path(
    "/kaggle/working/OceanEmbed_B9_MSSTP_ProfileNet.py"
)


B9_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


MODEL_ID = "B9_MSSTP_PROFILE_NET"
MODEL_VERSION = "9.1.0"
MODEL_CONTRACT = "MSSTPPROFILE9.1"

GRID_H = 100
GRID_W = 240

WINDOW_DAYS = 14

DYNAMIC_CHANNELS = 24
STATIC_CHANNELS = 6
SPATIAL_CONTEXT_CHANNELS = 4
TEMPORAL_CONTEXT_CHANNELS = 2

TOTAL_CHANNELS = 36
N_DEPTH = 15

DEPTHS_M = (
    0.0,
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
    125.0,
    150.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
)

OISST_SST_INDEX = 4
OISST_ANOM_INDEX = 5

PROFILE_MODES = 8

OCEAN_FRACTION_STATIC_INDEX = 2
OCEAN_FRACTION_TRAIN_MEAN = 0.494109282395264
OCEAN_FRACTION_TRAIN_STD = 0.49318098582328196

EXPECTED_LAND_OCEAN_FRACTION_Z = (
    (0.0 - OCEAN_FRACTION_TRAIN_MEAN)
    / OCEAN_FRACTION_TRAIN_STD
)

OCEAN_FRACTION_Z_TOL = 5e-5


def contract_hash(
    contract: Dict[str, Any]
) -> str:

    raw = json.dumps(
        contract,
        sort_keys=True,
        default=str,
    ).encode()

    return hashlib.sha256(raw).hexdigest()


def derive_ocean_mask(
    x_static: torch.Tensor,
    batch: Dict[str, Any],
) -> torch.Tensor:
    """
    Use the authoritative Backbone-provided physical_depth_mask.

    The 0 m physical-depth mask is the structural ocean support map.
    This avoids hard-coding normalization statistics for ocean_fraction.
    """
    explicit = batch.get("ocean_mask")

    if explicit is not None:

        if explicit.ndim == 2:
            explicit = explicit[None, None]

        elif explicit.ndim == 3:
            explicit = explicit[:, None]

        if explicit.ndim != 4:
            raise RuntimeError(
                "ocean_mask must be [H,W], [B,H,W], or [B,1,H,W]"
            )

        expected = (
            x_static.shape[0],
            1,
            GRID_H,
            GRID_W,
        )

        if tuple(explicit.shape) != expected:
            raise RuntimeError(
                "ocean_mask shape mismatch: "
                f"{tuple(explicit.shape)} != {expected}"
            )

        return (
            explicit > 0.5
        ).to(
            dtype=x_static.dtype,
            device=x_static.device,
        )

    physical = batch.get(
        "physical_depth_mask"
    )

    if physical is None:
        raise RuntimeError(
            "B9 requires Backbone-provided physical_depth_mask."
        )

    expected = (
        x_static.shape[0],
        N_DEPTH,
        GRID_H,
        GRID_W,
    )

    if tuple(physical.shape) != expected:
        raise RuntimeError(
            "physical_depth_mask shape mismatch: "
            f"{tuple(physical.shape)} != {expected}"
        )

    # Surface depth (0 m) provides structural ocean support.
    return (
        physical[:, 0:1].bool()
    ).to(
        dtype=x_static.dtype,
        device=x_static.device,
    )


class ResidualConvBlock(
    nn.Module
):

    def __init__(
        self,
        channels: int,
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        identity = x

        y = self.conv1(
            x
        )

        y = self.act(
            y
        )

        y = self.conv2(
            y
        )

        y = y + identity

        y = self.act(
            y
        )

        return y * mask


class Model(nn.Module):

    model_id = MODEL_ID
    model_version = MODEL_VERSION
    model_contract = MODEL_CONTRACT

    def __init__(
        self,
        contract: Dict[str, Any],
    ):

        super().__init__()

        self.contract = contract

        self.contract_hash = contract_hash(
            contract
        )

        self.enc1 = nn.Conv2d(
            TOTAL_CHANNELS + 1,
            48,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="replicate",
        )

        self.enc1_block = (
            ResidualConvBlock(48)
        )

        self.enc2 = nn.Conv2d(
            48,
            96,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="replicate",
        )

        self.enc2_block = (
            ResidualConvBlock(96)
        )

        temporal_layer = (
            nn.TransformerEncoderLayer(
                d_model=96,
                nhead=4,
                dim_feedforward=192,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.temporal_encoder = (
            nn.TransformerEncoder(
                temporal_layer,
                num_layers=2,
                enable_nested_tensor=False,
            )
        )

        self.temporal_pos = nn.Parameter(
            torch.zeros(
                WINDOW_DAYS,
                96,
            )
        )

        nn.init.normal_(
            self.temporal_pos,
            mean=0.0,
            std=0.02,
        )

        self.coarse_projection = (
            nn.Sequential(
                nn.Conv2d(
                    96,
                    64,
                    kernel_size=1,
                ),
                nn.GELU(),
            )
        )

        self.mid_skip = nn.Conv2d(
            TOTAL_CHANNELS + 1,
            48,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode="replicate",
        )

        self.mid_skip_block = (
            ResidualConvBlock(48)
        )

        self.high_skip = nn.Conv2d(
            TOTAL_CHANNELS + 1,
            32,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )

        self.high_skip_block = (
            ResidualConvBlock(32)
        )

        self.surface_branch = nn.Sequential(
            nn.Conv2d(
                2,
                16,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),

            nn.Conv2d(
                16,
                24,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )

        self.dec_mid = nn.Sequential(
            nn.Conv2d(
                64 + 48,
                64,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )

        self.dec_high = nn.Sequential(
            nn.Conv2d(
                64 + 32 + 24,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),

            nn.Conv2d(
                48,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
        )

        self.profile_coefficients = nn.Conv2d(
            48,
            PROFILE_MODES,
            kernel_size=1,
        )

        self.depth_basis = nn.Sequential(
            nn.Linear(
                1,
                32,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                PROFILE_MODES,
            ),
        )

        self.surface_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        self.surface_bias = nn.Parameter(
            torch.tensor(0.0)
        )

        self.surface_correction = (
            nn.Conv2d(
                48 + 24,
                1,
                kernel_size=1,
            )
        )

        nn.init.zeros_(
            self.surface_correction.weight
        )

        nn.init.zeros_(
            self.surface_correction.bias
        )

    def _prepare_day(
        self,
        batch: Dict[str, Any],
        t: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        xd = batch[
            "x_dynamic"
        ]

        xs = batch[
            "x_static"
        ]

        xdv = batch[
            "x_dynamic_valid"
        ].bool()

        xsv = batch[
            "x_static_valid"
        ].bool()

        b = xd.shape[0]

        dynamic = xd[
            :,
            t
        ]

        dynamic_valid = xdv[
            :,
            t
        ]

        spatial = batch[
            "context_spatial"
        ]

        if spatial.ndim == 3:

            spatial = (
                spatial
                .unsqueeze(0)
                .expand(
                    b,
                    -1,
                    -1,
                    -1,
                )
            )

        temporal = batch[
            "context_temporal"
        ][:, t]

        temporal_map = (
            temporal
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                GRID_H,
                GRID_W,
            )
        )

        spatial_valid = torch.ones(
            b,
            SPATIAL_CONTEXT_CHANNELS,
            GRID_H,
            GRID_W,
            dtype=torch.bool,
            device=xd.device,
        )

        temporal_valid = torch.ones(
            b,
            TEMPORAL_CONTEXT_CHANNELS,
            GRID_H,
            GRID_W,
            dtype=torch.bool,
            device=xd.device,
        )

        x = torch.cat(
            [
                dynamic,
                xs,
                spatial,
                temporal_map,
            ],
            dim=1,
        )

        valid = torch.cat(
            [
                dynamic_valid,
                xsv,
                spatial_valid,
                temporal_valid,
            ],
            dim=1,
        )

        ocean_mask = (
            derive_ocean_mask(
                xs,
                batch,
            ).bool()
        )

        valid = (
            valid
            & ocean_mask
        )

        x = torch.where(
            valid,
            x,
            torch.zeros_like(x),
        )

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        support = (
            valid.float()
            .mean(
                dim=1,
                keepdim=True,
            )
        )

        return (
            x,
            support,
            ocean_mask,
        )

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> torch.Tensor:

        xd = batch[
            "x_dynamic"
        ]

        b, t = xd.shape[:2]

        if t != WINDOW_DAYS:

            raise RuntimeError(
                f"Expected {WINDOW_DAYS} input days, got {t}"
            )

        if tuple(
            xd.shape[2:]
        ) != (
            DYNAMIC_CHANNELS,
            GRID_H,
            GRID_W,
        ):

            raise RuntimeError(
                "Unexpected x_dynamic shape: "
                f"{tuple(xd.shape)}"
            )

        temporal_states = []

        final_x37 = None
        final_ocean_mask = None

        final_mid_input = None

        for day in range(
            WINDOW_DAYS
        ):

            x, support, ocean_mask = (
                self._prepare_day(
                    batch,
                    day,
                )
            )

            x37 = torch.cat(
                [
                    x,
                    support,
                ],
                dim=1,
            )

            mid = self.enc1(
                x37
            )

            mid_mask = F.interpolate(
                ocean_mask.float(),
                size=mid.shape[-2:],
                mode="nearest",
            )

            mid = mid * mid_mask

            mid = self.enc1_block(
                mid,
                mid_mask,
            )

            low = self.enc2(
                mid
            )

            low_mask = F.interpolate(
                ocean_mask.float(),
                size=low.shape[-2:],
                mode="nearest",
            )

            low = low * low_mask

            low = self.enc2_block(
                low,
                low_mask,
            )

            temporal_states.append(
                low
            )

            if day == (
                WINDOW_DAYS - 1
            ):

                final_x37 = x37

                final_ocean_mask = (
                    ocean_mask
                )

                final_mid_input = mid

        seq = torch.stack(
            temporal_states,
            dim=1,
        )

        _, _, c, h, w = (
            seq.shape
        )

        tokens = (
            seq.permute(
                0,
                3,
                4,
                1,
                2,
            )
            .contiguous()
            .view(
                b * h * w,
                t,
                c,
            )
        )

        tokens = (
            tokens
            + self.temporal_pos[
                None,
                :t,
                :,
            ]
        )

        attended = (
            self.temporal_encoder(
                tokens
            )
        )

        temporal_state = (
            attended[:, -1]
            .view(
                b,
                h,
                w,
                c,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .contiguous()
        )

        temporal_state = (
            self.coarse_projection(
                temporal_state
            )
        )

        coarse_mask = F.interpolate(
            final_ocean_mask.float(),
            size=temporal_state.shape[-2:],
            mode="nearest",
        )

        temporal_state = (
            temporal_state
            * coarse_mask
        )

        mid_skip = self.mid_skip(
            final_x37
        )

        mid_mask = F.interpolate(
            final_ocean_mask.float(),
            size=mid_skip.shape[-2:],
            mode="nearest",
        )

        mid_skip = (
            mid_skip
            * mid_mask
        )

        mid_skip = self.mid_skip_block(
            mid_skip,
            mid_mask,
        )

        high_skip = self.high_skip(
            final_x37
        )

        high_mask = (
            final_ocean_mask.float()
        )

        high_skip = (
            high_skip
            * high_mask
        )

        high_skip = (
            self.high_skip_block(
                high_skip,
                high_mask,
            )
        )

        raw_sst = batch[
            "x_dynamic"
        ][:,
          -1,
          OISST_SST_INDEX:
          OISST_SST_INDEX + 1
        ]

        raw_anom = batch[
            "x_dynamic"
        ][:,
          -1,
          OISST_ANOM_INDEX:
          OISST_ANOM_INDEX + 1
        ]

        sst_valid = batch[
            "x_dynamic_valid"
        ][:,
          -1,
          OISST_SST_INDEX:
          OISST_SST_INDEX + 1
        ].bool()

        anom_valid = batch[
            "x_dynamic_valid"
        ][:,
          -1,
          OISST_ANOM_INDEX:
          OISST_ANOM_INDEX + 1
        ].bool()

        raw_sst = torch.where(
            sst_valid,
            raw_sst,
            torch.zeros_like(raw_sst),
        )

        raw_anom = torch.where(
            anom_valid,
            raw_anom,
            torch.zeros_like(raw_anom),
        )

        raw_sst = torch.nan_to_num(
            raw_sst,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        raw_anom = torch.nan_to_num(
            raw_anom,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        surface_input = torch.cat(
            [
                raw_sst,
                raw_anom,
            ],
            dim=1,
        )

        surface_features = (
            self.surface_branch(
                surface_input
            )
        )

        surface_features = (
            surface_features
            * high_mask
        )

        up_mid = F.interpolate(
            temporal_state,
            size=mid_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        up_mid = (
            up_mid
            * mid_mask
        )

        mid_fused = torch.cat(
            [
                up_mid,
                mid_skip,
            ],
            dim=1,
        )

        mid_fused = (
            self.dec_mid(
                mid_fused
            )
        )

        mid_fused = (
            mid_fused
            * mid_mask
        )

        up_high = F.interpolate(
            mid_fused,
            size=(GRID_H, GRID_W),
            mode="bilinear",
            align_corners=False,
        )

        up_high = (
            up_high
            * high_mask
        )

        high_fused = torch.cat(
            [
                up_high,
                high_skip,
                surface_features,
            ],
            dim=1,
        )

        high_fused = (
            self.dec_high(
                high_fused
            )
        )

        high_fused = (
            high_fused
            * high_mask
        )

        coefficients = (
            self.profile_coefficients(
                high_fused
            )
        )

        coefficients = (
            coefficients
            * high_mask
        )

        depths = torch.tensor(
            DEPTHS_M,
            dtype=high_fused.dtype,
            device=high_fused.device,
        )

        z = (
            torch.log1p(
                depths
            )
            / torch.log1p(
                torch.tensor(
                    1000.0,
                    dtype=high_fused.dtype,
                    device=high_fused.device,
                ),
            )
        )

        z = z.view(
            N_DEPTH,
            1,
        )

        basis = (
            self.depth_basis(
                z
            )
        )

        profile = torch.einsum(
            "bkhw,dk->bdhw",
            coefficients,
            basis,
        )

        surface_base = (
            self.surface_scale
            * raw_anom[:, 0]
            + self.surface_bias
        )

        surface_correction = (
            self.surface_correction(
                torch.cat(
                    [
                        high_fused,
                        surface_features,
                    ],
                    dim=1,
                )
            )[:, 0]
        )

        surface_prediction = (
            surface_base
            + surface_correction
        )

        profile = profile.clone()

        profile[:, 0] = (
            surface_prediction
        )

        profile = (
            profile
            * final_ocean_mask
        )

        expected = (
            b,
            N_DEPTH,
            GRID_H,
            GRID_W,
        )

        if tuple(
            profile.shape
        ) != expected:

            raise RuntimeError(
                "B9 output shape mismatch: "
                f"{tuple(profile.shape)} != {expected}"
            )

        if not torch.isfinite(
            profile
        ).all():

            raise FloatingPointError(
                "B9 produced non-finite output."
            )

        return profile

    def model_summary(
        self
    ) -> Dict[str, Any]:

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "model_id":
                MODEL_ID,

            "model_version":
                MODEL_VERSION,

            "model_contract":
                MODEL_CONTRACT,

            "input_channels":
                TOTAL_CHANNELS,

            "window_days":
                WINDOW_DAYS,

            "grid":
                [GRID_H, GRID_W],

            "depths_m":
                list(DEPTHS_M),

            "profile_modes":
                PROFILE_MODES,

            "total_parameters":
                int(total),

            "trainable_parameters":
                int(trainable),

            "architecture":
                {
                    "multi_scale_encoder":
                        "100x240 -> 50x120 -> 25x60",

                    "temporal_attention":
                        "2-layer 4-head Transformer over 14 daily states",

                    "surface_path":
                        "high-resolution OISST SST + anomaly branch",

                    "decoder":
                        "multi-scale skip fusion",

                    "vertical_decoder":
                        "8-mode continuous log-depth basis",

                    "output":
                        "[B,15,100,240]",

                    "target":
                        "canonical anomaly target",

                    "imputation":
                        False,

                    "physics_loss":
                        False,

                    "checkpoint_resume":
                        False,
                },
        }


def build_model(
    contract: Dict[str, Any]
) -> nn.Module:

    return Model(contract)
'''


MODEL_PATH.write_text(
    B9_SOURCE,
    encoding="utf-8",
)

print(
    "\n[B9 MODEL SOURCE]",
    flush=True,
)

print(
    "Written:",
    MODEL_PATH,
    flush=True,
)


# ============================================================
# 5. IMPORT B9
# ============================================================

b9 = import_module_from_path(
    MODEL_PATH,
    "oceanembed_b9_model",
)


# ============================================================
# 6. TRAINING CONFIG
# ============================================================

RUN_ROOT = Path(
    "/kaggle/working/OceanEmbed_Runs"
)

RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPERIMENT_NAME = (
    "B9_MSSTP_PROFILE_NET"
)

EPOCHS = 10
BATCH_SIZE = 1
NUM_WORKERS = 2

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

PATIENCE = 3

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

AMP_ENABLED = (
    DEVICE == "cuda"
)

LOG_EVERY = 50
CHECKPOINT_EVERY = 1


print(
    "\n[DEVICE]",
    flush=True,
)

print(
    "CUDA available:",
    torch.cuda.is_available(),
    flush=True,
)

print(
    "Using:",
    DEVICE,
    flush=True,
)

if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# 7. SEED
# ============================================================

def set_seed(
    seed: int
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


set_seed(SEED)


# ============================================================
# 8. CONFIG BUILDER
# ============================================================

def make_cfg(
    *,
    workers: int,
    epochs: int,
):
    cfg = framework.TrainConfig(

        dataset_root=str(
            CANONICAL_DATA_ROOT.resolve()
        ),

        backbone_path=str(
            BACKBONE_PATH.resolve()
        ),

        loader_path=str(
            LOADER_PATH.resolve()
        ),

        model_module=str(
            MODEL_PATH.resolve()
        ),

        model_class="Model",

        run_root=str(
            RUN_ROOT.resolve()
        ),

        experiment_name=EXPERIMENT_NAME,

        epochs=epochs,

        batch_size=BATCH_SIZE,

        num_workers=workers,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        grad_clip_norm=GRAD_CLIP_NORM,

        patience=PATIENCE,

        device=DEVICE,

        amp=AMP_ENABLED,

        seed=SEED,

        log_every=LOG_EVERY,

        checkpoint_every=CHECKPOINT_EVERY,

        resume=None,
    )

    cfg.validate()

    return cfg


# ============================================================
# 9. REAL-DATA PREFLIGHT
# ============================================================

print(
    "\n" + "=" * 110,
    flush=True,
)

print(
    "B9 REAL-DATA PREFLIGHT",
    flush=True,
)

print(
    "=" * 110,
    flush=True,
)


probe_cfg = make_cfg(
    workers=0,
    epochs=1,
)

probe_backbone = (
    framework.BackboneAdapter(
        probe_cfg
    )
)

print(
    "\n[BACKBONE]",
    flush=True,
)

print(
    "Bundle fingerprint:",
    getattr(
        probe_backbone.backbone,
        "fp",
        "UNKNOWN",
    ),
    flush=True,
)

EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)

actual_fp = getattr(
    probe_backbone.backbone,
    "fp",
    None,
)

if actual_fp != EXPECTED_BUNDLE_FP:

    raise RuntimeError(
        "\nFrozen ModelData_V1 fingerprint mismatch.\n"
        f"Expected: {EXPECTED_BUNDLE_FP}\n"
        f"Found:    {actual_fp}"
    )


contract = (
    probe_backbone.contract
)

print(
    "Target contract:",
    contract.get(
        "target",
        {},
    ),
    flush=True,
)


if contract.get(
    "target",
    {}
).get(
    "mode"
) != "anomaly":

    raise RuntimeError(
        "B9 expected canonical anomaly target mode."
    )


probe_trainer = framework.Trainer(
    probe_cfg,
    probe_backbone,
)

probe_model = (
    probe_trainer.model
)


total_params = sum(
    p.numel()
    for p in probe_model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in probe_model.parameters()
    if p.requires_grad
)

print(
    "\n[B9 MODEL]",
    flush=True,
)

print(
    "Total parameters:",
    total_params,
    flush=True,
)

print(
    "Trainable parameters:",
    trainable_params,
    flush=True,
)

print(
    json.dumps(
        b9.Model(
            contract
        ).model_summary(),
        indent=2,
    ),
    flush=True,
)


print(
    "\n[REAL BATCH]",
    flush=True,
)

real_batch = next(
    iter(
        probe_trainer.backbone.train_loader
    )
)

print(
    "Keys:",
    list(
        real_batch.keys()
    ),
    flush=True,
)


required_batch_keys = [
    "x_dynamic",
    "x_static",
    "x_dynamic_valid",
    "x_static_valid",
    "context_spatial",
    "context_temporal",
    "y",
    "loss_mask",
]


missing_keys = [
    k
    for k in required_batch_keys
    if k not in real_batch
]

if missing_keys:

    raise RuntimeError(
        "Real batch missing required keys:\n"
        + "\n".join(
            f"  {k}"
            for k in missing_keys
        )
    )


for key in required_batch_keys:

    value = real_batch[key]

    if torch.is_tensor(value):

        print(
            f"{key:<24} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}",
            flush=True,
        )

    else:

        print(
            f"{key:<24} "
            f"type={type(value).__name__}",
            flush=True,
        )


print(
    "\n[FORWARD PREFLIGHT]",
    flush=True,
)

moved = (
    framework.move_batch_to_device(
        real_batch,
        probe_trainer.device,
    )
)

probe_model.eval()

with torch.no_grad():

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=probe_trainer.amp_enabled,
    ):

        pred = (
            framework.forward_prediction(
                probe_model,
                moved,
            )
        )

        loss = (
            framework.masked_mse_loss(
                pred,
                moved["y"],
                moved["loss_mask"],
            )
        )


expected_shape = (
    real_batch[
        "x_dynamic"
    ].shape[0],
    15,
    100,
    240,
)


if tuple(
    pred.shape
) != expected_shape:

    raise RuntimeError(
        "\nB9 forward preflight FAILED.\n"
        f"Got:      {tuple(pred.shape)}\n"
        f"Expected: {expected_shape}"
    )


if not torch.isfinite(
    pred
).all():

    raise RuntimeError(
        "B9 forward preflight FAILED: non-finite prediction."
    )


if not torch.isfinite(
    loss
):

    raise RuntimeError(
        "B9 forward preflight FAILED: non-finite loss."
    )


print(
    "[PASS] Forward",
    flush=True,
)

print(
    "Prediction:",
    tuple(pred.shape),
    flush=True,
)

print(
    "Loss:",
    float(
        loss.detach().cpu()
    ),
    flush=True,
)


print(
    "\n[BACKWARD PREFLIGHT]",
    flush=True,
)

probe_model.train()

probe_optimizer = (
    torch.optim.AdamW(
        probe_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
)

probe_optimizer.zero_grad(
    set_to_none=True
)

with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=probe_trainer.amp_enabled,
):

    pred2 = (
        framework.forward_prediction(
            probe_model,
            moved,
        )
    )

    loss2 = (
        framework.masked_mse_loss(
            pred2,
            moved["y"],
            moved["loss_mask"],
        )
    )


loss2.backward()

grad_norm = (
    torch.nn.utils.clip_grad_norm_(
        probe_model.parameters(),
        GRAD_CLIP_NORM,
    )
)

if not torch.isfinite(
    grad_norm
):

    raise RuntimeError(
        "B9 backward preflight FAILED: non-finite gradient norm."
    )


probe_optimizer.step()

print(
    "[PASS] Backward + optimizer step",
    flush=True,
)

print(
    "Gradient norm:",
    float(
        grad_norm.detach().cpu()
    ),
    flush=True,
)


try:

    ld = getattr(
        probe_trainer.backbone.backbone,
        "loader",
        None,
    )

    if ld is not None:
        ld.close()

except Exception:
    pass


del pred
del pred2
del loss
del loss2
del moved
del real_batch
del probe_optimizer
del probe_model
del probe_trainer
del probe_backbone

gc.collect()

if DEVICE == "cuda":
    torch.cuda.empty_cache()


print(
    "\n" + "=" * 110,
    flush=True,
)

print(
    "B9 PREFLIGHT PASSED",
    flush=True,
)

print(
    "=" * 110,
    flush=True,
)



# ============================================================
# 10. RESUME CHECKPOINT DISCOVERY
# ============================================================
#
# This cell resumes ONLY from a completed-epoch latest_model.pt.
# It does NOT restart B9 from scratch.
#
# CHECKPOINT_PATH_OVERRIDE can be set to an exact Kaggle path when
# desired. Otherwise the code performs a bounded search of common
# Kaggle input layouts and accepts only checkpoints that pass all
# B9 identity checks below.
# ============================================================

CHECKPOINT_PATH_OVERRIDE = None
RESUME_CHECKPOINT_NAME = "latest_model.pt"

EXPECTED_BUNDLE_FP = (
    "ededfcfe6f43068fb9aa70f95479db5b2502ff32696e22134ba06c14dd7b229f"
)


def _bounded_latest_checkpoint_candidates():
    """Find latest_model.pt without rglob()/recursive filesystem scans."""
    candidates = []
    seen = set()

    def add(path):
        try:
            path = path.resolve()
        except Exception:
            return
        if path in seen:
            return
        seen.add(path)
        if path.is_file() and path.name == RESUME_CHECKPOINT_NAME:
            candidates.append(path)

    roots = [
        Path("/kaggle/input"),
    ]

    for root in roots:
        if not root.is_dir():
            continue

        # Depth 1: /kaggle/input/<dataset>/latest_model.pt
        try:
            children = sorted(root.iterdir(), key=lambda p: str(p))
        except OSError:
            children = []

        for child in children:
            if not child.is_dir():
                continue

            add(child / RESUME_CHECKPOINT_NAME)

            # Depth 2: /kaggle/input/<dataset>/<subdir>/latest_model.pt
            try:
                grandchildren = sorted(child.iterdir(), key=lambda p: str(p))
            except OSError:
                grandchildren = []

            for grandchild in grandchildren:
                if not grandchild.is_dir():
                    continue

                add(grandchild / RESUME_CHECKPOINT_NAME)

                # Depth 3: common /kaggle/input/datasets/<owner>/<dataset>/latest_model.pt
                try:
                    great_grandchildren = sorted(
                        grandchild.iterdir(),
                        key=lambda p: str(p),
                    )
                except OSError:
                    great_grandchildren = []

                for great_grandchild in great_grandchildren:
                    if not great_grandchild.is_dir():
                        continue
                    add(great_grandchild / RESUME_CHECKPOINT_NAME)

    return candidates


def _load_checkpoint_payload(path: Path):
    """Load a PyTorch checkpoint payload safely on CPU for metadata checks."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "latest_model.pt is not a dictionary checkpoint payload."
        )

    return payload


def _validate_b9_checkpoint_identity(
    path: Path,
    payload: Dict[str, Any],
):
    """Validate the checkpoint before the Trainer is allowed to load it."""
    if path.name != RESUME_CHECKPOINT_NAME:
        raise RuntimeError(
            f"Refusing resume: expected {RESUME_CHECKPOINT_NAME}, got {path.name!r}"
        )

    if payload.get("checkpoint_kind") != "epoch":
        raise RuntimeError(
            "Refusing resume: latest_model.pt must be a completed-epoch "
            f"checkpoint, got checkpoint_kind={payload.get('checkpoint_kind')!r}"
        )

    stored_fp = payload.get("backbone_fingerprint")
    if stored_fp != EXPECTED_BUNDLE_FP:
        raise RuntimeError(
            "Refusing resume: checkpoint Backbone fingerprint mismatch.\n"
            f"Expected: {EXPECTED_BUNDLE_FP}\n"
            f"Found:    {stored_fp}"
        )

    cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError(
            "Refusing resume: checkpoint config metadata is missing."
        )

    experiment = str(cfg.get("experiment_name", ""))
    if experiment != EXPERIMENT_NAME:
        raise RuntimeError(
            "Refusing cross-experiment resume.\n"
            f"Expected experiment: {EXPERIMENT_NAME}\n"
            f"Checkpoint experiment: {experiment}"
        )

    epoch = int(payload.get("epoch", 0))
    if epoch < 1:
        raise RuntimeError(
            f"Refusing resume: invalid checkpoint epoch {epoch}."
        )

    required_state = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    ]
    missing = [
        key
        for key in required_state
        if key not in payload
    ]
    if missing:
        raise RuntimeError(
            "Refusing resume: checkpoint missing required state: "
            + ", ".join(missing)
        )

    model_hash = payload.get("model_contract_hash")
    if not isinstance(model_hash, str) or not model_hash:
        raise RuntimeError(
            "Refusing resume: checkpoint model_contract_hash is missing."
        )

    return {
        "epoch": epoch,
        "global_step": int(payload.get("global_step", 0)),
        "val_loss": payload.get("val_loss"),
        "best_val_loss": payload.get("best_val_loss"),
        "model_contract_hash": model_hash,
    }


print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "B9 RESUME CHECKPOINT DISCOVERY",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)

if DEVICE != "cuda":
    raise RuntimeError(
        "This continuation cell is intended for a CUDA Kaggle runtime. "
        f"Configured DEVICE={DEVICE!r}."
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable in this new Kaggle session. "
        "Do not start the continuation on CPU. Enable a GPU runtime first."
    )


if CHECKPOINT_PATH_OVERRIDE:
    CHECKPOINT_PATH = Path(
        CHECKPOINT_PATH_OVERRIDE
    ).expanduser().resolve()
    CHECKPOINT_CANDIDATES = [CHECKPOINT_PATH]
else:
    CHECKPOINT_CANDIDATES = _bounded_latest_checkpoint_candidates()


if not CHECKPOINT_CANDIDATES:
    raise FileNotFoundError(
        "No latest_model.pt was found in the bounded Kaggle input search.\n"
        "Attach the dataset containing latest_model.pt, or set "
        "CHECKPOINT_PATH_OVERRIDE to the exact path."
    )


VALID_CHECKPOINTS = []
INVALID_CHECKPOINTS = []

for candidate in CHECKPOINT_CANDIDATES:
    try:
        payload = _load_checkpoint_payload(candidate)
        info = _validate_b9_checkpoint_identity(
            candidate,
            payload,
        )
        VALID_CHECKPOINTS.append(
            (candidate, payload, info)
        )
    except Exception as exc:
        INVALID_CHECKPOINTS.append(
            (candidate, f"{type(exc).__name__}: {exc}")
        )


if not VALID_CHECKPOINTS:
    details = "\n".join(
        f"  - {path}: {reason}"
        for path, reason in INVALID_CHECKPOINTS
    )
    raise RuntimeError(
        "latest_model.pt files were found, but none passed the pinned "
        "OceanEmbed B9 checkpoint identity checks.\n"
        + details
    )


# If more than one valid B9 latest_model.pt is mounted, use the one
# with the highest completed epoch. This avoids accidentally resuming
# from an older uploaded copy.
VALID_CHECKPOINTS.sort(
    key=lambda item: (
        item[2]["epoch"],
        str(item[0]),
    ),
    reverse=True,
)

CHECKPOINT_PATH, CHECKPOINT_PAYLOAD, CHECKPOINT_INFO = (
    VALID_CHECKPOINTS[0]
)


print(
    "Candidate latest_model.pt files:",
    len(CHECKPOINT_CANDIDATES),
    flush=True,
)
print(
    "Valid B9 checkpoints:",
    len(VALID_CHECKPOINTS),
    flush=True,
)
print(
    "Selected checkpoint:",
    CHECKPOINT_PATH,
    flush=True,
)
print(
    "Checkpoint epoch:",
    CHECKPOINT_INFO["epoch"],
    flush=True,
)
print(
    "Checkpoint global_step:",
    CHECKPOINT_INFO["global_step"],
    flush=True,
)
print(
    "Checkpoint val_loss:",
    CHECKPOINT_INFO["val_loss"],
    flush=True,
)
print(
    "Checkpoint best_val_loss:",
    CHECKPOINT_INFO["best_val_loss"],
    flush=True,
)


# ============================================================
# 11. CREATE RESUMED PRODUCTION TRAINER
# ============================================================

print(
    "\nCreating RESUMED production Trainer...",
    flush=True,
)


production_cfg = framework.TrainConfig(
    dataset_root=str(
        CANONICAL_DATA_ROOT.resolve()
    ),
    backbone_path=str(
        BACKBONE_PATH.resolve()
    ),
    loader_path=str(
        LOADER_PATH.resolve()
    ),
    model_module=str(
        MODEL_PATH.resolve()
    ),
    model_class="Model",
    run_root=str(
        RUN_ROOT.resolve()
    ),
    experiment_name=EXPERIMENT_NAME,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    grad_clip_norm=GRAD_CLIP_NORM,
    patience=PATIENCE,
    device=DEVICE,
    amp=AMP_ENABLED,
    seed=SEED,
    log_every=LOG_EVERY,
    checkpoint_every=CHECKPOINT_EVERY,
    resume=str(
        CHECKPOINT_PATH
    ),
)

production_cfg.validate()

production_backbone = (
    framework.BackboneAdapter(
        production_cfg
    )
)


# Validate the checkpoint model-contract hash against the exact
# current B9 model before Trainer.load_checkpoint() is relied upon.
contract_probe = b9.Model(
    production_backbone.contract
)
expected_model_hash = (
    contract_probe.contract_hash
)

if CHECKPOINT_INFO["model_contract_hash"] != expected_model_hash:
    raise RuntimeError(
        "Refusing resume: checkpoint model contract hash does not match "
        "the current B9 source + current Backbone contract.\n"
        f"Checkpoint: {CHECKPOINT_INFO['model_contract_hash']}\n"
        f"Current:    {expected_model_hash}"
    )

del contract_probe


print(
    "[PASS] Current B9 model contract hash matches checkpoint.",
    flush=True,
)


trainer = framework.Trainer(
    production_cfg,
    production_backbone,
)


EXPECTED_RESUME_START_EPOCH = (
    CHECKPOINT_INFO["epoch"] + 1
)

if trainer.start_epoch != EXPECTED_RESUME_START_EPOCH:
    raise RuntimeError(
        "Resume epoch mismatch.\n"
        f"Checkpoint epoch: {CHECKPOINT_INFO['epoch']}\n"
        f"Expected start_epoch: {EXPECTED_RESUME_START_EPOCH}\n"
        f"Trainer start_epoch: {trainer.start_epoch}"
    )

if trainer.resume_source is None:
    raise RuntimeError(
        "Trainer did not record the requested resume checkpoint."
    )

if trainer.resume_policy != "continue_after_completed_epoch":
    raise RuntimeError(
        "Unexpected resume policy for latest_model.pt: "
        f"{trainer.resume_policy!r}"
    )


print(
    "[PASS] Trainer loaded checkpoint state.",
    flush=True,
)
print(
    "Resume source:",
    trainer.resume_source,
    flush=True,
)
print(
    "Resume policy:",
    trainer.resume_policy,
    flush=True,
)
print(
    "Resume start epoch:",
    trainer.start_epoch,
    flush=True,
)
print(
    "Restored global step:",
    trainer.global_step,
    flush=True,
)
print(
    "Restored best validation loss:",
    trainer.best_val_loss,
    flush=True,
)


if trainer.start_epoch > EPOCHS:
    raise RuntimeError(
        "Checkpoint is already at/after the configured final epoch.\n"
        f"Checkpoint epoch: {CHECKPOINT_INFO['epoch']}\n"
        f"Configured total epochs: {EPOCHS}"
    )


# ============================================================
# 12. ONE-BATCH RESUME FORWARD PREFLIGHT
# ============================================================
#
# This is intentionally forward-only. The original training run
# already passed real-data backward preflight, and the checkpoint
# loader has restored optimizer/scaler state. We only need to prove
# that the restored model executes correctly on the new device before
# spending hours on the continuation.
# ============================================================

print(
    "\n[B9 RESUME FORWARD PREFLIGHT]",
    flush=True,
)

resume_batch = next(
    iter(
        production_backbone.train_loader
    )
)

resume_device_batch = (
    framework.move_batch_to_device(
        resume_batch,
        trainer.device,
    )
)

trainer.model.eval()

with torch.no_grad():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=trainer.amp_enabled,
    ):
        resume_prediction = (
            framework.forward_prediction(
                trainer.model,
                resume_device_batch,
            )
        )
        resume_loss = (
            framework.masked_mse_loss(
                resume_prediction,
                resume_device_batch["y"],
                resume_device_batch["loss_mask"],
            )
        )


expected_prediction_shape = (
    int(resume_device_batch["x_dynamic"].shape[0]),
    15,
    100,
    240,
)

if tuple(resume_prediction.shape) != expected_prediction_shape:
    raise RuntimeError(
        "Resumed B9 prediction shape mismatch.\n"
        f"Expected: {expected_prediction_shape}\n"
        f"Found:    {tuple(resume_prediction.shape)}"
    )

if not torch.isfinite(resume_prediction).all():
    raise FloatingPointError(
        "Resumed B9 model produced NaN/Inf during forward preflight."
    )

if not torch.isfinite(resume_loss):
    raise FloatingPointError(
        "Resumed B9 model produced NaN/Inf loss during forward preflight."
    )

print(
    "[PASS] Resumed model forward on new device.",
    flush=True,
)
print(
    "Prediction:",
    tuple(resume_prediction.shape),
    flush=True,
)
print(
    "Resume forward loss:",
    float(resume_loss.detach().cpu()),
    flush=True,
)


del resume_prediction
del resume_loss
del resume_device_batch
del resume_batch

gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 13. RESUMED RUN MANIFEST
# ============================================================

run_dir = Path(
    trainer.run_dir
)


# Preserve an exact copy of the checkpoint that started this run.
# This lives in /kaggle/working and does not mutate /kaggle/input.
shutil.copy2(
    CHECKPOINT_PATH,
    run_dir / "resumed_from_latest_model.pt",
)


# When latest_model.pt is itself the best checkpoint, carry that
# known-best state forward as the new run's best_model.pt. Otherwise
# leave best_model.pt for the Trainer to regenerate when a later
# epoch improves on the restored best_val_loss.
source_val_loss = CHECKPOINT_INFO["val_loss"]
source_best_val_loss = CHECKPOINT_INFO["best_val_loss"]

if (
    source_val_loss is not None
    and source_best_val_loss is not None
    and math.isclose(
        float(source_val_loss),
        float(source_best_val_loss),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
):
    shutil.copy2(
        CHECKPOINT_PATH,
        run_dir / "best_model.pt",
    )
    carried_best = True
else:
    carried_best = False


architecture_manifest = {
    "experiment_id": EXPERIMENT_NAME,
    "model_id": b9.MODEL_ID,
    "model_version": b9.MODEL_VERSION,
    "model_contract": b9.MODEL_CONTRACT,
    "dataset_root_physical": str(PHYSICAL_ROOT),
    "dataset_root_canonical": str(CANONICAL_DATA_ROOT),
    "bundle_fingerprint": EXPECTED_BUNDLE_FP,
    "runtime": {
        "framework": str(FRAMEWORK_PATH),
        "backbone": str(BACKBONE_PATH),
        "loader": str(LOADER_PATH),
    },
    "training": {
        "total_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "patience": PATIENCE,
        "device": DEVICE,
        "amp": AMP_ENABLED,
        "seed": SEED,
    },
    "resume": {
        "source_checkpoint": str(CHECKPOINT_PATH),
        "source_epoch": CHECKPOINT_INFO["epoch"],
        "source_global_step": CHECKPOINT_INFO["global_step"],
        "resume_start_epoch": trainer.start_epoch,
        "resume_policy": trainer.resume_policy,
        "restored_best_val_loss": trainer.best_val_loss,
        "carried_forward_best_checkpoint": carried_best,
    },
    "scientific_contract": {
        "window_days": 14,
        "input_channels": 36,
        "grid": [100, 240],
        "depths_m": [
            0, 5, 10, 20, 30, 50, 75, 100,
            125, 150, 200, 300, 500, 700, 1000,
        ],
        "target_mode": "anomaly",
        "GLORYS_target_only": True,
        "ARGO_training": False,
    },
    "architecture": b9.Model(
        production_backbone.contract
    ).model_summary(),
    "design_hypothesis": [
        "high-resolution SST-preserving path",
        "multi-scale spatial encoding",
        "14-day temporal attention",
        "multi-scale decoder",
        "coupled 15-depth profile reconstruction",
    ],
}


with open(
    run_dir / "b9_architecture_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        architecture_manifest,
        f,
        indent=2,
        default=str,
    )


print(
    "\nResumed production run directory:",
    run_dir,
    flush=True,
)
print(
    "Source checkpoint copy:",
    run_dir / "resumed_from_latest_model.pt",
    flush=True,
)
if carried_best:
    print(
        "[PASS] Carried forward best_model.pt from the selected latest checkpoint.",
        flush=True,
    )
else:
    print(
        "[INFO] Selected latest checkpoint was not itself the recorded best; "
        "Trainer will create a new best_model.pt when a later epoch improves.",
        flush=True,
    )


# ============================================================
# 14. START RESUMED TRAINING
# ============================================================

print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "STARTING OCEANEMBED B9 RESUMED TRAINING",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)

print(
    f"""
Experiment          : {EXPERIMENT_NAME}
Total epochs        : {EPOCHS}
Checkpoint epoch    : {CHECKPOINT_INFO['epoch']}
Resume start epoch  : {trainer.start_epoch}
Remaining epochs    : {EPOCHS - trainer.start_epoch + 1}
Batch               : {BATCH_SIZE}
Workers             : {NUM_WORKERS}
LR                  : {LEARNING_RATE}
Weight decay        : {WEIGHT_DECAY}
Patience            : {PATIENCE}
Device              : {DEVICE}
GPU                 : {torch.cuda.get_device_name(0)}
AMP                 : {AMP_ENABLED}
Restored global step: {trainer.global_step}
Restored best val   : {trainer.best_val_loss}

B9:
  Multi-scale spatial encoder
  14-day temporal attention
  High-resolution SST/SST-anomaly path
  Multi-scale decoder
  8-mode coupled vertical profile
  15 output depths
""",
    flush=True,
)


training_summary = (
    trainer.fit()
)


print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "B9 RESUMED TRAINING FINISHED",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)
print(
    json.dumps(
        training_summary,
        indent=2,
        default=str,
    ),
    flush=True,
)


# ============================================================
# 15. TEST EVALUATION
# ============================================================

print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "B9 TEST EVALUATION",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)

test_metrics = (
    trainer.evaluate_test()
)

print(
    json.dumps(
        test_metrics,
        indent=2,
        default=str,
    ),
    flush=True,
)


# ============================================================
# 16. PRINT DEPTH-WISE RESULTS
# ============================================================

print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "B9 DEPTH-WISE TEST RMSE",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)

for row in test_metrics.get(
    "by_depth",
    [],
):
    print(
        f"{float(row['depth_m']):7.0f} m"
        f" | RMSE={float(row['RMSE']):.6f}"
        f" | MAE={float(row['MAE']):.6f}"
        f" | r={float(row['Pearson_r']):.6f}"
        f" | ACC={float(row['ACC_spatial_daily_mean']):.6f}",
        flush=True,
    )


# ============================================================
# 17. SAVE FINAL B9 MANIFEST
# ============================================================

final_manifest = {
    **architecture_manifest,
    "training_summary": training_summary,
    "test_metrics": test_metrics,
    "status": "PASS",
}


with open(
    run_dir / "b9_final_manifest.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        final_manifest,
        f,
        indent=2,
        default=str,
    )


print(
    "\n" + "=" * 110,
    flush=True,
)
print(
    "OCEANEMBED B9 RESUME COMPLETE",
    flush=True,
)
print(
    "Run directory:",
    run_dir,
    flush=True,
)
print(
    "Best checkpoint:",
    run_dir / "best_model.pt",
    flush=True,
)
print(
    "Final manifest:",
    run_dir / "b9_final_manifest.json",
    flush=True,
)
print(
    "=" * 110,
    flush=True,
)
