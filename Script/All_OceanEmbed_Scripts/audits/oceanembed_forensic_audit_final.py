#!/usr/bin/env python3
"""
OceanEmbed Dataset Forensic Audit Engine - FINAL

A read-only, evidence-driven audit of the OceanEmbed NetCDF collection.

Design goals:
- Never modify raw input data.
- CF-aware time decoding.
- File-level provenance and schema auditing.
- Exact temporal coverage and intersection.
- Grid geometry and actual canonical-cell coverage.
- Explicit, dataset-specific variable semantics.
- Fill/missing/finite-value auditing.
- Exact GLORYS target-depth verification.
- Numerically stable aggregate statistics and deterministic quantile sampling.
- Actual spatial interpolation cross-validation ("holdout"), not nearest-vs-linear substitution.
- Vector-component-aware regridding diagnostics.
- GlobCurrent vs OSCAR comparability and redundancy analysis.
- Joint valid-mask analysis based on the actual variables intended for integration.
- Evidence-derived integration recommendations; registry flags are priors, not conclusions.
- Truthful completeness status.
- Deterministic sampling, cache invalidation, and machine-readable reports.

Canonical OceanEmbed domain:
    latitude  5.00 .. 30.00 N
    longitude 45.00 .. 105.00 E
    spacing   0.25 degree
    shape     101 x 241

Target:
    GLORYS potential temperature at exactly:
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

This program is an AUDITOR. It does not create or alter the production training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

try:
    import cftime
except Exception:
    cftime = None

try:
    import dask
    import dask.array as darray
except Exception:
    dask = None
    darray = None

AUDIT_VERSION = "FINAL-1.0"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RES = 0.25
LAT_SIZE, LON_SIZE = 101, 241
GLORYS_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=float,
)
DEPTH_TOL = 1e-6

CORE_KEYS = ["oisst", "oisss", "duacs", "globecurrent", "ccmp"]
TARGET_KEY = "glorys"
VALIDATION_KEYS = ["oscar", "argo"]

REGISTRY = {
    "oisst": dict(name="OISST", root="oisst", purpose="surface sea-surface temperature",
                  prior="CORE_INPUT", semantic_groups=["sst"]),
    "oisss": dict(name="OISSS", root="oisss", purpose="surface sea-surface salinity",
                  prior="CORE_INPUT", semantic_groups=["sss"]),
    "duacs": dict(name="DUACS", root="duacs", purpose="sea-level / SLA / ADT",
                  prior="CORE_INPUT", semantic_groups=["sla", "adt", "ugos", "vgos"]),
    "globecurrent": dict(name="GlobCurrent", root="globecurrent", purpose="surface ocean current",
                         prior="CORE_INPUT", semantic_groups=["u", "v", "ugos", "vgos", "uek", "vek"]),
    "ccmp": dict(name="CCMP", root="ccmp", purpose="surface wind",
                 prior="CORE_INPUT", semantic_groups=["u_wind", "v_wind", "wind_speed"]),
    "glorys": dict(name="GLORYS", root="glorys", purpose="subsurface target / reference grid",
                   prior="TARGET", semantic_groups=["thetao"]),
    "oscar": dict(name="OSCAR", root="oscar", purpose="independent surface current source",
                  prior="VALIDATION_ONLY", semantic_groups=["u", "v"]),
    "argo": dict(name="Argo", root="argo", purpose="independent vertical-profile validation",
                 prior="VALIDATION_ONLY", semantic_groups=["temp", "sal", "pres"]),
}

CANON_LAT = np.round(np.arange(LAT_MIN, LAT_MAX + RES / 2, RES), 10)
CANON_LON = np.round(np.arange(LON_MIN, LON_MAX + RES / 2, RES), 10)


# ---------------------------------------------------------------------------
# Logging / JSON utilities
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("oceanembed.audit")


def setup_logging(verbose: bool) -> None:
    LOGGER.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if cftime is not None and isinstance(x, cftime.datetime):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, set):
        return sorted(x)
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=json_default, allow_nan=False)
    os.replace(tmp, path)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_grid() -> dict[str, Any]:
    return {
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "resolution": RES,
        "lat_size": LAT_SIZE, "lon_size": LON_SIZE,
        "lat_values": CANON_LAT.tolist(),
        "lon_values": CANON_LON.tolist(),
    }


# ---------------------------------------------------------------------------
# Configuration / cache
# ---------------------------------------------------------------------------

@dataclass
class Config:
    root: Path = Path(r"D:\OceanEmbed\oceanembed-data\raw")
    output: Path = Path(r"D:\OceanEmbed\oceanembed-data\audit_final")
    datasets: list[str] = field(default_factory=lambda: ["all"])
    full: bool = True
    include_argo: bool = False
    include_argo_full: bool = False
    checksums: bool = False
    resume: bool = True
    force: bool = False
    verbose: bool = False
    workers: int = 1
    sample_files: int = 10
    quantile_sample: int = 250000
    joint_dates: int = 10
    joint_all: bool = False
    redundancy_dates: int = 10
    regrid_files: int = 5
    holdout_fraction: float = 0.20
    holdout_points: int = 2000
    seed: int = 42

    def validate(self) -> None:
        if self.sample_files < 1:
            raise ValueError("sample_files must be >= 1")
        if self.quantile_sample < 100:
            raise ValueError("quantile_sample must be >= 100")
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5)")
        if self.holdout_points < 10:
            raise ValueError("holdout_points must be >= 10")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")

    def hash(self) -> str:
        d = asdict(self)
        d["root"] = str(self.root)
        d["output"] = str(self.output)
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, default=json_default).encode()
        ).hexdigest()[:20]


class Cache:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.output / "cache" / f"cache_{cfg.hash()}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {}
        if cfg.resume and self.path.exists() and not cfg.force:
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                LOGGER.warning("Cache could not be loaded; starting clean.")

    @staticmethod
    def key(path: Path) -> str:
        st = path.stat()
        return f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"

    def get(self, path: Path, stage: str) -> Any:
        return self.data.get(self.key(path), {}).get(stage)

    def set(self, path: Path, stage: str, value: Any) -> None:
        if self.cfg.force:
            return
        self.data.setdefault(self.key(path), {})[stage] = value

    def save(self) -> None:
        if self.cfg.force:
            return
        write_json(self.path, self.data)


# ---------------------------------------------------------------------------
# NetCDF / CF helpers
# ---------------------------------------------------------------------------

def open_nc(path: Path) -> xr.Dataset:
    """
    Decode CF conventions on open. mask_and_scale=True ensures scale_factor,
    add_offset, _FillValue and missing_value are interpreted by xarray.
    """
    return xr.open_dataset(
        path,
        engine="netcdf4",
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
        use_cftime=True,
        chunks="auto",
        cache=False,
    )


def close_quiet(ds: Optional[xr.Dataset]) -> None:
    if ds is not None:
        try:
            ds.close()
        except Exception:
            pass


def coord_name(ds: xr.Dataset, kind: str) -> Optional[str]:
    candidates = []
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if kind == "lat" and (low in {"lat", "latitude"} or "latitude" in low):
            candidates.append(name)
        if kind == "lon" and (low in {"lon", "longitude"} or "longitude" in low):
            candidates.append(name)
    return candidates[0] if candidates else None


def depth_name(ds: xr.Dataset) -> Optional[str]:
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if low in {"depth", "deptht", "depthu", "depthv", "lev", "level"} or "depth" in low:
            return name
    return None


def time_name(ds: xr.Dataset) -> Optional[str]:
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if low == "time" or "time" in low:
            return name
    return None


def to_time_objects(ds: xr.Dataset) -> list[Any]:
    tn = time_name(ds)
    if tn is None:
        return []
    vals = ds[tn].values
    return list(np.asarray(vals).reshape(-1))


def time_key(t: Any) -> str:
    try:
        return pd.Timestamp(t).isoformat()
    except Exception:
        return str(t)


def date_key(t: Any) -> str:
    try:
        return pd.Timestamp(t).strftime("%Y-%m-%d")
    except Exception:
        return f"{t.year:04d}-{t.month:02d}-{t.day:02d}"


def seconds_between(a: Any, b: Any) -> Optional[float]:
    try:
        return float((b - a) / np.timedelta64(1, "s"))
    except Exception:
        try:
            return float((b - a).total_seconds())
        except Exception:
            return None


def is_regular(values: Sequence[Any]) -> Optional[bool]:
    if len(values) < 3:
        return True if len(values) >= 2 else None
    diffs = [seconds_between(values[i], values[i + 1]) for i in range(len(values) - 1)]
    if any(x is None for x in diffs):
        return None
    return bool(np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-6))


def time_semantics(ds: xr.Dataset) -> dict[str, Any]:
    tn = time_name(ds)
    if tn is None:
        return {"present": False}
    t = ds[tn]
    vals = to_time_objects(ds)
    info: dict[str, Any] = {
        "present": True,
        "name": tn,
        "dtype": str(t.dtype),
        "n": len(vals),
        "first": time_key(vals[0]) if vals else None,
        "last": time_key(vals[-1]) if vals else None,
        "first_date": date_key(vals[0]) if vals else None,
        "last_date": date_key(vals[-1]) if vals else None,
        "calendar": t.encoding.get("calendar", t.attrs.get("calendar", "standard")),
        "units": t.encoding.get("units", t.attrs.get("units")),
        "bounds": t.attrs.get("bounds"),
        "cell_methods": t.attrs.get("cell_methods"),
    }
    if len(vals) >= 2:
        diffs = [seconds_between(vals[i], vals[i + 1]) for i in range(len(vals) - 1)]
        valid = [x for x in diffs if x is not None]
        if valid:
            info["timestep_seconds_median"] = float(np.median(valid))
            info["timestep_seconds_min"] = float(np.min(valid))
            info["timestep_seconds_max"] = float(np.max(valid))
            info["regular"] = bool(np.allclose(valid, valid[0], rtol=1e-9, atol=1e-6))
    # Record explicit time bounds values if a bounds variable exists.
    bname = info.get("bounds")
    if bname and bname in ds.variables:
        try:
            info["bounds_shape"] = list(ds[bname].shape)
            info["bounds_first"] = [time_key(x) for x in np.asarray(ds[bname].values[0]).reshape(-1)]
            info["bounds_last"] = [time_key(x) for x in np.asarray(ds[bname].values[-1]).reshape(-1)]
        except Exception:
            info["bounds_decode_status"] = "FAILED"
    return info


def filename_date_tokens(path: Path) -> list[str]:
    stem = path.stem
    out = []
    for pattern in (r"\b\d{8}\b", r"\b\d{4}[-_]\d{2}[-_]\d{2}\b", r"\b\d{4}[-_]\d{2}\b"):
        out.extend(re.findall(pattern, stem))
    return out


def parse_filename_token(token: str) -> Optional[pd.Timestamp]:
    token = token.replace("_", "-")
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y-%m"):
        try:
            return pd.to_datetime(token, format=fmt)
        except Exception:
            pass
    return None


def safe_dataarray_values(da: xr.DataArray) -> np.ndarray:
    arr = da.data
    if dask is not None and dask.is_dask_collection(arr):
        arr = arr.compute()
    return np.asarray(arr)


def finite_stats(arr: np.ndarray, qsample: int, seed: int) -> dict[str, Any]:
    a = np.asarray(arr)
    flat = a.reshape(-1)
    finite = flat[np.isfinite(flat)]
    n = int(flat.size)
    nf = int(finite.size)
    ni = int(np.isinf(flat).sum())
    nn = int(np.isnan(flat).sum())
    result = {
        "count": n,
        "finite_count": nf,
        "nan_count": nn,
        "inf_count": ni,
        "invalid_count": int(n - nf),
        "valid_percent": float(100 * nf / n) if n else None,
        "min": None, "max": None, "mean": None, "std": None,
        "quantiles": {},
        "quantile_method": None,
    }
    if nf == 0:
        result["quantile_method"] = "NO_FINITE_VALUES"
        return result
    # Stable enough for audit summaries; values are already decoded/masked.
    result["min"] = float(np.min(finite))
    result["max"] = float(np.max(finite))
    result["mean"] = float(np.mean(finite, dtype=np.float64))
    result["std"] = float(np.std(finite, dtype=np.float64))
    if nf <= qsample:
        qvals = finite
        method = "EXACT_ALL_FINITE_VALUES"
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(nf, size=qsample, replace=False)
        qvals = finite[idx]
        method = f"DETERMINISTIC_RANDOM_SAMPLE_{qsample}"
    qs = [1, 5, 25, 50, 75, 95, 99]
    result["quantiles"] = {f"p{q:02d}": float(np.percentile(qvals, q)) for q in qs}
    result["quantile_method"] = method
    return result


def robust_stats_for_dataarray(da: xr.DataArray, cfg: Config, variable_seed: int) -> dict[str, Any]:
    # Compute chunks separately where possible; a single compute is reliable and
    # bounded by the quantile sampler only after the data are materialized.
    arr = safe_dataarray_values(da)
    return finite_stats(arr, cfg.quantile_sample, variable_seed)


# ---------------------------------------------------------------------------
# Variable semantics
# ---------------------------------------------------------------------------

SEMANTIC_RULES: dict[str, list[tuple[str, str]]] = {
    "oisst": [("sst", "sea_surface_temperature")],
    "oisss": [("sss", "sea_surface_salinity")],
    "duacs": [
        ("sla", "sea_surface_height_above_mean_sea_level"),
        ("adt", "sea_surface_height_above_geoid"),
        ("ugos", "eastward_geostrophic_sea_water_velocity"),
        ("vgos", "northward_geostrophic_sea_water_velocity"),
    ],
    "globecurrent": [
        ("u", "eastward_sea_water_velocity"),
        ("v", "northward_sea_water_velocity"),
        ("ugos", "eastward_geostrophic_sea_water_velocity"),
        ("vgos", "northward_geostrophic_sea_water_velocity"),
        ("uek", "eastward_ekman_sea_water_velocity"),
        ("vek", "northward_ekman_sea_water_velocity"),
    ],
    "ccmp": [
        ("u_wind", "eastward_wind"),
        ("v_wind", "northward_wind"),
        ("wind_speed", "wind_speed"),
    ],
    "glorys": [("thetao", "sea_water_potential_temperature")],
    "oscar": [
        ("u", "eastward_sea_water_velocity"),
        ("v", "northward_sea_water_velocity"),
    ],
    "argo": [
        ("temp", "sea_water_temperature"),
        ("sal", "sea_water_salinity"),
        ("pres", "sea_water_pressure"),
    ],
}

NAME_FALLBACKS = {
    "oisst": {"sst": ["sst", "analysed_sst", "sea_surface_temperature"]},
    "oisss": {"sss": ["sss", "salinity", "sea_surface_salinity"]},
    "glorys": {"thetao": ["thetao", "potential_temperature", "theta", "votemper"]},
    "globecurrent": {"u": ["uo", "ugos", "u", "eastward_velocity"],
                     "v": ["vo", "vgos", "v", "northward_velocity"]},
    "oscar": {"u": ["u", "u_current", "ugos"], "v": ["v", "v_current", "vgos"]},
    "ccmp": {"u_wind": ["uwnd", "u_wind", "u"], "v_wind": ["vwnd", "v_wind", "v"]},
}


def normalize_text(x: Any) -> str:
    return str(x or "").strip().lower()


def resolve_variables(dataset_key: str, ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """
    Returns semantic role -> evidence. Metadata is preferred. Name fallback is
    accepted only when the variable's dimensions/units make it plausible.
    """
    out: dict[str, dict[str, Any]] = {}
    for role, required_std in SEMANTIC_RULES.get(dataset_key, []):
        exact = []
        for v in ds.data_vars:
            std = normalize_text(ds[v].attrs.get("standard_name"))
            if std == required_std or required_std in std:
                exact.append(v)
        if len(exact) == 1:
            v = exact[0]
            out[role] = {"variable": v, "basis": "standard_name", "confidence": "HIGH"}
        elif len(exact) > 1:
            # Ambiguity is evidence, not permission to guess.
            out[role] = {
                "variable": None, "basis": "ambiguous_standard_name",
                "candidates": exact, "confidence": "AMBIGUOUS",
            }

    for role, names in NAME_FALLBACKS.get(dataset_key, {}).items():
        if role in out:
            continue
        candidates = []
        for v in ds.data_vars:
            low = v.lower()
            if any(low == n or n in low for n in names):
                candidates.append(v)
        plausible = []
        for v in candidates:
            units = normalize_text(ds[v].attrs.get("units"))
            dims = ds[v].dims
            has_space = coord_name(ds, "lat") in dims if coord_name(ds, "lat") else False
            has_space = has_space or (coord_name(ds, "lon") in dims if coord_name(ds, "lon") else False)
            if has_space:
                plausible.append(v)
        if len(plausible) == 1:
            out[role] = {"variable": plausible[0], "basis": "name_fallback", "confidence": "MEDIUM"}
        elif len(plausible) > 1:
            out[role] = {
                "variable": None, "basis": "ambiguous_name_fallback",
                "candidates": plausible, "confidence": "AMBIGUOUS",
            }
    return out


def semantic_audit(dataset_key: str, ds: xr.Dataset) -> dict[str, Any]:
    resolved = resolve_variables(dataset_key, ds)
    rows = []
    for role, evidence in resolved.items():
        vname = evidence.get("variable")
        if not vname:
            rows.append({"role": role, **evidence})
            continue
        v = ds[vname]
        std = v.attrs.get("standard_name")
        units = v.attrs.get("units")
        long_name = v.attrs.get("long_name")
        sign = "UNKNOWN"
        if std:
            if "eastward" in std:
                sign = "positive_eastward"
            elif "northward" in std:
                sign = "positive_northward"
            elif "westward" in std:
                sign = "positive_westward"
            elif "southward" in std:
                sign = "positive_southward"

        expected = {
            "sst": {"degc", "degree_celsius", "c", "k", "kelvin"},
            "thetao": {"degc", "degree_celsius", "c", "k", "kelvin"},
            "sss": {"1", "psu", "1e-3", "practical_salinity_unit"},
            "sla": {"m", "meter", "meters"},
            "adt": {"m", "meter", "meters"},
            "ugos": {"m/s", "m s-1", "m s^-1", "meter second-1"},
            "vgos": {"m/s", "m s-1", "m s^-1", "meter second-1"},
            "u": {"m/s", "m s-1", "m s^-1", "meter second-1", "cm/s"},
            "v": {"m/s", "m s-1", "m s^-1", "meter second-1", "cm/s"},
            "uek": {"m/s", "m s-1", "m s^-1"},
            "vek": {"m/s", "m s-1", "m s^-1"},
            "u_wind": {"m/s", "m s-1", "m s^-1"},
            "v_wind": {"m/s", "m s-1", "m s^-1"},
        }.get(role, set())
        unit_norm = normalize_text(units)
        unit_ok = any(u == unit_norm for u in expected) if expected else None
        if units is None:
            unit_status = "MISSING"
        elif unit_ok:
            unit_status = "CONFIRMED"
        else:
            unit_status = "REQUIRES_REVIEW"

        rows.append({
            "role": role,
            "variable": vname,
            "basis": evidence.get("basis"),
            "confidence": evidence.get("confidence"),
            "standard_name": std,
            "long_name": long_name,
            "units": units,
            "sign_convention": sign,
            "unit_status": unit_status,
            "dimensions": list(v.dims),
            "dtype": str(v.dtype),
        })
    return {
        "resolved": resolved,
        "variables": rows,
        "semantic_status": (
            "CONFIRMED"
            if rows and all(r.get("confidence") == "HIGH" and r.get("unit_status") == "CONFIRMED"
                            for r in rows if r.get("variable"))
            else "REQUIRES_REVIEW"
        ),
    }


# ---------------------------------------------------------------------------
# File index / metadata / schema
# ---------------------------------------------------------------------------

def discover(cfg: Config, key: str) -> list[Path]:
    root = cfg.root / REGISTRY[key]["root"]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.nc") if p.is_file())


def file_index(cfg: Config, key: str, files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in files:
        try:
            st = p.stat()
            row = {
                "dataset": key,
                "file": str(p),
                "relative": str(p.relative_to(cfg.root)),
                "size_bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
            if cfg.checksums:
                row["sha256"] = sha256_file(p)
            rows.append(row)
        except Exception as e:
            rows.append({"dataset": key, "file": str(p), "error": str(e)})
    return rows


def schema_signature(ds: xr.Dataset) -> dict[str, Any]:
    vars_sig = {}
    for name in sorted(ds.data_vars):
        v = ds[name]
        vars_sig[name] = {
            "dtype": str(v.dtype),
            "dims": list(v.dims),
            "shape": list(v.shape),
            "attrs": {k: str(vv) for k, vv in sorted(v.attrs.items())},
        }
    return {
        "dims": {k: int(v) for k, v in sorted(ds.sizes.items())},
        "coords": sorted(ds.coords),
        "data_vars": vars_sig,
        "global_attrs": {k: str(v) for k, v in sorted(ds.attrs.items())},
    }


def metadata_snapshot(ds: xr.Dataset, path: Path) -> dict[str, Any]:
    coords = {}
    for c in ds.coords:
        a = ds[c]
        coords[c] = {
            "dtype": str(a.dtype),
            "dims": list(a.dims),
            "shape": list(a.shape),
            "attrs": dict(a.attrs),
            "encoding": {k: str(v) for k, v in a.encoding.items()
                         if k in {"units", "calendar", "_FillValue", "dtype"}},
        }
    vars_ = {}
    for v in ds.data_vars:
        a = ds[v]
        vars_[v] = {
            "dtype": str(a.dtype),
            "dims": list(a.dims),
            "shape": list(a.shape),
            "attrs": dict(a.attrs),
            "encoding": {k: str(vv) for k, vv in a.encoding.items()
                         if k in {"units", "calendar", "_FillValue", "dtype",
                                  "scale_factor", "add_offset"}},
        }
    return {
        "file": str(path),
        "global_attributes": dict(ds.attrs),
        "dimensions": {k: int(v) for k, v in ds.sizes.items()},
        "coordinates": coords,
        "variables": vars_,
        "time": time_semantics(ds),
    }


def schema_audit(cfg: Config, key: str, files: list[Path], cache: Cache) -> dict[str, Any]:
    counts = Counter()
    failures = []
    checked = 0
    representative = None
    for p in files:
        cached = cache.get(p, "schema")
        if cached is not None:
            sig = json.dumps(cached, sort_keys=True, default=json_default)
            counts[sig] += 1
            checked += 1
            if representative is None:
                representative = cached
            continue
        ds = None
        try:
            ds = open_nc(p)
            sig_obj = schema_signature(ds)
            counts[json.dumps(sig_obj, sort_keys=True, default=json_default)] += 1
            cache.set(p, "schema", sig_obj)
            checked += 1
            if representative is None:
                representative = sig_obj
        except Exception as e:
            failures.append({"file": str(p), "stage": "schema", "error": repr(e)})
        finally:
            close_quiet(ds)
    cache.save()
    return {
        "files_total": len(files),
        "files_checked": checked,
        "files_failed": len(failures),
        "unique_schemas": len(counts),
        "schema_counts": dict(counts),
        "failures": failures,
        "status": "COMPLETE" if checked == len(files) else "PARTIAL",
    }


# ---------------------------------------------------------------------------
# Temporal audit
# ---------------------------------------------------------------------------

def temporal_audit(cfg: Config, key: str, files: list[Path]) -> dict[str, Any]:
    records = []
    all_timestamps: list[str] = []
    date_counts = Counter()
    timestamp_counts = Counter()
    filename_issues = []
    failures = []

    for p in files:
        ds = None
        try:
            ds = open_nc(p)
            vals = to_time_objects(ds)
            sem = time_semantics(ds)
            keys = [time_key(x) for x in vals]
            dates = [date_key(x) for x in vals]
            for k in keys:
                all_timestamps.append(k)
                timestamp_counts[k] += 1
            for d in dates:
                date_counts[d] += 1

            tokens = filename_date_tokens(p)
            parsed = [parse_filename_token(t) for t in tokens]
            parsed = [x for x in parsed if x is not None]
            filename_check = "NO_DATE_TOKEN"
            if parsed and vals:
                first_d = pd.Timestamp(date_key(vals[0]))
                last_d = pd.Timestamp(date_key(vals[-1]))
                outside = [x.strftime("%Y-%m-%d") for x in parsed
                           if x < first_d or x > last_d]
                filename_check = "OUTSIDE_TIME_RANGE" if outside else "CONSISTENT_RANGE"
                if outside:
                    filename_issues.append({
                        "file": str(p), "filename_dates": outside,
                        "actual_first": date_key(vals[0]), "actual_last": date_key(vals[-1]),
                    })
            records.append({
                "file": str(p),
                **sem,
                "filename_date_check": filename_check,
                "timestamps": keys,
            })
        except Exception as e:
            failures.append({"file": str(p), "stage": "temporal", "error": repr(e)})
        finally:
            close_quiet(ds)

    dates = sorted(date_counts)
    timestamps = sorted(timestamp_counts)
    duplicate_dates = {d: c for d, c in date_counts.items() if c > 1}
    duplicate_timestamps = {t: c for t, c in timestamp_counts.items() if c > 1}

    missing_dates = []
    if dates:
        try:
            expected = pd.date_range(dates[0], dates[-1], freq="D")
            present = set(dates)
            missing_dates = [d.strftime("%Y-%m-%d") for d in expected if d.strftime("%Y-%m-%d") not in present]
        except Exception:
            missing_dates = []

    return {
        "files_total": len(files),
        "files_checked": len(records),
        "files_failed": len(failures),
        "records": records,
        "first_timestamp": timestamps[0] if timestamps else None,
        "last_timestamp": timestamps[-1] if timestamps else None,
        "unique_timestamps": timestamps,
        "unique_dates": dates,
        "date_counts": dict(date_counts),
        "timestamp_counts": dict(timestamp_counts),
        "duplicate_dates": duplicate_dates,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_dates": missing_dates,
        "filename_date_issues": filename_issues,
        "status": "COMPLETE" if len(records) == len(files) else "PARTIAL",
    }


# ---------------------------------------------------------------------------
# Grid / spatial audit
# ---------------------------------------------------------------------------

def axis_info(a: xr.DataArray) -> dict[str, Any]:
    vals = np.asarray(a.values).astype(float).reshape(-1)
    if vals.size == 0:
        raise ValueError("empty coordinate")
    diffs = np.diff(vals)
    return {
        "size": int(vals.size),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "spacing_median": float(np.nanmedian(np.abs(diffs))) if diffs.size else None,
        "spacing_min": float(np.nanmin(np.abs(diffs))) if diffs.size else None,
        "spacing_max": float(np.nanmax(np.abs(diffs))) if diffs.size else None,
        "uniform": bool(np.allclose(diffs, diffs[0], rtol=1e-8, atol=1e-10)) if diffs.size > 1 else None,
        "ascending": bool(vals[-1] > vals[0]) if vals.size > 1 else None,
        "descending": bool(vals[-1] < vals[0]) if vals.size > 1 else None,
        "values": vals.tolist(),
    }


def actual_grid_audit(ds: xr.Dataset) -> dict[str, Any]:
    la, lo = coord_name(ds, "lat"), coord_name(ds, "lon")
    if not la or not lo:
        return {"status": "INCONCLUSIVE", "reason": "latitude/longitude coordinate not found"}
    try:
        lat = axis_info(ds[la])
        lon = axis_info(ds[lo])
        lon_vals = np.asarray(ds[lo].values).astype(float)
        lat_vals = np.asarray(ds[la].values).astype(float)
        lon_norm = lon_vals.copy()
        # Convert only if clearly 0..360; retain original values in report.
        if np.nanmin(lon_norm) >= 0 and np.nanmax(lon_norm) > 180:
            lon_norm = ((lon_norm + 180) % 360) - 180
        in_lat = (lat_vals >= LAT_MIN - 1e-10) & (lat_vals <= LAT_MAX + 1e-10)
        in_lon = (lon_norm >= LON_MIN - 1e-10) & (lon_norm <= LON_MAX + 1e-10)
        return {
            "status": "COMPLETE",
            "lat_name": la, "lon_name": lo,
            "lat": lat, "lon": lon,
            "longitude_normalized_range": [float(lon_norm.min()), float(lon_norm.max())],
            "canonical_lat_points_present": int(in_lat.sum()),
            "canonical_lon_points_present": int(in_lon.sum()),
            "canonical_point_grid_exact": bool(
                lat["size"] == LAT_SIZE and lon["size"] == LON_SIZE
                and np.allclose(lat_vals, CANON_LAT, atol=1e-8)
                and np.allclose(lon_norm, CANON_LON, atol=1e-8)
            ),
        }
    except Exception as e:
        return {"status": "FAILED", "error": repr(e)}


def spatial_audit(cfg: Config, key: str, files: list[Path]) -> dict[str, Any]:
    sample = files if len(files) <= cfg.sample_files else [
        files[0], files[len(files)//2], files[-1]
    ]
    records, failures = [], []
    for p in sample:
        ds = None
        try:
            ds = open_nc(p)
            records.append({"file": str(p), "grid": actual_grid_audit(ds)})
        except Exception as e:
            failures.append({"file": str(p), "stage": "spatial", "error": repr(e)})
        finally:
            close_quiet(ds)
    return {
        "sample_files": len(sample),
        "records": records,
        "failures": failures,
        "status": "COMPLETE" if records else "INCONCLUSIVE",
    }


def canonical_interp(da: xr.DataArray, lat: str, lon: str, method: str = "linear") -> xr.DataArray:
    x = da.sortby(lat).sortby(lon)
    # normalize 0..360 source longitude before interpolation when needed.
    vals = np.asarray(x[lon].values).astype(float)
    if np.nanmin(vals) >= 0 and np.nanmax(vals) > 180:
        newlon = ((vals + 180) % 360) - 180
        x = x.assign_coords({lon: newlon}).sortby(lon)
    return x.interp({lat: CANON_LAT, lon: CANON_LON}, method=method)


def spatial_cell_coverage(ds: xr.Dataset, varname: str) -> dict[str, Any]:
    v = ds[varname]
    lat, lon = coord_name(ds, "lat"), coord_name(ds, "lon")
    if not lat or not lon:
        return {"status": "INCONCLUSIVE", "reason": "coordinates unavailable"}
    try:
        # A source point contributes to canonical overlap if a canonical point
        # lies within its coordinate extent. This is geometry, not data validity.
        la = np.asarray(ds[lat].values).astype(float)
        lo = np.asarray(ds[lon].values).astype(float)
        if np.nanmin(lo) >= 0 and np.nanmax(lo) > 180:
            lo = ((lo + 180) % 360) - 180
        return {
            "status": "COMPLETE",
            "source_lat_points_in_domain": int(((la >= LAT_MIN) & (la <= LAT_MAX)).sum()),
            "source_lon_points_in_domain": int(((lo >= LON_MIN) & (lo <= LON_MAX)).sum()),
            "canonical_cells": int(LAT_SIZE * LON_SIZE),
            "canonical_lat_cells": int(np.sum(np.isin(CANON_LAT, la))),
            "canonical_lon_cells": int(np.sum(np.isin(CANON_LON, lo))),
            "source_extent_intersects": bool(
                np.nanmin(la) <= LAT_MAX and np.nanmax(la) >= LAT_MIN and
                np.nanmin(lo) <= LON_MAX and np.nanmax(lo) >= LON_MIN
            ),
            "variable_dims": list(v.dims),
        }
    except Exception as e:
        return {"status": "FAILED", "error": repr(e)}


# ---------------------------------------------------------------------------
# GLORYS target audit
# ---------------------------------------------------------------------------

def glorys_target_audit(files: list[Path]) -> dict[str, Any]:
    failures = []
    observations = []
    for p in files[:max(3, min(len(files), 5))]:
        ds = None
        try:
            ds = open_nc(p)
            sem = resolve_variables("glorys", ds)
            ev = sem.get("thetao")
            if not ev or not ev.get("variable"):
                observations.append({"file": str(p), "status": "FAIL", "reason": "thetao not resolved"})
                continue
            vname = ev["variable"]
            dn = depth_name(ds)
            if dn is None:
                observations.append({"file": str(p), "status": "FAIL", "reason": "depth coordinate not found"})
                continue
            depth = np.asarray(ds[dn].values).astype(float).reshape(-1)
            # Common ocean convention is positive downward; report actual sign.
            nearest = []
            for target in GLORYS_DEPTHS:
                i = int(np.argmin(np.abs(depth - target)))
                nearest.append({
                    "requested_m": float(target),
                    "nearest_source_m": float(depth[i]),
                    "abs_error_m": float(abs(depth[i] - target)),
                })
            exact = all(x["abs_error_m"] <= DEPTH_TOL for x in nearest)
            observations.append({
                "file": str(p),
                "thetao_variable": vname,
                "depth_name": dn,
                "depth_units": ds[dn].attrs.get("units"),
                "n_depths": int(depth.size),
                "min_depth": float(depth.min()),
                "max_depth": float(depth.max()),
                "ascending": bool(depth[-1] > depth[0]) if depth.size > 1 else None,
                "requested_depths_exactly_present": exact,
                "requested_depths": nearest,
            })
        except Exception as e:
            failures.append({"file": str(p), "stage": "glorys_depth", "error": repr(e)})
        finally:
            close_quiet(ds)
    statuses = [o.get("requested_depths_exactly_present") for o in observations]
    return {
        "observations": observations,
        "failures": failures,
        "exact_15_depths_verified": bool(statuses) and all(statuses),
        "status": "COMPLETE" if observations and not failures else ("PARTIAL" if observations else "FAILED"),
    }


# ---------------------------------------------------------------------------
# Numerical audit
# ---------------------------------------------------------------------------

def numerical_audit(cfg: Config, key: str, files: list[Path]) -> dict[str, Any]:
    """
    Full variable-level audit. To avoid misleading cross-file aggregation,
    variables are aggregated only by exact variable name AND exact dimensional
    signature. Per-file stats are preserved.
    """
    per_var = defaultdict(list)
    failures = []
    for i, p in enumerate(files):
        ds = None
        try:
            ds = open_nc(p)
            for vname in ds.data_vars:
                v = ds[vname]
                # Skip scalar metadata variables only if they have no data.
                seed = cfg.seed + (hash((key, vname)) & 0xFFFF)
                stats = robust_stats_for_dataarray(v, cfg, seed)
                per_var[(vname, tuple(v.dims), str(v.dtype))].append({
                    "file": str(p), **stats
                })
        except Exception as e:
            failures.append({"file": str(p), "stage": "numerical", "error": repr(e)})
        finally:
            close_quiet(ds)

    aggregate = {}
    for (vname, dims, dtype), rows in per_var.items():
        # Combine exact moments from finite values using numerically stable
        # two-pass over per-file means/stds.
        total_n = sum(r["finite_count"] for r in rows)
        invalid = sum(r["invalid_count"] for r in rows)
        if total_n:
            means = np.array([r["mean"] for r in rows if r["finite_count"]], dtype=float)
            ns = np.array([r["finite_count"] for r in rows if r["finite_count"]], dtype=float)
            vars_ = np.array([r["std"]**2 for r in rows if r["finite_count"]], dtype=float)
            grand_mean = float(np.sum(ns * means) / total_n)
            # pooled population variance
            pooled = np.sum(ns * (vars_ + (means - grand_mean)**2)) / total_n
            minv = min(r["min"] for r in rows if r["min"] is not None)
            maxv = max(r["max"] for r in rows if r["max"] is not None)
        else:
            grand_mean = pooled = minv = maxv = None
        aggregate[f"{vname}|{','.join(dims)}|{dtype}"] = {
            "variable": vname,
            "dimensions": list(dims),
            "dtype": dtype,
            "files_contributing": len(rows),
            "finite_count": int(total_n),
            "invalid_count": int(invalid),
            "valid_percent": float(100 * total_n / (total_n + invalid)) if total_n + invalid else None,
            "min": minv, "max": maxv, "mean": grand_mean,
            "std": float(math.sqrt(max(pooled, 0.0))) if pooled is not None else None,
            "per_file": rows,
            "note": "Quantiles are per-file deterministic samples; they are not pooled quantiles.",
        }
    return {
        "aggregate": aggregate,
        "failures": failures,
        "files_checked": len(files) - len(failures),
        "status": "COMPLETE" if not failures else "PARTIAL",
    }


# ---------------------------------------------------------------------------
# Coverage / masks
# ---------------------------------------------------------------------------

def coverage_audit(cfg: Config, key: str, files: list[Path]) -> dict[str, Any]:
    rows = []
    failures = []
    for p in files:
        ds = None
        try:
            ds = open_nc(p)
            tn = time_name(ds)
            times = to_time_objects(ds)
            dates = [date_key(x) for x in times] if times else [None]
            for vname in ds.data_vars:
                v = ds[vname]
                lat, lon = coord_name(ds, "lat"), coord_name(ds, "lon")
                if not lat or not lon or lat not in v.dims and lon not in v.dims:
                    continue
                # Slice only the canonical geographic region. Avoid assuming
                # coordinate ordering.
                la = np.asarray(ds[lat].values).astype(float)
                lo = np.asarray(ds[lon].values).astype(float)
                if np.nanmin(lo) >= 0 and np.nanmax(lo) > 180:
                    lo = ((lo + 180) % 360) - 180
                li = np.where((la >= LAT_MIN) & (la <= LAT_MAX))[0]
                oi = np.where((lo >= LON_MIN) & (lo <= LON_MAX))[0]
                if not li.size or not oi.size:
                    continue
                sub = v.isel({lat: li, lon: oi})
                arr = safe_dataarray_values(sub)
                finite = np.isfinite(arr)
                rows.append({
                    "dataset": key,
                    "file": str(p),
                    "date_first": dates[0] if dates else None,
                    "n_time": len(times),
                    "variable": vname,
                    "cells_examined": int(arr.size),
                    "valid": int(finite.sum()),
                    "invalid": int((~finite).sum()),
                    "valid_percent": float(100 * finite.mean()) if arr.size else None,
                })
        except Exception as e:
            failures.append({"file": str(p), "stage": "coverage", "error": repr(e)})
        finally:
            close_quiet(ds)
    return {
        "rows": rows,
        "failures": failures,
        "status": "COMPLETE" if not failures else "PARTIAL",
    }


# ---------------------------------------------------------------------------
# Exact temporal intersection
# ---------------------------------------------------------------------------

def temporal_intersection(dataset_results: dict[str, Any]) -> dict[str, Any]:
    date_sets = {}
    timestamp_sets = {}
    for key, result in dataset_results.items():
        t = result.get("temporal", {})
        if t.get("unique_dates"):
            date_sets[key] = set(t["unique_dates"])
        if t.get("unique_timestamps"):
            timestamp_sets[key] = set(t["unique_timestamps"])

    def intersection(sets: dict[str, set[str]]) -> list[str]:
        if not sets:
            return []
        return sorted(set.intersection(*sets.values()))

    common_dates_all = intersection(date_sets)
    common_dates_core_target = intersection({
        k: date_sets[k] for k in CORE_KEYS + [TARGET_KEY] if k in date_sets
    })
    common_ts_all = intersection(timestamp_sets)
    common_ts_core_target = intersection({
        k: timestamp_sets[k] for k in CORE_KEYS + [TARGET_KEY] if k in timestamp_sets
    })

    return {
        "all_dataset_common_dates": common_dates_all,
        "core_plus_target_common_dates": common_dates_core_target,
        "all_dataset_common_timestamps": common_ts_all,
        "core_plus_target_common_timestamps": common_ts_core_target,
        "date_counts": {k: len(v) for k, v in date_sets.items()},
        "timestamp_counts": {k: len(v) for k, v in timestamp_sets.items()},
        "important_note": "Date intersection is not physical-time equivalence; exact timestamp intersection is reported separately.",
    }


# ---------------------------------------------------------------------------
# Joint validity using actual intended variables
# ---------------------------------------------------------------------------

INTEGRATION_ROLE_MAP = {
    "oisst": ["sst"],
    "oisss": ["sss"],
    "duacs": ["sla", "adt", "ugos", "vgos"],
    "globecurrent": ["u", "v"],
    "ccmp": ["u_wind", "v_wind"],
    "glorys": ["thetao"],
}


def select_resolved_variable(dataset_key: str, ds: xr.Dataset, role: str) -> Optional[str]:
    ev = resolve_variables(dataset_key, ds).get(role)
    return ev.get("variable") if ev and ev.get("confidence") != "AMBIGUOUS" else None


def file_for_date(files: list[Path], date_str: str) -> Optional[Path]:
    target = pd.Timestamp(date_str).date()
    for p in files:
        ds = None
        try:
            ds = open_nc(p)
            if any(date_key(t) == str(target) for t in to_time_objects(ds)):
                return p
        except Exception:
            pass
        finally:
            close_quiet(ds)
    return None


def joint_validity(cfg: Config, files_by_key: dict[str, list[Path]], dates: list[str]) -> dict[str, Any]:
    required = [k for k in CORE_KEYS + [TARGET_KEY] if k in files_by_key]
    rows = []
    failures = []

    if not all(k in files_by_key and files_by_key[k] for k in required):
        return {"status": "INCONCLUSIVE", "reason": "not all required datasets are available"}

    chosen_dates = dates if cfg.joint_all else deterministic_sample(dates, cfg.joint_dates, cfg.seed)
    cache_map: dict[tuple[str, str], Optional[Path]] = {}

    for date_str in chosen_dates:
        masks: dict[str, np.ndarray] = {}
        missing = []
        for key in required:
            fp = file_for_date(files_by_key[key], date_str)
            cache_map[(key, date_str)] = fp
            if fp is None:
                missing.append(key)
                continue
            ds = None
            try:
                ds = open_nc(fp)
                role_list = INTEGRATION_ROLE_MAP[key]
                component_masks = []
                for role in role_list:
                    vname = select_resolved_variable(key, ds, role)
                    if vname is None:
                        continue
                    v = ds[vname]
                    # If multiple dimensions include time, choose the exact
                    # date/time slice rather than arbitrary index 0.
                    tn = time_name(ds)
                    if tn and tn in v.dims:
                        times = to_time_objects(ds)
                        idx = next((i for i, t in enumerate(times) if date_key(t) == date_str), None)
                        if idx is None:
                            continue
                        v = v.isel({tn: idx})
                    lat, lon = coord_name(ds, "lat"), coord_name(ds, "lon")
                    if not lat or not lon or lat not in v.dims or lon not in v.dims:
                        continue
                    r = canonical_interp(v, lat, lon, "linear")
                    component_masks.append(np.isfinite(safe_dataarray_values(r)))
                if not component_masks:
                    missing.append(key)
                else:
                    # All variables actually intended for integration must be
                    # jointly valid within a dataset.
                    masks[key] = np.logical_and.reduce(component_masks)
            except Exception as e:
                failures.append({"date": date_str, "dataset": key, "error": repr(e)})
                missing.append(key)
            finally:
                close_quiet(ds)

        if missing:
            rows.append({
                "date": date_str,
                "status": "INCOMPLETE",
                "missing_or_unresolved": ",".join(missing),
            })
            continue

        joint = np.logical_and.reduce([masks[k] for k in required])
        total = int(joint.size)
        row = {
            "date": date_str,
            "status": "COMPLETE",
            "total_canonical_cells": total,
            "joint_valid_cells": int(joint.sum()),
            "joint_valid_percent": float(100 * joint.mean()) if total else None,
        }
        for key, m in masks.items():
            row[f"{key}_valid_cells"] = int(m.sum())
            row[f"{key}_valid_percent"] = float(100 * m.mean())
        rows.append(row)

    return {
        "status": "COMPLETE" if rows else "INCONCLUSIVE",
        "dates_requested": chosen_dates,
        "rows": rows,
        "failures": failures,
        "definition": "A canonical cell is jointly valid only when every resolved variable intended for that dataset's integration role is finite after linear interpolation to the canonical grid.",
    }


# ---------------------------------------------------------------------------
# Regridding cross-validation
# ---------------------------------------------------------------------------

def spatial_holdout_cv(
    da: xr.DataArray,
    lat: str,
    lon: str,
    cfg: Config,
    seed: int,
) -> dict[str, Any]:
    """
    Genuine spatial holdout:
      1. select finite source-grid points;
      2. randomly hold out a deterministic subset;
      3. interpolate the remaining source points to held-out source coordinates;
      4. compare prediction to the actual held-out source values.

    This tests interpolation fidelity on the source field itself. It does not
    claim to be a validation of the ocean truth; it is a diagnostic of the
    interpolation operator and field smoothness.
    """
    x = da.squeeze(drop=True)
    # Only 2-D fields are admissible here. Time/depth must be sliced by caller.
    if lat not in x.dims or lon not in x.dims:
        return {"status": "INCONCLUSIVE", "reason": "spatial dimensions unavailable"}
    extra = [d for d in x.dims if d not in {lat, lon}]
    if extra:
        return {"status": "INCONCLUSIVE", "reason": f"unsliced dimensions: {extra}"}

    arr = safe_dataarray_values(x)
    if arr.ndim != 2:
        return {"status": "INCONCLUSIVE", "reason": f"expected 2-D, got {arr.ndim}-D"}

    la = np.asarray(x[lat].values).astype(float)
    lo = np.asarray(x[lon].values).astype(float)
    la_order = np.argsort(la)
    lo_order = np.argsort(lo)
    la = la[la_order]
    lo = lo[lo_order]
    arr = arr[np.ix_(la_order, lo_order)]

    valid = np.isfinite(arr)
    coords = np.argwhere(valid)
    if len(coords) < 20:
        return {"status": "INCONCLUSIVE", "reason": "too few finite source points", "finite_points": int(len(coords))}

    rng = np.random.default_rng(seed)
    n_hold = min(cfg.holdout_points, max(10, int(len(coords) * cfg.holdout_fraction)))
    hold_idx = rng.choice(len(coords), size=n_hold, replace=False)
    hold = coords[hold_idx]
    train_mask = valid.copy()
    train_mask[hold[:, 0], hold[:, 1]] = False

    # Use scattered-point interpolation so that a held-out source point is
    # genuinely absent from the training set. Interpolating a masked regular
    # grid with xarray at the exact held-out coordinate would otherwise return
    # the NaN at that coordinate and is not a real holdout.
    train_coords = coords[~np.isin(np.arange(len(coords)), hold_idx)]
    train_values = arr[train_coords[:, 0], train_coords[:, 1]]
    points = np.column_stack((la[train_coords[:, 0]], lo[train_coords[:, 1]]))
    query = np.column_stack((la[hold[:, 0]], lo[hold[:, 1]]))
    pred = griddata(points, train_values, query, method="linear")
    truth = arr[hold[:, 0], hold[:, 1]]
    ok = np.isfinite(pred) & np.isfinite(truth)
    if int(ok.sum()) < 10:
        return {
            "status": "INCONCLUSIVE",
            "reason": "insufficient interpolatable holdout points",
            "holdout_points": int(n_hold),
            "predicted_points": int(ok.sum()),
        }
    err = pred[ok] - truth[ok]
    return {
        "status": "COMPLETE",
        "holdout_points": int(n_hold),
        "predicted_points": int(ok.sum()),
        "prediction_coverage_percent": float(100 * ok.mean()),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "truth_std": float(np.std(truth[ok])),
        "normalized_rmse": float(np.sqrt(np.mean(err ** 2)) / np.std(truth[ok]))
        if np.std(truth[ok]) > 0 else None,
    }


def regridding_audit(cfg: Config, key: str, files: list[Path], glorys_files: list[Path]) -> dict[str, Any]:
    if not glorys_files:
        return {"dataset": key, "status": "INCONCLUSIVE", "reason": "GLORYS target grid unavailable"}

    chosen = deterministic_sample(files, cfg.regrid_files, cfg.seed)
    target = glorys_files[0]
    tds = None
    try:
        tds = open_nc(target)
        tla, tlo = coord_name(tds, "lat"), coord_name(tds, "lon")
        if not tla or not tlo:
            return {"dataset": key, "status": "FAILED", "reason": "GLORYS coordinates unavailable"}
        target_lat = np.asarray(tds[tla].values).astype(float)
        target_lon = np.asarray(tds[tlo].values).astype(float)
    finally:
        close_quiet(tds)

    records = []
    failures = []
    for p in chosen:
        ds = None
        try:
            ds = open_nc(p)
            sem = resolve_variables(key, ds)
            # Evaluate the variables actually relevant to the dataset, not an
            # arbitrary first variable.
            roles = INTEGRATION_ROLE_MAP.get(key, [])
            for role in roles:
                ev = sem.get(role)
                if not ev or not ev.get("variable"):
                    continue
                v = ds[ev["variable"]]
                tn = time_name(ds)
                if tn and tn in v.dims:
                    v = v.isel({tn: 0})
                dn = depth_name(ds)
                if dn and dn in v.dims:
                    v = v.isel({dn: 0})
                la, lo = coord_name(ds, "lat"), coord_name(ds, "lon")
                if not la or not lo:
                    continue
                cv = canonical_interp(v, la, lo, "linear")
                arr = safe_dataarray_values(cv)
                finite = arr[np.isfinite(arr)]
                holdout = spatial_holdout_cv(v, la, lo, cfg, cfg.seed + len(records) + 1)
                records.append({
                    "file": str(p),
                    "role": role,
                    "variable": ev["variable"],
                    "source_shape": list(v.shape),
                    "target_shape": [int(target_lat.size), int(target_lon.size)],
                    "target_grid_exact_canonical": bool(
                        target_lat.size == LAT_SIZE and target_lon.size == LON_SIZE
                        and np.allclose(target_lat, CANON_LAT, atol=1e-8)
                        and np.allclose(target_lon, CANON_LON, atol=1e-8)
                    ),
                    "canonical_valid_cells": int(np.isfinite(arr).sum()),
                    "canonical_valid_percent": float(100 * np.isfinite(arr).mean()),
                    "holdout": holdout,
                })
        except Exception as e:
            failures.append({"file": str(p), "stage": "regridding", "error": repr(e)})
        finally:
            close_quiet(ds)
    return {
        "dataset": key,
        "target_file": str(target),
        "method_for_production_candidate": "linear",
        "records": records,
        "failures": failures,
        "status": "COMPLETE" if records and not failures else ("PARTIAL" if records else "FAILED"),
        "scientific_note": "Holdout metrics measure interpolation self-consistency on withheld source-grid observations; they do not establish agreement with independent ocean observations.",
    }


# ---------------------------------------------------------------------------
# GlobCurrent / OSCAR redundancy
# ---------------------------------------------------------------------------

def comparable_current_semantics(ds: xr.Dataset, key: str) -> dict[str, Any]:
    sem = semantic_audit(key, ds)
    rows = {r["role"]: r for r in sem["variables"]}
    u, v = rows.get("u"), rows.get("v")
    if not u or not v:
        return {"comparable": False, "reason": "u/v unresolved", "semantic": sem}
    units_ok = u.get("unit_status") == "CONFIRMED" and v.get("unit_status") == "CONFIRMED"
    return {
        "comparable": units_ok,
        "u": u, "v": v,
        "semantic": sem,
        "reason": None if units_ok else "units require review",
    }


def current_redundancy(cfg: Config, files_gc: list[Path], files_osc: list[Path]) -> dict[str, Any]:
    gc_dates = set()
    osc_dates = set()
    for key, files in (("globecurrent", files_gc), ("oscar", files_osc)):
        for p in files:
            ds = None
            try:
                ds = open_nc(p)
                s = gc_dates if key == "globecurrent" else osc_dates
                s.update(date_key(t) for t in to_time_objects(ds))
            except Exception:
                pass
            finally:
                close_quiet(ds)
    common = sorted(gc_dates & osc_dates)
    chosen = deterministic_sample(common, cfg.redundancy_dates, cfg.seed)
    rows, failures = [], []

    for date_str in chosen:
        gfp = file_for_date(files_gc, date_str)
        ofp = file_for_date(files_osc, date_str)
        if not gfp or not ofp:
            continue
        gds = ods = None
        try:
            gds, ods = open_nc(gfp), open_nc(ofp)
            gs = comparable_current_semantics(gds, "globecurrent")
            os_ = comparable_current_semantics(ods, "oscar")
            if not gs["comparable"] or not os_["comparable"]:
                rows.append({
                    "date": date_str, "status": "INCOMPARABLE",
                    "reason": f"GlobCurrent={gs['reason']}; OSCAR={os_['reason']}",
                })
                continue

            gu = gds[gs["u"]["variable"]]
            gv = gds[gs["v"]["variable"]]
            ou = ods[os_["u"]["variable"]]
            ov = ods[os_["v"]["variable"]]
            for v in ("gu", "gv", "ou", "ov"):
                pass

            def slice_date(x: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
                tn = time_name(ds)
                if tn and tn in x.dims:
                    times = to_time_objects(ds)
                    idx = next((i for i, t in enumerate(times) if date_key(t) == date_str), None)
                    if idx is None:
                        raise ValueError("date not present in selected file")
                    return x.isel({tn: idx})
                return x

            gu, gv = slice_date(gu, gds), slice_date(gv, gds)
            ou, ov = slice_date(ou, ods), slice_date(ov, ods)

            glat, glon = coord_name(gds, "lat"), coord_name(gds, "lon")
            olat, olon = coord_name(ods, "lat"), coord_name(ods, "lon")
            if not all([glat, glon, olat, olon]):
                raise ValueError("missing coordinates")

            gu = canonical_interp(gu, glat, glon)
            gv = canonical_interp(gv, glat, glon)
            ou = canonical_interp(ou, olat, olon)
            ov = canonical_interp(ov, olat, olon)

            a, b, c, d = [safe_dataarray_values(x) for x in (gu, gv, ou, ov)]
            mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
            if int(mask.sum()) < 10:
                rows.append({"date": date_str, "status": "INSUFFICIENT_VALID_CELLS",
                             "valid_cells": int(mask.sum())})
                continue

            a, b, c, d = a[mask], b[mask], c[mask], d[mask]
            speed_g, speed_o = np.hypot(a, b), np.hypot(c, d)

            def corr(x, y):
                if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                    return None
                return float(np.corrcoef(x, y)[0, 1])

            dir_g, dir_o = np.arctan2(b, a), np.arctan2(d, c)
            angle = np.abs(np.angle(np.exp(1j * (dir_o - dir_g))))

            rows.append({
                "date": date_str, "status": "COMPLETE", "valid_cells": int(mask.sum()),
                "u_bias_oscar_minus_globcurrent": float(np.mean(c - a)),
                "u_mae": float(np.mean(np.abs(c - a))),
                "u_rmse": float(np.sqrt(np.mean((c - a) ** 2))),
                "u_corr": corr(a, c),
                "v_bias_oscar_minus_globcurrent": float(np.mean(d - b)),
                "v_mae": float(np.mean(np.abs(d - b))),
                "v_rmse": float(np.sqrt(np.mean((d - b) ** 2))),
                "v_corr": corr(b, d),
                "speed_bias_oscar_minus_globcurrent": float(np.mean(speed_o - speed_g)),
                "speed_mae": float(np.mean(np.abs(speed_o - speed_g))),
                "speed_rmse": float(np.sqrt(np.mean((speed_o - speed_g) ** 2))),
                "speed_corr": corr(speed_g, speed_o),
                "mean_direction_difference_rad": float(np.mean(angle)),
            })
        except Exception as e:
            failures.append({"date": date_str, "error": repr(e)})
        finally:
            close_quiet(gds)
            close_quiet(ods)

    return {
        "status": "COMPLETE" if rows else "INCONCLUSIVE",
        "common_dates": common,
        "sample_dates": chosen,
        "rows": rows,
        "failures": failures,
        "interpretation_rule": "High correlation/low error supports redundancy only after physical semantics, units and temporal meaning are shown comparable.",
    }


# ---------------------------------------------------------------------------
# Evidence-derived integration recommendation
# ---------------------------------------------------------------------------

def recommendation_for_dataset(key: str, result: dict[str, Any]) -> dict[str, Any]:
    prior = REGISTRY[key]["prior"]
    evidence = []
    score = 0
    blockers = []

    if result.get("files_found", 0) == 0:
        return {"dataset": key, "recommendation": "EXCLUDE", "confidence": "HIGH",
                "score": 0, "evidence": ["No files found"], "blockers": ["No data"]}

    evidence.append(f"{result['files_found']} files discovered")
    if result.get("schema", {}).get("status") == "COMPLETE":
        score += 1; evidence.append("schema audit complete")
    else:
        blockers.append("schema audit incomplete")

    if result.get("temporal", {}).get("status") == "COMPLETE":
        score += 1; evidence.append("temporal audit complete")
    else:
        blockers.append("temporal audit incomplete")

    if result.get("spatial", {}).get("status") == "COMPLETE":
        score += 1; evidence.append("spatial geometry audited")
    else:
        blockers.append("spatial geometry unresolved")

    sem = result.get("semantic", {})
    if sem.get("semantic_status") == "CONFIRMED":
        score += 2; evidence.append("resolved variables have high-confidence semantics and confirmed units")
    else:
        blockers.append("physical semantics/units require review")

    if key == "glorys":
        gd = result.get("glorys_depths", {})
        if gd.get("exact_15_depths_verified"):
            score += 3; evidence.append("exact 15 requested target depths verified")
        else:
            blockers.append("exact 15 GLORYS target depths not verified")
        rec = "TARGET" if not blockers else "INCONCLUSIVE"
    elif key == "argo":
        rec = "VALIDATION_ONLY" if not blockers else "INCONCLUSIVE"
    elif key == "oscar":
        rec = "VALIDATION_ONLY" if not blockers else "INCONCLUSIVE"
    else:
        # A core prior can be retained only if audit evidence does not expose a blocker.
        if not blockers and score >= 5:
            rec = "CORE_INPUT"
        elif score >= 3:
            rec = "OPTIONAL_INPUT"
        else:
            rec = "INCONCLUSIVE"

    confidence = "HIGH" if rec in {"CORE_INPUT", "TARGET", "VALIDATION_ONLY"} and not blockers else (
        "MEDIUM" if rec == "OPTIONAL_INPUT" else "LOW"
    )
    return {
        "dataset": key,
        "recommendation": rec,
        "confidence": confidence,
        "score": score,
        "prior_registry_role": prior,
        "evidence": evidence,
        "blockers": blockers,
    }


def derive_recommendations(dataset_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [recommendation_for_dataset(k, v) for k, v in dataset_results.items()]

    # Specific cross-dataset evidence: if current sources are demonstrably
    # redundant, do not silently discard one; mark the secondary as ABLATION
    # candidate only when the comparison actually ran and supports redundancy.
    red = dataset_results.get("_cross", {}).get("redundancy", {})
    complete_red = [r for r in red.get("rows", []) if r.get("status") == "COMPLETE"]
    if len(complete_red) >= 1:
        gc = next((r for r in rows if r["dataset"] == "globecurrent"), None)
        os_ = next((r for r in rows if r["dataset"] == "oscar"), None)
        if gc and os_ and gc["recommendation"] == "CORE_INPUT" and os_["recommendation"] == "VALIDATION_ONLY":
            os_["cross_dataset_note"] = "Independent current-source comparison completed; retain for validation/ablation, not as automatic core input."
    return rows


# ---------------------------------------------------------------------------
# Completeness / report
# ---------------------------------------------------------------------------

def completeness(result: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    cats = {
        "file_index": result.get("files_found") is not None,
        "schema": result.get("schema", {}).get("status") == "COMPLETE",
        "temporal": result.get("temporal", {}).get("status") == "COMPLETE",
        "spatial": result.get("spatial", {}).get("status") == "COMPLETE",
        "semantic": result.get("semantic", {}).get("semantic_status") == "CONFIRMED",
        "numerical": result.get("numerical", {}).get("status") == "COMPLETE" if cfg.full else False,
        "coverage": result.get("coverage", {}).get("status") == "COMPLETE" if cfg.full else False,
        "target_depths": result.get("glorys_depths", {}).get("status") == "COMPLETE" if result["dataset_key"] == "glorys" else None,
    }
    rows = []
    for cat, ok in cats.items():
        if ok is True:
            status = "COMPLETE"
        elif ok is False:
            status = "PARTIAL" if cfg.full else "NOT_RUN"
        else:
            status = "NOT_APPLICABLE"
        rows.append({
            "dataset": result["dataset_key"],
            "category": cat,
            "status": status,
            "scope": "FULL" if cfg.full else "QUICK",
        })
    return rows


def deterministic_sample(items: Sequence[Any], n: int, seed: int) -> list[Any]:
    items = list(items)
    if n <= 0 or n >= len(items):
        return items
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(items), size=n, replace=False))
    return [items[int(i)] for i in idx]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def generate_report(cfg: Config, dataset_results: dict[str, Any], cross: dict[str, Any]) -> None:
    out = cfg.output
    out.mkdir(parents=True, exist_ok=True)

    inventory = []
    completeness_rows = []
    recommendations = []

    for key, result in dataset_results.items():
        if key.startswith("_"):
            continue
        inventory.append({
            "dataset": key,
            "name": REGISTRY[key]["name"],
            "files_found": result.get("files_found", 0),
            "files_audited": result.get("files_audited", 0),
            "schema_status": result.get("schema", {}).get("status"),
            "temporal_status": result.get("temporal", {}).get("status"),
            "spatial_status": result.get("spatial", {}).get("status"),
            "semantic_status": result.get("semantic", {}).get("semantic_status"),
        })
        completeness_rows.extend(completeness(result, cfg))
        recommendations.append(recommendation_for_dataset(key, result))

        write_json(out / "datasets" / f"{key}.json", result)

    write_csv(out / "dataset_inventory.csv", inventory)
    write_csv(out / "audit_completeness.csv", completeness_rows)
    write_csv(out / "integration_recommendation.csv", recommendations)

    temporal = cross.get("temporal_intersection", {})
    write_json(out / "temporal_intersection.json", temporal)
    write_json(out / "joint_validity.json", cross.get("joint_validity", {}))
    write_json(out / "regridding.json", cross.get("regridding", {}))
    write_json(out / "redundancy.json", cross.get("redundancy", {}))

    summary_lines = [
        "# OceanEmbed Dataset Forensic Audit",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Timestamp UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Raw root: `{cfg.root}`",
        f"- Output: `{cfg.output}`",
        "",
        "## Executive result",
        "",
    ]
    for r in recommendations:
        summary_lines.append(
            f"- **{r['dataset']}** → **{r['recommendation']}** "
            f"(confidence {r['confidence']}, score {r['score']})"
        )
        if r["blockers"]:
            summary_lines.append(f"  - Blockers: {'; '.join(r['blockers'])}")

    summary_lines += [
        "",
        "## Temporal compatibility",
        f"- Core + target common dates: `{len(temporal.get('core_plus_target_common_dates', []))}`",
        f"- Core + target exact common timestamps: `{len(temporal.get('core_plus_target_common_timestamps', []))}`",
        "",
        "## GLORYS target",
        f"- Exact requested 15 depths verified: `{dataset_results.get('glorys', {}).get('glorys_depths', {}).get('exact_15_depths_verified')}`",
        "",
        "## Scientific caution",
        "- Date overlap does not imply physical-time equivalence.",
        "- Interpolation holdout error diagnoses the interpolation operator; it is not independent ocean-observation validation.",
        "- Integration roles are evidence-derived and may remain INCONCLUSIVE when semantics, coverage or target compatibility are unresolved.",
        "- Raw NetCDF inputs are opened read-only and are never rewritten.",
    ]
    (out / "audit_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    write_json(out / "audit_run.json", {
        "audit_version": AUDIT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "canonical_grid": canonical_grid(),
        "dataset_count": len([k for k in dataset_results if not k.startswith("_")]),
        "recommendations": recommendations,
    })


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def audit_dataset(cfg: Config, key: str, files: list[Path], cache: Cache) -> dict[str, Any]:
    LOGGER.info("[%s] %d files", REGISTRY[key]["name"], len(files))
    result: dict[str, Any] = {
        "dataset_key": key,
        "dataset_name": REGISTRY[key]["name"],
        "purpose": REGISTRY[key]["purpose"],
        "files_found": len(files),
        "files_audited": len(files),
        "raw_root": str(cfg.root / REGISTRY[key]["root"]),
    }

    idx = file_index(cfg, key, files)
    write_csv(cfg.output / "indexes" / f"file_index_{key}.csv", idx)

    # Schema and time always run across all files in FULL mode.
    result["schema"] = schema_audit(cfg, key, files, cache)
    result["temporal"] = temporal_audit(cfg, key, files)

    # Representative metadata: use first successfully opened file.
    meta = None
    for p in files:
        ds = None
        try:
            ds = open_nc(p)
            meta = metadata_snapshot(ds, p)
            break
        except Exception:
            pass
        finally:
            close_quiet(ds)
    result["representative_metadata"] = meta

    result["spatial"] = spatial_audit(cfg, key, files)

    # Semantic evidence is evaluated on representative + sampled files.
    semantic_records = []
    sem_fail = []
    for p in deterministic_sample(files, min(cfg.sample_files, len(files)), cfg.seed):
        ds = None
        try:
            ds = open_nc(p)
            semantic_records.append({"file": str(p), **semantic_audit(key, ds)})
        except Exception as e:
            sem_fail.append({"file": str(p), "error": repr(e)})
        finally:
            close_quiet(ds)
    result["semantic"] = {
        "sample_records": semantic_records,
        "failures": sem_fail,
        "semantic_status": (
            "CONFIRMED"
            if semantic_records and all(r["semantic_status"] == "CONFIRMED" for r in semantic_records)
            else "REQUIRES_REVIEW"
        ),
    }

    if cfg.full:
        result["numerical"] = numerical_audit(cfg, key, files)
        result["coverage"] = coverage_audit(cfg, key, files)
    else:
        result["numerical"] = {"status": "NOT_RUN"}
        result["coverage"] = {"status": "NOT_RUN"}

    if key == "glorys":
        result["glorys_depths"] = glorys_target_audit(files)
    return result


def run(cfg: Config) -> int:
    cfg.validate()
    setup_logging(cfg.verbose)
    cfg.output.mkdir(parents=True, exist_ok=True)
    for sub in ("datasets", "indexes", "cache"):
        (cfg.output / sub).mkdir(parents=True, exist_ok=True)

    LOGGER.info("OceanEmbed Forensic Audit %s", AUDIT_VERSION)
    LOGGER.info("RAW INPUTS ARE READ-ONLY")
    LOGGER.info("Root: %s", cfg.root)
    LOGGER.info("Output: %s", cfg.output)

    requested = list(REGISTRY) if "all" in cfg.datasets else [x.strip() for x in cfg.datasets]
    requested = [x for x in requested if x in REGISTRY]
    if not cfg.include_argo:
        requested = [x for x in requested if x != "argo"]
    if not requested:
        LOGGER.error("No valid datasets selected.")
        return 2

    files_by_key = {k: discover(cfg, k) for k in requested}
    cache = Cache(cfg)
    dataset_results = {}
    for key in requested:
        if not files_by_key[key]:
            dataset_results[key] = {
                "dataset_key": key, "dataset_name": REGISTRY[key]["name"],
                "files_found": 0, "files_audited": 0,
                "status": "NO_FILES",
                "purpose": REGISTRY[key]["purpose"],
            }
            LOGGER.warning("[%s] NO FILES", key)
            continue
        dataset_results[key] = audit_dataset(cfg, key, files_by_key[key], cache)

    cross = {}
    cross["temporal_intersection"] = temporal_intersection(dataset_results)

    # Joint validity is only meaningful when all core + target data exist.
    if cfg.full:
        common = cross["temporal_intersection"].get("core_plus_target_common_dates", [])
        cross["joint_validity"] = joint_validity(cfg, files_by_key, common)
    else:
        cross["joint_validity"] = {"status": "NOT_RUN"}

    # Regridding diagnostics.
    cross["regridding"] = {}
    if cfg.full and "glorys" in files_by_key and files_by_key["glorys"]:
        for key in CORE_KEYS:
            if key in files_by_key and files_by_key[key]:
                cross["regridding"][key] = regridding_audit(
                    cfg, key, files_by_key[key], files_by_key["glorys"]
                )
    else:
        cross["regridding"] = {"status": "NOT_RUN"}

    if cfg.full and "globecurrent" in files_by_key and "oscar" in files_by_key:
        cross["redundancy"] = current_redundancy(
            cfg, files_by_key["globecurrent"], files_by_key["oscar"]
        )
    else:
        cross["redundancy"] = {"status": "NOT_RUN"}

    dataset_results["_cross"] = cross
    generate_report(cfg, dataset_results, cross)

    # Final machine-readable gate.
    recs = derive_recommendations(dataset_results)
    gate = {
        "audit_version": AUDIT_VERSION,
        "final": True,
        "raw_read_only": True,
        "datasets": {k: {
            "files_found": v.get("files_found", 0),
            "schema": v.get("schema", {}).get("status"),
            "temporal": v.get("temporal", {}).get("status"),
            "spatial": v.get("spatial", {}).get("status"),
            "semantic": v.get("semantic", {}).get("semantic_status"),
        } for k, v in dataset_results.items() if not k.startswith("_")},
        "recommendations": recs,
        "gate_rules": [
            "No registry-only role is accepted as scientific evidence.",
            "GLORYS must verify all requested depths before TARGET is considered high confidence.",
            "Regridding holdout must be a genuine withheld-source-point test.",
            "Exact timestamps and calendar-aware date coverage are reported separately.",
            "Incomplete audit stages are never labelled COMPLETE.",
        ],
    }
    write_json(cfg.output / "final_gate.json", gate)

    LOGGER.info("AUDIT COMPLETE")
    LOGGER.info("Reports: %s", cfg.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OceanEmbed final scientific forensic audit; raw NetCDF inputs are never modified."
    )
    p.add_argument("--root", type=Path, default=Path(r"D:\OceanEmbed\oceanembed-data\raw"))
    p.add_argument("--output", type=Path, default=Path(r"D:\OceanEmbed\oceanembed-data\audit_final"))
    p.add_argument("--datasets", default="all", help="comma-separated dataset keys or all")
    p.add_argument("--quick", action="store_true", help="metadata/coverage diagnostics only; does not claim FULL audit")
    p.add_argument("--full", action="store_true", help="run the complete audit (default)")
    p.add_argument("--include-argo", action="store_true")
    p.add_argument("--include-argo-full", action="store_true")
    p.add_argument("--checksums", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--force", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sample-files", type=int, default=10)
    p.add_argument("--quantile-sample", type=int, default=250000)
    p.add_argument("--joint-dates", type=int, default=10)
    p.add_argument("--joint-all", action="store_true")
    p.add_argument("--redundancy-dates", type=int, default=10)
    p.add_argument("--regrid-files", type=int, default=5)
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    p.add_argument("--holdout-points", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.quick and args.full:
        parser.error("--quick and --full are mutually exclusive")
    full = not args.quick if not args.full else True
    if args.include_argo_full:
        args.include_argo = True
    cfg = Config(
        root=args.root,
        output=args.output,
        datasets=args.datasets.split(","),
        full=full,
        include_argo=args.include_argo,
        include_argo_full=args.include_argo_full,
        checksums=args.checksums,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        workers=args.workers,
        sample_files=args.sample_files,
        quantile_sample=args.quantile_sample,
        joint_dates=args.joint_dates,
        joint_all=args.joint_all,
        redundancy_dates=args.redundancy_dates,
        regrid_files=args.regrid_files,
        holdout_fraction=args.holdout_fraction,
        holdout_points=args.holdout_points,
        seed=args.seed,
    )
    try:
        return run(cfg)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted.")
        return 130
    except Exception:
        LOGGER.error("Fatal audit failure:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
