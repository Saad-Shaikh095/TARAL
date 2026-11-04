#!/usr/bin/env python3
"""
OceanEmbed Forensic Audit Engine - STAGED / FAST / EVIDENCE-FIRST

Stage 1
-------
One NetCDF open per file. Collects:
- file inventory
- schema
- CF time semantics, dates and exact timestamps
- grid geometry
- dataset metadata/provenance fields
- explicit variable semantics and units
- exact GLORYS 15-depth verification
No cross-dataset work is done here.

Stage "stats"
------------
One NetCDF open per file. Reads ONLY variables actually intended for
integration (and only the requested GLORYS depths). Produces finite-value
statistics and invalid/valid counts.

Stage 2
-------
Uses Stage-1 indexes and opens only selected files for:
- exact date/timestamp intersection
- canonical-grid regridding diagnostics
- genuine source-point spatial holdout
- joint validity
- GlobCurrent vs OSCAR comparison

Argo is intentionally handled separately because it is validation data with
a very different profile/file structure and scale.

Raw inputs are read-only.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata


VERSION = "STAGED-2.0"

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RES = 0.25
LAT_SIZE, LON_SIZE = 101, 241

CANON_LAT = np.round(np.arange(LAT_MIN, LAT_MAX + RES / 2, RES), 10)
CANON_LON = np.round(np.arange(LON_MIN, LON_MAX + RES / 2, RES), 10)

GLORYS_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    dtype=float,
)
DEPTH_TOL = 1e-6

CORE = ["oisst", "oisss", "duacs", "globecurrent", "ccmp"]
TARGET = "glorys"

REGISTRY = {
    "oisst": {"name": "OISST", "root": "oisst", "prior": "CORE_INPUT"},
    "oisss": {"name": "OISSS", "root": "oisss", "prior": "CORE_INPUT"},
    "duacs": {"name": "DUACS", "root": "duacs", "prior": "CORE_INPUT"},
    "globecurrent": {"name": "GlobCurrent", "root": "globecurrent", "prior": "CORE_INPUT"},
    "ccmp": {"name": "CCMP", "root": "ccmp", "prior": "CORE_INPUT"},
    "glorys": {"name": "GLORYS", "root": "glorys", "prior": "TARGET"},
    "oscar": {"name": "OSCAR", "root": "oscar", "prior": "VALIDATION_ONLY"},
}

ROLES = {
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
}

FALLBACKS = {
    "oisst": {"sst": ["sst", "analysed_sst", "sea_surface_temperature"]},
    "oisss": {"sss": ["sss", "salinity", "sea_surface_salinity"]},
    "glorys": {"thetao": ["thetao", "potential_temperature", "theta", "votemper"]},
    "globecurrent": {
        "u": ["uo", "ugos", "u", "eastward_velocity"],
        "v": ["vo", "vgos", "v", "northward_velocity"],
    },
    "oscar": {
        "u": ["u", "u_current", "ugos"],
        "v": ["v", "v_current", "vgos"],
    },
    "ccmp": {
        "u_wind": ["uwnd", "u_wind", "u"],
        "v_wind": ["vwnd", "v_wind", "v"],
        "wind_speed": ["ws", "wind_speed", "windspeed"],
    },
}

INTEGRATION = {
    "oisst": ["sst"],
    "oisss": ["sss"],
    "duacs": ["sla", "adt", "ugos", "vgos"],
    "globecurrent": ["u", "v"],
    "ccmp": ["u_wind", "v_wind"],
    "glorys": ["thetao"],
}

LOG = logging.getLogger("oceanembed.audit")


@dataclass
class Config:
    root: Path
    output: Path
    datasets: list[str]
    stage: str = "1"
    stats: bool = False
    sample_cross: int = 10
    joint_all: bool = False
    holdout_points: int = 2000
    holdout_fraction: float = 0.20
    seed: int = 42
    progress_every: int = 100


def setup_log(verbose: bool = False) -> None:
    LOG.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)


def json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, default=json_default, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def norm(x: Any) -> str:
    return str(x or "").strip().lower()


def open_nc(path: Path) -> xr.Dataset:
    # Current xarray API. Do NOT use use_cftime=True as a top-level kwarg.
    return xr.open_dataset(
        path,
        engine="netcdf4",
        decode_cf=True,
        mask_and_scale=True,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        chunks=None,
        cache=False,
    )


def close(ds: xr.Dataset | None) -> None:
    if ds is not None:
        try:
            ds.close()
        except Exception:
            pass


def coord(ds: xr.Dataset, kind: str) -> str | None:
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if kind == "lat" and (low in {"lat", "latitude"} or "latitude" in low):
            return str(name)
        if kind == "lon" and (low in {"lon", "longitude"} or "longitude" in low):
            return str(name)
    return None


def depth_name(ds: xr.Dataset) -> str | None:
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if low in {"depth", "deptht", "depthu", "depthv", "lev", "level"} or "depth" in low:
            return str(name)
    return None


def time_name(ds: xr.Dataset) -> str | None:
    for name in list(ds.coords) + list(ds.dims):
        low = str(name).lower()
        if low == "time" or "time" in low:
            return str(name)
    return None


def times(ds: xr.Dataset) -> list[Any]:
    name = time_name(ds)
    if name is None:
        return []
    return list(np.asarray(ds[name].values).reshape(-1))


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


def axis_info(a: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(a.values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("empty coordinate")
    diffs = np.diff(values)
    return {
        "size": int(values.size),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "spacing_median": float(np.nanmedian(np.abs(diffs))) if diffs.size else None,
        "spacing_min": float(np.nanmin(np.abs(diffs))) if diffs.size else None,
        "spacing_max": float(np.nanmax(np.abs(diffs))) if diffs.size else None,
        "uniform": bool(np.allclose(diffs, diffs[0], rtol=1e-8, atol=1e-10))
        if diffs.size > 1
        else None,
        "ascending": bool(values[-1] > values[0]) if values.size > 1 else None,
        "descending": bool(values[-1] < values[0]) if values.size > 1 else None,
        "values": values.tolist(),
    }


def grid_info(ds: xr.Dataset) -> dict[str, Any]:
    lat_name = coord(ds, "lat")
    lon_name = coord(ds, "lon")
    if not lat_name or not lon_name:
        return {"status": "INCONCLUSIVE", "reason": "latitude/longitude not found"}

    try:
        lat = np.asarray(ds[lat_name].values, dtype=float).reshape(-1)
        lon = np.asarray(ds[lon_name].values, dtype=float).reshape(-1)
        lon_norm = lon.copy()
        if np.nanmin(lon_norm) >= 0 and np.nanmax(lon_norm) > 180:
            lon_norm = ((lon_norm + 180) % 360) - 180

        return {
            "status": "COMPLETE",
            "lat_name": lat_name,
            "lon_name": lon_name,
            "lat": axis_info(ds[lat_name]),
            "lon": axis_info(ds[lon_name]),
            "normalized_lon_range": [
                float(np.nanmin(lon_norm)),
                float(np.nanmax(lon_norm)),
            ],
            "canonical_exact": bool(
                lat.size == LAT_SIZE
                and lon.size == LON_SIZE
                and np.allclose(lat, CANON_LAT, atol=1e-8)
                and np.allclose(lon_norm, CANON_LON, atol=1e-8)
            ),
            "canonical_lat_points_in_domain": int(
                ((lat >= LAT_MIN) & (lat <= LAT_MAX)).sum()
            ),
            "canonical_lon_points_in_domain": int(
                ((lon_norm >= LON_MIN) & (lon_norm <= LON_MAX)).sum()
            ),
            "source_extent_intersects_domain": bool(
                np.nanmin(lat) <= LAT_MAX
                and np.nanmax(lat) >= LAT_MIN
                and np.nanmin(lon_norm) <= LON_MAX
                and np.nanmax(lon_norm) >= LON_MIN
            ),
        }
    except Exception as exc:
        return {"status": "FAILED", "error": repr(exc)}


def resolve_variables(dataset_key: str, ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for role, required_standard_name in ROLES.get(dataset_key, []):
        exact = [
            v
            for v in ds.data_vars
            if required_standard_name in norm(ds[v].attrs.get("standard_name"))
        ]
        if len(exact) == 1:
            result[role] = {
                "variable": exact[0],
                "basis": "standard_name",
                "confidence": "HIGH",
            }
        elif len(exact) > 1:
            result[role] = {
                "variable": None,
                "basis": "ambiguous_standard_name",
                "candidates": exact,
                "confidence": "AMBIGUOUS",
            }

    for role, names in FALLBACKS.get(dataset_key, {}).items():
        if role in result:
            continue
        candidates = []
        lat_name = coord(ds, "lat")
        lon_name = coord(ds, "lon")

        for variable in ds.data_vars:
            low = str(variable).lower()
            if any(low == name or name in low for name in names):
                if (
                    lat_name
                    and lon_name
                    and lat_name in ds[variable].dims
                    and lon_name in ds[variable].dims
                ):
                    candidates.append(variable)

        if len(candidates) == 1:
            result[role] = {
                "variable": candidates[0],
                "basis": "name_fallback",
                "confidence": "MEDIUM",
            }
        elif len(candidates) > 1:
            result[role] = {
                "variable": None,
                "basis": "ambiguous_name_fallback",
                "candidates": candidates,
                "confidence": "AMBIGUOUS",
            }

    return result


def expected_units(role: str) -> set[str]:
    return {
        "sst": {"c", "degc", "degree_celsius", "k", "kelvin"},
        "thetao": {"c", "degc", "degree_celsius", "k", "kelvin"},
        "sss": {"1", "psu", "1e-3", "practical_salinity_unit"},
        "sla": {"m", "meter", "meters"},
        "adt": {"m", "meter", "meters"},
        "ugos": {"m/s", "m s-1", "m s^-1", "meter second-1"},
        "vgos": {"m/s", "m s-1", "m s^-1", "meter second-1"},
        "u": {"m/s", "m s-1", "m s^-1", "meter second-1", "cm/s"},
        "v": {"m/s", "m s-1", "m s^-1", "meter second-1", "cm/s"},
        "uek": {"m/s", "m s-1", "m s^-1", "meter second-1"},
        "vek": {"m/s", "m s-1", "m s^-1", "meter second-1"},
        "u_wind": {"m/s", "m s-1", "m s^-1"},
        "v_wind": {"m/s", "m s-1", "m s^-1"},
        "wind_speed": {"m/s", "m s-1", "m s^-1"},
    }.get(role, set())


def semantic_info(dataset_key: str, ds: xr.Dataset) -> dict[str, Any]:
    resolved = resolve_variables(dataset_key, ds)
    rows = []
    missing_roles = []

    for role, _ in ROLES.get(dataset_key, []):
        evidence = resolved.get(role)
        if not evidence or not evidence.get("variable"):
            missing_roles.append(role)
            rows.append(
                {
                    "role": role,
                    **(
                        evidence
                        if evidence
                        else {
                            "variable": None,
                            "confidence": "UNRESOLVED",
                        }
                    ),
                }
            )
            continue

        variable = evidence["variable"]
        da = ds[variable]
        units = da.attrs.get("units")
        unit_norm = norm(units)
        expected = expected_units(role)

        standard_name = norm(da.attrs.get("standard_name"))
        sign = "UNKNOWN"
        if "eastward" in standard_name:
            sign = "positive_eastward"
        elif "northward" in standard_name:
            sign = "positive_northward"
        elif "westward" in standard_name:
            sign = "positive_westward"
        elif "southward" in standard_name:
            sign = "positive_southward"

        rows.append(
            {
                "role": role,
                "variable": variable,
                "basis": evidence["basis"],
                "confidence": evidence["confidence"],
                "standard_name": da.attrs.get("standard_name"),
                "long_name": da.attrs.get("long_name"),
                "units": units,
                "unit_status": (
                    "MISSING"
                    if units is None
                    else ("CONFIRMED" if unit_norm in expected else "REQUIRES_REVIEW")
                ),
                "sign_convention": sign,
                "dimensions": list(da.dims),
                "dtype": str(da.dtype),
            }
        )

    confirmed = (
        not missing_roles
        and bool(rows)
        and all(
            row.get("confidence") == "HIGH"
            and row.get("unit_status") == "CONFIRMED"
            for row in rows
        )
    )

    return {
        "resolved": resolved,
        "variables": rows,
        "missing_roles": missing_roles,
        "semantic_status": "CONFIRMED" if confirmed else "REQUIRES_REVIEW",
    }


def time_info(ds: xr.Dataset) -> dict[str, Any]:
    values = times(ds)
    timestamps = [time_key(x) for x in values]
    dates = [date_key(x) for x in values]

    result: dict[str, Any] = {
        "n": len(values),
        "first": timestamps[0] if timestamps else None,
        "last": timestamps[-1] if timestamps else None,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "timestamps": timestamps,
        "dates": dates,
    }

    if len(values) >= 2:
        diffs = []
        for i in range(len(values) - 1):
            try:
                diffs.append(float((values[i + 1] - values[i]) / np.timedelta64(1, "s")))
            except Exception:
                try:
                    diffs.append(float((values[i + 1] - values[i]).total_seconds()))
                except Exception:
                    pass

        if diffs:
            result.update(
                timestep_seconds_median=float(np.median(diffs)),
                timestep_seconds_min=float(np.min(diffs)),
                timestep_seconds_max=float(np.max(diffs)),
                regular=bool(
                    np.allclose(diffs, diffs[0], rtol=1e-9, atol=1e-6)
                ),
            )

    name = time_name(ds)
    if name:
        result["calendar"] = ds[name].encoding.get(
            "calendar", ds[name].attrs.get("calendar", "standard")
        )
        result["units"] = ds[name].encoding.get(
            "units", ds[name].attrs.get("units")
        )
        result["cell_methods"] = ds[name].attrs.get("cell_methods")

    tokens = re.findall(
        r"\b\d{8}\b|\b\d{4}[-_]\d{2}[-_]\d{2}\b|\b\d{4}[-_]\d{2}\b",
        "",
    )
    del tokens  # Filename consistency is evaluated in audit_one.

    return result


def schema_signature(ds: xr.Dataset) -> dict[str, Any]:
    return {
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
        "coords": sorted(map(str, ds.coords)),
        "data_vars": {
            str(name): {
                "dtype": str(ds[name].dtype),
                "dims": list(ds[name].dims),
                "shape": list(ds[name].shape),
            }
            for name in sorted(ds.data_vars)
        },
    }


def compact_metadata(ds: xr.Dataset) -> dict[str, Any]:
    keys = (
        "title",
        "institution",
        "source",
        "product",
        "product_version",
        "version",
        "references",
        "history",
        "summary",
        "date_created",
        "Conventions",
        "license",
    )
    return {key: ds.attrs[key] for key in keys if key in ds.attrs}


def target_depth_info(ds: xr.Dataset) -> dict[str, Any]:
    name = depth_name(ds)
    if name is None:
        return {
            "status": "INCONCLUSIVE",
            "reason": "depth coordinate not found",
        }

    depth = np.asarray(ds[name].values, dtype=float).reshape(-1)
    nearest = []

    for target in GLORYS_DEPTHS:
        index = int(np.argmin(np.abs(depth - target)))
        nearest.append(
            {
                "requested_m": float(target),
                "nearest_source_m": float(depth[index]),
                "abs_error_m": float(abs(depth[index] - target)),
            }
        )

    return {
        "status": "COMPLETE",
        "depth_name": name,
        "units": ds[name].attrs.get("units"),
        "n_depths": int(depth.size),
        "min_depth": float(np.min(depth)),
        "max_depth": float(np.max(depth)),
        "ascending": bool(depth[-1] > depth[0]) if depth.size > 1 else None,
        "exact_15_depths": bool(
            all(x["abs_error_m"] <= DEPTH_TOL for x in nearest)
        ),
        "requested_depths": nearest,
    }


def finite_stats(
    arr: np.ndarray,
    sample: int = 250_000,
    seed: int = 42,
) -> dict[str, Any]:
    values = np.asarray(arr).reshape(-1)
    finite = values[np.isfinite(values)]
    count = int(values.size)
    finite_count = int(finite.size)

    result = {
        "count": count,
        "finite_count": finite_count,
        "invalid_count": int(count - finite_count),
        "valid_percent": float(100 * finite_count / count) if count else None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "quantiles": {},
    }

    if finite_count == 0:
        return result

    result["min"] = float(np.min(finite))
    result["max"] = float(np.max(finite))
    result["mean"] = float(np.mean(finite, dtype=np.float64))
    result["std"] = float(np.std(finite, dtype=np.float64))

    if finite_count <= sample:
        qvalues = finite
        method = "EXACT"
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(finite_count, sample, replace=False)
        qvalues = finite[indices]
        method = f"DETERMINISTIC_SAMPLE_{sample}"

    result["quantiles"] = {
        f"p{q:02d}": float(np.percentile(qvalues, q))
        for q in (1, 5, 25, 50, 75, 95, 99)
    }
    result["quantile_method"] = method
    return result


def audit_one_file(
    dataset_key: str,
    path: Path,
    cfg: Config,
) -> tuple[dict[str, Any], str | None]:
    ds = None

    try:
        ds = open_nc(path)
        tinfo = time_info(ds)
        semantic = semantic_info(dataset_key, ds)

        filename_tokens = re.findall(
            r"\b\d{8}\b|\b\d{4}[-_]\d{2}[-_]\d{2}\b|\b\d{4}[-_]\d{2}\b",
            path.stem,
        )
        filename_dates = []
        for token in filename_tokens:
            try:
                filename_dates.append(
                    pd.to_datetime(token.replace("_", "-")).strftime("%Y-%m-%d")
                )
            except Exception:
                pass

        if filename_dates and tinfo["dates"]:
            tinfo["filename_dates"] = filename_dates
            tinfo["filename_date_consistent"] = all(
                tinfo["first_date"] <= item <= tinfo["last_date"]
                for item in filename_dates
            )

        record: dict[str, Any] = {
            "file": str(path),
            "relative": str(path.relative_to(cfg.root)),
            "size_bytes": path.stat().st_size,
            "schema": schema_signature(ds),
            "grid": grid_info(ds),
            "time": tinfo,
            "metadata": compact_metadata(ds),
            "semantic": semantic,
        }

        if dataset_key == TARGET:
            record["target_depths"] = target_depth_info(ds)

        return record, None

    except Exception as exc:
        record = {
            "file": str(path),
            "relative": str(path.relative_to(cfg.root)),
            "status": "FAILED",
            "error": repr(exc),
        }
        return record, repr(exc)

    finally:
        close(ds)


def discover(cfg: Config, key: str) -> list[Path]:
    root = cfg.root / REGISTRY[key]["root"]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.nc") if p.is_file())


def stage1(cfg: Config, keys: list[str]) -> None:
    out = cfg.output / "stage1"
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_grid": {
            "lat_min": LAT_MIN,
            "lat_max": LAT_MAX,
            "lon_min": LON_MIN,
            "lon_max": LON_MAX,
            "resolution": RES,
            "shape": [LAT_SIZE, LON_SIZE],
        },
        "datasets": {},
    }

    for key in keys:
        files = discover(cfg, key)
        LOG.info("[%s] %d files", REGISTRY[key]["name"], len(files))

        records = []
        failures = []
        date_index: dict[str, list[str]] = defaultdict(list)
        timestamp_index: dict[str, list[str]] = defaultdict(list)

        total = len(files)

        for i, path in enumerate(files, 1):
            record, error = audit_one_file(key, path, cfg)
            records.append(record)

            if error:
                failures.append(record)

            for date in record.get("time", {}).get("dates", []):
                date_index[date].append(str(path))

            for timestamp in record.get("time", {}).get("timestamps", []):
                timestamp_index[timestamp].append(str(path))

            if (
                i == 1
                or i % cfg.progress_every == 0
                or i == total
            ):
                LOG.info(
                    "[%s] %d/%d files (%.1f%%)",
                    key,
                    i,
                    total,
                    100 * i / total if total else 100,
                )

        write_json(out / f"{key}_files.json", records)
        write_json(out / f"{key}_date_index.json", dict(date_index))
        write_json(out / f"{key}_timestamp_index.json", dict(timestamp_index))

        summary = {
            "dataset": key,
            "name": REGISTRY[key]["name"],
            "files_found": len(files),
            "files_checked": len(records) - len(failures),
            "files_failed": len(failures),
            "status": "COMPLETE" if len(records) == len(files) else "PARTIAL",
            "semantic_confirmed_files": sum(
                r.get("semantic", {}).get("semantic_status") == "CONFIRMED"
                for r in records
            ),
            "canonical_grid_exact_files": sum(
                r.get("grid", {}).get("canonical_exact") is True
                for r in records
            ),
        }

        if key == TARGET:
            summary["exact_15_depth_files"] = sum(
                r.get("target_depths", {}).get("exact_15_depths") is True
                for r in records
            )

        write_json(out / f"{key}_summary.json", summary)
        manifest["datasets"][key] = summary

    write_json(out / "manifest.json", manifest)
    LOG.info("STAGE 1 COMPLETE: %s", out)


def load_stage1(cfg: Config, key: str) -> list[dict[str, Any]]:
    path = cfg.output / "stage1" / f"{key}_files.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-1 evidence for {key}: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(cfg: Config, key: str, kind: str) -> dict[str, list[str]]:
    path = cfg.output / "stage1" / f"{key}_{kind}_index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage-1 {kind} index for {key}: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def selected_variable_array(
    ds: xr.Dataset,
    dataset_key: str,
    role: str,
    variable: str,
) -> xr.DataArray:
    da = ds[variable]

    # For GLORYS statistics, only the requested 15 target depths are loaded.
    if dataset_key == TARGET and role == "thetao":
        name = depth_name(ds)
        if name and name in da.dims:
            depth = np.asarray(ds[name].values, dtype=float).reshape(-1)
            indices = [
                int(np.argmin(np.abs(depth - target)))
                for target in GLORYS_DEPTHS
            ]
            da = da.isel({name: sorted(set(indices))})

    return da


def stage_stats(cfg: Config, keys: list[str]) -> None:
    out = cfg.output / "stage1_stats"
    out.mkdir(parents=True, exist_ok=True)

    for key in keys:
        records = load_stage1(cfg, key)
        failures = []
        aggregate = defaultdict(
            lambda: {
                "files": 0,
                "finite": 0,
                "invalid": 0,
                "min": None,
                "max": None,
                "sum": 0.0,
                "sumsq": 0.0,
            }
        )

        total = len(records)

        for i, record in enumerate(records, 1):
            path = Path(record["file"])
            ds = None

            try:
                ds = open_nc(path)
                resolved = record.get("semantic", {}).get("resolved", {})

                for role in INTEGRATION.get(key, []):
                    evidence = resolved.get(role)

                    if not evidence or not evidence.get("variable"):
                        continue

                    variable = evidence["variable"]
                    da = selected_variable_array(
                        ds, key, role, variable
                    )

                    values = np.asarray(da.values)
                    flat = values.reshape(-1)
                    finite = flat[np.isfinite(flat)]

                    per_file = finite_stats(
                        flat,
                        seed=cfg.seed + sum(map(ord, role)),
                    )

                    signature = (
                        role,
                        variable,
                        ",".join(da.dims),
                        str(da.dtype),
                    )
                    item = aggregate[signature]

                    item["files"] += 1
                    item["finite"] += int(finite.size)
                    item["invalid"] += int(flat.size - finite.size)

                    if finite.size:
                        current_min = float(np.min(finite))
                        current_max = float(np.max(finite))

                        item["min"] = (
                            current_min
                            if item["min"] is None
                            else min(item["min"], current_min)
                        )
                        item["max"] = (
                            current_max
                            if item["max"] is None
                            else max(item["max"], current_max)
                        )

                        f64 = finite.astype(np.float64, copy=False)
                        item["sum"] += float(np.sum(f64))
                        item["sumsq"] += float(np.sum(f64 * f64))

                    record.setdefault("targeted_statistics", {})[role] = {
                        "variable": variable,
                        "stats": per_file,
                    }

            except Exception as exc:
                failures.append(
                    {
                        "file": str(path),
                        "stage": "stats",
                        "error": repr(exc),
                    }
                )

            finally:
                close(ds)

            if (
                i == 1
                or i % cfg.progress_every == 0
                or i == total
            ):
                LOG.info(
                    "[STATS:%s] %d/%d files (%.1f%%)",
                    key,
                    i,
                    total,
                    100 * i / total if total else 100,
                )

        final = {}

        for (role, variable, dimensions, dtype), item in aggregate.items():
            finite_count = item["finite"]

            if finite_count:
                mean = item["sum"] / finite_count
                variance = max(
                    item["sumsq"] / finite_count - mean * mean,
                    0.0,
                )
                std = math.sqrt(variance)
            else:
                mean = None
                std = None

            final[
                f"{role}|{variable}|{dimensions}|{dtype}"
            ] = {
                "role": role,
                "variable": variable,
                "dimensions": dimensions,
                "dtype": dtype,
                "files": item["files"],
                "finite_count": finite_count,
                "invalid_count": item["invalid"],
                "valid_percent": (
                    100
                    * finite_count
                    / (finite_count + item["invalid"])
                    if finite_count + item["invalid"]
                    else None
                ),
                "min": item["min"],
                "max": item["max"],
                "mean": mean,
                "std": std,
                "note": (
                    "Quantiles are retained per file and are not "
                    "treated as pooled quantiles."
                ),
            }

        write_json(
            out / f"{key}_files.json",
            records,
        )
        write_json(
            out / f"{key}_aggregate.json",
            {
                "dataset": key,
                "status": "COMPLETE" if not failures else "PARTIAL",
                "files_checked": total - len(failures),
                "files_failed": len(failures),
                "aggregate": final,
                "failures": failures,
            },
        )

    LOG.info("TARGETED NUMERICAL STATS COMPLETE: %s", out)


def canonical_interp(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    method: str = "linear",
) -> xr.DataArray:
    x = da.sortby(lat_name).sortby(lon_name)

    lon_values = np.asarray(x[lon_name].values, dtype=float)
    if np.nanmin(lon_values) >= 0 and np.nanmax(lon_values) > 180:
        x = x.assign_coords(
            {
                lon_name: (
                    (lon_values + 180) % 360
                ) - 180
            }
        ).sortby(lon_name)

    return x.interp(
        {lat_name: CANON_LAT, lon_name: CANON_LON},
        method=method,
    )


def genuine_holdout(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    cfg: Config,
    seed: int,
) -> dict[str, Any]:
    x = da.squeeze(drop=True)

    extra_dims = [
        dim
        for dim in x.dims
        if dim not in {lat_name, lon_name}
    ]
    if extra_dims:
        return {
            "status": "INCONCLUSIVE",
            "reason": f"unsliced dimensions: {extra_dims}",
        }

    array = np.asarray(x.values)
    if array.ndim != 2:
        return {
            "status": "INCONCLUSIVE",
            "reason": f"expected 2-D field, got {array.ndim}-D",
        }

    lat = np.asarray(x[lat_name].values, dtype=float)
    lon = np.asarray(x[lon_name].values, dtype=float)

    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)

    lat = lat[lat_order]
    lon = lon[lon_order]
    array = array[np.ix_(lat_order, lon_order)]

    valid = np.isfinite(array)
    coordinates = np.argwhere(valid)

    if len(coordinates) < 20:
        return {
            "status": "INCONCLUSIVE",
            "reason": "too few finite source points",
            "finite_points": int(len(coordinates)),
        }

    n_holdout = min(
        cfg.holdout_points,
        max(10, int(len(coordinates) * cfg.holdout_fraction)),
    )

    rng = np.random.default_rng(seed)
    hold_indices = rng.choice(
        len(coordinates),
        n_holdout,
        replace=False,
    )
    holdout = coordinates[hold_indices]

    training = coordinates[
        ~np.isin(np.arange(len(coordinates)), hold_indices)
    ]

    training_points = np.column_stack(
        (
            lat[training[:, 0]],
            lon[training[:, 1]],
        )
    )
    query_points = np.column_stack(
        (
            lat[holdout[:, 0]],
            lon[holdout[:, 1]],
        )
    )

    predictions = griddata(
        training_points,
        array[training[:, 0], training[:, 1]],
        query_points,
        method="linear",
    )

    truth = array[holdout[:, 0], holdout[:, 1]]
    ok = np.isfinite(predictions) & np.isfinite(truth)

    if int(ok.sum()) < 10:
        return {
            "status": "INCONCLUSIVE",
            "reason": "insufficient interpolatable holdout points",
            "holdout_points": int(n_holdout),
            "predicted_points": int(ok.sum()),
        }

    errors = predictions[ok] - truth[ok]
    truth_std = float(np.std(truth[ok]))

    return {
        "status": "COMPLETE",
        "holdout_points": int(n_holdout),
        "predicted_points": int(ok.sum()),
        "prediction_coverage_percent": float(
            100 * ok.mean()
        ),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "normalized_rmse": (
            float(
                np.sqrt(np.mean(errors * errors))
                / truth_std
            )
            if truth_std > 0
            else None
        ),
    }


def pick_file(index: dict[str, list[str]], date: str) -> Path | None:
    values = index.get(date, [])
    return Path(values[0]) if values else None


def slice_date(
    da: xr.DataArray,
    ds: xr.Dataset,
    requested_date: str,
) -> xr.DataArray:
    name = time_name(ds)

    if name and name in da.dims:
        values = times(ds)
        index = next(
            (
                i
                for i, value in enumerate(values)
                if date_key(value) == requested_date
            ),
            None,
        )

        if index is None:
            raise ValueError(
                f"requested date {requested_date} absent from file"
            )

        da = da.isel({name: index})

    return da


def stage2(cfg: Config, keys: list[str]) -> None:
    out = cfg.output / "stage2"
    out.mkdir(parents=True, exist_ok=True)

    indexes = {
        key: load_index(cfg, key, "date")
        for key in keys
    }
    timestamp_indexes = {
        key: load_index(cfg, key, "timestamp")
        for key in keys
    }

    date_sets = {
        key: set(index.keys())
        for key, index in indexes.items()
        if index
    }
    timestamp_sets = {
        key: set(index.keys())
        for key, index in timestamp_indexes.items()
        if index
    }

    required = [
        key
        for key in CORE + [TARGET]
        if key in date_sets and date_sets[key]
    ]

    common_dates = (
        sorted(set.intersection(*(date_sets[key] for key in required)))
        if required
        else []
    )

    common_timestamps = (
        sorted(
            set.intersection(
                *(timestamp_sets[key] for key in required)
            )
        )
        if required
        else []
    )

    temporal = {
        "core_plus_target_common_dates": common_dates,
        "core_plus_target_common_timestamps": common_timestamps,
        "date_counts": {
            key: len(values)
            for key, values in date_sets.items()
        },
        "timestamp_counts": {
            key: len(values)
            for key, values in timestamp_sets.items()
        },
        "note": (
            "Date overlap is not physical-time equivalence; "
            "exact timestamps are reported separately."
        ),
    }

    write_json(
        out / "temporal_intersection.json",
        temporal,
    )

    # ---------------------------------------------------------------
    # Regridding diagnostics
    # ---------------------------------------------------------------
    regridding = {}
    glorys_files = discover(cfg, TARGET)

    target_grid_exact = None
    if glorys_files:
        ds = None
        try:
            ds = open_nc(glorys_files[0])
            lat_name = coord(ds, "lat")
            lon_name = coord(ds, "lon")

            target_grid_exact = bool(
                lat_name
                and lon_name
                and len(ds[lat_name]) == LAT_SIZE
                and len(ds[lon_name]) == LON_SIZE
                and np.allclose(
                    ds[lat_name].values,
                    CANON_LAT,
                    atol=1e-8,
                )
                and np.allclose(
                    ds[lon_name].values,
                    CANON_LON,
                    atol=1e-8,
                )
            )
        finally:
            close(ds)

    for key in [item for item in CORE if item in indexes]:
        files = [
            Path(record["file"])
            for record in load_stage1(cfg, key)
        ]

        if len(files) <= cfg.sample_cross:
            chosen = files
        else:
            chosen = [
                files[0],
                files[len(files) // 2],
                files[-1],
            ]

        rows = []

        for sample_number, path in enumerate(chosen, 1):
            ds = None

            try:
                ds = open_nc(path)
                resolved = resolve_variables(key, ds)
                lat_name = coord(ds, "lat")
                lon_name = coord(ds, "lon")

                if not lat_name or not lon_name:
                    continue

                for role in INTEGRATION.get(key, []):
                    evidence = resolved.get(role)
                    if not evidence or not evidence.get("variable"):
                        continue

                    da = ds[evidence["variable"]]

                    if times(ds):
                        da = slice_date(
                            da,
                            ds,
                            date_key(times(ds)[0]),
                        )

                    depth = depth_name(ds)
                    if depth and depth in da.dims:
                        da = da.isel({depth: 0})

                    canonical = canonical_interp(
                        da,
                        lat_name,
                        lon_name,
                    )
                    array = np.asarray(canonical.values)
                    finite = np.isfinite(array)

                    rows.append(
                        {
                            "file": str(path),
                            "role": role,
                            "variable": evidence["variable"],
                            "canonical_valid_cells": int(
                                finite.sum()
                            ),
                            "canonical_valid_percent": (
                                float(100 * finite.mean())
                                if array.size
                                else None
                            ),
                            "holdout": genuine_holdout(
                                da,
                                lat_name,
                                lon_name,
                                cfg,
                                cfg.seed + sample_number,
                            ),
                        }
                    )

            except Exception as exc:
                rows.append(
                    {
                        "file": str(path),
                        "status": "FAILED",
                        "error": repr(exc),
                    }
                )
            finally:
                close(ds)

        regridding[key] = {
            "target_grid_exact": target_grid_exact,
            "sample_rows": rows,
            "status": "COMPLETE" if rows else "INCONCLUSIVE",
            "note": (
                "Holdout metrics diagnose interpolation self-consistency "
                "on withheld source-grid observations. They are not "
                "independent ocean-observation validation."
            ),
        }

    write_json(
        out / "regridding.json",
        regridding,
    )

    # ---------------------------------------------------------------
    # Joint validity
    # ---------------------------------------------------------------
    if common_dates:
        if cfg.joint_all:
            joint_dates = common_dates
        else:
            n = min(cfg.sample_cross, len(common_dates))
            positions = np.linspace(
                0,
                len(common_dates) - 1,
                n,
                dtype=int,
            )
            joint_dates = [
                common_dates[int(position)]
                for position in positions
            ]
    else:
        joint_dates = []

    joint_rows = []

    for date_number, date in enumerate(joint_dates, 1):
        masks = {}
        missing = []

        for key in required:
            path = pick_file(indexes[key], date)

            if path is None:
                missing.append(key)
                continue

            ds = None

            try:
                ds = open_nc(path)
                resolved = resolve_variables(key, ds)
                lat_name = coord(ds, "lat")
                lon_name = coord(ds, "lon")
                component_masks = []

                for role in INTEGRATION.get(key, []):
                    evidence = resolved.get(role)
                    if not evidence or not evidence.get("variable"):
                        continue

                    da = slice_date(
                        ds[evidence["variable"]],
                        ds,
                        date,
                    )

                    if (
                        not lat_name
                        or not lon_name
                        or lat_name not in da.dims
                        or lon_name not in da.dims
                    ):
                        continue

                    canonical = canonical_interp(
                        da,
                        lat_name,
                        lon_name,
                    )
                    component_masks.append(
                        np.isfinite(
                            np.asarray(canonical.values)
                        )
                    )

                if not component_masks:
                    missing.append(key)
                else:
                    masks[key] = np.logical_and.reduce(
                        component_masks
                    )

            except Exception:
                missing.append(key)

            finally:
                close(ds)

        row = {"date": date}

        if missing:
            row.update(
                {
                    "status": "INCOMPLETE",
                    "missing_or_unresolved": missing,
                }
            )
        else:
            joint = np.logical_and.reduce(
                [masks[key] for key in required]
            )

            row.update(
                {
                    "status": "COMPLETE",
                    "total_canonical_cells": int(joint.size),
                    "joint_valid_cells": int(joint.sum()),
                    "joint_valid_percent": float(
                        100 * joint.mean()
                    ),
                }
            )

            for key, mask in masks.items():
                row[f"{key}_valid_cells"] = int(mask.sum())
                row[f"{key}_valid_percent"] = float(
                    100 * mask.mean()
                )

        joint_rows.append(row)

        if (
            date_number == 1
            or date_number % 10 == 0
            or date_number == len(joint_dates)
        ):
            LOG.info(
                "[JOINT] %d/%d dates",
                date_number,
                len(joint_dates),
            )

    write_json(
        out / "joint_validity.json",
        {
            "status": "COMPLETE" if joint_rows else "INCONCLUSIVE",
            "dates_requested": joint_dates,
            "rows": joint_rows,
            "note": (
                "Joint validity is computed after linear interpolation "
                "to the canonical 0.25-degree grid."
            ),
        },
    )

    # ---------------------------------------------------------------
    # GlobCurrent vs OSCAR
    # ---------------------------------------------------------------
    redundancy = {
        "status": "NOT_RUN",
        "reason": "OSCAR not included in selected datasets",
    }

    if "globecurrent" in indexes and "oscar" in indexes:
        common_current_dates = sorted(
            set(indexes["globecurrent"])
            & set(indexes["oscar"])
        )

        chosen = (
            common_current_dates
            if cfg.joint_all
            else common_current_dates[:cfg.sample_cross]
        )

        rows = []

        for date in chosen:
            gc_path = pick_file(
                indexes["globecurrent"],
                date,
            )
            oscar_path = pick_file(
                indexes["oscar"],
                date,
            )

            if not gc_path or not oscar_path:
                continue

            gc_ds = None
            os_ds = None

            try:
                gc_ds = open_nc(gc_path)
                os_ds = open_nc(oscar_path)

                gc_resolved = resolve_variables(
                    "globecurrent",
                    gc_ds,
                )
                os_resolved = resolve_variables(
                    "oscar",
                    os_ds,
                )

                if not all(
                    gc_resolved.get(role, {}).get("variable")
                    for role in ("u", "v")
                ):
                    continue

                if not all(
                    os_resolved.get(role, {}).get("variable")
                    for role in ("u", "v")
                ):
                    continue

                gc_u = slice_date(
                    gc_ds[
                        gc_resolved["u"]["variable"]
                    ],
                    gc_ds,
                    date,
                )
                gc_v = slice_date(
                    gc_ds[
                        gc_resolved["v"]["variable"]
                    ],
                    gc_ds,
                    date,
                )
                os_u = slice_date(
                    os_ds[
                        os_resolved["u"]["variable"]
                    ],
                    os_ds,
                    date,
                )
                os_v = slice_date(
                    os_ds[
                        os_resolved["v"]["variable"]
                    ],
                    os_ds,
                    date,
                )

                gc_lat = coord(gc_ds, "lat")
                gc_lon = coord(gc_ds, "lon")
                os_lat = coord(os_ds, "lat")
                os_lon = coord(os_ds, "lon")

                if not all(
                    (gc_lat, gc_lon, os_lat, os_lon)
                ):
                    continue

                arrays = [
                    np.asarray(
                        canonical_interp(
                            da,
                            lat,
                            lon,
                        ).values
                    )
                    for da, lat, lon in (
                        (gc_u, gc_lat, gc_lon),
                        (gc_v, gc_lat, gc_lon),
                        (os_u, os_lat, os_lon),
                        (os_v, os_lat, os_lon),
                    )
                ]

                gc_u_a, gc_v_a, os_u_a, os_v_a = arrays

                valid = (
                    np.isfinite(gc_u_a)
                    & np.isfinite(gc_v_a)
                    & np.isfinite(os_u_a)
                    & np.isfinite(os_v_a)
                )

                if int(valid.sum()) < 10:
                    continue

                a = gc_u_a[valid]
                b = gc_v_a[valid]
                c = os_u_a[valid]
                d = os_v_a[valid]

                def corr(x: np.ndarray, y: np.ndarray) -> float | None:
                    if (
                        len(x) < 2
                        or np.std(x) == 0
                        or np.std(y) == 0
                    ):
                        return None
                    return float(
                        np.corrcoef(x, y)[0, 1]
                    )

                gc_speed = np.hypot(a, b)
                os_speed = np.hypot(c, d)

                angle_difference = np.abs(
                    np.angle(
                        np.exp(
                            1j
                            * (
                                np.arctan2(d, c)
                                - np.arctan2(b, a)
                            )
                        )
                    )
                )

                rows.append(
                    {
                        "date": date,
                        "valid_cells": int(valid.sum()),
                        "u_bias_oscar_minus_globcurrent": float(
                            np.mean(c - a)
                        ),
                        "u_mae": float(
                            np.mean(np.abs(c - a))
                        ),
                        "u_rmse": float(
                            np.sqrt(np.mean((c - a) ** 2))
                        ),
                        "u_corr": corr(a, c),
                        "v_bias_oscar_minus_globcurrent": float(
                            np.mean(d - b)
                        ),
                        "v_mae": float(
                            np.mean(np.abs(d - b))
                        ),
                        "v_rmse": float(
                            np.sqrt(np.mean((d - b) ** 2))
                        ),
                        "v_corr": corr(b, d),
                        "speed_corr": corr(
                            gc_speed,
                            os_speed,
                        ),
                        "speed_rmse": float(
                            np.sqrt(
                                np.mean(
                                    (os_speed - gc_speed) ** 2
                                )
                            )
                        ),
                        "mean_direction_difference_rad": float(
                            np.mean(angle_difference)
                        ),
                    }
                )

            except Exception as exc:
                rows.append(
                    {
                        "date": date,
                        "status": "FAILED",
                        "error": repr(exc),
                    }
                )

            finally:
                close(gc_ds)
                close(os_ds)

        redundancy = {
            "status": "COMPLETE" if rows else "INCONCLUSIVE",
            "common_dates": common_current_dates,
            "sample_dates": chosen,
            "rows": rows,
            "note": (
                "Redundancy inference requires comparable physical "
                "semantics, units and temporal meaning; correlation "
                "alone is insufficient."
            ),
        }

    write_json(
        out / "redundancy.json",
        redundancy,
    )

    LOG.info("STAGE 2 COMPLETE: %s", out)


def write_summary(cfg: Config, keys: list[str]) -> None:
    lines = [
        "# OceanEmbed Dataset Forensic Audit",
        "",
        f"Version: `{VERSION}`",
        "",
        "## Stage 1",
        "",
    ]

    for key in keys:
        path = (
            cfg.output
            / "stage1"
            / f"{key}_summary.json"
        )
        if not path.exists():
            continue

        summary = json.loads(
            path.read_text(encoding="utf-8")
        )

        lines.append(
            f"- **{summary['name']}**: "
            f"{summary['files_found']} files; "
            f"{summary['files_checked']} checked; "
            f"{summary['files_failed']} failed; "
            f"{summary['semantic_confirmed_files']} "
            f"semantic-confirmed."
        )

        if key == TARGET:
            lines.append(
                f"  - Exact 15-depth files: "
                f"{summary.get('exact_15_depth_files', 0)}"
            )

    lines += [
        "",
        "## Stage 2",
        "",
    ]

    temporal_path = (
        cfg.output
        / "stage2"
        / "temporal_intersection.json"
    )

    if temporal_path.exists():
        temporal = json.loads(
            temporal_path.read_text(
                encoding="utf-8"
            )
        )
        lines.append(
            "- Core + target common dates: "
            f"{len(temporal.get('core_plus_target_common_dates', []))}"
        )
        lines.append(
            "- Core + target exact timestamps: "
            f"{len(temporal.get('core_plus_target_common_timestamps', []))}"
        )

    lines += [
        "",
        "Raw NetCDF inputs are read-only.",
        "Stage 1 opens each file once.",
        "Stage 2 performs cross-dataset diagnostics on selected files/dates.",
        "Argo is intentionally separated as a later validation stage.",
    ]

    (cfg.output / "audit_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OceanEmbed staged forensic audit. "
            "Raw NetCDF inputs are never modified."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            r"D:\OceanEmbed\oceanembed-data\raw"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\OceanEmbed\oceanembed-data\audit_final"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["1", "stats", "2", "all"],
        default="1",
        help=(
            "1 = fast one-open-per-file forensic inventory; "
            "stats = targeted numerical pass; "
            "2 = cross-dataset diagnostics; "
            "all = 1 + stats + 2"
        ),
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="comma-separated dataset keys or all",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="kept as an explicit alias for stage=stats when stage=1",
    )
    parser.add_argument(
        "--sample-cross",
        type=int,
        default=10,
        help="number of dates/files used for cross-dataset diagnostics",
    )
    parser.add_argument(
        "--joint-all",
        action="store_true",
        help=(
            "compute joint-validity/redundancy across all common dates; "
            "this is intentionally expensive"
        ),
    )
    parser.add_argument(
        "--holdout-points",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_log(args.verbose)

    keys = (
        list(REGISTRY)
        if args.datasets == "all"
        else [
            item.strip()
            for item in args.datasets.split(",")
        ]
    )
    keys = [
        key
        for key in keys
        if key in REGISTRY
    ]

    if not keys:
        LOG.error("No valid datasets selected.")
        return 2

    if args.stats and args.stage == "1":
        args.stage = "stats"

    cfg = Config(
        root=args.root,
        output=args.output,
        datasets=keys,
        stage=args.stage,
        stats=args.stats,
        sample_cross=args.sample_cross,
        joint_all=args.joint_all,
        holdout_points=args.holdout_points,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        progress_every=max(1, args.progress_every),
    )

    cfg.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG.info(
        "OceanEmbed Forensic Audit %s",
        VERSION,
    )
    LOG.info("RAW INPUTS ARE READ-ONLY")
    LOG.info("Root: %s", cfg.root)
    LOG.info("Output: %s", cfg.output)
    LOG.info(
        "Stage=%s | datasets=%s",
        cfg.stage,
        ",".join(keys),
    )

    if cfg.stage in {"1", "all"}:
        stage1(cfg, keys)

    if cfg.stage in {"stats", "all"}:
        stage_stats(cfg, keys)

    if cfg.stage in {"2", "all"}:
        stage2(cfg, keys)

    if cfg.stage in {"stats", "2", "all"}:
        write_summary(cfg, keys)

    LOG.info("AUDIT COMPLETE")
    LOG.info("Reports: %s", cfg.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted.")
        raise SystemExit(130)
    except Exception:
        LOG.error(
            "Fatal audit failure:\n%s",
            traceback.format_exc(),
        )
        raise SystemExit(1)
