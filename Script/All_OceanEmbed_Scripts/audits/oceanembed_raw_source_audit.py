#!/usr/bin/env python3
"""OceanEmbed raw-source integrity audit: CCMP, OSCAR, and the supplied Argo tree.

Purpose
-------
1) Audit raw CCMP wind variables and any raw validity/quality/flag variables.
2) Audit raw OSCAR current variables and any raw validity/quality/flag variables.
3) Inventory and sanity-check raw Argo profile files using the supplied YYYY_MM/DAC/float/profiles layout.

Outputs are written under:
    D:\\OceanEmbed\\oceanembed-data\\raw\\audit_results\\raw_source_audit_<timestamp>\\

Important
---------
The prompt supplied the Argo folder layout, but not the exact CCMP/OSCAR raw folders.
The script tries these defaults first:
    <raw>\\ccmp
    <raw>\\oscar
It then searches for directories whose names contain CCMP / OSCAR. If your folders are
stored elsewhere, edit CCMP_RAW_ROOT and OSCAR_RAW_ROOT below before running.

Dependencies (run in the VS Code environment):
    pip install numpy pandas xarray netCDF4

This script does not alter source files. CCMP/OSCAR file inventory covers all discoverable
files; detailed numeric statistics are computed on an evenly spaced, reproducible sample of
files to keep the audit practical. Set DATA_FILES_TO_AUDIT_PER_SOURCE = 0 to read every file.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

# ------------------------------- CONFIG ------------------------------------
RAW_BASE = Path(r"D:\OceanEmbed\oceanembed-data\raw")
CCMP_RAW_ROOT = RAW_BASE / "ccmp"   # EDIT if your actual folder is named differently
OSCAR_RAW_ROOT = RAW_BASE / "oscar" # EDIT if your actual folder is named differently
ARGO_RAW_ROOT = RAW_BASE / "argo"
OUTPUT_BASE = RAW_BASE / "audit_results"

# 0 = detailed statistics for every CCMP/OSCAR file; 500 is a faster, broad time sample.
DATA_FILES_TO_AUDIT_PER_SOURCE = 500
# 0 = audit all Argo profiles. If the archive is exceptionally large, set e.g. 3000.
ARGO_MAX_FILES = 0

# OceanEmbed domain. Used only when regular 1-D latitude/longitude coordinates are present.
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
MAX_QUANTILE_SAMPLE = 200_000
SUPPORTED_SUFFIXES = {".nc", ".nc4", ".cdf", ".h5", ".hdf5"}
FLAG_WORDS = ("mask", "valid", "flag", "qc", "quality", "status")
COORD_WORDS = {"lat", "latitude", "lon", "longitude", "time", "depth", "lev", "level", "z", "crs"}

# ---------------------------------------------------------------------------


def safe_str(value: Any, limit: int = 800) -> str:
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    s = s.replace("\r", " ").replace("\n", " ")
    return s[:limit]


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return safe_str(value)


def find_source_root(preferred: Path, keyword: str) -> Path | None:
    """Use configured root if it exists; otherwise find likely named subfolders."""
    if preferred.exists() and preferred.is_dir():
        return preferred
    if not RAW_BASE.exists():
        return None
    matches: list[Path] = []
    try:
        # Search only a few folder levels; do not traverse the entire archive just to find roots.
        for p in RAW_BASE.iterdir():
            if p.is_dir() and keyword.casefold() in p.name.casefold() and p.name.casefold() != "audit_results":
                matches.append(p)
            if p.is_dir() and p.name.casefold() != "audit_results":
                try:
                    for q in p.iterdir():
                        if q.is_dir() and keyword.casefold() in q.name.casefold():
                            matches.append(q)
                except OSError:
                    pass
    except OSError:
        return None
    matches = sorted(set(matches), key=lambda p: (len(str(p)), str(p).casefold()))
    return matches[0] if matches else None


def discover_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.casefold() in SUPPORTED_SUFFIXES:
            # Ignore accidental output folders if a broad root was selected.
            if "audit_results" in {part.casefold() for part in p.parts}:
                continue
            files.append(p)
    return sorted(files, key=lambda p: str(p).casefold())


def sample_evenly(files: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(files) <= limit:
        return files
    indexes = np.linspace(0, len(files) - 1, num=limit, dtype=int)
    # Stable de-duplication, relevant for small source collections.
    return [files[i] for i in sorted(set(indexes.tolist()) )]


def infer_month(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    # Folder convention YYYY_MM.
    m = re.search(r"(?<!\d)(20\d{2})[_-](0[1-9]|1[0-2])(?!\d)", rel)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Dates in filenames: YYYYMMDD or YYYY-MM-DD.
    m = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", path.name)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{m.group(1)}-{m.group(2)}"
    return "unknown"


def get_coord(ds: xr.Dataset, choices: Iterable[str]) -> xr.DataArray | None:
    lookup = {name.casefold(): name for name in ds.variables}
    for candidate in choices:
        name = lookup.get(candidate.casefold())
        if name and ds[name].ndim == 1 and np.issubdtype(ds[name].dtype, np.number):
            return ds[name]
    # Common fallback: coordinates may be named y/x, but use only when clearly tagged by attrs.
    for name in ds.variables:
        da = ds[name]
        if da.ndim != 1 or not np.issubdtype(da.dtype, np.number):
            continue
        standard = safe_str(da.attrs.get("standard_name", "")).casefold()
        axis = safe_str(da.attrs.get("axis", "")).upper()
        if "latitude" in standard or axis == "Y":
            if any(c in choices for c in ("lat", "latitude")):
                return da
        if "longitude" in standard or axis == "X":
            if any(c in choices for c in ("lon", "longitude")):
                return da
    return None


def crop_to_domain(da: xr.DataArray, ds: xr.Dataset) -> tuple[xr.DataArray, str]:
    """Crop regular lat/lon data to the NIO region before materializing values."""
    notes: list[str] = []
    lat = get_coord(ds, ("latitude", "lat"))
    lon = get_coord(ds, ("longitude", "lon"))
    indexers: dict[str, slice] = {}
    if lat is not None and lat.dims[0] in da.dims:
        try:
            lv = np.asarray(lat.values, dtype=float)
            inds = np.flatnonzero(np.isfinite(lv) & (lv >= LAT_MIN) & (lv <= LAT_MAX))
            if inds.size:
                indexers[lat.dims[0]] = slice(int(inds.min()), int(inds.max()) + 1)
                notes.append("lat cropped")
        except Exception:
            pass
    if lon is not None and lon.dims[0] in da.dims:
        try:
            xv = np.asarray(lon.values, dtype=float)
            # The OceanEmbed region lies east of Greenwich; this supports 0..360 or -180..180 axes.
            inds = np.flatnonzero(np.isfinite(xv) & (xv >= LON_MIN) & (xv <= LON_MAX))
            if inds.size:
                indexers[lon.dims[0]] = slice(int(inds.min()), int(inds.max()) + 1)
                notes.append("lon cropped")
        except Exception:
            pass
    if indexers:
        try:
            da = da.isel(indexers)
        except Exception:
            notes.append("crop failed")
    else:
        notes.append("no regular lat/lon crop")
    return da, "; ".join(notes)


def quantile_values(arr: np.ndarray, max_n: int = MAX_QUANTILE_SAMPLE) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    if flat.size > max_n:
        step = max(1, int(math.ceil(flat.size / max_n)))
        flat = flat[::step]
    return flat


def get_attr(da: xr.DataArray, key: str) -> Any:
    if key in da.attrs:
        return da.attrs[key]
    if key in da.encoding:
        return da.encoding[key]
    return ""


def flag_rows(source: str, filepath: Path, varname: str, vals: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    flat = np.asarray(vals).reshape(-1)
    flat = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.number) else flat
    if flat.size == 0:
        return rows
    # For fields that are explicitly flag-like, retain value counts even if non-integer.
    is_flag_name = any(w in varname.casefold() for w in FLAG_WORDS)
    if not is_flag_name and not np.issubdtype(flat.dtype, np.integer):
        return rows
    if np.issubdtype(flat.dtype, np.number):
        rounded = np.rint(flat)
        integral_ratio = float(np.mean(np.isclose(flat, rounded)))
        if integral_ratio < 0.999:
            if not is_flag_name:
                return rows
            value_labels = Counter(np.round(flat, 5).tolist())
            for value, count in value_labels.most_common(100):
                rows.append({"source": source, "file": str(filepath), "variable": varname,
                             "entry_type": "value_frequency", "value_or_bit": value,
                             "count": int(count), "fraction": float(count / flat.size),
                             "note": "non-integer values; not decoded as bitmask"})
            return rows
        ints = rounded.astype(np.int64, copy=False)
        uniques, counts = np.unique(ints, return_counts=True)
        if is_flag_name or (len(uniques) <= 64 and np.min(uniques) >= 0 and np.max(uniques) <= 65535):
            if len(uniques) <= 256:
                for value, count in zip(uniques, counts):
                    rows.append({"source": source, "file": str(filepath), "variable": varname,
                                 "entry_type": "value_frequency", "value_or_bit": int(value),
                                 "count": int(count), "fraction": float(count / ints.size),
                                 "note": ""})
            # Decode bit-presence counts only for flag-like fields with nonnegative integers.
            if is_flag_name and np.min(ints) >= 0:
                maxval = int(np.max(ints)) if ints.size else 0
                for bit in range(min(32, maxval.bit_length())):
                    count = int(np.count_nonzero((ints & (1 << bit)) != 0))
                    rows.append({"source": source, "file": str(filepath), "variable": varname,
                                 "entry_type": "bit_presence", "value_or_bit": bit,
                                 "count": count, "fraction": float(count / ints.size),
                                 "note": f"bit value {1 << bit}; presence is not proof of meaning"})
    else:
        counts = Counter(safe_str(x) for x in flat)
        for value, count in counts.most_common(100):
            rows.append({"source": source, "file": str(filepath), "variable": varname,
                         "entry_type": "value_frequency", "value_or_bit": value,
                         "count": int(count), "fraction": float(count / flat.size), "note": ""})
    return rows


def dataset_file_row(source: str, root: Path, path: Path, status: str,
                     dims: str = "", variables: str = "", title: str = "", error: str = "") -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    return {"source": source, "relative_path": rel, "month_from_path_or_name": infer_month(path, root),
            "size_bytes": size, "status": status, "dimensions": dims, "variables": variables,
            "global_title": title, "error": error}


def audit_netcdf_source(source: str, root: Path | None, inventory_rows: list[dict[str, Any]],
                        stats_rows: list[dict[str, Any]], flag_count_rows: list[dict[str, Any]],
                        report_lines: list[str]) -> None:
    report_lines.append(f"\n=== {source} RAW DATA AUDIT ===")
    if root is None or not root.exists():
        report_lines.append("STATUS: ROOT NOT FOUND. Edit the source-root setting in the script.")
        report_lines.append(f"Configured/default root: {CCMP_RAW_ROOT if source == 'CCMP' else OSCAR_RAW_ROOT}")
        return
    files = discover_files(root)
    report_lines.append(f"Root: {root}")
    report_lines.append(f"Discoverable NetCDF/HDF files: {len(files):,}")
    if not files:
        report_lines.append("No supported files found under this root.")
        return

    # Full file inventory, including open failures, gives coverage context even when detailed
    # variable statistics are sampled.
    file_rows_by_path: dict[str, dict[str, Any]] = {}
    for i, path in enumerate(files, 1):
        try:
            with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
                dims = ";".join(f"{k}={v}" for k, v in ds.sizes.items())
                vars_text = ";".join(ds.data_vars.keys())
                row = dataset_file_row(source, root, path, "inventory_ok", dims, vars_text,
                                       safe_str(ds.attrs.get("title", ds.attrs.get("summary", ""))))
        except Exception as exc:
            row = dataset_file_row(source, root, path, "open_error", error=f"{type(exc).__name__}: {exc}")
        inventory_rows.append(row)
        file_rows_by_path[str(path)] = row
        if i % 500 == 0:
            print(f"[{source}] inventoried {i:,}/{len(files):,} files...", flush=True)

    sampled = sample_evenly(files, DATA_FILES_TO_AUDIT_PER_SOURCE)
    report_lines.append(f"Files with detailed variable statistics: {len(sampled):,} of {len(files):,}")
    report_lines.append("Detailed file sample: evenly spaced over sorted file paths (deterministic).")
    var_names_seen: Counter[str] = Counter()
    mask_var_observations: Counter[str] = Counter()
    open_errors = 0
    for i, path in enumerate(sampled, 1):
        try:
            with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
                lat = get_coord(ds, ("latitude", "lat"))
                lon = get_coord(ds, ("longitude", "lon"))
                coord_names = {n for n in (lat.name if lat is not None else None,
                                           lon.name if lon is not None else None) if n}
                for name, da0 in ds.data_vars.items():
                    var_names_seen[name] += 1
                    # Ignore text and scalars; numerical source variables/flags are the audit target.
                    try:
                        if da0.ndim == 0 or not np.issubdtype(da0.dtype, np.number):
                            continue
                    except TypeError:
                        continue
                    da, crop_note = crop_to_domain(da0, ds)
                    try:
                        arr = np.asarray(da.values)
                        if arr.size == 0:
                            continue
                        # Numeric data should usually decode to float if fill values exist.
                        if not np.issubdtype(arr.dtype, np.number):
                            continue
                        finite = np.isfinite(arr)
                        finite_vals = arr[finite]
                        n = int(arr.size)
                        nfinite = int(finite.sum())
                        sampled_for_quantiles = quantile_values(finite_vals) if nfinite else np.array([])
                        q = np.nanpercentile(sampled_for_quantiles, [1, 50, 99]).tolist() if sampled_for_quantiles.size else [np.nan, np.nan, np.nan]
                        is_flag = any(w in name.casefold() for w in FLAG_WORDS)
                        if is_flag:
                            mask_var_observations[name] += 1
                        stats_rows.append({
                            "source": source, "file": str(path), "month": infer_month(path, root),
                            "variable": name, "dims": ",".join(da0.dims), "dtype_decoded": str(da0.dtype),
                            "units": safe_str(da0.attrs.get("units", "")),
                            "long_name": safe_str(da0.attrs.get("long_name", da0.attrs.get("standard_name", ""))),
                            "_FillValue": safe_str(get_attr(da0, "_FillValue")),
                            "missing_value": safe_str(get_attr(da0, "missing_value")),
                            "scale_factor": safe_str(get_attr(da0, "scale_factor")),
                            "add_offset": safe_str(get_attr(da0, "add_offset")),
                            "n_values_in_region_or_file": n, "finite_count": nfinite,
                            "missing_or_nonfinite_count": int(n - nfinite),
                            "finite_fraction": float(nfinite / n) if n else np.nan,
                            "min": float(np.min(finite_vals)) if nfinite else np.nan,
                            "p01_sampled": float(q[0]), "median_sampled": float(q[1]),
                            "p99_sampled": float(q[2]), "max": float(np.max(finite_vals)) if nfinite else np.nan,
                            "mean": float(np.mean(finite_vals, dtype=np.float64)) if nfinite else np.nan,
                            "std": float(np.std(finite_vals, dtype=np.float64)) if nfinite else np.nan,
                            "zero_fraction_of_finite": float(np.count_nonzero(finite_vals == 0) / nfinite) if nfinite else np.nan,
                            "domain_slice": crop_note,
                            "attributes_json": json.dumps({k: safe_str(v, 300) for k, v in da0.attrs.items()}, ensure_ascii=False),
                        })
                        flag_count_rows.extend(flag_rows(source, path, name, arr))
                    except Exception as exc:
                        report_lines.append(f"WARNING: {source} {path.name}:{name} statistics failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            open_errors += 1
            report_lines.append(f"WARNING: {source} detailed open failed for {path}: {type(exc).__name__}: {exc}")
        if i % 100 == 0:
            print(f"[{source}] detailed-audited {i:,}/{len(sampled):,} files...", flush=True)

    # Month coverage based on all file paths, not only the detailed sample.
    months = Counter(row["month_from_path_or_name"] for row in inventory_rows if row["source"] == source)
    report_lines.append(f"Files that failed inventory open: {sum(1 for r in inventory_rows if r['source']==source and r['status']=='open_error'):,}")
    report_lines.append(f"Detailed-open failures: {open_errors:,}")
    report_lines.append("Month/file counts (path/filename-derived; validate unusual filename conventions):")
    for month, count in sorted(months.items()):
        report_lines.append(f"  {month}: {count}")
    report_lines.append("Numeric variables appearing in detailed sample:")
    for name, count in var_names_seen.most_common():
        report_lines.append(f"  {name}: present in {count} sampled file(s)")
    report_lines.append("Flag/mask-like variables observed in detailed sample:")
    if mask_var_observations:
        for name, count in mask_var_observations.most_common():
            report_lines.append(f"  {name}: observed in {count} file-variable sample(s); inspect raw_variable_statistics.csv and flag_value_counts.csv")
    else:
        report_lines.append("  None identified by variable name. This does not rule out encoded flags under other names.")


def decode_string_values(arr: np.ndarray) -> list[str]:
    """Best-effort decode for Argo |S1 arrays including char-array dimensions."""
    a = np.asarray(arr)
    if a.size == 0:
        return []
    if a.dtype.kind in {"S", "U"}:
        if a.dtype.kind == "S" and a.dtype.itemsize == 1 and a.ndim >= 2:
            # NetCDF char arrays conventionally store characters along their final axis.
            flat = a.reshape((-1, a.shape[-1]))
            out = []
            for row in flat:
                out.append(b"".join(row.tolist()).decode("utf-8", "replace").replace("\x00", "").strip())
            return out
        return [safe_str(x.decode("utf-8", "replace") if isinstance(x, bytes) else x).replace("\x00", "").strip() for x in a.reshape(-1)]
    return [safe_str(x) for x in a.reshape(-1)]


def variable_valid_stats(ds: xr.Dataset, name: str) -> tuple[int, int, float, float, float]:
    if name not in ds.variables:
        return (0, 0, np.nan, np.nan, np.nan)
    try:
        da = ds[name]
        if not np.issubdtype(da.dtype, np.number):
            return (0, 0, np.nan, np.nan, np.nan)
        values = np.asarray(da.values, dtype=float)
        valid = np.isfinite(values)
        # Mask both decoded NaNs and exceptionally common Argo fill sentinel if not decoded.
        for sentinel in (99999.0, 9999.0, 999.0, -999.0, -9999.0, 1e20, 1e35):
            valid &= ~np.isclose(values, sentinel, rtol=0, atol=max(abs(sentinel) * 1e-7, 1e-7))
        v = values[valid]
        return (int(values.size), int(v.size), float(np.min(v)) if v.size else np.nan,
                float(np.max(v)) if v.size else np.nan, float(np.mean(v)) if v.size else np.nan)
    except Exception:
        return (0, 0, np.nan, np.nan, np.nan)


def qc_distribution(ds: xr.Dataset, name: str) -> str:
    if name not in ds.variables:
        return ""
    try:
        vals = decode_string_values(np.asarray(ds[name].values))
        counter = Counter(v for v in vals if v and v not in {"\x00", " "})
        return json.dumps(dict(counter.most_common()), ensure_ascii=False)
    except Exception:
        return "unreadable"


def audit_argo(root: Path, inventory_rows: list[dict[str, Any]], report_lines: list[str]) -> None:
    report_lines.append("\n=== ARGO RAW PROFILE AUDIT ===")
    report_lines.append(f"Root: {root}")
    if not root.exists():
        report_lines.append("STATUS: ARGO ROOT NOT FOUND.")
        return
    files = sorted((p for p in root.rglob("*.nc") if p.is_file()), key=lambda p: str(p).casefold())
    report_lines.append(f"Argo .nc profile files found: {len(files):,}")
    if ARGO_MAX_FILES > 0 and len(files) > ARGO_MAX_FILES:
        files = sample_evenly(files, ARGO_MAX_FILES)
        report_lines.append(f"Argo files detailed-audited: {len(files):,} (evenly sampled; change ARGO_MAX_FILES=0 for all).")
    else:
        report_lines.append(f"Argo files detailed-audited: {len(files):,}.")

    prefix_counts: Counter[str] = Counter()
    dac_counts: Counter[str] = Counter()
    month_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    all_station_parameters: Counter[str] = Counter()
    qc_all: dict[str, Counter[str]] = {n: Counter() for n in ("PRES_QC", "TEMP_QC", "PSAL_QC", "PRES_ADJUSTED_QC", "TEMP_ADJUSTED_QC", "PSAL_ADJUSTED_QC")}
    core_missing: Counter[str] = Counter()
    errors = 0

    for i, path in enumerate(files, 1):
        try:
            with xr.open_dataset(path, decode_times=False, mask_and_scale=True) as ds:
                rel = path.relative_to(root)
                parts = rel.parts
                month = parts[0] if len(parts) > 0 else "unknown"
                dac = parts[1] if len(parts) > 1 else "unknown"
                prefix_match = re.match(r"([DRA])", path.name, flags=re.IGNORECASE)
                prefix = prefix_match.group(1).upper() if prefix_match else "unknown"
                prefix_counts[prefix] += 1
                dac_counts[dac] += 1
                month_counts[month] += 1
                core_vars = ["LATITUDE", "LONGITUDE", "JULD", "PRES", "TEMP", "PSAL", "PRES_QC", "TEMP_QC", "PSAL_QC"]
                missing_core = [v for v in core_vars if v not in ds.variables]
                for v in missing_core:
                    core_missing[v] += 1
                station_parameters = ""
                if "STATION_PARAMETERS" in ds.variables:
                    items = decode_string_values(np.asarray(ds["STATION_PARAMETERS"].values))
                    station_parameters = " | ".join(x for x in items if x)
                    for x in items:
                        for token in re.findall(r"PRES|TEMP|PSAL|MTIME|NB_SAMPLE_CTD|PRES_MED|TEMP_MED|PSAL_MED|TEMP_STD|PSAL_STD", x.upper()):
                            all_station_parameters[token] += 1
                # Schema signature based on STATION_PARAMETERS tokens rather than a hard-coded schema number.
                params_upper = station_parameters.upper()
                base = all(p in ds.variables for p in ("PRES", "TEMP", "PSAL"))
                extras = []
                for v in ("MTIME", "NB_SAMPLE_CTD", "PRES_MED", "TEMP_MED", "PSAL_MED", "TEMP_STD", "PSAL_STD"):
                    if v in ds.variables:
                        extras.append(v)
                if not base:
                    schema = "missing_one_or_more_PRES_TEMP_PSAL"
                elif "MTIME" in extras and "NB_SAMPLE_CTD" in extras:
                    schema = "MTIME+NB_SAMPLE_CTD+other"
                elif "MTIME" in extras and any(v in extras for v in ("PRES_MED", "TEMP_MED", "PSAL_MED", "TEMP_STD", "PSAL_STD")):
                    schema = "MTIME+MED_STD"
                elif "MTIME" in extras:
                    schema = "MTIME"
                elif "NB_SAMPLE_CTD" in extras:
                    schema = "NB_SAMPLE_CTD"
                elif extras:
                    schema = "+".join(extras)
                else:
                    schema = "base_PRES_TEMP_PSAL"
                schema_counts[schema] += 1
                dims_text = ";".join(f"{k}={v}" for k, v in ds.sizes.items())
                row: dict[str, Any] = {
                    "source": "ARGO", "relative_path": str(rel), "month_folder": month,
                    "dac": dac, "float_directory": parts[2] if len(parts) > 2 else "unknown",
                    "file_name": path.name, "mode_prefix": prefix, "schema_signature": schema,
                    "station_parameters_decoded": station_parameters, "dimensions": dims_text,
                    "core_variables_missing": ";".join(missing_core),
                    "has_adjusted_temp": "TEMP_ADJUSTED" in ds.variables,
                    "has_adjusted_pressure": "PRES_ADJUSTED" in ds.variables,
                    "has_adjusted_salinity": "PSAL_ADJUSTED" in ds.variables,
                    "data_mode": " | ".join(decode_string_values(np.asarray(ds["DATA_MODE"].values))) if "DATA_MODE" in ds.variables else "",
                    "position_qc": " | ".join(decode_string_values(np.asarray(ds["POSITION_QC"].values))) if "POSITION_QC" in ds.variables else "",
                    "latitude": float(np.asarray(ds["LATITUDE"].values).reshape(-1)[0]) if "LATITUDE" in ds.variables and np.asarray(ds["LATITUDE"].values).size else np.nan,
                    "longitude": float(np.asarray(ds["LONGITUDE"].values).reshape(-1)[0]) if "LONGITUDE" in ds.variables and np.asarray(ds["LONGITUDE"].values).size else np.nan,
                    "juld_raw_first": float(np.asarray(ds["JULD"].values).reshape(-1)[0]) if "JULD" in ds.variables and np.asarray(ds["JULD"].values).size else np.nan,
                    "global_title": safe_str(ds.attrs.get("title", "")),
                    "file_size_bytes": path.stat().st_size,
                }
                for name in ("PRES", "TEMP", "PSAL", "PRES_ADJUSTED", "TEMP_ADJUSTED", "PSAL_ADJUSTED"):
                    n, nv, vmin, vmax, mean = variable_valid_stats(ds, name)
                    row[f"{name}_n"] = n
                    row[f"{name}_valid_n"] = nv
                    row[f"{name}_min"] = vmin
                    row[f"{name}_max"] = vmax
                    row[f"{name}_mean"] = mean
                for name in qc_all:
                    dist = qc_distribution(ds, name)
                    row[f"{name}_distribution"] = dist
                    if dist:
                        try:
                            for qc, count in json.loads(dist).items():
                                qc_all[name][qc] += int(count)
                        except Exception:
                            pass
                inventory_rows.append(row)
        except Exception as exc:
            errors += 1
            inventory_rows.append({"source": "ARGO", "relative_path": str(path), "file_name": path.name,
                                   "status": "open_error", "error": f"{type(exc).__name__}: {exc}"})
        if i % 1000 == 0:
            print(f"[ARGO] audited {i:,}/{len(files):,} profiles...", flush=True)

    report_lines.append(f"Profile open/read errors: {errors:,}")
    report_lines.append("File counts by D/R/A prefix:")
    for key, count in sorted(prefix_counts.items()):
        report_lines.append(f"  {key}: {count:,}")
    report_lines.append("File counts by DAC:")
    for key, count in sorted(dac_counts.items()):
        report_lines.append(f"  {key}: {count:,}")
    report_lines.append("File counts by month folder:")
    for key, count in sorted(month_counts.items()):
        report_lines.append(f"  {key}: {count:,}")
    report_lines.append("Observed schema signatures:")
    for key, count in schema_counts.most_common():
        report_lines.append(f"  {key}: {count:,}")
    report_lines.append("STATION_PARAMETERS tokens observed (best-effort char-array decode):")
    for key, count in all_station_parameters.most_common():
        report_lines.append(f"  {key}: token occurrences={count:,}")
    report_lines.append("Core variable missing-file counts:")
    if core_missing:
        for key, count in core_missing.most_common():
            report_lines.append(f"  {key}: missing in {count:,} file(s)")
    else:
        report_lines.append("  No core-variable absence found in successfully read files.")
    report_lines.append("QC character distributions aggregated across successfully read profiles:")
    for name, counts in qc_all.items():
        if counts:
            report_lines.append(f"  {name}: {dict(counts.most_common())}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("No rows generated.\n", encoding="utf-8")
        return
    # Preserve all keys, including fields specific to Argo rows.
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    pd.DataFrame(rows, columns=keys).to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    start = datetime.now()
    stamp = start.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"raw_source_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ccmp_root = find_source_root(CCMP_RAW_ROOT, "ccmp")
    oscar_root = find_source_root(OSCAR_RAW_ROOT, "oscar")

    report_lines: list[str] = [
        "OceanEmbed Raw-Source Integrity Audit",
        f"Started: {start.isoformat(timespec='seconds')}",
        f"Raw base: {RAW_BASE}",
        f"Output directory: {out_dir}",
        f"OceanEmbed domain crop (when 1-D coordinate axes are found): lat {LAT_MIN}..{LAT_MAX}, lon {LON_MIN}..{LON_MAX}",
        "Interpretation warning: a variable name containing 'mask' or 'valid' is not enough to establish its semantics.",
        "For bit-like variables, bit presence is reported as a diagnostic only; bit meanings require source metadata/code.",
        "",
    ]
    file_inventory_rows: list[dict[str, Any]] = []
    variable_stats_rows: list[dict[str, Any]] = []
    flag_count_rows: list[dict[str, Any]] = []
    argo_rows: list[dict[str, Any]] = []

    audit_netcdf_source("CCMP", ccmp_root, file_inventory_rows, variable_stats_rows, flag_count_rows, report_lines)
    audit_netcdf_source("OSCAR", oscar_root, file_inventory_rows, variable_stats_rows, flag_count_rows, report_lines)
    audit_argo(ARGO_RAW_ROOT, argo_rows, report_lines)

    # One-line interpretation guidance for the metadata issue raised in the conversation.
    report_lines.extend([
        "\n=== HOW TO USE THESE RESULTS ===",
        "CCMP: inspect variable names that contain 'valid', 'mask', 'flag', or 'qc', and compare their value/bit distributions with the actual raw wind-variable finite coverage.",
        "OSCAR: inspect the same flag-like fields, if present. Integer values such as 0,1,2,3,5,7,11,12,15 may be bit combinations, but do not assign meanings without the producer's definition.",
        "If the raw CCMP/OSCAR files contain no such mask fields, the source_valid_mask variables are likely created by the merge/preprocessing pipeline; audit that code and the merged files next.",
        "Metadata defects on merged helper variables (e.g., *_valid or both_valid inheriting physical-variable units/ranges) should be checked primarily in the merged NetCDF files and the merge writer; raw source attributes are the comparison baseline.",
        "Argo is audited here as a separate raw-profile inventory/QA check. This script does not use Argo to train or reconstruct OceanEmbed fields.",
        "",
    ])
    report_lines.append(f"Finished: {datetime.now().isoformat(timespec='seconds')}")

    report_path = out_dir / "raw_source_audit_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    write_csv(out_dir / "raw_file_inventory.csv", file_inventory_rows)
    write_csv(out_dir / "raw_variable_statistics.csv", variable_stats_rows)
    write_csv(out_dir / "raw_flag_value_counts.csv", flag_count_rows)
    write_csv(out_dir / "argo_profile_inventory.csv", argo_rows)

    run_info = {
        "started": start.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "raw_base": str(RAW_BASE),
        "ccmp_root_used": str(ccmp_root) if ccmp_root else None,
        "oscar_root_used": str(oscar_root) if oscar_root else None,
        "argo_root_used": str(ARGO_RAW_ROOT),
        "output_dir": str(out_dir),
        "config": {
            "DATA_FILES_TO_AUDIT_PER_SOURCE": DATA_FILES_TO_AUDIT_PER_SOURCE,
            "ARGO_MAX_FILES": ARGO_MAX_FILES,
            "domain": [LAT_MIN, LAT_MAX, LON_MIN, LON_MAX],
        },
        "row_counts": {
            "file_inventory": len(file_inventory_rows),
            "variable_statistics": len(variable_stats_rows),
            "flag_value_counts": len(flag_count_rows),
            "argo_profile_inventory": len(argo_rows),
        },
        "outputs": [
            "raw_source_audit_report.txt", "raw_file_inventory.csv",
            "raw_variable_statistics.csv", "raw_flag_value_counts.csv",
            "argo_profile_inventory.csv", "run_manifest.json",
        ],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(run_info, indent=2, default=json_default), encoding="utf-8")

    print("\nAUDIT COMPLETED")
    print(f"Output folder: {out_dir}")
    print(f"Main report:   {report_path}")
    print(f"CCMP root:     {ccmp_root if ccmp_root else 'NOT FOUND - edit CCMP_RAW_ROOT'}")
    print(f"OSCAR root:    {oscar_root if oscar_root else 'NOT FOUND - edit OSCAR_RAW_ROOT'}")
    print(f"Argo root:     {ARGO_RAW_ROOT if ARGO_RAW_ROOT.exists() else 'NOT FOUND'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nAudit interrupted by user. Any already-created output files may be incomplete.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
