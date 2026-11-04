from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


# ============================================================
# OSCAR SCIENTIFIC + PIPELINE READINESS AUDIT
# ============================================================
#
# PURPOSE
# -------
# Audit the OSCAR surface current dataset for:
#
# 1. Scientific suitability
# 2. Spatial coverage
# 3. Temporal coverage
# 4. File/calendar completeness
# 5. NetCDF readability
# 6. Variables and dimensions
# 7. Coordinates and resolution
# 8. Units and metadata
# 9. Missing-value behavior
# 10. Physical sanity
# 11. U/V consistency
# 12. Duplicate dates/files
# 13. Dataset structure consistency
# 14. Pipeline automation readiness
#
# IMPORTANT
# ---------
# This script NEVER modifies raw OSCAR data.
#
# It only READS the raw files and writes audit outputs.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\oscar_audit"
)

START_DATE = "2012-01-01"
END_DATE = "2024-12-31"

EXPECTED_DAYS = 4749

# OceanEmbed target domain
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

# Expected OSCAR surface current product characteristics.
# These are EXPECTATIONS, not hard rejection rules.
EXPECTED_RESOLUTION = 0.25

# Broad physical sanity limits for surface ocean currents.
# These are intentionally conservative.
U_MIN = -10.0
U_MAX = 10.0
V_MIN = -10.0
V_MAX = 10.0
SPEED_MAX = 15.0


# ============================================================
# OUTPUT FILES
# ============================================================

REPORT_TXT = OUTPUT_DIR / "oscar_scientific_audit_report.txt"
REPORT_JSON = OUTPUT_DIR / "oscar_scientific_audit.json"

FILE_INVENTORY = OUTPUT_DIR / "oscar_file_inventory.csv"
DAILY_COVERAGE = OUTPUT_DIR / "oscar_daily_coverage.csv"
MONTHLY_COVERAGE = OUTPUT_DIR / "oscar_monthly_coverage.csv"
YEARLY_COVERAGE = OUTPUT_DIR / "oscar_yearly_coverage.csv"

PROBLEM_FILES = OUTPUT_DIR / "oscar_problem_files.csv"
DUPLICATE_DATES = OUTPUT_DIR / "oscar_duplicate_dates.csv"
STRUCTURE_SUMMARY = OUTPUT_DIR / "oscar_structure_summary.csv"
VARIABLE_SUMMARY = OUTPUT_DIR / "oscar_variable_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ist_from_utc(dt: datetime) -> datetime:
    from datetime import timedelta
    return dt.astimezone(timezone.utc).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )


def iso(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.isoformat()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        arr = np.asarray(value)

        if arr.size == 0:
            return None

        x = arr.reshape(-1)[0]

        if np.ma.isMaskedArray(x):
            if np.ma.is_masked(x):
                return None
            x = x.item()

        x = float(x)

        if not math.isfinite(x):
            return None

        return x

    except Exception:
        return None


def safe_str(value: Any) -> str:
    try:
        if value is None:
            return ""

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")

        if isinstance(value, np.ndarray):
            if value.size == 1:
                return safe_str(value.reshape(-1)[0])
            return json.dumps(value.tolist(), default=str)

        return str(value)

    except Exception:
        return ""


def json_safe(value: Any) -> Any:
    """
    Convert arbitrary xarray / numpy metadata into JSON-safe objects.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    return str(value)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def extract_date_from_filename(path: Path) -> str | None:
    """
    Expected examples:

    oscar_currents_final_20120101_uv.nc
    oscar_currents_final_20241231_uv.nc
    """

    match = re.search(r"(20\d{6})", path.name)

    if match:
        raw = match.group(1)

        try:
            datetime.strptime(raw, "%Y%m%d")
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        except ValueError:
            return None

    return None


def flatten_attrs(attrs: dict[str, Any]) -> str:
    try:
        return json.dumps(
            json_safe(attrs),
            sort_keys=True,
            ensure_ascii=False,
        )
    except Exception:
        return "{}"


def coordinate_signature(ds: xr.Dataset) -> str:
    parts = []

    for name in sorted(ds.coords):
        try:
            coord = ds[name]

            values = np.asarray(coord.values)

            if values.size == 0:
                signature = {
                    "name": name,
                    "dims": list(coord.dims),
                    "shape": list(values.shape),
                }
            else:
                signature = {
                    "name": name,
                    "dims": list(coord.dims),
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "first": json_safe(values.reshape(-1)[0]),
                    "last": json_safe(values.reshape(-1)[-1]),
                }

            parts.append(signature)

        except Exception as exc:
            parts.append({
                "name": name,
                "error": str(exc),
            })

    payload = json.dumps(
        parts,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def variable_signature(ds: xr.Dataset) -> str:
    payload = []

    for name in sorted(ds.data_vars):
        try:
            var = ds[name]

            payload.append({
                "name": name,
                "dims": list(var.dims),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "attrs": json_safe(dict(var.attrs)),
            })

        except Exception as exc:
            payload.append({
                "name": name,
                "error": str(exc),
            })

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_uv_variables(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """
    Try to identify U/V current variables without assuming
    exact capitalization or naming.
    """

    names = list(ds.data_vars)

    normalized = {
        normalize_name(name): name
        for name in names
    }

    u_candidates = [
        "u",
        "uo",
        "ugos",
        "u_current",
        "ucurrent",
        "eastwardvelocity",
        "eastwardcurrent",
        "zonalcurrent",
        "zonalvelocity",
    ]

    v_candidates = [
        "v",
        "vo",
        "vgos",
        "v_current",
        "vcurrent",
        "northwardvelocity",
        "northwardcurrent",
        "meridionalcurrent",
        "meridionalvelocity",
    ]

    u_name = None
    v_name = None

    for candidate in u_candidates:
        key = normalize_name(candidate)

        if key in normalized:
            u_name = normalized[key]
            break

    for candidate in v_candidates:
        key = normalize_name(candidate)

        if key in normalized:
            v_name = normalized[key]
            break

    # Fallback: names containing u/v
    if u_name is None:
        for name in names:
            n = normalize_name(name)

            if n.startswith("u") and len(n) <= 8:
                u_name = name
                break

    if v_name is None:
        for name in names:
            n = normalize_name(name)

            if n.startswith("v") and len(n) <= 8:
                v_name = name
                break

    return u_name, v_name


def find_lat_lon(ds: xr.Dataset) -> tuple[str | None, str | None]:

    lat_candidates = [
        "lat",
        "latitude",
        "y",
    ]

    lon_candidates = [
        "lon",
        "longitude",
        "x",
    ]

    normalized = {
        normalize_name(name): name
        for name in ds.coords
    }

    lat_name = None
    lon_name = None

    for candidate in lat_candidates:
        key = normalize_name(candidate)

        if key in normalized:
            lat_name = normalized[key]
            break

    for candidate in lon_candidates:
        key = normalize_name(candidate)

        if key in normalized:
            lon_name = normalized[key]
            break

    return lat_name, lon_name


def find_time_variable(ds: xr.Dataset) -> str | None:

    candidates = [
        "time",
        "date",
        "datetime",
    ]

    normalized = {
        normalize_name(name): name
        for name in ds.coords
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def get_valid_numeric_array(
    da: xr.DataArray,
) -> np.ndarray:

    try:
        arr = da.values

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        arr = np.asarray(arr, dtype=np.float64)

        return arr

    except Exception:
        return np.array([], dtype=np.float64)


def numeric_stats(arr: np.ndarray) -> dict[str, Any]:

    if arr.size == 0:
        return {
            "count": 0,
            "valid": 0,
            "nan": 0,
            "min": None,
            "max": None,
            "mean": None,
        }

    finite = np.isfinite(arr)

    valid = arr[finite]

    if valid.size == 0:
        return {
            "count": int(arr.size),
            "valid": 0,
            "nan": int(arr.size),
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "count": int(arr.size),
        "valid": int(valid.size),
        "nan": int(arr.size - valid.size),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
    }


# ============================================================
# CSV SAFETY
# ============================================================

def write_csv_safe(
    path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: list[str] | None = None,
) -> None:
    """
    Robust CSV writer.

    IMPORTANT:
    The previous OSCAR audit failed here because some rows contained
    additional keys not present in fieldnames.

    This function automatically discovers ALL keys across ALL rows
    and adds them to the CSV schema.

    Therefore an unexpected audit field can never terminate the
    complete audit at the final serialization stage.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    all_fields = []

    if preferred_fields:
        all_fields.extend(preferred_fields)

    for row in rows:
        for key in row.keys():
            if key not in all_fields:
                all_fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:

            safe_row = {}

            for field in all_fields:

                value = row.get(field, "")

                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(
                        json_safe(value),
                        ensure_ascii=False,
                    )

                elif isinstance(value, np.generic):
                    value = json_safe(value)

                safe_row[field] = value

            writer.writerow(safe_row)


# ============================================================
# MAIN AUDIT
# ============================================================

def main() -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_start_utc = utc_now()
    audit_start_ist = ist_from_utc(audit_start_utc)

    run_id = (
        "OSCAR-"
        + audit_start_utc.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + hashlib.sha1(
            f"{audit_start_utc.timestamp()}".encode()
        ).hexdigest()[:8].upper()
    )

    print()
    print("=" * 72)
    print("OSCAR SCIENTIFIC + PIPELINE READINESS AUDIT")
    print("=" * 72)
    print()
    print("Run ID:", run_id)
    print("Start UTC:", iso(audit_start_utc))
    print("Start IST:", iso(audit_start_ist))
    print("Raw root:", RAW_ROOT)
    print("Output:", OUTPUT_DIR)
    print()

    if not RAW_ROOT.exists():
        print("ERROR: RAW_ROOT does not exist:")
        print(RAW_ROOT)
        return 1

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = sorted(
        RAW_ROOT.rglob("*.nc")
    )

    print("NetCDF files discovered:", len(files))
    print()

    # --------------------------------------------------------
    # Expected dates
    # --------------------------------------------------------

    expected_dates = []

    current = datetime.strptime(
        START_DATE,
        "%Y-%m-%d",
    )

    end = datetime.strptime(
        END_DATE,
        "%Y-%m-%d",
    )

    while current <= end:
        expected_dates.append(
            current.strftime("%Y-%m-%d")
        )

        from datetime import timedelta
        current += timedelta(days=1)

    expected_date_set = set(expected_dates)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    inventory_rows: list[dict[str, Any]] = []
    problem_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []

    structure_counter = Counter()
    variable_counter = Counter()

    date_to_files: defaultdict[str, list[str]] = defaultdict(list)

    observed_dates = set()

    total_opened = 0
    total_failed = 0

    total_lat_outside = 0
    total_lon_outside = 0

    total_u_nan = 0
    total_v_nan = 0
    total_u_zero = 0
    total_v_zero = 0
    total_both_zero = 0

    total_u_values = 0
    total_v_values = 0

    total_u_outlier_files = 0
    total_v_outlier_files = 0
    total_speed_outlier_files = 0

    global_u_min = None
    global_u_max = None
    global_v_min = None
    global_v_max = None
    global_speed_max = None

    global_lat_min = None
    global_lat_max = None
    global_lon_min = None
    global_lon_max = None

    coordinate_signatures = Counter()
    variable_signatures = Counter()

    representative_metadata = None

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    for index, path in enumerate(files, start=1):

        print(
            f"[{index}/{len(files)}] {path.name}",
            flush=True,
        )

        row = {
            "file": path.name,
            "relative_path": str(
                path.relative_to(RAW_ROOT)
            ),
            "full_path": str(path),
            "size_bytes": path.stat().st_size,
            "filename_date": extract_date_from_filename(path),
            "opened": False,
            "error": "",
            "u_variable": "",
            "v_variable": "",
            "lat_variable": "",
            "lon_variable": "",
            "time_variable": "",
            "variables": "",
            "coordinates": "",
            "dataset_attrs": "",
            "coordinate_signature": "",
            "variable_signature": "",
            "lat_min": "",
            "lat_max": "",
            "lon_min": "",
            "lon_max": "",
            "u_min": "",
            "u_max": "",
            "v_min": "",
            "v_max": "",
            "speed_max": "",
            "u_valid": "",
            "u_nan": "",
            "v_valid": "",
            "v_nan": "",
            "u_zero": "",
            "v_zero": "",
            "both_zero": "",
            "lat_inside_nio": "",
            "lon_inside_nio": "",
            "physical_sanity": "",
            "structure_signature": "",
        }

        try:

            with xr.open_dataset(
                path,
                decode_cf=True,
                mask_and_scale=True,
            ) as ds:

                total_opened += 1
                row["opened"] = True

                # --------------------------------------------
                # Variables / coordinates
                # --------------------------------------------

                u_name, v_name = find_uv_variables(ds)
                lat_name, lon_name = find_lat_lon(ds)
                time_name = find_time_variable(ds)

                row["u_variable"] = u_name or ""
                row["v_variable"] = v_name or ""
                row["lat_variable"] = lat_name or ""
                row["lon_variable"] = lon_name or ""
                row["time_variable"] = time_name or ""

                row["variables"] = json.dumps(
                    list(ds.data_vars),
                    ensure_ascii=False,
                )

                row["coordinates"] = json.dumps(
                    list(ds.coords),
                    ensure_ascii=False,
                )

                row["dataset_attrs"] = flatten_attrs(
                    dict(ds.attrs)
                )

                # --------------------------------------------
                # Structure
                # --------------------------------------------

                structure_payload = {
                    "dims": {
                        str(k): int(v)
                        for k, v in ds.sizes.items()
                    },
                    "data_vars": sorted(
                        str(v)
                        for v in ds.data_vars
                    ),
                    "coords": sorted(
                        str(v)
                        for v in ds.coords
                    ),
                }

                structure_signature = hashlib.sha256(
                    json.dumps(
                        structure_payload,
                        sort_keys=True,
                    ).encode()
                ).hexdigest()

                row["structure_signature"] = (
                    structure_signature
                )

                structure_counter[
                    json.dumps(
                        structure_payload,
                        sort_keys=True,
                    )
                ] += 1

                # --------------------------------------------
                # Signatures
                # --------------------------------------------

                coord_sig = coordinate_signature(ds)
                var_sig = variable_signature(ds)

                row["coordinate_signature"] = coord_sig
                row["variable_signature"] = var_sig

                coordinate_signatures[coord_sig] += 1
                variable_signatures[var_sig] += 1

                if representative_metadata is None:

                    representative_metadata = {
                        "variables": list(ds.data_vars),
                        "coordinates": list(ds.coords),
                        "dims": {
                            str(k): int(v)
                            for k, v in ds.sizes.items()
                        },
                        "attrs": json_safe(
                            dict(ds.attrs)
                        ),
                    }

                # --------------------------------------------
                # Date
                # --------------------------------------------

                file_date = row["filename_date"]

                if file_date:
                    observed_dates.add(file_date)
                    date_to_files[file_date].append(
                        str(path)
                    )

                # --------------------------------------------
                # Coordinates
                # --------------------------------------------

                lat_arr = None
                lon_arr = None

                if lat_name:
                    lat_arr = get_valid_numeric_array(
                        ds[lat_name]
                    )

                if lon_name:
                    lon_arr = get_valid_numeric_array(
                        ds[lon_name]
                    )

                if lat_arr is not None and lat_arr.size:

                    row["lat_min"] = float(
                        np.nanmin(lat_arr)
                    )

                    row["lat_max"] = float(
                        np.nanmax(lat_arr)
                    )

                    global_lat_min = (
                        row["lat_min"]
                        if global_lat_min is None
                        else min(
                            global_lat_min,
                            row["lat_min"],
                        )
                    )

                    global_lat_max = (
                        row["lat_max"]
                        if global_lat_max is None
                        else max(
                            global_lat_max,
                            row["lat_max"],
                        )
                    )

                    if (
                        row["lat_min"] < LAT_MIN
                        or row["lat_max"] > LAT_MAX
                    ):
                        total_lat_outside += 1

                    row["lat_inside_nio"] = (
                        row["lat_min"] >= LAT_MIN
                        and row["lat_max"] <= LAT_MAX
                    )

                if lon_arr is not None and lon_arr.size:

                    row["lon_min"] = float(
                        np.nanmin(lon_arr)
                    )

                    row["lon_max"] = float(
                        np.nanmax(lon_arr)
                    )

                    global_lon_min = (
                        row["lon_min"]
                        if global_lon_min is None
                        else min(
                            global_lon_min,
                            row["lon_min"],
                        )
                    )

                    global_lon_max = (
                        row["lon_max"]
                        if global_lon_max is None
                        else max(
                            global_lon_max,
                            row["lon_max"],
                        )
                    )

                    if (
                        row["lon_min"] < LON_MIN
                        or row["lon_max"] > LON_MAX
                    ):
                        total_lon_outside += 1

                    row["lon_inside_nio"] = (
                        row["lon_min"] >= LON_MIN
                        and row["lon_max"] <= LON_MAX
                    )

                # --------------------------------------------
                # U/V scientific audit
                # --------------------------------------------

                u_arr = np.array([], dtype=np.float64)
                v_arr = np.array([], dtype=np.float64)

                if u_name:
                    u_arr = get_valid_numeric_array(
                        ds[u_name]
                    )

                if v_name:
                    v_arr = get_valid_numeric_array(
                        ds[v_name]
                    )

                u_stats = numeric_stats(u_arr)
                v_stats = numeric_stats(v_arr)

                row["u_valid"] = u_stats["valid"]
                row["u_nan"] = u_stats["nan"]

                row["v_valid"] = v_stats["valid"]
                row["v_nan"] = v_stats["nan"]

                total_u_values += u_stats["count"]
                total_v_values += v_stats["count"]

                total_u_nan += u_stats["nan"]
                total_v_nan += v_stats["nan"]

                if u_arr.size:

                    finite_u = np.isfinite(u_arr)

                    valid_u = u_arr[finite_u]

                    row["u_min"] = (
                        float(np.min(valid_u))
                        if valid_u.size
                        else ""
                    )

                    row["u_max"] = (
                        float(np.max(valid_u))
                        if valid_u.size
                        else ""
                    )

                    if valid_u.size:

                        global_u_min = (
                            row["u_min"]
                            if global_u_min is None
                            else min(
                                global_u_min,
                                row["u_min"],
                            )
                        )

                        global_u_max = (
                            row["u_max"]
                            if global_u_max is None
                            else max(
                                global_u_max,
                                row["u_max"],
                            )
                        )

                        zeros = int(
                            np.count_nonzero(
                                valid_u == 0
                            )
                        )

                        row["u_zero"] = zeros
                        total_u_zero += zeros

                        if (
                            np.any(valid_u < U_MIN)
                            or np.any(valid_u > U_MAX)
                        ):
                            total_u_outlier_files += 1

                if v_arr.size:

                    finite_v = np.isfinite(v_arr)

                    valid_v = v_arr[finite_v]

                    row["v_min"] = (
                        float(np.min(valid_v))
                        if valid_v.size
                        else ""
                    )

                    row["v_max"] = (
                        float(np.max(valid_v))
                        if valid_v.size
                        else ""
                    )

                    if valid_v.size:

                        global_v_min = (
                            row["v_min"]
                            if global_v_min is None
                            else min(
                                global_v_min,
                                row["v_min"],
                            )
                        )

                        global_v_max = (
                            row["v_max"]
                            if global_v_max is None
                            else max(
                                global_v_max,
                                row["v_max"],
                            )
                        )

                        zeros = int(
                            np.count_nonzero(
                                valid_v == 0
                            )
                        )

                        row["v_zero"] = zeros
                        total_v_zero += zeros

                        if (
                            np.any(valid_v < V_MIN)
                            or np.any(valid_v > V_MAX)
                        ):
                            total_v_outlier_files += 1

                # --------------------------------------------
                # Paired U/V zero test
                # --------------------------------------------

                if (
                    u_arr.size
                    and v_arr.size
                    and u_arr.shape == v_arr.shape
                ):

                    both_valid = (
                        np.isfinite(u_arr)
                        & np.isfinite(v_arr)
                    )

                    both_zero = int(
                        np.count_nonzero(
                            both_valid
                            & (u_arr == 0)
                            & (v_arr == 0)
                        )
                    )

                    row["both_zero"] = both_zero
                    total_both_zero += both_zero

                    speed = np.sqrt(
                        u_arr[both_valid] ** 2
                        + v_arr[both_valid] ** 2
                    )

                    if speed.size:

                        row["speed_max"] = float(
                            np.max(speed)
                        )

                        global_speed_max = (
                            row["speed_max"]
                            if global_speed_max is None
                            else max(
                                global_speed_max,
                                row["speed_max"],
                            )
                        )

                        if np.any(
                            speed > SPEED_MAX
                        ):
                            total_speed_outlier_files += 1

                # --------------------------------------------
                # Variable units
                # --------------------------------------------

                if u_name:
                    variable_counter[
                        f"{u_name} | "
                        f"{ds[u_name].attrs.get('units', '')}"
                    ] += 1

                if v_name:
                    variable_counter[
                        f"{v_name} | "
                        f"{ds[v_name].attrs.get('units', '')}"
                    ] += 1

                # --------------------------------------------
                # Physical sanity
                # --------------------------------------------

                row["physical_sanity"] = (
                    (
                        u_name is not None
                        and v_name is not None
                    )
                    and (
                        row["u_min"] == ""
                        or (
                            row["u_min"] >= U_MIN
                            and row["u_max"] <= U_MAX
                        )
                    )
                    and (
                        row["v_min"] == ""
                        or (
                            row["v_min"] >= V_MIN
                            and row["v_max"] <= V_MAX
                        )
                    )
                    and (
                        row["speed_max"] == ""
                        or row["speed_max"] <= SPEED_MAX
                    )
                )

                # --------------------------------------------
                # Problem classification
                # --------------------------------------------

                problems = []

                if u_name is None:
                    problems.append(
                        "U_VARIABLE_NOT_FOUND"
                    )

                if v_name is None:
                    problems.append(
                        "V_VARIABLE_NOT_FOUND"
                    )

                if lat_name is None:
                    problems.append(
                        "LATITUDE_NOT_FOUND"
                    )

                if lon_name is None:
                    problems.append(
                        "LONGITUDE_NOT_FOUND"
                    )

                if file_date is None:
                    problems.append(
                        "DATE_NOT_FOUND_IN_FILENAME"
                    )

                if (
                    row["lat_inside_nio"] is False
                ):
                    problems.append(
                        "LATITUDE_OUTSIDE_EXPECTED_DOMAIN"
                    )

                if (
                    row["lon_inside_nio"] is False
                ):
                    problems.append(
                        "LONGITUDE_OUTSIDE_EXPECTED_DOMAIN"
                    )

                if not row["physical_sanity"]:
                    problems.append(
                        "PHYSICAL_SANITY_WARNING"
                    )

                if problems:

                    problem_rows.append({
                        "file": row["file"],
                        "relative_path": row[
                            "relative_path"
                        ],
                        "date": file_date or "",
                        "problems": ";".join(
                            problems
                        ),
                        "u_min": row["u_min"],
                        "u_max": row["u_max"],
                        "v_min": row["v_min"],
                        "v_max": row["v_max"],
                        "speed_max": row[
                            "speed_max"
                        ],
                    })

        except Exception as exc:

            total_failed += 1

            row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            problem_rows.append({
                "file": row["file"],
                "relative_path": row[
                    "relative_path"
                ],
                "date": row["filename_date"] or "",
                "problems": "OPEN_FAILED",
                "u_min": "",
                "u_max": "",
                "v_min": "",
                "v_max": "",
                "speed_max": "",
            })

        inventory_rows.append(row)

    # ========================================================
    # DUPLICATE DATES
    # ========================================================

    for date_value, paths in sorted(
        date_to_files.items()
    ):

        if len(paths) > 1:

            duplicate_rows.append({
                "date": date_value,
                "file_count": len(paths),
                "files": json.dumps(
                    paths,
                    ensure_ascii=False,
                ),
                "sizes_bytes": json.dumps([
                    Path(p).stat().st_size
                    for p in paths
                ]),
            })

    # ========================================================
    # COVERAGE TABLES
    # ========================================================

    daily_rows = []

    for date_value in expected_dates:

        paths = date_to_files.get(
            date_value,
            [],
        )

        daily_rows.append({
            "date": date_value,
            "file_count": len(paths),
            "present": bool(paths),
            "files": json.dumps(
                paths,
                ensure_ascii=False,
            ),
        })

    monthly_counter = Counter()

    for date_value in observed_dates:

        month = date_value[:7]

        monthly_counter[month] += 1

    monthly_rows = []

    for month in sorted(
        {
            d[:7]
            for d in expected_dates
        }
    ):

        year, mon = month.split("-")

        expected_month_days = sum(
            1
            for d in expected_dates
            if d.startswith(month)
        )

        observed = monthly_counter.get(
            month,
            0,
        )

        monthly_rows.append({
            "month": month,
            "year": int(year),
            "month_number": int(mon),
            "expected_days": expected_month_days,
            "observed_days": observed,
            "missing_days": (
                expected_month_days
                - observed
            ),
            "complete": (
                observed
                == expected_month_days
            ),
        })

    yearly_rows = []

    for year in range(
        int(START_DATE[:4]),
        int(END_DATE[:4]) + 1,
    ):

        year_str = str(year)

        expected_year_days = sum(
            1
            for d in expected_dates
            if d.startswith(year_str)
        )

        observed_year_days = sum(
            1
            for d in observed_dates
            if d.startswith(year_str)
        )

        yearly_rows.append({
            "year": year,
            "expected_days": expected_year_days,
            "observed_days": observed_year_days,
            "missing_days": (
                expected_year_days
                - observed_year_days
            ),
            "complete": (
                expected_year_days
                == observed_year_days
            ),
        })

    # ========================================================
    # STRUCTURE SUMMARY
    # ========================================================

    structure_rows = []

    for signature, count in structure_counter.items():

        structure_rows.append({
            "structure": signature,
            "file_count": count,
        })

    # ========================================================
    # VARIABLE SUMMARY
    # ========================================================

    variable_rows = []

    for signature, count in variable_counter.items():

        if " | " in signature:

            variable, units = signature.split(
                " | ",
                1,
            )

        else:

            variable = signature
            units = ""

        variable_rows.append({
            "variable": variable,
            "units": units,
            "file_count": count,
        })

    # ========================================================
    # DECISION FLAGS
    # ========================================================

    all_dates_present = (
        len(
            expected_date_set
            - observed_dates
        )
        == 0
    )

    no_out_of_period_dates = (
        len(
            observed_dates
            - expected_date_set
        )
        == 0
    )

    all_files_opened = (
        total_failed == 0
        and total_opened == len(files)
    )

    all_files_have_uv = all(
        bool(row["u_variable"])
        and bool(row["v_variable"])
        for row in inventory_rows
        if row["opened"]
    )

    all_files_have_coordinates = all(
        bool(row["lat_variable"])
        and bool(row["lon_variable"])
        for row in inventory_rows
        if row["opened"]
    )

    coordinate_consistent = (
        len(coordinate_signatures) <= 1
    )

    variable_structure_consistent = (
        len(variable_signatures) <= 1
    )

    spatial_ok = (
        total_lat_outside == 0
        and total_lon_outside == 0
    )

    physical_ok = (
        total_u_outlier_files == 0
        and total_v_outlier_files == 0
        and total_speed_outlier_files == 0
    )

    structural_ok = (
        all_files_opened
        and all_files_have_uv
        and all_files_have_coordinates
    )

    calendar_ok = (
        all_dates_present
        and no_out_of_period_dates
    )

    # ========================================================
    # SCIENTIFIC DECISION
    # ========================================================

    if (
        structural_ok
        and spatial_ok
        and calendar_ok
        and physical_ok
    ):

        decision = "GREEN"

        decision_reason = (
            "OSCAR passes the primary scientific and "
            "pipeline-readiness audit. Files are readable, "
            "the expected date range is represented, "
            "U/V variables and coordinates are present, "
            "the requested NIO domain is covered, and no "
            "broad physical sanity failures were detected."
        )

    elif (
        structural_ok
        and calendar_ok
        and spatial_ok
    ):

        decision = "YELLOW"

        decision_reason = (
            "OSCAR is structurally and spatially suitable, "
            "but one or more scientific sanity checks "
            "require targeted investigation before final "
            "acceptance."
        )

    else:

        decision = "RED"

        decision_reason = (
            "OSCAR has one or more fundamental acquisition/"
            "structure/coverage problems that prevent "
            "immediate acceptance."
        )

    # ========================================================
    # END TIMESTAMP
    # ========================================================

    audit_end_utc = utc_now()
    audit_end_ist = ist_from_utc(
        audit_end_utc
    )

    duration_seconds = (
        audit_end_utc - audit_start_utc
    ).total_seconds()

    # ========================================================
    # FINAL REPORT OBJECT
    # ========================================================

    report = {

        "audit": {

            "run_id": run_id,

            "start_utc": iso(
                audit_start_utc
            ),

            "end_utc": iso(
                audit_end_utc
            ),

            "start_ist": iso(
                audit_start_ist
            ),

            "end_ist": iso(
                audit_end_ist
            ),

            "duration_seconds": duration_seconds,

            "raw_root": str(
                RAW_ROOT
            ),

            "output_dir": str(
                OUTPUT_DIR
            ),

            "raw_data_modified": False,

            "expected_period": {
                "start": START_DATE,
                "end": END_DATE,
                "expected_days": EXPECTED_DAYS,
            },

            "expected_domain": {
                "latitude_min": LAT_MIN,
                "latitude_max": LAT_MAX,
                "longitude_min": LON_MIN,
                "longitude_max": LON_MAX,
            },
        },

        "inventory": {

            "netcdf_files_discovered": len(files),

            "files_opened": total_opened,

            "files_failed": total_failed,

            "observed_unique_dates": len(
                observed_dates
            ),

            "expected_dates": len(
                expected_dates
            ),

            "missing_dates": len(
                expected_date_set
                - observed_dates
            ),

            "unexpected_dates": len(
                observed_dates
                - expected_date_set
            ),

            "duplicate_dates": len(
                duplicate_rows
            ),
        },

        "geographic": {

            "global_lat_min": global_lat_min,

            "global_lat_max": global_lat_max,

            "global_lon_min": global_lon_min,

            "global_lon_max": global_lon_max,

            "files_lat_outside": total_lat_outside,

            "files_lon_outside": total_lon_outside,
        },

        "variables": {

            "u_variable_files": Counter(
                row["u_variable"]
                for row in inventory_rows
                if row["u_variable"]
            ),

            "v_variable_files": Counter(
                row["v_variable"]
                for row in inventory_rows
                if row["v_variable"]
            ),

            "coordinate_signature_count": len(
                coordinate_signatures
            ),

            "variable_signature_count": len(
                variable_signatures
            ),
        },

        "scientific": {

            "u_global_min": global_u_min,

            "u_global_max": global_u_max,

            "v_global_min": global_v_min,

            "v_global_max": global_v_max,

            "maximum_current_speed": global_speed_max,

            "total_u_values": total_u_values,

            "total_v_values": total_v_values,

            "total_u_nan": total_u_nan,

            "total_v_nan": total_v_nan,

            "total_u_zero": total_u_zero,

            "total_v_zero": total_v_zero,

            "total_both_zero": total_both_zero,

            "u_outlier_files": total_u_outlier_files,

            "v_outlier_files": total_v_outlier_files,

            "speed_outlier_files": total_speed_outlier_files,
        },

        "decision": {

            "structural_ok": structural_ok,

            "geographic_ok": spatial_ok,

            "calendar_ok": calendar_ok,

            "physical_sanity_ok": physical_ok,

            "coordinate_structure_consistent": (
                coordinate_consistent
            ),

            "variable_structure_consistent": (
                variable_structure_consistent
            ),

            "all_files_scanned_successfully": (
                all_files_opened
            ),

            "status": decision,

            "reason": decision_reason,
        },

        "representative_metadata": (
            representative_metadata
        ),
    }

    # Convert Counters to normal dicts.
    report["variables"][
        "u_variable_files"
    ] = dict(
        report["variables"]["u_variable_files"]
    )

    report["variables"][
        "v_variable_files"
    ] = dict(
        report["variables"]["v_variable_files"]
    )

    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    print()
    print("=" * 72)
    print("Writing audit outputs...")
    print("=" * 72)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        with REPORT_JSON.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                json_safe(report),
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "OK:",
            REPORT_JSON,
        )

    except Exception as exc:

        print(
            "WARNING: JSON report failed:",
            exc,
        )

    # --------------------------------------------------------
    # CSVs
    # --------------------------------------------------------

    csv_jobs = [

        (
            FILE_INVENTORY,
            inventory_rows,
            None,
        ),

        (
            DAILY_COVERAGE,
            daily_rows,
            [
                "date",
                "file_count",
                "present",
                "files",
            ],
        ),

        (
            MONTHLY_COVERAGE,
            monthly_rows,
            [
                "month",
                "year",
                "month_number",
                "expected_days",
                "observed_days",
                "missing_days",
                "complete",
            ],
        ),

        (
            YEARLY_COVERAGE,
            yearly_rows,
            [
                "year",
                "expected_days",
                "observed_days",
                "missing_days",
                "complete",
            ],
        ),

        (
            PROBLEM_FILES,
            problem_rows,
            None,
        ),

        (
            DUPLICATE_DATES,
            duplicate_rows,
            None,
        ),

        (
            STRUCTURE_SUMMARY,
            structure_rows,
            None,
        ),

        (
            VARIABLE_SUMMARY,
            variable_rows,
            None,
        ),
    ]

    csv_failures = []

    for path, rows, fields in csv_jobs:

        try:

            write_csv_safe(
                path,
                rows,
                fields,
            )

            print(
                "OK:",
                path,
            )

        except Exception as exc:

            csv_failures.append({
                "file": str(path),
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            })

            print(
                "WARNING: CSV failed:",
                path,
                exc,
            )

    # --------------------------------------------------------
    # TXT REPORT
    # --------------------------------------------------------

    try:

        with REPORT_TXT.open(
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                "============================================================\n"
            )
            f.write(
                "OSCAR SCIENTIFIC + PIPELINE READINESS AUDIT\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"Audit Run ID: {run_id}\n"
            )

            f.write(
                f"Audit Start UTC: "
                f"{iso(audit_start_utc)}\n"
            )

            f.write(
                f"Audit End UTC: "
                f"{iso(audit_end_utc)}\n"
            )

            f.write(
                f"Audit Start IST: "
                f"{iso(audit_start_ist)}\n"
            )

            f.write(
                f"Audit End IST: "
                f"{iso(audit_end_ist)}\n"
            )

            f.write(
                f"Duration seconds: "
                f"{duration_seconds:.2f}\n"
            )

            f.write(
                f"Raw root: {RAW_ROOT}\n"
            )

            f.write(
                f"Output directory: {OUTPUT_DIR}\n"
            )

            f.write(
                "Raw data modified: NO\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "EXPECTED DATASET\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"Expected period: "
                f"{START_DATE} -> {END_DATE}\n"
            )

            f.write(
                f"Expected days: "
                f"{EXPECTED_DAYS}\n"
            )

            f.write(
                f"Expected latitude: "
                f"{LAT_MIN} -> {LAT_MAX}\n"
            )

            f.write(
                f"Expected longitude: "
                f"{LON_MIN} -> {LON_MAX}\n"
            )

            f.write("\n")

            f.write(
                "============================================================\n"
            )
            f.write(
                "FILE INVENTORY\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"NetCDF files discovered: "
                f"{len(files)}\n"
            )

            f.write(
                f"Files opened successfully: "
                f"{total_opened}\n"
            )

            f.write(
                f"Files failed: "
                f"{total_failed}\n"
            )

            f.write(
                f"Unique observed dates: "
                f"{len(observed_dates)}\n"
            )

            f.write(
                f"Expected dates: "
                f"{len(expected_dates)}\n"
            )

            f.write(
                f"Missing dates: "
                f"{len(expected_date_set - observed_dates)}\n"
            )

            f.write(
                f"Unexpected dates: "
                f"{len(observed_dates - expected_date_set)}\n"
            )

            f.write(
                f"Duplicate dates: "
                f"{len(duplicate_rows)}\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "GEOGRAPHIC COVERAGE\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"Global latitude minimum: "
                f"{global_lat_min}\n"
            )

            f.write(
                f"Global latitude maximum: "
                f"{global_lat_max}\n"
            )

            f.write(
                f"Global longitude minimum: "
                f"{global_lon_min}\n"
            )

            f.write(
                f"Global longitude maximum: "
                f"{global_lon_max}\n"
            )

            f.write(
                f"Files with latitude outside domain: "
                f"{total_lat_outside}\n"
            )

            f.write(
                f"Files with longitude outside domain: "
                f"{total_lon_outside}\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "SCIENTIFIC VALUES\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"U global minimum: {global_u_min}\n"
            )

            f.write(
                f"U global maximum: {global_u_max}\n"
            )

            f.write(
                f"V global minimum: {global_v_min}\n"
            )

            f.write(
                f"V global maximum: {global_v_max}\n"
            )

            f.write(
                f"Maximum current speed: "
                f"{global_speed_max}\n"
            )

            f.write(
                f"Total U values: "
                f"{total_u_values}\n"
            )

            f.write(
                f"Total V values: "
                f"{total_v_values}\n"
            )

            f.write(
                f"Total U NaN/missing values: "
                f"{total_u_nan}\n"
            )

            f.write(
                f"Total V NaN/missing values: "
                f"{total_v_nan}\n"
            )

            f.write(
                f"Total U exact-zero values: "
                f"{total_u_zero}\n"
            )

            f.write(
                f"Total V exact-zero values: "
                f"{total_v_zero}\n"
            )

            f.write(
                f"Total locations where U=0 AND V=0: "
                f"{total_both_zero}\n"
            )

            f.write(
                f"Files with U broad-range outliers: "
                f"{total_u_outlier_files}\n"
            )

            f.write(
                f"Files with V broad-range outliers: "
                f"{total_v_outlier_files}\n"
            )

            f.write(
                f"Files with speed > {SPEED_MAX}: "
                f"{total_speed_outlier_files}\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "STRUCTURE\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"Unique coordinate signatures: "
                f"{len(coordinate_signatures)}\n"
            )

            f.write(
                f"Unique variable signatures: "
                f"{len(variable_signatures)}\n"
            )

            f.write(
                f"Coordinate structure consistent: "
                f"{coordinate_consistent}\n"
            )

            f.write(
                f"Variable structure consistent: "
                f"{variable_structure_consistent}\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "DECISION FLAGS\n"
            )
            f.write(
                "============================================================\n\n"
            )

            f.write(
                f"Structural OK: {structural_ok}\n"
            )

            f.write(
                f"Geographic OK: {spatial_ok}\n"
            )

            f.write(
                f"Calendar OK: {calendar_ok}\n"
            )

            f.write(
                f"Physical sanity OK: {physical_ok}\n"
            )

            f.write(
                f"All files scanned successfully: "
                f"{all_files_opened}\n"
            )

            f.write(
                f"FINAL STATUS: {decision}\n\n"
            )

            f.write(
                "Decision reason:\n"
            )

            f.write(
                decision_reason
                + "\n\n"
            )

            f.write(
                "============================================================\n"
            )
            f.write(
                "OUTPUT FILES\n"
            )
            f.write(
                "============================================================\n\n"
            )

            for path, _, _ in csv_jobs:

                f.write(
                    f"{path}\n"
                )

            f.write(
                f"{REPORT_JSON}\n"
            )

            f.write(
                f"{REPORT_TXT}\n"
            )

            f.write("\n")

            f.write(
                "CSV serialization failures: "
                f"{len(csv_failures)}\n"
            )

            for failure in csv_failures:

                f.write(
                    f"- {failure['file']}: "
                    f"{failure['error']}\n"
                )

        print(
            "OK:",
            REPORT_TXT,
        )

    except Exception as exc:

        print(
            "WARNING: TXT report failed:",
            exc,
        )

    # ========================================================
    # FINAL CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("OSCAR AUDIT COMPLETE")
    print("=" * 72)
    print()

    print("Run ID:", run_id)
    print("Files discovered:", len(files))
    print("Files opened:", total_opened)
    print("Files failed:", total_failed)
    print("Expected dates:", len(expected_dates))
    print("Observed dates:", len(observed_dates))
    print(
        "Missing dates:",
        len(expected_date_set - observed_dates),
    )
    print(
        "Duplicate dates:",
        len(duplicate_rows),
    )

    print()
    print("Structural OK:", structural_ok)
    print("Geographic OK:", spatial_ok)
    print("Calendar OK:", calendar_ok)
    print("Physical sanity OK:", physical_ok)
    print(
        "Coordinate structure consistent:",
        coordinate_consistent,
    )
    print(
        "Variable structure consistent:",
        variable_structure_consistent,
    )

    print()
    print("FINAL STATUS:", decision)

    print()
    print(
        "CSV serialization failures:",
        len(csv_failures),
    )

    print()
    print("Report:")
    print(REPORT_TXT)

    print()
    print("JSON:")
    print(REPORT_JSON)

    print()
    print("=" * 72)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "Audit interrupted by user."
        )

        sys.exit(130)

    except Exception:

        print()
        print(
            "FATAL AUDIT ERROR"
        )

        traceback.print_exc()

        sys.exit(1)
