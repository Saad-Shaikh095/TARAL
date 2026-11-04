from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# ARGO SCIENTIFIC AUDIT — OCEANEMBED
# ============================================================
#
# PURPOSE
# -------
# Full scientific audit of all Argo profile NetCDF files.
#
# IMPORTANT
# ---------
# This script ONLY READS the source files.
# It does NOT modify, delete, interpolate, regrid, fill, or
# otherwise alter the original Argo data.
#
# DATE / TIME
# -----------
# Every audit run records:
#   - Run ID
#   - Start/end time UTC
#   - Start/end time IST
#   - Duration
#   - Audit configuration
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

ARGO_ROOT = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\argo"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\argo_audit"
)

EXPECTED_START = pd.Timestamp("2012-01-01")
EXPECTED_END = pd.Timestamp("2024-12-31")

# North Indian Ocean target domain
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

# Broad physical sanity ranges.
# These are NOT the final scientific QC rules.
TEMP_MIN = -5.0
TEMP_MAX = 40.0

PSAL_MIN = 0.0
PSAL_MAX = 45.0

PRES_MIN = 0.0
PRES_MAX = 12000.0

# Pressure/depth coverage thresholds.
DEPTH_THRESHOLDS = [
    0,
    50,
    100,
    200,
    500,
    1000,
    1500,
    2000,
]

# QC flags that are normally considered "good enough" for
# basic scientific screening.
#
# We REPORT all flags. We do NOT remove profiles based on these
# values in this script.
GOOD_QC_FLAGS = {"1", "2"}

CORE_VARS = [
    "LATITUDE",
    "LONGITUDE",
    "JULD",
    "PRES",
    "TEMP",
    "PSAL",
]

QC_VARS = [
    "PRES_QC",
    "TEMP_QC",
    "PSAL_QC",
    "PROFILE_PRES_QC",
    "PROFILE_TEMP_QC",
    "PROFILE_PSAL_QC",
]

ADJUSTED_VARS = [
    "PRES_ADJUSTED",
    "TEMP_ADJUSTED",
    "PSAL_ADJUSTED",
]

IDENTITY_VARS = [
    "PLATFORM_NUMBER",
    "CYCLE_NUMBER",
    "DIRECTION",
    "JULD",
    "LATITUDE",
    "LONGITUDE",
]


# ============================================================
# TIME / RUN ID
# ============================================================

AUDIT_START_UTC = datetime.now(timezone.utc)
AUDIT_START_IST = AUDIT_START_UTC.astimezone(
    timezone(pd.Timedelta(hours=5, minutes=30))
)

RUN_SEED = (
    f"{AUDIT_START_UTC.isoformat()}|"
    f"{ARGO_ROOT}|"
    f"{EXPECTED_START}|"
    f"{EXPECTED_END}"
)

RUN_ID = (
    "ARGO-SCI-"
    + hashlib.sha1(RUN_SEED.encode("utf-8")).hexdigest()[:10].upper()
)


# ============================================================
# OUTPUT SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_TXT = OUTPUT_DIR / "argo_scientific_audit_report.txt"
REPORT_JSON = OUTPUT_DIR / "argo_scientific_audit.json"

PROFILE_CSV = OUTPUT_DIR / "argo_profile_inventory.csv"
DAILY_CSV = OUTPUT_DIR / "argo_daily_coverage.csv"
MONTHLY_CSV = OUTPUT_DIR / "argo_monthly_coverage.csv"
YEARLY_CSV = OUTPUT_DIR / "argo_yearly_coverage.csv"
FLOAT_CSV = OUTPUT_DIR / "argo_float_summary.csv"
QC_CSV = OUTPUT_DIR / "argo_qc_summary.csv"
DEPTH_CSV = OUTPUT_DIR / "argo_depth_coverage.csv"
DUPLICATE_CSV = OUTPUT_DIR / "argo_duplicate_profiles.csv"
PROBLEM_CSV = OUTPUT_DIR / "argo_problem_profiles.csv"


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


def ist_now():
    return utc_now().astimezone(
        timezone(pd.Timedelta(hours=5, minutes=30))
    )


def safe_scalar(ds, name):
    """
    Safely extract a scalar-ish variable.
    """
    if name not in ds.variables:
        return None

    try:
        arr = np.asarray(ds[name].values).reshape(-1)

        if arr.size == 0:
            return None

        value = arr[0]

        if np.ma.isMaskedArray(value):
            if np.ma.is_masked(value):
                return None
            value = value.item()

        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip()

        if isinstance(value, np.generic):
            value = value.item()

        return value

    except Exception:
        return None


def safe_array(ds, name):
    """
    Return decoded NumPy array for a variable.
    """
    if name not in ds.variables:
        return None

    try:
        arr = ds[name].values

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        return np.asarray(arr)

    except Exception:
        return None


def flatten_numeric(arr):
    """
    Flatten numeric array and remove NaN/inf.
    """
    if arr is None:
        return np.array([], dtype=float)

    try:
        a = np.asarray(arr, dtype=float).reshape(-1)
        return a[np.isfinite(a)]
    except Exception:
        return np.array([], dtype=float)


def flatten_strings(arr):
    """
    Flatten QC/string-like arrays safely.
    """
    if arr is None:
        return []

    result = []

    try:
        for x in np.asarray(arr).reshape(-1):
            if x is None:
                continue

            if isinstance(x, bytes):
                s = x.decode("utf-8", errors="ignore")
            else:
                s = str(x)

            s = s.strip()

            if s:
                result.append(s)

    except Exception:
        pass

    return result


def normalize_wmo(ds):
    """
    Extract PLATFORM_NUMBER / WMO identifier.
    """
    for name in [
        "PLATFORM_NUMBER",
        "PLATFORM_NUMBER_ADJUSTED",
    ]:
        if name in ds.variables:
            values = flatten_strings(safe_array(ds, name))

            if values:
                return values[0]

    return None


def normalize_cycle(ds):
    value = safe_scalar(ds, "CYCLE_NUMBER")

    if value is None:
        return None

    try:
        return int(float(value))
    except Exception:
        return str(value)


def normalize_direction(ds):
    value = safe_scalar(ds, "DIRECTION")

    if value is None:
        return None

    return str(value)


def extract_datetime(ds):
    """
    JULD is decoded by xarray when possible.
    """
    if "JULD" not in ds.variables:
        return None

    try:
        values = ds["JULD"].values

        flat = np.asarray(values).reshape(-1)

        for value in flat:
            if value is None:
                continue

            try:
                ts = pd.Timestamp(value)

                if pd.isna(ts):
                    continue

                # Normalize timezone if necessary
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)

                return ts

            except Exception:
                continue

    except Exception:
        return None

    return None


def numeric_extrema(arr):
    values = flatten_numeric(arr)

    if values.size == 0:
        return None, None

    return float(np.min(values)), float(np.max(values))


def count_valid(arr):
    return int(flatten_numeric(arr).size)


def count_total(arr):
    if arr is None:
        return 0

    try:
        return int(np.asarray(arr).size)
    except Exception:
        return 0


def qc_counts(arr):
    """
    Return counts for QC flags.
    """
    counts = Counter()

    for value in flatten_strings(arr):
        # Some Argo QC arrays can be represented as characters.
        value = value.strip()

        if len(value) > 1:
            # Keep first meaningful character where applicable.
            value = value[0]

        counts[value] += 1

    return counts


def pressure_statistics(pres):
    """
    Analyze pressure ordering and duplicate levels.

    Returns:
      valid_count
      monotonic_non_decreasing
      strictly_increasing
      duplicate_adjacent_count
      minimum
      maximum
    """
    values = flatten_numeric(pres)

    if values.size == 0:
        return {
            "valid_count": 0,
            "monotonic_non_decreasing": None,
            "strictly_increasing": None,
            "duplicate_adjacent_count": 0,
            "min": None,
            "max": None,
        }

    # Remove no values here; ordering must be checked on the actual
    # finite sequence, preserving original order.
    raw = np.asarray(pres, dtype=float).reshape(-1)
    raw = raw[np.isfinite(raw)]

    if raw.size <= 1:
        return {
            "valid_count": int(raw.size),
            "monotonic_non_decreasing": True,
            "strictly_increasing": True,
            "duplicate_adjacent_count": 0,
            "min": float(raw.min()) if raw.size else None,
            "max": float(raw.max()) if raw.size else None,
        }

    diffs = np.diff(raw)

    return {
        "valid_count": int(raw.size),
        "monotonic_non_decreasing": bool(np.all(diffs >= 0)),
        "strictly_increasing": bool(np.all(diffs > 0)),
        "duplicate_adjacent_count": int(np.sum(diffs == 0)),
        "min": float(raw.min()),
        "max": float(raw.max()),
    }


def has_level_dimension(arr):
    """
    Just a convenience helper for counting measurements.
    """
    if arr is None:
        return False

    return np.asarray(arr).ndim >= 1


def profile_identity(wmo, cycle, juld, lat, lon, direction):
    """
    Construct a reasonably strong profile identity.

    Exact byte-for-byte duplicate files are not assumed.
    """
    if juld is not None:
        time_key = pd.Timestamp(juld).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        time_key = "NA"

    return (
        str(wmo),
        str(cycle),
        time_key,
        round(float(lat), 5) if lat is not None else None,
        round(float(lon), 5) if lon is not None else None,
        str(direction),
    )


def is_in_domain(lat, lon):
    if lat is None or lon is None:
        return False

    try:
        return (
            LAT_MIN <= float(lat) <= LAT_MAX
            and LON_MIN <= float(lon) <= LON_MAX
        )
    except Exception:
        return False


def date_in_expected_range(ts):
    if ts is None:
        return False

    try:
        return EXPECTED_START <= ts <= EXPECTED_END + pd.Timedelta(days=1)
    except Exception:
        return False


def year_month_day(ts):
    if ts is None:
        return None, None, None

    return ts.year, ts.month, ts.date()


def add_qc_counter(global_counter, name, arr):
    counts = qc_counts(arr)

    for flag, count in counts.items():
        global_counter[(name, flag)] += count


def write_csv(records, path):
    if not records:
        pd.DataFrame().to_csv(path, index=False)
        return

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


# ============================================================
# DISCOVER FILES
# ============================================================

print("=" * 80)
print("ARGO SCIENTIFIC AUDIT — OCEANEMBED")
print("=" * 80)
print(f"Run ID        : {RUN_ID}")
print(f"Start UTC     : {AUDIT_START_UTC.isoformat()}")
print(f"Start IST     : {AUDIT_START_IST.isoformat()}")
print(f"ARGO ROOT     : {ARGO_ROOT}")
print(f"OUTPUT DIR    : {OUTPUT_DIR}")
print("=" * 80)

if not ARGO_ROOT.exists():
    raise FileNotFoundError(
        f"Argo root directory does not exist: {ARGO_ROOT}"
    )

nc_files = sorted(ARGO_ROOT.rglob("*.nc"))

print(f"NetCDF files discovered: {len(nc_files):,}")

if len(nc_files) == 0:
    raise RuntimeError("No NetCDF files found.")


# ============================================================
# GLOBAL ACCUMULATORS
# ============================================================

profile_records = []
problem_records = []

daily = defaultdict(
    lambda: {
        "profiles": 0,
        "floats": set(),
        "nio_profiles": 0,
        "nio_floats": set(),
    }
)

monthly = defaultdict(
    lambda: {
        "profiles": 0,
        "floats": set(),
        "nio_profiles": 0,
        "nio_floats": set(),
    }
)

yearly = defaultdict(
    lambda: {
        "profiles": 0,
        "floats": set(),
        "nio_profiles": 0,
        "nio_floats": set(),
    }
)

float_stats = defaultdict(
    lambda: {
        "profiles": 0,
        "nio_profiles": 0,
        "first_date": None,
        "last_date": None,
        "years": set(),
    }
)

qc_counter = Counter()

data_mode_counter = Counter()

adjusted_availability = Counter()

duplicate_groups = defaultdict(list)

depth_coverage_counter = Counter()

# Overall counters
total_files = len(nc_files)
opened_files = 0
failed_open = 0

nio_profiles = 0
outside_nio_profiles = 0
invalid_location_profiles = 0
invalid_time_profiles = 0
out_of_period_profiles = 0

missing_core_variable_profiles = 0

profiles_with_bad_physical_values = 0
profiles_with_pressure_problem = 0
profiles_with_duplicate_pressure = 0

profiles_with_any_missing_pres = 0
profiles_with_any_missing_temp = 0
profiles_with_any_missing_psal = 0

profiles_with_adjusted_temp = 0
profiles_with_adjusted_psal = 0
profiles_with_adjusted_pres = 0

data_mode_profile_counts = Counter()

# Extrema across all profiles
global_temp_min = math.inf
global_temp_max = -math.inf
global_psal_min = math.inf
global_psal_max = -math.inf
global_pres_min = math.inf
global_pres_max = -math.inf

# Level totals
total_pres_levels = 0
valid_pres_levels = 0

total_temp_levels = 0
valid_temp_levels = 0

total_psal_levels = 0
valid_psal_levels = 0

# Sample errors
error_examples = []


# ============================================================
# MAIN SCAN
# ============================================================

for idx, nc_path in enumerate(nc_files, start=1):

    if idx == 1 or idx % 250 == 0 or idx == total_files:
        print(
            f"[{idx:>6,}/{total_files:,}] "
            f"{idx / total_files * 100:6.2f}% | "
            f"NIO={nio_profiles:,} | "
            f"Failed={failed_open:,}"
        )

    record = {
        "file": str(nc_path),
        "relative_file": str(
            nc_path.relative_to(ARGO_ROOT)
        ),
        "run_id": RUN_ID,
        "opened": False,
        "in_nio": False,
        "problem": False,
    }

    try:
        # --------------------------------------------------------
        # Open file
        # --------------------------------------------------------
        with xr.open_dataset(
            nc_path,
            decode_cf=True,
            mask_and_scale=True,
        ) as ds:

            opened_files += 1
            record["opened"] = True

            # ----------------------------------------------------
            # Core variables
            # ----------------------------------------------------
            missing_core = [
                v for v in CORE_VARS
                if v not in ds.variables
            ]

            if missing_core:
                missing_core_variable_profiles += 1
                record["missing_core_variables"] = ",".join(
                    missing_core
                )
                record["problem"] = True

            # ----------------------------------------------------
            # Identity / metadata
            # ----------------------------------------------------
            wmo = normalize_wmo(ds)
            cycle = normalize_cycle(ds)
            direction = normalize_direction(ds)

            lat_value = safe_scalar(ds, "LATITUDE")
            lon_value = safe_scalar(ds, "LONGITUDE")

            try:
                lat = float(lat_value) if lat_value is not None else None
            except Exception:
                lat = None

            try:
                lon = float(lon_value) if lon_value is not None else None
            except Exception:
                lon = None

            juld = extract_datetime(ds)

            record["wmo"] = wmo
            record["cycle_number"] = cycle
            record["direction"] = direction
            record["latitude"] = lat
            record["longitude"] = lon
            record["datetime"] = (
                juld.isoformat() if juld is not None else None
            )

            # ----------------------------------------------------
            # Location validity
            # ----------------------------------------------------
            location_valid = (
                lat is not None
                and lon is not None
                and np.isfinite(lat)
                and np.isfinite(lon)
                and -90 <= lat <= 90
                and -180 <= lon <= 180
            )

            record["valid_location"] = location_valid

            if not location_valid:
                invalid_location_profiles += 1
                record["problem"] = True

            in_nio = is_in_domain(lat, lon)

            record["in_nio"] = in_nio

            if in_nio:
                nio_profiles += 1
            else:
                outside_nio_profiles += 1

            # ----------------------------------------------------
            # Time validity
            # ----------------------------------------------------
            valid_time = juld is not None
            record["valid_time"] = valid_time

            if not valid_time:
                invalid_time_profiles += 1
                record["problem"] = True

            in_period = date_in_expected_range(juld)
            record["in_expected_period"] = in_period

            if not in_period:
                out_of_period_profiles += 1
                record["problem"] = True

            # ----------------------------------------------------
            # Profile measurements
            # ----------------------------------------------------
            pres = safe_array(ds, "PRES")
            temp = safe_array(ds, "TEMP")
            psal = safe_array(ds, "PSAL")

            pres_values = flatten_numeric(pres)
            temp_values = flatten_numeric(temp)
            psal_values = flatten_numeric(psal)

            pres_total = count_total(pres)
            temp_total = count_total(temp)
            psal_total = count_total(psal)

            pres_valid = len(pres_values)
            temp_valid = len(temp_values)
            psal_valid = len(psal_values)

            total_pres_levels += pres_total
            valid_pres_levels += pres_valid

            total_temp_levels += temp_total
            valid_temp_levels += temp_valid

            total_psal_levels += psal_total
            valid_psal_levels += psal_valid

            record["pres_total_levels"] = pres_total
            record["pres_valid_levels"] = pres_valid

            record["temp_total_levels"] = temp_total
            record["temp_valid_levels"] = temp_valid

            record["psal_total_levels"] = psal_total
            record["psal_valid_levels"] = psal_valid

            record["pres_missing_pct"] = (
                100.0 * (pres_total - pres_valid) / pres_total
                if pres_total else None
            )

            record["temp_missing_pct"] = (
                100.0 * (temp_total - temp_valid) / temp_total
                if temp_total else None
            )

            record["psal_missing_pct"] = (
                100.0 * (psal_total - psal_valid) / psal_total
                if psal_total else None
            )

            if pres_total > pres_valid:
                profiles_with_any_missing_pres += 1

            if temp_total > temp_valid:
                profiles_with_any_missing_temp += 1

            if psal_total > psal_valid:
                profiles_with_any_missing_psal += 1

            # ----------------------------------------------------
            # Physical ranges
            # ----------------------------------------------------
            temp_min, temp_max = numeric_extrema(temp)
            psal_min, psal_max = numeric_extrema(psal)
            pres_min, pres_max = numeric_extrema(pres)

            record["temp_min"] = temp_min
            record["temp_max"] = temp_max
            record["psal_min"] = psal_min
            record["psal_max"] = psal_max
            record["pres_min"] = pres_min
            record["pres_max"] = pres_max

            bad_physical = False

            if temp_min is not None:
                global_temp_min = min(global_temp_min, temp_min)
                global_temp_max = max(global_temp_max, temp_max)

                if temp_min < TEMP_MIN or temp_max > TEMP_MAX:
                    bad_physical = True

            if psal_min is not None:
                global_psal_min = min(global_psal_min, psal_min)
                global_psal_max = max(global_psal_max, psal_max)

                if psal_min < PSAL_MIN or psal_max > PSAL_MAX:
                    bad_physical = True

            if pres_min is not None:
                global_pres_min = min(global_pres_min, pres_min)
                global_pres_max = max(global_pres_max, pres_max)

                if pres_min < PRES_MIN or pres_max > PRES_MAX:
                    bad_physical = True

            if bad_physical:
                profiles_with_bad_physical_values += 1
                record["bad_physical_values"] = True
                record["problem"] = True
            else:
                record["bad_physical_values"] = False

            # ----------------------------------------------------
            # Pressure structure
            # ----------------------------------------------------
            pressure_stats = pressure_statistics(pres)

            record["pressure_monotonic_non_decreasing"] = (
                pressure_stats["monotonic_non_decreasing"]
            )

            record["pressure_strictly_increasing"] = (
                pressure_stats["strictly_increasing"]
            )

            record["duplicate_adjacent_pressure_levels"] = (
                pressure_stats["duplicate_adjacent_count"]
            )

            pressure_problem = (
                pressure_stats["monotonic_non_decreasing"] is False
            )

            duplicate_pressure = (
                pressure_stats["duplicate_adjacent_count"] > 0
            )

            if pressure_problem:
                profiles_with_pressure_problem += 1
                record["problem"] = True

            if duplicate_pressure:
                profiles_with_duplicate_pressure += 1

            # ----------------------------------------------------
            # DATA_MODE
            # ----------------------------------------------------
            data_mode = safe_scalar(ds, "DATA_MODE")

            if data_mode is not None:
                data_mode = str(data_mode).strip()

            record["data_mode"] = data_mode

            if data_mode:
                data_mode_counter[data_mode] += 1
                data_mode_profile_counts[data_mode] += 1

            # ----------------------------------------------------
            # QC variables
            # ----------------------------------------------------
            for qc_var in QC_VARS:
                if qc_var in ds.variables:
                    arr = safe_array(ds, qc_var)
                    add_qc_counter(qc_counter, qc_var, arr)

            # ----------------------------------------------------
            # Adjusted variables
            # ----------------------------------------------------
            adjusted_presence = {}

            for adj_var in ADJUSTED_VARS:
                if adj_var in ds.variables:
                    arr = safe_array(ds, adj_var)
                    values = flatten_numeric(arr)

                    has_adjusted = values.size > 0

                    adjusted_presence[adj_var] = has_adjusted

                    if has_adjusted:
                        adjusted_availability[adj_var] += 1

                else:
                    adjusted_presence[adj_var] = False

            record["has_pres_adjusted"] = adjusted_presence[
                "PRES_ADJUSTED"
            ]
            record["has_temp_adjusted"] = adjusted_presence[
                "TEMP_ADJUSTED"
            ]
            record["has_psal_adjusted"] = adjusted_presence[
                "PSAL_ADJUSTED"
            ]

            if record["has_pres_adjusted"]:
                profiles_with_adjusted_pres += 1

            if record["has_temp_adjusted"]:
                profiles_with_adjusted_temp += 1

            if record["has_psal_adjusted"]:
                profiles_with_adjusted_psal += 1

            # ----------------------------------------------------
            # Adjusted vs raw differences
            # ----------------------------------------------------
            for raw_name, adj_name, diff_name in [
                ("TEMP", "TEMP_ADJUSTED", "temp_adjusted_raw_abs_diff"),
                ("PSAL", "PSAL_ADJUSTED", "psal_adjusted_raw_abs_diff"),
                ("PRES", "PRES_ADJUSTED", "pres_adjusted_raw_abs_diff"),
            ]:
                if raw_name in ds.variables and adj_name in ds.variables:

                    raw_values = flatten_numeric(
                        safe_array(ds, raw_name)
                    )

                    adj_values = flatten_numeric(
                        safe_array(ds, adj_name)
                    )

                    n = min(
                        raw_values.size,
                        adj_values.size
                    )

                    if n > 0:
                        raw_cmp = raw_values[:n]
                        adj_cmp = adj_values[:n]

                        diff = np.abs(adj_cmp - raw_cmp)

                        finite_diff = diff[
                            np.isfinite(diff)
                        ]

                        if finite_diff.size:
                            record[diff_name + "_mean"] = float(
                                np.mean(finite_diff)
                            )
                            record[diff_name + "_max"] = float(
                                np.max(finite_diff)
                            )
                        else:
                            record[diff_name + "_mean"] = None
                            record[diff_name + "_max"] = None
                    else:
                        record[diff_name + "_mean"] = None
                        record[diff_name + "_max"] = None

            # ----------------------------------------------------
            # Depth coverage
            # ----------------------------------------------------
            deepest_pressure = (
                float(np.max(pres_values))
                if pres_values.size
                else None
            )

            record["deepest_pressure_dbar"] = deepest_pressure

            if deepest_pressure is not None:

                for threshold in DEPTH_THRESHOLDS:

                    if deepest_pressure >= threshold:
                        depth_coverage_counter[
                            threshold
                        ] += 1

            # ----------------------------------------------------
            # Time coverage accumulators
            # ----------------------------------------------------
            if juld is not None:

                year, month, day = year_month_day(juld)

                day_key = str(day)
                month_key = f"{year:04d}-{month:02d}"
                year_key = str(year)

                daily[day_key]["profiles"] += 1

                if wmo:
                    daily[day_key]["floats"].add(str(wmo))

                monthly[month_key]["profiles"] += 1

                if wmo:
                    monthly[month_key]["floats"].add(str(wmo))

                yearly[year_key]["profiles"] += 1

                if wmo:
                    yearly[year_key]["floats"].add(str(wmo))

                if in_nio:

                    daily[day_key]["nio_profiles"] += 1

                    if wmo:
                        daily[day_key]["nio_floats"].add(
                            str(wmo)
                        )

                    monthly[month_key]["nio_profiles"] += 1

                    if wmo:
                        monthly[month_key]["nio_floats"].add(
                            str(wmo)
                        )

                    yearly[year_key]["nio_profiles"] += 1

                    if wmo:
                        yearly[year_key]["nio_floats"].add(
                            str(wmo)
                        )

                # Float statistics
                if wmo:
                    wmo_key = str(wmo)

                    float_stats[wmo_key]["profiles"] += 1

                    if in_nio:
                        float_stats[wmo_key]["nio_profiles"] += 1

                    float_stats[wmo_key]["years"].add(year)

                    current_first = float_stats[
                        wmo_key
                    ]["first_date"]

                    current_last = float_stats[
                        wmo_key
                    ]["last_date"]

                    if (
                        current_first is None
                        or juld < current_first
                    ):
                        float_stats[
                            wmo_key
                        ]["first_date"] = juld

                    if (
                        current_last is None
                        or juld > current_last
                    ):
                        float_stats[
                            wmo_key
                        ]["last_date"] = juld

            # ----------------------------------------------------
            # Duplicate identity
            # ----------------------------------------------------
            identity = profile_identity(
                wmo,
                cycle,
                juld,
                lat,
                lon,
                direction,
            )

            duplicate_groups[identity].append(
                str(nc_path)
            )

            # ----------------------------------------------------
            # Problem summary
            # ----------------------------------------------------
            if record.get("problem"):
                problem_records.append(record.copy())

            profile_records.append(record)

    except Exception as exc:

        failed_open += 1

        record["opened"] = False
        record["problem"] = True
        record["error"] = str(exc)

        problem_records.append(record.copy())

        if len(error_examples) < 25:
            error_examples.append(
                {
                    "file": str(nc_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

duplicate_records = []

duplicate_group_count = 0
duplicate_profile_count = 0

for identity, files in duplicate_groups.items():

    if len(files) <= 1:
        continue

    duplicate_group_count += 1
    duplicate_profile_count += len(files)

    wmo, cycle, juld, lat, lon, direction = identity

    duplicate_records.append(
        {
            "wmo": wmo,
            "cycle_number": cycle,
            "datetime": juld,
            "latitude": lat,
            "longitude": lon,
            "direction": direction,
            "file_count": len(files),
            "files": " | ".join(files),
        }
    )


# ============================================================
# DATAFRAME OUTPUTS
# ============================================================

profile_df = pd.DataFrame(profile_records)

if not profile_df.empty:
    profile_df.to_csv(
        PROFILE_CSV,
        index=False,
    )
else:
    pd.DataFrame().to_csv(
        PROFILE_CSV,
        index=False,
    )


# ------------------------------------------------------------
# DAILY
# ------------------------------------------------------------

daily_records = []

for day_key in sorted(daily.keys()):

    item = daily[day_key]

    daily_records.append(
        {
            "date": day_key,
            "profiles_all": item["profiles"],
            "floats_all": len(item["floats"]),
            "profiles_nio": item["nio_profiles"],
            "floats_nio": len(item["nio_floats"]),
        }
    )

write_csv(
    daily_records,
    DAILY_CSV,
)


# ------------------------------------------------------------
# MONTHLY
# ------------------------------------------------------------

monthly_records = []

for month_key in sorted(monthly.keys()):

    item = monthly[month_key]

    monthly_records.append(
        {
            "month": month_key,
            "profiles_all": item["profiles"],
            "floats_all": len(item["floats"]),
            "profiles_nio": item["nio_profiles"],
            "floats_nio": len(item["nio_floats"]),
        }
    )

write_csv(
    monthly_records,
    MONTHLY_CSV,
)


# ------------------------------------------------------------
# YEARLY
# ------------------------------------------------------------

yearly_records = []

for year_key in sorted(yearly.keys()):

    item = yearly[year_key]

    yearly_records.append(
        {
            "year": int(year_key),
            "profiles_all": item["profiles"],
            "floats_all": len(item["floats"]),
            "profiles_nio": item["nio_profiles"],
            "floats_nio": len(item["nio_floats"]),
        }
    )

write_csv(
    yearly_records,
    YEARLY_CSV,
)


# ------------------------------------------------------------
# FLOAT SUMMARY
# ------------------------------------------------------------

float_records = []

for wmo, item in sorted(float_stats.items()):

    first_date = item["first_date"]
    last_date = item["last_date"]

    duration_days = None

    if first_date is not None and last_date is not None:
        duration_days = (
            last_date - first_date
        ).days

    float_records.append(
        {
            "wmo": wmo,
            "profiles_all": item["profiles"],
            "profiles_nio": item["nio_profiles"],
            "first_date": (
                first_date.isoformat()
                if first_date is not None
                else None
            ),
            "last_date": (
                last_date.isoformat()
                if last_date is not None
                else None
            ),
            "active_duration_days": duration_days,
            "years_active": len(item["years"]),
        }
    )

write_csv(
    float_records,
    FLOAT_CSV,
)


# ------------------------------------------------------------
# QC SUMMARY
# ------------------------------------------------------------

qc_records = []

for (variable, flag), count in sorted(
    qc_counter.items()
):

    qc_records.append(
        {
            "variable": variable,
            "qc_flag": flag,
            "count": count,
        }
    )

write_csv(
    qc_records,
    QC_CSV,
)


# ------------------------------------------------------------
# DEPTH SUMMARY
# ------------------------------------------------------------

depth_records = []

for threshold in DEPTH_THRESHOLDS:

    count = depth_coverage_counter[threshold]

    depth_records.append(
        {
            "minimum_pressure_dbar": threshold,
            "profiles_reaching_threshold": count,
            "percentage_of_all_profiles": (
                100.0 * count / total_files
                if total_files
                else 0
            ),
            "percentage_of_nio_profiles": (
                100.0 * count / nio_profiles
                if nio_profiles
                else 0
            ),
        }
    )

write_csv(
    depth_records,
    DEPTH_CSV,
)


# ------------------------------------------------------------
# DUPLICATES
# ------------------------------------------------------------

write_csv(
    duplicate_records,
    DUPLICATE_CSV,
)


# ------------------------------------------------------------
# PROBLEMS
# ------------------------------------------------------------

write_csv(
    problem_records,
    PROBLEM_CSV,
)


# ============================================================
# FINAL TIME
# ============================================================

AUDIT_END_UTC = datetime.now(timezone.utc)
AUDIT_END_IST = AUDIT_END_UTC.astimezone(
    timezone(pd.Timedelta(hours=5, minutes=30))
)

duration_seconds = (
    AUDIT_END_UTC - AUDIT_START_UTC
).total_seconds()


# ============================================================
# EXPECTED CALENDAR COVERAGE
# ============================================================

expected_dates = pd.date_range(
    EXPECTED_START,
    EXPECTED_END,
    freq="D",
)

observed_dates = set(
    daily.keys()
)

expected_date_strings = {
    d.date().isoformat()
    for d in expected_dates
}

missing_calendar_dates = sorted(
    expected_date_strings - observed_dates
)

extra_dates = sorted(
    observed_dates - expected_date_strings
)


# ============================================================
# PERCENTAGES
# ============================================================

def pct(value, denominator):
    if not denominator:
        return 0.0

    return 100.0 * value / denominator


nio_pct = pct(
    nio_profiles,
    total_files,
)

outside_nio_pct = pct(
    outside_nio_profiles,
    total_files,
)

failed_pct = pct(
    failed_open,
    total_files,
)

invalid_location_pct = pct(
    invalid_location_profiles,
    total_files,
)

invalid_time_pct = pct(
    invalid_time_profiles,
    total_files,
)

out_of_period_pct = pct(
    out_of_period_profiles,
    total_files,
)

bad_physical_pct = pct(
    profiles_with_bad_physical_values,
    total_files,
)

pressure_problem_pct = pct(
    profiles_with_pressure_problem,
    total_files,
)

duplicate_pressure_pct = pct(
    profiles_with_duplicate_pressure,
    total_files,
)


# ============================================================
# DECISION FLAGS
# ============================================================

structural_ok = (
    failed_open == 0
    and missing_core_variable_profiles == 0
)

geographic_ok = (
    nio_profiles > 0
)

temporal_calendar_ok = (
    len(missing_calendar_dates) == 0
    and len(extra_dates) == 0
)

physical_basic_ok = (
    profiles_with_bad_physical_values == 0
)

pressure_structure_ok = (
    profiles_with_pressure_problem == 0
)

duplicate_review_needed = (
    duplicate_group_count > 0
)

adjusted_data_present = (
    profiles_with_adjusted_temp > 0
    or profiles_with_adjusted_psal > 0
    or profiles_with_adjusted_pres > 0
)

scientific_audit_complete = (
    opened_files == total_files
)


# ============================================================
# JSON REPORT
# ============================================================

json_report = {
    "audit_identity": {
        "audit_name": "OceanEmbed Argo Scientific Audit",
        "run_id": RUN_ID,
        "script": "argo_scientific_audit.py",
        "audit_started_utc": AUDIT_START_UTC.isoformat(),
        "audit_started_ist": AUDIT_START_IST.isoformat(),
        "audit_finished_utc": AUDIT_END_UTC.isoformat(),
        "audit_finished_ist": AUDIT_END_IST.isoformat(),
        "duration_seconds": duration_seconds,
        "raw_data_modified": False,
    },

    "configuration": {
        "argo_root": str(ARGO_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "expected_start": EXPECTED_START.date().isoformat(),
        "expected_end": EXPECTED_END.date().isoformat(),
        "nio_domain": {
            "latitude_min": LAT_MIN,
            "latitude_max": LAT_MAX,
            "longitude_min": LON_MIN,
            "longitude_max": LON_MAX,
        },
        "physical_screening_ranges": {
            "temperature_c": [
                TEMP_MIN,
                TEMP_MAX,
            ],
            "salinity_psu": [
                PSAL_MIN,
                PSAL_MAX,
            ],
            "pressure_dbar": [
                PRES_MIN,
                PRES_MAX,
            ],
        },
        "depth_thresholds_dbar": DEPTH_THRESHOLDS,
    },

    "file_inventory": {
        "netcdf_files": total_files,
        "opened_files": opened_files,
        "failed_open": failed_open,
        "failed_open_pct": failed_pct,
    },

    "geographic_coverage": {
        "profiles_in_nio": nio_profiles,
        "profiles_in_nio_pct": nio_pct,
        "profiles_outside_nio": outside_nio_profiles,
        "profiles_outside_nio_pct": outside_nio_pct,
        "invalid_location_profiles": invalid_location_profiles,
        "invalid_location_pct": invalid_location_pct,
    },

    "temporal_coverage": {
        "invalid_time_profiles": invalid_time_profiles,
        "invalid_time_pct": invalid_time_pct,
        "out_of_expected_period_profiles": out_of_period_profiles,
        "out_of_expected_period_pct": out_of_period_pct,
        "expected_calendar_days": len(expected_dates),
        "observed_calendar_days": len(observed_dates),
        "missing_calendar_days": missing_calendar_dates,
        "extra_dates": extra_dates,
    },

    "profile_quality": {
        "missing_core_variable_profiles": missing_core_variable_profiles,
        "bad_physical_profiles": profiles_with_bad_physical_values,
        "bad_physical_pct": bad_physical_pct,
        "pressure_ordering_problem_profiles": profiles_with_pressure_problem,
        "pressure_ordering_problem_pct": pressure_problem_pct,
        "profiles_with_duplicate_adjacent_pressure": profiles_with_duplicate_pressure,
        "duplicate_pressure_pct": duplicate_pressure_pct,
    },

    "global_physical_extrema": {
        "temperature_min_c": (
            None
            if global_temp_min == math.inf
            else global_temp_min
        ),
        "temperature_max_c": (
            None
            if global_temp_max == -math.inf
            else global_temp_max
        ),
        "salinity_min_psu": (
            None
            if global_psal_min == math.inf
            else global_psal_min
        ),
        "salinity_max_psu": (
            None
            if global_psal_max == -math.inf
            else global_psal_max
        ),
        "pressure_min_dbar": (
            None
            if global_pres_min == math.inf
            else global_pres_min
        ),
        "pressure_max_dbar": (
            None
            if global_pres_max == -math.inf
            else global_pres_max
        ),
    },

    "missingness": {
        "profiles_with_missing_pres": profiles_with_any_missing_pres,
        "profiles_with_missing_temp": profiles_with_any_missing_temp,
        "profiles_with_missing_psal": profiles_with_any_missing_psal,
        "total_pres_levels": total_pres_levels,
        "valid_pres_levels": valid_pres_levels,
        "total_temp_levels": total_temp_levels,
        "valid_temp_levels": valid_temp_levels,
        "total_psal_levels": total_psal_levels,
        "valid_psal_levels": valid_psal_levels,
    },

    "data_mode": dict(data_mode_counter),

    "adjusted_data": {
        "profiles_with_pres_adjusted": profiles_with_adjusted_pres,
        "profiles_with_temp_adjusted": profiles_with_adjusted_temp,
        "profiles_with_psal_adjusted": profiles_with_adjusted_psal,
        "availability_counts": dict(adjusted_availability),
    },

    "duplicates": {
        "duplicate_identity_groups": duplicate_group_count,
        "profiles_in_duplicate_groups": duplicate_profile_count,
    },

    "float_inventory": {
        "unique_wmo_floats": len(float_stats),
    },

    "depth_coverage": {
        str(threshold): depth_coverage_counter[threshold]
        for threshold in DEPTH_THRESHOLDS
    },

    "qc_counts": {
        f"{variable}:{flag}": count
        for (variable, flag), count
        in sorted(qc_counter.items())
    },

    "decision_flags": {
        "structural_ok": structural_ok,
        "geographic_coverage_available": geographic_ok,
        "calendar_coverage_ok": temporal_calendar_ok,
        "basic_physical_screen_ok": physical_basic_ok,
        "pressure_structure_ok": pressure_structure_ok,
        "duplicate_review_needed": duplicate_review_needed,
        "adjusted_data_present": adjusted_data_present,
        "all_files_scanned_successfully": scientific_audit_complete,
    },

    "important_interpretation": {
        "note_1": (
            "This audit does not remove or modify any Argo profile."
        ),
        "note_2": (
            "Physical ranges are broad sanity checks, not final "
            "scientific QC rules."
        ),
        "note_3": (
            "QC flags are reported but are not used here to delete "
            "measurements."
        ),
        "note_4": (
            "Adjusted variables are reported separately from raw "
            "variables; final selection policy requires review."
        ),
        "note_5": (
            "Duplicate identities are flagged for review and are "
            "not automatically deleted."
        ),
    },

    "outputs": {
        "report_txt": str(REPORT_TXT),
        "report_json": str(REPORT_JSON),
        "profile_inventory": str(PROFILE_CSV),
        "daily_coverage": str(DAILY_CSV),
        "monthly_coverage": str(MONTHLY_CSV),
        "yearly_coverage": str(YEARLY_CSV),
        "float_summary": str(FLOAT_CSV),
        "qc_summary": str(QC_CSV),
        "depth_coverage": str(DEPTH_CSV),
        "duplicate_profiles": str(DUPLICATE_CSV),
        "problem_profiles": str(PROBLEM_CSV),
    },
}


# ============================================================
# TEXT REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 100)
report_lines.append("OCEANEMBED — ARGO SCIENTIFIC AUDIT REPORT")
report_lines.append("=" * 100)
report_lines.append("")

report_lines.append("AUDIT IDENTITY")
report_lines.append("-" * 100)
report_lines.append(f"Audit Run ID          : {RUN_ID}")
report_lines.append(
    f"Audit Started UTC     : {AUDIT_START_UTC.isoformat()}"
)
report_lines.append(
    f"Audit Started IST     : {AUDIT_START_IST.isoformat()}"
)
report_lines.append(
    f"Audit Finished UTC    : {AUDIT_END_UTC.isoformat()}"
)
report_lines.append(
    f"Audit Finished IST    : {AUDIT_END_IST.isoformat()}"
)
report_lines.append(
    f"Audit Duration        : {duration_seconds:.2f} seconds"
)
report_lines.append(
    "Raw Argo files modified: NO"
)
report_lines.append("")

report_lines.append("AUDIT CONFIGURATION")
report_lines.append("-" * 100)
report_lines.append(f"Argo Root             : {ARGO_ROOT}")
report_lines.append(f"Expected Start        : {EXPECTED_START.date()}")
report_lines.append(f"Expected End          : {EXPECTED_END.date()}")
report_lines.append(
    f"NIO Latitude          : {LAT_MIN} to {LAT_MAX} N"
)
report_lines.append(
    f"NIO Longitude         : {LON_MIN} to {LON_MAX} E"
)
report_lines.append("")

report_lines.append("1. FILE INVENTORY")
report_lines.append("-" * 100)
report_lines.append(
    f"NetCDF files discovered : {total_files:,}"
)
report_lines.append(
    f"Successfully opened     : {opened_files:,}"
)
report_lines.append(
    f"Failed to open          : {failed_open:,}"
)
report_lines.append(
    f"Open failure percentage : {failed_pct:.4f}%"
)
report_lines.append("")

report_lines.append("2. ACTUAL NIO GEOGRAPHIC COVERAGE")
report_lines.append("-" * 100)
report_lines.append(
    f"Profiles inside NIO     : {nio_profiles:,}"
)
report_lines.append(
    f"Percentage inside NIO   : {nio_pct:.4f}%"
)
report_lines.append(
    f"Profiles outside NIO    : {outside_nio_profiles:,}"
)
report_lines.append(
    f"Percentage outside NIO  : {outside_nio_pct:.4f}%"
)
report_lines.append(
    f"Invalid location        : {invalid_location_profiles:,}"
)
report_lines.append("")

report_lines.append("3. TEMPORAL COVERAGE")
report_lines.append("-" * 100)
report_lines.append(
    f"Expected calendar days  : {len(expected_dates):,}"
)
report_lines.append(
    f"Observed calendar days  : {len(observed_dates):,}"
)
report_lines.append(
    f"Missing calendar days   : {len(missing_calendar_dates):,}"
)

if missing_calendar_dates:
    report_lines.append(
        "Missing dates (first 50):"
    )

    for value in missing_calendar_dates[:50]:
        report_lines.append(
            f"  {value}"
        )

report_lines.append(
    f"Invalid profile times   : {invalid_time_profiles:,}"
)
report_lines.append(
    f"Out-of-period profiles  : {out_of_period_profiles:,}"
)
report_lines.append("")

report_lines.append("4. UNIQUE FLOAT INVENTORY")
report_lines.append("-" * 100)
report_lines.append(
    f"Unique WMO floats       : {len(float_stats):,}"
)
report_lines.append("")

report_lines.append("5. DATA MODE")
report_lines.append("-" * 100)

if data_mode_counter:
    for mode, count in sorted(
        data_mode_counter.items()
    ):
        report_lines.append(
            f"{mode:>4} : {count:,} profiles "
            f"({pct(count, total_files):.3f}%)"
        )
else:
    report_lines.append(
        "No DATA_MODE values found."
    )

report_lines.append("")

report_lines.append("6. ADJUSTED DATA AVAILABILITY")
report_lines.append("-" * 100)
report_lines.append(
    f"PRES_ADJUSTED profiles : "
    f"{profiles_with_adjusted_pres:,} "
    f"({pct(profiles_with_adjusted_pres, total_files):.3f}%)"
)
report_lines.append(
    f"TEMP_ADJUSTED profiles : "
    f"{profiles_with_adjusted_temp:,} "
    f"({pct(profiles_with_adjusted_temp, total_files):.3f}%)"
)
report_lines.append(
    f"PSAL_ADJUSTED profiles : "
    f"{profiles_with_adjusted_psal:,} "
    f"({pct(profiles_with_adjusted_psal, total_files):.3f}%)"
)
report_lines.append("")

report_lines.append("7. GLOBAL PHYSICAL EXTREMA")
report_lines.append("-" * 100)

def fmt(value):
    return "N/A" if value is None else f"{value:.6f}"

report_lines.append(
    f"TEMP minimum           : {fmt(json_report['global_physical_extrema']['temperature_min_c'])} °C"
)
report_lines.append(
    f"TEMP maximum           : {fmt(json_report['global_physical_extrema']['temperature_max_c'])} °C"
)
report_lines.append(
    f"PSAL minimum           : {fmt(json_report['global_physical_extrema']['salinity_min_psu'])} PSU"
)
report_lines.append(
    f"PSAL maximum           : {fmt(json_report['global_physical_extrema']['salinity_max_psu'])} PSU"
)
report_lines.append(
    f"PRES minimum           : {fmt(json_report['global_physical_extrema']['pressure_min_dbar'])} dbar"
)
report_lines.append(
    f"PRES maximum           : {fmt(json_report['global_physical_extrema']['pressure_max_dbar'])} dbar"
)
report_lines.append("")

report_lines.append("8. BASIC PHYSICAL SANITY CHECK")
report_lines.append("-" * 100)
report_lines.append(
    f"Temperature range      : {TEMP_MIN} to {TEMP_MAX} °C"
)
report_lines.append(
    f"Salinity range         : {PSAL_MIN} to {PSAL_MAX} PSU"
)
report_lines.append(
    f"Pressure range         : {PRES_MIN} to {PRES_MAX} dbar"
)
report_lines.append(
    f"Profiles with failures : {profiles_with_bad_physical_values:,}"
)
report_lines.append(
    f"Failure percentage     : {bad_physical_pct:.4f}%"
)
report_lines.append("")

report_lines.append("9. PRESSURE STRUCTURE")
report_lines.append("-" * 100)
report_lines.append(
    f"Non-monotonic profiles : {profiles_with_pressure_problem:,}"
)
report_lines.append(
    f"Non-monotonic percent   : {pressure_problem_pct:.4f}%"
)
report_lines.append(
    f"Profiles with adjacent duplicate pressure levels: "
    f"{profiles_with_duplicate_pressure:,}"
)
report_lines.append("")

report_lines.append("10. MISSINGNESS")
report_lines.append("-" * 100)
report_lines.append(
    f"Profiles with missing PRES: {profiles_with_any_missing_pres:,}"
)
report_lines.append(
    f"Profiles with missing TEMP: {profiles_with_any_missing_temp:,}"
)
report_lines.append(
    f"Profiles with missing PSAL: {profiles_with_any_missing_psal:,}"
)
report_lines.append("")

report_lines.append("11. DEPTH / PRESSURE COVERAGE")
report_lines.append("-" * 100)

for threshold in DEPTH_THRESHOLDS:

    count = depth_coverage_counter[threshold]

    report_lines.append(
        f">= {threshold:>4} dbar : "
        f"{count:>7,} profiles "
        f"({pct(count, total_files):>7.3f}% of all; "
        f"{pct(count, nio_profiles):>7.3f}% of NIO)"
    )

report_lines.append("")

report_lines.append("12. QC FLAG DISTRIBUTION")
report_lines.append("-" * 100)

if qc_counter:

    current_var = None

    for (variable, flag), count in sorted(
        qc_counter.items()
    ):

        if variable != current_var:
            report_lines.append("")
            report_lines.append(
                f"{variable}:"
            )
            current_var = variable

        report_lines.append(
            f"  flag {flag}: {count:,}"
        )

else:
    report_lines.append(
        "No QC variables found."
    )

report_lines.append("")

report_lines.append("13. DUPLICATE PROFILE IDENTITIES")
report_lines.append("-" * 100)
report_lines.append(
    f"Duplicate identity groups : {duplicate_group_count:,}"
)
report_lines.append(
    f"Profiles in duplicate groups: {duplicate_profile_count:,}"
)
report_lines.append(
    "Duplicates are FLAGGED only. No files were removed."
)
report_lines.append("")

report_lines.append("14. SCIENTIFIC AUDIT DECISION FLAGS")
report_lines.append("-" * 100)
report_lines.append(
    f"All files opened successfully      : "
    f"{'PASS' if scientific_audit_complete else 'FAIL'}"
)
report_lines.append(
    f"Core variables available           : "
    f"{'PASS' if structural_ok else 'FAIL'}"
)
report_lines.append(
    f"NIO profiles found                 : "
    f"{'PASS' if geographic_ok else 'FAIL'}"
)
report_lines.append(
    f"Calendar coverage                  : "
    f"{'PASS' if temporal_calendar_ok else 'REVIEW'}"
)
report_lines.append(
    f"Broad physical sanity              : "
    f"{'PASS' if physical_basic_ok else 'REVIEW'}"
)
report_lines.append(
    f"Pressure ordering                  : "
    f"{'PASS' if pressure_structure_ok else 'REVIEW'}"
)
report_lines.append(
    f"Duplicate identities               : "
    f"{'REVIEW REQUIRED' if duplicate_review_needed else 'NONE DETECTED'}"
)
report_lines.append(
    f"Adjusted data present              : "
    f"{'YES' if adjusted_data_present else 'NO'}"
)
report_lines.append("")

report_lines.append("15. IMPORTANT SCIENTIFIC INTERPRETATION")
report_lines.append("-" * 100)
report_lines.append(
    "This report is an evidence-gathering audit."
)
report_lines.append(
    "It does not automatically declare all profiles scientifically usable."
)
report_lines.append(
    "QC flags, DATA_MODE, adjusted measurements, duplicates and"
)
report_lines.append(
    "profile completeness must be considered when defining the"
)
report_lines.append(
    "final OceanEmbed Argo validation dataset."
)
report_lines.append("")
report_lines.append(
    "No interpolation, regridding, gap filling, smoothing,"
)
report_lines.append(
    "deletion or modification was performed."
)
report_lines.append("")

report_lines.append("16. OUTPUT FILES")
report_lines.append("-" * 100)

for path in [
    REPORT_TXT,
    REPORT_JSON,
    PROFILE_CSV,
    DAILY_CSV,
    MONTHLY_CSV,
    YEARLY_CSV,
    FLOAT_CSV,
    QC_CSV,
    DEPTH_CSV,
    DUPLICATE_CSV,
    PROBLEM_CSV,
]:
    report_lines.append(
        f"{path.name}"
    )

report_lines.append("")
report_lines.append("=" * 100)
report_lines.append("END OF ARGO SCIENTIFIC AUDIT")
report_lines.append("=" * 100)


REPORT_TXT.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# WRITE JSON
# ============================================================

REPORT_JSON.write_text(
    json.dumps(
        json_report,
        indent=2,
        default=str,
    ),
    encoding="utf-8",
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("")
print("=" * 80)
print("ARGO SCIENTIFIC AUDIT COMPLETE")
print("=" * 80)
print(f"Run ID                  : {RUN_ID}")
print(f"Files scanned            : {total_files:,}")
print(f"Files opened             : {opened_files:,}")
print(f"Files failed             : {failed_open:,}")
print(f"NIO profiles             : {nio_profiles:,} ({nio_pct:.3f}%)")
print(f"Outside NIO              : {outside_nio_profiles:,}")
print(f"Unique WMO floats        : {len(float_stats):,}")
print(f"Missing calendar days    : {len(missing_calendar_dates):,}")
print(
    f"Bad physical profiles    : "
    f"{profiles_with_bad_physical_values:,}"
)
print(
    f"Pressure problems        : "
    f"{profiles_with_pressure_problem:,}"
)
print(
    f"Duplicate identity groups: "
    f"{duplicate_group_count:,}"
)
print(
    f"TEMP adjusted profiles   : "
    f"{profiles_with_adjusted_temp:,}"
)
print(
    f"PSAL adjusted profiles   : "
    f"{profiles_with_adjusted_psal:,}"
)
print("")
print("Reports written to:")
print(f"  {OUTPUT_DIR}")
print("")
print(f"TXT report : {REPORT_TXT}")
print(f"JSON report: {REPORT_JSON}")
print("=" * 80)