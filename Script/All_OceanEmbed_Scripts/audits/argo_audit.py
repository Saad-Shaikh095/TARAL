from pathlib import Path
from collections import Counter, defaultdict
import re
import json
import math
import traceback

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

ARGO_ROOT = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\argo"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\argo_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "argo_initial_audit_report.txt"
CSV_FILE = OUTPUT_DIR / "argo_sample_file_audit.csv"
JSON_FILE = OUTPUT_DIR / "argo_initial_audit.json"


# OceanEmbed target region
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

EXPECTED_START = pd.Timestamp("2012-01-01")
EXPECTED_END = pd.Timestamp("2024-12-31")


# Number of files to inspect scientifically
# We deliberately keep this manageable initially.
MAX_SCIENTIFIC_SAMPLES = 100


# ============================================================
# HELPERS
# ============================================================

def write(report_lines, text=""):
    report_lines.append(str(text))


def safe_str(value):
    try:
        return str(value)
    except Exception:
        return repr(value)


def parse_year_month_from_parent(path):
    """
    Expected structure:

        argo/
            2012_01/
            2012_02/
            ...

    Return (year, month) if detectable.
    """
    for parent in path.parents:
        m = re.fullmatch(r"(20\d{2})_(\d{2})", parent.name)

        if m:
            return int(m.group(1)), int(m.group(2))

    return None, None


def detect_dac(path):
    """
    Find DAC in path.

    Expected:

        2012_01/aoml/2900106/profiles/file.nc
    """
    parts = path.parts

    for i, part in enumerate(parts):
        if re.fullmatch(r"20\d{2}_\d{2}", part):
            if i + 1 < len(parts):
                return parts[i + 1]

    return "UNKNOWN"


def detect_wmo(path):
    """
    Find WMO directory.
    """
    for part in path.parts:
        if re.fullmatch(r"\d{7}", part):
            return part

    return "UNKNOWN"


def classify_file(path):
    """
    Basic classification from path/name.
    """
    lower = str(path).lower()

    if "\\profiles\\" in lower:
        return "profile"

    if "meta" in path.name.lower():
        return "metadata"

    if "traj" in path.name.lower():
        return "trajectory"

    if "synthetic" in path.name.lower():
        return "synthetic"

    return "other"


def finite_stats(values):
    arr = np.asarray(values)

    if arr.size == 0:
        return {}

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "count": int(arr.size),
            "finite": 0,
            "nan_or_nonfinite": int(arr.size),
        }

    return {
        "count": int(arr.size),
        "finite": int(finite.size),
        "nan_or_nonfinite": int(arr.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def check_required_variables(ds):
    """
    Argo core variables we care about.
    """
    candidates = {
        "latitude": ["LATITUDE", "latitude"],
        "longitude": ["LONGITUDE", "longitude"],
        "time": ["JULD", "juld", "TIME", "time"],
        "pressure": ["PRES", "pres", "pressure"],
        "temperature": ["TEMP", "temp", "temperature"],
        "salinity": ["PSAL", "psal", "salinity"],
    }

    found = {}

    for logical_name, names in candidates.items():
        found[logical_name] = None

        for name in names:
            if name in ds.variables:
                found[logical_name] = name
                break

    return found


def extract_scalar_or_first(value):
    try:
        arr = np.asarray(value)

        if arr.size == 0:
            return None

        return arr.flat[0].item()

    except Exception:
        return None


# ============================================================
# REPORT SETUP
# ============================================================

report = []

write(report, "=" * 100)
write(report, "OCEANEMBED — ARGO INITIAL SCIENTIFIC AUDIT")
write(report, "=" * 100)
write(report)
write(report, f"ARGO ROOT: {ARGO_ROOT}")
write(report, f"OUTPUT DIR: {OUTPUT_DIR}")
write(report, f"Expected period: {EXPECTED_START.date()} -> {EXPECTED_END.date()}")
write(report, f"Target region: {LAT_MIN}–{LAT_MAX} N, {LON_MIN}–{LON_MAX} E")
write(report)


# ============================================================
# CHECK ROOT
# ============================================================

if not ARGO_ROOT.exists():
    raise FileNotFoundError(
        f"Argo directory does not exist:\n{ARGO_ROOT}"
    )


# ============================================================
# FULL FILE INVENTORY
# ============================================================

write(report, "=" * 100)
write(report, "SECTION 1 — COMPLETE FILE INVENTORY")
write(report, "=" * 100)

print("Scanning all Argo NetCDF files...")

all_nc_files = sorted(
    ARGO_ROOT.rglob("*.nc")
)

write(report, f"Total NetCDF files: {len(all_nc_files):,}")


if not all_nc_files:
    raise RuntimeError("No NetCDF files found.")


# ============================================================
# FILE CLASSIFICATION
# ============================================================

classification_counts = Counter()
year_counts = Counter()
month_counts = Counter()
dac_counts = Counter()
wmo_counts = Counter()

files_by_year = defaultdict(list)
files_by_dac = defaultdict(list)

for path in all_nc_files:

    classification = classify_file(path)
    classification_counts[classification] += 1

    year, month = parse_year_month_from_parent(path)

    if year is not None:
        year_counts[year] += 1
        month_counts[(year, month)] += 1
        files_by_year[year].append(path)

    dac = detect_dac(path)

    dac_counts[dac] += 1
    files_by_dac[dac].append(path)

    wmo = detect_wmo(path)

    if wmo != "UNKNOWN":
        wmo_counts[wmo] += 1


write(report)
write(report, "File classification:")
for key, value in classification_counts.most_common():
    write(report, f"  {key:15s}: {value:,}")


# ============================================================
# YEAR COVERAGE
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 2 — YEAR COVERAGE")
write(report, "=" * 100)

expected_years = list(range(2012, 2025))

for year in expected_years:

    count = year_counts.get(year, 0)

    write(
        report,
        f"{year}: {count:,} files"
    )


extra_years = sorted(
    y for y in year_counts
    if y not in expected_years
)

if extra_years:
    write(report)
    write(report, f"WARNING — files found outside expected years: {extra_years}")
else:
    write(report)
    write(report, "No files found outside 2012–2024.")


# ============================================================
# MONTH COVERAGE
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 3 — MONTH COVERAGE")
write(report, "=" * 100)

for year in expected_years:

    missing_months = []

    for month in range(1, 13):

        count = month_counts.get((year, month), 0)

        if count == 0:
            missing_months.append(month)

    if missing_months:
        write(
            report,
            f"{year}: MISSING MONTHS -> {missing_months}"
        )
    else:
        write(
            report,
            f"{year}: all 12 months present"
        )


# ============================================================
# DAC SUMMARY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 4 — DAC / DATA PROVIDER DISTRIBUTION")
write(report, "=" * 100)

write(report, f"Unique DACs detected: {len(dac_counts)}")
write(report)

for dac, count in dac_counts.most_common():
    write(report, f"{dac:20s}: {count:,} files")


# ============================================================
# FLOAT SUMMARY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 5 — FLOAT / WMO SUMMARY")
write(report, "=" * 100)

write(report, f"Unique WMO identifiers detected: {len(wmo_counts):,}")

write(report)
write(report, "Top 30 WMO floats by file count:")

for wmo, count in wmo_counts.most_common(30):
    write(report, f"  {wmo}: {count:,}")


# ============================================================
# REPRESENTATIVE SAMPLE SELECTION
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 6 — SCIENTIFIC SAMPLE SELECTION")
write(report, "=" * 100)

print("Selecting representative scientific sample...")


sample_files = []

# ------------------------------------------------------------
# 1. One file per year
# ------------------------------------------------------------

for year in expected_years:

    candidates = files_by_year.get(year, [])

    if candidates:
        sample_files.append(candidates[len(candidates) // 2])


# ------------------------------------------------------------
# 2. One file per DAC
# ------------------------------------------------------------

for dac, candidates in files_by_dac.items():

    if candidates:
        sample_files.append(candidates[len(candidates) // 2])


# ------------------------------------------------------------
# 3. Add beginning/end files
# ------------------------------------------------------------

sample_files.extend(all_nc_files[:5])
sample_files.extend(all_nc_files[-5:])


# ------------------------------------------------------------
# Remove duplicates
# ------------------------------------------------------------

unique_samples = []

seen = set()

for path in sample_files:

    if path not in seen:
        unique_samples.append(path)
        seen.add(path)


sample_files = unique_samples[:MAX_SCIENTIFIC_SAMPLES]

write(
    report,
    f"Scientific files selected: {len(sample_files)}"
)


# ============================================================
# SCIENTIFIC FILE AUDIT
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 7 — SCIENTIFIC NETCDF AUDIT")
write(report, "=" * 100)

sample_results = []

required_presence = Counter()
dimension_signatures = Counter()
variable_signatures = Counter()

lat_values = []
lon_values = []

global_temp_values = []
global_psal_values = []
global_pres_values = []

files_with_errors = []
files_missing_core_variables = []


for index, path in enumerate(sample_files, start=1):

    print(
        f"[{index:03d}/{len(sample_files):03d}] "
        f"{path.name}"
    )

    result = {
        "file": str(path),
        "relative_path": str(path.relative_to(ARGO_ROOT)),
        "year": None,
        "month": None,
        "dac": detect_dac(path),
        "wmo": detect_wmo(path),
        "classification": classify_file(path),
        "open_status": "UNKNOWN",
    }

    year, month = parse_year_month_from_parent(path)

    result["year"] = year
    result["month"] = month

    try:

        # ----------------------------------------------------
        # Open
        # ----------------------------------------------------

        with xr.open_dataset(
            path,
            decode_cf=True,
            mask_and_scale=True
        ) as ds:

            result["open_status"] = "PASS"

            result["dimensions"] = {
                str(k): int(v)
                for k, v in ds.sizes.items()
            }

            result["variables"] = sorted(
                [str(v) for v in ds.variables]
            )

            result["data_variables"] = sorted(
                [str(v) for v in ds.data_vars]
            )

            dimension_signatures[
                tuple(sorted(ds.sizes.items()))
            ] += 1

            variable_signatures[
                tuple(sorted(ds.variables))
            ] += 1

            # ------------------------------------------------
            # Required variables
            # ------------------------------------------------

            found = check_required_variables(ds)

            result["required_variables"] = found

            for logical_name, actual_name in found.items():

                if actual_name is not None:
                    required_presence[logical_name] += 1

            missing_core = [
                name
                for name, actual
                in found.items()
                if actual is None
            ]

            result["missing_core_variables"] = missing_core

            if missing_core:
                files_missing_core_variables.append(
                    str(path)
                )

            # ------------------------------------------------
            # Coordinate information
            # ------------------------------------------------

            lat_name = found["latitude"]
            lon_name = found["longitude"]

            if lat_name:

                lat = np.asarray(
                    ds[lat_name].values,
                    dtype=float
                )

                lat_stats = finite_stats(lat)

                result["latitude_stats"] = lat_stats

                finite_lat = lat[np.isfinite(lat)]

                if finite_lat.size:
                    lat_values.extend(
                        finite_lat.tolist()
                    )

                    result["latitude_in_nio"] = bool(
                        np.any(
                            (finite_lat >= LAT_MIN)
                            & (finite_lat <= LAT_MAX)
                        )
                    )

            if lon_name:

                lon = np.asarray(
                    ds[lon_name].values,
                    dtype=float
                )

                lon_stats = finite_stats(lon)

                result["longitude_stats"] = lon_stats

                finite_lon = lon[np.isfinite(lon)]

                if finite_lon.size:
                    lon_values.extend(
                        finite_lon.tolist()
                    )

                    # Handle possible 0–360 longitude convention
                    lon_for_region = finite_lon.copy()

                    lon_for_region = np.where(
                        lon_for_region > 180,
                        lon_for_region - 360,
                        lon_for_region
                    )

                    result["longitude_in_nio"] = bool(
                        np.any(
                            (lon_for_region >= LON_MIN)
                            & (lon_for_region <= LON_MAX)
                        )
                    )

            # ------------------------------------------------
            # Time
            # ------------------------------------------------

            time_name = found["time"]

            if time_name:

                try:

                    time_values = ds[time_name].values

                    result["time_dtype"] = str(
                        np.asarray(time_values).dtype
                    )

                    result["time_count"] = int(
                        np.asarray(time_values).size
                    )

                    try:

                        decoded_times = pd.to_datetime(
                            time_values
                        )

                        result["time_min"] = str(
                            decoded_times.min()
                        )

                        result["time_max"] = str(
                            decoded_times.max()
                        )

                        # Check target range
                        result["time_in_expected_range"] = bool(
                            (
                                decoded_times.min()
                                >= EXPECTED_START
                            )
                            and
                            (
                                decoded_times.max()
                                <= EXPECTED_END + pd.Timedelta(days=1)
                            )
                        )

                    except Exception as exc:

                        result["time_decode_error"] = str(exc)

                except Exception as exc:

                    result["time_error"] = str(exc)

            # ------------------------------------------------
            # Pressure
            # ------------------------------------------------

            pressure_name = found["pressure"]

            if pressure_name:

                arr = np.asarray(
                    ds[pressure_name].values,
                    dtype=float
                )

                stats = finite_stats(arr)

                result["pressure_stats"] = stats

                finite = arr[np.isfinite(arr)]

                if finite.size:

                    global_pres_values.extend(
                        finite.ravel().tolist()
                    )

                    result["pressure_negative_values"] = int(
                        np.sum(finite < 0)
                    )

            # ------------------------------------------------
            # Temperature
            # ------------------------------------------------

            temp_name = found["temperature"]

            if temp_name:

                arr = np.asarray(
                    ds[temp_name].values,
                    dtype=float
                )

                stats = finite_stats(arr)

                result["temperature_stats"] = stats

                finite = arr[np.isfinite(arr)]

                if finite.size:

                    # Keep aggregate sample bounded
                    if finite.size > 50000:
                        finite = finite.ravel()[
                            ::max(1, finite.size // 50000)
                        ]

                    global_temp_values.extend(
                        finite.ravel().tolist()
                    )

            # ------------------------------------------------
            # Salinity
            # ------------------------------------------------

            psal_name = found["salinity"]

            if psal_name:

                arr = np.asarray(
                    ds[psal_name].values,
                    dtype=float
                )

                stats = finite_stats(arr)

                result["salinity_stats"] = stats

                finite = arr[np.isfinite(arr)]

                if finite.size:

                    if finite.size > 50000:
                        finite = finite.ravel()[
                            ::max(1, finite.size // 50000)
                        ]

                    global_psal_values.extend(
                        finite.ravel().tolist()
                    )

            # ------------------------------------------------
            # Variable metadata
            # ------------------------------------------------

            metadata = {}

            for logical_name, actual_name in found.items():

                if actual_name is None:
                    continue

                var = ds[actual_name]

                metadata[logical_name] = {
                    "name": actual_name,
                    "dtype": str(var.dtype),
                    "dimensions": [
                        str(x)
                        for x in var.dims
                    ],
                    "shape": [
                        int(x)
                        for x in var.shape
                    ],
                    "attrs": {
                        str(k): safe_str(v)
                        for k, v in var.attrs.items()
                    },
                    "encoding": {
                        str(k): safe_str(v)
                        for k, v in var.encoding.items()
                        if k in [
                            "_FillValue",
                            "missing_value",
                            "scale_factor",
                            "add_offset",
                            "dtype"
                        ]
                    }
                }

            result["core_variable_metadata"] = metadata

            # ------------------------------------------------
            # Global metadata
            # ------------------------------------------------

            important_global_attrs = {}

            for key in [
                "title",
                "source",
                "institution",
                "history",
                "date_created",
                "data_type",
                "featureType",
                "platform_type",
                "conventions",
                "reference_date_time",
            ]:

                if key in ds.attrs:
                    important_global_attrs[key] = safe_str(
                        ds.attrs[key]
                    )

            result["global_metadata"] = important_global_attrs

    except Exception as exc:

        result["open_status"] = "FAIL"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

        files_with_errors.append(str(path))

    sample_results.append(result)


# ============================================================
# CORE VARIABLE PRESENCE SUMMARY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 8 — CORE ARGO VARIABLE PRESENCE")
write(report, "=" * 100)

for name in [
    "latitude",
    "longitude",
    "time",
    "pressure",
    "temperature",
    "salinity",
]:

    count = required_presence[name]

    write(
        report,
        f"{name:15s}: {count:3d}/{len(sample_files)} "
        f"sample files"
    )


# ============================================================
# DIMENSION SIGNATURES
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 9 — DIMENSION STRUCTURES")
write(report, "=" * 100)

write(report, f"Unique dimension signatures: {len(dimension_signatures)}")

for dims, count in dimension_signatures.most_common(20):

    write(
        report,
        f"{count:3d} files -> {dict(dims)}"
    )


# ============================================================
# VARIABLE SIGNATURES
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 10 — VARIABLE STRUCTURES")
write(report, "=" * 100)

write(
    report,
    f"Unique variable signatures: {len(variable_signatures)}"
)

for variables, count in variable_signatures.most_common(20):

    write(
        report,
        f"{count:3d} files -> {list(variables)}"
    )


# ============================================================
# AGGREGATE GEOGRAPHIC SUMMARY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 11 — GEOGRAPHIC COVERAGE")
write(report, "=" * 100)

if lat_values:

    write(
        report,
        f"Sample latitude min: {min(lat_values):.6f}"
    )

    write(
        report,
        f"Sample latitude max: {max(lat_values):.6f}"
    )

if lon_values:

    # Normalize for reporting
    lon_norm = np.where(
        np.asarray(lon_values) > 180,
        np.asarray(lon_values) - 360,
        np.asarray(lon_values)
    )

    write(
        report,
        f"Sample longitude min: {np.min(lon_norm):.6f}"
    )

    write(
        report,
        f"Sample longitude max: {np.max(lon_norm):.6f}"
    )


# ============================================================
# AGGREGATE SCIENTIFIC RANGE CHECKS
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 12 — PHYSICAL VALUE RANGE CHECK")
write(report, "=" * 100)


def report_range(name, values, lower, upper):

    if not values:

        write(
            report,
            f"{name}: no finite values collected"
        )

        return

    arr = np.asarray(values, dtype=float)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:

        write(
            report,
            f"{name}: no finite values"
        )

        return

    min_v = float(np.min(finite))
    max_v = float(np.max(finite))

    suspicious = np.sum(
        (finite < lower) | (finite > upper)
    )

    write(
        report,
        f"{name}: min={min_v:.6f}, max={max_v:.6f}"
    )

    write(
        report,
        f"  Values outside [{lower}, {upper}]: "
        f"{int(suspicious):,}"
    )


# These are broad screening limits, NOT scientific QC thresholds.
report_range(
    "Temperature",
    global_temp_values,
    -5,
    40
)

report_range(
    "Salinity",
    global_psal_values,
    0,
    45
)

report_range(
    "Pressure",
    global_pres_values,
    0,
    12000
)


# ============================================================
# ERROR SUMMARY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 13 — ERRORS / WARNINGS")
write(report, "=" * 100)

write(
    report,
    f"Files that failed to open: {len(files_with_errors)}"
)

write(
    report,
    f"Files missing one or more core variables: "
    f"{len(files_missing_core_variables)}"
)

if files_with_errors:

    write(report)
    write(report, "Files with open errors:")

    for item in files_with_errors[:50]:
        write(report, f"  {item}")

if files_missing_core_variables:

    write(report)
    write(report, "Files missing core variables:")

    for item in files_missing_core_variables[:50]:
        write(report, f"  {item}")


# ============================================================
# SAMPLE CSV
# ============================================================

rows = []

for result in sample_results:

    row = {
        "file": result.get("file"),
        "relative_path": result.get("relative_path"),
        "year": result.get("year"),
        "month": result.get("month"),
        "dac": result.get("dac"),
        "wmo": result.get("wmo"),
        "classification": result.get("classification"),
        "open_status": result.get("open_status"),
        "missing_core_variables": ",".join(
            result.get("missing_core_variables", [])
        ),
        "time_min": result.get("time_min"),
        "time_max": result.get("time_max"),
        "time_in_expected_range":
            result.get("time_in_expected_range"),
        "latitude_min":
            result.get("latitude_stats", {}).get("min"),
        "latitude_max":
            result.get("latitude_stats", {}).get("max"),
        "longitude_min":
            result.get("longitude_stats", {}).get("min"),
        "longitude_max":
            result.get("longitude_stats", {}).get("max"),
        "temperature_min":
            result.get("temperature_stats", {}).get("min"),
        "temperature_max":
            result.get("temperature_stats", {}).get("max"),
        "salinity_min":
            result.get("salinity_stats", {}).get("min"),
        "salinity_max":
            result.get("salinity_stats", {}).get("max"),
        "pressure_min":
            result.get("pressure_stats", {}).get("min"),
        "pressure_max":
            result.get("pressure_stats", {}).get("max"),
        "error":
            result.get("error"),
    }

    rows.append(row)


pd.DataFrame(rows).to_csv(
    CSV_FILE,
    index=False
)


# ============================================================
# JSON REPORT
# ============================================================

json_summary = {
    "argo_root": str(ARGO_ROOT),
    "total_nc_files": len(all_nc_files),
    "classification_counts":
        dict(classification_counts),
    "year_counts":
        {str(k): int(v) for k, v in year_counts.items()},
    "dac_counts":
        dict(dac_counts),
    "unique_wmo_count":
        len(wmo_counts),
    "sample_count":
        len(sample_files),
    "required_variable_presence":
        dict(required_presence),
    "dimension_signatures":
        {
            str(k): int(v)
            for k, v in dimension_signatures.items()
        },
    "files_with_errors":
        files_with_errors,
    "files_missing_core_variables":
        files_missing_core_variables,
    "sample_results":
        sample_results,
}

with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_summary,
        f,
        indent=2,
        default=str
    )


# ============================================================
# FINAL VERDICT — PRELIMINARY ONLY
# ============================================================

write(report)
write(report, "=" * 100)
write(report, "SECTION 14 — PRELIMINARY AUDIT STATUS")
write(report, "=" * 100)

preliminary_flags = []

if not all_nc_files:
    preliminary_flags.append(
        "NO_NETCDF_FILES"
    )

if files_with_errors:
    preliminary_flags.append(
        "SOME_FILES_FAILED_TO_OPEN"
    )

if files_missing_core_variables:
    preliminary_flags.append(
        "CORE_VARIABLES_MISSING_IN_SAMPLE"
    )

for required in [
    "latitude",
    "longitude",
    "time",
    "pressure",
    "temperature",
    "salinity",
]:

    if required_presence[required] == 0:
        preliminary_flags.append(
            f"NO_{required.upper()}_FOUND"
        )


if preliminary_flags:

    write(report, "STATUS: INVESTIGATION REQUIRED")

    write(report)
    write(report, "Flags:")

    for flag in preliminary_flags:
        write(report, f"  - {flag}")

else:

    write(
        report,
        "STATUS: PRELIMINARY STRUCTURAL CHECK PASSED"
    )

    write(
        report,
        "IMPORTANT: This is NOT yet the final Argo scientific verdict."
    )

write(report)
write(report, "=" * 100)
write(report, "END OF ARGO INITIAL SCIENTIFIC AUDIT")
write(report, "=" * 100)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report))


print()
print("=" * 80)
print("ARGO INITIAL AUDIT COMPLETE")
print("=" * 80)
print(f"NetCDF files found : {len(all_nc_files):,}")
print(f"Scientific samples : {len(sample_files):,}")
print()
print("Reports saved:")
print(f"  {REPORT_FILE}")
print(f"  {CSV_FILE}")
print(f"  {JSON_FILE}")
print("=" * 80)