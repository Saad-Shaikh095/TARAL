from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import hashlib
import traceback
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oisss"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\oisss_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2012-01-01")
END_DATE = pd.Timestamp("2024-12-31")

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0
EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0

# Broad physical sanity limits.
# These are NOT used to modify the data.
SSS_MIN_REASONABLE = 0.0
SSS_MAX_REASONABLE = 50.0

# Files whose missing fraction is unusually large will be highlighted.
HIGH_MISSING_THRESHOLD = 0.80

# Years specifically suspected by the user.
SUSPECT_YEARS = {2019, 2022, 2024}


# ============================================================
# RUN METADATA
# ============================================================

RUN_ID = (
    "OISSS-SCI-"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + hashlib.sha1(
        datetime.now(timezone.utc).isoformat().encode()
    ).hexdigest()[:8].upper()
)

RUN_START_UTC = datetime.now(timezone.utc)
RUN_START_IST = RUN_START_UTC.astimezone(
    timezone(timedelta(hours=5, minutes=30))
)


# ============================================================
# OUTPUT FILES
# ============================================================

REPORT_TXT = OUTPUT_DIR / "oisss_scientific_audit_report.txt"
REPORT_JSON = OUTPUT_DIR / "oisss_scientific_audit_report.json"

FILE_INVENTORY_CSV = OUTPUT_DIR / "oisss_file_inventory.csv"
FILE_STATS_CSV = OUTPUT_DIR / "oisss_file_statistics.csv"
YEAR_STATS_CSV = OUTPUT_DIR / "oisss_year_statistics.csv"
MONTH_STATS_CSV = OUTPUT_DIR / "oisss_month_statistics.csv"

VARIABLE_INVENTORY_CSV = OUTPUT_DIR / "oisss_variable_inventory.csv"
SIGNATURES_CSV = OUTPUT_DIR / "oisss_structure_signatures.csv"

MISSING_CELLS_CSV = OUTPUT_DIR / "oisss_missing_cell_statistics.csv"
SUSPECT_FILES_CSV = OUTPUT_DIR / "oisss_suspect_files.csv"

PROBLEM_FILES_CSV = OUTPUT_DIR / "oisss_problem_files.csv"


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def normalize_attr(value):
    """
    Convert NetCDF attribute values into JSON-safe Python objects.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [
            normalize_attr(v)
            for v in value
        ]

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def find_sss_variable(ds):
    """
    Try to identify the salinity variable without assuming
    a single exact variable name.
    """

    preferred_names = [
        "sss",
        "sos",
        "so",
        "salinity",
        "sea_surface_salinity",
        "sea_surface_salinity_anomaly",
    ]

    lower_map = {
        name.lower(): name
        for name in ds.data_vars
    }

    for preferred in preferred_names:
        if preferred.lower() in lower_map:
            return lower_map[preferred.lower()]

    # Second-pass heuristic using metadata.
    candidates = []

    for name, da in ds.data_vars.items():

        text = " ".join(
            [
                str(name),
                str(da.attrs.get("long_name", "")),
                str(da.attrs.get("standard_name", "")),
                str(da.attrs.get("description", "")),
            ]
        ).lower()

        if "salinity" in text:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    return None


def find_lat_lon(ds):

    lat_candidates = [
        "lat",
        "latitude",
        "nav_lat",
    ]

    lon_candidates = [
        "lon",
        "longitude",
        "nav_lon",
    ]

    lat_name = None
    lon_name = None

    for name in ds.coords:
        lname = name.lower()

        if lname in lat_candidates:
            lat_name = name

        if lname in lon_candidates:
            lon_name = name

    if lat_name is None:
        for name in ds.variables:
            lname = name.lower()

            if lname in lat_candidates:
                lat_name = name
                break

    if lon_name is None:
        for name in ds.variables:
            lname = name.lower()

            if lname in lon_candidates:
                lon_name = name
                break

    return lat_name, lon_name


def extract_date_from_filename(path):

    # Expected:
    # OISSS_L4_multimission_global_daily_v3.0_2012-01-01.nc

    stem = path.stem

    for token in stem.split("_"):

        try:
            dt = pd.Timestamp(token)

            if (
                START_DATE <= dt <= END_DATE
            ):
                return dt.normalize()

        except Exception:
            pass

    # Fallback: search all hyphenated date-like substrings.
    import re

    match = re.search(
        r"(20\d{2}-\d{2}-\d{2})",
        stem
    )

    if match:
        try:
            return pd.Timestamp(
                match.group(1)
            ).normalize()
        except Exception:
            pass

    return None


def get_fill_values(da):

    values = []

    for key in [
        "_FillValue",
        "missing_value",
    ]:
        if key in da.attrs:
            value = da.attrs[key]

            if np.ndim(value) == 0:
                values.append(
                    safe_float(value)
                )
            else:
                for item in np.ravel(value):
                    values.append(
                        safe_float(item)
                    )

    # Remove None and duplicates.
    return sorted(
        set(
            v for v in values
            if v is not None
        )
    )


def build_structure_signature(ds, sss_name):

    signature = {
        "sss_variable": sss_name,
        "variables": [],
        "dimensions": {},
        "coordinates": {},
    }

    for name in sorted(ds.data_vars):

        da = ds[name]

        signature["variables"].append({
            "name": name,
            "dims": list(da.dims),
            "shape": list(da.shape),
            "dtype": str(da.dtype),
        })

    for name, size in ds.sizes.items():
        signature["dimensions"][name] = int(size)

    lat_name, lon_name = find_lat_lon(ds)

    for coord_name in [lat_name, lon_name]:

        if coord_name is None:
            continue

        da = ds[coord_name]

        signature["coordinates"][coord_name] = {
            "dims": list(da.dims),
            "shape": list(da.shape),
            "dtype": str(da.dtype),
        }

    return json.dumps(
        signature,
        sort_keys=True,
    )


def array_nan_mask(arr):

    arr = np.asarray(arr)

    return ~np.isfinite(arr)


# ============================================================
# DISCOVER FILES
# ============================================================

print("=" * 80)
print("OISSS SCIENTIFIC + PIPELINE READINESS AUDIT")
print("=" * 80)
print(f"Run ID:       {RUN_ID}")
print(f"Raw root:     {RAW_ROOT}")
print(f"Output:       {OUTPUT_DIR}")
print(f"Period:       {START_DATE.date()} -> {END_DATE.date()}")
print(f"Domain:       {EXPECTED_LAT_MIN}-{EXPECTED_LAT_MAX} N")
print(f"              {EXPECTED_LON_MIN}-{EXPECTED_LON_MAX} E")
print("Raw data modified: NO")
print()

all_files = sorted(
    RAW_ROOT.rglob("*.nc")
)

print(f"NetCDF files discovered: {len(all_files):,}")

if not all_files:
    raise RuntimeError(
        f"No NetCDF files found under {RAW_ROOT}"
    )


# ============================================================
# DATE INVENTORY
# ============================================================

file_inventory = []

for path in all_files:

    file_date = extract_date_from_filename(path)

    file_inventory.append({
        "path": str(path),
        "filename": path.name,
        "date": (
            file_date.strftime("%Y-%m-%d")
            if file_date is not None
            else None
        ),
        "year": (
            int(file_date.year)
            if file_date is not None
            else None
        ),
        "month": (
            int(file_date.month)
            if file_date is not None
            else None
        ),
        "size_bytes": path.stat().st_size,
    })


inventory_df = pd.DataFrame(file_inventory)

inventory_df.to_csv(
    FILE_INVENTORY_CSV,
    index=False,
)


# ============================================================
# EXPECTED CALENDAR
# ============================================================

expected_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D",
)

expected_date_set = set(
    expected_dates.strftime("%Y-%m-%d")
)

observed_dates = set(
    inventory_df["date"]
    .dropna()
)

missing_dates = sorted(
    expected_date_set - observed_dates
)

unexpected_dates = sorted(
    observed_dates - expected_date_set
)

duplicate_dates = (
    inventory_df[
        inventory_df["date"].notna()
    ]
    .groupby("date")
    .size()
)

duplicate_dates = duplicate_dates[
    duplicate_dates > 1
]


# ============================================================
# FULL SCIENTIFIC FILE SCAN
# ============================================================

file_stats = []
variable_inventory = []
problem_files = []

coordinate_signatures = {}
structure_signatures = {}

# For determining whether a location is persistently valid/missing.
global_valid_counts = None
global_total_counts = None

reference_lat = None
reference_lon = None

files_opened = 0
files_failed = 0

total_values = 0
total_missing = 0
total_zero = 0

global_sss_min = np.inf
global_sss_max = -np.inf

global_bad_physical = 0

suspect_year_stats = {
    year: {
        "files": 0,
        "values": 0,
        "missing": 0,
        "zero": 0,
        "bad_physical": 0,
    }
    for year in SUSPECT_YEARS
}


for idx, row in inventory_df.iterrows():

    path = Path(row["path"])

    print(
        f"[{idx + 1:,}/{len(inventory_df):,}] "
        f"{path.name}"
    )

    stats = {
        "path": str(path),
        "filename": path.name,
        "date": row["date"],
        "year": row["year"],
        "month": row["month"],
        "open_ok": False,
        "sss_variable": None,
        "sss_units": None,
        "sss_long_name": None,
        "dimensions": None,
        "lat_name": None,
        "lon_name": None,
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "shape": None,
        "dtype": None,
        "fill_values": None,
        "total_values": 0,
        "missing_values": 0,
        "missing_fraction": None,
        "zero_values": 0,
        "zero_fraction": None,
        "finite_values": 0,
        "min_sss": None,
        "max_sss": None,
        "bad_physical_values": 0,
        "structure_signature": None,
        "error": None,
    }

    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ds = xr.open_dataset(
                path,
                decode_times=True,
                mask_and_scale=True,
            )

        files_opened += 1
        stats["open_ok"] = True

        # ----------------------------------------------------
        # Identify SSS variable
        # ----------------------------------------------------

        sss_name = find_sss_variable(ds)

        stats["sss_variable"] = sss_name

        if sss_name is None:

            stats["error"] = (
                "Could not identify salinity variable"
            )

            problem_files.append({
                "path": str(path),
                "date": row["date"],
                "year": row["year"],
                "problem": (
                    "NO_SSS_VARIABLE_IDENTIFIED"
                ),
            })

            ds.close()

            file_stats.append(stats)
            continue

        da = ds[sss_name]

        stats["sss_units"] = (
            da.attrs.get("units")
        )

        stats["sss_long_name"] = (
            da.attrs.get("long_name")
        )

        stats["fill_values"] = get_fill_values(
            da
        )

        stats["dimensions"] = json.dumps(
            {
                k: int(v)
                for k, v in ds.sizes.items()
            },
            sort_keys=True,
        )

        stats["shape"] = list(da.shape)
        stats["dtype"] = str(da.dtype)

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        lat_name, lon_name = find_lat_lon(ds)

        stats["lat_name"] = lat_name
        stats["lon_name"] = lon_name

        if lat_name is not None:
            lat = np.asarray(
                ds[lat_name].values
            ).squeeze()

            stats["lat_min"] = safe_float(
                np.nanmin(lat)
            )

            stats["lat_max"] = safe_float(
                np.nanmax(lat)
            )

        else:
            lat = None

        if lon_name is not None:
            lon = np.asarray(
                ds[lon_name].values
            ).squeeze()

            stats["lon_min"] = safe_float(
                np.nanmin(lon)
            )

            stats["lon_max"] = safe_float(
                np.nanmax(lon)
            )

        else:
            lon = None

        # ----------------------------------------------------
        # Coordinate signature
        # ----------------------------------------------------

        coord_signature = json.dumps(
            {
                "lat": (
                    np.asarray(lat).shape
                    if lat is not None
                    else None
                ),
                "lon": (
                    np.asarray(lon).shape
                    if lon is not None
                    else None
                ),
                "lat_min": stats["lat_min"],
                "lat_max": stats["lat_max"],
                "lon_min": stats["lon_min"],
                "lon_max": stats["lon_max"],
            },
            sort_keys=True,
        )

        coordinate_signatures.setdefault(
            coord_signature,
            []
        ).append(str(path))

        # ----------------------------------------------------
        # Structure signature
        # ----------------------------------------------------

        structure_signature = (
            build_structure_signature(
                ds,
                sss_name,
            )
        )

        stats["structure_signature"] = (
            structure_signature
        )

        structure_signatures.setdefault(
            structure_signature,
            []
        ).append(str(path))

        # ----------------------------------------------------
        # Load SSS
        # ----------------------------------------------------

        values = np.asarray(
            da.values
        )

        finite_mask = np.isfinite(
            values
        )

        missing_mask = ~finite_mask

        total_count = values.size
        missing_count = int(
            missing_mask.sum()
        )
        finite_count = int(
            finite_mask.sum()
        )

        zero_count = int(
            np.count_nonzero(
                finite_mask & (values == 0)
            )
        )

        stats["total_values"] = total_count
        stats["missing_values"] = missing_count
        stats["missing_fraction"] = (
            missing_count / total_count
            if total_count
            else None
        )

        stats["zero_values"] = zero_count
        stats["zero_fraction"] = (
            zero_count / total_count
            if total_count
            else None
        )

        stats["finite_values"] = finite_count

        # ----------------------------------------------------
        # Physical range
        # ----------------------------------------------------

        if finite_count:

            finite_values = values[
                finite_mask
            ]

            min_value = float(
                np.min(finite_values)
            )

            max_value = float(
                np.max(finite_values)
            )

            stats["min_sss"] = min_value
            stats["max_sss"] = max_value

            global_sss_min = min(
                global_sss_min,
                min_value,
            )

            global_sss_max = max(
                global_sss_max,
                max_value,
            )

            bad_physical_mask = (
                finite_mask
                &
                (
                    (values < SSS_MIN_REASONABLE)
                    |
                    (values > SSS_MAX_REASONABLE)
                )
            )

            bad_physical = int(
                bad_physical_mask.sum()
            )

            stats["bad_physical_values"] = (
                bad_physical
            )

            global_bad_physical += (
                bad_physical
            )

        # ----------------------------------------------------
        # Global counters
        # ----------------------------------------------------

        total_values += total_count
        total_missing += missing_count
        total_zero += zero_count

        # ----------------------------------------------------
        # Suspect-year counters
        # ----------------------------------------------------

        year = row["year"]

        if year in SUSPECT_YEARS:

            suspect_year_stats[year][
                "files"
            ] += 1

            suspect_year_stats[year][
                "values"
            ] += total_count

            suspect_year_stats[year][
                "missing"
            ] += missing_count

            suspect_year_stats[year][
                "zero"
            ] += zero_count

            suspect_year_stats[year][
                "bad_physical"
            ] += stats[
                "bad_physical_values"
            ]

        # ----------------------------------------------------
        # Variable inventory
        # ----------------------------------------------------

        variable_inventory.append({
            "path": str(path),
            "date": row["date"],
            "year": row["year"],
            "month": row["month"],
            "variable": sss_name,
            "dims": json.dumps(
                list(da.dims)
            ),
            "shape": json.dumps(
                list(da.shape)
            ),
            "dtype": str(da.dtype),
            "units": da.attrs.get(
                "units"
            ),
            "long_name": da.attrs.get(
                "long_name"
            ),
            "fill_values": json.dumps(
                stats["fill_values"]
            ),
        })

        # ----------------------------------------------------
        # Problem classification
        # ----------------------------------------------------

        if (
            stats["missing_fraction"]
            is not None
            and
            stats["missing_fraction"]
            >= HIGH_MISSING_THRESHOLD
        ):

            problem_files.append({
                "path": str(path),
                "date": row["date"],
                "year": row["year"],
                "problem": (
                    "HIGH_MISSING_FRACTION"
                ),
                "missing_fraction": (
                    stats["missing_fraction"]
                ),
            })

        if (
            stats["bad_physical_values"]
            > 0
        ):

            problem_files.append({
                "path": str(path),
                "date": row["date"],
                "year": row["year"],
                "problem": (
                    "PHYSICAL_RANGE_OUTLIERS"
                ),
                "bad_physical_values": (
                    stats[
                        "bad_physical_values"
                    ]
                ),
            })

        # ----------------------------------------------------
        # Spatial temporal availability
        #
        # We only initialize this after the first
        # successfully read SSS array.
        #
        # IMPORTANT:
        # This is initially done on the full native grid.
        # We will later use it to distinguish persistent
        # missing cells from intermittent missing cells.
        # ----------------------------------------------------

        # Collapse any singleton time/depth dimensions.
        spatial_values = np.squeeze(
            values
        )

        if (
            spatial_values.ndim == 2
            and
            lat is not None
            and
            lon is not None
        ):

            current_valid = np.isfinite(
                spatial_values
            )

            # Ensure orientation is compatible.
            if global_valid_counts is None:

                global_valid_counts = (
                    np.zeros_like(
                        current_valid,
                        dtype=np.int64,
                    )
                )

                global_total_counts = (
                    np.zeros_like(
                        current_valid,
                        dtype=np.int64,
                    )
                )

                reference_lat = np.asarray(
                    lat
                ).copy()

                reference_lon = np.asarray(
                    lon
                ).copy()

            if (
                current_valid.shape
                == global_valid_counts.shape
            ):

                global_valid_counts += (
                    current_valid.astype(
                        np.int64
                    )
                )

                global_total_counts += 1

        ds.close()

    except Exception as exc:

        files_failed += 1

        stats["error"] = (
            repr(exc)
        )

        problem_files.append({
            "path": str(path),
            "date": row["date"],
            "year": row["year"],
            "problem": "OPEN_OR_PROCESS_FAILURE",
            "error": repr(exc),
        })

        print(
            f"    ERROR: {exc}"
        )

    file_stats.append(stats)


# ============================================================
# DATAFRAMES
# ============================================================

file_stats_df = pd.DataFrame(
    file_stats
)

file_stats_df.to_csv(
    FILE_STATS_CSV,
    index=False,
)

variable_inventory_df = pd.DataFrame(
    variable_inventory
)

variable_inventory_df.to_csv(
    VARIABLE_INVENTORY_CSV,
    index=False,
)

problem_files_df = pd.DataFrame(
    problem_files
)

problem_files_df.to_csv(
    PROBLEM_FILES_CSV,
    index=False,
)


# ============================================================
# YEAR STATISTICS
# ============================================================

year_records = []

for year in range(
    START_DATE.year,
    END_DATE.year + 1,
):

    subset = file_stats_df[
        file_stats_df["year"] == year
    ]

    total = int(
        subset["total_values"]
        .fillna(0)
        .sum()
    )

    missing = int(
        subset["missing_values"]
        .fillna(0)
        .sum()
    )

    zero = int(
        subset["zero_values"]
        .fillna(0)
        .sum()
    )

    bad = int(
        subset["bad_physical_values"]
        .fillna(0)
        .sum()
    )

    year_records.append({
        "year": year,
        "files": len(subset),
        "total_values": total,
        "missing_values": missing,
        "missing_fraction": (
            missing / total
            if total
            else None
        ),
        "zero_values": zero,
        "zero_fraction": (
            zero / total
            if total
            else None
        ),
        "bad_physical_values": bad,
        "files_high_missing": int(
            (
                subset[
                    "missing_fraction"
                ]
                >= HIGH_MISSING_THRESHOLD
            ).sum()
        ),
    })


year_stats_df = pd.DataFrame(
    year_records
)

year_stats_df.to_csv(
    YEAR_STATS_CSV,
    index=False,
)


# ============================================================
# MONTH STATISTICS
# ============================================================

month_records = []

for year in range(
    START_DATE.year,
    END_DATE.year + 1,
):

    for month in range(1, 13):

        subset = file_stats_df[
            (file_stats_df["year"] == year)
            &
            (file_stats_df["month"] == month)
        ]

        if subset.empty:
            continue

        total = int(
            subset["total_values"]
            .fillna(0)
            .sum()
        )

        missing = int(
            subset["missing_values"]
            .fillna(0)
            .sum()
        )

        month_records.append({
            "year": year,
            "month": month,
            "files": len(subset),
            "total_values": total,
            "missing_values": missing,
            "missing_fraction": (
                missing / total
                if total
                else None
            ),
            "files_high_missing": int(
                (
                    subset[
                        "missing_fraction"
                    ]
                    >= HIGH_MISSING_THRESHOLD
                ).sum()
            ),
        })


month_stats_df = pd.DataFrame(
    month_records
)

month_stats_df.to_csv(
    MONTH_STATS_CSV,
    index=False,
)


# ============================================================
# STRUCTURE SIGNATURE REPORT
# ============================================================

signature_records = []

for sig, paths in structure_signatures.items():

    signature_records.append({
        "signature_id": hashlib.sha1(
            sig.encode()
        ).hexdigest()[:12],
        "file_count": len(paths),
        "representative_file": paths[0],
        "signature": sig,
    })


signatures_df = pd.DataFrame(
    signature_records
)

signatures_df.to_csv(
    SIGNATURES_CSV,
    index=False,
)


# ============================================================
# TEMPORAL CELL AVAILABILITY
# ============================================================

missing_cell_records = []

if (
    global_valid_counts is not None
    and
    global_total_counts is not None
):

    availability = (
        global_valid_counts
        /
        np.maximum(
            global_total_counts,
            1,
        )
    )

    always_missing = (
        availability == 0
    )

    always_valid = (
        availability == 1
    )

    intermittent = (
        (availability > 0)
        &
        (availability < 1)
    )

    for category, mask in [
        (
            "always_missing",
            always_missing,
        ),
        (
            "always_valid",
            always_valid,
        ),
        (
            "intermittent",
            intermittent,
        ),
    ]:

        indices = np.argwhere(mask)

        for index in indices:

            i, j = index

            lat_value = (
                reference_lat[i]
                if reference_lat.ndim == 1
                and i < len(reference_lat)
                else np.nan
            )

            lon_value = (
                reference_lon[j]
                if reference_lon.ndim == 1
                and j < len(reference_lon)
                else np.nan
            )

            missing_cell_records.append({
                "category": category,
                "lat_index": int(i),
                "lon_index": int(j),
                "latitude": safe_float(
                    lat_value
                ),
                "longitude": safe_float(
                    lon_value
                ),
                "valid_count": int(
                    global_valid_counts[i, j]
                ),
                "total_count": int(
                    global_total_counts[i, j]
                ),
                "availability_fraction": (
                    float(
                        availability[i, j]
                    )
                ),
                "missing_count": int(
                    global_total_counts[i, j]
                    -
                    global_valid_counts[i, j]
                ),
            })


missing_cells_df = pd.DataFrame(
    missing_cell_records
)

missing_cells_df.to_csv(
    MISSING_CELLS_CSV,
    index=False,
)


# ============================================================
# SUSPECT-YEAR SUMMARY
# ============================================================

suspect_records = []

for year, stats in suspect_year_stats.items():

    total = stats["values"]
    missing = stats["missing"]

    suspect_records.append({
        "year": year,
        "files": stats["files"],
        "total_values": total,
        "missing_values": missing,
        "missing_fraction": (
            missing / total
            if total
            else None
        ),
        "zero_values": stats["zero"],
        "bad_physical_values": (
            stats["bad_physical"]
        ),
    })


suspect_df = pd.DataFrame(
    suspect_records
)

suspect_df.to_csv(
    SUSPECT_FILES_CSV,
    index=False,
)


# ============================================================
# FINAL COUNTS
# ============================================================

RUN_END_UTC = datetime.now(timezone.utc)
RUN_END_IST = RUN_END_UTC.astimezone(
    timezone(timedelta(hours=5, minutes=30))
)

duration_seconds = (
    RUN_END_UTC - RUN_START_UTC
).total_seconds()


# ============================================================
# DECISION FLAGS
# ============================================================

calendar_ok = (
    len(missing_dates) == 0
    and
    len(unexpected_dates) == 0
)

readability_ok = (
    files_failed == 0
)

structure_ok = (
    len(structure_signatures) == 1
)

sss_identified_all = (
    file_stats_df[
        "sss_variable"
    ].notna().all()
)

geographic_ok = (
    (
        file_stats_df["lat_min"]
        >= EXPECTED_LAT_MIN
    ).all()
    and
    (
        file_stats_df["lat_max"]
        <= EXPECTED_LAT_MAX
    ).all()
    and
    (
        file_stats_df["lon_min"]
        >= EXPECTED_LON_MIN
    ).all()
    and
    (
        file_stats_df["lon_max"]
        <= EXPECTED_LON_MAX
    ).all()
)

physical_ok = (
    global_bad_physical == 0
)


# ============================================================
# IMPORTANT:
# DO NOT automatically make GREEN just because the
# technical scan passes.
#
# OISSS gets a provisional status because the suspected
# ocean missing-value issue requires interpretation.
# ============================================================

if not calendar_ok:
    decision = "RED"
    decision_reason = (
        "Calendar coverage is incomplete or contains "
        "unexpected dates."
    )

elif not readability_ok:
    decision = "RED"
    decision_reason = (
        "One or more OISSS files failed to open/process."
    )

elif not sss_identified_all:
    decision = "RED"
    decision_reason = (
        "The SSS variable could not be consistently "
        "identified across the archive."
    )

elif not geographic_ok:
    decision = "RED"
    decision_reason = (
        "One or more files do not provide the expected "
        "NIO spatial domain."
    )

else:
    decision = "YELLOW"
    decision_reason = (
        "Technical acquisition and structural checks "
        "completed, but missing-value behavior must be "
        "scientifically interpreted before accepting "
        "OISSS or replacing any values."
    )


# ============================================================
# FINAL JSON REPORT
# ============================================================

summary = {
    "run": {
        "run_id": RUN_ID,
        "start_utc": RUN_START_UTC.isoformat(),
        "end_utc": RUN_END_UTC.isoformat(),
        "start_ist": RUN_START_IST.isoformat(),
        "end_ist": RUN_END_IST.isoformat(),
        "duration_seconds": duration_seconds,
    },

    "configuration": {
        "raw_root": str(RAW_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "start_date": str(
            START_DATE.date()
        ),
        "end_date": str(
            END_DATE.date()
        ),
        "domain": {
            "lat_min": EXPECTED_LAT_MIN,
            "lat_max": EXPECTED_LAT_MAX,
            "lon_min": EXPECTED_LON_MIN,
            "lon_max": EXPECTED_LON_MAX,
        },
        "raw_data_modified": False,
    },

    "files": {
        "discovered": len(all_files),
        "opened": files_opened,
        "failed": files_failed,
        "expected_days": len(expected_dates),
        "observed_dates": len(observed_dates),
        "missing_dates": missing_dates,
        "unexpected_dates": unexpected_dates,
        "duplicate_dates": (
            duplicate_dates.to_dict()
        ),
    },

    "global_data": {
        "total_values": total_values,
        "missing_values": total_missing,
        "missing_fraction": (
            total_missing / total_values
            if total_values
            else None
        ),
        "zero_values": total_zero,
        "zero_fraction": (
            total_zero / total_values
            if total_values
            else None
        ),
        "min_sss": (
            None
            if global_sss_min == np.inf
            else global_sss_min
        ),
        "max_sss": (
            None
            if global_sss_max == -np.inf
            else global_sss_max
        ),
        "bad_physical_values": (
            global_bad_physical
        ),
    },

    "structure": {
        "unique_structure_signatures": len(
            structure_signatures
        ),
        "unique_coordinate_signatures": len(
            coordinate_signatures
        ),
    },

    "suspect_years": suspect_records,

    "decision": {
        "status": decision,
        "reason": decision_reason,
        "calendar_ok": calendar_ok,
        "readability_ok": readability_ok,
        "sss_identified_all": (
            sss_identified_all
        ),
        "geographic_ok": geographic_ok,
        "physical_ok": physical_ok,
        "structure_ok": structure_ok,
    },
}


with open(
    REPORT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        default=str,
    )


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

report_lines = []

report_lines.append(
    "OISSS SCIENTIFIC + PIPELINE READINESS AUDIT"
)
report_lines.append("=" * 72)
report_lines.append("")

report_lines.append(
    f"Run ID: {RUN_ID}"
)

report_lines.append(
    f"Start UTC: {RUN_START_UTC.isoformat()}"
)

report_lines.append(
    f"End UTC:   {RUN_END_UTC.isoformat()}"
)

report_lines.append(
    f"Start IST: {RUN_START_IST.isoformat()}"
)

report_lines.append(
    f"End IST:   {RUN_END_IST.isoformat()}"
)

report_lines.append(
    f"Duration:  {duration_seconds:.2f} seconds"
)

report_lines.append("")

report_lines.append(
    f"Raw root: {RAW_ROOT}"
)

report_lines.append(
    f"Output:   {OUTPUT_DIR}"
)

report_lines.append(
    "Raw data modified: NO"
)

report_lines.append("")

report_lines.append(
    "EXPECTED DOMAIN / PERIOD"
)

report_lines.append("-" * 72)

report_lines.append(
    f"Period: {START_DATE.date()} -> "
    f"{END_DATE.date()}"
)

report_lines.append(
    f"Domain: {EXPECTED_LAT_MIN} -> "
    f"{EXPECTED_LAT_MAX} N"
)

report_lines.append(
    f"        {EXPECTED_LON_MIN} -> "
    f"{EXPECTED_LON_MAX} E"
)

report_lines.append("")

report_lines.append(
    "FILE / CALENDAR AUDIT"
)

report_lines.append("-" * 72)

report_lines.append(
    f"Files discovered: {len(all_files):,}"
)

report_lines.append(
    f"Files opened:    {files_opened:,}"
)

report_lines.append(
    f"Files failed:    {files_failed:,}"
)

report_lines.append(
    f"Expected days:   {len(expected_dates):,}"
)

report_lines.append(
    f"Observed dates:  {len(observed_dates):,}"
)

report_lines.append(
    f"Missing dates:   {len(missing_dates):,}"
)

report_lines.append(
    f"Unexpected dates:{len(unexpected_dates):,}"
)

report_lines.append(
    f"Duplicate dates: {len(duplicate_dates):,}"
)

if missing_dates:
    report_lines.append(
        "Missing date examples: "
        + ", ".join(
            missing_dates[:20]
        )
    )

report_lines.append("")

report_lines.append(
    "GLOBAL SSS STATISTICS"
)

report_lines.append("-" * 72)

report_lines.append(
    f"Total values: {total_values:,}"
)

report_lines.append(
    f"Missing:      {total_missing:,}"
)

report_lines.append(
    f"Missing %:    "
    f"{100 * total_missing / total_values:.4f}%"
    if total_values
    else "Missing %: N/A"
)

report_lines.append(
    f"Exact zeros:  {total_zero:,}"
)

report_lines.append(
    f"Min SSS:      {global_sss_min}"
)

report_lines.append(
    f"Max SSS:      {global_sss_max}"
)

report_lines.append(
    f"Physical outliers: "
    f"{global_bad_physical:,}"
)

report_lines.append("")

report_lines.append(
    "SUSPECT YEARS"
)

report_lines.append("-" * 72)

for record in suspect_records:

    report_lines.append(
        f"{record['year']}: "
        f"files={record['files']:,}, "
        f"missing={record['missing_values']:,}, "
        f"missing_fraction="
        f"{record['missing_fraction']:.6f}"
        if record["missing_fraction"]
        is not None
        else
        f"{record['year']}: "
        f"files={record['files']:,}"
    )

report_lines.append("")

report_lines.append(
    "STRUCTURE"
)

report_lines.append("-" * 72)

report_lines.append(
    f"Unique structure signatures: "
    f"{len(structure_signatures)}"
)

report_lines.append(
    f"Unique coordinate signatures: "
    f"{len(coordinate_signatures)}"
)

report_lines.append("")

report_lines.append(
    "TEMPORAL CELL AVAILABILITY"
)

report_lines.append("-" * 72)

if not missing_cells_df.empty:

    category_counts = (
        missing_cells_df[
            "category"
        ]
        .value_counts()
        .to_dict()
    )

    report_lines.append(
        f"Always valid cells: "
        f"{category_counts.get('always_valid', 0):,}"
    )

    report_lines.append(
        f"Always missing cells: "
        f"{category_counts.get('always_missing', 0):,}"
    )

    report_lines.append(
        f"Intermittent cells: "
        f"{category_counts.get('intermittent', 0):,}"
    )

else:

    report_lines.append(
        "Cell availability analysis unavailable."
    )

report_lines.append("")

report_lines.append(
    "DECISION"
)

report_lines.append("-" * 72)

report_lines.append(
    f"STATUS: {decision}"
)

report_lines.append(
    f"REASON: {decision_reason}"
)

report_lines.append("")

report_lines.append(
    "IMPORTANT: No values were replaced, filled, "
    "interpolated, clipped, or otherwise modified."
)

report_lines.append(
    "The missing-value analysis is diagnostic only."
)

with open(
    REPORT_TXT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report_lines)
    )


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print()
print("=" * 80)
print("OISSS AUDIT COMPLETE")
print("=" * 80)

print(f"Run ID:              {RUN_ID}")
print(f"Files discovered:    {len(all_files):,}")
print(f"Files opened:        {files_opened:,}")
print(f"Files failed:        {files_failed:,}")

print(
    f"Expected dates:      {len(expected_dates):,}"
)

print(
    f"Missing dates:       {len(missing_dates):,}"
)

print(
    f"Duplicate dates:     {len(duplicate_dates):,}"
)

print(
    f"Global missing:      "
    f"{100 * total_missing / total_values:.4f}%"
    if total_values
    else "Global missing:      N/A"
)

print(
    f"Global exact zeros:  {total_zero:,}"
)

print(
    f"Global SSS range:    "
    f"{global_sss_min} -> {global_sss_max}"
)

print(
    f"Physical outliers:   "
    f"{global_bad_physical:,}"
)

print()
print("SUSPECT YEARS:")

for record in suspect_records:

    print(
        f"  {record['year']}: "
        f"missing="
        f"{record['missing_fraction'] * 100:.4f}%"
        if record["missing_fraction"]
        is not None
        else
        f"  {record['year']}: N/A"
    )

print()
print(
    f"Structure signatures: "
    f"{len(structure_signatures)}"
)

print(
    f"Decision: {decision}"
)

print()
print("REPORTS:")
print(f"  {REPORT_TXT}")
print(f"  {REPORT_JSON}")
print(f"  {FILE_INVENTORY_CSV}")
print(f"  {FILE_STATS_CSV}")
print(f"  {YEAR_STATS_CSV}")
print(f"  {MONTH_STATS_CSV}")
print(f"  {MISSING_CELLS_CSV}")
print(f"  {SUSPECT_FILES_CSV}")
print(f"  {PROBLEM_FILES_CSV}")
print(f"  {SIGNATURES_CSV}")

print("=" * 80)
