from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# OISSS SPECIAL FORENSIC GAP AUDIT
#
# PURPOSE
# -------
# Identify exact missing-data patches and classify them into:
#
#   1. CONTROLLED_INTERPOLATION
#   2. INDEPENDENT_SSS_FALLBACK
#   3. KEEP_MASKED
#
# IMPORTANT
# ---------
# This script DOES NOT modify raw OISSS files.
# It DOES NOT fill or interpolate values.
#
# It only creates an evidence-based patch inventory.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

RAW_ROOT = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oisss"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\audits\oisss_special_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

START_DATE = pd.Timestamp("2012-01-01")
END_DATE = pd.Timestamp("2024-12-31")

EXPECTED_DAYS = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

EXPECTED_DAY_COUNT = len(EXPECTED_DAYS)

LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

# ------------------------------------------------------------
# Scientific policy
# ------------------------------------------------------------

# Maximum internal missing streak eligible for
# controlled temporal interpolation.
MAX_INTERPOLATION_GAP_DAYS = 3

# Any internal missing streak longer than this goes
# to independent SSS fallback.
#
# Edge gaps ALWAYS go to fallback because they lack
# a valid OISSS value on both sides.
#
# Permanent masks are handled separately.
# ------------------------------------------------------------


# ============================================================
# RUN METADATA
# ============================================================

RUN_START_UTC = datetime.now(timezone.utc)
RUN_START_IST = RUN_START_UTC.astimezone(
    timezone(timedelta(hours=5, minutes=30))
)

RUN_ID = (
    "OISSS-SPECIAL-"
    + RUN_START_UTC.strftime("%Y%m%dT%H%M%SZ")
    + "-"
    + hashlib.sha1(
        RUN_START_UTC.isoformat().encode()
    ).hexdigest()[:8].upper()
)


# ============================================================
# OUTPUT FILES
# ============================================================

REPORT_TXT = (
    OUTPUT_DIR /
    "oisss_special_audit_report.txt"
)

REPORT_JSON = (
    OUTPUT_DIR /
    "oisss_special_audit_report.json"
)

PATCHES_CSV = (
    OUTPUT_DIR /
    "oisss_gap_patches.csv"
)

INTERPOLATION_CSV = (
    OUTPUT_DIR /
    "oisss_controlled_interpolation_patches.csv"
)

FALLBACK_CSV = (
    OUTPUT_DIR /
    "oisss_independent_fallback_patches.csv"
)

MASKED_CSV = (
    OUTPUT_DIR /
    "oisss_keep_masked_cells.csv"
)

CELL_SUMMARY_CSV = (
    OUTPUT_DIR /
    "oisss_cell_gap_summary.csv"
)

YEAR_SUMMARY_CSV = (
    OUTPUT_DIR /
    "oisss_gap_year_summary.csv"
)

PROGRESS_CSV = (
    OUTPUT_DIR /
    "oisss_special_audit_progress.csv"
)

ERRORS_CSV = (
    OUTPUT_DIR /
    "oisss_special_audit_errors.csv"
)


# ============================================================
# FIXED SCHEMAS
#
# THIS IS THE IMPORTANT PART THAT PREVENTS THE
# KeyError: 'first_missing_date'
#
# Every dataframe is created with an explicit schema,
# even when it contains zero rows.
# ============================================================

PATCH_COLUMNS = [
    "patch_id",
    "action",
    "lat_index",
    "lon_index",
    "latitude",
    "longitude",
    "start_date",
    "end_date",
    "gap_days",
    "before_date",
    "after_date",
    "before_valid",
    "after_valid",
    "is_edge_gap",
    "total_missing_days_cell",
    "cell_availability_fraction",
    "reason",
]

CELL_COLUMNS = [
    "lat_index",
    "lon_index",
    "latitude",
    "longitude",
    "total_days",
    "valid_days",
    "missing_days",
    "availability_fraction",
    "missing_fraction",
    "first_missing_date",
    "last_missing_date",
    "longest_missing_streak",
    "number_of_missing_runs",
    "has_internal_gap",
    "has_edge_gap",
    "action",
]

YEAR_COLUMNS = [
    "year",
    "missing_cells_with_gap",
    "interpolation_patches",
    "fallback_patches",
    "missing_days_in_patches",
]


# ============================================================
# JSON SERIALIZATION SAFETY
# ============================================================

def json_safe(value):
    """
    Recursively convert NumPy/Pandas scalar/container types into
    native Python JSON-serializable types.

    This prevents end-of-run failures such as:
        TypeError: Object of type bool is not JSON serializable

    It is intentionally applied to the FINAL report only; the
    scientific calculations remain unchanged.
    """
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def validate_json_serializable(payload):
    """
    Fail BEFORE writing the final JSON if anything is still
    not serializable. This is a cheap final guardrail.
    """
    try:
        json.dumps(payload, indent=2)
    except TypeError as exc:
        raise TypeError(
            "FINAL REPORT IS NOT JSON SERIALIZABLE. "
            "The audit results are preserved in memory, but the "
            "script will stop here rather than silently writing "
            "an incomplete report."
        ) from exc


# ============================================================
# HELPERS
# ============================================================

def empty_df(columns):
    """
    Always return a dataframe with the expected columns.
    Prevents KeyError when a category is empty.
    """
    return pd.DataFrame(
        columns=columns
    )


def extract_date(path):
    """
    Extract YYYY-MM-DD from filename.
    """

    match = re.search(
        r"(20\d{2}-\d{2}-\d{2})",
        path.name
    )

    if not match:
        return None

    try:
        return pd.Timestamp(
            match.group(1)
        ).normalize()

    except Exception:
        return None


def find_sss_variable(ds):
    """
    Identify SSS variable robustly.
    """

    preferred = [
        "sss",
        "sos",
        "sea_surface_salinity",
        "salinity",
    ]

    lower_map = {
        name.lower(): name
        for name in ds.data_vars
    }

    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    candidates = []

    for name, da in ds.data_vars.items():

        text = " ".join([
            str(name),
            str(
                da.attrs.get(
                    "long_name",
                    ""
                )
            ),
            str(
                da.attrs.get(
                    "standard_name",
                    ""
                )
            ),
            str(
                da.attrs.get(
                    "description",
                    ""
                )
            ),
        ]).lower()

        if "salinity" in text:
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    return None


def find_lat_lon(ds):

    lat_names = {
        "lat",
        "latitude",
        "nav_lat",
    }

    lon_names = {
        "lon",
        "longitude",
        "nav_lon",
    }

    lat_name = None
    lon_name = None

    for name in ds.variables:

        lower = name.lower()

        if lower in lat_names:
            lat_name = name

        if lower in lon_names:
            lon_name = name

    return lat_name, lon_name


def normalize_sss_array(da):
    """
    Reduce singleton dimensions.

    Expected final form:
        (lat, lon)

    This intentionally fails loudly if the dataset does not
    reduce to a 2-D spatial field.
    """

    arr = np.asarray(
        da.values
    )

    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"SSS array did not reduce to 2-D. "
            f"Original shape={da.shape}, "
            f"squeezed shape={arr.shape}"
        )

    return arr


# ============================================================
# DISCOVER FILES
# ============================================================

print("=" * 80)
print("OISSS SPECIAL FORENSIC GAP AUDIT")
print("=" * 80)

print(f"Run ID:      {RUN_ID}")
print(f"Raw root:    {RAW_ROOT}")
print(f"Output:      {OUTPUT_DIR}")
print(
    f"Period:      {START_DATE.date()} -> "
    f"{END_DATE.date()}"
)
print(
    f"Domain:      {LAT_MIN}-{LAT_MAX} N, "
    f"{LON_MIN}-{LON_MAX} E"
)
print(
    f"Interpolation threshold: "
    f"{MAX_INTERPOLATION_GAP_DAYS} day(s)"
)
print("Raw data modified: NO")
print()

all_files = sorted(
    RAW_ROOT.rglob("*.nc")
)

dated_files = []

for path in all_files:

    date = extract_date(path)

    if date is None:
        continue

    if (
        START_DATE
        <= date
        <= END_DATE
    ):

        dated_files.append(
            (date, path)
        )


dated_files.sort(
    key=lambda x: x[0]
)

print(
    f"NetCDF files discovered: "
    f"{len(all_files):,}"
)

print(
    f"Dated files in requested period: "
    f"{len(dated_files):,}"
)


# ============================================================
# VALIDATE CALENDAR BEFORE SCIENTIFIC PASS
# ============================================================

date_to_files = {}

for date, path in dated_files:

    key = date.strftime(
        "%Y-%m-%d"
    )

    date_to_files.setdefault(
        key,
        []
    ).append(path)


expected_keys = {
    d.strftime("%Y-%m-%d")
    for d in EXPECTED_DAYS
}

observed_keys = set(
    date_to_files.keys()
)

missing_dates = sorted(
    expected_keys - observed_keys
)

unexpected_dates = sorted(
    observed_keys - expected_keys
)

duplicate_dates = {
    date: paths
    for date, paths in date_to_files.items()
    if len(paths) > 1
}


if missing_dates:
    raise RuntimeError(
        f"Calendar validation failed BEFORE scientific gap analysis: "
        f"{len(missing_dates)} expected dates are missing."
    )

if unexpected_dates:
    raise RuntimeError(
        f"Calendar validation failed BEFORE scientific gap analysis: "
        f"{len(unexpected_dates)} unexpected dates were detected."
    )

if duplicate_dates:
    raise RuntimeError(
        f"Calendar validation failed BEFORE scientific gap analysis: "
        f"{len(duplicate_dates)} dates have multiple NetCDF files. "
        f"Temporal patch classification requires exactly one file per day."
    )

if len(dated_files) != EXPECTED_DAY_COUNT:
    raise RuntimeError(
        "Calendar validation failed BEFORE scientific gap analysis: "
        f"expected exactly {EXPECTED_DAY_COUNT:,} dated files, "
        f"found {len(dated_files):,}."
    )

print("Calendar validation: PASS")


# ============================================================
# REFERENCE GRID
# ============================================================

reference_lat = None
reference_lon = None

lat_name_reference = None
lon_name_reference = None

reference_shape = None

sss_variable_name = None


# ============================================================
# AVAILABILITY ARRAYS
# ============================================================

valid_count = None
missing_count = None


# ============================================================
# DAILY MASK STORAGE
#
# We need the daily mask to derive exact contiguous
# missing runs.
#
# Shape:
#   (time, lat, lon)
#
# 4749 x 100 x 240 ~= 114 million booleans
# ~= 114 MB.
#
# Acceptable for a machine with sufficient RAM.
# ============================================================

daily_valid_masks = []

daily_dates = []


# ============================================================
# ERRORS
# ============================================================

errors = []


# ============================================================
# DAILY PASS
# ============================================================

for idx, (date, path) in enumerate(
    dated_files,
    start=1
):

    if idx == 1 or idx % 100 == 0:

        print(
            f"[{idx:,}/{len(dated_files):,}] "
            f"{date.date()}"
        )

        pd.DataFrame({
            "run_id": [RUN_ID],
            "processed_files": [idx],
            "total_files": [
                len(dated_files)
            ],
            "current_date": [
                str(date.date())
            ],
            "timestamp_utc": [
                datetime.now(
                    timezone.utc
                ).isoformat()
            ],
        }).to_csv(
            PROGRESS_CSV,
            index=False
        )

    try:

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )

            ds = xr.open_dataset(
                path,
                decode_times=True,
                mask_and_scale=True,
            )

        sss_name = find_sss_variable(
            ds
        )

        if sss_name is None:
            raise ValueError(
                "Could not identify SSS variable"
            )

        if sss_variable_name is None:
            sss_variable_name = sss_name

        elif sss_name != sss_variable_name:
            raise ValueError(
                "SSS variable changed from "
                f"{sss_variable_name} to {sss_name}"
            )

        lat_name, lon_name = find_lat_lon(
            ds
        )

        if lat_name is None or lon_name is None:
            raise ValueError(
                "Latitude/longitude variable "
                "could not be identified."
            )

        lat = np.asarray(
            ds[lat_name].values
        ).squeeze()

        lon = np.asarray(
            ds[lon_name].values
        ).squeeze()

        if reference_lat is None:

            reference_lat = lat.copy()
            reference_lon = lon.copy()

            lat_name_reference = lat_name
            lon_name_reference = lon_name

        else:

            if not np.array_equal(
                reference_lat,
                lat
            ):
                raise ValueError(
                    "Latitude grid changed."
                )

            if not np.array_equal(
                reference_lon,
                lon
            ):
                raise ValueError(
                    "Longitude grid changed."
                )

        values = normalize_sss_array(
            ds[sss_name]
        )

        if reference_shape is None:

            reference_shape = (
                values.shape
            )

            valid_count = np.zeros(
                reference_shape,
                dtype=np.int32
            )

            missing_count = np.zeros(
                reference_shape,
                dtype=np.int32
            )

        elif values.shape != reference_shape:

            raise ValueError(
                "SSS spatial shape changed: "
                f"{values.shape} vs "
                f"{reference_shape}"
            )

        valid = np.isfinite(
            values
        )

        valid_count += (
            valid.astype(
                np.int32
            )
        )

        missing_count += (
            (~valid).astype(
                np.int32
            )
        )

        daily_valid_masks.append(
            valid
        )

        daily_dates.append(
            date
        )

        ds.close()

    except Exception as exc:

        errors.append({
            "date": str(date.date()),
            "path": str(path),
            "error": repr(exc),
        })

        print(
            f"    ERROR: {exc}"
        )

        # We cannot silently invent a mask for a failed file.
        # Stop because the temporal gap analysis would otherwise
        # become scientifically ambiguous.
        raise


# ============================================================
# CONVERT TO 3-D BOOLEAN ARRAY
# ============================================================

daily_valid_masks = np.stack(
    daily_valid_masks,
    axis=0
)

daily_dates = pd.DatetimeIndex(
    daily_dates
)

print()
print(
    "Daily mask assembled:"
)
print(
    f"  shape = {daily_valid_masks.shape}"
)


# ============================================================
# GRID CONSISTENCY
# ============================================================

if (
    reference_lat is None
    or reference_lon is None
):
    raise RuntimeError(
        "No valid reference grid."
    )

lat_count = len(
    reference_lat
)

lon_count = len(
    reference_lon
)

print(
    f"  latitude cells  = {lat_count}"
)

print(
    f"  longitude cells = {lon_count}"
)


# ============================================================
# NIO DOMAIN CHECK
# ============================================================

lat_ok = (
    np.nanmin(reference_lat)
    >= LAT_MIN
    and
    np.nanmax(reference_lat)
    <= LAT_MAX
)

lon_ok = (
    np.nanmin(reference_lon)
    >= LON_MIN
    and
    np.nanmax(reference_lon)
    <= LON_MAX
)


# ============================================================
# CELL-LEVEL SUMMARY
# ============================================================

cell_summary_records = []

for i in range(lat_count):

    print(
        f"Analyzing latitude "
        f"{i + 1}/{lat_count}"
    )

    for j in range(lon_count):

        series = (
            daily_valid_masks[:, i, j]
        )

        total = len(series)

        valid_days = int(
            series.sum()
        )

        missing_days = (
            total - valid_days
        )

        availability = (
            valid_days / total
            if total
            else np.nan
        )

        missing_fraction = (
            missing_days / total
            if total
            else np.nan
        )

        # ----------------------------------------------------
        # Completely valid
        # ----------------------------------------------------

        if missing_days == 0:

            cell_summary_records.append({
                "lat_index": i,
                "lon_index": j,
                "latitude": float(
                    reference_lat[i]
                ),
                "longitude": float(
                    reference_lon[j]
                ),
                "total_days": total,
                "valid_days": valid_days,
                "missing_days": 0,
                "availability_fraction": availability,
                "missing_fraction": 0.0,
                "first_missing_date": None,
                "last_missing_date": None,
                "longest_missing_streak": 0,
                "number_of_missing_runs": 0,
                "has_internal_gap": False,
                "has_edge_gap": False,
                "action": "KEEP_OISSS",
            })

            continue

        # ----------------------------------------------------
        # Permanently missing
        # ----------------------------------------------------

        if valid_days == 0:

            cell_summary_records.append({
                "lat_index": i,
                "lon_index": j,
                "latitude": float(
                    reference_lat[i]
                ),
                "longitude": float(
                    reference_lon[j]
                ),
                "total_days": total,
                "valid_days": 0,
                "missing_days": missing_days,
                "availability_fraction": 0.0,
                "missing_fraction": 1.0,
                "first_missing_date": str(
                    daily_dates[0].date()
                ),
                "last_missing_date": str(
                    daily_dates[-1].date()
                ),
                "longest_missing_streak": total,
                "number_of_missing_runs": 1,
                "has_internal_gap": False,
                "has_edge_gap": True,
                "action": "KEEP_MASKED",
            })

            continue

        # ----------------------------------------------------
        # Missing runs
        # ----------------------------------------------------

        missing = ~series

        padded = np.concatenate(
            [
                [False],
                missing,
                [False],
            ]
        )

        starts = np.where(
            padded[1:]
            &
            ~padded[:-1]
        )[0]

        ends = np.where(
            ~padded[1:]
            &
            padded[:-1]
        )[0] - 1

        run_lengths = (
            ends - starts + 1
        )

        longest_run = int(
            run_lengths.max()
        )

        first_missing = daily_dates[
            starts[0]
        ]

        last_missing = daily_dates[
            ends[-1]
        ]

        has_internal_gap = False
        has_edge_gap = False

        for s, e in zip(
            starts,
            ends
        ):

            if s == 0 or e == total - 1:
                has_edge_gap = True

            else:
                has_internal_gap = True

        # ----------------------------------------------------
        # Cell-level action
        #
        # If ALL gaps are short internal gaps:
        #   CONTROLLED_INTERPOLATION
        #
        # Otherwise:
        #   INDEPENDENT_SSS_FALLBACK
        # ----------------------------------------------------

        all_runs_short_internal = True

        for s, e, length in zip(
            starts,
            ends,
            run_lengths
        ):

            is_edge = (
                s == 0
                or
                e == total - 1
            )

            if (
                is_edge
                or
                length
                > MAX_INTERPOLATION_GAP_DAYS
            ):

                all_runs_short_internal = False
                break

        if all_runs_short_internal:

            action = (
                "CONTROLLED_INTERPOLATION"
            )

        else:

            action = (
                "INDEPENDENT_SSS_FALLBACK"
            )

        cell_summary_records.append({
            "lat_index": i,
            "lon_index": j,
            "latitude": float(
                reference_lat[i]
            ),
            "longitude": float(
                reference_lon[j]
            ),
            "total_days": total,
            "valid_days": valid_days,
            "missing_days": missing_days,
            "availability_fraction": availability,
            "missing_fraction": missing_fraction,
            "first_missing_date": str(
                first_missing.date()
            ),
            "last_missing_date": str(
                last_missing.date()
            ),
            "longest_missing_streak": longest_run,
            "number_of_missing_runs": len(
                starts
            ),
            "has_internal_gap": has_internal_gap,
            "has_edge_gap": has_edge_gap,
            "action": action,
        })


# ============================================================
# CELL DATAFRAME
# ============================================================

cell_df = pd.DataFrame(
    cell_summary_records,
    columns=CELL_COLUMNS
)

cell_df.to_csv(
    CELL_SUMMARY_CSV,
    index=False
)


# ============================================================
# CREATE EXACT GAP PATCHES
# ============================================================

patch_records = []

patch_counter = 0


for _, cell in cell_df.iterrows():

    action = cell["action"]

    if action == "KEEP_OISSS":
        continue

    if action == "KEEP_MASKED":
        continue

    i = int(
        cell["lat_index"]
    )

    j = int(
        cell["lon_index"]
    )

    series = (
        daily_valid_masks[:, i, j]
    )

    missing = ~series

    padded = np.concatenate(
        [
            [False],
            missing,
            [False],
        ]
    )

    starts = np.where(
        padded[1:]
        &
        ~padded[:-1]
    )[0]

    ends = np.where(
        ~padded[1:]
        &
        padded[:-1]
    )[0] - 1

    for s, e in zip(
        starts,
        ends
    ):

        gap_days = int(
            e - s + 1
        )

        is_edge = (
            s == 0
            or
            e == len(series) - 1
        )

        before_index = (
            s - 1
            if s > 0
            else None
        )

        after_index = (
            e + 1
            if e < len(series) - 1
            else None
        )

        before_valid = (
            before_index is not None
            and
            bool(
                series[before_index]
            )
        )

        after_valid = (
            after_index is not None
            and
            bool(
                series[after_index]
            )
        )

        # ----------------------------------------------------
        # EXACT PATCH DECISION
        # ----------------------------------------------------

        if (
            not is_edge
            and
            gap_days
            <= MAX_INTERPOLATION_GAP_DAYS
            and
            before_valid
            and
            after_valid
        ):

            patch_action = (
                "CONTROLLED_INTERPOLATION"
            )

            reason = (
                "Internal missing run within "
                f"{MAX_INTERPOLATION_GAP_DAYS} days "
                "with valid OISSS values on both sides."
            )

        else:

            patch_action = (
                "INDEPENDENT_SSS_FALLBACK"
            )

            if is_edge:

                reason = (
                    "Edge gap cannot be safely "
                    "bracketed by OISSS values."
                )

            else:

                reason = (
                    "Missing run exceeds the controlled "
                    "interpolation threshold."
                )

        patch_counter += 1

        patch_records.append({
            "patch_id": (
                f"OISSS-PATCH-{patch_counter:07d}"
            ),
            "action": patch_action,
            "lat_index": i,
            "lon_index": j,
            "latitude": float(
                cell["latitude"]
            ),
            "longitude": float(
                cell["longitude"]
            ),
            "start_date": str(
                daily_dates[s].date()
            ),
            "end_date": str(
                daily_dates[e].date()
            ),
            "gap_days": gap_days,
            "before_date": (
                str(
                    daily_dates[
                        before_index
                    ].date()
                )
                if before_index is not None
                else None
            ),
            "after_date": (
                str(
                    daily_dates[
                        after_index
                    ].date()
                )
                if after_index is not None
                else None
            ),
            "before_valid": before_valid,
            "after_valid": after_valid,
            "is_edge_gap": is_edge,
            "total_missing_days_cell": int(
                cell["missing_days"]
            ),
            "cell_availability_fraction": float(
                cell[
                    "availability_fraction"
                ]
            ),
            "reason": reason,
        })


# ============================================================
# PATCH DATAFRAME
# ============================================================

patch_df = pd.DataFrame(
    patch_records,
    columns=PATCH_COLUMNS
)

patch_df.to_csv(
    PATCHES_CSV,
    index=False
)


# ============================================================
# THREE-WAY OUTPUTS
# ============================================================

interpolation_df = patch_df[
    patch_df["action"]
    ==
    "CONTROLLED_INTERPOLATION"
].copy()

fallback_df = patch_df[
    patch_df["action"]
    ==
    "INDEPENDENT_SSS_FALLBACK"
].copy()


# ------------------------------------------------------------
# IMPORTANT:
# These explicit column selections are now SAFE even when
# there are zero rows.
# ------------------------------------------------------------

interpolation_df = (
    interpolation_df.reindex(
        columns=PATCH_COLUMNS
    )
)

fallback_df = (
    fallback_df.reindex(
        columns=PATCH_COLUMNS
    )
)


interpolation_df.to_csv(
    INTERPOLATION_CSV,
    index=False
)

fallback_df.to_csv(
    FALLBACK_CSV,
    index=False
)


# ============================================================
# PERMANENT MASK
# ============================================================

masked_df = cell_df[
    cell_df["action"]
    ==
    "KEEP_MASKED"
].copy()

# Explicit schema.
#
# THIS DIRECTLY FIXES THE ERROR YOU ENCOUNTERED.
#
# first_missing_date and last_missing_date are guaranteed
# to exist because cell_df was created with CELL_COLUMNS.
# ============================================================

masked_df = masked_df.reindex(
    columns=CELL_COLUMNS
)

masked_df.to_csv(
    MASKED_CSV,
    index=False
)


# ============================================================
# YEAR-LEVEL PATCH SUMMARY
# ============================================================

year_records = []

if not patch_df.empty:

    patch_df["_start_date"] = pd.to_datetime(
        patch_df["start_date"]
    )

    patch_df["_year"] = (
        patch_df["_start_date"]
        .dt.year
    )

    for year in range(
        START_DATE.year,
        END_DATE.year + 1
    ):

        subset = patch_df[
            patch_df["_year"] == year
        ]

        interpolation_count = int(
            (
                subset["action"]
                ==
                "CONTROLLED_INTERPOLATION"
            ).sum()
        )

        fallback_count = int(
            (
                subset["action"]
                ==
                "INDEPENDENT_SSS_FALLBACK"
            ).sum()
        )

        year_records.append({
            "year": year,
            "missing_cells_with_gap": int(
                len(subset)
            ),
            "interpolation_patches": (
                interpolation_count
            ),
            "fallback_patches": (
                fallback_count
            ),
            "missing_days_in_patches": int(
                subset["gap_days"]
                .sum()
            ),
        })

else:

    for year in range(
        START_DATE.year,
        END_DATE.year + 1
    ):

        year_records.append({
            "year": year,
            "missing_cells_with_gap": 0,
            "interpolation_patches": 0,
            "fallback_patches": 0,
            "missing_days_in_patches": 0,
        })


year_df = pd.DataFrame(
    year_records,
    columns=YEAR_COLUMNS
)

year_df.to_csv(
    YEAR_SUMMARY_CSV,
    index=False
)


# ============================================================
# REMOVE INTERNAL TEMP COLUMNS
# ============================================================

if "_start_date" in patch_df.columns:

    patch_df = patch_df.drop(
        columns=[
            "_start_date",
            "_year",
        ]
    )

    patch_df.to_csv(
        PATCHES_CSV,
        index=False
    )


# ============================================================
# FINAL COUNTS
# ============================================================

permanent_mask_count = int(
    (
        cell_df["action"]
        ==
        "KEEP_MASKED"
    ).sum()
)

keep_ois_count = int(
    (
        cell_df["action"]
        ==
        "KEEP_OISSS"
    ).sum()
)

interpolation_patch_count = len(
    interpolation_df
)

fallback_patch_count = len(
    fallback_df
)

total_patch_count = len(
    patch_df
)

total_patch_days = int(
    patch_df["gap_days"].sum()
) if not patch_df.empty else 0


# ============================================================
# GAP LENGTH DISTRIBUTION
# ============================================================

if not patch_df.empty:

    gap_distribution = (
        patch_df[
            "gap_days"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

else:

    gap_distribution = {}


# ============================================================
# CELL ACTION DISTRIBUTION
# ============================================================

cell_action_counts = (
    cell_df[
        "action"
    ]
    .value_counts()
    .to_dict()
)


# ============================================================
# FINAL TIME
# ============================================================

RUN_END_UTC = datetime.now(
    timezone.utc
)

RUN_END_IST = RUN_END_UTC.astimezone(
    timezone(timedelta(hours=5, minutes=30))
)

duration_seconds = (
    RUN_END_UTC
    - RUN_START_UTC
).total_seconds()


# ============================================================
# FINAL DECISION
# ============================================================

if errors:

    status = "RED"

    decision_reason = (
        "One or more OISSS files failed during the "
        "forensic temporal scan."
    )

elif (
    len(missing_dates) > 0
    or
    len(duplicate_dates) > 0
):

    status = "YELLOW"

    decision_reason = (
        "The forensic scan completed, but calendar "
        "irregularities require investigation."
    )

else:

    status = "YELLOW"

    decision_reason = (
        "Forensic gap classification completed. "
        "Raw OISSS remains unchanged. "
        "Three-way treatment patches are identified, "
        "but no interpolation or external SSS replacement "
        "has been performed."
    )


# ============================================================
# JSON REPORT
# ============================================================

report = {

    "run": {
        "run_id": RUN_ID,
        "start_utc": RUN_START_UTC.isoformat(),
        "end_utc": RUN_END_UTC.isoformat(),
        "start_ist": RUN_START_IST.isoformat(),
        "end_ist": RUN_END_IST.isoformat(),
        "duration_seconds": duration_seconds,
    },

    "configuration": {
        "raw_root": str(
            RAW_ROOT
        ),
        "output_dir": str(
            OUTPUT_DIR
        ),
        "start_date": str(
            START_DATE.date()
        ),
        "end_date": str(
            END_DATE.date()
        ),
        "expected_days": EXPECTED_DAY_COUNT,
        "latitude_min": LAT_MIN,
        "latitude_max": LAT_MAX,
        "longitude_min": LON_MIN,
        "longitude_max": LON_MAX,
        "max_interpolation_gap_days": (
            MAX_INTERPOLATION_GAP_DAYS
        ),
        "raw_data_modified": False,
    },

    "calendar": {
        "files_discovered": len(
            all_files
        ),
        "dated_files": len(
            dated_files
        ),
        "expected_days": EXPECTED_DAY_COUNT,
        "observed_dates": len(
            observed_keys
        ),
        "missing_dates": missing_dates,
        "unexpected_dates": unexpected_dates,
        "duplicate_dates": {
            k: [
                str(p)
                for p in v
            ]
            for k, v in duplicate_dates.items()
        },
    },

    "grid": {
        "latitude_count": lat_count,
        "longitude_count": lon_count,
        "shape": list(
            reference_shape
        ),
        "latitude_min": float(
            np.nanmin(
                reference_lat
            )
        ),
        "latitude_max": float(
            np.nanmax(
                reference_lat
            )
        ),
        "longitude_min": float(
            np.nanmin(
                reference_lon
            )
        ),
        "longitude_max": float(
            np.nanmax(
                reference_lon
            )
        ),
        "domain_ok": bool(
            lat_ok
            and
            lon_ok
        ),
        "sss_variable": (
            sss_variable_name
        ),
    },

    "cells": {
        "total_grid_cells": int(
            len(cell_df)
        ),
        "keep_oisss": keep_ois_count,
        "keep_masked": permanent_mask_count,
        "cells_with_gaps": int(
            (
                cell_df["action"]
                !=
                "KEEP_OISSS"
            ).sum()
        ),
        "action_counts": {
            str(k): int(v)
            for k, v in
            cell_action_counts.items()
        },
    },

    "patches": {
        "total_patches": total_patch_count,
        "controlled_interpolation": (
            interpolation_patch_count
        ),
        "independent_sss_fallback": (
            fallback_patch_count
        ),
        "total_missing_days_in_patches": (
            total_patch_days
        ),
        "gap_length_distribution": {
            str(k): int(v)
            for k, v in
            gap_distribution.items()
        },
    },

    "three_way_policy": {

        "CONTROLLED_INTERPOLATION": {
            "condition": (
                f"Internal gap <= "
                f"{MAX_INTERPOLATION_GAP_DAYS} days "
                "with valid OISSS immediately before "
                "and after the gap."
            ),
            "values_generated": False,
        },

        "INDEPENDENT_SSS_FALLBACK": {
            "condition": (
                "Longer internal gaps or edge gaps "
                "that cannot be safely bracketed."
            ),
            "values_generated": False,
        },

        "KEEP_MASKED": {
            "condition": (
                "Cells missing for the entire archive."
            ),
            "values_generated": False,
        },
    },

    "decision": {
        "status": status,
        "reason": decision_reason,
    },

    "errors": errors,
}


# ------------------------------------------------------------
# FINAL JSON SAFETY GATE
# ------------------------------------------------------------
# Convert every NumPy/Pandas scalar/container recursively and
# validate serialization BEFORE opening the destination file.
#
# This guarantees that an end-of-run NumPy scalar cannot destroy
# the completed audit after 100% of the scientific analysis has
# already finished.
# ------------------------------------------------------------

report = json_safe(report)
validate_json_serializable(report)

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        allow_nan=False,
    )


# ============================================================
# HUMAN REPORT
# ============================================================

lines = []

lines.append(
    "OISSS SPECIAL FORENSIC GAP AUDIT"
)

lines.append(
    "=" * 80
)

lines.append("")

lines.append(
    f"Run ID: {RUN_ID}"
)

lines.append(
    f"Start UTC: {RUN_START_UTC.isoformat()}"
)

lines.append(
    f"End UTC:   {RUN_END_UTC.isoformat()}"
)

lines.append(
    f"Start IST: {RUN_START_IST.isoformat()}"
)

lines.append(
    f"End IST:   {RUN_END_IST.isoformat()}"
)

lines.append(
    f"Duration:  {duration_seconds:.2f} seconds"
)

lines.append("")

lines.append(
    f"Raw root: {RAW_ROOT}"
)

lines.append(
    f"Output:   {OUTPUT_DIR}"
)

lines.append(
    "Raw data modified: NO"
)

lines.append("")

lines.append(
    "PERIOD / DOMAIN"
)

lines.append(
    "-" * 80
)

lines.append(
    f"{START_DATE.date()} -> "
    f"{END_DATE.date()}"
)

lines.append(
    f"{LAT_MIN} -> {LAT_MAX} N"
)

lines.append(
    f"{LON_MIN} -> {LON_MAX} E"
)

lines.append("")

lines.append(
    "CALENDAR"
)

lines.append(
    "-" * 80
)

lines.append(
    f"Expected days: {EXPECTED_DAY_COUNT:,}"
)

lines.append(
    f"Observed dates: {len(observed_keys):,}"
)

lines.append(
    f"Missing dates: {len(missing_dates):,}"
)

lines.append(
    f"Duplicate dates: {len(duplicate_dates):,}"
)

lines.append("")

lines.append(
    "GRID"
)

lines.append(
    "-" * 80
)

lines.append(
    f"Shape: {reference_shape}"
)

lines.append(
    f"Latitude: "
    f"{np.nanmin(reference_lat):.6f} -> "
    f"{np.nanmax(reference_lat):.6f}"
)

lines.append(
    f"Longitude: "
    f"{np.nanmin(reference_lon):.6f} -> "
    f"{np.nanmax(reference_lon):.6f}"
)

lines.append(
    f"SSS variable: {sss_variable_name}"
)

lines.append(
    f"Domain OK: {bool(lat_ok and lon_ok)}"
)

lines.append("")

lines.append(
    "THREE-WAY CELL CLASSIFICATION"
)

lines.append(
    "-" * 80
)

lines.append(
    f"KEEP OISSS: "
    f"{keep_ois_count:,}"
)

lines.append(
    f"KEEP MASKED: "
    f"{permanent_mask_count:,}"
)

lines.append(
    f"Cells requiring treatment: "
    f"{(
        len(cell_df)
        - keep_ois_count
        - permanent_mask_count
    ):,}"
)

lines.append("")

lines.append(
    "EXACT GAP PATCHES"
)

lines.append(
    "-" * 80
)

lines.append(
    f"Total patches: "
    f"{total_patch_count:,}"
)

lines.append(
    f"Controlled interpolation: "
    f"{interpolation_patch_count:,}"
)

lines.append(
    f"Independent SSS fallback: "
    f"{fallback_patch_count:,}"
)

lines.append(
    f"Total missing days represented: "
    f"{total_patch_days:,}"
)

lines.append("")

lines.append(
    "GAP LENGTH DISTRIBUTION"
)

lines.append(
    "-" * 80
)

for length, count in sorted(
    gap_distribution.items(),
    key=lambda x: int(x[0])
):

    lines.append(
        f"{length} day(s): "
        f"{count:,} patch(es)"
    )

lines.append("")

lines.append(
    "POLICY"
)

lines.append(
    "-" * 80
)

lines.append(
    f"Internal gaps <= "
    f"{MAX_INTERPOLATION_GAP_DAYS} days "
    f"with valid values on both sides: "
    "CONTROLLED_INTERPOLATION"
)

lines.append(
    "Longer gaps: "
    "INDEPENDENT_SSS_FALLBACK"
)

lines.append(
    "Edge gaps: "
    "INDEPENDENT_SSS_FALLBACK"
)

lines.append(
    "Permanent missing cells: "
    "KEEP_MASKED"
)

lines.append("")

lines.append(
    "IMPORTANT"
)

lines.append(
    "-" * 80
)

lines.append(
    "No OISSS values were generated."
)

lines.append(
    "No interpolation was performed."
)

lines.append(
    "No independent SSS fallback values were inserted."
)

lines.append(
    "No raw OISSS file was modified."
)

lines.append("")

lines.append(
    f"FINAL STATUS: {status}"
)

lines.append(
    f"REASON: {decision_reason}"
)

lines.append("")

lines.append(
    "OUTPUT FILES"
)

lines.append(
    "-" * 80
)

for p in [
    REPORT_TXT,
    REPORT_JSON,
    PATCHES_CSV,
    INTERPOLATION_CSV,
    FALLBACK_CSV,
    MASKED_CSV,
    CELL_SUMMARY_CSV,
    YEAR_SUMMARY_CSV,
    PROGRESS_CSV,
]:

    lines.append(
        str(p)
    )

with open(
    REPORT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(lines)
    )


# ============================================================
# ERROR FILE
# ============================================================

error_df = pd.DataFrame(
    errors,
    columns=[
        "date",
        "path",
        "error",
    ]
)

error_df.to_csv(
    ERRORS_CSV,
    index=False
)


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 80)
print("OISSS SPECIAL FORENSIC AUDIT COMPLETE")
print("=" * 80)

print(
    f"Run ID: {RUN_ID}"
)

print(
    f"Files processed: "
    f"{len(dated_files):,}"
)

print(
    f"Grid: "
    f"{reference_shape}"
)

print()

print(
    "THREE-WAY RESULT"
)

print(
    f"  KEEP OISSS: "
    f"{keep_ois_count:,} cells"
)

print(
    f"  KEEP MASKED: "
    f"{permanent_mask_count:,} cells"
)

print(
    f"  INTERPOLATION PATCHES: "
    f"{interpolation_patch_count:,}"
)

print(
    f"  FALLBACK PATCHES: "
    f"{fallback_patch_count:,}"
)

print(
    f"  TOTAL PATCHES: "
    f"{total_patch_count:,}"
)

print()

print(
    f"Final status: {status}"
)

print()

print(
    "KEY OUTPUTS:"
)

print(
    f"  {PATCHES_CSV}"
)

print(
    f"  {INTERPOLATION_CSV}"
)

print(
    f"  {FALLBACK_CSV}"
)

print(
    f"  {MASKED_CSV}"
)

print(
    f"  {CELL_SUMMARY_CSV}"
)

print(
    f"  {REPORT_TXT}"
)

print(
    f"  {REPORT_JSON}"
)

print("=" * 80)