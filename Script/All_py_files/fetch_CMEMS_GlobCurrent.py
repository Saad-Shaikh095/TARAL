# All year acquisition code:
# ==============================================================================
# CMEMS GLOBCURRENT — 2012–2024 YEARLY ACQUISITION
# ==============================================================================
#
# ARCHITECTURE
# ------------------------------------------------------------------------------
# One CMEMS server-side request per year.
#
# 2012 → one NetCDF → E:\...\raw\2012\
# 2013 → one NetCDF → E:\...\raw\2013\
# ...
# 2024 → one NetCDF → E:\...\raw\2024\
#
# Each file contains ALL daily records for that year.
#
# CONFIGURATION
# ------------------------------------------------------------------------------
# Product:
#   MULTIOBS_GLO_PHY_MYNRT_015_003
#
# Dataset:
#   cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m
#
# Variables:
#   uo
#   vo
#
# Depth:
#   0 m ONLY
#
# BBOX:
#   45–105°E
#   5–30°N
#
# Grid:
#   0.25° × 0.25°
#
# Period:
#   2012-01-01 → 2024-12-31
#
# IMPORTANT
# ------------------------------------------------------------------------------
# - Folder structure is NOT created or modified.
# - Existing valid yearly files are skipped.
# - Existing invalid yearly files can be replaced.
# - Years are acquired SEQUENTIALLY.
# - No 15 m data is requested.
# - force_download is NOT used because it is deprecated.
#
# ==============================================================================


from pathlib import Path
import calendar
import sys
import time

import numpy as np
import xarray as xr
import copernicusmarine


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

START_YEAR = 2012
END_YEAR = 2024

# ------------------------------------------------------------------------------
# CMEMS PRODUCT
# ------------------------------------------------------------------------------

PRODUCT_ID = "MULTIOBS_GLO_PHY_MYNRT_015_003"

DATASET_ID = (
    "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"
)

# ------------------------------------------------------------------------------
# VARIABLES — SURFACE U/V ONLY
# ------------------------------------------------------------------------------

VARIABLES = [
    "uo",
    "vo",
]

DEPTH = 0.0

# ------------------------------------------------------------------------------
# PROJECT BBOX
# ------------------------------------------------------------------------------

LON_MIN = 45.0
LON_MAX = 105.0

LAT_MIN = 5.0
LAT_MAX = 30.0

# ------------------------------------------------------------------------------
# EXISTING FOLDER STRUCTURE
#
# The script expects:
#
# E:\Ocean_data\CMEMS_GlobCurrent\data\raw\2012
# E:\Ocean_data\CMEMS_GlobCurrent\data\raw\2013
# ...
# E:\Ocean_data\CMEMS_GlobCurrent\data\raw\2024
#
# It does NOT create these folders.
# ------------------------------------------------------------------------------

OUTPUT_ROOT = Path(
    r"E:\Ocean_data\CMEMS_GlobCurrent\data\raw"
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def format_bytes(size_bytes):
    """Convert bytes to a readable size."""

    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"

    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"

    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"

    return f"{size_bytes / (1024 ** 3):.3f} GB"


def format_duration(seconds):
    """Convert seconds to HH:MM:SS."""

    seconds = int(seconds)

    hours, remainder = divmod(
        seconds,
        3600
    )

    minutes, seconds = divmod(
        remainder,
        60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def expected_days(year):
    """Return 365 or 366 depending on leap year."""

    return (
        366
        if calendar.isleap(year)
        else 365
    )


# ==============================================================================
# YEAR FILE VALIDATION
# ==============================================================================

def validate_existing_file(
    filepath,
    expected_day_count,
):
    """
    Perform lightweight structural validation.

    This is intentionally NOT a full scientific audit.
    """

    try:

        with xr.open_dataset(
            filepath,
            decode_cf=True,
        ) as ds:

            # ------------------------------------------------------------------
            # Variables
            # ------------------------------------------------------------------

            for variable in VARIABLES:

                if variable not in ds.data_vars:

                    return False, (
                        f"missing variable '{variable}'"
                    )

            # ------------------------------------------------------------------
            # Coordinates
            # ------------------------------------------------------------------

            required_coords = [
                "time",
                "depth",
                "latitude",
                "longitude",
            ]

            for coord in required_coords:

                if coord not in ds.coords:

                    return False, (
                        f"missing coordinate '{coord}'"
                    )

            # ------------------------------------------------------------------
            # Time
            # ------------------------------------------------------------------

            actual_days = len(
                ds["time"]
            )

            if actual_days != expected_day_count:

                return False, (
                    f"expected {expected_day_count} "
                    f"daily records, found {actual_days}"
                )

            # ------------------------------------------------------------------
            # Depth
            # ------------------------------------------------------------------

            depth_values = np.asarray(
                ds["depth"].values
            )

            if depth_values.size != 1:

                return False, (
                    f"expected one depth level, "
                    f"found {depth_values.size}"
                )

            actual_depth = float(
                depth_values.reshape(-1)[0]
            )

            if not np.isclose(
                actual_depth,
                DEPTH,
                atol=1e-6,
            ):

                return False, (
                    f"expected depth 0 m, "
                    f"found {actual_depth} m"
                )

            # ------------------------------------------------------------------
            # Dimensions
            # ------------------------------------------------------------------

            expected_dims = (
                "time",
                "depth",
                "latitude",
                "longitude",
            )

            for variable in VARIABLES:

                if ds[variable].dims != expected_dims:

                    return False, (
                        f"{variable} dimensions are "
                        f"{ds[variable].dims}"
                    )

            # ------------------------------------------------------------------
            # Spatial resolution
            # ------------------------------------------------------------------

            longitude = ds["longitude"].values
            latitude = ds["latitude"].values

            if len(longitude) < 2:

                return False, (
                    "longitude coordinate has "
                    "insufficient points"
                )

            if len(latitude) < 2:

                return False, (
                    "latitude coordinate has "
                    "insufficient points"
                )

            lon_resolution = float(
                np.median(
                    np.abs(
                        np.diff(longitude)
                    )
                )
            )

            lat_resolution = float(
                np.median(
                    np.abs(
                        np.diff(latitude)
                    )
                )
            )

            if not np.isclose(
                lon_resolution,
                0.25,
                atol=0.001,
            ):

                return False, (
                    f"longitude resolution is "
                    f"{lon_resolution}"
                )

            if not np.isclose(
                lat_resolution,
                0.25,
                atol=0.001,
            ):

                return False, (
                    f"latitude resolution is "
                    f"{lat_resolution}"
                )

            # ------------------------------------------------------------------
            # BBOX
            #
            # CMEMS returns cell-centered coordinates.
            # Therefore the returned coordinate extrema can be
            # 0.125° inside the requested boundaries.
            # ------------------------------------------------------------------

            actual_lon_min = float(
                np.min(longitude)
            )

            actual_lon_max = float(
                np.max(longitude)
            )

            actual_lat_min = float(
                np.min(latitude)
            )

            actual_lat_max = float(
                np.max(latitude)
            )

            tolerance = 0.25

            if (
                actual_lon_min
                < LON_MIN - tolerance
                or
                actual_lon_max
                > LON_MAX + tolerance
                or
                actual_lat_min
                < LAT_MIN - tolerance
                or
                actual_lat_max
                > LAT_MAX + tolerance
            ):

                return False, (
                    "returned grid exceeds requested BBOX"
                )

            # ------------------------------------------------------------------
            # Time continuity
            # ------------------------------------------------------------------

            times = ds["time"].values

            if len(times) > 1:

                differences = np.diff(
                    times
                )

                one_day = np.timedelta64(
                    1,
                    "D"
                )

                if not np.all(
                    differences == one_day
                ):

                    return False, (
                        "time coordinate contains "
                        "gaps or irregular intervals"
                    )

            # ------------------------------------------------------------------
            # All checks passed
            # ------------------------------------------------------------------

            return True, "all structural checks passed"

    except Exception as exc:

        return False, (
            f"{type(exc).__name__}: {exc}"
        )


# ==============================================================================
# INITIAL CHECK
# ==============================================================================

print()
print("=" * 80)
print("CMEMS GLOBCURRENT — MULTI-YEAR YEARLY ACQUISITION")
print("=" * 80)

print()
print(
    f"Years          : "
    f"{START_YEAR} → {END_YEAR}"
)

print(
    f"Total years    : "
    f"{END_YEAR - START_YEAR + 1}"
)

print(
    f"Product        : "
    f"{PRODUCT_ID}"
)

print(
    f"Dataset        : "
    f"{DATASET_ID}"
)

print()
print("Configuration:")

print(
    f"  BBOX         : "
    f"{LON_MIN}–{LON_MAX}°E, "
    f"{LAT_MIN}–{LAT_MAX}°N"
)

print(
    f"  Grid         : "
    f"0.25° × 0.25°"
)

print(
    f"  Variables    : "
    f"uo, vo"
)

print(
    f"  Depth        : "
    f"{DEPTH} m ONLY"
)

print(
    "  Temporal     : "
    "Daily"
)

print(
    "  Acquisition  : "
    "1 server-side request per year"
)

print()
print(
    f"Output root:"
)

print(
    f"  {OUTPUT_ROOT}"
)

print()


# ==============================================================================
# ROOT DIRECTORY CHECK
# ==============================================================================

if not OUTPUT_ROOT.exists():

    print("=" * 80)
    print("FATAL ERROR")
    print("=" * 80)

    print()
    print(
        f"Output root does not exist:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    print()
    print(
        "The existing folder structure was expected "
        "to already be present."
    )

    sys.exit(1)


# ==============================================================================
# BUILD YEAR PLAN
# ==============================================================================

years = list(
    range(
        START_YEAR,
        END_YEAR + 1
    )
)


# ==============================================================================
# YEAR STATUS TRACKING
# ==============================================================================

successful_years = []
skipped_years = []
failed_years = []

year_sizes = {}

overall_start = time.perf_counter()


# ==============================================================================
# YEAR LOOP
# ==============================================================================

for index, year in enumerate(years, start=1):

    # ==========================================================================
    # YEAR-SPECIFIC PATHS
    #
    # THIS IS THE IMPORTANT PART:
    #
    # Every year's file is constructed using that year's directory.
    # ==========================================================================

    year_dir = (
        OUTPUT_ROOT
        / str(year)
    )

    output_file = (
        year_dir
        / f"GlobCurrent_{year}_surface_uv.nc"
    )

    expected = expected_days(
        year
    )

    start_datetime = (
        f"{year}-01-01T00:00:00"
    )

    end_datetime = (
        f"{year}-12-31T23:59:59"
    )

    # ==========================================================================
    # YEAR HEADER
    # ==========================================================================

    print()
    print("=" * 80)

    print(
        f"YEAR {index}/{len(years)} — {year}"
    )

    print("=" * 80)

    print()
    print(
        f"Period        : "
        f"{start_datetime} → {end_datetime}"
    )

    print(
        f"Expected days : "
        f"{expected}"
    )

    print(
        f"Leap year     : "
        f"{'YES' if calendar.isleap(year) else 'NO'}"
    )

    print(
        f"Output folder : "
        f"{year_dir}"
    )

    print(
        f"Output file   : "
        f"{output_file.name}"
    )

    # ==========================================================================
    # YEAR DIRECTORY CHECK
    # ==========================================================================

    if not year_dir.exists():

        print()
        print(
            "[FAIL] Expected year folder does not exist."
        )

        print(
            f"       {year_dir}"
        )

        print(
            "       Folder structure will NOT be created "
            "automatically."
        )

        failed_years.append(
            year
        )

        continue

    # ==========================================================================
    # EXISTING FILE CHECK
    # ==========================================================================

    if output_file.exists():

        existing_size = (
            output_file.stat().st_size
        )

        print()
        print(
            "[EXISTS] Yearly file already present."
        )

        print(
            f"         Size: "
            f"{format_bytes(existing_size)}"
        )

        print()
        print(
            "Validating existing file..."
        )

        valid, reason = validate_existing_file(
            output_file,
            expected,
        )

        if valid:

            print(
                "[PASS] Existing file passed "
                "structural validation."
            )

            print(
                "[SKIP] No download required."
            )

            skipped_years.append(
                year
            )

            year_sizes[year] = (
                existing_size
            )

            continue

        print(
            "[WARNING] Existing file failed "
            "structural validation."
        )

        print(
            f"          Reason: {reason}"
        )

        print()

        response = input(
            f"Replace {year} file? [y/N]: "
        ).strip().lower()

        if response != "y":

            print(
                "[SKIP] File retained."
            )

            skipped_years.append(
                year
            )

            year_sizes[year] = (
                existing_size
            )

            continue

        try:

            output_file.unlink()

            print(
                "[INFO] Invalid file removed."
            )

        except PermissionError:

            print(
                "[FAIL] File is open by another program."
            )

            failed_years.append(
                year
            )

            continue

    # ==========================================================================
    # CMEMS REQUEST
    # ==========================================================================

    print()
    print(
        "Starting server-side yearly acquisition..."
    )

    print()
    print(
        "CMEMS operations:"
    )

    print(
        "  ✓ Temporal subset"
    )

    print(
        "  ✓ Spatial subset"
    )

    print(
        "  ✓ Variable subset"
    )

    print(
        "  ✓ Surface-depth selection"
    )

    print()
    print(
        "Request: 1/1"
    )

    print()

    year_start = time.perf_counter()

    try:

        # ----------------------------------------------------------------------
        # ONE request for the ENTIRE YEAR
        #
        # No force_download because it is deprecated.
        # ----------------------------------------------------------------------

        copernicusmarine.subset(

            dataset_id=DATASET_ID,

            variables=VARIABLES,

            start_datetime=start_datetime,

            end_datetime=end_datetime,

            minimum_longitude=LON_MIN,
            maximum_longitude=LON_MAX,

            minimum_latitude=LAT_MIN,
            maximum_latitude=LAT_MAX,

            minimum_depth=DEPTH,
            maximum_depth=DEPTH,

            output_directory=str(
                year_dir
            ),

            output_filename=output_file.name,
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - year_start
        )

        print()
        print(
            f"[FAILED] {year}"
        )

        print(
            f"Elapsed : "
            f"{format_duration(elapsed)}"
        )

        print(
            f"Error   : "
            f"{type(exc).__name__}: {exc}"
        )

        failed_years.append(
            year
        )

        continue

    # ==========================================================================
    # FILE EXISTENCE
    # ==========================================================================

    if not output_file.exists():

        print()
        print(
            "[FAILED] CMEMS request completed, "
            "but expected output file was not found."
        )

        failed_years.append(
            year
        )

        continue

    # ==========================================================================
    # FILE SIZE
    # ==========================================================================

    file_size = (
        output_file.stat().st_size
    )

    elapsed = (
        time.perf_counter()
        - year_start
    )

    if file_size == 0:

        print()
        print(
            "[FAILED] Output file is empty."
        )

        failed_years.append(
            year
        )

        continue

    # ==========================================================================
    # DOWNLOAD SUMMARY
    # ==========================================================================

    print()
    print(
        f"[DOWNLOADED] {year}"
    )

    print(
        f"File size   : "
        f"{format_bytes(file_size)}"
    )

    print(
        f"Elapsed     : "
        f"{format_duration(elapsed)}"
    )

    if elapsed > 0:

        speed = (
            file_size
            / (1024 ** 2)
            / elapsed
        )

        print(
            f"Throughput  : "
            f"{speed:.2f} MB/s"
        )

    # ==========================================================================
    # VALIDATE DOWNLOADED FILE
    # ==========================================================================

    print()
    print(
        "Validating downloaded NetCDF..."
    )

    valid, reason = validate_existing_file(
        output_file,
        expected,
    )

    if not valid:

        print()
        print(
            "[FAIL] Yearly file failed validation."
        )

        print(
            f"       Reason: {reason}"
        )

        failed_years.append(
            year
        )

        continue

    # ==========================================================================
    # SUCCESS
    # ==========================================================================

    print()
    print(
        "[PASS] Variables verified."
    )

    print(
        "[PASS] Surface depth verified."
    )

    print(
        f"[PASS] {expected} daily records verified."
    )

    print(
        "[PASS] 0.25° grid verified."
    )

    print(
        "[PASS] BBOX verified."
    )

    print(
        "[PASS] Time continuity verified."
    )

    successful_years.append(
        year
    )

    year_sizes[year] = (
        file_size
    )

    # ==========================================================================
    # OVERALL PROGRESS
    # ==========================================================================

    completed = (
        len(successful_years)
        + len(skipped_years)
    )

    percentage = (
        completed
        / len(years)
        * 100
    )

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )

    print()
    print(
        "-" * 80
    )

    print(
        f"OVERALL PROGRESS: "
        f"{completed}/{len(years)} "
        f"({percentage:.1f}%)"
    )

    print(
        f"Overall elapsed : "
        f"{format_duration(overall_elapsed)}"
    )

    print(
        f"Successful      : "
        f"{len(successful_years)}"
    )

    print(
        f"Skipped         : "
        f"{len(skipped_years)}"
    )

    print(
        f"Failed          : "
        f"{len(failed_years)}"
    )

    print(
        "-" * 80
    )


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

overall_elapsed = (
    time.perf_counter()
    - overall_start
)

total_size = sum(
    year_sizes.values()
)


print()
print()
print("=" * 80)
print("MULTI-YEAR GLOBCURRENT ACQUISITION COMPLETE")
print("=" * 80)

print()
print("REQUEST")

print(
    f"  Years requested : "
    f"{START_YEAR} → {END_YEAR}"
)

print(
    f"  Total years     : "
    f"{len(years)}"
)

print(
    "  Requests/year   : 1"
)

print(
    "  Total requests  : "
    f"{len(years)}"
)

print()
print("RESULTS")

print(
    f"  Successful      : "
    f"{len(successful_years)}"
)

print(
    f"  Skipped         : "
    f"{len(skipped_years)}"
)

print(
    f"  Failed          : "
    f"{len(failed_years)}"
)

print()
print("STORAGE")

print(
    f"  Total data      : "
    f"{format_bytes(total_size)}"
)

print()
print("TIME")

print(
    f"  Total elapsed   : "
    f"{format_duration(overall_elapsed)}"
)


# ==============================================================================
# YEAR-BY-YEAR TABLE
# ==============================================================================

print()
print("=" * 80)
print("YEAR-BY-YEAR STATUS")
print("=" * 80)

print()
print(
    f"{'YEAR':<8}"
    f"{'STATUS':<14}"
    f"{'EXPECTED DAYS':<16}"
    f"{'FILE SIZE':<15}"
)

print(
    "-" * 53
)

for year in years:

    expected = expected_days(
        year
    )

    if year in successful_years:

        status = "SUCCESS"

    elif year in skipped_years:

        status = "SKIPPED"

    else:

        status = "FAILED"

    size = year_sizes.get(
        year,
        0
    )

    print(
        f"{year:<8}"
        f"{status:<14}"
        f"{expected:<16}"
        f"{format_bytes(size):<15}"
    )


# ==============================================================================
# FAILED YEARS
# ==============================================================================

if failed_years:

    print()
    print("=" * 80)
    print("FAILED YEARS")
    print("=" * 80)

    for year in failed_years:

        print(
            f"  ✗ {year}"
        )

    print()
    print(
        "Successful years remain untouched."
    )

    print(
        "Rerunning the script will validate existing "
        "successful files and only retry incomplete years."
    )


# ==============================================================================
# FINAL VERDICT
# ==============================================================================

print()
print("=" * 80)

if len(failed_years) == 0:

    print(
        "[SUCCESS] All requested years completed "
        "or were already valid."
    )

    print()
    print(
        "Each year is stored as one NetCDF inside "
        "its respective year folder."
    )

else:

    print(
        f"[WARNING] {len(failed_years)} year(s) "
        "require attention."
    )

print()
print("=" * 80)