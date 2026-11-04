# Final code: (needs trial run)
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import requests
import xarray as xr
import numpy as np
import time
import threading
import os
import gc


# ============================================================
# OCEANEMBED — NOAA OISST v2.1
# 2012–2024 FULL ACQUISITION + BBOX SUBSETTING
# ============================================================
#
# ARCHITECTURE:
#
#   3 CHANNELS
#   10 WORKERS / CHANNEL
#   30 MAX CONCURRENT DOWNLOAD REQUESTS
#
#
# DOWNLOAD:
#
#   NOAA NCEI OISST v2.1
#
#
# TEMPORAL RANGE:
#
#   2012-01-01 → 2024-12-31
#
#
# SPATIAL BBOX:
#
#   Latitude  : 5°N  → 30°N
#   Longitude : 45°E → 105°E
#
#
# PRODUCTION STRUCTURE:
#
#   E:\Ocean_data\OISST\data\raw\2012\01\
#   E:\Ocean_data\OISST\data\raw\2012\02\
#   ...
#   E:\Ocean_data\OISST\data\raw\2024\12\
#
#
# IMPORTANT:
#
# The BBOX subset is stored AT THE SAME PATH as the downloaded
# file.
#
# There is NO separate "subset" directory.
#
#
# PROCESS:
#
#   1. Download global NetCDF
#   2. Validate download
#   3. If already BBOX-sized → keep it
#   4. If global → subset to BBOX
#   5. Write subset to temporary file
#   6. Validate subset
#   7. Replace production file
#
#
# SAFETY:
#
# The production file is only replaced after successful
# validation of the new subset.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

START_YEAR = 2012
END_YEAR = 2024


# ============================================================
# OCEANEMBED BOUNDING BOX
# ============================================================

LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0


# ============================================================
# NOAA NCEI DIRECT DOWNLOAD
# ============================================================

BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr/"
)


# ============================================================
# PARALLEL CONFIGURATION
# ============================================================

CHANNEL_COUNT = 3

WORKERS_PER_CHANNEL = 10

TOTAL_WORKERS = (
    CHANNEL_COUNT
    *
    WORKERS_PER_CHANNEL
)


# ============================================================
# REQUEST CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = 300


# ============================================================
# PRODUCTION ROOT
# ============================================================

PRODUCTION_ROOT = Path(
    r"E:\Ocean_data\OISST\data\raw"
)

PRODUCTION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED OISST GRID
# ============================================================

GLOBAL_LAT_COUNT = 720
GLOBAL_LON_COUNT = 1440

BBOX_LAT_COUNT = 100
BBOX_LON_COUNT = 240


# ============================================================
# VARIABLES
# ============================================================
#
# OISST v2.1 files contain:
#
#   sst
#   anom
#   err
#   ice
#
# We retain all four.
# ============================================================

RETAIN_VARIABLES = [
    "sst",
    "anom",
    "err",
    "ice",
]


# ============================================================
# THREAD-SAFE COUNTERS
# ============================================================

counter_lock = threading.Lock()

print_lock = threading.Lock()

downloaded_count = 0

skipped_count = 0

failed_count = 0

subsetted_count = 0

already_subset_count = 0

active_requests = 0


# ============================================================
# DATE GENERATOR
# ============================================================

def generate_dates(
    start_date,
    end_date
):

    current = start_date

    while current <= end_date:

        yield current

        current += timedelta(
            days=1
        )


# ============================================================
# BUILD FILE URL
# ============================================================

def build_file_url(day):

    filename = (
        f"oisst-avhrr-v02r01."
        f"{day.strftime('%Y%m%d')}.nc"
    )

    month_dir = (
        day.strftime("%Y%m")
    )

    return (
        BASE_URL
        + month_dir
        + "/"
        + filename
    )


# ============================================================
# BUILD OUTPUT DIRECTORY
# ============================================================

def build_output_directory(day):

    output_dir = (
        PRODUCTION_ROOT
        / f"{day.year:04d}"
        / f"{day.month:02d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir


# ============================================================
# BUILD PRODUCTION FILE
# ============================================================

def build_output_file(day):

    output_dir = (
        build_output_directory(day)
    )

    filename = (
        f"oisst-avhrr-v02r01."
        f"{day.strftime('%Y%m%d')}.nc"
    )

    return (
        output_dir
        / filename
    )


# ============================================================
# BASIC NETCDF VALIDATION
# ============================================================

def is_valid_file(filepath):

    if not filepath.exists():
        return False

    try:

        if filepath.stat().st_size <= 0:
            return False

    except OSError:

        return False


    try:

        with open(
            filepath,
            "rb"
        ) as f:

            header = f.read(8)

    except OSError:

        return False


    # NetCDF-4 / HDF5
    if header.startswith(
        b"\x89HDF"
    ):

        return True


    # NetCDF classic
    if header.startswith(
        b"CDF"
    ):

        return True


    return False


# ============================================================
# CHECK WHETHER FILE IS ALREADY BBOX-SUBSETTED
# ============================================================

def is_already_subsetted(
    filepath
):

    ds = None

    try:

        if not is_valid_file(
            filepath
        ):

            return False


        ds = xr.open_dataset(
            filepath,
            decode_times=True
        )


        if "lat" not in ds.coords:

            return False


        if "lon" not in ds.coords:

            return False


        if len(ds.lat) != BBOX_LAT_COUNT:

            return False


        if len(ds.lon) != BBOX_LON_COUNT:

            return False


        lat = ds.lat.values

        lon = ds.lon.values


        if lat.min() < LAT_MIN:

            return False


        if lat.max() > LAT_MAX:

            return False


        if lon.min() < LON_MIN:

            return False


        if lon.max() > LON_MAX:

            return False


        return True


    except Exception:

        return False


    finally:

        if ds is not None:

            try:
                ds.close()
            except Exception:
                pass


# ============================================================
# VALIDATE FINAL BBOX FILE
# ============================================================

def validate_bbox_file(
    filepath,
    expected_date
):

    ds = None

    try:

        # ----------------------------------------------------
        # BASIC FILE
        # ----------------------------------------------------

        if not is_valid_file(
            filepath
        ):

            return (
                False,
                "Invalid NetCDF file."
            )


        # ----------------------------------------------------
        # OPEN
        # ----------------------------------------------------

        ds = xr.open_dataset(
            filepath,
            decode_times=True
        )


        # ----------------------------------------------------
        # REQUIRED VARIABLES
        # ----------------------------------------------------

        missing = []

        for variable in RETAIN_VARIABLES:

            if variable not in ds:

                missing.append(
                    variable
                )


        if missing:

            return (
                False,
                "Missing variables: "
                + ", ".join(
                    missing
                )
            )


        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        if "lat" not in ds.coords:

            return (
                False,
                "Latitude coordinate missing."
            )


        if "lon" not in ds.coords:

            return (
                False,
                "Longitude coordinate missing."
            )


        # ----------------------------------------------------
        # DIMENSIONS
        # ----------------------------------------------------

        if len(ds.lat) != BBOX_LAT_COUNT:

            return (
                False,
                "Incorrect latitude count: "
                + str(len(ds.lat))
            )


        if len(ds.lon) != BBOX_LON_COUNT:

            return (
                False,
                "Incorrect longitude count: "
                + str(len(ds.lon))
            )


        # ----------------------------------------------------
        # COORDINATE RANGE
        # ----------------------------------------------------

        lat = ds.lat.values

        lon = ds.lon.values


        if lat.min() < LAT_MIN:

            return (
                False,
                "Latitude below BBOX."
            )


        if lat.max() > LAT_MAX:

            return (
                False,
                "Latitude above BBOX."
            )


        if lon.min() < LON_MIN:

            return (
                False,
                "Longitude below BBOX."
            )


        if lon.max() > LON_MAX:

            return (
                False,
                "Longitude above BBOX."
            )


        # ----------------------------------------------------
        # MONOTONIC COORDINATES
        # ----------------------------------------------------

        if not np.all(
            np.diff(lat) > 0
        ):

            return (
                False,
                "Latitude is not increasing."
            )


        if not np.all(
            np.diff(lon) > 0
        ):

            return (
                False,
                "Longitude is not increasing."
            )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        if "time" not in ds.coords:

            return (
                False,
                "Time coordinate missing."
            )


        if len(ds.time) != 1:

            return (
                False,
                "Expected one time value."
            )


        # ----------------------------------------------------
        # VERIFY DATE
        # ----------------------------------------------------

        actual_time = (
            ds.time.values[0]
        )


        actual_date = (
            np.datetime_as_string(
                actual_time,
                unit="D"
            )
        )


        expected_date_string = (
            expected_date.isoformat()
        )


        if actual_date != (
            expected_date_string
        ):

            return (
                False,
                "Dataset date "
                + actual_date
                + " does not match "
                + expected_date_string
            )


        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        for variable in RETAIN_VARIABLES:

            ds[variable].load()


        return (
            True,
            "PASS"
        )


    except Exception as exc:

        return (
            False,
            str(exc)
        )


    finally:

        if ds is not None:

            try:
                ds.close()
            except Exception:
                pass


# ============================================================
# FIND BBOX INDICES
# ============================================================

def find_bbox_indices(
    ds
):

    if "lat" not in ds.coords:

        raise RuntimeError(
            "Latitude coordinate missing."
        )


    if "lon" not in ds.coords:

        raise RuntimeError(
            "Longitude coordinate missing."
        )


    lat = ds.lat.values

    lon = ds.lon.values


    lat_indices = np.where(
        (
            lat >= LAT_MIN
        )
        &
        (
            lat <= LAT_MAX
        )
    )[0]


    lon_indices = np.where(
        (
            lon >= LON_MIN
        )
        &
        (
            lon <= LON_MAX
        )
    )[0]


    if len(lat_indices) != (
        BBOX_LAT_COUNT
    ):

        raise RuntimeError(
            "Expected "
            + str(BBOX_LAT_COUNT)
            + " latitude cells, got "
            + str(len(lat_indices))
        )


    if len(lon_indices) != (
        BBOX_LON_COUNT
    ):

        raise RuntimeError(
            "Expected "
            + str(BBOX_LON_COUNT)
            + " longitude cells, got "
            + str(len(lon_indices))
        )


    return (
        lat_indices,
        lon_indices
    )


# ============================================================
# BUILD SAFE ENCODING FOR SUBSET
# ============================================================
#
# Preserve useful NetCDF encoding information from the source:
#
#   _FillValue
#   dtype
#   scale_factor
#   add_offset
#   compression where possible
#
# Chunk sizes are intentionally removed because the original
# global chunks may not fit the 100 × 240 subset grid.
# ============================================================

def build_subset_encoding(
    ds
):

    encoding = {}


    for variable in RETAIN_VARIABLES:

        source_encoding = dict(
            ds[variable].encoding
        )


        safe_encoding = {}


        keys_to_preserve = [

            "_FillValue",
            "dtype",
            "scale_factor",
            "add_offset",
            "zlib",
            "complevel",
            "shuffle",
            "fletcher32",
            "contiguous",
        ]


        for key in keys_to_preserve:

            if key in source_encoding:

                value = (
                    source_encoding[key]
                )


                # ------------------------------------------------
                # Avoid incompatible compression/chunk options.
                # ------------------------------------------------

                if key == "contiguous":

                    # Contiguous storage cannot be combined
                    # with compression.

                    if (
                        source_encoding.get(
                            "zlib",
                            False
                        )
                    ):

                        continue


                safe_encoding[key] = value


        # --------------------------------------------------------
        # Explicitly remove chunking from copied encoding.
        # NetCDF4 will choose appropriate storage.
        # --------------------------------------------------------

        safe_encoding.pop(
            "chunksizes",
            None
        )


        safe_encoding.pop(
            "chunksizes",
            None
        )


        encoding[
            variable
        ] = safe_encoding


    return encoding


# ============================================================
# SUBSET ONE EXISTING GLOBAL FILE
# ============================================================

def subset_existing_file(
    input_file,
    expected_date
):

    global subsetted_count
    global already_subset_count
    global failed_count


    start = time.perf_counter()


    # ========================================================
    # ALREADY SUBSETTED?
    # ========================================================

    if is_already_subsetted(
        input_file
    ):

        valid, message = (
            validate_bbox_file(
                input_file,
                expected_date
            )
        )


        if valid:

            with counter_lock:

                already_subset_count += 1


            return {

                "status":
                    "already_subsetted",

                "file":
                    input_file,

                "date":
                    expected_date,
            }


    # ========================================================
    # TEMP FILE
    # ========================================================

    temp_file = (
        input_file.with_name(
            input_file.stem
            + ".subset.tmp.nc"
        )
    )


    temp_file.unlink(
        missing_ok=True
    )


    ds = None

    subset = None


    try:

        # ====================================================
        # OPEN SOURCE
        # ====================================================

        ds = xr.open_dataset(
            input_file,
            decode_times=True
        )


        # ====================================================
        # VERIFY GLOBAL GRID
        # ====================================================

        if len(ds.lat) != (
            GLOBAL_LAT_COUNT
        ):

            raise RuntimeError(
                "Unexpected latitude grid: "
                + str(len(ds.lat))
            )


        if len(ds.lon) != (
            GLOBAL_LON_COUNT
        ):

            raise RuntimeError(
                "Unexpected longitude grid: "
                + str(len(ds.lon))
            )


        # ====================================================
        # VERIFY VARIABLES
        # ====================================================

        missing = []

        for variable in RETAIN_VARIABLES:

            if variable not in ds:

                missing.append(
                    variable
                )


        if missing:

            raise RuntimeError(
                "Missing variables: "
                + ", ".join(
                    missing
                )
            )


        # ====================================================
        # VERIFY DATE
        # ====================================================

        if "time" in ds.coords:

            source_time = (
                ds.time.values[0]
            )

            source_date = (
                np.datetime_as_string(
                    source_time,
                    unit="D"
                )
            )


            if source_date != (
                expected_date.isoformat()
            ):

                raise RuntimeError(
                    "Source date "
                    + source_date
                    + " does not match "
                    + expected_date.isoformat()
                )


        # ====================================================
        # FIND BBOX
        # ====================================================

        (
            lat_indices,
            lon_indices
        ) = find_bbox_indices(
            ds
        )


        # ====================================================
        # CREATE SUBSET
        # ====================================================

        subset = (
            ds[
                RETAIN_VARIABLES
            ]
            .isel(
                lat=lat_indices,
                lon=lon_indices
            )
        )


        # ====================================================
        # FORCE LOAD
        # ====================================================

        subset.load()


        # ====================================================
        # BUILD ENCODING
        # ====================================================

        encoding = (
            build_subset_encoding(
                ds
            )
        )


        # ====================================================
        # CLOSE SOURCE
        # ====================================================

        ds.close()

        ds = None


        # ====================================================
        # WRITE TEMPORARY FILE
        # ====================================================

        subset.to_netcdf(
            temp_file,
            mode="w",
            format="NETCDF4",
            encoding=encoding
        )


        # ====================================================
        # RELEASE MEMORY
        # ====================================================

        del subset

        subset = None

        gc.collect()


        # ====================================================
        # VALIDATE TEMP FILE
        # ====================================================

        valid, message = (
            validate_bbox_file(
                temp_file,
                expected_date
            )
        )


        if not valid:

            temp_file.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Subset validation failed: "
                + message
            )


        # ====================================================
        # SIZE
        # ====================================================

        original_bytes = (
            input_file.stat().st_size
        )


        subset_bytes = (
            temp_file.stat().st_size
        )


        original_mb = (
            original_bytes
            / (1024 * 1024)
        )


        subset_mb = (
            subset_bytes
            / (1024 * 1024)
        )


        if original_bytes > 0:

            reduction_pct = (

                1
                -
                (
                    subset_bytes
                    /
                    original_bytes
                )

            ) * 100

        else:

            reduction_pct = 0.0


        # ====================================================
        # REPLACE ORIGINAL
        # ====================================================

        temp_file.replace(
            input_file
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        with counter_lock:

            subsetted_count += 1


        return {

            "status":
                "subsetted",

            "file":
                input_file,

            "date":
                expected_date,

            "original_mb":
                original_mb,

            "subset_mb":
                subset_mb,

            "reduction_pct":
                reduction_pct,

            "elapsed":
                elapsed
        }


    except Exception as exc:

        # ----------------------------------------------------
        # CRITICAL:
        #
        # Original file is NOT deleted/replaced here.
        # ----------------------------------------------------

        temp_file.unlink(
            missing_ok=True
        )


        with counter_lock:

            failed_count += 1


        return {

            "status":
                "failed",

            "file":
                input_file,

            "date":
                expected_date,

            "error":
                str(exc)
        }


    finally:

        if ds is not None:

            try:
                ds.close()
            except Exception:
                pass


        if subset is not None:

            try:
                del subset
            except Exception:
                pass


        gc.collect()


# ============================================================
# DOWNLOAD ONE DAILY FILE
# ============================================================

def download_file(
    channel_id,
    day
):

    global downloaded_count
    global skipped_count
    global failed_count
    global active_requests


    output_file = (
        build_output_file(day)
    )


    # ========================================================
    # EXISTING FILE
    # ========================================================

    if is_valid_file(
        output_file
    ):

        with counter_lock:

            skipped_count += 1


        return {

            "channel":
                channel_id,

            "date":
                day,

            "status":
                "existing",

            "file":
                output_file,
        }


    # ========================================================
    # URL
    # ========================================================

    url = build_file_url(
        day
    )


    start = time.perf_counter()


    # ========================================================
    # ACTIVE REQUEST
    # ========================================================

    with counter_lock:

        active_requests += 1

        current_active = (
            active_requests
        )


    temp_file = (
        output_file.with_suffix(
            ".download"
        )
    )


    try:

        # ====================================================
        # REQUEST
        # ====================================================

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT
        )


        response.raise_for_status()


        # ====================================================
        # RESPONSE
        # ====================================================

        if not response.content:

            raise RuntimeError(
                "Downloaded response is empty."
            )


        # ====================================================
        # WRITE TEMP DOWNLOAD
        # ====================================================

        temp_file.write_bytes(
            response.content
        )


        # ====================================================
        # VALIDATE DOWNLOAD
        # ====================================================

        if not is_valid_file(
            temp_file
        ):

            temp_file.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Downloaded file failed "
                "NetCDF validation."
            )


        # ====================================================
        # MOVE DOWNLOAD INTO PRODUCTION
        # ====================================================

        temp_file.replace(
            output_file
        )


        # ====================================================
        # DOWNLOAD COUNTER
        # ====================================================

        with counter_lock:

            downloaded_count += 1

            active_requests -= 1


        elapsed = (
            time.perf_counter()
            - start
        )


        size_bytes = (
            output_file.stat().st_size
        )


        size_mb = (
            size_bytes
            / (1024 * 1024)
        )


        return {

            "channel":
                channel_id,

            "date":
                day,

            "status":
                "downloaded",

            "file":
                output_file,

            "size":
                size_bytes,

            "size_mb":
                size_mb,

            "elapsed":
                elapsed,

            "active":
                current_active,
        }


    except Exception as exc:

        # ====================================================
        # CLEAN DOWNLOAD TEMP FILE
        # ====================================================

        temp_file.unlink(
            missing_ok=True
        )


        with counter_lock:

            failed_count += 1

            active_requests -= 1


        return {

            "channel":
                channel_id,

            "date":
                day,

            "status":
                "failed",

            "file":
                output_file,

            "error":
                str(exc)
        }


# ============================================================
# DOWNLOAD + SUBSET ONE FILE
# ============================================================

def acquire_one_file(
    channel_id,
    day
):

    # ========================================================
    # STEP 1 — DOWNLOAD / CHECK EXISTING
    # ========================================================

    download_result = (
        download_file(
            channel_id,
            day
        )
    )


    # ========================================================
    # DOWNLOAD FAILED
    # ========================================================

    if download_result["status"] == (
        "failed"
    ):

        return download_result


    output_file = (
        download_result["file"]
    )


    # ========================================================
    # STEP 2 — SUBSET
    # ========================================================
    #
    # This runs for:
    #
    #   a) newly downloaded global files
    #   b) existing global files
    #
    # If the file is already a valid BBOX subset,
    # subset_existing_file() simply skips it.
    # ========================================================

    subset_result = (
        subset_existing_file(
            output_file,
            day
        )
    )


    # ========================================================
    # MERGE RESULTS
    # ========================================================

    return {

        "channel":
            channel_id,

        "date":
            day,

        "download_status":
            download_result["status"],

        "status":
            subset_result["status"],

        "file":
            output_file,

        "size":
            subset_result.get(
                "subset_mb",
                download_result.get(
                    "size_mb",
                    0.0
                )
            ),

        "original_mb":
            subset_result.get(
                "original_mb",
                download_result.get(
                    "size_mb",
                    0.0
                )
            ),

        "subset_mb":
            subset_result.get(
                "subset_mb",
                0.0
            ),

        "reduction_pct":
            subset_result.get(
                "reduction_pct",
                0.0
            ),

        "elapsed":
            subset_result.get(
                "elapsed",
                0.0
            ),

        "error":
            subset_result.get(
                "error",
                None
            ),
    }


# ============================================================
# CHANNEL DATE RANGES
# ============================================================

def build_channel_ranges(
    year
):

    return {

        1: (
            date(
                year,
                1,
                1
            ),

            date(
                year,
                4,
                30
            )
        ),

        2: (
            date(
                year,
                5,
                1
            ),

            date(
                year,
                8,
                31
            )
        ),

        3: (
            date(
                year,
                9,
                1
            ),

            date(
                year,
                12,
                31
            )
        ),
    }


# ============================================================
# RUN ONE CHANNEL
# ============================================================

def run_channel(
    year,
    channel_id,
    start_date,
    end_date
):

    dates = list(
        generate_dates(
            start_date,
            end_date
        )
    )


    print()

    print("-" * 100)

    print(
        f"YEAR {year} — CHANNEL {channel_id}"
    )

    print("-" * 100)

    print(
        f"Period : "
        f"{start_date} → {end_date}"
    )

    print(
        f"Files  : "
        f"{len(dates)}"
    )

    print(
        f"Workers: "
        f"{WORKERS_PER_CHANNEL}"
    )

    print(
        f"Output : "
        f"{PRODUCTION_ROOT / f'{year:04d}'}"
    )

    print()


    channel_start = (
        time.perf_counter()
    )


    results = []

    completed = 0


    # ========================================================
    # 10 WORKERS
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=WORKERS_PER_CHANNEL
    ) as executor:


        futures = {

            executor.submit(
                acquire_one_file,
                channel_id,
                day
            ):
                day

            for day in dates
        }


        for future in as_completed(
            futures
        ):

            result = (
                future.result()
            )


            results.append(
                result
            )


            completed += 1


            elapsed = (
                time.perf_counter()
                - channel_start
            )


            with counter_lock:

                active = (
                    active_requests
                )


            rate = (

                completed
                /
                elapsed

                if elapsed > 0

                else 0
            )


            # ------------------------------------------------
            # LIVE PROGRESS
            # ------------------------------------------------

            print(
                "\r"
                + f"Y{year} C{channel_id}"
                + " | "
                + f"{completed:3d}"
                + "/"
                + f"{len(dates)}"
                + " | Active: "
                + f"{active:2d}"
                + " | Rate: "
                + f"{rate:5.2f}"
                + " files/s | "
                + f"Elapsed: {elapsed:7.2f}s",
                end="",
                flush=True
            )


    channel_elapsed = (
        time.perf_counter()
        - channel_start
    )


    print()
    print()


    # ========================================================
    # CHANNEL RESULT COUNTS
    # ========================================================

    successful = sum(

        1

        for result
        in results

        if result["status"]
        in (
            "subsetted",
            "already_subsetted"
        )
    )


    subsetted = sum(

        1

        for result
        in results

        if result["status"]
        == "subsetted"
    )


    already_subsetted = sum(

        1

        for result
        in results

        if result["status"]
        == "already_subsetted"
    )


    failed = sum(

        1

        for result
        in results

        if result["status"]
        == "failed"
    )


    total_mb = sum(

        result.get(
            "subset_mb",
            0.0
        )

        for result
        in results
    )


    print(
        f"Year {year} Channel {channel_id} "
        f"finished in "
        f"{channel_elapsed:.2f} seconds"
    )


    print(
        f"Successful : "
        f"{successful}/{len(dates)}"
    )


    print(
        f"Subsetted  : "
        f"{subsetted}"
    )


    print(
        f"Already BBOX: "
        f"{already_subsetted}"
    )


    print(
        f"Failed     : "
        f"{failed}"
    )


    print(
        f"BBOX data  : "
        f"{total_mb:.3f} MB"
    )


    return {

        "year":
            year,

        "channel":
            channel_id,

        "expected":
            len(dates),

        "successful":
            successful,

        "subsetted":
            subsetted,

        "already_subsetted":
            already_subsetted,

        "failed":
            failed,

        "bytes":
            int(
                total_mb
                * 1024
                * 1024
            ),

        "elapsed":
            channel_elapsed,

        "results":
            results
    }


# ============================================================
# RUN ONE YEAR
# ============================================================

def run_year(
    year
):

    channel_ranges = (
        build_channel_ranges(
            year
        )
    )


    print()
    print()
    print("#" * 100)

    print(
        f"STARTING YEAR {year}"
    )

    print("#" * 100)


    print()

    print(
        "Production output:"
    )

    print(
        PRODUCTION_ROOT
        / f"{year:04d}"
    )


    total_expected = 0


    # ========================================================
    # SHOW CHANNEL ALLOCATION
    # ========================================================

    for (
        channel_id,
        (
            start_date,
            end_date
        )
    ) in channel_ranges.items():

        days = list(
            generate_dates(
                start_date,
                end_date
            )
        )


        total_expected += (
            len(days)
        )


        print(
            f"Channel {channel_id}: "
            f"{start_date} → "
            f"{end_date} "
            f"({len(days)} files)"
        )


    print()

    print(
        f"Total expected for {year}: "
        f"{total_expected}"
    )


    print()

    print(
        "Starting all three channels "
        "simultaneously..."
    )


    year_start = (
        time.perf_counter()
    )


    channel_results = []


    # ========================================================
    # THREE CHANNEL SUPERVISORS
    # ========================================================

    with ThreadPoolExecutor(
        max_workers=CHANNEL_COUNT
    ) as channel_executor:


        channel_futures = {

            channel_executor.submit(
                run_channel,
                year,
                channel_id,
                start_date,
                end_date
            ):
                channel_id

            for (
                channel_id,
                (
                    start_date,
                    end_date
                )
            )
            in channel_ranges.items()
        }


        for future in as_completed(
            channel_futures
        ):

            channel_id = (
                channel_futures[
                    future
                ]
            )


            try:

                result = (
                    future.result()
                )


                channel_results.append(
                    result
                )


            except Exception as exc:

                print()

                print(
                    f"YEAR {year} "
                    f"CHANNEL {channel_id} "
                    f"CRASHED:"
                )

                print(
                    f"  {exc}"
                )


    year_elapsed = (
        time.perf_counter()
        - year_start
    )


    # ========================================================
    # YEAR STATISTICS
    # ========================================================

    successful = sum(

        result["successful"]

        for result
        in channel_results
    )


    subsetted = sum(

        result["subsetted"]

        for result
        in channel_results
    )


    already_subsetted = sum(

        result["already_subsetted"]

        for result
        in channel_results
    )


    failed = sum(

        result["failed"]

        for result
        in channel_results
    )


    total_bytes = sum(

        result["bytes"]

        for result
        in channel_results
    )


    total_mb = (
        total_bytes
        / (1024 * 1024)
    )


    coverage = (

        successful
        /
        total_expected
        *
        100

        if total_expected > 0

        else 0
    )


    # ========================================================
    # YEAR REPORT
    # ========================================================

    print()

    print("=" * 100)

    print(
        f"YEAR {year} SUMMARY"
    )

    print("=" * 100)


    print()

    print(
        f"Expected files : "
        f"{total_expected}"
    )


    print(
        f"Valid BBOX files: "
        f"{successful}"
    )


    print(
        f"Newly subsetted : "
        f"{subsetted}"
    )


    print(
        f"Already BBOX    : "
        f"{already_subsetted}"
    )


    print(
        f"Failed          : "
        f"{failed}"
    )


    print(
        f"Coverage        : "
        f"{coverage:.2f}%"
    )


    print(
        f"BBOX data       : "
        f"{total_mb:.3f} MB"
    )


    print(
        f"Year time       : "
        f"{year_elapsed:.2f} seconds"
    )


    if year_elapsed > 0:

        print(
            f"Effective rate  : "
            f"{successful / year_elapsed:.3f} files/s"
        )


    # ========================================================
    # CHANNEL TIMINGS
    # ========================================================

    print()

    print(
        "CHANNEL TIMINGS"
    )

    print("-" * 100)


    for result in sorted(
        channel_results,
        key=lambda x: x["channel"]
    ):

        channel_mb = (
            result["bytes"]
            / (1024 * 1024)
        )


        print(
            f"Channel "
            f"{result['channel']} : "
            f"{result['elapsed']:.2f} s | "
            f"{result['successful']}/"
            f"{result['expected']} valid | "
            f"{channel_mb:.3f} MB"
        )


    print()


    return {

        "year":
            year,

        "expected":
            total_expected,

        "successful":
            successful,

        "subsetted":
            subsetted,

        "already_subsetted":
            already_subsetted,

        "failed":
            failed,

        "bytes":
            total_bytes,

        "elapsed":
            year_elapsed,

        "channel_results":
            channel_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    global downloaded_count
    global skipped_count
    global failed_count
    global subsetted_count
    global already_subset_count
    global active_requests


    # ========================================================
    # RESET COUNTERS
    # ========================================================

    downloaded_count = 0

    skipped_count = 0

    failed_count = 0

    subsetted_count = 0

    already_subset_count = 0

    active_requests = 0


    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 100)

    print(
        "OCEANEMBED — NOAA OISST v2.1"
    )

    print(
        "2012–2024 FULL ACQUISITION "
        "+ AUTOMATIC BBOX SUBSETTING"
    )

    print(
        "3 CHANNELS / "
        "10 WORKERS PER CHANNEL / "
        "30 CONCURRENT REQUESTS"
    )

    print("=" * 100)


    # ========================================================
    # DATASET
    # ========================================================

    print()

    print(
        "Dataset:"
    )

    print(
        "  NOAA Optimum Interpolation "
        "Sea Surface Temperature"
    )

    print(
        "  OISST Version 2.1"
    )

    print(
        "  AVHRR Daily 1/4 Degree"
    )


    # ========================================================
    # PERIOD
    # ========================================================

    print()

    print(
        "Temporal range:"
    )

    print(
        f"  {START_YEAR}-01-01 → "
        f"{END_YEAR}-12-31"
    )


    # ========================================================
    # BBOX
    # ========================================================

    print()

    print(
        "BBOX:"
    )

    print(
        f"  Latitude : "
        f"{LAT_MIN}° → {LAT_MAX}°"
    )

    print(
        f"  Longitude: "
        f"{LON_MIN}° → {LON_MAX}°"
    )


    print()

    print(
        "Global grid:"
    )

    print(
        f"  {GLOBAL_LAT_COUNT} × "
        f"{GLOBAL_LON_COUNT}"
    )


    print()

    print(
        "BBOX grid:"
    )

    print(
        f"  {BBOX_LAT_COUNT} × "
        f"{BBOX_LON_COUNT} = "
        f"{BBOX_LAT_COUNT * BBOX_LON_COUNT:,} cells"
    )


    # ========================================================
    # VARIABLES
    # ========================================================

    print()

    print(
        "Variables retained:"
    )

    for variable in RETAIN_VARIABLES:

        print(
            f"  ✓ {variable}"
        )


    # ========================================================
    # PRODUCTION ROOT
    # ========================================================

    print()

    print(
        "Production root:"
    )

    print(
        PRODUCTION_ROOT
    )


    # ========================================================
    # ARCHITECTURE
    # ========================================================

    print()

    print(
        "Parallel architecture:"
    )

    print(
        f"  Channels            : "
        f"{CHANNEL_COUNT}"
    )

    print(
        f"  Workers/channel     : "
        f"{WORKERS_PER_CHANNEL}"
    )

    print(
        f"  Maximum concurrent  : "
        f"{TOTAL_WORKERS}"
    )


    # ========================================================
    # EXPECTED FILES
    # ========================================================

    total_expected_all_years = 0


    print()

    print(
        "YEAR ALLOCATION"
    )

    print("-" * 100)


    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        year_start = date(
            year,
            1,
            1
        )


        year_end = date(
            year,
            12,
            31
        )


        days = list(
            generate_dates(
                year_start,
                year_end
            )
        )


        total_expected_all_years += (
            len(days)
        )


        print(
            f"{year}: "
            f"{len(days):3d} daily files"
        )


    print("-" * 100)


    print(
        f"TOTAL EXPECTED: "
        f"{total_expected_all_years:,} files"
    )


    # ========================================================
    # START
    # ========================================================

    overall_start = (
        time.perf_counter()
    )


    all_year_results = []


    # ========================================================
    # YEARS RUN SEQUENTIALLY
    # ========================================================
    #
    # IMPORTANT:
    #
    # We preserve the proven architecture:
    #
    #     3 channels
    #     × 10 workers
    #     = 30 concurrent HTTP requests
    #
    # We do NOT launch all 13 years simultaneously.
    #
    # ========================================================

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        result = run_year(
            year
        )


        all_year_results.append(
            result
        )


    # ========================================================
    # OVERALL TIME
    # ========================================================

    overall_elapsed = (
        time.perf_counter()
        - overall_start
    )


    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    total_successful = sum(

        result["successful"]

        for result
        in all_year_results
    )


    total_subsetted = sum(

        result["subsetted"]

        for result
        in all_year_results
    )


    total_already_subsetted = sum(

        result["already_subsetted"]

        for result
        in all_year_results
    )


    total_failed = sum(

        result["failed"]

        for result
        in all_year_results
    )


    total_bytes = sum(

        result["bytes"]

        for result
        in all_year_results
    )


    total_mb = (
        total_bytes
        / (1024 * 1024)
    )


    missing = (
        total_expected_all_years
        -
        total_successful
    )


    coverage = (

        total_successful
        /
        total_expected_all_years
        *
        100

        if total_expected_all_years > 0

        else 0
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print()

    print("=" * 100)

    print(
        "FINAL 2012–2024 "
        "OISST ACQUISITION + SUBSETTING REPORT"
    )

    print("=" * 100)


    print()

    print(
        f"Expected daily files       : "
        f"{total_expected_all_years:,}"
    )


    print(
        f"Valid BBOX files           : "
        f"{total_successful:,}"
    )


    print(
        f"Newly subsetted            : "
        f"{total_subsetted:,}"
    )


    print(
        f"Already BBOX-subsetted     : "
        f"{total_already_subsetted:,}"
    )


    print(
        f"Failed                     : "
        f"{total_failed:,}"
    )


    print(
        f"Missing                    : "
        f"{missing:,}"
    )


    print(
        f"Coverage                   : "
        f"{coverage:.2f}%"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    print()

    print(
        "FINAL BBOX STORAGE"
    )

    print("-" * 100)


    print(
        f"Total BBOX data size       : "
        f"{total_mb:.3f} MB"
    )


    if total_successful > 0:

        print(
            f"Average BBOX file size     : "
            f"{total_mb / total_successful:.3f} MB"
        )


    # ========================================================
    # SPEED
    # ========================================================

    if overall_elapsed > 0:

        files_per_second = (
            total_successful
            /
            overall_elapsed
        )


        mb_per_second = (
            total_mb
            /
            overall_elapsed
        )


        print()

        print(
            f"Average completion rate    : "
            f"{files_per_second:.3f} files/s"
        )


        print(
            f"Effective BBOX throughput  : "
            f"{mb_per_second:.3f} MB/s"
        )


    # ========================================================
    # WALL CLOCK
    # ========================================================

    print()

    print(
        f"Total wall-clock time      : "
        f"{overall_elapsed:.2f} seconds"
    )


    # ========================================================
    # YEAR TIMINGS
    # ========================================================

    print()

    print(
        "YEAR TIMINGS"
    )

    print("-" * 100)


    for result in all_year_results:

        year_mb = (
            result["bytes"]
            / (1024 * 1024)
        )


        print(
            f"{result['year']} : "
            f"{result['elapsed']:.2f} s | "
            f"{result['successful']:,}/"
            f"{result['expected']:,} valid | "
            f"{year_mb:.3f} MB"
        )


    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print()

    print("=" * 100)


    if (

        total_successful
        ==
        total_expected_all_years

        and
        total_failed
        ==
        0

    ):

        print(
            "✓ 2012–2024 FULL OISST "
            "ACQUISITION + BBOX SUBSETTING PASSED"
        )


    else:

        print(
            "✗ 2012–2024 OISST "
            "ACQUISITION/SUBSETTING INCOMPLETE"
        )


    print("=" * 100)


    # ========================================================
    # PRODUCTION STRUCTURE
    # ========================================================

    print()

    print(
        "ALL PRODUCTION FILES REMAIN UNDER:"
    )

    print(
        PRODUCTION_ROOT
    )


    print()

    print(
        "Example:"
    )

    print(
        PRODUCTION_ROOT
        / "2012"
        / "01"
        / "oisst-avhrr-v02r01.20120101.nc"
    )


    print()

    print(
        "The files above contain the "
        "OceanEmbed BBOX."
    )


    print()

    print("=" * 100)

    print(
        "DONE"
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()