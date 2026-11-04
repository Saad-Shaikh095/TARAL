# Year code test
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import requests
import time
import threading


# ============================================================
# OceanEmbed — NOAA OISST v2.1
# 1-YEAR / 3-CHANNEL / 30-WORKER ACQUISITION
# ============================================================
#
# Year:
#     2011
#
# Channel 1:
#     January → April
#     10 workers
#
# Channel 2:
#     May → August
#     10 workers
#
# Channel 3:
#     September → December
#     10 workers
#
# Total:
#     30 concurrent workers
#
# IMPORTANT:
#     Channels are only an execution strategy.
#     Files are stored in the normal monthly directories.
#
# Final structure:
#
# E:\Ocean_data\OISST\data\raw\2011\
#     01\
#     02\
#     ...
#     12\
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2011

LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0

# NOAA NCEI direct-download endpoint
BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr/"
)

# Workers assigned to each channel
WORKERS_PER_CHANNEL = 10

# Number of parallel channels
CHANNEL_COUNT = 3

# HTTP timeout per request
REQUEST_TIMEOUT = 300


# ============================================================
# PRODUCTION OUTPUT ROOT
# ============================================================

PRODUCTION_ROOT = Path(
    r"E:\Ocean_data\OISST\data\raw\2011"
)

PRODUCTION_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHANNEL DATE RANGES
# ============================================================

CHANNEL_RANGES = {

    # --------------------------------------------------------
    # CHANNEL 1
    # January → April
    # --------------------------------------------------------

    1: (
        date(YEAR, 1, 1),
        date(YEAR, 4, 30),
    ),

    # --------------------------------------------------------
    # CHANNEL 2
    # May → August
    # --------------------------------------------------------

    2: (
        date(YEAR, 5, 1),
        date(YEAR, 8, 31),
    ),

    # --------------------------------------------------------
    # CHANNEL 3
    # September → December
    # --------------------------------------------------------

    3: (
        date(YEAR, 9, 1),
        date(YEAR, 12, 31),
    ),
}


# ============================================================
# THREAD-SAFE GLOBAL COUNTERS
# ============================================================

counter_lock = threading.Lock()

downloaded_count = 0
skipped_count = 0
failed_count = 0
active_requests = 0


# ============================================================
# DATE GENERATOR
# ============================================================

def generate_dates(start_date, end_date):

    current = start_date

    while current <= end_date:

        yield current

        current += timedelta(days=1)


# ============================================================
# BUILD NOAA FILE URL
# ============================================================

def build_file_url(day):

    filename = (
        f"oisst-avhrr-v02r01."
        f"{day.strftime('%Y%m%d')}.nc"
    )

    month_dir = day.strftime("%Y%m")

    return (
        BASE_URL
        + month_dir
        + "/"
        + filename
    )


# ============================================================
# DOWNLOAD ONE DAILY FILE
# ============================================================

def download_file(channel_id, day):

    global downloaded_count
    global skipped_count
    global failed_count
    global active_requests

    # --------------------------------------------------------
    # MONTHLY OUTPUT DIRECTORY
    #
    # Example:
    #
    # 2011-01-15 →
    # E:\Ocean_data\OISST\data\raw\2011\01\
    #
    # 2011-07-15 →
    # E:\Ocean_data\OISST\data\raw\2011\07\
    # --------------------------------------------------------

    output_dir = (
        PRODUCTION_ROOT
        / f"{day.month:02d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"oisst-avhrr-v02r01."
        f"{day.strftime('%Y%m%d')}.nc"
    )

    output_file = (
        output_dir / filename
    )

    # --------------------------------------------------------
    # EXISTING FILE CHECK
    # --------------------------------------------------------

    if (
        output_file.exists()
        and output_file.stat().st_size > 0
    ):

        with counter_lock:
            skipped_count += 1

        return {
            "channel": channel_id,
            "date": day,
            "status": "skipped",
            "size": output_file.stat().st_size,
        }

    # --------------------------------------------------------
    # BUILD URL
    # --------------------------------------------------------

    url = build_file_url(day)

    start = time.perf_counter()

    # --------------------------------------------------------
    # TRACK ACTIVE REQUEST
    # --------------------------------------------------------

    with counter_lock:

        active_requests += 1

        current_active = active_requests

    try:

        # ----------------------------------------------------
        # HTTP REQUEST
        # ----------------------------------------------------

        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # BASIC RESPONSE VALIDATION
        # ----------------------------------------------------

        if not response.content:

            raise RuntimeError(
                "Downloaded response is empty."
            )

        # ----------------------------------------------------
        # TEMPORARY DOWNLOAD
        #
        # This prevents a partially downloaded file from
        # appearing as a valid production file.
        # ----------------------------------------------------

        temp_file = output_file.with_suffix(
            ".download"
        )

        temp_file.write_bytes(
            response.content
        )

        if temp_file.stat().st_size == 0:

            temp_file.unlink(
                missing_ok=True
            )

            raise RuntimeError(
                "Downloaded file has zero bytes."
            )

        # ----------------------------------------------------
        # ATOMIC MOVE INTO FINAL LOCATION
        # ----------------------------------------------------

        temp_file.replace(
            output_file
        )

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

        # ----------------------------------------------------
        # UPDATE COUNTERS
        # ----------------------------------------------------

        with counter_lock:

            downloaded_count += 1
            active_requests -= 1

        return {
            "channel": channel_id,
            "date": day,
            "status": "downloaded",
            "size": size_bytes,
            "size_mb": size_mb,
            "elapsed": elapsed,
            "active": current_active,
        }

    except Exception as exc:

        # ----------------------------------------------------
        # CLEAN UP FAILED TEMPORARY FILE
        # ----------------------------------------------------

        temp_file = output_file.with_suffix(
            ".download"
        )

        temp_file.unlink(
            missing_ok=True
        )

        with counter_lock:

            failed_count += 1
            active_requests -= 1

        return {
            "channel": channel_id,
            "date": day,
            "status": "failed",
            "error": str(exc),
        }


# ============================================================
# RUN ONE CHANNEL
# ============================================================

def run_channel(channel_id):

    start_date, end_date = (
        CHANNEL_RANGES[channel_id]
    )

    dates = list(
        generate_dates(
            start_date,
            end_date,
        )
    )

    print()
    print("-" * 100)

    print(
        f"CHANNEL {channel_id}"
    )

    print("-" * 100)

    print(
        f"Period : "
        f"{start_date} → {end_date}"
    )

    print(
        f"Files  : {len(dates)}"
    )

    print(
        f"Workers: {WORKERS_PER_CHANNEL}"
    )

    print(
        f"Output : {PRODUCTION_ROOT}"
    )

    print()

    channel_start = time.perf_counter()

    results = []

    completed = 0

    # --------------------------------------------------------
    # CHANNEL WORKER POOL
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=WORKERS_PER_CHANNEL
    ) as executor:

        futures = {

            executor.submit(
                download_file,
                channel_id,
                day,
            ): day

            for day in dates
        }

        for future in as_completed(
            futures
        ):

            result = future.result()

            results.append(result)

            completed += 1

            elapsed = (
                time.perf_counter()
                - channel_start
            )

            with counter_lock:

                active = active_requests

            # ------------------------------------------------
            # DYNAMIC PROGRESS
            # ------------------------------------------------

            print(
                f"\r"
                f"Channel {channel_id} | "
                f"{completed:3d}/{len(dates)} | "
                f"Active requests: {active:2d} | "
                f"Elapsed: {elapsed:7.2f}s",
                end="",
                flush=True,
            )

    channel_elapsed = (
        time.perf_counter()
        - channel_start
    )

    print()
    print()

    successful = sum(
        1
        for r in results
        if r["status"]
        in ("downloaded", "skipped")
    )

    failures = sum(
        1
        for r in results
        if r["status"] == "failed"
    )

    total_bytes = sum(
        r.get("size", 0)
        for r in results
        if r["status"]
        in ("downloaded", "skipped")
    )

    print(
        f"Channel {channel_id} finished in "
        f"{channel_elapsed:.2f} seconds"
    )

    print(
        f"Successful : "
        f"{successful}/{len(dates)}"
    )

    print(
        f"Failed     : "
        f"{failures}"
    )

    print(
        f"Data       : "
        f"{total_bytes / (1024 * 1024):.3f} MB"
    )

    return {

        "channel": channel_id,

        "expected": len(dates),

        "successful": successful,

        "failed": failures,

        "elapsed": channel_elapsed,

        "bytes": total_bytes,

        "results": results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    global downloaded_count
    global skipped_count
    global failed_count

    print("=" * 100)

    print(
        "OCEANEMBED — NOAA OISST v2.1"
    )

    print(
        "2011 FULL YEAR / "
        "3 CHANNEL / "
        "30 WORKER ACQUISITION"
    )

    print("=" * 100)

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print()

    print("Dataset:")

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

    print()

    print("Year:")

    print(
        f"  {YEAR}"
    )

    print()

    print("Area of Interest:")

    print(
        f"  Latitude : "
        f"{LAT_MIN}° → {LAT_MAX}°"
    )

    print(
        f"  Longitude: "
        f"{LON_MIN}° → {LON_MAX}°"
    )

    # ========================================================
    # PARALLEL CONFIGURATION
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
        f"{CHANNEL_COUNT * WORKERS_PER_CHANNEL}"
    )

    # ========================================================
    # CHANNEL ALLOCATION
    # ========================================================

    print()

    print(
        "Channel allocation:"
    )

    total_expected = 0

    for (
        channel_id,
        (start_date, end_date)
    ) in CHANNEL_RANGES.items():

        days = list(
            generate_dates(
                start_date,
                end_date,
            )
        )

        total_expected += len(days)

        print(
            f"  Channel {channel_id}: "
            f"{start_date} → {end_date} "
            f"({len(days)} files)"
        )

    print()

    print(
        f"Total expected files: "
        f"{total_expected}"
    )

    # ========================================================
    # OUTPUT STRUCTURE
    # ========================================================

    print()

    print(
        "Production output:"
    )

    print(
        f"  {PRODUCTION_ROOT}"
    )

    print()

    print(
        "Monthly directories:"
    )

    for month in range(1, 13):

        month_dir = (
            PRODUCTION_ROOT
            / f"{month:02d}"
        )

        print(
            f"  {month_dir}"
        )

    # ========================================================
    # START ALL CHANNELS
    # ========================================================

    print()

    print("=" * 100)

    print(
        "STARTING ALL THREE CHANNELS "
        "SIMULTANEOUSLY"
    )

    print("=" * 100)

    global_start_time = (
        time.perf_counter()
    )

    channel_results = []

    # --------------------------------------------------------
    # THREE CHANNEL SUPERVISORS
    #
    # Each channel has 10 workers.
    #
    # 3 × 10 = 30 concurrent requests maximum.
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=CHANNEL_COUNT
    ) as channel_executor:

        channel_futures = {

            channel_executor.submit(
                run_channel,
                channel_id,
            ): channel_id

            for channel_id
            in CHANNEL_RANGES
        }

        for future in as_completed(
            channel_futures
        ):

            channel_id = (
                channel_futures[future]
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
                    f"CHANNEL {channel_id} "
                    f"CRASHED: {exc}"
                )

    # ========================================================
    # TOTAL WALL CLOCK TIME
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - global_start_time
    )

    # ========================================================
    # FINAL ACQUISITION REPORT
    # ========================================================

    print()
    print()

    print("=" * 100)

    print(
        "FINAL ACQUISITION REPORT"
    )

    print("=" * 100)

    total_successful = sum(
        r["successful"]
        for r in channel_results
    )

    total_failed = sum(
        r["failed"]
        for r in channel_results
    )

    total_bytes = sum(
        r["bytes"]
        for r in channel_results
    )

    # ========================================================
    # OVERALL STATISTICS
    # ========================================================

    print()

    print(
        f"Expected daily files       : "
        f"{total_expected}"
    )

    print(
        f"Valid files                : "
        f"{total_successful}"
    )

    print(
        f"Downloaded this run        : "
        f"{downloaded_count}"
    )

    print(
        f"Skipped as already valid   : "
        f"{skipped_count}"
    )

    print(
        f"Failed downloads           : "
        f"{total_failed}"
    )

    missing = (
        total_expected
        - total_successful
    )

    print(
        f"Missing files              : "
        f"{missing}"
    )

    coverage = (

        total_successful
        / total_expected
        * 100

        if total_expected
        else 0

    )

    print(
        f"Coverage                   : "
        f"{coverage:.2f}%"
    )

    # ========================================================
    # DATA SIZE
    # ========================================================

    print()

    total_mb = (
        total_bytes
        / (1024 * 1024)
    )

    print(
        f"Total valid data size      : "
        f"{total_mb:.3f} MB"
    )

    if total_successful > 0:

        print(
            f"Average daily file size    : "
            f"{total_mb / total_successful:.3f} MB"
        )

    # ========================================================
    # THROUGHPUT
    # ========================================================

    if total_elapsed > 0:

        files_per_second = (
            total_successful
            / total_elapsed
        )

        mb_per_second = (
            total_mb
            / total_elapsed
        )

        print()

        print(
            f"Average completion rate    : "
            f"{files_per_second:.3f} files/s"
        )

        print(
            f"Overall effective speed    : "
            f"{mb_per_second:.3f} MB/s"
        )

    # ========================================================
    # WALL CLOCK TIME
    # ========================================================

    print()

    print(
        f"Total wall-clock time      : "
        f"{total_elapsed:.2f} seconds"
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
        key=lambda x: x["channel"],
    ):

        print(

            f"Channel {result['channel']} : "

            f"{result['elapsed']:.2f} s | "

            f"{result['successful']}/"
            f"{result['expected']} valid | "

            f"{result['bytes'] / (1024 * 1024):.3f} MB"

        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print("=" * 100)

    if (

        total_successful
        == total_expected

        and total_failed == 0

    ):

        print(
            "✓ 2011 FULL-YEAR "
            "30-CONCURRENT-REQUEST "
            "TEST PASSED"
        )

    else:

        print(
            "✗ 2011 TEST FAILED — "
            "CHECK MISSING/FAILED FILES"
        )

    print("=" * 100)

    # ========================================================
    # FINAL LOCATION
    # ========================================================

    print()

    print(
        "All OISST files are stored under:"
    )

    print(
        f"  {PRODUCTION_ROOT}"
    )

    print()

    print(
        "Files are organized by month:"
    )

    for month in range(1, 13):

        month_dir = (
            PRODUCTION_ROOT
            / f"{month:02d}"
        )

        print(
            f"  {month_dir}"
        )

    print()

    print("=" * 100)

    print("DONE")

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if _name_ == "_main_":

    main()