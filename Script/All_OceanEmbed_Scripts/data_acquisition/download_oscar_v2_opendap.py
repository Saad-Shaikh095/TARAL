from pathlib import Path
from getpass import getpass
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import os
import sys
import time

import earthaccess
import numpy as np
import requests
import xarray as xr


# ============================================================
# OSCAR V2 PRODUCTION DOWNLOADER
# ============================================================
#
# Dataset:
#   OSCAR L4 OC FINAL v2.0
#
# Period:
#   2012-01-01 -> 2024-12-31
#
# Variables:
#   u = zonal surface current
#   v = meridional surface current
#
# Native OSCAR grid:
#   0.25 degree
#
# Requested region:
#   Longitude: 45.00E -> 105.00E
#   Latitude : 5.25N  -> 29.75N
#
# Native subset:
#   longitude = 241
#   latitude  = 99
#
# Acquisition method:
#   Authenticated NASA OPeNDAP
#   Server-side spatial + variable subsetting
#
# IMPORTANT:
#   This script performs ACQUISITION only.
#
#   It does NOT:
#     - regrid
#     - interpolate
#     - change resolution
#     - convert float64 -> float32
#     - calculate speed/direction
#     - merge days
#
# Those operations belong to the processing layer.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_CONCEPT_ID = (
    "C2098858642-POCLOUD"
)

START_DATE = datetime(
    2012,
    1,
    1,
).date()

END_DATE = datetime(
    2024,
    12,
    31,
).date()


# ------------------------------------------------------------
# OSCAR native subset
# ------------------------------------------------------------

LON_START = 180
LON_END = 420

LAT_START = 380
LAT_END = 478

EXPECTED_TIME = 1
EXPECTED_LON = LON_END - LON_START + 1
EXPECTED_LAT = LAT_END - LAT_START + 1


# ------------------------------------------------------------
# Variables
# ------------------------------------------------------------

REQUESTED_VARIABLES = (
    "u",
    "v",
)


# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\data\raw\oscar_v2"
)

MANIFEST_FILE = (
    OUTPUT_DIR /
    "oscar_v2_manifest.csv"
)

LOG_FILE = (
    OUTPUT_DIR /
    "oscar_v2_download.log"
)


# ------------------------------------------------------------
# Download configuration
# ------------------------------------------------------------

# Start conservatively.
#
# NASA server-side OPeNDAP requests are relatively small,
# so 4 workers should already provide useful throughput
# without unnecessarily hammering the service.
#
# Increase later only if the service remains stable.
MAX_WORKERS = 4

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300

MAX_RETRIES = 5

RETRY_BASE_SECONDS = 5
RETRY_MAX_SECONDS = 120

CHUNK_SIZE = 1024 * 1024


# ------------------------------------------------------------
# Validation configuration
# ------------------------------------------------------------

COORD_ATOL = 1e-10
COORD_RTOL = 0.0


# ============================================================
# GLOBAL COUNTERS
# ============================================================

EXPECTED_DAYS = (
    END_DATE - START_DATE
).days + 1


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log(message: str) -> None:

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = (
        f"[{timestamp}] {message}"
    )

    print(line, flush=True)

    try:
        with open(
            LOG_FILE,
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                line + "\n"
            )

    except Exception:
        pass


def print_header(title: str) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def date_range(start_date, end_date):

    current = start_date

    while current <= end_date:

        yield current

        from datetime import timedelta

        current += timedelta(days=1)


def expected_dates():

    return list(
        date_range(
            START_DATE,
            END_DATE,
        )
    )


def granule_name(date_value) -> str:

    return (
        "oscar_currents_final_"
        f"{date_value:%Y%m%d}"
    )


def output_file_for_date(date_value) -> Path:

    year_dir = (
        OUTPUT_DIR /
        f"{date_value:%Y}"
    )

    return (
        year_dir /
        f"{granule_name(date_value)}_uv.nc"
    )


def temporary_file_for_date(date_value) -> Path:

    output_file = (
        output_file_for_date(
            date_value
        )
    )

    return output_file.with_suffix(
        ".nc.part"
    )


def opendap_url(granule: str) -> str:

    return (
        "https://opendap.earthdata.nasa.gov/"
        "collections/"
        f"{COLLECTION_CONCEPT_ID}/"
        "granules/"
        f"{granule}"
    )


def build_constraint() -> str:

    return (
        f"/u[0][{LON_START}:{LON_END}]"
        f"[{LAT_START}:{LAT_END}];"
        f"/v[0][{LON_START}:{LON_END}]"
        f"[{LAT_START}:{LAT_END}];"
        f"/lon[{LON_START}:{LON_END}];"
        f"/lat[{LAT_START}:{LAT_END}]"
    )


# ============================================================
# EARTHDATA AUTHENTICATION
# ============================================================

def authenticate_earthdata():

    print_header(
        "EARTHDATA AUTHENTICATION"
    )

    print()
    print(
        "Enter your NASA Earthdata Login credentials."
    )
    print()

    username = input(
        "Earthdata username:\n> "
    ).strip()

    if not username:

        raise RuntimeError(
            "Earthdata username cannot be empty."
        )

    print()

    password = getpass(
        "Earthdata password:\n> "
    )

    if not password:

        raise RuntimeError(
            "Earthdata password cannot be empty."
        )

    os.environ[
        "EARTHDATA_USERNAME"
    ] = username

    os.environ[
        "EARTHDATA_PASSWORD"
    ] = password

    print()
    print(
        "Authenticating with Earthdata..."
    )

    auth = earthaccess.login(
        strategy="environment",
        persist=False,
    )

    if not auth:

        raise RuntimeError(
            "Earthdata authentication failed."
        )

    print(
        "Authentication OK."
    )

    return auth


# ============================================================
# MANIFEST
# ============================================================

MANIFEST_FIELDS = [
    "date",
    "granule",
    "status",
    "output",
    "bytes",
    "elapsed_seconds",
    "attempts",
    "http_status",
    "error",
    "completed_at",
]


def create_manifest_if_needed():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if MANIFEST_FILE.exists():

        return

    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()


def load_manifest():

    completed = {}

    if not MANIFEST_FILE.exists():

        return completed

    with open(
        MANIFEST_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:

            date_value = row.get(
                "date",
                "",
            )

            status = row.get(
                "status",
                "",
            )

            if (
                date_value
                and status == "SUCCESS"
            ):

                completed[
                    date_value
                ] = row

    return completed


def append_manifest(row):

    with open(
        MANIFEST_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writerow(row)


# ============================================================
# NETCDF VALIDATION
# ============================================================

def validate_downloaded_file(
    file_path: Path,
    expected_date,
):

    if not file_path.exists():

        return (
            False,
            "file does not exist",
        )

    file_size = file_path.stat().st_size

    if file_size <= 0:

        return (
            False,
            "file is empty",
        )

    try:

        ds = xr.open_dataset(
            file_path,
            decode_cf=True,
            mask_and_scale=True,
        )

    except Exception as exc:

        return (
            False,
            f"NetCDF open failed: {exc}",
        )

    try:

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        required = {
            "u",
            "v",
            "lon",
            "lat",
        }

        missing = (
            required -
            set(ds.variables)
        )

        if missing:

            return (
                False,
                "missing variables: "
                f"{sorted(missing)}",
            )

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        if ds.sizes.get("time") != 1:

            return (
                False,
                "time dimension is not 1",
            )

        if (
            ds.sizes.get("longitude")
            != EXPECTED_LON
        ):

            return (
                False,
                "longitude dimension is "
                f"{ds.sizes.get('longitude')}",
            )

        if (
            ds.sizes.get("latitude")
            != EXPECTED_LAT
        ):

            return (
                False,
                "latitude dimension is "
                f"{ds.sizes.get('latitude')}",
            )

        # ----------------------------------------------------
        # Dimensions of u/v
        # ----------------------------------------------------

        expected_dims = (
            "time",
            "longitude",
            "latitude",
        )

        if ds["u"].dims != expected_dims:

            return (
                False,
                f"unexpected u dims: "
                f"{ds['u'].dims}",
            )

        if ds["v"].dims != expected_dims:

            return (
                False,
                f"unexpected v dims: "
                f"{ds['v'].dims}",
            )

        # ----------------------------------------------------
        # Coordinates
        # ----------------------------------------------------

        lon = np.asarray(
            ds["lon"].values
        )

        lat = np.asarray(
            ds["lat"].values
        )

        if len(lon) != EXPECTED_LON:

            return (
                False,
                "unexpected longitude length",
            )

        if len(lat) != EXPECTED_LAT:

            return (
                False,
                "unexpected latitude length",
            )

        if not np.allclose(
            lon[0],
            45.0,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                f"unexpected first longitude: "
                f"{lon[0]}",
            )

        if not np.allclose(
            lon[-1],
            105.0,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                f"unexpected last longitude: "
                f"{lon[-1]}",
            )

        if not np.allclose(
            lat[0],
            5.25,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                f"unexpected first latitude: "
                f"{lat[0]}",
            )

        if not np.allclose(
            lat[-1],
            29.75,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                f"unexpected last latitude: "
                f"{lat[-1]}",
            )

        if not np.allclose(
            np.diff(lon),
            0.25,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                "longitude spacing is not 0.25",
            )

        if not np.allclose(
            np.diff(lat),
            0.25,
            atol=COORD_ATOL,
            rtol=COORD_RTOL,
        ):

            return (
                False,
                "latitude spacing is not 0.25",
            )

        # ----------------------------------------------------
        # Data arrays
        # ----------------------------------------------------

        u = np.asarray(
            ds["u"].isel(time=0).values
        )

        v = np.asarray(
            ds["v"].isel(time=0).values
        )

        expected_shape = (
            EXPECTED_LON,
            EXPECTED_LAT,
        )

        if u.shape != expected_shape:

            return (
                False,
                f"unexpected u shape: "
                f"{u.shape}",
            )

        if v.shape != expected_shape:

            return (
                False,
                f"unexpected v shape: "
                f"{v.shape}",
            )

        # ----------------------------------------------------
        # Date verification
        # ----------------------------------------------------

        if "time" in ds:

            try:

                actual_time = (
                    ds["time"]
                    .values[0]
                )

                actual_date = (
                    np.datetime_as_string(
                        actual_time,
                        unit="D",
                    )
                )

                expected_string = (
                    expected_date.isoformat()
                )

                if actual_date != expected_string:

                    return (
                        False,
                        "time mismatch: "
                        f"{actual_date} != "
                        f"{expected_string}",
                    )

            except Exception:
                # Do not reject an otherwise valid file
                # solely because of unusual time decoding.
                pass

        return (
            True,
            "OK",
        )

    finally:

        ds.close()


# ============================================================
# EXISTING FILE CHECK
# ============================================================

def check_existing_file(
    date_value,
):

    output_file = (
        output_file_for_date(
            date_value
        )
    )

    if not output_file.exists():

        return False

    valid, reason = (
        validate_downloaded_file(
            output_file,
            date_value,
        )
    )

    if valid:

        return True

    log(
        f"{date_value}: existing file invalid "
        f"and will be redownloaded: {reason}"
    )

    try:
        output_file.unlink()
    except Exception:
        pass

    return False


# ============================================================
# SINGLE-DAY DOWNLOAD
# ============================================================

def download_one(
    date_value,
    session,
):

    granule = granule_name(
        date_value
    )

    output_file = (
        output_file_for_date(
            date_value
        )
    )

    temp_file = (
        temporary_file_for_date(
            date_value
        )
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Existing valid file
    # --------------------------------------------------------

    if check_existing_file(
        date_value
    ):

        return {
            "date": date_value.isoformat(),
            "granule": granule,
            "status": "SKIPPED",
            "output": str(output_file),
            "bytes": output_file.stat().st_size,
            "elapsed_seconds": 0,
            "attempts": 0,
            "http_status": 200,
            "error": "",
            "completed_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    # --------------------------------------------------------
    # Build request
    # --------------------------------------------------------

    constraint = build_constraint()

    encoded_constraint = quote(
        constraint,
        safe="",
    )

    request_url = (
        opendap_url(granule)
        + ".dap.nc4?dap4.ce="
        + encoded_constraint
    )

    last_error = ""
    last_status = ""

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            # Remove stale partial file.
            if temp_file.exists():

                try:
                    temp_file.unlink()
                except Exception:
                    pass

            response = session.get(
                request_url,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
            )

            last_status = (
                response.status_code
            )

            # ------------------------------------------------
            # HTTP validation
            # ------------------------------------------------

            if response.status_code != 200:

                raise RuntimeError(
                    "HTTP status "
                    f"{response.status_code}"
                )

            content_type = (
                response.headers.get(
                    "Content-Type",
                    "",
                )
            )

            if (
                "netcdf" not in
                content_type.lower()
            ):

                raise RuntimeError(
                    "Unexpected Content-Type: "
                    f"{content_type}"
                )

            # ------------------------------------------------
            # Download to temporary file
            # ------------------------------------------------

            total_bytes = 0

            with open(
                temp_file,
                "wb",
            ) as handle:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if not chunk:
                        continue

                    handle.write(chunk)

                    total_bytes += len(
                        chunk
                    )

            response.close()

            if total_bytes <= 0:

                raise RuntimeError(
                    "Downloaded file is empty."
                )

            # ------------------------------------------------
            # Validate before committing
            # ------------------------------------------------

            valid, reason = (
                validate_downloaded_file(
                    temp_file,
                    date_value,
                )
            )

            if not valid:

                raise RuntimeError(
                    "Downloaded file failed "
                    f"validation: {reason}"
                )

            # ------------------------------------------------
            # Atomic commit
            # ------------------------------------------------

            if output_file.exists():

                output_file.unlink()

            temp_file.replace(
                output_file
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            log(
                f"{date_value} SUCCESS "
                f"{total_bytes:,} bytes "
                f"attempt={attempt} "
                f"time={elapsed:.2f}s"
            )

            return {
                "date": date_value.isoformat(),
                "granule": granule,
                "status": "SUCCESS",
                "output": str(output_file),
                "bytes": total_bytes,
                "elapsed_seconds": round(
                    elapsed,
                    3,
                ),
                "attempts": attempt,
                "http_status": last_status,
                "error": "",
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        except (
            requests.RequestException,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:

            last_error = str(exc)

            log(
                f"{date_value} attempt "
                f"{attempt}/{MAX_RETRIES} FAILED: "
                f"{last_error}"
            )

            if attempt < MAX_RETRIES:

                delay = min(
                    RETRY_BASE_SECONDS
                    * (
                        2 ** (
                            attempt - 1
                        )
                    ),
                    RETRY_MAX_SECONDS,
                )

                time.sleep(delay)

    # --------------------------------------------------------
    # Final failure
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if temp_file.exists():

        try:
            temp_file.unlink()
        except Exception:
            pass

    log(
        f"{date_value} FINAL FAILURE "
        f"after {MAX_RETRIES} attempts"
    )

    return {
        "date": date_value.isoformat(),
        "granule": granule,
        "status": "FAILED",
        "output": str(output_file),
        "bytes": 0,
        "elapsed_seconds": round(
            elapsed,
            3,
        ),
        "attempts": MAX_RETRIES,
        "http_status": last_status,
        "error": last_error,
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }


# ============================================================
# FINAL AUDIT
# ============================================================

def final_audit():

    print_header(
        "FINAL OSCAR V2 ACQUISITION AUDIT"
    )

    dates = expected_dates()

    missing = []
    invalid = []
    valid = []

    for index, date_value in enumerate(
        dates,
        start=1,
    ):

        output_file = (
            output_file_for_date(
                date_value
            )
        )

        if not output_file.exists():

            missing.append(
                date_value
            )

            continue

        ok, reason = (
            validate_downloaded_file(
                output_file,
                date_value,
            )
        )

        if ok:

            valid.append(
                date_value
            )

        else:

            invalid.append(
                (
                    date_value,
                    reason,
                )
            )

        if (
            index % 250 == 0
            or index == len(dates)
        ):

            print(
                f"Audit: "
                f"{index:,}/{len(dates):,}",
                flush=True,
            )

    print()
    print(
        f"Expected days : "
        f"{len(dates):,}"
    )

    print(
        f"Valid files   : "
        f"{len(valid):,}"
    )

    print(
        f"Missing files : "
        f"{len(missing):,}"
    )

    print(
        f"Invalid files : "
        f"{len(invalid):,}"
    )

    if missing:

        print()
        print("Missing dates:")

        for date_value in missing[
            :50
        ]:

            print(
                f"  {date_value}"
            )

        if len(missing) > 50:

            print(
                f"  ... and "
                f"{len(missing) - 50} more"
            )

    if invalid:

        print()
        print("Invalid files:")

        for date_value, reason in invalid[
            :50
        ]:

            print(
                f"  {date_value}: "
                f"{reason}"
            )

        if len(invalid) > 50:

            print(
                f"  ... and "
                f"{len(invalid) - 50} more"
            )

    print()

    if (
        len(valid) == len(dates)
        and not missing
        and not invalid
    ):

        print(
            "FINAL AUDIT: PASS"
        )

        return True

    print(
        "FINAL AUDIT: FAIL"
    )

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OSCAR V2 PRODUCTION ACQUISITION"
    )

    print()
    print("Dataset:")
    print(
        "  OSCAR L4 OC FINAL v2.0"
    )

    print()
    print("Period:")
    print(
        f"  {START_DATE} -> {END_DATE}"
    )

    print()
    print(
        f"Expected days: "
        f"{EXPECTED_DAYS:,}"
    )

    print()
    print("Variables:")
    print("  u")
    print("  v")

    print()
    print("Server-side subset:")
    print("  Longitude: 45.00E -> 105.00E")
    print("  Latitude : 5.25N -> 29.75N")

    print()
    print("Native subset dimensions:")
    print(
        f"  longitude: {EXPECTED_LON}"
    )
    print(
        f"  latitude : {EXPECTED_LAT}"
    )
    print(
        f"  time     : {EXPECTED_TIME}"
    )

    print()
    print("Output:")
    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        f"Workers: {MAX_WORKERS}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    create_manifest_if_needed()

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    authenticate_earthdata()

    # --------------------------------------------------------
    # Authenticated session
    # --------------------------------------------------------

    print_header(
        "CREATING AUTHENTICATED HTTP SESSION"
    )

    session = (
        earthaccess.get_requests_https_session()
    )

    if session is None:

        raise RuntimeError(
            "Could not create authenticated "
            "Earthdata HTTP session."
        )

    print(
        "Authenticated HTTP session created."
    )

    # --------------------------------------------------------
    # Existing files
    # --------------------------------------------------------

    print_header(
        "SCANNING EXISTING ACQUISITION"
    )

    dates = expected_dates()

    pending_dates = []
    existing_valid = 0

    for date_value in dates:

        if check_existing_file(
            date_value
        ):

            existing_valid += 1

        else:

            pending_dates.append(
                date_value
            )

    print()
    print(
        f"Expected days      : "
        f"{len(dates):,}"
    )

    print(
        f"Already valid      : "
        f"{existing_valid:,}"
    )

    print(
        f"Remaining to fetch : "
        f"{len(pending_dates):,}"
    )

    if not pending_dates:

        print()
        print(
            "All files already exist "
            "and passed validation."
        )

        final_audit()

        return

    # --------------------------------------------------------
    # Acquisition
    # --------------------------------------------------------

    print_header(
        "STARTING OSCAR V2 ACQUISITION"
    )

    acquisition_start = (
        time.perf_counter()
    )

    success_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # Thread pool
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                download_one,
                date_value,
                session,
            ): date_value
            for date_value in pending_dates
        }

        completed = 0

        try:

            for future in as_completed(
                futures
            ):

                date_value = (
                    futures[future]
                )

                completed += 1

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    result = {
                        "date":
                            date_value.isoformat(),
                        "granule":
                            granule_name(
                                date_value
                            ),
                        "status":
                            "FAILED",
                        "output":
                            str(
                                output_file_for_date(
                                    date_value
                                )
                            ),
                        "bytes":
                            0,
                        "elapsed_seconds":
                            0,
                        "attempts":
                            0,
                        "http_status":
                            "",
                        "error":
                            f"worker exception: "
                            f"{exc}",
                        "completed_at":
                            datetime.now(
                                timezone.utc
                            ).isoformat(),
                    }

                append_manifest(
                    result
                )

                if result["status"] in (
                    "SUCCESS",
                    "SKIPPED",
                ):

                    success_count += 1

                else:

                    failed_count += 1

                print(
                    f"[{completed:,}/"
                    f"{len(pending_dates):,}] "
                    f"{date_value} "
                    f"{result['status']}",
                    flush=True,
                )

        except KeyboardInterrupt:

            print()
            print(
                "Keyboard interrupt received."
            )

            print(
                "Waiting for active workers "
                "to finish their current requests..."
            )

            raise

    acquisition_time = (
        time.perf_counter()
        - acquisition_start
    )

    print()
    print(
        f"Acquisition elapsed time: "
        f"{acquisition_time / 3600:.2f} hours"
    )

    print(
        f"Successful/skipped: "
        f"{success_count:,}"
    )

    print(
        f"Failed: "
        f"{failed_count:,}"
    )

    # --------------------------------------------------------
    # Final audit
    # --------------------------------------------------------

    audit_pass = final_audit()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print_header(
        "OSCAR V2 ACQUISITION RESULT"
    )

    if audit_pass:

        print()
        print(
            "RESULT: SUCCESS"
        )

        print()
        print(
            "All 4,749 daily OSCAR files "
            "were acquired and validated."
        )

        print()
        print(
            "Acquisition layer is complete."
        )

    else:

        print()
        print(
            "RESULT: INCOMPLETE"
        )

        print()
        print(
            "Some files are missing or invalid."
        )

        print()
        print(
            "Run this same script again to "
            "resume the acquisition."
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("=" * 72)
        print(
            "ACQUISITION INTERRUPTED"
        )
        print("=" * 72)

        print()
        print(
            "Already completed files are safe."
        )

        print(
            "Run the script again to resume."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print(
            "ACQUISITION FAILED"
        )
        print("=" * 72)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        sys.exit(1)