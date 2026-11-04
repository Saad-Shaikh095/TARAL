"""
OSCAR v2.0 FINAL - Full Historical Downloader
==============================================

Dataset:
    Ocean Surface Current Analyses Real-time (OSCAR)
    Final 0.25 Degree, Version 2.0

Collection:
    OSCAR_L4_OC_FINAL_V2.0

Period:
    2012-01-01 through 2024-12-31

Expected granules:
    4,749 daily files

Download method:
    1. Query NASA CMR for all OSCAR granules in the date range.
    2. Extract protected PO.DAAC archive URLs.
    3. Authenticate using Earthdata Login via earthaccess.
    4. Download files concurrently.
    5. Resume interrupted downloads when possible.
    6. Retry transient failures.
    7. Skip already-complete files.
    8. Maintain a persistent download manifest.
    9. Produce a final completeness report.

IMPORTANT:
    This downloads the COMPLETE GLOBAL native OSCAR files.

    It does NOT:
        - subset spatially
        - regrid
        - change precision
        - modify NetCDF files
        - discard variables

That processing will happen later.

Requirements:
    pip install earthaccess requests

Authentication:
    Earthdata Login is required.

Recommended Python:
    Python 3.11/3.12/3.13

Tested acquisition approach:
    earthaccess.get_requests_https_session()
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import earthaccess
import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

COLLECTION_CONCEPT_ID = "C2098858642-POCLOUD"
COLLECTION_SHORT_NAME = "OSCAR_L4_OC_FINAL_V2.0"

START_DATE = "2012-01-01T00:00:00Z"
END_DATE = "2024-12-31T23:59:59Z"

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar"
)

MANIFEST_PATH = OUTPUT_DIR / "oscar_download_manifest.json"
FAILED_PATH = OUTPUT_DIR / "oscar_failed_downloads.json"
LOG_PATH = OUTPUT_DIR / "oscar_download.log"

# Number of simultaneous downloads.
#
# Start conservatively.
# 8 is a good initial value for a normal broadband connection.
MAX_WORKERS = 8

# HTTP behavior
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300

# Retry behavior
MAX_RETRIES = 6
INITIAL_RETRY_DELAY = 2.0
MAX_RETRY_DELAY = 60.0

# Chunk used while writing files.
CHUNK_SIZE = 1024 * 1024  # 1 MB

# How often progress is printed.
PROGRESS_EVERY = 10


# ============================================================================
# GLOBAL STATE
# ============================================================================

print_lock = Lock()
manifest_lock = Lock()

manifest = {}
failed_downloads = []


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Granule:
    producer_granule_id: str
    concept_id: str
    url: str
    filename: str
    start_time: str
    end_time: str


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    """
    Thread-safe console + file logging.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    line = f"[{timestamp}] {message}"

    with print_lock:
        print(line, flush=True)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================================
# UTILITIES
# ============================================================================

def format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(value)

    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:,.2f} {unit}"

        value /= 1024.0

    return f"{value:,.2f} TB"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = minutes / 60

    if hours < 24:
        return f"{hours:.1f}h"

    return f"{hours / 24:.1f}d"


def atomic_json_write(path: Path, data) -> None:
    """
    Write JSON atomically so an interrupted process does not corrupt
    the manifest.
    """

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(temp_path, path)


# ============================================================================
# MANIFEST
# ============================================================================

def load_manifest() -> None:
    global manifest

    if not MANIFEST_PATH.exists():
        manifest = {}
        return

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        log(
            f"Loaded existing manifest: "
            f"{len(manifest):,} entries"
        )

    except Exception as exc:
        log(
            f"WARNING: Could not load manifest: {exc}"
        )

        manifest = {}


def save_manifest() -> None:
    with manifest_lock:
        atomic_json_write(
            MANIFEST_PATH,
            manifest,
        )


def mark_manifest(
    filename: str,
    status: str,
    **extra,
) -> None:

    with manifest_lock:

        entry = manifest.setdefault(
            filename,
            {},
        )

        entry.update(
            {
                "status": status,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                **extra,
            }
        )

        atomic_json_write(
            MANIFEST_PATH,
            manifest,
        )


# ============================================================================
# CMR GRANULE DISCOVERY
# ============================================================================

def extract_download_url(granule: dict) -> Optional[str]:
    """
    Extract the protected PO.DAAC data URL from a CMR granule.

    We deliberately select the actual data URL rather than metadata,
    OPeNDAP, browse, or checksum links.
    """

    links = granule.get("links", [])

    candidates = []

    for link in links:

        href = link.get("href")

        if not href:
            continue

        rel = str(link.get("rel", "")).lower()

        title = str(link.get("title", "")).lower()

        # Actual PO.DAAC archive URL.
        if (
            "archive.podaac.earthdata.nasa.gov" in href
            and "protected" in href
            and href.endswith(".nc")
        ):
            return href

        # Keep other likely data links as fallback candidates.
        if (
            "data" in rel
            or "download" in rel
            or "data" in title
        ):
            candidates.append(href)

    # Fallback.
    for href in candidates:

        if href.endswith(".nc"):
            return href

    return None


def discover_granules() -> list[Granule]:
    """
    Query CMR using pagination.

    CMR granule searches are paginated, so we never assume that
    one request contains the entire 13-year dataset.
    """

    base_url = (
        "https://cmr.earthdata.nasa.gov"
        "/search/granules.json"
    )

    all_granules: list[Granule] = []

    page_num = 1
    page_size = 2000

    log("Starting CMR granule discovery...")

    while True:

        params = {
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "temporal": f"{START_DATE},{END_DATE}",
            "page_num": page_num,
            "page_size": page_size,
        }

        log(
            f"CMR page {page_num}: "
            f"requesting up to {page_size:,} granules..."
        )

        response = requests.get(
            base_url,
            params=params,
            timeout=(30, 120),
        )

        response.raise_for_status()

        data = response.json()

        items = data.get("feed", {}).get("entry", [])

        if not items:
            break

        for item in items:

            producer_id = item.get(
                "producer_granule_id",
                ""
            )

            concept_id = item.get(
                "id",
                ""
            )

            url = extract_download_url(item)

            if not url:
                raise RuntimeError(
                    "Could not find protected .nc URL for "
                    f"granule {producer_id}"
                )

            time_start = item.get(
                "time_start",
                "",
            )

            time_end = item.get(
                "time_end",
                "",
            )

            filename = Path(
                url.split("?")[0]
            ).name

            all_granules.append(
                Granule(
                    producer_granule_id=producer_id,
                    concept_id=concept_id,
                    url=url,
                    filename=filename,
                    start_time=time_start,
                    end_time=time_end,
                )
            )

        log(
            f"CMR page {page_num}: "
            f"{len(items):,} granules"
        )

        if len(items) < page_size:
            break

        page_num += 1

    # Remove accidental duplicates.
    unique = {}

    for granule in all_granules:
        unique[granule.filename] = granule

    granules = list(unique.values())

    # Sort chronologically by filename.
    granules.sort(
        key=lambda g: g.filename
    )

    log(
        f"CMR discovery complete: "
        f"{len(granules):,} unique granules"
    )

    return granules


# ============================================================================
# AUTHENTICATION
# ============================================================================

def authenticate() -> None:
    """
    Authenticate with NASA Earthdata.

    If credentials are already configured in earthaccess,
    this will use them.

    Otherwise Earthdata Login will prompt interactively.
    """

    log("Authenticating with NASA Earthdata...")

    auth = earthaccess.login(
        persist=True
    )

    if auth is None:
        raise RuntimeError(
            "Earthdata authentication failed."
        )

    log("Earthdata authentication successful.")


# ============================================================================
# DOWNLOAD
# ============================================================================

def download_one(
    granule: Granule,
) -> tuple[str, str, Optional[str]]:

    output_path = OUTPUT_DIR / granule.filename

    # ----------------------------------------------------------------------
    # Already complete according to manifest
    # ----------------------------------------------------------------------

    existing_entry = manifest.get(
        granule.filename,
        {},
    )

    if (
        existing_entry.get("status") == "complete"
        and output_path.exists()
        and output_path.stat().st_size > 0
    ):
        return (
            granule.filename,
            "skipped",
            None,
        )

    # ----------------------------------------------------------------------
    # Retry loop
    # ----------------------------------------------------------------------

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            # Each worker gets its own authenticated requests session.
            session = (
                earthaccess
                .get_requests_https_session()
            )

            existing_size = (
                output_path.stat().st_size
                if output_path.exists()
                else 0
            )

            headers = {}

            if existing_size > 0:
                headers["Range"] = (
                    f"bytes={existing_size}-"
                )

            log(
                f"START {granule.filename} "
                f"(attempt {attempt}/{MAX_RETRIES}, "
                f"resume={format_bytes(existing_size)})"
            )

            response = session.get(
                granule.url,
                headers=headers,
                stream=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT,
                ),
            )

            # --------------------------------------------------------------
            # Resume behavior
            # --------------------------------------------------------------

            if existing_size > 0:

                if response.status_code == 206:
                    mode = "ab"

                elif response.status_code == 200:
                    # Server ignored Range.
                    # Restart safely rather than corrupting the file.
                    log(
                        f"{granule.filename}: "
                        f"server ignored Range request; "
                        f"restarting download."
                    )

                    existing_size = 0
                    mode = "wb"

                else:
                    response.raise_for_status()
                    mode = "wb"

            else:

                response.raise_for_status()
                mode = "wb"

            # --------------------------------------------------------------
            # Expected total size
            # --------------------------------------------------------------

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length:
                content_length = int(
                    content_length
                )

                if mode == "ab":
                    expected_total = (
                        existing_size
                        + content_length
                    )
                else:
                    expected_total = content_length
            else:
                expected_total = None

            # --------------------------------------------------------------
            # Write file
            # --------------------------------------------------------------

            downloaded_this_attempt = 0

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with output_path.open(mode) as f:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if not chunk:
                        continue

                    f.write(chunk)

                    downloaded_this_attempt += len(
                        chunk
                    )

            final_size = output_path.stat().st_size

            # --------------------------------------------------------------
            # Verify size if server supplied it
            # --------------------------------------------------------------

            if (
                expected_total is not None
                and final_size != expected_total
            ):
                raise IOError(
                    f"Incomplete download: "
                    f"expected "
                    f"{format_bytes(expected_total)}, "
                    f"got "
                    f"{format_bytes(final_size)}"
                )

            # --------------------------------------------------------------
            # Basic NetCDF sanity check
            # --------------------------------------------------------------

            if final_size < 1024:
                raise IOError(
                    "Downloaded file is suspiciously small."
                )

            # NetCDF4/HDF5 files begin with the HDF5 signature.
            with output_path.open("rb") as f:
                signature = f.read(8)

            if signature != b"\x89HDF\r\n\x1a\n":
                raise IOError(
                    "File does not have a valid "
                    "NetCDF4/HDF5 signature."
                )

            # --------------------------------------------------------------
            # Mark complete
            # --------------------------------------------------------------

            mark_manifest(
                granule.filename,
                "complete",
                producer_granule_id=(
                    granule.producer_granule_id
                ),
                concept_id=granule.concept_id,
                url=granule.url,
                start_time=granule.start_time,
                end_time=granule.end_time,
                local_path=str(output_path),
                size_bytes=final_size,
            )

            log(
                f"OK    {granule.filename} "
                f"({format_bytes(final_size)})"
            )

            return (
                granule.filename,
                "complete",
                None,
            )

        except Exception as exc:

            last_error = str(exc)

            log(
                f"ERROR {granule.filename}: "
                f"{last_error}"
            )

            # Don't immediately retry if this is the last attempt.
            if attempt >= MAX_RETRIES:
                break

            delay = min(
                INITIAL_RETRY_DELAY
                * (2 ** (attempt - 1)),
                MAX_RETRY_DELAY,
            )

            log(
                f"RETRY {granule.filename} "
                f"in {delay:.1f}s"
            )

            time.sleep(delay)

    # ----------------------------------------------------------------------
    # Permanent failure
    # ----------------------------------------------------------------------

    mark_manifest(
        granule.filename,
        "failed",
        producer_granule_id=(
            granule.producer_granule_id
        ),
        concept_id=granule.concept_id,
        url=granule.url,
        start_time=granule.start_time,
        end_time=granule.end_time,
        local_path=str(output_path),
        error=last_error,
    )

    return (
        granule.filename,
        "failed",
        last_error,
    )


# ============================================================================
# DOWNLOAD MANAGER
# ============================================================================

def download_all(
    granules: list[Granule],
) -> None:

    total = len(granules)

    complete = 0
    skipped = 0
    failed = 0

    started = time.time()

    log("=" * 70)
    log("STARTING FULL OSCAR DOWNLOAD")
    log("=" * 70)

    log(
        f"Granules : {total:,}"
    )

    log(
        f"Workers  : {MAX_WORKERS}"
    )

    log(
        f"Output   : {OUTPUT_DIR}"
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                download_one,
                granule,
            ): granule
            for granule in granules
        }

        for index, future in enumerate(
            concurrent.futures.as_completed(
                futures
            ),
            start=1,
        ):

            filename, status, error = (
                future.result()
            )

            if status == "complete":
                complete += 1

            elif status == "skipped":
                skipped += 1

            else:
                failed += 1

                failed_downloads.append(
                    {
                        "filename": filename,
                        "error": error,
                    }
                )

            if (
                index % PROGRESS_EVERY == 0
                or index == total
            ):

                elapsed = time.time() - started

                rate = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    total - index
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                log(
                    "PROGRESS "
                    f"{index:,}/{total:,} "
                    f"({index / total * 100:.2f}%) | "
                    f"complete={complete:,} | "
                    f"skipped={skipped:,} | "
                    f"failed={failed:,} | "
                    f"rate={rate:.2f}/s | "
                    f"ETA={format_seconds(eta)}"
                )

    atomic_json_write(
        FAILED_PATH,
        failed_downloads,
    )

    elapsed = time.time() - started

    log("=" * 70)
    log("DOWNLOAD PHASE COMPLETE")
    log("=" * 70)

    log(
        f"Total    : {total:,}"
    )

    log(
        f"Complete : {complete:,}"
    )

    log(
        f"Skipped  : {skipped:,}"
    )

    log(
        f"Failed   : {failed:,}"
    )

    log(
        f"Elapsed  : {format_seconds(elapsed)}"
    )

    log(
        f"Failed manifest: {FAILED_PATH}"
    )


# ============================================================================
# FINAL AUDIT
# ============================================================================

def final_audit(
    granules: list[Granule],
) -> None:

    log("=" * 70)
    log("FINAL COMPLETENESS AUDIT")
    log("=" * 70)

    missing = []
    zero_size = []
    invalid_signature = []

    total_bytes = 0

    for granule in granules:

        path = OUTPUT_DIR / granule.filename

        if not path.exists():
            missing.append(
                granule.filename
            )
            continue

        size = path.stat().st_size

        if size <= 0:
            zero_size.append(
                granule.filename
            )
            continue

        total_bytes += size

        try:
            with path.open("rb") as f:
                signature = f.read(8)

            if signature != b"\x89HDF\r\n\x1a\n":
                invalid_signature.append(
                    granule.filename
                )

        except Exception:
            invalid_signature.append(
                granule.filename
            )

    expected = len(granules)

    present = (
        expected
        - len(missing)
        - len(zero_size)
    )

    log(
        f"Expected files       : {expected:,}"
    )

    log(
        f"Present files        : {present:,}"
    )

    log(
        f"Missing files        : {len(missing):,}"
    )

    log(
        f"Zero-size files      : {len(zero_size):,}"
    )

    log(
        f"Invalid signatures   : "
        f"{len(invalid_signature):,}"
    )

    log(
        f"Total disk usage     : "
        f"{format_bytes(total_bytes)}"
    )

    if missing:
        log(
            "Missing files:"
        )

        for filename in missing[:50]:
            log(
                f"  {filename}"
            )

    if zero_size:
        log(
            "Zero-size files:"
        )

        for filename in zero_size[:50]:
            log(
                f"  {filename}"
            )

    if invalid_signature:
        log(
            "Invalid NetCDF signatures:"
        )

        for filename in invalid_signature[:50]:
            log(
                f"  {filename}"
            )

    if (
        not missing
        and not zero_size
        and not invalid_signature
    ):
        log(
            "SUCCESS: All OSCAR granules passed "
            "the basic completeness audit."
        )

    else:
        log(
            "WARNING: Dataset is incomplete. "
            "Review the manifest and failed-download file."
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print("OSCAR v2.0 FINAL")
    print("FULL 2012-2024 DOWNLOAD")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Disk space warning
    # ----------------------------------------------------------------------

    usage = os.statvfs(
        OUTPUT_DIR.drive + "\\"
    ) if os.name != "nt" else None

    # We deliberately don't hard-fail on disk space because Windows
    # statvfs is not consistently available. The user should verify
    # available space before starting.

    log(
        f"Output directory: {OUTPUT_DIR}"
    )

    log(
        "Expected source size: approximately 150 GB"
    )

    log(
        "Make sure sufficient disk space is available."
    )

    # ----------------------------------------------------------------------
    # Load previous state
    # ----------------------------------------------------------------------

    load_manifest()

    # ----------------------------------------------------------------------
    # Authenticate
    # ----------------------------------------------------------------------

    authenticate()

    # ----------------------------------------------------------------------
    # Discover
    # ----------------------------------------------------------------------

    granules = discover_granules()

    if not granules:
        raise RuntimeError(
            "CMR returned zero OSCAR granules."
        )

    # ----------------------------------------------------------------------
    # Safety check
    # ----------------------------------------------------------------------

    log(
        f"Discovered {len(granules):,} granules."
    )

    if len(granules) != 4749:

        log(
            "WARNING: Expected approximately "
            "4,749 daily granules for "
            "2012-01-01 through 2024-12-31."
        )

        log(
            f"CMR returned {len(granules):,}."
        )

        log(
            "The downloader will NOT automatically "
            "abort. Review the count before proceeding."
        )

    # ----------------------------------------------------------------------
    # Save discovered granule list
    # ----------------------------------------------------------------------

    granule_manifest_path = (
        OUTPUT_DIR
        / "oscar_cmr_granules_2012_2024.json"
    )

    atomic_json_write(
        granule_manifest_path,
        [
            asdict(g)
            for g in granules
        ],
    )

    log(
        f"Saved CMR granule manifest: "
        f"{granule_manifest_path}"
    )

    # ----------------------------------------------------------------------
    # Download
    # ----------------------------------------------------------------------

    download_all(
        granules
    )

    # ----------------------------------------------------------------------
    # Final audit
    # ----------------------------------------------------------------------

    final_audit(
        granules
    )

    print()
    print("=" * 70)
    print("OSCAR DOWNLOAD FINISHED")
    print("=" * 70)
    print()
    print(
        f"Raw data directory:\n{OUTPUT_DIR}"
    )
    print()
    print(
        f"Download manifest:\n{MANIFEST_PATH}"
    )
    print()
    print(
        f"Failed downloads:\n{FAILED_PATH}"
    )
    print()
    print(
        f"Log:\n{LOG_PATH}"
    )
    print()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        log(
            "INTERRUPTED BY USER."
        )

        log(
            "Already completed files are safe. "
            "Run the script again to resume."
        )

        sys.exit(130)

    except Exception as exc:

        log(
            f"FATAL ERROR: {exc}"
        )

        sys.exit(1)