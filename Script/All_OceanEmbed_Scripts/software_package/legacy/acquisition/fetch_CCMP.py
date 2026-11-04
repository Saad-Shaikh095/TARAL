"""
================================================================================
OCEANEMBED CCMP V5
DIRECT PO.DAAC OPeNDAP-IN-THE-CLOUD SUBSET PIPELINE
================================================================================
...
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock, local
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import requests
import xarray as xr


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\OceanEmbed\oceanembed-data")

RAW_ROOT = ROOT / "raw" / "ccmp"
TEMP_ROOT = RAW_ROOT / "_temp_v5_opendap"
PROCESSED_ROOT = RAW_ROOT / "processed"
STATE_ROOT = RAW_ROOT / "_state"
LOG_ROOT = RAW_ROOT / "_logs"

START_DATE = date(2012, 1, 1)
END_DATE = date(2024, 12, 31)

# SAFETY: keep TRUE for the first run.
TEST_MODE = False
TEST_DATE = date(2012, 3, 1)
AUTO_START_FULL_AFTER_TEST = True
PRODUCTION_PASSES = 2

COLLECTION_SHORTNAME = "CCMP_WINDS_10M6HR_L4_V3.1"
COLLECTION_ID = "C2916514952-POCLOUD"

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

RSS_BASE = "https://data.remss.com/ccmp/v03.1"

TARGET_LON_MIN = 45.0
TARGET_LON_MAX = 105.0
TARGET_LAT_MIN = 5.0
TARGET_LAT_MAX = 30.0
TARGET_RESOLUTION = 0.25

NATIVE_BUFFER = 0.25

MAX_WORKERS = 6

CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 300
OPENDAP_TIMEOUT_SECONDS = 10 * 60

MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 4

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
NETCDF_COMPRESSION_LEVEL = 4

KEEP_RAW_AFTER_PROCESSING = False

VALIDATION_ATOL = 1e-6
VALIDATION_RTOL = 1e-6

COORD_TOLERANCE = 1e-5

USER_AGENT = "OceanEmbed-CCMP-V5-OPeNDAP/1.0"

STATE_FILE = STATE_ROOT / "ccmp_v5_opendap_state.json"
LOG_FILE = LOG_ROOT / "ccmp_v5_opendap.log"

STATE_LOCK = Lock()
THREAD_LOCAL = local()

MANIFEST_FILE = TEMP_ROOT / "ccmp_v5_manifest.json"
GRID_FILE = TEMP_ROOT / "ccmp_v5_grid.json"
CMR_PAGE_SIZE = 2000
EXPECTED_DAYS = (END_DATE - START_DATE).days + 1


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("oceanembed_ccmp_v5_opendap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()


# =============================================================================
# HELPERS
# =============================================================================

def human_size(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_granule_name(target_date: date) -> str:
    return f"CCMP_Wind_Analysis_{target_date:%Y%m%d}_V03.1_L4.nc"


def build_rss_url(target_date: date) -> str:
    return (
        f"{RSS_BASE}/"
        f"Y{target_date.year:04d}/"
        f"M{target_date.month:02d}/"
        f"CCMP_Wind_Analysis_{target_date:%Y%m%d}_V03.1_L4.nc"
    )


def build_temp_day_dir(target_date: date) -> Path:
    day_dir = (
        TEMP_ROOT
        / f"{target_date:%Y}"
        / f"{target_date:%m}"
        / f"{target_date:%d}"
    )
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def build_subset_path(target_date: date) -> Path:
    return (
        build_temp_day_dir(target_date)
        / f"{target_date:%Y%m%d}_opendap_subset.nc4"
    )


def build_coordinate_path(target_date: date) -> Path:
    return (
        build_temp_day_dir(target_date)
        / f"{target_date:%Y%m%d}_opendap_coordinates.nc4"
    )


def build_cmr_metadata_path(target_date: date) -> Path:
    return (
        build_temp_day_dir(target_date)
        / f"{target_date:%Y%m%d}_cmr.json"
    )


def build_processed_path(target_date: date) -> Path:
    day_dir = (
        PROCESSED_ROOT
        / f"{target_date:%Y}"
        / f"{target_date:%m}"
    )
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"CCMP_daily_NIO_{target_date:%Y%m%d}.nc"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_complete(target_date: date) -> bool:
    path = build_processed_path(target_date)
    if not path.exists():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def mark_complete(target_date: date, processed_path: Path) -> None:
    with STATE_LOCK:
        if STATE_FILE.exists():
            try:
                with STATE_FILE.open("r", encoding="utf-8") as handle:
                    state = json.load(handle)
            except Exception as exc:
                logger.warning("State file could not be read during commit: %s", exc)
                state = {"completed": {}}
        else:
            state = {"completed": {}}

        if not isinstance(state, dict):
            state = {"completed": {}}
        state.setdefault("completed", {})

        state["completed"][target_date.isoformat()] = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "processed_file": str(processed_path),
            "processed_size_bytes": processed_path.stat().st_size,
            "processed_sha256": file_sha256(processed_path),
            "pipeline_version": "CCMP_V5_OPENDAP",
        }

        atomic_write_json(STATE_FILE, state)


def get_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
            }
        )
        session.trust_env = True
        THREAD_LOCAL.session = session
    return session


def validate_netcdf_magic(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(8)
    if not (magic[:3] == b"CDF" or magic == b"\x89HDF\r\n\x1a\n"):
        raise RuntimeError(f"File is not a recognized NetCDF file: {path}; first bytes={magic!r}")


# =============================================================================
# TARGET GRID
# =============================================================================

def expected_target_grid() -> tuple[np.ndarray, np.ndarray]:
    target_lats = np.arange(
        TARGET_LAT_MIN,
        TARGET_LAT_MAX + TARGET_RESOLUTION / 2,
        TARGET_RESOLUTION,
        dtype=np.float32,
    )
    target_lons = np.arange(
        TARGET_LON_MIN,
        TARGET_LON_MAX + TARGET_RESOLUTION / 2,
        TARGET_RESOLUTION,
        dtype=np.float32,
    )
    return target_lats, target_lons


# =============================================================================
# CMR PARSING
# =============================================================================

def _recursive_find_keys(obj, key, max_depth=5):
    """
    Recursively search for a key in a nested dict/list, returning list of values.
    Does not descend beyond max_depth to avoid excessive recursion.
    """
    results = []
    if max_depth < 0:
        return results
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                results.append(v)
            results.extend(_recursive_find_keys(v, key, max_depth - 1))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_recursive_find_keys(item, key, max_depth - 1))
    return results


def _parse_datetime(value) -> datetime | None:
    """Parse a datetime string or object to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Try common formats
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        # ISO format fallback
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def extract_granule_datetime(umm: dict) -> datetime | None:
    """
    Extract granule start datetime from CMR UMM-JSON.

    Searches TemporalExtents and other temporal fields recursively.
    Returns datetime (naive UTC) or None.
    """
    candidates = []
    # Direct TemporalExtents
    if "umm" in umm:
        root = umm.get("umm", {})
        temporal_extents = root.get("TemporalExtents", [])
        if isinstance(temporal_extents, dict):
            temporal_extents = [temporal_extents]
        if isinstance(temporal_extents, list):
            for extent in temporal_extents:
                if not isinstance(extent, dict):
                    continue
                # Try different possible field names
                range_dt = extent.get("RangeDateTime") or extent.get("RangeDateTimes")
                if isinstance(range_dt, dict):
                    range_dt = [range_dt]
                if isinstance(range_dt, list):
                    for r in range_dt:
                        if isinstance(r, dict):
                            for key in ("BeginningDateTime", "BeginningDate", "StartDateTime", "StartDate"):
                                if key in r:
                                    dt = _parse_datetime(r[key])
                                    if dt:
                                        candidates.append(dt)
                                        break  # take first valid
                # Also directly nested?
                for key in ("BeginningDateTime", "BeginningDate", "StartDateTime", "StartDate"):
                    if key in extent:
                        dt = _parse_datetime(extent[key])
                        if dt:
                            candidates.append(dt)

    # Fallback: search recursively for known temporal keys
    for key in ("BeginningDateTime", "BeginningDate", "StartDateTime", "StartDate"):
        vals = _recursive_find_keys(umm, key, max_depth=6)
        for v in vals:
            dt = _parse_datetime(v)
            if dt:
                candidates.append(dt)

    # Also check meta fields like 'temporal'?
    if not candidates:
        meta = umm.get("meta", {})
        if isinstance(meta, dict):
            for key in ("beginningDateTime", "beginningDate", "startDateTime", "startDate"):
                if key in meta:
                    dt = _parse_datetime(meta[key])
                    if dt:
                        candidates.append(dt)

    for dt in candidates:
        if dt is not None:
            # Convert to UTC naive
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
    return None


def extract_granule_filename(umm: dict) -> str:
    """
    Extract granule filename from CMR UMM-JSON.
    Searches common identifier fields.
    """
    candidates = []
    root = umm.get("umm", {})
    meta = umm.get("meta", {})

    # Direct common fields
    for key in ("GranuleUR", "ProducerGranuleId", "ProducerGranuleID"):
        val = root.get(key)
        if val and isinstance(val, str):
            candidates.append(val)
        val = meta.get(key)
        if val and isinstance(val, str):
            candidates.append(val)

    # DataGranule Identifiers
    identifiers = root.get("DataGranule", {}).get("Identifiers", [])
    if isinstance(identifiers, list):
        for ident in identifiers:
            if isinstance(ident, dict):
                for key in ("Identifier", "identifier"):
                    val = ident.get(key)
                    if val and isinstance(val, str):
                        candidates.append(val)

    # meta.native-id
    native_id = meta.get("native-id")
    if native_id and isinstance(native_id, str):
        candidates.append(native_id)

    # Fallback: search all strings ending with .nc or containing "CCMP"
    for s in candidates:
        if s.endswith(".nc"):
            return s
    for s in candidates:
        if "CCMP" in s:
            return s
    if candidates:
        return candidates[0]
    # Last resort: search recursively for any string that looks like a granule name
    all_strings = _recursive_find_keys(umm, "GranuleUR", max_depth=4)
    for s in all_strings:
        if isinstance(s, str) and s.endswith(".nc"):
            return s
    return ""


def extract_opendap_url(umm: dict) -> str:
    """
    Extract OPeNDAP URL from CMR UMM-JSON RelatedUrls.
    Prefer URLs containing 'opendap.earthdata.nasa.gov'.
    """
    related = umm.get("umm", {}).get("RelatedUrls", [])
    candidates = []
    for item in related:
        if not isinstance(item, dict):
            continue
        url = item.get("URL")
        if not url:
            continue
        subtype = str(item.get("Subtype", "")).upper()
        description = str(item.get("Description", "")).upper()
        if "OPENDAP" in subtype or "OPENDAP" in description or "DAP" in subtype or "DAP" in description:
            candidates.append(url)
    if not candidates:
        # Fallback: look for any URL containing 'opendap'
        for item in related:
            url = item.get("URL", "")
            if url and "opendap" in url.lower():
                candidates.append(url)
    # Prefer Earthdata OPeNDAP endpoint
    for url in candidates:
        if "opendap.earthdata.nasa.gov" in url:
            return url
    if candidates:
        return candidates[0]
    raise RuntimeError("No OPeNDAP URL found in CMR metadata")


def parse_cmr_item(item: dict) -> tuple[date | None, str, str]:
    """
    Parse one CMR UMM-JSON item and return (date, filename, opendap_url).
    If date cannot be extracted, returns None.
    """
    dt = extract_granule_datetime(item)
    if dt is None:
        # Fallback to filename date
        filename = extract_granule_filename(item)
        if filename:
            # Try to extract 8-digit date pattern from filename
            match = re.search(r"(\d{8})", filename)
            if match:
                date_str = match.group(1)
                try:
                    dt = datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    pass
    if dt is None:
        return None, "", ""
    granule_date = dt.date()
    if granule_date < START_DATE or granule_date > END_DATE:
        return None, "", ""
    filename = extract_granule_filename(item)
    try:
        url = extract_opendap_url(item)
    except Exception:
        url = ""
    return granule_date, filename, url


# =============================================================================
# CMR INVENTORY
# =============================================================================

def fetch_cmr_inventory() -> dict[str, str]:
    """
    Fetch/cache all daily CCMP OPeNDAP URLs for 2012-2024 in paged CMR calls.
    Uses robust CMR parsing and returns date->URL mapping.
    """
    if MANIFEST_FILE.exists():
        try:
            with MANIFEST_FILE.open("r", encoding="utf-8") as h:
                cached = json.load(h)
            if (
                isinstance(cached, dict)
                and cached.get("collection_id") == COLLECTION_ID
                and cached.get("start_date") == START_DATE.isoformat()
                and cached.get("end_date") == END_DATE.isoformat()
                and isinstance(cached.get("granules"), dict)
                and len(cached["granules"]) == EXPECTED_DAYS
            ):
                logger.info("Using cached CMR manifest: %d daily granules.", EXPECTED_DAYS)
                return {str(k): str(v) for k, v in cached["granules"].items()}
        except Exception as exc:
            logger.warning("Cached CMR manifest invalid; rebuilding: %s", exc)

    start_iso = f"{START_DATE.isoformat()}T00:00:00Z"
    end_exclusive = END_DATE + timedelta(days=1)
    end_iso = f"{end_exclusive.isoformat()}T00:00:00Z"
    session = get_session()
    all_items = []
    page = 1
    logger.info("Building one CMR inventory for %s through %s...", START_DATE, END_DATE)

    while True:
        params = [
            ("collection_concept_id", COLLECTION_ID),
            ("temporal[]", f"{start_iso},{end_iso}"),
            ("options[temporal][exclude_boundary]", "true"),
            ("page_size", CMR_PAGE_SIZE),
            ("page_num", page),
        ]
        response = session.get(
            CMR_GRANULES_URL,
            params=params,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        hits = int(payload.get("hits", 0) or 0)
        logger.info("CMR inventory page %d: %d items (hits=%d)", page, len(items), hits)
        all_items.extend(items)
        if not items or len(all_items) >= hits or len(items) < CMR_PAGE_SIZE:
            break
        page += 1
        if page > 20:
            raise RuntimeError("CMR pagination exceeded safety limit.")

    # Debug sample if parsing fails
    manifest = {}
    parse_fail_count = 0
    sample_saved = False
    for item in all_items:
        granule_date, filename, url = parse_cmr_item(item)
        if granule_date is None:
            parse_fail_count += 1
            if not sample_saved and parse_fail_count == 1:
                sample_saved = True
                sample_path = TEMP_ROOT / "cmr_debug_sample.json"
                try:
                    atomic_write_json(sample_path, item)
                    logger.warning("Saved first unparseable CMR item to %s", sample_path)
                except Exception as exc:
                    logger.error("Could not save debug sample: %s", exc)
            continue
        day_iso = granule_date.isoformat()
        if day_iso in manifest and manifest[day_iso] != url:
            logger.warning(
                "Duplicate date %s with different URLs: existing=%s, new=%s. Keeping existing.",
                day_iso,
                manifest[day_iso],
                url,
            )
            continue
        manifest[day_iso] = url

    if parse_fail_count > 0:
        logger.warning("Failed to parse %d CMR items out of %d.", parse_fail_count, len(all_items))

    missing = [d.isoformat() for d in date_range(START_DATE, END_DATE) if d.isoformat() not in manifest]
    if missing:
        # Try filename fallback on unparsed items
        logger.info("Attempting filename date fallback for missing days...")
        for item in all_items:
            if granule_date is not None:
                continue
            filename = extract_granule_filename(item)
            if filename:
                match = re.search(r"(\d{8})", filename)
                if match:
                    date_str = match.group(1)
                    try:
                        dt = datetime.strptime(date_str, "%Y%m%d")
                        day_iso = dt.date().isoformat()
                        if day_iso not in manifest:
                            try:
                                url = extract_opendap_url(item)
                                manifest[day_iso] = url
                            except Exception:
                                pass
                    except ValueError:
                        pass

    missing = [d.isoformat() for d in date_range(START_DATE, END_DATE) if d.isoformat() not in manifest]
    if missing:
        raise RuntimeError(
            f"CMR inventory incomplete: {len(missing)} missing days. "
            f"First missing: {', '.join(missing[:10])}"
        )
    if len(manifest) != EXPECTED_DAYS:
        raise RuntimeError(f"CMR inventory expected {EXPECTED_DAYS}, built {len(manifest)}.")

    atomic_write_json(MANIFEST_FILE, {
        "pipeline_version": "CCMP_V5_OPENDAP",
        "collection_id": COLLECTION_ID,
        "collection": COLLECTION_SHORTNAME,
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "granules": manifest,
    })
    logger.info("CMR manifest READY: %d/%d daily granules.", len(manifest), EXPECTED_DAYS)
    return manifest


def get_opendap_for_date(target_date: date, manifest: dict[str, str]) -> str:
    try:
        return manifest[target_date.isoformat()]
    except KeyError as exc:
        raise RuntimeError(f"No OPeNDAP granule in manifest for {target_date}") from exc


# =============================================================================
# OPeNDAP URL / CONSTRAINTS
# =============================================================================

def normalize_opendap_granule_url(url: str) -> str:
    url = url.strip()
    if url.endswith(".dap.nc4"):
        return url
    if url.endswith(".nc4"):
        return url[:-4] + ".dap.nc4"
    if url.endswith(".nc"):
        return url[:-3] + ".dap.nc4"
    return url + ".dap.nc4"


def build_dap4_url(opendap_base: str, projection: list[str]) -> str:
    base = normalize_opendap_granule_url(opendap_base)
    expression = ";".join(projection)
    encoded = quote(expression, safe="")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}dap4.ce={encoded}"


def dap_slice(name: str, start: int, stop: int, stride: int = 1) -> str:
    return f"/{name}[{start}:{stride}:{stop}]"


# =============================================================================
# OPeNDAP DOWNLOAD
# =============================================================================

def stream_download(url: str, destination: Path, timeout_seconds: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    session = get_session()
    started = time.perf_counter()

    with session.get(
        url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, timeout_seconds),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" in content_type or "json" in content_type:
            preview = response.text[:500]
            raise RuntimeError(
                f"OPeNDAP returned an error/document response instead of NetCDF. "
                f"Content-Type={content_type}; preview={preview!r}"
            )

        total = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)

    if total <= 0:
        raise RuntimeError("OPeNDAP returned zero bytes")

    validate_netcdf_magic(temporary)
    temporary.replace(destination)
    elapsed = time.perf_counter() - started
    logger.info(
        "OPeNDAP response: %s in %.2fs (%.2f MB/s)",
        human_size(total),
        elapsed,
        total / 1024 / 1024 / max(elapsed, 1e-9),
    )


def download_with_retries(url: str, destination: Path, target_date: date, label: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[%s] %s attempt %d/%d", target_date, label, attempt, MAX_RETRIES)
            stream_download(url, destination, OPENDAP_TIMEOUT_SECONDS)
            return True
        except Exception as exc:
            logger.warning("[%s] %s failed: %s", target_date, label, exc)
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    logger.error("[%s] %s FAILED after %d attempts", target_date, label, MAX_RETRIES)
    return False


# =============================================================================
# COORDINATE DISCOVERY
# =============================================================================

def load_coordinates_from_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True, mask_and_scale=True) as ds:
        ds = normalize_source_coordinates(ds)
        latitude = ds["latitude"].values.astype(np.float64)
        longitude = ds["longitude"].values.astype(np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise RuntimeError("CCMP latitude/longitude are not 1-D")
    if len(latitude) < 2 or len(longitude) < 2:
        raise RuntimeError("CCMP coordinate vectors are unexpectedly small")
    return latitude, longitude


def normalize_source_coordinates(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" not in ds.coords and "lat" in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "longitude" not in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lon": "longitude"})
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError("Source is missing latitude/longitude coordinates")
    if ds["latitude"].ndim != 1 or ds["longitude"].ndim != 1:
        raise RuntimeError("Expected 1-D latitude/longitude coordinates")
    return ds.sortby("latitude").sortby("longitude")


def coordinate_indices(values: np.ndarray, minimum: float, maximum: float) -> tuple[int, int]:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.diff(values) > 0):
        raise RuntimeError("Coordinate vector is not strictly increasing")
    mask = (values >= minimum - COORD_TOLERANCE) & (values <= maximum + COORD_TOLERANCE)
    indices = np.flatnonzero(mask)
    if len(indices) < 2:
        raise RuntimeError(f"Could not find enough coordinate cells for {minimum}..{maximum}")
    return int(indices[0]), int(indices[-1])


def get_coordinate_indices(target_date: date, opendap_url: str) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    coordinate_path = build_coordinate_path(target_date)
    if coordinate_path.exists():
        try:
            validate_netcdf_magic(coordinate_path)
            latitude, longitude = load_coordinates_from_file(coordinate_path)
        except Exception:
            try:
                coordinate_path.unlink()
            except OSError:
                pass
    if not coordinate_path.exists():
        projection = ["/latitude", "/longitude"]
        url = build_dap4_url(opendap_url, projection)
        if not download_with_retries(url, coordinate_path, target_date, "coordinate metadata"):
            raise RuntimeError("Could not download CCMP coordinate metadata")
        latitude, longitude = load_coordinates_from_file(coordinate_path)

    lat_start, lat_stop = coordinate_indices(latitude, TARGET_LAT_MIN - NATIVE_BUFFER, TARGET_LAT_MAX + NATIVE_BUFFER)
    lon_start, lon_stop = coordinate_indices(longitude, TARGET_LON_MIN - NATIVE_BUFFER, TARGET_LON_MAX + NATIVE_BUFFER)
    return latitude, longitude, lat_start, lat_stop, lon_start, lon_stop


def initialize_global_grid(reference_date: date, opendap_url: str) -> tuple[int, int, int, int]:
    if GRID_FILE.exists():
        try:
            with GRID_FILE.open("r", encoding="utf-8") as h:
                payload = json.load(h)
            indices = tuple(int(v) for v in payload["indices"])
            if len(indices) == 4:
                logger.info("Using cached CCMP grid indices: %s", indices)
                return indices
        except Exception as exc:
            logger.warning("Cached grid invalid; rediscovering: %s", exc)

    lat, lon, ls, le, os_, oe = get_coordinate_indices(reference_date, opendap_url)
    if not np.allclose(np.diff(lat), 0.25, rtol=0, atol=1e-5):
        raise RuntimeError("CCMP latitude grid is not regular 0.25 degree.")
    if not np.allclose(np.diff(lon), 0.25, rtol=0, atol=1e-5):
        raise RuntimeError("CCMP longitude grid is not regular 0.25 degree.")
    indices = (ls, le, os_, oe)
    atomic_write_json(GRID_FILE, {
        "pipeline_version": "CCMP_V5_OPENDAP",
        "reference_date": reference_date.isoformat(),
        "latitude_count": len(lat),
        "longitude_count": len(lon),
        "latitude_start": float(lat[0]),
        "latitude_end": float(lat[-1]),
        "longitude_start": float(lon[0]),
        "longitude_end": float(lon[-1]),
        "indices": list(indices),
    })
    logger.info("CCMP grid READY: lat=%d:%d lon=%d:%d", ls, le, os_, oe)
    return indices


# =============================================================================
# SCIENCE SUBSET
# =============================================================================

def build_science_constraint(lat_start: int, lat_stop: int, lon_start: int, lon_stop: int) -> str:
    uwnd = f"/uwnd[0:1:3][{lat_start}:1:{lat_stop}][{lon_start}:1:{lon_stop}]"
    vwnd = f"/vwnd[0:1:3][{lat_start}:1:{lat_stop}][{lon_start}:1:{lon_stop}]"
    time_var = "/time[0:1:3]"
    latitude = f"/latitude[{lat_start}:1:{lat_stop}]"
    longitude = f"/longitude[{lon_start}:1:{lon_stop}]"
    return ";".join([uwnd, vwnd, time_var, latitude, longitude])


def validate_subset_source(path: Path, target_date: date) -> None:
    with xr.open_dataset(path, engine="netcdf4", decode_times=True, mask_and_scale=True) as ds:
        ds = normalize_source_coordinates(ds)
        required = {"time", "latitude", "longitude"}
        missing = required - set(ds.coords)
        if missing:
            raise RuntimeError(f"Subset missing coordinates: {sorted(missing)}")
        for variable in ("uwnd", "vwnd"):
            if variable not in ds.data_vars:
                raise RuntimeError(f"Subset missing {variable}")
        if ds.sizes.get("time") != 4:
            raise RuntimeError(f"Expected exactly four CCMP analyses; found {ds.sizes.get('time')}")
        time_values = ds["time"].values
        actual_hours = {
            int(np.datetime64(value, "h").astype("datetime64[h]").astype(int) % 24)
            for value in time_values
        }
        if actual_hours != {0, 6, 12, 18}:
            raise RuntimeError(f"Unexpected CCMP hours: {sorted(actual_hours)}")
        first_day = np.datetime64(time_values[0], "D")
        expected_day = np.datetime64(target_date.isoformat(), "D")
        if first_day != expected_day:
            raise RuntimeError(f"Subset date mismatch: expected {expected_day}, found {first_day}")
        expected_dims = ("time", "latitude", "longitude")
        for variable in ("uwnd", "vwnd"):
            if ds[variable].dims != expected_dims:
                raise RuntimeError(f"{variable} dimensions are {ds[variable].dims}; expected {expected_dims}")
            values = ds[variable].values
            if not np.isfinite(values).any():
                raise RuntimeError(f"{variable} contains no finite values")
            minimum = float(np.nanmin(values))
            maximum = float(np.nanmax(values))
            if minimum < -100 or maximum > 100:
                raise RuntimeError(f"{variable} has suspicious values: {minimum}..{maximum} m/s")
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if lat.min() > TARGET_LAT_MIN:
            raise RuntimeError("Subset lacks southern interpolation neighbor")
        if lat.max() < TARGET_LAT_MAX:
            raise RuntimeError("Subset lacks northern interpolation neighbor")
        if lon.min() > TARGET_LON_MIN:
            raise RuntimeError("Subset lacks western interpolation neighbor")
        if lon.max() < TARGET_LON_MAX:
            raise RuntimeError("Subset lacks eastern interpolation neighbor")


def download_science_subset(
    target_date: date,
    opendap_url: str,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    destination: Path,
) -> bool:
    projection = build_science_constraint(lat_start, lat_stop, lon_start, lon_stop)
    url = build_dap4_url(opendap_url, projection.split(";"))
    logger.info(
        "[%s] OPeNDAP science indices: lat=%d:%d lon=%d:%d",
        target_date, lat_start, lat_stop, lon_start, lon_stop,
    )
    logger.info("[%s] OPeNDAP constraint: %s", target_date, projection)
    if not download_with_retries(url, destination, target_date, "science subset"):
        return False
    try:
        validate_subset_source(destination, target_date)
        return True
    except Exception as exc:
        logger.error("[%s] Science subset QC failed: %s", target_date, exc)
        try:
            destination.unlink()
        except OSError:
            pass
        return False


# =============================================================================
# DAILY OCEANEMBED PROCESSING
# =============================================================================

def daily_oceanembed_uv(source_path: Path, target_date: date) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(source_path, engine="netcdf4", decode_times=True, mask_and_scale=True) as ds:
        ds = normalize_source_coordinates(ds)
        validate_subset_source(source_path, target_date)
        target_lats, target_lons = expected_target_grid()
        regional = ds[["uwnd", "vwnd"]]
        regional = regional.interp(
            latitude=xr.DataArray(target_lats, dims="latitude"),
            longitude=xr.DataArray(target_lons, dims="longitude"),
            method="linear",
        )
        daily_u = regional["uwnd"].mean(dim="time", skipna=True).astype("float32").values
        daily_v = regional["vwnd"].mean(dim="time", skipna=True).astype("float32").values
        if not np.isfinite(daily_u).any():
            raise RuntimeError("Daily U field has no finite values")
        if not np.isfinite(daily_v).any():
            raise RuntimeError("Daily V field has no finite values")
        return daily_u, daily_v


def write_processed(target_date: date, daily_u: np.ndarray, daily_v: np.ndarray) -> Path:
    target_lats, target_lons = expected_target_grid()
    expected_shape = (len(target_lats), len(target_lons))
    if daily_u.shape != expected_shape:
        raise RuntimeError(f"Unexpected U shape {daily_u.shape}; expected {expected_shape}")
    if daily_v.shape != expected_shape:
        raise RuntimeError(f"Unexpected V shape {daily_v.shape}; expected {expected_shape}")

    processed_path = build_processed_path(target_date)
    temporary = processed_path.with_suffix(processed_path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    output = xr.Dataset(
        data_vars={
            "uwnd": (("time", "latitude", "longitude"), daily_u[np.newaxis, :, :]),
            "vwnd": (("time", "latitude", "longitude"), daily_v[np.newaxis, :, :]),
        },
        coords={
            "time": [np.datetime64(target_date.isoformat())],
            "latitude": target_lats,
            "longitude": target_lons,
        },
        attrs={
            "title": "OceanEmbed processed CCMP V3.1 daily surface winds",
            "source": "RSS CCMP 6-Hourly 10 Meter Surface Winds L4 V3.1",
            "source_doi": "10.5067/CCMP-6HW10M-L4V31",
            "source_url": RSS_BASE,
            "podaac_collection": COLLECTION_SHORTNAME,
            "podaac_concept_id": COLLECTION_ID,
            "processing": (
                "Direct PO.DAAC OPeNDAP-in-the-Cloud native spatial/variable subset, "
                "linear spatial interpolation to OceanEmbed 0.25 degree grid, "
                "daily vector mean of 00/06/12/18 UTC"
            ),
            "region": "North Indian Ocean: 45E-105E, 5N-30N",
            "spatial_resolution": "0.25 degree",
            "temporal_resolution": "daily",
            "variables": "uwnd, vwnd",
            "wind_speed_note": "Wind speed can be derived as sqrt(uwnd^2 + vwnd^2).",
            "pipeline_version": "OceanEmbed CCMP V5 OPeNDAP",
        },
    )

    output["uwnd"].attrs = {
        "long_name": "Daily mean eastward wind component at 10 meters",
        "standard_name": "eastward_wind",
        "units": "m s-1",
    }
    output["vwnd"].attrs = {
        "long_name": "Daily mean northward wind component at 10 meters",
        "standard_name": "northward_wind",
        "units": "m s-1",
    }
    output["latitude"].attrs = {"standard_name": "latitude", "long_name": "latitude", "units": "degrees_north"}
    output["longitude"].attrs = {"standard_name": "longitude", "long_name": "longitude", "units": "degrees_east"}
    output["time"].attrs = {"standard_name": "time"}

    encoding = {
        "uwnd": {"dtype": "float32", "zlib": True, "complevel": NETCDF_COMPRESSION_LEVEL, "shuffle": True},
        "vwnd": {"dtype": "float32", "zlib": True, "complevel": NETCDF_COMPRESSION_LEVEL, "shuffle": True},
        "latitude": {"dtype": "float32"},
        "longitude": {"dtype": "float32"},
    }

    try:
        output.to_netcdf(temporary, engine="netcdf4", format="NETCDF4", encoding=encoding)
        output.close()
        temporary.replace(processed_path)
    except Exception:
        try:
            output.close()
        except Exception:
            pass
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise

    with xr.open_dataset(processed_path, engine="netcdf4") as final_ds:
        expected_final_shape = (1, len(target_lats), len(target_lons))
        if final_ds["uwnd"].shape != expected_final_shape:
            raise RuntimeError(f"Final U shape {final_ds['uwnd'].shape}; expected {expected_final_shape}")
        if final_ds["vwnd"].shape != expected_final_shape:
            raise RuntimeError(f"Final V shape {final_ds['vwnd'].shape}; expected {expected_final_shape}")
        if set(final_ds.data_vars) != {"uwnd", "vwnd"}:
            raise RuntimeError("Unexpected final data variables")
        for variable in ("uwnd", "vwnd"):
            values = final_ds[variable].values
            if not np.isfinite(values).any():
                raise RuntimeError(f"Final {variable} has no finite values")

    return processed_path


# =============================================================================
# OFFICIAL RSS REFERENCE
# =============================================================================

def download_direct_rss(target_date: date, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    session = get_session()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if temporary.exists():
                temporary.unlink()
            started = time.perf_counter()
            with session.get(
                build_rss_url(target_date),
                stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                expected_size = int(expected) if expected else None
                downloaded = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            handle.write(chunk)
                            downloaded += len(chunk)
            if downloaded <= 0:
                raise RuntimeError("RSS reference returned zero bytes")
            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(f"RSS size mismatch: expected {expected_size}, received {downloaded}")
            validate_netcdf_magic(temporary)
            temporary.replace(destination)
            elapsed = time.perf_counter() - started
            logger.info("[%s] RSS reference: %s in %.1fs", target_date, human_size(downloaded), elapsed)
            return True
        except Exception as exc:
            logger.warning("[%s] RSS reference attempt %d/%d failed: %s", target_date, attempt, MAX_RETRIES, exc)
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return False


def compare_against_rss(target_date: date, subset_path: Path, rss_path: Path) -> None:
    logger.info("[%s] Running OPeNDAP subset vs official RSS numerical comparison...", target_date)
    subset_u, subset_v = daily_oceanembed_uv(subset_path, target_date)
    rss_u, rss_v = daily_oceanembed_uv(rss_path, target_date)

    for variable, actual, reference in (("uwnd", subset_u, rss_u), ("vwnd", subset_v, rss_v)):
        if actual.shape != reference.shape:
            raise RuntimeError(f"{variable} shape mismatch: subset={actual.shape}, RSS={reference.shape}")
        difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        max_abs = float(np.nanmax(difference))
        mean_abs = float(np.nanmean(difference))
        if not np.allclose(actual, reference, rtol=VALIDATION_RTOL, atol=VALIDATION_ATOL, equal_nan=True):
            raise RuntimeError(
                f"{variable} OPeNDAP/RSS mismatch: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}"
            )
        logger.info("[%s] %s OPeNDAP/RSS PASS | max_abs=%.8g | mean_abs=%.8g", target_date, variable, max_abs, mean_abs)


# =============================================================================
# ONE-DAY ACQUISITION TRANSACTION
# =============================================================================

def acquire_one_day(
    target_date: date,
    opendap_url: str,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
) -> tuple[date, bool, str]:
    processed_path = build_processed_path(target_date)
    if is_complete(target_date):
        return target_date, True, "already complete"

    subset_path = build_subset_path(target_date)
    started = time.perf_counter()
    logger.info("[%s] START", target_date)

    try:
        if not subset_path.exists():
            ok = download_science_subset(
                target_date, opendap_url, lat_start, lat_stop, lon_start, lon_stop, subset_path
            )
            if not ok:
                return target_date, False, "OPeNDAP science subset failed"
        else:
            logger.info("[%s] Existing OPeNDAP subset found: %s", target_date, human_size(subset_path.stat().st_size))
            validate_subset_source(subset_path, target_date)

        daily_u, daily_v = daily_oceanembed_uv(subset_path, target_date)
        processed_path = write_processed(target_date, daily_u, daily_v)
        mark_complete(target_date, processed_path)

    except Exception as exc:
        logger.error("[%s] FAILED: %s", target_date, exc, exc_info=True)
        return target_date, False, str(exc)

    if not KEEP_RAW_AFTER_PROCESSING:
        for path in (subset_path, build_coordinate_path(target_date), build_cmr_metadata_path(target_date)):
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                logger.warning("[%s] Could not delete temporary %s: %s", target_date, path, exc)

    elapsed = time.perf_counter() - started
    logger.info("[%s] COMPLETE in %.1fs", target_date, elapsed)
    return target_date, True, "complete"


# =============================================================================
# TEST MODE
# =============================================================================

def run_test() -> int:
    target_date = TEST_DATE
    subset_path = build_subset_path(target_date)
    rss_path = build_temp_day_dir(target_date) / f"{target_date:%Y%m%d}_RSS_GLOBAL.nc"

    logger.info("")
    logger.info("=" * 90)
    logger.info("CCMP V5 OPeNDAP VALIDATION TEST")
    logger.info("=" * 90)
    logger.info("Test date: %s", target_date)
    logger.info("Collection: %s", COLLECTION_SHORTNAME)
    logger.info("Target: %.2f..%.2f E / %.2f..%.2f N", TARGET_LON_MIN, TARGET_LON_MAX, TARGET_LAT_MIN, TARGET_LAT_MAX)
    logger.info("Native support: %.2f..%.2f E / %.2f..%.2f N",
                TARGET_LON_MIN - NATIVE_BUFFER, TARGET_LON_MAX + NATIVE_BUFFER,
                TARGET_LAT_MIN - NATIVE_BUFFER, TARGET_LAT_MAX + NATIVE_BUFFER)
    logger.info("Access: direct PO.DAAC OPeNDAP-in-the-Cloud")

    try:
        manifest = fetch_cmr_inventory()
        opendap_url = get_opendap_for_date(target_date, manifest)
        logger.info("OPeNDAP URL: %s", opendap_url)

        lat_start, lat_stop, lon_start, lon_stop = initialize_global_grid(target_date, opendap_url)
        latitude, longitude = load_coordinates_from_file(build_coordinate_path(target_date))

        logger.info("Global source dimensions: lat=%d lon=%d", len(latitude), len(longitude))
        logger.info("Selected indices: lat=%d:%d lon=%d:%d", lat_start, lat_stop, lon_start, lon_stop)

        if not download_science_subset(
            target_date, opendap_url, lat_start, lat_stop, lon_start, lon_stop, subset_path
        ):
            raise RuntimeError("OPeNDAP science subset download failed")

        logger.info("OPeNDAP subset size: %s", human_size(subset_path.stat().st_size))

        if not download_direct_rss(target_date, rss_path):
            raise RuntimeError("Official RSS reference download failed")

        compare_against_rss(target_date, subset_path, rss_path)

        daily_u, daily_v = daily_oceanembed_uv(subset_path, target_date)
        processed_path = write_processed(target_date, daily_u, daily_v)
        mark_complete(target_date, processed_path)

        logger.info("")
        logger.info("=" * 90)
        logger.info("CCMP V5 OPeNDAP TEST PASSED")
        logger.info("=" * 90)
        logger.info("Processed output: %s", processed_path)
        logger.info("Output size: %s", human_size(processed_path.stat().st_size))
        logger.info("Set TEST_MODE = False for full acquisition.")
        return 0

    except Exception as exc:
        logger.error("CCMP V5 TEST FAILED: %s", exc, exc_info=True)
        return 1
    finally:
        # Keep processed output; remove disposable test inputs.
        for path in (subset_path, rss_path, build_coordinate_path(target_date), build_cmr_metadata_path(target_date)):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(all_dates: list[date], results: list[tuple[date, bool, str]]) -> None:
    successful = [item for item in results if item[1]]
    failed = [item for item in results if not item[1]]

    processed_files = 0
    total_processed_bytes = 0
    for target_date in all_dates:
        path = build_processed_path(target_date)
        if path.exists():
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 0:
                processed_files += 1
                total_processed_bytes += size

    logger.info("")
    logger.info("=" * 90)
    logger.info("CCMP V5 OPeNDAP ACQUISITION SUMMARY")
    logger.info("=" * 90)
    logger.info("Requested dates : %d", len(all_dates))
    logger.info("Successful      : %d", len(successful))
    logger.info("Failed          : %d", len(failed))
    logger.info("Processed files : %d", processed_files)
    logger.info("Processed size  : %s", human_size(total_processed_bytes))
    logger.info("State file      : %s", STATE_FILE)
    logger.info("Log file        : %s", LOG_FILE)

    if failed:
        logger.error("")
        logger.error("FAILED DATES:")
        for target_date, _, reason in sorted(failed):
            logger.error("  %s -> %s", target_date, reason)
    logger.info("=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    logger.info("")
    logger.info("=" * 90)
    logger.info("OCEANEMBED CCMP V5")
    logger.info("DIRECT PO.DAAC OPeNDAP-IN-THE-CLOUD")
    logger.info("=" * 90)
    logger.info("Source: RSS CCMP V3.1 / PO.DAAC")
    logger.info("Collection: %s", COLLECTION_SHORTNAME)
    logger.info("Collection concept: %s", COLLECTION_ID)
    logger.info("DOI: 10.5067/CCMP-6HW10M-L4V31")
    logger.info("Target: %.2f..%.2f E, %.2f..%.2f N", TARGET_LON_MIN, TARGET_LON_MAX, TARGET_LAT_MIN, TARGET_LAT_MAX)
    logger.info("Native support: %.2f..%.2f E, %.2f..%.2f N",
                TARGET_LON_MIN - NATIVE_BUFFER, TARGET_LON_MAX + NATIVE_BUFFER,
                TARGET_LAT_MIN - NATIVE_BUFFER, TARGET_LAT_MAX + NATIVE_BUFFER)
    logger.info("Workers: %d", MAX_WORKERS)
    logger.info("TEST_MODE: %s", TEST_MODE)
    logger.info("Python: %s", sys.version.replace("\n", " "))

    ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        test_result = run_test()
        if test_result != 0 or not AUTO_START_FULL_AFTER_TEST:
            return test_result
        logger.info("")
        logger.info("=" * 90)
        logger.info("VALIDATION PASSED — AUTOMATICALLY STARTING FULL 2012-2024 ACQUISITION")
        logger.info("=" * 90)

    dates = list(date_range(START_DATE, END_DATE))
    pending_dates = [d for d in dates if not is_complete(d)]

    logger.info("Total calendar days: %d", len(dates))
    logger.info("Already complete: %d", len(dates) - len(pending_dates))
    logger.info("Pending: %d", len(pending_dates))

    if not pending_dates:
        print_summary(dates, [(d, True, "already complete") for d in dates])
        return 0

    # One source manifest + one grid discovery, then parallel OPeNDAP downloads.
    manifest = fetch_cmr_inventory()
    first_key = min(manifest)
    reference_day = date.fromisoformat(first_key)
    grid_indices = initialize_global_grid(reference_day, manifest[first_key])

    results = []
    remaining = pending_dates

    for production_pass in range(1, PRODUCTION_PASSES + 1):
        if not remaining:
            break

        logger.info("Starting production pass %d/%d with %d workers...", production_pass, PRODUCTION_PASSES, MAX_WORKERS)
        logger.info("Zero per-day CMR queries; cached manifest + cached grid.")

        pass_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    acquire_one_day,
                    d,
                    manifest[d.isoformat()],
                    *grid_indices,
                ): d
                for d in remaining
            }
            for future in as_completed(future_map):
                d = future_map[future]
                try:
                    result = future.result()
                    pass_results.append(result)
                    _, ok, msg = result
                    logger.info("FINISHED %s -> %s", d, msg) if ok else logger.error("FAILED %s -> %s", d, msg)
                except Exception as exc:
                    logger.error("UNEXPECTED ERROR for %s: %s", d, exc, exc_info=True)
                    pass_results.append((d, False, f"unexpected error: {exc}"))

        results.extend(pass_results)
        remaining = [d for d in remaining if not is_complete(d)]
        logger.info("Production pass %d complete. Remaining unfinished days: %d", production_pass, len(remaining))

    final_failed = [d for d in dates if not is_complete(d)]
    summary_results = [(d, is_complete(d), "complete" if is_complete(d) else "unfinished after retries") for d in dates]
    print_summary(dates, summary_results)
    return int(bool(final_failed))


if __name__ == "__main__":
    raise SystemExit(main())