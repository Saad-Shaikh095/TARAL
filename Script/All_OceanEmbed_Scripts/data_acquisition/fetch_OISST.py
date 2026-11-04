#!/usr/bin/env python3
"""
NOAA OISST Daily Downloader, Subsetter, Organizer, and Validator
Fixed version with dynamic coordinate detection and no manual time encoding.
"""

import os, sys, time, threading, gc, warnings
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
START_YEAR, END_YEAR = 2012, 2024
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
EXPECTED_RES = 0.25
RES_TOLERANCE = 0.01

PRODUCTION_ROOT = Path(r"E:\Ocean_data\OISST\data\raw")
REPORT_DIR = Path(r"E:\Ocean_data\OISST\reports")
REPORT_PATH = REPORT_DIR / "oisst_download_report.txt"
TEMP_ROOT = Path(r"E:\Ocean_data\OISST\temp")

RETAIN_VARIABLES = ["sst", "anom", "err", "ice"]
SST_VAR = "sst"

CHANNEL_COUNT = 3
WORKERS_PER_CHANNEL = 10
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5
TIMEOUT = 300

MIN_FILE_SIZE_KB = 50
MAX_NAN_FRACTION = 0.95

GLOBAL_LAT_COUNT = 720
GLOBAL_LON_COUNT = 1440
BBOX_LAT_COUNT = 100
BBOX_LON_COUNT = 240

BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"

counter_lock = threading.Lock()
print_lock = threading.Lock()
downloaded_count = skipped_count = failed_count = subsetted_count = already_subset_count = active_requests = 0

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def generate_dates(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def build_file_url(day: date) -> str:
    filename = f"oisst-avhrr-v02r01.{day.strftime('%Y%m%d')}.nc"
    month_dir = day.strftime("%Y%m")
    return f"{BASE_URL}{month_dir}/{filename}"

def build_output_file(day: date) -> Path:
    outdir = PRODUCTION_ROOT / f"{day.year:04d}" / f"{day.month:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"oisst_{day.strftime('%Y%m%d')}.nc"

def is_valid_file(fp: Path) -> bool:
    if not fp.exists() or fp.stat().st_size < MIN_FILE_SIZE_KB * 1024:
        return False
    try:
        with open(fp, "rb") as f:
            header = f.read(8)
        return header.startswith(b"\x89HDF") or header.startswith(b"CDF")
    except OSError:
        return False

def find_coord_names(ds: xr.Dataset) -> tuple:
    """Return (lat_name, lon_name, time_name) dynamically."""
    lat_name = lon_name = time_name = None
    for c in ds.coords:
        cl = c.lower()
        if 'lat' in cl and lat_name is None:
            lat_name = c
        elif 'lon' in cl and lon_name is None:
            lon_name = c
        elif 'time' in cl and time_name is None:
            time_name = c
    # Fallbacks
    if lat_name is None and 'latitude' in ds.coords: lat_name = 'latitude'
    if lon_name is None and 'longitude' in ds.coords: lon_name = 'longitude'
    if time_name is None and 'time' in ds.coords: time_name = 'time'
    return lat_name, lon_name, time_name

def is_already_subsetted(fp: Path) -> bool:
    ds = None
    try:
        if not is_valid_file(fp): return False
        ds = xr.open_dataset(fp, decode_times=True)
        lat_name, lon_name, _ = find_coord_names(ds)
        if not lat_name or not lon_name: return False
        if len(ds[lat_name]) != BBOX_LAT_COUNT or len(ds[lon_name]) != BBOX_LON_COUNT:
            return False
        lat = ds[lat_name].values
        lon = ds[lon_name].values
        return lat.min() >= LAT_MIN - 0.125 and lat.max() <= LAT_MAX + 0.125 and lon.min() >= LON_MIN - 0.125 and lon.max() <= LON_MAX + 0.125
    except Exception:
        return False
    finally:
        if ds is not None: ds.close()

def validate_bbox_file(fp: Path, expected_date: date) -> Tuple[bool, str, dict]:
    info = {}
    errors = []
    warnings = []
    if not fp.exists():
        return False, "File missing", info
    if not is_valid_file(fp):
        return False, "Invalid NetCDF", info
    try:
        ds = xr.open_dataset(fp, decode_times=True)
    except Exception as e:
        return False, f"Cannot open: {e}", info

    # Variables
    missing = [v for v in RETAIN_VARIABLES if v not in ds.variables]
    if missing: errors.append(f"Missing vars: {missing}")
    if SST_VAR not in ds.variables: errors.append(f"Missing SST var: {SST_VAR}")

    # Coords
    lat_name, lon_name, time_name = find_coord_names(ds)
    if not lat_name: errors.append("Lat coord not found")
    if not lon_name: errors.append("Lon coord not found")
    if not time_name: errors.append("Time coord not found")

    if lat_name and lon_name:
        lat = ds[lat_name].values
        lon = ds[lon_name].values
        if len(lat) != BBOX_LAT_COUNT: errors.append(f"Lat count {len(lat)} != {BBOX_LAT_COUNT}")
        if len(lon) != BBOX_LON_COUNT: errors.append(f"Lon count {len(lon)} != {BBOX_LON_COUNT}")
        if len(lat) > 1:
            spacing = np.mean(np.diff(lat))
            info['lat_spacing'] = spacing
            if abs(spacing - EXPECTED_RES) > RES_TOLERANCE: errors.append(f"Lat spacing bad")
            if not np.all(np.diff(lat) > 0): errors.append("Lat not monotonic")
        if len(lon) > 1:
            spacing = np.mean(np.diff(lon))
            info['lon_spacing'] = spacing
            if abs(spacing - EXPECTED_RES) > RES_TOLERANCE: errors.append(f"Lon spacing bad")
            if not np.all(np.diff(lon) > 0): errors.append("Lon not monotonic")
        info['lat_min'], info['lat_max'] = float(lat.min()), float(lat.max())
        info['lon_min'], info['lon_max'] = float(lon.min()), float(lon.max())
        info['lat_count'] = len(lat)
        info['lon_count'] = len(lon)
        # Bounds
        if lat.min() < LAT_MIN - 0.125 or lat.max() > LAT_MAX + 0.125: errors.append("Lat out of bbox")
        if lon.min() < LON_MIN - 0.125 or lon.max() > LON_MAX + 0.125: errors.append("Lon out of bbox")

    if time_name:
        times = ds[time_name].values
        if len(times) != 1: errors.append(f"Time count {len(times)} != 1")
        else:
            try:
                t0 = times[0]
                if isinstance(t0, np.datetime64): d = pd.Timestamp(t0).date()
                else: d = pd.Timestamp(t0.isoformat()).date()
                if d != expected_date: errors.append(f"Time mismatch: file {expected_date}, internal {d}")
                else: info['time_date'] = str(d)
            except Exception as e: errors.append(f"Time decode error: {e}")

    # Data stats
    for var in RETAIN_VARIABLES:
        if var in ds.variables:
            v = ds[var]
            attrs = v.attrs
            info[f'{var}_units'] = attrs.get('units', None)
            info[f'{var}_fill'] = attrs.get('_FillValue', None)
            try:
                vals = v.values
                info[f'{var}_nan_count'] = int(np.isnan(vals).sum())
                info[f'{var}_inf_count'] = int(np.isinf(vals).sum())
                finite = vals[np.isfinite(vals)]
                info[f'{var}_min'] = float(np.nanmin(vals)) if finite.size else None
                info[f'{var}_max'] = float(np.nanmax(vals)) if finite.size else None
                total = vals.size
                info[f'{var}_nan_frac'] = info[f'{var}_nan_count'] / total if total > 0 else 0
                if info[f'{var}_nan_frac'] > MAX_NAN_FRACTION: warnings.append(f"{var} high NaN")
            except Exception as e:
                errors.append(f"Stats error {var}: {e}")

    ds.close()
    valid = len(errors) == 0
    return valid, ("PASS" if valid else "; ".join(errors)), info

# ============================================================
# SUBSET GLOBAL FILE
# ============================================================
def subset_global_file(input_file: Path, expected_date: date) -> Dict:
    global subsetted_count, already_subset_count, failed_count

    start = time.perf_counter()
    if is_already_subsetted(input_file):
        valid, msg, _ = validate_bbox_file(input_file, expected_date)
        if valid:
            with counter_lock: already_subset_count += 1
            return {"status": "already_subsetted", "file": input_file, "date": expected_date}

    temp_file = input_file.with_name(input_file.stem + ".subset.tmp.nc")
    temp_file.unlink(missing_ok=True)
    ds = None
    subset = None
    try:
        ds = xr.open_dataset(input_file, decode_times=True)
        lat_name, lon_name, time_name = find_coord_names(ds)
        if not lat_name or not lon_name:
            raise RuntimeError("Could not identify lat/lon coords")

        # Verify global size
        if len(ds[lat_name]) != GLOBAL_LAT_COUNT or len(ds[lon_name]) != GLOBAL_LON_COUNT:
            raise RuntimeError(f"Unexpected global grid: {len(ds[lat_name])}x{len(ds[lon_name])}")

        # Verify date
        if time_name and len(ds[time_name]) > 0:
            t0 = ds[time_name].values[0]
            d = pd.Timestamp(t0).date() if isinstance(t0, np.datetime64) else pd.Timestamp(t0.isoformat()).date()
            if d != expected_date:
                raise RuntimeError(f"Source date {d} != expected {expected_date}")

        # Subset using sel with slice (robust)
        subset = ds[RETAIN_VARIABLES].sel(
            {lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)}
        )
        # Ensure exact sizes
        if subset[lat_name].size != BBOX_LAT_COUNT or subset[lon_name].size != BBOX_LON_COUNT:
            raise RuntimeError(f"Subset size wrong: {subset[lat_name].size}x{subset[lon_name].size}")

        subset.load()

        # Encoding for data variables only (no time encoding)
        encoding = {}
        for var in RETAIN_VARIABLES:
            enc = dict(ds[var].encoding)
            safe = {k: enc[k] for k in ["_FillValue", "dtype", "scale_factor", "add_offset", "zlib", "complevel", "shuffle", "fletcher32"] if k in enc}
            safe.pop("chunksizes", None)
            encoding[var] = safe

        ds.close(); ds = None

        subset.to_netcdf(temp_file, mode="w", format="NETCDF4", encoding=encoding)
        del subset; subset = None
        gc.collect()

        valid, msg, _ = validate_bbox_file(temp_file, expected_date)
        if not valid:
            temp_file.unlink(missing_ok=True)
            raise RuntimeError(f"Subset validation failed: {msg}")

        # Replace original
        orig_size = input_file.stat().st_size
        sub_size = temp_file.stat().st_size
        temp_file.replace(input_file)

        with counter_lock: subsetted_count += 1
        return {"status": "subsetted", "file": input_file, "date": expected_date,
                "original_mb": orig_size/(1024*1024), "subset_mb": sub_size/(1024*1024),
                "reduction_pct": (1 - sub_size/orig_size)*100 if orig_size>0 else 0,
                "elapsed": time.perf_counter()-start}
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        with counter_lock: failed_count += 1
        return {"status": "failed", "file": input_file, "date": expected_date, "error": str(e)}
    finally:
        if ds is not None: ds.close()
        if subset is not None: del subset
        gc.collect()

# ============================================================
# DOWNLOAD
# ============================================================
def download_file(channel_id: int, day: date) -> Dict:
    global downloaded_count, skipped_count, failed_count, active_requests
    output_file = build_output_file(day)
    if is_valid_file(output_file):
        with counter_lock: skipped_count += 1
        return {"channel": channel_id, "date": day, "status": "existing", "file": output_file}

    url = build_file_url(day)
    start = time.perf_counter()
    with counter_lock:
        active_requests += 1
        current_active = active_requests
    temp_file = output_file.with_suffix(".download")
    temp_file.unlink(missing_ok=True)

    for attempt in range(1, MAX_RETRIES+1):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            temp_file.write_bytes(response.content)
            if not is_valid_file(temp_file):
                raise RuntimeError("Invalid NetCDF after download")
            temp_file.replace(output_file)
            with counter_lock:
                downloaded_count += 1
                active_requests -= 1
            return {"channel": channel_id, "date": day, "status": "downloaded", "file": output_file,
                    "size_mb": output_file.stat().st_size/(1024*1024),
                    "elapsed": time.perf_counter()-start, "active": current_active}
        except Exception as e:
            temp_file.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_BASE * (2**(attempt-1)))
            else:
                with counter_lock:
                    failed_count += 1
                    active_requests -= 1
                return {"channel": channel_id, "date": day, "status": "failed", "file": output_file, "error": str(e)}

# ============================================================
# ACQUIRE ONE
# ============================================================
def acquire_one_file(channel_id: int, day: date) -> Dict:
    dl = download_file(channel_id, day)
    if dl["status"] == "failed":
        return dl
    subset = subset_global_file(dl["file"], day)
    return {
        "channel": channel_id, "date": day,
        "download_status": dl["status"],
        "status": subset["status"],
        "file": dl["file"],
        "size_mb": subset.get("subset_mb", dl.get("size_mb", 0)),
        "original_mb": subset.get("original_mb", 0),
        "subset_mb": subset.get("subset_mb", 0),
        "reduction_pct": subset.get("reduction_pct", 0),
        "elapsed": subset.get("elapsed", dl.get("elapsed", 0)),
        "error": subset.get("error")
    }

# ============================================================
# CHANNEL & YEAR
# ============================================================
def build_channel_ranges(year: int) -> Dict[int, Tuple[date, date]]:
    return {1: (date(year,1,1), date(year,4,30)),
            2: (date(year,5,1), date(year,8,31)),
            3: (date(year,9,1), date(year,12,31))}

def run_channel(year: int, channel_id: int, start_date: date, end_date: date) -> Dict:
    dates = list(generate_dates(start_date, end_date))
    print(f"\nYear {year} Channel {channel_id}: {len(dates)} files")
    results = []
    completed = 0
    channel_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS_PER_CHANNEL) as executor:
        futures = {executor.submit(acquire_one_file, channel_id, day): day for day in dates}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            elapsed = time.perf_counter() - channel_start
            with counter_lock: active = active_requests
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"\r  Y{year} C{channel_id} | {completed:3d}/{len(dates)} | Active: {active:2d} | Rate: {rate:.2f} files/s", end="", flush=True)

    channel_elapsed = time.perf_counter() - channel_start
    print(f"\n  Channel {channel_id} finished in {channel_elapsed:.2f}s")

    successful = sum(1 for r in results if r["status"] in ("subsetted", "already_subsetted"))
    subsetted = sum(1 for r in results if r["status"] == "subsetted")
    already = sum(1 for r in results if r["status"] == "already_subsetted")
    failed = sum(1 for r in results if r["status"] == "failed")
    total_mb = sum(r.get("subset_mb", 0) for r in results)

    # Print first 3 errors for debugging
    errors_shown = 0
    for r in results:
        if r["status"] == "failed" and errors_shown < 3:
            print(f"    ERROR example: {r['error']}")
            errors_shown += 1

    return {"year": year, "channel": channel_id, "expected": len(dates),
            "successful": successful, "subsetted": subsetted, "already_subsetted": already,
            "failed": failed, "bytes": int(total_mb*1024*1024), "elapsed": channel_elapsed, "results": results}

def run_year(year: int) -> Dict:
    ranges = build_channel_ranges(year)
    total_expected = sum(len(list(generate_dates(s,e))) for s,e in ranges.values())
    print(f"\n{'#'*80}\nSTARTING YEAR {year} (expected {total_expected} files)\n{'#'*80}")
    year_start = time.perf_counter()
    channel_results = []
    with ThreadPoolExecutor(max_workers=CHANNEL_COUNT) as ce:
        futures = {ce.submit(run_channel, year, cid, s, e): cid for cid,(s,e) in ranges.items()}
        for future in as_completed(futures):
            channel_results.append(future.result())
    year_elapsed = time.perf_counter() - year_start
    successful = sum(r["successful"] for r in channel_results)
    failed = sum(r["failed"] for r in channel_results)
    total_bytes = sum(r["bytes"] for r in channel_results)
    print(f"\nYear {year} summary: {successful}/{total_expected} valid, {failed} failed, {total_bytes/(1024*1024):.2f} MB")
    return {"year": year, "expected": total_expected, "successful": successful,
            "failed": failed, "bytes": total_bytes, "elapsed": year_elapsed, "channel_results": channel_results}

def generate_report(year_results, overall_start):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    total_expected = sum(r["expected"] for r in year_results)
    total_success = sum(r["successful"] for r in year_results)
    total_failed = sum(r["failed"] for r in year_results)
    lines = [
        "="*100,
        "NOAA OISST Download and Subsetting Report",
        "="*100,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: NOAA/NCEI OISST v2.1 (AVHRR-only, 0.25°)",
        f"Period: {START_YEAR}-01-01 to {END_YEAR}-12-31",
        f"Bounding box: Lat [{LAT_MIN},{LAT_MAX}], Lon [{LON_MIN},{LON_MAX}]",
        f"Expected files: {total_expected}",
        f"Valid files: {total_success}",
        f"Failed files: {total_failed}",
        f"Coverage: {total_success/total_expected*100:.2f}%",
        f"Total wall time: {time.perf_counter()-overall_start:.2f}s",
        "="*100
    ]
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

def main():
    global downloaded_count, skipped_count, failed_count, subsetted_count, already_subset_count, active_requests
    PRODUCTION_ROOT.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    print("="*100)
    print("OceanEmbed NOAA OISST v2.1 Downloader (Fixed)")
    print(f"Period: {START_YEAR}–{END_YEAR}, BBox: Lat [{LAT_MIN},{LAT_MAX}], Lon [{LON_MIN},{LON_MAX}]")
    print(f"Parallel: {CHANNEL_COUNT} channels × {WORKERS_PER_CHANNEL} workers = {CHANNEL_COUNT*WORKERS_PER_CHANNEL} concurrent")
    print("="*100)
    overall_start = time.perf_counter()
    year_results = []
    for year in range(START_YEAR, END_YEAR+1):
        year_results.append(run_year(year))
    generate_report(year_results, overall_start)
    print(f"\nDone. Report: {REPORT_PATH}")

if __name__ == "__main__":
    main()