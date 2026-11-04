#!/usr/bin/env python3
"""
CMEMS DUACS 0.25° Daily Sea Level Downloader for OceanEmbed

Progress bars enabled, concurrency 4, warnings suppressed.

Source: c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D
Resolution: 0.25° (twosat product — two-satellite merged)
Chunking: quarterly (Q1–Q4 per year)
Variables: sla, adt, ugos, vgos

IMPORTANT — OceanEmbed engineering notes:
------------------------------------------
1. The output path (OUTPUT_ROOT) is hard-coded to E:\Ocean_data\CMEMS_DUACS\data\raw.
   In production OceanEmbed, this must be replaced with ${data_root}/raw/duacs/.
   The path is preserved here exactly as provided for auditing purposes.

2. This script uses the C3S/Copernicus twosat 0.25° product
   (c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D), which is different
   from the previously audited allsat 0.125° product
   (cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202411).
   The native resolution has changed from 0.125° to 0.25°.
   This change resolves the regridding requirement (OSD-005 scope for DUACS reduced).

3. Chunking strategy changed from monthly to quarterly. One output file covers
   approximately 90 days (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec).

4. No scientific transformations are applied. This is raw source acquisition only.

5. CMEMS credentials are required. Do NOT hardcode credentials in this script.
   Use the copernicusmarine credential store or environment variables.

Staged: 2026-09-10
Audited: PARTIAL — see configs/datasets/duacs.yaml and
         docs/datasets/LEGACY_ACQUISITION_AUDIT.md for full audit record.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")  # suppress all warnings (including chunk warnings)

try:
    import copernicusmarine as cm
except ImportError:
    print("ERROR: copernicusmarine package not found. Install it with:")
    print("    pip install copernicusmarine")
    sys.exit(1)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATASET_ID  = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
VARIABLES   = ["sla", "adt", "ugos", "vgos"]

LON_MIN, LON_MAX = 45.0, 105.0
LAT_MIN, LAT_MAX =  5.0,  30.0

START_YEAR = 2012
END_YEAR   = 2024

# NOTE: These paths are hard-coded to the original developer's machine.
# In OceanEmbed production, use ${data_root}/raw/duacs/ resolved from config.
OUTPUT_ROOT = Path(r"E:\Ocean_data\CMEMS_DUACS\data\raw")
REPORT_PATH = Path(r"E:\Ocean_data\CMEMS_DUACS\data\duacs_download_report.txt")

RES_TOLERANCE       = 0.01
MIN_FILE_SIZE_KB    = 50
MAX_RETRIES         = 3
RETRY_DELAY_SECONDS = 10
CONCURRENT_DOWNLOADS = 4

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    q_start_month = 3 * (quarter - 1) + 1
    q_end_month   = q_start_month + 2
    start = datetime(year, q_start_month, 1)
    if q_end_month == 12:
        end = datetime(year, q_end_month, 31)
    else:
        next_month = q_end_month + 1
        end = datetime(year, next_month, 1) - timedelta(days=1)
    return start, end


def expected_daily_count(start: datetime, end: datetime) -> int:
    return (end - start).days + 1


def validate_file(filepath: Path, start: datetime, end: datetime) -> Dict:
    errors = []
    warnings_list = []
    info = {}

    if not filepath.exists():
        errors.append(f"File does not exist: {filepath}")
        return {"valid": False, "errors": errors, "warnings": warnings_list, "info": info}

    if filepath.stat().st_size < MIN_FILE_SIZE_KB * 1024:
        errors.append("File size below minimum.")
        return {"valid": False, "errors": errors, "warnings": warnings_list, "info": info}

    try:
        ds = xr.open_dataset(filepath, decode_times=True, mask_and_scale=True)
    except Exception as e:
        errors.append(f"Cannot open: {e}")
        return {"valid": False, "errors": errors, "warnings": warnings_list, "info": info}

    missing_vars = [v for v in VARIABLES if v not in ds.variables]
    if missing_vars:
        errors.append(f"Missing variables: {missing_vars}")
    else:
        info["variables_found"] = list(VARIABLES)

    lat_name = lon_name = time_name = None
    for coord in ds.coords:
        if "lat"  in coord.lower(): lat_name  = coord
        if "lon"  in coord.lower(): lon_name  = coord
        if "time" in coord.lower(): time_name = coord

    if not lat_name:  errors.append("Latitude coordinate not found")
    if not lon_name:  errors.append("Longitude coordinate not found")
    if not time_name: errors.append("Time coordinate not found")

    if lat_name and lon_name and time_name:
        lat      = ds[lat_name].values
        lon      = ds[lon_name].values
        time_vals = ds[time_name].values

        if len(lat) > 1:
            lat_spacing = np.mean(np.diff(lat))
            info["lat_spacing"] = lat_spacing
            if abs(lat_spacing - 0.25) > RES_TOLERANCE:
                errors.append("Lat spacing not 0.25°")
            lat_min, lat_max = lat.min(), lat.max()
            info["lat_min"], info["lat_max"] = lat_min, lat_max
            info["lat_count"] = len(lat)
            if lat_min < LAT_MIN - 0.125 or lat_max > LAT_MAX + 0.125:
                errors.append("Lat bounds exceed requested")
        else:
            errors.append("Latitude only one point")

        if len(lon) > 1:
            lon_spacing = np.mean(np.diff(lon))
            info["lon_spacing"] = lon_spacing
            if abs(lon_spacing - 0.25) > RES_TOLERANCE:
                errors.append("Lon spacing not 0.25°")
            lon_min, lon_max = lon.min(), lon.max()
            info["lon_min"], info["lon_max"] = lon_min, lon_max
            info["lon_count"] = len(lon)
            if lon_min < LON_MIN - 0.125 or lon_max > LON_MAX + 0.125:
                errors.append("Lon bounds exceed requested")
        else:
            errors.append("Longitude only one point")

        if hasattr(time_vals, "dtype") and np.issubdtype(time_vals.dtype, np.datetime64):
            times = pd.DatetimeIndex(time_vals)
        else:
            try:
                times = pd.to_datetime([t.isoformat() for t in time_vals])
            except Exception:
                errors.append("Could not decode time")
                times = None

        if times is not None:
            if times.duplicated().any():
                errors.append("Duplicate timestamps")
            daily_diff = np.diff(times).astype("timedelta64[D]").astype(int)
            if not np.all(daily_diff == 1):
                errors.append("Not daily")
            expected_start = pd.Timestamp(start)
            expected_end   = pd.Timestamp(end)
            if times.min() != expected_start or times.max() != expected_end:
                errors.append("Time range mismatch")
            expected_count = expected_daily_count(start, end)
            if len(times) != expected_count:
                errors.append(f"Expected {expected_count} steps, got {len(times)}")
            else:
                info["time_count"] = len(times)

        for var in VARIABLES:
            if var in ds.variables and ds[var].ndim < 3:
                errors.append(f"Variable {var} wrong dimensions")

        info["coords"] = {"lat": lat_name, "lon": lon_name, "time": time_name}

    ds.close()
    valid = len(errors) == 0
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings_list,
        "info": info,
        "filepath": str(filepath),
    }


def download_quarter(year: int, quarter: int) -> Optional[Path]:
    start, end = get_quarter_dates(year, quarter)
    quarter_str = f"{year}_Q{quarter}"

    year_dir    = OUTPUT_ROOT / str(year)
    quarter_dir = year_dir / f"Q{quarter}"
    quarter_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"duacs_{year}_Q{quarter}.nc"
    output_path     = quarter_dir / output_filename

    if output_path.exists():
        validation = validate_file(output_path, start, end)
        if validation["valid"]:
            print(f"  [SKIP] {output_filename} already valid")
            return output_path
        else:
            print(f"  [WARN] {output_filename} invalid, re-downloading")
            output_path.unlink()

    download_kwargs = {
        "dataset_id":        DATASET_ID,
        "variables":         VARIABLES,
        "minimum_longitude": LON_MIN,
        "maximum_longitude": LON_MAX,
        "minimum_latitude":  LAT_MIN,
        "maximum_latitude":  LAT_MAX,
        "start_datetime":    start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_datetime":      end.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_filename":   output_filename,
        "output_directory":  str(quarter_dir),
        "overwrite":         True,
        # Progress bar enabled by default; no 'disable_progress_bar' argument
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            cm.subset(**download_kwargs)
            if not output_path.exists():
                raise FileNotFoundError(f"File {output_path} not created")
            validation = validate_file(output_path, start, end)
            if validation["valid"]:
                print(f"  [OK] {output_filename}")
                return output_path
            else:
                print(f"  [FAIL] {output_filename} validation failed: {validation['errors']}")
                output_path.unlink(missing_ok=True)
                if attempt == MAX_RETRIES:
                    return None
        except Exception as e:
            print(f"  [ERROR] {quarter_str}: {e}")
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return None
            time.sleep(RETRY_DELAY_SECONDS)

    return None


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_quarters = 0
    success_count  = 0
    failed_quarters: List[str] = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Year {year}: downloading Q1-Q4 concurrently")
        quarters = list(range(1, 5))
        total_quarters += len(quarters)

        with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as executor:
            future_to_quarter = {
                executor.submit(download_quarter, year, q): q
                for q in quarters
            }
            for future in as_completed(future_to_quarter):
                q = future_to_quarter[future]
                try:
                    result = future.result()
                    if result is not None:
                        success_count += 1
                    else:
                        failed_quarters.append(f"{year} Q{q}")
                except Exception as e:
                    print(f"  [ERROR] Unexpected error for {year} Q{q}: {e}")
                    failed_quarters.append(f"{year} Q{q}")

        time.sleep(1)

    # Write report
    report_lines = [
        "=" * 80,
        "DUACS 0.25° Download Report",
        "=" * 80,
        f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset ID: {DATASET_ID}",
        f"Requested time range: {START_YEAR}-01-01 to {END_YEAR}-12-31",
        f"Requested bounding box: Lon [{LON_MIN}, {LON_MAX}], Lat [{LAT_MIN}, {LAT_MAX}]",
        f"Requested resolution: 0.25° x 0.25°",
        f"Variables requested: {', '.join(VARIABLES)}",
        "",
        f"Expected number of quarterly files: 52",
        f"Successful downloads: {success_count}",
        f"Failed downloads: {len(failed_quarters)}",
    ]
    if failed_quarters:
        report_lines.append("Failed quarters:")
        for q in failed_quarters:
            report_lines.append(f"  - {q}")
    report_lines.append("")
    report_lines.append("Note: Coordinate spacing is verified from actual coordinate arrays.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nDownload complete: {success_count}/{total_quarters} successful")
    if failed_quarters:
        print(f"Failed quarters: {failed_quarters}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
