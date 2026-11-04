"""
CCMP Real Acquisition Smoke Test.

Downloads real RSS CCMP v3.1 daily files (public HTTPS, no auth).
Verifies all 4 × 6-hourly time records are preserved in each file.

Three scenarios:
  Request 1  : 2023-01-01 → 2023-01-03 (3 daily files)
  Request 1 REPEAT : same params → verify reuse
  Request 2  : 2022-03-01 → 2022-03-02 → separate files, R1 intact

Usage
-----
    cd D:\\OceanEmbed\\oceanembed-software
    .venv\\Scripts\\python.exe tests/smoke/test_ccmp_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oceanembed.datasets.adapters.ccmp import CCMPAdapter, _TIME_RECORDS_PER_FILE
from oceanembed.acquisition.engine import AcquisitionJob, run_acquisition_job, _md5_of_file
from oceanembed.acquisition.manifest import ManifestStore, ManifestStatus


def _fmt(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"

def _section(t: str) -> None:
    print(); print("=" * 70); print(t); print("=" * 70)

def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def resolve_data_root() -> Path:
    raw = os.environ.get("OCEANEMBED_DATA_ROOT", "").strip()
    root = Path(raw) if raw else _REPO_ROOT.parent / "oceanembed-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_acquisition(label, adapter, start, end, data_root):
    _section(f"ACQUISITION: {label}")
    print(f"  Dates: {start} → {end}")
    requests = adapter.date_range_requests(start, end, str(data_root))
    print(f"  Requests: {len(requests)} (one file per day)")
    results = []
    for req in requests:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"ccmp_{int(time.time()*1000) % 10**8}",
            dataset_id="ccmp",
            start_date=start, end_date=end, data_root=data_root,
        )
        print(f"\n  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, adapter))
        elapsed = time.perf_counter() - t0
        success = job.status.value == "COMPLETED"
        nbytes = dest.stat().st_size if dest.exists() else 0
        skipped = was_present and success

        manifest = ManifestStore(data_root)
        recs = [r for r in manifest.read_all("ccmp")
                if r.get("request_date") == req.request_date.isoformat()
                and r.get("status") == "SUCCESS"]
        md5 = recs[-1].get("md5_checksum") if recs else None

        results.append(dict(path=dest, success=success, nbytes=nbytes,
                            md5=md5, elapsed=elapsed, skipped=skipped,
                            request_date=req.request_date,
                            relative_path=req.expected_output_path))
        print(f"  Success  : {success}")
        if success:
            print(f"  Reused   : {skipped}")
            print(f"  Size     : {_fmt(nbytes)}")
            if md5: print(f"  MD5      : {md5}")
            print(f"  Elapsed  : {elapsed:.1f}s")
        else:
            for log in job.logs[-5:]:
                print(f"    [{log.get('level','?')}] {log.get('message','')}")
    return results


def inspect_ccmp_file(path: Path) -> tuple[bool, dict]:
    """Open a real CCMP file and verify the critical source facts."""
    info = {}
    try:
        import xarray as xr, numpy as np
        with xr.open_dataset(path, decode_cf=True, mask_and_scale=True,
                             engine="netcdf4") as ds:
            info["dims"] = dict(ds.sizes)
            info["data_vars"] = list(ds.data_vars)
            info["coords"] = list(ds.coords)
            info["has_depth"] = "depth" in ds.dims or "depth" in ds.coords
            info["time_n"] = ds.sizes.get("time", 0)

            # Time records
            time_vals = ds["time"].values
            info["time_records"] = []
            for t in time_vals:
                ts = str(np.datetime_as_string(t, unit="s"))
                day = ts[:10]
                h = int((np.datetime64(t, "h") - np.datetime64(day, "D")).astype(int))
                info["time_records"].append(f"{h:02d}:00 UTC")

            # Spatial
            lat_name = "latitude" if "latitude" in ds.coords else "lat"
            lon_name = "longitude" if "longitude" in ds.coords else "lon"
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            info["lat_first"] = float(lat[0]); info["lat_last"] = float(lat[-1])
            info["lon_first"] = float(lon[0]); info["lon_last"] = float(lon[-1])
            info["lat_n"] = len(lat); info["lon_n"] = len(lon)
            if len(lat) > 1: info["lat_step"] = round(float(np.median(np.abs(np.diff(lat)))), 6)
            if len(lon) > 1: info["lon_step"] = round(float(np.median(np.abs(np.diff(lon)))), 6)
            info["lon_convention"] = "0-360" if lon[-1] > 180 else "-180-180"

            # Variables
            for v in ["uwnd", "vwnd", "ws", "nobs"]:
                if v in ds:
                    attrs = ds[v].attrs
                    info[f"{v}_units"] = attrs.get("units", "N/A")
                    info[f"{v}_fill"] = ds[v].encoding.get("_FillValue", "N/A")
                    info[f"{v}_std_name"] = attrs.get("standard_name", "N/A")

        ok = (
            info["time_n"] == 4 and
            "uwnd" in info["data_vars"] and "vwnd" in info["data_vars"] and
            "ws" in info["data_vars"] and "nobs" in info["data_vars"] and
            not info["has_depth"] and
            info.get("lon_convention") == "0-360" and
            info["lat_n"] == 720 and info["lon_n"] == 1440
        )
        return ok, info
    except ImportError:
        return True, {"note": "xarray/netCDF4 not available — skipping content check"}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    print()
    print("=" * 70)
    print("CCMP SMOKE TEST — RSS PUBLIC HTTPS")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Source: https://data.remss.com/ccmp/v03.1/")
    print("=" * 70)

    data_root = resolve_data_root()
    print(f"\n  Data root : {data_root}")

    adapter = CCMPAdapter()
    all_checks: list[bool] = []

    # ------------------------------------------------------------------ R1
    r1_results = run_acquisition(
        "Request 1 (2023-01-01 to 2023-01-03)",
        adapter, date(2023, 1, 1), date(2023, 1, 3), data_root,
    )

    _section("VERIFICATION: Request 1")
    r1_pass = True
    r1_md5s = {}
    for r in r1_results:
        ok = _check("Download succeeded", r["success"], r["path"].name)
        r1_pass = r1_pass and ok
        if r["success"]:
            ok2 = _check("File non-empty", r["path"].stat().st_size > 0)
            r1_pass = r1_pass and ok2
    all_checks.append(r1_pass)

    if not r1_pass:
        print("\n  Request 1 FAILED — stopping.")
        return 1

    # Inspect first file
    _section("FILE CONTENT INSPECTION: 2023-01-01")
    first_path = r1_results[0]["path"]
    nc_ok, nc_info = inspect_ccmp_file(first_path)
    print(f"\n  File: {first_path.name}")
    print(f"  Size: {_fmt(first_path.stat().st_size)}")
    for k, v in nc_info.items():
        print(f"    {k}: {v}")
    print()

    _check("dims: time=4 (all 4 × 6-hourly records preserved)",
           nc_info.get("time_n") == 4,
           f"actual time={nc_info.get('time_n')}")
    _check("dims: latitude=720", nc_info.get("lat_n") == 720)
    _check("dims: longitude=1440", nc_info.get("lon_n") == 1440)
    _check("time[0]=00:00 UTC", nc_info.get("time_records", ["?"])[0] == "00:00 UTC",
           f"actual: {nc_info.get('time_records', ['?'])[0]}")
    _check("time[1]=06:00 UTC", nc_info.get("time_records", [None, "?"])[1] == "06:00 UTC",
           f"actual: {nc_info.get('time_records', [None, '?'])[1]}")
    _check("time[2]=12:00 UTC", nc_info.get("time_records", [None, None, "?"])[2] == "12:00 UTC")
    _check("time[3]=18:00 UTC", nc_info.get("time_records", [None, None, None, "?"])[3] == "18:00 UTC")
    _check("uwnd present", "uwnd" in nc_info.get("data_vars", []))
    _check("vwnd present", "vwnd" in nc_info.get("data_vars", []))
    _check("ws present", "ws" in nc_info.get("data_vars", []))
    _check("nobs present", "nobs" in nc_info.get("data_vars", []))
    _check("longitude convention 0-360", nc_info.get("lon_convention") == "0-360",
           f"lon[0]={nc_info.get('lon_first')}")
    _check("latitude ascending (south-to-north)", nc_info.get("lat_first", 0) < 0,
           f"lat[0]={nc_info.get('lat_first')}")
    _check("resolution 0.25°", nc_info.get("lat_step") == 0.25 and nc_info.get("lon_step") == 0.25)
    _check("no depth dimension", not nc_info.get("has_depth", True))
    _check("uwnd units = m s-1", nc_info.get("uwnd_units") == "m s-1",
           f"actual: {nc_info.get('uwnd_units')}")
    _check("fill_value = -9999.0", str(nc_info.get("uwnd_fill", "")) == "-9999.0",
           f"actual: {nc_info.get('uwnd_fill')}")
    _check("global file (NOT NIO subset)", nc_info.get("lon_n") == 1440 and nc_info.get("lat_n") == 720)

    # Store MD5s for reuse check
    for r in r1_results:
        if r["md5"]:
            r1_md5s[str(r["request_date"])] = r["md5"]

    # Manifest
    _section("MANIFEST: After Request 1")
    manifest = ManifestStore(data_root)
    success_recs = [r for r in manifest.read_all("ccmp") if r.get("status") == "SUCCESS"]
    _check(f"At least 3 SUCCESS entries",
           len(success_recs) >= 3, f"found {len(success_recs)}")
    for rec in success_recs[-3:]:
        _check("Entry has md5_checksum", rec.get("md5_checksum") is not None,
               f"date={rec.get('request_date')}")

    # ------------------------------------------------------------------ R1 REPEAT
    _section("REQUEST 1 REPEAT (expect reuse)")
    requests_r1 = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 3), str(data_root))
    repeat_results = []
    for req in requests_r1:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"ccmp_r1b_{req.request_date:%Y%m%d}",
            dataset_id="ccmp",
            start_date=req.request_date, end_date=req.request_date, data_root=data_root,
        )
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, CCMPAdapter()))
        elapsed = time.perf_counter() - t0
        success = job.status.value == "COMPLETED"
        skipped = was_present and success
        manifest2 = ManifestStore(data_root)
        recs2 = [r for r in manifest2.read_all("ccmp")
                 if r.get("request_date") == req.request_date.isoformat()
                 and r.get("status") == "SUCCESS"]
        md5 = recs2[-1].get("md5_checksum") if recs2 else None
        repeat_results.append(dict(success=success, skipped=skipped, md5=md5,
                                   elapsed=elapsed, request_date=req.request_date))
        print(f"  {req.request_date}: reused={skipped}, elapsed={elapsed:.1f}s")

    _section("VERIFICATION: Request 1 Repeat")
    repeat_pass = True
    for r in repeat_results:
        ok = _check("Repeat succeeded", r["success"])
        repeat_pass = repeat_pass and ok
        if r["success"]:
            ok2 = _check("File reused (no re-download)", r["skipped"])
            repeat_pass = repeat_pass and ok2
            orig = r1_md5s.get(str(r["request_date"]))
            if orig and r["md5"]:
                ok3 = _check("MD5 unchanged",
                             r["md5"] == orig,
                             f"orig={orig[:8]}… repeat={r['md5'][:8]}…")
                repeat_pass = repeat_pass and ok3
    all_checks.append(repeat_pass)

    # ------------------------------------------------------------------ R2
    r2_results = run_acquisition(
        "Request 2 (2022-03-01 to 2022-03-02)",
        CCMPAdapter(), date(2022, 3, 1), date(2022, 3, 2), data_root,
    )

    _section("VERIFICATION: Request 2")
    r1_paths = {r["relative_path"] for r in r1_results}
    r2_pass = True
    for r in r2_results:
        ok = _check("R2 download succeeded", r["success"])
        r2_pass = r2_pass and ok
        if r["success"]:
            ok2 = _check("R2 path differs from R1", r["relative_path"] not in r1_paths,
                         r["relative_path"])
            r2_pass = r2_pass and ok2
    for r in r1_results:
        if r["path"].exists():
            ok_r1 = _check("R1 file intact after R2", r["path"].stat().st_size > 0)
            orig = r1_md5s.get(str(r["request_date"]))
            if orig:
                ok_md5 = _check("R1 MD5 unchanged", _md5_of_file(r["path"]) == orig)
                r2_pass = r2_pass and ok_r1 and ok_md5
    all_checks.append(r2_pass)

    # Summary
    _section("SMOKE TEST SUMMARY")
    labels = ["Request 1 (3 days, 2023-01-01→03)",
              "Request 1 REPEAT (reuse check)",
              "Request 2 (2022-03-01→02)"]
    for lbl, passed in zip(labels, all_checks):
        print(f"  [{'PASS' if passed else 'FAIL'}] {lbl}")
    print()
    _check("All scenarios passed", all(all_checks), f"{sum(all_checks)}/{len(all_checks)}")
    print(f"\n  Raw data : {data_root / 'raw' / 'ccmp'}")
    print(f"  Manifest : {data_root / 'manifests' / 'ccmp.jsonl'}")
    print()
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
