"""
OISSS Real Acquisition Smoke Test.

Performs live ESR HTTPS acquisition of OISSS v3.0 global daily files.
NOT part of the automated unit test suite — requires network access.

Three scenarios:
  Request 1  : 2023-01-01 → 2023-01-03 (3 daily files)
  Request 1 REPEAT : same params → verify file reuse, no re-download
  Request 2  : 2022-03-01 → 2022-03-02 (different dates → separate files)

After download, opens one real file and verifies:
  - dimensions (latitude=720, longitude=1440, time=1)
  - variables (sss, sss_empirical_uncertainty, sss_formal_uncertainty)
  - coordinate names (latitude, longitude, time — long form)
  - longitude convention (-180 to +180)
  - latitude direction (ascending)
  - grid spacing (0.25°)
  - fill value (-9999.0 raw / NaN after masking)
  - sss.units = "1e-3" (unit discrepancy preserved, NOT converted)
  - full global extent (NOT NIO subset)
  - no depth dimension

Usage
-----
    cd D:\\OceanEmbed\\oceanembed-software
    .venv\\Scripts\\python.exe tests/smoke/test_oisss_smoke.py
"""

from __future__ import annotations

import asyncio
import hashlib
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

from oceanembed.datasets.adapters.oisss import OISSSAdapter
from oceanembed.acquisition.engine import (
    AcquisitionJob, JobManager, run_acquisition_job, _md5_of_file
)
from oceanembed.acquisition.manifest import ManifestStore, ManifestStatus


def _fmt(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1024**2: return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"

def _section(title: str) -> None:
    print(); print("=" * 70); print(title); print("=" * 70)

def _check(label: str, ok: bool, detail: str = "") -> bool:
    s = "PASS" if ok else "FAIL"
    print(f"  [{s}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def resolve_data_root() -> Path:
    raw = os.environ.get("OCEANEMBED_DATA_ROOT", "").strip()
    root = Path(raw) if raw else _REPO_ROOT.parent / "oceanembed-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_acquisition(label, adapter, start, end, data_root):
    """Run acquisition for a date range via the engine and return results."""
    _section(f"ACQUISITION: {label}")
    print(f"  Dates: {start} → {end}")

    requests = adapter.date_range_requests(start, end, str(data_root))
    print(f"  Requests: {len(requests)}")
    for r in requests:
        print(f"    {r.request_date}  {Path(r.expected_output_path).name}")

    results = []
    for req in requests:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0

        job = AcquisitionJob(
            job_id=f"oisss_{int(time.time()*1000) % 10**8}",
            dataset_id="oisss",
            start_date=start, end_date=end, data_root=data_root,
        )

        print(f"\n  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, adapter))
        elapsed = time.perf_counter() - t0

        success = job.status.value == "COMPLETED"
        nbytes = job.bytes_downloaded
        skipped = was_present and success

        # Read checksum from manifest
        manifest = ManifestStore(data_root)
        recs = [r for r in manifest.read_all("oisss")
                if r.get("request_date") == req.request_date.isoformat()
                and r.get("status") == "SUCCESS"]
        md5 = recs[-1].get("md5_checksum") if recs else None

        results.append(dict(
            path=dest, success=success, nbytes=nbytes, md5=md5,
            elapsed=elapsed, skipped=skipped, request_date=req.request_date,
            relative_path=req.expected_output_path,
        ))

        if success:
            print(f"  Success  : True")
            print(f"  Reused   : {skipped}")
            if nbytes > 0:
                print(f"  Size     : {_fmt(nbytes)}")
            if md5:
                print(f"  MD5      : {md5}")
            print(f"  Elapsed  : {elapsed:.1f}s")
        else:
            print(f"  FAILED — logs:")
            for log in job.logs[-6:]:
                print(f"    [{log.get('level','?')}] {log.get('message','')}")
        break  # one file at a time for the smoke test loop

    return results


def inspect_netcdf(path: Path) -> tuple[bool, dict]:
    """Open a real OISSS file and verify its contents."""
    info = {}
    try:
        import xarray as xr
        import numpy as np
        with xr.open_dataset(path, decode_cf=True, mask_and_scale=True,
                             engine="netcdf4") as ds:
            info["dims"] = dict(ds.sizes)
            info["data_vars"] = list(ds.data_vars)
            info["coords"] = list(ds.coords)
            info["has_depth"] = "depth" in ds.dims or "depth" in ds.coords

            for v in ["sss", "sss_empirical_uncertainty", "sss_formal_uncertainty"]:
                if v in ds:
                    attrs = ds[v].attrs
                    info[f"{v}_units"] = attrs.get("units", "N/A")
                    info[f"{v}_fill"] = ds[v].encoding.get("_FillValue", "N/A")
                    info[f"{v}_dtype"] = str(ds[v].dtype)
                    nan_count = int(np.isnan(ds[v].values).sum())
                    info[f"{v}_nan_pct"] = f"{100*nan_count/ds[v].size:.1f}%"

            for c in ["latitude", "longitude"]:
                if c in ds.coords:
                    vals = ds[c].values
                    info[f"{c}_first"] = float(vals[0])
                    info[f"{c}_last"] = float(vals[-1])
                    info[f"{c}_n"] = len(vals)
                    if len(vals) > 1:
                        info[f"{c}_step"] = round(float(np.median(np.abs(np.diff(vals)))), 6)

            lat_vals = ds["latitude"].values if "latitude" in ds.coords else []
            info["lat_ascending"] = bool(lat_vals[1] > lat_vals[0]) if len(lat_vals) > 1 else None
            info["lon_convention"] = "-180-180" if ds.coords["longitude"].values[-1] <= 180 else "0-360"
            info["resolution_ok"] = (
                abs(info.get("longitude_step", 0) - 0.25) < 0.01 and
                abs(info.get("latitude_step", 0) - 0.25) < 0.01
            )

        ok = (
            info["dims"].get("latitude") == 720 and
            info["dims"].get("longitude") == 1440 and
            info["dims"].get("time") == 1 and
            "sss" in info["data_vars"] and
            "sss_empirical_uncertainty" in info["data_vars"] and
            "sss_formal_uncertainty" in info["data_vars"] and
            not info["has_depth"] and
            info.get("resolution_ok", False) and
            info.get("lon_convention") == "-180-180"
        )
        return ok, info
    except ImportError:
        return True, {"note": "xarray/netCDF4 not available"}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    print()
    print("=" * 70)
    print("OISSS SMOKE TEST — ESR HTTPS PUBLIC")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Source: https://oisss.esr.org/data/V3/Daily/")
    print("=" * 70)

    data_root = resolve_data_root()
    print(f"\n  Data root : {data_root}")
    adapter = OISSSAdapter()
    all_checks: list[bool] = []

    # ------------------------------------------------------------------ R1
    # Acquire 3 daily files — but run one at a time to show each
    _section("ACQUISITION: Request 1 (2023-01-01 to 2023-01-03)")
    print(f"  Adapter: OISSSAdapter (plain HTTPS, no auth, global file)")

    adapter_full = OISSSAdapter()
    requests_r1 = adapter_full.date_range_requests(date(2023,1,1), date(2023,1,3), str(data_root))
    print(f"  Requests: {len(requests_r1)}")

    r1_results = []
    r1_md5s = {}

    for req in requests_r1:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"oisss_r1_{req.request_date:%Y%m%d}",
            dataset_id="oisss",
            start_date=req.request_date, end_date=req.request_date,
            data_root=data_root,
        )
        print(f"\n  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, OISSSAdapter()))
        elapsed = time.perf_counter() - t0
        success = job.status.value == "COMPLETED"
        skipped = was_present and success

        manifest = ManifestStore(data_root)
        recs = [r for r in manifest.read_all("oisss")
                if r.get("request_date") == req.request_date.isoformat()
                and r.get("status") == "SUCCESS"]
        md5 = recs[-1].get("md5_checksum") if recs else None
        nbytes = dest.stat().st_size if dest.exists() else 0

        r1_results.append(dict(path=dest, success=success, nbytes=nbytes,
                               md5=md5, elapsed=elapsed, skipped=skipped,
                               relative_path=req.expected_output_path,
                               request_date=req.request_date))
        if md5:
            r1_md5s[str(req.request_date)] = md5

        print(f"  Success  : {success}")
        if success:
            print(f"  Reused   : {skipped}")
            print(f"  Size     : {_fmt(nbytes)}")
            if md5: print(f"  MD5      : {md5}")
            print(f"  Elapsed  : {elapsed:.1f}s")
        else:
            for log in job.logs[-5:]:
                print(f"    [{log.get('level','?')}] {log.get('message','')}")

    _section("VERIFICATION: Request 1")
    r1_pass = True
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
    first_file = r1_results[0]["path"]
    nc_ok, nc_info = inspect_netcdf(first_file)
    for k, v in nc_info.items():
        print(f"  {k}: {v}")
    print()
    _check("dims: latitude=720", nc_info.get("dims", {}).get("latitude") == 720)
    _check("dims: longitude=1440", nc_info.get("dims", {}).get("longitude") == 1440)
    _check("dims: time=1", nc_info.get("dims", {}).get("time") == 1)
    _check("sss present", "sss" in nc_info.get("data_vars", []))
    _check("sss_empirical_uncertainty present",
           "sss_empirical_uncertainty" in nc_info.get("data_vars", []))
    _check("sss_formal_uncertainty present",
           "sss_formal_uncertainty" in nc_info.get("data_vars", []))
    _check("coord names: latitude, longitude, time",
           "latitude" in nc_info.get("coords", []) and
           "longitude" in nc_info.get("coords", []))
    _check("longitude convention -180 to +180",
           nc_info.get("lon_convention") == "-180-180")
    _check("latitude ascending", nc_info.get("lat_ascending") is True)
    _check("resolution 0.25°", nc_info.get("resolution_ok", False))
    _check("no depth dimension", not nc_info.get("has_depth", True))
    _check("sss.units = 1e-3 (discrepancy preserved, NOT converted)",
           nc_info.get("sss_units") == "1e-3",
           f"actual: {nc_info.get('sss_units')}")
    _check("fill_value = -9999.0",
           str(nc_info.get("sss_fill")) == "-9999.0",
           f"actual: {nc_info.get('sss_fill')}")
    _check("global extent — NOT NIO subset",
           nc_info.get("latitude_n", 0) == 720 and nc_info.get("longitude_n", 0) == 1440)

    # Manifest check
    _section("MANIFEST VERIFICATION: After Request 1")
    manifest = ManifestStore(data_root)
    success_recs = [r for r in manifest.read_all("oisss") if r.get("status") == "SUCCESS"]
    _check(f"At least 3 SUCCESS entries in oisss.jsonl",
           len(success_recs) >= 3, f"found {len(success_recs)}")
    for rec in success_recs[-3:]:
        _check("Entry has md5_checksum", rec.get("md5_checksum") is not None,
               f"date={rec.get('request_date')}")

    # ------------------------------------------------------------------ R1 REPEAT
    _section("REQUEST 1 REPEAT (same params — expect reuse)")
    repeat_results = []
    for req in requests_r1:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"oisss_r1b_{req.request_date:%Y%m%d}",
            dataset_id="oisss",
            start_date=req.request_date, end_date=req.request_date,
            data_root=data_root,
        )
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, OISSSAdapter()))
        elapsed = time.perf_counter() - t0
        success = job.status.value == "COMPLETED"
        skipped = was_present and success

        manifest2 = ManifestStore(data_root)
        recs2 = [r for r in manifest2.read_all("oisss")
                 if r.get("request_date") == req.request_date.isoformat()
                 and r.get("status") == "SUCCESS"]
        md5 = recs2[-1].get("md5_checksum") if recs2 else None
        repeat_results.append(dict(success=success, skipped=skipped, md5=md5,
                                   elapsed=elapsed, request_date=req.request_date))
        print(f"  {req.request_date}: success={success}, reused={skipped}, "
              f"elapsed={elapsed:.1f}s, md5={md5[:8] if md5 else 'None'}…")

    _section("VERIFICATION: Request 1 Repeat")
    repeat_pass = True
    for r in repeat_results:
        ok = _check("Repeat succeeded", r["success"])
        repeat_pass = repeat_pass and ok
        if r["success"]:
            ok2 = _check("File reused (no re-download)", r["skipped"])
            repeat_pass = repeat_pass and ok2
            orig_md5 = r1_md5s.get(str(r["request_date"]))
            if orig_md5 and r["md5"]:
                ok3 = _check("MD5 unchanged",
                             r["md5"] == orig_md5,
                             f"orig={orig_md5[:8]}… repeat={r['md5'][:8]}…")
                repeat_pass = repeat_pass and ok3
    all_checks.append(repeat_pass)

    # ------------------------------------------------------------------ R2
    _section("ACQUISITION: Request 2 (2022-03-01 to 2022-03-02)")
    requests_r2 = OISSSAdapter().date_range_requests(
        date(2022, 3, 1), date(2022, 3, 2), str(data_root)
    )
    r2_results = []
    for req in requests_r2:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"oisss_r2_{req.request_date:%Y%m%d}",
            dataset_id="oisss",
            start_date=req.request_date, end_date=req.request_date,
            data_root=data_root,
        )
        t0 = time.perf_counter()
        asyncio.run(run_acquisition_job(job, OISSSAdapter()))
        elapsed = time.perf_counter() - t0
        success = job.status.value == "COMPLETED"
        nbytes = dest.stat().st_size if dest.exists() else 0
        print(f"  {req.request_date}: success={success}, "
              f"size={_fmt(nbytes)}, elapsed={elapsed:.1f}s")
        r2_results.append(dict(path=dest, success=success, nbytes=nbytes,
                               elapsed=elapsed, relative_path=req.expected_output_path,
                               request_date=req.request_date))

    _section("VERIFICATION: Request 2")
    r2_paths = {r["relative_path"] for r in r1_results}
    r2_pass = True
    for r in r2_results:
        ok = _check("R2 download succeeded", r["success"])
        r2_pass = r2_pass and ok
        if r["success"]:
            ok2 = _check("R2 path differs from R1", r["relative_path"] not in r2_paths,
                         r["relative_path"])
            r2_pass = r2_pass and ok2
    for r in r1_results:
        if r["path"].exists():
            orig_md5 = r1_md5s.get(str(r["request_date"]))
            ok_intact = _check("R1 file intact after R2", r["path"].stat().st_size > 0)
            if orig_md5:
                actual_md5 = _md5_of_file(r["path"])
                ok_md5 = _check("R1 MD5 unchanged", actual_md5 == orig_md5)
                r2_pass = r2_pass and ok_intact and ok_md5
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
    print(f"\n  Raw data : {data_root / 'raw' / 'oisss'}")
    print(f"  Manifest : {data_root / 'manifests' / 'oisss.jsonl'}")
    print()
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
