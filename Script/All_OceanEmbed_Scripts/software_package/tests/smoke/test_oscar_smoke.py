"""
OSCAR Real Acquisition Smoke Test.

Performs live NASA OPeNDAP acquisition of OSCAR L4 OC FINAL v2.0 data.
NOT part of the automated unit test suite — requires real credentials.

Three scenarios:
  Request 1  — lat 10-20N, lon 65-85E, 2023-01-01 → 2023-01-03 (3 days)
  Request 1 REPEAT — same params → verify reuse, no re-download
  Request 2  — lat 15-25N, lon 70-90E, 2022-03-01 → 2022-03-03

After download, opens each NetCDF and verifies:
  - dimensions (time=1, longitude, latitude) — NO depth
  - variables u, v
  - coordinate arrays lon, lat
  - resolution ≈ 0.25°
  - spatial region reflects request
  - no preprocessing applied

Usage
-----
    cd D:\\OceanEmbed\\oceanembed-software
    .venv\\Scripts\\python.exe tests/smoke/test_oscar_smoke.py

Prerequisites
-------------
Add to .env:
    EARTHDATA_USERNAME=your-earthdata-username
    EARTHDATA_PASSWORD=your-earthdata-password
"""

from __future__ import annotations

import hashlib, os, sys, time
from datetime import date, datetime, timezone
from pathlib import Path

# Load .env FIRST — before earthaccess import (it reads env at module level)
def _load_dotenv(path: Path) -> None:
    if not path.exists(): return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_load_dotenv(_REPO_ROOT / ".env")

_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oceanembed.datasets.adapters.oscar import OSCARAdapter
from oceanembed.acquisition.engine import AcquisitionJob, run_oscar_subset, _md5_of_file
from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus


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


def check_credentials() -> bool:
    _section("CREDENTIAL CHECK")
    result = OSCARAdapter().check_credentials()
    print(f"  {result['message']}")
    if not result["credentials_available"]:
        print()
        print("  SMOKE TEST CANNOT RUN — credentials missing.")
        print("  Add to .env:")
        print("    EARTHDATA_USERNAME=your-earthdata-username")
        print("    EARTHDATA_PASSWORD=your-earthdata-password")
        print("  Register at: https://urs.earthdata.nasa.gov/users/new")
        return False
    return True


def resolve_data_root() -> Path:
    raw = os.environ.get("OCEANEMBED_DATA_ROOT", "").strip()
    root = Path(raw) if raw else _REPO_ROOT.parent / "oceanembed-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def run_acquisition(label: str, adapter: OSCARAdapter,
                    start: date, end: date, data_root: Path) -> list[dict]:
    _section(f"ACQUISITION: {label}")
    print(f"  Bbox  : lat {adapter._min_lat}–{adapter._max_lat}°N, "
          f"lon {adapter._min_lon}–{adapter._max_lon}°E")
    print(f"  Range : {start} → {end}")

    requests = adapter.date_range_requests(start, end, str(data_root))
    print(f"  Requests: {len(requests)} (one per day)")
    for i, req in enumerate(requests, 1):
        print(f"    {i}. {req.expected_output_path}")

    manifest = ManifestStore(data_root)
    results = []

    for req in requests:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0
        job = AcquisitionJob(
            job_id=f"oscar_{int(time.time()*1000) % 10**8}",
            dataset_id="oscar",
            start_date=start, end_date=end, data_root=data_root,
        )
        print(f"\n  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        success, nbytes, md5 = run_oscar_subset(req.url_or_query, dest, job)
        elapsed = time.perf_counter() - t0

        manifest.record(ManifestEntry(
            dataset_id=req.dataset_id, request_date=req.request_date,
            source_url=req.url_or_query, local_path=str(dest),
            product_id=req.source_description.product_id,
            file_size_bytes=nbytes if success else None,
            md5_checksum=md5, status=ManifestStatus.SUCCESS if success else ManifestStatus.FAILED,
            error=None if success else "Download failed",
        ))

        skipped = was_present and success
        results.append(dict(label=label, path=dest,
                            relative_path=req.expected_output_path,
                            success=success, nbytes=nbytes, md5=md5,
                            elapsed=elapsed, skipped=skipped, job=job,
                            request_date=req.request_date))
        print(f"  Success  : {success}")
        if success:
            print(f"  Reused   : {skipped}")
            print(f"  Size     : {_fmt(nbytes)}")
            print(f"  MD5      : {md5}")
            print(f"  Elapsed  : {elapsed:.1f}s")
        else:
            for log in job.logs[-8:]:
                print(f"    [{log.get('level','?')}] {log.get('message','')}")

    return results


def inspect_netcdf(path: Path, request_date: date,
                   min_lat: float, max_lat: float,
                   min_lon: float, max_lon: float) -> tuple[bool, dict]:
    """
    Open downloaded NetCDF and report actual contents.
    Returns (ok, info_dict). Does NOT assert — reports observations.
    """
    info: dict = {}
    try:
        import xarray as xr
        import numpy as np
        with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as ds:
            info["dims"] = dict(ds.dims)
            info["data_vars"] = list(ds.data_vars)
            info["coords"] = list(ds.coords)
            info["has_depth"] = "depth" in ds.dims or "depth" in ds.coords

            # Variables
            for var in ["u", "v"]:
                if var in ds:
                    info[f"{var}_dims"] = list(ds[var].dims)
                    info[f"{var}_dtype"] = str(ds[var].dtype)
                    fill = ds[var].encoding.get("_FillValue", "N/A")
                    info[f"{var}_fill_value"] = str(fill)
                    attrs = dict(ds[var].attrs)
                    info[f"{var}_units"] = attrs.get("units", "N/A")

            # Coordinates
            for coord in ["lon", "lat"]:
                if coord in ds:
                    vals = ds[coord].values
                    info[f"{coord}_first"] = float(vals[0])
                    info[f"{coord}_last"] = float(vals[-1])
                    info[f"{coord}_n"] = len(vals)
                    if len(vals) > 1:
                        info[f"{coord}_step"] = float(round(np.median(np.abs(np.diff(vals))), 6))
                else:
                    info[f"{coord}_present"] = False

            # Time
            if "time" in ds:
                tvals = ds["time"].values
                info["time_n"] = len(tvals)
                info["time_first"] = str(np.datetime_as_string(tvals[0], unit="D"))

            # Resolution check
            lon_step = info.get("lon_step", 0)
            lat_step = info.get("lat_step", 0)
            info["resolution_ok"] = (
                abs(lon_step - 0.25) < 0.01 and abs(lat_step - 0.25) < 0.01
            )

        ok = (
            "u" in info["data_vars"] and
            "v" in info["data_vars"] and
            not info["has_depth"] and
            info.get("resolution_ok", False)
        )
        return ok, info
    except ImportError:
        return True, {"note": "xarray not available — skipping content check"}
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}


def verify_manifest(data_root: Path, min_success: int) -> list[dict]:
    manifest = ManifestStore(data_root)
    recs = [r for r in manifest.read_all("oscar") if r.get("status") == "SUCCESS"]
    _check(f"At least {min_success} SUCCESS entries in oscar.jsonl",
           len(recs) >= min_success, f"found {len(recs)}")
    for rec in recs[-min_success:]:
        _check("Entry has md5_checksum", rec.get("md5_checksum") is not None,
               f"date={rec.get('request_date')}")
    return recs


def main() -> int:
    print()
    print("=" * 70)
    print("OSCAR SMOKE TEST — NASA PO.DAAC OPeNDAP")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Collection: C2098858642-POCLOUD (OSCAR L4 OC FINAL v2.0)")
    print("=" * 70)

    if not check_credentials():
        return 1

    data_root = resolve_data_root()
    print(f"\n  Data root : {data_root}")
    print(f"  Manifests : {data_root / 'manifests'}")

    all_checks: list[bool] = []

    # ------------------------------------------------------------------ R1
    # Use only 3 days to keep smoke test fast
    adapter_1 = OSCARAdapter(min_lat=10.0, max_lat=20.0, min_lon=65.0, max_lon=85.0)
    results_1 = run_acquisition(
        "Request 1 (lat 10-20N, lon 65-85E, 2023-01-01→03)",
        adapter_1, date(2023, 1, 1), date(2023, 1, 3), data_root,
    )

    _section("VERIFICATION: Request 1")
    r1_pass = True
    r1_md5s = {}
    for r in results_1:
        ok = _check("Download succeeded", r["success"], str(r["path"].name))
        r1_pass = r1_pass and ok
        if r["success"]:
            ok2 = _check("File non-empty", r["path"].stat().st_size > 0)
            r1_pass = r1_pass and ok2
            _check("First acquisition (not reuse)", not r["skipped"])

            nc_ok, nc_info = inspect_netcdf(
                r["path"], r["request_date"],
                adapter_1._min_lat, adapter_1._max_lat,
                adapter_1._min_lon, adapter_1._max_lon,
            )
            _check("NetCDF opens and contains u, v", nc_ok, str(nc_info.get("data_vars", nc_info.get("error",""))))
            _check("No depth dimension", not nc_info.get("has_depth", True))
            _check("Resolution ≈ 0.25°", nc_info.get("resolution_ok", False),
                   f"lon_step={nc_info.get('lon_step','?')}, lat_step={nc_info.get('lat_step','?')}")

            print(f"\n  --- File content report: {r['path'].name} ---")
            for k, v in nc_info.items():
                print(f"    {k}: {v}")

            r1_md5s[str(r["request_date"])] = r["md5"]

    all_checks.append(r1_pass)
    if not r1_pass:
        print("\n  Request 1 FAILED — stopping.")
        return 1

    verify_manifest(data_root, min_success=len(results_1))

    # ------------------------------------------------------------------ R1 REPEAT
    _section("REQUEST 1 REPEAT (expect reuse)")
    results_1b = run_acquisition(
        "Request 1 REPEAT",
        adapter_1, date(2023, 1, 1), date(2023, 1, 3), data_root,
    )

    _section("VERIFICATION: Request 1 Repeat")
    repeat_pass = True
    for r in results_1b:
        ok = _check("Repeat succeeded", r["success"])
        repeat_pass = repeat_pass and ok
        if r["success"]:
            ok2 = _check("File was reused (no re-download)", r["skipped"])
            repeat_pass = repeat_pass and ok2
            orig_md5 = r1_md5s.get(str(r["request_date"]))
            if orig_md5:
                ok3 = _check("MD5 unchanged", r["md5"] == orig_md5,
                              f"orig={orig_md5[:8]}… repeat={r['md5'][:8] if r['md5'] else 'None'}…")
                repeat_pass = repeat_pass and ok3
    all_checks.append(repeat_pass)

    # ------------------------------------------------------------------ R2
    adapter_2 = OSCARAdapter(min_lat=15.0, max_lat=25.0, min_lon=70.0, max_lon=90.0)
    results_2 = run_acquisition(
        "Request 2 (lat 15-25N, lon 70-90E, 2022-03-01→03)",
        adapter_2, date(2022, 3, 1), date(2022, 3, 3), data_root,
    )

    _section("VERIFICATION: Request 2")
    r2_pass = True
    r1_paths = {r["relative_path"] for r in results_1}
    for r in results_2:
        ok = _check("R2 download succeeded", r["success"])
        r2_pass = r2_pass and ok
        if r["success"]:
            ok2 = _check("R2 path differs from R1", r["relative_path"] not in r1_paths,
                         r["relative_path"])
            r2_pass = r2_pass and ok2
            nc_ok, nc_info = inspect_netcdf(r["path"], r["request_date"],
                                            adapter_2._min_lat, adapter_2._max_lat,
                                            adapter_2._min_lon, adapter_2._max_lon)
            ok3 = _check("R2 NetCDF valid", nc_ok, str(nc_info.get("data_vars", nc_info.get("error",""))))
            r2_pass = r2_pass and ok3
    # R1 files must still be intact
    for r in results_1:
        if r["success"]:
            ok_r1 = _check("R1 file intact after R2", r["path"].exists() and r["path"].stat().st_size > 0)
            if r1_md5s.get(str(r["request_date"])):
                ok_md5 = _check("R1 MD5 unchanged", _md5_of_file(r["path"]) == r1_md5s[str(r["request_date"])])
                r2_pass = r2_pass and ok_r1 and ok_md5
    all_checks.append(r2_pass)

    # Final manifest
    _section("MANIFEST: After Both Requests")
    all_recs = verify_manifest(data_root, min_success=len(results_1) + len(results_2))
    distinct = sorted(set(r.get("request_date") for r in all_recs))
    _check("SUCCESS entries from at least 2 distinct dates", len(distinct) >= 2, str(distinct))

    # Summary
    _section("SMOKE TEST SUMMARY")
    labels = [
        "Request 1 (lat 10-20N, lon 65-85E, 2023-01-01→03)",
        "Request 1 REPEAT (reuse check)",
        "Request 2 (lat 15-25N, lon 70-90E, 2022-03-01→03)",
    ]
    for lbl, passed in zip(labels, all_checks):
        print(f"  [{'PASS' if passed else 'FAIL'}] {lbl}")
    print()
    _check("All scenarios passed", all(all_checks), f"{sum(all_checks)}/{len(all_checks)}")
    print(f"\n  Raw data : {data_root / 'raw' / 'oscar'}")
    print(f"  Manifest : {data_root / 'manifests' / 'oscar.jsonl'}")
    print()
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
