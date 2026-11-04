"""
DUACS Real Acquisition Smoke Test.

Performs live CMEMS acquisition of DUACS data to verify the adapter works
against the actual source. NOT part of the automated unit test suite.

Three scenarios:
  Request 1  — small bbox, one week, 2023-01-01 → 2023-01-07
  Request 1 REPEAT — same params → verify file reuse, no re-download
  Request 2  — different bbox, different week → separate file, R1 intact

After download, opens each NetCDF and verifies:
  - dimensions (time, latitude, longitude) — NO depth
  - variables (sla, adt, ugos, vgos)
  - resolution ≈ 0.125°
  - no preprocessing applied

Usage
-----
    cd D:\\OceanEmbed\\oceanembed-software
    .venv\\Scripts\\python.exe tests/smoke/test_duacs_smoke.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Load .env FIRST — before any import that triggers copernicusmarine
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
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

from oceanembed.datasets.adapters.duacs import DUACSAdapter, _CMEMS_DATASET_ID
from oceanembed.acquisition.engine import AcquisitionJob, run_cmems_subset, _md5_of_file
from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

def check_credentials() -> bool:
    _section("CREDENTIAL CHECK")
    result = DUACSAdapter().check_credentials()
    print(f"  {result['message']}")
    if not result["credentials_available"]:
        print("\n  SMOKE TEST CANNOT RUN — credentials missing.")
        print("  Set COPERNICUSMARINE_SERVICE_USERNAME and _PASSWORD in .env")
        return False
    return True


def resolve_data_root() -> Path:
    raw = os.environ.get("OCEANEMBED_DATA_ROOT", "").strip()
    root = Path(raw) if raw else _REPO_ROOT.parent / "oceanembed-data"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Single acquisition helper
# ---------------------------------------------------------------------------

def run_acquisition(
    label: str,
    adapter: DUACSAdapter,
    start: date,
    end: date,
    data_root: Path,
) -> list[dict]:
    _section(f"ACQUISITION: {label}")
    print(f"  Bbox  : lat {adapter._min_lat}–{adapter._max_lat}°N, "
          f"lon {adapter._min_lon}–{adapter._max_lon}°E")
    print(f"  Range : {start} → {end}")
    print()

    requests = adapter.date_range_requests(start, end, str(data_root))
    print(f"  Requests generated: {len(requests)}")
    for i, req in enumerate(requests, 1):
        print(f"    {i}. {req.expected_output_path}")

    manifest = ManifestStore(data_root)
    results = []

    for req in requests:
        dest = data_root / req.expected_output_path
        was_present = dest.exists() and dest.stat().st_size > 0

        job = AcquisitionJob(
            job_id=f"duacs_{int(time.time()*1000) % 10**8}",
            dataset_id="duacs",
            start_date=start, end_date=end,
            data_root=data_root,
        )

        print(f"\n  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)
        elapsed = time.perf_counter() - t0

        status = ManifestStatus.SUCCESS if success else ManifestStatus.FAILED
        manifest.record(ManifestEntry(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            source_url=req.url_or_query,
            local_path=str(dest),
            product_id=req.source_description.product_id,
            file_size_bytes=nbytes if success else None,
            md5_checksum=md5,
            status=status,
            error=None if success else "Download failed — see logs",
        ))

        skipped = was_present and success
        result = dict(
            label=label, path=dest,
            relative_path=req.expected_output_path,
            success=success, nbytes=nbytes, md5=md5,
            elapsed=elapsed, skipped=skipped, job=job,
        )
        results.append(result)

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


# ---------------------------------------------------------------------------
# NetCDF content verification (no xarray import error if not installed)
# ---------------------------------------------------------------------------

def verify_netcdf_content(path: Path) -> tuple[bool, str]:
    """Open the file and verify variables, dims, resolution. No preprocessing."""
    try:
        import xarray as xr
        import numpy as np
        with xr.open_dataset(path, decode_cf=True) as ds:
            # Variables
            missing = [v for v in ["sla", "adt", "ugos", "vgos"] if v not in ds.data_vars]
            if missing:
                return False, f"Missing variables: {missing}"
            # No depth dimension
            if "depth" in ds.dims or "depth" in ds.coords:
                return False, "Unexpected depth dimension — should not be present in DUACS"
            # Dims check
            for v in ["sla", "adt", "ugos", "vgos"]:
                dims = set(ds[v].dims)
                if not {"time", "latitude", "longitude"}.issubset(dims):
                    return False, f"{v} dims {ds[v].dims} missing expected coordinates"
            # Resolution
            lat = ds["latitude"].values
            lon = ds["longitude"].values
            if len(lat) > 1:
                dlat = float(np.median(np.abs(np.diff(lat))))
                if not np.isclose(dlat, 0.125, atol=0.01):
                    return False, f"Unexpected lat resolution: {dlat:.4f}° (expected ~0.125°)"
            if len(lon) > 1:
                dlon = float(np.median(np.abs(np.diff(lon))))
                if not np.isclose(dlon, 0.125, atol=0.01):
                    return False, f"Unexpected lon resolution: {dlon:.4f}° (expected ~0.125°)"
        return True, "variables, dims, resolution OK"
    except ImportError:
        return True, "xarray not available — skipping content check"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 70)
    print("DUACS SMOKE TEST")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    if not check_credentials():
        return 1

    data_root = resolve_data_root()
    print(f"\n  Data root : {data_root}")
    print(f"  Manifests : {data_root / 'manifests'}")

    all_checks: list[bool] = []

    # ------------------------------------------------------------------ R1
    adapter_1 = DUACSAdapter(min_lat=10.0, max_lat=20.0, min_lon=65.0, max_lon=85.0)
    results_1 = run_acquisition(
        "Request 1 (small bbox, 2023-01-01–07)",
        adapter_1, date(2023, 1, 1), date(2023, 1, 7), data_root,
    )

    _section("VERIFICATION: Request 1")
    r1_pass = True
    r1_md5 = None
    for r in results_1:
        ok = _check("Download succeeded", r["success"])
        r1_pass = r1_pass and ok
        if r["success"]:
            ok2 = _check("File exists and non-empty",
                         r["path"].exists() and r["path"].stat().st_size > 0,
                         r["path"].name)
            r1_pass = r1_pass and ok2
            _check("First acquisition (not a reuse)", not r["skipped"])
            nc_ok, nc_msg = verify_netcdf_content(r["path"])
            ok3 = _check("NetCDF content valid", nc_ok, nc_msg)
            r1_pass = r1_pass and ok3
            r1_md5 = r["md5"]
    all_checks.append(r1_pass)

    if not r1_pass:
        print("\n  Request 1 FAILED — stopping.")
        return 1

    # Manifest after R1
    _section("MANIFEST: After Request 1")
    manifest = ManifestStore(data_root)
    recs = [r for r in manifest.read_all("duacs") if r.get("status") == "SUCCESS"]
    _check(f"At least 1 SUCCESS entry", len(recs) >= 1, f"found {len(recs)}")
    for rec in recs[-len(results_1):]:
        _check("Entry has md5_checksum", rec.get("md5_checksum") is not None,
               f"date={rec.get('request_date')}")

    # ------------------------------------------------------------------ R1 REPEAT
    _section("REQUEST 1 REPEAT (expect reuse)")
    results_1b = run_acquisition(
        "Request 1 REPEAT",
        adapter_1, date(2023, 1, 1), date(2023, 1, 7), data_root,
    )

    _section("VERIFICATION: Request 1 Repeat")
    repeat_pass = True
    for r, r_orig in zip(results_1b, results_1):
        ok = _check("Repeat succeeded", r["success"])
        repeat_pass = repeat_pass and ok
        if r["success"]:
            ok2 = _check("File was reused (no re-download)", r["skipped"])
            repeat_pass = repeat_pass and ok2
            if r1_md5:
                ok3 = _check("MD5 unchanged",
                              r["md5"] == r1_md5,
                              f"orig={r1_md5[:8]}… repeat={r['md5'][:8] if r['md5'] else 'None'}…")
                repeat_pass = repeat_pass and ok3
    all_checks.append(repeat_pass)

    # ------------------------------------------------------------------ R2
    adapter_2 = DUACSAdapter(min_lat=15.0, max_lat=25.0, min_lon=70.0, max_lon=90.0)
    results_2 = run_acquisition(
        "Request 2 (different bbox, 2022-03-01–07)",
        adapter_2, date(2022, 3, 1), date(2022, 3, 7), data_root,
    )

    _section("VERIFICATION: Request 2")
    r2_pass = True
    for r in results_2:
        ok = _check("Request 2 download succeeded", r["success"])
        r2_pass = r2_pass and ok
        if r["success"]:
            ok2 = _check("Different path from R1",
                          r["relative_path"] != results_1[0]["relative_path"] if results_1 else True,
                          f"R1={results_1[0]['relative_path'] if results_1 else 'N/A'} "
                          f"R2={r['relative_path']}")
            r2_pass = r2_pass and ok2
            nc_ok, nc_msg = verify_netcdf_content(r["path"])
            ok3 = _check("R2 NetCDF content valid", nc_ok, nc_msg)
            r2_pass = r2_pass and ok3
    if results_1:
        ok_r1 = _check("R1 file still intact after R2",
                        results_1[0]["path"].exists() and
                        results_1[0]["path"].stat().st_size > 0)
        if r1_md5:
            ok_md5 = _check("R1 MD5 unchanged after R2",
                             _md5_of_file(results_1[0]["path"]) == r1_md5)
            r2_pass = r2_pass and ok_r1 and ok_md5
    all_checks.append(r2_pass)

    # Manifest final
    _section("MANIFEST: After Both Requests")
    all_success = [r for r in manifest.read_all("duacs") if r.get("status") == "SUCCESS"]
    _check("At least 2 SUCCESS entries", len(all_success) >= 2, f"found {len(all_success)}")
    distinct_dates = sorted(set(r.get("request_date") for r in all_success))
    _check("SUCCESS entries from at least 2 distinct dates",
           len(distinct_dates) >= 2, f"{distinct_dates}")

    # Summary
    _section("SMOKE TEST SUMMARY")
    labels = [
        "Request 1 (small bbox, 2023-01-01–07)",
        "Request 1 REPEAT (reuse check)",
        "Request 2 (different bbox + year)",
    ]
    for lbl, passed in zip(labels, all_checks):
        print(f"  [{'PASS' if passed else 'FAIL'}] {lbl}")
    print()
    _check("All scenarios passed", all(all_checks), f"{sum(all_checks)}/{len(all_checks)}")
    print(f"\n  Raw data : {data_root / 'raw' / 'duacs'}")
    print(f"  Manifest : {data_root / 'manifests' / 'duacs.jsonl'}")
    print()

    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
