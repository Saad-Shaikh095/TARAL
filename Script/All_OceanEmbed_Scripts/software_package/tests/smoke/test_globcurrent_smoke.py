"""
GlobCurrent Real Acquisition Smoke Test.

This script performs live CMEMS acquisition of GlobCurrent data.
It is NOT part of the automated unit test suite (which runs offline with mocks).
Run it manually once CMEMS credentials are available.

Prerequisites
-------------
Set environment variables before running:
    COPERNICUSMARINE_SERVICE_USERNAME=<your-cmems-username>
    COPERNICUSMARINE_SERVICE_PASSWORD=<your-cmems-password>
    OCEANEMBED_DATA_ROOT=D:\\OceanEmbed\\oceanembed-data

Or create a .env file at D:\\OceanEmbed\\oceanembed-software\\.env:
    COPERNICUSMARINE_SERVICE_USERNAME=your-username
    COPERNICUSMARINE_SERVICE_PASSWORD=your-password
    OCEANEMBED_DATA_ROOT=D:\\OceanEmbed\\oceanembed-data

Usage
-----
    # From oceanembed-software directory:
    .\.venv\Scripts\python.exe tests/smoke/test_globcurrent_smoke.py

What this test proves
---------------------
1. Request 1 (small region, one week in 2023):
       lat 10–20°N, lon 65–85°E
       2023-01-01 → 2023-01-07 (partial year → Jan-only file)
   → Download, verify file exists, verify manifest entry, verify checksum.

2. Request 1 REPEAT (same parameters):
   → Verify existing file is reused (no re-download).
   → Verify checksum matches first acquisition.

3. Request 2 (different bbox, different year — Jan 2022):
       lat 15–25°N, lon 70–90°E
       2022-03-01 → 2022-03-31
   → Verify a DIFFERENT file is created (no overwrite of request 1 output).
   → Verify manifest has two separate entries.

Scientific guardrail
--------------------
This test only verifies acquisition reliability. It does NOT:
- Regrid, interpolate, or harmonize the data.
- Validate the scientific content of the NetCDF files.
- Make any claim about what depth=0m means physically.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup: load .env FIRST — before any import that triggers copernicusmarine,
# which reads COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD at import time.
# ---------------------------------------------------------------------------

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets env vars from KEY=VALUE lines."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent  # tests/smoke/ → repo root
_DOTENV = _REPO_ROOT / ".env"
_load_dotenv(_DOTENV)

# Add src/ to path so we can import oceanembed without pip install
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
from oceanembed.acquisition.engine import (
    AcquisitionJob, JobStatus, run_cmems_subset, _md5_of_file
)
from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024**2):.2f} MB"


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return condition


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

def check_credentials() -> bool:
    _section("CREDENTIAL CHECK")
    result = GlobCurrentAdapter().check_credentials()
    print(f"  {result['message']}")
    if not result["credentials_available"]:
        print()
        print("  SMOKE TEST CANNOT RUN: CMEMS credentials not found.")
        print()
        print("  Set environment variables:")
        print("    COPERNICUSMARINE_SERVICE_USERNAME=<your-cmems-username>")
        print("    COPERNICUSMARINE_SERVICE_PASSWORD=<your-cmems-password>")
        print()
        print("  Or create .env at: D:\\OceanEmbed\\oceanembed-software\\.env")
        print("  (See .env.example for the format)")
        return False
    return True


# ---------------------------------------------------------------------------
# Data root resolution
# ---------------------------------------------------------------------------

def resolve_data_root() -> Path:
    raw_root = os.environ.get("OCEANEMBED_DATA_ROOT", "").strip()
    if raw_root:
        data_root = Path(raw_root)
    else:
        # Fall back to sibling of repo root
        data_root = _REPO_ROOT.parent / "oceanembed-data"
    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


# ---------------------------------------------------------------------------
# Single acquisition helper
# ---------------------------------------------------------------------------

def run_single_acquisition(
    label: str,
    adapter: GlobCurrentAdapter,
    start_date: date,
    end_date: date,
    data_root: Path,
) -> dict:
    """
    Run one acquisition request and return result dict.
    Does NOT raise — captures failures and returns them.
    """
    _section(f"ACQUISITION: {label}")
    print(f"  Adapter bbox : lat {adapter._min_lat}–{adapter._max_lat}°N, "
          f"lon {adapter._min_lon}–{adapter._max_lon}°E")
    print(f"  Date range   : {start_date} → {end_date}")
    print(f"  Data root    : {data_root}")
    print()

    requests = adapter.date_range_requests(start_date, end_date, str(data_root))
    print(f"  Requests generated: {len(requests)}")
    for i, req in enumerate(requests, 1):
        print(f"    {i}. {req.expected_output_path}")
        print(f"       period: {req.extra.get('period_start')} → {req.extra.get('period_end')}")

    results = []
    manifest = ManifestStore(data_root)

    for req in requests:
        dest = data_root / req.expected_output_path

        job = AcquisitionJob(
            job_id=f"smoke_{int(time.time())}",
            dataset_id="globcurrent",
            start_date=start_date,
            end_date=end_date,
            data_root=data_root,
        )

        print()
        print(f"  Acquiring: {dest.name}")
        t0 = time.perf_counter()
        was_present_before = dest.exists() and dest.stat().st_size > 0

        success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        elapsed = time.perf_counter() - t0

        status = ManifestStatus.SUCCESS if success else ManifestStatus.FAILED
        entry = ManifestEntry(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            source_url=req.url_or_query,
            local_path=str(dest),
            product_id=req.source_description.product_id,
            file_size_bytes=nbytes if success else None,
            md5_checksum=md5,
            status=status,
            error=None if success else "Download failed — see logs",
        )
        manifest.record(entry)

        # Determine if this was a skip or a fresh download
        skipped = was_present_before and success

        result = {
            "label": label,
            "path": dest,
            "relative_path": req.expected_output_path,
            "success": success,
            "nbytes": nbytes,
            "md5": md5,
            "elapsed": elapsed,
            "skipped": skipped,
            "job_logs": job.logs,
        }
        results.append(result)

        print(f"  Success    : {success}")
        if success:
            print(f"  Reused     : {skipped}")
            print(f"  File size  : {_fmt_bytes(nbytes)}")
            print(f"  MD5        : {md5}")
            print(f"  Elapsed    : {elapsed:.1f}s")
        else:
            # Print job logs for diagnosis
            print(f"  FAILURE — job logs:")
            for log in job.logs[-10:]:
                print(f"    [{log.get('level','?')}] {log.get('message','')}")

    return results


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_file_integrity(path: Path, expected_md5: str | None) -> bool:
    """Check file exists, is non-empty, and MD5 matches if provided."""
    if not path.exists():
        print(f"  [FAIL] File does not exist: {path}")
        return False
    size = path.stat().st_size
    if size == 0:
        print(f"  [FAIL] File is empty: {path}")
        return False
    if expected_md5:
        actual_md5 = _md5_of_file(path)
        if actual_md5 != expected_md5:
            print(f"  [FAIL] MD5 mismatch: expected {expected_md5}, got {actual_md5}")
            return False
    return True


def verify_manifest(data_root: Path, dataset_id: str, min_entries: int) -> list[dict]:
    """Read manifest and return entries."""
    manifest = ManifestStore(data_root)
    records = manifest.read_all(dataset_id)
    success_records = [r for r in records if r.get("status") == "SUCCESS"]
    _check(
        f"Manifest has at least {min_entries} SUCCESS entry/entries for {dataset_id}",
        len(success_records) >= min_entries,
        f"found {len(success_records)} SUCCESS (total entries including prior failures: {len(records)})",
    )
    for rec in success_records:
        _check(
            f"Manifest SUCCESS entry has md5_checksum",
            rec.get("md5_checksum") is not None,
            f"entry date={rec.get('request_date')}",
        )
        _check(
            f"Manifest SUCCESS entry has status=SUCCESS",
            rec.get("status") == "SUCCESS",
            f"status={rec.get('status')}",
        )
    return records


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 70)
    print("GLOBCURRENT SMOKE TEST")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    if not check_credentials():
        return 1

    data_root = resolve_data_root()
    print()
    print(f"  Data root    : {data_root}")
    print(f"  Manifests at : {data_root / 'manifests'}")

    all_checks: list[bool] = []

    # -----------------------------------------------------------------------
    # REQUEST 1: small region, partial year (Jan 2023 only)
    # Small enough to download quickly but real enough to verify.
    # -----------------------------------------------------------------------
    adapter_1 = GlobCurrentAdapter(
        min_lat=10.0, max_lat=20.0, min_lon=65.0, max_lon=85.0
    )
    start_1 = date(2023, 1, 1)
    end_1 = date(2023, 1, 7)  # one week — partial year file

    results_1 = run_single_acquisition(
        label="Request 1 (small bbox, Jan 2023 week)",
        adapter=adapter_1,
        start_date=start_1,
        end_date=end_1,
        data_root=data_root,
    )

    _section("VERIFICATION: Request 1")
    all_pass = True
    for r in results_1:
        ok = _check("Download succeeded", r["success"])
        all_pass = all_pass and ok
        if r["success"]:
            ok2 = _check(
                "File exists and non-empty",
                verify_file_integrity(r["path"], r["md5"]),
                str(r["path"].name),
            )
            all_pass = all_pass and ok2
            _check("Not a reuse (first acquisition)", not r["skipped"])
    all_checks.append(all_pass)

    if not all_pass:
        print()
        print("  Request 1 FAILED — stopping smoke test.")
        print("  Check CMEMS credentials and dataset availability.")
        return 1

    # Manifest check after request 1
    _section("MANIFEST VERIFICATION: After Request 1")
    records_after_1 = verify_manifest(data_root, "globcurrent", min_entries=1)
    r1_md5 = results_1[0]["md5"] if results_1 else None

    # -----------------------------------------------------------------------
    # REQUEST 1 REPEAT: same parameters → should reuse, no re-download
    # -----------------------------------------------------------------------
    _section("REQUEST 1 REPEAT (same params — expect reuse)")
    results_1b = run_single_acquisition(
        label="Request 1 REPEAT",
        adapter=adapter_1,
        start_date=start_1,
        end_date=end_1,
        data_root=data_root,
    )

    _section("VERIFICATION: Request 1 Repeat")
    reuse_pass = True
    for r in results_1b:
        ok = _check("Repeat succeeded (file reused)", r["success"])
        reuse_pass = reuse_pass and ok
        if r["success"]:
            ok2 = _check("File was reused (not re-downloaded)", r["skipped"])
            reuse_pass = reuse_pass and ok2
            if r1_md5:
                ok3 = _check(
                    "MD5 unchanged after reuse",
                    r["md5"] == r1_md5,
                    f"first={r1_md5[:8]}… repeat={r['md5'][:8] if r['md5'] else 'None'}…",
                )
                reuse_pass = reuse_pass and ok3
    all_checks.append(reuse_pass)

    # -----------------------------------------------------------------------
    # REQUEST 2: different bbox, different year → must NOT overwrite request 1
    # -----------------------------------------------------------------------
    adapter_2 = GlobCurrentAdapter(
        min_lat=15.0, max_lat=25.0, min_lon=70.0, max_lon=90.0
    )
    start_2 = date(2022, 3, 1)
    end_2 = date(2022, 3, 7)  # one week in 2022

    results_2 = run_single_acquisition(
        label="Request 2 (different bbox, Mar 2022 week)",
        adapter=adapter_2,
        start_date=start_2,
        end_date=end_2,
        data_root=data_root,
    )

    _section("VERIFICATION: Request 2")
    r2_pass = True
    for r in results_2:
        ok = _check("Request 2 download succeeded", r["success"])
        r2_pass = r2_pass and ok
        if r["success"]:
            ok2 = _check(
                "Request 2 path differs from Request 1",
                r["relative_path"] != results_1[0]["relative_path"] if results_1 else True,
                f"R1={results_1[0]['relative_path'] if results_1 else 'N/A'} "
                f"R2={r['relative_path']}",
            )
            r2_pass = r2_pass and ok2
            # Request 1 file must still exist and be unchanged
            if results_1:
                ok3 = _check(
                    "Request 1 file still exists after Request 2",
                    results_1[0]["path"].exists() and results_1[0]["path"].stat().st_size > 0,
                )
                if r1_md5:
                    ok4 = _check(
                        "Request 1 MD5 unchanged after Request 2",
                        _md5_of_file(results_1[0]["path"]) == r1_md5,
                    )
                    r2_pass = r2_pass and ok4
                r2_pass = r2_pass and ok3
    all_checks.append(r2_pass)

    # -----------------------------------------------------------------------
    # Final manifest: should have entries for both requests
    # -----------------------------------------------------------------------
    _section("MANIFEST VERIFICATION: After Both Requests")
    all_records = verify_manifest(data_root, "globcurrent", min_entries=2)
    success_records = [r for r in all_records if r.get("status") == "SUCCESS"]
    _check(
        "Manifest has SUCCESS entries from different dates",
        len(set(r.get("request_date") for r in success_records)) >= 2,
        f"distinct dates: {sorted(set(r.get('request_date') for r in success_records))}",
    )

    # -----------------------------------------------------------------------
    # Output summary
    # -----------------------------------------------------------------------
    _section("SMOKE TEST SUMMARY")

    labels = [
        "Request 1 (small bbox, Jan 2023 week)",
        "Request 1 REPEAT (reuse check)",
        "Request 2 (different bbox + year)",
    ]
    for label, passed in zip(labels, all_checks):
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")

    print()
    _check(
        "All smoke test scenarios passed",
        all(all_checks),
        f"{sum(all_checks)}/{len(all_checks)} passed",
    )

    print()
    print(f"  Raw data location: {data_root / 'raw' / 'globcurrent'}")
    print(f"  Manifest location: {data_root / 'manifests' / 'globcurrent.jsonl'}")
    print()

    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
