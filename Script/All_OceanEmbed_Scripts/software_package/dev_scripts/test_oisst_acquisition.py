"""
Real OISST acquisition integration test (2023-01-01 to 2023-01-07, 7 files).

Usage:
    python scripts/development/test_oisst_acquisition.py

Requires backend running on localhost:8000.
"""
import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

BASE = "http://localhost:8000"
DATA_ROOT = pathlib.Path(
    os.environ.get("OCEANEMBED_DATA_ROOT", r"D:\OceanEmbed\oceanembed-data")
).expanduser()
TEST_START = "2023-01-01"
TEST_END   = "2023-01-07"


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    results = []
    overall_ok = True

    # 1. Health
    try:
        h = get(f"{BASE}/api/health")
        results.append(
            f"PASS  /api/health  status={h['status']} version={h['version']}"
        )
    except Exception as e:
        results.append(f"FAIL  /api/health  {e}")
        overall_ok = False

    # 2. Confirm data_root has no trailing whitespace
    data_root_str = str(DATA_ROOT)
    if data_root_str != data_root_str.strip():
        results.append(f"FAIL  DATA_ROOT has surrounding whitespace: {data_root_str!r}")
        overall_ok = False
    else:
        results.append(f"PASS  DATA_ROOT is clean: {data_root_str}")

    # 3. Create job
    code, job = post(
        f"{BASE}/api/acquisition/jobs",
        {"dataset_id": "oisst", "start_date": TEST_START, "end_date": TEST_END},
    )
    if code not in (200, 202):
        results.append(f"FAIL  POST /api/acquisition/jobs  HTTP {code}  {job}")
        overall_ok = False
        _write(results)
        return 1

    job_id = job["job_id"]
    results.append(f"PASS  Job created  id={job_id}  status={job['status']}")

    # 4. Poll until terminal
    deadline = time.time() + 420
    terminal = {"COMPLETED", "FAILED", "CANCELLED"}
    last_status = job["status"]
    print(f"  Polling job {job_id} (up to 420 s for 7 files)...")

    while time.time() < deadline:
        time.sleep(4)
        try:
            j = get(f"{BASE}/api/acquisition/jobs/{job_id}")
        except Exception as e:
            results.append(f"WARN  Poll error: {e}")
            continue
        if j["status"] != last_status:
            print(f"  {last_status} -> {j['status']}  "
                  f"({j['files_completed']}/{j['files_requested']} files)")
            last_status = j["status"]
        if j["status"] in terminal:
            break

    final = get(f"{BASE}/api/acquisition/jobs/{job_id}")

    # 5. Collect logs
    logs_raw = get(f"{BASE}/api/acquisition/jobs/{job_id}/logs")
    log_lines = []
    for entry in logs_raw.get("logs", []):
        ts = (entry.get("ts") or "")[:19].replace("T", " ")
        log_lines.append(f"  {ts}  [{entry['level']:8}]  {entry['message']}")

    # 6. Evaluate outcome
    if final["status"] == "COMPLETED":
        results.append(
            f"PASS  Job COMPLETED  files={final['files_completed']}/"
            f"{final['files_requested']}  bytes={final['bytes_downloaded']:,}  "
            f"elapsed={final.get('elapsed_seconds', 0):.1f}s"
        )

        # Verify each expected file
        raw_dir = DATA_ROOT / "raw" / "oisst" / "2023"
        results.append(f"PASS  Output directory: {raw_dir}")
        start_d = date(2023, 1, 1)
        for i in range(7):
            d = start_d + timedelta(days=i)
            fpath = raw_dir / f"oisst_{d.strftime('%Y%m%d')}.nc"
            if fpath.exists() and fpath.stat().st_size > 0:
                sz = fpath.stat().st_size
                results.append(f"PASS  {fpath.name}  ({sz:,} bytes)")
            else:
                results.append(f"FAIL  {fpath.name}  NOT FOUND or empty")
                overall_ok = False

        # Verify no whitespace in paths
        for path_str in [str(raw_dir), str(DATA_ROOT / "raw")]:
            if " \\" in path_str or " /" in path_str:
                results.append(f"FAIL  Whitespace in path: {path_str!r}")
                overall_ok = False

        # Manifest
        mpath = DATA_ROOT / "manifests" / "oisst.jsonl"
        if mpath.exists():
            recs = [l for l in mpath.read_text(encoding="utf-8").splitlines() if l.strip()]
            # Check last 7 for checksums
            checksums_ok = 0
            for line in recs[-7:]:
                try:
                    rec = json.loads(line)
                    if rec.get("md5_checksum"):
                        checksums_ok += 1
                except Exception:
                    pass
            results.append(
                f"PASS  Manifest: {mpath}  ({len(recs)} records, "
                f"{checksums_ok}/7 recent records have MD5)"
            )
        else:
            results.append(f"WARN  Manifest not found at {mpath}")

    elif final["status"] == "FAILED":
        results.append(
            f"FAIL  Job FAILED  completed={final['files_completed']}  "
            f"failed={final['files_failed']}  error={final.get('error')}"
        )
        overall_ok = False
    else:
        results.append(f"FAIL  Unexpected final status: {final['status']}")
        overall_ok = False

    _write(results, log_lines, final)
    return 0 if overall_ok else 1


def _write(results, log_lines=None, final=None):
    out = pathlib.Path(__file__).parent.parent.parent / "oisst_acq_result.txt"
    lines = ["=== OISST Acquisition Test (2023-01-01 to 2023-01-07) ===", ""]
    lines += results
    if log_lines:
        lines += ["", "--- Log ---"]
        lines += log_lines
    if final:
        lines += ["", "--- Final state ---", json.dumps(final, indent=2, default=str)]
    overall = "PASS" if all("FAIL" not in r for r in results if not r.startswith("WARN")) else "FAIL"
    lines += ["", f"OVERALL: {overall}"]
    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
