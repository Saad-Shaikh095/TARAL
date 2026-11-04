"""
Quick backend connectivity check.
Run: python scripts/development/check_backend.py
"""
import json
import sys
import urllib.request
from pathlib import Path

ENDPOINTS = [
    ("http://localhost:8000/api/health", "Health"),
    ("http://localhost:8000/api/datasets", "Dataset List"),
    ("http://localhost:8000/api/datasets/oisst", "OISST Detail"),
    ("http://localhost:8000/api/datasets/glorys", "GLORYS Detail"),
]

results = []
all_ok = True

for url, label in ENDPOINTS:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
            status = r.status

            if label == "Health":
                detail = f"status={data.get('status')} version={data.get('version')}"
            elif label == "Dataset List":
                total = data.get("total", 0)
                ids = [d["dataset_id"] for d in data.get("datasets", [])]
                detail = f"total={total} ids={ids}"
            elif "Detail" in label:
                ds_id = label.split()[0].lower()
                detail = (
                    f"product={data.get('product_name','?')[:40]} "
                    f"adapter={data.get('adapter_implemented')}"
                )
            else:
                detail = str(data)[:80]

            results.append(f"PASS  HTTP {status}  {label}: {detail}")
    except Exception as exc:
        results.append(f"FAIL  {label}: {exc}")
        all_ok = False

out_path = Path(__file__).parent.parent.parent / "check_backend_result.txt"
out_path.write_text("\n".join(results) + f"\n\nOVERALL: {'PASS' if all_ok else 'FAIL'}\n",
                    encoding="ascii", errors="replace")

print("\n".join(results))
print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
