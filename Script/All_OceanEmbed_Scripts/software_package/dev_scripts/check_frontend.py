"""Quick frontend connectivity check. Run: python scripts/development/check_frontend.py"""
import urllib.request
from pathlib import Path

results = []

try:
    with urllib.request.urlopen("http://localhost:3000", timeout=10) as r:
        html = r.read().decode("utf-8", errors="replace")
        results.append(f"HTTP status: {r.status}")
        results.append(f"OceanEmbed in page: {'OceanEmbed' in html}")
        results.append(f"Next.js present: {'__NEXT_DATA__' in html or '_next' in html}")
        results.append(f"Page length: {len(html)} chars")
except Exception as exc:
    results.append(f"ERROR: {exc}")

try:
    with urllib.request.urlopen("http://localhost:3000/api/health", timeout=10) as r:
        import json
        d = json.loads(r.read())
        results.append(f"Proxy /api/health: HTTP {r.status} -> {d.get('status')}")
except Exception as exc:
    results.append(f"Proxy ERROR: {exc}")

try:
    with urllib.request.urlopen("http://localhost:3000/api/datasets", timeout=10) as r:
        import json
        d = json.loads(r.read())
        results.append(f"Proxy /api/datasets: HTTP {r.status} -> total={d.get('total')}")
except Exception as exc:
    results.append(f"Proxy datasets ERROR: {exc}")

out = Path(__file__).parent.parent.parent / "frontend_check_result.txt"
out.write_text("\n".join(results), encoding="ascii", errors="replace")
print("\n".join(results))
