import requests
from pathlib import Path
import gzip
import shutil

URL = "https://data-argo.ifremer.fr/ar_index_global_prof.txt.gz"

OUTPUT_DIR = Path("./data/argo/indexes")
GZ_FILE = OUTPUT_DIR / "ar_index_global_prof.txt.gz"
TXT_FILE = OUTPUT_DIR / "ar_index_global_prof.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading ARGO global profile index...")
print(f"URL: {URL}")
print(f"Destination: {GZ_FILE}")

with requests.get(URL, stream=True, timeout=120) as response:
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(GZ_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100
                    print(
                        f"\rProgress: {percent:6.2f}% "
                        f"({downloaded / 1024 / 1024:.1f} MB)",
                        end=""
                    )

print("\nDownload complete.")

print("Extracting index...")

with gzip.open(GZ_FILE, "rb") as src:
    with open(TXT_FILE, "wb") as dst:
        shutil.copyfileobj(src, dst)

print("Extraction complete.")

print(f"\nCompressed file:")
print(f"  {GZ_FILE}")

print(f"\nExtracted file:")
print(f"  {TXT_FILE}")