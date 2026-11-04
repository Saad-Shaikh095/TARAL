from pathlib import Path
from datetime import datetime
import csv

# ============================================================
# CONFIG
# ============================================================

INDEX_FILE = Path(
    r"D:\OceanEmbed\argo\indexes\ar_index_global_prof.txt"
)

START_DATE = datetime(2012, 1, 1)
END_DATE = datetime(2025, 1, 1)  # exclusive

OUTPUT_FILE = Path(
    r"D:\OceanEmbed\argo\manifests\argo_profile_manifest_2012_2024.csv"
)

# ============================================================
# SETUP
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

total_records = 0
selected_records = 0

min_date = None
max_date = None

print("=" * 70)
print("ARGO GLOBAL PROFILE INDEX ANALYSIS")
print("=" * 70)

print(f"Index : {INDEX_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

# ============================================================
# PROCESS INDEX
# ============================================================

with open(INDEX_FILE, "r", encoding="utf-8", errors="replace") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.reader(
        line for line in infile
        if not line.startswith("#")
    )

    writer = csv.writer(outfile)

    # Read header
    header = next(reader)

    print("Columns:")
    print(header)
    print()

    writer.writerow(header)

    for row in reader:

        if len(row) != len(header):
            continue

        total_records += 1

        # ----------------------------------------------------
        # Parse date
        # ----------------------------------------------------

        try:
            profile_date = datetime.strptime(
                row[1],
                "%Y%m%d%H%M%S"
            )
        except ValueError:
            continue

        # Track global date range
        if min_date is None or profile_date < min_date:
            min_date = profile_date

        if max_date is None or profile_date > max_date:
            max_date = profile_date

        # ----------------------------------------------------
        # Select 2012-2024
        # ----------------------------------------------------

        if START_DATE <= profile_date < END_DATE:

            writer.writerow(row)
            selected_records += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if total_records % 1_000_000 == 0:
            print(
                f"Processed: {total_records:,} | "
                f"Selected: {selected_records:,}"
            )

# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Total index records : {total_records:,}")
print(f"Selected 2012-2024  : {selected_records:,}")

if min_date:
    print(f"Oldest profile      : {min_date}")

if max_date:
    print(f"Newest profile      : {max_date}")

print()
print(f"Manifest created:")
print(OUTPUT_FILE)

print()
print("Done.")