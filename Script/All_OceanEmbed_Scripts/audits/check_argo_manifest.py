from pathlib import Path
import pandas as pd

MANIFEST = Path(
    r"D:\OceanEmbed\argo\manifests\argo_profile_manifest_2012_2024.csv"
)

print("=" * 70)
print("ARGO MANIFEST CHECK")
print("=" * 70)

print(f"Manifest: {MANIFEST}")
print()

# Read only the file column
df = pd.read_csv(
    MANIFEST,
    usecols=["file"]
)

total_rows = len(df)
unique_files = df["file"].nunique()

duplicates = total_rows - unique_files

print(f"Total manifest rows : {total_rows:,}")
print(f"Unique file paths   : {unique_files:,}")
print(f"Duplicate paths     : {duplicates:,}")

print()

if duplicates == 0:
    print("PASS: Every profile points to a unique NetCDF file.")
else:
    print("WARNING: Duplicate file paths detected.")

    duplicate_rows = df[
        df["file"].duplicated(keep=False)
    ].sort_values("file")

    print()
    print("First duplicate examples:")
    print(duplicate_rows.head(20).to_string(index=False))

print()
print("=" * 70)