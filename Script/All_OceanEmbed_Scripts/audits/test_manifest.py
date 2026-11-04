from pathlib import Path
import pandas as pd

INPUT = Path(
    r"D:\OceanEmbed\argo\manifests\argo_profile_manifest_2012_2024.csv"
)

OUTPUT = Path(
    r"D:\OceanEmbed\argo\manifests\argo_profile_manifest_2012_01.csv"
)

print("=" * 70)
print("CREATING ARGO JANUARY 2012 TEST MANIFEST")
print("=" * 70)

df = pd.read_csv(
    INPUT,
    dtype={
        "file": "string",
        "date": "string",
    }
)

# Convert date column
dates = pd.to_datetime(
    df["date"],
    format="%Y%m%d%H%M%S",
    errors="coerce"
)

mask = (
    (dates >= "2012-01-01") &
    (dates < "2012-02-01")
)

test_df = df.loc[mask].copy()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

test_df.to_csv(
    OUTPUT,
    index=False
)

print()
print(f"Input profiles  : {len(df):,}")
print(f"January profiles : {len(test_df):,}")
print()
print(f"Output:")
print(OUTPUT)

print()
print("=" * 70)