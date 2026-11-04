from pathlib import Path
import time

import requests
import xarray as xr


# ============================================================
# OceanEmbed - NOAA OISST v2.1 Acquisition Test
# ============================================================

BASE_URL = (
    "https://www.ncei.noaa.gov/erddap/griddap/"
    "ncdc_oisst_v2_avhrr_by_time_zlev_lat_lon.nc"
)

# ------------------------------------------------------------
# Test configuration
# ------------------------------------------------------------

START_DATE = "2021-01-01T00:00:00Z"
END_DATE = "2021-01-07T00:00:00Z"

LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0

DEPTH = 0.0
VARIABLE = "sst"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "oisst"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "oisst_test_20210101_20210107.nc"


def download_oisst():

    print("=" * 70)
    print("OceanEmbed - NOAA OISST v2.1 Acquisition Test")
    print("=" * 70)

    print(f"Dataset : NOAA OISST v2.1")
    print(f"Period  : {START_DATE} → {END_DATE}")

    print(
        f"Region  : {LAT_MIN}°N → {LAT_MAX}°N\n"
        f"          {LON_MIN}°E → {LON_MAX}°E"
    )

    print(f"Variable: {VARIABLE}")
    print()

    # --------------------------------------------------------
    # ERDDAP griddap query
    #
    # OISST dimensions:
    #
    # time
    # depth
    # latitude
    # longitude
    #
    # Each dimension uses:
    #
    # [start:stride:stop]
    # --------------------------------------------------------

    query = (
        f"{VARIABLE}"
        f"[({START_DATE}):1:({END_DATE})]"
        f"[({DEPTH}):1:({DEPTH})]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )

    url = BASE_URL + "?" + query

    print("Requesting NOAA ERDDAP subset...")
    print(url)
    print()

    start_time = time.perf_counter()

    response = requests.get(
        url,
        timeout=300,
    )

    elapsed = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Helpful error output
    # --------------------------------------------------------

    if not response.ok:

        print("NOAA request failed.")
        print()
        print(f"HTTP status: {response.status_code}")
        print(f"URL: {response.url}")
        print()

        print("Server response:")
        print(response.text[:2000])

        response.raise_for_status()

    # --------------------------------------------------------
    # Save response
    # --------------------------------------------------------

    OUTPUT_FILE.write_bytes(response.content)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print("Download successful.")
    print()
    print(f"Output file : {OUTPUT_FILE}")
    print(f"File size   : {size_mb:.2f} MB")
    print(f"Time        : {elapsed:.2f} seconds")

    if elapsed > 0:
        print(f"Speed       : {size_mb / elapsed:.2f} MB/s")

    print()

    # --------------------------------------------------------
    # Open using xarray
    # --------------------------------------------------------

    print("Opening NetCDF with xarray...")

    ds = xr.open_dataset(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    print(ds)

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print()
    print("Dimensions:")

    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    print()
    print("Coordinates:")

    for name in ds.coords:

        values = ds[name].values

        if values.size > 10:

            print(
                f"  {name}: "
                f"{values.min()} → {values.max()} "
                f"({values.size} values)"
            )

        else:

            print(f"  {name}: {values}")

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    print()
    print("Data variables:")

    for name in ds.data_vars:

        variable = ds[name]

        print(
            f"  {name}: "
            f"dims={variable.dims}, "
            f"shape={variable.shape}"
        )

    # --------------------------------------------------------
    # SST statistics
    # --------------------------------------------------------

    sst = ds["sst"]

    print()
    print("SST statistics:")

    print(
        f"  Minimum : "
        f"{float(sst.min(skipna=True).values):.2f} °C"
    )

    print(
        f"  Maximum : "
        f"{float(sst.max(skipna=True).values):.2f} °C"
    )

    print(
        f"  Mean    : "
        f"{float(sst.mean(skipna=True).values):.2f} °C"
    )

    missing = int(sst.isnull().sum().values)

    print(f"  Missing : {missing:,} values")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    assert "sst" in ds.data_vars
    assert "time" in ds.dims
    assert "depth" in ds.dims
    assert "latitude" in ds.dims
    assert "longitude" in ds.dims

    assert ds.sizes["depth"] == 1

    print("✓ SST variable found")
    print("✓ Time dimension found")
    print("✓ Depth dimension found")
    print("✓ Latitude dimension found")
    print("✓ Longitude dimension found")
    print("✓ NetCDF opened successfully")
    print("✓ OISST acquisition test PASSED")

    ds.close()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    download_oisst()