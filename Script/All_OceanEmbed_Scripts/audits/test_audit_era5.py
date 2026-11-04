from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# OceanEmbed - ERA5 Deep Audit
# ============================================================

FILE = Path(
    r"D:\OceanEmbed\data\raw\era5\era5_arabian_sea_hourly_2012_01.nc"
)

# OceanEmbed Arabian Sea acquisition region
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 81.0


def section(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def audit():

    section("OceanEmbed - ERA5 DEEP AUDIT")

    print(f"File: {FILE}")

    if not FILE.exists():
        raise FileNotFoundError(f"File not found: {FILE}")

    size_gb = FILE.stat().st_size / (1024 ** 3)

    print(f"File size: {size_gb:.3f} GiB")

    # --------------------------------------------------------
    # 1. Open lazily
    # --------------------------------------------------------

    ds = xr.open_dataset(FILE)

    section("1. DATASET SUMMARY")

    print(ds)

    # --------------------------------------------------------
    # 2. Dimensions
    # --------------------------------------------------------

    section("2. DIMENSIONS")

    for name, size in ds.sizes.items():
        print(f"{name:15s}: {size}")

    # --------------------------------------------------------
    # 3. Coordinates
    # --------------------------------------------------------

    section("3. COORDINATE AUDIT")

    for name in ds.coords:

        coord = ds[name]

        print()
        print(f"{name}")
        print(f"  dtype : {coord.dtype}")
        print(f"  shape : {coord.shape}")

        if coord.size > 0:

            try:
                print(f"  first : {coord.values[0]}")
                print(f"  last  : {coord.values[-1]}")
            except Exception:
                pass

    # --------------------------------------------------------
    # 4. Variables
    # --------------------------------------------------------

    section("4. DATA VARIABLES")

    for name, var in ds.data_vars.items():

        print()
        print(f"Variable: {name}")
        print(f"  dimensions : {var.dims}")
        print(f"  shape      : {var.shape}")
        print(f"  dtype      : {var.dtype}")

        print("  attributes:")

        for key, value in var.attrs.items():
            print(f"    {key}: {value}")

    # --------------------------------------------------------
    # 5. Global metadata
    # --------------------------------------------------------

    section("5. GLOBAL METADATA")

    for key, value in ds.attrs.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # 6. Detect wind variables
    # --------------------------------------------------------

    section("6. OCEANEMBED WIND VARIABLE CHECK")

    variables = list(ds.data_vars)

    print("Variables found:")

    for name in variables:
        print(f"  - {name}")

    wind_candidates = []

    for name in variables:

        lower = name.lower()

        if (
            "10m_u_component_of_wind" in lower
            or "10u" in lower
            or lower in ["u10", "u10m"]
            or lower == "u"
        ):
            wind_candidates.append(name)

        if (
            "10m_v_component_of_wind" in lower
            or "10v" in lower
            or lower in ["v10", "v10m"]
            or lower == "v"
        ):
            wind_candidates.append(name)

    print()

    if wind_candidates:

        print("Potential OceanEmbed wind variables:")

        for name in wind_candidates:
            print(f"  ✓ {name}")

    else:

        print("⚠ Could not automatically identify U10/V10.")
        print("Inspect the variable list above.")

    # --------------------------------------------------------
    # 7. Time audit
    # --------------------------------------------------------

    section("7. TIME AUDIT")

    time_name = None

    for candidate in ["time", "valid_time", "date"]:

        if candidate in ds.coords:
            time_name = candidate
            break

        if candidate in ds:
            time_name = candidate
            break

    if time_name is None:

        print("⚠ No obvious time coordinate found.")

    else:

        time = ds[time_name]

        print(f"Time coordinate: {time_name}")
        print(f"Number of timestamps: {time.size}")

        values = time.values

        print(f"First timestamp: {values[0]}")
        print(f"Last timestamp : {values[-1]}")

        if len(values) > 1:

            differences = np.diff(values)

            print()
            print("Unique time intervals:")

            for diff in np.unique(differences):
                print(f"  {diff}")

            # ERA5 should normally be hourly here.
            expected = np.timedelta64(1, "h")

            if np.all(differences == expected):

                print("✓ Exactly hourly")

            else:

                print("⚠ Not perfectly hourly")

        if len(np.unique(values)) == len(values):

            print("✓ No duplicate timestamps")

        else:

            print("⚠ Duplicate timestamps detected")

    # --------------------------------------------------------
    # 8. Spatial audit
    # --------------------------------------------------------

    section("8. SPATIAL AUDIT")

    lat_name = None
    lon_name = None

    for candidate in ["latitude", "lat"]:

        if candidate in ds.coords:
            lat_name = candidate
            break

    for candidate in ["longitude", "lon"]:

        if candidate in ds.coords:
            lon_name = candidate
            break

    print(f"Latitude coordinate : {lat_name}")
    print(f"Longitude coordinate: {lon_name}")

    if lat_name and lon_name:

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        print()
        print("Latitude:")
        print(f"  count: {len(lat)}")
        print(f"  min  : {lat.min()}")
        print(f"  max  : {lat.max()}")

        print()
        print("Longitude:")
        print(f"  count: {len(lon)}")
        print(f"  min  : {lon.min()}")
        print(f"  max  : {lon.max()}")

        if len(lat) > 1:

            lat_diff = np.diff(lat)

            print(
                f"  spacing: "
                f"{lat_diff.min()} → {lat_diff.max()}"
            )

        if len(lon) > 1:

            lon_diff = np.diff(lon)

            print(
                f"  spacing: "
                f"{lon_diff.min()} → {lon_diff.max()}"
            )

        # ----------------------------------------------------
        # Check orientation
        # ----------------------------------------------------

        print()

        if np.all(np.diff(lat) < 0):

            print("ℹ Latitude is descending")

        elif np.all(np.diff(lat) > 0):

            print("✓ Latitude is ascending")

        else:

            print("⚠ Latitude ordering is irregular")

        if np.all(np.diff(lon) > 0):

            print("✓ Longitude is ascending")

        elif np.all(np.diff(lon) < 0):

            print("ℹ Longitude is descending")

        else:

            print("⚠ Longitude ordering is irregular")

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        covers_lat = (
            lat.min() <= LAT_MIN
            and lat.max() >= LAT_MAX
        )

        covers_lon = (
            lon.min() <= LON_MIN
            and lon.max() >= LON_MAX
        )

        print()

        print(
            f"Arabian Sea latitude coverage : "
            f"{'✓ YES' if covers_lat else '✗ NO'}"
        )

        print(
            f"Arabian Sea longitude coverage: "
            f"{'✓ YES' if covers_lon else '✗ NO'}"
        )

    # --------------------------------------------------------
    # 9. Resolution check
    # --------------------------------------------------------

    section("9. RESOLUTION CHECK")

    if lat_name and lon_name:

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        if len(lat) > 1:

            lat_spacing = np.median(
                np.abs(np.diff(lat))
            )

            print(
                f"Latitude resolution : "
                f"{lat_spacing:.6f}°"
            )

        if len(lon) > 1:

            lon_spacing = np.median(
                np.abs(np.diff(lon))
            )

            print(
                f"Longitude resolution: "
                f"{lon_spacing:.6f}°"
            )

        print()
        print("OceanEmbed target: 0.25° × 0.25°")

    # --------------------------------------------------------
    # 10. Wind value audit
    # --------------------------------------------------------

    section("10. WIND VALUE AUDIT")

    for name in wind_candidates:

        var = ds[name]

        print()
        print(f"Variable: {name}")

        # Only first time step.
        # Prevents unnecessary full-file loading.

        if time_name and time_name in var.dims:

            sample = var.isel(
                {time_name: 0}
            )

        else:

            sample = var

        values = sample.values

        valid = values[np.isfinite(values)]

        if valid.size == 0:

            print("⚠ No valid values")

            continue

        print(f"Min : {valid.min():.4f}")
        print(f"Max : {valid.max():.4f}")
        print(f"Mean: {valid.mean():.4f}")
        print(f"Std : {valid.std():.4f}")

        missing = np.sum(~np.isfinite(values))

        print(
            f"Missing: "
            f"{missing:,} / {values.size:,} "
            f"({missing / values.size * 100:.2f}%)"
        )

    # --------------------------------------------------------
    # 11. Hourly → daily aggregation preview
    # --------------------------------------------------------

    section("11. DAILY AGGREGATION TEST")

    if time_name:

        for name in wind_candidates:

            var = ds[name]

            if time_name not in var.dims:
                continue

            print()
            print(f"Testing daily aggregation for: {name}")

            daily = var.resample(
                {time_name: "1D"}
            ).mean()

            print(
                f"Hourly shape: {var.shape}"
            )

            print(
                f"Daily shape : {daily.shape}"
            )

            print(
                f"Daily timestamps: "
                f"{daily[time_name].size}"
            )

            print("✓ Daily aggregation successful")

    # --------------------------------------------------------
    # 12. NaN / missing analysis
    # --------------------------------------------------------

    section("12. MISSING DATA ANALYSIS")

    for name in wind_candidates:

        var = ds[name]

        print()
        print(f"{name}")

        # First day only
        if time_name in var.dims:

            sample = var.isel(
                {time_name: 0}
            )

        else:

            sample = var

        values = sample.values

        missing = np.sum(~np.isfinite(values))
        total = values.size

        print(f"Total   : {total:,}")
        print(f"Missing : {missing:,}")
        print(
            f"Missing %: "
            f"{missing / total * 100:.2f}%"
        )

    # --------------------------------------------------------
    # 13. Generate maps
    # --------------------------------------------------------

    section("13. GENERATING VISUALIZATIONS")

    # First available wind variable

    if wind_candidates:

        wind_name = wind_candidates[0]

        var = ds[wind_name]

        if time_name in var.dims:

            sample = var.isel(
                {time_name: 0}
            )

        else:

            sample = var

        plt.figure(figsize=(12, 6))

        sample.plot()

        plt.title(
            f"ERA5 {wind_name} - First Timestamp"
        )

        plt.tight_layout()
        plt.show()

        # Daily mean map
        if time_name in var.dims:

            daily = var.resample(
                {time_name: "1D"}
            ).mean()

            plt.figure(figsize=(12, 6))

            daily.isel(
                {time_name: 0}
            ).plot()

            plt.title(
                f"ERA5 {wind_name} - First Daily Mean"
            )

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------------
    # 14. Storage / encoding
    # --------------------------------------------------------

    section("14. STORAGE / ENCODING")

    print("Dataset encoding:")

    for key, value in ds.encoding.items():

        print(f"  {key}: {value}")

    print()

    for name, var in ds.data_vars.items():

        print(f"{name} encoding:")

        for key, value in var.encoding.items():

            print(f"  {key}: {value}")

    # --------------------------------------------------------
    # 15. Final verdict helpers
    # --------------------------------------------------------

    section("15. FINAL CHECKLIST")

    checks = {}

    checks["File exists"] = FILE.exists()

    checks["Time coordinate found"] = (
        time_name is not None
    )

    checks["Latitude found"] = (
        lat_name is not None
    )

    checks["Longitude found"] = (
        lon_name is not None
    )

    checks["Wind variable detected"] = (
        len(wind_candidates) >= 2
    )

    if lat_name and lon_name:

        checks["Latitude covers target"] = (
            lat.min() <= LAT_MIN
            and lat.max() >= LAT_MAX
        )

        checks["Longitude covers target"] = (
            lon.min() <= LON_MIN
            and lon.max() >= LON_MAX
        )

    for name, result in checks.items():

        print(
            f"{'✓' if result else '✗'} {name}"
        )

    ds.close()

    print()
    print("=" * 75)
    print("ERA5 AUDIT COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    audit()