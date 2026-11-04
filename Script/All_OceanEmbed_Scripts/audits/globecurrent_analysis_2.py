from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FILE = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\globecurrent\2012"
    r"\GlobCurrent_2012_surface_uv.nc"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\globecurrent_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OPEN DATASET
# ============================================================

print("=" * 80)
print("GLOBECURRENT STATIC MASK VERIFICATION")
print("=" * 80)

print(f"\nFile:\n{FILE}")

if not FILE.exists():
    raise FileNotFoundError(
        f"\nFile does not exist:\n{FILE}"
    )

ds = xr.open_dataset(
    FILE,
    decode_times=True,
    mask_and_scale=True
)

print("\nDataset opened successfully.")


# ============================================================
# BASIC STRUCTURE
# ============================================================

print("\n" + "-" * 80)
print("DATASET STRUCTURE")
print("-" * 80)

print("\nDimensions:")

for name, size in ds.sizes.items():
    print(f"  {name}: {size}")


uo = ds["uo"]
vo = ds["vo"]

lat = ds["latitude"].values
lon = ds["longitude"].values

print("\nUO shape:")
print(uo.shape)

print("\nVO shape:")
print(vo.shape)

print("\nDepth:")
print(ds["depth"].values)


# ============================================================
# COORDINATE CHECK
# ============================================================

print("\n" + "-" * 80)
print("GEOGRAPHIC GRID")
print("-" * 80)

print(
    f"\nLatitude range : "
    f"{lat.min():.6f} to {lat.max():.6f}"
)

print(
    f"Longitude range: "
    f"{lon.min():.6f} to {lon.max():.6f}"
)

if len(lat) > 1:
    lat_spacing = np.median(
        np.abs(np.diff(lat))
    )
else:
    lat_spacing = np.nan

if len(lon) > 1:
    lon_spacing = np.median(
        np.abs(np.diff(lon))
    )
else:
    lon_spacing = np.nan

print(
    f"Latitude spacing : {lat_spacing:.6f} degrees"
)

print(
    f"Longitude spacing: {lon_spacing:.6f} degrees"
)

print(
    f"Spatial grid     : {len(lat)} × {len(lon)}"
)


# ============================================================
# BUILD U/V MASKS
# ============================================================

print("\n" + "-" * 80)
print("BUILDING MASKS")
print("-" * 80)

# ------------------------------------------------------------
# Missing = True
# ------------------------------------------------------------

uo_missing = ~np.isfinite(
    uo.values
)

vo_missing = ~np.isfinite(
    vo.values
)

print(
    "\nRaw UO mask shape:",
    uo_missing.shape
)

print(
    "Raw VO mask shape:",
    vo_missing.shape
)


# ============================================================
# U/V MASK CONSISTENCY
# ============================================================

print("\n" + "-" * 80)
print("U/V MASK CONSISTENCY")
print("-" * 80)

different_uv = (
    uo_missing != vo_missing
)

different_uv_count = int(
    different_uv.sum()
)

total_elements = int(
    different_uv.size
)

print(
    f"\nU/V mask disagreement cells: "
    f"{different_uv_count:,}"
)

print(
    f"Percentage: "
    f"{100 * different_uv_count / total_elements:.8f}%"
)

if different_uv_count == 0:
    print(
        "\nRESULT: PASS"
    )
    print(
        "UO and VO have exactly the same missing-value mask."
    )
else:
    print(
        "\nRESULT: WARNING"
    )
    print(
        "UO and VO do not have identical masks."
    )


# ============================================================
# TEMPORAL MASK CONSISTENCY
# ============================================================

print("\n" + "-" * 80)
print("TEMPORAL MASK CONSISTENCY")
print("-" * 80)

# uo_missing dimensions:
#
# time × depth × latitude × longitude
#
# We have exactly one depth level.
#
# Take first day as reference.

reference_mask = uo_missing[0, 0, :, :]

# Compare every day with day 0.

mask_difference_from_first = (
    uo_missing[:, 0, :, :]
    != reference_mask[np.newaxis, :, :]
)

difference_count = int(
    mask_difference_from_first.sum()
)

print(
    f"\nTotal mask disagreements against first day: "
    f"{difference_count:,}"
)

if difference_count == 0:

    print("\nRESULT: PASS")

    print(
        "The geographic missing-value mask is identical "
        "on every day of 2012."
    )

else:

    print("\nRESULT: WARNING")

    print(
        "The mask changes over time."
    )


# ============================================================
# STATIC SPATIAL MASK
# ============================================================

static_missing_mask = reference_mask

static_valid_mask = ~static_missing_mask

total_spatial_cells = static_missing_mask.size

missing_cells = int(
    static_missing_mask.sum()
)

valid_cells = int(
    static_valid_mask.sum()
)

missing_percent = (
    100.0 *
    missing_cells /
    total_spatial_cells
)

valid_percent = (
    100.0 *
    valid_cells /
    total_spatial_cells
)

print("\n" + "-" * 80)
print("STATIC SPATIAL MASK")
print("-" * 80)

print(
    f"\nTotal spatial cells : "
    f"{total_spatial_cells:,}"
)

print(
    f"Always missing      : "
    f"{missing_cells:,}"
)

print(
    f"Always valid        : "
    f"{valid_cells:,}"
)

print(
    f"\nMissing percentage: "
    f"{missing_percent:.6f}%"
)

print(
    f"Valid percentage  : "
    f"{valid_percent:.6f}%"
)


# ============================================================
# EXPECTED 50.475% CHECK
# ============================================================

EXPECTED = 50.475

difference = abs(
    missing_percent - EXPECTED
)

print("\n" + "-" * 80)
print("50.475% VERIFICATION")
print("-" * 80)

print(
    f"\nObserved : {missing_percent:.6f}%"
)

print(
    f"Expected : {EXPECTED:.6f}%"
)

print(
    f"Difference: {difference:.10f} percentage points"
)

if difference < 1e-9:

    print(
        "\nRESULT: EXACT MATCH"
    )

else:

    print(
        "\nRESULT: Different from reported value."
    )


# ============================================================
# CHECK WHETHER EVERY MASKED CELL IS PERMANENTLY MASKED
# ============================================================

print("\n" + "-" * 80)
print("PERMANENT MASK CHECK")
print("-" * 80)

# Number of missing days for every spatial cell.

missing_days = np.sum(
    uo_missing[:, 0, :, :],
    axis=0
)

always_missing = (
    missing_days == len(ds["time"])
)

always_valid = (
    missing_days == 0
)

partially_missing = (
    (missing_days > 0) &
    (missing_days < len(ds["time"]))
)

print(
    f"\nAlways missing cells : "
    f"{always_missing.sum():,}"
)

print(
    f"Always valid cells   : "
    f"{always_valid.sum():,}"
)

print(
    f"Partially missing    : "
    f"{partially_missing.sum():,}"
)

if partially_missing.sum() == 0:

    print(
        "\nRESULT: PASS"
    )

    print(
        "There are no spatial cells that intermittently "
        "disappear during the year."
    )

else:

    print(
        "\nRESULT: WARNING"
    )

    print(
        "Some spatial cells are intermittently missing."
    )


# ============================================================
# CREATE MASK DATASET
# ============================================================

print("\n" + "-" * 80)
print("SAVING MASK DATA")
print("-" * 80)

mask_ds = xr.Dataset(
    {
        "missing_mask": (
            ("latitude", "longitude"),
            static_missing_mask.astype(np.int8)
        ),

        "valid_mask": (
            ("latitude", "longitude"),
            static_valid_mask.astype(np.int8)
        ),

        "missing_days": (
            ("latitude", "longitude"),
            missing_days.astype(np.int16)
        ),
    },

    coords={
        "latitude": lat,
        "longitude": lon,
    }
)

mask_ds["missing_mask"].attrs = {
    "description":
        "1 = UO is missing at this geographic cell; "
        "0 = UO is valid",
}

mask_ds["valid_mask"].attrs = {
    "description":
        "1 = UO is valid at this geographic cell; "
        "0 = UO is missing",
}

mask_ds["missing_days"].attrs = {
    "description":
        "Number of days in 2012 for which UO is missing "
        "at this geographic cell",
}

mask_file = (
    OUTPUT_DIR /
    "globecurrent_2012_static_mask.nc"
)

mask_ds.to_netcdf(
    mask_file
)

print(
    f"\nMask NetCDF written to:\n{mask_file}"
)


# ============================================================
# SAVE CSV
# ============================================================

print("\nSaving CSV...")

lon_grid, lat_grid = np.meshgrid(
    lon,
    lat
)

csv_data = np.column_stack(
    [
        lat_grid.ravel(),
        lon_grid.ravel(),
        static_missing_mask.ravel().astype(int),
        static_valid_mask.ravel().astype(int),
        missing_days.ravel(),
    ]
)

csv_file = (
    OUTPUT_DIR /
    "globecurrent_2012_mask_cells.csv"
)

np.savetxt(
    csv_file,
    csv_data,
    delimiter=",",
    header=(
        "latitude,longitude,"
        "missing_mask,valid_mask,missing_days"
    ),
    comments="",
    fmt=[
        "%.8f",
        "%.8f",
        "%d",
        "%d",
        "%d",
    ]
)

print(
    f"CSV written to:\n{csv_file}"
)


# ============================================================
# PLOT 1: VALID / MASKED
# ============================================================

print("\n" + "-" * 80)
print("CREATING GEOGRAPHIC MASK PLOT")
print("-" * 80)

fig, ax = plt.subplots(
    figsize=(14, 7)
)

# 1 = missing
# 0 = valid

im = ax.pcolormesh(
    lon,
    lat,
    static_missing_mask.astype(float),
    shading="auto"
)

ax.set_xlabel(
    "Longitude (degrees east)"
)

ax.set_ylabel(
    "Latitude (degrees north)"
)

ax.set_title(
    "GlobCurrent 2012 Static Missing-Value Mask\n"
    "0 = valid current cell, 1 = permanently masked"
)

ax.set_xlim(
    lon.min(),
    lon.max()
)

ax.set_ylim(
    lat.min(),
    lat.max()
)

fig.colorbar(
    im,
    ax=ax,
    label="Missing mask"
)

fig.tight_layout()

plot_file = (
    OUTPUT_DIR /
    "globecurrent_2012_static_mask.png"
)

fig.savefig(
    plot_file,
    dpi=200
)

plt.close(fig)

print(
    f"\nMask plot written to:\n{plot_file}"
)


# ============================================================
# PLOT 2: MISSING DAYS
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 7)
)

im = ax.pcolormesh(
    lon,
    lat,
    missing_days,
    shading="auto"
)

ax.set_xlabel(
    "Longitude (degrees east)"
)

ax.set_ylabel(
    "Latitude (degrees north)"
)

ax.set_title(
    "GlobCurrent 2012 Missing Days per Spatial Cell"
)

ax.set_xlim(
    lon.min(),
    lon.max()
)

ax.set_ylim(
    lat.min(),
    lat.max()
)

fig.colorbar(
    im,
    ax=ax,
    label="Number of missing days"
)

fig.tight_layout()

missing_days_plot = (
    OUTPUT_DIR /
    "globecurrent_2012_missing_days.png"
)

fig.savefig(
    missing_days_plot,
    dpi=200
)

plt.close(fig)

print(
    f"Missing-days plot written to:\n"
    f"{missing_days_plot}"
)


# ============================================================
# SAVE NUMPY MASK
# ============================================================

npz_file = (
    OUTPUT_DIR /
    "globecurrent_2012_static_mask.npz"
)

np.savez_compressed(
    npz_file,
    latitude=lat,
    longitude=lon,
    missing_mask=static_missing_mask,
    valid_mask=static_valid_mask,
    missing_days=missing_days,
)

print(
    f"\nNPZ written to:\n{npz_file}"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n")
print("=" * 80)
print("FINAL MASK VERIFICATION")
print("=" * 80)

checks = {
    "U/V masks identical":
        different_uv_count == 0,

    "Mask identical across all days":
        difference_count == 0,

    "No partially missing spatial cells":
        partially_missing.sum() == 0,

    "50.475% figure reproduced":
        difference < 1e-9,
}


for name, passed in checks.items():

    print(
        f"\n{'PASS' if passed else 'WARNING'}"
        f"  {name}"
    )


all_pass = all(
    checks.values()
)

print("\n" + "-" * 80)

if all_pass:

    print(
        "OVERALL RESULT: STATIC MASK CONFIRMED"
    )

    print(
        "\nThe ~50.475% missing-value fraction is a "
        "persistent geographic mask rather than "
        "time-varying missing observations."
    )

    print(
        "\nNo NaN values were modified."
    )

else:

    print(
        "OVERALL RESULT: FURTHER INVESTIGATION REQUIRED"
    )


print("\n" + "=" * 80)

ds.close()
mask_ds.close()