from pathlib import Path
import json
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# =============================================================================
# OCEANEMBED GLORYS SCIENTIFIC SPATIAL AUDIT
# =============================================================================
#
# Audits the reduced GLORYS January 2012 dataset.
#
# Checks:
#   1. File structure
#   2. Coordinate/grid correctness
#   3. Temperature ranges
#   4. Surface / 100 / 500 / 1000 m maps
#   5. Valid-data masks
#   6. Zonal/meridional spatial behavior
#   7. Vertical temperature section
#   8. Temporal behavior
#   9. NaN structure
#
# This script DOES NOT modify the source NetCDF.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = Path(
    r"D:\OceanEmbed\oceanembed-data\GLORYS\processed"
    r"\GLORYS_thetao_20120101_20120131_NIO_0p25deg_15depth.nc"
)

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\GLORYS\audit_2012_01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# EXPECTED OCEANEMBED GRID
# =============================================================================

EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0

EXPECTED_HORIZONTAL_STEP = 0.25

EXPECTED_DEPTHS = np.array(
    [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ],
    dtype=np.float32,
)

EXPECTED_TIME_COUNT = 31


# =============================================================================
# PLOT SETTINGS
# =============================================================================

PLOT_DEPTHS = [
    0,
    100,
    500,
    1000,
]


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_depth_index(depth_values, target):
    """
    Find exact target depth index.
    """

    matches = np.where(
        np.isclose(
            depth_values,
            target,
            atol=1e-4,
        )
    )[0]

    if len(matches) == 0:
        raise ValueError(
            f"Depth {target} m not found."
        )

    return int(matches[0])


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"  Saved: {path.name}"
    )


# =============================================================================
# LOAD DATA
# =============================================================================

print_header(
    "OCEANEMBED GLORYS SCIENTIFIC SPATIAL AUDIT"
)

print(
    f"Input:\n{INPUT_FILE}"
)

print(
    f"Output:\n{OUTPUT_DIR}"
)

if not INPUT_FILE.exists():
    print()
    print(
        "ERROR: Input NetCDF does not exist."
    )
    sys.exit(1)


print()
print(
    "Opening NetCDF..."
)

try:
    ds = xr.open_dataset(
        INPUT_FILE,
        decode_cf=True,
    )

except Exception as exc:
    print(
        f"ERROR: Could not open NetCDF:\n{exc}"
    )
    sys.exit(1)


# =============================================================================
# BASIC STRUCTURE
# =============================================================================

print_header(
    "1. DATASET STRUCTURE"
)

print(
    ds
)

if "thetao" not in ds:
    print(
        "ERROR: thetao variable is missing."
    )
    ds.close()
    sys.exit(1)

thetao = ds["thetao"]

print()
print(
    f"thetao dimensions : {thetao.dims}"
)

print(
    f"thetao shape      : {thetao.shape}"
)

print(
    f"thetao dtype      : {thetao.dtype}"
)


# =============================================================================
# COORDINATE CHECK
# =============================================================================

print_header(
    "2. COORDINATE / GRID AUDIT"
)

lat = ds["latitude"].values
lon = ds["longitude"].values
depth = ds["depth"].values
time = ds["time"].values

print(
    f"Longitude : {lon[0]:.4f} -> {lon[-1]:.4f}"
)

print(
    f"Latitude  : {lat[0]:.4f} -> {lat[-1]:.4f}"
)

print(
    f"Depth     : {depth}"
)

print(
    f"Time      : {time[0]} -> {time[-1]}"
)

# -------------------------------------------------------------------------
# Longitude
# -------------------------------------------------------------------------

lon_step = np.diff(lon)

lon_ok = (
    np.isclose(
        lon[0],
        EXPECTED_LON_MIN,
        atol=1e-5,
    )
    and
    np.isclose(
        lon[-1],
        EXPECTED_LON_MAX,
        atol=1e-5,
    )
    and
    np.allclose(
        lon_step,
        EXPECTED_HORIZONTAL_STEP,
        atol=1e-5,
    )
)

print()
print(
    "Longitude grid : "
    + ("PASS" if lon_ok else "FAIL")
)

# -------------------------------------------------------------------------
# Latitude
# -------------------------------------------------------------------------

lat_step = np.diff(lat)

lat_ok = (
    np.isclose(
        lat[0],
        EXPECTED_LAT_MIN,
        atol=1e-5,
    )
    and
    np.isclose(
        lat[-1],
        EXPECTED_LAT_MAX,
        atol=1e-5,
    )
    and
    np.allclose(
        lat_step,
        EXPECTED_HORIZONTAL_STEP,
        atol=1e-5,
    )
)

print(
    "Latitude grid  : "
    + ("PASS" if lat_ok else "FAIL")
)

# -------------------------------------------------------------------------
# Depth
# -------------------------------------------------------------------------

depth_ok = np.allclose(
    depth,
    EXPECTED_DEPTHS,
    atol=1e-4,
)

print(
    "Depth grid      : "
    + ("PASS" if depth_ok else "FAIL")
)

# -------------------------------------------------------------------------
# Time
# -------------------------------------------------------------------------

time_diff = np.diff(
    time.astype(
        "datetime64[h]"
    )
)

time_ok = (
    len(time) == EXPECTED_TIME_COUNT
    and
    np.all(
        time_diff
        == np.timedelta64(
            24,
            "h",
        )
    )
)

print(
    "Daily time axis  : "
    + ("PASS" if time_ok else "FAIL")
)


# =============================================================================
# NUMERICAL DATA AUDIT
# =============================================================================

print_header(
    "3. TEMPERATURE DATA AUDIT"
)

values = thetao.values

finite = np.isfinite(values)

finite_values = values[finite]

if len(finite_values) == 0:
    print(
        "ERROR: Dataset contains no finite "
        "temperature values."
    )

    ds.close()
    sys.exit(1)

overall_min = float(
    np.min(finite_values)
)

overall_max = float(
    np.max(finite_values)
)

overall_mean = float(
    np.mean(finite_values)
)

overall_std = float(
    np.std(finite_values)
)

valid_fraction = (
    np.count_nonzero(finite)
    / values.size
)

print(
    f"Minimum temperature : "
    f"{overall_min:.4f} °C"
)

print(
    f"Maximum temperature : "
    f"{overall_max:.4f} °C"
)

print(
    f"Mean temperature    : "
    f"{overall_mean:.4f} °C"
)

print(
    f"Std deviation       : "
    f"{overall_std:.4f} °C"
)

print(
    f"Valid fraction      : "
    f"{valid_fraction * 100:.2f}%"
)

print(
    f"NaN fraction        : "
    f"{(1 - valid_fraction) * 100:.2f}%"
)

# Basic physical sanity limits.
physical_ok = (
    overall_min >= -5.0
    and
    overall_max <= 40.0
)

print()
print(
    "Basic physical range: "
    + (
        "PASS"
        if physical_ok
        else "WARNING"
    )
)


# =============================================================================
# DEPTH STATISTICS
# =============================================================================

print_header(
    "4. DEPTH-BY-DEPTH STATISTICS"
)

depth_statistics = []

for d in depth:

    layer = thetao.sel(
        depth=d
    )

    layer_values = layer.values

    finite_layer = (
        np.isfinite(
            layer_values
        )
    )

    count = int(
        np.count_nonzero(
            finite_layer
        )
    )

    total = int(
        layer_values.size
    )

    fraction = (
        count / total
        if total
        else 0.0
    )

    if count:

        layer_min = float(
            np.min(
                layer_values[
                    finite_layer
                ]
            )
        )

        layer_max = float(
            np.max(
                layer_values[
                    finite_layer
                ]
            )
        )

        layer_mean = float(
            np.mean(
                layer_values[
                    finite_layer
                ]
            )
        )

    else:

        layer_min = np.nan
        layer_max = np.nan
        layer_mean = np.nan

    depth_statistics.append(
        {
            "depth_m": float(d),
            "min_C": layer_min,
            "max_C": layer_max,
            "mean_C": layer_mean,
            "valid_fraction": fraction,
        }
    )

    print(
        f"{float(d):7.1f} m | "
        f"min={layer_min:7.3f} | "
        f"max={layer_max:7.3f} | "
        f"mean={layer_mean:7.3f} | "
        f"valid={fraction * 100:6.2f}%"
    )


# =============================================================================
# SPATIAL TEMPERATURE MAPS
# =============================================================================

print_header(
    "5. SPATIAL TEMPERATURE MAPS"
)

for target_depth in PLOT_DEPTHS:

    print(
        f"Creating {target_depth} m temperature map..."
    )

    layer = thetao.sel(
        depth=target_depth
    ).mean(
        dim="time"
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        layer.values,
        shading="auto",
    )

    contour = ax.contour(
        lon,
        lat,
        layer.values,
        levels=10,
        linewidths=0.5,
    )

    ax.clabel(
        contour,
        inline=True,
        fontsize=7,
        fmt="%.1f",
    )

    ax.set_xlabel(
        "Longitude (°E)"
    )

    ax.set_ylabel(
        "Latitude (°N)"
    )

    ax.set_title(
        f"GLORYS January 2012 Mean Potential "
        f"Temperature at {target_depth} m"
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    cbar.set_label(
        "Potential temperature (°C)"
    )

    ax.set_xlim(
        EXPECTED_LON_MIN,
        EXPECTED_LON_MAX,
    )

    ax.set_ylim(
        EXPECTED_LAT_MIN,
        EXPECTED_LAT_MAX,
    )

    save_figure(
        fig,
        f"temperature_{target_depth:04d}m.png",
    )


# =============================================================================
# VALID-DATA / OCEAN MASK MAPS
# =============================================================================

print_header(
    "6. VALID-DATA / OCEAN MASK MAPS"
)

for target_depth in PLOT_DEPTHS:

    print(
        f"Creating {target_depth} m validity map..."
    )

    layer = thetao.sel(
        depth=target_depth
    )

    valid_fraction_map = (
        layer.notnull()
        .mean(
            dim="time"
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        valid_fraction_map.values,
        shading="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel(
        "Longitude (°E)"
    )

    ax.set_ylabel(
        "Latitude (°N)"
    )

    ax.set_title(
        f"GLORYS Valid-Data Fraction "
        f"at {target_depth} m\n"
        f"January 2012"
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    cbar.set_label(
        "Fraction of days with valid data"
    )

    ax.set_xlim(
        EXPECTED_LON_MIN,
        EXPECTED_LON_MAX,
    )

    ax.set_ylim(
        EXPECTED_LAT_MIN,
        EXPECTED_LAT_MAX,
    )

    save_figure(
        fig,
        f"valid_fraction_{target_depth:04d}m.png",
    )


# =============================================================================
# VERTICAL TEMPERATURE SECTION
# =============================================================================

print_header(
    "7. VERTICAL TEMPERATURE STRUCTURE"
)

# Use the temporal mean.
vertical_mean = thetao.mean(
    dim="time"
)

# Average longitude to show lat-depth structure.
lat_depth = vertical_mean.mean(
    dim="longitude"
)

fig, ax = plt.subplots(
    figsize=(11, 7)
)

mesh = ax.pcolormesh(
    lat,
    depth,
    lat_depth.values,
    shading="auto",
)

ax.invert_yaxis()

ax.set_xlabel(
    "Latitude (°N)"
)

ax.set_ylabel(
    "Depth (m)"
)

ax.set_title(
    "GLORYS January 2012 Mean "
    "Temperature: Latitude–Depth Section"
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
)

cbar.set_label(
    "Potential temperature (°C)"
)

save_figure(
    fig,
    "vertical_latitude_depth_section.png",
)


# =============================================================================
# ZONAL VERTICAL SECTION
# =============================================================================

print(
    "Creating longitude-depth section..."
)

lon_depth = vertical_mean.mean(
    dim="latitude"
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

mesh = ax.pcolormesh(
    lon,
    depth,
    lon_depth.values,
    shading="auto",
)

ax.invert_yaxis()

ax.set_xlabel(
    "Longitude (°E)"
)

ax.set_ylabel(
    "Depth (m)"
)

ax.set_title(
    "GLORYS January 2012 Mean "
    "Temperature: Longitude–Depth Section"
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
)

cbar.set_label(
    "Potential temperature (°C)"
)

save_figure(
    fig,
    "vertical_longitude_depth_section.png",
)


# =============================================================================
# TEMPORAL SANITY CHECK
# =============================================================================

print_header(
    "8. TEMPORAL SANITY CHECK"
)

# Domain-wide daily mean.
daily_mean = thetao.mean(
    dim=[
        "depth",
        "latitude",
        "longitude",
    ],
    skipna=True,
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    time,
    daily_mean.values,
    marker="o",
    markersize=3,
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Mean potential temperature (°C)"
)

ax.set_title(
    "GLORYS January 2012 "
    "Daily Domain-Depth Mean Temperature"
)

ax.grid(
    True,
    alpha=0.3,
)

fig.autofmt_xdate()

save_figure(
    fig,
    "daily_mean_temperature.png",
)


# =============================================================================
# DAILY VALID FRACTION
# =============================================================================

print(
    "Creating daily valid-data fraction plot..."
)

daily_valid_fraction = (
    thetao.notnull()
    .mean(
        dim=[
            "depth",
            "latitude",
            "longitude",
        ]
    )
)

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    time,
    daily_valid_fraction.values,
    marker="o",
    markersize=3,
)

ax.set_xlabel(
    "Date"
)

ax.set_ylabel(
    "Valid fraction"
)

ax.set_title(
    "GLORYS January 2012 "
    "Daily Valid-Data Fraction"
)

ax.set_ylim(
    0,
    1,
)

ax.grid(
    True,
    alpha=0.3,
)

fig.autofmt_xdate()

save_figure(
    fig,
    "daily_valid_fraction.png",
)


# =============================================================================
# DEPTH VALID FRACTION
# =============================================================================

print_header(
    "9. DEPTH VALID-DATA PROFILE"
)

depth_valid_fraction = (
    thetao.notnull()
    .mean(
        dim=[
            "time",
            "latitude",
            "longitude",
        ]
    )
)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.plot(
    depth_valid_fraction.values,
    depth,
    marker="o",
)

ax.invert_yaxis()

ax.set_xlabel(
    "Valid-data fraction"
)

ax.set_ylabel(
    "Depth (m)"
)

ax.set_title(
    "GLORYS January 2012 "
    "Valid-Data Fraction by Depth"
)

ax.set_xlim(
    0,
    1,
)

ax.grid(
    True,
    alpha=0.3,
)

save_figure(
    fig,
    "depth_valid_fraction.png",
)


# =============================================================================
# NaN MASK CONSISTENCY
# =============================================================================

print_header(
    "10. TEMPORAL MASK CONSISTENCY"
)

# A grid cell is classified as:
#
#   1. Always valid
#   2. Always invalid
#   3. Variable availability
#
# across the month.

valid_count_per_cell = (
    thetao.notnull()
    .sum(
        dim="time"
    )
)

always_valid = (
    valid_count_per_cell
    == EXPECTED_TIME_COUNT
)

always_invalid = (
    valid_count_per_cell
    == 0
)

variable = (
    (~always_valid)
    &
    (~always_invalid)
)

print(
    f"Always valid cells   : "
    f"{int(always_valid.sum())}"
)

print(
    f"Always invalid cells : "
    f"{int(always_invalid.sum())}"
)

print(
    f"Variable cells       : "
    f"{int(variable.sum())}"
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

mask_class = (
    always_valid.astype(
        np.int8
    )
    +
    variable.astype(
        np.int8
    ) * 2
)

mesh = ax.pcolormesh(
    lon,
    lat,
    mask_class.isel(
        depth=0
    ).values,
    shading="auto",
)

ax.set_xlabel(
    "Longitude (°E)"
)

ax.set_ylabel(
    "Latitude (°N)"
)

ax.set_title(
    "Surface Temporal Data-Availability Classification\n"
    "1 = Always valid, 2 = Variable"
)

cbar = fig.colorbar(
    mesh,
    ax=ax,
)

cbar.set_label(
    "Availability class"
)

save_figure(
    fig,
    "surface_temporal_mask_consistency.png",
)


# =============================================================================
# REPORT
# =============================================================================

print_header(
    "11. GENERATING AUDIT REPORT"
)

report = {
    "input_file": str(
        INPUT_FILE
    ),
    "dataset": {
        "dimensions": {
            key: int(value)
            for key, value
            in ds.sizes.items()
        },
        "variable": "thetao",
        "dtype": str(
            thetao.dtype
        ),
    },
    "coordinates": {
        "longitude": {
            "min": float(lon[0]),
            "max": float(lon[-1]),
            "count": int(len(lon)),
            "step": float(
                np.median(lon_step)
            ),
            "pass": bool(lon_ok),
        },
        "latitude": {
            "min": float(lat[0]),
            "max": float(lat[-1]),
            "count": int(len(lat)),
            "step": float(
                np.median(lat_step)
            ),
            "pass": bool(lat_ok),
        },
        "depths_m": [
            float(x)
            for x in depth
        ],
        "depth_pass": bool(
            depth_ok
        ),
        "time_count": int(
            len(time)
        ),
        "time_pass": bool(
            time_ok
        ),
    },
    "temperature": {
        "minimum_C": overall_min,
        "maximum_C": overall_max,
        "mean_C": overall_mean,
        "std_C": overall_std,
        "valid_fraction": valid_fraction,
        "physical_range_pass": bool(
            physical_ok
        ),
    },
    "depth_statistics": depth_statistics,
    "mask": {
        "always_valid_surface_cells": int(
            always_valid.isel(
                depth=0
            ).sum()
        ),
        "always_invalid_surface_cells": int(
            always_invalid.isel(
                depth=0
            ).sum()
        ),
        "variable_surface_cells": int(
            variable.isel(
                depth=0
            ).sum()
        ),
    },
}


report_path = (
    OUTPUT_DIR
    / "glorys_spatial_audit_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )

print(
    f"Report saved:\n{report_path}"
)


# =============================================================================
# FINAL VERDICT
# =============================================================================

print_header(
    "GLORYS SPATIAL AUDIT COMPLETE"
)

all_structural_checks = (
    lon_ok
    and lat_ok
    and depth_ok
    and time_ok
    and physical_ok
)

if all_structural_checks:

    print(
        "STRUCTURAL STATUS: PASS"
    )

else:

    print(
        "STRUCTURAL STATUS: REVIEW"
    )

print()
print(
    "IMPORTANT:"
)

print(
    "The plots in the audit directory must be "
    "visually inspected before approving the "
    "full 2012-2024 acquisition."
)

print()
print(
    f"Audit directory:\n{OUTPUT_DIR}"
)

print(
    "=" * 80
)

ds.close()