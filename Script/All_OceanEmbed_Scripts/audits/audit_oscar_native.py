"""
OSCAR v2.0 FINAL
Native NetCDF Audit
===================

Audits one downloaded OSCAR native granule.

Input:
    D:\\OceanEmbed\\oceanembed-data\\raw\\oscar\\test\\
        oscar_currents_final_20120101.nc

Reports:
    - file size
    - dimensions
    - coordinates
    - resolution
    - variables
    - dtype
    - units
    - fill values
    - valid ranges
    - NaN / missing values
    - basic statistics
    - global attributes
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

FILE = Path(
    r"D:\OceanEmbed\oceanembed-data\raw\oscar\test"
    r"\oscar_currents_final_20120101.nc"
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def format_bytes(size: int) -> str:
    """Human-readable file size."""

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    value = float(size)

    for unit in units:

        if value < 1024 or unit == units[-1]:
            return f"{value:,.2f} {unit}"

        value /= 1024

    return f"{size:,} B"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return value


# ============================================================
# CHECK FILE
# ============================================================

def check_file() -> None:

    print_header(
        "FILE CHECK"
    )

    if not FILE.exists():

        print(
            "ERROR: OSCAR NetCDF file was not found."
        )

        print(
            f"Expected:\n{FILE}"
        )

        sys.exit(1)

    size = FILE.stat().st_size

    print(
        f"File:\n{FILE}"
    )

    print(
        f"Size: {format_bytes(size)}"
    )


# ============================================================
# OPEN DATASET
# ============================================================

def audit_dataset():

    print_header(
        "OPENING NETCDF"
    )

    print(
        "Opening with xarray..."
    )

    try:

        ds = xr.open_dataset(
            FILE,
            engine="netcdf4",
            decode_cf=True,
            mask_and_scale=True,
        )

    except Exception as exc:

        print()
        print(
            "ERROR: Could not open NetCDF."
        )

        print(
            exc
        )

        sys.exit(1)

    print(
        "NetCDF opened successfully."
    )

    return ds


# ============================================================
# DIMENSIONS
# ============================================================

def audit_dimensions(ds):

    print_header(
        "DIMENSIONS"
    )

    for name, size in ds.sizes.items():

        print(
            f"{name:15s}: {size:,}"
        )


# ============================================================
# COORDINATES
# ============================================================

def audit_coordinates(ds):

    print_header(
        "COORDINATES"
    )

    for name in ds.coords:

        coord = ds[name]

        print()
        print(
            f"[{name}]"
        )

        print(
            f"  Dimensions : "
            f"{coord.dims}"
        )

        print(
            f"  Shape      : "
            f"{coord.shape}"
        )

        print(
            f"  Dtype      : "
            f"{coord.dtype}"
        )

        if coord.size > 0:

            values = coord.values

            print(
                f"  First      : "
                f"{values.flat[0]}"
            )

            print(
                f"  Last       : "
                f"{values.flat[-1]}"
            )

        if coord.size > 1:

            values = np.asarray(
                coord.values,
                dtype=float,
            )

            differences = np.diff(
                values
            )

            print(
                f"  Min step   : "
                f"{differences.min()}"
            )

            print(
                f"  Max step   : "
                f"{differences.max()}"
            )

            print(
                f"  Mean step  : "
                f"{differences.mean()}"
            )

        if coord.attrs:

            print(
                "  Attributes:"
            )

            for key, value in coord.attrs.items():

                print(
                    f"    {key}: {value}"
                )


# ============================================================
# VARIABLES
# ============================================================

def audit_variables(ds):

    print_header(
        "DATA VARIABLES"
    )

    for name, variable in ds.data_vars.items():

        print()
        print(
            f"[{name}]"
        )

        print(
            f"  Dimensions : "
            f"{variable.dims}"
        )

        print(
            f"  Shape      : "
            f"{variable.shape}"
        )

        print(
            f"  Dtype      : "
            f"{variable.dtype}"
        )

        print(
            f"  Size       : "
            f"{variable.size:,} values"
        )

        print(
            f"  Bytes/item : "
            f"{variable.dtype.itemsize}"
        )

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        if variable.attrs:

            print(
                "  Attributes:"
            )

            for key, value in variable.attrs.items():

                print(
                    f"    {key}: {value}"
                )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print(
            "  Statistics:"
        )

        try:

            values = variable.values

            finite = np.isfinite(
                values
            )

            finite_count = int(
                finite.sum()
            )

            total_count = values.size

            missing_count = (
                total_count
                - finite_count
            )

            missing_pct = (
                missing_count
                / total_count
                * 100
            )

            print(
                f"    Total       : "
                f"{total_count:,}"
            )

            print(
                f"    Finite      : "
                f"{finite_count:,}"
            )

            print(
                f"    Missing     : "
                f"{missing_count:,}"
            )

            print(
                f"    Missing %   : "
                f"{missing_pct:.6f}%"
            )

            if finite_count > 0:

                valid_values = values[
                    finite
                ]

                print(
                    f"    Min         : "
                    f"{np.min(valid_values)}"
                )

                print(
                    f"    Max         : "
                    f"{np.max(valid_values)}"
                )

                print(
                    f"    Mean        : "
                    f"{np.mean(valid_values)}"
                )

                print(
                    f"    Std         : "
                    f"{np.std(valid_values)}"
                )

        except Exception as exc:

            print(
                f"    Statistics unavailable: "
                f"{exc}"
            )


# ============================================================
# GLOBAL ATTRIBUTES
# ============================================================

def audit_global_attributes(ds):

    print_header(
        "GLOBAL ATTRIBUTES"
    )

    if not ds.attrs:

        print(
            "No global attributes found."
        )

        return

    for key, value in ds.attrs.items():

        print(
            f"{key}: {value}"
        )


# ============================================================
# DATASET SUMMARY
# ============================================================

def audit_summary(ds):

    print_header(
        "XARRAY DATASET SUMMARY"
    )

    print(
        ds
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "OSCAR v2.0 FINAL"
    )

    print(
        "NATIVE NETCDF AUDIT"
    )

    check_file()

    ds = audit_dataset()

    try:

        audit_dimensions(ds)

        audit_coordinates(ds)

        audit_variables(ds)

        audit_global_attributes(ds)

        audit_summary(ds)

    finally:

        ds.close()

    print()
    print_header(
        "AUDIT COMPLETE"
    )

    print()
    print(
        "No data was modified."
    )

    print(
        "No resampling or subsetting was performed."
    )

    print(
        "This audit only reads the native OSCAR file."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()