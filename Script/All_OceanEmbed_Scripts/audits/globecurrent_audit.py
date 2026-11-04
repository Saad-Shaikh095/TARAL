from pathlib import Path
import json
import numpy as np
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

FILES = {
    2012: Path(
        r"D:\OceanEmbed\oceanembed-data\raw\globecurrent\2012\GlobCurrent_2012_surface_uv.nc"
    ),
    2013: Path(
        r"D:\OceanEmbed\oceanembed-data\raw\globecurrent\2013\GlobCurrent_2013_surface_uv.nc"
    ),
}

OUTPUT_DIR = Path(
    r"D:\OceanEmbed\oceanembed-data\processed\globecurrent_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """
    Convert numpy scalar / array scalar to Python float.
    """
    try:
        return float(value)
    except Exception:
        return None


def summarize_variable(var):
    """
    Extract useful metadata from an xarray DataArray.
    """
    return {
        "name": var.name,
        "dims": list(var.dims),
        "shape": list(var.shape),
        "dtype": str(var.dtype),
        "attrs": {
            str(k): str(v)
            for k, v in var.attrs.items()
        },
        "encoding": {
            str(k): str(v)
            for k, v in var.encoding.items()
            if k in {
                "dtype",
                "_FillValue",
                "missing_value",
                "zlib",
                "complevel",
                "chunksizes"
            }
        },
    }


def finite_stats(values):
    """
    Statistics over finite values only.
    """
    values = np.asarray(values)

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    return {
        "count": int(finite.size),
        "min": safe_float(np.min(finite)),
        "max": safe_float(np.max(finite)),
        "mean": safe_float(np.mean(finite)),
        "median": safe_float(np.median(finite)),
        "std": safe_float(np.std(finite)),
    }


# ============================================================
# DEPTH ANALYSIS
# ============================================================

def inspect_depth(ds):
    """
    Determine whether depth exists and what levels are present.
    """

    depth_candidates = [
        name
        for name in ds.variables
        if name.lower() in {
            "depth",
            "depths",
            "z",
            "lev",
            "level",
            "deptht",
        }
    ]

    result = {
        "depth_variables_found": depth_candidates,
        "depth_values": [],
        "depth_attrs": {},
        "interpretation": None,
    }

    if not depth_candidates:
        result["interpretation"] = (
            "No conventional depth variable found."
        )
        return result

    depth_name = depth_candidates[0]
    depth = ds[depth_name]

    try:
        values = np.asarray(depth.values).flatten()

        result["depth_values"] = [
            safe_float(v) for v in values
        ]

        result["depth_attrs"] = {
            str(k): str(v)
            for k, v in depth.attrs.items()
        }

        if len(values) == 1 and np.isclose(values[0], 0):
            result["interpretation"] = (
                "File contains exactly one vertical level at 0 m. "
                "This is consistent with a surface-only extraction, "
                "but does not by itself prove whether 15 m was discarded "
                "during acquisition."
            )

        elif len(values) >= 2:
            result["interpretation"] = (
                "File contains multiple vertical levels. "
                "Check whether 0 m and approximately 15 m are present."
            )

        else:
            result["interpretation"] = (
                "A depth variable exists but its structure is unusual."
            )

    except Exception as exc:
        result["interpretation"] = (
            f"Could not inspect depth values: {exc}"
        )

    return result


# ============================================================
# MASK ANALYSIS
# ============================================================

def analyze_mask(uo, vo):
    """
    Analyze NaN structure.

    Important:
    We distinguish:
        1. cells invalid in both U and V
        2. cells valid in both
        3. cells where only U is missing
        4. cells where only V is missing

    Since land cells should normally be masked consistently
    between U and V, this gives us useful evidence about the mask.
    """

    u = np.asarray(uo.values)
    v = np.asarray(vo.values)

    u_missing = ~np.isfinite(u)
    v_missing = ~np.isfinite(v)

    both_missing = u_missing & v_missing
    both_valid = (~u_missing) & (~v_missing)

    u_only_missing = u_missing & (~v_missing)
    v_only_missing = v_missing & (~u_missing)

    total = u.size

    result = {
        "total_elements": int(total),

        "uo": {
            "missing_count": int(u_missing.sum()),
            "missing_percent": float(
                100.0 * u_missing.sum() / total
            ),
        },

        "vo": {
            "missing_count": int(v_missing.sum()),
            "missing_percent": float(
                100.0 * v_missing.sum() / total
            ),
        },

        "joint": {
            "both_missing_count": int(both_missing.sum()),
            "both_missing_percent": float(
                100.0 * both_missing.sum() / total
            ),

            "both_valid_count": int(both_valid.sum()),
            "both_valid_percent": float(
                100.0 * both_valid.sum() / total
            ),

            "uo_only_missing_count": int(u_only_missing.sum()),
            "uo_only_missing_percent": float(
                100.0 * u_only_missing.sum() / total
            ),

            "vo_only_missing_count": int(v_only_missing.sum()),
            "vo_only_missing_percent": float(
                100.0 * v_only_missing.sum() / total
            ),
        },
    }

    return result


# ============================================================
# SPATIAL MASK ANALYSIS
# ============================================================

def analyze_spatial_mask(uo, lat_name, lon_name):
    """
    Collapse the time dimension and determine how consistently
    each geographic cell is masked.

    This is critical for distinguishing geographic land masking
    from sporadic missing observations.
    """

    data = np.asarray(uo.values)

    # Identify dimensions
    dims = list(uo.dims)

    if lat_name not in dims or lon_name not in dims:
        return {
            "status": "Could not identify latitude/longitude dimensions."
        }

    time_axes = [
        i
        for i, dim in enumerate(dims)
        if dim not in {lat_name, lon_name, "depth"}
    ]

    if len(time_axes) == 0:
        missing_fraction = ~np.isfinite(data)
    else:
        # Collapse all non-spatial dimensions.
        missing_fraction = np.mean(
            ~np.isfinite(data),
            axis=tuple(time_axes)
        )

    result = {
        "status": "ok",

        "cells_always_valid": int(
            np.sum(missing_fraction == 0)
        ),

        "cells_always_missing": int(
            np.sum(missing_fraction == 1)
        ),

        "cells_partially_missing": int(
            np.sum(
                (missing_fraction > 0) &
                (missing_fraction < 1)
            )
        ),

        "total_spatial_cells": int(
            missing_fraction.size
        ),

        "mean_missing_fraction_per_spatial_cell": safe_float(
            np.mean(missing_fraction)
        ),
    }

    return result


# ============================================================
# DIMENSION DISCOVERY
# ============================================================

def identify_coordinate(ds, candidates):
    """
    Find coordinate variable using common names.
    """
    for candidate in candidates:
        if candidate in ds.variables:
            return candidate

    for name in ds.variables:
        lname = name.lower()

        for candidate in candidates:
            if candidate.lower() == lname:
                return name

    return None


# ============================================================
# MAIN AUDIT
# ============================================================

def audit_file(year, path):

    print()
    print("=" * 80)
    print(f"GLOBECURRENT FORENSIC AUDIT — {year}")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist:\n{path}"
        )

    print(f"\nFile:")
    print(path)

    file_size_mb = path.stat().st_size / (1024 ** 2)

    print(
        f"Size: {file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Open dataset
    # --------------------------------------------------------

    print("\nOpening NetCDF...")

    ds = xr.open_dataset(
        path,
        decode_times=True,
        mask_and_scale=True,
    )

    print("Opened successfully.")

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    print("\nDimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print("\nVariables:")
    for name in ds.variables:
        var = ds[name]
        print(
            f"  {name}: dims={var.dims}, shape={var.shape}"
        )

    # --------------------------------------------------------
    # Identify important variables
    # --------------------------------------------------------

    uo_name = identify_coordinate(
        ds,
        ["uo", "u", "eastward_velocity", "eastward_current"]
    )

    vo_name = identify_coordinate(
        ds,
        ["vo", "v", "northward_velocity", "northward_current"]
    )

    lat_name = identify_coordinate(
        ds,
        ["lat", "latitude", "nav_lat", "y"]
    )

    lon_name = identify_coordinate(
        ds,
        ["lon", "longitude", "nav_lon", "x"]
    )

    time_name = identify_coordinate(
        ds,
        ["time", "TIME", "time_counter"]
    )

    print("\nDetected variables:")
    print(f"  U velocity : {uo_name}")
    print(f"  V velocity : {vo_name}")
    print(f"  Latitude   : {lat_name}")
    print(f"  Longitude  : {lon_name}")
    print(f"  Time       : {time_name}")

    if uo_name is None:
        raise RuntimeError(
            "Could not identify U velocity variable."
        )

    if vo_name is None:
        raise RuntimeError(
            "Could not identify V velocity variable."
        )

    uo = ds[uo_name]
    vo = ds[vo_name]

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("DEPTH ANALYSIS")
    print("-" * 80)

    depth_info = inspect_depth(ds)

    print(
        "Depth variables:",
        depth_info["depth_variables_found"]
    )

    print(
        "Depth values:",
        depth_info["depth_values"]
    )

    print(
        "Interpretation:",
        depth_info["interpretation"]
    )

    # --------------------------------------------------------
    # Velocity metadata
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("UO METADATA")
    print("-" * 80)

    print(
        json.dumps(
            summarize_variable(uo),
            indent=2
        )
    )

    print("\n" + "-" * 80)
    print("VO METADATA")
    print("-" * 80)

    print(
        json.dumps(
            summarize_variable(vo),
            indent=2
        )
    )

    # --------------------------------------------------------
    # Coordinate metadata
    # --------------------------------------------------------

    coordinates = {}

    for name in [
        lat_name,
        lon_name,
        time_name,
    ]:
        if name is None:
            continue

        var = ds[name]

        coordinates[name] = {
            "dims": list(var.dims),
            "shape": list(var.shape),
            "attrs": {
                str(k): str(v)
                for k, v in var.attrs.items()
            },
        }

        try:
            values = np.asarray(var.values)

            if values.size > 0:
                coordinates[name]["min"] = safe_float(
                    np.nanmin(values)
                )
                coordinates[name]["max"] = safe_float(
                    np.nanmax(values)
                )

                if values.size > 1:
                    flat = values.flatten()

                    diffs = np.diff(flat)

                    finite_diffs = diffs[
                        np.isfinite(diffs)
                    ]

                    if finite_diffs.size:
                        coordinates[name]["approx_spacing"] = (
                            safe_float(
                                np.median(
                                    np.abs(finite_diffs)
                                )
                            )
                        )

        except Exception:
            pass

    # --------------------------------------------------------
    # NaN / mask analysis
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("MISSING VALUE ANALYSIS")
    print("-" * 80)

    mask_info = analyze_mask(
        uo,
        vo
    )

    print(
        json.dumps(
            mask_info,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Spatial analysis
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("SPATIAL MASK ANALYSIS")
    print("-" * 80)

    if lat_name and lon_name:

        spatial_info = analyze_spatial_mask(
            uo,
            lat_name,
            lon_name
        )

        print(
            json.dumps(
                spatial_info,
                indent=2
            )
        )

    else:

        spatial_info = {
            "status": "Latitude/longitude not identified."
        }

    # --------------------------------------------------------
    # Value statistics
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("VELOCITY STATISTICS")
    print("-" * 80)

    uo_stats = finite_stats(
        uo.values
    )

    vo_stats = finite_stats(
        vo.values
    )

    print("\nUO:")
    print(
        json.dumps(
            uo_stats,
            indent=2
        )
    )

    print("\nVO:")
    print(
        json.dumps(
            vo_stats,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Time analysis
    # --------------------------------------------------------

    time_info = {}

    if time_name:

        time_values = ds[time_name].values

        time_info = {
            "count": int(time_values.size),
            "first": str(time_values[0])
            if time_values.size else None,
            "last": str(time_values[-1])
            if time_values.size else None,
        }

    # --------------------------------------------------------
    # Dimension consistency
    # --------------------------------------------------------

    dimension_consistency = {
        "uo_dims": list(uo.dims),
        "uo_shape": list(uo.shape),
        "vo_dims": list(vo.dims),
        "vo_shape": list(vo.shape),
        "uo_vo_same_shape": (
            uo.shape == vo.shape
        ),
    }

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    depth_values = depth_info["depth_values"]

    if len(depth_values) == 1 and np.isclose(
        depth_values[0], 0
    ):
        surface_only = True
    else:
        surface_only = False

    total_missing = mask_info["joint"][
        "both_missing_percent"
    ]

    permanently_missing = spatial_info.get(
        "cells_always_missing",
        None
    )

    partially_missing = spatial_info.get(
        "cells_partially_missing",
        None
    )

    interpretation = {
        "surface_only_file": surface_only,

        "surface_only_reason": (
            "Exactly one depth level at 0 m."
            if surface_only
            else
            "File contains more than one depth level or "
            "depth could not be established."
        ),

        "both_uo_vo_missing_percent": total_missing,

        "spatial_cells_always_missing": permanently_missing,

        "spatial_cells_partially_missing": partially_missing,

        "mask_assessment": (
            "A large fraction of cells are permanently masked "
            "spatially. This is consistent with geographic "
            "land/coastal masking, but this audit alone cannot "
            "prove that every permanently masked cell is land."
            if permanently_missing is not None
            else
            "Spatial mask classification unavailable."
        ),
    }

    # --------------------------------------------------------
    # Full report
    # --------------------------------------------------------

    report = {
        "dataset": "GlobCurrent",
        "year": year,
        "file": str(path),
        "file_size_mb": file_size_mb,

        "dimensions": {
            str(k): int(v)
            for k, v in ds.sizes.items()
        },

        "detected_variables": {
            "uo": uo_name,
            "vo": vo_name,
            "latitude": lat_name,
            "longitude": lon_name,
            "time": time_name,
        },

        "depth": depth_info,

        "coordinates": coordinates,

        "time": time_info,

        "dimension_consistency": dimension_consistency,

        "uo_metadata": summarize_variable(uo),

        "vo_metadata": summarize_variable(vo),

        "missing_values": mask_info,

        "spatial_mask": spatial_info,

        "uo_statistics": uo_stats,

        "vo_statistics": vo_stats,

        "interpretation": interpretation,
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / (
        f"globecurrent_{year}_audit.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print("\n" + "=" * 80)
    print(f"AUDIT COMPLETE — {year}")
    print("=" * 80)

    print(
        f"\nReport written to:\n{output_file}"
    )

    ds.close()

    return report


# ============================================================
# RUN ALL AUDITS
# ============================================================

def main():

    reports = {}

    for year, path in FILES.items():

        try:

            reports[year] = audit_file(
                year,
                path
            )

        except Exception as exc:

            print("\n" + "!" * 80)
            print(f"AUDIT FAILED — {year}")
            print("!" * 80)

            print(
                f"\nError: {exc}"
            )

    # --------------------------------------------------------
    # Cross-year comparison
    # --------------------------------------------------------

    if len(reports) >= 2:

        print("\n")
        print("=" * 80)
        print("CROSS-YEAR COMPARISON")
        print("=" * 80)

        years = sorted(reports)

        for year in years:

            report = reports[year]

            print(f"\nYEAR {year}")

            print(
                "Dimensions:",
                report["dimensions"]
            )

            print(
                "Depth:",
                report["depth"]["depth_values"]
            )

            print(
                "UO missing:",
                report["missing_values"]["uo"][
                    "missing_percent"
                ],
                "%"
            )

            print(
                "VO missing:",
                report["missing_values"]["vo"][
                    "missing_percent"
                ],
                "%"
            )

            print(
                "Both missing:",
                report["missing_values"]["joint"][
                    "both_missing_percent"
                ],
                "%"
            )

            print(
                "Always-missing spatial cells:",
                report["spatial_mask"].get(
                    "cells_always_missing"
                )
            )

            print(
                "Partially-missing spatial cells:",
                report["spatial_mask"].get(
                    "cells_partially_missing"
                )
            )

        # Compare structural properties
        first = reports[years[0]]

        structural_match = True

        for year in years[1:]:

            current = reports[year]

            checks = [
                current["dimensions"]
                == first["dimensions"],

                current["depth"]["depth_values"]
                == first["depth"]["depth_values"],

                current["dimension_consistency"]
                ["uo_shape"]
                ==
                first["dimension_consistency"]
                ["uo_shape"],

                current["dimension_consistency"]
                ["vo_shape"]
                ==
                first["dimension_consistency"]
                ["vo_shape"],
            ]

            if not all(checks):
                structural_match = False

        print("\nStructural consistency:", end=" ")

        if structural_match:
            print("PASS")
            print(
                "The audited years have matching "
                "dataset structure."
            )
        else:
            print("WARNING")
            print(
                "The audited years do NOT have identical "
                "structure. Investigate before bulk acquisition."
            )

    # --------------------------------------------------------
    # Save combined report
    # --------------------------------------------------------

    combined_file = OUTPUT_DIR / (
        "globecurrent_combined_audit.json"
    )

    with open(
        combined_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            reports,
            f,
            indent=2
        )

    print("\n")
    print("=" * 80)
    print("ALL AUDITS FINISHED")
    print("=" * 80)

    print(
        f"\nCombined report:\n{combined_file}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()