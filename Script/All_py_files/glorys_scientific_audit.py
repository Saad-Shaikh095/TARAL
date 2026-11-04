from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


# ============================================================
# EXACT PATHS FROM YOUR FOLDER STRUCTURE
# ============================================================

RAW_FILE = Path(
    r"E:\Ocean_data\Ocean_extra_data\raw"
    r"\GLORYS_thetao_20120101_20120101.nc.part.nc"
)

GLORYS_FOLDER = Path(
    r"E:\Ocean_data\Ocean_extra_data\raw\glorys"
)

FIRST_PROCESSED_FILE = GLORYS_FOLDER / (
    "GLORYS_thetao_20120101_20120131_NIO_0p25deg_15depth.nc"
)

AUDIT_FOLDER = Path(
    r"E:\Ocean_data\Ocean_extra_data\scientific_audit\GLORYS"
)


# ============================================================
# EXPECTED SCIENTIFIC VALUES
# ============================================================

TEMPERATURE_VARIABLE = "thetao"

EXPECTED_DEPTH_COUNT = 15

# Broad physical range for seawater temperature.
MIN_REASONABLE_TEMPERATURE = -5.0
MAX_REASONABLE_TEMPERATURE = 45.0

# Expected approximate North Indian Ocean region.
EXPECTED_LATITUDE_MIN = 0.0
EXPECTED_LATITUDE_MAX = 30.0

EXPECTED_LONGITUDE_MIN = 40.0
EXPECTED_LONGITUDE_MAX = 110.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    """
    Convert NumPy and other special values into JSON-compatible
    Python values.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None

        if np.isinf(value):
            return str(value)

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        file.write(text)


def find_variable(
    dataset: xr.Dataset,
) -> str | None:
    """
    Find the temperature variable.
    """
    if TEMPERATURE_VARIABLE in dataset.data_vars:
        return TEMPERATURE_VARIABLE

    alternatives = [
        "thetao",
        "theta",
        "temperature",
        "potential_temperature",
        "water_potential_temperature",
    ]

    for name in alternatives:
        if name in dataset.data_vars:
            return name

    return None


def find_coordinate(
    dataset: xr.Dataset,
    possible_names: list[str],
) -> str | None:
    """
    Find coordinates despite common naming differences.
    """
    available_names = list(dataset.coords) + list(dataset.dims)

    normalized_names = {
        name.lower().replace("_", "").replace("-", ""): name
        for name in available_names
    }

    for candidate in possible_names:
        normalized_candidate = (
            candidate.lower()
            .replace("_", "")
            .replace("-", "")
        )

        if normalized_candidate in normalized_names:
            return normalized_names[normalized_candidate]

    return None


def standardize_coordinates(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Rename common coordinate names to:
    latitude, longitude, depth, time
    """
    rename_map = {}

    latitude_name = find_coordinate(
        dataset,
        ["latitude", "lat", "nav_lat"],
    )

    longitude_name = find_coordinate(
        dataset,
        ["longitude", "lon", "nav_lon"],
    )

    depth_name = find_coordinate(
        dataset,
        ["depth", "deptht", "lev", "level"],
    )

    time_name = find_coordinate(
        dataset,
        ["time", "time_counter", "date"],
    )

    if latitude_name and latitude_name != "latitude":
        rename_map[latitude_name] = "latitude"

    if longitude_name and longitude_name != "longitude":
        rename_map[longitude_name] = "longitude"

    if depth_name and depth_name != "depth":
        rename_map[depth_name] = "depth"

    if time_name and time_name != "time":
        rename_map[time_name] = "time"

    if rename_map:
        dataset = dataset.rename(rename_map)

    return dataset


def select_first_day(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Select only the first time record.

    For the raw file, this selects its only/first day.
    For the monthly processed file, this selects 2012-01-01
    and discards the remaining January days.
    """
    if "time" not in dataset.dims:
        return dataset

    if dataset.sizes["time"] == 0:
        raise ValueError(
            "The dataset has an empty time dimension."
        )

    return dataset.isel(time=0)


def get_numeric_values(
    data: xr.DataArray,
) -> np.ndarray:
    values = np.asarray(
        data.values,
        dtype=np.float64,
    )

    return values[np.isfinite(values)]


def get_statistics(
    data: xr.DataArray,
) -> dict[str, Any]:
    values = np.asarray(
        data.values,
        dtype=np.float64,
    )

    finite_values = values[np.isfinite(values)]

    result = {
        "shape": list(values.shape),
        "total_value_count": int(values.size),
        "finite_value_count": int(finite_values.size),
        "nan_value_count": int(
            np.count_nonzero(np.isnan(values))
        ),
    }

    if finite_values.size == 0:
        result["statistics_available"] = False
        return result

    result.update(
        {
            "statistics_available": True,
            "minimum": float(np.min(finite_values)),
            "maximum": float(np.max(finite_values)),
            "mean": float(np.mean(finite_values)),
            "median": float(np.median(finite_values)),
            "standard_deviation": float(
                np.std(finite_values)
            ),
            "physical_range_passed": bool(
                np.min(finite_values)
                >= MIN_REASONABLE_TEMPERATURE
                and np.max(finite_values)
                <= MAX_REASONABLE_TEMPERATURE
            ),
        }
    )

    return result


def coordinate_statistics(
    dataset: xr.Dataset,
    coordinate_name: str,
) -> dict[str, Any]:
    if coordinate_name not in dataset:
        return {
            "exists": False,
            "name": coordinate_name,
        }

    coordinate = dataset[coordinate_name]
    values = np.asarray(coordinate.values)

    result = {
        "exists": True,
        "name": coordinate_name,
        "dimensions": list(coordinate.dims),
        "shape": list(coordinate.shape),
        "dtype": str(coordinate.dtype),
        "count": int(values.size),
        "attributes": dict(coordinate.attrs),
    }

    if values.size > 0:
        numeric_values = values.astype(np.float64)

        result["minimum"] = float(
            np.nanmin(numeric_values)
        )

        result["maximum"] = float(
            np.nanmax(numeric_values)
        )

        result["values"] = numeric_values

    return result


def compare_arrays(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """
    Compare two coordinate arrays.
    """
    first = np.asarray(first)
    second = np.asarray(second)

    result = {
        "first_shape": list(first.shape),
        "second_shape": list(second.shape),
        "same_shape": bool(
            first.shape == second.shape
        ),
    }

    if first.shape != second.shape:
        result["match"] = False
        return result

    differences = np.abs(
        first.astype(np.float64)
        - second.astype(np.float64)
    )

    result.update(
        {
            "maximum_absolute_difference": float(
                np.max(differences)
            ),
            "mean_absolute_difference": float(
                np.mean(differences)
            ),
            "match": bool(
                np.all(differences <= tolerance)
            ),
        }
    )

    return result


# ============================================================
# BASIC AUDIT OF ONE FILE
# ============================================================

def audit_single_file(
    file_path: Path,
    label: str,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "label": label,
        "file": str(file_path),
        "exists": file_path.exists(),
        "opened_successfully": False,
        "errors": [],
    }

    if not file_path.exists():
        result["errors"].append(
            f"File does not exist: {file_path}"
        )
        return result

    try:
        result["file_size_bytes"] = file_path.stat().st_size

        with xr.open_dataset(
            file_path,
            decode_times=True,
            mask_and_scale=True,
        ) as original_dataset:

            dataset = standardize_coordinates(
                original_dataset
            )

            # Only inspect the first day.
            dataset = select_first_day(dataset)

            result["opened_successfully"] = True

            result["dataset_attributes"] = dict(
                dataset.attrs
            )

            result["dimensions"] = {
                name: int(size)
                for name, size in dataset.sizes.items()
            }

            result["coordinates"] = list(dataset.coords)
            result["data_variables"] = list(dataset.data_vars)

            variable_name = find_variable(dataset)

            result["variable_name"] = variable_name

            if variable_name is None:
                result["errors"].append(
                    "Could not find the thetao variable."
                )
                return result

            variable = dataset[variable_name]

            result["variable"] = {
                "name": variable_name,
                "dimensions": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "attributes": dict(variable.attrs),
                "statistics": get_statistics(variable),
            }

            result["coordinates_summary"] = {}

            for coordinate_name in [
                "time",
                "depth",
                "latitude",
                "longitude",
            ]:
                result["coordinates_summary"][
                    coordinate_name
                ] = coordinate_statistics(
                    dataset,
                    coordinate_name,
                )

    except Exception as exc:
        result["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

        result["traceback"] = traceback.format_exc()

    return result


# ============================================================
# RAW DAY ONE VS PROCESSED JANUARY DAY ONE
# ============================================================

def compare_first_day(
    raw_file: Path,
    processed_file: Path,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "raw_file": str(raw_file),
        "processed_file": str(processed_file),
        "comparison_completed": False,
        "errors": [],
    }

    if not raw_file.exists():
        result["errors"].append(
            f"Raw file does not exist: {raw_file}"
        )
        return result

    if not processed_file.exists():
        result["errors"].append(
            f"Processed file does not exist: {processed_file}"
        )
        return result

    try:
        with xr.open_dataset(
            raw_file,
            decode_times=True,
            mask_and_scale=True,
        ) as raw_original, xr.open_dataset(
            processed_file,
            decode_times=True,
            mask_and_scale=True,
        ) as processed_original:

            raw_dataset = standardize_coordinates(
                raw_original
            )

            processed_dataset = standardize_coordinates(
                processed_original
            )

            # IMPORTANT:
            # The raw file is one day.
            # The processed file is one month.
            # We select only the first processed day.
            raw_dataset = select_first_day(
                raw_dataset
            )

            processed_dataset = select_first_day(
                processed_dataset
            )

            raw_variable_name = find_variable(
                raw_dataset
            )

            processed_variable_name = find_variable(
                processed_dataset
            )

            result["raw_variable_name"] = (
                raw_variable_name
            )

            result["processed_variable_name"] = (
                processed_variable_name
            )

            if raw_variable_name is None:
                raise ValueError(
                    "No temperature variable was found "
                    "in the raw file."
                )

            if processed_variable_name is None:
                raise ValueError(
                    "No temperature variable was found "
                    "in the processed file."
                )

            raw_temperature = raw_dataset[
                raw_variable_name
            ]

            processed_temperature = processed_dataset[
                processed_variable_name
            ]

            result["raw_dimensions"] = {
                name: int(size)
                for name, size in raw_dataset.sizes.items()
            }

            result["processed_first_day_dimensions"] = {
                name: int(size)
                for name, size in processed_dataset.sizes.items()
            }

            result["raw_temperature_dimensions"] = list(
                raw_temperature.dims
            )

            result[
                "processed_temperature_dimensions"
            ] = list(
                processed_temperature.dims
            )

            # ------------------------------------------------
            # Check that required coordinates exist
            # ------------------------------------------------

            required_coordinates = [
                "latitude",
                "longitude",
                "depth",
            ]

            missing_raw_coordinates = [
                name
                for name in required_coordinates
                if name not in raw_dataset
            ]

            missing_processed_coordinates = [
                name
                for name in required_coordinates
                if name not in processed_dataset
            ]

            result["missing_raw_coordinates"] = (
                missing_raw_coordinates
            )

            result["missing_processed_coordinates"] = (
                missing_processed_coordinates
            )

            if missing_raw_coordinates:
                raise ValueError(
                    "Raw file is missing coordinates: "
                    + ", ".join(missing_raw_coordinates)
                )

            if missing_processed_coordinates:
                raise ValueError(
                    "Processed file is missing coordinates: "
                    + ", ".join(
                        missing_processed_coordinates
                    )
                )

            # ------------------------------------------------
            # Extract coordinates
            # ------------------------------------------------

            raw_latitude = np.asarray(
                raw_dataset["latitude"].values
            )

            processed_latitude = np.asarray(
                processed_dataset["latitude"].values
            )

            raw_longitude = np.asarray(
                raw_dataset["longitude"].values
            )

            processed_longitude = np.asarray(
                processed_dataset["longitude"].values
            )

            raw_depth = np.asarray(
                raw_dataset["depth"].values
            )

            processed_depth = np.asarray(
                processed_dataset["depth"].values
            )

            result["coordinate_comparisons"] = {
                "latitude": compare_arrays(
                    raw_latitude,
                    processed_latitude,
                ),
                "longitude": compare_arrays(
                    raw_longitude,
                    processed_longitude,
                ),
                "depth": compare_arrays(
                    raw_depth,
                    processed_depth,
                ),
            }

            # ------------------------------------------------
            # Geographic checks on processed day one
            # ------------------------------------------------

            processed_latitude_min = float(
                np.nanmin(processed_latitude)
            )

            processed_latitude_max = float(
                np.nanmax(processed_latitude)
            )

            processed_longitude_min = float(
                np.nanmin(processed_longitude)
            )

            processed_longitude_max = float(
                np.nanmax(processed_longitude)
            )

            result["geographic_validation"] = {
                "processed_latitude_minimum": (
                    processed_latitude_min
                ),
                "processed_latitude_maximum": (
                    processed_latitude_max
                ),
                "processed_longitude_minimum": (
                    processed_longitude_min
                ),
                "processed_longitude_maximum": (
                    processed_longitude_max
                ),
                "latitude_inside_expected_region": bool(
                    processed_latitude_min
                    >= EXPECTED_LATITUDE_MIN
                    and processed_latitude_max
                    <= EXPECTED_LATITUDE_MAX
                ),
                "longitude_inside_expected_region": bool(
                    processed_longitude_min
                    >= EXPECTED_LONGITUDE_MIN
                    and processed_longitude_max
                    <= EXPECTED_LONGITUDE_MAX
                ),
            }

            # ------------------------------------------------
            # Depth checks
            # ------------------------------------------------

            result["depth_validation"] = {
                "raw_depth_count": int(
                    raw_depth.size
                ),
                "processed_depth_count": int(
                    processed_depth.size
                ),
                "expected_depth_count": (
                    EXPECTED_DEPTH_COUNT
                ),
                "processed_depth_count_correct": bool(
                    processed_depth.size
                    == EXPECTED_DEPTH_COUNT
                ),
                "depth_values_match": bool(
                    raw_depth.shape
                    == processed_depth.shape
                    and np.allclose(
                        raw_depth,
                        processed_depth,
                        rtol=0.0,
                        atol=1e-6,
                    )
                ),
            }

            # ------------------------------------------------
            # Grid spacing
            # ------------------------------------------------

            def median_spacing(
                values: np.ndarray,
            ) -> float | None:
                if values.size < 2:
                    return None

                return float(
                    np.median(
                        np.abs(np.diff(values))
                    )
                )

            result["grid_spacing"] = {
                "raw_latitude_spacing": median_spacing(
                    raw_latitude
                ),
                "processed_latitude_spacing": (
                    median_spacing(processed_latitude)
                ),
                "raw_longitude_spacing": median_spacing(
                    raw_longitude
                ),
                "processed_longitude_spacing": (
                    median_spacing(processed_longitude)
                ),
            }

            # ------------------------------------------------
            # Temperature comparison
            # ------------------------------------------------

            # First remove time if it still exists.
            if "time" in raw_temperature.dims:
                raw_temperature = raw_temperature.isel(
                    time=0
                )

            if "time" in processed_temperature.dims:
                processed_temperature = (
                    processed_temperature.isel(time=0)
                )

            # The processed file may be on a different grid.
            # If grids differ, interpolate raw day-one data
            # onto the processed grid.
            grids_are_identical = (
                raw_latitude.shape
                == processed_latitude.shape
                and raw_longitude.shape
                == processed_longitude.shape
                and np.allclose(
                    raw_latitude,
                    processed_latitude,
                    rtol=0.0,
                    atol=1e-6,
                )
                and np.allclose(
                    raw_longitude,
                    processed_longitude,
                    rtol=0.0,
                    atol=1e-6,
                )
            )

            result["grids_are_identical"] = (
                grids_are_identical
            )

            if grids_are_identical:
                raw_for_comparison = raw_temperature

            else:
                if "latitude" not in raw_temperature.dims:
                    raise ValueError(
                        "Raw latitude is not a dimension. "
                        "Cannot interpolate raw data."
                    )

                if "longitude" not in raw_temperature.dims:
                    raise ValueError(
                        "Raw longitude is not a dimension. "
                        "Cannot interpolate raw data."
                    )

                raw_for_comparison = (
                    raw_temperature.interp(
                        latitude=processed_latitude,
                        longitude=processed_longitude,
                        method="nearest",
                    )
                )

            # Match raw depth levels to processed depth levels.
            if "depth" in raw_for_comparison.dims:
                raw_for_comparison = (
                    raw_for_comparison.sel(
                        depth=xr.DataArray(
                            processed_depth,
                            dims="depth",
                        ),
                        method="nearest",
                    )
                )

            # Align the arrays before comparing.
            processed_aligned, raw_aligned = xr.align(
                processed_temperature,
                raw_for_comparison,
                join="inner",
            )

            processed_values = np.asarray(
                processed_aligned.values,
                dtype=np.float64,
            )

            raw_values = np.asarray(
                raw_aligned.values,
                dtype=np.float64,
            )

            if processed_values.shape != raw_values.shape:
                raise ValueError(
                    "Comparison arrays have different shapes: "
                    f"raw={raw_values.shape}, "
                    f"processed={processed_values.shape}"
                )

            valid_mask = (
                np.isfinite(processed_values)
                & np.isfinite(raw_values)
            )

            valid_processed = processed_values[
                valid_mask
            ]

            valid_raw = raw_values[
                valid_mask
            ]

            if valid_processed.size == 0:
                raise ValueError(
                    "No valid overlapping values were found."
                )

            differences = (
                valid_processed - valid_raw
            )

            absolute_differences = np.abs(
                differences
            )

            result["temperature_comparison"] = {
                "comparison_shape": list(
                    processed_values.shape
                ),
                "valid_value_count": int(
                    valid_processed.size
                ),
                "raw_minimum": float(
                    np.min(valid_raw)
                ),
                "raw_maximum": float(
                    np.max(valid_raw)
                ),
                "processed_minimum": float(
                    np.min(valid_processed)
                ),
                "processed_maximum": float(
                    np.max(valid_processed)
                ),
                "raw_mean": float(
                    np.mean(valid_raw)
                ),
                "processed_mean": float(
                    np.mean(valid_processed)
                ),
                "mean_difference": float(
                    np.mean(differences)
                ),
                "mean_absolute_difference": float(
                    np.mean(absolute_differences)
                ),
                "median_absolute_difference": float(
                    np.median(absolute_differences)
                ),
                "maximum_absolute_difference": float(
                    np.max(absolute_differences)
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(differences ** 2)
                    )
                ),
                "matching_fraction_within_1e-6": float(
                    np.mean(
                        np.isclose(
                            valid_processed,
                            valid_raw,
                            rtol=0.0,
                            atol=1e-6,
                        )
                    )
                ),
            }

            # ------------------------------------------------
            # Scientific validation
            # ------------------------------------------------

            processed_statistics = get_statistics(
                processed_temperature
            )

            result["scientific_validation"] = {
                "temperature_physical_range_passed": (
                    processed_statistics.get(
                        "physical_range_passed"
                    )
                ),
                "depth_count_passed": bool(
                    processed_depth.size
                    == EXPECTED_DEPTH_COUNT
                ),
                "latitude_region_passed": (
                    result["geographic_validation"][
                        "latitude_inside_expected_region"
                    ]
                ),
                "longitude_region_passed": (
                    result["geographic_validation"][
                        "longitude_inside_expected_region"
                    ]
                ),
                "depth_values_match": (
                    result["depth_validation"][
                        "depth_values_match"
                    ]
                ),
            }

            result["comparison_completed"] = True

    except Exception as exc:
        result["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

        result["traceback"] = traceback.format_exc()

    return result


# ============================================================
# TEXT REPORT
# ============================================================

def create_text_report(
    raw_audit: dict[str, Any],
    processed_audit: dict[str, Any],
    comparison: dict[str, Any],
) -> str:

    lines = []

    lines.append(
        "GLORYS FIRST-DAY SCIENTIFIC VALIDATION REPORT"
    )
    lines.append("=" * 80)
    lines.append(
        datetime.now().strftime(
            "Generated: %Y-%m-%d %H:%M:%S"
        )
    )
    lines.append("")

    lines.append("FILES USED")
    lines.append("-" * 80)
    lines.append(f"Raw file:")
    lines.append(f"  {RAW_FILE}")
    lines.append("")
    lines.append("Processed file:")
    lines.append(f"  {FIRST_PROCESSED_FILE}")
    lines.append("")
    lines.append(
        "Comparison scope:"
    )
    lines.append(
        "  Raw 2012-01-01 versus only the first day "
        "of the January 2012 processed file"
    )
    lines.append("")

    lines.append("OPENING CHECKS")
    lines.append("-" * 80)
    lines.append(
        f"Raw file opened: "
        f"{raw_audit.get('opened_successfully')}"
    )
    lines.append(
        f"Processed first day opened: "
        f"{processed_audit.get('opened_successfully')}"
    )
    lines.append(
        f"Comparison completed: "
        f"{comparison.get('comparison_completed')}"
    )
    lines.append("")

    lines.append("DIMENSIONS")
    lines.append("-" * 80)
    lines.append(
        f"Raw dimensions: "
        f"{comparison.get('raw_dimensions')}"
    )
    lines.append(
        f"Processed first-day dimensions: "
        f"{comparison.get('processed_first_day_dimensions')}"
    )
    lines.append("")

    lines.append("DEPTH VALIDATION")
    lines.append("-" * 80)

    depth = comparison.get(
        "depth_validation",
        {},
    )

    lines.append(
        f"Raw depth count: "
        f"{depth.get('raw_depth_count')}"
    )

    lines.append(
        f"Processed depth count: "
        f"{depth.get('processed_depth_count')}"
    )

    lines.append(
        f"Expected depth count: "
        f"{depth.get('expected_depth_count')}"
    )

    lines.append(
        f"Depth count correct: "
        f"{depth.get('processed_depth_count_correct')}"
    )

    lines.append(
        f"Depth values match: "
        f"{depth.get('depth_values_match')}"
    )

    lines.append("")

    lines.append("GRID VALIDATION")
    lines.append("-" * 80)

    lines.append(
        f"Grids identical: "
        f"{comparison.get('grids_are_identical')}"
    )

    coordinate_comparisons = comparison.get(
        "coordinate_comparisons",
        {},
    )

    for coordinate_name in [
        "latitude",
        "longitude",
        "depth",
    ]:
        coordinate_result = coordinate_comparisons.get(
            coordinate_name,
            {},
        )

        lines.append(
            f"{coordinate_name} match: "
            f"{coordinate_result.get('match')}"
        )

    lines.append("")

    lines.append("TEMPERATURE COMPARISON")
    lines.append("-" * 80)

    temperature = comparison.get(
        "temperature_comparison",
        {},
    )

    lines.append(
        f"Comparison shape: "
        f"{temperature.get('comparison_shape')}"
    )

    lines.append(
        f"Valid values compared: "
        f"{temperature.get('valid_value_count')}"
    )

    lines.append(
        f"Raw mean: "
        f"{temperature.get('raw_mean')}"
    )

    lines.append(
        f"Processed mean: "
        f"{temperature.get('processed_mean')}"
    )

    lines.append(
        f"Mean absolute difference: "
        f"{temperature.get('mean_absolute_difference')}"
    )

    lines.append(
        f"Maximum absolute difference: "
        f"{temperature.get('maximum_absolute_difference')}"
    )

    lines.append(
        f"RMSE: "
        f"{temperature.get('rmse')}"
    )

    lines.append(
        f"Matching fraction within 1e-6: "
        f"{temperature.get('matching_fraction_within_1e-6')}"
    )

    lines.append("")

    lines.append("SCIENTIFIC VALIDATION")
    lines.append("-" * 80)

    scientific = comparison.get(
        "scientific_validation",
        {},
    )

    for name, value in scientific.items():
        lines.append(
            f"{name}: {value}"
        )

    lines.append("")

    lines.append("ERRORS")
    lines.append("-" * 80)

    errors = (
        raw_audit.get("errors", [])
        + processed_audit.get("errors", [])
        + comparison.get("errors", [])
    )

    if errors:
        for error in errors:
            lines.append(error)
    else:
        lines.append("No errors reported.")

    lines.append("")
    lines.append("IMPORTANT SCOPE NOTE")
    lines.append("-" * 80)
    lines.append(
        "Only 2012-01-01 was compared."
    )
    lines.append(
        "The rest of January 2012 was not compared."
    )
    lines.append(
        "The other monthly GLORYS files were not compared."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    AUDIT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("GLORYS FIRST-DAY SCIENTIFIC VALIDATION")
    print("=" * 80)

    print("Raw file:")
    print(f"  {RAW_FILE}")

    print("Processed file:")
    print(f"  {FIRST_PROCESSED_FILE}")

    print()

    # --------------------------------------------------------
    # Verify exact paths
    # --------------------------------------------------------

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw file was not found:\n{RAW_FILE}"
        )

    if not GLORYS_FOLDER.exists():
        raise FileNotFoundError(
            f"GLORYS folder was not found:\n{GLORYS_FOLDER}"
        )

    if not FIRST_PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "The first processed January file was not found:\n"
            f"{FIRST_PROCESSED_FILE}"
        )

    # --------------------------------------------------------
    # Audit raw one-day file
    # --------------------------------------------------------

    print("Auditing raw 2012-01-01 file...")

    raw_audit = audit_single_file(
        file_path=RAW_FILE,
        label="raw_2012_01_01",
    )

    save_json(
        raw_audit,
        AUDIT_FOLDER / "01_raw_20120101_audit.json",
    )

    # --------------------------------------------------------
    # Audit only day one of January processed file
    # --------------------------------------------------------

    print(
        "Auditing only 2012-01-01 from January file..."
    )

    processed_audit = audit_single_file(
        file_path=FIRST_PROCESSED_FILE,
        label="processed_2012_01_01",
    )

    save_json(
        processed_audit,
        AUDIT_FOLDER
        / "02_processed_20120101_audit.json",
    )

    # --------------------------------------------------------
    # Compare the two day-one datasets
    # --------------------------------------------------------

    print(
        "Comparing raw 2012-01-01 with processed "
        "2012-01-01..."
    )

    comparison = compare_first_day(
        raw_file=RAW_FILE,
        processed_file=FIRST_PROCESSED_FILE,
    )

    save_json(
        comparison,
        AUDIT_FOLDER
        / "03_raw_vs_processed_20120101_comparison.json",
    )

    # --------------------------------------------------------
    # Combined JSON report
    # --------------------------------------------------------

    combined_report = {
        "generated": datetime.now().isoformat(),
        "paths": {
            "raw_file": str(RAW_FILE),
            "glorys_folder": str(GLORYS_FOLDER),
            "first_processed_file": str(
                FIRST_PROCESSED_FILE
            ),
            "audit_folder": str(AUDIT_FOLDER),
        },
        "comparison_scope": {
            "raw_day": "2012-01-01",
            "processed_day": "2012-01-01",
            "only_first_processed_day": True,
            "full_january_compared": False,
            "other_months_compared": False,
        },
        "raw_audit": raw_audit,
        "processed_first_day_audit": processed_audit,
        "comparison": comparison,
    }

    save_json(
        combined_report,
        AUDIT_FOLDER
        / "00_combined_first_day_validation.json",
    )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    text_report = create_text_report(
        raw_audit=raw_audit,
        processed_audit=processed_audit,
        comparison=comparison,
    )

    save_text(
        text_report,
        AUDIT_FOLDER
        / "GLORYS_first_day_validation_report.txt",
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION FINISHED")
    print("=" * 80)

    print(
        "Comparison performed:"
    )
    print(
        "  Raw 2012-01-01"
    )
    print(
        "  versus processed 2012-01-01"
    )

    print()
    print("Reports saved in:")
    print(f"  {AUDIT_FOLDER}")

    print()
    print("Main report:")
    print(
        AUDIT_FOLDER
        / "GLORYS_first_day_validation_report.txt"
    )

    print()
    print("Combined JSON report:")
    print(
        AUDIT_FOLDER
        / "00_combined_first_day_validation.json"
    )


if __name__ == "__main__":
    main()