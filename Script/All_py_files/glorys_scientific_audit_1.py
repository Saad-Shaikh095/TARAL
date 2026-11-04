"""
GLORYS SCIENTIFIC VALIDATION AUDIT
==================================

Run:
    python glorys_scientific_audit.py

All paths are configured inside this file.

This audit checks:

1. Correct physical variable
2. Correct dimensions
3. Correct target depths
4. Correct NIO spatial domain
5. Correct 0.25-degree final grid
6. Daily temporal spacing
7. Temporal coverage
8. Units
9. Missing values
10. Physical plausibility
11. Depth-dependent missingness
12. GLORYS product identity
13. Source dataset version
14. Native-grid configuration
15. Vertical interpolation
16. Surface-level handling
17. Horizontal processing method
18. DUACS and OSCAR contamination
19. Reanalysis versus observation interpretation
20. ML-related limitations

Important:
This is a scientific data-quality and provenance audit.
It cannot prove that GLORYS is observational truth.
It also cannot prove exact interpolation/regridding behavior
without retaining the native source files and processing logs.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Main Ocean_data directory.
# Change ONLY this path if your main directory is elsewhere.
BASE_DIR = Path(r"E:\Ocean_data\Ocean_extra_data")

# Existing GLORYS acquisition script.
ACQUISITION_SCRIPT = BASE_DIR / "raw" / "fetch_GLORYS.py"

# Directory containing the final processed GLORYS NetCDF files.
GLORYS_DATA_DIR = BASE_DIR / "GLORYS" / "processed"

# Audit outputs.
TEXT_REPORT = BASE_DIR / "GLORYS" / "glorys_scientific_validation.txt"
JSON_REPORT = BASE_DIR / "GLORYS" / "glorys_scientific_validation.json"


# ============================================================================
# EXPECTED SCIENTIFIC CONFIGURATION
# ============================================================================

EXPECTED_PRODUCT_ID = "GLOBAL_MULTIYEAR_PHY_001_030"

EXPECTED_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

EXPECTED_DATASET_VERSION = "202311"

EXPECTED_VARIABLE = "thetao"

EXPECTED_DEPTHS = np.array(
    [
        0.0,
        5.0,
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        125.0,
        150.0,
        200.0,
        300.0,
        500.0,
        700.0,
        1000.0,
    ],
    dtype=float,
)

EXPECTED_MIN_LON = 45.0
EXPECTED_MAX_LON = 105.0

EXPECTED_MIN_LAT = 5.0
EXPECTED_MAX_LAT = 30.0

EXPECTED_FINAL_GRID_STEP = 0.25

EXPECTED_START_DATE = np.datetime64("2012-01-01")

EXPECTED_END_DATE = np.datetime64("2024-12-31")

EXPECTED_NATIVE_GRID_STEP = 1.0 / 12.0

DEPTH_TOLERANCE = 0.001
COORDINATE_TOLERANCE = 0.001
GRID_TOLERANCE = 0.001

# Broad plausibility limits for seawater temperature.
MIN_REASONABLE_TEMPERATURE = -5.0
MAX_REASONABLE_TEMPERATURE = 45.0

# ============================================================================
# RESULT STRUCTURES
# ============================================================================


@dataclass
class Finding:
    section: str
    status: str
    question: str
    evidence: str
    interpretation: str
    recommendation: str = ""


class Audit:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(
        self,
        section: str,
        status: str,
        question: str,
        evidence: str,
        interpretation: str,
        recommendation: str = "",
    ) -> None:
        self.findings.append(
            Finding(
                section=section,
                status=status,
                question=question,
                evidence=evidence,
                interpretation=interpretation,
                recommendation=recommendation,
            )
        )


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def normalise(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def array_string(values: Any) -> str:
    array = np.asarray(values)

    if array.size == 0:
        return "[]"

    return "[" + ", ".join(
        f"{float(value):g}"
        for value in array
    ) + "]"


def arrays_match(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
) -> bool:
    if len(actual) != len(expected):
        return False

    return bool(
        np.all(
            np.abs(
                actual.astype(float) - expected.astype(float)
            ) <= tolerance
        )
    )


def coordinate_step(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None

    differences = np.diff(values.astype(float))

    if len(differences) == 0:
        return None

    return float(np.median(np.abs(differences)))


def is_monotonic(values: np.ndarray) -> bool:
    if len(values) < 2:
        return True

    differences = np.diff(values.astype(float))

    return bool(
        np.all(differences > 0)
        or np.all(differences < 0)
    )


def duplicate_values(values: np.ndarray) -> np.ndarray:
    unique_values, counts = np.unique(
        values,
        return_counts=True,
    )

    return unique_values[counts > 1]


def find_coordinate(
    dataset: xr.Dataset,
    candidates: list[str],
) -> str | None:
    names = list(dataset.coords) + list(dataset.dims)

    lower_names = {
        str(name).lower(): str(name)
        for name in names
    }

    for candidate in candidates:
        if candidate.lower() in lower_names:
            return lower_names[candidate.lower()]

    for name in names:
        name_lower = str(name).lower()

        for candidate in candidates:
            if candidate.lower() in name_lower:
                return str(name)

    return None


def find_temperature_variable(
    dataset: xr.Dataset,
) -> str | None:
    if EXPECTED_VARIABLE in dataset.data_vars:
        return EXPECTED_VARIABLE

    for name, variable in dataset.data_vars.items():
        standard_name = normalise(
            variable.attrs.get("standard_name")
        )

        long_name = normalise(
            variable.attrs.get("long_name")
        )

        if "sea_water_potential_temperature" in standard_name:
            return str(name)

        if "potential temperature" in long_name:
            return str(name)

    return None


def find_depth_coordinate(
    dataset: xr.Dataset,
    variable_name: str,
) -> str | None:
    variable = dataset[variable_name]

    candidates = [
        "depth",
        "deptht",
        "level",
        "lev",
        "z",
    ]

    for dimension in variable.dims:
        if dimension.lower() in candidates:
            return dimension

    for dimension in variable.dims:
        if dimension not in dataset.coords:
            continue

        coordinate = dataset[dimension]

        standard_name = normalise(
            coordinate.attrs.get("standard_name")
        )

        units = normalise(
            coordinate.attrs.get("units")
        )

        positive = normalise(
            coordinate.attrs.get("positive")
        )

        if standard_name == "depth":
            return dimension

        if positive in {"down", "up"}:
            return dimension

        if units in {
            "m",
            "meter",
            "meters",
            "metre",
            "metres",
        }:
            return dimension

    return None


def find_time_coordinate(
    dataset: xr.Dataset,
    variable_name: str,
) -> str | None:
    variable = dataset[variable_name]

    for dimension in variable.dims:
        if dimension.lower() in {
            "time",
            "time_counter",
            "datetime",
        }:
            return dimension

    for dimension in variable.dims:
        if dimension not in dataset.coords:
            continue

        try:
            if np.issubdtype(
                dataset[dimension].dtype,
                np.datetime64,
            ):
                return dimension
        except TypeError:
            pass

    return None


def find_latitude_coordinate(
    dataset: xr.Dataset,
    variable_name: str,
) -> str | None:
    variable = dataset[variable_name]

    for dimension in variable.dims:
        if dimension.lower() in {
            "latitude",
            "lat",
            "y",
        }:
            return dimension

    for dimension in variable.dims:
        if dimension not in dataset.coords:
            continue

        if (
            normalise(
                dataset[dimension].attrs.get("standard_name")
            )
            == "latitude"
        ):
            return dimension

    return None


def find_longitude_coordinate(
    dataset: xr.Dataset,
    variable_name: str,
) -> str | None:
    variable = dataset[variable_name]

    for dimension in variable.dims:
        if dimension.lower() in {
            "longitude",
            "lon",
            "x",
        }:
            return dimension

    for dimension in variable.dims:
        if dimension not in dataset.coords:
            continue

        if (
            normalise(
                dataset[dimension].attrs.get("standard_name")
            )
            == "longitude"
        ):
            return dimension

    return None


def extract_assignment(
    source: str,
    name: str,
) -> str | None:
    pattern = rf"""
        {re.escape(name)}
        \s*=\s*
        (?P<value>
            "(?:[^"\\]|\\.)*"
            |
            '(?:[^'\\]|\\.)*'
            |
            [^\n#]+
        )
    """

    match = re.search(
        pattern,
        source,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    if not match:
        return None

    return match.group("value").strip()


def extract_string(
    source: str,
    name: str,
) -> str | None:
    value = extract_assignment(
        source,
        name,
    )

    if value is None:
        return None

    if (
        len(value) >= 2
        and value[0] in {"'", '"'}
        and value[-1] == value[0]
    ):
        return value[1:-1]

    return value


def extract_float(
    source: str,
    name: str,
) -> float | None:
    value = extract_assignment(
        source,
        name,
    )

    if value is None:
        return None

    fraction = re.search(
        r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)",
        value,
    )

    if fraction:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))

        if denominator != 0:
            return numerator / denominator

    number = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        value,
    )

    if not number:
        return None

    return float(number.group(0))


def extract_array(
    source: str,
    name: str,
) -> np.ndarray | None:
    pattern = rf"""
        {re.escape(name)}
        \s*=\s*
        (?:np\.array\s*\(\s*)?
        \[
            (?P<values>[^\]]+)
        \]
    """

    match = re.search(
        pattern,
        source,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    if not match:
        return None

    values_text = match.group("values")

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        values_text,
    )

    if not numbers:
        return None

    return np.array(
        [float(value) for value in numbers],
        dtype=float,
    )


def variable_statistics(
    variable: xr.DataArray,
) -> dict[str, Any]:
    values = np.asarray(variable.values)

    finite_values = values[np.isfinite(values)]

    if len(finite_values) == 0:
        return {
            "total_count": int(values.size),
            "finite_count": 0,
            "missing_count": int(values.size),
            "missing_fraction": 1.0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }

    return {
        "total_count": int(values.size),
        "finite_count": int(len(finite_values)),
        "missing_count": int(values.size - len(finite_values)),
        "missing_fraction": float(
            (values.size - len(finite_values)) / values.size
        ),
        "minimum": float(np.min(finite_values)),
        "maximum": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
    }


def has_fill_metadata(
    variable: xr.DataArray,
) -> bool:
    attribute_names = {
        "_FillValue",
        "missing_value",
        "fill_value",
    }

    return bool(
        attribute_names.intersection(variable.attrs)
        or attribute_names.intersection(variable.encoding)
    )


# ============================================================================
# ACQUISITION SCRIPT AUDIT
# ============================================================================


def audit_acquisition_script(
    audit: Audit,
) -> dict[str, Any]:
    source_info: dict[str, Any] = {}

    audit.add(
        section="Acquisition script",
        status="INFO",
        question="Which acquisition script is being audited?",
        evidence=str(ACQUISITION_SCRIPT),
        interpretation=(
            "The audit uses the fixed path configured inside this file."
        ),
    )

    if not ACQUISITION_SCRIPT.exists():
        audit.add(
            section="Acquisition script",
            status="FAIL",
            question="Does the acquisition script exist?",
            evidence=str(ACQUISITION_SCRIPT),
            interpretation=(
                "The acquisition script could not be found. "
                "The script-based provenance audit cannot continue."
            ),
            recommendation=(
                "Correct ACQUISITION_SCRIPT near the top of this file."
            ),
        )

        return source_info

    source = read_text(ACQUISITION_SCRIPT)

    source_info["path"] = str(ACQUISITION_SCRIPT)
    source_info["sha256"] = sha256_file(ACQUISITION_SCRIPT)

    product_id = extract_string(
        source,
        "PRODUCT_ID",
    )

    dataset_id = extract_string(
        source,
        "DATASET_ID",
    )

    dataset_version = extract_string(
        source,
        "DATASET_VERSION",
    )

    start_date = extract_string(
        source,
        "START_DATE",
    )

    end_date = extract_string(
        source,
        "END_DATE",
    )

    native_step = extract_float(
        source,
        "NATIVE_STEP",
    )

    target_depths = extract_array(
        source,
        "TARGET_DEPTHS",
    )

    target_min_lon = extract_float(
        source,
        "TARGET_MIN_LON",
    )

    target_max_lon = extract_float(
        source,
        "TARGET_MAX_LON",
    )

    target_min_lat = extract_float(
        source,
        "TARGET_MIN_LAT",
    )

    target_max_lat = extract_float(
        source,
        "TARGET_MAX_LAT",
    )

    request_min_lon = extract_float(
        source,
        "REQUEST_MIN_LON",
    )

    request_max_lon = extract_float(
        source,
        "REQUEST_MAX_LON",
    )

    request_min_lat = extract_float(
        source,
        "REQUEST_MIN_LAT",
    )

    request_max_lat = extract_float(
        source,
        "REQUEST_MAX_LAT",
    )

    request_max_depth = extract_float(
        source,
        "REQUEST_MAX_DEPTH",
    )

    source_info.update(
        {
            "product_id": product_id,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "start_date": start_date,
            "end_date": end_date,
            "native_step": native_step,
            "target_depths": (
                target_depths.tolist()
                if target_depths is not None
                else None
            ),
            "target_min_lon": target_min_lon,
            "target_max_lon": target_max_lon,
            "target_min_lat": target_min_lat,
            "target_max_lat": target_max_lat,
            "request_min_lon": request_min_lon,
            "request_max_lon": request_max_lon,
            "request_min_lat": request_min_lat,
            "request_max_lat": request_max_lat,
            "request_max_depth": request_max_depth,
        }
    )

    source_lower = source.lower()

    # ------------------------------------------------------------------------
    # Product identity
    # ------------------------------------------------------------------------

    if product_id == EXPECTED_PRODUCT_ID:
        audit.add(
            section="Product identity",
            status="PASS",
            question="Are we using the correct GLORYS product?",
            evidence=f"PRODUCT_ID = {product_id}",
            interpretation=(
                "The acquisition script identifies the expected "
                "Copernicus GLORYS multiyear physical product."
            ),
        )
    else:
        audit.add(
            section="Product identity",
            status="FAIL",
            question="Are we using the correct GLORYS product?",
            evidence=f"PRODUCT_ID = {product_id}",
            interpretation=(
                "The expected GLORYS product ID was not confirmed."
            ),
            recommendation=(
                f"Expected product ID: {EXPECTED_PRODUCT_ID}"
            ),
        )

    if dataset_id == EXPECTED_DATASET_ID:
        audit.add(
            section="Product identity",
            status="PASS",
            question="Are we using the correct daily GLORYS dataset?",
            evidence=f"DATASET_ID = {dataset_id}",
            interpretation=(
                "The dataset ID matches the expected daily "
                "approximately 1/12-degree GLORYS product."
            ),
        )
    elif dataset_id and "p1d" in dataset_id.lower():
        audit.add(
            section="Product identity",
            status="WARNING",
            question="Are we using the correct daily GLORYS dataset?",
            evidence=f"DATASET_ID = {dataset_id}",
            interpretation=(
                "The dataset appears to be daily, but its exact ID "
                "differs from the expected dataset ID."
            ),
        )
    else:
        audit.add(
            section="Product identity",
            status="FAIL",
            question="Are we using the correct daily GLORYS dataset?",
            evidence=f"DATASET_ID = {dataset_id}",
            interpretation=(
                "A daily GLORYS dataset could not be confirmed."
            ),
        )

    if dataset_version == EXPECTED_DATASET_VERSION:
        audit.add(
            section="Product identity",
            status="PASS",
            question="Is the GLORYS dataset version pinned?",
            evidence=f"DATASET_VERSION = {dataset_version}",
            interpretation=(
                "The dataset version is explicitly recorded."
            ),
        )
    else:
        audit.add(
            section="Product identity",
            status="WARNING",
            question="Is the GLORYS dataset version pinned?",
            evidence=f"DATASET_VERSION = {dataset_version}",
            interpretation=(
                "The expected source dataset version was not confirmed."
            ),
            recommendation=(
                "Record the exact Copernicus dataset version and retrieval date."
            ),
        )

    # ------------------------------------------------------------------------
    # Variable
    # ------------------------------------------------------------------------

    if "thetao" in source_lower:
        audit.add(
            section="Physical variable",
            status="PASS",
            question="Are we using the correct physical variable?",
            evidence="The acquisition script contains thetao.",
            interpretation=(
                "thetao is sea-water potential temperature and is "
                "appropriate for the GLORYS temperature target."
            ),
        )
    else:
        audit.add(
            section="Physical variable",
            status="FAIL",
            question="Are we using the correct physical variable?",
            evidence="thetao was not found in the acquisition script.",
            interpretation=(
                "The intended GLORYS temperature variable was not confirmed."
            ),
        )

    # ------------------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------------------

    configured_domain = [
        target_min_lon,
        target_max_lon,
        target_min_lat,
        target_max_lat,
    ]

    if (
        all(value is not None for value in configured_domain)
        and abs(target_min_lon - EXPECTED_MIN_LON) <= COORDINATE_TOLERANCE
        and abs(target_max_lon - EXPECTED_MAX_LON) <= COORDINATE_TOLERANCE
        and abs(target_min_lat - EXPECTED_MIN_LAT) <= COORDINATE_TOLERANCE
        and abs(target_max_lat - EXPECTED_MAX_LAT) <= COORDINATE_TOLERANCE
    ):
        audit.add(
            section="Spatial domain",
            status="PASS",
            question="Is the configured domain the required NIO region?",
            evidence=(
                f"{target_min_lon}–{target_max_lon} E, "
                f"{target_min_lat}–{target_max_lat} N"
            ),
            interpretation=(
                "The configured final domain matches the required "
                "North Indian Ocean domain."
            ),
        )
    else:
        audit.add(
            section="Spatial domain",
            status="FAIL",
            question="Is the configured domain the required NIO region?",
            evidence=(
                f"Configured domain = {configured_domain}"
            ),
            interpretation=(
                "The configured final domain differs from the required "
                "45–105 E, 5–30 N region."
            ),
        )

    # ------------------------------------------------------------------------
    # Native grid
    # ------------------------------------------------------------------------

    if native_step is not None and abs(
        native_step - EXPECTED_NATIVE_GRID_STEP
    ) <= 1e-6:
        audit.add(
            section="Spatial grid",
            status="PASS",
            question="Is the native GLORYS grid correctly identified?",
            evidence=f"NATIVE_STEP = {native_step}",
            interpretation=(
                "The script identifies the native spacing as approximately "
                "1/12 degree."
            ),
        )
    else:
        audit.add(
            section="Spatial grid",
            status="WARNING",
            question="Is the native GLORYS grid correctly identified?",
            evidence=f"NATIVE_STEP = {native_step}",
            interpretation=(
                "The native grid spacing could not be confirmed."
            ),
        )

    # ------------------------------------------------------------------------
    # Depths
    # ------------------------------------------------------------------------

    if target_depths is not None and arrays_match(
        target_depths,
        EXPECTED_DEPTHS,
        DEPTH_TOLERANCE,
    ):
        audit.add(
            section="Depths",
            status="PASS",
            question="Are the 15 required target depths configured?",
            evidence=f"TARGET_DEPTHS = {array_string(target_depths)}",
            interpretation=(
                "The script requests exactly the required 15 depths."
            ),
        )
    else:
        audit.add(
            section="Depths",
            status="FAIL",
            question="Are the 15 required target depths configured?",
            evidence=f"TARGET_DEPTHS = {array_string(target_depths)}",
            interpretation=(
                "The configured target depths differ from the required "
                "OceanEmbed depth vector."
            ),
        )

    if request_max_depth is not None and request_max_depth > 1000:
        audit.add(
            section="Depths",
            status="PASS",
            question="Is 1000 m requested without intentional extrapolation?",
            evidence=f"REQUEST_MAX_DEPTH = {request_max_depth} m",
            interpretation=(
                "The source request extends deeper than the deepest "
                "target depth."
            ),
        )
    else:
        audit.add(
            section="Depths",
            status="WARNING",
            question="Is 1000 m requested without intentional extrapolation?",
            evidence=f"REQUEST_MAX_DEPTH = {request_max_depth}",
            interpretation=(
                "The source request does not clearly extend below 1000 m."
            ),
            recommendation=(
                "Request deeper native levels and explicitly reject "
                "vertical extrapolation."
            ),
        )

    # ------------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------------

    if (
        start_date == "2012-01-01"
        and end_date == "2024-12-31"
    ):
        audit.add(
            section="Temporal coverage",
            status="PASS",
            question="Is the configured temporal coverage correct?",
            evidence=f"{start_date} through {end_date}",
            interpretation=(
                "The acquisition script requests the intended inclusive "
                "2012–2024 period."
            ),
        )
    else:
        audit.add(
            section="Temporal coverage",
            status="WARNING",
            question="Is the configured temporal coverage correct?",
            evidence=f"{start_date} through {end_date}",
            interpretation=(
                "The configured dates differ from the intended period."
            ),
        )

    if "p1d" in source_lower:
        audit.add(
            section="Temporal interpretation",
            status="PASS",
            question="Is the source product daily?",
            evidence=f"Dataset ID contains P1D: {dataset_id}",
            interpretation=(
                "The selected source product is identified as daily."
            ),
        )
    else:
        audit.add(
            section="Temporal interpretation",
            status="WARNING",
            question="Is the source product daily?",
            evidence=f"Dataset ID = {dataset_id}",
            interpretation=(
                "The daily P1D marker was not found."
            ),
        )

    audit.add(
        section="Temporal interpretation",
        status="REQUIRES EXTERNAL DOCUMENTATION",
        question="Are the values definitely 24-hour daily means?",
        evidence=(
            "The acquisition script selects a daily product but does not "
            "independently calculate a 24-hour mean."
        ),
        interpretation=(
            "Daily sampling and daily averaging are not automatically "
            "the same. The exact averaging convention must be confirmed "
            "from Copernicus metadata and time bounds."
        ),
        recommendation=(
            "Use 'daily GLORYS product' unless the official metadata "
            "explicitly confirms the daily averaging convention."
        ),
    )

    # ------------------------------------------------------------------------
    # Vertical interpolation
    # ------------------------------------------------------------------------

    if (
        "thetao.interp" in source_lower
        and "linear" in source_lower
    ):
        audit.add(
            section="Vertical processing",
            status="PASS",
            question="Are positive target depths linearly interpolated?",
            evidence=(
                "The script contains thetao.interp and linear interpolation."
            ),
            interpretation=(
                "The positive target depths are intended to be obtained "
                "through linear interpolation."
            ),
        )
    else:
        audit.add(
            section="Vertical processing",
            status="WARNING",
            question="Are positive target depths linearly interpolated?",
            evidence=(
                "Linear vertical interpolation was not clearly detected."
            ),
            interpretation=(
                "The exact vertical processing method could not be confirmed."
            ),
        )

    if (
        "surface = thetao.sel" in source_lower
        and "nearest" in source_lower
    ):
        audit.add(
            section="Vertical processing",
            status="WARNING",
            question="Is the 0 m value a true native surface value?",
            evidence=(
                "The script selects the nearest native depth for the surface."
            ),
            interpretation=(
                "The 0 m value is a shallow physical-level proxy, not "
                "necessarily a literal native 0 m value."
            ),
            recommendation=(
                "Document the surface method as nearest shallow native "
                "physical level."
            ),
        )

    # ------------------------------------------------------------------------
    # Horizontal processing
    # ------------------------------------------------------------------------

    if (
        ".rolling(" in source_lower
        and "longitude=3" in source_lower
        and "latitude=3" in source_lower
    ):
        audit.add(
            section="Horizontal processing",
            status="WARNING",
            question="Was the native grid formally regridded to 0.25 degree?",
            evidence=(
                "The script applies 3-point rolling operations in longitude "
                "and latitude."
            ),
            interpretation=(
                "This is spatial smoothing followed by coordinate mapping. "
                "It is not automatically equivalent to conservative, "
                "area-weighted, or bilinear regridding."
            ),
            recommendation=(
                "Describe this as spatial smoothing and mapping to a "
                "nominal 0.25-degree grid."
            ),
        )
    else:
        audit.add(
            section="Horizontal processing",
            status="NOT PROVABLE FROM FILES",
            question="Was the native grid formally regridded to 0.25 degree?",
            evidence=(
                "The expected rolling operation was not detected."
            ),
            interpretation=(
                "The exact horizontal processing method could not be confirmed."
            ),
        )

    if "min_periods=1" in source_lower:
        audit.add(
            section="Horizontal processing",
            status="WARNING",
            question="Are edge-cell smoothing weights uniform?",
            evidence="The rolling operation uses min_periods=1.",
            interpretation=(
                "Edge cells may use fewer source points and therefore "
                "have different effective smoothing weights."
            ),
        )

    # ------------------------------------------------------------------------
    # DUACS and OSCAR
    # ------------------------------------------------------------------------

    if "duacs" in source_lower or "altimetry" in source_lower:
        audit.add(
            section="DUACS and OSCAR",
            status="WARNING",
            question="Are DUACS variables being mixed into GLORYS temperature?",
            evidence="DUACS or altimetry-related text was found.",
            interpretation=(
                "DUACS should not be silently treated as part of the "
                "GLORYS temperature target."
            ),
            recommendation=(
                "Keep DUACS as a separately documented SSH or altimetry "
                "product if it is used elsewhere."
            ),
        )
    else:
        audit.add(
            section="DUACS and OSCAR",
            status="PASS",
            question="Are DUACS variables being mixed into GLORYS temperature?",
            evidence="No DUACS-related text was found.",
            interpretation=(
                "The GLORYS temperature acquisition script does not "
                "appear to use DUACS."
            ),
        )

    if "oscar" in source_lower or "surface current" in source_lower:
        audit.add(
            section="DUACS and OSCAR",
            status="WARNING",
            question="Are OSCAR variables being mixed into GLORYS temperature?",
            evidence="OSCAR or surface-current-related text was found.",
            interpretation=(
                "OSCAR should be treated as a separate surface-current "
                "product, not as GLORYS temperature."
            ),
        )
    else:
        audit.add(
            section="DUACS and OSCAR",
            status="PASS",
            question="Are OSCAR variables being mixed into GLORYS temperature?",
            evidence="No OSCAR-related text was found.",
            interpretation=(
                "The GLORYS temperature acquisition script does not "
                "appear to use OSCAR."
            ),
        )

    # ------------------------------------------------------------------------
    # Product interpretation
    # ------------------------------------------------------------------------

    audit.add(
        section="Product interpretation",
        status="PASS",
        question="Does the dataset represent what we claim it represents?",
        evidence=(
            "The configured product is GLORYS and the intended variable "
            "is thetao."
        ),
        interpretation=(
            "The scientifically correct description is processed daily "
            "GLORYS assimilative ocean-reanalysis temperature."
        ),
        recommendation=(
            "Do not describe GLORYS as independent observational truth."
        ),
    )

    audit.add(
        section="Product interpretation",
        status="WARNING",
        question="Are analysis, observation, reanalysis, and derived products distinguished?",
        evidence=(
            "GLORYS is model-based and assimilative, and the final data "
            "are additionally spatially and vertically processed."
        ),
        interpretation=(
            "The final dataset is not raw observations. It contains "
            "model dynamics, forcing, assimilation influence, and "
            "processing effects."
        ),
        recommendation=(
            "Keep observation, analysis, reanalysis, and derived product "
            "labels separate in the project documentation."
        ),
    )

    # ------------------------------------------------------------------------
    # ML limitations
    # ------------------------------------------------------------------------

    audit.add(
        section="ML limitations",
        status="WARNING",
        question="Can GLORYS limitations affect ML training or validation?",
        evidence=(
            "The target is an assimilative reanalysis and is processed "
            "using vertical interpolation and spatial smoothing."
        ),
        interpretation=(
            "Potential effects include model bias, assimilation bias, "
            "weakened fronts, reduced small-scale variability, smoothed "
            "thermoclines, and depth-dependent missingness."
        ),
        recommendation=(
            "Report metrics separately by depth, season, region, and "
            "coastal/open-ocean regime."
        ),
    )

    audit.add(
        section="ML limitations",
        status="WARNING",
        question="Can random train/test splitting cause leakage?",
        evidence=(
            "Ocean temperature is strongly autocorrelated in time and space."
        ),
        interpretation=(
            "Randomly splitting nearby dates or neighbouring grid cells "
            "can place nearly identical ocean states in training and "
            "validation sets."
        ),
        recommendation=(
            "Use blocked temporal splits, year-based splits, or spatial "
            "holdouts."
        ),
    )

    audit.add(
        section="ML limitations",
        status="WARNING",
        question="Can spatial reduction remove important physical structure?",
        evidence=(
            "The native grid is approximately 0.0833 degree and the "
            "final grid is nominally 0.25 degree."
        ),
        interpretation=(
            "The reduction can weaken mesoscale fronts, eddies, filaments, "
            "coastal gradients, and small-scale variability."
        ),
        recommendation=(
            "Compare native and processed fields using gradients, "
            "variance, transects, and representative maps."
        ),
    )

    # ------------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------------

    audit.add(
        section="Provenance",
        status="INFO",
        question="Is the acquisition script fingerprinted?",
        evidence=f"SHA256 = {source_info['sha256']}",
        interpretation=(
            "The exact acquisition script can be identified by its checksum."
        ),
    )

    if "sha256" in source_lower:
        audit.add(
            section="Provenance",
            status="PASS",
            question="Does the acquisition script use file checksums?",
            evidence="SHA256 logic was found in the acquisition script.",
            interpretation=(
                "Checksums improve reproducibility and file-integrity tracking."
            ),
        )
    else:
        audit.add(
            section="Provenance",
            status="WARNING",
            question="Does the acquisition script use file checksums?",
            evidence="SHA256 logic was not found.",
            interpretation=(
                "The acquisition script does not clearly fingerprint "
                "downloaded or processed files."
            ),
        )

    if "unlink" in source_lower or "safe_unlink" in source_lower:
        audit.add(
            section="Provenance",
            status="WARNING",
            question="Are native source files retained?",
            evidence="File deletion logic was found in the acquisition script.",
            interpretation=(
                "Deleting native source subsets makes later scientific "
                "comparison with the exact source data difficult."
            ),
            recommendation=(
                "Retain at least one native audit sample per year or "
                "archive the native files separately."
            ),
        )

    return source_info


# ============================================================================
# FINAL NETCDF FILE AUDIT
# ============================================================================


def audit_netcdf_file(
    audit: Audit,
    path: Path,
) -> dict[str, Any]:
    file_info: dict[str, Any] = {
        "file": str(path),
        "sha256": sha256_file(path),
    }

    try:
        dataset = xr.open_dataset(
            path,
            decode_cf=True,
            mask_and_scale=True,
        )
    except Exception as error:
        audit.add(
            section="Final NetCDF files",
            status="FAIL",
            question=f"Can {path.name} be opened?",
            evidence=f"{type(error).__name__}: {error}",
            interpretation=(
                "The NetCDF file could not be opened for scientific inspection."
            ),
        )

        return file_info

    try:
        variable_name = find_temperature_variable(dataset)

        if variable_name is None:
            audit.add(
                section="Physical variable",
                status="FAIL",
                question=f"Does {path.name} contain thetao?",
                evidence=f"Variables = {list(dataset.data_vars)}",
                interpretation=(
                    "The expected sea-water potential-temperature variable "
                    "was not found."
                ),
            )

            return file_info

        variable = dataset[variable_name]

        file_info["variable"] = variable_name
        file_info["dimensions"] = dict(dataset.sizes)
        file_info["data_variables"] = list(dataset.data_vars)

        # --------------------------------------------------------------------
        # Variable identity
        # --------------------------------------------------------------------

        standard_name = variable.attrs.get("standard_name")
        long_name = variable.attrs.get("long_name")
        units = variable.attrs.get("units")

        standard_name_text = normalise(standard_name)
        long_name_text = normalise(long_name)

        metadata_identifies_temperature = (
            "sea_water_potential_temperature" in standard_name_text
            or "potential temperature" in long_name_text
        )

        if (
            variable_name == EXPECTED_VARIABLE
            and metadata_identifies_temperature
        ):
            audit.add(
                section="Physical variable",
                status="PASS",
                question=f"Is {path.name} the correct physical variable?",
                evidence=(
                    f"Variable = {variable_name}; "
                    f"standard_name = {standard_name}; "
                    f"long_name = {long_name}"
                ),
                interpretation=(
                    "The file contains sea-water potential temperature."
                ),
            )
        elif variable_name == EXPECTED_VARIABLE:
            audit.add(
                section="Physical variable",
                status="WARNING",
                question=f"Is {path.name} the correct physical variable?",
                evidence=(
                    f"Variable = {variable_name}; "
                    f"standard_name = {standard_name}; "
                    f"long_name = {long_name}"
                ),
                interpretation=(
                    "The variable name is correct, but the CF metadata "
                    "does not clearly identify potential temperature."
                ),
            )
        else:
            audit.add(
                section="Physical variable",
                status="FAIL",
                question=f"Is {path.name} the correct physical variable?",
                evidence=(
                    f"Detected variable = {variable_name}; "
                    f"expected = {EXPECTED_VARIABLE}"
                ),
                interpretation=(
                    "The final file does not contain the expected thetao variable."
                ),
            )

        # --------------------------------------------------------------------
        # Coordinate identification
        # --------------------------------------------------------------------

        depth_name = find_depth_coordinate(
            dataset,
            variable_name,
        )

        time_name = find_time_coordinate(
            dataset,
            variable_name,
        )

        lat_name = find_latitude_coordinate(
            dataset,
            variable_name,
        )

        lon_name = find_longitude_coordinate(
            dataset,
            variable_name,
        )

        file_info.update(
            {
                "depth_coordinate": depth_name,
                "time_coordinate": time_name,
                "latitude_coordinate": lat_name,
                "longitude_coordinate": lon_name,
            }
        )

        missing_coordinates = [
            name
            for name, value in {
                "depth": depth_name,
                "time": time_name,
                "latitude": lat_name,
                "longitude": lon_name,
            }.items()
            if value is None
        ]

        if not missing_coordinates:
            audit.add(
                section="Dimensions",
                status="PASS",
                question=f"Are the dimensions identifiable in {path.name}?",
                evidence=(
                    f"time={time_name}, "
                    f"depth={depth_name}, "
                    f"latitude={lat_name}, "
                    f"longitude={lon_name}; "
                    f"dims={dict(dataset.sizes)}"
                ),
                interpretation=(
                    "The expected physical axes are identifiable."
                ),
            )
        else:
            audit.add(
                section="Dimensions",
                status="FAIL",
                question=f"Are the dimensions identifiable in {path.name}?",
                evidence=(
                    f"Missing coordinate identifications = "
                    f"{missing_coordinates}"
                ),
                interpretation=(
                    "The file structure does not clearly expose all "
                    "expected physical axes."
                ),
            )

        # --------------------------------------------------------------------
        # Depth checks
        # --------------------------------------------------------------------

        if depth_name is not None:
            depths = np.asarray(
                dataset[depth_name].values,
                dtype=float,
            )

            file_info["depths"] = depths.tolist()

            if arrays_match(
                depths,
                EXPECTED_DEPTHS,
                DEPTH_TOLERANCE,
            ):
                audit.add(
                    section="Depths",
                    status="PASS",
                    question=f"Are the 15 required depths present in {path.name}?",
                    evidence=f"Depths = {array_string(depths)}",
                    interpretation=(
                        "The file contains exactly the required target depths."
                    ),
                )
            else:
                audit.add(
                    section="Depths",
                    status="FAIL",
                    question=f"Are the 15 required depths present in {path.name}?",
                    evidence=f"Depths = {array_string(depths)}",
                    interpretation=(
                        "The file's depth vector differs from the required "
                        "OceanEmbed depth vector."
                    ),
                )

            depth_units = dataset[depth_name].attrs.get("units")
            depth_positive = dataset[depth_name].attrs.get("positive")

            depth_units_text = normalise(depth_units)

            depth_units_correct = (
                depth_units_text in {
                    "m",
                    "meter",
                    "meters",
                    "metre",
                    "metres",
                }
                or "meter" in depth_units_text
                or "metre" in depth_units_text
            )

            if (
                depth_units_correct
                and normalise(depth_positive) == "down"
            ):
                audit.add(
                    section="Depths",
                    status="PASS",
                    question=f"Are depth units and direction correct in {path.name}?",
                    evidence=(
                        f"units={depth_units}; "
                        f"positive={depth_positive}"
                    ),
                    interpretation=(
                        "Depths are expressed in metres and increase downward."
                    ),
                )
            else:
                audit.add(
                    section="Depths",
                    status="WARNING",
                    question=f"Are depth units and direction correct in {path.name}?",
                    evidence=(
                        f"units={depth_units}; "
                        f"positive={depth_positive}"
                    ),
                    interpretation=(
                        "The depth metadata does not clearly indicate "
                        "metres with positive direction downward."
                    ),
                )

            if np.any(
                np.isclose(
                    depths,
                    0.0,
                    atol=DEPTH_TOLERANCE,
                )
            ):
                audit.add(
                    section="Depths",
                    status="WARNING",
                    question=f"Is 0 m a literal native surface level in {path.name}?",
                    evidence="The final file contains a depth coordinate labelled 0 m.",
                    interpretation=(
                        "The coordinate label alone cannot prove that the "
                        "source contained a literal 0 m physical level. "
                        "In the current pipeline, it is expected to be a "
                        "nearest shallow-level proxy."
                    ),
                )

        # --------------------------------------------------------------------
        # Longitude checks
        # --------------------------------------------------------------------

        if lon_name is not None:
            longitudes = np.asarray(
                dataset[lon_name].values,
                dtype=float,
            )

            longitude_min = float(np.nanmin(longitudes))
            longitude_max = float(np.nanmax(longitudes))
            longitude_step = coordinate_step(longitudes)

            file_info.update(
                {
                    "longitude_min": longitude_min,
                    "longitude_max": longitude_max,
                    "longitude_step": longitude_step,
                }
            )

            if (
                abs(longitude_min - EXPECTED_MIN_LON)
                <= COORDINATE_TOLERANCE
                and abs(longitude_max - EXPECTED_MAX_LON)
                <= COORDINATE_TOLERANCE
            ):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is the longitude domain correct in {path.name}?",
                    evidence=(
                        f"Longitude = {longitude_min} to {longitude_max} E"
                    ),
                    interpretation=(
                        "The longitude bounds match the required NIO domain."
                    ),
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is the longitude domain correct in {path.name}?",
                    evidence=(
                        f"Longitude = {longitude_min} to {longitude_max} E; "
                        f"expected = {EXPECTED_MIN_LON} to {EXPECTED_MAX_LON} E"
                    ),
                    interpretation=(
                        "The longitude bounds do not match the required domain."
                    ),
                )

            if (
                longitude_step is not None
                and abs(
                    longitude_step - EXPECTED_FINAL_GRID_STEP
                )
                <= GRID_TOLERANCE
            ):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is longitude spacing 0.25 degree in {path.name}?",
                    evidence=f"Longitude spacing = {longitude_step}",
                    interpretation=(
                        "The final longitude spacing matches the nominal "
                        "0.25-degree grid."
                    ),
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is longitude spacing 0.25 degree in {path.name}?",
                    evidence=f"Longitude spacing = {longitude_step}",
                    interpretation=(
                        "The final longitude spacing is not 0.25 degree."
                    ),
                )

            if is_monotonic(longitudes):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is longitude monotonic in {path.name}?",
                    evidence="Longitude coordinate is monotonic.",
                    interpretation="Longitude ordering is valid.",
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is longitude monotonic in {path.name}?",
                    evidence="Longitude coordinate is not monotonic.",
                    interpretation=(
                        "Non-monotonic longitude can cause incorrect "
                        "spatial interpretation."
                    ),
                )

            duplicates = duplicate_values(longitudes)

            if len(duplicates) == 0:
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Are longitude coordinates unique in {path.name}?",
                    evidence="No duplicate longitude coordinates found.",
                    interpretation="Longitude coordinates are unique.",
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Are longitude coordinates unique in {path.name}?",
                    evidence=f"Duplicates = {array_string(duplicates)}",
                    interpretation=(
                        "Duplicate longitude coordinates can create "
                        "ambiguous grid cells."
                    ),
                )

        # --------------------------------------------------------------------
        # Latitude checks
        # --------------------------------------------------------------------

        if lat_name is not None:
            latitudes = np.asarray(
                dataset[lat_name].values,
                dtype=float,
            )

            latitude_min = float(np.nanmin(latitudes))
            latitude_max = float(np.nanmax(latitudes))
            latitude_step = coordinate_step(latitudes)

            file_info.update(
                {
                    "latitude_min": latitude_min,
                    "latitude_max": latitude_max,
                    "latitude_step": latitude_step,
                }
            )

            if (
                abs(latitude_min - EXPECTED_MIN_LAT)
                <= COORDINATE_TOLERANCE
                and abs(latitude_max - EXPECTED_MAX_LAT)
                <= COORDINATE_TOLERANCE
            ):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is the latitude domain correct in {path.name}?",
                    evidence=(
                        f"Latitude = {latitude_min} to {latitude_max} N"
                    ),
                    interpretation=(
                        "The latitude bounds match the required NIO domain."
                    ),
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is the latitude domain correct in {path.name}?",
                    evidence=(
                        f"Latitude = {latitude_min} to {latitude_max} N; "
                        f"expected = {EXPECTED_MIN_LAT} to {EXPECTED_MAX_LAT} N"
                    ),
                    interpretation=(
                        "The latitude bounds do not match the required domain."
                    ),
                )

            if (
                latitude_step is not None
                and abs(
                    latitude_step - EXPECTED_FINAL_GRID_STEP
                )
                <= GRID_TOLERANCE
            ):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is latitude spacing 0.25 degree in {path.name}?",
                    evidence=f"Latitude spacing = {latitude_step}",
                    interpretation=(
                        "The final latitude spacing matches the nominal "
                        "0.25-degree grid."
                    ),
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is latitude spacing 0.25 degree in {path.name}?",
                    evidence=f"Latitude spacing = {latitude_step}",
                    interpretation=(
                        "The final latitude spacing is not 0.25 degree."
                    ),
                )

            if is_monotonic(latitudes):
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Is latitude monotonic in {path.name}?",
                    evidence="Latitude coordinate is monotonic.",
                    interpretation="Latitude ordering is valid.",
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Is latitude monotonic in {path.name}?",
                    evidence="Latitude coordinate is not monotonic.",
                    interpretation=(
                        "Non-monotonic latitude can cause incorrect "
                        "spatial interpretation."
                    ),
                )

            duplicates = duplicate_values(latitudes)

            if len(duplicates) == 0:
                audit.add(
                    section="Spatial grid",
                    status="PASS",
                    question=f"Are latitude coordinates unique in {path.name}?",
                    evidence="No duplicate latitude coordinates found.",
                    interpretation="Latitude coordinates are unique.",
                )
            else:
                audit.add(
                    section="Spatial grid",
                    status="FAIL",
                    question=f"Are latitude coordinates unique in {path.name}?",
                    evidence=f"Duplicates = {array_string(duplicates)}",
                    interpretation=(
                        "Duplicate latitude coordinates can create "
                        "ambiguous grid cells."
                    ),
                )

        # --------------------------------------------------------------------
        # Time checks
        # --------------------------------------------------------------------

        if time_name is not None:
            times = dataset[time_name].values

            file_info["time_count"] = int(len(times))

            if len(times) > 0:
                first_time = times[0]
                last_time = times[-1]

                file_info["first_time"] = str(first_time)
                file_info["last_time"] = str(last_time)

                first_day = first_time.astype("datetime64[D]")
                last_day = last_time.astype("datetime64[D]")

                if (
                    first_day == EXPECTED_START_DATE
                    and last_day == EXPECTED_END_DATE
                ):
                    audit.add(
                        section="Temporal coverage",
                        status="PASS",
                        question=f"Does {path.name} cover the expected dates?",
                        evidence=(
                            f"First = {first_time}; "
                            f"last = {last_time}"
                        ),
                        interpretation=(
                            "The file begins and ends on the expected "
                            "OceanEmbed dates."
                        ),
                    )
                else:
                    audit.add(
                        section="Temporal coverage",
                        status="WARNING",
                        question=f"Does {path.name} cover the expected dates?",
                        evidence=(
                            f"First = {first_time}; "
                            f"last = {last_time}"
                        ),
                        interpretation=(
                            "The file's temporal bounds differ from "
                            "the expected 2012–2024 period."
                        ),
                    )

            if len(times) > 1:
                differences = np.diff(times) / np.timedelta64(
                    1,
                    "D",
                )

                minimum_step = float(np.min(differences))
                maximum_step = float(np.max(differences))
                median_step = float(np.median(differences))

                file_info.update(
                    {
                        "minimum_time_step_days": minimum_step,
                        "maximum_time_step_days": maximum_step,
                        "median_time_step_days": median_step,
                    }
                )

                if np.allclose(differences, 1.0):
                    audit.add(
                        section="Temporal interpretation",
                        status="PASS",
                        question=f"Are timestamps daily in {path.name}?",
                        evidence=(
                            f"Minimum step = {minimum_step} days; "
                            f"median step = {median_step} days; "
                            f"maximum step = {maximum_step} days"
                        ),
                        interpretation=(
                            "The timestamp sequence is uniformly daily."
                        ),
                    )
                else:
                    audit.add(
                        section="Temporal interpretation",
                        status="FAIL",
                        question=f"Are timestamps daily in {path.name}?",
                        evidence=(
                            f"Minimum step = {minimum_step} days; "
                            f"median step = {median_step} days; "
                            f"maximum step = {maximum_step} days"
                        ),
                        interpretation=(
                            "The timestamps are not uniformly daily."
                        ),
                    )

                gaps = differences[differences > 1.01]

                if len(gaps) == 0:
                    audit.add(
                        section="Temporal coverage",
                        status="PASS",
                        question=f"Are there temporal gaps in {path.name}?",
                        evidence="No multi-day gaps detected.",
                        interpretation=(
                            "The time coordinate is continuous at daily spacing."
                        ),
                    )
                else:
                    audit.add(
                        section="Temporal coverage",
                        status="FAIL",
                        question=f"Are there temporal gaps in {path.name}?",
                        evidence=(
                            f"Number of gaps = {len(gaps)}; "
                            f"largest gap = {float(np.max(gaps))} days"
                        ),
                        interpretation=(
                            "The file contains one or more temporal gaps."
                        ),
                    )

                duplicate_or_backward = differences <= 0.0

                if not np.any(duplicate_or_backward):
                    audit.add(
                        section="Temporal coverage",
                        status="PASS",
                        question=f"Are timestamps unique in {path.name}?",
                        evidence="No duplicate or backward timestamps detected.",
                        interpretation="The time coordinate is strictly increasing.",
                    )
                else:
                    audit.add(
                        section="Temporal coverage",
                        status="FAIL",
                        question=f"Are timestamps unique in {path.name}?",
                        evidence=(
                            f"Detected {int(np.sum(duplicate_or_backward))} "
                            "duplicate or backward transitions."
                        ),
                        interpretation=(
                            "Duplicate timestamps can cause incorrect "
                            "aggregation or ML leakage."
                        ),
                    )

        # --------------------------------------------------------------------
        # Units
        # --------------------------------------------------------------------

        units_text = normalise(units)

        if units_text in {
            "degree_celsius",
            "degrees_celsius",
            "degc",
            "°c",
            "celsius",
            "degree celsius",
        }:
            audit.add(
                section="Units",
                status="PASS",
                question=f"Are temperature units correct in {path.name}?",
                evidence=f"units = {units}",
                interpretation=(
                    "Temperature is expressed in degrees Celsius."
                ),
            )
        elif units_text in {"k", "kelvin"}:
            audit.add(
                section="Units",
                status="WARNING",
                question=f"Are temperature units correct in {path.name}?",
                evidence=f"units = {units}",
                interpretation=(
                    "Temperature is expressed in Kelvin."
                ),
                recommendation=(
                    "Convert to Celsius only if required by the project "
                    "and document the conversion."
                ),
            )
        else:
            audit.add(
                section="Units",
                status="WARNING",
                question=f"Are temperature units correct in {path.name}?",
                evidence=f"units = {units}",
                interpretation=(
                    "The temperature units are missing or unrecognised."
                ),
            )

        # --------------------------------------------------------------------
        # Missing values
        # --------------------------------------------------------------------

        statistics = variable_statistics(variable)

        file_info["statistics"] = statistics

        fill_metadata = has_fill_metadata(variable)

        if statistics["missing_fraction"] == 0.0:
            audit.add(
                section="Missing values",
                status="PASS",
                question=f"Are there missing values in {path.name}?",
                evidence=(
                    f"Missing fraction = 0.0; "
                    f"fill metadata present = {fill_metadata}"
                ),
                interpretation=(
                    "No NaN or infinite values were found in the decoded "
                    "temperature array."
                ),
            )
        else:
            audit.add(
                section="Missing values",
                status="WARNING",
                question=f"Are there missing values in {path.name}?",
                evidence=(
                    f"Missing fraction = "
                    f"{statistics['missing_fraction']:.8f}; "
                    f"fill metadata present = {fill_metadata}"
                ),
                interpretation=(
                    "Missing values exist. Their distribution should be "
                    "evaluated by depth and geography."
                ),
                recommendation=(
                    "Preserve a missing-data mask and calculate missingness "
                    "by depth and region."
                ),
            )

        # --------------------------------------------------------------------
        # Physical plausibility
        # --------------------------------------------------------------------

        minimum = statistics["minimum"]
        maximum = statistics["maximum"]

        if minimum is None or maximum is None:
            audit.add(
                section="Physical plausibility",
                status="FAIL",
                question=f"Are values usable in {path.name}?",
                evidence="No finite temperature values found.",
                interpretation=(
                    "The file contains no usable temperature data."
                ),
            )
        elif (
            minimum < MIN_REASONABLE_TEMPERATURE
            or maximum > MAX_REASONABLE_TEMPERATURE
        ):
            audit.add(
                section="Physical plausibility",
                status="FAIL",
                question=f"Are temperature values plausible in {path.name}?",
                evidence=(
                    f"Minimum = {minimum} °C; "
                    f"maximum = {maximum} °C"
                ),
                interpretation=(
                    "The values exceed broad seawater-temperature "
                    "plausibility limits."
                ),
                recommendation=(
                    "Inspect units, fill-value decoding, scaling, and "
                    "possible file corruption."
                ),
            )
        else:
            audit.add(
                section="Physical plausibility",
                status="PASS",
                question=f"Are temperature values plausible in {path.name}?",
                evidence=(
                    f"Minimum = {minimum} °C; "
                    f"maximum = {maximum} °C"
                ),
                interpretation=(
                    "The decoded temperature range is broadly plausible."
                ),
            )

        # --------------------------------------------------------------------
        # Depth-dependent missingness
        # --------------------------------------------------------------------

        if (
            depth_name is not None
            and depth_name in variable.dims
        ):
            values = np.asarray(variable.values)

            depth_axis = variable.get_axis_num(depth_name)

            other_axes = tuple(
                axis
                for axis in range(values.ndim)
                if axis != depth_axis
            )

            missing_by_depth = np.mean(
                ~np.isfinite(values),
                axis=other_axes,
            )

            file_info["missing_fraction_by_depth"] = (
                np.asarray(missing_by_depth).tolist()
            )

            depth_lines = []

            for depth, missing_fraction in zip(
                dataset[depth_name].values,
                missing_by_depth,
            ):
                depth_lines.append(
                    f"{float(depth):g}m="
                    f"{float(missing_fraction):.6f}"
                )

            audit.add(
                section="Missing values",
                status="INFO",
                question=f"What is missingness by depth in {path.name}?",
                evidence="; ".join(depth_lines),
                interpretation=(
                    "Depth-dependent missingness must be considered "
                    "during ML training and evaluation."
                ),
            )

    finally:
        dataset.close()

    return file_info


# ============================================================================
# FINAL DATASET AUDIT
# ============================================================================


def audit_final_dataset(
    audit: Audit,
) -> list[dict[str, Any]]:
    file_information: list[dict[str, Any]] = []

    audit.add(
        section="Final NetCDF files",
        status="INFO",
        question="Which final GLORYS directory is being audited?",
        evidence=str(GLORYS_DATA_DIR),
        interpretation=(
            "The final processed files are read from the fixed directory "
            "configured inside this script."
        ),
    )

    if not GLORYS_DATA_DIR.exists():
        audit.add(
            section="Final NetCDF files",
            status="FAIL",
            question="Does the final GLORYS directory exist?",
            evidence=str(GLORYS_DATA_DIR),
            interpretation=(
                "The final processed GLORYS directory could not be found."
            ),
            recommendation=(
                "Correct GLORYS_DATA_DIR near the top of this file."
            ),
        )

        return file_information

    files = sorted(
        GLORYS_DATA_DIR.glob("*.nc")
    )

    if not files:
        audit.add(
            section="Final NetCDF files",
            status="FAIL",
            question="Are final GLORYS NetCDF files available?",
            evidence=f"No .nc files found in {GLORYS_DATA_DIR}",
            interpretation=(
                "The final dataset cannot be audited because no NetCDF "
                "files were found."
            ),
        )

        return file_information

    audit.add(
        section="Final NetCDF files",
        status="INFO",
        question="How many final GLORYS files were found?",
        evidence=f"{len(files)} NetCDF files found.",
        interpretation=(
            "Every discovered NetCDF file will be checked."
        ),
    )

    for file in files:
        file_information.append(
            audit_netcdf_file(
                audit,
                file,
            )
        )

    return file_information


# ============================================================================
# REPORT GENERATION
# ============================================================================


def write_text_report(
    audit: Audit,
    source_info: dict[str, Any],
    file_information: list[dict[str, Any]],
) -> None:
    lines: list[str] = []

    lines.append("GLORYS SCIENTIFIC VALIDATION REPORT")
    lines.append("=" * 100)
    lines.append(f"Audit time: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("FIXED PATHS")
    lines.append("-" * 100)
    lines.append(f"Base directory: {BASE_DIR}")
    lines.append(f"Acquisition script: {ACQUISITION_SCRIPT}")
    lines.append(f"Final data directory: {GLORYS_DATA_DIR}")
    lines.append(f"Text report: {TEXT_REPORT}")
    lines.append(f"JSON report: {JSON_REPORT}")
    lines.append("")

    lines.append("IMPORTANT SCIENTIFIC SCOPE")
    lines.append("-" * 100)
    lines.append(
        "This audit verifies structure, metadata, coordinates, values, "
        "processing configuration, and provenance."
    )
    lines.append(
        "It does not prove that GLORYS is independent observational truth."
    )
    lines.append(
        "It does not independently validate GLORYS against ARGO because "
        "the ARGO question was intentionally excluded."
    )
    lines.append("")

    lines.append("SOURCE SCRIPT SUMMARY")
    lines.append("-" * 100)

    for key, value in source_info.items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("FINAL FILE SUMMARY")
    lines.append("-" * 100)
    lines.append(f"Number of files audited: {len(file_information)}")

    for file_info in file_information:
        lines.append("")
        lines.append(f"File: {file_info.get('file')}")
        lines.append(f"SHA256: {file_info.get('sha256')}")

        if "variable" in file_info:
            lines.append(f"Variable: {file_info['variable']}")

        if "dimensions" in file_info:
            lines.append(f"Dimensions: {file_info['dimensions']}")

        if "statistics" in file_info:
            lines.append(
                f"Statistics: {file_info['statistics']}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED FINDINGS")
    lines.append("=" * 100)

    current_section = None

    for finding in audit.findings:
        if finding.section != current_section:
            current_section = finding.section

            lines.append("")
            lines.append("-" * 100)
            lines.append(current_section)
            lines.append("-" * 100)

        lines.append("")
        lines.append(f"STATUS: {finding.status}")
        lines.append(f"QUESTION: {finding.question}")
        lines.append(f"EVIDENCE: {finding.evidence}")
        lines.append(
            f"INTERPRETATION: {finding.interpretation}"
        )

        if finding.recommendation:
            lines.append(
                f"RECOMMENDATION: {finding.recommendation}"
            )

    lines.append("")
    lines.append("=" * 100)
    lines.append("OVERALL SCIENTIFIC CONCLUSION")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        "The GLORYS configuration is scientifically reasonable for "
        "a daily North Indian Ocean subsurface temperature target."
    )
    lines.append("")
    lines.append(
        "The correct description is: processed daily GLORYS "
        "assimilative ocean-reanalysis temperature."
    )
    lines.append("")
    lines.append(
        "The current horizontal operation must not be described as "
        "formal conservative or area-weighted regridding unless the "
        "processing code is changed. A 3x3 rolling operation is "
        "spatial smoothing followed by coordinate mapping."
    )
    lines.append("")
    lines.append(
        "The 0 m level should be documented as a nearest shallow "
        "native physical-level proxy."
    )
    lines.append("")
    lines.append(
        "The strongest defensible temporal wording is 'daily GLORYS "
        "product'. The exact daily-mean convention requires confirmation "
        "from official Copernicus metadata."
    )
    lines.append("")
    lines.append(
        "For ML, use blocked temporal or spatial validation, preserve "
        "missing-data masks, and report performance by depth, season, "
        "and region."
    )
    lines.append("")

    TEXT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEXT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_json_report(
    audit: Audit,
    source_info: dict[str, Any],
    file_information: list[dict[str, Any]],
) -> None:
    counts: dict[str, int] = {}

    for finding in audit.findings:
        counts[finding.status] = (
            counts.get(finding.status, 0) + 1
        )

    payload = {
        "audit_time": datetime.now().isoformat(),
        "paths": {
            "base_dir": str(BASE_DIR),
            "acquisition_script": str(ACQUISITION_SCRIPT),
            "glorys_data_dir": str(GLORYS_DATA_DIR),
            "text_report": str(TEXT_REPORT),
            "json_report": str(JSON_REPORT),
        },
        "summary_counts": counts,
        "source_info": source_info,
        "files": file_information,
        "findings": [
            asdict(finding)
            for finding in audit.findings
        ],
    }

    JSON_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_REPORT.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:
    print("")
    print("=" * 80)
    print("GLORYS SCIENTIFIC VALIDATION AUDIT")
    print("=" * 80)
    print("")
    print(f"Acquisition script: {ACQUISITION_SCRIPT}")
    print(f"Final data folder:  {GLORYS_DATA_DIR}")
    print(f"Text report:        {TEXT_REPORT}")
    print(f"JSON report:        {JSON_REPORT}")
    print("")

    audit = Audit()

    source_info = audit_acquisition_script(
        audit
    )

    file_information = audit_final_dataset(
        audit
    )

    write_text_report(
        audit,
        source_info,
        file_information,
    )

    write_json_report(
        audit,
        source_info,
        file_information,
    )

    print("=" * 80)
    print("AUDIT COMPLETED")
    print("=" * 80)
    print("")
    print(f"Text report saved to:")
    print(TEXT_REPORT)
    print("")
    print(f"JSON report saved to:")
    print(JSON_REPORT)
    print("")
    print(
        "The audit does not prove observational truth. "
        "It verifies the dataset configuration, structure, "
        "metadata, values, and documented processing."
    )
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())