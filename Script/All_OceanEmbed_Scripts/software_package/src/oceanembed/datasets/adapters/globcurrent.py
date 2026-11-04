"""
OceanEmbed GlobCurrent Dataset Adapter.

Constructs acquisition requests for CMEMS GlobCurrent daily current vector files
via the copernicusmarine Python client.

Primary source
--------------
Product  : MULTIOBS_GLO_PHY_MYNRT_015_003
Dataset  : cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m
Provider : Copernicus Marine Service (CMEMS)
Access   : copernicusmarine Python client (authenticated)
Variables: uo (eastward sea water velocity), vo (northward sea water velocity)

Acquisition strategy
--------------------
GlobCurrent is acquired in YEARLY CHUNKS: one copernicusmarine.subset() call
per calendar year, returning a single NetCDF file containing all daily records
for that year.

This is the strategy verified in the legacy acquisition script
(legacy/acquisition/fetch_CMEMS_GlobCurrent.py) and documented in the
dataset registry (configs/datasets/globcurrent.yaml).

The public interface accepts arbitrary start_date / end_date; the adapter
internally translates the date range into one request per year needed to
cover the requested period.

Depth
-----
The legacy script requests depth=0m only. This is preserved as the ACQUISITION
BEHAVIOR. It is NOT a scientific claim about the physical meaning of 0m in this
product. See configs/datasets/globcurrent.yaml for the documented open decision
status and legacy note.

Parameterization
----------------
All spatial and temporal bounds are request-time parameters. No coordinates are
hardcoded. A caller can request, for example:

    start_date=2023-01-01, end_date=2023-01-07
    min_lat=10.0, max_lat=25.0, min_lon=60.0, max_lon=90.0

without modifying source code.

Authentication
--------------
Credentials are NOT hardcoded. The copernicusmarine client reads them from:
    COPERNICUSMARINE_SERVICE_USERNAME
    COPERNICUSMARINE_SERVICE_PASSWORD
environment variables (documented in .env.example). Credentials are never
logged or printed.

Engine integration
------------------
GlobCurrent uses a different execution path from OISST because:
  - OISST: plain HTTPS URL → download_file() → atomic .tmp rename
  - GlobCurrent: copernicusmarine.subset() API → writes file directly to
    output directory; there is no plain HTTP URL to hand to download_file().

The adapter overrides date_range_requests() to produce one AcquisitionRequest
per year. The url_or_query field contains a sentinel "cmems://<dataset_id>"
that the engine recognises as a CMEMS-flavoured request, delegating execution
to run_cmems_subset() instead of download_file().

Scientific guardrail
--------------------
This adapter constructs requests only. It does NOT:
- Regrid the data
- Convert monthly to daily
- Fill missing values (OSD-012 open)
- Interpolate depth layers
- Convert units
- Align with the OceanEmbed canonical grid (OSD-005 open)
"""

from __future__ import annotations

import calendar
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants (all verified from legacy script + YAML)
# ---------------------------------------------------------------------------

_CMEMS_PRODUCT_ID = "MULTIOBS_GLO_PHY_MYNRT_015_003"
_CMEMS_DATASET_ID = "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"
_CMEMS_VARIABLES = ["uo", "vo"]
_CMEMS_DEPTH = 0.0  # LEGACY BEHAVIOR — see module docstring

# Sentinel prefix: url_or_query values starting with this string signal
# to the acquisition engine that this is a CMEMS subset request.
_CMEMS_URL_SENTINEL = f"cmems://{_CMEMS_DATASET_ID}"

# Source temporal coverage (verified from YAML / legacy script)
_SOURCE_COVERAGE_START = date(2012, 1, 1)
_SOURCE_COVERAGE_END = date(2024, 12, 31)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_LAT_MIN_VALID = -90.0
_LAT_MAX_VALID = 90.0
_LON_MIN_VALID = -180.0
_LON_MAX_VALID = 180.0


def _validate_globcurrent_request_params(
    start_date: date,
    end_date: date,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> list[str]:
    """
    Validate request parameters before building AcquisitionRequest objects.

    Returns a list of error messages. An empty list means the parameters
    are valid. Each message describes a specific problem in plain English.

    Does NOT silently clamp invalid values — callers receive explicit errors.
    """
    errors: list[str] = []

    # Date ordering
    if start_date > end_date:
        errors.append(
            f"start_date ({start_date}) must be on or before end_date ({end_date})."
        )

    # Date coverage — source provides 2012-01-01 to 2024-12-31
    if start_date < _SOURCE_COVERAGE_START:
        errors.append(
            f"start_date ({start_date}) is before the GlobCurrent source coverage "
            f"start ({_SOURCE_COVERAGE_START}). No data exists before this date."
        )
    if end_date > _SOURCE_COVERAGE_END:
        errors.append(
            f"end_date ({end_date}) is after the GlobCurrent source coverage "
            f"end ({_SOURCE_COVERAGE_END}). No data exists after this date."
        )

    # Latitude range
    if min_lat >= max_lat:
        errors.append(
            f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})."
        )
    if min_lat < _LAT_MIN_VALID or min_lat > _LAT_MAX_VALID:
        errors.append(
            f"min_lat ({min_lat}) is outside the valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )
    if max_lat < _LAT_MIN_VALID or max_lat > _LAT_MAX_VALID:
        errors.append(
            f"max_lat ({max_lat}) is outside the valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )

    # Longitude range
    if min_lon >= max_lon:
        errors.append(
            f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})."
        )
    if min_lon < _LON_MIN_VALID or min_lon > _LON_MAX_VALID:
        errors.append(
            f"min_lon ({min_lon}) is outside the valid range "
            f"[{_LON_MIN_VALID}, {_LON_MAX_VALID}]."
        )
    if max_lon < _LON_MIN_VALID or max_lon > _LON_MAX_VALID:
        errors.append(
            f"max_lon ({max_lon}) is outside the valid range "
            f"[{_LON_MIN_VALID}, {_LON_MAX_VALID}]."
        )

    return errors


def _years_in_range(start_date: date, end_date: date) -> list[int]:
    """Return a sorted list of calendar years that overlap [start_date, end_date]."""
    return list(range(start_date.year, end_date.year + 1))


def _year_period_overlap(
    year: int, start_date: date, end_date: date
) -> tuple[date, date]:
    """
    Return the (start, end) of the intersection between a calendar year and the
    requested date range.

    For example, if year=2023 and start_date=2022-11-01, end_date=2023-03-31:
        returns (2023-01-01, 2023-03-31)

    If year=2023 and start_date=2023-01-01, end_date=2024-06-30:
        returns (2023-01-01, 2023-12-31)
    """
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    overlap_start = max(start_date, year_start)
    overlap_end = min(end_date, year_end)
    return overlap_start, overlap_end


def _expected_days(year: int) -> int:
    """Return 366 for leap years, 365 otherwise."""
    return 366 if calendar.isleap(year) else 365


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GlobCurrentAdapter(DatasetAdapterBase):
    """
    Adapter for CMEMS GlobCurrent daily ocean current vectors.

    Produces one AcquisitionRequest per calendar year (or partial year if the
    requested range does not align with full years). The copernicusmarine
    client executes the actual download outside of the generic HTTPS download_file()
    path; the engine recognises the "cmems://" sentinel in url_or_query and
    calls run_cmems_subset() accordingly.

    Parameters
    ----------
    min_lat, max_lat, min_lon, max_lon:
        Spatial bounding box for the request. Defaults to the North Indian Ocean
        domain documented in configs/datasets/globcurrent.yaml. Any valid
        geographic bounding box is accepted.

    All parameters are optional at construction time; they can also be
    overridden per-request via date_range_requests().

    Notes
    -----
    - Authentication: required. COPERNICUSMARINE_SERVICE_USERNAME and
      COPERNICUSMARINE_SERVICE_PASSWORD must be set in the environment.
    - Temporal resolution: daily (P1D-m product). Acquisition is yearly.
    - Native resolution: 0.25 degree global.
    - Variables: uo (eastward), vo (northward) only.
    - Depth: 0.0 m (legacy behavior — see module docstring).
    """

    def __init__(
        self,
        min_lat: float = 5.0,
        max_lat: float = 30.0,
        min_lon: float = 45.0,
        max_lon: float = 105.0,
    ) -> None:
        self._min_lat = min_lat
        self._max_lat = max_lat
        self._min_lon = min_lon
        self._max_lon = max_lon
        # Maximum date range allowed per job (366 = one full leap year).
        # The API reads this attribute to enforce per-dataset range limits.
        self._max_job_days = 366

    @property
    def dataset_id(self) -> str:
        return "globcurrent"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="globcurrent",
            product_id=_CMEMS_PRODUCT_ID,
            product_version="UNVERIFIED",
            provider_name="Copernicus Marine Service (CMEMS)",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.25,
            variables_requested=list(_CMEMS_VARIABLES),
            authentication_required=True,
            notes=(
                "GlobCurrent MULTIOBS_GLO_PHY_MYNRT_015_003 — daily total surface "
                "and 15m current vectors. Acquired via copernicusmarine.subset() "
                "with server-side spatial, temporal, variable, and depth subsetting. "
                "One yearly NetCDF per year. Depth=0m only (legacy acquisition behavior). "
                "Credentials from COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build a CMEMS acquisition request for the calendar year containing request_date.

        Note: For GlobCurrent, one AcquisitionRequest covers the ENTIRE calendar year
        containing request_date. This method is required by the base class interface,
        but callers should prefer date_range_requests() to avoid multiple requests for
        the same year.

        Parameters
        ----------
        request_date:
            Any date within the year to acquire. The request will cover the
            full calendar year (Jan 1 to Dec 31).
        data_root:
            Root path of the data workspace (unused in path construction, kept for
            interface compatibility).
        """
        year = request_date.year
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        return self._build_year_request(year, start, end)

    def _build_year_request(
        self,
        year: int,
        period_start: date,
        period_end: date,
    ) -> AcquisitionRequest:
        """
        Build a CMEMS acquisition request for one year (or partial year).

        Parameters
        ----------
        year:
            Calendar year being acquired.
        period_start:
            First date of the period to acquire (>= year-01-01).
        period_end:
            Last date of the period to acquire (<= year-12-31).
        """
        start_str = period_start.strftime("%Y-%m-%d")
        end_str = period_end.strftime("%Y-%m-%d")

        # Deterministic filename:
        # Full year  → GlobCurrent_YYYY_surface_uv.nc
        # Partial    → GlobCurrent_YYYY_MMDD_MMDD_surface_uv.nc
        is_full_year = (
            period_start == date(year, 1, 1) and period_end == date(year, 12, 31)
        )
        if is_full_year:
            filename = f"GlobCurrent_{year}_surface_uv.nc"
        else:
            s = period_start.strftime("%m%d")
            e = period_end.strftime("%m%d")
            filename = f"GlobCurrent_{year}_{s}_{e}_surface_uv.nc"

        # Relative output path (NEVER absolute)
        relative_path = f"raw/globcurrent/{year}/{filename}"

        # Sentinel URL carries all parameters needed by the engine executor
        # Format: cmems://<dataset_id>?start=<date>&end=<date>&...
        # The engine parses this to call copernicusmarine.subset()
        sentinel_url = (
            f"{_CMEMS_URL_SENTINEL}"
            f"?start={start_str}"
            f"&end={end_str}"
            f"&lat_min={self._min_lat}"
            f"&lat_max={self._max_lat}"
            f"&lon_min={self._min_lon}"
            f"&lon_max={self._max_lon}"
            f"&depth={_CMEMS_DEPTH}"
            f"&variables={','.join(_CMEMS_VARIABLES)}"
        )

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=period_start,
            url_or_query=sentinel_url,
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                "lat_min": self._min_lat,
                "lat_max": self._max_lat,
                "lon_min": self._min_lon,
                "lon_max": self._max_lon,
            },
            extra={
                "cmems_dataset_id": _CMEMS_DATASET_ID,
                "cmems_product_id": _CMEMS_PRODUCT_ID,
                "variables": list(_CMEMS_VARIABLES),
                "depth_m": _CMEMS_DEPTH,
                "period_start": start_str,
                "period_end": end_str,
                "acquisition_year": year,
                "is_full_year": is_full_year,
                "expected_days": _expected_days(year) if is_full_year else None,
                "acquisition_method": "copernicusmarine.subset",
                "note": (
                    "Depth=0m is LEGACY BEHAVIOR. Not a scientifically approved "
                    "decision on GlobCurrent representative depth. See YAML."
                ),
            },
        )

    def date_range_requests(
        self,
        start_date: date,
        end_date: date,
        data_root: str,
    ) -> list[AcquisitionRequest]:
        """
        Build one AcquisitionRequest per calendar year in [start_date, end_date].

        Validates the request parameters before producing requests. Raises
        ValueError with a descriptive message if any parameter is invalid.
        Invalid requests never silently clamp — the caller sees the exact problem.

        Parameters
        ----------
        start_date:
            First date of the requested range (inclusive).
        end_date:
            Last date of the requested range (inclusive).
        data_root:
            Root path of the data workspace.

        Returns
        -------
        list[AcquisitionRequest]
            One request per year in the requested range. If the range spans
            part of a year (e.g. 2023-06-01 to 2023-09-30) the request covers
            that partial year only — a full yearly file is NOT acquired unless
            the full year is within the requested range.
        """
        errors = _validate_globcurrent_request_params(
            start_date=start_date,
            end_date=end_date,
            min_lat=self._min_lat,
            max_lat=self._max_lat,
            min_lon=self._min_lon,
            max_lon=self._max_lon,
        )
        if errors:
            raise ValueError(
                "Invalid GlobCurrent acquisition request:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        years = _years_in_range(start_date, end_date)
        requests = []
        for year in years:
            period_start, period_end = _year_period_overlap(year, start_date, end_date)
            requests.append(
                self._build_year_request(year, period_start, period_end)
            )
        return requests

    def validate_request(self, request: AcquisitionRequest) -> list[str]:
        """
        Validate a GlobCurrent AcquisitionRequest.

        Extends the base class structural checks with GlobCurrent-specific
        validation (sentinel URL format, required extra fields, spatial bounds).
        """
        issues = super().validate_request(request)

        # Check sentinel URL format
        if not request.url_or_query.startswith("cmems://"):
            issues.append(
                f"GlobCurrent url_or_query must start with 'cmems://', "
                f"got: {request.url_or_query[:60]!r}"
            )

        # Check required extra fields
        for field in ("cmems_dataset_id", "period_start", "period_end", "depth_m"):
            if field not in request.extra:
                issues.append(f"Missing required extra field: '{field}'")

        # Check spatial region populated
        if not request.spatial_region:
            issues.append("spatial_region is empty.")

        return issues

    def check_credentials(self) -> dict[str, Any]:
        """
        Check whether CMEMS credentials are available in the environment.

        Returns
        -------
        dict with:
            - has_username (bool)
            - has_password (bool)
            - credentials_available (bool)
            - message (str) — human-readable status; never contains credential values

        This method is safe to call in tests and logging.
        Credential VALUES are never returned or logged.
        """
        username_set = bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "").strip())
        password_set = bool(os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip())
        available = username_set and password_set

        if available:
            msg = "CMEMS credentials are present in environment."
        elif username_set and not password_set:
            msg = "COPERNICUSMARINE_SERVICE_USERNAME is set but PASSWORD is missing."
        elif password_set and not username_set:
            msg = "COPERNICUSMARINE_SERVICE_PASSWORD is set but USERNAME is missing."
        else:
            msg = (
                "CMEMS credentials not found. Set COPERNICUSMARINE_SERVICE_USERNAME "
                "and COPERNICUSMARINE_SERVICE_PASSWORD environment variables. "
                "See .env.example for details."
            )
        return {
            "has_username": username_set,
            "has_password": password_set,
            "credentials_available": available,
            "message": msg,
        }


# ---------------------------------------------------------------------------
# Sentinel URL parser (used by the acquisition engine)
# ---------------------------------------------------------------------------

def parse_cmems_sentinel_url(url: str) -> dict[str, Any]:
    """
    Parse a CMEMS sentinel URL into parameters for copernicusmarine.subset().

    Produced by GlobCurrentAdapter._build_year_request() and
    DUACSAdapter._build_month_request().

    Raises ValueError if the URL is malformed or missing required parameters.
    This is a pure function — safe to call without credentials or network.

    The ``depth`` parameter is OPTIONAL: datasets without a depth dimension
    (e.g. DUACS) omit it.  When absent, the returned dict has ``depth=None``
    and callers must NOT pass minimum_depth/maximum_depth to subset().

    Example input (with depth — GlobCurrent):
        cmems://cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m
            ?start=2023-01-01&end=2023-12-31
            &lat_min=5.0&lat_max=30.0&lon_min=45.0&lon_max=105.0
            &depth=0.0&variables=uo,vo

    Example input (no depth — DUACS):
        cmems://cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D
            ?start=2023-01-01&end=2023-01-31
            &lat_min=10.0&lat_max=20.0&lon_min=65.0&lon_max=85.0
            &variables=sla,adt,ugos,vgos

    Returns dict with keys:
        dataset_id, start, end, lat_min, lat_max, lon_min, lon_max,
        depth (float or None), variables (list[str])
    """
    if not url.startswith("cmems://"):
        raise ValueError(f"Not a CMEMS sentinel URL: {url!r}")

    if "?" not in url:
        raise ValueError(f"CMEMS sentinel URL has no query parameters: {url!r}")

    path_part, query_part = url[len("cmems://"):].split("?", 1)
    dataset_id = path_part.strip()
    if not dataset_id:
        raise ValueError(f"CMEMS sentinel URL has empty dataset_id: {url!r}")

    params: dict[str, str] = {}
    for segment in query_part.split("&"):
        if "=" not in segment:
            continue
        k, v = segment.split("=", 1)
        params[k.strip()] = v.strip()

    # depth is optional — DUACS and other no-depth datasets omit it
    required = ("start", "end", "lat_min", "lat_max", "lon_min", "lon_max", "variables")
    missing = [r for r in required if r not in params]
    if missing:
        raise ValueError(
            f"CMEMS sentinel URL missing required parameters: {missing!r}\n"
            f"URL: {url!r}"
        )

    try:
        result: dict[str, Any] = {
            "dataset_id": dataset_id,
            "start": f"{params['start']}T00:00:00",
            "end": f"{params['end']}T23:59:59",
            "lat_min": float(params["lat_min"]),
            "lat_max": float(params["lat_max"]),
            "lon_min": float(params["lon_min"]),
            "lon_max": float(params["lon_max"]),
            # depth: single value (GlobCurrent), range (GLORYS), or None (DUACS)
            "depth": float(params["depth"]) if "depth" in params else None,
            "depth_min": float(params["depth_min"]) if "depth_min" in params else None,
            "depth_max": float(params["depth_max"]) if "depth_max" in params else None,
            "variables": [v.strip() for v in params["variables"].split(",") if v.strip()],
            # dataset_version is optional — GLORYS pins version; GlobCurrent/DUACS omit it
            "dataset_version": params.get("dataset_version"),
        }
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Failed to parse CMEMS sentinel URL parameters: {exc}\n"
            f"URL: {url!r}"
        ) from exc

    if not result["variables"]:
        raise ValueError(f"CMEMS sentinel URL has empty variables list: {url!r}")

    return result
