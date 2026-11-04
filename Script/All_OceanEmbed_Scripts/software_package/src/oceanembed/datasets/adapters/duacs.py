"""
OceanEmbed DUACS Dataset Adapter.

Constructs acquisition requests for CMEMS DUACS daily sea-level and
geostrophic-current files via the copernicusmarine Python client.

Primary source
--------------
Dataset  : cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D
           (version suffix omitted — copernicusmarine auto-selects latest)
Provider : Copernicus Marine Service (CMEMS)
Access   : copernicusmarine Python client (authenticated)
Variables: sla, adt, ugos, vgos
Product  : SEA_LEVEL_GLO_PHY_L4_MY_008_047 (delayed-time, UNVERIFIED product_id)

Live-verified (2026-09-09, copernicusmarine 2.4.1)
--------------------------------------------------
- Dataset ID without version suffix resolves to version 202411 automatically.
- Variables [sla, adt, ugos, vgos] confirmed present.
- Dimensions: (time, latitude, longitude) — NO depth dimension.
- Coordinate names: latitude, longitude, time.
- Resolution: 0.125° (8 cells per degree confirmed in dry run).

Acquisition strategy
--------------------
DUACS is acquired in MONTHLY CHUNKS: one copernicusmarine.subset() call
per calendar month, returning a single NetCDF file with all daily records
for that month.

This matches the legacy acquisition strategy verified in
legacy/acquisition/fetch_CMEMS_DUACS.py (monthly chunks, up to 12
concurrent downloads). The new adapter uses the Python API rather than
the CLI subprocess; the public interface remains the same:

    start_date / end_date → adapter produces one request per month needed.

No depth parameter
------------------
DUACS has no depth dimension. The sentinel URL for DUACS omits the
``depth=`` parameter. The engine's parse_cmems_sentinel_url() handles
this by returning depth=None and skipping minimum_depth/maximum_depth
arguments to copernicusmarine.subset().

Parameterization
----------------
All spatial and temporal bounds are request-time parameters. A caller can
request any valid geographic bounding box and any date range within source
coverage without modifying source code.

Authentication
--------------
Credentials are NOT hardcoded. The copernicusmarine client reads them from:
    COPERNICUSMARINE_SERVICE_USERNAME
    COPERNICUSMARINE_SERVICE_PASSWORD
environment variables (documented in .env.example).

Scientific guardrail
--------------------
This adapter constructs requests only. It does NOT:
- Regrid from 0.125° to 0.25° (OSD-005 open)
- Compute daily means
- Fill missing values
- Merge with OISST or GlobCurrent
- Derive new variables
- Apply the OceanEmbed canonical grid
"""

from __future__ import annotations

import calendar
import os
from datetime import date
from typing import Any

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants — all verified from legacy script + live CMEMS check
# ---------------------------------------------------------------------------

# Dataset ID without version suffix: copernicusmarine auto-selects latest.
# CMEMS issues a WARNING if the version suffix is included in dataset_id.
# Verified live: auto-selects version "202411".
_CMEMS_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"

# Product family (UNVERIFIED from live source — YAML documents as UNVERIFIED).
# The legacy script uses dataset_id only; product_id for this family is
# SEA_LEVEL_GLO_PHY_L4_MY_008_047 per CMEMS portal, but not confirmed via API.
_CMEMS_PRODUCT_ID = "SEA_LEVEL_GLO_PHY_L4_MY_008_047_UNVERIFIED"

# Variables confirmed live via copernicusmarine.subset() dry run (2026-09-09).
# Exactly 4: sla, adt, ugos, vgos. No error/flag variables.
_CMEMS_VARIABLES = ["sla", "adt", "ugos", "vgos"]

# Sentinel prefix (shared CMEMS scheme; dataset_id distinguishes the product)
_CMEMS_URL_SENTINEL = f"cmems://{_CMEMS_DATASET_ID}"

# Source temporal coverage per project documentation (duacs.yaml).
# OceanEmbed period: 2012-01-01 to 2024-12-31.
# Full delayed-time archive starts 1993-01-01.
_SOURCE_COVERAGE_START = date(1993, 1, 1)
_SOURCE_COVERAGE_END = date(2024, 12, 31)

# Geographic validity bounds
_LAT_MIN_VALID = -90.0
_LAT_MAX_VALID = 90.0
_LON_MIN_VALID = -180.0
_LON_MAX_VALID = 180.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_duacs_request_params(
    start_date: date,
    end_date: date,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> list[str]:
    """
    Validate DUACS request parameters.

    Returns a list of error messages. Empty list = valid.
    Never silently clamps — callers see the exact problem.
    """
    errors: list[str] = []

    if start_date > end_date:
        errors.append(
            f"start_date ({start_date}) must be on or before end_date ({end_date})."
        )
    if start_date < _SOURCE_COVERAGE_START:
        errors.append(
            f"start_date ({start_date}) is before the DUACS source coverage "
            f"start ({_SOURCE_COVERAGE_START}). No data exists before this date."
        )
    if end_date > _SOURCE_COVERAGE_END:
        errors.append(
            f"end_date ({end_date}) is after the DUACS source coverage "
            f"end ({_SOURCE_COVERAGE_END}). No data exists after this date."
        )
    if min_lat >= max_lat:
        errors.append(
            f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})."
        )
    if not (_LAT_MIN_VALID <= min_lat <= _LAT_MAX_VALID):
        errors.append(
            f"min_lat ({min_lat}) is outside the valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )
    if not (_LAT_MIN_VALID <= max_lat <= _LAT_MAX_VALID):
        errors.append(
            f"max_lat ({max_lat}) is outside the valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )
    if min_lon >= max_lon:
        errors.append(
            f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})."
        )
    if not (_LON_MIN_VALID <= min_lon <= _LON_MAX_VALID):
        errors.append(
            f"min_lon ({min_lon}) is outside the valid range "
            f"[{_LON_MIN_VALID}, {_LON_MAX_VALID}]."
        )
    if not (_LON_MIN_VALID <= max_lon <= _LON_MAX_VALID):
        errors.append(
            f"max_lon ({max_lon}) is outside the valid range "
            f"[{_LON_MIN_VALID}, {_LON_MAX_VALID}]."
        )
    return errors


def _months_in_range(start_date: date, end_date: date) -> list[tuple[int, int]]:
    """
    Return a sorted list of (year, month) tuples covering [start_date, end_date].

    Examples
    --------
    (2023-01-05, 2023-03-20) → [(2023,1), (2023,2), (2023,3)]
    (2022-11-01, 2023-02-28) → [(2022,11),(2022,12),(2023,1),(2023,2)]
    """
    months = []
    y, m = start_date.year, start_date.month
    while (y, m) <= (end_date.year, end_date.month):
        months.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return months


def _month_period_overlap(
    year: int, month: int, start_date: date, end_date: date
) -> tuple[date, date]:
    """
    Return (overlap_start, overlap_end) of the requested range and the given month.
    """
    last_day = calendar.monthrange(year, month)[1]
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)
    return max(start_date, month_start), min(end_date, month_end)


def _is_full_month(year: int, month: int, period_start: date, period_end: date) -> bool:
    """Return True if period_start/end exactly cover the full calendar month."""
    last_day = calendar.monthrange(year, month)[1]
    return period_start == date(year, month, 1) and period_end == date(year, month, last_day)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DUACSAdapter(DatasetAdapterBase):
    """
    Adapter for CMEMS DUACS daily sea-level altimetry and geostrophic currents.

    Produces one AcquisitionRequest per calendar month in the requested date
    range. The copernicusmarine Python client executes the actual download via
    the engine's CMEMS dispatch path (run_cmems_subset).

    Parameters
    ----------
    min_lat, max_lat, min_lon, max_lon:
        Spatial bounding box. Defaults to the North Indian Ocean domain
        documented in configs/datasets/duacs.yaml. Any valid geographic
        bounding box is accepted — caller specifies per-request.

    Notes
    -----
    - Authentication: required. COPERNICUSMARINE_SERVICE_USERNAME and
      COPERNICUSMARINE_SERVICE_PASSWORD must be set in the environment.
    - Temporal resolution: daily (P1D product). Acquisition is monthly.
    - Native resolution: 0.125° global.
    - Variables: sla, adt, ugos, vgos (no depth dimension).
    - No depth parameter is passed to copernicusmarine.subset() — DUACS has
      no depth coordinate.
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
        # Maximum date range per API job: 366 days (one full year of monthly files).
        self._max_job_days = 366

    @property
    def dataset_id(self) -> str:
        return "duacs"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="duacs",
            product_id=_CMEMS_PRODUCT_ID,
            product_version="202411",
            provider_name="Copernicus Marine Service (CMEMS)",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.125,
            variables_requested=list(_CMEMS_VARIABLES),
            authentication_required=True,
            notes=(
                "DUACS multimission altimeter sea-level product at 0.125° resolution. "
                "Dataset: cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D "
                "(version auto-selected by copernicusmarine client; 202411 as of 2026-09-09). "
                "Variables: sla, adt, ugos, vgos. No depth dimension. "
                "Acquired via copernicusmarine.subset() with monthly chunking. "
                "Credentials from COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars. "
                "Source coverage: 1993-01-01 to present (delayed-time ~6 month lag). "
                "OceanEmbed acquisition period: 2012-01-01 to 2024-12-31."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build a CMEMS acquisition request for the calendar month containing request_date.

        DUACS is acquired monthly. This method builds a request covering the
        full calendar month that contains request_date. Callers should prefer
        date_range_requests() which avoids producing duplicate month requests.

        Parameters
        ----------
        request_date:
            Any date within the month to acquire.
        data_root:
            Root path of the data workspace (kept for interface compatibility).
        """
        year, month = request_date.year, request_date.month
        last_day = calendar.monthrange(year, month)[1]
        return self._build_month_request(
            year, month,
            period_start=date(year, month, 1),
            period_end=date(year, month, last_day),
        )

    def _build_month_request(
        self,
        year: int,
        month: int,
        period_start: date,
        period_end: date,
    ) -> AcquisitionRequest:
        """
        Build a CMEMS acquisition request for one month (or partial month).

        Parameters
        ----------
        year, month:
            Calendar year and month being acquired.
        period_start:
            First date of the period (>= month's first day).
        period_end:
            Last date of the period (<= month's last day).
        """
        start_str = period_start.strftime("%Y-%m-%d")
        end_str = period_end.strftime("%Y-%m-%d")

        # Deterministic filename:
        # Full month  → duacs_YYYY_MM.nc
        # Partial     → duacs_YYYY_MM_DD_DD.nc
        full = _is_full_month(year, month, period_start, period_end)
        if full:
            filename = f"duacs_{year}_{month:02d}.nc"
        else:
            s = period_start.strftime("%d")
            e = period_end.strftime("%d")
            filename = f"duacs_{year}_{month:02d}_{s}_{e}.nc"

        # Relative output path: raw/duacs/YYYY/duacs_YYYY_MM.nc
        relative_path = f"raw/duacs/{year}/{filename}"

        # Sentinel URL — no depth parameter (DUACS has no depth dimension)
        sentinel_url = (
            f"{_CMEMS_URL_SENTINEL}"
            f"?start={start_str}"
            f"&end={end_str}"
            f"&lat_min={self._min_lat}"
            f"&lat_max={self._max_lat}"
            f"&lon_min={self._min_lon}"
            f"&lon_max={self._max_lon}"
            f"&variables={','.join(_CMEMS_VARIABLES)}"
            # NOTE: no &depth= — DUACS has no depth coordinate.
            # parse_cmems_sentinel_url returns depth=None for this URL.
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
                "period_start": start_str,
                "period_end": end_str,
                "acquisition_year": year,
                "acquisition_month": month,
                "is_full_month": full,
                "has_depth": False,
                "acquisition_method": "copernicusmarine.subset",
                "note": (
                    "DUACS has no depth dimension. depth args are not passed "
                    "to copernicusmarine.subset(). OSD-005 (regridding) is open."
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
        Build one AcquisitionRequest per calendar month in [start_date, end_date].

        Validates parameters before producing requests. Raises ValueError with
        a clear message if any parameter is invalid — never silently clamps.

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
            One request per month in the requested range. Partial months at
            the edges use the actual requested boundary, not the full month.
        """
        errors = _validate_duacs_request_params(
            start_date=start_date,
            end_date=end_date,
            min_lat=self._min_lat,
            max_lat=self._max_lat,
            min_lon=self._min_lon,
            max_lon=self._max_lon,
        )
        if errors:
            raise ValueError(
                "Invalid DUACS acquisition request:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        requests = []
        for year, month in _months_in_range(start_date, end_date):
            period_start, period_end = _month_period_overlap(
                year, month, start_date, end_date
            )
            requests.append(
                self._build_month_request(year, month, period_start, period_end)
            )
        return requests

    def validate_request(self, request: AcquisitionRequest) -> list[str]:
        """
        Validate a DUACS AcquisitionRequest.

        Extends base class structural checks with DUACS-specific validation.
        """
        issues = super().validate_request(request)

        if not request.url_or_query.startswith("cmems://"):
            issues.append(
                f"DUACS url_or_query must start with 'cmems://', "
                f"got: {request.url_or_query[:60]!r}"
            )
        for f in ("cmems_dataset_id", "period_start", "period_end"):
            if f not in request.extra:
                issues.append(f"Missing required extra field: '{f}'")
        if not request.spatial_region:
            issues.append("spatial_region is empty.")
        # DUACS must NOT have depth in its sentinel URL
        if "depth=" in request.url_or_query:
            issues.append(
                "DUACS sentinel URL must not contain a depth parameter "
                "(DUACS has no depth dimension)."
            )
        return issues

    def check_credentials(self) -> dict[str, Any]:
        """
        Check whether CMEMS credentials are available in the environment.

        Never logs or returns credential values.
        """
        username_set = bool(
            os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "").strip()
        )
        password_set = bool(
            os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip()
        )
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
                "and COPERNICUSMARINE_SERVICE_PASSWORD. See .env.example."
            )
        return {
            "has_username": username_set,
            "has_password": password_set,
            "credentials_available": available,
            "message": msg,
        }
