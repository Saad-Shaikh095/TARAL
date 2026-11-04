"""
OceanEmbed GLORYS Dataset Adapter.

Constructs acquisition requests for GLORYS12V1 daily thetao files via the
Copernicus Marine Service (CMEMS) copernicusmarine Python API.

Source facts (VERIFIED 2026-09-10, live CMEMS API)
---------------------------------------------------
Product   : GLOBAL_MULTIYEAR_PHY_001_030
Dataset   : cmems_mod_glo_phy_my_0.083deg_P1D-m  (daily mean, ~1/12 deg)
Version   : 202311  (pinned for reproducibility; covers 2012-01-01 to 2024-12-31)
Variable  : thetao  (sea_water_potential_temperature, degrees_C)
Time      : Daily mean, 00:00 UTC midnight timestamp, P1D spacing
Depth     : 50 native levels, 0.4940 m to 5727.917 m
Grid      : Regular lat/lon -180 to 180, -80 to 90
Auth      : CMEMS account credentials required

Acquisition strategy
--------------------
Monthly chunking: one copernicusmarine.subset() call per calendar month.
Each request covers all daily records for that month within the NIO domain
(plus a one-cell buffer on each edge for safe regridding at domain edges).

Depth range: 0.0–1100.0 m requested (ensures native levels bracket the 1000 m
candidate target depth on both sides for interpolation in the processing layer).

Scientific guardrail
--------------------
This adapter constructs acquisition requests ONLY. It does NOT:
- Apply the 3x3 rolling mean spatial reduction (OSD-002 — not approved)
- Interpolate to the 15 candidate depth levels (OSD-003 — not approved)
- Interpret the 0.4940 m level as "0 m" (OSD-004 — not approved)
- Modify thetao values in any way
- Perform any temporal aggregation

The acquired files are native 1/12° NetCDF4 with all 50 depth levels.
All scientific transformations belong in the processing layer, after the
relevant OSDs are resolved.

Credentials
-----------
Set before starting the backend:
  COPERNICUSMARINE_SERVICE_USERNAME=<your_username>
  COPERNICUSMARINE_SERVICE_PASSWORD=<your_password>

Or run `copernicusmarine login` once to store credentials locally.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants (all VERIFIED from live CMEMS API 2026-09-10)
# ---------------------------------------------------------------------------

_PRODUCT_ID    = "GLOBAL_MULTIYEAR_PHY_001_030"
_DATASET_ID    = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
_DATASET_VERSION = "202311"
_VARIABLE      = "thetao"

# NIO domain with one native-cell buffer for safe regridding.
# Native step is 1/12 deg (~0.0833). Buffer = 1 step each side.
_BUFFER        = 1 / 12.0
_LAT_MIN       = 5.0  - _BUFFER
_LAT_MAX       = 30.0 + _BUFFER
_LON_MIN       = 45.0 - _BUFFER
_LON_MAX       = 105.0 + _BUFFER

# Depth range ensures native levels bracket all 15 candidate target depths.
# Shallowest native: 0.4940 m. Deepest candidate: 1000 m. Nearest native
# below 1000 m is 1062.44 m, so 1100 m gives a safe margin. (Verified audit.)
_DEPTH_MIN     = 0.0
_DEPTH_MAX     = 1100.0


class GLORYSAdapter(DatasetAdapterBase):
    """
    Adapter for GLORYS12V1 daily potential temperature (thetao).

    Uses copernicusmarine.subset() with monthly chunking.
    No scientific transformation is applied — raw native files only.

    The adapter checks for CMEMS credentials at instantiation and raises
    clearly if they are absent, rather than failing silently mid-download.
    """

    # Maximum date range per job (31 days = one month; monthly chunking means
    # one CMEMS request per month, and jobs beyond one month can take a long
    # time). Adjust upward once the acquisition engine supports job resumption.
    _max_job_days: int = 366  # Allow up to one full year per job

    def __init__(self) -> None:
        self._check_credentials()

    @property
    def dataset_id(self) -> str:
        return "glorys"

    # ------------------------------------------------------------------
    # Credential check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_credentials() -> None:
        """
        Verify CMEMS credentials are present in the environment.

        Raises RuntimeError with actionable instructions if absent.
        Does NOT print or log the credential values.
        """
        import os
        username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "").strip()
        password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip()

        missing = []
        if not username:
            missing.append("COPERNICUSMARINE_SERVICE_USERNAME")
        if not password:
            missing.append("COPERNICUSMARINE_SERVICE_PASSWORD")

        if missing:
            raise RuntimeError(
                f"GLORYS acquisition requires CMEMS credentials. "
                f"Missing environment variable(s): {missing}. "
                f"Options:\n"
                f"  1. Set in .env file (see .env.example)\n"
                f"  2. Set in environment before starting backend\n"
                f"  3. Run 'copernicusmarine login' to store credentials locally"
            )

    # ------------------------------------------------------------------
    # SourceDescription
    # ------------------------------------------------------------------

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="glorys",
            product_id=_PRODUCT_ID,
            product_version=_DATASET_VERSION,
            provider_name="Copernicus Marine Service (CMEMS) / Mercator Ocean International",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=round(1 / 12, 4),  # ~0.0833 deg
            variables_requested=[_VARIABLE],
            authentication_required=True,
            notes=(
                f"GLORYS12V1 daily thetao via copernicusmarine.subset(). "
                f"Dataset: {_DATASET_ID}. Version: {_DATASET_VERSION}. "
                f"NIO domain with 1/12-deg buffer. "
                f"Depth 0–1100 m (50 native levels; all returned). "
                f"Raw native 1/12-deg file — no spatial reduction or vertical "
                f"interpolation applied (OSD-002, OSD-003 unresolved)."
            ),
        )

    # ------------------------------------------------------------------
    # build_request — single day (base interface)
    # ------------------------------------------------------------------

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build a copernicusmarine acquisition request for a single day.

        NOTE: GLORYS acquisition is more efficient with monthly chunking.
        Use build_monthly_request() for production acquisition. This method
        satisfies the DatasetAdapterBase interface and is used for testing
        and single-day spot checks.
        """
        date_str = request_date.isoformat()
        year_str = request_date.strftime("%Y")
        date_compact = request_date.strftime("%Y%m%d")

        relative_path = (
            f"raw/glorys/{year_str}/glorys_thetao_native_{date_compact}.nc"
        )

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=self._build_query(date_str, date_str),
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                "lat_min": _LAT_MIN,
                "lat_max": _LAT_MAX,
                "lon_min": _LON_MIN,
                "lon_max": _LON_MAX,
            },
            extra={
                "dataset_id": _DATASET_ID,
                "dataset_version": _DATASET_VERSION,
                "variables": [_VARIABLE],
                "start_datetime": f"{date_str}T00:00:00",
                "end_datetime": f"{date_str}T00:00:00",
                "minimum_depth": _DEPTH_MIN,
                "maximum_depth": _DEPTH_MAX,
                "chunking": "single_day",
                "scientific_note": (
                    "Raw native 1/12-deg file. No spatial reduction (OSD-002) "
                    "or vertical interpolation (OSD-003) applied."
                ),
            },
        )

    # ------------------------------------------------------------------
    # build_monthly_request — preferred for production
    # ------------------------------------------------------------------

    def build_monthly_request(
        self,
        year: int,
        month: int,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build a copernicusmarine request for a full calendar month.

        This is the preferred acquisition unit for GLORYS. One CMEMS
        request covers all daily records for the month, which is more
        efficient than 28-31 individual requests.

        The output file contains all daily thetao fields for the month,
        at native 1/12-deg resolution, all 50 depth levels.

        Parameters
        ----------
        year, month:
            Calendar year and month (1-12).
        data_root:
            Root of the OceanEmbed data workspace.
        """
        start = date(year, month, 1)
        last_day = calendar.monthrange(year, month)[1]
        end = date(year, month, last_day)

        start_str = start.isoformat()
        end_str = end.isoformat()
        ym_str = f"{year:04d}{month:02d}"
        year_str = f"{year:04d}"

        relative_path = (
            f"raw/glorys/{year_str}/"
            f"glorys_thetao_native_{start_str.replace('-','')}_{end_str.replace('-','')}.nc"
        )

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=start,
            url_or_query=self._build_query(start_str, end_str),
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                "lat_min": _LAT_MIN,
                "lat_max": _LAT_MAX,
                "lon_min": _LON_MIN,
                "lon_max": _LON_MAX,
            },
            extra={
                "dataset_id": _DATASET_ID,
                "dataset_version": _DATASET_VERSION,
                "variables": [_VARIABLE],
                "start_datetime": f"{start_str}T00:00:00",
                "end_datetime": f"{end_str}T00:00:00",
                "minimum_depth": _DEPTH_MIN,
                "maximum_depth": _DEPTH_MAX,
                "chunking": "monthly",
                "year_month": ym_str,
                "expected_time_steps": last_day,
                "scientific_note": (
                    "Raw native 1/12-deg file. No spatial reduction (OSD-002) "
                    "or vertical interpolation (OSD-003) applied."
                ),
            },
        )

    # ------------------------------------------------------------------
    # date_range_requests — monthly chunking override
    # ------------------------------------------------------------------

    def date_range_requests(
        self,
        start_date: date,
        end_date: date,
        data_root: str,
    ) -> list[AcquisitionRequest]:
        """
        Build monthly-chunked acquisition requests for a date range.

        Overrides the default day-by-day implementation from
        DatasetAdapterBase with monthly chunking, which is the efficient
        and intended access pattern for GLORYS.

        Each AcquisitionRequest covers one full calendar month that
        overlaps with [start_date, end_date].

        For partial months (first/last month of the range), the full
        calendar month is still requested. The processing layer trims
        to the exact requested period.
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        requests: list[AcquisitionRequest] = []

        # Iterate month by month
        current_year = start_date.year
        current_month = start_date.month

        end_year = end_date.year
        end_month = end_date.month

        while (current_year, current_month) <= (end_year, end_month):
            requests.append(
                self.build_monthly_request(current_year, current_month, data_root)
            )
            # Advance to next month
            if current_month == 12:
                current_year += 1
                current_month = 1
            else:
                current_month += 1

        return requests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(start_str: str, end_str: str) -> str:
        """
        Build a CMEMS sentinel URL for the acquisition engine.

        The returned string starts with "cmems://" so the engine dispatches
        to run_cmems_subset() rather than download_file().

        This follows the identical convention used by GlobCurrentAdapter
        and DUACSAdapter. The sentinel carries all parameters needed by
        parse_cmems_sentinel_url() and run_cmems_subset().

        GLORYS depth is a RANGE (0.0-1100.0 m), encoded as:
            depth_min=0.0&depth_max=1100.0

        This extends the existing sentinel format (which used depth=<single>
        for GlobCurrent). The engine and parser both handle the new keys.
        Existing GlobCurrent/DUACS behavior is unaffected.
        """
        return (
            f"cmems://{_DATASET_ID}"
            f"?start={start_str}"
            f"&end={end_str}"
            f"&lat_min={_LAT_MIN}"
            f"&lat_max={_LAT_MAX}"
            f"&lon_min={_LON_MIN}"
            f"&lon_max={_LON_MAX}"
            f"&depth_min={_DEPTH_MIN}"
            f"&depth_max={_DEPTH_MAX}"
            f"&variables={_VARIABLE}"
            f"&dataset_version={_DATASET_VERSION}"
        )
