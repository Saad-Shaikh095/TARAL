"""
OceanEmbed CCMP Dataset Adapter.

Constructs acquisition requests for RSS CCMP v3.1 daily global 6-hourly
wind analysis files via direct RSS HTTPS (public, no authentication).

Primary source
--------------
Product   : RSS CCMP 6-Hourly 10 Meter Surface Winds Level 4 Version 3.1
Short ID  : CCMP_WINDS_10M6HR_L4_V3.1
DOI       : 10.5067/CCMP-6HW10M-L4V31
Provider  : Remote Sensing Systems (RSS)
Distributor: NASA/JPL/PO.DAAC
Access    : RSS public HTTPS (no authentication required)
Version   : 3.1

URL pattern (verified 2026-09-10)
----------------------------------
  https://data.remss.com/ccmp/v03.1/Y{YYYY}/M{MM}/
  CCMP_Wind_Analysis_{YYYYMMDD}_V03.1_L4.nc

Server coverage verified: 1993 through 2026.
OceanEmbed acquisition period: 2012-01-01 through 2024-12-31.

Verified source characteristics (live file inspection 2026-09-10)
------------------------------------------------------------------
  Dimensions   : time=4, latitude=720, longitude=1440
  Variables    : uwnd, vwnd, ws, nobs
  uwnd/vwnd    : eastward/northward wind at 10 m, units m s-1
                 standard_name: eastward_wind / northward_wind
  ws           : wind speed scalar, m s-1
  nobs         : number of observations (float32, no units)
  Coordinate names : latitude, longitude, time (long form)
  Longitude    : 0.125 to 359.875 (0-360 convention)
  Latitude     : -89.875 to 89.875 (ascending)
  Grid spacing : 0.25 degrees (cell-centred at +0.125 offset)
  _FillValue   : -9999.0 (becomes NaN with mask_and_scale=True)
  File size    : ~31.87 MB per daily file (global, all 4 times)

CRITICAL TEMPORAL STRUCTURE
----------------------------
Each daily file contains EXACTLY 4 independent analysis records:

    time[0] = 00:00 UTC
    time[1] = 06:00 UTC
    time[2] = 12:00 UTC
    time[3] = 18:00 UTC

These are 6-hourly variational analysis (VAM) fields, NOT a daily mean.

The acquisition adapter MUST preserve all four records.

    MUST NOT: select one timestamp
    MUST NOT: average the four timestamps
    MUST NOT: drop three timestamps
    MUST NOT: perform temporal aggregation

OSD-007 (CCMP daily aggregation method) remains OPEN.
Temporal selection/aggregation is a processing-layer decision.

Wind semantics
--------------
CCMP provides 10-metre wind VELOCITY components in m/s.
It does NOT provide wind stress.
Do NOT derive stress during acquisition.
OSD-013 (velocity vs stress for model input) remains open at the model level.
The source-level fact is confirmed: velocity, not stress.

Longitude convention
--------------------
CCMP uses 0-360 longitude (first cell at 0.125°).
This differs from OISSS and DUACS (-180 to +180).
The raw stored file retains the native 0-360 convention.
Do NOT convert longitude in the stored file.
OSD-005 (canonical grid) remains OPEN.

Local storage layout
--------------------
  raw/ccmp/{YYYY}/CCMP_Wind_Analysis_{YYYYMMDD}_V03.1_L4.nc
  (relative to data_root)

The full global file (time=4, latitude=720, longitude=1440) is stored.
The NIO subset (5-30N, 45-105E) is NOT applied during acquisition.
Subsetting is a processing-layer responsibility.

Scientific guardrail
--------------------
This adapter constructs URLs and paths only.  It does NOT:
  - Select one of the 4 time records
  - Average or aggregate the 4 time records
  - Subset to the NIO domain
  - Regrid or interpolate
  - Convert longitude convention in the stored file
  - Fill NaN / replace -9999.0
  - Derive wind stress
  - Drop auxiliary variables (ws, nobs)
  - Perform any scientific processing

Open scientific decisions preserved
------------------------------------
  OSD-005: Canonical OceanEmbed grid
  OSD-007: CCMP daily aggregation (OPEN — 4 records preserved raw)
  OSD-008: OceanEmbed Day definition
  OSD-013: Wind velocity vs stress (source: velocity; model use: open)
  OSD-016: Temporal alignment for non-daily datasets
"""

from __future__ import annotations

from datetime import date, timedelta

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants — all verified from live source inspection 2026-09-10
# ---------------------------------------------------------------------------

_RSS_BASE = "https://data.remss.com/ccmp/v03.1"

# Filename template matches the RSS distribution exactly.
# Example: CCMP_Wind_Analysis_20230101_V03.1_L4.nc
_FILENAME_TEMPLATE = "CCMP_Wind_Analysis_{date_compact}_V03.1_L4.nc"

# Source temporal coverage (verified: RSS server has 1993-2026)
# OceanEmbed acquisition range: 2012-01-01 to 2024-12-31
_SOURCE_COVERAGE_START = date(2012, 1, 1)
_SOURCE_COVERAGE_END = date(2024, 12, 31)

# Verified native temporal structure
_TIME_RECORDS_PER_FILE = 4
_ANALYSIS_TIMES_UTC = ("00:00", "06:00", "12:00", "18:00")


class CCMPAdapter(DatasetAdapterBase):
    """
    Adapter for RSS CCMP v3.1 6-hourly 10m surface wind analysis.

    Uses the RSS public HTTPS direct file distribution — no authentication.

    Notes
    -----
    - Authentication: not required (public RSS server)
    - Temporal structure: 4 × 6-hourly analysis records per daily file
      (00Z, 06Z, 12Z, 18Z) — all four are preserved in raw acquisition
    - Native spatial resolution: 0.25 degree global
    - Grid dimensions: time=4, latitude=720, longitude=1440
    - Longitude convention: 0-360 (0.125 to 359.875)
    - Coordinate names: latitude, longitude, time (long form)
    - Variables: uwnd, vwnd, ws, nobs
    - uwnd/vwnd: 10m wind velocity (m/s) — NOT stress
    - Fill value: -9999.0 (becomes NaN with mask_and_scale=True)
    - Raw global file stored; no NIO subsetting, no temporal selection
    - OSD-007 OPEN: temporal aggregation is a processing-layer decision
    """

    @property
    def dataset_id(self) -> str:
        return "ccmp"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="ccmp",
            product_id="CCMP_WINDS_10M6HR_L4_V3.1",
            product_version="3.1",
            provider_name=(
                "Remote Sensing Systems (RSS); "
                "distributed by NASA/JPL/PO.DAAC"
            ),
            native_temporal_resolution="6-hourly",
            native_spatial_resolution_degrees=0.25,
            variables_requested=["uwnd", "vwnd", "ws", "nobs"],
            authentication_required=False,
            notes=(
                "CCMP v3.1 6-hourly 10m wind analysis at 0.25 degree global resolution. "
                "Public RSS HTTPS — no credentials required. "
                "CRITICAL: 4 analysis times per daily file: "
                "00:00, 06:00, 12:00, 18:00 UTC. "
                "All 4 records preserved in raw acquisition. "
                "OSD-007 (daily aggregation) OPEN — temporal selection "
                "belongs to processing layer. "
                "uwnd/vwnd = 10m wind velocity (m/s), NOT stress. "
                "OSD-013 model-level decision OPEN. "
                "Longitude convention: 0-360 (0.125-359.875). "
                "Fill value: -9999.0. "
                "Raw global file (time=4, lat=720, lon=1440); no NIO subset."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build an RSS HTTPS acquisition request for a single CCMP daily file.

        The returned request targets the full global daily file containing
        all 4 × 6-hourly analysis records.  No spatial or temporal subsetting
        is applied.

        Parameters
        ----------
        request_date:
            The calendar date to request.
        data_root:
            Root of the OceanEmbed data workspace (for path context only;
            the returned path is always relative).

        Returns
        -------
        AcquisitionRequest
            Contains the RSS download URL and the deterministic relative
            output path.  url_or_query is a plain HTTPS URL routed through
            the existing download_file() engine path.
        """
        date_compact = request_date.strftime("%Y%m%d")  # YYYYMMDD for filename
        year_str = request_date.strftime("%Y")          # YYYY for directory
        month_str = request_date.strftime("%m")         # MM for directory

        filename = _FILENAME_TEMPLATE.format(date_compact=date_compact)

        # RSS URL: {base}/Y{YYYY}/M{MM}/{filename}
        url = f"{_RSS_BASE}/Y{year_str}/M{month_str}/{filename}"

        # Relative output path (never absolute — relative to data_root)
        relative_path = f"raw/ccmp/{year_str}/{filename}"

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=url,
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                # Global extent — full 0-360 longitude, no NIO subset
                "lat_min": -89.875,
                "lat_max": 89.875,
                "lon_min": 0.125,     # native 0-360 convention
                "lon_max": 359.875,
            },
            extra={
                "source": "RSS_HTTPS_public",
                "rss_base": _RSS_BASE,
                "filename": filename,
                "format": "NetCDF4",
                "authentication": "none",
                "time_records_per_file": _TIME_RECORDS_PER_FILE,
                "analysis_times_utc": list(_ANALYSIS_TIMES_UTC),
                "global_time_cells": 4,
                "global_lat_cells": 720,
                "global_lon_cells": 1440,
                "fill_value": -9999.0,
                "lon_convention": "0-360",
                "reference_height_m": 10,
                "wind_type": "velocity",
                "wind_note": (
                    "uwnd/vwnd are 10m wind VELOCITY components (m/s). "
                    "NOT wind stress. OSD-013 model-level decision OPEN."
                ),
                "temporal_note": (
                    "File contains 4 × 6-hourly records: 00Z, 06Z, 12Z, 18Z. "
                    "All 4 preserved in raw acquisition. "
                    "OSD-007 (daily aggregation) OPEN — processing-layer decision."
                ),
                "subsetting_note": (
                    "Full global file acquired (time=4, lat=720, lon=1440). "
                    "NIO subsetting (5-30N, 45-105E) is a processing-layer "
                    "responsibility. OSD-005 and OSD-007 remain open."
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
        Build one AcquisitionRequest per day in [start_date, end_date].

        CCMP is delivered as one daily file per calendar day.  Each file
        contains all 4 × 6-hourly analysis records.  This method generates
        one request per day — not four.

        Parameters
        ----------
        start_date:
            First date of the requested range (inclusive).
        end_date:
            Last date of the requested range (inclusive).
        data_root:
            Root of the OceanEmbed data workspace.

        Returns
        -------
        list[AcquisitionRequest]
            One request per day, in chronological order.

        Raises
        ------
        ValueError
            If start_date > end_date.
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be on or before "
                f"end_date ({end_date})."
            )

        requests = []
        current = start_date
        while current <= end_date:
            requests.append(self.build_request(current, data_root))
            current += timedelta(days=1)
        return requests
