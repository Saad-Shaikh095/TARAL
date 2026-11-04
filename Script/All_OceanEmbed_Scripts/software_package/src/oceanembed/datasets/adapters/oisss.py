"""
OceanEmbed OISSS Dataset Adapter.

Constructs acquisition requests for ESR OISSS v3.0 daily global files.

Primary source
--------------
Product  : Multi-Mission Optimally Interpolated Sea Surface Salinity
           Daily Global Dataset v3.0
Short ID : OISSS_L4_multimission_daily_v3
Provider : Earth & Space Research (ESR), Seattle, WA
Access   : ESR HTTPS (public, no authentication required)
Version  : v3.0

URL pattern (verified 2026-09-09)
----------------------------------
  https://oisss.esr.org/data/V3/Daily/{YYYY}/
  OISSS_L4_multimission_global_daily_v3.0_{YYYY-MM-DD}.nc

Server coverage verified: 2011 through 2026.
OceanEmbed acquisition period: 2012-01-01 through 2024-12-31.

Verified source characteristics (live file inspection 2026-09-09)
------------------------------------------------------------------
  Dimensions   : latitude=720, longitude=1440, time=1
  Coordinate names : latitude, longitude, time  (long form)
  Longitude    : -179.875 to +179.875 (-180 to +180 convention)
  Latitude     : -89.875 to +89.875 (ascending)
  Grid spacing : 0.25 degrees (cell-centred at ±0.125 offset)
  Variables    : sss, sss_empirical_uncertainty, sss_formal_uncertainty
  sss dtype    : float64
  _FillValue   : -9999.0  (becomes NaN with mask_and_scale=True)
  File size    : ~12.5 MB per daily file

Unit attribute discrepancy (UNRESOLVED)
---------------------------------------
The live source file reports:

    sss.units = "1e-3"

The registry previously documented native_units as "PSU".

These two representations are NOT the same and the scientific interpretation
has NOT been resolved.

This adapter does NOT apply any unit scaling or conversion.
The raw source values are stored exactly as distributed by ESR.
The unit ambiguity is a processing-layer concern pending scientific decision.

See: configs/datasets/oisss.yaml  →  source_unit_attribute / native_units

Local storage layout
--------------------
  raw/oisss/{YYYY}/OISSS_L4_multimission_global_daily_v3.0_{YYYY-MM-DD}.nc
  (relative to data_root)

The full global file (720×1440) is stored.
The NIO subset (5–30°N, 45–105°E) is NOT applied during acquisition.
Subsetting is a processing-layer responsibility, consistent with the
OISST adapter architecture and OSD-005/OSD-006 being open.

Scientific guardrail
--------------------
This adapter constructs URLs and paths only.  It does NOT:
  - Subset to the NIO domain
  - Regrid
  - Interpolate
  - Fill NaN / replace -9999.0
  - Convert or scale SSS values
  - Drop uncertainty variables
  - Aggregate time
  - Perform any scientific processing
"""

from __future__ import annotations

from datetime import date, timedelta

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants — all verified from live source inspection 2026-09-09
# ---------------------------------------------------------------------------

_ESR_BASE = "https://oisss.esr.org/data/V3/Daily"

# Filename template matches the ESR distribution exactly.
_FILENAME_TEMPLATE = "OISSS_L4_multimission_global_daily_v3.0_{date_str}.nc"

# Source temporal coverage (verified: server has 2011–2026)
# OceanEmbed acquisition range: 2012-01-01 to 2024-12-31
_SOURCE_COVERAGE_START = date(2012, 1, 1)
_SOURCE_COVERAGE_END = date(2024, 12, 31)


class OISSSAdapter(DatasetAdapterBase):
    """
    Adapter for ESR OISSS v3.0 daily global sea surface salinity.

    Uses the ESR public HTTPS direct file distribution.

    Notes
    -----
    - Authentication: not required (public ESR server)
    - Temporal resolution: daily (one file per day)
    - Native spatial resolution: 0.25 degree global
    - Grid dimensions: latitude=720, longitude=1440, time=1
    - Longitude convention: -180 to +180
    - Coordinate names: latitude, longitude, time  (long form)
    - Variables in source file: sss, sss_empirical_uncertainty,
      sss_formal_uncertainty
    - Fill value: -9999.0  (becomes NaN with mask_and_scale=True)
    - Unit attribute on sss: "1e-3"  (scientific interpretation unresolved)
    - Raw global file stored; no NIO subsetting at acquisition time
    """

    @property
    def dataset_id(self) -> str:
        return "oisss"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="oisss",
            product_id="OISSS_L4_multimission_daily_v3",
            product_version="3.0",
            provider_name=(
                "Earth & Space Research (ESR), Seattle WA; "
                "IPRC/SOEST University of Hawaii; "
                "Remote Sensing Systems (RSS)"
            ),
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.25,
            variables_requested=[
                "sss",
                "sss_empirical_uncertainty",
                "sss_formal_uncertainty",
            ],
            authentication_required=False,
            notes=(
                "OISSS v3.0 daily global SSS at 0.25 degree resolution. "
                "Public ESR HTTPS access — no credentials required. "
                "Coordinate names: latitude, longitude, time (long form). "
                "Longitude convention: -180 to +180. "
                "Fill value: -9999.0 (NaN after mask_and_scale=True). "
                "Unit attribute on sss: '1e-3' — scientific interpretation "
                "unresolved; raw value stored without conversion. "
                "Raw global file acquired (720×1440); "
                "NIO subsetting is a processing-layer concern (OSD-005/OSD-006)."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build an ESR HTTPS acquisition request for a single OISSS daily file.

        The full global file is requested.  No spatial subsetting is applied.

        Parameters
        ----------
        request_date:
            The calendar date to request.
        data_root:
            Root of the OceanEmbed data workspace (used for path context only;
            the returned path is always relative).

        Returns
        -------
        AcquisitionRequest
            Contains the ESR download URL and expected local output path.
        """
        date_str = request_date.strftime("%Y-%m-%d")
        year_str = request_date.strftime("%Y")

        filename = _FILENAME_TEMPLATE.format(date_str=date_str)

        # ESR URL: {base}/{YYYY}/{filename}
        url = f"{_ESR_BASE}/{year_str}/{filename}"

        # Relative output path (never absolute — relative to data_root)
        relative_path = f"raw/oisss/{year_str}/{filename}"

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=url,
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                # Global extent — no NIO subset at acquisition time
                "lat_min": -89.875,
                "lat_max": 89.875,
                "lon_min": -179.875,
                "lon_max": 179.875,
            },
            extra={
                "source": "ESR_HTTPS_public",
                "esr_base": _ESR_BASE,
                "filename": filename,
                "format": "NetCDF4",
                "authentication": "none",
                "global_lat_cells": 720,
                "global_lon_cells": 1440,
                "fill_value": -9999.0,
                "unit_attribute": "1e-3",
                "unit_note": (
                    "Live source reports sss.units='1e-3'. "
                    "Scientific interpretation is unresolved. "
                    "Raw values stored without conversion."
                ),
                "subsetting_note": (
                    "Full global file acquired (720×1440). "
                    "NIO subsetting (5–30°N, 45–105°E) is a processing-layer "
                    "responsibility. OSD-005 and OSD-006 remain open."
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

        OISSS is a daily product: one file per calendar day.

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
