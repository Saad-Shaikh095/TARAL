"""
OceanEmbed OISST Dataset Adapter.

Constructs acquisition requests for NOAA OISST v2.1 daily global files.

Primary source
--------------
Product  : NOAA OI SST v2.1 (AVHRR_OI-NCEI-L4-GLOB-v2.1)
Provider : NOAA National Centers for Environmental Information (NCEI)
Access   : NCEI HTTPS (public, no authentication required)

URL pattern
-----------
  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/
  v2.1/access/avhrr/{YYYY}{MM}/oisst-avhrr-v02r01.{YYYYMMDD}.nc

This is the canonical, stable NCEI distribution path.  Each file contains the
full global daily field (all variables: sst, anom, err, ice).

The NOAA CoastWatch ERDDAP endpoint (ncdcOisst21Agg_LonPM180) is retained as a
secondary/fallback method but its griddap query interface returned HTTP 404 for
data requests in testing (September 2026), while the NCEI direct path returned
HTTP 200 with a 1.5 MB NetCDF file.

Local storage layout
--------------------
  raw/oisst/YYYY/oisst_YYYYMMDD.nc  (relative to data_root)

Scientific guardrail
--------------------
This adapter constructs URLs and paths only.  It does NOT:
- Regrid the data
- Modify the time coordinate
- Fill missing values
- Convert units
- Subset to the OceanEmbed domain (a processing-stage decision)

The acquisition result is the raw source file as distributed by NOAA.
"""

from __future__ import annotations

from datetime import date

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Primary source — NCEI HTTPS direct file path
# ---------------------------------------------------------------------------

_NCEI_BASE = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation"
    "/v2.1/access/avhrr"
)
# URL pattern: {_NCEI_BASE}/{YYYY}{MM}/oisst-avhrr-v02r01.{YYYYMMDD}.nc
# Confirmed HTTP 200 / Content-Type application/x-netcdf / ~1.5 MB per daily file.

# ---------------------------------------------------------------------------
# Secondary source — NOAA CoastWatch ERDDAP griddap
# (griddap data queries returned HTTP 404 in September 2026 testing;
#  the dataset info endpoint returns HTTP 200 so the dataset may be moved)
# ---------------------------------------------------------------------------

_ERDDAP_BASE = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.nc"
)
_ERDDAP_QUERY_TEMPLATE = (
    "?sst[({date}T12:00:00Z):1:({date}T12:00:00Z)]"
    "[(-89.875):1:(89.875)]"
    "[(-179.875):1:(179.875)]"
)


class OISSTAdapter(DatasetAdapterBase):
    """
    Adapter for NOAA OISST v2.1 daily global sea surface temperature.

    Uses the NCEI HTTPS direct file distribution as the primary source.

    Notes
    -----
    - Authentication: not required (public NOAA data)
    - Temporal resolution: daily (one file per day)
    - Native spatial resolution: 0.25 degree global
    - Longitude convention: -180 to 180 in the NCEI files
    - Output format: NetCDF4 (HDF5-based)
    - Variables in source file: sst, anom, err, ice
    """

    @property
    def dataset_id(self) -> str:
        return "oisst"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="oisst",
            product_id="AVHRR_OI-NCEI-L4-GLOB-v2.1",
            product_version="2.1",
            provider_name="NOAA National Centers for Environmental Information (NCEI)",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.25,
            variables_requested=["sst", "err"],
            authentication_required=False,
            notes=(
                "OISST v2.1 global daily SST at 0.25 degree resolution. "
                "Primary access via NCEI HTTPS direct file distribution. "
                "Time coordinate: 12:00:00 UTC (noon) for each calendar day. "
                "Source file contains: sst, anom, err, ice variables. "
                "Raw source acquisition only — no scientific processing applied."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build an NCEI HTTPS acquisition request for a single OISST daily file.

        Parameters
        ----------
        request_date:
            The calendar date to request.
        data_root:
            Root of the OceanEmbed data workspace.

        Returns
        -------
        AcquisitionRequest
            Contains the NCEI download URL and expected local output path.
        """
        date_compact = request_date.strftime("%Y%m%d")
        year_str = request_date.strftime("%Y")
        month_str = request_date.strftime("%m")
        ym_str = f"{year_str}{month_str}"

        # NCEI canonical path:
        # /v2.1/access/avhrr/YYYYMM/oisst-avhrr-v02r01.YYYYMMDD.nc
        url = f"{_NCEI_BASE}/{ym_str}/oisst-avhrr-v02r01.{date_compact}.nc"

        # Expected relative output path (relative to data_root)
        relative_path = f"raw/oisst/{year_str}/oisst_{date_compact}.nc"

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=url,
            expected_output_path=relative_path,
            source_description=self.source_description(),
            spatial_region={
                "lat_min": -89.875,
                "lat_max": 89.875,
                "lon_min": -179.875,
                "lon_max": 179.875,
            },
            extra={
                "source": "NCEI_HTTPS_primary",
                "ncei_base": _NCEI_BASE,
                "format": "NetCDF4",
                "note": (
                    "Full global file acquired from NCEI. "
                    "Domain subsetting to North Indian Ocean (5-30N, 45-105E) "
                    "is a processing-stage step, not acquisition."
                ),
            },
        )

    def build_request_erddap(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build an ERDDAP griddap request (secondary/fallback path).

        NOTE: As of September 2026 testing, the ERDDAP griddap data endpoint
        returned HTTP 404 for this dataset while the NCEI direct path worked.
        This method is retained for future use if the ERDDAP service resumes.

        Parameters
        ----------
        request_date:
            The calendar date to request.
        data_root:
            Root of the OceanEmbed data workspace.
        """
        date_str = request_date.strftime("%Y-%m-%d")
        date_compact = request_date.strftime("%Y%m%d")
        year_str = request_date.strftime("%Y")

        query = _ERDDAP_QUERY_TEMPLATE.format(date=date_str)
        url = _ERDDAP_BASE + query
        relative_path = f"raw/oisst/{year_str}/oisst_{date_compact}.nc"

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=url,
            expected_output_path=relative_path,
            source_description=self.source_description(),
            extra={
                "source": "ERDDAP_griddap_secondary",
                "erddap_dataset_id": "ncdcOisst21Agg_LonPM180",
                "note": "ERDDAP returned HTTP 404 in Sep 2026; use NCEI primary instead.",
            },
        )
