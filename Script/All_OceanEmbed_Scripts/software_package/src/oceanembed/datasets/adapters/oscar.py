"""
OceanEmbed OSCAR Dataset Adapter.

Constructs acquisition requests for NASA OSCAR L4 OC FINAL v2.0 daily
surface current files via authenticated NASA OPeNDAP (earthaccess).

Primary source
--------------
Collection : C2098858642-POCLOUD
Product    : OSCAR L4 Ocean Surface Current Analysis Final v2.0
Provider   : NASA Jet Propulsion Laboratory / PO.DAAC
Access     : Authenticated NASA OPeNDAP via earthaccess Python library
Variables  : u (eastward, m/s), v (northward, m/s)

Live-verified (2026-09-09, earthaccess 0.17.0)
----------------------------------------------
- Collection ID C2098858642-POCLOUD is valid.
- Daily granules: oscar_currents_final_{YYYYMMDD}
- OPeNDAP URL pattern:
    https://opendap.earthdata.nasa.gov/collections/C2098858642-POCLOUD/
    granules/{granule}.dap.nc4?dap4.ce={constraint}
- Constraint uses DAP4 index-based subsetting (not coordinate-based).
- Variable dims: (time, longitude, latitude) — NO depth dimension.
- Coordinate arrays: lon (0-360°), lat.
- Resolution: 0.25°.
- NIO subset indices: lon[180:420], lat[380:478]
    → lon 45.00°E → 105.00°E, lat 5.25°N → 29.75°N

Longitude convention
--------------------
OSCAR uses 0-360° longitude. The native grid index 180 = 45.00°E, 420 = 105.00°E.
Coordinates do NOT use a -180→180 convention. The adapter converts user-supplied
-180→180 (or 0-360) bbox to the OSCAR 0-360 grid when computing OPeNDAP indices.

Acquisition strategy
--------------------
One request per calendar day. The adapter computes OPeNDAP index bounds from
the requested geographic bbox and constructs a DAP4 constraint URL. The generic
engine's OSCAR dispatch path calls run_oscar_subset() via asyncio.to_thread,
which uses an earthaccess authenticated requests session.

Depth
-----
OSCAR v2.0 has no depth dimension. The product nominally represents currents at
~15 m depth (upper mixed layer). This is NOT 0 m.
OSD-009 (OSCAR representative depth) remains OPEN — do not close it here.

Authentication
--------------
Credentials from EARTHDATA_USERNAME + EARTHDATA_PASSWORD environment variables.
Never hardcoded. Never logged.

Scientific guardrail
--------------------
This adapter constructs requests only. It does NOT:
- Regrid from 0.25° to any other resolution
- Convert lon convention post-acquisition
- Aggregate multiple days
- Fill missing values
- Convert units
- Close OSD-009
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants — all verified from YAML + legacy script + live check
# ---------------------------------------------------------------------------

_COLLECTION_ID = "C2098858642-POCLOUD"
_OPENDAP_BASE = (
    "https://opendap.earthdata.nasa.gov"
    "/collections/{collection_id}"
    "/granules/{granule}.dap.nc4"
)

# Variables confirmed from legacy script validation
_OSCAR_VARIABLES = ["u", "v"]

# Sentinel prefix for engine dispatch
_OSCAR_URL_SENTINEL = "oscar://opendap"

# Source temporal coverage (verified from legacy script + YAML)
_SOURCE_COVERAGE_START = date(2012, 1, 1)
_SOURCE_COVERAGE_END = date(2024, 12, 31)

# Geographic validity bounds (OSCAR native grid is global, but near-polar
# data quality degrades; we use standard geographic limits for validation)
_LAT_MIN_VALID = -80.0   # OSCAR lat range documented as -80 to 80
_LAT_MAX_VALID = 80.0
_LON_MIN_VALID = -180.0
_LON_MAX_VALID = 360.0   # Accept both -180→180 and 0→360

# OSCAR native 0.25° grid parameters — verified from live source (2026-09-09)
# Fetched full lon[0:1439] and lat[0:639] from a real granule:
#   lon[0] = 0.0,   lon[-1] = 359.75,  step = 0.25° (lon[i] = i * 0.25)
#   lat[0] = -89.75, lat[-1] = 70.0,   step = 0.25° (lat[i] = -89.75 + i * 0.25)
#   Total: 1440 lon cells × 640 lat cells
#
# NOTE: This differs from the legacy script's assumed grid constants.
# The legacy script hard-coded LON_START=180 → lon=45.0, LAT_START=380 → lat=5.25.
# With the corrected grid:
#   lon[i] = i * 0.25     → lon=45.0 at i=180 ✓ (same as legacy)
#   lat[i] = -89.75 + i*0.25 → lat=5.25 at i=380  (legacy value)
#                            → lat=5.0  at i=379.9 → round → 380  (effectively same)
# The legacy script's indices were correct for the NIO subset even though
# the assumed first-cell values were wrong.
_OSCAR_RESOLUTION = 0.25
_OSCAR_LON_FIRST = 0.0      # VERIFIED: lon[0] = 0.0
_OSCAR_LAT_FIRST = -89.75   # VERIFIED: lat[0] = -89.75
_OSCAR_LON_N = 1440
_OSCAR_LAT_N = 640


# ---------------------------------------------------------------------------
# Index helpers — convert geographic coords to OSCAR OPeNDAP array indices
# ---------------------------------------------------------------------------

def _lon_to_oscar_360(lon: float) -> float:
    """
    Convert any longitude to the OSCAR 0-360° convention.
    OSCAR's native grid uses 0-360°, not -180→180.
    Input may be -180→180 or 0→360; output is always 0→360.
    """
    return lon % 360.0


def _geo_to_oscar_lon_idx(lon_deg: float) -> int:
    """
    Map a geographic longitude (in OSCAR 0-360° convention) to the nearest
    OPeNDAP array index in the OSCAR native grid.

    VERIFIED grid: lon[i] = i * 0.25   for i in 0..1439
    (lon[0]=0.0, lon[180]=45.0, lon[420]=105.0, lon[-1]=359.75)
    """
    lon_360 = _lon_to_oscar_360(lon_deg)
    idx = round((lon_360 - _OSCAR_LON_FIRST) / _OSCAR_RESOLUTION)
    return max(0, min(_OSCAR_LON_N - 1, idx))


def _geo_to_oscar_lat_idx(lat_deg: float) -> int:
    """
    Map a geographic latitude to the nearest OPeNDAP array index in the
    OSCAR native grid.

    VERIFIED grid: lat[i] = -89.75 + i * 0.25   for i in 0..639
    (lat[0]=-89.75, lat[380]=5.25, lat[399]=10.0, lat[439]=20.0, lat[-1]=70.0)
    """
    idx = round((lat_deg - _OSCAR_LAT_FIRST) / _OSCAR_RESOLUTION)
    return max(0, min(_OSCAR_LAT_N - 1, idx))


def _build_dap4_constraint(
    lon_idx_start: int,
    lon_idx_end: int,
    lat_idx_start: int,
    lat_idx_end: int,
) -> str:
    """
    Build a DAP4 constraint expression for OSCAR u, v, lon, lat.

    OSCAR variable dimensions are (time, longitude, latitude).
    We select time=0 (single-time granule) and the requested index ranges.
    """
    return (
        f"/u[0][{lon_idx_start}:{lon_idx_end}][{lat_idx_start}:{lat_idx_end}];"
        f"/v[0][{lon_idx_start}:{lon_idx_end}][{lat_idx_start}:{lat_idx_end}];"
        f"/lon[{lon_idx_start}:{lon_idx_end}];"
        f"/lat[{lat_idx_start}:{lat_idx_end}]"
    )


def _granule_name(request_date: date) -> str:
    """Return the OSCAR granule name for a given date."""
    return f"oscar_currents_final_{request_date:%Y%m%d}"


def _build_opendap_url(request_date: date, constraint: str) -> str:
    """Build the full authenticated OPeNDAP URL for a date + constraint."""
    granule = _granule_name(request_date)
    base = _OPENDAP_BASE.format(
        collection_id=_COLLECTION_ID,
        granule=granule,
    )
    encoded = quote(constraint, safe="")
    return f"{base}?dap4.ce={encoded}"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_oscar_request_params(
    start_date: date,
    end_date: date,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> list[str]:
    """
    Validate OSCAR request parameters. Returns list of error messages.
    Empty list = valid. Never silently clamps — callers see exact problems.
    """
    errors: list[str] = []

    if start_date > end_date:
        errors.append(
            f"start_date ({start_date}) must be on or before end_date ({end_date})."
        )
    if start_date < _SOURCE_COVERAGE_START:
        errors.append(
            f"start_date ({start_date}) is before OSCAR source coverage "
            f"start ({_SOURCE_COVERAGE_START})."
        )
    if end_date > _SOURCE_COVERAGE_END:
        errors.append(
            f"end_date ({end_date}) is after OSCAR source coverage "
            f"end ({_SOURCE_COVERAGE_END})."
        )
    if min_lat >= max_lat:
        errors.append(
            f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})."
        )
    if min_lat < _LAT_MIN_VALID or min_lat > _LAT_MAX_VALID:
        errors.append(
            f"min_lat ({min_lat}) is outside valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )
    if max_lat < _LAT_MIN_VALID or max_lat > _LAT_MAX_VALID:
        errors.append(
            f"max_lat ({max_lat}) is outside valid range "
            f"[{_LAT_MIN_VALID}, {_LAT_MAX_VALID}]."
        )

    # Longitude: accept -180→180 or 0→360; after normalising to 0→360 min < max required
    lon_min_360 = _lon_to_oscar_360(min_lon)
    lon_max_360 = _lon_to_oscar_360(max_lon)
    if lon_min_360 >= lon_max_360:
        errors.append(
            f"After normalising to 0-360°, min_lon ({lon_min_360:.3f}) must be "
            f"strictly less than max_lon ({lon_max_360:.3f}). "
            f"Input: min_lon={min_lon}, max_lon={max_lon}."
        )

    return errors


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class OSCARAdapter(DatasetAdapterBase):
    """
    Adapter for NASA OSCAR L4 OC FINAL v2.0 daily surface current data.

    Produces one AcquisitionRequest per calendar day in the requested date
    range. Each request carries a sentinel URL (oscar://) and the computed
    OPeNDAP index bounds for the requested geographic region. The engine
    dispatches to run_oscar_subset() for the actual authenticated download.

    Parameters
    ----------
    min_lat, max_lat : float
        Latitude bounds (degrees N). Valid range: -80 to 80.
    min_lon, max_lon : float
        Longitude bounds. Accepts -180→180 or 0→360. The OSCAR native grid
        uses 0-360; conversion is handled internally.

    Notes
    -----
    - Authentication: required. EARTHDATA_USERNAME and EARTHDATA_PASSWORD
      must be set in the environment.
    - Temporal resolution: daily — one request per day.
    - Native resolution: 0.25°.
    - Variables: u (eastward), v (northward).
    - Dimensions: (time=1, longitude, latitude) — NO depth.
    - Longitude in output: OSCAR native 0-360° convention preserved in
      the downloaded file. Do NOT remap here.
    - OSD-009 (OSCAR depth semantics) remains OPEN.
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
        # API date-range limit: 31 days (daily files — matches OISST pattern)
        self._max_job_days = 31

        # Pre-compute OPeNDAP indices from bbox
        self._lon_idx_start = _geo_to_oscar_lon_idx(min_lon)
        self._lon_idx_end = _geo_to_oscar_lon_idx(max_lon)
        self._lat_idx_start = _geo_to_oscar_lat_idx(min_lat)
        self._lat_idx_end = _geo_to_oscar_lat_idx(max_lat)

    @property
    def dataset_id(self) -> str:
        return "oscar"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="oscar",
            product_id=_COLLECTION_ID,
            product_version="2.0",
            provider_name="NASA Jet Propulsion Laboratory / PO.DAAC",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.25,
            variables_requested=list(_OSCAR_VARIABLES),
            authentication_required=True,
            notes=(
                "OSCAR L4 OC FINAL v2.0 — daily total near-surface ocean currents. "
                f"Collection {_COLLECTION_ID}. "
                "Variables: u, v (m/s). Dims: (time, longitude, latitude). No depth. "
                "Nominal 15 m depth (NOT 0 m — OSD-009 open). "
                "Longitude: 0-360° native convention. "
                "Acquired via authenticated NASA OPeNDAP (earthaccess). "
                "Credentials: EARTHDATA_USERNAME + EARTHDATA_PASSWORD. "
                "Source coverage: 2012-01-01 to 2024-12-31."
            ),
        )

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build one OSCAR acquisition request for a specific date.

        The sentinel URL encodes the date, collection, and OPeNDAP index
        bounds. The engine's run_oscar_subset() reads these parameters
        to construct the actual OPeNDAP request URL.
        """
        # Sentinel URL: oscar://<collection>?date=YYYY-MM-DD&...
        sentinel_url = (
            f"{_OSCAR_URL_SENTINEL}"
            f"/{_COLLECTION_ID}"
            f"?date={request_date.isoformat()}"
            f"&lon_idx_start={self._lon_idx_start}"
            f"&lon_idx_end={self._lon_idx_end}"
            f"&lat_idx_start={self._lat_idx_start}"
            f"&lat_idx_end={self._lat_idx_end}"
        )

        # Deterministic relative output path
        year = request_date.year
        relative_path = (
            f"raw/oscar/{year}/"
            f"oscar_currents_final_{request_date:%Y%m%d}_uv.nc"
        )

        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
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
                "collection_id": _COLLECTION_ID,
                "granule": _granule_name(request_date),
                "variables": list(_OSCAR_VARIABLES),
                "lon_idx_start": self._lon_idx_start,
                "lon_idx_end": self._lon_idx_end,
                "lat_idx_start": self._lat_idx_start,
                "lat_idx_end": self._lat_idx_end,
                "acquisition_date": request_date.isoformat(),
                "has_depth": False,
                "nominal_depth_m": 15,
                "depth_note": (
                    "Nominal 15 m depth. NOT a 0 m product. OSD-009 OPEN."
                ),
                "lon_convention": "0-360 (OSCAR native; preserved in output)",
                "acquisition_method": "NASA_OPeNDAP_earthaccess",
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

        Validates parameters before producing requests. Raises ValueError
        with a clear message if any parameter is invalid.
        """
        errors = _validate_oscar_request_params(
            start_date=start_date,
            end_date=end_date,
            min_lat=self._min_lat,
            max_lat=self._max_lat,
            min_lon=self._min_lon,
            max_lon=self._max_lon,
        )
        if errors:
            raise ValueError(
                "Invalid OSCAR acquisition request:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        requests = []
        current = start_date
        while current <= end_date:
            requests.append(self.build_request(current, data_root))
            current += timedelta(days=1)
        return requests

    def validate_request(self, request: AcquisitionRequest) -> list[str]:
        """Validate an OSCAR AcquisitionRequest with OSCAR-specific checks."""
        issues = super().validate_request(request)

        if not request.url_or_query.startswith("oscar://"):
            issues.append(
                f"OSCAR url_or_query must start with 'oscar://', "
                f"got: {request.url_or_query[:60]!r}"
            )
        for f in ("collection_id", "granule", "acquisition_date"):
            if f not in request.extra:
                issues.append(f"Missing required extra field: '{f}'")
        if request.extra.get("has_depth") is True:
            issues.append(
                "OSCAR has no depth dimension — has_depth must be False."
            )
        if not request.spatial_region:
            issues.append("spatial_region is empty.")
        return issues

    def check_credentials(self) -> dict[str, Any]:
        """
        Check whether NASA Earthdata credentials are in the environment.

        Returns a status dict. Credential values are NEVER returned or logged.
        """
        username_set = bool(os.environ.get("EARTHDATA_USERNAME", "").strip())
        password_set = bool(os.environ.get("EARTHDATA_PASSWORD", "").strip())
        available = username_set and password_set

        if available:
            msg = "NASA Earthdata credentials are present in environment."
        elif username_set and not password_set:
            msg = "EARTHDATA_USERNAME is set but EARTHDATA_PASSWORD is missing."
        elif password_set and not username_set:
            msg = "EARTHDATA_PASSWORD is set but EARTHDATA_USERNAME is missing."
        else:
            msg = (
                "NASA Earthdata credentials not found. "
                "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD. "
                "See .env.example for details."
            )
        return {
            "has_username": username_set,
            "has_password": password_set,
            "credentials_available": available,
            "message": msg,
        }


# ---------------------------------------------------------------------------
# Sentinel URL parser
# ---------------------------------------------------------------------------

def parse_oscar_sentinel_url(url: str) -> dict[str, Any]:
    """
    Parse an OSCAR sentinel URL produced by OSCARAdapter.build_request().

    Returns a dict with all parameters needed by run_oscar_subset().
    Raises ValueError if malformed or missing required parameters.

    Example input:
        oscar://opendap/C2098858642-POCLOUD
            ?date=2023-01-01
            &lon_idx_start=180&lon_idx_end=420
            &lat_idx_start=380&lat_idx_end=478
    """
    if not url.startswith("oscar://"):
        raise ValueError(f"Not an OSCAR sentinel URL: {url!r}")
    if "?" not in url:
        raise ValueError(f"OSCAR sentinel URL has no query parameters: {url!r}")

    _, query_part = url.split("?", 1)
    params: dict[str, str] = {}
    for seg in query_part.split("&"):
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        params[k.strip()] = v.strip()

    required = ("date", "lon_idx_start", "lon_idx_end", "lat_idx_start", "lat_idx_end")
    missing = [r for r in required if r not in params]
    if missing:
        raise ValueError(
            f"OSCAR sentinel URL missing required parameters: {missing!r}\n"
            f"URL: {url!r}"
        )

    try:
        request_date = date.fromisoformat(params["date"])
        granule = _granule_name(request_date)
        lon_s = int(params["lon_idx_start"])
        lon_e = int(params["lon_idx_end"])
        lat_s = int(params["lat_idx_start"])
        lat_e = int(params["lat_idx_end"])
        constraint = _build_dap4_constraint(lon_s, lon_e, lat_s, lat_e)
        opendap_url = _build_opendap_url(request_date, constraint)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Failed to parse OSCAR sentinel URL: {exc}\nURL: {url!r}"
        ) from exc

    return {
        "request_date": request_date,
        "granule": granule,
        "opendap_url": opendap_url,
        "lon_idx_start": lon_s,
        "lon_idx_end": lon_e,
        "lat_idx_start": lat_s,
        "lat_idx_end": lat_e,
        "constraint": constraint,
    }
