"""
OceanEmbed Argo Dataset Adapter.

Constructs acquisition requests for raw Argo float core profiles
from the official GDAC sources.

Source facts (VERIFIED 2026-09-10, live S3 profile download)
-------------------------------------------------------------
Provider     : International Argo Program / Global Data Assembly Centres (GDAC)
Role         : Independent observational VALIDATION source (NOT a model input)
Access       : Public — no authentication required
Primary URL  : https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/<file>
Fallback 1   : https://data-argo.ifremer.fr/<file>
Fallback 2   : https://nrlgodae1.nrlmry.navy.mil/pub/outgoing/argo/<file>

Acquisition strategy
--------------------
Index-driven: The official Argo GDAC index file (ar_index_global_prof.txt)
maps each profile to its DAC path and metadata.  The adapter reads this index,
filters to the OceanEmbed NIO domain and project period, and constructs one
AcquisitionRequest per profile.

Acquisition unit: one Argo profile NetCDF file.

Profile file path convention (from index):
    <dac>/<float_id>/profiles/<mode><float_id>_<cycle>.nc
    e.g.  aoml/1900272/profiles/D1900272_314.nc

Local storage layout (relative to data_root):
    raw/argo/<YYYY_MM>/<dac>/<float_id>/profiles/<filename>.nc
    (preserves the GDAC directory structure under a YYYY_MM time bucket)

Verified profile structure (live, 2026-09-10)
----------------------------------------------
File      : aoml/1900272/profiles/D1900272_314.nc  (20,064 bytes)
Mode      : D (delayed)
N_PROF    : 1
N_LEVELS  : 50
PRES      : decibar,      FillValue=99999.0,  shape=(1,50)
TEMP      : degree_Celsius, FillValue=99999.0, shape=(1,50)
PSAL      : psu,          FillValue=99999.0,  shape=(1,50)
JULD      : days since 1950-01-01 00:00:00 UTC, FillValue=999999.0
LATITUDE  : degree_north, FillValue=99999.0
LONGITUDE : degree_east,  FillValue=99999.0
DATA_MODE : (1,) char  — D=delayed, R=real-time, A=adjusted real-time
QC vars   : PRES_QC, TEMP_QC, PSAL_QC, POSITION_QC, JULD_QC (all preserved)

Scientific guardrail
--------------------
THIS ADAPTER CONSTRUCTS ACQUISITION REQUESTS ONLY.  It does NOT:
- Interpolate pressure/depth to target levels
- Map profiles to the OceanEmbed grid
- Compute validation metrics
- Compare against GLORYS
- Remove observations based on QC flags
- Convert pressure to depth
- Subset variables
- Modify any measurement values

All scientific processing and validation decisions belong to the
validation pipeline, after OSD-010 is resolved.

Argo is an independent validation source.
It must NOT be converted into model training targets.
OSD-010 remains OPEN.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

from oceanembed.datasets.adapters.base import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)

# ---------------------------------------------------------------------------
# Source constants (VERIFIED 2026-09-10)
# ---------------------------------------------------------------------------

# Primary: public Argo GDAC S3 mirror (no AWS account required)
_S3_BASE      = "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub/dac/"
# Fallback: official Coriolis GDAC HTTPS
_CORIOLIS_BASE = "https://data-argo.ifremer.fr/"
# Fallback: US GDAC HTTPS (can be slow from some networks)
_USGDAC_BASE  = "https://nrlgodae1.nrlmry.navy.mil/pub/outgoing/argo/"

# Index file name as distributed by Coriolis GDAC
_INDEX_FILENAME = "ar_index_global_prof.txt"

# OceanEmbed NIO domain
_LAT_MIN = 5.0
_LAT_MAX = 30.0
_LON_MIN = 45.0
_LON_MAX = 105.0

# OceanEmbed historical period
_PERIOD_START = date(2012, 1, 1)
_PERIOD_END   = date(2024, 12, 31)

# Minimum file size considered non-empty (1 KB)
_MIN_BYTES = 1024

# Required variables that must be present for a profile to be structurally valid
REQUIRED_VARIABLES = ("PRES", "TEMP")


# ---------------------------------------------------------------------------
# Index record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArgoIndexRecord:
    """
    One row from the Argo GDAC global profile index.

    The index file (ar_index_global_prof.txt) header:
        file,date,latitude,longitude,ocean,profiler_type,institution,date_update

    date format: YYYYMMDDHHMMSS
    """
    file: str           # relative path: <dac>/<float_id>/profiles/<name>.nc
    date_raw: str       # YYYYMMDDHHMMSS
    latitude: float
    longitude: float
    ocean: str
    profiler_type: str
    institution: str
    date_update: str

    @property
    def profile_date(self) -> date:
        """Parse the observation date from the index date string."""
        return datetime.strptime(self.date_raw[:8], "%Y%m%d").date()

    @property
    def float_id(self) -> str:
        """Extract the WMO float ID from the file path."""
        # e.g. aoml/1900272/profiles/D1900272_314.nc → 1900272
        parts = self.file.replace("\\", "/").split("/")
        return parts[1] if len(parts) >= 2 else "unknown"

    @property
    def dac(self) -> str:
        """Extract the DAC name from the file path."""
        return self.file.replace("\\", "/").split("/")[0]

    @property
    def profile_key(self) -> str:
        """Canonical profile key (normalised relative path)."""
        return self.file.replace("\\", "/").strip("/")

    def s3_url(self) -> str:
        return _S3_BASE + self.profile_key

    def coriolis_url(self) -> str:
        return _CORIOLIS_BASE + "dac/" + self.profile_key

    def usgdac_url(self) -> str:
        return _USGDAC_BASE + "dac/" + self.profile_key

    def local_relative_path(self) -> str:
        """
        Relative output path under data_root.

        Layout: raw/argo/<YYYY_MM>/<dac>/<float_id>/profiles/<filename>.nc
        Preserves the GDAC directory structure under a time bucket,
        making it easy to find profiles by month and DAC.
        """
        d = self.profile_date
        ym = f"{d.year:04d}_{d.month:02d}"
        return f"raw/argo/{ym}/{self.profile_key}"


# ---------------------------------------------------------------------------
# Index reader
# ---------------------------------------------------------------------------

def read_nio_index(index_path: Path) -> list[ArgoIndexRecord]:
    """
    Read the Argo GDAC global profile index and return records that fall
    within the OceanEmbed NIO domain and project period.

    Parameters
    ----------
    index_path:
        Path to ar_index_global_prof.txt (local copy).

    Returns
    -------
    list[ArgoIndexRecord]
        Filtered records, sorted by profile_date then file path.
        Duplicates (same file path) are removed.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"Argo index not found: {index_path}\n"
            f"Expected file: {_INDEX_FILENAME}\n"
            f"Place a local copy of the GDAC global profile index at this path."
        )

    seen: dict[str, ArgoIndexRecord] = {}
    malformed = 0

    with open(index_path, encoding="utf-8", errors="replace") as fh:
        # Skip comment lines (start with #), then parse CSV
        reader = csv.DictReader(
            line for line in fh if not line.startswith("#")
        )
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                date_raw = row["date"].strip()
                year = int(date_raw[:4])
                month = int(date_raw[4:6])
                day = int(date_raw[6:8])
                profile_date = date(year, month, day)
            except (ValueError, KeyError, TypeError):
                malformed += 1
                continue

            # Domain + period filter
            if not (_LAT_MIN <= lat <= _LAT_MAX):
                continue
            if not (_LON_MIN <= lon <= _LON_MAX):
                continue
            if not (_PERIOD_START <= profile_date <= _PERIOD_END):
                continue

            file_path = row["file"].replace("\\", "/").strip("/")
            if not file_path.endswith(".nc"):
                continue

            # Validate path structure: <dac>/<float>/<subdir>/<name>.nc
            parts = file_path.split("/")
            if len(parts) < 4:
                continue

            if file_path not in seen:
                seen[file_path] = ArgoIndexRecord(
                    file=file_path,
                    date_raw=date_raw,
                    latitude=lat,
                    longitude=lon,
                    ocean=row.get("ocean", ""),
                    profiler_type=row.get("profiler_type", ""),
                    institution=row.get("institution", ""),
                    date_update=row.get("date_update", ""),
                )

    records = sorted(seen.values(), key=lambda r: (r.profile_date, r.profile_key))
    return records


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ArgoAdapter(DatasetAdapterBase):
    """
    Adapter for raw Argo float core profiles.

    IMPORTANT: Argo is an INDEPENDENT VALIDATION source.
    It must never be treated as model training data.
    OSD-010 remains OPEN.

    This adapter:
    - Reads the local GDAC profile index
    - Filters to the NIO domain and 2012-2024 period
    - Constructs one AcquisitionRequest per profile
    - Uses the S3 primary URL with GDAC HTTPS fallbacks
    - Records full provenance in the request extra metadata

    It does NOT:
    - Interpolate pressure/depth
    - Apply QC filters
    - Grid profiles
    - Compute validation metrics
    """

    def __init__(self, index_path: Path) -> None:
        """
        Parameters
        ----------
        index_path:
            Path to the local ar_index_global_prof.txt copy.
            Typically: <repo_root>/argo index/ar_index_global_prof.txt
        """
        self._index_path = Path(index_path)
        self._records: list[ArgoIndexRecord] | None = None

    @property
    def dataset_id(self) -> str:
        return "argo"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="argo",
            product_id="argo_core_profiles_gdac",
            product_version="GDAC_current",
            provider_name="International Argo Program / Global Data Assembly Centres (GDAC)",
            native_temporal_resolution="irregular (~10-day float cycle)",
            native_spatial_resolution_degrees=None,  # point profiles — not gridded
            variables_requested=["PRES", "TEMP", "PSAL"],
            authentication_required=False,
            notes=(
                "Raw Argo core profiles from GDAC. "
                "Primary source: public S3 mirror (argo-gdac-sandbox). "
                "Argo is an independent validation source — NOT model training data. "
                "OSD-010 open: validation methodology not yet defined. "
                "No interpolation, gridding, or QC filtering applied at acquisition."
            ),
        )

    def _load_index(self) -> list[ArgoIndexRecord]:
        """Load and cache the NIO-filtered index records."""
        if self._records is None:
            self._records = read_nio_index(self._index_path)
        return self._records

    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build acquisition requests for all NIO profiles on a specific date.

        NOTE: Argo profiles are sparse and irregular. Most dates will have
        zero or very few profiles in the NIO domain. This method returns
        the FIRST profile matching the date. Use build_requests_for_date()
        for all profiles on a given day, or date_range_requests() for a
        full date range.

        This method satisfies the DatasetAdapterBase interface contract.
        """
        records = self._load_index()
        matching = [r for r in records if r.profile_date == request_date]
        if not matching:
            # Return a no-op request when no profile exists for this date.
            # The acquisition engine will record 0 files for this date.
            return AcquisitionRequest(
                dataset_id=self.dataset_id,
                request_date=request_date,
                url_or_query="",   # engine skips empty URL
                expected_output_path=f"raw/argo/no_profile_{request_date.isoformat()}.placeholder",
                source_description=self.source_description(),
                extra={"note": f"No NIO Argo profiles in index for {request_date}"},
            )
        return self._record_to_request(matching[0], data_root)

    def build_requests_for_date(
        self,
        request_date: date,
        data_root: str,
    ) -> list[AcquisitionRequest]:
        """Return one AcquisitionRequest per NIO profile on request_date."""
        records = self._load_index()
        return [
            self._record_to_request(r, data_root)
            for r in records
            if r.profile_date == request_date
        ]

    def date_range_requests(
        self,
        start_date: date,
        end_date: date,
        data_root: str,
    ) -> list[AcquisitionRequest]:
        """
        Return one AcquisitionRequest per NIO profile in [start_date, end_date].

        This overrides the default day-by-day iteration from DatasetAdapterBase.
        The index already contains only NIO+period profiles; we simply filter
        by the requested date range.
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )
        records = self._load_index()
        return [
            self._record_to_request(r, data_root)
            for r in records
            if start_date <= r.profile_date <= end_date
        ]

    def profile_count_in_range(
        self,
        start_date: date,
        end_date: date,
    ) -> int:
        """Return the number of NIO profiles in the index for a date range."""
        records = self._load_index()
        return sum(1 for r in records if start_date <= r.profile_date <= end_date)

    def total_nio_profiles(self) -> int:
        """Return total NIO profiles available in 2012–2024."""
        return len(self._load_index())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record_to_request(
        self,
        record: ArgoIndexRecord,
        data_root: str,
    ) -> AcquisitionRequest:
        """Convert one ArgoIndexRecord to an AcquisitionRequest."""
        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=record.profile_date,
            url_or_query=record.s3_url(),
            expected_output_path=record.local_relative_path(),
            source_description=self.source_description(),
            spatial_region={
                "latitude":  record.latitude,
                "longitude": record.longitude,
            },
            extra={
                "profile_key":    record.profile_key,
                "dac":            record.dac,
                "float_id":       record.float_id,
                "institution":    record.institution,
                "profiler_type":  record.profiler_type,
                "ocean":          record.ocean,
                "date_update":    record.date_update,
                "fallback_urls":  [
                    record.coriolis_url(),
                    record.usgdac_url(),
                ],
                "source_primary": "argo_gdac_s3",
                "scientific_note": (
                    "Raw Argo core profile — no interpolation, gridding, "
                    "QC filtering, or depth resampling applied. "
                    "Validation methodology: OSD-010 (OPEN)."
                ),
            },
        )
