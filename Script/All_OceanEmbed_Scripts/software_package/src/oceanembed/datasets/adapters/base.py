"""
OceanEmbed Dataset Adapter — abstract base class and data transfer objects.

Responsibility boundary
-----------------------

The adapter layer sits between the dataset registry and the acquisition engine:

    Dataset Registry  →  Dataset Adapter  →  Acquisition Engine
         (metadata)       (source-specific)    (shared infrastructure)

ADAPTER is responsible for (SOURCE-SPECIFIC):
  - Building the download URL or API request for a given date / date range
  - Resolving expected output filenames and paths
  - Declaring which variables and spatial region are being requested
  - Documenting what processing has NOT been applied (important for traceability)
  - Providing enough metadata to reconstruct provenance after the fact

ACQUISITION ENGINE will be responsible for (GENERIC):
  - Executing HTTP/FTP/API downloads
  - Retry logic with backoff
  - Resume/partial-file detection
  - Checksum / integrity verification
  - Manifest updates
  - Progress reporting
  - Job state management

Scientific guardrail
--------------------
Adapters MUST NOT perform scientific harmonization:

  ✗  Do not regrid source data.
  ✗  Do not resample or aggregate temporally.
  ✗  Do not fill NaN / masked values.
  ✗  Do not convert monthly GLORYS to daily.
  ✗  Do not convert CCMP 6-hourly to daily.
  ✗  Do not change depth coordinates.
  ✗  Do not convert units silently.

Acquisition retrieves the source representation.
Processing transforms it.  These are separate pipeline stages.

Any adapter that violates this boundary introduces a hidden scientific decision
into the acquisition layer.  See docs/development/SCIENTIFIC_DECISION_POLICY.md.

Extending the contract
----------------------
When implementing a real adapter (e.g. OISSTAdapter), subclass DatasetAdapterBase
and implement all abstract methods.  Do not add scientific transformation logic
to the subclass — put that in src/oceanembed/processing/ after a scientific
decision has been recorded.

See docs/datasets/DATASET_REGISTRY.md §5 for the complete design rationale.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDescription:
    """
    Machine-readable description of what source data an adapter targets.

    This is metadata only — it describes the source, not a specific download
    request.  The acquisition engine uses this to populate manifest records
    and provenance metadata.

    Fields
    ------
    dataset_id:
        OceanEmbed dataset identifier (e.g. 'oisst', 'glorys').
    product_id:
        Provider product identifier (e.g. 'AVHRR_OI-NCEI-L4-GLOB-v2.1').
        May be 'UNVERIFIED' if not yet confirmed.
    product_version:
        Dataset version string.  May be 'UNVERIFIED'.
    provider_name:
        Provider organisation name.
    native_temporal_resolution:
        Native temporal resolution description (e.g. 'daily', 'monthly',
        '6-hourly').  This is a product fact, not a processing choice.
    native_spatial_resolution_degrees:
        Native resolution in degrees.  None for non-gridded datasets (Argo).
    variables_requested:
        List of variable names that this adapter retrieves.
    authentication_required:
        Whether credentials are needed to access this source.
    notes:
        Free-text notes about this source.  Include any known caveats
        that the acquisition engine or caller should be aware of.
    """

    dataset_id: str
    product_id: str
    product_version: str
    provider_name: str
    native_temporal_resolution: str
    native_spatial_resolution_degrees: float | None
    variables_requested: list[str]
    authentication_required: bool
    notes: str = ""


@dataclass(frozen=True)
class AcquisitionRequest:
    """
    A single, self-contained acquisition request produced by an adapter.

    The acquisition engine consumes AcquisitionRequest objects.  Each request
    describes exactly what to retrieve from the source — the engine is
    responsible for actually executing the download.

    The adapter produces requests; the engine fulfils them.

    Fields
    ------
    dataset_id:
        OceanEmbed dataset identifier.
    request_date:
        The calendar date being requested.  For datasets with sub-daily or
        multi-day temporal resolution, this is the nominal date.
    url_or_query:
        The download URL, API query string, or other source-specific request
        descriptor.  The exact format is adapter-specific.
    expected_output_path:
        Relative path (below data_root) where the acquired file should be
        stored.  Must NOT be an absolute path.  Uses forward slashes.
    source_description:
        Metadata about the source for provenance tracking.
    spatial_region:
        Dictionary describing the spatial region being requested, e.g.
        {'lat_min': 5.0, 'lat_max': 30.0, 'lon_min': 45.0, 'lon_max': 105.0}.
    extra:
        Adapter-specific additional metadata.  Must not contain scientific
        transformation instructions.
    """

    dataset_id: str
    request_date: date
    url_or_query: str
    expected_output_path: str
    source_description: SourceDescription
    spatial_region: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Guard against hard-coded absolute paths.
        import re
        if re.match(r"^[A-Za-z]:\\", self.expected_output_path):
            raise ValueError(
                f"expected_output_path must be a relative path, "
                f"not an absolute Windows path: {self.expected_output_path!r}"
            )
        if self.expected_output_path.startswith("/"):
            raise ValueError(
                f"expected_output_path must be a relative path, "
                f"not an absolute Unix path: {self.expected_output_path!r}"
            )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DatasetAdapterBase(abc.ABC):
    """
    Abstract base class for all OceanEmbed dataset adapters.

    Every dataset (OISST, GLORYS, CCMP, OSCAR, OISSS, DUACS, GlobCurrent, Argo)
    will have a concrete subclass of this base class.

    Subclasses MUST implement:
    - ``dataset_id``       (property)
    - ``source_description()``
    - ``build_request(request_date, data_root)``

    Subclasses MUST NOT:
    - Perform regridding, temporal aggregation, unit conversion, or any other
      scientific transformation.
    - Make HTTP requests or access the network directly.
    - Access the local filesystem except to construct expected paths.
    - Encode scientific decisions that have not been formally approved.

    See the module docstring for the full responsibility boundary.
    """

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """
        OceanEmbed dataset identifier (e.g. 'oisst', 'glorys').

        Must match the corresponding registry YAML filename stem.
        """

    # ------------------------------------------------------------------
    # Required methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def source_description(self) -> SourceDescription:
        """
        Return a SourceDescription describing what source data this adapter targets.

        This is static metadata about the source product — it does not depend
        on a specific date or date range.  The acquisition engine uses it to
        populate manifest records and provenance metadata.

        Returns
        -------
        SourceDescription
            Metadata about the source: product ID, version, provider, temporal
            resolution, spatial resolution, requested variables, authentication.
        """

    @abc.abstractmethod
    def build_request(
        self,
        request_date: date,
        data_root: str,
    ) -> AcquisitionRequest:
        """
        Build an acquisition request for a single date.

        This method constructs the URL/query and expected output path for
        one date.  It does NOT execute the download.

        Parameters
        ----------
        request_date:
            The calendar date for which to build the request.
        data_root:
            Root path of the data workspace (the ${data_root} configured path).
            Used to construct the expected_output_path without hard-coding.

        Returns
        -------
        AcquisitionRequest
            A self-contained acquisition request for the acquisition engine.

        Notes
        -----
        - The returned path must be relative to data_root, not absolute.
        - Do not retrieve the actual data in this method.
        - Do not apply any scientific transformation.
        - For datasets with sub-daily temporal resolution (e.g. CCMP 6-hourly),
          return the request for the source file(s) that cover the requested
          date — do NOT aggregate or subsample.  Temporal handling is a
          processing decision, not an acquisition decision.
        """

    # ------------------------------------------------------------------
    # Optional hooks (may be overridden)
    # ------------------------------------------------------------------

    def validate_request(self, request: AcquisitionRequest) -> list[str]:
        """
        Validate an AcquisitionRequest before handing it to the engine.

        Returns a list of warning or error messages.  An empty list means
        the request is valid.  The engine may choose to refuse requests
        that produce non-empty validation results.

        The default implementation performs basic structural checks.
        Subclasses may override to add source-specific validation.

        Parameters
        ----------
        request:
            The request to validate.

        Returns
        -------
        list[str]
            Validation messages.  Empty means valid.
        """
        issues: list[str] = []
        if request.dataset_id != self.dataset_id:
            issues.append(
                f"Request dataset_id '{request.dataset_id}' does not match "
                f"adapter dataset_id '{self.dataset_id}'."
            )
        if not request.url_or_query:
            issues.append("url_or_query is empty.")
        if not request.expected_output_path:
            issues.append("expected_output_path is empty.")
        return issues

    def date_range_requests(
        self,
        start_date: date,
        end_date: date,
        data_root: str,
    ) -> list[AcquisitionRequest]:
        """
        Build acquisition requests for every date in [start_date, end_date].

        The default implementation calls build_request() for each date in
        the range.  Subclasses may override for datasets where a single
        request covers multiple dates (e.g. monthly GLORYS files).

        Parameters
        ----------
        start_date:
            First date of the range (inclusive).
        end_date:
            Last date of the range (inclusive).
        data_root:
            Root path of the data workspace.

        Returns
        -------
        list[AcquisitionRequest]
            One request per date in the range.
        """
        from datetime import timedelta

        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})"
            )

        requests = []
        current = start_date
        while current <= end_date:
            requests.append(self.build_request(current, data_root))
            current += timedelta(days=1)
        return requests

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dataset_id={self.dataset_id!r})"
