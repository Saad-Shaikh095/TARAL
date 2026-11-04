"""
Unit tests for the OISSS dataset adapter.

Coverage
--------
- Adapter identity (dataset_id, source_description)
- URL construction for known dates
- Date range produces correct number of daily requests
- Output paths are deterministic and relative
- Source metadata is correct
- No scientific processing occurs (no subsetting, no unit conversion,
  no NaN filling, no regridding)
- Unit discrepancy is documented, not silently resolved
- API registration is correct
- Regression: existing OISST adapter unaffected

All tests are offline. No network access.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from oceanembed.datasets.adapters.oisss import (
    OISSSAdapter,
    _ESR_BASE,
    _FILENAME_TEMPLATE,
    _SOURCE_COVERAGE_START,
    _SOURCE_COVERAGE_END,
)
from oceanembed.datasets.adapters.base import AcquisitionRequest, SourceDescription


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return OISSSAdapter()


@pytest.fixture
def data_root(tmp_path):
    return str(tmp_path)


# ===========================================================================
# Adapter identity
# ===========================================================================

class TestAdapterIdentity:
    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "oisss"

    def test_repr_contains_class_and_id(self, adapter):
        r = repr(adapter)
        assert "OISSSAdapter" in r
        assert "oisss" in r

    def test_source_description_returns_correct_type(self, adapter):
        assert isinstance(adapter.source_description(), SourceDescription)

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "oisss"

    def test_source_description_product_id(self, adapter):
        assert adapter.source_description().product_id == "OISSS_L4_multimission_daily_v3"

    def test_source_description_version(self, adapter):
        assert adapter.source_description().product_version == "3.0"

    def test_source_description_provider_contains_esr(self, adapter):
        assert "ESR" in adapter.source_description().provider_name or \
               "Earth" in adapter.source_description().provider_name

    def test_auth_not_required(self, adapter):
        assert adapter.source_description().authentication_required is False

    def test_temporal_resolution_daily(self, adapter):
        assert adapter.source_description().native_temporal_resolution == "daily"

    def test_spatial_resolution_025(self, adapter):
        assert adapter.source_description().native_spatial_resolution_degrees == 0.25

    def test_variables_include_sss(self, adapter):
        vars_ = adapter.source_description().variables_requested
        assert "sss" in vars_

    def test_variables_include_empirical_uncertainty(self, adapter):
        vars_ = adapter.source_description().variables_requested
        assert "sss_empirical_uncertainty" in vars_

    def test_variables_include_formal_uncertainty(self, adapter):
        vars_ = adapter.source_description().variables_requested
        assert "sss_formal_uncertainty" in vars_

    def test_exactly_three_variables(self, adapter):
        assert len(adapter.source_description().variables_requested) == 3


# ===========================================================================
# URL construction
# ===========================================================================

class TestUrlConstruction:
    def test_url_for_known_date_2023_01_01(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        expected = (
            "https://oisss.esr.org/data/V3/Daily/2023/"
            "OISSS_L4_multimission_global_daily_v3.0_2023-01-01.nc"
        )
        assert req.url_or_query == expected

    def test_url_starts_with_esr_base(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 15), data_root)
        assert req.url_or_query.startswith(_ESR_BASE)

    def test_url_contains_year_directory(self, adapter, data_root):
        req = adapter.build_request(date(2022, 3, 7), data_root)
        assert "/2022/" in req.url_or_query

    def test_url_contains_iso_date(self, adapter, data_root):
        req = adapter.build_request(date(2020, 11, 5), data_root)
        assert "2020-11-05" in req.url_or_query

    def test_url_ends_with_nc(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.url_or_query.endswith(".nc")

    def test_url_contains_official_filename_prefix(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert "OISSS_L4_multimission_global_daily_v3.0" in req.url_or_query

    def test_url_for_leap_day(self, adapter, data_root):
        req = adapter.build_request(date(2024, 2, 29), data_root)
        assert "2024-02-29" in req.url_or_query
        assert "/2024/" in req.url_or_query

    def test_url_for_year_boundary_dec31(self, adapter, data_root):
        req = adapter.build_request(date(2023, 12, 31), data_root)
        assert "2023-12-31" in req.url_or_query

    def test_url_for_year_boundary_jan1(self, adapter, data_root):
        req = adapter.build_request(date(2024, 1, 1), data_root)
        assert "2024-01-01" in req.url_or_query
        assert "/2024/" in req.url_or_query


# ===========================================================================
# Date range — daily chunking
# ===========================================================================

class TestDateRange:
    def test_single_day_produces_one_request(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 1), data_root)
        assert len(reqs) == 1

    def test_three_days_produces_three_requests(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 3), data_root)
        assert len(reqs) == 3

    def test_seven_days_produces_seven_requests(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        assert len(reqs) == 7

    def test_requests_are_in_chronological_order(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 5), data_root)
        dates = [r.request_date for r in reqs]
        assert dates == sorted(dates)

    def test_dates_are_sequential_no_gaps(self, adapter, data_root):
        from datetime import timedelta
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        for i in range(1, len(reqs)):
            assert reqs[i].request_date == reqs[i-1].request_date + timedelta(days=1)

    def test_reversed_dates_raise_value_error(self, adapter, data_root):
        with pytest.raises(ValueError):
            adapter.date_range_requests(date(2023, 6, 1), date(2023, 1, 1), data_root)

    def test_all_requests_have_correct_dataset_id(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 5), data_root)
        for req in reqs:
            assert req.dataset_id == "oisss"

    def test_cross_year_boundary(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2022, 12, 30), date(2023, 1, 2), data_root)
        assert len(reqs) == 4
        assert reqs[0].request_date.year == 2022
        assert reqs[-1].request_date.year == 2023


# ===========================================================================
# Output paths
# ===========================================================================

class TestOutputPaths:
    def test_path_is_relative(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert not req.expected_output_path.startswith("/")
        assert not req.expected_output_path.startswith("D:")

    def test_path_never_absolute_with_absolute_data_root(self, adapter):
        req = adapter.build_request(date(2023, 1, 1), "D:\\some\\absolute\\path")
        # AcquisitionRequest rejects absolute paths — but adapter must not produce one
        assert not req.expected_output_path.startswith("D:")
        assert not req.expected_output_path.startswith("/")

    def test_path_structure_raw_oisss_year(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        parts = Path(req.expected_output_path).parts
        assert parts[0] == "raw"
        assert parts[1] == "oisss"
        assert parts[2] == "2023"

    def test_path_filename_matches_official_pattern(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        fname = Path(req.expected_output_path).name
        assert fname == "OISSS_L4_multimission_global_daily_v3.0_2023-01-01.nc"

    def test_path_ends_with_nc(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 15), data_root)
        assert req.expected_output_path.endswith(".nc")

    def test_same_date_same_path_regardless_of_data_root(self, adapter):
        req_a = adapter.build_request(date(2023, 1, 1), "/path/a")
        req_b = adapter.build_request(date(2023, 1, 1), "/path/b")
        assert req_a.expected_output_path == req_b.expected_output_path

    def test_different_dates_different_paths(self, adapter, data_root):
        req1 = adapter.build_request(date(2023, 1, 1), data_root)
        req2 = adapter.build_request(date(2023, 1, 2), data_root)
        assert req1.expected_output_path != req2.expected_output_path

    def test_different_years_different_directories(self, adapter, data_root):
        req_2022 = adapter.build_request(date(2022, 6, 1), data_root)
        req_2023 = adapter.build_request(date(2023, 6, 1), data_root)
        assert "2022" in req_2022.expected_output_path
        assert "2023" in req_2023.expected_output_path
        assert req_2022.expected_output_path != req_2023.expected_output_path


# ===========================================================================
# Source identity and metadata
# ===========================================================================

class TestSourceMetadata:
    def test_request_has_correct_dataset_id(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.dataset_id == "oisss"

    def test_request_has_correct_date(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 15), data_root)
        assert req.request_date == date(2023, 6, 15)

    def test_spatial_region_is_global(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        sr = req.spatial_region
        assert sr["lat_min"] == pytest.approx(-89.875)
        assert sr["lat_max"] == pytest.approx(89.875)
        assert sr["lon_min"] == pytest.approx(-179.875)
        assert sr["lon_max"] == pytest.approx(179.875)

    def test_spatial_region_is_not_nio_subset(self, adapter, data_root):
        """The adapter must NOT limit to NIO bbox — must stay global."""
        req = adapter.build_request(date(2023, 1, 1), data_root)
        sr = req.spatial_region
        # NIO subset would be lat 5-30N, lon 45-105E — must NOT be this
        assert sr["lat_min"] < 0      # global extent extends into southern hemisphere
        assert sr["lon_min"] < 0      # global extent extends into western longitudes

    def test_extra_records_esr_source(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["source"] == "ESR_HTTPS_public"

    def test_extra_records_fill_value(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["fill_value"] == -9999.0

    def test_extra_records_global_dimensions(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["global_lat_cells"] == 720
        assert req.extra["global_lon_cells"] == 1440

    def test_validate_request_passes(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        issues = adapter.validate_request(req)
        assert issues == [], f"Unexpected validation issues: {issues}"


# ===========================================================================
# No scientific processing
# ===========================================================================

class TestNoScientificProcessing:
    """
    These tests document and enforce that the adapter does NOT perform
    any scientific transformation.
    """

    def test_url_is_plain_https_not_cmems_sentinel(self, adapter, data_root):
        """OISSS uses plain HTTPS — not a cmems:// or oscar:// sentinel."""
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.url_or_query.startswith("https://")
        assert not req.url_or_query.startswith("cmems://")
        assert not req.url_or_query.startswith("oscar://")

    def test_no_depth_parameter_in_url(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert "depth" not in req.url_or_query.lower()

    def test_no_bbox_in_url(self, adapter, data_root):
        """Global file is requested — no coordinate subsetting in the URL."""
        req = adapter.build_request(date(2023, 1, 1), data_root)
        # No server-side subset parameters
        assert "lat_min" not in req.url_or_query
        assert "lat_max" not in req.url_or_query
        assert "lon_min" not in req.url_or_query
        assert "lon_max" not in req.url_or_query

    def test_extra_subsetting_note_warns_against_subsetting(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("subsetting_note", "")
        assert "processing" in note.lower() or "global" in note.lower()

    def test_unit_attribute_documented_not_converted(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        # The unit discrepancy must be documented in extra
        assert req.extra.get("unit_attribute") == "1e-3"
        # The unit note must exist and mention the unresolved status
        note = req.extra.get("unit_note", "")
        assert "unresolved" in note.lower() or "1e-3" in note

    def test_no_scaling_factor_in_url(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert "scale" not in req.url_or_query.lower()
        assert "multiply" not in req.url_or_query.lower()

    def test_source_description_does_not_claim_psu_unit(self, adapter):
        """
        The source_description must not claim 'PSU' as units without
        acknowledging the 1e-3 discrepancy from the live file.
        """
        notes = adapter.source_description().notes
        # Either the notes mention the discrepancy, or they mention 1e-3
        assert "1e-3" in notes or "unresolved" in notes.lower()


# ===========================================================================
# Unit discrepancy explicitly tested
# ===========================================================================

class TestUnitDiscrepancy:
    """
    These tests verify that the unit discrepancy is explicitly documented
    and NOT silently resolved in any direction.
    """

    def test_unit_attribute_is_1e_3(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["unit_attribute"] == "1e-3"

    def test_unit_note_exists_and_is_non_empty(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("unit_note", "")
        assert len(note) > 0

    def test_unit_note_does_not_claim_psu_resolved(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("unit_note", "").lower()
        # The note must not claim the unit is definitively PSU
        assert "confirmed psu" not in note
        assert "unit is psu" not in note

    def test_unit_note_does_not_multiply_by_1000(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("unit_note", "").lower()
        assert "multiply by 1000" not in note
        assert "×1000" not in note

    def test_source_notes_mention_unit_discrepancy(self, adapter):
        notes = adapter.source_description().notes
        assert "1e-3" in notes


# ===========================================================================
# Registry consistency
# ===========================================================================

class TestRegistryConsistency:
    """Verify the OISSS registry entry is valid and consistent."""

    @pytest.fixture(scope="class")
    @classmethod
    def oisss_definition(cls):
        from oceanembed.datasets.registry import load_registry
        from pathlib import Path
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / "configs" / "datasets"
            if candidate.exists():
                registry = load_registry(candidate)
                return registry.get("oisss")
        pytest.skip("Registry directory not found")

    def test_registry_loads_oisss(self, oisss_definition):
        assert oisss_definition is not None

    def test_registry_dataset_id_matches_adapter(self, oisss_definition):
        assert oisss_definition.dataset_id == "oisss"

    def test_registry_version_is_30(self, oisss_definition):
        assert oisss_definition.version == "3.0"

    def test_registry_has_no_hardcoded_absolute_paths_in_storage(self, oisss_definition):
        if oisss_definition.storage:
            raw = oisss_definition.storage.raw_path_template or ""
            assert not raw.startswith("D:")
            assert not raw.startswith("E:")
            assert not raw.startswith("/home")

    def test_registry_authentication_not_required(self, oisss_definition):
        if oisss_definition.acquisition:
            assert oisss_definition.acquisition.authentication_required is False

    def test_registry_has_open_scientific_decisions(self, oisss_definition):
        osds = [o.id for o in oisss_definition.open_scientific_decisions]
        assert "OSD-005" in osds
        assert "OSD-006" in osds


# ===========================================================================
# API registration
# ===========================================================================

class TestApiRegistration:
    def test_oisss_adapter_implemented_in_api(self):
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        app_module._registry = None
        client = TestClient(app_module.app)
        resp = client.get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "oisss")
        assert ds["adapter_implemented"] is True
        assert ds["engineering_status"] == "ACQUISITION_READY"

    def test_oisss_does_not_return_501(self):
        """OISSS must accept a job request (not return 501 Not Implemented)."""
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        from unittest.mock import MagicMock, patch
        app_module._registry = None

        mock_adapter = MagicMock()
        mock_adapter._max_job_days = 31
        mock_adapter.date_range_requests.return_value = []

        with patch.object(app_module, "get_adapter", return_value=mock_adapter):
            client = TestClient(app_module.app)
            resp = client.post(
                "/api/acquisition/jobs",
                json={
                    "dataset_id": "oisss",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-03",
                },
            )
        assert resp.status_code == 202, f"Expected 202, got {resp.status_code}"
        assert resp.json()["dataset_id"] == "oisss"


# ===========================================================================
# Engine dispatch — OISSS uses plain HTTPS (download_file), not CMEMS/OSCAR
# ===========================================================================

class TestEngineDispatch:
    def test_oisss_dispatches_to_download_file(self, tmp_path):
        """OISSS plain HTTPS URL must route to download_file(), not CMEMS or OSCAR."""
        import asyncio
        from unittest.mock import patch
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob

        job = AcquisitionJob(
            job_id="disp_oisss",
            dataset_id="oisss",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 1),
            data_root=tmp_path,
        )

        async def fake_dl(*a, **kw):
            return (True, 500, "abc123")

        with patch("oceanembed.acquisition.engine.download_file",
                   side_effect=fake_dl) as mock_dl:
            with patch("oceanembed.acquisition.engine.run_cmems_subset") as mock_cmems:
                with patch("oceanembed.acquisition.engine.run_oscar_subset") as mock_oscar:
                    asyncio.run(run_acquisition_job(job, OISSSAdapter()))

        mock_dl.assert_called_once()
        mock_cmems.assert_not_called()
        mock_oscar.assert_not_called()


# ===========================================================================
# OISST regression — must be unaffected
# ===========================================================================

class TestOISSTRegression:
    def test_oisst_adapter_still_works(self):
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        adapter = OISSTAdapter()
        assert adapter.dataset_id == "oisst"
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert "ncei.noaa.gov" in req.url_or_query
        assert req.expected_output_path == "raw/oisst/2023/oisst_20230115.nc"

    def test_oisst_auth_still_not_required(self):
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        assert OISSTAdapter().source_description().authentication_required is False

    def test_oisst_still_in_implemented_adapters(self):
        import oceanembed.api.app as app_module
        assert "oisst" in app_module._IMPLEMENTED_ADAPTERS
