"""
Unit tests for the CCMP dataset adapter.

Critical invariants verified
-----------------------------
- 4 time records per file: the raw acquisition MUST NOT reduce time=4 to time=1
- No temporal aggregation or selection
- No spatial subsetting (global file)
- 0-360 longitude convention preserved
- Wind velocity (not stress)
- All 4 source variables: uwnd, vwnd, ws, nobs
- Public HTTPS — no authentication
- OSD-007 remains OPEN (temporal aggregation is processing-layer decision)

All tests are offline. No network access.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from oceanembed.datasets.adapters.ccmp import (
    CCMPAdapter,
    _RSS_BASE,
    _FILENAME_TEMPLATE,
    _SOURCE_COVERAGE_START,
    _SOURCE_COVERAGE_END,
    _TIME_RECORDS_PER_FILE,
    _ANALYSIS_TIMES_UTC,
)
from oceanembed.datasets.adapters.base import AcquisitionRequest, SourceDescription


@pytest.fixture
def adapter():
    return CCMPAdapter()


@pytest.fixture
def data_root(tmp_path):
    return str(tmp_path)


# ===========================================================================
# Identity
# ===========================================================================

class TestAdapterIdentity:
    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "ccmp"

    def test_repr_contains_class_and_id(self, adapter):
        assert "CCMPAdapter" in repr(adapter)
        assert "ccmp" in repr(adapter)

    def test_source_description_type(self, adapter):
        assert isinstance(adapter.source_description(), SourceDescription)

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "ccmp"

    def test_product_id_is_short_name(self, adapter):
        assert "CCMP_WINDS_10M6HR_L4_V3.1" in adapter.source_description().product_id

    def test_product_version_31(self, adapter):
        assert adapter.source_description().product_version == "3.1"

    def test_provider_contains_rss(self, adapter):
        assert "RSS" in adapter.source_description().provider_name or \
               "Remote Sensing" in adapter.source_description().provider_name

    def test_auth_not_required(self, adapter):
        assert adapter.source_description().authentication_required is False

    def test_temporal_resolution_6hourly(self, adapter):
        assert "6" in adapter.source_description().native_temporal_resolution or \
               "hourly" in adapter.source_description().native_temporal_resolution

    def test_spatial_resolution_025(self, adapter):
        assert adapter.source_description().native_spatial_resolution_degrees == 0.25

    def test_variables_include_uwnd_vwnd(self, adapter):
        vars_ = adapter.source_description().variables_requested
        assert "uwnd" in vars_
        assert "vwnd" in vars_

    def test_variables_include_ws_and_nobs(self, adapter):
        """All 4 source variables must be represented."""
        vars_ = adapter.source_description().variables_requested
        assert "ws" in vars_
        assert "nobs" in vars_

    def test_exactly_four_variables(self, adapter):
        assert len(adapter.source_description().variables_requested) == 4

    def test_source_notes_mention_4_time_records(self, adapter):
        notes = adapter.source_description().notes
        assert "4" in notes

    def test_source_notes_mention_osd_007(self, adapter):
        notes = adapter.source_description().notes
        assert "OSD-007" in notes

    def test_time_records_constant_is_4(self):
        assert _TIME_RECORDS_PER_FILE == 4

    def test_analysis_times_are_4(self):
        assert len(_ANALYSIS_TIMES_UTC) == 4

    def test_analysis_times_are_correct_utc(self):
        assert "00:00" in _ANALYSIS_TIMES_UTC
        assert "06:00" in _ANALYSIS_TIMES_UTC
        assert "12:00" in _ANALYSIS_TIMES_UTC
        assert "18:00" in _ANALYSIS_TIMES_UTC


# ===========================================================================
# URL construction
# ===========================================================================

class TestUrlConstruction:
    def test_url_for_known_date_2023_01_01(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        expected = (
            "https://data.remss.com/ccmp/v03.1/Y2023/M01/"
            "CCMP_Wind_Analysis_20230101_V03.1_L4.nc"
        )
        assert req.url_or_query == expected

    def test_url_starts_with_rss_base(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 15), data_root)
        assert req.url_or_query.startswith(_RSS_BASE)

    def test_url_contains_year_directory(self, adapter, data_root):
        req = adapter.build_request(date(2022, 3, 7), data_root)
        assert "/Y2022/" in req.url_or_query

    def test_url_contains_month_directory(self, adapter, data_root):
        req = adapter.build_request(date(2022, 3, 7), data_root)
        assert "/M03/" in req.url_or_query

    def test_url_contains_compact_date(self, adapter, data_root):
        req = adapter.build_request(date(2020, 11, 5), data_root)
        assert "20201105" in req.url_or_query

    def test_url_ends_with_nc(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.url_or_query.endswith(".nc")

    def test_url_contains_v031_l4(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert "V03.1_L4" in req.url_or_query

    def test_url_is_plain_https(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.url_or_query.startswith("https://")
        assert not req.url_or_query.startswith("cmems://")
        assert not req.url_or_query.startswith("oscar://")

    def test_url_for_year_boundary_dec31(self, adapter, data_root):
        req = adapter.build_request(date(2023, 12, 31), data_root)
        assert "20231231" in req.url_or_query
        assert "/Y2023/" in req.url_or_query

    def test_url_for_year_boundary_jan01(self, adapter, data_root):
        req = adapter.build_request(date(2024, 1, 1), data_root)
        assert "20240101" in req.url_or_query
        assert "/Y2024/" in req.url_or_query

    def test_url_for_leap_day(self, adapter, data_root):
        req = adapter.build_request(date(2024, 2, 29), data_root)
        assert "20240229" in req.url_or_query


# ===========================================================================
# Date range — daily chunking (one request per day, not four)
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

    def test_not_four_requests_per_day(self, adapter, data_root):
        """One daily file contains 4 records — adapter must NOT create 4 requests."""
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 1), data_root)
        assert len(reqs) == 1, (
            "CCMP adapter must produce 1 request per day (not 4). "
            "The 4 time records are in the single daily file."
        )

    def test_requests_chronological(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 5), data_root)
        dates = [r.request_date for r in reqs]
        assert dates == sorted(dates)

    def test_dates_sequential_no_gaps(self, adapter, data_root):
        from datetime import timedelta
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        for i in range(1, len(reqs)):
            assert reqs[i].request_date == reqs[i - 1].request_date + timedelta(days=1)

    def test_reversed_dates_raise_value_error(self, adapter, data_root):
        with pytest.raises(ValueError):
            adapter.date_range_requests(date(2023, 6, 1), date(2023, 1, 1), data_root)

    def test_all_requests_have_correct_dataset_id(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 5), data_root)
        for req in reqs:
            assert req.dataset_id == "ccmp"

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

    def test_path_structure_raw_ccmp_year(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        parts = Path(req.expected_output_path).parts
        assert parts[0] == "raw"
        assert parts[1] == "ccmp"
        assert parts[2] == "2023"

    def test_path_filename_matches_official_pattern(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        fname = Path(req.expected_output_path).name
        assert fname == "CCMP_Wind_Analysis_20230101_V03.1_L4.nc"

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


# ===========================================================================
# Temporal preservation — the most critical invariant
# ===========================================================================

class TestTemporalPreservation:
    """
    Verify that the acquisition adapter does NOT perform temporal aggregation
    or selection. The 4 × 6-hourly records must be preserved.
    """

    def test_extra_records_4_time_records_per_file(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["time_records_per_file"] == 4

    def test_extra_global_time_cells_is_4(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["global_time_cells"] == 4

    def test_extra_analysis_times_all_four(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        times = req.extra["analysis_times_utc"]
        assert len(times) == 4
        assert "00:00" in times
        assert "06:00" in times
        assert "12:00" in times
        assert "18:00" in times

    def test_extra_temporal_note_mentions_osd007(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("temporal_note", "")
        assert "OSD-007" in note

    def test_extra_temporal_note_mentions_all_four_times(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("temporal_note", "")
        assert "00Z" in note and "06Z" in note and "12Z" in note and "18Z" in note

    def test_one_request_per_day_not_four(self, adapter, data_root):
        """Regression: must not produce 4 requests/day (one per analysis time)."""
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 3), data_root)
        assert len(reqs) == 3

    def test_source_description_notes_mention_4_records(self, adapter):
        notes = adapter.source_description().notes
        assert "4" in notes
        assert "00:00" in notes or "00Z" in notes or "6-hourly" in notes.lower()

    def test_url_points_to_daily_file_not_6hourly_file(self, adapter, data_root):
        """URL must target the daily container file, not a sub-daily file."""
        req = adapter.build_request(date(2023, 1, 1), data_root)
        # Daily file: CCMP_Wind_Analysis_20230101_V03.1_L4.nc
        # There is no 'z00' or '00Z' component in the URL — that would indicate
        # a legacy 6-hourly individual file selection.
        url = req.url_or_query
        assert "z00" not in url.lower()
        assert "z06" not in url.lower()
        assert "z12" not in url.lower()
        assert "z18" not in url.lower()
        assert "00Z" not in url
        assert "_00_" not in url


# ===========================================================================
# Spatial safety — global, no NIO subset, 0-360 longitude preserved
# ===========================================================================

class TestSpatialSafety:
    def test_spatial_region_is_global_lat(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        sr = req.spatial_region
        assert sr["lat_min"] == pytest.approx(-89.875)
        assert sr["lat_max"] == pytest.approx(89.875)

    def test_spatial_region_lon_is_0_360(self, adapter, data_root):
        """Longitude must use 0-360 convention (not -180 to +180)."""
        req = adapter.build_request(date(2023, 1, 1), data_root)
        sr = req.spatial_region
        assert sr["lon_min"] == pytest.approx(0.125)
        assert sr["lon_max"] == pytest.approx(359.875)

    def test_spatial_region_is_not_nio_subset(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        sr = req.spatial_region
        # NIO would be lat 5-30N, lon 45-105E — must NOT be this
        assert sr["lat_min"] < 0        # global extent south of equator
        assert sr["lon_max"] > 180      # global extent west of 180

    def test_extra_lon_convention_is_0_360(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["lon_convention"] == "0-360"

    def test_extra_global_lat_cells_is_720(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["global_lat_cells"] == 720

    def test_extra_global_lon_cells_is_1440(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["global_lon_cells"] == 1440

    def test_extra_subsetting_note_warns_against_nio_subset(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("subsetting_note", "")
        assert "global" in note.lower() or "processing" in note.lower()

    def test_no_bbox_in_url(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        for param in ["lat_min", "lat_max", "lon_min", "lon_max", "bbox"]:
            assert param not in req.url_or_query


# ===========================================================================
# Wind semantics — velocity not stress
# ===========================================================================

class TestWindSemantics:
    def test_extra_wind_type_is_velocity(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["wind_type"] == "velocity"

    def test_extra_reference_height_is_10m(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["reference_height_m"] == 10

    def test_extra_wind_note_mentions_velocity_not_stress(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        note = req.extra.get("wind_note", "")
        assert "velocity" in note.lower() or "VELOCITY" in note
        assert "NOT" in note or "not" in note.lower()

    def test_extra_wind_note_mentions_osd013(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert "OSD-013" in req.extra.get("wind_note", "")

    def test_source_notes_confirm_velocity(self, adapter):
        notes = adapter.source_description().notes.lower()
        assert "velocity" in notes or "uwnd" in notes

    def test_four_variables_not_stress_derived(self, adapter):
        """ws is wind speed, not stress. nobs is count."""
        vars_ = adapter.source_description().variables_requested
        # Wind stress variables would be named tau_x, tau_y, taux, tauy, etc.
        for v in vars_:
            assert "tau" not in v.lower()
            assert "stress" not in v.lower()


# ===========================================================================
# Fill value and missing data
# ===========================================================================

class TestFillValue:
    def test_extra_fill_value_is_minus_9999(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        assert req.extra["fill_value"] == -9999.0

    def test_fill_value_constant_in_extra(self, adapter, data_root):
        # All three date ranges get the same fill value
        for d in [date(2012, 1, 1), date(2018, 6, 15), date(2024, 12, 31)]:
            req = adapter.build_request(d, "/data")
            assert req.extra["fill_value"] == -9999.0


# ===========================================================================
# Validate request
# ===========================================================================

class TestValidateRequest:
    def test_valid_request_passes(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        issues = adapter.validate_request(req)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_wrong_dataset_id_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        bad = AcquisitionRequest(
            dataset_id="oisst",  # wrong
            request_date=req.request_date,
            url_or_query=req.url_or_query,
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
            extra=req.extra,
        )
        issues = adapter.validate_request(bad)
        assert any("oisst" in m for m in issues)

    def test_empty_url_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 1, 1), data_root)
        bad = AcquisitionRequest(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            url_or_query="",
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
            extra=req.extra,
        )
        issues = adapter.validate_request(bad)
        assert len(issues) > 0


# ===========================================================================
# Engine dispatch — CCMP uses plain HTTPS (download_file)
# ===========================================================================

class TestEngineDispatch:
    def test_ccmp_dispatches_to_download_file(self, tmp_path):
        """CCMP plain HTTPS URL must route to download_file(), not CMEMS or OSCAR."""
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob

        job = AcquisitionJob(
            job_id="disp_ccmp",
            dataset_id="ccmp",
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
                    asyncio.run(run_acquisition_job(job, CCMPAdapter()))

        mock_dl.assert_called_once()
        mock_cmems.assert_not_called()
        mock_oscar.assert_not_called()


# ===========================================================================
# API registration
# ===========================================================================

class TestApiRegistration:
    def test_ccmp_adapter_implemented_in_api(self):
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        app_module._registry = None
        client = TestClient(app_module.app)
        resp = client.get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "ccmp")
        assert ds["adapter_implemented"] is True
        assert ds["engineering_status"] == "ACQUISITION_READY"

    def test_ccmp_does_not_return_501(self):
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        from unittest.mock import MagicMock
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        app_module._registry = None

        mock_adapter = MagicMock()
        mock_adapter._max_job_days = 31
        mock_adapter.date_range_requests.return_value = []

        with patch.object(app_module, "get_adapter", return_value=mock_adapter):
            client = TestClient(app_module.app)
            resp = client.post(
                "/api/acquisition/jobs",
                json={
                    "dataset_id": "ccmp",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-03",
                },
            )
        assert resp.status_code == 202
        assert resp.json()["dataset_id"] == "ccmp"


# ===========================================================================
# Registry consistency
# ===========================================================================

class TestRegistryConsistency:
    @pytest.fixture(scope="class")
    @classmethod
    def ccmp_definition(cls):
        from oceanembed.datasets.registry import load_registry
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / "configs" / "datasets"
            if candidate.exists():
                registry = load_registry(candidate)
                return registry.get("ccmp")
        pytest.skip("Registry directory not found")

    def test_registry_loads_ccmp(self, ccmp_definition):
        assert ccmp_definition is not None

    def test_registry_dataset_id_matches_adapter(self, ccmp_definition):
        assert ccmp_definition.dataset_id == "ccmp"

    def test_registry_version_is_31(self, ccmp_definition):
        assert ccmp_definition.version == "3.1"

    def test_registry_has_osd_007(self, ccmp_definition):
        osds = [o.id for o in ccmp_definition.open_scientific_decisions]
        assert "OSD-007" in osds

    def test_registry_has_osd_013(self, ccmp_definition):
        osds = [o.id for o in ccmp_definition.open_scientific_decisions]
        assert "OSD-013" in osds

    def test_registry_has_no_hardcoded_absolute_paths(self, ccmp_definition):
        if ccmp_definition.storage:
            raw = ccmp_definition.storage.raw_path_template or ""
            assert not raw.startswith("D:")
            assert not raw.startswith("/home")


# ===========================================================================
# Regression — previously verified datasets unaffected
# ===========================================================================

class TestOISSTRegression:
    def test_oisst_still_works(self):
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        adapter = OISSTAdapter()
        assert adapter.dataset_id == "oisst"
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert "ncei.noaa.gov" in req.url_or_query

    def test_oisst_still_in_implemented_adapters(self):
        import oceanembed.api.app as app_module
        assert "oisst" in app_module._IMPLEMENTED_ADAPTERS


class TestOISSSRegression:
    def test_oisss_still_works(self):
        from oceanembed.datasets.adapters.oisss import OISSSAdapter
        adapter = OISSSAdapter()
        assert adapter.dataset_id == "oisss"
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert "oisss.esr.org" in req.url_or_query

    def test_oisss_auth_not_required(self):
        from oceanembed.datasets.adapters.oisss import OISSSAdapter
        assert not OISSSAdapter().source_description().authentication_required
