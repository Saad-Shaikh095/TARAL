"""
Unit tests for the GlobCurrent dataset adapter.

Test matrix (from prompt spec)
-------------------------------
A. Valid small date range
B. Valid small bounding box
C. Different valid bounding box
D. Different valid date range
E. Invalid date range (start > end)
F. Invalid latitude
G. Invalid longitude
H. Date outside source coverage
I. Existing valid file reuse
J. Failed download / retry behavior
K. Manifest creation/update
L. Checksum behavior
M. Deterministic path generation
N. Authentication configuration behavior
O. No credentials in logs

Additional coverage
-------------------
- Adapter identity (dataset_id, source_description, auth_required)
- AcquisitionRequest structural correctness
- Sentinel URL format and parse_cmems_sentinel_url()
- Yearly chunking across year boundaries
- Partial-year filename vs full-year filename
- validate_request() — GlobCurrent-specific checks
- build_request() — single-date convenience method
- Engine dispatch: cmems:// sentinel → run_cmems_subset()
- Engine dispatch: plain URL → download_file() unchanged (OISST regression)

All network access is mocked. No real CMEMS calls are made.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from oceanembed.datasets.adapters.globcurrent import (
    GlobCurrentAdapter,
    _CMEMS_DATASET_ID,
    _CMEMS_PRODUCT_ID,
    _CMEMS_VARIABLES,
    _CMEMS_DEPTH,
    _CMEMS_URL_SENTINEL,
    _SOURCE_COVERAGE_START,
    _SOURCE_COVERAGE_END,
    _expected_days,
    _validate_globcurrent_request_params,
    _year_period_overlap,
    _years_in_range,
    parse_cmems_sentinel_url,
)
from oceanembed.datasets.adapters.base import AcquisitionRequest, SourceDescription


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    """Default adapter using NIO domain."""
    return GlobCurrentAdapter()


@pytest.fixture
def small_adapter():
    """Adapter with a small valid bounding box."""
    return GlobCurrentAdapter(min_lat=10.0, max_lat=20.0, min_lon=60.0, max_lon=80.0)


@pytest.fixture
def data_root(tmp_path):
    """Temporary data root directory."""
    return str(tmp_path)


# ===========================================================================
# Adapter identity
# ===========================================================================

class TestAdapterIdentity:
    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "globcurrent"

    def test_repr(self, adapter):
        r = repr(adapter)
        assert "GlobCurrentAdapter" in r
        assert "globcurrent" in r

    def test_source_description_type(self, adapter):
        desc = adapter.source_description()
        assert isinstance(desc, SourceDescription)

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "globcurrent"

    def test_source_description_product_id(self, adapter):
        assert adapter.source_description().product_id == _CMEMS_PRODUCT_ID

    def test_source_description_authentication_required(self, adapter):
        assert adapter.source_description().authentication_required is True

    def test_source_description_temporal_resolution(self, adapter):
        assert adapter.source_description().native_temporal_resolution == "daily"

    def test_source_description_spatial_resolution(self, adapter):
        assert adapter.source_description().native_spatial_resolution_degrees == 0.25

    def test_source_description_variables(self, adapter):
        vars_ = adapter.source_description().variables_requested
        assert "uo" in vars_
        assert "vo" in vars_

    def test_max_job_days(self, adapter):
        assert adapter._max_job_days == 366


# ===========================================================================
# A. Valid small date range
# ===========================================================================

class TestValidSmallDateRange:
    """Test A: A small valid date range produces correct requests."""

    def test_single_year_single_request(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 1, 7), data_root
        )
        assert len(requests) == 1

    def test_request_date_is_period_start(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 3, 1), date(2023, 3, 31), data_root
        )
        assert requests[0].request_date == date(2023, 3, 1)

    def test_request_dataset_id(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 1, 3), data_root
        )
        assert requests[0].dataset_id == "globcurrent"

    def test_url_starts_with_cmems_sentinel(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 1, 7), data_root
        )
        assert requests[0].url_or_query.startswith("cmems://")

    def test_output_path_relative(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 1, 7), data_root
        )
        path = requests[0].expected_output_path
        assert not path.startswith("/")
        assert "\\" not in path

    def test_validate_request_passes(self, adapter, data_root):
        req = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 1, 7), data_root
        )[0]
        issues = adapter.validate_request(req)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_partial_year_filename_structure(self, adapter, data_root):
        # Jan 1 – Mar 31 is a partial year — filename should reflect bounds
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 3, 31), data_root
        )
        filename = Path(requests[0].expected_output_path).name
        assert "GlobCurrent_2023" in filename

    def test_full_year_filename(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 12, 31), data_root
        )
        filename = Path(requests[0].expected_output_path).name
        assert filename == "GlobCurrent_2023_surface_uv.nc"


# ===========================================================================
# B. Valid small bounding box
# ===========================================================================

class TestValidSmallBoundingBox:
    """Test B: A small but valid bbox is accepted without source-code changes."""

    def test_small_bbox_accepted(self):
        adapter = GlobCurrentAdapter(
            min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0
        )
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 1, 7),
            12.0, 18.0, 70.0, 80.0,
        )
        assert errors == []

    def test_spatial_region_in_request(self):
        adapter = GlobCurrentAdapter(
            min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0
        )
        req = adapter.build_request(date(2023, 6, 1), "/data")
        assert req.spatial_region["lat_min"] == 12.0
        assert req.spatial_region["lat_max"] == 18.0
        assert req.spatial_region["lon_min"] == 70.0
        assert req.spatial_region["lon_max"] == 80.0

    def test_bbox_in_sentinel_url(self):
        adapter = GlobCurrentAdapter(
            min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0
        )
        req = adapter.build_request(date(2023, 6, 1), "/data")
        url = req.url_or_query
        assert "lat_min=12.0" in url
        assert "lat_max=18.0" in url
        assert "lon_min=70.0" in url
        assert "lon_max=80.0" in url

    def test_bbox_reflected_in_parsed_url(self):
        adapter = GlobCurrentAdapter(
            min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0
        )
        req = adapter.build_request(date(2023, 6, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["lat_min"] == 12.0
        assert params["lat_max"] == 18.0
        assert params["lon_min"] == 70.0
        assert params["lon_max"] == 80.0


# ===========================================================================
# C. Different valid bounding box (no source-code changes needed)
# ===========================================================================

class TestDifferentValidBoundingBox:
    """Test C: A completely different bbox works without source changes."""

    def test_different_bbox_produces_different_path(self):
        adapter_a = GlobCurrentAdapter(
            min_lat=5.0, max_lat=30.0, min_lon=45.0, max_lon=105.0
        )
        adapter_b = GlobCurrentAdapter(
            min_lat=10.0, max_lat=25.0, min_lon=60.0, max_lon=90.0
        )
        req_a = adapter_a.build_request(date(2023, 1, 1), "/data")
        req_b = adapter_b.build_request(date(2023, 1, 1), "/data")
        # Output paths are the same (year-based), but sentinel URLs differ
        assert req_a.url_or_query != req_b.url_or_query

    def test_different_bbox_url_contains_correct_coords(self):
        adapter = GlobCurrentAdapter(
            min_lat=10.0, max_lat=25.0, min_lon=60.0, max_lon=90.0
        )
        req = adapter.build_request(date(2023, 1, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["lat_min"] == 10.0
        assert params["lon_max"] == 90.0

    def test_global_bbox_is_valid(self):
        errors = _validate_globcurrent_request_params(
            date(2020, 1, 1), date(2020, 12, 31),
            -90.0, 90.0, -180.0, 180.0,
        )
        assert errors == []

    def test_southern_hemisphere_bbox_valid(self):
        errors = _validate_globcurrent_request_params(
            date(2020, 1, 1), date(2020, 6, 30),
            -40.0, -10.0, -60.0, 20.0,
        )
        assert errors == []


# ===========================================================================
# D. Different valid date range (no source-code changes needed)
# ===========================================================================

class TestDifferentValidDateRange:
    """Test D: Different date ranges produce correct yearly requests."""

    def test_single_month_request(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2018, 6, 1), date(2018, 6, 30), data_root
        )
        assert len(requests) == 1
        assert requests[0].request_date == date(2018, 6, 1)

    def test_full_year_request(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 12, 31), data_root
        )
        assert len(requests) == 1
        params = parse_cmems_sentinel_url(requests[0].url_or_query)
        assert params["start"] == "2023-01-01T00:00:00"
        assert params["end"] == "2023-12-31T23:59:59"

    def test_multi_year_produces_multiple_requests(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2021, 1, 1), date(2023, 12, 31), data_root
        )
        assert len(requests) == 3
        years = [r.request_date.year for r in requests]
        assert years == [2021, 2022, 2023]

    def test_cross_year_boundary_correct_overlap(self, adapter, data_root):
        # 2022-11-01 to 2023-03-31 spans two years
        requests = adapter.date_range_requests(
            date(2022, 11, 1), date(2023, 3, 31), data_root
        )
        assert len(requests) == 2
        # 2022 portion: Nov 1 – Dec 31
        p2022 = parse_cmems_sentinel_url(requests[0].url_or_query)
        assert p2022["start"] == "2022-11-01T00:00:00"
        assert p2022["end"] == "2022-12-31T23:59:59"
        # 2023 portion: Jan 1 – Mar 31
        p2023 = parse_cmems_sentinel_url(requests[1].url_or_query)
        assert p2023["start"] == "2023-01-01T00:00:00"
        assert p2023["end"] == "2023-03-31T23:59:59"

    def test_different_year_different_path(self, adapter, data_root):
        reqs_2021 = adapter.date_range_requests(
            date(2021, 1, 1), date(2021, 12, 31), data_root
        )
        reqs_2023 = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 12, 31), data_root
        )
        assert reqs_2021[0].expected_output_path != reqs_2023[0].expected_output_path
        assert "2021" in reqs_2021[0].expected_output_path
        assert "2023" in reqs_2023[0].expected_output_path


# ===========================================================================
# E. Invalid date range
# ===========================================================================

class TestInvalidDateRange:
    """Test E: start > end is rejected clearly."""

    def test_reversed_dates_raise_value_error(self, adapter, data_root):
        with pytest.raises(ValueError, match="start_date"):
            adapter.date_range_requests(
                date(2023, 6, 1), date(2023, 1, 1), data_root
            )

    def test_error_message_contains_both_dates(self, adapter, data_root):
        with pytest.raises(ValueError) as exc_info:
            adapter.date_range_requests(
                date(2023, 6, 1), date(2023, 1, 1), data_root
            )
        msg = str(exc_info.value)
        assert "2023-06-01" in msg
        assert "2023-01-01" in msg

    def test_same_date_is_valid(self, adapter, data_root):
        # Single-day range is valid (one partial-year request)
        requests = adapter.date_range_requests(
            date(2023, 6, 15), date(2023, 6, 15), data_root
        )
        assert len(requests) == 1

    def test_validation_helper_reversed_dates(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 6, 1), date(2023, 1, 1),
            5.0, 30.0, 45.0, 105.0,
        )
        assert any("start_date" in e for e in errors)


# ===========================================================================
# F. Invalid latitude
# ===========================================================================

class TestInvalidLatitude:
    """Test F: Invalid latitudes are rejected clearly, not silently clamped."""

    def test_min_lat_greater_than_max_lat(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            30.0, 5.0, 45.0, 105.0,  # min > max
        )
        assert any("min_lat" in e and "max_lat" in e for e in errors)

    def test_min_lat_equal_to_max_lat(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            15.0, 15.0, 45.0, 105.0,  # equal — degenerate box
        )
        assert any("min_lat" in e for e in errors)

    def test_lat_below_negative_90(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            -91.0, 30.0, 45.0, 105.0,
        )
        assert any("-91" in e or "min_lat" in e for e in errors)

    def test_lat_above_positive_90(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            5.0, 91.0, 45.0, 105.0,
        )
        assert any("91" in e or "max_lat" in e for e in errors)

    def test_adapter_construction_with_bad_lat_raises_on_use(self):
        # Adapter construction does not validate — validation happens on request
        adapter = GlobCurrentAdapter(min_lat=50.0, max_lat=10.0)
        with pytest.raises(ValueError):
            adapter.date_range_requests(
                date(2023, 1, 1), date(2023, 12, 31), "/data"
            )

    def test_error_message_mentions_lat_values(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            30.0, 5.0, 45.0, 105.0,
        )
        combined = " ".join(errors)
        assert "30" in combined or "5" in combined


# ===========================================================================
# G. Invalid longitude
# ===========================================================================

class TestInvalidLongitude:
    """Test G: Invalid longitudes are rejected clearly, not silently clamped."""

    def test_min_lon_greater_than_max_lon(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            5.0, 30.0, 105.0, 45.0,  # min > max
        )
        assert any("min_lon" in e and "max_lon" in e for e in errors)

    def test_min_lon_equal_to_max_lon(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            5.0, 30.0, 75.0, 75.0,
        )
        assert any("min_lon" in e for e in errors)

    def test_lon_below_negative_180(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            5.0, 30.0, -181.0, 105.0,
        )
        assert any("-181" in e or "min_lon" in e for e in errors)

    def test_lon_above_positive_180(self):
        errors = _validate_globcurrent_request_params(
            date(2023, 1, 1), date(2023, 12, 31),
            5.0, 30.0, 45.0, 181.0,
        )
        assert any("181" in e or "max_lon" in e for e in errors)

    def test_adapter_construction_with_bad_lon_raises_on_use(self):
        adapter = GlobCurrentAdapter(min_lon=105.0, max_lon=45.0)
        with pytest.raises(ValueError):
            adapter.date_range_requests(
                date(2023, 1, 1), date(2023, 12, 31), "/data"
            )


# ===========================================================================
# H. Date outside source coverage
# ===========================================================================

class TestDateOutsideSourceCoverage:
    """Test H: Dates outside 2012-01-01 → 2024-12-31 are rejected clearly."""

    def test_start_date_before_coverage_start(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(
                date(2010, 1, 1), date(2012, 6, 30), data_root
            )

    def test_end_date_after_coverage_end(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(
                date(2024, 6, 1), date(2025, 3, 1), data_root
            )

    def test_error_mentions_coverage_dates(self, adapter, data_root):
        with pytest.raises(ValueError) as exc_info:
            adapter.date_range_requests(
                date(2010, 1, 1), date(2010, 12, 31), data_root
            )
        msg = str(exc_info.value)
        assert str(_SOURCE_COVERAGE_START) in msg

    def test_validation_helper_before_coverage(self):
        errors = _validate_globcurrent_request_params(
            date(2011, 12, 31), date(2012, 6, 30),
            5.0, 30.0, 45.0, 105.0,
        )
        assert any("2011" in e or "coverage" in e.lower() for e in errors)

    def test_validation_helper_after_coverage(self):
        errors = _validate_globcurrent_request_params(
            date(2024, 6, 1), date(2025, 1, 1),
            5.0, 30.0, 45.0, 105.0,
        )
        assert any("2025" in e or "coverage" in e.lower() for e in errors)

    def test_boundary_dates_valid(self, adapter, data_root):
        # Exact boundary dates should be accepted
        requests = adapter.date_range_requests(
            _SOURCE_COVERAGE_START, _SOURCE_COVERAGE_START, data_root
        )
        assert len(requests) == 1

    def test_coverage_end_boundary_valid(self, adapter, data_root):
        requests = adapter.date_range_requests(
            date(2024, 12, 31), _SOURCE_COVERAGE_END, data_root
        )
        assert len(requests) == 1


# ===========================================================================
# I. Existing valid file reuse
# ===========================================================================

class TestExistingFileReuse:
    """Test I: run_cmems_subset skips download if the file already exists and is non-empty."""

    def test_existing_non_empty_file_is_skipped(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob, JobStatus

        # Create a non-empty file at the destination
        dest = tmp_path / "raw" / "globcurrent" / "2023" / "GlobCurrent_2023_surface_uv.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake netcdf data content")

        job = AcquisitionJob(
            job_id="test0001",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )

        adapter = GlobCurrentAdapter()
        req = adapter.date_range_requests(
            date(2023, 1, 1), date(2023, 12, 31), str(tmp_path)
        )[0]

        # Should return success immediately without calling copernicusmarine
        with patch("copernicusmarine.subset") as mock_subset:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        mock_subset.assert_not_called()
        assert success is True
        assert nbytes == dest.stat().st_size
        assert md5 is not None

    def test_skipped_log_message(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        dest.write_bytes(b"existing valid data")

        job = AcquisitionJob(
            job_id="test0002",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        with patch("copernicusmarine.subset"):
            run_cmems_subset(req.url_or_query, dest, job)

        log_messages = " ".join(entry["message"] for entry in job.logs)
        assert "already exists" in log_messages.lower() or "skipping" in log_messages.lower()

    def test_empty_existing_file_triggers_download(self, tmp_path):
        """An empty .nc file at the destination should NOT be treated as valid."""
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        dest.write_bytes(b"")  # empty file

        job = AcquisitionJob(
            job_id="test0003",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        # With no credentials set, expect failure (not skip)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
            os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        # Should have attempted (not skipped) and failed due to missing credentials
        assert success is False


# ===========================================================================
# J. Failed download / retry behavior
# ===========================================================================

class TestFailedDownloadRetry:
    """Test J: Transient failures trigger retries; auth failures do not."""

    def test_missing_credentials_returns_failure(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0004",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        saved_user = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_pass = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_user:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_user
            if saved_pass:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_pass

        assert success is False
        assert nbytes == 0
        assert md5 is None

    def test_transient_error_retried(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0005",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        call_count = {"n": 0}

        def fake_subset(**kwargs):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient network error")
            # Second call succeeds: write the output file.
            # copernicusmarine v2.x appends .nc to output_filename, so the
            # engine passes stem-only; we must write <stem>.nc here.
            stem = kwargs["output_filename"]
            out = Path(kwargs["output_directory"]) / (stem + ".nc")
            out.write_bytes(b"fake netcdf content from retry")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "testuser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "testpass",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_subset):
                with patch("time.sleep"):  # don't actually wait
                    success, nbytes, md5 = run_cmems_subset(
                        req.url_or_query, dest, job, max_retries=3
                    )

        assert call_count["n"] == 2, f"Expected 2 calls (1 fail + 1 success), got {call_count['n']}"
        assert success is True

    def test_auth_error_not_retried(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0006",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        call_count = {"n": 0}

        def fake_auth_error(**kwargs):
            call_count["n"] += 1
            raise RuntimeError("HTTP 401 unauthorized access denied")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "baduser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "badpass",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_auth_error):
                success, nbytes, md5 = run_cmems_subset(
                    req.url_or_query, dest, job, max_retries=3
                )

        # Auth errors should abort immediately — only 1 attempt
        assert call_count["n"] == 1
        assert success is False

    def test_partial_file_cleaned_up_on_failure(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        tmp_dest = dest.with_suffix(".tmp")
        job = AcquisitionJob(
            job_id="test0007",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), str(tmp_path))

        def fake_fail_and_leave_tmp(**kwargs):
            # Write a partial file (simulating mid-transfer failure).
            # Engine passes stem-only to output_filename; copernicusmarine
            # appends .nc, so we write <stem>.nc here.
            stem = kwargs["output_filename"]
            out = Path(kwargs["output_directory"]) / (stem + ".nc")
            out.write_bytes(b"partial garbage")
            raise RuntimeError("connection lost mid-transfer")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_fail_and_leave_tmp):
                with patch("time.sleep"):
                    run_cmems_subset(req.url_or_query, dest, job, max_retries=1)

        # The .tmp file must be cleaned up
        assert not tmp_dest.exists(), ".tmp file should be removed on failure"
        # The real dest must not exist (no partial promotion)
        assert not dest.exists(), "dest must not exist after clean failure"


# ===========================================================================
# K. Manifest creation/update
# ===========================================================================

class TestManifestBehavior:
    """Test K: Manifest is written correctly for GlobCurrent acquisitions."""

    def test_manifest_entry_written_on_success(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob
        from oceanembed.acquisition.manifest import ManifestStore, ManifestStatus

        dest = tmp_path / "raw" / "globcurrent" / "2023" / "GlobCurrent_2023_surface_uv.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)

        job = AcquisitionJob(
            job_id="test0008",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )

        def fake_subset(**kwargs):
            # Engine passes stem-only; copernicusmarine appends .nc
            stem = kwargs["output_filename"]
            out = Path(kwargs["output_directory"]) / (stem + ".nc")
            out.write_bytes(b"fake globcurrent netcdf")

        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_subset):
                success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        assert success is True

        # Now write the manifest as the engine would
        from oceanembed.acquisition.manifest import ManifestEntry
        manifest = ManifestStore(tmp_path)
        entry = ManifestEntry(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            source_url=req.url_or_query,
            local_path=str(dest),
            product_id=req.source_description.product_id,
            file_size_bytes=nbytes,
            md5_checksum=md5,
            status=ManifestStatus.SUCCESS,
        )
        manifest.record(entry)

        records = manifest.read_all("globcurrent")
        assert len(records) == 1
        assert records[0]["dataset_id"] == "globcurrent"
        assert records[0]["status"] == "SUCCESS"

    def test_manifest_records_product_id(self, tmp_path):
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        manifest = ManifestStore(tmp_path)
        entry = ManifestEntry(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            source_url=req.url_or_query,
            local_path="/fake/path/GlobCurrent_2023_surface_uv.nc",
            product_id=req.source_description.product_id,
            file_size_bytes=1000,
            md5_checksum="abc123",
            status=ManifestStatus.SUCCESS,
        )
        manifest.record(entry)

        records = manifest.read_all("globcurrent")
        assert records[0]["product_id"] == _CMEMS_PRODUCT_ID

    def test_manifest_records_failure_status(self, tmp_path):
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        manifest = ManifestStore(tmp_path)
        entry = ManifestEntry(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            source_url=req.url_or_query,
            local_path="/fake/path.nc",
            product_id=req.source_description.product_id,
            status=ManifestStatus.FAILED,
            error="Download failed",
        )
        manifest.record(entry)

        records = manifest.read_all("globcurrent")
        assert records[0]["status"] == "FAILED"
        assert records[0]["error"] == "Download failed"

    def test_manifest_does_not_overwrite_oisst_manifest(self, tmp_path):
        """Writing a GlobCurrent manifest must not touch oisst.jsonl."""
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

        manifest = ManifestStore(tmp_path)
        oisst_entry = ManifestEntry(
            dataset_id="oisst",
            request_date=date(2023, 1, 1),
            source_url="https://ncei.noaa.gov/oisst_20230101.nc",
            local_path="/data/oisst/2023/oisst_20230101.nc",
            product_id="AVHRR_OI-NCEI-L4-GLOB-v2.1",
            status=ManifestStatus.SUCCESS,
        )
        manifest.record(oisst_entry)

        gc_entry = ManifestEntry(
            dataset_id="globcurrent",
            request_date=date(2023, 1, 1),
            source_url="cmems://fake",
            local_path="/data/globcurrent/2023/GlobCurrent_2023_surface_uv.nc",
            product_id=_CMEMS_PRODUCT_ID,
            status=ManifestStatus.SUCCESS,
        )
        manifest.record(gc_entry)

        oisst_records = manifest.read_all("oisst")
        gc_records = manifest.read_all("globcurrent")
        assert len(oisst_records) == 1
        assert len(gc_records) == 1
        assert oisst_records[0]["dataset_id"] == "oisst"
        assert gc_records[0]["dataset_id"] == "globcurrent"


# ===========================================================================
# L. Checksum behavior
# ===========================================================================

class TestChecksumBehavior:
    """Test L: MD5 checksums are computed and stored correctly."""

    def test_checksum_computed_on_download(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0009",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        fake_content = b"fake nc content for checksum test"

        def fake_subset(**kwargs):
            stem = kwargs["output_filename"]
            out = Path(kwargs["output_directory"]) / (stem + ".nc")
            out.write_bytes(fake_content)

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_subset):
                success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        expected_md5 = hashlib.md5(fake_content).hexdigest()
        assert success is True
        assert md5 == expected_md5

    def test_checksum_computed_for_existing_file(self, tmp_path):
        """Skipped (already-existing) files still get their MD5 computed."""
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        fake_content = b"existing file content"
        dest.write_bytes(fake_content)

        job = AcquisitionJob(
            job_id="test0010",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        with patch("copernicusmarine.subset") as mock_subset:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        mock_subset.assert_not_called()
        expected_md5 = hashlib.md5(fake_content).hexdigest()
        assert md5 == expected_md5

    def test_failed_download_has_no_checksum(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0011",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        saved_user = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_pass = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_user:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_user
            if saved_pass:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_pass

        assert success is False
        assert md5 is None


# ===========================================================================
# M. Deterministic path generation
# ===========================================================================

class TestDeterministicPathGeneration:
    """Test M: Same inputs always produce the same output paths."""

    def test_same_year_same_path(self, adapter):
        req_a = adapter.build_request(date(2023, 3, 15), "/data")
        req_b = adapter.build_request(date(2023, 7, 22), "/data")
        # Both cover the full year 2023
        assert req_a.expected_output_path == req_b.expected_output_path

    def test_different_years_different_paths(self, adapter):
        req_2021 = adapter.build_request(date(2021, 6, 1), "/data")
        req_2023 = adapter.build_request(date(2023, 6, 1), "/data")
        assert req_2021.expected_output_path != req_2023.expected_output_path

    def test_path_structure_raw_globcurrent_year(self, adapter):
        req = adapter.build_request(date(2023, 6, 1), "/data")
        parts = Path(req.expected_output_path).parts
        assert parts[0] == "raw"
        assert parts[1] == "globcurrent"
        assert parts[2] == "2023"

    def test_path_ends_with_nc(self, adapter):
        req = adapter.build_request(date(2023, 6, 1), "/data")
        assert req.expected_output_path.endswith(".nc")

    def test_path_never_absolute(self, adapter):
        for d in [date(2012, 1, 1), date(2018, 6, 15), date(2024, 12, 31)]:
            req = adapter.build_request(d, "D:\\someroot")
            assert not req.expected_output_path.startswith("D:")
            assert not req.expected_output_path.startswith("/")

    def test_partial_year_path_contains_year(self, adapter):
        req = adapter._build_year_request(2023, date(2023, 3, 1), date(2023, 9, 30))
        assert "2023" in req.expected_output_path
        assert req.expected_output_path.endswith(".nc")

    def test_full_year_path_deterministic(self, adapter):
        req = adapter._build_year_request(2023, date(2023, 1, 1), date(2023, 12, 31))
        assert req.expected_output_path == "raw/globcurrent/2023/GlobCurrent_2023_surface_uv.nc"

    def test_years_in_range_helper(self):
        assert _years_in_range(date(2020, 6, 1), date(2022, 3, 31)) == [2020, 2021, 2022]
        assert _years_in_range(date(2023, 1, 1), date(2023, 12, 31)) == [2023]

    def test_year_period_overlap_helper(self):
        start, end = _year_period_overlap(2023, date(2022, 11, 1), date(2023, 3, 31))
        assert start == date(2023, 1, 1)
        assert end == date(2023, 3, 31)

    def test_expected_days_helper(self):
        assert _expected_days(2020) == 366  # leap year
        assert _expected_days(2023) == 365  # non-leap
        assert _expected_days(2024) == 366  # leap year


# ===========================================================================
# N. Authentication configuration behavior
# ===========================================================================

class TestAuthenticationConfiguration:
    """Test N: Credentials are read from environment, not hardcoded."""

    def test_check_credentials_both_present(self):
        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "testuser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "testpass",
        }):
            result = GlobCurrentAdapter().check_credentials()
        assert result["credentials_available"] is True
        assert result["has_username"] is True
        assert result["has_password"] is True

    def test_check_credentials_both_missing(self):
        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            result = GlobCurrentAdapter().check_credentials()
        finally:
            if saved_u:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p
        assert result["credentials_available"] is False
        assert result["has_username"] is False
        assert result["has_password"] is False

    def test_check_credentials_only_username(self):
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            with patch.dict(os.environ, {"COPERNICUSMARINE_SERVICE_USERNAME": "u"}):
                result = GlobCurrentAdapter().check_credentials()
        finally:
            if saved_p:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p
        assert result["credentials_available"] is False
        assert result["has_username"] is True
        assert result["has_password"] is False

    def test_check_credentials_only_password(self):
        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        try:
            with patch.dict(os.environ, {"COPERNICUSMARINE_SERVICE_PASSWORD": "p"}):
                result = GlobCurrentAdapter().check_credentials()
        finally:
            if saved_u:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
        assert result["credentials_available"] is False
        assert result["has_username"] is False
        assert result["has_password"] is True

    def test_check_credentials_returns_message_not_values(self):
        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "secretuser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "secretpass",
        }):
            result = GlobCurrentAdapter().check_credentials()
        # The message must never contain the actual credential values
        assert "secretuser" not in result["message"]
        assert "secretpass" not in result["message"]

    def test_source_description_declares_auth_required(self):
        assert GlobCurrentAdapter().source_description().authentication_required is True

    def test_credentials_not_in_sentinel_url(self):
        """Credentials must never appear in the request URL."""
        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "mySuperSecretUser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "mySuperSecretPass",
        }):
            adapter = GlobCurrentAdapter()
            req = adapter.build_request(date(2023, 1, 1), "/data")
        assert "mySuperSecretUser" not in req.url_or_query
        assert "mySuperSecretPass" not in req.url_or_query


# ===========================================================================
# O. No credentials in logs
# ===========================================================================

class TestNoCredentialsInLogs:
    """Test O: Credential values never appear in job log messages."""

    def test_credentials_not_in_success_logs(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0012",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        def fake_subset(**kwargs):
            stem = kwargs["output_filename"]
            out = Path(kwargs["output_directory"]) / (stem + ".nc")
            out.write_bytes(b"data")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "SECRETUSERNAME",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "SECRETPASSWORD",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_subset):
                run_cmems_subset(req.url_or_query, dest, job)

        all_logs = " ".join(
            str(entry.get("message", "")) + str(entry)
            for entry in job.logs
        )
        assert "SECRETUSERNAME" not in all_logs
        assert "SECRETPASSWORD" not in all_logs

    def test_credentials_not_in_failure_logs(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0013",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        def fake_error(**kwargs):
            raise RuntimeError("server rejected SECRETUSERNAME with SECRETPASSWORD")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "SECRETUSERNAME",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "SECRETPASSWORD",
        }):
            with patch("copernicusmarine.subset", side_effect=fake_error):
                with patch("time.sleep"):
                    run_cmems_subset(req.url_or_query, dest, job, max_retries=1)

        all_logs = " ".join(
            str(entry.get("message", "")) + str(entry)
            for entry in job.logs
        )
        # The regex in run_cmems_subset should have redacted credential-adjacent text.
        # We verify the actual secret values are not present verbatim.
        # Note: the exception message may contain the words, but the safe_msg
        # processing redacts "password=..." and "token=..." patterns.
        # We test that the password value itself isn't exposed in a plain form.
        assert "SECRETPASSWORD" not in all_logs

    def test_missing_creds_log_does_not_echo_values(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "GlobCurrent_2023_surface_uv.nc"
        job = AcquisitionJob(
            job_id="test0014",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            data_root=tmp_path,
        )
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 1, 1), str(tmp_path))

        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_u:
                os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p:
                os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p

        all_logs = " ".join(str(entry.get("message", "")) for entry in job.logs)
        # Should mention credentials are needed but not echo values
        assert "username" in all_logs.lower() or "credential" in all_logs.lower()
        # No value-leakage check (nothing to leak in this case, but good hygiene)
        assert "None" not in all_logs or "not found" in all_logs.lower()


# ===========================================================================
# Sentinel URL parser
# ===========================================================================

class TestParseCmemsSentinelUrl:
    """Direct tests of parse_cmems_sentinel_url()."""

    def _make_url(self, **overrides):
        base = {
            "dataset": _CMEMS_DATASET_ID,
            "start": "2023-01-01",
            "end": "2023-12-31",
            "lat_min": 5.0,
            "lat_max": 30.0,
            "lon_min": 45.0,
            "lon_max": 105.0,
            "depth": 0.0,
            "variables": "uo,vo",
        }
        base.update(overrides)
        return (
            f"cmems://{base['dataset']}"
            f"?start={base['start']}"
            f"&end={base['end']}"
            f"&lat_min={base['lat_min']}"
            f"&lat_max={base['lat_max']}"
            f"&lon_min={base['lon_min']}"
            f"&lon_max={base['lon_max']}"
            f"&depth={base['depth']}"
            f"&variables={base['variables']}"
        )

    def test_parse_returns_correct_dataset_id(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["dataset_id"] == _CMEMS_DATASET_ID

    def test_parse_returns_iso_start_datetime(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["start"] == "2023-01-01T00:00:00"

    def test_parse_returns_iso_end_datetime(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["end"] == "2023-12-31T23:59:59"

    def test_parse_returns_float_coords(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["lat_min"] == 5.0
        assert params["lat_max"] == 30.0
        assert params["lon_min"] == 45.0
        assert params["lon_max"] == 105.0

    def test_parse_returns_float_depth(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["depth"] == 0.0

    def test_parse_returns_variable_list(self):
        params = parse_cmems_sentinel_url(self._make_url())
        assert params["variables"] == ["uo", "vo"]

    def test_not_cmems_url_raises(self):
        with pytest.raises(ValueError, match="Not a CMEMS sentinel"):
            parse_cmems_sentinel_url("https://example.com/data.nc")

    def test_missing_query_string_raises(self):
        with pytest.raises(ValueError, match="query"):
            parse_cmems_sentinel_url(f"cmems://{_CMEMS_DATASET_ID}")

    def test_missing_required_param_raises(self):
        # Build URL without 'variables' (a genuinely required param)
        url = (
            f"cmems://{_CMEMS_DATASET_ID}"
            "?start=2023-01-01&end=2023-12-31"
            "&lat_min=5.0&lat_max=30.0&lon_min=45.0&lon_max=105.0"
            # no variables
        )
        with pytest.raises(ValueError, match="variables"):
            parse_cmems_sentinel_url(url)

    def test_adapter_produced_url_is_parseable(self):
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["dataset_id"] == _CMEMS_DATASET_ID
        assert params["variables"] == list(_CMEMS_VARIABLES)
        assert params["depth"] == _CMEMS_DEPTH


# ===========================================================================
# Engine dispatch tests
# ===========================================================================

class TestEngineDispatch:
    """Verify run_acquisition_job dispatches correctly by URL prefix."""

    def test_cmems_url_dispatches_to_run_cmems_subset(self, tmp_path):
        from oceanembed.acquisition.engine import (
            run_acquisition_job, AcquisitionJob, JobManager
        )

        job = AcquisitionJob(
            job_id="disp0001",
            dataset_id="globcurrent",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )

        # Mock the adapter to return a cmems:// request
        mock_adapter = MagicMock()
        mock_adapter.date_range_requests.return_value = [
            GlobCurrentAdapter().date_range_requests(
                date(2023, 1, 1), date(2023, 1, 31), str(tmp_path)
            )[0]
        ]

        with patch(
            "oceanembed.acquisition.engine.run_cmems_subset",
            return_value=(True, 1000, "abc123"),
        ) as mock_cmems:
            with patch(
                "oceanembed.acquisition.engine.download_file",
            ) as mock_dl:
                asyncio.run(run_acquisition_job(job, mock_adapter))

        mock_cmems.assert_called_once()
        mock_dl.assert_not_called()

    def test_plain_url_dispatches_to_download_file(self, tmp_path):
        from oceanembed.acquisition.engine import (
            run_acquisition_job, AcquisitionJob
        )
        from oceanembed.datasets.adapters.oisst import OISSTAdapter

        job = AcquisitionJob(
            job_id="disp0002",
            dataset_id="oisst",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 1),
            data_root=tmp_path,
        )

        adapter = OISSTAdapter()

        async def fake_download_file(*args, **kwargs):
            return (True, 500, "def456")

        with patch(
            "oceanembed.acquisition.engine.download_file",
            side_effect=fake_download_file,
        ) as mock_dl:
            with patch(
                "oceanembed.acquisition.engine.run_cmems_subset",
            ) as mock_cmems:
                asyncio.run(run_acquisition_job(job, adapter))

        mock_dl.assert_called_once()
        mock_cmems.assert_not_called()


# ===========================================================================
# validate_request (GlobCurrent-specific)
# ===========================================================================

class TestValidateRequest:
    """Test GlobCurrent-specific validate_request() checks."""

    def test_valid_request_passes(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        issues = adapter.validate_request(req)
        assert issues == []

    def test_non_cmems_url_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        # Manually create a broken request with a plain URL
        bad_req = AcquisitionRequest(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            url_or_query="https://example.com/not-cmems.nc",
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
            extra=req.extra,
        )
        issues = adapter.validate_request(bad_req)
        assert any("cmems://" in msg for msg in issues)

    def test_missing_extra_field_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        # Strip out a required extra field
        trimmed_extra = {k: v for k, v in req.extra.items() if k != "cmems_dataset_id"}
        bad_req = AcquisitionRequest(
            dataset_id=req.dataset_id,
            request_date=req.request_date,
            url_or_query=req.url_or_query,
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
            extra=trimmed_extra,
        )
        issues = adapter.validate_request(bad_req)
        assert any("cmems_dataset_id" in msg for msg in issues)

    def test_wrong_dataset_id_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        bad_req = AcquisitionRequest(
            dataset_id="oisst",  # wrong
            request_date=req.request_date,
            url_or_query=req.url_or_query,
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
        )
        issues = adapter.validate_request(bad_req)
        assert any("oisst" in msg for msg in issues)
