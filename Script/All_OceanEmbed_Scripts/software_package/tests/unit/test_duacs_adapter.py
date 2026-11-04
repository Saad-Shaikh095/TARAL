"""
Unit tests for the DUACS dataset adapter.

Test matrix (A-T from prompt spec)
------------------------------------
A. Adapter identity
B. Valid small date range
C. Valid small bounding box
D. Different valid bounding box
E. Different valid date range
F. Invalid date range
G. Invalid latitude
H. Invalid longitude
I. Source coverage boundaries
J. Existing file reuse
K. Failed download / retry behavior
L. Authentication configuration
M. No credentials in logs
N. Deterministic output path
O. Manifest behavior
P. Checksum behavior
Q. Engine dispatch (cmems:// -> run_cmems_subset, plain -> download_file)
R. Correct CMEMS request construction (no depth args for DUACS)
S. Correct DUACS variable selection
T. Correct dataset/product selection

Additional coverage
-------------------
- Monthly chunking helpers (_months_in_range, _month_period_overlap, _is_full_month)
- parse_cmems_sentinel_url depth=None regression (GlobCurrent still works)
- validate_request() DUACS-specific depth check
- API registration

All network access is mocked. No real CMEMS calls are made.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oceanembed.datasets.adapters.duacs import (
    DUACSAdapter,
    _CMEMS_DATASET_ID,
    _CMEMS_PRODUCT_ID,
    _CMEMS_VARIABLES,
    _SOURCE_COVERAGE_START,
    _SOURCE_COVERAGE_END,
    _validate_duacs_request_params,
    _months_in_range,
    _month_period_overlap,
    _is_full_month,
)
from oceanembed.datasets.adapters.base import AcquisitionRequest, SourceDescription
from oceanembed.datasets.adapters.globcurrent import parse_cmems_sentinel_url


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    return DUACSAdapter()


@pytest.fixture
def data_root(tmp_path):
    return str(tmp_path)


# ===========================================================================
# A. Adapter identity
# ===========================================================================

class TestAdapterIdentity:
    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "duacs"

    def test_repr(self, adapter):
        assert "DUACSAdapter" in repr(adapter)
        assert "duacs" in repr(adapter)

    def test_source_description_type(self, adapter):
        assert isinstance(adapter.source_description(), SourceDescription)

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "duacs"

    def test_authentication_required(self, adapter):
        assert adapter.source_description().authentication_required is True

    def test_temporal_resolution(self, adapter):
        assert adapter.source_description().native_temporal_resolution == "daily"

    def test_spatial_resolution_0125(self, adapter):
        assert adapter.source_description().native_spatial_resolution_degrees == 0.125

    def test_variables_sla_adt_ugos_vgos(self, adapter):
        vars_ = adapter.source_description().variables_requested
        for v in ["sla", "adt", "ugos", "vgos"]:
            assert v in vars_

    def test_exactly_four_variables(self, adapter):
        assert len(adapter.source_description().variables_requested) == 4

    def test_max_job_days_366(self, adapter):
        assert adapter._max_job_days == 366

    def test_cmems_dataset_id_in_request(self, adapter):
        req = adapter.build_request(date(2023, 6, 1), "/data")
        assert _CMEMS_DATASET_ID in req.url_or_query


# ===========================================================================
# B. Valid small date range
# ===========================================================================

class TestValidSmallDateRange:
    def test_one_week_one_request(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        assert len(reqs) == 1

    def test_request_dataset_id(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        assert reqs[0].dataset_id == "duacs"

    def test_url_starts_with_cmems(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        assert reqs[0].url_or_query.startswith("cmems://")

    def test_output_path_relative(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        path = reqs[0].expected_output_path
        assert not path.startswith("/")
        assert "\\" not in path

    def test_validate_request_passes(self, adapter, data_root):
        req = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)[0]
        assert adapter.validate_request(req) == []

    def test_full_month_filename(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 6, 1), date(2023, 6, 30), data_root)
        assert Path(reqs[0].expected_output_path).name == "duacs_2023_06.nc"

    def test_partial_month_filename_has_days(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 6, 5), date(2023, 6, 20), data_root)
        fname = Path(reqs[0].expected_output_path).name
        assert "2023_06" in fname
        # partial: should not match the clean full-month name
        assert fname != "duacs_2023_06.nc"

    def test_no_depth_in_sentinel_url(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 7), data_root)
        assert "depth=" not in reqs[0].url_or_query

    def test_single_day_range(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 3, 15), date(2023, 3, 15), data_root)
        assert len(reqs) == 1


# ===========================================================================
# C. Valid small bounding box
# ===========================================================================

class TestValidSmallBoundingBox:
    def test_small_bbox_no_validation_errors(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 7), 12.0, 18.0, 70.0, 80.0,
        )
        assert errors == []

    def test_spatial_region_populated(self):
        a = DUACSAdapter(min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0)
        req = a.build_request(date(2023, 6, 1), "/data")
        assert req.spatial_region["lat_min"] == 12.0
        assert req.spatial_region["lon_max"] == 80.0

    def test_bbox_reflected_in_parsed_url(self):
        a = DUACSAdapter(min_lat=12.0, max_lat=18.0, min_lon=70.0, max_lon=80.0)
        params = parse_cmems_sentinel_url(a.build_request(date(2023, 6, 1), "/data").url_or_query)
        assert params["lat_min"] == 12.0
        assert params["lon_max"] == 80.0

    def test_parsed_depth_is_none(self):
        req = DUACSAdapter().build_request(date(2023, 1, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["depth"] is None


# ===========================================================================
# D. Different valid bounding box
# ===========================================================================

class TestDifferentValidBoundingBox:
    def test_different_bbox_different_url(self):
        a = DUACSAdapter(min_lat=5.0, max_lat=30.0, min_lon=45.0, max_lon=105.0)
        b = DUACSAdapter(min_lat=10.0, max_lat=25.0, min_lon=60.0, max_lon=90.0)
        assert (a.build_request(date(2023, 1, 1), "/data").url_or_query !=
                b.build_request(date(2023, 1, 1), "/data").url_or_query)

    def test_southern_hemisphere_valid(self):
        assert _validate_duacs_request_params(
            date(2020, 1, 1), date(2020, 6, 30), -40.0, -10.0, -60.0, 20.0,
        ) == []

    def test_global_bbox_valid(self):
        assert _validate_duacs_request_params(
            date(2020, 1, 1), date(2020, 12, 31), -90.0, 90.0, -180.0, 180.0,
        ) == []


# ===========================================================================
# E. Different valid date range
# ===========================================================================

class TestDifferentValidDateRange:
    def test_two_months_two_requests(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 6, 1), date(2023, 7, 31), data_root)
        assert len(reqs) == 2

    def test_cross_year_boundary_four_months(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2022, 11, 1), date(2023, 2, 28), data_root)
        assert len(reqs) == 4
        assert [(r.request_date.year, r.request_date.month) for r in reqs] == [
            (2022, 11), (2022, 12), (2023, 1), (2023, 2)
        ]

    def test_cross_year_overlap_period_correctness(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2022, 12, 15), date(2023, 1, 10), data_root)
        assert len(reqs) == 2
        p_dec = parse_cmems_sentinel_url(reqs[0].url_or_query)
        assert p_dec["start"] == "2022-12-15T00:00:00"
        assert p_dec["end"] == "2022-12-31T23:59:59"
        p_jan = parse_cmems_sentinel_url(reqs[1].url_or_query)
        assert p_jan["start"] == "2023-01-01T00:00:00"
        assert p_jan["end"] == "2023-01-10T23:59:59"

    def test_different_years_different_paths(self, adapter, data_root):
        r2021 = adapter.date_range_requests(date(2021, 3, 1), date(2021, 3, 31), data_root)
        r2023 = adapter.date_range_requests(date(2023, 3, 1), date(2023, 3, 31), data_root)
        assert r2021[0].expected_output_path != r2023[0].expected_output_path
        assert "2021" in r2021[0].expected_output_path
        assert "2023" in r2023[0].expected_output_path

    def test_historical_1993_boundary_valid(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(1993, 1, 1), date(1993, 1, 31), data_root)
        assert len(reqs) == 1


# ===========================================================================
# F. Invalid date range
# ===========================================================================

class TestInvalidDateRange:
    def test_reversed_dates_raise(self, adapter, data_root):
        with pytest.raises(ValueError, match="start_date"):
            adapter.date_range_requests(date(2023, 6, 1), date(2023, 1, 1), data_root)

    def test_error_contains_both_dates(self, adapter, data_root):
        with pytest.raises(ValueError) as exc_info:
            adapter.date_range_requests(date(2023, 6, 1), date(2023, 1, 1), data_root)
        assert "2023-06-01" in str(exc_info.value)
        assert "2023-01-01" in str(exc_info.value)

    def test_same_date_valid(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023, 6, 15), date(2023, 6, 15), data_root)
        assert len(reqs) == 1

    def test_validation_helper_reversed(self):
        errors = _validate_duacs_request_params(
            date(2023, 6, 1), date(2023, 1, 1), 5.0, 30.0, 45.0, 105.0,
        )
        assert any("start_date" in e for e in errors)


# ===========================================================================
# G. Invalid latitude
# ===========================================================================

class TestInvalidLatitude:
    def test_min_lat_greater_than_max_lat(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 30.0, 5.0, 45.0, 105.0,
        )
        assert any("min_lat" in e for e in errors)

    def test_equal_lat_rejected(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 15.0, 15.0, 45.0, 105.0,
        )
        assert any("min_lat" in e for e in errors)

    def test_lat_below_minus_90(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), -91.0, 30.0, 45.0, 105.0,
        )
        assert len(errors) > 0

    def test_lat_above_plus_90(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 5.0, 91.0, 45.0, 105.0,
        )
        assert len(errors) > 0

    def test_bad_lat_raises_on_use(self):
        with pytest.raises(ValueError):
            DUACSAdapter(min_lat=50.0, max_lat=10.0).date_range_requests(
                date(2023, 1, 1), date(2023, 1, 31), "/data"
            )


# ===========================================================================
# H. Invalid longitude
# ===========================================================================

class TestInvalidLongitude:
    def test_min_lon_greater_than_max_lon(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 5.0, 30.0, 105.0, 45.0,
        )
        assert any("min_lon" in e for e in errors)

    def test_equal_lon_rejected(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 5.0, 30.0, 75.0, 75.0,
        )
        assert any("min_lon" in e for e in errors)

    def test_lon_below_minus_180(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 5.0, 30.0, -181.0, 105.0,
        )
        assert len(errors) > 0

    def test_lon_above_plus_180(self):
        errors = _validate_duacs_request_params(
            date(2023, 1, 1), date(2023, 1, 31), 5.0, 30.0, 45.0, 181.0,
        )
        assert len(errors) > 0

    def test_bad_lon_raises_on_use(self):
        with pytest.raises(ValueError):
            DUACSAdapter(min_lon=105.0, max_lon=45.0).date_range_requests(
                date(2023, 1, 1), date(2023, 1, 31), "/data"
            )


# ===========================================================================
# I. Source coverage boundaries
# ===========================================================================

class TestSourceCoverageBoundaries:
    def test_before_1993_rejected(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(date(1992, 12, 31), date(1993, 1, 15), data_root)

    def test_after_coverage_end_rejected(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(date(2024, 12, 1), date(2025, 3, 1), data_root)

    def test_exact_start_accepted(self, adapter, data_root):
        reqs = adapter.date_range_requests(
            _SOURCE_COVERAGE_START, _SOURCE_COVERAGE_START, data_root
        )
        assert len(reqs) == 1

    def test_exact_end_accepted(self, adapter, data_root):
        reqs = adapter.date_range_requests(
            _SOURCE_COVERAGE_END, _SOURCE_COVERAGE_END, data_root
        )
        assert len(reqs) == 1

    def test_error_mentions_coverage_start(self, adapter, data_root):
        with pytest.raises(ValueError) as exc_info:
            adapter.date_range_requests(date(1990, 1, 1), date(1990, 12, 31), data_root)
        assert str(_SOURCE_COVERAGE_START) in str(exc_info.value)

    def test_validation_helper_before_1993(self):
        errors = _validate_duacs_request_params(
            date(1992, 6, 1), date(1992, 12, 31), 5.0, 30.0, 45.0, 105.0,
        )
        assert any("1992" in e or "coverage" in e.lower() for e in errors)


# ===========================================================================
# J. Existing file reuse
# ===========================================================================

class TestExistingFileReuse:
    def test_non_empty_file_skipped(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "raw" / "duacs" / "2023" / "duacs_2023_01.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake duacs netcdf content")

        job = AcquisitionJob(
            job_id="j_reuse01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        with patch("copernicusmarine.subset") as mock_sub:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        mock_sub.assert_not_called()
        assert success is True
        assert md5 is not None

    def test_empty_file_not_reused(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        dest.write_bytes(b"")  # empty — must not be reused

        job = AcquisitionJob(
            job_id="j_reuse02", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            success, _, _ = run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_u: os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p: os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p

        # No credentials → must fail, but not skip (empty file is not valid)
        assert success is False


# ===========================================================================
# K. Failed download / retry behavior
# ===========================================================================

class TestFailedDownloadRetry:
    def test_missing_credentials_fails(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_retry01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_u: os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p: os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p

        assert success is False
        assert nbytes == 0
        assert md5 is None

    def test_transient_error_retried(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_retry02", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        calls = {"n": 0}

        def fake(**kw):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"data")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake):
                with patch("time.sleep"):
                    success, _, _ = run_cmems_subset(req.url_or_query, dest, job, max_retries=3)

        assert calls["n"] == 2
        assert success is True

    def test_auth_error_not_retried(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_retry03", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        calls = {"n": 0}

        def auth_err(**kw):
            calls["n"] += 1
            raise RuntimeError("HTTP 401 unauthorized")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "bad",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "bad",
        }):
            with patch("copernicusmarine.subset", side_effect=auth_err):
                success, _, _ = run_cmems_subset(req.url_or_query, dest, job, max_retries=3)

        assert calls["n"] == 1
        assert success is False


# ===========================================================================
# L. Authentication configuration
# ===========================================================================

class TestAuthenticationConfiguration:
    def test_both_credentials_detected(self):
        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "testuser",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "testpass",
        }):
            result = DUACSAdapter().check_credentials()
        assert result["credentials_available"] is True

    def test_both_missing(self):
        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            result = DUACSAdapter().check_credentials()
        finally:
            if saved_u: os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p: os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p
        assert result["credentials_available"] is False

    def test_only_username_insufficient(self):
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            with patch.dict(os.environ, {"COPERNICUSMARINE_SERVICE_USERNAME": "u"}):
                result = DUACSAdapter().check_credentials()
        finally:
            if saved_p: os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p
        assert result["credentials_available"] is False
        assert result["has_username"] is True
        assert result["has_password"] is False

    def test_credentials_not_in_sentinel_url(self):
        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "SECRETDUACSUSER",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "SECRETDUACSPASS",
        }):
            req = DUACSAdapter().build_request(date(2023, 1, 1), "/data")
        assert "SECRETDUACSUSER" not in req.url_or_query
        assert "SECRETDUACSPASS" not in req.url_or_query


# ===========================================================================
# M. No credentials in logs
# ===========================================================================

class TestNoCredentialsInLogs:
    def test_success_logs_no_creds(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_logs01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        def fake(**kw):
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"ok")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "DUACS_SECRET_USER",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "DUACS_SECRET_PASS",
        }):
            with patch("copernicusmarine.subset", side_effect=fake):
                run_cmems_subset(req.url_or_query, dest, job)

        all_logs = " ".join(str(e.get("message", "")) + str(e) for e in job.logs)
        assert "DUACS_SECRET_USER" not in all_logs
        assert "DUACS_SECRET_PASS" not in all_logs

    def test_failure_logs_no_password(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_logs02", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        def err(**kw):
            raise RuntimeError("reject DUACS_SECRET_PASS value")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "DUACS_SECRET_PASS",
        }):
            with patch("copernicusmarine.subset", side_effect=err):
                with patch("time.sleep"):
                    run_cmems_subset(req.url_or_query, dest, job, max_retries=1)

        all_logs = " ".join(str(e.get("message", "")) + str(e) for e in job.logs)
        assert "DUACS_SECRET_PASS" not in all_logs


# ===========================================================================
# N. Deterministic output path
# ===========================================================================

class TestDeterministicPath:
    def test_full_month_path(self, adapter):
        req = adapter._build_month_request(2023, 6, date(2023, 6, 1), date(2023, 6, 30))
        assert req.expected_output_path == "raw/duacs/2023/duacs_2023_06.nc"

    def test_partial_month_path_contains_year_month(self, adapter):
        req = adapter._build_month_request(2023, 6, date(2023, 6, 5), date(2023, 6, 20))
        assert "2023" in req.expected_output_path
        assert "06" in req.expected_output_path
        assert req.expected_output_path.endswith(".nc")

    def test_path_never_absolute(self, adapter):
        req = adapter.build_request(date(2023, 6, 1), "D:\\data")
        assert not req.expected_output_path.startswith("D:")
        assert not req.expected_output_path.startswith("/")

    def test_different_months_different_paths(self, adapter):
        r1 = adapter.build_request(date(2023, 1, 1), "/data")
        r6 = adapter.build_request(date(2023, 6, 1), "/data")
        assert r1.expected_output_path != r6.expected_output_path

    def test_same_month_same_path(self, adapter):
        # build_request for any day in the same month → same full-month file
        r1 = adapter.build_request(date(2023, 6, 1), "/data")
        r15 = adapter.build_request(date(2023, 6, 15), "/data")
        assert r1.expected_output_path == r15.expected_output_path

    def test_raw_duacs_year_directory_structure(self, adapter):
        req = adapter.build_request(date(2023, 6, 1), "/data")
        parts = Path(req.expected_output_path).parts
        assert parts[0] == "raw"
        assert parts[1] == "duacs"
        assert parts[2] == "2023"

    def test_months_in_range_helper(self):
        assert _months_in_range(date(2020, 11, 1), date(2021, 2, 28)) == [
            (2020, 11), (2020, 12), (2021, 1), (2021, 2),
        ]

    def test_month_period_overlap_helper_start(self):
        start, end = _month_period_overlap(2023, 6, date(2023, 5, 15), date(2023, 6, 20))
        assert start == date(2023, 6, 1)
        assert end == date(2023, 6, 20)

    def test_is_full_month_helper_true(self):
        assert _is_full_month(2023, 6, date(2023, 6, 1), date(2023, 6, 30)) is True

    def test_is_full_month_helper_false(self):
        assert _is_full_month(2023, 6, date(2023, 6, 1), date(2023, 6, 29)) is False

    def test_is_full_month_leap_year(self):
        assert _is_full_month(2020, 2, date(2020, 2, 1), date(2020, 2, 29)) is True


# ===========================================================================
# O. Manifest behavior
# ===========================================================================

class TestManifestBehavior:
    def test_manifest_written_on_success(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

        dest = tmp_path / "raw" / "duacs" / "2023" / "duacs_2023_01.nc"
        dest.parent.mkdir(parents=True, exist_ok=True)
        job = AcquisitionJob(
            job_id="j_mfst01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        def fake(**kw):
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"duacs")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake):
                success, nbytes, md5 = run_cmems_subset(req.url_or_query, dest, job)

        assert success is True
        manifest = ManifestStore(tmp_path)
        manifest.record(ManifestEntry(
            dataset_id=req.dataset_id, request_date=req.request_date,
            source_url=req.url_or_query, local_path=str(dest),
            product_id=req.source_description.product_id,
            file_size_bytes=nbytes, md5_checksum=md5,
            status=ManifestStatus.SUCCESS,
        ))
        records = manifest.read_all("duacs")
        assert len(records) == 1
        assert records[0]["status"] == "SUCCESS"

    def test_duacs_manifest_separate_from_other_datasets(self, tmp_path):
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus

        m = ManifestStore(tmp_path)
        for ds in ("globcurrent", "oisst", "duacs"):
            m.record(ManifestEntry(
                dataset_id=ds, request_date=date(2023, 1, 1),
                source_url="cmems://fake", local_path=f"/fake/{ds}.nc",
                product_id="FAKE", status=ManifestStatus.SUCCESS,
            ))
        assert len(m.read_all("duacs")) == 1
        assert len(m.read_all("oisst")) == 1
        assert len(m.read_all("globcurrent")) == 1


# ===========================================================================
# P. Checksum behavior
# ===========================================================================

class TestChecksumBehavior:
    def test_checksum_computed_on_download(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_cksum01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        content = b"duacs checksum content"

        def fake(**kw):
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(content)

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=fake):
                success, _, md5 = run_cmems_subset(req.url_or_query, dest, job)

        assert success is True
        assert md5 == hashlib.md5(content).hexdigest()

    def test_checksum_computed_for_existing_file(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        content = b"existing duacs file"
        dest = tmp_path / "duacs_2023_01.nc"
        dest.write_bytes(content)
        job = AcquisitionJob(
            job_id="j_cksum02", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        with patch("copernicusmarine.subset") as mock_sub:
            success, _, md5 = run_cmems_subset(req.url_or_query, dest, job)

        mock_sub.assert_not_called()
        assert md5 == hashlib.md5(content).hexdigest()

    def test_failed_download_md5_none(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_cksum03", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))

        saved_u = os.environ.pop("COPERNICUSMARINE_SERVICE_USERNAME", None)
        saved_p = os.environ.pop("COPERNICUSMARINE_SERVICE_PASSWORD", None)
        try:
            success, _, md5 = run_cmems_subset(req.url_or_query, dest, job)
        finally:
            if saved_u: os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = saved_u
            if saved_p: os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = saved_p

        assert success is False
        assert md5 is None


# ===========================================================================
# Q. Engine dispatch
# ===========================================================================

class TestEngineDispatch:
    def test_cmems_url_dispatches_to_run_cmems_subset(self, tmp_path):
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob

        job = AcquisitionJob(
            job_id="j_disp01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        mock_adapter = MagicMock()
        mock_adapter.date_range_requests.return_value = [
            DUACSAdapter().date_range_requests(
                date(2023, 1, 1), date(2023, 1, 31), str(tmp_path)
            )[0]
        ]

        with patch("oceanembed.acquisition.engine.run_cmems_subset",
                   return_value=(True, 1000, "abc")) as mock_cmems:
            with patch("oceanembed.acquisition.engine.download_file") as mock_dl:
                asyncio.run(run_acquisition_job(job, mock_adapter))

        mock_cmems.assert_called_once()
        mock_dl.assert_not_called()

    def test_oisst_plain_url_dispatches_to_download_file(self, tmp_path):
        """OISST path must remain unaffected by DUACS changes."""
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob
        from oceanembed.datasets.adapters.oisst import OISSTAdapter

        job = AcquisitionJob(
            job_id="j_disp02", dataset_id="oisst",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 1),
            data_root=tmp_path,
        )

        async def fake_dl(*a, **kw):
            return (True, 500, "abc")

        with patch("oceanembed.acquisition.engine.download_file",
                   side_effect=fake_dl) as mock_dl:
            with patch("oceanembed.acquisition.engine.run_cmems_subset") as mock_cmems:
                asyncio.run(run_acquisition_job(job, OISSTAdapter()))

        mock_dl.assert_called_once()
        mock_cmems.assert_not_called()


# ===========================================================================
# R. Correct CMEMS request construction (no depth)
# ===========================================================================

class TestCmemsRequestConstruction:
    def test_no_depth_in_parsed_url(self):
        params = parse_cmems_sentinel_url(
            DUACSAdapter().build_request(date(2023, 1, 1), "/data").url_or_query
        )
        assert params["depth"] is None

    def test_engine_subset_not_called_with_depth_args(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_nodepth01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        captured: list[dict] = []

        def cap(**kw):
            captured.append(dict(kw))
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"ok")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=cap):
                run_cmems_subset(req.url_or_query, dest, job)

        assert "minimum_depth" not in captured[0]
        assert "maximum_depth" not in captured[0]

    def test_correct_date_range_in_request(self, adapter):
        req = adapter._build_month_request(2023, 1, date(2023, 1, 1), date(2023, 1, 31))
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["start"] == "2023-01-01T00:00:00"
        assert params["end"] == "2023-01-31T23:59:59"

    def test_partial_month_dates_in_request(self, adapter):
        req = adapter._build_month_request(2023, 1, date(2023, 1, 5), date(2023, 1, 20))
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["start"] == "2023-01-05T00:00:00"
        assert params["end"] == "2023-01-20T23:59:59"


# ===========================================================================
# S. Variable selection
# ===========================================================================

class TestVariableSelection:
    def test_all_four_variables_in_url(self):
        params = parse_cmems_sentinel_url(
            DUACSAdapter().build_request(date(2023, 1, 1), "/data").url_or_query
        )
        assert set(params["variables"]) == {"sla", "adt", "ugos", "vgos"}

    def test_exactly_four_variables(self):
        params = parse_cmems_sentinel_url(
            DUACSAdapter().build_request(date(2023, 1, 1), "/data").url_or_query
        )
        assert len(params["variables"]) == 4

    def test_removed_variables_absent(self):
        params = parse_cmems_sentinel_url(
            DUACSAdapter().build_request(date(2023, 1, 1), "/data").url_or_query
        )
        removed = {"ugosa", "vgosa", "err_sla", "err_ugosa", "err_vgosa", "flag_ice"}
        for v in removed:
            assert v not in params["variables"], f"Removed var {v!r} must not be requested"

    def test_engine_receives_correct_variables(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_vars01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        captured: list[dict] = []

        def cap(**kw):
            captured.append(dict(kw))
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"ok")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=cap):
                run_cmems_subset(req.url_or_query, dest, job)

        assert set(captured[0]["variables"]) == {"sla", "adt", "ugos", "vgos"}


# ===========================================================================
# T. Dataset/product selection
# ===========================================================================

class TestDatasetProductSelection:
    def test_dataset_id_in_url(self):
        req = DUACSAdapter().build_request(date(2023, 1, 1), "/data")
        assert _CMEMS_DATASET_ID in req.url_or_query

    def test_dataset_id_no_version_suffix(self):
        """Version suffix must be omitted — copernicusmarine warns if included."""
        req = DUACSAdapter().build_request(date(2023, 1, 1), "/data")
        # The dataset ID itself contains _P1D — check the version suffix _202411 is absent
        assert "_P1D_202411" not in req.url_or_query

    def test_extra_has_no_depth_flag(self):
        req = DUACSAdapter().build_request(date(2023, 1, 1), "/data")
        assert req.extra["has_depth"] is False

    def test_engine_receives_correct_dataset_id(self, tmp_path):
        from oceanembed.acquisition.engine import run_cmems_subset, AcquisitionJob

        dest = tmp_path / "duacs_2023_01.nc"
        job = AcquisitionJob(
            job_id="j_dsid01", dataset_id="duacs",
            start_date=date(2023, 1, 1), end_date=date(2023, 1, 31),
            data_root=tmp_path,
        )
        req = DUACSAdapter().build_request(date(2023, 1, 1), str(tmp_path))
        captured: list[dict] = []

        def cap(**kw):
            captured.append(dict(kw))
            stem = kw["output_filename"]
            (Path(kw["output_directory"]) / (stem + ".nc")).write_bytes(b"ok")

        with patch.dict(os.environ, {
            "COPERNICUSMARINE_SERVICE_USERNAME": "u",
            "COPERNICUSMARINE_SERVICE_PASSWORD": "p",
        }):
            with patch("copernicusmarine.subset", side_effect=cap):
                run_cmems_subset(req.url_or_query, dest, job)

        assert captured[0]["dataset_id"] == _CMEMS_DATASET_ID

    def test_product_version_in_source_description(self):
        assert DUACSAdapter().source_description().product_version == "202411"


# ===========================================================================
# parse_cmems_sentinel_url — depth optional regression
# ===========================================================================

class TestParseSentinelUrlDepthHandling:
    def test_depth_present_returns_float(self):
        url = ("cmems://ds?start=2023-01-01&end=2023-12-31"
               "&lat_min=5.0&lat_max=30.0&lon_min=45.0&lon_max=105.0"
               "&depth=0.0&variables=uo,vo")
        assert parse_cmems_sentinel_url(url)["depth"] == 0.0

    def test_depth_absent_returns_none(self):
        url = ("cmems://ds?start=2023-01-01&end=2023-01-31"
               "&lat_min=5.0&lat_max=30.0&lon_min=45.0&lon_max=105.0"
               "&variables=sla,adt,ugos,vgos")
        assert parse_cmems_sentinel_url(url)["depth"] is None

    def test_globcurrent_url_still_parsed_correctly(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        req = GlobCurrentAdapter().build_request(date(2023, 6, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["depth"] == 0.0
        assert params["variables"] == ["uo", "vo"]

    def test_duacs_url_depth_none(self):
        req = DUACSAdapter().build_request(date(2023, 6, 1), "/data")
        assert parse_cmems_sentinel_url(req.url_or_query)["depth"] is None

    def test_missing_required_param_raises(self):
        url = "cmems://ds?start=2023-01-01&end=2023-01-31&lat_min=5.0&lat_max=30.0"
        with pytest.raises(ValueError, match="missing required parameters"):
            parse_cmems_sentinel_url(url)


# ===========================================================================
# validate_request — DUACS specific
# ===========================================================================

class TestValidateRequest:
    def test_valid_passes(self, adapter, data_root):
        assert adapter.validate_request(
            adapter.build_request(date(2023, 6, 1), data_root)
        ) == []

    def test_non_cmems_url_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        bad = AcquisitionRequest(
            dataset_id=req.dataset_id, request_date=req.request_date,
            url_or_query="https://example.com/file.nc",
            expected_output_path=req.expected_output_path,
            source_description=req.source_description, extra=req.extra,
        )
        assert any("cmems://" in m for m in adapter.validate_request(bad))

    def test_depth_in_url_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023, 6, 1), data_root)
        bad = AcquisitionRequest(
            dataset_id=req.dataset_id, request_date=req.request_date,
            url_or_query=req.url_or_query + "&depth=0.0",
            expected_output_path=req.expected_output_path,
            source_description=req.source_description, extra=req.extra,
        )
        assert any("depth" in m.lower() for m in adapter.validate_request(bad))


# ===========================================================================
# API registration
# ===========================================================================

class TestApiRegistration:
    def test_duacs_adapter_implemented(self):
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        app_module._registry = None
        resp = TestClient(app_module.app).get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "duacs")
        assert ds["adapter_implemented"] is True
        assert ds["engineering_status"] == "ACQUISITION_READY"
