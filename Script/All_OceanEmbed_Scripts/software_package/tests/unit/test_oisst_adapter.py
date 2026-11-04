"""
Unit tests for the OISSTAdapter.

Tests cover:
- URL construction for known dates
- Output path structure
- SourceDescription fields
- date_range_requests generates correct count
- No scientific transformation is applied (verified by checking adapter raises no errors
  and produces only path/URL metadata, not data)
- Absolute paths are rejected
"""

from __future__ import annotations

from datetime import date

import pytest

from oceanembed.datasets.adapters.oisst import OISSTAdapter, _NCEI_BASE


class TestOISSTAdapterIdentity:
    def test_dataset_id(self):
        assert OISSTAdapter().dataset_id == "oisst"

    def test_source_description_dataset_id(self):
        desc = OISSTAdapter().source_description()
        assert desc.dataset_id == "oisst"

    def test_source_description_product_id(self):
        desc = OISSTAdapter().source_description()
        assert desc.product_id == "AVHRR_OI-NCEI-L4-GLOB-v2.1"

    def test_source_description_auth_not_required(self):
        desc = OISSTAdapter().source_description()
        assert desc.authentication_required is False

    def test_source_description_temporal_resolution(self):
        desc = OISSTAdapter().source_description()
        assert desc.native_temporal_resolution == "daily"

    def test_source_description_spatial_resolution(self):
        desc = OISSTAdapter().source_description()
        assert desc.native_spatial_resolution_degrees == 0.25

    def test_source_description_variables(self):
        desc = OISSTAdapter().source_description()
        assert "sst" in desc.variables_requested


class TestOISSTAdapterBuildRequest:
    @pytest.fixture
    def adapter(self):
        return OISSTAdapter()

    def test_build_request_dataset_id(self, adapter):
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert req.dataset_id == "oisst"

    def test_build_request_date(self, adapter):
        d = date(2023, 6, 1)
        req = adapter.build_request(d, "/data")
        assert req.request_date == d

    def test_build_request_url_starts_with_ncei(self, adapter):
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert req.url_or_query.startswith(_NCEI_BASE)

    def test_build_request_url_contains_date_compact(self, adapter):
        req = adapter.build_request(date(2023, 3, 22), "/data")
        assert "20230322" in req.url_or_query

    def test_build_request_url_contains_ym(self, adapter):
        req = adapter.build_request(date(2023, 3, 22), "/data")
        assert "202303" in req.url_or_query

    def test_build_request_url_contains_filename(self, adapter):
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert "oisst-avhrr-v02r01.20230115.nc" in req.url_or_query

    def test_build_request_output_path_relative(self, adapter):
        req = adapter.build_request(date(2023, 1, 15), "/data")
        assert not req.expected_output_path.startswith("/")
        assert "\\" not in req.expected_output_path

    def test_build_request_output_path_structure(self, adapter):
        req = adapter.build_request(date(2023, 1, 15), "/data")
        # Should be raw/oisst/YYYY/oisst_YYYYMMDD.nc
        assert req.expected_output_path == "raw/oisst/2023/oisst_20230115.nc"

    def test_build_request_output_path_year_2012(self, adapter):
        req = adapter.build_request(date(2012, 3, 5), "/data")
        assert req.expected_output_path == "raw/oisst/2012/oisst_20120305.nc"

    def test_build_request_spatial_region(self, adapter):
        req = adapter.build_request(date(2023, 1, 1), "/data")
        assert "lat_min" in req.spatial_region
        assert req.spatial_region["lat_min"] == -89.875
        assert req.spatial_region["lat_max"] == 89.875

    def test_build_request_no_absolute_path_returned(self, adapter):
        req = adapter.build_request(date(2023, 1, 1), "D:\\data")
        # Path is relative even when data_root is absolute
        assert not req.expected_output_path.startswith("D:")
        assert not req.expected_output_path.startswith("/")

    def test_validate_request_passes(self, adapter):
        req = adapter.build_request(date(2023, 1, 1), "/data")
        issues = adapter.validate_request(req)
        assert issues == []

    def test_date_range_three_days(self, adapter):
        reqs = adapter.date_range_requests(date(2023, 1, 1), date(2023, 1, 3), "/data")
        assert len(reqs) == 3
        assert reqs[0].request_date == date(2023, 1, 1)
        assert reqs[2].request_date == date(2023, 1, 3)

    def test_date_range_single_day(self, adapter):
        reqs = adapter.date_range_requests(date(2023, 6, 15), date(2023, 6, 15), "/data")
        assert len(reqs) == 1

    def test_date_range_wrong_order_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.date_range_requests(date(2023, 1, 5), date(2023, 1, 1), "/data")

    def test_erddap_fallback_url(self, adapter):
        req = adapter.build_request_erddap(date(2023, 1, 15), "/data")
        assert "coastwatch.pfeg.noaa.gov" in req.url_or_query
        assert "ncdcOisst21Agg" in req.url_or_query
        assert req.expected_output_path == "raw/oisst/2023/oisst_20230115.nc"
