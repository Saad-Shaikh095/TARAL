"""
Unit tests for the OceanEmbed dataset adapter contract.

Tests cover:
- DatasetAdapterBase cannot be instantiated directly (abstract).
- A concrete test adapter that satisfies the interface works correctly.
- AcquisitionRequest rejects hard-coded absolute paths.
- SourceDescription is correctly structured.
- validate_request() catches basic structural issues.
- date_range_requests() generates the correct number of requests.
- Adapters that violate the interface contract are detected.

No real datasets or network access are used.
"""

from __future__ import annotations

import abc
from datetime import date

import pytest

from oceanembed.datasets.adapters import (
    AcquisitionRequest,
    DatasetAdapterBase,
    SourceDescription,
)


# ---------------------------------------------------------------------------
# Minimal concrete test adapter
# ---------------------------------------------------------------------------

class _MinimalAdapter(DatasetAdapterBase):
    """Minimal concrete adapter for interface testing."""

    @property
    def dataset_id(self) -> str:
        return "test_dataset"

    def source_description(self) -> SourceDescription:
        return SourceDescription(
            dataset_id="test_dataset",
            product_id="TEST_PRODUCT_v1",
            product_version="1.0",
            provider_name="Test Provider",
            native_temporal_resolution="daily",
            native_spatial_resolution_degrees=0.25,
            variables_requested=["var1", "var2"],
            authentication_required=False,
            notes="Synthetic test adapter.",
        )

    def build_request(self, request_date: date, data_root: str) -> AcquisitionRequest:
        date_str = request_date.strftime("%Y%m%d")
        return AcquisitionRequest(
            dataset_id=self.dataset_id,
            request_date=request_date,
            url_or_query=f"https://example.com/data/{date_str}.nc",
            expected_output_path=f"raw/test_dataset/{date_str}.nc",
            source_description=self.source_description(),
            spatial_region={"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 105.0},
        )


# ---------------------------------------------------------------------------
# Abstract class enforcement
# ---------------------------------------------------------------------------

class TestAbstractBase:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            DatasetAdapterBase()  # type: ignore[abstract]

    def test_missing_dataset_id_property_prevents_instantiation(self):
        class MissingDatasetId(DatasetAdapterBase):
            def source_description(self) -> SourceDescription:
                return _MinimalAdapter().source_description()
            def build_request(self, request_date, data_root):
                return _MinimalAdapter().build_request(request_date, data_root)

        with pytest.raises(TypeError):
            MissingDatasetId()

    def test_missing_source_description_prevents_instantiation(self):
        class MissingSourceDesc(DatasetAdapterBase):
            @property
            def dataset_id(self): return "x"
            def build_request(self, request_date, data_root):
                return _MinimalAdapter().build_request(request_date, data_root)

        with pytest.raises(TypeError):
            MissingSourceDesc()

    def test_missing_build_request_prevents_instantiation(self):
        class MissingBuildRequest(DatasetAdapterBase):
            @property
            def dataset_id(self): return "x"
            def source_description(self): return _MinimalAdapter().source_description()

        with pytest.raises(TypeError):
            MissingBuildRequest()


# ---------------------------------------------------------------------------
# Concrete adapter behaviour
# ---------------------------------------------------------------------------

class TestConcreteAdapter:
    @pytest.fixture
    def adapter(self) -> _MinimalAdapter:
        return _MinimalAdapter()

    def test_dataset_id_correct(self, adapter: _MinimalAdapter):
        assert adapter.dataset_id == "test_dataset"

    def test_source_description_returns_correct_type(self, adapter: _MinimalAdapter):
        desc = adapter.source_description()
        assert isinstance(desc, SourceDescription)

    def test_source_description_fields(self, adapter: _MinimalAdapter):
        desc = adapter.source_description()
        assert desc.dataset_id == "test_dataset"
        assert desc.native_temporal_resolution == "daily"
        assert desc.native_spatial_resolution_degrees == 0.25
        assert "var1" in desc.variables_requested
        assert desc.authentication_required is False

    def test_build_request_returns_correct_type(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 15), "/data")
        assert isinstance(req, AcquisitionRequest)

    def test_build_request_date_correct(self, adapter: _MinimalAdapter):
        d = date(2020, 6, 1)
        req = adapter.build_request(d, "/data")
        assert req.request_date == d

    def test_build_request_dataset_id_correct(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert req.dataset_id == "test_dataset"

    def test_build_request_url_not_empty(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert req.url_or_query

    def test_build_request_output_path_relative(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert not req.expected_output_path.startswith("/")
        assert ":\\" not in req.expected_output_path

    def test_repr_contains_class_and_id(self, adapter: _MinimalAdapter):
        r = repr(adapter)
        assert "_MinimalAdapter" in r
        assert "test_dataset" in r


# ---------------------------------------------------------------------------
# AcquisitionRequest validation
# ---------------------------------------------------------------------------

class TestAcquisitionRequest:
    def _valid_request(self, **overrides) -> dict:
        base = dict(
            dataset_id="testds",
            request_date=date(2020, 1, 1),
            url_or_query="https://example.com/file.nc",
            expected_output_path="raw/testds/20200101.nc",
            source_description=_MinimalAdapter().source_description(),
        )
        base.update(overrides)
        return base

    def test_valid_request_created(self):
        req = AcquisitionRequest(**self._valid_request())
        assert req.dataset_id == "testds"

    def test_windows_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            AcquisitionRequest(**self._valid_request(
                expected_output_path=r"D:\OceanEmbed\data\raw\testds\file.nc"
            ))

    def test_unix_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            AcquisitionRequest(**self._valid_request(
                expected_output_path="/home/user/data/raw/testds/file.nc"
            ))

    def test_relative_path_accepted(self):
        req = AcquisitionRequest(**self._valid_request(
            expected_output_path="raw/testds/20200101.nc"
        ))
        assert req.expected_output_path == "raw/testds/20200101.nc"

    def test_spatial_region_stored(self):
        region = {"lat_min": 5.0, "lat_max": 30.0, "lon_min": 45.0, "lon_max": 105.0}
        req = AcquisitionRequest(**self._valid_request(spatial_region=region))
        assert req.spatial_region["lat_min"] == 5.0

    def test_extra_metadata_stored(self):
        req = AcquisitionRequest(**self._valid_request(
            extra={"depth_level": 15, "note": "mixed layer"}
        ))
        assert req.extra["depth_level"] == 15


# ---------------------------------------------------------------------------
# validate_request (default implementation)
# ---------------------------------------------------------------------------

class TestValidateRequest:
    @pytest.fixture
    def adapter(self) -> _MinimalAdapter:
        return _MinimalAdapter()

    def test_valid_request_passes(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        issues = adapter.validate_request(req)
        assert issues == []

    def test_wrong_dataset_id_flagged(self, adapter: _MinimalAdapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        # Manually override dataset_id to create a mismatch
        bad_req = AcquisitionRequest(
            dataset_id="wrong_id",
            request_date=req.request_date,
            url_or_query=req.url_or_query,
            expected_output_path=req.expected_output_path,
            source_description=req.source_description,
        )
        issues = adapter.validate_request(bad_req)
        assert len(issues) > 0
        assert any("wrong_id" in msg for msg in issues)

    def test_empty_url_flagged(self, adapter: _MinimalAdapter):
        req = AcquisitionRequest(
            dataset_id=adapter.dataset_id,
            request_date=date(2020, 1, 1),
            url_or_query="",
            expected_output_path="raw/test/file.nc",
            source_description=adapter.source_description(),
        )
        issues = adapter.validate_request(req)
        assert any("url" in msg.lower() for msg in issues)

    def test_empty_output_path_flagged(self, adapter: _MinimalAdapter):
        req = AcquisitionRequest(
            dataset_id=adapter.dataset_id,
            request_date=date(2020, 1, 1),
            url_or_query="https://example.com/file.nc",
            expected_output_path="",
            source_description=adapter.source_description(),
        )
        issues = adapter.validate_request(req)
        assert any("path" in msg.lower() or "output" in msg.lower() for msg in issues)


# ---------------------------------------------------------------------------
# date_range_requests (default implementation)
# ---------------------------------------------------------------------------

class TestDateRangeRequests:
    @pytest.fixture
    def adapter(self) -> _MinimalAdapter:
        return _MinimalAdapter()

    def test_single_day_range(self, adapter: _MinimalAdapter):
        requests = adapter.date_range_requests(date(2020, 1, 1), date(2020, 1, 1), "/data")
        assert len(requests) == 1
        assert requests[0].request_date == date(2020, 1, 1)

    def test_five_day_range(self, adapter: _MinimalAdapter):
        requests = adapter.date_range_requests(date(2020, 1, 1), date(2020, 1, 5), "/data")
        assert len(requests) == 5
        dates = [r.request_date for r in requests]
        assert dates == [date(2020, 1, d) for d in range(1, 6)]

    def test_start_after_end_raises(self, adapter: _MinimalAdapter):
        with pytest.raises(ValueError, match="start_date"):
            adapter.date_range_requests(date(2020, 1, 5), date(2020, 1, 1), "/data")

    def test_all_requests_have_correct_dataset_id(self, adapter: _MinimalAdapter):
        requests = adapter.date_range_requests(date(2020, 3, 1), date(2020, 3, 3), "/data")
        for req in requests:
            assert req.dataset_id == "test_dataset"
