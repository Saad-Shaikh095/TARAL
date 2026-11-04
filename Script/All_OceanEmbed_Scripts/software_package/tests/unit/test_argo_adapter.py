"""
Unit tests for the OceanEmbed ArgoAdapter.

All tests are offline — no network access, no real GDAC downloads.
A synthetic index fixture is used for index-related tests.

Tests cover:
- ArgoIndexRecord fields and properties
- read_nio_index: filtering, deduplication, malformed rows
- ArgoAdapter: identity, source_description, request construction
- AcquisitionRequest: correct URL, relative path, no scientific transforms
- date_range_requests: correct count for a date range
- build_requests_for_date: all profiles on one day
- Safety: no interpolation, no credentials in URLs
- Idempotency: same record → same request output
"""

from __future__ import annotations

import csv
import io
import textwrap
from datetime import date
from pathlib import Path

import pytest

from oceanembed.datasets.adapters.argo import (
    ArgoAdapter,
    ArgoIndexRecord,
    read_nio_index,
    _S3_BASE,
    _CORIOLIS_BASE,
    _USGDAC_BASE,
    REQUIRED_VARIABLES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INDEX_HEADER = (
    "# Title : Profile directory file of the Argo GDAC\n"
    "# Description : test fixture\n"
    "file,date,latitude,longitude,ocean,profiler_type,institution,date_update\n"
)

def _make_index_txt(rows: list[dict]) -> str:
    """Build a minimal ar_index_global_prof.txt string from row dicts."""
    lines = [_INDEX_HEADER]
    for r in rows:
        lines.append(
            f"{r['file']},{r['date']},{r['lat']},{r['lon']},"
            f"{r.get('ocean','I')},{r.get('profiler_type','845')},"
            f"{r.get('institution','AO')},{r.get('date_update','20260101000000')}\n"
        )
    return "".join(lines)


def _write_index(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "ar_index_global_prof.txt"
    p.write_text(_make_index_txt(rows), encoding="utf-8")
    return p


# Sample NIO profile rows (inside domain, inside period)
_NIO_ROWS = [
    {"file": "aoml/1900272/profiles/D1900272_314.nc",
     "date": "20131127120000", "lat": "5.050", "lon": "49.025"},
    {"file": "aoml/1900272/profiles/D1900272_315.nc",
     "date": "20131207120000", "lat": "5.171", "lon": "48.972"},
    {"file": "incois/1902345/profiles/R1902345_010.nc",
     "date": "20150615060000", "lat": "15.200", "lon": "72.100"},
]

# Out-of-domain or out-of-period rows (should be filtered out)
_OUTSIDE_ROWS = [
    # latitude out of range
    {"file": "aoml/9999999/profiles/D9999999_001.nc",
     "date": "20150101120000", "lat": "40.0", "lon": "70.0"},
    # before 2012
    {"file": "aoml/8888888/profiles/D8888888_001.nc",
     "date": "20100101120000", "lat": "10.0", "lon": "65.0"},
    # longitude out of range
    {"file": "aoml/7777777/profiles/D7777777_001.nc",
     "date": "20150101120000", "lat": "10.0", "lon": "30.0"},
    # after 2024
    {"file": "aoml/6666666/profiles/D6666666_001.nc",
     "date": "20250101120000", "lat": "10.0", "lon": "65.0"},
]


# ---------------------------------------------------------------------------
# ArgoIndexRecord
# ---------------------------------------------------------------------------

class TestArgoIndexRecord:
    def _make_record(self, **kwargs) -> ArgoIndexRecord:
        defaults = dict(
            file="aoml/1900272/profiles/D1900272_314.nc",
            date_raw="20131127120000",
            latitude=5.050,
            longitude=49.025,
            ocean="I",
            profiler_type="845",
            institution="AO",
            date_update="20260101000000",
        )
        defaults.update(kwargs)
        return ArgoIndexRecord(**defaults)

    def test_profile_date(self):
        r = self._make_record(date_raw="20131127120000")
        assert r.profile_date == date(2013, 11, 27)

    def test_float_id(self):
        r = self._make_record()
        assert r.float_id == "1900272"

    def test_dac(self):
        r = self._make_record()
        assert r.dac == "aoml"

    def test_profile_key_normalised(self):
        r = self._make_record()
        assert r.profile_key == "aoml/1900272/profiles/D1900272_314.nc"

    def test_s3_url(self):
        r = self._make_record()
        assert r.s3_url() == _S3_BASE + "aoml/1900272/profiles/D1900272_314.nc"
        assert r.s3_url().startswith("https://")

    def test_coriolis_url(self):
        r = self._make_record()
        assert "data-argo.ifremer.fr" in r.coriolis_url()

    def test_usgdac_url(self):
        r = self._make_record()
        assert "nrlgodae1" in r.usgdac_url()

    def test_local_relative_path_structure(self):
        r = self._make_record(date_raw="20131127120000")
        path = r.local_relative_path()
        assert path.startswith("raw/argo/2013_11/aoml/1900272/profiles/")
        assert path.endswith(".nc")
        assert " " not in path
        assert "\\" not in path

    def test_no_absolute_path(self):
        r = self._make_record()
        path = r.local_relative_path()
        assert not path.startswith("/")
        assert ":" not in path[:3]


# ---------------------------------------------------------------------------
# read_nio_index
# ---------------------------------------------------------------------------

class TestReadNioIndex:
    def test_loads_nio_profiles(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_filters_out_of_domain(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS + _OUTSIDE_ROWS)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_deduplicate_same_file(self, tmp_path):
        dup = _NIO_ROWS + [_NIO_ROWS[0]]  # exact duplicate
        idx = _write_index(tmp_path, dup)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_sorted_by_date_then_key(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        records = read_nio_index(idx)
        dates = [r.profile_date for r in records]
        assert dates == sorted(dates)

    def test_malformed_lat_skipped(self, tmp_path):
        bad = [{"file": "aoml/1/profiles/D1_001.nc",
                "date": "20150101120000",
                "lat": "not_a_number", "lon": "65.0"}]
        idx = _write_index(tmp_path, _NIO_ROWS + bad)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_malformed_date_skipped(self, tmp_path):
        bad = [{"file": "aoml/2/profiles/D2_001.nc",
                "date": "invalid_date", "lat": "10.0", "lon": "65.0"}]
        idx = _write_index(tmp_path, _NIO_ROWS + bad)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_bad_path_structure_skipped(self, tmp_path):
        """Path must be at least 4 parts deep."""
        bad = [{"file": "aoml/tooshort.nc",
                "date": "20150101120000", "lat": "10.0", "lon": "65.0"}]
        idx = _write_index(tmp_path, _NIO_ROWS + bad)
        records = read_nio_index(idx)
        assert len(records) == 3

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_nio_index(tmp_path / "does_not_exist.txt")

    def test_empty_index_returns_empty(self, tmp_path):
        p = tmp_path / "ar_index_global_prof.txt"
        p.write_text(_INDEX_HEADER, encoding="utf-8")
        records = read_nio_index(p)
        assert records == []


# ---------------------------------------------------------------------------
# ArgoAdapter identity
# ---------------------------------------------------------------------------

class TestArgoAdapterIdentity:
    @pytest.fixture
    def adapter(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        return ArgoAdapter(index_path=idx)

    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "argo"

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "argo"

    def test_source_description_no_auth(self, adapter):
        assert adapter.source_description().authentication_required is False

    def test_source_description_has_pres_temp_psal(self, adapter):
        desc = adapter.source_description()
        for v in ("PRES", "TEMP", "PSAL"):
            assert v in desc.variables_requested

    def test_source_description_resolution_is_none(self, adapter):
        """Argo is point data — spatial resolution is None (not gridded)."""
        assert adapter.source_description().native_spatial_resolution_degrees is None

    def test_total_nio_profiles(self, adapter):
        assert adapter.total_nio_profiles() == 3


# ---------------------------------------------------------------------------
# build_request (single day)
# ---------------------------------------------------------------------------

class TestBuildRequest:
    @pytest.fixture
    def adapter(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        return ArgoAdapter(index_path=idx)

    def test_known_date_returns_request(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        assert req.dataset_id == "argo"
        assert req.request_date == date(2013, 11, 27)

    def test_url_starts_with_s3(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        assert req.url_or_query.startswith("https://argo-gdac-sandbox.s3.")

    def test_url_contains_profile_key(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        assert "D1900272_314.nc" in req.url_or_query

    def test_output_path_relative(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        assert not req.expected_output_path.startswith("/")
        assert "\\" not in req.expected_output_path

    def test_output_path_structure(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        assert req.expected_output_path.startswith("raw/argo/2013_11/aoml/")
        assert req.expected_output_path.endswith(".nc")

    def test_no_interpolation_in_url(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        url_lower = req.url_or_query.lower()
        assert "interp" not in url_lower
        assert "resample" not in url_lower
        assert "regrid" not in url_lower
        assert "depth_target" not in url_lower

    def test_fallback_urls_in_extra(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        fallbacks = req.extra.get("fallback_urls", [])
        assert len(fallbacks) == 2
        assert any("ifremer.fr" in u for u in fallbacks)
        assert any("nrlgodae1" in u for u in fallbacks)

    def test_no_credentials_in_url(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        # Ensure no credentials appear in URL
        url = req.url_or_query
        assert "@" not in url or url.count("@") == 0
        assert "password" not in url.lower()
        assert "username" not in url.lower()

    def test_scientific_note_in_extra(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        note = req.extra.get("scientific_note", "")
        assert "OSD-010" in note
        assert "interpolation" in note.lower()

    def test_date_with_no_profile_returns_placeholder(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        # No profile on 2020-01-01 in fixture — should be a placeholder
        assert req.url_or_query == "" or "no_profile" in req.expected_output_path

    def test_validate_request_passes_for_real_profile(self, adapter):
        req = adapter.build_request(date(2013, 11, 27), "/data")
        if req.url_or_query:
            issues = adapter.validate_request(req)
            assert issues == []


# ---------------------------------------------------------------------------
# build_requests_for_date
# ---------------------------------------------------------------------------

class TestBuildRequestsForDate:
    @pytest.fixture
    def adapter(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        return ArgoAdapter(index_path=idx)

    def test_known_date_returns_all_profiles(self, adapter):
        # 2013-11-27 has exactly 1 profile in fixture
        reqs = adapter.build_requests_for_date(date(2013, 11, 27), "/data")
        assert len(reqs) == 1

    def test_empty_date_returns_empty_list(self, adapter):
        reqs = adapter.build_requests_for_date(date(2020, 6, 1), "/data")
        assert reqs == []

    def test_all_requests_have_argo_id(self, adapter):
        reqs = adapter.build_requests_for_date(date(2013, 11, 27), "/data")
        for req in reqs:
            assert req.dataset_id == "argo"


# ---------------------------------------------------------------------------
# date_range_requests
# ---------------------------------------------------------------------------

class TestDateRangeRequests:
    @pytest.fixture
    def adapter(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        return ArgoAdapter(index_path=idx)

    def test_full_range_returns_all_three(self, adapter):
        reqs = adapter.date_range_requests(date(2012,1,1), date(2024,12,31), "/data")
        assert len(reqs) == 3

    def test_narrow_range_filters_correctly(self, adapter):
        # Only 2013-11-27 and 2013-12-07 are in Nov-Dec 2013
        reqs = adapter.date_range_requests(date(2013,11,1), date(2013,12,31), "/data")
        assert len(reqs) == 2

    def test_range_with_no_profiles_returns_empty(self, adapter):
        reqs = adapter.date_range_requests(date(2020,1,1), date(2020,12,31), "/data")
        assert reqs == []

    def test_start_after_end_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.date_range_requests(date(2020,6,1), date(2020,1,1), "/data")

    def test_all_output_paths_relative(self, adapter):
        reqs = adapter.date_range_requests(date(2012,1,1), date(2024,12,31), "/data")
        for req in reqs:
            assert not req.expected_output_path.startswith("/")
            assert "\\" not in req.expected_output_path

    def test_no_duplicate_paths(self, adapter):
        reqs = adapter.date_range_requests(date(2012,1,1), date(2024,12,31), "/data")
        paths = [req.expected_output_path for req in reqs if req.url_or_query]
        assert len(paths) == len(set(paths))


# ---------------------------------------------------------------------------
# Profile count helper
# ---------------------------------------------------------------------------

class TestProfileCount:
    def test_profile_count_in_range(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        adapter = ArgoAdapter(index_path=idx)
        count = adapter.profile_count_in_range(date(2012,1,1), date(2024,12,31))
        assert count == 3

    def test_total_nio_profiles(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        adapter = ArgoAdapter(index_path=idx)
        assert adapter.total_nio_profiles() == 3


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_request_twice_produces_same_output(self, tmp_path):
        idx = _write_index(tmp_path, _NIO_ROWS)
        adapter = ArgoAdapter(index_path=idx)
        req1 = adapter.build_request(date(2013, 11, 27), "/data")
        req2 = adapter.build_request(date(2013, 11, 27), "/data")
        assert req1.url_or_query == req2.url_or_query
        assert req1.expected_output_path == req2.expected_output_path


# ---------------------------------------------------------------------------
# Required variables constant
# ---------------------------------------------------------------------------

class TestRequiredVariables:
    def test_pres_required(self):
        assert "PRES" in REQUIRED_VARIABLES

    def test_temp_required(self):
        assert "TEMP" in REQUIRED_VARIABLES
