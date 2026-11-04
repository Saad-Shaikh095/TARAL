"""
Unit tests for the GLORYSAdapter.

All tests are offline — no CMEMS network access required.
CMEMS credentials are mocked so the adapter can be instantiated without
real credentials in CI or on machines without a .env file.

Verifies:
- Source description matches audit findings
- build_request() produces correct query string and output path
- build_monthly_request() produces correct monthly chunking
- date_range_requests() produces correct number of monthly chunks
- Output paths use pathlib and have no whitespace artifacts
- Credential check raises clearly when credentials are absent
- Scientific guardrail: no spatial reduction or vertical interpolation in requests
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: mock CMEMS credentials so adapter can be instantiated offline
# ---------------------------------------------------------------------------

MOCK_ENV = {
    "COPERNICUSMARINE_SERVICE_USERNAME": "test_user",
    "COPERNICUSMARINE_SERVICE_PASSWORD": "test_pass",
}


@pytest.fixture
def adapter():
    with patch.dict(os.environ, MOCK_ENV):
        from oceanembed.datasets.adapters.glorys import GLORYSAdapter
        return GLORYSAdapter()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestGLORYSAdapterIdentity:
    def test_dataset_id(self, adapter):
        assert adapter.dataset_id == "glorys"

    def test_source_description_dataset_id(self, adapter):
        assert adapter.source_description().dataset_id == "glorys"

    def test_source_description_product_id(self, adapter):
        assert adapter.source_description().product_id == "GLOBAL_MULTIYEAR_PHY_001_030"

    def test_source_description_version(self, adapter):
        assert adapter.source_description().product_version == "202311"

    def test_source_description_auth_required(self, adapter):
        assert adapter.source_description().authentication_required is True

    def test_source_description_temporal_resolution(self, adapter):
        assert adapter.source_description().native_temporal_resolution == "daily"

    def test_source_description_spatial_resolution(self, adapter):
        # ~1/12 degree
        assert abs(adapter.source_description().native_spatial_resolution_degrees - 0.0833) < 0.001

    def test_source_description_variable(self, adapter):
        assert "thetao" in adapter.source_description().variables_requested


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

class TestCredentialCheck:
    def test_raises_without_username(self):
        env = {"COPERNICUSMARINE_SERVICE_PASSWORD": "pass"}
        # Remove username if present
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "COPERNICUSMARINE_SERVICE_USERNAME"}
        clean_env["COPERNICUSMARINE_SERVICE_PASSWORD"] = "pass"
        with patch.dict(os.environ, clean_env, clear=True):
            from importlib import reload
            import oceanembed.datasets.adapters.glorys as mod
            reload(mod)
            with pytest.raises(RuntimeError, match="COPERNICUSMARINE_SERVICE_USERNAME"):
                mod.GLORYSAdapter()

    def test_raises_without_password(self):
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "COPERNICUSMARINE_SERVICE_PASSWORD"}
        clean_env["COPERNICUSMARINE_SERVICE_USERNAME"] = "user"
        with patch.dict(os.environ, clean_env, clear=True):
            from importlib import reload
            import oceanembed.datasets.adapters.glorys as mod
            reload(mod)
            with pytest.raises(RuntimeError, match="COPERNICUSMARINE_SERVICE_PASSWORD"):
                mod.GLORYSAdapter()

    def test_raises_without_both(self):
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("COPERNICUSMARINE_SERVICE_USERNAME",
                                  "COPERNICUSMARINE_SERVICE_PASSWORD")}
        with patch.dict(os.environ, clean_env, clear=True):
            from importlib import reload
            import oceanembed.datasets.adapters.glorys as mod
            reload(mod)
            with pytest.raises(RuntimeError):
                mod.GLORYSAdapter()

    def test_succeeds_with_credentials(self):
        with patch.dict(os.environ, MOCK_ENV):
            from importlib import reload
            import oceanembed.datasets.adapters.glorys as mod
            reload(mod)
            adapter = mod.GLORYSAdapter()  # must not raise
            assert adapter.dataset_id == "glorys"


# ---------------------------------------------------------------------------
# build_request (single day)
# ---------------------------------------------------------------------------

class TestBuildRequest:
    def test_dataset_id_in_request(self, adapter):
        req = adapter.build_request(date(2023, 6, 15), "/data")
        assert req.dataset_id == "glorys"

    def test_request_date_correct(self, adapter):
        d = date(2018, 1, 1)
        req = adapter.build_request(d, "/data")
        assert req.request_date == d

    # ── Core dispatch fix verification ──────────────────────────────────────

    def test_url_starts_with_cmems_sentinel(self, adapter):
        """url_or_query must start with cmems:// so engine routes to run_cmems_subset."""
        req = adapter.build_request(date(2023, 1, 1), "/data")
        assert req.url_or_query.startswith("cmems://"), (
            f"DISPATCH DEFECT: url starts with {req.url_or_query[:40]!r} "
            "instead of 'cmems://'"
        )

    def test_engine_dispatches_to_cmems_not_download(self, adapter):
        """Replicate the engine dispatch logic and confirm it picks run_cmems_subset."""
        req = adapter.build_request(date(2023, 1, 1), "/data")
        url = req.url_or_query
        if url.startswith("cmems://"):
            dispatch = "run_cmems_subset"
        elif url.startswith("oscar://"):
            dispatch = "run_oscar_subset"
        else:
            dispatch = "download_file"
        assert dispatch == "run_cmems_subset", (
            f"Engine would dispatch to {dispatch!r}. url prefix: {url[:50]!r}"
        )

    def test_url_parseable_by_cmems_sentinel_parser(self, adapter):
        """The url must be parseable by parse_cmems_sentinel_url."""
        from oceanembed.datasets.adapters.globcurrent import parse_cmems_sentinel_url
        req = adapter.build_request(date(2023, 1, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        assert params["depth_min"] == 0.0
        assert params["depth_max"] == 1100.0
        assert "thetao" in params["variables"]
        assert params["dataset_version"] == "202311"

    # ── Semantics preserved ─────────────────────────────────────────────────

    def test_url_contains_dataset_id(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert "cmems_mod_glo_phy_my_0.083deg_P1D-m" in req.url_or_query

    def test_url_contains_version(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert "202311" in req.url_or_query

    def test_url_contains_thetao(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert "thetao" in req.url_or_query

    def test_url_contains_date(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert "2020-03-01" in req.url_or_query

    def test_url_contains_depth_min_max(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert "depth_min=0.0" in req.url_or_query
        assert "depth_max=1100.0" in req.url_or_query

    def test_output_path_relative(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert not req.expected_output_path.startswith("/")
        assert "\\" not in req.expected_output_path

    def test_output_path_structure(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert req.expected_output_path.startswith("raw/glorys/2020/")
        assert req.expected_output_path.endswith(".nc")

    def test_output_path_no_whitespace(self, adapter):
        req = adapter.build_request(date(2020, 3, 1), "/data")
        assert " " not in req.expected_output_path

    def test_spatial_region_covers_nio(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert req.spatial_region["lat_min"] <= 5.0
        assert req.spatial_region["lat_max"] >= 30.0
        assert req.spatial_region["lon_min"] <= 45.0
        assert req.spatial_region["lon_max"] >= 105.0

    def test_extra_contains_depth_range(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert req.extra["minimum_depth"] == 0.0
        assert req.extra["maximum_depth"] == 1100.0

    def test_extra_has_no_spatial_reduction(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        query = req.url_or_query.lower()
        assert "rolling" not in query
        assert "regrid" not in query
        assert "smooth" not in query

    def test_extra_has_no_vertical_interpolation(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        assert "interp" not in req.url_or_query.lower()

    def test_validate_request_passes(self, adapter):
        req = adapter.build_request(date(2020, 1, 1), "/data")
        issues = adapter.validate_request(req)
        assert issues == []


# ---------------------------------------------------------------------------
# build_monthly_request
# ---------------------------------------------------------------------------

class TestBuildMonthlyRequest:
    def test_monthly_path_structure(self, adapter):
        req = adapter.build_monthly_request(2018, 1, "/data")
        assert "raw/glorys/2018/" in req.expected_output_path
        assert "20180101" in req.expected_output_path
        assert "20180131" in req.expected_output_path
        assert req.expected_output_path.endswith(".nc")

    def test_monthly_url_starts_with_cmems_sentinel(self, adapter):
        """Monthly request must also use cmems:// sentinel."""
        req = adapter.build_monthly_request(2018, 1, "/data")
        assert req.url_or_query.startswith("cmems://"), (
            f"Monthly request url starts with {req.url_or_query[:40]!r}"
        )

    def test_monthly_url_parseable(self, adapter):
        from oceanembed.datasets.adapters.globcurrent import parse_cmems_sentinel_url
        req = adapter.build_monthly_request(2023, 1, "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        assert params["depth_min"] == 0.0
        assert params["depth_max"] == 1100.0
        assert params["dataset_version"] == "202311"

    def test_monthly_query_contains_start_and_end(self, adapter):
        req = adapter.build_monthly_request(2018, 1, "/data")
        assert "2018-01-01" in req.url_or_query
        assert "2018-01-31" in req.url_or_query

    def test_february_28_days(self, adapter):
        req = adapter.build_monthly_request(2019, 2, "/data")  # non-leap
        assert "2019-02-28" in req.url_or_query
        assert req.extra["expected_time_steps"] == 28

    def test_february_29_days_leap(self, adapter):
        req = adapter.build_monthly_request(2020, 2, "/data")  # leap year
        assert "2020-02-29" in req.url_or_query
        assert req.extra["expected_time_steps"] == 29

    def test_december_31_days(self, adapter):
        req = adapter.build_monthly_request(2020, 12, "/data")
        assert "2020-12-31" in req.url_or_query
        assert req.extra["expected_time_steps"] == 31

    def test_chunking_label(self, adapter):
        req = adapter.build_monthly_request(2020, 6, "/data")
        assert req.extra["chunking"] == "monthly"

    def test_no_whitespace_in_path(self, adapter):
        req = adapter.build_monthly_request(2020, 6, "/data")
        assert " " not in req.expected_output_path


# ---------------------------------------------------------------------------
# date_range_requests (monthly chunking)
# ---------------------------------------------------------------------------

class TestDateRangeRequests:
    def test_single_month(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 1, 1), date(2020, 1, 31), "/data")
        assert len(reqs) == 1

    def test_two_months(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 1, 1), date(2020, 2, 28), "/data")
        assert len(reqs) == 2

    def test_full_year(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 1, 1), date(2020, 12, 31), "/data")
        assert len(reqs) == 12

    def test_13_years_156_months(self, adapter):
        """OceanEmbed period 2012-2024 = 156 monthly chunks."""
        reqs = adapter.date_range_requests(date(2012, 1, 1), date(2024, 12, 31), "/data")
        assert len(reqs) == 156

    def test_partial_start_month(self, adapter):
        """Partial first month still produces one chunk."""
        reqs = adapter.date_range_requests(date(2020, 3, 15), date(2020, 4, 30), "/data")
        assert len(reqs) == 2

    def test_start_after_end_raises(self, adapter):
        with pytest.raises(ValueError):
            adapter.date_range_requests(date(2020, 6, 1), date(2020, 1, 1), "/data")

    def test_single_day_produces_one_chunk(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 6, 15), date(2020, 6, 15), "/data")
        assert len(reqs) == 1

    def test_cross_year_boundary(self, adapter):
        reqs = adapter.date_range_requests(date(2019, 11, 1), date(2020, 2, 28), "/data")
        assert len(reqs) == 4  # Nov, Dec, Jan, Feb

    def test_all_paths_relative(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 1, 1), date(2020, 3, 31), "/data")
        for req in reqs:
            assert not req.expected_output_path.startswith("/")
            assert "\\" not in req.expected_output_path
            assert " " not in req.expected_output_path

    def test_all_requests_have_glorys_id(self, adapter):
        reqs = adapter.date_range_requests(date(2020, 1, 1), date(2020, 3, 31), "/data")
        for req in reqs:
            assert req.dataset_id == "glorys"


# ---------------------------------------------------------------------------
# Path construction — pathlib integration
# ---------------------------------------------------------------------------

class TestPathConstruction:
    def test_output_path_joinable_with_pathlib(self, adapter, tmp_path):
        from pathlib import Path
        req = adapter.build_request(date(2020, 6, 15), str(tmp_path))
        dest = tmp_path / req.expected_output_path
        assert isinstance(dest, Path)
        assert " " not in str(dest)

    def test_monthly_output_path_joinable(self, adapter, tmp_path):
        from pathlib import Path
        req = adapter.build_monthly_request(2020, 6, str(tmp_path))
        dest = tmp_path / req.expected_output_path
        assert isinstance(dest, Path)
        assert dest.suffix == ".nc"


# ---------------------------------------------------------------------------
# Existing CMEMS adapters unaffected
# ---------------------------------------------------------------------------

class TestExistingCMEMSAdaptersUnaffected:
    """Verify GlobCurrent and DUACS still produce cmems:// URLs correctly."""

    def test_globcurrent_still_uses_cmems_sentinel(self):
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        adapter = GlobCurrentAdapter()
        from datetime import date
        req = adapter.build_request(date(2023, 6, 1), "/data")
        assert req.url_or_query.startswith("cmems://")
        assert req.dataset_id == "globcurrent"

    def test_duacs_still_uses_cmems_sentinel(self):
        from oceanembed.datasets.adapters.duacs import DUACSAdapter
        adapter = DUACSAdapter()
        from datetime import date
        req = adapter.build_request(date(2023, 6, 1), "/data")
        assert req.url_or_query.startswith("cmems://")
        assert req.dataset_id == "duacs"

    def test_globcurrent_parser_still_works(self):
        from oceanembed.datasets.adapters.globcurrent import (
            GlobCurrentAdapter,
            parse_cmems_sentinel_url,
        )
        from datetime import date
        adapter = GlobCurrentAdapter()
        req = adapter.build_request(date(2023, 6, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        # GlobCurrent uses single depth=, not depth_min/depth_max
        assert params["depth"] == 0.0
        assert params["depth_min"] is None
        assert params["depth_max"] is None
        # No dataset_version for GlobCurrent
        assert params["dataset_version"] is None

    def test_duacs_parser_depth_is_none(self):
        from oceanembed.datasets.adapters.duacs import DUACSAdapter
        from oceanembed.datasets.adapters.globcurrent import parse_cmems_sentinel_url
        from datetime import date
        adapter = DUACSAdapter()
        req = adapter.build_request(date(2023, 6, 1), "/data")
        params = parse_cmems_sentinel_url(req.url_or_query)
        # DUACS has no depth dimension
        assert params["depth"] is None
        assert params["depth_min"] is None
        assert params["depth_max"] is None
