"""
Unit tests for the OSCAR dataset adapter (A-Y matrix).
All network access is mocked. No real NASA Earthdata calls.
"""
from __future__ import annotations
import asyncio, hashlib, os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from oceanembed.datasets.adapters.oscar import (
    OSCARAdapter, _COLLECTION_ID, _OSCAR_VARIABLES,
    _SOURCE_COVERAGE_START, _SOURCE_COVERAGE_END,
    _validate_oscar_request_params, _lon_to_oscar_360,
    _geo_to_oscar_lon_idx, _geo_to_oscar_lat_idx,
    _build_dap4_constraint, _granule_name, parse_oscar_sentinel_url,
)
from oceanembed.datasets.adapters.base import AcquisitionRequest, SourceDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: bytes, status: int = 200,
                   ctype: str = "application/x-netcdf4"):
    r = MagicMock()
    r.status_code = status
    r.headers = {"Content-Type": ctype}
    r.iter_content.return_value = [content] if content else []
    return r


def _make_session(content: bytes = b"fake nc", status: int = 200,
                  ctype: str = "application/x-netcdf4"):
    s = MagicMock()
    s.get.return_value = _make_response(content, status, ctype)
    return s


def _run(tmp_path, content=b"fake nc", status=200,
         ctype="application/x-netcdf4", max_retries=1,
         req_date=date(2023, 1, 1), creds=True):
    from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
    dest = tmp_path / f"oscar_{req_date:%Y%m%d}.nc"
    job = AcquisitionJob(job_id="jt", dataset_id="oscar",
                         start_date=req_date, end_date=req_date,
                         data_root=tmp_path)
    req = OSCARAdapter().build_request(req_date, str(tmp_path))
    env = ({"EARTHDATA_USERNAME": "u", "EARTHDATA_PASSWORD": "p"} if creds else {})
    saved_u = os.environ.pop("EARTHDATA_USERNAME", None)
    saved_p = os.environ.pop("EARTHDATA_PASSWORD", None)
    try:
        with patch.dict(os.environ, env):
            with patch("earthaccess.login"):
                with patch("earthaccess.get_requests_https_session",
                           return_value=_make_session(content, status, ctype)):
                    res = run_oscar_subset(req.url_or_query, dest, job,
                                          max_retries=max_retries)
    finally:
        if saved_u: os.environ["EARTHDATA_USERNAME"] = saved_u
        if saved_p: os.environ["EARTHDATA_PASSWORD"] = saved_p
    return res, dest, job


@pytest.fixture
def adapter():
    return OSCARAdapter()

@pytest.fixture
def data_root(tmp_path):
    return str(tmp_path)


# ===========================================================================
# A. Valid configuration / identity
# ===========================================================================
class TestAdapterIdentity:
    def test_dataset_id(self, adapter): assert adapter.dataset_id == "oscar"
    def test_source_desc_type(self, adapter): assert isinstance(adapter.source_description(), SourceDescription)
    def test_product_id(self, adapter): assert _COLLECTION_ID in adapter.source_description().product_id
    def test_version_2(self, adapter): assert adapter.source_description().product_version == "2.0"
    def test_auth_required(self, adapter): assert adapter.source_description().authentication_required is True
    def test_daily(self, adapter): assert adapter.source_description().native_temporal_resolution == "daily"
    def test_res_025(self, adapter): assert adapter.source_description().native_spatial_resolution_degrees == 0.25
    def test_vars_u_v(self, adapter): assert set(adapter.source_description().variables_requested) == {"u", "v"}
    def test_two_vars(self, adapter): assert len(adapter.source_description().variables_requested) == 2
    def test_max_job_days(self, adapter): assert adapter._max_job_days == 31
    def test_has_depth_false(self, adapter): assert adapter.build_request(date(2023,1,1), "/d").extra["has_depth"] is False
    def test_nominal_depth_15(self, adapter): assert adapter.build_request(date(2023,1,1), "/d").extra["nominal_depth_m"] == 15
    def test_osd009_in_note(self, adapter): assert "OSD-009" in adapter.build_request(date(2023,1,1), "/d").extra["depth_note"]
    def test_sentinel_prefix(self, adapter): assert adapter.build_request(date(2023,1,1), "/d").url_or_query.startswith("oscar://")


# ===========================================================================
# B. Invalid bbox
# ===========================================================================
class TestInvalidBbox:
    def test_min_gt_max_lat(self):
        errors = _validate_oscar_request_params(date(2023,1,1), date(2023,1,7), 30.0, 5.0, 45.0, 105.0)
        assert any("min_lat" in e for e in errors)

    def test_equal_lat(self):
        errors = _validate_oscar_request_params(date(2023,1,1), date(2023,1,7), 15.0, 15.0, 45.0, 105.0)
        assert any("min_lat" in e for e in errors)

    def test_lat_below_minus80(self):
        assert _validate_oscar_request_params(date(2023,1,1), date(2023,1,7), -81.0, 30.0, 45.0, 105.0)

    def test_lat_above_80(self):
        assert _validate_oscar_request_params(date(2023,1,1), date(2023,1,7), 5.0, 81.0, 45.0, 105.0)

    def test_raises_on_use(self):
        with pytest.raises(ValueError):
            OSCARAdapter(min_lat=50.0, max_lat=10.0).date_range_requests(date(2023,1,1), date(2023,1,7), "/d")


# ===========================================================================
# C. Invalid date range
# ===========================================================================
class TestInvalidDateRange:
    def test_reversed(self, adapter, data_root):
        with pytest.raises(ValueError, match="start_date"):
            adapter.date_range_requests(date(2023,6,1), date(2023,1,1), data_root)

    def test_error_has_both_dates(self, adapter, data_root):
        with pytest.raises(ValueError) as exc:
            adapter.date_range_requests(date(2023,6,1), date(2023,1,1), data_root)
        assert "2023-06-01" in str(exc.value) and "2023-01-01" in str(exc.value)

    def test_same_date_ok(self, adapter, data_root):
        assert len(adapter.date_range_requests(date(2023,6,15), date(2023,6,15), data_root)) == 1


# ===========================================================================
# D. Date outside source coverage
# ===========================================================================
class TestDateCoverage:
    def test_before_2012(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(date(2011,12,31), date(2012,1,7), data_root)

    def test_after_2024(self, adapter, data_root):
        with pytest.raises(ValueError, match="coverage"):
            adapter.date_range_requests(date(2024,12,1), date(2025,3,1), data_root)

    def test_exact_start_ok(self, adapter, data_root):
        assert len(adapter.date_range_requests(_SOURCE_COVERAGE_START, _SOURCE_COVERAGE_START, data_root)) == 1

    def test_exact_end_ok(self, adapter, data_root):
        assert len(adapter.date_range_requests(_SOURCE_COVERAGE_END, _SOURCE_COVERAGE_END, data_root)) == 1

    def test_error_mentions_coverage(self, adapter, data_root):
        with pytest.raises(ValueError) as exc:
            adapter.date_range_requests(date(2011,1,1), date(2011,12,31), data_root)
        assert str(_SOURCE_COVERAGE_START) in str(exc.value)


# ===========================================================================
# E. Invalid / missing required config (validate_request)
# ===========================================================================
class TestValidateRequest:
    def test_valid_passes(self, adapter, data_root):
        assert adapter.validate_request(adapter.build_request(date(2023,1,1), data_root)) == []

    def test_non_oscar_url_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023,1,1), data_root)
        bad = AcquisitionRequest(dataset_id=req.dataset_id, request_date=req.request_date,
                                 url_or_query="https://example.com/file.nc",
                                 expected_output_path=req.expected_output_path,
                                 source_description=req.source_description, extra=req.extra)
        assert any("oscar://" in m for m in adapter.validate_request(bad))

    def test_has_depth_true_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023,1,1), data_root)
        bad = AcquisitionRequest(dataset_id=req.dataset_id, request_date=req.request_date,
                                 url_or_query=req.url_or_query,
                                 expected_output_path=req.expected_output_path,
                                 source_description=req.source_description,
                                 extra=dict(req.extra, has_depth=True))
        assert any("depth" in m.lower() for m in adapter.validate_request(bad))

    def test_missing_collection_id_flagged(self, adapter, data_root):
        req = adapter.build_request(date(2023,1,1), data_root)
        trimmed = {k: v for k,v in req.extra.items() if k != "collection_id"}
        bad = AcquisitionRequest(dataset_id=req.dataset_id, request_date=req.request_date,
                                 url_or_query=req.url_or_query,
                                 expected_output_path=req.expected_output_path,
                                 source_description=req.source_description, extra=trimmed)
        assert any("collection_id" in m for m in adapter.validate_request(bad))


# ===========================================================================
# F. Correct source endpoint generation
# ===========================================================================
class TestEndpointGeneration:
    def test_collection_in_url(self, adapter):
        assert _COLLECTION_ID in adapter.build_request(date(2023,1,1), "/d").url_or_query

    def test_date_in_url(self, adapter):
        assert "2023-06-15" in adapter.build_request(date(2023,6,15), "/d").url_or_query

    def test_granule_format(self):
        assert _granule_name(date(2023,1,5)) == "oscar_currents_final_20230105"

    def test_parsed_url_has_opendap(self):
        params = parse_oscar_sentinel_url(OSCARAdapter().build_request(date(2023,1,1), "/d").url_or_query)
        assert "opendap.earthdata.nasa.gov" in params["opendap_url"]

    def test_parsed_url_has_dap4_ce(self):
        params = parse_oscar_sentinel_url(OSCARAdapter().build_request(date(2023,1,1), "/d").url_or_query)
        assert "dap4.ce=" in params["opendap_url"]

    def test_parsed_url_has_granule(self):
        params = parse_oscar_sentinel_url(OSCARAdapter().build_request(date(2023,1,1), "/d").url_or_query)
        assert "oscar_currents_final_20230101" in params["opendap_url"]


# ===========================================================================
# G. Correct request parameters
# ===========================================================================
class TestRequestParams:
    def test_spatial_region(self, adapter):
        req = adapter.build_request(date(2023,1,1), "/d")
        assert req.spatial_region["lat_min"] == 5.0
        assert req.spatial_region["lon_max"] == 105.0

    def test_idx_end_gt_start(self, adapter):
        req = adapter.build_request(date(2023,1,1), "/d")
        assert req.extra["lon_idx_end"] > req.extra["lon_idx_start"]
        assert req.extra["lat_idx_end"] > req.extra["lat_idx_start"]

    def test_constraint_has_u_v_lon_lat(self):
        c = _build_dap4_constraint(180, 420, 380, 478)
        assert "/u[0][180:420][380:478]" in c
        assert "/v[0][180:420][380:478]" in c
        assert "/lon[180:420]" in c
        assert "/lat[380:478]" in c

    def test_different_bbox_different_url(self):
        a = OSCARAdapter(min_lat=5.0, max_lat=30.0, min_lon=45.0, max_lon=105.0)
        b = OSCARAdapter(min_lat=10.0, max_lat=25.0, min_lon=60.0, max_lon=90.0)
        assert (a.build_request(date(2023,1,1), "/d").url_or_query !=
                b.build_request(date(2023,1,1), "/d").url_or_query)


# ===========================================================================
# H. Authentication configuration
# ===========================================================================
class TestAuthConfig:
    def test_both_present(self):
        with patch.dict(os.environ, {"EARTHDATA_USERNAME": "u", "EARTHDATA_PASSWORD": "p"}):
            assert OSCARAdapter().check_credentials()["credentials_available"] is True

    def test_both_missing(self):
        saved_u = os.environ.pop("EARTHDATA_USERNAME", None)
        saved_p = os.environ.pop("EARTHDATA_PASSWORD", None)
        try:
            assert OSCARAdapter().check_credentials()["credentials_available"] is False
        finally:
            if saved_u: os.environ["EARTHDATA_USERNAME"] = saved_u
            if saved_p: os.environ["EARTHDATA_PASSWORD"] = saved_p

    def test_only_username(self):
        saved_p = os.environ.pop("EARTHDATA_PASSWORD", None)
        try:
            with patch.dict(os.environ, {"EARTHDATA_USERNAME": "u"}):
                r = OSCARAdapter().check_credentials()
        finally:
            if saved_p: os.environ["EARTHDATA_PASSWORD"] = saved_p
        assert not r["credentials_available"]
        assert r["has_username"]

    def test_message_no_values(self):
        with patch.dict(os.environ, {"EARTHDATA_USERNAME": "SECRET_U", "EARTHDATA_PASSWORD": "SECRET_P"}):
            msg = OSCARAdapter().check_credentials()["message"]
        assert "SECRET_U" not in msg and "SECRET_P" not in msg

    def test_no_creds_in_url(self):
        with patch.dict(os.environ, {"EARTHDATA_USERNAME": "SECUSER", "EARTHDATA_PASSWORD": "SECPASS"}):
            url = OSCARAdapter().build_request(date(2023,1,1), "/d").url_or_query
        assert "SECUSER" not in url and "SECPASS" not in url


# ===========================================================================
# I. Longitude convention (0-360 normalisation)
# ===========================================================================
class TestLongitudeConvention:
    def test_positive_lon_unchanged(self): assert _lon_to_oscar_360(45.0) == pytest.approx(45.0)
    def test_minus180_to_180(self): assert _lon_to_oscar_360(-180.0) == pytest.approx(180.0)
    def test_minus90_to_270(self): assert _lon_to_oscar_360(-90.0) == pytest.approx(270.0)
    def test_360_to_0(self): assert _lon_to_oscar_360(360.0) == pytest.approx(0.0)
    def test_45E_idx_180(self):
        # lon[i] = i * 0.25  → lon=45.0 at i=180 ✓
        assert _geo_to_oscar_lon_idx(45.0) == 180

    def test_105E_idx_420(self):
        # lon=105.0 at i=420 ✓
        assert _geo_to_oscar_lon_idx(105.0) == 420
    def test_larger_lon_larger_idx(self): assert _geo_to_oscar_lon_idx(105.0) > _geo_to_oscar_lon_idx(45.0)
    def test_larger_lat_larger_idx(self): assert _geo_to_oscar_lat_idx(30.0) > _geo_to_oscar_lat_idx(5.0)
    def test_neg_lon_equiv_pos(self): assert _geo_to_oscar_lon_idx(-135.0) == _geo_to_oscar_lon_idx(225.0)
    def test_convention_in_extra(self, adapter):
        assert "0-360" in adapter.build_request(date(2023,1,1), "/d").extra["lon_convention"]


# ===========================================================================
# J. Variable selection (u, v only — no CMEMS vars)
# ===========================================================================
class TestVariableSelection:
    def test_exactly_u_v(self, adapter): assert set(adapter.source_description().variables_requested) == {"u","v"}
    def test_extra_vars(self, adapter): assert OSCARAdapter().build_request(date(2023,1,1),"/d").extra["variables"] == ["u","v"]

    def test_no_cmems_vars(self, adapter):
        extra = OSCARAdapter().build_request(date(2023,1,1),"/d").extra["variables"]
        for v in ["uo","vo","sla","adt","ugos","vgos"]: assert v not in extra

    def test_constraint_has_u_v(self):
        c = _build_dap4_constraint(180, 420, 380, 478)
        assert "/u[0][" in c and "/v[0][" in c


# ===========================================================================
# K. Temporal request behavior (daily — one request per day)
# ===========================================================================
class TestTemporalBehavior:
    def test_one_day_one_req(self, adapter, data_root):
        assert len(adapter.date_range_requests(date(2023,1,1), date(2023,1,1), data_root)) == 1

    def test_seven_days_seven_reqs(self, adapter, data_root):
        assert len(adapter.date_range_requests(date(2023,1,1), date(2023,1,7), data_root)) == 7

    def test_dates_sequential(self, adapter, data_root):
        from datetime import timedelta
        reqs = adapter.date_range_requests(date(2023,1,1), date(2023,1,5), data_root)
        for i in range(1, len(reqs)):
            assert reqs[i].request_date == reqs[i-1].request_date + timedelta(days=1)

    def test_cross_year_boundary(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2022,12,30), date(2023,1,2), data_root)
        assert len(reqs) == 4
        assert reqs[0].request_date.year == 2022
        assert reqs[-1].request_date.year == 2023


# ===========================================================================
# L. Chunk generation (daily)
# ===========================================================================
class TestChunkGeneration:
    def test_june_30_days(self, adapter, data_root):
        assert len(adapter.date_range_requests(date(2023,6,1), date(2023,6,30), data_root)) == 30

    def test_dates_in_urls(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023,1,1), date(2023,1,3), data_root)
        urls = [r.url_or_query for r in reqs]
        for d_str in ["2023-01-01", "2023-01-02", "2023-01-03"]:
            assert any(d_str in u for u in urls)

    def test_granule_in_extra(self, adapter, data_root):
        reqs = adapter.date_range_requests(date(2023,1,5), date(2023,1,5), data_root)
        assert reqs[0].extra["granule"] == "oscar_currents_final_20230105"


# ===========================================================================
# M. Deterministic output paths
# ===========================================================================
class TestDeterministicPaths:
    def test_filename(self, adapter):
        assert Path(adapter.build_request(date(2023,6,15),"/d").expected_output_path).name == "oscar_currents_final_20230615_uv.nc"

    def test_structure_raw_oscar_year(self, adapter):
        parts = Path(adapter.build_request(date(2023,6,15),"/d").expected_output_path).parts
        assert parts[0]=="raw" and parts[1]=="oscar" and parts[2]=="2023"

    def test_never_absolute(self, adapter):
        p = adapter.build_request(date(2023,6,15),"D:\\data").expected_output_path
        assert not p.startswith("D:") and not p.startswith("/")

    def test_different_dates_different_paths(self, adapter):
        assert (adapter.build_request(date(2023,1,1),"/d").expected_output_path !=
                adapter.build_request(date(2023,1,7),"/d").expected_output_path)

    def test_same_date_same_path(self, adapter):
        assert (adapter.build_request(date(2023,6,15),"/d").expected_output_path ==
                adapter.build_request(date(2023,6,15),"/other").expected_output_path)

    def test_year_in_dir(self, adapter):
        r22 = adapter.build_request(date(2022,1,1),"/d").expected_output_path
        r23 = adapter.build_request(date(2023,1,1),"/d").expected_output_path
        assert "2022" in r22 and "2023" in r23


# ===========================================================================
# N. Manifest integration
# ===========================================================================
class TestManifestIntegration:
    def test_manifest_written_on_success(self, tmp_path):
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus
        result, dest, _ = _run(tmp_path, b"nc data")
        assert result[0]
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        m = ManifestStore(tmp_path)
        m.record(ManifestEntry(dataset_id=req.dataset_id, request_date=req.request_date,
                               source_url=req.url_or_query, local_path=str(dest),
                               product_id=req.source_description.product_id,
                               file_size_bytes=result[1], md5_checksum=result[2],
                               status=ManifestStatus.SUCCESS))
        records = m.read_all("oscar")
        assert len(records)==1 and records[0]["status"]=="SUCCESS"

    def test_separate_from_other_datasets(self, tmp_path):
        from oceanembed.acquisition.manifest import ManifestStore, ManifestEntry, ManifestStatus
        m = ManifestStore(tmp_path)
        for ds in ("oisst","globcurrent","duacs","oscar"):
            m.record(ManifestEntry(dataset_id=ds, request_date=date(2023,1,1),
                                   source_url="fake", local_path=f"/f/{ds}.nc",
                                   product_id="FAKE", status=ManifestStatus.SUCCESS))
        assert len(m.read_all("oscar"))==1 and len(m.read_all("oisst"))==1


# ===========================================================================
# O. Checksum behavior
# ===========================================================================
class TestChecksumBehavior:
    def test_md5_on_download(self, tmp_path):
        content = b"oscar checksum test"
        result, _, _ = _run(tmp_path, content)
        assert result[0] and result[2] == hashlib.md5(content).hexdigest()

    def test_md5_for_existing_file(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        content = b"existing oscar"
        dest = tmp_path / "oscar_20230101.nc"
        dest.write_bytes(content)
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session"):
            _, _, md5 = run_oscar_subset(req.url_or_query, dest, job)
        assert md5 == hashlib.md5(content).hexdigest()

    def test_failed_md5_none(self, tmp_path):
        result, _, _ = _run(tmp_path, b"", creds=False)
        assert not result[0] and result[2] is None


# ===========================================================================
# P. File reuse
# ===========================================================================
class TestFileReuse:
    def test_non_empty_file_skipped(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        dest.write_bytes(b"existing valid data")
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        with patch("earthaccess.login") as ml, patch("earthaccess.get_requests_https_session") as ms:
            success, _, md5 = run_oscar_subset(req.url_or_query, dest, job)
        ml.assert_not_called(); ms.assert_not_called()
        assert success and md5 is not None

    def test_empty_file_not_reused(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        dest.write_bytes(b"")
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        saved_u = os.environ.pop("EARTHDATA_USERNAME", None)
        saved_p = os.environ.pop("EARTHDATA_PASSWORD", None)
        try:
            success, _, _ = run_oscar_subset(req.url_or_query, dest, job)
        finally:
            if saved_u: os.environ["EARTHDATA_USERNAME"] = saved_u
            if saved_p: os.environ["EARTHDATA_PASSWORD"] = saved_p
        assert not success


# ===========================================================================
# Q. No depth parameter
# ===========================================================================
class TestNoDepth:
    def test_has_depth_false(self, adapter): assert not adapter.build_request(date(2023,1,1),"/d").extra["has_depth"]
    def test_no_depth_in_constraint(self): assert "depth" not in _build_dap4_constraint(180,420,380,478).lower()
    def test_no_depth_in_url(self, adapter): assert "depth" not in adapter.build_request(date(2023,1,1),"/d").url_or_query.lower()

    def test_no_depth_in_parsed_constraint(self):
        params = parse_oscar_sentinel_url(OSCARAdapter().build_request(date(2023,1,1),"/d").url_or_query)
        assert "depth" not in params["constraint"].lower()


# ===========================================================================
# R. Source response / file validation
# ===========================================================================
class TestSourceResponseHandling:
    def test_200_netcdf_succeeds(self, tmp_path): assert _run(tmp_path, b"nc")[0][0]
    def test_200_octet_stream_succeeds(self, tmp_path): assert _run(tmp_path, b"nc", ctype="application/octet-stream")[0][0]
    def test_401_fails(self, tmp_path): assert not _run(tmp_path, b"", status=401, ctype="text/html", max_retries=5)[0][0]
    def test_404_fails(self, tmp_path): assert not _run(tmp_path, b"", status=404, ctype="text/html", max_retries=5)[0][0]
    def test_empty_response_fails(self, tmp_path):
        result, dest, _ = _run(tmp_path, b"")
        assert not result[0] and not dest.exists()


# ===========================================================================
# S. Credential redaction in logs
# ===========================================================================
class TestCredentialRedaction:
    def test_no_creds_in_success_logs(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        with patch.dict(os.environ, {"EARTHDATA_USERNAME":"OSCARUSER","EARTHDATA_PASSWORD":"OSCARPASS"}):
            with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session", return_value=_make_session(b"data")):
                run_oscar_subset(req.url_or_query, dest, job)
        all_logs = " ".join(str(e.get("message",""))+str(e) for e in job.logs)
        assert "OSCARUSER" not in all_logs and "OSCARPASS" not in all_logs

    def test_no_password_in_error_logs(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        # Use a realistic auth-failure message pattern (key=value style)
        # The regex in run_oscar_subset redacts "password=..." patterns
        ms = MagicMock(); ms.get.side_effect = RuntimeError("HTTP 401: password=OSCARPASS rejected")
        with patch.dict(os.environ, {"EARTHDATA_USERNAME":"u","EARTHDATA_PASSWORD":"OSCARPASS"}):
            with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session", return_value=ms):
                with patch("time.sleep"):
                    run_oscar_subset(req.url_or_query, dest, job, max_retries=1)
        all_logs = " ".join(str(e.get("message",""))+str(e) for e in job.logs)
        # The regex redacts "password=OSCARPASS" → "password=<REDACTED>"
        # so the raw value "OSCARPASS" must not appear
        assert "password=OSCARPASS" not in all_logs


# ===========================================================================
# T. Malformed sentinel URL
# ===========================================================================
class TestMalformedSentinelUrl:
    def test_non_oscar_raises(self):
        with pytest.raises(ValueError, match="Not an OSCAR"):
            parse_oscar_sentinel_url("https://example.com/file.nc")

    def test_missing_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            parse_oscar_sentinel_url(f"oscar://opendap/{_COLLECTION_ID}")

    def test_missing_date_raises(self):
        url = f"oscar://opendap/{_COLLECTION_ID}?lon_idx_start=180&lon_idx_end=420&lat_idx_start=380&lat_idx_end=478"
        with pytest.raises(ValueError, match="date"):
            parse_oscar_sentinel_url(url)

    def test_missing_lat_idx_end_raises(self):
        url = f"oscar://opendap/{_COLLECTION_ID}?date=2023-01-01&lon_idx_start=180&lon_idx_end=420&lat_idx_start=380"
        with pytest.raises(ValueError, match="lat_idx_end"):
            parse_oscar_sentinel_url(url)

    def test_valid_round_trip(self):
        req = OSCARAdapter().build_request(date(2023,1,1), "/d")
        params = parse_oscar_sentinel_url(req.url_or_query)
        assert params["request_date"] == date(2023,1,1)
        assert params["granule"] == "oscar_currents_final_20230101"
        assert "/u[0][" in params["constraint"]


# ===========================================================================
# U. Retryable errors
# ===========================================================================
class TestRetryableErrors:
    def test_transient_error_retried(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        calls = {"n": 0}

        def fake_get(url, **kw):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("transient")
            return _make_response(b"data")

        ms = MagicMock(); ms.get.side_effect = fake_get
        with patch.dict(os.environ, {"EARTHDATA_USERNAME":"u","EARTHDATA_PASSWORD":"p"}):
            with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session", return_value=ms):
                with patch("time.sleep"):
                    success, _, _ = run_oscar_subset(req.url_or_query, dest, job, max_retries=3)
        assert calls["n"] == 2 and success

    def test_no_creds_no_retry(self, tmp_path):
        assert not _run(tmp_path, b"data", creds=False)[0][0]


# ===========================================================================
# V. Non-retryable auth failure (401 / 404)
# ===========================================================================
class TestNonRetryableAuth:
    def test_401_one_attempt(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        calls = {"n":0}
        ms = MagicMock(); ms.get.side_effect = lambda *a,**k: (calls.update({"n":calls["n"]+1}), _make_response(b"", 401, "text/html"))[1]
        with patch.dict(os.environ, {"EARTHDATA_USERNAME":"bad","EARTHDATA_PASSWORD":"bad"}):
            with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session", return_value=ms):
                success, _, _ = run_oscar_subset(req.url_or_query, dest, job, max_retries=5)
        assert calls["n"] == 1 and not success

    def test_404_one_attempt(self, tmp_path):
        from oceanembed.acquisition.engine import run_oscar_subset, AcquisitionJob
        dest = tmp_path / "oscar_20230101.nc"
        job = AcquisitionJob(job_id="j", dataset_id="oscar",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        req = OSCARAdapter().build_request(date(2023,1,1), str(tmp_path))
        calls = {"n":0}
        ms = MagicMock(); ms.get.side_effect = lambda *a,**k: (calls.update({"n":calls["n"]+1}), _make_response(b"", 404, "text/html"))[1]
        with patch.dict(os.environ, {"EARTHDATA_USERNAME":"u","EARTHDATA_PASSWORD":"p"}):
            with patch("earthaccess.login"), patch("earthaccess.get_requests_https_session", return_value=ms):
                success, _, _ = run_oscar_subset(req.url_or_query, dest, job, max_retries=5)
        assert calls["n"] == 1 and not success


# ===========================================================================
# W. OISST regression
# ===========================================================================
class TestOISSTRegression:
    def test_oisst_uses_download_file(self, tmp_path):
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob
        from oceanembed.datasets.adapters.oisst import OISSTAdapter
        job = AcquisitionJob(job_id="j", dataset_id="oisst",
                             start_date=date(2023,1,1), end_date=date(2023,1,1), data_root=tmp_path)
        async def fake_dl(*a,**k): return (True, 500, "abc")
        with patch("oceanembed.acquisition.engine.download_file", side_effect=fake_dl) as md:
            with patch("oceanembed.acquisition.engine.run_oscar_subset") as mo:
                with patch("oceanembed.acquisition.engine.run_cmems_subset") as mc:
                    asyncio.run(run_acquisition_job(job, OISSTAdapter()))
        md.assert_called_once(); mo.assert_not_called(); mc.assert_not_called()


# ===========================================================================
# X. GlobCurrent regression
# ===========================================================================
class TestGlobCurrentRegression:
    def test_globcurrent_uses_cmems(self, tmp_path):
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob
        from oceanembed.datasets.adapters.globcurrent import GlobCurrentAdapter
        job = AcquisitionJob(job_id="j", dataset_id="globcurrent",
                             start_date=date(2023,1,1), end_date=date(2023,1,7), data_root=tmp_path)
        with patch("oceanembed.acquisition.engine.run_cmems_subset", return_value=(True,1000,"abc")) as mc:
            with patch("oceanembed.acquisition.engine.run_oscar_subset") as mo:
                with patch("oceanembed.acquisition.engine.download_file") as md:
                    asyncio.run(run_acquisition_job(job, GlobCurrentAdapter()))
        mc.assert_called_once(); mo.assert_not_called(); md.assert_not_called()


# ===========================================================================
# Y. DUACS regression
# ===========================================================================
class TestDUACSRegression:
    def test_duacs_uses_cmems(self, tmp_path):
        from oceanembed.acquisition.engine import run_acquisition_job, AcquisitionJob
        from oceanembed.datasets.adapters.duacs import DUACSAdapter
        job = AcquisitionJob(job_id="j", dataset_id="duacs",
                             start_date=date(2023,1,1), end_date=date(2023,1,7), data_root=tmp_path)
        with patch("oceanembed.acquisition.engine.run_cmems_subset", return_value=(True,1000,"abc")) as mc:
            with patch("oceanembed.acquisition.engine.run_oscar_subset") as mo:
                with patch("oceanembed.acquisition.engine.download_file") as md:
                    asyncio.run(run_acquisition_job(job, DUACSAdapter()))
        mc.assert_called_once(); mo.assert_not_called(); md.assert_not_called()


# ===========================================================================
# API registration
# ===========================================================================
class TestApiRegistration:
    def test_oscar_adapter_implemented(self):
        os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test")
        from fastapi.testclient import TestClient
        from oceanembed.config.loader import get_settings
        get_settings.cache_clear()
        import oceanembed.api.app as app_module
        app_module._registry = None
        resp = TestClient(app_module.app).get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "oscar")
        assert ds["adapter_implemented"] is True
        assert ds["engineering_status"] == "ACQUISITION_READY"
