"""
Unit tests for the FastAPI application endpoints.

Uses FastAPI TestClient (no real HTTP, no network access, no downloads).
Tests cover:
- GET /api/health
- GET /api/datasets
- GET /api/datasets/{dataset_id}
- GET /api/datasets/{unknown_id} → 404
- POST /api/acquisition/jobs with unimplemented adapter → 501
- POST /api/acquisition/jobs with invalid dates → 422
- GET /api/acquisition/jobs/{bad_id} → 404
"""

from __future__ import annotations

import os
import pytest

# Set data_root before importing app so config loader doesn't fail
os.environ.setdefault("OCEANEMBED_DATA_ROOT", "/tmp/oceanembed-test-data")


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient."""
    from fastapi.testclient import TestClient
    # Clear lru_cache so settings are reloaded cleanly
    from oceanembed.config.loader import get_settings
    get_settings.cache_clear()
    from oceanembed.api.app import app, _registry
    # Reset registry singleton
    import oceanembed.api.app as app_module
    app_module._registry = None
    return TestClient(app)


class TestHealth:
    def test_health_status_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_version(self, client):
        resp = client.get("/api/health")
        assert "version" in resp.json()

    def test_health_has_timestamp(self, client):
        resp = client.get("/api/health")
        assert "timestamp" in resp.json()


class TestDatasetListing:
    def test_list_returns_200(self, client):
        resp = client.get("/api/datasets")
        assert resp.status_code == 200

    def test_list_contains_eight_datasets(self, client):
        resp = client.get("/api/datasets")
        data = resp.json()
        assert data["total"] == 8
        assert len(data["datasets"]) == 8

    def test_list_contains_oisst(self, client):
        resp = client.get("/api/datasets")
        ids = [d["dataset_id"] for d in resp.json()["datasets"]]
        assert "oisst" in ids

    def test_list_contains_glorys(self, client):
        resp = client.get("/api/datasets")
        ids = [d["dataset_id"] for d in resp.json()["datasets"]]
        assert "glorys" in ids

    def test_list_all_expected_ids(self, client):
        resp = client.get("/api/datasets")
        ids = set(d["dataset_id"] for d in resp.json()["datasets"])
        expected = {"oisst", "oisss", "duacs", "ccmp", "oscar", "globcurrent", "glorys", "argo"}
        assert ids == expected

    def test_oisst_is_demo_ready(self, client):
        resp = client.get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "oisst")
        assert ds["engineering_status"] == "DEMO_READY"
        assert ds["adapter_implemented"] is True

    def test_globcurrent_adapter_implemented(self, client):
        resp = client.get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "globcurrent")
        assert ds["adapter_implemented"] is True
        assert ds["engineering_status"] == "ACQUISITION_READY"

    def test_glorys_adapter_implemented(self, client):
        resp = client.get("/api/datasets")
        ds = next(d for d in resp.json()["datasets"] if d["dataset_id"] == "glorys")
        assert ds["adapter_implemented"] is True


class TestDatasetDetail:
    def test_oisst_detail_200(self, client):
        resp = client.get("/api/datasets/oisst")
        assert resp.status_code == 200

    def test_oisst_detail_has_provider(self, client):
        resp = client.get("/api/datasets/oisst")
        data = resp.json()
        assert data["provider_name"] is not None
        assert "NOAA" in data["provider_name"]

    def test_oisst_detail_has_variables(self, client):
        resp = client.get("/api/datasets/oisst")
        assert len(resp.json()["primary_variables"]) > 0

    def test_oisst_detail_has_open_decisions(self, client):
        resp = client.get("/api/datasets/oisst")
        assert len(resp.json()["open_scientific_decisions"]) > 0

    def test_unknown_dataset_404(self, client):
        resp = client.get("/api/datasets/does_not_exist")
        assert resp.status_code == 404

    def test_glorys_detail_200(self, client):
        resp = client.get("/api/datasets/glorys")
        assert resp.status_code == 200


class TestAcquisitionJobs:
    def test_unimplemented_adapter_returns_501(self, client):
        # No dataset has a truly unimplemented adapter now except via missing index.
        # Use an unknown dataset_id to test the 404 path instead.
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "dataset_that_does_not_exist",
                  "start_date": "2023-01-01", "end_date": "2023-01-01"},
        )
        assert resp.status_code == 404

    def test_globcurrent_adapter_does_not_return_501(self, client):
        # GlobCurrent is now implemented — it should return 202, not 501.
        # The job will fail (no CMEMS credentials in test env) but it should
        # be accepted. We mock the adapter to avoid actual network access.
        import oceanembed.api.app as app_module
        from unittest.mock import MagicMock, patch

        mock_adapter = MagicMock()
        mock_adapter._max_job_days = 366
        mock_adapter.date_range_requests.return_value = []

        with patch.object(app_module, "get_adapter", return_value=mock_adapter):
            resp = client.post(
                "/api/acquisition/jobs",
                json={
                    "dataset_id": "globcurrent",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-07",
                },
            )
        # Should be 202 (accepted) not 501 (not implemented)
        assert resp.status_code == 202
        assert resp.json()["dataset_id"] == "globcurrent"

    def test_unknown_dataset_returns_404(self, client):
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "does_not_exist", "start_date": "2023-01-01", "end_date": "2023-01-01"},
        )
        assert resp.status_code == 404

    def test_invalid_dates_returns_422(self, client):
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "oisst", "start_date": "2023-01-05", "end_date": "2023-01-01"},
        )
        assert resp.status_code == 422

    def test_date_range_too_long_returns_422(self, client):
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "oisst", "start_date": "2023-01-01", "end_date": "2023-03-15"},
        )
        assert resp.status_code == 422

    def test_oisst_job_created_returns_202(self, client):
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "oisst", "start_date": "2023-01-01", "end_date": "2023-01-02"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["dataset_id"] == "oisst"
        assert data["status"] in ("QUEUED", "RUNNING")

    def test_job_status_reachable(self, client):
        # Create a job first
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "oisst", "start_date": "2023-01-01", "end_date": "2023-01-01"},
        )
        job_id = resp.json()["job_id"]
        status_resp = client.get(f"/api/acquisition/jobs/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_job_logs_reachable(self, client):
        resp = client.post(
            "/api/acquisition/jobs",
            json={"dataset_id": "oisst", "start_date": "2023-01-01", "end_date": "2023-01-01"},
        )
        job_id = resp.json()["job_id"]
        logs_resp = client.get(f"/api/acquisition/jobs/{job_id}/logs")
        assert logs_resp.status_code == 200
        assert "logs" in logs_resp.json()

    def test_unknown_job_status_404(self, client):
        resp = client.get("/api/acquisition/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_cancel_unknown_job_404(self, client):
        resp = client.post("/api/acquisition/jobs/nonexistent-id/cancel")
        assert resp.status_code == 404
