"""
Phase 10 tests — FastAPI REST service layer.

Uses FastAPI's TestClient for in-process testing (no server needed).
"""

import os
import tempfile
import uuid

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    Create a FastAPI TestClient with a fresh temp GeoPackage store.
    Sets GPKG_PATH env var so routes use the temp file.
    """
    fastapi = pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    tmp = tmp_path_factory.mktemp("db")
    gpkg = str(tmp / "test.gpkg")

    os.environ["GPKG_PATH"] = gpkg

    from fastapi.testclient import TestClient
    from src.api.app import app

    with TestClient(app) as c:
        yield c

    # Cleanup
    os.environ.pop("GPKG_PATH", None)


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_docs_endpoint_returns_200(self, client):
        r = client.get("/docs")
        assert r.status_code == 200, f"/docs returned {r.status_code}"

    def test_openapi_json_returns_200(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "paths" in schema

    def test_map_viewer_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_all_expected_endpoints_present(self, client):
        r = client.get("/openapi.json")
        paths = list(r.json()["paths"].keys())
        for expected in ["/api/v1/detect", "/api/v1/detections"]:
            assert expected in paths, f"Missing endpoint: {expected}"


# ---------------------------------------------------------------------------
# /api/v1/detect
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    def test_missing_paths_returns_422(self, client):
        r = client.post("/api/v1/detect", json={
            "img_a_path": "/dev/null/nonexistent_a.png",
            "img_b_path": "/dev/null/nonexistent_b.png",
        })
        assert r.status_code in (422, 400), (
            f"Expected 422/400 for missing files, got {r.status_code}"
        )

    def test_missing_body_returns_422(self, client):
        r = client.post("/api/v1/detect", json={})
        assert r.status_code == 422

    def test_with_existing_files_returns_202(self, client, tmp_path_factory):
        """Submit real image files — should queue successfully."""
        tmp = tmp_path_factory.mktemp("imgs")
        from PIL import Image
        import numpy as np

        for name in ("a.png", "b.png"):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(str(tmp / name))

        r = client.post("/api/v1/detect", json={
            "img_a_path": str(tmp / "a.png"),
            "img_b_path": str(tmp / "b.png"),
        })
        assert r.status_code in (200, 202), f"Expected 200/202, got {r.status_code}: {r.text}"
        body = r.json()
        assert "job_id" in body
        assert "status" in body


# ---------------------------------------------------------------------------
# /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------

class TestJobStatusEndpoint:
    def test_unknown_job_returns_404(self, client):
        r = client.get(f"/api/v1/jobs/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_known_job_after_submit(self, client, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("imgs2")
        from PIL import Image
        import numpy as np

        for name in ("a.png", "b.png"):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img.save(str(tmp / name))

        submit = client.post("/api/v1/detect", json={
            "img_a_path": str(tmp / "a.png"),
            "img_b_path": str(tmp / "b.png"),
        })
        if submit.status_code not in (200, 202):
            pytest.skip("Submit failed, skipping job status test")

        job_id = submit.json()["job_id"]
        r = client.get(f"/api/v1/jobs/{job_id}")
        assert r.status_code == 200
        body = r.json()
        assert body["job_id"] == job_id
        assert "status" in body


# ---------------------------------------------------------------------------
# /api/v1/detections
# ---------------------------------------------------------------------------

class TestDetectionsEndpoint:
    def test_list_returns_200(self, client):
        r = client.get("/api/v1/detections")
        assert r.status_code == 200

    def test_list_response_has_count_and_detections(self, client):
        r = client.get("/api/v1/detections")
        body = r.json()
        assert "count" in body
        assert "detections" in body
        assert isinstance(body["detections"], list)

    def test_single_detection_404_for_unknown_id(self, client):
        r = client.get(f"/api/v1/detections/nonexistent-{uuid.uuid4()}")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /api/v1/detections/query (bbox)
# ---------------------------------------------------------------------------

class TestBboxQueryEndpoint:
    def test_returns_200(self, client):
        r = client.post("/api/v1/detections/query", json={
            "xmin": -180.0,
            "ymin": -90.0,
            "xmax": 180.0,
            "ymax": 90.0,
        })
        assert r.status_code == 200

    def test_empty_bbox_returns_no_results(self, client):
        r = client.post("/api/v1/detections/query", json={
            "xmin": 999.0,
            "ymin": 999.0,
            "xmax": 1000.0,
            "ymax": 1000.0,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 0

    def test_missing_fields_returns_422(self, client):
        r = client.post("/api/v1/detections/query", json={"xmin": 0})
        assert r.status_code == 422
