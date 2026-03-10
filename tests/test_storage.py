"""
Phase 9 tests — Spatial Storage (GeoPackageStore).
"""

import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Return a fresh GeoPackageStore pointed at a temp file."""
    from src.storage.geopackage import GeoPackageStore

    db_path = str(tmp_path / "test_store.gpkg")
    return GeoPackageStore(db_path)


@pytest.fixture
def sample_gdf():
    """A tiny GeoDataFrame with one change detection polygon."""
    geopandas = pytest.importorskip("geopandas")
    from shapely.geometry import box

    return geopandas.GeoDataFrame(
        {
            "geometry": [box(10.0, 10.0, 50.0, 50.0)],
            "area_m2": [1600.0],
            "confidence": [0.92],
            "timestamp_a": ["2023-01-01"],
            "timestamp_b": ["2024-01-01"],
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# GeoPackageStore
# ---------------------------------------------------------------------------

class TestGeoPackageStore:
    def test_empty_on_init(self, store):
        result = store.query_all()
        assert len(result) == 0, "Store should be empty on init"

    def test_insert_and_query_all(self, store, sample_gdf):
        store.insert(sample_gdf)
        result = store.query_all()
        assert len(result) == 1, f"Expected 1 record, got {len(result)}"

    def test_multiple_inserts_accumulate(self, store, sample_gdf):
        store.insert(sample_gdf)
        store.insert(sample_gdf)
        result = store.query_all()
        assert len(result) == 2, f"Expected 2 records, got {len(result)}"

    def test_detection_id_added(self, store, sample_gdf):
        """Ensure detection_id is auto-assigned if absent."""
        gdf_no_id = sample_gdf.copy()
        if "detection_id" in gdf_no_id.columns:
            gdf_no_id = gdf_no_id.drop(columns=["detection_id"])

        store.insert(gdf_no_id)
        result = store.query_all()
        assert "detection_id" in result.columns
        assert result["detection_id"].notna().all()

    def test_query_bbox_hits(self, store, sample_gdf):
        store.insert(sample_gdf)
        # BBox that overlaps with box(10, 10, 50, 50)
        result = store.query_bbox(0.0, 0.0, 100.0, 100.0)
        assert len(result) >= 1, "Bbox query should return the overlapping record"

    def test_query_bbox_miss(self, store, sample_gdf):
        store.insert(sample_gdf)
        # BBox far away — no hit
        result = store.query_bbox(500.0, 500.0, 600.0, 600.0)
        assert len(result) == 0, "Distant bbox should return no results"

    def test_get_by_id_found(self, store, sample_gdf):
        import uuid
        gdf = sample_gdf.copy()
        gdf["detection_id"] = [str(uuid.uuid4())]
        detection_id = gdf.iloc[0]["detection_id"]

        store.insert(gdf)
        result = store.get_by_id(detection_id)
        assert len(result) == 1
        assert result.iloc[0]["detection_id"] == detection_id

    def test_get_by_id_not_found(self, store, sample_gdf):
        store.insert(sample_gdf)
        result = store.get_by_id("nonexistent-id-12345")
        assert len(result) == 0

    def test_crs_preserved(self, store, sample_gdf):
        store.insert(sample_gdf)
        result = store.query_all()
        assert result.crs is not None
        assert "4326" in str(result.crs)

    def test_store_file_created(self, tmp_path, sample_gdf):
        from src.storage.geopackage import GeoPackageStore

        db_path = str(tmp_path / "fresh.gpkg")
        store = GeoPackageStore(db_path)
        store.insert(sample_gdf)
        assert os.path.exists(db_path), "GeoPackage file should exist after insert"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_module_importable(self):
        """models.py should be importable regardless of SQLAlchemy being available."""
        from src.storage import models
        assert hasattr(models, "ChangeDetection")

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("sqlalchemy"),
        reason="SQLAlchemy not installed"
    )
    def test_change_detection_fields(self):
        from src.storage.models import ChangeDetection

        obj = ChangeDetection()
        # Should have all expected fields without raising
        attrs = ["detection_id", "area_m2", "confidence", "timestamp_a", "timestamp_b"]
        for attr in attrs:
            assert hasattr(obj, attr), f"Missing attribute: {attr}"
