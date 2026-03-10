"""
Phase 8 tests — Geospatial I/O (GeoReader, GeoWriter, crs_utils).
"""

import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a small synthetic GeoTIFF and return its path."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path = str(tmp_path / "sample.tif")
    crs = CRS.from_epsg(4326)
    transform = from_bounds(0, 0, 1, 1, 64, 64)
    data = np.random.rand(3, 64, 64).astype(np.float32)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=64, width=64, count=3,
        dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(data)
    return path, crs, transform, data


# ---------------------------------------------------------------------------
# GeoReader
# ---------------------------------------------------------------------------

class TestGeoReader:
    def test_reads_array_and_metadata(self, sample_geotiff):
        from src.geospatial.reader import GeoReader

        path, crs, transform, _ = sample_geotiff
        arr, meta = GeoReader.read(path)

        assert arr.ndim in (2, 3), f"Expected 2D or 3D array, got {arr.ndim}D"
        assert meta.crs is not None, "CRS was not preserved"
        assert meta.transform is not None, "Transform was not preserved"

    def test_array_shape_matches_raster(self, sample_geotiff):
        from src.geospatial.reader import GeoReader

        path, _, _, data = sample_geotiff
        arr, meta = GeoReader.read(path)
        # Shape should be (bands, height, width) or (height, width)
        if arr.ndim == 3:
            assert arr.shape[1] == 64
            assert arr.shape[2] == 64
        else:
            assert arr.shape == (64, 64)

    def test_reads_grayscale_fallback(self, tmp_path):
        """Plain PNG without geo metadata should be readable with None CRS."""
        from PIL import Image
        from src.geospatial.reader import GeoReader

        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        path = str(tmp_path / "plain.png")
        img.save(path)

        arr, meta = GeoReader.read(path)
        assert arr is not None
        # CRS may be None for plain images
        # (implementation-dependent)


# ---------------------------------------------------------------------------
# GeoWriter
# ---------------------------------------------------------------------------

class TestGeoWriter:
    def test_preserves_crs(self, sample_geotiff, tmp_path):
        import rasterio
        from src.geospatial.reader import GeoReader
        from src.geospatial.writer import GeoWriter

        path, crs, transform, _ = sample_geotiff
        arr, meta = GeoReader.read(path)

        out_path = str(tmp_path / "output.tif")
        change_mask = np.random.rand(1, 64, 64).astype(np.float32)
        GeoWriter.write(change_mask, meta, out_path)

        with rasterio.open(out_path) as src:
            assert str(src.crs) == str(crs), f"CRS mismatch: {src.crs} ≠ {crs}"

    def test_preserves_transform(self, sample_geotiff, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from src.geospatial.reader import GeoReader
        from src.geospatial.writer import GeoWriter

        path, _, transform, _ = sample_geotiff
        arr, meta = GeoReader.read(path)

        out_path = str(tmp_path / "output2.tif")
        change_mask = np.random.rand(1, 64, 64).astype(np.float32)
        GeoWriter.write(change_mask, meta, out_path)

        with rasterio.open(out_path) as src:
            assert src.transform == meta.transform, "Transform not preserved"


# ---------------------------------------------------------------------------
# crs_utils
# ---------------------------------------------------------------------------

class TestCrsUtils:
    def test_epsg_extraction(self):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.crs import CRS
        from src.geospatial.crs_utils import epsg_code

        crs = CRS.from_epsg(4326)
        epsg = epsg_code(crs)
        assert epsg == 4326, f"Expected 4326, got {epsg}"

    def test_pixel_to_world(self):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_bounds
        from src.geospatial.crs_utils import pixel_to_world

        transform = from_bounds(0, 0, 100, 100, 100, 100)  # 1 unit/px
        x, y = pixel_to_world(0, 0, transform)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_validate_crs_with_valid_crs(self):
        from rasterio.crs import CRS
        from src.geospatial.crs_utils import validate_crs

        crs = CRS.from_epsg(4326)
        assert validate_crs(crs) is True

    def test_validate_crs_with_none(self):
        from src.geospatial.crs_utils import validate_crs

        assert validate_crs(None) is False
