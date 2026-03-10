"""
Phase 9 tests — Post-processing (connected_components, polygonizer, attributes).
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask_with_regions():
    """256×256 mask with three regions (two above 50px min, one below)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[10:50, 10:50] = 1     # 40×40 = 1 600 px  ✅
    mask[100:120, 100:120] = 1  # 20×20 =   400 px  ✅
    mask[200:203, 200:203] = 1  # 3×3   =     9 px  ❌ (below min_area)
    return mask


# ---------------------------------------------------------------------------
# connected_components.py
# ---------------------------------------------------------------------------

class TestExtractRegions:
    def test_correct_region_count(self):
        from src.postprocessing.connected_components import extract_regions

        mask = _make_mask_with_regions()
        regions = extract_regions(mask, min_area_px=50)
        assert len(regions) == 2, (
            f"Expected 2 regions (1 filtered by min_area), got {len(regions)}"
        )

    def test_minimum_area_filter(self):
        from src.postprocessing.connected_components import extract_regions

        mask = _make_mask_with_regions()
        # With a high min_area, only the largest region survives
        regions = extract_regions(mask, min_area_px=1000)
        assert len(regions) == 1, f"Expected 1 large region, got {len(regions)}"

    def test_empty_mask_returns_empty(self):
        from src.postprocessing.connected_components import extract_regions

        mask = np.zeros((256, 256), dtype=np.uint8)
        regions = extract_regions(mask, min_area_px=50)
        assert len(regions) == 0

    def test_full_mask_returns_one_region(self):
        from src.postprocessing.connected_components import extract_regions

        mask = np.ones((64, 64), dtype=np.uint8)
        regions = extract_regions(mask, min_area_px=0)
        assert len(regions) == 1

    def test_region_has_expected_attributes(self):
        from src.postprocessing.connected_components import extract_regions

        mask = _make_mask_with_regions()
        regions = extract_regions(mask, min_area_px=50)
        for region in regions:
            assert hasattr(region, "pixel_count") or hasattr(region, "area"), (
                "Region should have pixel_count or area attribute"
            )
            assert hasattr(region, "bbox"), "Region should have bbox"


# ---------------------------------------------------------------------------
# polygonizer.py
# ---------------------------------------------------------------------------

class TestMaskToPolygons:
    def test_basic_square_produces_polygon(self):
        from src.postprocessing.polygonizer import mask_to_polygons

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 1  # 50×50 square
        polygons = mask_to_polygons(mask, min_area_px=10)
        assert len(polygons) >= 1, "Expected at least one polygon"

    def test_returns_valid_shapely_polygons(self):
        from shapely.geometry import Polygon
        from src.postprocessing.polygonizer import mask_to_polygons

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        polygons = mask_to_polygons(mask)
        for poly in polygons:
            assert poly.is_valid, "Polygon is invalid"
            assert poly.area > 0, "Polygon has zero area"

    def test_empty_mask_returns_empty_list(self):
        from src.postprocessing.polygonizer import mask_to_polygons

        mask = np.zeros((128, 128), dtype=np.uint8)
        assert mask_to_polygons(mask) == []

    def test_geo_transform_applied(self):
        """With a transform, polygon coordinates should be in geo units, not pixels."""
        from rasterio.transform import from_bounds
        from src.postprocessing.polygonizer import mask_to_polygons

        transform = from_bounds(100, 50, 200, 150, 256, 256)  # ~1 degree/px chunks
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0:50, 0:50] = 1
        polygons = mask_to_polygons(mask, transform=transform)
        if polygons:
            # Centroid should be in geo space (not pixel [0–256])
            cx, cy = polygons[0].centroid.x, polygons[0].centroid.y
            assert 100 <= cx <= 200 or 50 <= cy <= 150, (
                f"Centroid ({cx:.2f}, {cy:.2f}) not in expected geo range"
            )


# ---------------------------------------------------------------------------
# attributes.py
# ---------------------------------------------------------------------------

class TestComputeAttributes:
    def test_geodataframe_columns(self):
        from rasterio.transform import from_bounds
        from src.postprocessing.polygonizer import mask_to_polygons
        from src.postprocessing.attributes import compute_attributes

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 1
        confidence = np.ones((256, 256), dtype=np.float32) * 0.85
        transform = from_bounds(0, 0, 256, 256, 256, 256)

        polygons = mask_to_polygons(mask, min_area_px=10)
        assert polygons, "No polygons to compute attributes on"

        gdf = compute_attributes(polygons, confidence, transform)
        assert "area_m2" in gdf.columns, "Missing area_m2"
        assert "centroid_x" in gdf.columns, "Missing centroid_x"
        assert "mean_confidence" in gdf.columns, "Missing mean_confidence"

    def test_area_is_positive(self):
        from rasterio.transform import from_bounds
        from src.postprocessing.polygonizer import mask_to_polygons
        from src.postprocessing.attributes import compute_attributes

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:60, 10:60] = 1
        confidence = np.full((256, 256), 0.9, dtype=np.float32)
        transform = from_bounds(0, 0, 256, 256, 256, 256)

        polygons = mask_to_polygons(mask)
        gdf = compute_attributes(polygons, confidence, transform)
        assert (gdf["area_m2"] > 0).all(), "All areas should be positive"

    def test_confidence_in_range(self):
        from rasterio.transform import from_bounds
        from src.postprocessing.polygonizer import mask_to_polygons
        from src.postprocessing.attributes import compute_attributes

        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        confidence = np.full((128, 128), 0.72, dtype=np.float32)
        transform = from_bounds(0, 0, 128, 128, 128, 128)

        polygons = mask_to_polygons(mask)
        gdf = compute_attributes(polygons, confidence, transform)
        assert (gdf["mean_confidence"] >= 0).all()
        assert (gdf["mean_confidence"] <= 1).all()
