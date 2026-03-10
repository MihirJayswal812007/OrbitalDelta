"""
Attribute computation for detected change polygons.

Given a list of Shapely polygons and a confidence map, computes per-region
attributes and returns a GeoDataFrame for downstream storage / export.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


def compute_attributes(
    polygons: list[Polygon],
    confidence_map: np.ndarray,
    transform: Any = None,
    crs: Any = None,
    timestamp_a: str = "",
    timestamp_b: str = "",
) -> "geopandas.GeoDataFrame":  # noqa: F821
    """
    Compute attributes for each detected change polygon.

    Parameters
    ----------
    polygons        : List of Shapely Polygon objects (pixel or world coords)
    confidence_map  : 2-D float32 array — model output probability in [0, 1]
    transform       : rasterio Affine transform (for area in real units)
    crs             : rasterio CRS (attached to the returned GeoDataFrame)
    timestamp_a     : ISO-8601 date string for image A (e.g. "2023-01-01")
    timestamp_b     : ISO-8601 date string for image B

    Returns
    -------
    GeoDataFrame with columns:
        geometry, area_m2, centroid_x, centroid_y, bbox_minx, bbox_miny,
        bbox_maxx, bbox_maxy, perimeter_m, mean_confidence,
        timestamp_a, timestamp_b
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "geopandas is required: pip install geopandas"
        ) from e

    if not polygons:
        return gpd.GeoDataFrame(
            columns=[
                "geometry", "area_m2", "centroid_x", "centroid_y",
                "bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy",
                "perimeter_m", "mean_confidence", "timestamp_a", "timestamp_b",
            ],
            crs=crs,
        )

    # Pixel area in m² from transform
    pixel_area = _pixel_area_m2(transform)

    rows = []
    for poly in polygons:
        centroid = poly.centroid
        minx, miny, maxx, maxy = poly.bounds

        # Area in m² (polygon.area is in CRS units²)
        if pixel_area is not None:
            area_m2 = poly.area * pixel_area
            perim_m = poly.length * (pixel_area ** 0.5)
        else:
            # Fallback: area in pixels if no transform
            area_m2 = poly.area
            perim_m = poly.length

        # Mean confidence in the polygon's bounding box (fast approximation)
        mean_conf = _mean_confidence_in_polygon(poly, confidence_map, transform)

        rows.append({
            "geometry": poly,
            "area_m2": float(area_m2),
            "centroid_x": float(centroid.x),
            "centroid_y": float(centroid.y),
            "bbox_minx": float(minx),
            "bbox_miny": float(miny),
            "bbox_maxx": float(maxx),
            "bbox_maxy": float(maxy),
            "perimeter_m": float(perim_m),
            "mean_confidence": float(mean_conf),
            "timestamp_a": timestamp_a,
            "timestamp_b": timestamp_b,
        })

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    return gdf


def _pixel_area_m2(transform) -> float | None:
    """Return area of one pixel in m² from an affine transform, or None."""
    if transform is None:
        return None
    px = abs(float(transform.a))
    return px * px if px > 0 else None


def _mean_confidence_in_polygon(
    poly: Polygon,
    conf_map: np.ndarray,
    transform: Any,
) -> float:
    """
    Approximate mean confidence inside a polygon by sampling the confidence map.

    Uses the polygon bounding box for speed; falls back to global mean if
    coordinate mapping is unavailable.
    """
    if conf_map is None or conf_map.size == 0:
        return float("nan")

    if transform is None:
        # No geo transform — return global map mean as best estimate
        return float(conf_map.mean())

    minx, miny, maxx, maxy = poly.bounds

    # Map world bbox → pixel bbox using inverse transform
    from rasterio.transform import rowcol
    try:
        r_min, c_min = rowcol(transform, minx, maxy)  # top-left
        r_max, c_max = rowcol(transform, maxx, miny)  # bottom-right
    except Exception:
        return float(conf_map.mean())

    h, w = conf_map.shape[:2]
    r_min = max(0, min(int(r_min), h - 1))
    r_max = max(0, min(int(r_max) + 1, h))
    c_min = max(0, min(int(c_min), w - 1))
    c_max = max(0, min(int(c_max) + 1, w))

    patch = conf_map[r_min:r_max, c_min:c_max]
    return float(patch.mean()) if patch.size > 0 else float(conf_map.mean())
