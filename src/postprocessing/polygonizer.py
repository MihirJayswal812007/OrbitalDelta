"""
Polygonizer — converts binary change masks to Shapely polygon geometries.

Uses rasterio.features.shapes (backed by GDAL) for fast, accurate polygon
extraction via the Marching Squares algorithm.  Falls back to scipy contour
tracing if rasterio is not available.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from shapely.geometry import shape, Polygon

logger = logging.getLogger(__name__)


def mask_to_polygons(
    binary_mask: np.ndarray,
    min_area_px: int = 10,
    transform: Any = None,
    simplify_tolerance: float = 0.5,
) -> list[Polygon]:
    """
    Convert a binary change mask to a list of Shapely Polygons.

    If *transform* (rasterio Affine) is provided, polygons are in the image's
    coordinate reference system (metres or degrees).  Otherwise they are in
    pixel coordinates.

    Parameters
    ----------
    binary_mask        : 2-D uint8 / bool array; non-zero → change
    min_area_px        : Discard polygons with fewer pixels than this
    transform          : rasterio Affine transform (optional)
    simplify_tolerance : Douglas-Peucker simplification tolerance in CRS units

    Returns
    -------
    List of shapely Polygon objects (may be empty if no changes detected)
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got shape {binary_mask.shape}")

    mask_u8 = (binary_mask > 0).astype(np.uint8)

    if mask_u8.max() == 0:
        return []  # no changes

    polygons: list[Polygon] = []

    try:
        from rasterio.features import shapes as rasterio_shapes

        # rasterio.features.shapes requires a real Affine transform or omitting it.
        # Passing None causes an error in rasterio's internal GDAL checks.
        shape_kwargs: dict = {"mask": mask_u8}
        if transform is not None:
            shape_kwargs["transform"] = transform

        # rasterio.features.shapes expects C-contiguous uint8/int32/float32
        for geom_dict, value in rasterio_shapes(mask_u8, **shape_kwargs):
            if value == 0:
                continue
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)  # fix self-intersections
            if poly.area < min_area_px:
                continue
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
            polygons.append(poly)

    except ImportError:
        logger.warning("rasterio not available — using scipy contour fallback")
        polygons = _scipy_fallback(mask_u8, min_area_px, transform)

    return polygons


def _scipy_fallback(
    mask_u8: np.ndarray,
    min_area_px: int,
    transform: Any,
) -> list[Polygon]:
    """Pure scipy / shapely polygon extraction (slower, no rasterio required)."""
    from scipy.ndimage import label as nd_label
    from shapely.geometry import MultiPoint

    labeled, n = nd_label(mask_u8)
    polygons: list[Polygon] = []

    for lab in range(1, n + 1):
        region = labeled == lab
        count = int(region.sum())
        if count < min_area_px:
            continue

        rows, cols = np.where(region)

        # Map to world coordinates if transform is available
        if transform is not None:
            xs, ys = [], []
            for r, c in zip(rows, cols):
                x, y = transform * (c + 0.5, r + 0.5)
                xs.append(x)
                ys.append(y)
            pts = list(zip(xs, ys))
        else:
            pts = list(zip(cols.tolist(), rows.tolist()))  # (x=col, y=row)

        if len(pts) >= 3:
            try:
                poly = MultiPoint(pts).convex_hull
                if isinstance(poly, Polygon):
                    polygons.append(poly)
            except Exception:
                pass

    return polygons
