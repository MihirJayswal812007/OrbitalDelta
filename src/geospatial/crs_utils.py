"""
CRS utilities — coordinate reference system helpers.

Provides thin, dependency-free helpers around rasterio.crs and pyproj
for CRS validation, EPSG extraction, and pixel↔coordinate conversions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.geospatial.geotiff_io import GeoMetadata


def epsg_code(crs) -> int | None:
    """
    Return the EPSG integer code for a rasterio CRS, or None if unknown.

    Parameters
    ----------
    crs : rasterio.crs.CRS | None

    Returns
    -------
    int | None
    """
    if crs is None:
        return None
    try:
        return int(crs.to_epsg())
    except Exception:
        return None


def is_geographic(crs) -> bool:
    """True if the CRS uses geographic (degrees) coordinates (e.g. EPSG:4326)."""
    if crs is None:
        return False
    try:
        return crs.is_geographic
    except AttributeError:
        return False


def is_projected(crs) -> bool:
    """True if the CRS uses projected (metres/feet) coordinates."""
    if crs is None:
        return False
    try:
        return not crs.is_geographic
    except AttributeError:
        return False


def validate_crs(crs) -> bool:
    """
    Return True if the CRS object is valid and usable.

    Checks that it can be converted to a WKT string without error.
    """
    if crs is None:
        return False
    try:
        _ = crs.to_wkt()
        return True
    except Exception:
        return False


def pixel_to_world(
    row: int,
    col: int,
    transform,
) -> tuple[float, float]:
    """
    Convert pixel (row, col) to world (x, y) using an affine transform.

    Parameters
    ----------
    row, col  : Zero-based pixel indices
    transform : rasterio Affine transform

    Returns
    -------
    (x, y) in the CRS units (degrees or metres)
    """
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def world_to_pixel(
    x: float,
    y: float,
    transform,
) -> tuple[int, int]:
    """
    Convert world (x, y) to pixel (row, col) using inverse of affine transform.

    Parameters
    ----------
    x, y      : World coordinates in the CRS units
    transform : rasterio Affine transform

    Returns
    -------
    (row, col) — integer pixel indices (may be outside image bounds)
    """
    from rasterio.transform import rowcol
    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def pixel_resolution_m(transform) -> float | None:
    """
    Estimate the ground sampling distance in metres from the affine transform.

    For geographic CRS the pixel size is in degrees — this returns None in
    that case (convert using pyproj if you need metres for geographic CRS).

    Returns None if transform is None.
    """
    if transform is None:
        return None
    # Affine.a = x-pixel width; absolute value gives GSD
    return abs(float(transform.a))
