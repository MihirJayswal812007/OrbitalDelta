"""
Connected-component extraction from binary change masks.

Uses scipy.ndimage.label for reliable, dependency-light component labelling.
Returns structured RegionInfo objects rather than raw label arrays so
downstream code (polygonizer, attribute computation) always has a stable API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as nd_label


@dataclass
class RegionInfo:
    """Metadata for a single connected change region."""

    label_id: int
    pixel_mask: np.ndarray   # boolean (H, W) — True where this region exists
    bbox: tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max)
    pixel_count: int

    @property
    def area_px(self) -> int:
        """Alias for pixel_count (sugar)."""
        return self.pixel_count

    @property
    def centroid_rc(self) -> tuple[float, float]:
        """(row, col) centroid of the region in pixel coordinates."""
        rows, cols = np.where(self.pixel_mask)
        return float(rows.mean()), float(cols.mean())


# 4-connected structure (strict connectivity, fewer noise merges)
_STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
# 8-connected structure (more permissive connectivity)
_STRUCT_8 = np.ones((3, 3), dtype=int)


def extract_regions(
    binary_mask: np.ndarray,
    min_area_px: int = 50,
    connectivity: int = 8,
) -> list[RegionInfo]:
    """
    Extract connected change regions from a binary mask.

    Parameters
    ----------
    binary_mask  : 2-D array, dtype uint8 or bool.  Non-zero → change.
    min_area_px  : Regions smaller than this (in pixels) are discarded.
    connectivity : 4 or 8 neighbourhood connectivity.

    Returns
    -------
    List of RegionInfo objects, sorted by descending pixel_count.

    Raises
    ------
    ValueError : If binary_mask is not 2-D.
    """
    if binary_mask.ndim != 2:
        raise ValueError(
            f"binary_mask must be 2-D, got shape {binary_mask.shape}"
        )

    mask_bool = binary_mask.astype(bool)

    struct = _STRUCT_8 if connectivity == 8 else _STRUCT_4
    labeled, n_components = nd_label(mask_bool, structure=struct)

    regions: list[RegionInfo] = []
    for lab in range(1, n_components + 1):
        region_mask = labeled == lab
        count = int(region_mask.sum())

        if count < min_area_px:
            continue  # filter noise

        rows, cols = np.where(region_mask)
        bbox = (
            int(rows.min()),
            int(cols.min()),
            int(rows.max()),
            int(cols.max()),
        )

        regions.append(
            RegionInfo(
                label_id=lab,
                pixel_mask=region_mask,
                bbox=bbox,
                pixel_count=count,
            )
        )

    # Largest regions first
    regions.sort(key=lambda r: r.pixel_count, reverse=True)
    return regions
