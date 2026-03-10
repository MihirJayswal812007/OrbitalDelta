"""
Geospatial I/O utilities for satellite image change detection.

Handles:
  - Reading GeoTIFF files with CRS and geotransform preservation
  - Writing georeferenced change maps as GeoTIFF
  - Coordinate transformation utilities
  - Spatial resolution extraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeoMetadata:
    """Encapsulates all geospatial metadata for a raster image."""

    crs: Any = None                    # rasterio CRS object
    transform: Any = None             # Affine transform (rasterio.transform.Affine)
    width: int = 0
    height: int = 0
    count: int = 0                    # Number of bands
    dtype: str = "uint8"
    nodata: float | None = None
    resolution_m: float | None = None  # Spatial resolution in metres (if known)
    source_path: str = ""

    @property
    def has_geo(self) -> bool:
        """True if CRS and transform are available."""
        return self.crs is not None and self.transform is not None

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel (row, col) → world (x, y) coordinates."""
        if not self.has_geo:
            raise ValueError("No geospatial metadata available")
        x, y = self.transform * (col + 0.5, row + 0.5)
        return float(x), float(y)

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert world (x, y) → pixel (row, col) — inverse of transform."""
        if not self.has_geo:
            raise ValueError("No geospatial metadata available")
        from rasterio.transform import rowcol
        row, col = rowcol(self.transform, x, y)
        return int(row), int(col)

    def pixel_area_m2(self) -> float | None:
        """Return area of a single pixel in m² (if resolution available)."""
        if self.resolution_m is not None:
            return self.resolution_m ** 2
        if self.transform is not None:
            # Estimate from transform pixel size
            px_size = abs(self.transform.a)
            if px_size > 0:
                return px_size ** 2
        return None


def load_geotiff(path: str | Path) -> tuple[np.ndarray, GeoMetadata]:
    """
    Load a GeoTIFF and extract spatial metadata.

    Returns:
        data:  (C, H, W) float32 array, band-normalised to [0, 1]
        meta:  GeoMetadata object
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("rasterio is required for GeoTIFF I/O: pip install rasterio") from e

    path = Path(path)
    with rasterio.open(path) as src:
        meta = GeoMetadata(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            source_path=str(path),
        )

        # Estimate spatial resolution
        if src.transform:
            meta.resolution_m = abs(float(src.transform.a))

        # Read data (all bands)
        data = src.read().astype(np.float32)  # (C, H, W)

        # Normalize each band to [0, 1] using 2nd–98th percentile
        for c in range(data.shape[0]):
            band = data[c]
            valids = band[band != (src.nodata if src.nodata else np.nan)]
            if len(valids) == 0:
                continue
            p2, p98 = np.percentile(valids, [2, 98])
            rng = p98 - p2
            if rng > 0:
                data[c] = np.clip((band - p2) / rng, 0.0, 1.0)
            else:
                data[c] = np.clip(band / max(band.max(), 1e-6), 0.0, 1.0)

    logger.info(
        f"Loaded GeoTIFF: {path.name} | "
        f"{meta.width}x{meta.height} | "
        f"{meta.count} bands | "
        f"res={meta.resolution_m:.2f}m | "
        f"CRS={meta.crs}"
    )
    return data, meta


def save_geotiff_mask(
    mask: np.ndarray,
    meta: GeoMetadata,
    output_path: str | Path,
    prob_map: np.ndarray | None = None,
    compress: str = "lzw",
) -> Path:
    """
    Save a binary change mask (and optional probability map) as a GeoTIFF.

    Preserves the original CRS and geotransform from the source image.

    Args:
        mask:        2D binary array (H, W) in {0, 1}
        meta:        GeoMetadata from the source image
        output_path: Output .tif file path
        prob_map:    Optional (H, W) float32 probability array
        compress:    Compression codec ('lzw', 'deflate', 'none')

    Returns:
        Path to saved file
    """
    try:
        import rasterio
        from rasterio.enums import Compression
    except ImportError as e:
        raise ImportError("rasterio is required") from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_bands = 2 if prob_map is not None else 1
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": meta.width,
        "height": meta.height,
        "count": n_bands,
        "compress": compress,
    }
    if meta.has_geo:
        profile["crs"] = meta.crs
        profile["transform"] = meta.transform

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.float32), 1)
        dst.update_tags(1, DESCRIPTION="binary_change_mask", VALUES="0=no_change,1=change")

        if prob_map is not None:
            dst.write(prob_map.astype(np.float32), 2)
            dst.update_tags(2, DESCRIPTION="change_probability", VALUES="[0,1]")

        dst.update_tags(
            SOURCE="OrbitalDelta",
            MODEL="SiameseUNet",
        )

    logger.info(f"Saved georeferenced change map: {output_path}")
    return output_path


def geotiff_to_rgb(data: np.ndarray, bands: tuple[int, int, int] = (0, 1, 2)) -> np.ndarray:
    """
    Convert (C, H, W) float32 array to (H, W, 3) uint8 RGB for display.

    Args:
        data:  Float array in [0, 1], shape (C, H, W)
        bands: Band indices to use as (R, G, B)
    """
    n_ch = data.shape[0]
    rgb_bands = [min(b, n_ch - 1) for b in bands]
    rgb = np.stack([data[b] for b in rgb_bands], axis=2)  # (H, W, 3)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
