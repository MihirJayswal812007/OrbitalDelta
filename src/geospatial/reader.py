"""
GeoReader — reads satellite imagery (GeoTIFF, JP2, PNG, JPEG) and extracts
geospatial metadata.  Plain images (no CRS) are silently accepted with empty
GeoMetadata so every downstream module stays consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.geospatial.geotiff_io import GeoMetadata, load_geotiff


class GeoReader:
    """
    Reads raster images as (C, H, W) float32 numpy arrays, together with
    full geospatial metadata.

    Usage::

        arr, meta = GeoReader.read("scene.tif")
        arr, meta = GeoReader.read("photo.png")   # meta.has_geo == False
    """

    # Extensions handled by rasterio (GeoTIFF, JP2, ERDAS, …)
    _RASTERIO_EXTS = {".tif", ".tiff", ".jp2", ".img", ".vrt", ".nc"}

    @classmethod
    def read(cls, path: str | Path) -> tuple[np.ndarray, GeoMetadata]:
        """
        Read a raster file.

        Returns
        -------
        array : np.ndarray
            Shape (C, H, W), dtype float32, values normalised to [0, 1].
        meta  : GeoMetadata
            Populated if the file carries geospatial information;
            ``meta.has_geo`` is False for plain images (PNG/JPEG).
        """
        path = Path(path)
        ext = path.suffix.lower()

        if ext in cls._RASTERIO_EXTS:
            return load_geotiff(path)

        # Plain image (PNG / JPEG / BMP) — fall back to PIL
        return cls._read_plain(path)

    @classmethod
    def _read_plain(cls, path: Path) -> tuple[np.ndarray, GeoMetadata]:
        """Load a plain image without geo-metadata."""
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("Pillow is required for non-GeoTIFF images") from e

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)                    # (3, H, W)

        meta = GeoMetadata(
            width=img.width,
            height=img.height,
            count=3,
            dtype="float32",
            source_path=str(path),
        )
        return arr, meta
