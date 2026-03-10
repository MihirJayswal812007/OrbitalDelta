"""
GeoWriter — writes change maps as GeoTIFF with preserved CRS and geotransform.

Thin wrapper around geotiff_io.save_geotiff_mask that follows the same
interface pattern as GeoReader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.geospatial.geotiff_io import GeoMetadata, save_geotiff_mask


class GeoWriter:
    """
    Writes raster data as GeoTIFF, preserving the original CRS and transform.

    Usage::

        GeoWriter.write(change_mask, meta, "outputs/change_map.tif")
        GeoWriter.write(change_mask, meta, "outputs/change_map.tif", prob_map=probs)
    """

    @staticmethod
    def write(
        array: np.ndarray,
        meta: GeoMetadata,
        path: str | Path,
        prob_map: np.ndarray | None = None,
        compress: str = "lzw",
    ) -> Path:
        """
        Write a change mask (and optional probability map) as a GeoTIFF.

        Parameters
        ----------
        array    : 2-D binary (H, W) or 3-D (1, H, W) change mask, values {0, 1}
        meta     : GeoMetadata from the source image (carries CRS + transform)
        path     : Output file path (.tif / .tiff)
        prob_map : Optional (H, W) float32 probability array written as band 2
        compress : LZW (default), deflate, or none

        Returns
        -------
        Path object pointing to the written file.
        """
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]  # (1, H, W) → (H, W)

        return save_geotiff_mask(
            mask=array,
            meta=meta,
            output_path=path,
            prob_map=prob_map,
            compress=compress,
        )
