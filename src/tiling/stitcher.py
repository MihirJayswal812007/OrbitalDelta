"""
Tile stitcher — reassembles overlapping tile predictions into a full-resolution map.

Uses Hann-window (cosine taper) blending to eliminate seam artifacts at tile boundaries.
"""

from __future__ import annotations

import numpy as np


class TileStitcher:
    """
    Reassembles tiles into the original image resolution with seam-free blending.

    Uses a cosine (Hann) weight function so tiles fade smoothly into each other
    in the overlap region, eliminating hard seam artifacts.

    Args:
        tile_size:    Size of each tile (must match TileSplitter)
        overlap:      Overlap in pixels (must match TileSplitter)
        output_shape: (H, W) of the original (unpadded) image
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        output_shape: tuple[int, int] = (1024, 1024),
    ) -> None:
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.output_h, self.output_w = output_shape

        # Accumulation buffers (padded to handle out-of-bounds tile writes)
        pad_h = (self.stride - (self.output_h - tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (self.output_w - tile_size) % self.stride) % self.stride
        self._acc_h = self.output_h + pad_h
        self._acc_w = self.output_w + pad_w

        self._data = np.zeros((self._acc_h, self._acc_w), dtype=np.float64)
        self._weight = np.zeros((self._acc_h, self._acc_w), dtype=np.float64)

        # Build 2D Hann (cosine taper) blending kernel — peaks at center, fades at edges
        hann_1d = np.hanning(tile_size).astype(np.float64)
        self._blend_kernel = np.outer(hann_1d, hann_1d)  # (tile_size, tile_size)

    def add_tile(
        self,
        prediction: np.ndarray,
        row: int,
        col: int,
    ) -> None:
        """
        Add a tile prediction to the accumulation buffer.

        Args:
            prediction: 2D float array (tile_size, tile_size) — probabilities in [0, 1]
            row:        Tile grid row index (NOT pixel offset)
            col:        Tile grid col index
        """
        r = row * self.stride
        c = col * self.stride

        tile_h, tile_w = prediction.shape[:2]
        r_end = r + tile_h
        c_end = c + tile_w

        # Safety: clip to accumulation buffer bounds
        r_end = min(r_end, self._acc_h)
        c_end = min(c_end, self._acc_w)
        th = r_end - r
        tw = c_end - c

        kernel = self._blend_kernel[:th, :tw]
        self._data[r:r_end, c:c_end] += prediction[:th, :tw] * kernel
        self._weight[r:r_end, c:c_end] += kernel

    def add_tile_by_offset(
        self,
        prediction: np.ndarray,
        orig_row: int,
        orig_col: int,
    ) -> None:
        """
        Add a tile prediction using pixel-level grid offsets (from TileRecord).

        Args:
            prediction: 2D float array
            orig_row:   Pixel row in the padded image where this tile begins
            orig_col:   Pixel col in the padded image where this tile begins
        """
        r, c = orig_row, orig_col
        tile_h, tile_w = prediction.shape[:2]
        r_end = min(r + tile_h, self._acc_h)
        c_end = min(c + tile_w, self._acc_w)
        th = r_end - r
        tw = c_end - c

        kernel = self._blend_kernel[:th, :tw]
        self._data[r:r_end, c:c_end] += prediction[:th, :tw] * kernel
        self._weight[r:r_end, c:c_end] += kernel

    def stitch(self) -> np.ndarray:
        """
        Compute the final blended prediction map.

        Returns:
            (H, W) float32 array clipped to the original (unpadded) image shape.
        """
        safe_weight = np.where(self._weight > 1e-9, self._weight, 1.0)
        result = self._data / safe_weight
        result = np.clip(result, 0.0, 1.0)
        return result[: self.output_h, : self.output_w].astype(np.float32)

    def reset(self) -> None:
        """Clear accumulation buffers for reuse."""
        self._data[:] = 0.0
        self._weight[:] = 0.0
