"""
Tile splitter for large satellite image processing.

Splits large images into overlapping patches compatible with the model input size.
Handles edge padding and tracks tile coordinates for reconstruction.
"""

from __future__ import annotations

import numpy as np


class TileRecord:
    """Metadata + data for a single tile."""

    __slots__ = ("tile", "row", "col", "orig_row", "orig_col", "tile_h", "tile_w")

    def __init__(
        self,
        tile: np.ndarray,
        row: int,
        col: int,
        orig_row: int,
        orig_col: int,
    ) -> None:
        self.tile = tile
        self.row = row        # tile row index
        self.col = col        # tile col index
        self.orig_row = orig_row  # pixel row offset in padded image
        self.orig_col = orig_col  # pixel col offset in padded image
        self.tile_h = tile.shape[0]
        self.tile_w = tile.shape[1]

    def __iter__(self):
        """Allow unpacking: tile, row, col = tile_record"""
        return iter((self.tile, self.row, self.col))


class TileSplitter:
    """
    Splits large images into overlapping tiles.

    Args:
        tile_size:  Spatial size of each tile (height == width)
        overlap:    Overlap in pixels between adjacent tiles
        pad_mode:   Padding mode for edge tiles ('reflect' | 'constant')
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 32,
        pad_mode: str = "reflect",
    ) -> None:
        if overlap >= tile_size // 2:
            raise ValueError(f"overlap ({overlap}) must be < tile_size/2 ({tile_size // 2})")
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.pad_mode = pad_mode

    def _pad_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Pad image so all tiles fit without cutoff.

        Returns:
            padded:          Padded image array
            (pad_h, pad_w):  Padding amounts on bottom and right
        """
        h, w = image.shape[:2]
        # How much we need to add so the last tile is fully covered
        pad_h = (self.stride - (h - self.tile_size) % self.stride) % self.stride
        pad_w = (self.stride - (w - self.tile_size) % self.stride) % self.stride

        if h < self.tile_size:
            pad_h = self.tile_size - h
        if w < self.tile_size:
            pad_w = self.tile_size - w

        if pad_h > 0 or pad_w > 0:
            pad_args = ((0, pad_h), (0, pad_w))
            if image.ndim == 3:
                pad_args = pad_args + ((0, 0),)
            image = np.pad(image, pad_args, mode=self.pad_mode)

        return image, (pad_h, pad_w)

    def split(self, image: np.ndarray) -> list[TileRecord]:
        """
        Split image into overlapping tiles.

        Args:
            image: HWC or HW numpy array

        Returns:
            List of TileRecord objects (iterable as (tile, row, col))
        """
        padded, _ = self._pad_image(image)
        h, w = padded.shape[:2]
        tiles: list[TileRecord] = []

        row_idx = 0
        for r in range(0, h - self.tile_size + 1, self.stride):
            col_idx = 0
            for c in range(0, w - self.tile_size + 1, self.stride):
                tile = padded[r : r + self.tile_size, c : c + self.tile_size]
                tiles.append(TileRecord(tile, row_idx, col_idx, r, c))
                col_idx += 1
            row_idx += 1

        return tiles

    def split_pair(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
    ) -> list[tuple[TileRecord, TileRecord]]:
        """
        Split two co-registered images into matching tile pairs.
        Both images must have the same spatial dimensions.
        """
        assert image_a.shape[:2] == image_b.shape[:2], (
            f"Images must have same spatial dimensions: "
            f"{image_a.shape[:2]} vs {image_b.shape[:2]}"
        )
        tiles_a = self.split(image_a)
        tiles_b = self.split(image_b)
        return list(zip(tiles_a, tiles_b))

    def get_grid_shape(self, image: np.ndarray) -> tuple[int, int]:
        """Return (n_rows, n_cols) tiles for given image shape."""
        padded, _ = self._pad_image(image)
        h, w = padded.shape[:2]
        n_rows = (h - self.tile_size) // self.stride + 1
        n_cols = (w - self.tile_size) // self.stride + 1
        return n_rows, n_cols
