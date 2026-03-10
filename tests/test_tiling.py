"""
Phase 8 tests — Tiling Engine.

Tests: TileSplitter, TileStitcher, split→stitch roundtrip.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TileSplitter
# ---------------------------------------------------------------------------

class TestTileSplitter:
    def test_basic_split(self):
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(512, 512).astype(np.float32)
        splitter = TileSplitter(tile_size=256, overlap=0)
        tiles = splitter.split(img)
        assert len(tiles) > 0, "No tiles produced"

    def test_tiles_have_correct_size(self):
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(512, 512).astype(np.float32)
        splitter = TileSplitter(tile_size=256, overlap=0)
        tiles = splitter.split(img)
        for tile, row, col in tiles:
            assert tile.shape[0] == 256, f"Tile height {tile.shape[0]} ≠ 256"
            assert tile.shape[1] == 256, f"Tile width {tile.shape[1]} ≠ 256"

    def test_non_square_image(self):
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(384, 512).astype(np.float32)
        splitter = TileSplitter(tile_size=128, overlap=16)
        tiles = splitter.split(img)
        assert len(tiles) > 0

    def test_overlap_increases_tile_count(self):
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(512, 512).astype(np.float32)
        tiles_no_overlap = TileSplitter(tile_size=256, overlap=0).split(img)
        tiles_with_overlap = TileSplitter(tile_size=256, overlap=32).split(img)
        # Overlapping produces at least as many tiles
        assert len(tiles_with_overlap) >= len(tiles_no_overlap)

    def test_rgb_image(self):
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(512, 512, 3).astype(np.float32)
        tiles = TileSplitter(tile_size=256, overlap=0).split(img)
        assert len(tiles) > 0
        for tile, row, col in tiles:
            assert tile.ndim == 3, "RGB tiles should have 3 dims"

    def test_image_smaller_than_tile(self):
        """When image < tile_size, should return a single padded tile."""
        from src.tiling.splitter import TileSplitter

        img = np.random.rand(64, 64).astype(np.float32)
        tiles = TileSplitter(tile_size=256, overlap=0).split(img)
        assert len(tiles) == 1


# ---------------------------------------------------------------------------
# TileStitcher
# ---------------------------------------------------------------------------

class TestTileStitcher:
    def test_roundtrip_no_overlap(self):
        from src.tiling.splitter import TileSplitter
        from src.tiling.stitcher import TileStitcher

        img = np.random.rand(512, 512).astype(np.float32)
        splitter = TileSplitter(tile_size=256, overlap=0)
        tiles = splitter.split(img)

        stitcher = TileStitcher(tile_size=256, overlap=0, output_shape=img.shape)
        for tile, row, col in tiles:
            stitcher.add_tile(tile, row, col)
        result = stitcher.stitch()

        assert result.shape == img.shape, f"Shape mismatch: {result.shape} ≠ {img.shape}"
        error = np.abs(result - img).mean()
        assert error < 0.01, f"Reconstruction error too high: {error:.4f}"

    def test_roundtrip_with_overlap(self):
        from src.tiling.splitter import TileSplitter
        from src.tiling.stitcher import TileStitcher

        img = np.random.rand(1024, 1024).astype(np.float32)
        overlap = 32
        splitter = TileSplitter(tile_size=256, overlap=overlap)
        tiles = splitter.split(img)

        stitcher = TileStitcher(tile_size=256, overlap=overlap, output_shape=img.shape)
        for tile, row, col in tiles:
            stitcher.add_tile(tile, row, col)
        result = stitcher.stitch()

        assert result.shape == img.shape
        error = np.abs(result - img).mean()
        assert error < 0.05, f"Reconstruction error with overlap too high: {error:.4f}"

    def test_various_sizes(self):
        from src.tiling.splitter import TileSplitter
        from src.tiling.stitcher import TileStitcher

        for h, w in [(256, 256), (300, 400), (1024, 768)]:
            img = np.random.rand(h, w).astype(np.float32)
            tiles = TileSplitter(tile_size=128, overlap=16).split(img)
            stitcher = TileStitcher(tile_size=128, overlap=16, output_shape=img.shape)
            for tile, row, col in tiles:
                stitcher.add_tile(tile, row, col)
            result = stitcher.stitch()
            assert result.shape == img.shape, f"{h}×{w}: shape mismatch {result.shape}"


# ---------------------------------------------------------------------------
# padding.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# padding.py
# ---------------------------------------------------------------------------

class TestPadding:
    def test_compute_pad_basic(self):
        from src.tiling.padding import compute_pad

        # 300×400 image, tile=256, stride=256 → needs some padding
        pad_h, pad_w = compute_pad(300, 400, tile_size=256, stride=256)
        assert isinstance(pad_h, int) and pad_h >= 0
        assert isinstance(pad_w, int) and pad_w >= 0
        # After padding, (300+pad_h - 256) % stride == 0
        assert (300 + pad_h - 256) % 256 == 0 or (300 + pad_h) == 256

    def test_apply_and_strip_padding(self):
        from src.tiling.padding import compute_pad, apply_padding, strip_padding

        img = np.random.rand(300, 400).astype(np.float32)
        ph, pw = compute_pad(300, 400, tile_size=256, stride=256)
        padded = apply_padding(img, ph, pw)

        assert padded.shape[0] >= 300
        assert padded.shape[1] >= 400

        restored = strip_padding(padded, 300, 400)
        assert restored.shape == img.shape

    def test_already_aligned_no_padding(self):
        from src.tiling.padding import compute_pad

        # 512×256 is already a perfect multiple of 256
        pad_h, pad_w = compute_pad(512, 256, tile_size=256, stride=256)
        assert pad_h == 0 and pad_w == 0, (
            f"Already-aligned image should not be padded, got ({pad_h}, {pad_w})"
        )

    def test_apply_padding_smaller_than_tile(self):
        """Image smaller than tile_size should be padded up to tile_size."""
        from src.tiling.padding import compute_pad, apply_padding

        img = np.random.rand(64, 64).astype(np.float32)
        ph, pw = compute_pad(64, 64, tile_size=256, stride=256)
        padded = apply_padding(img, ph, pw)
        assert padded.shape[0] >= 256
        assert padded.shape[1] >= 256

    def test_strip_padding_rgb(self):
        from src.tiling.padding import strip_padding

        img = np.random.rand(300, 400, 3).astype(np.float32)
        # Simulate a padded result
        padded = np.pad(img, ((0, 12), (0, 112), (0, 0)), mode="reflect")
        restored = strip_padding(padded, 300, 400)
        assert restored.shape == (300, 400, 3)

