"""
Padding utilities for tiling engine.

Computes how much to pad an image so that a given tile_size and stride
cover the entire image without leftover pixels, and provides apply/strip
helpers so the rest of the pipeline never has to think about padding math.
"""

from __future__ import annotations

import numpy as np


def compute_pad(
    h: int,
    w: int,
    tile_size: int,
    stride: int,
) -> tuple[int, int]:
    """
    Compute the minimum bottom/right padding required.

    After padding, every tile of `tile_size` extracted at `stride` intervals
    will be fully within the padded image boundaries.

    Parameters
    ----------
    h, w       : Original image height and width
    tile_size  : Spatial size of each tile
    stride     : Step between adjacent tiles (tile_size - overlap)

    Returns
    -------
    (pad_h, pad_w) : Padding amounts (non-negative integers)
    """
    # If image is smaller than tile_size, pad to exactly tile_size
    if h < tile_size:
        pad_h = tile_size - h
    else:
        remainder_h = (h - tile_size) % stride
        pad_h = (stride - remainder_h) % stride

    if w < tile_size:
        pad_w = tile_size - w
    else:
        remainder_w = (w - tile_size) % stride
        pad_w = (stride - remainder_w) % stride

    return int(pad_h), int(pad_w)


def apply_padding(
    image: np.ndarray,
    pad_h: int,
    pad_w: int,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Pad image on the bottom and right edges.

    Parameters
    ----------
    image   : HW or HWC numpy array
    pad_h   : Pixels to add to the bottom
    pad_w   : Pixels to add to the right
    mode    : Numpy pad mode ('reflect', 'constant', 'edge')

    Returns
    -------
    Padded array (same dtype as input)
    """
    if pad_h == 0 and pad_w == 0:
        return image

    pad_config: tuple
    if image.ndim == 2:
        pad_config = ((0, pad_h), (0, pad_w))
    elif image.ndim == 3:
        pad_config = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        raise ValueError(f"Expected 2-D or 3-D image, got ndim={image.ndim}")

    return np.pad(image, pad_config, mode=mode)


def strip_padding(
    image: np.ndarray,
    original_h: int,
    original_w: int,
) -> np.ndarray:
    """
    Remove padding from a processed image, restoring it to original spatial dimensions.

    Parameters
    ----------
    image      : HW or HWC numpy array (possibly padded)
    original_h : Target height
    original_w : Target width

    Returns
    -------
    Array cropped to (original_h, original_w[, C])
    """
    return image[:original_h, :original_w]
