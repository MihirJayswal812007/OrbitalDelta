"""
Homography estimation using RANSAC for robust satellite image registration.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def estimate_homography(
    pts_a: list[tuple[float, float]],
    pts_b: list[tuple[float, float]],
    ransac_reproj_threshold: float = 3.0,
    confidence: float = 0.995,
    max_iters: int = 2000,
) -> tuple[np.ndarray | None, np.ndarray, float, int]:
    """
    Estimate the homography that maps pts_b → pts_a using RANSAC.

    Args:
        pts_a:                    Source points (from image A)
        pts_b:                    Target points (from image B, to be aligned)
        ransac_reproj_threshold:  Max reprojection error for RANSAC inliers (pixels)
        confidence:               RANSAC confidence level
        max_iters:                Maximum RANSAC iterations

    Returns:
        H:          3×3 homography matrix (None if estimation fails)
        mask:       Boolean mask of inliers (1=inlier, 0=outlier)
        error_px:   Mean reprojection error over inliers (pixels)
        n_inliers:  Number of RANSAC inliers
    """
    pts_a_np = np.float32(pts_a).reshape(-1, 1, 2)
    pts_b_np = np.float32(pts_b).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        pts_b_np,  # source: image B points
        pts_a_np,  # dest:   image A points
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=confidence,
        maxIters=max_iters,
    )

    if H is None or mask is None:
        logger.warning("RANSAC failed to find a valid homography")
        return None, np.array([]), float("inf"), 0

    mask = mask.ravel().astype(bool)
    n_inliers = int(mask.sum())

    # Compute mean reprojection error for inliers
    error_px = _reprojection_error(pts_a_np, pts_b_np, H, mask)

    logger.debug(
        f"Homography: {n_inliers}/{len(mask)} inliers, "
        f"reprojection error = {error_px:.2f}px"
    )

    return H, mask, error_px, n_inliers


def _reprojection_error(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute mean reprojection error for inlier points.
    Error = ||H * pts_b - pts_a|| in pixels.
    """
    if mask.sum() == 0:
        return float("inf")

    inlier_b = pts_b[mask]   # (N, 1, 2)
    inlier_a = pts_a[mask]   # (N, 1, 2)

    # Project B points using H
    projected = cv2.perspectiveTransform(inlier_b, H)  # (N, 1, 2)

    # Compute Euclidean distance
    diff = projected - inlier_a
    errors = np.sqrt((diff ** 2).sum(axis=2)).ravel()
    return float(errors.mean())


def is_homography_valid(
    H: np.ndarray | None,
    error_px: float,
    n_inliers: int,
    max_error: float = 5.0,
    min_inliers: int = 10,
) -> bool:
    """
    Validate that a homography is acceptable for registration.

    Args:
        H:           Estimated homography matrix
        error_px:    Mean reprojection error
        n_inliers:   Number of RANSAC inliers
        max_error:   Maximum acceptable error (pixels)
        min_inliers: Minimum acceptable inliers

    Returns:
        True if homography is acceptable, False to reject the image pair
    """
    if H is None:
        return False
    if error_px > max_error:
        logger.warning(f"Homography rejected: error {error_px:.2f}px > {max_error}px")
        return False
    if n_inliers < min_inliers:
        logger.warning(f"Homography rejected: {n_inliers} inliers < {min_inliers}")
        return False

    # Check for degenerate homography (extreme skew / near-singular)
    det = np.linalg.det(H)
    if abs(det) < 0.1 or abs(det) > 10.0:
        logger.warning(f"Homography rejected: degenerate matrix (det={det:.3f})")
        return False

    return True
