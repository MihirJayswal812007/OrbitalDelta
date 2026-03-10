"""
Image warping and full registration pipeline.

Orchestrates feature matching → homography estimation → perspective warp
into a single reusable align_images() function.
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

from src.registration.feature_matching import detect_and_match
from src.registration.homography import estimate_homography, is_homography_valid

logger = logging.getLogger(__name__)


class RegistrationError(Exception):
    """Raised when image pair cannot be registered within acceptable error."""
    pass


def align_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    detector: str = "orb",
    max_features: int = 2000,
    ransac_threshold: float = 3.0,
    max_error: float = 5.0,
    min_inliers: int = 10,
    raise_on_failure: bool = False,
) -> tuple[np.ndarray, float, int]:
    """
    Align img_b to the coordinate frame of img_a.

    Full pipeline:
        1. Detect and match feature keypoints
        2. Estimate homography with RANSAC
        3. Validate alignment quality
        4. Warp img_b using the estimated homography

    Args:
        img_a:            Reference image (HWC uint8) — stays fixed
        img_b:            Moving image (HWC uint8) — will be warped
        detector:         Feature detector: 'orb' | 'sift' | 'akaze'
        max_features:     Maximum keypoints per image
        ransac_threshold: RANSAC reprojection threshold (pixels)
        max_error:        Maximum acceptable alignment error (pixels)
        min_inliers:      Minimum acceptable RANSAC inliers
        raise_on_failure: If True, raise RegistrationError on bad alignment;
                          otherwise return unmodified img_b with error=inf

    Returns:
        aligned_b:  Warped image B (same shape as img_a)
        error_px:   Mean reprojection error (px). float('inf') if failed.
        n_inliers:  Number of RANSAC inliers. 0 if failed.
    """
    h, w = img_a.shape[:2]

    try:
        # Step 1: Feature detection + matching
        pts_a, pts_b, _, _ = detect_and_match(img_a, img_b, detector=detector, max_features=max_features)

        # Step 2: Homography estimation
        H, mask, error_px, n_inliers = estimate_homography(
            pts_a, pts_b, ransac_reproj_threshold=ransac_threshold
        )

        # Step 3: Validation
        if not is_homography_valid(H, error_px, n_inliers, max_error=max_error, min_inliers=min_inliers):
            if raise_on_failure:
                raise RegistrationError(
                    f"Registration quality unacceptable: error={error_px:.2f}px, "
                    f"inliers={n_inliers}"
                )
            logger.warning("Registration failed validation — returning unaligned image")
            return img_b.copy(), float("inf"), 0

        # Step 4: Warp img_b using the estimated homography
        aligned_b = cv2.warpPerspective(
            img_b, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        logger.info(
            f"Registration OK: error={error_px:.2f}px, inliers={n_inliers}/{len(mask)}"
        )
        return aligned_b, error_px, n_inliers

    except RuntimeError as e:
        if raise_on_failure:
            raise RegistrationError(str(e)) from e
        logger.warning(f"Registration failed: {e} — returning unaligned image")
        return img_b.copy(), float("inf"), 0


def align_image_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    config: dict | None = None,
) -> dict:
    """
    Higher-level wrapper returning a full result dict (for API/pipeline use).

    Args:
        img_a, img_b: HWC uint8 images
        config:       Optional dict with keys: detector, max_features, max_error, min_inliers

    Returns:
        {
            "aligned_b": np.ndarray,    # warped image B
            "error_px": float,          # alignment error in pixels
            "n_inliers": int,           # RANSAC inlier count
            "success": bool,            # whether registration passed validation
        }
    """
    cfg = config or {}
    aligned_b, error_px, n_inliers = align_images(
        img_a=img_a,
        img_b=img_b,
        detector=cfg.get("detector", "orb"),
        max_features=cfg.get("max_features", 2000),
        ransac_threshold=cfg.get("ransac_threshold", 3.0),
        max_error=cfg.get("max_error", 5.0),
        min_inliers=cfg.get("min_inliers", 10),
        raise_on_failure=False,
    )
    success = error_px < cfg.get("max_error", 5.0) and n_inliers >= cfg.get("min_inliers", 10)
    return {
        "aligned_b": aligned_b,
        "error_px": error_px,
        "n_inliers": n_inliers,
        "success": success,
    }
