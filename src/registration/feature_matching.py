"""
Feature detection and matching for satellite image registration.

Uses ORB (fast, free) with BFMatcher + Lowe's ratio test.
Falls back to SIFT if ORB produces insufficient keypoints.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of valid matches required for homography estimation
MIN_MATCHES = 10
# Lowe's ratio test threshold
RATIO_THRESH = 0.75


def detect_and_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
    detector: str = "orb",
    max_features: int = 2000,
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """
    Detect feature keypoints in both images and find good matches.

    Args:
        img_a:        HWC uint8 source image (time-1)
        img_b:        HWC uint8 target image (time-2)
        detector:     'orb' | 'sift' | 'akaze'
        max_features: Maximum number of features to detect

    Returns:
        good_kp_a:  List of matched keypoints from image A
        good_kp_b:  List of matched keypoints from image B
        desc_a:     All descriptors from A (for debugging)
        desc_b:     All descriptors from B

    Raises:
        RuntimeError: If insufficient matches found
    """
    # Convert to grayscale for detection
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY) if img_b.ndim == 3 else img_b

    # Create detector
    det = _create_detector(detector, max_features)

    # Detect keypoints and compute descriptors
    kp_a, desc_a = det.detectAndCompute(gray_a, None)
    kp_b, desc_b = det.detectAndCompute(gray_b, None)

    logger.debug(f"Detected {len(kp_a)} / {len(kp_b)} keypoints in A / B")

    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        if detector != "sift":
            logger.warning(f"Too few keypoints with {detector}, retrying with SIFT")
            return detect_and_match(img_a, img_b, detector="sift", max_features=max_features)
        raise RuntimeError(
            f"Too few keypoints: A={len(kp_a)}, B={len(kp_b)}. "
            "Images may be too uniform or too noisy."
        )

    # Match using BFMatcher
    norm = cv2.NORM_HAMMING if detector in ("orb", "akaze") else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)

    try:
        raw_matches = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error as e:
        raise RuntimeError(f"BFMatcher failed: {e}") from e

    # Lowe's ratio test
    good_matches = [
        m for m, n in raw_matches if m.distance < RATIO_THRESH * n.distance
    ]

    if len(good_matches) < MIN_MATCHES:
        if detector != "sift":
            logger.warning(
                f"Only {len(good_matches)} good matches with {detector}, retrying SIFT"
            )
            return detect_and_match(img_a, img_b, detector="sift", max_features=max_features)
        raise RuntimeError(
            f"Insufficient good matches: {len(good_matches)} < {MIN_MATCHES}. "
            "Image pair may be too dissimilar for registration."
        )

    # Extract matched point coordinates
    good_kp_a = [kp_a[m.queryIdx].pt for m in good_matches]
    good_kp_b = [kp_b[m.trainIdx].pt for m in good_matches]

    logger.debug(f"Good matches after ratio test: {len(good_matches)}")
    return good_kp_a, good_kp_b, desc_a, desc_b


def _create_detector(name: str, max_features: int):
    """Create feature detector by name."""
    name = name.lower()
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
    elif name == "akaze":
        return cv2.AKAZE_create()
    elif name == "sift":
        try:
            return cv2.SIFT_create(nfeatures=max_features)
        except AttributeError:
            # Older OpenCV might call it differently
            return cv2.xfeatures2d.SIFT_create(nfeatures=max_features)
    else:
        raise ValueError(f"Unknown detector: {name!r}. Choose 'orb', 'sift', or 'akaze'.")
