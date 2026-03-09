"""
Paired augmentation transforms for change detection.
All transforms are applied IDENTICALLY to image A, image B, and the mask
using Albumentations' additional_targets mechanism.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _paired_targets() -> dict[str, str]:
    """Define additional_targets so B and mask get same transform as A."""
    return {"image_b": "image", "mask": "mask"}


def get_train_transforms(crop_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline.
    Geometric transforms applied identically to A, B, and mask.
    Color transforms applied independently to A and B (realistic).
    """
    return A.Compose(
        [
            # --- Spatial transforms (same for A, B, mask) ---
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.3,
            ),
            # --- Color transforms (applied to images only, not mask) ---
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # --- Normalization + Tensor ---
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets=_paired_targets(),
    )


def get_val_transforms(crop_size: int = 256) -> A.Compose:
    """
    Validation / test transforms — normalize only, no augmentation.
    Crop ensures consistent size if images are larger than crop_size.
    """
    return A.Compose(
        [
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets=_paired_targets(),
    )


def get_inference_transforms() -> A.Compose:
    """
    Inference transforms — normalize only, no crop (tiles are already correct size).
    """
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets=_paired_targets(),
    )
