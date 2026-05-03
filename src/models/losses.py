"""
Combined BCE + Dice loss for binary change detection.
Handles class imbalance (change pixels are typically <20% of the image).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Differentiable approximation of the Dice coefficient.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Sigmoid-activated predictions (B, 1, H, W) in [0, 1]
            target: Binary ground truth (B, 1, H, W) in {0, 1}
        """
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    L = α * BCE + β * Dice

    BCE handles per-pixel classification; Dice handles class imbalance.

    Args:
        bce_weight:  Weight for BCE term (α)
        dice_weight: Weight for Dice term (β)
        smooth:      Smoothing constant for Dice denominator
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        assert abs(bce_weight + dice_weight - 1.0) < 1e-6, (
            f"bce_weight + dice_weight must equal 1.0, got {bce_weight + dice_weight}"
        )
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Sigmoid-activated predictions (B, 1, H, W) in [0, 1]
            target: Binary ground truth (B, 1, H, W) in {0, 1}

        Returns:
            Scalar loss tensor with gradient.
        """
        # BCE expects float targets in [0, 1]
        target = target.float()

        bce = F.binary_cross_entropy(pred, target, reduction="mean")
        dice = self.dice(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss for hard negative mining.
    Use when BCE+Dice isn't sufficient for very imbalanced datasets.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        # Focal loss = -(1-p_t)^gamma * log(p_t)
        # AMP-safe: convert sigmoid output back to logits, then use with_logits
        pred_logits = torch.log(pred.float().clamp(1e-6, 1 - 1e-6) / (1 - pred.float().clamp(1e-6, 1 - 1e-6)))
        bce = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        focal = bce * (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = (alpha_t * focal).mean()

        dice = self.dice(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice
