"""
Siamese U-Net for satellite image change detection.
Uses a single shared-weight encoder for both time points.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.decoders import UNetDecoder
from src.models.encoders import SiameseEncoder


class SiameseUNet(nn.Module):
    """
    Full Siamese U-Net change detection model.

    Architecture:
        1. Shared ResNet encoder (same weights used for both images)
        2. Feature differencing at each scale: concat(|FA-FB|, FA, FB)
        3. U-Net decoder with skip connections
        4. Sigmoid output: change probability map

    Args:
        encoder_name: "resnet18" | "resnet34" | "resnet50"
        pretrained:   Load ImageNet pretrained encoder weights
        dropout:      Dropout probability in decoder blocks
    """

    def __init__(
        self,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name

        # SINGLE encoder — shared weights for both input images
        self.encoder = SiameseEncoder(encoder_name, pretrained=pretrained)

        # Decoder
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            dropout=dropout,
        )

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            img_a: Time-1 image tensor (B, 3, H, W)
            img_b: Time-2 image tensor (B, 3, H, W)

        Returns:
            Change probability map (B, 1, H, W) in [0, 1]
        """
        # Both images pass through the SAME encoder (shared weights)
        feats_a = self.encoder(img_a)
        feats_b = self.encoder(img_b)

        # Decoder computes difference and produces change map
        return self.decoder(feats_a, feats_b)

    def count_parameters(self) -> dict[str, int]:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "total_M": total // 1_000_000}
