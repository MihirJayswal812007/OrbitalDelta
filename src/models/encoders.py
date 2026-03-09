"""
Siamese encoder using a shared-weight ResNet backbone.
Returns multi-scale feature maps for U-Net skip connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SiameseEncoder(nn.Module):
    """
    Shared-weight ResNet encoder for Siamese change detection.

    Strips the FC head and avgpool from a pretrained ResNet and exposes
    feature maps at 4 scales for U-Net skip connections.

    Feature map sizes for 256×256 input (stride-2 at each level):
        Level 0: (B, 64,  64, 64)   — after layer1
        Level 1: (B, 128, 32, 32)   — after layer2
        Level 2: (B, 256, 16, 16)   — after layer3
        Level 3: (B, 512,  8,  8)   — after layer4

    Args:
        encoder_name: One of "resnet18", "resnet34", "resnet50"
        pretrained:   Use ImageNet pretrained weights
    """

    # Output channels per backbone
    CHANNELS: dict[str, list[int]] = {
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
    }

    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if encoder_name not in self.CHANNELS:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Choose from: {list(self.CHANNELS.keys())}"
            )
        self.encoder_name = encoder_name

        # Load backbone
        weights_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        }
        weights = weights_map[encoder_name] if pretrained else None
        backbone_fn = getattr(models, encoder_name)
        backbone = backbone_fn(weights=weights)

        # Split into sequential stages
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @property
    def out_channels(self) -> list[int]:
        return self.CHANNELS[self.encoder_name]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            List of 4 feature tensors at successive scales.
        """
        x = self.stem(x)       # (B, 64, H/4, W/4)
        f1 = self.layer1(x)    # (B, 64,  H/4,  W/4)
        f2 = self.layer2(f1)   # (B, 128, H/8,  W/8)
        f3 = self.layer3(f2)   # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)   # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4]
