"""
U-Net style decoder with skip connections for change detection.
Takes differenced feature maps from the Siamese encoder and produces
a single-channel sigmoid change probability map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """3×3 Conv → BN → ReLU block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two ConvBnRelu blocks with optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """
    U-Net decoder accepting 4-level feature maps from a Siamese encoder.

    For each encoder level, the input features are:
        concat(diff_features, skip_a, skip_b)
    where diff_features = |F_A - F_B|

    Args:
        encoder_channels: List of [ch_l1, ch_l2, ch_l3, ch_l4] from encoder.
        dropout:          Dropout probability in decoder blocks.
    """

    def __init__(
        self,
        encoder_channels: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        # Each decoder block receives: diff + skip_a + skip_b = 3× channel
        # Then upsamples to previous level
        ch = encoder_channels
        # Decoder from bottom (l4) upward
        # Input to each block: 3*ch[i] (diff + skip_a + skip_b from level i)
        # concatenated with upsampled output from previous block
        self.dec4 = DoubleConv(ch[3] * 3, ch[2], dropout)          # 512*3 → 256
        self.dec3 = DoubleConv(ch[2] * 3 + ch[2], ch[1], dropout)  # 256*3 + 256 → 128
        self.dec2 = DoubleConv(ch[1] * 3 + ch[1], ch[0], dropout)  # 128*3 + 128 → 64
        self.dec1 = DoubleConv(ch[0] * 3 + ch[0], ch[0], dropout)  # 64*3 + 64 → 64

        # Final classification head
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _diff_concat(
        self,
        fa: torch.Tensor,
        fb: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate [|fa-fb|, fa, fb] along channel dim."""
        return torch.cat([torch.abs(fa - fb), fa, fb], dim=1)

    def forward(
        self,
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            feats_a: List of 4 tensors from encoder forward(img_a)
            feats_b: List of 4 tensors from encoder forward(img_b)

        Returns:
            Change probability map (B, 1, H, W) in [0, 1]
        """
        f1a, f2a, f3a, f4a = feats_a
        f1b, f2b, f3b, f4b = feats_b

        # Bottom of U (deepest level)
        d4 = self.dec4(self._diff_concat(f4a, f4b))   # (B, 256, H/32, W/32)
        d4_up = self.up(d4)                            # (B, 256, H/16, W/16)

        # Align spatial dims (handles odd sizes)
        d4_up = F.interpolate(d4_up, size=f3a.shape[2:], mode="bilinear", align_corners=False)
        d3_in = torch.cat([self._diff_concat(f3a, f3b), d4_up], dim=1)
        d3 = self.dec3(d3_in)                          # (B, 128, H/16, W/16)
        d3_up = self.up(d3)

        d3_up = F.interpolate(d3_up, size=f2a.shape[2:], mode="bilinear", align_corners=False)
        d2_in = torch.cat([self._diff_concat(f2a, f2b), d3_up], dim=1)
        d2 = self.dec2(d2_in)                          # (B, 64, H/8, W/8)
        d2_up = self.up(d2)

        d2_up = F.interpolate(d2_up, size=f1a.shape[2:], mode="bilinear", align_corners=False)
        d1_in = torch.cat([self._diff_concat(f1a, f1b), d2_up], dim=1)
        d1 = self.dec1(d1_in)                          # (B, 64, H/4, W/4)

        # Upsample to original image size
        d1_up = F.interpolate(d1, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(d1_up)                        # (B, 1, H, W)
