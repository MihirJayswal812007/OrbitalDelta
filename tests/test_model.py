"""
Unit tests for model architecture: encoder, decoder, full model, loss.
"""

import pytest
import torch

from src.models.decoders import UNetDecoder
from src.models.encoders import SiameseEncoder
from src.models.losses import BCEDiceLoss, DiceLoss, FocalDiceLoss
from src.models.siamese_unet import SiameseUNet


class TestEncoder:
    def test_output_length(self):
        enc = SiameseEncoder("resnet18", pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        feats = enc(x)
        assert len(feats) == 4

    def test_output_channels_resnet18(self):
        enc = SiameseEncoder("resnet18", pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        feats = enc(x)
        expected_ch = [64, 128, 256, 512]
        for f, ch in zip(feats, expected_ch):
            assert f.shape[1] == ch, f"Expected {ch} channels, got {f.shape[1]}"

    def test_output_channels_resnet34(self):
        enc = SiameseEncoder("resnet34", pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        feats = enc(x)
        expected_ch = [64, 128, 256, 512]
        for f, ch in zip(feats, expected_ch):
            assert f.shape[1] == ch

    def test_spatial_downsampling(self):
        enc = SiameseEncoder("resnet18", pretrained=False)
        x = torch.randn(1, 3, 256, 256)
        feats = enc(x)
        # Each level should be smaller than the previous
        prev_size = 256
        for f in feats:
            assert f.shape[2] < prev_size or prev_size == 64
            prev_size = f.shape[2]

    def test_invalid_encoder_name(self):
        with pytest.raises(ValueError, match="Unknown encoder"):
            SiameseEncoder("resnet99")


class TestDecoder:
    def _make_feats(self, channels):
        sizes = [64, 32, 16, 8]
        return [torch.randn(2, c, s, s) for c, s in zip(channels, sizes)]

    def test_output_shape(self):
        dec = UNetDecoder(encoder_channels=[64, 128, 256, 512])
        feats_a = self._make_feats([64, 128, 256, 512])
        feats_b = self._make_feats([64, 128, 256, 512])
        out = dec(feats_a, feats_b)
        assert out.shape == (2, 1, 256, 256)

    def test_output_range(self):
        dec = UNetDecoder(encoder_channels=[64, 128, 256, 512])
        feats_a = self._make_feats([64, 128, 256, 512])
        feats_b = self._make_feats([64, 128, 256, 512])
        out = dec(feats_a, feats_b)
        assert 0 <= out.min().item(), "Output below 0 (sigmoid missing)"
        assert out.max().item() <= 1.0, "Output above 1 (sigmoid missing)"


class TestSiameseUNet:
    def test_forward_shape(self):
        model = SiameseUNet("resnet18", pretrained=False)
        a, b = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
        out = model(a, b)
        assert out.shape == (1, 1, 256, 256)

    def test_output_is_probability(self):
        model = SiameseUNet("resnet18", pretrained=False)
        a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        out = model(a, b)
        assert 0 <= out.min().item()
        assert out.max().item() <= 1.0

    def test_weight_sharing(self):
        """Verify both images use the same encoder (no encoder_b)."""
        model = SiameseUNet("resnet18", pretrained=False)
        assert not hasattr(model, "encoder_b"), "encoder_b should not exist"
        assert hasattr(model, "encoder"), "encoder must exist"

    def test_weight_sharing_by_param_count(self):
        """Siamese should have same params as single ResNet18 + decoder."""
        model = SiameseUNet("resnet18", pretrained=False)
        p = model.count_parameters()
        # ResNet-18 has ~11M params; adding decoder should stay under 20M
        assert p["total_M"] < 20, f"Too many params: {p['total_M']}M (likely double encoder)"

    def test_gradient_flow(self):
        model = SiameseUNet("resnet18", pretrained=False)
        loss_fn = BCEDiceLoss()
        a, b = torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)
        target = (torch.rand(1, 1, 64, 64) > 0.8).float()
        out = model(a, b)
        loss = loss_fn(out, target)
        loss.backward()
        # All encoder and decoder params should have gradients
        no_grad = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not no_grad, f"Params with no gradient: {no_grad[:5]}"

    def test_batch_size_invariance(self):
        model = SiameseUNet("resnet18", pretrained=False)
        for bs in [1, 2]:
            a = torch.randn(bs, 3, 64, 64)
            b = torch.randn(bs, 3, 64, 64)
            out = model(a, b)
            assert out.shape[0] == bs


class TestLoss:
    def test_bce_dice_positive(self):
        loss_fn = BCEDiceLoss()
        pred = torch.sigmoid(torch.randn(1, 1, 64, 64))
        target = (torch.rand(1, 1, 64, 64) > 0.8).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_bce_dice_no_nan(self):
        loss_fn = BCEDiceLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = (torch.rand(2, 1, 64, 64) > 0.8).float()
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_perfect_prediction(self):
        loss_fn = BCEDiceLoss()
        target = (torch.rand(1, 1, 64, 64) > 0.5).float()
        # Near-perfect prediction
        pred = target * 0.99 + (1 - target) * 0.01
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1, f"Perfect pred should give low loss, got {loss.item():.4f}"

    def test_all_zeros_mask(self):
        """Loss should be finite even when no change pixels present."""
        loss_fn = BCEDiceLoss()
        pred = torch.full((1, 1, 64, 64), 0.01)
        target = torch.zeros(1, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)

    def test_all_ones_mask(self):
        """Loss should be finite when all pixels are change."""
        loss_fn = BCEDiceLoss()
        pred = torch.full((1, 1, 64, 64), 0.99)
        target = torch.ones(1, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)

    def test_weight_sum_constraint(self):
        with pytest.raises(AssertionError):
            BCEDiceLoss(bce_weight=0.6, dice_weight=0.6)

    def test_focal_dice_loss(self):
        loss_fn = FocalDiceLoss()
        pred = torch.sigmoid(torch.randn(1, 1, 64, 64))
        target = (torch.rand(1, 1, 64, 64) > 0.8).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)
