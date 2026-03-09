"""
Unit tests for Phase 3 training components:
- ChangeDetectionMetrics
- EarlyStopping
- CheckpointManager
- Trainer (2-epoch smoke test on synthetic data)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import CDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.losses import BCEDiceLoss
from src.models.siamese_unet import SiameseUNet
from src.training.trainer import CheckpointManager, EarlyStopping, Trainer
from src.utils.metrics import ChangeDetectionMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_dataset_dir():
    """Tiny synthetic change detection dataset on disk."""
    tmpdir = Path(tempfile.mkdtemp())
    for split in ["train", "val"]:
        for sub in ["A", "B", "label"]:
            (tmpdir / split / sub).mkdir(parents=True)
        for i in range(8):
            img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            lbl = (np.random.rand(300, 300) > 0.8).astype(np.uint8) * 255
            Image.fromarray(img).save(tmpdir / split / "A" / f"img_{i:04d}.png")
            Image.fromarray(img).save(tmpdir / split / "B" / f"img_{i:04d}.png")
            Image.fromarray(lbl, mode="L").save(
                tmpdir / split / "label" / f"img_{i:04d}.png"
            )
    return tmpdir


@pytest.fixture(scope="module")
def smoke_config(tmp_path_factory, synthetic_dataset_dir):
    """Minimal config for 2-epoch smoke tests."""
    base = tmp_path_factory.mktemp("smoke")
    return {
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_T0": 2,
            "early_stopping_patience": 5,
            "mixed_precision": False,
            "gradient_clip_norm": 1.0,
            "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
        },
        "data": {
            "root": str(synthetic_dataset_dir),
            "crop_size": 64,
            "num_workers": 0,
        },
        "model": {"encoder": "resnet18", "pretrained": False},
        "logging": {
            "backend": "tensorboard",
            "log_dir": str(base / "logs"),
            "log_every_n_steps": 1,
            "save_top_k": 2,
        },
        "checkpoint_dir": str(base / "checkpoints"),
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# ChangeDetectionMetrics tests
# ---------------------------------------------------------------------------


class TestChangeDetectionMetrics:
    def test_perfect_prediction_f1_is_one(self):
        m = ChangeDetectionMetrics()
        pred = torch.ones(4)
        target = torch.ones(4)
        m.update(pred, target)
        r = m.compute()
        assert abs(r["f1"] - 1.0) < 1e-4

    def test_all_wrong_f1_is_zero(self):
        m = ChangeDetectionMetrics()
        pred = torch.zeros(4)
        target = torch.ones(4)
        m.update(pred, target)
        r = m.compute()
        assert r["f1"] == 0.0

    def test_iou_range(self):
        m = ChangeDetectionMetrics()
        pred = torch.sigmoid(torch.randn(64))
        target = (torch.rand(64) > 0.7).long()
        m.update(pred, target)
        r = m.compute()
        assert 0.0 <= r["iou"] <= 1.0

    def test_batched_accumulation_consistency(self):
        """Two half-calls should match one full call."""
        pred = torch.tensor([0.9, 0.9, 0.1, 0.1])
        target = torch.tensor([1, 1, 0, 0])

        # Full call
        m_full = ChangeDetectionMetrics()
        m_full.update(pred, target)
        r_full = m_full.compute()

        # Split into two calls
        m_split = ChangeDetectionMetrics()
        m_split.update(pred[:2], target[:2])
        m_split.update(pred[2:], target[2:])
        r_split = m_split.compute()

        assert abs(r_full["f1"] - r_split["f1"]) < 1e-4

    def test_reset_clears_state(self):
        m = ChangeDetectionMetrics()
        pred = torch.ones(4)
        target = torch.ones(4)
        m.update(pred, target)
        m.reset()
        # After reset, torchmetrics raises on compute with no data
        # (or returns 0) — simply verify we can update again cleanly
        m.update(pred, target)
        r = m.compute()
        assert 0.0 <= r["f1"] <= 1.0

    def test_4d_input(self):
        """Real inference produces (B, 1, H, W) tensors — must not crash."""
        m = ChangeDetectionMetrics()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = (torch.rand(2, 1, 64, 64) > 0.7).long()
        m.update(pred, target)
        r = m.compute()
        assert "f1" in r and "iou" in r

    def test_kappa_range(self):
        m = ChangeDetectionMetrics()
        pred = torch.sigmoid(torch.randn(100))
        target = (torch.rand(100) > 0.8).long()
        m.update(pred, target)
        r = m.compute()
        assert -1.0 <= r["kappa"] <= 1.0


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3)
        state = {}
        for score in [0.5, 0.6, 0.7, 0.8]:
            stopped = es(score, state)
        assert not stopped

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3)
        state = {}
        es(0.8, state)  # best
        for _ in range(3):
            stopped = es(0.7, state)  # no improvement
        assert stopped

    def test_best_score_tracked(self):
        es = EarlyStopping(patience=5)
        state = {}
        for score in [0.5, 0.8, 0.7, 0.9]:
            es(score, state)
        assert abs(es.best_score - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def test_saves_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpts", top_k=3)
        state = {"epoch": 1, "model_state": {}}
        mgr.save(state, 0.85, epoch=1)
        assert len(list((tmp_path / "ckpts").glob("*.pt"))) == 1

    def test_saves_best_pt(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpts", top_k=3)
        state = {"epoch": 1}
        mgr.save(state, 0.90, epoch=1, is_best=True)
        assert (tmp_path / "ckpts" / "best.pt").exists()

    def test_prunes_beyond_top_k(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpts", top_k=2)
        for i, score in enumerate([0.5, 0.6, 0.7, 0.8]):
            mgr.save({"epoch": i}, score, epoch=i)
        # Should keep only top-2 regular checkpoints
        ckpts = [p for p in (tmp_path / "ckpts").glob("epoch_*.pt")]
        assert len(ckpts) <= 2


# ---------------------------------------------------------------------------
# Trainer smoke test
# ---------------------------------------------------------------------------


class TestTrainer:
    """Each test is fully self-contained — creates its own config + temp dirs."""

    def _build_dataloaders(self, synthetic_dataset_dir, crop_size=64):
        train_ds = CDDataset(
            synthetic_dataset_dir / "train", "train", get_train_transforms(crop_size)
        )
        val_ds = CDDataset(
            synthetic_dataset_dir / "val", "val", get_val_transforms(crop_size)
        )
        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
        return train_dl, val_dl

    def _make_config(self, base: Path, synthetic_dataset_dir: Path) -> dict:
        return {
            "training": {
                "epochs": 2,
                "batch_size": 2,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "scheduler_T0": 2,
                "early_stopping_patience": 5,
                "mixed_precision": False,
                "gradient_clip_norm": 1.0,
                "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
            },
            "data": {
                "root": str(synthetic_dataset_dir),
                "crop_size": 64,
                "num_workers": 0,
            },
            "model": {"encoder": "resnet18", "pretrained": False},
            "logging": {
                "backend": "tensorboard",
                "log_dir": str(base / "logs"),
                "log_every_n_steps": 1,
                "save_top_k": 2,
            },
            "checkpoint_dir": str(base / "checkpoints"),
            "seed": 42,
        }

    def test_two_epoch_smoke(self, tmp_path, synthetic_dataset_dir):
        config = self._make_config(tmp_path, synthetic_dataset_dir)
        train_dl, val_dl = self._build_dataloaders(synthetic_dataset_dir)
        model = SiameseUNet("resnet18", pretrained=False)
        trainer = Trainer(model, train_dl, val_dl, BCEDiceLoss(), config)
        history = trainer.train()

        assert len(history["train_loss"]) == 2
        assert len(history["val_f1"]) == 2
        assert all(not torch.isnan(torch.tensor(v)) for v in history["val_f1"])
        assert all(not torch.isnan(torch.tensor(v)) for v in history["train_loss"])

    def test_checkpoints_created(self, tmp_path, synthetic_dataset_dir):
        config = self._make_config(tmp_path, synthetic_dataset_dir)
        train_dl, val_dl = self._build_dataloaders(synthetic_dataset_dir)
        model = SiameseUNet("resnet18", pretrained=False)
        trainer = Trainer(model, train_dl, val_dl, BCEDiceLoss(), config)
        trainer.train()

        ckpt_dir = Path(config["checkpoint_dir"])
        ckpts = list(ckpt_dir.glob("*.pt"))
        assert len(ckpts) >= 1, f"No checkpoints in {ckpt_dir}"

    def test_history_keys(self, tmp_path, synthetic_dataset_dir):
        config = self._make_config(tmp_path, synthetic_dataset_dir)
        config["training"]["epochs"] = 1
        train_dl, val_dl = self._build_dataloaders(synthetic_dataset_dir)
        model = SiameseUNet("resnet18", pretrained=False)
        trainer = Trainer(model, train_dl, val_dl, BCEDiceLoss(), config)
        history = trainer.train()

        for key in ["train_loss", "val_loss", "val_f1", "val_iou", "lr"]:
            assert key in history, f"Missing key: {key}"
        assert len(history["train_loss"]) == 1

