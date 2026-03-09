"""
Gate script for Phase 3 Task 3.2 — Trainer smoke test.

Runs 2 epochs on a tiny synthetic dataset (no real data needed) to verify:
- Trainer initializes and runs
- History dict contains expected keys and correct epoch count
- Checkpoint is written
- Metrics are computed
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Create a tiny synthetic dataset on disk
tmpdir = Path(tempfile.mkdtemp())
for split in ["train", "val"]:
    for sub in ["A", "B", "label"]:
        (tmpdir / split / sub).mkdir(parents=True)
    for i in range(8):
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        lbl = (np.random.rand(300, 300) > 0.8).astype(np.uint8) * 255
        Image.fromarray(img).save(tmpdir / split / "A" / f"img_{i:04d}.png")
        Image.fromarray(img).save(tmpdir / split / "B" / f"img_{i:04d}.png")
        Image.fromarray(lbl, mode="L").save(tmpdir / split / "label" / f"img_{i:04d}.png")

from src.data.dataset import CDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.siamese_unet import SiameseUNet
from src.models.losses import BCEDiceLoss
from src.training.trainer import Trainer

config = {
    "training": {
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "scheduler_T0": 2,
        "early_stopping_patience": 5,
        "mixed_precision": False,  # CPU: no AMP
        "gradient_clip_norm": 1.0,
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
    },
    "data": {"root": str(tmpdir), "crop_size": 64, "num_workers": 0},
    "model": {"encoder": "resnet18", "pretrained": False},
    "logging": {
        "backend": "tensorboard",
        "log_dir": str(tmpdir / "logs"),
        "log_every_n_steps": 1,
        "save_top_k": 2,
    },
    "checkpoint_dir": str(tmpdir / "checkpoints"),
    "seed": 42,
}

train_ds = CDDataset(tmpdir / "train", "train", get_train_transforms(64))
val_ds = CDDataset(tmpdir / "val", "val", get_val_transforms(64))
train_dl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

model = SiameseUNet("resnet18", pretrained=False)
loss_fn = BCEDiceLoss()
trainer = Trainer(model, train_dl, val_dl, loss_fn, config)
history = trainer.train()

# Structural checks
assert "train_loss" in history, "Missing train_loss in history"
assert "val_f1" in history, "Missing val_f1 in history"
assert "val_iou" in history, "Missing val_iou in history"
assert "lr" in history, "Missing lr in history"
assert len(history["train_loss"]) == 2, f"Expected 2 epochs, got {len(history['train_loss'])}"
assert all(not torch.isnan(torch.tensor(v)) for v in history["val_f1"]), "NaN in val_f1"

# Checkpoint must exist
ckpt_dir = Path(config["checkpoint_dir"])
ckpts = list(ckpt_dir.glob("*.pt"))
assert len(ckpts) >= 1, f"No checkpoints saved! dir: {list(ckpt_dir.glob('*'))}"

print("===================================")
print("  Task 3.2: Trainer smoke test")
print("===================================")
print(f"  Epochs: {len(history['train_loss'])}")
print(f"  train_loss: {history['train_loss']}")
print(f"  val_f1:     {history['val_f1']}")
print(f"  val_iou:    {history['val_iou']}")
print(f"  Checkpoints saved: {len(ckpts)} files")
print("  STATUS: TRAINER OK")
print("===================================")
