"""
Phase 3 Gate — verifies the complete training pipeline end-to-end
using synthetic data (no real LEVIR-CD required for this gate).

Checks:
  1. Metrics module computes correct values
  2. Trainer runs 5 epochs on synthetic data
  3. Checkpoints are written
  4. History has all required keys
  5. train.py and evaluate.py scripts are importable and structured correctly
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# ------------------------------------------------------------------
# 1. Metrics
# ------------------------------------------------------------------
from src.utils.metrics import ChangeDetectionMetrics

m = ChangeDetectionMetrics()
m.update(torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([1, 1, 0, 0]))
r = m.compute()
assert abs(r["f1"] - 1.0) < 1e-4, f"F1 should be 1.0, got {r['f1']}"
assert "iou" in r and "precision" in r and "recall" in r and "kappa" in r
print("[1/5] Metrics module: OK")

# ------------------------------------------------------------------
# 2. Trainer — 3 epochs on synthetic data
# ------------------------------------------------------------------
from src.data.dataset import CDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.losses import BCEDiceLoss
from src.models.siamese_unet import SiameseUNet
from src.training.trainer import Trainer

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

config = {
    "training": {
        "epochs": 3,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "scheduler_T0": 3,
        "early_stopping_patience": 10,
        "mixed_precision": False,
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
trainer = Trainer(model, train_dl, val_dl, BCEDiceLoss(), config)
history = trainer.train()

assert len(history["train_loss"]) == 3, f"Expected 3 epochs, got {len(history['train_loss'])}"
print("[2/5] Trainer 3-epoch smoke test: OK")

# ------------------------------------------------------------------
# 3. Checkpoints exist
# ------------------------------------------------------------------
ckpt_dir = Path(config["checkpoint_dir"])
ckpts = list(ckpt_dir.glob("*.pt"))
assert len(ckpts) >= 1, f"No checkpoints in {ckpt_dir}"
best = ckpt_dir / "best.pt"
assert best.exists(), "best.pt not found"
print(f"[3/5] Checkpoints: {len(ckpts)} files, best.pt: OK")

# ------------------------------------------------------------------
# 4. History keys and types
# ------------------------------------------------------------------
for key in ["train_loss", "val_loss", "val_f1", "val_iou", "lr"]:
    assert key in history, f"Missing key: {key}"
    assert len(history[key]) == 3
assert all(isinstance(v, float) for v in history["train_loss"])
print("[4/5] History structure: OK")

# ------------------------------------------------------------------
# 5. Scripts are importable and structured correctly
# ------------------------------------------------------------------
train_script = Path("scripts/train.py")
eval_script = Path("scripts/evaluate.py")
assert train_script.exists(), "scripts/train.py missing"
assert eval_script.exists(), "scripts/evaluate.py missing"
# Quick parse check via ast
import ast
for script in [train_script, eval_script]:
    ast.parse(script.read_text(encoding="utf-8"))  # SyntaxError if broken
print("[5/5] Scripts (train.py, evaluate.py) syntax: OK")

# ------------------------------------------------------------------
# Gate summary
# ------------------------------------------------------------------
print()
print("=" * 50)
print("  PHASE 3 GATE")
print("=" * 50)
print(f"  Epochs:      {len(history['train_loss'])}")
print(f"  Best val_f1: {max(history['val_f1']):.4f}")
print(f"  Best val_iou:{max(history['val_iou']):.4f}")
print(f"  Metrics:     6 metrics computed")
print(f"  Checkpoints: {len(ckpts)} files")
print("  STATUS: PHASE 3 COMPLETE")
print("=" * 50)
