"""
Training loop for Siamese U-Net change detection.

Features:
  - AdamW optimizer with CosineAnnealingWarmRestarts scheduler
  - Mixed precision training (torch.cuda.amp) — graceful CPU fallback
  - Gradient clipping (max_norm=1.0)
  - Early stopping on val F1 (patience configurable)
  - Top-k checkpoint saving (by val F1)
  - TensorBoard and optional W&B logging
  - Reproducibility via seed_everything
  - Resume from checkpoint
  - Progress bars via tqdm
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import ChangeDetectionMetrics

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Stops training when validation metric stops improving.
    Saves the best model state in-memory for final checkpoint restore.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float = -float("inf")
        self.best_state: dict | None = None
        self.should_stop = False

    def __call__(self, score: float, model_state: dict) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = model_state
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    """
    Keeps the top-k checkpoints by val F1.
    Automatically prunes checkpoints outside the top-k.
    """

    def __init__(self, save_dir: Path, top_k: int = 3) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self._checkpoints: list[tuple[float, Path]] = []  # (score, path)

    def save(
        self,
        state: dict,
        score: float,
        epoch: int,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint, prune if over top_k."""
        fname = self.save_dir / f"epoch_{epoch:03d}_f1_{score:.4f}.pt"
        torch.save(state, fname)

        self._checkpoints.append((score, fname))
        self._checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Prune checkpoints beyond top_k (but never prune 'best.pt')
        while len(self._checkpoints) > self.top_k:
            _, old_path = self._checkpoints.pop()
            if old_path.exists() and "best" not in old_path.name:
                old_path.unlink()

        if is_best:
            best_path = self.save_dir / "best.pt"
            shutil.copy2(fname, best_path)
            logger.info(f"Saved best checkpoint → {best_path} (F1={score:.4f})")

        return fname


class Trainer:
    """
    Orchestrates training and validation loops.

    Args:
        model:     SiameseUNet (or any nn.Module with same signature)
        train_dl:  Training DataLoader yielding (img_a, img_b, mask)
        val_dl:    Validation DataLoader
        loss_fn:   e.g. BCEDiceLoss()
        config:    Dict loaded from train_*.yaml
    """

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        loss_fn: nn.Module,
        config: dict[str, Any],
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer device: {self.device}")

        self.model = model.to(self.device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn.to(self.device)

        tc = config["training"]
        self.max_epochs = tc["epochs"]
        self.grad_clip = tc.get("gradient_clip_norm", 1.0)
        self.use_amp = tc.get("mixed_precision", True) and self.device.type == "cuda"

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=tc["lr"],
            weight_decay=tc.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=tc.get("scheduler_T0", 10),
            T_mult=1,
            eta_min=1e-6,
        )

        # Mixed precision scaler (no-op on CPU)
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        # Early stopping
        patience = tc.get("early_stopping_patience", 15)
        self.early_stopping = EarlyStopping(patience=patience)

        # Checkpoint manager
        save_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        top_k = config.get("logging", {}).get("save_top_k", 3)
        self.ckpt_manager = CheckpointManager(save_dir, top_k=top_k)

        # Metrics
        self.metrics = ChangeDetectionMetrics(device=str(self.device))

        # Logging
        log_cfg = config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        self.log_every = log_cfg.get("log_every_n_steps", 50)

        # Optional W&B
        self._wandb = None
        if log_cfg.get("backend") == "wandb":
            try:
                import wandb

                self._wandb = wandb
                wandb.init(
                    project="orbital-delta-cd",
                    config=config,
                    resume="allow",
                )
            except Exception as e:
                logger.warning(f"W&B init failed, falling back to TensorBoard: {e}")

        # Seed
        seed_everything(config.get("seed", 42))

        # Track best val F1
        self._best_val_f1: float = -1.0
        self._start_epoch: int = 1

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training from a saved checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self._start_epoch = state.get("epoch", 1) + 1
        self._best_val_f1 = state.get("val_f1", -1.0)
        logger.info(
            f"Resumed from {checkpoint_path} "
            f"(epoch={state.get('epoch')}, val_f1={self._best_val_f1:.4f})"
        )

    # ------------------------------------------------------------------
    #  Core loops
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """One training epoch. Returns mean training loss."""
        self.model.train()
        total_loss = 0.0
        step = 0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, (img_a, img_b, mask) in enumerate(pbar):
            img_a = img_a.to(self.device, non_blocking=True)
            img_b = img_b.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model(img_a, img_b)
                loss = self.loss_fn(pred, mask)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            step += 1

            if step % self.log_every == 0:
                global_step = (epoch - 1) * len(self.train_dl) + batch_idx
                self.writer.add_scalar("train/loss_step", loss.item(), global_step)
                if self._wandb:
                    self._wandb.log(
                        {"train/loss_step": loss.item()}, step=global_step
                    )

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Step scheduler once per epoch
        self.scheduler.step(epoch)

        return total_loss / max(step, 1)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        """Validation epoch. Returns (val_loss, metrics_dict)."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        step = 0

        pbar = tqdm(self.val_dl, desc=f"Epoch {epoch} [val]  ", leave=False)
        for img_a, img_b, mask in pbar:
            img_a = img_a.to(self.device, non_blocking=True)
            img_b = img_b.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                pred = self.model(img_a, img_b)
                loss = self.loss_fn(pred, mask)

            self.metrics.update(pred, mask)
            total_loss += loss.item()
            step += 1
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = total_loss / max(step, 1)
        results = self.metrics.compute()
        return val_loss, results

    # ------------------------------------------------------------------
    #  Main train() entry
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list]:
        """
        Run the full training loop.

        Returns:
            history dict with keys: train_loss, val_loss, val_f1, val_iou, lr
        """
        history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_iou": [],
            "lr": [],
        }

        logger.info(
            f"Starting training: {self.max_epochs} epochs, "
            f"device={self.device}, AMP={self.use_amp}"
        )

        for epoch in range(self._start_epoch, self.max_epochs + 1):
            # --- Train ---
            train_loss = self._train_epoch(epoch)

            # --- Validate ---
            val_loss, val_metrics = self._val_epoch(epoch)
            val_f1 = val_metrics["f1"]
            val_iou = val_metrics["iou"]
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Log ---
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_iou"].append(val_iou)
            history["lr"].append(current_lr)

            self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/f1", val_f1, epoch)
            self.writer.add_scalar("val/iou", val_iou, epoch)
            self.writer.add_scalar("val/precision", val_metrics["precision"], epoch)
            self.writer.add_scalar("val/recall", val_metrics["recall"], epoch)
            self.writer.add_scalar("val/kappa", val_metrics["kappa"], epoch)
            self.writer.add_scalar("lr", current_lr, epoch)

            if self._wandb:
                self._wandb.log(
                    {
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/f1": val_f1,
                        "val/iou": val_iou,
                        "lr": current_lr,
                        "epoch": epoch,
                    }
                )

            is_best = val_f1 > self._best_val_f1
            if is_best:
                self._best_val_f1 = val_f1

            # --- Checkpoint ---
            state = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "val_f1": val_f1,
                "val_iou": val_iou,
                "config": self.config,
            }
            self.ckpt_manager.save(state, val_f1, epoch, is_best=is_best)

            print(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"F1={val_f1:.4f} | "
                f"IoU={val_iou:.4f} | "
                f"lr={current_lr:.2e}"
                + (" << BEST" if is_best else "")
            )

            # --- Early stopping ---
            if self.early_stopping(val_f1, state):
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val_f1={self._best_val_f1:.4f}"
                )
                break

        self.writer.close()
        if self._wandb:
            self._wandb.finish()

        return history
