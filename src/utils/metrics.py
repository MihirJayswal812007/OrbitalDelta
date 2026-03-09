"""
Metrics for binary change detection evaluation.

Computes: F1, IoU (Jaccard), Precision, Recall, Overall Accuracy, Cohen's Kappa.
Uses torchmetrics for correct batched accumulation (not per-batch averaging).
"""

from __future__ import annotations

import torch
import torchmetrics
from torchmetrics import MetricCollection


class ChangeDetectionMetrics:
    """
    Accumulates predictions and targets across batches, then computes
    final metrics over the entire dataset — avoids micro-averaging bias.

    Usage:
        metrics = ChangeDetectionMetrics(threshold=0.5, device='cuda')
        for batch in dataloader:
            pred_probs = model(batch.a, batch.b)   # (B, 1, H, W) in [0, 1]
            metrics.update(pred_probs, batch.mask)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(
        self,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.threshold = threshold
        self.device = torch.device(device)

        # torchmetrics handles stateful accumulation across .update() calls
        self._collection = MetricCollection(
            {
                "f1": torchmetrics.F1Score(task="binary", threshold=threshold),
                "iou": torchmetrics.JaccardIndex(task="binary", threshold=threshold),
                "precision": torchmetrics.Precision(
                    task="binary", threshold=threshold
                ),
                "recall": torchmetrics.Recall(task="binary", threshold=threshold),
                "accuracy": torchmetrics.Accuracy(task="binary", threshold=threshold),
                "kappa": torchmetrics.CohenKappa(task="binary", threshold=threshold),
            }
        ).to(self.device)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Accumulate a batch of predictions and targets.

        Args:
            pred:   Sigmoid-activated change map (B, 1, H, W) or (B, H, W) in [0, 1]
            target: Ground-truth change mask (B, 1, H, W) or (B, H, W) in {0, 1}
        """
        # Flatten to (N,) regardless of input shape (handles 1D, 2D, 3D, 4D)
        pred = pred.to(self.device).reshape(-1)
        target = target.to(self.device).reshape(-1).long()
        self._collection.update(pred, target)

    def compute(self) -> dict[str, float]:
        """
        Compute final metrics over all accumulated predictions.

        Returns:
            dict with keys: f1, iou, precision, recall, accuracy, kappa
        """
        raw = self._collection.compute()
        return {k: v.item() for k, v in raw.items()}

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._collection.reset()

    def to(self, device: str | torch.device) -> "ChangeDetectionMetrics":
        """Move metrics to device."""
        self.device = torch.device(device)
        self._collection = self._collection.to(self.device)
        return self
