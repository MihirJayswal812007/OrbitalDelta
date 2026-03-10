"""
Visualization script for change detection results.

Generates 4-panel comparison images (Image A | Image B | Ground Truth | Prediction)
and overlay heatmaps. Identifies failure cases by F1 score.

Usage:
    python scripts/visualize.py --checkpoint checkpoints/best.pt --config configs/train_levir.yaml
    python scripts/visualize.py --checkpoint checkpoints/best.pt --config configs/train_levir.yaml --num-samples 25
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import CDDataset
from src.data.transforms import get_val_transforms
from src.models.siamese_unet import SiameseUNet
from src.utils.metrics import ChangeDetectionMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet mean/std for un-normalizing display
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor (C, H, W) → uint8 HWC array for display."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD + _MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def compute_sample_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Per-sample F1 (avoids torchmetrics overhead for single images)."""
    tp = (pred_mask & gt_mask).sum()
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom).item() if denom > 0 else 1.0


def save_sample_figure(
    img_a: np.ndarray,
    img_b: np.ndarray,
    gt_mask: np.ndarray,
    pred_probs: np.ndarray,
    pred_mask: np.ndarray,
    sample_idx: int,
    f1: float,
    output_dir: Path,
    prefix: str = "sample",
) -> Path:
    """
    Save a 5-panel figure: Image A | Image B | GT | Pred Probs | Pred Binary
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), facecolor="#111111")
    fig.suptitle(
        f"Sample {sample_idx:04d}  |  F1 = {f1:.4f}",
        color="white",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    panels = [
        (img_a, "Image A (t1)", None, None),
        (img_b, "Image B (t2)", None, None),
        (gt_mask.astype(np.float32), "Ground Truth", "Reds", [0, 1]),
        (pred_probs, "Change Probability", "RdYlGn_r", [0, 1]),
        (pred_mask.astype(np.float32), f"Prediction (thr=0.5)", "Reds", [0, 1]),
    ]

    for ax, (data, title, cmap, vrange) in zip(axes, panels):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.axis("off")
        if cmap is None:
            ax.imshow(data)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])

    plt.tight_layout()
    out_path = output_dir / f"{prefix}_{sample_idx:04d}_f1_{f1:.4f}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def save_overlay_figure(
    img_b: np.ndarray,
    pred_probs: np.ndarray,
    pred_mask: np.ndarray,
    sample_idx: int,
    f1: float,
    output_dir: Path,
) -> Path:
    """Save heatmap overlay on Image B."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#0d0d0d")
    fig.suptitle(
        f"Overlay — Sample {sample_idx:04d}  |  F1 = {f1:.4f}",
        color="white", fontsize=11, y=1.02
    )

    axes[0].imshow(img_b)
    axes[0].imshow(pred_probs, cmap="hot", alpha=0.45, vmin=0, vmax=1)
    axes[0].set_title("Heatmap Overlay", color="white", fontsize=9)
    axes[0].axis("off")

    overlay = img_b.copy()
    overlay[pred_mask] = [255, 60, 60]
    axes[1].imshow(overlay)
    axes[1].set_title("Binary Change Overlay", color="white", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()
    out_path = output_dir / f"overlay_{sample_idx:04d}_f1_{f1:.4f}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate change detection visualizations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--num-samples", type=int, default=25)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", default="outputs/visualizations")
    p.add_argument("--failure-mode", action="store_true",
                   help="Also save top-10 worst F1 samples separately")
    p.add_argument("--data-root", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config: dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.data_root:
        config["data"]["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    state = torch.load(args.checkpoint, map_location=device)
    saved_cfg = state.get("config", config)
    encoder = saved_cfg.get("model", {}).get("encoder", "resnet18")

    model = SiameseUNet(encoder_name=encoder, pretrained=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    # Load dataset
    crop_size = config["data"].get("crop_size", 256)
    data_root = config["data"]["root"]
    dataset = CDDataset(
        root=data_root,
        split=args.split,
        transform=get_val_transforms(crop_size),
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    if args.failure_mode:
        failures_dir.mkdir(exist_ok=True)

    # Collect samples
    samples_generated = 0
    sample_scores: list[tuple[float, int]] = []

    indices = list(range(min(args.num_samples * 4, len(dataset))))  # buffer for failures

    with torch.no_grad():
        for i, idx in enumerate(indices):
            if samples_generated >= args.num_samples:
                break

            img_a_t, img_b_t, mask_t = dataset[idx]

            # Run inference
            a = img_a_t.unsqueeze(0).to(device)
            b = img_b_t.unsqueeze(0).to(device)
            pred_prob = model(a, b).squeeze().cpu().numpy()  # (H, W)

            # Prepare arrays
            img_a_np = denormalize(img_a_t)
            img_b_np = denormalize(img_b_t)
            gt_mask_np = (mask_t.squeeze().numpy() > 0.5)
            pred_mask_np = (pred_prob > args.threshold)

            # Per-sample F1
            f1 = compute_sample_f1(pred_mask_np, gt_mask_np)
            sample_scores.append((f1, idx))

            # Save main figure
            save_sample_figure(
                img_a_np, img_b_np, gt_mask_np, pred_prob, pred_mask_np,
                sample_idx=idx, f1=f1, output_dir=output_dir
            )
            # Save overlay
            save_overlay_figure(
                img_b_np, pred_prob, pred_mask_np,
                sample_idx=idx, f1=f1, output_dir=output_dir
            )
            samples_generated += 1

    # Failure analysis: top-10 worst
    if args.failure_mode and sample_scores:
        worst = sorted(sample_scores, key=lambda x: x[0])[:10]
        logger.info(f"Failure analysis: re-saving {len(worst)} worst samples")
        with torch.no_grad():
            for f1, idx in worst:
                img_a_t, img_b_t, mask_t = dataset[idx]
                a = img_a_t.unsqueeze(0).to(device)
                b = img_b_t.unsqueeze(0).to(device)
                pred_prob = model(a, b).squeeze().cpu().numpy()
                gt_mask_np = (mask_t.squeeze().numpy() > 0.5)
                pred_mask_np = (pred_prob > args.threshold)
                save_sample_figure(
                    denormalize(img_a_t), denormalize(img_b_t),
                    gt_mask_np, pred_prob, pred_mask_np,
                    sample_idx=idx, f1=f1,
                    output_dir=failures_dir, prefix="failure"
                )

    # Summary stats
    if sample_scores:
        f1_values = [s[0] for s in sample_scores]
        avg_f1 = np.mean(f1_values)
        worst_f1 = min(f1_values)
        best_f1 = max(f1_values)
    else:
        avg_f1 = worst_f1 = best_f1 = 0.0

    output_files = list(output_dir.glob("*.png"))
    logger.info(f"Saved {len(output_files)} visualization files to {output_dir}")

    print("\n" + "=" * 50)
    print("  VISUALIZATION COMPLETE")
    print("=" * 50)
    print(f"  Samples visualized: {samples_generated}")
    print(f"  Output files:       {len(output_files)}")
    print(f"  Avg per-sample F1:  {avg_f1:.4f}")
    print(f"  Best sample F1:     {best_f1:.4f}")
    print(f"  Worst sample F1:    {worst_f1:.4f}")
    print(f"  Output dir:         {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
