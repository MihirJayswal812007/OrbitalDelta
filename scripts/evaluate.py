"""
Evaluation entry point for OrbitalDelta change detection system.

Loads a trained checkpoint, runs inference on the test set,
computes all metrics, and saves a JSON results file.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/train_levir.yaml
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/train_levir.yaml --subset 32
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/train_levir.yaml --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, Subset

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained change detection model")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization threshold for change probability",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Evaluate on first N samples only (smoke test)",
    )
    p.add_argument(
        "--output",
        default="outputs/eval_results.json",
        help="Path to save JSON results",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data.root in config",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config: dict = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        config["data"]["root"] = args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    # Build model
    model_cfg = config.get("model", {})
    # Support both dict state and full state with config embedded
    if "config" in state:
        saved_cfg = state["config"]
        encoder = saved_cfg.get("model", {}).get("encoder", model_cfg.get("encoder", "resnet18"))
    else:
        encoder = model_cfg.get("encoder", "resnet18")

    model = SiameseUNet(encoder_name=encoder, pretrained=False)
    model.load_state_dict(state["model_state"])
    model = model.to(device)
    model.eval()

    params = model.count_parameters()
    logger.info(f"Model: {encoder} | {params['total_M']}M params")

    # If checkpoint has known metrics, report them
    if "val_f1" in state:
        logger.info(
            f"Checkpoint saved at epoch {state.get('epoch', '?')} "
            f"with val_f1={state['val_f1']:.4f}"
        )

    # Build dataset
    crop_size = config["data"].get("crop_size", 256)
    data_root = config["data"]["root"]

    dataset = CDDataset(
        root=data_root,
        split=args.split,
        transform=get_val_transforms(crop_size),
    )
    if args.subset:
        dataset = Subset(dataset, range(min(args.subset, len(dataset))))

    batch_size = config["training"].get("batch_size", 8)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Evaluating {len(dataset)} samples from '{args.split}' split...")
    logger.info(f"Binarization threshold: {args.threshold}")

    # Evaluate
    metrics = ChangeDetectionMetrics(threshold=args.threshold, device=str(device))

    with torch.no_grad():
        for img_a, img_b, mask in dl:
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            pred = model(img_a, img_b)
            metrics.update(pred, mask)

    results = metrics.compute()

    # Print results table
    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS ({args.split} split)")
    print("=" * 50)
    targets = {"f1": 0.88, "iou": 0.80}
    for k, v in results.items():
        target_str = ""
        if k in targets:
            met = "✅" if v >= targets[k] else "❌"
            target_str = f" {met} (target ≥ {targets[k]})"
        print(f"  {k:12s}: {v:.4f}{target_str}")
    print("=" * 50)

    # Overall verdict
    f1_ok = results.get("f1", 0.0) >= 0.88
    iou_ok = results.get("iou", 0.0) >= 0.80
    if f1_ok and iou_ok:
        print("\n  🎉 SUCCESS: Both F1 ≥ 0.88 and IoU ≥ 0.80 targets met!")
    else:
        print("\n  ⚠️  Targets not yet met — review Task 4.2 for tuning guidance")
    print()

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Enrich results with metadata
    full_results = {
        **results,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "threshold": args.threshold,
        "n_samples": len(dataset),
        "encoder": encoder,
    }
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
