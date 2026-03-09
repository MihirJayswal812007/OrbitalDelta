"""
Training entry point for OrbitalDelta change detection system.

Usage:
    python scripts/train.py --config configs/train_levir.yaml
    python scripts/train.py --config configs/train_levir.yaml --epochs 5 --batch-size 4 --subset 64
    python scripts/train.py --config configs/train_levir.yaml --resume checkpoints/epoch_010_f1_0.8500.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path (works both as script and module)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, Subset

from src.data.dataset import CDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.losses import BCEDiceLoss, FocalDiceLoss
from src.models.siamese_unet import SiameseUNet
from src.training.trainer import Trainer, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train OrbitalDelta change detection model")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument(
        "--batch-size", type=int, default=None, help="Override config batch_size"
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use only first N samples per split (for smoke tests)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    p.add_argument(
        "--loss",
        choices=["bce_dice", "focal_dice"],
        default="bce_dice",
        help="Loss function to use",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data.root in config",
    )
    return p.parse_args()


def build_dataloaders(
    config: dict,
    crop_size: int,
    batch_size: int,
    subset: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    data_root = config["data"]["root"]
    num_workers = config["data"].get("num_workers", 4)

    train_ds: CDDataset | Subset = CDDataset(
        root=data_root, split="train", transform=get_train_transforms(crop_size)
    )
    val_ds: CDDataset | Subset = CDDataset(
        root=data_root, split="val", transform=get_val_transforms(crop_size)
    )

    if subset is not None:
        train_ds = Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds = Subset(val_ds, range(min(subset // 4, len(val_ds))))
        logger.info(f"Using subset: train={len(train_ds)}, val={len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders: train={len(train_ds)} samples, "
        f"val={len(val_ds)} samples, batch_size={batch_size}"
    )
    return train_dl, val_dl


def main() -> None:
    args = parse_args()

    # Load config
    config: dict = yaml.safe_load(Path(args.config).read_text())

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.data_root is not None:
        config["data"]["root"] = args.data_root

    # Seed before building anything
    seed_everything(config.get("seed", 42))

    crop_size = config["data"].get("crop_size", 256)
    batch_size = config["training"]["batch_size"]

    # Data
    train_dl, val_dl = build_dataloaders(config, crop_size, batch_size, args.subset)

    # Model
    model = SiameseUNet(
        encoder_name=config["model"]["encoder"],
        pretrained=config["model"].get("pretrained", True),
        dropout=config["model"].get("dropout", 0.1),
    )
    params = model.count_parameters()
    logger.info(f"Model: {params['total_M']}M params total, {params['trainable']} trainable")

    # Loss
    loss_cfg = config["training"].get("loss", {})
    if args.loss == "focal_dice":
        loss_fn = FocalDiceLoss()
        logger.info("Using FocalDiceLoss")
    else:
        loss_fn = BCEDiceLoss(
            bce_weight=loss_cfg.get("bce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
        )
        logger.info(
            f"Using BCEDiceLoss(bce={loss_cfg.get('bce_weight', 0.5)}, "
            f"dice={loss_cfg.get('dice_weight', 0.5)})"
        )

    # Trainer
    trainer = Trainer(model, train_dl, val_dl, loss_fn, config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("=" * 60)
    logger.info("  STARTING TRAINING")
    logger.info("=" * 60)
    history = trainer.train()

    # Save final history
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    history_path = output_dir / "train_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. History saved to {history_path}")
    logger.info(f"Best val_f1: {max(history['val_f1']):.4f}")
    logger.info(f"Best val_iou: {max(history['val_iou']):.4f}")

    # Verify checkpoint exists
    best_ckpt = Path(config.get("checkpoint_dir", "checkpoints")) / "best.pt"
    if best_ckpt.exists():
        logger.info(f"Best checkpoint: {best_ckpt}")
    else:
        logger.warning("best.pt not found — training may have run too few epochs")


if __name__ == "__main__":
    main()
