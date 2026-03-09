"""
PyTorch Dataset for satellite change detection image pairs.
Loads (image_A, image_B, mask) triplets from preprocessed directory structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError as e:
    raise ImportError("albumentations is required. Run: pip install albumentations") from e


class CDDataset(Dataset):
    """
    Change Detection Dataset.

    Expects directory structure:
        root/
        ├── A/          ← time-1 images (.png)
        ├── B/          ← time-2 images (.png)
        └── label/      ← binary change masks (.png)

    Args:
        root:      Path to split directory (e.g. data/processed/levir-cd/train/)
        split:     Informational only — used in repr
        transform: Albumentations Compose with additional_targets={"image_b": "image", "mask": "mask"}
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Validate directory structure
        for sub in ("A", "B", "label"):
            d = self.root / sub
            if not d.exists():
                raise FileNotFoundError(
                    f"Directory not found: {d}. "
                    "Run preprocessing: python -m src.data.preprocess"
                )

        # Collect image IDs from A/ directory
        self.ids = sorted([p.stem for p in (self.root / "A").glob("*.png")])
        if len(self.ids) == 0:
            raise RuntimeError(f"No images found in {self.root / 'A'}")

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"CDDataset(split={self.split!r}, n={len(self)}, root={self.root})"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]

        # Load images
        img_a = np.array(Image.open(self.root / "A" / f"{img_id}.png").convert("RGB"))
        img_b = np.array(Image.open(self.root / "B" / f"{img_id}.png").convert("RGB"))
        mask = np.array(Image.open(self.root / "label" / f"{img_id}.png").convert("L"))

        # Binarize mask: >128 → 1, else → 0
        mask = (mask > 128).astype(np.uint8)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=img_a, image_b=img_b, mask=mask)
            img_a = transformed["image"]        # torch.Tensor (3, H, W)
            img_b = transformed["image_b"]      # torch.Tensor (3, H, W)
            mask = transformed["mask"]          # torch.Tensor (H, W)
        else:
            # Fallback: convert to tensor without normalization
            img_a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 255.0
            img_b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        # Ensure mask has channel dim: (H, W) → (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return img_a, img_b, mask

    def get_image_id(self, idx: int) -> str:
        """Return the image ID for a given index (useful for visualization)."""
        return self.ids[idx]
