"""
Preprocessing pipeline for satellite change detection datasets.

Crops large images into 256×256 patches and splits by image-level
(not patch-level) to prevent data leakage across train/val/test.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def find_image_ids(root: Path) -> list[str]:
    """
    Find image IDs (filenames without extension) from the A/ subdirectory.
    Works with LEVIR-CD raw format: root/{A,B,label}/*.png
    """
    a_dir = root / "A"
    if not a_dir.exists():
        raise FileNotFoundError(
            f"Expected directory {a_dir}. "
            "Did you download the dataset? Run: python -m src.data.download --dataset levir-cd"
        )
    return sorted([p.stem for p in a_dir.glob("*.png")])


def find_split_dirs(root: Path) -> list[tuple[str, Path]]:
    """
    Handle multiple raw dataset formats:
    - LEVIR-CD raw: root/{train,val,test}/{A,B,label}/
    - Generic raw:  root/{A,B,label}/
    Returns list of (split_name, split_root) paths.
    """
    # LEVIR-CD comes pre-split into train/val/test
    standard_splits = ["train", "val", "test"]
    if all((root / s / "A").exists() for s in standard_splits):
        return [(s, root / s) for s in standard_splits]

    # Generic: single flat directory → we split ourselves
    if (root / "A").exists():
        return [("all", root)]

    raise FileNotFoundError(
        f"No recognizable dataset structure at {root}. "
        "Expected either {root}/{A,B,label}/ or {root}/{train,val,test}/{A,B,label}/"
    )


def crop_image_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    label: np.ndarray,
    crop_size: int,
    image_id: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """
    Crop a triplet (A, B, label) into non-overlapping patches of crop_size×crop_size.
    Returns list of (patch_a, patch_b, patch_label, patch_id).
    """
    h, w = img_a.shape[:2]
    patches = []
    idx = 0
    for row in range(0, h - crop_size + 1, crop_size):
        for col in range(0, w - crop_size + 1, crop_size):
            pa = img_a[row:row + crop_size, col:col + crop_size]
            pb = img_b[row:row + crop_size, col:col + crop_size]
            pl = label[row:row + crop_size, col:col + crop_size]
            patch_id = f"{image_id}_{idx:04d}"
            patches.append((pa, pb, pl, patch_id))
            idx += 1
    return patches


def save_patch(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    patch_label: np.ndarray,
    patch_id: str,
    output_root: Path,
    split: str,
) -> None:
    """Save a patch triplet to disk."""
    for sub, data in [("A", patch_a), ("B", patch_b), ("label", patch_label)]:
        out_dir = output_root / split / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(data).save(out_dir / f"{patch_id}.png")


def preprocess_single_split(
    input_root: Path,
    output_root: Path,
    split_name: str,
    crop_size: int,
) -> int:
    """Process all images in a single split directory. Returns patch count."""
    image_ids = find_image_ids(input_root)
    total_patches = 0

    for img_id in tqdm(image_ids, desc=f"Processing {split_name}", unit="img"):
        a_path = input_root / "A" / f"{img_id}.png"
        b_path = input_root / "B" / f"{img_id}.png"
        l_path = input_root / "label" / f"{img_id}.png"

        if not (a_path.exists() and b_path.exists() and l_path.exists()):
            print(f"  SKIP: missing pair for {img_id}")
            continue

        img_a = np.array(Image.open(a_path).convert("RGB"))
        img_b = np.array(Image.open(b_path).convert("RGB"))
        label = np.array(Image.open(l_path).convert("L"))  # grayscale

        # Binarize: >128 → 255 (change), else → 0 (no change)
        label = (label > 128).astype(np.uint8) * 255

        patches = crop_image_pair(img_a, img_b, label, crop_size, img_id)
        for pa, pb, pl, pid in patches:
            save_patch(pa, pb, pl, pid, output_root, split_name)
            total_patches += 1

    return total_patches


def preprocess_and_split(
    input_root: Path,
    output_root: Path,
    crop_size: int = 256,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """
    Main preprocessing entrypoint.
    Handles both pre-split and unsplit dataset formats.
    Ensures NO data leakage: patches from same image always go to same split.
    """
    random.seed(seed)
    np.random.seed(seed)

    split_dirs = find_split_dirs(input_root)
    patch_counts = {}

    if len(split_dirs) > 1 and split_dirs[0][0] != "all":
        # Dataset is already split (e.g. LEVIR-CD)
        print("Dataset is pre-split (train/val/test). Processing each split separately.")
        for split_name, split_root in split_dirs:
            n = preprocess_single_split(split_root, output_root, split_name, crop_size)
            patch_counts[split_name] = n
    else:
        # Generic unsplit dataset: split by image ID
        print("Dataset is flat. Splitting by image-level ratio.")
        _, only_root = split_dirs[0]
        image_ids = find_image_ids(only_root)
        random.shuffle(image_ids)

        n = len(image_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        split_map = (
            {img_id: "train" for img_id in image_ids[:n_train]}
            | {img_id: "val" for img_id in image_ids[n_train:n_train + n_val]}
            | {img_id: "test" for img_id in image_ids[n_train + n_val:]}
        )

        for split_name in ["train", "val", "test"]:
            patch_counts[split_name] = 0

        for img_id in tqdm(image_ids, desc="Preprocessing", unit="img"):
            split_name = split_map[img_id]
            a_path = only_root / "A" / f"{img_id}.png"
            b_path = only_root / "B" / f"{img_id}.png"
            l_path = only_root / "label" / f"{img_id}.png"

            if not (a_path.exists() and b_path.exists() and l_path.exists()):
                continue

            img_a = np.array(Image.open(a_path).convert("RGB"))
            img_b = np.array(Image.open(b_path).convert("RGB"))
            label = np.array(Image.open(l_path).convert("L"))
            label = (label > 128).astype(np.uint8) * 255

            patches = crop_image_pair(img_a, img_b, label, crop_size, img_id)
            for pa, pb, pl, pid in patches:
                save_patch(pa, pb, pl, pid, output_root, split_name)
                patch_counts[split_name] += 1

    return patch_counts


def verify_no_leakage(output_root: Path) -> None:
    """Assert that no image ID appears in more than one split."""
    splits = ["train", "val", "test"]
    split_ids: dict[str, set[str]] = {}
    for split in splits:
        a_dir = output_root / split / "A"
        if a_dir.exists():
            # ID = everything before the _NNNN patch index
            split_ids[split] = {
                "_".join(p.stem.split("_")[:-1]) for p in a_dir.glob("*.png")
            }
        else:
            split_ids[split] = set()

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            overlap = split_ids[s1] & split_ids[s2]
            if overlap:
                raise RuntimeError(
                    f"DATA LEAKAGE: {len(overlap)} images appear in both {s1} and {s2}: "
                    f"{list(overlap)[:5]}"
                )
    print("Leakage check: PASSED — no image-level overlap between splits.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess satellite change detection dataset")
    parser.add_argument("--input", required=True, help="Raw dataset directory")
    parser.add_argument("--output", required=True, help="Processed output directory")
    parser.add_argument("--crop-size", type=int, default=256, help="Patch size (default: 256)")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    print(f"\nPreprocessing: {input_root} → {output_root}")
    print(f"Crop size: {args.crop_size}px | Splits: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio:.2f}")

    patch_counts = preprocess_and_split(
        input_root=input_root,
        output_root=output_root,
        crop_size=args.crop_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\nPatch counts:")
    for split, count in patch_counts.items():
        print(f"  {split}: {count} patches")

    verify_no_leakage(output_root)
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
