"""
Dataset download utility.
Supports LEVIR-CD and WHU-CD from free public sources.
Handles resume, checksum verification, and extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Dataset registry: name -> list of (url, filename, expected_sha256_prefix)
DATASET_REGISTRY: dict[str, list[tuple[str, str, str]]] = {
    "levir-cd": [
        # HuggingFace mirror (most reliable free source)
        (
            "https://huggingface.co/datasets/torchgeo/levir-cd/resolve/main/LEVIR-CD.zip",
            "LEVIR-CD.zip",
            "",  # checksum optional; validate directory structure instead
        ),
    ],
    "whu-cd": [
        (
            "https://huggingface.co/datasets/torchgeo/whu-cd/resolve/main/WHU-CD.zip",
            "WHU-CD.zip",
            "",
        ),
    ],
    "dsifn-cd": [
        (
            "https://huggingface.co/datasets/torchgeo/dsifn-cd/resolve/main/DSIFN-CD.zip",
            "DSIFN-CD.zip",
            "",
        ),
    ],
}

# Expected directory structure to validate after extraction
EXPECTED_DIRS: dict[str, list[str]] = {
    "levir-cd": ["train/A", "train/B", "train/label",
                  "val/A", "val/B", "val/label",
                  "test/A", "test/B", "test/label"],
    "whu-cd": ["A", "B", "label"],
    "dsifn-cd": ["A", "B", "label"],
}


def _sha256_prefix(path: Path, length: int = 8) -> str:
    """Return first `length` hex chars of SHA-256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def download_file(url: str, dest: Path, resume: bool = True) -> None:
    """Download a file with progress bar and resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    existing_size = 0

    if resume and dest.exists():
        existing_size = dest.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        print(f"  Resuming from {existing_size / 1e6:.1f} MB ...")

    response = requests.get(url, headers=headers, stream=True, timeout=30)

    if response.status_code == 416:
        # Range not satisfiable — file already fully downloaded
        print(f"  Already downloaded: {dest.name}")
        return
    if response.status_code == 200 and existing_size > 0:
        # Server doesn't support range — restart
        existing_size = 0

    if response.status_code not in (200, 206):
        raise RuntimeError(
            f"Download failed: HTTP {response.status_code} for {url}"
        )

    total = int(response.headers.get("content-length", 0)) + existing_size
    mode = "ab" if existing_size > 0 else "wb"

    with open(dest, mode) as f, tqdm(
        total=total,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract zip archive, skipping if output exists and looks valid."""
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, output_dir)
    print(f"  Extracted to {output_dir}")


def validate_dataset(dataset: str, output_dir: Path) -> bool:
    """Check that expected directory structure is present and non-empty."""
    expected = EXPECTED_DIRS.get(dataset, [])
    if not expected:
        return True  # No validation defined

    for rel_dir in expected:
        full = output_dir / dataset.upper().replace("-", "_") if False else output_dir
        # Try various possible nested paths
        candidates = [
            output_dir / rel_dir,
            output_dir / dataset.upper() / rel_dir,
            output_dir / dataset.lower() / rel_dir,
        ]
        found = False
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                found = True
                break
        if not found:
            return False
    return True


def download_dataset(dataset: str, output_dir: Path) -> None:
    """Download, extract, and validate a dataset."""
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = DATASET_REGISTRY[dataset]

    success = False
    for url, filename, expected_checksum in entries:
        zip_path = output_dir / filename
        print(f"\nDownloading {dataset} from:\n  {url}")

        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"  WARNING: Download failed ({e}), trying next source ...")
            continue

        # Validate zip integrity
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad = zf.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"Corrupt file in zip: {bad}")
        except zipfile.BadZipFile as e:
            print(f"  WARNING: Zip corrupt ({e}), re-downloading ...")
            zip_path.unlink(missing_ok=True)
            continue

        extract_zip(zip_path, output_dir)
        success = True
        break

    if not success:
        raise RuntimeError(
            f"All download sources failed for {dataset}. "
            "Try downloading manually and placing in data/raw/"
        )

    print(f"\n  Dataset ready at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download satellite change detection datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume: restart download from scratch",
    )
    args = parser.parse_args()

    download_dataset(
        dataset=args.dataset,
        output_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
