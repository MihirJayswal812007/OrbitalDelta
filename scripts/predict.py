"""
Inference CLI for OrbitalDelta — predicts change map between two satellite images.

Handles:
  - Arbitrary input sizes (auto-tiling for large images)
  - Multiple formats (PNG, JPEG, GeoTIFF)
  - GeoTIFF output with preserved spatial metadata
  - Threshold-tunable binary map

Usage:
    python scripts/predict.py \\
        --img-a path/to/before.tif \\
        --img-b path/to/after.tif \\
        --checkpoint checkpoints/best.pt \\
        --output outputs/change_map.png

    # With GeoTIFF georeferenced output:
    python scripts/predict.py \\
        --img-a before.tif --img-b after.tif \\
        --checkpoint checkpoints/best.pt \\
        --output outputs/change_map.tif \\
        --geotiff
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.siamese_unet import SiameseUNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ImageNet normalization constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Maximum single-pass size; larger images are auto-tiled
SINGLE_PASS_MAX_PX = 640


def normalize_image(img_np: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 array to normalized CHW float tensor."""
    t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2, 0, 1)
    return (t - _MEAN) / _STD


def load_image(path: str | Path, target_channels: int = 3) -> tuple[np.ndarray, dict]:
    """
    Load an image from any supported format.

    Returns:
        img: HWC uint8 array (H, W, 3)
        meta: dict with geospatial metadata (populated for GeoTIFF, empty otherwise)
    """
    path = Path(path)
    meta: dict = {}

    # Try rasterio for GeoTIFF first
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                meta = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "count": src.count,
                }
                # Read first 3 bands (RGB or first 3 multispectral)
                bands = min(src.count, target_channels)
                data = src.read(list(range(1, bands + 1)))  # (C, H, W)
                # Normalize to 0-255 uint8
                data = data.astype(np.float32)
                for c in range(data.shape[0]):
                    p2, p98 = np.percentile(data[c], [2, 98])
                    if p98 > p2:
                        data[c] = np.clip((data[c] - p2) / (p98 - p2) * 255, 0, 255)
                    else:
                        data[c] = np.clip(data[c] / max(data[c].max(), 1) * 255, 0, 255)
                img = data.astype(np.uint8).transpose(1, 2, 0)  # HWC
                # Pad to 3 channels if needed
                if img.shape[2] < 3:
                    img = np.stack([img[:, :, 0]] * 3, axis=2)
                return img[:, :, :3], meta
        except ImportError:
            logger.warning("rasterio not available, falling back to PIL for .tif")
        except Exception as e:
            logger.warning(f"rasterio failed: {e}, trying PIL")

    # Fallback: PIL
    pil_img = Image.open(path).convert("RGB")
    return np.array(pil_img), meta


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, tuple]:
    """Pad tensor to be divisible by `multiple`. Returns (padded, (pad_h, pad_w))."""
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (pad_h, pad_w)


def tile_inference(
    model: SiameseUNet,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 64,
    device: torch.device = torch.device("cpu"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Run inference on a large image pair using overlapping tiles.
    Uses weighted blending in overlap regions to eliminate seams.

    Args:
        img_a, img_b: (C, H, W) normalized tensors
        tile_size:    Tile dimension (model input size)
        overlap:      Overlap between adjacent tiles (pixels)
        threshold:    Not applied here — returns raw probabilities

    Returns:
        (H, W) probability map in [0, 1]
    """
    C, H, W = img_a.shape
    stride = tile_size - overlap

    output = torch.zeros(H, W, device=device)
    weight = torch.zeros(H, W, device=device)

    # Build a cosine weight map for blending
    cos_y = torch.hann_window(tile_size, periodic=False, device=device)
    cos_x = torch.hann_window(tile_size, periodic=False, device=device)
    blend_kernel = cos_y.unsqueeze(1) * cos_x.unsqueeze(0)  # (tile_size, tile_size)

    # Pad image to cover all tiles uniformly
    pad_h = max(0, tile_size - H % stride) if H % stride else 0
    pad_w = max(0, tile_size - W % stride) if W % stride else 0
    a_pad = F.pad(img_a, (0, pad_w, 0, pad_h), mode="reflect")
    b_pad = F.pad(img_b, (0, pad_w, 0, pad_h), mode="reflect")

    _, Hp, Wp = a_pad.shape
    rows = range(0, Hp - tile_size + 1, stride)
    cols = range(0, Wp - tile_size + 1, stride)

    model.eval()
    with torch.no_grad():
        for r in rows:
            for c in cols:
                tile_a = a_pad[:, r : r + tile_size, c : c + tile_size].unsqueeze(0).to(device)
                tile_b = b_pad[:, r : r + tile_size, c : c + tile_size].unsqueeze(0).to(device)

                pred = model(tile_a, tile_b).squeeze()  # (tile_size, tile_size)

                # Write to output with blending, clipped to original image bounds
                r_end = min(r + tile_size, H)
                c_end = min(c + tile_size, W)
                tile_r_end = r_end - r
                tile_c_end = c_end - c

                output[r:r_end, c:c_end] += (
                    pred[:tile_r_end, :tile_c_end] * blend_kernel[:tile_r_end, :tile_c_end]
                )
                weight[r:r_end, c:c_end] += blend_kernel[:tile_r_end, :tile_c_end]

    # Normalize by accumulated weights
    weight = torch.clamp(weight, min=1e-8)
    return (output / weight).cpu()


def run_inference(
    model: SiameseUNet,
    img_a_np: np.ndarray,
    img_b_np: np.ndarray,
    device: torch.device,
    tile_size: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full inference pipeline.

    Returns:
        prob_map: (H, W) float32 in [0, 1]
        binary_map: (H, W) uint8 {0, 255}
    """
    H, W = img_a_np.shape[:2]

    # Normalize
    a_t = normalize_image(img_a_np)  # (C, H, W)
    b_t = normalize_image(img_b_np)

    if H <= SINGLE_PASS_MAX_PX and W <= SINGLE_PASS_MAX_PX:
        # Single-pass: pad to 32-multiple, forward, unpad
        a_padded, (ph, pw) = pad_to_multiple(a_t)
        b_padded, _ = pad_to_multiple(b_t)

        with torch.no_grad():
            pred = model(
                a_padded.unsqueeze(0).to(device),
                b_padded.unsqueeze(0).to(device),
            ).squeeze().cpu()

        # Remove padding
        pred = pred[:H, :W]
    else:
        logger.info(f"Large image ({H}x{W}): using tiling engine (tile={tile_size}, overlap={overlap})")
        pred = tile_inference(model, a_t, b_t, tile_size, overlap, device, threshold)

    prob_map = pred.numpy().astype(np.float32)
    binary_map = (prob_map >= threshold).astype(np.uint8) * 255
    return prob_map, binary_map


def save_output(
    prob_map: np.ndarray,
    binary_map: np.ndarray,
    output_path: Path,
    geo_meta: dict | None,
    save_geotiff: bool = False,
) -> None:
    """Save change map. Uses GeoTIFF if metadata available and requested."""
    if save_geotiff and geo_meta and output_path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            from rasterio.transform import from_gcps

            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                height=binary_map.shape[0],
                width=binary_map.shape[1],
                count=2,
                dtype=np.float32,
                crs=geo_meta.get("crs"),
                transform=geo_meta.get("transform"),
            ) as dst:
                dst.write(prob_map, 1)
                dst.write(binary_map.astype(np.float32) / 255.0, 2)
                dst.update_tags(
                    BAND1="change_probability",
                    BAND2="binary_change_mask",
                    SOURCE="OrbitalDelta",
                )
            logger.info(f"Saved georeferenced GeoTIFF: {output_path}")
            return
        except Exception as e:
            logger.warning(f"GeoTIFF save failed ({e}), falling back to PNG")
            output_path = output_path.with_suffix(".png")

    # Default: save binary map as PNG
    Image.fromarray(binary_map, mode="L").save(output_path)
    # Also save probability map
    prob_path = output_path.with_stem(output_path.stem + "_prob")
    prob_uint8 = (prob_map * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(prob_uint8, mode="L").save(prob_path)
    logger.info(f"Saved binary map: {output_path}")
    logger.info(f"Saved probability map: {prob_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OrbitalDelta: predict change between two images")
    p.add_argument("--img-a", required=True, help="Path to time-1 image (PNG/TIFF/JPEG)")
    p.add_argument("--img-b", required=True, help="Path to time-2 image (PNG/TIFF/JPEG)")
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    p.add_argument("--output", default="outputs/change_map.png", help="Output file path")
    p.add_argument("--config", default=None, help="Optional YAML config for model settings")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold [0,1]")
    p.add_argument("--tile-size", type=int, default=256, help="Tile size for large images")
    p.add_argument("--overlap", type=int, default=64, help="Tile overlap for blending")
    p.add_argument("--geotiff", action="store_true", help="Save georeferenced GeoTIFF output")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load checkpoint
    state = torch.load(args.checkpoint, map_location=device)

    # Determine encoder from checkpoint or config
    encoder = "resnet18"
    if "config" in state:
        encoder = state["config"].get("model", {}).get("encoder", encoder)
    elif args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        encoder = cfg.get("model", {}).get("encoder", encoder)

    model = SiameseUNet(encoder_name=encoder, pretrained=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    params = model.count_parameters()
    logger.info(f"Model: {encoder} | {params['total_M']}M params")

    # Load images
    logger.info(f"Loading images...")
    img_a, meta_a = load_image(args.img_a)
    img_b, meta_b = load_image(args.img_b)
    logger.info(f"  Image A: {img_a.shape} ({Path(args.img_a).name})")
    logger.info(f"  Image B: {img_b.shape} ({Path(args.img_b).name})")

    # Validate resolution match
    if img_a.shape[:2] != img_b.shape[:2]:
        logger.warning(
            f"Image sizes differ: A={img_a.shape[:2]}, B={img_b.shape[:2]}. "
            "Auto-resizing B to match A."
        )
        pil_b = Image.fromarray(img_b).resize(
            (img_a.shape[1], img_a.shape[0]), Image.BILINEAR
        )
        img_b = np.array(pil_b)

    # Run inference
    logger.info("Running inference...")
    prob_map, binary_map = run_inference(
        model, img_a, img_b, device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geo_meta = meta_a if meta_a else meta_b
    save_output(prob_map, binary_map, output_path, geo_meta, save_geotiff=args.geotiff)

    # Stats
    change_pct = (binary_map > 0).mean() * 100
    print("\n" + "=" * 50)
    print("  PREDICTION COMPLETE")
    print("=" * 50)
    print(f"  Input:       {img_a.shape[1]}x{img_a.shape[0]} px")
    print(f"  Change area: {change_pct:.2f}% of image")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Output:      {output_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
