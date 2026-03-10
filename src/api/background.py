"""
Background pipeline — runs the full processing chain in a FastAPI BackgroundTask.

Pipeline steps:
  1. Load images (GeoReader supports GeoTIFF + plain images)
  2. Image registration (ORB/SIFT + RANSAC)
  3. Tile large images
  4. Model inference (Siamese U-Net)
  5. Stitch tiles back
  6. Post-process: connected components → polygonize → attributes
  7. Persist to spatial store (GeoPackage by default)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# In-memory job registry  {job_id: {"status": str, "message": str, "result": dict | None}}
_jobs: dict[str, dict] = {}


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def submit_job(request_data: dict) -> str:
    """Register a job and return its UUID (without running it yet)."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "message": "", "result": None}
    return job_id


def run_pipeline(job_id: str, req: dict, store_path: str = "data/detections.gpkg") -> None:
    """
    Full pipeline executed as a FastAPI BackgroundTask.

    Parameters
    ----------
    job_id      : UUID assigned at submission
    req         : Dict from DetectRequest model
    store_path  : Path to the GeoPackage file
    """
    _jobs[job_id]["status"] = "running"
    try:
        result = _execute(req, store_path)
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["result"] = result
    except Exception as exc:
        logger.exception(f"Pipeline failed for job {job_id}: {exc}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["message"] = str(exc)


def _execute(req: dict, store_path: str) -> dict:
    """Core processing logic (raising exceptions on failure)."""
    from src.geospatial.reader import GeoReader
    from src.registration.warping import align_images
    from src.tiling.splitter import TileSplitter
    from src.tiling.stitcher import TileStitcher
    from src.postprocessing.connected_components import extract_regions
    from src.postprocessing.polygonizer import mask_to_polygons
    from src.postprocessing.attributes import compute_attributes
    from src.storage.geopackage import GeoPackageStore

    img_a_path = req["img_a_path"]
    img_b_path = req["img_b_path"]
    timestamp_a = req.get("timestamp_a", "")
    timestamp_b = req.get("timestamp_b", "")
    checkpoint = req.get("checkpoint", "checkpoints/best.pt")

    # ── 1. Load images ────────────────────────────────────────────────
    logger.info(f"Loading images: {img_a_path}, {img_b_path}")
    arr_a, meta_a = GeoReader.read(img_a_path)
    arr_b, meta_b = GeoReader.read(img_b_path)

    # Convert (C, H, W) float32 → (H, W, C) uint8 for OpenCV
    def to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
        hwc = arr.transpose(1, 2, 0)
        return (np.clip(hwc, 0, 1) * 255).astype(np.uint8)

    img_a_u8 = to_hwc_uint8(arr_a)
    img_b_u8 = to_hwc_uint8(arr_b)

    # ── 2. Registration ───────────────────────────────────────────────
    logger.info("Registering images…")
    aligned_b_u8, error_px, n_inliers = align_images(
        img_a_u8, img_b_u8, max_error=5.0
    )
    logger.info(f"Registration: error={error_px:.2f}px, inliers={n_inliers}")

    # Convert back to float (C, H, W)
    aligned_b = (aligned_b_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)

    # ── 3–4. Tiling + Inference ───────────────────────────────────────
    logger.info("Running tiled inference…")
    prob_map = _tiled_inference(arr_a, aligned_b, checkpoint)

    # ── 5. Threshold to binary mask ───────────────────────────────────
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    # ── 6. Post-process ───────────────────────────────────────────────
    logger.info("Extracting polygons…")
    transform = meta_a.transform if meta_a.has_geo else None
    crs = meta_a.crs if meta_a.has_geo else None

    polygons = mask_to_polygons(binary_mask, min_area_px=25, transform=transform)
    n_polys = len(polygons)
    logger.info(f"Detected {n_polys} change regions")

    # ── 7. Compute attributes + persist ───────────────────────────────
    if n_polys > 0:
        gdf = compute_attributes(
            polygons, prob_map,
            transform=transform, crs=crs,
            timestamp_a=timestamp_a, timestamp_b=timestamp_b,
        )
        store = GeoPackageStore(store_path)
        store.insert(gdf)

    return {
        "n_changes": n_polys,
        "registration_error_px": round(error_px, 2),
        "inliers": n_inliers,
    }


def _tiled_inference(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    checkpoint: str,
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """
    Run model inference on potentially large image pairs using tile split/stitch.

    Returns a (H, W) float32 probability map.
    """
    import torch
    from src.models.siamese_unet import SiameseUNet
    from src.tiling.splitter import TileSplitter
    from src.tiling.stitcher import TileStitcher

    # ── Load model ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseUNet("resnet18", pretrained=False)

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info(f"Loaded checkpoint: {checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint} — using random weights")

    model.eval().to(device)

    _, h, w = arr_a.shape
    splitter = TileSplitter(tile_size=tile_size, overlap=overlap)
    stitcher = TileStitcher(tile_size=tile_size, overlap=overlap, output_shape=(h, w))

    # Split both images into matching tile pairs
    pairs = splitter.split_pair(
        arr_a.transpose(1, 2, 0),  # HWC
        arr_b.transpose(1, 2, 0),
    )

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for rec_a, rec_b in pairs:
            # Normalise: (H, W, C) float32 → (1, C, H, W) tensor
            ta = _to_tensor(rec_a.tile, IMAGENET_MEAN, IMAGENET_STD, device)
            tb = _to_tensor(rec_b.tile, IMAGENET_MEAN, IMAGENET_STD, device)

            prob = model(ta, tb).squeeze().cpu().numpy()  # (tile_H, tile_W)
            stitcher.add_tile(prob, rec_a.row, rec_a.col)

    return stitcher.stitch()


def _to_tensor(tile_hwc: np.ndarray, mean, std, device) -> "torch.Tensor":
    """(H, W, C) float32 → (1, C, H, W) normalised tensor."""
    import torch
    t = (tile_hwc.astype(np.float32) - mean) / std   # (H, W, C)
    t = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)  # (1, C, H, W)
    return t.to(device)
