"""
Full end-to-end processing pipeline CLI.

Usage:
    python scripts/pipeline.py \\
        --img-a data/sample/t1.tif \\
        --img-b data/sample/t2.tif \\
        --checkpoint checkpoints/best.pt \\
        --output outputs/pipeline_test/

Also importable:
    from src.pipeline import run_pipeline
    result = run_pipeline(img_a_path, img_b_path, checkpoint, output_dir)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API (importable by other modules)
# ---------------------------------------------------------------------------

def run_pipeline(
    img_a_path: str,
    img_b_path: str,
    checkpoint: str = "checkpoints/best.pt",
    output_dir: str = "outputs/pipeline/",
    gpkg_path: str = "data/detections.gpkg",
) -> dict:
    """
    Run the complete OrbitalDelta processing pipeline.

    Steps:
      1. Load images (plain PNG or GeoTIFF)
      2. Register images (ORB + RANSAC)
      3. Run tiled Siamese U-Net inference
      4. Threshold → binary change mask
      5. Polygonize + compute attributes
      6. Save outputs: change_map.tif, detections.geojson
      7. Persist to GeoPackage spatial store

    Parameters
    ----------
    img_a_path   : Path to Time-1 image
    img_b_path   : Path to Time-2 image
    checkpoint   : Model checkpoint path
    output_dir   : Directory to write change_map.tif and detections.geojson
    gpkg_path    : GeoPackage file path for persistent storage

    Returns
    -------
    dict with keys: n_changes, registration_error_px, n_inliers, output_dir
    """
    from src.geospatial.reader import GeoReader
    from src.geospatial.writer import GeoWriter
    from src.registration.warping import align_images
    from src.postprocessing.connected_components import extract_regions
    from src.postprocessing.polygonizer import mask_to_polygons
    from src.postprocessing.attributes import compute_attributes
    from src.storage.geopackage import GeoPackageStore
    from src.api.background import _tiled_inference

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────
    logger.info(f"Loading:  A={img_a_path}")
    logger.info(f"          B={img_b_path}")
    arr_a, meta_a = GeoReader.read(img_a_path)
    arr_b, _      = GeoReader.read(img_b_path)

    def to_hwc_u8(arr):
        return (np.clip(arr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

    # ── 2. Register ───────────────────────────────────────────────────
    logger.info("Registering images…")
    aligned_b_u8, error_px, n_inliers = align_images(to_hwc_u8(arr_a), to_hwc_u8(arr_b))
    aligned_b = (aligned_b_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)
    logger.info(f"Registration: error={error_px:.2f}px  inliers={n_inliers}")

    # ── 3. Inference ──────────────────────────────────────────────────
    logger.info("Running tiled inference…")
    prob_map = _tiled_inference(arr_a, aligned_b, checkpoint)

    # ── 4. Threshold ──────────────────────────────────────────────────
    binary_mask = (prob_map > 0.5).astype(np.uint8)

    # ── 5. Save change map ────────────────────────────────────────────
    map_path = out / "change_map.tif"
    GeoWriter.write(binary_mask, meta_a, map_path, prob_map=prob_map)
    logger.info(f"Saved change map: {map_path}")

    # ── 6. Polygonize + attributes ────────────────────────────────────
    transform = meta_a.transform if meta_a.has_geo else None
    crs = meta_a.crs if meta_a.has_geo else None

    polygons = mask_to_polygons(binary_mask, min_area_px=25, transform=transform)
    n_polys = len(polygons)
    logger.info(f"Detected {n_polys} change regions")

    # ── 7. Export GeoJSON + persist ───────────────────────────────────
    geojson_path = out / "detections.geojson"
    if n_polys > 0:
        gdf = compute_attributes(
            polygons, prob_map, transform=transform, crs=crs
        )
        gdf.to_file(geojson_path, driver="GeoJSON")
        store = GeoPackageStore(gpkg_path)
        store.insert(gdf)
        logger.info(f"Saved GeoJSON: {geojson_path}")
    else:
        # Write empty GeoJSON so downstream checks pass
        geojson_path.write_text('{"type":"FeatureCollection","features":[]}')
        logger.info("No changes detected — wrote empty GeoJSON")

    result = {
        "n_changes": n_polys,
        "registration_error_px": round(float(error_px), 2),
        "n_inliers": int(n_inliers),
        "output_dir": str(out.resolve()),
    }

    # Save result summary
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    logger.info(f"Pipeline complete: {result}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="OrbitalDelta end-to-end processing pipeline"
    )
    p.add_argument("--img-a",       required=True, help="Time-1 image (PNG or GeoTIFF)")
    p.add_argument("--img-b",       required=True, help="Time-2 image (PNG or GeoTIFF)")
    p.add_argument("--checkpoint",  default="checkpoints/best.pt", help="Model checkpoint")
    p.add_argument("--output",      default="outputs/pipeline/", help="Output directory")
    p.add_argument("--gpkg",        default="data/detections.gpkg", help="GeoPackage path")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    result = run_pipeline(
        img_a_path=args.img_a,
        img_b_path=args.img_b,
        checkpoint=args.checkpoint,
        output_dir=args.output,
        gpkg_path=args.gpkg,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["n_changes"] >= 0 else 1)
