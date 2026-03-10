"""
API route definitions for OrbitalDelta REST service.

Endpoints:
  POST /api/v1/detect           — submit image pair for processing
  GET  /api/v1/jobs/{job_id}    — check job status
  GET  /api/v1/detections       — list all stored detections
  GET  /api/v1/detections/{id}  — retrieve a single detection
  POST /api/v1/detections/query — spatial bounding-box query
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.api.schemas import (
    BBoxQuery,
    DetectRequest,
    DetectionList,
    DetectionResult,
    JobStatus,
)
from src.api import background as bg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["detections"])

# Path to the default GeoPackage store
_STORE_PATH = os.environ.get("GPKG_PATH", "data/detections.gpkg")


# ---------------------------------------------------------------------------
# Detection submission
# ---------------------------------------------------------------------------

@router.post(
    "/detect",
    response_model=JobStatus,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an image pair for change detection",
)
async def submit_detection(
    req: DetectRequest,
    background_tasks: BackgroundTasks,
) -> JobStatus:
    """
    Enqueue an image pair for asynchronous processing.

    The pipeline runs in the background — poll ``GET /api/v1/jobs/{job_id}``
    to check completion before querying results.
    """
    # Validate input paths exist
    for p, label in [(req.img_a_path, "img_a_path"), (req.img_b_path, "img_b_path")]:
        if not Path(p).exists():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{label} does not exist: {p}",
            )

    job_id = bg.submit_job(req.model_dump())
    background_tasks.add_task(
        bg.run_pipeline, job_id, req.model_dump(), _STORE_PATH
    )
    logger.info(f"Job {job_id} queued for {req.img_a_path} / {req.img_b_path}")
    return JobStatus(job_id=job_id, status="queued", message="Processing in background")


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Check processing job status",
)
async def get_job_status(job_id: str) -> JobStatus:
    """Poll the status of a submitted detection job."""
    job = bg.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message", ""),
    )


# ---------------------------------------------------------------------------
# Detection queries
# ---------------------------------------------------------------------------

@router.get(
    "/detections",
    response_model=DetectionList,
    summary="List all stored change detections",
)
async def list_detections() -> DetectionList:
    """Return all change detections from the spatial store."""
    store = _get_store()
    gdf = store.query_all()
    return _gdf_to_list(gdf)


@router.get(
    "/detections/{detection_id}",
    response_model=DetectionResult,
    summary="Retrieve a single detection by ID",
)
async def get_detection(detection_id: str) -> DetectionResult:
    """Return the detection with the given UUID."""
    store = _get_store()
    gdf = store.get_by_id(detection_id)
    if gdf.empty:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id!r} not found")
    return _row_to_result(gdf.iloc[0])


@router.post(
    "/detections/query",
    response_model=DetectionList,
    summary="Spatial bounding-box query",
)
async def query_detections_bbox(q: BBoxQuery) -> DetectionList:
    """Return detections intersecting the supplied bounding box."""
    store = _get_store()
    gdf = store.query_bbox(q.xmin, q.ymin, q.xmax, q.ymax)
    return _gdf_to_list(gdf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_store():
    from src.storage.geopackage import GeoPackageStore
    return GeoPackageStore(_STORE_PATH)


def _gdf_to_list(gdf) -> DetectionList:
    """Convert a GeoDataFrame to DetectionList schema."""
    results = [_row_to_result(row) for _, row in gdf.iterrows()]
    return DetectionList(count=len(results), detections=results)


def _row_to_result(row) -> DetectionResult:
    """Convert a single GeoDataFrame row to DetectionResult."""
    import json
    geom = row.get("geometry")
    geojson = json.loads(geom.to_json()) if geom and hasattr(geom, "to_json") else {}

    return DetectionResult(
        detection_id=str(row.get("detection_id", "")),
        geometry_geojson=geojson,
        area_m2=float(row.get("area_m2", 0)),
        centroid_x=float(row.get("centroid_x", 0)),
        centroid_y=float(row.get("centroid_y", 0)),
        bbox={
            "minx": float(row.get("bbox_minx", 0)),
            "miny": float(row.get("bbox_miny", 0)),
            "maxx": float(row.get("bbox_maxx", 0)),
            "maxy": float(row.get("bbox_maxy", 0)),
        },
        mean_confidence=float(row.get("mean_confidence", 0)),
        timestamp_a=str(row.get("timestamp_a", "")),
        timestamp_b=str(row.get("timestamp_b", "")),
        created_at=str(row.get("created_at", "")),
    )
