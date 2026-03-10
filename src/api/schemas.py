"""
Pydantic schemas for the OrbitalDelta REST API.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    """Submit a new image pair for change detection."""
    img_a_path: str = Field(..., description="Absolute or relative path to Time-1 image")
    img_b_path: str = Field(..., description="Absolute or relative path to Time-2 image")
    timestamp_a: str = Field(default="", description="ISO-8601 date of image A (optional)")
    timestamp_b: str = Field(default="", description="ISO-8601 date of image B (optional)")
    checkpoint: str = Field(
        default="checkpoints/best.pt",
        description="Path to model checkpoint (.pt file)",
    )


class BBoxQuery(BaseModel):
    """Spatial bounding box query parameters."""
    xmin: float = Field(..., description="Western longitude / X-min (CRS units)")
    ymin: float = Field(..., description="Southern latitude / Y-min (CRS units)")
    xmax: float = Field(..., description="Eastern longitude / X-max (CRS units)")
    ymax: float = Field(..., description="Northern latitude / Y-max (CRS units)")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class JobStatus(BaseModel):
    """Response returned immediately after submitting a detection job."""
    job_id: str = Field(..., description="Unique UUID for this processing job")
    status: str = Field(..., description="'queued' | 'running' | 'complete' | 'failed'")
    message: str = Field(default="")


class DetectionResult(BaseModel):
    """A single detected change region."""
    detection_id: str
    geometry_geojson: dict[str, Any] = Field(..., description="GeoJSON geometry of the polygon")
    area_m2: float
    centroid_x: float
    centroid_y: float
    bbox: dict[str, float] = Field(..., description="{minx, miny, maxx, maxy}")
    mean_confidence: float
    timestamp_a: str = ""
    timestamp_b: str = ""
    created_at: str = ""


class DetectionList(BaseModel):
    """Paginated list of detection results."""
    count: int
    detections: list[DetectionResult]


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str
    code: int = 500
