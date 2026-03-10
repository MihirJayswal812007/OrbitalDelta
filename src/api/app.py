"""
OrbitalDelta FastAPI application.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Or via the serve script:
    python scripts/serve.py
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OrbitalDelta — Satellite Change Detection API",
    description=(
        "REST API for geospatial change detection using Siamese U-Net. "
        "Submit image pairs, query detected changes by bounding box, "
        "and visualise results on an interactive Leaflet map."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow browser clients on any origin during development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files (CSS, JS for the map viewer)
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent.parent / "web" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
app.include_router(router)

# ---------------------------------------------------------------------------
# Root: serve the Leaflet map viewer
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def map_viewer():
    """Serve the Leaflet-based visualization interface."""
    template_path = Path(__file__).parent.parent / "web" / "templates" / "map.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(content=_fallback_html(), status_code=200)


def _fallback_html() -> str:
    """Minimal HTML served when web/templates/map.html does not exist yet."""
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>OrbitalDelta</title></head>
<body style="font-family:monospace;padding:2rem;background:#111;color:#0f0">
  <h1>OrbitalDelta API is running</h1>
  <p>Map viewer not yet built. API docs: <a href="/docs" style="color:#0f0">/docs</a></p>
</body>
</html>"""
