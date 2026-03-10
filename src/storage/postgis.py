"""
PostGIS spatial storage backend (optional).

Activated only when the DATABASE_URL environment variable is set to a
PostgreSQL + PostGIS connection string.
Falls back silently to GeoPackageStore when DATABASE_URL is absent.

Same public interface as GeoPackageStore (duck typing):
    insert(gdf), query_all(), query_bbox(xmin,ymin,xmax,ymax), get_by_id(id)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)


def get_store(default_gpkg_path: str = "data/detections.gpkg"):
    """
    Factory that returns the appropriate store backend.

    If DATABASE_URL is set → PostGISStore.
    Otherwise           → GeoPackageStore (zero infrastructure).
    """
    if os.environ.get("DATABASE_URL"):
        logger.info("DATABASE_URL found — using PostGIS backend")
        return PostGISStore(os.environ["DATABASE_URL"])

    from src.storage.geopackage import GeoPackageStore
    logger.info("No DATABASE_URL — using GeoPackage backend")
    return GeoPackageStore(default_gpkg_path)


class PostGISStore:
    """
    Spatial storage backed by PostgreSQL / PostGIS.

    Requires:  psycopg2-binary, SQLAlchemy, geopandas, GeoAlchemy2
    Activated: DATABASE_URL env var (e.g. postgresql://user:pw@host/db)

    Parameters
    ----------
    url : SQLAlchemy database URL
    """

    _TABLE = "change_detections"

    def __init__(self, url: str) -> None:
        self._url = url
        self._engine = self._make_engine(url)
        self._ensure_table()

    # ------------------------------------------------------------------
    # Public interface (same as GeoPackageStore)
    # ------------------------------------------------------------------

    def insert(self, gdf: Any) -> None:
        """Insert a GeoDataFrame of change detections into PostGIS."""
        import pandas as pd
        gdf = gdf.copy()
        if "detection_id" not in gdf.columns:
            gdf["detection_id"] = [str(uuid.uuid4()) for _ in range(len(gdf))]
        if "created_at" not in gdf.columns:
            gdf["created_at"] = pd.Timestamp.utcnow().isoformat()

        gdf.to_postgis(
            self._TABLE,
            self._engine,
            if_exists="append",
            index=False,
        )
        logger.info(f"Inserted {len(gdf)} detections into PostGIS/{self._TABLE}")

    def query_all(self) -> Any:
        """Return all detections from PostGIS."""
        import geopandas as gpd
        return gpd.read_postgis(
            f"SELECT * FROM {self._TABLE}",
            self._engine,
            geom_col="geometry",
        )

    def query_bbox(
        self, xmin: float, ymin: float, xmax: float, ymax: float
    ) -> Any:
        """Spatial query using PostGIS ST_Intersects."""
        import geopandas as gpd
        sql = (
            f"SELECT * FROM {self._TABLE} "
            f"WHERE ST_Intersects(geometry, "
            f"ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, 4326))"
        )
        return gpd.read_postgis(sql, self._engine, geom_col="geometry")

    def get_by_id(self, detection_id: str) -> Any:
        """Return single detection by UUID."""
        import geopandas as gpd
        sql = (
            f"SELECT * FROM {self._TABLE} "
            f"WHERE detection_id = '{detection_id}'"
        )
        return gpd.read_postgis(sql, self._engine, geom_col="geometry")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_engine(self, url: str):
        from sqlalchemy import create_engine
        return create_engine(url)

    def _ensure_table(self) -> None:
        """Create the PostGIS table if it does not exist."""
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS postgis;
        CREATE TABLE IF NOT EXISTS {self._TABLE} (
            id              SERIAL PRIMARY KEY,
            detection_id    TEXT UNIQUE,
            geometry        GEOMETRY(Polygon, 4326),
            area_m2         DOUBLE PRECISION,
            centroid_x      DOUBLE PRECISION,
            centroid_y      DOUBLE PRECISION,
            bbox_minx       DOUBLE PRECISION,
            bbox_miny       DOUBLE PRECISION,
            bbox_maxx       DOUBLE PRECISION,
            bbox_maxy       DOUBLE PRECISION,
            perimeter_m     DOUBLE PRECISION,
            mean_confidence DOUBLE PRECISION,
            timestamp_a     TEXT,
            timestamp_b     TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS {self._TABLE}_geom_idx
            ON {self._TABLE} USING GIST(geometry);
        """
        from sqlalchemy import text
        with self._engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()
        logger.info(f"PostGIS table '{self._TABLE}' ready")
