"""
GeoPackage spatial storage backend.

Zero-infrastructure spatial database using OGC GeoPackage (SQLite-based),
accessed through geopandas + fiona.  Works out of the box on any OS without
a server process.  Same interface as PostGISStore so backends are swappable.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Layer name inside the GeoPackage file
LAYER = "change_detections"


class GeoPackageStore:
    """
    Spatial storage backed by a GeoPackage (.gpkg) file.

    Supports insert, query_all, query_bbox, and get_by_id.

    Parameters
    ----------
    path : Path or str to the .gpkg file (created if it does not exist)
    crs  : Default CRS for new tables (EPSG code or rasterio CRS object)
    """

    def __init__(self, path: str | Path, crs: Any = "EPSG:4326") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.crs = crs
        self._ensure_layer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, gdf: "geopandas.GeoDataFrame") -> None:  # noqa: F821
        """
        Insert a GeoDataFrame of new change detections.

        A unique ``detection_id`` (UUID) and ``created_at`` timestamp are
        added if not already present.

        Parameters
        ----------
        gdf : GeoDataFrame with at minimum a ``geometry`` column
        """
        import geopandas as gpd
        import pandas as pd

        gdf = gdf.copy()
        if "detection_id" not in gdf.columns:
            gdf["detection_id"] = [str(uuid.uuid4()) for _ in range(len(gdf))]
        if "created_at" not in gdf.columns:
            gdf["created_at"] = pd.Timestamp.utcnow().isoformat()

        if gdf.crs is None:
            gdf = gdf.set_crs(self.crs)

        # Read existing rows, concatenate with new ones, then rewrite the
        # whole layer. pyogrio mode='a' can misalign columns when the schema
        # differs from the on-disk layer, so a full rewrite is safer.
        existing = self._read()
        import pandas as pd
        if existing.empty:
            combined = gdf
        else:
            import geopandas as gpd  # noqa: F811
            combined = gpd.GeoDataFrame(
                pd.concat([existing, gdf], ignore_index=True),
                crs=gdf.crs,
            )

        combined.to_file(
            self.path,
            layer=LAYER,
            driver="GPKG",
            mode="w",          # always full overwrite — safe & correct
            engine="pyogrio",
        )
        logger.info(f"Inserted {len(gdf)} detections into {self.path.name}")

    def query_all(self) -> "geopandas.GeoDataFrame":  # noqa: F821
        """Return all stored change detections as a GeoDataFrame."""
        return self._read()

    def query_bbox(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ) -> "geopandas.GeoDataFrame":  # noqa: F821
        """
        Spatial query — return detections intersecting the given bounding box.

        Parameters
        ----------
        xmin, ymin, xmax, ymax : Bounding box in the store's CRS
        """
        from shapely.geometry import box
        gdf = self._read()
        if gdf.empty:
            return gdf
        bbox_geom = box(xmin, ymin, xmax, ymax)
        mask = gdf.geometry.intersects(bbox_geom)
        return gdf[mask].copy()

    def get_by_id(self, detection_id: str) -> "geopandas.GeoDataFrame":  # noqa: F821
        """Return the single row matching *detection_id*, or empty GeoDataFrame."""
        gdf = self._read()
        if gdf.empty or "detection_id" not in gdf.columns:
            return gdf.iloc[0:0]
        return gdf[gdf["detection_id"] == detection_id].copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> "geopandas.GeoDataFrame":  # noqa: F821
        """Read the change_detections layer; return empty GDF if it doesn't exist."""
        import geopandas as gpd
        if not self.path.exists() or not self._layer_exists():
            return gpd.GeoDataFrame(
                columns=["geometry", "detection_id", "area_m2", "confidence"],
                crs=self.crs,
            )
        return gpd.read_file(self.path, layer=LAYER, engine="pyogrio")

    def _layer_exists(self) -> bool:
        """Check whether the layer already exists in the file."""
        if not self.path.exists():
            return False
        try:
            import geopandas as gpd
            from fiona import listlayers
            return LAYER in listlayers(str(self.path))
        except Exception:
            return False

    def _ensure_layer(self) -> None:
        """Create an empty layer on first use so the file is valid."""
        if not self._layer_exists():
            import geopandas as gpd
            from shapely.geometry import Point
            # Write a minimal placeholder and immediately drop it
            placeholder = gpd.GeoDataFrame(
                {"geometry": [], "detection_id": [], "area_m2": []},
                crs=self.crs,
            )
            placeholder.to_file(
                self.path, layer=LAYER, driver="GPKG", mode="w", engine="pyogrio"
            )
            logger.debug(f"Initialised GeoPackage layer '{LAYER}' at {self.path}")
