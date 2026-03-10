"""
SQLAlchemy / GeoAlchemy2 ORM model for change detections.

This module is used by the PostGIS backend (src/storage/postgis.py).
The GeoPackage backend (src/storage/geopackage.py) does NOT use SQLAlchemy —
it works with plain GeoDataFrames via geopandas/fiona.

Environment variable DATABASE_URL must be set to activate the PostGIS backend,
e.g.::

    DATABASE_URL=postgresql://user:pass@localhost:5432/orbitaldelta

If the variable is absent, all storage defaults to GeoPackage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

try:
    from geoalchemy2 import Geometry
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        String,
        Text,
    )
    from sqlalchemy.orm import DeclarativeBase

    _GEO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GEO_AVAILABLE = False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


if _GEO_AVAILABLE:

    class Base(DeclarativeBase):
        pass

    class ChangeDetection(Base):
        """
        Persistent record of one change detection polygon.

        Columns
        -------
        id           : Auto-increment integer primary key
        detection_id : Globally unique UUID for this detection
        geometry     : PostGIS geometry (POLYGON, EPSG:4326)
        timestamp_a  : ISO-8601 date/time of the earlier image
        timestamp_b  : ISO-8601 date/time of the later image
        area_m2      : Polygon area in square metres
        centroid_lon : Centroid longitude (EPSG:4326)
        centroid_lat : Centroid latitude (EPSG:4326)
        confidence   : Mean model confidence score [0, 1]
        source_dataset : Optional label (e.g. "levir-cd")
        created_at   : UTC timestamp of record insertion
        """

        __tablename__ = "change_detections"

        id = Column(Integer, primary_key=True, autoincrement=True)
        detection_id = Column(
            String(36),
            unique=True,
            nullable=False,
            default=lambda: str(uuid.uuid4()),
        )
        geometry = Column(
            Geometry(geometry_type="POLYGON", srid=4326),
            nullable=False,
        )
        timestamp_a = Column(Text, nullable=True)
        timestamp_b = Column(Text, nullable=True)
        area_m2 = Column(Float, nullable=True)
        centroid_lon = Column(Float, nullable=True)
        centroid_lat = Column(Float, nullable=True)
        confidence = Column(Float, nullable=True)
        source_dataset = Column(Text, nullable=True)
        created_at = Column(
            DateTime(timezone=True),
            nullable=False,
            default=_utcnow,
        )

        def __repr__(self) -> str:
            return (
                f"<ChangeDetection id={self.id} "
                f"area_m2={self.area_m2:.1f} "
                f"conf={self.confidence:.3f}>"
            )

else:  # pragma: no cover
    # Stub so the module can always be imported even without SQLAlchemy/GeoAlchemy2
    class Base:  # type: ignore[no-redef]
        pass

    class ChangeDetection:  # type: ignore[no-redef]
        """Stub — install sqlalchemy geoalchemy2 for full support."""
        pass
