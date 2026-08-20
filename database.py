"""
Database access for RACZoneGen.

Reads RAC health events and autonomous lane geometry from the MineStar
SQL Server instance (mshist / msmodel) over the ODBC driver.

Speed extraction: the MineStar payload for these health events contains
a speech-speed value for the vehicle.  We try several source formats in
order and fall back to a configurable default when it cannot be parsed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("rac_zone_monitor")


def read_sql_file(path: Path, time_cutoff_minutes: int) -> str:
    """Read SQL and substitute the configurable time-cutoff token."""
    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")

    query = path.read_text(encoding="utf-8-sig")

    query = query.replace(
        "{time_cutoff_minutes}",
        str(time_cutoff_minutes),
    )

    return query.strip()


def normalize_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """Normalize SQL column names without case sensitivity."""
    actual_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    rename_map: dict[object, str] = {}

    for required in required_columns:
        actual = actual_columns.get(required.lower())

        if actual is None:
            raise ValueError(
                f"SQL result is missing required column: {required}. "
                f"Returned columns: {list(dataframe.columns)}"
            )

        rename_map[actual] = required

    return dataframe.rename(columns=rename_map)


def _payload_speed(payload: object) -> float | None:
    """
    Attempt to pull a speed (km/h) out of a RAC event payload.

    MineStar stores the payload for health events in several shapes:
    JSON, a naive float string, or "speed=<value>" style text.  This
    tries each in turn.
    """
    if payload is None:
        return None

    if isinstance(payload, (int, float, np.integer, np.floating)):
        value = float(payload)
        return value if np.isfinite(value) and value > 0 else None

    text_value = str(payload).strip()

    if not text_value:
        return None

    try:
        parsed_json = json.loads(text_value)
    except (ValueError, TypeError):
        parsed_json = None

    if isinstance(parsed_json, dict):
        for key in ("Speed", "speed", "SPEED", "VehicleSpeed", "machSpeed"):
            candidate = parsed_json.get(key)
            if candidate is not None:
                try:
                    value = float(candidate)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value

    try:
        value = float(text_value)
        return value if np.isfinite(value) and value > 0 else None
    except ValueError:
        pass

    import re

    match = re.search(
        r"(?i)speed[\"'=:=]\s*([\d.]+)",
        text_value.replace("\\", ""),
    )

    if match:
        try:
            value = float(match.group(1))
            return value if np.isfinite(value) and value > 0 else None
        except ValueError:
            return None

    return None


class Database:
    """Wraps one SQLAlchemy engine with typed query helpers."""

    def __init__(
        self,
        connection_string: str,
        query_timeout: int = 60,
    ) -> None:
        self.engine = create_engine(
            "mssql+pyodbc:///?odbc_connect="
            + quote_plus(connection_string),
            pool_pre_ping=True,
            pool_recycle=300,
            connect_args={
                "timeout": query_timeout,
            },
        )

    def dispose(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # RAC events
    # ------------------------------------------------------------------

    def read_rac_events(
        self,
        query_file: Path,
        time_cutoff_minutes: int,
    ) -> pd.DataFrame:
        """Read and validate recent RAC events, adding a Speed column."""
        query = read_sql_file(query_file, time_cutoff_minutes)

        with self.engine.connect() as connection:
            events = pd.read_sql_query(query, connection)

        events = normalize_columns(
            events,
            ["Time", "X", "Y", "Level", "Payload"],
        )

        events["Time"] = pd.to_datetime(
            events["Time"],
            errors="coerce",
        )

        for column in ("X", "Y", "Level"):
            events[column] = pd.to_numeric(
                events[column],
                errors="coerce",
            )

        events["Speed"] = events["Payload"].map(_payload_speed)

        events = events.dropna(
            subset=["Time", "X", "Y", "Level"]
        ).copy()

        events = events.reset_index(drop=True)

        return events

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------

    def read_lanes(
        self,
        query_file: Path,
    ) -> pd.DataFrame:
        """Read active autonomous lane edge points."""
        query = read_sql_file(query_file, time_cutoff_minutes=0)

        with self.engine.connect() as connection:
            lanes = pd.read_sql_query(query, connection)

        lanes = normalize_columns(
            lanes,
            ["LANE_OID", "Segment", "POINT_NR", "X", "Y"],
        )

        for column in ("POINT_NR", "X", "Y"):
            lanes[column] = pd.to_numeric(
                lanes[column],
                errors="coerce",
            )

        lanes = lanes.dropna(
            subset=["LANE_OID", "Segment", "POINT_NR", "X", "Y"]
        ).copy()

        logger.info(
            "Loaded %d lane edge points from %d active lanes",
            len(lanes),
            lanes["LANE_OID"].nunique(),
        )

        return lanes

    # ------------------------------------------------------------------
    # Existing zones (direct DB read — avoids the slow mstarrun export)
    # ------------------------------------------------------------------

    def read_existing_zones(
        self,
        query_file: Path,
        name_prefix: str,
    ) -> list["ExistingZone"]:
        """
        Read active MineStar zones directly from `msmodel.dbo.ZONE`.

        This is dramatically faster than invoking `mstarrun exportZones`
        (which starts a full MineStar client).  The query file returns
        the envelope bounding box of every active zone; we filter to the
        configured name prefix and reduce each zone to a centre + half
        extent.
        """
        from minestar import ExistingZone

        if not query_file.is_file():
            raise FileNotFoundError(
                f"Zones SQL file not found: {query_file}"
            )

        query = read_sql_file(query_file, time_cutoff_minutes=0)

        with self.engine.connect() as connection:
            zones_frame = pd.read_sql_query(query, connection)

        zones_frame = normalize_columns(
            zones_frame,
            ["Name", "MinX", "MaxX", "MinY", "MaxY"],
        )

        for column in ("MinX", "MaxX", "MinY", "MaxY"):
            zones_frame[column] = pd.to_numeric(
                zones_frame[column],
                errors="coerce",
            )

        zones: list[ExistingZone] = []

        for _, row in zones_frame.iterrows():
            name = str(row["Name"]).strip() if pd.notna(row["Name"]) else ""

            if not name.upper().startswith(name_prefix.upper()):
                continue

            min_x = float(row["MinX"])
            max_x = float(row["MaxX"])
            min_y = float(row["MinY"])
            max_y = float(row["MaxY"])

            if any(
                not np.isfinite(value)
                for value in (min_x, max_x, min_y, max_y)
            ):
                continue

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            half_size = max(max_x - min_x, max_y - min_y) / 2.0

            zones.append(
                ExistingZone(
                    name=name,
                    center_x=center_x,
                    center_y=center_y,
                    half_size=half_size,
                )
            )

        logger.info(
            "Read %d active %s* zone(s) directly from MineStar DB",
            len(zones),
            name_prefix,
        )

        return zones