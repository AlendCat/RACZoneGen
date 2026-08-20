"""
RAC Automatic Zone Monitor

Every 30 seconds:
1. Reads RAC events from SQL Server.
2. Reads active autonomous lanes.
3. Detects RAC events not present in the previous sample.
4. Groups nearby RAC events by combined severity.
5. Exports current MineStar zones.
6. Blocks clusters near existing RAC_AUTO zones.
7. Creates a circular MineStar zone XML.
8. Optionally imports it with mstarrun.
9. Displays lanes, RAC events, groups, zones, and suppression areas.

The first RAC sample is a baseline. It never creates zones.
"""

from __future__ import annotations

import configparser
import copy
import logging
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    center_x: float
    center_y: float
    search_x: float
    search_y: float
    score: float
    event_count: int
    new_event_count: int


@dataclass
class ExistingZone:
    name: str
    center_x: float
    center_y: float
    radius: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def application_directory() -> Path:
    """Return the script or packaged executable directory."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parent


BASE_DIR = application_directory()
CONFIG_FILE = BASE_DIR / "config.ini"

config = configparser.ConfigParser(interpolation=None)

if not config.read(CONFIG_FILE, encoding="utf-8-sig"):
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_FILE}"
    )


def configured_path(section: str, option: str) -> Path:
    """Resolve a configured path relative to the application folder."""

    configured_value = config.get(section, option).strip()
    path = Path(os.path.expandvars(configured_value))

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


POLL_SECONDS = config.getint(
    "application",
    "poll_interval_seconds",
)

CONNECTION_STRING = config.get(
    "database",
    "odbc_connection_string",
)

RAC_QUERY_FILE = configured_path("rac", "query_file")
TIME_CUTOFF = config.getint("rac", "time_cutoff_minutes")
GROUP_RADIUS = config.getfloat("rac", "grouping_radius_metres")
GROUP_SCORE = config.getfloat("rac", "grouping_score")
MINIMUM_EVENTS = config.getint("rac", "minimum_event_count")
TOP_N = config.getint("rac", "top_n")
INCLUSIVE_THRESHOLD = config.getboolean(
    "rac",
    "threshold_inclusive",
)

LANE_QUERY_FILE = configured_path("lanes", "query_file")

ZONE_TEMPLATE = configured_path("zone", "template_file")
ZONE_OUTPUT = configured_path("zone", "output_directory")
ZONE_PREFIX = config.get("zone", "name_prefix").strip()
ZONE_RADIUS = config.getfloat("zone", "radius_metres")
ZONE_POINTS = config.getint("zone", "polygon_points")
ZONE_ELEVATION = config.getfloat("zone", "elevation")
SUPPRESSION_RADIUS = config.getfloat(
    "zone",
    "suppression_radius_metres",
)

MSTAR_DIRECTORY = configured_path(
    "minestar",
    "mstar_bin_directory",
)
MSTAR_EXECUTABLE = config.get(
    "minestar",
    "executable",
).strip()
CURRENT_ZONES_FILE = configured_path(
    "minestar",
    "current_zones_file",
)
COMMAND_TIMEOUT = config.getint(
    "minestar",
    "timeout_seconds",
)
IMPORT_ENABLED = config.getboolean(
    "minestar",
    "import_enabled",
)
DRY_RUN = config.getboolean(
    "minestar",
    "dry_run",
)
ALLOW_OUTSIDE = config.getboolean(
    "minestar",
    "allow_outside_mine_boundary",
)

DISPLAY_ENABLED = config.getboolean(
    "display",
    "enabled",
)


if POLL_SECONDS < 1:
    raise ValueError("poll_interval_seconds must be at least 1")

if GROUP_RADIUS <= 0:
    raise ValueError("grouping_radius_metres must be greater than 0")

if MINIMUM_EVENTS < 1:
    raise ValueError("minimum_event_count must be at least 1")

if ZONE_RADIUS <= 0:
    raise ValueError("radius_metres must be greater than 0")

if ZONE_POINTS < 8:
    raise ValueError("polygon_points must be at least 8")

if SUPPRESSION_RADIUS < ZONE_RADIUS:
    raise ValueError(
        "suppression_radius_metres must be at least as large "
        "as radius_metres"
    )


# ---------------------------------------------------------------------------
# Logging and folders
# ---------------------------------------------------------------------------

LOG_DIRECTORY = BASE_DIR / "logs"

LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
ZONE_OUTPUT.mkdir(parents=True, exist_ok=True)
CURRENT_ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rac_zone_monitor")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = RotatingFileHandler(
    LOG_DIRECTORY / "rac_zone_monitor.log",
    maxBytes=5_000_000,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

engine = create_engine(
    "mssql+pyodbc:///?odbc_connect="
    + quote_plus(CONNECTION_STRING),
    pool_pre_ping=True,
    pool_recycle=300,
)


def read_sql_file(path: Path) -> str:
    """Read SQL and replace the configurable time cutoff token."""

    if not path.is_file():
        raise FileNotFoundError(f"SQL file not found: {path}")

    query = path.read_text(encoding="utf-8-sig")

    query = query.replace(
        "{time_cutoff_minutes}",
        str(TIME_CUTOFF),
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


def read_rac_events() -> pd.DataFrame:
    """Read and validate recent RAC events."""

    query = read_sql_file(RAC_QUERY_FILE)

    with engine.connect() as connection:
        events = pd.read_sql_query(query, connection)

    events = normalize_columns(
        events,
        ["Time", "X", "Y", "Level"],
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

    events = events.dropna(
        subset=["Time", "X", "Y", "Level"]
    ).copy()

    events = events.reset_index(drop=True)

    return events


def read_lanes() -> pd.DataFrame:
    """Read active autonomous lane edge points."""

    query = read_sql_file(LANE_QUERY_FILE)

    with engine.connect() as connection:
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
        subset=[
            "LANE_OID",
            "Segment",
            "POINT_NR",
            "X",
            "Y",
        ]
    ).copy()

    logger.info(
        "Loaded %d lane edge points from %d active lanes",
        len(lanes),
        lanes["LANE_OID"].nunique(),
    )

    return lanes


# ---------------------------------------------------------------------------
# New-event detection
# ---------------------------------------------------------------------------

def event_signatures(
    events: pd.DataFrame,
) -> list[tuple[int, float, float, float]]:
    """
    Build stable event signatures.

    Counter handles multiple events with the same time and location.
    """

    signatures: list[tuple[int, float, float, float]] = []

    for _, row in events.iterrows():
        timestamp = pd.Timestamp(row["Time"])

        signatures.append(
            (
                int(timestamp.value),
                round(float(row["X"]), 3),
                round(float(row["Y"]), 3),
                round(float(row["Level"]), 3),
            )
        )

    return signatures


def mark_new_events(
    events: pd.DataFrame,
    previous_counts: Counter | None,
) -> tuple[pd.DataFrame, Counter, int]:
    """
    Mark events that were not in the previous successful sample.

    The first sample is a baseline. Nothing is marked new.
    """

    events = events.copy()
    signatures = event_signatures(events)
    current_counts = Counter(signatures)

    if previous_counts is None:
        events["_IsNew"] = False
        return events, current_counts, 0

    remaining_new: Counter = Counter()

    for signature, count in current_counts.items():
        previous_count = previous_counts.get(signature, 0)
        additional_count = count - previous_count

        if additional_count > 0:
            remaining_new[signature] = additional_count

    new_flags: list[bool] = []

    for signature in signatures:
        if remaining_new.get(signature, 0) > 0:
            new_flags.append(True)
            remaining_new[signature] -= 1
        else:
            new_flags.append(False)

    events["_IsNew"] = new_flags

    return events, current_counts, int(sum(new_flags))


# ---------------------------------------------------------------------------
# RAC grouping
# ---------------------------------------------------------------------------

def find_clusters(events: pd.DataFrame) -> list[Cluster]:
    """
    Use the original hub-radius grouping mechanism.

    A possible group contains currently unassigned events within the
    configured radius of a seed event. Qualifying groups consume their
    members. The generated zone is centered on the member average.
    """

    if events.empty:
        return []

    working = (
        events.sort_values("X", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    x_values = working["X"].to_numpy(dtype=float)
    y_values = working["Y"].to_numpy(dtype=float)
    levels = working["Level"].to_numpy(dtype=float)

    if "_IsNew" in working.columns:
        new_flags = working["_IsNew"].to_numpy(dtype=bool)
    else:
        new_flags = np.zeros(len(working), dtype=bool)

    assigned = np.zeros(len(working), dtype=bool)
    clusters: list[Cluster] = []

    for seed_index in range(len(working)):
        if assigned[seed_index]:
            continue

        distances = np.hypot(
            x_values - x_values[seed_index],
            y_values - y_values[seed_index],
        )

        member_indices = np.flatnonzero(
            (~assigned) & (distances < GROUP_RADIUS)
        )

        if len(member_indices) < MINIMUM_EVENTS:
            continue

        score = float(levels[member_indices].sum())

        if INCLUSIVE_THRESHOLD:
            qualifies = score >= GROUP_SCORE
        else:
            qualifies = score > GROUP_SCORE

        if not qualifies:
            continue

        assigned[member_indices] = True

        clusters.append(
            Cluster(
                center_x=float(
                    x_values[member_indices].mean()
                ),
                center_y=float(
                    y_values[member_indices].mean()
                ),
                search_x=float(x_values[seed_index]),
                search_y=float(y_values[seed_index]),
                score=score,
                event_count=len(member_indices),
                new_event_count=int(
                    new_flags[member_indices].sum()
                ),
            )
        )

    clusters.sort(
        key=lambda cluster: cluster.score,
        reverse=True,
    )

    return clusters[:TOP_N]


# ---------------------------------------------------------------------------
# MineStar commands and existing zones
# ---------------------------------------------------------------------------

def mstar_command(
    arguments: list[str],
) -> subprocess.CompletedProcess:
    """Run mstarrun or mstarrun.bat from the MineStar bin directory."""

    executable = MSTAR_DIRECTORY / MSTAR_EXECUTABLE

    if not executable.is_file():
        raise FileNotFoundError(
            f"mstarrun not found: {executable}"
        )

    creation_flags = 0

    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creation_flags = subprocess.CREATE_NO_WINDOW

    if executable.suffix.lower() in {".bat", ".cmd"}:
        command_line = subprocess.list2cmdline(
            [str(executable), *arguments]
        )

        command = [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            command_line,
        ]
    else:
        command = [str(executable), *arguments]

    logger.debug(
        "Running MineStar command: %s",
        subprocess.list2cmdline(command),
    )

    return subprocess.run(
        command,
        cwd=MSTAR_DIRECTORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=COMMAND_TIMEOUT,
        check=False,
        creationflags=creation_flags,
    )


def polygon_center(
    points: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return the average center of polygon vertices."""

    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]

    if not points:
        raise ValueError("Zone has no valid polygon points")

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)

    return center_x, center_y


def export_current_zones() -> list[ExistingZone]:
    """
    Export and parse active MineStar RAC_AUTO zones.

    This is refreshed every cycle. If a user deletes a zone, it
    disappears from this collection on the next successful export.
    """

    if CURRENT_ZONES_FILE.exists():
        CURRENT_ZONES_FILE.unlink()

    result = mstar_command(
        [
            "-b",
            "exportZones",
            "-file",
            str(CURRENT_ZONES_FILE),
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(
            "exportZones failed. "
            f"Return code={result.returncode}. "
            f"stdout={result.stdout.strip()!r}. "
            f"stderr={result.stderr.strip()!r}"
        )

    if not CURRENT_ZONES_FILE.is_file():
        raise RuntimeError(
            "exportZones returned success but did not create: "
            f"{CURRENT_ZONES_FILE}"
        )

    root = ET.parse(CURRENT_ZONES_FILE).getroot()
    zones: list[ExistingZone] = []

    for zone_element in root.findall(".//zone"):
        active_text = zone_element.findtext(
            "active",
            default="true",
        )

        active = active_text.strip().lower() == "true"
        name = zone_element.findtext(
            "name",
            default="",
        ).strip()

        if not active:
            continue

        if not name.upper().startswith(ZONE_PREFIX.upper()):
            continue

        polygon = zone_element.find("polygon")

        if polygon is None:
            continue

        points: list[tuple[float, float]] = []

        for point in polygon.findall("point"):
            try:
                x_value = float(point.get("x", ""))
                y_value = float(point.get("y", ""))
            except ValueError:
                continue

            points.append((x_value, y_value))

        if not points:
            continue

        center_x, center_y = polygon_center(points)

        radius = max(
            math.hypot(
                x_value - center_x,
                y_value - center_y,
            )
            for x_value, y_value in points
        )

        zones.append(
            ExistingZone(
                name=name,
                center_x=center_x,
                center_y=center_y,
                radius=radius,
            )
        )

    logger.info(
        "Read %d active %s zone(s) from MineStar",
        len(zones),
        ZONE_PREFIX,
    )

    return zones


def cluster_is_suppressed(
    cluster: Cluster,
    existing_zones: list[ExistingZone],
) -> bool:
    """Return True when the cluster is too close to an existing zone."""

    for zone in existing_zones:
        distance = math.hypot(
            cluster.center_x - zone.center_x,
            cluster.center_y - zone.center_y,
        )

        if distance <= SUPPRESSION_RADIUS:
            return True

    return False


# ---------------------------------------------------------------------------
# Zone XML generation and import
# ---------------------------------------------------------------------------

def create_zone_xml(
    cluster: Cluster,
) -> tuple[Path, str]:
    """Create a circular MineStar zone from the exported template."""

    if not ZONE_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Zone template not found: {ZONE_TEMPLATE}"
        )

    source_root = ET.parse(ZONE_TEMPLATE).getroot()
    template_zone = source_root.find("zone")

    if template_zone is None:
        raise ValueError(
            f"{ZONE_TEMPLATE} does not contain a <zone>"
        )

    output_root = ET.Element("zones")
    zone_element = copy.deepcopy(template_zone)
    output_root.append(zone_element)

    now = datetime.now().astimezone()
    timestamp = now.isoformat(timespec="milliseconds")

    zone_name = (
        f"{ZONE_PREFIX}_"
        f"{now.strftime('%Y%m%d_%H%M%S')}_"
        f"{cluster.center_x:.1f}_"
        f"{cluster.center_y:.1f}"
    )

    name_element = zone_element.find("name")
    polygon_element = zone_element.find("polygon")

    if name_element is None:
        raise ValueError(
            "Zone template does not contain <name>"
        )

    if polygon_element is None:
        raise ValueError(
            "Zone template does not contain <polygon>"
        )

    name_element.text = zone_name

    created_element = zone_element.find("createdDate")
    updated_element = zone_element.find("lastUpdatedDate")

    if created_element is not None:
        created_element.text = timestamp

    if updated_element is not None:
        updated_element.text = timestamp

    polygon_position = list(zone_element).index(
        polygon_element
    )

    zone_element.remove(polygon_element)

    new_polygon = ET.Element("polygon")
    generated_points: list[tuple[float, float]] = []

    for point_index in range(ZONE_POINTS):
        angle = (
            2.0
            * math.pi
            * point_index
            / ZONE_POINTS
        )

        x_value = (
            cluster.center_x
            + ZONE_RADIUS * math.cos(angle)
        )

        y_value = (
            cluster.center_y
            + ZONE_RADIUS * math.sin(angle)
        )

        generated_points.append((x_value, y_value))

    # MineStar exports repeat the first point to close the polygon.
    generated_points.append(generated_points[0])

    for x_value, y_value in generated_points:
        ET.SubElement(
            new_polygon,
            "point",
            {
                "x": f"{x_value:.2f}",
                "y": f"{y_value:.2f}",
                "z": f"{ZONE_ELEVATION:.2f}",
            },
        )

    zone_element.insert(
        polygon_position,
        new_polygon,
    )

    ET.indent(output_root, space="    ")

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        zone_name,
    )

    output_file = ZONE_OUTPUT / f"{safe_name}.xml"
    temporary_file = output_file.with_suffix(".xml.tmp")

    xml_body = ET.tostring(
        output_root,
        encoding="unicode",
        short_empty_elements=True,
    )

    xml_document = (
        '<?xml version="1.0" encoding="UTF-8" '
        'standalone="yes"?>\n'
        f"{xml_body}\n"
    )

    temporary_file.write_text(
        xml_document,
        encoding="utf-8",
    )

    temporary_file.replace(output_file)

    return output_file, zone_name


def import_zone(xml_file: Path) -> str:
    """
    Import one zone.

    Returns:
        imported
        disabled
        dry_run
        failed
    """

    command = [
        "-b",
        "importZones",
        "-file",
        str(xml_file.resolve()),
    ]

    if ALLOW_OUTSIDE:
        command.extend(
            [
                "-AllowOutsideMineBoundary",
                "NO_VALIDATION",
            ]
        )

    if not IMPORT_ENABLED:
        logger.info(
            "Import disabled. XML generated only: %s",
            xml_file,
        )
        return "disabled"

    if DRY_RUN:
        logger.info(
            "Dry-run import command: %s",
            subprocess.list2cmdline(
                [
                    str(
                        MSTAR_DIRECTORY
                        / MSTAR_EXECUTABLE
                    ),
                    *command,
                ]
            ),
        )
        return "dry_run"

    result = mstar_command(command)

    if result.stdout.strip():
        logger.info(
            "mstarrun output: %s",
            result.stdout.strip(),
        )

    if result.returncode != 0:
        logger.error(
            "importZones failed for %s. "
            "Return code=%d. stderr=%s",
            xml_file,
            result.returncode,
            result.stderr.strip(),
        )
        return "failed"

    logger.info(
        "Successfully imported MineStar zone: %s",
        xml_file.name,
    )

    return "imported"


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class Viewer:
    """Live site map."""

    def __init__(self) -> None:
        plt.ion()

        self.figure = plt.figure(
            figsize=(13, 10),
            facecolor="#202020",
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "RAC Automatic Zone Monitor"
            )
        except AttributeError:
            pass

        plt.show(block=False)

    def is_open(self) -> bool:
        return plt.fignum_exists(self.figure.number)

    def update(
        self,
        events: pd.DataFrame,
        lanes: pd.DataFrame,
        clusters: list[Cluster],
        existing_zones: list[ExistingZone],
    ) -> None:
        """Redraw the complete site map."""

        if not self.is_open():
            return

        self.figure.clear()
        axis = self.figure.add_subplot(111)

        axis.set_facecolor("#202020")
        axis.set_aspect("equal", adjustable="box")

        all_x: list[float] = []
        all_y: list[float] = []

        # Draw lane polygons.
        if not lanes.empty:
            for _, lane in lanes.groupby("LANE_OID"):
                left = lane[
                    lane["Segment"]
                    .astype(str)
                    .str.lower()
                    == "left"
                ].sort_values("POINT_NR")

                right = lane[
                    lane["Segment"]
                    .astype(str)
                    .str.lower()
                    == "right"
                ].sort_values("POINT_NR")

                if left.empty or right.empty:
                    continue

                x_coordinates = (
                    left["X"].astype(float).tolist()
                    + right["X"]
                    .astype(float)
                    .tolist()[::-1]
                )

                y_coordinates = (
                    left["Y"].astype(float).tolist()
                    + right["Y"]
                    .astype(float)
                    .tolist()[::-1]
                )

                axis.add_patch(
                    Polygon(
                        list(
                            zip(
                                x_coordinates,
                                y_coordinates,
                            )
                        ),
                        closed=True,
                        facecolor="#164916",
                        edgecolor="#32CD32",
                        linewidth=0.7,
                        alpha=0.45,
                        zorder=1,
                    )
                )

                all_x.extend(x_coordinates)
                all_y.extend(y_coordinates)

        # Draw RAC events.
        if not events.empty:
            minimum_level = float(events["Level"].min())
            maximum_level = float(events["Level"].max())

            if minimum_level == maximum_level:
                maximum_level = minimum_level + 1.0

            color_map = LinearSegmentedColormap.from_list(
                "rac_severity",
                ["yellow", "orange", "red"],
            )

            collection = axis.scatter(
                events["X"],
                events["Y"],
                c=events["Level"],
                cmap=color_map,
                norm=Normalize(
                    minimum_level,
                    maximum_level,
                ),
                s=40,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.9,
                zorder=3,
                label="RAC events",
            )

            color_bar = self.figure.colorbar(
                collection,
                ax=axis,
                pad=0.02,
            )

            color_bar.set_label(
                "RAC Level",
                color="#A9A9A9",
                fontweight="bold",
            )

            color_bar.ax.tick_params(
                colors="#A9A9A9"
            )

            # Highlight RAC events first seen this cycle.
            if "_IsNew" in events.columns:
                new_events = events[events["_IsNew"]]

                if not new_events.empty:
                    axis.scatter(
                        new_events["X"],
                        new_events["Y"],
                        facecolors="none",
                        edgecolors="#00FFFF",
                        s=100,
                        linewidths=2,
                        zorder=5,
                        label="New RAC events",
                    )

            all_x.extend(
                events["X"].astype(float).tolist()
            )
            all_y.extend(
                events["Y"].astype(float).tolist()
            )

        # Draw existing automatic zones and suppression areas.
        for zone in existing_zones:
            axis.add_patch(
                Circle(
                    (zone.center_x, zone.center_y),
                    zone.radius,
                    facecolor="#FF00FF",
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.45,
                    zorder=4,
                )
            )

            axis.add_patch(
                Circle(
                    (zone.center_x, zone.center_y),
                    SUPPRESSION_RADIUS,
                    fill=False,
                    edgecolor="#FF69B4",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=3,
                )
            )

            axis.text(
                zone.center_x,
                zone.center_y - zone.radius - 0.5,
                zone.name,
                color="#FFB6FF",
                fontsize=7,
                horizontalalignment="center",
                zorder=6,
            )

            all_x.extend(
                [
                    zone.center_x - SUPPRESSION_RADIUS,
                    zone.center_x + SUPPRESSION_RADIUS,
                ]
            )
            all_y.extend(
                [
                    zone.center_y - SUPPRESSION_RADIUS,
                    zone.center_y + SUPPRESSION_RADIUS,
                ]
            )

        # Draw qualifying RAC groups.
        for cluster in clusters:
            blocked = cluster_is_suppressed(
                cluster,
                existing_zones,
            )

            if blocked:
                color = "#808080"
            elif cluster.new_event_count > 0:
                color = "#00FFFF"
            else:
                color = "#4169E1"

            axis.add_patch(
                Circle(
                    (cluster.search_x, cluster.search_y),
                    GROUP_RADIUS,
                    fill=False,
                    edgecolor=color,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.6,
                    zorder=2,
                )
            )

            axis.scatter(
                [cluster.center_x],
                [cluster.center_y],
                marker="x",
                color=color,
                s=100,
                linewidths=2,
                zorder=6,
            )

            axis.text(
                cluster.center_x,
                cluster.center_y,
                (
                    f" {cluster.event_count} events"
                    f", {cluster.new_event_count} new"
                    f", score {cluster.score:g}"
                ),
                color="white",
                fontsize=8,
                zorder=7,
            )

            all_x.append(cluster.center_x)
            all_y.append(cluster.center_y)

        # Map bounds.
        if all_x and all_y:
            minimum_x = min(all_x)
            maximum_x = max(all_x)
            minimum_y = min(all_y)
            maximum_y = max(all_y)

            x_range = max(maximum_x - minimum_x, 10.0)
            y_range = max(maximum_y - minimum_y, 10.0)
            padding = max(x_range, y_range) * 0.05

            axis.set_xlim(
                minimum_x - padding,
                maximum_x + padding,
            )
            axis.set_ylim(
                minimum_y - padding,
                maximum_y + padding,
            )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_edgecolor("#A9A9A9")
            spine.set_linewidth(2)

        new_event_count = 0

        if "_IsNew" in events.columns:
            new_event_count = int(
                events["_IsNew"].sum()
            )

        axis.set_title(
            (
                f"RAC events: {len(events)} | "
                f"New: {new_event_count} | "
                f"Groups: {len(clusters)} | "
                f"Automatic zones: {len(existing_zones)}"
            ),
            color="white",
            fontsize=14,
            fontweight="bold",
        )

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def wait_for_next_cycle(
    seconds: int,
    viewer: Viewer | None,
) -> bool:
    """Wait while continuing to process viewer events."""

    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        if viewer is not None:
            if not viewer.is_open():
                return False

            plt.pause(0.1)

        time.sleep(0.1)

    return True


def main() -> None:
    logger.info("Starting RAC Automatic Zone Monitor")
    logger.info(
        "Polling interval: %d seconds",
        POLL_SECONDS,
    )
    logger.info(
        "RAC time window: %d minutes",
        TIME_CUTOFF,
    )
    logger.info(
        "Actual zone diameter: %.2f m",
        ZONE_RADIUS * 2.0,
    )
    logger.info(
        "Suppression diameter: %.2f m",
        SUPPRESSION_RADIUS * 2.0,
    )
    logger.info(
        "Automatic imports enabled=%s, dry_run=%s",
        IMPORT_ENABLED,
        DRY_RUN,
    )

    viewer = Viewer() if DISPLAY_ENABLED else None

    previous_event_counts: Counter | None = None

    existing_zones: list[ExistingZone] = []

    last_lanes = pd.DataFrame(
        columns=[
            "LANE_OID",
            "Segment",
            "POINT_NR",
            "X",
            "Y",
        ]
    )

    try:
        while True:
            try:
                # RAC query must succeed for this cycle to continue.
                raw_events = read_rac_events()

                (
                    events,
                    current_event_counts,
                    new_event_count,
                ) = mark_new_events(
                    raw_events,
                    previous_event_counts,
                )

                first_sample = previous_event_counts is None

                clusters = find_clusters(events)

                # Lane failure does not stop RAC processing.
                try:
                    last_lanes = read_lanes()
                except Exception:
                    logger.exception(
                        "Lane refresh failed. Retaining previous lanes."
                    )

                # Refresh current zones every cycle so asynchronous user
                # deletions are detected.
                zones_are_current = False

                try:
                    existing_zones = export_current_zones()
                    zones_are_current = True
                except Exception:
                    logger.exception(
                        "MineStar zone refresh failed. "
                        "New zone imports are blocked this cycle."
                    )

                # Normally the current sample becomes the comparison
                # baseline for the next cycle.
                commit_event_sample = True

                if first_sample:
                    logger.info(
                        "Initial RAC baseline captured: %d event(s). "
                        "No zones will be generated or imported.",
                        len(events),
                    )

                elif events.empty:
                    logger.info(
                        "No RAC events in the last %d minutes. "
                        "No action.",
                        TIME_CUTOFF,
                    )

                elif new_event_count == 0:
                    logger.info(
                        "No new RAC events since the previous poll. "
                        "No action."
                    )

                else:
                    logger.info(
                        "Detected %d new RAC event(s).",
                        new_event_count,
                    )

                    actionable_clusters = [
                        cluster
                        for cluster in clusters
                        if cluster.new_event_count > 0
                    ]

                    if not actionable_clusters:
                        logger.info(
                            "%d new RAC event(s) detected, but no "
                            "group exceeded the configured threshold. "
                            "No action.",
                            new_event_count,
                        )

                    elif not zones_are_current:
                        # Do not forget these events. Retry them on the
                        # next cycle after MineStar becomes available.
                        commit_event_sample = False

                        logger.error(
                            "%d qualifying cluster(s) found, but "
                            "current MineStar zones could not be read. "
                            "No XML or imports created. Will retry.",
                            len(actionable_clusters),
                        )

                    else:
                        import_failed = False

                        logger.info(
                            "%d qualifying cluster(s) contain new "
                            "RAC events.",
                            len(actionable_clusters),
                        )

                        for cluster in actionable_clusters:
                            if cluster_is_suppressed(
                                cluster,
                                existing_zones,
                            ):
                                logger.info(
                                    "Cluster suppressed at "
                                    "X=%.2f Y=%.2f. "
                                    "Events=%d, new=%d, score=%.2f.",
                                    cluster.center_x,
                                    cluster.center_y,
                                    cluster.event_count,
                                    cluster.new_event_count,
                                    cluster.score,
                                )
                                continue

                            xml_file, zone_name = create_zone_xml(
                                cluster
                            )

                            logger.info(
                                "Generated zone %s at "
                                "X=%.2f Y=%.2f. "
                                "Events=%d, new=%d, score=%.2f.",
                                zone_name,
                                cluster.center_x,
                                cluster.center_y,
                                cluster.event_count,
                                cluster.new_event_count,
                                cluster.score,
                            )

                            import_status = import_zone(xml_file)

                            if import_status == "imported":
                                # Prevent another cluster in this same
                                # cycle from creating a nearby zone.
                                existing_zones.append(
                                    ExistingZone(
                                        name=zone_name,
                                        center_x=cluster.center_x,
                                        center_y=cluster.center_y,
                                        radius=ZONE_RADIUS,
                                    )
                                )

                            elif import_status == "failed":
                                import_failed = True

                        # Retry new events after a genuine import failure.
                        # Disabled and dry-run tests are not repeated
                        # continuously every 30 seconds.
                        if import_failed:
                            commit_event_sample = False

                if commit_event_sample:
                    previous_event_counts = current_event_counts
                else:
                    logger.warning(
                        "Current RAC sample was not committed. "
                        "New events will be retried next cycle."
                    )

                if viewer is not None:
                    viewer.update(
                        events,
                        last_lanes,
                        clusters,
                        existing_zones,
                    )

            except Exception:
                logger.exception(
                    "Polling cycle failed. "
                    "No zones imported this cycle."
                )

            if not wait_for_next_cycle(
                POLL_SECONDS,
                viewer,
            ):
                logger.info("Viewer window closed.")
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user.")

    finally:
        engine.dispose()
        logger.info("RAC Automatic Zone Monitor stopped.")


if __name__ == "__main__":
    main()
