"""
MineStar integration.

Builds MineStar zone XML documents from a template, computes the
speed limit from the average vehicle speed of the RAC cluster, and
runs mstarrun.bat to export / import zones.

Zone naming convention:  <name_prefix>_<timestamp>_<X>_<Y>
The prefix acts as a filter so exported zones can be grouped into an
in-memory library on startup and used to suppress duplicates.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clustering import Cluster

logger = logging.getLogger("rac_zone_monitor")


@dataclass
class ExistingZone:
    """A zone that already exists in MineStar (exported from it)."""

    name: str
    center_x: float
    center_y: float
    half_size: float  # half the bounding-box extent (metres)


def polygon_bounds(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Return (min_x, max_x, min_y, max_y) of polygon points."""
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return min(x_values), max(x_values), min(y_values), max(y_values)


class MineStar:
    """Handles mstarrun commands and zone XML generation."""

    def __init__(self, config) -> None:
        self.config = config
        self.executable = config.mstar_bin_directory / config.mstar_executable

    # ------------------------------------------------------------------
    # mstarrun
    # ------------------------------------------------------------------

    def _command(self, arguments: list[str]) -> subprocess.CompletedProcess:
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"mstarrun not found: {self.executable}"
            )

        creation_flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creation_flags = subprocess.CREATE_NO_WINDOW

        if self.executable.suffix.lower() in {".bat", ".cmd"}:
            command_line = subprocess.list2cmdline(
                [str(self.executable), *arguments]
            )
            command = [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/d",
                "/c",
                command_line,
            ]
        else:
            command = [str(self.executable), *arguments]

        logger.debug(
            "Running MineStar command: %s",
            subprocess.list2cmdline(command),
        )

        return subprocess.run(
            command,
            cwd=self.config.mstar_bin_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.config.command_timeout_seconds,
            check=False,
            creationflags=creation_flags,
        )

    # ------------------------------------------------------------------
    # Existing zones
    # ------------------------------------------------------------------

    def export_current_zones(self) -> list[ExistingZone]:
        """
        Export and parse active zones whose name uses our prefix.

        A successful export that produces no file is treated as "zero
        zones" (with a warning) so a first deployment on a site with no
        RAC_AUTO zones does not block zone creation forever.
        """
        current_file = self.config.current_zones_file

        if current_file.exists():
            current_file.unlink()

        result = self._command(
            [
                "-b",
                "exportZones",
                "-file",
                str(current_file),
            ]
        )

        if result.returncode != 0:
            raise RuntimeError(
                "exportZones failed. "
                f"Return code={result.returncode}. "
                f"stdout={result.stdout.strip()!r}. "
                f"stderr={result.stderr.strip()!r}"
            )

        if not current_file.is_file():
            logger.warning(
                "exportZones returned success but no zone file was "
                "created (%s). Treating as zero existing zones.",
                current_file,
            )
            return []

        root = ET.parse(current_file).getroot()
        zones: list[ExistingZone] = []

        prefix = self.config.zone_name_prefix

        for zone_element in root.findall(".//zone"):
            active_text = zone_element.findtext("active", default="true")
            active = active_text.strip().lower() == "true"
            name = zone_element.findtext("name", default="").strip()

            if not active or not re.match(
                re.escape(prefix), name, re.IGNORECASE
            ):
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

            min_x, max_x, min_y, max_y = polygon_bounds(points)
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
            "Read %d active %s* zone(s) from MineStar",
            len(zones),
            prefix,
        )

        return zones

    # ------------------------------------------------------------------
    # Zone XML
    # ------------------------------------------------------------------

    def compute_speed_limit(self, cluster: Cluster) -> float:
        """
        Zone speed limit = average vehicle speed - offset, clamped to
        a configured floor/ceiling.

        average_speed_kmh may be None when no speed could be parsed; in
        that case the configured default speed is used.
        """
        average = cluster.average_speed_kmh
        if average is None or average <= 0:
            average = self.config.default_speed_kmh

        limit = average - self.config.speed_offset_kmh
        limit = max(limit, self.config.minimum_speed_kmh)
        limit = min(limit, self.config.maximum_speed_kmh)

        return limit

    def create_zone_xml(
        self,
        cluster: Cluster,
        speed_limit_kmh: float,
    ) -> tuple[Path, str]:
        """Create a square MineStar zone XML from the exported template."""
        template_file = self.config.zone_template_file

        if not template_file.is_file():
            raise FileNotFoundError(
                f"Zone template not found: {template_file}"
            )

        source_root = ET.parse(template_file).getroot()
        template_zone = source_root.find("zone")

        if template_zone is None:
            raise ValueError(
                f"{template_file} does not contain a <zone>"
            )

        output_root = ET.Element("zones")
        zone_element = copy.deepcopy(template_zone)
        output_root.append(zone_element)

        now = datetime.now().astimezone()
        timestamp = now.isoformat(timespec="milliseconds")

        zone_name = (
            f"{self.config.zone_name_prefix}_"
            f"{now.strftime('%Y%m%d_%H%M%S')}_"
            f"{cluster.center_x:.1f}_"
            f"{cluster.center_y:.1f}"
        )

        name_element = zone_element.find("name")
        polygon_element = zone_element.find("polygon")

        if name_element is None:
            raise ValueError("Zone template does not contain <name>")

        if polygon_element is None:
            raise ValueError("Zone template does not contain <polygon>")

        name_element.text = zone_name

        created_element = zone_element.find("createdDate")
        updated_element = zone_element.find("lastUpdatedDate")

        if created_element is not None:
            created_element.text = timestamp

        if updated_element is not None:
            updated_element.text = timestamp

        polygon_position = list(zone_element).index(polygon_element)
        zone_element.remove(polygon_element)

        new_polygon = ET.Element("polygon")
        half_size = self.config.zone_size_metres / 2.0
        elevation = self.config.zone_elevation

        corners = [
            (cluster.center_x - half_size, cluster.center_y - half_size),
            (cluster.center_x + half_size, cluster.center_y - half_size),
            (cluster.center_x + half_size, cluster.center_y + half_size),
            (cluster.center_x - half_size, cluster.center_y + half_size),
            (cluster.center_x - half_size, cluster.center_y - half_size),
        ]

        for x_value, y_value in corners:
            ET.SubElement(
                new_polygon,
                "point",
                {
                    "x": f"{x_value:.2f}",
                    "y": f"{y_value:.2f}",
                    "z": f"{elevation:.2f}",
                },
            )

        zone_element.insert(polygon_position, new_polygon)

        speed_limit = zone_element.find("speedLimit")

        if speed_limit is not None:
            speed_limit.set(
                "magnitude",
                f"{speed_limit_kmh:.1f}",
            )

        ET.indent(output_root, space="    ")

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", zone_name)
        output_directory = self.config.zone_output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / f"{safe_name}.xml"
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

        temporary_file.write_text(xml_document, encoding="utf-8")
        temporary_file.replace(output_file)

        return output_file, zone_name

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_zone(self, xml_file: Path) -> str:
        """
        Import one zone.

        Returns one of: imported / disabled / dry_run / failed.
        """
        command = [
            "-b",
            "importZones",
            "-file",
            str(xml_file.resolve()),
        ]

        if self.config.allow_outside_mine_boundary:
            command.extend(["-AllowOutsideMineBoundary", "NO_VALIDATION"])

        if not self.config.import_enabled:
            logger.info(
                "Import disabled. XML generated only: %s",
                xml_file,
            )
            return "disabled"

        if self.config.dry_run:
            logger.info(
                "Dry-run import command: %s",
                subprocess.list2cmdline(
                    [str(self.executable), *command]
                ),
            )
            return "dry_run"

        result = self._command(command)

        if result.stdout.strip():
            logger.info("mstarrun output: %s", result.stdout.strip())

        if result.returncode != 0:
            logger.error(
                "importZones failed for %s. Return code=%d. stderr=%s",
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


def cluster_is_suppressed(
    cluster: Cluster,
    existing_zones: list[ExistingZone],
    suppression_radius: float,
) -> bool:
    """True when the cluster centroid is too close to an existing zone."""
    for zone in existing_zones:
        distance = math.hypot(
            cluster.center_x - zone.center_x,
            cluster.center_y - zone.center_y,
        )
        if distance <= suppression_radius:
            return True

    return False