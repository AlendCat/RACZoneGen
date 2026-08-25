from __future__ import annotations

import copy
import logging
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from clustering import Cluster

logger = logging.getLogger("rac_zone_monitor")


@dataclass
class ExistingZone:
    name: str
    center_x: float
    center_y: float
    half_size: float
    proposed: bool = False


def polygon_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    return min(x_values), max(x_values), min(y_values), max(y_values)


class MineStar:
    def __init__(self, config) -> None:
        self.config = config
        self.executable = getattr(config, "mstarrun_location", getattr(config, "mstar_bin_directory", Path(r"D:\mstar\mstarHome\bus\bin")) / getattr(config, "mstar_executable", "mstarrun.bat"))

    def _command(self, arguments: list[str]) -> subprocess.CompletedProcess:
        if not self.executable.is_file():
            raise FileNotFoundError(f"mstarrun not found: {self.executable}")
        creation_flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creation_flags = subprocess.CREATE_NO_WINDOW
        if self.executable.suffix.lower() in {".bat", ".cmd"}:
            command_line = subprocess.list2cmdline([str(self.executable), *arguments])
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", command_line]
        else:
            command = [str(self.executable), *arguments]
        logger.debug("Running MineStar command: %s", subprocess.list2cmdline(command))
        return subprocess.run(command, cwd=self.executable.parent, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=False, creationflags=creation_flags)

    def compute_speed_limit(self, cluster: Cluster) -> float | None:
        average = cluster.average_speed_kmh
        if average is None or average <= 0 or not math.isfinite(average):
            logger.warning("No parseable speed for cluster at %.1f,%.1f (avg %s) — skipping zone", cluster.center_x, cluster.center_y, average)
            return None
        limit = average - 10.0
        if limit < 5.0:
            limit = 5.0
        return limit

    def _template_zone(self) -> ET.Element:
        template_file = getattr(self.config, "zone_template_file", None)
        if template_file is not None:
            try:
                p = Path(template_file)
                if p.is_file():
                    root = ET.parse(p).getroot()
                    tz = root.find("zone")
                    if tz is not None:
                        return copy.deepcopy(tz)
            except Exception:
                pass
        return ET.fromstring('<zone><OID>0</OID><id>0</id><version>0</version><revision>1</revision><active>true</active><layerUpdateVersion>0</layerUpdateVersion><editable>true</editable><name>New Zone</name><polygon><point x="0" y="0" z="300"/><point x="0" y="0" z="300"/><point x="0" y="0" z="300"/><point x="0" y="0" z="300"/><point x="0" y="0" z="300"/></polygon><speedLimit magnitude="0.0" unit="kilometres per hour" unitType="speed"/><tractionLevel>100</tractionLevel><flags><bedDown>false</bedDown><level1EntryIncident>false</level1EntryIncident><level1ExitIncident>false</level1ExitIncident><beepIfOtherMachinesEnter>false</beepIfOtherMachinesEnter><beepIfOtherMachinesArePresent>false</beepIfOtherMachinesArePresent><silenceProximityAlarms>false</silenceProximityAlarms><silenceRadarAlarms>false</silenceRadarAlarms><mannedExclusionZone>false</mannedExclusionZone><cleanUp>false</cleanUp><canBeRemovedByPanel>false</canBeRemovedByPanel><autonomousInclusionZone>false</autonomousInclusionZone><autonomousExclusionZone>false</autonomousExclusionZone><multicastUpdateZone>false</multicastUpdateZone><passable>false</passable><outsideAOZ>false</outsideAOZ></flags><loadedSpeedLimit magnitude="0.0" unit="kilometres per hour" unitType="speed"/><automaticallyManaged>false</automaticallyManaged><barrier>false</barrier><displayColorRed>255</displayColorRed><displayColorBlue>255</displayColorBlue><displayColorGreen>255</displayColorGreen><displayColorAlpha>255</displayColorAlpha><createdDate>2026-01-01T00:00:00.000+00:00</createdDate><lastUpdatedDate>2026-01-01T00:00:00.000+00:00</lastUpdatedDate></zone>')

    def create_zone_xml(self, cluster: Cluster, speed_limit_kmh: float) -> tuple[Path, str]:
        template_zone = self._template_zone()
        output_root = ET.Element("zones")
        zone_element = copy.deepcopy(template_zone)
        output_root.append(zone_element)
        now = datetime.now().astimezone()
        timestamp = now.isoformat(timespec="milliseconds")
        base_name = f"{self.config.zone_name_prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
        zone_name = base_name
        output_directory = self.config.zone_output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        counter = 1
        while (output_directory / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', zone_name)}.xml").exists():
            counter += 1
            zone_name = f"{base_name}_{counter}"
        name_element = zone_element.find("name")
        polygon_element = zone_element.find("polygon")
        if name_element is None:
            raise ValueError("Zone template does not contain <name>")
        if polygon_element is None:
            raise ValueError("Zone template does not contain <polygon>")
        name_element.text = zone_name
        for tag in ("OID", "id", "version", "revision", "layerUpdateVersion"):
            elem = zone_element.find(tag)
            if elem is not None:
                elem.text = "0"
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
        elevation = 300.0
        corners = [(cluster.center_x - half_size, cluster.center_y - half_size), (cluster.center_x + half_size, cluster.center_y - half_size), (cluster.center_x + half_size, cluster.center_y + half_size), (cluster.center_x - half_size, cluster.center_y + half_size), (cluster.center_x - half_size, cluster.center_y - half_size)]
        for x_value, y_value in corners:
            ET.SubElement(new_polygon, "point", {"x": f"{x_value:.2f}", "y": f"{y_value:.2f}", "z": f"{elevation:.2f}"})
        zone_element.insert(polygon_position, new_polygon)
        speed_limit = zone_element.find("speedLimit")
        if speed_limit is not None:
            speed_limit.set("magnitude", f"{speed_limit_kmh:.1f}")
        ET.indent(output_root, space="    ")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", zone_name)
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / f"{safe_name}.xml"
        temporary_file = output_file.with_suffix(".xml.tmp")
        xml_body = ET.tostring(output_root, encoding="unicode", short_empty_elements=True)
        xml_document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n{xml_body}\n'
        temporary_file.write_text(xml_document, encoding="utf-8")
        temporary_file.replace(output_file)
        return output_file, zone_name

    def import_zone(self, xml_file: Path) -> str:
        is_sim = getattr(self.config, "simulation_enabled", False) or getattr(self.config, "_is_simulated", False)
        # simulation import control via simulation.import_enabled
        if is_sim:
            do_import = bool(getattr(self.config, "simulation_import_enabled", False))
        else:
            do_import = True
        if not do_import:
            logger.info("Import disabled (simulation import_enabled=false). XML kept: %s", xml_file)
            return "disabled"
        bin_target = self.executable.parent / "zones.xml"
        try:
            bin_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(xml_file, bin_target)
        except Exception as copy_error:
            logger.error("Failed to stage import file %s -> %s: %s", xml_file, bin_target, copy_error)
            return "failed"
        result = self._command(["-b", "importZones"])
        if result.stdout.strip():
            logger.info("mstarrun output: %s", result.stdout.strip())
        if result.returncode != 0:
            logger.error("importZones failed for %s. Return code=%d. stderr=%s", xml_file, result.returncode, result.stderr.strip())
            return "failed"
        logger.info("Successfully imported MineStar zone: %s", xml_file.name)
        try:
            xml_file.unlink(missing_ok=True)
            xml_file.with_suffix(".xml.tmp").unlink(missing_ok=True)
            bin_target.unlink(missing_ok=True)
        except Exception:
            pass
        return "imported"


def cluster_is_suppressed(cluster: Cluster, existing_zones: list[ExistingZone], suppression_radius: float) -> bool:
    for zone in existing_zones:
        distance = math.hypot(cluster.center_x - zone.center_x, cluster.center_y - zone.center_y)
        if distance <= suppression_radius:
            return True
    return False
