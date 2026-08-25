from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path

DEFAULT_CONFIG_NAME = "config.ini"
PUBLIC_APP_FOLDER = "RACZoneGen"


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def default_public_directory() -> Path:
    public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    return public_root / PUBLIC_APP_FOLDER


def find_config_file() -> tuple[Path, Path]:
    candidates = [
        application_directory() / DEFAULT_CONFIG_NAME,
        default_public_directory() / DEFAULT_CONFIG_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.parent
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError("Configuration file not found. Looked in: " + searched)


class Config:
    def __init__(self, config_file: Path | None = None) -> None:
        if config_file is None:
            self.config_file, self.config_dir = find_config_file()
        else:
            self.config_file = config_file.resolve()
            self.config_dir = self.config_file.parent
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(self.config_file, encoding="utf-8-sig")
        self._parser = parser
        self.poll_interval_seconds = parser.getint("application", "poll_interval_seconds")
        self.log_level = "INFO"
        server = parser.get("database", "server").strip()
        database = parser.get("database", "database").strip()
        self.odbc_connection_string = f"Driver={{SQL Server}};Server={server};Database={database};Trusted_Connection=yes;"
        self.db_query_timeout = 60
        if parser.has_option("database", "sql_directory"):
            self.sql_directory = self._path("database", "sql_directory")
        else:
            self.sql_directory = (self.config_dir / "sql").resolve()
        self.rac_query_file = (self.sql_directory / "rac_events.sql").resolve()
        self.lane_query_file = (self.sql_directory / "lanes.sql").resolve()
        self.zones_query_file = (self.sql_directory / "zones.sql").resolve()
        self.time_cutoff_minutes = 120
        self.grouping_radius_metres = parser.getfloat("rac", "grouping_radius_metres")
        self.grouping_score = parser.getfloat("rac", "grouping_score")
        self.minimum_event_count = parser.getint("rac", "minimum_event_count")
        self.top_n = 1000
        self.threshold_inclusive = True
        self.zone_output_directory = self._path("zone", "output_directory")
        self.zone_size_metres = parser.getfloat("zone", "zone_size_metres")
        self.suppression_radius_metres = parser.getfloat("zone", "suppression_radius_metres")
        self.zones_refresh_interval_seconds = parser.getint("zone", "zones_refresh_interval_seconds")
        self.zone_name_prefix = "RacMonitorGen_zone"
        self.zone_elevation = 300.0
        self.zone_template_file = self.config_dir / "zones.xml"
        self.lane_refresh_interval_seconds = parser.getint("lanes", "refresh_interval_seconds")
        self.default_speed_kmh = 40
        self.speed_offset_kmh = 10.0
        self.minimum_speed_kmh = 5.0
        self.maximum_speed_kmh = 999.0
        if parser.has_option("minestar", "mstarrun_location"):
            val = parser.get("minestar", "mstarrun_location").strip()
            p = Path(os.path.expandvars(val))
            if not p.is_absolute():
                p = self.config_dir / p
            self.mstar_bin_directory = p.parent.resolve()
            self.mstar_executable = p.name
            self.mstarrun_location = p.resolve()
        else:
            self.mstar_bin_directory = Path(r"D:\mstar\mstarHome\bus\bin").resolve()
            self.mstar_executable = "mstarrun.bat"
            self.mstarrun_location = (self.mstar_bin_directory / self.mstar_executable).resolve()
        self.current_zones_file = (self.config_dir / "runtime" / "current_zones.xml").resolve()
        self.command_timeout_seconds = 120
        self.import_enabled = True
        self.dry_run = False
        self.show_only = False
        self.allow_outside_mine_boundary = False
        self.import_workers = 1
        self.simulation_enabled = parser.getboolean("simulation", "enabled", fallback=False)
        self.simulation_import_enabled = parser.getboolean("simulation", "import_enabled", fallback=False)
        self.simulation_real_import = self.simulation_import_enabled
        self.simulation_seed = parser.getint("simulation", "random_seed", fallback=20260824)
        self.simulation_spawn_per_cycle = parser.getint("simulation", "spawn_per_cycle", fallback=1)
        self.simulation_jitter_metres = parser.getfloat("simulation", "jitter_metres", fallback=500.0)
        self.simulation_level_min = parser.getint("simulation", "level_min", fallback=3)
        self.simulation_level_max = parser.getint("simulation", "level_max", fallback=4)
        self.simulation_speed_min = parser.getfloat("simulation", "speed_min", fallback=45.0)
        self.simulation_speed_max = parser.getfloat("simulation", "speed_max", fallback=60.0)
        self.simulation_time_window_minutes = parser.getint("simulation", "time_window_minutes", fallback=120)
        self.simulation_hotspots = self._parse_hotspots("simulation", "hotspots")
        self.display_enabled = True
        self.log_directory = self.config_dir / "logs"
        self.runtime_directory = self.config_dir / "runtime"
        self.validate()

    def _path(self, section: str, option: str) -> Path:
        configured_value = self._parser.get(section, option).strip()
        path = Path(os.path.expandvars(configured_value))
        if not path.is_absolute():
            path = self.config_dir / path
        return path.resolve()

    def _parse_hotspots(self, section: str, option: str) -> list[tuple[float, float, float | None, float | None]]:
        raw = self._parser.get(section, option, fallback="")
        result: list[tuple[float, float, float | None, float | None]] = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",")]
            if len(parts) < 2:
                continue
            try:
                x_value = float(parts[0])
                y_value = float(parts[1])
            except ValueError:
                continue
            try:
                level = float(parts[2]) if len(parts) > 2 else None
            except ValueError:
                level = None
            try:
                speed = float(parts[3]) if len(parts) > 3 else None
            except ValueError:
                speed = None
            result.append((x_value, y_value, level, speed))
        return result

    def validate(self) -> None:
        if self.poll_interval_seconds < 1:
            raise ValueError("poll_interval_seconds must be at least 1")
        if self.grouping_radius_metres <= 0:
            raise ValueError("grouping_radius_metres must be > 0")
        if self.minimum_event_count < 1:
            raise ValueError("minimum_event_count must be at least 1")
        if self.zone_size_metres <= 0:
            raise ValueError("zone_size_metres must be > 0")
        if self.suppression_radius_metres < 0:
            raise ValueError("suppression_radius_metres must be >= 0")
