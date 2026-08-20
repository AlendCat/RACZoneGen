"""
Configuration loading for RACZoneGen.

The config file is looked up in order:
    1. <application directory>/config.ini   (local dev / portable)
    2. %PUBLIC%/RACZoneGen/config.ini        (machine-global deployment)

All configured paths are resolved relative to the directory that owns
the active config file, so a machine-global install is fully relocatable.
"""

from __future__ import annotations

import configparser
import os
import sys
from pathlib import Path

DEFAULT_CONFIG_NAME = "config.ini"
PUBLIC_APP_FOLDER = "RACZoneGen"


def application_directory() -> Path:
    """Return the script or packaged executable directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def default_public_directory() -> Path:
    """Return %PUBLIC%/RACZoneGen (machine-global location)."""
    public_root = Path(os.environ.get("PUBLIC", r"C:\Users\Public"))
    return public_root / PUBLIC_APP_FOLDER


def find_config_file() -> tuple[Path, Path]:
    """
    Locate config.ini and the directory it lives in.

    Returns:
        (config_file, config_directory)

    Raises:
        FileNotFoundError if no config.ini exists in either candidate.
    """
    candidates = [
        application_directory() / DEFAULT_CONFIG_NAME,
        default_public_directory() / DEFAULT_CONFIG_NAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.parent

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Configuration file not found. Looked in: " + searched
    )


class Config:
    """Typed, validated access to every runtime setting."""

    def __init__(self) -> None:
        self.config_file, self.config_dir = find_config_file()

        parser = configparser.ConfigParser(interpolation=None)
        parser.read(self.config_file, encoding="utf-8-sig")

        self._parser = parser

        # --- application ---------------------------------------------------
        self.poll_interval_seconds = parser.getint(
            "application", "poll_interval_seconds"
        )
        self.log_level = parser.get(
            "application", "log_level", fallback="INFO"
        ).strip().upper()

        # --- database ------------------------------------------------------
        self.odbc_connection_string = parser.get(
            "database", "odbc_connection_string"
        )
        self.db_query_timeout = parser.getint(
            "database", "query_timeout_seconds", fallback=60
        )

        # --- rac -----------------------------------------------------------
        self.rac_query_file = self._path("rac", "query_file")
        self.time_cutoff_minutes = parser.getint(
            "rac", "time_cutoff_minutes"
        )
        self.grouping_radius_metres = parser.getfloat(
            "rac", "grouping_radius_metres"
        )
        self.grouping_score = parser.getfloat("rac", "grouping_score")
        self.minimum_event_count = parser.getint(
            "rac", "minimum_event_count"
        )
        self.top_n = parser.getint("rac", "top_n")
        self.threshold_inclusive = parser.getboolean(
            "rac", "threshold_inclusive"
        )

        # --- lanes ---------------------------------------------------------
        self.lane_query_file = self._path("lanes", "query_file")
        self.lane_refresh_interval_seconds = parser.getint(
            "lanes", "refresh_interval_seconds", fallback=300
        )

        # --- zone ----------------------------------------------------------
        self.zone_template_file = self._path("zone", "template_file")
        self.zone_output_directory = self._path("zone", "output_directory")
        self.zone_name_prefix = parser.get(
            "zone", "name_prefix"
        ).strip()
        self.zone_size_metres = parser.getfloat(
            "zone", "zone_size_metres"
        )
        self.zone_elevation = parser.getfloat("zone", "elevation")
        self.suppression_radius_metres = parser.getfloat(
            "zone", "suppression_radius_metres"
        )
        self.zones_query_file = self._path("zone", "zones_query_file")
        self.zones_refresh_interval_seconds = parser.getint(
            "zone", "zones_refresh_interval_seconds", fallback=300
        )

        # --- speed limit ---------------------------------------------------
        self.default_speed_kmh = parser.getfloat(
            "speed", "default_speed_kmh"
        )
        self.speed_offset_kmh = parser.getfloat(
            "speed", "speed_offset_kmh"
        )
        self.minimum_speed_kmh = parser.getfloat(
            "speed", "minimum_speed_kmh"
        )
        self.maximum_speed_kmh = parser.getfloat(
            "speed", "maximum_speed_kmh"
        )

        # --- minestar ------------------------------------------------------
        self.mstar_bin_directory = self._path(
            "minestar", "mstar_bin_directory"
        )
        self.mstar_executable = parser.get(
            "minestar", "executable"
        ).strip()
        self.current_zones_file = self._path(
            "minestar", "current_zones_file"
        )
        self.command_timeout_seconds = parser.getint(
            "minestar", "timeout_seconds"
        )
        self.import_enabled = parser.getboolean(
            "minestar", "import_enabled"
        )
        self.dry_run = parser.getboolean("minestar", "dry_run")
        self.allow_outside_mine_boundary = parser.getboolean(
            "minestar", "allow_outside_mine_boundary"
        )
        self.import_workers = max(
            1, parser.getint("minestar", "import_workers", fallback=1)
        )

        # --- display -------------------------------------------------------
        self.display_enabled = parser.getboolean(
            "display", "enabled"
        )

        self.log_directory = self.config_dir / "logs"
        self.runtime_directory = self.config_dir / "runtime"

        self.validate()

    def _path(self, section: str, option: str) -> Path:
        """Resolve a configured path relative to the config directory."""
        configured_value = self._parser.get(section, option).strip()
        path = Path(os.path.expandvars(configured_value))

        if not path.is_absolute():
            path = self.config_dir / path

        return path.resolve()

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
            raise ValueError(
                "suppression_radius_metres must be >= 0"
            )

        if self.minimum_speed_kmh < 0:
            raise ValueError("minimum_speed_kmh must be >= 0")

        if self.maximum_speed_kmh < self.minimum_speed_kmh:
            raise ValueError(
                "maximum_speed_kmh must be >= minimum_speed_kmh"
            )