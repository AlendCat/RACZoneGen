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
    return public_root / "Documents" / PUBLIC_APP_FOLDER


def _create_public_base_config() -> Path | None:
    import shutil
    public_dir = default_public_directory()
    public_config = public_dir / DEFAULT_CONFIG_NAME
    if public_config.is_file():
        return public_config
    try:
        public_dir.mkdir(parents=True, exist_ok=True)
        template = application_directory() / DEFAULT_CONFIG_NAME
        if template.is_file():
            shutil.copy2(template, public_config)
        else:
            public_config.write_text(
                application_directory().joinpath(DEFAULT_CONFIG_NAME).read_text(encoding="utf-8-sig")
                if (application_directory() / DEFAULT_CONFIG_NAME).exists()
                else DEFAULT_CONFIG_TEXT,
                encoding="utf-8",
            )
        sql_src = application_directory() / "sql"
        sql_dst = public_dir / "sql"
        if sql_src.is_dir() and not sql_dst.is_dir():
            shutil.copytree(sql_src, sql_dst, dirs_exist_ok=True)
        zones_src = application_directory() / "zones.xml"
        zones_dst = public_dir / "zones.xml"
        if zones_src.is_file() and not zones_dst.is_file():
            shutil.copy2(zones_src, zones_dst)
        return public_config if public_config.is_file() else None
    except Exception:
        return None


DEFAULT_CONFIG_TEXT = """[application]
poll_interval_seconds = 8
[database]
server = C001749812\\MINESTAR
database = mshist
sql_directory = sql
use_windows_authentication = true
username =
password =
[rac]
grouping_radius_metres = 50
grouping_score = 8
minimum_event_count = 5
# Levels (0-5) allowed to FORM/JOIN a cluster for zone creation.
# Comma-separated list. Empty = all levels count.
# To ignore level-1 noise:  cluster_levels = 2,3,4,5
# Full set (same as empty): cluster_levels = 1,2,3,4,5
cluster_levels =
[zone]
output_directory = output
zone_size_metres = 100
suppression_radius_metres = 300
zones_refresh_interval_seconds = 5
[lanes]
refresh_interval_seconds = 21600
[minestar]
mstarrun_location = D:\\mstar\\mstarHome\\bus\\bin\\mstarrun.bat
[simulation]
enabled = false
import_enabled = false
random_seed = 20260824
spawn_per_cycle = 1
jitter_metres = 500
level_min = 0
level_max = 5
speed_min = 1
speed_max = 60
time_window_minutes = 120
hotspots = 100.0,100.0;340.0,180.0;620.0,420.0
"""


def find_config_file() -> tuple[Path, Path]:
    is_frozen = getattr(sys, "frozen", False)
    if is_frozen:
        candidates = [
            default_public_directory() / DEFAULT_CONFIG_NAME,
            application_directory() / DEFAULT_CONFIG_NAME,
        ]
    else:
        candidates = [
            application_directory() / DEFAULT_CONFIG_NAME,
            default_public_directory() / DEFAULT_CONFIG_NAME,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.parent
    created = _create_public_base_config()
    if created is not None and created.is_file():
        return created, created.parent
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
        self.use_windows_authentication = parser.getboolean(
            "database", "use_windows_authentication", fallback=True
        )
        raw_username = parser.get("database", "username", fallback="").strip()
        raw_password = parser.get("database", "password", fallback="").strip()
        # Allow %ENV_VAR% expansion so passwords need not live in cleartext
        # in config.ini (e.g. password = %RAC_DB_PASSWORD%).
        self.db_username = os.path.expandvars(raw_username)
        self.db_password = os.path.expandvars(raw_password)
        if self.use_windows_authentication:
            self.odbc_connection_string = (
                f"Driver={{SQL Server}};Server={server};"
                f"Database={database};Trusted_Connection=yes;"
            )
        else:
            if not self.db_username or not self.db_password:
                raise ValueError(
                    "database username/password must be set when "
                    "use_windows_authentication = false"
                )
            self.odbc_connection_string = (
                f"Driver={{SQL Server}};Server={server};"
                f"Database={database};"
                f"UID={self.db_username};PWD={self._escape_odbc_value(self.db_password)};"
            )
        self.db_query_timeout = 60
        if parser.has_option("database", "sql_directory"):
            self.sql_directory = self._path("database", "sql_directory")
        else:
            self.sql_directory = (self.config_dir / "sql").resolve()
        self.rac_query_file = (self.sql_directory / "rac_events.sql").resolve()
        self.lane_query_file = (self.sql_directory / "lanes.sql").resolve()
        self.zones_query_file = (self.sql_directory / "zones.sql").resolve()
        self.time_cutoff_minutes = 120
        raw_levels = parser.get("rac", "cluster_levels", fallback="")
        self.grouping_radius_metres = parser.getfloat("rac", "grouping_radius_metres")
        self.grouping_score = parser.getfloat("rac", "grouping_score")
        self.minimum_event_count = parser.getint("rac", "minimum_event_count")
        # Levels allowed to FORM/JOIN a cluster (allow-list). Empty set =
        # no filtering. e.g. {2,3,4,5} makes level-1 events invisible to
        # zone creation (they still display on the map and table).
        self.cluster_levels = self._parse_cluster_levels(raw_levels)
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
        self.minimum_speed_kmh = 11.0
        self.maximum_speed_kmh = 60.0
        # Optional [speed] overrides (all have safe fallbacks so old
        # config.ini files without a [speed] section keep working).
        try:
            if parser.has_option("speed", "payload_speed_divisor"):
                self.payload_speed_divisor = parser.getfloat("speed", "payload_speed_divisor")
            else:
                self.payload_speed_divisor = 0.0
            if parser.has_option("speed", "speed_offset_kmh"):
                self.speed_offset_kmh = parser.getfloat("speed", "speed_offset_kmh")
            if parser.has_option("speed", "minimum_speed_kmh"):
                self.minimum_speed_kmh = parser.getfloat("speed", "minimum_speed_kmh")
            if parser.has_option("speed", "maximum_speed_kmh"):
                self.maximum_speed_kmh = parser.getfloat("speed", "maximum_speed_kmh")
            if parser.has_option("speed", "default_speed_kmh"):
                self.default_speed_kmh = parser.getfloat("speed", "default_speed_kmh")
        except Exception:
            pass
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
        # Import gates — actually read from config (previously hardcoded, so
        # these settings were silently ignored and live ALWAYS imported).
        # Safe defaults: read-only until explicitly enabled (matches docs).
        self.import_enabled = parser.getboolean("minestar", "import_enabled", fallback=False)
        self.dry_run = parser.getboolean("minestar", "dry_run", fallback=True)
        self.show_only = parser.getboolean("minestar", "show_only", fallback=False)
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

    @staticmethod
    def _escape_odbc_value(value: str) -> str:
        # ODBC connection-string escaping: wrap in {} if the value contains
        # a semicolon or braces; a literal } is escaped as }}.
        if any(char in value for char in (";", "{", "}")):
            return "{" + value.replace("}", "}}") + "}"
        return value

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

    def _parse_cluster_levels(self, raw: str) -> frozenset[int]:
        """[rac] cluster_levels — allow-list of severities for clustering.

        Accepts comma (or space/semicolon) separated integers, empty for
        "all levels". Anything non-numeric or outside 0-5 is a hard error
        so a typo never silently widens zone creation.
        """
        levels: set[int] = set()
        for token in raw.replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(float(token))
            except ValueError:
                raise ValueError(
                    f"cluster_levels has a non-numeric entry: {token!r}"
                )
            if not 0 <= value <= 5:
                raise ValueError(
                    f"cluster_levels entry out of range 0-5: {value}"
                )
            levels.add(value)
        return frozenset(levels)

    def validate(self) -> None:
        if self.poll_interval_seconds < 1:
            raise ValueError("poll_interval_seconds must be at least 1")
        if self.grouping_radius_metres <= 0:
            raise ValueError("grouping_radius_metres must be > 0")
        if self.minimum_event_count < 1:
            raise ValueError("minimum_event_count must be at least 1")
        for level in self.cluster_levels:
            if not 0 <= level <= 5:
                raise ValueError(
                    f"cluster_levels entry out of range 0-5: {level}"
                )
        if self.zone_size_metres <= 0:
            raise ValueError("zone_size_metres must be > 0")
        if self.suppression_radius_metres < 0:
            raise ValueError("suppression_radius_metres must be >= 0")
