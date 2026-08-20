"""
RACZoneGen — RAC Automatic Zone Monitor

Polls the MineStar `mshist` database for RAC health events, groups
nearby events into severity clusters, and — when a qualifying cluster
appears and is not suppressed by an existing zone — builds a square
MineStar speed-limit zone and imports it with mstarrun.bat on a
background thread so the live map stays responsive.

Zones are only ever generated for RAC events that occur *after* the
program starts.  The first successful sample is a pure baseline.

Flow per cycle:
1. Read RAC events (must succeed or the cycle is skipped).
2. Mark events new since the baseline.
3. Group events into clusters (Kyle's hub-radius algorithm).
4. Refresh lanes at a throttled rate (failure is non-fatal).
5. Export current zones (failure blocks imports that cycle).
6. For each non-suppressed qualifying cluster: build the zone XML and
   submit the MineStar import to a background worker.
7. Withhold the sample on import failure so new events are retried.
8. Redraw the map and event panel.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from clustering import find_clusters, mark_new_events
from config import Config
from database import Database
from minestar import (
    ExistingZone,
    MineStar,
    cluster_is_suppressed,
)
from viewer import Viewer, wait_for_next_cycle

logger = logging.getLogger("rac_zone_monitor")


def setup_logging(config: Config) -> None:
    """Rotating file + console logging under the config directory."""

    log_directory = config.log_directory
    log_directory.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rac_zone_monitor")
    root.setLevel(getattr(logging, config.log_level, logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_directory / "rac_zone_monitor.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def wait_with_viewer(
    future,
    viewer: Viewer | None,
    timeout: int,
) -> object:
    """Wait for a background future while keeping the viewer responsive.

    Blocks until the future completes (or the timeout elapses) but pumps
    the matplotlib event loop in small slices so the window never shows
    "Not responding" while a MineStar/DB worker is busy.
    """
    import concurrent.futures

    time_budget = time.monotonic() + timeout

    while True:
        try:
            return future.result(timeout=0.05)
        except concurrent.futures.TimeoutError:
            if viewer is not None:
                viewer.pump()
            if time.monotonic() > time_budget:
                raise TimeoutError(
                    "Background MineStar task exceeded its timeout."
                )
        except concurrent.futures.CancelledError:
            raise


def _read_zones(
    database: Database,
    minestar: MineStar,
    config: Config,
    viewer: Viewer | None,
    executor: ThreadPoolExecutor,
) -> list:
    """
    Refresh the in-memory zone library.

    Prefers a fast direct SQL read of `msmodel.dbo.ZONE`; falls back to
    `mstarrun exportZones` if the direct query is unavailable (handled
    in MineStar.export_current_zones).  The work runs on the background
    executor with the GUI pumped while it completes.
    """
    try:
        future = executor.submit(
            database.read_existing_zones,
            config.zones_query_file,
            config.zone_name_prefix,
        )
        return wait_with_viewer(
            future, viewer, timeout=config.db_query_timeout
        )
    except Exception as direct_error:
        logger.warning(
            "Direct zone read failed, falling back to mstarrun "
            "exportZones: %s",
            direct_error,
        )
        future = executor.submit(minestar.export_current_zones)
        return wait_with_viewer(
            future, viewer, timeout=config.command_timeout_seconds
        )


def run_cycle(
    config: Config,
    database: Database,
    minestar: MineStar,
    viewer: Viewer,
    executor: ThreadPoolExecutor,
    previous_event_counts: Counter | None,
    existing_zones: list,
    last_lanes: pd.DataFrame,
    last_lane_refresh: float,
    last_zone_refresh: float,
) -> tuple[Counter | None, list, pd.DataFrame, float, float]:
    """One full polling cycle. Returns updated state."""

    # ---- 1. RAC query must succeed for the cycle to continue -----------
    raw_events = database.read_rac_events(
        config.rac_query_file,
        config.time_cutoff_minutes,
    )

    events, current_event_counts, new_event_count = mark_new_events(
        raw_events,
        previous_event_counts,
    )

    first_sample = previous_event_counts is None

    clusters = find_clusters(
        events,
        radius=config.grouping_radius_metres,
        minimum_events=config.minimum_event_count,
        score_threshold=config.grouping_score,
        threshold_inclusive=config.threshold_inclusive,
        top_n=config.top_n,
    )

    # ---- 2. Lanes (throttled; failure is non-fatal) ---------------------
    now = time.monotonic()

    if (
        lanes_are_stale := (
            (now - last_lane_refresh)
            >= config.lane_refresh_interval_seconds
        )
        or last_lanes.empty
    ):
        try:
            last_lanes = database.read_lanes(config.lane_query_file)
            last_lane_refresh = now
        except Exception:
            logger.exception(
                "Lane refresh failed. Retaining previous lanes."
            )

    # ---- 3. Refresh existing zones (throttled; failure blocks imports) --
    zones_are_current = True

    if (now - last_zone_refresh) >= config.zones_refresh_interval_seconds:
        zones_are_current = False
        try:
            existing_zones = _read_zones(database, minestar, config, viewer, executor)
            last_zone_refresh = now
            zones_are_current = True
        except Exception:
            logger.exception(
                "MineStar zone refresh failed. New zone imports are "
                "blocked this cycle."
            )
    else:
        zones_are_current = True

    # ---- 4. New events → clusters → zones → background import ----------
    commit_event_sample = True

    if first_sample:
        logger.info(
            "Initial RAC baseline captured: %d event(s). No zones "
            "will be generated or imported.",
            len(events),
        )
    elif events.empty:
        logger.info(
            "No RAC events in the last %d minutes. No action.",
            config.time_cutoff_minutes,
        )
    elif new_event_count == 0:
        logger.info("No new RAC events since the previous poll. No action.")
    else:
        actionable_clusters = [
            cluster for cluster in clusters if cluster.is_new
        ]

        if not actionable_clusters:
            logger.info(
                "%d new RAC event(s) detected, but no group exceeded "
                "the configured threshold. No action.",
                new_event_count,
            )
        elif not zones_are_current:
            commit_event_sample = False
            logger.error(
                "%d qualifying cluster(s) found, but current MineStar "
                "zones could not be read. No XML or imports created. "
                "Will retry.",
                len(actionable_clusters),
            )
        else:
            import_failed = False
            submitted = 0

            for cluster in actionable_clusters:
                if cluster_is_suppressed(
                    cluster,
                    existing_zones,
                    config.suppression_radius_metres,
                ):
                    logger.info(
                        "Cluster suppressed at X=%.2f Y=%.2f. "
                        "Events=%d, new=%d, score=%.2f.",
                        cluster.center_x,
                        cluster.center_y,
                        cluster.event_count,
                        cluster.new_event_count,
                        cluster.score,
                    )
                    continue

                speed_limit = minestar.compute_speed_limit(cluster)

                xml_file, zone_name = minestar.create_zone_xml(
                    cluster, speed_limit
                )

                logger.info(
                    "Generated zone %s at X=%.2f Y=%.2f. "
                    "Events=%d, new=%d, score=%.2f, speed limit=%.1f "
                    "km/h (avg %.1f km/h).",
                    zone_name,
                    cluster.center_x,
                    cluster.center_y,
                    cluster.event_count,
                    cluster.new_event_count,
                    cluster.score,
                    speed_limit,
                    (
                        cluster.average_speed_kmh
                        if cluster.average_speed_kmh
                        else config.default_speed_kmh
                    ),
                )

                future = executor.submit(minestar.import_zone, xml_file)
                cluster._import_future = future  # type: ignore[attr-defined]
                cluster._zone_name = zone_name  # type: ignore[attr-defined]
                submitted += 1

            # Collect background import results (pumping the GUI while
            # any import subprocess is still running).
            done = 0
            for cluster in actionable_clusters:
                future = getattr(cluster, "_import_future", None)
                if future is None:
                    continue

                done += 1
                try:
                    status = wait_with_viewer(
                        future,
                        viewer,
                        timeout=config.command_timeout_seconds,
                    )
                except Exception:
                    logger.exception(
                        "Failed to await import result for %s. "
                        "Treating it as an import failure.",
                        getattr(cluster, "_zone_name", "unknown"),
                    )
                    import_failed = True
                    continue

                if status == "imported":
                    # Prevent another cluster in this same cycle from
                    # creating a nearby zone.
                    existing_zones.append(
                        ExistingZone(
                            name=getattr(
                                cluster, "_zone_name", "unknown"
                            ),
                            center_x=cluster.center_x,
                            center_y=cluster.center_y,
                            half_size=config.zone_size_metres / 2.0,
                        )
                    )
                elif status == "failed":
                    import_failed = True

            logger.info(
                "%d zone(s) submitted for import, %d result(s) collected.",
                submitted,
                done,
            )

            if import_failed:
                # Retry new events after a genuine import failure.
                # Disabled and dry-run tests are not repeated.
                commit_event_sample = False
                logger.warning(
                    "At least one import failed. Current RAC sample "
                    "will be retried next cycle."
                )

    # ---- 5. Commit semantics --------------------------------------------
    if commit_event_sample:
        previous_event_counts = current_event_counts
    else:
        logger.warning(
            "Current RAC sample was not committed. New events will be "
            "retried next cycle."
        )

    # ---- 6. Viewer ------------------------------------------------------
    if viewer is not None:
        for cluster in clusters:
            viewer.log_activity(
                f"Clu X={cluster.center_x:.0f} Y={cluster.center_y:.0f} "
                f"{cluster.event_count}evt {cluster.score:.0f}pt"
            )

        for _, row in events.head(14).iterrows():
            viewer.add_event(row)

        viewer.update(
            events,
            last_lanes,
            clusters,
            existing_zones,
            suppression_radius=config.suppression_radius_metres,
            zone_size=config.zone_size_metres,
            grouping_radius=config.grouping_radius_metres,
        )

    return (
        previous_event_counts,
        existing_zones,
        last_lanes,
        last_lane_refresh,
        last_zone_refresh,
    )


def main() -> None:
    config = Config()
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("RACZoneGen starting")
    logger.info("Config file: %s", config.config_file)
    logger.info("Polling interval: %d s", config.poll_interval_seconds)
    logger.info(
        "Zone size: %.0f m x %.0f m | suppression radius: %.0f m",
        config.zone_size_metres,
        config.zone_size_metres,
        config.suppression_radius_metres,
    )
    logger.info(
        "Speed limit = avg - %.0f km/h, floor %.0f, default %.0f",
        config.speed_offset_kmh,
        config.minimum_speed_kmh,
        config.default_speed_kmh,
    )
    logger.info(
        "Automatic imports enabled=%s, dry_run=%s",
        config.import_enabled,
        config.dry_run,
    )

    config.zone_output_directory.mkdir(parents=True, exist_ok=True)
    config.runtime_directory.mkdir(parents=True, exist_ok=True)

    database = Database(
        config.odbc_connection_string,
        query_timeout=config.db_query_timeout,
    )

    minestar = MineStar(config)

    viewer = Viewer() if config.display_enabled else None

    executor = ThreadPoolExecutor(
        max_workers=config.import_workers,
        thread_name_prefix="mstar-import",
    )

    previous_event_counts: Counter | None = None
    existing_zones: list = []

    last_lanes = pd.DataFrame(
        columns=["LANE_OID", "Segment", "POINT_NR", "X", "Y"]
    )
    last_lane_refresh = 0.0
    last_zone_refresh = 0.0

    # ---- Startup: pull lanes + build in-memory zone library -------------
    try:
        lane_future = executor.submit(
            database.read_lanes, config.lane_query_file
        )
        last_lanes = wait_with_viewer(
            lane_future, viewer, timeout=config.db_query_timeout * 2
        )
        last_lane_refresh = time.monotonic()
        logger.info("Initial lane load complete.")
    except Exception:
        logger.exception("Initial lane load failed. The map will be empty.")

    try:
        existing_zones = _read_zones(
            database, minestar, config, viewer, executor
        )
        last_zone_refresh = time.monotonic()
        logger.info(
            "In-memory zone library: %d existing %s* zone(s).",
            len(existing_zones),
            config.zone_name_prefix,
        )
    except Exception:
        logger.exception(
            "Initial MineStar zone library could not be built. "
            "Zone creation will be blocked until zones refresh."
        )

    try:
        while True:
            cycle_started = time.monotonic()

            try:
                (
                    previous_event_counts,
                    existing_zones,
                    last_lanes,
                    last_lane_refresh,
                    last_zone_refresh,
                ) = run_cycle(
                    config,
                    database,
                    minestar,
                    viewer,
                    executor,
                    previous_event_counts,
                    existing_zones,
                    last_lanes,
                    last_lane_refresh,
                    last_zone_refresh,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception(
                    "Polling cycle failed. No zones imported this cycle."
                )

            elapsed = time.monotonic() - cycle_started
            wait_seconds = max(
                int(config.poll_interval_seconds - elapsed), 1
            )

            if not wait_for_next_cycle(wait_seconds, viewer):
                logger.info("Viewer window closed.")
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user.")

    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        database.dispose()
        logger.info("RACZoneGen stopped.")


if __name__ == "__main__":
    main()