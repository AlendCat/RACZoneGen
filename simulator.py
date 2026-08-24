"""
Simulated RAC event source for RACZoneGen validation.

Wraps the live `Database` so that **everything except RAC events is read
from the real MineStar SQL Server** — lanes, existing zones, and the
direct zone queries all pass through to the production database.

Only the RAC health events are faked: this source synthesises an
*additive* stream of new events so you can watch the real pipeline
(baseline → clustering → suppression → zone creation/import) process
growing event counts over time, without waiting for real RAC alarms.

Interface (duck-typed to `Database`):

    read_rac_events(query_file, time_cutoff_minutes)
    read_lanes(query_file)
    read_existing_zones(query_file, name_prefix)
    dispose()

Every `read_rac_events` call after the first (the pure baseline) spawns
`spawn_per_cycle` new events per configured hotspot with Gaussian jitter
and random level / speed.  Event timestamps are stored and stable, so the
real new-event detection (`mark_new_events`) flags exactly the freshly
spawned events; events older than the configured cutoff are pruned.

Payloads are emitted as JSON (e.g. ``{"Speed": 51.4}``) so the genuine
payload-speed parser in `database._payload_speed` is exercised too.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from database import Database, _payload_speed

logger = logging.getLogger("rac_zone_monitor")


class SimulatedDatabase:
    """
    Fake RAC-event source layered over the live database.

    Only `read_rac_events` is synthetic; `read_lanes` and
    `read_existing_zones` delegate to the wrapped real `Database`, so the
    map shows the current autonomous lanes and MineStar zones.
    """

    def __init__(self, config, database: Database) -> None:
        self.config = config
        self._database = database
        self._rng = np.random.default_rng(config.simulation_seed)
        self._events: list[dict] = []
        self._read_count = 0

        self._site_bounds = self._resolve_site_bounds()

    # ------------------------------------------------------------------
    # Site extent (so synthetic events fall inside the drawn map)
    # ------------------------------------------------------------------

    def _resolve_site_bounds(
        self,
    ) -> tuple[float, float, float, float] | None:
        """
        Derive the map X/Y extent from the live lane network.

        Generated events are clamped to these bounds so they always land
        inside the drawn map.  Returns None when the extent cannot be
        determined (no lanes / DB offline) — events are then unclamped.
        """
        try:
            lanes = self._database.read_lanes(self.config.lane_query_file)
        except Exception as error:
            logger.warning(
                "[sim] Could not read lanes for site bounds; RAC events "
                "will be unclamped. %s",
                error,
            )
            return None

        if lanes is None or lanes.empty:
            return None

        min_x = float(lanes["X"].min())
        max_x = float(lanes["X"].max())
        min_y = float(lanes["Y"].min())
        max_y = float(lanes["Y"].max())

        if max_x - min_x < 1.0 or max_y - min_y < 1.0:
            return None

        logger.info(
            "[sim] Site bounds for RAC generation: "
            "X [%.1f, %.1f], Y [%.1f, %.1f].",
            min_x,
            max_x,
            min_y,
            max_y,
        )

        return min_x, max_x, min_y, max_y

    def _clamp_point(
        self,
        x_value: float,
        y_value: float,
        inset: float = 0.0,
    ) -> tuple[float, float]:
        """Clamp a coordinate into the site bounds (with a margin)."""
        if self._site_bounds is None:
            return x_value, y_value

        min_x, max_x, min_y, max_y = self._site_bounds
        return (
            min(max(x_value, min_x + inset), max_x - inset),
            min(max(y_value, min_y + inset), max_y - inset),
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        """Release the wrapped real database."""
        self._database.dispose()

    def read_rac_events(
        self,
        query_file: Path,
        time_cutoff_minutes: int,
    ) -> pd.DataFrame:
        """
        Return the synthetic (pruned) event store, spawning a new batch
        first.  The first call returns an empty store — that is the
        baseline poll.  Every subsequent call spawns `spawn_per_cycle`
        new events per configured hotspot so clusters appear within a
        couple of cycles.
        """
        del query_file  # unused; interface parity only

        self._read_count += 1

        if self._read_count > 1:
            self._spawn_batch()

        self._prune(time_cutoff_minutes)

        events = pd.DataFrame(
            self._events,
            columns=["Time", "X", "Y", "Level", "Payload"],
        )

        if events.empty:
            logger.info(
                "[sim] RAC sample: 0 event(s). Synthetic source only.",
            )
        else:
            events["Speed"] = events["Payload"].map(_payload_speed)
            events[["X", "Y", "Level"]] = events[
                ["X", "Y", "Level"]
            ].astype(float)
            logger.info(
                "[sim] Synthetic RAC sample: %d event(s) across %d "
                "hotspot(s).",
                len(events),
                len(self.config.simulation_hotspots),
            )

        return events.reset_index(drop=True)

    def read_lanes(
        self,
        query_file: Path,
    ) -> pd.DataFrame:
        """Read the live autonomous lane network from MineStar."""
        return self._database.read_lanes(query_file)

    def read_existing_zones(
        self,
        query_file: Path,
        name_prefix: str,
    ) -> list:
        """Read the current MineStar zones directly from the database."""
        return self._database.read_existing_zones(
            query_file,
            name_prefix,
        )

    # ------------------------------------------------------------------
    # Synthetic RAC generation
    # ------------------------------------------------------------------

    def _spawn_batch(self) -> None:
        """Spawn one fresh batch of events across the configured hotspots."""
        spawn_count = max(1, self.config.simulation_spawn_per_cycle)
        now = dt.datetime.now().astimezone()

        site_inset = 5.0

        for center_x, center_y, level, speed in (
            self.config.simulation_hotspots
        ):
            # Pull the anchor hotspot inside the site footprint so every
            # generated event lands on the map.
            anchor_x, anchor_y = self._clamp_point(
                center_x,
                center_y,
                inset=site_inset,
            )

            for _ in range(spawn_count):
                jitter = self._rng.normal(
                    0.0, self.config.simulation_jitter_metres, size=2
                )

                if level is None:
                    level = float(
                        self._rng.integers(
                            self.config.simulation_level_min,
                            self.config.simulation_level_max + 1,
                        )
                    )
                if speed is None:
                    speed = float(
                        self._rng.uniform(
                            self.config.simulation_speed_min,
                            self.config.simulation_speed_max,
                        )
                    )

                event_x, event_y = self._clamp_point(
                    anchor_x + float(jitter[0]),
                    anchor_y + float(jitter[1]),
                )

                self._events.append(
                    {
                        "Time": now,
                        "X": event_x,
                        "Y": event_y,
                        "Level": level,
                        "Payload": json.dumps(
                            {
                                "Speed": round(float(speed), 1),
                                "VehicleSpeed": round(float(speed), 1),
                            }
                        ),
                    }
                )

    def _prune(self, cutoff_minutes: int) -> None:
        """Drop synthetic events older than the configured time window."""
        if not self._events:
            return

        now = dt.datetime.now().astimezone()
        cutoff = now - dt.timedelta(minutes=cutoff_minutes)

        self._events = [
            event for event in self._events if event["Time"] >= cutoff
        ]