"""
RAC event clustering.

Implements the hub-radius grouping originally written by Kyle McIlroy:

* A seed event (highest X, then row order) that has not been consumed
  becomes the centre of a candidate group.
* Every *currently unassigned* event within ``grouping_radius`` of the
  seed is a member.
* The group qualifies when it has enough members and the summed level
  exceeds the configured threshold.  All members are then consumed.
* The zone is centred on the member average, inherits the grid score,
  and reports how many members were first seen this cycle.

Also provides the new-event detection used to decide *when* a group
should be acted on (only events that appeared after program start).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Cluster:
    """A qualifying group of RAC events."""

    center_x: float
    center_y: float
    search_x: float  # hub (seed) location
    search_y: float
    score: float
    event_count: int
    new_event_count: int
    average_speed_kmh: float | None = None
    member_speeds: list[float] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        """True when at least one member event is new this cycle."""
        return self.new_event_count > 0


def event_signatures(
    events: pd.DataFrame,
) -> list[tuple[int, float, float, float]]:
    """
    Build stable event signatures (time, X, Y, level).

    Level is included so an event whose severity escalates between polls
    is treated as a new event rather than silently ignored.
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
    Mark events that were not present in the previous successful sample.

    The very first sample becomes a pure baseline: nothing is flagged
    new and nothing triggers zone creation.  This guarantees the program
    never generates zones for RAC events that happened before startup.
    """
    events = events.copy()
    signatures = event_signatures(events)
    current_counts = Counter(signatures)

    if previous_counts is None:
        events["_IsNew"] = False
        return events, current_counts, 0

    remaining_new: Counter = Counter()

    for signature, count in current_counts.items():
        additional = count - previous_counts.get(signature, 0)
        if additional > 0:
            remaining_new[signature] = additional

    new_flags: list[bool] = []

    for signature in signatures:
        if remaining_new.get(signature, 0) > 0:
            new_flags.append(True)
            remaining_new[signature] -= 1
        else:
            new_flags.append(False)

    events["_IsNew"] = new_flags

    return events, current_counts, int(sum(new_flags))


def _speed_value(row: pd.Series) -> float | None:
    """Best-effort speed extraction from a RAC event row."""
    speed = row.get("Speed")

    if speed is None:
        return None

    try:
        value = float(pd.to_numeric(speed, errors="coerce"))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value) or value < 0:
        return None

    return value


def find_clusters(
    events: pd.DataFrame,
    *,
    radius: float,
    minimum_events: int,
    score_threshold: float,
    threshold_inclusive: bool,
    top_n: int,
) -> list[Cluster]:
    """
    Group RAC events using the hub-radius algorithm.

    The seed order is deterministic: sorted by X descending, then row
    order, so identical input always produces identical clusters.
    """
    if events.empty:
        return []

    working = (
        events
        .sort_values(["X", "_IsNew"], ascending=[False, True])
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

    speeds = np.asarray(
        [_speed_value(row) or np.nan for _, row in working.iterrows()],
        dtype=float,
    )

    assigned = np.zeros(len(working), dtype=bool)
    clusters: list[Cluster] = []

    for seed_index in range(len(working)):
        if assigned[seed_index] or not new_flags[seed_index]:
            continue

        distances = np.hypot(
            x_values - x_values[seed_index],
            y_values - y_values[seed_index],
        )

        member_indices = np.flatnonzero(
            (~assigned) & (distances < radius)
        )

        if len(member_indices) < minimum_events:
            continue

        score = float(levels[member_indices].sum())

        if threshold_inclusive:
            qualifies = score >= score_threshold
        else:
            qualifies = score > score_threshold

        if not qualifies:
            continue

        assigned[member_indices] = True

        member_speeds = [
            float(s) for s in speeds[member_indices]
            if np.isfinite(s) and s > 0
        ]

        clusters.append(
            Cluster(
                center_x=float(x_values[member_indices].mean()),
                center_y=float(y_values[member_indices].mean()),
                search_x=float(x_values[seed_index]),
                search_y=float(y_values[seed_index]),
                score=score,
                event_count=len(member_indices),
                new_event_count=int(new_flags[member_indices].sum()),
                average_speed_kmh=(
                    float(np.mean(member_speeds))
                    if member_speeds else None
                ),
                member_speeds=member_speeds,
            )
        )

    clusters.sort(key=lambda c: c.score, reverse=True)

    return clusters[:top_n]