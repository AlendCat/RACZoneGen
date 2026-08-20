"""
Live site map and event panel.

A responsive interactive matplotlib window:

* autonomous lane network drawn as a *cached* PatchCollection so the
  repaint per cycle is cheap even with ~4800 lanes / ~74k points
* RAC events scattered by severity + cyan halo on events new this cycle
* existing / newly placed square zones and their dashed suppression areas
* qualifying clusters (dotted hub radius, X marks the centroid)
* a side panel listing recent RAC events (time/X/Y/level/speed) and an
  activity log of zone generation / import decisions

The GUI event loop is pumped inside every blocking wait (`pump`) so the
window never freezes while a MineStar command or a DB query is running.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon, Rectangle

from clustering import Cluster
from minestar import ExistingZone, cluster_is_suppressed

PALETE_BG = "#121418"
PANEL_BG = "#0d0f12"
TEXT_PRIMARY = "#EAEAEA"
TEXT_GREY = "#9AA0A6"


class Viewer:
    """Live-updating site map with an event/activity side panel."""

    def __init__(self) -> None:
        plt.ion()

        self.figure = plt.figure(
            figsize=(15, 8.5),
            facecolor=PALETE_BG,
            constrained_layout=True,
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "RACZoneGen — RAC Automatic Zone Monitor"
            )
        except AttributeError:
            pass

        grid = self.figure.add_gridspec(
            1, 2, width_ratios=[3.15, 1.0], wspace=0.08
        )
        self.map_axis = self.figure.add_subplot(grid[0, 0])
        self.panel_axis = self.figure.add_subplot(grid[0, 1])

        self.map_axis.set_facecolor(PALETE_BG)
        self.map_axis.set_aspect("equal", adjustable="box")
        self.map_axis.set_xticks([])
        self.map_axis.set_yticks([])
        for spine in self.map_axis.spines.values():
            spine.set_edgecolor(TEXT_GREY)
            spine.set_linewidth(1.5)

        self.event_log: deque[str] = deque(maxlen=14)
        self.activity_log: deque[str] = deque(maxlen=12)

        # Cached lane collection.
        self._lane_collection: PatchCollection | None = None
        self._lane_signature: object = None
        self._lane_extent_x: list[float] = []
        self._lane_extent_y: list[float] = []

        # Dynamic artists that get discarded on every repaint.
        self._dynamic: list = []
        self._colorbar = None
        self._legend = None

        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._closed = False

        plt.show(block=False)

    def _on_close(self, _event) -> None:
        self._closed = True

    def is_open(self) -> bool:
        if self._closed:
            return False
        return plt.fignum_exists(self.figure.number)

    def pump(self, seconds: float = 0.01) -> None:
        """Process GUI events so the window stays responsive."""
        if self._closed or not self.is_open():
            return
        try:
            self.figure.canvas.flush_events()
            self.figure.canvas.draw_idle()
            plt.pause(seconds)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Logs / panel data
    # ------------------------------------------------------------------

    def add_event(self, row: pd.Series) -> None:
        """Add a single RAC event line to the panel."""
        time_text = pd.Timestamp(row["Time"]).strftime("%H:%M:%S")
        speed = row.get("Speed")
        speed_text = f"{float(speed):.0f}" if pd.notna(speed) else "-"
        self.event_log.appendleft(
            f"{time_text}  X {row['X']:>10.1f}  "
            f"Y {row['Y']:>10.1f}  Lv {int(row['Level']):>2}  "
            f"{speed_text:>3} km/h"
        )

    def log_activity(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.appendleft(f"{timestamp}  {message}")

    # ------------------------------------------------------------------
    # Lanes (cached)
    # ------------------------------------------------------------------

    @staticmethod
    def _lanes_signature(lanes: pd.DataFrame) -> object:
        if lanes.empty:
            return 0
        return (
            lanes.shape,
            tuple(sorted(lanes["LANE_OID"].unique().tolist())),
        )

    def _rebuild_lanes(self, lanes: pd.DataFrame) -> None:
        signature = self._lanes_signature(lanes)

        if signature == self._lane_signature:
            return

        if self._lane_collection is not None:
            self._lane_collection.remove()
            self._lane_collection = None

        self._lane_extent_x = []
        self._lane_extent_y = []

        if not lanes.empty:
            patches: list[Polygon] = []

            for _, lane in lanes.groupby("LANE_OID"):
                left = lane[
                    lane["Segment"].astype(str).str.lower() == "left"
                ].sort_values("POINT_NR")
                right = lane[
                    lane["Segment"].astype(str).str.lower() == "right"
                ].sort_values("POINT_NR")

                if left.empty or right.empty:
                    continue

                x_coords = (
                    left["X"].astype(float).tolist()
                    + right["X"].astype(float).tolist()[::-1]
                )
                y_coords = (
                    left["Y"].astype(float).tolist()
                    + right["Y"].astype(float).tolist()[::-1]
                )

                patches.append(
                    Polygon(
                        list(zip(x_coords, y_coords)),
                        closed=True,
                        facecolor="#1F3A22",
                        edgecolor="#35C759",
                        linewidth=0.3,
                        alpha=0.55,
                    )
                )

                self._lane_extent_x.extend(x_coords)
                self._lane_extent_y.extend(y_coords)

            if patches:
                self._lane_collection = PatchCollection(
                    patches, match_original=True
                )
                self.map_axis.add_collection(self._lane_collection)

        self._lane_signature = signature

    # ------------------------------------------------------------------

    def _discard_dynamic(self) -> None:
        if self._legend is not None:
            try:
                self._legend.remove()
            except (ValueError, AttributeError):
                pass
            self._legend = None

        for artist in self._dynamic:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._dynamic = []

    def update(
        self,
        events: pd.DataFrame,
        lanes: pd.DataFrame,
        clusters: list[Cluster],
        existing_zones: list[ExistingZone],
        suppression_radius: float,
        zone_size: float,
        grouping_radius: float,
    ) -> None:
        """Redraw the map and panel; lanes only when they actually change."""
        if not self.is_open():
            return

        axis = self.map_axis
        self._discard_dynamic()
        self._rebuild_lanes(lanes)

        all_x = list(self._lane_extent_x)
        all_y = list(self._lane_extent_y)

        handles = []

        # RAC events -----------------------------------------------------
        if not events.empty:
            minimum_level = float(events["Level"].min())
            maximum_level = float(events["Level"].max())
            if minimum_level == maximum_level:
                maximum_level = minimum_level + 1.0

            color_map = LinearSegmentedColormap.from_list(
                "rac_severity",
                ["#E4C45C", "#E8873C", "#E23B3B"],
            )

            collection = axis.scatter(
                events["X"],
                events["Y"],
                c=events["Level"],
                cmap=color_map,
                norm=Normalize(minimum_level, maximum_level),
                s=34,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.95,
                zorder=3,
                label="RAC events",
            )
            self._dynamic.append(collection)
            handles.append(collection)

            if self._colorbar is None:
                self._colorbar = self.figure.colorbar(
                    collection,
                    ax=axis,
                    pad=0.015,
                    shrink=0.8,
                )
                self._colorbar.ax.tick_params(colors=TEXT_GREY)
                self._colorbar.set_label(
                    "RAC Level", color=TEXT_GREY, fontweight="bold"
                )
            else:
                self._colorbar.update_normal(collection)
                self._colorbar.ax.set_visible(True)

            if "_IsNew" in events.columns:
                new_events = events[events["_IsNew"]]
                if not new_events.empty:
                    halo = axis.scatter(
                        new_events["X"],
                        new_events["Y"],
                        facecolors="none",
                        edgecolors="#35C9FF",
                        s=110,
                        linewidths=2,
                        zorder=5,
                        label="New RAC events",
                    )
                    self._dynamic.append(halo)
                    handles.append(halo)

            all_x.extend(events["X"].astype(float).tolist())
            all_y.extend(events["Y"].astype(float).tolist())
        elif self._colorbar is not None:
            self._colorbar.ax.set_visible(False)

        # Existing zones + suppression areas -------------------------------
        for zone in existing_zones:
            center = (zone.center_x, zone.center_y)
            half = max(zone.half_size, zone_size / 2.0)

            zone_patch = Rectangle(
                (zone.center_x - half, zone.center_y - half),
                2 * half,
                2 * half,
                facecolor="#B024E0",
                edgecolor="white",
                linewidth=1.4,
                alpha=0.45,
                zorder=4,
            )
            suppression_patch = Circle(
                center,
                suppression_radius,
                fill=False,
                edgecolor="#E06AFF",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
            )
            name_text = axis.text(
                zone.center_x,
                zone.center_y - half - 1.0,
                zone.name,
                color="#E7B4FF",
                fontsize=6.5,
                horizontalalignment="center",
                zorder=6,
            )
            axis.add_patch(zone_patch)
            axis.add_patch(suppression_patch)
            self._dynamic.extend(
                [zone_patch, suppression_patch, name_text]
            )

            all_x.append(zone.center_x - suppression_radius)
            all_x.append(zone.center_x + suppression_radius)
            all_y.append(zone.center_y - suppression_radius)
            all_y.append(zone.center_y + suppression_radius)

        # Qualifying clusters ---------------------------------------------
        for cluster in clusters:
            blocked = cluster_is_suppressed(
                cluster, existing_zones, suppression_radius
            )

            if blocked:
                color = "#6B7280"
            elif cluster.is_new:
                color = "#35C9FF"
            else:
                color = "#4C7DFF"

            hub = Circle(
                (cluster.search_x, cluster.search_y),
                grouping_radius,
                fill=False,
                edgecolor=color,
                linestyle=":",
                linewidth=1.1,
                alpha=0.65,
                zorder=2,
            )
            mark = axis.scatter(
                [cluster.center_x],
                [cluster.center_y],
                marker="x",
                color=color,
                s=110,
                linewidths=2.2,
                zorder=6,
            )
            speed_text = (
                f"{cluster.average_speed_kmh:.0f}"
                if cluster.average_speed_kmh
                else "-"
            )
            label = axis.text(
                cluster.center_x,
                cluster.center_y,
                (
                    f" {cluster.event_count} evt"
                    f", {cluster.new_event_count} new"
                    f", spd {speed_text}"
                ),
                color="white",
                fontsize=7,
                zorder=7,
            )
            axis.add_patch(hub)
            self._dynamic.extend([hub, mark, label])

            all_x.append(cluster.center_x)
            all_y.append(cluster.center_y)

        # Bounds -----------------------------------------------------------
        if all_x and all_y:
            pad = max(
                max(all_x) - min(all_x),
                max(all_y) - min(all_y),
            ) * 0.04
            pad = max(pad, 25.0)
            axis.set_xlim(min(all_x) - pad, max(all_x) + pad)
            axis.set_ylim(min(all_y) - pad, max(all_y) + pad)

        new_event_count = int(
            events["_IsNew"].sum() if "_IsNew" in events.columns else 0
        )

        axis.set_title(
            (
                f"RAC events: {len(events)}  |  New: {new_event_count}"
                f"  |  Zones: {len(existing_zones)}"
            ),
            color=TEXT_PRIMARY,
            fontsize=12,
            fontweight="bold",
            loc="left",
        )

        if handles:
            self._legend = axis.legend(
                handles=handles,
                loc="upper right",
                fontsize=8,
                framealpha=0.5,
            )

        self._draw_panel()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _draw_panel(self) -> None:
        panel = self.panel_axis
        panel.clear()
        panel.set_facecolor(PANEL_BG)
        panel.axis("off")

        panel.text(
            0.02,
            0.975,
            "LIVE RAC EVENTS",
            transform=panel.transAxes,
            color=TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

        panel.text(
            0.02,
            0.945,
            "time   X            Y          Lv  km/h",
            transform=panel.transAxes,
            color=TEXT_GREY,
            fontsize=6.5,
        )

        event_count = len(self.event_log)
        event_top = 0.925
        event_bottom = 0.42

        for index, line in enumerate(self.event_log):
            y_position = event_top - index * (
                (event_top - event_bottom) / max(event_count - 1, 1)
            )
            panel.text(
                0.02,
                y_position,
                line,
                transform=panel.transAxes,
                color=TEXT_PRIMARY,
                fontsize=6.5,
                fontfamily="monospace",
            )

        panel.text(
            0.02,
            0.38,
            "ACTIVITY LOG",
            transform=panel.transAxes,
            color=TEXT_PRIMARY,
            fontsize=10,
            fontweight="bold",
        )

        activity_count = len(self.activity_log)
        activity_top = 0.34
        activity_bottom = 0.03

        for index, line in enumerate(self.activity_log):
            y_position = activity_top - index * (
                (activity_top - activity_bottom)
                / max(activity_count - 1, 1)
            )
            panel.text(
                0.02,
                y_position,
                line,
                transform=panel.transAxes,
                color="#B8C0C8",
                fontsize=6.5,
                fontfamily="monospace",
            )


def wait_for_next_cycle(
    seconds: int,
    viewer: Viewer | None,
) -> bool:
    """Wait while keeping the viewer's event loop live."""
    import time

    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        if viewer is not None:
            if not viewer.is_open():
                return False
            viewer.pump()
        else:
            time.sleep(0.1)

    return True