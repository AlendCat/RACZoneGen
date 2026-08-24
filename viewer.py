"""
Live site map and event panel.

A responsive interactive matplotlib window:

* autonomous lane network drawn as a *cached* PatchCollection so the
  repaint per cycle is cheap even with ~4800 lanes / ~74k points
* RAC events (only those reported since program start) scattered by
  severity, with a colour scale legend on the *left* of the map
* existing / newly placed square zones and their dashed suppression areas
* qualifying clusters (dotted hub radius, X marks the centroid)
* a compact, data-entry style side table listing the RAC events
  (time / X / Y / level / speed)

The map view is frozen to a fixed extent once the site footprint is
known, so it keeps the same size and shape on every cycle.  The GUI
event loop is pumped inside every blocking wait (`pump`) so the window
never freezes while a MineStar command or a DB query is running.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon, Rectangle

from clustering import Cluster
from minestar import ExistingZone, cluster_is_suppressed

# `adjustable="datalim"` keeps the map box a fixed shape (equal units,
# zooming never stretches it).  Matplotlib logs a "Ignoring fixed ...
# limits" message whenever it re-fits the limits to keep that aspect —
# expected here, so matplotlib logging is dropped to WARNING+ to keep
# the console clean.
logging.getLogger("matplotlib").setLevel(logging.ERROR)

PALETE_BG = "#121418"
PANEL_BG = "#0d0f12"
TEXT_PRIMARY = "#EAEAEA"
TEXT_GREY = "#9AA0A6"


class Viewer:
    """Live-updating site map with an event side table."""

    def __init__(self) -> None:
        plt.ion()

        self.figure = plt.figure(
            figsize=(15, 8.5),
            facecolor=PALETE_BG,
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "RACZoneGen — RAC Automatic Zone Monitor"
            )
        except AttributeError:
            pass

        # Manual, aspect-driven layout (no constrained layout, so the
        # geometry is fixed once and never re-fitted).  Placeholder
        # positions fill in at startup; `_freeze_layout` re-sizes the
        # map from the site's data ratio once the site bounds are known,
        # keeping the severity bar on the map's left and the RAC table
        # hugging the map's right edge, all centred in the window —
        # the table never sits at the far right of the window.
        placeholder = {
            "map": [0.09, 0.09, 0.60, 0.82],
            "panel": [0.73, 0.09, 0.20, 0.82],
            "colorbar": [0.065, 0.16, 0.013, 0.68],
        }
        self.map_axis = self.figure.add_axes(placeholder["map"])
        self.panel_axis = self.figure.add_axes(placeholder["panel"])

        self.map_axis.set_facecolor(PALETE_BG)
        # `set_box_aspect` (applied once the site bounds are known) pins
        # the axes box to a fixed width:height ratio on every draw, so
        # the map keeps the same shape and unit scale cycle after cycle;
        # the data limits are only touched when a zoom strays >0.5% from
        # that ratio, which prevents the per-cycle zoom creep seen with
        # a plain `adjustable="datalim"`.
        self.map_axis.set_aspect("equal")
        self.map_axis.set_xticks([])
        self.map_axis.set_yticks([])
        for spine in self.map_axis.spines.values():
            spine.set_edgecolor(TEXT_GREY)
            spine.set_linewidth(1.5)

        self.event_log: deque[tuple[str, str, str, str, str]] = deque(
            maxlen=14
        )

        # Cached lane collection.
        self._lane_collection: PatchCollection | None = None
        self._lane_signature: object = None
        self._lane_extent_x: list[float] = []
        self._lane_extent_y: list[float] = []

        # Dynamic artists that get discarded on every repaint.
        self._dynamic: list = []

        # The map view is frozen once meaningful data exists, so the map
        # keeps the same size and shape from cycle to cycle.
        self._fixed_bounds: tuple[float, float, float, float] | None = None
        self._bounds_home: tuple[float, float, float, float] | None = None

        # Eager severity legend (left of the map) so layout never shifts.
        self._severity_colormap = LinearSegmentedColormap.from_list(
            "rac_severity",
            ["#E4C45C", "#E8873C", "#E23B3B"],
        )
        self._severity_mappable = ScalarMappable(
            norm=Normalize(1, 4),
            cmap=self._severity_colormap,
        )
        self._severity_mappable.set_array([])
        self._colorbar = self.figure.colorbar(
            self._severity_mappable,
            cax=self.figure.add_axes(placeholder["colorbar"]),
        )
        self._colorbar.ax.set_title("RAC", color=TEXT_GREY, fontsize=8)
        self._colorbar.ax.tick_params(
            colors=TEXT_GREY,
            labelsize=7,
            left=True,
            right=False,
            labelleft=True,
            labelright=False,
        )
        self._colorbar.set_label(
            "Level", color=TEXT_GREY, fontweight="bold"
        )

        self._legend = None

        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._closed = False

        plt.show(block=False)

        # Keep the window from stealing focus when it repaints while the
        # operator is working in another application.
        try:
            from matplotlib.backends.qt_compat import QtCore

            window = self.figure.canvas.manager.window
            window.setAttribute(
                QtCore.Qt.WA_ShowWithoutActivating,
                True,
            )
        except Exception:
            pass

    def _on_close(self, _event) -> None:
        self._closed = True

    def is_open(self) -> bool:
        if self._closed:
            return False
        return plt.fignum_exists(self.figure.number)

    def pump(self, seconds: float = 0.05) -> None:
        """Process GUI events so the window stays responsive.

        Uses a plain sleep + Qt-backed `flush_events` instead of
        `plt.pause`, which can re-raise/re-focus the window on every
        poll cycle and steal focus from other applications.
        """
        if self._closed or not self.is_open():
            return
        try:
            self.figure.canvas.flush_events()
            self.figure.canvas.draw_idle()
            time.sleep(seconds)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Panel data
    # ------------------------------------------------------------------

    def add_event(self, row: pd.Series) -> None:
        """Add a single RAC event row to the panel."""
        time_text = pd.Timestamp(row["Time"]).strftime("%H:%M:%S")
        speed = row.get("Speed")
        speed_text = f"{float(speed):.0f}" if pd.notna(speed) else "-"
        self.event_log.appendleft(
            (
                time_text,
                f"{row['X']:>8.1f}",
                f"{row['Y']:>8.1f}",
                f"{int(row['Level']):>2}",
                f"{speed_text:>3}",
            )
        )

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

        handles = []

        # RAC events -----------------------------------------------------
        if not events.empty:
            minimum_level = float(events["Level"].min())
            maximum_level = float(events["Level"].max())
            if minimum_level == maximum_level:
                maximum_level = minimum_level + 1.0

            collection = axis.scatter(
                events["X"],
                events["Y"],
                c=events["Level"],
                cmap=self._severity_colormap,
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

            self._colorbar.update_normal(collection)

        # Existing zones + suppression areas -------------------------------
        for zone in existing_zones:
            center = (zone.center_x, zone.center_y)
            half = max(zone.half_size, zone_size / 2.0)

            if zone.proposed:
                zone_patch = Rectangle(
                    (zone.center_x - half, zone.center_y - half),
                    2 * half,
                    2 * half,
                    facecolor="#1B873B",
                    edgecolor="#3DFF6F",
                    linestyle=(0, (5, 2)),
                    linewidth=1.8,
                    alpha=0.45,
                    zorder=4,
                )
                suppression_patch = Circle(
                    center,
                    suppression_radius,
                    fill=False,
                    edgecolor="#3DFF6F",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=3,
                )
                name_text = axis.text(
                    zone.center_x,
                    zone.center_y - half - 1.0,
                    f"{zone.name} (PROPOSED)",
                    color="#9DFFB5",
                    fontsize=6.5,
                    horizontalalignment="center",
                    zorder=6,
                )
            else:
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

        # Qualifying clusters ---------------------------------------------
        for cluster in clusters:
            blocked = cluster_is_suppressed(
                cluster, existing_zones, suppression_radius
            )

            color = "#6B7280" if blocked else "#35C9FF"

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
                    f", spd {speed_text}"
                ),
                color="white",
                fontsize=7,
                zorder=7,
            )
            axis.add_patch(hub)
            self._dynamic.extend([hub, mark, label])

        # Bounds: frozen once the site extent is known ----------------------
        if self._fixed_bounds is None:
            self._fixed_bounds = self._resolve_bounds(
                events,
                existing_zones,
                suppression_radius,
            )

        # Freeze the map view exactly once: size the layout from the site's
        # aspect ratio (table hugging the map) and pin the box so the
        # limits are never re-fitted and the map neither creeps nor
        # stretches while zooming.
        if self._fixed_bounds is not None and self._bounds_home is None:
            self._freeze_layout(self._fixed_bounds)

        axis.set_title(
            (
                f"RAC events: {len(events)}"
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
    # Map bounds (frozen)
    # ------------------------------------------------------------------

    def _freeze_layout(
        self, bounds: tuple[float, float, float, float]
    ) -> None:
        """
        Size the map from the site's data ratio and lay the severity bar
        and RAC table tight against its edges.

        The whole block is centred in the window, so the table hugs the
        map instead of sitting at the far right of the window.  Runs
        once, the first cycle that has a site extent.
        """
        min_x, max_x, min_y, max_y = bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        if span_x <= 0 or span_y <= 0:
            return

        # Desired map display ratio (width/height in pixels) for equal
        # unit scaling.  Converted to figure-fraction ratio using the
        # figure's own aspect so the axes box fills the allocated rect
        # exactly (otherwise matplotlib shrinks it and the table no
        # longer hugs the map).
        fig_w, fig_h = self.figure.get_size_inches()
        fig_aspect = fig_w / fig_h
        pixel_ratio = span_x / span_y
        fraction_ratio = pixel_ratio / fig_aspect

        left_gap = 0.05
        right_gap = 0.05
        bottom_gap = 0.07
        top_gap = 0.06
        colorbar_w = 0.018
        colorbar_gap = 0.011
        table_w = 0.16
        table_gap = 0.012
        colorbar_height = 0.78

        avail_w = 1.0 - left_gap - right_gap
        avail_h = 1.0 - bottom_gap - top_gap
        usable_w = avail_w - (
            colorbar_w + colorbar_gap + table_gap + table_w
        )

        if avail_h * fraction_ratio <= usable_w:
            map_h = avail_h
            map_w = map_h * fraction_ratio
            block_h = map_h
            block_w = colorbar_w + colorbar_gap + map_w + table_gap + table_w
            block_x = left_gap + (avail_w - block_w) / 2.0
            block_y = bottom_gap
        else:
            map_w = usable_w
            map_h = map_w / fraction_ratio
            block_h = map_h
            block_w = avail_w
            block_x = left_gap
            block_y = bottom_gap + (avail_h - block_h) / 2.0

        colorbar_x = block_x
        colorbar_side_margin = block_h * (1.0 - colorbar_height) / 2.0
        colorbar_y = block_y + colorbar_side_margin
        colorbar_h = block_h * colorbar_height

        map_x = colorbar_x + colorbar_w + colorbar_gap
        map_y = block_y

        table_x = map_x + map_w + table_gap
        table_w = block_w - (table_x - block_x)

        self._colorbar.ax.set_position(
            [colorbar_x, colorbar_y, colorbar_w, colorbar_h]
        )
        self.map_axis.set_position([map_x, map_y, map_w, map_h])
        self.panel_axis.set_position([table_x, block_y, table_w, map_h])

        self.map_axis.set_box_aspect(span_y / span_x)
        self._bounds_home = bounds
        self._set_view(bounds)

    def _set_view(self, bounds) -> None:
        """Apply the frozen home limits (the box is pinned to its ratio)."""
        min_x, max_x, min_y, max_y = bounds
        self.map_axis.set_xlim(min_x, max_x)
        self.map_axis.set_ylim(min_y, max_y)

    def _resolve_bounds(
        self,
        events: pd.DataFrame,
        existing_zones: list[ExistingZone],
        suppression_radius: float,
    ) -> tuple[float, float, float, float] | None:
        """
        Compute a fixed map extent (min_x, max_x, min_y, max_y).

        The lane network defines the site footprint; the suppression
        circles around the zones extend beyond it, so both are included.
        Falls back to events/zones if lanes are not yet loaded.
        """
        x_values = list(self._lane_extent_x)
        y_values = list(self._lane_extent_y)

        for zone in existing_zones:
            x_values.extend(
                [
                    zone.center_x - suppression_radius,
                    zone.center_x + suppression_radius,
                ]
            )
            y_values.extend(
                [
                    zone.center_y - suppression_radius,
                    zone.center_y + suppression_radius,
                ]
            )

        if not x_values and not events.empty:
            x_values.extend(events["X"].astype(float).tolist())
            y_values.extend(events["Y"].astype(float).tolist())

        if not x_values:
            return None

        pad = max(
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
        ) * 0.06
        pad = max(pad, 50.0)

        return (
            min(x_values) - pad,
            max(x_values) + pad,
            min(y_values) - pad,
            max(y_values) + pad,
        )

    # ------------------------------------------------------------------
    # Side panel
    # ------------------------------------------------------------------

    def _draw_panel(self) -> None:
        """
        Draw the RAC event list as a compact data-entry style table.

        Rows have a fixed height (no stretching when the list is short),
        columns are evenly spaced, and the newest event sits on top.
        """
        panel = self.panel_axis
        panel.clear()
        panel.set_facecolor(PANEL_BG)
        panel.axis("off")
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)

        top = 0.98
        bottom_margin = 0.02
        header_height = 0.06
        row_height = 0.05
        left = 0.035
        right = 0.965
        fontsize = 9.5

        columns = ["TIME", "X", "Y", "Lv", "km/h"]
        step = (right - left) / len(columns)
        column_x = [
            left + step * (index + 0.5) for index in range(len(columns))
        ]

        panel.add_patch(
            Rectangle(
                (0.0, top - header_height),
                1.0,
                header_height,
                facecolor="#1C2A24",
                edgecolor="#35C759",
                linewidth=1.0,
                transform=panel.transAxes,
                zorder=1,
            )
        )

        for header_text, x in zip(columns, column_x):
            panel.text(
                x,
                top - header_height / 2.0,
                header_text,
                va="center",
                ha="left",
                transform=panel.transAxes,
                color="#8CFFA8",
                fontsize=fontsize,
                fontfamily="monospace",
                fontweight="bold",
                zorder=2,
            )

        rows = list(self.event_log)

        if not rows:
            panel.text(
                (left + right) / 2.0,
                (top + bottom_margin) / 2.0,
                "No RAC events since start",
                va="center",
                ha="center",
                transform=panel.transAxes,
                color=TEXT_GREY,
                fontsize=10,
            )
            return

        max_rows = int(
            (top - header_height - bottom_margin) // row_height
        )

        for index, values in enumerate(rows[:max_rows]):
            row_top = top - header_height - index * row_height

            if index % 2 == 0:
                panel.add_patch(
                    Rectangle(
                        (0.0, row_top - row_height),
                        1.0,
                        row_height,
                        facecolor="#101418",
                        edgecolor="none",
                        transform=panel.transAxes,
                        zorder=0,
                    )
                )

            for text, x in zip(values, column_x):
                panel.text(
                    x,
                    row_top - row_height / 2.0,
                    text,
                    va="center",
                    ha="left",
                    transform=panel.transAxes,
                    color=TEXT_PRIMARY,
                    fontsize=fontsize,
                    fontfamily="monospace",
                    zorder=2,
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