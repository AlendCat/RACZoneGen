"""Live site map and event panel.

Fast, fixed-geometry matplotlib window:

* fixed axes fractions — the map box and side table never move, whatever
  the window size or data volume.  The map keeps equal units
  (``aspect='equal', adjustable='datalim'``) so the site never stretches;
  resizing only scales the whole figure, it never re-fits the layout.
* autonomous lanes drawn as ONE cached PatchCollection (rebuilt only when
  the lane data actually changes, decimated for draw speed)
* RAC events scattered by severity, colour legend on the map's left
* square zones + dashed suppression rings, cluster X marks
* side table with fixed row height listing RAC events since start
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon, Rectangle

from clustering import Cluster
from minestar import ExistingZone

try:
    matplotlib.rcParams["path.simplify"] = True
    matplotlib.rcParams["path.simplify_threshold"] = 0.5
    matplotlib.rcParams["agg.path.chunksize"] = 20000
except Exception:
    pass

logging.getLogger("matplotlib").setLevel(logging.ERROR)

logger = logging.getLogger("rac_zone_monitor")

PALETE_BG = "#121418"
PANEL_BG = "#0d0f12"
TEXT_PRIMARY = "#EAEAEA"
TEXT_GREY = "#9AA0A6"

# Control-bar chip palette (matches the table header theme).
CHIP_ACTIVE_FACE = "#1C2A24"
CHIP_ACTIVE_EDGE = "#35C759"
CHIP_ACTIVE_TEXT = "#8CFFA8"
CHIP_OFF_FACE = "#14171B"
CHIP_OFF_EDGE = "#3A444E"
CHIP_OFF_TEXT = "#9AA0A6"
CHIP_WARN_EDGE = "#E23B3B"
CHIP_WARN_TEXT = "#FF9B9B"
CHIP_HINT_TEXT = "#6E7681"

# Fixed geometry — set once in __init__, never recomputed.  Fractions of
# the figure, so resize scales everything proportionally with no jumps.
LAYOUT = {
    "colorbar": [0.040, 0.150, 0.018, 0.700],
    "map": [0.075, 0.080, 0.545, 0.840],
    "panel": [0.650, 0.080, 0.320, 0.840],
    "controls": [0.650, 0.905, 0.320, 0.085],
}

ZONE_IMPORTED_FACE = "#C232A8"
ZONE_IMPORTED_EDGE = "#FF8AE2"
ZONE_PROPOSED_FACE = "#2F7BFF"
ZONE_PROPOSED_EDGE = "#9DC0FF"
SUPPRESSION_EDGE = "#5B8DEF"

# Max lane vertices drawn per edge before decimation kicks in.
LANE_STRIDE_2_AT = 40
LANE_STRIDE_3_AT = 120

# Minimum seconds between full redraws (update/pump/hover all respect it).
MIN_DRAW_INTERVAL = 0.20


def _compact_coord(value: float) -> str:
    av = abs(value)
    if av >= 1000:
        return f"{value / 1000:.1f}K"
    return f"{value:.1f}"


def _fmt_speed(value: object) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v) or v <= 0 or v > 200:
        return "-"
    return f"{v:.1f}"


class Viewer:
    """Live-updating site map with an event side table (fixed geometry)."""

    def __init__(self) -> None:
        plt.ion()

        self.figure = plt.figure(
            figsize=(13.5, 7.8),
            facecolor=PALETE_BG,
            dpi=100,
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "RACZoneGen — RAC Automatic Zone Monitor"
            )
        except AttributeError:
            pass

        self.map_axis = self.figure.add_axes(LAYOUT["map"])
        self.panel_axis = self.figure.add_axes(LAYOUT["panel"])
        self._controls_axis = self.figure.add_axes(LAYOUT["controls"])
        self._controls_axis.set_facecolor(PANEL_BG)
        self._controls_axis.set_xlim(0, 1)
        self._controls_axis.set_ylim(0, 1)
        self._controls_axis.axis("off")

        self.map_axis.set_facecolor(PALETE_BG)
        # Equal units, fixed box: extra data space pads the limits instead
        # of reshaping the axes — the table/map never shift on resize.
        self.map_axis.set_aspect("equal", adjustable="datalim")
        self.map_axis.set_xticks([])
        self.map_axis.set_yticks([])
        for spine in self.map_axis.spines.values():
            spine.set_edgecolor(TEXT_GREY)
            spine.set_linewidth(1.2)

        self.event_log: deque[tuple[str, str, str, str]] = deque(
            maxlen=100
        )
        self._event_coords: deque[tuple[float, float]] = deque(maxlen=100)
        self._panel_scroll: int = 0

        # ---- Runtime user controls (control bar above the table) ---------
        # creation_enabled: pause/resume zone creation. visible_levels:
        # display-only severity filter for the MAP and the table. Zone-
        # severity selection itself is config-only ([rac] cluster_levels).
        self.creation_enabled: bool = True
        # Display filter covers severities 1-5 (level 0 does not occur in
        # real RAC data; unparsable/0 levels always stay visible).
        self.visible_levels: set[int] = {1, 2, 3, 4, 5}
        self._chips: list[tuple[dict, Rectangle]] = []
        self._panel_rows: list = []
        self._hover_event_data: tuple | None = None
        self._last_frame: dict | None = None

        self._lane_collection: PatchCollection | None = None
        self._lane_signature: object = None
        self._lane_extent_x: list[float] = []
        self._lane_extent_y: list[float] = []

        # Pooled map artists (incremental redraw: updated in place, never
        # discarded/recreated per cycle).
        self._event_scatter = None
        self._zone_artists: list = []
        self._cluster_scatter = None
        self._cluster_labels: list = []
        self._frame_sig: object = None
        self._loading_text = self.map_axis.text(
            0.5,
            0.5,
            "Loading site…",
            transform=self.map_axis.transAxes,
            color=TEXT_GREY,
            fontsize=11,
            ha="center",
            va="center",
        )

        self._fixed_bounds: tuple[float, float, float, float] | None = None

        self._severity_colormap = LinearSegmentedColormap.from_list(
            "rac_severity",
            ["#E4C45C", "#E8873C", "#E23B3B"],
        )
        self._severity_mappable = ScalarMappable(
            norm=Normalize(1, 5),
            cmap=self._severity_colormap,
        )
        self._severity_mappable.set_array([])
        self._colorbar = self.figure.colorbar(
            self._severity_mappable,
            cax=self.figure.add_axes(LAYOUT["colorbar"]),
        )
        self._style_colorbar()

        self._last_draw: float = 0.0
        self._last_hover: float = 0.0
        self._pan_active = False
        self._pan_origin: tuple[float, float] | None = None
        self._pan_xlim: tuple[float, float] | None = None
        self._pan_ylim: tuple[float, float] | None = None
        self._ctrl_down = False
        self._hover_zones: list[ExistingZone] = []
        self._zone_annotation = self.map_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.35", fc="#1F2937", ec="#93C5FD", alpha=0.92),
            color="#DBEAFE",
            fontsize=7,
            ha="left",
            va="bottom",
            visible=False,
            zorder=12,
        )

        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.figure.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.figure.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._closed = False

        # Draw the (empty) table immediately so the right panel is never a
        # white flash on startup.
        self._draw_controls()
        self._draw_panel()
        self.figure.canvas.draw_idle()

        plt.show(block=False)

        try:
            from matplotlib.backends.qt_compat import QtCore

            window = self.figure.canvas.manager.window
            window.setAttribute(
                QtCore.Qt.WA_ShowWithoutActivating,
                True,
            )
        except Exception:
            pass

    # -- window events ----------------------------------------------------

    def _on_close(self, _event) -> None:
        self._closed = True

    def _on_key_press(self, event) -> None:
        key = getattr(event, "key", None)
        if key is not None and "control" in key.lower():
            self._ctrl_down = True

    def _on_key_release(self, event) -> None:
        key = getattr(event, "key", None)
        if key is not None and "control" in key.lower():
            self._ctrl_down = False
        elif key is None:
            self._ctrl_down = False

    def _request_draw(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_draw) < MIN_DRAW_INTERVAL:
            return
        self._last_draw = now
        try:
            self.figure.canvas.draw_idle()
        except Exception:
            pass

    def _zoom_to_point(self, x: float, y: float, half_window: float = 400.0) -> None:
        try:
            self.map_axis.set_xlim(x - half_window, x + half_window)
            self.map_axis.set_ylim(y - half_window, y + half_window)
            self._request_draw(force=True)
        except Exception:
            pass

    def _on_button_press(self, event) -> None:
        if getattr(event, "button", None) != 1:
            return
        if getattr(event, "inaxes", None) is self._controls_axis:
            self._on_controls_click(event)
            return
        in_panel = getattr(event, "inaxes", None) is self.panel_axis or self._point_in_panel(event)
        if in_panel:
            try:
                y = getattr(event, "ydata", None)
                if y is None:
                    return
                top = 0.98
                header_height = 0.06
                bottom_margin = 0.02
                row_height = 0.05
                if y > top - header_height or y < bottom_margin:
                    return
                visible_index = int((top - header_height - y) // row_height)
                if visible_index < 0:
                    return
                pairs = self._panel_rows
                total = len(pairs)
                if total == 0:
                    return
                log_index = self._panel_scroll + visible_index
                if log_index < 0 or log_index >= total:
                    return
                rx, ry = pairs[log_index][1]
                self._zoom_to_point(rx, ry)
            except Exception:
                pass
            return
        if event.inaxes is not self.map_axis:
            return
        key = getattr(event, "key", None)
        is_ctrl = self._ctrl_down or (key is not None and "control" in key.lower())
        if not is_ctrl:
            return
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return
        self._pan_active = True
        self._pan_origin = (event.x, event.y)
        self._pan_xlim = self.map_axis.get_xlim()
        self._pan_ylim = self.map_axis.get_ylim()

    def _on_mouse_move(self, event) -> None:
        if self._pan_active:
            if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
                return
            if self._pan_origin is None or self._pan_xlim is None or self._pan_ylim is None:
                return
            bbox = self.map_axis.bbox
            if bbox.width == 0 or bbox.height == 0:
                return
            dx_disp = event.x - self._pan_origin[0]
            dy_disp = event.y - self._pan_origin[1]
            x0, x1 = self._pan_xlim
            y0, y1 = self._pan_ylim
            x_range = x1 - x0
            scale = x_range / bbox.width
            dx_data = dx_disp * scale
            dy_data = dy_disp * scale
            self.map_axis.set_xlim(x0 - dx_data, x1 - dx_data)
            self.map_axis.set_ylim(y0 - dy_data, y1 - dy_data)
            self._request_draw()
            return

        # Hover tooltip, debounced so moving the mouse never hammers redraw.
        now = time.monotonic()
        if (now - self._last_hover) < 0.08:
            return
        self._last_hover = now
        if getattr(event, "inaxes", None) is not self.map_axis:
            if self._zone_annotation.get_visible():
                self._zone_annotation.set_visible(False)
                self._request_draw(force=True)
            return
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            if self._zone_annotation.get_visible():
                self._zone_annotation.set_visible(False)
                self._request_draw(force=True)
            return

        # Event dots first: unit · speed · severity tooltip.
        hit = self._event_dot_hit(event)
        if hit is not None:
            machine, speed, level, hit_x, hit_y = hit
            if np.isfinite(speed) and speed > 0:
                speed_text = f"{speed:.0f} km/h"
            else:
                speed_text = "speed ?"
            self._zone_annotation.xy = (hit_x, hit_y)
            self._zone_annotation.set_text(
                f"{machine} · {speed_text} · Lv {level}"
            )
            if not self._zone_annotation.get_visible():
                self._zone_annotation.set_visible(True)
                self._request_draw(force=True)
            return

        for zone in self._hover_zones:
            half = max(zone.half_size, 50.0)
            if abs(xdata - zone.center_x) <= half and abs(ydata - zone.center_y) <= half:
                self._zone_annotation.xy = (zone.center_x, zone.center_y)
                self._zone_annotation.set_text(zone.name)
                if not self._zone_annotation.get_visible():
                    self._zone_annotation.set_visible(True)
                    self._request_draw(force=True)
                return
        if self._zone_annotation.get_visible():
            self._zone_annotation.set_visible(False)
            self._request_draw(force=True)

    def _on_button_release(self, event) -> None:
        if getattr(event, "button", None) != 1:
            return
        self._pan_active = False
        self._pan_origin = None
        self._pan_xlim = None
        self._pan_ylim = None

    def _event_dot_hit(self, event) -> tuple | None:
        """Nearest RAC-event dot within ~10 px, else None."""
        data = self._hover_event_data
        if not data:
            return None
        ex, ey, el, machines, speeds = data
        if len(ex) == 0:
            return None
        try:
            display_points = self.map_axis.transData.transform(
                np.column_stack([ex, ey])
            )
        except Exception:
            return None
        cursor = np.array(
            [getattr(event, "x", None), getattr(event, "y", None)],
            dtype=float,
        )
        if not np.all(np.isfinite(cursor)):
            return None
        deltas = display_points - cursor
        squared = (deltas * deltas).sum(axis=1)
        index = int(np.argmin(squared))
        if squared[index] > 100.0:  # ~10 px radius
            return None
        return (
            str(machines[index]),
            float(speeds[index]),
            int(np.clip(round(float(el[index])), 0, 9)),
            float(ex[index]),
            float(ey[index]),
        )

    def _point_in_panel(self, event) -> bool:
        try:
            bbox = self.panel_axis.bbox
            x, y = getattr(event, "x", None), getattr(event, "y", None)
            if x is None or y is None:
                return False
            return bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Control bar (pause · min severity · table level filter)
    # ------------------------------------------------------------------

    def _draw_controls(self) -> None:
        """Render the themed control strip above the side table."""
        axis = self._controls_axis
        axis.clear()
        axis.set_facecolor(PANEL_BG)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        self._chips = []

        def add_chip(x, y, w, h, text, active, kind=None, value=None,
                     warn=False):
            if active:
                face = CHIP_ACTIVE_FACE
                edge = CHIP_WARN_EDGE if warn else CHIP_ACTIVE_EDGE
                text_color = CHIP_WARN_TEXT if warn else CHIP_ACTIVE_TEXT
            else:
                face, edge, text_color = (
                    CHIP_OFF_FACE, CHIP_OFF_EDGE, CHIP_OFF_TEXT
                )
            rect = Rectangle(
                (x, y),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
                transform=axis.transAxes,
                zorder=2,
                clip_on=False,
            )
            axis.add_patch(rect)
            axis.text(
                x + w / 2.0,
                y + h / 2.0,
                text,
                va="center",
                ha="center",
                color=text_color,
                fontsize=7.0,
                fontfamily="monospace",
                fontweight="bold" if active else "normal",
                transform=axis.transAxes,
                zorder=3,
            )
            if kind is not None:
                self._chips.append(
                    ({"kind": kind, "value": value}, rect)
                )
            return x + w + 0.015

        row1_y, row1_h = 0.55, 0.40
        row2_y, row2_h = 0.05, 0.40

        # Row 1: zone-creation toggle, anchored right ----------------------
        create_w = 0.27
        add_chip(
            0.98 - 0.015 - create_w, row1_y, create_w, row1_h,
            "CREATE ZONES: OFF" if not self.creation_enabled
            else "CREATE ZONES: ON",
            kind="create",
            active=True,
            warn=not self.creation_enabled,
        )

        # Row 2: severity display filter (map + table) ---------------------
        axis.text(
            0.02,
            row2_y + row2_h / 2.0,
            "Filter by Severity:",
            va="center", ha="left", color="#FFFFFF",
            fontsize=9.0, fontfamily="monospace",
            transform=axis.transAxes, zorder=3,
        )
        # Chips anchored from the right edge so the label can never run
        # into them at smaller window sizes.
        chip_w, chip_gap = 0.085, 0.015
        edge = 0.98
        for level in (5, 4, 3, 2, 1):
            chip_x = edge - chip_gap - chip_w
            add_chip(
                chip_x, row2_y, chip_w, row2_h,
                str(level),
                kind="level",
                value=level,
                active=(level in self.visible_levels),
            )
            edge = chip_x

    def _on_controls_click(self, event) -> None:
        """Handle clicks on the control-bar chips (axis data coords 0-1)."""
        x, y = getattr(event, "xdata", None), getattr(event, "ydata", None)
        if x is None or y is None:
            return
        for meta, rect in self._chips:
            bx, by = rect.get_xy()
            bw, bh = rect.get_width(), rect.get_height()
            if not (bx <= x <= bx + bw and by <= y <= by + bh):
                continue
            kind = meta.get("kind")
            value = meta.get("value")
            if kind == "create":
                self.creation_enabled = not self.creation_enabled
                logger.info(
                    "User control: zone creation -> %s.",
                    "ENABLED" if self.creation_enabled else "PAUSED",
                )
            elif kind == "minsev":
                value = int(value)
                self.min_severity = (
                    0 if self.min_severity == value else value
                )
                logger.info(
                    "User control: minimum zone severity -> %s.",
                    f">= {self.min_severity}" if self.min_severity else "OFF",
                )
            elif kind == "level":
                value = int(value)
                if value in self.visible_levels:
                    if len(self.visible_levels) == 1:
                        # Never blank the map/table on the user.
                        self.visible_levels = {1, 2, 3, 4, 5}
                    else:
                        self.visible_levels.discard(value)
                else:
                    self.visible_levels.add(value)
                logger.info(
                    "User control: severity display filter -> %s.",
                    ", ".join(
                        str(level) for level in sorted(self.visible_levels)
                    ),
                )
            # Force a full map/panel resync and apply it right now so the
            # dots respond instantly, not on the next poll.
            self._frame_sig = None
            if self._last_frame is not None:
                try:
                    self.update(**self._last_frame)
                except Exception:
                    logger.exception(
                        "Map resync after filter change failed."
                    )
            self._draw_controls()
            self._draw_panel()
            self._request_draw(force=True)
            return

    def _on_scroll(self, event) -> None:
        in_panel = getattr(event, "inaxes", None) is self.panel_axis or self._point_in_panel(event)
        if in_panel:
            top = 0.98
            header_height = 0.06
            bottom_margin = 0.02
            row_height = 0.05
            max_rows = int((top - header_height - bottom_margin) // row_height)
            total = len(self._panel_rows) or len(self.event_log)
            max_offset = max(0, total - max_rows)
            if max_offset == 0:
                return
            step = 3
            try:
                if getattr(event, "step", None) is not None:
                    step = max(1, int(abs(float(event.step))) * 3)
            except Exception:
                pass
            btn = getattr(event, "button", None)
            if btn == "up" or btn == 4:
                self._panel_scroll = max(0, self._panel_scroll - step)
            elif btn == "down" or btn == 5:
                self._panel_scroll = min(max_offset, self._panel_scroll + step)
            else:
                step_val = getattr(event, "step", None)
                if step_val is not None:
                    try:
                        s = float(step_val)
                        if s > 0:
                            self._panel_scroll = max(0, self._panel_scroll - step)
                        elif s < 0:
                            self._panel_scroll = min(max_offset, self._panel_scroll + step)
                        else:
                            return
                    except Exception:
                        return
                else:
                    return
            self._draw_panel()
            self._request_draw(force=True)
            try:
                self.figure.canvas.flush_events()
            except Exception:
                pass
            return
        if getattr(event, "inaxes", None) is not self.map_axis:
            return
        key = getattr(event, "key", None)
        is_ctrl = self._ctrl_down or (key is not None and "control" in key.lower())
        if not is_ctrl:
            return
        if getattr(event, "xdata", None) is None or getattr(event, "ydata", None) is None:
            return
        factor = 0.85 if event.button == "up" else 1.15
        if event.step is not None:
            try:
                step = float(event.step)
                factor = 0.85 ** step if step > 0 else 1.15 ** (-step)
            except Exception:
                pass
        x0, x1 = self.map_axis.get_xlim()
        y0, y1 = self.map_axis.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_x0 = xdata - (xdata - x0) * factor
        new_x1 = xdata + (x1 - xdata) * factor
        new_y0 = ydata - (ydata - y0) * factor
        new_y1 = ydata + (y1 - ydata) * factor
        self.map_axis.set_xlim(new_x0, new_x1)
        self.map_axis.set_ylim(new_y0, new_y1)
        self._request_draw()

    def is_open(self) -> bool:
        if self._closed:
            return False
        return plt.fignum_exists(self.figure.number)

    def pump(self, seconds: float = 0.05) -> None:
        """Process GUI events so the window stays responsive."""
        if self._closed or not self.is_open():
            return
        try:
            self.figure.canvas.flush_events()
            time.sleep(seconds)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Panel data
    # ------------------------------------------------------------------

    def add_event(self, row: pd.Series) -> None:
        """Add a single RAC event row to the panel (newest at top)."""
        time_text = pd.Timestamp(row["Time"]).strftime("%H:%M:%S")
        speed_text = _fmt_speed(row.get("Speed"))
        unit = row.get("Machine")
        unit_text = str(unit).strip() if pd.notna(unit) else ""
        unit_text = unit_text if unit_text else "-"
        raw_x = float(row["X"])
        raw_y = float(row["Y"])
        was_at_top = self._panel_scroll == 0
        before = len(self.event_log)
        self.event_log.appendleft(
            (
                time_text,
                unit_text,
                f"{int(row['Level'])}",
                speed_text,
            )
        )
        self._event_coords.appendleft((raw_x, raw_y))
        after = len(self.event_log)
        grew = 1 if after > before else 0
        if grew == 0:
            grew = 1
        if was_at_top:
            self._panel_scroll = 0
        else:
            max_rows_est = 18
            total = len(self.event_log)
            max_offset = max(0, total - max_rows_est)
            self._panel_scroll = min(self._panel_scroll + grew, max_offset)

    # ------------------------------------------------------------------
    # Lanes (cached, decimated)
    # ------------------------------------------------------------------

    @staticmethod
    def _lanes_signature(lanes: pd.DataFrame) -> object:
        if lanes.empty:
            return 0
        try:
            x_sum = round(float(pd.to_numeric(lanes["X"], errors="coerce").sum()), 1)
        except Exception:
            x_sum = 0.0
        return (len(lanes), int(lanes["LANE_OID"].nunique()), x_sum)

    def _rebuild_lanes(self, lanes: pd.DataFrame) -> None:
        signature = self._lanes_signature(lanes)

        if signature == self._lane_signature:
            return

        if self._lane_collection is not None:
            try:
                self._lane_collection.remove()
            except (ValueError, AttributeError):
                pass
            self._lane_collection = None

        self._lane_extent_x = []
        self._lane_extent_y = []

        if not lanes.empty:
            work = lanes[["LANE_OID", "Segment", "POINT_NR", "X", "Y"]].copy()
            work["X"] = pd.to_numeric(work["X"], errors="coerce")
            work["Y"] = pd.to_numeric(work["Y"], errors="coerce")
            work["POINT_NR"] = pd.to_numeric(work["POINT_NR"], errors="coerce")
            work = work.dropna(subset=["X", "Y", "POINT_NR"])
            seg = work["Segment"].astype(str).str.lower().to_numpy()
            left_mask = seg == "left"
            right_mask = seg == "right"
            left = work[left_mask].sort_values(["LANE_OID", "POINT_NR"])
            right = work[right_mask].sort_values(["LANE_OID", "POINT_NR"])

            patches: list[Polygon] = []
            xs_all: list[float] = []
            ys_all: list[float] = []

            for oid, lg in left.groupby("LANE_OID", sort=False):
                try:
                    rg = right[right["LANE_OID"] == oid]
                except Exception:
                    continue
                if rg.empty:
                    continue
                lx = lg["X"].to_numpy(dtype=float)
                ly = lg["Y"].to_numpy(dtype=float)
                rx = rg["X"].to_numpy(dtype=float)[::-1]
                ry = rg["Y"].to_numpy(dtype=float)[::-1]
                # Decimate long edges: halves/thirds the vertex count.
                if len(lx) > LANE_STRIDE_3_AT:
                    lx = lx[::3]
                    ly = ly[::3]
                elif len(lx) > LANE_STRIDE_2_AT:
                    lx = lx[::2]
                    ly = ly[::2]
                if len(rx) > LANE_STRIDE_3_AT:
                    rx = rx[::3]
                    ry = ry[::3]
                elif len(rx) > LANE_STRIDE_2_AT:
                    rx = rx[::2]
                    ry = ry[::2]
                if len(lx) < 2 or len(rx) < 2:
                    continue
                x_coords = np.concatenate([lx, rx])
                y_coords = np.concatenate([ly, ry])
                patches.append(
                    Polygon(
                        np.column_stack([x_coords, y_coords]),
                        closed=True,
                    )
                )
                xs_all.extend([float(x_coords.min()), float(x_coords.max())])
                ys_all.extend([float(y_coords.min()), float(y_coords.max())])

            if patches:
                self._lane_collection = PatchCollection(
                    patches,
                    facecolor="#1F3A22",
                    edgecolor="#35C759",
                    linewidths=0.35,
                    alpha=0.55,
                )
                self.map_axis.add_collection(self._lane_collection)
                self._lane_extent_x = xs_all
                self._lane_extent_y = ys_all
                # New footprint invalidates the frozen view once.
                if self._fixed_bounds is not None:
                    self._fixed_bounds = None

        self._lane_signature = signature

    # ------------------------------------------------------------------

    def _style_colorbar(self) -> None:
        """Pin the severity legend: ticks + numbers on the LEFT of the bar.

        Re-applied after every ``update_normal`` because resyncing the
        mappable can reset tick placement back to the right side.
        """
        try:
            self._colorbar.set_ticks([1, 2, 3, 4, 5])
            self._colorbar.set_ticklabels(["1", "2", "3", "4", "5"])
            self._colorbar.ax.set_title(
                "RAC", color=TEXT_GREY, fontsize=8, loc="left", pad=6
            )
            self._colorbar.ax.yaxis.set_label_position("left")
            self._colorbar.ax.yaxis.set_ticks_position("left")
            self._colorbar.ax.tick_params(
                colors=TEXT_GREY,
                labelsize=7,
                left=True,
                right=False,
                labelleft=True,
                labelright=False,
                pad=2,
            )
            self._colorbar.set_label(
                "Level", color=TEXT_GREY, fontweight="bold", labelpad=8
            )
        except Exception:
            pass

    def _sync_zones(
        self,
        existing_zones: list[ExistingZone],
        suppression_radius: float,
        zone_size: float,
    ) -> None:
        """Update the pooled zone + suppression-ring + tag artists."""
        axis = self.map_axis
        pool = self._zone_artists
        for i, zone in enumerate(existing_zones):
            half = max(zone.half_size, zone_size / 2.0)
            if zone.proposed:
                face, edge = ZONE_PROPOSED_FACE, ZONE_PROPOSED_EDGE
            else:
                face, edge = ZONE_IMPORTED_FACE, ZONE_IMPORTED_EDGE
            if i < len(pool):
                rect, circ, tag = pool[i]
                rect.set_xy((zone.center_x - half, zone.center_y - half))
                rect.set_width(2 * half)
                rect.set_height(2 * half)
                rect.set_facecolor(face)
                rect.set_edgecolor(edge)
                rect.set_visible(True)
                circ.center = (zone.center_x, zone.center_y)
                circ.set_radius(suppression_radius)
                circ.set_visible(True)
            else:
                rect = Rectangle(
                    (zone.center_x - half, zone.center_y - half),
                    2 * half,
                    2 * half,
                    facecolor=face,
                    edgecolor=edge,
                    linestyle="-",
                    linewidth=1.6,
                    alpha=0.80,
                    zorder=2,
                )
                circ = Circle(
                    (zone.center_x, zone.center_y),
                    suppression_radius,
                    fill=False,
                    edgecolor=SUPPRESSION_EDGE,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=2,
                )
                tag = axis.text(
                    zone.center_x,
                    zone.center_y + half + 12.0,
                    "PROPOSED",
                    color=ZONE_PROPOSED_EDGE,
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    zorder=7,
                )
                axis.add_patch(rect)
                axis.add_patch(circ)
                pool.append((rect, circ, tag))
            tag.set_position((zone.center_x, zone.center_y + half + 12.0))
            tag.set_visible(bool(zone.proposed))
        for j in range(len(existing_zones), len(pool)):
            for artist in pool[j]:
                try:
                    artist.set_visible(False)
                except (ValueError, AttributeError):
                    pass

    def _sync_events(self, x, y, lv) -> None:
        """Update the pooled RAC-event scatter collection in place."""
        axis = self.map_axis
        if len(x) == 0:
            if self._event_scatter is not None:
                try:
                    self._event_scatter.remove()
                except (ValueError, AttributeError):
                    pass
                self._event_scatter = None
            return
        if self._event_scatter is None:
            self._event_scatter = axis.scatter(
                x,
                y,
                c=lv,
                cmap=self._severity_colormap,
                norm=Normalize(1, 5),
                s=28,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.95,
                zorder=5,
            )
        else:
            try:
                self._event_scatter.set_offsets(np.column_stack([x, y]))
            except Exception:
                try:
                    self._event_scatter.remove()
                except (ValueError, AttributeError):
                    pass
                self._event_scatter = None
                self._sync_events(x, y, lv)
                return
            try:
                self._event_scatter.set_array(np.asarray(lv, dtype=float))
            except Exception:
                pass
        try:
            self._colorbar.update_normal(self._event_scatter)
        except Exception:
            pass
        self._style_colorbar()

    def _sync_clusters(self, clusters: list[Cluster]) -> None:
        """Redraw cluster X marks (one pooled scatter) + pooled labels."""
        axis = self.map_axis
        if self._cluster_scatter is not None:
            try:
                self._cluster_scatter.remove()
            except (ValueError, AttributeError):
                pass
            self._cluster_scatter = None
        points: list[tuple[float, float]] = []
        for cluster in clusters:
            avg_level = cluster.score / max(cluster.event_count, 1)
            lvl = max(1.0, min(5.0, float(avg_level)))
            x_rgba = self._severity_colormap(
                self._severity_mappable.norm(lvl)
            )
            anchor = getattr(cluster, "anchor_zone", None)
            if anchor is not None:
                draw_x, draw_y = anchor.center_x, anchor.center_y
            else:
                draw_x, draw_y = cluster.center_x, cluster.center_y
            points.append((draw_x, draw_y, x_rgba, cluster))
        if points:
            self._cluster_scatter = axis.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="x",
                c=[p[2] for p in points],
                s=120,
                linewidths=2.2,
                zorder=6,
            )
        for i, (draw_x, draw_y, _rgba, cluster) in enumerate(points):
            avg_spd = getattr(cluster, "average_speed_kmh", None)
            speed_text = (
                f"{float(avg_spd):.0f}"
                if avg_spd and np.isfinite(float(avg_spd)) else "-"
            )
            text = f" {cluster.event_count} evt, spd {speed_text}"
            if i < len(self._cluster_labels):
                label = self._cluster_labels[i]
                label.set_position((draw_x, draw_y))
                label.set_text(text)
                label.set_visible(True)
            else:
                label = axis.text(
                    draw_x,
                    draw_y,
                    text,
                    color="white",
                    fontsize=7,
                    zorder=7,
                )
                self._cluster_labels.append(label)
        for j in range(len(points), len(self._cluster_labels)):
            try:
                self._cluster_labels[j].set_visible(False)
            except (ValueError, AttributeError):
                pass

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
        """Incremental redraw: pooled artists, cached lanes, skip identical frames."""
        if not self.is_open():
            return

        axis = self.map_axis
        # Keep the args so chip clicks can re-sync the map immediately
        # without waiting for the next poll cycle.
        self._last_frame = dict(
            events=events,
            lanes=lanes,
            clusters=clusters,
            existing_zones=existing_zones,
            suppression_radius=suppression_radius,
            zone_size=zone_size,
            grouping_radius=grouping_radius,
        )
        self._rebuild_lanes(lanes)

        # Event vectors (shared by the scatter pool + frame signature).
        if not events.empty:
            try:
                ex = pd.to_numeric(events["X"], errors="coerce").to_numpy(dtype=float)
                ey = pd.to_numeric(events["Y"], errors="coerce").to_numpy(dtype=float)
                el = pd.to_numeric(events["Level"], errors="coerce").to_numpy(dtype=float)
            except Exception:
                ex = np.asarray(events["X"].to_numpy(), dtype=float)
                ey = np.asarray(events["Y"].to_numpy(), dtype=float)
                el = np.asarray(events["Level"].to_numpy(), dtype=float)
            finite = np.isfinite(ex) & np.isfinite(ey)
        else:
            ex = ey = el = np.empty(0, dtype=float)
            finite = np.empty(0, dtype=bool)
        # Severity DISPLAY filter: the shortfall chips hide dots on the map
        # exactly like they hide table rows. Unparsable levels stay visible.
        # (Zone-cluster severity is config [rac] cluster_levels — separate.)
        if self.visible_levels and len(el):
            level_int = np.rint(
                np.nan_to_num(el, nan=-9999.0)
            ).astype(int)
            in_set = np.isin(
                level_int, np.fromiter(self.visible_levels, dtype=int)
            )
            # Level 0 / garbage levels: treat like unparsable, keep shown.
            visible = (level_int < 1) | in_set
        else:
            visible = np.ones(len(ex), dtype=bool)
        keep = finite & visible
        ex, ey, el = ex[keep], ey[keep], el[keep]

        # Hover payload for map dots: unit · speed · severity (aligned to
        # the coordinate arrays the scatter actually draws).
        try:
            if "Machine" in events.columns:
                machines = (
                    events["Machine"]
                    .fillna("-")
                    .astype(str)
                    .to_numpy(dtype=object)
                )
            else:
                machines = np.full(len(events), "-", dtype=object)
        except Exception:
            machines = np.full(len(events), "-", dtype=object)
        try:
            speeds = (
                pd.to_numeric(events.get("Speed"), errors="coerce")
                .to_numpy(dtype=float)
            )
        except Exception:
            speeds = np.full(len(events), np.nan, dtype=float)
        machines = np.asarray(machines, dtype=object)
        speeds = np.asarray(speeds, dtype=float)
        # Labels must follow the SAME keep-mask that filtered the
        # coordinates — drop positions, never just truncate prefixes.
        if len(machines) == len(keep):
            machines = machines[keep]
        if len(speeds) == len(keep):
            speeds = speeds[keep]
        self._hover_event_data = (
            np.asarray(ex, dtype=float),
            np.asarray(ey, dtype=float),
            np.asarray(el, dtype=float),
            machines[: len(ex)],
            speeds[: len(ex)],
        )

        # Skip the entire frame when nothing changed (idle cycles are free).
        newest = self.event_log[0] if self.event_log else None
        frame_sig = (
            self._lane_signature,
            len(ex),
            round(float(np.nansum(ex)), 1) if len(ex) else 0.0,
            round(float(np.nansum(ey)), 1) if len(ey) else 0.0,
            tuple(
                (z.name, round(z.center_x, 1), round(z.center_y, 1),
                 round(z.half_size, 1), bool(z.proposed))
                for z in existing_zones
            ),
            tuple(
                (round(c.center_x, 1), round(c.center_y, 1),
                 c.event_count, round(c.score, 1))
                for c in clusters
            ),
            len(self.event_log),
            self._panel_scroll,
            newest,
        )
        if frame_sig == self._frame_sig:
            return
        self._frame_sig = frame_sig

        self._sync_zones(existing_zones, suppression_radius, zone_size)

        self._hover_zones = list(existing_zones)

        self._sync_events(ex, ey, el)
        self._sync_clusters(clusters)

        # Frozen view: lanes define the footprint; set once, then never
        # touch the limits again (user pan/zoom is preserved).  The axes
        # box itself never moves — only data limits change, once.
        if self._fixed_bounds is None:
            self._fixed_bounds = self._resolve_bounds(events)
        if self._fixed_bounds is not None:
            if self._loading_text is not None:
                try:
                    self._loading_text.remove()
                except (ValueError, AttributeError):
                    pass
                self._loading_text = None  # type: ignore[assignment]
            # Only apply when the user hasn't taken over the view: don't
            # stomp an active pan/zoom.  First set always applies.
            if not self._pan_active and not getattr(self, "_view_set", False):
                self._set_view(self._fixed_bounds)
                self._view_set = True  # type: ignore[attr-defined]

        axis.set_title(
            (
                f"RAC events: {len(events)}"
                f"  |  Zones: {len(existing_zones)}"
                f"  |  Clusters: {len(clusters)}"
            ),
            color=TEXT_PRIMARY,
            fontsize=11,
            fontweight="bold",
            loc="left",
        )

        self._draw_panel()
        self._request_draw()

    # ------------------------------------------------------------------
    # Map bounds (frozen, lanes-only)
    # ------------------------------------------------------------------

    def _set_view(self, bounds) -> None:
        """Apply the frozen home limits (axes box itself never moves)."""
        min_x, max_x, min_y, max_y = bounds
        self.map_axis.set_xlim(min_x, max_x)
        self.map_axis.set_ylim(min_y, max_y)

    def _resolve_bounds(
        self,
        events: pd.DataFrame,
    ) -> tuple[float, float, float, float] | None:
        """Fixed map extent from the lane footprint, else events."""
        x_values = list(self._lane_extent_x)
        y_values = list(self._lane_extent_y)

        if not x_values and not events.empty:
            try:
                x_values.extend(
                    pd.to_numeric(events["X"], errors="coerce").dropna().astype(float).tolist()
                )
                y_values.extend(
                    pd.to_numeric(events["Y"], errors="coerce").dropna().astype(float).tolist()
                )
            except Exception:
                pass

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
        """Draw the RAC event list as a compact data-entry style table."""
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
        left = 0.02
        right = 0.98
        fontsize = 9.0
        header_fontsize = 9.0

        weights = [0.30, 0.40, 0.12, 0.18]
        columns = ["TIME", "UNIT", "Lv", "km/h"]
        usable = right - left
        total_w = sum(weights)
        norm = [w / total_w * usable for w in weights]
        edges = [left]
        for w in norm:
            edges.append(edges[-1] + w)
        centres = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(columns))]

        rows_all = list(self.event_log)
        coords_all = list(self._event_coords)

        def _row_level(entry):
            try:
                return int(float(entry[2]))
            except Exception:
                return None

        # Display-only filter: the Filter-by-Severity chips choose which
        # severities appear on the map AND in this table. Level 0 or
        # unparsable levels always stay visible (they do not occur in
        # real data; a stray one should never silently vanish).
        pairs = [
            (row, coord)
            for row, coord in zip(rows_all, coords_all)
            if _row_level(row) is None
            or _row_level(row) < 1
            or _row_level(row) in self.visible_levels
        ]
        self._panel_rows = pairs
        rows = [row for row, _ in pairs]

        if not rows:
            message = (
                "No RAC events since start"
                if not rows_all
                else "All levels hidden (severity chips)"
            )
            panel.add_patch(
                Rectangle(
                    (0.0, top - header_height),
                    1.0,
                    header_height,
                    facecolor="#1C2A24",
                    edgecolor="#35C759",
                    linewidth=1.2,
                    transform=panel.transAxes,
                    zorder=1,
                    clip_on=False,
                )
            )
            for header_text, cx in zip(columns, centres):
                panel.text(
                    cx,
                    top - header_height / 2.0,
                    header_text,
                    va="center",
                    ha="center",
                    transform=panel.transAxes,
                    color="#8CFFA8",
                    fontsize=header_fontsize,
                    fontfamily="monospace",
                    fontweight="bold",
                    zorder=2,
                )
            panel.text(
                0.5,
                0.5,
                message,
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
        total = len(rows)
        max_offset = max(0, total - max_rows)
        if self._panel_scroll > max_offset:
            self._panel_scroll = max_offset
        if self._panel_scroll < 0:
            self._panel_scroll = 0
        visible = rows[self._panel_scroll : self._panel_scroll + max_rows] if total else []

        panel.add_patch(
            Rectangle(
                (0.0, top - header_height),
                1.0,
                header_height,
                facecolor="#1C2A24",
                edgecolor="#35C759",
                linewidth=1.2,
                transform=panel.transAxes,
                zorder=1,
                clip_on=False,
            )
        )
        for header_text, cx in zip(columns, centres):
            panel.text(
                cx,
                top - header_height / 2.0,
                header_text,
                va="center",
                ha="center",
                transform=panel.transAxes,
                color="#8CFFA8",
                fontsize=header_fontsize,
                fontfamily="monospace",
                fontweight="bold",
                zorder=2,
            )

        for index, values in enumerate(visible):
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

            try:
                lvl_val = int(float(values[2]))
            except Exception:
                lvl_val = 0
            if lvl_val <= 2:
                row_color = TEXT_PRIMARY
            else:
                lvl_c = max(1.0, min(5.0, float(lvl_val)))
                row_color = self._severity_colormap(
                    self._severity_mappable.norm(lvl_c)
                )

            for text, cx in zip(values, centres):
                panel.text(
                    cx,
                    row_top - row_height / 2.0,
                    text,
                    va="center",
                    ha="center",
                    transform=panel.transAxes,
                    color=row_color,
                    fontsize=fontsize,
                    fontfamily="monospace",
                    zorder=2,
                )

        if total > max_rows:
            track_x = 0.985
            track_w = 0.012
            track_y0 = bottom_margin
            track_h = top - header_height - bottom_margin
            panel.add_patch(
                Rectangle(
                    (track_x, track_y0),
                    track_w,
                    track_h,
                    facecolor="#2A2E33",
                    edgecolor="#3A444E",
                    linewidth=0.6,
                    transform=panel.transAxes,
                    zorder=5,
                )
            )
            thumb_h = max(0.04, track_h * (max_rows / total))
            if max_offset > 0:
                thumb_y0 = track_y0 + track_h - thumb_h - (self._panel_scroll / max_offset) * (track_h - thumb_h)
            else:
                thumb_y0 = track_y0 + track_h - thumb_h
            panel.add_patch(
                Rectangle(
                    (track_x, thumb_y0),
                    track_w,
                    thumb_h,
                    facecolor="#5A6573",
                    edgecolor="none",
                    transform=panel.transAxes,
                    zorder=6,
                )
            )


def wait_for_next_cycle(
    seconds: int,
    viewer: Viewer | None,
) -> bool:
    """Wait while keeping the viewer's event loop live (throttled)."""
    import time

    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        if viewer is not None:
            if not viewer.is_open():
                return False
            viewer.pump(0.1)
        else:
            time.sleep(0.1)

    return True
