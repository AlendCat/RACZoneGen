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

PALETE_BG = "#121418"
PANEL_BG = "#0d0f12"
TEXT_PRIMARY = "#EAEAEA"
TEXT_GREY = "#9AA0A6"

# Fixed geometry — set once in __init__, never recomputed.  Fractions of
# the figure, so resize scales everything proportionally with no jumps.
LAYOUT = {
    "colorbar": [0.040, 0.150, 0.018, 0.700],
    "map": [0.075, 0.080, 0.545, 0.840],
    "panel": [0.650, 0.080, 0.320, 0.840],
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
                total = len(self.event_log)
                if total == 0:
                    return
                log_index = self._panel_scroll + visible_index
                if log_index < 0 or log_index >= total:
                    return
                coords = list(self._event_coords)
                if log_index >= len(coords):
                    return
                rx, ry = coords[log_index]
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

    def _point_in_panel(self, event) -> bool:
        try:
            bbox = self.panel_axis.bbox
            x, y = getattr(event, "x", None), getattr(event, "y", None)
            if x is None or y is None:
                return False
            return bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1
        except Exception:
            return False

    def _on_scroll(self, event) -> None:
        in_panel = getattr(event, "inaxes", None) is self.panel_axis or self._point_in_panel(event)
        if in_panel:
            top = 0.98
            header_height = 0.06
            bottom_margin = 0.02
            row_height = 0.05
            max_rows = int((top - header_height - bottom_margin) // row_height)
            total = len(self.event_log)
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
            ex, ey, el = ex[finite], ey[finite], el[finite]
        else:
            ex = ey = el = np.empty(0, dtype=float)

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

        rows = list(self.event_log)

        if not rows:
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
