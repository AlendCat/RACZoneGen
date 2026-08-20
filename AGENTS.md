# AGENTS.md

## What this project does

**RACZoneGen** — a single-file Windows executable (built with PyInstaller)
that automates MineStar speed-limit zone creation on a mine site.

It polls the live MineStar SQL Server (`mshist` on `C001749812\MINESTAR`,
ODBC driver) every `poll_interval_seconds`. When a RAC health event is
created, it groups nearby events into severity clusters (the hub-radius
algorithm inherited from Kyle's prototype), and when a cluster qualifies it
builds a 100 m x 100 m square MineStar zone whose speed limit equals the
average vehicle speed of the RAC events minus a configurable offset
(default 10 km/h, floored at 5 km/h). The zone is imported into MineStar on
a **background thread** via `mstarrun.bat importZones`, so the live map
never blocks on MineStar I/O.

Key behavior guarantees:
* **Only events that happen after program start create zones.** The first
  successful RAC sample is a pure baseline; nothing from history is acted on.
* **No duplicate zones.** Before creating a zone the app checks whether any
  existing zone (from its startup in-memory library, refreshed every cycle)
  sits within `suppression_radius_metres` of the cluster centroid.
* **Graceful when idle.** No new RAC events → nothing is generated, nothing
  is imported, the loop simply waits for the next poll.
* **Everything is logged** to `logs/rac_zone_monitor.log` and the console.

## Configuration

The config file is located by checking, in order:
1. `<application dir>/config.ini` (local / portable run)
2. `%PUBLIC%\RACZoneGen\config.ini` (machine-global deployment)

All relative paths in config resolve against the directory that owns the
active config file, so a machine-global install is fully relocatable.

Key settings (`config.ini`):
* `[application] poll_interval_seconds`, `log_level`
* `[rac]` grouping radius, grouping score, `minimum_event_count`, `top_n`
* `[zone]` name prefix (`RacMonitorGen_zone`), `zone_size_metres` (100),
  `suppression_radius_metres` (300), `zones_refresh_interval_seconds` (300)
* `[speed]` `default_speed_kmh`, `speed_offset_kmh` (10), `minimum_speed_kmh` (5)
* `[minestar]` mstarrun path, `import_enabled`, `dry_run`, `import_workers`
* `[display] enabled`

## Runtime flow (`app.py main()` → `run_cycle`)

1. **Read RAC events** from `sql/rac_events.sql`. Must succeed or the whole
   cycle is skipped.
2. **Mark new events** (`mark_new_events`). First sample is a pure baseline.
3. **Group events** (`find_clusters` in `clustering.py`) — greedy
   hub-radius clustering on `X,Y` (radius `grouping_radius_metres`, min
   events `minimum_event_count`, score = summed `Level` above
   `grouping_score`, capped at `top_n`). Only seeds that contain a *new*
   event can form a cluster. Carries the members' average speed.
4. **Read lanes** — throttled by `lanes.refresh_interval_seconds`; failure
   is non-fatal, previous lane data is retained.
5. **Read existing zones** — throttled by `zone.zones_refresh_interval_seconds`
   (default 300 s). Prefers a **fast direct SQL read** of `msmodel.dbo.ZONE`
   (`sql/zones.sql`, returns the bounding box of every active zone) with a
   fallback to `mstarrun -b exportZones` if the direct query fails. Failure
   blocks imports that cycle. Success-with-no-zones is treated as *zero
   zones with a warning* (fixed bug — previously an empty export blocked
   zone creation forever on first deployment).
6. **Suppress + create zones** — clusters within
   `suppression_radius_metres` of an existing `RacMonitorGen_zone*` zone
   are skipped (`cluster_is_suppressed`). Remaining clusters get a square
   polygon XML from the `zones.xml` template written to `output/`.
7. **Speed limit** = `cluster.average_speed_kmh - speed_offset_kmh`
   clamped to `[minimum_speed_kmh, maximum_speed_kmh]` (no speed parsed →
   `default_speed_kmh`). Written to the `speedLimit` magnitude.
8. **Import** is submitted to a single background `ThreadPoolExecutor` so
   MineStar commands are serialized while the map stays live. Successful
   imports are appended to the in-memory zone library so the same cycle
   cannot create overlapping zones.
9. **Sample-commit semantics** — the snapshot commits as the next baseline
   *unless* the zone export failed or a real import failed, in which case
   it is withheld so those new events get retried next cycle.
10. **Viewer** — live map: lanes drawn as a *cached* PatchCollection (only
    rebuilt when lane data changes), RAC events coloured by severity, cyan
    halos on new events, square zones + dashed suppression circles,
    cluster hubs, and a side panel with a recent RAC event list
    (time + X/Y + level + speed) and an activity log.

## Zone naming

`<name_prefix>_<YYYYMMDD_HHMMSS>_<X>_<Y>` e.g.
`RacMonitorGen_zone_20260820_124955_102.0_100.8`. The prefix is the
filter used to build the in-memory zone library at startup and for
duplicate suppression.

## Layout

| Path | Purpose |
|------|---------|
| `app.py` | Main entry, cycle orchestration, background import, retry/commit logic. |
| `config.py` | Config discovery (local → `%PUBLIC%\RACZoneGen`) + typed, validated settings. |
| `database.py` | ODBC engine, RAC event reads (+ payload speed parsing), lane reads. |
| `clustering.py` | `mark_new_events` + Kyle's hub-radius `find_clusters`. |
| `minestar.py` | mstarrun commands, square zone XML generation, speed limit calc, import. |
| `viewer.py` | Live matplotlib map + RAC event/activity side panel. |
| `config.ini` | All runtime settings (resolved relative to the app/exe folder). |
| `sql/rac_events.sql` | RAC query; `{time_cutoff_minutes}` substituted at runtime. |
| `sql/lanes.sql` | Left/right edge points of active autonomous lanes. |
| `sql/zones.sql` | Fast direct read of active zones (envelope bounding boxes). |
| `zones.xml` | Zone XML **template** (edited in place to create new zones). |
| `Kyles_code/` | Original Kyle McIlroy prototype scripts (reference only, gitignored). |
| `build_exe.bat` | Pip-installs deps, builds onefile `dist\RACZoneGen.exe`. |

## Build & run (Windows / PowerShell)

```powershell
.venv\Scripts\Activate.ps1
python app.py          # run from the project root
.\build_exe.bat        # produce dist\RACZoneGen.exe
```

No test suite is configured; verification is manual against the live
MineStar/SQL Server. `config.ini` ships with `import_enabled=false` /
`dry_run=true` — imports must be explicitly enabled and dry-run turned off
to create real zones. `display.enabled` can be set to `false` for
headless/service use.

## Issues & constraints

* **Payload speed parsing is best-effort.** `database._payload_speed`
  tries JSON keys (`Speed`, `VehicleSpeed`, …), a bare float, or a
  `speed=` text match; anything unparseable falls back to
  `default_speed_kmh`. Confirm the actual `HEALTH_EVENT.PAYLOAD` shape on
  the test site and adjust the key list if speeds look wrong.
* **Cluster seeds are X-descending** (deterministic but coordinate-
  dependent); changing the sort order changes cluster membership.
* `find_clusters` is O(n²) distance work per seed — fine for typical RAC
  volumes, but watch it if event counts explode.
* **Viewer lanes are cached** — ~74k points / ~4800 lanes are rebuilt
  into the PatchCollection *only when lane data changes*, so per-cycle
  repaints are cheap. Lanes are only *fetched* on the refresh interval.
* `wait_with_viewer` pumps the GUI event loop while background MineStar
  tasks run, so the window never shows "Not responding" during the ~20 s
  `exportZones` subprocess.
* Matplotlib must stay on the main thread; MineStar imports run on a
  worker thread and never touch matplotlib.
* `mstarrun.bat` must be invoked via `cmd /d /c` (handled in
  `MineStar._command`).

## Recommended optimizations (future)

1. Incremental map updates (offsets only) instead of a full dynamic-artist
   rebuild, and a PNG snapshot mode for headless runs.
2. Add pytest coverage for `clustering`, XML generation, and suppression.
3. Exponential backoff for DB/MineStar outages instead of a fixed retry.
4. Deterministic, hashed seed ordering in `find_clusters` with a unit test
   locking the behaviour.
5. PyInstaller `.spec` file with version/icon instead of the shell copy
   steps (keeps `sql/`, `config.ini`, `zones.xml` bundled reproducibly).