# RACZoneGen

Automated MineStar speed-limit zone creation from RAC health events.

Poll the MineStar `mshist` database for RAC events, group nearby events
into severity clusters, and create square 100 m x 100 m speed-limit zones
(avg vehicle speed − 10 km/h, floored at 5 km/h) automatically with
`mstarrun.bat importZones` — all on a background thread so the live map
never blocks.

## Running the simulation (synthetic RAC events, live lanes/zones)

Simulation fakes **only the RAC event stream** — an additive stream of
synthetic RAC events that grows each poll cycle, so you can watch the
real pipeline (baseline → clustering → suppression → zone XML creation)
process new events from start to finish. **Lanes and existing zones are
read live from your MineStar SQL Server**, exactly as in production.

```powershell
.venv\Scripts\Activate.ps1
python app.py --simulate --cycles 5
```

The simulator auto-spawns `spawn_per_cycle` events at each configured
hotspot under the `[simulation]` section of `config.ini` (hotspots,
jitter, level/speed ranges). Pair with `[minestar] show_only = true` to
see zones appear on the map as green "PROPOSED" without touching
mstarrun, or with `import_enabled=true` + `dry_run=false` to actually
import synthetic-event zones. A live SQL Server connection is required
(for lanes/zones); `mstarrun.bat` is never invoked in simulate/show-only
mode. Pass `--config <path>` to use an alternative config file.

## Show-only mode (preview zones without touching mstarrun)

Set `[minestar] show_only = true` to run the real pipeline against the
live MineStar database but **never invoke mstarrun**. Qualifying zones
are computed and drawn on the map as green dashed "PROPOSED" zones (a
review copy is still written to `output/`), so you can see exactly what
would be created before enabling real imports. This is the only layer
that suppresses the MineStar command — `import_enabled` / `dry_run`
only gate the command once show-only is off.

Full documentation lives in [AGENTS.md](AGENTS.md).