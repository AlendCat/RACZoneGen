# Plan: Guarantee simulation creates visible zones + complete visual chain

**Context:** User reports no zones in simulation and wants to see full chain: raw RAC points → cluster X → 100×100m blue zone (80% opacity, lighter border) → 300m suppression circle. Current `config.ini` had `spawn_per_cycle=2`, `jitter=30`, `level 1-4` random → ~19% of 2-event groups fail `sum(Level)>3` (`threshold_inclusive=false`) and jitter spreads groups near `radius 50m` edge. Need deterministic clustering to reliably demo pipeline.

**Changes Made:**
- `config.ini:54-62` → `spawn_per_cycle=4`, `jitter=15`, `level 3-4`, `speed 45-60`, `hotspots=100,100,4,50;340,180,4,50;620,420,4,50` (fixed level/speed per hotspot). With 4 events/hotspot, min sum `12 >3` → always qualifies; jitter 15 keeps all members inside 50m hub. Verified `StubDB` 4-cycle run creates 2 `RacMonitorGen_zone*` (`viewer.py:458-522` blue `#2563EB α0.80` `#93C5FD` border + dashed `#60A5FA` 300m `Circle`) and `X`/`hub` (`viewer.py:525-569`). Suppression correctly holds at 1 hotspot (`dist 253m <300m`) showing 2 vs 3 zones.

**Verification:**
- `Config()` loads `4/15/3-4/hotspots(4,50)` correctly.
- Headless `run_cycle` with `show_only=true` over `StubDB` → Cycle0 baseline 0, Cycle1+ 12 events → 2 proposed zones, 12→36 synthetic events pruned by `time_window 120`, 2 XMLs in `output/`, respects `zone_size 100` (`half=50`) and `suppression_radius 300`.
- Existing smoke `smoke_sim_test.py` still PASS (its own ini isolated).

**How to run:**
```powershell
python app.py --simulate                     # infinite, synthetic RAC only
python app.py --simulate --cycles 5          # auto-closes after 5 polls (scripted)
python app.py --simulate --config config.ini # explicit file
```
Live DB still used for lanes/zones (`simulator.py:14`); only RAC faked. For visual suppression demo keep `[zone] suppression_radius_metres=300` (`config.ini:28`); set `show_only=true` to keep proposed blue zones in `existing_zones` without `mstarrun` import.

**Risks:**
- Fixed hotspot level 4 hides random-level variation; reset `hotspots` to `X,Y` without level to re-enable random if needed.
- 4×3=12 events/cycle × 120m window → ~360 events retained, still O(n²) `find_clusters` fine.
- Suppression demo shows 2 zones not 3 due to 300m radius; move hotspots >300m apart for 3-zone demo (e.g., `500,100`).

**References:**
- `config.ini:22-30,49-62` zone/suppression/simulation
- `clustering.py:132-225` hub-radius `radius 50`, `minimum_events 2`, `score>3`, `is_new`
- `simulator.py:199-255` `_spawn_batch` jitter/level, `read_rac_events` baseline
- `app.py:165-310` `mark_new_events` → `find_clusters` → `cluster_is_suppressed` → `create_zone_xml`
- `minestar.py:279-288,394-408` square `half=zone_size/2`, `suppression_radius 300`
- `viewer.py:317-350,353-424,525-569` scatter points, `X`, blue zone, dashed 300m circle
