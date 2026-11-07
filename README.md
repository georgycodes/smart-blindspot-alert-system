# Smart BlindSpot Alert System

AI-based predictive blind-spot monitoring for vehicles. Uses YOLOv8 + ByteTrack to detect and track people, maps them to configurable danger zones, and runs a rule-based, explainable risk engine — with short-term movement prediction, automatic safety-event logging, and a Streamlit dashboard — fully software-only, no hardware required.

Started as a Semester 6 mini-project (detection + distance heuristic + alerts); currently being upgraded for Semester 7 into a tracked, zone-aware, predictive, explainable safety-monitoring platform, built entirely around a laptop webcam so no bus, Arduino, LED, or buzzer is needed to demo it.

## What it does right now

- **Person detection** — YOLOv8n (`models/yolov8n.pt`), filtered to the COCO "person" class.
- **Persistent multi-person tracking** — Ultralytics' built-in ByteTrack (`model.track(..., persist=True)`) assigns each person a stable `PERSON #ID` across frames, instead of re-detecting independently every frame.
- **Configurable danger zones** — polygon-based zones defined per vehicle profile in [`config/vehicle_profiles.json`](config/vehicle_profiles.json). The default `bus` profile defines FRONT/LEFT/RIGHT/REAR zones, each meant to correspond to one physical camera on a real bus. Only `FRONT` is active by default, since a single webcam can only ever represent one side at a time.
- **Movement direction analysis** — each tracked person's recent position history is smoothed and classified as APPROACHING / MOVING AWAY / MOVING LEFT / MOVING RIGHT / STATIONARY / CROSSING / UNKNOWN. These are image-space directions, not real-world directions, until the camera is calibrated.
- **Short-term trajectory prediction** — a simple linear extrapolation of recent movement (not a learned model) projects a few seconds ahead. When that projection would cross into a danger zone the person isn't in yet, the system reports *which* zone and an estimated time-to-entry, clearly as an estimate, and only when there's enough tracking history to support one (otherwise it says so rather than guessing).
- **Predictive zone-conflict detection** — the risk engine escalates *before* someone actually enters a zone if their projected path says they're about to, not just once they're already in it.
- **Explainable, rule-based risk engine** — combines zone occupancy, movement direction, and predicted entry into a risk level (`SAFE → LOW → MODERATE → HIGH → CRITICAL`), with temporal hysteresis to stop the level flickering frame-to-frame, and a plain-English reason attached to every decision (e.g. *"Person is inside the FRONT danger zone. Getting closer to the camera over recent frames."*). No LLM involved — every reason is built directly from the same rule-based logic that set the risk level.
- **Safety-risk event logging** — when risk reaches HIGH/CRITICAL, one event is opened and persisted to SQLite (`data/events/events.db`); it's extended while the situation continues and closed when it clears, so one real encounter is one row, not one row per frame. These are rule-based flags, not verified collisions or confirmed near-misses.
- **Audio alerts** — a looping warning/critical tone (`pygame`) fires automatically once risk reaches `HIGH`/`CRITICAL`.
- **Optional Arduino hook** — a serial link that can relay simple alert codes to real hardware if/when it exists. It auto-detects a port and silently does nothing if no device is connected, so hardware is never required to run the system.
- **Streamlit dashboard** (`dashboard/app.py`) — five tabs, all driven by the same pipeline as the CLI script:
  - **Live Monitoring** — start/stop webcam monitoring, pick a vehicle profile, adjust detection confidence, mute/unmute audio, live annotated feed, FPS/risk/zone-occupancy readout.
  - **Predictive Safety** — a live per-person panel: zone, direction, predicted entry + ETA, and the reason behind the current risk level.
  - **Zone Configuration** — a visual, click-to-place polygon editor: click points directly on the camera preview to draw a zone's boundary, with live preview, undo/clear/reset, and save back to `config/vehicle_profiles.json`.
  - **Incident Analytics** — real charts (Plotly) and a filterable event table computed from the SQLite log; shows an honest empty state with no events yet, never fabricated numbers. Includes **incident replay** for events whose source video is still in `data/videos/` (re-runs the full pipeline over the frames around that event so the original detection/zone/risk overlays show up again, scrubbable frame by frame). Live webcam events have no saved recording, so they're honestly marked as not replayable.
  - **Scenario Simulator** — upload or pick a test video and run it through the exact same pipeline with start/pause/resume/stop controls, independent of the live webcam.
- **Evaluation suite** (`src/evaluation.py`) — detection precision/recall/F1 (IoU-matched against hand-annotated ground-truth boxes), tracking ID-switch counting, predictive zone-entry accuracy + average warning lead time, and performance (FPS, per-frame processing time, CPU/memory via `psutil`). Detection/tracking/predictive metrics need ground truth this project doesn't have real footage for yet — see "Running an evaluation" below; the module refuses to fabricate numbers in its place. Performance metrics need no ground truth and have been run for real (see `reports/performance_report.md`).
- **Automated tests** (`tests/`, pytest) — 42 tests covering zone geometry, the risk engine (including hysteresis), trajectory prediction, the event state machine, the evaluation math (via synthetic/fake detector output), and one real end-to-end integration test that loads the actual YOLO model and runs it through the full pipeline.

**Known limitations:**
- The on-screen distance figure comes from a monocular pinhole-camera heuristic (`estimate_distance()` in `src/blindspot_detection.py`) that assumes an uncalibrated focal length. Treat it as a relative indicator, not a measured distance, until the camera is calibrated.
- Zone polygons for the `bus` profile are placeholders sized for a laptop webcam demo, not calibrated against real vehicle camera footage.
- Trajectory prediction is a simple linear extrapolation of recent motion, not a learned motion model — it's deliberately simple and says so.
- The Zone Configuration editor places points by clicking, not full drag-to-reposition.
- Detection/tracking/predictive-safety evaluation have not actually been *run* yet — no real annotated test footage exists for this project. The tooling is real and tested (see `tests/test_evaluation.py`); only the performance (FPS/timing) metrics have real measured numbers so far.

## Project structure

```
blindspot-human-detection-main/
├── models/
│   └── yolov8n.pt              # YOLOv8-nano weights (gitignored, see Setup below)
├── sounds/
│   ├── adult_warning.mp3       # HIGH-risk alert tone
│   └── child_alert.mp3         # CRITICAL-risk alert tone
├── config/
│   └── vehicle_profiles.json   # zone geometry per vehicle profile (bus/car/truck/custom)
├── data/
│   ├── events/                 # SQLite safety-event log (gitignored, created at runtime)
│   └── videos/                 # uploaded/test videos for the Scenario Simulator (gitignored)
├── src/
│   ├── blindspot_detection.py  # original Semester 6 script (detection + distance + alerts)
│   ├── live_demo.py            # CLI entry point: thin wrapper around pipeline.py
│   ├── pipeline.py             # shared core: detection+tracking+zones+risk+prediction+events
│   ├── zone_manager.py         # polygon zone loading, point-in-zone test, overlay drawing
│   ├── risk_engine.py          # direction classification + rule-based risk + explanations
│   ├── trajectory_prediction.py # linear extrapolation + predicted zone-entry/ETA
│   ├── event_manager.py        # safety-risk event state machine + SQLite persistence
│   ├── video_processor.py      # shared webcam/file frame source
│   └── evaluation.py           # detection/tracking/predictive/performance metrics
├── dashboard/
│   ├── app.py                  # Streamlit shell (5 tabs)
│   ├── monitoring.py           # Tab 1: Live Monitoring
│   ├── predictive_safety.py    # Tab 2: Predictive Safety
│   ├── zone_configuration.py   # Tab 3: Zone Configuration (click-to-place editor)
│   ├── analytics.py            # Tab 4: Incident Analytics (+ incident replay)
│   ├── simulator.py            # Tab 5: Scenario Simulator
│   └── paths.py                # shared path constants
├── tests/                      # pytest suite (unit + one real integration test)
│   └── fixtures/sample_annotations.json  # ground-truth annotation schema example
├── reports/                     # evaluation reports (e.g. performance_report.md)
├── logs/                       # runtime-generated CSV detection logs (gitignored, from blindspot_detection.py)
├── requirements.txt
└── .gitignore
```

## Setup

Requires Python 3.11+ (tested on 3.13) on Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The YOLOv8n weights file (`models/yolov8n.pt`) is **not** committed to this repo (see `.gitignore` — large binary weights are better fetched than versioned). Get it with either:

```powershell
# downloads yolov8n.pt into the current directory on first use
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

then move the downloaded `yolov8n.pt` into `models/`, or download it directly from the [official Ultralytics releases](https://github.com/ultralytics/assets/releases) into `models/yolov8n.pt`.

## Running it

From the project root (`blindspot-human-detection-main/`):

```powershell
# dashboard (recommended): all 5 tabs, webcam + simulator + analytics
.venv\Scripts\python.exe -m streamlit run dashboard/app.py

# CLI version: tracking + zones + prediction + risk + events, plain OpenCV window
.venv\Scripts\python.exe src\live_demo.py

# original Semester 6 script (detection + distance heuristic only, no tracking/zones)
.venv\Scripts\python.exe src\blindspot_detection.py
```

Useful flags for `live_demo.py`:

| Flag | Meaning |
|---|---|
| `--source 0` | camera index (default `0`), or a path to a video file |
| `--model models/yolov8n.pt` | path to the YOLO weights |
| `--profile bus` | vehicle profile to load from `config/vehicle_profiles.json` |
| `--conf 0.35` | detection confidence threshold |
| `--no-serial` | disable the optional Arduino link entirely |
| `--headless` | run with no display window (e.g. on a server) |
| `--fullscreen` | open the display window fullscreen |

Press `q` in the display window to quit. In the dashboard, use the Start/Stop buttons on the Live Monitoring tab instead.

## Running the tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

42 tests, including one real integration test that loads the actual YOLO model (so the first run is a little slower while it loads).

## Running an evaluation

Performance (FPS / processing time / resource usage) needs no ground truth and can be run right now:

```powershell
.venv\Scripts\python.exe src\evaluation.py --source 0 --max-frames 200
```

This writes `reports/performance_report.json` and `.md`. A real run against this project's development machine is already checked into `reports/performance_report.md`.

Detection, tracking, and predictive-safety evaluation need manually annotated ground truth (bounding boxes, track identities, or confirmed zone-entry frames) — see the schema and worked examples in `tests/fixtures/sample_annotations.json`, and call `run_detection_eval` / `run_tracking_eval` / `run_predictive_eval` from `src/evaluation.py` once you have a real annotated clip. No results are reported here because no annotated footage exists for this project yet.

## Roadmap

- [x] Short-term trajectory prediction + predictive zone-conflict detection
- [x] Event/incident persistence (SQLite)
- [x] Streamlit dashboard: live monitoring, predictive safety panel, zone editor, incident analytics, scenario simulator
- [x] Incident replay (for events whose source video is still available)
- [x] Visual zone editor (click-to-place; not full drag-and-drop yet)
- [x] Evaluation tooling (detection/tracking/predictive/performance) + automated test suite
- [ ] Actually run detection/tracking/predictive evaluation against real annotated footage (tooling exists, data doesn't yet)
- [ ] Zone geometry calibrated against real bus camera footage (currently webcam-demo placeholders)
- [ ] Full drag-and-drop zone editor (currently click-to-place)

## Notes for anyone evaluating this project

This is an active academic project (Semester 6 → Semester 7 upgrade), not a production safety system. It does not control a real vehicle's brakes, "safety-risk events" are rule-based flags rather than verified collisions or confirmed near-misses, and distance/zone geometry are software heuristics, not calibrated measurements, until validated against real vehicle camera footage.
