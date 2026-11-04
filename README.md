# Smart BlindSpot Alert System

AI-based predictive blind-spot monitoring for vehicles. Uses YOLOv8 + ByteTrack to detect and track people, maps them to configurable danger zones, and runs a rule-based, explainable risk engine with audio alerts — fully software-only, no hardware required.

Started as a Semester 6 mini-project (detection + distance heuristic + alerts); currently being upgraded for Semester 7 into a tracked, zone-aware, explainable safety-monitoring system, built entirely around a laptop webcam so no bus, Arduino, LED, or buzzer is needed to demo it.

## What it does right now

- **Person detection** — YOLOv8n (`models/yolov8n.pt`), filtered to the COCO "person" class.
- **Persistent multi-person tracking** — Ultralytics' built-in ByteTrack (`model.track(..., persist=True)`) assigns each person a stable `PERSON #ID` across frames, instead of re-detecting independently every frame.
- **Configurable danger zones** — polygon-based zones defined per vehicle profile in [`config/vehicle_profiles.json`](config/vehicle_profiles.json). The default `bus` profile defines FRONT/LEFT/RIGHT/REAR zones, each meant to correspond to one physical camera on a real bus. Only `FRONT` is active by default, since a single webcam can only ever represent one side at a time.
- **Explainable, rule-based risk engine** — for each tracked person, combines *which zone they're in* with *recent movement direction* (APPROACHING / MOVING AWAY / MOVING LEFT / MOVING RIGHT / STATIONARY / CROSSING / UNKNOWN) to produce a risk level (`SAFE → LOW → MODERATE → HIGH → CRITICAL`), with temporal hysteresis to stop the level flickering frame-to-frame, and a plain-English reason attached to every decision (e.g. *"Person is inside the FRONT danger zone. Getting closer to the camera over recent frames."*).
- **Audio alerts** — a looping warning/critical tone (`pygame`) fires automatically once risk reaches `HIGH`/`CRITICAL`.
- **CSV detection logging** — every in-zone detection is appended to `logs/detections_log.csv` with timestamp, risk, estimated distance, and confidence.
- **Optional Arduino hook** — a serial link that can relay simple alert codes to real hardware if/when it exists. It auto-detects a port and silently does nothing if no device is connected, so hardware is never required to run the system.

**Known limitation:** the on-screen distance figure comes from a monocular pinhole-camera heuristic (`estimate_distance()` in `src/blindspot_detection.py`) that assumes an uncalibrated focal length. Treat it as a relative indicator, not a measured distance, until the camera is calibrated.

## Project structure

```
blindspot-human-detection-main/
├── models/
│   └── yolov8n.pt            # YOLOv8-nano weights (gitignored, see Setup below)
├── sounds/
│   ├── adult_warning.mp3     # HIGH-risk alert tone
│   └── child_alert.mp3       # CRITICAL-risk alert tone
├── config/
│   └── vehicle_profiles.json # zone geometry per vehicle profile (bus/car/truck/custom)
├── src/
│   ├── blindspot_detection.py # original Semester 6 script (detection + distance + alerts)
│   ├── live_demo.py          # current entry point: tracking + zones + explainable risk
│   ├── zone_manager.py       # polygon zone loading, point-in-zone test, overlay drawing
│   └── risk_engine.py        # direction classification + rule-based risk + explanations
├── logs/                     # runtime-generated CSV detection logs (gitignored)
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
# current version: tracking + danger zones + explainable risk
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

Press `q` in the display window to quit.

## Roadmap

- [ ] Short-term trajectory prediction (not just current direction)
- [ ] Event/incident persistence (SQLite) with replay
- [ ] Streamlit dashboard: live monitoring, predictive safety panel, zone editor, incident analytics, scenario simulator
- [ ] Formal evaluation suite (detection precision/recall, tracking ID-switch rate, warning lead time)

## Notes for anyone evaluating this project

This is an active academic project (Semester 6 → Semester 7 upgrade), not a production safety system. It does not control a real vehicle's brakes, and distance/zone geometry are software heuristics, not calibrated measurements, until validated against real vehicle camera footage.
