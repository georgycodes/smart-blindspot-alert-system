"""
BlindSpot Guardian -- Stage 2 live demo.

Adds on top of the original blindspot_detection.py:
  - persistent person tracking (ByteTrack, via Ultralytics' built-in tracker)
  - polygon-based danger zones loaded from a vehicle profile (zone_manager.py)
  - a rule-based, explainable risk engine (risk_engine.py): zone occupancy +
    recent movement direction -> SAFE/LOW/MODERATE/HIGH/CRITICAL, each with
    a plain-English reason

Removed: the old bounding-box-height child/adult classification -- every
detected person is now labelled PERSON, since that heuristic is not a
reliable way to judge age.

Not yet implemented (later stages): trajectory prediction/extrapolation,
event persistence (SQLite), and the Streamlit dashboard. This is still a
plain OpenCV window, same as the original script.

Reuses AlertSound, ArduinoLink and DetectionLogger unchanged from
blindspot_detection.py -- audio alerts, the optional Arduino serial link,
and CSV logging all behave exactly as before.

Run with `python src/live_demo.py --help` for all options.
"""
import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blindspot_detection import AlertSound, ArduinoLink, DetectionLogger  # noqa: E402

from risk_engine import (  # noqa: E402
    RiskEngine, RISK_ORDER, RISK_SAFE, RISK_LOW, RISK_MODERATE, RISK_HIGH, RISK_CRITICAL,
)
from zone_manager import ZoneManager  # noqa: E402


RISK_COLORS = {
    RISK_SAFE: (0, 255, 0),
    RISK_LOW: (0, 255, 0),
    RISK_MODERATE: (0, 255, 255),
    RISK_HIGH: (0, 165, 255),
    RISK_CRITICAL: (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BlindSpot Guardian - Stage 2 (tracking + zones + explainable risk)"
    )
    p.add_argument("--source", default="0", help="camera index (e.g. 0) or path to a video file")
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--config", default="config/vehicle_profiles.json")
    p.add_argument("--profile", default=None, help="vehicle profile name; defaults to the config's active_profile")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--serial-port", default="auto", help='"auto", a port name, or "" to disable the Arduino link')
    p.add_argument("--no-serial", action="store_true", help="disable the Arduino link entirely")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--headless", action="store_true", help="no display window")
    p.add_argument("--log-path", default="logs/detections_log.csv")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if not os.path.exists(args.model):
        print(f"[Fatal] YOLO model not found at '{args.model}'. Pass --model with the correct path.")
        sys.exit(1)
    model = YOLO(args.model)

    zones = ZoneManager(args.config, profile=args.profile)
    if not zones.validated:
        print(f"[ZoneManager] Profile '{zones.profile_name}' zones are placeholders "
              f"sized for a webcam demo, not calibrated against real vehicle footage.")
    if not zones.active_zones():
        print(f"[ZoneManager] Warning: profile '{zones.profile_name}' has no active zones; "
              f"nothing will be flagged as in-danger.")

    risk_engine = RiskEngine()
    sound = AlertSound("sounds", "child_alert.mp3", "adult_warning.mp3")
    serial_port = "" if args.no_serial else args.serial_port
    arduino = ArduinoLink(serial_port, 9600)
    logger = DetectionLogger(args.log_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Fatal] Could not open video source '{source}'.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "BlindSpot Guardian - Stage 2"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] Frame grab failed; stopping.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            now = time.time()
            fps = 1.0 / (now - prev_time) if prev_time else 0.0
            prev_time = now

            results = model.track(frame, classes=[0], conf=args.conf, tracker=args.tracker,
                                   persist=True, verbose=False)

            if not args.headless:
                zones.draw(frame, frame_w, frame_h)

            worst_risk = RISK_SAFE
            seen_ids = set()
            people_in_zone = 0

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue
                for box, track_id in zip(r.boxes, r.boxes.id):
                    tid = int(track_id)
                    seen_ids.add(tid)

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0]) if box.conf is not None else 0.0

                    cx_frac = ((x1 + x2) / 2) / frame_w
                    cy_frac = ((y1 + y2) / 2) / frame_h
                    box_h_frac = (y2 - y1) / frame_h

                    zone = zones.zone_for_point(cx_frac, cy_frac, frame_w, frame_h)
                    zone_name = zone.name if zone else None

                    state = risk_engine.update(tid, cx_frac, box_h_frac, zone_name, now)

                    if zone_name:
                        people_in_zone += 1
                        logger.log("PERSON", state.risk, box_h_frac * frame_h, confidence)
                    if RISK_ORDER[state.risk] > RISK_ORDER[worst_risk]:
                        worst_risk = state.risk

                    if not args.headless:
                        color = RISK_COLORS[state.risk]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"PERSON #{tid:02d}  {state.direction}  {state.risk}"
                        cv2.putText(frame, label, (int(x1), max(15, int(y1) - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        reason = state.reasons[-1] if state.reasons else ""
                        cv2.putText(frame, reason, (int(x1), min(frame_h - 5, int(y2) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            risk_engine.forget_stale(seen_ids)

            if worst_risk == RISK_CRITICAL:
                sound.set_level(AlertSound.LEVEL_CRITICAL)
                arduino.send('3')
            elif worst_risk == RISK_HIGH:
                sound.set_level(AlertSound.LEVEL_WARNING)
                arduino.send('2')
            else:
                sound.set_level(AlertSound.LEVEL_NONE)
                arduino.send('0')

            if not args.headless:
                cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PEOPLE IN DANGER ZONE: {people_in_zone}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"OVERALL RISK: {worst_risk}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, RISK_COLORS[worst_risk], 2)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        sound.shutdown()
        arduino.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopped by user.")
