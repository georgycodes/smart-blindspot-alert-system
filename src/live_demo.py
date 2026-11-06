"""
BlindSpot Guardian -- CLI entry point (Stage 3).

Thin wrapper around GuardianPipeline (pipeline.py): tracking + zones +
explainable risk + trajectory prediction + event logging, all in one
plain OpenCV window. The real logic lives in pipeline.py and is shared
with the Streamlit dashboard -- this file is just argument parsing, the
frame loop, and the audio/Arduino alert hookup.

Run with `python src/live_demo.py --help` for all options.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blindspot_detection import AlertSound, ArduinoLink  # noqa: E402  (reused unchanged)

from pipeline import GuardianPipeline, RISK_CRITICAL, RISK_HIGH  # noqa: E402
from video_processor import frames, open_capture  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BlindSpot Guardian - Stage 3 (tracking + zones + prediction + risk + events)"
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
    p.add_argument("--db-path", default="data/events/events.db")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    mirror = isinstance(source, int)  # mirror a live webcam, not a prerecorded file

    if not os.path.exists(args.model):
        print(f"[Fatal] YOLO model not found at '{args.model}'. Pass --model with the correct path.")
        sys.exit(1)

    pipeline = GuardianPipeline(
        model_path=args.model, config_path=args.config, profile=args.profile,
        conf_threshold=args.conf, tracker=args.tracker, db_path=args.db_path,
        source_label=str(args.source),
    )
    if not pipeline.zones.validated:
        print(f"[ZoneManager] Profile '{pipeline.zones.profile_name}' zones are placeholders "
              f"sized for a webcam demo, not calibrated against real vehicle footage.")

    sound = AlertSound("sounds", "child_alert.mp3", "adult_warning.mp3")
    serial_port = "" if args.no_serial else args.serial_port
    arduino = ArduinoLink(serial_port, 9600)

    try:
        cap = open_capture(source, args.width, args.height)
    except RuntimeError as exc:
        print(f"[Fatal] {exc}")
        sys.exit(1)

    window_name = "BlindSpot Guardian - Stage 3"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        for frame in frames(cap, mirror=mirror):
            result = pipeline.process_frame(frame, draw=not args.headless)

            if result.overall_risk == RISK_CRITICAL:
                sound.set_level(AlertSound.LEVEL_CRITICAL)
                arduino.send('3')
            elif result.overall_risk == RISK_HIGH:
                sound.set_level(AlertSound.LEVEL_WARNING)
                arduino.send('2')
            else:
                sound.set_level(AlertSound.LEVEL_NONE)
                arduino.send('0')

            if not args.headless:
                cv2.imshow(window_name, result.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        sound.shutdown()
        arduino.close()
        pipeline.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopped by user.")
