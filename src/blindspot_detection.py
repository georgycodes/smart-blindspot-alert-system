"""
AI Bus Blind Spot Human Detection System
-----------------------------------------
Detects pedestrians (adults vs. children) in a bus blind-spot camera feed,
estimates their distance, raises audio/visual/hardware alerts, and logs
every detection event to CSV.

Run with `python src/blindspot_detection.py --help` to see all options.
"""

import argparse
import csv
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from ultralytics import YOLO


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
@dataclass
class Config:
    source: int = 0                 # camera index, or a video file path
    width: int = 1280
    height: int = 720

    model_path: str = "models/yolov8n.pt"

    # Monocular pinhole-camera distance estimate.
    # NOTE: this is a heuristic, not a calibrated measurement. It assumes
    # the full body height is visible and upright, and focal_length_px
    # must be calibrated for your specific camera for the numbers to mean
    # anything in real units. Treat distances as relative risk bands,
    # not as exact metres.
    focal_length_px: float = 700.0
    adult_height_m: float = 1.7
    child_height_m: float = 1.1
    child_height_ratio_threshold: float = 0.4  # box_height/frame_height below this => CHILD

    # Risk bands (metres), nearest-first
    critical_dist_m: float = 2.5
    warning_dist_m: float = 3.5
    caution_dist_m: float = 4.5
    safe_dist_m: float = 5.5

    conf_threshold: float = 0.35

    # Blind-spot region of interest, as a fraction of frame (x1, y1, x2, y2).
    # Only detections whose box center falls inside this box are treated as
    # "in the blind spot." Defaults to the full frame.
    roi: tuple = (0.0, 0.0, 1.0, 1.0)

    serial_port: str = "auto"       # "auto", a port name like "COM3"/"/dev/ttyUSB0", or "" to disable
    serial_baud: int = 9600

    sound_dir: str = "sounds"
    child_sound: str = "child_alert.mp3"
    adult_sound: str = "adult_warning.mp3"

    log_path: str = "logs/detections_log.csv"

    fullscreen: bool = False
    headless: bool = False          # run with no display window (e.g. on a server)


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Bus blind-spot human detection")
    p.add_argument("--source", default=str(cfg.source),
                    help="camera index (e.g. 0) or path to a video file")
    p.add_argument("--model", default=cfg.model_path)
    p.add_argument("--serial-port", default=cfg.serial_port,
                    help='"auto", a port name, or "" to disable Arduino link')
    p.add_argument("--no-serial", action="store_true", help="disable Arduino link entirely")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--headless", action="store_true", help="no display window")
    p.add_argument("--log-path", default=cfg.log_path)
    args = p.parse_args()

    cfg.source = int(args.source) if args.source.isdigit() else args.source
    cfg.model_path = args.model
    cfg.serial_port = "" if args.no_serial else args.serial_port
    cfg.fullscreen = args.fullscreen
    cfg.headless = args.headless
    cfg.log_path = args.log_path
    return cfg


# --------------------------------------------------
# SERIAL (ARDUINO) — optional, auto-detecting, never fatal
# --------------------------------------------------
class ArduinoLink:
    """Best-effort serial link. If no Arduino is present or pyserial isn't
    installed, every method becomes a harmless no-op so the rest of the
    system runs in software-only mode."""

    def __init__(self, port: str, baud: int):
        self.conn = None
        if serial is None or not port:
            return

        target_port = port
        if port == "auto":
            target_port = self._autodetect()
            if target_port is None:
                print("[Arduino] No serial device found, running without hardware alerts.")
                return

        try:
            self.conn = serial.Serial(target_port, baud, timeout=1)
            time.sleep(2)  # allow the board to reset
            print(f"[Arduino] Connected on {target_port}")
        except (serial.SerialException, OSError) as exc:
            print(f"[Arduino] Could not open {target_port}: {exc}. Running without hardware alerts.")
            self.conn = None

    @staticmethod
    def _autodetect():
        if list_ports is None:
            return None
        ports = list(list_ports.comports())
        return ports[0].device if ports else None

    def send(self, signal: str):
        if self.conn is None:
            return
        try:
            self.conn.write(signal.encode())
        except (serial.SerialException, OSError) as exc:
            print(f"[Arduino] Write failed ({exc}); disabling hardware link.")
            self.conn = None

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass


# --------------------------------------------------
# SOUND SYSTEM — plays the actual alert/warning audio files
# --------------------------------------------------
class AlertSound:
    """Loops one of two real audio files (child alarm vs adult warning beep)
    on a background thread. Falls back to silent no-op if pygame or the
    sound files aren't available, so a missing audio asset never crashes
    detection."""

    LEVEL_NONE = 0
    LEVEL_WARNING = 1
    LEVEL_CRITICAL = 2

    def __init__(self, sound_dir: str, child_sound: str, adult_sound: str):
        self.enabled = pygame is not None
        self._level = self.LEVEL_NONE
        self._lock = threading.Lock()
        self._thread = None
        self._stop_flag = threading.Event()
        self.sounds = {}

        if not self.enabled:
            print("[Audio] pygame not installed; running without audio alerts.")
            return

        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
        except pygame.error as exc:
            print(f"[Audio] Could not init audio device ({exc}); running without audio alerts.")
            self.enabled = False
            return

        for level, fname in ((self.LEVEL_CRITICAL, child_sound), (self.LEVEL_WARNING, adult_sound)):
            path = os.path.join(sound_dir, fname)
            if os.path.exists(path):
                try:
                    self.sounds[level] = pygame.mixer.Sound(path)
                except pygame.error as exc:
                    print(f"[Audio] Failed to load {path}: {exc}")
            else:
                print(f"[Audio] Missing sound file: {path}")

    def set_level(self, level: int):
        if not self.enabled:
            return
        with self._lock:
            if level == self._level:
                return
            self._level = level

        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._stop_flag.clear()

        if level == self.LEVEL_NONE or level not in self.sounds:
            return

        self._thread = threading.Thread(target=self._loop, args=(level,), daemon=True)
        self._thread.start()

    def _loop(self, level: int):
        sound = self.sounds[level]
        channel = sound.play(loops=-1)
        while not self._stop_flag.is_set():
            time.sleep(0.1)
        if channel is not None:
            channel.stop()

    def shutdown(self):
        self.set_level(self.LEVEL_NONE)


# --------------------------------------------------
# DETECTION LOGGING
# --------------------------------------------------
class DetectionLogger:
    """Appends one row per detected person per frame to a CSV file.
    Creates the file with a header if it doesn't already exist."""

    FIELDS = ["timestamp", "type", "risk", "distance_m", "confidence"]

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, person_type: str, risk: str, distance_m: float, confidence: float):
        try:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    person_type,
                    risk,
                    round(distance_m, 2),
                    round(confidence, 3),
                ])
        except OSError as exc:
            print(f"[Log] Could not write detection: {exc}")


# --------------------------------------------------
# DISTANCE ESTIMATION
# --------------------------------------------------
def estimate_distance(box_height_px: float, real_height_m: float, focal_px: float) -> float:
    if box_height_px <= 0:
        return float("inf")
    return (real_height_m * focal_px) / box_height_px


def classify_risk(dist_m: float, cfg: Config) -> tuple:
    if dist_m < cfg.critical_dist_m:
        return "CRITICAL", (0, 0, 255)
    if dist_m < cfg.warning_dist_m:
        return "WARNING", (0, 165, 255)
    if dist_m < cfg.caution_dist_m:
        return "CAUTION", (0, 255, 255)
    if dist_m < cfg.safe_dist_m:
        return "SAFE", (0, 255, 0)
    return "OUT", (200, 255, 200)


# --------------------------------------------------
# MAIN ENGINE
# --------------------------------------------------
class BlindSpotGuardian:

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(
                f"YOLO model not found at '{cfg.model_path}'. "
                f"Pass --model with the correct path."
            )
        self.model = YOLO(cfg.model_path)

        self.sound = AlertSound(cfg.sound_dir, cfg.child_sound, cfg.adult_sound)
        self.arduino = ArduinoLink(cfg.serial_port, cfg.serial_baud)
        self.logger = DetectionLogger(cfg.log_path)

        self.running = False

    def _in_roi(self, cx_frac: float, cy_frac: float) -> bool:
        x1, y1, x2, y2 = self.cfg.roi
        return x1 <= cx_frac <= x2 and y1 <= cy_frac <= y2

    def run(self):
        cfg = self.cfg
        cap = cv2.VideoCapture(cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source '{cfg.source}'.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

        window_name = "BlindSpot Guardian"
        if not cfg.headless:
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            if cfg.fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.running = True
        prev_time = 0.0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[Camera] Frame grab failed; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                frame_h, frame_w = frame.shape[:2]

                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
                prev_time = current_time

                results = self.model.predict(frame, classes=[0], conf=cfg.conf_threshold, verbose=False)

                people_count = 0
                blindspot_critical = False
                blindspot_warning = False

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0]) if box.conf is not None else 0.0
                        box_height = y2 - y1

                        cx_frac = ((x1 + x2) / 2) / frame_w
                        cy_frac = ((y1 + y2) / 2) / frame_h
                        in_blindspot = self._in_roi(cx_frac, cy_frac)

                        height_ratio = box_height / frame_h
                        if height_ratio < cfg.child_height_ratio_threshold:
                            person_type = "CHILD"
                            real_height = cfg.child_height_m
                        else:
                            person_type = "ADULT"
                            real_height = cfg.adult_height_m

                        dist = estimate_distance(box_height, real_height, cfg.focal_length_px)
                        risk, color = classify_risk(dist, cfg)

                        if in_blindspot:
                            people_count += 1
                            self.logger.log(person_type, risk, dist, confidence)
                            if risk == "CRITICAL":
                                blindspot_critical = True
                            elif risk == "WARNING":
                                blindspot_warning = True
                        else:
                            # Detected, but outside the monitored blind-spot zone.
                            color = tuple(int(c * 0.5) for c in color)

                        if not cfg.headless:
                            box_color = color if in_blindspot else (120, 120, 120)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2 if in_blindspot else 1)
                            label = f"{person_type} {risk} {dist:.1f}m"
                            cv2.putText(frame, label, (int(x1), max(15, int(y1) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # ---------------- ALERT LOGIC ----------------
                if blindspot_critical:
                    self.sound.set_level(AlertSound.LEVEL_CRITICAL)
                    self.arduino.send('3')
                elif blindspot_warning:
                    self.sound.set_level(AlertSound.LEVEL_WARNING)
                    self.arduino.send('2')
                else:
                    self.sound.set_level(AlertSound.LEVEL_NONE)
                    self.arduino.send('0')

                if not cfg.headless:
                    self._draw_roi_and_hud(frame, frame_w, frame_h, fps, people_count)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            self.running = False
            cap.release()
            if not cfg.headless:
                cv2.destroyAllWindows()
            self.sound.shutdown()
            self.arduino.close()

    def _draw_roi_and_hud(self, frame, frame_w, frame_h, fps, people_count):
        x1f, y1f, x2f, y2f = self.cfg.roi
        if (x1f, y1f, x2f, y2f) != (0.0, 0.0, 1.0, 1.0):
            pt1 = (int(x1f * frame_w), int(y1f * frame_h))
            pt2 = (int(x2f * frame_w), int(y2f * frame_h))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame, "BLIND SPOT ZONE", (pt1[0], max(15, pt1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PEOPLE IN BLIND SPOT: {people_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    cfg = parse_args()
    try:
        engine = BlindSpotGuardian(cfg)
        engine.run()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[Fatal] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopped by user.")
