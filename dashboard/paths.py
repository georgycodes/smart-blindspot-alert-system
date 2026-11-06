"""Central path constants for the dashboard, resolved relative to the repo root."""
import os

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(DASHBOARD_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "vehicle_profiles.json")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt")
DB_PATH = os.path.join(ROOT_DIR, "data", "events", "events.db")
SOUNDS_DIR = os.path.join(ROOT_DIR, "sounds")
VIDEOS_DIR = os.path.join(ROOT_DIR, "data", "videos")
