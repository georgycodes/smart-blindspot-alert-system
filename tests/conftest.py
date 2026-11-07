import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pytest


@pytest.fixture(scope="session")
def config_path():
    return os.path.join(ROOT_DIR, "config", "vehicle_profiles.json")


@pytest.fixture(scope="session")
def model_path():
    return os.path.join(ROOT_DIR, "models", "yolov8n.pt")
