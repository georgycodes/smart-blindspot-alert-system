"""
Shared video input helper: webcam or a prerecorded file, same interface
either way, used by both the CLI script and the Streamlit dashboard.
"""
from typing import Generator, Union

import cv2


def open_capture(source: Union[int, str], width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def frames(cap: cv2.VideoCapture, mirror: bool) -> Generator:
    """Yields BGR frames until the source runs out or a read fails. `mirror`
    should be True for a live selfie-style webcam, False for a prerecorded
    file (flipping someone else's footage makes no sense)."""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield frame
    finally:
        cap.release()
