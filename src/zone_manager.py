"""
Polygon-based blind-spot danger zones, loaded from a vehicle profile config.

Replaces the Semester 6 script's single full-frame rectangular ROI. A vehicle
profile (e.g. "bus") can define multiple named zones (FRONT/LEFT/RIGHT/REAR),
each meant to correspond to one physical camera on the real vehicle. Only the
zones marked "active" in the config are drawn/checked -- with a single
webcam you can only usefully activate one at a time.
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Zone:
    name: str
    polygon_frac: List[Tuple[float, float]]
    color: Tuple[int, int, int]
    active: bool = True
    note: str = ""

    def polygon_px(self, frame_w: int, frame_h: int) -> np.ndarray:
        pts = [(int(x * frame_w), int(y * frame_h)) for x, y in self.polygon_frac]
        return np.array(pts, dtype=np.int32)

    def contains(self, cx_frac: float, cy_frac: float, frame_w: int, frame_h: int) -> bool:
        poly = self.polygon_px(frame_w, frame_h)
        point = (cx_frac * frame_w, cy_frac * frame_h)
        return cv2.pointPolygonTest(poly, point, False) >= 0


class ZoneManager:
    def __init__(self, config_path: str, profile: Optional[str] = None):
        with open(config_path, "r") as f:
            data = json.load(f)

        self.profile_name = profile or data.get("active_profile", "bus")
        if self.profile_name not in data["profiles"]:
            raise ValueError(f"Vehicle profile '{self.profile_name}' not found in {config_path}")

        profile_data = data["profiles"][self.profile_name]
        self.description = profile_data.get("description", "")
        self.validated = profile_data.get("validated", False)
        self.zones: List[Zone] = [
            Zone(
                name=z["name"],
                polygon_frac=[tuple(p) for p in z["polygon"]],
                color=tuple(z.get("color", [255, 0, 0])),
                active=z.get("active", False),
                note=z.get("note", ""),
            )
            for z in profile_data.get("zones", [])
        ]

    def active_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.active]

    def zone_for_point(self, cx_frac: float, cy_frac: float, frame_w: int, frame_h: int) -> Optional[Zone]:
        for zone in self.active_zones():
            if zone.contains(cx_frac, cy_frac, frame_w, frame_h):
                return zone
        return None

    def draw(self, frame, frame_w: int, frame_h: int) -> None:
        active = self.active_zones()
        if not active:
            return
        overlay = frame.copy()
        for zone in active:
            cv2.fillPoly(overlay, [zone.polygon_px(frame_w, frame_h)], zone.color)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
        for zone in active:
            poly = zone.polygon_px(frame_w, frame_h)
            cv2.polylines(frame, [poly], True, zone.color, 2)
            label_pt = tuple(poly[-1])
            cv2.putText(frame, f"{zone.name} ZONE", (int(label_pt[0]), max(20, int(label_pt[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, zone.color, 2)
