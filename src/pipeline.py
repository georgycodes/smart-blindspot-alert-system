"""
GuardianPipeline -- the one place that wires detection + tracking + zones +
risk + trajectory prediction + event logging together. Both the CLI script
(live_demo.py) and the Streamlit dashboard call this same class, so there is
exactly one implementation of the core logic rather than two copies that can
drift apart.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

from event_manager import EventManager
from risk_engine import RISK_CRITICAL, RISK_HIGH, RISK_ORDER, RISK_SAFE, RiskEngine
from trajectory_prediction import estimate_zone_entry, predict_path
from zone_manager import ZoneManager

RISK_COLORS = {
    RISK_SAFE: (0, 255, 0),
    "LOW": (0, 255, 0),
    "MODERATE": (0, 255, 255),
    RISK_HIGH: (0, 165, 255),
    RISK_CRITICAL: (0, 0, 255),
}


@dataclass
class PersonInfo:
    track_id: int
    bbox: tuple
    confidence: float
    zone: Optional[str]
    direction: str
    risk: str
    reasons: List[str]
    predicted_zone: Optional[str]
    predicted_eta_s: Optional[float]


@dataclass
class FrameResult:
    annotated: "cv2.Mat"
    people: List[PersonInfo] = field(default_factory=list)
    overall_risk: str = RISK_SAFE
    people_in_zone: int = 0
    fps: float = 0.0


class GuardianPipeline:
    def __init__(self, model_path: str, config_path: str, profile: Optional[str] = None,
                 conf_threshold: float = 0.35, tracker: str = "bytetrack.yaml",
                 db_path: Optional[str] = "data/events/events.db", source_label: str = "webcam",
                 prediction_horizon_s: float = 3.0):
        self.model = YOLO(model_path)
        self.zones = ZoneManager(config_path, profile=profile)
        self.risk_engine = RiskEngine()
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self.prediction_horizon_s = prediction_horizon_s
        self._prev_time = 0.0
        self.frame_index = -1  # incremented to 0 on the first process_frame() call

        self.event_manager = (
            EventManager(db_path, vehicle_profile=self.zones.profile_name, source_video=source_label)
            if db_path else None
        )

    def process_frame(self, frame, draw: bool = True) -> FrameResult:
        self.frame_index += 1
        frame_h, frame_w = frame.shape[:2]
        now = time.time()
        fps = 1.0 / (now - self._prev_time) if self._prev_time else 0.0
        self._prev_time = now

        results = self.model.track(frame, classes=[0], conf=self.conf_threshold,
                                    tracker=self.tracker, persist=True, verbose=False)

        annotated = frame
        if draw:
            self.zones.draw(annotated, frame_w, frame_h)

        people: List[PersonInfo] = []
        overall_risk = RISK_SAFE
        people_in_zone = 0
        seen_ids = set()

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

                zone = self.zones.zone_for_point(cx_frac, cy_frac, frame_w, frame_h)
                zone_name = zone.name if zone else None

                state = self.risk_engine.update(tid, cx_frac, cy_frac, box_h_frac, zone_name, now)

                predicted = estimate_zone_entry(
                    self.risk_engine.get_positions(tid), self.zones, frame_w, frame_h,
                    horizon_s=self.prediction_horizon_s, current_zone=zone_name,
                )
                self.risk_engine.apply_prediction(tid, predicted)

                if zone_name:
                    people_in_zone += 1
                if RISK_ORDER[state.risk] > RISK_ORDER[overall_risk]:
                    overall_risk = state.risk

                if self.event_manager:
                    self.event_manager.update(tid, state.risk, zone_name, state.direction, state.reasons,
                                               frame_index=self.frame_index,
                                               predicted_zone=predicted.zone_name if predicted else None)

                people.append(PersonInfo(
                    track_id=tid, bbox=(x1, y1, x2, y2), confidence=confidence,
                    zone=zone_name, direction=state.direction, risk=state.risk,
                    reasons=list(state.reasons),
                    predicted_zone=predicted.zone_name if predicted else None,
                    predicted_eta_s=predicted.eta_seconds if predicted else None,
                ))

                if draw:
                    self._draw_person(annotated, tid, (x1, y1, x2, y2), state, predicted,
                                       frame_w, frame_h)

        self.risk_engine.forget_stale(seen_ids)
        if self.event_manager:
            for tid in self.event_manager.active_track_ids():
                if tid not in seen_ids:
                    self.event_manager.close_track(tid)

        if draw:
            self._draw_hud(annotated, fps, people_in_zone, overall_risk)

        return FrameResult(annotated=annotated, people=people, overall_risk=overall_risk,
                            people_in_zone=people_in_zone, fps=fps)

    def _draw_person(self, frame, tid, bbox, state, predicted, frame_w, frame_h) -> None:
        x1, y1, x2, y2 = bbox
        color = RISK_COLORS.get(state.risk, (255, 255, 255))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"PERSON #{tid:02d}  {state.direction}  {state.risk}"
        cv2.putText(frame, label, (int(x1), max(15, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        reason = state.reasons[-1] if state.reasons else ""
        cv2.putText(frame, reason, (int(x1), min(frame_h - 5, int(y2) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        path = predict_path(self.risk_engine.get_positions(tid), self.prediction_horizon_s)
        for i, (px_frac, py_frac) in enumerate(path):
            if i % 2 == 0:  # dotted
                cv2.circle(frame, (int(px_frac * frame_w), int(py_frac * frame_h)), 3, color, -1)

    def _draw_hud(self, frame, fps, people_in_zone, overall_risk) -> None:
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PEOPLE IN DANGER ZONE: {people_in_zone}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"OVERALL RISK: {overall_risk}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RISK_COLORS.get(overall_risk, (255, 255, 255)), 2)

    def close(self) -> None:
        if self.event_manager:
            self.event_manager.close()
