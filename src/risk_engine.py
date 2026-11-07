"""
Rule-based, explainable risk engine.

Stage 2 scope: risk is derived from (a) whether a tracked person is inside an
active danger zone and (b) their recent movement direction, smoothed over a
few frames of tracking history. There is no trajectory extrapolation yet
(that is a later stage) -- this only reasons about current state and recent
history, not predicted future position.

Every decision comes with a human-readable reason string built from the
actual inputs; nothing here is invented or guessed by an LLM at runtime.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from trajectory_prediction import PredictedZoneEntry

RISK_SAFE = "SAFE"
RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH = "HIGH"
RISK_CRITICAL = "CRITICAL"

RISK_ORDER = {RISK_SAFE: 0, RISK_LOW: 1, RISK_MODERATE: 2, RISK_HIGH: 3, RISK_CRITICAL: 4}

DIR_APPROACHING = "APPROACHING"
DIR_AWAY = "MOVING AWAY"
DIR_LEFT = "MOVING LEFT"
DIR_RIGHT = "MOVING RIGHT"
DIR_STATIONARY = "STATIONARY"
DIR_CROSSING = "CROSSING"
DIR_UNKNOWN = "UNKNOWN"


@dataclass
class TrackState:
    positions: Deque[Tuple[float, float, float, float]]  # (t, cx_frac, cy_frac, box_h_frac)
    direction: str = DIR_UNKNOWN
    risk: str = RISK_SAFE
    pending_risk: Optional[str] = None  # a lower risk waiting out the hysteresis window
    pending_streak: int = 0
    zone_name: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    predicted_entry: Optional[PredictedZoneEntry] = None


class RiskEngine:
    """
    history_len: how many recent frame samples are kept per tracked ID.
    smoothing_window: how many of the most recent samples are compared to
        decide direction, so a single noisy frame can't flip the label.
    hysteresis_frames: how many consecutive frames a *lower* risk must hold
        before it's accepted -- risk can escalate immediately but only
        de-escalates after holding steady, to avoid flicker between e.g.
        HIGH and MODERATE frame-to-frame.
    move_threshold_frac: minimum frame-fraction change treated as real
        movement rather than detector/tracker jitter.
    """

    def __init__(self, history_len: int = 15, smoothing_window: int = 5, hysteresis_frames: int = 3,
                 move_threshold_frac: float = 0.012):
        self.history_len = history_len
        self.smoothing_window = smoothing_window
        self.hysteresis_frames = hysteresis_frames
        self.move_threshold_frac = move_threshold_frac
        self.tracks: Dict[int, TrackState] = {}

    def _get(self, track_id: int) -> TrackState:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(positions=deque(maxlen=self.history_len))
        return self.tracks[track_id]

    def get_positions(self, track_id: int) -> List[Tuple[float, float, float, float]]:
        state = self.tracks.get(track_id)
        return list(state.positions) if state else []

    def forget_stale(self, seen_ids) -> None:
        for tid in [t for t in self.tracks if t not in seen_ids]:
            del self.tracks[tid]

    def _classify_direction(self, state: TrackState) -> str:
        if len(state.positions) < 2:
            return DIR_UNKNOWN
        window = list(state.positions)[-self.smoothing_window:]
        _, x0, _, h0 = window[0]
        _, x1, _, h1 = window[-1]
        dx = x1 - x0
        dh = h1 - h0  # growing box height fraction ~ getting closer to the camera

        moving_x = abs(dx) > self.move_threshold_frac
        moving_h = abs(dh) > self.move_threshold_frac * 0.5

        if not moving_x and not moving_h:
            return DIR_STATIONARY
        if moving_h and not moving_x:
            return DIR_APPROACHING if dh > 0 else DIR_AWAY
        if moving_x and moving_h:
            return DIR_CROSSING
        if moving_x:
            return DIR_LEFT if dx < 0 else DIR_RIGHT
        return DIR_UNKNOWN

    def _risk_for(self, zone_name: Optional[str], direction: str) -> str:
        if zone_name is None:
            return RISK_SAFE
        if direction == DIR_APPROACHING:
            return RISK_CRITICAL
        if direction in (DIR_STATIONARY, DIR_CROSSING, DIR_UNKNOWN):
            return RISK_HIGH
        if direction in (DIR_LEFT, DIR_RIGHT):
            return RISK_MODERATE
        if direction == DIR_AWAY:
            return RISK_LOW
        return RISK_MODERATE

    def _build_reasons(self, zone_name: Optional[str], direction: str) -> List[str]:
        reasons = []
        if zone_name:
            reasons.append(f"Person is inside the {zone_name} danger zone.")
        else:
            reasons.append("Person is outside all monitored danger zones.")

        if direction == DIR_APPROACHING:
            reasons.append("Getting closer to the camera (bounding box growing) over recent frames.")
        elif direction == DIR_AWAY:
            reasons.append("Moving away from the camera over recent frames.")
        elif direction in (DIR_LEFT, DIR_RIGHT):
            reasons.append(f"Moving sideways ({direction.lower()}) across the frame.")
        elif direction == DIR_CROSSING:
            reasons.append("Moving sideways and toward/away from the camera at the same time (crossing path).")
        elif direction == DIR_STATIONARY:
            reasons.append("Not moving significantly over recent frames.")
        else:
            reasons.append("Not enough tracking history yet to judge movement.")
        return reasons

    def update(self, track_id: int, cx_frac: float, cy_frac: float, box_h_frac: float,
               zone_name: Optional[str], t: float) -> TrackState:
        state = self._get(track_id)
        state.positions.append((t, cx_frac, cy_frac, box_h_frac))
        state.direction = self._classify_direction(state)
        state.zone_name = zone_name
        state.predicted_entry = None
        state.reasons = self._build_reasons(zone_name, state.direction)

        self._set_risk(state, self._risk_for(zone_name, state.direction))
        return state

    def apply_prediction(self, track_id: int, predicted_entry: Optional[PredictedZoneEntry]) -> TrackState:
        """Folds a predictive zone-conflict result (from trajectory_prediction.py)
        into the track's risk/explanation. Only ever escalates -- a predicted
        future conflict never overrides a worse *current* situation, and a
        person already inside a zone doesn't get a redundant prediction for
        the same zone (the caller excludes that case)."""
        state = self._get(track_id)
        state.predicted_entry = predicted_entry
        if predicted_entry is None:
            return state

        if predicted_entry.eta_seconds <= 1.5:
            predicted_risk = RISK_HIGH
        else:
            predicted_risk = RISK_MODERATE

        if RISK_ORDER[predicted_risk] > RISK_ORDER[state.risk]:
            self._set_risk(state, predicted_risk)

        state.reasons.append(
            f"Projected path (simple linear extrapolation) would enter the "
            f"{predicted_entry.zone_name} zone in ~{predicted_entry.eta_seconds}s if this trend continues."
        )
        return state

    def _set_risk(self, state: TrackState, target_risk: str) -> None:
        if target_risk == state.risk:
            # stable at the committed risk -- any pending de-escalation lapses
            state.pending_risk = None
            state.pending_streak = 0
            return

        if RISK_ORDER[target_risk] > RISK_ORDER[state.risk]:
            state.risk = target_risk  # escalate immediately
            state.pending_risk = None
            state.pending_streak = 0
            return

        # target is lower than the committed risk: only commit once it has
        # been requested for `hysteresis_frames` consecutive calls, so a
        # one-frame blip doesn't instantly erase a real CRITICAL/HIGH.
        if state.pending_risk == target_risk:
            state.pending_streak += 1
        else:
            state.pending_risk = target_risk
            state.pending_streak = 1

        if state.pending_streak >= self.hysteresis_frames:
            state.risk = target_risk
            state.pending_risk = None
            state.pending_streak = 0
