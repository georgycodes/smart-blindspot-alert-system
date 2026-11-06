"""
Short-term trajectory prediction.

Uses a simple linear extrapolation over the same recent-position history the
risk engine already keeps per tracked person (a handful of (t, cx_frac,
cy_frac) samples). This is intentionally simple -- a smoothed average
velocity projected forward -- not a learned motion model. Predictions are
explicitly labelled estimates, and the module refuses to guess (returns
None / "insufficient history") when there isn't enough history to support
one, rather than inventing a confident-looking number.
"""
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from zone_manager import ZoneManager

Sample = Tuple[float, float, float, float]  # (t, cx_frac, cy_frac, box_h_frac)


@dataclass
class PredictedZoneEntry:
    zone_name: str
    eta_seconds: float


def _velocity(positions: List[Sample], smoothing_window: int) -> Optional[Tuple[float, float, float]]:
    """Average (vx, vy) in frame-fractions/second over the smoothing window, plus dt covered."""
    if len(positions) < 2:
        return None
    window = positions[-smoothing_window:]
    t0, x0, y0, _ = window[0]
    t1, x1, y1, _ = window[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    return (x1 - x0) / dt, (y1 - y0) / dt, dt


def predict_position(positions: List[Sample], horizon_s: float, smoothing_window: int = 5,
                      min_samples: int = 3) -> Optional[Tuple[float, float]]:
    """Predicted (cx_frac, cy_frac) `horizon_s` seconds from now, or None if there
    isn't enough history to support a prediction."""
    if len(positions) < min_samples:
        return None
    vel = _velocity(positions, smoothing_window)
    if vel is None:
        return None
    vx, vy, _ = vel
    _, cx, cy, _ = positions[-1]
    return cx + vx * horizon_s, cy + vy * horizon_s


def predict_path(positions: List[Sample], horizon_s: float, steps: int = 6,
                  smoothing_window: int = 5, min_samples: int = 3) -> List[Tuple[float, float]]:
    """A short list of (cx_frac, cy_frac) points from now out to `horizon_s`,
    for drawing a dotted predicted-path line. Empty if there's not enough history."""
    if len(positions) < min_samples:
        return []
    vel = _velocity(positions, smoothing_window)
    if vel is None:
        return []
    vx, vy, _ = vel
    _, cx, cy, _ = positions[-1]
    return [(cx + vx * horizon_s * (i / steps), cy + vy * horizon_s * (i / steps))
            for i in range(1, steps + 1)]


def estimate_zone_entry(positions: List[Sample], zones: ZoneManager, frame_w: int, frame_h: int,
                         horizon_s: float = 3.0, step_s: float = 0.2,
                         smoothing_window: int = 5, min_samples: int = 3,
                         current_zone: Optional[str] = None) -> Optional[PredictedZoneEntry]:
    """Walks the linear projection forward in `step_s` increments up to
    `horizon_s` and returns the first active zone the projected point enters,
    with the estimated seconds-until-entry. Returns None if there's not
    enough history (caller should show UNKNOWN) or if no entry is predicted
    within the horizon. Does not re-report a zone the person is already in.
    """
    if len(positions) < min_samples:
        return None
    vel = _velocity(positions, smoothing_window)
    if vel is None:
        return None
    vx, vy, _ = vel
    _, cx, cy, _ = positions[-1]

    t = step_s
    while t <= horizon_s:
        px, py = cx + vx * t, cy + vy * t
        zone = zones.zone_for_point(px, py, frame_w, frame_h)
        if zone is not None and zone.name != current_zone:
            return PredictedZoneEntry(zone_name=zone.name, eta_seconds=round(t, 1))
        t += step_s
    return None
