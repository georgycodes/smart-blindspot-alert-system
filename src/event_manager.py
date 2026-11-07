"""
Safety-risk event state machine + SQLite persistence.

A "safety-risk event" (the brief's term -- these are rule-based flags, not
verified collisions or confirmed near-misses) is opened for a tracked person
the moment their risk reaches HIGH or CRITICAL, and closed once their risk
drops back below that for a short grace period (so a one-frame flicker
doesn't split one real encounter into several events) or the track is lost.
One continuous encounter = one row, updated in place while it's ongoing --
never a new row per frame.
"""
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

TRIGGER_RISKS = {"HIGH", "CRITICAL"}
DEFAULT_GRACE_SECONDS = 1.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracking_id INTEGER NOT NULL,
    vehicle_profile TEXT NOT NULL,
    zone TEXT,
    risk_level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    direction TEXT,
    explanation TEXT,
    source_video TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    created_at TEXT NOT NULL,
    start_frame INTEGER,
    end_frame INTEGER
);
"""


@dataclass
class _OpenEvent:
    row_id: int
    last_seen_monotonic: float
    peak_risk: str


class EventManager:
    def __init__(self, db_path: str, vehicle_profile: str, source_video: Optional[str] = None,
                 grace_seconds: float = DEFAULT_GRACE_SECONDS):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.vehicle_profile = vehicle_profile
        self.source_video = source_video or "webcam"
        self.grace_seconds = grace_seconds
        self._open: Dict[int, _OpenEvent] = {}

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Migrates an events.db created before start_frame/end_frame existed."""
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "start_frame" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN start_frame INTEGER")
        if "end_frame" not in cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN end_frame INTEGER")
        self.conn.commit()

    def update(self, track_id: int, risk_level: str, zone: Optional[str], direction: str,
               reasons: List[str], now_monotonic: Optional[float] = None,
               frame_index: Optional[int] = None) -> None:
        now_monotonic = now_monotonic if now_monotonic is not None else time.monotonic()
        triggered = risk_level in TRIGGER_RISKS
        open_event = self._open.get(track_id)

        if triggered:
            explanation = " ".join(reasons)
            if open_event is None:
                row_id = self._insert(track_id, zone, risk_level, direction, explanation, frame_index)
                self._open[track_id] = _OpenEvent(row_id=row_id, last_seen_monotonic=now_monotonic,
                                                   peak_risk=risk_level)
            else:
                open_event.last_seen_monotonic = now_monotonic
                peak = risk_level if _risk_rank(risk_level) > _risk_rank(open_event.peak_risk) else open_event.peak_risk
                open_event.peak_risk = peak
                self._extend(open_event.row_id, peak, zone, direction, explanation, frame_index)
        elif open_event is not None and now_monotonic - open_event.last_seen_monotonic > self.grace_seconds:
            self._close(open_event.row_id)
            del self._open[track_id]

    def close_track(self, track_id: int) -> None:
        open_event = self._open.pop(track_id, None)
        if open_event is not None:
            self._close(open_event.row_id)

    def active_track_ids(self) -> List[int]:
        return list(self._open.keys())

    def close_all(self) -> None:
        for track_id in list(self._open.keys()):
            self.close_track(track_id)

    def _insert(self, track_id: int, zone: Optional[str], risk_level: str, direction: str,
                explanation: str, frame_index: Optional[int]) -> int:
        now = datetime.now().isoformat(timespec="seconds")
        event_type = "ZONE_OCCUPANCY" if zone else "PREDICTED_CONFLICT"
        cur = self.conn.execute(
            "INSERT INTO events (tracking_id, vehicle_profile, zone, risk_level, event_type, "
            "direction, explanation, source_video, start_time, end_time, created_at, "
            "start_frame, end_frame) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)",
            (track_id, self.vehicle_profile, zone, risk_level, event_type, direction,
             explanation, self.source_video, now, now, frame_index, frame_index),
        )
        self.conn.commit()
        return cur.lastrowid

    def _extend(self, row_id: int, peak_risk: str, zone: Optional[str], direction: str,
                explanation: str, frame_index: Optional[int]) -> None:
        params = {"risk_level": peak_risk, "direction": direction, "explanation": explanation,
                  "id": row_id}
        set_clause = "risk_level = :risk_level, direction = :direction, explanation = :explanation"
        if zone:
            # once actually inside a zone, that supersedes an earlier predicted-only event
            set_clause += ", zone = :zone, event_type = 'ZONE_OCCUPANCY'"
            params["zone"] = zone
        if frame_index is not None:
            set_clause += ", end_frame = :end_frame"
            params["end_frame"] = frame_index
        self.conn.execute(f"UPDATE events SET {set_clause} WHERE id = :id", params)
        self.conn.commit()

    def _close(self, row_id: int) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute("UPDATE events SET end_time = ? WHERE id = ?", (now, row_id))
        self.conn.commit()

    def close(self) -> None:
        self.close_all()
        self.conn.close()


def _risk_rank(risk: str) -> int:
    order = {"SAFE": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
    return order.get(risk, 0)


# -------- read-side helpers for the dashboard / analytics --------

def query_events(db_path: str, risk_level: Optional[str] = None, zone: Optional[str] = None,
                  limit: int = 500) -> List[sqlite3.Row]:
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    clauses, params = [], []
    if risk_level:
        clauses.append("risk_level = ?")
        params.append(risk_level)
    if zone:
        clauses.append("zone = ?")
        params.append(zone)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM events {where} ORDER BY created_at DESC LIMIT ?", (*params, limit)
    ).fetchall()
    conn.close()
    return rows


def summary_stats(db_path: str) -> dict:
    if not os.path.exists(db_path):
        return {"total": 0, "critical": 0, "high": 0, "by_zone": {}, "by_risk": {}, "avg_duration_s": None}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events").fetchall()
    conn.close()

    total = len(rows)
    critical = sum(1 for r in rows if r["risk_level"] == "CRITICAL")
    high = sum(1 for r in rows if r["risk_level"] == "HIGH")
    by_zone: Dict[str, int] = {}
    by_risk: Dict[str, int] = {}
    durations = []
    for r in rows:
        by_zone[r["zone"] or "UNKNOWN"] = by_zone.get(r["zone"] or "UNKNOWN", 0) + 1
        by_risk[r["risk_level"]] = by_risk.get(r["risk_level"], 0) + 1
        if r["end_time"]:
            try:
                start = datetime.fromisoformat(r["start_time"])
                end = datetime.fromisoformat(r["end_time"])
                durations.append((end - start).total_seconds())
            except ValueError:
                pass
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None

    return {
        "total": total, "critical": critical, "high": high,
        "by_zone": by_zone, "by_risk": by_risk, "avg_duration_s": avg_duration,
    }
