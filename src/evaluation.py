"""
Evaluation tools: detection precision/recall/F1, tracking ID-switch rate,
predictive zone-entry accuracy + warning lead time, and performance (FPS /
processing time / resource usage).

Performance evaluation needs no ground truth and can be run on any video or
live source right now -- see run_performance_eval / the CLI at the bottom.

Detection, tracking, and predictive evaluation all need manually annotated
ground truth (see the annotation schema in each function's docstring and
`tests/fixtures/sample_annotations.json` for a worked example). This module
deliberately does not invent or approximate ground truth -- if you don't
have annotated data yet, these functions simply cannot be run, and nothing
here will print a plausible-looking number in their place.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import cv2

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# -------------------------------------------------------------------------
# Performance evaluation -- no ground truth needed
# -------------------------------------------------------------------------

@dataclass
class PerformanceReport:
    source: str
    frame_count: int
    duration_s: float
    avg_fps: float
    min_fps: float
    max_fps: float
    avg_frame_time_ms: float
    p95_frame_time_ms: float
    avg_cpu_percent: Optional[float]
    peak_memory_mb: Optional[float]


def run_performance_eval(pipeline, source, max_frames: int = 200, mirror: bool = False) -> PerformanceReport:
    """Runs `pipeline` over up to `max_frames` frames of `source` (a webcam
    index or video path), measuring real wall-clock processing time per
    frame plus CPU/memory of this process if `psutil` is available."""
    from video_processor import frames, open_capture

    cap = open_capture(source)
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc:
        proc.cpu_percent()  # first call always returns 0.0; primes the measurement

    frame_times: List[float] = []
    cpu_samples: List[float] = []
    mem_samples: List[float] = []

    t_start = time.time()
    count = 0
    for frame in frames(cap, mirror=mirror):
        t0 = time.time()
        pipeline.process_frame(frame, draw=False)
        frame_times.append(time.time() - t0)
        if proc:
            cpu_samples.append(proc.cpu_percent())
            mem_samples.append(proc.memory_info().rss / (1024 * 1024))
        count += 1
        if count >= max_frames:
            break
    duration = time.time() - t_start

    fps_series = [1.0 / t for t in frame_times if t > 0]
    frame_times_ms = [t * 1000 for t in frame_times]

    return PerformanceReport(
        source=str(source),
        frame_count=count,
        duration_s=round(duration, 2),
        avg_fps=round(sum(fps_series) / len(fps_series), 2) if fps_series else 0.0,
        min_fps=round(min(fps_series), 2) if fps_series else 0.0,
        max_fps=round(max(fps_series), 2) if fps_series else 0.0,
        avg_frame_time_ms=round(sum(frame_times_ms) / len(frame_times_ms), 1) if frame_times_ms else 0.0,
        p95_frame_time_ms=round(_percentile(frame_times_ms, 95), 1) if frame_times_ms else 0.0,
        avg_cpu_percent=round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else None,
        peak_memory_mb=round(max(mem_samples), 1) if mem_samples else None,
    )


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.percentile(values, pct))
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# -------------------------------------------------------------------------
# Detection evaluation -- needs ground-truth boxes
# -------------------------------------------------------------------------

@dataclass
class DetectionReport:
    frames_evaluated: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_detection_eval(model, annotations: dict, conf_threshold: float = 0.35,
                        iou_threshold: float = 0.5) -> DetectionReport:
    """
    annotations schema:
        {"video": "path/to/clip.mp4",
         "frames": {"<frame_index>": [[x1, y1, x2, y2], ...], ...}}
    Ground truth boxes must be hand-annotated (e.g. with CVAT, Label Studio,
    or by hand) -- this function only scores a model against them, never
    generates them.
    """
    video_path = annotations["video"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open annotated video '{video_path}'.")

    gt_frames: Dict[int, list] = {int(k): v for k, v in annotations["frames"].items()}
    max_frame = max(gt_frames) if gt_frames else -1

    tp = fp = fn = 0
    frame_idx = 0
    try:
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in gt_frames:
                gt_boxes = gt_frames[frame_idx]
                results = model.predict(frame, classes=[0], conf=conf_threshold, verbose=False)
                pred_boxes = [b.xyxy[0].tolist() for r in results for b in r.boxes]

                matched = set()
                for pb in pred_boxes:
                    best_iou, best_j = 0.0, -1
                    for j, gb in enumerate(gt_boxes):
                        if j in matched:
                            continue
                        iou = _iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= iou_threshold:
                        tp += 1
                        matched.add(best_j)
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(matched)
            frame_idx += 1
    finally:
        cap.release()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionReport(frames_evaluated=len(gt_frames), true_positives=tp, false_positives=fp,
                            false_negatives=fn, precision=round(precision, 3), recall=round(recall, 3),
                            f1=round(f1, 3))


# -------------------------------------------------------------------------
# Tracking evaluation -- needs ground-truth track IDs, needs a YOLO model
# -------------------------------------------------------------------------

@dataclass
class TrackingReport:
    frames_evaluated: int
    ground_truth_tracks: int
    id_switches: int


def run_tracking_eval(model, annotations: dict, tracker: str = "bytetrack.yaml",
                       conf_threshold: float = 0.35, iou_threshold: float = 0.3) -> TrackingReport:
    """
    annotations schema:
        {"video": "path/to/clip.mp4",
         "frames": {"<frame_index>": [{"gt_id": 1, "bbox": [x1,y1,x2,y2]}, ...], ...}}
    For each ground-truth identity, finds whichever predicted track ID
    overlaps it most (by IoU) in each annotated frame, then counts how many
    times that matched predicted ID changes across consecutive annotated
    frames for the same ground-truth identity -- a standard proxy for
    ID-switch counting (full MOTA/IDF1 scoring is out of scope here).
    """
    video_path = annotations["video"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open annotated video '{video_path}'.")

    gt_frames: Dict[int, list] = {int(k): v for k, v in annotations["frames"].items()}
    max_frame = max(gt_frames) if gt_frames else -1

    last_matched_pred_id: Dict[int, Optional[int]] = {}
    id_switches = 0
    gt_ids_seen = set()

    frame_idx = 0
    try:
        while frame_idx <= max_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in gt_frames:
                results = model.track(frame, classes=[0], conf=conf_threshold, tracker=tracker,
                                       persist=True, verbose=False)
                pred = []
                for r in results:
                    if r.boxes is None or r.boxes.id is None:
                        continue
                    for box, tid in zip(r.boxes, r.boxes.id):
                        pred.append((int(tid), box.xyxy[0].tolist()))

                for gt in gt_frames[frame_idx]:
                    gt_id = gt["gt_id"]
                    gt_ids_seen.add(gt_id)
                    best_iou, best_pred_id = 0.0, None
                    for pid, pbox in pred:
                        iou = _iou(gt["bbox"], pbox)
                        if iou > best_iou:
                            best_iou, best_pred_id = iou, pid
                    if best_iou >= iou_threshold:
                        prev = last_matched_pred_id.get(gt_id)
                        if prev is not None and best_pred_id != prev:
                            id_switches += 1
                        last_matched_pred_id[gt_id] = best_pred_id
            frame_idx += 1
    finally:
        cap.release()

    return TrackingReport(frames_evaluated=len(gt_frames), ground_truth_tracks=len(gt_ids_seen),
                           id_switches=id_switches)


# -------------------------------------------------------------------------
# Predictive-safety evaluation -- needs ground-truth zone-entry events
# -------------------------------------------------------------------------

@dataclass
class PredictiveReport:
    ground_truth_entries: int
    correct_predictions: int
    false_predictions: int
    missed_predictions: int
    avg_lead_time_s: Optional[float]


def run_predictive_eval(pipeline, video_path: str, ground_truth_entries: List[dict],
                         fps_hint: float = 30.0, lead_time_tolerance_s: float = 2.0) -> PredictiveReport:
    """
    ground_truth_entries schema: [{"zone": "FRONT", "entry_frame": 120}, ...]
    -- the frame at which a person is manually confirmed to have actually
    entered that zone, regardless of which internal track ID the tracker
    assigned them (identities aren't required to line up, only the event).

    Runs the pipeline over the whole video, recording every frame where any
    tracked person had a `predicted_zone`/`predicted_eta_s`. A ground-truth
    entry counts as "correctly predicted" if some prediction for that zone
    existed in the window before `entry_frame` (within `lead_time_tolerance_s`
    of actually elapsed time); "missed" if no such prediction existed.
    "False predictions" are predictions for a zone that no ground-truth
    entry in `ground_truth_entries` ever confirms.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video '{video_path}'.")

    predictions_by_zone: List[dict] = []  # [{"frame": i, "zone": ..., "eta_s": ...}]
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process_frame(frame, draw=False)
            for person in result.people:
                if person.predicted_zone:
                    predictions_by_zone.append({
                        "frame": frame_idx, "zone": person.predicted_zone, "eta_s": person.predicted_eta_s,
                    })
            frame_idx += 1
    finally:
        cap.release()

    tolerance_frames = lead_time_tolerance_s * fps_hint
    correct, missed, lead_times = 0, 0, []
    confirmed_prediction_indices = set()

    for gt in ground_truth_entries:
        zone, entry_frame = gt["zone"], gt["entry_frame"]
        candidates = [
            (i, p) for i, p in enumerate(predictions_by_zone)
            if p["zone"] == zone and p["frame"] <= entry_frame
            and (entry_frame - p["frame"]) <= tolerance_frames + (p["eta_s"] or 0) * fps_hint
        ]
        if candidates:
            i, best = max(candidates, key=lambda ip: ip[1]["frame"])
            confirmed_prediction_indices.add(i)
            correct += 1
            lead_times.append((entry_frame - best["frame"]) / fps_hint)
        else:
            missed += 1

    false_predictions = len(predictions_by_zone) - len(confirmed_prediction_indices)

    return PredictiveReport(
        ground_truth_entries=len(ground_truth_entries), correct_predictions=correct,
        false_predictions=max(0, false_predictions), missed_predictions=missed,
        avg_lead_time_s=round(sum(lead_times) / len(lead_times), 2) if lead_times else None,
    )


# -------------------------------------------------------------------------
# Report export
# -------------------------------------------------------------------------

def export_report(report: dict, out_path: str) -> None:
    """Writes the given report dict to `out_path` as JSON (and a sibling
    .md file with the same name), creating parent directories if needed."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.splitext(out_path)[0] + ".md"
    lines = ["# Evaluation report", ""]
    for section, data in report.items():
        lines.append(f"## {section}")
        if isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"- **{k}**: {v}")
        else:
            lines.append(str(data))
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import argparse

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline import GuardianPipeline

    p = argparse.ArgumentParser(
        description="Run the performance evaluation (FPS / processing time / resource usage) -- "
                    "the only metric family here that needs no ground-truth annotations."
    )
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--config", default="config/vehicle_profiles.json")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--out", default="reports/performance_report.json")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = GuardianPipeline(model_path=args.model, config_path=args.config, db_path=None)
    report = run_performance_eval(pipeline, source, max_frames=args.max_frames, mirror=isinstance(source, int))
    pipeline.close()

    print(json.dumps(asdict(report), indent=2))
    export_report({"performance": asdict(report)}, args.out)
    print(f"\nSaved report to {args.out} (and {os.path.splitext(args.out)[0]}.md)")
