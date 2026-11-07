"""
Tests for the evaluation math (IoU matching, precision/recall/F1, ID-switch
counting, predictive lead-time scoring) using small synthetic videos and
fake detector/tracker outputs -- these check the *arithmetic* is correct
without needing real annotated footage, which the project doesn't have yet.
"""
import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from evaluation import (
    _iou, export_report, run_detection_eval, run_performance_eval,
    run_predictive_eval, run_tracking_eval,
)


# -------------------- fakes for the detector/tracker --------------------

class _FakeBox:
    def __init__(self, xyxy, conf=0.9):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [conf]


class _FakeBoxes(list):
    def __init__(self, items, ids=None):
        super().__init__(items)
        self.id = ids


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    def __init__(self, predict_script=None, track_script=None):
        self.predict_script = predict_script or []
        self.track_script = track_script or []
        self._pi = 0
        self._ti = 0

    def predict(self, frame, classes=None, conf=None, verbose=False):
        boxes = self.predict_script[self._pi]
        self._pi += 1
        return [_FakeResult(_FakeBoxes([_FakeBox(b) for b in boxes]))]

    def track(self, frame, classes=None, conf=None, tracker=None, persist=None, verbose=False):
        boxes, ids = self.track_script[self._ti]
        self._ti += 1
        return [_FakeResult(_FakeBoxes([_FakeBox(b) for b in boxes], ids=ids))]


def _make_video(path: str, n_frames: int, size=(320, 240)) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


# -------------------------------- IoU --------------------------------

def test_iou_identical_boxes_is_one():
    box = [10, 10, 50, 50]
    assert _iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert _iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_iou_partial_overlap():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    # intersection 5x5=25, union 100+100-25=175
    assert _iou(a, b) == pytest.approx(25 / 175)


# -------------------------- detection evaluation --------------------------

def test_detection_eval_perfect_match_gives_precision_recall_one(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=2)

    gt_box = [10, 10, 50, 50]
    model = _FakeModel(predict_script=[[gt_box], [gt_box]])
    annotations = {"video": video_path, "frames": {"0": [gt_box], "1": [gt_box]}}

    report = run_detection_eval(model, annotations)
    assert report.true_positives == 2
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


def test_detection_eval_counts_false_positive_and_false_negative(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=1)

    gt_box = [10, 10, 50, 50]
    spurious_box = [200, 200, 220, 220]  # nowhere near gt -> false positive
    model = _FakeModel(predict_script=[[spurious_box]])  # misses gt entirely
    annotations = {"video": video_path, "frames": {"0": [gt_box]}}

    report = run_detection_eval(model, annotations)
    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.false_negatives == 1
    assert report.precision == 0.0
    assert report.recall == 0.0


# -------------------------- tracking evaluation --------------------------

def test_tracking_eval_no_switch_when_id_stays_consistent(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=3)

    box = [10, 10, 50, 50]
    model = _FakeModel(track_script=[
        ([box], [7]), ([box], [7]), ([box], [7]),
    ])
    annotations = {
        "video": video_path,
        "frames": {str(i): [{"gt_id": 1, "bbox": box}] for i in range(3)},
    }
    report = run_tracking_eval(model, annotations)
    assert report.id_switches == 0
    assert report.ground_truth_tracks == 1


def test_tracking_eval_counts_one_switch(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=3)

    box = [10, 10, 50, 50]
    model = _FakeModel(track_script=[
        ([box], [7]), ([box], [7]), ([box], [9]),  # id flips on the 3rd frame
    ])
    annotations = {
        "video": video_path,
        "frames": {str(i): [{"gt_id": 1, "bbox": box}] for i in range(3)},
    }
    report = run_tracking_eval(model, annotations)
    assert report.id_switches == 1


# -------------------------- predictive evaluation --------------------------

class _FakePipeline:
    """Scripts person-level predictions per frame without running real detection."""

    def __init__(self, people_per_frame):
        self.people_per_frame = people_per_frame
        self._i = 0

    def process_frame(self, frame, draw=False):
        people = self.people_per_frame[self._i] if self._i < len(self.people_per_frame) else []
        self._i += 1
        return SimpleNamespace(people=people)


def _person(predicted_zone, eta_s):
    return SimpleNamespace(predicted_zone=predicted_zone, predicted_eta_s=eta_s)


def test_predictive_eval_counts_a_correct_prediction(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=5)

    # a FRONT prediction appears at frame 2 with eta ~1s; actual entry at frame 2+10=12 (fps_hint=10)
    people_per_frame = [[], [], [_person("FRONT", 1.0)], [], []]
    pipeline = _FakePipeline(people_per_frame)
    ground_truth = [{"zone": "FRONT", "entry_frame": 12}]

    report = run_predictive_eval(pipeline, video_path, ground_truth, fps_hint=10.0,
                                  lead_time_tolerance_s=2.0)
    assert report.correct_predictions == 1
    assert report.missed_predictions == 0
    assert report.avg_lead_time_s == pytest.approx(1.0)


def test_predictive_eval_counts_a_missed_entry(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=5)

    pipeline = _FakePipeline([[], [], [], [], []])  # no predictions at all
    ground_truth = [{"zone": "FRONT", "entry_frame": 4}]

    report = run_predictive_eval(pipeline, video_path, ground_truth, fps_hint=10.0)
    assert report.correct_predictions == 0
    assert report.missed_predictions == 1


def test_predictive_eval_counts_a_false_prediction(tmp_path):
    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=5)

    # predicts REAR, but no ground-truth entry ever confirms a REAR entry
    pipeline = _FakePipeline([[], [_person("REAR", 1.0)], [], [], []])
    ground_truth = []

    report = run_predictive_eval(pipeline, video_path, ground_truth, fps_hint=10.0)
    assert report.false_predictions == 1


# -------------------------- performance evaluation (real pipeline) --------------------------

def test_performance_eval_runs_real_pipeline_and_reports_timing(tmp_path, model_path, config_path):
    if not os.path.exists(model_path):
        pytest.skip("models/yolov8n.pt not present -- see README Setup")
    from pipeline import GuardianPipeline

    video_path = str(tmp_path / "clip.mp4")
    _make_video(video_path, n_frames=5)

    pipeline = GuardianPipeline(model_path=model_path, config_path=config_path, db_path=None)
    try:
        report = run_performance_eval(pipeline, video_path, max_frames=5, mirror=False)
    finally:
        pipeline.close()

    assert report.frame_count == 5
    assert report.avg_fps >= 0
    assert report.avg_frame_time_ms >= 0


# -------------------------------- export --------------------------------

def test_export_report_writes_json_and_markdown(tmp_path):
    out_path = str(tmp_path / "report.json")
    export_report({"performance": {"avg_fps": 12.3}}, out_path)

    assert os.path.exists(out_path)
    with open(out_path) as f:
        data = json.load(f)
    assert data["performance"]["avg_fps"] == 12.3

    md_path = str(tmp_path / "report.md")
    assert os.path.exists(md_path)
    assert "avg_fps" in open(md_path).read()
