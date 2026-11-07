"""
Integration test: the real GuardianPipeline (actual YOLO model load, actual
ByteTrack config, actual zone/risk/event wiring) processing real frames.
Slower than the unit tests (loads the YOLO model) but it's the only thing
that proves all the pieces actually fit together end to end.
"""
import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pipeline(model_path, config_path, tmp_path_factory):
    if not os.path.exists(model_path):
        pytest.skip("models/yolov8n.pt not present -- see README Setup")
    from pipeline import GuardianPipeline

    db_path = str(tmp_path_factory.mktemp("events") / "events.db")
    p = GuardianPipeline(model_path=model_path, config_path=config_path, db_path=db_path,
                          source_label="integration-test")
    yield p
    p.close()


def test_pipeline_processes_a_blank_frame_without_crashing(pipeline):
    frame = np.full((720, 1280, 3), 20, dtype=np.uint8)
    result = pipeline.process_frame(frame, draw=True)

    assert result.annotated.shape == frame.shape
    assert result.people == []  # nothing in a blank frame
    assert result.overall_risk == "SAFE"
    assert result.people_in_zone == 0


def test_pipeline_runs_several_frames_in_a_row(pipeline):
    for _ in range(3):
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, draw=False)
        assert result.fps >= 0
