import time

from event_manager import EventManager, query_events, summary_stats, zone_label


def test_continuous_high_risk_creates_one_event_not_one_per_frame(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")

    now = time.monotonic()
    for i in range(10):
        em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["test reason"], now_monotonic=now + i * 0.05)
    em.close_all()

    rows = query_events(db_path)
    assert len(rows) == 1
    assert rows[0]["tracking_id"] == 1
    assert rows[0]["risk_level"] == "CRITICAL"
    assert rows[0]["event_type"] == "ZONE_OCCUPANCY"


def test_risk_dropping_below_trigger_closes_the_event(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test", grace_seconds=0.1)

    now = time.monotonic()
    em.update(1, "HIGH", "FRONT", "STATIONARY", ["reason"], now_monotonic=now)
    em.update(1, "SAFE", None, "MOVING AWAY", ["reason"], now_monotonic=now + 0.5)  # past grace period

    rows = query_events(db_path)
    assert len(rows) == 1
    assert rows[0]["end_time"] is not None


def test_two_different_tracks_create_two_events(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")

    now = time.monotonic()
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["a"], now_monotonic=now)
    em.update(2, "HIGH", "FRONT", "STATIONARY", ["b"], now_monotonic=now)
    em.close_all()

    rows = query_events(db_path)
    assert len(rows) == 2
    assert {r["tracking_id"] for r in rows} == {1, 2}


def test_safe_risk_never_opens_an_event(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")
    em.update(1, "SAFE", None, "STATIONARY", ["reason"], now_monotonic=time.monotonic())
    assert query_events(db_path) == []


def test_close_track_closes_an_open_event_immediately(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["reason"], now_monotonic=time.monotonic())
    em.close_track(1)
    rows = query_events(db_path)
    assert len(rows) == 1
    assert rows[0]["end_time"] is not None
    assert em.active_track_ids() == []


def test_summary_stats_on_empty_db_is_honestly_empty(tmp_path):
    db_path = str(tmp_path / "does_not_exist.db")
    stats = summary_stats(db_path)
    assert stats["total"] == 0
    assert stats["critical"] == 0
    assert stats["by_zone"] == {}


def test_predicted_only_event_stores_predicted_zone_not_unknown(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")

    # risk escalated purely from a predicted future zone entry -- not actually
    # inside any zone yet, so zone=None but predicted_zone is known
    em.update(1, "HIGH", None, "APPROACHING", ["projected to enter FRONT"],
              now_monotonic=time.monotonic(), predicted_zone="FRONT")
    em.close_all()

    rows = query_events(db_path)
    assert len(rows) == 1
    assert rows[0]["zone"] is None
    assert rows[0]["predicted_zone"] == "FRONT"
    assert zone_label(rows[0]) == "FRONT (predicted)"


def test_zone_label_falls_back_to_unknown_when_nothing_known(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")
    em.update(1, "HIGH", None, "STATIONARY", ["reason"], now_monotonic=time.monotonic())
    em.close_all()

    rows = query_events(db_path)
    assert zone_label(rows[0]) == "UNKNOWN"


def test_actual_zone_entry_supersedes_an_earlier_prediction_in_summary(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")
    now = time.monotonic()

    em.update(1, "HIGH", None, "APPROACHING", ["predicted"], now_monotonic=now, predicted_zone="FRONT")
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["inside now"], now_monotonic=now + 0.1)
    em.close_all()

    rows = query_events(db_path)
    assert len(rows) == 1
    assert rows[0]["zone"] == "FRONT"
    assert zone_label(rows[0]) == "FRONT"

    stats = summary_stats(db_path)
    assert stats["by_zone"] == {"FRONT": 1}


def test_start_and_end_frame_are_tracked_for_replay(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="clip.mp4")

    now = time.monotonic()
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["a"], now_monotonic=now, frame_index=50)
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["a"], now_monotonic=now + 0.1, frame_index=51)
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["a"], now_monotonic=now + 0.2, frame_index=52)
    em.close_all()

    rows = query_events(db_path)
    assert rows[0]["start_frame"] == 50
    assert rows[0]["end_frame"] == 52


def test_summary_stats_counts_by_risk_and_zone(tmp_path):
    db_path = str(tmp_path / "events.db")
    em = EventManager(db_path, vehicle_profile="bus", source_video="unit-test")
    now = time.monotonic()
    em.update(1, "CRITICAL", "FRONT", "APPROACHING", ["a"], now_monotonic=now)
    em.update(2, "HIGH", "FRONT", "STATIONARY", ["b"], now_monotonic=now)
    em.close_all()

    stats = summary_stats(db_path)
    assert stats["total"] == 2
    assert stats["critical"] == 1
    assert stats["high"] == 1
    assert stats["by_zone"]["FRONT"] == 2
