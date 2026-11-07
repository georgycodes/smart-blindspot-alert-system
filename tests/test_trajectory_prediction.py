from trajectory_prediction import estimate_zone_entry, predict_path, predict_position
from zone_manager import ZoneManager


def _walk_toward_front_zone(n=10):
    """(t, cx, cy, h) samples for a person moving straight down toward the
    FRONT zone's boundary (y ~ 0.40 at x=0.5), starting outside it."""
    return [(i * 0.1, 0.5, 0.28 + i * 0.015, 0.15 + i * 0.015) for i in range(n)]


def test_predict_position_returns_none_without_enough_history():
    assert predict_position([(0.0, 0.5, 0.5, 0.2)], horizon_s=1.0) is None


def test_predict_position_extrapolates_forward():
    positions = _walk_toward_front_zone()
    predicted = predict_position(positions, horizon_s=1.0)
    assert predicted is not None
    px, py = predicted
    # still moving down and roughly centered
    assert py > positions[-1][2]
    assert 0.4 < px < 0.6


def test_predict_path_returns_ordered_points_toward_the_projection():
    positions = _walk_toward_front_zone()
    path = predict_path(positions, horizon_s=1.0, steps=4)
    assert len(path) == 4
    ys = [p[1] for p in path]
    assert ys == sorted(ys)  # monotonically increasing y (continuing downward)


def test_estimate_zone_entry_predicts_front_before_actual_entry(config_path):
    zm = ZoneManager(config_path)
    # only the first 8 samples -- still outside the zone at this point
    positions = _walk_toward_front_zone()[:8]
    result = estimate_zone_entry(positions, zm, 1280, 720, horizon_s=3.0, current_zone=None)
    assert result is not None
    assert result.zone_name == "FRONT"
    assert result.eta_seconds > 0


def test_estimate_zone_entry_excludes_the_zone_already_occupied(config_path):
    zm = ZoneManager(config_path)
    positions = _walk_toward_front_zone()
    result = estimate_zone_entry(positions, zm, 1280, 720, horizon_s=3.0, current_zone="FRONT")
    assert result is None or result.zone_name != "FRONT"


def test_estimate_zone_entry_returns_none_without_enough_history(config_path):
    zm = ZoneManager(config_path)
    assert estimate_zone_entry([(0.0, 0.5, 0.3, 0.2)], zm, 1280, 720) is None


def test_stationary_person_has_no_predicted_entry(config_path):
    zm = ZoneManager(config_path)
    positions = [(i * 0.1, 0.5, 0.1, 0.2) for i in range(10)]  # not moving, far outside zone
    result = estimate_zone_entry(positions, zm, 1280, 720, horizon_s=3.0)
    assert result is None
