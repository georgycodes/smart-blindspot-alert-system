from risk_engine import (
    DIR_APPROACHING, DIR_STATIONARY, DIR_UNKNOWN,
    RISK_CRITICAL, RISK_HIGH, RISK_SAFE,
    RiskEngine,
)
from trajectory_prediction import PredictedZoneEntry


def test_first_frame_is_unknown_direction_and_safe():
    engine = RiskEngine()
    state = engine.update(1, 0.5, 0.5, 0.2, None, t=0.0)
    assert state.direction == DIR_UNKNOWN
    assert state.risk == RISK_SAFE


def test_stationary_outside_zone_stays_safe():
    engine = RiskEngine()
    for i in range(6):
        state = engine.update(1, 0.5, 0.5, 0.2, None, t=i * 0.1)
    assert state.direction == DIR_STATIONARY
    assert state.risk == RISK_SAFE


def test_approaching_inside_zone_escalates_to_critical():
    engine = RiskEngine()
    for i in range(8):
        state = engine.update(1, 0.5, 0.5, 0.15 + i * 0.03, "FRONT", t=i * 0.1)
    assert state.direction == DIR_APPROACHING
    assert state.risk == RISK_CRITICAL
    assert any("FRONT" in r for r in state.reasons)


def test_risk_escalates_immediately_but_deescalates_with_hysteresis():
    engine = RiskEngine(hysteresis_frames=3)
    for i in range(8):
        engine.update(1, 0.5, 0.5, 0.15 + i * 0.03, "FRONT", t=i * 0.1)
    state = engine.tracks[1]
    assert state.risk == RISK_CRITICAL

    # leaving the zone: risk must not drop on the very next frame (hysteresis)
    state = engine.update(1, 0.5, 0.5, 0.39, None, t=0.9)
    assert state.risk == RISK_CRITICAL


def test_forget_stale_removes_untracked_ids():
    engine = RiskEngine()
    engine.update(1, 0.5, 0.5, 0.2, None, t=0.0)
    engine.update(2, 0.3, 0.3, 0.2, None, t=0.0)
    engine.forget_stale(seen_ids={1})
    assert 1 in engine.tracks
    assert 2 not in engine.tracks


def test_apply_prediction_escalates_risk_before_actual_entry():
    engine = RiskEngine()
    engine.update(1, 0.5, 0.3, 0.2, None, t=0.0)
    predicted = PredictedZoneEntry(zone_name="FRONT", eta_seconds=1.0)
    state = engine.apply_prediction(1, predicted)
    assert state.risk == RISK_HIGH
    assert state.predicted_entry is predicted
    assert any("FRONT" in r for r in state.reasons)


def test_apply_prediction_never_downgrades_a_worse_current_risk():
    engine = RiskEngine()
    for i in range(8):
        engine.update(1, 0.5, 0.5, 0.15 + i * 0.03, "FRONT", t=i * 0.1)
    assert engine.tracks[1].risk == RISK_CRITICAL

    # a (hypothetical) weak prediction must not pull a CRITICAL track down
    weak_prediction = PredictedZoneEntry(zone_name="REAR", eta_seconds=2.9)
    state = engine.apply_prediction(1, weak_prediction)
    assert state.risk == RISK_CRITICAL
