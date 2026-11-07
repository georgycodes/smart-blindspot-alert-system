from zone_manager import ZoneManager


def test_loads_bus_profile_with_one_active_zone(config_path):
    zm = ZoneManager(config_path)
    assert zm.profile_name == "bus"
    assert zm.validated is False
    active = zm.active_zones()
    assert [z.name for z in active] == ["FRONT"]


def test_point_near_bottom_is_inside_front_zone(config_path):
    zm = ZoneManager(config_path)
    zone = zm.zone_for_point(0.5, 0.9, 1280, 720)
    assert zone is not None
    assert zone.name == "FRONT"


def test_point_near_top_is_outside_any_zone(config_path):
    zm = ZoneManager(config_path)
    zone = zm.zone_for_point(0.5, 0.05, 1280, 720)
    assert zone is None


def test_inactive_zones_are_never_matched(config_path):
    zm = ZoneManager(config_path)
    # LEFT zone's own polygon region, but it's inactive -- must not match.
    left_zone_point = (0.1, 0.6)
    zone = zm.zone_for_point(*left_zone_point, 1280, 720)
    assert zone is None or zone.name != "LEFT"


def test_explicit_profile_selection(config_path):
    zm = ZoneManager(config_path, profile="car")
    assert zm.profile_name == "car"


def test_unknown_profile_raises(config_path):
    import pytest
    with pytest.raises(ValueError):
        ZoneManager(config_path, profile="spaceship")
