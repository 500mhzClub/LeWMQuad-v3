from lewm.safety import deployment_valid_strong_braking_mode_v1 as S


def test_binding_fixture_is_deterministic_and_stops_before_physics():
    first = S.fixture_payload()
    second = S.fixture_payload()
    assert first == second
    assert first["pass"]
    assert first["content_digest"] == second["content_digest"]
    assert S.choose_primary_mode(S.frozen_mode_bindings()) is None


def test_no_zero_velocity_alias_is_eligible():
    bindings = {binding.experiment_name: binding for binding in S.frozen_mode_bindings()}
    assert not bindings["ACTIVE_STOP"].eligible_for_fixtures
    assert "zero-velocity" in bindings["ACTIVE_STOP"].exclusion_reason
    assert not bindings["BALANCE_STAND_TRANSITION"].platform_equivalent
    assert not bindings["DAMPING_MODE"].platform_equivalent


def test_unqualified_stopping_envelope_does_not_invent_clearance():
    value = S.stopping_envelope(
        mode_qualified=False,
        planar_speed_m_s=0.5,
        yaw_rate_rad_s=0.3,
        current_command=(0.5, 0.0, 0.3),
        candidate_command=(0.0, 0.0, 0.0),
        stopping_distance_m=None,
        stopping_time_s=None,
        uncertainty_margin_m=0.1,
    )
    assert not value["guard_defined"]
    assert value["required_clearance_m"] is None
    assert value["route_authorisation"].startswith("blocked")


def test_qualified_stopping_envelope_includes_margin():
    value = S.stopping_envelope(
        mode_qualified=True,
        planar_speed_m_s=0.4,
        yaw_rate_rad_s=-0.2,
        current_command=(0.4, 0.0, -0.2),
        candidate_command=(0.0, 0.0, 0.0),
        stopping_distance_m=0.31,
        stopping_time_s=0.62,
        uncertainty_margin_m=0.09,
    )
    assert value["guard_defined"]
    assert abs(value["required_clearance_m"] - 0.40) < 1e-12
