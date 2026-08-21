from lewm.safety import h1_safe_action_set_successor_v1 as S


def test_fixture_is_deterministic():
    fixture = S.fixture_payload()
    assert fixture["pass"]
    assert fixture["byte_identical_regeneration"]


def test_stop_requires_three_complete_samples():
    assert S.stopped_tick([.01, .01], [0, 0], [False, False]) is None
    assert S.stopped_tick([.01, .01, .01], [0, 0, 0], [False, False, False]) == 2


def test_predecessor_failure_classification():
    assert S.classify_predecessor(
        {"contact": True, "boundary_contact": False},
        {"qualified_safe_brake": False, "boundary_contact": False},
    ) == "EMERGENCY_BRAKE_INSUFFICIENT"
