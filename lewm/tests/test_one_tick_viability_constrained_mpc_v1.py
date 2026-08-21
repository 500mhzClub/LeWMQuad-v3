from lewm.safety import one_tick_viability_constrained_mpc_v1 as V


def test_fixture_is_deterministic_and_passes():
    first = V.fixture_payload()
    assert first == V.fixture_payload()
    assert first["pass"]


def test_route_order_uses_distance_tie_then_heading():
    rows = [
        {"candidate_index": 0, "d": .20, "h": .01},
        {"candidate_index": 1, "d": .18, "h": .20},
        {"candidate_index": 2, "d": .10, "h": .50},
    ]
    assert V.route_order(rows, "d", "h") == [1, 0, 2]


def test_safe_prefix_without_viable_successor_is_distinct():
    rows = [{"safe_prefix": True, "viable": False, "immediate_progress_m": 1.0}]
    assert V.state_classification(
        rows, pre_existing=False, contact_before_authority=False
    ) == "SAFE_PREFIX_ONLY_NO_VIABLE_SUCCESSOR"


def test_contact_before_authority_is_distinct():
    rows = [{"safe_prefix": False, "viable": False, "immediate_progress_m": 0.0}]
    assert V.state_classification(
        rows, pre_existing=False, contact_before_authority=True
    ) == "CONTACT_BEFORE_CONTROL_AUTHORITY"
