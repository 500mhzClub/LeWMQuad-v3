from lewm.safety import multi_cycle_viability_envelope_v1 as subject


def test_fixture_passes_and_is_deterministic():
    first = subject.fixture_payload()
    assert first["pass"]
    assert first == subject.fixture_payload()


def test_route_tie_uses_heading_before_index():
    rows = [
        {"candidate_index": 0, "h3_progress_m": 0.20, "h3_heading_improvement_rad": 0.0},
        {"candidate_index": 1, "h3_progress_m": 0.21, "h3_heading_improvement_rad": 0.1},
    ]
    assert subject.route_order(rows) == [1, 0]


def test_stable_depth_requires_three_consecutive_boundaries():
    rows = [
        {"depth": 1, "viability_admissible_count": 1},
        {"depth": 2, "viability_admissible_count": 0},
        {"depth": 3, "viability_admissible_count": 2},
        {"depth": 4, "viability_admissible_count": 1},
        {"depth": 5, "viability_admissible_count": 3},
    ]
    assert subject.stable_predecessor_depth(rows) == 5


def test_persistent_failure_requires_full_ten_depths():
    short = [{"depth": depth, "viability_admissible_count": 0} for depth in range(1, 10)]
    full = short + [{"depth": 10, "viability_admissible_count": 0}]
    assert subject.intervention_class(short) == "UNRESOLVED"
    assert subject.intervention_class(full) == "PERSISTENT_CANDIDATE_BANK_VIABILITY_FAILURE"
