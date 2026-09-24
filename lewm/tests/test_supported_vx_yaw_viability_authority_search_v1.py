from lewm.safety import supported_vx_yaw_viability_authority_search_v1 as subject


def test_grid_is_capped_mirrored_and_deterministic():
    first = subject.requested_grid()
    assert first == subject.requested_grid()
    assert len(first) == 21
    assert subject.fixture_payload()["pass"]


def test_dedup_separates_historical_duplicates():
    rows = [
        {"search_index": 0, "applied_vx_vy_wz": [0.0, 0.0, 0.0]},
        {"search_index": 1, "applied_vx_vy_wz": [0.0, 0.0, 0.0]},
        {"search_index": 2, "applied_vx_vy_wz": [-0.1, 0.0, 0.0]},
    ]
    historical = [{"candidate_index": 11, "target_command": [0.0, 0.0, 0.0]}]
    result = subject.deduplicate_applied(rows, historical)
    assert result["unique_applied_count"] == 2
    assert result["duplicates_within_grid"] == 1
    assert result["duplicates_of_historical"] == 1
    assert result["genuinely_new_applied"] == 1


def test_residual_classes_prioritize_simple_supported_recovery():
    rows = [
        {"family": "MIRRORED_REVERSE_ARC_RETREAT", "safe_prefix": True,
         "viability_admissible": True},
        {"family": "PURE_REVERSE_RETREAT", "safe_prefix": True,
         "viability_admissible": True},
    ]
    assert subject.residual_classification(rows) == "PURE_REVERSE_RECOVERS_VIABILITY"


def test_contact_inside_first_policy_interval_is_before_control_authority():
    rows = [
        {"family": "PURE_REVERSE_RETREAT", "safe_prefix": False,
         "viability_admissible": False, "outcome": {"first_contact_step": 4}},
        {"family": "MIRRORED_IN_PLACE_ESCAPE_TURN", "safe_prefix": False,
         "viability_admissible": False, "outcome": {"first_contact_step": 8}},
    ]
    assert subject.residual_classification(rows) == "CONTACT_BEFORE_SUPPORTED_CONTROL_AUTHORITY"
