import numpy as np

from lewm.safety import genesis_narrowphase_reconciliation_v1 as R


def test_fixture_passes_and_is_deterministic():
    fixture = R.fixture_payload()
    assert fixture["pass"]
    assert fixture["byte_identical_regeneration"]


def test_feasibility_excludes_no_safe_states_from_mobility_denominators():
    rows = [
        {"state_id": "a", "family": "f", "candidate_index": 0, "hard_contact": False, "p_d": .2, "p_theta": 0, "kinematic": np.array([0, 0, 0, 0, .2, 0])},
        {"state_id": "a", "family": "f", "candidate_index": 1, "hard_contact": True, "p_d": .3, "p_theta": 0, "kinematic": np.array([0, 0, 0, 0, .3, 0])},
        {"state_id": "b", "family": "f", "candidate_index": 0, "hard_contact": True, "p_d": .2, "p_theta": 0, "kinematic": np.array([0, 0, 0, 0, .2, 0])},
        {"state_id": "b", "family": "f", "candidate_index": 1, "hard_contact": True, "p_d": .1, "p_theta": 0, "kinematic": np.array([0, 0, 0, 0, .1, 0])},
    ]
    metric = R.feasibility_metrics(rows, np.asarray([True, False, False, False]))
    assert metric["safe_candidate_available_states"] == 1
    assert metric["states_retaining_contact_negative"] == 1
    assert metric["false_abstentions"] == 0
    assert metric["correct_abstentions_no_safe"] == 1


def test_no_safe_causal_classes_are_frozen():
    assert R.classify_no_safe_state(
        boundary_contact=False,
        first_contact_steps=[9] * 12,
        trajectory_divergence_step=3,
        avoiding_response_step=4,
        candidate_effect_evidence=True,
    ) == "CANDIDATE_BANK_SAFETY_COVERAGE_FAILURE"
