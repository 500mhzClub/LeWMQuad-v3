import numpy as np

from scripts import run_development_micro_viability_model_screen_v1 as subject


def candidate(index, *, contact=False, n_safe=2, progress=0.1):
    return {"action_index": index, "contact": contact, "n_safe": n_safe,
            "h3_progress_m": progress if index < 12 else None,
            "h3_heading_improvement_rad": 0.0 if index < 12 else None,
            "decision_progress_m": progress, "candidate": f"a{index}"}


def test_state_ontology_separates_oracle_nonviable():
    state = {"state_id": "s", "family": "large_enclosed_maze",
             "candidates": [candidate(i, contact=i < 7, n_safe=0) for i in range(14)]}
    result = subject.decision_metrics([state], np.ones((1, 14)), np.ones((1, 14)), np.zeros((1, 14)), 0.5, 0.5)
    assert result["oracle_nonviable_states"] == 1
    assert result["correct_abstentions"] == 1
    assert result["false_abstentions"] == 0


def test_viable_false_abstention_is_counted_only_on_viable_population():
    state = {"state_id": "s", "family": "large_enclosed_maze",
             "candidates": [candidate(i, contact=False, n_safe=2) for i in range(14)]}
    result = subject.decision_metrics([state], np.ones((1, 14)), np.ones((1, 14)), np.zeros((1, 14)), 0.5, 0.5)
    assert result["oracle_viable_states"] == 1
    assert result["false_abstentions"] == 1
    assert result["correct_abstentions"] == 0


def test_frozen_split_is_disjoint_and_exact():
    split = subject.freeze_split()
    assert len(split["development_training_state_ids"]) == 128
    assert len(split["internal_calibration_state_ids"]) == 24
    assert len(split["development_heldout_state_ids"]) == 24
    assert all(split["disjointness"].values())
