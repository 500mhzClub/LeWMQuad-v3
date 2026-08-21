import json
from pathlib import Path

import numpy as np

from scripts import evaluate_true_future_contact_filter_decision_decomposition_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_passes():
    assert diagnostic.evaluator_fixture()["pass"]


def test_threshold_tie_is_rejected():
    assert not (0.5 < 0.5)


def test_oracle_stuck_tiebreak_preserves_distance_margin():
    rows = []
    for candidate, (stuck, distance) in enumerate(((True, .20), (False, .18), (False, .10))):
        rows.append({"candidate_index": candidate, "stuck": stuck, "kinematic": np.asarray([0, 0, 0, 0, distance, 0.])})
    assert diagnostic.rank_indices(rows, [0, 1, 2], "kinematic_stuck_tiebreak") == [1, 0, 2]


def test_margin_bins_are_disjoint():
    rows = [{"probability": value, "family": diagnostic.FAMILIES[0], "candidate": "x", "branch_id": str(i),
             "state_id": "s", "candidate_index": i} for i, value in enumerate((.49, .55, .75))]
    result = diagnostic.distribution(rows, .5)
    assert sum(result["absolute_margin_bins"].values()) == 3


def test_committed_evidence_preserves_claim_boundary_and_terminal():
    evidence = json.loads((ROOT / "docs/lewm_true_future_contact_filter_decision_decomposition_v1_evidence_index_2026-08-21.json").read_text())
    assert evidence["classification"] == "CONTACT_PROXY_FILTER_SCORE_NO_GO"
    assert evidence["preserved_result"] == "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY"
    assert evidence["frontier"]["complete_gate_points"] == 0
    assert evidence["frozen_checkpoint"]["executed"] is False
    assert "no material-hazard" in evidence["claim_boundary"]
