import json
from pathlib import Path

import numpy as np

from scripts import evaluate_control_commitment_aligned_contact_filter_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_and_first_contact_intervals():
    assert diagnostic.fixture()["pass"]
    assert diagnostic.first_interval(np.zeros(15, dtype=np.uint8)) == "NO_CONTACT_THROUGH_H3"
    cumulative = np.zeros(15, dtype=np.uint8); cumulative[10:] = 1
    assert diagnostic.first_interval(cumulative) == "FIRST_CONTACT_H2_TO_H3"


def test_temperature_fit_is_finite_and_deterministic():
    logits = np.asarray([-3.0, -1.0, 1.0, 3.0])
    labels = np.asarray([0, 0, 1, 1])
    first = diagnostic.fit_temperature(logits, labels)
    second = diagnostic.fit_temperature(logits, labels)
    assert 0.05 <= first <= 20.0
    assert first == second


def test_soft_ranking_preserves_distance_tolerance_before_risk():
    rows = [
        {"candidate_index": 0, "kinematic": np.asarray([0, 0, 0, 0, .20, 0]), "p_h2": .9, "p_h3": .9},
        {"candidate_index": 1, "kinematic": np.asarray([0, 0, 0, 0, .18, 0]), "p_h2": .1, "p_h3": .1},
        {"candidate_index": 2, "kinematic": np.asarray([0, 0, 0, 0, .10, 0]), "p_h2": 0., "p_h3": 0.},
    ]
    assert diagnostic.ranked_order(rows, [0, 1, 2], "soft_scores") == [1, 0, 2]


def test_committed_evidence_has_strict_claim_and_no_execution():
    evidence = json.loads((ROOT / "docs/lewm_control_commitment_aligned_contact_filter_v1_evidence_index_2026-08-21.json").read_text())
    assert evidence["classification"] == "CONTACT_SCORE_NO_GO_ACROSS_CONTROL_HORIZONS"
    assert evidence["secondary_classification"] == "CONTINUATION_RISK_RANKING_NO_SIGNAL"
    assert evidence["checkpoint"]["executed"] is False
    assert evidence["committed_horizon"] == {"blocks": 1, "command_ticks": 5, "duration_s": .5, "label": "H1"}
    assert evidence["stopping_horizon_status"] == "CONSERVATIVE_UNVALIDATED_STOPPING_PROXY"
    assert "no material-hazard" in evidence["claim_boundary"]
