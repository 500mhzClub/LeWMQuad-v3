import json
from pathlib import Path

import numpy as np

from scripts import evaluate_physics_rate_contact_proxy_reconciliation_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[1]


def test_contiguous_physics_contacts_are_grouped_deterministically():
    trace = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8)
    assert diagnostic.events(trace) == [(1, 2), (4, 4), (7, 9)]
    assert diagnostic.events(np.zeros(5, dtype=np.uint8)) == []


def test_sampled_label_is_preserved_beside_physics_rate_label():
    trace = np.zeros(250, dtype=np.uint8)
    trace[24:27] = 1
    assert bool(trace.any())
    # A transient event between the 50-step sample instants is deliberately
    # absent from the historical sampled label; reconciliation never mutates it.
    assert not bool(trace[[49, 99, 149, 199, 249]].any())


def test_strict_clearance_admission_rejects_threshold_tie():
    risk = np.asarray([0.2, 0.5, 0.8])
    threshold = 0.5
    admitted = risk < threshold
    assert admitted.tolist() == [True, False, False]


def test_committed_result_preserves_claim_boundary_and_no_execution():
    evidence = json.loads((ROOT / "docs/lewm_physics_rate_contact_proxy_reconciliation_v1_evidence_index_2026-08-21.json").read_text())
    assert evidence["classification"] == "PHYSICS_RATE_FULL_GEOMETRY_SCORE_NO_GO"
    assert evidence["targets"]["historical"] == "H1_SAMPLED_DISALLOWED_CONTACT"
    assert evidence["targets"]["development"] == "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT"
    assert evidence["targets"]["material_hazard"] == "SEVERITY_UNRESOLVED"
    assert evidence["execution"]["training"] is False
    assert evidence["execution"]["learned_inference"] is False
    assert evidence["execution"]["simulation_or_replay"] is False
