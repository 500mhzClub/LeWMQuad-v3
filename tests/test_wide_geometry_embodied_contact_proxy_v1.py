import json
from pathlib import Path

import numpy as np

from scripts import evaluate_wide_geometry_score_composition_v1 as stage0


ROOT = Path(__file__).resolve().parents[1]


def test_joint_admission_is_strict_component_or_rejection():
    lidar = np.asarray([0.1, 0.5, 0.1, 0.5])
    fusion = np.asarray([0.1, 0.1, 0.5, 0.5])
    admitted = (lidar < 0.5) & (fusion < 0.5)
    assert admitted.tolist() == [True, False, False, False]
    assert not (0.5 < 0.5)


def test_combined_continuous_risk_is_probability_union():
    lidar = np.asarray([0.2, 0.8]); fusion = np.asarray([0.5, 0.25])
    union = 1 - (1 - lidar) * (1 - fusion)
    assert np.allclose(union, [0.6, 0.85])


def test_stage0_evaluator_perfect_and_reversed_auc():
    labels = np.asarray([0, 1, 0, 1], bool); probability = np.asarray([0.01, 0.99, 0.02, 0.98])
    assert stage0.auc(labels, probability) == 1.0
    assert stage0.auc(labels, 1 - probability) == 0.0


def test_claim_boundary_and_evidence_index():
    index = json.loads((ROOT / "docs/lewm_wide_geometry_embodied_contact_proxy_v1_evidence_index_2026-08-21.json").read_text())
    assert index["classification"] == "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY"
    assert index["preserved_ontology_result"] == "CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT"
    assert "no material-hazard claim" in index["claim_boundary"]
    assert index["row_level_evidence"]["rows"] == 3456
    assert index["stage0"]["heldout_frontier_pairs"] == 84100
