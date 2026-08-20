from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "factorised_micro_safety",
    ROOT / "scripts/train_evaluate_factorised_micro_safety_world_model_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_specialists_are_disjoint_and_below_parameter_cap():
    model = MODULE.FactorisedModel()
    contact = {id(value) for value in model.contact.parameters()}
    stuck = {id(value) for value in model.stuck.parameters()}
    assert not contact & stuck
    assert sum(value.numel() for value in model.parameters()) == 205252
    assert sum(value.numel() for value in model.parameters()) < 500000


def test_specialist_shapes_and_causality():
    model = MODULE.FactorisedModel().eval()
    current_contact = torch.randn(2, len(MODULE.CONTACT_SENSOR))
    future_contact = torch.randn(2, 15, len(MODULE.CONTACT_SENSOR))
    current_stuck = torch.randn(2, len(MODULE.STUCK_SENSOR))
    future_stuck = torch.randn(2, 15, len(MODULE.STUCK_SENSOR))
    action = torch.randn(2, 15, 6)
    with torch.inference_mode():
        contact = model.contact(current_contact, future_contact, action)
        stuck = model.stuck(current_stuck, future_stuck, action)
        changed = future_contact.clone(); changed[:, 10:] += 100
        contact_changed = model.contact(current_contact, changed, action)
    assert contact.shape == stuck.shape == (2, 15, 2)
    assert torch.equal(contact[:, :10], contact_changed[:, :10])


def test_strict_threshold_and_fixture_contract():
    values = MODULE.threshold_values(np.asarray([.2, .2, .8]))
    assert values[0] < 0 and values[-1] > 1
    assert not np.any(np.asarray([.2, .8]) < .2)
    assert MODULE.evaluator_fixture()["pass"]


def test_kinematic_tie_contract():
    rows = [
        {"candidate_index": index, "kinematic": np.asarray([0, 0, 0, 1, index / 100., 0.], np.float32)}
        for index in range(12)
    ]
    assert MODULE.route_order(rows, list(range(12)))[0] == 9


def test_fresh_manifest_is_prospectively_frozen_and_disjoint_if_present():
    path = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_panel_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    assert manifest["frozen_before_candidate_execution"]
    assert manifest["state_count"] == 48
    assert manifest["split_state_count"] == {"calibration": 24, "heldout": 24}
    assert manifest["disjointness"]["distinct_scene_count"] == 48
    assert manifest["disjointness"]["old_panel_scene_overlap"] == 0
    assert manifest["disjointness"]["predictor_scene_overlap"] == 0


def test_completed_result_persists_final_checkpoint_and_row_evidence():
    path = ROOT / ".generated/factorised_micro_safety_world_model_v1/result.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    assert result["classification"] in {
        "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_SIGNAL",
        "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_NO_SIGNAL",
    }
    assert result["training"]["seed"] == 2026082010
    assert result["training"]["epochs"] == 60
    assert result["row_level_evidence"]["rows"] == 288
    assert result["row_level_evidence"]["row_level_evidence_persistence"]
    assert result["custody"]["jepa_predictor_opened_or_trained"] is False
