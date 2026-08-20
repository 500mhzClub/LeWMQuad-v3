from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "mechanism_specific_recovery",
    ROOT / "scripts/run_mechanism_specific_safety_composition_inference_recovery_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_primary_binding_is_physically_frozen():
    assert MODULE.PRIMARY_BINDING == {
        "contact": "ENHANCED_EMBODIED",
        "stuck": "ACTION_CONTROL_ONLY",
    }


def test_component_threshold_respects_recall_and_conservative_tie():
    probability = np.asarray([.1, .2, .3, .8, .9])
    labels = np.asarray([0, 0, 0, 1, 1])
    result = MODULE.choose_component_threshold(probability, labels, .90)
    admitted = probability < result["threshold"]
    assert np.mean(~admitted[labels.astype(bool)]) >= .90
    assert result["tie_rule"].startswith("more conservative")


def test_recovered_ledger_has_reusable_row_contract():
    index_path = MODULE.OUT / "row_level_component_predictions_v1_index.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text())
    assert index["rows"] == 576 and index["ticks"] == 15
    assert index["raw_logit_dtype"] == "float32"
    for field in ("branch_id", "labels", "action_control_logits", "enhanced_embodied_logits",
                  "primary_contact_probability", "primary_stuck_probability", "primary_admitted"):
        assert field in index["fields"]
