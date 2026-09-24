from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COLLECT = load("factorised_scaling_collect", "scripts/collect_factorised_micro_safety_data_scaling_v2.py")
TRAIN = load("factorised_scaling_train", "scripts/train_evaluate_factorised_micro_safety_data_scaling_v2.py")


def test_nested_inventory_and_seed_contracts():
    assert COLLECT.TOTAL_PER_FAMILY == 36
    assert COLLECT.split_identity(0, 0) == ("fit192_extra", "scale-fit-0-00")
    assert COLLECT.split_identity(0, 24) == ("calibration", "scale-cal-0-00")
    assert COLLECT.split_identity(0, 30) == ("heldout", "scale-held-0-00")
    assert TRAIN.derived_seed("fit96") != TRAIN.derived_seed("fit192")
    assert TRAIN.derived_seed("fit96") == TRAIN.derived_seed("fit96")


def test_architecture_is_exactly_v1():
    model = TRAIN.BASE.FactorisedModel()
    assert sum(value.numel() for value in model.contact.parameters()) == 97346
    assert sum(value.numel() for value in model.stuck.parameters()) == 107906
    assert sum(value.numel() for value in model.parameters()) == 205252
    assert not ({id(value) for value in model.contact.parameters()} & {id(value) for value in model.stuck.parameters()})


def test_manifest_is_frozen_and_disjoint_if_present():
    path = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text())
    assert manifest["frozen_before_candidate_execution"]
    assert manifest["state_count"] == 144
    assert manifest["split_state_count"] == {"fit192_extra": 96, "calibration": 24, "heldout": 24}
    assert manifest["disjointness"]["distinct_scene_count"] == 144
    assert manifest["disjointness"]["original_fit48_scene_overlap"] == 0
    assert manifest["disjointness"]["prior_fresh48_scene_overlap"] == 0
    assert manifest["disjointness"]["predictor_scene_overlap"] == 0


def test_completed_result_preserves_evidence_if_present():
    path = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/result.json"
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    assert result["classification"] in {
        "DATA_SCALING_FRESH_PANEL_INADEQUATE",
        "FACTORISED_MICRO_SAFETY_DATA_SCALING_SIGNAL",
        "FACTORISED_MICRO_SAFETY_DATA_SCALING_POSITIVE_TENDENCY",
        "FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL",
    }
    if result["classification"] != "DATA_SCALING_FRESH_PANEL_INADEQUATE":
        assert result["conditions"]["fit48"]["training"]["retrained"] is False
        assert result["conditions"]["fit96"]["row_level_evidence"]["row_level_evidence_persistence"]
        assert result["conditions"]["fit192"]["row_level_evidence"]["row_level_evidence_persistence"]
        assert result["custody"]["one_seed_family_used"]
        assert not result["custody"]["jepa_predictor_or_rgb_or_depth_lidar_model_trained"]
