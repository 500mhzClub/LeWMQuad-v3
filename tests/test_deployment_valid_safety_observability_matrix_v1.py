from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deployment_valid_matrix",
    ROOT / "scripts/train_evaluate_deployment_valid_safety_observability_matrix_v1.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_all_architectures_remain_below_parameter_cap():
    for condition in MODULE.CONDITIONS:
        count = sum(parameter.numel() for parameter in MODULE.SafetyModalityModel(condition).parameters())
        assert count < 750_000


def test_condition_keyed_seeds_are_distinct_and_repeatable():
    first = [MODULE.condition_seed(condition) for condition in MODULE.CONDITIONS]
    second = [MODULE.condition_seed(condition) for condition in MODULE.CONDITIONS]
    assert first == second
    assert len(set(first)) == len(first)


def test_input_allow_lists_preserve_modality_isolation():
    action = MODULE.input_allow_list("ACTION_CONTROL_ONLY")
    raw = MODULE.input_allow_list("RAW_RGB")
    proprio = MODULE.input_allow_list("PROPRIOCEPTION")
    fusion = MODULE.input_allow_list("RGB_PLUS_PROPRIOCEPTION")
    assert not any("rgb" in value for value in action)
    assert not any("proprio" in value for value in action)
    assert any("rgb" in value for value in raw) and not any("proprio" in value for value in raw)
    assert any("proprio" in value for value in proprio) and not any("rgb" in value for value in proprio)
    assert any("rgb" in value for value in fusion) and any("proprio" in value for value in fusion)


def test_common_evaluator_fixture_passes():
    assert MODULE.evaluator_fixture()["pass"] is True
