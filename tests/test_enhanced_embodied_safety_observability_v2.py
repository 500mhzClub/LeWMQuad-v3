from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "enhanced_embodied_safety_v2",
    ROOT / "scripts/train_evaluate_enhanced_embodied_safety_observability_v2.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_model_remains_below_parameter_cap():
    count = sum(parameter.numel() for parameter in MODULE.EnhancedSafetyModel().parameters())
    assert count < 500_000


def test_enhanced_contract_excludes_privileged_inputs():
    excluded = set(MODULE.SENSOR.CHANNELS)
    assert not any("global" in value for value in excluded)
    assert not any("label" in value for value in excluded)
    assert not any("rgb" in value for value in excluded)


def test_channel_contract_dimensions_are_frozen():
    assert len(MODULE.SENSOR.CHANNELS) == 73
    assert len(MODULE.SENSOR.ACTION_CONTROL_CHANNELS) == 6


def test_common_evaluator_fixture_passes():
    assert MODULE.evaluator_fixture()["pass"] is True
