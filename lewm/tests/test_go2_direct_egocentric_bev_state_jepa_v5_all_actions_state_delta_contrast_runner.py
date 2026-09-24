from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "scripts/run_go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V5_"
    "ALL_ACTIONS_STATE_DELTA_CONTRAST_PREFLIGHT_JSON"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("_test_direct_bev_v5_state_delta_runner")


def test_isolated_import_is_stdlib_only_and_deeply_rebound() -> None:
    program = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_v5_runner_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
owners = (
    module._V4, module._V4._V3, module._V4._V3._V2,
    module._V4._V3._V2._V1,
)
assert all(owner.contract is module.contract for owner in owners)
assert all(owner.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r} for owner in owners)
assert all(pathlib.Path(owner.__file__).resolve() == path for owner in owners)
assert module._V4.V4_MODEL_RUNTIME_MODULE_NAME == module.V5_MODEL_RUNTIME_MODULE_NAME
assert module._V4._V3.V3_MODEL_RUNTIME_MODULE_NAME == module.V5_MODEL_RUNTIME_MODULE_NAME
assert module._V4._V3._V2.V2_MODEL_RUNTIME_MODULE_NAME == module.V5_MODEL_RUNTIME_MODULE_NAME
deepest = module._V4._V3._V2._V1
assert deepest._gradient_integrity_probe is module._V4._v4_gradient_integrity_probe
assert deepest._initialize_model is module._V4._v4_initialize_model
assert deepest.contract.validate_failure_status_chain is module.contract.validate_failure_status_chain
args = module.parse_args([
    '--run',
    '--review-sha256', '0' * 64,
    '--authorization-sha256', '1' * 64,
])
assert args.review_sha256 == '0' * 64
assert args.authorization_sha256 == '1' * 64
print('PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_parser_authority_and_parent_delegation(monkeypatch) -> None:
    args = runner.parse_args([
        "--run",
        "--review-sha256", "2" * 64,
        "--authorization-sha256", "3" * 64,
    ])
    assert args.review_sha256 == "2" * 64
    assert args.authorization_sha256 == "3" * 64
    assert runner.contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert runner.contract.EXECUTION_AUTHORITY["maximum_attempts"] == 1
    assert runner.contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert runner.contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000

    calls: list[tuple[str, str]] = []

    def fake_run_parent(
        *, review_file_sha256: str, authorization_file_sha256: str
    ) -> int:
        assert runner._V4._V3._V2._V1.contract is runner.contract
        calls.append((review_file_sha256, authorization_file_sha256))
        return 29

    monkeypatch.setattr(runner._V4, "run_parent", fake_run_parent)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 29
    assert calls == [("a" * 64, "b" * 64)]


def test_model_loader_uses_only_v5_runtime_identity(monkeypatch) -> None:
    calls: list[tuple[str, Path]] = []
    sentinel = object()

    def fake_loader(name: str, path: Path) -> object:
        calls.append((name, Path(path)))
        return sentinel

    v3 = runner._V4._V3
    monkeypatch.setattr(v3, "_FROZEN_V1_SOURCE_ONLY_MODULE", fake_loader)
    model_path = ROOT / runner.contract.MODEL_RELATIVE_PATH
    assert v3._source_only_runtime_module("ignored", model_path) is sentinel
    assert calls == [(runner.V5_MODEL_RUNTIME_MODULE_NAME, model_path)]

    calls.clear()
    other_path = ROOT / "lewm/models/vision_transformer.py"
    assert v3._source_only_runtime_module("kept", other_path) is sentinel
    assert calls == [("kept", other_path)]


class _FakeHead:
    in_channels = 16
    out_channels = 3
    kernel_size = (3, 3)
    padding = (1, 1)
    bias = object()


class _FakePredictor:
    def __init__(self) -> None:
        self.residual_head = _FakeHead()
        self.net = (self.residual_head,)


class _FakeModel:
    def __init__(self) -> None:
        self.predictor = _FakePredictor()
        self.marker = object()


def test_v4_gradient_hook_adapter_is_inherited_unchanged(monkeypatch) -> None:
    real = _FakeModel()

    def fake_frozen(_runtime, model, _partition, _batch):
        assert isinstance(model, runner._V4._ResidualHeadHookModelView)
        assert model.predictor is real.predictor.residual_head
        assert model.marker is real.marker
        return {
            "training_objective_call_counts": {
                "online_state_stack": 3,
                "predictor": 1,
                "target_state_stack": 3,
            },
            "six_call_graph_isolation_exact": True,
        }

    monkeypatch.setattr(
        runner._V4,
        "_FROZEN_GRADIENT_INTEGRITY_PROBE",
        fake_frozen,
    )
    deepest = runner._V4._V3._V2._V1
    assert deepest._gradient_integrity_probe is (
        runner._V4._v4_gradient_integrity_probe
    )
    result = deepest._gradient_integrity_probe(None, real, {}, {})
    assert result["six_call_graph_isolation_exact"] is True
    assert result["training_objective_call_counts"]["predictor"] == 1


def test_fresh_initial_state_hash_adapter_is_inherited_fail_closed(
    monkeypatch,
) -> None:
    model = object()
    partition = {"groups": object()}
    expected = runner.contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256
    monkeypatch.setattr(
        runner._V4,
        "_FROZEN_INITIALIZE_MODEL",
        lambda *_args: (
            model,
            partition,
            {"complete_initial_state_sha256": expected},
        ),
    )
    initialized = runner._V4._v4_initialize_model(None, None, None, None)
    assert initialized[0] is model
    assert initialized[1] is partition
    assert initialized[2]["complete_initial_state_sha256"] == expected

    monkeypatch.setattr(
        runner._V4,
        "_FROZEN_INITIALIZE_MODEL",
        lambda *_args: (
            model,
            partition,
            {"complete_initial_state_sha256": "f" * 64},
        ),
    )
    with pytest.raises(RuntimeError, match="fresh V4 initial model"):
        runner._V4._v4_initialize_model(None, None, None, None)


def test_version_local_failure_chain_accepts_only_one_failure_control() -> None:
    validator = runner.contract.validate_failure_status_chain
    assert validator.__module__ == runner.contract.__name__
    for control in runner.contract.FAILURE_CONTROLS:
        chain = {
            "metrics": control,
            "artifact": control,
            "result": control,
            "completion": control,
        }
        assert validator(chain) == chain

    pass_chain = {
        "metrics": runner.contract.CONTROL_PASS,
        "artifact": runner.contract.CONTROL_PASS,
        "result": runner.contract.CONTROL_PASS,
        "completion": runner.contract.CONTROL_PASS,
    }
    with pytest.raises(ValueError, match=r"one exact (?:V5 )?gate control"):
        validator(pass_chain)

    mixed = {
        "metrics": runner.contract.FAILURE_CONTROLS[0],
        "artifact": runner.contract.FAILURE_CONTROLS[0],
        "result": runner.contract.FAILURE_CONTROLS[-1],
        "completion": runner.contract.FAILURE_CONTROLS[0],
    }
    with pytest.raises(ValueError, match=r"one exact (?:V5 )?gate control"):
        validator(mixed)
