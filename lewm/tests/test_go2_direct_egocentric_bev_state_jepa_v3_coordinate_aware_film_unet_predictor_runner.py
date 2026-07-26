from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT / "scripts/run_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V3_"
    "COORDINATE_AWARE_FILM_UNET_PREDICTOR_PREFLIGHT_JSON"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_and_complete_stack_is_rebound() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location('_direct_bev_v3_runner', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert not any(name.startswith('numpy.') for name in sys.modules)
assert 'PIL' not in sys.modules
assert module._V2.contract is module.contract
assert module._V2._V1.contract is module.contract
assert module._V2.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._V2._V1.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert Path(module._V2.__file__).resolve() == path
assert Path(module._V2._V1.__file__).resolve() == path
assert module._V2._V1._evaluate_observation_impl is module._v3_evaluate_observation_impl
assert module.contract.MODEL_PARAMETER_INVENTORY['predictor'][
    'ordered_parameter_name_sha256'
] == '0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd'
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


def test_entrypoints_delegate_after_rebinding(monkeypatch) -> None:
    runner = _load("_direct_bev_v3_runner_delegate", RUNNER)
    calls: list[tuple[str, str]] = []

    def fake_run_parent(
        *,
        review_file_sha256: str,
        authorization_file_sha256: str,
    ) -> int:
        assert runner._V2.contract is runner.contract
        assert runner._V2._V1.contract is runner.contract
        calls.append((review_file_sha256, authorization_file_sha256))
        return 31

    monkeypatch.setattr(runner._V2, "run_parent", fake_run_parent)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 31
    assert calls == [("a" * 64, "b" * 64)]


def test_only_v3_model_runtime_import_name_is_adapted(monkeypatch) -> None:
    runner = _load("_direct_bev_v3_model_name_adapter", RUNNER)
    calls: list[tuple[str, Path]] = []

    def fake_source_loader(name: str, path: Path) -> tuple[str, Path]:
        calls.append((name, path))
        return name, path

    monkeypatch.setattr(
        runner,
        "_FROZEN_V1_SOURCE_ONLY_MODULE",
        fake_source_loader,
    )
    model_path = runner.ROOT / runner.contract.MODEL_RELATIVE_PATH
    observed = runner._source_only_runtime_module(
        "lewm.models.direct_egocentric_bev_state_jepa_v1",
        model_path,
    )
    assert observed == (runner.V3_MODEL_RUNTIME_MODULE_NAME, model_path)

    other_path = runner.ROOT / runner.contract.FROZEN_V2_CONTRACT_RELATIVE_PATH
    observed_other = runner._source_only_runtime_module(
        "unchanged_module_name",
        other_path,
    )
    assert observed_other == ("unchanged_module_name", other_path)
    assert calls == [
        (runner.V3_MODEL_RUNTIME_MODULE_NAME, model_path),
        ("unchanged_module_name", other_path),
    ]


def test_runtime_uses_contract_bound_inventory_directly() -> None:
    runner = _load("_direct_bev_v3_inventory_adapter", RUNNER)
    inventory = runner.contract.MODEL_PARAMETER_INVENTORY
    declared = runner.contract.model_config()["parameter_inventory"]
    assert inventory["predictor"] == {
        "parameter_count": 317_107,
        "tensor_count": 79,
        "ordered_parameter_name_sha256": (
            "0398031cb776c10a23b14c7935d2566f4a3087175213e87b49c2a05cadf6e1dd"
        ),
    }
    assert inventory["total"] == {
        "parameter_count": 6_552_249,
        "tensor_count": 277,
    }
    for group in (
        "encoder", "decoder_state", "detached_target_encoder_decoder_state"
    ):
        assert inventory[group] == declared[group]
    assert not hasattr(runner, "V3_MODEL_PARAMETER_INVENTORY")


def _update_zero_metrics() -> dict[str, object]:
    return {
        "G": 1.0,
        "J": 1.0,
        "C": 1.0,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "three_logit_bottleneck_exact": False,
        "no_hidden_or_auxiliary_bypass": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "target_parameters_gradient_free": True,
        "intended_online_path_gradient_nonzero": True,
        "six_call_graph_isolation_exact": True,
        "all_registered_values_finite": True,
    }


def test_update_zero_adapts_only_architecture_integrity_receipt(
    monkeypatch,
) -> None:
    runner = _load("_direct_bev_v3_observation_adapter", RUNNER)
    metrics = _update_zero_metrics()

    def fake_observation(*args, **kwargs):
        return {"metrics": dict(metrics), "gate": {"passed": False}}

    monkeypatch.setattr(
        runner,
        "_FROZEN_V1_EVALUATE_OBSERVATION_IMPL",
        fake_observation,
    )
    predictor = SimpleNamespace(
        enc64=SimpleNamespace(conv1=SimpleNamespace(in_channels=5)),
        residual_head=SimpleNamespace(in_channels=16, out_channels=3),
    )
    model = SimpleNamespace(
        state_head=SimpleNamespace(out_channels=3),
        predictor=predictor,
    )
    observed = runner._v3_evaluate_observation_impl(
        object(),
        object(),
        model,
        {},
        object(),
        [],
        {},
        object(),
        update=0,
        update_zero=None,
        prior_gates_passed=True,
    )
    assert observed["metrics"]["three_logit_bottleneck_exact"] is True
    assert observed["gate"]["passed"] is True
    assert observed["gate"]["control"] == "CONTINUE_AFTER_UPDATE_ZERO_GATE"


def test_update_zero_architecture_receipt_is_fail_closed(monkeypatch) -> None:
    runner = _load("_direct_bev_v3_observation_reject", RUNNER)

    def fake_observation(*args, **kwargs):
        return {"metrics": _update_zero_metrics(), "gate": {"passed": False}}

    monkeypatch.setattr(
        runner,
        "_FROZEN_V1_EVALUATE_OBSERVATION_IMPL",
        fake_observation,
    )
    predictor = SimpleNamespace(
        enc64=SimpleNamespace(conv1=SimpleNamespace(in_channels=6)),
        residual_head=SimpleNamespace(in_channels=16, out_channels=3),
    )
    model = SimpleNamespace(
        state_head=SimpleNamespace(out_channels=3),
        predictor=predictor,
    )
    observed = runner._v3_evaluate_observation_impl(
        object(), object(), model, {}, object(), [], {}, object(),
        update=0,
        update_zero=None,
        prior_gates_passed=True,
    )
    assert observed["metrics"]["three_logit_bottleneck_exact"] is False
    assert observed["gate"]["passed"] is False
    assert observed["gate"]["control"] == (
        "FAIL_UPDATE_ZERO_INTEGRITY_GATE_TERMINAL_NO_RETRY"
    )
