from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

v21 = importlib.import_module(
    "scripts.execute_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)


def _accounting(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in v21.ACCOUNTING_MULTIPLIERS_V21.items()
    }


def _losses() -> dict[str, float]:
    values = {
        "S": 0.1,
        "P": 0.2,
        "U": 0.3,
        "R": 0.4,
        "O": 0.5,
        "I_fit": 1.0,
        "I_rank": 0.5,
        "C": 0.6,
    }
    values["I_scene"] = values["I_fit"] + values["I_rank"]
    values["N"] = sum(values[name] for name in ("S", "P", "U", "R", "O"))
    values["L"] = values["N"] + values["C"] + values["I_scene"]
    return values


def _diagnostics() -> dict[str, float | int]:
    return {
        "positive_energy_mean": 1.0,
        "negative_energy_mean": 1.25,
        "advantage_sum": 4.0,
        "advantage_count": 16,
        "advantage_mean": 0.25,
        "matching_predictor_gradient_cosine": 0.1,
        "valid_cell_count": 3_000,
        "high_salience_cell_count": 128,
        "low_salience_cell_count": 128,
    }


class _Parameter:
    def __init__(self, count: int) -> None:
        self._count = count

    def numel(self) -> int:
        return self._count


class _Model:
    def named_parameters(self):
        counts = [19_923] * 12 + [19_932]
        return tuple(
            (f"predictor.transition_{index}", _Parameter(count))
            for index, count in enumerate(counts)
        )


def _result(update: int = 2) -> SimpleNamespace:
    inherited_route = {
        "preclip_l2": 0.5,
        "applied_scale": 1.0,
        "parameter_tensor_count": 1,
        "absent_tensor_gradient_count": 0,
    }
    innovation_route = {
        "preclip_l2": 2.0,
        "applied_scale": 0.5,
        "parameter_tensor_count": 13,
        "absent_tensor_gradient_count": 0,
    }
    return SimpleNamespace(
        accounting=_accounting(update),
        gradient_routes={
            "camera_shared": dict(inherited_route),
            "joint_shared": dict(inherited_route),
            "representation": dict(inherited_route),
            "predictor": dict(inherited_route),
            "scene_innovation_predictor": innovation_route,
        },
        mean_losses=_losses(),
        scene_innovation_diagnostics=_diagnostics(),
        ranking_active_microbatches=4,
        ranking_eligible_pairs=8,
        survival_supervised_decisions=16,
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def _inherited_receipt() -> dict[str, object]:
    route = {
        "preclip_l2": 0.5,
        "applied_scale": 1.0,
        "parameter_tensor_count": 1,
        "absent_tensor_gradient_count": 0,
    }
    return {
        "update": 2,
        "accounting": {},
        "gradient_routes": {
            name: dict(route)
            for name in (
                "camera_shared",
                "joint_shared",
                "representation",
                "predictor",
            )
        },
        "mean_losses": {},
        "factual_successor_diagnostics": {},
        "v19_executed_successor_semantic_grounding": {},
        "passed": True,
    }


def _summary(
    *,
    passed: int = 89,
    shortfall: float = 53.0,
    depth: float = 1.62,
    pixel: float = 0.82,
    ground: float = 0.66,
    complete: int = 0,
) -> dict[str, object]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -0.5,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _controls(value: bool = True) -> dict[str, dict[str, bool]]:
    return {
        name: {check: value for check in v21.CONTROL_CHECK_NAMES}
        for name in v21.CONTROL_NAMES
    }


def test_private_v20_adapter_binds_fresh_v21_identity_and_denies_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert v21.PRIVATE_V20_MODULE_NAME not in sys.modules
    assert v21._base.__name__ != v21.V20_PUBLIC_MODULE_NAME
    assert v21.SCHEMA_PREFIX.endswith("contrastive_innovation_joint_jepa_v21")
    assert v21.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "contrastive_innovation_joint_jepa_v21/attempt_v1"
    )
    assert v21.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert v21.TERMINAL_UPDATES == (400, 1_000)
    assert v21.CURRENT_EXECUTION_AUTHORIZED is False
    assert v21.main([]) == 4
    denied = json.loads(capsys.readouterr().out)
    assert denied["status"] == "DENIED_SOURCE_ONLY"
    assert denied["scientific_payload_opened"] is False
    receipt = v21.private_adapter_receipt_v21()
    assert receipt["preregistration_commit"] == (
        "c2bbce067175dd980c9ed2511dc14db5a222afe4"
    )
    assert receipt["v20_scientific_result_commit"] == (
        "8321d76004aa1f3c87dfa04c3b18d701267a89ec"
    )
    assert receipt["inherited_batch_registry_unchanged"] is True
    assert receipt["v20_update400_and_update1000_gates_unchanged"] is True


def test_parent_bindings_include_exact_preregistration_and_v20_result() -> None:
    expected = {
        v21.PREREGISTRATION_PATH: (
            v21.PREREGISTRATION_FILE_SHA256,
            v21.PREREGISTRATION_BYTE_COUNT,
        ),
        v21.V20_SCIENTIFIC_RESULT_PATH: (
            v21.V20_SCIENTIFIC_RESULT_FILE_SHA256,
            v21.V20_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
    for path, binding in expected.items():
        assert v21.BOUND_PARENT_SOURCES[path] == binding
    receipt = v21.validate_bound_sources_v21(ROOT, expected)
    assert receipt["validated_path_count"] == 2


def test_training_api_requires_exact_one_field_extension(monkeypatch) -> None:
    monkeypatch.setattr(
        v21,
        "_original_validate_training_api",
        lambda module: {
            "required_function_count": 5,
            "required_batch_key_count": len(v21.TRAINING_REQUIRED_BATCH_KEYS),
            "presentations_per_update": 16,
        },
    )
    module = SimpleNamespace(
        JointTrainingAccountingV21=type("JointTrainingAccountingV21", (), {}),
        JointUpdateResultV21=type("JointUpdateResultV21", (), {}),
        joint_training_update_v21=lambda: None,
        validate_accounting_v21=lambda: None,
        SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21=(
            v21.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
        ),
        REQUIRED_BATCH_KEYS_V21=v21.TRAINING_REQUIRED_BATCH_KEYS_V21,
    )
    receipt = v21.validate_training_api_v21(module)
    assert receipt["required_batch_key_count_v21"] == (
        len(v21.TRAINING_REQUIRED_BATCH_KEYS) + 1
    )
    broken = copy.copy(module)
    broken.REQUIRED_BATCH_KEYS_V21 = tuple(
        reversed(v21.TRAINING_REQUIRED_BATCH_KEYS_V21)
    )
    with pytest.raises(RuntimeError, match="batch extension changed"):
        v21.validate_training_api_v21(broken)


def test_engine_microbatch_hook_accepts_only_ordered_v21_extension(
    monkeypatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        v21._engine,
        "_validate_batch_query_identity_v13",
        lambda model, batch: calls.append(model),
    )
    training = SimpleNamespace(
        _validate_microbatches_v21=lambda torch, batches: calls.append(batches)
    )
    runtime = SimpleNamespace(training_module=training, torch=object())
    batches = tuple(
        {name: object() for name in v21.TRAINING_REQUIRED_BATCH_KEYS_V21}
        for _ in range(4)
    )
    model = object()
    v21.validate_microbatches_for_engine_v21(runtime, model, batches)
    assert calls[:4] == [model] * 4
    assert calls[4] is batches

    malformed = list(batches)
    row = dict(malformed[0])
    value = row.pop(v21.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21)
    row = {
        v21.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21: value,
        **row,
    }
    malformed[0] = row
    with pytest.raises(PermissionError, match="microbatch schema changed"):
        v21.validate_microbatches_for_engine_v21(runtime, model, tuple(malformed))


def test_update_integrity_projects_through_v20_and_publishes_only_v21_receipts(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    captured: list[object] = []

    def inherited(runtime, model, result, **kwargs):
        captured.append(result)
        return _inherited_receipt()

    monkeypatch.setattr(v21, "_original_validate_update_integrity", inherited)
    runtime = SimpleNamespace(torch=torch)
    receipt = v21.validate_update_integrity_v21(
        runtime,
        _Model(),
        _result(),
        update=2,
        access_receipt={},
    )
    projected = captured[0]
    assert projected.accounting["factual_successor_grad_calls"] == 8
    assert "scene_innovation_grad_calls" not in projected.accounting
    assert projected.mean_losses["Q"] == _losses()["I_scene"]
    assert "scene_innovation_predictor" not in projected.gradient_routes
    assert receipt["accounting"] == _accounting(2)
    assert set(receipt["gradient_routes"]) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "scene_innovation_predictor",
    }
    assert receipt["mean_losses"] == _losses()
    assert receipt["scene_innovation_diagnostics"] == _diagnostics()
    assert "factual_successor_diagnostics" not in receipt
    assert "v19_executed_successor_semantic_grounding" not in receipt
    assert runtime.scene_innovation_diagnostics_v21 == {2: _diagnostics()}
    grounding = receipt["v21_same_action_cross_scene_contrastive_innovation"]
    assert grounding["parameter_tensor_count"] == 13
    assert grounding["parameter_count"] == 259_008
    assert grounding["negative_row_batch_key"] == "scene_innovation_negative_row"
    assert grounding["target_gradient_from_i_scene"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.accounting.__setitem__("backward_calls", 23),
            "per-update accounting",
        ),
        (
            lambda result: result.gradient_routes[
                "scene_innovation_predictor"
            ].__setitem__("parameter_tensor_count", 12),
            "gradient route failed integrity",
        ),
        (
            lambda result: result.mean_losses.__setitem__("L", 99.0),
            "loss equations",
        ),
        (
            lambda result: result.scene_innovation_diagnostics.__setitem__(
                "advantage_count", 15
            ),
            "diagnostics are inconsistent",
        ),
    ),
)
def test_update_integrity_rejects_extended_contract_breaks(
    monkeypatch,
    mutation,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        v21,
        "_original_validate_update_integrity",
        lambda *args, **kwargs: _inherited_receipt(),
    )
    result = _result()
    mutation(result)
    with pytest.raises(RuntimeError, match=message):
        v21.validate_update_integrity_v21(
            SimpleNamespace(torch=torch),
            _Model(),
            result,
            update=2,
            access_receipt={},
        )


def test_observation_adds_only_current_update_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        v21,
        "_original_observation",
        lambda *args, **kwargs: {"schema": "inherited", "update": kwargs["update"]},
    )
    runtime = SimpleNamespace(scene_innovation_diagnostics_v21={100: _diagnostics()})
    observed = v21.observation_v21(
        runtime,
        object(),
        update=100,
        integrity_pass=True,
    )
    assert observed["scene_innovation_diagnostics"] == _diagnostics()
    assert v21.observation_v21(
        SimpleNamespace(),
        object(),
        update=0,
        integrity_pass=True,
    ) == {"schema": "inherited", "update": 0}
    with pytest.raises(RuntimeError, match="lacks current-update"):
        v21.observation_v21(
            SimpleNamespace(scene_innovation_diagnostics_v21={}),
            object(),
            update=400,
            integrity_pass=True,
        )


def test_terminal_accounting_and_v20_gates_are_unchanged() -> None:
    expected = v21.validate_terminal_accounting_v21(
        _accounting(400), terminal_update=400
    )
    assert expected["presentations"] == 6_400
    assert expected["scene_innovation_grad_calls"] == 1_600
    assert expected["scene_innovation_objectives"] == 1_600
    assert v21.TERMINAL_UPDATES == (400, 1_000)

    before = _summary(passed=60, shortfall=90.0, depth=2.2)
    after = _summary()
    controls = _controls()
    observed = v21.evaluate_update400_gate_v21(
        before,
        after,
        controls,
        integrity_pass=True,
    )
    expected_gate = v21._base.evaluate_update400_gate_v19(
        before,
        after,
        controls,
        integrity_pass=True,
    )
    assert observed == expected_gate
    assert observed["passed"] is True
    assert observed["schema"].startswith(v21.SCHEMA_PREFIX)

    v12_gate = {
        "passed": True,
        "checks": {name: True for name in v21.V12_GATE_CHECK_NAMES},
    }
    final_physical = _summary(
        passed=120,
        shortfall=30.0,
        depth=0.90,
        pixel=0.84,
        ground=0.68,
        complete=1,
    )
    observed_final = v21.evaluate_final_gate_v21(
        v12_gate,
        final_physical,
        integrity_pass=True,
    )
    expected_final = v21._base.evaluate_final_gate_v19(
        v12_gate,
        final_physical,
        integrity_pass=True,
    )
    assert observed_final == expected_final
    assert observed_final["passed"] is True
    assert observed_final["schema"].startswith(v21.SCHEMA_PREFIX)
