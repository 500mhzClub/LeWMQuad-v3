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

v18 = importlib.import_module(
    "scripts.execute_go2_rgb_object_space_height_volume_joint_jepa_v18"
)
v19 = importlib.import_module(
    "scripts.execute_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)


def _summary(
    *,
    passed: int = 89,
    shortfall: float = 53.0,
    pixel: float = 0.82,
    ground: float = 0.66,
    depth: float = 1.62,
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
        name: {check: value for check in v19.CONTROL_CHECK_NAMES}
        for name in v19.CONTROL_NAMES
    }


def _comparisons() -> dict[str, dict[str, object]]:
    return {
        control: {
            "scene_count": 8,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": v19.BOOTSTRAP_SEED,
            "equal_scene_mean_delta": 0.25,
            "bootstrap_lower_95": 0.10,
            "per_scene_delta": {
                f"scene_{index}": 0.02 + index * 0.01 for index in range(8)
            },
            "positive_family_count": len(v19.REGISTERED_FAMILIES),
            "family_deltas": {
                family: 0.05 + index * 0.01
                for index, family in enumerate(v19.REGISTERED_FAMILIES)
            },
        }
        for control in v19.CONTROL_NAMES
    }


def _accounting(update: int) -> dict[str, int]:
    return {
        name: update * multiplier
        for name, multiplier in v19.ACCOUNTING_MULTIPLIERS_V19.items()
    }


def _losses() -> dict[str, float]:
    values = {
        "S": 0.1,
        "P": 0.2,
        "U": 0.3,
        "R": 0.4,
        "O": 0.5,
        "Q": 1.2,
        "C": 0.6,
    }
    values["N"] = sum(values[name] for name in ("S", "P", "U", "R", "O"))
    values["L"] = values["N"] + values["C"] + values["Q"]
    return values


def _diagnostics() -> dict[str, float | int]:
    return {
        "successor_semantic_nll_normalized": 1.2,
        "persistence_semantic_nll_normalized": 1.4,
        "successor_minus_persistence_nll_normalized": -0.2,
        "changed_cell_fraction": 0.3,
        "non_hold_row_count": 10,
        "matching_predictor_gradient_cosine": 0.1,
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


def _result(*, torch: object, update: int = 2) -> SimpleNamespace:
    inherited_route = {
        "preclip_l2": 0.5,
        "applied_scale": 1.0,
        "parameter_tensor_count": 1,
        "absent_tensor_gradient_count": 0,
    }
    q_route = {
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
            "factual_successor_predictor": q_route,
        },
        mean_losses=_losses(),
        factual_successor_diagnostics=_diagnostics(),
        ranking_active_microbatches=4,
        ranking_eligible_pairs=8,
        survival_supervised_decisions=16,
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )


def _inherited_integrity_receipt() -> dict[str, object]:
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
        "passed": True,
    }


def test_private_adapter_binds_frozen_v18_without_mutating_public_module(
    capsys,
) -> None:
    assert v19._base is not v18
    assert v19.PRIVATE_V18_MODULE_NAME not in sys.modules
    assert v19.V18_EXECUTOR_COMMIT == "5567c9aa152b8aedcc085cfff46a7975668f7bfa"
    assert v19.V18_EXECUTOR_FILE_SHA256 == (
        "5ce4259126c21d0f474c0548f0ee6757f78225daa8ed778540f83764496d0e92"
    )
    assert v18.SCHEMA_PREFIX.endswith("joint_jepa_v18_integrity_replacement_v3")
    assert v19.SCHEMA_PREFIX.endswith("semantic_grounding_joint_jepa_v20")
    assert v19.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20/attempt_v1"
    )
    assert v19.MODEL_CLASS_NAME == v18.MODEL_CLASS_NAME
    assert v19.MAXIMUM_UPDATES == v18.MAXIMUM_UPDATES == 1_000
    assert v19.MAXIMUM_PRESENTATIONS == v18.MAXIMUM_PRESENTATIONS == 16_000
    assert v19.OBSERVATION_UPDATES == v18.OBSERVATION_UPDATES
    assert v19.TERMINAL_UPDATES == v18.TERMINAL_UPDATES
    assert v19.MATCHED_UPDATE400_THRESHOLDS == v18.MATCHED_UPDATE400_THRESHOLDS
    assert v19._engine.ACCOUNTING_MULTIPLIERS == (
        v19.INHERITED_ACCOUNTING_MULTIPLIERS_V13
    )
    assert v19._INHERITED_ACCOUNTING_MULTIPLIERS == (
        v19.INHERITED_ACCOUNTING_MULTIPLIERS_V13
    )
    assert "factual_successor_grad_calls" not in (
        v19._engine.ACCOUNTING_MULTIPLIERS
    )
    adapter = v19.private_adapter_receipt_v19()
    assert adapter["preregistration_commit"] == (
        "c99837b91aeb959e07da94e898e3ba11ccbb4c04"
    )
    assert adapter["v19_integrity_replacement_v1_preregistration_commit"] == (
        "691ed5d39f0b8d1b40071045dc181b9a4b215573"
    )
    assert (
        adapter["v19_integrity_replacement_v1_terminal_failure_result_commit"]
        == "7105e2d9ed6e724f364c837e84177b6b4c4cd163"
    )
    assert adapter["original_v19_preregistration_commit"] == (
        "6255a9a2cccffde4e777169eacf95105a828cf7e"
    )
    assert adapter["v19_terminal_failure_result_commit"] == (
        "37a87ac49ebcdebe57263476c20b1476877e36c2"
    )
    assert adapter["output_root"] == v19.OUTPUT_ROOT_RELATIVE_PATH
    assert adapter["inherited_accounting_registry"] == (
        v19.INHERITED_ACCOUNTING_MULTIPLIERS_V13
    )
    assert adapter["extended_accounting_is_local"] is True
    assert adapter["execution_authorized"] is False
    assert v19.main([]) == 4
    denial = v19.validate_content_bound_v19(json.loads(capsys.readouterr().out))
    assert denial["status"] == "DENIED_SOURCE_ONLY"
    assert denial["scientific_payload_opened"] is False
    assert denial["reservation_created"] is False


def test_parent_bindings_include_exact_preregistration_and_v18_result() -> None:
    expected = {
        v19.PREREGISTRATION_PATH: (
            v19.PREREGISTRATION_FILE_SHA256,
            v19.PREREGISTRATION_BYTE_COUNT,
        ),
        v19.V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_PATH: (
            v19.V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_FILE_SHA256,
            v19.V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_BYTE_COUNT,
        ),
        v19.V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_PATH: (
            v19.V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            v19.V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        v19.ORIGINAL_V19_PREREGISTRATION_PATH: (
            v19.ORIGINAL_V19_PREREGISTRATION_FILE_SHA256,
            v19.ORIGINAL_V19_PREREGISTRATION_BYTE_COUNT,
        ),
        v19.V19_TERMINAL_FAILURE_RESULT_PATH: (
            v19.V19_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            v19.V19_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        v19.V18_SCIENTIFIC_RESULT_PATH: (
            v19.V18_SCIENTIFIC_RESULT_FILE_SHA256,
            v19.V18_SCIENTIFIC_RESULT_BYTE_COUNT,
        ),
    }
    assert v19.PREREGISTRATION_COMMIT == (
        "c99837b91aeb959e07da94e898e3ba11ccbb4c04"
    )
    assert v19.V19_INTEGRITY_REPLACEMENT_V1_PREREGISTRATION_COMMIT == (
        "691ed5d39f0b8d1b40071045dc181b9a4b215573"
    )
    assert v19.V19_INTEGRITY_REPLACEMENT_V1_TERMINAL_FAILURE_RESULT_COMMIT == (
        "7105e2d9ed6e724f364c837e84177b6b4c4cd163"
    )
    assert v19.ORIGINAL_V19_PREREGISTRATION_COMMIT == (
        "6255a9a2cccffde4e777169eacf95105a828cf7e"
    )
    assert v19.V19_TERMINAL_FAILURE_RESULT_COMMIT == (
        "37a87ac49ebcdebe57263476c20b1476877e36c2"
    )
    assert v19.V18_SCIENTIFIC_RESULT_COMMIT == (
        "f2e290ce42f7b0cd142131f3272d1119b7b5d3d1"
    )
    for path, binding in expected.items():
        assert v19.BOUND_PARENT_SOURCES[path] == binding
    receipt = v19.validate_bound_sources_v19(ROOT, expected)
    assert receipt["validated_path_count"] == len(expected)


def test_v19_accounting_is_exactly_twelve_backward_and_eight_predictor_objectives() -> None:
    expected = v19.validate_terminal_accounting_v19(
        _accounting(400), terminal_update=400
    )
    assert expected["backward_calls"] == 4_800
    assert expected["predictor_objectives"] == 3_200
    assert expected["factual_successor_grad_calls"] == 1_600
    assert expected["factual_successor_objectives"] == 1_600
    assert expected["presentations"] == 6_400

    changed = dict(expected)
    changed["backward_calls"] -= 1
    with pytest.raises(RuntimeError, match="terminal accounting"):
        v19.validate_terminal_accounting_v19(changed, terminal_update=400)


def test_update_integrity_validates_q_route_losses_diagnostics_and_subset(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        v19,
        "_original_validate_update_integrity",
        lambda *args, **kwargs: _inherited_integrity_receipt(),
    )
    receipt = v19.validate_update_integrity_v19(
        SimpleNamespace(torch=torch),
        _Model(),
        _result(torch=torch),
        update=2,
        access_receipt={},
    )
    assert receipt["accounting"] == _accounting(2)
    assert set(receipt["gradient_routes"]) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "factual_successor_predictor",
    }
    assert receipt["mean_losses"]["L"] == pytest.approx(3.3)
    assert receipt["factual_successor_diagnostics"] == _diagnostics()
    grounding = receipt["v19_executed_successor_semantic_grounding"]
    assert grounding["parameter_tensor_count"] == 13
    assert grounding["parameter_count"] == 259_008
    assert grounding["representation_gradient_from_q"] is False
    assert grounding["passed"] is True


def test_real_synthetic_update_crosses_unmocked_inherited_accounting_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = importlib.import_module(
        "lewm.tests.test_run_go2_rgb_object_space_height_volume_executed_"
        "successor_semantic_grounding_joint_jepa_v19"
    )
    torch = synthetic.torch
    model = synthetic._TinyModel()
    partition = synthetic._partition(model)
    synthetic._install_tiny_apis(monkeypatch, model, partition)
    optimizer = synthetic._CountingSgd(list(partition.online))
    result = synthetic.runner.joint_training_update_v19(
        model,
        optimizer,
        synthetic._microbatches(),
    )

    class _TargetView:
        training = False

        def parameters(self):
            return (model.target,)

    class _RegisteredModelView:
        target_hard_sync_count = torch.tensor(1, dtype=torch.long)

        def __getattr__(self, name):
            return getattr(model, name)

        def named_parameters(self):
            return (
                ("encoder.synthetic", model.shared),
                ("semantic_head.synthetic", model.representation),
                *tuple(
                    zip(
                        synthetic.TRANSITION_NAMES,
                        model.transition,
                        strict=True,
                    )
                ),
                *tuple(
                    zip(
                        synthetic.SURVIVAL_NAMES,
                        model.survival,
                        strict=True,
                    )
                ),
                ("target_encoder.synthetic", model.target),
            )

        def target_modules(self):
            return (_TargetView(),)

    runtime = SimpleNamespace(torch=torch)
    receipt = v19.validate_update_integrity_v19(
        runtime,
        _RegisteredModelView(),
        result,
        update=1,
        access_receipt={
            "forbidden_input_count": 0,
            "probability_calibration_open_count": 0,
            "opened_roles": ("train",),
        },
    )
    assert receipt["passed"] is True
    assert receipt["accounting"] == _accounting(1)
    assert receipt["gradient_routes"][
        "factual_successor_predictor"
    ]["parameter_tensor_count"] == 13
    assert receipt["gradient_routes"][
        "factual_successor_predictor"
    ]["absent_tensor_gradient_count"] == 0
    assert receipt["v19_executed_successor_semantic_grounding"] == {
        "parameter_tensor_count": 13,
        "parameter_count": 259_008,
        "representation_gradient_from_q": False,
        "semantic_head_gradient_from_q": False,
        "target_gradient_from_q": False,
        "passed": True,
    }

    monkeypatch.setattr(
        v19._engine,
        "ACCOUNTING_MULTIPLIERS",
        dict(v19.ACCOUNTING_MULTIPLIERS_V19),
    )
    with pytest.raises(RuntimeError, match="inherited accounting registry changed"):
        v19.validate_update_integrity_v19(
            runtime,
            _RegisteredModelView(),
            result,
            update=1,
            access_receipt={
                "forbidden_input_count": 0,
                "probability_calibration_open_count": 0,
                "opened_roles": ("train",),
            },
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.accounting.__setitem__("backward_calls", 23),
            "per-update accounting",
        ),
        (
            lambda result: result.gradient_routes[
                "factual_successor_predictor"
            ].__setitem__("parameter_tensor_count", 12),
            "gradient route failed integrity",
        ),
        (
            lambda result: result.mean_losses.__setitem__("L", 99.0),
            "loss equations",
        ),
        (
            lambda result: result.factual_successor_diagnostics.__setitem__(
                "successor_minus_persistence_nll_normalized", 0.0
            ),
            "diagnostics are inconsistent",
        ),
    ),
)
def test_update_integrity_rejects_each_extended_contract_break(
    monkeypatch,
    mutation,
    message: str,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        v19,
        "_original_validate_update_integrity",
        lambda *args, **kwargs: _inherited_integrity_receipt(),
    )
    result = _result(torch=torch)
    mutation(result)
    with pytest.raises(RuntimeError, match=message):
        v19.validate_update_integrity_v19(
            SimpleNamespace(torch=torch),
            _Model(),
            result,
            update=2,
            access_receipt={},
        )


def test_observation_retains_cached_numeric_comparisons_without_changing_controls(
    monkeypatch,
) -> None:
    inherited = {
        "schema": "inherited",
        "update": 100,
        "physical": {"passed": True},
        "v12_gate": {"passed": False},
        "controls": None,
        "integrity_pass": True,
    }
    monkeypatch.setattr(
        v19,
        "_original_observation",
        lambda *args, **kwargs: dict(inherited),
    )
    cache = _comparisons()
    runtime = SimpleNamespace(causal_comparisons_v19={100: cache})
    observed = v19.observation_v19(
        runtime,
        object(),
        update=100,
        integrity_pass=True,
    )
    assert observed["controls"] is None
    assert observed["physical"] == inherited["physical"]
    assert observed["v12_gate"] == inherited["v12_gate"]
    assert observed["causal_comparisons"] == cache
    cache[v19.CONTROL_NAMES[0]]["family_deltas"][v19.REGISTERED_FAMILIES[0]] = 9.0
    assert (
        observed["causal_comparisons"][v19.CONTROL_NAMES[0]]["family_deltas"]
        [v19.REGISTERED_FAMILIES[0]]
        != 9.0
    )


def test_observation_rejects_incomplete_numeric_comparisons(monkeypatch) -> None:
    monkeypatch.setattr(
        v19,
        "_original_observation",
        lambda *args, **kwargs: {"controls": _controls()},
    )
    comparisons = _comparisons()
    comparisons.pop(v19.CONTROL_NAMES[-1])
    runtime = SimpleNamespace(causal_comparisons_v19={400: comparisons})
    with pytest.raises(RuntimeError, match="control set changed"):
        v19.observation_v19(
            runtime,
            object(),
            update=400,
            integrity_pass=True,
        )


def test_update400_and_final_gates_retain_v18_decisions() -> None:
    update100 = _summary(passed=60, shortfall=90.0, depth=2.2)
    update400 = _summary()
    controls = _controls()
    expected_400 = v18.evaluate_update400_gate_v18(
        update100,
        update400,
        controls,
        integrity_pass=True,
    )
    observed_400 = v19.evaluate_update400_gate_v19(
        update100,
        update400,
        controls,
        integrity_pass=True,
    )
    assert observed_400["schema"].startswith(v19.SCHEMA_PREFIX)
    assert {k: v for k, v in observed_400.items() if k != "schema"} == {
        k: v for k, v in expected_400.items() if k != "schema"
    }

    v12_gate = {
        "passed": True,
        "checks": {name: True for name in v19.V12_GATE_CHECK_NAMES},
    }
    physical = _summary(
        passed=120,
        shortfall=30.0,
        pixel=0.84,
        ground=0.68,
        depth=0.90,
        complete=1,
    )
    expected_final = v18.evaluate_final_gate_v18(
        v12_gate, physical, integrity_pass=True
    )
    observed_final = v19.evaluate_final_gate_v19(
        v12_gate, physical, integrity_pass=True
    )
    assert observed_final["schema"].startswith(v19.SCHEMA_PREFIX)
    assert {k: v for k, v in observed_final.items() if k != "schema"} == {
        k: v for k, v in expected_final.items() if k != "schema"
    }


def test_training_api_requires_v19_types_and_entrypoints(monkeypatch) -> None:
    monkeypatch.setattr(
        v19,
        "_original_validate_training_api",
        lambda module: {
            "required_function_count": 5,
            "required_batch_key_count": len(v19.TRAINING_REQUIRED_BATCH_KEYS),
            "presentations_per_update": 16,
        },
    )
    module = SimpleNamespace(
        JointTrainingAccountingV19=type("JointTrainingAccountingV19", (), {}),
        JointUpdateResultV19=type("JointUpdateResultV19", (), {}),
        joint_training_update_v19=lambda: None,
        validate_accounting_v19=lambda: None,
    )
    receipt = v19.validate_training_api_v19(module)
    assert receipt["backward_calls_per_update"] == 12
    assert receipt["predictor_objectives_per_update"] == 8
    assert receipt["factual_successor_objectives_per_update"] == 4

    broken = copy.copy(module)
    del broken.joint_training_update_v19
    with pytest.raises(RuntimeError, match="training callable is absent"):
        v19.validate_training_api_v19(broken)
