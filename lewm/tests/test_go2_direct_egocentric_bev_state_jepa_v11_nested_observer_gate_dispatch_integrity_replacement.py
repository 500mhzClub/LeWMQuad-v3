from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "go2_direct_egocentric_bev_state_jepa_v11_"
    "nested_observer_gate_dispatch_integrity_replacement"
)
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
V10_MODEL = (
    ROOT
    / "lewm/models/"
    "direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding.py"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_v11_sources_import_without_tensor_runtime(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_v11_source_only', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
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


def test_science_governance_model_and_source_closure_are_exact() -> None:
    contract = _load(CONTRACT, "_v11_contract_identity_test")
    checker = _load(CHECKER, "_v11_checker_identity_test")

    assert contract.science_contract() == contract.frozen_v10_science_contract()
    identity = contract.science_identity_receipt()
    assert identity["scientific_delta_count"] == 0
    assert identity["normalized_exactly_equals_frozen_v10"] is True
    assert identity["v11_science_contract_sha256"] == (
        "bf839c0897d73f21b789b8e4c0d9277cba6c2c387e4ccbe347aa4cf91eadff43"
    )
    assert contract.MODEL_RELATIVE_PATH == V10_MODEL.relative_to(ROOT).as_posix()
    assert not any("/models/" in path for path in contract.ADDITIVE_SOURCE_PATHS)
    assert len(contract.REUSED_SOURCE_PATHS) == 133
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 138
    assert contract.GATE_DISPATCH_ACCOUNTING == {
        0: {
            "preliminary_call_count": 2,
            "final_delegate_call_count": 1,
            "total_evaluate_gate_call_count": 3,
        },
        50: {
            "preliminary_call_count": 1,
            "final_delegate_call_count": 1,
            "total_evaluate_gate_call_count": 2,
        },
        100: {
            "preliminary_call_count": 1,
            "final_delegate_call_count": 1,
            "total_evaluate_gate_call_count": 2,
        },
        250: {
            "preliminary_call_count": 1,
            "final_delegate_call_count": 1,
            "total_evaluate_gate_call_count": 2,
        },
    }
    governing = contract.validate_governing_documents()
    for binding in (
        contract.frozen_v10_source_manifest_binding(),
        contract.frozen_v10_review_binding(),
        contract.frozen_v10_authorization_binding(),
        contract.v10_terminal_audit_binding(),
        contract.preregistration_binding(),
    ):
        assert governing[binding["path"]] == binding["file_sha256"]

    manifest = checker.build_manifest()
    assert manifest["source_count"] == 138
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0


def test_both_absent_is_exact_nonauthorizing_preliminary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load(CONTRACT, "_v11_preliminary_test")
    calls: list[object] = []
    monkeypatch.setattr(
        contract._V10,
        "evaluate_gate",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    for update in contract.OBSERVATION_UPDATES:
        gate = contract.evaluate_gate(update, {}, prior_gates_passed=True)
        assert gate == {
            "update": update,
            "active_training_scope_v8_present": False,
            "passed": True,
            "control": contract.CONTROL_PRELIMINARY_NESTED_DISPATCH,
            "gate_mode": (
                "PRELIMINARY_NESTED_OBSERVER_GATE_DISPATCH_NOT_FINAL_"
                "SCIENTIFIC_EVIDENCE"
            ),
            "v8_mechanism_receipt_ready": False,
            "scientific_gate_evidence": False,
            "execution_training_checkpoint_terminal_pass_or_downstream_"
            "authority": False,
            "must_be_overwritten_by_frozen_outer_v8_final_dispatch": True,
            "final_gate_evaluated": False,
            "thresholds": dict(contract.GATE_THRESHOLDS[update]),
            "thresholds_applied": False,
            "perception_accounting": dict(contract.PERCEPTION_ACCOUNTING[update]),
            "perception_accounting_applied": False,
            "prior_gates_passed_validated_only": True,
        }
    assert calls == []
    assert contract.CONTROL_PRELIMINARY_NESTED_DISPATCH not in (
        *contract.FAILURE_CONTROLS,
        contract.CONTROL_PASS,
    )
    with pytest.raises(ValueError, match="one exact V10 gate control"):
        contract.validate_failure_status_chain(
            dict.fromkeys(
                ("metrics", "artifact", "result", "completion"),
                contract.CONTROL_PRELIMINARY_NESTED_DISPATCH,
            )
        )
    with pytest.raises(ValueError, match="update must be"):
        contract.evaluate_gate(1, {})
    with pytest.raises(ValueError, match="prior_gates_passed"):
        contract.evaluate_gate(0, {}, prior_gates_passed=1)


@pytest.mark.parametrize(
    "metrics",
    [
        {"v8_mechanism_receipt_ready": True},
        {"active_training_scope_v8": "perception_only"},
        {
            "v8_mechanism_receipt_ready": True,
            "active_training_scope_v8": "perception_only",
        },
        {"v8_mechanism_receipt_ready": False},
        {"v8_mechanism_receipt_ready": None},
        {"active_training_scope_v8": "wrong"},
    ],
)
def test_every_other_presence_combination_delegates_once_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    metrics: dict[str, Any],
) -> None:
    contract = _load(CONTRACT, f"_v11_delegate_test_{id(metrics)}")
    update_zero = {"G": 1.0}
    update_100 = {"aggregate_raster_nll": 0.4}
    sentinel = object()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def delegate(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(contract._V10, "evaluate_gate", delegate)
    before = dict(metrics)
    result = contract.evaluate_gate(
        250,
        metrics,
        update_zero=update_zero,
        update_100=update_100,
        prior_gates_passed=False,
    )
    assert result is sentinel
    assert metrics == before
    assert calls == [
        (
            (250, metrics),
            {
                "update_zero": update_zero,
                "update_100": update_100,
                "prior_gates_passed": False,
            },
        )
    ]

    error = RuntimeError("frozen V10 sentinel")
    monkeypatch.setattr(
        contract._V10,
        "evaluate_gate",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(RuntimeError) as raised:
        contract.evaluate_gate(250, metrics, update_100=update_100)
    assert raised.value is error


def _update_zero_metrics(contract: Any) -> dict[str, Any]:
    exact_fields = (
        "fresh_v8_model_and_optimizer_zero_prior_runtime_reuse",
        "n320_encoder_only_migration_exact",
        "registered_seed_draw_order_exact",
        "initial_model_state_matches_frozen_v8",
        "model_parameter_inventory_exact",
        "v8_decoder_parameter_inventory_exact",
        "learned_only_forbidden_geometry_absent",
        "two_residual_cross_attention_ffn_blocks_exact",
        "negative_squared_prototype_distance_formula_exact",
        "online_target_perception_bitwise_equal",
        "three_channel_state_exact",
        "all_logits_in_closed_interval_minus4_to0",
        "v8_intended_gradient_coverage_exact",
        "predictor_target_and_fixed_negative_gradients_absent",
        "no_hidden_auxiliary_bypass",
        "all_forbidden_access_counts_zero",
    )
    return {
        **contract.PERCEPTION_ACCOUNTING[0],
        **{field: True for field in exact_fields},
        "v8_mechanism_receipt_ready": True,
        "active_training_scope_v8": "perception_only",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "initial_online_to_target_hard_sync_count": 1,
        "correct_rgb_scene_win_count": 8,
    }


def test_marker_present_final_gate_is_exact_frozen_v10_result() -> None:
    contract = _load(CONTRACT, "_v11_final_gate_test")
    metrics = _update_zero_metrics(contract)
    expected = contract._V10.evaluate_gate(0, metrics)
    actual = contract.evaluate_gate(0, metrics)
    assert actual == expected
    assert actual["gate_mode"] == "FINAL_V10_FINAL_CLASS_MACRO_GROUNDING_RECEIPT"

    failing = dict(metrics)
    failing["correct_rgb_scene_win_count"] = 7
    assert contract.evaluate_gate(0, failing) == contract._V10.evaluate_gate(
        0, failing
    )
    assert contract.evaluate_gate(0, failing)["passed"] is False

    # The markerless update-250 calls must not require the baseline that only
    # the outer frozen V8 observer owns.
    assert contract.evaluate_gate(250, {})["passed"] is True


def test_actual_captured_topology_and_wrapper_seams_are_frozen() -> None:
    contract = _load(CONTRACT, "_v11_topology_contract_test")
    runner = _load(RUNNER, "_v11_topology_runner_test")
    launcher = _load(LAUNCHER, "_v11_topology_launcher_test")
    runner._assert_v11_bindings()
    launcher._assert_v11_bindings()

    v8 = runner._V9._V8
    captured = v8._V6._FROZEN_EVALUATE_OBSERVATION_IMPL
    assert captured.__name__ == "_v3_evaluate_observation_impl"
    assert captured is not v8._V6._v6_evaluate_observation_impl
    captured_source = inspect.getsource(captured)
    assert "_FROZEN_V1_EVALUATE_OBSERVATION_IMPL" in captured_source
    assert captured_source.count("contract.evaluate_gate(") == 1
    assert "if update == 0:" in captured_source
    v8_source = inspect.getsource(v8._v8_evaluate_observation_impl)
    assert "_V6._FROZEN_EVALUATE_OBSERVATION_IMPL" in v8_source
    assert v8_source.count("contract.evaluate_gate(") == 1
    assert '"v8_mechanism_receipt_ready": True' in v8_source
    assert '"active_training_scope_v8": "perception_only"' in v8_source

    expected_v9 = dict(runner._V9._V9_SEAM_TABLE)
    for name, expected_v8 in runner._V9._V8._V8_SEAM_TABLE:
        assert getattr(runner._LEAF, name) is expected_v9.get(name, expected_v8)
    assert runner._LEAF._snapshot_model is runner._V9._v9_snapshot_model
    assert runner._LEAF._terminal_failure is runner._V9._v9_terminal_failure
    assert runner._LEAF._evaluate_observation_impl is (
        runner._V9._v9_evaluate_observation_impl
    )
    assert all(
        name == runner.V10_MODEL_RUNTIME_MODULE_NAME
        for name in runner._runtime_module_names()
    )
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER
    assert contract.MODEL_RELATIVE_PATH == runner.MODEL_RELATIVE_PATH
