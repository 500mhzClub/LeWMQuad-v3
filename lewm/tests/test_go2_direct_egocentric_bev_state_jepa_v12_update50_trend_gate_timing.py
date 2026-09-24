from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_state_jepa_v12_update50_trend_gate_timing"
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
def test_v12_sources_import_without_tensor_runtime(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_v12_source_only', path)
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


def test_only_decision_policy_changes_and_closure_is_143_sources() -> None:
    contract = _load(CONTRACT, "_v12_science_identity")
    checker = _load(CHECKER, "_v12_closure_identity")
    active = contract.science_contract()
    frozen = contract.frozen_v11_science_contract()

    for field in (
        "repository_goal",
        "model",
        "data",
        "loader",
        "objective",
        "optimizer",
        "schedule",
        "phase_adapter",
        "access_policy",
    ):
        assert active[field] == frozen[field]

    active_gates = copy.deepcopy(active["gates"])
    frozen_gates = copy.deepcopy(frozen["gates"])
    active_u50 = active_gates["thresholds"].pop("50")
    frozen_u50 = frozen_gates["thresholds"].pop("50")
    active_u50_controls = active_gates["controls"].pop("50")
    frozen_u50_controls = frozen_gates["controls"].pop("50")
    assert active_gates == frozen_gates
    assert active_u50 == contract.GATE_THRESHOLDS[50]
    assert active_u50_controls == list(contract.GATE_CONTROLS[50])
    assert frozen_u50 == contract._V11.GATE_THRESHOLDS[50]
    assert frozen_u50_controls == list(contract._V11.GATE_CONTROLS[50])
    assert active_u50 != frozen_u50

    identity = contract.science_identity_receipt()
    assert identity["scientific_delta_count"] == 1
    assert contract.MODEL_RELATIVE_PATH == V10_MODEL.relative_to(ROOT).as_posix()
    assert not any("/models/" in path for path in contract.ADDITIVE_SOURCE_PATHS)
    assert len(contract.REUSED_SOURCE_PATHS) == 138
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 143
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 143
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0


def _common(contract: Any, update: int) -> dict[str, Any]:
    return {
        **contract.PERCEPTION_ACCOUNTING[update],
        "v8_mechanism_receipt_ready": True,
        "active_training_scope_v8": "perception_only",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
    }


def _update_zero(contract: Any) -> dict[str, Any]:
    exact = (
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
        **_common(contract, 0),
        **{field: True for field in exact},
        "initial_online_to_target_hard_sync_count": 1,
        "correct_rgb_scene_win_count": 8,
        "G": 1.0,
        "aggregate_raster_balanced_accuracy": 0.40,
        "aggregate_occupied_recall": 0.60,
        "rough_raster_balanced_accuracy": 0.10,
        "rough_raster_occupied_recall": 0.10,
    }


def _update_50(contract: Any) -> dict[str, Any]:
    return {
        **_common(contract, 50),
        "G": 0.90,
        # Both are below V10's absolute floors but improve over update zero.
        "aggregate_raster_balanced_accuracy": 0.50,
        "aggregate_free_recall": 0.25,
        "aggregate_occupied_recall": 0.70,
        "aggregate_raster_nll": 0.80,
        "rough_raster_balanced_accuracy": 0.01,
        "rough_raster_occupied_recall": 0.01,
        "correct_rgb_scene_win_count": 8,
    }


def test_v11_preliminary_and_non_u50_final_gates_are_exact() -> None:
    contract = _load(CONTRACT, "_v12_unchanged_dispatch")
    for update in contract.OBSERVATION_UPDATES:
        assert contract.evaluate_gate(update, {}) == contract._V11.evaluate_gate(
            update, {}
        )

    zero = _update_zero(contract)
    hundred = {
        **_common(contract, 100),
        "G": 0.80,
        "aggregate_raster_balanced_accuracy": 0.70,
        "aggregate_free_recall": 0.50,
        "aggregate_occupied_recall": 0.80,
        "aggregate_raster_nll": 0.46,
        "rough_raster_balanced_accuracy": 0.100001,
        "rough_raster_occupied_recall": 0.100001,
        "correct_rgb_scene_win_count": 8,
    }
    terminal = {
        **_common(contract, 250),
        "G": 0.70,
        "aggregate_raster_balanced_accuracy": 0.80,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "aggregate_raster_nll": 0.42,
        "rough_raster_balanced_accuracy": 0.7719525,
        "rough_raster_occupied_recall": 0.4319467,
        "correct_rgb_scene_win_count": 8,
    }
    cases = (
        (0, zero, {}),
        (100, hundred, {"update_zero": zero}),
        (
            250,
            terminal,
            {
                "update_zero": zero,
                "update_100": {"aggregate_raster_nll": 0.41},
            },
        ),
    )
    for update, metrics, kwargs in cases:
        assert contract.evaluate_gate(update, metrics, **kwargs) == (
            contract._V11.evaluate_gate(update, metrics, **kwargs)
        )


def test_u50_replaces_only_two_predicates_and_uses_strict_trends() -> None:
    contract = _load(CONTRACT, "_v12_u50_delta")
    zero = _update_zero(contract)
    metrics = _update_50(contract)
    frozen = contract._V11.evaluate_gate(50, metrics, update_zero=zero)
    active = contract.evaluate_gate(50, metrics, update_zero=zero)
    old = {
        "aggregate_raster_balanced_accuracy_at_least_point60",
        "aggregate_occupied_recall_at_least_point75",
    }
    new = {
        "aggregate_raster_balanced_accuracy_strictly_higher_than_update_zero",
        "aggregate_occupied_recall_strictly_higher_than_update_zero",
    }
    assert old.issubset(frozen["conjuncts"])
    assert new.issubset(active["conjuncts"])
    assert {
        key: value for key, value in frozen["conjuncts"].items() if key not in old
    } == {
        key: value for key, value in active["conjuncts"].items() if key not in new
    }
    assert all(active["conjuncts"].values())
    assert frozen["passed"] is False
    assert active["passed"] is True
    assert active["control"] == contract.GATE_CONTROLS[50][1]

    for field, baseline in (
        ("aggregate_raster_balanced_accuracy", 0.40),
        ("aggregate_occupied_recall", 0.60),
    ):
        equal = dict(metrics)
        equal[field] = baseline
        failed = contract.evaluate_gate(50, equal, update_zero=zero)
        assert failed["passed"] is False
        assert failed["control"] == contract.GATE_CONTROLS[50][0]


def test_v12_failure_controls_and_runtime_wrapper_seams() -> None:
    contract = _load(CONTRACT, "_v12_controls")
    old_u50_failure = contract._V11.GATE_CONTROLS[50][0]
    new_u50_failure = contract.GATE_CONTROLS[50][0]
    assert old_u50_failure != new_u50_failure
    assert contract.CONTROL_PASS == contract._V11.CONTROL_PASS
    fields = ("metrics", "artifact", "result", "completion")
    for control in contract.FAILURE_CONTROLS:
        chain = dict.fromkeys(fields, control)
        assert contract.validate_failure_status_chain(chain) == chain
    with pytest.raises(ValueError, match="one exact V12 gate control"):
        contract.validate_failure_status_chain(
            dict.fromkeys(fields, old_u50_failure)
        )

    runner = _load(RUNNER, "_v12_runner_seams")
    launcher = _load(LAUNCHER, "_v12_launcher_seams")
    runner._assert_v12_bindings()
    launcher._assert_v12_bindings()
    expected_v9 = dict(runner._V9._V9_SEAM_TABLE)
    for name, expected_v8 in runner._V9._V8._V8_SEAM_TABLE:
        assert getattr(runner._LEAF, name) is expected_v9.get(name, expected_v8)
    assert runner._LEAF._evaluate_observation_impl is (
        runner._V9._v9_evaluate_observation_impl
    )
    assert runner._LEAF._snapshot_model is runner._V9._v9_snapshot_model
    assert runner._LEAF._terminal_failure is runner._V9._v9_terminal_failure
    assert runner._LEAF.contract.validate_failure_status_chain is (
        runner.contract.validate_failure_status_chain
    )
    assert all(
        name == runner.V10_MODEL_RUNTIME_MODULE_NAME
        for name in runner._runtime_module_names()
    )
    assert launcher._LEAF._V11._BASE.RUNNER_PATH == RUNNER
