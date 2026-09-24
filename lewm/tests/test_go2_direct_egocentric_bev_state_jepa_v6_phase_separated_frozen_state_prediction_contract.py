from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
V5_CONTRACT = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("_direct_bev_v6_phase_contract_test", CONTRACT)
v5 = _load("_direct_bev_v6_phase_frozen_v5_test", V5_CONTRACT)


def _phase_receipt(update: int) -> dict[str, object]:
    value: dict[str, object] = {
        "v6_phase_receipt_ready": True,
        "active_phase_v6": "phase_one" if update in (0, 100) else "phase_two",
    }
    value.update(contract.PHASE_ACCOUNTING[update])
    return value


def _update_zero_metrics() -> dict[str, object]:
    return {
        **_phase_receipt(0),
        "initial_model_state_matches_frozen_v3": True,
        "model_parameter_inventory_exact": True,
        "three_logit_bottleneck_exact": True,
        "no_hidden_or_auxiliary_bypass": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "target_parameters_gradient_free": True,
        "intended_online_path_gradient_nonzero": True,
        "six_call_graph_isolation_exact": True,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "registered_state_and_target_nonconstant": True,
        "no_prior_runtime_or_protected_input": True,
        "phase_one_trainability_exact": True,
        "phase_one_gradient_isolation_exact": True,
        "phase_two_gradient_isolation_exact": True,
        "dual_gradient_probe_nonmutating_exact": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "G": 1.0,
        "aggregate_raster_nll": 0.8,
        "rough_raster_balanced_accuracy": 0.3,
    }


def _update_100_metrics() -> dict[str, object]:
    return {
        **_phase_receipt(100),
        "G": 0.5,
        "aggregate_raster_balanced_accuracy": 0.70,
        "aggregate_raster_nll": 0.4,
        "rough_raster_balanced_accuracy": 0.6,
        "correct_rgb_scene_win_count": 8,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "predictor_matches_initialization": True,
        "predictor_residual_head_exact_zero": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "phase_one_trainability_exact": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "predictor_state_sha256": "b" * 64,
    }


def _perception_metrics() -> dict[str, object]:
    return {
        "aggregate_raster_balanced_accuracy": 0.95,
        "aggregate_free_recall": 0.95,
        "aggregate_occupied_recall": 0.90,
        "aggregate_raster_nll": 0.10,
        "rough_raster_balanced_accuracy": 0.85,
        "rough_raster_occupied_recall": 0.60,
        "correct_rgb_scene_win_count": 8,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
    }


def _phase_two_integrity() -> dict[str, object]:
    return {
        "online_target_perception_bitwise_equal": True,
        "phase_two_trainability_exact": True,
        "online_perception_eval_mode": True,
        "target_perception_eval_mode": True,
        "predictor_train_mode": True,
        "phase_two_module_modes_exact": True,
        "zero_rgb_online_repeat_bitwise_equal": True,
        "zero_rgb_target_repeat_bitwise_equal": True,
        "zero_rgb_witness_exact": True,
        "online_perception_state_sha256": "a" * 64,
        "target_perception_state_sha256": "a" * 64,
        "predictor_state_sha256": "d" * 64,
        "predictor_update400_sha256": "d" * 64,
        "perception_metrics_update400_baseline_sha256": "e" * 64,
        "J_update400_boundary": 0.50,
        "C_update400_boundary": 0.80,
    }


def _update_400_metrics() -> dict[str, object]:
    return {
        **_phase_receipt(400),
        **_perception_metrics(),
        **_phase_two_integrity(),
        "predictor_matches_initialization": True,
        "predictor_residual_head_exact_zero": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "boundary_phase_two_gradient_isolation_exact": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
        "J": 0.50,
        "C": 0.80,
    }


def _update_1000_metrics() -> dict[str, object]:
    return {
        **_phase_receipt(1_000),
        **_perception_metrics(),
        **_phase_two_integrity(),
        "online_perception_unchanged_from_update400": True,
        "target_perception_unchanged_from_update400": True,
        "perception_metrics_unchanged_from_update400": True,
        "J": 0.44,
        "C": 0.70,
        "action_nll": 2.0,
        "action_macro_balanced_accuracy": 0.30,
        "hardest_wrong_positive_scene_count": 8,
        "same_action_target_nll": 0.60,
        "same_action_target_strict_win_rate": 0.80,
        "target_positive_scene_count": 8,
    }


def _synthetic_manifest() -> bytes:
    bindings = [
        {"path": path, "file_sha256": "1" * 64, "byte_count": 1}
        for path in contract.SOURCE_PATHS
    ]
    core = {
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_CLOSURE",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources": list(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES),
        "excluded_runtime_categories": list(contract.PROHIBITED_RUNTIME_CATEGORIES),
        "source_paths": list(contract.SOURCE_PATHS),
        "source_bindings": bindings,
        "source_bindings_sha256": contract.canonical_json_sha256(bindings),
        "source_count": 111,
        "generated_input_open_count": 0,
        "checkpoint_or_tensor_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
        "authority": contract.SOURCE_ONLY_AUTHORITY,
    }
    value = contract.with_content_sha256(core)
    return contract.canonical_json_bytes(value) + b"\n"


def test_isolated_import_is_stdlib_source_only() -> None:
    program = f"""
import importlib.util, json, pathlib, sys
path = pathlib.Path({str(CONTRACT)!r})
spec = importlib.util.spec_from_file_location('_v6_contract_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({{
    'experiment': module.EXPERIMENT_ID,
    'sources': len(module.SOURCE_PATHS),
    'torch': 'torch' in sys.modules,
    'numpy': 'numpy' in sys.modules,
    'PIL': 'PIL' in sys.modules,
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "PIL": False,
        "experiment": contract.EXPERIMENT_ID,
        "numpy": False,
        "sources": 111,
        "torch": False,
    }


def test_exact_frozen_v5_terminal_and_v6_preregistration_bindings() -> None:
    assert contract.v5_terminal_audit_binding() == {
        "path": contract.V5_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": "458f590605178f1460d043a48ed629c181f593a4",
        "file_sha256": (
            "e4c9a329322e641b9c096ae3bc163876991e4d90c1bb24dc48146a2dd30acd20"
        ),
        "content_sha256": (
            "b89afbcfaff1a703bb924f5cc028613bd927c316db5a0c1066bccae3e567526e"
        ),
        "byte_count": 11_272,
        "status": contract.V5_TERMINAL_AUDIT_STATUS,
        "classification": contract.V5_TERMINAL_AUDIT_CLASSIFICATION,
    }
    assert contract.preregistration_binding() == {
        "path": contract.PREREGISTRATION_RELATIVE_PATH,
        "commit": "2ec3c7a2e216544acab6f43b29b113fdc538a74f",
        "file_sha256": (
            "e71dac233d89aa49e97998afdeaadc6c806671945e25101ceb078fcbac0af4e7"
        ),
        "byte_count": 14_618,
    }
    governed = contract.validate_governing_documents(ROOT)
    assert governed[contract.V5_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.V5_TERMINAL_AUDIT_FILE_SHA256
    )
    assert governed[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )


def test_exact_ten_additive_paths_and_111_source_closure() -> None:
    assert len(contract.REUSED_SOURCE_PATHS) == 101
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 10
    assert len(contract.SOURCE_PATHS) == 111
    assert set(contract.REUSED_SOURCE_PATHS).isdisjoint(contract.ADDITIVE_SOURCE_PATHS)
    assert contract.validate_source_manifest(_synthetic_manifest())["source_count"] == 111


def test_v6_paths_aliases_and_fresh_output_root_are_exact() -> None:
    assert contract.FROZEN_V5_CONTRACT_RELATIVE_PATH == (
        "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
        "all_actions_state_delta_contrast.py"
    )
    assert contract.FROZEN_V5_RUNNER_RELATIVE_PATH == v5.RUNNER_RELATIVE_PATH
    assert contract.FROZEN_V5_LAUNCHER_RELATIVE_PATH == v5.LAUNCHER_RELATIVE_PATH
    assert contract.FROZEN_V5_SOURCE_CLOSURE_CHECKER_RELATIVE_PATH == (
        v5.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH
    )
    assert contract.MODEL_RELATIVE_PATH == (
        "lewm/models/direct_egocentric_bev_state_jepa_v6_"
        "phase_separated_frozen_state_prediction.py"
    )
    assert contract.MODEL_RELATIVE_PATH != v5.MODEL_RELATIVE_PATH
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_shared_observable_camera_ray_jepa_v6/"
        "rgb_direct_egocentric_bev_state_jepa_probe_v6_"
        "phase_separated_frozen_state_prediction"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v5.OUTPUT_ROOT_RELATIVE_PATH


def test_v5_A_is_removed_and_frozen_v3_science_is_phase_routed() -> None:
    objective = contract.objective_contract()
    frozen = v5._v4.objective_contract()
    for key in ("G", "J", "C", "wrong_rgb_control", "same_action_target_metrics"):
        assert objective[key] == frozen[key]
    assert "A" not in objective
    assert "C_v3" not in objective
    assert objective["v5_all_actions_state_delta_contrast_A"] == "absent"
    assert objective["phase_one_total"] == "G/log(2)"
    assert objective["phase_two_total"] == "J/log(2)+C"
    assert contract.MODEL_PARAMETER_INVENTORY == v5._v4.MODEL_PARAMETER_INVENTORY
    assert contract.optimizer_contract() == v5._v4.optimizer_contract()
    assert contract.build_schedule_identity() == v5._v4.build_schedule_identity()


def test_preliminary_mode_is_labelled_and_not_v6_phase_evidence() -> None:
    metrics = {
        "three_logit_bottleneck_exact": True,
        "no_hidden_or_auxiliary_bypass": True,
        "prediction_is_exact_persistence": True,
        "all_nine_action_predictions_bitwise_equal": True,
        "target_parameters_gradient_free": True,
        "intended_online_path_gradient_nonzero": True,
        "six_call_graph_isolation_exact": True,
        "all_registered_values_finite": True,
        "action_nll": math.log(9.0),
        "action_macro_balanced_accuracy": 1.0 / 9.0,
    }
    result = contract.evaluate_gate(0, metrics)
    assert result["passed"] is True
    assert result["v6_phase_receipt_ready"] is False
    assert result["gate_mode"] == "PRELIMINARY_INHERITED_V3_NOT_V6_PHASE_EVIDENCE"
    assert result["control"] == contract.CONTROL_CONTINUE_UPDATE_ZERO


def test_final_u0_u100_u400_u1000_pass_exact_registered_contract() -> None:
    update_zero = _update_zero_metrics()
    for update, metrics in (
        (0, update_zero),
        (100, _update_100_metrics()),
        (400, _update_400_metrics()),
        (1_000, _update_1000_metrics()),
    ):
        result = contract.evaluate_gate(
            update,
            metrics,
            update_zero=update_zero if update else None,
        )
        assert result["passed"] is True
        assert result["gate_mode"] == "FINAL_V6_PHASE_RECEIPT"
        assert result["control"] == contract.GATE_CONTROLS[update][1]
        assert result["phase_accounting"] == contract.PHASE_ACCOUNTING[update]


def test_final_gates_are_strict_and_fail_closed() -> None:
    update_zero = _update_zero_metrics()

    false_ready = _update_zero_metrics()
    false_ready["v6_phase_receipt_ready"] = False
    assert contract.evaluate_gate(0, false_ready)["passed"] is False

    u100 = _update_100_metrics()
    u100["G"] = 0.90 * float(update_zero["G"])
    result100 = contract.evaluate_gate(100, u100, update_zero=update_zero)
    assert result100["passed"] is False
    assert result100["control"] == contract.CONTROL_UPDATE_100_FAIL

    u400 = _update_400_metrics()
    u400["aggregate_raster_balanced_accuracy"] = (
        contract.GATE_THRESHOLDS[400][
            "aggregate_raster_balanced_accuracy_strictly_greater_than"
        ]
    )
    assert contract.evaluate_gate(400, u400)["passed"] is False

    u1000 = _update_1000_metrics()
    u1000["C"] = u1000["C_update400_boundary"]
    result1000 = contract.evaluate_gate(1_000, u1000)
    assert result1000["passed"] is False
    assert result1000["control"] == contract.CONTROL_UPDATE_1000_FAIL

    missing = _update_100_metrics()
    missing.pop("phase_one_trainability_exact")
    with pytest.raises(ValueError, match="phase_one_trainability_exact must be bool"):
        contract.evaluate_gate(100, missing, update_zero=update_zero)


@pytest.mark.parametrize("control", contract.FAILURE_CONTROLS)
def test_version_local_failure_chain_accepts_every_v6_failure_control(
    control: str,
) -> None:
    chain = {
        "metrics": control,
        "artifact": control,
        "result": control,
        "completion": control,
    }
    assert contract.validate_failure_status_chain(chain) == chain


def test_version_local_failure_chain_rejects_mismatch_and_pass() -> None:
    control = contract.CONTROL_UPDATE_100_FAIL
    chain = {
        "metrics": control,
        "artifact": control,
        "result": control,
        "completion": control,
    }
    mismatch = dict(chain)
    mismatch["result"] = contract.CONTROL_UPDATE_400_FAIL
    with pytest.raises(ValueError, match="one exact V6 gate control"):
        contract.validate_failure_status_chain(mismatch)
    with pytest.raises(ValueError, match="one exact V6 gate control"):
        contract.validate_failure_status_chain(
            {field: contract.CONTROL_PASS for field in chain}
        )


def test_source_contract_grants_no_execution_or_downstream_authority() -> None:
    assert contract.PRESENT_AUTHORITY == contract.SOURCE_ONLY_AUTHORITY
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert contract.PRESENT_AUTHORITY["generated_input_access_authorized"] is False
    science = contract.science_contract()
    assert science["phase_successor"] == contract.SCIENTIFIC_DELTA
    assert science["phase_successor"]["v5_all_actions_state_delta_contrast_A"] == (
        "absent"
    )
    assert science["gates"]["preliminary_mode_authorizes_execution"] is False
    assert science["lifecycle"]["one_fresh_attempt"] is True
    assert science["lifecycle"]["maximum_updates"] == 1_000
    assert science["lifecycle"]["maximum_presentations"] == 16_000
    for field in (
        "g2_authorized",
        "navigation_authorized",
        "heldout_authorized",
        "sealed_authorized",
        "promotion_authorized",
    ):
        assert science["authority"][field] is False
