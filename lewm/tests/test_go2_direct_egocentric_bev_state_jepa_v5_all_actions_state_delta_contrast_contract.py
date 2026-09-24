from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v5_"
    "all_actions_state_delta_contrast.py"
)
V4_CONTRACT = (
    ROOT
    / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v4_"
    "residual_head_hook_integrity.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _load("_direct_bev_v5_delta_contract_test", CONTRACT)
v4 = _load("_direct_bev_v5_delta_frozen_v4_test", V4_CONTRACT)


def _update_zero_metrics(*, c_value: float) -> dict[str, object]:
    return {
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
        "C": c_value,
    }


def test_frozen_v4_and_preregistration_bindings_are_exact() -> None:
    assert contract.frozen_v4_source_manifest_binding() == {
        "path": contract.FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH,
        "commit": "d82e386c1b846442fa4f2f66d6233ca98380fd74",
        "file_sha256": (
            "299a54b683d6926cf2cec4d3887991d4b5df53b0540a10d936c06679d2dc6d98"
        ),
        "content_sha256": (
            "f78fbed617f7f1319ededbf5ae76cb5f822d146c9d407c10f4379052e6cb97be"
        ),
        "byte_count": 29_853,
        "status": "PASS_SOURCE_CLOSURE",
        "source_count": 91,
    }
    assert contract.frozen_v4_review_binding()["commit"] == (
        "478ca4845249f2ab1d79e09c31c46e88a25c5c89"
    )
    assert contract.frozen_v4_authorization_binding()["commit"] == (
        "d9cc2fad0c1f953487756b34226e03a9607f8d3e"
    )
    assert contract.v4_terminal_audit_binding()["commit"] == (
        "dcd509d9ded153d07c6a4513da328c92398d1b7c"
    )
    assert contract.preregistration_binding() == {
        "path": contract.PREREGISTRATION_RELATIVE_PATH,
        "commit": "5b503a27b1f3ee6f94b0e9ba1cde339b0d007bb8",
        "file_sha256": (
            "215de2dd0978862acf1f527778642a7151abbe75c35ff30d9ee875b196477ad9"
        ),
        "byte_count": 7_007,
    }


def test_governing_documents_validate_source_only() -> None:
    bindings = contract.validate_governing_documents(ROOT)
    assert bindings[contract.FROZEN_V4_SOURCE_MANIFEST_RELATIVE_PATH] == (
        contract.FROZEN_V4_SOURCE_MANIFEST_FILE_SHA256
    )
    assert bindings[contract.V4_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.V4_TERMINAL_AUDIT_FILE_SHA256
    )
    assert bindings[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )


def test_exact_ten_additive_paths_and_101_source_closure() -> None:
    assert len(contract.REUSED_SOURCE_PATHS) == 91
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 10
    assert len(contract.SOURCE_PATHS) == 101
    assert set(contract.ADDITIVE_SOURCE_PATHS) == {
        contract.MODEL_RELATIVE_PATH,
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        contract.MODEL_TEST_RELATIVE_PATH,
        contract.CONTRACT_TEST_RELATIVE_PATH,
        contract.RUNNER_TEST_RELATIVE_PATH,
        contract.LAUNCHER_TEST_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    }
    assert set(contract.REUSED_SOURCE_PATHS).isdisjoint(
        contract.ADDITIVE_SOURCE_PATHS
    )


def test_v5_identities_output_and_frozen_runner_paths_are_exact() -> None:
    assert contract.EXPERIMENT_ID == (
        "go2_rgb_direct_egocentric_bev_state_jepa_v5_"
        "all_actions_state_delta_contrast"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "rgb_direct_egocentric_bev_state_jepa_probe_v5_"
        "all_actions_state_delta_contrast"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v4.OUTPUT_ROOT_RELATIVE_PATH
    assert contract.FROZEN_V4_RUNNER_RELATIVE_PATH == v4.RUNNER_RELATIVE_PATH
    assert contract.FROZEN_V4_LAUNCHER_RELATIVE_PATH == v4.LAUNCHER_RELATIVE_PATH
    assert contract.MODEL_RELATIVE_PATH != v4.MODEL_RELATIVE_PATH


def test_only_objective_changes_model_optimizer_schedule_and_inventory_frozen() -> None:
    assert contract.model_config() == v4.model_config()
    assert contract.optimizer_contract() == v4.optimizer_contract()
    assert contract.build_schedule_identity() == v4.build_schedule_identity()
    assert contract.MODEL_PARAMETER_INVENTORY == v4.MODEL_PARAMETER_INVENTORY
    assert contract.FROZEN_V3_INITIAL_MODEL_STATE_SHA256 == (
        "84748bc66f0639b9dae1c81880f5c0fa756f4c4d9e75d0ffddac1310c7d05d0a"
    )

    frozen = v4.objective_contract()
    objective = contract.objective_contract()
    for key in ("G", "J", "wrong_rgb_control", "same_action_target_metrics"):
        assert objective[key] == frozen[key]
    assert objective["C_v3"] == frozen["C"]
    assert objective["C"]["formula"] == "C_v3+A"
    assert objective["C"]["A_weight"] == 1.0
    assert objective["A"]["both_ema_target_terms_detached"] is True
    assert objective["A"][
        "adds_parameter_buffer_module_state_call_target_call_or_output_head"
    ] is False
    assert objective["A"][
        "uses_raster_pose_depth_odometry_geometry_ray_warp_or_navigation_signal"
    ] is False
    assert objective["total"] == "1*G/log(2) + 1*J/log(2) + 1*C_v5"


@pytest.mark.parametrize("c_value", [1.99, 2.0, 2.01])
def test_update_zero_v5_C_closed_interval_accepts_boundaries(c_value: float) -> None:
    result = contract.evaluate_gate(0, _update_zero_metrics(c_value=c_value))
    assert result["passed"] is True
    assert result["control"] == contract.CONTROL_CONTINUE_UPDATE_ZERO
    assert result["thresholds"] == contract.GATE_THRESHOLDS[0]
    assert result["conjuncts"][
        "v5_C_in_closed_interval_1point99_2point01"
    ] is True


@pytest.mark.parametrize("c_value", [1.989999, 2.010001])
def test_update_zero_v5_C_outside_interval_fails(c_value: float) -> None:
    result = contract.evaluate_gate(0, _update_zero_metrics(c_value=c_value))
    assert result["passed"] is False
    assert result["control"] == contract.CONTROL_UPDATE_ZERO_FAIL


def test_update_100_400_1000_thresholds_are_frozen_v3_v4() -> None:
    for update in (100, 400, 1_000):
        assert contract.GATE_THRESHOLDS[update] == v4.GATE_THRESHOLDS[update]


@pytest.mark.parametrize("control", contract.FAILURE_CONTROLS)
def test_local_failure_status_chain_accepts_every_v5_failure_control(
    control: str,
) -> None:
    chain = {
        "metrics": control,
        "artifact": control,
        "result": control,
        "completion": control,
    }
    assert contract.validate_failure_status_chain(chain) == chain


def test_local_failure_status_chain_rejects_mismatch_and_pass_control() -> None:
    control = contract.CONTROL_UPDATE_100_FAIL
    chain = {
        "metrics": control,
        "artifact": control,
        "result": control,
        "completion": control,
    }
    mismatch = dict(chain)
    mismatch["result"] = contract.CONTROL_UPDATE_400_FAIL
    with pytest.raises(ValueError, match="one exact V5 gate control"):
        contract.validate_failure_status_chain(mismatch)

    pass_chain = {field: contract.CONTROL_PASS for field in chain}
    with pytest.raises(ValueError, match="one exact V5 gate control"):
        contract.validate_failure_status_chain(pass_chain)


def test_source_contract_has_no_execution_or_downstream_authority() -> None:
    assert contract.PRESENT_AUTHORITY == contract.SOURCE_ONLY_AUTHORITY
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert contract.PRESENT_AUTHORITY["generated_input_access_authorized"] is False
    science = contract.science_contract()
    assert "integrity_replacement" not in science
    assert science["objective"] == contract.objective_contract()
    assert science["loss_successor"] == contract.SCIENTIFIC_DELTA
    assert science["lifecycle"]["v4_retry"] is False
    assert science["lifecycle"][
        "v4_checkpoint_tensor_trace_or_runtime_output_reuse"
    ] is False
    for field in (
        "g2_authorized",
        "navigation_authorized",
        "heldout_authorized",
        "sealed_authorized",
        "promotion_authorized",
    ):
        assert science["authority"][field] is False
