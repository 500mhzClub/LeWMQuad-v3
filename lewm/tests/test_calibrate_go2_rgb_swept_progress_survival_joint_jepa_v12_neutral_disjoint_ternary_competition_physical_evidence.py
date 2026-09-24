from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import (
    calibrate_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence
    as runner,
)


def _activity(
    *,
    schema: str,
    suffixes: frozenset[str],
    inventory_sha256: str,
    count: int,
    target: int,
    minimum: int,
) -> dict[str, Any]:
    first = {name: 1 for name in sorted(suffixes)}
    return {
        "schema": schema,
        "update_count": 1_000,
        "online_parameter_count": count,
        "online_parameter_tensor_count": len(suffixes),
        "parameter_suffix_inventory_sha256": inventory_sha256,
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": first,
        "latest_first_active_update": 1,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": minimum,
        "maximum_active_parameter_tensor_count": len(suffixes),
        "minimum_gradient_l2": 0.1,
        "maximum_gradient_l2": 0.2,
        "target_parameter_tensor_count": target,
        "target_gradient_tensor_count": 0,
    }


def _initial_model() -> dict[str, Any]:
    migration = {
        "predecessor_experiment_checkpoint_read": False,
        "all_common_v10_parameter_values_bit_exact": True,
        "all_common_v10_buffer_values_bit_exact": True,
        "online_branch_attention_parameter_count": 14_528,
        "target_branch_attention_parameter_count": 14_528,
        "factorized_semantic_parameter_count": 18_628,
        "online_target_branch_attention_initial_copy_exact": True,
        "target_branch_attention_initial_gradient_tensor_count": 0,
    }
    identity = {
        "schema": "lewm_v12_fresh_v11_zero_parameter_state_identity_v1",
        "predecessor_experiment_checkpoint_read": False,
        "v11_source_migration_witness": migration,
        "v12_parameter_tensor_count": 233,
        "v11_parameter_tensor_count": 233,
        "v12_parameter_count": 6_122_053,
        "v11_parameter_count": 6_122_053,
        "added_parameter_tensor_count": 0,
        "added_parameter_count": 0,
        "all_parameter_values_bit_exact": True,
        "all_buffer_values_bit_exact": True,
        "semantic_axis_modules_reused_without_aliasing": True,
        "neutral_algebra_exact": True,
        "supported_probabilities_finite_and_normalized": True,
        "branch_invalid_evidence_fixed_to_minus_20": True,
        "all_invalid_logits_exact": True,
        "shared_predictor_state_unchanged": True,
        "ema_target_state_unchanged_and_frozen": True,
    }
    return {
        "schema": "lewm_v12_neutral_disjoint_ternary_initial_model_v1",
        "architecture": runner.EXPECTED_ARCHITECTURE_V12,
        "fresh_v11_state_identity": identity,
        "online_branch_attention_parameter_count": 14_528,
        "online_branch_attention_parameter_tensor_count": 14,
        "target_branch_attention_parameter_count": 14_528,
        "target_branch_attention_parameter_tensor_count": 14,
        "factorized_semantic_parameter_count": 18_628,
        "factorized_semantic_parameter_tensor_count": 12,
        "all_v11_parameters_partitioned_exactly_once": True,
        "optimizer_parameter_membership_changed_from_v11": False,
        "target_initial_gradient_tensor_count": 0,
        "initial_hard_sync_count": 1,
        "initial_ema_update_count": 0,
    }


def _gate() -> dict[str, Any]:
    return {
        "status": "PASS_FULL_ARM",
        "passed": True,
        "checks": {name: True for name in runner.EXPECTED_GATE_CHECKS_V12},
        "failed_checks": [],
        "thresholds": dict(runner.EXPECTED_GATE_THRESHOLDS_V12),
    }


def _candidate_result() -> dict[str, Any]:
    gate = _gate()
    branch = _activity(
        schema="lewm_v11_height_role_branch_attention_training_activity_v1",
        suffixes=runner.ATTENTION_PARAMETER_SUFFIXES_V12,
        inventory_sha256=runner.ATTENTION_PARAMETER_INVENTORY_SHA256_V12,
        count=14_528,
        target=14,
        minimum=14,
    )
    semantic = _activity(
        schema="lewm_v11_factorized_semantic_axes_training_activity_v1",
        suffixes=runner.SEMANTIC_PARAMETER_SUFFIXES_V12,
        inventory_sha256=runner.SEMANTIC_PARAMETER_INVENTORY_SHA256_V12,
        count=18_628,
        target=0,
        minimum=8,
    )
    diagnostics = {
        "height_role_branch_attention": branch,
        "factorized_semantic_axes": semantic,
        "v12_contract": {
            "schema": "lewm_v12_unchanged_joint_training_contract_v1",
            "training_helper": (
                "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
                "height_role_factorized_evidence_lift"
            ),
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_weight": False,
            "height_role_branch_attention": branch,
            "factorized_semantic_axes": semantic,
        },
    }
    core = {
        "access": {
            "forbidden_input_count": 0,
            "g2_navigation_final_evaluation_open_count": 0,
            "narrow_loader": {
                "rgb_request_count": {"fixed_negative": 0},
                "forbidden_semantic_counters": {"general_raw": 0},
            },
        },
        "action_prior_mean_progress_m": [],
        "authority": {
            "development_only": True,
            "g2_navigation_final_evaluation_opened": False,
            "heldout_or_sealed_opened": False,
            "physical_calibration_run": False,
            "physical_evidence_gate_passed": False,
            "checkpoint_qualified": False,
            "promotion_performed": False,
            "retry_or_resume_authorized": False,
            "checkpoint_access_authorized_for_physical_calibration": False,
            "separate_physical_preregistration_required": True,
        },
        "caps": {
            "updates": 1_000,
            "microbatch_graphs": 4_000,
            "presentations": 16_000,
        },
        "determinism": {},
        "full_arm_gate": gate,
        "gate": gate,
        "hardware": {},
        "label_manifest": {},
        "masks": {},
        "n320": {
            "encoder_only_initialization": True,
            "predecessor_experiment_checkpoint_read": False,
        },
        "physical_evidence_calibration": {
            "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT",
            "physical_calibration_run_in_this_attempt": False,
            "requires_full_arm_pass": True,
            "protocol_changed_from_reviewed_v4_calibration": False,
            "threshold_tuple_count": 2_016,
            "physical_gate_passed": False,
            "schema": "lewm_v12_unchanged_physical_calibration_stage_v1",
            "source": "numerically_unchanged_v10_v4_2016_tuple_protocol",
            "v10_directional_baselines_are_interpretation_only": True,
            "physical_calibration_authorized_in_this_attempt": False,
        },
        "preregistration_commit": runner.CANDIDATE_PREREGISTRATION_COMMIT,
        "roles": {},
        "schedule_prefix_sha256": (
            "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528"
        ),
        "schema": runner.CANDIDATE_RESULT_SCHEMA,
        "scientific_change_from_v11": {
            "only_change": "neutral_disjoint_ternary_semantic_algebra",
            "initial_v12_model": _initial_model(),
            "architecture": runner.EXPECTED_ARCHITECTURE_V12,
            "objective": "S+P+U+R+O",
            "inherited_occupied_auxiliary": runner.EXPECTED_AUXILIARY_OBJECTIVE_V12,
            "model_code_changed": True,
            "parameter_or_buffer_state_changed": False,
            "added_parameter_count": 0,
            "data_changed": False,
            "dataset_identity_changed": False,
            "input_tensorization_changed": False,
            "optimizer_rules_changed": False,
            "optimizer_parameter_tensor_membership_changed": False,
            "loss_source_or_coefficient_changed": False,
            "loss_gradient_surface_changed_by_registered_semantic_algebra": True,
            "new_loss_or_loss_weight": False,
            "schedule_changed": False,
            "evaluation_changed": False,
        },
        "seeds": dict(runner.EXPECTED_SEEDS_V12),
        "selection_control_comparisons": {},
        "selection_semantic": {},
        "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION",
        "training": {
            "accounting": dict(runner.EXPECTED_ACCOUNTING_V12),
            "checkpoint": {
                "path": "checkpoint_update_1000.pt",
                "byte_count": runner.CANDIDATE_CHECKPOINT_BYTE_COUNT,
                "file_sha256": runner.CANDIDATE_CHECKPOINT_SHA256,
            },
            "checkpoint_access_status": "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION",
            "core": (
                "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v11_"
                "height_role_factorized_evidence_lift"
            ),
            "diagnostics": diagnostics,
            "factorized_semantic_axes_activity": semantic,
            "height_role_branch_attention_activity": branch,
            "joint_from_update_one": True,
            "separate_head_or_predictor_training": False,
            "trace": {},
        },
        "wrong_rgb_mapping_sha256": {},
    }
    return runner._hashed(core)


def _raw(receipt: dict[str, Any]) -> bytes:
    return runner._canonical_bytes(receipt) + b"\n"


def _bind_result(
    monkeypatch: pytest.MonkeyPatch, raw: bytes, receipt: dict[str, Any]
) -> None:
    monkeypatch.setattr(runner, "CANDIDATE_RESULT_BYTE_COUNT", len(raw))
    monkeypatch.setattr(
        runner, "CANDIDATE_RESULT_FILE_SHA256", hashlib.sha256(raw).hexdigest()
    )
    monkeypatch.setattr(
        runner, "CANDIDATE_RESULT_CONTENT_SHA256", receipt["content_sha256"]
    )


def test_runner_is_a_direct_v4_protocol_adapter() -> None:
    for name in (
        "_canonical_bytes",
        "_content_sha256",
        "_hashed",
        "_parse_canonical",
        "_read_regular",
        "_atomic_write",
        "_write_json",
        "_build_data_boundary",
        "_collect_role",
        "_fit_select_score",
        "_raw_access_snapshot",
    ):
        assert getattr(runner, name) is getattr(runner._v4, name)
    assert runner.ROLE_COUNTS == {
        "probability_calibration": 415,
        "checkpoint_selection": 495,
    }
    assert runner.ROLE_CELL_COUNTS == {
        "probability_calibration": 1_699_840,
        "checkpoint_selection": 2_027_520,
    }
    assert len(runner.EXPECTED_GATE_CHECKS_V12) == 24
    assert runner.PREREGISTRATION_COMMIT == (
        "c63e98162a1b03a33225e6e0a04b67a357c7ed89"
    )


def test_candidate_result_binds_v12_state_activity_and_closed_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _candidate_result()
    monkeypatch.setattr(
        runner, "CANDIDATE_RESULT_CONTENT_SHA256", receipt["content_sha256"]
    )
    runner._validate_candidate_result_v12(receipt)

    receipt["scientific_change_from_v11"]["initial_v12_model"][
        "fresh_v11_state_identity"
    ]["neutral_algebra_exact"] = False
    with pytest.raises(PermissionError, match="fresh-state identity"):
        runner._validate_candidate_result_v12(receipt)


def test_result_is_validated_before_exactly_one_checkpoint_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = _candidate_result()
    checkpoint = b"synthetic-v12-checkpoint"
    checksum = hashlib.sha256(checkpoint).hexdigest()
    monkeypatch.setattr(runner, "CANDIDATE_CHECKPOINT_BYTE_COUNT", len(checkpoint))
    monkeypatch.setattr(runner, "CANDIDATE_CHECKPOINT_SHA256", checksum)
    receipt["training"]["checkpoint"] = {
        "path": "checkpoint_update_1000.pt",
        "byte_count": len(checkpoint),
        "file_sha256": checksum,
    }
    receipt = runner._hashed(
        {name: value for name, value in receipt.items() if name != "content_sha256"}
    )
    result_raw = _raw(receipt)
    _bind_result(monkeypatch, result_raw, receipt)
    reads: list[str] = []

    def read_regular(path: Path) -> bytes:
        reads.append(path.name)
        return result_raw if path.name == "result.json" else checkpoint

    model = object()
    adapter = SimpleNamespace(
        PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT=runner.PREREGISTRATION_COMMIT,
        load_checkpoint=lambda raw: model if raw == checkpoint else None,
    )
    monkeypatch.setattr(runner, "_read_regular", read_regular)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: adapter)
    access = runner._new_access_v12()
    assert runner._load_candidate_v12(tmp_path, access) is model
    assert reads == ["result.json", "checkpoint_update_1000.pt"]
    assert access["candidate_result_validations"] == 1
    assert access["candidate_checkpoint_read_attempts"] == 1
    assert access["candidate_checkpoint_load_successes"] == 1


def test_invalid_result_cannot_reach_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = _candidate_result()
    receipt["authority"]["checkpoint_access_authorized_for_physical_calibration"] = True
    receipt = runner._hashed(
        {name: value for name, value in receipt.items() if name != "content_sha256"}
    )
    result_raw = _raw(receipt)
    _bind_result(monkeypatch, result_raw, receipt)
    reads: list[str] = []

    def read_regular(path: Path) -> bytes:
        reads.append(path.name)
        if path.name != "result.json":
            raise AssertionError("checkpoint read before admission")
        return result_raw

    monkeypatch.setattr(runner, "_read_regular", read_regular)
    with pytest.raises(PermissionError, match="terminal authority"):
        runner._load_candidate_v12(tmp_path, runner._new_access_v12())
    assert reads == ["result.json"]


def test_source_closure_requires_adapter_binding_and_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "ADAPTER_SOURCE_SHA256", None)
    with pytest.raises(PermissionError, match="adapter source is not frozen"):
        runner._expected_sources_v12()
    payloads = {"base.py": b"base", "adapter.py": b"adapter"}
    monkeypatch.setattr(
        runner,
        "SOURCE_SHA256",
        {"base.py": hashlib.sha256(payloads["base.py"]).hexdigest()},
    )
    monkeypatch.setattr(runner, "ADAPTER_SOURCE_RELATIVE_PATH", "adapter.py")
    monkeypatch.setattr(
        runner,
        "ADAPTER_SOURCE_SHA256",
        hashlib.sha256(payloads["adapter.py"]).hexdigest(),
    )
    monkeypatch.setattr(runner, "_read_regular", lambda path: payloads[path.name])
    assert runner._validate_sources_v12(tmp_path) == runner._expected_sources_v12()
    payloads["adapter.py"] = b"changed"
    with pytest.raises(PermissionError, match="dependency source changed"):
        runner._validate_sources_v12(tmp_path)


class _Inputs:
    def __init__(self) -> None:
        self.consumed = {
            "cal": {"kind": "development_rgb", "roles": ["probability_calibration"]},
            "sel": {"kind": "development_rgb", "roles": ["checkpoint_selection"]},
        }

    def role_pairs(self, role: str) -> list[dict[str, str]]:
        return [{"role": role}]


class _Loader:
    def receipt(self) -> dict[str, Any]:
        return {
            "raw_inputs_frame_attribute_invocation_count": 0,
            "forbidden_semantic_counters": {"general_raw": 0},
        }

    def model_facing_access_counts(self) -> dict[str, int]:
        total = sum(runner.ROLE_COUNTS.values())
        return {
            "endpoint_rgb_row_request_count": total,
            "raster_label_row_request_count": total,
            "current_rgb_row_request_count": 0,
            "next_rgb_row_request_count": 0,
            "fixed_negative_rgb_row_request_count": 0,
        }


def _science(passed: bool) -> dict[str, Any]:
    return {
        "calibration": {
            "schema": "fixture",
            "content_sha256": "a" * 64,
            "id": "fixture-id",
        },
        "threshold_selection": {"candidate_count": 2_016},
        "selection": {},
        "gate": {
            "status": (
                "PASS_DEVELOPMENT_PHYSICAL_EVIDENCE"
                if passed
                else "FAIL_DEVELOPMENT_PHYSICAL_EVIDENCE"
            ),
            "passed": passed,
            "checks": {"fixture": passed},
            "failed_checks": [] if passed else ["fixture"],
        },
    }


@pytest.mark.parametrize("passed", [True, False])
def test_execute_uses_two_roles_one_v4_fit_and_no_forbidden_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE_PATH", "output/attempt_v1")
    monkeypatch.setattr(runner, "_validate_sources_v12", lambda root: {"x": "y"})

    def load(root: Path, access: dict[str, int]) -> object:
        for name in (
            "candidate_result_read_attempts",
            "candidate_result_read_successes",
            "candidate_result_validations",
            "candidate_checkpoint_read_attempts",
            "candidate_checkpoint_read_successes",
            "candidate_checkpoint_load_attempts",
            "candidate_checkpoint_load_successes",
        ):
            access[name] += 1
        return object()

    monkeypatch.setattr(runner, "_load_candidate_v12", load)
    inputs, loader = _Inputs(), _Loader()
    monkeypatch.setattr(
        runner,
        "_build_data_boundary",
        lambda root: (
            SimpleNamespace(torch=object()),
            inputs,
            loader,
            {"_raw_constructor_reads": {}},
        ),
    )
    roles: list[str] = []

    def collect(model: Any, loader: Any, pairs: Any, *, role: str, torch: Any):
        roles.append(role)
        return (
            object(),
            object(),
            {
                "role": role,
                "pair_count": runner.ROLE_COUNTS[role],
                "cell_count": runner.ROLE_CELL_COUNTS[role],
                "next_endpoint_order_sha256": role[0] * 64,
                "all_cells_used": True,
            },
        )

    monkeypatch.setattr(runner, "_collect_role", collect)

    def fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        counts = kwargs["operation_counts"]
        counts["calibration_fit_calls"] += 1
        counts["threshold_selection_calls"] += 1
        assert kwargs["provenance"]["all_cells_used"] is True
        return _science(passed)

    monkeypatch.setattr(runner, "_fit_select_score", fit)
    result = runner.execute_v12(repository_root=tmp_path)
    assert roles == ["probability_calibration", "checkpoint_selection"]
    assert result["gate"]["passed"] is passed
    assert result["access"]["calibration_fit_calls"] == 1
    assert result["access"]["threshold_selection_calls"] == 1
    for name in (
        "model_backward_calls",
        "model_optimizer_steps",
        "model_ema_steps",
        "predictor_calls",
        "g2_operations",
        "navigation_operations",
        "heldout_reads",
        "sealed_reads",
    ):
        assert result["access"][name] == 0
    assert result["authority"]["g2_binding_preparation_authorized"] is passed


def test_source_has_one_science_call_and_no_training_trace_or_g2() -> None:
    source = Path(runner.__file__).read_text()
    assert "predict_all_actions" not in source
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "torch.load(" not in source
    assert "training_trace.json" not in source
    assert "fit_hierarchical_probability_calibration(" not in source
    assert "select_conservative_thresholds(" not in source
    assert "evaluation_mask" not in source
    assert source.count("_fit_select_score(") == 1
    assert source.count("_collect_role(") == 1
