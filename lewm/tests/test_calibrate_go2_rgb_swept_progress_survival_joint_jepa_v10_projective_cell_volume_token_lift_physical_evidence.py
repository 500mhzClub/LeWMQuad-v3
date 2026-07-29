from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from scripts import (
    calibrate_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence
    as runner,
)


def _activity() -> dict[str, Any]:
    return {
        "schema": "lewm_v10_cell_volume_attention_training_activity_v1",
        "update_count": 1_000,
        "online_parameter_count": 16_576,
        "online_parameter_tensor_count": 7,
        "parameter_suffix_inventory_sha256": "a" * 64,
        "all_online_parameter_tensors_active_by_update_2": True,
        "first_active_update": {
            name: 1 for name in runner.ATTENTION_PARAMETER_SUFFIXES_V10
        },
        "latest_first_active_update": 1,
        "active_update_count": 1_000,
        "minimum_active_parameter_tensor_count": 7,
        "maximum_active_parameter_tensor_count": 7,
        "minimum_gradient_l2": 0.1,
        "maximum_gradient_l2": 0.2,
        "target_gradient_tensor_count": 0,
        "implementation": "unchanged_v9_attention_gradient_receipts",
    }


def _gate() -> dict[str, Any]:
    return {
        "status": "PASS_FULL_ARM",
        "passed": True,
        "checks": {name: True for name in runner.EXPECTED_GATE_CHECKS_V10},
        "failed_checks": [],
        "thresholds": dict(runner.EXPECTED_GATE_THRESHOLDS_V10),
    }


def _candidate_result() -> dict[str, Any]:
    gate = _gate()
    activity = _activity()
    core = {
        "access": {
            "forbidden_input_count": 0,
            "g2_navigation_final_evaluation_open_count": 0,
        },
        "action_prior_mean_progress_m": [],
        "authority": {
            "checkpoint_access_authorized_for_physical_calibration": True,
            "checkpoint_qualified": False,
            "development_only": True,
            "g2_navigation_final_evaluation_opened": False,
            "heldout_or_sealed_opened": False,
            "physical_evidence_gate_passed": False,
            "promotion_performed": False,
            "retry_or_resume_authorized": False,
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
            "schema": "lewm_v10_unchanged_physical_calibration_stage_v1",
            "source": "unchanged_v9_v4_2016_tuple_protocol",
            "physical_calibration_run_in_this_attempt": False,
            "physical_gate_passed": False,
            "protocol_changed_from_reviewed_v4_calibration": False,
            "requires_full_arm_pass": True,
            "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT",
            "threshold_tuple_count": 2_016,
        },
        "preregistration_commit": runner.CANDIDATE_PREREGISTRATION_COMMIT,
        "roles": {},
        "schedule_prefix_sha256": "b" * 64,
        "schema": runner.CANDIDATE_RESULT_SCHEMA,
        "scientific_change_from_v9": {
            "only_change": (
                "projective_cell_volume_token_lift_geometry_and_masked_mean_base"
            ),
            "initial_v10_model": {},
            "architecture": {},
            "objective": "S+P+U+R+O",
            "inherited_occupied_auxiliary": {},
            "model_changed": True,
            "data_changed": False,
            "dataset_identity_changed": False,
            "input_tensorization_changed": False,
            "inherited_parameter_state_bit_exact": True,
            "optimizer_rules_changed": False,
            "optimizer_parameter_tensor_membership_changed": False,
            "losses_changed": False,
            "new_loss_or_head": False,
            "schedule_changed": False,
            "evaluation_changed": False,
        },
        "seeds": {},
        "selection_control_comparisons": {},
        "selection_semantic": {},
        "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION",
        "training": {
            "accounting": dict(runner.EXPECTED_ACCOUNTING_V10),
            "checkpoint": {
                "path": "checkpoint_update_1000.pt",
                "byte_count": runner.CANDIDATE_CHECKPOINT_BYTE_COUNT,
                "file_sha256": runner.CANDIDATE_CHECKPOINT_SHA256,
            },
            "checkpoint_access_status": (
                "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION"
            ),
            "core": (
                "unchanged scripts.run_go2_rgb_swept_progress_survival_joint_"
                "jepa_v9_content_adaptive_dense_local_token_lift"
            ),
            "cell_volume_attention_activity": activity,
            "diagnostics": {
                "dense_local_attention": activity,
                "v10_contract": {
                    "schema": "lewm_v10_unchanged_joint_training_contract_v1",
                    "objective": "S+P+U+R+O",
                    "occupied_auxiliary_coefficient": 0.5,
                    "new_loss_or_head": False,
                    "training_core": "unchanged_v9_wrapper_over_v3_v4",
                },
            },
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


def test_runner_directly_aliases_every_v4_scientific_procedure() -> None:
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
    assert runner.ROLE_COUNTS is runner._v4.ROLE_COUNTS
    assert runner.ROLE_CELL_COUNTS is runner._v4.ROLE_CELL_COUNTS
    assert runner.FREE_CANDIDATES is runner._v4.FREE_CANDIDATES
    assert runner.OCCUPIED_CANDIDATES is runner._v4.OCCUPIED_CANDIDATES
    assert runner.UNKNOWN_CANDIDATES is runner._v4.UNKNOWN_CANDIDATES
    assert (
        runner.OCCUPIED_DETECTION_CANDIDATES
        is runner._v4.OCCUPIED_DETECTION_CANDIDATES
    )
    assert runner.PREREGISTRATION_COMMIT == (
        "6bc4dca93daf0e220bbaa4fc524470addb880e21"
    )


def test_candidate_result_contract_is_exactly_v10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _candidate_result()
    monkeypatch.setattr(
        runner, "CANDIDATE_RESULT_CONTENT_SHA256", receipt["content_sha256"]
    )
    runner._validate_candidate_result_v10(receipt)
    receipt["physical_evidence_calibration"]["schema"] = "wrong"
    with pytest.raises(PermissionError, match="physical-calibration staging"):
        runner._validate_candidate_result_v10(receipt)


def test_result_is_validated_before_single_checkpoint_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = _candidate_result()
    checkpoint = b"synthetic-checkpoint"
    monkeypatch.setattr(runner, "CANDIDATE_CHECKPOINT_BYTE_COUNT", len(checkpoint))
    monkeypatch.setattr(
        runner, "CANDIDATE_CHECKPOINT_SHA256", hashlib.sha256(checkpoint).hexdigest()
    )
    receipt["training"]["checkpoint"]["byte_count"] = len(checkpoint)
    receipt["training"]["checkpoint"]["file_sha256"] = hashlib.sha256(
        checkpoint
    ).hexdigest()
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
        load_checkpoint=lambda encoded: model if encoded == checkpoint else None,
    )
    monkeypatch.setattr(runner, "_read_regular", read_regular)
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: adapter if name == runner.ADAPTER_MODULE else None,
    )
    access = runner._new_access_v10()
    assert runner._load_candidate_v10(tmp_path, access) is model
    assert reads == ["result.json", "checkpoint_update_1000.pt"]
    assert access["candidate_result_validations"] == 1
    assert access["candidate_checkpoint_read_attempts"] == 1
    assert access["candidate_checkpoint_load_successes"] == 1


def test_invalid_result_cannot_reach_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = _candidate_result()
    receipt["authority"]["checkpoint_access_authorized_for_physical_calibration"] = (
        False
    )
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
    access = runner._new_access_v10()
    with pytest.raises(PermissionError, match="terminal authority"):
        runner._load_candidate_v10(tmp_path, access)
    assert reads == ["result.json"]
    assert access["candidate_checkpoint_read_attempts"] == 0


def test_source_closure_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert len(runner.SOURCE_SHA256) == 12
    assert (
        "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v10_"
        "projective_cell_volume_token_lift.py"
    ) in runner.SOURCE_SHA256
    assert "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py" in (
        runner.SOURCE_SHA256
    )
    assert "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py" in (
        runner.SOURCE_SHA256
    )
    payloads = {"a.py": b"a", "b.py": b"b"}
    monkeypatch.setattr(
        runner,
        "SOURCE_SHA256",
        {
            name: hashlib.sha256(value).hexdigest()
            for name, value in payloads.items()
        },
    )
    monkeypatch.setattr(runner, "_read_regular", lambda path: payloads[path.name])
    assert runner._validate_sources_v10(tmp_path) == runner.SOURCE_SHA256
    payloads["a.py"] = b"changed"
    with pytest.raises(PermissionError, match="dependency source changed"):
        runner._validate_sources_v10(tmp_path)


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
            "forbidden_semantic_counters": {
                "general_raw_frame_loader_call_count": 0,
                "other_supervision_array_open_count": 0,
            },
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


def _mock_science(passed: bool) -> dict[str, Any]:
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
def test_execute_uses_two_roles_one_fit_all_cells_and_no_forbidden_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, passed: bool
) -> None:
    monkeypatch.setattr(runner, "OUTPUT_RELATIVE_PATH", "output/attempt_v1")
    monkeypatch.setattr(runner, "_validate_sources_v10", lambda root: {"x": "y"})

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

    monkeypatch.setattr(runner, "_load_candidate_v10", load)
    inputs, loader = _Inputs(), _Loader()
    monkeypatch.setattr(
        runner,
        "_build_data_boundary",
        lambda root: (
            SimpleNamespace(torch=torch),
            inputs,
            loader,
            {"_raw_constructor_reads": {}},
        ),
    )
    roles: list[str] = []

    def collect(model: Any, loader: Any, pairs: Any, *, role: str, torch: Any):
        roles.append(role)
        return (
            torch.zeros((1, 3, 2, 2)),
            torch.zeros((1, 2, 2), dtype=torch.long),
            {
                "role": role,
                "pair_count": runner.ROLE_COUNTS[role],
                "cell_count": runner.ROLE_CELL_COUNTS[role],
                "next_endpoint_order_sha256": role[0] * 64,
                "batch_count": 1,
                "model_state_mutated": False,
            },
        )

    monkeypatch.setattr(runner, "_collect_role", collect)

    def fit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        counts = kwargs["operation_counts"]
        counts["calibration_fit_calls"] += 1
        counts["threshold_selection_calls"] += 1
        assert kwargs["provenance"]["all_cells_used"] is True
        return _mock_science(passed)

    monkeypatch.setattr(runner, "_fit_select_score", fit)
    result = runner.execute_v10(repository_root=tmp_path)
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


def test_source_does_not_reimplement_science_or_mask_calibration_cells() -> None:
    source = Path(runner.__file__).read_text()
    assert "predict_all_actions" not in source
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "fit_hierarchical_probability_calibration(" not in source
    assert "select_conservative_thresholds(" not in source
    assert "evaluation_mask" not in source
    assert source.count("_fit_select_score(") == 1
    assert source.count("_collect_role(") == 1
