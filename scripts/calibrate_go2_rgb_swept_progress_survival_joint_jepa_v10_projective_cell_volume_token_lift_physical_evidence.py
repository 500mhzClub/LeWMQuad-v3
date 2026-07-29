#!/usr/bin/env python3
"""One-shot V10 physical-evidence calibration using the frozen V4 protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence
    as _v4,
)


PREREGISTRATION_COMMIT = "6bc4dca93daf0e220bbaa4fc524470addb880e21"
REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT = (
    "e983e0abd9349426f69262563e12d90a4488180e"
)
TERMINAL_RESULT_COMMIT = "7ccb9cc88f1ddfa687a6d9b5cef847bbb3f11cfe"
CANDIDATE_PREREGISTRATION_COMMIT = (
    "b9eaae6560c42e588c86fb8bf949cc95bd9e29e9"
)
ADAPTER_MODULE = (
    "lewm.benchmarks.go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_physical_evidence_adapter"
)

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_physical_evidence_calibration/"
    "attempt_v1"
)
CANDIDATE_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift/attempt_v1"
)
CANDIDATE_RESULT_BYTE_COUNT = 70_550
CANDIDATE_RESULT_FILE_SHA256 = (
    "f62fa6c908fe8cfb4ae838878d40b615e14ad343d5f123c1dd24e16f274bbb70"
)
CANDIDATE_RESULT_CONTENT_SHA256 = (
    "01ce5f55d3b2cc264b21a9924d27e64568873dfaf2a2364e1448991adda0b6b6"
)
CANDIDATE_CHECKPOINT_BYTE_COUNT = 29_741_203
CANDIDATE_CHECKPOINT_SHA256 = (
    "f63a037868de1e4db465fb4f85af2b8e6eba9883880c19d908216db20d82faa0"
)
CANDIDATE_RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_result_v1"
)
CANDIDATE_CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_checkpoint_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_physical_evidence_calibration_"
    "result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v10_"
    "projective_cell_volume_token_lift_physical_evidence_calibration_"
    "failure_v1"
)

ROLE_COUNTS = _v4.ROLE_COUNTS
ROLE_CELL_COUNTS = _v4.ROLE_CELL_COUNTS
BATCH_SIZE = _v4.BATCH_SIZE
FREE_CANDIDATES = _v4.FREE_CANDIDATES
OCCUPIED_CANDIDATES = _v4.OCCUPIED_CANDIDATES
UNKNOWN_CANDIDATES = _v4.UNKNOWN_CANDIDATES
OCCUPIED_DETECTION_CANDIDATES = _v4.OCCUPIED_DETECTION_CANDIDATES

# Direct aliases keep every scientific and custody procedure on the reviewed
# V4 implementation. V10 adapts only candidate validation/loading and receipts.
_canonical_bytes = _v4._canonical_bytes
_content_sha256 = _v4._content_sha256
_hashed = _v4._hashed
_parse_canonical = _v4._parse_canonical
_read_regular = _v4._read_regular
_atomic_write = _v4._atomic_write
_write_json = _v4._write_json
_build_data_boundary = _v4._build_data_boundary
_collect_role = _v4._collect_role
_fit_select_score = _v4._fit_select_score
_raw_access_snapshot = _v4._raw_access_snapshot

SOURCE_SHA256 = {
    "lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_adapter.py": (
        "03b6d8e0e69e31adb1a9bf4b8227769b4440de8f589d1b066efb36a40d45b414"
    ),
    "lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v4_g2_adapter.py": (
        "1ddbfd743d89614932823ae2247534ac6a76e2eaaf031911617a9311562b4b58"
    ),
    "lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py": (
        "68ade72ef4293bd23136ad739c269af360a962c901216c996c2247d494a88196"
    ),
    "scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v4_physical_evidence.py": (
        "cee7c9c70e6bb9d2bacc6528ef77d009c80e2f484400de9f6445ebfd0c010313"
    ),
    "lewm/hierarchical_probability_calibration.py": (
        "2a41a69d4bf981415f3c3ae6c437e78b3c07e781a603602f7ca58e4e6f785f2b"
    ),
    "lewm/benchmarks/traversability_metrics.py": (
        "97be0acb1a9cf6e170db90945c908a1a30b2ce0a230a5664024b8c06edd03396"
    ),
    "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py": (
        "79e66a4ca5bd814030f374413e4ac0a2edda2552d0614ec23b54b6b0e52ff1b6"
    ),
    "scripts/run_go2_direct_egocentric_bev_state_jepa_v1.py": (
        "33617086a5481f2fa0bf8ae6993110c40bf8db85f066d1d6e874dde12fb07000"
    ),
    "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py": (
        "ce256dcb1ef67dff313855680365ce07d867aca986dfcad7b8e9493373fe099c"
    ),
    "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py": (
        "8c35f0cbafe78185ac74d4412914c177de20f899b0f009a9b9dc7aafdf7695a5"
    ),
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py": (
        "e98bd8cceed26288ebcbf8a02eac03c72be6d06a539953927754353e049a5578"
    ),
    "lewm/benchmarks/go2_shared_jepa_v5_matched_training_v1.py": (
        "53a7fac793a1b46764d49e7259fd637ec02b20111927effd01cdcd09682c206a"
    ),
}

EXPECTED_ACCOUNTING_V10 = {
    "backward_calls": 4_000,
    "ema_steps": 1_000,
    "microbatch_graphs": 4_000,
    "optimizer_steps": 1_000,
    "predictor_forwards": 4_000,
    "predictor_objectives": 4_000,
    "presentations": 16_000,
    "updates": 1_000,
}
EXPECTED_GATE_THRESHOLDS_V10 = {
    "family_informative_utility_min": 0.70,
    "family_pair_concordance_min": 0.60,
    "family_selected_zero_prefix_rate_max": 0.20,
    "informative_utility_min": 0.85,
    "pair_concordance_min": 0.75,
    "positive_control_family_count_min": 6,
    "selected_zero_prefix_rate_max": 0.05,
    "semantic_balanced_accuracy_min": 0.80,
    "semantic_free_recall_min": 0.85,
    "semantic_occupied_recall_min": 0.70,
    "semantic_rough_occupied_recall_min": 0.65,
    "semantic_unknown_recall_min": 0.90,
}
_CONTROL_NAMES_V10 = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "train_action_mean_prior",
    "wrong_rgb",
)
EXPECTED_GATE_CHECKS_V10 = frozenset(
    {
        "all_family_pair_concordance",
        "all_family_utility",
        "all_family_zero_prefix_rate",
        "selection_informative_utility",
        "selection_pair_concordance",
        "selection_registered_families",
        "selection_zero_prefix_rate",
        "semantic_balanced_accuracy",
        "semantic_free_recall",
        "semantic_occupied_recall",
        "semantic_rough_occupied_recall",
        "semantic_unknown_recall",
    }
    | {
        f"{control}:{suffix}"
        for control in _CONTROL_NAMES_V10
        for suffix in (
            "positive_bootstrap_lower_95",
            "positive_equal_scene_delta",
            "positive_family_count",
        )
    }
)
EXPECTED_RESULT_TOP_LEVEL_KEYS_V10 = frozenset(
    {
        "access",
        "action_prior_mean_progress_m",
        "authority",
        "caps",
        "content_sha256",
        "determinism",
        "full_arm_gate",
        "gate",
        "hardware",
        "label_manifest",
        "masks",
        "n320",
        "physical_evidence_calibration",
        "preregistration_commit",
        "roles",
        "schedule_prefix_sha256",
        "schema",
        "scientific_change_from_v9",
        "seeds",
        "selection_control_comparisons",
        "selection_semantic",
        "status",
        "training",
        "wrong_rgb_mapping_sha256",
    }
)
ATTENTION_PARAMETER_SUFFIXES_V10 = frozenset(
    {
        "query_projection.weight",
        "query_projection.bias",
        "key_projection.weight",
        "value_projection.weight",
        "value_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    }
)


def _fresh_output_v10(repository_root: Path) -> Path:
    output = repository_root / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh V10 physical-evidence attempt_v1 exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_sources_v10(repository_root: Path) -> Mapping[str, str]:
    observed = {
        relative: hashlib.sha256(
            _read_regular(repository_root / relative)
        ).hexdigest()
        for relative in SOURCE_SHA256
    }
    if observed != SOURCE_SHA256:
        raise PermissionError("frozen V10 calibration dependency source changed")
    return observed


def _validate_attention_activity_v10(activity: Any) -> None:
    if type(activity) is not dict:
        raise PermissionError("V10 attention activity receipt changed")
    first_active = activity.get("first_active_update")
    if (
        activity.get("schema")
        != "lewm_v10_cell_volume_attention_training_activity_v1"
        or activity.get("implementation")
        != "unchanged_v9_attention_gradient_receipts"
        or activity.get("update_count") != 1_000
        or activity.get("online_parameter_count") != 16_576
        or activity.get("online_parameter_tensor_count") != 7
        or activity.get("all_online_parameter_tensors_active_by_update_2") is not True
        or activity.get("latest_first_active_update") not in (1, 2)
        or activity.get("active_update_count") != 1_000
        or activity.get("minimum_active_parameter_tensor_count") != 7
        or activity.get("maximum_active_parameter_tensor_count") != 7
        or activity.get("target_gradient_tensor_count") != 0
        or type(first_active) is not dict
        or set(first_active) != ATTENTION_PARAMETER_SUFFIXES_V10
        or any(update not in (1, 2) for update in first_active.values())
    ):
        raise PermissionError("V10 attention activity receipt changed")


def _validate_candidate_result_v10(receipt: Mapping[str, Any]) -> None:
    if type(receipt) is not dict or set(receipt) != EXPECTED_RESULT_TOP_LEVEL_KEYS_V10:
        raise PermissionError("V10 result top-level inventory changed")
    gate = receipt.get("gate")
    training = receipt.get("training")
    authority = receipt.get("authority")
    physical = receipt.get("physical_evidence_calibration")
    access = receipt.get("access")
    science = receipt.get("scientific_change_from_v9")
    if (
        receipt.get("schema") != CANDIDATE_RESULT_SCHEMA
        or receipt.get("status")
        != "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION"
        or receipt.get("content_sha256") != CANDIDATE_RESULT_CONTENT_SHA256
        or receipt.get("preregistration_commit")
        != CANDIDATE_PREREGISTRATION_COMMIT
        or type(gate) is not dict
        or receipt.get("full_arm_gate") != gate
        or gate.get("status") != "PASS_FULL_ARM"
        or gate.get("passed") is not True
        or gate.get("failed_checks") != []
        or gate.get("thresholds") != EXPECTED_GATE_THRESHOLDS_V10
        or type(gate.get("checks")) is not dict
        or set(gate["checks"]) != EXPECTED_GATE_CHECKS_V10
        or len(gate["checks"]) != 24
        or any(value is not True for value in gate["checks"].values())
        or receipt.get("caps")
        != {"updates": 1_000, "microbatch_graphs": 4_000, "presentations": 16_000}
        or type(training) is not dict
        or set(training)
        != {
            "accounting",
            "checkpoint",
            "checkpoint_access_status",
            "core",
            "cell_volume_attention_activity",
            "diagnostics",
            "joint_from_update_one",
            "separate_head_or_predictor_training",
            "trace",
        }
        or training.get("accounting") != EXPECTED_ACCOUNTING_V10
        or training.get("checkpoint")
        != {
            "path": "checkpoint_update_1000.pt",
            "byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
            "file_sha256": CANDIDATE_CHECKPOINT_SHA256,
        }
        or training.get("checkpoint_access_status")
        != "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION"
        or training.get("core")
        != "unchanged scripts.run_go2_rgb_swept_progress_survival_joint_jepa_"
        "v9_content_adaptive_dense_local_token_lift"
        or training.get("joint_from_update_one") is not True
        or training.get("separate_head_or_predictor_training") is not False
        or type(training.get("diagnostics")) is not dict
        or training["diagnostics"].get("dense_local_attention")
        != training.get("cell_volume_attention_activity")
        or training["diagnostics"].get("v10_contract")
        != {
            "schema": "lewm_v10_unchanged_joint_training_contract_v1",
            "objective": "S+P+U+R+O",
            "occupied_auxiliary_coefficient": 0.5,
            "new_loss_or_head": False,
            "training_core": "unchanged_v9_wrapper_over_v3_v4",
        }
    ):
        raise PermissionError("V10 terminal result contract changed")
    _validate_attention_activity_v10(training["cell_volume_attention_activity"])
    if authority != {
        "checkpoint_access_authorized_for_physical_calibration": True,
        "checkpoint_qualified": False,
        "development_only": True,
        "g2_navigation_final_evaluation_opened": False,
        "heldout_or_sealed_opened": False,
        "physical_evidence_gate_passed": False,
        "promotion_performed": False,
        "retry_or_resume_authorized": False,
    }:
        raise PermissionError("V10 terminal authority changed")
    if physical != {
        "schema": "lewm_v10_unchanged_physical_calibration_stage_v1",
        "source": "unchanged_v9_v4_2016_tuple_protocol",
        "physical_calibration_run_in_this_attempt": False,
        "physical_gate_passed": False,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "requires_full_arm_pass": True,
        "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT",
        "threshold_tuple_count": 2_016,
    }:
        raise PermissionError("V10 physical-calibration staging changed")
    if (
        type(access) is not dict
        or access.get("forbidden_input_count") != 0
        or access.get("g2_navigation_final_evaluation_open_count") != 0
        or type(science) is not dict
        or set(science)
        != {
            "only_change",
            "initial_v10_model",
            "architecture",
            "objective",
            "inherited_occupied_auxiliary",
            "model_changed",
            "data_changed",
            "dataset_identity_changed",
            "input_tensorization_changed",
            "inherited_parameter_state_bit_exact",
            "optimizer_rules_changed",
            "optimizer_parameter_tensor_membership_changed",
            "losses_changed",
            "new_loss_or_head",
            "schedule_changed",
            "evaluation_changed",
        }
        or science.get("only_change")
        != "projective_cell_volume_token_lift_geometry_and_masked_mean_base"
        or science.get("objective") != "S+P+U+R+O"
        or science.get("model_changed") is not True
        or science.get("inherited_parameter_state_bit_exact") is not True
        or science.get("new_loss_or_head") is not False
        or science.get("optimizer_parameter_tensor_membership_changed") is not False
        or any(
            science.get(name) is not False
            for name in (
                "data_changed",
                "dataset_identity_changed",
                "input_tensorization_changed",
                "optimizer_rules_changed",
                "losses_changed",
                "schedule_changed",
                "evaluation_changed",
            )
        )
        or receipt.get("n320", {}).get("predecessor_experiment_checkpoint_read")
        is not False
    ):
        raise PermissionError("V10 result custody or science receipt changed")


def _new_access_v10() -> dict[str, int]:
    access = dict(_v4._new_access())
    for name in (
        "candidate_receipt_reads",
        "candidate_checkpoint_reads",
        "candidate_checkpoint_loads",
    ):
        access.pop(name)
    access.update(
        {
            "candidate_result_read_attempts": 0,
            "candidate_result_read_successes": 0,
            "candidate_result_validations": 0,
            "candidate_checkpoint_read_attempts": 0,
            "candidate_checkpoint_read_successes": 0,
            "candidate_checkpoint_load_attempts": 0,
            "candidate_checkpoint_load_successes": 0,
        }
    )
    return access


def _load_candidate_v10(repository_root: Path, access: dict[str, int]) -> Any:
    root = repository_root / CANDIDATE_ROOT_RELATIVE_PATH
    access["candidate_result_read_attempts"] += 1
    result_raw = _read_regular(root / "result.json")
    access["candidate_result_read_successes"] += 1
    if (
        len(result_raw) != CANDIDATE_RESULT_BYTE_COUNT
        or hashlib.sha256(result_raw).hexdigest() != CANDIDATE_RESULT_FILE_SHA256
    ):
        raise PermissionError("V10 result file identity changed")
    receipt = _parse_canonical(result_raw, name="V10 terminal result")
    _validate_candidate_result_v10(receipt)
    access["candidate_result_validations"] += 1

    # This read is deliberately unreachable until every result check passes.
    access["candidate_checkpoint_read_attempts"] += 1
    checkpoint_raw = _read_regular(root / "checkpoint_update_1000.pt")
    access["candidate_checkpoint_read_successes"] += 1
    if (
        len(checkpoint_raw) != CANDIDATE_CHECKPOINT_BYTE_COUNT
        or hashlib.sha256(checkpoint_raw).hexdigest() != CANDIDATE_CHECKPOINT_SHA256
    ):
        raise PermissionError("V10 checkpoint identity changed")
    adapter = importlib.import_module(ADAPTER_MODULE)
    if (
        getattr(adapter, "PHYSICAL_CALIBRATION_PREREGISTRATION_COMMIT", None)
        != PREREGISTRATION_COMMIT
    ):
        raise PermissionError("V10 physical-evidence adapter authority changed")
    load_checkpoint = getattr(adapter, "load_checkpoint", None)
    if not callable(load_checkpoint):
        raise PermissionError("V10 physical-evidence adapter API changed")
    access["candidate_checkpoint_load_attempts"] += 1
    model = load_checkpoint(checkpoint_raw)
    access["candidate_checkpoint_load_successes"] += 1
    return model


def _validate_development_access_v10(
    inputs: Any, loader: Any, access: dict[str, int]
) -> None:
    loader_counts = loader.model_facing_access_counts()
    loader_receipt = loader.receipt()
    if (
        loader_counts["endpoint_rgb_row_request_count"]
        != sum(ROLE_COUNTS.values())
        or loader_counts["raster_label_row_request_count"]
        != sum(ROLE_COUNTS.values())
        or any(
            loader_counts[name] != 0
            for name in (
                "current_rgb_row_request_count",
                "next_rgb_row_request_count",
                "fixed_negative_rgb_row_request_count",
            )
        )
    ):
        raise PermissionError("model-facing development access changed")
    if (
        loader_receipt.get("raw_inputs_frame_attribute_invocation_count") != 0
        or any(
            int(value) != 0
            for value in loader_receipt.get("forbidden_semantic_counters", {}).values()
        )
    ):
        raise PermissionError("forbidden development access was recorded")
    payload_records = [
        record
        for record in inputs.consumed.values()
        if record.get("kind") in {"development_rgb", "raw_supervision"}
    ]
    if any("train" in record.get("roles", []) for record in payload_records):
        access["train_role_payload_requests"] += 1
        raise PermissionError("train-role payload entered calibration")


def _authority_v10(*, physical_passed: bool) -> Mapping[str, Any]:
    return {
        "development_only": True,
        "development_physical_evidence_passed": physical_passed,
        "g2_binding_preparation_authorized": physical_passed,
        "g2_opened": False,
        "g2_qualified": False,
        "navigation_qualified": False,
        "promotion_performed": False,
        "deployment_authorized": False,
        "training_or_resume_authorized": False,
        "scientific_retry_authorized": False,
        "heldout_or_sealed_opened": False,
    }


def _candidate_receipt_v10(*, validated: bool) -> Mapping[str, Any]:
    return {
        "terminal_result_commit": TERMINAL_RESULT_COMMIT,
        "result": {
            "path": "result.json",
            "byte_count": CANDIDATE_RESULT_BYTE_COUNT,
            "file_sha256": CANDIDATE_RESULT_FILE_SHA256,
            "content_sha256": CANDIDATE_RESULT_CONTENT_SHA256,
        },
        "checkpoint": {
            "path": "checkpoint_update_1000.pt",
            "schema": CANDIDATE_CHECKPOINT_SCHEMA,
            "byte_count": CANDIDATE_CHECKPOINT_BYTE_COUNT,
            "file_sha256": CANDIDATE_CHECKPOINT_SHA256,
        },
        "result_validated_before_checkpoint_read": validated,
        "adapter_module": ADAPTER_MODULE,
    }


def execute_v10(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    output = _fresh_output_v10(repository_root)
    access = _new_access_v10()
    stage = "reserved_output"
    source_hashes: Mapping[str, str] | None = None
    runtime = None
    inputs = None
    loader = None
    progress: Mapping[str, Any] | None = None
    try:
        stage = "validated_sources"
        source_hashes = _validate_sources_v10(repository_root)
        stage = "loaded_candidate"
        model = _load_candidate_v10(repository_root, access)
        stage = "constructed_development_boundary"
        runtime, inputs, loader, progress = _build_data_boundary(repository_root)
        role_arrays: dict[str, tuple[Any, Any]] = {}
        role_receipts: dict[str, Mapping[str, Any]] = {}
        for role in ("probability_calibration", "checkpoint_selection"):
            stage = f"collected_{role}"
            pairs = inputs.role_pairs(role)
            logits, labels, receipt = _collect_role(
                model,
                loader,
                pairs,
                role=role,
                torch=runtime.torch,
            )
            role_arrays[role] = (logits, labels)
            role_receipts[role] = receipt
        stage = "fit_select_score"
        science = _fit_select_score(
            *role_arrays["probability_calibration"],
            *role_arrays["checkpoint_selection"],
            provenance={
                "role": "probability_calibration",
                "candidate_result_content_sha256": CANDIDATE_RESULT_CONTENT_SHA256,
                "candidate_checkpoint_sha256": CANDIDATE_CHECKPOINT_SHA256,
                "pair_count": ROLE_COUNTS["probability_calibration"],
                "cell_count": ROLE_CELL_COUNTS["probability_calibration"],
                "next_endpoint_order_sha256": role_receipts[
                    "probability_calibration"
                ]["next_endpoint_order_sha256"],
                "all_cells_used": True,
            },
            operation_counts=access,
        )
        stage = "validated_access"
        _validate_development_access_v10(inputs, loader, access)
        if (
            access["calibration_fit_calls"] != 1
            or access["threshold_selection_calls"] != 1
        ):
            raise RuntimeError("V10 calibration or threshold selection count changed")
        stage = "write_outputs"
        calibration_raw = _canonical_bytes(science["calibration"]) + b"\n"
        calibration_binding = _atomic_write(
            output / "calibration.json", calibration_raw
        )
        physical_passed = bool(science["gate"]["passed"])
        result, _ = _write_json(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": science["gate"]["status"],
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "reference_calibration_preregistration_commit": (
                    REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT
                ),
                "candidate": _candidate_receipt_v10(validated=True),
                "source_sha256": source_hashes,
                "protocol": {
                    "scientific_change_from_v4": False,
                    "fit_select_score_implementation": (
                        "scripts.calibrate_go2_rgb_swept_progress_survival_"
                        "joint_jepa_v4_physical_evidence._fit_select_score"
                    ),
                    "routing": "NOT_APPLICABLE",
                },
                "roles": role_receipts,
                "calibration_artifact": {
                    **calibration_binding,
                    "content_sha256": science["calibration"]["content_sha256"],
                    "id": science["calibration"]["id"],
                },
                "threshold_selection": science["threshold_selection"],
                "selection": science["selection"],
                "gate": science["gate"],
                "routing": {
                    "status": "NOT_APPLICABLE",
                    "included_in_gate": False,
                    "reason": "physical_evidence_is_not_configuration_space",
                    "deferred_to": "G3_post_memory_multi_view_fusion",
                },
                "raw_access": _raw_access_snapshot(inputs, loader, progress),
                "access": access,
                "authority": _authority_v10(physical_passed=physical_passed),
            },
        )
        return result
    except Exception as error:
        failure, _ = _write_json(
            output / "failure.json",
            {
                "schema": FAILURE_SCHEMA,
                "status": "FAILED_OPERATIONALLY",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "reference_calibration_preregistration_commit": (
                    REFERENCE_CALIBRATION_PREREGISTRATION_COMMIT
                ),
                "stage": stage,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "candidate": _candidate_receipt_v10(
                    validated=access["candidate_result_validations"] == 1
                ),
                "source_sha256": source_hashes,
                "raw_access": _raw_access_snapshot(inputs, loader, progress),
                "access": access,
                "authority": _authority_v10(physical_passed=False),
            },
        )
        return failure


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = execute_v10()
    print(
        json.dumps(
            {"status": result["status"], "output": OUTPUT_RELATIVE_PATH},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    if result["status"] == "FAILED_OPERATIONALLY":
        return 1
    return 0 if result.get("gate", {}).get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
