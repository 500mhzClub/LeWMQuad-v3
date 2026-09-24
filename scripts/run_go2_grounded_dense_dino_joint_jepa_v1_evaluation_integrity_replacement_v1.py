#!/usr/bin/env python3
"""Evaluate the frozen grounded dense-DINO V1 checkpoints exactly once.

This runner is an evaluation-only integrity replacement for the consumed V1
attempt.  It never trains, resumes, mutates, or writes a checkpoint, and its
data API has no evaluation-successor path.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as base  # noqa: E402
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (  # noqa: E402
    ActionSpecificRidgeReadoutsV1,
    fit_action_specific_ridge_readouts_v1,
    predict_action_specific_scores_v1,
    task_conditioned_feature_v1,
)


SCHEMA = (
    "lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1_result_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1_terminal_v1"
)
AUTHORITY_SCHEMA = (
    "lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1_execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_EXACT_GROUNDED_DENSE_DINO_JOINT_JEPA_V1_"
    "EVALUATION_INTEGRITY_REPLACEMENT_V1"
)
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1_source_review_v1"
)
SOURCE_REVIEW_STATUS = "PASS_INDEPENDENT_SOURCE_REVIEW"

PREREGISTRATION = REPO_ROOT / (
    "docs/lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1_preregistration_2026-08-04.md"
)
PREREGISTRATION_SHA256 = (
    "0c563e2dd6cd7fce4fa8b00958eed28148faf5eb981fe01a6f8dde5576704e74"
)
PREREGISTRATION_BYTE_COUNT = 8_708
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1/attempt_v1"
)
ORIGINAL_OUTPUT_ROOT = REPO_ROOT / (
    ".generated/dev/go2_grounded_dense_dino_joint_jepa_v1/attempt_v1"
)

EXPECTED_TRAIN_RUNTIME_IDENTITY = (
    "2fda67b0eca18c90004e7a099ef6fd46c4f623adfd6c85d093574fdc62a09699"
)
EXPECTED_TRAIN_PLAN_IDENTITY = (
    "f6f94cf589ec44324fdefe0939aa7076e25543d984464d5b264a0b2f0ff9535b"
)
EXPECTED_TASK_IDENTITY = (
    "8163103162bcad02b5d06542ed6087cfe6f99bd9e9deccb81f66f051a12ebc7d"
)
EXPECTED_TASK_EVAL_REGRET = 0.17441406250000002
TASK_RIDGE_LAMBDA = 1.0e-3

ORIGINAL_AUTHORITY = REPO_ROOT / (
    "docs/lewm_go2_grounded_dense_dino_joint_jepa_v1_"
    "execution_authority_2026-08-04.json"
)
ORIGINAL_RESERVATION = ORIGINAL_OUTPUT_ROOT / "reservation.json"
ORIGINAL_TERMINAL = ORIGINAL_OUTPUT_ROOT / "terminal.json"
ORIGINAL_PHYSICAL_CHECKPOINT = (
    ORIGINAL_OUTPUT_ROOT / "checkpoints/physical_only_matched_update_800.pt"
)
ORIGINAL_JOINT_CHECKPOINT = (
    ORIGINAL_OUTPUT_ROOT / "checkpoints/joint_jepa_grounded_update_800.pt"
)


class RecoveryError(base.GroundedRunnerError):
    """Raised for any replacement authority, integrity, or science failure."""


def predecessor_bindings_v1() -> dict[str, dict[str, Any]]:
    return {
        "original_authority": base._binding(  # noqa: SLF001
            ORIGINAL_AUTHORITY,
            "c5a1627e5b0528507a52ee07e2e8442a32a7c0017161aeb4d080e0d3357c76e4",
            10_445,
        ),
        "original_reservation": base._binding(  # noqa: SLF001
            ORIGINAL_RESERVATION,
            "509e2a839e07caf40cf2b21fc7c7991b53c8bc6f257ddb9e09a36dfca2f1c97c",
            9_088,
        ),
        "original_terminal": base._binding(  # noqa: SLF001
            ORIGINAL_TERMINAL,
            "481f0f5eb46c11151241b8ce577e41154a3eefae2bc40bfb3671e5a96f45209d",
            270,
        ),
        "physical_only_update_800_checkpoint": base._binding(  # noqa: SLF001
            ORIGINAL_PHYSICAL_CHECKPOINT,
            "58ac1dfe4f083db038839766235c1ddeee83bb48f338082028da4f28662ab7bc",
            263_193_051,
        ),
        "joint_jepa_update_800_checkpoint": base._binding(  # noqa: SLF001
            ORIGINAL_JOINT_CHECKPOINT,
            "a9d2f4cb585ce2098b08024edd5faba8a8bdc6e88bfe1fa4d5a35822769311bc",
            263_193_051,
        ),
    }


SOURCE_PATHS = {
    "recovery_runner": Path(__file__).resolve(),
    "recovery_runner_test": REPO_ROOT
    / "lewm/tests/test_run_go2_grounded_dense_dino_joint_jepa_v1_"
    "evaluation_integrity_replacement_v1.py",
} | {f"v1_{label}": path for label, path in base.SOURCE_PATHS.items()}


def numpy_runtime_v1() -> dict[str, str]:
    configuration = np.show_config(mode="dicts")
    dependency = configuration["Build Dependencies"]["blas"]
    return {
        "numpy": np.__version__,
        "blas_name": str(dependency["name"]),
        "blas_version": str(dependency["version"]),
        "blas_configuration": str(dependency["openblas configuration"]),
    }


def config_v1() -> dict[str, Any]:
    return {
        "action_count": base.ACTION_COUNT,
        "train_state_count": base.STATE_COUNT,
        "eval_state_count": base.STATE_COUNT,
        "context_frames_per_eval_state": base.CONTEXT_COUNT,
        "learned_arms": list(base.ARM_ORDER),
        "checkpoint_update": base.MAX_UPDATES,
        "prediction_repeats": 2,
        "task_action_only": {
            "features": ["goal_x_body_m", "goal_y_body_m", "constant_one"],
            "heads": base.ACTION_COUNT,
            "ridge_lambda": TASK_RIDGE_LAMBDA,
            "expected_train_identity_sha256": EXPECTED_TASK_IDENTITY,
            "expected_eval_regret": EXPECTED_TASK_EVAL_REGRET,
            "same_process_exact_refits": 2,
        },
        "bootstrap_draws": 10_000,
        "bootstrap_seed": base.BOOTSTRAP_SEED,
        "checkpoint_policy": "one_byte_read_per_arm_no_optimizer_restore",
        "training_allowed": False,
    }


def permissions_v1() -> dict[str, bool]:
    return {
        "predecessor_evidence_access": True,
        "predecessor_checkpoint_read_once": True,
        "train_receipt_access": True,
        "train_rgb_access": False,
        "eval_receipt_access_after_qualification": True,
        "eval_context_rgb_access_after_qualification": True,
        "eval_successor_rgb_access": False,
        "training_or_optimizer_access": False,
        "checkpoint_write_or_mutation": False,
        "data_generation": False,
        "protected_or_sealed_access": False,
        "retry_resume_overwrite": False,
    }


def _read_bound_bytes_once(binding: Mapping[str, Any], *, label: str) -> bytes:
    selected = base._require_binding(binding, label=label, rehash=False)  # noqa: SLF001
    path = base.safe_path_v1(Path(str(selected["path"])), label=label)
    if not path.is_file():
        raise RecoveryError(f"{label} is not a file")
    raw = path.read_bytes()
    if (
        len(raw) != int(selected["byte_count"])
        or hashlib.sha256(raw).hexdigest() != str(selected["sha256"])
    ):
        raise RecoveryError(f"{label} binding changed")
    return raw


def _read_bound_json_once(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return base._strict_json_loads(  # noqa: SLF001
        _read_bound_bytes_once(binding, label=label), label=label
    )


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def _readouts_exactly_equal(
    left: ActionSpecificRidgeReadoutsV1,
    right: ActionSpecificRidgeReadoutsV1,
) -> bool:
    if left.identity_sha256 != right.identity_sha256 or len(left.heads) != len(
        right.heads
    ):
        return False
    for first, second in zip(left.heads, right.heads, strict=True):
        if (
            first.identity_sha256 != second.identity_sha256
            or first.ridge_lambda != second.ridge_lambda
            or first.training_rows != second.training_rows
            or first.solver != second.solver
            or not np.array_equal(first.feature_mean, second.feature_mean)
            or not np.array_equal(first.feature_scale, second.feature_scale)
            or not np.array_equal(first.coefficients, second.coefficients)
        ):
            return False
    return True


def fit_current_task_action_only_v1(plan: Any) -> ActionSpecificRidgeReadoutsV1:
    """Refit the unchanged analytic task control under the bound environment."""

    if (
        getattr(plan, "identity_sha256", None) != EXPECTED_TRAIN_PLAN_IDENTITY
        or len(getattr(plan, "states", ())) != base.STATE_COUNT
    ):
        raise RecoveryError("training feature-plan identity changed")
    features = np.stack(
        [
            task_conditioned_feature_v1(
                None, relative_target_xy_body_m=state.relative_target_xy_body_m
            )
            for state in plan.states
        ]
    ).astype(np.float64, copy=False)
    targets = []
    for state in plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        if ranks.shape != (base.ACTION_COUNT,) or not np.isfinite(ranks).all():
            raise RecoveryError("task-control ranks changed")
        maximum = float(ranks.max())
        if maximum <= 0.0:
            raise RecoveryError("task-control ranks have no positive maximum")
        targets.append(ranks / maximum)
    target_matrix = np.stack(targets)
    feature_sets = [features for _ in range(base.ACTION_COUNT)]
    target_sets = [target_matrix[:, action] for action in range(base.ACTION_COUNT)]

    def fit_once() -> ActionSpecificRidgeReadoutsV1:
        return fit_action_specific_ridge_readouts_v1(
            feature_sets, target_sets, ridge_lambda=TASK_RIDGE_LAMBDA
        )

    first = fit_once()
    second = fit_once()
    if not _readouts_exactly_equal(first, second):
        raise RecoveryError("same-process task-control refit changed")
    if first.identity_sha256 != EXPECTED_TASK_IDENTITY:
        raise RecoveryError("current-environment task-control identity changed")
    if any(
        head.solver != "primal"
        or head.training_rows != base.STATE_COUNT
        or head.ridge_lambda != TASK_RIDGE_LAMBDA
        for head in first.heads
    ):
        raise RecoveryError("task-control solver contract changed")
    return first


def score_task_action_only_v1(
    plan: Any, readouts: ActionSpecificRidgeReadoutsV1
) -> np.ndarray:
    rows = []
    for state in plan.states:
        feature = task_conditioned_feature_v1(
            None, relative_target_xy_body_m=state.relative_target_xy_body_m
        )
        rows.append(
            predict_action_specific_scores_v1(
                readouts, [feature for _ in range(base.ACTION_COUNT)]
            )
        )
    scores = np.stack(rows)
    if (
        scores.shape != (base.STATE_COUNT, base.ACTION_COUNT)
        or not np.isfinite(scores).all()
    ):
        raise RecoveryError("task-control evaluation scores changed")
    return scores


def _trace_row(trace: Sequence[Mapping[str, Any]], update: int) -> Mapping[str, Any]:
    selected = [row for row in trace if int(row.get("update", -1)) == update]
    if len(selected) != 1:
        raise RecoveryError(f"checkpoint trace update {update} changed")
    return selected[0]


def read_checkpoint_once_v1(
    binding: Mapping[str, Any], *, expected_arm: str
) -> dict[str, Any]:
    raw = _read_bound_bytes_once(binding, label=f"{expected_arm} checkpoint")
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    required = {
        "schema",
        "arm",
        "update",
        "model_seed",
        "sampler_seed",
        "config",
        "train_identity_sha256",
        "initial_model_identity_sha256",
        "input_statistics",
        "outcome_statistics",
        "model_state_dict",
        "optimizer_state_dict",
        "trace",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != base.CHECKPOINT_SCHEMA
        or payload.get("arm") != expected_arm
        or payload.get("update") != base.MAX_UPDATES
        or payload.get("model_seed") != base.MODEL_SEED
        or payload.get("sampler_seed") != base.SAMPLER_SEED
        or payload.get("config") != base.runner_config_v1()
        or payload.get("train_identity_sha256") != EXPECTED_TRAIN_RUNTIME_IDENTITY
        or not isinstance(payload.get("model_state_dict"), Mapping)
        or not isinstance(payload.get("optimizer_state_dict"), Mapping)
        or not isinstance(payload.get("trace"), list)
    ):
        raise RecoveryError(f"{expected_arm} checkpoint contract changed")
    trace = payload["trace"]
    if [int(row.get("update", -1)) for row in trace] != [0, 400, 800]:
        raise RecoveryError(f"{expected_arm} checkpoint trace changed")
    if any(row.get("all_finite") is not True for row in trace):
        raise RecoveryError(f"{expected_arm} checkpoint records nonfinite state")
    result = dict(payload)
    result.pop("optimizer_state_dict")
    return result


def qualify_checkpoints_v1(
    physical: Mapping[str, Any], joint: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        physical.get("initial_model_identity_sha256")
        != joint.get("initial_model_identity_sha256")
        or not _same_value(physical.get("input_statistics"), joint.get("input_statistics"))
        or not _same_value(
            physical.get("outcome_statistics"), joint.get("outcome_statistics")
        )
    ):
        raise RecoveryError("learned checkpoint pairing changed")
    required_fields = {
        "normalized_physical_rank_regret",
        "branch_retrieval_accuracy",
        "successor_cosine_error",
        "persistence_cosine_error",
        "all_finite",
    }
    trace_0 = _trace_row(joint["trace"], 0)
    trace_400 = _trace_row(joint["trace"], 400)
    futility = base.train_only_futility_v1(
        {key: trace_0[key] for key in required_fields},
        {key: trace_400[key] for key in required_fields},
    )
    if futility.get("continue_to_update_800") is not True:
        raise RecoveryError("joint checkpoint was ineligible to reach update 800")
    return {
        "common_initial_model_identity_sha256": physical[
            "initial_model_identity_sha256"
        ],
        "train_role_identity_sha256": physical["train_identity_sha256"],
        "input_statistics_identity_sha256": physical["input_statistics"][
            "identity_sha256"
        ],
        "outcome_statistics_identity_sha256": physical["outcome_statistics"][
            "identity_sha256"
        ],
        "joint_update_400_futility": futility,
        "physical_trace": physical["trace"],
        "joint_trace": joint["trace"],
    }


def _validate_original_authority_v1(document: Mapping[str, Any]) -> None:
    if (
        document.get("schema") != base.AUTHORITY_SCHEMA
        or document.get("status") != base.AUTHORITY_STATUS
        or document.get("preregistration_binding")
        != base._binding(  # noqa: SLF001
            base.PREREGISTRATION,
            base.PREREGISTRATION_SHA256,
            base.PREREGISTRATION_BYTE_COUNT,
        )
        or document.get("input_bindings") != base.fixed_input_bindings_v1()
        or document.get("config") != base.runner_config_v1()
        or document.get("output_root") != str(ORIGINAL_OUTPUT_ROOT.resolve())
    ):
        raise RecoveryError("original execution authority changed")


def validate_original_inventory_v1() -> None:
    root = base.safe_path_v1(ORIGINAL_OUTPUT_ROOT, label="original attempt root")
    checkpoints = base.safe_path_v1(
        root / "checkpoints", label="original checkpoint directory"
    )
    if not root.is_dir() or not checkpoints.is_dir():
        raise RecoveryError("original attempt directories changed")
    root_entries = {path.name for path in root.iterdir()}
    checkpoint_entries = {path.name for path in checkpoints.iterdir()}
    if root_entries != {"reservation.json", "terminal.json", "checkpoints"} or checkpoint_entries != {
        "physical_only_matched_update_800.pt",
        "joint_jepa_grounded_update_800.pt",
    }:
        raise RecoveryError("original attempt inventory changed")


def validate_predecessor_attempt_v1(
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bindings = authority["predecessor_bindings"]
    original_authority = _read_bound_json_once(
        bindings["original_authority"], label="original authority"
    )
    _validate_original_authority_v1(original_authority)
    reservation = _read_bound_json_once(
        bindings["original_reservation"], label="original reservation"
    )
    if (
        reservation.get("schema")
        != "lewm_go2_grounded_dense_dino_joint_jepa_v1_reservation_v1"
        or reservation.get("status") != "CONSUMED_ONE_SHOT_ATTEMPT"
        or reservation.get("authority") != original_authority
    ):
        raise RecoveryError("original reservation changed")
    terminal = _read_bound_json_once(
        bindings["original_terminal"], label="original terminal"
    )
    if terminal != {
        "schema": base.TERMINAL_SCHEMA,
        "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
        "error_type": "DenseSharedCalibrationError",
        "error_message": "task/action-only control identity changed",
        "retry_authorized": False,
        "result_binding": None,
    }:
        raise RecoveryError("original terminal is not the admitted failure")
    validate_original_inventory_v1()
    physical = read_checkpoint_once_v1(
        bindings["physical_only_update_800_checkpoint"],
        expected_arm="physical_only_matched",
    )
    joint = read_checkpoint_once_v1(
        bindings["joint_jepa_update_800_checkpoint"],
        expected_arm="joint_jepa_grounded",
    )
    return original_authority, physical, joint


def _model_from_payload_v1(
    payload: Mapping[str, Any],
    *,
    dino: Any,
    model_class: type[torch.nn.Module],
    device: torch.device,
) -> torch.nn.Module:
    blocks, norm = dino.fresh_tail()
    model = model_class(
        blocks,
        norm,
        initialization_seed=base.MODEL_SEED,
        ema_momentum=base.EMA_MOMENTUM,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RecoveryError("recovery model retained trainable parameters")
    return model.to(device)


def evaluate_v1(
    *,
    train: Any,
    evaluation: Any,
    eval_context_trunks: torch.Tensor,
    task: ActionSpecificRidgeReadoutsV1,
    physical_payload: Mapping[str, Any],
    joint_payload: Mapping[str, Any],
    dino: Any,
    model_class: type[torch.nn.Module],
    benchmark: Any,
    device: torch.device,
) -> dict[str, Any]:
    from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as physical

    disjointness = base.assert_role_disjointness_v1(train.plan, evaluation.plan)
    payloads = {
        "physical_only_matched": physical_payload,
        "joint_jepa_grounded": joint_payload,
    }
    reports: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for arm, payload in payloads.items():
            model = _model_from_payload_v1(
                payload, dino=dino, model_class=model_class, device=device
            )
            first = base.predict_role_outcomes_v1(
                model=model,
                role=evaluation,
                context_trunks=eval_context_trunks,
                input_statistics=physical_payload["input_statistics"],
                outcome_statistics=physical_payload["outcome_statistics"],
                benchmark=benchmark,
                device=device,
            )
            second = base.predict_role_outcomes_v1(
                model=model,
                role=evaluation,
                context_trunks=eval_context_trunks,
                input_statistics=physical_payload["input_statistics"],
                outcome_statistics=physical_payload["outcome_statistics"],
                benchmark=benchmark,
                device=device,
            )
            if not torch.equal(first, second):
                raise RecoveryError(f"{arm} deterministic repeat changed")
            predictions[arm] = first
            reports[arm] = benchmark.report_physical_scores_v1(
                evaluation.plan,
                benchmark.physical_score_matrix_v1(evaluation.plan, first),
            )
            del model

    task_scores = score_task_action_only_v1(evaluation.plan, task)
    reports["task_action_only"] = benchmark.report_physical_scores_v1(
        evaluation.plan, task_scores
    )
    task_regret = float(
        reports["task_action_only"]["summary"]["normalized_rank_regret"]
    )
    if task_regret != EXPECTED_TASK_EVAL_REGRET:
        raise RecoveryError("task-control behavioral parity changed")
    oracle_scores = np.asarray(
        [state.dense_ranks for state in evaluation.plan.states], dtype=np.float64
    )
    reports["privileged_physical_oracle"] = benchmark.report_physical_scores_v1(
        evaluation.plan, oracle_scores
    )
    reports["random_expected"] = physical.prior._random_expected_report(  # noqa: SLF001
        evaluation.plan
    )
    joint_vs_task = benchmark.paired_family_scene_bootstrap_v1(
        base.report_group_results_v1(
            reports["joint_jepa_grounded"], label="joint_jepa_grounded"
        ),
        base.report_group_results_v1(
            reports["task_action_only"], label="task_action_only"
        ),
    )
    joint_vs_matched = benchmark.paired_family_scene_bootstrap_v1(
        base.report_group_results_v1(
            reports["joint_jepa_grounded"], label="joint_jepa_grounded"
        ),
        base.report_group_results_v1(
            reports["physical_only_matched"], label="physical_only_matched"
        ),
    )
    gate = benchmark.fixed_gate_v1(
        joint_report=reports["joint_jepa_grounded"],
        task_report=reports["task_action_only"],
        matched_report=reports["physical_only_matched"],
        random_report=reports["random_expected"],
        oracle_report=reports["privileged_physical_oracle"],
        joint_vs_task=joint_vs_task,
        joint_vs_matched=joint_vs_matched,
        integrity_passed=True,
    )
    return {
        "reports": reports,
        "comparisons": {
            "joint_vs_task_action_only": joint_vs_task,
            "joint_vs_physical_only_matched": joint_vs_matched,
        },
        "prediction_identities": {
            arm: base._tensor_sha256(value)  # noqa: SLF001
            for arm, value in predictions.items()
        },
        "integrity_evidence": {
            "role_disjointness": disjointness,
            "task_action_train_identity_sha256": task.identity_sha256,
            "task_action_eval_regret": task_regret,
            "expected_task_action_eval_regret": EXPECTED_TASK_EVAL_REGRET,
            "failed_v1_eval_context_was_previously_opened": True,
            "fresh_or_held_out_evaluation": False,
        },
        "deterministic_repeat_passed": True,
        "gate": gate,
    }


def recovery_access_audit_v1(
    ledger: base.AccessLedgerV1,
    *,
    checkpoint_reads: Mapping[str, int],
) -> dict[str, Any]:
    audit = ledger.audit()
    if (
        audit["stage"] != "evaluation"
        or audit["physical_checkpoint_durable"] is not True
        or audit["joint_checkpoint_durable"] is not True
        or audit["receipt_loads"] != {"train": 1, "eval": 1}
        or audit["role_index_opens"] != {"train": 1, "eval": 1}
        or audit["state_receipt_opens"]
        != {"train": base.STATE_COUNT, "eval": base.STATE_COUNT}
        or audit["rgb_opens"]
        != {
            "train_context": 0,
            "train_successor": 0,
            "eval_context": base.STATE_COUNT * base.CONTEXT_COUNT,
            "eval_successor": 0,
        }
        or audit["unique_rgb_artifacts"] != base.STATE_COUNT * base.CONTEXT_COUNT
        or checkpoint_reads
        != {"physical_only_matched": 1, "joint_jepa_grounded": 1}
    ):
        raise RecoveryError("replacement access audit changed")
    return audit | {
        "predecessor_checkpoint_semantic_reads": dict(checkpoint_reads),
        "training_performed": False,
        "optimizer_instantiated_or_restored": False,
        "checkpoint_written_or_mutated": False,
        "evaluation_repeated_after_v1_infrastructure_failure": True,
    }


def _source_bindings_unchanged(authority: Mapping[str, Any]) -> None:
    for label, expected in authority["source_bindings"].items():
        if base.file_binding_v1(Path(str(expected["path"]))) != expected:
            raise RecoveryError(f"source {label} changed during replacement")


def _load_authority_v1(
    path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
) -> dict[str, Any]:
    binding = base.file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise RecoveryError("replacement authority caller binding changed")
    document = base._strict_json_loads(  # noqa: SLF001
        Path(path).read_bytes(), label="replacement authority"
    )
    required = {
        "schema",
        "status",
        "preregistration_binding",
        "source_review_binding",
        "source_bindings",
        "input_bindings",
        "predecessor_bindings",
        "dino",
        "environment",
        "permissions",
        "config",
        "output_root",
        "reviewed_git_commit",
    }
    if (
        set(document) != required
        or document.get("schema") != AUTHORITY_SCHEMA
        or document.get("status") != AUTHORITY_STATUS
        or document.get("permissions") != permissions_v1()
        or document.get("config") != config_v1()
        or document.get("output_root") != str(DEFAULT_OUTPUT_ROOT.resolve())
    ):
        raise RecoveryError("replacement authority contract changed")
    base._validate_git_commit(document["reviewed_git_commit"])  # noqa: SLF001
    prereg = base._require_binding(  # noqa: SLF001
        document["preregistration_binding"], label="replacement preregistration"
    )
    if prereg != base._binding(  # noqa: SLF001
        PREREGISTRATION, PREREGISTRATION_SHA256, PREREGISTRATION_BYTE_COUNT
    ):
        raise RecoveryError("replacement preregistration binding changed")
    sources = document["source_bindings"]
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_PATHS):
        raise RecoveryError("replacement source closure changed")
    for label, expected_path in SOURCE_PATHS.items():
        actual = base._require_binding(sources[label], label=f"source {label}")  # noqa: SLF001
        if actual["path"] != str(expected_path.resolve()):
            raise RecoveryError(f"source {label} path changed")
    review_binding = base._require_binding(  # noqa: SLF001
        document["source_review_binding"], label="replacement source review"
    )
    review = base._strict_json_loads(  # noqa: SLF001
        Path(str(review_binding["path"])).read_bytes(), label="replacement source review"
    )
    if (
        review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != SOURCE_REVIEW_STATUS
        or review.get("preregistration_binding") != prereg
        or review.get("source_bindings") != sources
        or review.get("findings") != []
        or review.get("protected_material_opened") is not False
        or not isinstance(review.get("checks"), Mapping)
        or not review["checks"]
        or any(value is not True for value in review["checks"].values())
    ):
        raise RecoveryError("replacement source review did not pass exactly")
    if document.get("input_bindings") != base.fixed_input_bindings_v1():
        raise RecoveryError("replacement scientific input bindings changed")
    predecessor = document.get("predecessor_bindings")
    if predecessor != predecessor_bindings_v1():
        raise RecoveryError("replacement predecessor bindings changed")
    for label, selected in predecessor.items():
        base._require_binding(selected, label=f"predecessor {label}", rehash=False)  # noqa: SLF001
    dino = document.get("dino")
    if (
        not isinstance(dino, Mapping)
        or set(dino) != {"repository_path", "repository_commit", "checkpoint_binding"}
        or dino.get("repository_commit") != base.DINO_REPOSITORY_COMMIT
        or dino.get("checkpoint_binding")
        != base._binding(  # noqa: SLF001
            Path(str(dino["checkpoint_binding"]["path"])),
            base.DINO_CHECKPOINT_SHA256,
            base.DINO_CHECKPOINT_BYTE_COUNT,
        )
    ):
        raise RecoveryError("replacement DINO binding changed")
    environment = document.get("environment")
    runtime = numpy_runtime_v1()
    if (
        not isinstance(environment, Mapping)
        or set(environment)
        != {
            "python",
            "torch",
            "hip",
            "numpy",
            "blas_name",
            "blas_version",
            "blas_configuration",
            "device_index",
            "device_name",
        }
        or environment.get("python") != str(Path(sys.executable).resolve())
        or environment.get("torch") != torch.__version__
        or environment.get("hip") != torch.version.hip
        or {key: environment.get(key) for key in runtime} != runtime
        or environment.get("device_index") != 0
    ):
        raise RecoveryError("replacement numerical environment changed")
    base.validate_live_device_v1(environment)
    return document


def execute_v1(authority: Mapping[str, Any]) -> dict[str, Any]:
    from lewm.benchmarks import go2_grounded_dense_dino_joint_jepa_v1 as benchmark
    from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (
        read_bound_rgb_bytes_v1,
    )
    from lewm.models.go2_grounded_dense_dino_joint_jepa_v1 import (
        GroundedDenseDINOJointJEPAV1,
    )

    output_root = base.safe_path_v1(
        Path(str(authority["output_root"])), label="replacement output root", must_exist=False
    )
    output_root.mkdir(parents=True, exist_ok=False)
    base._write_json_exclusive(  # noqa: SLF001
        output_root / "reservation.json",
        {
            "schema": (
                "lewm_go2_grounded_dense_dino_joint_jepa_v1_"
                "evaluation_integrity_replacement_v1_reservation_v1"
            ),
            "status": "CONSUMED_ONE_SHOT_EVALUATION_INTEGRITY_REPLACEMENT",
            "authority": dict(authority),
        },
    )

    original_authority, physical_payload, joint_payload = (
        validate_predecessor_attempt_v1(authority)
    )
    checkpoint_qualification = qualify_checkpoints_v1(
        physical_payload, joint_payload
    )
    determinism = base.configure_determinism_v1()
    ledger = base.AccessLedgerV1()
    shared = base._load_shared_role_metadata_v1(original_authority)  # noqa: SLF001
    ledger.load_receipts("train")
    train = base.load_role_runtime_data_v1(
        original_authority, shared, role="train", ledger=ledger
    )
    if train.identity_sha256 != EXPECTED_TRAIN_RUNTIME_IDENTITY:
        raise RecoveryError("replacement train-role identity changed")
    task = fit_current_task_action_only_v1(train.plan)

    ledger.checkpoint("physical_only_matched")
    ledger.checkpoint("joint_jepa_grounded")
    ledger.load_receipts("eval")
    evaluation = base.load_role_runtime_data_v1(
        original_authority, shared, role="eval", ledger=ledger
    )

    device = torch.device("cuda:0")
    dino = base.load_dino_trunk_v1(
        Path(str(authority["dino"]["repository_path"])),
        Path(str(authority["dino"]["checkpoint_binding"]["path"])),
        device=device,
    )
    eval_context_ids = tuple(
        itertools.chain.from_iterable(evaluation.context_artifact_ids)
    )
    eval_context_flat = base.precompute_trunks_v1(
        eval_context_ids,
        role="eval",
        kind="context",
        ledger=ledger,
        bound_reader=lambda artifact_id: read_bound_rgb_bytes_v1(
            evaluation.bundle, artifact_id
        ),
        trunk=dino,
    )
    eval_layout = benchmark.extract_dense_trunk_layout_v1(
        evaluation.plan,
        eval_context_ids,
        eval_context_flat,
        include_successors=False,
    )
    if eval_layout.successor_trunk_tokens is not None:
        raise RecoveryError("replacement evaluation exposed successor trunks")
    evaluation_result = evaluate_v1(
        train=train,
        evaluation=evaluation,
        eval_context_trunks=eval_layout.context_trunk_tokens,
        task=task,
        physical_payload=physical_payload,
        joint_payload=joint_payload,
        dino=dino,
        model_class=GroundedDenseDINOJointJEPAV1,
        benchmark=benchmark,
        device=device,
    )
    access_audit = recovery_access_audit_v1(
        ledger,
        checkpoint_reads={
            "physical_only_matched": 1,
            "joint_jepa_grounded": 1,
        },
    )
    _source_bindings_unchanged(authority)
    gate = evaluation_result["gate"]
    eligible = bool(gate["passed"])
    result = {
        "schema": SCHEMA,
        "status": (
            "PASS_CLOSED_LOOP_EXPERIMENT_ELIGIBLE"
            if eligible
            else "FAIL_STOP_GROUNDED_DENSE_DINO_MECHANISM"
        ),
        "closed_loop_eligible": eligible,
        "evaluation_replacement": True,
        "fresh_or_held_out_evaluation": False,
        "predecessor_checkpoint_qualification": checkpoint_qualification,
        "evaluation": evaluation_result,
        "gate": gate,
        "runtime_provenance": {
            "determinism": determinism,
            "environment": dict(authority["environment"]),
            "dino": dict(authority["dino"]),
            "train_role_identity_sha256": train.identity_sha256,
            "eval_role_identity_sha256": evaluation.identity_sha256,
            "task_action_train_identity_sha256": task.identity_sha256,
            "predecessor_attempt_status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
        },
        "access_audit": access_audit,
        "authority": dict(authority),
    }
    base._write_json_exclusive(output_root / "result.json", result)  # noqa: SLF001
    result_binding = base.file_binding_v1(output_root / "result.json")
    base._write_json_exclusive(  # noqa: SLF001
        output_root / "terminal.json",
        {
            "schema": TERMINAL_SCHEMA,
            "status": result["status"],
            "closed_loop_eligible": eligible,
            "retry_authorized": False,
            "result_binding": result_binding,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority = _load_authority_v1(
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"]))
    existed = output_root.exists()
    try:
        result = execute_v1(authority)
    except Exception as error:
        if not existed and output_root.is_dir() and not (output_root / "terminal.json").exists():
            base._write_json_exclusive(  # noqa: SLF001
                output_root / "terminal.json",
                {
                    "schema": TERMINAL_SCHEMA,
                    "status": "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "retry_authorized": False,
                    "result_binding": None,
                },
            )
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "closed_loop_eligible": result["closed_loop_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
