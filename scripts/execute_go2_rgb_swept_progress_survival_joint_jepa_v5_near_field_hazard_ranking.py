#!/usr/bin/env python3
"""Run the one-shot V5 near-field-hazard-ranking development probe.

V5 reconstructs the unchanged V4 model from the accepted N320 encoder and
adds only the preregistered, parameter-free near-field hazard-ranking loss to
joint training.  No predecessor runtime artifact is named, opened, or used.
Passing this executor's unchanged V4 full-arm gate only stages a development
checkpoint for the separately frozen physical-evidence calibration.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import io
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_v4 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v4_"
    "residual_local_semantic_decoder"
)
_v3 = _v4._v3
_v1 = _v4._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/"
    "go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking/"
    "attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_"
    "hazard_ranking_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_"
    "hazard_ranking_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_"
    "hazard_ranking_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_"
    "hazard_ranking_failure_v1"
)

PREREGISTRATION_COMMIT = "7fe075d752b5d14c539eaed213c9f28510659c79"

LABEL_ROOT_RELATIVE_PATH = _v4.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v4.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v4.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v4.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v4.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v4.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v4.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v4.ACTION_ORDER
ROLE_FILES = _v4.ROLE_FILES
MICROBATCH_SIZE = _v4.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v4.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v4.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v4.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v4.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v4.CONSTRUCTOR_INITIALIZATION_SEED
SEMANTIC_DECODER_INITIALIZATION_SEED = (
    _v4.SEMANTIC_DECODER_INITIALIZATION_SEED
)
EXPERIMENT_SEED = _v4.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v4.BOOTSTRAP_SEED
CONTROL_NAMES = _v4.CONTROL_NAMES
ALL_ARM_NAMES = _v4.ALL_ARM_NAMES
REGISTERED_FAMILIES = _v4.REGISTERED_FAMILIES
GATE_THRESHOLDS = _v4.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v4.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v4.AUXILIARY_OBJECTIVE)

NEAR_FIELD_RANGE_M = 2.0
HAZARD_RANKING_COEFFICIENT = 1.0
HAZARD_RANKING_NORMALIZATION = math.log(2.0)
HAZARD_RANKING_OBJECTIVE = {
    "schema": "lewm_near_field_hazard_ranking_objective_v1",
    "coefficient": HAZARD_RANKING_COEFFICIENT,
    "new_parameter_count": 0,
    "joint_from_update_one": True,
    "raster_centers_m": {
        "forward": {"start": -0.95, "stop": 5.35, "count": 64},
        "left": {"start": -3.15, "stop": 3.15, "count": 64},
    },
    "near_field": "euclidean_distance_lte_2m",
    "near_field_range_m": NEAR_FIELD_RANGE_M,
    "near_field_cell_count": 1_016,
    "hazard_score": "occupied_logit-logsumexp(unknown_logit,free_logit)",
    "positive_cells": "true_occupied_inside_near_field",
    "negative_cells": "true_free_inside_near_field",
    "pair_set": "complete_cartesian_per_raster_row",
    "per_pair": "softplus(free_hazard-occupied_hazard)/log(2)",
    "normalization": HAZARD_RANKING_NORMALIZATION,
    "row_reduction": "arithmetic_mean_all_pairs",
    "view_reduction": "equal_mean_of_present_current_and_next_view_means",
    "inactive": "exact_graph_connected_zero_when_neither_view_is_eligible",
    "sampling_or_mining": False,
    "margin": 0.0,
}

scientific_metrics_v5 = _v4.scientific_metrics_v4
semantic_metrics_v5 = _v4.semantic_metrics_v4
paired_control_comparison_v5 = _v4.paired_control_comparison_v4
evaluate_gate_v5 = _v4.evaluate_gate_v4


def hazard_ranking_objective_receipt_v5() -> dict[str, Any]:
    return copy.deepcopy(HAZARD_RANKING_OBJECTIVE)


def inherited_v4_model_receipt_v5(
    decoder_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lewm_v5_inherited_v4_model_identity_v1",
        "model_changed_from_v4": False,
        "model_class": (
            "GeometryAnchoredSweptProgressSurvivalJointJepaV4"
        ),
        "residual_local_semantic_decoder_changed_from_v4": False,
        "initial_v4_semantic_decoder": copy.deepcopy(dict(decoder_receipt)),
    }


def _fresh_output_root_v5(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh near-field-hazard-ranking attempt_v1 exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v5(
    training_v1: Any, training_v3: Any, training_v5: Any
) -> None:
    """Fail closed unless V4's core and V5's sole loss delta are exact."""

    _v4._validate_training_core_v4(training_v1, training_v3)
    for name in (
        "ACTION_ORDER",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX",
        "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        if getattr(training_v5, name, None) != getattr(training_v3, name):
            raise PermissionError(f"V5 training core changed inherited {name}")
    if (
        getattr(training_v5, "RASTER_SIZE", None) != 64
        or getattr(training_v5, "FORWARD_MIN_M", None) != -0.95
        or getattr(training_v5, "FORWARD_MAX_M", None) != 5.35
        or getattr(training_v5, "LEFT_MIN_M", None) != -3.15
        or getattr(training_v5, "LEFT_MAX_M", None) != 3.15
        or getattr(training_v5, "NEAR_FIELD_RANGE_M", None)
        != NEAR_FIELD_RANGE_M
        or getattr(training_v5, "NEAR_FIELD_CELL_COUNT", None) != 1_016
        or getattr(training_v5, "HAZARD_RANKING_COEFFICIENT", None)
        != HAZARD_RANKING_COEFFICIENT
        or getattr(training_v5, "HAZARD_RANKING_NORMALIZATION", None)
        != HAZARD_RANKING_NORMALIZATION
    ):
        raise PermissionError("V5 hazard-ranking constants changed")
    for name in (
        "near_field_hazard_ranking_loss_v5",
        "run_fixed_training_v5",
    ):
        if not callable(getattr(training_v5, name, None)):
            raise PermissionError(f"V5 training entrypoint {name} is absent")


def _hazard_training_receipt_v5(
    trace: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Validate and summarize all 4,000 preregistered H receipts."""

    if len(trace) != MAXIMUM_UPDATES:
        raise RuntimeError("V5 trace does not contain exactly 1,000 updates")
    totals = {
        "active_microbatch_count": 0,
        "current_eligible_row_count": 0,
        "next_eligible_row_count": 0,
        "current_ranked_pair_count": 0,
        "next_ranked_pair_count": 0,
    }
    windows: list[dict[str, Any]] = []
    window = {name: 0 for name in totals}
    h_sum = 0.0
    microbatch_count = 0
    for expected_update, row in enumerate(trace, start=1):
        if int(row.get("update", -1)) != expected_update:
            raise RuntimeError("V5 trace update order changed")
        losses = row.get("losses")
        if not isinstance(losses, Mapping) or set(losses) != {
            "S", "P", "U", "R", "O", "H", "L"
        }:
            raise RuntimeError("V5 trace loss identity changed")
        hazard = row.get("hazard_ranking_activity")
        microbatches = (
            hazard.get("microbatches") if isinstance(hazard, Mapping) else None
        )
        if not isinstance(microbatches, (list, tuple)) or len(microbatches) != (
            MICROBATCHES_PER_UPDATE
        ):
            raise RuntimeError("V5 update lacks four hazard microbatch receipts")
        update_h = 0.0
        for receipt in microbatches:
            if not isinstance(receipt, Mapping):
                raise RuntimeError("V5 hazard microbatch receipt changed type")
            h = float(receipt.get("H", math.nan))
            if not math.isfinite(h) or h < 0.0:
                raise RuntimeError("V5 hazard microbatch H is invalid")
            active = receipt.get("hazard_active")
            if not isinstance(active, bool):
                raise RuntimeError("V5 hazard activity receipt is not boolean")
            values: dict[str, int] = {}
            for name in (
                "current_eligible_row_count",
                "next_eligible_row_count",
                "current_ranked_pair_count",
                "next_ranked_pair_count",
            ):
                value = receipt.get(f"hazard_{name}")
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise RuntimeError(f"V5 hazard receipt has invalid {name}")
                values[name] = value
            if (
                values["current_eligible_row_count"] > MICROBATCH_SIZE
                or values["next_eligible_row_count"] > MICROBATCH_SIZE
            ):
                raise RuntimeError("V5 hazard eligible-row count exceeds microbatch")
            for view in ("current", "next"):
                rows = values[f"{view}_eligible_row_count"]
                pairs = values[f"{view}_ranked_pair_count"]
                if (rows == 0) != (pairs == 0) or pairs < rows:
                    raise RuntimeError("V5 hazard row/pair receipt is inconsistent")
            expected_active = bool(
                values["current_eligible_row_count"]
                or values["next_eligible_row_count"]
            )
            if active != expected_active or (not active and h != 0.0):
                raise RuntimeError("V5 inactive hazard receipt is inconsistent")
            activity = int(active)
            totals["active_microbatch_count"] += activity
            window["active_microbatch_count"] += activity
            for name, value in values.items():
                totals[name] += value
                window[name] += value
            update_h += h
            h_sum += h
            microbatch_count += 1
        if not math.isclose(
            float(losses["H"]),
            update_h / MICROBATCHES_PER_UPDATE,
            rel_tol=2.0e-6,
            abs_tol=2.0e-6,
        ):
            raise RuntimeError("V5 trace H mean disagrees with microbatch receipts")
        expected_total = sum(float(losses[name]) for name in "SPUROH")
        if not math.isclose(
            float(losses["L"]), expected_total, rel_tol=2.0e-6, abs_tol=2.0e-6
        ):
            raise RuntimeError("V5 total loss is not S+P+U+R+O+H")
        if expected_update % 100 == 0:
            windows.append(
                {
                    "start_update": expected_update - 99,
                    "end_update": expected_update,
                    **window,
                }
            )
            window = {name: 0 for name in totals}
    expected_microbatches = MAXIMUM_UPDATES * MICROBATCHES_PER_UPDATE
    if microbatch_count != expected_microbatches or len(windows) != 10:
        raise RuntimeError("V5 hazard trace accounting changed")
    return {
        "schema": "lewm_v5_near_field_hazard_training_activity_v1",
        "update_count": len(trace),
        "microbatch_count": microbatch_count,
        "inactive_microbatch_count": (
            microbatch_count - totals["active_microbatch_count"]
        ),
        **totals,
        "hazard_loss_microbatch_mean": h_sum / microbatch_count,
        "windows_100_updates": windows,
    }


def _physical_calibration_stage_v5(full_arm_passed: bool) -> Mapping[str, Any]:
    return {
        "status": (
            "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT"
            if full_arm_passed
            else "CLOSED_FULL_ARM_GATE_FAILED"
        ),
        "physical_calibration_run_in_this_attempt": False,
        "requires_full_arm_pass": True,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "threshold_tuple_count": 2_016,
        "physical_gate_passed": False,
    }


def execute_v5(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v5(repository_root)
    inherited_model_receipt: Mapping[str, Any] | None = None
    hazard_activity: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_labels_v1"
        )
        manifest, rows_by_role = _v1.load_label_bundle_v1(
            repository_root, labels_api=labels_api
        )
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        preflight = labels_api.summarize_preflight_v1(
            rows_by_role, context["schedule"]
        )
        if preflight != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")

        training_v1 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
        )
        training_v3 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux"
        )
        training_v5 = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v5_near_field_hazard_ranking"
        )
        _validate_training_core_v5(training_v1, training_v3, training_v5)
        frozen = {
            role: training_v1.freeze_role_labels_v1(rows, role=role, np=np)
            for role, rows in rows_by_role.items()
        }
        informative = {
            role: np.asarray(
                [group[0]["informative_state"] for group in labels.state_groups],
                dtype=np.bool_,
            )
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder"
        )
        _v4._validate_model_api_v4(model_api)
        parent_model_api = importlib.import_module(
            "lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1"
        )
        survival_scoring = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
        )
        metrics_api = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_metrics_v1"
        )
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {
            name: value.detach().cpu().float().contiguous().clone()
            for name, value in context["fit"].encoder.state_dict().items()
        }
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = (
            survival_scoring.build_current_frame_swept_progress_masks_v1()
        )
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(
            n320_state, masks
        ).to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        decoder_receipt = _v4._initial_decoder_receipt_v4(
            model,
            partition,
            torch=torch,
            inherited_semantic_method=(
                parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
                semantic_logits_from_latent
            ),
        )
        inherited_model_receipt = inherited_v4_model_receipt_v5(decoder_receipt)
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(
            name.startswith("predictor.swept_progress_head.")
            for name in partition.names["predictor"]
        ):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, training_diagnostics = (
            training_v5.run_fixed_training_v5(
                model,
                optimizer,
                context["loader"],
                pairs["train"],
                frozen["train"],
                context["schedule"],
                context["device"],
            )
        )
        accounting = dict(accounting_state.__dict__)
        hazard_activity = _hazard_training_receipt_v5(trace)
        model.eval()
        model.requires_grad_(False)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "development_only": True,
                "resume_authorized": False,
                "qualified": False,
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": "exact_n320_encoder_only",
                "predecessor_experiment_checkpoint_read": False,
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "hazard_ranking_objective": hazard_ranking_objective_receipt_v5(),
                "inherited_v4_model": inherited_model_receipt,
                "hazard_ranking_activity": hazard_activity,
                "training_diagnostics": training_diagnostics,
                "accounting": accounting,
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _v1._atomic_write_v1(
            output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue()
        )
        _, trace_binding = _v1._write_json_v1(
            output / "training_trace.json",
            {
                "schema": TRACE_SCHEMA,
                "status": "COMPLETE",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "hazard_ranking_objective": hazard_ranking_objective_receipt_v5(),
                "hazard_ranking_activity": hazard_activity,
                "training_diagnostics": training_diagnostics,
                "accounting": accounting,
                "rows": list(trace),
            },
        )

        action_prior_m = (
            frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64)
            * PROGRESS_SEGMENT_M
        )
        scored = {
            role: _v1.score_role_v1(
                model,
                context["loader"],
                pairs[role],
                frozen[role],
                action_prior_m,
                context["device"],
                torch=torch,
                np=np,
                training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            )
            for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v5(
                    scored[role]["scores_m"][arm],
                    frozen[role].prefix_lengths,
                    informative[role],
                    frozen[role].scene_ids,
                    frozen[role].family_ids,
                    np=np,
                )
                for arm in ALL_ARM_NAMES
            }
            for role in scored
        }
        selection_semantic = semantic_metrics_v5(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v5(
                selection_scores["full"],
                selection_scores[name],
                selection_labels.prefix_lengths,
                informative["checkpoint_selection"],
                selection_labels.scene_ids,
                selection_labels.family_ids,
                np=np,
            )
            for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v5(
            role_metrics["checkpoint_selection"], selection_semantic, comparisons
        )
        full_arm_passed = bool(gate["passed"])
        calibration_stage = _physical_calibration_stage_v5(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(
                current_frame_persistence_masks
            ),
        }
        result, _ = _v1._write_json_v1(
            output / "result.json",
            {
                "schema": RESULT_SCHEMA,
                "status": (
                    "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION"
                    if full_arm_passed
                    else "FAIL_DEVELOPMENT_FULL_ARM"
                ),
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "full_arm_gate": gate,
                "gate": gate,
                "physical_evidence_calibration": calibration_stage,
                "caps": {
                    "updates": MAXIMUM_UPDATES,
                    "microbatch_graphs": 4_000,
                    "presentations": MAXIMUM_PRESENTATIONS,
                },
                "seeds": {
                    "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                    "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                    "experiment_and_stochastic_execution": EXPERIMENT_SEED,
                    "bootstrap": BOOTSTRAP_SEED,
                },
                "label_manifest": {
                    "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                    "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                    "content_sha256": manifest["content_sha256"],
                    "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                    "role_files": manifest["files"],
                },
                "n320": {
                    "gate_content_sha256": context["n320_gate"]["content_sha256"],
                    "checkpoint": context["n320_checkpoint"],
                    "encoder_only_initialization": True,
                    "predecessor_experiment_checkpoint_read": False,
                },
                "hardware": context["hardware"],
                "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
                "masks": mask_receipts,
                "scientific_change_from_v4": {
                    "only_change": "near_field_hazard_ranking_loss",
                    "inherited_v4_model": inherited_model_receipt,
                    "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                    "hazard_ranking_objective": hazard_ranking_objective_receipt_v5(),
                    "model_changed": False,
                    "data_changed": False,
                    "optimizer_rules_changed": False,
                    "losses_changed": True,
                    "schedule_changed": False,
                    "evaluation_changed": False,
                },
                "training": {
                    "core": (
                        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_"
                        "v5_near_field_hazard_ranking"
                    ),
                    "accounting": accounting,
                    "diagnostics": training_diagnostics,
                    "hazard_ranking_activity": hazard_activity,
                    "joint_from_update_one": True,
                    "separate_head_or_predictor_training": False,
                    "checkpoint": checkpoint_binding,
                    "trace": trace_binding,
                },
                "action_prior_mean_progress_m": action_prior_m.tolist(),
                "roles": role_metrics,
                "selection_semantic": selection_semantic,
                "selection_control_comparisons": comparisons,
                "wrong_rgb_mapping_sha256": {
                    role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored
                },
                "determinism": {
                    "algorithms_enabled": bool(
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "warn_only": True,
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                },
                "access": access_receipt,
                "authority": {
                    "development_only": True,
                    "g2_navigation_final_evaluation_opened": False,
                    "heldout_or_sealed_opened": False,
                    "physical_evidence_gate_passed": False,
                    "checkpoint_qualified": False,
                    "promotion_performed": False,
                    "retry_or_resume_authorized": False,
                },
            },
        )
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (
            output / "failure.json"
        ).exists():
            try:
                fallback_decoder = _v4.semantic_decoder_architecture_receipt_v4()
                _v1._write_json_v1(
                    output / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "FAILED_NO_RETRY_OR_RESUME",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "preregistration_commit": PREREGISTRATION_COMMIT,
                        "hazard_ranking_objective": hazard_ranking_objective_receipt_v5(),
                        "hazard_ranking_activity": hazard_activity,
                        "inherited_v4_model": (
                            inherited_model_receipt
                            if inherited_model_receipt is not None
                            else inherited_v4_model_receipt_v5(fallback_decoder)
                        ),
                        "predecessor_experiment_checkpoint_read": False,
                        "physical_calibration_run_in_this_attempt": False,
                        "authority": {
                            "development_only": True,
                            "g2_navigation_final_evaluation_opened": False,
                            "heldout_or_sealed_opened": False,
                            "checkpoint_qualified": False,
                            "retry_or_resume_authorized": False,
                        },
                    },
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v5(repository_root=args.repository_root)
    print(
        _v1._canonical_json_bytes(
            {
                "status": result["status"],
                "result": f"{OUTPUT_RELATIVE_PATH}/result.json",
            }
        ).decode("utf-8")
    )
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
