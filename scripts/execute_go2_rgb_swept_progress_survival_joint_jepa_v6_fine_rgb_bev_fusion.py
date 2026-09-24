#!/usr/bin/env python3
"""Execute the single fresh V6 fine-RGB BEV-fusion development probe."""
from __future__ import annotations

import argparse
import copy
import importlib
import io
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
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v6_"
    "fine_rgb_bev_fusion/attempt_v1"
)
CHECKPOINT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion_checkpoint_v1"
TRACE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion_trace_v1"
RESULT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion_result_v1"
FAILURE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion_failure_v1"
PREREGISTRATION_COMMIT = "cc9ec66d796b37724e0a9e15d737813817e95265"

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
SEMANTIC_DECODER_INITIALIZATION_SEED = _v4.SEMANTIC_DECODER_INITIALIZATION_SEED
EXPERIMENT_SEED = _v4.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v4.BOOTSTRAP_SEED
CONTROL_NAMES = _v4.CONTROL_NAMES
ALL_ARM_NAMES = _v4.ALL_ARM_NAMES
GATE_THRESHOLDS = _v4.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v4.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v4.AUXILIARY_OBJECTIVE)

FINE_RGB_BRANCH_INITIALIZATION_SEED = 20_260_714
FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT = 12_256
FINE_RGB_ARCHITECTURE = {
    "schema": "lewm_fine_rgb_bev_fusion_v6_architecture_v1",
    "input": {"source": "normalized_rgb", "shape": [3, 112, 112]},
    "branch": [
        {"type": "Conv2d", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True},
        {"type": "GELU", "approximate": "none"},
        {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "bias": True},
        {"type": "GELU", "approximate": "none"},
        {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 1, "stride": 1, "padding": 0, "bias": True, "initialization": "exact_zero"},
    ],
    "initialization_seed": FINE_RGB_BRANCH_INITIALIZATION_SEED,
    "added_trainable_parameter_count": FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT,
    "sampling": {"grids": "exact_inherited_four", "weights": "exact_inherited_normalized", "mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
    "fusion": "inherited_final_latent_plus_weighted_fine_residual",
    "invalid_cells": "exact_zero_residual",
    "consumers": ["semantic_decoder", "action_conditioned_jepa_predictor", "online_and_ema_persistence"],
}

scientific_metrics_v6 = _v4.scientific_metrics_v4
semantic_metrics_v6 = _v4.semantic_metrics_v4
paired_control_comparison_v6 = _v4.paired_control_comparison_v4
evaluate_gate_v6 = _v4.evaluate_gate_v4


def fine_rgb_architecture_receipt_v6() -> dict[str, Any]:
    return copy.deepcopy(FINE_RGB_ARCHITECTURE)


def _fresh_output_root_v6(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh fine-RGB-BEV-fusion attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v6(training_v1: Any, training_v3: Any, training_v6: Any) -> None:
    _v4._validate_training_core_v4(training_v1, training_v3)
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        if getattr(training_v6, name, None) != getattr(training_v3, name):
            raise PermissionError(f"V6 training wrapper changed inherited {name}")
    if (
        getattr(training_v6, "FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6", None)
        != FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT
        or not callable(getattr(training_v6, "run_fixed_training_v6", None))
    ):
        raise PermissionError("V6 training wrapper contract changed")


def _validate_model_api_v6(model_api: Any) -> None:
    if (
        getattr(model_api, "FINE_RGB_BRANCH_INITIALIZATION_SEED_V6", None)
        != FINE_RGB_BRANCH_INITIALIZATION_SEED
        or getattr(model_api, "FINE_RGB_BRANCH_ADDED_TRAINABLE_PARAMETER_COUNT_V6", None)
        != FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT
        or not callable(getattr(model_api, "GeometryAnchoredSweptProgressSurvivalJointJepaV6", None))
    ):
        raise PermissionError("V6 model API or frozen fine-RGB constants changed")


def _initial_model_receipt_v6(
    model: Any,
    partition: Any,
    *,
    torch: Any,
    model_api: Any,
    inherited_semantic_method: Any,
) -> Mapping[str, Any]:
    online = model.bev_lift.fine_rgb_branch
    target = model.target_bev_lift.fine_rgb_branch
    nn = torch.nn
    expected = (
        (online.conv1, 3, 32, (3, 3), (1, 1), (1, 1)),
        (online.conv2, 32, 32, (3, 3), (1, 1), (1, 1)),
        (online.output, 32, 64, (1, 1), (1, 1), (0, 0)),
    )
    for layer, in_channels, out_channels, kernel, stride, padding in expected:
        if not isinstance(layer, nn.Conv2d) or (
            layer.in_channels, layer.out_channels, layer.kernel_size,
            layer.stride, layer.padding, layer.bias is not None,
        ) != (in_channels, out_channels, kernel, stride, padding, True):
            raise RuntimeError("V6 fine-RGB convolution architecture changed")
    for activation in (online.activation1, online.activation2):
        if not isinstance(activation, nn.GELU) or activation.approximate != "none":
            raise RuntimeError("V6 fine-RGB GELU changed")
    online_named = tuple(
        name for name in partition.names["lift_semantic"]
        if name.startswith("bev_lift.fine_rgb_branch.")
    )
    target_named = tuple(
        name for name in partition.names["target"]
        if name.startswith("target_bev_lift.fine_rgb_branch.")
    )
    if len(online_named) != 6 or len(target_named) != 6:
        raise RuntimeError("V6 fine-RGB parameter partition changed")
    online_parameters = tuple(online.parameters())
    target_parameters = tuple(target.parameters())
    if sum(parameter.numel() for parameter in online_parameters) != FINE_RGB_BRANCH_ADDED_PARAMETER_COUNT:
        raise RuntimeError("V6 fine-RGB parameter count changed")
    if any(parameter.requires_grad for parameter in target_parameters):
        raise RuntimeError("V6 target fine-RGB branch is trainable")
    if int(torch.count_nonzero(online.output.weight)) or int(torch.count_nonzero(online.output.bias)):
        raise RuntimeError("V6 fine-RGB output is not exactly zero initialized")
    if any(not torch.equal(left.detach(), right.detach()) for left, right in zip(online_parameters, target_parameters, strict=True)):
        raise RuntimeError("V6 target fine-RGB branch is not an exact initial copy")
    inherited_decoder = _v4._initial_decoder_receipt_v4(
        model,
        partition,
        torch=torch,
        inherited_semantic_method=inherited_semantic_method,
    )
    parameter = next(model.parameters())
    probe = torch.zeros((1, 3, 112, 112), dtype=parameter.dtype, device=parameter.device)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        patch_tokens = model.encoder.forward_tokens(probe)[:, 1:]
        inherited_sampling = model.bev_lift.forward_with_sampling(patch_tokens)
        fused_sampling = model_api._fuse_fine_rgb_v6(probe, inherited_sampling, online)
        latent_parity = torch.equal(inherited_sampling.latent, fused_sampling.latent)
        semantic_parity = torch.equal(
            model.semantic_logits_from_latent(inherited_sampling.latent),
            model.semantic_logits_from_latent(fused_sampling.latent),
        )
    model.train(was_training)
    if not latent_parity or not semantic_parity:
        raise RuntimeError("V6 initial zero-projection parity with V4 failed")
    return {
        "architecture": fine_rgb_architecture_receipt_v6(),
        "inherited_v4_decoder": inherited_decoder,
        "initial_projection_exactly_zero": True,
        "initial_fused_latent_parity_with_v4": True,
        "initial_semantic_logits_parity_with_v4": True,
        "initial_predictor_input_parity_with_v4": True,
        "parity_basis": "bitwise synthetic forward through actual fusion route",
        "online_parameter_count": sum(parameter.numel() for parameter in online_parameters),
        "online_parameter_tensor_count": len(online_parameters),
        "target_parameter_count": sum(parameter.numel() for parameter in target_parameters),
        "all_online_parameters_in_lift_semantic_exactly_once": True,
        "all_target_parameters_frozen_in_target_exactly_once": True,
        "target_initial_copy_exact": True,
    }


def _physical_calibration_stage_v6(full_arm_passed: bool) -> Mapping[str, Any]:
    return {
        "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED",
        "physical_calibration_run_in_this_attempt": False,
        "requires_full_arm_pass": True,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "threshold_tuple_count": 2_016,
        "physical_gate_passed": False,
    }


def execute_v6(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v6(repository_root)
    initial_model: Mapping[str, Any] | None = None
    branch_activity: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_labels_v1")
        manifest, rows_by_role = _v1.load_label_bundle_v1(repository_root, labels_api=labels_api)
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        if labels_api.summarize_preflight_v1(rows_by_role, context["schedule"]) != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_v1 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1")
        training_v3 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux")
        training_v6 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion")
        _validate_training_core_v6(training_v1, training_v3, training_v6)
        frozen = {role: training_v1.freeze_role_labels_v1(rows, role=role, np=np) for role, rows in rows_by_role.items()}
        informative = {
            role: np.asarray([group[0]["informative_state"] for group in labels.state_groups], dtype=np.bool_)
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion")
        _validate_model_api_v6(model_api)
        parent_model_api = importlib.import_module("lewm.models.geometry_anchored_deformable_bev_lift_joint_jepa_v1")
        survival_scoring = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1")
        metrics_api = importlib.import_module("lewm.benchmarks.go2_post_action_projective_support_metrics_v1")
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        n320_state = {name: value.detach().cpu().float().contiguous().clone() for name, value in context["fit"].encoder.state_dict().items()}
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = survival_scoring.build_current_frame_swept_progress_masks_v1()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV6(n320_state, masks).to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v6(
            model,
            partition,
            torch=torch,
            model_api=model_api,
            inherited_semantic_method=(
                parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
                semantic_logits_from_latent
            ),
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(name.startswith("predictor.swept_progress_head.") for name in partition.names["predictor"]):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, training_diagnostics = training_v6.run_fixed_training_v6(
            model, optimizer, context["loader"], pairs["train"], frozen["train"], context["schedule"], context["device"]
        )
        accounting = dict(accounting_state.__dict__)
        branch_activity = training_diagnostics["fine_rgb_branch"]
        model.eval()
        model.requires_grad_(False)
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        checkpoint_buffer = io.BytesIO()
        torch.save({
            "schema": CHECKPOINT_SCHEMA, "development_only": True,
            "resume_authorized": False, "qualified": False,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
            "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
            "fine_rgb_branch_initialization_seed": FINE_RGB_BRANCH_INITIALIZATION_SEED,
            "experiment_seed": EXPERIMENT_SEED,
            "initialization_source": "exact_n320_encoder_only",
            "predecessor_experiment_checkpoint_read": False,
            "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
            "initial_v6_model": initial_model, "fine_rgb_branch_activity": branch_activity,
            "training_diagnostics": training_diagnostics, "accounting": accounting,
            "model_state_dict": state,
        }, checkpoint_buffer)
        checkpoint_binding = _v1._atomic_write_v1(output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue())
        _, trace_binding = _v1._write_json_v1(output / "training_trace.json", {
            "schema": TRACE_SCHEMA, "status": "COMPLETE",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "initial_v6_model": initial_model, "fine_rgb_branch_activity": branch_activity,
            "training_diagnostics": training_diagnostics, "accounting": accounting,
            "rows": list(trace),
        })

        action_prior_m = frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64) * PROGRESS_SEGMENT_M
        scored = {
            role: _v1.score_role_v1(
                model, context["loader"], pairs[role], frozen[role], action_prior_m,
                context["device"], torch=torch, np=np, training_core=training_v1,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            ) for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v6(
                    scored[role]["scores_m"][arm], frozen[role].prefix_lengths,
                    informative[role], frozen[role].scene_ids, frozen[role].family_ids, np=np,
                ) for arm in ALL_ARM_NAMES
            } for role in scored
        }
        selection_semantic = semantic_metrics_v6(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"], np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v6(
                selection_scores["full"], selection_scores[name], selection_labels.prefix_lengths,
                informative["checkpoint_selection"], selection_labels.scene_ids,
                selection_labels.family_ids, np=np,
            ) for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v6(role_metrics["checkpoint_selection"], selection_semantic, comparisons)
        full_arm_passed = bool(gate["passed"])
        calibration_stage = _physical_calibration_stage_v6(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(current_frame_persistence_masks),
        }
        result, _ = _v1._write_json_v1(output / "result.json", {
            "schema": RESULT_SCHEMA,
            "status": "PASS_FULL_ARM_STAGED_FOR_PHYSICAL_CALIBRATION" if full_arm_passed else "FAIL_DEVELOPMENT_FULL_ARM",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "full_arm_gate": gate, "gate": gate,
            "physical_evidence_calibration": calibration_stage,
            "caps": {"updates": MAXIMUM_UPDATES, "microbatch_graphs": 4_000, "presentations": MAXIMUM_PRESENTATIONS},
            "seeds": {
                "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "fine_rgb_branch_isolated": FINE_RGB_BRANCH_INITIALIZATION_SEED,
                "experiment_and_stochastic_execution": EXPERIMENT_SEED, "bootstrap": BOOTSTRAP_SEED,
            },
            "label_manifest": {
                "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                "content_sha256": manifest["content_sha256"], "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                "role_files": manifest["files"],
            },
            "n320": {
                "gate_content_sha256": context["n320_gate"]["content_sha256"],
                "checkpoint": context["n320_checkpoint"], "encoder_only_initialization": True,
                "predecessor_experiment_checkpoint_read": False,
            },
            "hardware": context["hardware"],
            "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
            "masks": mask_receipts,
            "scientific_change_from_v4": {
                "only_change": "fine_rgb_bev_fusion_branch", "initial_v6_model": initial_model,
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "model_changed": True, "data_changed": False,
                "optimizer_rules_changed": False, "optimizer_membership_changed_for_new_parameters": True,
                "losses_changed": False, "schedule_changed": False, "evaluation_changed": False,
            },
            "training": {
                "core": "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v6_fine_rgb_bev_fusion",
                "accounting": accounting, "diagnostics": training_diagnostics,
                "fine_rgb_branch_activity": branch_activity, "joint_from_update_one": True,
                "separate_head_or_predictor_training": False,
                "checkpoint": checkpoint_binding, "trace": trace_binding,
            },
            "action_prior_mean_progress_m": action_prior_m.tolist(), "roles": role_metrics,
            "selection_semantic": selection_semantic, "selection_control_comparisons": comparisons,
            "wrong_rgb_mapping_sha256": {role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored},
            "determinism": {
                "algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()), "warn_only": True,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            },
            "access": access_receipt,
            "authority": {
                "development_only": True, "g2_navigation_final_evaluation_opened": False,
                "heldout_or_sealed_opened": False, "physical_evidence_gate_passed": False,
                "checkpoint_qualified": False, "promotion_performed": False,
                "retry_or_resume_authorized": False,
            },
        })
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (output / "failure.json").exists():
            try:
                _v1._write_json_v1(output / "failure.json", {
                    "schema": FAILURE_SCHEMA, "status": "FAILED_NO_RETRY_OR_RESUME",
                    "error_type": type(error).__name__, "error_message": str(error),
                    "traceback": traceback.format_exc(), "preregistration_commit": PREREGISTRATION_COMMIT,
                    "fine_rgb_architecture": fine_rgb_architecture_receipt_v6(),
                    "initial_v6_model": initial_model, "fine_rgb_branch_activity": branch_activity,
                    "predecessor_experiment_checkpoint_read": False,
                    "physical_calibration_run_in_this_attempt": False,
                    "authority": {
                        "development_only": True, "g2_navigation_final_evaluation_opened": False,
                        "heldout_or_sealed_opened": False, "checkpoint_qualified": False,
                        "retry_or_resume_authorized": False,
                    },
                })
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v6(repository_root=args.repository_root)
    print(_v1._canonical_json_bytes({"status": result["status"], "result": f"{OUTPUT_RELATIVE_PATH}/result.json"}).decode("utf-8"))
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
