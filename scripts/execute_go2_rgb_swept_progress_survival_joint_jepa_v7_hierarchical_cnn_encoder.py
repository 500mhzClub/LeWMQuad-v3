#!/usr/bin/env python3
"""Execute the single fresh V7 hierarchical-CNN development probe."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
_v1 = _v4._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v7_"
    "hierarchical_cnn_encoder/attempt_v1"
)
CHECKPOINT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder_checkpoint_v1"
TRACE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder_trace_v1"
RESULT_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder_result_v1"
FAILURE_SCHEMA = "lewm_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder_failure_v1"
PREREGISTRATION_COMMIT = "34c4a33e2fa25926b3127e0c893755757426cfd4"

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

HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED = 20_260_715
HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT = 1_994_880
HIERARCHICAL_CNN_ARCHITECTURE = {
    "schema": "lewm_hierarchical_cnn_encoder_v7_architecture_v1",
    "input": {"source": "normalized_rgb", "shape": [3, 112, 112]},
    "stem": {"conv": [3, 48, 5, 2, 2], "group_norm": [6, 48], "activation": "GELU_none"},
    "stages": [
        {"width": 48, "residual_blocks": 2, "group_norm_groups": 6},
        {"downsample_conv": [48, 96, 3, 2, 1], "width": 96, "residual_blocks": 2, "group_norm_groups": 8},
        {"downsample_conv": [96, 192, 3, 2, 1], "width": 192, "residual_blocks": 2, "group_norm_groups": 12},
    ],
    "residual_block": "conv3x3_groupnorm_gelu_conv3x3_groupnorm_residual_gelu",
    "spatial_adapter": {"type": "bilinear_interpolation", "size": [16, 16], "align_corners": False},
    "output_projection": {"conv": [192, 192, 1, 1, 0]},
    "tokens": {"spatial_count": 256, "dimension": 192, "order": "row_major", "cls": "mean_spatial", "output_shape": [257, 192]},
    "initialization_seed": HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED,
    "trainable_parameter_count": HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT,
}

scientific_metrics_v7 = _v4.scientific_metrics_v4
semantic_metrics_v7 = _v4.semantic_metrics_v4
paired_control_comparison_v7 = _v4.paired_control_comparison_v4
evaluate_gate_v7 = _v4.evaluate_gate_v4


def hierarchical_cnn_architecture_receipt_v7() -> dict[str, Any]:
    return copy.deepcopy(HIERARCHICAL_CNN_ARCHITECTURE)


def _fresh_output_root_v7(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh hierarchical-CNN attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v7(training_v3: Any, training_v7: Any) -> None:
    _v4._validate_training_core_v4(_v1_training(), training_v3)
    for name in (
        "ACTION_ORDER", "MICROBATCH_SIZE", "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE", "MAXIMUM_UPDATES", "MAXIMUM_PRESENTATIONS",
        "OCCUPIED_CLASS_INDEX", "OCCUPIED_SAFETY_AUX_COEFFICIENT",
        "OCCUPIED_SAFETY_AUX_NORMALIZATION",
    ):
        if getattr(training_v7, name, None) != getattr(training_v3, name):
            raise PermissionError(f"V7 training wrapper changed inherited {name}")
    if (
        getattr(training_v7, "HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7", None)
        != HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT
        or not callable(getattr(training_v7, "run_fixed_training_v7", None))
    ):
        raise PermissionError("V7 training wrapper contract changed")


def _v1_training() -> Any:
    return importlib.import_module(
        "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
    )


def _validate_model_api_v7(model_api: Any) -> None:
    if (
        getattr(model_api, "HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED_V7", None)
        != HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED
        or getattr(model_api, "HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT_V7", None)
        != HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT
        or not callable(getattr(model_api, "HierarchicalCnnEncoderV7", None))
        or not callable(getattr(model_api, "GeometryAnchoredSweptProgressSurvivalJointJepaV7", None))
    ):
        raise PermissionError("V7 model API or hierarchical-CNN constants changed")


def _names_sha256_v7(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _inherited_components_receipt_v7(
    model: Any, clean_v4: Any, *, torch: Any
) -> Mapping[str, Any]:
    excluded = ("encoder.", "target_encoder.")
    actual = {
        name: value for name, value in model.state_dict().items()
        if not name.startswith(excluded)
    }
    reference = {
        name: value for name, value in clean_v4.state_dict().items()
        if not name.startswith(excluded)
    }
    if actual.keys() != reference.keys() or any(
        not torch.equal(actual[name].detach().cpu(), reference[name].detach().cpu())
        for name in actual
    ):
        raise RuntimeError("V7 changed an inherited V4 non-encoder state tensor")
    parameter_names = tuple(
        name for name, _ in model.named_parameters()
        if not name.startswith(excluded)
    )
    parameter_count = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith(excluded)
    )
    return {
        "reference": "fresh clean V4 construction with identical inputs",
        "excluded_replacement_prefixes": list(excluded),
        "state_tensor_count": len(actual),
        "state_name_inventory_sha256": _names_sha256_v7(tuple(actual)),
        "parameter_tensor_count": len(parameter_names),
        "parameter_count": parameter_count,
        "parameter_name_inventory_sha256": _names_sha256_v7(parameter_names),
        "initial_state_exactly_equal": True,
    }


def _initial_model_receipt_v7(
    model: Any,
    partition: Any,
    inherited_components: Mapping[str, Any],
    *,
    torch: Any,
    model_api: Any,
    inherited_semantic_method: Any,
) -> Mapping[str, Any]:
    if not isinstance(model.encoder, model_api.HierarchicalCnnEncoderV7) or not isinstance(
        model.target_encoder, model_api.HierarchicalCnnEncoderV7
    ):
        raise RuntimeError("V7 hierarchical CNN type changed")
    online = tuple(model.encoder.named_parameters())
    target = tuple(model.target_encoder.named_parameters())
    if tuple(name for name, _ in online) != tuple(name for name, _ in target):
        raise RuntimeError("V7 online/target CNN inventories differ")
    if sum(parameter.numel() for _, parameter in online) != HIERARCHICAL_CNN_ENCODER_TRAINABLE_PARAMETER_COUNT:
        raise RuntimeError("V7 hierarchical CNN parameter count changed")
    if any(not parameter.requires_grad for _, parameter in online):
        raise RuntimeError("V7 online CNN is frozen")
    if any(parameter.requires_grad for _, parameter in target):
        raise RuntimeError("V7 target CNN is trainable")
    if any(
        not torch.equal(left.detach(), right.detach())
        for (_, left), (_, right) in zip(online, target, strict=True)
    ):
        raise RuntimeError("V7 target CNN is not an exact initial copy")
    online_names = tuple(f"encoder.{name}" for name, _ in online)
    target_names = tuple(f"target_encoder.{name}" for name, _ in target)
    if tuple(partition.names["encoder"]) != online_names:
        raise RuntimeError("V7 online CNN partition coverage changed")
    partition_target = tuple(
        name for name in partition.names["target"]
        if name.startswith("target_encoder.")
    )
    if partition_target != target_names:
        raise RuntimeError("V7 target CNN partition coverage changed")
    parameter = next(model.parameters())
    probe = torch.zeros((1, 3, 112, 112), dtype=parameter.dtype, device=parameter.device)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        online_tokens = model.encoder.forward_tokens(probe)
        target_tokens = model.target_encoder.forward_tokens(probe)
    model.train(was_training)
    if tuple(online_tokens.shape) != (1, 257, 192):
        raise RuntimeError("V7 hierarchical CNN token contract changed")
    if not torch.equal(online_tokens, target_tokens) or not torch.equal(
        online_tokens[:, :1], online_tokens[:, 1:].mean(dim=1, keepdim=True)
    ):
        raise RuntimeError("V7 initial online/target token equality changed")
    inherited_decoder = _v4._initial_decoder_receipt_v4(
        model, partition, torch=torch,
        inherited_semantic_method=inherited_semantic_method,
    )
    return {
        "architecture": hierarchical_cnn_architecture_receipt_v7(),
        "inherited_v4_decoder": inherited_decoder,
        "inherited_nonencoder_components": dict(inherited_components),
        "n320_encoder_parameter_initialization_used_by_v7_cnn": False,
        "token_interface_shape": [257, 192],
        "online_parameter_count": sum(parameter.numel() for _, parameter in online),
        "online_parameter_tensor_count": len(online),
        "online_parameter_suffix_inventory_sha256": _names_sha256_v7(tuple(name for name, _ in online)),
        "target_parameter_count": sum(parameter.numel() for _, parameter in target),
        "all_online_parameters_in_encoder_partition_exactly_once": True,
        "all_target_parameters_frozen_in_target_partition_exactly_once": True,
        "target_initial_copy_exact": True,
        "initial_online_target_tokens_exact": True,
    }


def _physical_calibration_stage_v7(full_arm_passed: bool) -> Mapping[str, Any]:
    return {
        "status": "STAGED_FOR_SEPARATELY_FROZEN_ONE_SHOT" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED",
        "physical_calibration_run_in_this_attempt": False,
        "requires_full_arm_pass": True,
        "protocol_changed_from_reviewed_v4_calibration": False,
        "threshold_tuple_count": 2_016,
        "physical_gate_passed": False,
    }


def execute_v7(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v7(repository_root)
    initial_model: Mapping[str, Any] | None = None
    encoder_activity: Mapping[str, Any] | None = None
    try:
        labels_api = importlib.import_module("lewm.benchmarks.go2_swept_progress_survival_labels_v1")
        manifest, rows_by_role = _v1.load_label_bundle_v1(repository_root, labels_api=labels_api)
        context = _v1._prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        if labels_api.summarize_preflight_v1(rows_by_role, context["schedule"]) != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_v1 = _v1_training()
        training_v3 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux")
        training_v7 = importlib.import_module("scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder")
        _validate_training_core_v7(training_v3, training_v7)
        frozen = {role: training_v1.freeze_role_labels_v1(rows, role=role, np=np) for role, rows in rows_by_role.items()}
        informative = {
            role: np.asarray([group[0]["informative_state"] for group in labels.state_groups], dtype=np.bool_)
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_v1.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder")
        _validate_model_api_v7(model_api)
        v4_model_api = importlib.import_module("lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder")
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
        persistence_masks = survival_scoring.build_current_frame_swept_progress_masks_v1()
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV7(n320_state, masks)
        clean_v4 = v4_model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV4(n320_state, masks)
        inherited_components = _inherited_components_receipt_v7(model, clean_v4, torch=torch)
        del clean_v4
        model = model.to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
        initial_model = _initial_model_receipt_v7(
            model, partition, inherited_components, torch=torch, model_api=model_api,
            inherited_semantic_method=parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.semantic_logits_from_latent,
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(name.startswith("predictor.swept_progress_head.") for name in partition.names["predictor"]):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, diagnostics = training_v7.run_fixed_training_v7(
            model, optimizer, context["loader"], pairs["train"], frozen["train"], context["schedule"], context["device"]
        )
        accounting = dict(accounting_state.__dict__)
        encoder_activity = diagnostics["hierarchical_cnn_encoder"]
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
            "hierarchical_cnn_encoder_initialization_seed": HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED,
            "experiment_seed": EXPERIMENT_SEED,
            "initialization_source": "fresh_hierarchical_cnn_plus_inherited_v4_nonencoder_components",
            "n320_encoder_parameter_initialization_used_by_v7_cnn": False,
            "predecessor_experiment_checkpoint_read": False,
            "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
            "initial_v7_model": initial_model,
            "hierarchical_cnn_encoder_activity": encoder_activity,
            "training_diagnostics": diagnostics, "accounting": accounting,
            "model_state_dict": state,
        }, checkpoint_buffer)
        checkpoint_binding = _v1._atomic_write_v1(output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue())
        _, trace_binding = _v1._write_json_v1(output / "training_trace.json", {
            "schema": TRACE_SCHEMA, "status": "COMPLETE",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "initial_v7_model": initial_model,
            "hierarchical_cnn_encoder_activity": encoder_activity,
            "training_diagnostics": diagnostics, "accounting": accounting,
            "rows": list(trace),
        })

        action_prior_m = frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64) * PROGRESS_SEGMENT_M
        scored = {
            role: _v1.score_role_v1(
                model, context["loader"], pairs[role], frozen[role], action_prior_m,
                context["device"], torch=torch, np=np, training_core=training_v1,
                current_frame_persistence_masks=persistence_masks, metrics_api=metrics_api,
            ) for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v7(
                    scored[role]["scores_m"][arm], frozen[role].prefix_lengths,
                    informative[role], frozen[role].scene_ids, frozen[role].family_ids, np=np,
                ) for arm in ALL_ARM_NAMES
            } for role in scored
        }
        selection_semantic = semantic_metrics_v7(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"], np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v7(
                selection_scores["full"], selection_scores[name], selection_labels.prefix_lengths,
                informative["checkpoint_selection"], selection_labels.scene_ids,
                selection_labels.family_ids, np=np,
            ) for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v7(role_metrics["checkpoint_selection"], selection_semantic, comparisons)
        full_arm_passed = bool(gate["passed"])
        checkpoint_access = "STAGED_FOR_SEPARATE_PHYSICAL_CALIBRATION" if full_arm_passed else "CLOSED_FULL_ARM_GATE_FAILED"
        calibration_stage = _physical_calibration_stage_v7(full_arm_passed)
        access_receipt = _v1._access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _v1._mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _v1._mask_receipt_v1(persistence_masks),
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
                "hierarchical_cnn_encoder_isolated": HIERARCHICAL_CNN_ENCODER_INITIALIZATION_SEED,
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
                "checkpoint": context["n320_checkpoint"],
                "constructor_compatibility_validation_performed": True,
                "n320_encoder_parameter_initialization_used_by_v7_cnn": False,
                "predecessor_experiment_checkpoint_read": False,
            },
            "hardware": context["hardware"],
            "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
            "masks": mask_receipts,
            "scientific_change_from_v4": {
                "only_change": "wholesale_hierarchical_cnn_encoder_replacement",
                "initial_v7_model": initial_model,
                "inherited_nonencoder_components_unchanged": True,
                "inherited_occupied_auxiliary": dict(AUXILIARY_OBJECTIVE),
                "model_changed": True, "data_changed": False,
                "optimizer_rules_changed": False,
                "optimizer_membership_changed_for_replacement_parameters": True,
                "losses_changed": False, "schedule_changed": False, "evaluation_changed": False,
            },
            "training": {
                "core": "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v7_hierarchical_cnn_encoder",
                "accounting": accounting, "diagnostics": diagnostics,
                "hierarchical_cnn_encoder_activity": encoder_activity,
                "joint_from_update_one": True, "separate_head_or_predictor_training": False,
                "checkpoint_access_status": checkpoint_access,
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
                "checkpoint_access_authorized_for_physical_calibration": full_arm_passed,
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
                    "hierarchical_cnn_architecture": hierarchical_cnn_architecture_receipt_v7(),
                    "initial_v7_model": initial_model,
                    "hierarchical_cnn_encoder_activity": encoder_activity,
                    "n320_encoder_parameter_initialization_used_by_v7_cnn": False,
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
    result = execute_v7(repository_root=args.repository_root)
    print(_v1._canonical_json_bytes({"status": result["status"], "result": f"{OUTPUT_RELATIVE_PATH}/result.json"}).decode("utf-8"))
    return 0 if result["full_arm_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
