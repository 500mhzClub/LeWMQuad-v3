#!/usr/bin/env python3
"""Run the one-shot V4 residual-local-semantic-decoder development probe.

V4 changes only the semantic readout architecture.  It reuses the frozen V3
coefficient-0.5 joint training and every reviewed input, schedule, optimizer,
control, metric, gate, cap, and hardware binding.  No predecessor experiment
output is named or opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

_v3 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux"
)
_v1 = _v3._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/"
    "go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_semantic_decoder/"
    "attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_residual_local_"
    "semantic_decoder_failure_v1"
)

LABEL_ROOT_RELATIVE_PATH = _v3.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v3.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v3.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v3.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v3.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v3.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v3.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v3.ACTION_ORDER
ROLE_FILES = _v3.ROLE_FILES
MICROBATCH_SIZE = _v3.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v3.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v3.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v3.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v3.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v3.CONSTRUCTOR_INITIALIZATION_SEED
EXPERIMENT_SEED = _v3.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v3.BOOTSTRAP_SEED
CONTROL_NAMES = _v3.CONTROL_NAMES
ALL_ARM_NAMES = _v3.ALL_ARM_NAMES
REGISTERED_FAMILIES = _v3.REGISTERED_FAMILIES
GATE_THRESHOLDS = _v3.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v3.PROGRESS_SEGMENT_M
AUXILIARY_OBJECTIVE = dict(_v3.AUXILIARY_OBJECTIVE)

SEMANTIC_DECODER_INITIALIZATION_SEED = 20_260_713
SEMANTIC_DECODER_ADDED_PARAMETER_COUNT = 37_123
SEMANTIC_DECODER_ARCHITECTURE = {
    "schema": "lewm_residual_local_semantic_decoder_v4_architecture_v1",
    "merge": "base_logits_plus_residual_logits",
    "base": {
        "type": "Conv2d",
        "in_channels": 64,
        "out_channels": 3,
        "kernel_size": [1, 1],
        "bias": True,
        "identity": "exact_existing_v3_semantic_head",
    },
    "residual": {
        "local": {
            "type": "Conv2d",
            "in_channels": 64,
            "out_channels": 64,
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
            "bias": True,
        },
        "activation": {"type": "GELU", "approximate": "none"},
        "output": {
            "type": "Conv2d",
            "in_channels": 64,
            "out_channels": 3,
            "kernel_size": [1, 1],
            "bias": True,
            "weight_initialization": "exact_zeros",
            "bias_initialization": "exact_zeros",
        },
    },
    "added_trainable_parameter_count": SEMANTIC_DECODER_ADDED_PARAMETER_COUNT,
    "initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
    "visibility_mask": "inherited_bev_lift_anchor_in_frustum_post_logits",
    "normalization_layers": 0,
}

scientific_metrics_v4 = _v3.scientific_metrics_v3
semantic_metrics_v4 = _v3.semantic_metrics_v3
paired_control_comparison_v4 = _v3.paired_control_comparison_v3
evaluate_gate_v4 = _v3.evaluate_gate_v3


def semantic_decoder_architecture_receipt_v4() -> dict[str, Any]:
    return copy.deepcopy(SEMANTIC_DECODER_ARCHITECTURE)


def _fresh_output_root_v4(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh residual-local-decoder attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v4(training_v1: Any, training_v3: Any) -> None:
    _v3._validate_training_core_v3(training_v1, training_v3)
    if (
        getattr(training_v3, "OCCUPIED_SAFETY_AUX_COEFFICIENT", None) != 0.5
        or getattr(training_v3, "OCCUPIED_SAFETY_AUX_NORMALIZATION", None)
        != math.log(2.0)
    ):
        raise PermissionError("V4 must retain the exact V3 auxiliary")


def _validate_model_api_v4(model_api: Any) -> None:
    if (
        getattr(
            model_api,
            "RESIDUAL_LOCAL_SEMANTIC_DECODER_ADDED_PARAMETER_COUNT_V4",
            None,
        )
        != SEMANTIC_DECODER_ADDED_PARAMETER_COUNT
        or getattr(
            model_api, "RESIDUAL_BRANCH_INITIALIZATION_SEED_OFFSET_V4", None
        )
        != 1
        or not callable(
            getattr(
                model_api, "GeometryAnchoredSweptProgressSurvivalJointJepaV4", None
            )
        )
    ):
        raise PermissionError("V4 model API or frozen decoder constants changed")


def _initial_decoder_receipt_v4(
    model: Any,
    partition: Any,
    *,
    torch: Any,
    inherited_semantic_method: Any,
) -> Mapping[str, Any]:
    """Validate and receipt the fresh V4 decoder before its first update."""

    decoder = model.semantic_head
    nn = torch.nn
    for name in ("base", "local", "activation", "residual_output"):
        if not hasattr(decoder, name):
            raise RuntimeError(f"V4 semantic decoder lacks {name}")
    base, local = decoder.base, decoder.local
    activation, residual_output = decoder.activation, decoder.residual_output
    if not isinstance(base, nn.Conv2d) or (
        base.in_channels,
        base.out_channels,
        base.kernel_size,
        base.stride,
        base.padding,
        base.bias is not None,
    ) != (64, 3, (1, 1), (1, 1), (0, 0), True):
        raise RuntimeError("V4 base semantic head changed")
    if not isinstance(local, nn.Conv2d) or (
        local.in_channels,
        local.out_channels,
        local.kernel_size,
        local.stride,
        local.padding,
        local.bias is not None,
    ) != (64, 64, (3, 3), (1, 1), (1, 1), True):
        raise RuntimeError("V4 local semantic convolution changed")
    if not isinstance(activation, nn.GELU) or activation.approximate != "none":
        raise RuntimeError("V4 semantic GELU changed")
    if not isinstance(residual_output, nn.Conv2d) or (
        residual_output.in_channels,
        residual_output.out_channels,
        residual_output.kernel_size,
        residual_output.stride,
        residual_output.padding,
        residual_output.bias is not None,
    ) != (64, 3, (1, 1), (1, 1), (0, 0), True):
        raise RuntimeError("V4 residual output changed")
    if (
        int(model.config.initialization_seed) + 1
        != SEMANTIC_DECODER_INITIALIZATION_SEED
    ):
        raise RuntimeError("V4 decoder initialization seed changed")
    if type(model).semantic_logits_from_latent is not inherited_semantic_method:
        raise RuntimeError("V4 changed the inherited semantic visibility mask route")
    if int(torch.count_nonzero(residual_output.weight).item()) != 0 or int(
        torch.count_nonzero(residual_output.bias).item()
    ) != 0:
        raise RuntimeError("V4 residual output is not exactly zero initialized")

    added_parameters = tuple(local.parameters()) + tuple(residual_output.parameters())
    added_count = sum(parameter.numel() for parameter in added_parameters)
    if added_count != SEMANTIC_DECODER_ADDED_PARAMETER_COUNT:
        raise RuntimeError("V4 added semantic parameter count changed")
    semantic_parameters = tuple(decoder.parameters())
    lift_ids = {id(parameter) for parameter in partition.lift_semantic}
    other_ids = {
        id(parameter)
        for group in (partition.encoder, partition.predictor, partition.target)
        for parameter in group
    }
    if (
        not all(id(parameter) in lift_ids for parameter in semantic_parameters)
        or any(id(parameter) in other_ids for parameter in semantic_parameters)
    ):
        raise RuntimeError("V4 semantic decoder escaped the lift/semantic partition")
    inventory = [
        id(parameter)
        for group in (
            partition.encoder,
            partition.lift_semantic,
            partition.predictor,
            partition.target,
        )
        for parameter in group
    ]
    if any(inventory.count(id(parameter)) != 1 for parameter in semantic_parameters):
        raise RuntimeError("V4 semantic parameter partition is not exactly once")
    if not all(
        parameter.dtype == torch.float32
        and bool(torch.isfinite(parameter).all())
        for parameter in semantic_parameters
    ):
        raise RuntimeError("V4 semantic parameters changed dtype or finiteness")

    visibility = model.bev_lift.anchor_in_frustum.detach().cpu().contiguous()
    if tuple(visibility.shape) != (64, 64) or visibility.dtype != torch.bool:
        raise RuntimeError("V4 inherited visibility mask changed")
    visibility_payload = visibility.numpy().tobytes(order="C")
    return {
        "architecture": semantic_decoder_architecture_receipt_v4(),
        "initial_residual_output_exactly_zero": True,
        "semantic_parameter_count": sum(
            parameter.numel() for parameter in semantic_parameters
        ),
        "added_parameter_count": added_count,
        "all_semantic_parameters_in_lift_semantic_exactly_once": True,
        "visibility_mask": {
            "shape": [64, 64],
            "dtype": "bool",
            "true_cell_count": int(visibility.sum().item()),
            "sha256": hashlib.sha256(visibility_payload).hexdigest(),
            "application": "inherited_post_logits",
        },
    }


def execute_v4(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v4(repository_root)
    initial_decoder_receipt: Mapping[str, Any] | None = None
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
        _validate_training_core_v4(training_v1, training_v3)
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
        _validate_model_api_v4(model_api)
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
        initial_decoder_receipt = _initial_decoder_receipt_v4(
            model,
            partition,
            torch=torch,
            inherited_semantic_method=(
                parent_model_api.GeometryAnchoredDeformableBevLiftJointJepaV1.
                semantic_logits_from_latent
            ),
        )
        optimizer = training_v1.build_frozen_optimizer_v1(partition)
        if not any(
            name.startswith("predictor.swept_progress_head.")
            for name in partition.names["predictor"]
        ):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, training_diagnostics = (
            training_v3.run_fixed_training_v3(
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
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "semantic_decoder_initialization_seed": SEMANTIC_DECODER_INITIALIZATION_SEED,
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": "exact_n320_encoder_only",
                "predecessor_experiment_checkpoint_read": False,
                "auxiliary_objective": dict(AUXILIARY_OBJECTIVE),
                "initial_semantic_decoder": initial_decoder_receipt,
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
                "auxiliary_objective": dict(AUXILIARY_OBJECTIVE),
                "initial_semantic_decoder": initial_decoder_receipt,
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
                arm: scientific_metrics_v4(
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
        selection_semantic = semantic_metrics_v4(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v4(
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
        gate = evaluate_gate_v4(
            role_metrics["checkpoint_selection"], selection_semantic, comparisons
        )
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
                "status": gate["status"],
                "gate": gate,
                "caps": {
                    "updates": MAXIMUM_UPDATES,
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
                "scientific_change_from_v3": {
                    "only_change": "residual_local_semantic_decoder",
                    "initial_semantic_decoder": initial_decoder_receipt,
                    "auxiliary_objective_unchanged": dict(AUXILIARY_OBJECTIVE),
                    "model_changed": True,
                    "data_changed": False,
                    "optimizer_rules_changed": False,
                    "losses_changed": False,
                    "schedule_changed": False,
                    "evaluation_changed": False,
                },
                "training": {
                    "accounting": accounting,
                    "diagnostics": training_diagnostics,
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
                "matched_no_jepa": {
                    "status": "STAGED_ONLY_IF_FULL_ARM_PASSES",
                    "run_in_this_attempt": False,
                    "jepa_treatment_effect_claimed": False,
                    "must_use_identical_v4_decoder": True,
                },
                "authority": {
                    "development_only": True,
                    "g2_navigation_final_evaluation_opened": False,
                    "heldout_or_sealed_opened": False,
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
                _v1._write_json_v1(
                    output / "failure.json",
                    {
                        "schema": FAILURE_SCHEMA,
                        "status": "FAILED_NO_RETRY_OR_RESUME",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "auxiliary_objective": dict(AUXILIARY_OBJECTIVE),
                        "semantic_decoder": (
                            initial_decoder_receipt
                            if initial_decoder_receipt is not None
                            else semantic_decoder_architecture_receipt_v4()
                        ),
                        "predecessor_experiment_checkpoint_read": False,
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
    result = execute_v4(repository_root=args.repository_root)
    print(
        _v1._canonical_json_bytes(
            {
                "status": result["status"],
                "result": f"{OUTPUT_RELATIVE_PATH}/result.json",
            }
        ).decode("utf-8")
    )
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
