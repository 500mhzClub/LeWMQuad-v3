#!/usr/bin/env python3
"""Run the single capped RGB joint-JEPA V3 half occupied-safety probe.

V3 changes only the occupied-safety auxiliary coefficient from one to one
half.  Input, initialization, model, schedule, optimizer, controls, gates, and
hardware remain bound to the frozen predecessor contracts.  Rejected
experiment outputs are never named or opened.
"""
from __future__ import annotations

import argparse
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

_v2 = importlib.import_module(
    "scripts.execute_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux"
)
_v1 = _v2._v1

OUTPUT_RELATIVE_PATH = (
    ".generated/"
    "go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux/attempt_v1"
)
CHECKPOINT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux_"
    "checkpoint_v1"
)
TRACE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux_"
    "trace_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux_"
    "result_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux_"
    "failure_v1"
)

LABEL_ROOT_RELATIVE_PATH = _v2.LABEL_ROOT_RELATIVE_PATH
LABEL_MANIFEST_NAME = _v2.LABEL_MANIFEST_NAME
LABEL_MANIFEST_CONTENT_SHA256 = _v2.LABEL_MANIFEST_CONTENT_SHA256
LABEL_MANIFEST_FILE_SHA256 = _v2.LABEL_MANIFEST_FILE_SHA256
LABEL_MANIFEST_BYTE_COUNT = _v2.LABEL_MANIFEST_BYTE_COUNT
REQUIRED_GPU_NAME = _v2.REQUIRED_GPU_NAME
REQUIRED_GPU_MEMORY_BYTES = _v2.REQUIRED_GPU_MEMORY_BYTES
ACTION_ORDER = _v2.ACTION_ORDER
ROLE_FILES = _v2.ROLE_FILES
MICROBATCH_SIZE = _v2.MICROBATCH_SIZE
MICROBATCHES_PER_UPDATE = _v2.MICROBATCHES_PER_UPDATE
PRESENTATIONS_PER_UPDATE = _v2.PRESENTATIONS_PER_UPDATE
MAXIMUM_UPDATES = _v2.MAXIMUM_UPDATES
MAXIMUM_PRESENTATIONS = _v2.MAXIMUM_PRESENTATIONS
CONSTRUCTOR_INITIALIZATION_SEED = _v2.CONSTRUCTOR_INITIALIZATION_SEED
EXPERIMENT_SEED = _v2.EXPERIMENT_SEED
BOOTSTRAP_SEED = _v2.BOOTSTRAP_SEED
CONTROL_NAMES = _v2.CONTROL_NAMES
ALL_ARM_NAMES = _v2.ALL_ARM_NAMES
REGISTERED_FAMILIES = _v2.REGISTERED_FAMILIES
GATE_THRESHOLDS = _v2.GATE_THRESHOLDS
PROGRESS_SEGMENT_M = _v2.PROGRESS_SEGMENT_M

AUXILIARY_OBJECTIVE = {
    **_v2.AUXILIARY_OBJECTIVE,
    "coefficient": 0.5,
}

scientific_metrics_v3 = _v2.scientific_metrics_v2
semantic_metrics_v3 = _v2.semantic_metrics_v2
paired_control_comparison_v3 = _v2.paired_control_comparison_v2
evaluate_gate_v3 = _v2.evaluate_gate_v2


def auxiliary_objective_receipt_v3() -> dict[str, Any]:
    """Return the exact sole V3 scientific change as a detached record."""

    return dict(AUXILIARY_OBJECTIVE)


def _fresh_output_root_v3(repository_root: Path) -> Path:
    output = Path(repository_root) / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh half-occupied-safety attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    return output


def _validate_training_core_v3(training_v1: Any, training_v3: Any) -> None:
    """Fail closed unless the V3 core retains the frozen execution contract."""

    for name in (
        "ACTION_ORDER",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "PRESENTATIONS_PER_UPDATE",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
    ):
        if getattr(training_v3, name) != getattr(training_v1, name):
            raise PermissionError(f"V3 training core changed frozen {name}")
    if not callable(getattr(training_v3, "run_fixed_training_v3", None)):
        raise PermissionError("V3 fixed training entrypoint is absent")
    if (
        getattr(training_v3, "OCCUPIED_CLASS_INDEX", None) != 2
        or getattr(training_v3, "OCCUPIED_SAFETY_AUX_COEFFICIENT", None) != 0.5
        or getattr(training_v3, "OCCUPIED_SAFETY_AUX_NORMALIZATION", None)
        != math.log(2.0)
    ):
        raise PermissionError("V3 half occupied-safety auxiliary contract changed")


def execute_v3(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _v1._install_repository_import_roots_v1(repository_root)
    output = _fresh_output_root_v3(repository_root)
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
        _validate_training_core_v3(training_v1, training_v3)
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
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v1"
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
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV1(
            n320_state, masks
        ).to(context["device"])
        model.train()
        partition = training_v1.partition_parameters_v1(model)
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
                "experiment_seed": EXPERIMENT_SEED,
                "initialization_source": "exact_n320_encoder_only",
                "predecessor_experiment_checkpoint_read": False,
                "auxiliary_objective": auxiliary_objective_receipt_v3(),
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
                "auxiliary_objective": auxiliary_objective_receipt_v3(),
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
                arm: scientific_metrics_v3(
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
        selection_semantic = semantic_metrics_v3(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v3(
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
        gate = evaluate_gate_v3(
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
                "scientific_change_from_v2": {
                    "only_change": "occupied_safety_auxiliary_coefficient_1_to_0.5",
                    "auxiliary_objective": auxiliary_objective_receipt_v3(),
                    "model_changed": False,
                    "data_changed": False,
                    "optimizer_changed": False,
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
                },
                "authority": {
                    "development_only": True,
                    "g2_navigation_final_evaluation_opened": False,
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
                        "auxiliary_objective": auxiliary_objective_receipt_v3(),
                        "predecessor_experiment_checkpoint_read": False,
                        "authority": {
                            "development_only": True,
                            "g2_navigation_final_evaluation_opened": False,
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
    result = execute_v3(repository_root=args.repository_root)
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
