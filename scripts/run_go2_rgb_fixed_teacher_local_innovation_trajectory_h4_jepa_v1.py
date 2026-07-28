#!/usr/bin/env python3
"""Run one fixed-teacher local-innovation trajectory H4 JEPA probe."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1 as base,
)


core = base.core
BASE_WRAPPER_SOURCE = ROOT / (
    "scripts/run_go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1.py"
)
BASE_WRAPPER_SOURCE_SHA256 = (
    "ce133263ec17feae9c729072153567253bd1529c3b27d350cb0c290fc6552b4c"
)
BASE_WRAPPER_SOURCE_BYTES = 32_000
MODEL_MODULE = (
    "lewm.models.go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "f71639ee80bcadfa0e5c3238b55164deb57d196c2bdfa4bd0ced3bb5cba1bb71"
MODEL_SOURCE_BYTES = 17_248
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1/probe_v1"
)
SCHEMA = "lewm_go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1"
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_LOCAL_INNOVATION_TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_LOCAL_INNOVATION_TRAJECTORY_H4_JEPA_V1"
)
TRAJECTORY_MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1.py"
)
TRAJECTORY_MODEL_SOURCE_SHA256 = (
    "1d07f8f67ab7f3e93534454a2aaf2ff8626f984b5346a2717c1643a331eb6d8e"
)
TRAJECTORY_MODEL_SOURCE_BYTES = 15_967
_BASE_DECISION = base._trajectory_decision
_BASE_RUN = base._trajectory_run


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get(
        "LEWM_LOCAL_INNOVATION_TRAJECTORY_H4_WRAPPER_SHA256", ""
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_LOCAL_INNOVATION_TRAJECTORY_H4_WRAPPER_BYTES", ""
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external local-innovation trajectory wrapper binding is required"
        ) from error
    return {
        "local_innovation_trajectory_h4_wrapper": base._source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "trajectory_h4_wrapper_dependency": base._source_binding(
            BASE_WRAPPER_SOURCE,
            BASE_WRAPPER_SOURCE_SHA256,
            BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": base._source_binding(
            base.CORE_SOURCE,
            base.CORE_SOURCE_SHA256,
            base.CORE_SOURCE_BYTES,
        ),
        "local_innovation_trajectory_h4_model": base._source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": base._source_binding(
            TRAJECTORY_MODEL_SOURCE,
            TRAJECTORY_MODEL_SOURCE_SHA256,
            TRAJECTORY_MODEL_SOURCE_BYTES,
        ),
        "dense_h4_model_dependency": base._source_binding(
            base.DENSE_MODEL_SOURCE,
            base.DENSE_MODEL_SOURCE_SHA256,
            base.DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": base._source_binding(
            base.BASE_MODEL_SOURCE,
            base.BASE_MODEL_SOURCE_SHA256,
            base.BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": base._source_binding(
            base.ENCODER_SOURCE,
            base.ENCODER_SOURCE_SHA256,
            base.ENCODER_SOURCE_BYTES,
        ),
    }


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    base.MODEL_MODULE = MODEL_MODULE
    base.MODEL_SOURCE = MODEL_SOURCE
    base.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    base.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SCHEMA = SCHEMA
    base.PASS_DECISION = PASS_DECISION
    base.STOP_DECISION = STOP_DECISION
    base._configure_core(source_bindings)
    core.TARGET_DESCRIPTION = (
        "successive_fixed_N320_normalized_patch_innovations_stop_gradient_no_ema"
    )
    core.OBJECTIVE_DESCRIPTION = (
        "1*joint_plus_marginal_local_innovation_energy_score+"
        "1*three_frame_online_to_fixed_teacher_alignment+"
        "1*normalized_cyclic_wrong_action_margin_0.05+"
        "1*normalized_ordered_history_margin_0.03"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+dense_history+action_path+four_atom_trajectory_predictor_"
            "jointly_trained_in_one_backward"
        ),
        "support": "four_equal_mass_coherent_H1_to_H4_trajectory_atoms",
        "target": (
            "successive_fixed_teacher_normalized_patch_changes_"
            "e2_to_e3_e3_to_e4_e4_to_e5_e5_to_e6"
        ),
        "prediction": (
            "local_increments_recursively_integrated_and_normalized_from_e2"
        ),
        "innovation_metric": (
            "proper_50_50_joint_plus_mean_marginal_uniform_energy_score"
        ),
        "training_controls": {
            "wrong_action": "cyclic_plus_one_modulo_nine_margin_0.05",
            "history": (
                "minimum_of_reversed_e1_e0_e2_and_reset_e2_only_margin_0.03"
            ),
            "normalization": "detached_zero_innovation_target_energy",
        },
        "evaluation": "cumulative_absolute_trajectory_metrics_unchanged",
        "absent": [
            "learned_target_compressor",
            "absolute_future_training_loss",
            "best_of_k_loss",
            "learned_variance_scale_or_mixture_weights",
            "variance_covariance_or_whitening_loss",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_or_geometry_target",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    # Wrong-action, reversed-history, and reset-history paths all reuse the
    # current loaded batch inside the model's auxiliary objective.
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 3
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _local_innovation_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    result = dict(_BASE_DECISION(observations, updates_completed))
    prior_decision = result.get("decision")
    if prior_decision == base.PASS_DECISION:
        result["decision"] = PASS_DECISION
    elif prior_decision == base.STOP_DECISION:
        result["decision"] = STOP_DECISION
    else:
        raise core.ContractError("trajectory gate returned an unknown decision")
    result["authority"] = (
        "A pass establishes bounded development fixed-teacher local-innovation "
        "trajectory JEPA feasibility only; it grants no checkpoint access, "
        "navigation, held-out access, promotion, or deployment authority. A "
        "stop closes this exact local-innovation/counterfactual mechanism."
    )
    return result


def _local_innovation_run(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
    access = kwargs.get("access")
    if access is None:
        raise core.ContractError("local-innovation run requires access accounting")
    try:
        metrics, artifact, decision = _BASE_RUN(*args, **kwargs)
    finally:
        completed = int(access["optimizer_update_count"])
        expected = completed * core.BATCH_SIZE * 3
        if int(access["auxiliary_training_control_sequence_count"]) != expected:
            raise core.ContractError("local-innovation control accounting changed")
        if int(access["wrong_action_counterfactual_sequence_count"]) != 0:
            raise core.ContractError("built-in wrong-action control unexpectedly ran")
        access["auxiliary_training_control_sequence_count"] = (
            completed * core.BATCH_SIZE * 2
        )
        access["wrong_action_counterfactual_sequence_count"] = (
            completed * core.BATCH_SIZE
        )
    training = metrics.get("training_losses")
    if not isinstance(training, dict):
        raise core.ContractError("local-innovation training receipt is absent")
    for bucket_name in ("mean_over_completed_updates", "last_completed_update"):
        bucket = training.get(bucket_name)
        if not isinstance(bucket, dict):
            raise core.ContractError("local-innovation loss bucket is absent")
        diagnostic = bucket.pop("prediction", None)
        if diagnostic is None or "diagnostic_centroid_absolute_future_error" in bucket:
            raise core.ContractError("centroid diagnostic receipt changed")
        bucket["diagnostic_centroid_absolute_future_error"] = diagnostic
        for disabled_name in ("variance", "wrong_action_ranking"):
            value = bucket.pop(disabled_name, None)
            if value is None or value != 0.0:
                raise core.ContractError("disabled inherited training term changed")
    training["disabled_terms"] = [
        "absolute_future_centroid_prediction_in_objective",
        "inherited_variance_regularization",
        "built_in_centroid_wrong_action_ranking",
    ]
    training["receipt_field_semantics"] = {
        "diagnostic_centroid_absolute_future_error": (
            "measured_by_shared_runner_but_weight_zero"
        ),
        "future_teacher_local_innovation_energy_score": "objective_weight_one",
        "history_teacher_alignment": "objective_weight_one",
        "cyclic_wrong_action_score_ranking": "objective_weight_one",
        "history_counterfactual_score_ranking": "objective_weight_one",
    }
    adapted = dict(artifact)
    built_in = adapted.pop("wrong_action_training_contrast_enabled", None)
    if built_in is not False:
        raise core.ContractError("built-in wrong-action receipt changed")
    adapted["built_in_centroid_wrong_action_training_contrast_enabled"] = False
    adapted[
        "local_innovation_cyclic_wrong_action_training_contrast_enabled"
    ] = True
    adapted[
        "reversed_and_reset_history_training_contrasts_enabled"
    ] = True
    return metrics, adapted, decision


def _install_runtime_adapters() -> None:
    base._install_runtime_adapters()
    if core._decision not in (_BASE_DECISION, _local_innovation_decision):
        raise core.ContractError(
            "trajectory decision changed before local-innovation adapter"
        )
    core._decision = _local_innovation_decision
    if core._run not in (_BASE_RUN, _local_innovation_run):
        raise core.ContractError("trajectory run changed before receipt adapter")
    core._run = _local_innovation_run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(base.__file__).resolve() != BASE_WRAPPER_SOURCE:
        raise core.ContractError(
            "trajectory wrapper imported from an unexpected path"
        )
    source_bindings = _verify_source_closure()
    base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
