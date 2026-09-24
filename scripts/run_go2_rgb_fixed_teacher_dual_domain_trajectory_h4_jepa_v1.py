#!/usr/bin/env python3
"""Run one fixed-teacher dual-domain trajectory H4 JEPA probe."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1 as parent,
)


base = parent.base
core = parent.core
PARENT_WRAPPER_SOURCE = ROOT / (
    "scripts/run_go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1.py"
)
PARENT_WRAPPER_SOURCE_SHA256 = (
    "8f2e84ec28fb67440fa624720a7e15fab9aaf88537413872a2f723cf43301a66"
)
PARENT_WRAPPER_SOURCE_BYTES = 11_453
PARENT_MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1.py"
)
PARENT_MODEL_SOURCE_SHA256 = (
    "f71639ee80bcadfa0e5c3238b55164deb57d196c2bdfa4bd0ced3bb5cba1bb71"
)
PARENT_MODEL_SOURCE_BYTES = 17_248
MODEL_MODULE = "lewm.models.go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1"
MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "001bc8507dfe7c4e9f815e67e09f96c898628b9aa7bffa3ad1e0f8a1777b971f"
MODEL_SOURCE_BYTES = 9_608
OUTPUT_ROOT = ROOT / (
    ".generated/go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1/probe_v1"
)
SCHEMA = "lewm_go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1"
PASS_DECISION = "PASS_MAIN_POOL_RGB_FIXED_TEACHER_DUAL_DOMAIN_TRAJECTORY_H4_JEPA_V1"
STOP_DECISION = "STOP_MAIN_POOL_RGB_FIXED_TEACHER_DUAL_DOMAIN_TRAJECTORY_H4_JEPA_V1"
_PARENT_DECISION = parent._local_innovation_decision
_PARENT_RUN = parent._local_innovation_run


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get(
        "LEWM_DUAL_DOMAIN_TRAJECTORY_H4_WRAPPER_SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_DUAL_DOMAIN_TRAJECTORY_H4_WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external dual-domain trajectory wrapper binding is required"
        ) from error
    return {
        "dual_domain_trajectory_h4_wrapper": base._source_binding(
            Path(__file__).resolve(),
            wrapper_sha256,
            wrapper_bytes,
        ),
        "local_innovation_trajectory_h4_wrapper_dependency": base._source_binding(
            PARENT_WRAPPER_SOURCE,
            PARENT_WRAPPER_SOURCE_SHA256,
            PARENT_WRAPPER_SOURCE_BYTES,
        ),
        "trajectory_h4_wrapper_dependency": base._source_binding(
            parent.BASE_WRAPPER_SOURCE,
            parent.BASE_WRAPPER_SOURCE_SHA256,
            parent.BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": base._source_binding(
            base.CORE_SOURCE,
            base.CORE_SOURCE_SHA256,
            base.CORE_SOURCE_BYTES,
        ),
        "dual_domain_trajectory_h4_model": base._source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "local_innovation_trajectory_h4_model_dependency": base._source_binding(
            PARENT_MODEL_SOURCE,
            PARENT_MODEL_SOURCE_SHA256,
            PARENT_MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": base._source_binding(
            parent.TRAJECTORY_MODEL_SOURCE,
            parent.TRAJECTORY_MODEL_SOURCE_SHA256,
            parent.TRAJECTORY_MODEL_SOURCE_BYTES,
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
    parent.MODEL_MODULE = MODEL_MODULE
    parent.MODEL_SOURCE = MODEL_SOURCE
    parent.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    parent.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    parent.OUTPUT_ROOT = OUTPUT_ROOT
    parent.SCHEMA = SCHEMA
    parent.PASS_DECISION = PASS_DECISION
    parent.STOP_DECISION = STOP_DECISION
    parent._configure_core(source_bindings)
    core.TARGET_DESCRIPTION = (
        "fixed_N320_local_innovations_and_integrated_future_trajectory_"
        "stop_gradient_no_ema"
    )
    core.OBJECTIVE_DESCRIPTION = (
        "0.5*proper_local_innovation_energy_score+"
        "0.5*proper_cumulative_trajectory_energy_score+"
        "1*three_frame_online_to_fixed_teacher_alignment+"
        "1*normalized_dual_domain_cyclic_wrong_action_margin_0.05+"
        "1*normalized_dual_domain_ordered_history_margin_0.03"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+dense_history+action_path+four_atom_trajectory_predictor_"
            "jointly_trained_in_one_backward"
        ),
        "support": "four_equal_mass_coherent_H1_to_H4_trajectory_atoms",
        "prediction": (
            "local_increments_recursively_integrated_and_normalized_from_e2"
        ),
        "targets": {
            "local": (
                "successive_fixed_teacher_normalized_patch_changes_"
                "e2_to_e3_e3_to_e4_e4_to_e5_e5_to_e6"
            ),
            "cumulative": "fixed_teacher_normalized_future_latents_e3_to_e6",
        },
        "proper_score": {
            "local_weight": 0.5,
            "cumulative_weight": 0.5,
            "each_domain": (
                "50_50_joint_plus_mean_marginal_uniform_energy_score"
            ),
            "prediction_normalization": "none",
        },
        "training_controls": {
            "score": "same_50_50_local_plus_cumulative_score_as_prediction",
            "wrong_action": "cyclic_plus_one_modulo_nine_margin_0.05",
            "history": (
                "minimum_of_complete_mixed_reversed_and_reset_scores_margin_0.03"
            ),
            "normalization": (
                "detached_target_only_mixed_persistence_energy_clamped_1e-6"
            ),
        },
        "inherited_receipt_semantics": {
            "wrong_action_contrast_training_enabled_false": (
                "disabled_built_in_centroid_contrast_only"
            ),
            "dual_domain_cyclic_and_history_training_controls": "enabled",
        },
        "evaluation": (
            "cumulative_absolute_trajectory_metrics_with_hold_family_"
            "breadth_and_floor_gates"
        ),
        "absent": [
            "learned_target_compressor",
            "absolute_future_centroid_squared_error_training_loss",
            "best_of_k_loss",
            "learned_variance_scale_or_mixture_weights",
            "variance_covariance_or_whitening_loss",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_or_geometry_target",
            "all_hold_training_control",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 3
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _dual_domain_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    result = dict(_PARENT_DECISION(observations, updates_completed))
    gates = dict(result["gates"])
    diagnostics = dict(result["diagnostics"])
    selected_update = diagnostics.get("selected_update")
    selected = next(
        (item for item in observations if item["update"] == selected_update),
        None,
    )
    if selected is None:
        hold_positive = 0
        minimum_hold = None
        gates["hold_positive_in_six_families"] = False
        gates["no_family_hold_gap_below_minus_point02"] = False
    else:
        family_holds = [
            selected["family"][family]["hold_gap"][3]
            for family in core.FAMILIES
        ]
        hold_positive = sum(value > 0.0 for value in family_holds)
        minimum_hold = min(family_holds)
        gates["hold_positive_in_six_families"] = hold_positive >= 6
        gates["no_family_hold_gap_below_minus_point02"] = minimum_hold >= -0.02
    diagnostics["hold_positive_family_count"] = hold_positive
    diagnostics["minimum_family_h4_hold_gap"] = minimum_hold
    failed_gates = sorted(name for name, value in gates.items() if not value)
    return {
        "decision": PASS_DECISION if not failed_gates else STOP_DECISION,
        "gates": gates,
        "failed_gates": failed_gates,
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded development fixed-teacher dual-domain "
            "trajectory JEPA feasibility only; it grants no checkpoint access, "
            "navigation, held-out access, scale promotion, or deployment "
            "authority. A stop closes this exact 50/50 dual-domain mechanism."
        ),
    }


def _dual_domain_run(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
    metrics, artifact, decision = _PARENT_RUN(*args, **kwargs)
    training = metrics.get("training_losses")
    if not isinstance(training, dict):
        raise core.ContractError("dual-domain training receipt is absent")
    expected_bucket = {
        "total",
        "diagnostic_centroid_absolute_future_error",
        "history_teacher_alignment",
        "half_future_teacher_local_innovation_energy_score",
        "half_future_teacher_cumulative_trajectory_energy_score",
        "dual_domain_cyclic_wrong_action_score_ranking",
        "dual_domain_history_counterfactual_score_ranking",
    }
    for bucket_name in ("mean_over_completed_updates", "last_completed_update"):
        bucket = training.get(bucket_name)
        if not isinstance(bucket, dict) or set(bucket) != expected_bucket:
            raise core.ContractError("dual-domain loss receipt fields changed")
    training["receipt_field_semantics"] = {
        "diagnostic_centroid_absolute_future_error": (
            "measured_by_shared_runner_but_weight_zero"
        ),
        "history_teacher_alignment": "objective_weight_one",
        "half_future_teacher_local_innovation_energy_score": (
            "objective_term_already_weighted_one_half"
        ),
        "half_future_teacher_cumulative_trajectory_energy_score": (
            "objective_term_already_weighted_one_half"
        ),
        "dual_domain_cyclic_wrong_action_score_ranking": "objective_weight_one",
        "dual_domain_history_counterfactual_score_ranking": "objective_weight_one",
    }
    adapted = dict(artifact)
    local_action = adapted.pop(
        "local_innovation_cyclic_wrong_action_training_contrast_enabled",
        None,
    )
    local_history = adapted.pop(
        "reversed_and_reset_history_training_contrasts_enabled",
        None,
    )
    if local_action is not True or local_history is not True:
        raise core.ContractError("parent counterfactual artifact receipt changed")
    adapted["dual_domain_prediction_score_enabled"] = True
    adapted["dual_domain_score_weights"] = {
        "local_innovation": 0.5,
        "cumulative_trajectory": 0.5,
    }
    adapted["dual_domain_cyclic_wrong_action_training_contrast_enabled"] = True
    adapted["dual_domain_reversed_and_reset_history_training_contrasts_enabled"] = (
        True
    )
    adapted["all_hold_training_contrast_enabled"] = False
    return metrics, adapted, decision


def _install_runtime_adapters() -> None:
    parent._install_runtime_adapters()
    if core._decision not in (_PARENT_DECISION, _dual_domain_decision):
        raise core.ContractError(
            "local-innovation decision changed before dual-domain adapter"
        )
    core._decision = _dual_domain_decision
    if core._run not in (_PARENT_RUN, _dual_domain_run):
        raise core.ContractError("local-innovation run changed before dual adapter")
    core._run = _dual_domain_run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(parent.__file__).resolve() != PARENT_WRAPPER_SOURCE:
        raise core.ContractError(
            "local-innovation wrapper imported from an unexpected path"
        )
    if Path(base.__file__).resolve() != parent.BASE_WRAPPER_SOURCE:
        raise core.ContractError("trajectory wrapper imported from an unexpected path")
    source_bindings = _verify_source_closure()
    base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
