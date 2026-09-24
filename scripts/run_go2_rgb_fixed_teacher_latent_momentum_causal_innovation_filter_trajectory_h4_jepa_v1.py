#!/usr/bin/env python3
"""Run one latent-momentum causal innovation-filter trajectory-H4 JEPA probe.

This is a thin source-bound adapter over the factorized conditional-increment
runner.  It preserves the causal V2 schedules, evaluator, proper-score
objective, selection rule, all 32 gates, seed, work caps, and complete terminal
handler.  Only the model, mechanism receipts, output identity, and terminal
decision identity change.

Importing this module is source-only and opens no runtime input.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1
    as factorized,
)


v2 = factorized.v2
v1 = factorized.v1
core = factorized.core

FACTORIZED_RUNNER_SOURCE = ROOT / (
    "scripts/"
    "run_go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_"
    "h4_jepa_v1.py"
)
FACTORIZED_RUNNER_SOURCE_SHA256 = (
    "459b1a2837704e4c9534ef3229070461f507630e07cb77ad5bacb96aac3f0c56"
)
FACTORIZED_RUNNER_SOURCE_BYTES = 19_239
MODEL_MODULE = (
    "lewm.models."
    "go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_"
    "trajectory_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_"
    "trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "46fe5f22ff7b2416f9f6bdc4feb362d895b183ba6266b9db96ff98e4eaf9eb3e"
MODEL_SOURCE_BYTES = 16_417
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_"
    "trajectory_h4_jepa_v1/probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_"
    "trajectory_h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_LATENT_MOMENTUM_CAUSAL_INNOVATION_"
    "FILTER_TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_LATENT_MOMENTUM_CAUSAL_INNOVATION_"
    "FILTER_TRAJECTORY_H4_JEPA_V1"
)
INHERITED_OBJECTIVE_DESCRIPTION = (
    "0.5*proper_all_six_factual_local_innovation_energy_score+"
    "0.5*proper_open_loop_future_cumulative_trajectory_energy_score+"
    "1*three_frame_online_to_fixed_teacher_alignment;"
    "counterfactual_controls_evaluation_only"
)
OBJECTIVE_DESCRIPTION = (
    "0.5*proper_all_six_realized_local_innovation_energy_score_with_"
    "observed_online_z_t_and_future_recursive_q_t_baselines+"
    "0.5*proper_open_loop_future_cumulative_trajectory_energy_score+"
    "1*three_frame_online_to_fixed_teacher_alignment;"
    "counterfactual_controls_evaluation_only"
)

_FACTORIZED_DECISION = factorized._factorized_conditional_increment_decision
_FACTORIZED_RUN = factorized._factorized_conditional_increment_run
_FACTORIZED_TERMINAL_FAILURE = (
    factorized._factorized_conditional_increment_terminal_failure
)


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    """Bind this wrapper, the new model, and the full factorized closure."""

    wrapper_sha256 = os.environ.get(
        "LEWM_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_V1_"
        "WRAPPER_SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_V1_"
        "WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external latent-momentum innovation-filter wrapper binding is "
            "required"
        ) from error

    source_binding = v1.base._source_binding
    return {
        "latent_momentum_causal_innovation_filter_wrapper": source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "factorized_conditional_increment_wrapper_dependency": source_binding(
            FACTORIZED_RUNNER_SOURCE,
            FACTORIZED_RUNNER_SOURCE_SHA256,
            FACTORIZED_RUNNER_SOURCE_BYTES,
        ),
        "latent_momentum_causal_innovation_filter_model": source_binding(
            MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
        ),
        "factorized_conditional_increment_model_dependency": source_binding(
            factorized.MODEL_SOURCE,
            factorized.MODEL_SOURCE_SHA256,
            factorized.MODEL_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_wrapper_dependency": source_binding(
            factorized.V2_RUNNER_SOURCE,
            factorized.V2_RUNNER_SOURCE_SHA256,
            factorized.V2_RUNNER_SOURCE_BYTES,
        ),
        "factual_shared_transition_v1_runner_dependency": source_binding(
            v2.V1_RUNNER_SOURCE,
            v2.V1_RUNNER_SOURCE_SHA256,
            v2.V1_RUNNER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_adapter": source_binding(
            v2.V2_ADAPTER_SOURCE,
            v2.V2_ADAPTER_SOURCE_SHA256,
            v2.V2_ADAPTER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_builder": source_binding(
            v2.V2_BUILDER_SOURCE,
            v2.V2_BUILDER_SOURCE_SHA256,
            v2.V2_BUILDER_SOURCE_BYTES,
        ),
        "factual_shared_transition_model_dependency": source_binding(
            v1.MODEL_SOURCE,
            v1.MODEL_SOURCE_SHA256,
            v1.MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_wrapper_dependency": source_binding(
            v1.BASE_WRAPPER_SOURCE,
            v1.BASE_WRAPPER_SOURCE_SHA256,
            v1.BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": source_binding(
            v1.base.CORE_SOURCE,
            v1.base.CORE_SOURCE_SHA256,
            v1.base.CORE_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": source_binding(
            v1.TRAJECTORY_MODEL_SOURCE,
            v1.TRAJECTORY_MODEL_SOURCE_SHA256,
            v1.TRAJECTORY_MODEL_SOURCE_BYTES,
        ),
        "local_innovation_trajectory_h4_model_dependency": source_binding(
            v1.LOCAL_INNOVATION_MODEL_SOURCE,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_SHA256,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_BYTES,
        ),
        "dense_h4_model_dependency": source_binding(
            v1.base.DENSE_MODEL_SOURCE,
            v1.base.DENSE_MODEL_SOURCE_SHA256,
            v1.base.DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": source_binding(
            v1.base.BASE_MODEL_SOURCE,
            v1.base.BASE_MODEL_SOURCE_SHA256,
            v1.base.BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": source_binding(
            v1.base.ENCODER_SOURCE,
            v1.base.ENCODER_SOURCE_SHA256,
            v1.base.ENCODER_SOURCE_BYTES,
        ),
    }


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    """Install factorized V1, then replace only the reviewed mechanism."""

    factorized._configure_core(source_bindings)
    if core.OBJECTIVE_DESCRIPTION != INHERITED_OBJECTIVE_DESCRIPTION:
        raise core.ContractError("factorized objective description changed")
    inherited_schedule = core.ADDITIONAL_SCIENCE.get("schedule_integrity")
    if not isinstance(inherited_schedule, dict):
        raise core.ContractError("factorized schedule-integrity receipt changed")
    schedule_integrity = dict(inherited_schedule)
    inherited_reuse = schedule_integrity.get("reuse")
    if inherited_reuse != "exact_causal_v2_schedule_with_new_factorized_model":
        raise core.ContractError("factorized schedule reuse label changed")
    schedule_integrity["reuse"] = (
        "exact_causal_v2_schedule_with_new_latent_momentum_filter_model"
    )

    core.MODEL_MODULE = MODEL_MODULE
    core.MODEL_SOURCE = MODEL_SOURCE
    core.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    core.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.OBJECTIVE_DESCRIPTION = OBJECTIVE_DESCRIPTION
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+four_q_v_state_atoms+centered_modes+shared_observer+"
            "shared_state_context+complete_tower_centered_action_code+one_"
            "bias_free_zero_initialized_acceleration_head_jointly_trained_in_"
            "one_backward"
        ),
        "support": "four_equal_mass_coherent_six_transition_trajectory_atoms",
        "state": {
            "atoms": "exactly_four_equal_mass_q_content_v_momentum_pairs",
            "initialization": "q0_equals_online_z0_and_v0_equals_zero",
            "physical_interpretation": "none_feature_lattice_only",
            "future_belief": "packed_q2_v2_only",
        },
        "transition": {
            "shared_core": "one_exact_parameter_set_for_p0_through_p5",
            "prior": (
                "centered_action_acceleration_updates_tangent_momentum_then_"
                "radius_preserving_content"
            ),
            "observed_steps": (
                "emit_and_score_prior_before_post_prior_innovation_"
                "assimilation_on_p0_and_p1"
            ),
            "future_steps": (
                "same_prior_open_loop_over_p2_through_p5_from_packed_q2_v2"
            ),
            "readout": "prior_q_content_only",
            "target_leakage": "none",
        },
        "observer": {
            "calls": "exactly_twice_after_scored_observed_priors",
            "input": "prior_q_v_plus_new_online_z_minus_prior_q_innovation",
            "content_gain": "one_plus_tanh_of_zero_initialized_head",
            "momentum_update": "residual_through_zero_initialized_head",
            "reinitializer": False,
            "parallel_factual_route": False,
        },
        "factorization": {
            "categorical_action_code": (
                "c_a=A(E[a])-uniform_mean_over_complete_action_tower"
            ),
            "selected_action_interaction": "B(q_t,v_t,centered_mode)*c_a",
            "shared_acceleration_projection": (
                "one_bias_free_zero_initialized_W0"
            ),
            "uniform_action_mean_acceleration": "exactly_zero",
            "hold_special_case": False,
        },
        "proper_score": {
            "all_six_realized_local_innovation_weight": 0.5,
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "observed_local_baseline": "registered_online_factual_z_t",
            "future_local_baseline": "recursively_realized_q_t",
            "each_domain": "50_50_joint_plus_mean_marginal_uniform_energy_score",
            "prediction_normalization": "none",
        },
        "training_losses": [
            "half_all_six_realized_local_innovation_energy_score",
            "half_open_loop_future_cumulative_trajectory_energy_score",
            "history_teacher_alignment",
        ],
        "training_controls": {
            "enabled": False,
            "wrong_action": False,
            "all_hold": False,
            "reversed_history": False,
            "reset_history": False,
        },
        "evaluation_only_controls": [
            "cyclic_wrong_action",
            "all_hold_action",
            "persistence",
            "reordered_or_reset_history",
            "collapsed_spherical_centroid",
        ],
        "evaluation": "exact_V2_evaluator_selection_and_all_32_gates",
        "absent": [
            "raw_z2_or_explicit_incoming_increment_predictor_bypass",
            "factual_carrier_replacement_or_anchor_slot",
            "dense_history_or_direct_horizon_query",
            "action_cross_entropy_inverse_or_control_ranking_loss",
            "per_action_operator_bank_or_action_query_successor",
            "learned_target_compressor_target_ema_whitening_or_covariance_loss",
            "reconstruction_navigation_pose_depth_flow_bev_warp_or_geometry",
            "separate_predictor_training_or_predictor_checkpoint",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
        "schedule_integrity": schedule_integrity,
    }
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _latent_momentum_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    """Reuse all 32 inherited gates and replace only terminal identity."""

    result = dict(_FACTORIZED_DECISION(observations, updates_completed))
    failed_gates = result.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise core.ContractError("factorized decision failure list changed")
    expected = (
        factorized.PASS_DECISION
        if not failed_gates
        else factorized.STOP_DECISION
    )
    if result.get("decision") != expected:
        raise core.ContractError("factorized decision identity disagrees with gates")
    result["decision"] = PASS_DECISION if not failed_gates else STOP_DECISION
    result["authority"] = (
        "A pass establishes bounded perception/world-model development evidence "
        "for this latent-momentum causal innovation filter on the frozen V2 "
        "requested-action schedule only; it grants no checkpoint access, "
        "navigation, G2, held-out or sealed access, promotion, production, or "
        "deployment authority. A stop closes this exact one-shot mechanism "
        "without retry, resume, or replacement."
    )
    return result


def _latent_momentum_run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Reuse factorized execution and replace predecessor artifact claims."""

    metrics, artifact, decision = _FACTORIZED_RUN(*args, **kwargs)
    adapted_metrics = dict(metrics)
    inherited_training = metrics.get("training_losses")
    if not isinstance(inherited_training, dict):
        raise core.ContractError("inherited training-loss receipt changed")
    training = dict(inherited_training)
    if training.get("objective") != OBJECTIVE_DESCRIPTION:
        raise core.ContractError("latent-momentum objective receipt changed")
    inherited_loss_name = (
        "half_all_six_factual_local_innovation_energy_score"
    )
    realized_loss_name = (
        "half_all_six_realized_local_innovation_energy_score"
    )
    for bucket_name in (
        "mean_over_completed_updates",
        "last_completed_update",
    ):
        inherited_bucket = training.get(bucket_name)
        if not isinstance(inherited_bucket, dict):
            raise core.ContractError("inherited training-loss bucket changed")
        bucket = dict(inherited_bucket)
        if inherited_loss_name not in bucket:
            raise core.ContractError("inherited local-loss field changed")
        bucket[realized_loss_name] = bucket.pop(inherited_loss_name)
        training[bucket_name] = bucket
    inherited_semantics = training.get("receipt_field_semantics")
    if not isinstance(inherited_semantics, dict):
        raise core.ContractError("inherited loss semantics changed")
    semantics = dict(inherited_semantics)
    inherited_local_semantics = semantics.pop(inherited_loss_name, None)
    if inherited_local_semantics != "objective_term_already_weighted_one_half":
        raise core.ContractError("inherited local-loss semantics changed")
    semantics[realized_loss_name] = (
        "objective_term_already_weighted_one_half_with_observed_z_t_and_"
        "future_recursive_q_t_baselines"
    )
    training["receipt_field_semantics"] = semantics
    adapted_metrics["training_losses"] = training

    adapted = dict(artifact)
    expected_true = (
        "fresh_factorized_belief_increment_action_and_shared_projection_"
        "initialization",
        "factorized_conditional_increment_mechanism_enabled",
        "factual_shared_transition_objective_enabled",
    )
    for name in expected_true:
        if adapted.pop(name, None) is not True:
            raise core.ContractError(f"inherited artifact field changed: {name}")
    if not isinstance(
        adapted.pop("factorized_conditional_increment_contract", None), dict
    ):
        raise core.ContractError("factorized mechanism artifact changed")
    inherited_weights = adapted.pop(
        "factual_shared_transition_score_weights", None
    )
    if inherited_weights != {
        "all_six_factual_local_innovation": 0.5,
        "open_loop_future_cumulative_trajectory": 0.5,
    }:
        raise core.ContractError("inherited proper-score artifact changed")

    adapted[
        "fresh_latent_momentum_modes_observer_context_action_and_"
        "acceleration_initialization"
    ] = True
    adapted["latent_momentum_causal_innovation_filter_enabled"] = True
    adapted["latent_momentum_causal_innovation_filter_score_weights"] = {
        "all_six_realized_local_innovations": 0.5,
        "open_loop_future_cumulative_trajectory": 0.5,
    }
    adapted["latent_momentum_causal_innovation_filter_contract"] = {
        "state": "four_equal_mass_q_content_v_momentum_atoms",
        "shared_prior_calls": 6,
        "post_prior_observer_calls": 2,
        "observed_local_baseline": "registered_online_factual_z_t",
        "future_local_baseline": "recursive_q_t",
        "action_code": "uniformly_centered_after_complete_action_tower",
        "acceleration_projection_bias": False,
        "acceleration_projection_zero_initialized": True,
        "future_raw_z_or_explicit_increment_bypass": False,
        "parallel_factual_or_anchor_route": False,
    }
    return adapted_metrics, adapted, decision


def _install_runtime_adapters() -> None:
    """Install factorized runtime, retaining evaluator and terminal handler."""

    if core._decision is _latent_momentum_decision:
        if core._evaluate is not v1._factual_shared_transition_evaluate:
            raise core.ContractError("latent-momentum evaluator identity changed")
        if core._run is not _latent_momentum_run:
            raise core.ContractError("latent-momentum run handler identity changed")
        if core._terminal_failure is not _FACTORIZED_TERMINAL_FAILURE:
            raise core.ContractError("factorized terminal handler identity changed")
        return

    factorized._install_runtime_adapters()
    if core._evaluate is not v1._factual_shared_transition_evaluate:
        raise core.ContractError("factorized evaluator was not preserved")
    if core._run is not _FACTORIZED_RUN:
        raise core.ContractError("factorized run adapter was not preserved")
    if core._decision is not _FACTORIZED_DECISION:
        raise core.ContractError("factorized decision adapter was not preserved")
    if core._terminal_failure is not _FACTORIZED_TERMINAL_FAILURE:
        raise core.ContractError("factorized terminal handler was not preserved")
    core._run = _latent_momentum_run
    core._decision = _latent_momentum_decision


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != v1.base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(factorized.__file__).resolve() != FACTORIZED_RUNNER_SOURCE:
        raise core.ContractError(
            "factorized runner imported from an unexpected path"
        )
    source_bindings = _verify_source_closure()
    v1.base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
