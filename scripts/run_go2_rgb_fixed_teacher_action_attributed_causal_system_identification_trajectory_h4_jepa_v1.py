#!/usr/bin/env python3
"""Run one action-attributed causal system-identification H4 JEPA probe.

This is a thin source-bound adapter over the frozen latent-momentum causal
innovation-filter runner.  It preserves the causal V2 schedules, evaluator,
proper-score objective, selection rule, all 32 gates, seed, work caps, and
complete terminal handler.  Only the model, mechanism receipts, output
identity, and terminal decision identity change.

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
    run_go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_trajectory_h4_jepa_v1
    as latent_momentum,
)


factorized = latent_momentum.factorized
v2 = latent_momentum.v2
v1 = latent_momentum.v1
core = latent_momentum.core

LATENT_MOMENTUM_RUNNER_SOURCE = ROOT / (
    "scripts/"
    "run_go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_"
    "trajectory_h4_jepa_v1.py"
)
LATENT_MOMENTUM_RUNNER_SOURCE_SHA256 = (
    "398bd99a9fdb9f9e6d7b6ef54089f96440fadf7544d0eedabf68344e0fec4e59"
)
LATENT_MOMENTUM_RUNNER_SOURCE_BYTES = 19_659
MODEL_MODULE = (
    "lewm.models."
    "go2_rgb_fixed_teacher_action_attributed_causal_system_identification_"
    "trajectory_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_action_attributed_causal_system_identification_"
    "trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = (
    "8edff571ca262fcf3b1e505017beb3f73eee027fc2ae195caaa127bbee6b6f02"
)
MODEL_SOURCE_BYTES = 21_854
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_action_attributed_causal_system_identification_"
    "trajectory_h4_jepa_v1/probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_action_attributed_causal_system_"
    "identification_trajectory_h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_"
    "IDENTIFICATION_TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_"
    "IDENTIFICATION_TRAJECTORY_H4_JEPA_V1"
)
INHERITED_OBJECTIVE_DESCRIPTION = latent_momentum.OBJECTIVE_DESCRIPTION
OBJECTIVE_DESCRIPTION = INHERITED_OBJECTIVE_DESCRIPTION

_LATENT_MOMENTUM_DECISION = latent_momentum._latent_momentum_decision
_LATENT_MOMENTUM_RUN = latent_momentum._latent_momentum_run
_LATENT_MOMENTUM_TERMINAL_FAILURE = (
    latent_momentum._FACTORIZED_TERMINAL_FAILURE
)


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    """Bind this wrapper, the new model, and the full inherited closure."""

    wrapper_sha256 = os.environ.get(
        "LEWM_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_IDENTIFICATION_TRAJECTORY_H4_"
        "V1_WRAPPER_SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_ACTION_ATTRIBUTED_CAUSAL_SYSTEM_IDENTIFICATION_TRAJECTORY_H4_"
        "V1_WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external action-attributed system-identification wrapper binding "
            "is required"
        ) from error

    source_binding = v1.base._source_binding
    return {
        "action_attributed_causal_system_identification_wrapper": (
            source_binding(
                Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
            )
        ),
        "latent_momentum_causal_innovation_filter_wrapper_dependency": (
            source_binding(
                LATENT_MOMENTUM_RUNNER_SOURCE,
                LATENT_MOMENTUM_RUNNER_SOURCE_SHA256,
                LATENT_MOMENTUM_RUNNER_SOURCE_BYTES,
            )
        ),
        "action_attributed_causal_system_identification_model": (
            source_binding(
                MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
            )
        ),
        "latent_momentum_causal_innovation_filter_model_dependency": (
            source_binding(
                latent_momentum.MODEL_SOURCE,
                latent_momentum.MODEL_SOURCE_SHA256,
                latent_momentum.MODEL_SOURCE_BYTES,
            )
        ),
        "factorized_conditional_increment_wrapper_dependency": source_binding(
            latent_momentum.FACTORIZED_RUNNER_SOURCE,
            latent_momentum.FACTORIZED_RUNNER_SOURCE_SHA256,
            latent_momentum.FACTORIZED_RUNNER_SOURCE_BYTES,
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
    """Install latent momentum, then replace only the reviewed mechanism."""

    latent_momentum._configure_core(source_bindings)
    if core.OBJECTIVE_DESCRIPTION != INHERITED_OBJECTIVE_DESCRIPTION:
        raise core.ContractError("latent-momentum objective description changed")
    inherited_schedule = core.ADDITIONAL_SCIENCE.get("schedule_integrity")
    if not isinstance(inherited_schedule, dict):
        raise core.ContractError(
            "latent-momentum schedule-integrity receipt changed"
        )
    schedule_integrity = dict(inherited_schedule)
    inherited_reuse = schedule_integrity.get("reuse")
    expected_reuse = (
        "exact_causal_v2_schedule_with_new_latent_momentum_filter_model"
    )
    if inherited_reuse != expected_reuse:
        raise core.ContractError("latent-momentum schedule reuse label changed")
    schedule_integrity["reuse"] = (
        "exact_causal_v2_schedule_with_new_action_attributed_causal_system_"
        "identification_model"
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
            "encoder+four_q_M_state_atoms+centered_modes+shared_action_free_"
            "spatial_context+complete_tower_centered_action_code+shared_"
            "response_key_memory_projections+one_bias_free_zero_initialized_"
            "increment_head_jointly_trained_in_one_backward"
        ),
        "support": "four_equal_mass_coherent_six_transition_trajectory_atoms",
        "state": {
            "atoms": (
                "exactly_four_equal_mass_q_content_M_16x16_nonspatial_"
                "action_response_matrix_pairs"
            ),
            "initialization": "q0_equals_online_z0_and_M0_equals_zero",
            "memory_matrix_shape": [16, 16],
            "maximum_rank_after_two_writes": 2,
            "physical_interpretation": "none_feature_statistic_only",
            "future_belief": "packed_q2_M2_only",
            "serialized_padding": "fixed_and_exact_zero",
        },
        "transition": {
            "shared_core": "one_exact_parameter_set_for_p0_through_p5",
            "prior": (
                "action_free_B_times_one_plus_tanh_P_M_of_row_major_M_times_"
                "current_centered_action_then_one_shared_bias_free_W0"
            ),
            "observed_steps": (
                "emit_and_score_prior_then_fixed_outer_product_write_then_"
                "factual_q_assimilation_on_p0_and_p1"
            ),
            "future_steps": (
                "same_prior_open_loop_over_p2_through_p5_with_M2_bitwise_fixed"
            ),
            "readout": "prior_q_content_only",
            "target_leakage": "none",
        },
        "writer": {
            "calls": "exactly_twice_after_scored_observed_priors",
            "input": (
                "new_online_z_minus_prior_q_error_and_corresponding_known_"
                "requested_past_action_only"
            ),
            "response": (
                "tanh_of_bias_free_P_r_after_non_affine_token_LN_and_fixed_"
                "token_mean"
            ),
            "action_key": (
                "centered_tanh_of_bias_free_P_c_on_complete_tower_centered_"
                "requested_action_code"
            ),
            "update": (
                "M_next_equals_M_plus_one_over_sqrt16_times_outer_response_key"
            ),
            "future_calls": 0,
            "learned_gain_decay_gate_or_recurrent_updater": False,
        },
        "memory_read": {
            "sole_consumer": "bias_free_P_M_then_one_plus_tanh",
            "current_action_requirement": (
                "M_modulation_reaches_W0_only_multiplicatively_with_current_"
                "centered_action"
            ),
            "M_zero_path": "ordinary_centered_action_interaction_with_mu_one",
            "uniform_action_mean_pre_renormalization_delta": "exactly_zero",
            "direct_or_action_independent_route": False,
        },
        "factorization": {
            "categorical_action_code": (
                "c_a=A(E[a])-uniform_mean_over_complete_action_tower"
            ),
            "attribution_key": (
                "k_a=centered_tanh_bias_free_projection_of_c_a"
            ),
            "selected_action_interaction": "B(q_t,mode)*mu(M_t)*c_a",
            "shared_increment_projection": (
                "one_bias_free_zero_initialized_W0"
            ),
            "uniform_action_mean_pre_renormalization_delta": "exactly_zero",
            "hold_special_case": False,
        },
        "proper_score": {
            "all_six_realized_local_innovation_weight": 0.5,
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "observed_local_baseline": "registered_online_factual_z_t",
            "future_local_baseline": "recursively_realized_q_t",
            "each_domain": (
                "50_50_joint_plus_mean_marginal_uniform_energy_score"
            ),
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
            "learned_memory_gain_decay_gate_or_generic_recurrent_updater",
            "token_memory_nine_action_slots_or_per_action_operator_bank",
            "raw_z2_or_explicit_incoming_increment_predictor_bypass",
            "momentum_velocity_factual_carrier_or_anchor_slot",
            "dense_history_direct_horizon_query_or_attention_memory",
            "action_cross_entropy_inverse_or_control_ranking_loss",
            "correspondence_transport_cost_volume_flow_warp_or_retrieval",
            "learned_target_compressor_target_ema_whitening_or_covariance_loss",
            "reconstruction_navigation_pose_depth_bev_or_geometry",
            "separate_system_identifier_or_predictor_training_checkpoint",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
        "schedule_integrity": schedule_integrity,
    }
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _action_attributed_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    """Reuse all 32 inherited gates and replace only terminal identity."""

    result = dict(_LATENT_MOMENTUM_DECISION(observations, updates_completed))
    failed_gates = result.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise core.ContractError("latent-momentum decision failure list changed")
    expected = (
        latent_momentum.PASS_DECISION
        if not failed_gates
        else latent_momentum.STOP_DECISION
    )
    if result.get("decision") != expected:
        raise core.ContractError(
            "latent-momentum decision identity disagrees with gates"
        )
    result["decision"] = PASS_DECISION if not failed_gates else STOP_DECISION
    result["authority"] = (
        "A pass establishes bounded perception/world-model development evidence "
        "for this action-attributed causal system-identification statistic on "
        "the frozen V2 requested-action schedule only; it grants no checkpoint "
        "access, navigation, G2, held-out or sealed access, promotion, "
        "production, or deployment authority. A stop closes this exact "
        "one-shot mechanism without retry, resume, or replacement."
    )
    return result


def _action_attributed_run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Reuse latent-momentum execution and replace mechanism artifact claims."""

    metrics, artifact, decision = _LATENT_MOMENTUM_RUN(*args, **kwargs)
    adapted_metrics = dict(metrics)
    inherited_training = metrics.get("training_losses")
    if not isinstance(inherited_training, dict):
        raise core.ContractError("inherited training-loss receipt changed")
    if inherited_training.get("objective") != OBJECTIVE_DESCRIPTION:
        raise core.ContractError(
            "action-attributed objective receipt changed"
        )

    adapted = dict(artifact)
    expected_true = (
        "fresh_latent_momentum_modes_observer_context_action_and_"
        "acceleration_initialization",
        "latent_momentum_causal_innovation_filter_enabled",
    )
    for name in expected_true:
        if adapted.pop(name, None) is not True:
            raise core.ContractError(f"inherited artifact field changed: {name}")
    inherited_weights = adapted.pop(
        "latent_momentum_causal_innovation_filter_score_weights", None
    )
    if inherited_weights != {
        "all_six_realized_local_innovations": 0.5,
        "open_loop_future_cumulative_trajectory": 0.5,
    }:
        raise core.ContractError("inherited proper-score artifact changed")
    inherited_contract = adapted.pop(
        "latent_momentum_causal_innovation_filter_contract", None
    )
    if inherited_contract != {
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
    }:
        raise core.ContractError("latent-momentum mechanism artifact changed")

    adapted[
        "fresh_action_attributed_modes_response_key_memory_context_action_and_"
        "increment_initialization"
    ] = True
    adapted["action_attributed_causal_system_identification_enabled"] = True
    adapted["action_attributed_causal_system_identification_score_weights"] = {
        "all_six_realized_local_innovations": 0.5,
        "open_loop_future_cumulative_trajectory": 0.5,
    }
    adapted["action_attributed_causal_system_identification_contract"] = {
        "state": (
            "four_equal_mass_q_content_M_16x16_nonspatial_response_matrix_atoms"
        ),
        "memory_matrix_shape": [16, 16],
        "maximum_rank_after_observed_writes": 2,
        "shared_prior_calls": 6,
        "post_prior_outer_product_writer_calls": 2,
        "future_writer_calls": 0,
        "writer_rule": (
            "M_next=M+(1/sqrt(16))*outer(prior_error_response,"
            "centered_known_past_action_key)"
        ),
        "observed_local_baseline": "registered_online_factual_z_t",
        "future_local_baseline": "recursive_q_t",
        "action_code": "uniformly_centered_after_complete_action_tower",
        "memory_read": (
            "one_plus_tanh_bias_free_projection_multiplying_current_centered_"
            "action_interaction_only"
        ),
        "increment_projection_bias": False,
        "increment_projection_zero_initialized": True,
        "learned_writer_gain_decay_gate_or_recurrent_updater": False,
        "future_memory_bitwise_fixed": True,
        "future_raw_z_or_explicit_increment_bypass": False,
        "parallel_factual_momentum_or_anchor_route": False,
    }
    return adapted_metrics, adapted, decision


def _install_runtime_adapters() -> None:
    """Install latent-momentum runtime, preserving evaluator and terminal."""

    if core._decision is _action_attributed_decision:
        if core._evaluate is not v1._factual_shared_transition_evaluate:
            raise core.ContractError(
                "action-attributed evaluator identity changed"
            )
        if core._run is not _action_attributed_run:
            raise core.ContractError(
                "action-attributed run handler identity changed"
            )
        if core._terminal_failure is not _LATENT_MOMENTUM_TERMINAL_FAILURE:
            raise core.ContractError(
                "latent-momentum terminal handler identity changed"
            )
        return

    latent_momentum._install_runtime_adapters()
    if core._evaluate is not v1._factual_shared_transition_evaluate:
        raise core.ContractError("latent-momentum evaluator was not preserved")
    if core._run is not _LATENT_MOMENTUM_RUN:
        raise core.ContractError("latent-momentum run adapter was not preserved")
    if core._decision is not _LATENT_MOMENTUM_DECISION:
        raise core.ContractError(
            "latent-momentum decision adapter was not preserved"
        )
    if core._terminal_failure is not _LATENT_MOMENTUM_TERMINAL_FAILURE:
        raise core.ContractError(
            "latent-momentum terminal handler was not preserved"
        )
    core._run = _action_attributed_run
    core._decision = _action_attributed_decision


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != v1.base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(latent_momentum.__file__).resolve() != LATENT_MOMENTUM_RUNNER_SOURCE:
        raise core.ContractError(
            "latent-momentum runner imported from an unexpected path"
        )
    source_bindings = _verify_source_closure()
    v1.base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
