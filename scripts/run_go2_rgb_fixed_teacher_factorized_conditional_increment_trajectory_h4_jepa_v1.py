#!/usr/bin/env python3
"""Run one factorized conditional-increment trajectory-H4 JEPA probe.

This is a thin source-bound adapter over the factual shared-transition V2
schedule-integrity runner.  It preserves that runner's causal indexes,
evaluator, factual proper losses, training loop, selection rule, 32 decision
gates, and work caps.  Only the model/mechanism identity, output/receipt
identity, terminal labels, and mechanism-specific artifact fields change.

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
    run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity
    as v2,
)


v1 = v2.v1
core = v2.core

V2_RUNNER_SOURCE = ROOT / (
    "scripts/"
    "run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_"
    "v2_schedule_integrity.py"
)
V2_RUNNER_SOURCE_SHA256 = (
    "b8d4f861b8a465da6530dd7997a27875dbc431a875bb214badc87c8bb798b14e"
)
V2_RUNNER_SOURCE_BYTES = 10_129
MODEL_MODULE = (
    "lewm.models."
    "go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_"
    "jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_"
    "jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "ff4b0b00b9f6bc165c603ea729b0082d5fc543e02f379427cf288ad8337d2f8c"
MODEL_SOURCE_BYTES = 15_009
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_"
    "jepa_v1/probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_"
    "h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_FACTORIZED_CONDITIONAL_INCREMENT_"
    "TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTORIZED_CONDITIONAL_INCREMENT_"
    "TRAJECTORY_H4_JEPA_V1"
)

_V2_DECISION = v2._schedule_integrity_decision
_V1_FACTUAL_RUN = v1._factual_shared_transition_run


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    """Bind this wrapper, the new model, and the complete frozen V2 closure."""

    wrapper_sha256 = os.environ.get(
        "LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_"
        "SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external factorized conditional-increment wrapper binding is "
            "required"
        ) from error

    source_binding = v1.base._source_binding
    return {
        "factorized_conditional_increment_trajectory_h4_wrapper": (
            source_binding(
                Path(__file__).resolve(),
                wrapper_sha256,
                wrapper_bytes,
            )
        ),
        "v2_schedule_integrity_wrapper_dependency": source_binding(
            V2_RUNNER_SOURCE,
            V2_RUNNER_SOURCE_SHA256,
            V2_RUNNER_SOURCE_BYTES,
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
        "factorized_conditional_increment_trajectory_h4_model": source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "factual_shared_transition_trajectory_h4_model_dependency": (
            source_binding(
                v1.MODEL_SOURCE,
                v1.MODEL_SOURCE_SHA256,
                v1.MODEL_SOURCE_BYTES,
            )
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
    """Install exact V2 science, then replace only the reviewed mechanism."""

    v2._configure_core(source_bindings)
    inherited_schedule = core.ADDITIONAL_SCIENCE.get("schedule_integrity")
    if not isinstance(inherited_schedule, dict):
        raise core.ContractError("V2 schedule-integrity science receipt changed")
    schedule_integrity = dict(inherited_schedule)
    inherited_replacement = schedule_integrity.pop("replacement", None)
    if inherited_replacement != "science_identical_v1_model_and_objective":
        raise core.ContractError("V2 schedule-integrity replacement label changed")
    schedule_integrity["reuse"] = (
        "exact_causal_v2_schedule_with_new_factorized_model"
    )

    core.MODEL_MODULE = MODEL_MODULE
    core.MODEL_SOURCE = MODEL_SOURCE
    core.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    core.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+four_particle_belief_context+incoming_increment_"
            "modulator+complete_tower_centered_categorical_action_code+one_"
            "shared_bias_free_zero_initialized_projection_jointly_trained_in_"
            "one_backward"
        ),
        "support": "four_equal_mass_coherent_six_transition_trajectory_atoms",
        "transition": {
            "shared_core": "one_exact_parameter_set_for_p0_through_p5",
            "observed_steps": (
                "predict_e1_and_e2_before_factual_carrier_insertion_with_"
                "incoming_increment_zero_for_p0_and_realized_e1_minus_e0_for_"
                "p1"
            ),
            "future_steps": (
                "same_core_open_loop_over_p2_through_p5_with_post_"
                "renormalization_realized_increment_carried_recursively"
            ),
            "target_leakage": "none",
        },
        "factorization": {
            "action_free_belief": "B(z_t,h_t,D(d_t))",
            "incoming_increment_modulation": "1+tanh(D(d_t))",
            "categorical_action_code": (
                "c_a=A(E[a])-uniform_mean_over_complete_action_tower"
            ),
            "preprojection_increment": (
                "d_t+B(z_t,h_t,D(d_t))*(1+tanh(D(d_t)))*c_a"
            ),
            "shared_projection": "one_bias_free_zero_initialized_W0",
            "uniform_action_mean": "W0(d_t)",
            "generic_current_state_successor_bypass": False,
        },
        "proper_score": {
            "all_six_factual_local_innovation_weight": 0.5,
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "each_domain": "50_50_joint_plus_mean_marginal_uniform_energy_score",
            "prediction_normalization": "none",
        },
        "training_losses": [
            "half_all_six_factual_local_innovation_energy_score",
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
            "action_cross_entropy_or_inverse_dynamics",
            "control_ranking_margin_or_action_gain_loss",
            "per_action_operator_bank_or_action_query_successor",
            "action_independent_learned_current_state_delta_path",
            "separate_predictor_training_or_predictor_checkpoint",
            "learned_target_compressor_or_target_ema",
            "best_of_k_loss_or_learned_mixture_weights",
            "variance_covariance_or_whitening_loss",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_or_geometry_target",
            "executed_or_clipped_command_input",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
        "schedule_integrity": schedule_integrity,
    }
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _factorized_conditional_increment_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    """Reuse every V2 gate and replace only terminal receipt identity."""

    result = dict(_V2_DECISION(observations, updates_completed))
    failed_gates = result.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise core.ContractError("V2 decision failure list changed")
    expected_v2 = v2.PASS_DECISION if not failed_gates else v2.STOP_DECISION
    if result.get("decision") != expected_v2:
        raise core.ContractError("V2 decision identity disagrees with its gates")
    result["decision"] = PASS_DECISION if not failed_gates else STOP_DECISION
    result["authority"] = (
        "A pass establishes bounded development evidence for this factorized "
        "conditional-increment trajectory JEPA on the frozen candidate-valid "
        "V2 requested-action schedule only; it grants no checkpoint access, "
        "navigation, held-out access, scale promotion, or deployment "
        "authority. A stop closes this exact one-shot mechanism without retry "
        "or resume."
    )
    return result


def _factorized_conditional_increment_run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Reuse the V1 factual run adapter and relabel mechanism artifacts."""

    metrics, artifact, decision = _V1_FACTUAL_RUN(*args, **kwargs)
    adapted = dict(artifact)
    inherited_initialization = adapted.pop(
        "fresh_shared_transition_mode_and_residual_head_initialization",
        None,
    )
    if inherited_initialization is not True:
        raise core.ContractError("factual initialization receipt changed")
    adapted[
        "fresh_factorized_belief_increment_action_and_shared_projection_"
        "initialization"
    ] = True
    adapted["factorized_conditional_increment_mechanism_enabled"] = True
    adapted["factorized_conditional_increment_contract"] = {
        "incoming_increment": (
            "factual_for_observed_edges_and_post_renormalization_realized_for_"
            "open_loop_edges"
        ),
        "action_code": "uniformly_centered_after_complete_action_tower",
        "action_free_belief_current_action_access": False,
        "shared_projection_bias": False,
        "shared_projection_zero_initialized": True,
        "generic_current_state_successor_bypass": False,
    }
    return metrics, adapted, decision


def _install_runtime_adapters() -> None:
    """Install V2, retaining its evaluator and wrapping only run/decision."""

    if core._decision is _factorized_conditional_increment_decision:
        if core._evaluate is not v1._factual_shared_transition_evaluate:
            raise core.ContractError("factorized evaluator identity changed")
        if core._run is not _factorized_conditional_increment_run:
            raise core.ContractError("factorized run handler identity changed")
        return

    v2._install_runtime_adapters()
    if core._evaluate is not v1._factual_shared_transition_evaluate:
        raise core.ContractError("V2 factual evaluator was not preserved")
    if core._run is not _V1_FACTUAL_RUN:
        raise core.ContractError("V1 factual run adapter was not preserved")
    if core._decision is not _V2_DECISION:
        raise core.ContractError("V2 decision adapter was not preserved")
    core._run = _factorized_conditional_increment_run
    core._decision = _factorized_conditional_increment_decision


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != v1.base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(v2.__file__).resolve() != V2_RUNNER_SOURCE:
        raise core.ContractError("frozen V2 runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    v1.base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
