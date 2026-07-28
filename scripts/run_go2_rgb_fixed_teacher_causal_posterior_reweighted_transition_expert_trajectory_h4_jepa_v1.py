#!/usr/bin/env python3
"""Run one causal posterior-reweighted transition-expert H4 JEPA probe.

This source-bound wrapper retains the causal V2 schedules, caps, terminal
handler, selection rule, and 32-gate decision from the frozen system-ID line.
It replaces only the model, truthful mechanism receipts, and the
preregistered posterior-weighted future evaluator.  Import is source-only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_go2_rgb_fixed_teacher_action_attributed_causal_system_identification_trajectory_h4_jepa_v1
    as system_id,
)


latent_momentum = system_id.latent_momentum
factorized = system_id.factorized
v2 = system_id.v2
v1 = system_id.v1
base = v1.base
core = system_id.core

SYSTEM_ID_RUNNER_SOURCE = ROOT / (
    "scripts/"
    "run_go2_rgb_fixed_teacher_action_attributed_causal_system_"
    "identification_trajectory_h4_jepa_v1.py"
)
SYSTEM_ID_RUNNER_SOURCE_SHA256 = (
    "5ab759e0353366b6c5172a0a854ff150a10321b80dee3d645d56f5f286401759"
)
SYSTEM_ID_RUNNER_SOURCE_BYTES = 21_651
MODEL_MODULE = (
    "lewm.models."
    "go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_"
    "trajectory_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_"
    "trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = (
    "cbbc7c7f27021dc77a38405136de473552809fd5141fe60ae773e2fb4772bb99"
)
MODEL_SOURCE_BYTES = 31_330
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_"
    "trajectory_h4_jepa_v1/probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_"
    "expert_trajectory_h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_CAUSAL_POSTERIOR_REWEIGHTED_"
    "TRANSITION_EXPERT_TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_CAUSAL_POSTERIOR_REWEIGHTED_"
    "TRANSITION_EXPERT_TRAJECTORY_H4_JEPA_V1"
)
OBJECTIVE_DESCRIPTION = (
    "0.5*proper_equal_mass_all_six_realized_local_innovation_energy_score_"
    "with_observed_online_z_t_and_future_recursive_q_t_baselines+"
    "0.5*proper_w2_posterior_weighted_open_loop_future_cumulative_"
    "trajectory_energy_score+1*three_frame_online_to_fixed_teacher_"
    "alignment;counterfactual_controls_evaluation_only"
)

_SYSTEM_ID_DECISION = system_id._action_attributed_decision
_SYSTEM_ID_RUN = system_id._action_attributed_run
_FACTORIZED_RUN = factorized._factorized_conditional_increment_run
_SYSTEM_ID_TERMINAL_FAILURE = system_id._LATENT_MOMENTUM_TERMINAL_FAILURE


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    """Bind this wrapper, its model, and every inherited source dependency."""

    prefix = (
        "LEWM_CAUSAL_POSTERIOR_REWEIGHTED_TRANSITION_EXPERT_TRAJECTORY_H4_"
        "V1_WRAPPER_"
    )
    wrapper_sha256 = os.environ.get(prefix + "SHA256", "")
    try:
        wrapper_bytes = int(os.environ.get(prefix + "BYTES", ""))
    except ValueError as error:
        raise core.ContractError(
            "external posterior-reweighted wrapper binding is required"
        ) from error

    source_binding = base._source_binding
    entries = {
        "causal_posterior_reweighted_transition_expert_wrapper": (
            Path(__file__).resolve(),
            wrapper_sha256,
            wrapper_bytes,
        ),
        "action_attributed_system_id_wrapper_dependency": (
            SYSTEM_ID_RUNNER_SOURCE,
            SYSTEM_ID_RUNNER_SOURCE_SHA256,
            SYSTEM_ID_RUNNER_SOURCE_BYTES,
        ),
        "causal_posterior_reweighted_transition_expert_model": (
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "action_attributed_system_id_model_dependency": (
            system_id.MODEL_SOURCE,
            system_id.MODEL_SOURCE_SHA256,
            system_id.MODEL_SOURCE_BYTES,
        ),
        "latent_momentum_wrapper_dependency": (
            system_id.LATENT_MOMENTUM_RUNNER_SOURCE,
            system_id.LATENT_MOMENTUM_RUNNER_SOURCE_SHA256,
            system_id.LATENT_MOMENTUM_RUNNER_SOURCE_BYTES,
        ),
        "latent_momentum_model_dependency": (
            latent_momentum.MODEL_SOURCE,
            latent_momentum.MODEL_SOURCE_SHA256,
            latent_momentum.MODEL_SOURCE_BYTES,
        ),
        "factorized_wrapper_dependency": (
            latent_momentum.FACTORIZED_RUNNER_SOURCE,
            latent_momentum.FACTORIZED_RUNNER_SOURCE_SHA256,
            latent_momentum.FACTORIZED_RUNNER_SOURCE_BYTES,
        ),
        "factorized_model_dependency": (
            factorized.MODEL_SOURCE,
            factorized.MODEL_SOURCE_SHA256,
            factorized.MODEL_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_wrapper_dependency": (
            factorized.V2_RUNNER_SOURCE,
            factorized.V2_RUNNER_SOURCE_SHA256,
            factorized.V2_RUNNER_SOURCE_BYTES,
        ),
        "factual_shared_transition_v1_runner_dependency": (
            v2.V1_RUNNER_SOURCE,
            v2.V1_RUNNER_SOURCE_SHA256,
            v2.V1_RUNNER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_adapter": (
            v2.V2_ADAPTER_SOURCE,
            v2.V2_ADAPTER_SOURCE_SHA256,
            v2.V2_ADAPTER_SOURCE_BYTES,
        ),
        "v2_schedule_integrity_index_builder": (
            v2.V2_BUILDER_SOURCE,
            v2.V2_BUILDER_SOURCE_SHA256,
            v2.V2_BUILDER_SOURCE_BYTES,
        ),
        "factual_shared_transition_model_dependency": (
            v1.MODEL_SOURCE,
            v1.MODEL_SOURCE_SHA256,
            v1.MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_wrapper_dependency": (
            v1.BASE_WRAPPER_SOURCE,
            v1.BASE_WRAPPER_SOURCE_SHA256,
            v1.BASE_WRAPPER_SOURCE_BYTES,
        ),
        "shared_runner": (
            base.CORE_SOURCE,
            base.CORE_SOURCE_SHA256,
            base.CORE_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": (
            v1.TRAJECTORY_MODEL_SOURCE,
            v1.TRAJECTORY_MODEL_SOURCE_SHA256,
            v1.TRAJECTORY_MODEL_SOURCE_BYTES,
        ),
        "local_innovation_model_dependency": (
            v1.LOCAL_INNOVATION_MODEL_SOURCE,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_SHA256,
            v1.LOCAL_INNOVATION_MODEL_SOURCE_BYTES,
        ),
        "dense_h4_model_dependency": (
            base.DENSE_MODEL_SOURCE,
            base.DENSE_MODEL_SOURCE_SHA256,
            base.DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": (
            base.BASE_MODEL_SOURCE,
            base.BASE_MODEL_SOURCE_SHA256,
            base.BASE_MODEL_SOURCE_BYTES,
        ),
        "encoder_dependency": (
            base.ENCODER_SOURCE,
            base.ENCODER_SOURCE_SHA256,
            base.ENCODER_SOURCE_BYTES,
        ),
    }
    return {
        name: source_binding(path, sha256, byte_count)
        for name, (path, sha256, byte_count) in entries.items()
    }


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    """Install the system-ID runtime, then replace only registered science."""

    system_id._configure_core(source_bindings)
    inherited_schedule = core.ADDITIONAL_SCIENCE.get("schedule_integrity")
    if not isinstance(inherited_schedule, dict):
        raise core.ContractError("system-ID schedule-integrity receipt changed")
    schedule_integrity = dict(inherited_schedule)
    expected_reuse = (
        "exact_causal_v2_schedule_with_new_action_attributed_causal_system_"
        "identification_model"
    )
    if schedule_integrity.get("reuse") != expected_reuse:
        raise core.ContractError("system-ID schedule reuse label changed")
    schedule_integrity["reuse"] = (
        "exact_causal_v2_schedule_with_new_causal_posterior_reweighted_"
        "transition_expert_model"
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
            "encoder+four_centered_mode_transition_experts+shared_action_free_"
            "spatial_context+complete_tower_centered_action_code+one_bias_free_"
            "zero_initialized_head+fixed_two_update_posterior_jointly_trained_"
            "in_one_backward"
        ),
        "support": (
            "four_coherent_transition_experts_with_causal_posterior_mass"
        ),
        "state": {
            "content": "four_q_lattices_assimilated_to_each_factual_online_z",
            "sole_history": "four_strictly_positive_simplex_probabilities",
            "initialization": "q0_equals_online_z0_and_w0_equals_uniform_quarter",
            "future_belief": "packed_q2_and_w2_only",
            "serialized_probability_scalars": 4,
            "serialized_padding": "fixed_and_exact_zero",
            "final_hidden_particles": (
                "compatibility_alias_of_posterior_probabilities_not_extra_state"
            ),
            "other_hidden_state": False,
        },
        "transition": {
            "shared_core": "one_exact_parameter_set_for_p0_through_p5",
            "prior": (
                "action_free_B_of_q_and_centered_mode_spatial_context_times_"
                "current_complete_tower_centered_action_then_shared_W0"
            ),
            "observed_order": (
                "predict_and_score_then_fixed_evidence_update_then_factual_q_"
                "assimilation_on_p0_and_p1"
            ),
            "future": "same_prior_over_p2_through_p5_with_w2_bitwise_fixed",
            "probabilities_enter_expert_location_or_increment": False,
            "target_leakage": "none",
        },
        "posterior": {
            "initial_mass": [0.25, 0.25, 0.25, 0.25],
            "update_calls": 2,
            "future_update_calls": 0,
            "squared_error": (
                "mean_token_sum_feature_squared_prior_minus_online_destination"
            ),
            "likelihood": "exp(-d_k/(mean_four_d+1e-6))",
            "update": "normalize(w_previous_times_likelihood)",
            "learned_temperature_gain_gate_prior_or_detach": False,
        },
        "proper_score": {
            "all_six_realized_local_innovation_weight": 0.5,
            "all_six_local_mass": "equal_quarter",
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "future_mass": "causal_w2",
            "future_combination": "50_50_joint_plus_mean_marginal",
            "p0_p1_diagnostic_mass": "equal_quarter",
            "prediction_normalization": "none",
        },
        "evaluation": (
            "posterior_weighted_future_evaluator_with_exact_V2_selection_and_"
            "all_32_unchanged_gates"
        ),
        "evaluation_weight_rules": {
            "real_wrong_action_and_all_hold": "factual_branch_w2",
            "reversed_history": "independently_recomputed_reversed_w2",
            "reset_history": "independently_recomputed_reset_w2",
            "centroid_and_pair_spread": "posterior_weighted",
            "persistence": "weight_invariant_identical_atoms",
        },
        "training_losses": [
            "half_all_six_realized_local_innovation_energy_score",
            "half_open_loop_future_posterior_weighted_cumulative_trajectory_"
            "energy_score",
            "history_teacher_alignment",
        ],
        "training_controls": {
            "enabled": False,
            "wrong_action": False,
            "all_hold": False,
            "reversed_history": False,
            "reset_history": False,
        },
        "absent": [
            "generic_learned_history_updater_or_continuous_history_statistic",
            "writable_matrix_momentum_increment_or_dense_token_memory",
            "learned_likelihood_temperature_gain_entropy_or_resampling",
            "per_action_operator_bank_inverse_classifier_or_ranking_loss",
            "correspondence_transport_cost_volume_flow_warp_or_retrieval",
            "target_compressor_whitening_covariance_reconstruction_or_ema",
            "navigation_pose_depth_bev_geometry_reward_or_labels",
            "separate_predictor_inference_model_optimizer_or_checkpoint",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
        "schedule_integrity": schedule_integrity,
    }
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _probabilities(output: Any, atoms: Any, runtime: Any) -> Any:
    """Extract and fail closed on one Bx4 strict posterior simplex."""

    torch = runtime.torch
    weights = core._extract_tensor(output, "posterior_probabilities")
    expected = (int(atoms.shape[0]), 4)
    if weights.ndim != 2 or tuple(weights.shape) != expected:
        raise core.ContractError("posterior probabilities must have shape B,4")
    if not torch.is_floating_point(weights):
        raise core.ContractError("posterior probabilities must be floating")
    if weights.device != atoms.device or weights.dtype != atoms.dtype:
        raise core.ContractError("posterior and atoms must share device/dtype")
    if not bool(torch.isfinite(weights).all()):
        raise core.ContractError("posterior probabilities are nonfinite")
    if not bool((weights > 0.0).all()):
        raise core.ContractError("posterior probabilities must be positive")
    totals = weights.sum(dim=1)
    if not torch.allclose(
        totals,
        torch.ones_like(totals),
        rtol=0.0,
        atol=1e-6,
    ):
        raise core.ContractError("posterior probabilities must sum to one")
    return weights


def _weighted_energy(
    atoms: Any,
    target: Any,
    weights: Any,
    runtime: Any,
) -> tuple[Any, Any, Any]:
    """Exact posterior-weighted marginal, joint, and combined energy score."""

    if atoms.ndim != 5 or target.ndim != 4:
        raise core.ContractError("weighted energy shapes changed")
    if atoms.shape[1] != 4 or atoms.shape[0] != target.shape[0]:
        raise core.ContractError("weighted energy expert/batch shape changed")
    if atoms.shape[2:] != target.shape[1:]:
        raise core.ContractError("weighted energy atom/target shape changed")
    if tuple(weights.shape) != (atoms.shape[0], atoms.shape[1]):
        raise core.ContractError("weighted energy probability shape changed")
    fit_distance = base._lattice_distance(atoms, target[:, None], runtime)
    pair_distance = base._lattice_distance(
        atoms[:, :, None], atoms[:, None, :], runtime
    )
    fit_horizon = (fit_distance * weights[:, :, None]).sum(dim=1)
    pair_weights = weights[:, :, None] * weights[:, None, :]
    pair_horizon = (
        pair_distance * pair_weights[:, :, :, None]
    ).sum(dim=(1, 2))
    horizon_score = fit_horizon - 0.5 * pair_horizon

    batch, experts, horizons, tokens, dim = atoms.shape
    flat_atoms = atoms.reshape(batch, experts, horizons * tokens, dim)
    flat_target = target.reshape(batch, horizons * tokens, dim)
    joint_fit_distance = base._lattice_distance(
        flat_atoms, flat_target[:, None], runtime
    )
    joint_pair_distance = base._lattice_distance(
        flat_atoms[:, :, None], flat_atoms[:, None, :], runtime
    )
    joint_fit = (joint_fit_distance * weights).sum(dim=1)
    joint_pair = (joint_pair_distance * pair_weights).sum(dim=(1, 2))
    joint_score = joint_fit - 0.5 * joint_pair
    combined = 0.5 * joint_score + 0.5 * horizon_score.mean(dim=1)
    return horizon_score, joint_score, combined


def _weighted_centroid(atoms: Any, weights: Any, runtime: Any) -> Any:
    mean = (atoms * weights[:, :, None, None, None]).sum(dim=1)
    return runtime.torch.nn.functional.normalize(mean, dim=-1, eps=1e-6)


def _weighted_spread(atoms: Any, weights: Any, runtime: Any) -> Any:
    pair_distance = base._lattice_distance(
        atoms[:, :, None], atoms[:, None, :], runtime
    )
    pair_weights = weights[:, :, None] * weights[:, None, :]
    return (pair_distance * pair_weights[:, :, :, None]).sum(dim=(1, 2))


def _weighted_local_ratio(
    atoms: Any,
    target: Any,
    weights: Any,
    runtime: Any,
) -> Any:
    _horizon, _joint, combined = _weighted_energy(
        atoms, target, weights, runtime
    )
    _zero_horizon, _zero_joint, persistence = _weighted_energy(
        runtime.torch.zeros_like(atoms), target, weights, runtime
    )
    return combined / persistence.clamp_min(1e-4)


def _control_distribution(
    model: Any,
    belief: Any,
    actions: Any,
    expected_weights: Any,
    reference_atoms: Any,
    runtime: Any,
) -> Any:
    method = getattr(
        model,
        "predict_trajectory_atoms_and_probabilities_from_belief",
        None,
    )
    if not callable(method):
        raise core.ContractError("posterior control prediction API is absent")
    value = method(belief, actions)
    if not isinstance(value, tuple) or len(value) != 2:
        raise core.ContractError("posterior control prediction API changed")
    atoms, weights = value
    if atoms.shape != reference_atoms.shape:
        raise core.ContractError("posterior control atom shape changed")
    checked = _probabilities(
        {"posterior_probabilities": weights}, atoms, runtime
    )
    if not runtime.torch.equal(checked, expected_weights):
        raise core.ContractError("future control changed factual posterior")
    return atoms


def _belief_probabilities(
    model: Any,
    output: Any,
    atoms: Any,
    output_weights: Any,
    runtime: Any,
) -> Any:
    """Require an output posterior to equal its serialized belief posterior."""

    belief = core._extract_tensor(output, "belief_latents", "belief")
    method = getattr(model, "posterior_probabilities_from_belief", None)
    if not callable(method):
        raise core.ContractError("posterior belief read API is absent")
    belief_weights = _probabilities(
        {"posterior_probabilities": method(belief)}, atoms, runtime
    )
    if not runtime.torch.equal(belief_weights, output_weights):
        raise core.ContractError("output and belief posterior differ")
    return belief


def _posterior_evaluate(
    model: Any,
    rows: Sequence[Any],
    *,
    root_fd: int,
    runtime: Any,
    access: Counter[str],
    device: Any,
    update: int,
) -> dict[str, Any]:
    """Evaluate future distributions under their causal posterior masses."""

    torch = runtime.torch
    model.eval()
    metric_names = (
        "real_normalized_energy_score",
        "action_gap",
        "hold_gap",
        "persistence_gap",
        "history_gap",
        "distribution_value_gap",
        "normalized_pairwise_spread",
        "best_atom_normalized_squared_error",
        "centroid_normalized_squared_error",
    )
    sums: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [0.0] * 4 for name in metric_names}
    )
    joint_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    combined_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    combined_value_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    p0_p1_score_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    p0_p1_gap_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    future_local_score_sums: defaultdict[tuple[str, str], float] = defaultdict(
        float
    )
    counts: Counter[tuple[str, str]] = Counter()
    target_features = []
    online_features = []
    with torch.no_grad():
        for start in range(0, len(rows), core.BATCH_SIZE):
            batch_rows = rows[start : start + core.BATCH_SIZE]
            rgb, actions = core._load_batch(
                batch_rows,
                root_fd=root_fd,
                runtime=runtime,
                access=access,
                device=device,
            )
            history = rgb[:, :3]
            future_rgb = rgb[:, 3:]
            past = actions[:, :2]
            future = actions[:, 2:]
            output = core._model_forward(model, history, past, future)
            atoms = base._trajectory_atoms(output)
            weights = _probabilities(output, atoms, runtime)
            factual_priors = v1._observed_prior_atoms(output)
            future_innovations = v1._future_innovations(output)
            online = core._extract_tensor(
                output, "online_latents", "history_latents"
            )
            belief = _belief_probabilities(
                model, output, atoms, weights, runtime
            )

            target = core._target_encode(model, future_rgb)
            teacher_history_method = getattr(
                model, "_encode_fixed_teacher_history", None
            )
            if not callable(teacher_history_method):
                raise core.ContractError("fixed-teacher history encoder is absent")
            teacher_history = teacher_history_method(history)
            expected_history = (
                history.shape[0],
                3,
                atoms.shape[-2],
                atoms.shape[-1],
            )
            if tuple(teacher_history.shape) != expected_history:
                raise core.ContractError("fixed-teacher history shape changed")
            current_target = teacher_history[:, 2:3].expand(-1, 4, -1, -1)
            persistence_atoms = current_target[:, None].expand_as(atoms)

            wrong_future = (future + 1) % len(core.PRIMITIVES)
            hold_future = torch.full_like(future, core.HOLD_ACTION)
            wrong_atoms = _control_distribution(
                model,
                belief,
                wrong_future,
                weights,
                atoms,
                runtime,
            )
            hold_atoms = _control_distribution(
                model,
                belief,
                hold_future,
                weights,
                atoms,
                runtime,
            )
            reversed_output = core._model_forward(
                model,
                history[:, [1, 0, 2]],
                past[:, [1, 0]],
                future,
            )
            reset_output = core._model_forward(
                model,
                history[:, 2:3].expand(-1, 3, -1, -1, -1).contiguous(),
                torch.full_like(past, core.HOLD_ACTION),
                future,
            )
            reverse_atoms = base._trajectory_atoms(reversed_output)
            reset_atoms = base._trajectory_atoms(reset_output)
            reverse_weights = _probabilities(
                reversed_output, reverse_atoms, runtime
            )
            reset_weights = _probabilities(reset_output, reset_atoms, runtime)
            _belief_probabilities(
                model,
                reversed_output,
                reverse_atoms,
                reverse_weights,
                runtime,
            )
            _belief_probabilities(
                model,
                reset_output,
                reset_atoms,
                reset_weights,
                runtime,
            )

            real_score, joint_real, combined_real = _weighted_energy(
                atoms, target, weights, runtime
            )
            wrong_score, _wrong_joint, _wrong_combined = _weighted_energy(
                wrong_atoms, target, weights, runtime
            )
            hold_score, _hold_joint, _hold_combined = _weighted_energy(
                hold_atoms, target, weights, runtime
            )
            reverse_score, _reverse_joint, _reverse_combined = _weighted_energy(
                reverse_atoms, target, reverse_weights, runtime
            )
            reset_score, _reset_joint, _reset_combined = _weighted_energy(
                reset_atoms, target, reset_weights, runtime
            )
            persistence_score, joint_persistence, combined_persistence = (
                _weighted_energy(
                    persistence_atoms,
                    target,
                    weights,
                    runtime,
                )
            )
            scale = persistence_score.clamp_min(1e-4)
            centroid = _weighted_centroid(atoms, weights, runtime)
            centroid_score = base._lattice_distance(centroid, target, runtime)
            pair_spread = _weighted_spread(atoms, weights, runtime)
            squared = (
                (atoms - target[:, None]).square().sum(dim=-1).mean(dim=-1)
            )
            persistence_squared = (
                (current_target - target).square().sum(dim=-1).mean(dim=-1)
            ).clamp_min(1e-4)
            values = {
                "real_normalized_energy_score": real_score / scale,
                "action_gap": (wrong_score - real_score) / scale,
                "hold_gap": (hold_score - real_score) / scale,
                "persistence_gap": (persistence_score - real_score) / scale,
                "history_gap": (
                    torch.minimum(reverse_score, reset_score) - real_score
                )
                / scale,
                "distribution_value_gap": (
                    centroid_score - real_score
                )
                / scale,
                "normalized_pairwise_spread": pair_spread / scale,
                "best_atom_normalized_squared_error": (
                    squared.min(dim=1).values / persistence_squared
                ),
                "centroid_normalized_squared_error": (
                    (centroid - target).square().sum(dim=-1).mean(dim=-1)
                    / persistence_squared
                ),
            }
            joint_ratio = joint_real / joint_persistence.clamp_min(1e-4)
            combined_scale = combined_persistence.clamp_min(1e-4)
            combined_ratio = combined_real / combined_scale
            flat_centroid = centroid.reshape(
                centroid.shape[0],
                centroid.shape[1] * centroid.shape[2],
                centroid.shape[3],
            )
            flat_target = target.reshape(
                target.shape[0],
                target.shape[1] * target.shape[2],
                target.shape[3],
            )
            joint_centroid = base._lattice_distance(
                flat_centroid, flat_target, runtime
            )
            combined_centroid = (
                0.5 * joint_centroid + 0.5 * centroid_score.mean(dim=1)
            )
            combined_value = (
                combined_centroid - combined_real
            ) / combined_scale

            online_normalized = torch.nn.functional.normalize(
                online, p=2.0, dim=-1, eps=1e-6
            )
            factual_innovations = (
                factual_priors - online_normalized[:, None, :2]
            )
            factual_targets = teacher_history[:, 1:] - teacher_history[:, :-1]
            factual_ratio, factual_gap = v1._normalized_local_combined_score(
                factual_innovations,
                factual_targets,
                runtime,
            )
            future_targets = torch.cat(
                (
                    target[:, :1] - teacher_history[:, 2:3],
                    target[:, 1:] - target[:, :-1],
                ),
                dim=1,
            )
            future_local_ratio = _weighted_local_ratio(
                future_innovations,
                future_targets,
                weights,
                runtime,
            )

            for row_index, row in enumerate(batch_rows):
                key = (row.family, row.scene_id)
                counts[key] += 1
                joint_sums[key] += float(joint_ratio[row_index].item())
                combined_sums[key] += float(combined_ratio[row_index].item())
                combined_value_sums[key] += float(
                    combined_value[row_index].item()
                )
                p0_p1_score_sums[key] += float(factual_ratio[row_index].item())
                p0_p1_gap_sums[key] += float(factual_gap[row_index].item())
                future_local_score_sums[key] += float(
                    future_local_ratio[row_index].item()
                )
                for name in metric_names:
                    vector = values[name][row_index].detach().cpu().tolist()
                    for horizon in range(4):
                        sums[key][name][horizon] += float(vector[horizon])
            target_features.append(
                core._pool_features(target, time_index=3).detach().cpu()
            )
            online_features.append(
                core._pool_features(online, time_index=2).detach().cpu()
            )
            access["validation_sequence_presentation_count"] += len(batch_rows)

    scene_metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for key, values in sums.items():
        scene_metrics[key] = {
            name: [item / counts[key] for item in vector]
            for name, vector in values.items()
        }
        scene_metrics[key]["joint_trajectory_normalized_energy_score"] = (
            joint_sums[key] / counts[key]
        )
        scene_metrics[key]["combined_normalized_energy_score"] = (
            combined_sums[key] / counts[key]
        )
        scene_metrics[key]["combined_distribution_value_gap"] = (
            combined_value_sums[key] / counts[key]
        )
        scene_metrics[key][
            "p0_p1_local_prior_combined_normalized_energy_score"
        ] = p0_p1_score_sums[key] / counts[key]
        scene_metrics[key]["p0_p1_local_prior_persistence_gap"] = (
            p0_p1_gap_sums[key] / counts[key]
        )
        scene_metrics[key][
            "future_p2_p5_local_combined_normalized_energy_score"
        ] = future_local_score_sums[key] / counts[key]

    aggregate: dict[str, Any] = {}
    family_metrics: dict[str, dict[str, Any]] = {
        family: {} for family in core.FAMILIES
    }
    for name in metric_names:
        family_vectors = []
        for family in core.FAMILIES:
            scene_vectors = [
                metrics[name]
                for (item_family, _scene), metrics in scene_metrics.items()
                if item_family == family
            ]
            if not scene_vectors:
                raise core.ContractError("validation macro lost a family")
            vector = [
                sum(item[horizon] for item in scene_vectors)
                / len(scene_vectors)
                for horizon in range(4)
            ]
            family_metrics[family][name] = vector
            family_vectors.append(vector)
        aggregate[name] = [
            sum(vector[horizon] for vector in family_vectors)
            / len(family_vectors)
            for horizon in range(4)
        ]
    scalar_names = (
        "joint_trajectory_normalized_energy_score",
        "combined_normalized_energy_score",
        "combined_distribution_value_gap",
        "p0_p1_local_prior_combined_normalized_energy_score",
        "p0_p1_local_prior_persistence_gap",
        "future_p2_p5_local_combined_normalized_energy_score",
    )
    for name in scalar_names:
        family_values = []
        for family in core.FAMILIES:
            scene_values = [
                metrics[name]
                for (item_family, _scene), metrics in scene_metrics.items()
                if item_family == family
            ]
            value = sum(scene_values) / len(scene_values)
            family_metrics[family][name] = value
            family_values.append(value)
        aggregate[name] = sum(family_values) / len(family_values)

    lower_bounds: dict[str, float] = {}
    for offset, name in enumerate(
        ("action_gap", "persistence_gap", "history_gap", "distribution_value_gap")
    ):
        values_by_family = {
            family: {
                scene: metrics[name][3]
                for (item_family, scene), metrics in scene_metrics.items()
                if item_family == family
            }
            for family in core.FAMILIES
        }
        lower_bounds[f"{name}_h4"] = core._bootstrap_lower(
            values_by_family,
            seed=core.SEED + update * 10 + offset,
        )
    combined_value_by_family = {
        family: {
            scene: metrics["combined_distribution_value_gap"]
            for (item_family, scene), metrics in scene_metrics.items()
            if item_family == family
        }
        for family in core.FAMILIES
    }
    lower_bounds["combined_distribution_value_gap"] = core._bootstrap_lower(
        combined_value_by_family,
        seed=core.SEED + update * 10 + 4,
    )
    p0_p1_gap_by_family = {
        family: {
            scene: metrics["p0_p1_local_prior_persistence_gap"]
            for (item_family, scene), metrics in scene_metrics.items()
            if item_family == family
        }
        for family in core.FAMILIES
    }
    lower_bounds["p0_p1_local_prior_persistence_gap"] = core._bootstrap_lower(
        p0_p1_gap_by_family,
        seed=core.SEED + update * 10 + 5,
    )
    target_rank, target_near_zero = core._effective_rank(
        torch.cat(target_features, dim=0), runtime
    )
    online_rank, online_near_zero = core._effective_rank(
        torch.cat(online_features, dim=0), runtime
    )
    finite_values = [
        value
        for name, vectors in aggregate.items()
        if name not in scalar_names
        for value in vectors
    ] + [
        *(aggregate[name] for name in scalar_names),
        *lower_bounds.values(),
        target_rank,
        target_near_zero,
        online_rank,
        online_near_zero,
    ]
    result = {
        "update": update,
        "presentations": update * core.BATCH_SIZE,
        "validation_rows": len(rows),
        "aggregate": aggregate,
        "family": family_metrics,
        "bootstrap_lower_95": lower_bounds,
        "noncollapse": {
            "target_effective_rank_ratio": target_rank,
            "online_effective_rank_ratio": online_rank,
            "target_near_zero_variance_fraction": target_near_zero,
            "online_near_zero_variance_fraction": online_near_zero,
        },
        "all_registered_values_finite": all(
            math.isfinite(value) for value in finite_values
        ),
    }
    model.train()
    return result


def _posterior_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    """Preserve all 32 gates and replace only terminal identity/authority."""

    result = dict(_SYSTEM_ID_DECISION(observations, updates_completed))
    failed_gates = result.get("failed_gates")
    if not isinstance(failed_gates, list):
        raise core.ContractError("system-ID decision failure list changed")
    expected = (
        system_id.PASS_DECISION
        if not failed_gates
        else system_id.STOP_DECISION
    )
    if result.get("decision") != expected:
        raise core.ContractError("system-ID decision identity disagrees with gates")
    result["decision"] = PASS_DECISION if not failed_gates else STOP_DECISION
    result["authority"] = (
        "A pass establishes bounded perception/world-model development evidence "
        "for this causal posterior-reweighted transition-expert JEPA on the "
        "frozen V2 requested-action schedule only; it grants no checkpoint "
        "access, navigation, G2, held-out or sealed access, promotion, "
        "production, or deployment authority. A stop closes this exact "
        "one-shot mechanism without retry, resume, or replacement."
    )
    return result


def _posterior_run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Reuse bounded execution and replace only truthful mechanism receipts."""

    metrics, artifact, decision = _FACTORIZED_RUN(*args, **kwargs)
    adapted_metrics = dict(metrics)
    inherited_training = metrics.get("training_losses")
    if not isinstance(inherited_training, dict):
        raise core.ContractError("inherited training-loss receipt changed")
    training = dict(inherited_training)
    if training.get("objective") != OBJECTIVE_DESCRIPTION:
        raise core.ContractError("posterior-weighted objective receipt changed")
    factual_name = "half_all_six_factual_local_innovation_energy_score"
    realized_name = "half_all_six_realized_local_innovation_energy_score"
    inherited_cumulative_name = (
        "half_open_loop_future_cumulative_trajectory_energy_score"
    )
    weighted_cumulative_name = (
        "half_open_loop_future_posterior_weighted_cumulative_trajectory_"
        "energy_score"
    )
    for bucket_name in ("mean_over_completed_updates", "last_completed_update"):
        inherited_bucket = training.get(bucket_name)
        if not isinstance(inherited_bucket, dict):
            raise core.ContractError("inherited training-loss bucket changed")
        bucket = dict(inherited_bucket)
        if factual_name not in bucket or realized_name in bucket:
            raise core.ContractError("inherited local-loss field changed")
        if (
            inherited_cumulative_name not in bucket
            or weighted_cumulative_name in bucket
        ):
            raise core.ContractError("inherited cumulative-loss field changed")
        bucket[realized_name] = bucket.pop(factual_name)
        bucket[weighted_cumulative_name] = bucket.pop(
            inherited_cumulative_name
        )
        training[bucket_name] = bucket
    inherited_semantics = training.get("receipt_field_semantics")
    if not isinstance(inherited_semantics, dict):
        raise core.ContractError("inherited loss semantics changed")
    semantics = dict(inherited_semantics)
    if semantics.pop(factual_name, None) != (
        "objective_term_already_weighted_one_half"
    ):
        raise core.ContractError("inherited local-loss semantics changed")
    if semantics.pop(inherited_cumulative_name, None) != (
        "objective_term_already_weighted_one_half"
    ):
        raise core.ContractError("inherited cumulative-loss semantics changed")
    semantics[realized_name] = (
        "objective_term_already_weighted_one_half_equal_mass_with_observed_"
        "z_t_and_future_recursive_q_t_baselines"
    )
    semantics[weighted_cumulative_name] = (
        "objective_term_already_weighted_one_half_with_causal_w2_"
        "posterior_mass"
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
    inherited_contract = adapted.pop(
        "factorized_conditional_increment_contract", None
    )
    if inherited_contract != {
        "incoming_increment": (
            "factual_for_observed_edges_and_post_renormalization_realized_for_"
            "open_loop_edges"
        ),
        "action_code": "uniformly_centered_after_complete_action_tower",
        "action_free_belief_current_action_access": False,
        "shared_projection_bias": False,
        "shared_projection_zero_initialized": True,
        "generic_current_state_successor_bypass": False,
    }:
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
        "fresh_causal_posterior_reweighted_modes_context_action_and_shared_"
        "head_initialization"
    ] = True
    adapted["causal_posterior_reweighted_transition_expert_enabled"] = True
    adapted["causal_posterior_reweighted_score_weights"] = {
        "equal_mass_all_six_realized_local_innovations": 0.5,
        "w2_weighted_open_loop_future_cumulative_trajectory": 0.5,
    }
    adapted["causal_posterior_reweighted_transition_expert_contract"] = {
        "state": "four_q_content_experts_plus_four_probability_simplex",
        "initial_probabilities": [0.25, 0.25, 0.25, 0.25],
        "shared_prior_calls": 6,
        "post_prior_evidence_update_calls": 2,
        "future_evidence_update_calls": 0,
        "evidence_error": (
            "mean_token_sum_feature_squared_prior_minus_online_destination"
        ),
        "likelihood": "exp(-d_k/(mean_four_d+1e-6))",
        "posterior_update": "normalize(w_previous_times_likelihood)",
        "learned_temperature_gain_gate_prior_or_detach": False,
        "factual_assimilation": "q_next_equals_online_destination_for_all_four",
        "final_hidden_particles": (
            "compatibility_alias_of_posterior_probabilities_not_extra_state"
        ),
        "future_probabilities_bitwise_fixed": True,
        "probabilities_move_expert_content_or_increment": False,
        "observed_local_score_mass": "equal_quarter",
        "p0_p1_diagnostic_mass": "equal_quarter",
        "future_score_mass": "causal_w2",
        "wrong_and_hold_mass": "factual_w2",
        "reverse_and_reset_mass": "independently_recomputed_branch_w2",
        "action_code": "uniformly_centered_after_complete_action_tower",
        "increment_projection_bias": False,
        "increment_projection_zero_initialized": True,
        "future_raw_z_or_continuous_history_bypass": False,
    }
    return adapted_metrics, adapted, decision


def _install_runtime_adapters() -> None:
    """Install system-ID runtime, replacing evaluator/run/decision only."""

    if core._decision is _posterior_decision:
        if core._evaluate is not _posterior_evaluate:
            raise core.ContractError("posterior evaluator identity changed")
        if core._run is not _posterior_run:
            raise core.ContractError("posterior run handler identity changed")
        if core._terminal_failure is not _SYSTEM_ID_TERMINAL_FAILURE:
            raise core.ContractError("system-ID terminal handler identity changed")
        return

    system_id._install_runtime_adapters()
    if core._evaluate is not v1._factual_shared_transition_evaluate:
        raise core.ContractError("system-ID evaluator was not preserved")
    if core._run is not _SYSTEM_ID_RUN:
        raise core.ContractError("system-ID run adapter was not preserved")
    if core._decision is not _SYSTEM_ID_DECISION:
        raise core.ContractError("system-ID decision adapter was not preserved")
    if core._terminal_failure is not _SYSTEM_ID_TERMINAL_FAILURE:
        raise core.ContractError("system-ID terminal handler was not preserved")
    core._evaluate = _posterior_evaluate
    core._run = _posterior_run
    core._decision = _posterior_decision


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(system_id.__file__).resolve() != SYSTEM_ID_RUNNER_SOURCE:
        raise core.ContractError("system-ID runner imported from unexpected path")
    source_bindings = _verify_source_closure()
    base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
