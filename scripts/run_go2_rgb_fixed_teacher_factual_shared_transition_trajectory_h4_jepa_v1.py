#!/usr/bin/env python3
"""Run one fixed-teacher factual shared-transition trajectory H4 JEPA probe."""
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
    "lewm.models."
    "go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "38e264f8e18ffa3c3da4775fdd7d4a38549e8544f99cd863bfd2534999cd5b36"
MODEL_SOURCE_BYTES = 21_734
OUTPUT_ROOT = ROOT / (
    ".generated/"
    "go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1/"
    "probe_v1"
)
SCHEMA = (
    "lewm_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v1"
)
PASS_DECISION = (
    "PASS_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_"
    "TRAJECTORY_H4_JEPA_V1"
)
STOP_DECISION = (
    "STOP_MAIN_POOL_RGB_FIXED_TEACHER_FACTUAL_SHARED_TRANSITION_"
    "TRAJECTORY_H4_JEPA_V1"
)
TRAJECTORY_MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1.py"
)
TRAJECTORY_MODEL_SOURCE_SHA256 = base.MODEL_SOURCE_SHA256
TRAJECTORY_MODEL_SOURCE_BYTES = base.MODEL_SOURCE_BYTES
LOCAL_INNOVATION_MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1.py"
)
LOCAL_INNOVATION_MODEL_SOURCE_SHA256 = (
    "f71639ee80bcadfa0e5c3238b55164deb57d196c2bdfa4bd0ced3bb5cba1bb71"
)
LOCAL_INNOVATION_MODEL_SOURCE_BYTES = 17_248
_BASE_DECISION = base._trajectory_decision
_BASE_RUN = base._trajectory_run


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get(
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_WRAPPER_SHA256",
        "",
    )
    wrapper_bytes_text = os.environ.get(
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_WRAPPER_BYTES",
        "",
    )
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external factual shared-transition wrapper binding is required"
        ) from error
    return {
        "factual_shared_transition_trajectory_h4_wrapper": base._source_binding(
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
        "factual_shared_transition_trajectory_h4_model": base._source_binding(
            MODEL_SOURCE,
            MODEL_SOURCE_SHA256,
            MODEL_SOURCE_BYTES,
        ),
        "trajectory_h4_model_dependency": base._source_binding(
            TRAJECTORY_MODEL_SOURCE,
            TRAJECTORY_MODEL_SOURCE_SHA256,
            TRAJECTORY_MODEL_SOURCE_BYTES,
        ),
        "local_innovation_trajectory_h4_model_dependency": base._source_binding(
            LOCAL_INNOVATION_MODEL_SOURCE,
            LOCAL_INNOVATION_MODEL_SOURCE_SHA256,
            LOCAL_INNOVATION_MODEL_SOURCE_BYTES,
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
        "fixed_N320_all_six_factual_innovations_and_open_loop_future_"
        "trajectory_stop_gradient_no_ema"
    )
    core.OBJECTIVE_DESCRIPTION = (
        "0.5*proper_all_six_factual_local_innovation_energy_score+"
        "0.5*proper_open_loop_future_cumulative_trajectory_energy_score+"
        "1*three_frame_online_to_fixed_teacher_alignment;"
        "counterfactual_controls_evaluation_only"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+one_shared_spatial_transition+four_fixed_mode_embeddings+"
            "one_shared_zero_initialized_residual_head_jointly_trained_in_one_"
            "backward"
        ),
        "support": "four_equal_mass_coherent_six_transition_trajectory_atoms",
        "transition": {
            "shared_core": "one_exact_parameter_set_for_p0_through_p5",
            "observed_steps": (
                "predict_e1_and_e2_before_observation_then_teacher_force_the_"
                "factual_online_carrier_while_retaining_transition_hidden_state"
            ),
            "future_steps": "same_core_open_loop_over_p2_through_p5",
            "target_leakage": "none",
        },
        "proper_score": {
            "all_six_factual_local_innovation_weight": 0.5,
            "open_loop_future_cumulative_trajectory_weight": 0.5,
            "each_domain": (
                "50_50_joint_plus_mean_marginal_uniform_energy_score"
            ),
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
        "evaluation": (
            "inherited_trajectory_distribution_metrics_and_all_28_"
            "dual_domain_gates_including_hold_breadth_and_floor"
        ),
        "absent": [
            "horizon_specific_queries_or_heads",
            "future_action_prefix_decoder",
            "separate_history_and_future_transition_cells",
            "control_ranking_or_margin_loss",
            "learned_target_compressor_or_target_ema",
            "best_of_k_loss_or_learned_mixture_weights",
            "variance_covariance_or_whitening_loss",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_or_geometry_target",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _observed_prior_atoms(output: Any) -> Any:
    value = core._extract_tensor(output, "observed_prior_latents")
    if value.ndim != 5 or value.shape[1] != 4 or value.shape[2] != 2:
        raise core.ContractError("factual prior atom shape contract changed")
    return value


def _future_innovations(output: Any) -> Any:
    value = core._extract_tensor(output, "trajectory_innovations")
    if value.ndim != 5 or value.shape[1] != 4 or value.shape[2] != 4:
        raise core.ContractError("future innovation atom shape contract changed")
    return value


def _normalized_local_combined_score(
    atoms: Any,
    target: Any,
    runtime: Any,
) -> tuple[Any, Any]:
    """Normalize a local-innovation proper score by exact zero innovation."""

    zero = runtime.torch.zeros_like(atoms)
    marginal = base._marginal_energy(atoms, target, runtime)
    joint = base._joint_energy(atoms, target, runtime)
    zero_marginal = base._marginal_energy(zero, target, runtime)
    zero_joint = base._joint_energy(zero, target, runtime)
    combined = 0.5 * joint + 0.5 * marginal.mean(dim=1)
    zero_combined = 0.5 * zero_joint + 0.5 * zero_marginal.mean(dim=1)
    denominator = zero_combined.clamp_min(1e-6)
    return (
        combined / denominator,
        (zero_combined - combined) / denominator,
    )


def _factual_shared_transition_evaluate(
    model: Any,
    rows: Sequence[Any],
    *,
    root_fd: int,
    runtime: Any,
    access: Counter[str],
    device: Any,
    update: int,
) -> dict[str, Any]:
    """Run the inherited future evaluator plus factual p0/p1 diagnostics."""

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
            factual_priors = _observed_prior_atoms(output)
            future_innovations = _future_innovations(output)
            online = core._extract_tensor(output, "online_latents", "history_latents")
            belief = core._extract_tensor(output, "belief_latents", "belief")
            target = core._target_encode(model, future_rgb)
            teacher_history_method = getattr(model, "_encode_fixed_teacher_history", None)
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
            wrong_atoms = model.predict_trajectory_atoms_from_belief(
                belief, wrong_future
            )
            hold_atoms = model.predict_trajectory_atoms_from_belief(
                belief, hold_future
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

            real_score = base._marginal_energy(atoms, target, runtime)
            wrong_score = base._marginal_energy(wrong_atoms, target, runtime)
            hold_score = base._marginal_energy(hold_atoms, target, runtime)
            reverse_score = base._marginal_energy(reverse_atoms, target, runtime)
            reset_score = base._marginal_energy(reset_atoms, target, runtime)
            persistence_score = base._marginal_energy(
                persistence_atoms, target, runtime
            )
            scale = persistence_score.clamp_min(1e-4)
            centroid = torch.nn.functional.normalize(
                atoms.mean(dim=1), dim=-1, eps=1e-6
            )
            centroid_score = base._lattice_distance(centroid, target, runtime)
            pair_spread = base._lattice_distance(
                atoms[:, :, None], atoms[:, None, :], runtime
            ).mean(dim=(1, 2))
            squared = (atoms - target[:, None]).square().sum(dim=-1).mean(dim=-1)
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
                "distribution_value_gap": (centroid_score - real_score) / scale,
                "normalized_pairwise_spread": pair_spread / scale,
                "best_atom_normalized_squared_error": (
                    squared.min(dim=1).values / persistence_squared
                ),
                "centroid_normalized_squared_error": (
                    (centroid - target).square().sum(dim=-1).mean(dim=-1)
                    / persistence_squared
                ),
            }
            joint_real = base._joint_energy(atoms, target, runtime)
            joint_persistence = base._joint_energy(
                persistence_atoms, target, runtime
            )
            joint_ratio = joint_real / joint_persistence.clamp_min(1e-4)
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
            combined_real = 0.5 * joint_real + 0.5 * real_score.mean(dim=1)
            combined_persistence = (
                0.5 * joint_persistence + 0.5 * persistence_score.mean(dim=1)
            ).clamp_min(1e-4)
            combined_ratio = combined_real / combined_persistence
            combined_centroid = (
                0.5 * joint_centroid + 0.5 * centroid_score.mean(dim=1)
            )
            combined_value = (
                combined_centroid - combined_real
            ) / combined_persistence

            online_normalized = torch.nn.functional.normalize(
                online,
                p=2.0,
                dim=-1,
                eps=1e-6,
            )
            factual_innovations = (
                factual_priors - online_normalized[:, None, :2]
            )
            factual_targets = teacher_history[:, 1:] - teacher_history[:, :-1]
            factual_ratio, factual_gap = _normalized_local_combined_score(
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
            future_local_ratio, _future_local_gap = (
                _normalized_local_combined_score(
                future_innovations,
                future_targets,
                runtime,
                )
            )

            for row_index, row in enumerate(batch_rows):
                key = (row.family, row.scene_id)
                counts[key] += 1
                joint_sums[key] += float(joint_ratio[row_index].item())
                combined_sums[key] += float(combined_ratio[row_index].item())
                combined_value_sums[key] += float(combined_value[row_index].item())
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
                sum(item[horizon] for item in scene_vectors) / len(scene_vectors)
                for horizon in range(4)
            ]
            family_metrics[family][name] = vector
            family_vectors.append(vector)
        aggregate[name] = [
            sum(vector[horizon] for vector in family_vectors) / len(family_vectors)
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


def _factual_shared_transition_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    result = dict(_BASE_DECISION(observations, updates_completed))
    gates = dict(result["gates"])
    diagnostics = dict(result["diagnostics"])
    baseline = next(item for item in observations if item["update"] == 0)
    baseline_aggregate = baseline["aggregate"]
    gates["update_zero_is_exact_persistence"] = bool(
        gates["update_zero_is_exact_persistence"]
        and abs(
            baseline_aggregate[
                "p0_p1_local_prior_combined_normalized_energy_score"
            ]
            - 1.0
        )
        <= 1e-5
        and abs(
            baseline_aggregate["p0_p1_local_prior_persistence_gap"]
        )
        <= 1e-5
    )
    selected_update = diagnostics.get("selected_update")
    selected = next(
        (item for item in observations if item["update"] == selected_update),
        None,
    )
    if selected is None:
        hold_positive = 0
        minimum_hold = None
        p0_p1_positive = 0
        minimum_p0_p1_gap = None
        gates["hold_positive_in_six_families"] = False
        gates["no_family_hold_gap_below_minus_point02"] = False
        gates[
            "p0_p1_local_prior_combined_score_below_persistence"
        ] = False
        gates[
            "p0_p1_local_prior_persistence_gap_bootstrap_lower_positive"
        ] = False
        gates[
            "p0_p1_local_prior_persistence_positive_in_six_families"
        ] = False
        gates[
            "no_family_p0_p1_local_prior_persistence_gap_below_minus_point02"
        ] = False
    else:
        family_holds = [
            selected["family"][family]["hold_gap"][3]
            for family in core.FAMILIES
        ]
        hold_positive = sum(value > 0.0 for value in family_holds)
        minimum_hold = min(family_holds)
        gates["hold_positive_in_six_families"] = hold_positive >= 6
        gates["no_family_hold_gap_below_minus_point02"] = minimum_hold >= -0.02
        p0_p1_score = selected["aggregate"][
            "p0_p1_local_prior_combined_normalized_energy_score"
        ]
        family_p0_p1_gaps = [
            selected["family"][family]["p0_p1_local_prior_persistence_gap"]
            for family in core.FAMILIES
        ]
        p0_p1_positive = sum(value > 0.0 for value in family_p0_p1_gaps)
        minimum_p0_p1_gap = min(family_p0_p1_gaps)
        gates[
            "p0_p1_local_prior_combined_score_below_persistence"
        ] = p0_p1_score < 1.0
        gates[
            "p0_p1_local_prior_persistence_gap_bootstrap_lower_positive"
        ] = (
            selected["bootstrap_lower_95"][
                "p0_p1_local_prior_persistence_gap"
            ]
            > 0.0
        )
        gates[
            "p0_p1_local_prior_persistence_positive_in_six_families"
        ] = p0_p1_positive >= 6
        gates[
            "no_family_p0_p1_local_prior_persistence_gap_below_minus_point02"
        ] = minimum_p0_p1_gap >= -0.02
        diagnostics[
            "p0_p1_local_prior_combined_normalized_energy_score"
        ] = p0_p1_score
        diagnostics["p0_p1_local_prior_persistence_gap"] = selected[
            "aggregate"
        ]["p0_p1_local_prior_persistence_gap"]
        diagnostics[
            "future_p2_p5_local_combined_normalized_energy_score"
        ] = selected["aggregate"][
            "future_p2_p5_local_combined_normalized_energy_score"
        ]
    diagnostics["hold_positive_family_count"] = hold_positive
    diagnostics["minimum_family_h4_hold_gap"] = minimum_hold
    diagnostics["p0_p1_persistence_positive_family_count"] = p0_p1_positive
    diagnostics["minimum_family_p0_p1_persistence_gap"] = minimum_p0_p1_gap
    failed_gates = sorted(name for name, value in gates.items() if not value)
    return {
        "decision": PASS_DECISION if not failed_gates else STOP_DECISION,
        "gates": gates,
        "failed_gates": failed_gates,
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded development factual shared-transition "
            "trajectory JEPA feasibility only; it grants no checkpoint access, "
            "navigation, held-out access, scale promotion, or deployment "
            "authority. A stop closes this exact factual shared-transition "
            "mechanism."
        ),
    }


def _factual_shared_transition_run(
    *args: Any,
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    access = kwargs.get("access")
    if access is None:
        raise core.ContractError("factual shared-transition run requires access")
    try:
        metrics, artifact, decision = _BASE_RUN(*args, **kwargs)
    finally:
        if int(access.get("auxiliary_training_control_sequence_count", 0)) != 0:
            raise core.ContractError(
                "factual shared-transition auxiliary training control ran"
            )
        if int(access.get("wrong_action_counterfactual_sequence_count", 0)) != 0:
            raise core.ContractError(
                "factual shared-transition wrong-action training control ran"
            )

    training = metrics.get("training_losses")
    if not isinstance(training, dict):
        raise core.ContractError("factual shared-transition training receipt absent")
    objective_names = {
        "history_teacher_alignment",
        "half_all_six_factual_local_innovation_energy_score",
        "half_open_loop_future_cumulative_trajectory_energy_score",
    }
    inherited_names = {
        "prediction",
        "variance",
        "wrong_action_ranking",
        "total",
    }
    for bucket_name in ("mean_over_completed_updates", "last_completed_update"):
        bucket = training.get(bucket_name)
        if not isinstance(bucket, dict) or set(bucket) != (
            objective_names | inherited_names
        ):
            raise core.ContractError(
                "factual shared-transition loss receipt fields changed"
            )
        diagnostic = bucket.pop("prediction")
        if not math.isfinite(diagnostic):
            raise core.ContractError("centroid prediction diagnostic is non-finite")
        for disabled_name in ("variance", "wrong_action_ranking"):
            value = bucket.pop(disabled_name)
            if value != 0.0:
                raise core.ContractError("disabled inherited training term changed")
        bucket["diagnostic_centroid_absolute_future_error"] = diagnostic
        expected_total = sum(float(bucket[name]) for name in objective_names)
        if not math.isclose(
            float(bucket["total"]),
            expected_total,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise core.ContractError(
                "factual shared-transition objective arithmetic changed"
            )
    training["disabled_terms"] = [
        "absolute_future_centroid_prediction_in_objective",
        "inherited_variance_regularization",
        "built_in_centroid_wrong_action_ranking",
        "cyclic_wrong_action_score_ranking",
        "all_hold_score_ranking",
        "reversed_or_reset_history_score_ranking",
    ]
    training["receipt_field_semantics"] = {
        "diagnostic_centroid_absolute_future_error": (
            "measured_by_shared_runner_but_weight_zero"
        ),
        "history_teacher_alignment": "objective_weight_one",
        "half_all_six_factual_local_innovation_energy_score": (
            "objective_term_already_weighted_one_half"
        ),
        "half_open_loop_future_cumulative_trajectory_energy_score": (
            "objective_term_already_weighted_one_half"
        ),
    }

    adapted = dict(artifact)
    inherited_initialization = adapted.pop(
        "fresh_dense_history_mode_embeddings_action_path_and_shared_delta_head_"
        "initialization",
        None,
    )
    if inherited_initialization is not True:
        raise core.ContractError("trajectory initialization receipt changed")
    built_in_control = adapted.pop("wrong_action_training_contrast_enabled", None)
    if built_in_control is not False:
        raise core.ContractError("built-in wrong-action receipt changed")
    adapted["fresh_shared_transition_mode_and_residual_head_initialization"] = True
    adapted["factual_shared_transition_objective_enabled"] = True
    adapted["factual_shared_transition_score_weights"] = {
        "all_six_factual_local_innovation": 0.5,
        "open_loop_future_cumulative_trajectory": 0.5,
    }
    adapted["built_in_centroid_wrong_action_training_contrast_enabled"] = False
    adapted["cyclic_wrong_action_training_contrast_enabled"] = False
    adapted["all_hold_training_contrast_enabled"] = False
    adapted["reversed_and_reset_history_training_contrasts_enabled"] = False
    return metrics, adapted, decision


def _install_runtime_adapters() -> None:
    base._install_runtime_adapters()
    if core._evaluate not in (
        base._trajectory_evaluate,
        _factual_shared_transition_evaluate,
    ):
        raise core.ContractError(
            "trajectory evaluator changed before factual shared-transition adapter"
        )
    core._evaluate = _factual_shared_transition_evaluate
    if core._decision not in (_BASE_DECISION, _factual_shared_transition_decision):
        raise core.ContractError(
            "trajectory decision changed before factual shared-transition adapter"
        )
    core._decision = _factual_shared_transition_decision
    if core._run not in (_BASE_RUN, _factual_shared_transition_run):
        raise core.ContractError(
            "trajectory runner changed before factual shared-transition adapter"
        )
    core._run = _factual_shared_transition_run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != base.CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    if Path(base.__file__).resolve() != BASE_WRAPPER_SOURCE:
        raise core.ContractError("trajectory wrapper imported from unexpected path")
    source_bindings = _verify_source_closure()
    base._install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
