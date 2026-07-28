#!/usr/bin/env python3
"""Run one fresh RGB fixed-teacher trajectory-distribution H4 JEPA probe."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import os
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_go2_recurrent_h4_joint_jepa_v1 as core  # noqa: E402


CORE_SOURCE = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"
CORE_SOURCE_SHA256 = "fc35d535e1c07b56c667474e6e10c5c7587fa01e627567148c022331983616fc"
CORE_SOURCE_BYTES = 70_301
MODEL_MODULE = (
    "lewm.models.go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1"
)
MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "1d07f8f67ab7f3e93534454a2aaf2ff8626f984b5346a2717c1643a331eb6d8e"
MODEL_SOURCE_BYTES = 15_967
DENSE_MODEL_SOURCE = ROOT / (
    "lewm/models/"
    "go2_rgb_fixed_teacher_dense_spatiotemporal_cross_attention_h4_jepa_v1.py"
)
DENSE_MODEL_SOURCE_SHA256 = (
    "5c74675b93667e6035fc21c9fe497880ba4bff22641b3e735272e4cc1ede3d30"
)
DENSE_MODEL_SOURCE_BYTES = 23_712
BASE_MODEL_SOURCE = ROOT / "lewm/models/go2_recurrent_h4_joint_jepa.py"
BASE_MODEL_SOURCE_SHA256 = "ddd84561aba5a36df1255ab942bb29db943cc1bf7b0e496ae41b3d1cdc218f55"
BASE_MODEL_SOURCE_BYTES = 21_166
ENCODER_SOURCE = ROOT / "lewm/models/encoders.py"
ENCODER_SOURCE_SHA256 = "5eed7bbe424d5ddd293ea67ed1596e74504c68dd8da93f8420795f216cb7599d"
ENCODER_SOURCE_BYTES = 7_028
OUTPUT_ROOT = ROOT / (
    ".generated/go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1/"
    "probe_v1"
)
SCHEMA = "lewm_go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1"
PASS_DECISION = "PASS_MAIN_POOL_RGB_FIXED_TEACHER_TRAJECTORY_DISTRIBUTION_H4_JEPA_V1"
STOP_DECISION = "STOP_MAIN_POOL_RGB_FIXED_TEACHER_TRAJECTORY_DISTRIBUTION_H4_JEPA_V1"
_GEOMETRY_TOLERANCE = 1e-6
_INITIAL_TOLERANCE = 1e-5
_CORE_RUN = core._run
_CORE_EVALUATE = core._evaluate
_CORE_DECISION = core._decision


def _source_binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    core._read_regular_bound(
        path,
        expected_sha256=sha256,
        expected_bytes=byte_count,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get("LEWM_TRAJECTORY_H4_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_TRAJECTORY_H4_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError(
            "external trajectory-H4 wrapper byte binding is required"
        ) from error
    return {
        "trajectory_h4_wrapper": _source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "shared_runner": _source_binding(
            CORE_SOURCE, CORE_SOURCE_SHA256, CORE_SOURCE_BYTES
        ),
        "trajectory_h4_model": _source_binding(
            MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
        ),
        "dense_h4_model_dependency": _source_binding(
            DENSE_MODEL_SOURCE,
            DENSE_MODEL_SOURCE_SHA256,
            DENSE_MODEL_SOURCE_BYTES,
        ),
        "inherited_v1_model": _source_binding(
            BASE_MODEL_SOURCE, BASE_MODEL_SOURCE_SHA256, BASE_MODEL_SOURCE_BYTES
        ),
        "encoder_dependency": _source_binding(
            ENCODER_SOURCE, ENCODER_SOURCE_SHA256, ENCODER_SOURCE_BYTES
        ),
    }


def _install_bound_model_package_stubs() -> None:
    if any(name == "lewm" or name.startswith("lewm.") for name in sys.modules):
        raise core.ContractError("a lewm package was imported before bound model loading")

    def package(name: str, path: Path) -> ModuleType:
        module = ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        module.__spec__ = spec
        return module

    lewm_package = package("lewm", ROOT / "lewm")
    models_package = package("lewm.models", ROOT / "lewm/models")
    lewm_package.models = models_package
    sys.modules["lewm"] = lewm_package
    sys.modules["lewm.models"] = models_package


def _configure_core(source_bindings: Mapping[str, Mapping[str, Any]]) -> None:
    core.MODEL_MODULE = MODEL_MODULE
    core.MODEL_SOURCE = MODEL_SOURCE
    core.MODEL_SOURCE_SHA256 = MODEL_SOURCE_SHA256
    core.MODEL_SOURCE_BYTES = MODEL_SOURCE_BYTES
    core.OUTPUT_ROOT = OUTPUT_ROOT
    core.SCHEMA = SCHEMA
    core.PASS_DECISION = PASS_DECISION
    core.STOP_DECISION = STOP_DECISION
    core.PREDICTION_WEIGHT = 0.0
    core.VARIANCE_WEIGHT = 0.0
    core.ACTION_RANKING_WEIGHT = 0.0
    core.TRAIN_WRONG_ACTION_CONTRAST = False
    core.UPDATE_TARGET_EMA = False
    core.TARGET_DESCRIPTION = "fixed_accepted_N320_teacher_stop_gradient_no_ema"
    core.OBJECTIVE_DESCRIPTION = (
        "1.0*(0.5*joint_trajectory_energy_score+"
        "0.5*mean_marginal_horizon_energy_score)+"
        "1.0*three_frame_online_to_fixed_teacher_alignment; "
        "all_other_training_terms_absent"
    )
    core.ADDITIONAL_SCIENCE = {
        "teacher": "accepted_N320_fixed_for_entire_probe",
        "online_components": (
            "encoder+dense_history+action_path+four_atom_trajectory_predictor_"
            "jointly_trained"
        ),
        "support": "four_equal_mass_coherent_H1_to_H4_trajectory_atoms",
        "history": "complete_e0_p0_e1_p1_e2_dense_token_context",
        "future_action": "ordered_fixed_four_slot_zero_suffix_prefix_per_horizon",
        "prediction": "direct_nonrecursive_e2_relative_normalized_latent_atoms",
        "delta_head_initialization": "shared_final_linear_weight_and_bias_zero",
        "training_losses": [
            "proper_uniform_empirical_distribution_energy_score",
            "three_frame_online_to_fixed_teacher_alignment",
        ],
        "absent": [
            "learned_variance_or_scale",
            "learned_mixture_weights",
            "best_of_k_loss",
            "diversity_bonus",
            "control_ranking_loss",
            "navigation_labels_or_loss",
        ],
        "evaluation_only_controls": [
            "wrong_action",
            "hold_action",
            "persistence",
            "reordered_or_reset_history",
            "collapsed_spherical_centroid",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _lattice_distance(left: Any, right: Any, runtime: Any) -> Any:
    return runtime.torch.linalg.vector_norm(left - right, dim=(-2, -1)) / math.sqrt(
        float(left.shape[-2])
    )


def _marginal_energy(atoms: Any, target: Any, runtime: Any) -> Any:
    fit = _lattice_distance(atoms, target[:, None], runtime).mean(dim=1)
    pair = _lattice_distance(atoms[:, :, None], atoms[:, None, :], runtime).mean(
        dim=(1, 2)
    )
    return fit - 0.5 * pair


def _joint_energy(atoms: Any, target: Any, runtime: Any) -> Any:
    batch, atom_count, horizons, tokens, dim = atoms.shape
    flat_atoms = atoms.reshape(batch, atom_count, horizons * tokens, dim)
    flat_target = target.reshape(batch, horizons * tokens, dim)
    fit = _lattice_distance(flat_atoms, flat_target[:, None], runtime).mean(dim=1)
    pair = _lattice_distance(
        flat_atoms[:, :, None], flat_atoms[:, None, :], runtime
    ).mean(dim=(1, 2))
    return fit - 0.5 * pair


def _trajectory_atoms(output: Any) -> Any:
    value = core._extract_tensor(output, "trajectory_latents")
    if value.ndim != 5 or value.shape[1] != 4 or value.shape[2] != 4:
        raise core.ContractError("trajectory atom shape contract changed")
    return value


def _trajectory_evaluate(
    model: Any,
    rows: Sequence[Any],
    *,
    root_fd: int,
    runtime: Any,
    access: Counter[str],
    device: Any,
    update: int,
) -> dict[str, Any]:
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
            atoms = _trajectory_atoms(output)
            online = core._extract_tensor(output, "online_latents", "history_latents")
            belief = core._extract_tensor(output, "belief_latents", "belief")
            target = core._target_encode(model, future_rgb)
            repeated_current = history[:, 2:3].expand(
                -1, 4, -1, -1, -1
            ).contiguous()
            current_target = core._target_encode(model, repeated_current)
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
            reverse_atoms = _trajectory_atoms(reversed_output)
            reset_atoms = _trajectory_atoms(reset_output)

            real_score = _marginal_energy(atoms, target, runtime)
            wrong_score = _marginal_energy(wrong_atoms, target, runtime)
            hold_score = _marginal_energy(hold_atoms, target, runtime)
            reverse_score = _marginal_energy(reverse_atoms, target, runtime)
            reset_score = _marginal_energy(reset_atoms, target, runtime)
            persistence_score = _marginal_energy(
                persistence_atoms, target, runtime
            )
            scale = persistence_score.clamp_min(1e-4)
            centroid = torch.nn.functional.normalize(
                atoms.mean(dim=1), dim=-1, eps=1e-6
            )
            centroid_score = _lattice_distance(centroid, target, runtime)
            pair_spread = _lattice_distance(
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
            joint_real = _joint_energy(atoms, target, runtime)
            joint_persistence = _joint_energy(
                persistence_atoms, target, runtime
            )
            joint_scale = joint_persistence.clamp_min(1e-4)
            joint_ratio = joint_real / joint_scale
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
            joint_centroid = _lattice_distance(
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
            for row_index, row in enumerate(batch_rows):
                key = (row.family, row.scene_id)
                counts[key] += 1
                joint_sums[key] += float(joint_ratio[row_index].item())
                combined_sums[key] += float(combined_ratio[row_index].item())
                combined_value_sums[key] += float(combined_value[row_index].item())
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
    joint_family_values = []
    for family in core.FAMILIES:
        scene_values = [
            metrics["joint_trajectory_normalized_energy_score"]
            for (item_family, _scene), metrics in scene_metrics.items()
            if item_family == family
        ]
        value = sum(scene_values) / len(scene_values)
        family_metrics[family]["joint_trajectory_normalized_energy_score"] = value
        joint_family_values.append(value)
    aggregate["joint_trajectory_normalized_energy_score"] = (
        sum(joint_family_values) / len(joint_family_values)
    )
    for name in (
        "combined_normalized_energy_score",
        "combined_distribution_value_gap",
    ):
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
    target_rank, target_near_zero = core._effective_rank(
        torch.cat(target_features, dim=0), runtime
    )
    online_rank, online_near_zero = core._effective_rank(
        torch.cat(online_features, dim=0), runtime
    )
    finite_values = [
        value
        for name, vectors in aggregate.items()
        if name
        not in (
            "joint_trajectory_normalized_energy_score",
            "combined_normalized_energy_score",
            "combined_distribution_value_gap",
        )
        for value in vectors
    ] + [
        aggregate["joint_trajectory_normalized_energy_score"],
        aggregate["combined_normalized_energy_score"],
        aggregate["combined_distribution_value_gap"],
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


def _trajectory_decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    baseline = next(item for item in observations if item["update"] == 0)
    base_noncollapse = baseline["noncollapse"]
    target_rank = base_noncollapse["target_effective_rank_ratio"]
    target_near_zero = base_noncollapse["target_near_zero_variance_fraction"]
    target_rank_drift = max(
        abs(item["noncollapse"]["target_effective_rank_ratio"] - target_rank)
        for item in observations
    )
    target_near_zero_drift = max(
        abs(
            item["noncollapse"]["target_near_zero_variance_fraction"]
            - target_near_zero
        )
        for item in observations
    )
    initial_values = (
        baseline["aggregate"]["action_gap"]
        + baseline["aggregate"]["hold_gap"]
        + baseline["aggregate"]["persistence_gap"]
        + baseline["aggregate"]["history_gap"]
        + baseline["aggregate"]["distribution_value_gap"]
        + baseline["aggregate"]["normalized_pairwise_spread"]
        + [baseline["aggregate"]["combined_distribution_value_gap"]]
    )
    candidates = [
        item for item in observations if item["update"] > 0 and core._noncollapsed(item)
    ]
    selected = (
        min(
            candidates,
            key=lambda item: item["aggregate"][
                "combined_normalized_energy_score"
            ],
        )
        if candidates
        else None
    )
    gates: dict[str, bool] = {
        "completed_exact_cap": updates_completed == core.UPDATES,
        "all_observations_finite": all(
            bool(item["all_registered_values_finite"]) for item in observations
        ),
        "fixed_teacher_metric_geometry_unchanged": (
            target_rank_drift <= _GEOMETRY_TOLERANCE
            and target_near_zero_drift <= _GEOMETRY_TOLERANCE
        ),
        "fixed_teacher_rank_floor_all_observations": all(
            item["noncollapse"]["target_effective_rank_ratio"] >= 0.10
            and item["noncollapse"]["target_near_zero_variance_fraction"] <= 0.05
            for item in observations
        ),
        "update_zero_is_exact_persistence": (
            max(abs(value) for value in initial_values) <= _INITIAL_TOLERANCE
            and abs(
                baseline["aggregate"]["joint_trajectory_normalized_energy_score"]
                - 1.0
            )
            <= _INITIAL_TOLERANCE
            and abs(
                baseline["aggregate"]["combined_normalized_energy_score"] - 1.0
            )
            <= _INITIAL_TOLERANCE
            and max(
                abs(value - 1.0)
                for value in baseline["aggregate"][
                    "real_normalized_energy_score"
                ]
            )
            <= _INITIAL_TOLERANCE
        ),
        "eligible_noncollapsed_checkpoint_exists": selected is not None,
    }
    selected_gate_names = (
        "combined_energy_score_beats_persistence",
        "joint_trajectory_energy_score_beats_persistence",
        "h1_h3_energy_scores_below_persistence",
        "h4_energy_score_at_most_point90",
        "h4_persistence_gap_bootstrap_lower_positive",
        "persistence_positive_in_six_families",
        "no_family_persistence_gap_below_minus_point02",
        "combined_distribution_value_gap_at_least_point05",
        "combined_distribution_value_bootstrap_lower_positive",
        "combined_distribution_value_positive_in_six_families",
        "h4_normalized_pairwise_spread_at_least_point05",
        "h4_action_gap_at_least_point05",
        "h4_action_gap_bootstrap_lower_positive",
        "h1_h3_action_gaps_nonnegative",
        "action_positive_in_six_families",
        "no_family_action_gap_below_minus_point02",
        "h4_history_gap_at_least_point03",
        "h4_history_gap_bootstrap_lower_positive",
        "history_positive_in_six_families",
        "h4_hold_gap_positive",
    )
    diagnostics: dict[str, Any] = {
        "selected_update": None,
        "selected_presentations": None,
        "fixed_teacher_target_rank_drift": target_rank_drift,
        "fixed_teacher_target_near_zero_drift": target_near_zero_drift,
    }
    if selected is None:
        gates.update({name: False for name in selected_gate_names})
    else:
        aggregate = selected["aggregate"]
        real = aggregate["real_normalized_energy_score"]
        action = aggregate["action_gap"]
        hold = aggregate["hold_gap"]
        history = aggregate["history_gap"]
        spread = aggregate["normalized_pairwise_spread"]
        action_positive = sum(
            selected["family"][family]["action_gap"][3] > 0
            for family in core.FAMILIES
        )
        persistence_positive = sum(
            selected["family"][family]["persistence_gap"][3] > 0
            for family in core.FAMILIES
        )
        history_positive = sum(
            selected["family"][family]["history_gap"][3] > 0
            for family in core.FAMILIES
        )
        combined_distribution_positive = sum(
            selected["family"][family]["combined_distribution_value_gap"] > 0
            for family in core.FAMILIES
        )
        gates.update(
            {
                "combined_energy_score_beats_persistence": aggregate[
                    "combined_normalized_energy_score"
                ]
                < 1.0,
                "joint_trajectory_energy_score_beats_persistence": aggregate[
                    "joint_trajectory_normalized_energy_score"
                ]
                < 1.0,
                "h1_h3_energy_scores_below_persistence": all(
                    value < 1.0 for value in real[:3]
                ),
                "h4_energy_score_at_most_point90": real[3] <= 0.90,
                "h4_persistence_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["persistence_gap_h4"]
                > 0,
                "persistence_positive_in_six_families": persistence_positive >= 6,
                "no_family_persistence_gap_below_minus_point02": min(
                    selected["family"][family]["persistence_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "combined_distribution_value_gap_at_least_point05": aggregate[
                    "combined_distribution_value_gap"
                ]
                >= 0.05,
                "combined_distribution_value_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["combined_distribution_value_gap"]
                > 0,
                "combined_distribution_value_positive_in_six_families": (
                    combined_distribution_positive >= 6
                ),
                "h4_normalized_pairwise_spread_at_least_point05": spread[3]
                >= 0.05,
                "h4_action_gap_at_least_point05": action[3] >= 0.05,
                "h4_action_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["action_gap_h4"]
                > 0,
                "h1_h3_action_gaps_nonnegative": all(
                    value >= 0 for value in action[:3]
                ),
                "action_positive_in_six_families": action_positive >= 6,
                "no_family_action_gap_below_minus_point02": min(
                    selected["family"][family]["action_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "h4_history_gap_at_least_point03": history[3] >= 0.03,
                "h4_history_gap_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["history_gap_h4"]
                > 0,
                "history_positive_in_six_families": history_positive >= 6,
                "h4_hold_gap_positive": hold[3] > 0,
            }
        )
        diagnostics.update(
            {
                "selected_update": selected["update"],
                "selected_presentations": selected["presentations"],
                "joint_trajectory_normalized_energy_score": aggregate[
                    "joint_trajectory_normalized_energy_score"
                ],
                "combined_normalized_energy_score": aggregate[
                    "combined_normalized_energy_score"
                ],
                "action_positive_family_count": action_positive,
                "persistence_positive_family_count": persistence_positive,
                "history_positive_family_count": history_positive,
                "combined_distribution_value_positive_family_count": (
                    combined_distribution_positive
                ),
            }
        )
    failed_gates = sorted(name for name, value in gates.items() if not value)
    return {
        "decision": PASS_DECISION if not failed_gates else STOP_DECISION,
        "gates": gates,
        "failed_gates": failed_gates,
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded development trajectory-distribution JEPA "
            "substrate feasibility only; it grants no navigation, held-out, "
            "promotion, or deployment authority. A stop closes this exact "
            "finite-support/full-fixed-teacher-latent formulation."
        ),
    }


def _trajectory_run(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
    metrics, artifact, decision = _CORE_RUN(*args, **kwargs)
    if artifact.get("fresh_recurrent_and_predictor_initialization") is not True:
        raise core.ContractError("shared runner initialization receipt changed")
    trajectory_metrics = dict(metrics)
    trajectory_metrics["selection_rule"] = (
        "minimum validation 50/50 joint-plus-marginal normalized energy score among "
        "registered noncollapsed trained checkpoints"
    )
    trajectory_artifact = dict(artifact)
    del trajectory_artifact["fresh_recurrent_and_predictor_initialization"]
    trajectory_artifact[
        "fresh_dense_history_mode_embeddings_action_path_and_shared_delta_head_initialization"
    ] = True
    return trajectory_metrics, trajectory_artifact, decision


def _install_runtime_adapters() -> None:
    if core._evaluate not in (_CORE_EVALUATE, _trajectory_evaluate):
        raise core.ContractError("shared evaluator changed before adapter install")
    if core._decision not in (_CORE_DECISION, _trajectory_decision):
        raise core.ContractError("shared decision changed before adapter install")
    if core._run not in (_CORE_RUN, _trajectory_run):
        raise core.ContractError("shared runner changed before adapter install")
    core._evaluate = _trajectory_evaluate
    core._decision = _trajectory_decision
    core._run = _trajectory_run


def main(argv: Sequence[str] | None = None) -> int:
    if Path(core.__file__).resolve() != CORE_SOURCE:
        raise core.ContractError("shared runner imported from an unexpected path")
    source_bindings = _verify_source_closure()
    _install_bound_model_package_stubs()
    _configure_core(source_bindings)
    _install_runtime_adapters()
    return core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
