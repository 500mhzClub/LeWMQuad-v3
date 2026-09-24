#!/usr/bin/env python3
"""Run one fresh RGB whitened-delta predictive-state H4 JEPA probe."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import os
from importlib.machinery import ModuleSpec
from pathlib import Path
import random
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_go2_recurrent_h4_joint_jepa_v1 as core  # noqa: E402


CORE_SOURCE = ROOT / "scripts/run_go2_recurrent_h4_joint_jepa_v1.py"
CORE_SOURCE_SHA256 = "fc35d535e1c07b56c667474e6e10c5c7587fa01e627567148c022331983616fc"
CORE_SOURCE_BYTES = 70_301
MODEL_MODULE = "lewm.models.go2_rgb_whitened_delta_predictive_state_h4_jepa_v1"
MODEL_SOURCE = ROOT / (
    "lewm/models/go2_rgb_whitened_delta_predictive_state_h4_jepa_v1.py"
)
MODEL_SOURCE_SHA256 = "2d69d0bb0a31e2e95ffb29723a0bc559c425b10b381708a8cf985e36d19e1b5d"
MODEL_SOURCE_BYTES = 18_567
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
    ".generated/go2_rgb_whitened_delta_predictive_state_h4_jepa_v1/probe_v1"
)
SCHEMA = "lewm_go2_rgb_whitened_delta_predictive_state_h4_jepa_v1"
PASS_DECISION = "PASS_MAIN_POOL_RGB_WHITENED_DELTA_PREDICTIVE_STATE_H4_JEPA_V1"
STOP_DECISION = "STOP_MAIN_POOL_RGB_WHITENED_DELTA_PREDICTIVE_STATE_H4_JEPA_V1"
_GEOMETRY_TOLERANCE = 1e-6
_INITIAL_TOLERANCE = 1e-5
_CORE_RUN = core._run
_CORE_PREFLIGHT = core._preflight
_CORE_DECISION = core._decision


def _source_binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    core._read_regular_bound(path, expected_sha256=sha256, expected_bytes=byte_count)
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }


def _verify_source_closure() -> dict[str, dict[str, Any]]:
    wrapper_sha256 = os.environ.get("LEWM_WDPS_H4_WRAPPER_SHA256", "")
    wrapper_bytes_text = os.environ.get("LEWM_WDPS_H4_WRAPPER_BYTES", "")
    try:
        wrapper_bytes = int(wrapper_bytes_text)
    except ValueError as error:
        raise core.ContractError("external WDPS-H4 wrapper byte binding is required") from error
    return {
        "wdps_h4_wrapper": _source_binding(
            Path(__file__).resolve(), wrapper_sha256, wrapper_bytes
        ),
        "shared_runner": _source_binding(
            CORE_SOURCE, CORE_SOURCE_SHA256, CORE_SOURCE_BYTES
        ),
        "wdps_h4_model": _source_binding(
            MODEL_SOURCE, MODEL_SOURCE_SHA256, MODEL_SOURCE_BYTES
        ),
        "dense_h4_model_dependency": _source_binding(
            DENSE_MODEL_SOURCE, DENSE_MODEL_SOURCE_SHA256, DENSE_MODEL_SOURCE_BYTES
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
    core.TARGET_DESCRIPTION = (
        "jointly_learned_eight_dimensional_whitened_future_change_state_"
        "with_fixed_N320_target_and_history_teacher_no_ema"
    )
    core.OBJECTIVE_DESCRIPTION = (
        "25*predicted_to_target_state_similarity+25*mean_predicted_and_target_"
        "variance_floor+25*mean_predicted_and_target_zero_mean+1*mean_"
        "predicted_and_target_covariance_decorrelation+1*online_history_to_"
        "fixed_N320_alignment; all_controls_absent"
    )
    core.ADDITIONAL_SCIENCE = {
        "state": "four_horizons_times_eight_learned_future_change_dimensions",
        "target_state": (
            "shared_zero_preserving_spatial_attention_pool_of_fixed_N320_"
            "normalized_future_minus_e2_patch_deltas"
        ),
        "prediction": "dense_three_frame_history_plus_ordered_action_prefix",
        "joint_training": (
            "online_encoder+target_compressor+history+action+predictor_one_backward"
        ),
        "noncollapse": (
            "VICReg_variance_floor_zero_mean_and_covariance_decorrelation"
        ),
        "persistence": "exact_zero_compact_change_state",
        "fixed_teacher_role": "fixed_target_and_history_teacher",
        "absent": [
            "distribution_atoms_or_learned_variance",
            "best_of_k_or_diversity_loss",
            "wrong_action_history_persistence_or_hold_training_control",
            "contrastive_negatives_or_codebook",
            "reconstruction_or_navigation_loss",
            "pose_depth_flow_bev_warp_transport_or_geometry_target",
        ],
        "predecessor_predictor_checkpoint_tensor_open_count": 0,
    }
    core.AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
    core.EXECUTION_SOURCE_BINDINGS = {
        name: dict(binding) for name, binding in source_bindings.items()
    }


def _build_model(runtime: Any, n320_encoder: Mapping[str, Any]) -> Any:
    module = runtime.model_module
    config_cls = getattr(module, "WhitenedDeltaPredictiveStateConfig", None)
    model_cls = getattr(module, "WhitenedDeltaPredictiveStateH4JEPA", None)
    if config_cls is None or model_cls is None:
        raise core.ContractError("WDPS model module lacks its reviewed API")
    config = config_cls(
        image_size=core.IMAGE_SIZE,
        target_ema_momentum=core.EMA_MOMENTUM,
        variance_weight=0.0,
        action_vocabulary=core.PRIMITIVES,
    )
    model = model_cls(n320_encoder_state_dict=n320_encoder, config=config)
    for name in (
        "encode_history",
        "predict_from_belief",
        "encode_target_state",
        "hard_sync_target",
        "update_target",
    ):
        if not callable(getattr(model, name, None)):
            raise core.ContractError(f"WDPS model is missing {name}()")
    if tuple(model.action_vocabulary) != core.PRIMITIVES:
        raise core.ContractError("WDPS primitive vocabulary changed")
    return model


def _parameter_groups(model: Any) -> dict[str, list[Any]]:
    inventories = {
        "encoder": (model.encoder,),
        "history": (model.initial_belief,),
        "predictor": (
            model.action_embedding,
            model.future_cell,
            model.state_projector,
        ),
        "target_state": (model.target_state_compressor,),
    }
    groups = {
        name: [parameter for module in modules for parameter in module.parameters()]
        for name, modules in inventories.items()
    }
    if any(not values for values in groups.values()):
        raise core.ContractError("every WDPS optimizer group must be nonempty")
    flattened = [parameter for values in groups.values() for parameter in values]
    ids = [id(parameter) for parameter in flattened]
    if len(set(ids)) != len(ids) or any(not p.requires_grad for p in flattened):
        raise core.ContractError("WDPS optimizer groups overlap or contain frozen tensors")
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    if target_ids & set(ids) or any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    ):
        raise core.ContractError("fixed N320 target/history teacher entered the WDPS optimizer")
    all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if set(ids) != all_trainable:
        raise core.ContractError("WDPS groups do not cover trainable parameters exactly")
    return groups


def _forward(
    model: Any,
    history: Any,
    past: Any,
    future: Any,
    future_rgb: Any | None = None,
) -> Any:
    output = model(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
        future_rgb=future_rgb,
    )
    predicted = getattr(output, "predicted_state", None)
    if predicted is None or predicted.ndim != 3 or predicted.shape[1:] != (4, 8):
        raise core.ContractError("WDPS predicted state shape changed")
    if future_rgb is not None:
        target = getattr(output, "target_state", None)
        total = getattr(output, "total_loss", None)
        if target is None or target.shape != predicted.shape or total is None:
            raise core.ContractError("WDPS training output contract changed")
    return output


def _squared_error(predicted: Any, target: Any) -> Any:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise core.ContractError("WDPS state error shape changed")
    return (predicted - target).square().mean(dim=-1)


def _state_geometry(state: Any, runtime: Any) -> dict[str, list[float]]:
    torch = runtime.torch
    if state.ndim != 3 or state.shape[1:] != (4, 8):
        raise core.ContractError("WDPS state geometry shape changed")
    state_float = state.float()
    state_mean = state_float.mean(dim=0)
    centered = state_float - state_mean[None]
    covariance = torch.einsum("bhd,bhe->hde", centered, centered) / max(
        1, int(state.shape[0]) - 1
    )
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    numerator = eigenvalues.sum(dim=-1).square()
    denominator = eigenvalues.square().sum(dim=-1).clamp_min(1e-12)
    rank_ratio = numerator / denominator / float(state.shape[-1])
    std = state_float.var(dim=0, unbiased=False).add(1e-12).sqrt()
    return {
        "participation_rank_ratio": rank_ratio.detach().cpu().tolist(),
        "minimum_std": std.min(dim=-1).values.detach().cpu().tolist(),
        "maximum_std": std.max(dim=-1).values.detach().cpu().tolist(),
        "maximum_abs_mean": state_mean.abs().max(dim=-1).values.detach().cpu().tolist(),
    }


def _evaluate(
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
        "real_normalized_error",
        "action_gap",
        "hold_gap",
        "persistence_gap",
        "history_gap",
    )
    sums: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [0.0] * 4 for name in metric_names}
    )
    scale_sums: dict[tuple[str, str], list[float]] = defaultdict(
        lambda: [0.0] * 4
    )
    counts: Counter[tuple[str, str]] = Counter()
    predicted_states = []
    target_states = []
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
            output = _forward(model, history, past, future, future_rgb)
            predicted = output.predicted_state
            target = output.target_state
            belief = output.belief_latents
            wrong = model.predict_from_belief(
                belief, (future + 1) % len(core.PRIMITIVES)
            )
            hold = model.predict_from_belief(
                belief, torch.full_like(future, core.HOLD_ACTION)
            )
            _reverse_history, reverse_belief = model.encode_history(
                history[:, [1, 0, 2]], past[:, [1, 0]]
            )
            reverse = model.predict_from_belief(reverse_belief, future)
            _reset_history, reset_belief = model.encode_history(
                history[:, 2:3].expand(-1, 3, -1, -1, -1).contiguous(),
                torch.full_like(past, core.HOLD_ACTION),
            )
            reset = model.predict_from_belief(reset_belief, future)

            real_error = _squared_error(predicted, target)
            scale = target.square().mean(dim=-1)
            values = {
                "real_normalized_error": real_error,
                "action_gap": _squared_error(wrong, target) - real_error,
                "hold_gap": _squared_error(hold, target) - real_error,
                "persistence_gap": scale - real_error,
                "history_gap": (
                    torch.minimum(
                        _squared_error(reverse, target),
                        _squared_error(reset, target),
                    )
                    - real_error
                ),
            }
            for row_index, row in enumerate(batch_rows):
                key = (row.family, row.scene_id)
                counts[key] += 1
                scale_vector = scale[row_index].detach().cpu().tolist()
                for horizon in range(4):
                    scale_sums[key][horizon] += float(scale_vector[horizon])
                for name in metric_names:
                    vector = values[name][row_index].detach().cpu().tolist()
                    for horizon in range(4):
                        sums[key][name][horizon] += float(vector[horizon])
            predicted_states.append(predicted.detach().cpu())
            target_states.append(target.detach().cpu())
            online_features.append(
                core._pool_features(output.history_latents, time_index=2).detach().cpu()
            )
            fixed = output.fixed_history_latents
            if fixed is None:
                raise core.ContractError("WDPS evaluation lacks fixed history latents")
            target_features.append(
                core._pool_features(fixed, time_index=2).detach().cpu()
            )
            access["validation_sequence_presentation_count"] += len(batch_rows)

    scene_metrics = {
        key: {
            name: [
                item / max(scale_sums[key][horizon], 1e-20)
                for horizon, item in enumerate(vector)
            ]
            for name, vector in values.items()
        }
        for key, values in sums.items()
    }
    aggregate: dict[str, list[float]] = {}
    family_metrics: dict[str, dict[str, list[float]]] = {
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
                raise core.ContractError("WDPS validation macro lost a family")
            vector = [
                sum(item[h] for item in scene_vectors) / len(scene_vectors)
                for h in range(4)
            ]
            family_metrics[family][name] = vector
            family_vectors.append(vector)
        aggregate[name] = [
            sum(vector[h] for vector in family_vectors) / len(family_vectors)
            for h in range(4)
        ]

    lower_bounds: dict[str, float] = {}
    for offset, name in enumerate(
        ("action_gap", "persistence_gap", "history_gap", "hold_gap")
    ):
        by_family = {
            family: {
                scene: metrics[name][3]
                for (item_family, scene), metrics in scene_metrics.items()
                if item_family == family
            }
            for family in core.FAMILIES
        }
        lower_bounds[f"{name}_h4"] = core._bootstrap_lower(
            by_family, seed=core.SEED + update * 10 + offset
        )
    predicted_all = torch.cat(predicted_states, dim=0).float()
    target_all = torch.cat(target_states, dim=0).float()
    predicted_geometry = _state_geometry(predicted_all, runtime)
    target_geometry = _state_geometry(target_all, runtime)
    predicted_total_energy = predicted_all.square().mean(dim=(0, 2))
    target_total_energy = target_all.square().mean(dim=(0, 2))
    predicted_mean_energy = predicted_all.mean(dim=0).square().mean(dim=-1)
    target_mean_energy = target_all.mean(dim=0).square().mean(dim=-1)
    state_energy = {
        "predicted_rms": predicted_total_energy.sqrt().tolist(),
        "target_rms": target_total_energy.sqrt().tolist(),
        "predicted_mean_energy_fraction": (
            predicted_mean_energy / predicted_total_energy.clamp_min(1e-20)
        ).tolist(),
        "target_mean_energy_fraction": (
            target_mean_energy / target_total_energy.clamp_min(1e-20)
        ).tolist(),
        "target_mean_squared_energy": target_total_energy.tolist(),
        "near_zero_scene_denominator_count": [
            sum(scale_sums[key][horizon] <= 1e-8 for key in scale_sums)
            for horizon in range(4)
        ],
    }
    target_rank, target_near_zero = core._effective_rank(
        torch.cat(target_features, dim=0), runtime
    )
    online_rank, online_near_zero = core._effective_rank(
        torch.cat(online_features, dim=0), runtime
    )
    finite_values = [
        value for vectors in aggregate.values() for value in vectors
    ] + [
        *lower_bounds.values(),
        target_rank,
        target_near_zero,
        online_rank,
        online_near_zero,
        *(
            value
            for geometry in (predicted_geometry, target_geometry)
            for vector in geometry.values()
            for value in vector
        ),
        *(
            value
            for name, vector in state_energy.items()
            if name != "near_zero_scene_denominator_count"
            for value in vector
        ),
    ]
    result = {
        "update": update,
        "presentations": update * core.BATCH_SIZE,
        "validation_rows": len(rows),
        "aggregate": aggregate,
        "family": family_metrics,
        "bootstrap_lower_95": lower_bounds,
        "state_geometry": {
            "predicted": predicted_geometry,
            "target": target_geometry,
        },
        "state_energy": state_energy,
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


def _state_eligible(observation: Mapping[str, Any]) -> bool:
    if not core._noncollapsed(observation):
        return False
    for role in ("predicted", "target"):
        geometry = observation["state_geometry"][role]
        if not all(value >= 0.75 for value in geometry["participation_rank_ratio"]):
            return False
        if not all(value >= 0.50 for value in geometry["minimum_std"]):
            return False
        if not all(value <= 2.0 for value in geometry["maximum_std"]):
            return False
        if not all(value <= 0.25 for value in geometry["maximum_abs_mean"]):
            return False
    energy = observation["state_energy"]
    if any(energy["near_zero_scene_denominator_count"]):
        return False
    if not all(value <= 3.0 for value in energy["predicted_rms"]):
        return False
    if not all(value <= 3.0 for value in energy["target_rms"]):
        return False
    if not all(value <= 0.25 for value in energy["predicted_mean_energy_fraction"]):
        return False
    if not all(value <= 0.25 for value in energy["target_mean_energy_fraction"]):
        return False
    return True


def _decision(
    observations: Sequence[Mapping[str, Any]],
    updates_completed: int,
) -> dict[str, Any]:
    baseline = next(item for item in observations if item["update"] == 0)
    base_target_rank = baseline["noncollapse"]["target_effective_rank_ratio"]
    base_target_zero = baseline["noncollapse"]["target_near_zero_variance_fraction"]
    target_rank_drift = max(
        abs(item["noncollapse"]["target_effective_rank_ratio"] - base_target_rank)
        for item in observations
    )
    target_zero_drift = max(
        abs(
            item["noncollapse"]["target_near_zero_variance_fraction"]
            - base_target_zero
        )
        for item in observations
    )
    initial_values = (
        baseline["aggregate"]["action_gap"]
        + baseline["aggregate"]["hold_gap"]
        + baseline["aggregate"]["persistence_gap"]
        + baseline["aggregate"]["history_gap"]
    )
    candidates = [
        item for item in observations if item["update"] > 0 and _state_eligible(item)
    ]
    selected = (
        min(
            candidates,
            key=lambda item: sum(item["aggregate"]["real_normalized_error"])
            / 4.0,
        )
        if candidates
        else None
    )
    gates: dict[str, bool] = {
        "completed_exact_cap": updates_completed == core.UPDATES,
        "all_observations_finite": all(
            item["all_registered_values_finite"] for item in observations
        ),
        "fixed_teacher_metric_geometry_unchanged": (
            target_rank_drift <= _GEOMETRY_TOLERANCE
            and target_zero_drift <= _GEOMETRY_TOLERANCE
        ),
        "update_zero_is_exact_persistence": (
            max(abs(value) for value in initial_values) <= _INITIAL_TOLERANCE
            and max(
                abs(value - 1.0)
                for value in baseline["aggregate"]["real_normalized_error"]
            )
            <= _INITIAL_TOLERANCE
        ),
        "eligible_noncollapsed_compact_state_exists": selected is not None,
    }
    selected_gate_names = (
        "all_horizon_errors_below_persistence",
        "mean_error_at_most_point90",
        "h4_error_at_most_point90",
        "h4_persistence_bootstrap_lower_positive",
        "persistence_positive_in_six_families",
        "no_family_persistence_below_minus_point02",
        "h4_action_gap_at_least_point03",
        "h4_action_bootstrap_lower_positive",
        "h1_h3_action_gaps_nonnegative",
        "action_positive_in_six_families",
        "no_family_action_below_minus_point02",
        "h4_history_gap_at_least_point03",
        "h4_history_bootstrap_lower_positive",
        "history_positive_in_six_families",
        "h4_hold_gap_positive",
    )
    diagnostics: dict[str, Any] = {
        "selected_update": None,
        "selected_presentations": None,
        "fixed_teacher_target_rank_drift": target_rank_drift,
        "fixed_teacher_target_near_zero_drift": target_zero_drift,
    }
    if selected is None:
        gates.update({name: False for name in selected_gate_names})
    else:
        real = selected["aggregate"]["real_normalized_error"]
        action = selected["aggregate"]["action_gap"]
        hold = selected["aggregate"]["hold_gap"]
        history = selected["aggregate"]["history_gap"]
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
        gates.update(
            {
                "all_horizon_errors_below_persistence": all(v < 1.0 for v in real),
                "mean_error_at_most_point90": sum(real) / 4.0 <= 0.90,
                "h4_error_at_most_point90": real[3] <= 0.90,
                "h4_persistence_bootstrap_lower_positive": selected[
                    "bootstrap_lower_95"
                ]["persistence_gap_h4"]
                > 0,
                "persistence_positive_in_six_families": persistence_positive >= 6,
                "no_family_persistence_below_minus_point02": min(
                    selected["family"][family]["persistence_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "h4_action_gap_at_least_point03": action[3] >= 0.03,
                "h4_action_bootstrap_lower_positive": selected["bootstrap_lower_95"][
                    "action_gap_h4"
                ]
                > 0,
                "h1_h3_action_gaps_nonnegative": all(v >= 0 for v in action[:3]),
                "action_positive_in_six_families": action_positive >= 6,
                "no_family_action_below_minus_point02": min(
                    selected["family"][family]["action_gap"][3]
                    for family in core.FAMILIES
                )
                >= -0.02,
                "h4_history_gap_at_least_point03": history[3] >= 0.03,
                "h4_history_bootstrap_lower_positive": selected[
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
                "selected_mean_normalized_error": sum(real) / 4.0,
                "action_positive_family_count": action_positive,
                "persistence_positive_family_count": persistence_positive,
                "history_positive_family_count": history_positive,
            }
        )
    failed = sorted(name for name, value in gates.items() if not value)
    return {
        "decision": PASS_DECISION if not failed else STOP_DECISION,
        "gates": gates,
        "failed_gates": failed,
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded compact predictive-state JEPA feasibility "
            "only; it grants no checkpoint, navigation, held-out, promotion, or "
            "deployment authority. A stop closes this exact WDPS-D8 formulation."
        ),
    }


def _preflight(args: Any, census_binding: Mapping[str, Any]) -> int:
    access: Counter[str] = Counter()
    access["census_receipt_open_count"] = 1
    train_rows, val_rows, train_binding, val_binding = core._load_index_contract(
        args, access=access
    )
    runtime = core._late_runtime(args.model_sha256, args.model_bytes, access)
    n320_encoder, n320_binding = core._load_n320_encoder(runtime, access)
    model = _build_model(runtime, n320_encoder)
    groups = _parameter_groups(model)
    print(
        json.dumps(
            {
                "decision": "PREFLIGHT_PASS_NO_OUTPUT_RESERVED_NO_RGB_OPENED",
                "census": dict(census_binding),
                "train": train_binding,
                "val": val_binding,
                "n320_encoder_initialization": n320_binding,
                "row_counts": {"train": len(train_rows), "val": len(val_rows)},
                "trainable_parameter_counts": {
                    name: sum(p.numel() for p in parameters)
                    for name, parameters in groups.items()
                },
                "access": dict(sorted(access.items())),
                "rgb_open_count": 0,
                "output_reservation_count": 0,
                "training_update_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run(
    args: Any,
    *,
    output_fd: int,
    access: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for name in (
        "target_ema_update_count",
        "optimizer_updates_with_fixed_target_count",
        "wrong_action_counterfactual_sequence_count",
        "auxiliary_training_control_sequence_count",
    ):
        access[name] += 0
    train_rows, val_rows, train_binding, val_binding = core._load_index_contract(
        args, access=access
    )
    runtime = core._late_runtime(args.model_sha256, args.model_bytes, access)
    torch = runtime.torch
    if args.device != "cuda" or not torch.cuda.is_available():
        raise core.ContractError("the capped WDPS probe requires CUDA/ROCm")
    device = torch.device("cuda")
    random.seed(core.SEED)
    torch.manual_seed(core.SEED)
    torch.cuda.manual_seed_all(core.SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    n320_encoder, n320_binding = core._load_n320_encoder(runtime, access)
    model = _build_model(runtime, n320_encoder).to(device)
    del n320_encoder
    fixed_target_initial = core._state_sha256(model.target_encoder, runtime)
    groups = _parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["encoder"], "lr": 1e-4, "name": "encoder"},
            {"params": groups["history"], "lr": 3e-4, "name": "history"},
            {"params": groups["predictor"], "lr": 3e-4, "name": "predictor"},
            {
                "params": groups["target_state"],
                "lr": 3e-4,
                "name": "target_state",
            },
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    rgb_root_fd = core._open_absolute_directory(core.RGB_ROOT)
    observations: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    updates_completed = 0
    presentations_completed = 0
    loss_sums: defaultdict[str, float] = defaultdict(float)
    last_losses: dict[str, float] | None = None
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        observations.append(
            _evaluate(
                model,
                val_rows,
                root_fd=rgb_root_fd,
                runtime=runtime,
                access=access,
                device=device,
                update=0,
            )
        )
        for update in range(1, core.UPDATES + 1):
            start = (update - 1) * core.BATCH_SIZE
            batch_rows = train_rows[start : start + core.BATCH_SIZE]
            if len(batch_rows) != core.BATCH_SIZE:
                raise core.ContractError("WDPS train schedule exhausted before cap")
            rgb, actions = core._load_batch(
                batch_rows,
                root_fd=rgb_root_fd,
                runtime=runtime,
                access=access,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = _forward(
                model,
                rgb[:, :3],
                actions[:, :2],
                actions[:, 2:],
                rgb[:, 3:],
            )
            loss = output.total_loss
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise core.ContractError("nonfinite WDPS joint objective")
            loss.backward()
            for values in groups.values():
                torch.nn.utils.clip_grad_norm_(values, max_norm=1.0)
            optimizer.step()
            access["optimizer_updates_with_fixed_target_count"] += 1
            updates_completed = update
            presentations_completed = update * core.BATCH_SIZE
            access["optimizer_update_count"] = update
            access["train_sequence_presentation_count"] = presentations_completed
            last_losses = {
                "state_similarity": float(output.state_prediction_loss.detach().item()),
                "predicted_variance": float(
                    output.predicted_variance_loss.detach().item()
                ),
                "target_variance": float(output.target_variance_loss.detach().item()),
                "predicted_mean": float(output.predicted_mean_loss.detach().item()),
                "target_mean": float(output.target_mean_loss.detach().item()),
                "predicted_covariance": float(
                    output.predicted_covariance_loss.detach().item()
                ),
                "target_covariance": float(
                    output.target_covariance_loss.detach().item()
                ),
                "history_teacher_alignment": float(
                    output.history_teacher_alignment_loss.detach().item()
                ),
                "total": float(loss.detach().item()),
            }
            if not all(math.isfinite(value) for value in last_losses.values()):
                raise core.ContractError("nonfinite detached WDPS loss receipt")
            for name, value in last_losses.items():
                loss_sums[name] += value
            if update in core.OBSERVATION_UPDATES[1:]:
                observations.append(
                    _evaluate(
                        model,
                        val_rows,
                        root_fd=rgb_root_fd,
                        runtime=runtime,
                        access=access,
                        device=device,
                        update=update,
                    )
                )
                checkpoints[str(update)] = core._save_checkpoint(
                    output_fd,
                    model=model,
                    runtime=runtime,
                    update=update,
                    presentations=presentations_completed,
                )
            torch.cuda.synchronize()
            if time.monotonic() - started > core.MAX_GPU_SECONDS:
                raise core.ContractError("90-minute active GPU cap exceeded")
    finally:
        os.close(rgb_root_fd)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if updates_completed != core.UPDATES or presentations_completed != core.PRESENTATIONS:
        raise core.ContractError("WDPS exact cap did not complete")
    fixed_target_final = core._state_sha256(model.target_encoder, runtime)
    if fixed_target_final != fixed_target_initial:
        raise core.ContractError("fixed N320 target/history teacher changed")
    if int(model.ema_update_count.detach().cpu().item()) != 0:
        raise core.ContractError("fixed N320 target/history teacher recorded EMA")
    decision = _decision(observations, updates_completed)
    metrics = {
        "schema": f"{SCHEMA}_metrics_v1",
        "observations": observations,
        "training_losses": {
            "mean_over_completed_updates": {
                name: value / updates_completed for name, value in loss_sums.items()
            },
            "last_completed_update": last_losses,
            "objective": core.OBJECTIVE_DESCRIPTION,
        },
        "selection_rule": (
            "minimum mean H1-H4 normalized compact-state error among registered "
            "trained checkpoints satisfying encoder and compact-state geometry"
        ),
    }
    artifact = {
        "schema": f"{SCHEMA}_artifact_v1",
        "checkpoints": checkpoints,
        "updates_completed": updates_completed,
        "presentations_completed": presentations_completed,
        "gpu_active_seconds": elapsed,
        "input_bindings": {
            "train": train_binding,
            "val": val_binding,
            "n320_encoder_initialization": n320_binding,
        },
        "execution_source_bindings": {
            name: dict(binding)
            for name, binding in sorted(core.EXECUTION_SOURCE_BINDINGS.items())
        },
        "fresh_history_action_target_compressor_and_state_predictor_initialization": True,
        "online_encoder_initialized_from_accepted_n320_and_jointly_trained": True,
        "fixed_target_and_history_teacher_identity": {
            "initial_state_sha256": fixed_target_initial,
            "final_state_sha256": fixed_target_final,
            "ema_update_count": 0,
            "unchanged": True,
        },
        "target_ema_update_enabled": False,
        "wrong_action_training_contrast_enabled": False,
        "n320_encoder_initialization_checkpoint_open_count": 1,
        "retry_or_resume_checkpoint_input_open_count": 0,
        "retry_or_resume": False,
    }
    return metrics, artifact, decision


def _install_runtime_adapters() -> None:
    if core._preflight not in (_CORE_PREFLIGHT, _preflight):
        raise core.ContractError("shared preflight changed before WDPS adapter")
    if core._run not in (_CORE_RUN, _run):
        raise core.ContractError("shared runner changed before WDPS adapter")
    if core._decision not in (_CORE_DECISION, _decision):
        raise core.ContractError("shared decision changed before WDPS adapter")
    core._preflight = _preflight
    core._run = _run
    core._decision = _decision


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
