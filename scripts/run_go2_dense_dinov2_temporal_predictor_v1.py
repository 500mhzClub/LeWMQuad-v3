#!/usr/bin/env python3
"""Train and evaluate the single dense-DINO temporal-predictor experiment.

The runner opens only the exact SHA-bound H6 V2 train/validation indices and
the RGB leaves named by selected rows.  DINOv2 stays frozen; only the temporal
predictor is optimized.  There is deliberately no feature cache or model grid.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.actions import ACTIVE_BLOCK_DIM, encode_active_block  # noqa: E402
from lewm.benchmarks import go2_matched_branch_successor_screen_v1 as dino_data  # noqa: E402
from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import PRIMITIVES  # noqa: E402
from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6  # noqa: E402


SCHEMA = "lewm_go2_dense_dinov2_temporal_predictor_v1"
TRACE_SCHEMA = f"{SCHEMA}_trace"
CHECKPOINT_SCHEMA = f"{SCHEMA}_checkpoint"
TERMINAL_SCHEMA = f"{SCHEMA}_terminal"
DEFAULT_RGB_ROOT = REPO_ROOT / ".generated/datagen_full/render_textured_v03"
DEFAULT_PRIMITIVE_REGISTRY = REPO_ROOT / "config/go2_primitive_registry.yaml"
DEFAULT_DINO_REPOSITORY = Path(
    "/home/andrewknowles/.cache/"
    "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
DEFAULT_DINO_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/torch/hub/checkpoints/"
    "dinov2_vits14_pretrain.pth"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / ".generated/dev/go2_dense_dinov2_temporal_predictor_v1/attempt_v1"
)
DINO_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DINO_CHECKPOINT_BYTE_COUNT = 88_283_115
TOKEN_SHAPE = (256, 384)
CONTEXT_FRAMES = 3
HISTORY_ACTIONS = 2
ROLLOUT_HORIZON = 2
DEFAULT_SEED = 2_026_080_405
DEFAULT_BOOTSTRAP_SEED = 2_026_080_406
DEFAULT_BOOTSTRAP_DRAWS = 2_000
DEFAULT_TRACE_UPDATES = (0, 250, 500, 1_000)
CONTROL_NAMES = (
    "correct",
    "persistence",
    "wrong_a2",
    "wrong_a3",
    "reset_history",
    "current_only",
    "reversed_history",
)
COLLAPSE_STD_FLOOR = 1.0e-4


class DenseDINORunnerError(RuntimeError):
    """Raised when the fixed experiment or an input binding is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _is_sealed_path(path: Path) -> bool:
    return path.name == "sealed_test.json" or any(
        part == "sealed" or part.startswith("sealed_") for part in path.parts
    )


def _safe_path(path: Path, *, must_exist: bool, label: str) -> Path:
    if _is_sealed_path(path):
        raise DenseDINORunnerError(f"refusing sealed {label} path")
    try:
        resolved = path.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise DenseDINORunnerError(f"{label} does not exist: {path}") from exc
    if _is_sealed_path(resolved):
        raise DenseDINORunnerError(f"refusing sealed {label} path")
    return resolved


def file_binding(path: Path) -> dict[str, Any]:
    selected = _safe_path(path, must_exist=True, label="file binding")
    if not selected.is_file() or selected.is_symlink():
        raise DenseDINORunnerError(f"bound input is not a regular file: {selected}")
    raw = selected.read_bytes()
    return {
        "path": str(selected),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_trace_updates(raw: str, *, updates: int) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise DenseDINORunnerError("trace updates must be comma-separated integers") from exc
    if (
        not values
        or tuple(sorted(set(values))) != values
        or values[0] != 0
        or values[-1] != updates
        or any(value < 0 or value > updates for value in values)
    ):
        raise DenseDINORunnerError(
            "trace updates must be unique, sorted, start at zero, and end at --updates"
        )
    return values


def primitive_action_table_from_document(
    document: Mapping[str, Any],
    *,
    primitive_names: Sequence[str] = PRIMITIVES,
) -> torch.Tensor:
    """Convert canonical five-tick velocity primitives to channel-major 15-D."""
    if document.get("block_size") != 5 or document.get("command_order") != [
        "vx_body_mps",
        "vy_body_mps",
        "yaw_rate_radps",
    ]:
        raise DenseDINORunnerError("primitive registry block contract changed")
    primitives = document.get("primitives")
    if not isinstance(primitives, Mapping):
        raise DenseDINORunnerError("primitive registry has no primitive mapping")
    vectors: list[np.ndarray] = []
    for name in primitive_names:
        spec = primitives.get(name)
        if not isinstance(spec, Mapping) or spec.get("type") != "velocity_block":
            raise DenseDINORunnerError(f"primitive {name!r} is not a velocity block")
        command = spec.get("command")
        if not isinstance(command, Mapping):
            raise DenseDINORunnerError(f"primitive {name!r} has no command")
        vx = float(command.get("vx_body_mps", 0.0))
        vy = float(command.get("vy_body_mps", 0.0))
        yaw = float(command.get("yaw_rate_radps", 0.0))
        if not all(math.isfinite(value) for value in (vx, vy, yaw)):
            raise DenseDINORunnerError(f"primitive {name!r} is nonfinite")
        vectors.append(
            encode_active_block([vx] * 5, [vy] * 5, [yaw] * 5)
        )
    result = torch.from_numpy(np.stack(vectors)).to(torch.float32)
    if result.shape != (len(primitive_names), ACTIVE_BLOCK_DIM):
        raise DenseDINORunnerError("canonical primitive action table shape changed")
    return result


def load_primitive_action_table(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    selected = _safe_path(path, must_exist=True, label="primitive registry")
    try:
        document = yaml.safe_load(selected.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DenseDINORunnerError("cannot decode primitive registry") from exc
    if not isinstance(document, Mapping):
        raise DenseDINORunnerError("primitive registry must be an object")
    return primitive_action_table_from_document(document), file_binding(selected)


def deterministic_batch_indices(
    *, row_count: int, batch_size: int, update_index: int, seed: int
) -> np.ndarray:
    """Return a resumable epoch-permutation batch for a zero-based update."""
    if row_count <= 0 or batch_size <= 0 or update_index < 0:
        raise DenseDINORunnerError("sampler dimensions must be positive")
    start = update_index * batch_size
    remaining = batch_size
    selected: list[int] = []
    while remaining:
        epoch, offset = divmod(start, row_count)
        permutation = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(epoch)])
        ).permutation(row_count)
        take = min(remaining, row_count - offset)
        selected.extend(int(value) for value in permutation[offset : offset + take])
        start += take
        remaining -= take
    return np.asarray(selected, dtype=np.int64)


def select_wrong_action_donor_indices(
    action_sequences: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Select a deterministic cyclic donor maximizing H1/H2 action contrast."""
    if len(action_sequences) < 2:
        raise DenseDINORunnerError("wrong-action control requires at least two rows")
    futures: list[tuple[int, int]] = []
    for index, sequence in enumerate(action_sequences):
        if len(sequence) < 4 or any(type(value) is not int for value in sequence):
            raise DenseDINORunnerError(f"action sequence {index} is malformed")
        futures.append((sequence[2], sequence[3]))
    donors: list[int] = []
    count = len(futures)
    for row_index, target in enumerate(futures):
        candidates = []
        for donor_index, donor in enumerate(futures):
            if donor_index == row_index:
                continue
            contrast = int(donor[0] != target[0]) + int(donor[1] != target[1])
            offset = (donor_index - row_index) % count
            candidates.append((-contrast, offset, donor_index))
        donors.append(min(candidates)[2])
    return tuple(donors)


def build_temporal_controls(
    tokens: torch.Tensor,
    actions: torch.Tensor,
    wrong_future_actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the exact factual and intervention inputs from frames 0:5."""
    if tokens.ndim != 4 or tuple(tokens.shape[1:]) != (5, *TOKEN_SHAPE):
        raise DenseDINORunnerError("tokens must have shape [B,5,256,384]")
    if actions.shape != (tokens.shape[0], 4, ACTIVE_BLOCK_DIM):
        raise DenseDINORunnerError("actions must have shape [B,4,15]")
    if wrong_future_actions.shape != (tokens.shape[0], 2, ACTIVE_BLOCK_DIM):
        raise DenseDINORunnerError("wrong future actions must have shape [B,2,15]")
    context = tokens[:, :3]
    history = actions[:, :2]
    future = actions[:, 2:4]
    current = tokens[:, 2]
    wrong_a2 = future.clone()
    wrong_a2[:, 0] = wrong_future_actions[:, 0]
    wrong_a3 = future.clone()
    wrong_a3[:, 1] = wrong_future_actions[:, 1]
    return {
        "context": context,
        "history_actions": history,
        "future_actions": future,
        "targets": tokens[:, 3:5].detach(),
        "persistence": current[:, None].expand(-1, 2, -1, -1),
        "wrong_a2_future_actions": wrong_a2,
        "wrong_a3_future_actions": wrong_a3,
        "reset_history_actions": torch.zeros_like(history),
        "current_only_context": current[:, None].expand(-1, 3, -1, -1),
        "reversed_context": torch.flip(context, dims=(1,)),
        "reversed_history_actions": torch.flip(history, dims=(1,)),
    }


def free_running_rollout(
    model: torch.nn.Module,
    context: torch.Tensor,
    history_actions: torch.Tensor,
    action_sequence: torch.Tensor,
) -> torch.Tensor:
    """Lock the exact H1/H2 causal alignment without teacher-forcing ``z3``.

    H1 uses ``z0,z1,z2 + a0,a1,a2 -> z3_hat``.  H2 then shifts the predicted
    state into ``z1,z2,z3_hat + a1,a2,a3 -> z4_hat``.  No target token is an
    input to either call.
    """
    if context.ndim != 4 or tuple(context.shape[1:]) != (3, *TOKEN_SHAPE):
        raise DenseDINORunnerError("rollout context must have shape [B,3,256,384]")
    if history_actions.shape != (context.shape[0], 2, ACTIVE_BLOCK_DIM):
        raise DenseDINORunnerError("rollout history must have shape [B,2,15]")
    if action_sequence.shape != (context.shape[0], 2, ACTIVE_BLOCK_DIM):
        raise DenseDINORunnerError("rollout actions must have shape [B,2,15]")
    z3_hat = model(context, history_actions, action_sequence[:, 0])
    if z3_hat.shape != (context.shape[0], *TOKEN_SHAPE):
        raise DenseDINORunnerError("predictor H1 output shape changed")
    shifted_context = torch.cat((context[:, 1:], z3_hat[:, None]), dim=1)
    shifted_history = torch.cat(
        (history_actions[:, 1:], action_sequence[:, 0, None]), dim=1
    )
    z4_hat = model(shifted_context, shifted_history, action_sequence[:, 1])
    if z4_hat.shape != z3_hat.shape:
        raise DenseDINORunnerError("predictor H2 output shape changed")
    result = torch.stack((z3_hat, z4_hat), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise DenseDINORunnerError("free-running rollout became nonfinite")
    return result


def cosine_error_per_row_horizon(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise DenseDINORunnerError("prediction and target token shapes differ")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise DenseDINORunnerError("token metric received a nonfinite value")
    result = 1.0 - F.cosine_similarity(
        prediction.to(torch.float32), target.to(torch.float32), dim=-1, eps=1.0e-8
    ).mean(dim=-1)
    if result.shape != prediction.shape[:2] or not bool(torch.isfinite(result).all()):
        raise DenseDINORunnerError("cosine-error metric is invalid")
    return result


def _mean_ci(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise DenseDINORunnerError("bootstrap vector is invalid")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(draws, vector.size))
    replicates = vector[indices].mean(axis=1)
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    return {
        "point": float(vector.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _scene_values(values: np.ndarray, scene_ids: Sequence[str]) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene_id, value in zip(scene_ids, values.tolist(), strict=True):
        grouped[scene_id].append(float(value))
    return np.asarray(
        [np.mean(grouped[scene]) for scene in sorted(grouped)], dtype=np.float64
    )


def _scene_means_by_family(
    values: np.ndarray,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    vector = np.asarray(values, dtype=np.float64)
    if (
        vector.ndim != 1
        or vector.size != len(scene_ids)
        or vector.size != len(family_ids)
        or not np.isfinite(vector).all()
    ):
        raise DenseDINORunnerError("family-scene bootstrap inputs are invalid")
    scene_family: dict[str, str] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene_id, family_id, value in zip(
        scene_ids, family_ids, vector.tolist(), strict=True
    ):
        if not isinstance(scene_id, str) or not isinstance(family_id, str):
            raise DenseDINORunnerError("scene and family identifiers must be strings")
        previous = scene_family.setdefault(scene_id, family_id)
        if previous != family_id:
            raise DenseDINORunnerError("one scene appeared in multiple families")
        grouped[scene_id].append(float(value))
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for scene_id in sorted(grouped):
        result[scene_family[scene_id]][scene_id] = float(np.mean(grouped[scene_id]))
    return {family: result[family] for family in sorted(result)}


def _family_equal_mean_ci(
    values: np.ndarray,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Resample whole scenes within each family, then weight families equally."""
    grouped = _scene_means_by_family(values, scene_ids, family_ids)
    if not grouped or draws <= 0:
        raise DenseDINORunnerError("family-scene bootstrap panel is empty")
    rng = np.random.default_rng(seed)
    family_replicates: list[np.ndarray] = []
    family_points: list[float] = []
    for family in sorted(grouped):
        vector = np.asarray(list(grouped[family].values()), dtype=np.float64)
        indices = rng.integers(0, vector.size, size=(draws, vector.size))
        family_replicates.append(vector[indices].mean(axis=1))
        family_points.append(float(vector.mean()))
    replicates = np.stack(family_replicates, axis=1).mean(axis=1)
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    return {
        "point": float(np.mean(family_points)),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def _paired_statistic(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    statistic: str,
    draws: int,
    seed: int,
) -> dict[str, float]:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise DenseDINORunnerError("paired statistic vectors changed")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, left.size, size=(draws, left.size))
    if statistic == "gap":
        point = float((left - right).mean())
        replicates = (left[indices] - right[indices]).mean(axis=1)
    elif statistic == "ratio":
        denominator_mean = float(right.mean())
        if denominator_mean <= 1.0e-12:
            raise DenseDINORunnerError("persistence denominator is zero")
        point = float(left.mean() / denominator_mean)
        sampled_denominator = right[indices].mean(axis=1)
        if bool((sampled_denominator <= 1.0e-12).any()):
            raise DenseDINORunnerError("bootstrap persistence denominator is zero")
        replicates = left[indices].mean(axis=1) / sampled_denominator
    else:
        raise DenseDINORunnerError("unknown paired statistic")
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    return {"point": point, "lower_95": float(lower), "upper_95": float(upper)}


def _family_equal_paired_statistic(
    numerator: np.ndarray,
    denominator: np.ndarray,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    statistic: str,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap paired whole-scene statistics with equal family weighting."""
    left = _scene_means_by_family(numerator, scene_ids, family_ids)
    right = _scene_means_by_family(denominator, scene_ids, family_ids)
    if left.keys() != right.keys() or not left or draws <= 0:
        raise DenseDINORunnerError("paired family-scene panel changed")
    rng = np.random.default_rng(seed)
    left_replicates = np.zeros(draws, dtype=np.float64)
    right_replicates = np.zeros(draws, dtype=np.float64)
    left_points: list[float] = []
    right_points: list[float] = []
    for family in sorted(left):
        if left[family].keys() != right[family].keys():
            raise DenseDINORunnerError("paired family scenes are not aligned")
        ordered_scenes = sorted(left[family])
        left_vector = np.asarray(
            [left[family][scene] for scene in ordered_scenes], dtype=np.float64
        )
        right_vector = np.asarray(
            [right[family][scene] for scene in ordered_scenes], dtype=np.float64
        )
        indices = rng.integers(
            0, left_vector.size, size=(draws, left_vector.size)
        )
        left_replicates += left_vector[indices].mean(axis=1)
        right_replicates += right_vector[indices].mean(axis=1)
        left_points.append(float(left_vector.mean()))
        right_points.append(float(right_vector.mean()))
    family_count = len(left)
    left_replicates /= family_count
    right_replicates /= family_count
    left_point = float(np.mean(left_points))
    right_point = float(np.mean(right_points))
    if statistic == "gap":
        point = left_point - right_point
        replicates = left_replicates - right_replicates
    elif statistic == "ratio":
        if right_point <= 1.0e-12 or bool((right_replicates <= 1.0e-12).any()):
            raise DenseDINORunnerError("family persistence denominator is zero")
        point = left_point / right_point
        replicates = left_replicates / right_replicates
    else:
        raise DenseDINORunnerError("unknown paired statistic")
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    return {"point": point, "lower_95": float(lower), "upper_95": float(upper)}


def summarize_evaluation_metrics(
    row_metrics: Mapping[str, np.ndarray],
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if set(row_metrics) != set(CONTROL_NAMES) or bootstrap_draws <= 0:
        raise DenseDINORunnerError("evaluation metric panel changed")
    row_count = len(scene_ids)
    if len(family_ids) != row_count:
        raise DenseDINORunnerError("evaluation family panel length changed")
    arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in row_metrics.items()
    }
    if any(
        values.shape != (row_count, 2) or not np.isfinite(values).all()
        for values in arrays.values()
    ):
        raise DenseDINORunnerError("evaluation row metrics are invalid")
    summaries: dict[str, Any] = {}
    for name in CONTROL_NAMES:
        summaries[name] = {}
        for horizon in range(2):
            row_values = arrays[name][:, horizon]
            scene_values = _scene_values(row_values, scene_ids)
            summaries[name][f"h{horizon + 1}"] = {
                "row": _mean_ci(
                    row_values, draws=bootstrap_draws, seed=bootstrap_seed
                ),
                "scene": _mean_ci(
                    scene_values, draws=bootstrap_draws, seed=bootstrap_seed
                ),
                "family_equal_scene": _family_equal_mean_ci(
                    row_values,
                    scene_ids,
                    family_ids,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                ),
            }

    comparisons: dict[str, Any] = {}
    for horizon in range(2):
        hname = f"h{horizon + 1}"
        comparisons[hname] = {}
        correct = arrays["correct"][:, horizon]
        persistence = arrays["persistence"][:, horizon]
        for comparison, left, right, statistic in (
            ("correct_to_persistence_ratio", correct, persistence, "ratio"),
            ("persistence_minus_correct", persistence, correct, "gap"),
            ("wrong_a2_minus_correct", arrays["wrong_a2"][:, horizon], correct, "gap"),
            ("wrong_a3_minus_correct", arrays["wrong_a3"][:, horizon], correct, "gap"),
            (
                "reset_history_minus_correct",
                arrays["reset_history"][:, horizon],
                correct,
                "gap",
            ),
            (
                "current_only_minus_correct",
                arrays["current_only"][:, horizon],
                correct,
                "gap",
            ),
            (
                "reversed_minus_correct",
                arrays["reversed_history"][:, horizon],
                correct,
                "gap",
            ),
        ):
            left_scene = _scene_values(left, scene_ids)
            right_scene = _scene_values(right, scene_ids)
            comparisons[hname][comparison] = {
                "row": _paired_statistic(
                    left,
                    right,
                    statistic=statistic,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                ),
                "scene": _paired_statistic(
                    left_scene,
                    right_scene,
                    statistic=statistic,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                ),
                "family_equal_scene": _family_equal_paired_statistic(
                    left,
                    right,
                    scene_ids,
                    family_ids,
                    statistic=statistic,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed,
                ),
            }
        for comparison in ("wrong_a2_minus_correct", "wrong_a3_minus_correct"):
            wrong_gap = comparisons[hname][comparison]
            for unit in ("row", "scene", "family_equal_scene"):
                denominator = float(summaries["persistence"][hname][unit]["point"])
                if denominator <= 1.0e-12:
                    raise DenseDINORunnerError("persistence normalization is zero")
                wrong_gap[unit]["normalized_by_persistence"] = (
                    wrong_gap[unit]["point"] / denominator
                )
    return {
        "row_count": row_count,
        "scene_count": len(set(scene_ids)),
        "family_count": len(set(family_ids)),
        "controls": summaries,
        "comparisons": comparisons,
        "bootstrap": {
            "draws": bootstrap_draws,
            "seed": bootstrap_seed,
            "row_unit": "individual_h6_v2_row",
            "scene_unit": "equal_weight_scene_mean",
            "family_equal_scene_unit": (
                "whole_scenes_resampled_within_family_then_families_equal_weighted"
            ),
            "percentiles": [2.5, 97.5],
        },
        "interpretation": {
            "wrong_action_gaps": (
                "observational_H6_action_association_evidence_not_same_state_"
                "counterfactual_causality"
            ),
            "h2_rollout": "free_running_predicted_z3_never_teacher_z3",
            "qualification_scope": (
                "H6_is_associative_only_matched_branch_qualification_is_required_"
                "before_any_planner_experiment"
            ),
        },
    }


def fixed_offline_gate(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = evaluation["comparisons"]
    audit = evaluation["prediction_audit"]
    unit = "family_equal_scene"

    def horizon_values(hname: str) -> dict[str, float]:
        return {
            "correct_to_persistence_ratio": float(
                comparisons[hname]["correct_to_persistence_ratio"][unit]["point"]
            ),
            "normalized_wrong_a2_gap": float(
                comparisons[hname]["wrong_a2_minus_correct"][unit][
                    "normalized_by_persistence"
                ]
            ),
            "normalized_wrong_a3_gap": float(
                comparisons[hname]["wrong_a3_minus_correct"][unit][
                    "normalized_by_persistence"
                ]
            ),
            "persistence_advantage_bootstrap_lower_95": float(
                comparisons[hname]["persistence_minus_correct"][unit]["lower_95"]
            ),
            "wrong_a2_gap_bootstrap_lower_95": float(
                comparisons[hname]["wrong_a2_minus_correct"][unit]["lower_95"]
            ),
            "wrong_a3_gap_bootstrap_lower_95": float(
                comparisons[hname]["wrong_a3_minus_correct"][unit]["lower_95"]
            ),
        }

    h1_values = horizon_values("h1")
    h1_criteria = {
        "correct_to_persistence_ratio_le_0p95": (
            h1_values["correct_to_persistence_ratio"] <= 0.95
        ),
        "normalized_wrong_a2_gap_ge_0p01": (
            h1_values["normalized_wrong_a2_gap"] >= 0.01
        ),
        "persistence_advantage_family_scene_bootstrap_lower_95_gt_zero": (
            h1_values["persistence_advantage_bootstrap_lower_95"] > 0.0
        ),
        "wrong_a2_gap_family_scene_bootstrap_lower_95_gt_zero": (
            h1_values["wrong_a2_gap_bootstrap_lower_95"] > 0.0
        ),
    }
    h2_values = horizon_values("h2")
    h2_criteria = {
        "correct_to_persistence_ratio_le_0p95": (
            h2_values["correct_to_persistence_ratio"] <= 0.95
        ),
        "normalized_wrong_a2_gap_ge_0p01": (
            h2_values["normalized_wrong_a2_gap"] >= 0.01
        ),
        "normalized_wrong_a3_gap_ge_0p01": (
            h2_values["normalized_wrong_a3_gap"] >= 0.01
        ),
        "persistence_advantage_family_scene_bootstrap_lower_95_gt_zero": (
            h2_values["persistence_advantage_bootstrap_lower_95"] > 0.0
        ),
        "wrong_a2_gap_family_scene_bootstrap_lower_95_gt_zero": (
            h2_values["wrong_a2_gap_bootstrap_lower_95"] > 0.0
        ),
        "wrong_a3_gap_family_scene_bootstrap_lower_95_gt_zero": (
            h2_values["wrong_a3_gap_bootstrap_lower_95"] > 0.0
        ),
    }
    h1_passes = all(h1_criteria.values())
    h2_passes = all(h2_criteria.values())
    if h1_passes and h2_passes:
        decision = "PASS_H1_MPC_AND_H2_COMPOSABILITY_OFFLINE_GATE"
    elif h1_passes:
        decision = "PASS_H1_MPC_OFFLINE_GATE_H2_COMPOSABILITY_NOT_ESTABLISHED"
    else:
        decision = "STOP_H1_MPC_OFFLINE_GATE_NOT_MET"
    return {
        "h1_mpc_gate": {
            "criteria": h1_criteria,
            "values": h1_values,
            "passes_all": h1_passes,
        },
        "h2_composability_gate": {
            "criteria": h2_criteria,
            "values": h2_values,
            "passes_all": h2_passes,
        },
        "prediction_audit_not_a_scientific_gate": {
            "all_finite": bool(audit["all_finite"]),
            "row_descriptor_std": float(audit["row_descriptor_std"]),
            "within_row_token_std": float(audit["within_row_token_std"]),
            "collapse_std_floor": COLLAPSE_STD_FLOOR,
        },
        "diagnostics_excluded_from_gate": [
            "reset_history",
            "current_only",
            "reversed_history",
        ],
        "thresholds": {
            "maximum_correct_to_persistence_ratio": 0.95,
            "minimum_normalized_wrong_action_gap": 0.01,
            "minimum_family_scene_bootstrap_lower_95": 0.0,
        },
        "primary_passes": h1_passes,
        "passes_all": h1_passes,
        "h2_composability_passes": h2_passes,
        "decision": decision,
        "claim_scope": (
            "H1_is_an_offline_MPC_eligibility_screen_only_H6_action_evidence_is_"
            "associative_and_matched_branch_qualification_remains_required"
        ),
    }


def training_continuation_decision(
    evaluations_by_update: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered update-500 futility and progress rule."""
    thresholds = {
        "minimum_persistence_advantage_ratio": 1.0,
        "minimum_positive_normalized_wrong_a2_gap": 0.0,
        "minimum_ratio_improvement_from_u250": 0.01,
        "minimum_action_gap_improvement_from_u250": 0.005,
    }
    if 500 not in evaluations_by_update:
        return {
            "applicable": False,
            "should_continue": True,
            "decision": "CONTINUE_UPDATE_500_RULE_NOT_YET_APPLICABLE",
            "thresholds": thresholds,
        }

    def values(update: int) -> tuple[float, float]:
        comparison = evaluations_by_update[update]["comparisons"]["h1"]
        ratio = float(
            comparison["correct_to_persistence_ratio"]["family_equal_scene"][
                "point"
            ]
        )
        normalized_gap = float(
            comparison["wrong_a2_minus_correct"]["family_equal_scene"][
                "normalized_by_persistence"
            ]
        )
        return ratio, normalized_gap

    ratio_500, gap_500 = values(500)
    result: dict[str, Any] = {
        "applicable": True,
        "should_continue": False,
        "values": {
            "u500_correct_to_persistence_ratio": ratio_500,
            "u500_normalized_wrong_a2_gap": gap_500,
        },
        "thresholds": thresholds,
    }
    if ratio_500 >= 1.0:
        result["decision"] = "STOP_AT_U500_NO_H1_PERSISTENCE_ADVANTAGE"
        return result
    if gap_500 <= 0.0:
        result["decision"] = "STOP_AT_U500_NONPOSITIVE_H1_WRONG_A2_GAP"
        return result
    if 250 not in evaluations_by_update:
        result.update(
            {
                "applicable": False,
                "should_continue": True,
                "decision": "CONTINUE_U250_PROGRESS_REFERENCE_UNAVAILABLE",
            }
        )
        return result
    ratio_250, gap_250 = values(250)
    ratio_improvement = ratio_250 - ratio_500
    gap_improvement = gap_500 - gap_250
    result["values"].update(
        {
            "u250_correct_to_persistence_ratio": ratio_250,
            "u250_normalized_wrong_a2_gap": gap_250,
            "ratio_improvement_from_u250": ratio_improvement,
            "action_gap_improvement_from_u250": gap_improvement,
        }
    )
    should_continue = ratio_improvement >= 0.01 or gap_improvement >= 0.005
    result["should_continue"] = should_continue
    result["decision"] = (
        "CONTINUE_TO_U1000_MEANINGFUL_H1_PROGRESS"
        if should_continue
        else "STOP_AT_U500_H1_PROGRESS_STALLED"
    )
    return result


class RGBReadAudit:
    def __init__(self) -> None:
        self.read_count = 0
        self._bindings: dict[str, tuple[str, int]] = {}

    def observe(self, leaf: str, raw: bytes) -> None:
        binding = (hashlib.sha256(raw).hexdigest(), len(raw))
        previous = self._bindings.setdefault(leaf, binding)
        if previous != binding:
            raise DenseDINORunnerError("RGB bytes changed during execution")
        self.read_count += 1

    def report(self, root: Path) -> dict[str, Any]:
        ordered = [
            {"leaf": leaf, "sha256": binding[0], "byte_count": binding[1]}
            for leaf, binding in sorted(self._bindings.items())
        ]
        return {
            "root": str(root),
            "read_count": self.read_count,
            "unique_leaf_count": len(ordered),
            "ordered_unique_leaf_bindings_sha256": hashlib.sha256(
                _canonical_bytes(ordered)
            ).hexdigest(),
        }


def _read_rgb(root: Path, leaf: str, audit: RGBReadAudit) -> bytes:
    relative = Path(leaf)
    if relative.is_absolute() or ".." in relative.parts or _is_sealed_path(relative):
        raise DenseDINORunnerError("RGB leaf escaped its bound root")
    path = root.joinpath(relative)
    resolved = _safe_path(path, must_exist=True, label="RGB leaf")
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DenseDINORunnerError("RGB leaf escaped its bound root") from exc
    if resolved != path.absolute() or not resolved.is_file() or resolved.is_symlink():
        raise DenseDINORunnerError("RGB leaf is not a direct regular file")
    raw = resolved.read_bytes()
    audit.observe(leaf, raw)
    return raw


def _load_dino_encoder(
    repository: Path, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    repo = _safe_path(repository, must_exist=True, label="DINO repository")
    ckpt = file_binding(checkpoint)
    if ckpt["sha256"] != DINO_CHECKPOINT_SHA256 or ckpt["byte_count"] != DINO_CHECKPOINT_BYTE_COUNT:
        raise DenseDINORunnerError("DINO checkpoint identity changed")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != DINO_REPOSITORY_COMMIT:
        raise DenseDINORunnerError("DINO repository commit changed")
    encoder = torch.hub.load(
        str(repo), "dinov2_vits14", source="local", pretrained=False
    )
    state = torch.load(ckpt["path"], map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    return encoder, {
        "architecture": "dinov2_vits14",
        "repository": {"path": str(repo), "commit": commit},
        "checkpoint": ckpt,
        "preprocessing": {
            "input_size": [224, 224],
            "normalization_mean": list(dino_data.IMAGENET_MEAN),
            "normalization_std": list(dino_data.IMAGENET_STD),
            "token_shape": list(TOKEN_SHAPE),
            "per_token_l2_normalization": True,
        },
    }


@torch.no_grad()
def _encode_rows(
    rows: Sequence[h6.H6V2Row],
    *,
    encoder: torch.nn.Module,
    device: torch.device,
    rgb_root: Path,
    dino_batch_size: int,
    audit: RGBReadAudit,
) -> torch.Tensor:
    prepared = [
        dino_data.preprocess_dinov2_png_bytes_v1(_read_rgb(rgb_root, leaf, audit))
        for row in rows
        for leaf in row.rgb[:5]
    ]
    batches: list[torch.Tensor] = []
    for start in range(0, len(prepared), dino_batch_size):
        inputs = torch.stack(prepared[start : start + dino_batch_size]).to(device)
        raw = encoder.forward_features(inputs)["x_norm_patchtokens"]
        batches.append(dino_data.normalize_dense_token_grid_v1(raw))
    tokens = torch.cat(batches).reshape(len(rows), 5, *TOKEN_SHAPE)
    if not bool(torch.isfinite(tokens).all()):
        raise DenseDINORunnerError("DINO produced nonfinite tokens")
    return tokens


def _action_tensor(rows: Sequence[h6.H6V2Row], table: torch.Tensor, device: torch.device) -> torch.Tensor:
    ids = torch.tensor([row.actions[:4] for row in rows], dtype=torch.long)
    return table[ids].to(device)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    rows: Sequence[h6.H6V2Row],
    *,
    donor_indices: Sequence[int],
    action_table: torch.Tensor,
    encoder: torch.nn.Module,
    device: torch.device,
    rgb_root: Path,
    batch_size: int,
    dino_batch_size: int,
    audit: RGBReadAudit,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(rows) != len(donor_indices):
        raise DenseDINORunnerError("wrong-action donor panel length changed")
    model.eval()
    metrics: dict[str, list[np.ndarray]] = {name: [] for name in CONTROL_NAMES}
    descriptors: list[torch.Tensor] = []
    token_spreads: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tokens = _encode_rows(
            batch_rows,
            encoder=encoder,
            device=device,
            rgb_root=rgb_root,
            dino_batch_size=dino_batch_size,
            audit=audit,
        )
        actions = _action_tensor(batch_rows, action_table, device)
        batch_donors = [rows[donor_indices[index]] for index in range(start, start + len(batch_rows))]
        donor_actions = _action_tensor(batch_donors, action_table, device)[:, 2:4]
        controls = build_temporal_controls(tokens, actions, donor_actions)
        correct = free_running_rollout(
            model,
            controls["context"], controls["history_actions"], controls["future_actions"]
        )
        wrong_a2 = free_running_rollout(
            model,
            controls["context"],
            controls["history_actions"],
            controls["wrong_a2_future_actions"],
        )
        wrong_a3 = free_running_rollout(
            model,
            controls["context"],
            controls["history_actions"],
            controls["wrong_a3_future_actions"],
        )
        reset_history = free_running_rollout(
            model,
            controls["context"],
            controls["reset_history_actions"],
            controls["future_actions"],
        )
        current_only = free_running_rollout(
            model,
            controls["current_only_context"],
            controls["history_actions"],
            controls["future_actions"],
        )
        reversed_prediction = free_running_rollout(
            model,
            controls["reversed_context"],
            controls["reversed_history_actions"],
            controls["future_actions"],
        )
        predictions = {
            "correct": correct,
            "persistence": controls["persistence"],
            "wrong_a2": wrong_a2,
            "wrong_a3": wrong_a3,
            "reset_history": reset_history,
            "current_only": current_only,
            "reversed_history": reversed_prediction,
        }
        for name, prediction in predictions.items():
            metrics[name].append(
                cosine_error_per_row_horizon(prediction, controls["targets"])
                .cpu()
                .numpy()
            )
        descriptors.append(correct.mean(dim=2).to(torch.float32).cpu())
        token_spreads.append(
            correct.to(torch.float32)
            .std(dim=2, unbiased=False)
            .mean(dim=-1)
            .cpu()
        )
    arrays = {name: np.concatenate(values, axis=0) for name, values in metrics.items()}
    summary = summarize_evaluation_metrics(
        arrays,
        [row.scene_id for row in rows],
        [row.family for row in rows],
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    descriptor_tensor = torch.cat(descriptors, dim=0)
    prediction_audit = {
        "all_finite": bool(torch.isfinite(descriptor_tensor).all()),
        "row_descriptor_std": float(
            descriptor_tensor.std(dim=0, unbiased=False).mean().item()
        ),
        "within_row_token_std": float(torch.cat(token_spreads, dim=0).mean().item()),
        "collapse_std_floor": COLLAPSE_STD_FLOOR,
    }
    summary["prediction_audit"] = prediction_audit
    summary["gate"] = fixed_offline_gate(summary)
    return summary


def _static_bindings(
    *,
    train_audit: Mapping[str, Any],
    val_audit: Mapping[str, Any],
    primitive_binding: Mapping[str, Any],
    dino_binding: Mapping[str, Any],
    model_source: Path,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "indices": {"train": dict(train_audit), "val": dict(val_audit)},
        "primitive_registry": dict(primitive_binding),
        "dinov2": dict(dino_binding),
        "sources": {
            "runner": file_binding(Path(__file__)),
            "model": file_binding(model_source),
            "h6_loader": file_binding(Path(h6.__file__)),
            "dino_preprocessing": file_binding(Path(dino_data.__file__)),
        },
        "environment": {
            "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
        },
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.updates <= 0
        or args.batch_size <= 0
        or args.eval_batch_size <= 0
        or args.dino_batch_size <= 0
        or args.bootstrap_draws <= 0
        or args.train_row_limit < 0
        or args.val_row_limit < 0
    ):
        raise DenseDINORunnerError("training dimensions must be positive")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
        or not math.isfinite(args.gradient_clip_norm)
        or args.gradient_clip_norm <= 0.0
    ):
        raise DenseDINORunnerError("optimizer configuration is invalid")
    trace_updates = parse_trace_updates(args.trace_updates, updates=args.updates)
    output_root = _safe_path(args.output_root, must_exist=False, label="output")
    output_root.mkdir(parents=True, exist_ok=False)
    rgb_root = _safe_path(args.rgb_root, must_exist=True, label="RGB root")
    if not rgb_root.is_dir():
        raise DenseDINORunnerError("RGB root is not a directory")
    device = torch.device(args.device)
    _seed_everything(args.seed)

    train_rows_all, train_audit = h6.load_bound_index(REPO_ROOT, role="train")
    val_rows_all, val_audit = h6.load_bound_index(REPO_ROOT, role="val")
    train_rows = train_rows_all[: args.train_row_limit or None]
    val_rows = val_rows_all[: args.val_row_limit or None]
    if len(train_rows) < args.batch_size or len(val_rows) < 2:
        raise DenseDINORunnerError("row limits are too small")
    action_table, primitive_binding = load_primitive_action_table(args.primitive_registry)
    donor_indices = select_wrong_action_donor_indices([row.actions for row in val_rows])
    encoder, dino_binding = _load_dino_encoder(
        args.dino_repository, args.dino_checkpoint, device
    )
    from lewm.models import dense_dinov2_temporal_predictor as model_module

    model = model_module.DenseDINOv2TemporalPredictorV1().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise DenseDINORunnerError("DINO encoder was not frozen")
    encoder_parameter_ids = {id(parameter) for parameter in encoder.parameters()}
    if any(
        id(parameter) in encoder_parameter_ids
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise DenseDINORunnerError("DINO encoder entered the predictor optimizer")
    bindings = _static_bindings(
        train_audit=train_audit,
        val_audit=val_audit,
        primitive_binding=primitive_binding,
        dino_binding=dino_binding,
        model_source=Path(model_module.__file__),
        device=device,
    )
    config = {
        "seed": args.seed,
        "attempt_policy": "one_seed_no_hyperparameter_tuning",
        "updates": args.updates,
        "trace_updates": list(trace_updates),
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "dino_batch_size": args.dino_batch_size,
        "train_row_count": len(train_rows),
        "val_row_count": len(val_rows),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "observation_policy": "only_preregistered_trace_updates",
        "frames": [0, 1, 2, 3, 4],
        "actions": [0, 1, 2, 3],
        "rollout_contract": {
            "h1": "z0,z1,z2+a0,a1,a2->z3_hat",
            "h2": "z1,z2,z3_hat+a1,a2,a3->z4_hat",
            "teacher_z3_at_h2": False,
        },
    }
    rgb_audit = RGBReadAudit()
    trace: list[dict[str, Any]] = []
    started = time.time()
    last_train_loss: float | None = None
    actual_final_update = 0
    continuation_decision: dict[str, Any] = {
        "applicable": False,
        "should_continue": True,
        "decision": "UPDATE_500_RULE_NOT_REACHED",
    }

    for update in range(args.updates + 1):
        if update in trace_updates:
            evaluation = evaluate_model(
                model,
                val_rows,
                donor_indices=donor_indices,
                action_table=action_table,
                encoder=encoder,
                device=device,
                rgb_root=rgb_root,
                batch_size=args.eval_batch_size,
                dino_batch_size=args.dino_batch_size,
                audit=rgb_audit,
                bootstrap_draws=args.bootstrap_draws,
                bootstrap_seed=args.bootstrap_seed,
            )
            trace.append(
                {
                    "update": update,
                    "train_loss": last_train_loss,
                    "evaluation": evaluation,
                    "elapsed_seconds": time.time() - started,
                }
            )
            if update == 500 and args.updates > 500 and 250 in trace_updates:
                continuation_decision = training_continuation_decision(
                    {
                        int(entry["update"]): entry["evaluation"]
                        for entry in trace
                    }
                )
                trace[-1]["continuation_decision"] = continuation_decision
            _write_json(
                output_root / "trace.json",
                {
                    "schema": TRACE_SCHEMA,
                    "config": config,
                    "bindings": bindings,
                    "trace": trace,
                    "continuation_decision": continuation_decision,
                    "rgb_input_audit": rgb_audit.report(rgb_root),
                },
            )
            if (
                continuation_decision.get("applicable") is True
                and continuation_decision.get("should_continue") is False
            ):
                actual_final_update = update
                break
        if update == args.updates:
            actual_final_update = update
            break

        positions = deterministic_batch_indices(
            row_count=len(train_rows),
            batch_size=args.batch_size,
            update_index=update,
            seed=args.seed,
        )
        batch_rows = [train_rows[int(index)] for index in positions]
        tokens = _encode_rows(
            batch_rows,
            encoder=encoder,
            device=device,
            rgb_root=rgb_root,
            dino_batch_size=args.dino_batch_size,
            audit=rgb_audit,
        )
        actions = _action_tensor(batch_rows, action_table, device)
        controls = build_temporal_controls(tokens, actions, actions[:, 2:4])
        model.train()
        prediction = free_running_rollout(
            model,
            controls["context"], controls["history_actions"], controls["future_actions"]
        )
        loss = cosine_error_per_row_horizon(prediction, controls["targets"]).mean()
        if not bool(torch.isfinite(loss)):
            raise DenseDINORunnerError("training loss became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()
        last_train_loss = float(loss.detach().cpu())

    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "config": config,
        "bindings": bindings,
        "update": actual_final_update,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "final_gate": trace[-1]["evaluation"]["gate"],
        "continuation_decision": continuation_decision,
        "rgb_input_audit": rgb_audit.report(rgb_root),
    }
    checkpoint_path = output_root / "final_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_binding = file_binding(checkpoint_path)
    stopped_for_futility = (
        continuation_decision.get("applicable") is True
        and continuation_decision.get("should_continue") is False
    )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": (
            continuation_decision["decision"]
            if stopped_for_futility
            else trace[-1]["evaluation"]["gate"]["decision"]
        ),
        "actual_final_update": actual_final_update,
        "config": config,
        "bindings": bindings,
        "rgb_input_audit": rgb_audit.report(rgb_root),
        "trace_binding": file_binding(output_root / "trace.json"),
        "checkpoint_binding": checkpoint_binding,
        "final_evaluation": trace[-1]["evaluation"],
        "continuation_decision": continuation_decision,
        "claim_scope": {
            "wrong_action_gap": "action_association_not_counterfactual_causality",
            "h6_evidence": "associative_only",
            "matched_branch_qualification_required_before_planner": True,
            "planner_authority": False,
            "physical_or_navigation_claim": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output_root / "terminal_report.json", terminal)
    return terminal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_RGB_ROOT)
    parser.add_argument(
        "--primitive-registry", type=Path, default=DEFAULT_PRIMITIVE_REGISTRY
    )
    parser.add_argument("--dino-repository", type=Path, default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--trace-updates", default=",".join(map(str, DEFAULT_TRACE_UPDATES)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dino-batch-size", type=int, default=32)
    parser.add_argument("--train-row-limit", type=int, default=0)
    parser.add_argument("--val-row-limit", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    terminal = run(args)
    print(json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
