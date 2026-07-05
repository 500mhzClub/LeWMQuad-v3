"""Phase 2S action-conditioned swept-geometry affordance features."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from lewm.actions import active_block_to_matrix

from .phase2_data import action_name, source_key
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION
from .phase2m_primitive_affordance import primitive_vocabulary
from .phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    FactorizedPrimitiveAffordanceExample,
    build_factorized_primitive_affordance_examples,
)
from .phase2r_geometry_affordance import (
    FramePose2D,
    _aabbs_from_manifest,
    _clamped,
    _load_frame_poses,
    _point_clearance,
    _source_metadata,
)

PHASE2S_SWEPT_FEATURE_SCHEMA = "phase2s_action_conditioned_swept_geometry_v0"


@dataclass(frozen=True)
class Phase2SSweptGeometryAffordanceExample:
    """One source state with per-primitive swept geometry features."""

    factorized_example: FactorizedPrimitiveAffordanceExample
    swept_geometry_features: tuple[tuple[float, ...], ...]
    source_pose_found: bool
    goal_pose_found: bool
    continuation_counts: tuple[int, ...]


@dataclass
class Phase2SSweptGeometryBatch:
    """Materialized per-primitive swept-geometry training batch."""

    example_indices: tuple[int, ...]
    examples: tuple[Phase2SSweptGeometryAffordanceExample, ...]
    base_examples: tuple[FactorizedPrimitiveAffordanceExample, ...]
    swept_geometry_features: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor
    factor_targets: torch.Tensor
    factor_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2SSweptGeometryBatch":
        return Phase2SSweptGeometryBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            base_examples=self.base_examples,
            swept_geometry_features=self.swept_geometry_features.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
            factor_targets=self.factor_targets.to(device),
            factor_mask=self.factor_mask.to(device),
        )


def phase2s_swept_feature_names(primitive_names: Sequence[str]) -> tuple[str, ...]:
    """Return the deterministic Phase 2S per-primitive feature order."""

    names = tuple(str(name) for name in primitive_names)
    if not names:
        raise ValueError("primitive_names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("primitive_names must be unique")
    return (
        *(f"primitive_onehot_{name}" for name in names),
        "first_command_mean_vx_norm",
        "first_command_mean_vy_norm",
        "first_command_mean_yaw_rate_norm",
        "first_command_path_length_norm",
        "first_command_abs_yaw_delta_norm",
        "first_final_forward_norm",
        "first_final_left_norm",
        "first_final_yaw_delta_sin",
        "first_final_yaw_delta_cos",
        "first_min_clearance_norm",
        "first_p05_clearance_norm",
        "first_mean_clearance_norm",
        "first_unsafe_margin_fraction",
        "first_goal_progress_norm",
        "first_final_goal_distance_norm",
        "first_heading_alignment",
        "continuation_fraction",
        "continuation_min_unsafe_fraction",
        "continuation_mean_unsafe_fraction",
        "continuation_max_min_clearance_norm",
        "continuation_max_p05_clearance_norm",
        "continuation_mean_min_clearance_norm",
        "continuation_max_goal_progress_norm",
        "continuation_mean_goal_progress_norm",
        "continuation_max_heading_alignment",
        "continuation_min_final_goal_distance_norm",
        "best_progress_unsafe_fraction",
        "best_progress_min_clearance_norm",
        "best_progress_heading_alignment",
        "safest_goal_progress_norm",
        "safest_min_clearance_norm",
        *(f"best_progress_second_onehot_{name}" for name in names),
    )


def _angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clearance_norm(clearance_m: float) -> float:
    return _clamped((float(clearance_m) + 0.2) / 1.2, 0.0, 1.0)


def _progress_norm(progress_m: float) -> float:
    return _clamped(float(progress_m) / 0.3, -1.0, 1.0)


def _goal_alignment(
    *,
    pose: FramePose2D,
    goal_pose: FramePose2D | None,
) -> float:
    if goal_pose is None:
        return 0.0
    bearing = math.atan2(goal_pose.y_m - pose.y_m, goal_pose.x_m - pose.x_m)
    error = abs(_angle_wrap(bearing - pose.yaw_rad))
    return 1.0 - _clamped(error, 0.0, math.pi) / math.pi


def _goal_metrics(
    *,
    start_pose: FramePose2D,
    final_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    max_ray_m: float,
) -> dict[str, float]:
    if goal_pose is None:
        return {
            "progress_norm": 0.0,
            "final_distance_norm": 0.0,
            "heading_alignment": 0.0,
        }
    start_distance = math.hypot(
        goal_pose.x_m - start_pose.x_m,
        goal_pose.y_m - start_pose.y_m,
    )
    final_distance = math.hypot(
        goal_pose.x_m - final_pose.x_m,
        goal_pose.y_m - final_pose.y_m,
    )
    return {
        "progress_norm": _progress_norm(start_distance - final_distance),
        "final_distance_norm": _clamped(final_distance / max_ray_m, 0.0, 1.0),
        "heading_alignment": _goal_alignment(pose=final_pose, goal_pose=goal_pose),
    }


def _rollout_pose_samples(
    *,
    source_pose: FramePose2D,
    active_blocks: Sequence[Sequence[float]],
    command_dt_s: float,
) -> tuple[FramePose2D, ...]:
    x = float(source_pose.x_m)
    y = float(source_pose.y_m)
    yaw = float(source_pose.yaw_rad)
    samples = [FramePose2D(x_m=x, y_m=y, yaw_rad=yaw)]
    for block in active_blocks:
        matrix = active_block_to_matrix(block)
        for vx_body, vy_body, yaw_rate in matrix.tolist():
            mid_yaw = yaw + 0.5 * float(yaw_rate) * command_dt_s
            x += (
                math.cos(mid_yaw) * float(vx_body)
                - math.sin(mid_yaw) * float(vy_body)
            ) * command_dt_s
            y += (
                math.sin(mid_yaw) * float(vx_body)
                + math.cos(mid_yaw) * float(vy_body)
            ) * command_dt_s
            yaw = _angle_wrap(yaw + float(yaw_rate) * command_dt_s)
            samples.append(FramePose2D(x_m=x, y_m=y, yaw_rad=yaw))
    return tuple(samples)


def _command_summary(
    active_blocks: Sequence[Sequence[float]],
    *,
    command_dt_s: float,
    max_ray_m: float,
) -> dict[str, float]:
    matrices = [active_block_to_matrix(block) for block in active_blocks]
    if not matrices:
        raise ValueError("cannot summarize empty active block sequence")
    matrix = torch.tensor(
        [row for item in matrices for row in item.tolist()],
        dtype=torch.float32,
    )
    linear_speed = torch.linalg.vector_norm(matrix[:, :2], dim=1)
    path_length = float(linear_speed.sum() * command_dt_s)
    yaw_delta = float(torch.sum(matrix[:, 2]) * command_dt_s)
    return {
        "mean_vx_norm": _clamped(float(torch.mean(matrix[:, 0])) / 0.5, -1.0, 1.0),
        "mean_vy_norm": _clamped(float(torch.mean(matrix[:, 1])) / 0.5, -1.0, 1.0),
        "mean_yaw_rate_norm": _clamped(float(torch.mean(matrix[:, 2])) / 1.0, -1.0, 1.0),
        "path_length_norm": _clamped(path_length / max_ray_m, 0.0, 1.0),
        "abs_yaw_delta_norm": _clamped(abs(yaw_delta) / math.pi, 0.0, 1.0),
    }


def _trajectory_metrics(
    *,
    source_pose: FramePose2D,
    samples: Sequence[FramePose2D],
    goal_pose: FramePose2D | None,
    aabbs: torch.Tensor,
    max_ray_m: float,
    unsafe_clearance_m: float,
) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot score an empty trajectory")
    clearances = [
        _point_clearance(sample.x_m, sample.y_m, aabbs)
        for sample in samples
    ]
    ordered = sorted(clearances)
    p05_index = int(0.05 * (len(ordered) - 1))
    final = samples[-1]
    cos_y = math.cos(-source_pose.yaw_rad)
    sin_y = math.sin(-source_pose.yaw_rad)
    dx_world = final.x_m - source_pose.x_m
    dy_world = final.y_m - source_pose.y_m
    forward = cos_y * dx_world - sin_y * dy_world
    left = sin_y * dx_world + cos_y * dy_world
    goal = _goal_metrics(
        start_pose=source_pose,
        final_pose=final,
        goal_pose=goal_pose,
        max_ray_m=max_ray_m,
    )
    return {
        "final_forward_norm": _clamped(forward / max_ray_m, -1.0, 1.0),
        "final_left_norm": _clamped(left / max_ray_m, -1.0, 1.0),
        "final_yaw_delta_sin": math.sin(_angle_wrap(final.yaw_rad - source_pose.yaw_rad)),
        "final_yaw_delta_cos": math.cos(_angle_wrap(final.yaw_rad - source_pose.yaw_rad)),
        "min_clearance_norm": _clearance_norm(min(clearances)),
        "p05_clearance_norm": _clearance_norm(ordered[p05_index]),
        "mean_clearance_norm": _clearance_norm(sum(clearances) / len(clearances)),
        "unsafe_fraction": sum(
            clearance <= unsafe_clearance_m for clearance in clearances
        )
        / len(clearances),
        **goal,
    }


def _sequence_metrics(
    *,
    row: Mapping,
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    aabbs: torch.Tensor,
    command_dt_s: float,
    max_ray_m: float,
    unsafe_clearance_m: float,
) -> dict[str, float | tuple[str, ...]]:
    active_blocks = row.get("active_blocks", ())
    samples = _rollout_pose_samples(
        source_pose=source_pose,
        active_blocks=active_blocks,
        command_dt_s=command_dt_s,
    )
    return {
        "sequence": tuple(str(value) for value in row.get("primitive_sequence", ())),
        **_trajectory_metrics(
            source_pose=source_pose,
            samples=samples,
            goal_pose=goal_pose,
            aabbs=aabbs,
            max_ray_m=max_ray_m,
            unsafe_clearance_m=unsafe_clearance_m,
        ),
    }


def _first_block_features(
    *,
    row: Mapping,
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    aabbs: torch.Tensor,
    command_dt_s: float,
    max_ray_m: float,
    unsafe_clearance_m: float,
) -> list[float]:
    block = row["active_blocks"][0]
    samples = _rollout_pose_samples(
        source_pose=source_pose,
        active_blocks=[block],
        command_dt_s=command_dt_s,
    )
    command = _command_summary(
        [block],
        command_dt_s=command_dt_s,
        max_ray_m=max_ray_m,
    )
    metrics = _trajectory_metrics(
        source_pose=source_pose,
        samples=samples,
        goal_pose=goal_pose,
        aabbs=aabbs,
        max_ray_m=max_ray_m,
        unsafe_clearance_m=unsafe_clearance_m,
    )
    return [
        command["mean_vx_norm"],
        command["mean_vy_norm"],
        command["mean_yaw_rate_norm"],
        command["path_length_norm"],
        command["abs_yaw_delta_norm"],
        metrics["final_forward_norm"],
        metrics["final_left_norm"],
        metrics["final_yaw_delta_sin"],
        metrics["final_yaw_delta_cos"],
        metrics["min_clearance_norm"],
        metrics["p05_clearance_norm"],
        metrics["mean_clearance_norm"],
        metrics["unsafe_fraction"],
        metrics["progress_norm"],
        metrics["final_distance_norm"],
        metrics["heading_alignment"],
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _primitive_swept_features(
    *,
    primitive_name: str,
    primitive_names: Sequence[str],
    rows: Sequence[Mapping],
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    aabbs: torch.Tensor,
    command_dt_s: float,
    max_ray_m: float,
    unsafe_clearance_m: float,
) -> tuple[float, ...]:
    names = tuple(primitive_names)
    if not rows:
        return tuple(0.0 for _name in phase2s_swept_feature_names(names))

    first = rows[0]
    onehot = [1.0 if primitive_name == name else 0.0 for name in names]
    first_features = _first_block_features(
        row=first,
        source_pose=source_pose,
        goal_pose=goal_pose,
        aabbs=aabbs,
        command_dt_s=command_dt_s,
        max_ray_m=max_ray_m,
        unsafe_clearance_m=unsafe_clearance_m,
    )
    records = [
        _sequence_metrics(
            row=row,
            source_pose=source_pose,
            goal_pose=goal_pose,
            aabbs=aabbs,
            command_dt_s=command_dt_s,
            max_ray_m=max_ray_m,
            unsafe_clearance_m=unsafe_clearance_m,
        )
        for row in rows
    ]
    best_progress = max(
        records,
        key=lambda item: (
            float(item["progress_norm"]),
            -float(item["unsafe_fraction"]),
            float(item["min_clearance_norm"]),
        ),
    )
    safest = max(
        records,
        key=lambda item: (
            -float(item["unsafe_fraction"]),
            float(item["min_clearance_norm"]),
            float(item["progress_norm"]),
        ),
    )
    best_sequence = tuple(best_progress["sequence"])
    second_name = best_sequence[1] if len(best_sequence) >= 2 else ""
    best_second_onehot = [1.0 if second_name == name else 0.0 for name in names]
    aggregate = [
        _clamped(len(rows) / max(1, len(names)), 0.0, 1.0),
        min(float(item["unsafe_fraction"]) for item in records),
        _mean([float(item["unsafe_fraction"]) for item in records]),
        max(float(item["min_clearance_norm"]) for item in records),
        max(float(item["p05_clearance_norm"]) for item in records),
        _mean([float(item["min_clearance_norm"]) for item in records]),
        max(float(item["progress_norm"]) for item in records),
        _mean([float(item["progress_norm"]) for item in records]),
        max(float(item["heading_alignment"]) for item in records),
        min(float(item["final_distance_norm"]) for item in records),
        float(best_progress["unsafe_fraction"]),
        float(best_progress["min_clearance_norm"]),
        float(best_progress["heading_alignment"]),
        float(safest["progress_norm"]),
        float(safest["min_clearance_norm"]),
    ]
    return tuple(float(value) for value in [*onehot, *first_features, *aggregate, *best_second_onehot])


def _source_rows_by_first_primitive(
    rows: Sequence[dict],
) -> dict[tuple[str, int], dict[str, list[dict]]]:
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[source_key(row)][action_name(row, 0)].append(row)
    return {key: dict(value) for key, value in grouped.items()}


def build_phase2s_swept_geometry_affordance_examples(
    rows: Sequence[dict],
    *,
    primitive_names: Sequence[str] | None = None,
    command_dt_s: float = 0.1,
    max_ray_m: float = 4.0,
    unsafe_clearance_m: float = 0.02,
) -> tuple[Phase2SSweptGeometryAffordanceExample, ...]:
    """Build action-conditioned swept-geometry examples for Phase 2S."""

    if command_dt_s <= 0.0:
        raise ValueError("command_dt_s must be positive")
    if max_ray_m <= 0.0:
        raise ValueError("max_ray_m must be positive")
    if unsafe_clearance_m < 0.0:
        raise ValueError("unsafe_clearance_m must be non-negative")
    names = (
        primitive_vocabulary(rows)
        if primitive_names is None
        else tuple(str(name) for name in primitive_names)
    )
    feature_count = len(phase2s_swept_feature_names(names))
    base_examples = build_factorized_primitive_affordance_examples(
        rows,
        primitive_names=names,
    )
    metadata = _source_metadata(rows)
    frame_paths = []
    for item in metadata.values():
        frame_paths.append(item["start_frame"])
        if item["goal_frame"] is not None:
            frame_paths.append(item["goal_frame"])
    poses = _load_frame_poses(frame_paths)
    source_rows = _source_rows_by_first_primitive(rows)
    manifest_cache: dict[str, Mapping] = {}
    examples = []
    for base in base_examples:
        key = (base.scene_id, base.source_index)
        item = metadata[key]
        manifest_path = item["scene_manifest"]
        manifest = manifest_cache.get(manifest_path)
        if manifest is None:
            manifest = json.loads(Path(manifest_path).read_text())
            manifest_cache[manifest_path] = manifest
        aabbs = _aabbs_from_manifest(manifest)
        source_pose = poses[item["start_frame"]]
        goal_pose = poses.get(item["goal_frame"]) if item["goal_frame"] else None
        by_primitive = source_rows[key]
        features = []
        continuation_counts = []
        for primitive in names:
            primitive_rows = by_primitive.get(primitive, [])
            continuation_counts.append(len(primitive_rows))
            feature_values = _primitive_swept_features(
                primitive_name=primitive,
                primitive_names=names,
                rows=primitive_rows,
                source_pose=source_pose,
                goal_pose=goal_pose,
                aabbs=aabbs,
                command_dt_s=command_dt_s,
                max_ray_m=max_ray_m,
                unsafe_clearance_m=unsafe_clearance_m,
            )
            if len(feature_values) != feature_count:
                raise ValueError("Phase 2S feature count mismatch")
            features.append(feature_values)
        examples.append(
            Phase2SSweptGeometryAffordanceExample(
                factorized_example=base,
                swept_geometry_features=tuple(features),
                source_pose_found=True,
                goal_pose_found=goal_pose is not None,
                continuation_counts=tuple(continuation_counts),
            )
        )
    return tuple(examples)


def materialize_phase2s_swept_geometry_batch(
    examples: Sequence[Phase2SSweptGeometryAffordanceExample],
    indices: Sequence[int],
) -> Phase2SSweptGeometryBatch:
    """Build one Phase 2S swept-geometry training batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2S batch")
    selected = tuple(examples[index] for index in example_indices)
    base_examples = tuple(example.factorized_example for example in selected)
    primitive_names = base_examples[0].primitive_names
    if any(example.primitive_names != primitive_names for example in base_examples):
        raise ValueError("all Phase 2S examples in a batch must share vocabulary")
    return Phase2SSweptGeometryBatch(
        example_indices=example_indices,
        examples=selected,
        base_examples=base_examples,
        swept_geometry_features=torch.tensor(
            [example.swept_geometry_features for example in selected],
            dtype=torch.float32,
        ),
        primitive_utility_targets=torch.tensor(
            [example.utility_targets for example in base_examples],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.tensor(
            [example.utility_mask for example in base_examples],
            dtype=torch.bool,
        ),
        factor_targets=torch.tensor(
            [example.factor_targets for example in base_examples],
            dtype=torch.float32,
        ),
        factor_mask=torch.tensor(
            [example.factor_mask for example in base_examples],
            dtype=torch.bool,
        ),
    )


def phase2s_swept_geometry_dataset_audit(
    examples: Sequence[Phase2SSweptGeometryAffordanceExample],
    *,
    split_name: str,
    feature_names: Sequence[str],
) -> dict:
    """Summarize Phase 2S swept-geometry feature coverage."""

    feature_count = len(feature_names)
    feature_tensor = (
        torch.tensor(
            [
                primitive_features
                for example in examples
                for primitive_features in example.swept_geometry_features
            ],
            dtype=torch.float32,
        )
        if examples
        else torch.zeros((0, feature_count))
    )
    continuation_counts = [
        count for example in examples for count in example.continuation_counts
    ]
    return {
        "schema": "jepa_phase2s_swept_geometry_affordance_dataset_audit_v0",
        "split": split_name,
        "feature_schema": PHASE2S_SWEPT_FEATURE_SCHEMA,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_states": len(examples),
        "primitive_feature_rows": int(feature_tensor.shape[0]),
        "feature_names": list(feature_names),
        "feature_count": feature_count,
        "source_pose_found": sum(example.source_pose_found for example in examples),
        "goal_pose_found": sum(example.goal_pose_found for example in examples),
        "minimum_continuations_per_primitive": min(continuation_counts, default=0),
        "maximum_continuations_per_primitive": max(continuation_counts, default=0),
        "mean_continuations_per_primitive": (
            sum(continuation_counts) / len(continuation_counts)
            if continuation_counts
            else 0.0
        ),
        "finite_features": bool(torch.isfinite(feature_tensor).all()),
        "feature_mean": (
            feature_tensor.mean(dim=0).tolist() if feature_tensor.numel() else []
        ),
        "feature_std": (
            feature_tensor.std(dim=0, unbiased=False).tolist()
            if feature_tensor.numel()
            else []
        ),
    }
