"""Phase 2Z action-conditioned local-occupancy affordance features."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

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
from .phase2s_swept_geometry_affordance import (
    _clearance_norm,
    _command_summary,
    _rollout_pose_samples,
)

PHASE2Z_OCCUPANCY_FEATURE_SCHEMA = (
    "phase2z_action_conditioned_local_occupancy_grid_v0"
)


@dataclass(frozen=True)
class Phase2ZOccupancyAffordanceExample:
    """One source state with per-primitive local occupancy and factor targets."""

    factorized_example: FactorizedPrimitiveAffordanceExample
    occupancy_action_grids: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    vector_features: tuple[tuple[float, ...], ...]
    source_pose_found: bool
    goal_pose_found: bool
    continuation_counts: tuple[int, ...]


@dataclass
class Phase2ZOccupancyBatch:
    """Materialized action-conditioned local-occupancy training batch."""

    example_indices: tuple[int, ...]
    examples: tuple[Phase2ZOccupancyAffordanceExample, ...]
    base_examples: tuple[FactorizedPrimitiveAffordanceExample, ...]
    occupancy_action_grids: torch.Tensor
    vector_features: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor
    factor_targets: torch.Tensor
    factor_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2ZOccupancyBatch":
        return Phase2ZOccupancyBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            base_examples=self.base_examples,
            occupancy_action_grids=self.occupancy_action_grids.to(device),
            vector_features=self.vector_features.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
            factor_targets=self.factor_targets.to(device),
            factor_mask=self.factor_mask.to(device),
        )


def phase2z_grid_channel_names() -> tuple[str, ...]:
    """Return the deterministic Phase 2Z grid channel order."""

    return (
        "occupied_or_out_of_bounds",
        "clearance_norm",
        "goal_heat",
        "candidate_first_primitive_swept_path",
    )


def phase2z_vector_feature_names(primitive_names: Sequence[str]) -> tuple[str, ...]:
    """Return the deterministic Phase 2Z per-primitive vector feature order."""

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
        "source_clearance_norm",
        "goal_present",
        "goal_distance_norm",
        "goal_forward_norm",
        "goal_left_norm",
        "goal_bearing_sin",
        "goal_bearing_cos",
    )


def _bounds_from_manifest(manifest: Mapping) -> tuple[float, float, float, float]:
    bounds = manifest.get("world_bounds_xy_m", [[-1.0, -1.0], [1.0, 1.0]])
    return (
        float(bounds[0][0]),
        float(bounds[0][1]),
        float(bounds[1][0]),
        float(bounds[1][1]),
    )


def _inside_bounds(x_m: float, y_m: float, manifest: Mapping) -> bool:
    min_x, min_y, max_x, max_y = _bounds_from_manifest(manifest)
    return min_x <= float(x_m) <= max_x and min_y <= float(y_m) <= max_y


def _world_from_local(
    *,
    source_pose: FramePose2D,
    forward_m: float,
    left_m: float,
) -> tuple[float, float]:
    cos_y = math.cos(source_pose.yaw_rad)
    sin_y = math.sin(source_pose.yaw_rad)
    return (
        source_pose.x_m + cos_y * forward_m - sin_y * left_m,
        source_pose.y_m + sin_y * forward_m + cos_y * left_m,
    )


def _local_from_world(
    *,
    source_pose: FramePose2D,
    x_m: float,
    y_m: float,
) -> tuple[float, float]:
    dx_world = float(x_m) - source_pose.x_m
    dy_world = float(y_m) - source_pose.y_m
    cos_y = math.cos(-source_pose.yaw_rad)
    sin_y = math.sin(-source_pose.yaw_rad)
    return (
        cos_y * dx_world - sin_y * dy_world,
        sin_y * dx_world + cos_y * dy_world,
    )


def _goal_vector(
    *,
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    half_extent_m: float,
) -> tuple[float, float, float, float, float, float]:
    if goal_pose is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    forward, left = _local_from_world(
        source_pose=source_pose,
        x_m=goal_pose.x_m,
        y_m=goal_pose.y_m,
    )
    distance = math.hypot(forward, left)
    bearing = math.atan2(left, forward) if distance > 1e-6 else 0.0
    scale = max(float(half_extent_m), 1e-6)
    return (
        1.0,
        _clamped(distance / scale, 0.0, 1.0),
        _clamped(forward / scale, -1.0, 1.0),
        _clamped(left / scale, -1.0, 1.0),
        math.sin(bearing),
        math.cos(bearing),
    )


def _first_block_for_rows(rows: Sequence[Mapping]) -> Sequence[float] | None:
    if not rows:
        return None
    active_blocks = rows[0].get("active_blocks", ())
    if not active_blocks:
        return None
    return active_blocks[0]


def _path_samples_local(
    *,
    rows: Sequence[Mapping],
    source_pose: FramePose2D,
    command_dt_s: float,
) -> tuple[tuple[float, float], ...]:
    block = _first_block_for_rows(rows)
    if block is None:
        return ()
    samples = _rollout_pose_samples(
        source_pose=source_pose,
        active_blocks=[block],
        command_dt_s=command_dt_s,
    )
    return tuple(
        _local_from_world(source_pose=source_pose, x_m=sample.x_m, y_m=sample.y_m)
        for sample in samples
    )


def _nearest_path_distance(
    *,
    forward_m: float,
    left_m: float,
    path_samples: Sequence[tuple[float, float]],
) -> float:
    if not path_samples:
        return float("inf")
    return min(
        math.hypot(float(forward_m) - sample[0], float(left_m) - sample[1])
        for sample in path_samples
    )


def _occupancy_action_grid(
    *,
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    manifest: Mapping,
    aabbs: torch.Tensor,
    path_samples: Sequence[tuple[float, float]],
    grid_size: int,
    half_extent_m: float,
    occupied_clearance_m: float,
    path_radius_m: float,
    goal_heat_sigma_m: float,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    cell = 2.0 * float(half_extent_m) / float(grid_size)
    effective_path_radius_m = max(float(path_radius_m), 0.5 * math.sqrt(2.0) * cell)
    goal_local = (
        None
        if goal_pose is None
        else _local_from_world(
            source_pose=source_pose,
            x_m=goal_pose.x_m,
            y_m=goal_pose.y_m,
        )
    )
    occupied_rows: list[list[float]] = []
    clearance_rows: list[list[float]] = []
    goal_rows: list[list[float]] = []
    path_rows: list[list[float]] = []
    sigma_sq = max(float(goal_heat_sigma_m) ** 2, 1e-8)
    for row in range(int(grid_size)):
        # Image-like orientation: row 0 is far forward; columns run left to right.
        forward = float(half_extent_m) - (row + 0.5) * cell
        occupied_row: list[float] = []
        clearance_row: list[float] = []
        goal_row: list[float] = []
        path_row: list[float] = []
        for col in range(int(grid_size)):
            left = -float(half_extent_m) + (col + 0.5) * cell
            world_x, world_y = _world_from_local(
                source_pose=source_pose,
                forward_m=forward,
                left_m=left,
            )
            inside = _inside_bounds(world_x, world_y, manifest)
            clearance = (
                _point_clearance(world_x, world_y, aabbs) if inside else -float("inf")
            )
            occupied = (not inside) or clearance <= float(occupied_clearance_m)
            occupied_row.append(1.0 if occupied else 0.0)
            clearance_row.append(0.0 if not inside else _clearance_norm(clearance))
            if goal_local is None:
                goal_row.append(0.0)
            else:
                goal_distance_sq = (forward - goal_local[0]) ** 2 + (
                    left - goal_local[1]
                ) ** 2
                goal_row.append(float(math.exp(-0.5 * goal_distance_sq / sigma_sq)))
            path_distance = _nearest_path_distance(
                forward_m=forward,
                left_m=left,
                path_samples=path_samples,
            )
            path_row.append(1.0 if path_distance <= effective_path_radius_m else 0.0)
        occupied_rows.append(occupied_row)
        clearance_rows.append(clearance_row)
        goal_rows.append(goal_row)
        path_rows.append(path_row)
    return (
        tuple(tuple(row) for row in occupied_rows),
        tuple(tuple(row) for row in clearance_rows),
        tuple(tuple(row) for row in goal_rows),
        tuple(tuple(row) for row in path_rows),
    )


def _primitive_vector_features(
    *,
    primitive_name: str,
    primitive_names: Sequence[str],
    rows: Sequence[Mapping],
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    aabbs: torch.Tensor,
    command_dt_s: float,
    half_extent_m: float,
) -> tuple[float, ...]:
    names = tuple(primitive_names)
    onehot = [1.0 if primitive_name == name else 0.0 for name in names]
    block = _first_block_for_rows(rows)
    if block is None:
        command = {
            "mean_vx_norm": 0.0,
            "mean_vy_norm": 0.0,
            "mean_yaw_rate_norm": 0.0,
            "path_length_norm": 0.0,
            "abs_yaw_delta_norm": 0.0,
        }
    else:
        command = _command_summary(
            [block],
            command_dt_s=command_dt_s,
            max_ray_m=half_extent_m,
        )
    clearance = _point_clearance(source_pose.x_m, source_pose.y_m, aabbs)
    return tuple(
        float(value)
        for value in (
            *onehot,
            command["mean_vx_norm"],
            command["mean_vy_norm"],
            command["mean_yaw_rate_norm"],
            command["path_length_norm"],
            command["abs_yaw_delta_norm"],
            _clearance_norm(clearance),
            *_goal_vector(
                source_pose=source_pose,
                goal_pose=goal_pose,
                half_extent_m=half_extent_m,
            ),
        )
    )


def _source_rows_by_first_primitive(
    rows: Sequence[dict],
) -> dict[tuple[str, int], dict[str, list[dict]]]:
    grouped: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[source_key(row)][action_name(row, 0)].append(row)
    return {key: dict(value) for key, value in grouped.items()}


def build_phase2z_occupancy_affordance_examples(
    rows: Sequence[dict],
    *,
    primitive_names: Sequence[str] | None = None,
    command_dt_s: float = 0.1,
    grid_size: int = 32,
    half_extent_m: float = 4.0,
    occupied_clearance_m: float = 0.15,
    path_radius_m: float = 0.12,
    goal_heat_sigma_m: float = 0.35,
) -> tuple[Phase2ZOccupancyAffordanceExample, ...]:
    """Build action-conditioned local-occupancy examples for Phase 2Z."""

    if command_dt_s <= 0.0:
        raise ValueError("command_dt_s must be positive")
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    if half_extent_m <= 0.0:
        raise ValueError("half_extent_m must be positive")
    if occupied_clearance_m < 0.0:
        raise ValueError("occupied_clearance_m must be non-negative")
    if path_radius_m <= 0.0:
        raise ValueError("path_radius_m must be positive")
    if goal_heat_sigma_m <= 0.0:
        raise ValueError("goal_heat_sigma_m must be positive")
    names = (
        primitive_vocabulary(rows)
        if primitive_names is None
        else tuple(str(name) for name in primitive_names)
    )
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
        grids = []
        vectors = []
        continuation_counts = []
        for primitive in names:
            primitive_rows = by_primitive.get(primitive, [])
            continuation_counts.append(len(primitive_rows))
            path_samples = _path_samples_local(
                rows=primitive_rows,
                source_pose=source_pose,
                command_dt_s=command_dt_s,
            )
            grids.append(
                _occupancy_action_grid(
                    source_pose=source_pose,
                    goal_pose=goal_pose,
                    manifest=manifest,
                    aabbs=aabbs,
                    path_samples=path_samples,
                    grid_size=grid_size,
                    half_extent_m=half_extent_m,
                    occupied_clearance_m=occupied_clearance_m,
                    path_radius_m=path_radius_m,
                    goal_heat_sigma_m=goal_heat_sigma_m,
                )
            )
            vectors.append(
                _primitive_vector_features(
                    primitive_name=primitive,
                    primitive_names=names,
                    rows=primitive_rows,
                    source_pose=source_pose,
                    goal_pose=goal_pose,
                    aabbs=aabbs,
                    command_dt_s=command_dt_s,
                    half_extent_m=half_extent_m,
                )
            )
        examples.append(
            Phase2ZOccupancyAffordanceExample(
                factorized_example=base,
                occupancy_action_grids=tuple(grids),
                vector_features=tuple(vectors),
                source_pose_found=True,
                goal_pose_found=goal_pose is not None,
                continuation_counts=tuple(continuation_counts),
            )
        )
    return tuple(examples)


def materialize_phase2z_occupancy_batch(
    examples: Sequence[Phase2ZOccupancyAffordanceExample],
    indices: Sequence[int],
) -> Phase2ZOccupancyBatch:
    """Build one Phase 2Z local-occupancy training batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2Z batch")
    selected = tuple(examples[index] for index in example_indices)
    base_examples = tuple(example.factorized_example for example in selected)
    primitive_names = base_examples[0].primitive_names
    if any(example.primitive_names != primitive_names for example in base_examples):
        raise ValueError("all Phase 2Z examples in a batch must share vocabulary")
    return Phase2ZOccupancyBatch(
        example_indices=example_indices,
        examples=selected,
        base_examples=base_examples,
        occupancy_action_grids=torch.tensor(
            [example.occupancy_action_grids for example in selected],
            dtype=torch.float32,
        ),
        vector_features=torch.tensor(
            [example.vector_features for example in selected],
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


def phase2z_occupancy_dataset_audit(
    examples: Sequence[Phase2ZOccupancyAffordanceExample],
    *,
    split_name: str,
    grid_channel_names: Sequence[str],
    vector_feature_names: Sequence[str],
) -> dict:
    """Summarize Phase 2Z local-occupancy feature coverage."""

    grid_tensor = (
        torch.tensor(
            [
                grid
                for example in examples
                for grid in example.occupancy_action_grids
            ],
            dtype=torch.float32,
        )
        if examples
        else torch.zeros((0, len(grid_channel_names), 0, 0))
    )
    vector_tensor = (
        torch.tensor(
            [
                vector
                for example in examples
                for vector in example.vector_features
            ],
            dtype=torch.float32,
        )
        if examples
        else torch.zeros((0, len(vector_feature_names)))
    )
    continuation_counts = [
        count for example in examples for count in example.continuation_counts
    ]
    channel_mean = (
        grid_tensor.mean(dim=(0, 2, 3)).tolist() if grid_tensor.numel() else []
    )
    channel_std = (
        grid_tensor.std(dim=(0, 2, 3), unbiased=False).tolist()
        if grid_tensor.numel()
        else []
    )
    return {
        "schema": "jepa_phase2z_occupancy_affordance_dataset_audit_v0",
        "split": split_name,
        "feature_schema": PHASE2Z_OCCUPANCY_FEATURE_SCHEMA,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_states": len(examples),
        "primitive_grid_rows": int(grid_tensor.shape[0]),
        "grid_channel_names": list(grid_channel_names),
        "grid_channel_count": len(grid_channel_names),
        "grid_shape": list(grid_tensor.shape[1:]) if grid_tensor.ndim == 4 else [],
        "vector_feature_names": list(vector_feature_names),
        "vector_feature_count": len(vector_feature_names),
        "source_pose_found": sum(example.source_pose_found for example in examples),
        "goal_pose_found": sum(example.goal_pose_found for example in examples),
        "minimum_continuations_per_primitive": min(continuation_counts, default=0),
        "maximum_continuations_per_primitive": max(continuation_counts, default=0),
        "mean_continuations_per_primitive": (
            sum(continuation_counts) / len(continuation_counts)
            if continuation_counts
            else 0.0
        ),
        "finite_grids": bool(torch.isfinite(grid_tensor).all()),
        "finite_vectors": bool(torch.isfinite(vector_tensor).all()),
        "grid_channel_mean": channel_mean,
        "grid_channel_std": channel_std,
        "vector_feature_mean": (
            vector_tensor.mean(dim=0).tolist() if vector_tensor.numel() else []
        ),
        "vector_feature_std": (
            vector_tensor.std(dim=0, unbiased=False).tolist()
            if vector_tensor.numel()
            else []
        ),
    }
