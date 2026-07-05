"""Phase 2R geometry-exposed primitive affordance feature contract."""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .phase2_data import source_key
from .phase2d_training import ACTION_UTILITY_TARGET_VERSION
from .phase2m_primitive_affordance import primitive_vocabulary
from .phase2o_factorized_affordance import (
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    FactorizedPrimitiveAffordanceExample,
    build_factorized_primitive_affordance_examples,
)

PHASE2R_GEOMETRY_FEATURE_SCHEMA = "phase2r_source_goal_geometry_features_v0"
_FRAME_NAME_RE = re.compile(r"frame_(\d+)_env_(\d+)\.png$")


@dataclass(frozen=True)
class FramePose2D:
    """World-frame base pose recovered from render replay metadata."""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class Phase2RGeometryAffordanceExample:
    """One source state with geometry features and Phase 2O factor targets."""

    factorized_example: FactorizedPrimitiveAffordanceExample
    geometry_features: tuple[float, ...]
    source_pose_found: bool
    goal_pose_found: bool


@dataclass
class Phase2RGeometryBatch:
    """Materialized geometry-feature batch with factorized primitive targets."""

    example_indices: tuple[int, ...]
    examples: tuple[Phase2RGeometryAffordanceExample, ...]
    base_examples: tuple[FactorizedPrimitiveAffordanceExample, ...]
    geometry_features: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor
    factor_targets: torch.Tensor
    factor_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2RGeometryBatch":
        return Phase2RGeometryBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            base_examples=self.base_examples,
            geometry_features=self.geometry_features.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
            factor_targets=self.factor_targets.to(device),
            factor_mask=self.factor_mask.to(device),
        )


def phase2r_geometry_feature_names(ray_count: int) -> tuple[str, ...]:
    """Return the deterministic Phase 2R geometry feature order."""

    if ray_count < 4:
        raise ValueError("ray_count must be at least 4")
    return tuple(f"ray_{index:02d}_distance_norm" for index in range(ray_count)) + (
        "source_clearance_norm",
        "source_x_norm",
        "source_y_norm",
        "source_yaw_sin",
        "source_yaw_cos",
        "goal_present",
        "goal_distance_norm",
        "goal_forward_norm",
        "goal_left_norm",
        "goal_bearing_sin",
        "goal_bearing_cos",
    )


def _frame_request(frame_path: str) -> tuple[Path, int, int]:
    path = Path(frame_path)
    match = _FRAME_NAME_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse rendered frame name: {frame_path}")
    summary_path = path.parents[1] / "summary.json"
    summary = json.loads(summary_path.read_text())
    plan_path = Path(summary["plan"])
    plan = json.loads(plan_path.read_text())
    frames_path = Path(plan.get("frames_jsonl", plan_path.with_name("frames.jsonl")))
    return frames_path, int(match.group(1)), int(match.group(2))


def _load_frame_poses(frame_paths: Sequence[str]) -> dict[str, FramePose2D]:
    requests_by_frames: dict[Path, set[tuple[int, int]]] = defaultdict(set)
    request_lookup: dict[str, tuple[Path, int, int]] = {}
    for value in frame_paths:
        frames_path, frame_index, env_index = _frame_request(value)
        request_lookup[value] = (frames_path, frame_index, env_index)
        requests_by_frames[frames_path].add((frame_index, env_index))

    found: dict[tuple[Path, int, int], FramePose2D] = {}
    for frames_path, wanted in requests_by_frames.items():
        remaining = set(wanted)
        with frames_path.open() as stream:
            for line in stream:
                if not remaining:
                    break
                record = json.loads(line)
                key = (int(record["frame_index"]), int(record["env_index"]))
                if key not in remaining:
                    continue
                position = record["base_pose_world"]["position"]
                found[(frames_path, key[0], key[1])] = FramePose2D(
                    x_m=float(position["x"]),
                    y_m=float(position["y"]),
                    yaw_rad=float(record["base_rpy_rad"]["yaw"]),
                )
                remaining.remove(key)

    poses = {}
    missing = []
    for frame_path, request in request_lookup.items():
        pose = found.get(request)
        if pose is None:
            missing.append(frame_path)
        else:
            poses[frame_path] = pose
    if missing:
        raise ValueError(f"missing frame pose metadata for {len(missing)} frames")
    return poses


def _aabbs_from_manifest(manifest: Mapping) -> torch.Tensor:
    boxes = []
    for item in [*manifest.get("walls", ()), *manifest.get("obstacles", ())]:
        cx, cy, _cz = item["center_xyz_m"]
        sx, sy, _sz = item["size_xyz_m"]
        boxes.append(
            (
                float(cx),
                float(cy),
                0.5 * float(sx),
                0.5 * float(sy),
                float(item.get("yaw_rad", 0.0)),
            )
        )
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _point_clearance(x: float, y: float, aabbs: torch.Tensor) -> float:
    if aabbs.numel() == 0:
        return float("inf")
    cx = aabbs[:, 0]
    cy = aabbs[:, 1]
    hx = aabbs[:, 2]
    hy = aabbs[:, 3]
    yaw = aabbs[:, 4]
    cos_y = torch.cos(-yaw)
    sin_y = torch.sin(-yaw)
    dx_world = float(x) - cx
    dy_world = float(y) - cy
    local_x = cos_y * dx_world - sin_y * dy_world
    local_y = sin_y * dx_world + cos_y * dy_world
    outside_x = torch.clamp(torch.abs(local_x) - hx, min=0.0)
    outside_y = torch.clamp(torch.abs(local_y) - hy, min=0.0)
    return float(torch.min(torch.hypot(outside_x, outside_y)))


def _ray_aabb_hit(
    *,
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    half_x: float,
    half_y: float,
) -> float | None:
    t_min = -float("inf")
    t_max = float("inf")
    for origin, direction, half_extent in (
        (ox, dx, half_x),
        (oy, dy, half_y),
    ):
        if abs(direction) < 1e-9:
            if origin < -half_extent or origin > half_extent:
                return None
            continue
        t1 = (-half_extent - origin) / direction
        t2 = (half_extent - origin) / direction
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return max(t_min, 0.0)


def _ray_distance(
    *,
    x: float,
    y: float,
    yaw: float,
    aabbs: torch.Tensor,
    max_ray_m: float,
) -> float:
    if aabbs.numel() == 0:
        return float(max_ray_m)
    direction_x = math.cos(yaw)
    direction_y = math.sin(yaw)
    best = float(max_ray_m)
    for cx, cy, hx, hy, box_yaw in aabbs.tolist():
        cos_y = math.cos(-box_yaw)
        sin_y = math.sin(-box_yaw)
        local_x = cos_y * (x - cx) - sin_y * (y - cy)
        local_y = sin_y * (x - cx) + cos_y * (y - cy)
        local_dx = cos_y * direction_x - sin_y * direction_y
        local_dy = sin_y * direction_x + cos_y * direction_y
        hit = _ray_aabb_hit(
            ox=local_x,
            oy=local_y,
            dx=local_dx,
            dy=local_dy,
            half_x=hx,
            half_y=hy,
        )
        if hit is not None and 0.0 <= hit < best:
            best = float(hit)
    return min(best, float(max_ray_m))


def _clamped(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _geometry_features(
    *,
    source_pose: FramePose2D,
    goal_pose: FramePose2D | None,
    manifest: Mapping,
    ray_count: int,
    max_ray_m: float,
) -> tuple[float, ...]:
    aabbs = _aabbs_from_manifest(manifest)
    ray_values = []
    for index in range(ray_count):
        relative_yaw = -math.pi + 2.0 * math.pi * index / float(ray_count)
        distance = _ray_distance(
            x=source_pose.x_m,
            y=source_pose.y_m,
            yaw=source_pose.yaw_rad + relative_yaw,
            aabbs=aabbs,
            max_ray_m=max_ray_m,
        )
        ray_values.append(_clamped(distance / max_ray_m, 0.0, 1.0))

    clearance = _point_clearance(source_pose.x_m, source_pose.y_m, aabbs)
    bounds = manifest.get("world_bounds_xy_m", [[-1.0, -1.0], [1.0, 1.0]])
    min_x, min_y = float(bounds[0][0]), float(bounds[0][1])
    max_x, max_y = float(bounds[1][0]), float(bounds[1][1])
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_x = max(1e-6, 0.5 * (max_x - min_x))
    half_y = max(1e-6, 0.5 * (max_y - min_y))
    source_values = [
        _clamped(clearance / max_ray_m, 0.0, 1.0),
        _clamped((source_pose.x_m - center_x) / half_x, -1.0, 1.0),
        _clamped((source_pose.y_m - center_y) / half_y, -1.0, 1.0),
        math.sin(source_pose.yaw_rad),
        math.cos(source_pose.yaw_rad),
    ]

    if goal_pose is None:
        goal_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        dx_world = goal_pose.x_m - source_pose.x_m
        dy_world = goal_pose.y_m - source_pose.y_m
        cos_y = math.cos(-source_pose.yaw_rad)
        sin_y = math.sin(-source_pose.yaw_rad)
        forward = cos_y * dx_world - sin_y * dy_world
        left = sin_y * dx_world + cos_y * dy_world
        distance = math.hypot(forward, left)
        bearing = math.atan2(left, forward) if distance > 1e-6 else 0.0
        goal_values = [
            1.0,
            _clamped(distance / max_ray_m, 0.0, 1.0),
            _clamped(forward / max_ray_m, -1.0, 1.0),
            _clamped(left / max_ray_m, -1.0, 1.0),
            math.sin(bearing),
            math.cos(bearing),
        ]
    return tuple(float(value) for value in [*ray_values, *source_values, *goal_values])


def _source_metadata(rows: Sequence[dict]) -> dict[tuple[str, int], dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_key(row)].append(row)
    metadata = {}
    for key, source_rows in grouped.items():
        start_frames = {str(row["start_frame"]) for row in source_rows}
        scene_manifests = {str(row["scene_manifest"]) for row in source_rows}
        goal_frames = {
            str(row["goal_frame"])
            for row in source_rows
            if row.get("goal_frame") is not None
        }
        if len(start_frames) != 1:
            raise ValueError(f"source state has multiple start frames: {key}")
        if len(scene_manifests) != 1:
            raise ValueError(f"source state has multiple scene manifests: {key}")
        if len(goal_frames) > 1:
            raise ValueError(f"source state has multiple goal frames: {key}")
        metadata[key] = {
            "start_frame": next(iter(start_frames)),
            "scene_manifest": next(iter(scene_manifests)),
            "goal_frame": next(iter(goal_frames)) if goal_frames else None,
        }
    return metadata


def build_phase2r_geometry_affordance_examples(
    rows: Sequence[dict],
    *,
    primitive_names: Sequence[str] | None = None,
    ray_count: int = 16,
    max_ray_m: float = 4.0,
) -> tuple[Phase2RGeometryAffordanceExample, ...]:
    """Build source-local geometry feature examples for Phase 2R."""

    if max_ray_m <= 0.0:
        raise ValueError("max_ray_m must be positive")
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
        source_pose = poses[item["start_frame"]]
        goal_pose = poses.get(item["goal_frame"]) if item["goal_frame"] else None
        examples.append(
            Phase2RGeometryAffordanceExample(
                factorized_example=base,
                geometry_features=_geometry_features(
                    source_pose=source_pose,
                    goal_pose=goal_pose,
                    manifest=manifest,
                    ray_count=ray_count,
                    max_ray_m=max_ray_m,
                ),
                source_pose_found=True,
                goal_pose_found=goal_pose is not None,
            )
        )
    return tuple(examples)


def materialize_phase2r_geometry_batch(
    examples: Sequence[Phase2RGeometryAffordanceExample],
    indices: Sequence[int],
) -> Phase2RGeometryBatch:
    """Build one Phase 2R geometry-feature training batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2R batch")
    selected = tuple(examples[index] for index in example_indices)
    base_examples = tuple(example.factorized_example for example in selected)
    primitive_names = base_examples[0].primitive_names
    if any(example.primitive_names != primitive_names for example in base_examples):
        raise ValueError("all Phase 2R examples in a batch must share vocabulary")
    return Phase2RGeometryBatch(
        example_indices=example_indices,
        examples=selected,
        base_examples=base_examples,
        geometry_features=torch.tensor(
            [example.geometry_features for example in selected],
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


def phase2r_geometry_dataset_audit(
    examples: Sequence[Phase2RGeometryAffordanceExample],
    *,
    split_name: str,
    feature_names: Sequence[str],
) -> dict:
    """Summarize Phase 2R geometry feature coverage."""

    feature_count = len(feature_names)
    feature_tensor = (
        torch.tensor([example.geometry_features for example in examples])
        if examples
        else torch.zeros((0, feature_count))
    )
    return {
        "schema": "jepa_phase2r_geometry_affordance_dataset_audit_v0",
        "split": split_name,
        "feature_schema": PHASE2R_GEOMETRY_FEATURE_SCHEMA,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "source_target_version": ACTION_UTILITY_TARGET_VERSION,
        "source_states": len(examples),
        "feature_names": list(feature_names),
        "feature_count": feature_count,
        "source_pose_found": sum(example.source_pose_found for example in examples),
        "goal_pose_found": sum(example.goal_pose_found for example in examples),
        "finite_features": bool(torch.isfinite(feature_tensor).all()),
        "feature_mean": (
            feature_tensor.mean(dim=0).tolist() if len(examples) else []
        ),
        "feature_std": (
            feature_tensor.std(dim=0, unbiased=False).tolist()
            if len(examples)
            else []
        ),
    }
