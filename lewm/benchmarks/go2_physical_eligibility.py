"""Exact directional-footprint eligibility for deployment maze benchmarks.

The regular fixed-spawn audit deliberately uses a yaw-invariant planning disc.
This module adds the stricter deployment check bound by geometry contract v2:
the observed-maximum directional support polygon is evaluated at the manifest's
actual spawn yaw and on a staged-rotation/fore-aft SE(2) lattice.  A scene is
eligible only when every task beacon has a reachable, line-of-sight claim-state
witness on that lattice.

This is privileged benchmark-construction code.  It is not a deployed planner
and must never be used as an online observation source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from lewm.benchmarks.go2_physical_claim_evaluator import (
    CLAIM_ABSOLUTE_BEARING_RAD,
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    object_id_reference,
)
from lewm.planning.geometry_contract import GeometryContract
from lewm.planning.oriented_footprint import (
    DirectionalSupportFootprint,
    ManifestDirectionalFootprintFeasibility,
    Pose2D,
)
from lewm_worlds.manifest import SceneManifest
from lewm_worlds.scene_graph import SceneGraph


POLICY_SCHEMA = "lewm_go2_directional_footprint_policy_v1"
ELIGIBILITY_SCHEMA = "lewm_go2_physical_scene_eligibility_v1"
REQUIRED_GEOMETRY_SCHEMA = "lewm_go2_generalization_geometry_v2"
REQUIRED_PROFILE = "observed_max_plus_margin"
REQUIRED_COLLISION_REPRESENTATION = "directional_polygon_at_actual_yaw"
FOOTPRINT_SEMANTIC_SCHEMA = "lewm_go2_directional_footprint_semantics_v1"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def yaw_from_wxyz(quaternion: Iterable[float]) -> float:
    """Return planar yaw from a manifest ``(w, x, y, z)`` quaternion."""

    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("spawn quaternion must contain four finite values")
    qw, qx, qy, qz = values
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0:
        raise ValueError("spawn quaternion must have non-zero norm")
    qw, qx, qy, qz = (value / norm for value in values)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


@dataclass(frozen=True)
class PhysicalEligibilityConfig:
    """Discretization and task contract for the exact polygon audit."""

    cell_size_m: float
    maximum_translation_substep_m: float = 0.025
    yaw_bins: int = 16
    rotation_subsamples: int = 5
    claim_radius_m: float = 1.20
    preferred_standoff_m: float = 1.05
    require_line_of_sight: bool = True
    required_beacon_count: int = 4
    mask_validation_samples: int = 32

    def validate(self) -> None:
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        if (
            not math.isfinite(self.maximum_translation_substep_m)
            or self.maximum_translation_substep_m <= 0.0
        ):
            raise ValueError("maximum_translation_substep_m must be positive")
        if self.yaw_bins < 8 or self.yaw_bins % 4:
            raise ValueError("yaw_bins must be a multiple of four and at least eight")
        if self.rotation_subsamples < 1:
            raise ValueError("rotation_subsamples must be positive")
        if not math.isfinite(self.claim_radius_m) or self.claim_radius_m <= 0.0:
            raise ValueError("claim_radius_m must be positive")
        if (
            not math.isfinite(self.preferred_standoff_m)
            or self.preferred_standoff_m <= 0.0
            or self.preferred_standoff_m > self.claim_radius_m
        ):
            raise ValueError(
                "preferred_standoff_m must be positive and no larger than claim_radius_m"
            )
        if self.required_beacon_count < 1:
            raise ValueError("required_beacon_count must be positive")
        if self.mask_validation_samples < 0:
            raise ValueError("mask_validation_samples must be non-negative")


@dataclass(frozen=True)
class LoadedDirectionalPolicy:
    """Verified immutable policy artifact and reconstructed polygon."""

    source_path: Path
    file_sha256: str
    content_sha256: str
    policy_id: str
    profile_name: str
    footprint: DirectionalSupportFootprint
    policy_content_canonical_bytes: bytes
    footprint_semantic_sha256: str

    def provenance_dict(self, *, repository_root: Path | None = None) -> dict[str, Any]:
        path = self.source_path
        if repository_root is not None:
            try:
                path_text = str(path.relative_to(repository_root.resolve()))
            except ValueError:
                path_text = str(path)
        else:
            path_text = str(path)
        return {
            "path": path_text,
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "policy_id": self.policy_id,
            "profile": self.profile_name,
            "footprint_semantic_sha256": self.footprint_semantic_sha256,
            "vertex_count": len(self.footprint.vertices_xy_m),
            "maximum_vertex_radius_m": self.footprint.maximum_vertex_radius_m,
        }


def directional_footprint_semantic_sha256(
    footprint: DirectionalSupportFootprint,
) -> str:
    """Commit exact ordered binary64 geometry and its reconstruction inputs."""

    if type(footprint) is not DirectionalSupportFootprint:
        raise ValueError("directional footprint has the wrong exact type")
    if (
        type(footprint.vertices_xy_m) is not tuple
        or any(
            type(vertex) is not tuple
            or len(vertex) != 2
            or any(type(value) is not float or not math.isfinite(value) for value in vertex)
            for vertex in footprint.vertices_xy_m
        )
        or type(footprint.support_angles_deg) is not tuple
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in footprint.support_angles_deg
        )
        or type(footprint.support_values_m) is not tuple
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in footprint.support_values_m
        )
        or type(footprint.margin_m) is not float
        or not math.isfinite(footprint.margin_m)
    ):
        raise ValueError("directional footprint fields are not exact finite tuples")
    payload = {
        "schema": FOOTPRINT_SEMANTIC_SCHEMA,
        "vertices_xy_m_binary64_hex": [
            [x_m.hex(), y_m.hex()] for x_m, y_m in footprint.vertices_xy_m
        ],
        "support_angles_deg_binary64_hex": [
            value.hex() for value in footprint.support_angles_deg
        ],
        "support_values_m_binary64_hex": [
            value.hex() for value in footprint.support_values_m
        ],
        "margin_m_binary64_hex": footprint.margin_m.hex(),
    }
    return _payload_sha256(payload)


def validate_loaded_directional_policy_content(
    policy: LoadedDirectionalPolicy,
) -> str:
    """Bind an in-memory footprint to the exact content verified in its one file read."""

    if type(policy) is not LoadedDirectionalPolicy:
        raise ValueError("loaded directional policy has the wrong exact type")
    if (
        not isinstance(policy.source_path, Path)
        or type(policy.file_sha256) is not str
        or type(policy.content_sha256) is not str
        or type(policy.policy_id) is not str
        or type(policy.profile_name) is not str
        or type(policy.policy_content_canonical_bytes) is not bytes
        or type(policy.footprint_semantic_sha256) is not str
    ):
        raise ValueError("loaded directional policy has invalid field types")
    content_bytes = policy.policy_content_canonical_bytes
    if hashlib.sha256(content_bytes).hexdigest() != policy.content_sha256:
        raise ValueError("loaded directional policy content bytes do not match its hash")
    try:
        content = json.loads(content_bytes.decode("utf-8"))
        canonical_content = _canonical_bytes(content)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("loaded directional policy content bytes are not canonical JSON") from exc
    if canonical_content != content_bytes or type(content) is not dict:
        raise ValueError("loaded directional policy content bytes are not canonical JSON")
    if (
        content.get("schema") != POLICY_SCHEMA
        or content.get("policy_id") != policy.policy_id
        or content.get("recommended_profile") != policy.profile_name
    ):
        raise ValueError("loaded directional policy identity differs from bound content")
    selection = content.get("selection_policy")
    profiles = content.get("profiles")
    if (
        not isinstance(selection, Mapping)
        or selection.get("planning_and_collision") != policy.profile_name
        or not isinstance(profiles, Mapping)
        or not isinstance(profiles.get(policy.profile_name), Mapping)
    ):
        raise ValueError("loaded directional policy profile differs from bound content")
    expected_footprint = _footprint_from_profile(profiles[policy.profile_name])
    expected_semantic_sha256 = directional_footprint_semantic_sha256(
        expected_footprint
    )
    actual_semantic_sha256 = directional_footprint_semantic_sha256(policy.footprint)
    if (
        policy.footprint_semantic_sha256 != expected_semantic_sha256
        or actual_semantic_sha256 != expected_semantic_sha256
    ):
        raise ValueError("loaded directional footprint differs from bound policy content")
    return expected_semantic_sha256


def _footprint_from_profile(profile: Mapping[str, Any]) -> DirectionalSupportFootprint:
    planes = profile.get("support_planes")
    vertices = profile.get("vertices_xy_body_m")
    if not isinstance(planes, list) or not isinstance(vertices, list):
        raise ValueError("directional profile is missing support planes or vertices")
    raw_support: dict[float, float] = {}
    for plane in planes:
        if not isinstance(plane, Mapping):
            raise ValueError("support plane entries must be objects")
        angle = float(plane["angle_deg"])
        if angle in raw_support:
            raise ValueError(f"duplicate directional support angle: {angle}")
        raw_support[angle] = float(plane["raw_support_m"])
    footprint = DirectionalSupportFootprint.from_directional_support(
        raw_support,
        margin_m=float(profile["margin_m"]),
    )
    artifact_vertices = tuple(
        (float(vertex[0]), float(vertex[1])) for vertex in vertices
    )
    if len(artifact_vertices) != len(footprint.vertices_xy_m) or any(
        math.dist(expected, actual) > 1e-10
        for expected, actual in zip(
            artifact_vertices,
            footprint.vertices_xy_m,
            strict=True,
        )
    ):
        raise ValueError("policy vertices do not match reconstructed support planes")
    declared_radius = float(profile["maximum_vertex_radius_m"])
    if not math.isclose(
        declared_radius,
        footprint.maximum_vertex_radius_m,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("policy maximum vertex radius does not match its polygon")
    return footprint


def load_observed_max_directional_policy(
    path: Path,
    *,
    expected_content_sha256: str,
    expected_policy_id: str,
    expected_profile: str = REQUIRED_PROFILE,
) -> LoadedDirectionalPolicy:
    """Load and cryptographically bind the observed-max collision polygon."""

    source_path = path.resolve()
    source_bytes = source_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("directional footprint policy root must be an object")
    if payload.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"unsupported directional footprint schema: {payload.get('schema')!r}")
    declared_content = str(payload.get("content_sha256", ""))
    content = dict(payload)
    content.pop("content_sha256", None)
    policy_content_canonical_bytes = _canonical_bytes(content)
    actual_content = hashlib.sha256(policy_content_canonical_bytes).hexdigest()
    if not _is_sha256(declared_content) or declared_content != actual_content:
        raise ValueError("directional footprint content hash mismatch")
    if declared_content not in source_path.name:
        raise ValueError("directional footprint filename is not content-addressed")
    if declared_content != expected_content_sha256:
        raise ValueError(
            "directional footprint does not match geometry contract content ID"
        )
    if str(payload.get("policy_id", "")) != expected_policy_id:
        raise ValueError("directional footprint policy ID does not match geometry contract")
    if expected_profile != REQUIRED_PROFILE:
        raise ValueError("physical benchmark must use observed_max_plus_margin")
    selection = payload.get("selection_policy")
    if not isinstance(selection, Mapping) or selection.get(
        "planning_and_collision"
    ) != expected_profile:
        raise ValueError("policy does not select observed-max profile for collision")
    if payload.get("recommended_profile") != expected_profile:
        raise ValueError("policy recommended profile does not match geometry contract")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(
        profiles.get(expected_profile), Mapping
    ):
        raise ValueError(f"policy is missing profile {expected_profile!r}")
    footprint = _footprint_from_profile(profiles[expected_profile])
    policy = LoadedDirectionalPolicy(
        source_path=source_path,
        file_sha256=hashlib.sha256(source_bytes).hexdigest(),
        content_sha256=declared_content,
        policy_id=expected_policy_id,
        profile_name=expected_profile,
        footprint=footprint,
        policy_content_canonical_bytes=policy_content_canonical_bytes,
        footprint_semantic_sha256=directional_footprint_semantic_sha256(footprint),
    )
    validate_loaded_directional_policy_content(policy)
    return policy


def policy_from_geometry_contract(
    contract: GeometryContract,
    *,
    repository_root: Path,
    policy_override: Path | None = None,
) -> LoadedDirectionalPolicy:
    """Resolve the only physical policy authorized by geometry contract v2."""

    if contract.schema != REQUIRED_GEOMETRY_SCHEMA:
        raise ValueError("exact physical eligibility requires geometry contract v2")
    swept = contract.swept_footprint
    if swept.strict_collision_representation != REQUIRED_COLLISION_REPRESENTATION:
        raise ValueError("geometry contract does not require actual-yaw polygon collision")
    if swept.directional_profile != REQUIRED_PROFILE:
        raise ValueError("geometry contract must bind observed_max_plus_margin")
    content_sha = swept.directional_policy_content_sha256
    policy_id = swept.directional_policy_id
    if content_sha is None or policy_id is None:
        raise ValueError("geometry contract is missing directional policy identity")
    source = contract.source_artifacts.get("directional_footprint_policy")
    if not isinstance(source, Mapping):
        raise ValueError("geometry contract is missing directional policy source")
    bound_path = (repository_root.resolve() / str(source.get("path", ""))).resolve()
    if policy_override is not None and policy_override.resolve() != bound_path:
        raise ValueError("policy override differs from geometry-bound source path")
    policy = load_observed_max_directional_policy(
        bound_path,
        expected_content_sha256=content_sha,
        expected_policy_id=policy_id,
        expected_profile=swept.directional_profile,
    )
    expected_file_sha = str(source.get("sha256", ""))
    if policy.file_sha256 != expected_file_sha:
        raise ValueError("policy file hash differs from geometry-bound source hash")
    declared_radius = swept.maximum_vertex_radius_m
    if declared_radius is None or not math.isclose(
        declared_radius,
        policy.footprint.maximum_vertex_radius_m,
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("geometry contract polygon radius differs from policy")
    return policy


class DirectionalSE2Lattice:
    """Exact polygon SAT masks with staged yaw and fore-aft connectivity."""

    def __init__(
        self,
        checker: ManifestDirectionalFootprintFeasibility,
        *,
        cell_size_m: float,
        yaw_bins: int,
        rotation_subsamples: int,
        maximum_translation_substep_m: float = 0.025,
    ) -> None:
        config = PhysicalEligibilityConfig(
            cell_size_m=cell_size_m,
            maximum_translation_substep_m=maximum_translation_substep_m,
            yaw_bins=yaw_bins,
            rotation_subsamples=rotation_subsamples,
        )
        config.validate()
        self.checker = checker
        self.cell_size_m = float(cell_size_m)
        self.yaw_bins = int(yaw_bins)
        self.rotation_subsamples = int(rotation_subsamples)
        self.maximum_translation_substep_m = float(
            maximum_translation_substep_m
        )
        self.fine_yaw_bins = self.yaw_bins * self.rotation_subsamples
        (x_min, y_min), (x_max, y_max) = checker.world_bounds_xy_m

        def aligned_origin(lower: float, anchor: float) -> float:
            anchor_index = math.floor((anchor - lower) / self.cell_size_m)
            origin = anchor - (anchor_index + 0.5) * self.cell_size_m
            return origin - self.cell_size_m if origin > lower else origin

        self.origin_xy_m = (
            aligned_origin(float(x_min), float(checker.manifest.spawn.xyz_m[0])),
            aligned_origin(float(y_min), float(checker.manifest.spawn.xyz_m[1])),
        )
        self.shape = (
            int(math.ceil((float(x_max) - self.origin_xy_m[0]) / self.cell_size_m)),
            int(math.ceil((float(y_max) - self.origin_xy_m[1]) / self.cell_size_m)),
        )
        xs = self.origin_xy_m[0] + (
            np.arange(self.shape[0], dtype=np.float64) + 0.5
        ) * self.cell_size_m
        ys = self.origin_xy_m[1] + (
            np.arange(self.shape[1], dtype=np.float64) + 0.5
        ) * self.cell_size_m
        self.grid_x, self.grid_y = np.meshgrid(xs, ys, indexing="ij")
        self.fine_free = np.stack(
            [
                self._free_mask(2.0 * math.pi * index / self.fine_yaw_bins)
                for index in range(self.fine_yaw_bins)
            ],
            axis=0,
        )
        self.translation_free = np.zeros(
            (2, self.yaw_bins, *self.shape),
            dtype=bool,
        )
        for heading in range(self.yaw_bins):
            step = self._grid_step_for_heading(heading)
            if step is None:
                continue
            dx, dy = step
            yaw_rad = 2.0 * math.pi * heading / self.yaw_bins
            edge_length_m = self.cell_size_m * math.hypot(dx, dy)
            interval_count = max(
                1,
                int(
                    math.ceil(
                        edge_length_m / self.maximum_translation_substep_m
                    )
                ),
            )
            for direction_index, direction in enumerate((-1, 1)):
                free = np.ones(self.shape, dtype=bool)
                for sample_index in range(1, interval_count + 1):
                    fraction = sample_index / interval_count
                    free &= self._free_mask(
                        yaw_rad,
                        offset_xy_m=(
                            direction * dx * self.cell_size_m * fraction,
                            direction * dy * self.cell_size_m * fraction,
                        ),
                    )
                self.translation_free[direction_index, heading] = free

    def _free_mask(
        self,
        yaw_rad: float,
        *,
        offset_xy_m: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        body_vertices = np.asarray(
            self.checker.footprint.vertices_xy_m,
            dtype=np.float64,
        )
        relative_vertices = np.stack(
            (
                cos_yaw * body_vertices[:, 0] - sin_yaw * body_vertices[:, 1],
                sin_yaw * body_vertices[:, 0] + cos_yaw * body_vertices[:, 1],
            ),
            axis=1,
        )
        (x_min, y_min), (x_max, y_max) = self.checker.world_bounds_xy_m
        epsilon = self.checker.geometry_epsilon_m
        grid_x_all = self.grid_x + float(offset_xy_m[0])
        grid_y_all = self.grid_y + float(offset_xy_m[1])
        free = np.ones(self.shape, dtype=bool)
        for relative_x, relative_y in relative_vertices:
            vertex_x = grid_x_all + relative_x
            vertex_y = grid_y_all + relative_y
            free &= (
                (vertex_x >= x_min - epsilon)
                & (vertex_x <= x_max + epsilon)
                & (vertex_y >= y_min - epsilon)
                & (vertex_y <= y_max + epsilon)
            )

        polygon_axes: list[np.ndarray] = []
        following_vertices = np.concatenate(
            (relative_vertices[1:], relative_vertices[:1]), axis=0
        )
        for first, second in zip(
            relative_vertices,
            following_vertices,
            strict=True,
        ):
            edge = second - first
            length = float(np.linalg.norm(edge))
            polygon_axes.append(np.asarray((-edge[1] / length, edge[0] / length)))

        for obstacle in self.checker.collision_boxes:
            obstacle_axes = tuple(
                np.asarray(axis, dtype=np.float64) for axis in obstacle.axes
            )
            obstacle_vertices = np.asarray(obstacle.corners_xy_m, dtype=np.float64)
            base_x_min = float(
                np.min(obstacle_vertices[:, 0]) - np.max(relative_vertices[:, 0])
            )
            base_x_max = float(
                np.max(obstacle_vertices[:, 0]) - np.min(relative_vertices[:, 0])
            )
            base_y_min = float(
                np.min(obstacle_vertices[:, 1]) - np.max(relative_vertices[:, 1])
            )
            base_y_max = float(
                np.max(obstacle_vertices[:, 1]) - np.min(relative_vertices[:, 1])
            )
            x_values = grid_x_all[:, 0]
            y_values = grid_y_all[0, :]
            x_start = int(np.searchsorted(x_values, base_x_min - epsilon, side="left"))
            x_stop = int(np.searchsorted(x_values, base_x_max + epsilon, side="right"))
            y_start = int(np.searchsorted(y_values, base_y_min - epsilon, side="left"))
            y_stop = int(np.searchsorted(y_values, base_y_max + epsilon, side="right"))
            if x_start >= x_stop or y_start >= y_stop:
                continue
            grid_x = grid_x_all[x_start:x_stop, y_start:y_stop]
            grid_y = grid_y_all[x_start:x_stop, y_start:y_stop]
            collision = np.ones(grid_x.shape, dtype=bool)
            for axis in (*polygon_axes, *obstacle_axes):
                base_projection = grid_x * axis[0] + grid_y * axis[1]
                relative_projection = relative_vertices @ axis
                polygon_min = base_projection + float(np.min(relative_projection))
                polygon_max = base_projection + float(np.max(relative_projection))
                obstacle_center = (
                    obstacle.center_xy_m[0] * axis[0]
                    + obstacle.center_xy_m[1] * axis[1]
                )
                obstacle_radius = (
                    obstacle.half_extent_x_m
                    * abs(float(np.dot(obstacle_axes[0], axis)))
                    + obstacle.half_extent_y_m
                    * abs(float(np.dot(obstacle_axes[1], axis)))
                )
                obstacle_min = obstacle_center - obstacle_radius
                obstacle_max = obstacle_center + obstacle_radius
                collision &= (
                    (polygon_max >= obstacle_min - epsilon)
                    & (obstacle_max >= polygon_min - epsilon)
                )
            free[x_start:x_stop, y_start:y_stop] &= ~collision
        return free

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_xy_m[0] + (cell[0] + 0.5) * self.cell_size_m,
            self.origin_xy_m[1] + (cell[1] + 0.5) * self.cell_size_m,
        )

    def nearest_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            int(round((float(x_m) - self.origin_xy_m[0]) / self.cell_size_m - 0.5)),
            int(round((float(y_m) - self.origin_xy_m[1]) / self.cell_size_m - 0.5)),
        )

    def heading_index(self, yaw_rad: float) -> int:
        return int(
            round(float(yaw_rad) * self.yaw_bins / (2.0 * math.pi))
        ) % self.yaw_bins

    def pose_for_state(self, state: tuple[int, int, int]) -> Pose2D:
        heading, x, y = state
        world_x, world_y = self.cell_to_world((x, y))
        return Pose2D(
            world_x,
            world_y,
            2.0 * math.pi * heading / self.yaw_bins,
        )

    def _state_free(self, heading: int, x: int, y: int) -> bool:
        return bool(
            0 <= x < self.shape[0]
            and 0 <= y < self.shape[1]
            and self.fine_free[
                (heading % self.yaw_bins) * self.rotation_subsamples,
                x,
                y,
            ]
        )

    def _rotation_free(self, heading: int, x: int, y: int, direction: int) -> bool:
        start = heading * self.rotation_subsamples
        return all(
            self.fine_free[
                (start + direction * offset) % self.fine_yaw_bins,
                x,
                y,
            ]
            for offset in range(1, self.rotation_subsamples + 1)
        )

    def _grid_step_for_heading(self, heading: int) -> tuple[int, int] | None:
        angle = 2.0 * math.pi * heading / self.yaw_bins
        unit_x, unit_y = math.cos(angle), math.sin(angle)
        dx, dy = int(round(unit_x)), int(round(unit_y))
        length = math.hypot(dx, dy)
        if length <= 0.0:
            return None
        if not (
            math.isclose(unit_x, dx / length, rel_tol=0.0, abs_tol=1e-10)
            and math.isclose(unit_y, dy / length, rel_tol=0.0, abs_tol=1e-10)
        ):
            return None
        return dx, dy

    def _neighbors(
        self,
        state: tuple[int, int, int],
    ) -> Iterable[tuple[tuple[int, int, int], str]]:
        heading, x, y = state
        for rotation_direction, action in ((-1, "turn_right"), (1, "turn_left")):
            next_heading = (heading + rotation_direction) % self.yaw_bins
            if self._rotation_free(heading, x, y, rotation_direction):
                yield (next_heading, x, y), action
        step = self._grid_step_for_heading(heading)
        if step is None:
            return
        dx, dy = step
        for direction_index, (direction, action) in enumerate(
            ((-1, "reverse"), (1, "forward"))
        ):
            next_x = x + direction * dx
            next_y = y + direction * dy
            if not self._state_free(heading, next_x, next_y):
                continue
            if not self.translation_free[direction_index, heading, x, y]:
                continue
            yield (heading, next_x, next_y), action

    def snap_pose(self, pose: Pose2D) -> tuple[int, int, int] | None:
        heading = self.heading_index(pose.yaw_rad)
        x, y = self.nearest_cell(pose.x_m, pose.y_m)
        if not self._state_free(heading, x, y):
            return None
        snapped = self.pose_for_state((heading, x, y))
        if not self.checker.is_swept_pose_feasible(pose, snapped):
            return None
        return (heading, x, y)

    def validate_masks(self, *, samples: int) -> None:
        rng = np.random.default_rng(0x4C45574D)
        for _ in range(samples):
            heading = int(rng.integers(0, self.fine_yaw_bins))
            x = int(rng.integers(0, self.shape[0]))
            y = int(rng.integers(0, self.shape[1]))
            world_x, world_y = self.cell_to_world((x, y))
            pose = Pose2D(
                world_x,
                world_y,
                2.0 * math.pi * heading / self.fine_yaw_bins,
            )
            expected = self.checker.is_pose_feasible(pose)
            actual = bool(self.fine_free[heading, x, y])
            if actual != expected:
                raise AssertionError(
                    f"vectorized directional SAT mismatch at {pose}: "
                    f"{actual=} {expected=}"
                )

    def distances_from(self, spawn: Pose2D) -> np.ndarray:
        """Return minimum staged action count, or ``-1`` when unreachable."""

        distances = np.full((self.yaw_bins, *self.shape), -1, dtype=np.int32)
        if not self.checker.is_pose_feasible(spawn):
            return distances
        start = self.snap_pose(spawn)
        if start is None:
            return distances
        queue: deque[tuple[int, int, int]] = deque([start])
        distances[start] = 0
        while queue:
            state = queue.popleft()
            next_distance = int(distances[state]) + 1
            for neighbor, _action in self._neighbors(state):
                if distances[neighbor] >= 0:
                    continue
                distances[neighbor] = next_distance
                queue.append(neighbor)
        return distances

    def action_counts_to(
        self,
        spawn: Pose2D,
        distances: np.ndarray,
        goal: tuple[int, int, int],
    ) -> Mapping[str, int]:
        """Reconstruct one deterministic shortest path for audit evidence."""

        start = self.snap_pose(spawn)
        if start is None or distances[goal] < 0:
            raise ValueError("goal is not reachable from spawn")
        current = goal
        reversed_actions: list[str] = []
        while current != start:
            current_distance = int(distances[current])
            predecessors: list[tuple[tuple[int, int, int], str]] = []
            # The graph is bidirectional; enumerate candidates and verify that
            # their forward edge reaches the current state.
            heading, x, y = current
            candidate_states = (
                ((heading - 1) % self.yaw_bins, x, y),
                ((heading + 1) % self.yaw_bins, x, y),
            )
            step = self._grid_step_for_heading(heading)
            if step is not None:
                dx, dy = step
                candidate_states += (
                    (heading, x - dx, y - dy),
                    (heading, x + dx, y + dy),
                )
            for candidate in candidate_states:
                if not (
                    0 <= candidate[1] < self.shape[0]
                    and 0 <= candidate[2] < self.shape[1]
                    and distances[candidate] == current_distance - 1
                ):
                    continue
                for neighbor, action in self._neighbors(candidate):
                    if neighbor == current:
                        predecessors.append((candidate, action))
            if not predecessors:
                raise AssertionError("reachable state has no shortest-path predecessor")
            current, action = min(predecessors, key=lambda item: (item[1], item[0]))
            reversed_actions.append(action)
        counts = Counter(reversed(reversed_actions))
        return {name: int(counts.get(name, 0)) for name in (
            "forward",
            "reverse",
            "turn_left",
            "turn_right",
        )}


@dataclass(frozen=True)
class ClaimAnchorWitness:
    object_id: str
    target_xy_m: tuple[float, float]
    reachable: bool
    reachable_claim_state_count: int
    anchor_pose: tuple[float, float, float] | None
    anchor_lattice_state: tuple[int, int, int] | None
    anchor_target_distance_m: float | None
    anchor_has_line_of_sight: bool
    shortest_staged_action_count: int | None
    shortest_staged_action_counts: Mapping[str, int] | None
    physical_claim_decision: str | None = None
    physical_claim_credited: bool = False
    physical_claim_unverifiable_reasons: tuple[str, ...] = ()
    physical_claim_rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalEligibilityReport:
    scene_id: str
    family: str
    policy_content_sha256: str
    policy_profile: str
    config: PhysicalEligibilityConfig
    spawn_pose: tuple[float, float, float]
    spawn_clear_at_actual_yaw: bool
    spawn_snaps_to_lattice: bool
    reachable_pose_state_count: int
    reachable_center_cell_count: int
    claim_anchors: tuple[ClaimAnchorWitness, ...]
    canonical_physical_claim_trace: Mapping[str, Any]
    eligible: bool
    failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ELIGIBILITY_SCHEMA,
            "scene_id": self.scene_id,
            "family": self.family,
            "policy_content_sha256": self.policy_content_sha256,
            "policy_profile": self.policy_profile,
            "config": asdict(self.config),
            "spawn_pose": list(self.spawn_pose),
            "spawn_clear_at_actual_yaw": self.spawn_clear_at_actual_yaw,
            "spawn_snaps_to_lattice": self.spawn_snaps_to_lattice,
            "reachable_pose_state_count": self.reachable_pose_state_count,
            "reachable_center_cell_count": self.reachable_center_cell_count,
            "claim_anchors": [anchor.to_dict() for anchor in self.claim_anchors],
            "canonical_physical_claim_trace": dict(
                self.canonical_physical_claim_trace
            ),
            "eligible": self.eligible,
            "failure_reason": self.failure_reason,
        }

    @property
    def sha256(self) -> str:
        return _payload_sha256(self.to_dict())


def _claim_anchor_witnesses(
    manifest: SceneManifest,
    lattice: DirectionalSE2Lattice,
    distances: np.ndarray,
    spawn: Pose2D,
    *,
    config: PhysicalEligibilityConfig,
) -> tuple[ClaimAnchorWitness, ...]:
    scene = SceneGraph(manifest)
    reachable = distances >= 0
    reports: list[ClaimAnchorWitness] = []
    for landmark in sorted(manifest.landmarks, key=lambda item: item.object_id):
        target_xy = (
            float(landmark.center_xyz_m[0]),
            float(landmark.center_xyz_m[1]),
        )
        candidates: list[
            tuple[
                tuple[float, float, float, int, int, int],
                tuple[int, int, int],
                float,
            ]
        ] = []
        center_indices = np.argwhere(np.any(reachable, axis=0))
        for raw_x, raw_y in center_indices:
            x, y = int(raw_x), int(raw_y)
            world_xy = lattice.cell_to_world((x, y))
            target_distance = math.dist(world_xy, target_xy)
            if target_distance > config.claim_radius_m:
                continue
            if config.require_line_of_sight and not scene.has_line_of_sight(
                world_xy,
                target_xy,
                exclude_landmark_xy=target_xy,
            ):
                continue
            for raw_heading in np.flatnonzero(reachable[:, x, y]):
                heading = int(raw_heading)
                state = (heading, x, y)
                pose = lattice.pose_for_state(state)
                target_bearing = math.atan2(
                    target_xy[1] - pose.y_m,
                    target_xy[0] - pose.x_m,
                )
                bearing_error = abs(
                    math.atan2(
                        math.sin(target_bearing - pose.yaw_rad),
                        math.cos(target_bearing - pose.yaw_rad),
                    )
                )
                if bearing_error > CLAIM_ABSOLUTE_BEARING_RAD:
                    continue
                candidates.append(
                    (
                        (
                            abs(target_distance - config.preferred_standoff_m),
                            bearing_error,
                            target_distance,
                            x,
                            y,
                            heading,
                        ),
                        state,
                        target_distance,
                    )
                )
        if not candidates:
            reports.append(
                ClaimAnchorWitness(
                    object_id=str(landmark.object_id),
                    target_xy_m=target_xy,
                    reachable=False,
                    reachable_claim_state_count=0,
                    anchor_pose=None,
                    anchor_lattice_state=None,
                    anchor_target_distance_m=None,
                    anchor_has_line_of_sight=False,
                    shortest_staged_action_count=None,
                    shortest_staged_action_counts=None,
                )
            )
            continue
        _rank, state, target_distance = min(candidates, key=lambda item: item[0])
        pose = lattice.pose_for_state(state)
        action_counts = lattice.action_counts_to(spawn, distances, state)
        reports.append(
            ClaimAnchorWitness(
                object_id=str(landmark.object_id),
                target_xy_m=target_xy,
                reachable=True,
                reachable_claim_state_count=len(candidates),
                anchor_pose=(pose.x_m, pose.y_m, pose.yaw_rad),
                anchor_lattice_state=state,
                anchor_target_distance_m=float(target_distance),
                anchor_has_line_of_sight=True,
                shortest_staged_action_count=int(distances[state]),
                shortest_staged_action_counts=action_counts,
            )
        )
    return tuple(reports)


def evaluate_physical_claim_trace_eligibility_adapter(
    trace: Mapping[str, Any],
    physical_manifest: SceneManifest,
    expected_task_object_ids: tuple[str, ...],
    expected_task_object_set_sha256: str,
) -> dict[str, Any]:
    """Canonical evaluator boundary used by privileged eligibility auditing."""

    return evaluate_physical_claim_trace(
        trace,
        physical_manifest,
        expected_task_object_ids,
        expected_task_object_set_sha256,
    )


def _evaluate_anchor_witnesses(
    manifest: SceneManifest,
    anchors: tuple[ClaimAnchorWitness, ...],
) -> tuple[tuple[ClaimAnchorWitness, ...], dict[str, Any]]:
    trace_id = f"eligibility:{manifest.scene_id}"
    episode_id = f"eligibility:{manifest.scene_id}:episode"
    attempts = []
    for anchor in anchors:
        if anchor.anchor_pose is None:
            continue
        reference = object_id_reference(anchor.object_id)
        attempts.append(
            build_claim_attempt(
                manifest=manifest,
                trace_id=trace_id,
                episode_id=episode_id,
                event_id=f"eligibility:{manifest.scene_id}:{anchor.object_id}",
                tick=len(attempts),
                event_index=len(attempts),
                requested_target=reference,
                claimed_target=reference,
                robot_pose_world_xy_yaw=anchor.anchor_pose,
                pose_provenance="eligibility_candidate_full_precision",
            )
        )
    raw_trace, task_ids, task_hash = build_claim_trace(
        manifest=manifest,
        trace_id=trace_id,
        episode_id=episode_id,
        controller_claim_attempts=attempts,
    )
    evaluated = evaluate_physical_claim_trace_eligibility_adapter(
        raw_trace,
        manifest,
        task_ids,
        task_hash,
    )
    by_object = {
        item.get("claimed_target_object_id"): item
        for item in evaluated["physical_claim_evaluations"]
        if item.get("claimed_target_object_id") is not None
    }
    bound = tuple(
        replace(
            anchor,
            physical_claim_decision=(
                None
                if anchor.object_id not in by_object
                else str(by_object[anchor.object_id]["decision"])
            ),
            physical_claim_credited=bool(
                by_object.get(anchor.object_id, {}).get("credited", False)
            ),
            physical_claim_unverifiable_reasons=tuple(
                by_object.get(anchor.object_id, {}).get("unverifiable_reasons", ())
            ),
            physical_claim_rejection_reasons=tuple(
                by_object.get(anchor.object_id, {}).get("rejection_reasons", ())
            ),
        )
        for anchor in anchors
    )
    return bound, evaluated


def audit_physical_scene_eligibility(
    manifest: SceneManifest,
    *,
    policy: LoadedDirectionalPolicy,
    config: PhysicalEligibilityConfig,
) -> PhysicalEligibilityReport:
    """Audit actual-yaw spawn and staged polygon reachability to all beacons."""

    config.validate()
    checker = ManifestDirectionalFootprintFeasibility(
        manifest,
        policy.footprint,
    )
    spawn = Pose2D(
        float(manifest.spawn.xyz_m[0]),
        float(manifest.spawn.xyz_m[1]),
        yaw_from_wxyz(manifest.spawn.quat_wxyz),
    )
    spawn_clear = checker.is_pose_feasible(spawn)
    lattice = DirectionalSE2Lattice(
        checker,
        cell_size_m=config.cell_size_m,
        yaw_bins=config.yaw_bins,
        rotation_subsamples=config.rotation_subsamples,
        maximum_translation_substep_m=config.maximum_translation_substep_m,
    )
    lattice.validate_masks(samples=config.mask_validation_samples)
    snapped_spawn = lattice.snap_pose(spawn)
    distances = lattice.distances_from(spawn)
    anchors = _claim_anchor_witnesses(
        manifest,
        lattice,
        distances,
        spawn,
        config=config,
    )
    anchors, evaluated_claim_trace = _evaluate_anchor_witnesses(manifest, anchors)
    failures: list[str] = []
    if len(manifest.landmarks) != config.required_beacon_count:
        failures.append(
            f"beacon_count:{len(manifest.landmarks)}!={config.required_beacon_count}"
        )
    if not spawn_clear:
        failures.append("actual_yaw_spawn_not_polygon_clear")
    elif snapped_spawn is None:
        failures.append("actual_yaw_spawn_cannot_enter_se2_lattice")
    unreachable = [anchor.object_id for anchor in anchors if not anchor.reachable]
    if unreachable:
        failures.append("claim_anchors_unreachable:" + ",".join(unreachable))
    physical_summary = evaluated_claim_trace["physical_claim_summary"]
    if not bool(physical_summary["all_targets_claimed"]):
        failed_claims = [
            anchor.object_id
            for anchor in anchors
            if not anchor.physical_claim_credited
        ]
        failures.append("canonical_physical_claims_failed:" + ",".join(failed_claims))
    eligible = not failures
    return PhysicalEligibilityReport(
        scene_id=str(manifest.scene_id),
        family=str(manifest.family),
        policy_content_sha256=policy.content_sha256,
        policy_profile=policy.profile_name,
        config=config,
        spawn_pose=(spawn.x_m, spawn.y_m, spawn.yaw_rad),
        spawn_clear_at_actual_yaw=spawn_clear,
        spawn_snaps_to_lattice=snapped_spawn is not None,
        reachable_pose_state_count=int(np.count_nonzero(distances >= 0)),
        reachable_center_cell_count=int(np.count_nonzero(np.any(distances >= 0, axis=0))),
        claim_anchors=anchors,
        canonical_physical_claim_trace=evaluated_claim_trace,
        eligible=eligible,
        failure_reason=";".join(failures),
    )


def physical_config_from_geometry_contract(
    contract: GeometryContract,
    *,
    yaw_bins: int = 16,
    rotation_subsamples: int = 5,
    mask_validation_samples: int = 32,
) -> PhysicalEligibilityConfig:
    """Derive the exact audit without duplicating geometry-v2 task constants."""

    if contract.schema != REQUIRED_GEOMETRY_SCHEMA:
        raise ValueError("physical eligibility config requires geometry contract v2")
    return PhysicalEligibilityConfig(
        cell_size_m=float(contract.configuration_space.oracle_cell_size_m),
        maximum_translation_substep_m=float(
            contract.kinematic_execution.maximum_translation_substep_m
        ),
        yaw_bins=int(yaw_bins),
        rotation_subsamples=int(rotation_subsamples),
        claim_radius_m=float(contract.visibility_and_claim.claim_radius_m),
        preferred_standoff_m=float(contract.visibility_and_claim.standoff_m),
        require_line_of_sight=bool(
            contract.visibility_and_claim.require_line_of_sight_for_scene_validity
        ),
        required_beacon_count=4,
        mask_validation_samples=int(mask_validation_samples),
    )


__all__ = [
    "ClaimAnchorWitness",
    "DirectionalSE2Lattice",
    "LoadedDirectionalPolicy",
    "PhysicalEligibilityConfig",
    "PhysicalEligibilityReport",
    "audit_physical_scene_eligibility",
    "directional_footprint_semantic_sha256",
    "evaluate_physical_claim_trace_eligibility_adapter",
    "load_observed_max_directional_policy",
    "physical_config_from_geometry_contract",
    "policy_from_geometry_contract",
    "validate_loaded_directional_policy_content",
    "yaw_from_wxyz",
]
