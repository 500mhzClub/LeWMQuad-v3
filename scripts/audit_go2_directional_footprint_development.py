#!/usr/bin/env python3
"""Audit calibrated directional Go2 footprints on v3 development scenes."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for package_root in (ROOT, ROOT / "lewm_worlds"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from lewm.planning.oriented_footprint import (  # noqa: E402
    DirectionalSupportFootprint,
    ManifestDirectionalFootprintFeasibility,
    Pose2D,
)
from lewm_worlds.manifest import SceneManifest, parse_scene_manifest_dict  # noqa: E402
from scripts.audit_go2_oriented_footprint_development import (  # noqa: E402
    DEFAULT_DEVELOPMENT_MANIFEST,
    DEFAULT_SCENE_ROOT,
    _claimable_beacons,
    _load_json,
    _scene_paths,
    _topology_implications,
    _yaw_from_wxyz,
)


DEFAULT_OUTPUT = (
    ROOT
    / ".generated/go2_footprint_calibration/directional_footprint_development_v3_audit.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _discover_policy() -> Path:
    candidates = sorted(
        (ROOT / ".generated/go2_footprint_calibration").glob(
            "go2_directional_footprint_policy_v1_*.json"
        )
    )
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one directional footprint policy artifact; "
            f"found {len(candidates)}. Pass --policy explicitly."
        )
    return candidates[0]


def _load_policy(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != "lewm_go2_directional_footprint_policy_v1":
        raise ValueError("unsupported directional footprint policy schema")
    declared = str(payload.get("content_sha256", ""))
    content = dict(payload)
    content.pop("content_sha256", None)
    actual = _canonical_sha256(content)
    if declared != actual:
        raise ValueError(
            f"directional footprint content hash mismatch: {declared=} {actual=}"
        )
    if declared not in path.name:
        raise ValueError("directional footprint filename is not content-addressed")
    return payload


def _footprint_from_profile(profile: dict[str, Any]) -> DirectionalSupportFootprint:
    support_planes = profile.get("support_planes")
    vertices = profile.get("vertices_xy_body_m")
    if not isinstance(support_planes, list) or not isinstance(vertices, list):
        raise ValueError("directional profile is missing support planes or vertices")
    footprint = DirectionalSupportFootprint.from_directional_support(
        {
            float(plane["angle_deg"]): float(plane["raw_support_m"])
            for plane in support_planes
        },
        margin_m=float(profile["margin_m"]),
    )
    artifact_vertices = tuple(
        (float(vertex[0]), float(vertex[1])) for vertex in vertices
    )
    if len(artifact_vertices) != len(footprint.vertices_xy_m) or any(
        math.dist(actual, expected) > 1e-10
        for actual, expected in zip(
            footprint.vertices_xy_m,
            artifact_vertices,
            strict=True,
        )
    ):
        raise ValueError("directional profile vertices do not match its support planes")
    return footprint


class _DirectionalLattice:
    """Exact polygon SAT masks and swept-yaw reachability on an SE(2) lattice."""

    def __init__(
        self,
        checker: ManifestDirectionalFootprintFeasibility,
        *,
        cell_size_m: float,
        yaw_bins: int,
        rotation_subsamples: int,
    ) -> None:
        if cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        if yaw_bins < 8 or yaw_bins % 4:
            raise ValueError("yaw_bins must be a multiple of four and at least eight")
        if rotation_subsamples < 1:
            raise ValueError("rotation_subsamples must be positive")
        self.checker = checker
        self.cell_size_m = float(cell_size_m)
        self.yaw_bins = int(yaw_bins)
        self.rotation_subsamples = int(rotation_subsamples)
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

    def _free_mask(self, yaw_rad: float) -> np.ndarray:
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
        free = np.ones(self.shape, dtype=bool)
        for relative_x, relative_y in relative_vertices:
            vertex_x = self.grid_x + relative_x
            vertex_y = self.grid_y + relative_y
            free &= (
                (vertex_x >= x_min - epsilon)
                & (vertex_x <= x_max + epsilon)
                & (vertex_y >= y_min - epsilon)
                & (vertex_y <= y_max + epsilon)
            )

        polygon_axes: list[np.ndarray] = []
        for first, second in zip(
            relative_vertices,
            np.concatenate((relative_vertices[1:], relative_vertices[:1]), axis=0),
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
            base_x_min = float(np.min(obstacle_vertices[:, 0]) - np.max(relative_vertices[:, 0]))
            base_x_max = float(np.max(obstacle_vertices[:, 0]) - np.min(relative_vertices[:, 0]))
            base_y_min = float(np.min(obstacle_vertices[:, 1]) - np.max(relative_vertices[:, 1]))
            base_y_max = float(np.max(obstacle_vertices[:, 1]) - np.min(relative_vertices[:, 1]))
            x_values = self.grid_x[:, 0]
            y_values = self.grid_y[0, :]
            x_start = int(np.searchsorted(x_values, base_x_min - epsilon, side="left"))
            x_stop = int(np.searchsorted(x_values, base_x_max + epsilon, side="right"))
            y_start = int(np.searchsorted(y_values, base_y_min - epsilon, side="left"))
            y_stop = int(np.searchsorted(y_values, base_y_max + epsilon, side="right"))
            if x_start >= x_stop or y_start >= y_stop:
                continue
            grid_x = self.grid_x[x_start:x_stop, y_start:y_stop]
            grid_y = self.grid_y[x_start:x_stop, y_start:y_stop]
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
        return int(round(float(yaw_rad) * self.yaw_bins / (2.0 * math.pi))) % self.yaw_bins

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

    def _neighbors(
        self,
        state: tuple[int, int, int],
    ) -> Iterable[tuple[int, int, int]]:
        heading, x, y = state
        for rotation_direction in (-1, 1):
            next_heading = (heading + rotation_direction) % self.yaw_bins
            if self._rotation_free(heading, x, y, rotation_direction):
                yield (next_heading, x, y)
        angle = 2.0 * math.pi * heading / self.yaw_bins
        dx = int(round(math.cos(angle)))
        dy = int(round(math.sin(angle)))
        for direction in (-1, 1):
            next_x = x + direction * dx
            next_y = y + direction * dy
            if not self._state_free(heading, next_x, next_y):
                continue
            if dx and dy and not (
                self._state_free(heading, x + direction * dx, y)
                and self._state_free(heading, x, y + direction * dy)
            ):
                continue
            yield (heading, next_x, next_y)

    def snap_pose(self, pose: Pose2D) -> tuple[int, int, int] | None:
        heading = self.heading_index(pose.yaw_rad)
        x, y = self.nearest_cell(pose.x_m, pose.y_m)
        if not self._state_free(heading, x, y):
            return None
        snapped = self.pose_for_state((heading, x, y))
        if not self.checker.is_swept_pose_feasible(pose, snapped):
            return None
        return (heading, x, y)

    def validate_masks(self, *, samples: int = 48) -> None:
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
                    f"vectorized polygon SAT mismatch at {pose}: {actual=} {expected=}"
                )

    def reachable_from(self, spawn: Pose2D) -> np.ndarray:
        seen = np.zeros((self.yaw_bins, *self.shape), dtype=bool)
        if not self.checker.is_pose_feasible(spawn):
            return seen
        start = self.snap_pose(spawn)
        if start is None:
            return seen
        queue: deque[tuple[int, int, int]] = deque([start])
        seen[start] = True
        while queue:
            state = queue.popleft()
            for neighbor in self._neighbors(state):
                if seen[neighbor]:
                    continue
                seen[neighbor] = True
                queue.append(neighbor)
        return seen

    def local_reachable_from(
        self,
        start_pose: Pose2D,
        *,
        center_xy_m: tuple[float, float],
        radius_m: float,
    ) -> np.ndarray:
        seen = np.zeros((self.yaw_bins, *self.shape), dtype=bool)
        start = self.snap_pose(start_pose)
        if start is None:
            return seen

        def within_local(state: tuple[int, int, int]) -> bool:
            _, x, y = state
            world_x, world_y = self.cell_to_world((x, y))
            return (
                abs(world_x - center_xy_m[0]) <= radius_m
                and abs(world_y - center_xy_m[1]) <= radius_m
            )

        if not within_local(start):
            return seen
        queue: deque[tuple[int, int, int]] = deque([start])
        seen[start] = True
        while queue:
            state = queue.popleft()
            for neighbor in self._neighbors(state):
                if seen[neighbor] or not within_local(neighbor):
                    continue
                seen[neighbor] = True
                queue.append(neighbor)
        return seen


def _staged_turn_implications(
    manifest: SceneManifest,
    checker: ManifestDirectionalFootprintFeasibility,
    lattice: _DirectionalLattice,
    *,
    staging_distance_m: float,
) -> dict[str, int | float]:
    nodes = {node.node_id: node for node in manifest.graph_nodes}
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    for edge in manifest.graph_edges:
        if edge.traversable:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)

    turn_total = 0
    endpoint_feasible = 0
    staged_feasible = 0
    for node_id, neighbor_ids in adjacency.items():
        if len(neighbor_ids) < 2:
            continue
        node_xy = nodes[node_id].center_xy_m
        incident_distances = [
            math.dist(node_xy, nodes[neighbor_id].center_xy_m)
            for neighbor_id in neighbor_ids
        ]
        local_radius = 0.45 * min(incident_distances)
        ports: dict[int, Pose2D] = {}
        for neighbor_id in neighbor_ids:
            neighbor_xy = nodes[neighbor_id].center_xy_m
            delta_x = neighbor_xy[0] - node_xy[0]
            delta_y = neighbor_xy[1] - node_xy[1]
            distance = math.hypot(delta_x, delta_y)
            direction_x, direction_y = delta_x / distance, delta_y / distance
            ports[neighbor_id] = Pose2D(
                node_xy[0] + staging_distance_m * direction_x,
                node_xy[1] + staging_distance_m * direction_y,
                math.atan2(direction_y, direction_x),
            )

        for incoming_id in neighbor_ids:
            incoming_neighbor_xy = nodes[incoming_id].center_xy_m
            incoming_direction = (
                (node_xy[0] - incoming_neighbor_xy[0]) / incident_distances[
                    neighbor_ids.index(incoming_id)
                ],
                (node_xy[1] - incoming_neighbor_xy[1]) / incident_distances[
                    neighbor_ids.index(incoming_id)
                ],
            )
            start_pose = Pose2D(
                node_xy[0] - staging_distance_m * incoming_direction[0],
                node_xy[1] - staging_distance_m * incoming_direction[1],
                math.atan2(incoming_direction[1], incoming_direction[0]),
            )
            local_seen: np.ndarray | None = None
            for outgoing_id in neighbor_ids:
                if outgoing_id == incoming_id:
                    continue
                end_pose = ports[outgoing_id]
                delta_yaw = abs(
                    (end_pose.yaw_rad - start_pose.yaw_rad + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                if not math.isclose(delta_yaw, math.pi / 2.0, abs_tol=1e-6):
                    continue
                turn_total += 1
                if not (
                    checker.is_pose_feasible(start_pose)
                    and checker.is_pose_feasible(end_pose)
                ):
                    continue
                start_state = lattice.snap_pose(start_pose)
                end_state = lattice.snap_pose(end_pose)
                if start_state is None or end_state is None:
                    continue
                endpoint_feasible += 1
                if local_seen is None:
                    local_seen = lattice.local_reachable_from(
                        start_pose,
                        center_xy_m=node_xy,
                        radius_m=local_radius,
                    )
                if local_seen[end_state] and checker.is_swept_pose_feasible(
                    lattice.pose_for_state(end_state),
                    end_pose,
                ):
                    staged_feasible += 1
    return {
        "staging_distance_m": staging_distance_m,
        "ordered_quarter_turns": turn_total,
        "turns_with_feasible_staging_endpoints": endpoint_feasible,
        "staged_turns_feasible": staged_feasible,
        "staged_turns_blocked": endpoint_feasible - staged_feasible,
    }


def _audit_profile(
    profile_name: str,
    footprint: DirectionalSupportFootprint,
    manifests: tuple[SceneManifest, ...],
    *,
    audit_config: dict[str, Any],
    cell_size_m: float,
    yaw_bins: int,
    rotation_subsamples: int,
    staging_distance_m: float,
) -> dict[str, Any]:
    per_scene: list[dict[str, Any]] = []
    for scene_index, manifest in enumerate(manifests, start=1):
        checker = ManifestDirectionalFootprintFeasibility(manifest, footprint)
        lattice = _DirectionalLattice(
            checker,
            cell_size_m=cell_size_m,
            yaw_bins=yaw_bins,
            rotation_subsamples=rotation_subsamples,
        )
        lattice.validate_masks()
        spawn = Pose2D(
            float(manifest.spawn.xyz_m[0]),
            float(manifest.spawn.xyz_m[1]),
            _yaw_from_wxyz(manifest.spawn.quat_wxyz),
        )
        reachable_states = lattice.reachable_from(spawn)
        reachable_centers = np.any(reachable_states, axis=0)
        beacons = _claimable_beacons(
            manifest,
            reachable_centers,
            lattice,
            claim_radius_m=float(audit_config["claim_radius_m"]),
            require_line_of_sight=bool(audit_config["require_line_of_sight"]),
        )
        centerline = _topology_implications(manifest, checker)
        staged = _staged_turn_implications(
            manifest,
            checker,
            lattice,
            staging_distance_m=staging_distance_m,
        )
        scene_report = {
            "scene_id": manifest.scene_id,
            "spawn_clear": checker.is_pose_feasible(spawn),
            "fully_claimable": bool(beacons)
            and all(beacon["claimable"] for beacon in beacons),
            "claimable_beacons": sum(beacon["claimable"] for beacon in beacons),
            "reachable_center_cells": int(np.count_nonzero(reachable_centers)),
            "reachable_pose_states": int(np.count_nonzero(reachable_states)),
            "beacons": list(beacons),
            "straight_and_centerline_turns": centerline,
            "staged_turns": staged,
        }
        per_scene.append(scene_report)
        print(
            f"[{profile_name} {scene_index:02d}/{len(manifests)}] "
            f"{manifest.scene_id}: spawn={scene_report['spawn_clear']} "
            f"claims={scene_report['claimable_beacons']}/{len(beacons)} "
            f"centers={scene_report['reachable_center_cells']}",
            flush=True,
        )

    centerline_keys = per_scene[0]["straight_and_centerline_turns"]
    staged_count_keys = (
        "ordered_quarter_turns",
        "turns_with_feasible_staging_endpoints",
        "staged_turns_feasible",
        "staged_turns_blocked",
    )
    return {
        "profile_name": profile_name,
        "footprint": {
            "vertex_count": len(footprint.vertices_xy_m),
            "vertices_xy_body_m": [list(vertex) for vertex in footprint.vertices_xy_m],
            "maximum_vertex_radius_m": footprint.maximum_vertex_radius_m,
            "margin_m": footprint.margin_m,
        },
        "spawn_clear_scene_count": sum(scene["spawn_clear"] for scene in per_scene),
        "fully_claimable_scene_count": sum(
            scene["fully_claimable"] for scene in per_scene
        ),
        "claimable_beacon_count": sum(
            scene["claimable_beacons"] for scene in per_scene
        ),
        "reachable_center_cell_count": sum(
            scene["reachable_center_cells"] for scene in per_scene
        ),
        "straight_and_centerline_turn_totals": {
            key: sum(
                scene["straight_and_centerline_turns"][key] for scene in per_scene
            )
            for key in centerline_keys
        },
        "staged_turn_totals": {
            "staging_distance_m": staging_distance_m,
            **{
                key: sum(scene["staged_turns"][key] for scene in per_scene)
                for key in staged_count_keys
            },
        },
        "per_scene": per_scene,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    policy_path = (args.policy or _discover_policy()).resolve()
    policy = _load_policy(policy_path)
    development_path = args.development_manifest.resolve()
    scene_root = args.scene_root.resolve()
    development = _load_json(development_path)
    if development.get("schema") != "lewm_navigation_development_manifest_v0":
        raise ValueError("unsupported development manifest schema")
    scene_records = _scene_paths(development, scene_root)
    manifests = tuple(
        parse_scene_manifest_dict(_load_json(path)) for _record, path in scene_records
    )
    profiles = {
        name: _footprint_from_profile(profile)
        for name, profile in policy["profiles"].items()
    }
    audits = {
        name: _audit_profile(
            name,
            footprint,
            manifests,
            audit_config=development["audit_config"],
            cell_size_m=args.cell_size_m,
            yaw_bins=args.yaw_bins,
            rotation_subsamples=args.rotation_subsamples,
            staging_distance_m=args.staging_distance_m,
        )
        for name, footprint in profiles.items()
    }
    report = {
        "schema": "lewm_go2_directional_footprint_development_audit_v1",
        "scope": {
            "split": "validation",
            "scene_count": len(manifests),
            "beacon_count": sum(len(manifest.landmarks) for manifest in manifests),
            "sealed_test_accessed": False,
        },
        "lattice": {
            "cell_size_m": args.cell_size_m,
            "yaw_bins": args.yaw_bins,
            "rotation_subsamples": args.rotation_subsamples,
            "maximum_rotation_sample_step_deg": 360.0
            / (args.yaw_bins * args.rotation_subsamples),
            "translation_model": "fore_aft_only_with_no_diagonal_corner_cutting",
        },
        "recommended_profile": policy["recommended_profile"],
        "audits": audits,
        "source_artifacts": {
            "directional_policy": {
                "path": str(policy_path),
                "file_sha256": _sha256(policy_path),
                "content_sha256": policy["content_sha256"],
            },
            "development_manifest": {
                "path": str(development_path),
                "sha256": _sha256(development_path),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__)),
            },
            "geometry_helper": {
                "path": str(ROOT / "lewm/planning/oriented_footprint.py"),
                "sha256": _sha256(ROOT / "lewm/planning/oriented_footprint.py"),
            },
        },
    }
    content = dict(report)
    report["content_sha256"] = _canonical_sha256(content)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--yaw-bins", type=int, default=16)
    parser.add_argument("--rotation-subsamples", type=int, default=5)
    parser.add_argument("--staging-distance-m", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    for name, audit in report["audits"].items():
        print(
            f"{name}: spawn={audit['spawn_clear_scene_count']}/{report['scope']['scene_count']} "
            f"scenes={audit['fully_claimable_scene_count']}/{report['scope']['scene_count']} "
            f"beacons={audit['claimable_beacon_count']}/{report['scope']['beacon_count']} "
            f"staged_turns={audit['staged_turn_totals']['staged_turns_feasible']}/"
            f"{audit['staged_turn_totals']['turns_with_feasible_staging_endpoints']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
