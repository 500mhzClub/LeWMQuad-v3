#!/usr/bin/env python3
"""Compare calibrated oriented and circular footprints on v3 development mazes."""

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
    AsymmetricFootprint,
    ManifestFootprintFeasibility,
    Pose2D,
)
from lewm_worlds.fixed_spawn_audit import (  # noqa: E402
    FixedSpawnAuditConfig,
    audit_fixed_spawn,
)
from lewm_worlds.manifest import (  # noqa: E402
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402


DEFAULT_DEVELOPMENT_MANIFEST = ROOT / "config/go2_generalization_v3/development.json"
DEFAULT_SCENE_ROOT = ROOT / ".generated/scene_corpus/go2_generalization_v3/development"
DEFAULT_CALIBRATION = (
    ROOT / ".generated/go2_footprint_calibration/geometry_v1_calibration.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / ".generated/go2_footprint_calibration/development_v3_geometry_comparison.json"
)
DEFAULT_DISC_RADII_M = (0.20, 0.25, 0.46)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _yaw_from_wxyz(quaternion: Iterable[float]) -> float:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _scene_paths(
    development_manifest: dict[str, Any],
    scene_root: Path,
) -> tuple[tuple[dict[str, Any], Path], ...]:
    records = development_manifest.get("validation_scenes")
    if not isinstance(records, list) or not records:
        raise ValueError("development manifest has no validation_scenes")
    resolved: list[tuple[dict[str, Any], Path]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("validation_scenes entries must be objects")
        record = dict(raw_record)
        family = str(record["family"])
        scene_id = str(record["scene_id"])
        path = scene_root / family / scene_id / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        declared_hash = str(record.get("manifest_sha256", ""))
        payload = _load_json(path)
        actual_hash = manifest_sha256(parse_scene_manifest_dict(payload))
        if declared_hash and declared_hash != actual_hash:
            raise ValueError(
                f"development manifest hash mismatch for {scene_id}: "
                f"expected {declared_hash}, got {actual_hash}"
            )
        resolved.append((record, path))
    return tuple(resolved)


def _disc_config(
    audit_config: dict[str, Any],
    radius_m: float,
) -> FixedSpawnAuditConfig:
    values = dict(audit_config)
    values["body_radius_m"] = float(radius_m)
    return FixedSpawnAuditConfig(**values)


def _summarize_disc(
    manifests: tuple[SceneManifest, ...],
    *,
    audit_config: dict[str, Any],
    radius_m: float,
) -> dict[str, Any]:
    config = _disc_config(audit_config, radius_m)
    reports = tuple(audit_fixed_spawn(manifest, config=config) for manifest in manifests)
    per_scene = []
    for report in reports:
        per_scene.append(
            {
                "scene_id": report.scene_id,
                "spawn_clear": report.spawn_is_body_clear,
                "fully_claimable": report.fully_reachable,
                "claimable_beacons": sum(
                    beacon.claim_reachable for beacon in report.beacons
                ),
                "preferred_standoff_beacons": sum(
                    beacon.preferred_standoff_reachable for beacon in report.beacons
                ),
                "reachable_center_cells": report.reachable_cell_count,
                "failure_reason": report.failure_reason,
                "unclaimable_beacon_ids": [
                    beacon.object_id
                    for beacon in report.beacons
                    if not beacon.claim_reachable
                ],
            }
        )
    return {
        "model": "canonical_circular_inflation",
        "radius_m": float(radius_m),
        "fully_claimable_scene_count": sum(
            report.fully_reachable for report in reports
        ),
        "spawn_clear_scene_count": sum(
            report.spawn_is_body_clear for report in reports
        ),
        "claimable_beacon_count": sum(
            beacon.claim_reachable for report in reports for beacon in report.beacons
        ),
        "preferred_standoff_beacon_count": sum(
            beacon.preferred_standoff_reachable
            for report in reports
            for beacon in report.beacons
        ),
        "reachable_center_cell_count": sum(
            report.reachable_cell_count for report in reports
        ),
        "per_scene": per_scene,
    }


class _OrientedLattice:
    """Rasterized exact-SAT poses plus swept rotations on an SE(2) lattice."""

    def __init__(
        self,
        checker: ManifestFootprintFeasibility,
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
        (x_min, y_min), (x_max, y_max) = checker.world_bounds_xy_m
        spawn_x = float(checker.manifest.spawn.xyz_m[0])
        spawn_y = float(checker.manifest.spawn.xyz_m[1])

        def spawn_aligned_origin(lower: float, anchor: float) -> float:
            anchor_index = math.floor((anchor - lower) / self.cell_size_m)
            origin = anchor - (anchor_index + 0.5) * self.cell_size_m
            return origin - self.cell_size_m if origin > lower else origin

        self.origin_xy_m = (
            spawn_aligned_origin(float(x_min), spawn_x),
            spawn_aligned_origin(float(y_min), spawn_y),
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
        self.fine_yaw_bins = self.yaw_bins * self.rotation_subsamples
        self.fine_free = np.stack(
            [
                self._free_mask(2.0 * math.pi * index / self.fine_yaw_bins)
                for index in range(self.fine_yaw_bins)
            ],
            axis=0,
        )

    def _free_mask(self, yaw_rad: float) -> np.ndarray:
        footprint = self.checker.footprint
        local_center_x, local_center_y = footprint.local_center_xy_m
        half_x, half_y = footprint.half_extent_xy_m
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        axis_x = np.asarray((cos_yaw, sin_yaw), dtype=np.float64)
        axis_y = np.asarray((-sin_yaw, cos_yaw), dtype=np.float64)
        center_x = self.grid_x + cos_yaw * local_center_x - sin_yaw * local_center_y
        center_y = self.grid_y + sin_yaw * local_center_x + cos_yaw * local_center_y

        (x_min, y_min), (x_max, y_max) = self.checker.world_bounds_xy_m
        epsilon = self.checker.geometry_epsilon_m
        free = np.ones(self.shape, dtype=bool)
        for body_x, body_y in (
            (footprint.forward_m, footprint.left_m),
            (footprint.forward_m, -footprint.right_m),
            (-footprint.rear_m, footprint.left_m),
            (-footprint.rear_m, -footprint.right_m),
        ):
            corner_x = self.grid_x + cos_yaw * body_x - sin_yaw * body_y
            corner_y = self.grid_y + sin_yaw * body_x + cos_yaw * body_y
            free &= (
                (corner_x >= x_min - epsilon)
                & (corner_x <= x_max + epsilon)
                & (corner_y >= y_min - epsilon)
                & (corner_y <= y_max + epsilon)
            )

        for obstacle in self.checker.collision_boxes:
            obstacle_axis_x, obstacle_axis_y = (
                np.asarray(axis, dtype=np.float64) for axis in obstacle.axes
            )
            delta_x = float(obstacle.center_xy_m[0]) - center_x
            delta_y = float(obstacle.center_xy_m[1]) - center_y

            def projection_distance(axis: np.ndarray) -> np.ndarray:
                return np.abs(delta_x * axis[0] + delta_y * axis[1])

            def projection_radius(
                basis_x: np.ndarray,
                basis_y: np.ndarray,
                extent_x: float,
                extent_y: float,
                axis: np.ndarray,
            ) -> float:
                return float(
                    extent_x * abs(float(np.dot(basis_x, axis)))
                    + extent_y * abs(float(np.dot(basis_y, axis)))
                )

            collision = np.ones(self.shape, dtype=bool)
            for axis in (axis_x, axis_y, obstacle_axis_x, obstacle_axis_y):
                footprint_radius = projection_radius(
                    axis_x,
                    axis_y,
                    half_x,
                    half_y,
                    axis,
                )
                obstacle_radius = projection_radius(
                    obstacle_axis_x,
                    obstacle_axis_y,
                    obstacle.half_extent_x_m,
                    obstacle.half_extent_y_m,
                    axis,
                )
                collision &= projection_distance(axis) <= (
                    footprint_radius + obstacle_radius + epsilon
                )
            free &= ~collision
        return free

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            int(math.floor((float(x_m) - self.origin_xy_m[0]) / self.cell_size_m)),
            int(math.floor((float(y_m) - self.origin_xy_m[1]) / self.cell_size_m)),
        )

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_xy_m[0] + (cell[0] + 0.5) * self.cell_size_m,
            self.origin_xy_m[1] + (cell[1] + 0.5) * self.cell_size_m,
        )

    def heading_index(self, yaw_rad: float) -> int:
        return int(round(float(yaw_rad) * self.yaw_bins / (2.0 * math.pi))) % self.yaw_bins

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

    def validate_masks(self, *, samples: int = 32) -> None:
        """Cross-check vectorized SAT masks against the public exact helper."""

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
                    f"vectorized SAT mismatch at pose={pose}: {actual=} {expected=}"
                )

    def reachable_from(self, spawn: Pose2D) -> np.ndarray:
        """Return reachable heading/cell states using turn and fore-aft moves."""

        seen = np.zeros((self.yaw_bins, *self.shape), dtype=bool)
        if not self.checker.is_pose_feasible(spawn):
            return seen
        heading = self.heading_index(spawn.yaw_rad)
        start_x, start_y = self.world_to_cell(spawn.x_m, spawn.y_m)
        if not self._state_free(heading, start_x, start_y):
            return seen
        snapped_x, snapped_y = self.cell_to_world((start_x, start_y))
        snapped_yaw = 2.0 * math.pi * heading / self.yaw_bins
        if not self.checker.is_swept_pose_feasible(
            spawn,
            Pose2D(snapped_x, snapped_y, snapped_yaw),
        ):
            return seen

        queue: deque[tuple[int, int, int]] = deque([(heading, start_x, start_y)])
        seen[heading, start_x, start_y] = True
        while queue:
            current_heading, x, y = queue.popleft()
            for rotation_direction in (-1, 1):
                next_heading = (current_heading + rotation_direction) % self.yaw_bins
                if (
                    not seen[next_heading, x, y]
                    and self._rotation_free(
                        current_heading,
                        x,
                        y,
                        rotation_direction,
                    )
                ):
                    seen[next_heading, x, y] = True
                    queue.append((next_heading, x, y))

            angle = 2.0 * math.pi * current_heading / self.yaw_bins
            dx = int(round(math.cos(angle)))
            dy = int(round(math.sin(angle)))
            for direction in (-1, 1):
                next_x = x + direction * dx
                next_y = y + direction * dy
                if not (
                    0 <= next_x < self.shape[0]
                    and 0 <= next_y < self.shape[1]
                ):
                    continue
                if seen[current_heading, next_x, next_y]:
                    continue
                if not self._state_free(current_heading, next_x, next_y):
                    continue
                if dx and dy and not (
                    self._state_free(current_heading, x + direction * dx, y)
                    and self._state_free(current_heading, x, y + direction * dy)
                ):
                    continue
                seen[current_heading, next_x, next_y] = True
                queue.append((current_heading, next_x, next_y))
        return seen


def _claimable_beacons(
    manifest: SceneManifest,
    reachable_centers: np.ndarray,
    lattice: _OrientedLattice,
    *,
    claim_radius_m: float,
    require_line_of_sight: bool,
) -> tuple[dict[str, Any], ...]:
    scene = SceneGraph(manifest)
    center_indices = np.argwhere(reachable_centers)
    center_world = np.asarray(
        [lattice.cell_to_world((int(x), int(y))) for x, y in center_indices],
        dtype=np.float64,
    )
    reports: list[dict[str, Any]] = []
    for landmark in manifest.landmarks:
        target = np.asarray(landmark.center_xyz_m[:2], dtype=np.float64)
        if len(center_world):
            distances = np.linalg.norm(center_world - target[None, :], axis=1)
            candidate_indices = np.flatnonzero(distances <= claim_radius_m)
        else:
            distances = np.empty((0,), dtype=np.float64)
            candidate_indices = np.empty((0,), dtype=np.int64)
        claim_cells = 0
        closest = None if not len(distances) else float(np.min(distances))
        for candidate_index in candidate_indices:
            xy = tuple(float(value) for value in center_world[candidate_index])
            if require_line_of_sight and not scene.has_line_of_sight(
                xy,
                tuple(float(value) for value in target),
                exclude_landmark_xy=tuple(float(value) for value in target),
            ):
                continue
            claim_cells += 1
        reports.append(
            {
                "object_id": str(landmark.object_id),
                "claimable": claim_cells > 0,
                "reachable_claim_center_count": claim_cells,
                "closest_reachable_center_m": closest,
            }
        )
    return tuple(reports)


def _topology_implications(
    manifest: SceneManifest,
    checker: ManifestFootprintFeasibility,
) -> dict[str, int]:
    nodes = {node.node_id: node for node in manifest.graph_nodes}
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    directed_edge_total = 0
    directed_edge_endpoint_feasible = 0
    directed_edge_sweep_feasible = 0
    for edge in manifest.graph_edges:
        if not edge.traversable:
            continue
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
        for source_id, target_id in (
            (edge.source, edge.target),
            (edge.target, edge.source),
        ):
            directed_edge_total += 1
            source = nodes[source_id].center_xy_m
            target = nodes[target_id].center_xy_m
            yaw = math.atan2(target[1] - source[1], target[0] - source[0])
            start = Pose2D(source[0], source[1], yaw)
            end = Pose2D(target[0], target[1], yaw)
            if checker.is_pose_feasible(start) and checker.is_pose_feasible(end):
                directed_edge_endpoint_feasible += 1
                if checker.is_swept_pose_feasible(start, end):
                    directed_edge_sweep_feasible += 1

    turn_total = 0
    turn_endpoint_feasible = 0
    turn_sweep_feasible = 0
    for node_id, neighbors in adjacency.items():
        node_xy = nodes[node_id].center_xy_m
        for incoming_id in neighbors:
            incoming_xy = nodes[incoming_id].center_xy_m
            incoming_yaw = math.atan2(
                node_xy[1] - incoming_xy[1],
                node_xy[0] - incoming_xy[0],
            )
            for outgoing_id in neighbors:
                if outgoing_id == incoming_id:
                    continue
                outgoing_xy = nodes[outgoing_id].center_xy_m
                outgoing_yaw = math.atan2(
                    outgoing_xy[1] - node_xy[1],
                    outgoing_xy[0] - node_xy[0],
                )
                delta = abs(
                    (outgoing_yaw - incoming_yaw + math.pi) % (2.0 * math.pi)
                    - math.pi
                )
                if not math.isclose(delta, math.pi / 2.0, abs_tol=1e-6):
                    continue
                turn_total += 1
                start = Pose2D(node_xy[0], node_xy[1], incoming_yaw)
                end = Pose2D(node_xy[0], node_xy[1], outgoing_yaw)
                if checker.is_pose_feasible(start) and checker.is_pose_feasible(end):
                    turn_endpoint_feasible += 1
                    if checker.is_swept_pose_feasible(start, end):
                        turn_sweep_feasible += 1
    return {
        "directed_topology_edges": directed_edge_total,
        "directed_edges_with_feasible_endpoints": directed_edge_endpoint_feasible,
        "directed_edges_swept_feasible": directed_edge_sweep_feasible,
        "centerline_quarter_turns": turn_total,
        "centerline_turns_with_feasible_endpoints": turn_endpoint_feasible,
        "centerline_turns_swept_feasible": turn_sweep_feasible,
        "centerline_turns_blocked_mid_sweep": (
            turn_endpoint_feasible - turn_sweep_feasible
        ),
    }


def _summarize_oriented(
    manifests: tuple[SceneManifest, ...],
    *,
    footprint: AsymmetricFootprint,
    audit_config: dict[str, Any],
    cell_size_m: float,
    yaw_bins: int,
    rotation_subsamples: int,
) -> dict[str, Any]:
    per_scene: list[dict[str, Any]] = []
    for scene_index, manifest in enumerate(manifests, start=1):
        checker = ManifestFootprintFeasibility(manifest, footprint)
        lattice = _OrientedLattice(
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
        beacon_reports = _claimable_beacons(
            manifest,
            reachable_centers,
            lattice,
            claim_radius_m=float(audit_config["claim_radius_m"]),
            require_line_of_sight=bool(audit_config["require_line_of_sight"]),
        )
        topology = _topology_implications(manifest, checker)
        per_scene.append(
            {
                "scene_id": manifest.scene_id,
                "spawn_clear": checker.is_pose_feasible(spawn),
                "fully_claimable": bool(beacon_reports)
                and all(report["claimable"] for report in beacon_reports),
                "claimable_beacons": sum(
                    report["claimable"] for report in beacon_reports
                ),
                "reachable_center_cells": int(np.count_nonzero(reachable_centers)),
                "reachable_pose_states": int(np.count_nonzero(reachable_states)),
                "reachable_center_area_m2": float(
                    np.count_nonzero(reachable_centers) * cell_size_m**2
                ),
                "beacons": list(beacon_reports),
                "topology": topology,
            }
        )
        print(
            f"[{scene_index:02d}/{len(manifests)}] {manifest.scene_id}: "
            f"claims={per_scene[-1]['claimable_beacons']}/{len(beacon_reports)} "
            f"centers={per_scene[-1]['reachable_center_cells']}",
            flush=True,
        )
    topology_totals = {
        key: sum(scene["topology"][key] for scene in per_scene)
        for key in per_scene[0]["topology"]
    }
    return {
        "model": "calibrated_asymmetric_oriented_rectangle",
        "footprint": {
            "forward_m": footprint.forward_m,
            "rear_m": footprint.rear_m,
            "left_m": footprint.left_m,
            "right_m": footprint.right_m,
            "circumscribed_corner_radius_m": footprint.maximum_corner_radius_m,
        },
        "lattice": {
            "cell_size_m": cell_size_m,
            "yaw_bins": yaw_bins,
            "rotation_subsamples_per_heading_step": rotation_subsamples,
            "maximum_rotation_sample_step_deg": 360.0
            / (yaw_bins * rotation_subsamples),
            "translation_model": "fore_aft_only_with_no_diagonal_corner_cutting",
        },
        "fully_claimable_scene_count": sum(
            scene["fully_claimable"] for scene in per_scene
        ),
        "spawn_clear_scene_count": sum(scene["spawn_clear"] for scene in per_scene),
        "claimable_beacon_count": sum(
            scene["claimable_beacons"] for scene in per_scene
        ),
        "reachable_center_cell_count": sum(
            scene["reachable_center_cells"] for scene in per_scene
        ),
        "topology_totals": topology_totals,
        "per_scene": per_scene,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    development_path = args.development_manifest.resolve()
    scene_root = args.scene_root.resolve()
    calibration_path = args.calibration.resolve()
    development = _load_json(development_path)
    calibration = _load_json(calibration_path)
    if development.get("schema") != "lewm_navigation_development_manifest_v0":
        raise ValueError("unsupported development manifest schema")
    if calibration.get("additional_genesis_rollout", {}).get("required") is not False:
        raise ValueError("swept-footprint calibration coverage gate has not passed")
    scene_records = _scene_paths(development, scene_root)
    manifests = tuple(
        parse_scene_manifest_dict(_load_json(path)) for _record, path in scene_records
    )
    expected_ids = [str(record["scene_id"]) for record, _path in scene_records]
    if [manifest.scene_id for manifest in manifests] != expected_ids:
        raise ValueError("materialized development scene order does not match manifest")

    recommendation = calibration["recommendation"]
    probe = recommendation["action_probe"]
    footprint = AsymmetricFootprint.with_half_width(
        forward_m=float(probe["forward_m"]),
        rear_m=float(probe["rear_m"]),
        half_width_m=float(probe["half_width_m"]),
    )
    audit_config = development["audit_config"]
    discs = [
        _summarize_disc(
            manifests,
            audit_config=audit_config,
            radius_m=radius,
        )
        for radius in args.disc_radius_m
    ]
    oriented = _summarize_oriented(
        manifests,
        footprint=footprint,
        audit_config=audit_config,
        cell_size_m=args.cell_size_m,
        yaw_bins=args.yaw_bins,
        rotation_subsamples=args.rotation_subsamples,
    )
    return {
        "schema": "lewm_go2_development_footprint_comparison_v0",
        "scope": {
            "split": "validation",
            "scene_count": len(manifests),
            "beacon_count": sum(len(manifest.landmarks) for manifest in manifests),
            "sealed_test_accessed": False,
        },
        "source_artifacts": {
            "development_manifest": {
                "path": str(development_path.relative_to(ROOT)),
                "sha256": _sha256(development_path),
            },
            "calibration_report": {
                "path": str(calibration_path.relative_to(ROOT)),
                "sha256": _sha256(calibration_path),
            },
            "audit_script": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__)),
            },
            "oriented_feasibility_helper": {
                "path": "lewm/planning/oriented_footprint.py",
                "sha256": _sha256(ROOT / "lewm/planning/oriented_footprint.py"),
            },
        },
        "methodology": {
            "claim_radius_m": float(audit_config["claim_radius_m"]),
            "line_of_sight_required": bool(audit_config["require_line_of_sight"]),
            "landmarks_are_obstacles": bool(
                audit_config["treat_landmarks_as_obstacles"]
            ),
            "distractors_are_obstacles": bool(
                audit_config["treat_distractors_as_obstacles"]
            ),
            "disc_audit": "canonical fixed-spawn dense occupancy audit",
            "oriented_audit": (
                "exact-SAT pose masks on an SE(2) lattice; rotations require all "
                "intermediate yaw masks; translations are fore/aft only"
            ),
        },
        "disc_models": discs,
        "oriented_model": oriented,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MANIFEST,
    )
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--disc-radius-m",
        type=float,
        action="append",
        default=None,
    )
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--yaw-bins", type=int, default=16)
    parser.add_argument("--rotation-subsamples", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.disc_radius_m = tuple(args.disc_radius_m or DEFAULT_DISC_RADII_M)
    report = build_report(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    for disc in report["disc_models"]:
        print(
            f"disc {disc['radius_m']:.2f}m: "
            f"scenes={disc['fully_claimable_scene_count']}/{report['scope']['scene_count']} "
            f"beacons={disc['claimable_beacon_count']}/{report['scope']['beacon_count']}"
        )
    oriented = report["oriented_model"]
    print(
        "oriented: "
        f"scenes={oriented['fully_claimable_scene_count']}/{report['scope']['scene_count']} "
        f"beacons={oriented['claimable_beacon_count']}/{report['scope']['beacon_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
