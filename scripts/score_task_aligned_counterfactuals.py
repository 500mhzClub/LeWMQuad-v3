#!/usr/bin/env python3
"""Score all registered velocity primitives from mined rollout decision poses."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
try:
    import yaml as _yaml  # noqa: F401
except ModuleNotFoundError:
    _system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if _system_dist_packages.is_dir():
        sys.path.append(str(_system_dist_packages))
        import yaml as _yaml  # noqa: F401
        sys.path.remove(str(_system_dist_packages))

from lewm.actions import encode_active_block  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, expand_primitive_to_block  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402


@dataclass(frozen=True)
class SceneAssets:
    graph: SceneGraph
    grid: InflatedOccupancyGrid


def _wrap_angle_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _simulate_primitive(
    block: np.ndarray,
    *,
    start_xy: tuple[float, float],
    start_yaw: float,
    command_dt_s: float,
    grid: InflatedOccupancyGrid,
) -> tuple[tuple[float, float], float, bool]:
    x, y = start_xy
    yaw = float(start_yaw)
    collided = False
    for vx_body, vy_body, yaw_rate in block:
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        next_xy = (
            x + (float(vx_body) * cos_yaw - float(vy_body) * sin_yaw) * command_dt_s,
            y + (float(vx_body) * sin_yaw + float(vy_body) * cos_yaw) * command_dt_s,
        )
        if not grid.is_free(next_xy):
            collided = True
            break
        x, y = next_xy
        yaw = _wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
    return (x, y), yaw, collided


def _target_cell(row: dict) -> int | None:
    oracle_next = row.get("oracle_next_cell_id")
    if oracle_next is not None:
        return int(oracle_next)
    route_target = int(row.get("route_target_id", -1))
    return route_target if route_target >= 0 else None


def _score_row(
    row: dict,
    *,
    assets: SceneAssets,
    registry: PrimitiveRegistry,
    primitive_names: list[str],
    progress_weight: float,
    collision_penalty: float,
    clearance_target_m: float,
    clearance_penalty: float,
    heading_weight: float,
) -> dict:
    position = row["start_base_pose_world"]["position"]
    start_xy = (float(position["x"]), float(position["y"]))
    start_yaw = float(row["start_base_rpy_rad"]["yaw"])
    target_cell = _target_cell(row)
    target_xy = assets.graph.cell_center(target_cell) if target_cell is not None else None
    initial_distance = (
        math.dist(start_xy, target_xy) if target_xy is not None else None
    )

    candidates = []
    for primitive_name in primitive_names:
        block = expand_primitive_to_block(registry, primitive_name)
        endpoint, end_yaw, collided = _simulate_primitive(
            block,
            start_xy=start_xy,
            start_yaw=start_yaw,
            command_dt_s=registry.command_dt_s,
            grid=assets.grid,
        )
        clearance = assets.graph.clearance_to_walls(
            endpoint,
            include_landmarks=True,
        )
        distance = math.dist(endpoint, target_xy) if target_xy is not None else None
        progress = initial_distance - distance if distance is not None else None
        if target_xy is None:
            heading_error = None
            task_cost = 0.0
        else:
            bearing = math.atan2(target_xy[1] - endpoint[1], target_xy[0] - endpoint[0])
            heading_error = abs(_wrap_angle_pi(bearing - end_yaw))
            # Progress-dominant task term: reward distance reduction toward the
            # goal directly (meters), so a safe forward action outranks a
            # turn-in-place. Absolute distance was dwarfed by the collision and
            # clearance penalties, making the oracle a low-motion safe turn.
            task_cost = -progress_weight * progress + heading_weight * heading_error
        clearance_shortfall = max(0.0, clearance_target_m - clearance)
        cost = (
            task_cost
            + collision_penalty * float(collided)
            + clearance_penalty * clearance_shortfall
        )
        candidates.append(
            {
                "primitive_name": primitive_name,
                "active_block": encode_active_block(
                    block[:, 0],
                    block[:, 1],
                    block[:, 2],
                ).tolist(),
                "endpoint_xy": [float(endpoint[0]), float(endpoint[1])],
                "end_yaw_rad": float(end_yaw),
                "collided": collided,
                "clearance_m": float(clearance),
                "target_distance_m": float(distance) if distance is not None else None,
                "target_progress_m": float(progress) if progress is not None else None,
                "heading_error_rad": (
                    float(heading_error) if heading_error is not None else None
                ),
                "cost": float(cost),
            }
        )
    best = min(candidates, key=lambda item: (item["cost"], item["primitive_name"]))
    logged = next(
        (item for item in candidates if item["primitive_name"] == row["primitive_name"]),
        None,
    )
    return {
        **row,
        "counterfactual_schema": "task_aligned_counterfactual_v1",
        "counterfactual_target_cell_id": target_cell,
        "counterfactual_initial_target_distance_m": initial_distance,
        "counterfactual_best_primitive": best["primitive_name"],
        "counterfactual_best_cost": best["cost"],
        "counterfactual_logged_cost": logged["cost"] if logged is not None else None,
        "counterfactual_logged_regret": (
            logged["cost"] - best["cost"] if logged is not None else None
        ),
        "counterfactual_candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--primitive-registry",
        type=Path,
        default=REPO_ROOT / "config/go2_primitive_registry.yaml",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-weight", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=3.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.35)
    parser.add_argument("--clearance-penalty", type=float, default=0.5)
    parser.add_argument("--heading-weight", type=float, default=0.1)
    parser.add_argument(
        "--drop-no-safe-action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop decisions where every candidate primitive collides (no "
        "navigable action exists); removes the irreducible collision floor.",
    )
    args = parser.parse_args()

    registry = PrimitiveRegistry.from_yaml(args.primitive_registry)
    primitive_names = registry.trainable_velocity_names()
    assets_by_manifest: dict[str, SceneAssets] = {}
    best_counts: Counter[str] = Counter()
    logged_counts: Counter[str] = Counter()
    rows = 0
    rows_with_target = 0
    logged_optimal = 0
    logged_regret_sum = 0.0
    candidate_collisions = 0
    dropped_no_safe_action = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as source, args.output.open("w") as destination:
        for line in source:
            if args.max_rows > 0 and rows >= args.max_rows:
                break
            row = json.loads(line)
            manifest_path = str(row["scene_manifest"])
            assets = assets_by_manifest.get(manifest_path)
            if assets is None:
                manifest = parse_scene_manifest_dict(
                    json.loads(Path(manifest_path).read_text())
                )
                assets = SceneAssets(
                    graph=SceneGraph(manifest),
                    grid=InflatedOccupancyGrid(
                        manifest,
                        cell_size_m=0.05,
                        inflation_m=0.20,
                    ),
                )
                assets_by_manifest[manifest_path] = assets
            scored = _score_row(
                row,
                assets=assets,
                registry=registry,
                primitive_names=primitive_names,
                progress_weight=args.progress_weight,
                collision_penalty=args.collision_penalty,
                clearance_target_m=args.clearance_target_m,
                clearance_penalty=args.clearance_penalty,
                heading_weight=args.heading_weight,
            )
            if args.drop_no_safe_action and all(
                candidate["collided"]
                for candidate in scored["counterfactual_candidates"]
            ):
                dropped_no_safe_action += 1
                continue
            destination.write(json.dumps(scored, sort_keys=True) + "\n")
            rows += 1
            rows_with_target += scored["counterfactual_target_cell_id"] is not None
            best_counts[scored["counterfactual_best_primitive"]] += 1
            logged_counts[scored["primitive_name"]] += 1
            regret = scored["counterfactual_logged_regret"]
            if regret is not None:
                logged_regret_sum += float(regret)
                logged_optimal += float(regret) <= 1e-8
            candidate_collisions += sum(
                candidate["collided"]
                for candidate in scored["counterfactual_candidates"]
            )

    if rows == 0:
        raise SystemExit("no rows scored")
    summary = {
        "schema": "task_aligned_counterfactual_summary_v1",
        "input": str(args.input),
        "output": str(args.output),
        "row_count": rows,
        "dropped_no_safe_action": dropped_no_safe_action,
        "drop_no_safe_action": bool(args.drop_no_safe_action),
        "scene_count": len(assets_by_manifest),
        "primitive_names": primitive_names,
        "rows_with_target": rows_with_target,
        "best_primitive_counts": dict(best_counts),
        "logged_primitive_counts": dict(logged_counts),
        "logged_optimal_rate": logged_optimal / rows,
        "mean_logged_regret": logged_regret_sum / rows,
        "candidate_collision_rate": candidate_collisions / (rows * len(primitive_names)),
        "weights": {
            "progress_weight": args.progress_weight,
            "collision_penalty": args.collision_penalty,
            "clearance_target_m": args.clearance_target_m,
            "clearance_penalty": args.clearance_penalty,
            "heading_weight": args.heading_weight,
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
