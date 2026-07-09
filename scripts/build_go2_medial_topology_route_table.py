#!/usr/bin/env python3
"""Build a clearance-weighted route-memory table from an existing route table."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("route_table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path,
                        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--split", default="train")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--platform-manifest", type=Path,
                        default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--clearance-weight", type=float, default=4.0)
    parser.add_argument("--clearance-target-m", type=float, default=0.18)
    parser.add_argument("--min-spacing-m", type=float, default=0.12)
    parser.add_argument("--cardinal-only", action="store_true",
                        help="Disable diagonal grid moves when building medial routes.")
    args = parser.parse_args()

    table = json.loads(args.route_table.read_text())
    if table.get("schema") != "lewm_go2_learned_topology_route_table_v1":
        raise SystemExit(f"unsupported route table schema: {table.get('schema')}")

    scene_id = args.scene_id or table.get("source_scene")
    if not scene_id:
        raise SystemExit("--scene-id is required when route table has no source_scene")
    scene_dirs = find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family)
    matches = [path for path in scene_dirs if path.name == str(scene_id)]
    if not matches:
        raise SystemExit(f"scene-id not found: {scene_id}")
    platform = load_platform_manifest(args.platform_manifest.resolve())
    pack = load_scene_pack(matches[0], platform_manifest=platform, workspace_root=REPO_ROOT)
    grid = InflatedOccupancyGrid(
        pack.scene_graph.manifest,
        cell_size_m=float(args.cell_size_m),
        inflation_m=float(args.inflation_m),
    )

    routes: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for key, route in sorted(table.get("routes", {}).items(), key=lambda item: str(item[0])):
        waypoints = _normalise_waypoints(route.get("waypoints", []))
        target_color = str(route.get("target_color", key))
        target_index = int(route.get("target_index", key if str(key).isdigit() else 0))
        if len(waypoints) < 2:
            medial_waypoints = waypoints
            status = "fallback_short_route"
            cost_cells = None
        else:
            start = waypoints[0]
            goal = waypoints[-1]
            path = grid.astar(
                start,
                goal,
                clearance_weight=float(args.clearance_weight),
                clearance_target_m=float(args.clearance_target_m),
                allow_diagonal=not bool(args.cardinal_only),
            )
            if path is None:
                medial_waypoints = waypoints
                status = "fallback_astar_failed"
                cost_cells = None
            else:
                medial_waypoints = [start, *[(float(x), float(y)) for x, y in path.waypoints_xy]]
                medial_waypoints = _downsample_waypoints(
                    medial_waypoints,
                    min_spacing_m=max(0.0, float(args.min_spacing_m)),
                )
                status = "astar_medial"
                cost_cells = float(path.cost_cells)
        routes[str(key)] = {
            "target_color": target_color,
            "target_index": target_index,
            "waypoints": [[float(x), float(y)] for x, y in medial_waypoints],
            "source_waypoint_count": int(len(waypoints)),
            "builder": "clearance_weighted_astar_medial",
        }
        reports[str(key)] = {
            "target_color": target_color,
            "target_index": target_index,
            "status": status,
            "astar_cost_cells": cost_cells,
            "source_waypoint_count": int(len(waypoints)),
            "output_waypoint_count": int(len(medial_waypoints)),
            "source_clearance": _clearance_stats(grid, waypoints),
            "output_clearance": _clearance_stats(grid, medial_waypoints),
        }

    output = {
        "schema": "lewm_go2_learned_topology_route_table_v1",
        "source_dataset": table.get("source_dataset"),
        "source_scene": scene_id,
        "source_success": table.get("source_success"),
        "source_route_table": str(args.route_table),
        "postprocess": {
            "builder": "clearance_weighted_astar_medial",
            "scene_corpus": str(args.scene_corpus),
            "cell_size_m": float(args.cell_size_m),
            "inflation_m": float(args.inflation_m),
            "clearance_weight": float(args.clearance_weight),
            "clearance_target_m": float(args.clearance_target_m),
            "min_spacing_m": float(args.min_spacing_m),
            "allow_diagonal": not bool(args.cardinal_only),
        },
        "routes": routes,
        "route_reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(
        f"medial_route_table: output={args.output} routes={len(routes)} "
        f"waypoints={sum(len(route['waypoints']) for route in routes.values())}",
        flush=True,
    )
    for key, report in reports.items():
        print(
            f"  {key}: {report['status']} "
            f"{report['source_waypoint_count']}->{report['output_waypoint_count']} "
            f"min_cfg={report['source_clearance']['min_configuration_m']:.3f}"
            f"->{report['output_clearance']['min_configuration_m']:.3f}",
            flush=True,
        )
    return 0


def _normalise_waypoints(raw: Any) -> list[tuple[float, float]]:
    waypoints: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return waypoints
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            waypoints.append((float(item[0]), float(item[1])))
    return waypoints


def _downsample_waypoints(
    waypoints: list[tuple[float, float]],
    *,
    min_spacing_m: float,
) -> list[tuple[float, float]]:
    if len(waypoints) <= 2 or min_spacing_m <= 0.0:
        return list(waypoints)
    out = [waypoints[0]]
    for waypoint in waypoints[1:-1]:
        if _dist(out[-1], waypoint) >= min_spacing_m:
            out.append(waypoint)
    if _dist(out[-1], waypoints[-1]) > 1e-6:
        out.append(waypoints[-1])
    return out


def _clearance_stats(
    grid: InflatedOccupancyGrid,
    waypoints: list[tuple[float, float]],
) -> dict[str, float]:
    if not waypoints:
        return {
            "min_obstacle_m": math.nan,
            "mean_obstacle_m": math.nan,
            "min_configuration_m": math.nan,
            "mean_configuration_m": math.nan,
        }
    obstacle = np.asarray(
        [grid.obstacle_clearance_m((float(x), float(y))) for x, y in waypoints],
        dtype=np.float64,
    )
    configuration = obstacle - float(grid.inflation_m)
    return {
        "min_obstacle_m": float(np.min(obstacle)),
        "mean_obstacle_m": float(np.mean(obstacle)),
        "min_configuration_m": float(np.min(configuration)),
        "mean_configuration_m": float(np.mean(configuration)),
    }


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


if __name__ == "__main__":
    raise SystemExit(main())
