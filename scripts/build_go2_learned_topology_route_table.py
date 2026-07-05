#!/usr/bin/env python3
"""Build a same-scene learned topology route table from closed-loop teacher data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-spacing-m", type=float, default=0.22)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        schema = str(data["schema"][0]) if "schema" in data else ""
        if schema != "lewm_go2_closed_loop_learned_local_policy_dataset_v0":
            raise SystemExit(f"unsupported dataset schema: {schema}")
        meta = [json.loads(str(item)) for item in data["meta_json"].tolist()]
        source_result = json.loads(str(data["result_json"][0])) if "result_json" in data else {}

    routes: dict[str, dict[str, Any]] = {}
    for item in sorted(meta, key=lambda row: int(row.get("tick", 0))):
        if str(item.get("state", "")).upper() != "EXPLORE":
            continue
        pose_xy = item.get("pose_xy")
        if not isinstance(pose_xy, list) or len(pose_xy) < 2:
            continue
        target_color = str(item.get("target_color", ""))
        target_index = int(item.get("target_index", -1))
        if not target_color or target_index < 0:
            continue
        key = target_color
        route = routes.setdefault(
            key,
            {
                "target_color": target_color,
                "target_index": target_index,
                "waypoints": [],
            },
        )
        waypoint = [float(pose_xy[0]), float(pose_xy[1])]
        waypoints = route["waypoints"]
        if not waypoints or _dist(waypoints[-1], waypoint) >= float(args.min_spacing_m):
            waypoints.append(waypoint)
        else:
            waypoints[-1] = waypoint

    for route in routes.values():
        if len(route["waypoints"]) == 1:
            route["waypoints"].append(list(route["waypoints"][0]))

    output = {
        "schema": "lewm_go2_learned_topology_route_table_v1",
        "source_dataset": str(args.dataset),
        "source_scene": source_result.get("scene"),
        "source_success": source_result.get("success"),
        "min_spacing_m": float(args.min_spacing_m),
        "routes": routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(
        f"route_table: output={args.output} "
        f"routes={len(routes)} waypoints={sum(len(r['waypoints']) for r in routes.values())}",
        flush=True,
    )
    return 0


def _dist(a: list[float], b: list[float]) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


if __name__ == "__main__":
    raise SystemExit(main())
