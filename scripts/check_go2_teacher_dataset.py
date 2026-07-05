#!/usr/bin/env python3
"""Gate a privileged Go2 teacher rollout before using it for policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_COLORS = {"red", "yellow", "blue", "green"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-claims", type=int, default=4)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--require-all-beacons", action="store_true")
    parser.add_argument("--max-claim-distance-m", type=float, default=None)
    parser.add_argument("--max-contact-like-stalls", type=int, default=0)
    parser.add_argument("--max-hard-stalls", type=int, default=0)
    parser.add_argument("--max-body-violations", type=int, default=0)
    parser.add_argument("--min-examples", type=int, default=1)
    parser.add_argument("--forbid-pose-topology-features", action="store_true")
    args = parser.parse_args()

    payload = _load_json(args.result)
    result = payload.get("result", payload)
    wall = result.get("wall_metrics", {}) if isinstance(result, dict) else {}
    claimed = {str(item) for item in result.get("claimed_colors", [])}
    beacon_claims = result.get("beacon_claims", [])
    claim_distances = {
        str(item.get("target_color")): item.get("dist_to_target_m")
        for item in beacon_claims
        if isinstance(item, dict)
    }
    dataset_report = _dataset_report(args.dataset)
    feature_variant = str(wall.get("learned_local_policy_feature_variant", ""))

    gates = {
        "result_exists": args.result.is_file(),
        "dataset_exists": args.dataset is None or args.dataset.is_file(),
        "dataset_schema": args.dataset is None
        or dataset_report.get("schema") == "lewm_go2_closed_loop_learned_local_policy_dataset_v0",
        "dataset_min_examples": args.dataset is None
        or int(dataset_report.get("example_count") or 0) >= int(args.min_examples),
        "min_claims": len(claimed) >= int(args.min_claims),
        "contact_like_stalls": _metric_int(wall, "contact_like_stalls", "contact_like_stall_events")
        <= int(args.max_contact_like_stalls),
        "hard_stalls": _metric_int(wall, "hard_contact_like_stalls", "hard_stalls", "hard_stall_events")
        <= int(args.max_hard_stalls),
        "body_clearance_violations": _metric_int(wall, "body_clearance_violation_events")
        <= int(args.max_body_violations),
        "fall_tip_unstable": _metric_int(wall, "fall_events") == 0
        and _metric_int(wall, "tip_events") == 0
        and _metric_int(wall, "unstable_base_events") == 0,
    }
    if bool(args.require_success):
        gates["success"] = bool(result.get("success"))
        gates["all_beacons_claimed"] = REQUIRED_COLORS.issubset(claimed)
    if bool(args.require_all_beacons):
        gates["all_beacons_claimed"] = REQUIRED_COLORS.issubset(claimed)
    if args.max_claim_distance_m is not None:
        max_dist = float(args.max_claim_distance_m)
        required = REQUIRED_COLORS if bool(args.require_all_beacons) else claimed
        gates["claim_distances"] = all(
            claim_distances.get(color) is not None
            and float(claim_distances[color]) <= max_dist
            for color in required
        )
    if bool(args.forbid_pose_topology_features):
        gates["no_pose_topology_features"] = feature_variant != "pose_topology_v1"

    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "result": {
            "path": str(args.result),
            "scene": str(result.get("scene", "")),
            "success": bool(result.get("success")),
            "ticks_used": int(result.get("ticks_used") or 0),
            "claimed_colors": sorted(claimed),
            "claim_distances_m": claim_distances,
            "contact_like_stalls": _metric_int(wall, "contact_like_stalls", "contact_like_stall_events"),
            "hard_stalls": _metric_int(wall, "hard_contact_like_stalls", "hard_stalls", "hard_stall_events"),
            "body_clearance_violation_events": _metric_int(wall, "body_clearance_violation_events"),
            "feature_variant": feature_variant,
            "explore_goal_policy": wall.get("explore_goal_policy"),
            "explore_standoff_route": bool(wall.get("explore_standoff_route")),
        },
        "dataset": dataset_report,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_int(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _dataset_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        return {"path": str(path), "exists": False}
    try:
        with np.load(path, allow_pickle=False) as data:
            schema = str(data["schema"][0]) if "schema" in data else ""
            features = data["features"] if "features" in data else np.zeros((0, 0))
            labels = data["labels"] if "labels" in data else np.zeros((0,), dtype=np.int64)
            return {
                "path": str(path),
                "exists": True,
                "schema": schema,
                "example_count": int(features.shape[0]),
                "feature_dim": int(features.shape[1]) if features.ndim == 2 else 0,
                "label_count": int(labels.shape[0]),
            }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"path": str(path), "exists": True, "error": str(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
