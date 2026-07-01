#!/usr/bin/env python3
"""Gate a scene-disjoint generalized Go2 learned-local evaluation suite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_COLORS = {"red", "yellow", "blue", "green"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, default=None)
    parser.add_argument("--train-scenes", default="")
    parser.add_argument("--heldout-scenes", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-ticks", type=int, default=560)
    parser.add_argument("--max-contact-like-stalls", type=int, default=0)
    parser.add_argument("--max-hard-stalls", type=int, default=0)
    parser.add_argument("--max-body-violations", type=int, default=0)
    args = parser.parse_args()

    train_scenes = _csv_set(args.train_scenes)
    heldout_scenes = _csv_set(args.heldout_scenes)
    rows = [_result_row(path) for path in args.results]
    result_scenes = {str(row["scene"]) for row in rows if row.get("scene")}
    policy_report = _load_json(args.policy_report) if args.policy_report else {}
    report_train_scenes = _policy_report_scenes(policy_report)
    all_train_scenes = set(train_scenes) | set(report_train_scenes)

    success_count = sum(1 for row in rows if bool(row["success"]))
    all_claimed_count = sum(1 for row in rows if bool(row["all_beacons_claimed"]))
    required_successes = int(math.ceil(max(0.0, min(1.0, float(args.min_success_rate))) * len(rows)))
    if rows and required_successes == 0 and float(args.min_success_rate) > 0.0:
        required_successes = 1

    gates = {
        "has_results": bool(rows),
        "heldout_scene_count": len(result_scenes) == len(rows),
        "scene_disjoint_from_train": not (result_scenes & all_train_scenes),
        "matches_requested_heldout_scenes": (
            not heldout_scenes or result_scenes == heldout_scenes
        ),
        "generalized_contract": all(bool(row["generalized_contract"]) for row in rows),
        "learned_local_policy_runtime": all(bool(row["learned_local_policy_runtime"]) for row in rows),
        "no_privileged_explorer_for_policy_ticks": all(
            int(row["learned_local_policy_privileged_explorer_skipped_ticks"])
            >= int(row["learned_local_policy_explore_state_ticks"])
            for row in rows
        ),
        "no_route_memory": all(not bool(row["route_memory"]) for row in rows),
        "no_pose_topology_features": all(
            "pose_topology" not in str(row["policy_feature_variant"]) for row in rows
        ),
        "success_rate": success_count >= required_successes,
        "all_beacons_claimed": all_claimed_count >= required_successes,
        "ticks": all(int(row["ticks_used"]) <= int(args.max_ticks) for row in rows),
        "contact_like_stalls": all(
            int(row["contact_like_stalls"]) <= int(args.max_contact_like_stalls) for row in rows
        ),
        "hard_stalls": all(int(row["hard_stalls"]) <= int(args.max_hard_stalls) for row in rows),
        "body_clearance_violations": all(
            int(row["body_clearance_violation_events"]) <= int(args.max_body_violations)
            for row in rows
        ),
        "fall_tip_unstable": all(bool(row["fall_tip_unstable"]) for row in rows),
    }
    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "summary": {
            "result_count": len(rows),
            "train_scenes": sorted(all_train_scenes),
            "heldout_scenes": sorted(result_scenes),
            "success_count": success_count,
            "all_beacons_claimed_count": all_claimed_count,
            "required_successes": required_successes,
            "success_rate": (success_count / len(rows)) if rows else 0.0,
        },
        "results": rows,
        "policy_report": str(args.policy_report) if args.policy_report else None,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def _csv_set(raw: str) -> set[str]:
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_int(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _result_row(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    result = payload.get("result", payload)
    wall = result.get("wall_metrics", {}) if isinstance(result, dict) else {}
    contract = wall.get("fully_learned_runtime_contract_report", {})
    policy_contract = contract.get("learned_local_policy_checkpoint")
    feature_variant = (
        str(policy_contract.get("feature_variant", ""))
        if isinstance(policy_contract, dict)
        else str(wall.get("learned_local_policy_feature_variant", ""))
    )
    claimed = {str(item) for item in result.get("claimed_colors", [])}
    fall_tip_unstable = (
        _metric_int(wall, "fall_events") == 0
        and _metric_int(wall, "tip_events") == 0
        and _metric_int(wall, "unstable_base_events") == 0
    )
    return {
        "path": str(path),
        "scene": str(result.get("scene", "")),
        "execution_mode": str(result.get("execution_mode", "")),
        "success": bool(result.get("success")),
        "all_beacons_claimed": REQUIRED_COLORS.issubset(claimed),
        "claimed_colors": sorted(claimed),
        "ticks_used": int(result.get("ticks_used") or 0),
        "contact_like_stalls": _metric_int(wall, "contact_like_stalls", "contact_like_stall_events"),
        "hard_stalls": _metric_int(wall, "hard_contact_like_stalls", "hard_stalls", "hard_stall_events"),
        "body_clearance_violation_events": _metric_int(wall, "body_clearance_violation_events"),
        "fall_tip_unstable": fall_tip_unstable,
        "generalized_contract": bool(wall.get("generalized_runtime_contract"))
        and bool(contract.get("generalized"))
        and bool(contract.get("passed"))
        and str(contract.get("runtime_path", "")) == "learned_local_policy",
        "learned_local_policy_runtime": bool(
            str(wall.get("explore_goal_policy", "")).lower() == "learned_policy"
            and wall.get("learned_local_policy_checkpoint")
            and int(wall.get("learned_local_policy_ticks") or 0) > 0
        ),
        "learned_local_policy_ticks": int(wall.get("learned_local_policy_ticks") or 0),
        "learned_local_policy_explore_state_ticks": int(
            wall.get("learned_local_policy_explore_state_ticks")
            if wall.get("learned_local_policy_explore_state_ticks") is not None
            else wall.get("learned_local_policy_ticks") or 0
        ),
        "learned_local_policy_privileged_explorer_skipped_ticks": int(
            wall.get("learned_local_policy_privileged_explorer_skipped_ticks") or 0
        ),
        "policy_feature_variant": feature_variant,
        "route_memory": bool(wall.get("learned_topology_route_table"))
        or int(wall.get("learned_topology_route_ticks") or 0) > 0,
        "wall_source": wall.get("source"),
    }


def _policy_report_scenes(report: dict[str, Any]) -> set[str]:
    scenes: set[str] = set()
    for key in ("dataset_reports", "validation_dataset_reports"):
        values = report.get(key, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and item.get("scene"):
                scenes.add(str(item["scene"]))
    return scenes


if __name__ == "__main__":
    raise SystemExit(main())
