#!/usr/bin/env python3
"""Gate a Go2 all-beacon result against the fully learned runtime contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_COLORS = {"red", "yellow", "blue", "green"}
FORBIDDEN_FLAGS = {
    "--explore-standoff-route",
    "--learned-local-oracle-standoff-labels",
    "--learned-local-dataset-output",
    "--face-target",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--quality", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=400)
    parser.add_argument("--max-contact-like-stalls", type=int, default=0)
    parser.add_argument("--max-hard-stalls", type=int, default=0)
    parser.add_argument("--max-body-violations", type=int, default=0)
    parser.add_argument("--require-generalized-runtime-contract", action="store_true")
    parser.add_argument("--forbid-route-memory", action="store_true")
    parser.add_argument("--forbid-pose-topology-features", action="store_true")
    parser.add_argument(
        "--train-scenes",
        default="",
        help="Comma-separated scene ids that must be disjoint from this result.",
    )
    args = parser.parse_args()

    payload = _load_json(args.result)
    result = payload.get("result", payload)
    wall = result.get("wall_metrics", {}) if isinstance(result, dict) else {}
    argv = [str(item) for item in payload.get("provenance", {}).get("argv", [])]
    quality = _load_json(args.quality) if args.quality is not None else {}

    claimed = {str(item) for item in result.get("claimed_colors", [])}
    contract = wall.get("fully_learned_runtime_contract_report", {})
    forbidden_argv = _forbidden_argv(argv)
    contact_like = _metric_int(wall, "contact_like_stalls", "contact_like_stall_events")
    hard_stalls = _metric_int(wall, "hard_contact_like_stalls", "hard_stalls", "hard_stall_events")
    body_violations = _metric_int(wall, "body_clearance_violation_events")
    learned_policy_runtime = bool(
        str(wall.get("explore_goal_policy", "")).lower() == "learned_policy"
        and wall.get("learned_local_policy_checkpoint")
        and int(wall.get("learned_local_policy_ticks") or 0) > 0
    )
    online_frontier_runtime = bool(
        str(wall.get("explore_goal_policy", "")).lower() == "online_frontier"
    )
    learned_policy_disabled_ticks = int(wall.get("learned_local_policy_disabled_ticks") or 0)
    learned_policy_feature_mismatch_ticks = int(
        wall.get("learned_local_policy_feature_mismatch_ticks") or 0
    )
    if (
        "learned_local_policy_disabled_ticks" not in wall
        or "learned_local_policy_feature_mismatch_ticks" not in wall
    ):
        log_rows = payload.get("log", [])
        if isinstance(log_rows, list):
            disabled = 0
            feature_mismatch = 0
            for row in log_rows:
                if not isinstance(row, dict):
                    continue
                learned_policy_log = row.get("learned_local_policy")
                if not isinstance(learned_policy_log, dict):
                    continue
                if bool(learned_policy_log.get("enabled", False)):
                    continue
                disabled += 1
                if str(learned_policy_log.get("reason", "")) == "feature_dim_mismatch":
                    feature_mismatch += 1
            learned_policy_disabled_ticks = disabled
            learned_policy_feature_mismatch_ticks = feature_mismatch
    route_ticks = int(wall.get("learned_topology_route_ticks") or 0)
    route_skipped_ticks = int(
        wall.get("learned_topology_route_privileged_explorer_skipped_ticks") or 0
    )
    route_contract = contract.get("learned_topology_route_table")
    learned_route_runtime = bool(
        wall.get("learned_topology_route_table")
        and route_ticks > 0
        and route_skipped_ticks == route_ticks
        and isinstance(route_contract, dict)
        and route_contract.get("schema") == "lewm_go2_learned_topology_route_table_v1"
        and route_contract.get("source_success") is True
        and int(route_contract.get("waypoint_count") or 0) > 0
    )
    policy_contract = contract.get("learned_local_policy_checkpoint")
    policy_feature_variant = (
        str(policy_contract.get("feature_variant", ""))
        if isinstance(policy_contract, dict)
        else str(wall.get("learned_local_policy_feature_variant", ""))
    )
    post_claim_policy_contract = contract.get("learned_local_post_claim_policy_checkpoint")
    post_claim_policy_feature_variant = (
        str(post_claim_policy_contract.get("feature_variant", ""))
        if isinstance(post_claim_policy_contract, dict)
        else str(wall.get("learned_local_post_claim_policy_feature_variant", ""))
    )
    train_scenes = {item.strip() for item in str(args.train_scenes).split(",") if item.strip()}
    scene_id = str(result.get("scene", ""))

    gates = {
        "runtime_contract_flag": bool(wall.get("fully_learned_runtime_contract")),
        "runtime_contract_passed": bool(contract.get("passed")),
        "no_forbidden_argv": not forbidden_argv,
        "learned_runtime_action_source": (
            learned_policy_runtime or learned_route_runtime or online_frontier_runtime
        ),
        "learned_local_policy_enabled": (
            not learned_policy_runtime or learned_policy_disabled_ticks == 0
        ),
        "learned_local_policy_feature_match": (
            not learned_policy_runtime or learned_policy_feature_mismatch_ticks == 0
        ),
        "learned_route_memory_or_policy": (
            str(contract.get("runtime_path", ""))
            in {"learned_local_policy", "learned_topology_route_memory", "online_frontier"}
            or online_frontier_runtime
        ),
        "no_standoff_route": not bool(wall.get("explore_standoff_route"))
        and int(wall.get("explore_standoff_replans") or 0) == 0
        and int(wall.get("explore_standoff_final_path_len") or 0) == 0,
        "no_route_waypoints": not bool(wall.get("explore_route_waypoints")),
        "no_privileged_explorer_on_route_ticks": (
            not wall.get("learned_topology_route_table")
            or (route_ticks > 0 and route_skipped_ticks == route_ticks)
        ),
        "no_oracle_labels": not bool(wall.get("learned_local_oracle_standoff_labels"))
        and int(wall.get("learned_local_oracle_standoff_label_ticks") or 0) == 0,
        "learned_wall_source": str(wall.get("source")) == "learned_action_outcome",
        "success": bool(result.get("success")),
        "all_beacons_claimed": REQUIRED_COLORS.issubset(claimed),
        "ticks": int(result.get("ticks_used") or 0) <= int(args.max_ticks),
        "contact_like_stalls": contact_like <= int(args.max_contact_like_stalls),
        "hard_stalls": hard_stalls <= int(args.max_hard_stalls),
        "body_clearance_violations": body_violations <= int(args.max_body_violations),
        "fall_tip_unstable": _metric_int(wall, "fall_events") == 0
        and _metric_int(wall, "tip_events") == 0
        and _metric_int(wall, "unstable_base_events") == 0,
    }
    if quality:
        gates["quality_passed"] = bool(quality.get("passed", True))
    if bool(args.require_generalized_runtime_contract):
        policy_explore_ticks = int(
            wall.get("learned_local_policy_explore_state_ticks")
            if wall.get("learned_local_policy_explore_state_ticks") is not None
            else wall.get("learned_local_policy_ticks") or 0
        )
        gates["generalized_runtime_contract_flag"] = bool(wall.get("generalized_runtime_contract"))
        gates["generalized_contract_report"] = bool(contract.get("generalized"))
        gates["generalized_runtime_path"] = str(contract.get("runtime_path", "")) == "learned_local_policy"
        gates["generalized_uses_policy_ticks"] = learned_policy_runtime
        gates["no_privileged_explorer_for_policy_ticks"] = int(
            wall.get("learned_local_policy_privileged_explorer_skipped_ticks") or 0
        ) >= policy_explore_ticks
    if bool(args.forbid_route_memory):
        gates["no_learned_topology_route_memory"] = (
            not bool(wall.get("learned_topology_route_table"))
            and route_ticks == 0
        )
    if bool(args.forbid_pose_topology_features):
        gates["no_pose_topology_features"] = (
            "pose_topology" not in policy_feature_variant
            and "pose_topology" not in post_claim_policy_feature_variant
        )
    if train_scenes:
        gates["scene_disjoint_from_train"] = scene_id not in train_scenes

    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "result": {
            "path": str(args.result),
            "scene": scene_id,
            "success": bool(result.get("success")),
            "ticks_used": int(result.get("ticks_used") or 0),
            "claimed_colors": sorted(claimed),
            "contact_like_stalls": contact_like,
            "hard_stalls": hard_stalls,
            "body_clearance_violation_events": body_violations,
            "wall_source": wall.get("source"),
            "learned_local_policy_checkpoint": wall.get("learned_local_policy_checkpoint"),
            "learned_local_post_claim_policy_checkpoint": wall.get(
                "learned_local_post_claim_policy_checkpoint"
            ),
            "learned_local_policy_feature_variant": policy_feature_variant,
            "learned_local_post_claim_policy_feature_variant": post_claim_policy_feature_variant,
            "learned_local_policy_ticks": int(wall.get("learned_local_policy_ticks") or 0),
            "learned_local_post_claim_policy_ticks": int(
                wall.get("learned_local_post_claim_policy_ticks") or 0
            ),
            "learned_local_policy_disabled_ticks": learned_policy_disabled_ticks,
            "learned_local_policy_feature_mismatch_ticks": learned_policy_feature_mismatch_ticks,
            "learned_local_policy_privileged_explorer_skipped_ticks": int(
                wall.get("learned_local_policy_privileged_explorer_skipped_ticks") or 0
            ),
            "learned_topology_route_table": wall.get("learned_topology_route_table"),
            "learned_topology_route_ticks": route_ticks,
            "learned_topology_route_privileged_explorer_skipped_ticks": route_skipped_ticks,
        },
        "contract_failures": contract.get("failures", []),
        "forbidden_argv": forbidden_argv,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


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


def _forbidden_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    for idx, item in enumerate(argv):
        if item in FORBIDDEN_FLAGS:
            out.append(item)
        if item == "--wall-decision-source" and idx + 1 < len(argv) and argv[idx + 1] == "privileged_grid":
            out.append("--wall-decision-source privileged_grid")
        if item == "--explore-route-waypoints" and idx + 1 < len(argv) and argv[idx + 1].strip():
            out.append("--explore-route-waypoints")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
