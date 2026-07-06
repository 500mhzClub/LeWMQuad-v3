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
    parser.add_argument("--max-blocked-forward-executions", type=int, default=None)
    parser.add_argument("--max-blocked-forward-requests", type=int, default=None)
    parser.add_argument("--min-body-clearance-m", type=float, default=None)
    parser.add_argument("--max-learned-local-max-turn-run", type=int, default=None)
    parser.add_argument("--max-post-claim-ticks", type=int, default=None)
    parser.add_argument("--max-post-claim-turn-count", type=int, default=None)
    parser.add_argument("--max-post-claim-max-turn-run", type=int, default=None)
    parser.add_argument("--require-active-jepa-veto", action="store_true")
    parser.add_argument("--require-online-map-memory-active", action="store_true")
    parser.add_argument("--require-generalized-runtime-contract", action="store_true")
    parser.add_argument("--require-learned-claim-success-model", action="store_true")
    parser.add_argument("--require-learned-target-scheduler", action="store_true")
    parser.add_argument("--forbid-route-memory", action="store_true")
    parser.add_argument("--forbid-pose-topology-features", action="store_true")
    parser.add_argument("--forbid-rule-based-target-switching", action="store_true")
    parser.add_argument("--forbid-frontier-pressure", action="store_true")
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
    blocked_forward_executions = _metric_int(wall, "blocked_forward_executions")
    blocked_forward_requests = _metric_int(wall, "blocked_forward_requests")
    body_clearance_min = _metric_float(wall, "body_clearance_min_m")
    learned_local_max_turn_run = _metric_int(wall, "learned_local_policy_max_turn_run")
    post_claim_diag = wall.get("post_claim_acquisition_diagnostics", {})
    if not isinstance(post_claim_diag, dict):
        post_claim_diag = {}
    post_claim_ticks = _metric_int(post_claim_diag, "ticks")
    post_claim_primitives = post_claim_diag.get("primitive_counts", {})
    if not isinstance(post_claim_primitives, dict):
        post_claim_primitives = {}
    post_claim_turn_count = int(post_claim_primitives.get("yaw_left") or 0) + int(
        post_claim_primitives.get("yaw_right") or 0
    )
    log_rows = payload.get("log", [])
    if not isinstance(log_rows, list):
        log_rows = []
    post_claim_start_tick = post_claim_diag.get("start_tick")
    post_claim_max_turn_run = _max_primitive_run(
        log_rows,
        {"yaw_left", "yaw_right"},
        min_tick=int(post_claim_start_tick) if isinstance(post_claim_start_tick, (int, float)) else None,
    )
    active_jepa_veto_count = (
        _metric_int(wall, "body_clearance_learned_vetoes")
        + _metric_int(wall, "body_clearance_hard_vetoes")
        + _metric_int(wall, "body_clearance_saturated_vetoes")
        + _metric_int(wall, "body_clearance_yaw_direction_vetoes")
        + _metric_int(wall, "blocked_hard_vetoes")
        + _metric_int(wall, "low_progress_hard_vetoes")
        + _metric_int(wall, "learned_local_policy_outcome_rerank_overrides")
    )
    online_map_memory_ticks = (
        _metric_int(wall, "learned_local_policy_online_map_novelty_ticks")
        + _metric_int(wall, "learned_local_policy_online_map_novelty_overrides")
        + _metric_int(wall, "learned_local_policy_frontier_pressure_ticks")
        + _metric_int(wall, "learned_local_policy_frontier_pressure_overrides")
        + _metric_int(wall, "learned_local_policy_translation_pressure_ticks")
        + _metric_int(wall, "learned_local_policy_translation_pressure_overrides")
        + _metric_int(wall, "learned_local_online_map_low_progress_block_ticks")
        + _metric_int(wall, "learned_local_online_map_route_replay_guard_override_ticks")
    )
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
    target_switches = int(wall.get("target_switches") or 0)
    learned_target_scheduler_switches = int(
        wall.get("learned_target_scheduler_switches") or 0
    )
    rule_based_target_switches = max(0, target_switches - learned_target_scheduler_switches)
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
    if args.max_blocked_forward_executions is not None:
        gates["blocked_forward_executions"] = blocked_forward_executions <= int(
            args.max_blocked_forward_executions
        )
    if args.max_blocked_forward_requests is not None:
        gates["blocked_forward_requests"] = blocked_forward_requests <= int(
            args.max_blocked_forward_requests
        )
    if args.min_body_clearance_m is not None:
        gates["body_clearance_min"] = (
            body_clearance_min is not None
            and body_clearance_min >= float(args.min_body_clearance_m)
        )
    if args.max_learned_local_max_turn_run is not None:
        gates["learned_local_max_turn_run"] = learned_local_max_turn_run <= int(
            args.max_learned_local_max_turn_run
        )
    if args.max_post_claim_ticks is not None:
        gates["post_claim_ticks"] = post_claim_ticks <= int(args.max_post_claim_ticks)
    if args.max_post_claim_turn_count is not None:
        gates["post_claim_turn_count"] = post_claim_turn_count <= int(
            args.max_post_claim_turn_count
        )
    if args.max_post_claim_max_turn_run is not None:
        gates["post_claim_max_turn_run"] = post_claim_max_turn_run <= int(
            args.max_post_claim_max_turn_run
        )
    if bool(args.require_active_jepa_veto):
        gates["active_jepa_veto"] = (
            bool(wall.get("primitive_outcome_checkpoint"))
            and bool(wall.get("primitive_clearance_checkpoint"))
            and active_jepa_veto_count > 0
        )
    if bool(args.require_online_map_memory_active):
        gates["online_map_memory_active"] = (
            bool(wall.get("learned_local_online_map_features"))
            and online_map_memory_ticks > 0
        )
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
        gates["generalized_runtime_path"] = (
            str(contract.get("runtime_path", ""))
            in {"learned_local_policy", "online_frontier"}
            or online_frontier_runtime
        )
        gates["generalized_uses_policy_ticks"] = learned_policy_runtime or online_frontier_runtime
        gates["no_privileged_explorer_for_policy_ticks"] = int(
            wall.get("learned_local_policy_privileged_explorer_skipped_ticks") or 0
        ) >= policy_explore_ticks
    if bool(args.require_learned_claim_success_model):
        gates["learned_claim_success_model"] = bool(
            wall.get("claim_success_model_checkpoint")
            and wall.get("claim_success_model_threshold") is not None
            and int(wall.get("claim_success_model_evaluations") or 0) > 0
        )
    if bool(args.require_learned_target_scheduler):
        gates["learned_target_scheduler"] = bool(
            wall.get("learned_target_scheduler_checkpoint")
            and int(wall.get("learned_target_scheduler_ticks") or 0) > 0
        )
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
    if bool(args.forbid_rule_based_target_switching):
        gates["no_rule_based_target_switching"] = rule_based_target_switches == 0
    if bool(args.forbid_frontier_pressure):
        gates["no_frontier_pressure"] = (
            int(wall.get("learned_local_policy_frontier_pressure_ticks") or 0) == 0
            and int(wall.get("learned_local_policy_frontier_pressure_overrides") or 0) == 0
            and int(wall.get("learned_local_policy_translation_pressure_ticks") or 0) == 0
            and int(wall.get("learned_local_policy_translation_pressure_overrides") or 0) == 0
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
            "blocked_forward_executions": blocked_forward_executions,
            "blocked_forward_requests": blocked_forward_requests,
            "body_clearance_min_m": body_clearance_min,
            "learned_local_policy_max_turn_run": learned_local_max_turn_run,
            "post_claim_ticks": post_claim_ticks,
            "post_claim_turn_count": post_claim_turn_count,
            "post_claim_max_turn_run": post_claim_max_turn_run,
            "active_jepa_veto_count": active_jepa_veto_count,
            "online_map_memory_ticks": online_map_memory_ticks,
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
            "learned_target_scheduler_checkpoint": wall.get(
                "learned_target_scheduler_checkpoint"
            ),
            "learned_target_scheduler_ticks": int(
                wall.get("learned_target_scheduler_ticks") or 0
            ),
            "learned_target_scheduler_switches": learned_target_scheduler_switches,
            "target_switches": target_switches,
            "rule_based_target_switches": rule_based_target_switches,
            "learned_local_policy_frontier_pressure_ticks": int(
                wall.get("learned_local_policy_frontier_pressure_ticks") or 0
            ),
            "learned_local_policy_frontier_pressure_overrides": int(
                wall.get("learned_local_policy_frontier_pressure_overrides") or 0
            ),
            "learned_local_policy_translation_pressure_ticks": int(
                wall.get("learned_local_policy_translation_pressure_ticks") or 0
            ),
            "learned_local_policy_translation_pressure_overrides": int(
                wall.get("learned_local_policy_translation_pressure_overrides") or 0
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


def _metric_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _max_primitive_run(
    rows: list[Any],
    primitives: set[str],
    *,
    min_tick: int | None = None,
) -> int:
    max_run = 0
    run = 0
    last: str | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if min_tick is not None:
            tick = row.get("tick")
            if isinstance(tick, (int, float)) and int(tick) < int(min_tick):
                continue
        primitive = str(row.get("primitive") or "")
        if primitive in primitives and primitive == last:
            run += 1
        elif primitive in primitives:
            run = 1
        else:
            run = 0
        last = primitive if primitive in primitives else None
        max_run = max(max_run, run)
    return max_run


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
