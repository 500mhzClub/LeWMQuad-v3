#!/usr/bin/env python3
"""Check closed-loop Go2 wall-aware planner result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_FORWARD_PRIMITIVES = {
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "arc_left",
    "arc_right",
}


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _result(report: dict[str, Any]) -> dict[str, Any]:
    result = report.get("result")
    return result if isinstance(result, dict) else {}


def _log(report: dict[str, Any]) -> list[dict[str, Any]]:
    entries = report.get("log")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = _result(report).get("wall_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_metric(metrics: dict[str, Any], key: str) -> int:
    return int(_num(metrics.get(key), 0.0))


def _guard_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in _log(report) if isinstance(entry.get("wall_guard"), dict)]


def _veto_entries(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry for entry in _guard_entries(report)
        if bool(entry["wall_guard"].get("vetoed"))
    ]


def _claim_entry(report: dict[str, Any]) -> dict[str, Any] | None:
    claims = [entry for entry in _log(report) if str(entry.get("state")) == "CLAIM"]
    return claims[-1] if claims else None


def _safe_reduction(before: float, after: float) -> float:
    if before <= 0.0:
        return 1.0 if after <= 0.0 else 0.0
    return (before - after) / before


def _states(entries: list[dict[str, Any]]) -> list[str]:
    return sorted({str(entry.get("state")) for entry in entries})


def evaluate_gate(
    baseline_explore: dict[str, Any],
    wallaware_explore: dict[str, Any],
    wallaware_recall: dict[str, Any],
    wallaware_escape: dict[str, Any] | None,
    *,
    allowed_guard_states: set[str],
    max_explore_final_dist_m: float,
    max_recall_final_dist_m: float,
    max_first_seen_delay_ticks: int,
    max_tick_regression: int,
    min_wall_vetoes: int,
    min_blocked_forward_count_reduction: float,
    min_blocked_forward_rate_reduction: float,
    max_stall_regression: int,
    min_baseline_stalls: int,
    max_claim_abs_bearing: float,
    min_claim_area: float,
    require_escape_artifact: bool,
    min_stuck_recoveries: int,
) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(
        component: str,
        name: str,
        passed: bool,
        observed: dict[str, Any],
        threshold: dict[str, Any] | None = None,
    ) -> None:
        item = {
            "component": component,
            "name": name,
            "passed": bool(passed),
            "observed": observed,
        }
        if threshold is not None:
            item["threshold"] = threshold
        checks.append(item)
        if not passed:
            failures.append(f"{component}:{name}")

    base_result = _result(baseline_explore)
    wall_result = _result(wallaware_explore)
    recall_result = _result(wallaware_recall)
    base_metrics = _metrics(baseline_explore)
    wall_metrics = _metrics(wallaware_explore)
    recall_metrics = _metrics(wallaware_recall)

    base_guard_entries = _guard_entries(baseline_explore)
    wall_guard_entries = _guard_entries(wallaware_explore)
    veto_entries = _veto_entries(wallaware_explore)
    enabled_entries = [
        entry for entry in wall_guard_entries if bool(entry["wall_guard"].get("enabled"))
    ]
    enabled_states = set(_states(enabled_entries))
    veto_states = set(_states(veto_entries))

    check(
        "baseline_closed_loop",
        "physical_explore_succeeded_with_diagnostics",
        bool(base_result.get("success")) and base_metrics.get("source") == "diagnostic_only"
        and not bool(base_metrics.get("enabled")) and len(base_guard_entries) > 0,
        {
            "success": bool(base_result.get("success")),
            "source": base_metrics.get("source"),
            "enabled": bool(base_metrics.get("enabled")),
            "guard_log_entries": len(base_guard_entries),
        },
    )
    check(
        "wallaware_closed_loop",
        "physical_explore_succeeded",
        bool(wall_result.get("success"))
        and bool(wall_result.get("claimed"))
        and _num(wall_result.get("final_dist_to_target_m"), 999.0) <= max_explore_final_dist_m,
        {
            "success": bool(wall_result.get("success")),
            "claimed": bool(wall_result.get("claimed")),
            "final_dist_to_target_m": wall_result.get("final_dist_to_target_m"),
        },
        {
            "max_explore_final_dist_m": max_explore_final_dist_m,
        },
    )
    check(
        "wallaware_closed_loop",
        "target_discovery_and_runtime_not_regressed",
        int(_num(wall_result.get("first_seen_tick"), 9999.0))
        <= int(_num(base_result.get("first_seen_tick"), 0.0)) + max_first_seen_delay_ticks
        and int(_num(wall_result.get("ticks_used"), 9999.0))
        <= int(_num(base_result.get("ticks_used"), 0.0)) + max_tick_regression,
        {
            "baseline_first_seen_tick": base_result.get("first_seen_tick"),
            "wallaware_first_seen_tick": wall_result.get("first_seen_tick"),
            "baseline_ticks_used": base_result.get("ticks_used"),
            "wallaware_ticks_used": wall_result.get("ticks_used"),
        },
        {
            "max_first_seen_delay_ticks": max_first_seen_delay_ticks,
            "max_tick_regression": max_tick_regression,
        },
    )
    check(
        "clearance_diagnostic",
        "requested_clearance_logged_in_closed_loop",
        len(wall_guard_entries) == _int_metric(wall_metrics, "commands_total")
        and _int_metric(wall_metrics, "blocked_forward_requests") > 0
        and any(isinstance(entry["wall_guard"].get("candidates"), list) for entry in wall_guard_entries),
        {
            "guard_log_entries": len(wall_guard_entries),
            "commands_total": wall_metrics.get("commands_total"),
            "blocked_forward_requests": wall_metrics.get("blocked_forward_requests"),
        },
    )
    check(
        "primitive_veto_arbitration",
        "vetoes_blocked_forward_requests",
        _int_metric(wall_metrics, "wall_vetoes") >= min_wall_vetoes
        and len(veto_entries) == _int_metric(wall_metrics, "wall_vetoes")
        and any(
            bool(entry["wall_guard"].get("requested_blocked"))
            and not bool(entry["wall_guard"].get("selected_blocked"))
            for entry in veto_entries
        ),
        {
            "wall_vetoes_metric": wall_metrics.get("wall_vetoes"),
            "wall_vetoes_log": len(veto_entries),
            "veto_states": sorted(veto_states),
        },
        {
            "min_wall_vetoes": min_wall_vetoes,
        },
    )

    base_blocked = _int_metric(base_metrics, "blocked_forward_executions")
    wall_blocked = _int_metric(wall_metrics, "blocked_forward_executions")
    base_forward = _int_metric(base_metrics, "forward_executions")
    wall_forward = _int_metric(wall_metrics, "forward_executions")
    base_rate = base_blocked / base_forward if base_forward else 0.0
    wall_rate = wall_blocked / wall_forward if wall_forward else 0.0
    count_reduction = _safe_reduction(float(base_blocked), float(wall_blocked))
    rate_reduction = _safe_reduction(base_rate, wall_rate)
    check(
        "primitive_veto_arbitration",
        "blocked_forward_executions_materially_reduced",
        wall_blocked < base_blocked
        and count_reduction >= min_blocked_forward_count_reduction
        and rate_reduction >= min_blocked_forward_rate_reduction,
        {
            "baseline_blocked_forward_executions": base_blocked,
            "wallaware_blocked_forward_executions": wall_blocked,
            "baseline_blocked_forward_rate": round(base_rate, 4),
            "wallaware_blocked_forward_rate": round(wall_rate, 4),
            "count_reduction": round(count_reduction, 4),
            "rate_reduction": round(rate_reduction, 4),
        },
        {
            "min_blocked_forward_count_reduction": min_blocked_forward_count_reduction,
            "min_blocked_forward_rate_reduction": min_blocked_forward_rate_reduction,
        },
    )
    check(
        "state_scoping",
        "vetoes_stay_in_allowed_states",
        enabled_states.issubset(allowed_guard_states) and veto_states.issubset(allowed_guard_states),
        {
            "allowed_guard_states": sorted(allowed_guard_states),
            "enabled_states": sorted(enabled_states),
            "veto_states": sorted(veto_states),
        },
    )

    base_stalls = _int_metric(base_metrics, "contact_like_stalls")
    wall_stalls = _int_metric(wall_metrics, "contact_like_stalls")
    wall_stall_log = [
        entry for entry in _log(wallaware_explore) if bool(entry.get("stalled"))
    ]
    check(
        "stall_detector",
        "contact_like_stalls_measured_and_not_regressed",
        base_stalls >= min_baseline_stalls
        and wall_stalls <= base_stalls + max_stall_regression
        and len(wall_stall_log) == wall_stalls,
        {
            "baseline_contact_like_stalls": base_stalls,
            "wallaware_contact_like_stalls": wall_stalls,
            "wallaware_stall_log_entries": len(wall_stall_log),
        },
        {
            "min_baseline_stalls": min_baseline_stalls,
            "max_stall_regression": max_stall_regression,
        },
    )

    claim = _claim_entry(wallaware_explore)
    claim_bearing = _num(claim.get("bearing"), 999.0) if claim else 999.0
    claim_area = _num(claim.get("area"), -999.0) if claim else -999.0
    check(
        "claim_calibration",
        "claim_is_centered_and_large_enough",
        claim is not None
        and abs(claim_bearing) <= max_claim_abs_bearing
        and claim_area >= min_claim_area,
        {
            "claim_tick": claim.get("tick") if claim else None,
            "claim_bearing": claim.get("bearing") if claim else None,
            "claim_area": claim.get("area") if claim else None,
        },
        {
            "max_claim_abs_bearing": max_claim_abs_bearing,
            "min_claim_area": min_claim_area,
        },
    )
    check(
        "recall_preservation",
        "physical_recall_still_succeeds",
        bool(recall_result.get("success"))
        and bool(recall_result.get("claimed"))
        and _num(recall_result.get("final_dist_to_target_m"), 999.0) <= max_recall_final_dist_m,
        {
            "success": bool(recall_result.get("success")),
            "claimed": bool(recall_result.get("claimed")),
            "final_dist_to_target_m": recall_result.get("final_dist_to_target_m"),
            "wall_vetoes": recall_metrics.get("wall_vetoes"),
        },
        {
            "max_recall_final_dist_m": max_recall_final_dist_m,
        },
    )

    if wallaware_escape is not None:
        escape_result = _result(wallaware_escape)
        escape_metrics = _metrics(wallaware_escape)
        force_escape_log = [
            entry for entry in _guard_entries(wallaware_escape)
            if bool(entry["wall_guard"].get("force_escape"))
        ]
        check(
            "escape_hook",
            "stuck_recovery_exercised_in_closed_loop",
            bool(escape_result.get("success"))
            and _int_metric(escape_metrics, "stuck_recoveries") >= min_stuck_recoveries
            and _int_metric(escape_metrics, "escape_blocks_executed") >= min_stuck_recoveries
            and len(force_escape_log) >= min_stuck_recoveries,
            {
                "success": bool(escape_result.get("success")),
                "stuck_recoveries": escape_metrics.get("stuck_recoveries"),
                "escape_blocks_executed": escape_metrics.get("escape_blocks_executed"),
                "force_escape_log_entries": len(force_escape_log),
            },
            {
                "min_stuck_recoveries": min_stuck_recoveries,
            },
        )
    elif require_escape_artifact:
        check(
            "escape_hook",
            "stuck_recovery_exercised_in_closed_loop",
            False,
            {"wallaware_escape_artifact": None},
            {"required": True},
        )
    else:
        checks.append({
            "component": "escape_hook",
            "name": "stuck_recovery_exercised_in_closed_loop",
            "passed": None,
            "observed": {"wallaware_escape_artifact": None},
            "note": "not evaluated; pass --wallaware-escape or --require-escape-artifact",
        })

    return {
        "schema": "go2_wallaware_closed_loop_gate_v0",
        "passed": not failures,
        "failure_reasons": failures,
        "checks": checks,
        "summary": {
            "baseline_scene": base_result.get("scene"),
            "wallaware_scene": wall_result.get("scene"),
            "recall_scene": recall_result.get("scene"),
            "blocked_forward_executions": {
                "baseline": base_blocked,
                "wallaware": wall_blocked,
            },
            "contact_like_stalls": {
                "baseline": base_stalls,
                "wallaware": wall_stalls,
            },
            "wall_vetoes": wall_metrics.get("wall_vetoes"),
            "claim_bearing": claim.get("bearing") if claim else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-explore", type=Path, required=True)
    parser.add_argument("--wallaware-explore", type=Path, required=True)
    parser.add_argument("--wallaware-recall", type=Path, required=True)
    parser.add_argument("--wallaware-escape", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allowed-guard-states", default="EXPLORE")
    parser.add_argument("--max-explore-final-dist-m", type=float, default=1.2)
    parser.add_argument("--max-recall-final-dist-m", type=float, default=1.2)
    parser.add_argument("--max-first-seen-delay-ticks", type=int, default=5)
    parser.add_argument("--max-tick-regression", type=int, default=10)
    parser.add_argument("--min-wall-vetoes", type=int, default=1)
    parser.add_argument("--min-blocked-forward-count-reduction", type=float, default=0.25)
    parser.add_argument("--min-blocked-forward-rate-reduction", type=float, default=0.25)
    parser.add_argument("--max-stall-regression", type=int, default=0)
    parser.add_argument("--min-baseline-stalls", type=int, default=1)
    parser.add_argument("--max-claim-abs-bearing", type=float, default=0.3)
    parser.add_argument("--min-claim-area", type=float, default=1.5)
    parser.add_argument("--require-escape-artifact", action="store_true")
    parser.add_argument("--min-stuck-recoveries", type=int, default=1)
    args = parser.parse_args()

    allowed_states = {
        state.strip().upper()
        for state in str(args.allowed_guard_states).split(",")
        if state.strip()
    }
    result = evaluate_gate(
        _load_json(args.baseline_explore),
        _load_json(args.wallaware_explore),
        _load_json(args.wallaware_recall),
        _load_json(args.wallaware_escape) if args.wallaware_escape is not None else None,
        allowed_guard_states=allowed_states,
        max_explore_final_dist_m=float(args.max_explore_final_dist_m),
        max_recall_final_dist_m=float(args.max_recall_final_dist_m),
        max_first_seen_delay_ticks=int(args.max_first_seen_delay_ticks),
        max_tick_regression=int(args.max_tick_regression),
        min_wall_vetoes=int(args.min_wall_vetoes),
        min_blocked_forward_count_reduction=float(args.min_blocked_forward_count_reduction),
        min_blocked_forward_rate_reduction=float(args.min_blocked_forward_rate_reduction),
        max_stall_regression=int(args.max_stall_regression),
        min_baseline_stalls=int(args.min_baseline_stalls),
        max_claim_abs_bearing=float(args.max_claim_abs_bearing),
        min_claim_area=float(args.min_claim_area),
        require_escape_artifact=bool(args.require_escape_artifact),
        min_stuck_recoveries=int(args.min_stuck_recoveries),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
