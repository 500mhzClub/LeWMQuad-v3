#!/usr/bin/env python3
"""Check whether a Go2 all-beacon candidate is good enough to render/demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _primitive_summary(log: list[Any]) -> dict[str, Any]:
    primitives: list[str] = []
    dist_sum = 0.0
    for entry in log:
        if not isinstance(entry, dict):
            continue
        primitive = entry.get("primitive")
        if primitive is None:
            continue
        name = str(primitive)
        primitives.append(name)
        dist = entry.get("executed_displacement_m")
        if isinstance(dist, (int, float)) and math.isfinite(float(dist)):
            dist_sum += float(dist)

    counts: dict[str, int] = {}
    transitions = 0
    runs = []
    last = None
    run_len = 0
    for name in primitives:
        counts[name] = counts.get(name, 0) + 1
        if last is None:
            last = name
            run_len = 1
        elif name == last:
            run_len += 1
        else:
            transitions += 1
            runs.append(run_len)
            last = name
            run_len = 1
    if run_len:
        runs.append(run_len)

    yaw = counts.get("yaw_left", 0) + counts.get("yaw_right", 0)
    forward = sum(counts.get(name, 0) for name in ("forward_slow", "forward_medium", "forward_fast"))
    backward = counts.get("backward", 0)
    n = max(1, len(primitives))
    runs_sorted = sorted(runs)
    median_run = runs_sorted[len(runs_sorted) // 2] if runs_sorted else 0
    return {
        "commands_total": len(primitives),
        "counts": counts,
        "yaw_share": yaw / n,
        "translation_share": (forward + backward) / n,
        "transition_rate": transitions / n,
        "transitions": transitions,
        "median_run_ticks": median_run,
        "distance_sum_m": dist_sum,
    }


def _get(result: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = result
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, default=None)
    parser.add_argument("--max-ticks", type=int, default=520)
    parser.add_argument("--max-yaw-share", type=float, default=0.58)
    parser.add_argument("--min-translation-share", type=float, default=0.34)
    parser.add_argument("--max-transition-rate", type=float, default=0.48)
    parser.add_argument("--min-median-run", type=int, default=2)
    parser.add_argument("--max-body-violations", type=int, default=0)
    parser.add_argument("--max-open-loop-final-error-m", type=float, default=0.75)
    args = parser.parse_args()

    payload = _load_json(args.result)
    result = payload.get("result", payload)
    log = payload.get("log", [])
    summary = _primitive_summary(log if isinstance(log, list) else [])
    wall = result.get("wall_metrics", {}) if isinstance(result, dict) else {}
    claimed = result.get("claimed_colors", []) if isinstance(result, dict) else []
    claimed_set = set(str(c) for c in claimed)
    all_claimed = {"red", "yellow", "blue", "green"}.issubset(claimed_set)
    ticks = int(result.get("ticks_used", summary["commands_total"]))
    success = bool(result.get("success"))
    body_violations = int(wall.get("body_clearance_violation_events") or 0)
    fall_events = int(wall.get("fall_events") or 0)
    tip_events = int(wall.get("tip_events") or 0)

    replay = _load_json(args.replay_report)
    open_loop_final_error = None
    replay_final = replay.get("final_xy")
    result_final = result.get("final_xy") if isinstance(result, dict) else None
    if (
        isinstance(replay_final, list)
        and isinstance(result_final, list)
        and len(replay_final) >= 2
        and len(result_final) >= 2
    ):
        open_loop_final_error = math.hypot(
            float(replay_final[0]) - float(result_final[0]),
            float(replay_final[1]) - float(result_final[1]),
        )

    gates = {
        "success": success,
        "all_beacons_claimed": all_claimed,
        "ticks": ticks <= int(args.max_ticks),
        "yaw_share": summary["yaw_share"] <= float(args.max_yaw_share),
        "translation_share": summary["translation_share"] >= float(args.min_translation_share),
        "transition_rate": summary["transition_rate"] <= float(args.max_transition_rate),
        "median_run_ticks": summary["median_run_ticks"] >= int(args.min_median_run),
        "body_clearance_violations": body_violations <= int(args.max_body_violations),
        "fall_tip_events": fall_events == 0 and tip_events == 0,
    }
    if open_loop_final_error is not None:
        gates["open_loop_final_error"] = open_loop_final_error <= float(
            args.max_open_loop_final_error_m
        )

    report = {
        "passed": all(gates.values()),
        "gates": gates,
        "result": {
            "path": str(args.result),
            "success": success,
            "ticks_used": ticks,
            "claimed_colors": claimed,
            "final_xy": result.get("final_xy") if isinstance(result, dict) else None,
            "body_clearance_violation_events": body_violations,
            "fall_events": fall_events,
            "tip_events": tip_events,
            "wall_vetoes": int(wall.get("wall_vetoes") or 0),
            "escape_blocks_executed": int(wall.get("escape_blocks_executed") or 0),
        },
        "commands": summary,
        "open_loop_final_error_m": open_loop_final_error,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
