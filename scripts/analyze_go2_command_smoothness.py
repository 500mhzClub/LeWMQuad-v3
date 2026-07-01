#!/usr/bin/env python3
"""Summarize command smoothness for a Go2 closed-loop result JSON."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _result(data: dict[str, Any]) -> dict[str, Any]:
    result = data.get("result", {})
    return result if isinstance(result, dict) else {}


def _wall_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("wall_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _runs(values: list[str]) -> list[tuple[str, int]]:
    if not values:
        return []
    out: list[tuple[str, int]] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            out.append((current, count))
            current = value
            count = 1
    out.append((current, count))
    return out


def summarize(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    result = _result(data)
    metrics = _wall_metrics(result)
    log = data.get("log", [])
    if not isinstance(log, list):
        log = []
    primitives = [str(row.get("primitive")) for row in log if isinstance(row, dict)]
    requested = [str(row.get("requested_primitive")) for row in log if isinstance(row, dict)]
    runs = _runs(primitives)
    requested_runs = _runs(requested)
    run_lengths = [count for _, count in runs]
    requested_run_lengths = [count for _, count in requested_runs]
    transitions = sum(1 for prev, cur in zip(primitives, primitives[1:]) if prev != cur)
    requested_transitions = sum(1 for prev, cur in zip(requested, requested[1:]) if prev != cur)
    return {
        "path": str(path),
        "success": bool(result.get("success")),
        "ticks_used": result.get("ticks_used"),
        "claimed_colors": result.get("claimed_colors"),
        "claim_distances_m": result.get("claim_distances_m"),
        "commands_total": metrics.get("commands_total"),
        "forward_executions": metrics.get("forward_executions"),
        "blocked_forward_executions": metrics.get("blocked_forward_executions"),
        "contact_like_stalls": metrics.get("contact_like_stalls"),
        "hard_contact_like_stalls": metrics.get("hard_contact_like_stalls"),
        "wall_vetoes": metrics.get("wall_vetoes"),
        "escape_blocks_executed": metrics.get("escape_blocks_executed"),
        "turn_loop_recoveries": metrics.get("turn_loop_recoveries"),
        "mean_forward_execution_displacement_m": metrics.get("mean_forward_execution_displacement_m"),
        "primitive_transitions": transitions,
        "primitive_transition_rate": (
            transitions / max(1, len(primitives) - 1) if primitives else None
        ),
        "requested_transitions": requested_transitions,
        "requested_transition_rate": (
            requested_transitions / max(1, len(requested) - 1) if requested else None
        ),
        "primitive_run_count": len(runs),
        "primitive_singleton_runs": sum(1 for _, count in runs if count == 1),
        "primitive_median_run_ticks": statistics.median(run_lengths) if run_lengths else None,
        "requested_singleton_runs": sum(1 for _, count in requested_runs if count == 1),
        "requested_median_run_ticks": (
            statistics.median(requested_run_lengths) if requested_run_lengths else None
        ),
        "primitive_counts": dict(sorted(Counter(primitives).items())),
        "requested_counts": dict(sorted(Counter(requested).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.results]
    payload: Any = summaries[0] if len(summaries) == 1 else summaries
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
