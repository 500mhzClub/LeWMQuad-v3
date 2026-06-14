#!/usr/bin/env python3
"""Audit target alignment and safety-label semantics in a scored decision index."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _decision_type(row: dict) -> str:
    return "+".join(sorted(str(value) for value in row["decision_types"]))


def _candidate_by_name(row: dict) -> dict[str, dict]:
    return {
        str(candidate["primitive_name"]): candidate
        for candidate in row["counterfactual_candidates"]
    }


def _audit_scored(path: Path) -> dict:
    rows = 0
    decision_types: Counter[str] = Counter()
    scored_target_rows = 0
    target_frame_rows = 0
    target_without_frame = 0
    target_frame_matches_scored_target = 0
    target_frame_mismatches_scored_target = 0
    hold_collisions = 0
    yaw_left_collisions = 0
    yaw_right_collisions = 0
    identical_stationary_collision_labels = 0
    logged_counterfactual_collisions = 0
    execution_clipped = 0
    counterfactual_collision_matches_clipped = 0
    target_by_type: dict[str, Counter[str]] = {}

    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            row_type = _decision_type(row)
            decision_types[row_type] += 1
            type_counts = target_by_type.setdefault(row_type, Counter())

            target = row.get("counterfactual_target_cell_id")
            target_frame = row.get("target_frame")
            has_target = target is not None
            has_target_frame = target_frame is not None
            scored_target_rows += has_target
            target_frame_rows += has_target_frame
            target_without_frame += has_target and not has_target_frame
            type_counts["rows"] += 1
            type_counts["scored_target_rows"] += has_target
            type_counts["target_frame_rows"] += has_target_frame
            if has_target and has_target_frame:
                # NOTE: this compares the *scored* target cell to route_target_id.
                # It equals an image-vs-scored-target comparison only because
                # mine_task_aligned_decisions.py renders target_frame from
                # route_target_id (see mining ~line 300); the frame's depicted
                # cell is not read here directly.
                matches = int(target) == int(row["route_target_id"])
                target_frame_matches_scored_target += matches
                target_frame_mismatches_scored_target += not matches
                type_counts["target_frame_matches_scored_target"] += matches
                type_counts["target_frame_mismatches_scored_target"] += not matches

            candidates = _candidate_by_name(row)
            stationary = [
                bool(candidates[name]["collided"])
                for name in ("hold", "yaw_left", "yaw_right")
            ]
            hold_collisions += stationary[0]
            yaw_left_collisions += stationary[1]
            yaw_right_collisions += stationary[2]
            identical_stationary_collision_labels += len(set(stationary)) == 1

            logged = candidates.get(str(row["primitive_name"]))
            if logged is not None:
                predicted_collision = bool(logged["collided"])
                clipped = bool(row.get("execution_clipped", False))
                logged_counterfactual_collisions += predicted_collision
                execution_clipped += clipped
                counterfactual_collision_matches_clipped += predicted_collision == clipped

    aligned_target_rows = (
        target_frame_matches_scored_target + target_frame_mismatches_scored_target
    )
    return {
        "path": str(path),
        "rows": rows,
        "decision_type_counts": dict(decision_types),
        "target_contract": {
            "scored_target_rows": scored_target_rows,
            "target_frame_rows": target_frame_rows,
            "target_without_frame_rows": target_without_frame,
            "target_without_frame_rate": _rate(target_without_frame, rows),
            "target_frame_matches_scored_target_rows": target_frame_matches_scored_target,
            "target_frame_mismatches_scored_target_rows": (
                target_frame_mismatches_scored_target
            ),
            "target_frame_mismatch_rate_among_target_frames": _rate(
                target_frame_mismatches_scored_target,
                aligned_target_rows,
            ),
            "by_decision_type": {
                name: dict(counts) for name, counts in target_by_type.items()
            },
        },
        "safety_contract": {
            "hold_collision_rate": _rate(hold_collisions, rows),
            "yaw_left_collision_rate": _rate(yaw_left_collisions, rows),
            "yaw_right_collision_rate": _rate(yaw_right_collisions, rows),
            "identical_hold_yaw_collision_label_rate": _rate(
                identical_stationary_collision_labels,
                rows,
            ),
            "logged_counterfactual_collision_rate": _rate(
                logged_counterfactual_collisions,
                rows,
            ),
            "execution_clipped_rate": _rate(execution_clipped, rows),
            "counterfactual_collision_matches_clipped_rate": _rate(
                counterfactual_collision_matches_clipped,
                rows,
            ),
        },
    }


def _audit_no_safe_rows(path: Path) -> dict:
    rows = 0
    no_safe_rows = 0
    decision_types: Counter[str] = Counter()
    execution_clipped = 0
    start_stuck = 0
    end_stuck = 0
    positive_actual_motion = 0

    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows += 1
            if not all(
                bool(candidate["collided"])
                for candidate in row["counterfactual_candidates"]
            ):
                continue
            no_safe_rows += 1
            decision_types[_decision_type(row)] += 1
            execution_clipped += bool(row.get("execution_clipped", False))
            start_stuck += bool(row.get("start_stuck", False))
            end_stuck += bool(row.get("end_stuck", False))
            motion = row.get("integrated_body_motion_block", [0.0, 0.0, 0.0])
            positive_actual_motion += math.hypot(float(motion[0]), float(motion[1])) > 1e-3

    return {
        "path": str(path),
        "rows": rows,
        "no_safe_action_rows": no_safe_rows,
        "no_safe_action_rate": _rate(no_safe_rows, rows),
        "decision_type_counts": dict(decision_types),
        "execution_clipped_rate": _rate(execution_clipped, no_safe_rows),
        "start_stuck_rate": _rate(start_stuck, no_safe_rows),
        "end_stuck_rate": _rate(end_stuck, no_safe_rows),
        "positive_actual_motion_rate": _rate(positive_actual_motion, no_safe_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument(
        "--unfiltered-scored",
        type=Path,
        help="Optional scored index that still contains all-candidate-collision rows.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "schema": "task_aligned_contract_audit_v0",
        "scored": _audit_scored(args.scored),
        "unfiltered": (
            _audit_no_safe_rows(args.unfiltered_scored)
            if args.unfiltered_scored is not None
            else None
        ),
        "stop_conditions": [
            "declare and separate final-route-goal and local-subgoal task contracts",
            "a local-action benchmark must show the same subgoal used by scoring",
            "target existence must not be inferred from target-frame availability",
            "no-safe-action filtering requires validated safety-label semantics",
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
