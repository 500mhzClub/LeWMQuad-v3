#!/usr/bin/env python3
"""Audit coverage and consequence diversity in JEPA counterfactual indexes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _mean(total: float, count: int) -> float:
    return total / count if count else 0.0


def _new_group() -> dict:
    return {
        "rows": 0,
        "target_rows": 0,
        "target_rows_with_local_target_frame": 0,
        "starts_grid_unsafe": 0,
        "candidates": 0,
        "enters_grid_unsafe": 0,
        "ends_grid_unsafe": 0,
        "target_recoverable": 0,
        "target_recoverable_candidates": 0,
        "safe_positive_progress": 0,
        "target_progress_sum": 0.0,
        "target_progress_candidates": 0,
        "p05_clearance_sum": 0.0,
        "oracle_enters_grid_unsafe": 0,
        "oracle_ends_grid_unsafe": 0,
        "oracle_target_recoverable": 0,
        "oracle_target_recoverable_rows": 0,
        "oracle_safe_positive_progress": 0,
        "oracle_target_progress_sum": 0.0,
        "oracle_target_rows": 0,
        "oracle_first_primitives": Counter(),
    }


def _update(group: dict, row: dict) -> None:
    group["rows"] += 1
    target = row.get("counterfactual_target_cell_id") is not None
    local_frame = row.get("local_target_frame")
    group["target_rows"] += target
    group["target_rows_with_local_target_frame"] += (
        target and local_frame is not None and Path(str(local_frame)).is_file()
    )
    candidates = row["counterfactual_candidates"]
    group["starts_grid_unsafe"] += bool(candidates[0]["starts_grid_unsafe"])
    for candidate in candidates:
        group["candidates"] += 1
        group["enters_grid_unsafe"] += bool(candidate["enters_grid_unsafe"])
        group["ends_grid_unsafe"] += bool(candidate["ends_grid_unsafe"])
        group["p05_clearance_sum"] += float(
            candidate["p05_swept_configuration_clearance_m"]
        )
        if candidate["target_recoverable"] is not None:
            group["target_recoverable_candidates"] += 1
            group["target_recoverable"] += bool(candidate["target_recoverable"])
        if candidate["target_progress_m"] is not None:
            progress = float(candidate["target_progress_m"])
            group["target_progress_candidates"] += 1
            group["target_progress_sum"] += progress
            group["safe_positive_progress"] += (
                progress > 0.0
                and not candidate["enters_grid_unsafe"]
                and not candidate["ends_grid_unsafe"]
                and candidate["target_recoverable"] is not False
            )

    oracle = candidates[int(row["counterfactual_oracle_index"])]
    group["oracle_first_primitives"][oracle["primitive_sequence"][0]] += 1
    group["oracle_enters_grid_unsafe"] += bool(oracle["enters_grid_unsafe"])
    group["oracle_ends_grid_unsafe"] += bool(oracle["ends_grid_unsafe"])
    if oracle["target_recoverable"] is not None:
        group["oracle_target_recoverable_rows"] += 1
        group["oracle_target_recoverable"] += bool(oracle["target_recoverable"])
    if oracle["target_progress_m"] is not None:
        progress = float(oracle["target_progress_m"])
        group["oracle_target_rows"] += 1
        group["oracle_target_progress_sum"] += progress
        group["oracle_safe_positive_progress"] += (
            progress > 0.0
            and not oracle["enters_grid_unsafe"]
            and not oracle["ends_grid_unsafe"]
            and oracle["target_recoverable"] is not False
        )


def _finalize(group: dict) -> dict:
    rows = group["rows"]
    candidates = group["candidates"]
    target_candidates = group["target_progress_candidates"]
    oracle_target_rows = group["oracle_target_rows"]
    return {
        "rows": rows,
        "target_rows": group["target_rows"],
        "target_rows_with_local_target_frame": group[
            "target_rows_with_local_target_frame"
        ],
        "target_alignment_rate": _mean(
            group["target_rows_with_local_target_frame"], group["target_rows"]
        ),
        "starts_grid_unsafe_rate": _mean(group["starts_grid_unsafe"], rows),
        "candidate_count": candidates,
        "candidate_enters_grid_unsafe_rate": _mean(
            group["enters_grid_unsafe"], candidates
        ),
        "candidate_ends_grid_unsafe_rate": _mean(group["ends_grid_unsafe"], candidates),
        "candidate_target_recoverable_rate": _mean(
            group["target_recoverable"], group["target_recoverable_candidates"]
        ),
        "candidate_safe_positive_progress_rate": _mean(
            group["safe_positive_progress"], target_candidates
        ),
        "candidate_mean_target_progress_m": _mean(
            group["target_progress_sum"], target_candidates
        ),
        "candidate_mean_p05_clearance_m": _mean(
            group["p05_clearance_sum"], candidates
        ),
        "oracle_enters_grid_unsafe_rate": _mean(
            group["oracle_enters_grid_unsafe"], rows
        ),
        "oracle_ends_grid_unsafe_rate": _mean(group["oracle_ends_grid_unsafe"], rows),
        "oracle_target_recoverable_rate": _mean(
            group["oracle_target_recoverable"], group["oracle_target_recoverable_rows"]
        ),
        "oracle_safe_positive_progress_rate": _mean(
            group["oracle_safe_positive_progress"], oracle_target_rows
        ),
        "oracle_mean_target_progress_m": _mean(
            group["oracle_target_progress_sum"], oracle_target_rows
        ),
        "oracle_first_primitive_counts": dict(group["oracle_first_primitives"]),
    }


def _audit(path: Path) -> dict:
    overall = _new_group()
    by_family = defaultdict(_new_group)
    by_decision_type = defaultdict(_new_group)
    scenes: set[str] = set()
    sequence_count = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            scenes.add(str(row["scene_id"]))
            current_sequence_count = len(row["counterfactual_candidates"])
            if sequence_count is None:
                sequence_count = current_sequence_count
            elif sequence_count != current_sequence_count:
                raise ValueError("candidate sequence count differs between rows")
            _update(overall, row)
            _update(by_family[str(row["family"])], row)
            decision_type = "+".join(sorted(str(value) for value in row["decision_types"]))
            _update(by_decision_type[decision_type], row)
    result = _finalize(overall)
    result.update(
        {
            "path": str(path.resolve()),
            "scene_count": len(scenes),
            "scene_ids": sorted(scenes),
            "sequences_per_row": sequence_count or 0,
            "by_family": {
                name: _finalize(group) for name, group in sorted(by_family.items())
            },
            "by_decision_type": {
                name: _finalize(group)
                for name, group in sorted(by_decision_type.items())
            },
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = _audit(args.train)
    evaluation = _audit(args.eval)
    overlap = sorted(set(train["scene_ids"]) & set(evaluation["scene_ids"]))
    checks = {
        "scene_disjoint": not overlap,
        "train_target_alignment_complete": train["target_alignment_rate"] == 1.0,
        "eval_target_alignment_complete": evaluation["target_alignment_rate"] == 1.0,
        "train_has_unsafe_candidates": train["candidate_enters_grid_unsafe_rate"] >= 0.05,
        "eval_has_unsafe_candidates": evaluation["candidate_enters_grid_unsafe_rate"] >= 0.05,
        "train_has_safe_progress_candidates": (
            train["candidate_safe_positive_progress_rate"] >= 0.05
        ),
        "eval_has_safe_progress_candidates": (
            evaluation["candidate_safe_positive_progress_rate"] >= 0.05
        ),
        "train_oracle_avoids_new_unsafe": train["oracle_enters_grid_unsafe_rate"] == 0.0,
        "eval_oracle_avoids_new_unsafe": evaluation["oracle_enters_grid_unsafe_rate"] == 0.0,
    }
    report = {
        "schema": "jepa_counterfactual_coverage_audit_v0",
        "train": train,
        "eval": evaluation,
        "scene_overlap": overlap,
        "contract_checks": checks,
        "contract_gate_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "contract_checks": checks,
                "contract_gate_passed": report["contract_gate_passed"],
                "train": {key: value for key, value in train.items() if key not in {"scene_ids", "by_family", "by_decision_type"}},
                "eval": {key: value for key, value in evaluation.items() if key not in {"scene_ids", "by_family", "by_decision_type"}},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
