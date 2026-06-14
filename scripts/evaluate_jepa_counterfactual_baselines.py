#!/usr/bin/env python3
"""Evaluate model-agnostic controls on the JEPA counterfactual benchmark."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError(f"benchmark is empty: {path}")
    return rows


def _sequence(candidate: dict) -> tuple[str, ...]:
    return tuple(str(value) for value in candidate["primitive_sequence"])


def _candidate_map(row: dict) -> dict[tuple[str, ...], dict]:
    return {_sequence(candidate): candidate for candidate in row["counterfactual_candidates"]}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _selection_metrics(rows: list[dict], selected: list[dict]) -> dict:
    target_progress = [
        float(candidate["target_progress_m"])
        for candidate in selected
        if candidate["target_progress_m"] is not None
    ]
    heading_error = [
        float(candidate["target_heading_error_rad"])
        for candidate in selected
        if candidate["target_heading_error_rad"] is not None
    ]
    recoverable = [
        bool(candidate["target_recoverable"])
        for candidate in selected
        if candidate["target_recoverable"] is not None
    ]
    safe_progress = [
        candidate["target_progress_m"] is not None
        and float(candidate["target_progress_m"]) > 0.0
        and not bool(candidate["enters_grid_unsafe"])
        and not bool(candidate["ends_grid_unsafe"])
        and candidate["target_recoverable"] is not False
        for candidate in selected
    ]
    oracle_matches = [
        _sequence(candidate)
        == tuple(str(value) for value in row["counterfactual_oracle_primitive_sequence"])
        for row, candidate in zip(rows, selected, strict=True)
    ]
    return {
        "row_count": len(rows),
        "selected_enters_grid_unsafe_rate": _mean(
            [float(candidate["enters_grid_unsafe"]) for candidate in selected]
        ),
        "selected_ends_grid_unsafe_rate": _mean(
            [float(candidate["ends_grid_unsafe"]) for candidate in selected]
        ),
        "selected_target_recoverable_rate": _mean([float(value) for value in recoverable]),
        "mean_target_progress_m": _mean(target_progress),
        "mean_p05_swept_configuration_clearance_m": _mean(
            [
                float(candidate["p05_swept_configuration_clearance_m"])
                for candidate in selected
            ]
        ),
        "mean_minimum_swept_configuration_clearance_m": _mean(
            [
                float(candidate["minimum_swept_configuration_clearance_m"])
                for candidate in selected
            ]
        ),
        "mean_target_heading_error_rad": _mean(heading_error),
        "safe_positive_progress_rate": _mean([float(value) for value in safe_progress]),
        "oracle_sequence_match_rate": _mean([float(value) for value in oracle_matches]),
    }


def _random_expected_metrics(rows: list[dict]) -> dict:
    metric_names = (
        "enters_grid_unsafe",
        "ends_grid_unsafe",
        "p05_swept_configuration_clearance_m",
        "minimum_swept_configuration_clearance_m",
    )
    result = {
        f"selected_{name}_rate" if name.endswith("unsafe") else f"mean_{name}": _mean(
            [
                _mean([float(candidate[name]) for candidate in row["counterfactual_candidates"]])
                for row in rows
            ]
        )
        for name in metric_names
    }
    result["selected_target_recoverable_rate"] = _mean(
        [
            _mean(
                [
                    float(candidate["target_recoverable"])
                    for candidate in row["counterfactual_candidates"]
                    if candidate["target_recoverable"] is not None
                ]
            )
            for row in rows
        ]
    )
    result["mean_target_progress_m"] = _mean(
        [
            _mean(
                [
                    float(candidate["target_progress_m"])
                    for candidate in row["counterfactual_candidates"]
                    if candidate["target_progress_m"] is not None
                ]
            )
            for row in rows
        ]
    )
    result["mean_target_heading_error_rad"] = _mean(
        [
            _mean(
                [
                    float(candidate["target_heading_error_rad"])
                    for candidate in row["counterfactual_candidates"]
                    if candidate["target_heading_error_rad"] is not None
                ]
            )
            for row in rows
        ]
    )
    result["safe_positive_progress_rate"] = _mean(
        [
            _mean(
                [
                    float(
                        candidate["target_progress_m"] is not None
                        and float(candidate["target_progress_m"]) > 0.0
                        and not bool(candidate["enters_grid_unsafe"])
                        and not bool(candidate["ends_grid_unsafe"])
                        and candidate["target_recoverable"] is not False
                    )
                    for candidate in row["counterfactual_candidates"]
                ]
            )
            for row in rows
        ]
    )
    result["oracle_sequence_match_rate"] = _mean(
        [1.0 / len(row["counterfactual_candidates"]) for row in rows]
    )
    result["row_count"] = len(rows)
    return result


def _subset(values: list, indices: list[int]) -> list:
    return [values[index] for index in indices]


def _action_only_prior(train_rows: list[dict]) -> tuple[str, ...]:
    aggregates: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in train_rows:
        for candidate in row["counterfactual_candidates"]:
            values = aggregates[_sequence(candidate)]
            values["enters"].append(float(candidate["enters_grid_unsafe"]))
            values["ends"].append(float(candidate["ends_grid_unsafe"]))
            if candidate["target_recoverable"] is not None:
                values["unrecoverable"].append(float(not candidate["target_recoverable"]))
            task_gain = (
                candidate["target_progress_m"]
                if candidate["target_progress_m"] is not None
                else candidate["clearance_gain_m"]
            )
            values["task_gain"].append(float(task_gain))
            values["p05"].append(float(candidate["p05_swept_configuration_clearance_m"]))
            if candidate["target_heading_error_rad"] is not None:
                values["heading"].append(float(candidate["target_heading_error_rad"]))
            values["path"].append(float(candidate["path_length_m"]))

    def aggregate_key(sequence: tuple[str, ...]) -> tuple:
        values = aggregates[sequence]
        return (
            _mean(values["enters"]),
            _mean(values["ends"]),
            _mean(values["unrecoverable"]),
            -_mean(values["task_gain"]),
            -_mean(values["p05"]),
            _mean(values["heading"]),
            _mean(values["path"]),
            sequence,
        )

    return min(aggregates, key=aggregate_key)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-scene-overlap", action="store_true")
    args = parser.parse_args()

    train_rows = _load_rows(args.train)
    eval_rows = _load_rows(args.eval)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    eval_scenes = {str(row["scene_id"]) for row in eval_rows}
    scene_overlap = sorted(train_scenes & eval_scenes)
    if scene_overlap and not args.allow_scene_overlap:
        raise SystemExit(
            "train/eval scene overlap violates the benchmark contract: "
            + ", ".join(scene_overlap[:8])
        )
    action_only_sequence = _action_only_prior(train_rows)

    oracle = [
        row["counterfactual_candidates"][int(row["counterfactual_oracle_index"])]
        for row in eval_rows
    ]
    action_only = [_candidate_map(row)[action_only_sequence] for row in eval_rows]
    logged_then_hold = []
    logged_fallbacks = 0
    for row in eval_rows:
        candidate_map = _candidate_map(row)
        horizon = int(row["counterfactual_horizon_blocks"])
        sequence = (str(row["primitive_name"]),) + ("hold",) * (horizon - 1)
        if sequence not in candidate_map:
            sequence = ("hold",) * horizon
            logged_fallbacks += 1
        logged_then_hold.append(candidate_map[sequence])

    controls = {
        "oracle": _selection_metrics(eval_rows, oracle),
        "action_only_prior": _selection_metrics(eval_rows, action_only),
        "logged_then_hold": _selection_metrics(eval_rows, logged_then_hold),
        "random_expected": _random_expected_metrics(eval_rows),
    }
    target_indices = [
        index
        for index, row in enumerate(eval_rows)
        if row.get("counterfactual_target_cell_id") is not None
    ]
    target_rows = _subset(eval_rows, target_indices)
    target_conditioned_controls = {
        "oracle": _selection_metrics(target_rows, _subset(oracle, target_indices)),
        "action_only_prior": _selection_metrics(
            target_rows, _subset(action_only, target_indices)
        ),
        "logged_then_hold": _selection_metrics(
            target_rows, _subset(logged_then_hold, target_indices)
        ),
        "random_expected": _random_expected_metrics(target_rows),
    }
    contract_checks = {
        "scene_disjoint": not scene_overlap,
        "logged_actions_covered": logged_fallbacks == 0,
        "random_contains_new_unsafe_actions": (
            controls["random_expected"]["selected_enters_grid_unsafe_rate"] >= 0.05
        ),
        "random_contains_safe_progress_actions": (
            controls["random_expected"]["safe_positive_progress_rate"] >= 0.05
        ),
        "oracle_avoids_new_unsafe_actions": (
            controls["oracle"]["selected_enters_grid_unsafe_rate"] == 0.0
        ),
        "oracle_improves_safe_progress_over_action_only": (
            controls["oracle"]["safe_positive_progress_rate"]
            >= controls["action_only_prior"]["safe_positive_progress_rate"] + 0.10
        ),
    }
    report = {
        "schema": "jepa_counterfactual_baselines_v0",
        "train": str(args.train.resolve()),
        "eval": str(args.eval.resolve()),
        "train_scene_count": len(train_scenes),
        "eval_scene_count": len(eval_scenes),
        "scene_overlap": scene_overlap,
        "action_only_prior_sequence": list(action_only_sequence),
        "logged_then_hold_fallback_rows": logged_fallbacks,
        "contract_checks": contract_checks,
        "contract_gate_passed": all(contract_checks.values()),
        "controls": controls,
        "target_conditioned_controls": target_conditioned_controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
