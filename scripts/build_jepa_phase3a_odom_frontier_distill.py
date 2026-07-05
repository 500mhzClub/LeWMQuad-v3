#!/usr/bin/env python3
"""Build Phase 3A closed-loop odometry-frontier planner distillation rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    _state_dict,
    read_jsonl,
    step_state,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import (  # noqa: E402
    _candidate_rows,
    _goal_scene_from_row,
    _group_validation_sources,
    _select_odom_frontier_action,
    _state_from_dict,
    _subsample_candidates_by_first,
    _update_odom_frontier_memory,
)


def _build_split(
    groups: list[list[dict]],
    *,
    seed: int,
    width: int,
    height: int,
    view_size: int,
    horizon: int,
    max_steps: int,
    history_window: int,
    max_groups: int | None,
    max_candidates_per_state: int | None,
    positive_utility: float,
    negative_utility: float,
) -> list[dict]:
    output_rows: list[dict] = []
    for group_index, group in enumerate(groups[:max_groups]):
        template = group[0]
        scene = _goal_scene_from_row(template, seed=seed, width=width, height=height)
        state = _state_from_dict(template["start_state"])
        history_states = [_state_from_dict(item) for item in template["history_states"]]
        history_actions = [str(item) for item in template["history_primitive_sequence"]]
        memory = {"free": set(), "blocked": set(), "marker": None}
        for step in range(max_steps):
            if (state.x, state.y) == scene.goal:
                break
            _update_odom_frontier_memory(
                memory,
                scene=scene,
                state=state,
                view_size=view_size,
                current_goal_marker=True,
            )
            planner_action = _select_odom_frontier_action(memory, state)
            if history_window > 0:
                candidate_history_states = history_states[-history_window:]
                candidate_history_actions = history_actions[-history_window:]
            else:
                candidate_history_states = history_states
                candidate_history_actions = history_actions
            rows = _candidate_rows(
                scene=scene,
                source_index=group_index * max_steps + step,
                state=state,
                history_states=candidate_history_states,
                history_actions=candidate_history_actions,
                horizon=horizon,
                view_size=view_size,
                current_goal_marker=True,
            )
            rows = _subsample_candidates_by_first(rows, max_candidates_per_state)
            for row in rows:
                labels = dict(row["consequence_labels"])
                first_action = str(row["primitive_sequence"][0])
                labels["target_utility"] = (
                    positive_utility
                    if first_action == planner_action
                    else negative_utility
                )
                labels["target_planner_action"] = planner_action
                labels["target_planner_action_match"] = first_action == planner_action
                row["consequence_labels"] = labels
                row["utility_mode"] = "odom_frontier_planner_distill"
                row["history_policy"] = "odom_frontier_planner_distill"
                row["planner_state"] = {
                    "step": step,
                    "planner_action": planner_action,
                    "planner_marker_known": memory.get("marker") is not None,
                    "planner_known_free_cells": len(memory["free"]),
                    "planner_known_blocked_cells": len(memory["blocked"]),
                    "closed_loop_state": _state_dict(state),
                }
            output_rows.extend(rows)
            history_states.append(state)
            history_actions.append(planner_action)
            state, _collision = step_state(scene, state, planner_action)
    return output_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=56)
    parser.add_argument("--history-window", type=int, default=6)
    parser.add_argument("--max-train-groups", type=int, default=None)
    parser.add_argument("--max-validation-groups", type=int, default=16)
    parser.add_argument(
        "--max-candidates-per-state",
        type=int,
        default=None,
        help="subsample candidate rows per planner state for compact distillation",
    )
    parser.add_argument("--positive-utility", type=float, default=10.0)
    parser.add_argument("--negative-utility", type=float, default=0.0)
    args = parser.parse_args()

    train_rows = read_jsonl(args.input_dir / "train_phase3a_positive_control.jsonl")
    validation_rows = read_jsonl(
        args.input_dir / "validation_phase3a_positive_control.jsonl"
    )
    train_groups = _group_validation_sources(train_rows)
    validation_groups = _group_validation_sources(validation_rows)
    distill_train = _build_split(
        train_groups,
        seed=args.train_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        horizon=args.horizon,
        max_steps=args.max_steps,
        history_window=args.history_window,
        max_groups=args.max_train_groups,
        max_candidates_per_state=args.max_candidates_per_state,
        positive_utility=args.positive_utility,
        negative_utility=args.negative_utility,
    )
    distill_validation = _build_split(
        validation_groups,
        seed=args.validation_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        horizon=args.horizon,
        max_steps=args.max_steps,
        history_window=args.history_window,
        max_groups=args.max_validation_groups,
        max_candidates_per_state=args.max_candidates_per_state,
        positive_utility=args.positive_utility,
        negative_utility=args.negative_utility,
    )
    train_path = args.output_dir / "train_phase3a_positive_control.jsonl"
    validation_path = args.output_dir / "validation_phase3a_positive_control.jsonl"
    _write_jsonl(train_path, distill_train)
    _write_jsonl(validation_path, distill_validation)
    manifest = {
        "schema": "jepa_phase3a_odom_frontier_distill_manifest_v0",
        "input_dir": str(args.input_dir.resolve()),
        "train_data": str(train_path.resolve()),
        "validation_data": str(validation_path.resolve()),
        "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "train_groups": len(train_groups[: args.max_train_groups]),
        "validation_groups": len(validation_groups[: args.max_validation_groups]),
        "train_rows": len(distill_train),
        "validation_rows": len(distill_validation),
        "max_steps": args.max_steps,
        "max_candidates_per_state": args.max_candidates_per_state,
        "history_window": args.history_window,
        "positive_utility": args.positive_utility,
        "negative_utility": args.negative_utility,
    }
    manifest_path = args.output_dir / "phase3a_odom_frontier_distill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
