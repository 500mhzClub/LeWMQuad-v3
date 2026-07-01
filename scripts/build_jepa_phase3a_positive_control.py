#!/usr/bin/env python3
"""Build Phase 3A positive-control JEPA navigation datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    generate_phase3a_rows,
    phase3a_action_only_prior,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--train-scenes", type=int, default=16)
    parser.add_argument("--validation-scenes", type=int, default=8)
    parser.add_argument("--source-states-per-scene", type=int, default=16)
    parser.add_argument("--minimum-source-goal-distance", type=int, default=3)
    parser.add_argument("--maximum-source-goal-distance", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--view-size", type=int, default=9)
    parser.add_argument("--width", type=int, default=13)
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--history-steps", type=int, default=0)
    parser.add_argument(
        "--history-policy",
        choices=("turning", "explore"),
        default="turning",
        help="history action policy used before each source state",
    )
    parser.add_argument(
        "--utility-mode",
        choices=("goal_progress", "explore_then_claim", "novelty_then_claim"),
        default="goal_progress",
        help=(
            "target utility semantics: direct goal progress, novelty plus "
            "privileged discovery bonus before visual marker sighting, or pure "
            "novelty before marker sighting and goal claiming after sighting"
        ),
    )
    parser.add_argument(
        "--color-palette-mode",
        choices=("fixed", "scene_random"),
        default="fixed",
        help=(
            "rendering colors for floor/wall/marker: fixed legacy colors or "
            "deterministic per-scene randomized colors"
        ),
    )
    parser.add_argument(
        "--goal-variants-per-source",
        type=int,
        default=1,
        help=(
            "generate multiple hidden-goal aliases for each sampled source state; "
            "current observations remain identical when current goal cues are disabled"
        ),
    )
    parser.add_argument("--minimum-goal-variant-distance", type=int, default=3)
    parser.add_argument("--maximum-goal-variant-distance", type=int, default=None)
    parser.add_argument(
        "--no-current-goal-beacon",
        action="store_true",
        help="omit the goal/topology beacon from the current source observation",
    )
    parser.add_argument(
        "--no-history-goal-beacon",
        action="store_true",
        help="omit the goal/topology beacon from history observations",
    )
    parser.add_argument(
        "--no-current-goal-marker",
        action="store_true",
        help="render the goal cell as ordinary free space in the current source observation",
    )
    parser.add_argument(
        "--no-history-goal-marker",
        action="store_true",
        help="render the goal cell as ordinary free space in history observations",
    )
    parser.add_argument(
        "--no-future-goal-marker",
        action="store_true",
        help="render the goal cell as ordinary free space in future target observations",
    )
    args = parser.parse_args()

    train_rows, train_audit = generate_phase3a_rows(
        split="train",
        scene_count=args.train_scenes,
        source_states_per_scene=args.source_states_per_scene,
        seed=args.seed,
        horizon=args.horizon,
        view_size=args.view_size,
        width=args.width,
        height=args.height,
        history_steps=args.history_steps,
        current_goal_beacon=not args.no_current_goal_beacon,
        history_goal_beacon=not args.no_history_goal_beacon,
        current_goal_marker=not args.no_current_goal_marker,
        history_goal_marker=not args.no_history_goal_marker,
        future_goal_marker=not args.no_future_goal_marker,
        goal_variants_per_source=args.goal_variants_per_source,
        history_policy=args.history_policy,
        utility_mode=args.utility_mode,
        minimum_source_goal_distance=args.minimum_source_goal_distance,
        maximum_source_goal_distance=args.maximum_source_goal_distance,
        minimum_goal_variant_distance=args.minimum_goal_variant_distance,
        maximum_goal_variant_distance=args.maximum_goal_variant_distance,
        color_palette_mode=args.color_palette_mode,
    )
    validation_rows, validation_audit = generate_phase3a_rows(
        split="validation",
        scene_count=args.validation_scenes,
        source_states_per_scene=args.source_states_per_scene,
        seed=args.seed + 1_000_003,
        horizon=args.horizon,
        view_size=args.view_size,
        width=args.width,
        height=args.height,
        history_steps=args.history_steps,
        current_goal_beacon=not args.no_current_goal_beacon,
        history_goal_beacon=not args.no_history_goal_beacon,
        current_goal_marker=not args.no_current_goal_marker,
        history_goal_marker=not args.no_history_goal_marker,
        future_goal_marker=not args.no_future_goal_marker,
        goal_variants_per_source=args.goal_variants_per_source,
        history_policy=args.history_policy,
        utility_mode=args.utility_mode,
        minimum_source_goal_distance=args.minimum_source_goal_distance,
        maximum_source_goal_distance=args.maximum_source_goal_distance,
        minimum_goal_variant_distance=args.minimum_goal_variant_distance,
        maximum_goal_variant_distance=args.maximum_goal_variant_distance,
        color_palette_mode=args.color_palette_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_phase3a_positive_control.jsonl"
    validation_path = args.output_dir / "validation_phase3a_positive_control.jsonl"
    manifest_path = args.output_dir / "phase3a_positive_control_manifest.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)
    action_prior = phase3a_action_only_prior(train_rows, validation_rows)
    manifest = {
        "schema": "jepa_phase3a_positive_control_manifest_v0",
        "train_data": str(train_path.resolve()),
        "validation_data": str(validation_path.resolve()),
        "train_audit": train_audit,
        "validation_audit": validation_audit,
        "action_only_prior": action_prior,
        "args": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
