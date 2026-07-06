#!/usr/bin/env python3
"""Run the b7 learned-runtime stack on the 01732 maze with physical locomotion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / ".generated/go2_memory_closed_loop/b7_strict_learned_maze_20260704/"
    "medium_b7c_v297_packaged_all_artifacts_strict_seed1_result.json"
)
DEFAULT_SCENE_ID = "medium_enclosed_maze_01732aabc542"
DEFAULT_CLAIM_SUCCESS_MODEL = (
    REPO_ROOT
    / ".generated/go2_memory_closed_loop/b7_strict_learned_maze_20260704/"
    "go2_claim_success_head_v294_b7_strict.pt"
)
CLAIM_COLORS = ("red", "yellow", "blue", "green")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a physical 01732 learned-policy attempt from the packaged b7 "
            "learned-runtime command."
        )
    )
    parser.add_argument("--template-result", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--mode", choices=("physical", "kinematic"), default="physical")
    parser.add_argument("--no-generalized-runtime-contract", action="store_true")
    parser.add_argument("--target-colors", default="red,yellow,blue,green")
    parser.add_argument(
        "--multi-target-switch-policy",
        choices=("fixed", "seen_when_active_unseen", "visible_priority", "memory_priority"),
        default=None,
    )
    parser.add_argument("--multi-target-switch-conf", type=float, default=None)
    parser.add_argument("--learned-target-scheduler-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-target-scheduler-log-scores", action="store_true")
    parser.add_argument("--log-color-readouts", action="store_true")
    parser.add_argument("--learned-local-dataset-output", type=Path, default=None)
    parser.add_argument("--learned-local-dataset-states", default=None)
    parser.add_argument("--learned-local-oracle-standoff-labels", action="store_true")
    parser.add_argument("--learned-local-oracle-standoff-label-states", default=None)
    parser.add_argument("--max-ticks", type=int, default=700)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learned-local-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-local-post-claim-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-local-post-claim-policy-min-claims", type=int, default=None)
    parser.add_argument("--learned-local-policy-post-claim-states", default=None)
    parser.add_argument("--learned-local-policy-states", default=None)
    parser.add_argument("--learned-local-target-policy-checkpoints", default=None)
    parser.add_argument("--learned-local-target-policy-state-checkpoints", default=None)
    parser.add_argument("--learned-local-target-policy-priority-over-post-claim", action="store_true")
    parser.add_argument(
        "--learned-local-post-claim-priority-over-target-policy",
        action="store_true",
        help=(
            "Let the learned post-claim policy slot override target-color policies "
            "once its min-claim gate is active."
        ),
    )
    parser.add_argument(
        "--single-learned-local-policy",
        action="store_true",
        help="Remove post-claim and target-specific learned-local policy routers.",
    )
    parser.add_argument("--claim-success-threshold", type=float, default=None)
    parser.add_argument("--claim-success-model-checkpoint", type=Path, default=DEFAULT_CLAIM_SUCCESS_MODEL)
    parser.add_argument("--claim-trigger-min-seen-ticks", type=int, default=8)
    parser.add_argument("--learned-claim-only", action="store_true")
    parser.add_argument("--allow-visual-opportunistic-claims", action="store_true")
    parser.add_argument("--online-map-novelty-weight", type=float, default=None)
    parser.add_argument("--online-map-frontier-route-weight", type=float, default=None)
    parser.add_argument("--online-map-claim-repulsion-weight", type=float, default=None)
    parser.add_argument(
        "--disable-local-pressure",
        action="store_true",
        help="Disable rule-based learned-local frontier and translation pressure helpers.",
    )
    parser.add_argument(
        "--learned-local-policy-outcome-rerank",
        action="store_true",
        help="Enable the learned primitive-outcome reranker for learned-local policy actions.",
    )
    parser.add_argument(
        "--learned-local-post-claim-policy-outcome-rerank",
        choices=("inherit", "on", "off"),
        default=None,
    )
    parser.add_argument("--learned-local-policy-rerank-policy-weight", type=float, default=None)
    parser.add_argument("--learned-local-post-claim-policy-rerank-policy-weight", type=float, default=None)
    parser.add_argument("--learned-local-policy-rerank-top-k", type=int, default=None)
    parser.add_argument("--online-map-size", type=int, default=None)
    parser.add_argument("--online-map-cell-m", type=float, default=None)
    parser.add_argument("--wall-guard-states", default=None)
    parser.add_argument("--wall-guard-post-claim-states", default=None)
    parser.add_argument("--wall-guard-post-claim-min-claims", type=int, default=None)
    parser.add_argument("--primitive-outcome-preserve-arc-requests", action="store_true")
    parser.add_argument(
        "--keep-visited-after-claim",
        action="store_true",
        help=(
            "Preserve the learned policy's online visited map across claims instead "
            "of clearing it at every beacon."
        ),
    )
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--demo-fps", type=int, default=50)
    parser.add_argument("--demo-capture-rate", default="policy")
    parser.add_argument("--child-log", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = _load_template_argv(args.template_result)
    argv[0] = str(REPO_ROOT / "scripts/benchmark_go2_memory_closed_loop.py")

    if args.no_generalized_runtime_contract:
        _remove_flag(argv, "--generalized-runtime-contract")
    else:
        _ensure_flag(argv, "--generalized-runtime-contract")
    _set_value(argv, "--mode", args.mode)
    _set_value(argv, "--scene-id", args.scene_id)
    _set_value(argv, "--split", "train")
    _set_value(argv, "--seed", str(args.seed))
    _set_value(argv, "--max-ticks", str(args.max_ticks))
    _set_value(argv, "--target-colors", args.target_colors)
    _set_value(argv, "--output", str(args.output))
    if args.multi_target_switch_policy is not None:
        _set_value(argv, "--multi-target-switch-policy", str(args.multi_target_switch_policy))
    if args.multi_target_switch_conf is not None:
        _set_value(argv, "--multi-target-switch-conf", str(args.multi_target_switch_conf))
    if args.learned_target_scheduler_checkpoint is not None:
        _set_value(
            argv,
            "--learned-target-scheduler-checkpoint",
            str(args.learned_target_scheduler_checkpoint),
        )
    if args.learned_target_scheduler_log_scores:
        _ensure_flag(argv, "--learned-target-scheduler-log-scores")
    if args.log_color_readouts:
        _ensure_flag(argv, "--log-color-readouts")
    if args.learned_local_dataset_output is not None:
        _set_value(
            argv,
            "--learned-local-dataset-output",
            str(args.learned_local_dataset_output),
        )
        args.learned_local_dataset_output.parent.mkdir(parents=True, exist_ok=True)
    if args.learned_local_dataset_states is not None:
        _set_value(
            argv,
            "--learned-local-dataset-states",
            str(args.learned_local_dataset_states),
        )
    if args.learned_local_oracle_standoff_labels:
        _ensure_flag(argv, "--learned-local-oracle-standoff-labels")
    if args.learned_local_oracle_standoff_label_states is not None:
        _set_value(
            argv,
            "--learned-local-oracle-standoff-label-states",
            str(args.learned_local_oracle_standoff_label_states),
        )

    _remove_flag(argv, "--explore-standoff-route")
    _set_value(argv, "--explore-route-waypoints", "")
    _remove_option(argv, "--learned-topology-route-table")
    _remove_option(argv, "--learned-topology-route-until-area-logit")
    _set_value(argv, "--explore-goal-policy", "learned_policy")
    _set_value(argv, "--wall-decision-source", "learned_action")
    if args.learned_local_policy_checkpoint is not None:
        _set_value(
            argv,
            "--learned-local-policy-checkpoint",
            str(args.learned_local_policy_checkpoint),
        )
    if args.learned_local_post_claim_policy_checkpoint is not None:
        _set_value(
            argv,
            "--learned-local-post-claim-policy-checkpoint",
            str(args.learned_local_post_claim_policy_checkpoint),
        )
    if args.learned_local_post_claim_policy_min_claims is not None:
        _set_value(
            argv,
            "--learned-local-post-claim-policy-min-claims",
            str(args.learned_local_post_claim_policy_min_claims),
        )
    if args.learned_local_policy_post_claim_states is not None:
        _set_value(
            argv,
            "--learned-local-policy-post-claim-states",
            str(args.learned_local_policy_post_claim_states),
        )
    if args.learned_local_policy_states is not None:
        _set_value(argv, "--learned-local-policy-states", str(args.learned_local_policy_states))
    if args.learned_local_target_policy_checkpoints is not None:
        _set_value(
            argv,
            "--learned-local-target-policy-checkpoints",
            str(args.learned_local_target_policy_checkpoints),
        )
    if args.learned_local_target_policy_state_checkpoints is not None:
        _set_value(
            argv,
            "--learned-local-target-policy-state-checkpoints",
            str(args.learned_local_target_policy_state_checkpoints),
        )
    if args.learned_local_target_policy_priority_over_post_claim:
        _ensure_flag(argv, "--learned-local-target-policy-priority-over-post-claim")
    if args.learned_local_post_claim_priority_over_target_policy:
        _remove_flag(argv, "--learned-local-target-policy-priority-over-post-claim")
    if args.single_learned_local_policy:
        _remove_option(argv, "--learned-local-post-claim-policy-checkpoint")
        _remove_option(argv, "--learned-local-target-policy-checkpoints")
        _remove_option(argv, "--learned-local-target-policy-state-checkpoints")
        _remove_flag(argv, "--learned-local-target-policy-priority-over-post-claim")
        _set_value(argv, "--learned-local-policy-post-claim-states", "")
        _remove_option(argv, "--learned-local-post-claim-policy-outcome-rerank")
        _remove_option(argv, "--learned-local-target-policy-outcome-rerank")
        _remove_option(argv, "--learned-local-post-claim-policy-rerank-policy-weight")
        _remove_option(argv, "--learned-local-target-policy-rerank-policy-weight")
    if args.online_map_novelty_weight is not None:
        _set_value(
            argv,
            "--learned-local-policy-online-map-novelty-weight",
            str(args.online_map_novelty_weight),
        )
    if args.online_map_frontier_route_weight is not None:
        _set_value(
            argv,
            "--learned-local-policy-online-map-frontier-route-weight",
            str(args.online_map_frontier_route_weight),
        )
    if args.online_map_claim_repulsion_weight is not None:
        _set_value(
            argv,
            "--learned-local-policy-online-map-claim-repulsion-weight",
            str(args.online_map_claim_repulsion_weight),
        )
    if args.disable_local_pressure:
        _set_value(argv, "--learned-local-policy-frontier-pressure-after", "0")
        _set_value(argv, "--learned-local-policy-translation-pressure-after", "0")
        _set_value(argv, "--learned-local-policy-online-map-frontier-route-weight", "0.0")
        _set_value(argv, "--learned-local-policy-online-map-novelty-weight", "0.0")
    if args.learned_local_policy_outcome_rerank:
        _ensure_flag(argv, "--learned-local-policy-outcome-rerank")
    if args.learned_local_post_claim_policy_outcome_rerank is not None:
        _set_value(
            argv,
            "--learned-local-post-claim-policy-outcome-rerank",
            str(args.learned_local_post_claim_policy_outcome_rerank),
        )
    if args.learned_local_policy_rerank_policy_weight is not None:
        _set_value(
            argv,
            "--learned-local-policy-rerank-policy-weight",
            str(args.learned_local_policy_rerank_policy_weight),
        )
    if args.learned_local_post_claim_policy_rerank_policy_weight is not None:
        _set_value(
            argv,
            "--learned-local-post-claim-policy-rerank-policy-weight",
            str(args.learned_local_post_claim_policy_rerank_policy_weight),
        )
    if args.learned_local_policy_rerank_top_k is not None:
        _set_value(
            argv,
            "--learned-local-policy-rerank-top-k",
            str(args.learned_local_policy_rerank_top_k),
        )
    if args.online_map_size is not None:
        _set_value(argv, "--learned-local-online-map-size", str(args.online_map_size))
    if args.online_map_cell_m is not None:
        _set_value(argv, "--learned-local-online-map-cell-m", str(args.online_map_cell_m))
    if args.wall_guard_states is not None:
        _set_value(argv, "--wall-guard-states", str(args.wall_guard_states))
    if args.wall_guard_post_claim_states is not None:
        _set_value(argv, "--wall-guard-post-claim-states", str(args.wall_guard_post_claim_states))
    if args.wall_guard_post_claim_min_claims is not None:
        _set_value(
            argv,
            "--wall-guard-post-claim-min-claims",
            str(args.wall_guard_post_claim_min_claims),
        )
    if args.primitive_outcome_preserve_arc_requests:
        _ensure_flag(argv, "--primitive-outcome-preserve-arc-requests")
    if args.keep_visited_after_claim:
        _remove_flag(argv, "--explore-clear-visited-on-claim")

    _ensure_flag(argv, "--claim-success-model-positive-trigger")
    if args.claim_success_model_checkpoint is not None:
        _set_value(
            argv,
            "--claim-success-model-checkpoint",
            str(args.claim_success_model_checkpoint),
        )
    _set_value(
        argv,
        "--claim-success-model-trigger-min-seen-ticks",
        str(args.claim_trigger_min_seen_ticks),
    )
    if args.claim_success_threshold is not None:
        _set_value(argv, "--claim-success-model-threshold", str(args.claim_success_threshold))

    if args.learned_claim_only:
        _set_value(argv, "--claim-area-logit", "999.0")
        _set_value(argv, "--claim-near-area-logit", "999.0")
        _set_value(
            argv,
            "--claim-near-area-logit-by-color",
            ",".join(f"{color}:999.0" for color in CLAIM_COLORS),
        )
        _remove_option(argv, "--claim-contact-area-logit")
        _remove_option(argv, "--claim-stalled-area-logit")
        _remove_option(argv, "--claim-success-proxy-area-logit")
        _set_value(argv, "--claim-success-proxy-area-logit-by-color", "")
        _remove_option(argv, "--claim-success-proxy-bearing")
        _set_value(argv, "--claim-success-proxy-bearing-by-color", "")
        if not args.allow_visual_opportunistic_claims:
            _remove_flag(argv, "--multi-target-opportunistic-claims")

    if args.demo_video is None:
        _remove_option(argv, "--demo-video")
    else:
        _set_value(argv, "--demo-video", str(args.demo_video))
        _set_value(argv, "--demo-fps", str(args.demo_fps))
        _set_value(argv, "--demo-capture-rate", str(args.demo_capture_rate))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.demo_video is not None:
        args.demo_video.parent.mkdir(parents=True, exist_ok=True)
    if args.child_log is not None:
        args.child_log.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(" ".join(_shell_quote(item) for item in argv))
        return 0

    if args.child_log is None:
        try:
            return subprocess.run(
                [sys.executable, *argv],
                cwd=REPO_ROOT,
                timeout=args.timeout_s,
            ).returncode
        except subprocess.TimeoutExpired:
            return 124

    with args.child_log.open("w", encoding="utf-8") as log_file:
        try:
            return subprocess.run(
                [sys.executable, *argv],
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_s,
            ).returncode
        except subprocess.TimeoutExpired:
            log_file.write("\nTIMEOUT\n")
            return 124


def _load_template_argv(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    argv = payload.get("provenance", {}).get("argv")
    if not isinstance(argv, list) or not argv:
        raise SystemExit(f"{path} does not contain provenance.argv")
    return [str(item) for item in argv]


def _set_value(argv: list[str], flag: str, value: str) -> None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
            argv.insert(idx + 1, value)
        else:
            argv[idx + 1] = value
    else:
        argv.extend([flag, value])


def _ensure_flag(argv: list[str], flag: str) -> None:
    if flag not in argv:
        argv.append(flag)


def _remove_flag(argv: list[str], flag: str) -> None:
    while flag in argv:
        argv.pop(argv.index(flag))


def _remove_option(argv: list[str], flag: str) -> None:
    while flag in argv:
        idx = argv.index(flag)
        del argv[idx : min(idx + 2, len(argv))]


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "/._-:=,+" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
