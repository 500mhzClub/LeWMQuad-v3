#!/usr/bin/env python3
"""Collect a route-teacher dataset on 01732 with physical Go2 locomotion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / ".generated/go2_memory_closed_loop/"
    "clean_go2_candidate_try010_strict_blockedarc_targetbearing018_arcmax035_fwdslow_y245_policy50_result.json"
)
DEFAULT_SCENE_ID = "medium_enclosed_maze_01732aabc542"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the known successful physical standoff-route teacher on the "
            "01732 maze and save learned-local policy examples with runtime-safe "
            "features. The final policy run must disable the standoff route."
        )
    )
    parser.add_argument("--template-result", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--target-colors", default="red,yellow,blue,green")
    parser.add_argument("--max-ticks", type=int, default=420)
    parser.add_argument("--dataset-states", default="EXPLORE,SEEK,SERVO")
    parser.add_argument("--online-map-size", type=int, default=21)
    parser.add_argument("--online-map-cell-m", type=float, default=0.45)
    parser.add_argument("--learned-local-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-local-post-claim-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-local-post-claim-policy-min-claims", type=int, default=None)
    parser.add_argument("--learned-local-policy-post-claim-states", default=None)
    parser.add_argument("--child-log", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = _load_template_argv(args.template_result)
    argv[0] = str(REPO_ROOT / "scripts/benchmark_go2_memory_closed_loop.py")
    _set_value(argv, "--mode", "physical")
    _set_value(argv, "--scene-id", args.scene_id)
    _set_value(argv, "--target-colors", args.target_colors)
    _set_value(argv, "--max-ticks", str(args.max_ticks))
    _set_value(argv, "--output", str(args.output))
    _remove_option(argv, "--demo-video")
    _ensure_flag(argv, "--explore-standoff-route")
    _set_value(argv, "--learned-local-policy-states", args.dataset_states)
    _set_value(argv, "--learned-local-dataset-states", args.dataset_states)
    _set_value(argv, "--learned-local-dataset-output", str(args.dataset_output))
    _ensure_flag(argv, "--learned-local-clock-features")
    _ensure_flag(argv, "--learned-local-state-features")
    _ensure_flag(argv, "--learned-local-visual-readout-features")
    _ensure_flag(argv, "--learned-local-online-map-features")
    _set_value(argv, "--learned-local-online-map-size", str(args.online_map_size))
    _set_value(argv, "--learned-local-online-map-cell-m", str(args.online_map_cell_m))
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
    _ensure_flag(argv, "--log-color-readouts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
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
