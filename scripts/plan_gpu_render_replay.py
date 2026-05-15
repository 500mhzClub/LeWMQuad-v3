#!/usr/bin/env python3
"""Plan GPU render replay from a converted raw_rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))

from lewm_genesis.render_replay import build_render_replay_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_rollout", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--backend", default="genesis")
    parser.add_argument("--camera-hz", type=float, default=10.0)
    parser.add_argument(
        "--platform-manifest",
        type=Path,
        default=REPO_ROOT / "config" / "go2_platform_manifest.yaml",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Plan even when raw_rollout audits are failing.",
    )
    args = parser.parse_args()

    raw_rollout = args.raw_rollout.resolve()
    out_dir = (
        args.out
        if args.out is not None
        else raw_rollout.with_name(f"{raw_rollout.name}_rendered_vision_plan")
    )
    plan = build_render_replay_plan(
        raw_rollout,
        out_dir,
        backend=args.backend,
        camera_hz=args.camera_hz,
        platform_manifest=args.platform_manifest,
        max_frames=args.max_frames,
        require_quality_pass=not args.no_strict,
    )
    print(json.dumps(
        {
            "plan": str(Path(plan["output_dir"]) / "render_replay_plan.json"),
            "frames": plan["frames_jsonl"],
            "frame_count": plan["frame_count"],
            "backend": plan["backend"],
            "raw_quality_profile": plan["raw_quality_profile"],
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
