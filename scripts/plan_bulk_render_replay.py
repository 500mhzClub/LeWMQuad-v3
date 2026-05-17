#!/usr/bin/env python3
"""Plan render replay for one or more converted raw_rollout directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))

from lewm_genesis.render_replay import build_render_replay_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_rollouts",
        nargs="*",
        type=Path,
        help="Converted raw_rollout directories, or roots containing raw_rollout children.",
    )
    parser.add_argument(
        "--raw-root",
        action="append",
        type=Path,
        default=[],
        help="Discover converted raw_rollout directories under this root.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
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
        help="Plan even when a raw_rollout audit is failing.",
    )
    args = parser.parse_args()

    raw_dirs = _discover_raw_rollouts([*args.raw_rollouts, *args.raw_root])
    if not raw_dirs:
        raise SystemExit("no converted raw_rollout directories found")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures = 0
    for index, raw_dir in enumerate(raw_dirs):
        plan_out = out_root / f"{index:06d}_{raw_dir.name}"
        try:
            plan = build_render_replay_plan(
                raw_dir,
                plan_out,
                backend=args.backend,
                camera_hz=args.camera_hz,
                platform_manifest=args.platform_manifest,
                max_frames=args.max_frames,
                require_quality_pass=not args.no_strict,
            )
        except Exception as exc:
            failures += 1
            results.append(
                {
                    "raw_rollout_dir": str(raw_dir),
                    "output_dir": str(plan_out),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        results.append(
            {
                "raw_rollout_dir": str(raw_dir),
                "output_dir": str(plan_out),
                "status": "planned",
                "plan": str(Path(plan["output_dir"]) / "render_replay_plan.json"),
                "frames": plan["frames_jsonl"],
                "frame_count": plan["frame_count"],
                "raw_quality_profile": plan["raw_quality_profile"],
            }
        )

    summary = {
        "schema": "lewm_bulk_render_replay_plan_v0",
        "backend": args.backend,
        "camera_hz": args.camera_hz,
        "raw_rollout_count": len(raw_dirs),
        "planned_count": len(raw_dirs) - failures,
        "failed_count": failures,
        "output_root": str(out_root),
        "jobs": results,
    }
    summary_path = out_root / "bulk_render_replay_plan.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "planned_count": summary["planned_count"],
                "failed_count": failures,
                "raw_rollout_count": len(raw_dirs),
            },
            sort_keys=True,
        )
    )
    return 2 if failures else 0


def _discover_raw_rollouts(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    raw_dirs: list[Path] = []
    for path in paths:
        candidate = path.resolve()
        for raw_dir in _iter_raw_rollout_dirs(candidate):
            if raw_dir not in seen:
                seen.add(raw_dir)
                raw_dirs.append(raw_dir)
    return sorted(raw_dirs)


def _iter_raw_rollout_dirs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"raw_rollout path does not exist: {path}")
    if path.is_dir() and _is_converted_raw_rollout(path):
        yield path
        return
    if not path.is_dir():
        return
    for summary_path in sorted(path.rglob("summary.json")):
        raw_dir = summary_path.parent
        if _is_converted_raw_rollout(raw_dir):
            yield raw_dir


def _is_converted_raw_rollout(path: Path) -> bool:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(summary.get("messages_jsonl")) and "contract_audit" in summary


if __name__ == "__main__":
    raise SystemExit(main())
