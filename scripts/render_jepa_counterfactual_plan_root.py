#!/usr/bin/env python3
"""Render every JEPA counterfactual replay plan under a multi-scene plan root."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _output_dir(plan_root: Path, output_root: Path, plan_path: Path) -> Path:
    return output_root / plan_path.parent.relative_to(plan_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, required=True)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--camera-mode", default="replay")
    parser.add_argument("--replay-env-mode", default="single")
    parser.add_argument("--rgb-format", default="png")
    parser.add_argument("--store-resolution", default="training")
    parser.add_argument("--depth-validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    plan_root = args.plan_root.resolve()
    output_root = args.output_root.resolve()
    plans = sorted(plan_root.rglob("render_replay_plan.json"))
    if not plans:
        raise SystemExit(f"no render plans found under {plan_root}")

    scene_reports = []
    for index, plan in enumerate(plans, start=1):
        output = _output_dir(plan_root, output_root, plan)
        summary_path = output / "summary.json"
        if summary_path.is_file() and not args.overwrite:
            summary = json.loads(summary_path.read_text())
            return_code = 0 if int(summary.get("invalid_frame_count", 0)) == 0 else 2
            status = "reused"
        else:
            command = [
                str(repo_root / "scripts" / "render_replay_genesis.sh"),
                str(plan),
                "--scene-corpus",
                str(args.scene_corpus.resolve()),
                "--backend",
                args.backend,
                "--camera-mode",
                args.camera_mode,
                "--replay-env-mode",
                args.replay_env_mode,
                "--rgb-format",
                args.rgb_format,
                "--store-resolution",
                args.store_resolution,
                "--out",
                str(output),
            ]
            if args.depth_validate_only:
                command.append("--depth-validate-only")
            print(f"rendering scene {index}/{len(plans)}: {plan.parent.name}", flush=True)
            completed = subprocess.run(command, cwd=repo_root, check=False)
            return_code = completed.returncode
            if return_code not in (0, 2):
                raise SystemExit(
                    f"render failed with code {return_code}: {' '.join(command)}"
                )
            if not summary_path.is_file():
                raise SystemExit(f"render did not write summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            status = "rendered"
        scene_reports.append(
            {
                "plan": str(plan),
                "output": str(output),
                "status": status,
                "render_return_code": return_code,
                "frame_count": int(summary["frame_count"]),
                "invalid_frame_count": int(summary["invalid_frame_count"]),
            }
        )

    aggregate = {
        "schema": "jepa_counterfactual_render_root_summary_v0",
        "plan_root": str(plan_root),
        "output_root": str(output_root),
        "scene_count": len(scene_reports),
        "frame_count": sum(item["frame_count"] for item in scene_reports),
        "invalid_frame_count": sum(
            item["invalid_frame_count"] for item in scene_reports
        ),
        "scenes": scene_reports,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "root_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
