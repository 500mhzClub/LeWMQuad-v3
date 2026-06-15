#!/usr/bin/env python3
"""Render JEPA counterfactual replay plans with scene-level parallelism."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ACCEPTED_RENDER_RETURN_CODES = (0, 2)


def _output_dir(plan_root: Path, output_root: Path, plan_path: Path) -> Path:
    return output_root / plan_path.parent.relative_to(plan_root)


def _scene_log_path(plan_root: Path, log_root: Path, plan_path: Path) -> Path:
    return (log_root / plan_path.parent.relative_to(plan_root)).with_suffix(".log")


def _render_return_code_from_summary(summary: dict[str, Any]) -> int:
    invalid_frame_count = int(summary.get("invalid_frame_count", 0))
    return 0 if invalid_frame_count == 0 else 2


def _load_summary(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if "frame_count" not in summary or "invalid_frame_count" not in summary:
        raise ValueError(f"render summary is missing required counts: {summary_path}")
    return summary


def _build_command(
    *,
    repo_root: Path,
    plan: Path,
    output: Path,
    scene_corpus: Path,
    backend: str,
    camera_mode: str,
    replay_env_mode: str,
    rgb_format: str,
    store_resolution: str,
    depth_validate_only: bool,
) -> list[str]:
    command = [
        str(repo_root / "scripts" / "render_replay_genesis.sh"),
        str(plan),
        "--scene-corpus",
        str(scene_corpus),
        "--backend",
        backend,
        "--camera-mode",
        camera_mode,
        "--replay-env-mode",
        replay_env_mode,
        "--rgb-format",
        rgb_format,
        "--store-resolution",
        store_resolution,
        "--out",
        str(output),
    ]
    if depth_validate_only:
        command.append("--depth-validate-only")
    return command


def _run_command(command: list[str], *, repo_root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def _render_or_reuse_scene(
    *,
    index: int,
    total: int,
    plan_root: Path,
    output_root: Path,
    log_root: Path,
    repo_root: Path,
    plan: Path,
    scene_corpus: Path,
    backend: str,
    camera_mode: str,
    replay_env_mode: str,
    rgb_format: str,
    store_resolution: str,
    depth_validate_only: bool,
    overwrite: bool,
) -> dict[str, Any]:
    output = _output_dir(plan_root, output_root, plan)
    summary_path = output / "summary.json"
    log_path = _scene_log_path(plan_root, log_root, plan)
    if summary_path.is_file() and not overwrite:
        summary = _load_summary(summary_path)
        return_code = _render_return_code_from_summary(summary)
        status = "reused"
        print(f"reused scene {index}/{total}: {plan.parent.name}", flush=True)
    else:
        command = _build_command(
            repo_root=repo_root,
            plan=plan,
            output=output,
            scene_corpus=scene_corpus,
            backend=backend,
            camera_mode=camera_mode,
            replay_env_mode=replay_env_mode,
            rgb_format=rgb_format,
            store_resolution=store_resolution,
            depth_validate_only=depth_validate_only,
        )
        print(f"rendering scene {index}/{total}: {plan.parent.name}", flush=True)
        return_code = _run_command(command, repo_root=repo_root, log_path=log_path)
        if return_code not in ACCEPTED_RENDER_RETURN_CODES:
            return {
                "index": index,
                "plan": str(plan),
                "output": str(output),
                "status": "failed",
                "render_return_code": return_code,
                "frame_count": 0,
                "invalid_frame_count": 0,
                "log": str(log_path),
                "error": f"render failed with code {return_code}",
            }
        if not summary_path.is_file():
            return {
                "index": index,
                "plan": str(plan),
                "output": str(output),
                "status": "failed",
                "render_return_code": return_code,
                "frame_count": 0,
                "invalid_frame_count": 0,
                "log": str(log_path),
                "error": f"render did not write summary: {summary_path}",
            }
        summary = _load_summary(summary_path)
        status = "rendered"
        print(f"finished scene {index}/{total}: {plan.parent.name}", flush=True)
    return {
        "index": index,
        "plan": str(plan),
        "output": str(output),
        "status": status,
        "render_return_code": return_code,
        "frame_count": int(summary["frame_count"]),
        "invalid_frame_count": int(summary["invalid_frame_count"]),
        "log": str(log_path),
    }


def _aggregate_reports(
    *,
    plan_root: Path,
    output_root: Path,
    reports: list[dict[str, Any]],
    expected_scene_count: int,
    jobs: int,
) -> dict[str, Any]:
    ordered = sorted(reports, key=lambda item: int(item["index"]))
    failures = [item for item in ordered if item.get("status") == "failed"]
    successful = [item for item in ordered if item.get("status") != "failed"]
    return {
        "schema": "jepa_counterfactual_render_root_summary_v0",
        "renderer": "parallel_scene_subprocess",
        "jobs": jobs,
        "plan_root": str(plan_root),
        "output_root": str(output_root),
        "expected_scene_count": expected_scene_count,
        "scene_count": len(successful),
        "frame_count": sum(int(item["frame_count"]) for item in successful),
        "invalid_frame_count": sum(
            int(item["invalid_frame_count"]) for item in successful
        ),
        "failure_count": len(failures),
        "scenes": successful,
        "failures": failures,
    }


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
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--log-root", type=Path)
    args = parser.parse_args()

    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    plan_root = args.plan_root.resolve()
    output_root = args.output_root.resolve()
    scene_corpus = args.scene_corpus.resolve()
    log_root = (
        args.log_root.resolve()
        if args.log_root is not None
        else output_root / "_parallel_logs"
    )
    plans = sorted(plan_root.rglob("render_replay_plan.json"))
    if not plans:
        raise SystemExit(f"no render plans found under {plan_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                _render_or_reuse_scene,
                index=index,
                total=len(plans),
                plan_root=plan_root,
                output_root=output_root,
                log_root=log_root,
                repo_root=repo_root,
                plan=plan,
                scene_corpus=scene_corpus,
                backend=args.backend,
                camera_mode=args.camera_mode,
                replay_env_mode=args.replay_env_mode,
                rgb_format=args.rgb_format,
                store_resolution=args.store_resolution,
                depth_validate_only=args.depth_validate_only,
                overwrite=args.overwrite,
            )
            for index, plan in enumerate(plans, start=1)
        ]
        for future in as_completed(futures):
            reports.append(future.result())

    aggregate = _aggregate_reports(
        plan_root=plan_root,
        output_root=output_root,
        reports=reports,
        expected_scene_count=len(plans),
        jobs=args.jobs,
    )
    (output_root / "root_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))
    return 1 if aggregate["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
