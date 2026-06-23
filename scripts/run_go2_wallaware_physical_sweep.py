#!/usr/bin/env python3
"""Run a small physical closed-loop sweep for the Go2 wall-aware scaffold."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = (
    "000c67a65968",
    "01732aabc542",
    "04f670cb21f8",
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    scene_short: str
    demo_mode: str
    wallaware: bool
    claim_area_logit: float
    claim_bearing: float


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _result(path: Path) -> dict[str, Any]:
    data = _load(path)
    result = data.get("result", {})
    return result if isinstance(result, dict) else {}


def _metrics(path: Path) -> dict[str, Any]:
    metrics = _result(path).get("wall_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fp:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(proc.returncode)


def _controller_path(controller_dir: Path, scene_short: str, seed: str) -> Path:
    return controller_dir / f"exact_{scene_short}_s{seed}.pt"


def _build_command(
    *,
    python_bin: Path,
    spec: RunSpec,
    scene_corpus: Path,
    split: str,
    controller_dir: Path,
    controller_seed: str,
    frozen_jepa: Path,
    output: Path,
    backend: str,
    max_ticks: int,
    success_dist_m: float,
    extra_wall_args: list[str],
) -> list[str]:
    scene_id = f"medium_enclosed_maze_{spec.scene_short}"
    cmd = [
        str(python_bin),
        "scripts/benchmark_go2_memory_closed_loop.py",
        "--scene-corpus", str(scene_corpus),
        "--split", split,
        "--scene-id", scene_id,
        "--backend", backend,
        "--apply-textures",
        "--mode", "physical",
        "--policy-device", "cpu",
        "--controller", str(_controller_path(controller_dir, spec.scene_short, controller_seed)),
        "--frozen-jepa-checkpoint", str(frozen_jepa),
        "--target-color", "green",
        "--policy", "memory",
        "--demo-mode", spec.demo_mode,
        "--max-ticks", str(max_ticks),
        "--success-dist-m", str(success_dist_m),
        "--claim-area-logit", str(spec.claim_area_logit),
        "--claim-bearing", str(spec.claim_bearing),
        "--output", str(output),
    ]
    if spec.wallaware:
        cmd.append("--wall-aware-planner")
        cmd.extend(extra_wall_args)
    return cmd


def _summarize_scene(scene_short: str, paths: dict[str, Path]) -> dict[str, Any]:
    baseline = _result(paths["baseline_explore"])
    wall = _result(paths["wallaware_explore"])
    recall = _result(paths["wallaware_recall"])
    base_metrics = _metrics(paths["baseline_explore"])
    wall_metrics = _metrics(paths["wallaware_explore"])
    base_blocked = int(_num(base_metrics.get("blocked_forward_executions")))
    wall_blocked = int(_num(wall_metrics.get("blocked_forward_executions")))
    base_forward = int(_num(base_metrics.get("forward_executions")))
    wall_forward = int(_num(wall_metrics.get("forward_executions")))
    base_rate = base_blocked / base_forward if base_forward else 0.0
    wall_rate = wall_blocked / wall_forward if wall_forward else 0.0
    rate_reduction = 1.0 if base_rate <= 0.0 and wall_rate <= 0.0 else (
        (base_rate - wall_rate) / base_rate if base_rate > 0.0 else 0.0
    )
    return {
        "scene_short": scene_short,
        "baseline_explore_success": bool(baseline.get("success")),
        "wallaware_explore_success": bool(wall.get("success")),
        "wallaware_recall_success": bool(recall.get("success")),
        "baseline_first_seen_tick": baseline.get("first_seen_tick"),
        "wallaware_first_seen_tick": wall.get("first_seen_tick"),
        "baseline_ticks_used": baseline.get("ticks_used"),
        "wallaware_ticks_used": wall.get("ticks_used"),
        "wallaware_recall_ticks_used": recall.get("ticks_used"),
        "baseline_final_dist_to_target_m": baseline.get("final_dist_to_target_m"),
        "wallaware_final_dist_to_target_m": wall.get("final_dist_to_target_m"),
        "wallaware_recall_final_dist_to_target_m": recall.get("final_dist_to_target_m"),
        "baseline_blocked_forward_executions": base_blocked,
        "wallaware_blocked_forward_executions": wall_blocked,
        "baseline_blocked_forward_rate": round(base_rate, 6),
        "wallaware_blocked_forward_rate": round(wall_rate, 6),
        "blocked_forward_rate_reduction": round(rate_reduction, 6),
        "baseline_contact_like_stalls": int(_num(base_metrics.get("contact_like_stalls"))),
        "wallaware_contact_like_stalls": int(_num(wall_metrics.get("contact_like_stalls"))),
        "wall_vetoes": int(_num(wall_metrics.get("wall_vetoes"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path,
                        default=REPO_ROOT / ".generated/venvs/genesis_render_vulkan/bin/python")
    parser.add_argument("--scene-corpus", type=Path,
                        default=REPO_ROOT / ".generated/scene_corpus/minimum_20260520T080420Z")
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene-short", action="append", default=None,
                        help="Scene suffix such as 01732aabc542. May be repeated.")
    parser.add_argument("--controller-dir", type=Path,
                        default=REPO_ROOT / ".generated/go2_hidden_target_memory/observed_memory_gate_20260622/exact_valuenorm_cv")
    parser.add_argument("--controller-seed", default="20260820")
    parser.add_argument("--frozen-jepa-checkpoint", type=Path,
                        default=REPO_ROOT / ".generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / ".generated/go2_wallaware_physical_sweep/minimum_3scene")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--max-ticks", type=int, default=90)
    parser.add_argument("--success-dist-m", type=float, default=1.2)
    parser.add_argument("--baseline-claim-bearing", type=float, default=0.5)
    parser.add_argument("--wallaware-claim-bearing", type=float, default=0.3)
    parser.add_argument("--wallaware-recall-claim-bearing", type=float, default=None)
    parser.add_argument("--claim-area-logit", type=float, default=1.5)
    parser.add_argument("--min-explore-success-rate", type=float, default=0.67)
    parser.add_argument("--min-recall-success-rate", type=float, default=1.0)
    parser.add_argument("--min-blocked-rate-improvement-scenes", type=int, default=2)
    parser.add_argument("--max-stall-regression-scenes", type=int, default=0)
    parser.add_argument("--wall-extra-arg", action="append", default=[],
                        help="Extra argument for wall-aware benchmark runs. Repeat for value args.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    scenes = tuple(args.scene_short) if args.scene_short else DEFAULT_SCENES
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    recall_claim_bearing = (
        args.wallaware_claim_bearing
        if args.wallaware_recall_claim_bearing is None
        else args.wallaware_recall_claim_bearing
    )
    run_specs = []
    for scene_short in scenes:
        run_specs.extend([
            RunSpec("baseline_explore", scene_short, "explore", False, args.claim_area_logit, args.baseline_claim_bearing),
            RunSpec("wallaware_explore", scene_short, "explore", True, args.claim_area_logit, args.wallaware_claim_bearing),
            RunSpec("wallaware_recall", scene_short, "recall", True, args.claim_area_logit, recall_claim_bearing),
        ])

    run_records = []
    for spec in run_specs:
        output = output_dir / f"{spec.scene_short}_{spec.label}_result.json"
        log = output_dir / f"{spec.scene_short}_{spec.label}.log"
        cmd = _build_command(
            python_bin=args.python_bin,
            spec=spec,
            scene_corpus=args.scene_corpus,
            split=args.split,
            controller_dir=args.controller_dir,
            controller_seed=args.controller_seed,
            frozen_jepa=args.frozen_jepa_checkpoint,
            output=output,
            backend=args.backend,
            max_ticks=args.max_ticks,
            success_dist_m=args.success_dist_m,
            extra_wall_args=list(args.wall_extra_arg),
        )
        skipped = bool(args.skip_existing and output.exists())
        returncode = 0
        if not skipped:
            print(f"RUN {spec.scene_short} {spec.label}", flush=True)
            returncode = _run_command(cmd, log)
        run_records.append({
            "scene_short": spec.scene_short,
            "label": spec.label,
            "output": str(output),
            "log": str(log),
            "returncode": returncode,
            "skipped": skipped,
        })
        if returncode != 0:
            report = {
                "schema": "go2_wallaware_physical_sweep_v0",
                "passed": False,
                "failure_reasons": [f"run_failed:{spec.scene_short}:{spec.label}"],
                "runs": run_records,
            }
            (output_dir / "sweep_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            return 1

    scene_summaries = []
    for scene_short in scenes:
        paths = {
            label: output_dir / f"{scene_short}_{label}_result.json"
            for label in ("baseline_explore", "wallaware_explore", "wallaware_recall")
        }
        scene_summaries.append(_summarize_scene(scene_short, paths))

    explore_success_rate = sum(1 for s in scene_summaries if s["wallaware_explore_success"]) / len(scene_summaries)
    recall_success_rate = sum(1 for s in scene_summaries if s["wallaware_recall_success"]) / len(scene_summaries)
    improved_scenes = sum(1 for s in scene_summaries if s["blocked_forward_rate_reduction"] > 0.0)
    stall_regression_scenes = sum(
        1 for s in scene_summaries
        if s["wallaware_contact_like_stalls"] > s["baseline_contact_like_stalls"]
    )

    failures = []
    if explore_success_rate < float(args.min_explore_success_rate):
        failures.append("explore_success_rate_below_threshold")
    if recall_success_rate < float(args.min_recall_success_rate):
        failures.append("recall_success_rate_below_threshold")
    if improved_scenes < int(args.min_blocked_rate_improvement_scenes):
        failures.append("insufficient_blocked_forward_rate_improvement_scenes")
    if stall_regression_scenes > int(args.max_stall_regression_scenes):
        failures.append("stall_regression_scenes_above_threshold")

    report = {
        "schema": "go2_wallaware_physical_sweep_v0",
        "passed": not failures,
        "failure_reasons": failures,
        "thresholds": {
            "min_explore_success_rate": args.min_explore_success_rate,
            "min_recall_success_rate": args.min_recall_success_rate,
            "min_blocked_rate_improvement_scenes": args.min_blocked_rate_improvement_scenes,
            "max_stall_regression_scenes": args.max_stall_regression_scenes,
        },
        "summary": {
            "scene_count": len(scene_summaries),
            "explore_success_rate": round(explore_success_rate, 6),
            "recall_success_rate": round(recall_success_rate, 6),
            "blocked_rate_improvement_scenes": improved_scenes,
            "stall_regression_scenes": stall_regression_scenes,
        },
        "scenes": scene_summaries,
        "runs": run_records,
    }
    (output_dir / "sweep_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
