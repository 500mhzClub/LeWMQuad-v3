#!/usr/bin/env python3
"""Emit head-training rows from the mass datagen corpus (no re-rendering).

Joins render frames.jsonl (frame_index/env/timestamp/pose) with the derived
labels.jsonl (clearance_m) per scene and emits, per sampled frame:

- counterfactual pose rows (schema lewm_go2_result_pose_counterfactual_row_v0)
  for train_go2_jepa_primitive_outcome_predictor.py --label-mode
  counterfactual_body_clearance;
- optional direct-label rgb rows for
  train_go2_jepa_current_body_risk_predictor.py.

Sampling is clearance-stratified so near-wall poses (the guard's failure
regime on held-out scenes) are well represented.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FAMILIES_DEFAULT = "medium_enclosed_maze"


def _scene_dirs(rollout_root: Path, family: str, split: str) -> list[Path]:
    out = []
    base = rollout_root / split / family
    if not base.is_dir():
        return out
    for chunk in sorted(base.glob("chunk_*")):
        plan_dir = chunk / "plan"
        if not plan_dir.is_dir():
            continue
        for scene_plan in sorted(plan_dir.iterdir()):
            if (scene_plan / "frames.jsonl").is_file():
                out.append(scene_plan)
    return out


def _labels_by_key(chunk_dir: Path, scene_id: str) -> dict[tuple[int, int], dict]:
    labels_path = chunk_dir / "labels" / scene_id / "labels.jsonl"
    out: dict[tuple[int, int], dict] = {}
    if not labels_path.is_file():
        return out
    with labels_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row.get("env_idx", -1)), int(row.get("timestamp_ns", -1)))
            out[key] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, required=True)
    parser.add_argument("--family", default=FAMILIES_DEFAULT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--rows-per-scene", type=int, default=400)
    parser.add_argument("--low-clearance-fraction", type=float, default=0.5)
    parser.add_argument("--low-clearance-threshold-m", type=float, default=0.35)
    parser.add_argument("--min-episode-step", type=int, default=3,
                        help="Skip the first steps of each episode (reset transients).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-rows-output", type=Path, default=None)
    parser.add_argument("--risk-margin-m", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    scene_plans = _scene_dirs(args.rollout_root, str(args.family), str(args.split))
    if args.max_scenes is not None:
        scene_plans = scene_plans[: int(args.max_scenes)]
    if not scene_plans:
        raise SystemExit("no scene plans found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    risk_file = None
    if args.risk_rows_output is not None:
        args.risk_rows_output.parent.mkdir(parents=True, exist_ok=True)
        risk_file = args.risk_rows_output.open("w")

    total_rows = 0
    total_low = 0
    scenes_used = 0
    with args.output.open("w") as out:
        for scene_plan in scene_plans:
            scene_id = scene_plan.name.split("_", 1)[1]
            chunk_dir = scene_plan.parents[1]
            manifest = (
                args.scene_corpus / str(args.split) / str(args.family) / scene_id / "manifest.json"
            )
            if not manifest.is_file():
                continue
            rgb_dir = args.render_root / scene_id / "rgb"
            if not rgb_dir.is_dir():
                continue
            labels = _labels_by_key(chunk_dir, scene_id)
            candidates: list[tuple[float, dict]] = []
            with (scene_plan / "frames.jsonl").open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    episode = rec.get("episode") or {}
                    if int(episode.get("episode_step", 0)) < int(args.min_episode_step):
                        continue
                    key = (int(rec["env_index"]), int(rec["timestamp_ns"]))
                    label = labels.get(key)
                    if label is None or label.get("clearance_m") is None:
                        continue
                    candidates.append((float(label["clearance_m"]), rec))
            if len(candidates) < 32:
                continue
            low = [c for c in candidates if c[0] < float(args.low_clearance_threshold_m)]
            high = [c for c in candidates if c[0] >= float(args.low_clearance_threshold_m)]
            n = int(args.rows_per_scene)
            n_low = min(len(low), int(n * float(args.low_clearance_fraction)))
            n_high = min(len(high), n - n_low)
            picked = rng.sample(low, n_low) + rng.sample(high, n_high)
            wrote = 0
            for clearance_m, rec in picked:
                frame = rgb_dir / f"frame_{int(rec['frame_index']):06d}_env_{int(rec['env_index']):02d}.png"
                if not frame.is_file():
                    continue
                position = rec["base_pose_world"]["position"]
                rpy = rec["base_rpy_rad"]
                row = {
                    "schema": "lewm_go2_result_pose_counterfactual_row_v0",
                    "scene_id": scene_id,
                    "scene_manifest": str(manifest.resolve()),
                    "command_dt_s": 0.1,
                    "start_frame": str(frame.resolve()),
                    "start_base_pose_world": {"position": {
                        "x": float(position["x"]),
                        "y": float(position["y"]),
                        "z": float(position.get("z", 0.34)),
                    }},
                    "start_base_rpy_rad": {
                        "roll": float(rpy.get("roll", 0.0)),
                        "pitch": float(rpy.get("pitch", 0.0)),
                        "yaw": float(rpy["yaw"]),
                    },
                    "tick": int(rec["frame_index"]),
                    "variant": 0,
                    "source_clearance_m": float(clearance_m),
                }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                wrote += 1
                total_low += int(clearance_m < float(args.low_clearance_threshold_m))
                if risk_file is not None:
                    risk_file.write(json.dumps({
                        "rgb_path": str(frame.resolve()),
                        "scene_id": scene_id,
                        "source_clearance_m": float(clearance_m),
                        "traversability_forward_m": (
                            0.0 if float(clearance_m) < float(args.risk_margin_m) else 2.0
                        ),
                    }, sort_keys=True) + "\n")
            if wrote:
                scenes_used += 1
                total_rows += wrote
            print(f"[{scenes_used:4d}] {scene_id} rows={wrote}", flush=True)
    if risk_file is not None:
        risk_file.close()
    print(json.dumps({
        "scenes": scenes_used,
        "rows": total_rows,
        "low_clearance_rows": total_low,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
