#!/usr/bin/env python3
"""Ray-depth labels for the visual free-space head.

For each row of the BC sequence shards (frozen-JEPA latent + known world pose)
cast K rays across the camera FOV over the scene occupancy grid (uninflated)
and record free depth per ray, capped. Labels are privileged offline geometry;
the head runs from RGB latents at inference.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _process_scene(job) -> str:
    shard_path, rollout_root_s, corpus_s, family, split, n_envs, step_ns, k_rays, fov_deg, depth_cap = job
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
    sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
    from lewm_worlds.manifest import parse_scene_manifest_dict
    from lewm_worlds.planning_grid import InflatedOccupancyGrid

    shard = Path(shard_path)
    scene_id = shard.stem
    out_path = shard.parent.parent / "ray_depth" / f"{scene_id}.npz"
    if out_path.is_file():
        return f"SKIP {scene_id}"
    manifest_path = Path(corpus_s) / split / family / scene_id / "manifest.json"
    if not manifest_path.is_file():
        return f"NOMANIFEST {scene_id}"
    grid = InflatedOccupancyGrid(
        parse_scene_manifest_dict(json.loads(manifest_path.read_text())),
        cell_size_m=0.05,
        inflation_m=0.0,
    )
    # pose per row from frames.jsonl via the rollout plan
    plan = None
    for chunk in sorted((Path(rollout_root_s) / split / family).glob("chunk_*")):
        cand = chunk / "plan"
        for p in cand.glob(f"*_{scene_id}"):
            if (p / "frames.jsonl").is_file():
                plan = p / "frames.jsonl"
                break
        if plan:
            break
    if plan is None:
        return f"NOPLAN {scene_id}"
    pose_by_tick: dict[int, tuple[float, float, float]] = {}
    with plan.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            position = rec["base_pose_world"]["position"]
            pose_by_tick[int(rec["frame_index"])] = (
                float(position["x"]),
                float(position["y"]),
                float(rec["base_rpy_rad"]["yaw"]),
            )
    d = np.load(shard, allow_pickle=True)
    ticks = d["ticks"]
    half_fov = math.radians(float(fov_deg)) / 2.0
    angles = np.linspace(-half_fov, half_fov, int(k_rays))
    depths = np.full((len(ticks), int(k_rays)), np.nan, dtype=np.float32)
    for i, tick in enumerate(ticks):
        pose = pose_by_tick.get(int(tick))
        if pose is None:
            continue
        x0, y0, yaw = pose
        for j, a in enumerate(angles):
            ang = yaw + float(a)
            dx, dy = math.cos(ang) * 0.05, math.sin(ang) * 0.05
            depth = float(depth_cap)
            for step in range(1, int(depth_cap / 0.05) + 1):
                if grid.configuration_clearance_m((x0 + dx * step, y0 + dy * step)) <= 0.01:
                    depth = step * 0.05
                    break
            depths[i, j] = depth
    keep = ~np.isnan(depths).any(axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        latents=d["latents"][keep],
        depths=depths[keep],
        ticks=ticks[keep],
    )
    return f"WROTE {scene_id} rows={int(keep.sum())}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-sequences-dir", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, required=True)
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--split", default="train")
    parser.add_argument("--k-rays", type=int, default=15)
    parser.add_argument("--fov-deg", type=float, default=78.323)
    parser.add_argument("--depth-cap-m", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    shards = sorted(args.bc_sequences_dir.glob("*.npz"))
    jobs = [
        (str(s), str(args.rollout_root), str(args.scene_corpus), args.family, args.split,
         48, 100000000, args.k_rays, args.fov_deg, args.depth_cap_m)
        for s in shards
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        for msg in pool.map(_process_scene, jobs, chunksize=1):
            if msg.startswith("WROTE"):
                done += 1
            print(msg, flush=True)
    meta = {
        "schema": "go2_ray_depth_dataset_v0",
        "k_rays": int(args.k_rays),
        "fov_deg": float(args.fov_deg),
        "depth_cap_m": float(args.depth_cap_m),
        "scenes": done,
    }
    out_dir = args.bc_sequences_dir.parent / "ray_depth"
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
