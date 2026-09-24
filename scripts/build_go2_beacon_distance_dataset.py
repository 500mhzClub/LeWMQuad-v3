#!/usr/bin/env python3
"""Join BC sequence shards with bfs_distance_to_landmark labels.

Produces per-scene arrays for the action-conditioned beacon-distance head:
for each corpus tick t with a next tick t+1 in the same env stream, the
example is (latent_t, proprio_t[executed primitive one-hot inside]) →
Δbfs_distance(t→t+1) per beacon color, plus the absolute distance at t.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

COLORS = ("red", "yellow", "blue", "green")


def _labels_index(labels_path: Path) -> dict[tuple[int, int], list[float]]:
    out: dict[tuple[int, int], list[float]] = {}
    with labels_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            bfs = row.get("bfs_distance_to_landmark") or {}
            dists = [float("nan")] * len(COLORS)
            for key, value in bfs.items():
                m = re.search(r"landmark_(red|yellow|blue|green)", str(key))
                if m and value is not None:
                    dists[COLORS.index(m.group(1))] = float(value)
            out[(int(row["env_idx"]), int(row["timestamp_ns"]))] = dists
    return out


def _find_labels(rollout_root: Path, family: str, split: str, scene_id: str) -> Path | None:
    for chunk in sorted((rollout_root / split / family).glob("chunk_*")):
        p = chunk / "labels" / scene_id / "labels.jsonl"
        if p.is_file():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-sequences-dir", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-envs", type=int, default=48)
    parser.add_argument("--step-ns", type=int, default=100000000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.bc_sequences_dir / "meta.json").read_text())
    shards = sorted(args.bc_sequences_dir.glob("*.npz"))
    done = 0
    for shard in shards:
        scene_id = shard.stem
        out_path = args.output_dir / f"{scene_id}.npz"
        if out_path.is_file():
            continue
        labels_path = _find_labels(args.rollout_root, str(args.family), str(args.split), scene_id)
        if labels_path is None:
            continue
        lab = _labels_index(labels_path)
        d = np.load(shard, allow_pickle=True)
        ticks = d["ticks"]; lat = d["latents"]; pro = d["proprio"]
        n_envs = int(args.n_envs)
        keep_i, deltas, dists = [], [], []
        for i in range(len(ticks) - 1):
            if int(ticks[i + 1]) - int(ticks[i]) != n_envs:
                continue
            env = int(ticks[i]) % n_envs
            step = int(ticks[i]) // n_envs
            ts0 = (step + 1) * int(args.step_ns)
            ts1 = (step + 2) * int(args.step_ns)
            d0 = lab.get((env, ts0)); d1 = lab.get((env, ts1))
            if d0 is None or d1 is None:
                continue
            delta = [b - a for a, b in zip(d0, d1)]
            if any(np.isnan(delta)) or any(abs(x) > 5 for x in delta):
                continue
            keep_i.append(i); deltas.append(delta); dists.append(d0)
        if len(keep_i) < 64:
            continue
        np.savez_compressed(
            out_path,
            latents=lat[keep_i].astype(np.float32),
            proprio=pro[keep_i].astype(np.float32),
            delta_bfs=np.asarray(deltas, dtype=np.float32),
            bfs=np.asarray(dists, dtype=np.float32),
        )
        done += 1
        print(f"[{done:4d}] {scene_id} rows={len(keep_i)}", flush=True)
    (args.output_dir / "meta.json").write_text(json.dumps({
        "schema": "go2_beacon_distance_dataset_v0",
        "colors": list(COLORS),
        "latent_dim": meta["latent_dim"],
        "proprio_feature_dim": meta["proprio_feature_dim"],
        "primitives": meta["primitives"],
        "frozen_jepa_checkpoint": meta["frozen_jepa_checkpoint"],
        "image_size": meta["image_size"],
        "scenes": done,
    }, indent=2) + "\n")
    print(json.dumps({"scenes": done}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
