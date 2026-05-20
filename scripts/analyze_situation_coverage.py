#!/usr/bin/env python3
"""Measure §10 situation coverage from spec-mix raw rollouts.

For each converted ``raw_rollout`` under ``--raw-root`` (one dir per scene),
join the per-block command stream (``command_source``, ``primitive_name``)
with the per-tick base pose, locate each pose on the scene graph, and tally
situation counts — especially **dead-end entry** and **dead-end recovery**,
broken down by the collector that produced them. This answers whether the
recovery/frontier collectors actually generate dead-end behaviour (the
privileged route teacher avoids dead-ends by construction).

Counts are per-block (macro transitions) so they line up with the §10 /
§11 minimums. The script extrapolates the sampled rate to a full
train-tier target and prints the gap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lewm_worlds"))
sys.path.insert(0, str(REPO / "lewm_genesis"))

from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.scene_graph import SceneGraph
from lewm_worlds.labels.topology import local_graph_type_per_node

BASE_STATE = "/lewm/go2/base_state"
COMMAND_BLOCK = "/lewm/go2/command_block"
NEAR_COLLISION_M = 0.30


def _find_manifest(corpus: Path, scene_id: str) -> Path | None:
    for mf in corpus.rglob("manifest.json"):
        if mf.parent.name == scene_id:
            return mf
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--scene-corpus", type=Path, required=True)
    ap.add_argument("--train-scenes", type=int, default=1000)
    ap.add_argument("--streams-per-scene", type=int, default=48)
    args = ap.parse_args()

    # Per-command-source block tallies.
    src_blocks = Counter()
    src_deadend = Counter()       # blocks whose pose is on a dead-end cell
    src_backward = Counter()      # backward primitive on/adjacent to dead-end
    prim_blocks = Counter()
    near_collision = 0
    stuck_blocks = 0
    total_blocks = 0
    deadend_entry_blocks = 0      # any collector, pose at dead-end
    deadend_recovery_blocks = 0   # backward/yaw at dead-end (recovery behaviour)
    scenes = 0
    sampled_scene_streams = 0

    for summ in sorted(args.raw_root.rglob("summary.json")):
        scene_dir = summ.parent
        scene_id = scene_dir.name
        msgs = scene_dir / "messages.jsonl"
        if not msgs.is_file():
            continue
        mf = _find_manifest(args.scene_corpus, scene_id)
        if mf is None:
            continue
        manifest = parse_scene_manifest_dict(json.loads(mf.read_text()))
        scene = SceneGraph(manifest)
        node_type = local_graph_type_per_node(manifest)
        dead_cells = {c for c, t in node_type.items() if t == "dead_end"}
        scenes += 1

        cur_src: dict[int, str] = {}
        cur_prim: dict[int, str] = {}
        envs_seen = set()
        for line in msgs.open():
            r = json.loads(line)
            ct = r.get("canonical_topic", "")
            env = int(r.get("env_index") or 0)
            if ct == COMMAND_BLOCK:
                p = r["payload"]
                cur_src[env] = str(p.get("command_source") or "unknown")
                cur_prim[env] = str(p.get("primitive_name") or "unknown")
            elif ct == BASE_STATE:
                envs_seen.add(env)
                pos = r["payload"]["pose_world"]["position"]
                xy = (float(pos["x"]), float(pos["y"]))
                src = cur_src.get(env, "unknown")
                prim = cur_prim.get(env, "unknown")
                # Treat each base_state under the active block as one macro
                # sample (the command block is held across its ticks; we count
                # the executed-command stream at base_state rate as a proxy
                # for macro transitions).
                total_blocks += 1
                src_blocks[src] += 1
                prim_blocks[prim] += 1
                cell = int(scene.locate(xy).cell_id)
                clear = float(scene.clearance_to_walls(xy))
                if clear < NEAR_COLLISION_M:
                    near_collision += 1
                on_dead = cell in dead_cells
                if on_dead:
                    deadend_entry_blocks += 1
                    src_deadend[src] += 1
                    if prim in ("backward", "yaw_left", "yaw_right"):
                        deadend_recovery_blocks += 1
                        src_backward[src] += 1
        sampled_scene_streams += len(envs_seen)

    scale = (args.train_scenes * args.streams_per_scene) / max(1, sampled_scene_streams)

    def line(name, sampled, minimum):
        proj = int(sampled * scale)
        flag = "OK" if proj >= minimum else "SHORT"
        print(f"  {name:28s} sampled={sampled:>8d}  projected={proj:>10d}  min={minimum:>8d}  {flag}")

    print(f"=== situation coverage (spec mix) ===")
    print(f"scenes={scenes}  sampled_streams={sampled_scene_streams}  total_macro={total_blocks}")
    print(f"scale to {args.train_scenes} scenes x {args.streams_per_scene} streams = x{scale:.1f}\n")
    line("dead-end entry", deadend_entry_blocks, 200000)
    line("dead-end recovery/backout", deadend_recovery_blocks, 150000)
    line("near-collision (no contact)", near_collision, 250000)
    print()
    print("realized command-source mix (macro fraction):")
    for s, n in src_blocks.most_common():
        print(f"  {s:24s} {n/total_blocks*100:5.1f}%")
    print("\ndead-end visits attributed to collector (the key question):")
    for s in sorted(set(src_blocks)):
        d = src_deadend.get(s, 0); b = src_blocks.get(s, 1)
        print(f"  {s:24s} dead-end-steps={d:>7d}  ({d/b*100:4.1f}% of its blocks)")
    print("\nprimitive distribution:")
    tp = sum(prim_blocks.values()) or 1
    for p, n in prim_blocks.most_common():
        print(f"  {p:18s} {n/tp*100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
