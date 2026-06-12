#!/usr/bin/env python3
"""One-off probe: where in the demo maze does ANY colour appear in ego views?

Replays the same DFS tour as benchmark_topo_nav_e2e (same seed), then for each
visited cell renders 8 yaw views and reports the per-cell max saturation, the
best yaw, and the distance/bearing to the nearest landmark. Drives the goal
selection rule for the demo from measurement instead of guesswork.
"""
from __future__ import annotations

import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_lewm_closed_loop_mpc import _quat_wxyz_from_yaw, _render_tensor_from_base, _xy_distance  # noqa: E402
from benchmark_topo_nav_e2e import _saturation_score, graph_tour_cells  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import load_platform_manifest, load_scene_pack  # noqa: E402

SCENE = REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z/test_id/medium_enclosed_maze/medium_enclosed_maze_0198ae36dc38"
SEED = 20260609

platform = load_platform_manifest(REPO_ROOT / "config/go2_platform_manifest.yaml")
pack = load_scene_pack(SCENE, platform_manifest=platform, workspace_root=REPO_ROOT)
build = build_scene_from_pack(pack, n_envs=1, backend="vulkan", show_viewer=False,
                              render_robot=False, apply_textures=True)
graph = pack.scene_graph
base_z = float(pack.robot.spawn_xyz_m[2])
device = torch.device("cpu")

cells = [n.node_id for n in graph.manifest.graph_nodes]
spawn_xy = np.asarray(pack.robot.spawn_xyz_m[:2], dtype=np.float64)
start_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), spawn_xy))
trial_rng = random.Random(SEED + 0 * 1009 + zlib.crc32(pack.scene_id.encode()) % 100000)
tour = graph_tour_cells(graph, start_cell, 18, trial_rng)
visited = sorted(set(tour))
landmarks = [(str(o), np.asarray(graph.landmark_xy_for_cell(c) or graph.cell_center(c), dtype=np.float64))
             for o, c in graph.landmark_cells]
print(f"visited {len(visited)} cells: {visited}")
print("landmarks:", [(n, tuple(np.round(xy, 2))) for n, xy in landmarks])

rows = []
for c in visited:
    cx = np.asarray(graph.cell_center(c), dtype=np.float64)
    best = (0.0, 0.0)
    for k in range(8):
        yaw = -math.pi + k * math.pi / 4
        img = _render_tensor_from_base(build, pack,
                                       base_xyz_m=np.asarray([cx[0], cx[1], base_z], dtype=np.float32),
                                       base_quat_wxyz=_quat_wxyz_from_yaw(yaw), device=device)
        sat = _saturation_score(img)
        if sat > best[0]:
            best = (sat, yaw)
    name, d = min(((n, _xy_distance(cx, xy)) for n, xy in landmarks), key=lambda t: t[1])
    rows.append((best[0], c, best[1], name, d))
    print(f"cell {c:3d} at {tuple(np.round(cx,2))}: max_sat={best[0]:.4f} at yaw {best[1]:+.2f}; "
          f"nearest {name} d={d:.2f} m", flush=True)

rows.sort(reverse=True)
print("\nTOP 5 colourful (cell, sat, yaw):", [(r[1], round(r[0], 4), round(r[2], 2)) for r in rows[:5]])
