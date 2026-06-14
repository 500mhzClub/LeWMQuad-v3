#!/usr/bin/env python3
"""Single-pose diagnostic: does ego-depth occupancy block a known-clear corridor?

Places the camera at a graph cell facing a CONNECTED neighbor (a corridor the
privileged grid considers traversable), optionally sweeps a scan to accumulate
the rolling map, then compares ``DepthLocalObstacleModel.is_free`` against the
privileged ``InflatedOccupancyGrid`` along the corridor centerline. Isolates the
depth->occupancy projection / inflation from the full benchmark control loop.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_lewm_closed_loop_mpc import _quat_wxyz_from_yaw, _xy_distance  # noqa: E402
from benchmark_topo_nav_e2e import _render_perception_tensor_from_base  # noqa: E402
from lewm.planning.depth_local_obstacles import (  # noqa: E402
    DepthLocalObstacleConfig,
    DepthLocalObstacleModel,
)
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-corpus", type=Path, required=True)
    p.add_argument("--split", default="test_id")
    p.add_argument("--family", default="medium_enclosed_maze")
    p.add_argument("--scene-offset", type=int, default=0)
    p.add_argument("--platform-manifest", type=Path,
                   default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    p.add_argument("--backend", default="vulkan")
    p.add_argument("--apply-textures", action="store_true")
    p.add_argument("--scan-frames", type=int, default=0,
                   help="rotate-in-place frames fused before probing (0 = single forward frame)")
    p.add_argument("--inflation-m", type=float, default=0.22)
    p.add_argument("--max-age-frames", type=int, default=100000)
    p.add_argument("--robot-free-radius-m", type=float, default=0.20)
    p.add_argument("--stride-px", type=int, default=8)
    p.add_argument("--grid-inflation-m", type=float, default=0.20)
    p.add_argument("--dump-geometry", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    platform = load_platform_manifest(args.platform_manifest.resolve())
    scene_dirs = sorted(find_scene_dirs(args.scene_corpus.resolve(), split=args.split,
                                        family=args.family), key=lambda q: q.name)
    scene_dir = scene_dirs[args.scene_offset]
    pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
    print(f"scene {pack.scene_id}")
    build = build_scene_from_pack(pack, n_envs=1, backend=args.backend, show_viewer=False,
                                  render_robot=False, apply_textures=bool(args.apply_textures))
    graph = pack.scene_graph
    if args.dump_geometry:
        print("=== node geometry (cell center, neighbors) ===")
        for c in range(0, 14):
            try:
                ctr = graph.cell_center(c)
                print(f"  cell {c}: ({ctr[0]:.2f},{ctr[1]:.2f}) neighbors {sorted(graph.neighbors(c))}")
            except Exception:
                pass
    base_z = float(pack.robot.spawn_xyz_m[2])
    spawn_xy = np.asarray(pack.robot.spawn_xyz_m[:2], dtype=np.float64)
    cells = [int(n.node_id) for n in graph.manifest.graph_nodes]
    start = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), spawn_xy))
    nbrs = sorted(graph.neighbors(start))
    if not nbrs:
        raise SystemExit("start cell has no neighbors")
    nbr = nbrs[0]
    sc = np.asarray(graph.cell_center(start), dtype=np.float64)
    nc = np.asarray(graph.cell_center(nbr), dtype=np.float64)
    bearing = math.atan2(nc[1] - sc[1], nc[0] - sc[0])
    pitch = float(np.hypot(*(nc - sc)))
    print(f"start cell {start} center {sc.round(2)} -> neighbor {nbr} center {nc.round(2)} "
          f"bearing {math.degrees(bearing):.0f} deg, pitch {pitch:.2f} m")

    cfg = DepthLocalObstacleConfig(
        vertical_fov_deg=float(pack.camera.fov_deg),
        occupied_inflation_m=float(args.inflation_m),
        max_age_frames=int(args.max_age_frames),
        sample_stride_px=int(args.stride_px),
        robot_free_radius_m=float(args.robot_free_radius_m),
        unknown_is_free=False,
        debug_capture=True,
    )
    depth_model = DepthLocalObstacleModel(cfg, odometry_detail="probe sim pose",
                                          odometry_deployment_valid=False)
    grid = InflatedOccupancyGrid(graph.manifest, cell_size_m=0.05,
                                 inflation_m=float(args.grid_inflation_m))

    yaws = [bearing]
    if args.scan_frames > 0:
        yaws = [bearing + 2.0 * math.pi * k / args.scan_frames for k in range(args.scan_frames)]
        yaws.append(bearing)  # finish facing the corridor
    false_occ = []  # obstacle points landing where the privileged grid is free
    for yaw in yaws:
        base_xyz = np.asarray([sc[0], sc[1], base_z], dtype=np.float32)
        _img, depth, pose = _render_perception_tensor_from_base(
            build, pack, base_xyz_m=base_xyz, base_quat_wxyz=_quat_wxyz_from_yaw(yaw), device=device)
        depth_model.update(depth, camera_position_xyz=pose.position,
                           camera_rotation=pose.rotation, robot_xy=(float(sc[0]), float(sc[1])))
        for (row, col, dist, ex, ey, ez, obs) in depth_model.debug_points():
            if obs and grid.is_free((ex, ey)) and _xy_distance((ex, ey), sc) < 1.3:
                false_occ.append((math.degrees(yaw), row / max(depth.shape[-2], 1),
                                  dist, ex, ey, ez))

    print(f"\nfused {len(yaws)} frame(s); diagnostics:")
    d = depth_model.diagnostics()
    for k in ("recent_free_cells", "recent_occupied_cells", "last_depth_min_m", "last_depth_max_m",
              "last_endpoint_height_min_m", "last_endpoint_height_max_m"):
        print(f"  {k}: {d.get(k)}")
    print(f"  camera height z: {float(pose.position[2]):.3f} m | last depth shape: {depth.shape} "
          f"| fov_deg(vertical): {pack.camera.fov_deg}")
    print(f"\n  FALSE occupied (obstacle point where grid is FREE, within 1.3 m): {len(false_occ)} pts")
    print("   cam_yaw  row/H  dist_m   x      y      height")
    for (yawd, rr, dist, ex, ey, ez) in sorted(false_occ, key=lambda p: p[2])[:25]:
        print(f"   {yawd:6.0f}   {rr:4.2f}  {dist:5.2f}  {ex:6.2f} {ey:6.2f}  {ez:6.3f}")

    # Dump last-frame points landing on the corridor centerline (|dy| < 0.20 m,
    # between start and just past the neighbor), to reveal what creates the
    # mid-corridor obstacle cells: floor returns (bottom rows, low height) vs walls.
    yc = float(sc[1])
    h_depth = depth.shape[-2] if depth.ndim >= 2 else 0
    print("\n  centerline-region projected points (|dy|<0.20m):")
    print("   row/H   col    dist_m   x      y      height  obstacle?")
    pts = [p for p in depth_model.debug_points()
           if abs(p[4] - yc) < 0.20 and sc[0] - 0.1 <= p[3] <= nc[0] + 0.4]
    pts.sort(key=lambda p: p[3])
    for (row, col, dist, ex, ey, ez, obs) in pts[:40]:
        rr = row / max(h_depth, 1)
        print(f"   {rr:4.2f}  {col:4d}   {dist:5.2f}  {ex:6.2f} {ey:6.2f}  {ez:6.3f}   {obs}")
    n_obs = sum(1 for p in pts if p[6])
    print(f"  centerline-region points: {len(pts)} ({n_obs} obstacle); "
          f"row fraction 0=top,1=bottom")

    # Probe the corridor centerline start -> neighbor.
    print("\n  dist_m   depth_free  grid_free   nearest_occ_m")
    n_steps = int(pitch / 0.05) + 1
    disagree = 0
    for i in range(n_steps + 1):
        t = i / max(n_steps, 1)
        x = sc[0] + (nc[0] - sc[0]) * t
        y = sc[1] + (nc[1] - sc[1]) * t
        df = depth_model.is_free((x, y))
        gf = bool(grid.is_free((x, y)))
        # nearest occupied cell distance in the depth map
        nearest = None
        for (ci, cj) in depth_model._occupied:  # noqa: SLF001 (diagnostic)
            cx = (ci + 0.5) * cfg.cell_size_m
            cy = (cj + 0.5) * cfg.cell_size_m
            dd = math.hypot(cx - x, cy - y)
            nearest = dd if nearest is None else min(nearest, dd)
        flag = "" if df == gf else "  <-- DISAGREE"
        if df != gf:
            disagree += 1
        nn = "n/a" if nearest is None else f"{nearest:.2f}"
        print(f"  {t*pitch:5.2f}    {str(df):5s}      {str(gf):5s}      {nn:>6s}{flag}")
    print(f"\n  centerline disagreements (depth blocks where grid is free): {disagree}/{n_steps+1}")

    # Top-down occupancy comparison over a window around the start cell.
    # ego-depth: '#'=occupied, '.'=observed-free, '?'=unknown(blocked); grid: 'X'/'.'.
    half = 1.2
    res = 0.1
    n = int(half / res)
    print(f"\n  occupancy window +-{half:.1f} m around start (res {res:.2f} m), +x right, +y up")
    print("  ego-depth (#=occ .=free ?=unknown, S=start)        privileged grid (X=occ .=free)")
    for jj in range(n, -n - 1, -1):
        y = sc[1] + jj * res
        row_depth, row_grid = [], []
        for ii in range(-n, n + 1):
            x = sc[0] + ii * res
            at_start = (ii == 0 and jj == 0)
            if depth_model.is_free((x, y)):
                row_depth.append("S" if at_start else ".")
            else:
                occ = depth_model._has_recent_cell_within(  # noqa: SLF001
                    depth_model._occupied, (x, y), cfg.occupied_inflation_m)
                row_depth.append("#" if occ else "?")
            row_grid.append("." if grid.is_free((x, y)) else "X")
        print("  " + "".join(row_depth) + "      " + "".join(row_grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
