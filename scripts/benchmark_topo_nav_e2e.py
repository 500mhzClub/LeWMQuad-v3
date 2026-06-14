#!/usr/bin/env python3
"""Stage 4a — closed-loop topological navigation in Genesis (tour + seek).

The first end-to-end run of the learned stack
(`docs/lewm_topological_nav_stage3_wiring_2026-06-09.md`): on a held-out scene,
the robot is first DRIVEN along a scripted graph tour while the
`TopologicalNavigator` builds its memory from rendered ego frames only (the
tour's *motion* is privileged — it stands in for autonomous exploration, which
is deferred; *perception* is deployment-valid). It is then handed a goal
IMAGE and must navigate back to that place with the learned stack: filter
localization -> raw-frame goal match -> BFS + lookahead sub-goal -> LocalMPC
servoing on the sub-goal's stored representative observation -> perceptual
arrival. Ground-truth distance is logged for EVALUATION only.

Runtime collision avoidance consumes a ``LocalObstacleModel``. The default
benchmark adapter is an explicitly deployment-invalid privileged manifest grid;
inject a provider backed by accepted onboard sensing/odometry and pass
``--require-deployment-valid-local-obstacles`` for a deployable-nav claim.

Goal-image convention (the registered goal-facing constraint): the goal is
rendered at the goal cell center facing the heading the tour used when passing
through that cell — i.e. "a photo taken the way the place was seen", matching
how a user-supplied goal photo would relate to prior experience.

Policies on identical (start, goal):
  - topo  : the learned stack (fallback = servo at goal image when unplannable)
  - v2    : keyframe baseline — LocalMPC directly on the final goal image
  - bearing: privileged oracle toward the true goal xy
  - hold

Runs in the genesis vulkan venv (CPU torch):
  .generated/venvs/genesis_render_vulkan/bin/python scripts/benchmark_topo_nav_e2e.py \
    --checkpoint models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt \
    --belief-encoder .generated/topo_nav/belief_encoder_seq4_e9_v6_train32_encoders/belief_encoder_seed20260609.pt \
    --loop-head .generated/topo_nav/same_yaw_loop_head.pt \
    --apply-textures --backend vulkan --family medium_enclosed_maze --scene-limit 2
"""
from __future__ import annotations

import argparse
import zlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_lewm_closed_loop_mpc import (  # noqa: E402  (reuses env guards)
    _candidate_action_tensor,
    _choose_bearing_primitive,
    _choose_lewm_primitive,
    _lewm_primitive_costs,
    _current_pose,
    _execute_kinematic_primitive,
    _parse_csv,
    _primitive_active_blocks,
    _quat_xyzw_from_wxyz,
    _quat_wxyz_from_yaw,
    _render_tensor_from_base,
    _render_third_person,
    _set_pose,
    _xy_distance,
    _yaw_from_quat_wxyz,
)
from lewm.memory.topological_navigator import TopologicalNavigator  # noqa: E402
from lewm.models.belief_encoder import BeliefEncoder  # noqa: E402
from lewm.models.loop_closure import LoopClosureHead  # noqa: E402
from lewm.planning.depth_local_obstacles import (  # noqa: E402
    DepthLocalObstacleConfig,
    DepthLocalObstacleModel,
)
from lewm.planning.local_obstacles import (  # noqa: E402
    PrivilegedGridObstacleModel,
    require_deployment_valid_local_obstacles,
)
from benchmark_lewm_closed_loop_mpc import _execute_physical_primitive  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits, expand_primitive_to_block  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.collectors.base import wrap_angle_pi  # noqa: E402
from lewm_genesis.camera_safety import (  # noqa: E402
    camera_safety_config_from_pack,
    safe_camera_pose_from_base,
)
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import (  # noqa: E402
    effective_camera_mount_xyz_rpy,
    find_scene_dirs,
    load_platform_manifest,
    load_scene_pack,
)
from lewm_genesis.rollout import DEFAULT_GO2_STANCE_RAD, _resolve_rollout_leg_dof_indices  # noqa: E402
from lewm_worlds.planning_grid import InflatedOccupancyGrid  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402

import torch.nn.functional as F  # noqa: E402


def load_navigator_parts(args, device):
    blob = torch.load(args.belief_encoder, map_location="cpu", weights_only=False)
    config = blob["config"]
    encoder = BeliefEncoder(
        latent_dim=int(blob["latent_dim"]), embedding_dim=int(blob["embedding_dim"]),
        hidden=int(config["hidden"]), n_heads=int(config["n_heads"]),
        n_layers=int(config["n_layers"]), max_len=max(int(config["history"]), 16),
        dropout=float(config["dropout"]),
    )
    encoder.load_state_dict(blob["state_dict"])
    encoder = encoder.to(device).eval()
    head_blob = torch.load(args.loop_head, map_location="cpu", weights_only=False)
    head = LoopClosureHead(int(head_blob["embedding_dim"]), hidden=int(head_blob["hidden"]), dropout=0.0)
    head.load_state_dict(head_blob["state_dict"])
    head = head.to(device).eval()
    platt_a, platt_b = float(head_blob["platt_a"]), float(head_blob["platt_b"])
    tau_new = float(head_blob["tau_candidates"]["calP95"])

    def scorer(query, nodes):
        logits = head(query.unsqueeze(0).expand(len(nodes), -1), nodes)
        return torch.sigmoid(platt_a * logits + platt_b)

    return encoder, scorer, tau_new


def graph_tour_cells(graph, start_cell: int, max_cells: int, rng,
                     *, reprise_cells: int = 0) -> list[int]:
    """DFS traversal (with backtracking steps) over the cell graph — every
    consecutive pair is graph-adjacent, so a waypoint follower can drive it.

    ``reprise_cells > 0`` re-walks the first outbound legs after the DFS
    unwinds to the root: the memory is a time-DIRECTED chain, so at the bare
    tour end every edge points backward and NO goal is forward-routable
    (measured: all wide-maze candidate routes were 100% reversed edges). The
    reprise makes the filter re-localize onto the chain's START, minting the
    late->early loop-closure edge that puts the whole tour downstream of the
    seek start."""
    adjacency: dict[int, list[int]] = {}
    for node in graph.manifest.graph_nodes:
        adjacency[node.node_id] = sorted(graph.neighbors(node.node_id))
    visited, order = set(), []

    def dfs(cell):
        visited.add(cell)
        order.append(cell)
        neighbors = adjacency.get(cell, [])
        rng.shuffle(neighbors)
        for nxt in neighbors:
            if nxt not in visited and len(visited) < max_cells:
                dfs(nxt)
                order.append(cell)  # backtrack step keeps adjacency

    sys.setrecursionlimit(10000)
    dfs(int(start_cell))
    if reprise_cells > 0 and len(order) > 1:
        order = order + order[1:1 + int(reprise_cells)]
    return order


def _graph_shortest_path(graph, start: int, target: int) -> list[int] | None:
    if start == target:
        return [int(start)]
    frontier, parent = [int(start)], {int(start): None}
    while frontier and int(target) not in parent:
        nxt = []
        for node in frontier:
            for neighbor in graph.neighbors(node):
                neighbor = int(neighbor)
                if neighbor not in parent:
                    parent[neighbor] = node
                    nxt.append(neighbor)
        frontier = nxt
    if int(target) not in parent:
        return None
    path = [int(target)]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def _segment_is_traversable(grid, start_xy, end_xy, *, step_m: float = 0.05) -> bool:
    """Reject synthetic edges whose centerline crosses inflated occupancy.

    Do not extend a body probe beyond the endpoint: landmark photo standoffs
    intentionally terminate near the beacon, and runtime WALK's strict capsule
    gate stops the robot before contact. The mapping contract needed here is
    free-space continuity, which rules out the v28-v34 wall-crossing spur.
    """
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    delta = end - start
    span = float(np.linalg.norm(delta))
    if span <= 1e-9:
        return bool(grid.is_free((float(start[0]), float(start[1]))))
    n_steps = max(1, int(math.ceil(span / max(float(step_m), 1e-3))))
    for i in range(n_steps + 1):
        point = start + delta * (i / n_steps)
        if not grid.is_free((float(point[0]), float(point[1]))):
            return False
    return True


def _augment_tour_with_landmark_anchor(graph, tour, grid, *, goal_standoff_m: float,
                                       goal_min_hops: int, goal_max_hops: int,
                                       reprise_cells: int) -> list[int]:
    """Splice a minimal traversed excursion to a route-valid landmark anchor."""
    reprise_n = min(max(int(reprise_cells), 0), max(len(tour) - 1, 0))
    core = list(tour[:-reprise_n] if reprise_n else tour)
    reprise = list(tour[-reprise_n:] if reprise_n else [])
    if not core:
        return list(tour)
    seek_end = int(tour[-1])
    cells = [int(node.node_id) for node in graph.manifest.graph_nodes]
    best = None
    for _obj, landmark_cell in graph.landmark_cells:
        beacon = np.asarray(graph.landmark_xy_for_cell(landmark_cell)
                            or graph.cell_center(landmark_cell), dtype=np.float64)
        for candidate in cells:
            hops = graph.bfs_distance(seek_end, candidate)
            if hops is None or not (int(goal_min_hops) <= hops <= int(goal_max_hops)):
                continue
            anchor_xy = np.asarray(graph.cell_center(candidate), dtype=np.float64)
            gap = _xy_distance(anchor_xy, beacon)
            if not (0.55 <= gap <= 1.60):
                continue
            standoff = beacon + (anchor_xy - beacon) / gap * float(goal_standoff_m)
            if not _segment_is_traversable(grid, anchor_xy, standoff):
                continue
            for index, tour_cell in enumerate(core):
                path = _graph_shortest_path(graph, int(tour_cell), candidate)
                if path is None:
                    continue
                key = (len(path), gap)
                if best is None or key < best[0]:
                    best = (key, index, path, candidate)
    if best is None:
        return list(tour)
    _key, index, path, candidate = best
    if len(path) > 1:
        excursion = path[1:] + list(reversed(path[:-1]))
        core[index + 1:index + 1] = excursion
    print(f"  tour added route-valid landmark anchor cell {candidate} "
          f"(detour {2 * (len(path) - 1)} edges)", flush=True)
    return core + reprise


def tour_pose_sequence(graph, tour_cells, base_z, *, step_m: float = 0.12,
                       turn_step_rad: float = 0.5, max_steps: int,
                       lookats: dict | None = None, lookat_dwell: int = 4):
    """Dense (pos, yaw) walk along the DFS cell path — the tour's motion is
    privileged (stands in for exploration), so we interpolate poses directly at
    a ~bank-cadence step; yaw faces the direction of motion, with turn-in-place
    poses inserted at heading changes so DFS backtracks read as turns rather
    than instant 180° flips. Turn poses are flagged (pos, yaw, is_turn=True)
    and are VIDEO-ONLY: they must not feed the navigator, or they mint
    rotation-only memory edges that the align->walk traversal then executes as
    forward hops (verified-run memory had translation edges only).

    ``lookats`` maps cell_id -> (bearing_rad, standoff_xy): on first arrival at
    such a cell the tour steps aside to the landmark standoff ENTIRELY
    video-only — the locomotion chain must stay pure corridor — and emits one
    ``"spur"`` pose at the standoff apex, which the harness registers via
    ``navigator.insert_spur`` as a terminal goal-only node (anchor<->spur
    edges). Feeding look-at frames through the filter instead splices the
    standoff INLINE into the chain and every passing route detours into the
    pillar (v18/v19: veto/realign storms, final 5.74 m).

    Pose kinds: "walk" = fed to the navigator; "turn" = video-only;
    "spur" = video-only + spur registration."""
    poses = []
    current = np.asarray(graph.cell_center(tour_cells[0]), dtype=np.float64)
    yaw = None
    pending = dict(lookats or {})

    def emit(xy, yw, kind):
        poses.append((np.asarray([xy[0], xy[1], base_z], dtype=np.float32), float(yw), kind))
        return len(poses) >= max_steps

    for cell in tour_cells[1:]:
        target = np.asarray(graph.cell_center(cell), dtype=np.float64)
        delta = target - current
        span = float(np.linalg.norm(delta))
        if span < 1e-6:
            continue
        seg_yaw = math.atan2(delta[1], delta[0])
        if yaw is not None and abs(wrap_angle_pi(seg_yaw - yaw)) > 1e-6:
            dyaw = wrap_angle_pi(seg_yaw - yaw)
            n_turn = max(1, int(math.ceil(abs(dyaw) / turn_step_rad)))
            for k in range(1, n_turn + 1):
                if emit(current, wrap_angle_pi(yaw + dyaw * (k / n_turn)), "turn"):
                    return poses
        yaw = seg_yaw
        n_steps = max(1, int(math.ceil(span / step_m)))
        for i in range(1, n_steps + 1):
            if emit(current + delta * (i / n_steps), yaw, "walk"):
                return poses
        current = target
        if int(cell) in pending:
            look_yaw, look_xy = pending.pop(int(cell))
            entry_yaw = yaw
            dyaw = wrap_angle_pi(look_yaw - yaw)
            n_turn = max(1, int(math.ceil(abs(dyaw) / turn_step_rad)))
            for k in range(1, n_turn + 1):
                if emit(current, wrap_angle_pi(yaw + dyaw * (k / n_turn)), "turn"):
                    return poses
            delta_l = np.asarray(look_xy, dtype=np.float64) - current
            n_app = max(1, int(math.ceil(float(np.linalg.norm(delta_l)) / step_m)))
            for i in range(1, n_app + 1):
                if emit(current + delta_l * (i / n_app), look_yaw, "turn"):
                    return poses
            if emit(current + delta_l, look_yaw, "spur"):
                return poses
            for _ in range(max(int(lookat_dwell) - 1, 0)):
                if emit(current + delta_l, look_yaw, "turn"):
                    return poses
            for i in range(n_app - 1, -1, -1):
                if emit(current + delta_l * (i / n_app), look_yaw, "turn"):
                    return poses
            dyb = wrap_angle_pi(entry_yaw - look_yaw)
            n_turn = max(1, int(math.ceil(abs(dyb) / turn_step_rad)))
            for k in range(1, n_turn + 1):
                if emit(current, wrap_angle_pi(look_yaw + dyb * (k / n_turn)), "turn"):
                    return poses
            yaw = entry_yaw
    return poses


def _colour_signature(image_chw: torch.Tensor):
    """(saturated-pixel fraction, normalized mean RGB of those pixels) over the
    CENTER CROP. The frozen latent is place/yaw-dominant and barely encodes hue
    — beacon standoff close-ups of DIFFERENT colours alias at ~1.0 latent
    cosine (measured v14/v16: a blue beacon node was the worst alias of a
    non-blue goal) — but the goal is literally a photo, so raw pixel colour is
    deployment-valid arrival evidence. Center crop because saturated CC0
    floor/wall textures at the frame border blend into a full-frame mean and
    drag different beacons' signatures together; goal and arrival views face
    the beacon by construction, so the middle 40 % box is mostly beacon."""
    arr = image_chw.detach().float()
    h, w = arr.shape[-2], arr.shape[-1]
    arr = arr[:, int(0.30 * h):int(0.70 * h), int(0.30 * w):int(0.70 * w)]
    mx, _ = arr.max(dim=0)
    mn, _ = arr.min(dim=0)
    mask = ((mx - mn) > 0.25) & (mx > 0.25)
    frac = float(mask.float().mean())
    if frac < 1e-4:
        return frac, None
    rgb = torch.stack([arr[c][mask].mean() for c in range(3)])
    rgb = rgb / rgb.sum().clamp_min(1e-6)
    return frac, rgb


def _colour_match(goal_sig, cur_sig, *, tol: float = 0.30,
                  min_frac: float | None = None) -> bool:
    """True when ``cur`` could pass the colour-gated stop for ``goal``. Goals
    without a colour signature accept anything (legacy bland-goal behaviour).
    The claimed view must be proportionally as colourful as the goal (>= 40 %
    of its saturated fraction): a distant beacon sliver or a tinted wall
    texture shares the hue but not the coverage of a standoff close-up."""
    goal_frac, goal_rgb = goal_sig
    if goal_rgb is None:
        return True
    cur_frac, cur_rgb = cur_sig
    # Coverage bar capped at 0.25: a standoff goal's own crop saturates to
    # ~1.0, and demanding 40 % of that exceeds what the robot sees from the
    # veto-limited servo distance (v23: arrival never counted at the beacon).
    required_frac = (max(0.04, min(0.25, 0.4 * goal_frac))
                     if min_frac is None else max(float(min_frac), 0.0))
    if cur_frac < required_frac or cur_rgb is None:
        return False
    return float((goal_rgb - cur_rgb).abs().sum()) <= tol


def _goal_colour_centroid_x(image_chw: torch.Tensor, goal_rgb: torch.Tensor,
                            *, tol: float = 0.30):
    """Horizontal position of the goal-coloured pixels in [-1, 1] (left..right)
    and their full-frame fraction. Drives the goal-object visual servo: a beacon
    centroid > 0 is to the robot's right (yaw right to centre it). Full frame, not
    the centre crop, so the servo can re-acquire a beacon drifting toward an edge.
    Returns (fraction, None) when no goal-coloured pixels are visible."""
    arr = image_chw.detach().float()
    h, w = arr.shape[-2], arr.shape[-1]
    mx, _ = arr.max(dim=0)
    mn, _ = arr.min(dim=0)
    sat = ((mx - mn) > 0.25) & (mx > 0.25)
    norm = arr / arr.sum(dim=0).clamp_min(1e-6)
    hue_close = (norm - goal_rgb.view(3, 1, 1)).abs().sum(dim=0) <= float(tol)
    mask = sat & hue_close
    frac = float(mask.float().mean())
    if frac < 1e-4:
        return frac, None
    cols = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    centroid_col = float(cols[mask].mean())
    return frac, (centroid_col / max(w - 1, 1)) * 2.0 - 1.0


def _saturation_score(image_chw: torch.Tensor) -> float:
    """Fraction of pixels that are strongly coloured (saturated + bright) — in
    these scenes that is essentially 'a coloured landmark fills part of the
    frame', which is what makes a goal image legible to a viewer."""
    arr = image_chw.detach().float()
    mx, _ = arr.max(dim=0)
    mn, _ = arr.min(dim=0)
    return float((((mx - mn) > 0.25) & (mx > 0.25)).float().mean())


def _render_tour_view(build, base_xyz, yaw, size: int = 224) -> np.ndarray:
    """Raised follow view for the tour replay. The standard follow cam sits at
    +1.2 m — below the maze wall tops — so it keeps crossing walls and the
    teleported tour reads as 'flying through walls'. This one stays above the
    walls looking down at the robot."""
    from PIL import Image
    heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    cam_pos = np.array(
        [float(base_xyz[0]) - heading[0] * 2.4,
         float(base_xyz[1]) - heading[1] * 2.4,
         float(base_xyz[2]) + 3.1],
        dtype=np.float32,
    )
    lookat = np.array(
        [float(base_xyz[0]) + heading[0] * 0.7,
         float(base_xyz[1]) + heading[1] * 0.7,
         float(base_xyz[2])],
        dtype=np.float32,
    )
    build.camera.set_pose(pos=cam_pos, lookat=lookat, up=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    rgb = RolloutRunner._extract_rgb(build.camera.render())
    if rgb is None:
        raise RuntimeError("Genesis camera returned no RGB frame (tour view)")
    if rgb.ndim == 4:
        rgb = rgb[0]
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((size, size))
    return np.array(img, copy=True)


def _render_perception_tensor_from_base(build, pack, *, base_xyz_m, base_quat_wxyz,
                                        device):
    """Render synchronized ego RGB + depth and return the effective camera pose."""
    from PIL import Image

    quat_xyzw = _quat_xyzw_from_wxyz(np.asarray(base_quat_wxyz, dtype=np.float32))
    mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(pack)
    pose, _safety = safe_camera_pose_from_base(
        np.asarray(base_xyz_m, dtype=np.float32),
        quat_xyzw,
        mount_xyz_body=mount_xyz,
        mount_rpy_body=mount_rpy,
        objects=pack.static_objects,
        config=camera_safety_config_from_pack(pack),
    )
    build.camera.set_pose(pos=pose.position, lookat=pose.lookat, up=pose.up)
    rendered = build.camera.render(rgb=True, depth=True, force_render=True)
    if not isinstance(rendered, tuple) or len(rendered) < 2:
        raise RuntimeError("Genesis camera returned no synchronized RGB/depth tuple")
    rgb = RolloutRunner._as_np(rendered[0])
    depth = RolloutRunner._as_np(rendered[1])
    if rgb.ndim == 4:
        rgb = rgb[0]
    if depth.ndim >= 3 and depth.shape[0] == 1:
        depth = depth[0]
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    if depth.ndim != 2:
        raise RuntimeError(f"Genesis camera returned unexpected depth shape {depth.shape}")
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((224, 224))
    chw = np.array(image, copy=True).transpose(2, 0, 1)
    return torch.from_numpy(chw).float().div_(255.0).to(device), depth, pose


def _feasible_fraction(build, registry, primitive, local_obstacles, command_dt_s,
                       *, horizon_blocks: int = 1, body_radius_m: float = 0.0,
                       lateral_radius_m: float = 0.0):
    """Fraction of a primitive's sub-steps permitted by local obstacle sensing.

    Physical mode passes horizon_blocks > 1 and body_radius_m > 0: one block is
    0.5 s = 0.10 m of forward_slow, so a base-center single-block check cannot
    see a wall the Go2's nose (~0.25 m ahead of base) is about to hit, and the
    gait keeps moving past the block boundary. That gap is exactly the v9
    "confident walk into the wall": feasible 1.0 with the nose at the surface."""
    block = expand_primitive_to_block(registry, primitive)
    pos, quat = _current_pose(build)
    x, y = float(pos[0]), float(pos[1])
    yaw = _yaw_from_quat_wxyz(quat)
    steps = list(block) * max(int(horizon_blocks), 1)
    allowed = 0
    for vx, vy, yaw_rate in steps:
        nx = x + (float(vx) * math.cos(yaw) - float(vy) * math.sin(yaw)) * command_dt_s
        ny = y + (float(vx) * math.sin(yaw) + float(vy) * math.cos(yaw)) * command_dt_s
        nyaw = wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
        if local_obstacles is not None:
            probes = [(nx, ny)]
            if body_radius_m > 0.0:
                probes.append((nx + body_radius_m * math.cos(nyaw),
                               ny + body_radius_m * math.sin(nyaw)))
                probes.append((nx - body_radius_m * math.cos(nyaw),
                               ny - body_radius_m * math.sin(nyaw)))
            if lateral_radius_m > 0.0:
                # Side probes: the nose/tail capsule is blind to walls
                # PARALLEL to the path, which only the 0.20 m grid inflation
                # covers — about the body half-width plus gait sway, so a
                # "feasible 1.0" path can scrape a side wall. Used by the
                # padded clearance SCORE, not the strict gate.
                probes.append((nx - lateral_radius_m * math.sin(nyaw),
                               ny + lateral_radius_m * math.cos(nyaw)))
                probes.append((nx + lateral_radius_m * math.sin(nyaw),
                               ny - lateral_radius_m * math.cos(nyaw)))
            if not all(local_obstacles.is_free(p) for p in probes):
                break
        x, y, yaw = nx, ny, nyaw
        allowed += 1
    return allowed / max(len(steps), 1)


def _nearest_free_xy(local_obstacles, xy, *, max_radius_m: float = 0.45,
                     step_m: float = 0.09):
    """Closest locally free point to ``xy`` (ring search). Fall recovery stands the
    robot up here instead of at the wall-contact point it fell at — standing up
    pressed into the wall just re-falls."""
    if local_obstacles is None or local_obstacles.is_free((float(xy[0]), float(xy[1]))):
        return float(xy[0]), float(xy[1])
    r = step_m
    while r <= max_radius_m + 1e-9:
        for k in range(16):
            ang = 2.0 * math.pi * k / 16.0
            p = (float(xy[0] + r * math.cos(ang)), float(xy[1] + r * math.sin(ang)))
            if local_obstacles.is_free(p):
                return p
        r += step_m
    return float(xy[0]), float(xy[1])


def blocks_per_revolution(registry, command_dt_s, primitive="yaw_right"):
    block = expand_primitive_to_block(registry, primitive)
    per_block = abs(sum(float(t[2]) for t in block)) * command_dt_s
    return max(4, int(math.ceil(2.0 * math.pi / max(per_block, 1e-6))))


def choose_vetoed_primitive(model, build, registry, local_obstacles, command_dt_s,
                            image, goal_image,
                            sequences, action_tensor, *, horizon_blocks: int = 1,
                            body_radius_m: float = 0.0):
    """Latent-cost ranking among kinematically feasible candidates only."""
    costs, first_primitives = _lewm_primitive_costs(model, image, goal_image, sequences, action_tensor)
    feasibility = {name: _feasible_fraction(build, registry, name, local_obstacles, command_dt_s,
                                            horizon_blocks=horizon_blocks,
                                            body_radius_m=body_radius_m)
                   for name in set(first_primitives)}
    order = np.argsort(costs)
    for idx in order:
        name = first_primitives[int(idx)]
        if name == "hold" or feasibility[name] >= 0.7:
            return name, float(costs[int(idx)])
    return first_primitives[int(order[0])], float(costs[int(order[0])])


def _demo_capture(build, pack, device, demo, status, *, physical=False):
    """Pass-1 demo capture = bookkeeping ONLY (pose + status). Rendering happens
    in a separate replay pass on a robot-visible scene, so the policy run stays
    bit-identical to the verified non-demo configuration (with render_robot
    enabled, the robot's own legs enter the ego frame and stance-pinning was
    measurably perturbing the learned stack's decisions)."""
    pos, quat = _current_pose(build)
    yaw = _yaw_from_quat_wxyz(quat)
    demo["poses"].append((np.asarray(pos, dtype=np.float32).copy(), float(yaw)))
    demo["statuses"].append(status)


def _render_demo_frames(pack, demo, args, registry, safety, locomotion_policy):
    """Pass 2: rebuild the scene robot-visible. Tour = stance replay of recorded
    poses under a raised follow camera. Seek = GAIT REPLAY: the recorded
    primitive sequence executes through the real PPO locomotion policy (actual
    stepping); the base is snapped back to the verified route only when drift
    exceeds a gate (0.22 m / 0.6 rad), so the gait reads naturally on camera
    while still being unable to wander off the verified path."""
    build = build_scene_from_pack(pack, n_envs=1, backend=args.backend,
                                  show_viewer=False, render_robot=True,
                                  apply_textures=bool(args.apply_textures))
    leg_idx = _resolve_rollout_leg_dof_indices(build.robot, RolloutConfig().leg_dof_indices)
    device = torch.device("cpu")
    frames, statuses = [], []
    # --- tour: stance pose replay (privileged teleport phase) ---
    for (pos, yaw), status in zip(demo["poses"], demo["statuses"]):
        quat = _quat_wxyz_from_yaw(yaw)
        build.robot.set_pos(pos[None, :], envs_idx=[0], zero_velocity=True)
        build.robot.set_quat(quat[None, :], envs_idx=[0], zero_velocity=False)
        build.robot.set_dofs_position(DEFAULT_GO2_STANCE_RAD[None, :], leg_idx.tolist(), envs_idx=[0])
        build.robot.set_dofs_velocity(np.zeros((1, len(leg_idx)), dtype=np.float32), leg_idx.tolist(), envs_idx=[0])
        try:
            build.scene.step()
        except Exception:
            pass
        build.robot.set_pos(pos[None, :], envs_idx=[0], zero_velocity=True)
        build.robot.set_quat(quat[None, :], envs_idx=[0], zero_velocity=False)
        third = _render_tour_view(build, pos, yaw)
        ego = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
        ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append((third, ego_np, float(pos[0]), float(pos[1]), float(yaw)))
        statuses.append(status)
    # --- seek: RL-gait replay along the verified route ---
    if demo["seek_blocks"] and locomotion_policy is not None:
        runner = RolloutRunner(build, locomotion_policy, registry, safety, config=RolloutConfig(
            n_blocks=len(demo["seek_blocks"]) + 8, fall_z_threshold_m=0.05,
            rgb_capture_per_block=False, seed=int(args.seed),
            log_progress_every_blocks=0, foot_contact_source="zero",
            randomize_spawn_pose=False))
        start_pos, start_quat = demo["seek_start"]
        _set_pose(build=build, runner=runner, pos_xyz=start_pos, quat_wxyz=start_quat)
        scan_tick = n_snaps = 0
        for primitive, (target_pos, target_yaw), status in demo["seek_blocks"]:
            sink: list = []
            _execute_physical_primitive(runner, registry, primitive,
                                        frame_sink=sink, build=build, pack=pack, device=device)
            is_scan = status["state"].startswith(("ALIGN", "REPLAN"))
            scan_tick += 1 if is_scan else 0
            if is_scan:
                keep = sink[-1:] if scan_tick % 3 == 0 else []   # scans at ~3x
            else:
                keep = sink[::2]                                  # walking at half-tick rate
            for (t3, eg, fx, fy, fyaw) in keep:
                frames.append((t3, eg, fx, fy, fyaw))
                statuses.append(status)
            # Drift-gated correction to the verified route: only snap when the
            # gait has wandered enough to matter — the unconditional per-block
            # snap read as twitching/clipping on camera.
            cur_pos, cur_quat = _current_pose(build)
            drift_m = float(np.linalg.norm(np.asarray(cur_pos[:2], dtype=np.float64)
                                           - np.asarray(target_pos[:2], dtype=np.float64)))
            dyaw = abs(wrap_angle_pi(_yaw_from_quat_wxyz(cur_quat) - float(target_yaw)))
            if drift_m > 0.22 or dyaw > 0.6:
                n_snaps += 1
                build.robot.set_pos(target_pos[None, :], envs_idx=[0], zero_velocity=False)
                build.robot.set_quat(_quat_wxyz_from_yaw(target_yaw)[None, :], envs_idx=[0], zero_velocity=False)
        print(f"    [replay] route-correction snaps: {n_snaps}/{len(demo['seek_blocks'])} blocks", flush=True)
    return frames, statuses


_GOAL_COLORS = {
    "red": (225, 70, 70), "blue": (70, 130, 255),
    "green": (80, 205, 120), "yellow": (235, 200, 70),
}


def _write_topo_demo_video(path, pack, frames, statuses, goals_viz, fps, title, waypoint_thumbs=None):
    """HUD video in the repo's established demo format (title bar, third-person
    + robot-eye panels, minimap with trail) + topo-nav extras: the goal-image
    inset (the actual task input), phase/memory status, perceptual-stop banner.

    goals_viz: list of {xy, beacon_xy, name, image_np} — one entry per seek
    goal (multi-beacon demos chase several in sequence)."""
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    from benchmark_lewm_closed_loop_mpc import _draw_minimap

    manifest = pack.scene_graph.manifest
    (xlo, ylo), (xhi, yhi) = manifest.world_bounds_xy_m
    bounds = ((float(xlo), float(ylo)), (float(xhi), float(yhi)))
    raw = InflatedOccupancyGrid(manifest, cell_size_m=0.05, inflation_m=0.0)
    rn = 76
    occ = np.zeros((rn, rn), dtype=bool)
    for j in range(rn):
        for i in range(rn):
            x = xlo + (i + 0.5) / rn * (xhi - xlo)
            y = ylo + (j + 0.5) / rn * (yhi - ylo)
            occ[j, i] = not raw.is_free((float(x), float(y)))

    def _font(sz, bold=False):
        try:
            fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf"
            return ImageFont.truetype(fp, sz)
        except Exception:
            return ImageFont.load_default()

    waypoint_thumbs = waypoint_thumbs or {}
    f_title, f_lab, f_stat = _font(19, True), _font(14), _font(13)
    W, H = 896, 496
    out, trail = [], []
    goals_viz = goals_viz or []
    goal_thumbs = [Image.fromarray(g["image_np"]).resize((128, 128)) for g in goals_viz]
    goal_cols = [_GOAL_COLORS.get(g["name"], (235, 200, 70)) for g in goals_viz]
    multi = len(goals_viz) > 1
    for (third, ego, rx, ry, yaw), status in zip(frames, statuses):
        trail.append((rx, ry))
        canvas = Image.new("RGB", (W, H), (16, 16, 20))
        canvas.paste(Image.fromarray(third).resize((416, 416)), (12, 44))
        canvas.paste(Image.fromarray(ego).resize((300, 300)), (456, 44))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, W - 1, 36], fill=(10, 10, 13), outline=(70, 70, 80))
        draw.text((14, 9), title, fill=(0, 235, 120), font=f_title)
        draw.text((12, 462), "Third-person follow", fill=(195, 195, 200), font=f_lab)
        draw.text((456, 346), "Robot-eye (perception)", fill=(195, 195, 200), font=f_lab)
        gi = status.get("goal_idx")
        beacons, claimed, target_idx = [], set(), None
        if status["phase"] == "SEEK" and goals_viz:
            beacons = [(np.asarray(g.get("beacon_xy") or g["xy"], dtype=np.float64),
                        goal_cols[k], g["name"]) for k, g in enumerate(goals_viz)]
            claimed = set(range(int(status.get("claimed_n", 0))))
            if status.get("stopped") and gi is not None:
                claimed.add(gi)
            if gi is not None and gi not in claimed:
                target_idx = gi
        _draw_minimap(draw, bounds, occ, beacons, trail, (rx, ry), yaw,
                      target_idx, claimed, 456, 372, 300, 106)
        draw.text((766, 372), "Map", fill=(195, 195, 200), font=f_lab)
        if status["phase"] == "SEEK" and gi is not None and gi < len(goal_thumbs):
            canvas.paste(goal_thumbs[gi], (762, 44))
            draw.rectangle([761, 43, 891, 173], outline=goal_cols[gi])
            lab = f"Goal image ({goals_viz[gi]['name']})" if multi else "Goal image (input)"
            draw.text((762, 178), lab, fill=goal_cols[gi], font=f_stat)
        phase_col = (120, 180, 255) if status["phase"] == "TOUR" else (235, 200, 70)
        draw.text((766, 396), f"phase: {status['phase']}", fill=phase_col, font=f_stat)
        draw.text((766, 416), f"memory: {status['nodes']} nodes", fill=(200, 200, 205), font=f_stat)
        if status["phase"] == "SEEK":
            if multi:
                claimed_n = int(status.get("claimed_n", 0)) + (1 if status.get("stopped") else 0)
                draw.text((766, 436), f"goals: {claimed_n}/{len(goals_viz)}  d={status['dist']:.1f} m",
                          fill=(200, 200, 205), font=f_stat)
            else:
                draw.text((766, 436), f"goal dist: {status['dist']:.1f} m", fill=(200, 200, 205), font=f_stat)
        if status.get("state"):
            draw.rectangle([12, 44, 428, 66], fill=(10, 10, 13))
            draw.text((18, 48), status["state"], fill=(140, 220, 255), font=f_lab)
        cap = ("(mapping sweep: teleport replay - builds the topological memory)"
               if status["phase"] == "TOUR"
               else "(RL gait; drift-corrected to the verified route)")
        draw.text((140, 462), cap, fill=(120, 120, 130), font=f_stat)
        wp_ref = status.get("waypoint_ref")
        if wp_ref is not None and wp_ref in waypoint_thumbs and status["phase"] == "SEEK":
            canvas.paste(Image.fromarray(waypoint_thumbs[wp_ref]).resize((128, 128)), (762, 200))
            draw.rectangle([761, 199, 891, 329], outline=(140, 220, 255))
            draw.text((762, 334), "Current waypoint view", fill=(140, 220, 255), font=f_stat)
        if status.get("stopped"):
            name = str(status.get("goal_name", "goal"))
            txt = (f"PERCEPTUAL STOP - {name.upper()} REACHED" if multi
                   else "PERCEPTUAL STOP - GOAL REACHED")
            draw.rectangle([12, 200, 428, 260], fill=(10, 40, 16), outline=(70, 230, 120), width=2)
            draw.text((30, 215), txt, fill=(70, 230, 120), font=f_title)
        out.append(np.asarray(canvas))
    hold = [out[-1]] * max(1, int(round(fps * 2)))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), out + hold, fps=fps, macro_block_size=8)


def run_scene(pack, build, model, encoder, scorer, tau_new, registry, sequences, action_tensor,
              primitive_names, args, rng, device, demo=None, runner=None, local_obstacles=None):
    graph = pack.scene_graph
    # Route construction and kinematic simulation remain privileged benchmark
    # infrastructure. Runtime seek safety consumes only LocalObstacleModel.
    route_grid = InflatedOccupancyGrid(graph.manifest, cell_size_m=0.05, inflation_m=0.20)
    if local_obstacles is None:
        if getattr(args, "local_obstacle_source", "privileged-grid") == "ego-depth":
            local_obstacles = DepthLocalObstacleModel(
                DepthLocalObstacleConfig(
                    vertical_fov_deg=float(pack.camera.fov_deg),
                    max_range_m=float(args.depth_obstacle_max_range_m),
                    sample_stride_px=int(args.depth_obstacle_stride_px),
                    occupied_inflation_m=float(args.depth_obstacle_inflation_m),
                    max_age_frames=int(args.depth_obstacle_max_age_frames),
                    unknown_is_free=bool(args.depth_obstacle_unknown_is_free),
                ),
                odometry_detail=(
                    "simulator ground-truth pose; depth is simulator-rendered "
                    "and absent from the current mono-RGB platform contract"
                ),
                odometry_deployment_valid=False,
            )
        else:
            local_obstacles = PrivilegedGridObstacleModel(route_grid)
    if getattr(args, "require_deployment_valid_local_obstacles", False):
        require_deployment_valid_local_obstacles(local_obstacles)
    print(
        "  runtime local obstacles: "
        f"{local_obstacles.contract.source} "
        f"(deployment_valid={local_obstacles.contract.deployment_valid})",
        flush=True,
    )
    base_z = float(pack.robot.spawn_xyz_m[2])
    spawn_xy = np.asarray(pack.robot.spawn_xyz_m[:2], dtype=np.float64)
    cells = [n.node_id for n in graph.manifest.graph_nodes]
    start_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), spawn_xy))

    navigator = TopologicalNavigator(
        encoder, scorer, history=args.history, tau_new=tau_new,
        tau_goal=args.tau_goal, subgoal_lookahead=args.lookahead,
    )
    stored_frames: list[torch.Tensor] = []
    tour_cell_heading: dict[int, float] = {}

    def observe():
        pos, quat = _current_pose(build)
        image = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
        z_raw, _ = model.encode(image[None], None)
        stored_frames.append(image)
        nearest = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), pos[:2]))
        yaw = _yaw_from_quat_wxyz(quat)
        prev_map = navigator._last_map
        map_node = navigator.update((z_raw.squeeze(0), len(stored_frames) - 1), label=int(nearest))
        if prev_map is not None and map_node is not None and prev_map != map_node:
            # The mapping tour feeds only translational poses; its IMU yaw is
            # therefore the measured forward bearing for this directed memory
            # edge. Reusing it during seek avoids a full visual revolution per
            # edge (v38 still spent hundreds of blocks rescanning an otherwise
            # forward-only route).
            navigator._edge_bearings.setdefault((int(prev_map), int(map_node)), float(yaw))
        tour_cell_heading[nearest] = yaw

    # ---------------- Tour phase (privileged motion, learned perception) ----
    tour = graph_tour_cells(graph, start_cell, args.tour_max_cells, rng,
                            reprise_cells=int(args.tour_reprise_cells))
    lookats: dict[int, tuple[float, np.ndarray]] = {}
    if args.tour_landmark_lookat:
        tour = _augment_tour_with_landmark_anchor(
            graph, tour, route_grid, goal_standoff_m=float(args.goal_standoff_m),
            goal_min_hops=int(args.goal_min_hops),
            goal_max_hops=max(int(args.goal_max_hops) + 2, 7),
            reprise_cells=int(args.tour_reprise_cells))
        tour_set = set(int(c) for c in tour)
        for _obj, lcell in graph.landmark_cells:
            b = np.asarray(graph.landmark_xy_for_cell(lcell) or graph.cell_center(lcell),
                           dtype=np.float64)
            dists = {c: _xy_distance(graph.cell_center(c), b) for c in tour_set}
            cands = []
            for c, distance in dists.items():
                if not (0.55 <= distance <= max(1.60, min(dists.values()) + 0.25)):
                    continue
                cx = np.asarray(graph.cell_center(c), dtype=np.float64)
                standoff = b + (cx - b) / max(distance, 1e-6) * float(args.goal_standoff_m)
                if _segment_is_traversable(route_grid, cx, standoff):
                    cands.append(c)
            if not cands:
                continue
            c = min(cands, key=lambda c: dists[c])
            cx = np.asarray(graph.cell_center(c), dtype=np.float64)
            bearing = math.atan2(b[1] - cx[1], b[0] - cx[0])
            gap = max(_xy_distance(cx, b), 1e-6)
            standoff = b + (cx - b) / gap * float(args.goal_standoff_m)
            lookats[int(c)] = (bearing, standoff)
        if lookats:
            print(f"  tour landmark look-ats at cells {sorted(lookats)}", flush=True)
    poses = tour_pose_sequence(graph, tour, base_z, step_m=float(args.tour_step_m),
                               max_steps=args.tour_max_blocks,
                               lookats=lookats or None)
    tour_blocks = 0
    n_spurs = 0
    for pos_xyz, yaw, kind in poses:
        _set_pose(build=build, runner=None, pos_xyz=pos_xyz, quat_wxyz=_quat_wxyz_from_yaw(yaw))
        if kind == "walk":      # turn/spur poses are video-only (see tour_pose_sequence)
            observe()
        elif kind == "spur" and navigator._last_map is not None:
            # Landmark-facing view -> terminal goal-only spur node anchored at
            # the current chain node (see TopologicalNavigator.insert_spur).
            pos, quat = _current_pose(build)
            image = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
            z_raw, _ = model.encode(image[None], None)
            stored_frames.append(image)
            nearest = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), pos[:2]))
            # A standoff can be metrically nearer a cell across the beacon or
            # wall, but its graph attachment and goal contract are the anchor
            # cell represented by the current chain node.
            anchor_label = navigator.memory.node_majority_labels().get(
                navigator._last_map, (int(nearest), 0, False))[0]
            navigator.insert_spur(z_raw.squeeze(0), len(stored_frames) - 1,
                                  navigator._last_map, label=int(anchor_label),
                                  bearing_rad=_yaw_from_quat_wxyz(quat))
            n_spurs += 1
        tour_blocks += 1
        if demo is not None and tour_blocks % max(int(args.tour_capture_every), 1) == 0:
            _demo_capture(build, pack, device, demo,
                          {"phase": "TOUR", "state": "MAPPING SWEEP: driven tour builds the topological memory",
                           "nodes": len(navigator.memory.nodes)})
            _set_pose(build=build, runner=None, pos_xyz=pos_xyz, quat_wxyz=_quat_wxyz_from_yaw(yaw))
    n_nodes = len(navigator.memory.nodes)
    visited_cells = sorted(tour_cell_heading)
    print(f"  tour: {tour_blocks} blocks, {len(visited_cells)} cells seen, "
          f"{n_nodes} memory nodes ({n_spurs} landmark spurs, "
          f"{len(navigator._edge_bearings)} directed bearings)", flush=True)

    # ---------------- Goal selection (setup-time privilege; eval contract) --
    end_pos, _ = _current_pose(build)
    end_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), end_pos[:2]))

    def render_goal(cell, yaw):
        xy = graph.cell_center(cell)
        image = _render_tensor_from_base(
            build, pack, base_xyz_m=np.asarray([xy[0], xy[1], base_z], dtype=np.float32),
            base_quat_wxyz=_quat_wxyz_from_yaw(yaw), device=device,
        )
        return xy, image

    def render_goal_pose(xy, yaw):
        image = _render_tensor_from_base(
            build, pack, base_xyz_m=np.asarray([xy[0], xy[1], base_z], dtype=np.float32),
            base_quat_wxyz=_quat_wxyz_from_yaw(yaw), device=device,
        )
        return np.asarray(xy, dtype=np.float64), image

    def goal_node_allowlist(goal_sig=None, *, terminal_only: bool = False):
        """Deployment-valid goal-node gate shared by setup diagnostics and seek."""
        if not terminal_only and (goal_sig is None or goal_sig[1] is None):
            return None
        return {
            node_id for node_id in navigator._keyframes
            if (not terminal_only or not navigator.memory.nodes[node_id].in_filter)
            and ((goal_sig is None or goal_sig[1] is None)
                 or ((idx := navigator._observations.get(node_id)) is not None
                     and _colour_match(goal_sig, _colour_signature(stored_frames[idx]))))
        }

    def route_diag(z_goal, goal_sig=None, *, terminal_only: bool = False):
        """Reversed-edge count on the planned route from the CURRENT
        localization (tour end = seek start) to the goal match. Setup-time
        privilege used only to PICK demo goals honoring the forward-mostly
        rule (reversed-edge arrival never confirms: the robot faces away from
        every stored keyframe walking an edge backward). A colourful goal also
        requires the PLANNED goal node's stored view to colour-match: beacon
        close-ups of different colours are latent-identical (~1.0 cosine), so
        the latent matcher can route to the wrong beacon."""
        allowed_goal_nodes = goal_node_allowlist(goal_sig, terminal_only=terminal_only)
        plan = navigator.plan_node_path(z_goal, allowed_goal_nodes=allowed_goal_nodes)
        if plan is None:
            return None, None
        path, gnode, _score = plan
        if goal_sig is not None and goal_sig[1] is not None:
            idx = navigator._observations.get(gnode)
            if idx is not None and not _colour_match(goal_sig, _colour_signature(stored_frames[idx])):
                return None, None
        n_rev = sum(1 for i in range(1, len(path))
                    if (path[i - 1], path[i]) not in navigator.memory.edges)
        return n_rev, len(path)

    def forward_mostly(n_rev, path_len):
        if n_rev is None:
            return False
        # Demo seeks must be fully forward-routable. Reversed edges align to a
        # keyframe seen from the opposite travel direction and repeatedly fail
        # to confirm arrival (v29: the first reversed edge consumed 200 blocks
        # without leaving the start). Keep the looser benchmark contract for
        # non-demo evaluation, where measuring that failure remains useful.
        if args.demo_require_colourful and runner is not None:
            return n_rev == 0
        return n_rev <= max(1, (path_len - 1) // 3)

    # Median adjacent-cell spacing: the metric yardstick for "far" below.
    _spacings = [_xy_distance(graph.cell_center(n.node_id), graph.cell_center(nb))
                 for n in graph.manifest.graph_nodes for nb in graph.neighbors(n.node_id)]
    cell_pitch_m = float(np.median(_spacings)) if _spacings else 1.0

    def aliased_far_cosine(z_goal, goal_cell, *, min_dist_factor: float = 1.5, goal_sig=None,
                           return_worst: bool = False):
        """Max cosine between the goal latent and any memory keyframe whose
        majority cell is METRICALLY far (>= min_dist_factor cell pitches) from
        the goal cell. If this crosses ~tau_arrive the perceptual stop WILL
        false-fire on that distant view (measured: bland corridor goals stopped
        2.4-3.9 m out), so goal selection rejects such candidates up front.
        Far-ness is Euclidean, NOT graph hops: a landmark standoff frame gets
        labeled with the Euclidean-nearest cell, which can sit across a wall
        many graph-hops from the stand cell — graph hops then count the goal's
        OWN look-at view as far self-aliasing (v17: identical frame, cos 1.000,
        3 'hops' away). Under the colour-gated stop a far keyframe only counts
        as aliasing if its stored view could also pass the colour gate
        (different-colour beacon close-ups are latent-identical but cannot
        false-fire)."""
        labels = navigator.memory.node_majority_labels()
        z = F.normalize(z_goal, dim=-1)
        gxy = np.asarray(graph.cell_center(int(goal_cell)), dtype=np.float64)
        min_dist_m = float(min_dist_factor) * cell_pitch_m
        worst, worst_info = 0.0, None
        for node_id, keyframe in navigator._keyframes.items():
            lab = labels.get(node_id)
            if lab is None:
                continue
            dist_m = _xy_distance(graph.cell_center(int(lab[0])), gxy)
            if dist_m < min_dist_m:
                continue
            sig = None
            if goal_sig is not None and goal_sig[1] is not None:
                idx = navigator._observations.get(node_id)
                if idx is not None:
                    sig = _colour_signature(stored_frames[idx])
                    if not _colour_match(goal_sig, sig):
                        continue
            cos = float(keyframe @ z)
            if cos > worst:
                worst, worst_info = cos, (node_id, int(lab[0]), round(dist_m, 2), sig)
        if return_worst:
            return worst, worst_info
        return worst

    goals = []
    if args.goal_mode == "beacons":
        # Multi-goal demo: seek each coloured landmark in turn. The goal image
        # is rendered from a tour-visited cell near the beacon, facing it (the
        # standoff with the most saturated view = beacon clearly in frame) —
        # the "photo of the place" contract with a viewer-legible target.
        landmarks = [(str(object_id),
                      np.asarray(graph.landmark_xy_for_cell(cell_id) or graph.cell_center(cell_id), dtype=np.float64))
                     for object_id, cell_id in graph.landmark_cells]
        cur_xy = np.asarray(end_pos[:2], dtype=np.float64)
        ordered, remaining = [], landmarks[:]
        while remaining:  # farthest-first chaining: show real traversal legs
            remaining.sort(key=lambda b: -float(np.linalg.norm(b[1] - cur_xy)))
            ordered.append(remaining.pop(0))
            cur_xy = ordered[-1][1]
        for name, bxy in ordered[: max(int(args.goal_beacons), 1)]:
            # Stand cells: the few NEAREST tour-visited cells to the beacon
            # (the 0.55 m floor rejects the beacon's own cell when the landmark
            # sits at a cell center). Among them prefer a goal image the memory
            # actually recognizes (match >= tau_goal, else the planner can
            # never route to it), tie-broken by colour saturation (beacon
            # clearly in frame).
            stand = sorted((c for c in visited_cells
                            if _xy_distance(graph.cell_center(c), bxy) >= 0.55),
                           key=lambda c: _xy_distance(graph.cell_center(c), bxy))[:4]
            if not stand:
                continue
            scored = []
            for c in stand:
                cx = graph.cell_center(c)
                yaw_c = math.atan2(bxy[1] - cx[1], bxy[0] - cx[0])
                xy_c, img = render_goal(c, yaw_c)
                z_raw, _ = model.encode(img[None], None)
                _node, match = navigator.match_goal(z_raw.squeeze(0))
                scored.append((_saturation_score(img), float(match), c, yaw_c, xy_c, img))
            passing = [s for s in scored if s[1] >= float(args.tau_goal)]
            sat, match, c, yaw_c, xy_c, img = (max(passing, key=lambda s: s[0]) if passing
                                               else max(scored, key=lambda s: s[1]))
            goals.append({"cell": int(c), "xy": xy_c, "yaw": yaw_c, "image": img,
                          "name": name.split("landmark_")[-1], "beacon_xy": bxy,
                          "saturation": sat, "terminal_only": False})
        if len(goals) < 2:
            print("  [beacons] fewer than 2 reachable beacon goals - skipping trial", flush=True)
            return None
    else:
        terminal_only = False
        goal_candidates = [c for c in visited_cells
                           if (d := graph.bfs_distance(end_cell, c)) is not None
                           and args.goal_min_hops <= d <= args.goal_max_hops]
        if not goal_candidates:
            return None
        if args.goal_mode == "colourful":
            # Measured (probe_goal_saturation.py): colour is ONLY visible from
            # cells directly adjacent to a landmark while facing it (sat ~0.42
            # at 0.8 m); every tour-heading view in this corpus is bland. So
            # the legible goal = landmark-FACING image (the registered
            # goal-facing convention) at a landmark-adjacent tour-visited cell,
            # >= goal_min_hops from the start for a real traversal and
            # memory-matched >= tau_goal so the planner can route. Fall back to
            # the plain tour-heading contract if nothing qualifies.
            # Stand specs (cell, facing yaw, standoff point). When the tour did
            # landmark look-ats, score EXACTLY those standoffs: they are the
            # views now in memory, so re-deriving the geometry here can (and in
            # v13 did) disagree on the stand-distance cap and score zero stand
            # candidates while the look-at nodes sat unused.
            if lookats:
                stand_specs = [(c, yaw_c, standoff) for c, (yaw_c, standoff) in lookats.items()]
            else:
                landmark_pts = [np.asarray(graph.landmark_xy_for_cell(cid) or graph.cell_center(cid), dtype=np.float64)
                                for _o, cid in graph.landmark_cells]
                # Sparse-cell families (e.g. composite motifs) may have no visited
                # cell within the tight maze stand range; allow up to the nearest
                # visited cell per landmark (floor stays at 0.55 m).
                stand_cap = 1.3
                if landmark_pts:
                    nearest = [min(_xy_distance(graph.cell_center(c), b) for c in visited_cells)
                               for b in landmark_pts]
                    stand_cap = max(1.3, min(n for n in nearest) + 0.25)
                stand_specs = []
                for c in visited_cells:
                    cx = np.asarray(graph.cell_center(c), dtype=np.float64)
                    b = min(landmark_pts, key=lambda p: _xy_distance(cx, p)) if landmark_pts else None
                    if b is None or not (0.55 <= _xy_distance(cx, b) <= stand_cap):
                        continue
                    yaw_c = math.atan2(b[1] - cx[1], b[0] - cx[0])
                    # Goal image from a CLOSE standoff facing the landmark, not
                    # the cell center: wide-maze cell centers sit ~1.2 m out,
                    # beyond the measured ~0.8 m colour-visibility range (stand
                    # views came back sat 0.000 and aliased 0.97+ like every
                    # bland corridor). The standoff view is the salient-object
                    # regime where plan_cost servoing and the perceptual stop
                    # are validated (beacons 0.92).
                    gap = max(_xy_distance(cx, b), 1e-6)
                    stand_specs.append((c, yaw_c, b + (cx - b) / gap * float(args.goal_standoff_m)))
            scored = []
            for c, yaw_c, standoff in stand_specs:
                if c not in visited_cells:
                    continue
                hops = graph.bfs_distance(end_cell, c)
                # Cap the hop count too: long routes are all-reversed chains
                # (DFS memory is a time-directed graph) and reliably fail.
                if hops is None or not (int(args.goal_min_hops) <= hops <= max(int(args.goal_max_hops) + 2, 7)):
                    continue
                xy_c, img = render_goal_pose(standoff, yaw_c)
                z_raw, _ = model.encode(img[None], None)
                sig = _colour_signature(img)
                _n, match = navigator.match_goal(z_raw.squeeze(0))
                n_rev, plen = route_diag(z_raw.squeeze(0), goal_sig=sig, terminal_only=True)
                alias, worst = aliased_far_cosine(z_raw.squeeze(0), c, goal_sig=sig,
                                                  return_worst=True)
                if worst is not None:
                    wnode, wlab, whops, wsig = worst
                    def _sigtxt(s):
                        if s is None:
                            return "sig n/a"
                        if s[1] is None:
                            return f"frac {s[0]:.3f} rgb none"
                        return (f"frac {s[0]:.3f} rgb (" +
                                ",".join(f"{v:.2f}" for v in s[1].tolist()) + ")")
                    print(f"    [goal-cand-alias] stand cell {c}: goal {_sigtxt(sig)} | "
                          f"worst node {wnode} label {wlab} dist {whops} m cos {alias:.3f} "
                          f"{_sigtxt(wsig)}", flush=True)
                scored.append((_saturation_score(img), float(match), c, yaw_c, xy_c, img,
                               int(hops), n_rev, plen, alias, True))
            # Also score the standard tour-heading candidates: in open
            # families the landmark pillars are visible from afar, so a
            # forward-feasible tour-heading goal can be colourful too.
            scored_th = []
            for c in goal_candidates:
                xy_c, img = render_goal(c, tour_cell_heading[c])
                z_raw, _ = model.encode(img[None], None)
                _n, match = navigator.match_goal(z_raw.squeeze(0))
                n_rev, plen = route_diag(z_raw.squeeze(0))
                alias = aliased_far_cosine(z_raw.squeeze(0), c)
                scored_th.append((_saturation_score(img), float(match), c, tour_cell_heading[c],
                                  xy_c, img, int(graph.bfs_distance(end_cell, c) or 99), n_rev, plen,
                                  alias, False))
            alias_cap = float(args.tau_arrive) - 0.02
            tagged = [("stand", s) for s in scored] + [("tour-heading", s) for s in scored_th]
            for kind, s in sorted(tagged, key=lambda t: t[1][6]):
                rev = f"rev {s[7]}/{s[8] - 1}" if s[7] is not None else "unplannable"
                print(f"    [goal-cand] {kind:12s} cell {s[2]:3d} hops {s[6]} "
                      f"sat {s[0]:.3f} match {s[1]:.3f} alias {s[9]:.3f} {rev}", flush=True)
            passing_stand = [s for s in scored
                             if s[1] >= float(args.tau_goal) and s[0] >= 0.05
                             and forward_mostly(s[7], s[8]) and s[9] < alias_cap]
            passing_heading = [s for s in scored_th
                               if s[1] >= float(args.tau_goal) and s[0] >= 0.05
                               and forward_mostly(s[7], s[8]) and s[9] < alias_cap]
            # A route-valid terminal stand is the intended colourful-goal
            # contract. Tour-heading views remain a fallback for families where
            # no valid standoff exists; mixing both pools let a shorter ordinary
            # keyframe beat the valid terminal spur in v40.
            passing = passing_stand or passing_heading
            if passing:
                # Within the selected contract, prefer the fewest reversed
                # edges, then the shortest actual memory route.
                sat, match, goal_cell, goal_yaw_sel, goal_xy_sel, goal_img_sel, _h, _r, _l, _a, terminal_only = min(
                    passing, key=lambda s: (s[7], s[8], s[6]))
                print(f"  goal saturation: {sat:.3f} (match {match:.3f}, "
                      f"{len(passing)}/{len(scored) + len(scored_th)} candidates "
                      f"colourful+forward, {'terminal stand' if terminal_only else 'tour heading'})",
                      flush=True)
            else:
                if args.demo_require_colourful:
                    print(f"  [colourful] no colourful+forward goal this draw "
                          f"({len(scored) + len(scored_th)} candidates) - skipping trial", flush=True)
                    return None
                # Fall back to the bland tour-heading contract, but still honor
                # the forward-mostly rule when any candidate routes forward.
                plannable_th = [s for s in scored_th
                                if s[1] >= float(args.tau_goal) and forward_mostly(s[7], s[8])
                                and s[9] < alias_cap]
                if plannable_th:
                    best_rev = min(s[7] for s in plannable_th)
                    pool = [s for s in plannable_th if s[7] == best_rev]
                    _s, match, goal_cell, goal_yaw_sel, goal_xy_sel, goal_img_sel, _h, _r, _l, _a, terminal_only = \
                        pool[rng.randrange(len(pool))]
                    sat = None
                    print(f"  [colourful] no colourful+forward goal - tour-heading fallback "
                          f"cell {goal_cell} (match {match:.3f}, {best_rev} reversed edges, "
                          f"{len(pool)} forward+unaliased candidates)", flush=True)
                else:
                    if args.demo_require_colourful:
                        # Demo contract: the goal must BE a beacon photo. A
                        # bland fallback seek can "succeed" and steal the demo
                        # video from the actual task (v20 trial 2).
                        print(f"  [colourful] no colourful+forward goal this draw "
                              f"({len(scored) + len(scored_th)} candidates) - skipping trial",
                              flush=True)
                        return None
                    print(f"  [colourful] no forward-routable goal at all "
                          f"({len(scored) + len(scored_th)} candidates) - random tour-heading fallback",
                          flush=True)
                    goal_cell = rng.choice(goal_candidates)
                    goal_xy_sel, goal_img_sel = render_goal(goal_cell, tour_cell_heading[goal_cell])
                    goal_yaw_sel, sat = tour_cell_heading[goal_cell], None
        else:
            goal_cell = rng.choice(goal_candidates)
            goal_xy_sel, goal_img_sel = render_goal(goal_cell, tour_cell_heading[goal_cell])
            goal_yaw_sel, sat = tour_cell_heading[goal_cell], None
        goals = [{"cell": int(goal_cell), "xy": goal_xy_sel, "yaw": goal_yaw_sel,
                  "image": goal_img_sel, "name": "goal", "beacon_xy": None, "saturation": sat,
                  "terminal_only": terminal_only}]

    for g in goals:
        z_raw, _ = model.encode(g["image"][None], None)
        g["z"] = z_raw.squeeze(0)
        g["image_np"] = g["image"].mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        g["colour_sig"] = _colour_signature(g["image"])
        allowed_goal_nodes = goal_node_allowlist(
            g.get("colour_sig"), terminal_only=g.get("terminal_only", False))
        plan = navigator.plan_node_path(g["z"], allowed_goal_nodes=allowed_goal_nodes)
        if plan is not None:
            path, g["match_node"], g["match_score"] = plan
            g["planned_route_nodes"] = list(path)
        else:
            if g.get("terminal_only", False):
                print("  [colourful] selected terminal stand has no constrained route - "
                      "skipping trial", flush=True)
                return None
            g["match_node"], g["match_score"] = navigator.match_goal(g["z"])
            g["planned_route_nodes"] = None
        # Goal-conditioned arrival threshold: arrived = current view matches the
        # goal better than ANY far-away keyframe ever could (alias + margin).
        # Deployment-valid (alias comes from the robot's own memory + the goal
        # image at seek start). The fixed 0.90 bar misses gait arrivals — the
        # walking camera's pitch/height caps the cosine below the kinematic
        # value (measured: 0.37 m final, stop never fired).
        g["alias"] = aliased_far_cosine(g["z"], g["cell"], goal_sig=g["colour_sig"])
        g["tau_arrive_eff"] = min(float(args.tau_arrive),
                                  max(float(args.tau_arrive_floor), g["alias"] + 0.02))

    majority = navigator.memory.node_majority_labels()
    start_pose = (np.asarray([*graph.cell_center(end_cell), base_z], dtype=np.float32),
                  _quat_wxyz_from_yaw(rng.uniform(-math.pi, math.pi)))
    if demo is not None:
        demo["goals"] = [{"xy": (float(g["xy"][0]), float(g["xy"][1])),
                          "beacon_xy": (tuple(map(float, g["beacon_xy"])) if g["beacon_xy"] is not None else None),
                          "name": g["name"], "image_np": g["image_np"]} for g in goals]
    for gi, g in enumerate(goals):
        print(f"  goal[{gi}] '{g['name']}': cell {g['cell']}, "
              f"{graph.bfs_distance(end_cell, g['cell'])} hops from tour end; "
              f"matched node {g['match_node']} (score {g['match_score']:.3f}, "
              f"alias {g['alias']:.3f}, tau_arrive_eff {g['tau_arrive_eff']:.3f}, "
              f"{'terminal stand' if g.get('terminal_only', False) else 'tour heading'})",
              flush=True)

    # ---------------- Seek phase ----------------
    # Map is complete: freeze the memory to localization-only. Mid-seek node
    # commits are posterior black holes (v24 trace: commit resets the
    # posterior to {new:1.0}, the running mean absorbs every following gait
    # frame, and the MAP never escapes while the robot paces in place).
    navigator.memory.frozen = True
    results = {}
    n_scan = blocks_per_revolution(registry, args.command_dt_s)
    # Gait-aware veto (physical only; kinematic keeps the verified base-center
    # single-block semantics): capsule body probes + a continuation horizon for
    # walk gating, capsule-only for the per-block local servo.
    if runner is not None:
        veto_walk_kw = {"horizon_blocks": int(args.veto_horizon_blocks),
                        "body_radius_m": float(args.veto_body_radius_m)}
        veto_servo_kw = {"body_radius_m": float(args.veto_body_radius_m)}
        # Padded clearance SCORE (not a gate): longer nose/tail capsule plus
        # side probes, so steering can prefer corridor-centered paths before
        # the strict veto ever fires.
        veto_pad_kw = {"horizon_blocks": int(args.veto_horizon_blocks),
                       "body_radius_m": float(args.veto_pad_radius_m),
                       "lateral_radius_m": float(args.veto_pad_lateral_m)}
    else:
        veto_walk_kw, veto_servo_kw, veto_pad_kw = {}, {}, {}
    for policy in _parse_csv(args.policies):
        _set_pose(build=build, runner=runner, pos_xyz=start_pose[0], quat_wxyz=start_pose[1])
        reset_local_obstacles = getattr(local_obstacles, "reset", None)
        if callable(reset_local_obstacles):
            reset_local_obstacles()
        if demo is not None and policy == "topo":
            demo["seek_start"] = (np.asarray(start_pose[0], dtype=np.float32).copy(), start_pose[1].copy())
        check_pos, _ = _current_pose(build)
        print(f"    [{policy}] start intended={start_pose[0][:2]} actual={check_pos[:2]}", flush=True)
        primitive_counts: dict[str, int] = {}
        falls, used_fallback = 0, 0
        path_length, prev_xy = 0.0, np.asarray(start_pose[0][:2], dtype=np.float64)
        goal_rows, claimed_n = [], 0
        for goal_index, goal in enumerate(goals):
            z_goal_raw, goal_image = goal["z"], goal["image"]
            goal_xy, goal_cell = goal["xy"], goal["cell"]
            tau_arrive_eff = float(goal.get("tau_arrive_eff", args.tau_arrive))
            cur0, _ = _current_pose(build)
            d0 = _xy_distance(cur0[:2], goal_xy)
            navigator._window.clear()       # fresh perceptual window; memory kept
            arrive_hits, subgoal_nodes, cosine_profile, colour_profile = [], [], [], []
            walk_bearing = None             # post-ALIGN IMU bearing held during WALK
            beh = {"walk_fwd": 0, "head_corr": 0, "veto_escape": 0, "align_blocks": 0,
                   "align_timeout": 0,
                   "stuck_detect": 0, "stuck_escape": 0, "edge_realign": 0, "scan_reuse": 0,
                   "head_err_sq": 0.0, "head_err_n": 0}
            # Proprioceptive wall-contact recovery state (physical mode): the
            # The local-obstacle veto is a model, and when it is wrong the v9 failure mode
            # was pushing forward against the wall until falling.
            stuck_streak, escape_blocks, edge_veto_streak, realigns_this_edge = 0, 0, 0, 0
            last_scan = None        # completed-revolution views for scan reuse
            dead_edges: set = set() # edges that failed traversal -> penalized at replan
            max_goal_cosine, subgoal_progress, subgoal_scored = 0.0, 0, 0
            approach_extra = 0      # bounded final-approach servo past first arrival hit
            arrival_armed = False   # perceptual terminal-goal completion latch
            final_approach_blocks = 0  # total spur-servo budget; never reset by re-align/replan
            subgoals_reached = 0
            scan_remaining, scan_costs, scan_return = 0, [], 0
            node_path, path_index, mode, walk_blocks = None, 0, "align", 0
            align_blocks_this_edge = 0
            align_target, align_op = 0.0, "ge"
            align_phase, scan_samples, scan_cum, turn_target, turn_cap = "scan", [], 0.0, 0.0, 0
            stopped_perceptually = False
            # Goal selection just validated a route from the tour-end MAP, and
            # the physical reset places the robot at that same tour-end cell.
            # Do not replace that valid start with a one-frame random-yaw
            # localization before the first plan (v37: MAP jumped to node 2,
            # which did not match the physical start, then 822/1100 blocks
            # were spent aligning/replanning around the wrong graph origin).
            initial_plan_pending = goal_index == 0
            for _block in range(args.seek_max_blocks):
                pos, quat = _current_pose(build)
                if isinstance(local_obstacles, DepthLocalObstacleModel):
                    image, depth, camera_pose = _render_perception_tensor_from_base(
                        build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
                    local_obstacles.update(
                        depth,
                        camera_position_xyz=camera_pose.position,
                        camera_rotation=camera_pose.rotation,
                        robot_xy=(float(pos[0]), float(pos[1])),
                    )
                else:
                    image = _render_tensor_from_base(
                        build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
                z_now, _ = model.encode(image[None], None)
                z_now = z_now.squeeze(0)
                # Perceptual arrival (deployment-valid; gt only logged for eval).
                cosine = float(F.normalize(z_now, dim=-1) @ F.normalize(z_goal_raw, dim=-1))
                max_goal_cosine = max(max_goal_cosine, cosine)
                # Arrival only counts while walking/servoing: an ALIGN scan
                # sweeps every yaw, so an aliased corridor view can cross
                # tau_arrive for the 2-block streak mid-revolution (measured
                # false stops 2.4-3.9 m out). The verified Stage-4b stop fired
                # during the final-approach servo, which this keeps intact.
                # Streak = 2 hits in the last 4 walk blocks (not 2 consecutive):
                # gait bobbing makes the cosine flutter around the threshold, so
                # a consecutive streak misses physically-arrived robots
                # (measured: 0.37 m final with no stop under gait; kinematic
                # twin stopped). Same evidence bar, robust to single dips.
                # ALIGN-scan gating is goal-conditioned. For an UNALIASED goal
                # (tau_arrive_eff cleared below the cap), no far keyframe can
                # reach tau_eff - 0.02 by construction, so a scan hit IS
                # arrival evidence — and the gait arrival's 0.993 peak happens
                # while aligning at the goal place, where same-place sibling
                # nodes keep path_index short of "final" (measured: 0.37 m
                # final, stop swallowed by the all-align gate twice). ALIASED
                # goals keep the full intermediate-scan gate (sweep views
                # false-stopped 2.4-4.1 m out at the fixed 0.90 bar).
                at_final_node = node_path is not None and path_index >= len(node_path) - 1
                unaliased_goal = tau_arrive_eff < float(args.tau_arrive) - 1e-9
                in_align = policy == "topo" and mode == "align" and node_path is not None
                goal_colour_sig = goal.get("colour_sig", (0.0, None))
                cur_colour_sig = (_colour_signature(image)
                                  if goal_colour_sig[1] is not None else None)
                if cosine >= 0.80:
                    cosine_profile.append((mode if policy == "topo" else "walk",
                                           round(cosine, 4),
                                           round(_xy_distance(pos[:2], goal_xy), 3)))
                    if cur_colour_sig is not None:
                        frac, rgb = cur_colour_sig
                        colour_profile.append({
                            "block": int(_block),
                            "mode": mode if policy == "topo" else "walk",
                            "at_final_node": bool(at_final_node),
                            "cosine": round(cosine, 4),
                            "distance_m": round(_xy_distance(pos[:2], goal_xy), 3),
                            "colour_frac": round(float(frac), 4),
                            "colour_rgb": ([round(float(v), 4) for v in rgb.tolist()]
                                           if rgb is not None else None),
                            "strict_match": bool(_colour_match(goal_colour_sig, cur_colour_sig)),
                            "terminal_match": bool(_colour_match(
                                goal_colour_sig, cur_colour_sig, min_frac=0.04)),
                        })
                # Two-tier arrival evidence. Walk/servo blocks use tau_eff. An
                # ALIGN scan facing the goal direction from the ADJACENT cell
                # matches a bland corridor goal above tau_eff too (measured:
                # stop at 1.86 m — the alias measure only bounds views >=2 hops
                # out), so scan hits require the high bar that only the true
                # goal-place align reaches (its peak measured 0.993).
                if in_align and not at_final_node:
                    counted = unaliased_goal and cosine >= float(args.tau_arrive_scan)
                else:
                    counted = cosine >= tau_arrive_eff
                # Colour gate (colourful goals only): the latent cosine cannot
                # tell beacon close-ups apart by hue, but the raw pixels can —
                # a claimed arrival must also show the goal's colour.
                if counted and cur_colour_sig is not None:
                    # For a constrained terminal goal, high latent similarity
                    # plus the correct hue is sufficient. Localization can lag
                    # the physical route by several nodes (v42 saw the right
                    # beacon twice at 0.87/0.78 m while at_final_node was false),
                    # so graph-index state must not veto perceptual arrival.
                    # Strict coverage remains mandatory for ordinary goals.
                    min_frac = 0.04 if goal.get("terminal_only", False) else None
                    counted = _colour_match(
                        goal_colour_sig, cur_colour_sig, min_frac=min_frac)
                arrive_hits.append(counted)
                if sum(arrive_hits[-4:]) >= 2:
                    if (runner is not None and policy == "topo"
                            and goal.get("terminal_only", False)):
                        # The camera sees the beacon before the standoff, then
                        # the close beacon leaves the crop as the robot advances.
                        # Latch the two-frame perceptual evidence and complete a
                        # short straight approach on the observed bearing. Six
                        # gait blocks moved v42 from 0.78 m to 0.44 m; the strict
                        # feasibility gate still stops before contact.
                        arrival_armed = True
                    # Final-approach completion (physical): the view matches a
                    # standoff goal up to ~1 cell early (v25: stop at 0.86 m vs
                    # 0.65 radius with ~0.5 m of veto-legal approach left) —
                    # keep servoing a few more blocks while the goal cosine is
                    # still CLIMBING, and stop the moment it falls off its peak
                    # (walked past) or the small budget runs out. v26's
                    # unconditional while-feasible suppression never released:
                    # the robot passed through 0.44 m and drifted out to 1.05 m.
                    past_peak = cosine <= max_goal_cosine - 0.004
                    if arrival_armed:
                        pass
                    elif (runner is not None and policy == "topo" and at_final_node
                            and not past_peak and approach_extra < 6
                            and _feasible_fraction(build, registry, "forward_slow", local_obstacles,
                                                   args.command_dt_s, **veto_servo_kw) >= 0.7):
                        approach_extra += 1
                    else:
                        stopped_perceptually = True
                        break

                completion_primitive = None
                if arrival_armed:
                    _fwd_feas = _feasible_fraction(build, registry, "forward_slow", local_obstacles,
                                                   args.command_dt_s, **veto_servo_kw)
                    if getattr(args, "seek_goal_visual_servo", False) and goal_colour_sig[1] is not None:
                        # Goal-object visual servo. The straight-line completion
                        # only grazes the standoff viewpoint (the beacon is not
                        # dead ahead when arrival arms mid-route), so it stalls
                        # ~0.13 m short. Steer to keep the beacon centred (RGB
                        # centroid) and approach until the goal object fills the
                        # view or the capsule veto stops at it. Yaw blocks do not
                        # spend the forward budget. Off by default = verified
                        # privileged-grid straight-line completion.
                        g_frac, g_x = _goal_colour_centroid_x(image, goal_colour_sig[1])
                        if approach_extra >= 24 or _fwd_feas < 0.7 or g_frac >= 0.45:
                            stopped_perceptually = True
                            break
                        if g_x is not None and abs(g_x) > 0.12:
                            primitive = "yaw_right" if g_x > 0 else "yaw_left"
                            beh["head_corr"] += 1
                            if runner is not None:
                                _execute_physical_primitive(runner, registry, primitive)
                            else:
                                _execute_kinematic_primitive(build, registry, primitive,
                                                             command_dt_s=args.command_dt_s, grid=route_grid)
                            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
                            continue
                        completion_primitive = "forward_slow"
                        approach_extra += 1
                    elif approach_extra >= 6 or _fwd_feas < 0.7:
                        stopped_perceptually = True
                        break
                    else:
                        completion_primitive = "forward_slow"
                        approach_extra += 1

                if completion_primitive is not None:
                    primitive = completion_primitive
                    beh["walk_fwd"] += 1
                elif policy == "topo":
                    # Feed the filter ONLY during locomotion: the belief window
                    # (H=8 consecutive frames) matches the training distribution
                    # of smooth motion; scan rotations poison it for ~8 blocks
                    # after every alignment (measured: filter never confirmed
                    # arrival). The window is frozen during ALIGN.
                    if mode == "walk" or (node_path is None and not initial_plan_pending):
                        navigator.update((z_now, None))
                    # ---- Stage 4b traversal: ALIGN -> WALK per directed edge ----
                    # (plan_cost is flat between interior corridor views, so no
                    # image cost is used between nodes; alignment = raw-frame
                    # cosine to the NEXT node's keyframe, which faces the travel
                    # direction by construction of directed tour edges.)
                    if node_path is None or path_index >= len(node_path):
                        allowed_goal_nodes = goal_node_allowlist(
                            goal.get("colour_sig"),
                            terminal_only=goal.get("terminal_only", False))
                        plan = navigator.plan_node_path(
                            z_goal_raw, avoid_edges=dead_edges,
                            allowed_goal_nodes=allowed_goal_nodes)
                        if plan is None:
                            used_fallback += 1
                            primitive, _cost = choose_vetoed_primitive(
                                model, build, registry, local_obstacles, args.command_dt_s,
                                image, goal_image,
                                sequences, action_tensor, **veto_servo_kw)
                            primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
                            if runner is not None:
                                # The kinematic executor set_pos-slides the base;
                                # under physics that is a mid-gait teleport.
                                _execute_physical_primitive(runner, registry, primitive)
                            else:
                                _execute_kinematic_primitive(build, registry, primitive,
                                                             command_dt_s=args.command_dt_s,
                                                             grid=route_grid)
                            continue
                        node_path, _gnode, _s = plan
                        initial_plan_pending = False
                        path_index, mode, walk_blocks = 1, "align", 0
                        scan_remaining, scan_costs, scan_return = n_scan, [], 0
                        align_phase, scan_samples, scan_cum = "scan", [], 0.0
                        align_blocks_this_edge = 0
                        stuck_streak, escape_blocks, edge_veto_streak, realigns_this_edge = 0, 0, 0, 0
                        subgoal_nodes.append(node_path[-1])
                        n_rev = sum(1 for i in range(1, len(node_path))
                                    if (node_path[i-1], node_path[i]) not in navigator.memory.edges)
                        print(f"    path: {len(node_path)} nodes, {n_rev} reversed edges "
                              f"({node_path[0]}->{node_path[-1]})", flush=True)
                    next_node = node_path[min(path_index, len(node_path) - 1)]
                    at_goal_node = path_index >= len(node_path) - 1
                    prev_node = node_path[max(path_index - 1, 0)]
                    reversed_edge = (prev_node, next_node) not in navigator.memory.edges
                    edge_bearing = navigator._edge_bearings.get((prev_node, next_node))
                    # Prefer the mapping-time IMU bearing for directed forward
                    # traversals. Visual scan alignment remains the fallback for
                    # reversed edges or legacy maps without recorded bearings.
                    if edge_bearing is not None:
                        align_ref = None
                    elif at_goal_node or reversed_edge or navigator._keyframes.get(prev_node) is None:
                        align_ref = next_node
                    else:
                        align_ref = prev_node
                    if mode == "align" and runner is not None:
                        align_blocks_this_edge += 1
                    if (mode == "align" and runner is not None
                            and align_blocks_this_edge > max(4 * n_scan, 24)):
                        # A measured revolution plus the closed-loop turn-to
                        # phase is bounded. If the gait cannot rotate to this
                        # edge bearing/view within that allowance, retrying the
                        # same ALIGN forever only burns the global seek budget
                        # (v36: identical pose and waypoint for 230+ blocks).
                        beh["align_timeout"] += 1
                        if at_goal_node:
                            print("    [goal-approach] alignment budget exhausted - "
                                  "aborting goal", flush=True)
                            break
                        print(f"    [align] edge {prev_node}->{next_node} timed out - "
                              "marking dead and replanning", flush=True)
                        dead_edges.add((int(prev_node), int(next_node)))
                        node_path = None
                        primitive = "hold"
                    elif mode == "align" and runner is not None:
                        # Physical ALIGN: IMU-yaw referenced (spec 3.4: body-frame
                        # integrated Delta-yaw is deployment-admissible). Scan a
                        # MEASURED full revolution sampling (yaw, view-match), then
                        # turn closed-loop on IMU yaw to the best heading
                        # (reversed edge: best + pi).
                        yaw_now = _yaw_from_quat_wxyz(quat)
                        beh["align_blocks"] += 1
                        if edge_bearing is not None:
                            err = wrap_angle_pi(edge_bearing - yaw_now)
                            if abs(err) <= 0.15:
                                mode, walk_blocks = "walk", 0
                                walk_bearing = edge_bearing
                                align_phase, scan_samples, scan_cum = "scan", [], 0.0
                                navigator._window.clear()
                                primitive = "hold"
                            else:
                                primitive = "yaw_left" if err > 0 else "yaw_right"
                        else:
                            keyframe = navigator._keyframes.get(align_ref)
                            cosine_k = (float(F.normalize(z_now, dim=-1) @ keyframe)
                                        if keyframe is not None else 0.0)
                            if align_phase == "scan":
                                if (args.scan_reuse and scan_cum == 0.0 and not scan_samples
                                        and last_scan is not None and keyframe is not None
                                        and _xy_distance(pos[:2], last_scan["xy"]) < 0.2
                                        and len(last_scan["samples"]) >= max(8, n_scan // 2)):
                                    # Scan reuse: a full revolution was already
                                    # swept at (essentially) this spot — score the
                                    # remembered views against the NEW target
                                    # keyframe and turn straight to the best
                                    # bearing. Re-aligns and replans at the same
                                    # place re-spinning full revolutions dominated
                                    # v12 screen time (up to 487/700 blocks).
                                    best_yaw = max(last_scan["samples"],
                                                   key=lambda t: float(t[1] @ keyframe))[0]
                                    turn_target = wrap_angle_pi(best_yaw + (math.pi if reversed_edge else 0.0))
                                    align_phase, turn_cap = "turnto", 3 * n_scan
                                    beh["scan_reuse"] += 1
                                    primitive = "hold"
                                elif cosine_k >= float(args.align_early_cosine) and not reversed_edge:
                                    # Early-out (Stage 4c): already facing the
                                    # keyframe view — skip the rest of the
                                    # revolution. The bar is below the kinematic
                                    # 0.95: the walking camera's pitch/bob caps the
                                    # cosine, so 0.95 almost never fired under gait
                                    # and every align became a full revolution.
                                    # Safe at 0.92 because the latent is
                                    # yaw-dominant (wrong headings score low).
                                    mode, walk_blocks = "walk", 0
                                    walk_bearing = yaw_now
                                    align_phase, scan_samples, scan_cum = "scan", [], 0.0
                                    navigator._window.clear()
                                    primitive = "hold"
                                elif scan_cum >= 2.0 * math.pi:
                                    best_yaw = max(scan_samples, key=lambda t: t[1])[0]
                                    last_scan = {"xy": (float(pos[0]), float(pos[1])),
                                                 "samples": [(y, z) for y, _c, z in scan_samples]}
                                    turn_target = wrap_angle_pi(best_yaw + (math.pi if reversed_edge else 0.0))
                                    align_phase, turn_cap = "turnto", 3 * n_scan
                                    primitive = "yaw_right"
                                else:
                                    if scan_samples:
                                        step = abs(wrap_angle_pi(yaw_now - scan_samples[-1][0]))
                                        scan_cum += step
                                    scan_samples.append((yaw_now, cosine_k,
                                                         F.normalize(z_now, dim=-1)))
                                    primitive = "yaw_right"
                            else:  # turnto
                                err = wrap_angle_pi(turn_target - yaw_now)
                                turn_cap -= 1
                                if abs(err) <= 0.15 or turn_cap <= 0:
                                    mode, walk_blocks = "walk", 0
                                    walk_bearing = turn_target
                                    align_phase, scan_samples, scan_cum = "scan", [], 0.0
                                    navigator._window.clear()
                                    primitive = "hold"
                                else:
                                    primitive = "yaw_left" if err > 0 else "yaw_right"
                    elif mode == "align":
                        if edge_bearing is not None:
                            yaw_now = _yaw_from_quat_wxyz(quat)
                            err = wrap_angle_pi(edge_bearing - yaw_now)
                            if abs(err) <= 0.15:
                                mode, walk_blocks = "walk", 0
                                walk_bearing = edge_bearing
                                navigator._window.clear()
                                primitive = "hold"
                            else:
                                primitive = "yaw_left" if err > 0 else "yaw_right"
                        else:
                            keyframe = navigator._keyframes.get(align_ref)
                            cosine_k = (float(F.normalize(z_now, dim=-1) @ keyframe)
                                        if keyframe is not None else 0.0)
                            if (cosine_k >= 0.95 and not reversed_edge) or (scan_remaining == 0 and scan_return == 0):
                                mode, walk_blocks = "walk", 0
                                navigator._window.clear()
                                primitive = "hold"
                            elif scan_remaining > 0:
                                scan_costs.append(cosine_k)
                                scan_remaining -= 1
                                if scan_remaining == 0:
                                    # Kinematic: exact yaw per block -> open-loop
                                    # return, VERBATIM the verified-success logic.
                                    best = int(np.argmax(scan_costs))
                                    offset = n_scan // 2 if reversed_edge else 0
                                    scan_return = (best + 1 + offset) % n_scan
                                    align_op = "open"
                                primitive = "yaw_right"
                            elif align_op == "open":
                                scan_return -= 1
                                primitive = "yaw_right"
                            else:
                                hit = ((cosine_k >= align_target) if align_op == "ge"
                                       else (cosine_k <= align_target))
                                scan_return -= 1
                                if hit or scan_return <= 0:
                                    mode, walk_blocks = "walk", 0
                                    navigator._window.clear()
                                    primitive = "hold"
                                else:
                                    primitive = "yaw_right"
                    else:  # walk
                        posterior = navigator.current_belief()
                        # Arrival at ANY node at/after path_index counts (parallel
                        # view-nodes split the posterior; MAP may advance along a
                        # sibling chain).
                        remaining = node_path[path_index:]
                        arrived_at = None
                        if navigator._last_map in remaining:
                            arrived_at = navigator._last_map
                        else:
                            for candidate in remaining:
                                if posterior.get(candidate, 0.0) >= 0.5:
                                    arrived_at = candidate
                                    break
                        arrived_node = arrived_at is not None
                        if arrived_node:
                            next_node = arrived_at
                            path_index = node_path.index(arrived_at)
                            subgoals_reached += 1
                            sg_label = majority.get(next_node)
                            cur_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), pos[:2]))
                            if sg_label is not None:
                                d_cur = graph.bfs_distance(cur_cell, goal_cell)
                                d_sg = graph.bfs_distance(int(sg_label[0]), goal_cell)
                                if d_cur is not None and d_sg is not None:
                                    subgoal_scored += 1
                                    subgoal_progress += int(d_sg <= d_cur)
                            path_index += 1
                            mode = "align"
                            scan_remaining, scan_costs, scan_return = n_scan, [], 0
                            align_phase, scan_samples, scan_cum = "scan", [], 0.0
                            align_blocks_this_edge = 0
                            stuck_streak, escape_blocks, edge_veto_streak, realigns_this_edge = 0, 0, 0, 0
                            primitive = "hold"
                        else:
                            # The final spur uses the SAME clearance-aware
                            # bearing hold as every other edge, with its
                            # mapping-time measured bearing because a close-up
                            # spur keyframe cannot determine travel direction
                            # while still at the anchor. Perceptual arrival
                            # supplies the stop.
                            walk_blocks += 1
                            if at_goal_node:
                                final_approach_blocks += 1
                            if at_goal_node and final_approach_blocks > args.edge_budget:
                                print("    [goal-approach] budget exhausted - aborting goal",
                                      flush=True)
                                break
                            elif not at_goal_node and walk_blocks > args.edge_budget:
                                print(f"    [walk] edge {prev_node}->{next_node} exhausted "
                                      f"{args.edge_budget}-block budget at "
                                      f"({pos[0]:.2f},{pos[1]:.2f}); replanning", flush=True)
                                dead_edges.add((int(prev_node), int(next_node)))
                                node_path = None        # lost on this edge -> replan
                                primitive = "hold"
                            elif runner is not None and escape_blocks > 0:
                                # Proprioceptive wall contact (stuck detector
                                # fired): back straight off the obstacle, then
                                # re-align — the local-obstacle veto already mispredicted
                                # here, so forward is not to be trusted again
                                # on this bearing.
                                escape_blocks -= 1
                                beh["stuck_escape"] += 1
                                if _feasible_fraction(build, registry, "backward", local_obstacles,
                                                      args.command_dt_s, **veto_walk_kw) >= 0.5:
                                    primitive = "backward"
                                else:
                                    primitive = max(("yaw_left", "yaw_right"),
                                                    key=lambda p: _feasible_fraction(
                                                        build, registry, p, local_obstacles,
                                                        args.command_dt_s, **veto_walk_kw))
                                if escape_blocks == 0:
                                    mode = "align"
                                    scan_remaining, scan_costs, scan_return = n_scan, [], 0
                                    align_phase, scan_samples, scan_cum = "scan", [], 0.0
                                    align_blocks_this_edge = 0
                                    navigator._window.clear()
                            elif runner is not None:
                                # Stage 4c closed-loop WALK: the PPO gait's yaw
                                # tracking drifts, so open-loop forward blocks
                                # veer into corridor walls (the rejected v9
                                # video). Hold the post-ALIGN bearing on IMU
                                # yaw — arcs correct small error while keeping
                                # forward progress; pure yaw only for gross
                                # error.
                                yaw_now = _yaw_from_quat_wxyz(quat)
                                herr = (wrap_angle_pi(walk_bearing - yaw_now)
                                        if walk_bearing is not None else 0.0)
                                beh["head_err_sq"] += herr * herr
                                beh["head_err_n"] += 1
                                if abs(herr) > 0.45:
                                    primitive = "yaw_left" if herr > 0 else "yaw_right"
                                    beh["head_corr"] += 1
                                else:
                                    cand = ("arc_left" if herr > 0.12 else
                                            "arc_right" if herr < -0.12 else "forward_slow")
                                    # Clearance-graded steering (DWA-lite). The
                                    # binary veto only rejects predicted CONTACT,
                                    # so the bearing-hold choice may legally skim
                                    # the inflated boundary every block — and the
                                    # gait tracks loosely enough that "skim the
                                    # model" = "scrape the wall" on camera (v27:
                                    # 43 escapes, 36 backward, walls on video).
                                    # Among strict-feasible candidates, score by
                                    # padded-capsule clearance (longer nose +
                                    # side probes) minus a heading-deviation
                                    # penalty: in the open everything pads 1.0
                                    # and the bearing hold wins; near a wall the
                                    # away-arc outscores it and the robot bends
                                    # off BEFORE contact instead of escaping
                                    # after. Penalties sized so one padded
                                    # substep (0.1) of clearance flips an arc on.
                                    chosen, best_score = None, None
                                    for option, dev_pen in ((cand, 0.0),
                                                            ("forward_slow", 0.04),
                                                            ("arc_left", 0.08),
                                                            ("arc_right", 0.08)):
                                        if _feasible_fraction(build, registry, option, local_obstacles,
                                                              args.command_dt_s, **veto_walk_kw) < 0.8:
                                            continue
                                        pad = _feasible_fraction(build, registry, option, local_obstacles,
                                                                 args.command_dt_s, **veto_pad_kw)
                                        score = pad - dev_pen
                                        if best_score is None or score > best_score:
                                            chosen, best_score = option, score
                                    if chosen is not None:
                                        primitive = chosen
                                        beh["walk_fwd"] += 1
                                        beh["head_corr"] += int(chosen != "forward_slow")
                                        edge_veto_streak = 0
                                    else:
                                        # Wall-aware escape: back off or turn
                                        # toward open space, not the blind
                                        # yaw_right that read as a freak-out on
                                        # camera. Tie-break strict feasibility
                                        # (often 1.0 for several options) by
                                        # padded clearance so the escape moves
                                        # AWAY from the wall, not along it.
                                        # Perception occupancy is noisier than the
                                        # privileged grid; on a straight corridor a
                                        # transient forward veto (gait heading drift)
                                        # makes "backward" the highest-clearance
                                        # escape, so the robot RETREATS down the
                                        # corridor it came from instead of
                                        # re-centering. --seek-escape-no-backward
                                        # drops backward so the escape turns to
                                        # re-acquire the bearing (default keeps the
                                        # verified privileged-grid behaviour).
                                        options = (("yaw_left", "yaw_right", "arc_left", "arc_right")
                                                   if getattr(args, "seek_escape_no_backward", False)
                                                   else ("backward", "yaw_left", "yaw_right",
                                                         "arc_left", "arc_right"))
                                        primitive = max(options, key=lambda p: (
                                            _feasible_fraction(build, registry, p, local_obstacles,
                                                               args.command_dt_s, **veto_walk_kw),
                                            _feasible_fraction(build, registry, p, local_obstacles,
                                                               args.command_dt_s, **veto_pad_kw)))
                                        beh["veto_escape"] += 1
                                        edge_veto_streak += 1
                                        if edge_veto_streak >= int(getattr(args, "seek_edge_veto_streak", 3)):
                                            # The aligned bearing leads into a
                                            # wall (bad align or gait drift):
                                            # stop fighting the heading hold —
                                            # re-scan for the keyframe instead
                                            # of oscillating veto<->correction.
                                            # But only twice per edge: each
                                            # align completion resets
                                            # walk_blocks, so without this cap
                                            # a blocked edge NEVER exhausts its
                                            # budget (v19: 48 realigns, 195
                                            # escapes on one edge, whole seek
                                            # burned in place).
                                            edge_veto_streak = 0
                                            realigns_this_edge += 1
                                            beh["edge_realign"] += 1
                                            realign_cap = int(getattr(args, "seek_edge_realign_cap", 2))
                                            if realigns_this_edge >= realign_cap:
                                                if at_goal_node:
                                                    print(f"    [goal-approach] blocked after "
                                                          f"{realign_cap} realigns - aborting goal", flush=True)
                                                    break
                                                else:
                                                    print(f"    [walk] edge {prev_node}->{next_node} "
                                                          f"blocked after {realign_cap} realigns; replanning",
                                                          flush=True)
                                                    dead_edges.add((int(prev_node), int(next_node)))
                                                    node_path = None    # edge is dead -> replan
                                                    realigns_this_edge = 0
                                            else:
                                                mode = "align"
                                                scan_remaining, scan_costs, scan_return = n_scan, [], 0
                                                align_phase, scan_samples, scan_cum = "scan", [], 0.0
                                                align_blocks_this_edge = 0
                                                navigator._window.clear()
                            else:
                                forward_ok = _feasible_fraction(build, registry, "forward_medium",
                                                                local_obstacles,
                                                                args.command_dt_s) >= 0.5
                                primitive = "forward_medium" if forward_ok else "yaw_right"
                    if runner is not None and _block % 10 == 0:
                        belief = navigator.current_belief()
                        p_node, p_w = (max(belief.items(), key=lambda kv: kv[1])
                                       if belief else (None, 0.0))
                        print(f"    [trace] b{_block} {mode} idx {path_index}/"
                              f"{len(node_path) if node_path else 0} next "
                              f"{next_node if node_path else -1} map {navigator._last_map} "
                              f"p* {p_node}:{p_w:.2f} walk_blocks {walk_blocks} "
                              f"pos ({pos[0]:.2f},{pos[1]:.2f})", flush=True)
                elif policy == "v2":
                    primitive, _cost = choose_vetoed_primitive(
                        model, build, registry, local_obstacles, args.command_dt_s,
                        image, goal_image,
                        sequences, action_tensor, **veto_servo_kw)
                elif policy == "bearing":
                    primitive = _choose_bearing_primitive(build, (float(goal_xy[0]), float(goal_xy[1])))
                elif policy == "hold":
                    primitive = "hold"
                else:
                    raise ValueError(policy)
                primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
                if runner is not None:
                    _execute_physical_primitive(runner, registry, primitive)
                else:
                    _execute_kinematic_primitive(
                        build, registry, primitive,
                        command_dt_s=args.command_dt_s, grid=route_grid)
                new_pos, _ = _current_pose(build)
                if (runner is not None and policy == "topo" and mode == "walk"
                        and escape_blocks == 0
                        and primitive in ("forward_slow", "forward_medium", "forward_fast",
                                          "arc_left", "arc_right")):
                    # Stuck detector: commanded forward motion with essentially
                    # no displacement = wall contact the local-obstacle veto missed
                    # (commanded forward_slow moves ~0.10 m/block when free).
                    # The v9 falls came from pushing against the wall anyway.
                    disp = float(np.linalg.norm(np.asarray(new_pos[:2], dtype=np.float64)
                                                - np.asarray(pos[:2], dtype=np.float64)))
                    stuck_streak = stuck_streak + 1 if disp < 0.03 else 0
                    if stuck_streak >= 2:
                        stuck_streak, escape_blocks = 0, 2
                        beh["stuck_detect"] += 1
                        print(f"    [stuck] forward commanded, moved {disp:.3f} m - backing off", flush=True)
                if runner is not None and float(new_pos[2]) < 0.15:
                    falls += 1
                    if falls > 6:
                        print("    [fall] too many falls - aborting trial", flush=True)
                        break
                    # Recover at the nearest locally free point: stand back up clear
                    # of the wall it fell against, reset the gait policy state
                    # (locomotion robustness is not the thesis under test; real
                    # platforms have get-up controllers).
                    free_x, free_y = _nearest_free_xy(local_obstacles, new_pos[:2])
                    up = np.asarray([free_x, free_y, start_pose[0][2]], dtype=np.float32)
                    _, cur_quat = _current_pose(build)
                    cur_yaw = _yaw_from_quat_wxyz(cur_quat)
                    _set_pose(build=build, runner=runner, pos_xyz=up, quat_wxyz=_quat_wxyz_from_yaw(cur_yaw))
                    print(f"    [fall->recover #{falls}]", flush=True)
                    if policy == "topo":
                        # Tumble frames (camera on the floor) are far outside
                        # the smooth-locomotion distribution the filter was
                        # calibrated on — same reasoning as freezing the window
                        # during ALIGN. Purge them and re-align from the
                        # recovered pose instead of walking on with a poisoned
                        # posterior (measured: replans cluster right after
                        # falls).
                        navigator._window.clear()
                        mode, walk_blocks = "align", 0
                        scan_remaining, scan_costs, scan_return = n_scan, [], 0
                        align_phase, scan_samples, scan_cum = "scan", [], 0.0
                        align_blocks_this_edge = 0
                        stuck_streak, escape_blocks, edge_veto_streak, realigns_this_edge = 0, 0, 0, 0
                if demo is not None and policy == "topo":
                    if node_path is None:
                        demo_state, waypoint_ref = "REPLANNING (lost - re-localize)", None
                    elif mode == "align":
                        demo_state, waypoint_ref = "ALIGN: scanning for waypoint view (4x speed)", \
                            navigator._observations.get(next_node)
                    elif arrival_armed or (at_goal_node and walk_blocks > 2):
                        demo_state, waypoint_ref = "FINAL APPROACH: servo on goal image", None
                    else:
                        demo_state, waypoint_ref = "WALK: forward to waypoint", \
                            navigator._observations.get(next_node)
                    if waypoint_ref is not None and waypoint_ref not in demo["waypoint_thumbs"]:
                        thumb = stored_frames[waypoint_ref].mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        demo["waypoint_thumbs"][waypoint_ref] = thumb
                    status = {"phase": "SEEK", "state": demo_state, "waypoint_ref": waypoint_ref,
                              "nodes": len(navigator.memory.nodes),
                              "goal_idx": goal_index, "goal_name": goal["name"],
                              "claimed_n": claimed_n,
                              "dist": _xy_distance(new_pos[:2], goal_xy)}
                    _, _q = _current_pose(build)
                    demo["seek_blocks"].append((primitive,
                                                (np.asarray(new_pos, dtype=np.float32).copy(),
                                                 float(_yaw_from_quat_wxyz(_q))),
                                                status))
                path_length += float(np.linalg.norm(np.asarray(new_pos[:2], dtype=np.float64) - prev_xy))
                prev_xy = np.asarray(new_pos[:2], dtype=np.float64)

            final_pos, _ = _current_pose(build)
            final_distance = _xy_distance(final_pos[:2], goal_xy)
            if policy == "topo" and cosine_profile:
                top = sorted(cosine_profile, key=lambda e: -e[1])[:6]
                print("    cosine profile (top): " +
                      "  ".join(f"{m} {c:.3f}@{d:.2f}m" for m, c, d in top), flush=True)
            beh["head_rms_rad"] = (math.sqrt(beh["head_err_sq"] / beh["head_err_n"])
                                   if beh["head_err_n"] else None)
            if policy == "topo" and runner is not None:
                print(f"    behavior: walk_fwd {beh['walk_fwd']} head_corr {beh['head_corr']} "
                      f"veto_escape {beh['veto_escape']} align {beh['align_blocks']} "
                      f"align_timeout {beh['align_timeout']} "
                      f"stuck {beh['stuck_detect']} edge_realign {beh['edge_realign']} "
                      f"scan_reuse {beh['scan_reuse']} "
                      f"head_rms {beh['head_rms_rad'] if beh['head_rms_rad'] is None else round(beh['head_rms_rad'], 3)}",
                      flush=True)
            if demo is not None and policy == "topo" and stopped_perceptually and demo["seek_blocks"]:
                for _prim, _pose, st in demo["seek_blocks"][-2:]:
                    st["stopped"] = True
                if goal_index == len(goals) - 1:
                    demo["statuses"][-1]["stopped"] = True
            if stopped_perceptually:
                claimed_n += 1
            goal_rows.append({
                "goal_name": goal["name"],
                "goal_cell": int(goal_cell),
                "initial_distance_m": d0,
                "final_distance_m": final_distance,
                "progress_m": d0 - final_distance,
                "success_eval": bool(final_distance <= args.goal_radius_m),
                "stopped_perceptually": stopped_perceptually,
                "perceptual_stop_correct": bool(stopped_perceptually and final_distance <= args.goal_radius_m),
                "n_subgoal_switches": len(set(subgoal_nodes)),
                "max_goal_cosine": max_goal_cosine,
                "cosine_profile_top": sorted(cosine_profile, key=lambda e: -e[1])[:20],
                "colour_profile_top": sorted(
                    colour_profile, key=lambda e: -e["cosine"])[:20],
                "behavior": {k: v for k, v in beh.items() if k not in ("head_err_sq", "head_err_n")},
                "subgoals_reached": subgoals_reached,
                "subgoal_progress_rate": (subgoal_progress / subgoal_scored) if subgoal_scored else None,
            })
            print(f"  [{policy}] goal[{goal_index}] '{goal['name']}' final={final_distance:.2f} m "
                  f"success={goal_rows[-1]['success_eval']} perc_stop={stopped_perceptually} "
                  f"fallback={used_fallback}", flush=True)

        results[policy] = {
            "initial_distance_m": goal_rows[0]["initial_distance_m"],
            "final_distance_m": goal_rows[-1]["final_distance_m"],
            "progress_m": float(sum(r["progress_m"] for r in goal_rows)),
            "path_length_m": path_length,
            "success_eval": bool(all(r["success_eval"] for r in goal_rows)),
            "stopped_perceptually": bool(all(r["stopped_perceptually"] for r in goal_rows)),
            "perceptual_stop_correct": bool(all(r["perceptual_stop_correct"] for r in goal_rows)),
            "fallback_blocks": used_fallback,
            "n_subgoal_switches": sum(r["n_subgoal_switches"] for r in goal_rows),
            "max_goal_cosine": max(r["max_goal_cosine"] for r in goal_rows),
            "primitive_counts": primitive_counts,
            "subgoals_reached": sum(r["subgoals_reached"] for r in goal_rows),
            "falls": falls,
            "goals_reached": sum(int(r["success_eval"]) for r in goal_rows),
            "n_goals": len(goal_rows),
            "subgoal_progress_rate": (
                float(np.mean([r["subgoal_progress_rate"] for r in goal_rows
                               if r["subgoal_progress_rate"] is not None]))
                if any(r["subgoal_progress_rate"] is not None for r in goal_rows) else None),
            "goals": goal_rows,
        }

    g0 = goals[0]
    g0_node_cell = majority.get(g0["match_node"], (None, 0, False))[0] if g0["match_node"] is not None else None
    return {
        "scene_id": str(pack.scene_id),
        "tour_blocks": tour_blocks,
        "n_memory_nodes": n_nodes,
        "n_cells_visited": len(visited_cells),
        "goal_cell": int(g0["cell"]),
        "goal_hops": graph.bfs_distance(end_cell, g0["cell"]),
        "goal_match_node": g0["match_node"],
        "goal_match_score": g0["match_score"],
        "goal_match_cell_correct": bool(g0_node_cell == g0["cell"]),
        "local_obstacle_contract": local_obstacles.contract.to_dict(),
        "local_obstacle_diagnostics": (
            local_obstacles.diagnostics()
            if callable(getattr(local_obstacles, "diagnostics", None))
            else None
        ),
        "goals": [{"name": g["name"], "cell": int(g["cell"]),
                   "match_node": g["match_node"], "match_score": g["match_score"],
                   "saturation": g["saturation"],
                   "terminal_only": bool(g.get("terminal_only", False)),
                   "planned_route_nodes": g.get("planned_route_nodes")} for g in goals],
        "policies": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--belief-encoder", type=Path, required=True)
    parser.add_argument("--loop-head", type=Path, required=True)
    parser.add_argument("--scene-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--platform-manifest", type=Path, default=REPO_ROOT / "config/go2_platform_manifest.yaml")
    parser.add_argument("--primitive-registry", type=Path, default=REPO_ROOT / "config/go2_primitive_registry.yaml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".generated/topo_nav/topo_nav_e2e.json")
    parser.add_argument("--split", default="test_id")
    parser.add_argument("--family", default="medium_enclosed_maze")
    parser.add_argument("--scene-limit", type=int, default=2)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--goals-per-scene", type=int, default=2)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--mode", choices=("kinematic", "physical"), default="kinematic",
                        help="physical = real PPO locomotion for the SEEK phase (tour stays teleport)")
    parser.add_argument(
        "--require-deployment-valid-local-obstacles",
        action="store_true",
        help="refuse runtime seek safety backed by privileged manifest/oracle geometry; "
             "requires an injected perception-backed LocalObstacleModel",
    )
    parser.add_argument(
        "--local-obstacle-source",
        choices=("privileged-grid", "ego-depth"),
        default="privileged-grid",
        help="runtime seek collision source; ego-depth is perception-backed but "
             "this benchmark registers it with simulator pose, so strict "
             "deployment-valid mode rejects it",
    )
    parser.add_argument("--depth-obstacle-max-range-m", type=float, default=3.0)
    parser.add_argument("--depth-obstacle-stride-px", type=int, default=8)
    parser.add_argument("--depth-obstacle-inflation-m", type=float, default=0.22)
    parser.add_argument("--depth-obstacle-max-age-frames", type=int, default=12)
    parser.add_argument(
        "--depth-obstacle-unknown-is-free",
        action="store_true",
        help="permit unobserved local space; diagnostic only, off for conservative runs",
    )
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument("--policies", default="topo,v2,bearing,hold")
    parser.add_argument("--primitive-names", default="hold,forward_slow,forward_medium,forward_fast,yaw_left,yaw_right,arc_left,arc_right")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--tau-goal", type=float, default=0.80)
    parser.add_argument("--tau-arrive", type=float, default=0.90)
    parser.add_argument("--tau-arrive-floor", type=float, default=0.84,
                        help="lower bound for the goal-conditioned arrival threshold "
                             "min(tau_arrive, max(floor, goal_alias + 0.02))")
    parser.add_argument("--tau-arrive-scan", type=float, default=0.95,
                        help="arrival bar for ALIGN-scan blocks on unaliased goals (adjacent-cell "
                             "scans match bland goals above tau_eff; only the goal-place align "
                             "should count)")
    parser.add_argument("--tau-subgoal-reached", type=float, default=0.85)
    parser.add_argument("--align-early-cosine", type=float, default=0.95,
                        help="physical ALIGN scan early-out bar. Lowering it (0.92) cut scan time but "
                             "POISONED alignment: same-heading corridor views alias at 0.93-0.98, so "
                             "the early-out fired facing the wrong corridor and locked walk_bearing "
                             "onto walls (v18-v20 escape storms vs v12 with 0.95)")
    parser.add_argument("--demo-require-colourful", action="store_true",
                        help="skip trial draws without a colourful, fully forward-routable goal "
                             "instead of falling back to a bland or reversed-edge seek")
    parser.add_argument("--scan-reuse", action="store_true",
                        help="reuse the last full revolution's views when re-aligning at the same spot "
                             "(<0.2 m). Same aliasing caveat as the early-out bar; off by default")
    parser.add_argument("--tour-landmark-lookat", action="store_true",
                        help="tour turns to face each landmark from its nearest stand cell and walks "
                             "in to the goal standoff with FED frames: mints the landmark-facing "
                             "memory nodes a colourful goal image needs to be plannable")
    parser.add_argument("--veto-body-radius-m", type=float, default=0.25,
                        help="physical mode: capsule probe radius for the local-obstacle veto (Go2 nose/tail "
                             "extend ~0.25 m beyond base center; base-center checks walked into walls)")
    parser.add_argument("--veto-horizon-blocks", type=int, default=2,
                        help="physical mode: blocks of continuation simulated by the walk-gate veto "
                             "(the gait keeps moving past a single 0.5 s block)")
    parser.add_argument("--veto-pad-radius-m", type=float, default=0.40,
                        help="physical mode: nose/tail probe radius for the padded clearance SCORE "
                             "used by WALK steering (comfort margin, not a hard gate)")
    parser.add_argument("--veto-pad-lateral-m", type=float, default=0.35,
                        help="physical mode: side-probe radius for the padded clearance score (the "
                             "strict capsule is blind to walls parallel to the path)")
    parser.add_argument("--subgoal-budget", type=int, default=26)
    parser.add_argument("--edge-budget", type=int, default=14)
    parser.add_argument("--tour-max-cells", type=int, default=24)
    parser.add_argument("--tour-max-blocks", type=int, default=260)
    parser.add_argument("--tour-reprise-cells", type=int, default=0,
                        help="re-walk the first N outbound cells after the DFS unwinds: mints the "
                             "late->early loop-closure edge so the seek start has forward routes "
                             "(0 = off, the pre-reprise behaviour)")
    parser.add_argument("--tour-step-m", type=float, default=0.12,
                        help="tour pose stride; raise on large open scenes so the sweep covers them")
    parser.add_argument("--tour-capture-every", type=int, default=2,
                        help="demo: capture every Nth tour block (controls tour screen time)")
    parser.add_argument("--seek-max-blocks", type=int, default=200)
    parser.add_argument("--seek-edge-veto-streak", type=int, default=3,
                        help="walk vetoes in a row before a re-scan realign (default 3, "
                             "the verified privileged-grid value). Raise for noisier "
                             "perception occupancy so a transient veto does not realign.")
    parser.add_argument("--seek-edge-realign-cap", type=int, default=2,
                        help="realigns on one edge before it is marked dead and replanned "
                             "(default 2). Raise to persist on a sole-route edge that a "
                             "perception obstacle source intermittently vetoes.")
    parser.add_argument("--seek-escape-no-backward", action="store_true",
                        help="drop 'backward' from the walk veto-escape set so the robot "
                             "re-centers by turning instead of retreating down the corridor "
                             "(off by default = verified privileged-grid behaviour)")
    parser.add_argument("--seek-goal-visual-servo", action="store_true",
                        help="on perceptual arrival, steer to keep the goal beacon centred "
                             "(RGB colour centroid) and approach until it fills the view or "
                             "the capsule veto stops at it, instead of a straight-line "
                             "forward_slow that only grazes the standoff viewpoint. Closes "
                             "the goal approach for a perception obstacle source; off by "
                             "default = verified straight-line completion.")
    parser.add_argument("--goal-min-hops", type=int, default=3)
    parser.add_argument("--goal-max-hops", type=int, default=5)
    parser.add_argument("--goal-mode", choices=("random", "colourful", "beacons"), default="random",
                        help="random: hop-valid visited cell (eval contract). colourful: same contract, "
                        "prefer the most colour-saturated goal view (legible demos). beacons: seek each "
                        "coloured landmark in sequence (multi-goal demo).")
    parser.add_argument("--goal-beacons", type=int, default=4,
                        help="max beacons to chase with --goal-mode beacons")
    parser.add_argument("--goal-radius-m", type=float, default=0.65)
    parser.add_argument("--goal-standoff-m", type=float, default=0.72,
                        help="colourful mode: distance from the landmark at which the goal "
                             "image is rendered (within the measured colour-visibility range)")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--demo-video", type=Path, default=None)
    parser.add_argument("--demo-fps", type=float, default=12.0)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_grad_enabled(False)
    model, _ = load_model(SimpleNamespace(checkpoint=args.checkpoint.resolve(),
                                          max_seq_len=args.max_seq_len, sigreg_lambda=args.sigreg_lambda), device)
    encoder, scorer, tau_new = load_navigator_parts(args, device)
    platform = load_platform_manifest(args.platform_manifest.resolve())
    registry = PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    safety = SafetyLimits.from_manifest(platform)
    locomotion_policy = None
    if args.mode == "physical" or args.demo_video is not None:
        locomotion_policy = GenesisGo2PPOPolicy.from_platform_manifest(platform, REPO_ROOT, device="cpu")
        print("[physical] PPO locomotion policy loaded", flush=True)
    args.command_dt_s = float(platform.get("timing", {}).get("command_dt_s", 0.10))
    primitive_names = _parse_csv(args.primitive_names)
    if args.demo_video is not None:
        args.policies = "topo"
    primitive_blocks = _primitive_active_blocks(registry, primitive_names)
    rng = random.Random(args.seed)
    sequences, action_tensor = _candidate_action_tensor(
        primitive_blocks, primitive_names, args.horizon,
        max_candidates=args.max_candidates, rng=rng, device=device,
    )

    scene_dirs = sorted(find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family),
                        key=lambda p: p.name)[args.scene_offset:args.scene_offset + args.scene_limit]
    if not scene_dirs:
        raise SystemExit("no scenes found")

    all_results = []
    started = time.time()
    for scene_dir in scene_dirs:
        pack = load_scene_pack(scene_dir, platform_manifest=platform, workspace_root=REPO_ROOT)
        print(f"scene {pack.scene_id} ({pack.family}/{pack.split})", flush=True)
        build = build_scene_from_pack(pack, n_envs=1, backend=args.backend,
                                      show_viewer=False, render_robot=False,
                                      apply_textures=bool(args.apply_textures))
        runner = None
        if args.mode == "physical":
            runner = RolloutRunner(build, locomotion_policy, registry, safety, config=RolloutConfig(
                n_blocks=int(args.seek_max_blocks), fall_z_threshold_m=0.15,
                rgb_capture_per_block=False, seed=int(args.seed),
                log_progress_every_blocks=0, foot_contact_source="zero",
                randomize_spawn_pose=False))
        for trial in range(args.goals_per_scene):
            trial_rng = random.Random(args.seed + trial * 1009 + zlib.crc32(pack.scene_id.encode()) % 100000)
            demo = ({"poses": [], "statuses": [], "goals": [],
                     "waypoint_thumbs": {}, "seek_blocks": [], "seek_start": None}
                    if args.demo_video is not None else None)
            out = run_scene(pack, build, model, encoder, scorer, tau_new, registry,
                            sequences, action_tensor, primitive_names, args, trial_rng, device,
                            demo=demo, runner=runner)
            if out is not None:
                out["trial"] = trial
                all_results.append(out)
            if demo is not None and out is not None and (demo["poses"] or demo["seek_blocks"]):
                # Only render/keep the demo for a SUCCESSFUL trial (or the
                # final attempt): the replay pass is expensive and a failed
                # wander makes a bad video; report attempts honestly instead.
                topo_res = (out or {}).get("policies", {}).get("topo")
                demo_ok = bool(topo_res and topo_res.get("perceptual_stop_correct"))
                if not demo_ok and trial < int(args.goals_per_scene) - 1:
                    print(f"  [demo] trial {trial} did not succeed - retrying with next goal draw", flush=True)
                    continue
                print(f"rendering demo replay ({len(demo['poses'])} tour poses + "
                      f"{len(demo['seek_blocks'])} gait blocks)...", flush=True)
                demo_frames, demo_statuses = _render_demo_frames(
                    pack, demo, args, registry, safety, locomotion_policy)
                if demo_statuses and demo["statuses"] and demo["statuses"][-1].get("stopped"):
                    for st in demo_statuses[-3:]:
                        st["stopped"] = True
                title = ("Topological Nav - learned map + multi-beacon seek (goal images only)"
                         if args.goal_mode == "beacons"
                         else "Topological Nav - learned map + goal-image seek (held-out maze)")
                _write_topo_demo_video(
                    args.demo_video, pack, demo_frames, demo_statuses,
                    demo["goals"], args.demo_fps, title,
                    waypoint_thumbs=demo["waypoint_thumbs"])
                print(f"wrote demo video {args.demo_video} ({len(demo_frames)} frames)", flush=True)
                if demo_ok:
                    break   # keep the successful trial's video

    summary: dict = {"schema": "lewm_topo_nav_e2e_v0", "elapsed_s": time.time() - started,
                     "n_trials": len(all_results), "trials": all_results,
                     "config": {k: str(v) for k, v in vars(args).items()}}
    by_policy: dict[str, list] = {}
    for trial_result in all_results:
        for policy, metrics in trial_result["policies"].items():
            by_policy.setdefault(policy, []).append(metrics)
    summary["aggregate"] = {
        policy: {
            "n": len(rows),
            "success_rate_eval": float(np.mean([r["success_eval"] for r in rows])),
            "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in rows])),
            "mean_progress_m": float(np.mean([r["progress_m"] for r in rows])),
            "perceptual_stop_rate": float(np.mean([r["stopped_perceptually"] for r in rows])),
            "perceptual_stop_correct_rate": float(np.mean([r["perceptual_stop_correct"] for r in rows])),
        } for policy, rows in by_policy.items()
    }
    summary["goal_match_cell_accuracy"] = float(np.mean([t["goal_match_cell_correct"] for t in all_results])) if all_results else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
