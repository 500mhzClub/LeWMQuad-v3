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
from benchmark_lewm_closed_loop_mpc import _execute_physical_primitive  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, SafetyLimits, expand_primitive_to_block  # noqa: E402
from lewm_genesis.rollout import GenesisGo2PPOPolicy, RolloutConfig, RolloutRunner  # noqa: E402
from lewm_genesis.collectors.base import wrap_angle_pi  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import find_scene_dirs, load_platform_manifest, load_scene_pack  # noqa: E402
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


def graph_tour_cells(graph, start_cell: int, max_cells: int, rng) -> list[int]:
    """DFS traversal (with backtracking steps) over the cell graph — every
    consecutive pair is graph-adjacent, so a waypoint follower can drive it."""
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
    return order


def tour_pose_sequence(graph, tour_cells, base_z, *, step_m: float = 0.12, max_steps: int):
    """Dense (pos, yaw) walk along the DFS cell path — the tour's motion is
    privileged (stands in for exploration), so we interpolate poses directly at
    a ~bank-cadence step; yaw faces the direction of motion."""
    poses = []
    current = np.asarray(graph.cell_center(tour_cells[0]), dtype=np.float64)
    for cell in tour_cells[1:]:
        target = np.asarray(graph.cell_center(cell), dtype=np.float64)
        delta = target - current
        span = float(np.linalg.norm(delta))
        if span < 1e-6:
            continue
        yaw = math.atan2(delta[1], delta[0])
        n_steps = max(1, int(math.ceil(span / step_m)))
        for i in range(1, n_steps + 1):
            xy = current + delta * (i / n_steps)
            poses.append((np.asarray([xy[0], xy[1], base_z], dtype=np.float32), yaw))
            if len(poses) >= max_steps:
                return poses
        current = target
    return poses


def _feasible_fraction(build, registry, primitive, grid, command_dt_s):
    """Fraction of the primitive's sub-steps the inflated grid permits from the
    current pose (the spec's non-learned kinematic veto; stands in for onboard
    local obstacle sensing in this kinematic benchmark)."""
    block = expand_primitive_to_block(registry, primitive)
    pos, quat = _current_pose(build)
    x, y = float(pos[0]), float(pos[1])
    yaw = _yaw_from_quat_wxyz(quat)
    allowed = 0
    for vx, vy, yaw_rate in block:
        nx = x + (float(vx) * math.cos(yaw) - float(vy) * math.sin(yaw)) * command_dt_s
        ny = y + (float(vx) * math.sin(yaw) + float(vy) * math.cos(yaw)) * command_dt_s
        if grid is not None and not grid.is_free((nx, ny)):
            break
        x, y = nx, ny
        yaw = wrap_angle_pi(yaw + float(yaw_rate) * command_dt_s)
        allowed += 1
    return allowed / max(len(block), 1)


def blocks_per_revolution(registry, command_dt_s, primitive="yaw_right"):
    block = expand_primitive_to_block(registry, primitive)
    per_block = abs(sum(float(t[2]) for t in block)) * command_dt_s
    return max(4, int(math.ceil(2.0 * math.pi / max(per_block, 1e-6))))


def choose_vetoed_primitive(model, build, registry, grid, command_dt_s, image, goal_image,
                            sequences, action_tensor):
    """Latent-cost ranking among kinematically feasible candidates only."""
    costs, first_primitives = _lewm_primitive_costs(model, image, goal_image, sequences, action_tensor)
    feasibility = {name: _feasible_fraction(build, registry, name, grid, command_dt_s)
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


def _render_demo_frames(pack, poses, args):
    """Pass 2: rebuild the scene with the robot visible and replay the recorded
    poses, rendering third-person + ego for the video."""
    build = build_scene_from_pack(pack, n_envs=1, backend=args.backend,
                                  show_viewer=False, render_robot=True,
                                  apply_textures=bool(args.apply_textures))
    leg_idx = _resolve_rollout_leg_dof_indices(build.robot, RolloutConfig().leg_dof_indices)
    device = torch.device("cpu")
    frames = []
    for pos, yaw in poses:
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
        third = _render_third_person(build, pos, yaw)
        ego = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
        ego_np = ego.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        frames.append((third, ego_np, float(pos[0]), float(pos[1]), float(yaw)))
    return frames


def _write_topo_demo_video(path, pack, frames, statuses, goal_xy, goal_image_np, fps, title, waypoint_thumbs=None):
    """HUD video in the repo's established demo format (title bar, third-person
    + robot-eye panels, minimap with trail) + topo-nav extras: the goal-image
    inset (the actual task input), phase/memory status, perceptual-stop banner."""
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
    goal_thumb = Image.fromarray(goal_image_np).resize((128, 128)) if goal_image_np is not None else None
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
        beacons, claimed = [], set()
        if status["phase"] == "SEEK" and goal_xy is not None:
            beacons = [(np.asarray(goal_xy, dtype=np.float64), (235, 200, 70), "goal")]
            if status.get("stopped"):
                claimed = {0}
        _draw_minimap(draw, bounds, occ, beacons, trail, (rx, ry), yaw,
                      0 if beacons and not claimed else None, claimed, 456, 372, 300, 106)
        draw.text((766, 372), "Map", fill=(195, 195, 200), font=f_lab)
        if goal_thumb is not None and status["phase"] == "SEEK":
            canvas.paste(goal_thumb, (762, 44))
            draw.rectangle([761, 43, 891, 173], outline=(235, 200, 70))
            draw.text((762, 178), "Goal image (input)", fill=(235, 200, 70), font=f_stat)
        phase_col = (120, 180, 255) if status["phase"] == "TOUR" else (235, 200, 70)
        draw.text((766, 396), f"phase: {status['phase']}", fill=phase_col, font=f_stat)
        draw.text((766, 416), f"memory: {status['nodes']} nodes", fill=(200, 200, 205), font=f_stat)
        if status["phase"] == "SEEK":
            draw.text((766, 436), f"goal dist: {status['dist']:.1f} m", fill=(200, 200, 205), font=f_stat)
        if status.get("state"):
            draw.rectangle([12, 44, 428, 66], fill=(10, 10, 13))
            draw.text((18, 48), status["state"], fill=(140, 220, 255), font=f_lab)
        draw.text((140, 462), "(kinematic base - gait not simulated)", fill=(120, 120, 130), font=f_stat)
        wp_ref = status.get("waypoint_ref")
        if wp_ref is not None and wp_ref in waypoint_thumbs and status["phase"] == "SEEK":
            canvas.paste(Image.fromarray(waypoint_thumbs[wp_ref]).resize((128, 128)), (762, 200))
            draw.rectangle([761, 199, 891, 329], outline=(140, 220, 255))
            draw.text((762, 334), "Current waypoint view", fill=(140, 220, 255), font=f_stat)
        if status.get("stopped"):
            draw.rectangle([12, 200, 428, 260], fill=(10, 40, 16), outline=(70, 230, 120), width=2)
            draw.text((30, 215), "PERCEPTUAL STOP - GOAL REACHED", fill=(70, 230, 120), font=f_title)
        out.append(np.asarray(canvas))
    hold = [out[-1]] * max(1, int(round(fps * 2)))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), out + hold, fps=fps, macro_block_size=8)


def run_scene(pack, build, model, encoder, scorer, tau_new, registry, sequences, action_tensor,
              primitive_names, args, rng, device, demo=None, runner=None):
    graph = pack.scene_graph
    grid = InflatedOccupancyGrid(graph.manifest, cell_size_m=0.05, inflation_m=0.20)
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
        navigator.update((z_raw.squeeze(0), len(stored_frames) - 1), label=int(nearest))
        tour_cell_heading[nearest] = _yaw_from_quat_wxyz(quat)

    # ---------------- Tour phase (privileged motion, learned perception) ----
    tour = graph_tour_cells(graph, start_cell, args.tour_max_cells, rng)
    poses = tour_pose_sequence(graph, tour, base_z, max_steps=args.tour_max_blocks)
    tour_blocks = 0
    for pos_xyz, yaw in poses:
        _set_pose(build=build, runner=None, pos_xyz=pos_xyz, quat_wxyz=_quat_wxyz_from_yaw(yaw))
        observe()
        tour_blocks += 1
        if demo is not None and tour_blocks % 2 == 0:
            _demo_capture(build, pack, device, demo,
                          {"phase": "TOUR", "state": "exploring (driven tour)",
                           "nodes": len(navigator.memory.nodes)})
            _set_pose(build=build, runner=None, pos_xyz=pos_xyz, quat_wxyz=_quat_wxyz_from_yaw(yaw))
    n_nodes = len(navigator.memory.nodes)
    visited_cells = sorted(tour_cell_heading)
    print(f"  tour: {tour_blocks} blocks, {len(visited_cells)} cells seen, {n_nodes} memory nodes", flush=True)

    # ---------------- Goal selection (setup-time privilege; eval contract) --
    end_pos, _ = _current_pose(build)
    end_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), end_pos[:2]))
    goal_candidates = [c for c in visited_cells
                       if (d := graph.bfs_distance(end_cell, c)) is not None
                       and args.goal_min_hops <= d <= args.goal_max_hops]
    if not goal_candidates:
        return None
    goal_cell = rng.choice(goal_candidates)
    goal_xy = graph.cell_center(goal_cell)
    goal_yaw = tour_cell_heading[goal_cell]
    goal_image = _render_tensor_from_base(
        build, pack, base_xyz_m=np.asarray([*goal_xy, base_z], dtype=np.float32),
        base_quat_wxyz=_quat_wxyz_from_yaw(goal_yaw), device=device,
    )
    z_goal_raw, _ = model.encode(goal_image[None], None)
    z_goal_raw = z_goal_raw.squeeze(0)
    if demo is not None:
        demo["goal_xy"] = (float(goal_xy[0]), float(goal_xy[1]))
        demo["goal_image"] = goal_image.mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    goal_node, goal_score = navigator.match_goal(z_goal_raw)
    majority = navigator.memory.node_majority_labels()
    start_pose = (np.asarray([*graph.cell_center(end_cell), base_z], dtype=np.float32),
                  _quat_wxyz_from_yaw(rng.uniform(-math.pi, math.pi)))
    d0 = _xy_distance(start_pose[0][:2], goal_xy)
    print(f"  goal: cell {goal_cell} at {d0:.2f} m, {graph.bfs_distance(end_cell, goal_cell)} hops; "
          f"matched node {goal_node} (score {goal_score:.3f})", flush=True)

    # ---------------- Seek phase ----------------
    results = {}
    for policy in _parse_csv(args.policies):
        _set_pose(build=build, runner=runner, pos_xyz=start_pose[0], quat_wxyz=start_pose[1])
        check_pos, _ = _current_pose(build)
        print(f"    [{policy}] start intended={start_pose[0][:2]} actual={check_pos[:2]}", flush=True)
        navigator._window.clear()           # fresh perceptual window; memory kept
        arrive_streak, used_fallback, subgoal_nodes = 0, 0, []
        max_goal_cosine, subgoal_progress, subgoal_scored = 0.0, 0, 0
        primitive_counts: dict[str, int] = {}
        committed_subgoal, blocks_on_subgoal, subgoals_reached, falls = None, 0, 0, 0
        n_scan = blocks_per_revolution(registry, args.command_dt_s)
        scan_remaining, scan_costs, scan_return, z_subgoal_proj = 0, [], 0, None
        node_path, path_index, mode, walk_blocks = None, 0, "align", 0
        align_target, align_op = 0.0, "ge"
        path_length, prev_xy = 0.0, np.asarray(start_pose[0][:2], dtype=np.float64)
        stopped_perceptually = False
        for _block in range(args.seek_max_blocks):
            pos, quat = _current_pose(build)
            image = _render_tensor_from_base(build, pack, base_xyz_m=pos, base_quat_wxyz=quat, device=device)
            z_now, _ = model.encode(image[None], None)
            z_now = z_now.squeeze(0)
            # Perceptual arrival (deployment-valid; gt only logged for eval).
            cosine = float(F.normalize(z_now, dim=-1) @ F.normalize(z_goal_raw, dim=-1))
            max_goal_cosine = max(max_goal_cosine, cosine)
            if cosine >= args.tau_arrive:
                arrive_streak += 1
                if arrive_streak >= 2:
                    stopped_perceptually = True
                    break
            else:
                arrive_streak = 0

            if policy == "topo":
                # Feed the filter ONLY during locomotion: the belief window
                # (H=8 consecutive frames) matches the training distribution
                # of smooth motion; scan rotations poison it for ~8 blocks
                # after every alignment (measured: filter never confirmed
                # arrival). The window is frozen during ALIGN.
                if mode == "walk" or node_path is None:
                    navigator.update((z_now, None))
                # ---- Stage 4b traversal: ALIGN -> WALK per directed edge ----
                # (plan_cost is flat between interior corridor views, so no
                # image cost is used between nodes; alignment = raw-frame
                # cosine to the NEXT node's keyframe, which faces the travel
                # direction by construction of directed tour edges.)
                if node_path is None or path_index >= len(node_path):
                    plan = navigator.plan_node_path(z_goal_raw)
                    if plan is None:
                        used_fallback += 1
                        primitive, _cost = choose_vetoed_primitive(
                            model, build, registry, grid, args.command_dt_s, image, goal_image,
                            sequences, action_tensor)
                        primitive_counts[primitive] = primitive_counts.get(primitive, 0) + 1
                        _execute_kinematic_primitive(build, registry, primitive,
                                                     command_dt_s=args.command_dt_s, grid=grid)
                        continue
                    node_path, _gnode, _s = plan
                    path_index, mode, walk_blocks = 1, "align", 0
                    scan_remaining, scan_costs, scan_return = n_scan, [], 0
                    subgoal_nodes.append(node_path[-1])
                    n_rev = sum(1 for i in range(1, len(node_path))
                                if (node_path[i-1], node_path[i]) not in navigator.memory.edges)
                    print(f"    path: {len(node_path)} nodes, {n_rev} reversed edges", flush=True)
                next_node = node_path[min(path_index, len(node_path) - 1)]
                at_goal_node = path_index >= len(node_path) - 1
                reversed_edge = (node_path[path_index - 1], next_node) not in navigator.memory.edges
                if mode == "align":
                    keyframe = navigator._keyframes.get(next_node)
                    cosine_k = float(F.normalize(z_now, dim=-1) @ keyframe) if keyframe is not None else 0.0
                    if (cosine_k >= 0.95 and not reversed_edge) or (scan_remaining == 0 and scan_return == 0):
                        mode, walk_blocks = "walk", 0
                        navigator._window.clear()
                        primitive = "hold"
                    elif scan_remaining > 0:
                        scan_costs.append(cosine_k)
                        scan_remaining -= 1
                        if scan_remaining == 0:
                            if runner is None:
                                # Kinematic: exact yaw per block -> open-loop
                                # return, VERBATIM the verified-success logic.
                                best = int(np.argmax(scan_costs))
                                offset = n_scan // 2 if reversed_edge else 0
                                scan_return = (best + 1 + offset) % n_scan
                                align_op = "open"
                            else:
                                # Physical: gait yaw tracking error -> CLOSED-
                                # LOOP return on the live view (reversed edge:
                                # the anti-match).
                                if reversed_edge:
                                    align_target, align_op = float(min(scan_costs)) + 0.03, "le"
                                else:
                                    align_target, align_op = float(max(scan_costs)) - 0.03, "ge"
                                scan_return = 2 * n_scan  # cap
                        primitive = "yaw_right"
                    elif align_op == "open":
                        scan_return -= 1
                        primitive = "yaw_right"
                    else:
                        hit = (cosine_k >= align_target) if align_op == "ge" else (cosine_k <= align_target)
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
                        primitive = "hold"
                    elif at_goal_node and walk_blocks >= 2:
                        # Final hop: salient goal-facing view -> plan_cost servo.
                        primitive, _cost = choose_vetoed_primitive(
                            model, build, registry, grid, args.command_dt_s, image, goal_image,
                            sequences, action_tensor)
                        walk_blocks += 1
                    else:
                        walk_blocks += 1
                        if walk_blocks > args.edge_budget:
                            node_path = None        # lost on this edge -> replan
                            primitive = "hold"
                        else:
                            walk_prim = "forward_slow" if runner is not None else "forward_medium"
                            forward_ok = _feasible_fraction(build, registry, walk_prim,
                                                            grid, args.command_dt_s) >= (0.8 if runner is not None else 0.5)
                            primitive = walk_prim if forward_ok else "yaw_right"
            elif policy == "v2":
                primitive, _cost = choose_vetoed_primitive(
                    model, build, registry, grid, args.command_dt_s, image, goal_image, sequences, action_tensor)
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
                _execute_kinematic_primitive(build, registry, primitive, command_dt_s=args.command_dt_s, grid=grid)
            new_pos, _ = _current_pose(build)
            if runner is not None and float(new_pos[2]) < 0.15:
                falls += 1
                if falls > 6:
                    print("    [fall] too many falls - aborting trial", flush=True)
                    break
                # Recover in place: stand back up at the current xy, reset the
                # gait policy state (locomotion robustness is not the thesis
                # under test; real platforms have get-up controllers).
                up = np.asarray([new_pos[0], new_pos[1], start_pose[0][2]], dtype=np.float32)
                _, cur_quat = _current_pose(build)
                cur_yaw = _yaw_from_quat_wxyz(cur_quat)
                _set_pose(build=build, runner=runner, pos_xyz=up, quat_wxyz=_quat_wxyz_from_yaw(cur_yaw))
                print(f"    [fall->recover #{falls}]", flush=True)
            if demo is not None and policy == "topo":
                if node_path is None:
                    demo_state, waypoint_ref = "REPLANNING (lost - re-localize)", None
                elif mode == "align":
                    demo_state, waypoint_ref = "ALIGN: scanning for waypoint view (4x speed)", \
                        navigator._observations.get(next_node)
                elif at_goal_node and walk_blocks > 2:
                    demo_state, waypoint_ref = "FINAL APPROACH: servo on goal image", None
                else:
                    demo_state, waypoint_ref = "WALK: forward to waypoint", \
                        navigator._observations.get(next_node)
                if waypoint_ref is not None and waypoint_ref not in demo["waypoint_thumbs"]:
                    thumb = stored_frames[waypoint_ref].mul(255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                    demo["waypoint_thumbs"][waypoint_ref] = thumb
                is_scan_frame = demo_state.startswith("ALIGN")
                demo["scan_tick"] = demo.get("scan_tick", 0) + (1 if is_scan_frame else 0)
                if not is_scan_frame or demo["scan_tick"] % 4 == 0:
                    status = {"phase": "SEEK", "state": demo_state, "waypoint_ref": waypoint_ref,
                              "nodes": len(navigator.memory.nodes),
                              "dist": _xy_distance(new_pos[:2], goal_xy)}
                    repeats = 2 if demo_state.startswith(("WALK", "FINAL")) else 1
                    for _ in range(repeats):
                        _demo_capture(build, pack, device, demo, dict(status), physical=runner is not None)
            path_length += float(np.linalg.norm(np.asarray(new_pos[:2], dtype=np.float64) - prev_xy))
            prev_xy = np.asarray(new_pos[:2], dtype=np.float64)

        final_pos, _ = _current_pose(build)
        final_distance = _xy_distance(final_pos[:2], goal_xy)
        if demo is not None and policy == "topo" and stopped_perceptually and demo["statuses"]:
            for st in demo["statuses"][-3:]:
                st["stopped"] = True
        results[policy] = {
            "initial_distance_m": d0,
            "final_distance_m": final_distance,
            "progress_m": d0 - final_distance,
            "path_length_m": path_length,
            "success_eval": bool(final_distance <= args.goal_radius_m),
            "stopped_perceptually": stopped_perceptually,
            "perceptual_stop_correct": bool(stopped_perceptually and final_distance <= args.goal_radius_m),
            "fallback_blocks": used_fallback,
            "n_subgoal_switches": len(set(subgoal_nodes)),
            "max_goal_cosine": max_goal_cosine,
            "primitive_counts": primitive_counts,
            "subgoals_reached": subgoals_reached,
            "falls": falls,
            "subgoal_progress_rate": (subgoal_progress / subgoal_scored) if subgoal_scored else None,
        }
        print(f"  [{policy}] final={final_distance:.2f} m success={results[policy]['success_eval']} "
              f"perc_stop={stopped_perceptually} fallback={used_fallback}", flush=True)

    goal_node_cell = majority.get(goal_node, (None, 0, False))[0] if goal_node is not None else None
    return {
        "scene_id": str(pack.scene_id),
        "tour_blocks": tour_blocks,
        "n_memory_nodes": n_nodes,
        "n_cells_visited": len(visited_cells),
        "goal_cell": int(goal_cell),
        "goal_hops": graph.bfs_distance(end_cell, goal_cell),
        "goal_match_node": goal_node,
        "goal_match_score": goal_score,
        "goal_match_cell_correct": bool(goal_node_cell == goal_cell),
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
    parser.add_argument("--apply-textures", action="store_true")
    parser.add_argument("--policies", default="topo,v2,bearing,hold")
    parser.add_argument("--primitive-names", default="hold,forward_slow,forward_medium,forward_fast,yaw_left,yaw_right,arc_left,arc_right")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--tau-goal", type=float, default=0.80)
    parser.add_argument("--tau-arrive", type=float, default=0.90)
    parser.add_argument("--tau-subgoal-reached", type=float, default=0.85)
    parser.add_argument("--subgoal-budget", type=int, default=26)
    parser.add_argument("--edge-budget", type=int, default=14)
    parser.add_argument("--tour-max-cells", type=int, default=24)
    parser.add_argument("--tour-max-blocks", type=int, default=260)
    parser.add_argument("--seek-max-blocks", type=int, default=200)
    parser.add_argument("--goal-min-hops", type=int, default=3)
    parser.add_argument("--goal-max-hops", type=int, default=5)
    parser.add_argument("--goal-radius-m", type=float, default=0.65)
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
    if args.mode == "physical":
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
            demo = ({"poses": [], "statuses": [], "goal_xy": None, "goal_image": None,
                     "waypoint_thumbs": {}}
                    if args.demo_video is not None else None)
            out = run_scene(pack, build, model, encoder, scorer, tau_new, registry,
                            sequences, action_tensor, primitive_names, args, trial_rng, device,
                            demo=demo, runner=runner)
            if out is not None:
                out["trial"] = trial
                all_results.append(out)
            if demo is not None and demo["poses"]:
                print(f"rendering demo replay ({len(demo['poses'])} frames)...", flush=True)
                demo_frames = _render_demo_frames(pack, demo["poses"], args)
                _write_topo_demo_video(
                    args.demo_video, pack, demo_frames, demo["statuses"],
                    demo["goal_xy"], demo["goal_image"], args.demo_fps,
                    "Topological Nav - learned map + goal-image seek (held-out maze)",
                    waypoint_thumbs=demo["waypoint_thumbs"])
                print(f"wrote demo video {args.demo_video} ({len(demo_frames)} frames)", flush=True)

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
