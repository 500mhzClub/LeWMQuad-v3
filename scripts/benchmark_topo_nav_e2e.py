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
    _set_pose,
    _xy_distance,
    _yaw_from_quat_wxyz,
)
from lewm.memory.topological_navigator import TopologicalNavigator  # noqa: E402
from lewm.models.belief_encoder import BeliefEncoder  # noqa: E402
from lewm.models.loop_closure import LoopClosureHead  # noqa: E402
from lewm_genesis.lewm_contract import PrimitiveRegistry, expand_primitive_to_block  # noqa: E402
from lewm_genesis.collectors.base import wrap_angle_pi  # noqa: E402
from lewm_genesis.scene_builder import build_scene_from_pack  # noqa: E402
from lewm_genesis.scene_loader import find_scene_dirs, load_platform_manifest, load_scene_pack  # noqa: E402
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


def run_scene(pack, build, model, encoder, scorer, tau_new, registry, sequences, action_tensor,
              primitive_names, args, rng, device):
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
        _set_pose(build=build, runner=None, pos_xyz=start_pose[0], quat_wxyz=start_pose[1])
        check_pos, _ = _current_pose(build)
        print(f"    [{policy}] start intended={start_pose[0][:2]} actual={check_pos[:2]}", flush=True)
        navigator._window.clear()           # fresh perceptual window; memory kept
        arrive_streak, used_fallback, subgoal_nodes = 0, 0, []
        max_goal_cosine, subgoal_progress, subgoal_scored = 0.0, 0, 0
        primitive_counts: dict[str, int] = {}
        committed_subgoal, blocks_on_subgoal, subgoals_reached = None, 0, 0
        n_scan = blocks_per_revolution(registry, args.command_dt_s)
        scan_remaining, scan_costs, scan_return, z_subgoal_proj = 0, [], 0, None
        scan_needed = True  # initial orientation is unknown
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
                navigator.update((z_now, None))
                # --- state machine: SCAN (when lost) -> ALIGN -> SERVO ---
                if scan_remaining > 0:
                    _zr, z_now_proj = model.encode(image[None], None)
                    scan_costs.append(float((z_now_proj - z_subgoal_proj).square().sum()))
                    scan_remaining -= 1
                    if scan_remaining == 0:
                        best = int(np.argmin(scan_costs))
                        scan_return = (best + 1) % n_scan
                    primitive = "yaw_right"
                elif scan_return > 0:
                    scan_return -= 1
                    primitive = "yaw_right"
                else:
                    # Sub-goal accounting only while servoing (spec 6.2:
                    # reached = the filter localizes to the sub-goal node).
                    reached = False
                    if committed_subgoal is not None:
                        posterior = navigator.current_belief()
                        reached = (navigator._last_map == committed_subgoal
                                   or posterior.get(committed_subgoal, 0.0) >= 0.5)
                        if reached:
                            subgoals_reached += 1
                    exhausted = blocks_on_subgoal >= args.subgoal_budget
                    if committed_subgoal is None or reached or exhausted:
                        scan_needed = scan_needed or exhausted or committed_subgoal is None
                        plan = navigator.plan_to_goal_latent(z_goal_raw, lookahead=args.lookahead)
                        if plan is not None:
                            committed_subgoal, _gnode, _s = plan
                            blocks_on_subgoal = 0
                            subgoal_nodes.append(committed_subgoal)
                            sg_label = majority.get(committed_subgoal)
                            cur_cell = min(cells, key=lambda c: _xy_distance(graph.cell_center(c), pos[:2]))
                            if sg_label is not None:
                                d_cur = graph.bfs_distance(cur_cell, goal_cell)
                                d_sg = graph.bfs_distance(int(sg_label[0]), goal_cell)
                                if d_cur is not None and d_sg is not None:
                                    subgoal_scored += 1
                                    subgoal_progress += int(d_sg < d_cur)
                        else:
                            committed_subgoal = None
                    blocks_on_subgoal += 1
                    if committed_subgoal is not None:
                        ref = navigator._observations.get(committed_subgoal)
                        subgoal_image = stored_frames[ref] if ref is not None else goal_image
                    else:
                        used_fallback += 1
                        subgoal_image = goal_image
                    if scan_needed:
                        scan_needed = False
                        scan_remaining, scan_costs, scan_return = n_scan, [], 0
                        _zr, z_subgoal_proj = model.encode(subgoal_image[None], None)
                        primitive = "yaw_right"
                    else:
                        primitive, _cost = choose_vetoed_primitive(
                            model, build, registry, grid, args.command_dt_s, image, subgoal_image, sequences, action_tensor)
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
            _execute_kinematic_primitive(build, registry, primitive, command_dt_s=args.command_dt_s, grid=grid)
            new_pos, _ = _current_pose(build)
            path_length += float(np.linalg.norm(np.asarray(new_pos[:2], dtype=np.float64) - prev_xy))
            prev_xy = np.asarray(new_pos[:2], dtype=np.float64)

        final_pos, _ = _current_pose(build)
        final_distance = _xy_distance(final_pos[:2], goal_xy)
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
    parser.add_argument("--tour-max-cells", type=int, default=24)
    parser.add_argument("--tour-max-blocks", type=int, default=260)
    parser.add_argument("--seek-max-blocks", type=int, default=200)
    parser.add_argument("--goal-min-hops", type=int, default=3)
    parser.add_argument("--goal-max-hops", type=int, default=5)
    parser.add_argument("--goal-radius-m", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260609)
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
    args.command_dt_s = float(platform.get("timing", {}).get("command_dt_s", 0.10))
    primitive_names = _parse_csv(args.primitive_names)
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
        for trial in range(args.goals_per_scene):
            trial_rng = random.Random(args.seed + trial * 1009 + zlib.crc32(pack.scene_id.encode()) % 100000)
            out = run_scene(pack, build, model, encoder, scorer, tau_new, registry,
                            sequences, action_tensor, primitive_names, args, trial_rng, device)
            if out is not None:
                out["trial"] = trial
                all_results.append(out)

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
