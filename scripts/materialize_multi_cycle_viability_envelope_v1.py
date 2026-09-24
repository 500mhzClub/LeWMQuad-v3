#!/usr/bin/env python3
"""Materialise predecessor envelopes and bounded oracle viability rollouts.

No learned safety, planning, or JEPA model is imported.  The frozen low-level
locomotion policy remains the simulator controller.  Decisions use exact
Genesis physics-rate contact, a two-level one-tick viability tree, and the
frozen H3 kinematic route ordering.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import multi_cycle_viability_envelope_v1 as REDUCE
from lewm.safety import control_commitment_horizon_and_viability_v1 as KINEMATIC
from scripts import collect_factorised_micro_safety_world_model_v1 as EMBODIED
from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as NARROW
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep


SOURCE_COMMIT = "8ab19f4816aec7461072f45f48fd9a6f7ceac81e"
OUT = ROOT / ".generated/multi_cycle_viability_envelope_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/multi_cycle_viability_envelope_v1")
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
OLD_TREE = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/viability_tree_index.json"
OLD_RESULT = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/viability_result.json"
GEOMETRY_INDEX = ROOT / ".generated/h1_articulated_swept_geometry_sufficiency_v1/articulated_geometry_index.json"
SENSOR_INDEX = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
FIXTURE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
SELECTION = OUT / "frozen_state_selection.json"
PHYSICS_STEPS = 50
MAX_PREDECESSOR_DEPTH = 10
MAX_ROLLOUT_CYCLES = 10
IMPLEMENTATION_REVISION = 3


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _index(path: Path) -> dict[str, dict]:
    return {str(row["state_id"]): row for row in json.loads(path.read_text())["state_records"]}


def _quat_yaw_rate(current: np.ndarray) -> float:
    # The frozen enhanced current state contains the signed IMU z gyro.
    return float(current[5])


def _initial_speed(link_transform: np.ndarray) -> float:
    # Before candidate trajectories diverge, the first 2 ms base displacement
    # is an outcome-independent estimate of boundary planar speed.  Median over
    # all twelve registered candidates suppresses numerical noise.
    delta = link_transform[:, 1, 0, :2] - link_transform[:, 0, 0, :2]
    return float(np.median(np.linalg.norm(delta, axis=1)) / KINEMATIC.PHYSICS_DT_S)


def freeze_selection() -> dict:
    if SELECTION.is_file():
        payload = json.loads(SELECTION.read_text())
        expected = REDUCE.digest({key: value for key, value in payload.items() if key != "content_digest"})
        if payload.get("content_digest") != expected:
            raise RuntimeError("frozen selection digest mismatch")
        print(json.dumps({"status": "REUSED", "controls": payload["control_state_ids"],
                          "content_digest": payload["content_digest"]}, indent=2))
        return payload

    panel = json.loads(PANEL.read_text())
    tree = json.loads(OLD_TREE.read_text())
    result = json.loads(OLD_RESULT.read_text())
    state_manifest = {str(row["state_id"]): row for row in panel["states"]}
    tree_map = {str(row["state_id"]): row for row in tree["state_records"]}
    geometry_map = _index(GEOMETRY_INDEX)
    sensor_map = _index(SENSOR_INDEX)
    failed = {
        str(row["state_id"]): str(row["classification"])
        for row in result["current_state_viability"]["per_state"]
        if not row["viability_admissible_candidate_indices"]
    }
    if len(failed) != 8:
        raise RuntimeError(f"expected eight frozen failures, found {len(failed)}")

    features = []
    for state_id, state in sorted(state_manifest.items()):
        geometry_record = geometry_map[state_id]
        sensor_record = sensor_map[state_id]
        if sha(Path(geometry_record["shard_path"])) != geometry_record["shard_sha256"]:
            raise RuntimeError(f"geometry binding mismatch: {state_id}")
        if sha(Path(sensor_record["shard_path"])) != sensor_record["shard_sha256"]:
            raise RuntimeError(f"sensor binding mismatch: {state_id}")
        with np.load(geometry_record["shard_path"], allow_pickle=False) as loaded:
            speed = _initial_speed(np.asarray(loaded["link_transform"], np.float64))
        with np.load(sensor_record["shard_path"], allow_pickle=False) as loaded:
            current = np.asarray(loaded["current"][0], np.float64)
            action = np.asarray(loaded["action_control"][0, 0], np.float64)
        tree_record = tree_map[state_id]
        features.append({
            "state_id": state_id,
            "family": state["family"],
            "split": state["split"],
            "role": "failure" if state_id in failed else "control_candidate",
            "failure_classification": failed.get(state_id),
            "initial_planar_speed_m_s": speed,
            "initial_yaw_rate_rad_s": _quat_yaw_rate(current),
            "waypoint_distance_m": float(math.hypot(*state["waypoint_body_xy"])),
            "one_tick_safe_candidate_count": int(sum(row["safe_prefix"] for row in tree_record["current"])),
            "current_applied_command": action[3:6].tolist(),
        })

    controls: list[str] = []
    matching = []
    families = sorted({row["family"] for row in features})
    scales = np.asarray([0.10, 0.20, 0.50, 3.0], np.float64)
    for family in families:
        failures = [row for row in features if row["family"] == family and row["role"] == "failure"]
        candidates = [row for row in features if row["family"] == family and row["role"] == "control_candidate"]
        if not failures or len(candidates) < 2:
            raise RuntimeError(f"cannot bind two matched controls for {family}")
        target = np.mean([
            [row["initial_planar_speed_m_s"], row["initial_yaw_rate_rad_s"],
             row["waypoint_distance_m"], row["one_tick_safe_candidate_count"]]
            for row in failures
        ], axis=0)
        scored = []
        for row in candidates:
            value = np.asarray([
                row["initial_planar_speed_m_s"], row["initial_yaw_rate_rad_s"],
                row["waypoint_distance_m"], row["one_tick_safe_candidate_count"],
            ], np.float64)
            distance = float(np.sum(np.square((value - target) / scales)))
            scored.append((distance, row["state_id"], row, value))
        for distance, state_id, row, value in sorted(scored)[:2]:
            controls.append(state_id)
            matching.append({
                "family": family,
                "state_id": state_id,
                "squared_scaled_distance": distance,
                "target": target.tolist(),
                "features": value.tolist(),
            })

    payload = {
        "schema": "multi_cycle_viability_envelope_frozen_selection_v1",
        "source_commit": SOURCE_COMMIT,
        "frozen_before_multi_cycle_execution": True,
        "failure_state_ids": sorted(failed),
        "failure_classifications": failed,
        "control_state_ids": controls,
        "control_matching": matching,
        "all_pre_outcome_features": features,
        "matching_contract": {
            "controls_per_family": 2,
            "variables": ["initial_planar_speed_m_s", "initial_yaw_rate_rad_s",
                          "waypoint_distance_m", "one_tick_safe_candidate_count"],
            "fixed_scales": scales.tolist(),
            "selection": "lowest squared scaled distance to family failure centroid; state-id tie",
            "future_multi_cycle_outcomes_opened": False,
        },
        "bindings": {
            "panel_sha256": sha(PANEL),
            "old_tree_sha256": sha(OLD_TREE),
            "old_result_sha256": sha(OLD_RESULT),
            "geometry_index_sha256": sha(GEOMETRY_INDEX),
            "sensor_index_sha256": sha(SENSOR_INDEX),
        },
    }
    payload["content_digest"] = REDUCE.digest(payload)
    atomic_json(SELECTION, payload)
    print(json.dumps({"status": "FROZEN", "failures": payload["failure_state_ids"],
                      "controls": controls, "content_digest": payload["content_digest"]}, indent=2))
    return payload


def _advance_historical_tick(ctx, target: np.ndarray) -> None:
    ctx.runner._step_command_tick(target)
    ONE._advance_tick_counters(ctx, target)


def historical_snapshots(state: dict) -> tuple[object, dict[int, object], dict]:
    """Reconstruct depths 10..0 in one deterministic forward traversal."""

    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu",
                             shared=V.V1._load_shared("cpu"))
    ctx.begin_episode()
    current_tick = int(state["warmup_blocks"]) * int(ctx.runner._block_size)
    start_tick = current_tick - MAX_PREDECESSOR_DEPTH
    for _ in range(start_tick // int(ctx.runner._block_size)):
        ctx.drive_one_block()
    snapshots: dict[int, object] = {}
    active_block = None
    for absolute_tick in range(start_tick, current_tick + 1):
        depth = current_tick - absolute_tick
        snapshots[depth] = ONE._capture_tick_boundary(
            ctx,
            goal=dict(state["goal"]),
            identity={"state_id": f"{state['state_id']}:pred:{depth:02d}",
                      "scene_id": state["scene_id"], "family": state["family"]},
        )
        if absolute_tick == current_tick:
            break
        phase = absolute_tick % int(ctx.runner._block_size)
        if phase == 0:
            requested, _ = ctx.runner._collect_block()
            active_block = np.asarray(ctx.runner._clip_block(requested).executed, np.float32)
        if active_block is None:
            raise RuntimeError("historical block was not bound")
        _advance_historical_tick(ctx, active_block[:, phase])
    canonical_ctx, canonical = ONE._build_current(state)
    current_pose = ctx.pose()
    canonical_pose = canonical_ctx.pose()
    current_match = bool(np.allclose(np.asarray(current_pose[0] + (current_pose[1], current_pose[2])),
                                     np.asarray(canonical_pose[0] + (canonical_pose[1], canonical_pose[2])),
                                     atol=1e-8, rtol=0.0))
    del canonical_ctx
    if not current_match:
        raise RuntimeError(f"historical predecessor reconstruction mismatch: {state['state_id']}")
    return ctx, snapshots, {"current_pose_match": current_match,
                            "snapshots": len(snapshots), "depths": sorted(snapshots)}


def _route_contract(ctx, snapshot) -> dict:
    ONE._restore_tick_boundary(ctx, snapshot)
    pose = ctx.pose()
    hit = ctx.scene_graph.locate(pose[0])
    landmark = int(snapshot.goal["landmark_cell"])
    path = ctx.scene_graph.shortest_path(int(hit.cell_id), landmark)
    if path is None or not path:
        raise RuntimeError(f"no route at {snapshot.identity['state_id']}")
    waypoint_cell = int(path[min(2, len(path) - 1)])
    waypoint = tuple(map(float, ctx.scene_graph.cell_center(waypoint_cell)))
    body = EMBODIED.body_relative(pose, waypoint)
    if len(path) >= 2:
        heading = EMBODIED.route_heading(ctx, list(map(int, path)))
    else:
        heading = float(pose[1])
    return {
        "pose": [[float(pose[0][0]), float(pose[0][1])], float(pose[1]), float(pose[2])],
        "cell_id": int(hit.cell_id), "path": list(map(int, path)),
        "waypoint_cell": waypoint_cell, "waypoint_xy": list(waypoint),
        "waypoint_body_xy": list(map(float, body)), "route_heading_world_rad": float(heading),
    }


def _integrate(commands: np.ndarray, waypoint_body: list[float], heading_body: float) -> tuple[float, float]:
    x = y = yaw = 0.0
    for vx, vy, wz in np.asarray(commands, np.float64):
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * 0.1
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * 0.1
        yaw = KINEMATIC.wrap(yaw + wz * 0.1)
    wx, wy = waypoint_body
    progress = math.hypot(wx, wy) - math.hypot(wx - x, wy - y)
    heading = abs(KINEMATIC.wrap(heading_body)) - abs(KINEMATIC.wrap(heading_body - yaw))
    return float(progress), float(heading)


def _h3_plan(ctx, snapshot, candidate_index: int, route: dict) -> dict:
    ONE._restore_tick_boundary(ctx, snapshot)
    name, primitives = V.V1.CANDIDATE_BANK[candidate_index]
    executed = []
    for primitive in primitives[:3]:
        requested = V.V1.block_for(primitive)[None, ...]
        executed.append(np.asarray(ctx.runner._clip_block(requested).executed, np.float32)[0])
    commands = np.concatenate(executed, axis=0)
    heading_body = float(route["route_heading_world_rad"]) - float(route["pose"][1])
    progress, heading = _integrate(commands, route["waypoint_body_xy"], heading_body)
    return {"candidate": name, "h3_progress_m": progress,
            "h3_heading_improvement_rad": heading, "post_slew_h3": commands.tolist()}


def _current_features(ctx, snapshot, route: dict, topology, link_names, object_names) -> dict:
    ONE._restore_tick_boundary(ctx, snapshot)
    runner = ctx.runner
    robot = ctx.build.robot
    linear = np.asarray(runner._as_np(robot.get_vel()), np.float64).reshape(-1, 3)[0]
    angular = np.asarray(runner._as_np(robot.get_ang()), np.float64).reshape(-1, 3)[0]
    dofs = list(runner.policy.policy_dof_indices)
    qpos = np.asarray(runner._as_np(robot.get_dofs_position(dofs)), np.float64).reshape(-1)
    qvel = np.asarray(runner._as_np(robot.get_dofs_velocity(dofs)), np.float64).reshape(-1)
    embodied, _ = SENSOR.sensor_state(runner, previous_joint_velocity=None)
    native = NARROW.contact_summary(robot.get_contacts(), topology, link_names, object_names,
                                    force_threshold=True)
    scene_clearance = float(ctx.scene_graph.clearance_to_walls(route["pose"][0]))
    return {
        "current_applied_command": np.asarray(runner._last_executed, np.float32)[0].tolist(),
        "command_history_available": True,
        "base_planar_speed_m_s": float(np.linalg.norm(linear[:2])),
        "base_linear_velocity_m_s": linear.tolist(),
        "base_angular_velocity_rad_s": angular.tolist(),
        "joint_position_rad": qpos.tolist(), "joint_velocity_rad_s": qvel.tolist(),
        "enhanced_embodied_state": np.asarray(embodied, np.float32).tolist(),
        "front_depth": None, "lidar": None,
        "sensor_availability": {
            "front_depth": "not persisted at reconstructed predecessor/dynamic boundary",
            "lidar": "not persisted at reconstructed predecessor/dynamic boundary",
        },
        "genesis_exact_current_contact": bool(native["active"]),
        "genesis_contact_penetration_m": None if not np.isfinite(native["penetration"])
        else float(native["penetration"]),
        "exact_positive_clearance_available": False,
        "scene_graph_wall_clearance_m": scene_clearance,
        "exact_clearance_limitation": "Genesis 0.3.14 exact path exposes manifold contact/penetration, not positive pair distance",
    }


def boundary_tree(ctx, snapshot, *, identity: str) -> tuple[dict, dict[int, object]]:
    topology = V.link_topology(ctx)
    solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    route = _route_contract(ctx, snapshot)
    start_pose = route["pose"]
    start_distance = math.hypot(route["waypoint_xy"][0] - start_pose[0][0],
                                route["waypoint_xy"][1] - start_pose[0][1])
    start_heading_error = abs(KINEMATIC.wrap(route["route_heading_world_rad"] - start_pose[1]))
    features = _current_features(ctx, snapshot, route, topology, link_names, object_names)
    rows = []
    current_successors: dict[int, object] = {}
    successor_branches = 0
    for candidate_index in range(12):
        plan = _h3_plan(ctx, snapshot, candidate_index, route)
        current, contact, _robot, _other, successor = ONE._execute_one_tick(
            ctx, snapshot, candidate_index, topology, link_names, object_names
        )
        endpoint = current["endpoint_pose"]
        end_distance = math.hypot(route["waypoint_xy"][0] - endpoint[0][0],
                                  route["waypoint_xy"][1] - endpoint[0][1])
        end_heading_error = abs(KINEMATIC.wrap(route["route_heading_world_rad"] - endpoint[1]))
        safe_prefix = not bool(contact.any())
        safe_successors = []
        successor_rows = []
        if safe_prefix:
            current_successors[candidate_index] = successor
            for second_index in range(12):
                second, second_contact, _r, _o, _ = ONE._execute_one_tick(
                    ctx, successor, second_index, topology, link_names, object_names
                )
                safe_second = not bool(second_contact.any())
                if safe_second:
                    safe_successors.append(second_index)
                successor_rows.append({
                    "successor_candidate": second_index,
                    "contact": bool(second_contact.any()),
                    "first_contact_step": second["first_contact_step"],
                })
                successor_branches += 1
        row = {
            "candidate_index": candidate_index, "candidate": current["candidate"],
            "first_tick_contact": bool(contact.any()),
            "first_contact_step": current["first_contact_step"],
            "safe_prefix": safe_prefix, "successor_safe_action_count": len(safe_successors),
            "successor_safe_candidate_indices": safe_successors,
            "viable": bool(safe_successors), "admissible": bool(safe_prefix and safe_successors),
            "immediate_progress_m": float(start_distance - end_distance),
            "immediate_heading_improvement_rad": float(start_heading_error - end_heading_error),
            "endpoint_pose": endpoint, "termination": current["termination"],
            "successor_identity": f"{identity}:succ:{candidate_index:02d}",
            "successor_rows": successor_rows,
            **plan,
        }
        rows.append(row)
    features.update({
        "safe_one_tick_candidate_count": sum(row["safe_prefix"] for row in rows),
        "viability_admissible_candidate_count": sum(row["admissible"] for row in rows),
        "minimum_successor_safe_action_count": min(
            (row["successor_safe_action_count"] for row in rows if row["safe_prefix"]), default=0
        ),
        "time_to_first_candidate_contact_s": min(
            (float(row["first_contact_step"]) * 0.002 for row in rows
             if row["first_contact_step"] is not None), default=None
        ),
    })
    return {
        "identity": identity, "snapshot_digest": snapshot.digest,
        "route": route, "state_variables": features, "candidates": rows,
        "current_prefix_branches": 12, "successor_branches": successor_branches,
        "safe_prefix_count": sum(row["safe_prefix"] for row in rows),
        "viability_admissible_count": sum(row["admissible"] for row in rows),
    }, current_successors


def _selected_index(tree: dict) -> int | None:
    rows = [row for row in tree["candidates"] if row["admissible"]]
    if not rows:
        return None
    order = REDUCE.route_order(rows)
    return int(rows[order[0]]["candidate_index"])


def rollout(ctx, start_snapshot, *, state_id: str, role: str, start_depth: int) -> dict:
    snapshot = start_snapshot
    selected_rows = []
    current_branches = successor_branches = 0
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(ctx.runner.episode_states[0].episode_id)
    initial_pose = None
    for cycle in range(MAX_ROLLOUT_CYCLES):
        tree, snapshots = boundary_tree(ctx, snapshot, identity=f"{state_id}:roll:{cycle:02d}")
        current_branches += tree["current_prefix_branches"]
        successor_branches += tree["successor_branches"]
        if initial_pose is None:
            initial_pose = copy.deepcopy(tree["route"]["pose"])
        selected_index = _selected_index(tree)
        if selected_index is None:
            selected_rows.append({
                "cycle": cycle, "selected_candidate": None, "abstained": True,
                "safe_prefix_count": tree["safe_prefix_count"],
                "viability_admissible_count": tree["viability_admissible_count"],
                "tree": tree,
            })
            break
        selected = next(row for row in tree["candidates"] if row["candidate_index"] == selected_index)
        snapshot = snapshots[selected_index]
        ONE._restore_tick_boundary(ctx, snapshot)
        pose = ctx.pose()
        command = selected["target_command"] if "target_command" in selected else selected["post_slew_h3"][0]
        derived = label_computer.step(PoseStep(
            timestamp_ns=int(ctx.runner._sim_time_ns), env_idx=0, episode_id=episode_id,
            episode_step=int(ctx.runner.episode_states[0].episode_step),
            position_xy_world=(float(pose[0][0]), float(pose[0][1])), yaw_world_rad=float(pose[1]),
            last_command=tuple(float(value) for value in command),
        ))
        selected_rows.append({
            "cycle": cycle, "selected_candidate": selected_index,
            "selected_candidate_name": selected["candidate"], "abstained": False,
            "selected_first_tick_contact": selected["first_tick_contact"],
            "selected_successor_viable": selected["viable"],
            "selected_successor_safe_action_count": selected["successor_safe_action_count"],
            "safe_prefix_count": tree["safe_prefix_count"],
            "viability_admissible_count": tree["viability_admissible_count"],
            "immediate_progress_m": selected["immediate_progress_m"],
            "immediate_heading_improvement_rad": selected["immediate_heading_improvement_rad"],
            "h3_progress_m": selected["h3_progress_m"],
            "h3_heading_improvement_rad": selected["h3_heading_improvement_rad"],
            "reverse_progress": bool(selected["immediate_progress_m"] < 0.0),
            "stuck": bool(derived.stuck_label), "termination": selected["termination"],
            "waypoint_completed": bool(math.hypot(
                tree["route"]["waypoint_xy"][0] - pose[0][0],
                tree["route"]["waypoint_xy"][1] - pose[0][1],
            ) <= 0.35),
            "tree": tree,
        })
        if selected["first_tick_contact"] or any(selected["termination"].values()) or selected_rows[-1]["waypoint_completed"]:
            break
    return {
        "role": role, "start_depth": start_depth, "cycles": len(selected_rows),
        "completed_cycles": sum(not row["abstained"] for row in selected_rows),
        "selected": selected_rows,
        "selected_first_tick_contacts": sum(bool(row.get("selected_first_tick_contact")) for row in selected_rows),
        "transitions_to_nonviable_successor": sum(
            row.get("selected_successor_viable") is False for row in selected_rows
        ),
        "minimum_selected_successor_safe_actions": min(
            (int(row["selected_successor_safe_action_count"]) for row in selected_rows
             if row.get("selected_successor_safe_action_count") is not None), default=0
        ),
        "distance_progress_m": float(sum(float(row.get("immediate_progress_m", 0.0)) for row in selected_rows)),
        "heading_improvement_rad": float(sum(float(row.get("immediate_heading_improvement_rad", 0.0)) for row in selected_rows)),
        "reverse_progress_cycles": sum(bool(row.get("reverse_progress")) for row in selected_rows),
        "stuck_cycles": sum(bool(row.get("stuck")) for row in selected_rows),
        "abstained": bool(selected_rows and selected_rows[-1]["abstained"]),
        "waypoint_completed": any(bool(row.get("waypoint_completed")) for row in selected_rows),
        "fall_or_unsafe_termination": any(
            any(row.get("termination", {}).values()) for row in selected_rows
        ),
        "current_prefix_branches": current_branches,
        "successor_branches": successor_branches,
        "initial_pose": initial_pose,
    }


def run_fixture() -> dict:
    pure = REDUCE.fixture_payload()
    if not pure["pass"] or pure != REDUCE.fixture_payload():
        raise RuntimeError("pure fixture failure")
    fixture_states = {row["state_id"]: row for row in json.loads(FIXTURE_PANEL.read_text())["states"]}
    state = fixture_states["scale-fit-0-00"]
    ctx, snapshot = ONE._build_current(state)
    topology = V.link_topology(ctx)
    solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    first, c1, _r1, _o1, successor1 = ONE._execute_one_tick(
        ctx, snapshot, 11, topology, link_names, object_names
    )
    second, c2, _r2, _o2, successor2 = ONE._execute_one_tick(
        ctx, snapshot, 11, topology, link_names, object_names
    )
    deterministic = bool(np.array_equal(c1, c2) and first["endpoint_pose"] == second["endpoint_pose"]
                         and successor1.digest == successor2.digest)
    result = {
        "schema": "multi_cycle_viability_envelope_fixture_result_v1",
        "pure": pure, "training_state": state["state_id"],
        "snapshot_restore_deterministic": deterministic,
        "safe_prefix_viable_successor": not bool(c1.any()),
        "complete_row_serialization": True,
        "pass": bool(pure["pass"] and deterministic),
    }
    result["content_digest"] = REDUCE.digest(result)
    del ctx
    gc.collect()
    if not result["pass"]:
        raise RuntimeError("Genesis fixture failure")
    return result


def collect_state(state_id: str) -> dict:
    selection = freeze_selection()
    selected_ids = set(selection["failure_state_ids"] + selection["control_state_ids"])
    if state_id not in selected_ids:
        raise ValueError(f"state is not frozen for this experiment: {state_id}")
    output = OUT / "states" / f"{state_id}.json"
    if output.is_file():
        record = json.loads(output.read_text())
        if (record.get("status") == "PASS"
                and record.get("implementation_revision") == IMPLEMENTATION_REVISION
                and record.get("content_digest") == REDUCE.digest(
            {key: value for key, value in record.items() if key != "content_digest"}
        )):
            print(json.dumps({"state_id": state_id, "status": "REUSED"}), flush=True)
            return record
    started = time.time()
    panel = {row["state_id"]: row for row in json.loads(PANEL.read_text())["states"]}
    state = panel[state_id]
    role = "failure" if state_id in selection["failure_state_ids"] else "matched_control"
    predecessor_rows = []
    reconstruction = None
    if role == "failure":
        ctx, snapshots, reconstruction = historical_snapshots(state)
        consecutive = 0
        for depth in range(1, MAX_PREDECESSOR_DEPTH + 1):
            tree, _ = boundary_tree(ctx, snapshots[depth], identity=f"{state_id}:pred:{depth:02d}")
            tree["depth"] = depth
            predecessor_rows.append(tree)
            consecutive = consecutive + 1 if tree["viability_admissible_count"] else 0
            if consecutive >= 3:
                break
        compact = [{"depth": row["depth"],
                    "viability_admissible_count": row["viability_admissible_count"]}
                   for row in predecessor_rows]
        stable_depth = REDUCE.stable_predecessor_depth(compact)
        old_class = selection["failure_classifications"][state_id]
        unavoidable = bool(
            stable_depth is None
            and old_class == "CONTACT_BEFORE_CONTROL_AUTHORITY"
            and predecessor_rows
            and all(row["safe_prefix_count"] == 0 for row in predecessor_rows)
            and all(
                max((candidate["first_contact_step"] or 10**6) for candidate in row["candidates"]) <= 2
                for row in predecessor_rows
            )
        )
        classification = REDUCE.intervention_class(compact, contact_already_unavoidable=unavoidable)
        viable_depths = [row["depth"] for row in predecessor_rows if row["viability_admissible_count"]]
        # Start at the closest boundary that begins a verified three-boundary
        # viable envelope, not at an isolated later viable point.
        start_depth = stable_depth
        if start_depth is not None:
            multi = rollout(ctx, snapshots[start_depth], state_id=state_id, role=role,
                            start_depth=start_depth)
        else:
            multi = None
    else:
        ctx, snapshot = ONE._build_current(state)
        snapshots = {0: snapshot}
        stable_depth = start_depth = 0
        classification = "MATCHED_VIABILITY_POSITIVE_CONTROL"
        multi = rollout(ctx, snapshot, state_id=state_id, role=role, start_depth=0)

    record = {
        "schema": "multi_cycle_viability_envelope_state_v1", "status": "PASS",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_commit": SOURCE_COMMIT, "state_id": state_id, "scene_id": state["scene_id"],
        "family": state["family"], "split": state["split"], "role": role,
        "predecessor_reconstruction": reconstruction,
        "predecessors": predecessor_rows,
        "first_contact_free_depth": next((row["depth"] for row in predecessor_rows
                                          if row["safe_prefix_count"]), None),
        "first_viability_depth": next((row["depth"] for row in predecessor_rows
                                       if row["viability_admissible_count"]), None),
        "stable_predecessor_depth": stable_depth,
        "required_lead_time_ticks": start_depth if role == "failure" else 0,
        "required_lead_time_s": None if start_depth is None else float(start_depth * 0.1),
        "failure_classification": classification,
        "multi_cycle_rollout": multi,
        "runtime_s": time.time() - started,
    }
    record["content_digest"] = REDUCE.digest(record)
    atomic_json(output, record)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": state_id, "status": "PASS", "role": role,
                      "predecessors": len(predecessor_rows), "classification": classification,
                      "rollout_cycles": None if multi is None else multi["completed_cycles"],
                      "runtime_s": record["runtime_s"]}), flush=True)
    return record


def collect_all() -> None:
    selection = freeze_selection()
    started = time.time()
    fixture_started = time.time()
    fixture = run_fixture()
    atomic_json(OUT / "fixture.json", fixture)
    fixture_s = time.time() - fixture_started
    state_ids = selection["failure_state_ids"] + selection["control_state_ids"]
    logs = CACHE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(state_ids), 4):
        jobs = []
        for state_id in state_ids[start:start + 4]:
            log = logs / f"{state_id}.log"
            stream = log.open("wb")
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--collect-state", state_id],
                stdout=stream, stderr=subprocess.STDOUT,
            )
            jobs.append((state_id, process, stream, log))
        for state_id, process, stream, log in jobs:
            code = process.wait()
            stream.close()
            if code:
                raise RuntimeError(f"{state_id} failed with {code}; see {log}")
    atomic_json(OUT / "collection_receipt.json", {
        "states": state_ids, "parallel_processes": 4, "fixture_runtime_s": fixture_s,
        "wall_runtime_s": time.time() - started,
    })


def finalize() -> dict:
    selection = freeze_selection()
    records = []
    for state_id in selection["failure_state_ids"] + selection["control_state_ids"]:
        path = OUT / "states" / f"{state_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing state record: {state_id}")
        record = json.loads(path.read_text())
        expected = REDUCE.digest({key: value for key, value in record.items() if key != "content_digest"})
        if record.get("status") != "PASS" or record.get("content_digest") != expected:
            raise RuntimeError(f"invalid state record: {state_id}")
        records.append(record)
    receipt = json.loads((OUT / "collection_receipt.json").read_text())
    payload = {
        "schema": "multi_cycle_viability_envelope_index_v1", "source_commit": SOURCE_COMMIT,
        "selection": selection, "fixture": json.loads((OUT / "fixture.json").read_text()),
        "state_records": records,
        "counts": {
            "failure_states": sum(row["role"] == "failure" for row in records),
            "matched_controls": sum(row["role"] == "matched_control" for row in records),
            "predecessor_boundaries": sum(len(row["predecessors"]) for row in records),
            "predecessor_current_prefixes": sum(
                sum(p["current_prefix_branches"] for p in row["predecessors"]) for row in records
            ),
            "predecessor_successor_branches": sum(
                sum(p["successor_branches"] for p in row["predecessors"]) for row in records
            ),
            "rollout_current_prefixes": sum(
                row["multi_cycle_rollout"]["current_prefix_branches"]
                for row in records if row["multi_cycle_rollout"] is not None
            ),
            "rollout_successor_branches": sum(
                row["multi_cycle_rollout"]["successor_branches"]
                for row in records if row["multi_cycle_rollout"] is not None
            ),
        },
        "runtime": {
            "wall_s": receipt["wall_runtime_s"], "fixture_s": receipt["fixture_runtime_s"],
            "state_compute_s": sum(float(row["runtime_s"]) for row in records),
        },
        "bindings": {
            "selection_sha256": sha(SELECTION), "old_tree_sha256": sha(OLD_TREE),
            "old_result_sha256": sha(OLD_RESULT),
        },
    }
    payload["content_digest"] = REDUCE.digest(payload)
    atomic_json(OUT / "multi_cycle_index.json", payload)
    print(json.dumps({"counts": payload["counts"], "runtime": payload["runtime"],
                      "content_digest": payload["content_digest"]}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-selection", action="store_true")
    group.add_argument("--fixtures", action="store_true")
    group.add_argument("--collect-state")
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.freeze_selection:
        freeze_selection()
    elif args.fixtures:
        result = run_fixture()
        atomic_json(OUT / "fixture.json", result)
        print(json.dumps(result, indent=2))
    elif args.collect_state:
        collect_state(args.collect_state)
    elif args.collect_all:
        collect_all()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
