#!/usr/bin/env python3
"""Execute the frozen emergency-brake successor on registered snapshots.

No learned safety model is imported.  The existing low-level locomotion policy
is part of the frozen simulator/controller plant; the only new interface is a
zero velocity/yaw command that bypasses the planner-level command slew.
"""
from __future__ import annotations

import argparse
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
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from lewm.safety import h1_safe_action_set_successor_v1 as S
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as N
from scripts import run_go2_oracle_branch_pilot_v1_2 as V

OUT = ROOT / ".generated/h1_safe_action_set_successor_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_safe_action_set_successor_v1")
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
ENHANCED = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
FIXTURE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
FIXTURE_SENSOR = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/sensor_index.json"
MAX_STEPS = S.MAX_PHYSICS_STEPS

# Frozen before fixture or scientific execution.  The two wall fixtures are
# bound to training/development states whose registered forward/yaw candidates
# contact; no fresh or scientific evaluation identity is involved.
FIXTURES = (
    ("zero_initial_command", "scale-fit-0-00", None),
    ("maximum_forward_command", "scale-fit-0-00", (0.3, 0.0, 0.0)),
    ("maximum_reverse_command", "scale-fit-0-00", (-0.2, 0.0, 0.0)),
    ("maximum_left_yaw_command", "scale-fit-0-00", (0.0, 0.0, 0.5)),
    ("maximum_right_yaw_command", "scale-fit-0-00", (0.0, 0.0, -0.5)),
    ("combined_forward_yaw", "scale-fit-0-00", (0.3, 0.0, 0.5)),
    ("obstacle_free_stopping", "scale-fit-0-00", (0.3, 0.0, 0.0)),
    ("wall_directly_ahead", "scale-fit-0-01", (0.3, 0.0, 0.0)),
    ("wall_to_side_during_turn", "scale-fit-0-01", (0.0, 0.0, -0.5)),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def records(path: Path) -> dict[str, dict]:
    return {str(r["state_id"]): r for r in json.loads(path.read_text())["state_records"]}


def yaw_from_quat(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_snapshots(state, *, need_predecessor=True):
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=V.V1._load_shared("cpu"))
    ctx.begin_episode(); predecessor = None
    for block in range(int(state["warmup_blocks"])):
        if need_predecessor and block == int(state["warmup_blocks"]) - 1:
            predecessor = V.V1.capture_branch_state(ctx, goal={}, identity={"state_id": f"{state['state_id']}:predecessor", "scene_id": state["scene_id"], "family": state["family"]})
        ctx.drive_one_block()
    eligible = V.eligible_here(ctx, V.link_topology(ctx))
    if isinstance(eligible, str): raise RuntimeError(f"{state['state_id']}: eligibility changed: {eligible}")
    goal, _ = eligible
    identity = {"state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"]}
    current = V.V1.capture_branch_state(ctx, goal=dict(goal["goal"]), identity=identity)
    return ctx, current, predecessor


def prefix_motion(ctx, command) -> None:
    if command is None: return
    target = np.asarray(command, np.float32)[None, :]
    for _ in range(int(ctx.runner._policy_steps_per_command_tick)):
        ctx.runner._step_policy_step(target)


def empty_trace():
    return {key: [] for key in (
        "qpos_pre", "base_transform", "link_transform", "joint_position", "joint_velocity", "body_velocity", "body_angular_velocity",
        "requested_command", "applied_command", "native_contact", "native_count", "native_robot_link", "native_other_link",
        "native_penetration", "native_force", "exact_contact", "exact_count", "exact_robot_link", "exact_other_link", "exact_penetration",
        "planar_speed", "yaw_rate", "acceleration_norm", "angular_acceleration_norm", "peak_abs_torque", "fall", "tipped", "unsafe")}


def run_brake(ctx, snapshot, branch_id: str, *, prefix=None):
    V.V1.restore_branch_state(ctx, snapshot); prefix_motion(ctx, prefix)
    runner = ctx.runner; robot = ctx.build.robot; solver = ctx.build.scene.rigid_solver; topology = V.link_topology(ctx)
    link_names = {int(link.idx): str(link.name) for link in solver.links}; object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    boundary = N.contact_summary(robot.get_contacts(), topology, link_names, object_names, force_threshold=True)
    initial_pos = N.arr(robot.get_pos()).astype(np.float64); initial_quat = N.arr(robot.get_quat()).astype(np.float64)
    initial_vel = N.arr(robot.get_vel()).astype(np.float64); initial_ang = N.arr(robot.get_ang()).astype(np.float64)
    initial_command = np.asarray(runner._last_executed, np.float32)[0].copy()
    trace = empty_trace(); command_speeds = []; command_yaws = []; command_unsafe = []
    prev_vel = initial_vel.copy(); prev_ang = initial_ang.copy(); prev_pos = initial_pos.copy(); path_distance = 0.0
    stop_tick = None; terminated = None; first_contact = None; contact_summary = None
    command = S.BRAKE_COMMAND[None, :]
    physics_step = 0
    for command_tick in range(int(S.MAX_BRAKE_S / S.COMMAND_TICK_S)):
        completed_tick = True
        for _policy in range(int(runner._policy_steps_per_command_tick)):
            observation = runner._build_observation(command)
            joint_targets = runner.policy.act(observation)
            runner._apply_joint_targets(joint_targets)
            for _physics in range(int(runner._physics_steps_per_policy)):
                trace["qpos_pre"].append(N.arr(robot.get_qpos()).astype(np.float32))
                lp = N.arr(robot.get_links_pos()).astype(np.float32); lq = N.arr(robot.get_links_quat()).astype(np.float32)
                trace["link_transform"].append(np.concatenate((lp, lq), axis=-1)); trace["base_transform"].append(np.concatenate((lp[0], lq[0])))
                trace["joint_position"].append(N.arr(robot.get_dofs_position(runner._leg_dof_idx.tolist())).astype(np.float32))
                trace["joint_velocity"].append(N.arr(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())).astype(np.float32))
                trace["requested_command"].append(S.BRAKE_COMMAND.copy()); trace["applied_command"].append(S.BRAKE_COMMAND.copy())
                runner.build.scene.step(); physics_step += 1
                velocity = N.arr(robot.get_vel()).astype(np.float64); angular = N.arr(robot.get_ang()).astype(np.float64)
                pos = N.arr(robot.get_pos()).astype(np.float64); path_distance += float(np.linalg.norm(pos[:2] - prev_pos[:2])); prev_pos = pos
                acceleration = (velocity - prev_vel) / S.PHYSICS_DT_S; angular_acceleration = (angular - prev_ang) / S.PHYSICS_DT_S
                prev_vel, prev_ang = velocity.copy(), angular.copy()
                torque = N.arr(robot.get_dofs_control_force(runner._leg_dof_idx.tolist())).astype(np.float64)
                native = N.contact_summary(robot.get_contacts(), topology, link_names, object_names, force_threshold=True)
                flags = V.V1._termination_flags(ctx); unsafe = bool(any(flags.values()))
                for key, value in (("body_velocity", velocity), ("body_angular_velocity", angular),
                    ("planar_speed", float(np.linalg.norm(velocity[:2]))), ("yaw_rate", float(abs(angular[2]))),
                    ("acceleration_norm", float(np.linalg.norm(acceleration))), ("angular_acceleration_norm", float(np.linalg.norm(angular_acceleration))),
                    ("peak_abs_torque", float(np.max(np.abs(torque)))), ("native_contact", native["active"]), ("native_count", native["count"]),
                    ("native_robot_link", native["robot_link"]), ("native_other_link", native["other_link"]),
                    ("native_penetration", native["penetration"]), ("native_force", native["force"]),
                    ("fall", flags["fall"]), ("tipped", flags["tipped"]), ("unsafe", unsafe)):
                    trace[key].append(value)
                if native["active"] and first_contact is None:
                    first_contact = physics_step - 1; contact_summary = native
                if native["active"]:
                    terminated = "contact"; completed_tick = False; break
                if unsafe:
                    terminated = "unsafe_termination"; completed_tick = False; break
            runner._sim_time_ns += runner._policy_dt_ns
            if not completed_tick: break
        if not completed_tick: break
        command_speeds.append(float(trace["planar_speed"][-1])); command_yaws.append(float(trace["yaw_rate"][-1])); command_unsafe.append(bool(trace["unsafe"][-1]))
        stop_tick = S.stopped_tick(command_speeds, command_yaws, command_unsafe)
        if stop_tick is not None:
            terminated = "stopped"; break
    if terminated is None: terminated = "timeout"

    # Preserve the realised terminal pose before the read-only exact-query
    # pass rewinds the articulated configuration through captured pre-step
    # states.
    final_pos = prev_pos.copy()
    final_quat = N.arr(robot.get_quat()).astype(np.float64)

    # Query the same Genesis MPR/GJK narrowphase at the captured pre-step
    # articulated configurations.  This happens after dynamics have ended.
    for qpos in trace["qpos_pre"]:
        robot.set_qpos(qpos); robot.detect_collision()
        exact = N.contact_summary(solver.collider.get_contacts(as_tensor=False), topology, link_names, object_names, force_threshold=False)
        for key, value in (("exact_contact", exact["active"]), ("exact_count", exact["count"]),
                           ("exact_robot_link", exact["robot_link"]), ("exact_other_link", exact["other_link"]),
                           ("exact_penetration", exact["penetration"])):
            trace[key].append(value)
    arrays = {}
    for key, value in trace.items():
        if key in {"native_contact", "exact_contact", "fall", "tipped", "unsafe"}: dtype = np.uint8
        elif key in {"native_count", "exact_count", "native_robot_link", "native_other_link", "exact_robot_link", "exact_other_link"}: dtype = np.int16
        else: dtype = np.float32
        arrays[key] = np.asarray(value, dtype=dtype)
    exact_positive = arrays["exact_contact"].astype(bool); exact_pen = arrays["exact_penetration"][exact_positive]
    minimum_clearance = None if not len(exact_pen) else -float(np.nanmax(exact_pen))
    stopped = terminated == "stopped"; stable = not bool(arrays["unsafe"].any())
    initial_speed = float(np.linalg.norm(initial_vel[:2])); speed_before_contact = None
    if first_contact is not None:
        speed_before_contact = float(arrays["planar_speed"][max(0, first_contact - 1)])
    speed_reduced = bool(first_contact is None or speed_before_contact < initial_speed - 1e-4)
    metrics = {
        "branch_id": branch_id, "physics_steps": physics_step, "duration_s": physics_step * S.PHYSICS_DT_S,
        "initial_command_vx_vy_yaw": initial_command.tolist(), "initial_planar_speed_m_s": initial_speed,
        "initial_yaw_rate_rad_s": float(abs(initial_ang[2])), "boundary_contact": bool(boundary["active"]),
        "contact": first_contact is not None, "first_contact_step": first_contact,
        "first_contact_time_s": None if first_contact is None else (first_contact + 1) * S.PHYSICS_DT_S,
        "contact_robot_link": None if contact_summary is None else contact_summary.get("robot_link_name"),
        "contact_environment_object": None if contact_summary is None else contact_summary.get("environment_object"),
        "stopped": stopped, "stopping_time_s": physics_step * S.PHYSICS_DT_S if stopped else None,
        "stopping_distance_m": path_distance if stopped else None, "path_distance_until_termination_m": path_distance,
        "route_displacement_xy_m": (final_pos[:2] - initial_pos[:2]).tolist(),
        "route_displacement_norm_m": float(np.linalg.norm(final_pos[:2] - initial_pos[:2])),
        "yaw_change_rad": float((yaw_from_quat(final_quat) - yaw_from_quat(initial_quat) + math.pi) % (2 * math.pi) - math.pi),
        "peak_acceleration_m_s2": float(arrays["acceleration_norm"].max(initial=0)),
        "peak_angular_acceleration_rad_s2": float(arrays["angular_acceleration_norm"].max(initial=0)),
        "peak_actuator_torque_nm": float(arrays["peak_abs_torque"].max(initial=0)),
        "fall": bool(arrays["fall"].any()), "tipped": bool(arrays["tipped"].any()), "unsafe_termination": bool(arrays["unsafe"].any()),
        "stable": stable, "termination": terminated, "speed_reduced_before_contact": speed_reduced,
        "speed_before_contact_m_s": speed_before_contact, "exact_narrowphase_contact": bool(exact_positive.any()),
        "minimum_exact_clearance_m": minimum_clearance,
        "minimum_exact_clearance_semantics": "negative penetration; positive separation magnitude unavailable from Genesis 0.3.14 public query" if minimum_clearance is not None else "strictly positive lower bound only; magnitude unavailable",
        "qualified_safe_brake": bool(stopped and stable and first_contact is None),
        "command_tick_samples": len(command_speeds),
        "stop_sensitivity": {
            "speed_0.03_yaw_0.08_tick": S.stopped_tick(command_speeds, command_yaws, command_unsafe, speed_threshold=.03, yaw_threshold=.08),
            "speed_0.07_yaw_0.12_tick": S.stopped_tick(command_speeds, command_yaws, command_unsafe, speed_threshold=.07, yaw_threshold=.12),
        },
        "implementation_receipt": {"requested_zero": True, "applied_zero": bool(np.all(arrays["applied_command"] == 0)),
            "planner_slew_bypassed": True, "low_level_policy_used": True, "velocity_or_pose_clamp": False,
            "joint_freeze": False, "artificial_damping": False, "collision_modified": False},
    }
    metrics["state_classification"] = S.classify_brake(boundary_contact=metrics["boundary_contact"], contact=metrics["contact"],
        contact_time_s=metrics["first_contact_time_s"], stopped=metrics["stopped"], stable=metrics["stable"], speed_reduced_before_contact=speed_reduced)
    return metrics, arrays


def trace_signature(metrics, arrays):
    keep = {k: metrics[k] for k in ("physics_steps", "duration_s", "contact", "first_contact_step", "stopped", "stopping_time_s",
        "stopping_distance_m", "route_displacement_norm_m", "yaw_change_rad", "fall", "tipped", "unsafe_termination", "termination", "qualified_safe_brake")}
    keep["array_digest"] = hashlib.sha256(b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))).hexdigest()
    return S.digest(keep)


def run_fixtures():
    panel = {r["state_id"]: r for r in json.loads(FIXTURE_PANEL.read_text())["states"]}; grouped = {}
    for name, sid, prefix in FIXTURES: grouped.setdefault(sid, []).append((name, prefix))
    results = {}; started = time.time()
    for sid, fixtures in grouped.items():
        state = panel[sid]; ctx, snapshot, _ = build_snapshots(state, need_predecessor=False)
        first_runs = {}
        for name, prefix in fixtures:
            first_m, first_a = run_brake(ctx, snapshot, f"fixture:{name}:0", prefix=prefix)
            first_runs[name] = (first_m, first_a, prefix)
        del ctx; gc.collect()

        # A fresh scene distinguishes true deterministic regeneration from
        # collider scratch-state left by the post-run exact-query audit.
        ctx, snapshot, _ = build_snapshots(state, need_predecessor=False)
        for name, prefix in fixtures:
            first_m, first_a, _ = first_runs[name]
            second_m, second_a = run_brake(ctx, snapshot, f"fixture:{name}:1", prefix=prefix)
            first_sig = trace_signature(first_m, first_a); second_sig = trace_signature(second_m, second_a)
            finite_keys = (
                "qpos_pre", "base_transform", "link_transform", "joint_position", "joint_velocity",
                "body_velocity", "body_angular_velocity", "requested_command", "applied_command",
                "planar_speed", "yaw_rate", "acceleration_norm", "angular_acceleration_norm",
                "peak_abs_torque",
            )
            results[name] = {"state_id": sid, "training_only": True, "prefix_command": prefix,
                "first": {k: first_m[k] for k in ("contact", "stopped", "stopping_time_s", "stopping_distance_m", "fall", "tipped", "termination")},
                "deterministic_signature": first_sig, "byte_identical_numeric_regeneration": first_sig == second_sig,
                "finite": bool(all(np.isfinite(first_a[key]).all() for key in finite_keys)),
                "no_state_reset_or_velocity_clamp": bool(first_m["implementation_receipt"]["velocity_or_pose_clamp"] is False)}
        del ctx; gc.collect()
    pure = S.fixture_payload(); passed = pure["pass"] and all(x["byte_identical_numeric_regeneration"] and x["finite"] and x["no_state_reset_or_velocity_clamp"] for x in results.values())
    payload = {"schema": "h1_safe_action_set_successor_development_fixtures_v1", "pure_fixture": pure, "physics_fixtures": results,
        "fixture_count": len(results), "each_executed_twice": True, "pass": passed, "runtime_s": time.time() - started,
        "frozen_before_scientific_execution": True}
    payload["content_digest"] = S.digest(payload); atomic_json(OUT / "fixture_result.json", payload)
    if not passed: raise RuntimeError("emergency-brake fixture failure")
    print(json.dumps({"pass": passed, "fixture_count": len(results), "runtime_s": payload["runtime_s"], "content_digest": payload["content_digest"]}, indent=2))
    return payload


def collect_state(index: int):
    panel = json.loads(PANEL.read_text()); state = panel["states"][index]; sid = state["state_id"]; out = OUT / "states" / f"{sid}.json"
    if out.is_file():
        rec = json.loads(out.read_text()); shard = Path(rec["shard_path"])
        if rec.get("status") == "PASS" and shard.is_file() and sha(shard) == rec["shard_sha256"]:
            print(json.dumps({"state_id": sid, "status": "REUSED"}), flush=True); return rec
    started = time.time(); enhanced = records(ENHANCED)[sid]
    ctx, snapshot, predecessor = build_snapshots(state, need_predecessor=True)
    if snapshot.digest != enhanced["snapshot_digest"]: raise RuntimeError(f"{sid}: frozen snapshot digest mismatch")
    current_metrics, current_arrays = run_brake(ctx, snapshot, f"{sid}:emergency_brake_v1")
    predecessor_metrics = None; predecessor_arrays = None
    if current_metrics["contact"]:
        predecessor_metrics, predecessor_arrays = run_brake(ctx, predecessor, f"{sid}:predecessor:emergency_brake_v1")
    current_metrics["predecessor_classification"] = S.classify_predecessor(current_metrics, predecessor_metrics)
    arrays = {f"current_{k}": v for k, v in current_arrays.items()}
    if predecessor_arrays is not None: arrays.update({f"predecessor_{k}": v for k, v in predecessor_arrays.items()})
    shard = CACHE / "states" / f"{sid}.npz"; atomic_npz(shard, **arrays)
    rec = {"schema": "h1_safe_action_set_successor_state_v1", "status": "PASS", "state_index": index, "state_id": sid,
        "scene_id": state["scene_id"], "family": state["family"], "split": state["split"], "snapshot_digest": snapshot.digest,
        "snapshot_exact": True, "brake_branch_identity": f"{sid}:emergency_brake_v1", "current": current_metrics,
        "predecessor_executed": predecessor_metrics is not None, "predecessor": predecessor_metrics,
        "new_scientific_branches": 1, "predecessor_test_branches": int(predecessor_metrics is not None),
        "shard_path": str(shard), "shard_sha256": sha(shard), "storage_bytes": shard.stat().st_size, "runtime_s": time.time() - started}
    rec["content_digest"] = S.digest(rec); atomic_json(out, rec)
    print(json.dumps({"state_id": sid, "contact": current_metrics["contact"], "stopped": current_metrics["stopped"],
        "qualified": current_metrics["qualified_safe_brake"], "predecessor": current_metrics["predecessor_classification"], "runtime_s": rec["runtime_s"]}), flush=True)
    del ctx; gc.collect(); return rec


def finalize():
    panel = json.loads(PANEL.read_text()); rows = []
    for state in panel["states"]:
        row = json.loads((OUT / "states" / f"{state['state_id']}.json").read_text())
        if row["status"] != "PASS" or sha(Path(row["shard_path"])) != row["shard_sha256"]: raise RuntimeError(f"bad state receipt {state['state_id']}")
        rows.append(row)
    wall = json.loads((OUT / "collection_wall_receipt.json").read_text()); fixture = json.loads((OUT / "fixture_result.json").read_text())
    payload = {"schema": "h1_safe_action_set_successor_index_v1", "states": 48, "new_scientific_branches": 48,
        "predecessor_test_branches": sum(r["predecessor_test_branches"] for r in rows), "new_state_identities": 0,
        "snapshot_exact_states": sum(r["snapshot_exact"] for r in rows), "state_records": rows,
        "contract": {"name": "EMERGENCY_BRAKE_V1", "requested_command_vx_vy_yaw": [0.0, 0.0, 0.0],
            "planner_slew_bypass": True, "low_level_locomotion_policy_unchanged": True, "joint_and_actuator_limits_unchanged": True,
            "physics_and_collision_unchanged": True, "velocity_pose_clamp": False, "artificial_damping": False,
            "stopped_speed_m_s": S.SPEED_THRESHOLD_M_S, "stopped_yaw_rate_rad_s": S.YAW_RATE_THRESHOLD_RAD_S,
            "stopped_consecutive_command_ticks": S.CONSECUTIVE_COMMAND_TICKS, "maximum_duration_s": S.MAX_BRAKE_S},
        "fixture_digest": fixture["content_digest"], "runtime_compute_s": sum(r["runtime_s"] for r in rows),
        "parallel_wall_runtime_s": wall["wall_runtime_s"], "fixture_runtime_s": fixture["runtime_s"],
        "storage_bytes": sum(r["storage_bytes"] for r in rows), "bindings": {"panel_sha256": sha(PANEL), "enhanced_index_sha256": sha(ENHANCED)}}
    payload["content_digest"] = S.digest(payload); atomic_json(OUT / "brake_index.json", payload)
    print(json.dumps({k: payload[k] for k in ("states", "new_scientific_branches", "predecessor_test_branches", "snapshot_exact_states", "runtime_compute_s", "parallel_wall_runtime_s", "storage_bytes", "content_digest")}, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixtures", action="store_true"); group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.fixtures: run_fixtures()
    elif args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all:
        fixture = json.loads((OUT / "fixture_result.json").read_text()) if (OUT / "fixture_result.json").is_file() else None
        if not fixture or not fixture["pass"]: raise RuntimeError("fixtures must pass before scientific execution")
        started = time.time(); logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True)
        for start in range(0, 48, 4):
            processes = []
            for index in range(start, min(start + 4, 48)):
                stream = (logs / f"state_{index:03d}.log").open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
                processes.append((index, process, stream))
            for index, process, stream in processes:
                code = process.wait(); stream.close()
                if code: raise RuntimeError(f"state {index} failed; see {logs / f'state_{index:03d}.log'}")
        atomic_json(OUT / "collection_wall_receipt.json", {"parallel_processes": 4, "wall_runtime_s": time.time() - started})
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
