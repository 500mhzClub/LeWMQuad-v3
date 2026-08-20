#!/usr/bin/env python3
"""Replay frozen route branches solely to record enhanced embodied sensors."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import materialize_dense_route_intent_true_future_v1 as DENSE
from scripts import materialize_deployment_valid_dense_proprioception_v1 as OLD_PROP
from scripts import replay_safe_local_waypoint_route_intent_v2 as REPLAY
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_genesis.rollout import _rotate_world_to_body

OUT = ROOT / ".generated/enhanced_embodied_safety_observability_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/enhanced_embodied_safety_observability_v2")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
OLD_PROP_INDEX = ROOT / ".generated/deployment_valid_safety_observability_matrix_v1/proprio_index.json"
DT = 0.1
FOOT_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
CHANNELS = (
    *(f"imu_accelerometer_{axis}_m_s2" for axis in "xyz"),
    *(f"imu_gyroscope_{axis}_rad_s" for axis in "xyz"),
    *(f"projected_gravity_{axis}" for axis in "xyz"),
    *(f"joint_position_relative_default_{i:02d}_rad" for i in range(12)),
    *(f"joint_velocity_{i:02d}_rad_s" for i in range(12)),
    *(f"joint_acceleration_{i:02d}_rad_s2" for i in range(12)),
    *(f"actuator_control_torque_{i:02d}_nm" for i in range(12)),
    *(f"foot_net_contact_force_{name[:2].lower()}_n" for name in FOOT_NAMES),
    *(f"previous_policy_action_{i:02d}" for i in range(12)),
)
ACTION_CONTROL_CHANNELS = (
    "candidate_post_slew_vx_m_s", "candidate_post_slew_vy_m_s", "candidate_post_slew_yaw_rad_s",
    "previous_applied_vx_m_s", "previous_applied_vy_m_s", "previous_applied_yaw_rad_s",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def _one_env(value) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def sensor_state(runner, *, previous_joint_velocity: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the 73-D allowed state and current policy-order joint velocity."""
    robot = runner.build.robot
    policy = runner.policy
    observation = runner._build_observation(np.asarray(runner._last_executed, dtype=np.float32))
    quat_xyzw = np.asarray(observation["base_quat_xyzw"], np.float32)
    angular_body = _rotate_world_to_body(np.asarray(observation["base_ang_vel_world"], np.float32), quat_xyzw)[0]
    gravity_unit_world = np.zeros((1, 3), np.float32); gravity_unit_world[:, 2] = -1.
    projected_gravity = _rotate_world_to_body(gravity_unit_world, quat_xyzw)[0]
    base_acc_world = np.asarray(runner._as_np(robot.get_links_acc([0])), np.float32).reshape(-1, 3)[0]
    gravity_m_s2 = np.asarray([[0., 0., -9.81]], np.float32)
    accelerometer = _rotate_world_to_body(base_acc_world[None] - gravity_m_s2, quat_xyzw)[0]
    order = policy._policy_from_rollout
    position = np.asarray(observation["joint_pos"], np.float32)[:, order][0]
    velocity = np.asarray(observation["joint_vel"], np.float32)[:, order][0]
    relative_position = position - np.asarray(policy.default_dof_pos_policy, np.float32)
    acceleration = np.zeros(12, np.float32) if previous_joint_velocity is None else (velocity - previous_joint_velocity) / DT
    torque_rollout = _one_env(runner._as_np(robot.get_dofs_control_force(runner._leg_dof_idx.tolist()))).astype(np.float32)
    torque = torque_rollout[order]
    all_forces = np.asarray(runner._as_np(robot.get_links_net_contact_force()), np.float32)
    if all_forces.ndim == 2:
        all_forces = all_forces[None]
    foot_local = [int(next(link for link in robot.links if link.name == name).idx_local) for name in FOOT_NAMES]
    foot_force = np.linalg.norm(all_forces[0, foot_local], axis=-1).astype(np.float32)
    previous_action = np.asarray(policy._last_actions, np.float32)[0]
    value = np.concatenate((accelerometer, angular_body, projected_gravity, relative_position,
                            velocity, acceleration, torque, foot_force, previous_action)).astype(np.float32)
    if value.shape != (73,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid enhanced state: {value.shape}")
    return value, velocity.copy()


def execute_capture(ctx, snapshot, candidate, source: dict, dense_branch: dict,
                    old_current: np.ndarray, old_future: np.ndarray, *, topology) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep

    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    counter = {"episode_step": int(runner.episode_states[0].episode_step), "stamp_ns": int(runner._sim_time_ns)}
    current, previous_velocity = sensor_state(runner, previous_joint_velocity=None)
    # Every old-contract channel must still reproduce at the pre-action boundary.
    old_from_new = np.concatenate((current[6:9], current[3:6], current[9:21], current[21:33], current[61:73]))
    if not np.allclose(old_from_new, old_current, atol=2e-5, rtol=0):
        raise RuntimeError(f"{source['state_id']}:{source['candidate_index']}: current sensor lineage mismatch")
    previous_command = np.asarray(runner._last_executed, np.float32)[0].copy()
    ticks: list[np.ndarray] = []; controls: list[np.ndarray] = []; replay_components: list[tuple[bool, bool]] = []
    executed_all = []
    _name, primitives = candidate
    for block_index, primitive in enumerate(primitives[:3]):
        requested = V.V1.block_for(primitive)[None, ...]
        clipped = np.asarray(runner._clip_block(np.asarray(requested, np.float32)).executed, np.float32)

        def after_policy_step(tick_index: int, step_index: int, _block=clipped) -> None:
            nonlocal previous_velocity, previous_command
            if step_index != int(runner._policy_steps_per_command_tick) - 1:
                return
            counter["episode_step"] += 1; counter["stamp_ns"] += int(runner._command_dt_ns)
            command = _block[0, tick_index]
            state, velocity = sensor_state(runner, previous_joint_velocity=previous_velocity)
            previous_velocity = velocity
            offset = len(ticks); reference = dense_branch["ticks"][offset]
            robot = ctx.build.robot
            position_world = _one_env(REPLAY._to_numpy(robot.get_pos())).astype(np.float64)
            xy, yaw, _z = ctx.pose()
            label = label_computer.step(PoseStep(timestamp_ns=counter["stamp_ns"], env_idx=0,
                episode_id=episode_id, episode_step=counter["episode_step"], position_xy_world=xy,
                yaw_world_rad=float(yaw), last_command=tuple(float(v) for v in command)))
            contact = int(V._contact_count(ctx, topology)) > 0; stuck = bool(label.stuck_label)
            if contact != bool(reference["active_contact"]) or stuck != bool(reference["active_stuck"]):
                raise RuntimeError(f"{source['state_id']}:{source['candidate_index']} tick {offset + 1}: safety trace mismatch")
            if not np.allclose(position_world, reference["position_world_xyz"], atol=2e-5, rtol=0):
                raise RuntimeError(f"{source['state_id']}:{source['candidate_index']} tick {offset + 1}: pose mismatch")
            old = old_future[offset]
            old_from_state = np.concatenate((state[6:9], state[3:6], state[9:21], state[21:33], state[61:73]))
            if not np.allclose(old_from_state, old, atol=2e-5, rtol=0):
                raise RuntimeError(f"{source['state_id']}:{source['candidate_index']} tick {offset + 1}: old sensor mismatch")
            ticks.append(state); controls.append(np.concatenate((command, previous_command)).astype(np.float32))
            replay_components.append((contact, stuck)); previous_command = command.copy()

        block = runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        executed_all.append(np.asarray(block.executed, np.float32)[0])
        ctx.ticks_executed += runner._block_size; ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
        ctx.last_block_executed = np.asarray(block.executed, np.float32).copy()
    executed = np.concatenate(executed_all)
    expected = np.concatenate([np.asarray(block, np.float32) for block in source["post_slew"][:3]])
    if not np.allclose(executed, expected, atol=1e-7, rtol=0):
        raise RuntimeError(f"{source['state_id']}:{source['candidate_index']}: action trace mismatch")
    if len(ticks) != 15:
        raise RuntimeError("wrong enhanced tick count")
    return current, np.stack(ticks), np.stack(controls)


def collect_state(state_index: int, *, force: bool = False) -> dict:
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    state = manifest["state_candidates"][state_index]; sid = state["state_id"]
    record_path = OUT / "states" / f"{sid}.json"
    if record_path.is_file() and not force:
        payload = json.loads(record_path.read_text()); shard = Path(payload["shard_path"])
        if payload.get("status") == "PASS" and shard.is_file() and sha(shard) == payload["shard_sha256"]:
            print(json.dumps({"state_id": sid, "status": "REUSED"}), flush=True); return payload
    started = time.time()
    dense_state = json.loads((DENSE.OUT / "dense_replay" / f"{sid}.json").read_text())
    dense_by_candidate = {int(row["candidate_index"]): row for row in dense_state["branches"]}
    source = {ci: row for (state_id, ci), row in DENSE.source_rows().items() if state_id == sid}
    old_record = json.loads((OLD_PROP.OUT / "proprio_states" / f"{sid}.json").read_text())
    with np.load(old_record["shard_path"]) as old:
        old_current = np.asarray(old["current"], np.float32); old_future = np.asarray(old["future"], np.float32)
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40): ctx.drive_one_block()
    topology = V.link_topology(ctx); eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str): raise RuntimeError(f"{sid}: eligibility changed: {eligible}")
    record, _field = eligible
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(record["goal"]), identity={
        "state_id": sid, "scene_id": state["scene_id"], "family": state["family"]})
    current_rows, future_rows, control_rows = [], [], []
    for ci, candidate in enumerate(V.V1.CANDIDATE_BANK):
        current, future, control = execute_capture(ctx, snapshot, candidate, source[ci], dense_by_candidate[ci],
                                                   old_current[ci], old_future[ci], topology=topology)
        current_rows.append(current); future_rows.append(future); control_rows.append(control)
    current = np.stack(current_rows).astype(np.float32); future = np.stack(future_rows).astype(np.float32)
    controls = np.stack(control_rows).astype(np.float32)
    shard = CACHE / "sensors" / f"{sid}.npz"; atomic_npz(shard, current=current, future=future, action_control=controls)
    raw_h3_mismatches = sum(not bool(branch["aggregate_replay_match"][2]) for branch in dense_state["branches"])
    payload = {"schema": "enhanced_embodied_safety_state_v2", "status": "PASS", "state_index": state_index,
        "state_id": sid, "scene_id": state["scene_id"], "family": state["family"], "branches": 12, "ticks": 15,
        "channels": list(CHANNELS), "action_control_channels": list(ACTION_CONTROL_CHANNELS),
        "shapes": {"current": list(current.shape), "future": list(future.shape), "action_control": list(controls.shape)},
        "dtype": "float32", "shard_path": str(shard), "shard_sha256": sha(shard),
        "snapshot_digest": snapshot.digest, "registered_snapshot_digest": state.get("snapshot_digest"),
        "snapshot_digest_match": state.get("snapshot_digest") is None or snapshot.digest == state["snapshot_digest"],
        "post_slew_pose_contact_stuck_mismatches": 0,
        "raw_replay_h3_aggregate_mismatches_preserved": raw_h3_mismatches,
        "authoritative_training_aggregate_binding": "dense training_cumulative_unsafe right-censored to frozen aggregate",
        "runtime_s": time.time() - started, "new_state_identities": 0, "new_branch_identities": 0}
    payload["content_digest"] = canonical_digest(payload); atomic_json(record_path, payload)
    del ctx; gc.collect(); print(json.dumps({"state_id": sid, "status": "PASS", "runtime_s": payload["runtime_s"]}), flush=True)
    return payload


def finalize() -> dict:
    manifest = json.loads((V1 / "state_manifest.json").read_text()); rows = []; arrays = []
    branch_receipts = []
    for state in manifest["state_candidates"]:
        path = OUT / "states" / f"{state['state_id']}.json"; row = json.loads(path.read_text())
        if row.get("status") != "PASS" or sha(Path(row["shard_path"])) != row["shard_sha256"]:
            raise RuntimeError(f"invalid enhanced shard {state['state_id']}")
        rows.append(row)
        dense_state = json.loads((DENSE.OUT / "dense_replay" / f"{state['state_id']}.json").read_text())
        for branch in dense_state["branches"]:
            branch_receipts.append({
                "branch_id": f"{state['state_id']}:{int(branch['candidate_index']):02d}", "state_id": state["state_id"],
                "candidate_index": int(branch["candidate_index"]),
                "post_slew_action_match": True, "h1_h2_h3_pose_match": True,
                "contact_trace_match": True, "stuck_trace_match": True,
                "raw_aggregate_h3_match": bool(branch["aggregate_replay_match"][2]),
                "authoritative_aggregate_binding_preserved": True,
            })
        with np.load(row["shard_path"]) as loaded: arrays.append(np.asarray(loaded["future"], np.float64))
    flat = np.concatenate(arrays).reshape(-1, 73); ranges = np.ptp(flat, axis=0)
    paths = [OUT / "states" / f"{row['state_id']}.json" for row in rows]
    payload = {"schema": "enhanced_embodied_safety_index_v2", "complete": True,
        "states": 48, "branches": 576, "ticks_per_branch": 15, "channels": list(CHANNELS),
        "action_control_channels": list(ACTION_CONTROL_CHANNELS), "channel_count": 73,
        "channel_contract": {
            "imu_accelerometer": "Genesis base-link classical acceleration minus gravity, body-frame, m/s^2",
            "imu_gyroscope": "Genesis base angular velocity rotated world-to-body, rad/s",
            "projected_gravity": "unit gravity vector rotated world-to-body",
            "joint_position": "encoder position minus deployed default, policy joint order, rad",
            "joint_velocity": "encoder velocity, policy joint order, rad/s",
            "joint_acceleration": "causal 10 Hz finite difference of encoder velocity; current boundary is zero reference, rad/s^2",
            "actuator_torque": "Genesis PD control force from get_dofs_control_force, policy joint order, N m",
            "foot_force": "correct full-link net-contact-force magnitude sliced at four calf links, N",
            "previous_policy_action": "deployed policy adapter last raw action",
            "estimated_foot_force": "unavailable: Jacobian/IK support was not enabled in the frozen robot build; no value fabricated"},
        "excluded_inputs": ["global_position", "global_yaw", "simulator_body_linear_velocity", "scene_graph", "occupancy_grid", "future_rgb", "safety_labels", "privileged_collision_geometry"],
        "missing_channels": ["four_jacobian_based_estimated_foot_forces"],
        "degenerate_channels": [name for name, value in zip(CHANNELS, ranges) if value < 1e-7],
        "per_channel_range": {name: float(value) for name, value in zip(CHANNELS, ranges)},
        "replayed_branches": 576, "post_slew_pose_contact_stuck_mismatches": 0,
        "raw_replay_h3_aggregate_matches": 576 - sum(row["raw_replay_h3_aggregate_mismatches_preserved"] for row in rows),
        "raw_replay_h3_aggregate_mismatches_preserved": sum(row["raw_replay_h3_aggregate_mismatches_preserved"] for row in rows),
        "authoritative_aggregate_matches": 576,
        "branch_replay_receipts": branch_receipts,
        "snapshot_digest_matches": sum(bool(row["snapshot_digest_match"]) for row in rows),
        "snapshot_digest_mismatches_with_exact_tick_reproduction": sum(not bool(row["snapshot_digest_match"]) for row in rows),
        "runtime_compute_s": sum(float(row["runtime_s"]) for row in rows),
        "parallel_wall_runtime_s": max(path.stat().st_mtime for path in paths) - min(path.stat().st_mtime - float(row["runtime_s"]) for path, row in zip(paths, rows)),
        "storage_bytes": sum(Path(row["shard_path"]).stat().st_size for row in rows), "state_records": rows,
        "bindings": {"state_manifest_sha256": sha(V1 / "state_manifest.json"), "split_sha256": sha(V1 / "split.json"),
                     "old_proprio_index_sha256": sha(OLD_PROP_INDEX), "dense_evidence_sha256": sha(DENSE.OUT / "evidence_receipt.json")}}
    payload["content_digest"] = canonical_digest(payload); atomic_json(OUT / "enhanced_sensor_index.json", payload)
    print(json.dumps({key: payload[key] for key in ("states", "branches", "channel_count", "degenerate_channels", "raw_replay_h3_aggregate_mismatches_preserved", "runtime_compute_s", "storage_bytes", "content_digest")}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true"); args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.collect_state is not None: collect_state(args.collect_state, force=args.force)
    elif args.collect_all:
        for index in range(48): collect_state(index, force=args.force)
    else: finalize()
    return 0


if __name__ == "__main__": raise SystemExit(main())
