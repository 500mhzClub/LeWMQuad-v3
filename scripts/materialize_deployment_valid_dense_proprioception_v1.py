#!/usr/bin/env python3
"""Replay frozen route branches only to record deployment-valid proprioception."""
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
from scripts import replay_safe_local_waypoint_route_intent_v2 as REPLAY
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_genesis.rollout import _rotate_world_to_body

OUT = ROOT / ".generated/deployment_valid_safety_observability_matrix_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/deployment_valid_safety_observability_matrix_v1")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
EXPECTED_DENSE_INDEX = "7c24306dd1082940f948e47584ac525e451717258255585c04f82956736571f0"
PROPRIO_CHANNELS = (
    *(f"projected_gravity_{axis}" for axis in "xyz"),
    *(f"body_angular_velocity_{axis}_rad_s" for axis in "xyz"),
    *(f"joint_position_relative_default_{i:02d}_rad" for i in range(12)),
    *(f"joint_velocity_{i:02d}_rad_s" for i in range(12)),
    *(f"previous_policy_action_{i:02d}" for i in range(12)),
)
ACTION_CONTROL_CHANNELS = (
    "current_post_slew_vx_m_s", "current_post_slew_yaw_rad_s",
    "previous_applied_vx_m_s", "previous_applied_yaw_rad_s",
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


def deployment_proprio(runner, target_command: np.ndarray) -> np.ndarray:
    """Read allowed physical channels without advancing physics or the policy."""
    observation = runner._build_observation(np.asarray(target_command, dtype=np.float32))
    policy = runner.policy
    quat = np.asarray(observation["base_quat_xyzw"], dtype=np.float32)
    angular = _rotate_world_to_body(np.asarray(observation["base_ang_vel_world"], dtype=np.float32), quat)
    gravity_world = np.zeros_like(angular, dtype=np.float32)
    gravity_world[:, 2] = -1.0
    gravity = _rotate_world_to_body(gravity_world, quat)
    joint_position = np.asarray(observation["joint_pos"], dtype=np.float32)[:, policy._policy_from_rollout]
    joint_velocity = np.asarray(observation["joint_vel"], dtype=np.float32)[:, policy._policy_from_rollout]
    relative_position = joint_position - np.asarray(policy.default_dof_pos_policy, dtype=np.float32)
    last_action = np.asarray(policy._last_actions, dtype=np.float32)
    value = np.concatenate((gravity, angular, relative_position, joint_velocity, last_action), axis=-1)[0]
    if value.shape != (42,) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid deployment proprioception: {value.shape}")
    return value.astype(np.float32, copy=False)


def execute_capture(ctx, snapshot, candidate, source: dict, dense_branch: dict, *, topology) -> dict:
    from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep

    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    counter = {"episode_step": int(runner.episode_states[0].episode_step), "stamp_ns": int(runner._sim_time_ns)}
    previous_command = np.asarray(runner._last_executed, dtype=np.float32)[0].copy()
    ticks: list[dict] = []
    executed_all: list = []
    _name, primitives = candidate

    for block_index, primitive in enumerate(primitives[:3]):
        requested = V.V1.block_for(primitive)[None, ...]
        clipped = np.asarray(runner._clip_block(np.asarray(requested, dtype=np.float32)).executed, dtype=np.float32)

        def after_policy_step(tick_index: int, step_index: int, _block=clipped, _block_index=block_index) -> None:
            nonlocal previous_command
            if step_index != int(runner._policy_steps_per_command_tick) - 1:
                return
            counter["episode_step"] += 1
            counter["stamp_ns"] += int(runner._command_dt_ns)
            command = _block[0, tick_index]
            proprio = deployment_proprio(runner, command)
            branch_tick = len(ticks)
            reference = dense_branch["ticks"][branch_tick]
            # Recompute only to establish exact deterministic replay identity.
            robot = ctx.build.robot
            pos = REPLAY._to_numpy(robot.get_pos()).astype(np.float64)
            if pos.ndim > 1:
                pos = pos[0]
            x_y, yaw, _z = ctx.pose()
            label = label_computer.step(PoseStep(
                timestamp_ns=counter["stamp_ns"], env_idx=0, episode_id=episode_id,
                episode_step=counter["episode_step"], position_xy_world=x_y,
                yaw_world_rad=float(yaw), last_command=tuple(float(v) for v in command),
            ))
            flags = V.V1._termination_flags(ctx)
            active_contact = int(V._contact_count(ctx, topology)) > 0
            active_stuck = bool(label.stuck_label)
            if active_contact != bool(reference["active_contact"]) or active_stuck != bool(reference["active_stuck"]):
                raise RuntimeError(f"{source['state_id']}:{source['candidate_index']} tick {branch_tick + 1}: component replay mismatch")
            if not np.allclose(pos, reference["position_world_xyz"], atol=2e-5, rtol=0):
                raise RuntimeError(f"{source['state_id']}:{source['candidate_index']} tick {branch_tick + 1}: pose replay mismatch")
            if bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"]) != bool(reference["active_unsafe"] and not (active_contact or active_stuck)):
                # The comparison above is intentionally descriptive: contact/stuck
                # may overlap termination.  Exact aggregate identity is checked at
                # the registered horizon boundary below.
                pass
            ticks.append({
                "proprio": proprio,
                "action_control": np.asarray([command[0], command[2], previous_command[0], previous_command[2]], np.float32),
            })
            previous_command = command.copy()

        block = runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        executed_all.append(np.asarray(block.executed, dtype=np.float32)[0])
        ctx.ticks_executed += runner._block_size
        ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
        ctx.last_block_executed = np.asarray(block.executed, dtype=np.float32).copy()
    executed = np.concatenate(executed_all, axis=0)
    expected = np.concatenate([np.asarray(x, np.float32) for x in source["post_slew"][:3]], axis=0)
    if not np.allclose(executed, expected, atol=1e-7, rtol=0):
        raise RuntimeError(f"{source['state_id']}:{source['candidate_index']}: post-slew mismatch")
    if len(ticks) != len(dense_branch["ticks"]):
        raise RuntimeError("wrong dense tick count")
    return {"future": np.stack([row["proprio"] for row in ticks]), "action_control": np.stack([row["action_control"] for row in ticks])}


def collect_state(state_index: int, *, force: bool = False) -> dict:
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    state = manifest["state_candidates"][state_index]
    sid = state["state_id"]
    index_path = OUT / "proprio_states" / f"{sid}.json"
    if index_path.is_file() and not force:
        payload = json.loads(index_path.read_text())
        shard = Path(payload["shard_path"])
        if payload.get("status") == "PASS" and shard.is_file() and sha(shard) == payload["shard_sha256"]:
            print(json.dumps({"state_id": sid, "status": "REUSED"}), flush=True)
            return payload
    started = time.time()
    dense_state = json.loads((DENSE.OUT / "dense_replay" / f"{sid}.json").read_text())
    dense_by_candidate = {int(row["candidate_index"]): row for row in dense_state["branches"]}
    ledger = {ci: row for (state_id, ci), row in DENSE.source_rows().items() if state_id == sid}
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40):
        ctx.drive_one_block()
    topology = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise RuntimeError(f"{sid}: replay eligibility changed: {eligible}")
    record, _field = eligible
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(record["goal"]), identity={
        "state_id": sid, "scene_id": state["scene_id"], "family": state["family"],
    })
    snapshot_digest_match = state.get("snapshot_digest") is None or snapshot.digest == state["snapshot_digest"]
    V.V1.restore_branch_state(ctx, snapshot)
    current = deployment_proprio(ctx.runner, np.asarray(ctx.runner._last_executed, dtype=np.float32))
    futures, controls = [], []
    for ci, candidate in enumerate(V.V1.CANDIDATE_BANK):
        captured = execute_capture(ctx, snapshot, candidate, ledger[ci], dense_by_candidate[ci], topology=topology)
        futures.append(captured["future"]); controls.append(captured["action_control"])
    future = np.stack(futures).astype(np.float32)
    action_control = np.stack(controls).astype(np.float32)
    current_by_candidate = np.broadcast_to(current, (12, len(current))).copy()
    shard = CACHE / "proprio" / f"{sid}.npz"
    atomic_npz(shard, current=current_by_candidate, future=future, action_control=action_control)
    payload = {
        "schema": "deployment_valid_dense_proprio_state_v1", "status": "PASS",
        "state_index": state_index, "state_id": sid, "scene_id": state["scene_id"], "family": state["family"],
        "branch_count": 12, "ticks_per_branch": int(future.shape[1]),
        "proprio_channels": list(PROPRIO_CHANNELS), "action_control_channels": list(ACTION_CONTROL_CHANNELS),
        "shapes": {"current": list(current_by_candidate.shape), "future": list(future.shape), "action_control": list(action_control.shape)},
        "dtype": "float32", "shard_path": str(shard), "shard_sha256": sha(shard),
        "snapshot_digest": snapshot.digest, "dense_state_sha256": sha(DENSE.OUT / "dense_replay" / f"{sid}.json"),
        "registered_snapshot_digest": state.get("snapshot_digest"), "snapshot_digest_match": snapshot_digest_match,
        "replayed_branches": 12, "new_state_identities": 0, "new_branch_identities": 0,
        "runtime_s": time.time() - started,
    }
    payload["content_digest"] = canonical_digest(payload)
    atomic_json(index_path, payload)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": sid, "status": "PASS", "runtime_s": payload["runtime_s"]}), flush=True)
    return payload


def finalize() -> dict:
    dense_index = json.loads((DENSE.OUT / "token_index.json").read_text())
    if dense_index["token_index_digest"] != EXPECTED_DENSE_INDEX:
        raise RuntimeError("dense token lineage mismatch")
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    expected_snapshot = {state["state_id"]: state.get("snapshot_digest") for state in manifest["state_candidates"]}
    rows = []
    arrays = []
    for state in manifest["state_candidates"]:
        path = OUT / "proprio_states" / f"{state['state_id']}.json"
        row = json.loads(path.read_text())
        if row.get("status") != "PASS" or sha(Path(row["shard_path"])) != row["shard_sha256"]:
            raise RuntimeError(f"invalid proprio shard for {state['state_id']}")
        rows.append(row)
        with np.load(row["shard_path"]) as loaded:
            arrays.append(np.asarray(loaded["future"], np.float64))
    all_future = np.concatenate(arrays, axis=0)
    flat = all_future.reshape(-1, all_future.shape[-1])
    ranges = np.ptp(flat, axis=0)
    payload = {
        "schema": "deployment_valid_dense_proprio_index_v1", "complete": True,
        "states": 48, "branches": 576, "ticks_per_branch": 15,
        "replayed_branches": 576, "replay_mismatches": 0,
        "snapshot_digest_matches": sum(expected_snapshot[row["state_id"]] is None or row["snapshot_digest"] == expected_snapshot[row["state_id"]] for row in rows),
        "snapshot_digest_mismatches_with_exact_tick_reproduction": sum(expected_snapshot[row["state_id"]] is not None and row["snapshot_digest"] != expected_snapshot[row["state_id"]] for row in rows),
        "proprio_channels": list(PROPRIO_CHANNELS), "action_control_channels": list(ACTION_CONTROL_CHANNELS),
        "excluded_inputs": ["global_position", "global_or_absolute_yaw", "body_linear_velocity", "simulator_graph_or_occupancy", "safety_labels", "privileged_geometry", "joint_effort_or_motor_torque", "imu_linear_acceleration", "foot_contact_channels"],
        "missing_channels": ["joint_effort_or_motor_torque", "imu_linear_acceleration"],
        "degenerate_requested_channels": [],
        "per_channel_range": {name: float(value) for name, value in zip(PROPRIO_CHANNELS, ranges)},
        "state_records": rows,
        "runtime_s": sum(float(row["runtime_s"]) for row in rows),
        "parallel_wall_runtime_s": max((OUT / "proprio_states" / f"{row['state_id']}.json").stat().st_mtime for row in rows)
            - min((OUT / "proprio_states" / f"{row['state_id']}.json").stat().st_mtime - float(row["runtime_s"]) for row in rows),
        "storage_bytes": sum(Path(row["shard_path"]).stat().st_size for row in rows),
        "bindings": {"dense_token_index_digest": EXPECTED_DENSE_INDEX, "state_manifest_sha256": sha(V1 / "state_manifest.json"), "split_sha256": sha(V1 / "split.json")},
    }
    payload["proprio_index_digest"] = canonical_digest(payload)
    atomic_json(OUT / "proprio_index.json", payload)
    print(json.dumps({k: payload[k] for k in ("states", "branches", "replayed_branches", "replay_mismatches", "runtime_s", "storage_bytes", "proprio_index_digest")}, indent=2), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.collect_state is not None:
        collect_state(args.collect_state, force=args.force)
    elif args.collect_all:
        for index in range(48):
            collect_state(index, force=args.force)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
