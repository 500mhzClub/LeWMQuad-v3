#!/usr/bin/env python3
"""Generate the bounded two-level one-tick viability tree.

The script uses the frozen Genesis 0.3.14 controller/physics path.  It does not
load a learned safety or world model and it never changes the historical branch
identities or labels.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import one_tick_viability_constrained_mpc_v1 as REDUCE
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as NARROW
from scripts import run_go2_oracle_branch_pilot_v1_2 as V


SOURCE_COMMIT = "481253b5a504b0cd9fd05b14f5ad662b496fa0a8"
OUT = ROOT / ".generated/one_tick_viability_constrained_mpc_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/one_tick_viability_constrained_mpc_v1")
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
WIDE_STATES = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/states"
EXACT_INDEX = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1/narrowphase_index.json"
PREVIOUS = ROOT / ".generated/control_commitment_horizon_and_viability_v1/result.json"
FIXTURE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
PHYSICS_STEPS = 50


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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def records(path: Path) -> dict[str, dict]:
    return {str(row["state_id"]): row for row in json.loads(path.read_text())["state_records"]}


def _capture_tick_boundary(ctx, *, goal: dict, identity: dict):
    """Capture all branch state at a 100 ms boundary, including non-block boundaries."""

    runner, policy = ctx.runner, ctx.policy
    solver_state = V.V1.dump_solver_state(ctx.solver_fields)
    step_index = int(getattr(ctx.build.scene, "t", -1))
    harness = {
        name: np.array(getattr(runner, name), copy=True)
        for name in V.V1._HARNESS_ARRAY_FIELDS
    }
    harness_objects = {
        name: copy.deepcopy(getattr(runner, name))
        for name in V.V1._HARNESS_OBJECT_FIELDS
    }
    harness_objects["episode_states"] = copy.deepcopy(runner.episode_states)
    harness_objects["_sim_time_ns"] = int(runner._sim_time_ns)
    harness_objects["_sequence_id_counter"] = int(runner._sequence_id_counter)
    import torch

    rng = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "runner_rng": copy.deepcopy(runner._rng.bit_generator.state),
        "spawn_rng": runner._spawn_rng.getstate(),
        "torch": torch.get_rng_state().clone(),
    }
    counters = {
        "ticks_executed": int(ctx.ticks_executed),
        "policy_steps": int(ctx.policy_steps),
        "episode_ticks": int(ctx.episode_ticks),
        "reset_in_last_block": bool(ctx.reset_in_last_block),
        "episode_start_reset_count": int(ctx.episode_start_reset_count),
    }
    last_block = None if ctx.last_block_executed is None else np.array(ctx.last_block_executed, copy=True)
    boundary = {
        "kind": "one_tick_replanning_boundary_v1",
        "command_tick_phase": int(ctx.ticks_executed) % int(runner._block_size),
        "decimation_phase": int(ctx.policy_steps) % int(runner._policy_steps_per_command_tick),
        "sim_time_ns": int(runner._sim_time_ns),
        "episode_step": int(runner.episode_states[0].episode_step),
    }
    snapshot = V.V1.BranchSnapshot(
        solver_state=solver_state,
        step_index=step_index,
        last_actions=np.array(policy._last_actions, dtype=np.float32, copy=True),
        harness={"arrays": harness, "objects": harness_objects, "last_block": last_block},
        rng=rng,
        counters=counters,
        goal=dict(goal),
        identity=dict(identity),
        boundary=boundary,
        digest=None,
    )
    controller = b"".join(
        [
            snapshot.last_actions.tobytes(),
            np.ascontiguousarray(harness["_last_executed"], dtype=np.float32).tobytes(),
            pickle.dumps(rng["python"], protocol=4),
            pickle.dumps(rng["numpy_global"], protocol=4),
            pickle.dumps(rng["runner_rng"], protocol=4),
            pickle.dumps(rng["spawn_rng"], protocol=4),
            rng["torch"].numpy().tobytes(),
        ]
    )
    snapshot.digest = V.V1._domain_digest(
        [
            (b"GENESIS_SOLVER_V1", V.V1._solver_state_digest(solver_state, step_index).encode()),
            (b"CONTROLLER_RNG_V1", controller),
            (b"ONE_TICK_BOUNDARY_V1", json.dumps(boundary, sort_keys=True).encode()),
        ]
    )
    return snapshot


def _restore_tick_boundary(ctx, snapshot) -> None:
    """Restore the same state layers as the frozen branch code, without 5-tick assertion."""

    runner, policy = ctx.runner, ctx.policy
    V.V1.load_solver_state(ctx.solver_fields, snapshot.solver_state)
    if snapshot.step_index >= 0:
        ctx.build.scene._t = int(snapshot.step_index)
    policy._last_actions = np.array(snapshot.last_actions, dtype=np.float32, copy=True)
    for name, value in snapshot.harness["arrays"].items():
        setattr(runner, name, np.array(value, copy=True))
    objects = snapshot.harness["objects"]
    for name in V.V1._HARNESS_OBJECT_FIELDS:
        setattr(runner, name, copy.deepcopy(objects[name]))
    runner.episode_states = copy.deepcopy(objects["episode_states"])
    runner._sim_time_ns = int(objects["_sim_time_ns"])
    runner._sequence_id_counter = int(objects["_sequence_id_counter"])
    import torch

    random.setstate(snapshot.rng["python"])
    np.random.set_state(snapshot.rng["numpy_global"])
    runner._rng.bit_generator.state = copy.deepcopy(snapshot.rng["runner_rng"])
    runner._spawn_rng.setstate(snapshot.rng["spawn_rng"])
    torch.set_rng_state(snapshot.rng["torch"].clone())
    ctx.ticks_executed = int(snapshot.counters["ticks_executed"])
    ctx.policy_steps = int(snapshot.counters["policy_steps"])
    ctx.episode_ticks = int(snapshot.counters["episode_ticks"])
    ctx.reset_in_last_block = bool(snapshot.counters["reset_in_last_block"])
    ctx.episode_start_reset_count = int(snapshot.counters["episode_start_reset_count"])
    prior = snapshot.harness["last_block"]
    ctx.last_block_executed = None if prior is None else np.array(prior, copy=True)


def _advance_tick_counters(ctx, target: np.ndarray) -> None:
    runner = ctx.runner
    runner._last_executed = np.asarray(target, np.float32).copy()
    for state in runner.episode_states:
        state.step()
    ctx.ticks_executed += 1
    ctx.episode_ticks += 1
    ctx.policy_steps += int(runner._policy_steps_per_command_tick)
    ctx.last_block_executed = np.repeat(target[:, None, :], int(runner._block_size), axis=1)


def _execute_one_tick(ctx, snapshot, candidate_index: int, topology, link_names, object_names):
    _restore_tick_boundary(ctx, snapshot)
    runner = ctx.runner
    name, primitives = V.V1.CANDIDATE_BANK[candidate_index]
    requested = V.V1.block_for(primitives[0])[None, ...]
    clipped = runner._clip_block(requested)
    target = np.asarray(clipped.executed[:, 0], np.float32)
    contact = np.zeros(PHYSICS_STEPS, np.uint8)
    robot_link = np.full(PHYSICS_STEPS, -1, np.int16)
    other_link = np.full(PHYSICS_STEPS, -1, np.int16)
    counter = 0
    for _ in range(int(runner._policy_steps_per_command_tick)):
        observation = runner._build_observation(target)
        joint_targets = runner.policy.act(observation)
        runner._apply_joint_targets(joint_targets)
        for _ in range(int(runner._physics_steps_per_policy)):
            runner.build.scene.step()
            summary = NARROW.contact_summary(
                runner.build.robot.get_contacts(), topology, link_names, object_names,
                force_threshold=True,
            )
            contact[counter] = bool(summary["active"])
            robot_link[counter] = int(summary["robot_link"])
            other_link[counter] = int(summary["other_link"])
            counter += 1
        runner._sim_time_ns += runner._policy_dt_ns
    if counter != PHYSICS_STEPS:
        raise RuntimeError(f"one-tick physics count {counter} != {PHYSICS_STEPS}")
    _advance_tick_counters(ctx, target)
    endpoint = ctx.pose()
    flags = V.V1._termination_flags(ctx)
    successor = _capture_tick_boundary(
        ctx,
        goal=snapshot.goal,
        identity={**snapshot.identity, "successor_of": candidate_index},
    )
    return {
        "candidate_index": candidate_index,
        "candidate": name,
        "target_command": target[0].tolist(),
        "contact": bool(contact.any()),
        "first_contact_step": None if not contact.any() else int(np.flatnonzero(contact)[0]),
        "endpoint_pose": [[float(endpoint[0][0]), float(endpoint[0][1])], float(endpoint[1]), float(endpoint[2])],
        "termination": flags,
        "successor_snapshot_digest": successor.digest,
    }, contact, robot_link, other_link, successor


def _build_current(state: dict):
    ctx = V.V1.build_context(
        Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu",
        shared=V.V1._load_shared("cpu"),
    )
    ctx.begin_episode()
    for _ in range(int(state["warmup_blocks"])):
        ctx.drive_one_block()
    eligible = V.eligible_here(ctx, V.link_topology(ctx))
    if isinstance(eligible, str):
        raise RuntimeError(f"{state['state_id']}: eligibility changed: {eligible}")
    goal, _ = eligible
    snapshot = V.V1.capture_branch_state(
        ctx,
        goal=dict(goal["goal"]),
        identity={"state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"]},
    )
    return ctx, snapshot


def _build_predecessor(state: dict):
    """Reconstruct the state exactly 100 ms before the registered boundary."""

    ctx = V.V1.build_context(
        Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu",
        shared=V.V1._load_shared("cpu"),
    )
    ctx.begin_episode()
    for _ in range(int(state["warmup_blocks"]) - 1):
        ctx.drive_one_block()
    runner = ctx.runner
    requested, _choices = runner._collect_block()
    clipped = runner._clip_block(requested)
    for tick in range(int(runner._block_size) - 1):
        target = np.asarray(clipped.executed[:, tick], np.float32)
        runner._step_command_tick(target)
        _advance_tick_counters(ctx, target)
    snapshot = _capture_tick_boundary(
        ctx,
        goal=dict(state["goal"]),
        identity={"state_id": f"{state['state_id']}:predecessor", "scene_id": state["scene_id"], "family": state["family"]},
    )
    return ctx, snapshot


def _run_tree(ctx, snapshot, expected_current: dict[int, dict] | None = None):
    topology = V.link_topology(ctx)
    solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    current_contact = np.zeros((12, PHYSICS_STEPS), np.uint8)
    current_robot_link = np.full((12, PHYSICS_STEPS), -1, np.int16)
    current_other_link = np.full((12, PHYSICS_STEPS), -1, np.int16)
    successor_contact = np.full((12, 12, PHYSICS_STEPS), 255, np.uint8)
    successor_robot_link = np.full((12, 12, PHYSICS_STEPS), -1, np.int16)
    successor_other_link = np.full((12, 12, PHYSICS_STEPS), -1, np.int16)
    rows = []
    current_runs = []
    current_matches = []
    for current_candidate in range(12):
        record, contact, robot_link, other_link, successor = _execute_one_tick(
            ctx, snapshot, current_candidate, topology, link_names, object_names
        )
        current_contact[current_candidate] = contact
        current_robot_link[current_candidate] = robot_link
        current_other_link[current_candidate] = other_link
        if expected_current is not None:
            expected = expected_current[current_candidate]
            current_matches.append(
                bool(np.array_equal(np.asarray(record["target_command"], np.float32), expected["target"]))
                and bool(np.array_equal(contact, expected["contact"]))
            )
        safe_prefix = not bool(contact.any())
        successor_safe_indices: list[int] = []
        if safe_prefix:
            for successor_candidate in range(12):
                second, second_contact, second_robot, second_other, _ = _execute_one_tick(
                    ctx, successor, successor_candidate, topology, link_names, object_names
                )
                successor_contact[current_candidate, successor_candidate] = second_contact
                successor_robot_link[current_candidate, successor_candidate] = second_robot
                successor_other_link[current_candidate, successor_candidate] = second_other
                safe_second = not bool(second_contact.any())
                if safe_second:
                    successor_safe_indices.append(successor_candidate)
                rows.append(
                    {
                        "current_candidate": current_candidate,
                        "successor_identity": f"{snapshot.identity['state_id']}:succ:{current_candidate:02d}",
                        "successor_candidate": successor_candidate,
                        "successor_branch_identity": f"{snapshot.identity['state_id']}:succ:{current_candidate:02d}:{successor_candidate:02d}",
                        "successor_contact": bool(second_contact.any()),
                        "successor_first_contact_step": second["first_contact_step"],
                    }
                )
        record.update(
            {
                "branch_identity": f"{snapshot.identity['state_id']}:prefix:{current_candidate:02d}",
                "safe_prefix": safe_prefix,
                "viable": bool(successor_safe_indices),
                "successor_safe_candidate_indices": successor_safe_indices,
                "successor_safe_candidate_count": len(successor_safe_indices),
            }
        )
        current_runs.append(record)
    arrays = {
        "current_contact": current_contact,
        "current_robot_link": current_robot_link,
        "current_other_link": current_other_link,
        "successor_contact": successor_contact,
        "successor_robot_link": successor_robot_link,
        "successor_other_link": successor_other_link,
    }
    return current_runs, rows, arrays, current_matches, link_names, object_names


def run_fixture() -> dict:
    fixture = REDUCE.fixture_payload()
    if not fixture["pass"] or fixture != REDUCE.fixture_payload():
        raise RuntimeError("pure deterministic fixture failed")
    panel = {row["state_id"]: row for row in json.loads(FIXTURE_PANEL.read_text())["states"]}
    state = panel["scale-fit-0-00"]
    ctx, snapshot = _build_current(state)
    topology = V.link_topology(ctx)
    solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    first, c1, r1, o1, successor = _execute_one_tick(ctx, snapshot, 11, topology, link_names, object_names)
    second, c2, r2, o2, _ = _execute_one_tick(ctx, snapshot, 11, topology, link_names, object_names)
    next1, c3, r3, o3, _ = _execute_one_tick(ctx, successor, 11, topology, link_names, object_names)
    next2, c4, r4, o4, _ = _execute_one_tick(ctx, successor, 11, topology, link_names, object_names)
    deterministic = all(
        np.array_equal(a, b)
        for a, b in ((c1, c2), (r1, r2), (o1, o2), (c3, c4), (r3, r4), (o3, o4))
    ) and first["endpoint_pose"] == second["endpoint_pose"] and next1["endpoint_pose"] == next2["endpoint_pose"]
    result = {
        "schema": "one_tick_viability_fixture_result_v1",
        "pure": fixture,
        "training_state": state["state_id"],
        "snapshot_restore_deterministic": deterministic,
        "current_branch_repeated": 2,
        "successor_branch_repeated": 2,
        "row_ledger_serialization": True,
        "pass": bool(fixture["pass"] and deterministic),
    }
    result["content_digest"] = REDUCE.digest(result)
    del ctx
    gc.collect()
    if not result["pass"]:
        raise RuntimeError("simulation restoration fixture failed")
    return result


def collect_state(index: int) -> dict:
    panel = json.loads(PANEL.read_text())
    state = panel["states"][index]
    state_id = str(state["state_id"])
    record_path = OUT / "states" / f"{state_id}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            print(json.dumps({"state_id": state_id, "status": "REUSED"}), flush=True)
            return record
    started = time.time()
    wide = json.loads((WIDE_STATES / f"{state_id}.json").read_text())
    exact_record = records(EXACT_INDEX)[state_id]
    with np.load(exact_record["shard_path"], allow_pickle=False) as loaded:
        exact = {key: np.asarray(loaded[key]) for key in loaded.files}
    expected = {
        int(branch["candidate_index"]): {
            "target": np.asarray(branch["post_slew"][0][0], np.float32),
            "contact": exact["frozen_contact"][int(branch["candidate_index"]), :PHYSICS_STEPS].astype(np.uint8),
        }
        for branch in wide["branches"]
    }
    ctx, snapshot = _build_current(state)
    current, successors, arrays, matches, link_names, object_names = _run_tree(ctx, snapshot, expected)
    if len(matches) != 12 or not all(matches):
        raise RuntimeError(f"{state_id}: registered one-tick replay mismatch")
    shard = CACHE / "states" / f"{state_id}.npz"
    atomic_npz(shard, **arrays)
    record = {
        "schema": "one_tick_viability_tree_state_v1",
        "status": "PASS",
        "state_index": index,
        "state_id": state_id,
        "scene_id": state["scene_id"],
        "family": state["family"],
        "split": state["split"],
        "current_snapshot_digest": snapshot.digest,
        "current_prefix_branches": 12,
        "successor_branches": len(successors),
        "safe_current_prefixes": sum(row["safe_prefix"] for row in current),
        "viability_admissible_prefixes": sum(row["safe_prefix"] and row["viable"] for row in current),
        "current": current,
        "successors": successors,
        "link_names": {str(key): value for key, value in link_names.items()},
        "object_names": {str(key): value for key, value in object_names.items()},
        "registered_action_and_contact_matches": sum(matches),
        "shard_path": str(shard),
        "shard_sha256": sha(shard),
        "storage_bytes": shard.stat().st_size,
        "runtime_s": time.time() - started,
    }
    record["content_digest"] = REDUCE.digest(record)
    atomic_json(record_path, record)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": state_id, "status": "PASS", "successors": len(successors), "runtime_s": record["runtime_s"]}), flush=True)
    return record


def collect_predecessor(state_id: str) -> dict:
    panel = {row["state_id"]: row for row in json.loads(PANEL.read_text())["states"]}
    state = panel[state_id]
    output = OUT / "predecessors" / f"{state_id}.json"
    if output.is_file():
        record = json.loads(output.read_text())
        shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            print(json.dumps({"predecessor": state_id, "status": "REUSED"}), flush=True)
            return record
    started = time.time()
    ctx, snapshot = _build_predecessor(state)
    current, successors, arrays, _matches, link_names, object_names = _run_tree(ctx, snapshot)
    shard = CACHE / "predecessors" / f"{state_id}.npz"
    atomic_npz(shard, **arrays)
    record = {
        "schema": "one_tick_predecessor_viability_tree_v1",
        "status": "PASS",
        "original_state_id": state_id,
        "predecessor_identity": snapshot.identity["state_id"],
        "family": state["family"],
        "split": state["split"],
        "snapshot_digest": snapshot.digest,
        "current_prefix_branches": 12,
        "successor_branches": len(successors),
        "safe_prefixes": sum(row["safe_prefix"] for row in current),
        "viability_admissible_prefixes": sum(row["safe_prefix"] and row["viable"] for row in current),
        "current": current,
        "successors": successors,
        "link_names": {str(key): value for key, value in link_names.items()},
        "object_names": {str(key): value for key, value in object_names.items()},
        "shard_path": str(shard),
        "shard_sha256": sha(shard),
        "storage_bytes": shard.stat().st_size,
        "runtime_s": time.time() - started,
    }
    record["content_digest"] = REDUCE.digest(record)
    atomic_json(output, record)
    del ctx
    gc.collect()
    print(json.dumps({"predecessor": state_id, "status": "PASS", "successors": len(successors), "runtime_s": record["runtime_s"]}), flush=True)
    return record


def finalize() -> dict:
    panel = json.loads(PANEL.read_text())
    state_records = []
    for state in panel["states"]:
        path = OUT / "states" / f"{state['state_id']}.json"
        if not path.is_file():
            raise RuntimeError(f"missing state result {state['state_id']}")
        record = json.loads(path.read_text())
        if record.get("status") != "PASS" or sha(Path(record["shard_path"])) != record["shard_sha256"]:
            raise RuntimeError(f"invalid state result {state['state_id']}")
        state_records.append(record)
    previous = json.loads(PREVIOUS.read_text())
    failed = sorted(
        row["state_id"]
        for row in previous["horizon_results"]["combined"]["1"]["per_state"]
        if int(row["contact_negative_candidates"]) == 0
    )
    predecessor_records = []
    for state_id in failed:
        path = OUT / "predecessors" / f"{state_id}.json"
        if not path.is_file():
            raise RuntimeError(f"missing predecessor result {state_id}")
        record = json.loads(path.read_text())
        if record.get("status") != "PASS" or sha(Path(record["shard_path"])) != record["shard_sha256"]:
            raise RuntimeError(f"invalid predecessor result {state_id}")
        predecessor_records.append(record)
    wall = json.loads((OUT / "collection_wall_receipt.json").read_text())
    payload = {
        "schema": "one_tick_viability_tree_index_v1",
        "source_commit": SOURCE_COMMIT,
        "complete": True,
        "states": 48,
        "historical_current_prefixes_replayed_as_one_tick": sum(row["current_prefix_branches"] for row in state_records),
        "new_successor_branches": sum(row["successor_branches"] for row in state_records),
        "predecessor_states": len(predecessor_records),
        "predecessor_prefix_branches": sum(row["current_prefix_branches"] for row in predecessor_records),
        "predecessor_successor_branches": sum(row["successor_branches"] for row in predecessor_records),
        "state_records": state_records,
        "predecessor_records": predecessor_records,
        "fixture": json.loads((OUT / "fixture.json").read_text()),
        "runtime": {
            "state_compute_s": sum(float(row["runtime_s"]) for row in state_records),
            "predecessor_compute_s": sum(float(row["runtime_s"]) for row in predecessor_records),
            "parallel_wall_s": wall["wall_runtime_s"],
            "fixture_s": wall["fixture_runtime_s"],
        },
        "storage_bytes": sum(int(row["storage_bytes"]) for row in state_records + predecessor_records),
        "bindings": {
            "panel_sha256": sha(PANEL),
            "exact_index_sha256": sha(EXACT_INDEX),
            "previous_result_sha256": sha(PREVIOUS),
        },
    }
    payload["content_digest"] = REDUCE.digest(payload)
    atomic_json(OUT / "viability_tree_index.json", payload)
    print(json.dumps({key: payload[key] for key in ("states", "historical_current_prefixes_replayed_as_one_tick", "new_successor_branches", "predecessor_states", "predecessor_prefix_branches", "predecessor_successor_branches", "runtime", "storage_bytes", "content_digest")}, indent=2))
    return payload


def collect_all() -> None:
    started = time.time()
    fixture_started = time.time()
    fixture = run_fixture()
    atomic_json(OUT / "fixture.json", fixture)
    fixture_runtime = time.time() - fixture_started
    logs = CACHE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, 48, 4):
        jobs = []
        for index in range(start, min(start + 4, 48)):
            log = logs / f"state_{index:03d}.log"
            stream = log.open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((index, process, stream, log))
        for index, process, stream, log in jobs:
            code = process.wait()
            stream.close()
            if code:
                raise RuntimeError(f"state {index} failed with {code}; see {log}")
    previous = json.loads(PREVIOUS.read_text())
    failed = sorted(
        row["state_id"]
        for row in previous["horizon_results"]["combined"]["1"]["per_state"]
        if int(row["contact_negative_candidates"]) == 0
    )
    for state_id in failed:
        collect_predecessor(state_id)
    atomic_json(
        OUT / "collection_wall_receipt.json",
        {
            "states": 48,
            "parallel_processes": 4,
            "predecessor_states": failed,
            "fixture_runtime_s": fixture_runtime,
            "wall_runtime_s": time.time() - started,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixtures", action="store_true")
    group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-predecessor")
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.fixtures:
        fixture = run_fixture()
        atomic_json(OUT / "fixture.json", fixture)
        print(json.dumps(fixture, indent=2))
    elif args.collect_state is not None:
        collect_state(args.collect_state)
    elif args.collect_predecessor:
        collect_predecessor(args.collect_predecessor)
    elif args.collect_all:
        collect_all()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
