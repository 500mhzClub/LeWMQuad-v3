#!/usr/bin/env python3
"""Materialise solver-native and history-free Genesis contact evidence.

The only simulation in this tool is deterministic replay of the 576 already
registered branches.  It captures the pre-step articulated configuration and
the native contact manifold.  After a branch has finished, the same Genesis
0.3.14 collider is queried at each captured configuration without advancing
dynamics.  No learned checkpoint is imported or executed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.oracle import go2_branch_oracle_v1_2 as ORACLE
from lewm.safety import genesis_narrowphase_reconciliation_v1 as REDUCE
from scripts import run_go2_oracle_branch_pilot_v1_2 as V

OUT = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/genesis_narrowphase_candidate_feasibility_v1")
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
ENHANCED = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
OLD_INDEX = ROOT / ".generated/h1_articulated_swept_geometry_sufficiency_v1/articulated_geometry_index.json"
PHYSICS_STEPS = 250


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def arr(value):
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def records(path: Path) -> dict[str, dict]:
    return {str(row["state_id"]): row for row in json.loads(path.read_text())["state_records"]}


def contact_arrays(contacts) -> dict[str, np.ndarray]:
    if contacts is None:
        return {}
    return {str(k): arr(v) for k, v in contacts.items()}


def selected_contacts(contacts, topology, *, force_threshold: bool) -> list[int]:
    c = contact_arrays(contacts)
    a = np.asarray(c.get("link_a", ())).reshape(-1)
    b = np.asarray(c.get("link_b", ())).reshape(-1)
    if not len(a):
        return []
    forces = None
    for key in ("force_a", "force"):
        if key in c:
            value = np.asarray(c[key])
            if value.size:
                forces = np.linalg.norm(value.reshape(-1, 3), axis=-1)
                break
    low, high = topology["robot_link_range"]
    output = []
    for i, (left, right) in enumerate(zip(a, b, strict=True)):
        left = int(left); right = int(right)
        l_robot = low <= left < high; r_robot = low <= right < high
        if l_robot == r_robot:
            continue
        robot_link = left if l_robot else right
        other = right if l_robot else left
        if force_threshold and forces is not None and i < len(forces) and float(forces[i]) <= ORACLE.CONTACT_FORCE_THRESHOLD_N:
            continue
        if other in topology["ground_link_indices"] and robot_link in topology["foot_link_indices"]:
            continue
        output.append(i)
    return output


def contact_summary(contacts, topology, link_names, object_names, *, force_threshold: bool) -> dict:
    c = contact_arrays(contacts); ids = selected_contacts(c, topology, force_threshold=force_threshold)
    empty = {"active": False, "count": 0, "robot_link": -1, "other_link": -1, "robot_geom": -1,
             "other_geom": -1, "penetration": np.nan, "force": np.nan,
             "position": np.full(3, np.nan), "normal": np.full(3, np.nan)}
    if not ids:
        return empty
    a = np.asarray(c["link_a"]).reshape(-1); b = np.asarray(c["link_b"]).reshape(-1)
    ga = np.asarray(c.get("geom_a", np.full_like(a, -1))).reshape(-1)
    gb = np.asarray(c.get("geom_b", np.full_like(b, -1))).reshape(-1)
    penetration = np.asarray(c.get("penetration", np.full(len(a), np.nan))).reshape(-1)
    force_vector = None
    for key in ("force_a", "force"):
        if key in c and np.asarray(c[key]).size:
            force_vector = np.asarray(c[key]).reshape(-1, 3); break
    forces = np.full(len(a), np.nan) if force_vector is None else np.linalg.norm(force_vector, axis=-1)
    # Native replay prioritises force; history-free queries have zero force and
    # prioritise penetration.  Both choices are deterministic.
    if force_threshold and np.any(np.isfinite(forces[ids])):
        chosen = ids[int(np.nanargmax(forces[ids]))]
    elif np.any(np.isfinite(penetration[ids])):
        chosen = ids[int(np.nanargmax(penetration[ids]))]
    else:
        chosen = ids[0]
    low, high = topology["robot_link_range"]
    left, right = int(a[chosen]), int(b[chosen]); left_robot = low <= left < high
    robot_link = left if left_robot else right; other_link = right if left_robot else left
    robot_geom = int(ga[chosen] if left_robot else gb[chosen]); other_geom = int(gb[chosen] if left_robot else ga[chosen])
    position = np.asarray(c.get("position", np.full((len(a), 3), np.nan))).reshape(-1, 3)[chosen]
    normal = np.asarray(c.get("normal", np.full((len(a), 3), np.nan))).reshape(-1, 3)[chosen]
    return {"active": True, "count": len(ids), "robot_link": robot_link, "other_link": other_link,
            "robot_geom": robot_geom, "other_geom": other_geom,
            "penetration": float(penetration[chosen]), "force": float(forces[chosen]),
            "position": position.astype(np.float32), "normal": normal.astype(np.float32),
            "robot_link_name": link_names.get(robot_link, "unresolved"),
            "environment_object": object_names.get(other_link, "unresolved")}


def make_metadata_arrays() -> dict[str, np.ndarray]:
    shape = (12, PHYSICS_STEPS)
    return {
        "native_contact": np.zeros(shape, np.uint8), "native_count": np.zeros(shape, np.int16),
        "native_robot_link": np.full(shape, -1, np.int16), "native_other_link": np.full(shape, -1, np.int16),
        "native_robot_geom": np.full(shape, -1, np.int16), "native_other_geom": np.full(shape, -1, np.int16),
        "native_penetration": np.full(shape, np.nan, np.float32), "native_force": np.full(shape, np.nan, np.float32),
        "native_position": np.full(shape + (3,), np.nan, np.float32), "native_normal": np.full(shape + (3,), np.nan, np.float32),
        "exact_contact": np.zeros(shape, np.uint8), "exact_count": np.zeros(shape, np.int16),
        "exact_robot_link": np.full(shape, -1, np.int16), "exact_other_link": np.full(shape, -1, np.int16),
        "exact_robot_geom": np.full(shape, -1, np.int16), "exact_other_geom": np.full(shape, -1, np.int16),
        "exact_penetration": np.full(shape, np.nan, np.float32),
        "exact_position": np.full(shape + (3,), np.nan, np.float32), "exact_normal": np.full(shape + (3,), np.nan, np.float32),
    }


def assign_summary(store, prefix: str, candidate: int, step: int, summary: dict) -> None:
    store[f"{prefix}_contact"][candidate, step] = summary["active"]
    store[f"{prefix}_count"][candidate, step] = summary["count"]
    for key in ("robot_link", "other_link", "robot_geom", "other_geom", "penetration", "position", "normal"):
        store[f"{prefix}_{key}"][candidate, step] = summary[key]
    if prefix == "native":
        store["native_force"][candidate, step] = summary["force"]


def execute_branch(ctx, snapshot, candidate, expected_action, candidate_index, qpos_pre, store, topology, link_names, object_names):
    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner; counter = 0; original = runner._step_policy_step

    def instrumented(_runner, target_cmd):
        nonlocal counter
        observation = _runner._build_observation(target_cmd)
        targets = _runner.policy.act(observation)
        _runner._apply_joint_targets(targets)
        for _ in range(int(_runner._physics_steps_per_policy)):
            qpos_pre[candidate_index, counter] = arr(_runner.build.robot.get_qpos()).astype(np.float32)
            _runner.build.scene.step()
            summary = contact_summary(_runner.build.robot.get_contacts(), topology, link_names, object_names, force_threshold=True)
            assign_summary(store, "native", candidate_index, counter, summary)
            counter += 1
        _runner._sim_time_ns += _runner._policy_dt_ns

    runner._step_policy_step = types.MethodType(instrumented, runner)
    try:
        requested = V.V1.block_for(candidate[1][0])[None, ...]
        block = runner.execute_requested_block(requested)
    finally:
        runner._step_policy_step = original
    action_match = bool(np.array_equal(np.asarray(block.executed, np.float32)[0], expected_action))
    return action_match, counter


def collect_state(index: int):
    panel = json.loads(PANEL.read_text()); state = panel["states"][index]; sid = state["state_id"]
    out = OUT / "states" / f"{sid}.json"
    if out.is_file():
        rec = json.loads(out.read_text()); shard = Path(rec["shard_path"])
        if rec.get("status") == "PASS" and shard.is_file() and sha(shard) == rec["shard_sha256"]:
            print(json.dumps({"state_id": sid, "status": "REUSED"}), flush=True); return rec
    started = time.time(); enhanced = records(ENHANCED)[sid]; old = records(OLD_INDEX)[sid]
    with np.load(enhanced["shard_path"], allow_pickle=False) as f:
        sensor = {k: np.asarray(f[k]) for k in f.files}
    with np.load(old["shard_path"], allow_pickle=False) as f:
        previous = {k: np.asarray(f[k]) for k in f.files}
    branches = {int(x["candidate_index"]): x for x in enhanced["branches"]}
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(int(state["warmup_blocks"])):
        ctx.drive_one_block()
    topology = V.link_topology(ctx); eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise RuntimeError(f"{sid}: eligibility changed {eligible}")
    goal, _ = eligible
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(goal["goal"]), identity={"state_id": sid, "scene_id": state["scene_id"], "family": state["family"]})
    robot = ctx.build.robot; solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    boundary_native = contact_summary(robot.get_contacts(), topology, link_names, object_names, force_threshold=True)
    qpos_pre = np.empty((12, PHYSICS_STEPS, 19), np.float32); store = make_metadata_arrays(); action_matches = []
    for ci, candidate in enumerate(V.V1.CANDIDATE_BANK):
        match, count = execute_branch(ctx, snapshot, candidate, np.asarray(branches[ci]["post_slew"][0], np.float32), ci, qpos_pre, store, topology, link_names, object_names)
        action_matches.append(match)
        if count != PHYSICS_STEPS:
            raise RuntimeError(f"{sid}:{ci}: physics step count {count}")

    # History-free exact queries occur only after every native replay branch,
    # so collider.clear() cannot perturb any recorded trajectory.
    max_fk_position_error = 0.0; max_fk_orientation_error = 0.0
    for ci in range(12):
        for step in range(PHYSICS_STEPS):
            robot.set_qpos(qpos_pre[ci, step])
            # For step>0, the pre-step FK should equal the prior persisted
            # post-step transform.  This is the temporal-alignment receipt.
            if step > 0:
                lp = arr(robot.get_links_pos()).astype(np.float64)
                lq = arr(robot.get_links_quat()).astype(np.float64)
                ref = previous["link_transform"][ci, step - 1]
                max_fk_position_error = max(max_fk_position_error, float(np.max(np.abs(lp - ref[:, :3]))))
                lq = lq / np.maximum(np.linalg.norm(lq, axis=-1, keepdims=True), 1e-12)
                rq = ref[:, 3:].astype(np.float64)
                rq = rq / np.maximum(np.linalg.norm(rq, axis=-1, keepdims=True), 1e-12)
                dot = np.clip(np.abs(np.sum(lq * rq, axis=-1)), 0.0, 1.0)
                max_fk_orientation_error = max(max_fk_orientation_error, float(np.max(2 * np.arccos(dot))))
            robot.detect_collision()
            contacts = solver.collider.get_contacts(as_tensor=False)
            summary = contact_summary(contacts, topology, link_names, object_names, force_threshold=False)
            assign_summary(store, "exact", ci, step, summary)

    frozen = previous["physics_contact"].astype(bool); native = store["native_contact"].astype(bool); exact = store["exact_contact"].astype(bool)
    native_match = bool(np.array_equal(frozen, native))
    native_branch_match = np.all(frozen == native, axis=1)
    exact_branch_match = np.any(frozen, axis=1) == np.any(exact, axis=1)
    arrays = {"qpos_pre": qpos_pre, "frozen_contact": frozen.astype(np.uint8), "approx_clearance": previous["clearance"][:, :, 0].astype(np.float32), **store}
    shard = CACHE / "states" / f"{sid}.npz"; atomic_npz(shard, **arrays)
    rec = {
        "schema": "genesis_narrowphase_candidate_feasibility_state_v1", "status": "PASS" if native_match and all(action_matches) else "MISMATCH",
        "state_index": index, "state_id": sid, "scene_id": state["scene_id"], "family": state["family"], "split": state["split"],
        "branches": 12, "physics_steps_per_branch": PHYSICS_STEPS, "replayed_branches": 12,
        "action_trace_matches": int(sum(action_matches)), "native_trace_exact": native_match,
        "native_branch_matches": int(native_branch_match.sum()), "exact_branch_matches": int(exact_branch_match.sum()),
        "boundary_native_contact": bool(boundary_native["active"]), "boundary_native_robot_link": boundary_native.get("robot_link_name"),
        "boundary_native_environment_object": boundary_native.get("environment_object"),
        "max_fk_position_error_m": max_fk_position_error, "max_fk_orientation_error_rad": max_fk_orientation_error,
        "link_names": {str(k): v for k, v in link_names.items()}, "object_names": {str(k): v for k, v in object_names.items()},
        "shard_path": str(shard), "shard_sha256": sha(shard), "storage_bytes": shard.stat().st_size, "runtime_s": time.time() - started,
    }
    rec["content_digest"] = REDUCE.digest(rec); atomic_json(out, rec)
    print(json.dumps({"state_id": sid, "status": rec["status"], "native_trace_exact": native_match,
                      "exact_branch_matches": rec["exact_branch_matches"], "runtime_s": rec["runtime_s"]}), flush=True)
    del ctx; gc.collect(); return rec


def fixture():
    payload = REDUCE.fixture_payload(); atomic_json(OUT / "fixture.json", payload)
    if not payload["pass"] or not payload["byte_identical_regeneration"]:
        raise RuntimeError(payload)
    print(json.dumps(payload, indent=2)); return payload


def finalize():
    panel = json.loads(PANEL.read_text()); rows = []
    for state in panel["states"]:
        rec = json.loads((OUT / "states" / f"{state['state_id']}.json").read_text())
        if rec["status"] != "PASS" or sha(Path(rec["shard_path"])) != rec["shard_sha256"]:
            raise RuntimeError(f"bad state receipt: {state['state_id']}")
        rows.append(rec)
    wall = json.loads((OUT / "collection_wall_receipt.json").read_text())
    payload = {
        "schema": "genesis_narrowphase_candidate_feasibility_index_v1", "states": 48, "branches": 576,
        "physics_steps": 144000, "replayed_branches": 576, "new_identities": 0,
        "native_trace_exact_states": sum(r["native_trace_exact"] for r in rows),
        "native_branch_matches": sum(r["native_branch_matches"] for r in rows),
        "exact_branch_matches": sum(r["exact_branch_matches"] for r in rows),
        "action_trace_matches": sum(r["action_trace_matches"] for r in rows),
        "maximum_fk_position_error_m": max(r["max_fk_position_error_m"] for r in rows),
        "maximum_fk_orientation_error_rad": max(r["max_fk_orientation_error_rad"] for r in rows),
        "state_records": rows, "runtime_compute_s": sum(r["runtime_s"] for r in rows),
        "parallel_wall_runtime_s": wall["wall_runtime_s"], "storage_bytes": sum(r["storage_bytes"] for r in rows),
        "contract": {
            "genesis_version": "0.3.14", "collider": "broadphase + MPR, GJK fallback", "multi_contact": True,
            "box_box_detection": False, "collision_margin_m": 0.0, "contact_force_threshold_n": ORACLE.CONTACT_FORCE_THRESHOLD_N,
            "query_timing": "pre-physics-step articulated qpos; history-free detect_collision after all native branch replays",
            "pair_filter": "robot/environment only; excludes self-contact and permitted calf/foot-ground support",
        },
        "bindings": {"panel_sha256": sha(PANEL), "enhanced_index_sha256": sha(ENHANCED), "prior_geometry_index_sha256": sha(OLD_INDEX)},
    }
    payload["content_digest"] = REDUCE.digest(payload); atomic_json(OUT / "narrowphase_index.json", payload)
    print(json.dumps({k: payload[k] for k in ("states", "branches", "native_branch_matches", "exact_branch_matches", "runtime_compute_s", "parallel_wall_runtime_s", "storage_bytes", "content_digest")}, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture", action="store_true"); group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        fixture()
    elif args.collect_state is not None:
        collect_state(args.collect_state)
    elif args.collect_all:
        fixture(); started = time.time(); logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True)
        for start in range(0, 48, 4):
            processes = []
            for index in range(start, min(start + 4, 48)):
                stream = (logs / f"state_{index:03d}.log").open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
                processes.append((index, process, stream))
            for index, process, stream in processes:
                code = process.wait(); stream.close()
                if code:
                    raise RuntimeError(f"state {index} failed; see {logs / f'state_{index:03d}.log'}")
        atomic_json(OUT / "collection_wall_receipt.json", {"parallel_processes": 4, "wall_runtime_s": time.time() - started})
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
