#!/usr/bin/env python3
"""Replay the frozen development panel and instrument disallowed contacts.

The replay adds read-only physics-step contact instrumentation around the
unchanged controller/physics loop.  It creates no state or candidate identity
and does not alter the historical binary labels.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import contact_hazard_ontology_v1 as ONTOLOGY
from scripts import materialize_dense_route_intent_true_future_v1 as DENSE
from scripts import materialize_enhanced_embodied_safety_observability_v2 as EMBODIED
from scripts import replay_safe_local_waypoint_route_intent_v2 as REPLAY
from scripts import run_go2_oracle_branch_pilot_v1_2 as V


OUT = ROOT / ".generated/contact_hazard_ontology_and_instrumentation_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/contact_hazard_ontology_and_instrumentation_v1")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
PHYSICS_DT_S = ONTOLOGY.PHYSICS_DT_S


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n", compresslevel=6) as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(temporary, path)


def _array(value: Any) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def quat_rotation_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def roll_pitch_wxyz(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    return roll, pitch


def scene_metadata(ctx: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_objects = {
        str(row["object_id"]): row
        for field in ("walls", "obstacles", "landmarks")
        for row in manifest.get(field, [])
    }
    links: dict[int, dict[str, Any]] = {}
    for entity in ctx.build.scene.entities:
        entity_name = str(getattr(entity, "name", type(entity.morph).__name__))
        morph_name = type(entity.morph).__name__
        for link in getattr(entity, "links", ()):
            global_id = int(link.idx)
            source = manifest_objects.get(entity_name)
            if entity is ctx.build.robot:
                object_class = "robot"
            elif morph_name == "Plane":
                object_class = "ground"
            elif source is not None:
                object_class = str(source.get("kind", "static_object"))
            else:
                object_class = "unresolved"
            # Current maze manifests do not encode consequence properties.
            properties = {
                "fixed": None if entity is ctx.build.robot else True,
                "mass_kg": None,
                "mechanical_material": None,
                "visual_material_id": None if source is None else source.get("material_id"),
                "fragility_category": None,
                "safety_critical": None,
                "human_or_person_proxy": False if object_class in {"ground", "wall", "landmark"} else None,
                "permitted_contact": True if object_class == "ground" else None,
                "prohibited_contact": None,
                "damage_observed": None,
            }
            links[global_id] = {
                "link_name": str(getattr(link, "name", global_id)),
                "entity_name": entity_name,
                "object_id": "ground_plane" if object_class == "ground" else entity_name,
                "object_class": object_class,
                "properties": properties,
            }
    return links


def pre_physics_state(runner: Any) -> dict[str, np.ndarray]:
    robot = runner.build.robot
    return {
        "base_pos": _array(robot.get_pos()).astype(np.float64),
        "base_quat": _array(robot.get_quat()).astype(np.float64),
        "base_vel": _array(robot.get_vel()).astype(np.float64),
        "base_ang": _array(robot.get_ang()).astype(np.float64),
        "link_pos": _array(robot.get_links_pos()).astype(np.float64),
        "link_vel": _array(robot.get_links_vel()).astype(np.float64),
        "link_ang": _array(robot.get_links_ang()).astype(np.float64),
        "joint_vel": _array(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())).astype(np.float64),
    }


def post_response(runner: Any, before: dict[str, np.ndarray]) -> dict[str, Any]:
    robot = runner.build.robot
    quat = _array(robot.get_quat()).astype(np.float64)
    gravity_before = quat_rotation_wxyz(before["base_quat"]).T @ np.asarray([0., 0., -1.])
    gravity_after = quat_rotation_wxyz(quat).T @ np.asarray([0., 0., -1.])
    joint_after = _array(robot.get_dofs_velocity(runner._leg_dof_idx.tolist())).astype(np.float64)
    torque = _array(robot.get_dofs_control_force(runner._leg_dof_idx.tolist())).astype(np.float64)
    all_forces = _array(robot.get_links_net_contact_force()).astype(np.float64)
    foot_local = [int(next(link for link in robot.links if link.name == name).idx_local)
                  for name in EMBODIED.FOOT_NAMES]
    return {
        "body_linear_velocity_change_m_s": float(np.linalg.norm(_array(robot.get_vel()) - before["base_vel"])),
        "body_angular_velocity_change_rad_s": float(np.linalg.norm(_array(robot.get_ang()) - before["base_ang"])),
        "projected_gravity_change": float(np.linalg.norm(gravity_after - gravity_before)),
        "joint_velocity_change_rad_s": float(np.linalg.norm(joint_after - before["joint_vel"])),
        "joint_acceleration_peak_rad_s2": float(np.max(np.abs((joint_after - before["joint_vel"]) / PHYSICS_DT_S))),
        "actuator_torque_peak_nm": float(np.max(np.abs(torque))),
        "support_contact_force_peak_n": float(np.max(np.linalg.norm(all_forces[foot_local], axis=-1))),
    }


def contact_points(
    ctx: Any, topology: dict[str, Any], link_meta: dict[int, dict[str, Any]],
    before: dict[str, np.ndarray], response: dict[str, Any], *, branch_id: str,
    state: dict[str, Any], candidate_index: int, physics_step: int,
    command: np.ndarray, branch_stuck: bool, route_progress_m: float,
) -> list[dict[str, Any]]:
    robot = ctx.build.robot
    contacts = robot.get_contacts(exclude_self_contact=False)
    if not contacts:
        return []
    arrays = {key: _array(value) for key, value in contacts.items()}
    link_a = arrays.get("link_a", np.empty(0, dtype=np.int64)).reshape(-1)
    if not len(link_a):
        return []
    link_b = arrays["link_b"].reshape(-1)
    low, high = topology["robot_link_range"]
    base_pos = _array(robot.get_pos()).astype(np.float64)
    base_quat = _array(robot.get_quat()).astype(np.float64)
    body_from_world = quat_rotation_wxyz(base_quat).T
    rows: list[dict[str, Any]] = []
    for index, (a_raw, b_raw) in enumerate(zip(link_a, link_b, strict=True)):
        a, b = int(a_raw), int(b_raw)
        a_robot, b_robot = low <= a < high, low <= b < high
        if a_robot == b_robot:
            continue
        robot_is_a = a_robot
        robot_link = a if robot_is_a else b
        environment_link = b if robot_is_a else a
        force_key = "force_a" if robot_is_a else "force_b"
        force = np.asarray(arrays.get(force_key, np.zeros((len(link_a), 3)))[index], np.float64)
        magnitude = float(np.linalg.norm(force))
        if not ONTOLOGY.is_disallowed_contact(
            robot_link_id=robot_link, environment_link_id=environment_link,
            foot_link_ids=topology["foot_link_indices"],
            ground_link_ids=topology["ground_link_indices"],
            self_contact=False, force_magnitude_n=magnitude):
            continue
        point = np.asarray(arrays.get("position", np.full((len(link_a), 3), np.nan))[index], np.float64)
        normal = np.asarray(arrays.get("normal", np.full((len(link_a), 3), np.nan))[index], np.float64)
        normal_norm = float(np.linalg.norm(normal))
        normal_unit = normal / normal_norm if normal_norm > 1e-12 else np.full(3, np.nan)
        penetration = float(np.asarray(arrays.get("penetration", np.full(len(link_a), np.nan))).reshape(-1)[index])
        local = robot_link - int(robot.link_start)
        pre_com = before["link_pos"][local]
        pre_velocity = before["link_vel"][local] + np.cross(before["link_ang"][local], point - pre_com)
        relative = pre_velocity  # all current environment objects are fixed
        relative_normal = abs(float(np.dot(relative, normal_unit))) if np.isfinite(normal_unit).all() else None
        relative_tangent = (
            float(np.linalg.norm(relative - np.dot(relative, normal_unit) * normal_unit))
            if np.isfinite(normal_unit).all() else None)
        tangential_force = (
            float(np.linalg.norm(force - np.dot(force, normal_unit) * normal_unit))
            if np.isfinite(normal_unit).all() else None)
        point_body = body_from_world @ (point - base_pos)
        roll, pitch = roll_pitch_wxyz(base_quat)
        loss_stability = (
            float(base_pos[2]) < float(ctx.runner.config.fall_z_threshold_m)
            or max(abs(roll), abs(pitch)) > float(ctx.runner.config.tip_threshold_rad))
        metadata = link_meta.get(environment_link, {
            "link_name": None, "object_id": f"link-{environment_link}",
            "object_class": "unresolved", "properties": {},
        })
        row = {
            "schema": "lewm_contact_hazard_raw_point_v1",
            "branch_id": branch_id, "state_id": str(state["state_id"]),
            "scene_id": str(state["scene_id"]), "family": str(state["family"]),
            "candidate_index": int(candidate_index), "physics_step": int(physics_step),
            "policy_step": int((physics_step - 1) // int(ctx.runner._physics_steps_per_policy) + 1),
            "tick": int((physics_step - 1) // (
                int(ctx.runner._physics_steps_per_policy) * int(ctx.runner._policy_steps_per_command_tick)) + 1),
            "physics_step_within_tick": int((physics_step - 1) % (
                int(ctx.runner._physics_steps_per_policy) * int(ctx.runner._policy_steps_per_command_tick)) + 1),
            "contact_point_index": int(index), "robot_link_id": robot_link,
            "robot_link_name": link_meta.get(robot_link, {}).get("link_name"),
            "environment_link_id": environment_link,
            "environment_link_name": metadata.get("link_name"),
            "environment_object_id": metadata.get("object_id"),
            "environment_object_class": metadata.get("object_class"),
            "environment_properties": metadata.get("properties"),
            "contact_point_world_m": point.tolist(), "contact_point_body_m": point_body.tolist(),
            "contact_normal_world": normal.tolist(), "penetration_m": penetration,
            "force_robot_world_n": force.tolist(), "force_magnitude_n": magnitude,
            "normal_force_n": abs(float(np.dot(force, normal_unit))) if np.isfinite(normal_unit).all() else magnitude,
            "normal_impulse_increment_n_s": (
                abs(float(np.dot(force, normal_unit))) if np.isfinite(normal_unit).all() else magnitude) * PHYSICS_DT_S,
            "tangential_force_n": tangential_force,
            "tangential_impulse_increment_n_s": (
                None if tangential_force is None else tangential_force * PHYSICS_DT_S),
            "relative_normal_speed_m_s": relative_normal,
            "relative_tangential_speed_m_s": relative_tangent,
            "side_of_robot": ONTOLOGY.side_from_body_point(point_body),
            "simultaneous_contact_points": 0, "candidate_action": command.astype(float).tolist(),
            "loss_of_stability": bool(loss_stability), "fall": bool(float(base_pos[2]) < float(ctx.runner.config.fall_z_threshold_m)),
            "branch_stuck": bool(branch_stuck), "route_progress_m": float(route_progress_m),
            **response,
        }
        rows.append(row)
    for row in rows:
        row["simultaneous_contact_points"] = len(rows)
        row["raw_point_digest"] = ONTOLOGY.digest(row)
    return rows


def execute_capture(ctx: Any, snapshot: Any, candidate: Any, source: dict[str, Any],
                    dense_branch: dict[str, Any], *, topology: dict[str, Any],
                    link_meta: dict[int, dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    branch_id = f"{state['state_id']}:{int(source['candidate_index']):02d}"
    route_progress = float(source["horizons"]["3"]["progress"])
    branch_stuck = any(bool(row["active_stuck"]) for row in dense_branch["ticks"])
    point_rows: list[dict[str, Any]] = []
    tick_end_contacts: list[bool] = []
    physics_counter = 0
    original_step = runner._step_policy_step

    def instrumented_step(_runner: Any, target_cmd: np.ndarray) -> None:
        nonlocal physics_counter
        observation = _runner._build_observation(target_cmd)
        joint_targets = _runner.policy.act(observation)
        _runner._apply_joint_targets(joint_targets)
        for _ in range(int(_runner._physics_steps_per_policy)):
            before = pre_physics_state(_runner)
            _runner.build.scene.step()
            physics_counter += 1
            response = post_response(_runner, before)
            rows = contact_points(
                ctx, topology, link_meta, before, response, branch_id=branch_id,
                state=state, candidate_index=int(source["candidate_index"]),
                physics_step=physics_counter, command=np.asarray(target_cmd[0], np.float32),
                branch_stuck=branch_stuck, route_progress_m=route_progress)
            point_rows.extend(rows)
            steps_per_tick = int(_runner._physics_steps_per_policy) * int(_runner._policy_steps_per_command_tick)
            if physics_counter % steps_per_tick == 0:
                tick_end_contacts.append(bool(rows))
        _runner._sim_time_ns += _runner._policy_dt_ns

    runner._step_policy_step = types.MethodType(instrumented_step, runner)
    executed_all = []
    try:
        _name, primitives = candidate
        for primitive in primitives[:3]:
            requested = V.V1.block_for(primitive)[None, ...]
            block = runner.execute_requested_block(requested)
            executed_all.append(np.asarray(block.executed, np.float32)[0])
            ctx.ticks_executed += runner._block_size
            ctx.episode_ticks += runner._block_size
            ctx.policy_steps += runner._block_size * runner._policy_steps_per_command_tick
            ctx.last_block_executed = np.asarray(block.executed, np.float32).copy()
    finally:
        runner._step_policy_step = original_step
    executed = np.concatenate(executed_all)
    expected = np.concatenate([np.asarray(block, np.float32) for block in source["post_slew"][:3]])
    action_match = bool(np.allclose(executed, expected, atol=1e-7, rtol=0))
    expected_contact = [bool(row["active_contact"]) for row in dense_branch["ticks"]]
    contact_match = tick_end_contacts == expected_contact
    position = _array(ctx.build.robot.get_pos()).astype(np.float64)
    expected_position = np.asarray(dense_branch["ticks"][-1]["position_world_xyz"], np.float64)
    endpoint_match = bool(np.allclose(position, expected_position, atol=2e-5, rtol=0))
    expected_stuck = [bool(row["active_stuck"]) for row in dense_branch["ticks"]]
    # Stuck is not recomputed at physics rate; its frozen trace is bound exactly.
    if physics_counter != 750 or len(tick_end_contacts) != 15:
        raise RuntimeError(f"{branch_id}: wrong physics/tick count {physics_counter}/{len(tick_end_contacts)}")
    return {
        "branch_id": branch_id, "point_rows": point_rows,
        "verification": {
            "action_trace_match": action_match,
            "h3_endpoint_pose_match": endpoint_match,
            "binary_contact_tick_trace_match": contact_match,
            "stuck_trace_bound": len(expected_stuck) == 15,
            "aggregate_outcome_match": bool(source["horizons"]["3"]["unsafe"] == dense_branch["ticks"][-1]["training_cumulative_unsafe"]),
            "physics_steps": physics_counter, "policy_ticks": len(tick_end_contacts),
        },
        "route_progress_m": route_progress, "branch_stuck": branch_stuck,
        "historical_contact_positive": any(expected_contact),
    }


def collect_state(state_index: int, *, force: bool = False) -> dict[str, Any]:
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    state = manifest["state_candidates"][state_index]
    state_id = str(state["state_id"])
    record_path = OUT / "states" / f"{state_id}.json"
    if record_path.is_file() and not force:
        record = json.loads(record_path.read_text())
        raw_path = Path(record["raw_points_path"])
        if record.get("status") == "PASS" and raw_path.is_file() and sha256(raw_path) == record["raw_points_sha256"]:
            print(json.dumps({"state_id": state_id, "status": "REUSED"}), flush=True)
            return record
    started = time.time()
    dense_state = json.loads((DENSE.OUT / "dense_replay" / f"{state_id}.json").read_text())
    dense_by_candidate = {int(row["candidate_index"]): row for row in dense_state["branches"]}
    source = {candidate: row for (sid, candidate), row in DENSE.source_rows().items() if sid == state_id}
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40):
        ctx.drive_one_block()
    topology = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise RuntimeError(f"{state_id}: eligibility changed: {eligible}")
    goal_record, _field = eligible
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(goal_record["goal"]), identity={
        "state_id": state_id, "scene_id": state["scene_id"], "family": state["family"]})
    raw_manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    metadata = scene_metadata(ctx, raw_manifest)
    branches = []
    all_points: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(V.V1.CANDIDATE_BANK):
        branch = execute_capture(
            ctx, snapshot, candidate, source[candidate_index], dense_by_candidate[candidate_index],
            topology=topology, link_meta=metadata, state=state)
        all_points.extend(branch.pop("point_rows"))
        branches.append(branch)
    raw_path = CACHE / "raw_contact_points" / f"{state_id}.jsonl.gz"
    atomic_jsonl_gzip(raw_path, all_points)
    mismatches = [row["branch_id"] for row in branches if not all(row["verification"].values())]
    record = {
        "schema": "lewm_contact_hazard_instrumented_state_v1", "status": "PASS" if not mismatches else "MISMATCH",
        "state_index": state_index, "state_id": state_id, "scene_id": state["scene_id"],
        "family": state["family"], "branches": branches, "raw_contact_points": len(all_points),
        "raw_points_path": str(raw_path), "raw_points_sha256": sha256(raw_path),
        "snapshot_digest": snapshot.digest, "registered_snapshot_digest": state.get("snapshot_digest"),
        "snapshot_digest_match": snapshot.digest == state.get("snapshot_digest"),
        "mismatched_branches": mismatches, "runtime_s": time.time() - started,
        "new_state_identities": 0, "new_candidate_identities": 0,
    }
    record["content_digest"] = ONTOLOGY.digest(record)
    atomic_json(record_path, record)
    print(json.dumps({"state_id": state_id, "status": record["status"],
                      "raw_points": len(all_points), "runtime_s": record["runtime_s"]}), flush=True)
    del ctx
    gc.collect()
    return record


def finalize() -> dict[str, Any]:
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    records = []
    for state in manifest["state_candidates"]:
        record = json.loads((OUT / "states" / f"{state['state_id']}.json").read_text())
        if Path(record["raw_points_path"]).is_file() and sha256(Path(record["raw_points_path"])) == record["raw_points_sha256"]:
            records.append(record)
        else:
            raise RuntimeError(f"missing or changed raw evidence for {state['state_id']}")
    index = {
        "schema": "lewm_contact_hazard_raw_contact_event_index_v1",
        "states": 48, "branches": 576, "physics_steps_per_branch": 750,
        "physics_dt_s": PHYSICS_DT_S, "event_gap_physics_steps": ONTOLOGY.EVENT_GAP_PHYSICS_STEPS,
        "state_records": records,
        "passing_states": sum(row["status"] == "PASS" for row in records),
        "mismatched_states": sum(row["status"] != "PASS" for row in records),
        "raw_contact_points": sum(int(row["raw_contact_points"]) for row in records),
        "runtime_compute_s": sum(float(row["runtime_s"]) for row in records),
        "storage_bytes": sum(Path(row["raw_points_path"]).stat().st_size for row in records),
        "bindings": {
            "state_manifest_sha256": sha256(V1 / "state_manifest.json"),
            "branch_ledger_sha256": sha256(V1 / "branch_labels.jsonl"),
            "dense_evidence_sha256": sha256(DENSE.OUT / "evidence_receipt.json"),
        },
    }
    index["content_digest"] = ONTOLOGY.digest(index)
    atomic_json(OUT / "raw_contact_event_index.json", index)
    print(json.dumps({key: index[key] for key in (
        "passing_states", "mismatched_states", "raw_contact_points",
        "runtime_compute_s", "storage_bytes", "content_digest")}, indent=2))
    return index


def collect_case_study(branch_id: str, *, force: bool = False) -> dict[str, Any]:
    """Replay one previously selected scaling-panel branch descriptively."""
    state_id, candidate_text = branch_id.rsplit(":", 1)
    candidate_index = int(candidate_text)
    allowed = {"scale-held-0-02:00", "scale-held-0-03:06"}
    if branch_id not in allowed:
        raise RuntimeError(f"case study is not one of the two frozen selected contacts: {branch_id}")
    record_path = OUT / "case_studies" / f"{state_id}__{candidate_index:02d}.json"
    if record_path.is_file() and not force:
        record = json.loads(record_path.read_text())
        raw_path = Path(record["raw_points_path"])
        if raw_path.is_file() and sha256(raw_path) == record["raw_points_sha256"]:
            print(json.dumps({"branch_id": branch_id, "status": "REUSED"}))
            return record
    scaling = ROOT / ".generated/factorised_micro_safety_data_scaling_v2"
    panel = json.loads((scaling / "panel_manifest.json").read_text())
    index = json.loads((scaling / "sensor_index.json").read_text())
    state = next(row for row in panel["states"] if row["state_id"] == state_id)
    state_record = next(row for row in index["state_records"] if row["state_id"] == state_id)
    branch_record = next(row for row in state_record["branches"] if int(row["candidate_index"]) == candidate_index)
    with np.load(state_record["shard_path"], allow_pickle=False) as arrays:
        labels = np.asarray(arrays["labels"][candidate_index], np.float32)
        poses = np.asarray(arrays["poses"][candidate_index], np.float64)
    ticks = [{
        "active_contact": bool(labels[tick, 0]), "active_stuck": bool(labels[tick, 1]),
        "training_cumulative_unsafe": bool(labels[tick, 4]),
        "position_world_xyz": [float(poses[tick, 0]), float(poses[tick, 1]), float(poses[tick, 3])],
    } for tick in range(15)]
    dense_branch = {"candidate_index": candidate_index, "ticks": ticks}
    source = {
        "candidate_index": candidate_index, "post_slew": branch_record["post_slew"],
        "horizons": {"3": {"progress": float(branch_record["p_d"]), "unsafe": bool(branch_record["unsafe"])}}
    }
    started = time.time()
    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(int(state["warmup_blocks"])):
        ctx.drive_one_block()
    topology = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise RuntimeError(f"{state_id}: eligibility changed: {eligible}")
    goal_record, _field = eligible
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(goal_record["goal"]), identity={
        "state_id": state_id, "scene_id": state["scene_id"], "family": state["family"]})
    raw_manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    metadata = scene_metadata(ctx, raw_manifest)
    branch = execute_capture(
        ctx, snapshot, V.V1.CANDIDATE_BANK[candidate_index], source, dense_branch,
        topology=topology, link_meta=metadata, state=state)
    points = branch.pop("point_rows")
    raw_path = CACHE / "case_studies" / f"{state_id}__{candidate_index:02d}.jsonl.gz"
    atomic_jsonl_gzip(raw_path, points)
    record = {
        "schema": "lewm_contact_hazard_post_hoc_descriptive_case_study_v1",
        "status": "PASS" if all(branch["verification"].values()) else "MISMATCH",
        "branch_id": branch_id, "state_id": state_id, "candidate_index": candidate_index,
        "scene_id": state["scene_id"], "family": state["family"],
        "raw_contact_points": len(points), "raw_points_path": str(raw_path),
        "raw_points_sha256": sha256(raw_path), "verification": branch["verification"],
        "runtime_s": time.time() - started, "prospective_only": True,
        "historical_label_revised": False,
    }
    record["content_digest"] = ONTOLOGY.digest(record)
    atomic_json(record_path, record)
    print(json.dumps({"branch_id": branch_id, "status": record["status"],
                      "raw_points": len(points), "runtime_s": record["runtime_s"]}))
    del ctx
    gc.collect()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--case-study-branch")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.collect_state is not None:
        collect_state(args.collect_state, force=args.force)
    elif args.collect_all:
        for state_index in range(48):
            collect_state(state_index, force=args.force)
    elif args.case_study_branch:
        collect_case_study(args.case_study_branch, force=args.force)
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
