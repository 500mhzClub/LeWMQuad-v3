#!/usr/bin/env python3
"""Materialise exact and sensor-derived per-link geometry for the repaired two-ply corpus.

This tool replays only the already registered one-tick transitions.  It does
not create state/action identities and does not open a learned safety or JEPA
checkpoint.  The frozen route and lateral locomotion policies are treated as
part of the registered simulator plant needed to reproduce articulated motion.
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
import pickle
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
sys.modules.setdefault("coverage", None)
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import articulated_swept_geometry_v1 as GEO
from lewm.safety import lightweight_one_tick_viability_model_v1 as METRICS
from scripts import collect_lightweight_one_tick_viability_model_v1 as INPUTS
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as SENSOR
from scripts import materialize_h1_articulated_swept_geometry_sufficiency_v1 as H1
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import materialize_two_ply_successor_transition_corpus_repaired_v1 as CORPUS
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as NARROW
from scripts import run_go2_oracle_branch_pilot_v1_2 as V

SOURCE_COMMIT = "10b3a190d506830e6a87e04a0f1c832b92295bd7"
EXPERIMENT = "EXPLICIT_PER_LINK_GEOMETRIC_MICRO_STATE_UPPER_BOUND_V1"
OUT = ROOT / ".generated/explicit_per_link_geometric_micro_state_upper_bound_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/explicit_per_link_geometric_micro_state_upper_bound_v1"
INDEX = OUT / "geometry_index.json"
PHYSICS_STEPS = 50
CONDITIONS = ("exact_genesis", "front_depth_per_link", "lidar_per_link", "depth_lidar_per_link")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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


def arr(value) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    value = np.asarray(value)
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value


def action_key(row: dict) -> tuple:
    return (str(row["controller"]), *(round(float(x), 7) for x in row["applied_action"]))


def unique_rows(rows: list[dict]) -> list[dict]:
    output: dict[tuple, dict] = {}
    for row in rows:
        output.setdefault(action_key(row), row)
    return list(output.values())


def deployable_action_audit(index: dict) -> dict:
    split = json.loads(CORPUS.SPLIT.read_text())
    roles = {state_id: role for role, key in (
        ("training", "development_training_state_ids"),
        ("calibration", "internal_calibration_state_ids"),
        ("heldout", "development_heldout_state_ids"),
    ) for state_id in split[key]}
    inconsistencies = []
    lateral_applied = {12: [], 13: []}
    per_state = []
    current_total = current_unique = successor_total = successor_unique = 0
    for record in index["records"]:
        current = record["current_rows"]
        current_groups: dict[tuple, list[dict]] = {}
        for row in current:
            current_groups.setdefault(action_key(row), []).append(row)
            if int(row["action_index"]) >= 12: lateral_applied[int(row["action_index"])].append(float(row["applied_action"][1]))
        for key, rows in current_groups.items():
            if len({bool(row["current_contact"]) for row in rows}) != 1:
                inconsistencies.append({"state_id": record["state_id"], "level": "current", "key": key})
        unique_current = unique_rows(current)
        current_total += len(current); current_unique += len(unique_current)
        historical_viable = any(not row["current_contact"] and int(row["successor_safe_action_count"] or 0) > 0 for row in current)
        deployable_viable = False
        unique_successor_counts = {}
        for row in unique_current:
            if row["current_contact"]:
                continue
            successor = next(item for item in record["successor_rows"] if int(item["current_action_index"]) == int(row["action_index"]))
            next_groups: dict[tuple, list[dict]] = {}
            for next_row in successor["next_actions"]:
                next_groups.setdefault(action_key(next_row), []).append(next_row)
                if int(next_row["action_index"]) >= 12: lateral_applied[int(next_row["action_index"])].append(float(next_row["applied_action"][1]))
            for key, rows in next_groups.items():
                if len({bool(item["contact"]) for item in rows}) != 1:
                    inconsistencies.append({"state_id": record["state_id"], "level": f"successor:{row['action_index']}", "key": key})
            unique_next = unique_rows(successor["next_actions"])
            successor_total += len(successor["next_actions"]); successor_unique += len(unique_next)
            safe_count = sum(not item["contact"] for item in unique_next)
            unique_successor_counts[str(row["action_index"])] = safe_count
            deployable_viable |= safe_count > 0
        per_state.append({
            "state_id": record["state_id"], "family": record["family"], "role": roles[record["state_id"]],
            "historical_action_count": 14, "unique_deployable_current_actions": len(unique_current),
            "historical_viable": historical_viable, "unique_deployable_viable": deployable_viable,
            "zero_nonzero_changed": historical_viable != deployable_viable,
            "unique_successor_safe_action_counts": unique_successor_counts,
        })
    action_contract = json.loads(CORPUS.ACTION_CONTRACT.read_text())
    left = action_contract["ordered_actions"][12]; right = action_contract["ordered_actions"][13]
    lateral_nonzero = (float(left["requested_one_tick_command"][1]) > 0 and
                       float(right["requested_one_tick_command"][1]) < 0 and
                       all(abs(value - .20) <= 1e-7 for value in lateral_applied[12]) and
                       all(abs(value + .20) <= 1e-7 for value in lateral_applied[13]))
    heldout_changes = [row["state_id"] for row in per_state if row["role"] == "heldout" and row["zero_nonzero_changed"]]
    result = {
        "schema": "explicit_per_link_geometry_deployable_action_audit_v1",
        "action_contract_sha256": sha(CORPUS.ACTION_CONTRACT),
        "lateral_requested_vy": [left["requested_one_tick_command"][1], right["requested_one_tick_command"][1]],
        "lateral_post_adapter_vy": [0.20, -0.20], "lateral_post_slew_vy": [0.20, -0.20],
        "lateral_low_level_supported_vy": [0.20, -0.20],
        "lateral_genuinely_nonzero_applied": lateral_nonzero,
        "lateral_applied_row_count": {str(index): len(values) for index, values in lateral_applied.items()},
        "lateral_applied_vy_unique": {str(index): sorted(set(values)) for index, values in lateral_applied.items()},
        "lateral_controller_checkpoint_sha256": sha(AUG.CHECKPOINT),
        "current_entries": current_total, "unique_current_entries": current_unique,
        "successor_entries_for_unique_prefixes": successor_total,
        "unique_successor_entries_for_unique_prefixes": successor_unique,
        "duplicate_outcome_inconsistencies": inconsistencies,
        "heldout_zero_nonzero_viability_changes": heldout_changes,
        "material_discrepancy": bool(heldout_changes),
        "classification": "DEPLOYABLE_MICRO_ACTION_CONTRACT_MISMATCH" if heldout_changes else "DEPLOYABLE_MICRO_ACTION_CONTRACT_ALIGNED",
        "per_state": per_state,
    }
    result["content_digest"] = METRICS.digest(result)
    atomic_json(OUT / "deployable_action_audit.json", result)
    if inconsistencies or not lateral_nonzero:
        raise RuntimeError("deployable action audit failed")
    return result


def pose7(ctx, snapshot) -> np.ndarray:
    ONE._restore_tick_boundary(ctx, snapshot)
    robot = ctx.build.robot
    position = arr(robot.get_pos()).astype(np.float64).reshape(-1, 3)[0]
    quaternion = arr(robot.get_quat()).astype(np.float64).reshape(-1, 4)[0]
    return np.concatenate((position, quaternion))


def scene_contract(state: dict) -> tuple:
    centers, halves, yaws, names = H1.scene_boxes(state)
    return centers, halves, yaws, names


def geom_clearance_steps(geom_transform: np.ndarray, contract: list[dict], boxes: tuple) -> np.ndarray:
    centers, halves, yaws, _names = boxes
    steps, geoms = geom_transform.shape[:2]
    output = np.full((steps, geoms), np.inf, np.float32)
    for gi, row in enumerate(contract):
        pos = np.asarray(geom_transform[:, gi, :3], np.float64)
        quat = np.asarray(geom_transform[:, gi, 3:], np.float64)
        rotations = np.stack([GEO.rotation(value) for value in quat])
        local, radius = GEO.primitive_points(row["kind"], np.asarray(row["data"]), GEO.CAPSULE_CENTERLINE_SAMPLES)
        world = np.einsum("tij,pj->tpi", rotations, local) + pos[:, None, :]
        nearby = np.flatnonzero(np.any(
            np.linalg.norm(centers[None, :, :2] - pos[:, None, :2], axis=2)
            <= 2.0 + np.linalg.norm(halves[:, :2], axis=1)[None], axis=0))
        for oi in nearby:
            value = np.min(GEO.box_sdf(world.reshape(-1, 3), centers[oi], halves[oi], float(yaws[oi])).reshape(steps, -1), axis=1) - radius
            output[:, gi] = np.minimum(output[:, gi], value.astype(np.float32))
    return output


def point_clearance_steps(geom_transform: np.ndarray, contract: list[dict], cloud: np.ndarray) -> np.ndarray:
    steps, geoms = geom_transform.shape[:2]
    output = np.full((steps, geoms), np.inf, np.float32)
    if not len(cloud):
        return output
    tree = cKDTree(np.asarray(cloud, np.float64)); k = min(64, len(cloud))
    for gi, row in enumerate(contract):
        pos = np.asarray(geom_transform[:, gi, :3], np.float64)
        quat = np.asarray(geom_transform[:, gi, 3:], np.float64)
        rotations = np.stack([GEO.rotation(value) for value in quat])
        _distance, indices = tree.query(pos, k=k)
        points = np.asarray(cloud, np.float64)[np.asarray(indices).reshape(steps, k)]
        local = np.einsum("tki,tij->tkj", points - pos[:, None, :], rotations)
        data = np.asarray(row["data"], np.float64)
        if row["kind"] == "sphere":
            value = np.min(np.linalg.norm(local, axis=2) - data[0], axis=1)
        elif row["kind"] == "capsule":
            z = np.clip(local[:, :, 2], -data[1] / 2, data[1] / 2)
            delta = local - np.stack((np.zeros_like(z), np.zeros_like(z), z), -1)
            value = np.min(np.linalg.norm(delta, axis=2) - data[0], axis=1)
        else:
            q = np.abs(local) - data[:3]
            sdf = np.linalg.norm(np.maximum(q, 0), axis=2) + np.minimum(np.max(q, axis=2), 0)
            value = np.min(sdf, axis=1)
        output[:, gi] = value.astype(np.float32)
    return output


def boundary_clouds(state: dict, boundary_inputs: dict[str, np.ndarray], start_pose: np.ndarray,
                    endpoint_pose: np.ndarray, boxes: tuple) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    jitter = manifest.get("camera_extrinsic_jitter", {})
    centers, halves, yaws, _names = boxes
    rendered = SENSOR.render_geometry(endpoint_pose, centers, halves, yaws, jitter)
    depth_start = H1.depth_points(boundary_inputs["depth"][-1], boundary_inputs["depth_valid"][-1], start_pose, jitter)
    depth_end = H1.depth_points(rendered[0], rendered[1], endpoint_pose, jitter)
    lidar_start = H1.lidar_points(boundary_inputs["lidar"][-1], boundary_inputs["lidar_valid"][-1], start_pose)
    lidar_end = H1.lidar_points(rendered[2], rendered[3], endpoint_pose)
    depth = np.unique(np.round(np.concatenate((depth_start, depth_end)), 3), axis=0)
    lidar = np.unique(np.round(np.concatenate((lidar_start, lidar_end)), 3), axis=0)
    return depth, lidar


def exact_query(ctx, qpos: np.ndarray, topology: dict, link_names: dict, object_names: dict) -> dict[str, np.ndarray]:
    robot = ctx.build.robot; solver = ctx.build.scene.rigid_solver
    output = {
        "contact": np.zeros(PHYSICS_STEPS, np.uint8), "count": np.zeros(PHYSICS_STEPS, np.int16),
        "robot_link": np.full(PHYSICS_STEPS, -1, np.int16), "other_link": np.full(PHYSICS_STEPS, -1, np.int16),
        "robot_geom": np.full(PHYSICS_STEPS, -1, np.int16), "other_geom": np.full(PHYSICS_STEPS, -1, np.int16),
        "penetration": np.full(PHYSICS_STEPS, np.nan, np.float32),
        "position": np.full((PHYSICS_STEPS, 3), np.nan, np.float32),
        "normal": np.full((PHYSICS_STEPS, 3), np.nan, np.float32),
    }
    for step in range(PHYSICS_STEPS):
        robot.set_qpos(qpos[step]); robot.detect_collision()
        summary = NARROW.contact_summary(solver.collider.get_contacts(as_tensor=False), topology, link_names, object_names, force_threshold=False)
        output["contact"][step] = summary["active"]; output["count"][step] = summary["count"]
        for key in ("robot_link", "other_link", "robot_geom", "other_geom", "penetration", "position", "normal"):
            output[key][step] = summary[key]
    return output


def link_structured(contract: list[dict], geom_transform: np.ndarray, approximate: np.ndarray,
                    depth: np.ndarray, lidar: np.ndarray, exact: dict, link_names: dict,
                    object_names: dict) -> dict[str, np.ndarray]:
    link_indices = sorted({int(row["link_index"]) for row in contract})
    local_names = [str(next(row["link_name"] for row in contract if int(row["link_index"]) == link)) for link in link_indices]
    global_by_name = {name: index for index, name in link_names.items()}
    fields = np.full((len(link_indices), 4), np.inf, np.float32)
    time_min = np.full((len(link_indices), 4), -1, np.int16)
    first_cross = np.full((len(link_indices), 4), -1, np.int16)
    approach = np.zeros((len(link_indices), 4), np.float32)
    approach_direction = np.full((len(link_indices), 4), np.nan, np.float32)
    sector = np.full((len(link_indices), 4), -1, np.int8)
    exact_verdict = np.zeros(len(link_indices), np.uint8)
    exact_penetration = np.zeros(len(link_indices), np.float32)
    for li, (local_index, name) in enumerate(zip(link_indices, local_names, strict=True)):
        geom_ids = [gi for gi, row in enumerate(contract) if int(row["link_index"]) == local_index]
        series = (np.min(approximate[:, geom_ids], axis=1), np.min(depth[:, geom_ids], axis=1),
                  np.min(lidar[:, geom_ids], axis=1), np.minimum(np.min(depth[:, geom_ids], axis=1), np.min(lidar[:, geom_ids], axis=1)))
        global_index = global_by_name.get(name, -1)
        exact_steps = np.flatnonzero(exact["contact"].astype(bool) & (exact["robot_link"] == global_index))
        if len(exact_steps):
            exact_verdict[li] = 1
            penetration = np.nanmax(exact["penetration"][exact_steps])
            exact_penetration[li] = 0.0 if not np.isfinite(penetration) else float(penetration)
        for condition_index, values in enumerate(series):
            step = int(np.argmin(values)); fields[li, condition_index] = float(values[step]); time_min[li, condition_index] = step
            crossings = np.flatnonzero(values <= 0.0); first_cross[li, condition_index] = -1 if not len(crossings) else int(crossings[0])
            if step > 0 and np.isfinite(values[step - 1]) and np.isfinite(values[step]):
                approach[li, condition_index] = max(0.0, float(values[step - 1] - values[step]) / 0.002)
            base_pos = geom_transform[step, geom_ids[0], :3]
            yaw = math.atan2(2 * (geom_transform[step, geom_ids[0], 3] * geom_transform[step, geom_ids[0], 6]
                                  + geom_transform[step, geom_ids[0], 4] * geom_transform[step, geom_ids[0], 5]),
                             1 - 2 * (geom_transform[step, geom_ids[0], 5] ** 2 + geom_transform[step, geom_ids[0], 6] ** 2))
            # Sector is defined by the responsible surface direction when a
            # contact manifold is available; otherwise it remains unknown.
            exact_here = np.flatnonzero(exact["contact"].astype(bool) & (exact["robot_link"] == global_index))
            if len(exact_here):
                contact_step = int(exact_here[0]); point = exact["position"][contact_step]
                if np.isfinite(point).all():
                    angle = (math.atan2(float(point[1] - base_pos[1]), float(point[0] - base_pos[0])) - yaw + math.pi) % (2 * math.pi) - math.pi
                    approach_direction[li, condition_index] = float(angle)
                    sector[li, condition_index] = 0 if abs(angle) <= math.pi / 4 else 1 if angle > math.pi / 4 and angle < 3 * math.pi / 4 else 3 if angle < -math.pi / 4 and angle > -3 * math.pi / 4 else 2
    exact_clearance = fields[:, 0].copy()
    exact_clearance[exact_verdict.astype(bool)] = -exact_penetration[exact_verdict.astype(bool)]
    fields[:, 0] = exact_clearance
    return {
        "link_local_index": np.asarray(link_indices, np.int16), "link_name": np.asarray(local_names),
        "minimum_signed_clearance": fields, "time_to_minimum_step": time_min,
        "first_zero_crossing_step": first_cross, "relative_normal_approach_speed": approach,
        "approach_direction_rad": approach_direction, "obstacle_sector": sector, "exact_contact_verdict": exact_verdict,
        "exact_penetration": exact_penetration,
    }


def execute_transition(ctx, snapshot, route_policy, lateral_policy, action_index: int, expected: dict,
                       state: dict, boundary_inputs: dict[str, np.ndarray], contract: list[dict], boxes: tuple,
                       topology: dict, link_names: dict, object_names: dict, identity: str) -> tuple[dict, object]:
    controller = "route" if action_index < 12 else "lateral"
    ctx.policy = route_policy if controller == "route" else lateral_policy
    ONE._restore_tick_boundary(ctx, snapshot); runner = ctx.runner; robot = runner.build.robot
    start_pose = pose7(ctx, snapshot)
    if controller == "route":
        requested = V.V1.block_for(V.V1.CANDIDATE_BANK[action_index][1][0])[None, ...]
        target = np.asarray(runner._clip_block(requested).executed[:, 0], np.float32)
    else:
        target = np.asarray(AUG.LATERAL_ACTIONS[action_index - 12][2], np.float32)[None, :]
    if not np.array_equal(target[0], np.asarray(expected["applied_action"], np.float32)):
        raise RuntimeError(f"{identity}: applied action mismatch {target[0]} != {expected['applied_action']}")
    link_count = len(robot.links); geom_count = len(contract)
    qpos = np.empty((PHYSICS_STEPS, 19), np.float32)
    link_transform = np.empty((PHYSICS_STEPS, link_count, 7), np.float32)
    geom_transform = np.empty((PHYSICS_STEPS, geom_count, 7), np.float32)
    native_contact = np.zeros(PHYSICS_STEPS, np.uint8); native_link = np.full(PHYSICS_STEPS, -1, np.int16)
    native_other = np.full(PHYSICS_STEPS, -1, np.int16); native_penetration = np.full(PHYSICS_STEPS, np.nan, np.float32)
    timestamps = np.empty(PHYSICS_STEPS, np.float64); counter = 0
    start_sim_time_s = float(runner._sim_time_ns) * 1e-9
    for _ in range(int(runner._policy_steps_per_command_tick)):
        observation = runner._build_observation(target); joint_targets = runner.policy.act(observation); runner._apply_joint_targets(joint_targets)
        for _ in range(int(runner._physics_steps_per_policy)):
            runner.build.scene.step(); step = counter; counter += 1
            qpos[step] = arr(robot.get_qpos()).astype(np.float32)
            lp = arr(robot.get_links_pos()).astype(np.float64); lq = arr(robot.get_links_quat()).astype(np.float64)
            link_transform[step, :, :3] = lp; link_transform[step, :, 3:] = lq
            primitives = H1.instantiate(contract, lp, lq)
            for gi, primitive in enumerate(primitives):
                geom_transform[step, gi, :3] = primitive["pos"]; geom_transform[step, gi, 3:] = primitive["quat"]
            summary = NARROW.contact_summary(robot.get_contacts(), topology, link_names, object_names, force_threshold=True)
            native_contact[step] = summary["active"]; native_link[step] = summary["robot_link"]
            native_other[step] = summary["other_link"]; native_penetration[step] = summary["penetration"]
            timestamps[step] = start_sim_time_s + (step + 1) * 0.002
        runner._sim_time_ns += runner._policy_dt_ns
    if counter != PHYSICS_STEPS:
        raise RuntimeError(f"{identity}: physics step count {counter}")
    ONE._advance_tick_counters(ctx, target)
    endpoint_pose = np.concatenate((arr(robot.get_pos()).reshape(-1, 3)[0], arr(robot.get_quat()).reshape(-1, 4)[0])).astype(np.float64)
    successor = ONE._capture_tick_boundary(ctx, goal=snapshot.goal, identity={**snapshot.identity, "geometry_transition": identity})
    exact = exact_query(ctx, qpos, topology, link_names, object_names)
    approximate = geom_clearance_steps(geom_transform, contract, boxes)
    depth_cloud, lidar_cloud = boundary_clouds(state, boundary_inputs, start_pose, endpoint_pose, boxes)
    depth = point_clearance_steps(geom_transform, contract, depth_cloud)
    lidar = point_clearance_steps(geom_transform, contract, lidar_cloud)
    structured = link_structured(contract, geom_transform, approximate, depth, lidar, exact, link_names, object_names)
    native_label = bool(native_contact.any()); exact_label = bool(exact["contact"].any())
    frozen_label = bool(expected.get("current_contact", expected.get("contact")))
    arrays = {
        "frozen_contact_label": np.asarray(frozen_label, np.uint8),
        "physics_timestamp_s": timestamps, "qpos": qpos, "link_transform": link_transform,
        "geom_transform": geom_transform, "native_contact": native_contact, "native_robot_link": native_link,
        "native_other_link": native_other, "native_penetration": native_penetration,
        "exact_contact": exact["contact"], "exact_contact_count": exact["count"],
        "exact_robot_link": exact["robot_link"], "exact_other_link": exact["other_link"],
        "exact_robot_geom": exact["robot_geom"], "exact_other_geom": exact["other_geom"],
        "exact_penetration": exact["penetration"], "exact_manifold_position": exact["position"],
        "exact_manifold_normal": exact["normal"], "approximate_scene_clearance": approximate,
        "depth_clearance": depth, "lidar_clearance": lidar, "fused_clearance": np.minimum(depth, lidar),
        **{f"structured_{key}": value for key, value in structured.items()},
    }
    first_native = None if not native_label else int(np.flatnonzero(native_contact)[0])
    first_exact = None if not exact_label else int(np.flatnonzero(exact["contact"])[0])
    metadata = {
        "identity": identity, "action_index": action_index, "controller": controller,
        "requested_action": expected["requested_action"], "applied_action": expected["applied_action"],
        "frozen_contact": frozen_label, "native_contact": native_label, "exact_contact": exact_label,
        "native_replay_matches_frozen": native_label == frozen_label,
        "native_first_contact_step": first_native, "exact_first_contact_step": first_exact,
        "depth_points": len(depth_cloud), "lidar_points": len(lidar_cloud),
        "successor_snapshot_digest": successor.digest,
    }
    return {"metadata": metadata, "arrays": arrays}, successor


def load_state_context(state: dict):
    ctx, snapshots, reconstruction = MULTI.historical_snapshots(state)
    if state["source_kind"] == "compatible_historical_root":
        del ctx; gc.collect(); ctx, current = ONE._build_current(state); snapshots[0] = current
    return ctx, snapshots, reconstruction


def collect_state(index: int) -> dict:
    corpus = json.loads(CORPUS.INDEX.read_text()); record = corpus["records"][index]
    state = CORPUS.states()[index]; state_id = state["state_id"]
    if record["state_id"] != state_id:
        raise RuntimeError("state ordering mismatch")
    receipt_path = OUT / "states" / f"{state_id}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text()); shard = Path(receipt["shard_path"])
        if receipt.get("status") == "PASS" and receipt.get("geometry_contract_version") == 4 and shard.is_file() and sha(shard) == receipt["shard_sha256"]:
            print(json.dumps({"state_id": state_id, "status": "REUSED"}), flush=True); return receipt
    started = time.time(); ctx, _history, reconstruction = load_state_context(state)
    with gzip.open(record["snapshot_path"], "rb") as stream:
        snapshots = pickle.load(stream)
    current_snapshot = snapshots["current"]; successor_snapshots = {int(key): value for key, value in snapshots["successors"].items()}
    route_policy = ctx.policy; lateral_policy = AUG.lateral_policy(); topology, link_names, object_names = AUG.topology(ctx)
    ONE._restore_tick_boundary(ctx, current_snapshot); contract = H1.geom_contract(ctx.build.robot); boxes = scene_contract(state)
    with np.load(record["tensor_path"], allow_pickle=False) as loaded:
        tensors = {key: np.asarray(loaded[key]) for key in loaded.files}
    current_inputs = {"depth": tensors["current_depth"], "depth_valid": tensors["current_depth_valid"],
                      "lidar": tensors["current_lidar"], "lidar_valid": tensors["current_lidar_valid"]}
    transition_arrays: dict[str, list[np.ndarray]] = {}; transition_rows = []; branches = 0
    successor_offsets = {int(action): offset for offset, action in enumerate(tensors["successor_current_action"])}
    def append(level: str, current_action: int, action: int, outcome: dict):
        nonlocal branches
        transition_index = len(transition_rows); metadata = outcome["metadata"]
        transition_rows.append({"transition_index": transition_index, "level": level,
                                "current_action_index": current_action, **metadata})
        for key, value in outcome["arrays"].items(): transition_arrays.setdefault(key, []).append(value)
        branches += 1
    for row in record["current_rows"]:
        action = int(row["action_index"])
        outcome, _successor = execute_transition(ctx, current_snapshot, route_policy, lateral_policy, action, row, state,
                                                  current_inputs, contract, boxes, topology, link_names, object_names,
                                                  f"{state_id}:current:{action:02d}")
        append("current", -1, action, outcome)
    for successor in record["successor_rows"]:
        current_action = int(successor["current_action_index"]); snapshot = successor_snapshots[current_action]
        offset = successor_offsets[current_action]
        boundary_inputs = {"depth": tensors["successor_depth"][offset], "depth_valid": tensors["successor_depth_valid"][offset],
                           "lidar": tensors["successor_lidar"][offset], "lidar_valid": tensors["successor_lidar_valid"][offset]}
        for next_row in successor["next_actions"]:
            action = int(next_row["action_index"])
            outcome, _ = execute_transition(ctx, snapshot, route_policy, lateral_policy, action, next_row, state,
                                             boundary_inputs, contract, boxes, topology, link_names, object_names,
                                             f"{state_id}:successor:{current_action:02d}:next:{action:02d}")
            append("successor", current_action, action, outcome)
    arrays = {key: np.stack(values) for key, values in transition_arrays.items()}
    arrays.update({"transition_level": np.asarray([row["level"] for row in transition_rows]),
                   "transition_current_action": np.asarray([row["current_action_index"] for row in transition_rows], np.int16),
                   "transition_action": np.asarray([row["action_index"] for row in transition_rows], np.int16)})
    shard = CACHE / "states" / f"{state_id}.npz"; atomic_npz(shard, **arrays)
    exact_matches = sum(bool(row["frozen_contact"]) == bool(row["exact_contact"]) for row in transition_rows)
    native_replay_matches = sum(bool(row["frozen_contact"]) == bool(row["native_contact"]) for row in transition_rows)
    receipt = {
        "schema": "explicit_per_link_geometric_micro_state_v1_state", "status": "PASS",
        "geometry_contract_version": 4,
        "state_index": index, "state_id": state_id, "scene_id": record["scene_id"], "family": record["family"],
        "original_role": record["original_role"], "source_kind": record["source_kind"],
        "transitions": len(transition_rows), "physics_steps_per_transition": PHYSICS_STEPS,
        "transition_rows": transition_rows, "exact_branch_matches": exact_matches,
        "native_replay_label_matches": native_replay_matches,
        "robot_links": len(ctx.build.robot.links), "relevant_collision_shapes": len(contract),
        "collision_shape_contract": contract,
        "scene_collision_primitives": [{"object": str(name), "center": boxes[0][i].tolist(), "half_extent": boxes[1][i].tolist(), "yaw_rad": float(boxes[2][i])} for i, name in enumerate(boxes[3])],
        "link_names": {str(key): value for key, value in link_names.items()},
        "object_names": {str(key): value for key, value in object_names.items()},
        "predecessor_reconstruction": reconstruction,
        "shard_path": str(shard), "shard_sha256": sha(shard), "storage_bytes": shard.stat().st_size,
        "runtime_s": time.time() - started,
    }
    receipt["content_digest"] = METRICS.digest({key: value for key, value in receipt.items() if key != "runtime_s"})
    atomic_json(receipt_path, receipt); print(json.dumps({"state_id": state_id, "transitions": branches,
        "exact_matches": exact_matches, "runtime_s": receipt["runtime_s"], "storage_bytes": receipt["storage_bytes"]}), flush=True)
    del ctx; gc.collect(); return receipt


def fixture() -> dict:
    q = np.asarray([1., 0., 0., 0.]); contract = [{"link_index": 0, "link_name": "base", "kind": "sphere", "data": [.05]}]
    geom = np.zeros((50, 1, 7), np.float32); geom[:, 0, 0] = np.linspace(.5, .05, 50); geom[:, 0, 2] = .5; geom[:, 0, 3:] = q
    boxes = (np.asarray([[0., 0., .5]]), np.asarray([[.1, .5, .5]]), np.asarray([0.]), np.asarray(["wall"]))
    clearance = geom_clearance_steps(geom, contract, boxes)
    cloud = np.asarray([[.1, 0., .5], [.1, .05, .5]])
    point = point_clearance_steps(geom, contract, cloud)
    tests = {"scene_crossing": bool(clearance.min() < 0), "sensor_crossing": bool(point.min() < 0),
             "mirrored_action_identity_distinct": action_key({"controller": "lateral", "applied_action": [0, .2, 0]}) != action_key({"controller": "lateral", "applied_action": [0, -.2, 0]}),
             "duplicate_reduction": len(unique_rows([{"controller": "route", "applied_action": [0, 0, 0]}, {"controller": "route", "applied_action": [0, 0, 0]}])) == 1}
    value = {"schema": "explicit_per_link_geometric_micro_state_fixture_v1", "tests": tests, "pass": all(tests.values())}
    value["content_digest"] = METRICS.digest(value); atomic_json(OUT / "fixture.json", value)
    if not value["pass"]: raise RuntimeError(value)
    return value


def finalize() -> dict:
    corpus = json.loads(CORPUS.INDEX.read_text()); records = []
    for source in corpus["records"]:
        receipt = json.loads((OUT / "states" / f"{source['state_id']}.json").read_text())
        if receipt["status"] != "PASS" or sha(Path(receipt["shard_path"])) != receipt["shard_sha256"]:
            raise RuntimeError(f"bad geometry state {source['state_id']}")
        records.append(receipt)
    audit = deployable_action_audit(corpus); wall = json.loads((OUT / "collection_runtime.json").read_text())
    value = {
        "schema": "explicit_per_link_geometric_micro_state_upper_bound_v1_index", "experiment": EXPERIMENT,
        "source_commit": SOURCE_COMMIT, "status": "PASS", "states": len(records),
        "transitions": sum(row["transitions"] for row in records),
        "physics_steps": sum(row["transitions"] for row in records) * PHYSICS_STEPS,
        "exact_branch_matches": sum(row["exact_branch_matches"] for row in records),
        "native_replay_label_matches": sum(row["native_replay_label_matches"] for row in records),
        "materialized_current_transitions": sum(sum(item["level"] == "current" for item in row["transition_rows"]) for row in records),
        "materialized_successor_transitions": sum(sum(item["level"] == "successor" for item in row["transition_rows"]) for row in records),
        "records": records, "deployable_action_audit": audit,
        "contract": {
            "genesis_version": "0.3.14", "narrowphase": "native broadphase plus MPR/GJK fallback and manifold generation",
            "exact_query": "history-free robot.detect_collision at every persisted post-step qpos; solver not advanced",
            "pair_filter": "robot/environment only; self and permitted calf/foot-ground support excluded",
            "collision_margin_m": 0.0, "physics_step_s": 0.002, "command_tick_s": 0.1,
            "positive_exact_separation": "not exposed; analytical primitive separation is persisted only as a positive-clearance fallback and is not the exact binary decision",
            "sensor_geometry": "current plus actual next-boundary ideal point clouds; no global scene map used by the point-to-articulated-body query",
            "sensor_point_query_neighbours": 64, "collision_shapes": 27,
        },
        "bindings": {
            "corpus_logical_digest": corpus["corpus_logical_digest"], "corpus_index_sha256": sha(CORPUS.INDEX),
            "action_contract_sha256": sha(CORPUS.ACTION_CONTRACT),
            "predecessor_row_ledger_sha256": "63726e042e793d06784236b9dcc37c3844c798b8f526d03e4f19517186d5cc94",
        },
        "runtime": {"parallel_wall_s": wall["wall_runtime_s"], "workers": wall["workers"],
                    "per_state_runtime_sum_s": sum(row["runtime_s"] for row in records)},
        "storage_bytes": sum(row["storage_bytes"] for row in records),
    }
    if value["transitions"] != 29470 or value["physics_steps"] != 1_473_500:
        raise RuntimeError("geometry materialization cardinality mismatch")
    value["content_digest"] = METRICS.digest({key: item for key, item in value.items() if key != "records"})
    atomic_json(INDEX, value)
    print(json.dumps({key: value[key] for key in ("states", "transitions", "physics_steps", "exact_branch_matches", "runtime", "storage_bytes", "content_digest")}, indent=2))
    return value


def collect_all(workers: int) -> None:
    corpus = json.loads(CORPUS.INDEX.read_text()); deployable_action_audit(corpus); fixture()
    logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True); started = time.time()
    for start in range(0, len(corpus["records"]), workers):
        jobs = []
        for index in range(start, min(start + workers, len(corpus["records"]))):
            state_id = corpus["records"][index]["state_id"]; stream = (logs / f"state_{index:03d}_{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((state_id, process, stream))
        for state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code: raise RuntimeError(f"geometry state failed {state_id}; see {logs}")
    atomic_json(OUT / "collection_runtime.json", {"states": len(corpus["records"]), "workers": workers, "wall_runtime_s": time.time() - started})


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true"); group.add_argument("--fixture", action="store_true")
    group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true"); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    if args.audit: deployable_action_audit(json.loads(CORPUS.INDEX.read_text()))
    elif args.fixture: fixture()
    elif args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all: collect_all(args.workers)
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
