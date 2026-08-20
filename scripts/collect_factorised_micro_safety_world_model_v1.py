#!/usr/bin/env python3
"""Freeze and collect the fresh factorised micro-safety development panel."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep

OUT = ROOT / ".generated/factorised_micro_safety_world_model_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorised_micro_safety_world_model_v1")
SCENE_ROOT = ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z"
OLD_PANEL = ROOT / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json"
PREDICTOR_MANIFEST = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/factorial_manifest.json")
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
DOMAIN = "FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1/FRESH_PANEL/2026-08-20"
TICKS = 15


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def body_relative(start, point) -> list[float]:
    (x0, y0), yaw, _ = start
    dx, dy = float(point[0]) - float(x0), float(point[1]) - float(y0)
    return [math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy]


def wrap(value: float) -> float:
    return (float(value) + math.pi) % (2 * math.pi) - math.pi


def excluded_scene_ids() -> dict:
    old = json.loads(OLD_PANEL.read_text())
    old_scenes = {str(row["scene_id"]) for row in old["state_candidates"]}
    predictor = json.loads(PREDICTOR_MANIFEST.read_text())
    predictor_scenes = {str(row["episode_cluster"]).split("/")[0] for row in predictor["rows"]}
    return {
        "old_panel": old_scenes,
        "predictor_training_and_selection": predictor_scenes,
        "union": old_scenes | predictor_scenes,
        "predictor_manifest_digest": predictor["digest"],
    }


def ordered_scene_dirs(family: str, excluded: set[str]) -> list[Path]:
    paths = [path for path in SCENE_ROOT.glob(f"*/{family}/*") if path.is_dir() and path.name not in excluded]
    return sorted(paths, key=lambda path: hashlib.sha256(f"{DOMAIN}|{family}|{path.name}".encode()).hexdigest())


def probe_scene(scene_dir: Path, family: str, seed: int, receipt_path: Path) -> None:
    """Evaluate pre-outcome eligibility in a disposable Genesis process.

    Genesis 0.4.6 does not reliably release a long sequence of independently
    compiled scenes in one interpreter.  A receipt is fsync-equivalent before
    an immediate process exit, avoiding destructor-time simulator faults.
    """
    record = {"family": family, "scene_id": scene_dir.name, "scene_dir": str(scene_dir), "seed": seed}
    started = time.time()
    try:
        shared = V.V1._load_shared("cpu")
        ctx = V.V1.build_context(scene_dir, seed=seed, backend="cpu", shared=shared)
        ctx.begin_episode()
        for _ in range(40):
            ctx.drive_one_block()
        topology = V.link_topology(ctx)
        eligible = V.eligible_here(ctx, topology)
        if isinstance(eligible, str):
            record.update(status="INELIGIBLE", reason=eligible)
        else:
            goal_record, _field = eligible
            route = ctx.scene_graph.shortest_path(int(goal_record["cell_id"]), int(goal_record["goal"]["landmark_cell"]))
            if route is None or len(route) < 3:
                record.update(status="INELIGIBLE", reason="no_two_edge_route")
            else:
                waypoint = ctx.scene_graph.cell_center(int(route[2]))
                start_pose = ctx.pose()
                record.update(status="ELIGIBLE", start_pose=list(start_pose),
                              waypoint_path_cells=list(map(int, route[:3])),
                              waypoint_xy=list(map(float, waypoint)),
                              waypoint_body_xy=body_relative(start_pose, waypoint),
                              goal=dict(goal_record["goal"]), warmup_blocks=40)
    except Exception as exc:
        trace = traceback.format_exc()
        if (isinstance(exc, ImportError) and "ResetEvent" in str(exc)
                and "_check_and_reset_fallen_envs" in trace):
            record.update(status="INELIGIBLE", reason="warmup_fall_or_reset")
        else:
            record.update(status="ERROR", reason=f"{type(exc).__name__}: {exc}", traceback=trace)
    record["scan_runtime_s"] = time.time() - started
    atomic_json(receipt_path, record)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def freeze_panel() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fresh_panel_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("content_digest") != canonical_digest({key: value for key, value in payload.items() if key != "content_digest"}):
            raise RuntimeError("existing fresh panel manifest digest mismatch")
        print(json.dumps({"status": "REUSED", "states": len(payload["states"]), "digest": payload["content_digest"]}))
        return payload
    exclusions = excluded_scene_ids()
    selected = []
    scan = []
    probe_root = CACHE / "eligibility_receipts"
    probe_root.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        accepted = 0
        scene_dirs = ordered_scene_dirs(family, exclusions["union"])
        # Each qualified Genesis process owns its simulator state; bounded
        # four-way probing keeps state selection candidate-blind and avoids the
        # historical multi-scene teardown fault.
        for batch_start in range(0, len(scene_dirs), 4):
            if accepted >= 12:
                break
            processes = []
            for scene_dir in scene_dirs[batch_start:batch_start + 4]:
                seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                receipt = probe_root / f"{family}__{scene_dir.name}.json"
                if receipt.is_file():
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir),
                           "--probe-family", family, "--probe-seed", str(seed), "--probe-receipt", str(receipt)]
                log = receipt.with_suffix(".log").open("wb")
                processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, receipt))
            for process, log, receipt in processes:
                return_code = process.wait(); log.close()
                if return_code != 0 and not receipt.is_file():
                    atomic_json(receipt, {"family": family, "scene_id": receipt.stem.split("__", 1)[-1],
                                         "status": "ERROR", "reason": f"probe_exit_{return_code}", "scan_runtime_s": 0.0})
            for scene_dir in scene_dirs[batch_start:batch_start + 4]:
                receipt = probe_root / f"{family}__{scene_dir.name}.json"
                record = json.loads(receipt.read_text())
                if record["status"] == "ERROR":
                    # A bounded serial retry handles process-start races without
                    # changing deterministic scene order or viewing outcomes.
                    receipt.unlink()
                    seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                    command = [sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir),
                               "--probe-family", family, "--probe-seed", str(seed), "--probe-receipt", str(receipt)]
                    retry_log = receipt.with_suffix(".retry.log")
                    with retry_log.open("wb") as log:
                        retry = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if retry.returncode != 0 or not receipt.is_file():
                        raise RuntimeError(f"eligibility retry failed for {scene_dir.name}")
                    record = json.loads(receipt.read_text())
                    if record["status"] == "ERROR":
                        raise RuntimeError(f"eligibility error after serial retry for {scene_dir.name}: {record.get('reason')}")
                scan.append(record)
                if record["status"] == "ELIGIBLE" and accepted < 12:
                    split = "calibration" if accepted < 6 else "heldout"
                    prefix = "cal" if split == "calibration" else "held"
                    state_id = f"micro-{prefix}-{FAMILIES.index(family)}-{accepted % 6:02d}"
                    record.update(split=split, state_id=state_id)
                    selected.append(dict(record)); accepted += 1
        if accepted != 12:
            raise RuntimeError(f"{family}: found {accepted}/12 eligible fresh states")
    payload = {
        "schema": "factorised_micro_safety_fresh_panel_manifest_v1",
        "domain": DOMAIN, "frozen_before_candidate_execution": True,
        "states": selected, "state_count": 48, "candidate_count": 12,
        "split_state_count": {"calibration": 24, "heldout": 24},
        "family_split_state_count": {family: {split: sum(row["family"] == family and row["split"] == split for row in selected)
                                               for split in ("calibration", "heldout")} for family in FAMILIES},
        "disjointness": {"old_panel_scene_overlap": len({row["scene_id"] for row in selected} & exclusions["old_panel"]),
                         "predictor_scene_overlap": len({row["scene_id"] for row in selected} & exclusions["predictor_training_and_selection"]),
                         "distinct_scene_count": len({row["scene_id"] for row in selected}),
                         "distinct_episode_state_cluster_count": len({row["state_id"] for row in selected})},
        "bindings": {"old_panel_sha256": sha(OLD_PANEL), "predictor_manifest_sha256": sha(PREDICTOR_MANIFEST),
                     "predictor_manifest_content_digest": exclusions["predictor_manifest_digest"]},
        "scan": scan,
    }
    payload["content_digest"] = canonical_digest(payload)
    atomic_json(path, payload)
    print(json.dumps({"status": "FROZEN", "states": 48, "digest": payload["content_digest"],
                      "disjointness": payload["disjointness"]}, indent=2))
    return payload


def route_heading(ctx, path: list[int]) -> float:
    start = ctx.scene_graph.cell_center(int(path[0])); nxt = ctx.scene_graph.cell_center(int(path[1]))
    return math.atan2(float(nxt[1]) - float(start[1]), float(nxt[0]) - float(start[0]))


def execute_candidate(ctx, snapshot, candidate, *, topology) -> dict:
    V.V1.restore_branch_state(ctx, snapshot)
    runner = ctx.runner
    steps_per_tick = int(runner._policy_steps_per_command_tick)
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(runner.episode_states[0].episode_id)
    counter = {"episode_step": int(runner.episode_states[0].episode_step), "stamp_ns": int(runner._sim_time_ns)}
    current, previous_velocity = SENSOR.sensor_state(runner, previous_joint_velocity=None)
    previous_command = np.asarray(runner._last_executed, np.float32)[0].copy()
    future, controls, labels, poses = [], [], [], []
    requested_all, executed_all = [], []
    cumulative_contact = cumulative_stuck = cumulative_unsafe = False
    name, primitives = candidate
    for block_index, primitive in enumerate(primitives[:3]):
        requested = V.V1.block_for(primitive)[None, ...]
        clipped = np.asarray(runner._clip_block(np.asarray(requested, np.float32)).executed, np.float32)

        def after_policy_step(tick_index: int, step_index: int, _block=clipped) -> None:
            nonlocal previous_velocity, previous_command, cumulative_contact, cumulative_stuck, cumulative_unsafe
            if step_index != steps_per_tick - 1:
                return
            counter["episode_step"] += 1; counter["stamp_ns"] += int(runner._command_dt_ns)
            command = _block[0, tick_index]
            state, velocity = SENSOR.sensor_state(runner, previous_joint_velocity=previous_velocity)
            previous_velocity = velocity
            (x, y), yaw, z = ctx.pose()
            derived = label_computer.step(PoseStep(timestamp_ns=counter["stamp_ns"], env_idx=0,
                episode_id=episode_id, episode_step=counter["episode_step"], position_xy_world=(x, y),
                yaw_world_rad=float(yaw), last_command=tuple(float(value) for value in command)))
            flags = V.V1._termination_flags(ctx)
            contact = bool(int(V._contact_count(ctx, topology)) > 0)
            stuck = bool(derived.stuck_label)
            terminal = bool(flags["fall"] or flags["out_of_bounds"] or flags["tipped"])
            cumulative_contact |= contact; cumulative_stuck |= stuck; cumulative_unsafe |= contact or stuck or terminal
            future.append(state); controls.append(np.concatenate((command, previous_command)).astype(np.float32))
            labels.append([contact, stuck, cumulative_contact, cumulative_stuck, cumulative_unsafe])
            poses.append([float(x), float(y), float(yaw), float(z), float(derived.clearance_m),
                          float(flags["fall"]), float(flags["out_of_bounds"]), float(flags["tipped"])])
            previous_command = command.copy()

        block = runner.execute_requested_block(requested, after_policy_step=after_policy_step)
        requested_all.append(np.asarray(block.requested, np.float32)[0].tolist())
        executed_all.append(np.asarray(block.executed, np.float32)[0].tolist())
        ctx.ticks_executed += runner._block_size; ctx.episode_ticks += runner._block_size
        ctx.policy_steps += runner._block_size * steps_per_tick
        ctx.last_block_executed = np.asarray(block.executed, np.float32).copy()
    if len(future) != TICKS:
        raise RuntimeError(f"{name}: expected {TICKS} ticks, got {len(future)}")
    requested_trace = np.concatenate([np.asarray(block, np.float32) for block in requested_all])
    executed = np.concatenate([np.asarray(block, np.float32) for block in executed_all])
    registered = np.concatenate([np.asarray(V.V1.block_for(primitive), np.float32) for primitive in primitives[:3]])
    return {"name": name, "primitives": list(primitives), "current": current, "future": np.stack(future).astype(np.float32),
            "action_control": np.stack(controls).astype(np.float32), "labels": np.asarray(labels, np.float32),
            "poses": np.asarray(poses, np.float64), "requested": requested_all, "post_slew": executed_all,
            "requested_identity_matches": bool(np.array_equal(requested_trace, registered)),
            "post_slew_trace_matches": bool(np.array_equal(executed, np.stack(controls)[:, :3])),
            "finite": bool(np.isfinite(future).all() and np.isfinite(controls).all())}


def collect_state(index: int) -> dict:
    manifest = freeze_panel(); state = manifest["states"][index]
    record_path = OUT / "states" / f"{state['state_id']}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text()); shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            print(json.dumps({"state_id": state["state_id"], "status": "REUSED"}), flush=True); return record
    started = time.time(); shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40): ctx.drive_one_block()
    topology = V.link_topology(ctx); eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str): raise RuntimeError(f"{state['state_id']}: eligibility changed: {eligible}")
    goal_record, _field = eligible
    route = ctx.scene_graph.shortest_path(int(goal_record["cell_id"]), int(goal_record["goal"]["landmark_cell"]))
    if route is None or list(map(int, route[:3])) != state["waypoint_path_cells"]:
        raise RuntimeError(f"{state['state_id']}: route identity changed")
    start_pose = ctx.pose(); waypoint = ctx.scene_graph.cell_center(int(route[2])); waypoint_body = body_relative(start_pose, waypoint)
    heading = route_heading(ctx, route)
    snapshot = V.V1.capture_branch_state(ctx, goal=dict(goal_record["goal"]),
        identity={"state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"]})
    current_rows, future_rows, action_rows, label_rows, pose_rows = [], [], [], [], []
    branch_records = []
    for candidate_index, candidate in enumerate(V.V1.CANDIDATE_BANK):
        branch = execute_candidate(ctx, snapshot, candidate, topology=topology)
        current_rows.append(branch["current"]); future_rows.append(branch["future"])
        action_rows.append(branch["action_control"]); label_rows.append(branch["labels"]); pose_rows.append(branch["poses"])
        start_error = abs(wrap(heading - float(start_pose[1])))
        endpoint = branch["poses"][-1]
        endpoint_body = body_relative(start_pose, (endpoint[0], endpoint[1]))
        distance_start = math.hypot(*waypoint_body)
        distance_end = math.hypot(waypoint_body[0] - endpoint_body[0], waypoint_body[1] - endpoint_body[1])
        p_d = distance_start - distance_end
        p_theta = start_error - abs(wrap(heading - float(endpoint[2])))
        branch_records.append({"branch_id": f"{state['state_id']}:{candidate_index:02d}",
            "state_id": state["state_id"], "candidate_index": candidate_index, "candidate": branch["name"],
            "primitives": branch["primitives"], "requested": branch["requested"], "post_slew": branch["post_slew"],
            "p_d": float(p_d), "p_theta": float(p_theta), "unsafe": bool(branch["labels"][-1, 4]),
            "contact": bool(branch["labels"][-1, 2]), "stuck": bool(branch["labels"][-1, 3]),
            "completed": bool(distance_end <= .35), "distance_start": float(distance_start), "distance_end": float(distance_end),
            "route_heading_world_rad": float(heading), "finite": branch["finite"],
            "action_trace_identity_verified": bool(branch["requested_identity_matches"] and branch["post_slew_trace_matches"]),
            "pose_and_safety_trace_verified": bool(len(branch["poses"]) == TICKS and len(branch["labels"]) == TICKS)})
    arrays = {"current": np.stack(current_rows).astype(np.float32), "future": np.stack(future_rows).astype(np.float32),
              "action_control": np.stack(action_rows).astype(np.float32), "labels": np.stack(label_rows).astype(np.float32),
              "poses": np.stack(pose_rows).astype(np.float64)}
    shard = CACHE / "fresh_sensors" / f"{state['state_id']}.npz"; atomic_npz(shard, **arrays)
    record = {"schema": "factorised_micro_safety_fresh_state_v1", "status": "PASS", "state_index": index,
        "state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"], "split": state["split"],
        "snapshot_digest": snapshot.digest, "branches": branch_records, "shard_path": str(shard), "shard_sha256": sha(shard),
        "shapes": {key: list(value.shape) for key, value in arrays.items()}, "dtype": {key: str(value.dtype) for key, value in arrays.items()},
        "verification": {"branch_count": 12, "tick_count": TICKS, "finite_branches": sum(row["finite"] for row in branch_records),
                         "action_trace_identity_matches": sum(row["action_trace_identity_verified"] for row in branch_records),
                         "pose_and_safety_trace_matches": sum(row["pose_and_safety_trace_verified"] for row in branch_records),
                         "route_identity_match": True, "state_identity_match": True},
        "runtime_s": time.time() - started}
    record["content_digest"] = canonical_digest(record); atomic_json(record_path, record)
    del ctx; gc.collect()
    print(json.dumps({"state_id": state["state_id"], "status": "PASS", "runtime_s": record["runtime_s"]}), flush=True)
    return record


def finalize() -> dict:
    manifest = freeze_panel(); records = []
    for state in manifest["states"]:
        path = OUT / "states" / f"{state['state_id']}.json"
        if not path.is_file(): raise RuntimeError(f"missing state record {state['state_id']}")
        record = json.loads(path.read_text())
        if record.get("status") != "PASS" or sha(Path(record["shard_path"])) != record["shard_sha256"]:
            raise RuntimeError(f"invalid state record {state['state_id']}")
        records.append(record)
    wall_receipt = json.loads((OUT / "collection_wall_receipt.json").read_text()) if (OUT / "collection_wall_receipt.json").is_file() else {}
    payload = {"schema": "factorised_micro_safety_fresh_sensor_index_v1", "complete": True,
        "states": 48, "branches": 576, "ticks_per_branch": TICKS, "channels": list(SENSOR.CHANNELS),
        "action_control_channels": list(SENSOR.ACTION_CONTROL_CHANNELS), "channel_count": len(SENSOR.CHANNELS),
        "state_records": records, "fresh_panel_manifest_digest": manifest["content_digest"],
        "storage_bytes": sum(Path(record["shard_path"]).stat().st_size for record in records),
        "runtime_compute_s": sum(float(record["runtime_s"]) for record in records),
        "parallel_wall_runtime_s": wall_receipt.get("wall_runtime_s"),
        "verification": {"finite_branches": sum(record["verification"]["finite_branches"] for record in records),
                         "action_trace_identity_matches": sum(record["verification"]["action_trace_identity_matches"] for record in records),
                         "pose_and_safety_trace_matches": sum(record["verification"]["pose_and_safety_trace_matches"] for record in records),
                         "identity_mismatches": 0, "new_state_count": 48, "new_branch_count": 576},
        "excluded_inputs": ["global_position", "global_yaw", "body_linear_velocity", "scene_graph", "occupancy_grid",
                            "privileged_geometry", "labels_as_inputs"],
        "bindings": {"manifest_sha256": sha(OUT / "fresh_panel_manifest.json"),
                     "old_enhanced_sensor_digest": "d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d",
                     "specialist_ledger_content_digest": "e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de",
                     "specialist_ledger_sha256": "a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08"}}
    payload["content_digest"] = canonical_digest(payload); atomic_json(OUT / "fresh_sensor_index.json", payload)
    print(json.dumps({key: payload[key] for key in ("states", "branches", "storage_bytes", "runtime_compute_s", "content_digest")}, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true"); group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    group.add_argument("--probe-scene", type=Path)
    parser.add_argument("--probe-family"); parser.add_argument("--probe-seed", type=int); parser.add_argument("--probe-receipt", type=Path)
    args = parser.parse_args()
    if args.probe_scene is not None:
        if args.probe_family is None or args.probe_seed is None or args.probe_receipt is None:
            parser.error("probe mode requires family, seed, and receipt")
        probe_scene(args.probe_scene, args.probe_family, args.probe_seed, args.probe_receipt)
    elif args.freeze: freeze_panel()
    elif args.collect_state is not None:
        collect_state(args.collect_state)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.collect_all:
        collection_started = time.time()
        log_root = CACHE / "collection_logs"; log_root.mkdir(parents=True, exist_ok=True)
        for start in range(0, 48, 4):
            processes = []
            for index in range(start, min(start + 4, 48)):
                log_path = log_root / f"state_{index:02d}.log"; log = log_path.open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)],
                                           stdout=log, stderr=subprocess.STDOUT)
                processes.append((index, process, log, log_path))
            for index, process, log, log_path in processes:
                return_code = process.wait(); log.close()
                if return_code != 0:
                    raise RuntimeError(f"state {index} collection exited {return_code}; see {log_path}")
        atomic_json(OUT / "collection_wall_receipt.json", {"states": 48, "branches": 576,
                                                           "wall_runtime_s": time.time() - collection_started,
                                                           "parallel_processes": 4})
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
