#!/usr/bin/env python3
"""Freeze and collect the nested-fit and fresh-evaluation states for scaling V2."""
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import collect_factorised_micro_safety_world_model_v1 as BASE

OUT = ROOT / ".generated/factorised_micro_safety_data_scaling_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorised_micro_safety_data_scaling_v2")
OLD_PANEL = ROOT / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json"
V1_PANEL = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_panel_manifest.json"
PREDICTOR_MANIFEST = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/factorial_manifest.json")
FAMILIES = BASE.FAMILIES
DOMAIN = "FACTORISED_MICRO_SAFETY_DATA_SCALING_V2/NEW_PANEL/2026-08-20"
TICKS = 15
NEW_FIT_PER_FAMILY = 24
CAL_PER_FAMILY = 6
HELD_PER_FAMILY = 6
TOTAL_PER_FAMILY = NEW_FIT_PER_FAMILY + CAL_PER_FAMILY + HELD_PER_FAMILY


def sha(path: Path) -> str:
    return BASE.sha(path)


def canonical_digest(value) -> str:
    return BASE.canonical_digest(value)


def atomic_json(path: Path, payload: dict) -> None:
    BASE.atomic_json(path, payload)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    BASE.atomic_npz(path, **arrays)


def exclusions() -> dict:
    old = json.loads(OLD_PANEL.read_text())
    prior = json.loads(V1_PANEL.read_text())
    predictor = json.loads(PREDICTOR_MANIFEST.read_text())
    old_scenes = {str(row["scene_id"]) for row in old["state_candidates"]}
    prior_scenes = {str(row["scene_id"]) for row in prior["states"]}
    predictor_scenes = {str(row["episode_cluster"]).split("/")[0] for row in predictor["rows"]}
    return {"original_fit48": old_scenes, "prior_fresh48": prior_scenes,
            "predictor_training_and_selection": predictor_scenes,
            "union": old_scenes | prior_scenes | predictor_scenes,
            "predictor_manifest_digest": predictor["digest"]}


def ordered_scene_dirs(family: str, excluded: set[str]) -> list[Path]:
    paths = [path for path in BASE.SCENE_ROOT.glob(f"*/{family}/*") if path.is_dir() and path.name not in excluded]
    return sorted(paths, key=lambda path: hashlib.sha256(f"{DOMAIN}|{family}|{path.name}".encode()).hexdigest())


def probe_scene(scene_dir: Path, family: str, seed: int, receipt_path: Path) -> None:
    BASE.probe_scene(scene_dir, family, seed, receipt_path)


def split_identity(family_index: int, accepted: int) -> tuple[str, str]:
    if accepted < NEW_FIT_PER_FAMILY:
        return "fit192_extra", f"scale-fit-{family_index}-{accepted:02d}"
    if accepted < NEW_FIT_PER_FAMILY + CAL_PER_FAMILY:
        return "calibration", f"scale-cal-{family_index}-{accepted - NEW_FIT_PER_FAMILY:02d}"
    return "heldout", f"scale-held-{family_index}-{accepted - NEW_FIT_PER_FAMILY - CAL_PER_FAMILY:02d}"


def freeze_panel() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "panel_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("content_digest") != canonical_digest({k: v for k, v in payload.items() if k != "content_digest"}):
            raise RuntimeError("existing scaling panel manifest digest mismatch")
        print(json.dumps({"status": "REUSED", "states": len(payload["states"]), "digest": payload["content_digest"]}))
        return payload
    excluded = exclusions()
    selected, scan = [], []
    probe_root = CACHE / "eligibility_receipts"
    probe_root.mkdir(parents=True, exist_ok=True)
    for family_index, family in enumerate(FAMILIES):
        accepted = 0
        scene_dirs = ordered_scene_dirs(family, excluded["union"])
        for batch_start in range(0, len(scene_dirs), 4):
            if accepted >= TOTAL_PER_FAMILY:
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
                code = process.wait(); log.close()
                if code != 0 and not receipt.is_file():
                    atomic_json(receipt, {"family": family, "scene_id": receipt.stem.split("__", 1)[-1],
                                         "status": "ERROR", "reason": f"probe_exit_{code}", "scan_runtime_s": 0.0})
            for scene_dir in scene_dirs[batch_start:batch_start + 4]:
                receipt = probe_root / f"{family}__{scene_dir.name}.json"
                record = json.loads(receipt.read_text())
                if record["status"] == "ERROR":
                    receipt.unlink()
                    seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                    command = [sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir),
                               "--probe-family", family, "--probe-seed", str(seed), "--probe-receipt", str(receipt)]
                    with receipt.with_suffix(".retry.log").open("wb") as log:
                        retry = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if retry.returncode != 0 or not receipt.is_file():
                        raise RuntimeError(f"eligibility retry failed for {scene_dir.name}")
                    record = json.loads(receipt.read_text())
                    if record["status"] == "ERROR":
                        raise RuntimeError(f"eligibility error after retry for {scene_dir.name}: {record.get('reason')}")
                scan.append(record)
                if record["status"] == "ELIGIBLE" and accepted < TOTAL_PER_FAMILY:
                    split, state_id = split_identity(family_index, accepted)
                    record.update(split=split, state_id=state_id)
                    selected.append(dict(record)); accepted += 1
        if accepted != TOTAL_PER_FAMILY:
            raise RuntimeError(f"{family}: found {accepted}/{TOTAL_PER_FAMILY} eligible states")
    scenes = {row["scene_id"] for row in selected}
    split_names = ("fit192_extra", "calibration", "heldout")
    payload = {"schema": "factorised_micro_safety_data_scaling_v2_panel_manifest_v1", "domain": DOMAIN,
        "frozen_before_candidate_execution": True, "states": selected, "state_count": 144, "candidate_count": 12,
        "split_state_count": {split: sum(row["split"] == split for row in selected) for split in split_names},
        "family_split_state_count": {family: {split: sum(row["family"] == family and row["split"] == split for row in selected)
                                               for split in split_names} for family in FAMILIES},
        "disjointness": {"original_fit48_scene_overlap": len(scenes & excluded["original_fit48"]),
                          "prior_fresh48_scene_overlap": len(scenes & excluded["prior_fresh48"]),
                          "predictor_scene_overlap": len(scenes & excluded["predictor_training_and_selection"]),
                          "distinct_scene_count": len(scenes), "distinct_episode_state_cluster_count": len({row["state_id"] for row in selected})},
        "bindings": {"original_panel_sha256": sha(OLD_PANEL), "prior_v1_panel_sha256": sha(V1_PANEL),
                     "prior_v1_panel_digest": json.loads(V1_PANEL.read_text())["content_digest"],
                     "predictor_manifest_sha256": sha(PREDICTOR_MANIFEST),
                     "predictor_manifest_content_digest": excluded["predictor_manifest_digest"]}, "scan": scan}
    if any(payload["disjointness"][key] for key in ("original_fit48_scene_overlap", "prior_fresh48_scene_overlap", "predictor_scene_overlap")):
        raise RuntimeError("prospective panel disjointness failed")
    payload["content_digest"] = canonical_digest(payload)
    atomic_json(path, payload)
    print(json.dumps({"status": "FROZEN", "states": 144, "digest": payload["content_digest"],
                      "split_state_count": payload["split_state_count"], "disjointness": payload["disjointness"]}, indent=2))
    return payload


def collect_state(index: int) -> dict:
    manifest = freeze_panel(); state = manifest["states"][index]
    record_path = OUT / "states" / f"{state['state_id']}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text()); shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            print(json.dumps({"state_id": state["state_id"], "status": "REUSED"}), flush=True); return record
    started = time.time(); shared = BASE.V.V1._load_shared("cpu")
    ctx = BASE.V.V1.build_context(Path(state["scene_dir"]), seed=int(state["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40): ctx.drive_one_block()
    topology = BASE.V.link_topology(ctx); eligible = BASE.V.eligible_here(ctx, topology)
    if isinstance(eligible, str): raise RuntimeError(f"{state['state_id']}: eligibility changed: {eligible}")
    goal_record, _field = eligible
    route = ctx.scene_graph.shortest_path(int(goal_record["cell_id"]), int(goal_record["goal"]["landmark_cell"]))
    if route is None or list(map(int, route[:3])) != state["waypoint_path_cells"]:
        raise RuntimeError(f"{state['state_id']}: route identity changed")
    start_pose = ctx.pose(); waypoint = ctx.scene_graph.cell_center(int(route[2])); waypoint_body = BASE.body_relative(start_pose, waypoint)
    heading = BASE.route_heading(ctx, route)
    snapshot = BASE.V.V1.capture_branch_state(ctx, goal=dict(goal_record["goal"]),
        identity={"state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"]})
    current_rows, future_rows, action_rows, label_rows, pose_rows, branch_records = [], [], [], [], [], []
    for candidate_index, candidate in enumerate(BASE.V.V1.CANDIDATE_BANK):
        branch = BASE.execute_candidate(ctx, snapshot, candidate, topology=topology)
        current_rows.append(branch["current"]); future_rows.append(branch["future"]); action_rows.append(branch["action_control"])
        label_rows.append(branch["labels"]); pose_rows.append(branch["poses"])
        start_error = abs(BASE.wrap(heading - float(start_pose[1]))); endpoint = branch["poses"][-1]
        endpoint_body = BASE.body_relative(start_pose, (endpoint[0], endpoint[1]))
        distance_start = math.hypot(*waypoint_body); distance_end = math.hypot(waypoint_body[0] - endpoint_body[0], waypoint_body[1] - endpoint_body[1])
        branch_records.append({"branch_id": f"{state['state_id']}:{candidate_index:02d}", "state_id": state["state_id"],
            "candidate_index": candidate_index, "candidate": branch["name"], "primitives": branch["primitives"],
            "requested": branch["requested"], "post_slew": branch["post_slew"], "p_d": float(distance_start - distance_end),
            "p_theta": float(start_error - abs(BASE.wrap(heading - float(endpoint[2])))), "unsafe": bool(branch["labels"][-1, 4]),
            "contact": bool(branch["labels"][-1, 2]), "stuck": bool(branch["labels"][-1, 3]), "completed": bool(distance_end <= .35),
            "distance_start": float(distance_start), "distance_end": float(distance_end), "route_heading_world_rad": float(heading),
            "finite": branch["finite"], "action_trace_identity_verified": bool(branch["requested_identity_matches"] and branch["post_slew_trace_matches"]),
            "pose_and_safety_trace_verified": bool(len(branch["poses"]) == TICKS and len(branch["labels"]) == TICKS)})
    arrays = {"current": np.stack(current_rows).astype(np.float32), "future": np.stack(future_rows).astype(np.float32),
              "action_control": np.stack(action_rows).astype(np.float32), "labels": np.stack(label_rows).astype(np.float32),
              "poses": np.stack(pose_rows).astype(np.float64)}
    shard = CACHE / "sensors" / f"{state['state_id']}.npz"; atomic_npz(shard, **arrays)
    record = {"schema": "factorised_micro_safety_data_scaling_v2_state_v1", "status": "PASS", "state_index": index,
        "state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"], "split": state["split"],
        "snapshot_digest": snapshot.digest, "branches": branch_records, "shard_path": str(shard), "shard_sha256": sha(shard),
        "shapes": {key: list(value.shape) for key, value in arrays.items()}, "dtype": {key: str(value.dtype) for key, value in arrays.items()},
        "verification": {"branch_count": 12, "tick_count": TICKS, "finite_branches": sum(row["finite"] for row in branch_records),
                         "action_trace_identity_matches": sum(row["action_trace_identity_verified"] for row in branch_records),
                         "pose_and_safety_trace_matches": sum(row["pose_and_safety_trace_verified"] for row in branch_records),
                         "route_identity_match": True, "state_identity_match": True}, "runtime_s": time.time() - started}
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
    wall = json.loads((OUT / "collection_wall_receipt.json").read_text())
    payload = {"schema": "factorised_micro_safety_data_scaling_v2_sensor_index_v1", "complete": True,
        "states": 144, "branches": 1728, "ticks_per_branch": TICKS, "channels": list(BASE.SENSOR.CHANNELS),
        "action_control_channels": list(BASE.SENSOR.ACTION_CONTROL_CHANNELS), "channel_count": len(BASE.SENSOR.CHANNELS),
        "state_records": records, "panel_manifest_digest": manifest["content_digest"],
        "storage_bytes": sum(Path(record["shard_path"]).stat().st_size for record in records),
        "runtime_compute_s": sum(float(record["runtime_s"]) for record in records), "parallel_wall_runtime_s": wall["wall_runtime_s"],
        "verification": {"finite_branches": sum(r["verification"]["finite_branches"] for r in records),
                         "action_trace_identity_matches": sum(r["verification"]["action_trace_identity_matches"] for r in records),
                         "pose_and_safety_trace_matches": sum(r["verification"]["pose_and_safety_trace_matches"] for r in records),
                         "identity_mismatches": 0, "new_state_count": 144, "new_branch_count": 1728},
        "excluded_inputs": ["global_position", "global_yaw", "body_linear_velocity", "scene_graph", "occupancy_grid", "privileged_geometry", "labels_as_inputs"],
        "bindings": {**manifest["bindings"], "fit48_checkpoint_sha256": "93f919238ff7b757b77f5281f45c59818c9f2b33fa5fbd96a2554b7aea14776e",
                     "enhanced_sensor_index_digest": "d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d"}}
    payload["content_digest"] = canonical_digest(payload); atomic_json(OUT / "sensor_index.json", payload)
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
        if args.probe_family is None or args.probe_seed is None or args.probe_receipt is None: parser.error("probe mode requires family, seed, and receipt")
        probe_scene(args.probe_scene, args.probe_family, args.probe_seed, args.probe_receipt)
    elif args.freeze: freeze_panel()
    elif args.collect_state is not None:
        collect_state(args.collect_state); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.collect_all:
        started = time.time(); logs = CACHE / "collection_logs"; logs.mkdir(parents=True, exist_ok=True)
        for start in range(0, 144, 4):
            processes = []
            for index in range(start, min(start + 4, 144)):
                path = logs / f"state_{index:03d}.log"; handle = path.open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=handle, stderr=subprocess.STDOUT)
                processes.append((index, process, handle, path))
            for index, process, handle, path in processes:
                code = process.wait(); handle.close()
                if code != 0: raise RuntimeError(f"state {index} collection exited {code}; see {path}")
        atomic_json(OUT / "collection_wall_receipt.json", {"states": 144, "branches": 1728,
                    "wall_runtime_s": time.time() - started, "parallel_processes": 4})
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
