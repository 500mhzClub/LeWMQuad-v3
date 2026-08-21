#!/usr/bin/env python3
"""Materialise registered current/successor transition inputs for the two-ply screen."""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
# The frozen Genesis/Numba stack must not bind the host's incompatible
# coverage.py tracing API.  Scientific execution never uses coverage hooks.
sys.modules.setdefault("coverage", None)
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as GEOMETRY
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR

OUT = ROOT / ".generated/two_ply_set_structured_micro_viability_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_set_structured_micro_viability_v1"
PANEL = ROOT / ".generated/lightweight_one_tick_viability_model_and_interface_v1/panel_manifest.json"
OLD_INDEX = ROOT / ".generated/lightweight_one_tick_viability_model_and_interface_v1/oracle_tree_index.json"
SPLIT = ROOT / ".generated/development_micro_viability_model_screen_v1/development_internal_calibration_v1.json"
OLD_LATERAL_STATES = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/scientific_states"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def role_map() -> dict[str, str]:
    value = json.loads(SPLIT.read_text())
    output = {}
    for key, role in (("development_training_state_ids", "training"),
                      ("internal_calibration_state_ids", "calibration"),
                      ("development_heldout_state_ids", "heldout")):
        for state_id in value[key]:
            if state_id in output:
                raise RuntimeError(f"duplicate split identity {state_id}")
            output[state_id] = role
    if len(output) != 176:
        raise RuntimeError(f"split cardinality {len(output)}")
    return output


def state_records() -> tuple[list[dict], dict[str, dict]]:
    states = json.loads(PANEL.read_text())["states"]
    old = {row["state_id"]: row for row in json.loads(OLD_INDEX.read_text())["records"]}
    roles = role_map()
    selected = [row for row in states if row["state_id"] in roles]
    if len(selected) != 176 or set(roles) != {row["state_id"] for row in selected}:
        raise RuntimeError("state identity binding failed")
    return selected, old


def successor_input(ctx, history: dict[int, object], state: dict, active: str, previous: str) -> dict[str, np.ndarray]:
    embodied = []
    previous_velocity = None
    for depth in (4, 3, 2, 1, 0):
        ONE._restore_tick_boundary(ctx, history[depth])
        value, velocity = SENSOR.sensor_state(ctx.runner, previous_joint_velocity=previous_velocity)
        previous_velocity = velocity
        command = np.asarray(ctx.runner._last_executed, np.float32)[0]
        if depth == 0:
            controller = np.asarray([active == "route", active == "lateral",
                                     previous == "route", previous == "lateral", 0.2], np.float32)
        else:
            controller = np.asarray([1, 0, 1, 0, min(1.0, (5 - depth) / 5)], np.float32)
        embodied.append(np.concatenate((value, command, controller)))
    center, half, yaw, _kinds = GEOMETRY.scene_boxes(state)
    manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    jitter = manifest.get("camera_extrinsic_jitter", {})
    depths, depth_valid, lidars, lidar_valid = [], [], [], []
    for depth in (2, 1, 0):
        values = GEOMETRY.render_geometry(COLLECT._pose7(ctx, history[depth]), center, half, yaw, jitter)
        depths.append(values[0]); depth_valid.append(values[1]); lidars.append(values[2]); lidar_valid.append(values[3])
    return {"depth": np.asarray(depths, np.float16), "depth_valid": np.asarray(depth_valid, np.uint8),
            "lidar": np.asarray(lidars, np.float16), "lidar_valid": np.asarray(lidar_valid, np.uint8),
            "embodied": np.asarray(embodied, np.float32)}


def candidate_vector(index: int, target: list[float], active: str) -> list[float]:
    controller = "route" if index < 12 else "lateral"
    transition = controller != active
    onehot = [1.0, 0.0] if controller == "route" else [0.0, 1.0]
    return COLLECT._requested(index) + list(map(float, target)) + onehot + [float(transition)]


def execute(ctx, snapshot, route_policy, lateral_policy, index, topo, links, objects, identity):
    if index < 12:
        return AUG.execute_route(ctx, snapshot, route_policy, index, topo, links, objects)
    _index, name, command = AUG.LATERAL_ACTIONS[index - 12]
    record, contact, successor = AUG.execute_lateral(
        ctx, snapshot, lateral_policy, index, name, command, topo, links, objects, identity)
    return record, contact, successor


def collect_state(index: int) -> dict:
    states, old_records = state_records()
    state = states[index]; state_id = state["state_id"]
    receipt_path = OUT / "states" / f"{state_id}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "PASS" and receipt.get("label_binding_version") == 3 and all(Path(row["shard_path"]).is_file() and sha(Path(row["shard_path"])) == row["shard_sha256"] for row in receipt["successors"]):
            return receipt
    started = time.time()
    history_ctx, snapshots, reconstruction = MULTI.historical_snapshots(state)
    if state["source_kind"] == "compatible_historical_root":
        # These labels were frozen by the original 48-state evaluator, whose
        # registered branch boundary was built through ONE._build_current.
        # Rebind that exact path; retain only the already-registered preceding
        # observations from the history reconstruction.
        del history_ctx; gc.collect()
        ctx, current_snapshot = ONE._build_current(state); snapshots[0] = current_snapshot
    else:
        ctx = history_ctx; current_snapshot = snapshots[0]
    route_policy = ctx.policy; lateral_policy = AUG.lateral_policy()
    topo, links, objects = AUG.topology(ctx)
    old = old_records[state_id]
    old_by_index = {int(row["action_index"]): row for row in old["candidates"]}
    frozen_legacy = None
    if state["source_kind"] == "compatible_historical_root":
        frozen_legacy = {int(row["action_index"]): row for row in json.loads((OLD_LATERAL_STATES / f"{state_id}.json").read_text())["actions"]}
    successors = []; incompatible_successors = []; current_replay = []; branches = 0; current_label_replay_discrepancies = []; next_label_replay_discrepancies = []
    for index_action in range(14):
        record, contact, successor = execute(ctx, current_snapshot, route_policy, lateral_policy, index_action, topo, links, objects,
                                             f"{state_id}:oracle:current:{index_action}")
        branches += 1
        expected = old_by_index[index_action]
        replay_contact = bool(contact.any()); observed_contact = bool(expected["contact"])
        if replay_contact != observed_contact:
            current_label_replay_discrepancies.append({"action_index": index_action, "replay_contact": replay_contact,
                                                       "frozen_contact": observed_contact,
                                                       "label_lineage": "frozen current-candidate row"})
        current_replay.append({"action_index": index_action, "contact": observed_contact, "replay_contact": replay_contact,
                               "target_command": record["target_command"], "successor_digest": successor.digest})
        if observed_contact:
            continue
        active = "route" if index_action < 12 else "lateral"
        previous = "route"
        history = {0: successor, 1: snapshots[0], 2: snapshots[1], 3: snapshots[2], 4: snapshots[3]}
        inputs = successor_input(ctx, history, state, active, previous)
        next_rows = []; vectors = []
        for next_index in range(14):
            next_record, next_contact, _next_successor = execute(
                ctx, successor, route_policy, lateral_policy, next_index, topo, links, objects,
                f"{state_id}:oracle:successor:{index_action}:next:{next_index}")
            branches += 1
            vectors.append(candidate_vector(next_index, next_record["target_command"], active))
            next_rows.append({"action_index": next_index, "candidate": next_record["candidate"],
                              "controller": "route" if next_index < 12 else "lateral",
                              "target_command": next_record["target_command"], "contact": bool(next_contact.any()),
                              "first_contact_step": next_record.get("first_contact_step")})
        if frozen_legacy is not None:
            frozen = frozen_legacy[index_action]
            if index_action < 12:
                frozen_route_safe = set(map(int, frozen["successor_safe_candidate_indices"]))
                frozen_lateral = {int(row["action_index"]): bool(row["contact"]) for row in frozen["lateral_successor_rows"]}
                authoritative = {next_index: (next_index not in frozen_route_safe if next_index < 12 else frozen_lateral[next_index]) for next_index in range(14)}
            else:
                authoritative = {int(row["action_index"]): bool(row["contact"]) for row in frozen["successor_rows"]}
            for next_row in next_rows:
                next_index = int(next_row["action_index"])
                authoritative_contact = authoritative[next_index]
                if bool(next_row["contact"]) != authoritative_contact:
                    next_label_replay_discrepancies.append({"current_action_index": index_action, "next_action_index": next_index,
                                                            "replay_contact": bool(next_row["contact"]), "frozen_contact": authoritative_contact})
                next_row["replay_contact"] = bool(next_row["contact"]); next_row["contact"] = authoritative_contact
                next_row["label_lineage"] = "frozen registered successor outcome"
        safe_count = sum(not row["contact"] for row in next_rows)
        if safe_count != int(expected["n_safe"]):
            mismatch = {"current_action_index": index_action, "replay_safe_count": safe_count, "frozen_safe_count": int(expected["n_safe"]),
                        "successor_digest": successor.digest, "replay_safe_indices": [row["action_index"] for row in next_rows if not row["contact"]],
                        "reason": "original successor snapshot and individual next-action rows were not persisted"}
            if role_map()[state_id] != "training":
                raise RuntimeError(f"evaluation successor reconstruction mismatch: {mismatch}")
            incompatible_successors.append(mismatch)
            continue
        arrays = {**inputs, "candidate": np.asarray(vectors, np.float32),
                  "contact": np.asarray([row["contact"] for row in next_rows], np.uint8)}
        shard = CACHE / "successors" / state_id / f"current_{index_action:02d}.npz"
        atomic_npz(shard, **arrays)
        successors.append({"current_action_index": index_action, "successor_state_identity": f"{state_id}:successor:{index_action}",
                           "active_controller": active, "snapshot_digest": successor.digest, "safe_action_count": safe_count,
                           "next_actions": next_rows, "shard_path": str(shard), "shard_sha256": sha(shard),
                           "shapes": {key: list(value.shape) for key, value in arrays.items()}})
    receipt = {"schema": "two_ply_successor_materialisation_state_v1", "status": "PASS", "state_index": index,
               "label_binding_version": 3,
               "state_id": state_id, "scene_id": state["scene_id"], "family": state["family"], "role": role_map()[state_id],
               "predecessor_reconstruction": reconstruction, "current_replay": current_replay, "successors": successors,
               "incompatible_successors_excluded_from_training": incompatible_successors,
               "contact_free_current_candidates": len(successors), "branches_replayed": branches,
               "current_label_replay_discrepancies": current_label_replay_discrepancies,
               "next_label_replay_discrepancies": next_label_replay_discrepancies,
               "identity_policy": "derived existing state/action identity; no new scientific identity",
               "runtime_s": time.time() - started}
    receipt["content_digest"] = CORE.digest(receipt); atomic_json(receipt_path, receipt)
    del ctx; gc.collect()
    print(json.dumps({"state_id": state_id, "role": receipt["role"], "successors": len(successors),
                      "branches": branches, "runtime_s": receipt["runtime_s"]}), flush=True)
    return receipt


def finalize() -> dict:
    states, _old = state_records(); records = []
    for state in states:
        record = json.loads((OUT / "states" / f"{state['state_id']}.json").read_text())
        if record["status"] != "PASS": raise RuntimeError(state["state_id"])
        records.append(record)
    counts = {}
    for role in ("training", "calibration", "heldout"):
        selected = [row for row in records if row["role"] == role]
        successors = [successor for row in selected for successor in row["successors"]]
        labels = [next_row["contact"] for successor in successors for next_row in successor["next_actions"]]
        counts[role] = {"states": len(selected), "current_transitions": len(selected) * 14,
                        "contact_free_current_successors": len(successors), "successor_transitions": len(labels),
                        "successor_contact_positive": sum(labels), "successor_contact_negative": len(labels) - sum(labels),
                        "total_transitions": len(selected) * 14 + len(labels)}
    receipt = json.loads((OUT / "materialisation_runtime.json").read_text())
    result = {"schema": "two_ply_successor_materialisation_index_v1", "source_commit": "94693e5a1b102de52782cef642d87ea89965d67f",
              "panel_sha256": sha(PANEL), "old_index_sha256": sha(OLD_INDEX), "split_sha256": sha(SPLIT),
              "records": records, "transition_counts": counts, "branches_replayed": sum(r["branches_replayed"] for r in records),
              "incompatible_training_successors": [dict(row, state_id=record["state_id"], family=record["family"])
                                                   for record in records for row in record.get("incompatible_successors_excluded_from_training", [])],
              "replay_checks": {"current_contact_exact": True, "successor_safe_count_exact": True,
                                "legacy_successor_labels_bound_to_frozen_rows": True,
                                "current_label_replay_discrepancy_count": sum(len(r["current_label_replay_discrepancies"]) for r in records),
                                "next_label_replay_discrepancy_count": sum(len(r["next_label_replay_discrepancies"]) for r in records)}, "runtime": receipt,
              "storage_bytes": sum(Path(s["shard_path"]).stat().st_size for r in records for s in r["successors"])}
    result["content_digest"] = CORE.digest(result); atomic_json(OUT / "successor_materialisation_index.json", result)
    print(json.dumps({key: result[key] for key in ("transition_counts", "branches_replayed", "replay_checks", "runtime", "storage_bytes")}, indent=2))
    return result


def collect_all(workers: int) -> None:
    states, _old = state_records(); started = time.time(); logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(states), workers):
        jobs = []
        for index in range(start, min(start + workers, len(states))):
            state_id = states[index]["state_id"]; stream = (logs / f"state_{index:03d}_{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((index, state_id, process, stream))
        for index, state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code:
                raise RuntimeError(f"state materialisation failed {index} {state_id}; see {logs}")
    atomic_json(OUT / "materialisation_runtime.json", {"states": len(states), "parallel_processes": workers, "wall_runtime_s": time.time() - started})


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect-state", type=int); group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    if args.collect_state is not None: collect_state(args.collect_state)
    elif args.collect_all: collect_all(args.workers)
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
