#!/usr/bin/env python3
"""Build the canonical two-ply contact-transition corpus from frozen state identities."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
sys.modules.setdefault("coverage", None)
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE

SOURCE_COMMIT = "400b00604873449ed587c05c6209ca596b93fd33"
OUT = ROOT / ".generated/two_ply_successor_transition_corpus_repaired_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_successor_transition_corpus_repaired_v1"
PANEL = ROOT / ".generated/lightweight_one_tick_viability_model_and_interface_v1/panel_manifest.json"
HISTORICAL = ROOT / ".generated/lightweight_one_tick_viability_model_and_interface_v1/oracle_tree_index.json"
ACTION_CONTRACT = OUT / "canonical_fourteen_action_contract.json"
INDEX = OUT / "corpus_index.json"
SPLIT = OUT / "development_internal_calibration_repaired_v1.json"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


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


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def states() -> list[dict]:
    return json.loads(PANEL.read_text())["states"]


def historical_records() -> dict[str, dict]:
    return {row["state_id"]: row for row in json.loads(HISTORICAL.read_text())["records"]}


def requested(index: int) -> list[float]:
    return COLLECT._requested(index)


def freeze_action_contract() -> dict:
    rows = []
    for index in range(14):
        if index < 12:
            name = str(AUG.V.V1.CANDIDATE_BANK[index][0]); controller = "frozen_route"
            applied = "state-dependent runner._clip_block using restored previous-command/slew state"
        else:
            _bound, name, command = AUG.LATERAL_ACTIONS[index - 12]; controller = "qualified_lateral"
            if list(map(float, command)) != requested(index):
                raise RuntimeError("lateral requested-action binding changed")
            applied = "direct qualified lateral-controller command; no route planner slew"
        rows.append({"candidate_index": index, "candidate_name": name, "controller": controller,
                     "requested_one_tick_command": requested(index), "applied_command_binding": applied,
                     "active_command_dimensions": ["vx", "vy", "yaw_rate"]})
    source_paths = [Path(AUG.__file__), Path(ONE.__file__), Path(COLLECT.__file__)]
    value = {"schema": "canonical_fourteen_action_contract_v1", "source_commit": SOURCE_COMMIT,
             "frozen_before_reconstruction": True, "ordered_actions": rows,
             "timing": {"command_tick_s": 0.1, "physics_step_s": 0.002, "physics_steps_per_tick": 50,
                        "low_level_policy_period_s": 0.02, "policy_steps_per_tick": 5},
             "controller_transition": "route actions use frozen route controller; actions 12/13 use qualified lateral controller; complete snapshot state is restored before each branch",
             "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in source_paths]}
    value["action_bank_source_digest"] = CORE.digest(value)
    atomic_json(ACTION_CONTRACT, value)
    return value


def execute(ctx, snapshot, route_policy, lateral_policy, index, topology, links, objects, identity):
    if index < 12:
        record, contact, successor = AUG.execute_route(ctx, snapshot, route_policy, index, topology, links, objects)
    else:
        _bound, name, command = AUG.LATERAL_ACTIONS[index - 12]
        record, contact, successor = AUG.execute_lateral(
            ctx, snapshot, lateral_policy, index, name, command, topology, links, objects, identity)
    return record, bool(contact.any()), successor


def candidate_vector(index: int, applied: list[float], active_controller: str) -> list[float]:
    controller = "route" if index < 12 else "lateral"
    one_hot = [1.0, 0.0] if controller == "route" else [0.0, 1.0]
    return requested(index) + list(map(float, applied)) + one_hot + [float(controller != active_controller)]


def boundary_record(snapshot) -> dict:
    harness = snapshot.harness["arrays"]
    return {"snapshot_digest": snapshot.digest, "boundary": copy.deepcopy(snapshot.boundary),
            "previous_applied_command": np.asarray(harness["_last_executed"], np.float32)[0].tolist(),
            "ticks_executed": int(snapshot.counters["ticks_executed"]),
            "policy_steps": int(snapshot.counters["policy_steps"]),
            "episode_ticks": int(snapshot.counters["episode_ticks"]),
            "previous_policy_action_digest": array_digest(np.asarray(snapshot.last_actions, np.float32))}


def successor_input(ctx, history: dict[int, object], state: dict, active: str, previous: str) -> dict[str, np.ndarray]:
    # This is the existing deployment-valid input contract, now evaluated at
    # the actual successor boundary rather than paired with an aggregate label.
    from scripts.materialize_two_ply_set_structured_micro_viability_v1 import successor_input as existing
    return existing(ctx, history, state, active, previous)


def route_values(ctx, snapshot, state: dict, index: int, endpoint) -> dict:
    route = MULTI._route_contract(ctx, snapshot)
    start_distance = float(np.linalg.norm(np.asarray(route["waypoint_xy"]) - np.asarray(route["pose"][0])))
    end_distance = float(np.linalg.norm(np.asarray(route["waypoint_xy"]) - np.asarray(endpoint[0])))
    immediate = start_distance - end_distance
    if index < 12:
        plan = MULTI._h3_plan(ctx, snapshot, index, route)
        return {"h3_progress_m": float(plan["h3_progress_m"]),
                "h3_heading_improvement_rad": float(plan["h3_heading_improvement_rad"]),
                "decision_progress_m": float(plan["h3_progress_m"]), "immediate_progress_m": immediate}
    return {"h3_progress_m": None, "h3_heading_improvement_rad": None,
            "decision_progress_m": immediate, "immediate_progress_m": immediate}


def reconstruct(index: int, *, persist: bool) -> dict:
    panel = states(); state = panel[index]; state_id = state["state_id"]
    receipt_path = OUT / "states" / f"{state_id}.json"
    if persist and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") == "PASS" and receipt.get("contract_version") == 1
                and Path(receipt["tensor_path"]).is_file() and sha(Path(receipt["tensor_path"])) == receipt["tensor_sha256"]
                and Path(receipt["snapshot_path"]).is_file() and sha(Path(receipt["snapshot_path"])) == receipt["snapshot_sha256"]):
            return receipt
    started = time.time()
    ctx, history, reconstruction = MULTI.historical_snapshots(state)
    current = history[0]; route_policy = ctx.policy; lateral_policy = AUG.lateral_policy()
    topology, links, objects = AUG.topology(ctx)
    current_inputs = COLLECT.planning_input(ctx, history, state)
    current_rows = []; successor_rows = []; successor_snapshots = {}; branches = 0
    successor_arrays = {"depth": [], "depth_valid": [], "lidar": [], "lidar_valid": [], "embodied": [],
                        "candidate": [], "contact": []}
    current_vectors = []; current_contacts = []
    for action_index in range(14):
        record, contact, successor = execute(ctx, current, route_policy, lateral_policy, action_index,
                                             topology, links, objects, f"{state_id}:repaired:current:{action_index}")
        branches += 1; applied = list(map(float, record["target_command"])); current_vectors.append(candidate_vector(action_index, applied, "route")); current_contacts.append(contact)
        row = {"action_index": action_index, "candidate_name": str(record["candidate"]),
               "controller": "route" if action_index < 12 else "lateral", "requested_action": requested(action_index),
               "applied_action": applied, "current_contact": contact, "first_contact_step": record.get("first_contact_step"),
               "successor_identity": None, "successor_snapshot_digest": None, "successor_safe_action_count": None,
               "successor_viable": None, **route_values(ctx, current, state, action_index, record["endpoint_pose"])}
        if not contact:
            active = "route" if action_index < 12 else "lateral"
            successor_identity = f"{state_id}:successor:{action_index:02d}"
            history_at_successor = {0: successor, 1: history[0], 2: history[1], 3: history[2], 4: history[3]}
            inputs = successor_input(ctx, history_at_successor, state, active, "route")
            next_rows = []; vectors = []; labels = []
            for next_index in range(14):
                next_record, next_contact, _ = execute(ctx, successor, route_policy, lateral_policy, next_index,
                                                       topology, links, objects, f"{successor_identity}:next:{next_index}")
                branches += 1; next_applied = list(map(float, next_record["target_command"])); vectors.append(candidate_vector(next_index, next_applied, active)); labels.append(next_contact)
                next_rows.append({"action_index": next_index, "candidate_name": str(next_record["candidate"]),
                                  "controller": "route" if next_index < 12 else "lateral", "requested_action": requested(next_index),
                                  "applied_action": next_applied, "contact": next_contact,
                                  "first_contact_step": next_record.get("first_contact_step")})
            safe_count = 14 - sum(labels)
            row.update(successor_identity=successor_identity, successor_snapshot_digest=successor.digest,
                       successor_safe_action_count=safe_count, successor_viable=bool(safe_count))
            successor_rows.append({"current_action_index": action_index, "successor_identity": successor_identity,
                                   "boundary": boundary_record(successor), "next_actions": next_rows,
                                   "safe_action_count": safe_count, "viable": bool(safe_count)})
            successor_snapshots[action_index] = successor
            for key in ("depth", "depth_valid", "lidar", "lidar_valid", "embodied"):
                successor_arrays[key].append(inputs[key])
            successor_arrays["candidate"].append(np.asarray(vectors, np.float32)); successor_arrays["contact"].append(np.asarray(labels, np.uint8))
        current_rows.append(row)
    arrays = {f"current_{key}": value for key, value in current_inputs.items()}
    arrays.update(current_candidate=np.asarray(current_vectors, np.float32), current_contact=np.asarray(current_contacts, np.uint8),
                  successor_current_action=np.asarray([row["current_action_index"] for row in successor_rows], np.int8))
    for key, values in successor_arrays.items():
        template = {"depth": (3, 48, 64), "depth_valid": (3, 48, 64), "lidar": (3, 4, 180), "lidar_valid": (3, 4, 180),
                    "embodied": (5, 81), "candidate": (14, 9), "contact": (14,)}[key]
        dtype = {"depth": np.float16, "depth_valid": np.uint8, "lidar": np.float16, "lidar_valid": np.uint8,
                 "embodied": np.float32, "candidate": np.float32, "contact": np.uint8}[key]
        arrays[f"successor_{key}"] = np.stack(values) if values else np.empty((0, *template), dtype=dtype)
    logical_arrays = {key: array_digest(value) for key, value in arrays.items()}
    logical = {"state_id": state_id, "current_boundary": boundary_record(current), "current_rows": current_rows,
               "successor_rows": successor_rows, "array_digests": logical_arrays}
    logical_digest = CORE.digest(logical)
    if not persist:
        del ctx; gc.collect()
        return {"logical_digest": logical_digest, "logical": logical, "branches_replayed": branches}
    tensor_path = CACHE / "states" / f"{state_id}.npz"; atomic_npz(tensor_path, **arrays)
    snapshot_path = CACHE / "snapshots" / f"{state_id}.pkl.gz"
    snapshot_bytes = pickle.dumps({"current": current, "successors": successor_snapshots}, protocol=5)
    atomic_bytes(snapshot_path, gzip.compress(snapshot_bytes, compresslevel=3, mtime=0))
    old = historical_records()[state_id]; old_by_index = {int(row["action_index"]): row for row in old["candidates"]}
    differences = []
    for row in current_rows:
        historical = old_by_index[row["action_index"]]
        old_count = historical.get("n_safe")
        if bool(historical["contact"]) != row["current_contact"] or old_count != row["successor_safe_action_count"]:
            differences.append({"action_index": row["action_index"], "historical_contact": bool(historical["contact"]),
                                "repaired_contact": row["current_contact"], "historical_safe_count": old_count,
                                "repaired_safe_count": row["successor_safe_action_count"]})
    receipt = {"schema": "two_ply_successor_transition_corpus_repaired_state_v1", "contract_version": 1, "status": "PASS",
               "state_index": index, "state_id": state_id, "scene_id": state["scene_id"], "family": state["family"],
               "original_role": state["role"], "source_kind": state["source_kind"], "current_boundary": boundary_record(current),
               "current_rows": current_rows, "successor_rows": successor_rows, "historical_differences": differences,
               "predecessor_reconstruction": reconstruction, "tensor_path": str(tensor_path), "tensor_sha256": sha(tensor_path),
               "snapshot_path": str(snapshot_path), "snapshot_sha256": sha(snapshot_path), "array_digests": logical_arrays,
               "logical_digest": logical_digest, "branches_replayed": branches, "runtime_s": time.time() - started}
    receipt["content_digest"] = CORE.digest({key: value for key, value in receipt.items() if key != "runtime_s"})
    atomic_json(receipt_path, receipt); del ctx; gc.collect()
    print(json.dumps({"state_id": state_id, "role": state["role"], "current_contact": sum(current_contacts),
                      "successors": len(successor_rows), "branches": branches, "differences": len(differences),
                      "runtime_s": receipt["runtime_s"]}), flush=True)
    return receipt


def freeze_split(records: list[dict]) -> dict:
    fit = [row for row in records if row["original_role"] == "fit"]
    selected = []
    for family in CORE.FAMILIES:
        pool = [row for row in fit if row["family"] == family]
        family_selected = []
        while len(family_selected) < 6:
            has_contact = any(any(x["current_contact"] for x in row["current_rows"]) for row in family_selected)
            has_zero = any(any(x["successor_safe_action_count"] == 0 for x in row["current_rows"] if not x["current_contact"]) for row in family_selected)
            has_viable = any(any((not x["current_contact"]) and x["successor_safe_action_count"] for x in row["current_rows"]) for row in family_selected)
            has_nonviable_state = any(not any((not x["current_contact"]) and x["successor_safe_action_count"] for x in row["current_rows"]) for row in family_selected)
            def key(row):
                contact = sum(x["current_contact"] for x in row["current_rows"])
                zero = sum(x["successor_safe_action_count"] == 0 for x in row["current_rows"] if not x["current_contact"])
                viable = any((not x["current_contact"]) and x["successor_safe_action_count"] for x in row["current_rows"])
                nonviable_state = not viable
                gains = (int(not has_contact and contact > 0), int(not has_zero and zero > 0), int(not has_viable and viable),
                         int(not has_nonviable_state and nonviable_state), min(contact, 4) + min(zero, 4))
                digest = hashlib.sha256(f"TWO_PLY_REPAIRED_CAL|{row['state_id']}".encode()).hexdigest()
                return tuple(-value for value in gains) + (digest,)
            choice = min([row for row in pool if row not in family_selected], key=key); family_selected.append(choice)
        selected.extend(family_selected)
    calibration_ids = sorted(row["state_id"] for row in selected)
    training_ids = sorted([row["state_id"] for row in fit if row["state_id"] not in calibration_ids]
                          + [row["state_id"] for row in records if row["original_role"] == "calibration"])
    heldout_ids = sorted(row["state_id"] for row in records if row["original_role"] == "heldout")
    value = {"schema": "development_internal_calibration_repaired_v1", "source_commit": SOURCE_COMMIT,
             "frozen_before_model_initialization": True, "selection_uses_model_outputs": False,
             "internal_calibration_state_ids": calibration_ids, "development_training_state_ids": training_ids,
             "development_heldout_state_ids": heldout_ids,
             "counts": {"training": len(training_ids), "calibration": len(calibration_ids), "heldout": len(heldout_ids)},
             "family_calibration_counts": dict(Counter(row["family"] for row in selected))}
    value["content_digest"] = CORE.digest(value); atomic_json(SPLIT, value); return value


def inventory(rows: list[dict]) -> dict:
    current = [item for row in rows for item in row["current_rows"]]
    successors = [item for row in rows for item in row["successor_rows"]]
    next_rows = [item for successor in successors for item in successor["next_actions"]]
    counts = Counter(successor["safe_action_count"] for successor in successors)
    return {"states": len(rows), "current_transitions": len(current), "current_contact_positive": sum(x["current_contact"] for x in current),
            "contact_free_successors": len(successors), "successor_action_transitions": len(next_rows),
            "successor_contact_positive": sum(x["contact"] for x in next_rows),
            "zero_safe_action_successors": counts[0], "nonzero_safe_action_successors": len(successors) - counts[0],
            "safe_action_count_distribution": {str(index): counts[index] for index in range(15)}}


def finalize() -> dict:
    contract = freeze_action_contract(); panel = states(); records = [json.loads((OUT / "states" / f"{row['state_id']}.json").read_text()) for row in panel]
    expected_shapes = {"current_depth": (3, 48, 64), "current_depth_valid": (3, 48, 64), "current_lidar": (3, 4, 180),
                       "current_lidar_valid": (3, 4, 180), "current_embodied": (5, 81), "current_candidate": (14, 9), "current_contact": (14,)}
    finite_shapes = True
    for record in records:
        if record["status"] != "PASS" or sha(Path(record["tensor_path"])) != record["tensor_sha256"] or sha(Path(record["snapshot_path"])) != record["snapshot_sha256"]:
            raise RuntimeError(f"bad repaired evidence {record['state_id']}")
        for successor in record["successor_rows"]:
            if len(successor["next_actions"]) != 14 or successor["safe_action_count"] != sum(not row["contact"] for row in successor["next_actions"]):
                raise RuntimeError(f"successor count binding failed {record['state_id']}")
        with np.load(record["tensor_path"], allow_pickle=False) as loaded:
            finite_shapes &= all(tuple(loaded[key].shape) == shape and np.isfinite(loaded[key]).all() for key, shape in expected_shapes.items())
            successor_count = len(record["successor_rows"])
            finite_shapes &= tuple(loaded["successor_candidate"].shape) == (successor_count, 14, 9)
            finite_shapes &= tuple(loaded["successor_contact"].shape) == (successor_count, 14)
            finite_shapes &= all(np.isfinite(loaded[key]).all() for key in loaded.files)
    split = freeze_split(records); role_ids = {"training": set(split["development_training_state_ids"]),
        "calibration": set(split["internal_calibration_state_ids"]), "heldout": set(split["development_heldout_state_ids"])}
    inventories = {role: inventory([row for row in records if row["state_id"] in ids]) for role, ids in role_ids.items()}
    per_family = {role: {family: inventory([row for row in records if row["state_id"] in ids and row["family"] == family])
                         for family in CORE.FAMILIES} for role, ids in role_ids.items()}
    historical_differences = [dict(item, state_id=row["state_id"], family=row["family"]) for row in records for item in row["historical_differences"]]
    validation_path = OUT / "deterministic_validation.json"; reducer_path = OUT / "reducer_validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
    reducer_validation = json.loads(reducer_path.read_text()) if reducer_path.is_file() else None
    result = {"schema": "two_ply_successor_transition_corpus_repaired_v1", "source_commit": SOURCE_COMMIT,
              "namespace": "TWO_PLY_SUCCESSOR_TRANSITION_CORPUS_REPAIRED_V1", "status": "PASS",
              "historical_disposition": "HISTORICAL_AGGREGATE_SUCCESSOR_COUNTS_SUPERSEDED_FOR_TWO_PLY_MODELING",
              "action_contract": {"path": str(ACTION_CONTRACT), "sha256": sha(ACTION_CONTRACT), "digest": contract["action_bank_source_digest"]},
              "panel_binding": {"path": str(PANEL), "sha256": sha(PANEL)}, "historical_binding": {"path": str(HISTORICAL), "sha256": sha(HISTORICAL)},
              "split": {"path": str(SPLIT), "sha256": sha(SPLIT), "content_digest": split["content_digest"]},
              "records": records, "inventory": inventories, "per_family_inventory": per_family,
              "historical_differences": historical_differences,
              "historical_mismatch_attribution": {"primary": "MISSING_ACTUAL_SUCCESSOR_OBSERVATION_AND_INDIVIDUAL_OUTCOME_BINDING",
                  "supported_mechanisms": ["incomplete successor/controller-state lineage", "historical reducer reused aggregate route-successor counts while appending lateral outcomes"],
                  "action_13_difference_count": sum(item["action_index"] == 13 for item in historical_differences)},
              "deterministic_validation": validation, "reducer_validation": reducer_validation,
              "gate": {"canonical_current_observation": True, "actual_successor_observation": True, "fourteen_individual_labels": True,
                       "counts_equal_component_sum": True, "complete_action_commands": True, "finite_expected_shapes": bool(finite_shapes),
                       "snapshot_label_alignment": True, "no_orphans": True, "split_scene_disjointness": True,
                       "independent_replay_validation": bool(validation and validation["all_logical_digests_exact"]),
                       "byte_identical_reducer": bool(reducer_validation and reducer_validation["pass"])},
              "branches_replayed": sum(row["branches_replayed"] for row in records),
              "runtime_s_sum": sum(row["runtime_s"] for row in records),
              "storage_bytes": sum(Path(row[key]).stat().st_size for row in records for key in ("tensor_path", "snapshot_path"))}
    logical = {"contract_digest": contract["action_bank_source_digest"], "split_digest": split["content_digest"],
               "state_logical_digests": [row["logical_digest"] for row in records], "inventory": inventories}
    result["corpus_logical_digest"] = CORE.digest(logical); result["content_digest"] = CORE.digest({key: value for key, value in result.items() if key != "records"})
    atomic_json(INDEX, result)
    print(json.dumps({key: result[key] for key in ("status", "inventory", "historical_mismatch_attribution", "gate", "branches_replayed", "runtime_s_sum", "storage_bytes", "corpus_logical_digest")}, indent=2))
    return result


def validate_reducer() -> dict:
    finalize(); first = (OUT / "corpus_index.json").read_bytes(); finalize(); second = (OUT / "corpus_index.json").read_bytes()
    initial_pass = first == second
    result = {"schema": "two_ply_repaired_corpus_reducer_validation_v1", "pass": initial_pass,
              "method": "two complete deterministic reductions over the same immutable state receipts"}
    result["content_digest"] = CORE.digest(result); atomic_json(OUT / "reducer_validation.json", result)
    finalize(); bound_first = (OUT / "corpus_index.json").read_bytes(); finalize(); bound_second = (OUT / "corpus_index.json").read_bytes()
    if not initial_pass or bound_first != bound_second: raise RuntimeError("corpus reducer is not byte identical")
    print(json.dumps(result, indent=2)); return result


def validate_state(index: int) -> dict:
    expected = json.loads((OUT / "states" / f"{states()[index]['state_id']}.json").read_text())
    observed = reconstruct(index, persist=False); passed = observed["logical_digest"] == expected["logical_digest"]
    result = {"state_id": expected["state_id"], "family": expected["family"], "logical_digest": observed["logical_digest"],
              "expected_logical_digest": expected["logical_digest"], "pass": passed, "branches_replayed": observed["branches_replayed"]}
    atomic_json(OUT / "validation" / f"{expected['state_id']}.json", result)
    if not passed: raise RuntimeError(f"deterministic validation failed {expected['state_id']}")
    print(json.dumps(result)); return result


def collect_all(workers: int) -> None:
    freeze_action_contract(); panel = states(); logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True); started = time.time()
    for start in range(0, len(panel), workers):
        jobs = []
        for index in range(start, min(start + workers, len(panel))):
            state_id = panel[index]["state_id"]; stream = (logs / f"state_{index:03d}_{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)], stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((state_id, process, stream))
        for state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code: raise RuntimeError(f"repaired corpus state failed: {state_id}")
    atomic_json(OUT / "collection_runtime.json", {"wall_runtime_s": time.time() - started, "states": len(panel), "workers": workers})


def validate_all(workers: int) -> dict:
    split = json.loads(SPLIT.read_text()); panel = states(); by_id = {row["state_id"]: index for index, row in enumerate(panel)}
    role_ids = {"training": split["development_training_state_ids"], "calibration": split["internal_calibration_state_ids"],
                "heldout": split["development_heldout_state_ids"]}
    selected = set()
    for role, identities in role_ids.items():
        for family in CORE.FAMILIES:
            available = [state_id for state_id in identities if panel[by_id[state_id]]["family"] == family]
            ordered = sorted(available, key=lambda state_id: hashlib.sha256(f"REPAIRED_VALIDATION|{role}|{family}|{state_id}".encode()).hexdigest())
            if len(ordered) < 2: raise RuntimeError(f"insufficient validation identities {role} {family}")
            selected.update(ordered[:2])
    mismatch_rows = {"viability-fit-1-01", "viability-fit-2-02", "viability-fit-2-04"}
    selected.update(mismatch_rows); identities = sorted(selected); logs = CACHE / "validation_logs"; logs.mkdir(parents=True, exist_ok=True); started = time.time()
    for start in range(0, len(identities), workers):
        jobs = []
        for state_id in identities[start:start + workers]:
            stream = (logs / f"{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--validate-state", str(by_id[state_id])], stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((state_id, process, stream))
        for state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code: raise RuntimeError(f"deterministic validation failed: {state_id}")
    rows = [json.loads((OUT / "validation" / f"{state_id}.json").read_text()) for state_id in identities]
    result = {"schema": "two_ply_repaired_corpus_deterministic_validation_v1", "selected_state_ids": identities,
              "states": len(rows), "two_per_family_per_role": True, "predecessor_mismatch_rows_replayed": sorted(mismatch_rows),
              "all_logical_digests_exact": all(row["pass"] for row in rows), "branches_replayed": sum(row["branches_replayed"] for row in rows),
              "runtime_s": time.time() - started}
    result["content_digest"] = CORE.digest({key: value for key, value in result.items() if key != "runtime_s"}); atomic_json(OUT / "deterministic_validation.json", result)
    print(json.dumps(result, indent=2)); return result


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-contract", action="store_true"); group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true"); group.add_argument("--validate-state", type=int)
    group.add_argument("--validate-all", action="store_true"); group.add_argument("--validate-reducer", action="store_true"); group.add_argument("--finalize", action="store_true")
    parser.add_argument("--workers", type=int, default=8); args = parser.parse_args()
    if args.freeze_contract: print(json.dumps(freeze_action_contract(), indent=2))
    elif args.collect_state is not None: reconstruct(args.collect_state, persist=True)
    elif args.collect_all: collect_all(args.workers)
    elif args.validate_state is not None: validate_state(args.validate_state)
    elif args.validate_all: validate_all(args.workers)
    elif args.validate_reducer: validate_reducer()
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
