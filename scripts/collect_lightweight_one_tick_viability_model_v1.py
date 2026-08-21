#!/usr/bin/env python3
"""Freeze and collect the one-tick viability fit and fresh evaluation panels."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as GEOMETRY
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_worlds import build_corpus, plan_corpus


OUT = ROOT / ".generated/lightweight_one_tick_viability_model_and_interface_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/lightweight_one_tick_viability_model_and_interface_v1"
CORPUS = ROOT / ".generated/scene_corpus/lightweight_one_tick_viability_v1_20260821"
RESERVE_CORPUS = ROOT / ".generated/scene_corpus/lightweight_one_tick_viability_v1_reserve_20260821"
MANIFEST = OUT / "panel_manifest.json"
INDEX = OUT / "oracle_tree_index.json"
OLD_PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
OLD_LATERAL_STATES = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/scientific_states"
OLD_ORIGINAL_PANEL = ROOT / ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json"
OLD_FACTORISED_PANEL = ROOT / ".generated/factorised_micro_safety_world_model_v1/fresh_panel_manifest.json"
OLD_SCALING_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
PREDICTOR_MANIFEST = Path.home() / ".cache/lewm_go2_temporal_v03/proprio_v1/factorial_manifest.json"
DOMAIN = "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_AND_INTERFACE_V1/2026-08-21"
POOL_PER_FAMILY = 72
NEW_FIT_PER_FAMILY = 20
CAL_PER_FAMILY = 6
HELD_PER_FAMILY = 6


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


def ensure_corpus() -> dict:
    corpus_manifest = CORPUS / "corpus.json"
    if not corpus_manifest.is_file():
        totals = {"development_pool": {family: POOL_PER_FAMILY for family in CORE.FAMILIES}}
        plan = plan_corpus(plan_seed=CORE.SEED, totals=totals, validate=True)
        result = build_corpus(plan, CORPUS, emit_genesis=True)
        if result.scene_count != POOL_PER_FAMILY * len(CORE.FAMILIES):
            raise RuntimeError("new corpus cardinality mismatch")
    return json.loads(corpus_manifest.read_text())


def excluded_scenes() -> dict[str, set[str]]:
    old = json.loads(OLD_ORIGINAL_PANEL.read_text())["state_candidates"]
    factorised = json.loads(OLD_FACTORISED_PANEL.read_text())["states"]
    scaling = json.loads(OLD_SCALING_PANEL.read_text())["states"]
    wide = json.loads(OLD_PANEL.read_text())["states"]
    predictor = json.loads(PREDICTOR_MANIFEST.read_text())["rows"]
    groups = {
        "historical_development": {str(row["scene_id"]) for row in old + factorised + scaling + wide},
        "predictor": {str(row["episode_cluster"]).split("/")[0] for row in predictor},
    }
    groups["union"] = groups["historical_development"] | groups["predictor"]
    return groups


def point_clearance(scene_dir: Path, xy: tuple[float, float]) -> float:
    scene = json.loads((scene_dir / "genesis_scene.json").read_text())
    point = np.asarray(xy, np.float64); values = []
    for row in scene["objects"]:
        if row.get("kind") == "ground":
            continue
        center = np.asarray(row["center_xyz_m"][:2], np.float64)
        half = np.asarray(row["size_xyz_m"][:2], np.float64) / 2
        yaw = float(row.get("yaw_rad", 0.0)); delta = point - center
        local = np.asarray([
            math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
            -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1],
        ])
        outside = np.maximum(np.abs(local) - half, 0.0)
        values.append(float(np.linalg.norm(outside)))
    return min(values) if values else float("inf")


def probe_scene(scene_dir: Path, family: str, seed: int, receipt: Path) -> None:
    record = {"scene_dir": str(scene_dir), "scene_id": scene_dir.name, "family": family, "seed": seed}
    started = time.time()
    try:
        ctx = V.V1.build_context(scene_dir, seed=seed, backend="cpu", shared=V.V1._load_shared("cpu"))
        ctx.begin_episode()
        for _ in range(40):
            ctx.drive_one_block()
        topology = V.link_topology(ctx); eligible = V.eligible_here(ctx, topology)
        if isinstance(eligible, str):
            record.update(status="INELIGIBLE", reason=eligible)
        else:
            goal, _field = eligible; pose = ctx.pose()
            route = ctx.scene_graph.shortest_path(int(goal["cell_id"]), int(goal["goal"]["landmark_cell"]))
            native_contact = bool(V._contact_count(ctx, topology))
            if route is None or len(route) < 3 or native_contact:
                record.update(status="INELIGIBLE", reason="route_or_preexisting_contact")
            else:
                waypoint = ctx.scene_graph.cell_center(int(route[2]))
                command = np.asarray(ctx.runner._last_executed, np.float32)[0]
                heading = MULTI.EMBODIED.route_heading(ctx, list(map(int, route)))
                record.update(
                    status="ELIGIBLE", warmup_blocks=40, goal=dict(goal["goal"]),
                    start_pose=[list(map(float, pose[0])), float(pose[1]), float(pose[2])],
                    waypoint_path_cells=list(map(int, route[:3])), waypoint_xy=list(map(float, waypoint)),
                    waypoint_body_xy=MULTI.EMBODIED.body_relative(pose, waypoint),
                    route_heading_world_rad=float(heading), current_applied_command=command.tolist(),
                    current_command_magnitude=float(np.linalg.norm(command[[0, 2]])),
                    motion_class="turning" if abs(float(command[2])) >= 0.10 else "straight",
                    current_scene_object_point_clearance_m=point_clearance(scene_dir, pose[0]),
                    pre_existing_disallowed_contact=False,
                )
    except Exception as exc:
        record.update(status="ERROR", reason=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    record["probe_runtime_s"] = time.time() - started
    atomic_json(receipt, record); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


def _old_preaction(state: dict) -> dict:
    sensor_records = {row["state_id"]: row for row in json.loads(
        (ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json").read_text()
    )["state_records"]}
    with np.load(sensor_records[state["state_id"]]["shard_path"], allow_pickle=False) as loaded:
        command = np.asarray(loaded["action_control"][0, 0, 3:6], np.float32)
    return {
        **state, "role": "fit", "source_kind": "compatible_historical_root",
        "current_applied_command": command.tolist(),
        "current_command_magnitude": float(np.linalg.norm(command[[0, 2]])),
        "motion_class": "turning" if abs(float(command[2])) >= 0.10 else "straight",
        "current_scene_object_point_clearance_m": point_clearance(Path(state["scene_dir"]), tuple(state["start_pose"][0])),
        "pre_existing_disallowed_contact": False,
    }


def _scan_pool() -> list[dict]:
    ensure_corpus(); receipts = CACHE / "eligibility_receipts"; receipts.mkdir(parents=True, exist_ok=True)
    output = []
    for family in CORE.FAMILIES:
        paths = list((CORPUS / "development_pool" / family).iterdir())
        if (RESERVE_CORPUS / "corpus.json").is_file():
            paths.extend((RESERVE_CORPUS / "development_pool" / family).iterdir())
        paths = sorted(paths,
                       key=lambda path: hashlib.sha256(f"{DOMAIN}|{path.name}".encode()).hexdigest())
        accepted = 0
        for start in range(0, len(paths), 4):
            if accepted >= 40:
                break
            jobs = []
            for scene_dir in paths[start:start + 4]:
                receipt = receipts / f"{family}__{scene_dir.name}.json"
                if receipt.is_file():
                    continue
                seed = int(hashlib.sha256(f"{DOMAIN}|{scene_dir.name}".encode()).hexdigest()[:8], 16)
                stream = receipt.with_suffix(".log").open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--probe-scene", str(scene_dir),
                    "--probe-family", family, "--probe-seed", str(seed), "--probe-receipt", str(receipt)],
                    stdout=stream, stderr=subprocess.STDOUT)
                jobs.append((process, stream, receipt))
            for process, stream, receipt in jobs:
                code = process.wait(); stream.close()
                if code and not receipt.is_file():
                    raise RuntimeError(f"probe failed without receipt: {receipt}")
            for scene_dir in paths[start:start + 4]:
                receipt = receipts / f"{family}__{scene_dir.name}.json"
                if not receipt.is_file():
                    continue
                row = json.loads(receipt.read_text())
                if row["status"] == "ERROR":
                    if "cannot import name 'ResetEvent'" in row.get("reason", ""):
                        row.update(status="INELIGIBLE", reason="warmup_fall_or_reset")
                        atomic_json(receipt, row)
                    else:
                        raise RuntimeError(f"probe error {row['scene_id']}: {row['reason']}")
                if row["status"] == "ELIGIBLE":
                    output.append(row); accepted += 1
        if accepted < 34:
            raise RuntimeError(f"{family}: only {accepted} eligible new scenes")
    return output


def _band(value: float, boundaries: tuple[float, float]) -> str:
    return "low" if value <= boundaries[0] else "medium" if value <= boundaries[1] else "high"


def _balanced_select(rows: list[dict], count: int, salt: str) -> list[dict]:
    pool = sorted(rows, key=lambda row: hashlib.sha256(f"{salt}|{row['scene_id']}".encode()).hexdigest())
    selected = []; clearance_counts = {key: 0 for key in ("low", "medium", "high")}
    command_counts = {key: 0 for key in ("low", "high")}; motion_counts = {key: 0 for key in ("straight", "turning")}
    targets = ({key: count / 3 for key in clearance_counts}, {key: count / 2 for key in command_counts}, {key: count / 2 for key in motion_counts})
    while len(selected) < count:
        def score(row: dict) -> tuple:
            gain = max(0.0, targets[0][row["clearance_band"]] - clearance_counts[row["clearance_band"]])
            gain += max(0.0, targets[1][row["command_band"]] - command_counts[row["command_band"]])
            gain += max(0.0, targets[2][row["motion_class"]] - motion_counts[row["motion_class"]])
            return (-gain, hashlib.sha256(f"{salt}|pick|{row['scene_id']}".encode()).hexdigest())
        choice = min(pool, key=score); pool.remove(choice); selected.append(choice)
        clearance_counts[choice["clearance_band"]] += 1; command_counts[choice["command_band"]] += 1
        motion_counts[choice["motion_class"]] += 1
    return selected


def freeze_manifest() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.is_file():
        value = json.loads(MANIFEST.read_text())
        expected = CORE.digest({key: item for key, item in value.items() if key != "content_digest"})
        if value.get("content_digest") != expected:
            raise RuntimeError("panel manifest digest mismatch")
        return value
    excluded = excluded_scenes(); old_states = [_old_preaction(row) for row in json.loads(OLD_PANEL.read_text())["states"]]
    pool = _scan_pool(); new_fit = []
    for family_index, family in enumerate(CORE.FAMILIES):
        candidates = sorted([row for row in pool if row["family"] == family],
                            key=lambda row: hashlib.sha256(f"{DOMAIN}|fit|{row['scene_id']}".encode()).hexdigest())
        for offset, row in enumerate(candidates[:NEW_FIT_PER_FAMILY]):
            new_fit.append({**row, "role": "fit", "source_kind": "new_fit_root",
                            "state_id": f"viability-fit-{family_index}-{offset:02d}"})
    fit = old_states + new_fit
    clearances = np.asarray([row["current_scene_object_point_clearance_m"] for row in fit], np.float64)
    clearance_thresholds = tuple(float(value) for value in np.quantile(clearances, [1 / 3, 2 / 3]))
    magnitudes = np.asarray([row["current_command_magnitude"] for row in fit], np.float64)
    command_threshold = float(np.median(magnitudes))
    used = {row["scene_id"] for row in new_fit}; evaluation = []
    for family_index, family in enumerate(CORE.FAMILIES):
        candidates = []
        for row in pool:
            if row["family"] != family or row["scene_id"] in used:
                continue
            candidates.append({**row,
                "clearance_band": _band(row["current_scene_object_point_clearance_m"], clearance_thresholds),
                "command_band": "low" if row["current_command_magnitude"] <= command_threshold else "high"})
        calibration = _balanced_select(candidates, CAL_PER_FAMILY, f"{DOMAIN}|cal|{family}")
        cal_ids = {row["scene_id"] for row in calibration}
        heldout = _balanced_select([row for row in candidates if row["scene_id"] not in cal_ids], HELD_PER_FAMILY,
                                   f"{DOMAIN}|held|{family}")
        for split, rows in (("calibration", calibration), ("heldout", heldout)):
            for offset, row in enumerate(rows):
                evaluation.append({**row, "role": split, "source_kind": "fresh_evaluation_root",
                                   "state_id": f"viability-{'cal' if split == 'calibration' else 'held'}-{family_index}-{offset:02d}"})
    states = fit + evaluation; scenes = [row["scene_id"] for row in states]
    new_scenes = {row["scene_id"] for row in states if row["source_kind"] != "compatible_historical_root"}
    value = {
        "schema": "lightweight_one_tick_viability_panel_manifest_v1", "source_commit": CORE.SOURCE_COMMIT,
        "domain": DOMAIN, "frozen_before_candidate_execution": True, "states": states,
        "fit_state_count": 128, "fit_scene_count": len({row["scene_id"] for row in fit}),
        "split_state_count": {role: sum(row["role"] == role for row in states) for role in ("fit", "calibration", "heldout")},
        "family_role_counts": {family: {role: sum(row["family"] == family and row["role"] == role for row in states)
            for role in ("fit", "calibration", "heldout")} for family in CORE.FAMILIES},
        "preaction_stratification": {
            "clearance_measure": "exact analytic base-point distance to frozen scene collision boxes; not a Genesis articulated positive-distance claim",
            "clearance_quantile_thresholds_m": list(clearance_thresholds), "command_magnitude_median": command_threshold,
            "motion_rule": "turning iff abs(current applied yaw rate) >= 0.10 rad/s",
        },
        "disjointness": {
            "distinct_state_ids": len({row["state_id"] for row in states}), "distinct_scenes": len(set(scenes)),
            "fresh_evaluation_overlap_fit": len({row["scene_id"] for row in evaluation} & {row["scene_id"] for row in fit}),
            "new_corpus_overlap_historical_or_predictor": len(new_scenes & excluded["union"]),
        },
        "bindings": {"old_panel_sha256": sha(OLD_PANEL), "route_controller_sha256": sha(AUG.ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"),
                     "lateral_controller_sha256": sha(AUG.CHECKPOINT), "new_corpus_sha256": sha(CORPUS / "corpus.json")},
    }
    if value["split_state_count"] != {"fit": 128, "calibration": 24, "heldout": 24}:
        raise RuntimeError("panel cardinality mismatch")
    if value["disjointness"] != {"distinct_state_ids": 176, "distinct_scenes": 176,
                                 "fresh_evaluation_overlap_fit": 0, "new_corpus_overlap_historical_or_predictor": 0}:
        raise RuntimeError(f"panel disjointness failure: {value['disjointness']}")
    value["content_digest"] = CORE.digest(value); atomic_json(MANIFEST, value)
    return value


def _pose7(ctx, snapshot) -> np.ndarray:
    ONE._restore_tick_boundary(ctx, snapshot); robot = ctx.build.robot; runner = ctx.runner
    position = np.asarray(runner._as_np(robot.get_pos()), np.float64).reshape(-1, 3)[0]
    quaternion = np.asarray(runner._as_np(robot.get_quat()), np.float64).reshape(-1, 4)[0]
    return np.concatenate((position, quaternion))


def planning_input(ctx, snapshots: dict[int, object], state: dict) -> dict[str, np.ndarray]:
    embodied = []; previous_velocity = None
    for depth in (4, 3, 2, 1, 0):
        ONE._restore_tick_boundary(ctx, snapshots[depth]); value, velocity = SENSOR.sensor_state(ctx.runner, previous_joint_velocity=previous_velocity)
        previous_velocity = velocity; command = np.asarray(ctx.runner._last_executed, np.float32)[0]
        controller = np.asarray([1, 0, 1, 0, min(1.0, (5 - depth) / 5)], np.float32)
        embodied.append(np.concatenate((value, command, controller)))
    center, half, yaw, _kinds = GEOMETRY.scene_boxes(state)
    manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    jitter = manifest.get("camera_extrinsic_jitter", {}); depth_rows = []; depth_valid = []; lidar_rows = []; lidar_valid = []
    for predecessor_depth in (2, 1, 0):
        values = GEOMETRY.render_geometry(_pose7(ctx, snapshots[predecessor_depth]), center, half, yaw, jitter)
        depth_rows.append(values[0]); depth_valid.append(values[1]); lidar_rows.append(values[2]); lidar_valid.append(values[3])
    return {"depth": np.asarray(depth_rows, np.float16), "depth_valid": np.asarray(depth_valid, np.uint8),
            "lidar": np.asarray(lidar_rows, np.float16), "lidar_valid": np.asarray(lidar_valid, np.uint8),
            "embodied": np.asarray(embodied, np.float32)}


def _requested(index: int) -> list[float]:
    if index >= 12:
        return list(AUG.LATERAL_ACTIONS[index - 12][2])
    primitive = V.V1.CANDIDATE_BANK[index][1][0]
    return np.asarray(V.V1.block_for(primitive), np.float32)[0].tolist()


def normalize_rows(ctx, snapshot, state: dict, rows: list[dict]) -> list[dict]:
    route = MULTI._route_contract(ctx, snapshot); start = route["pose"]; start_distance = math.hypot(
        route["waypoint_xy"][0] - start[0][0], route["waypoint_xy"][1] - start[0][1])
    output = []
    for index, source in enumerate(rows):
        contact = bool(source.get("first_tick_contact", source.get("contact", False)))
        n_safe = None if contact else int(source.get("successor_safe_action_count", 0))
        if index < 12:
            plan = MULTI._h3_plan(ctx, snapshot, index, route); h3_progress = float(plan["h3_progress_m"]); h3_heading = float(plan["h3_heading_improvement_rad"])
        else:
            h3_progress = h3_heading = None
        endpoint = source.get("endpoint_pose"); immediate = 0.0
        if endpoint is not None:
            end_distance = math.hypot(route["waypoint_xy"][0] - float(endpoint[0][0]), route["waypoint_xy"][1] - float(endpoint[0][1]))
            immediate = float(start_distance - end_distance)
        applied = list(map(float, source["target_command"])); requested = _requested(index)
        output.append({
            "action_index": index, "candidate": str(source["candidate"]), "controller": "route" if index < 12 else "lateral",
            "requested_action": requested, "applied_first_tick_action": applied,
            "transition_required": bool(index >= 12), "contact": contact, "n_safe": n_safe,
            "n_safe_ge_1": bool(n_safe is not None and n_safe >= 1), "n_safe_ge_2": bool(n_safe is not None and n_safe >= 2),
            "n_safe_ge_3": bool(n_safe is not None and n_safe >= 3), "clipped_n_safe": None if n_safe is None else min(n_safe, 4),
            "nonviable": bool(n_safe == 0) if n_safe is not None else None, "h3_progress_m": h3_progress,
            "h3_heading_improvement_rad": h3_heading, "immediate_progress_m": immediate,
            "decision_progress_m": h3_progress if index < 12 else immediate,
            "first_contact_step": source.get("first_contact_step"),
        })
    return output


def collect_state(index: int) -> dict:
    manifest = freeze_manifest(); state = manifest["states"][index]; state_id = state["state_id"]
    path = OUT / "states" / f"{state_id}.json"
    if path.is_file():
        record = json.loads(path.read_text()); shard = Path(record["shard_path"])
        if record.get("status") == "PASS" and shard.is_file() and sha(shard) == record["shard_sha256"]:
            return record
    started = time.time(); ctx, snapshots, reconstruction = MULTI.historical_snapshots(state); snapshot = snapshots[0]
    route_policy = ctx.policy; lateral_policy = AUG.lateral_policy(); generated = 0
    if state["source_kind"] == "compatible_historical_root":
        previous = json.loads((OLD_LATERAL_STATES / f"{state_id}.json").read_text()); source_rows = previous["actions"]
        lineage = {"oracle_tree": "reused", "source_sha256": sha(OLD_LATERAL_STATES / f"{state_id}.json")}
    else:
        tree, _successors = AUG.augmented_tree(ctx, snapshot, route_policy, lateral_policy, f"{state_id}:oracle")
        source_rows = tree["candidates"]; generated = int(tree["current_prefix_branches"] + tree["successor_branches"])
        lineage = {"oracle_tree": "new", "tree_identity": tree["identity"]}
    candidates = normalize_rows(ctx, snapshot, state, source_rows); inputs = planning_input(ctx, snapshots, state)
    candidate_array = []
    for row in candidates:
        controller = [1.0, 0.0] if row["controller"] == "route" else [0.0, 1.0]
        candidate_array.append(row["requested_action"] + row["applied_first_tick_action"] + controller + [float(row["transition_required"])])
    arrays = {**inputs, "candidate": np.asarray(candidate_array, np.float32),
        "contact": np.asarray([row["contact"] for row in candidates], np.uint8),
        "n_safe": np.asarray([-1 if row["n_safe"] is None else row["n_safe"] for row in candidates], np.int8),
        "target_valid": np.asarray([row["n_safe"] is not None for row in candidates], np.uint8)}
    shard = CACHE / "states" / f"{state_id}.npz"; atomic_npz(shard, **arrays)
    record = {"schema": "lightweight_one_tick_viability_state_v1", "status": "PASS", "state_index": index,
        "state_id": state_id, "scene_id": state["scene_id"], "family": state["family"], "role": state["role"],
        "source_kind": state["source_kind"], "snapshot_digest": snapshot.digest, "predecessor_reconstruction": reconstruction,
        "input_contract": {"depth_history": 3, "lidar_history": 3, "embodied_history": 5, "future_sensor_inputs": 0,
                           "active_controller": "route", "previous_controller": "route"},
        "candidates": candidates, "shard_path": str(shard), "shard_sha256": sha(shard),
        "shapes": {key: list(value.shape) for key, value in arrays.items()}, "oracle_lineage": lineage,
        "generated_branches": generated, "runtime_s": time.time() - started}
    record["content_digest"] = CORE.digest(record); atomic_json(path, record); del ctx; gc.collect()
    print(json.dumps({"state_id": state_id, "role": state["role"], "branches": generated, "runtime_s": record["runtime_s"]}), flush=True)
    return record


def prevalence(records: list[dict]) -> dict:
    candidates = [row for state in records for row in state["candidates"]]
    contact_positive = sum(row["contact"] for row in candidates); contact_negative = len(candidates) - contact_positive
    valid = [row for row in candidates if not row["contact"]]; nonviable = sum(row["n_safe"] == 0 for row in valid)
    return {"states": len(records), "candidate_rows": len(candidates), "contact_positive": contact_positive,
        "contact_negative": contact_negative, "contact_prevalence": contact_positive / max(1, len(candidates)),
        "contact_free_successors": len(valid), "nonviable_successors": nonviable,
        "nonviability_prevalence_contact_free": nonviable / max(1, len(valid)),
        "safe_count_distribution": {str(value): sum(row["n_safe"] == value for row in valid) for value in range(15)},
        "states_with_one_viability_admissible": sum(any(not row["contact"] and row["n_safe"] >= 1 for row in state["candidates"]) for state in records),
        "states_with_two_viability_admissible": sum(sum(not row["contact"] and row["n_safe"] >= 1 for row in state["candidates"]) >= 2 for state in records)}


def finalize() -> dict:
    manifest = freeze_manifest(); records = []
    for state in manifest["states"]:
        path = OUT / "states" / f"{state['state_id']}.json"
        record = json.loads(path.read_text())
        if record["status"] != "PASS" or sha(Path(record["shard_path"])) != record["shard_sha256"]:
            raise RuntimeError(f"bad state evidence {state['state_id']}")
        records.append(record)
    roles = {role: [row for row in records if row["role"] == role] for role in ("fit", "calibration", "heldout")}
    inventory = {role: prevalence(rows) for role, rows in roles.items()}
    per_family = {role: {family: prevalence([row for row in rows if row["family"] == family]) for family in CORE.FAMILIES}
                  for role, rows in roles.items()}
    adequacy = {
        "both_contact_classes_each_split": all(inventory[role]["contact_positive"] > 0 and inventory[role]["contact_negative"] > 0 for role in ("calibration", "heldout")),
        "both_viable_nonviable_each_split": all(inventory[role]["nonviable_successors"] > 0 and inventory[role]["contact_free_successors"] > inventory[role]["nonviable_successors"] for role in ("calibration", "heldout")),
        "at_least_22_oracle_viable_each_split": all(inventory[role]["states_with_one_viability_admissible"] >= 22 for role in ("calibration", "heldout")),
        "every_family_contact_and_nonviability": all(per_family[role][family]["contact_positive"] > 0 and per_family[role][family]["nonviable_successors"] > 0 for role in ("calibration", "heldout") for family in CORE.FAMILIES),
    }
    wall = json.loads((OUT / "collection_receipt.json").read_text())
    result = {"schema": "lightweight_one_tick_viability_oracle_tree_index_v1", "manifest_digest": manifest["content_digest"],
        "records": records, "inventory": inventory, "per_family_inventory": per_family, "panel_adequacy": adequacy,
        "panel_adequate": all(adequacy.values()), "classification": None if all(adequacy.values()) else "FRESH_MICRO_VIABILITY_PANEL_INADEQUATE",
        "generated_branches": sum(row["generated_branches"] for row in records), "runtime": wall,
        "storage_bytes": sum(Path(row["shard_path"]).stat().st_size for row in records)}
    result["content_digest"] = CORE.digest(result); atomic_json(INDEX, result)
    print(json.dumps({key: result[key] for key in ("inventory", "panel_adequacy", "panel_adequate", "generated_branches", "runtime", "storage_bytes")}, indent=2))
    return result


def collect_all() -> None:
    manifest = freeze_manifest(); started = time.time(); logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(manifest["states"]), 4):
        jobs = []
        for index in range(start, min(start + 4, len(manifest["states"]))):
            state_id = manifest["states"][index]["state_id"]; stream = (logs / f"state_{index:03d}_{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--collect-state", str(index)],
                                       stdout=stream, stderr=subprocess.STDOUT)
            jobs.append((index, state_id, process, stream))
        for index, state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code:
                raise RuntimeError(f"state collection failed {index} {state_id}")
    atomic_json(OUT / "collection_receipt.json", {"states": len(manifest["states"]), "parallel_processes": 4,
        "wall_runtime_s": time.time() - started})


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate-corpus", action="store_true"); group.add_argument("--freeze", action="store_true")
    group.add_argument("--probe-scene", type=Path); group.add_argument("--collect-state", type=int)
    group.add_argument("--collect-all", action="store_true"); group.add_argument("--finalize", action="store_true")
    parser.add_argument("--probe-family"); parser.add_argument("--probe-seed", type=int); parser.add_argument("--probe-receipt", type=Path)
    args = parser.parse_args()
    if args.probe_scene is not None:
        probe_scene(args.probe_scene, args.probe_family, args.probe_seed, args.probe_receipt)
    elif args.generate_corpus:
        print(json.dumps(ensure_corpus(), indent=2))
    elif args.freeze:
        print(json.dumps(freeze_manifest(), indent=2))
    elif args.collect_state is not None:
        collect_state(args.collect_state); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    elif args.collect_all:
        collect_all()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
