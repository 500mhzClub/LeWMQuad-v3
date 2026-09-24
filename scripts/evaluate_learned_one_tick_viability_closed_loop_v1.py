#!/usr/bin/env python3
"""Conditional eight-start learned one-tick viability closed-loop evaluation."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import benchmark_lightweight_one_tick_viability_interface_v1 as BENCH
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import train_evaluate_lightweight_one_tick_viability_model_v1 as TRAIN
from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep


OUT = COLLECT.OUT
SELECTION = OUT / "closed_loop_selection.json"
RESULT = OUT / "closed_loop_result.json"
MAX_CYCLES = 30


def freeze_selection() -> dict:
    if SELECTION.is_file():
        value = json.loads(SELECTION.read_text())
        if value["content_digest"] != CORE.digest({key: item for key, item in value.items() if key != "content_digest"}):
            raise RuntimeError("closed-loop selection digest mismatch")
        return value
    used = {row["scene_id"] for row in json.loads(COLLECT.MANIFEST.read_text())["states"]}
    receipts = [json.loads(path.read_text()) for path in sorted((COLLECT.CACHE / "eligibility_receipts").glob("*.json"))]
    selected = []
    for family_index, family in enumerate(CORE.FAMILIES):
        candidates = [row for row in receipts if row.get("status") == "ELIGIBLE" and row["family"] == family and
                      row["scene_id"] not in used and abs(float(row["route_heading_world_rad"]) - float(row["start_pose"][1])) >= 0.15]
        candidates.sort(key=lambda row: hashlib.sha256(f"{COLLECT.DOMAIN}|closed|{row['scene_id']}".encode()).hexdigest())
        if len(candidates) < 2:
            raise RuntimeError(f"{family}: fewer than two unused turn-required starts")
        for offset, row in enumerate(candidates[:2]):
            selected.append({**row, "role": "closed_loop", "source_kind": "fresh_closed_loop_root",
                             "state_id": f"viability-closed-{family_index}-{offset:02d}"})
    value = {"schema": "learned_one_tick_viability_closed_loop_selection_v1", "frozen_before_closed_loop_execution": True,
        "states": selected, "state_count": 8, "family_counts": {family: sum(row["family"] == family for row in selected) for family in CORE.FAMILIES},
        "requirements": {"pre_existing_contact": False, "oracle_local_waypoint": True, "turn_required_min_abs_heading_error_rad": 0.15,
                         "final_beacon_omitted": True}, "overlap_fit_calibration_heldout": len({row["scene_id"] for row in selected} & used)}
    if value["overlap_fit_calibration_heldout"] or any(count != 2 for count in value["family_counts"].values()):
        raise RuntimeError("closed-loop selection contract failure")
    value["content_digest"] = CORE.digest(value); COLLECT.atomic_json(SELECTION, value); return value


def candidate_rows(ctx, snapshot, active_controller: str) -> list[dict]:
    route = MULTI._route_contract(ctx, snapshot); rows = []
    for index in range(14):
        if index < 12:
            ONE._restore_tick_boundary(ctx, snapshot); primitive = AUG.V.V1.CANDIDATE_BANK[index][1][0]
            target = np.asarray(ctx.runner._clip_block(AUG.V.V1.block_for(primitive)[None]).executed, np.float32)[0, 0]
            plan = MULTI._h3_plan(ctx, snapshot, index, route)
            rows.append({"action_index": index, "candidate": AUG.V.V1.CANDIDATE_BANK[index][0], "controller": "route",
                         "requested_action": COLLECT._requested(index), "applied_first_tick_action": target.tolist(),
                         "transition_required": active_controller == "lateral",
                         "h3_progress_m": plan["h3_progress_m"], "h3_heading_improvement_rad": plan["h3_heading_improvement_rad"],
                         "decision_progress_m": plan["h3_progress_m"]})
        else:
            name = AUG.LATERAL_ACTIONS[index - 12][1]; command = list(AUG.LATERAL_ACTIONS[index - 12][2])
            rows.append({"action_index": index, "candidate": name, "controller": "lateral", "requested_action": command,
                         "applied_first_tick_action": command, "transition_required": active_controller == "route",
                         "h3_progress_m": None, "h3_heading_improvement_rad": None, "decision_progress_m": 0.0})
    return rows


def state_input(ctx, snapshot, history: dict, state: dict, active: str, previous: str, ticks_since: int) -> tuple[dict, np.ndarray]:
    ONE._restore_tick_boundary(ctx, snapshot); value, velocity = COLLECT.SENSOR.sensor_state(ctx.runner, previous_joint_velocity=history["joint_velocity"])
    command = np.asarray(ctx.runner._last_executed, np.float32)[0]
    controller = np.asarray([active == "route", active == "lateral", previous == "route", previous == "lateral", min(ticks_since, 10) / 10], np.float32)
    embodied = np.concatenate((value, command, controller)).astype(np.float32)
    pose = COLLECT._pose7(ctx, snapshot); geometry = COLLECT.GEOMETRY.render_geometry(pose, history["center"], history["half"], history["yaw"], history["jitter"])
    history["depth"].append(geometry[0].astype(np.float16)); history["depth_valid"].append(geometry[1].astype(np.uint8))
    history["lidar"].append(geometry[2].astype(np.float16)); history["lidar_valid"].append(geometry[3].astype(np.uint8)); history["embodied"].append(embodied)
    for key in ("depth", "depth_valid", "lidar", "lidar_valid"):
        history[key] = history[key][-3:]
    history["embodied"] = history["embodied"][-5:]; history["joint_velocity"] = velocity
    return history, velocity


def initial_history(ctx, snapshots: dict[int, object], state: dict) -> dict:
    values = COLLECT.planning_input(ctx, snapshots, state); center, half, yaw, _ = COLLECT.GEOMETRY.scene_boxes(state)
    manifest = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
    ONE._restore_tick_boundary(ctx, snapshots[0]); _value, velocity = COLLECT.SENSOR.sensor_state(ctx.runner, previous_joint_velocity=None)
    return {"depth": list(values["depth"]), "depth_valid": list(values["depth_valid"]), "lidar": list(values["lidar"]),
            "lidar_valid": list(values["lidar_valid"]), "embodied": list(values["embodied"]), "joint_velocity": velocity,
            "center": center, "half": half, "yaw": yaw, "jitter": manifest.get("camera_extrinsic_jitter", {})}


def model_choice(model, package: dict, offline: dict, rows: list[dict], history: dict, device: torch.device) -> tuple[int | None, dict]:
    raw = {key: np.asarray(history[key]) for key in ("depth", "depth_valid", "lidar", "lidar_valid", "embodied")}
    candidate = []
    for row in rows:
        identity = [1, 0] if row["controller"] == "route" else [0, 1]
        candidate.append(row["requested_action"] + row["applied_first_tick_action"] + identity + [float(row["transition_required"])])
    raw["candidate"] = np.asarray(candidate, np.float32); tensors = BENCH.preprocess(raw, package["statistics"], device)
    with torch.inference_mode(): logits = model(*tensors)[0].float().cpu().numpy()
    temperatures = offline["calibration"]["temperatures"]; thresholds = offline["calibration"]["selected"]
    cp = 1 / (1 + np.exp(-logits[:, 0] / temperatures["contact"])); npv = 1 / (1 + np.exp(-logits[:, 1] / temperatures["nonviability"]))
    count = np.clip(logits[:, 5], 0, 4); admitted = (cp < thresholds["contact_threshold"]) & (npv < thresholds["nonviability_threshold"])
    return CORE.select_candidate(rows, admitted, count), {"contact_probability": cp.tolist(), "nonviability_probability": npv.tolist(),
                                                             "predicted_count": count.tolist(), "admitted": admitted.tolist()}


def execute(ctx, snapshot, route_policy, lateral_policy, choice: int, identity: str):
    topology, links, objects = AUG.topology(ctx)
    if choice < 12:
        record, contact, successor = AUG.execute_route(ctx, snapshot, route_policy, choice, topology, links, objects)
    else:
        _index, name, command = AUG.LATERAL_ACTIONS[choice - 12]
        record, contact, successor = AUG.execute_lateral(ctx, snapshot, lateral_policy, choice, name, command, topology, links, objects, identity)
    next_rows = AUG.successor_outcomes(ctx, successor, route_policy, lateral_policy, topology, links, objects, f"{identity}:successor")
    return record, bool(contact.any()), successor, sum(not row["contact"] for row in next_rows)


def rollout(state: dict, condition: str) -> dict:
    started = time.time(); ctx, snapshots, _ = MULTI.historical_snapshots(state); snapshot = snapshots[0]
    route_policy = ctx.policy; lateral_policy = AUG.lateral_policy(); history = initial_history(ctx, snapshots, state)
    package = torch.load(TRAIN.CHECKPOINT, map_location="cpu", weights_only=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CORE.LightweightOneTickViabilityModel().to(device); model.load_state_dict(package["state_dict"]); model.eval()
    offline = json.loads(TRAIN.RESULT.read_text()); active = previous = "route"; ticks_since = 10; cycles = []
    start_route = MULTI._route_contract(ctx, snapshot); start_distance = math.hypot(*start_route["waypoint_body_xy"])
    previous_heading_error = abs(MULTI.KINEMATIC.wrap(float(start_route["route_heading_world_rad"]) - float(start_route["pose"][1])))
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    for cycle in range(MAX_CYCLES):
        rows = candidate_rows(ctx, snapshot, active); prediction = None; oracle_tree = None
        if condition == "oracle":
            oracle_tree, successors = AUG.augmented_tree(ctx, snapshot, route_policy, lateral_policy, f"{state['state_id']}:{condition}:{cycle}")
            choice_row = AUG.choose(oracle_tree); choice = None if choice_row is None else int(choice_row["action_index"])
        elif condition == "learned":
            choice, prediction = model_choice(model, package, offline, rows, history, device)
        else:
            choice = int(rows[CORE.route_order(rows[:12])[0]]["action_index"])
        if choice is None:
            cycles.append({"cycle": cycle, "abstained": True}); break
        if condition == "oracle":
            record = oracle_tree["candidates"][choice]; contact = bool(record["first_tick_contact"]); successor = successors[choice]
            n_safe = int(record["successor_safe_action_count"])
        else:
            record, contact, successor, n_safe = execute(ctx, snapshot, route_policy, lateral_policy, choice,
                                                         f"{state['state_id']}:{condition}:{cycle}:{choice}")
        next_controller = "lateral" if choice >= 12 else "route"; transition = next_controller != active
        previous, active = active, next_controller; ticks_since = 0 if transition else ticks_since + 1
        snapshot = successor; history, _velocity = state_input(ctx, snapshot, history, state, active, previous, ticks_since)
        route_now = MULTI._route_contract(ctx, snapshot); distance = math.hypot(*route_now["waypoint_body_xy"])
        progress = start_distance - distance - sum(float(row.get("progress_m", 0)) for row in cycles)
        heading_error = abs(MULTI.KINEMATIC.wrap(float(route_now["route_heading_world_rad"]) - float(route_now["pose"][1])))
        heading_improvement = previous_heading_error - heading_error; previous_heading_error = heading_error
        ONE._restore_tick_boundary(ctx, snapshot); pose = ctx.pose(); episode = ctx.runner.episode_states[0]
        stuck = bool(label_computer.step(PoseStep(timestamp_ns=int(ctx.runner._sim_time_ns), env_idx=0,
            episode_id=int(episode.episode_id), episode_step=int(episode.episode_step),
            position_xy_world=(float(pose[0][0]), float(pose[0][1])), yaw_world_rad=float(pose[1]),
            last_command=tuple(float(value) for value in record["target_command"]))).stuck_label)
        complete = distance <= 0.35
        unsafe_termination = bool(any(record.get("termination", {}).values()))
        cycles.append({"cycle": cycle, "abstained": False, "selected_action": choice, "controller": active,
            "transition": transition, "contact": contact, "successor_n_safe": n_safe, "nonviable_successor": n_safe == 0,
            "progress_m": progress, "heading_improvement_rad": heading_improvement, "negative_progress": progress < 0,
            "stuck": stuck, "unsafe_termination": unsafe_termination, "waypoint_completed": complete, "prediction": prediction})
        if contact or n_safe == 0 or complete or unsafe_termination: break
    result = {"schema": "learned_one_tick_viability_closed_loop_rollout_v1", "state_id": state["state_id"], "scene_id": state["scene_id"],
        "family": state["family"], "condition": condition, "cycles": cycles,
        "executed_cycles": sum(not row["abstained"] for row in cycles), "contacts": sum(row.get("contact", False) for row in cycles),
        "nonviable_successors": sum(row.get("nonviable_successor", False) for row in cycles),
        "lateral_recoveries": sum(row.get("selected_action") in (12, 13) for row in cycles),
        "route_actions": sum(row.get("selected_action") is not None and row.get("selected_action", 99) < 12 for row in cycles if not row["abstained"]),
        "controller_transitions": sum(row.get("transition", False) for row in cycles), "abstentions": sum(row["abstained"] for row in cycles),
        "distance_progress_m": sum(float(row.get("progress_m", 0)) for row in cycles),
        "heading_improvement_rad": sum(float(row.get("heading_improvement_rad", 0)) for row in cycles),
        "negative_progress_cycles": sum(row.get("negative_progress", False) for row in cycles),
        "stuck_cycles": sum(row.get("stuck", False) for row in cycles),
        "unsafe_terminations": sum(row.get("unsafe_termination", False) for row in cycles),
        "waypoint_completed": any(row.get("waypoint_completed", False) for row in cycles), "runtime_s": time.time() - started}
    result["content_digest"] = CORE.digest(result); path = OUT / "closed_loop" / f"{state['state_id']}__{condition}.json"
    COLLECT.atomic_json(path, result); del ctx; gc.collect(); print(json.dumps({"state": state["state_id"], "condition": condition,
        "cycles": result["executed_cycles"], "contacts": result["contacts"], "nonviable": result["nonviable_successors"]}), flush=True)
    return result


def collect_all() -> None:
    selection = freeze_selection(); started = time.time(); logs = COLLECT.CACHE / "closed_loop_logs"; logs.mkdir(parents=True, exist_ok=True)
    jobspec = [(state["state_id"], condition) for state in selection["states"] for condition in ("oracle", "learned", "route_only")]
    for start in range(0, len(jobspec), 4):
        jobs = []
        for state_id, condition in jobspec[start:start + 4]:
            stream = (logs / f"{state_id}__{condition}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--state", state_id, "--condition", condition],
                                       stdout=stream, stderr=subprocess.STDOUT); jobs.append((state_id, condition, process, stream))
        for state_id, condition, process, stream in jobs:
            code = process.wait(); stream.close()
            if code: raise RuntimeError(f"closed-loop failure {state_id} {condition}")
    COLLECT.atomic_json(OUT / "closed_loop_collection_receipt.json", {"rollouts": len(jobspec), "parallel_processes": 4,
        "wall_runtime_s": time.time() - started})


def finalize() -> dict:
    selection = freeze_selection(); records = []
    for state in selection["states"]:
        for condition in ("oracle", "learned", "route_only"):
            records.append(json.loads((OUT / "closed_loop" / f"{state['state_id']}__{condition}.json").read_text()))
    groups = {condition: [row for row in records if row["condition"] == condition] for condition in ("oracle", "learned", "route_only")}
    def aggregate(rows):
        return {"starts": len(rows), "contacts": sum(row["contacts"] for row in rows), "nonviable_successors": sum(row["nonviable_successors"] for row in rows),
            "lateral_recoveries": sum(row["lateral_recoveries"] for row in rows), "route_actions": sum(row["route_actions"] for row in rows),
            "controller_transitions": sum(row["controller_transitions"] for row in rows), "abstentions": sum(row["abstentions"] for row in rows),
            "distance_progress_m": sum(row["distance_progress_m"] for row in rows),
            "heading_improvement_rad": sum(row["heading_improvement_rad"] for row in rows),
            "negative_progress_cycles": sum(row["negative_progress_cycles"] for row in rows), "stuck_cycles": sum(row["stuck_cycles"] for row in rows),
            "unsafe_terminations": sum(row["unsafe_terminations"] for row in rows),
            "waypoint_completions": sum(row["waypoint_completed"] for row in rows)}
    summary = {condition: aggregate(rows) for condition, rows in groups.items()}
    learned = summary["learned"]; oracle = summary["oracle"]; baseline = summary["route_only"]
    family = {name: {condition: aggregate([row for row in groups[condition] if row["family"] == name]) for condition in groups} for name in CORE.FAMILIES}
    resumed = all(not any(cycle.get("selected_action") in (12, 13) for cycle in row["cycles"]) or
                  any(cycle.get("selected_action", 99) < 12 for cycle in row["cycles"] if not cycle["abstained"])
                  for row in groups["learned"])
    gate = {"zero_contacts": learned["contacts"] == 0, "zero_nonviable_successors": learned["nonviable_successors"] == 0,
        "oracle_progress_80pct": learned["distance_progress_m"] / max(abs(oracle["distance_progress_m"]), 1e-9) >= 0.80,
        "additional_abstentions_at_most_one": learned["abstentions"] - oracle["abstentions"] <= 1,
        "route_progress_resumes_after_lateral": resumed,
        "outperforms_route_only_contact_nonviability": (learned["contacts"] + learned["nonviable_successors"]) < (baseline["contacts"] + baseline["nonviable_successors"]),
        "no_family_collapse": all(family[name]["learned"]["route_actions"] > 0 for name in CORE.FAMILIES)}
    result = {"schema": "learned_one_tick_viability_closed_loop_result_v1", "selection": selection,
        "summary": summary, "per_family": family, "gate": gate, "pass": all(gate.values()),
        "classification": "LEARNED_ONE_TICK_VIABILITY_CONTROL_SIGNAL" if all(gate.values()) else "LEARNED_ONE_TICK_VIABILITY_CONTROL_NO_SIGNAL",
        "records": records, "runtime": json.loads((OUT / "closed_loop_collection_receipt.json").read_text())}
    result["content_digest"] = CORE.digest(result); COLLECT.atomic_json(RESULT, result)
    print(json.dumps({key: result[key] for key in ("summary", "per_family", "gate", "pass", "classification", "runtime")}, indent=2)); return result


def main() -> int:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true"); group.add_argument("--collect-all", action="store_true")
    group.add_argument("--state"); group.add_argument("--finalize", action="store_true"); parser.add_argument("--condition")
    args = parser.parse_args()
    if args.freeze: print(json.dumps(freeze_selection(), indent=2))
    elif args.collect_all: collect_all()
    elif args.state:
        state = next(row for row in freeze_selection()["states"] if row["state_id"] == args.state); rollout(state, args.condition)
    else: finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
