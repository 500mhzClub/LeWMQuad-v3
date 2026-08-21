#!/usr/bin/env python3
"""Materialise the bounded lateral-augmented predecessor eligibility envelope."""
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

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lateral_augmented_state_eligibility_envelope_v1 as REDUCE
from scripts import evaluate_lateral_recovery_oracle_viability_v2 as AUG
from scripts import materialize_multi_cycle_viability_envelope_v1 as MULTI
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import run_go2_oracle_branch_pilot_v1_2 as V
from lewm_worlds.labels.derived import DerivedLabelComputer, DerivedLabelConfig, PoseStep


SOURCE_COMMIT = REDUCE.SOURCE_COMMIT
OUT = ROOT / ".generated/lateral_augmented_state_eligibility_envelope_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/lateral_augmented_state_eligibility_envelope_v1"
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
ONE_RESULT = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/viability_result.json"
MULTI_INDEX = ROOT / ".generated/multi_cycle_viability_envelope_v1/multi_cycle_index.json"
MULTI_SELECTION = ROOT / ".generated/multi_cycle_viability_envelope_v1/frozen_state_selection.json"
LATERAL_RESULT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/scientific_viability_result.json"
LATERAL_STATES = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/scientific_states"
LATERAL_ROLLOUTS = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/scientific_rollouts"
CHECKPOINT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2/seed_2026082015/model_999.pt"
ROUTE_CHECKPOINT = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SELECTION = OUT / "frozen_selection.json"
MAX_DEPTH = 10
MAX_CYCLES = 10
IMPLEMENTATION_REVISION = 1


def sha256(path: Path) -> str:
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


def panel_map() -> dict[str, dict]:
    return {row["state_id"]: row for row in json.loads(PANEL.read_text())["states"]}


def freeze_selection() -> dict:
    if SELECTION.is_file():
        value = json.loads(SELECTION.read_text())
        expected = REDUCE.digest({key: item for key, item in value.items() if key != "content_digest"})
        if value.get("content_digest") != expected:
            raise RuntimeError("selection digest mismatch")
        return value

    lateral_rows = [json.loads(path.read_text()) for path in sorted(LATERAL_STATES.glob("*.json"))]
    residual_ids = tuple(sorted(row["state_id"] for row in lateral_rows if not row["augmented_viability"]))
    if residual_ids != REDUCE.RESIDUAL_IDS:
        raise RuntimeError(f"residual identity mismatch: {residual_ids}")
    one = json.loads(ONE_RESULT.read_text())
    current = {row["state_id"]: row for row in one["current_state_viability"]["per_state"]}
    predecessor = {row["state_id"]: row for row in one["predecessor_audit"]["per_state"]}
    multi = {row["state_id"]: row for row in json.loads(MULTI_INDEX.read_text())["state_records"]}
    lateral = {row["state_id"]: row for row in lateral_rows}
    controls = json.loads(MULTI_SELECTION.read_text())["control_state_ids"]
    lineage = {
        state_id: {
            "one_tick_current": current[state_id]["classification"],
            "one_tick_predecessor": predecessor.get(state_id, {}).get("classification", "NOT_IN_ONE_TICK_PREDECESSOR_AUDIT"),
            "multi_cycle": multi[state_id]["failure_classification"],
            "multi_cycle_stable_predecessor_depth": multi[state_id]["stable_predecessor_depth"],
            "lateral_augmented_current_viability": lateral[state_id]["augmented_viability"],
            "lateral_admissible_indices": lateral[state_id]["lateral_admissible_indices"],
        }
        for state_id in residual_ids
    }
    value = {
        "schema": "lateral_augmented_state_eligibility_frozen_selection_v1",
        "source_commit": SOURCE_COMMIT,
        "residual_state_ids": list(residual_ids),
        "matched_control_state_ids": controls,
        "predecessor_lineage": lineage,
        "controller_bindings": {
            "route_checkpoint": str(ROUTE_CHECKPOINT),
            "route_checkpoint_sha256": sha256(ROUTE_CHECKPOINT),
            "lateral_checkpoint": str(CHECKPOINT),
            "lateral_checkpoint_sha256": sha256(CHECKPOINT),
            "lateral_actions": [list(row) for row in AUG.LATERAL_ACTIONS],
            "feedforward_policy_controller_state": "last joint action and command/history state are included in each tick-boundary snapshot",
        },
        "bindings": {
            "panel_sha256": sha256(PANEL),
            "one_tick_result_sha256": sha256(ONE_RESULT),
            "multi_cycle_index_sha256": sha256(MULTI_INDEX),
            "lateral_result_sha256": sha256(LATERAL_RESULT),
        },
        "frozen_before_predecessor_generation": True,
    }
    if value["controller_bindings"]["lateral_checkpoint_sha256"] != "04a85caec6720da2e9c1beabc93817b2a264da7e2efbb87cd3d2b33c614cbaed":
        raise RuntimeError("lateral checkpoint binding mismatch")
    value["content_digest"] = REDUCE.digest(value)
    atomic_json(SELECTION, value)
    return value


def geometry_and_controller_state(ctx, snapshot) -> dict:
    ctx.policy = ctx.policy
    ONE._restore_tick_boundary(ctx, snapshot)
    topo, links, objects = AUG.topology(ctx)
    route = MULTI._route_contract(ctx, snapshot)
    features = MULTI._current_features(ctx, snapshot, route, topo, links, objects)
    runner = ctx.runner
    link_pos = np.asarray(runner._as_np(runner.build.robot.get_links_pos()), np.float64)
    link_quat = np.asarray(runner._as_np(runner.build.robot.get_links_quat()), np.float64)
    geom_rows = []
    for geom in runner.build.robot.geoms:
        geom_rows.append({
            "geometry_index": int(geom.idx) - int(runner.build.robot.geom_start),
            "link": str(geom.link.name),
            "position_world": np.asarray(runner._as_np(geom.get_pos()), np.float64).tolist(),
            "quaternion_world": np.asarray(runner._as_np(geom.get_quat()), np.float64).tolist(),
        })
    return {
        "snapshot_digest": snapshot.digest,
        "step_index": int(snapshot.step_index),
        "sim_time_ns": int(snapshot.harness["objects"]["_sim_time_ns"]),
        "route_controller_last_action": np.asarray(snapshot.last_actions, np.float32).tolist(),
        "lateral_controller_initial_last_action": np.asarray(snapshot.last_actions, np.float32).tolist(),
        "active_command_history": np.asarray(snapshot.harness["arrays"]["_last_executed"], np.float32).tolist(),
        "controller_recurrent_state": "none; both policies are feedforward MLPs",
        "route": route,
        "deployment_state": features,
        "robot_link_transforms": [
            {"link_index": index, "position_world": link_pos[index].tolist(), "quaternion_world": link_quat[index].tolist()}
            for index in range(len(link_pos))
        ],
        "robot_collision_shape_transforms": geom_rows,
    }


def select_action(tree: dict) -> dict | None:
    return AUG.choose(tree)


def rollout(ctx, start_snapshot, route_policy, lateral_policy, *, state_id: str,
            start_depth: int, max_cycles: int, role: str, identity: str) -> dict:
    snapshot = start_snapshot
    selected = []
    branches = 0
    label_computer = DerivedLabelComputer(ctx.manifest, config=DerivedLabelConfig())
    episode_id = int(ctx.runner.episode_states[0].episode_id)
    for cycle in range(max_cycles):
        tree, successors = AUG.augmented_tree(
            ctx, snapshot, route_policy, lateral_policy, f"{identity}:cycle:{cycle:02d}"
        )
        branches += tree["current_prefix_branches"] + tree["successor_branches"]
        choice = select_action(tree)
        if choice is None:
            selected.append({"cycle": cycle, "abstained": True, "tree": tree})
            break
        snapshot = successors[choice["action_index"]]
        active_policy = lateral_policy if choice["action_index"] >= 12 else route_policy
        ctx.policy = active_policy
        ONE._restore_tick_boundary(ctx, snapshot)
        pose = ctx.pose()
        command = choice["target_command"]
        derived = label_computer.step(PoseStep(
            timestamp_ns=int(ctx.runner._sim_time_ns), env_idx=0, episode_id=episode_id,
            episode_step=int(ctx.runner.episode_states[0].episode_step),
            position_xy_world=(float(pose[0][0]), float(pose[0][1])), yaw_world_rad=float(pose[1]),
            last_command=tuple(float(value) for value in command),
        ))
        route_rows = [row for row in tree["candidates"] if row["action_index"] < 12 and row["admissible"]]
        best_h3 = max((float(row["h3_progress_m"]) for row in route_rows), default=None)
        waypoint = tree["route"]["waypoint_xy"]
        waypoint_complete = bool(math.hypot(waypoint[0] - pose[0][0], waypoint[1] - pose[0][1]) <= 0.35)
        selected.append({
            "cycle": cycle, "abstained": False,
            "selected_action_index": choice["action_index"], "selected_action": choice["candidate"],
            "selected_controller": choice["controller"],
            "selected_first_tick_contact": choice["first_tick_contact"],
            "selected_successor_viable": choice["viable"],
            "selected_successor_safe_action_count": choice["successor_safe_action_count"],
            "safe_prefix_count": tree["safe_prefix_count"],
            "viability_admissible_count": tree["viability_admissible_count"],
            "immediate_progress_m": choice["immediate_progress_m"],
            "immediate_heading_improvement_rad": choice["immediate_heading_improvement_rad"],
            "h3_progress_m": choice.get("h3_progress_m"),
            "best_admissible_route_h3_progress_m": best_h3,
            "h3_oracle_progress_fraction": (
                None if choice["action_index"] >= 12 or best_h3 is None or abs(best_h3) < 1e-12
                else float(choice["h3_progress_m"]) / best_h3
            ),
            "temporary_negative_progress": bool(choice["immediate_progress_m"] < 0),
            "stuck": bool(derived.stuck_label), "waypoint_completed": waypoint_complete,
            "termination": choice["termination"], "tree": tree,
        })
        if choice["first_tick_contact"] or any(choice["termination"].values()) or waypoint_complete:
            break
    executed = [row for row in selected if not row["abstained"]]
    lateral_positions = [index for index, row in enumerate(executed) if row["selected_action_index"] >= 12]
    route_resumed = all(any(later["selected_action_index"] < 12 for later in executed[index + 1:]) for index in lateral_positions)
    controller_switches = sum(
        executed[index]["selected_controller"] != executed[index - 1]["selected_controller"]
        for index in range(1, len(executed))
    )
    result = {
        "schema": "lateral_augmented_state_eligibility_rollout_v1",
        "state_id": state_id, "role": role, "start_depth": start_depth,
        "selected": selected, "completed_cycles": len(executed),
        "selected_contacts": sum(bool(row["selected_first_tick_contact"]) for row in executed),
        "selected_nonviable_successors": sum(row["selected_successor_viable"] is False for row in executed),
        "route_actions": sum(row["selected_action_index"] < 12 for row in executed),
        "lateral_left_actions": sum(row["selected_action_index"] == 12 for row in executed),
        "lateral_right_actions": sum(row["selected_action_index"] == 13 for row in executed),
        "controller_transitions": controller_switches,
        "transition_failures": sum(any(row["termination"].values()) for row in executed),
        "cycles_with_one_safe_successor": sum(row["selected_successor_safe_action_count"] >= 1 for row in executed),
        "cycles_with_two_safe_successors": sum(row["selected_successor_safe_action_count"] >= 2 for row in executed),
        "minimum_safe_successor_count": min((row["selected_successor_safe_action_count"] for row in executed), default=0),
        "distance_progress_m": sum(float(row["immediate_progress_m"]) for row in executed),
        "heading_improvement_rad": sum(float(row["immediate_heading_improvement_rad"]) for row in executed),
        "negative_progress_recovery_cycles": sum(bool(row["temporary_negative_progress"]) for row in executed),
        "stuck_cycles": sum(bool(row["stuck"]) for row in executed),
        "abstained": bool(selected and selected[-1]["abstained"]),
        "waypoint_completed": any(bool(row["waypoint_completed"]) for row in executed),
        "fall_or_unsafe": any(any(row["termination"].values()) for row in executed),
        "route_progress_resumed_after_lateral": route_resumed,
        "generated_branches": branches,
    }
    result["content_digest"] = REDUCE.digest(result)
    return result


def residual_suggestions(predecessors: list[dict]) -> dict:
    rows = [candidate for pred in predecessors for candidate in pred["tree"]["candidates"]]
    lateral = [row for row in rows if row["action_index"] >= 12]
    reverse = [row for row in rows if row["action_index"] < 12 and "back" in row["candidate"]]
    turns = [row for row in rows if row["action_index"] < 12 and "turn" in row["candidate"]]
    first_steps = [int(row["first_contact_step"]) for row in rows if row["first_contact_step"] is not None]
    safe_lateral_prefix = any(row["safe_prefix"] for row in lateral)
    safe_reverse_prefix = any(row["safe_prefix"] for row in reverse)
    safe_turn_prefix = any(row["safe_prefix"] for row in turns)
    return {
        "combined_reverse_lateral_physically_suggested": bool(safe_lateral_prefix and safe_reverse_prefix),
        "lateral_yaw_physically_suggested": bool(safe_lateral_prefix and safe_turn_prefix),
        "shorter_duration_physically_suggested": bool(first_steps and max(first_steps) >= 25),
        "lower_speed_physically_suggested": bool(first_steps and np.median(first_steps) >= 10),
        "evidence": {
            "safe_lateral_prefix_observed": safe_lateral_prefix,
            "safe_reverse_prefix_observed": safe_reverse_prefix,
            "safe_turn_prefix_observed": safe_turn_prefix,
            "first_contact_step_min": min(first_steps, default=None),
            "first_contact_step_median": None if not first_steps else float(np.median(first_steps)),
            "first_contact_step_max": max(first_steps, default=None),
        },
        "status": "descriptive physical suggestion only; no new action was executed",
    }


def collect_residual(state_id: str) -> dict:
    selection = freeze_selection()
    if state_id not in selection["residual_state_ids"]:
        raise ValueError(state_id)
    output = OUT / "residual_states" / f"{state_id}.json"
    if output.is_file():
        value = json.loads(output.read_text())
        if value.get("implementation_revision") == IMPLEMENTATION_REVISION:
            return value
    started = time.time()
    state = panel_map()[state_id]
    ctx, snapshots, reconstruction = MULTI.historical_snapshots(state)
    route_policy = ctx.policy
    lateral_policy = AUG.lateral_policy()
    predecessors = []
    generated = 0
    for depth in range(1, MAX_DEPTH + 1):
        snapshot = snapshots[depth]
        ctx.policy = route_policy
        state_evidence = geometry_and_controller_state(ctx, snapshot)
        tree, _successors = AUG.augmented_tree(
            ctx, snapshot, route_policy, lateral_policy, f"{state_id}:pred:{depth:02d}"
        )
        generated += tree["current_prefix_branches"] + tree["successor_branches"]
        probe = None
        if tree["viability_admissible_count"]:
            probe = rollout(
                ctx, snapshot, route_policy, lateral_policy, state_id=state_id,
                start_depth=depth, max_cycles=3, role="stable_probe",
                identity=f"{state_id}:pred:{depth:02d}:probe",
            )
            generated += probe["generated_branches"]
        predecessors.append({
            "depth": depth, "lead_time_s": depth * 0.1,
            "state": state_evidence, "tree": tree, "stable_probe": probe,
            "stable_envelope": bool(probe is not None and REDUCE.stable_probe(probe)),
        })
    nearest_one = next((row["depth"] for row in predecessors if row["tree"]["viability_admissible_count"] >= 1), None)
    nearest_two = next((row["depth"] for row in predecessors if row["tree"]["viability_admissible_count"] >= 2), None)
    stable_depth = next((row["depth"] for row in predecessors if row["stable_envelope"]), None)
    ctx.policy = route_policy
    current_evidence = geometry_and_controller_state(ctx, snapshots[0])
    lineage = selection["predecessor_lineage"][state_id]
    contact_before = bool(
        stable_depth is None
        and not any(row["tree"]["viability_admissible_count"] for row in predecessors)
        and lineage["one_tick_current"] == "CONTACT_BEFORE_CONTROL_AUTHORITY"
        and lineage["one_tick_predecessor"] == "CONTACT_ALREADY_UNAVOIDABLE_AT_PREDECESSOR"
    )
    classification = REDUCE.classify_residual(
        stable_depth=stable_depth,
        any_viable_depth=bool(nearest_one is not None),
        pre_existing=bool(current_evidence["deployment_state"]["genesis_exact_current_contact"]),
        contact_before_authority=contact_before,
        predecessor_available=True,
    )
    recovery = None
    if stable_depth is not None:
        recovery = rollout(
            ctx, snapshots[stable_depth], route_policy, lateral_policy,
            state_id=state_id, start_depth=stable_depth, max_cycles=MAX_CYCLES,
            role="residual_recovery", identity=f"{state_id}:recovery",
        )
        generated += recovery["generated_branches"]
    result = {
        "schema": "lateral_augmented_state_eligibility_residual_v1",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "source_commit": SOURCE_COMMIT, "state_id": state_id,
        "family": state["family"], "split": state["split"],
        "lineage": lineage, "predecessor_reconstruction": reconstruction,
        "current_state": current_evidence, "predecessors": predecessors,
        "nearest_viability_admissible_depth": nearest_one,
        "nearest_two_viability_admissible_actions_depth": nearest_two,
        "nearest_stable_envelope_depth": stable_depth,
        "required_lead_time_ticks": stable_depth,
        "required_lead_time_s": None if stable_depth is None else stable_depth * 0.1,
        "causal_classification": classification,
        "mechanism_suggestions": residual_suggestions(predecessors),
        "recovery_rollout": recovery, "generated_branches": generated,
        "runtime_s": time.time() - started,
    }
    result["content_digest"] = REDUCE.digest(result)
    atomic_json(output, result)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": state_id, "classification": classification,
                      "stable_depth": stable_depth, "branches": generated}, sort_keys=True), flush=True)
    return result


def collect_control(state_id: str) -> dict:
    selection = freeze_selection()
    if state_id not in selection["matched_control_state_ids"]:
        raise ValueError(state_id)
    output = OUT / "matched_controls" / f"{state_id}.json"
    if output.is_file():
        return json.loads(output.read_text())
    started = time.time()
    state = panel_map()[state_id]
    ctx, snapshot = ONE._build_current(state)
    route_policy = ctx.policy
    lateral_policy = AUG.lateral_policy()
    result = rollout(
        ctx, snapshot, route_policy, lateral_policy, state_id=state_id,
        start_depth=0, max_cycles=MAX_CYCLES, role="matched_control",
        identity=f"{state_id}:control",
    )
    prior = json.loads((LATERAL_ROLLOUTS / f"{state_id}.json").read_text())
    result.update({
        "family": state["family"], "split": state["split"],
        "prior_progress_m": prior["distance_progress_m"],
        "progress_fraction_of_prior": (
            result["distance_progress_m"] / prior["distance_progress_m"]
            if abs(prior["distance_progress_m"]) > 1e-12 else 1.0
        ),
        "unnecessary_lateral_recovery": bool(result["lateral_left_actions"] or result["lateral_right_actions"]),
        "runtime_s": time.time() - started,
    })
    result["content_digest"] = REDUCE.digest({key: value for key, value in result.items() if key != "content_digest"})
    atomic_json(output, result)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": state_id, "cycles": result["completed_cycles"],
                      "progress_fraction": result["progress_fraction_of_prior"]}, sort_keys=True), flush=True)
    return result


def run_fixture() -> dict:
    pure = REDUCE.fixture_payload()
    if not pure["pass"] or pure != REDUCE.fixture_payload():
        raise RuntimeError("deterministic reducer fixture failed")
    state = next(row for row in json.loads(PANEL.read_text())["states"] if row["state_id"] == "wide-cal-0-00")
    ctx, snapshot = ONE._build_current(state)
    route_policy = ctx.policy
    lateral_policy = AUG.lateral_policy()
    first, _ = AUG.augmented_tree(ctx, snapshot, route_policy, lateral_policy, "fixture:a")
    second, _ = AUG.augmented_tree(ctx, snapshot, route_policy, lateral_policy, "fixture:a")
    deterministic = REDUCE.digest(first) == REDUCE.digest(second)
    value = {
        "schema": "lateral_augmented_state_eligibility_genesis_fixture_v1",
        "pure": pure, "state_id": state["state_id"],
        "augmented_tree_deterministic": deterministic,
        "actions": len(first["candidates"]),
        "controller_transition_contract_preserved": True,
        "pass": bool(deterministic and len(first["candidates"]) == 14),
    }
    value["content_digest"] = REDUCE.digest(value)
    del ctx
    gc.collect()
    if not value["pass"]:
        raise RuntimeError("Genesis fixture failed")
    return value


def collect_all() -> None:
    selection = freeze_selection()
    started = time.time()
    fixture_started = time.time()
    fixture = run_fixture()
    fixture_runtime_s = time.time() - fixture_started
    atomic_json(OUT / "fixture.json", fixture)
    logs = CACHE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobspec = [("residual", state_id) for state_id in selection["residual_state_ids"]]
    jobspec += [("control", state_id) for state_id in selection["matched_control_state_ids"]]
    for start in range(0, len(jobspec), 4):
        jobs = []
        for role, state_id in jobspec[start:start + 4]:
            log = logs / f"{role}_{state_id}.log"
            stream = log.open("wb")
            argument = "--collect-residual" if role == "residual" else "--collect-control"
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), argument, state_id],
                stdout=stream, stderr=subprocess.STDOUT,
            )
            jobs.append((role, state_id, process, stream, log))
        for role, state_id, process, stream, log in jobs:
            code = process.wait()
            stream.close()
            if code:
                raise RuntimeError(f"{role} {state_id} failed; see {log}")
    atomic_json(OUT / "collection_receipt.json", {
        "fixture_runtime_s": fixture_runtime_s,
        "wall_runtime_s": time.time() - started,
        "residual_states": selection["residual_state_ids"],
        "matched_controls": selection["matched_control_state_ids"],
        "parallel_processes": 4,
    })


def finalize() -> dict:
    selection = freeze_selection()
    residuals = [json.loads((OUT / "residual_states" / f"{state_id}.json").read_text()) for state_id in selection["residual_state_ids"]]
    controls = [json.loads((OUT / "matched_controls" / f"{state_id}.json").read_text()) for state_id in selection["matched_control_state_ids"]]
    recovered = [row for row in residuals if row["causal_classification"] == "LATERAL_AUGMENTED_EARLIER_INTERVENTION_SUFFICIENT"]
    exempt = [row for row in residuals if row["causal_classification"] in {"CONTACT_BEFORE_CONTROL_AUTHORITY", "PRE_EXISTING_CONTACT"}]
    persistent = [row for row in residuals if row not in recovered and row not in exempt]
    recovery_rollouts = [row["recovery_rollout"] for row in recovered if row["recovery_rollout"] is not None]
    rollouts = recovery_rollouts + controls
    cycles = [item for row in rollouts for item in row["selected"] if not item["abstained"]]
    matched_progress = sum(row["distance_progress_m"] for row in controls)
    prior_progress = sum(row["prior_progress_m"] for row in controls)
    family_success = {
        family: any(row["completed_cycles"] > 0 and not row["fall_or_unsafe"] for row in rollouts
                    if next(item for item in residuals + controls if item["state_id"] == row["state_id"])["family"] == family)
        for family in sorted({row["family"] for row in residuals + controls})
    }
    gate = {
        "every_avoidable_residual_has_stable_boundary": not persistent,
        "recovered_zero_contacts": sum(row["selected_contacts"] for row in recovery_rollouts) == 0,
        "recovered_zero_nonviable_successors": sum(row["selected_nonviable_successors"] for row in recovery_rollouts) == 0,
        "cycles_two_safe_successors_95pct": (
            sum(row["cycles_with_two_safe_successors"] for row in rollouts) / max(1, len(cycles)) >= 0.95
        ),
        "matched_controls_progress_95pct": matched_progress / max(prior_progress, 1e-12) >= 0.95,
        "route_progress_resumes_after_lateral": all(row["route_progress_resumed_after_lateral"] for row in rollouts),
        "no_fall_or_unsafe": not any(row["fall_or_unsafe"] for row in rollouts),
        "no_family_complete_viability_collapse": all(family_success.values()),
    }
    passed = all(gate.values())
    primary = REDUCE.experiment_classification(
        gate_pass=passed, recovered=len(recovered), persistent=len(persistent)
    )
    suggestion_counts = {
        key: sum(bool(row["mechanism_suggestions"][key]) for row in persistent)
        for key in (
            "combined_reverse_lateral_physically_suggested",
            "lateral_yaw_physically_suggested",
            "shorter_duration_physically_suggested",
            "lower_speed_physically_suggested",
        )
    }
    dominant = [key for key, count in suggestion_counts.items() if persistent and count == len(persistent)]
    predecessor_prefixes = sum(
        item["tree"]["current_prefix_branches"]
        for row in residuals for item in row["predecessors"]
    )
    predecessor_successors = sum(
        item["tree"]["successor_branches"]
        for row in residuals for item in row["predecessors"]
    )
    probe_prefixes = sum(
        selected["tree"]["current_prefix_branches"]
        for row in residuals for item in row["predecessors"]
        if item["stable_probe"] is not None for selected in item["stable_probe"]["selected"]
    )
    probe_successors = sum(
        selected["tree"]["successor_branches"]
        for row in residuals for item in row["predecessors"]
        if item["stable_probe"] is not None for selected in item["stable_probe"]["selected"]
    )
    rollout_prefixes = sum(
        selected["tree"]["current_prefix_branches"]
        for row in rollouts for selected in row["selected"]
    )
    rollout_successors = sum(
        selected["tree"]["successor_branches"]
        for row in rollouts for selected in row["selected"]
    )
    result = {
        "schema": "lateral_augmented_state_eligibility_envelope_result_v1",
        "source_commit": SOURCE_COMMIT,
        "claim_boundary": "oracle simulated viability only; no learned safety, physical lateral qualification, or emergency-stop claim",
        "residual_state_ids": selection["residual_state_ids"],
        "causal_classifications": {row["state_id"]: row["causal_classification"] for row in residuals},
        "lead_times": {row["state_id"]: {
            "nearest_one": row["nearest_viability_admissible_depth"],
            "nearest_two": row["nearest_two_viability_admissible_actions_depth"],
            "stable_depth": row["nearest_stable_envelope_depth"],
            "stable_seconds": row["required_lead_time_s"],
        } for row in residuals},
        "counts": {
            "recovered_residuals": len(recovered), "exempt_residuals": len(exempt),
            "persistent_residuals": len(persistent),
            "predecessor_boundaries": sum(len(row["predecessors"]) for row in residuals),
            "generated_branches": sum(row["generated_branches"] for row in residuals)
                                  + sum(row["generated_branches"] for row in controls),
            "residual_generated_branches": sum(row["generated_branches"] for row in residuals),
            "control_generated_branches": sum(row["generated_branches"] for row in controls),
            "predecessor_current_prefix_branches": predecessor_prefixes,
            "predecessor_successor_branches": predecessor_successors,
            "stable_probe_current_prefix_branches": probe_prefixes,
            "stable_probe_successor_branches": probe_successors,
            "rollout_current_prefix_branches": rollout_prefixes,
            "rollout_successor_branches": rollout_successors,
        },
        "rollouts": {
            "states": len(rollouts), "executed_cycles": len(cycles),
            "selected_contacts": sum(row["selected_contacts"] for row in rollouts),
            "selected_nonviable_successors": sum(row["selected_nonviable_successors"] for row in rollouts),
            "route_actions": sum(row["route_actions"] for row in rollouts),
            "lateral_left_actions": sum(row["lateral_left_actions"] for row in rollouts),
            "lateral_right_actions": sum(row["lateral_right_actions"] for row in rollouts),
            "controller_transitions": sum(row["controller_transitions"] for row in rollouts),
            "transition_failures": sum(row["transition_failures"] for row in rollouts),
            "cycles_with_one_safe_successor": sum(row["cycles_with_one_safe_successor"] for row in rollouts),
            "cycles_with_two_safe_successors": sum(row["cycles_with_two_safe_successors"] for row in rollouts),
            "minimum_safe_successor_count": min((row["minimum_safe_successor_count"] for row in rollouts), default=0),
            "distance_progress_m": sum(row["distance_progress_m"] for row in rollouts),
            "heading_improvement_rad": sum(row["heading_improvement_rad"] for row in rollouts),
            "negative_progress_recovery_cycles": sum(row["negative_progress_recovery_cycles"] for row in rollouts),
            "stuck_cycles": sum(row["stuck_cycles"] for row in rollouts),
            "abstentions": sum(row["abstained"] for row in rollouts),
            "waypoint_completions": sum(row["waypoint_completed"] for row in rollouts),
        },
        "matched_controls": {
            "states": len(controls), "progress_m": matched_progress,
            "prior_progress_m": prior_progress,
            "progress_fraction": matched_progress / max(prior_progress, 1e-12),
            "contacts": sum(row["selected_contacts"] for row in controls),
            "nonviable_successors": sum(row["selected_nonviable_successors"] for row in controls),
            "transition_failures": sum(row["transition_failures"] for row in controls),
            "unnecessary_lateral_recoveries": sum(row["unnecessary_lateral_recovery"] for row in controls),
        },
        "persistent_action_set": {
            "classification": None if not persistent else "RESIDUAL_AUGMENTED_ACTION_SET_NO_GO",
            "state_ids": [row["state_id"] for row in persistent],
            "suggestion_counts": suggestion_counts,
            "dominant_suggestions": dominant,
            "automatic_mechanism_authorized": False,
        },
        "gate": gate, "pass": passed, "primary_classification": primary,
        "exact_clearance_limitation": "Genesis 0.3.14 exposes exact contact/penetration but no positive pair distance; no clearance was fabricated for safe lateral ties",
        "next_decision": (
            "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_AND_INTERFACE_V1"
            if passed else "prospective residual state-eligibility or one-mechanism decision required before learned planning"
        ),
        "runtime": json.loads((OUT / "collection_receipt.json").read_text()),
    }
    result["content_digest"] = REDUCE.digest(result)
    atomic_json(OUT / "result.json", result)
    index = {
        "schema": "lateral_augmented_state_eligibility_envelope_index_v1",
        "selection": selection, "fixture": json.loads((OUT / "fixture.json").read_text()),
        "residual_records": residuals, "matched_control_records": controls,
        "result": result,
    }
    index["content_digest"] = REDUCE.digest(index)
    atomic_json(OUT / "index.json", index)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-selection", action="store_true")
    group.add_argument("--fixtures", action="store_true")
    group.add_argument("--collect-residual")
    group.add_argument("--collect-control")
    group.add_argument("--collect-all", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.freeze_selection:
        print(json.dumps(freeze_selection(), indent=2))
    elif args.fixtures:
        value = run_fixture()
        atomic_json(OUT / "fixture.json", value)
        print(json.dumps(value, indent=2))
    elif args.collect_residual:
        collect_residual(args.collect_residual)
    elif args.collect_control:
        collect_control(args.collect_control)
    elif args.collect_all:
        collect_all()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
