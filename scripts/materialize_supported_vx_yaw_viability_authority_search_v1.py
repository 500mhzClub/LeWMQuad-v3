#!/usr/bin/env python3
"""Materialize the frozen supported-vx/yaw two-step oracle search."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import supported_vx_yaw_viability_authority_search_v1 as S
from scripts import materialize_genesis_narrowphase_candidate_feasibility_v1 as NARROW
from scripts import materialize_one_tick_viability_constrained_mpc_v1 as ONE
from scripts import run_go2_oracle_branch_pilot_v1_2 as V


SOURCE_COMMIT = "11a0c258e479f79a640ab237841f52ec0e6b6ecc"
OUT = ROOT / ".generated/supported_vx_yaw_viability_authority_search_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/supported_vx_yaw_viability_authority_search_v1"
PANEL = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_panel_manifest.json"
OLD_TREE = ROOT / ".generated/one_tick_viability_constrained_mpc_v1/viability_tree_index.json"
SELECTION = ROOT / ".generated/multi_cycle_viability_envelope_v1/frozen_state_selection.json"
FIXTURE_PANEL = ROOT / ".generated/factorised_micro_safety_data_scaling_v2/panel_manifest.json"
MANIFEST = ROOT / "config/go2_platform_manifest.yaml"
REGISTRY = ROOT / "config/go2_primitive_registry.yaml"
POLICY_CFG = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl"
GRID_RECEIPT = OUT / "frozen_requested_grid.json"
PHYSICS_STEPS = 50

FIXTURE_CONTEXTS = (
    ("obstacle_free_rest", "scale-fit-0-00", None),
    ("obstacle_free_forward_moving_initial_state", "scale-fit-0-00", (0.30, 0.0, 0.0)),
    ("obstacle_free_turning_initial_state", "scale-fit-0-00", (0.0, 0.0, 0.45)),
    ("rear_clearance", "scale-fit-0-00", None),
    ("left_wall", "scale-fit-0-01", (0.0, 0.0, 0.45)),
    ("right_wall", "scale-fit-0-01", (0.0, 0.0, -0.45)),
    ("front_left_corner", "scale-fit-0-01", (0.20, 0.0, 0.45)),
    ("front_right_corner", "scale-fit-0-01", (0.20, 0.0, -0.45)),
    ("narrow_corridor", "scale-fit-3-00", None),
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fixture_reduction(value):
    """Canonicalize sub-0.1 mm/rad backend noise in fixture-only reductions."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [fixture_reduction(item) for item in value]
    if isinstance(value, dict):
        return {key: fixture_reduction(item) for key, item in value.items()}
    return value


def freeze_grid() -> dict:
    if GRID_RECEIPT.is_file():
        result = json.loads(GRID_RECEIPT.read_text())
        expected = S.digest({key: value for key, value in result.items() if key != "content_digest"})
        if result["content_digest"] != expected:
            raise RuntimeError("grid receipt digest mismatch")
        return result
    manifest = yaml.safe_load(MANIFEST.read_text())
    with POLICY_CFG.open("rb") as stream:
        configs = pickle.load(stream)
    policy = configs[3]
    grid = S.requested_grid()
    result = {
        "schema": "supported_vx_yaw_frozen_requested_grid_v1",
        "source_commit": SOURCE_COMMIT,
        "frozen_before_scientific_outcomes": True,
        "grid_construction": "zero + four pure reverse + eight mirrored turns + eight matched-fraction mirrored reverse arcs",
        "grid": grid,
        "count": len(grid),
        "controller_contract": {
            "manifest_vx_range_m_s": [float(manifest["locomotion"]["safety"]["min_vx_mps"]),
                                       float(manifest["locomotion"]["safety"]["max_vx_mps"])],
            "policy_training_vx_range_m_s": [float(x) for x in policy["lin_vel_x_range"]],
            "bound_reverse_magnitude_m_s": S.MAX_REVERSE_M_S,
            "manifest_yaw_range_rad_s": [-float(manifest["locomotion"]["safety"]["max_yaw_rate_radps"]),
                                           float(manifest["locomotion"]["safety"]["max_yaw_rate_radps"])],
            "policy_training_yaw_range_rad_s": [float(x) for x in policy["ang_vel_range"]],
            "bound_yaw_magnitude_rad_s": S.MAX_YAW_RAD_S,
            "vy_m_s": 0.0,
            "command_period_s": float(manifest["timing"]["command_dt_s"]),
            "policy_period_s": float(manifest["timing"]["policy_dt_s"]),
            "max_command_delta": manifest["locomotion"]["safety"]["max_command_delta_per_tick"],
        },
        "bindings": {"manifest_sha256": sha(MANIFEST), "registry_sha256": sha(REGISTRY),
                     "policy_cfg_sha256": sha(POLICY_CFG), "policy_checkpoint_opened": False},
    }
    result["content_digest"] = S.digest(result)
    atomic_json(GRID_RECEIPT, result)
    return result


def topology(ctx):
    topo = V.link_topology(ctx)
    solver = ctx.build.scene.rigid_solver
    link_names = {int(link.idx): str(link.name) for link in solver.links}
    object_names = {int(link.idx): str(link.entity.name) for link in solver.links}
    return topo, link_names, object_names


def applied_for(ctx, snapshot, requested_command) -> list[float]:
    ONE._restore_tick_boundary(ctx, snapshot)
    requested = np.repeat(np.asarray(requested_command, np.float32)[None, None, :],
                          int(ctx.runner._block_size), axis=1)
    return np.asarray(ctx.runner._clip_block(requested).executed[0, 0], np.float32).tolist()


def execute_command(ctx, snapshot, requested_command, *, identity: str, topo, link_names, object_names):
    ONE._restore_tick_boundary(ctx, snapshot)
    runner = ctx.runner
    start = ctx.pose()
    requested = np.repeat(np.asarray(requested_command, np.float32)[None, None, :],
                          int(runner._block_size), axis=1)
    clipped = runner._clip_block(requested)
    target = np.asarray(clipped.executed[:, 0], np.float32)
    contact = np.zeros(PHYSICS_STEPS, np.uint8)
    first_summary = None
    counter = 0
    finite = True
    for _ in range(int(runner._policy_steps_per_command_tick)):
        observation = runner._build_observation(target)
        joint_targets = runner.policy.act(observation)
        finite = finite and bool(np.isfinite(runner._as_np(joint_targets)).all())
        runner._apply_joint_targets(joint_targets)
        for _ in range(int(runner._physics_steps_per_policy)):
            runner.build.scene.step()
            summary = NARROW.contact_summary(
                runner.build.robot.get_contacts(), topo, link_names, object_names,
                force_threshold=True,
            )
            contact[counter] = bool(summary["active"])
            if summary["active"] and first_summary is None:
                first_summary = summary
            counter += 1
        runner._sim_time_ns += runner._policy_dt_ns
    if counter != PHYSICS_STEPS:
        raise RuntimeError(counter)
    ONE._advance_tick_counters(ctx, target)
    endpoint = ctx.pose()
    flags = V.V1._termination_flags(ctx)
    successor = ONE._capture_tick_boundary(
        ctx, goal=snapshot.goal, identity={**snapshot.identity, "branch_identity": identity}
    )
    dx = float(endpoint[0][0] - start[0][0]); dy = float(endpoint[0][1] - start[0][1])
    body_dx = math.cos(start[1]) * dx + math.sin(start[1]) * dy
    body_dy = -math.sin(start[1]) * dx + math.cos(start[1]) * dy
    yaw_delta = (float(endpoint[1]) - float(start[1]) + math.pi) % (2 * math.pi) - math.pi
    return {
        "identity": identity,
        "requested_vx_vy_wz": [float(x) for x in requested_command],
        "applied_vx_vy_wz": target[0].tolist(),
        "clipped": bool(np.any(np.asarray(requested_command, np.float32) != target[0])),
        "contact": bool(contact.any()),
        "first_contact_step": None if not contact.any() else int(np.flatnonzero(contact)[0]),
        "contact_link": None if first_summary is None else first_summary.get("robot_link_name"),
        "contact_object": None if first_summary is None else first_summary.get("environment_object"),
        "endpoint_pose": [[float(endpoint[0][0]), float(endpoint[0][1])], float(endpoint[1]), float(endpoint[2])],
        "body_displacement_xy_m": [body_dx, body_dy], "yaw_change_rad": yaw_delta,
        "finite_controller_outputs": finite, "termination": flags,
        "successor_digest": successor.digest,
    }, contact, successor


def historical_requests() -> list[dict]:
    rows = []
    for index, (name, primitives) in enumerate(V.V1.CANDIDATE_BANK):
        command = np.asarray(V.V1.block_for(primitives[0])[0], np.float32).tolist()
        rows.append({"source": "historical", "candidate_index": index, "name": name,
                     "requested_vx_vy_wz": command})
    return rows


def successor_search(ctx, snapshot, *, identity: str, topo, link_names, object_names) -> tuple[list[dict], int]:
    requests = historical_requests() + [
        {"source": "search", **row} for row in S.requested_grid()
    ]
    groups = {}
    for row in requests:
        applied = applied_for(ctx, snapshot, row["requested_vx_vy_wz"])
        key = S.command_key(applied)
        groups.setdefault(key, []).append({**row, "applied_vx_vy_wz": applied})
    outcomes = []
    for group_index, (key, aliases) in enumerate(groups.items()):
        representative = aliases[0]
        outcome, _contact, _successor = execute_command(
            ctx, snapshot, representative["requested_vx_vy_wz"],
            identity=f"{identity}:next:{group_index:02d}", topo=topo,
            link_names=link_names, object_names=object_names,
        )
        outcomes.append({"applied_vx_vy_wz": list(key), "aliases": aliases,
                         "contact": outcome["contact"],
                         "first_contact_step": outcome["first_contact_step"]})
    return outcomes, len(groups)


def run_fixtures() -> dict:
    fixture_grid = freeze_grid()
    pure = S.fixture_payload()
    panel = {row["state_id"]: row for row in json.loads(FIXTURE_PANEL.read_text())["states"]}
    records = []
    started = time.time()
    for context_name, state_id, prefix in FIXTURE_CONTEXTS:
        state = panel[state_id]
        ctx, snapshot = ONE._build_current(state)
        topo, link_names, object_names = topology(ctx)
        if prefix is not None:
            _record, _contact, snapshot = execute_command(
                ctx, snapshot, prefix, identity=f"fixture:{context_name}:prefix",
                topo=topo, link_names=link_names, object_names=object_names,
            )
        for command in fixture_grid["grid"]:
            repeated = []
            for repeat in range(2):
                outcome, _contact, _successor = execute_command(
                    ctx, snapshot, command["requested_vx_vy_wz"],
                    identity=f"fixture:{context_name}:{command['search_index']:02d}:{repeat}",
                    topo=topo, link_names=link_names, object_names=object_names,
                )
                core = fixture_reduction({key: value for key, value in outcome.items()
                                          if key not in {"identity", "successor_digest"}})
                repeated.append({"repeat": repeat, "outcome": outcome,
                                 "core_digest": S.digest(core)})
            records.append({"context": context_name, "state_id": state_id,
                            "command": command, "runs": repeated,
                            "deterministic": repeated[0]["core_digest"] == repeated[1]["core_digest"]})
        del ctx
        gc.collect()
    obstacle_free = [row for row in records if row["context"].startswith("obstacle_free")]
    reverse_response = [
        -float(run["outcome"]["body_displacement_xy_m"][0])
        for row in obstacle_free if row["command"]["requested_vx_vy_wz"][0] < 0
        for run in row["runs"]
    ]
    yaw_response = [
        abs(float(run["outcome"]["yaw_change_rad"]))
        for row in obstacle_free if abs(row["command"]["requested_vx_vy_wz"][2]) > 0
        for run in row["runs"]
    ]
    result = {
        "schema": "supported_vx_yaw_training_fixture_result_v1", "pure": pure,
        "deterministic_reduction_precision": "continuous fixture-only values rounded to 1e-4; contact, termination, and commands exact",
        "contexts": len(FIXTURE_CONTEXTS), "grid_commands": len(fixture_grid["grid"]),
        "branches": sum(len(row["runs"]) for row in records),
        "all_deterministic": all(row["deterministic"] for row in records),
        "finite_outputs": all(run["outcome"]["finite_controller_outputs"] for row in records for run in row["runs"]),
        "obstacle_free_contacts": sum(run["outcome"]["contact"] for row in obstacle_free for run in row["runs"]),
        "falls_or_unsafe": sum(any(run["outcome"]["termination"].values()) for row in records for run in row["runs"]),
        "maximum_obstacle_free_reverse_response_m": max(reverse_response, default=0.0),
        "maximum_obstacle_free_yaw_response_rad": max(yaw_response, default=0.0),
        "meaningful_reverse_response": max(reverse_response, default=0.0) >= 0.0005,
        "meaningful_yaw_response": max(yaw_response, default=0.0) >= 0.001,
        "records": records, "runtime_s": time.time() - started,
    }
    result["pass"] = bool(result["all_deterministic"] and result["finite_outputs"]
                           and result["obstacle_free_contacts"] == 0
                           and result["falls_or_unsafe"] == 0
                           and result["meaningful_reverse_response"]
                           and result["meaningful_yaw_response"])
    result["classification"] = None if result["pass"] else "SUPPORTED_VX_YAW_CONTROLLER_AUTHORITY_NO_GO"
    result["content_digest"] = S.digest(result)
    atomic_json(OUT / "training_fixture_result.json", result)
    return result


def search_state(state_id: str) -> dict:
    selection = json.loads(SELECTION.read_text())
    allowed = selection["failure_state_ids"] + selection["control_state_ids"]
    if state_id not in allowed:
        raise ValueError(state_id)
    output = OUT / "search_states" / f"{state_id}.json"
    panel = {row["state_id"]: row for row in json.loads(PANEL.read_text())["states"]}
    old = {row["state_id"]: row for row in json.loads(OLD_TREE.read_text())["state_records"]}[state_id]
    ctx, snapshot = ONE._build_current(panel[state_id])
    topo, link_names, object_names = topology(ctx)
    previous_applied_command = np.asarray(ctx.runner._last_executed[0], np.float32).tolist()
    applied_rows = []
    for command in S.requested_grid():
        applied_rows.append({**command, "applied_vx_vy_wz": applied_for(ctx, snapshot, command["requested_vx_vy_wz"])})
    dedup = S.deduplicate_applied(applied_rows, old["current"])
    start_pose = ctx.pose()
    rows = []
    branches = successor_branches = 0
    for unique in dedup["unique"]:
        representative = next(row for row in applied_rows
                              if row["search_index"] == unique["representative_search_index"])
        outcome, _contact, successor = execute_command(
            ctx, snapshot, representative["requested_vx_vy_wz"],
            identity=f"{state_id}:search:{representative['search_index']:02d}",
            topo=topo, link_names=link_names, object_names=object_names,
        )
        branches += 1
        successor_outcomes = []
        if not outcome["contact"]:
            successor_outcomes, count = successor_search(
                ctx, successor, identity=f"{state_id}:search:{representative['search_index']:02d}",
                topo=topo, link_names=link_names, object_names=object_names,
            )
            successor_branches += count
        safe_successors = [row for row in successor_outcomes if not row["contact"]]
        endpoint = outcome["endpoint_pose"]
        progress = math.hypot(*panel[state_id]["waypoint_body_xy"]) - math.hypot(
            panel[state_id]["waypoint_body_xy"][0] - outcome["body_displacement_xy_m"][0],
            panel[state_id]["waypoint_body_xy"][1] - outcome["body_displacement_xy_m"][1],
        )
        rows.append({**representative, **unique, "outcome": outcome,
                     "safe_prefix": not outcome["contact"],
                     "successor_outcomes": successor_outcomes,
                     "successor_safe_action_count": len(safe_successors),
                     "viability_admissible": bool(not outcome["contact"] and safe_successors),
                     "immediate_route_progress_m": float(progress),
                     "resulting_scene_graph_clearance_m": float(ctx.scene_graph.clearance_to_walls(endpoint[0])),
                     "exact_positive_clearance_available": False})
    boundary_contact = bool(old["current"] and all(row["first_contact_step"] == 0 for row in old["current"]))
    classification = S.residual_classification(rows, boundary_contact=boundary_contact)
    result = {
        "schema": "supported_vx_yaw_search_state_v1", "source_commit": SOURCE_COMMIT,
        "state_id": state_id, "family": panel[state_id]["family"], "split": panel[state_id]["split"],
        "role": "failure" if state_id in selection["failure_state_ids"] else "matched_control",
        "previous_applied_command": previous_applied_command,
        "deduplication": dedup, "rows": rows, "classification": classification,
        "current_branches": branches, "successor_branches": successor_branches,
    }
    result["content_digest"] = S.digest(result)
    atomic_json(output, result)
    del ctx
    gc.collect()
    print(json.dumps({"state_id": state_id, "classification": classification,
                      "current": branches, "successor": successor_branches}), flush=True)
    return result


def collect_all() -> dict:
    grid = freeze_grid()
    fixture_path = OUT / "training_fixture_result.json"
    fixture = json.loads(fixture_path.read_text()) if fixture_path.is_file() else run_fixtures()
    if not fixture["pass"]:
        result = {"status": "STOP", "classification": "SUPPORTED_VX_YAW_CONTROLLER_AUTHORITY_NO_GO",
                  "grid": grid, "fixture": fixture, "scientific_states": 0}
        result["content_digest"] = S.digest(result)
        atomic_json(OUT / "search_index.json", result)
        return result
    selection = json.loads(SELECTION.read_text())
    state_ids = selection["failure_state_ids"] + selection["control_state_ids"]
    logs = CACHE / "logs"; logs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for start in range(0, len(state_ids), 4):
        jobs = []
        for state_id in state_ids[start:start + 4]:
            stream = (logs / f"{state_id}.log").open("wb")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                        "--search-state", state_id], stdout=stream,
                                       stderr=subprocess.STDOUT)
            jobs.append((state_id, process, stream))
        for state_id, process, stream in jobs:
            code = process.wait(); stream.close()
            if code:
                raise RuntimeError(f"{state_id} failed; see {logs / (state_id + '.log')}")
    records = [json.loads((OUT / "search_states" / f"{state_id}.json").read_text()) for state_id in state_ids]
    result = {
        "status": "PASS", "grid": grid, "fixture": fixture,
        "states": records, "runtime_s": time.time() - started,
        "counts": {"states": len(records),
                   "current_branches": sum(row["current_branches"] for row in records),
                   "successor_branches": sum(row["successor_branches"] for row in records)},
    }
    result["content_digest"] = S.digest(result)
    atomic_json(OUT / "search_index.json", result)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "runtime_s": result["runtime_s"],
                      "content_digest": result["content_digest"]}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-grid", action="store_true")
    group.add_argument("--fixtures", action="store_true")
    group.add_argument("--search-state")
    group.add_argument("--collect-all", action="store_true")
    args = parser.parse_args()
    if args.freeze_grid:
        print(json.dumps(freeze_grid(), indent=2))
    elif args.fixtures:
        print(json.dumps(run_fixtures(), indent=2))
    elif args.search_state:
        search_state(args.search_state)
    else:
        collect_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
