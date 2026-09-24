#!/usr/bin/env python3
"""Recover and evaluate the unchanged runtime veto on the frozen waypoint panel.

This is a development-only diagnostic.  The runtime adapter used by the
existing navigation benchmark is the privileged manifest occupancy grid and
is therefore explicitly *not* deployment valid.  Candidate outcomes are
never re-executed by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    sys.path.insert(0, str(extra))

import run_go2_oracle_branch_pilot_v1_2 as PILOT
from benchmark_topo_nav_e2e import _feasible_fraction
from lewm.planning.local_obstacles import PrivilegedGridObstacleModel
from lewm_worlds.manifest import parse_scene_manifest_dict
from lewm_worlds.planning_grid import InflatedOccupancyGrid
from lewm_worlds.scene_graph import SceneGraph

V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
OUT = ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1"
STATE_OUT = OUT / "states"
DELTA_D = 0.03
DELTA_THETA = math.radians(5.0)
GUARD = {
    "source": "privileged_manifest_grid",
    "deployment_valid": False,
    "cell_size_m": 0.05,
    "grid_inflation_m": 0.20,
    "body_radius_m": 0.25,
    "horizon_blocks": 2,
    "admission_fraction": 0.70,
    "hold_always_admitted": True,
    "candidate_contract": "existing benchmark veto evaluates the candidate's first primitive",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def counters(ctx):
    return {
        "episode_ticks": int(ctx.episode_ticks),
        "policy_steps": int(ctx.policy_steps),
        "ticks_executed": int(ctx.ticks_executed),
        "runner_episode_step": int(ctx.runner.episode_states[0].episode_step),
        "runner_sim_time_ns": int(ctx.runner._sim_time_ns),
    }


def evaluate_guard(ctx, registry, obstacle_model):
    before = counters(ctx)
    rows = []
    for candidate_index, (candidate, primitives) in enumerate(PILOT.V1.CANDIDATE_BANK):
        primitive = primitives[0]
        fraction = float(_feasible_fraction(
            ctx.build, registry, primitive, obstacle_model,
            float(registry.command_dt_s),
            horizon_blocks=GUARD["horizon_blocks"],
            body_radius_m=GUARD["body_radius_m"],
        ))
        admitted = primitive == "hold" or fraction >= GUARD["admission_fraction"]
        rows.append({
            "candidate_index": candidate_index,
            "candidate": candidate,
            "first_primitive": primitive,
            "feasible_fraction": fraction,
            "admitted": bool(admitted),
        })
    after = counters(ctx)
    return rows, before, after


def recover_state(state_index: int) -> dict:
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    entry = manifest["state_candidates"][state_index]
    sid = entry["state_id"]
    ledger = [json.loads(line) for line in (V1 / "branch_labels.jsonl").read_text().splitlines()]
    source = sorted((r for r in ledger if r["state_id"] == sid), key=lambda r: int(r["candidate_index"]))
    if len(source) != 12:
        raise RuntimeError(f"{sid}: expected 12 frozen branch rows, found {len(source)}")
    shared = PILOT.V1._load_shared("cpu")
    ctx = PILOT.V1.build_context(Path(entry["scene_dir"]), seed=int(entry["seed"]), backend="cpu", shared=shared)
    ctx.begin_episode()
    for _ in range(40):
        ctx.drive_one_block()
    topology = PILOT.link_topology(ctx)
    eligible = PILOT.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise RuntimeError(f"{sid}: frozen state no longer eligible: {eligible}")
    record, _field = eligible
    snapshot = PILOT.V1.capture_branch_state(
        ctx, goal=dict(record["goal"]),
        identity={"state_id": sid, "scene_id": entry["scene_id"], "family": entry["family"]},
    )
    source_snapshot_digests = sorted({str(row["snapshot_digest"]) for row in source if row.get("snapshot_digest")})
    if len(source_snapshot_digests) > 1:
        raise RuntimeError(f"{sid}: frozen branch rows disagree on snapshot digest")
    expected_snapshot_digest = entry.get("snapshot_digest") or (source_snapshot_digests[0] if source_snapshot_digests else None)
    snapshot_digest_match = expected_snapshot_digest is None or snapshot.digest == expected_snapshot_digest
    pose = ctx.pose()
    expected = entry["start_pose"]
    if not np.allclose([*pose[0], pose[1], pose[2]], [*expected[0], expected[1], expected[2]], atol=2e-5, rtol=0):
        raise RuntimeError(f"{sid}: pre-action pose mismatch")
    grid = InflatedOccupancyGrid(
        ctx.scene_graph.manifest,
        cell_size_m=GUARD["cell_size_m"],
        inflation_m=GUARD["grid_inflation_m"],
    )
    obstacle_model = PrivilegedGridObstacleModel(grid)
    if obstacle_model.contract.deployment_valid:
        raise RuntimeError("privileged guard unexpectedly marked deployment valid")
    for ci, row in enumerate(source):
        expected_candidate, expected_primitives = PILOT.V1.CANDIDATE_BANK[ci]
        if int(row["candidate_index"]) != ci or row["candidate"] != expected_candidate or tuple(row["primitives"]) != tuple(expected_primitives):
            raise RuntimeError(f"{sid}:{ci}: frozen candidate-bank mismatch")

    repeated = 2 if state_index in (0, 1) else 1
    evaluations = []
    for _ in range(repeated):
        PILOT.V1.restore_branch_state(ctx, snapshot)
        rows, before, after = evaluate_guard(ctx, shared["registry"], obstacle_model)
        evaluations.append({"rows": rows, "before": before, "after": after})
    first = evaluations[0]
    if first["before"] != first["after"]:
        raise RuntimeError(f"{sid}: runtime guard advanced simulator/controller counters")
    deterministic = all(e["rows"] == first["rows"] and e["before"] == e["after"] for e in evaluations)
    if not deterministic:
        raise RuntimeError(f"{sid}: repeated runtime guard evaluation differed")
    result = {
        "schema": "kinematic_route_with_runtime_safety_guard_v1_state",
        "state_index": state_index,
        "state_id": sid,
        "scene_id": entry["scene_id"],
        "family": entry["family"],
        "snapshot_digest": snapshot.digest,
        "expected_snapshot_digest": expected_snapshot_digest,
        "snapshot_digest_match": snapshot_digest_match,
        "current_pose": [[float(x) for x in pose[0]], float(pose[1]), float(pose[2])],
        "obstacle_contract": obstacle_model.contract.to_dict(),
        "guard_contract": GUARD,
        "candidate_rows": first["rows"],
        "guard_diagnostic_digest": digest_json(first["rows"]),
        "zero_simulator_step_after_snapshot": first["before"] == first["after"],
        "repeat_count": repeated,
        "repeat_deterministic": deterministic,
        "candidate_order_identical": [r["candidate_index"] for r in first["rows"]] == list(range(12)),
    }
    STATE_OUT.mkdir(parents=True, exist_ok=True)
    target = STATE_OUT / f"{sid}.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default))
    print(json.dumps({"state_id": sid, "admitted": sum(r["admitted"] for r in first["rows"]), "digest": sha(target)}), flush=True)
    return result


def confusion(records, truth_key="unsafe"):
    tp = sum(r[truth_key] and not r["admitted"] for r in records)
    fn = sum(r[truth_key] and r["admitted"] for r in records)
    tn = sum(not r[truth_key] and r["admitted"] for r in records)
    fp = sum(not r[truth_key] and not r["admitted"] for r in records)
    unsafe = tp + fn
    safe = tn + fp
    rejected = tp + fp
    return {
        "rows": len(records), "unsafe_rows": unsafe, "safe_rows": safe,
        "true_positive_rejected_unsafe": tp, "false_negative_admitted_unsafe": fn,
        "true_negative_admitted_safe": tn, "false_positive_rejected_safe": fp,
        "unsafe_recall": tp / unsafe if unsafe else None,
        "unsafe_false_negative_rate": fn / unsafe if unsafe else None,
        "safe_specificity": tn / safe if safe else None,
        "safe_candidate_retention": tn / safe if safe else None,
        "precision": tp / rejected if rejected else None,
        "balanced_accuracy": ((tp / unsafe) + (tn / safe)) / 2 if unsafe and safe else None,
        "candidate_count_rejected": rejected,
        "candidate_count_admitted": tn + fn,
    }


def route_order(indices, pd, ptheta):
    remaining = list(indices)
    ordered = []
    while remaining:
        best_pd = max(float(pd[i]) for i in remaining)
        near = [i for i in remaining if best_pd - float(pd[i]) <= DELTA_D]
        pick = min(near, key=lambda i: (-float(ptheta[i]), i))
        ordered.append(pick)
        remaining.remove(pick)
    return ordered


def preference(row_a, row_b):
    if row_a["unsafe"] != row_b["unsafe"]:
        return 1 if not row_a["unsafe"] else -1
    if row_a["unsafe"]:
        return 0
    dd = row_a["p_d"] - row_b["p_d"]
    if abs(dd) > DELTA_D:
        return 1 if dd > 0 else -1
    dt = row_a["p_theta"] - row_b["p_theta"]
    if abs(dt) > DELTA_THETA:
        return 1 if dt > 0 else -1
    return 0


def best_safe(ids, rows):
    safe = [i for i in ids if not rows[i]["unsafe"]]
    if not safe:
        return None
    best = safe[0]
    for idx in safe[1:]:
        if preference(rows[idx], rows[best]) > 0:
            best = idx
    return best


def planner_eval(name, rows, state_ids, admitted, nominal_pd, nominal_pt, mode):
    by_state = defaultdict(list)
    for i, row in enumerate(rows):
        if row["state_id"] in state_ids:
            by_state[row["state_id"]].append(i)
    per_state = []
    norm_regrets = []
    top1 = []
    top3 = []
    selected_progress = []
    selected_heading = []
    selected_unsafe = []
    false_abstentions = 0
    correct_abstentions = 0
    for sid, ids in sorted(by_state.items(), key=lambda item: int(item[0].split("-")[1])):
        ids = sorted(ids, key=lambda i: rows[i]["candidate_index"])
        safe = [i for i in ids if not rows[i]["unsafe"]]
        if mode == "hold":
            ranked, pick = [next(i for i in ids if rows[i]["candidate_index"] == 11)], next(i for i in ids if rows[i]["candidate_index"] == 11)
        elif mode == "lowest":
            ranked, pick = ids, ids[0]
        elif mode == "unguarded":
            ranked = route_order(ids, nominal_pd, nominal_pt); pick = ranked[0]
        elif mode == "guarded":
            candidates = [i for i in ids if admitted[i]]
            ranked = route_order(candidates, nominal_pd, nominal_pt) if candidates else []
            pick = ranked[0] if ranked else None
        elif mode == "oracle_kinematic":
            ranked = route_order(safe, nominal_pd, nominal_pt) if safe else []
            pick = ranked[0] if ranked else None
        elif mode == "oracle_best":
            pick = best_safe(ids, rows); ranked = [pick] if pick is not None else []
        else:
            raise ValueError(mode)
        best = best_safe(ids, rows)
        if pick is None:
            if safe: false_abstentions += 1
            else: correct_abstentions += 1
        else:
            selected_progress.append(rows[pick]["p_d"])
            selected_heading.append(rows[pick]["p_theta"])
            selected_unsafe.append(rows[pick]["unsafe"])
        if best is not None:
            top1.append(pick == best)
            top3.append(best in ranked[:3])
            if pick is not None and not rows[pick]["unsafe"] and len(safe) >= 2:
                vals = [rows[i]["p_d"] for i in safe]
                spread = max(vals) - min(vals)
                if spread > 1e-8:
                    norm_regrets.append((rows[best]["p_d"] - rows[pick]["p_d"]) / spread)
        per_state.append({
            "state_id": sid, "family": rows[ids[0]]["family"],
            "safe_candidates": len(safe), "admitted_candidates": sum(admitted[i] for i in ids),
            "selected_candidate": None if pick is None else rows[pick]["candidate_index"],
            "selected_safe": None if pick is None else not rows[pick]["unsafe"],
            "selected_distance_progress_m": None if pick is None else rows[pick]["p_d"],
            "selected_heading_improvement_deg": None if pick is None else math.degrees(rows[pick]["p_theta"]),
            "best_safe_candidate": None if best is None else rows[best]["candidate_index"],
            "best_safe_in_top3": None if best is None else best in ranked[:3],
        })
    per_family = {}
    for family in sorted({r["family"] for r in per_state}):
        subset = [r for r in per_state if r["family"] == family]
        moved = [r for r in subset if r["selected_candidate"] is not None]
        per_family[family] = {
            "states": len(subset),
            "selected_safe_rate": float(np.mean([r["selected_safe"] for r in moved])) if moved else None,
            "mean_selected_distance_progress_m": float(np.mean([r["selected_distance_progress_m"] for r in moved])) if moved else None,
            "abstention_rate": 1 - len(moved) / len(subset),
        }
    return {
        "condition": name, "states": len(by_state),
        "selected_unsafe_rate": float(np.mean(selected_unsafe)) if selected_unsafe else 0.0,
        "mean_selected_distance_progress_m": float(np.mean(selected_progress)) if selected_progress else 0.0,
        "mean_selected_heading_improvement_deg": float(math.degrees(np.mean(selected_heading))) if selected_heading else 0.0,
        "normalized_safe_progress_regret": float(np.mean(norm_regrets)) if norm_regrets else None,
        "normalized_regret_denominator_states": len(norm_regrets),
        "best_safe_top1": float(np.mean(top1)) if top1 else None,
        "best_safe_top3": float(np.mean(top3)) if top3 else None,
        "abstention_rate": sum(r["selected_candidate"] is None for r in per_state) / len(per_state),
        "correct_abstentions": correct_abstentions, "false_abstentions": false_abstentions,
        "per_state": per_state, "per_family": per_family,
    }


def integrate(post_slew, waypoint):
    x = y = yaw = 0.0
    for vx, vy, wz in [tick for block in post_slew[:3] for tick in block]:
        x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * 0.10
        y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * 0.10
        yaw = math.atan2(math.sin(yaw + float(wz) * 0.10), math.cos(yaw + float(wz) * 0.10))
    wx, wy, sg, cg = waypoint
    goal_heading = math.atan2(sg, cg)
    pd = math.hypot(wx, wy) - math.hypot(wx - x, wy - y)
    pt = abs(math.atan2(math.sin(goal_heading), math.cos(goal_heading))) - abs(math.atan2(math.sin(goal_heading - yaw), math.cos(goal_heading - yaw)))
    return pd, pt


def reduce_results() -> dict:
    state_files = sorted(STATE_OUT.glob("purpose-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    if len(state_files) != 48:
        raise RuntimeError(f"expected 48 recovered guard states, found {len(state_files)}")
    guard_states = [json.loads(path.read_text()) for path in state_files]
    guard_by = {(s["state_id"], r["candidate_index"]): r for s in guard_states for r in s["candidate_rows"]}
    ledger = [json.loads(line) for line in (V1 / "branch_labels.jsonl").read_text().splitlines()]
    route_rows = [json.loads(line) for line in (V2 / "route_intent_labels.jsonl").read_text().splitlines()]
    route_by = {r["branch_id"]: r for r in route_rows}
    manifest = json.loads((V1 / "state_manifest.json").read_text())
    state_manifest = {r["state_id"]: r for r in manifest["state_candidates"]}
    rows = []
    for source in ledger:
        sid, ci = source["state_id"], int(source["candidate_index"])
        route = route_by[f"{sid}:{ci:02d}"]
        h3, rh3 = source["horizons"]["3"], route["horizons"]["3"]
        state = state_manifest[sid]
        start_yaw = float(state["start_pose"][1])
        body = state.get("waypoint_body_xy")
        if body is None:
            scene_payload = json.loads((Path(state["scene_dir"]) / "manifest.json").read_text())
            graph = SceneGraph(parse_scene_manifest_dict(scene_payload))
            wx, wy = graph.cell_center(int(state["waypoint_path_cells"][2]))
            sx, sy = map(float, state["start_pose"][0])
            dx, dy = float(wx) - sx, float(wy) - sy
            body = [math.cos(start_yaw) * dx + math.sin(start_yaw) * dy,
                    -math.sin(start_yaw) * dx + math.cos(start_yaw) * dy]
        route_heading = float(rh3["route_heading_world_rad"])
        waypoint = [float(body[0]), float(body[1]), math.sin(route_heading - start_yaw), math.cos(route_heading - start_yaw)]
        guard = guard_by[(sid, ci)]
        rows.append({
            "state_id": sid, "candidate_index": ci, "family": source["family"], "split": route["split"],
            "unsafe": bool(h3["unsafe"]), "p_d": float(rh3["p_d"]), "p_theta": float(rh3["p_theta_rad"]),
            "admitted": bool(guard["admitted"]), "feasible_fraction": float(guard["feasible_fraction"]),
            "waypoint": waypoint, "post_slew": source["post_slew"],
        })

    # Frozen aggregate safety is authoritative.  Component fields are the
    # already-existing deterministic V2 replay sensitivity labels.
    component_by = {}
    component_mismatch = 0
    frozen_unsafe_by = {(r["state_id"], int(r["candidate_index"])): bool(r["horizons"]["3"]["unsafe"]) for r in ledger}
    for path in sorted((V2 / "replay").glob("purpose-*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            item = row["horizons"]["3"]
            key = (row["state_id"], int(row["candidate_index"]))
            component_by[key] = item["components"]
            replay_unsafe = bool(item.get("replay_path_unsafe", item.get("path_unsafe")))
            frozen_unsafe = bool(item.get("frozen_path_unsafe", frozen_unsafe_by[key]))
            component_mismatch += int(replay_unsafe != frozen_unsafe)
    if len(component_by) != 576:
        raise RuntimeError(f"expected 576 existing component rows, found {len(component_by)}")
    records = []
    for row in rows:
        comp = component_by[(row["state_id"], row["candidate_index"])]
        records.append(dict(row, **comp))

    splits = {name: {r["state_id"] for r in records if r["split"] == name} for name in ("fit", "calibration", "heldout")}
    heldout = [r for r in records if r["split"] == "heldout"]
    aggregate = {name: confusion([r for r in records if r["split"] == name]) for name in splits}
    aggregate["all"] = confusion(records)
    per_family = {family: confusion([r for r in heldout if r["family"] == family]) for family in sorted({r["family"] for r in heldout})}
    component_names = ("collision_or_disallowed_contact", "clearance_violation", "stuck", "fall", "unsafe_termination")
    per_component = {name: confusion([dict(r, component_truth=bool(r[name])) for r in heldout], "component_truth") for name in component_names}
    combination_counts = Counter("+".join(name for name in component_names if r[name]) or "none" for r in records)
    state_audit = []
    for sid in sorted(splits["heldout"], key=lambda x: int(x.split("-")[1])):
        q = [r for r in heldout if r["state_id"] == sid]
        admitted = [r for r in q if r["admitted"]]
        safe_admitted = [r for r in admitted if not r["unsafe"]]
        state_audit.append({
            "state_id": sid, "family": q[0]["family"], "safe_candidates": sum(not r["unsafe"] for r in q),
            "admitted_candidates": len(admitted), "admitted_safe_candidates": len(safe_admitted),
            "only_unsafe_admitted": bool(admitted and not safe_admitted), "no_admitted_candidates": not admitted,
        })
    h = aggregate["heldout"]
    family_collapse = any(
        (m["unsafe_rows"] and m["unsafe_recall"] == 0) or (m["safe_rows"] and m["safe_candidate_retention"] == 0)
        for m in per_family.values()
    )
    gate_checks = {
        "unsafe_recall_ge_0_95": h["unsafe_recall"] is not None and h["unsafe_recall"] >= 0.95,
        "unsafe_fnr_le_0_05": h["unsafe_false_negative_rate"] is not None and h["unsafe_false_negative_rate"] <= 0.05,
        "safe_retention_ge_0_50": h["safe_candidate_retention"] is not None and h["safe_candidate_retention"] >= 0.50,
        "six_of_eight_states_retain_safe": sum(s["admitted_safe_candidates"] > 0 for s in state_audit) >= 6,
        "no_state_only_unsafe_admitted": not any(s["only_unsafe_admitted"] for s in state_audit),
        "no_family_complete_collapse": not family_collapse,
    }
    qualified = all(gate_checks.values())

    nominal = [integrate(r["post_slew"], r["waypoint"]) for r in rows]
    nominal_pd = np.asarray([x[0] for x in nominal])
    nominal_pt = np.asarray([x[1] for x in nominal])
    admitted = np.asarray([r["admitted"] for r in rows], bool)
    planner = {
        "reached": qualified,
        "hold_brake": planner_eval("hold_brake", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "hold"),
        "lowest_index": planner_eval("lowest_index", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "lowest"),
        "unguarded_kinematic": planner_eval("unguarded_kinematic", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "unguarded"),
        "oracle_safety_kinematic": planner_eval("oracle_safety_kinematic", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "oracle_kinematic"),
        "oracle_best_safe": planner_eval("oracle_best_safe", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "oracle_best"),
        "runtime_guarded_kinematic": planner_eval("runtime_guarded_kinematic", rows, splits["heldout"], admitted, nominal_pd, nominal_pt, "guarded") if qualified else None,
    }
    planning_gate = None
    classification = "RUNTIME_SAFETY_GUARD_NO_GO_FOR_CANDIDATE_PLANNING"
    if qualified:
        guarded = planner["runtime_guarded_kinematic"]
        oracle_k = planner["oracle_safety_kinematic"]
        progress_fraction = guarded["mean_selected_distance_progress_m"] / oracle_k["mean_selected_distance_progress_m"] if oracle_k["mean_selected_distance_progress_m"] > 0 else None
        no_family_collapse = all(v["selected_safe_rate"] is not None and v["selected_safe_rate"] > 0 for v in guarded["per_family"].values())
        planning_gate = {
            "selects_no_unsafe": guarded["selected_unsafe_rate"] == 0,
            "retains_80pct_oracle_progress": progress_fraction is not None and progress_fraction >= 0.80,
            "normalized_regret_le_0_20": guarded["normalized_safe_progress_regret"] is not None and guarded["normalized_safe_progress_regret"] <= 0.20,
            "best_safe_top3_ge_0_75": guarded["best_safe_top3"] is not None and guarded["best_safe_top3"] >= 0.75,
            "false_abstentions_le_1": guarded["false_abstentions"] <= 1,
            "no_family_collapse": no_family_collapse,
            "oracle_progress_fraction": progress_fraction,
        }
        if all(v for k, v in planning_gate.items() if k != "oracle_progress_fraction"):
            classification = "KINEMATIC_RUNTIME_GUARD_PLANNING_SIGNAL"
        else:
            classification = "KINEMATIC_RUNTIME_GUARD_PLANNING_NO_SIGNAL"

    fixture = {
        "training_state_ids": ["purpose-0", "purpose-1"],
        "repeat_counts": {s["state_id"]: s["repeat_count"] for s in guard_states[:2]},
        "identical_verdicts_and_diagnostics": all(s["repeat_deterministic"] for s in guard_states[:2]),
        "identical_candidate_order": all(s["candidate_order_identical"] for s in guard_states[:2]),
        "zero_simulator_step_after_snapshot": all(s["zero_simulator_step_after_snapshot"] for s in guard_states[:2]),
        "pass": all(s["repeat_deterministic"] and s["candidate_order_identical"] and s["zero_simulator_step_after_snapshot"] for s in guard_states[:2]),
    }
    result = {
        "schema": "kinematic_route_with_runtime_safety_guard_v1",
        "source_commit": "026cd389deb5fa106b8a023530ac32e77013a34b",
        "preserved_terminal": "KINEMATIC_RESIDUAL_PLANNER_NO_SIGNAL",
        "bindings": {
            "state_manifest": sha(V1 / "state_manifest.json"),
            "branch_ledger": sha(V1 / "branch_labels.jsonl"),
            "route_labels": sha(V2 / "route_intent_labels.jsonl"),
            "runtime_guard_source": sha(ROOT / "scripts/benchmark_topo_nav_e2e.py"),
            "local_obstacle_source": sha(ROOT / "lewm/planning/local_obstacles.py"),
        },
        "guard_contract": GUARD,
        "guard_contract_digest": digest_json(GUARD),
        "guard_input_recovery": {
            "states": 48, "candidate_verdicts": 576,
            "method": "deterministic 40-block redrive to frozen pre-action snapshot; no candidate executed",
            "observation": "privileged manifest occupancy grid queried at frozen current simulator pose",
            "controller_history_restored": True, "future_outcome_access": False,
            "deployment_valid": False,
        },
        "determinism_fixture": fixture,
        "guard_discrimination": {"by_split": aggregate, "heldout_by_family": per_family, "heldout_by_component": per_component},
        "component_label_audit": {
            "source": "existing route-intent V2 deterministic replay sensitivity",
            "frozen_aggregate_remains_authoritative": True,
            "aggregate_replay_mismatch_rows": component_mismatch,
            "combination_counts_all_rows": dict(sorted(combination_counts.items())),
        },
        "heldout_state_audit": state_audit,
        "guard_qualification": {"passed": qualified, "checks": gate_checks},
        "planner_evaluation": planner,
        "planner_gate": planning_gate,
        "classification": classification,
        "model_training": False, "candidate_branch_execution": False, "predictor_opened": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "result.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default))
    print(json.dumps({"classification": classification, "guard_passed": qualified, "result_sha256": sha(target)}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=int)
    parser.add_argument("--reduce", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if args.state_index is not None:
        recover_state(args.state_index)
    elif args.reduce:
        result = reduce_results()
        result["reducer_runtime_s"] = time.time() - started
        target = OUT / "result.json"
        target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default))
        print(json.dumps({"classification": result["classification"], "result_sha256": sha(target)}, indent=2))
    else:
        parser.error("provide --state-index or --reduce")


if __name__ == "__main__":
    main()
