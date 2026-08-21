#!/usr/bin/env python3
"""Feasibility-aware Genesis narrowphase and sensor-geometry evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from lewm.safety import genesis_narrowphase_reconciliation_v1 as R
from scripts import evaluate_wide_geometry_score_composition_v1 as BASE

OUT = ROOT / ".generated/genesis_narrowphase_candidate_feasibility_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/genesis_narrowphase_candidate_feasibility_v1")
INDEX = OUT / "narrowphase_index.json"
PHYSICS_LEDGER = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/physics_rate_contact_proxy_reconciliation_v1/row_level_evidence_v1.npz")
PHYSICS_LEDGER_SHA = "3e5de8b6b4007f9ac066bb981e23f9fc59b28459caa23d93c9c222431b18b8ee"
OLD_INDEX = ROOT / ".generated/h1_articulated_swept_geometry_sufficiency_v1/articulated_geometry_index.json"
ENHANCED = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
WIDE_LEDGER = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
CONDITIONS = ("DEPTH_ARTICULATED_SWEEP", "LIDAR_ARTICULATED_SWEEP", "DEPTH_LIDAR_ARTICULATED_SWEEP")
SCORE_COLUMNS = (1, 2, 3)
FAMILIES = BASE.FAMILIES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as f: np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    return value


def events(trace):
    return R.event_spans(trace)


def load_all():
    if sha(PHYSICS_LEDGER) != PHYSICS_LEDGER_SHA:
        raise RuntimeError("physics-rate ledger SHA mismatch")
    index = json.loads(INDEX.read_text()); old_index = json.loads(OLD_INDEX.read_text())
    enhanced_index = json.loads(ENHANCED.read_text())
    with np.load(PHYSICS_LEDGER, allow_pickle=False) as f:
        physics = {k: np.asarray(f[k]) for k in f.files}
    with np.load(WIDE_LEDGER, allow_pickle=False) as f:
        wide = {k: np.asarray(f[k]) for k in f.files}
    wide_map = {str(x): i for i, x in enumerate(wide["branch_id"])}
    old_map = {r["state_id"]: r for r in old_index["state_records"]}
    enhanced_map = {r["state_id"]: r for r in enhanced_index["state_records"]}
    exact_map = {r["state_id"]: r for r in index["state_records"]}
    rows = []; state_arrays = {}
    for state in index["state_records"]:
        sid = state["state_id"]
        with np.load(state["shard_path"], allow_pickle=False) as f:
            exact = {k: np.asarray(f[k]) for k in f.files}
        with np.load(old_map[sid]["shard_path"], allow_pickle=False) as f:
            old = {k: np.asarray(f[k]) for k in f.files}
        state_arrays[sid] = {"exact": exact, "old": old}
    for i, bid_value in enumerate(physics["branch_id"]):
        bid = str(bid_value); sid = str(physics["state_id"][i]); ci = int(physics["candidate_index"][i]); wi = wide_map[bid]
        exact = state_arrays[sid]["exact"]
        if bool(physics["physics_rate_h1_contact"][i]) != bool(exact["frozen_contact"][ci].any()):
            raise RuntimeError(f"row alignment label mismatch {bid}")
        rows.append({
            "row_index": i, "branch_id": bid, "state_id": sid, "candidate_index": ci,
            "split": str(physics["split"][i]), "family": str(physics["family"][i]),
            "hard_contact": bool(physics["physics_rate_h1_contact"][i]),
            "stuck": bool(wide["stuck_labels"][wi, -1, 1]), "p_d": float(physics["p_d"][i]),
            "p_theta": float(physics["p_theta"][i]), "kinematic": np.asarray(physics["kinematic"][i], np.float64),
            "geometry_scores": np.asarray(physics["geometry_scores"][i], np.float64),
            "exact_contact": bool(exact["exact_contact"][ci].any()),
        })
    if len(rows) != 576 or len({r["branch_id"] for r in rows}) != 576:
        raise RuntimeError("row identity cardinality failure")
    return index, old_index, enhanced_index, enhanced_map, exact_map, physics, rows, state_arrays


def split_rows(rows, split):
    return [r for r in rows if r["split"] == split]


def per_family(rows, admitted):
    output = {}
    for family in FAMILIES:
        ids = [i for i, row in enumerate(rows) if row["family"] == family]
        output[family] = R.feasibility_metrics([rows[i] for i in ids], np.asarray(admitted)[ids])
    return output


def contact_metrics(rows, admitted, scores=None):
    labels = np.asarray([r["hard_contact"] for r in rows], bool); admitted = np.asarray(admitted, bool)
    result = R.feasibility_metrics(rows, admitted)
    result.update({
        "contact_recall": float((~admitted)[labels].mean()),
        "contact_false_negative_rate": float(admitted[labels].mean()),
        "contact_negative_retention": float(admitted[~labels].mean()),
        "admitted_contact_positive_count": int(np.sum(admitted & labels)),
        "admitted_contact_negative_count": int(np.sum(admitted & ~labels)),
    })
    if scores is not None:
        result["auc"] = BASE.auc(labels, scores); result["average_precision"] = BASE.average_precision(labels, scores)
    result["per_family"] = per_family(rows, admitted)
    return result


def sensor_gate(metric):
    family_ok = all(v["safe_candidate_available_states"] == 0 or (v["states_retaining_contact_negative"] > 0 and v["selected_contact_count"] == 0) for v in metric["per_family"].values())
    checks = {
        "auc_ge_0_90": metric["auc"] >= .90, "ap_ge_0_90": metric["average_precision"] >= .90,
        "recall_ge_0_95": metric["contact_recall"] >= .95, "fnr_le_0_05": metric["contact_false_negative_rate"] <= .05,
        "retention_ge_0_50": metric["contact_negative_retention"] >= .50,
        "safe_states_retained_ge_0_80": metric["safe_state_retention_rate"] >= .80,
        "zero_selected_contact": metric["selected_contact_count"] == 0,
        "false_abstention_rate_le_0_15": metric["false_abstention_rate"] <= .15,
        "progress_ge_0_80": metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"] >= .80,
        "regret_le_0_20": metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"] <= .20,
        "top3_ge_0_75": metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"] >= .75,
        "no_safe_correct_abstention_ge_0_90": metric["correct_abstention_rate_no_safe"] is None or metric["correct_abstention_rate_no_safe"] >= .90,
        "no_unsafe_move_no_safe": metric["unsafe_movements_no_safe"] == 0, "no_family_collapse": family_ok,
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_threshold(rows, scores, threshold):
    admitted = np.asarray(scores) < threshold
    return contact_metrics(rows, admitted, scores), admitted


def threshold_values(scores):
    x = np.asarray(scores, np.float64)
    return np.concatenate(([np.nextafter(x.min(), -np.inf)], np.unique(x), [np.nextafter(x.max(), np.inf)]))


def select_sensor_threshold(rows, scores):
    candidates = []; thresholds = threshold_values(scores)
    for threshold in thresholds:
        metric, _ = evaluate_threshold(rows, scores, threshold)
        if metric["contact_recall"] + 1e-15 < .95:
            continue
        regret = metric["normalized_route_progress_regret"]
        key = (metric["states_retaining_contact_negative"], metric["contact_negative_retention"],
               -metric["selected_contact_count"], metric["correct_abstentions_no_safe"],
               metric["mean_selected_route_progress_m"], -(1e9 if regret is None else regret),
               -metric["false_abstentions"], metric["best_contact_negative_top3"] or -1.0, -float(threshold))
        candidates.append((key, float(threshold), metric))
    if not candidates:
        raise RuntimeError("no sensor threshold reaches calibration recall")
    best = max(candidates, key=lambda x: x[0])
    return best[1], {"thresholds": len(thresholds), "eligible": len(candidates), "metrics": best[2]}


def sensor_frontier(rows, scores, name):
    values = defaultdict(list); gates = 0
    for threshold in threshold_values(scores):
        metric, _ = evaluate_threshold(rows, scores, threshold); gate = sensor_gate(metric); gates += gate["passed"]
        for key, value in {
            "risk_threshold": threshold, "clearance_threshold_m": -threshold, "recall": metric["contact_recall"],
            "retention": metric["contact_negative_retention"], "retained_states": metric["states_retaining_contact_negative"],
            "selected_contact": metric["selected_contact_count"], "false_abstention_rate": metric["false_abstention_rate"],
            "progress_fraction": metric["oracle_progress_fraction"], "regret": metric["normalized_route_progress_regret"],
            "top3": metric["best_contact_negative_top3"], "complete_gate": gate["passed"],
        }.items(): values[key].append(np.nan if value is None else value)
    arrays = {k: np.asarray(v) for k, v in values.items()}; path = CACHE / "frontiers" / f"{name.lower()}_heldout_v1.npz"
    atomic_npz(path, **arrays); recall = arrays["recall"] >= .95; zero = arrays["selected_contact"] == 0
    def best(key, mask, maximum=True):
        x = arrays[key][mask & np.isfinite(arrays[key])]
        return None if not len(x) else float(np.max(x) if maximum else np.min(x))
    return {"thresholds": len(arrays["risk_threshold"]), "complete_gate_points": int(gates), "any_complete_gate": bool(gates),
            "maximum_negative_retention_at_recall_ge_0_95": best("retention", recall),
            "maximum_safe_states_retained_at_recall_ge_0_95": best("retained_states", recall),
            "maximum_progress_fraction_with_zero_selected_contact": best("progress_fraction", zero),
            "minimum_regret_at_recall_ge_0_95_and_zero_selected": best("regret", recall & zero, False),
            "path": str(path), "sha256": sha(path)}


def exact_reproduction(index, rows, state_arrays):
    frozen_all = []; native_all = []; exact_all = []; approximate_all = []
    branch_ref = []; branch_query = []; first_errors = []; event_match = 0; pair_both = pair_match = 0; residual = []
    for state in index["state_records"]:
        sid = state["state_id"]; data = state_arrays[sid]["exact"]
        frozen = data["frozen_contact"].astype(bool); native = data["native_contact"].astype(bool); exact = data["exact_contact"].astype(bool)
        frozen_all.append(frozen.ravel()); native_all.append(native.ravel()); exact_all.append(exact.ravel()); approximate_all.append(data["approx_clearance"].ravel() <= 0)
        for ci in range(12):
            branch_ref.append(bool(frozen[ci].any())); branch_query.append(bool(exact[ci].any()))
            error = R.first_contact_error(frozen[ci], exact[ci])
            if error is not None and (frozen[ci].any() or exact[ci].any()): first_errors.append(error)
            event_match += events(frozen[ci]) == events(exact[ci])
            both = frozen[ci] & exact[ci]
            pair_both += int(both.sum())
            pair_match += int(np.sum(both & (data["native_robot_link"][ci] == data["exact_robot_link"][ci]) & (data["native_other_link"][ci] == data["exact_other_link"][ci])))
            for step in np.flatnonzero(frozen[ci] != exact[ci]):
                native_active = bool(frozen[ci, step]); exact_active = bool(exact[ci, step]); penetration = float(data["exact_penetration"][ci, step])
                if exact_active and not native_active:
                    category = "NUMERICAL_TOLERANCE" if penetration <= 1e-5 else "DYNAMIC_OR_CONSTRAINT_SOLVER_DEPENDENCE"
                elif native_active and not exact_active:
                    category = "HISTORY_DEPENDENT_COLLIDER_STATE"
                else:
                    category = "UNRESOLVED"
                source_prefix = "native" if native_active else "exact"
                residual.append({"branch_id": f"{sid}:{ci:02d}", "physics_step": int(step),
                    "robot_link": state["link_names"].get(str(int(data[f"{source_prefix}_robot_link"][ci, step])), "unresolved"),
                    "environment_object": state["object_names"].get(str(int(data[f"{source_prefix}_other_link"][ci, step])), "unresolved"),
                    "solver_verdict": native_active, "exact_query_verdict": exact_active,
                    "signed_separation_m": None if not np.isfinite(penetration) else -penetration,
                    "penetration_m": None if not np.isfinite(penetration) else penetration,
                    "contact_margin_m": 0.0, "manifold_status": int(data["exact_count"][ci, step]) > 0,
                    "approximate_query_verdict": bool(data["approx_clearance"][ci, step] <= 0), "classification": category})
    frozen = np.concatenate(frozen_all); native = np.concatenate(native_all); exact = np.concatenate(exact_all); approximate = np.concatenate(approximate_all)
    native_confusion = R.binary_confusion(frozen, native); exact_confusion = R.binary_confusion(frozen, exact); approximate_confusion = R.binary_confusion(frozen, approximate)
    branch_confusion = R.binary_confusion(branch_ref, branch_query)
    return {"native_replay": native_confusion, "exact_query": exact_confusion, "prior_approximate_query_at_zero": approximate_confusion,
            "branch_level": branch_confusion, "branch_agreement": branch_confusion["agreement"],
            "first_contact_step_error": {"count": len(first_errors), "median_signed_steps": None if not first_errors else float(np.median(first_errors)),
                "median_absolute_steps": None if not first_errors else float(np.median(np.abs(first_errors))), "maximum_absolute_steps": None if not first_errors else int(np.max(np.abs(first_errors)))},
            "responsible_pair_agreement": None if pair_both == 0 else pair_match / pair_both,
            "event_count_exact_branches": int(event_match), "event_count_agreement": event_match / 576,
            "residual_disagreements": len(residual), "residual_classification": dict(Counter(x["classification"] for x in residual)), "residual_inventory": residual}


def candidate_audit(rows, index, enhanced_index, enhanced_map, state_arrays):
    output = {}; class_counts = Counter(); reason_counts = Counter(); no_safe_by_split = Counter(); no_safe_by_family = Counter()
    row_by_state = defaultdict(list)
    for row in rows: row_by_state[row["state_id"]].append(row)
    for state in index["state_records"]:
        sid = state["state_id"]; state_rows = sorted(row_by_state[sid], key=lambda r: r["candidate_index"])
        labels = np.asarray([r["hard_contact"] for r in state_rows]); safe = int((~labels).sum())
        if safe:
            output[sid] = {"feasibility": "SAFE_CANDIDATE_AVAILABLE", "contact_negative_candidates": safe}; continue
        no_safe_by_split[state["split"]] += 1; no_safe_by_family[state["family"]] += 1
        data = state_arrays[sid]; exact = data["exact"]; old = data["old"]
        first = [int(np.flatnonzero(exact["frozen_contact"][ci])[0]) for ci in range(12)]
        divergence0 = R.candidate_divergence_step(old["link_transform"]); divergence = None if divergence0 is None else divergence0 + 1
        enhanced = enhanced_map[sid]; branch_map = {int(b["candidate_index"]): b for b in enhanced["branches"]}
        commands = np.asarray([branch_map[ci]["post_slew"][0] for ci in range(12)], np.float64)
        requested = np.asarray([branch_map[ci]["requested"][0] for ci in range(12)], np.float64)
        command_tick = R.command_divergence_tick(commands)
        with np.load(enhanced["shard_path"], allow_pickle=False) as f: action_control = np.asarray(f["action_control"])
        prior = action_control[0, 0, 3:6].astype(float)
        hold_contacts = bool(labels[11]); reverse_contacts = bool(labels[10])
        slew_limited = bool(not np.allclose(commands[11, 0], requested[11, 0], atol=1e-8) or not np.allclose(commands[10, 0], requested[10, 0], atol=1e-8))
        # A label-independent approach-speed diagnostic from the predeclared
        # prior analytical clearance: speed halves over a five-step window.
        reductions = []
        for ci in range(12):
            clearance = np.asarray(exact["approx_clearance"][ci], np.float64); stop = max(6, first[ci])
            velocity = -np.diff(clearance[:stop]) / .002
            initial = float(np.median(velocity[:5])) if len(velocity) >= 5 else 0.0
            later = float(np.min([np.median(velocity[j:j+5]) for j in range(5, max(6, len(velocity)-4))], initial=initial)) if len(velocity) >= 10 else initial
            reductions.append(bool(initial > 0 and later <= .5 * initial))
        candidate_effect = (max(first) - min(first) >= 5) or any(reductions)
        classification = R.classify_no_safe_state(boundary_contact=bool(state["boundary_native_contact"]), first_contact_steps=first,
            trajectory_divergence_step=divergence, avoiding_response_step=divergence, candidate_effect_evidence=candidate_effect)
        class_counts[classification] += 1
        if state["boundary_native_contact"]: cause = "initial-state eligibility"
        elif divergence is None or max(first) <= divergence: cause = "commitment latency"
        elif slew_limited and min(first) <= divergence + 50: cause = "slew-limiter authority"
        elif candidate_effect: cause = "candidate-bank coverage"
        else: cause = "unresolved physics"
        reason_counts[cause] += 1
        output[sid] = {"feasibility": "NO_SAFE_CANDIDATE_AVAILABLE", "contact_negative_candidates": 0,
            "family": state["family"], "split": state["split"], "prior_applied_command_vx_vy_yaw": prior.tolist(),
            "first_candidate_command_divergence_tick": command_tick,
            "first_material_base_or_link_divergence_physics_step": divergence,
            "first_contact_step_by_candidate": first, "contact_before_trajectory_divergence_by_candidate": [divergence is None or x <= divergence for x in first],
            "hold_contacts": hold_contacts, "reverse_contacts": reverse_contacts,
            "any_candidate_reduces_approach_speed": any(reductions), "approach_speed_reduction_candidates": [i for i, x in enumerate(reductions) if x],
            "slew_limiter_prevents_immediate_hold_or_reverse": slew_limited,
            "begins_in_native_contact": bool(state["boundary_native_contact"]),
            "begins_inside_exact_contact_envelope": bool(np.all(exact["exact_contact"][:, 0])),
            "classification": classification, "causal_factor": cause}
    total_by_split = {s: len({r["state_id"] for r in rows if r["split"] == s}) for s in ("calibration", "heldout")}
    adequacy = {s: {"states": total_by_split[s], "no_safe_states": no_safe_by_split[s], "safe_candidate_fraction": (total_by_split[s]-no_safe_by_split[s])/total_by_split[s]} for s in total_by_split}
    adequate = all(v["safe_candidate_fraction"] >= .90 for v in adequacy.values())
    if reason_counts["initial-state eligibility"] >= max(reason_counts.values(), default=0): correction = "state eligibility requiring a collision-free response envelope"
    elif reason_counts["commitment latency"] + reason_counts["slew-limiter authority"] >= reason_counts["candidate-bank coverage"]: correction = "true emergency brake"
    else: correction = "dedicated retreat candidates"
    return {"state_inventory": output, "no_safe_classification_counts": dict(class_counts), "causal_factor_counts": dict(reason_counts),
            "no_safe_by_family": dict(no_safe_by_family), "split_adequacy": adequacy,
            "classification": "CANDIDATE_BANK_H1_SAFETY_COVERAGE_ADEQUATE" if adequate else "CANDIDATE_BANK_H1_SAFETY_COVERAGE_NO_GO",
            "smallest_prospective_correction": None if adequate else correction}


def full_geometry_gate(reproduction, metric):
    checks = {
        "branch_agreement_ge_0_995": reproduction["branch_agreement"] >= .995,
        "step_sensitivity_ge_0_995": reproduction["exact_query"]["sensitivity"] >= .995,
        "step_specificity_ge_0_995": reproduction["exact_query"]["specificity"] >= .995,
        "first_step_median_error_le_1": reproduction["first_contact_step_error"]["median_absolute_steps"] <= 1,
        "safe_states_retained_ge_0_95": metric["safe_state_retention_rate"] >= .95,
        "zero_selected_contact": metric["selected_contact_count"] == 0,
        "false_abstention_rate_le_0_05": metric["false_abstention_rate"] <= .05,
        "progress_ge_0_95": metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"] >= .95,
        "regret_le_0_10": metric["normalized_route_progress_regret"] is not None and metric["normalized_route_progress_regret"] <= .10,
        "top3_ge_0_90": metric["best_contact_negative_top3"] is not None and metric["best_contact_negative_top3"] >= .90,
        "no_safe_correct_abstention": metric["correct_abstention_rate_no_safe"] is None or metric["correct_abstention_rate_no_safe"] == 1.0,
        "no_safe_no_execution": metric["unsafe_movements_no_safe"] == 0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    started = time.time(); fixture = R.fixture_payload()
    if not fixture["pass"] or not fixture["byte_identical_regeneration"]: raise RuntimeError("fixture failure")
    index, old_index, enhanced_index, enhanced_map, exact_map, physics, rows, state_arrays = load_all()
    reproduction = exact_reproduction(index, rows, state_arrays)
    calibration_rows = split_rows(rows, "calibration"); heldout_rows = split_rows(rows, "heldout")
    exact_cal = contact_metrics(calibration_rows, [not r["exact_contact"] for r in calibration_rows])
    exact_held = contact_metrics(heldout_rows, [not r["exact_contact"] for r in heldout_rows])
    exact_held["per_family"] = per_family(heldout_rows, [not r["exact_contact"] for r in heldout_rows])
    full_gate = full_geometry_gate(reproduction, exact_held)
    audit = candidate_audit(rows, index, enhanced_index, enhanced_map, state_arrays)
    sensor_results = {}; sensor_frontiers = {}
    if full_gate["passed"]:
        for name, column in zip(CONDITIONS, SCORE_COLUMNS, strict=True):
            cal_scores = np.asarray([r["geometry_scores"][column] for r in calibration_rows]); held_scores = np.asarray([r["geometry_scores"][column] for r in heldout_rows])
            threshold, selection = select_sensor_threshold(calibration_rows, cal_scores)
            held_metric, admitted = evaluate_threshold(heldout_rows, held_scores, threshold); gate = sensor_gate(held_metric); held_metric["gate"] = gate
            sensor_results[name] = {"calibration": {"risk_threshold": threshold, "clearance_threshold_m": -threshold, "selection": selection}, "heldout": held_metric}
            sensor_frontiers[name] = sensor_frontier(heldout_rows, held_scores, name)
    passing = [name for name, value in sensor_results.items() if value["heldout"]["gate"]["passed"]]
    if full_gate["passed"]:
        primary = "GENESIS_EXACT_GEOMETRY_SUFFICIENT"
    elif reproduction["exact_query"]["sensitivity"] >= .995 and reproduction["exact_query"]["specificity"] >= .995:
        primary = "GENESIS_EXACT_GEOMETRY_QUERY_UNRESOLVED"
    elif reproduction["residual_disagreements"] and all(k in {"HISTORY_DEPENDENT_COLLIDER_STATE", "DYNAMIC_OR_CONSTRAINT_SOLVER_DEPENDENCE"} for k in reproduction["residual_classification"]):
        primary = "ARTICULATED_CONTACT_DYNAMICS_REQUIRED"
    else:
        primary = "GENESIS_EXACT_GEOMETRY_QUERY_UNRESOLVED"
    secondary = [audit["classification"]]
    query_reproduction_passed = (reproduction["branch_agreement"] >= .995 and reproduction["exact_query"]["sensitivity"] >= .995
        and reproduction["exact_query"]["specificity"] >= .995 and reproduction["first_contact_step_error"]["median_absolute_steps"] <= 1)
    if query_reproduction_passed and not full_gate["checks"]["top3_ge_0_90"]:
        secondary.append("KINEMATIC_RANKING_LIMITATION")
    if primary == "GENESIS_EXACT_GEOMETRY_SUFFICIENT":
        mapping = {CONDITIONS[0]: "DEPTH_GEOMETRY_SIGNAL", CONDITIONS[1]: "LIDAR_GEOMETRY_SIGNAL", CONDITIONS[2]: "FUSED_GEOMETRY_SIGNAL"}
        secondary.extend(mapping[x] for x in passing)
        if not passing: secondary.append("SENSOR_GEOMETRY_COVERAGE_NO_GO")
    causes = audit["causal_factor_counts"]
    if causes.get("initial-state eligibility", 0): secondary.append("PRE_EXISTING_CONTACT_STATE_SELECTION_FAILURE")
    if causes.get("commitment latency", 0): secondary.append("CONTROL_RESPONSE_LATENCY_FAILURE")
    if causes.get("slew-limiter authority", 0): secondary.append("SLEW_LIMITED_BRAKING_FAILURE")
    # Immutable row ledger; physics-step detail remains in SHA-bound state shards.
    exact_branch = np.asarray([r["exact_contact"] for r in rows], np.uint8)
    selected_exact = {x["state_id"]: -1 if x["selected_candidate"] is None else x["selected_candidate"] for x in exact_held["per_state"]}
    ledger = CACHE / "row_level_evidence_v1.npz"
    sensor_thresholds = np.asarray([sensor_results[x]["calibration"]["risk_threshold"] for x in CONDITIONS], np.float64) if sensor_results else np.empty(0)
    sensor_admitted = np.stack([[r["geometry_scores"][c] < sensor_results[n]["calibration"]["risk_threshold"] for n, c in zip(CONDITIONS, SCORE_COLUMNS, strict=True)] for r in rows]).astype(np.uint8) if sensor_results else np.empty((len(rows), 0), np.uint8)
    atomic_npz(ledger, branch_id=np.asarray([r["branch_id"] for r in rows]), state_id=np.asarray([r["state_id"] for r in rows]),
        candidate_index=np.asarray([r["candidate_index"] for r in rows], np.int16), split=np.asarray([r["split"] for r in rows]), family=np.asarray([r["family"] for r in rows]),
        physics_contact=np.asarray([r["hard_contact"] for r in rows], np.uint8), exact_contact=exact_branch,
        approximate_and_sensor_scores=np.stack([r["geometry_scores"] for r in rows]).astype(np.float32), sensor_condition_names=np.asarray(CONDITIONS),
        sensor_risk_thresholds=sensor_thresholds, sensor_admitted=sensor_admitted,
        p_d=np.asarray([r["p_d"] for r in rows], np.float32), p_theta=np.asarray([r["p_theta"] for r in rows], np.float32),
        kinematic=np.stack([r["kinematic"] for r in rows]).astype(np.float32),
        exact_selected_candidate=np.asarray([selected_exact.get(r["state_id"], -1) if r["split"] == "heldout" else -1 for r in rows], np.int16))
    if primary == "GENESIS_EXACT_GEOMETRY_SUFFICIENT" and passing:
        next_experiment = "CANDIDATE_CONDITIONED_H1_GEOMETRIC_STATE_PREDICTOR_V1"
    elif primary == "GENESIS_EXACT_GEOMETRY_SUFFICIENT" and audit["classification"].endswith("NO_GO"):
        next_experiment = "H1_SAFE_ACTION_SET_SUCCESSOR_V1; qualify a true emergency brake under oracle physics-rate contact before learned prediction resumes"
    elif primary == "GENESIS_EXACT_GEOMETRY_SUFFICIENT":
        next_experiment = "prospectively change sensor coverage before another learned contact model"
    elif primary == "ARTICULATED_CONTACT_DYNAMICS_REQUIRED":
        next_experiment = "ARTICULATED_CONTACT_DYNAMICS_STATE_V1"
    else:
        next_experiment = "persist the missing Genesis collider/contact-cache state or bind a supported exact query API; do not train a dynamics model"
    runtime = {"evaluation_s": time.time() - started, "replay_wall_s": index["parallel_wall_runtime_s"], "replay_compute_s": index["runtime_compute_s"],
               "replayed_branches": index["replayed_branches"], "model_training": 0, "learned_inference": 0}
    storage_new = index["storage_bytes"] + ledger.stat().st_size + sum(Path(x["path"]).stat().st_size for x in sensor_frontiers.values())
    result = {"schema": "genesis_narrowphase_candidate_feasibility_v1_result", "experiment": "GENESIS_NARROWPHASE_AND_CANDIDATE_FEASIBILITY_V1",
        "source_commit": "0d490eb7651254c15ace65582cef06be6d007617", "claim_boundary": "H1_ANY_PHYSICS_STEP_DISALLOWED_CONTACT is a simulated contact/separation proxy only",
        "bindings": {"physics_rate_ledger_sha256": PHYSICS_LEDGER_SHA, "narrowphase_index_digest": index["content_digest"], "states": 48, "branches": 576, "physics_steps": 144000},
        "fixture": fixture, "exact_genesis_contract": index["contract"], "reproduction": reproduction,
        "feasibility": audit, "exact_geometry": {"calibration": exact_cal, "heldout": exact_held, "gate": full_gate},
        "sensor_geometry": sensor_results, "sensor_frontiers": sensor_frontiers,
        "primary_classification": primary, "secondary_classifications": secondary,
        "geometry_query_reconstruction_status": "RESOLVED" if query_reproduction_passed else "UNRESOLVED",
        "classification_protocol_note": ("Exact contact reproduction passes, but the supplied full upper-bound gate fails only the independent kinematic best-safe-top-3 requirement; therefore the strict GENESIS_EXACT_GEOMETRY_SUFFICIENT terminal is unavailable, while neither residual dynamics nor a missing query API is supported." if query_reproduction_passed and not full_gate["passed"] else None),
        "conditional_sensor_geometry_status": "EVALUATED" if full_gate["passed"] else "NOT_REACHED_BECAUSE_FULL_UPPER_BOUND_GATE_FAILED",
        "articulated_contact_dynamics_state_v1_justified": primary == "ARTICULATED_CONTACT_DYNAMICS_REQUIRED",
        "next_experiment": next_experiment, "row_level_evidence": {"path": str(ledger), "sha256": sha(ledger), "bytes": ledger.stat().st_size},
        "physics_step_evidence": {"state_shards": 48, "aggregate_bytes": index["storage_bytes"], "index_path": str(INDEX), "index_sha256": sha(INDEX)},
        "runtime": runtime, "storage": {"new_bytes": storage_new},
        "confirmations": {"model_training": False, "learned_model_inference": False, "jepa_access": False, "new_panel_or_identity": False,
            "replay_only_for_missing_narrowphase_evidence": True, "memory_navigation_novelty": False}}
    result = jsonable(result); result["content_digest"] = R.digest(result); atomic_json(OUT / "result.json", result)
    print(json.dumps({"primary": primary, "secondary": secondary, "step_exact": reproduction["exact_query"], "branch_agreement": reproduction["branch_agreement"],
        "heldout_feasibility": {k: exact_held[k] for k in ("safe_candidate_available_states", "no_safe_candidate_available_states", "safe_state_retention_rate", "selected_contact_count", "oracle_progress_fraction", "normalized_route_progress_regret", "best_contact_negative_top3")},
        "candidate_bank": audit["classification"], "sensor_passes": passing, "ledger_sha256": result["row_level_evidence"]["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
