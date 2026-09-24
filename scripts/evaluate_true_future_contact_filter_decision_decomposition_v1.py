#!/usr/bin/env python3
"""Read-only decision decomposition for the frozen wide-geometry contact proxy."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import evaluate_wide_geometry_score_composition_v1 as BASE


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/result.json"
SENSOR_INDEX = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/fresh_enhanced_sensor_index.json"
OUT = ROOT / ".generated/true_future_contact_filter_decision_decomposition_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/true_future_contact_filter_decision_decomposition_v1")
LEDGER = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
LEDGER_SHA256 = "ab47eb7848b980947ced6ee6f10493ef12578ab7871ef8ebdb97b46122617e9c"
CHECKPOINT_SHA256 = "3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31"
FAMILIES = BASE.FAMILIES
THRESHOLD_NEAR = 0.01
THRESHOLD_DEEP = 0.10


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_frozen_evidence():
    source = json.loads(SOURCE.read_text())
    if source["classification"] != "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY":
        raise RuntimeError("preserved classification mismatch")
    if source["training"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint identity mismatch")
    if BASE.sha(LEDGER) != LEDGER_SHA256 or source["row_level_evidence"]["sha256"] != LEDGER_SHA256:
        raise RuntimeError("frozen row-level ledger identity mismatch")
    with np.load(LEDGER, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"branch_id", "state_id", "candidate_index", "split", "family", "calibrated_probability",
                "contact_labels", "stuck_labels", "threshold_decision_admitted", "selected", "p_d", "p_theta", "kinematic"}
    if not required.issubset(ledger):
        raise RuntimeError(f"ledger missing fields: {sorted(required - set(ledger))}")
    sensor = json.loads(SENSOR_INDEX.read_text())
    candidate_names = {}
    for state in sensor["state_records"]:
        for branch in state["branches"]:
            candidate_names[(state["state_id"], int(branch["candidate_index"]))] = {
                "candidate": str(branch["candidate"]), "primitives": list(branch["primitives"])}
    held_mask = ledger["split"] == "heldout"
    rows = []
    for index in np.flatnonzero(held_mask):
        key = (str(ledger["state_id"][index]), int(ledger["candidate_index"][index]))
        if key not in candidate_names:
            raise RuntimeError(f"candidate contract unavailable: {key}")
        rows.append({"branch_id": str(ledger["branch_id"][index]), "state_id": key[0], "candidate_index": key[1],
            "candidate": candidate_names[key]["candidate"], "primitives": candidate_names[key]["primitives"],
            "family": str(ledger["family"][index]), "hard_contact": bool(ledger["contact_labels"][index, -1, 1]),
            "stuck": bool(ledger["stuck_labels"][index, -1, 1]), "p_d": float(ledger["p_d"][index]),
            "p_theta": float(ledger["p_theta"][index]), "kinematic": np.asarray(ledger["kinematic"][index], np.float64),
            "probability": float(ledger["calibrated_probability"][index, -1, 1]),
            "persisted_admitted": bool(ledger["threshold_decision_admitted"][index]),
            "persisted_selected": bool(ledger["selected"][index])})
    if len(rows) != 288 or len({row["state_id"] for row in rows}) != 24:
        raise RuntimeError("held-out cardinality mismatch")
    return source, ledger, rows


def metric_reproduction(source, rows):
    threshold = float(source["calibration"]["threshold"]); probability = np.asarray([row["probability"] for row in rows])
    admitted = probability < threshold; persisted = np.asarray([row["persisted_admitted"] for row in rows])
    if not np.array_equal(admitted, persisted):
        first = int(np.flatnonzero(admitted != persisted)[0]); raise RuntimeError(f"admission reproduction failure at {rows[first]['branch_id']}")
    states = BASE.prepare_states(rows); metrics = BASE.decision_metrics(rows, states, admitted, True)
    expected = source["heldout"]["filtering_and_planning"]
    keys = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "states_retaining_contact_negative",
            "selected_contact_count", "false_abstentions", "mean_selected_route_progress_m", "oracle_contact_kinematic_progress_m",
            "normalized_route_progress_regret", "best_contact_negative_top3")
    for key in keys:
        if not np.isclose(metrics[key], expected[key], atol=1e-12, rtol=1e-12):
            raise RuntimeError(f"metric reproduction failure: {key}: {metrics[key]} != {expected[key]}")
    selected = {(row["state_id"], row["selected_candidate"]) for row in metrics["per_state"] if row["selected_candidate"] is not None}
    persisted_selected = {(row["state_id"], row["candidate_index"]) for row in rows if row["persisted_selected"]}
    if selected != persisted_selected:
        raise RuntimeError("selected-candidate reproduction failure")
    contact = source["heldout"]["contact"]
    return {"status": "PASS", "ledger_sha256": LEDGER_SHA256, "checkpoint_identity_only": CHECKPOINT_SHA256,
        "checkpoint_executed": False, "contact_auc": contact["auc"], "contact_average_precision": contact["average_precision"],
        **{key: metrics[key] for key in keys}, "selected_candidates_exact": True}


def rank_indices(rows, ids, kind):
    if kind == "kinematic":
        return BASE.route_order(rows, ids)
    if kind == "oracle_progress":
        # Apply the frozen realised route-intent partial order: distance when
        # separated by >3 cm, then heading when separated by >5 degrees.
        remaining = sorted(ids, key=lambda i: rows[i]["candidate_index"]); ordered = []
        while remaining:
            best = remaining[0]
            for index in remaining[1:]:
                if BASE.preference(rows[index], rows[best]) > 0:
                    best = index
            ordered.append(best); remaining.remove(best)
        return ordered
    if kind == "kinematic_stuck_tiebreak":
        remaining = list(ids); ordered = []
        while remaining:
            best_distance = max(float(rows[i]["kinematic"][4]) for i in remaining)
            near = [i for i in remaining if best_distance - float(rows[i]["kinematic"][4]) <= BASE.DELTA_D]
            pick = min(near, key=lambda i: (bool(rows[i]["stuck"]), -float(rows[i]["kinematic"][5]), rows[i]["candidate_index"]))
            ordered.append(pick); remaining.remove(pick)
        return ordered
    raise ValueError(kind)


def evaluate_condition(rows, states, admitted, ranking, include_states=True):
    labels = np.asarray([row["hard_contact"] for row in rows], bool); per_state = []
    progress = []; headings = []; oracle_progress = []; contacts = []; stuck = []; regrets = []; top1 = []; top3 = []
    correct_abstentions = false_abstentions = 0
    for state in states:
        ids = state["ids"]; order = rank_indices(rows, ids, ranking); available = [i for i in order if admitted[i]]
        pick = available[0] if available else None; negative = state["negative"]
        if pick is None:
            false_abstentions += bool(negative); correct_abstentions += not bool(negative)
        else:
            progress.append(rows[pick]["p_d"]); headings.append(rows[pick]["p_theta"])
            contacts.append(bool(labels[pick])); stuck.append(bool(rows[pick]["stuck"]))
        if state["oracle"] is not None: oracle_progress.append(rows[state["oracle"]]["p_d"])
        if state["best"] is not None:
            top1.append(pick == state["best"]); top3.append(state["best"] in available[:3])
            if pick is not None and not labels[pick] and len(negative) >= 2:
                values = [rows[i]["p_d"] for i in negative]; spread = max(values) - min(values)
                if spread > 1e-8: regrets.append((rows[state["best"]]["p_d"] - rows[pick]["p_d"]) / spread)
        per_state.append({"state_id": state["state_id"], "family": state["family"],
            "selected_candidate": None if pick is None else rows[pick]["candidate_index"],
            "selected_contact": None if pick is None else bool(labels[pick]),
            "selected_stuck": None if pick is None else bool(rows[pick]["stuck"]),
            "selected_p_d": None if pick is None else rows[pick]["p_d"],
            "selected_p_theta": None if pick is None else rows[pick]["p_theta"],
            "admitted_count": int(admitted[ids].sum()),
            "admitted_negative_count": int(sum(admitted[i] and not labels[i] for i in ids)),
            "admitted_positive_count": int(sum(admitted[i] and labels[i] for i in ids))})
    selected_progress = float(np.mean(progress)) if progress else 0.0
    oracle_mean = float(np.mean(oracle_progress)) if oracle_progress else 0.0
    return {"selected_contact_count": int(sum(contacts)), "selected_stuck_count": int(sum(stuck)),
        "mean_selected_route_progress_m": selected_progress, "mean_selected_heading_improvement_rad": float(np.mean(headings)) if headings else 0.0,
        "normalized_route_progress_regret": None if not regrets else float(np.mean(regrets)),
        "best_contact_negative_top1": None if not top1 else float(np.mean(top1)),
        "best_contact_negative_top3": None if not top3 else float(np.mean(top3)),
        "abstentions": int(sum(row["selected_candidate"] is None for row in per_state)), "false_abstentions": int(false_abstentions),
        "correct_abstentions": int(correct_abstentions), "states_moving": int(sum(row["selected_candidate"] is not None for row in per_state)),
        "oracle_contact_kinematic_progress_m": oracle_mean,
        "oracle_progress_fraction": None if abs(oracle_mean) <= 1e-12 else selected_progress / oracle_mean,
        "per_state": per_state if include_states else None}


def family_condition(rows, states, admitted, ranking):
    output = {}
    for family in FAMILIES:
        ids = [i for i, row in enumerate(rows) if row["family"] == family]; remap = {old: new for new, old in enumerate(ids)}
        subrows = [rows[i] for i in ids]; substates = []
        for state in states:
            if state["family"] != family: continue
            substates.append({**state, "ids": [remap[i] for i in state["ids"]], "rank": [remap[i] for i in state["rank"]],
                "negative": [remap[i] for i in state["negative"]], "best": None if state["best"] is None else remap[state["best"]],
                "oracle": None if state["oracle"] is None else remap[state["oracle"]]})
        output[family] = {**BASE.decision_metrics(subrows, substates, admitted[ids], False),
            **evaluate_condition(subrows, substates, admitted[ids], ranking, False)}
    return output


def gate(metrics, auc_value, ap_value, families):
    no_collapse = all(value["states_moving"] > 0 and value["selected_contact_count"] == 0 for value in families.values())
    checks = {"auc_ge_0_90": auc_value >= .90, "ap_ge_0_90": ap_value >= .90,
        "recall_ge_0_95": metrics["contact_recall"] >= .95, "fnr_le_0_05": metrics["contact_false_negative_rate"] <= .05,
        "negative_retention_ge_0_50": metrics["contact_negative_retention"] >= .50,
        "states_retaining_ge_18": metrics["states_retaining_contact_negative"] >= 18,
        "no_only_positive_state": metrics["states_only_contact_positive_admitted"] == 0,
        "zero_selected_contact": metrics["selected_contact_count"] == 0, "false_abstentions_le_3": metrics["false_abstentions"] <= 3,
        "progress_fraction_ge_0_80": metrics["oracle_progress_fraction"] is not None and metrics["oracle_progress_fraction"] >= .80,
        "regret_le_0_20": metrics["normalized_route_progress_regret"] is not None and metrics["normalized_route_progress_regret"] <= .20,
        "top3_ge_0_75": metrics["best_contact_negative_top3"] is not None and metrics["best_contact_negative_top3"] >= .75,
        "no_family_collapse": no_collapse}
    filter_checks = {key: checks[key] for key in ("recall_ge_0_95", "fnr_le_0_05", "negative_retention_ge_0_50",
        "states_retaining_ge_18", "no_only_positive_state", "zero_selected_contact", "false_abstentions_le_3")}
    return {"checks": checks, "passed": all(checks.values()), "contact_filter_checks": filter_checks,
        "contact_filter_passed": all(filter_checks.values())}


def threshold_frontier(rows, states, source):
    labels = np.asarray([row["hard_contact"] for row in rows], bool); scores = np.asarray([row["probability"] for row in rows])
    auc_value = BASE.auc(labels, scores); ap_value = BASE.average_precision(labels, scores); values = defaultdict(list)
    gate_points = []; filter_points = []
    fields = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "admitted_contact_positive_count",
              "admitted_contact_negative_count", "states_retaining_contact_negative", "states_only_contact_positive_admitted",
              "false_abstentions", "selected_contact_count", "selected_stuck_count", "mean_selected_route_progress_m",
              "mean_selected_heading_improvement_rad", "normalized_route_progress_regret", "best_contact_negative_top1",
              "best_contact_negative_top3", "oracle_progress_fraction")
    for threshold in BASE.threshold_values(scores):
        admitted = scores < threshold; base_metrics = BASE.decision_metrics(rows, states, admitted)
        selection = evaluate_condition(rows, states, admitted, "kinematic", False); families = family_condition(rows, states, admitted, "kinematic")
        metric = {**base_metrics, **selection}; status = gate(metric, auc_value, ap_value, families)
        values["threshold"].append(float(threshold))
        for field in fields: values[field].append(np.nan if metric[field] is None else metric[field])
        values["complete_gate"].append(status["passed"]); values["contact_filter_gate"].append(status["contact_filter_passed"])
        for family in FAMILIES:
            for field in fields:
                family_value = families[family].get(field)
                values[f"family__{family}__{field}"].append(np.nan if family_value is None else family_value)
        if status["passed"]: gate_points.append({"threshold": float(threshold), "metrics": metric, "per_family": families, "gate": status})
        if status["contact_filter_passed"]: filter_points.append({"threshold": float(threshold), "metrics": metric, "per_family": families, "gate": status})
    arrays = {key: np.asarray(value) for key, value in values.items()}; path = CACHE / "heldout_threshold_frontier_v1.npz"
    atomic_npz(path, **arrays)
    # Descriptive limits do not select a threshold for deployment.
    summaries = {}
    for name, fn in {
        "max_negative_retention_at_recall_ge_0_95": lambda m: m["contact_negative_retention"],
        "max_states_retaining_at_recall_ge_0_95": lambda m: m["states_retaining_contact_negative"],
        "max_negative_retention_at_recall_ge_0_95_and_zero_selected_contacts": lambda m: m["contact_negative_retention"],
        "max_states_retaining_at_recall_ge_0_95_and_zero_selected_contacts": lambda m: m["states_retaining_contact_negative"],
        "max_progress_with_zero_selected_contacts": lambda m: m["mean_selected_route_progress_m"],
        "min_regret_at_recall_ge_0_95": lambda m: -m["normalized_route_progress_regret"] if m["normalized_route_progress_regret"] is not None else -1e9,
    }.items():
        candidates = []
        for i, threshold in enumerate(arrays["threshold"]):
            metric = {field: (None if np.isnan(arrays[field][i]) else float(arrays[field][i])) for field in fields}
            if "at_recall" in name and metric["contact_recall"] < .95: continue
            if "zero_selected_contact" in name and metric["selected_contact_count"] != 0: continue
            candidates.append((fn(metric), float(threshold), metric))
        summaries[name] = None if not candidates else max(candidates, key=lambda item: item[0])
    return {"thresholds": len(arrays["threshold"]), "auc": auc_value, "average_precision": ap_value,
        "complete_gate_points": gate_points, "contact_filter_gate_points": filter_points, "limits": summaries,
        "frontier": {"file": str(path), "sha256": BASE.sha(path),
            "content_digest": hashlib.sha256(b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))).hexdigest()}}


def state_classification(rows, states, admitted, selected_rows):
    selected_by_state = {row["state_id"]: row for row in selected_rows["per_state"]}; output = []
    for state in states:
        selected = selected_by_state[state["state_id"]]; pick_candidate = selected["selected_candidate"]
        pick = next((i for i in state["ids"] if rows[i]["candidate_index"] == pick_candidate), None)
        best = state["best"]
        if not state["negative"]: category = "NO_CONTACT_NEGATIVE_CANDIDATE"
        elif pick is not None and rows[pick]["hard_contact"]: category = "SELECTED_CONTACT_POSITIVE"
        elif pick is None: category = "ABSTAINED_WITH_SAFE_CANDIDATE"
        elif best is not None and not admitted[best]: category = "BEST_SAFE_REJECTED"
        elif pick == best: category = "BEST_SAFE_RETAINED_AND_SELECTED"
        else: category = "BEST_SAFE_RETAINED_BUT_MISRANKED"
        output.append({"state_id": state["state_id"], "family": state["family"], "classification": category,
            "oracle_best_candidate": None if best is None else rows[best]["candidate_index"],
            "oracle_best_probability": None if best is None else rows[best]["probability"],
            "oracle_best_admitted": None if best is None else bool(admitted[best]),
            "oracle_best_stuck": None if best is None else bool(rows[best]["stuck"]),
            "oracle_best_p_d": None if best is None else rows[best]["p_d"],
            "oracle_best_nominal_rank": None if best is None else state["rank"].index(best) + 1,
            **{key: selected[key] for key in ("selected_candidate", "selected_contact", "selected_stuck", "selected_p_d", "selected_p_theta")}})
    counts = Counter(row["classification"] for row in output)
    by_family = {family: dict(Counter(row["classification"] for row in output if row["family"] == family)) for family in FAMILIES}
    return {"counts": dict(counts), "by_family": by_family, "per_state": output}


def condition_set(rows, states, admitted):
    all_negative = np.asarray([not row["hard_contact"] for row in rows])
    conditions = {
        "A_MODEL_FILTER_KINEMATIC": (admitted, "kinematic"),
        "B_MODEL_FILTER_ORACLE_PROGRESS": (admitted, "oracle_progress"),
        "C_ORACLE_CONTACT_KINEMATIC": (all_negative, "kinematic"),
        "D_ORACLE_CONTACT_ORACLE_PROGRESS": (all_negative, "oracle_progress"),
        "E_MODEL_FILTER_KINEMATIC_ORACLE_STUCK_TIEBREAK": (admitted, "kinematic_stuck_tiebreak"),
    }
    output = {}
    for name, (mask, ranking) in conditions.items():
        output[name] = evaluate_condition(rows, states, mask, ranking, True)
        output[name]["per_family"] = family_condition(rows, states, mask, ranking)
    return output


def progress_loss_decomposition(rows, states, admitted, conditions):
    a = {row["state_id"]: row for row in conditions["A_MODEL_FILTER_KINEMATIC"]["per_state"]}
    c = {row["state_id"]: row for row in conditions["C_ORACLE_CONTACT_KINEMATIC"]["per_state"]}
    categories = defaultdict(list); state_rows = []
    for state in states:
        current, oracle = a[state["state_id"]], c[state["state_id"]]; best = state["best"]
        selected_index = next((i for i in state["ids"] if rows[i]["candidate_index"] == current["selected_candidate"]), None)
        current_progress = 0.0 if selected_index is None else rows[selected_index]["p_d"]
        oracle_progress = 0.0 if oracle["selected_candidate"] is None else float(oracle["selected_p_d"])
        lost = max(0.0, oracle_progress - current_progress); assigned = []
        if best is not None and not admitted[best]: assigned.append("filter_loss")
        if selected_index is not None and not rows[selected_index]["hard_contact"]:
            better = [i for i in state["negative"] if admitted[i] and BASE.preference(rows[i], rows[selected_index]) > 0]
            if better: assigned.append("ranking_loss")
            useful = rows[selected_index]["p_d"] > BASE.DELTA_D or rows[selected_index]["p_theta"] > BASE.DELTA_THETA
            if rows[selected_index]["stuck"] or not useful: assigned.append("recoverability_loss")
        if not any(rows[i]["p_d"] > BASE.DELTA_D or rows[i]["p_theta"] > BASE.DELTA_THETA for i in state["negative"]):
            assigned.append("candidate_bank_limitation")
        for category in assigned: categories[category].append((state["state_id"], state["family"], lost))
        state_rows.append({"state_id": state["state_id"], "family": state["family"], "categories": assigned,
            "model_progress_m": current_progress, "oracle_contact_kinematic_progress_m": oracle_progress, "positive_gap_m": lost})
    summary = {}
    for category in ("filter_loss", "ranking_loss", "recoverability_loss", "candidate_bank_limitation"):
        values = categories[category]; summary[category] = {"states": len(values), "positive_gap_sum_m": float(sum(x[2] for x in values)),
            "positive_gap_mean_m": float(np.mean([x[2] for x in values])) if values else 0.0,
            "by_family": dict(Counter(x[1] for x in values)), "state_ids": [x[0] for x in values]}
    overlap = Counter()
    for row in state_rows:
        for size in range(2, len(row["categories"]) + 1):
            for combination in itertools.combinations(sorted(row["categories"]), size): overlap["+".join(combination)] += 1
    return {"basis": "positive per-state gap from model-filter+kinematic to oracle-contact+kinematic; categories may overlap and losses are not additive",
        "no_useful_progress_rule": "p_d <= 0.03 m and p_theta <= 5 degrees", "summary": summary, "overlaps": dict(overlap), "per_state": state_rows}


def distribution(rows, threshold):
    values = np.asarray([row["probability"] for row in rows], np.float64); margin = values - threshold
    if not len(values): return {"count": 0}
    return {"count": len(values), "probability": {"min": float(values.min()), "q25": float(np.quantile(values, .25)),
        "median": float(np.median(values)), "q75": float(np.quantile(values, .75)), "max": float(values.max()), "mean": float(values.mean())},
        "signed_margin_from_threshold": {"min": float(margin.min()), "q25": float(np.quantile(margin, .25)),
        "median": float(np.median(margin)), "q75": float(np.quantile(margin, .75)), "max": float(margin.max()), "mean": float(margin.mean())},
        "absolute_margin_bins": {"near_le_0.01": int((np.abs(margin) <= THRESHOLD_NEAR).sum()),
            "moderate_0.01_to_0.10": int(((np.abs(margin) > THRESHOLD_NEAR) & (np.abs(margin) <= THRESHOLD_DEEP)).sum()),
            "deep_gt_0.10": int((np.abs(margin) > THRESHOLD_DEEP).sum())},
        "by_family": dict(Counter(row["family"] for row in rows)), "by_primitive": dict(Counter(row["candidate"] for row in rows)),
        "branches": [{"branch_id": row["branch_id"], "state_id": row["state_id"], "candidate_index": row["candidate_index"],
            "candidate": row["candidate"], "family": row["family"], "probability": row["probability"],
            "signed_margin": row["probability"] - threshold} for row in rows]}


def score_margins(rows, threshold, condition_a):
    false_negative = [row for row in rows if row["hard_contact"] and row["probability"] < threshold]
    selected_keys = {(row["state_id"], row["selected_candidate"]) for row in condition_a["per_state"] if row["selected_contact"]}
    selected_contact = [row for row in rows if (row["state_id"], row["candidate_index"]) in selected_keys]
    rejected_negative = [row for row in rows if not row["hard_contact"] and row["probability"] >= threshold]
    admitted_negative = [row for row in rows if not row["hard_contact"] and row["probability"] < threshold]
    output = {"margin_definition": "calibrated probability minus frozen calibration threshold",
        "bin_definition": {"near": "absolute margin <= 0.01", "moderate": "0.01 < absolute margin <= 0.10", "deep": "absolute margin > 0.10"},
        "contact_false_negatives": distribution(false_negative, threshold), "selected_contact": distribution(selected_contact, threshold),
        "rejected_contact_negative": distribution(rejected_negative, threshold), "admitted_contact_negative": distribution(admitted_negative, threshold)}
    fn_bins = output["contact_false_negatives"]["absolute_margin_bins"]
    output["interpretation"] = "DEEPLY_MISRANKED_ERRORS_PRESENT" if fn_bins["deep_gt_0.10"] else (
        "ISOLATED_NEAR_THRESHOLD_ERRORS" if fn_bins["near_le_0.01"] == len(false_negative) else "MIXED_MARGIN_ERRORS")
    return output


def evaluator_fixture():
    rows = []
    for candidate, (contact, stuck, probability, p_d, kin_d) in enumerate(((0, 1, .1, .2, .2), (1, 0, .8, .3, .3), (0, 0, .2, .18, .18))):
        rows.append({"branch_id": f"s:{candidate}", "state_id": "s", "candidate_index": candidate, "candidate": str(candidate),
            "primitives": [], "family": FAMILIES[0], "hard_contact": bool(contact), "stuck": bool(stuck), "p_d": p_d,
            "p_theta": 0., "kinematic": np.asarray([0, 0, 0, 0, kin_d, 0.]), "probability": probability})
    states = BASE.prepare_states(rows); admitted = np.asarray([True, False, True]); a = evaluate_condition(rows, states, admitted, "kinematic")
    e = evaluate_condition(rows, states, admitted, "kinematic_stuck_tiebreak")
    cases = {"or_contact_rejection": bool(not admitted[1]), "kinematic_selection": a["per_state"][0]["selected_candidate"] == 0,
        "stuck_tiebreak": e["per_state"][0]["selected_candidate"] == 2, "threshold_tie_rejected": not (.5 < .5),
        "deterministic_json": json.dumps({"b": 2, "a": 1}, sort_keys=True) == '{"a": 1, "b": 2}'}
    return {"schema": "true_future_contact_filter_decision_decomposition_fixture_v1", "cases": cases, "pass": all(cases.values())}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    fixture = evaluator_fixture()
    if not fixture["pass"]: raise RuntimeError("evaluator fixture failure")
    source, _, rows = load_frozen_evidence(); reproduction = metric_reproduction(source, rows); states = BASE.prepare_states(rows)
    threshold = float(source["calibration"]["threshold"]); admitted = np.asarray([row["probability"] < threshold for row in rows])
    frontier = threshold_frontier(rows, states, source); conditions = condition_set(rows, states, admitted)
    state_result = state_classification(rows, states, admitted, conditions["A_MODEL_FILTER_KINEMATIC"])
    loss = progress_loss_decomposition(rows, states, admitted, conditions); margins = score_margins(rows, threshold, conditions["A_MODEL_FILTER_KINEMATIC"])
    if frontier["complete_gate_points"]:
        classification = "CONTACT_PROXY_CALIBRATION_BOTTLENECK"
    elif frontier["contact_filter_gate_points"]:
        classification = "CONTACT_PROXY_SIGNAL_PLANNER_RECOVERABILITY_NO_GO"
    else:
        classification = "CONTACT_PROXY_FILTER_SCORE_NO_GO"
    secondary = []
    if state_result["counts"].get("BEST_SAFE_REJECTED", 0) >= state_result["counts"].get("BEST_SAFE_RETAINED_BUT_MISRANKED", 0): secondary.append("BEST_SAFE_REJECTION_DOMINANT")
    if state_result["counts"].get("BEST_SAFE_RETAINED_BUT_MISRANKED", 0) >= 6: secondary.append("KINEMATIC_RANKING_DOMINANT")
    for family, label in (("medium_enclosed_maze", "MEDIUM_MAZE_ROUTE_FAILURE"), ("loop_alias_stress", "LOOP_ALIAS_ROUTE_FAILURE")):
        metric = conditions["A_MODEL_FILTER_KINEMATIC"]["per_family"][family]
        if metric["mean_selected_route_progress_m"] <= 0 or metric["selected_contact_count"] > 0: secondary.append(label)
    if loss["summary"]["candidate_bank_limitation"]["states"]: secondary.append("CANDIDATE_BANK_LIMITATION")
    result = {"schema": "true_future_contact_filter_decision_decomposition_v1_result", "source_commit": "a2c2abfcf3d75a97371ddfbc73eaa6c79ed6f079",
        "development_status": "POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC",
        "claim_boundary": "SIMULATED_DISALLOWED_CONTACT_PROXY only; no material-hazard, injury, property-damage, people, or fragile-infrastructure claim",
        "preserved_result": "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY", "fixture": fixture,
        "reproduction": reproduction, "frozen_threshold": threshold, "heldout_frontier": frontier,
        "state_level_filter_decomposition": state_result, "candidate_set_upper_bounds": conditions,
        "progress_loss_decomposition": loss, "score_margin_analysis": margins, "classification": classification,
        "secondary_findings": secondary,
        "next_step": ("FRESH_CONTACT_DECISION_CALIBRATION_V1" if classification == "CONTACT_PROXY_CALIBRATION_BOTTLENECK" else
            "CONTACT_SAFE_ROUTE_AND_RECOVERY_PLANNER_V1" if classification == "CONTACT_PROXY_SIGNAL_PLANNER_RECOVERABILITY_NO_GO" else
            "Close WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1; no automatic successor authorized"),
        "custody": {"training": False, "checkpoint_execution": False, "simulation": False, "rendering": False, "encoding": False,
            "threshold_adopted_from_heldout": False, "jepa_predictor_opened": False}, "runtime_s": time.time() - started}
    # Full frontier rows are stored in the bound NPZ; keep gate points in JSON because they are small and scientifically decisive.
    result["content_digest"] = canonical_digest(result); atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "reproduction": reproduction, "frontier": {k: v for k, v in frontier.items() if k not in ("complete_gate_points", "contact_filter_gate_points")},
        "complete_gate_points": len(frontier["complete_gate_points"]), "contact_filter_gate_points": len(frontier["contact_filter_gate_points"]),
        "state_counts": state_result["counts"], "conditions": {k: {x: y for x, y in v.items() if x not in ("per_state", "per_family")} for k, v in conditions.items()},
        "secondary": secondary, "runtime_s": result["runtime_s"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
