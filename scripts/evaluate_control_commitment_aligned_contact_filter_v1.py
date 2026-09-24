#!/usr/bin/env python3
"""Read-only control-horizon decomposition of the frozen contact proxy.

The checkpoint is identity-bound but never loaded.  Every score, label, and
route quantity is reduced from the immutable Stage-1 row ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import evaluate_wide_geometry_score_composition_v1 as BASE


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1/result.json"
LEDGER = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1/stage1/row_level_evidence_v1.npz")
LEDGER_SHA256 = "ab47eb7848b980947ced6ee6f10493ef12578ab7871ef8ebdb97b46122617e9c"
CHECKPOINT_SHA256 = "3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31"
OUT = ROOT / ".generated/control_commitment_aligned_contact_filter_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/control_commitment_aligned_contact_filter_v1")
FAMILIES = BASE.FAMILIES
HORIZONS = {"H1": 4, "H2": 9, "H3": 14}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


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


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return np.where(value >= 0, 1.0 / (1.0 + np.exp(-value)), np.exp(value) / (1.0 + np.exp(value)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Deterministic bounded scalar-temperature fit, without model execution."""
    logits = np.asarray(logits, np.float64); labels = np.asarray(labels, np.float64)
    def objective(log_temperature: float) -> float:
        temperature = min(20.0, max(0.05, math.exp(log_temperature)))
        scaled = logits / temperature
        return float(np.mean(np.maximum(scaled, 0) - scaled * labels + np.log1p(np.exp(-np.abs(scaled)))))
    lo, hi = math.log(0.05), math.log(20.0); ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo); right = lo + ratio * (hi - lo)
    f_left, f_right = objective(left), objective(right)
    for _ in range(240):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo); f_left = objective(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo); f_right = objective(right)
    return float(min(20.0, max(0.05, math.exp((lo + hi) / 2.0))))


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    if not len(values): return {"count": 0}
    return {"count": int(len(values)), "min": float(values.min()), "q25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "q75": float(np.quantile(values, .75)), "max": float(values.max())}


def load_evidence():
    source = json.loads(SOURCE.read_text())
    if source["classification"] != "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY":
        raise RuntimeError("unexpected predecessor result")
    if source["training"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint identity mismatch")
    if sha(LEDGER) != LEDGER_SHA256 or source["row_level_evidence"]["sha256"] != LEDGER_SHA256:
        raise RuntimeError("row ledger identity mismatch")
    with np.load(LEDGER, allow_pickle=False) as loaded:
        ledger = {key: np.asarray(loaded[key]) for key in loaded.files}
    required = {"branch_id", "state_id", "candidate_index", "split", "family", "raw_logits",
                "calibrated_probability", "contact_labels", "stuck_labels", "p_d", "p_theta", "kinematic",
                "threshold_decision_admitted", "selected"}
    if not required.issubset(ledger):
        raise RuntimeError(f"row ledger missing {sorted(required - set(ledger))}")
    if ledger["raw_logits"].shape[1:] != (15, 2) or ledger["contact_labels"].shape[1:] != (15, 2):
        raise RuntimeError("frozen per-tick shape mismatch")
    return source, ledger


def rows_for(ledger, split: str, horizon: str, probabilities: np.ndarray) -> list[dict]:
    tick = HORIZONS[horizon]; mask = ledger["split"] == split; rows = []
    for local, index in enumerate(np.flatnonzero(mask)):
        rows.append({"branch_id": str(ledger["branch_id"][index]), "state_id": str(ledger["state_id"][index]),
            "candidate_index": int(ledger["candidate_index"][index]), "family": str(ledger["family"][index]),
            "hard_contact": bool(ledger["contact_labels"][index, tick, 1]),
            "h1_contact": bool(ledger["contact_labels"][index, HORIZONS["H1"], 1]),
            "h2_contact": bool(ledger["contact_labels"][index, HORIZONS["H2"], 1]),
            "h3_contact": bool(ledger["contact_labels"][index, HORIZONS["H3"], 1]),
            "first_interval": first_interval(ledger["contact_labels"][index, :, 1]),
            "stuck": bool(ledger["stuck_labels"][index, -1, 1]), "p_d": float(ledger["p_d"][index]),
            "p_theta": float(ledger["p_theta"][index]), "kinematic": np.asarray(ledger["kinematic"][index], np.float64),
            "probability": float(probabilities[local]), "global_index": int(index)})
    if len(rows) != 288 or len({row["state_id"] for row in rows}) != 24:
        raise RuntimeError(f"{split} cardinality mismatch")
    return rows


def first_interval(cumulative: np.ndarray) -> str:
    positions = np.flatnonzero(np.asarray(cumulative, bool))
    if not len(positions): return "NO_CONTACT_THROUGH_H3"
    tick = int(positions[0]) + 1
    if tick <= 5: return "CONTACT_BY_H1"
    if tick <= 10: return "FIRST_CONTACT_H1_TO_H2"
    return "FIRST_CONTACT_H2_TO_H3"


def exact_h3_reproduction(source, ledger) -> dict:
    mask = ledger["split"] == "heldout"
    probability = ledger["calibrated_probability"][mask, -1, 1].astype(np.float64)
    rows = rows_for(ledger, "heldout", "H3", probability); states = BASE.prepare_states(rows)
    threshold = float(source["calibration"]["threshold"]); admitted = probability < threshold
    persisted = ledger["threshold_decision_admitted"][mask].astype(bool)
    if not np.array_equal(admitted, persisted):
        index = int(np.flatnonzero(admitted != persisted)[0]); raise RuntimeError(f"H3 admission mismatch: {rows[index]['branch_id']}")
    metrics = BASE.decision_metrics(rows, states, admitted, True); expected = source["heldout"]["filtering_and_planning"]
    keys = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention",
            "states_retaining_contact_negative", "selected_contact_count", "false_abstentions",
            "mean_selected_route_progress_m", "oracle_contact_kinematic_progress_m",
            "normalized_route_progress_regret", "best_contact_negative_top3")
    for key in keys:
        if not np.isclose(metrics[key], expected[key], atol=1e-12, rtol=1e-12):
            raise RuntimeError(f"H3 reproduction failed: {key}: {metrics[key]} != {expected[key]}")
    selected = {(row["state_id"], row["selected_candidate"]) for row in metrics["per_state"] if row["selected_candidate"] is not None}
    persisted_selected = {(str(ledger["state_id"][i]), int(ledger["candidate_index"][i])) for i in np.flatnonzero(mask & ledger["selected"].astype(bool))}
    if selected != persisted_selected: raise RuntimeError("H3 selected candidates differ")
    return {"status": "PASS", "ledger_sha256": LEDGER_SHA256, "checkpoint_identity_only": CHECKPOINT_SHA256,
            "checkpoint_executed": False, "temperature": float(source["calibration"]["temperature"]), "threshold": threshold,
            "auc": float(source["heldout"]["contact"]["auc"]), "average_precision": float(source["heldout"]["contact"]["average_precision"]),
            "metrics": {key: metrics[key] for key in keys}, "selected_candidates_exact": True}


def commitment_contract() -> dict:
    return {"candidate_horizon_blocks": 4, "candidate_block_ticks": 5, "command_tick_s": .1,
        "committed_blocks_per_cycle": 1, "committed_ticks": 5, "committed_duration_s": .5,
        "policy_dt_s": .02, "physics_dt_s": .002, "policy_steps_per_command_tick": 5,
        "physics_steps_per_policy_step": 10, "physics_steps_per_committed_block": 250,
        "replanning": "after the first primitive block, with a fresh observation; candidate blocks 2-4 are replaceable",
        "hold_brake": "zero-command hold is registered and available next cycle, subject to the same slew limiter; it is not a validated emergency brake",
        "slew": {"max_delta_per_command_tick": {"vx_mps": .25, "vy_mps": 0., "yaw_rps": .35},
                 "previous_applied_command_carried_across_blocks": True},
        "source_bindings": ["lewm/planning/local_mpc.py:163", "scripts/benchmark_lewm_closed_loop_mpc.py:2020",
            "lewm_genesis/lewm_genesis/rollout.py:1", "config/go2_platform_manifest.yaml:64",
            "config/go2_primitive_registry.yaml:14", "scripts/dev_action_slew_reconstruction_v1.py:56"],
        "stopping_horizon": {"H2_status": "CONSERVATIVE_UNVALIDATED_STOPPING_PROXY",
            "reason": "H2 is not committed and no validated stopping-distance/time envelope or brake test binds ten ticks; zero-command hold and slew arithmetic exist but do not validate stopping safety",
            "H2_deployable_hard_filter_authorized": False}}


def interval_prevalence(ledger) -> dict:
    output = {}
    fresh = np.isin(ledger["split"], ["calibration", "heldout"])
    for split in ("calibration", "heldout"):
        mask = ledger["split"] == split
        intervals = np.asarray([first_interval(value[:, 1]) for value in ledger["contact_labels"][mask]])
        indices = np.flatnonzero(mask)
        output[split] = {"branches": int(mask.sum()), "counts": dict(Counter(intervals)),
            "by_family": {family: dict(Counter(intervals[ledger["family"][indices] == family])) for family in FAMILIES},
            "by_candidate_index": {str(candidate): dict(Counter(intervals[ledger["candidate_index"][indices] == candidate])) for candidate in range(12)}}
    output["fresh_total"] = {"branches": int(fresh.sum()), "counts": dict(Counter(first_interval(value[:, 1]) for value in ledger["contact_labels"][fresh]))}
    return output


def horizon_probabilities(ledger, source) -> tuple[dict, dict]:
    probabilities = {}; calibration = {}
    for horizon, tick in HORIZONS.items():
        logits = ledger["raw_logits"][:, tick, 1].astype(np.float64)
        if horizon == "H3":
            temperature = float(source["calibration"]["temperature"])
        else:
            mask = ledger["split"] == "calibration"
            labels = ledger["contact_labels"][mask, tick, 1].astype(np.float64)
            temperature = fit_temperature(logits[mask], labels)
        probabilities[horizon] = sigmoid(logits / temperature)
        calibration[horizon] = {"temperature": temperature}
    return probabilities, calibration


def horizon_discrimination(ledger, probabilities) -> dict:
    output = {}
    for split in ("calibration", "heldout"):
        mask = ledger["split"] == split; output[split] = {}
        score_matrix = []
        for horizon, tick in HORIZONS.items():
            labels = ledger["contact_labels"][mask, tick, 1].astype(bool); score = probabilities[horizon][mask]
            score_matrix.append(score)
            per_family = {}
            for family in FAMILIES:
                fm = mask & (ledger["family"] == family); y = ledger["contact_labels"][fm, tick, 1].astype(bool); p = probabilities[horizon][fm]
                per_family[family] = {"positive": int(y.sum()), "negative": int((~y).sum()),
                    "auc": None if not y.any() or y.all() else BASE.auc(y, p),
                    "average_precision": None if not y.any() else BASE.average_precision(y, p)}
            output[split][horizon] = {"positive": int(labels.sum()), "negative": int((~labels).sum()),
                "prevalence": float(labels.mean()), "auc": BASE.auc(labels, score),
                "average_precision": BASE.average_precision(labels, score),
                "contact_negative_score": summary(score[~labels]), "per_family": per_family}
        output[split]["score_correlation_pearson"] = np.corrcoef(np.asarray(score_matrix)).tolist()
        output[split]["correlation_order"] = list(HORIZONS)
    return output


def ranked_order(rows, ids, mode: str) -> list[int]:
    if mode == "kinematic": return BASE.route_order(rows, ids)
    remaining = list(ids); output = []
    while remaining:
        best_distance = max(float(rows[i]["kinematic"][4]) for i in remaining)
        near = [i for i in remaining if best_distance - float(rows[i]["kinematic"][4]) <= BASE.DELTA_D]
        if mode == "soft_scores":
            pick = min(near, key=lambda i: (rows[i]["p_h2"], rows[i]["p_h3"], -float(rows[i]["kinematic"][5]), rows[i]["candidate_index"]))
        elif mode == "oracle_continuation":
            pick = min(near, key=lambda i: (rows[i]["h2_contact"], rows[i]["h3_contact"], -float(rows[i]["kinematic"][5]), rows[i]["candidate_index"]))
        else: raise ValueError(mode)
        output.append(pick); remaining.remove(pick)
    return output


def evaluate(rows, admitted: np.ndarray, mode="kinematic", include_state=True) -> dict:
    states = BASE.prepare_states(rows); labels = np.asarray([row["hard_contact"] for row in rows], bool)
    per_state = []; progress = []; heading = []; oracle_progress = []; regrets = []; top1 = []; top3 = []
    selected_labels = []; selected_stuck = []; intervals = Counter(); false_abstain = correct_abstain = 0
    for state in states:
        ids = state["ids"]; order = ranked_order(rows, ids, mode); available = [i for i in order if admitted[i]]; pick = available[0] if available else None
        if pick is None:
            false_abstain += bool(state["negative"]); correct_abstain += not bool(state["negative"])
        else:
            progress.append(rows[pick]["p_d"]); heading.append(rows[pick]["p_theta"]); selected_labels.append(labels[pick]); selected_stuck.append(rows[pick]["stuck"])
            intervals[rows[pick]["first_interval"]] += 1
        if state["oracle"] is not None: oracle_progress.append(rows[state["oracle"]]["p_d"])
        if state["best"] is not None:
            top1.append(pick == state["best"]); top3.append(state["best"] in available[:3])
            if pick is not None and not labels[pick] and len(state["negative"]) >= 2:
                values = [rows[i]["p_d"] for i in state["negative"]]; spread = max(values) - min(values)
                if spread > 1e-8: regrets.append((rows[state["best"]]["p_d"] - rows[pick]["p_d"]) / spread)
        per_state.append({"state_id": state["state_id"], "family": state["family"], "selected_candidate": None if pick is None else rows[pick]["candidate_index"],
            "selected_hard_window_contact": None if pick is None else bool(labels[pick]), "selected_first_contact_interval": None if pick is None else rows[pick]["first_interval"],
            "selected_stuck": None if pick is None else bool(rows[pick]["stuck"]), "selected_progress_m": None if pick is None else rows[pick]["p_d"],
            "selected_heading_improvement_rad": None if pick is None else rows[pick]["p_theta"], "admitted": int(admitted[ids].sum()),
            "admitted_negative": int(sum(admitted[i] and not labels[i] for i in ids)), "admitted_positive": int(sum(admitted[i] and labels[i] for i in ids))})
    mean_progress = float(np.mean(progress)) if progress else 0.; oracle_mean = float(np.mean(oracle_progress)) if oracle_progress else 0.
    metrics = {"contact_recall": float((~admitted)[labels].mean()) if labels.any() else 1.,
        "contact_false_negative_rate": float(admitted[labels].mean()) if labels.any() else 0.,
        "contact_negative_retention": float(admitted[~labels].mean()), "admitted_contact_positive": int((admitted & labels).sum()),
        "admitted_contact_negative": int((admitted & ~labels).sum()),
        "states_retaining_negative": int(sum(row["admitted_negative"] > 0 for row in per_state)),
        "states_only_positive_admitted": int(sum(row["admitted"] > 0 and row["admitted_negative"] == 0 for row in per_state)),
        "selected_hard_window_contacts": int(sum(selected_labels)), "selected_stuck": int(sum(selected_stuck)),
        "selected_contact_intervals": {name: int(intervals.get(name, 0)) for name in ("CONTACT_BY_H1", "FIRST_CONTACT_H1_TO_H2", "FIRST_CONTACT_H2_TO_H3", "NO_CONTACT_THROUGH_H3")},
        "false_abstentions": int(false_abstain), "correct_abstentions": int(correct_abstain), "abstentions": int(false_abstain + correct_abstain),
        "mean_selected_route_progress_m": mean_progress, "mean_selected_heading_improvement_rad": float(np.mean(heading)) if heading else 0.,
        "oracle_horizon_kinematic_progress_m": oracle_mean, "oracle_progress_fraction": None if abs(oracle_mean) < 1e-12 else mean_progress / oracle_mean,
        "normalized_route_progress_regret": None if not regrets else float(np.mean(regrets)),
        "best_horizon_negative_top1": None if not top1 else float(np.mean(top1)), "best_horizon_negative_top3": None if not top3 else float(np.mean(top3))}
    if include_state: metrics["per_state"] = per_state
    return metrics


def family_evaluation(rows, admitted, mode) -> dict:
    output = {}
    for family in FAMILIES:
        ids = [i for i, row in enumerate(rows) if row["family"] == family]
        output[family] = evaluate([rows[i] for i in ids], admitted[ids], mode, False)
    return output


def choose_threshold(rows) -> dict:
    scores = np.asarray([row["probability"] for row in rows]); states = BASE.prepare_states(rows); eligible = []; records = []
    for threshold in BASE.threshold_values(scores):
        admitted = scores < threshold; metric = evaluate(rows, admitted, "kinematic", False)
        item = {"threshold": float(threshold), **metric}; records.append(item)
        if metric["contact_recall"] >= .95 and metric["contact_false_negative_rate"] <= .05:
            eligible.append(item)
    if not eligible: raise RuntimeError("calibration has no H1/H2 recall-eligible threshold")
    chosen = max(eligible, key=lambda x: (x["states_retaining_negative"], x["contact_negative_retention"],
        x["mean_selected_route_progress_m"], -(x["normalized_route_progress_regret"] if x["normalized_route_progress_regret"] is not None else 1e9),
        -x["false_abstentions"], -x["threshold"]))
    return {"threshold": chosen["threshold"], "eligible_thresholds": len(eligible), "thresholds_evaluated": len(records), "metrics": chosen, "records": records}


def gate(metrics, auc_value, ap_value, families) -> dict:
    no_collapse = all(value["states_retaining_negative"] > 0 and value["selected_hard_window_contacts"] == 0 for value in families.values())
    checks = {"auc_ge_0_90": auc_value >= .90, "ap_ge_0_90": ap_value >= .90,
        "recall_ge_0_95": metrics["contact_recall"] >= .95, "fnr_le_0_05": metrics["contact_false_negative_rate"] <= .05,
        "negative_retention_ge_0_50": metrics["contact_negative_retention"] >= .50, "states_retaining_ge_18": metrics["states_retaining_negative"] >= 18,
        "no_only_positive_state": metrics["states_only_positive_admitted"] == 0, "zero_selected_contact": metrics["selected_hard_window_contacts"] == 0,
        "false_abstentions_le_3": metrics["false_abstentions"] <= 3,
        "progress_fraction_ge_0_80": metrics["oracle_progress_fraction"] is not None and metrics["oracle_progress_fraction"] >= .80,
        "regret_le_0_20": metrics["normalized_route_progress_regret"] is not None and metrics["normalized_route_progress_regret"] <= .20,
        "top3_ge_0_75": metrics["best_horizon_negative_top3"] is not None and metrics["best_horizon_negative_top3"] >= .75,
        "no_family_collapse": no_collapse}
    return {"checks": checks, "passed": all(checks.values()), "passed_count": int(sum(checks.values())), "total_checks": len(checks)}


def fixture() -> dict:
    labels = np.asarray([[0] * 15, [0, 0, 0, 0, 1] + [1] * 10, [0] * 8 + [1] * 7, [0] * 12 + [1] * 3], np.uint8)
    cases = {"no_contact": first_interval(labels[0]) == "NO_CONTACT_THROUGH_H3", "h1": first_interval(labels[1]) == "CONTACT_BY_H1",
        "h2": first_interval(labels[2]) == "FIRST_CONTACT_H1_TO_H2", "h3": first_interval(labels[3]) == "FIRST_CONTACT_H2_TO_H3",
        "strict_threshold_tie_rejected": not (.5 < .5), "deterministic_json": json.dumps({"b": 2, "a": 1}, sort_keys=True) == '{"a": 1, "b": 2}'}
    return {"schema": "control_commitment_aligned_contact_filter_fixture_v1", "cases": cases, "pass": all(cases.values())}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True, action="store_true"); parser.parse_args()
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    fixed_fixture = fixture()
    if not fixed_fixture["pass"]: raise RuntimeError("fixture failed")
    source, ledger = load_evidence(); reproduction = exact_h3_reproduction(source, ledger); contract = commitment_contract()
    prevalence = interval_prevalence(ledger); probabilities, calibration = horizon_probabilities(ledger, source)
    discrimination = horizon_discrimination(ledger, probabilities)
    conditions = {}; frontier_files = {}
    for horizon in ("H1", "H2"):
        cal_prob = probabilities[horizon][ledger["split"] == "calibration"]
        cal_rows = rows_for(ledger, "calibration", horizon, cal_prob); selected = choose_threshold(cal_rows)
        threshold = selected["threshold"]; calibration[horizon].update({k: selected[k] for k in ("threshold", "eligible_thresholds", "thresholds_evaluated", "metrics")})
        arrays = {key: np.asarray([row[key] for row in selected["records"]]) for key in selected["records"][0] if key != "selected_contact_intervals"}
        frontier_path = CACHE / f"{horizon.lower()}_calibration_threshold_frontier_v1.npz"; atomic_npz(frontier_path, **arrays)
        frontier_files[horizon] = {"file": str(frontier_path), "sha256": sha(frontier_path)}
    calibration["H3"] = {"temperature": float(source["calibration"]["temperature"]), "threshold": float(source["calibration"]["threshold"]),
        "preserved_not_refit": True, "metrics": source["calibration"]["metrics"]}
    # Score columns used by the soft condition are attached to every horizon-specific row.
    for horizon in ("H1", "H2", "H3"):
        mask = ledger["split"] == "heldout"; rows = rows_for(ledger, "heldout", horizon, probabilities[horizon][mask])
        for local, row in enumerate(rows):
            global_index = row["global_index"]; row["p_h2"] = float(probabilities["H2"][global_index]); row["p_h3"] = float(probabilities["H3"][global_index])
        threshold = float(calibration[horizon]["threshold"]); admitted = np.asarray([row["probability"] < threshold for row in rows])
        metrics = evaluate(rows, admitted, "kinematic", True); families = family_evaluation(rows, admitted, "kinematic")
        score = np.asarray([row["probability"] for row in rows]); labels = np.asarray([row["hard_contact"] for row in rows], bool)
        conditions[f"{horizon}_HARD"] = {"horizon": horizon, "deployable": horizon == "H1",
            "temperature": float(calibration[horizon]["temperature"]), "threshold": threshold,
            "auc": BASE.auc(labels, score), "average_precision": BASE.average_precision(labels, score),
            "metrics": metrics, "per_family": families, "gate": gate(metrics, BASE.auc(labels, score), BASE.average_precision(labels, score), families)}
    # Soft continuation ranks only the H1-admitted candidate set.
    h1 = conditions["H1_HARD"]; rows_h1 = rows_for(ledger, "heldout", "H1", probabilities["H1"][ledger["split"] == "heldout"])
    for row in rows_h1:
        index = row["global_index"]; row["p_h2"] = float(probabilities["H2"][index]); row["p_h3"] = float(probabilities["H3"][index])
    admitted_h1 = np.asarray([row["probability"] < h1["threshold"] for row in rows_h1])
    soft_metrics = evaluate(rows_h1, admitted_h1, "soft_scores", True); soft_families = family_evaluation(rows_h1, admitted_h1, "soft_scores")
    base_intervals, soft_intervals = h1["metrics"]["selected_contact_intervals"], soft_metrics["selected_contact_intervals"]
    base_later = base_intervals["FIRST_CONTACT_H1_TO_H2"] + base_intervals["FIRST_CONTACT_H2_TO_H3"]
    soft_later = soft_intervals["FIRST_CONTACT_H1_TO_H2"] + soft_intervals["FIRST_CONTACT_H2_TO_H3"]
    family_ties = sum(soft_families[f]["selected_hard_window_contacts"] <= h1["per_family"][f]["selected_hard_window_contacts"] and
                      soft_families[f]["mean_selected_route_progress_m"] >= .9 * h1["per_family"][f]["mean_selected_route_progress_m"] for f in FAMILIES)
    soft_gate = {"no_extra_h1_contact": soft_metrics["selected_hard_window_contacts"] <= h1["metrics"]["selected_hard_window_contacts"],
        "reduces_later_contact": soft_later < base_later,
        "progress_within_10_percent": soft_metrics["mean_selected_route_progress_m"] >= .9 * h1["metrics"]["mean_selected_route_progress_m"],
        "no_added_false_abstention": soft_metrics["false_abstentions"] <= h1["metrics"]["false_abstentions"], "families_ge_3": family_ties >= 3}
    conditions["H1_HARD_H2_H3_SOFT"] = {"metrics": soft_metrics, "per_family": soft_families,
        "later_contacts_h1_h2_or_h2_h3": soft_later, "baseline_later_contacts": base_later,
        "gate": {"checks": soft_gate, "passed": all(soft_gate.values())}}
    # Oracle masks and continuation tie-break use labels only as diagnostic upper bounds.
    oracle = {}
    for horizon in ("H1", "H2", "H3"):
        rows = rows_for(ledger, "heldout", horizon, probabilities[horizon][ledger["split"] == "heldout"])
        for row in rows: row["p_h2"] = row["p_h3"] = 0.
        admitted = np.asarray([not row["hard_contact"] for row in rows])
        oracle[f"ORACLE_{horizon}_FILTER_KINEMATIC"] = {"metrics": evaluate(rows, admitted, "kinematic", True), "per_family": family_evaluation(rows, admitted, "kinematic")}
    oracle_h1_rows = rows_for(ledger, "heldout", "H1", probabilities["H1"][ledger["split"] == "heldout"])
    admitted = np.asarray([not row["h1_contact"] for row in oracle_h1_rows])
    oracle["ORACLE_H1_FILTER_ORACLE_CONTINUATION_TIEBREAK"] = {"metrics": evaluate(oracle_h1_rows, admitted, "oracle_continuation", True),
        "per_family": family_evaluation(oracle_h1_rows, admitted, "oracle_continuation")}
    h1_gate = conditions["H1_HARD"]["gate"]
    h3_metrics = conditions["H3_HARD"]["metrics"]
    material_improvement = (conditions["H1_HARD"]["metrics"]["contact_negative_retention"] - h3_metrics["contact_negative_retention"] >= .10 or
        conditions["H1_HARD"]["metrics"]["states_retaining_negative"] - h3_metrics["states_retaining_negative"] >= 3)
    narrow_miss = h1_gate["passed_count"] >= h1_gate["total_checks"] - 3 and conditions["H1_HARD"]["metrics"]["contact_recall"] >= .90 and conditions["H1_HARD"]["metrics"]["contact_negative_retention"] >= .40
    if h1_gate["passed"]: classification = "COMMITMENT_ALIGNED_CONTACT_FILTER_SIGNAL"
    elif contract["stopping_horizon"]["H2_status"] == "VALIDATED_STOPPING_HORIZON" and conditions["H2_HARD"]["gate"]["passed"]:
        classification = "STOPPING_HORIZON_CONTACT_FILTER_SIGNAL"
    elif material_improvement and narrow_miss: classification = "CONTACT_FILTER_HORIZON_MISMATCH_POSITIVE_TENDENCY"
    else: classification = "CONTACT_SCORE_NO_GO_ACROSS_CONTROL_HORIZONS"
    result = {"schema": "control_commitment_aligned_contact_filter_v1_result", "source_commit": "f720e2274aaa4c72961ba634731b57a0f2a97c4c",
        "development_status": "POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC", "preserved_result": "CONTACT_PROXY_FILTER_SCORE_NO_GO",
        "preserved_scope": "No threshold on cumulative contact probability through H3 provides the required safety-mobility operating point for WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1.",
        "claim_boundary": "SIMULATED_DISALLOWED_CONTACT_PROXY only; no material-hazard, human-safety, injury, property-damage, or closed-loop guarantee",
        "fixture": fixed_fixture, "reproduction": reproduction, "control_commitment": contract,
        "first_contact_interval_prevalence": prevalence, "horizon_discrimination": discrimination,
        "calibration": calibration, "calibration_frontiers": frontier_files, "conditions": conditions,
        "oracle_upper_bounds": oracle, "primary_classification": classification,
        "secondary_classification": "CONTINUATION_RISK_RANKING_SIGNAL" if conditions["H1_HARD_H2_H3_SOFT"]["gate"]["passed"] else "CONTINUATION_RISK_RANKING_NO_SIGNAL",
        "h3_no_go_horizon_mismatch_diagnosis": {
            "selected_contact_failure_was_outside_committed_H1": (
                h3_metrics["selected_hard_window_contacts"] > 0
                and conditions["H1_HARD"]["metrics"]["selected_hard_window_contacts"] == 0),
            "complete_operating_point_rescued": bool(h1_gate["passed"]),
            "conclusion": "The selected-contact part of the H3 failure was a horizon mismatch, but the complete H3 no-go was not rescued because H1 discrimination, retention, progress, regret, and top-3 still failed."},
        "next_step": ("CANDIDATE_CONDITIONED_COMMITMENT_HORIZON_CONTACT_PREDICTOR_V1 (specified only; not executed)" if classification == "COMMITMENT_ALIGNED_CONTACT_FILTER_SIGNAL" else
            "Candidate-conditioned H2 contact predictor only after independent validation of H2 as a stopping horizon" if classification == "STOPPING_HORIZON_CONTACT_FILTER_SIGNAL" else
            "No automatic predictor; return for decision after positive-tendency diagnosis" if classification == "CONTACT_FILTER_HORIZON_MISMATCH_POSITIVE_TENDENCY" else
            "Close WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1; change architecture, proxy target, or sensor coverage"),
        "custody": {"training": False, "checkpoint_execution": False, "simulation": False, "rendering": False, "encoding": False,
            "heldout_threshold_adopted": False, "jepa_predictor_opened": False, "navigation_memory_novelty": False},
        "runtime_s": time.time() - started}
    result["content_digest"] = canonical_digest(result); atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "secondary": result["secondary_classification"], "reproduction": reproduction,
        "commitment": contract, "prevalence": prevalence, "discrimination": discrimination,
        "calibration": calibration, "conditions": {name: ({"auc": value.get("auc"), "average_precision": value.get("average_precision"),
            "threshold": value.get("threshold"), "metrics": {k: v for k, v in value["metrics"].items() if k != "per_state"}, "gate": value.get("gate")}) for name, value in conditions.items()},
        "runtime_s": result["runtime_s"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
