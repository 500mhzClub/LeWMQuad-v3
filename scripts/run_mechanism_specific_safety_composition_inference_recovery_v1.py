#!/usr/bin/env python3
"""Recover frozen row predictions and evaluate mechanism-specific safety composition."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE
from scripts import train_evaluate_deployment_valid_safety_observability_matrix_v1 as MATRIX
from scripts import train_evaluate_enhanced_embodied_safety_observability_v2 as ENHANCED

OUT = ROOT / ".generated/mechanism_specific_safety_composition_inference_recovery_v1"
MATRIX_OUT = ROOT / ".generated/deployment_valid_safety_observability_matrix_v1"
ENHANCED_OUT = ROOT / ".generated/enhanced_embodied_safety_observability_v2"
GUARD_OUT = ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1"
ACTION_CHECKPOINT = MATRIX_OUT / "action_control_only_seed_2026082008.pt"
ENHANCED_CHECKPOINT = ENHANCED_OUT / "enhanced_proprio_action_safety_head_v1_seed_2026082009.pt"
EXPECTED_ACTION_SHA = "bc80ad410f83ab8503976a2cca850c833e05759af9e0cb85c46b406644eb8dcf"
EXPECTED_ENHANCED_SHA = "82b2704c770e2332a4a1e25b83fc6d0e8277877bee522e393f57cca3b5382a77"
EXPECTED_SENSOR_DIGEST = "d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d"
FAMILIES = FS.FAMILIES
CONDITION_COMPLEXITY = {"ACTION_CONTROL_ONLY": 0, "ENHANCED_EMBODIED": 1}
PRIMARY_BINDING = {"contact": "ENHANCED_EMBODIED", "stuck": "ACTION_CONTROL_ONLY"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=FS.json_default) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def array_content_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode()); digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def sigmoid(logits: np.ndarray, temperature: float) -> np.ndarray:
    value = np.asarray(logits, np.float64) / temperature
    return (1. / (1. + np.exp(-value))).astype(np.float64)


def load_and_align() -> tuple[list[dict], list[dict], dict]:
    action_branches, action_index, _ = MATRIX.load_branches()
    enhanced_branches, enhanced_index = ENHANCED.load_branches()
    if enhanced_index["content_digest"] != EXPECTED_SENSOR_DIGEST:
        raise RuntimeError("enhanced sensor content digest changed")
    if len(action_branches) != 576 or len(enhanced_branches) != 576:
        raise RuntimeError("frozen cardinality changed")
    rows = []
    for index, (action, enhanced) in enumerate(zip(action_branches, enhanced_branches)):
        fields = ("branch_id", "split", "family", "candidate_index")
        for field in fields:
            if action[field] != enhanced[field]:
                raise RuntimeError(f"row {index}: {field} alignment mismatch")
        if action["branch_id"] != f"{action['branch_id'].split(':')[0]}:{int(action['candidate_index']):02d}":
            raise RuntimeError(f"row {index}: malformed branch identity")
        action_target = DENSE.branch_targets(action); enhanced_target = DENSE.branch_targets(enhanced)
        if not np.array_equal(action_target, enhanced_target):
            raise RuntimeError(f"{action['branch_id']}: safety label mismatch")
        action_control = np.asarray(action["action_control"], np.float32)
        enhanced_control = np.asarray(enhanced["action_control"], np.float32)
        if action_control.shape != (15, 4) or enhanced_control.shape != (15, 6):
            raise RuntimeError(f"{action['branch_id']}: action/control shape mismatch")
        projection = enhanced_control[:, [0, 2, 3, 5]]
        if not np.array_equal(action_control, projection) or not np.array_equal(enhanced_control[:, [1, 4]], np.zeros((15, 2), np.float32)):
            raise RuntimeError(f"{action['branch_id']}: action/control lineage mismatch")
        rows.append({"row_index": index, "branch_id": action["branch_id"],
                     "state_id": action["branch_id"].split(":")[0], "candidate_index": int(action["candidate_index"]),
                     "split": action["split"], "family": action["family"], "ticks": 15})
    report = {
        "rows": 576, "states": len({row["state_id"] for row in rows}), "ticks_per_branch": 15,
        "split_rows": {split: sum(row["split"] == split for row in rows) for split in ("fit", "calibration", "heldout")},
        "families": list(FAMILIES), "candidate_identity_matches": 576, "split_matches": 576,
        "family_matches": 576, "tick_count_matches": 576, "action_control_matches": 576,
        "contact_stuck_aggregate_label_matches": 576, "enhanced_sensor_index_digest": enhanced_index["content_digest"],
        "action_sensor_index_digest": action_index["proprio_index_digest"],
    }
    return action_branches, enhanced_branches, {"rows": rows, "report": report}


def frozen_inference(action_branches: list[dict], enhanced_branches: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sha(ACTION_CHECKPOINT) != EXPECTED_ACTION_SHA or sha(ENHANCED_CHECKPOINT) != EXPECTED_ENHANCED_SHA:
        raise RuntimeError("frozen checkpoint hash mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_model = MATRIX.SafetyModalityModel("ACTION_CONTROL_ONLY").to(device)
    action_blob = torch.load(ACTION_CHECKPOINT, map_location=device, weights_only=True)
    action_model.load_state_dict(action_blob["state_dict"]); action_model.eval()
    enhanced_model = ENHANCED.EnhancedSafetyModel().to(device)
    enhanced_blob = torch.load(ENHANCED_CHECKPOINT, map_location=device, weights_only=True)
    enhanced_model.load_state_dict(enhanced_blob["state_dict"]); enhanced_model.eval()
    action_stats = json.loads((MATRIX_OUT / "proprio_fit_standardization.json").read_text())
    rgb_index = json.loads((MATRIX_OUT / "raw_rgb_index.json").read_text())
    enhanced_stats = json.loads((ENHANCED_OUT / "fit_standardization.json").read_text())
    action_logits, action_targets = MATRIX.predict(action_model, "ACTION_CONTROL_ONLY", action_branches, device, rgb_index, action_stats)
    enhanced_logits, enhanced_targets = ENHANCED.predict(enhanced_model, enhanced_branches, enhanced_stats, device)
    if action_logits.shape != (576, 15, 5) or enhanced_logits.shape != (576, 15, 5):
        raise RuntimeError("wrong frozen inference shape")
    if action_logits.dtype != np.float32 or enhanced_logits.dtype != np.float32:
        raise RuntimeError("raw logits were not recovered at FP32")
    if not np.array_equal(action_targets, enhanced_targets):
        raise RuntimeError("inference target alignment mismatch")
    del action_model, enhanced_model
    if device.type == "cuda": torch.cuda.empty_cache()
    return action_logits, enhanced_logits, action_targets.astype(np.float32)


def split_indices(rows: list[dict]) -> dict[str, np.ndarray]:
    return {split: np.asarray([i for i, row in enumerate(rows) if row["split"] == split], int)
            for split in ("fit", "calibration", "heldout")}


def reproduction_check(action_branches, enhanced_branches, action_logits, enhanced_logits, targets, route_rows, route_indices, kinematic):
    matrix_result = json.loads((MATRIX_OUT / "result.json").read_text())
    enhanced_result = json.loads((ENHANCED_OUT / "result.json").read_text())
    held_action = [branch for branch in action_branches if branch["split"] == "heldout"]
    held_enhanced = [branch for branch in enhanced_branches if branch["split"] == "heldout"]
    index = np.asarray([i for i, branch in enumerate(action_branches) if branch["split"] == "heldout"], int)
    action_cal = matrix_result["calibration"]["ACTION_CONTROL_ONLY"]
    enhanced_cal = enhanced_result["calibration"]
    reproduced = {
        "ACTION_CONTROL_ONLY": MATRIX.evaluate("ACTION_CONTROL_ONLY", held_action, action_logits[index], targets[index],
            action_cal["temperature"], action_cal["threshold"], route_rows, route_indices, kinematic),
        "ENHANCED_EMBODIED": MATRIX.evaluate("ENHANCED_PROPRIO_ACTION", held_enhanced, enhanced_logits[index], targets[index],
            enhanced_cal["temperature"], enhanced_cal["threshold"], route_rows, route_indices, kinematic),
    }
    expected = {"ACTION_CONTROL_ONLY": matrix_result["heldout"]["ACTION_CONTROL_ONLY"],
                "ENHANCED_EMBODIED": enhanced_result["heldout"]}
    checks = {}; first_divergence = None
    def compare_tree(actual, target, path, output):
        nonlocal first_divergence
        if isinstance(target, dict):
            if not isinstance(actual, dict):
                first_divergence = first_divergence or {"field": path, "actual_type": type(actual).__name__, "expected_type": "dict"}; return
            for key, value in target.items():
                if key not in actual:
                    first_divergence = first_divergence or {"field": f"{path}.{key}", "error": "missing regenerated field"}; continue
                compare_tree(actual[key], value, f"{path}.{key}" if path else key, output)
            return
        if isinstance(target, list):
            if not isinstance(actual, list) or len(actual) != len(target):
                first_divergence = first_divergence or {"field": path, "actual_length": len(actual), "expected_length": len(target)}; return
            for index, value in enumerate(target): compare_tree(actual[index], value, f"{path}[{index}]", output)
            return
        if target is None or isinstance(target, (str, bool, int)):
            passed = actual == target
        elif isinstance(target, float):
            passed = actual is not None and np.isclose(actual, target, atol=1e-8, rtol=1e-7)
        else:
            passed = actual == target
        output["fields_checked"] += 1
        if not passed and first_divergence is None:
            first_divergence = {"field": path, "actual": actual, "expected": target}
    for condition in reproduced:
        checks[condition] = {"fields_checked": 0, "subtrees": ["aggregate", "components", "temporal", "per_family"]}
        for root in checks[condition]["subtrees"]:
            compare_tree(reproduced[condition][root], expected[condition][root], root, checks[condition])
        actual_states = reproduced[condition]["candidate_filter_and_planning"]["planning"]["per_state"]
        target_states = expected[condition]["candidate_filter_and_planning"]["planning"]["per_state"]
        exact = [(row["state_id"], row["selected_candidate"]) for row in actual_states] == [(row["state_id"], row["selected_candidate"]) for row in target_states]
        checks[condition]["selected_candidates_exact"] = exact
        if not exact and first_divergence is None: first_divergence = {"condition": condition, "field": "selected_candidates"}
    passed = first_divergence is None and all(row["selected_candidates_exact"] for row in checks.values())
    return {"passed": passed, "tolerance": {"atol": 1e-8, "rtol": 1e-7}, "checks": checks,
            "first_divergence": first_divergence, "reproduced": reproduced}


def choose_component_threshold(probability: np.ndarray, labels: np.ndarray, minimum_recall: float = .90) -> dict:
    labels = labels.astype(bool)
    candidates = sorted({0., 1., *(float(np.nextafter(value, math.inf)) for value in probability)})
    scored = []
    for threshold in candidates:
        rejected = probability >= threshold
        recall = float(np.mean(rejected[labels])) if labels.any() else 1.
        retention = float(np.mean(~rejected[~labels])) if (~labels).any() else 0.
        scored.append((recall, retention, threshold))
    feasible = [row for row in scored if row[0] >= minimum_recall]
    chosen = max(feasible, key=lambda row: (row[1], -row[2])) if feasible else max(scored, key=lambda row: (row[0], row[1], -row[2]))
    return {"threshold": chosen[2], "calibration_recall": chosen[0], "calibration_negative_retention": chosen[1],
            "criterion_satisfied": bool(feasible), "minimum_recall": minimum_recall,
            "tie_rule": "more conservative (lower) threshold when negative retention ties"}


def component_calibration(logits: np.ndarray, targets: np.ndarray, active_index: int, cumulative_index: int) -> dict:
    h3_logits = logits[:, -1, cumulative_index]; labels = targets[:, -1, cumulative_index]
    temperature = DENSE.fit_temperature(h3_logits, labels)
    probability = sigmoid(h3_logits, temperature)
    return {"temperature": temperature, **choose_component_threshold(probability, labels),
            "calibration_average_precision": FS.average_precision(labels.astype(bool), probability),
            "calibration_auc": FS.auc(labels.astype(bool), probability), "active_output_index": active_index,
            "cumulative_output_index": cumulative_index}


def component_metrics(logits: np.ndarray, targets: np.ndarray, calibration: dict, active_index: int,
                      cumulative_index: int, families: list[str]) -> dict:
    probability = sigmoid(logits, calibration["temperature"]); threshold = calibration["threshold"]
    branch = DENSE.discrimination(targets[:, -1, cumulative_index], probability[:, -1, cumulative_index], threshold)
    active_truth = targets[:, :, active_index].astype(bool); active_prediction = probability[:, :, active_index] >= threshold
    delays = []; missed_transient = []
    for truth, prediction in zip(active_truth, active_prediction):
        events = np.flatnonzero(truth)
        if len(events):
            detected = np.flatnonzero(prediction & (np.arange(len(truth)) >= events[0]))
            delays.append(int(detected[0] - events[0]) if len(detected) else None)
        for lo, hi in DENSE.event_runs(truth.tolist()):
            if lo == hi: missed_transient.append(not bool(prediction[lo]))
    finite = [value for value in delays if value is not None]
    temporal = {"event_tick_recall": float(active_prediction[active_truth].mean()) if active_truth.any() else None,
                "first_event_detection_delay_ticks_median": float(np.median(finite)) if finite else None,
                "events_with_no_detection": sum(value is None for value in delays),
                "missed_transient_event_rate": float(np.mean(missed_transient)) if missed_transient else None,
                "transient_events": len(missed_transient)}
    per_family = {}
    families_array = np.asarray(families)
    for family in FAMILIES:
        mask = families_array == family
        per_family[family] = {
            "branch": DENSE.discrimination(targets[mask, -1, cumulative_index], probability[mask, -1, cumulative_index], threshold),
            "event_tick_recall": float(active_prediction[mask][active_truth[mask]].mean()) if active_truth[mask].any() else None,
        }
    return {"branch": branch, "temporal": temporal, "per_family": per_family,
            "probability": probability, "admitted": probability[:, -1, cumulative_index] < threshold}


def composition_evaluate(name: str, contact_probability: np.ndarray, stuck_probability: np.ndarray,
                         contact_calibration: dict, stuck_calibration: dict, targets: np.ndarray,
                         rows: list[dict], route_rows: list[dict], route_indices: np.ndarray, kinematic: np.ndarray) -> dict:
    contact_h3 = contact_probability[:, -1, 2]; stuck_h3 = stuck_probability[:, -1, 3]
    admitted = (contact_h3 < contact_calibration["threshold"]) & (stuck_h3 < stuck_calibration["threshold"])
    y = targets[:, -1, 4].astype(bool); tp = int(np.sum(y & ~admitted)); fn = int(np.sum(y & admitted))
    tn = int(np.sum(~y & admitted)); fp = int(np.sum(~y & ~admitted))
    risk_max = np.maximum(contact_h3, stuck_h3); risk_union = 1. - (1. - contact_h3) * (1. - stuck_h3)
    aggregate = {"rows": len(y), "unsafe": int(y.sum()), "safe": int((~y).sum()), "admitted": int(admitted.sum()),
                 "admitted_safe": tn, "admitted_unsafe": fn, "rejected_safe": fp, "rejected_unsafe": tp,
                 "unsafe_recall": tp / (tp + fn) if tp + fn else None,
                 "unsafe_false_negative_rate": fn / (tp + fn) if tp + fn else None,
                 "safe_candidate_retention": tn / (tn + fp) if tn + fp else None,
                 "risk_max": {"auc": FS.auc(y, risk_max), "average_precision": FS.average_precision(y, risk_max),
                              "ece": FS.ece(y, risk_max), "brier": float(np.mean((risk_max - y) ** 2))},
                 "risk_union": {"auc": FS.auc(y, risk_union), "average_precision": FS.average_precision(y, risk_union),
                                "ece": FS.ece(y, risk_union), "brier": float(np.mean((risk_union - y) ** 2))}}
    decision_risk = (~admitted).astype(np.float64)
    FS.rows_global = route_rows
    planning = FS.evaluate_condition(name, route_rows, route_indices, decision_risk, .5, kinematic)["planning"]
    improving_false_abstention = 0
    for state in planning["per_state"]:
        if state["selected_candidate"] is not None: continue
        candidates = [route_rows[index] for index in route_indices if route_rows[index]["state_id"] == state["state_id"]]
        if any((not row["unsafe"]) and (row["p_d"] > .03 or math.degrees(row["p_theta"]) > 5.) for row in candidates):
            improving_false_abstention += 1
    planning["states_hold_despite_safe_positive_progress"] = improving_false_abstention
    per_family = {}
    row_families = np.asarray([row["family"] for row in rows])
    for family in FAMILIES:
        mask = row_families == family; family_indices = route_indices[mask]
        fy = y[mask]; fa = admitted[mask]; fp_max = risk_max[mask]; fp_union = risk_union[mask]
        FS.rows_global = route_rows
        family_plan = FS.evaluate_condition(name, route_rows, family_indices, (~fa).astype(float), .5, kinematic)["planning"]
        per_family[family] = {"rows": int(mask.sum()), "unsafe_recall": float(np.mean(~fa[fy])) if fy.any() else None,
                              "unsafe_false_negative_rate": float(np.mean(fa[fy])) if fy.any() else None,
                              "safe_retention": float(np.mean(fa[~fy])) if (~fy).any() else None,
                              "risk_max_auc": FS.auc(fy, fp_max), "risk_union_auc": FS.auc(fy, fp_union), "planning": family_plan}
    return {"name": name, "aggregate": aggregate, "planning": planning, "per_family": per_family,
            "admitted": admitted, "risk_max": risk_max, "risk_union": risk_union}


def composition_gate(result: dict, contact: dict, stuck: dict, oracle_progress: float) -> dict:
    aggregate = result["aggregate"]; planning = result["planning"]
    no_family_collapse = all(row["planning"]["states_retaining_safe"] >= 1 and row["planning"]["abstention_rate"] < 1.
                             for row in result["per_family"].values())
    checks = {
        "aggregate_recall_ge_0_95": aggregate["unsafe_recall"] >= .95,
        "aggregate_fnr_le_0_05": aggregate["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": aggregate["safe_candidate_retention"] >= .40,
        "contact_recall_ge_0_90": contact["branch"]["unsafe_recall"] >= .90,
        "stuck_recall_ge_0_90": stuck["branch"]["unsafe_recall"] >= .90,
        "six_states_retain_safe": planning["states_retaining_safe"] >= 6,
        "no_state_only_unsafe_admitted": planning["states_only_unsafe_admitted"] == 0,
        "selected_unsafe_rate_zero": planning["selected_unsafe_rate"] == 0.,
        "false_abstention_le_1": planning["false_abstention"] <= 1,
        "progress_ge_80pct_oracle": planning["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": planning["normalized_safe_progress_regret"] is not None and planning["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": planning["best_safe_top3"] is not None and planning["best_safe_top3"] >= .75,
        "no_complete_family_collapse": no_family_collapse,
    }
    return {"passed": all(checks.values()), "checks": checks}


def public_component(value: dict) -> dict:
    return {key: row for key, row in value.items() if key not in ("probability", "admitted")}


def public_composition(value: dict) -> dict:
    return {key: row for key, row in value.items() if key not in ("admitted", "risk_max", "risk_union")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); started = time.time()
    action_branches, enhanced_branches, alignment = load_and_align(); rows = alignment["rows"]
    indices = split_indices(rows)
    route_rows = FS.load_metadata(); route_by_id = {row["state_id"] + f":{int(row['candidate_index']):02d}": i for i, row in enumerate(route_rows)}
    route_indices = {split: np.asarray([route_by_id[rows[index]["branch_id"]] for index in values], int) for split, values in indices.items()}
    kinematic = np.stack([row["kinematic"] for row in route_rows])
    inference_started = time.time(); action_logits, enhanced_logits, targets = frozen_inference(action_branches, enhanced_branches)
    inference_runtime = time.time() - inference_started
    reproduction = reproduction_check(action_branches, enhanced_branches, action_logits, enhanced_logits, targets,
                                     route_rows, route_indices["heldout"], kinematic)
    if not reproduction["passed"]:
        failure = {"classification": "FROZEN_SAFETY_CHECKPOINT_REPRODUCTION_FAILURE", "alignment": alignment["report"],
                   "reproduction": reproduction, "runtime_s": time.time() - started}
        atomic_json(OUT / "result.json", failure); print(json.dumps(failure, indent=2)); return 2
    calibrations = {}
    models = {"ACTION_CONTROL_ONLY": action_logits, "ENHANCED_EMBODIED": enhanced_logits}
    calibration_index = indices["calibration"]
    for condition, logits in models.items():
        calibrations[condition] = {
            "contact": component_calibration(logits[calibration_index], targets[calibration_index], 0, 2),
            "stuck": component_calibration(logits[calibration_index], targets[calibration_index], 1, 3),
        }
    primary_binding = dict(PRIMARY_BINDING)
    heldout = indices["heldout"]; heldout_rows = [rows[index] for index in heldout]
    primary_contact = component_metrics(enhanced_logits[heldout], targets[heldout], calibrations["ENHANCED_EMBODIED"]["contact"], 0, 2,
                                        [row["family"] for row in heldout_rows])
    primary_stuck = component_metrics(action_logits[heldout], targets[heldout], calibrations["ACTION_CONTROL_ONLY"]["stuck"], 1, 3,
                                      [row["family"] for row in heldout_rows])
    primary = composition_evaluate("PRIMARY_MECHANISM_SPECIFIC", primary_contact["probability"], primary_stuck["probability"],
        calibrations["ENHANCED_EMBODIED"]["contact"], calibrations["ACTION_CONTROL_ONLY"]["stuck"], targets[heldout], heldout_rows,
        route_rows, route_indices["heldout"], kinematic)
    FS.rows_global = route_rows; oracle_probability = np.asarray([float(route_rows[index]["unsafe"]) for index in route_indices["heldout"]])
    oracle = FS.evaluate_condition("oracle_safety", route_rows, route_indices["heldout"], oracle_probability, .5, kinematic)
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    primary["planning"]["oracle_progress_fraction"] = primary["planning"]["mean_selected_distance_progress_m"] / oracle_progress if abs(oracle_progress) > 1e-12 else None
    primary_gate = composition_gate(primary, primary_contact, primary_stuck, oracle_progress)
    # Secondary selection is intentionally restricted to the two lawfully regenerated conditions.
    selection = {}
    for component in ("contact", "stuck"):
        ranked = sorted(models, key=lambda name: (-calibrations[name][component]["calibration_average_precision"],
            -calibrations[name][component]["calibration_auc"], CONDITION_COMPLEXITY[name]))
        selection[component] = {"selected": ranked[0], "ranking": [{"condition": name,
            "average_precision": calibrations[name][component]["calibration_average_precision"],
            "auc": calibrations[name][component]["calibration_auc"], "complexity_tie_rank": CONDITION_COMPLEXITY[name]} for name in ranked]}
    secondary_contact_name = selection["contact"]["selected"]; secondary_stuck_name = selection["stuck"]["selected"]
    secondary_contact = component_metrics(models[secondary_contact_name][heldout], targets[heldout], calibrations[secondary_contact_name]["contact"], 0, 2,
                                          [row["family"] for row in heldout_rows])
    secondary_stuck = component_metrics(models[secondary_stuck_name][heldout], targets[heldout], calibrations[secondary_stuck_name]["stuck"], 1, 3,
                                        [row["family"] for row in heldout_rows])
    secondary = composition_evaluate("SECONDARY_CALIBRATION_SELECTED", secondary_contact["probability"], secondary_stuck["probability"],
        calibrations[secondary_contact_name]["contact"], calibrations[secondary_stuck_name]["stuck"], targets[heldout], heldout_rows,
        route_rows, route_indices["heldout"], kinematic)
    secondary["planning"]["oracle_progress_fraction"] = secondary["planning"]["mean_selected_distance_progress_m"] / oracle_progress if abs(oracle_progress) > 1e-12 else None
    secondary_gate = composition_gate(secondary, secondary_contact, secondary_stuck, oracle_progress)
    # Persist raw and calibrated row evidence for all later reducers.
    contact_probability_all = sigmoid(enhanced_logits, calibrations["ENHANCED_EMBODIED"]["contact"]["temperature"])
    stuck_probability_all = sigmoid(action_logits, calibrations["ACTION_CONTROL_ONLY"]["stuck"]["temperature"])
    contact_decision = contact_probability_all[:, -1, 2] >= calibrations["ENHANCED_EMBODIED"]["contact"]["threshold"]
    stuck_decision = stuck_probability_all[:, -1, 3] >= calibrations["ACTION_CONTROL_ONLY"]["stuck"]["threshold"]
    arrays = {
        "row_index": np.arange(576, dtype=np.int32), "branch_id": np.asarray([row["branch_id"] for row in rows]),
        "state_id": np.asarray([row["state_id"] for row in rows]), "candidate_index": np.asarray([row["candidate_index"] for row in rows], np.int16),
        "split": np.asarray([row["split"] for row in rows]), "family": np.asarray([row["family"] for row in rows]),
        "labels": targets.astype(np.float32), "action_control_logits": action_logits.astype(np.float32),
        "enhanced_embodied_logits": enhanced_logits.astype(np.float32),
        "action_control_sequence": np.stack([branch["action_control"] for branch in action_branches]).astype(np.float32),
        "enhanced_action_control_sequence": np.stack([branch["action_control"] for branch in enhanced_branches]).astype(np.float32),
        "primary_contact_probability": contact_probability_all.astype(np.float32),
        "primary_stuck_probability": stuck_probability_all.astype(np.float32),
        "primary_contact_reject": contact_decision, "primary_stuck_reject": stuck_decision,
        "primary_admitted": ~(contact_decision | stuck_decision),
    }
    ledger_path = OUT / "row_level_component_predictions_v1.npz"
    ledger_index_path = OUT / "row_level_component_predictions_v1_index.json"
    content_digest = array_content_digest(arrays)
    if ledger_path.is_file() or ledger_index_path.is_file():
        if not ledger_path.is_file() or not ledger_index_path.is_file():
            raise RuntimeError("partial immutable row-ledger custody")
        old_index = json.loads(ledger_index_path.read_text())
        if old_index.get("array_content_digest") != content_digest or old_index.get("file_sha256") != sha(ledger_path):
            raise RuntimeError("refusing to overwrite divergent immutable row ledger")
    else:
        atomic_npz(ledger_path, **arrays)
    ledger_index = {"schema": "row_level_component_predictions_v1", "rows": 576, "ticks": 15,
                    "raw_logit_dtype": "float32", "array_content_digest": content_digest,
                    "file_sha256": sha(ledger_path), "storage_bytes": ledger_path.stat().st_size,
                    "checkpoint_sha256": {"ACTION_CONTROL_ONLY": EXPECTED_ACTION_SHA, "ENHANCED_EMBODIED": EXPECTED_ENHANCED_SHA},
                    "fields": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
                    "sufficient_without_inference_for": ["aggregate metrics", "component composition", "operating-point analysis",
                        "candidate filtering", "state-level planning results"]}
    atomic_json(ledger_index_path, ledger_index)
    prior_matrix = json.loads((MATRIX_OUT / "result.json").read_text()); prior_enhanced = json.loads((ENHANCED_OUT / "result.json").read_text())
    guard = json.loads((GUARD_OUT / "result.json").read_text())
    comparators = {"ACTION_CONTROL_ONLY": prior_matrix["heldout"]["ACTION_CONTROL_ONLY"],
                   "PROPRIOCEPTION": prior_matrix["heldout"]["PROPRIOCEPTION"],
                   "RGB_PLUS_PROPRIOCEPTION": prior_matrix["heldout"]["RGB_PLUS_PROPRIOCEPTION"],
                   "ENHANCED_EMBODIED": prior_enhanced["heldout"],
                   "PRIVILEGED_STATIC_GRID_GUARD": {"classification": guard["classification"],
                       "heldout": guard["guard_discrimination"]["by_split"]["heldout"],
                       "per_family": guard["guard_discrimination"]["heldout_by_family"]},
                   "ORACLE_SAFETY_KINEMATIC": oracle}
    if primary_gate["passed"]:
        classification = "MECHANISM_SPECIFIC_SAFETY_COMPOSITION_SIGNAL"
        recommendation = "FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1 on a fresh panel: independent contact/impact and stuck/motion-shortfall states, losses, calibration, deterministic OR admission, and frozen kinematic ranking."
    elif secondary_gate["passed"]:
        classification = "SPECIALIST_COMPOSITION_POST_HOC_TENDENCY"
        recommendation = "Use only to prospectively freeze a specialist architecture; require a fresh evaluation panel before model development."
    else:
        classification = "MECHANISM_SPECIFIC_SAFETY_COMPOSITION_NO_SIGNAL"
        recommendation = "Prospectively design FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1 with independently trained contact/impact and stuck/motion-shortfall states and evaluate it on a fresh frozen panel; do not fit another post-hoc head automatically."
    result = {"schema": "mechanism_specific_safety_composition_inference_recovery_v1_result",
              "source_commit": "6982980178748cb1ca6eb42bd94245981593077a",
              "status": "POST_OUTCOME_DEVELOPMENT_DIAGNOSTIC", "preserved": ["ENHANCED_EMBODIED_SAFETY_POSITIVE_TENDENCY",
                  "MECHANISM_SPECIFIC_SAFETY_COMPOSITION_V1/ROW_ALIGNED_COMPONENT_PREDICTIONS_UNAVAILABLE"],
              "alignment": alignment["report"], "ledger": ledger_index, "metric_reproduction": reproduction,
              "component_calibration": calibrations, "primary_binding": primary_binding,
              "primary_contact": public_component(primary_contact), "primary_stuck": public_component(primary_stuck),
              "primary_composition": public_composition(primary), "primary_gate": primary_gate,
              "secondary": {"restricted_candidate_set": list(models), "excluded_without_inference": ["PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"],
                  "selection": selection, "binding": {"contact": secondary_contact_name, "stuck": secondary_stuck_name},
                  "contact": public_component(secondary_contact), "stuck": public_component(secondary_stuck),
                  "composition": public_composition(secondary), "gate": secondary_gate,
                  "differs_from_primary": secondary_contact_name != primary_binding["contact"] or secondary_stuck_name != primary_binding["stuck"]},
              "comparators": comparators, "classification": classification, "recommendation": recommendation,
              "prospective_persistence_rule": "ROW_LEVEL_EVIDENCE_PERSISTENCE",
              "runtime": {"total_s": time.time() - started, "inference_s": inference_runtime,
                          "ledger_storage_bytes": ledger_path.stat().st_size, "result_storage_bytes": 0},
              "custody": {"models_trained_or_finetuned": 0, "frozen_safety_checkpoints_executed": 2,
                          "simulation_or_rendering": False, "encoding": False, "jepa_predictor_opened": False,
                          "states_or_candidates_changed": False, "labels_or_splits_changed": False}}
    atomic_json(OUT / "result.json", result)
    result["runtime"]["result_storage_bytes"] = (OUT / "result.json").stat().st_size
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "ledger_digest": ledger_index["array_content_digest"],
        "ledger_sha256": ledger_index["file_sha256"], "ledger_bytes": ledger_index["storage_bytes"],
        "primary_gate": primary_gate["passed"], "secondary_binding": {"contact": secondary_contact_name, "stuck": secondary_stuck_name},
        "secondary_gate": secondary_gate["passed"], "result_sha256": sha(OUT / "result.json")}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
