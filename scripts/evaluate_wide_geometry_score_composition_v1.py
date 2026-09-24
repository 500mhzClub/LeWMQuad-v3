#!/usr/bin/env python3
"""Read-only frozen LiDAR/fusion score-composition frontier for V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".generated/geometry_modality_safety_sufficiency_v1/result.json"
OUT = ROOT / ".generated/wide_geometry_embodied_contact_proxy_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/wide_geometry_embodied_contact_proxy_v1")
LEDGER_ROOT = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/geometry_modality_safety_sufficiency_v1")
CHECKPOINTS = {
    "LIDAR_ONLY": "e225ac245e8e625750a59cd17cf8608e507b44a4d693946548e03f7db807d26b",
    "DEPTH_PLUS_EMBODIED": "8c51342d431c20496a60a69675851005cd9cc0d88f1440c9a583f1ae6d465204",
}
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
DELTA_D = 0.03
DELTA_THETA = np.deg2rad(5.0)


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


def auc(labels, scores) -> float:
    labels = np.asarray(labels, bool); scores = np.asarray(scores, np.float64)
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores), np.float64); start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1; start = end
    positive = int(labels.sum()); negative = int((~labels).sum())
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(labels, scores) -> float:
    labels = np.asarray(labels, bool); scores = np.asarray(scores, np.float64)
    order = np.argsort(-scores, kind="mergesort"); ranked = labels[order]
    return float((np.cumsum(ranked)[ranked] / (np.flatnonzero(ranked) + 1)).mean())


def threshold_values(probability) -> np.ndarray:
    return np.concatenate(([np.nextafter(0.0, -np.inf)], np.unique(probability), [np.nextafter(1.0, np.inf)]))


def route_order(rows, ids):
    remaining = list(ids); result = []
    while remaining:
        best_distance = max(float(rows[index]["kinematic"][4]) for index in remaining)
        near = [index for index in remaining if best_distance - float(rows[index]["kinematic"][4]) <= DELTA_D]
        pick = min(near, key=lambda index: (-float(rows[index]["kinematic"][5]), int(rows[index]["candidate_index"])))
        result.append(pick); remaining.remove(pick)
    return result


def preference(a, b) -> int:
    distance = float(a["p_d"]) - float(b["p_d"])
    if abs(distance) > DELTA_D:
        return 1 if distance > 0 else -1
    heading = float(a["p_theta"]) - float(b["p_theta"])
    if abs(heading) > DELTA_THETA:
        return 1 if heading > 0 else -1
    return 0


def load_ledgers():
    paths = {
        "LIDAR_ONLY": LEDGER_ROOT / "LIDAR_ONLY/row_level_evidence_v1.npz",
        "DEPTH_PLUS_EMBODIED": LEDGER_ROOT / "DEPTH_PLUS_EMBODIED/row_level_evidence_v1.npz",
    }
    ledgers = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as loaded:
            ledgers[name] = {key: np.asarray(loaded[key]) for key in loaded.files}
    identity = ("branch_id", "state_id", "candidate_index", "split", "family", "contact_labels", "stuck_labels", "p_d", "p_theta", "kinematic")
    for key in identity:
        if not np.array_equal(ledgers["LIDAR_ONLY"][key], ledgers["DEPTH_PLUS_EMBODIED"][key]):
            raise RuntimeError(f"row alignment mismatch: {key}")
    return ledgers, paths


def rows_for(ledger, split):
    mask = ledger["split"] == split; rows = []
    for index in np.flatnonzero(mask):
        rows.append({"branch_id": str(ledger["branch_id"][index]), "state_id": str(ledger["state_id"][index]),
            "candidate_index": int(ledger["candidate_index"][index]), "family": str(ledger["family"][index]),
            "hard_contact": bool(ledger["contact_labels"][index, -1, 1]), "stuck": bool(ledger["stuck_labels"][index, -1, 1]),
            "p_d": float(ledger["p_d"][index]), "p_theta": float(ledger["p_theta"][index]),
            "kinematic": np.asarray(ledger["kinematic"][index], np.float64)})
    return rows, mask


def prepare_states(rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows): grouped[row["state_id"]].append(index)
    output = []
    for state_id, ids in sorted(grouped.items()):
        rank = route_order(rows, ids); negative = [i for i in ids if not rows[i]["hard_contact"]]
        best = None
        for index in negative:
            if best is None or preference(rows[index], rows[best]) > 0: best = index
        oracle = next((index for index in rank if index in negative), None)
        output.append({"state_id": state_id, "family": rows[ids[0]]["family"], "ids": ids, "rank": rank,
                       "negative": negative, "best": best, "oracle": oracle})
    return output


def decision_metrics(rows, states, admitted, include_states=False):
    labels = np.asarray([row["hard_contact"] for row in rows], bool)
    progress = []; oracle_progress = []; selected_contact = []; selected_stuck = []; regrets = []; top1 = []; top3 = []
    false_abstentions = correct_abstentions = 0; per_state = []
    for state in states:
        ids, rank, negative = state["ids"], state["rank"], state["negative"]
        available = [index for index in rank if admitted[index]]; pick = available[0] if available else None
        admitted_negative = int(sum(admitted[index] and not labels[index] for index in ids))
        admitted_positive = int(sum(admitted[index] and labels[index] for index in ids))
        if pick is None:
            false_abstentions += bool(negative); correct_abstentions += not bool(negative)
        else:
            progress.append(rows[pick]["p_d"]); selected_contact.append(bool(labels[pick])); selected_stuck.append(rows[pick]["stuck"])
        if state["oracle"] is not None: oracle_progress.append(rows[state["oracle"]]["p_d"])
        if state["best"] is not None:
            top1.append(pick == state["best"]); top3.append(state["best"] in available[:3])
            if pick is not None and not labels[pick] and len(negative) >= 2:
                values = [rows[index]["p_d"] for index in negative]; spread = max(values) - min(values)
                if spread > 1e-8: regrets.append((rows[state["best"]]["p_d"] - rows[pick]["p_d"]) / spread)
        per_state.append({"state_id": state["state_id"], "family": state["family"], "admitted": int(admitted[ids].sum()),
            "admitted_contact_negative": admitted_negative, "admitted_contact_positive": admitted_positive,
            "selected_candidate": None if pick is None else rows[pick]["candidate_index"],
            "selected_contact": None if pick is None else bool(labels[pick]), "selected_stuck": None if pick is None else rows[pick]["stuck"],
            "selected_p_d": None if pick is None else rows[pick]["p_d"], "selected_p_theta": None if pick is None else rows[pick]["p_theta"],
            "oracle_contact_kinematic_p_d": None if state["oracle"] is None else rows[state["oracle"]]["p_d"]})
    selected_mean = float(np.mean(progress)) if progress else 0.0
    oracle_mean = float(np.mean(oracle_progress)) if oracle_progress else 0.0
    return {"contact_recall": float((~admitted)[labels].mean()), "contact_false_negative_rate": float(admitted[labels].mean()),
        "contact_negative_retention": float(admitted[~labels].mean()), "admitted_contact_negative_count": int((admitted & ~labels).sum()),
        "admitted_contact_positive_count": int((admitted & labels).sum()),
        "states_retaining_contact_negative": int(sum(row["admitted_contact_negative"] > 0 for row in per_state)),
        "states_only_contact_positive_admitted": int(sum(row["admitted"] > 0 and row["admitted_contact_negative"] == 0 for row in per_state)),
        "states_no_admitted": int(sum(row["admitted"] == 0 for row in per_state)), "false_abstentions": int(false_abstentions),
        "correct_abstentions": int(correct_abstentions), "selected_contact_count": int(sum(selected_contact)),
        "selected_stuck_count": int(sum(selected_stuck)), "mean_selected_route_progress_m": selected_mean,
        "oracle_contact_kinematic_progress_m": oracle_mean,
        "oracle_progress_fraction": None if abs(oracle_mean) <= 1e-9 else selected_mean / oracle_mean,
        "normalized_route_progress_regret": None if not regrets else float(np.mean(regrets)), "normalized_regret_states": len(regrets),
        "best_contact_negative_top1": None if not top1 else float(np.mean(top1)), "best_contact_negative_top3": None if not top3 else float(np.mean(top3)),
        "per_state": per_state if include_states else None}


def family_metrics(rows, states, admitted):
    output = {}
    for family in FAMILIES:
        state_ids = {row["state_id"] for row in states if row["family"] == family}
        ids = [index for index, row in enumerate(rows) if row["state_id"] in state_ids]
        remap = {old: new for new, old in enumerate(ids)}; subrows = [rows[index] for index in ids]
        substates = [{**state, "ids": [remap[i] for i in state["ids"]], "rank": [remap[i] for i in state["rank"]],
                      "negative": [remap[i] for i in state["negative"]],
                      "best": None if state["best"] is None else remap[state["best"]],
                      "oracle": None if state["oracle"] is None else remap[state["oracle"]]}
                     for state in states if state["family"] == family]
        output[family] = decision_metrics(subrows, substates, admitted[ids], True)
    return output


def complete_gate(metrics, combined_auc, families):
    no_collapse = all(value["states_retaining_contact_negative"] > 0 and value["selected_contact_count"] == 0 for value in families.values())
    checks = {"combined_auc_ge_0_90": combined_auc >= .90, "recall_ge_0_95": metrics["contact_recall"] >= .95,
        "fnr_le_0_05": metrics["contact_false_negative_rate"] <= .05, "negative_retention_ge_0_50": metrics["contact_negative_retention"] >= .50,
        "states_retaining_ge_18": metrics["states_retaining_contact_negative"] >= 18,
        "no_only_positive_state": metrics["states_only_contact_positive_admitted"] == 0,
        "zero_selected_contact": metrics["selected_contact_count"] == 0, "false_abstentions_le_3": metrics["false_abstentions"] <= 3,
        "progress_fraction_ge_0_80": metrics["oracle_progress_fraction"] is not None and metrics["oracle_progress_fraction"] >= .80,
        "regret_le_0_20": metrics["normalized_route_progress_regret"] is not None and metrics["normalized_route_progress_regret"] <= .20,
        "top3_ge_0_75": metrics["best_contact_negative_top3"] is not None and metrics["best_contact_negative_top3"] >= .75,
        "no_family_collapse": no_collapse}
    return {"checks": checks, "passed": all(checks.values())}


def frontier(split, rows, states, lidar_probability, fusion_probability, select=False):
    started = time.time(); labels = np.asarray([row["hard_contact"] for row in rows], bool)
    lidar_thresholds, fusion_thresholds = threshold_values(lidar_probability), threshold_values(fusion_probability)
    fields = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "admitted_contact_negative_count",
              "admitted_contact_positive_count", "states_retaining_contact_negative", "states_only_contact_positive_admitted",
              "states_no_admitted", "false_abstentions", "selected_contact_count", "mean_selected_route_progress_m",
              "oracle_progress_fraction", "normalized_route_progress_regret", "best_contact_negative_top1", "best_contact_negative_top3")
    values = defaultdict(list); eligible = []; gate_points = []
    union_risk = 1 - (1 - lidar_probability) * (1 - fusion_probability); union_auc = auc(labels, union_risk)
    max_risk = np.maximum(lidar_probability, fusion_probability)
    lidar_masks = [lidar_probability < threshold for threshold in lidar_thresholds]
    fusion_masks = [fusion_probability < threshold for threshold in fusion_thresholds]
    for lidar_index, lidar_admitted in enumerate(lidar_masks):
        for fusion_index, fusion_admitted in enumerate(fusion_masks):
            admitted = lidar_admitted & fusion_admitted; metric = decision_metrics(rows, states, admitted)
            values["lidar_threshold"].append(float(lidar_thresholds[lidar_index])); values["fusion_threshold"].append(float(fusion_thresholds[fusion_index]))
            for field in fields:
                value = metric[field]; values[field].append(np.nan if value is None else value)
            is_eligible = (metric["contact_recall"] >= .95 and metric["contact_false_negative_rate"] <= .05
                and metric["states_only_contact_positive_admitted"] == 0 and metric["selected_contact_count"] == 0)
            values["calibration_eligible"].append(is_eligible)
            if is_eligible:
                regret = metric["normalized_route_progress_regret"]
                key = (metric["states_retaining_contact_negative"], metric["contact_negative_retention"], metric["mean_selected_route_progress_m"],
                       -(regret if regret is not None else 1e9), -metric["false_abstentions"],
                       metric["best_contact_negative_top3"] or -1.0, -float(lidar_thresholds[lidar_index]), -float(fusion_thresholds[fusion_index]))
                eligible.append((key, float(lidar_thresholds[lidar_index]), float(fusion_thresholds[fusion_index]), metric))
            if not select and is_eligible:
                families = family_metrics(rows, states, admitted); gate = complete_gate(metric, union_auc, families)
                values["complete_gate"].append(gate["passed"])
                if gate["passed"]: gate_points.append((float(lidar_thresholds[lidar_index]), float(fusion_thresholds[fusion_index]), metric, families))
            else:
                values["complete_gate"].append(False)
    arrays = {key: np.asarray(value) for key, value in values.items()}
    path = CACHE / f"stage0_{split}_joint_threshold_frontier_v1.npz"; atomic_npz(path, **arrays)
    return {"pairs": len(arrays["lidar_threshold"]), "lidar_thresholds": len(lidar_thresholds), "fusion_thresholds": len(fusion_thresholds),
        "eligible_pairs": len(eligible), "complete_gate_pairs": len(gate_points), "combined_union_auc": union_auc,
        "combined_union_ap": average_precision(labels, union_risk), "combined_max_auc": auc(labels, max_risk),
        "frontier": {"file": str(path), "sha256": sha(path), "content_digest": hashlib.sha256(b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))).hexdigest()},
        "selected": None if not eligible else max(eligible, key=lambda item: item[0]), "gate_points": gate_points,
        "runtime_s": time.time() - started}


def reproduce(source, ledgers, paths):
    output = {}
    for name in ("LIDAR_ONLY", "DEPTH_PLUS_EMBODIED"):
        expected_sha = source["conditions"][name]["row_level_evidence"]["sha256"]
        if sha(paths[name]) != expected_sha: raise RuntimeError(f"{name}: ledger SHA mismatch")
        checkpoint = Path(source["conditions"][name]["training"]["checkpoint"])
        if sha(checkpoint) != CHECKPOINTS[name]: raise RuntimeError(f"{name}: checkpoint SHA mismatch")
        ledger = ledgers[name]; rows, mask = rows_for(ledger, "heldout"); states = prepare_states(rows)
        probability = ledger["calibrated_probability"][mask, -1, 1]
        threshold = float(source["conditions"][name]["calibration"]["threshold"])
        metric = decision_metrics(rows, states, probability < threshold, True)
        expected = source["conditions"][name]["heldout"]["filtering_and_planning"]
        keys = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "selected_contact_count",
                "states_retaining_contact_negative", "false_abstentions", "mean_selected_route_progress_m", "normalized_route_progress_regret")
        for key in keys:
            tolerance = 2e-7 if key in ("mean_selected_route_progress_m", "normalized_route_progress_regret") else 1e-12
            if not np.isclose(metric[key], expected[key], atol=tolerance, rtol=tolerance): raise RuntimeError(f"{name}: reproduction mismatch {key}")
        selected = {(row["state_id"], row["selected_candidate"]) for row in metric["per_state"] if row["selected_candidate"] is not None}
        persisted = {(str(ledger["state_id"][i]), int(ledger["candidate_index"][i])) for i in np.flatnonzero(ledger["selected"] & mask)}
        if selected != persisted: raise RuntimeError(f"{name}: selected-candidate mismatch")
        output[name] = {"status": "PASS", "checkpoint_sha256": CHECKPOINTS[name], "ledger_sha256": expected_sha,
                        "metrics": {key: metric[key] for key in keys}, "selected_candidates_exact": True,
                        "route_float_reduction_tolerance": 2e-7, "reason": "persisted route labels are FP32; contact counts, decisions, and selections reproduce exactly"}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text()); ledgers, paths = load_ledgers(); reproduction = reproduce(source, ledgers, paths)
    lidar, fusion = ledgers["LIDAR_ONLY"], ledgers["DEPTH_PLUS_EMBODIED"]
    cal_rows, cal_mask = rows_for(lidar, "calibration"); cal_states = prepare_states(cal_rows)
    cal = frontier("calibration", cal_rows, cal_states, lidar["calibrated_probability"][cal_mask, -1, 1],
                   fusion["calibrated_probability"][cal_mask, -1, 1], True)
    if cal["selected"] is None: raise RuntimeError("Stage-0 joint calibration feasibility failure")
    _, lidar_threshold, fusion_threshold, calibration_metrics = cal.pop("selected"); cal.pop("gate_points")
    held_rows, held_mask = rows_for(lidar, "heldout"); held_states = prepare_states(held_rows)
    held_lidar = lidar["calibrated_probability"][held_mask, -1, 1]; held_fusion = fusion["calibrated_probability"][held_mask, -1, 1]
    admitted = (held_lidar < lidar_threshold) & (held_fusion < fusion_threshold)
    primary_metrics = decision_metrics(held_rows, held_states, admitted, True); primary_families = family_metrics(held_rows, held_states, admitted)
    union_auc = auc(np.asarray([row["hard_contact"] for row in held_rows]), 1 - (1 - held_lidar) * (1 - held_fusion))
    primary_gate = complete_gate(primary_metrics, union_auc, primary_families)
    held = frontier("heldout", held_rows, held_states, held_lidar, held_fusion, False); held.pop("selected"); gate_points = held.pop("gate_points")
    if primary_gate["passed"]: classification = "FROZEN_WIDE_GEOMETRY_SCORE_COMPOSITION_SIGNAL"
    elif gate_points: classification = "WIDE_GEOMETRY_CALIBRATION_BOTTLENECK"
    else: classification = "WIDE_GEOMETRY_SCORE_FRONTIER_NO_GO"
    selected = {(row["state_id"], row["selected_candidate"]) for row in primary_metrics["per_state"] if row["selected_candidate"] is not None}
    arrays = {"branch_id": lidar["branch_id"], "state_id": lidar["state_id"], "candidate_index": lidar["candidate_index"], "split": lidar["split"],
        "family": lidar["family"], "lidar_probability": lidar["calibrated_probability"][:, -1, 1],
        "fusion_probability": fusion["calibrated_probability"][:, -1, 1], "contact_label": lidar["contact_labels"][:, -1, 1],
        "admitted_at_selected_pair": ((lidar["calibrated_probability"][:, -1, 1] < lidar_threshold) & (fusion["calibrated_probability"][:, -1, 1] < fusion_threshold)).astype(np.uint8),
        "selected_at_selected_pair": np.asarray([(str(lidar["state_id"][i]), int(lidar["candidate_index"][i])) in selected for i in range(len(lidar["state_id"]))], np.uint8)}
    ledger_path = CACHE / "stage0_row_level_composition_v1.npz"; atomic_npz(ledger_path, **arrays)
    result = {"schema": "wide_geometry_embodied_contact_proxy_v1_stage0_result", "source_commit": "58d91deb37d41a129e64a6a0c17ae8b9b6f135d2",
        "claim_boundary": "SIMULATED_DISALLOWED_CONTACT_PROXY only; no material-hazard, injury, property-damage, people, or fragile-infrastructure claim",
        "preserved_ontology_result": "CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT", "reproduction": reproduction,
        "continuous_combined_risk": "probabilistic union 1-(1-p_lidar)(1-p_fusion); admission remains separate-threshold OR rejection",
        "calibration_frontier": cal, "selected_thresholds": {"lidar": lidar_threshold, "fusion": fusion_threshold},
        "calibration_metrics": calibration_metrics, "heldout_primary": {"metrics": primary_metrics, "per_family": primary_families, "gate": primary_gate},
        "heldout_oracle_frontier": held, "heldout_operating_point_exists": bool(gate_points), "classification": classification,
        "row_level_ledger": {"file": str(ledger_path), "sha256": sha(ledger_path)}, "stage1_authorized": classification == "WIDE_GEOMETRY_SCORE_FRONTIER_NO_GO",
        "runtime_s": time.time() - started}
    result["content_digest"] = canonical_digest(result); atomic_json(OUT / "stage0_result.json", result)
    print(json.dumps({"classification": classification, "selected_thresholds": result["selected_thresholds"],
        "calibration_metrics": calibration_metrics, "heldout_primary": {k: primary_metrics[k] for k in primary_metrics if k != "per_state"},
        "heldout_complete_gate_pairs": held["complete_gate_pairs"], "runtime_s": result["runtime_s"]}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
