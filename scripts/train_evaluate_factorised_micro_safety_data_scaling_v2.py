#!/usr/bin/env python3
"""Train the two new nested-fit conditions and evaluate the frozen scaling curve."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import collect_factorised_micro_safety_data_scaling_v2 as COLLECT
from scripts import train_evaluate_factorised_micro_safety_world_model_v1 as BASE

OUT = ROOT / ".generated/factorised_micro_safety_data_scaling_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorised_micro_safety_data_scaling_v2")
INDEX = OUT / "sensor_index.json"
V1_RESULT = ROOT / ".generated/factorised_micro_safety_world_model_v1/result.json"
SEED_FAMILY = 2026082011
EXPECTED_FIT48_SHA = "93f919238ff7b757b77f5281f45c59818c9f2b33fa5fbd96a2554b7aea14776e"
FAMILIES = BASE.FAMILIES


def sha(path: Path) -> str: return BASE.sha(path)
def canonical_digest(value) -> str: return BASE.canonical_digest(value)
def atomic_json(path: Path, payload: dict) -> None: BASE.atomic_json(path, payload)
def atomic_npz(path: Path, **arrays) -> None: BASE.atomic_npz(path, **arrays)


def derived_seed(condition: str) -> int:
    value = hashlib.sha256(f"{SEED_FAMILY}|{condition}".encode()).digest()
    return int.from_bytes(value[:8], "big") % (2**31 - 1)


def load_new_rows() -> tuple[list[dict], dict]:
    index = json.loads(INDEX.read_text())
    without = {k: v for k, v in index.items() if k != "content_digest"}
    if not index.get("complete") or index.get("content_digest") != canonical_digest(without):
        raise RuntimeError("scaling sensor index incomplete or digest-invalid")
    manifest = json.loads((OUT / "panel_manifest.json").read_text())
    states = {row["state_id"]: row for row in manifest["states"]}; rows = []
    for record in index["state_records"]:
        state = states[record["state_id"]]
        if sha(Path(record["shard_path"])) != record["shard_sha256"]: raise RuntimeError(f"shard mismatch {record['state_id']}")
        with np.load(record["shard_path"], allow_pickle=False) as loaded:
            current = np.asarray(loaded["current"], np.float32); future = np.asarray(loaded["future"], np.float32)
            action = np.asarray(loaded["action_control"], np.float32); labels = np.asarray(loaded["labels"], np.float32)
        heading_body = float(record["branches"][0]["route_heading_world_rad"]) - float(state["start_pose"][1])
        waypoint = [*map(float, state["waypoint_body_xy"]), np.sin(heading_body), np.cos(heading_body)]
        for branch in record["branches"]:
            ci = int(branch["candidate_index"])
            rows.append({**branch, "split": record["split"], "family": record["family"], "scene_id": record["scene_id"],
                         "current_enhanced": current[ci], "future_enhanced": future[ci], "action_control": action[ci],
                         "target": labels[ci], "kinematic": BASE.integrate(branch["post_slew"], waypoint)})
    if len(rows) != 1728 or len({row["state_id"] for row in rows}) != 144: raise RuntimeError("scaling row cardinality mismatch")
    return rows, index


def one_prevalence(rows: list[dict]) -> dict:
    unsafe = np.asarray([row["target"][-1, 4] > .5 for row in rows]); contact = np.asarray([row["target"][-1, 2] > .5 for row in rows])
    stuck = np.asarray([row["target"][-1, 3] > .5 for row in rows]); grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    safe_counts = {key: sum(not bool(row["target"][-1, 4]) for row in value) for key, value in grouped.items()}
    return {"states": len(grouped), "branches": len(rows), "safe": int((~unsafe).sum()), "unsafe": int(unsafe.sum()),
            "contact_positive": int(contact.sum()), "stuck_positive": int(stuck.sum()), "contact_stuck_overlap": int((contact & stuck).sum()),
            "contact_positive_event_ticks": int(sum(row["target"][:, 0].sum() for row in rows)),
            "stuck_positive_event_ticks": int(sum(row["target"][:, 1].sum() for row in rows)),
            "states_no_safe_candidate": sum(value == 0 for value in safe_counts.values()), "safe_candidates_per_state": safe_counts}


def prevalence(inventories: dict[str, list[dict]]) -> dict:
    report = {"inventories": {}, "per_family": {}}
    for name, rows in inventories.items():
        report["inventories"][name] = one_prevalence(rows)
        report["per_family"][name] = {family: one_prevalence([row for row in rows if row["family"] == family]) for family in FAMILIES}
    checks = {}
    for split in ("calibration", "heldout"):
        item = report["inventories"][split]
        checks[split] = {"safe_and_unsafe": item["safe"] > 0 and item["unsafe"] > 0,
                         "contact_at_least_24": item["contact_positive"] >= 24,
                         "stuck_at_least_24": item["stuck_positive"] >= 24,
                         "four_safe_states_per_family": all(6 - report["per_family"][split][family]["states_no_safe_candidate"] >= 4 for family in FAMILIES),
                         "each_family_contact_and_stuck": all(report["per_family"][split][family]["contact_positive"] > 0 and report["per_family"][split][family]["stuck_positive"] > 0 for family in FAMILIES)}
    report["checks"] = checks; report["passed"] = all(all(item.values()) for item in checks.values())
    return report


def fit48_bundle(device):
    result = json.loads(V1_RESULT.read_text()); checkpoint = Path(result["training"]["checkpoint"])
    if sha(checkpoint) != EXPECTED_FIT48_SHA: raise RuntimeError("FIT-48 checkpoint SHA mismatch")
    package = torch.load(checkpoint, map_location=device, weights_only=True); model = BASE.FactorisedModel().to(device)
    model.load_state_dict(package["state_dict"]); model.eval()
    return model, package["contact_stats"], package["stuck_stats"], {"checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "retrained": False, "seed": 2026082010, "epochs": 60, "parameter_count": package["parameter_count"],
        "historical_result_classification": result["classification"], "historical_training": result["training"]}


def train_condition(name: str, rows: list[dict], device):
    condition_out = OUT / name; condition_cache = CACHE / name; condition_out.mkdir(parents=True, exist_ok=True); condition_cache.mkdir(parents=True, exist_ok=True)
    old_seed, old_out, old_cache = BASE.SEED, BASE.OUT, BASE.CACHE
    seed = derived_seed(name)
    try:
        BASE.SEED, BASE.OUT, BASE.CACHE = seed, condition_out, condition_cache
        model, contact_stats, stuck_stats, bundle = BASE.train(rows, device)
    finally:
        BASE.SEED, BASE.OUT, BASE.CACHE = old_seed, old_out, old_cache
    checkpoint = condition_out / f"factorised_micro_safety_{name}_seed_{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "seed_family": SEED_FAMILY, "derived_seed": seed, "condition": name,
                "epoch": 60, "contact_stats": contact_stats, "stuck_stats": stuck_stats,
                "parameter_count": bundle["training"]["parameter_count"]}, checkpoint)
    clone = BASE.FactorisedModel().to(device); package = torch.load(checkpoint, map_location=device, weights_only=True)
    clone.load_state_dict(package["state_dict"]); clone.eval()
    training = {**bundle["training"], "history": bundle["history"], "seed_family": SEED_FAMILY, "derived_seed": seed,
                "rng_derivation": "sha256(f'{seed_family}|{condition}')[:8] mod (2**31-1)", "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size, "retrained": True}
    return clone, contact_stats, stuck_stats, training


def calibrate_condition(name, rows, logits):
    condition_out = OUT / name; condition_cache = CACHE / name
    old_out, old_cache = BASE.OUT, BASE.CACHE
    try:
        BASE.OUT, BASE.CACHE = condition_out, condition_cache
        return BASE.calibrate(rows, logits)
    finally:
        BASE.OUT, BASE.CACHE = old_out, old_cache


def component_gate(components, decision, per_family) -> tuple[dict, bool]:
    no_family = all(value["states_retaining_safe"] > 0 and value["selected_unsafe_count"] == 0 for value in per_family.values())
    checks = {"unsafe_recall": decision["aggregate_unsafe_recall"] >= .95, "unsafe_fnr": decision["aggregate_false_negative_rate"] <= .05,
        "safe_retention": decision["safe_candidate_retention"] >= .40, "contact_recall": decision["contact_recall"] >= .90,
        "stuck_recall": decision["stuck_recall"] >= .90, "contact_auc": components["contact"]["auc"] >= .80,
        "stuck_auc": components["stuck"]["auc"] >= .85, "states_retaining_safe": decision["states_retaining_safe"] >= 18,
        "no_only_unsafe_state": decision["states_only_unsafe_admitted"] == 0, "zero_selected_unsafe": decision["selected_unsafe_rate"] == 0,
        "false_abstentions": decision["false_abstentions"] <= 3,
        "oracle_progress_fraction": decision["oracle_progress_fraction"] is not None and decision["oracle_progress_fraction"] >= .80,
        "normalized_regret": decision["normalized_safe_progress_regret"] is not None and decision["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3": decision["best_safe_top3"] is not None and decision["best_safe_top3"] >= .75, "no_family_collapse": no_family}
    return checks, all(checks.values())


def persist_ledger(name, rows, logits, temperatures, tc, ts):
    full_probability = {component: 1 / (1 + np.exp(-logits[component] / temperatures[component])) for component in ("contact", "stuck")}
    pc, ps = full_probability["contact"][:, -1, 1], full_probability["stuck"][:, -1, 1]; admitted = (pc < tc) & (ps < ts)
    selected = set()
    for split in ("calibration", "heldout"):
        ids = [index for index, row in enumerate(rows) if row["split"] == split]
        plan = BASE.planning_metrics([rows[index] for index in ids], admitted[ids], True)
        selected.update((split, row["state_id"], row["selected_candidate"]) for row in plan["per_state"] if row["selected_candidate"] is not None)
    arrays = {"branch_id": np.asarray([row["branch_id"] for row in rows]), "state_id": np.asarray([row["state_id"] for row in rows]),
        "candidate_index": np.asarray([row["candidate_index"] for row in rows], np.int16), "split": np.asarray([row["split"] for row in rows]),
        "family": np.asarray([row["family"] for row in rows]), "contact_logits": logits["contact"].astype(np.float32),
        "stuck_logits": logits["stuck"].astype(np.float32), "contact_probability": full_probability["contact"].astype(np.float64),
        "stuck_probability": full_probability["stuck"].astype(np.float64), "labels": np.stack([row["target"] for row in rows]).astype(np.uint8),
        "admitted": admitted.astype(np.uint8), "selected": np.asarray([(row["split"], row["state_id"], row["candidate_index"]) in selected for row in rows], np.uint8),
        "candidate_action_control": np.stack([row["action_control"] for row in rows]).astype(np.float32),
        "p_d": np.asarray([row["p_d"] for row in rows], np.float32), "p_theta": np.asarray([row["p_theta"] for row in rows], np.float32),
        "kinematic": np.stack([row["kinematic"] for row in rows]).astype(np.float32)}
    path = CACHE / name / "fresh_evaluation_row_level_evidence_v1.npz"; atomic_npz(path, **arrays)
    index = {"schema": "factorised_micro_safety_scaling_row_level_evidence_v1", "condition": name, "rows": len(rows),
        "states": len(set(arrays["state_id"])), "file": str(path), "sha256": sha(path), "content_digest": BASE.array_digest(arrays),
        "contact_threshold": tc, "stuck_threshold": ts, "fields": sorted(arrays), "row_level_evidence_persistence": True}
    atomic_json(OUT / name / "row_level_evidence_index.json", index); return index


def oracle_frontier(name, rows, logits, temperatures, components):
    pc = 1 / (1 + np.exp(-logits["contact"][:, -1, 1] / temperatures["contact"])); ps = 1 / (1 + np.exp(-logits["stuck"][:, -1, 1] / temperatures["stuck"]))
    fields = ("aggregate_unsafe_recall", "aggregate_false_negative_rate", "safe_candidate_retention", "states_retaining_safe",
              "selected_unsafe_count", "false_abstentions", "mean_selected_route_progress_m", "normalized_safe_progress_regret", "best_safe_top3")
    values = {key: [] for key in ("contact_threshold", "stuck_threshold", *fields)}; any_pass = False; pass_pairs = 0
    best = {"max_safe_retention_at_recall_095": 0., "max_states_retaining_at_recall_095": 0,
            "max_route_progress_zero_selected_unsafe": None, "min_regret_at_recall_095": None}
    started = time.time()
    for tc in BASE.threshold_values(pc):
        for ts in BASE.threshold_values(ps):
            metric = BASE.decision_metrics(rows, pc, ps, tc, ts)
            values["contact_threshold"].append(tc); values["stuck_threshold"].append(ts)
            for field in fields: values[field].append(np.nan if metric[field] is None else metric[field])
            if metric["aggregate_unsafe_recall"] >= .95:
                best["max_safe_retention_at_recall_095"] = max(best["max_safe_retention_at_recall_095"], metric["safe_candidate_retention"])
                best["max_states_retaining_at_recall_095"] = max(best["max_states_retaining_at_recall_095"], metric["states_retaining_safe"])
                if metric["normalized_safe_progress_regret"] is not None:
                    old = best["min_regret_at_recall_095"]; best["min_regret_at_recall_095"] = metric["normalized_safe_progress_regret"] if old is None else min(old, metric["normalized_safe_progress_regret"])
            if metric["selected_unsafe_count"] == 0:
                old = best["max_route_progress_zero_selected_unsafe"]
                best["max_route_progress_zero_selected_unsafe"] = metric["mean_selected_route_progress_m"] if old is None else max(old, metric["mean_selected_route_progress_m"])
            prelim = (metric["aggregate_unsafe_recall"] >= .95 and metric["aggregate_false_negative_rate"] <= .05 and
                      metric["safe_candidate_retention"] >= .40 and metric["contact_recall"] >= .90 and metric["stuck_recall"] >= .90 and
                      components["contact"]["auc"] >= .80 and components["stuck"]["auc"] >= .85 and metric["states_retaining_safe"] >= 18 and
                      metric["states_only_unsafe_admitted"] == 0 and metric["selected_unsafe_count"] == 0 and metric["false_abstentions"] <= 3 and
                      metric["oracle_progress_fraction"] is not None and metric["oracle_progress_fraction"] >= .80 and
                      metric["normalized_safe_progress_regret"] is not None and metric["normalized_safe_progress_regret"] <= .20 and
                      metric["best_safe_top3"] is not None and metric["best_safe_top3"] >= .75)
            if prelim:
                family = BASE.family_results(rows, pc, ps, tc, ts, logits["contact"], logits["stuck"], temperatures)
                complete = all(item["states_retaining_safe"] > 0 and item["selected_unsafe_count"] == 0 for item in family.values())
                any_pass |= complete; pass_pairs += int(complete)
    arrays = {key: np.asarray(value, np.float64) for key, value in values.items()}
    path = CACHE / name / "heldout_oracle_threshold_frontier_v1.npz"; atomic_npz(path, **arrays)
    return {**best, "pairs": len(arrays["contact_threshold"]), "complete_gate_pair_exists": any_pass,
            "complete_gate_pairs": pass_pairs, "runtime_s": time.time() - started, "file": str(path), "sha256": sha(path),
            "content_digest": BASE.array_digest(arrays), "post_hoc_oracle_diagnostic_only": True}


def evaluate_condition(name, model, contact_stats, stuck_stats, training, fit_rows, calibration_rows, heldout_rows, device):
    cal_logits = BASE.predict(model, calibration_rows, contact_stats, stuck_stats, device)
    temperatures, _unused, calibration = calibrate_condition(name, calibration_rows, cal_logits)
    if not calibration["feasible"]: raise RuntimeError(f"{name}: no eligible calibration threshold pair")
    tc, ts = calibration["contact_threshold"], calibration["stuck_threshold"]
    held_logits = BASE.predict(model, heldout_rows, contact_stats, stuck_stats, device)
    held_prob = {component: 1 / (1 + np.exp(-held_logits[component][:, -1, 1] / temperatures[component])) for component in ("contact", "stuck")}
    decision = BASE.decision_metrics(heldout_rows, held_prob["contact"], held_prob["stuck"], tc, ts, True)
    components = {component: BASE.component_metrics(heldout_rows, held_logits[component], temperatures[component], tc if component == "contact" else ts, component) for component in ("contact", "stuck")}
    per_family = BASE.family_results(heldout_rows, held_prob["contact"], held_prob["stuck"], tc, ts, held_logits["contact"], held_logits["stuck"], temperatures)
    checks, passed = component_gate(components, decision, per_family)
    fit_logits = BASE.predict(model, fit_rows, contact_stats, stuck_stats, device)
    fit_components = {component: BASE.component_metrics(fit_rows, fit_logits[component], temperatures[component], tc if component == "contact" else ts, component) for component in ("contact", "stuck")}
    all_rows = calibration_rows + heldout_rows
    all_logits = {component: np.concatenate((cal_logits[component], held_logits[component])) for component in ("contact", "stuck")}
    ledger = persist_ledger(name, all_rows, all_logits, temperatures, tc, ts)
    oracle = oracle_frontier(name, heldout_rows, held_logits, temperatures, components)
    return {"condition": name, "training": training, "calibration": {**calibration, "temperatures": temperatures},
        "heldout": {"components": components, "combined_admissibility_and_planning": decision, "per_family": per_family},
        "fit_diagnostics": {"states": len({row['state_id'] for row in fit_rows}), "branches": len(fit_rows), "components": fit_components,
                            "auc_gap_heldout_minus_fit": {component: components[component]["auc"] - fit_components[component]["auc"] for component in ("contact", "stuck")}},
        "prospective_gate": {"checks": checks, "passed": passed}, "oracle_frontier": oracle, "row_level_evidence": ledger}


def learning_curve(conditions: dict) -> dict:
    def values(item):
        held = item["heldout"]; dec = held["combined_admissibility_and_planning"]
        return {"contact_auc": held["components"]["contact"]["auc"], "stuck_auc": held["components"]["stuck"]["auc"],
            "safe_retention": dec["safe_candidate_retention"], "states_retaining_safe": dec["states_retaining_safe"],
            "false_abstentions": dec["false_abstentions"], "selected_progress": dec["mean_selected_route_progress_m"],
            "normalized_regret": dec["normalized_safe_progress_regret"], "best_safe_top3": dec["best_safe_top3"]}
    ordered = {name: values(conditions[name]) for name in ("fit48", "fit96", "fit192")}
    changes = {transition: {key: ordered[b][key] - ordered[a][key] if ordered[a][key] is not None and ordered[b][key] is not None else None
                            for key in ordered[a]} for transition, a, b in (("48_to_96", "fit48", "fit96"), ("96_to_192", "fit96", "fit192"))}
    lower = {"false_abstentions", "normalized_regret"}; monotonic = {}
    for key in ordered["fit48"]:
        triplet = [ordered[name][key] for name in ("fit48", "fit96", "fit192")]
        monotonic[key] = False if any(value is None for value in triplet) else ((triplet[0] >= triplet[1] >= triplet[2]) if key in lower else (triplet[0] <= triplet[1] <= triplet[2]))
    return {"ordered": ordered, "absolute_change": changes, "monotonic": monotonic,
            "all_primary_quantities_monotonic": all(monotonic.values()), "monotonic_count": sum(monotonic.values()), "primary_quantity_count": len(monotonic),
            "positive_tendency_rule_frozen_before_scoring": {"material_improvement": "at least five of eight primary metrics improve directionally FIT48-to-FIT192",
                "broad_monotonicity": "at least six of eight primary metrics are monotonic across all three sizes",
                "safe_retention_delta": ">=0.20", "state_retention_delta": ">=6", "no_new_unsafe_selection": True}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True); started = time.time()
    old_out = BASE.OUT
    try:
        BASE.OUT = OUT
        fixture = BASE.evaluator_fixture()
    finally: BASE.OUT = old_out
    old48, old_index = BASE.old_fit_rows(); prior48, prior_index = BASE.fresh_rows(); new, sensor_index = load_new_rows()
    fit96 = old48 + prior48; fit192 = fit96 + [row for row in new if row["split"] == "fit192_extra"]
    calibration_rows = [row for row in new if row["split"] == "calibration"]; heldout_rows = [row for row in new if row["split"] == "heldout"]
    if (len(fit96), len(fit192), len(calibration_rows), len(heldout_rows)) != (1152, 2304, 288, 288): raise RuntimeError("nested inventory cardinality failure")
    inventories = {"fit96": fit96, "fit192": fit192, "calibration": calibration_rows, "heldout": heldout_rows}
    audit = prevalence(inventories); atomic_json(OUT / "panel_adequacy.json", audit)
    if not audit["passed"]:
        result = {"schema": "factorised_micro_safety_data_scaling_v2_result", "source_commit": "056cc7d4b18384be97d9352eacd3b3409f146df6",
            "classification": "DATA_SCALING_FRESH_PANEL_INADEQUATE", "panel_adequacy": audit, "evaluator_fixture": fixture,
            "training_performed": False, "runtime": {"total_s": time.time() - started, "new_collection_wall_s": sensor_index["parallel_wall_runtime_s"],
            "new_collection_compute_s": sensor_index["runtime_compute_s"], "new_storage_bytes": sensor_index["storage_bytes"]}}
        atomic_json(OUT / "result.json", result); print(json.dumps({"classification": result["classification"]}, indent=2)); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model48, c48, s48, train48 = fit48_bundle(device)
    model96, c96, s96, train96 = train_condition("fit96", fit96, device)
    model192, c192, s192, train192 = train_condition("fit192", fit192, device)
    conditions = {
        "fit48": evaluate_condition("fit48", model48, c48, s48, train48, old48, calibration_rows, heldout_rows, device),
        "fit96": evaluate_condition("fit96", model96, c96, s96, train96, fit96, calibration_rows, heldout_rows, device),
        "fit192": evaluate_condition("fit192", model192, c192, s192, train192, fit192, calibration_rows, heldout_rows, device)}
    curve = learning_curve(conditions); fit192_pass = conditions["fit192"]["prospective_gate"]["passed"]
    v48, v192 = curve["ordered"]["fit48"], curve["ordered"]["fit192"]
    higher = ("contact_auc", "stuck_auc", "safe_retention", "states_retaining_safe", "selected_progress", "best_safe_top3")
    lower = ("false_abstentions", "normalized_regret")
    directional = sum(v192[key] is not None and v48[key] is not None and v192[key] > v48[key] for key in higher)
    directional += sum(v192[key] is not None and v48[key] is not None and v192[key] < v48[key] for key in lower)
    no_new_unsafe = conditions["fit192"]["heldout"]["combined_admissibility_and_planning"]["selected_unsafe_count"] <= conditions["fit48"]["heldout"]["combined_admissibility_and_planning"]["selected_unsafe_count"]
    positive = (not fit192_pass and directional >= 5 and curve["monotonic_count"] >= 6 and
                v192["safe_retention"] - v48["safe_retention"] >= .20 and v192["states_retaining_safe"] - v48["states_retaining_safe"] >= 6 and no_new_unsafe)
    classification = ("FACTORISED_MICRO_SAFETY_DATA_SCALING_SIGNAL" if fit192_pass else
                      "FACTORISED_MICRO_SAFETY_DATA_SCALING_POSITIVE_TENDENCY" if positive else "FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL")
    next_decision = ("CANDIDATE_CONDITIONED_MICRO_SAFETY_PREDICTOR_V1" if fit192_pass else
                     "DECIDE_WHETHER_SCALING_TREND_JUSTIFIES_FURTHER_INVESTMENT" if positive else "CHANGED_GEOMETRY_SENSOR_CONTRACT_OR_NARROWER_SAFETY_CLAIM")
    result = {"schema": "factorised_micro_safety_data_scaling_v2_result", "source_commit": "056cc7d4b18384be97d9352eacd3b3409f146df6",
        "experiment": "FACTORISED_MICRO_SAFETY_DATA_SCALING_V2", "preserved_result": "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_NO_SIGNAL",
        "bindings": {"fit48_checkpoint_sha256": EXPECTED_FIT48_SHA, "old_enhanced_sensor_index_digest": "d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d",
            "old_specialist_ledger_digest": "e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de",
            "old_specialist_ledger_sha256": "a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08",
            "new_panel_manifest_digest": sensor_index["panel_manifest_digest"], "new_sensor_index_digest": sensor_index["content_digest"]},
        "inventories": {"fit48": {"states": 48, "branches": 576}, "fit96": {"states": 96, "branches": 1152},
            "fit192": {"states": 192, "branches": 2304}, "fresh_calibration": {"states": 24, "branches": 288}, "fresh_heldout": {"states": 24, "branches": 288},
            "strictly_nested": True}, "panel_adequacy": audit, "evaluator_fixture": fixture,
        "seed_family": {"value": SEED_FAMILY, "fit48_existing_seed": 2026082010, "fit96_derived": derived_seed("fit96"), "fit192_derived": derived_seed("fit192"), "random_seed_family_count": 1},
        "architecture": {"name": "FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1", "contact_parameters": 97346, "stuck_parameters": 107906,
            "total_parameters": 205252, "shared_parameters": 0, "architecture_and_losses_unchanged": True},
        "conditions": conditions, "learning_curve": curve,
        "classification_evidence": {"fit192_full_gate_passed": fit192_pass, "fit48_to_fit192_directional_improvement_count": directional,
            "broad_monotonicity_passed": curve["monotonic_count"] >= 6, "safe_retention_delta": v192["safe_retention"] - v48["safe_retention"],
            "states_retaining_delta": v192["states_retaining_safe"] - v48["states_retaining_safe"], "no_new_unsafe_selection": no_new_unsafe},
        "classification": classification, "exact_next_decision": next_decision,
        "runtime": {"total_s": time.time() - started, "new_state_generation_wall_s": sensor_index["parallel_wall_runtime_s"],
            "new_state_generation_compute_s": sensor_index["runtime_compute_s"], "new_sensor_storage_bytes": sensor_index["storage_bytes"],
            "fit96_training_s": train96["runtime_s"], "fit192_training_s": train192["runtime_s"],
            "checkpoint_bytes": train96["checkpoint_bytes"] + train192["checkpoint_bytes"],
            "row_ledger_bytes": sum(Path(conditions[name]["row_level_evidence"]["file"]).stat().st_size for name in conditions)},
        "custody": {"one_seed_family_used": True, "new_conditions_trained": ["fit96", "fit192"], "fit48_retrained": False,
            "jepa_predictor_or_rgb_or_depth_lidar_model_trained": False, "memory_or_navigation_system_trained": False,
            "nothing_left_running_at_commit": True}}
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "learning_curve": curve["ordered"],
        "fit96_checkpoint": train96["checkpoint_sha256"], "fit192_checkpoint": train192["checkpoint_sha256"],
        "fit192_gate": conditions["fit192"]["prospective_gate"], "oracle_frontiers": {name: item["oracle_frontier"] for name, item in conditions.items()}}, indent=2, default=BASE.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
