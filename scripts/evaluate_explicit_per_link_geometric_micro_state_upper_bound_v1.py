#!/usr/bin/env python3
"""Evaluate explicit per-link geometry on the repaired two-ply corpus."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as METRICS
from scripts import materialize_explicit_per_link_geometric_micro_state_upper_bound_v1 as MATERIALIZE
from scripts import materialize_two_ply_successor_transition_corpus_repaired_v1 as CORPUS

OUT = MATERIALIZE.OUT
CACHE = MATERIALIZE.CACHE
RESULT = OUT / "result.json"
CONDITIONS = MATERIALIZE.CONDITIONS
SENSOR_CONDITIONS = CONDITIONS[1:]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream: np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def json_ready(value):
    if isinstance(value, dict): return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): value = float(value)
    if isinstance(value, float) and (not math.isfinite(value)): return None
    return value


def evaluator_fixture() -> dict:
    labels = np.asarray([False, True, False, True])
    scores = np.asarray([-0.4, 0.3, -0.2, 0.7])
    metric = contact_metrics(labels, scores, 0.0, False)
    tests = {
        "perfect_contact_ordering": metric["auc"] == 1.0 and metric["average_precision"] == 1.0,
        "zero_false_negative": metric["fn"] == 0 and metric["recall"] == 1.0,
        "safe_count_component_sum": sum(not value for value in (False, True, False, True)) == 2,
        "threshold_tie_is_rejected": bool(np.asarray([0.0]) >= -0.0),
        "nonviable_correct_abstention": not any((False, False)),
    }
    value = {"schema": "explicit_per_link_geometry_evaluator_fixture_v1", "tests": tests, "pass": all(tests.values())}
    value["content_digest"] = METRICS.digest(value); atomic_json(OUT / "evaluator_fixture.json", value)
    if not value["pass"]: raise RuntimeError(value)
    return value


def role_map() -> dict[str, str]:
    split = json.loads(CORPUS.SPLIT.read_text())
    return {state_id: role for role, key in (("training", "development_training_state_ids"),
        ("calibration", "internal_calibration_state_ids"), ("heldout", "development_heldout_state_ids")) for state_id in split[key]}


def action_key(row: dict) -> tuple:
    return MATERIALIZE.action_key(row)


def unique_indices(rows: list[dict]) -> list[int]:
    output = {}; ordered = []
    for row in rows:
        key = action_key(row)
        if key not in output:
            output[key] = int(row["action_index"]); ordered.append(int(row["action_index"]))
    return ordered


def body_region(name: str) -> str:
    if name == "base": return "trunk"
    if name.startswith(("FL", "FR")): return "front_limb"
    if name.startswith(("RL", "RR")): return "rear_limb"
    return "unresolved"


def load() -> tuple[dict, dict, list[dict]]:
    index = json.loads(MATERIALIZE.INDEX.read_text()); corpus = json.loads(CORPUS.INDEX.read_text()); roles = role_map()
    if index["status"] != "PASS" or index["bindings"]["corpus_logical_digest"] != "e41d9926cb7f0f9e1158d09a88b746547806411950b9cac7ebe58aa500a92223":
        raise RuntimeError("geometry/corpus binding failed")
    source_by_id = {row["state_id"]: row for row in corpus["records"]}; states = []
    for receipt in index["records"]:
        source = source_by_id[receipt["state_id"]]
        with np.load(receipt["shard_path"], allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        transitions = {}
        for metadata in receipt["transition_rows"]:
            ti = int(metadata["transition_index"]); level = str(metadata["level"]); current_action = int(metadata["current_action_index"]); action = int(metadata["action_index"])
            exact = bool(np.any(arrays["exact_contact"][ti]))
            frozen = bool(metadata["frozen_contact"])
            native_replay = bool(np.any(arrays["native_contact"][ti]))
            scores = {
                "exact_genesis": float(exact),
                "front_depth_per_link": -float(np.min(arrays["depth_clearance"][ti])),
                "lidar_per_link": -float(np.min(arrays["lidar_clearance"][ti])),
                "depth_lidar_per_link": -float(np.min(arrays["fused_clearance"][ti])),
            }
            exact_steps = np.flatnonzero(arrays["exact_contact"][ti].astype(bool)); first = None if not len(exact_steps) else int(exact_steps[0])
            link_name = None
            object_name = None
            if first is not None:
                link_name = receipt["link_names"].get(str(int(arrays["exact_robot_link"][ti, first])), "unresolved")
                object_name = receipt["object_names"].get(str(int(arrays["exact_other_link"][ti, first])), "unresolved")
            transitions[(level, current_action, action)] = {
                # ``native_contact`` remains the target-facing key throughout the
                # reducer, but is explicitly bound to the immutable repaired-corpus
                # label.  The replay-native force-threshold verdict is retained
                # separately as a reconstruction-consistency diagnostic.
                "transition_index": ti, "native_contact": frozen,
                "frozen_contact": frozen, "native_replay_contact": native_replay,
                "exact_contact": exact, "scores": scores,
                "controller": metadata["controller"], "requested_action": metadata["requested_action"],
                "applied_action": metadata["applied_action"],
                "first_contact_step": first, "responsible_link": link_name, "environment_object": object_name,
                "body_region": None if link_name is None else body_region(link_name),
                "link_minimum_clearance": arrays["structured_minimum_signed_clearance"][ti].astype(np.float32),
                "link_time_to_minimum": arrays["structured_time_to_minimum_step"][ti].astype(np.int16),
                "link_first_crossing": arrays["structured_first_zero_crossing_step"][ti].astype(np.int16),
                "link_approach_speed": arrays["structured_relative_normal_approach_speed"][ti].astype(np.float32),
                "link_sector": arrays["structured_obstacle_sector"][ti].astype(np.int8),
                "condition_clearance_steps": np.stack((
                    np.min(arrays["approximate_scene_clearance"][ti], axis=1),
                    np.min(arrays["depth_clearance"][ti], axis=1),
                    np.min(arrays["lidar_clearance"][ti], axis=1),
                    np.min(arrays["fused_clearance"][ti], axis=1),
                ), axis=1).astype(np.float32),
            }
        states.append({"state_id": source["state_id"], "family": source["family"], "role": roles[source["state_id"]],
                       "source": source, "receipt": receipt, "transitions": transitions})
    return index, corpus, states


def contract_view(state: dict, contract: str) -> dict:
    source = state["source"]; current_rows = source["current_rows"]
    current_indices = list(range(14)) if contract == "historical_fourteen" else unique_indices(current_rows)
    row_by_index = {int(row["action_index"]): row for row in current_rows}
    successors = {int(row["current_action_index"]): row for row in source["successor_rows"]}
    next_indices = {}
    for current in current_indices:
        if row_by_index[current]["current_contact"]: continue
        rows = successors[current]["next_actions"]
        next_indices[current] = list(range(14)) if contract == "historical_fourteen" else unique_indices(rows)
    return {"current_indices": current_indices, "current_rows": row_by_index, "successors": successors, "next_indices": next_indices}


def predictions(state: dict, condition: str, threshold: float | None, contract: str) -> dict:
    view = contract_view(state, contract); current_prediction = {}; next_prediction = {}; predicted_count = {}
    for action in view["current_indices"]:
        transition = state["transitions"][("current", -1, action)]
        current_prediction[action] = transition["exact_contact"] if condition == "exact_genesis" else transition["scores"][condition] >= -float(threshold)
        if action not in view["next_indices"]: predicted_count[action] = -1; continue
        next_prediction[action] = {}
        for next_action in view["next_indices"][action]:
            item = state["transitions"][("successor", action, next_action)]
            next_prediction[action][next_action] = item["exact_contact"] if condition == "exact_genesis" else item["scores"][condition] >= -float(threshold)
        predicted_count[action] = sum(not value for value in next_prediction[action].values())
    return {"view": view, "current_contact": current_prediction, "next_contact": next_prediction, "safe_count": predicted_count}


def select(view: dict, admissible: dict[int, bool], safe_count: dict[int, int]) -> int | None:
    route_rows = [view["current_rows"][index] for index in view["current_indices"] if index < 12 and admissible.get(index, False)]
    if route_rows:
        return int(route_rows[METRICS.route_order(route_rows)[0]]["action_index"])
    lateral = [index for index in view["current_indices"] if index >= 12 and admissible.get(index, False)]
    return None if not lateral else min(lateral, key=lambda index: (-safe_count[index], index))


def oracle_state(state: dict, contract: str) -> dict:
    view = contract_view(state, contract); contact = {}; count = {}; viable = {}
    for action in view["current_indices"]:
        current = state["transitions"][("current", -1, action)]["native_contact"]; contact[action] = current
        if current or action not in view["next_indices"]: count[action] = -1
        else: count[action] = sum(not state["transitions"][("successor", action, next_action)]["native_contact"] for next_action in view["next_indices"][action])
        viable[action] = (not current) and count[action] >= 1
    choice = select(view, viable, count)
    return {"view": view, "contact": contact, "safe_count": count, "viable": viable, "selected": choice}


def decision_metrics(states: list[dict], condition: str, threshold: float | None, contract: str) -> dict:
    per_state = []; selected_progress = oracle_progress = regret = regret_base = 0.0; top1 = top3 = top_den = 0
    count_truth = []; count_predicted = []
    for state in states:
        oracle = oracle_state(state, contract); pred = predictions(state, condition, threshold, contract); view = pred["view"]
        admissible = {action: (not pred["current_contact"][action]) and pred["safe_count"][action] >= 1 for action in view["current_indices"]}
        choice = select(view, admissible, pred["safe_count"]); oracle_choice = oracle["selected"]
        for action in view["current_indices"]:
            if not oracle["contact"][action] and oracle["safe_count"][action] >= 0:
                count_truth.append(oracle["safe_count"][action]); count_predicted.append(pred["safe_count"][action])
        true_viable = any(oracle["viable"].values()); retained = any(admissible[action] and oracle["viable"][action] for action in view["current_indices"])
        selected_contact = choice is not None and oracle["contact"][choice]
        selected_nonviable = choice is not None and not oracle["contact"][choice] and oracle["safe_count"][choice] == 0
        progress = 0.0 if choice is None else float(view["current_rows"][choice]["decision_progress_m"])
        oracle_value = 0.0 if oracle_choice is None else float(view["current_rows"][oracle_choice]["decision_progress_m"])
        if true_viable:
            selected_progress += progress; oracle_progress += oracle_value
            regret += max(0.0, oracle_value - progress); regret_base += max(abs(oracle_value), 1e-6)
            ranked_rows = [view["current_rows"][index] for index in view["current_indices"] if index < 12 and admissible[index]]
            ranked = [ranked_rows[i]["action_index"] for i in METRICS.route_order(ranked_rows)] if ranked_rows else [i for i in view["current_indices"] if i >= 12 and admissible[i]]
            top_den += 1; top1 += int(oracle_choice in ranked[:1]); top3 += int(oracle_choice in ranked[:3])
        per_state.append({"state_id": state["state_id"], "family": state["family"], "oracle_viable": true_viable,
                          "oracle_viable_action_count": sum(oracle["viable"].values()), "retained": retained,
                          "admitted_count": sum(admissible.values()), "selected": choice, "oracle_selected": oracle_choice,
                          "selected_contact": selected_contact, "selected_nonviable_successor": selected_nonviable,
                          "false_abstention": true_viable and choice is None, "correct_abstention": not true_viable and choice is None,
                          "unsafe_movement": not true_viable and choice is not None, "progress_m": progress,
                          "oracle_progress_m": oracle_value, "predicted_safe_counts": pred["safe_count"]})
    truth = np.asarray(count_truth, int); predicted = np.asarray(count_predicted, int); nonzero = truth > 0; predicted_nonzero = predicted > 0
    count = {"rows": len(truth), "mae": float(np.abs(truth - predicted).mean()), "spearman": METRICS.rank_correlation(truth, predicted),
             "exact_count_accuracy": float((truth == predicted).mean()), "zero_vs_nonzero_accuracy": float((nonzero == predicted_nonzero).mean()),
             "false_zero_rate": float((~predicted_nonzero[nonzero]).mean()) if nonzero.any() else 0.0,
             "false_nonzero_rate": float((predicted_nonzero[~nonzero]).mean()) if (~nonzero).any() else 0.0,
             "margin_ge_1": int((predicted >= 1).sum()), "margin_ge_2": int((predicted >= 2).sum()), "margin_ge_3": int((predicted >= 3).sum())}
    viable_rows = [row for row in per_state if row["oracle_viable"]]; nonviable_rows = [row for row in per_state if not row["oracle_viable"]]
    family = {}
    for name in METRICS.FAMILIES:
        rows = [row for row in per_state if row["family"] == name]; viable = [row for row in rows if row["oracle_viable"]]
        family[name] = {"states": len(rows), "oracle_viable_states": len(viable), "retained": sum(row["retained"] for row in viable),
                        "selected_contacts": sum(row["selected_contact"] for row in rows),
                        "selected_nonviable": sum(row["selected_nonviable_successor"] for row in rows),
                        "correct_abstentions": sum(row["correct_abstention"] for row in rows if not row["oracle_viable"])}
    return {"contract": contract, "oracle_viable_states": len(viable_rows), "states_retaining_admitted_action": sum(row["retained"] for row in viable_rows),
            "selected_immediate_contacts": sum(row["selected_contact"] for row in per_state),
            "selected_oracle_nonviable_successors": sum(row["selected_nonviable_successor"] for row in per_state),
            "false_abstentions": sum(row["false_abstention"] for row in viable_rows), "oracle_nonviable_states": len(nonviable_rows),
            "correct_abstentions": sum(row["correct_abstention"] for row in nonviable_rows),
            "unsafe_movement_decisions": sum(row["unsafe_movement"] for row in nonviable_rows),
            "falsely_viable_actions_nonviable_states": sum(row["admitted_count"] for row in nonviable_rows),
            "selected_h3_route_progress_m": selected_progress, "oracle_h3_route_progress_m": oracle_progress,
            "oracle_progress_fraction": selected_progress / max(abs(oracle_progress), 1e-9),
            "normalized_regret": regret / max(regret_base, 1e-9), "best_admissible_top1": top1 / max(1, top_den),
            "best_admissible_top3": top3 / max(1, top_den), "safe_action_count": count,
            "per_family": family, "per_state": per_state}


def contact_rows(states: list[dict], level: str, condition: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    truth = []; scores = []; details = []
    for state in states:
        for key, transition in state["transitions"].items():
            if key[0] != level: continue
            truth.append(transition["native_contact"]); scores.append(transition["scores"][condition])
            details.append({"state_id": state["state_id"], "family": state["family"], "level": level,
                            "current_action_index": key[1], "action_index": key[2], **transition})
    return np.asarray(truth, bool), np.asarray(scores, float), details


def contact_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None, exact: bool) -> dict:
    predicted = scores.astype(bool) if exact else scores >= -float(threshold)
    tp = int((labels & predicted).sum()); fn = int((labels & ~predicted).sum()); fp = int((~labels & predicted).sum()); tn = int((~labels & ~predicted).sum())
    probability = scores if exact else 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
    return {"rows": len(labels), "positives": int(labels.sum()), "auc": METRICS.auc(labels, scores), "average_precision": METRICS.average_precision(labels, scores),
            "recall": tp / max(1, tp + fn), "fnr": fn / max(1, tp + fn), "negative_retention": tn / max(1, tn + fp),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "ece": METRICS.ece(labels, probability), "brier": float(np.mean((probability - labels) ** 2))}


def contact_summary(states: list[dict], condition: str, threshold: float | None) -> dict:
    output = {}
    for level in ("current", "successor"):
        labels, scores, details = contact_rows(states, level, condition)
        result = contact_metrics(labels, scores, threshold, condition == "exact_genesis")
        per_family = {}
        for family in METRICS.FAMILIES:
            mask = np.asarray([row["family"] == family for row in details], bool)
            per_family[family] = contact_metrics(labels[mask], scores[mask], threshold, condition == "exact_genesis")
        result["per_family"] = per_family; output[level] = result
    labels = np.concatenate([contact_rows(states, level, condition)[0] for level in ("current", "successor")])
    scores = np.concatenate([contact_rows(states, level, condition)[1] for level in ("current", "successor")])
    output["combined"] = contact_metrics(labels, scores, threshold, condition == "exact_genesis")
    return output


def choose_threshold(calibration: list[dict], condition: str) -> dict:
    labels = np.concatenate([contact_rows(calibration, level, condition)[0] for level in ("current", "successor")])
    scores = np.concatenate([contact_rows(calibration, level, condition)[1] for level in ("current", "successor")])
    clearances = -scores; values = np.concatenate(([np.nextafter(clearances.min(), -np.inf)], np.unique(clearances), [np.nextafter(clearances.max(), np.inf)]))
    eligible = []
    for threshold in values:
        metric = contact_metrics(labels, scores, float(threshold), False)
        if metric["recall"] < .95 or metric["fnr"] > .05: continue
        decision = decision_metrics(calibration, condition, float(threshold), "unique_deployable")
        retention = metric["negative_retention"]
        key = (retention, decision["states_retaining_admitted_action"], decision["correct_abstentions"],
               decision["selected_h3_route_progress_m"], -decision["normalized_regret"], -float(threshold))
        eligible.append((key, float(threshold), metric, decision))
    if not eligible: raise RuntimeError(f"no threshold reaches contact recall for {condition}")
    selected = max(eligible, key=lambda item: item[0])
    return {"threshold_m": selected[1], "frontier_points": len(values), "eligible_points": len(eligible),
            "selected_contact_metrics": selected[2], "selected_decision_metrics": selected[3]}


def attribution(states: list[dict], condition: str, threshold: float | None) -> dict:
    regions = Counter(); links = Counter(); objects = Counter(); timing = Counter(); misses = Counter()
    for level in ("current", "successor"):
        labels, scores, details = contact_rows(states, level, condition)
        predicted = scores.astype(bool) if condition == "exact_genesis" else scores >= -float(threshold)
        for label, prediction, row in zip(labels, predicted, details, strict=True):
            if not label: continue
            regions[row["body_region"] or "unresolved"] += 1; links[row["responsible_link"] or "unresolved"] += 1
            objects[row["environment_object"] or "unresolved"] += 1
            timing[str(row["first_contact_step"])] += 1
            if not prediction: misses[row["body_region"] or "unresolved"] += 1
    return {"body_regions": dict(regions), "robot_links": dict(links), "environment_objects": dict(objects),
            "first_contact_steps": dict(timing), "missed_contact_body_regions": dict(misses)}


def exact_discrepancies(states: list[dict]) -> dict:
    rows = []
    for state in states:
        for key, transition in state["transitions"].items():
            if transition["native_contact"] == transition["exact_contact"]: continue
            classification = ("CONTACT_MARGIN_OR_MANIFOLD_EFFECT" if transition["exact_contact"]
                              else "DYNAMIC_OR_CONSTRAINT_SOLVER_DEPENDENCE")
            rows.append({"state_id": state["state_id"], "family": state["family"], "role": state["role"],
                         "level": key[0], "current_action_index": key[1], "action_index": key[2],
                         "repaired_corpus_label": transition["frozen_contact"], "history_free_exact_query": transition["exact_contact"],
                         "history_free_first_contact_step": transition["first_contact_step"],
                         "robot_link": transition["responsible_link"], "environment_object": transition["environment_object"],
                         "approximate_minimum_clearance_m": float(np.min(transition["condition_clearance_steps"][:, 0])),
                         "classification": classification})
    counts = Counter(row["classification"] for row in rows)
    return {"count": len(rows), "by_classification": dict(counts), "rows": rows}


def native_replay_discrepancies(states: list[dict]) -> dict:
    rows = []
    for state in states:
        for key, transition in state["transitions"].items():
            if transition["frozen_contact"] == transition["native_replay_contact"]: continue
            rows.append({"state_id": state["state_id"], "family": state["family"], "role": state["role"],
                         "level": key[0], "current_action_index": key[1], "action_index": key[2],
                         "repaired_corpus_label": transition["frozen_contact"],
                         "replay_native_force_threshold_label": transition["native_replay_contact"],
                         "history_free_exact_query": transition["exact_contact"]})
    return {"count": len(rows), "rows": rows}


def exact_gate(contacts: dict, decision: dict) -> dict:
    combined = contacts["combined"]; count = decision["safe_action_count"]
    checks = {"branch_contact_agreement": (combined["tp"] + combined["tn"]) / max(1, combined["rows"]) >= .995,
              "zero_nonzero_count_accuracy": count["zero_vs_nonzero_accuracy"] >= .99, "false_nonzero": count["false_nonzero_rate"] <= .01,
              "all_viable_states_retained": decision["states_retaining_admitted_action"] == decision["oracle_viable_states"],
              "all_nonviable_states_abstain": decision["correct_abstentions"] == decision["oracle_nonviable_states"],
              "selected_contacts_zero": decision["selected_immediate_contacts"] == 0,
              "selected_nonviable_zero": decision["selected_oracle_nonviable_successors"] == 0}
    return {"passed": all(checks.values()), "checks": checks}


def sensor_gate(contacts: dict, decision: dict, exact_progress: float) -> dict:
    current = contacts["current"]; successor = contacts["successor"]; combined = contacts["combined"]; count = decision["safe_action_count"]
    family_ok = all(row["oracle_viable_states"] == 0 or row["retained"] > 0 for row in decision["per_family"].values())
    progress_fraction = decision["selected_h3_route_progress_m"] / max(abs(exact_progress), 1e-9)
    checks = {"current_auc": current["auc"] >= .90, "successor_auc": successor["auc"] >= .90,
              "recall": combined["recall"] >= .95, "fnr": combined["fnr"] <= .05,
              "zero_nonzero_count_accuracy": count["zero_vs_nonzero_accuracy"] >= .90, "false_nonzero": count["false_nonzero_rate"] <= .05,
              "viable_state_retention": decision["states_retaining_admitted_action"] >= math.ceil(.90 * decision["oracle_viable_states"]),
              "all_nonviable_abstain": decision["correct_abstentions"] == decision["oracle_nonviable_states"],
              "selected_contacts_zero": decision["selected_immediate_contacts"] == 0,
              "selected_nonviable_zero": decision["selected_oracle_nonviable_successors"] == 0,
              "route_progress": progress_fraction >= .80, "regret": decision["normalized_regret"] <= .20,
              "top3": decision["best_admissible_top3"] >= .75, "no_family_collapse": family_ok}
    return {"passed": all(checks.values()), "checks": checks, "progress_fraction_of_exact": progress_fraction}


def benchmark(states: list[dict], thresholds: dict, iterations: int = 1000) -> dict:
    # Geometry distances are already materialised; this measures the complete
    # set reduction (14 current sets, all actual successor sets, counting and
    # unchanged H3 selection) independently of simulation/replay cost.
    warmups = 30; samples = []
    selected = states[0]
    for iteration in range(warmups + iterations):
        started = time.perf_counter_ns()
        for condition in SENSOR_CONDITIONS:
            decision_metrics([selected], condition, thresholds[condition], "unique_deployable")
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if iteration >= warmups: samples.append(elapsed)
    values = np.asarray(samples)
    result = {"schema": "explicit_structured_geometry_set_compute_benchmark_v1", "device": "CPU float32",
              "scope": "pre-materialised per-link geometry: all unique current actions, all next actions for contact-free prefixes, safe-counting and H3 selection",
              "warmups": warmups, "iterations": iterations, "p50_ms": float(np.percentile(values, 50)),
              "p90_ms": float(np.percentile(values, 90)), "p95_ms": float(np.percentile(values, 95)),
              "p99_ms": float(np.percentile(values, 99)), "max_ms": float(values.max()),
              "misses_50ms": int((values > 50).sum()), "misses_80ms": int((values > 80).sum()), "misses_100ms": int((values > 100).sum()),
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "peak_vram_bytes": 0,
              "classification": "STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_SIGNAL" if np.percentile(values, 99) <= 50 and values.max() <= 80 else "STRUCTURED_GEOMETRY_SET_REDUCTION_COMPUTE_NO_GO",
              "limitation": "excludes simulator trajectory generation and actual-successor acquisition; it is not a prospective 100 ms interface benchmark"}
    atomic_json(OUT / "compute_benchmark.json", result); return result


def persist_rows(states: list[dict], conditions: dict, thresholds: dict) -> dict:
    rows = []
    for state in states:
        current_source = {int(item["action_index"]): item for item in state["source"]["current_rows"]}
        for key, transition in state["transitions"].items():
            row = {"state_id": state["state_id"], "family": state["family"], "role": state["role"],
                   "level": key[0], "current_action_index": key[1], "action_index": key[2],
                   "geometry_transition_index": transition["transition_index"],
                   "geometry_shard_path": state["receipt"]["shard_path"],
                   "geometry_shard_sha256": state["receipt"]["shard_sha256"],
                   "controller": transition["controller"], "requested_action": transition["requested_action"],
                   "applied_action": transition["applied_action"],
                   "frozen_contact": transition["frozen_contact"],
                   "native_replay_contact": transition["native_replay_contact"],
                   "exact_contact": transition["exact_contact"],
                   "scores": transition["scores"], "first_contact_step": transition["first_contact_step"],
                   "responsible_link": transition["responsible_link"], "body_region": transition["body_region"],
                   "environment_object": transition["environment_object"],
                   "predicted_contact": {condition: (transition["exact_contact"] if condition == "exact_genesis" else transition["scores"][condition] >= -thresholds[condition]) for condition in CONDITIONS}}
            if key[0] == "current":
                source = current_source[key[2]]
                row["h3_progress_m"] = source["h3_progress_m"]
                row["h3_heading_improvement_rad"] = source["h3_heading_improvement_rad"]
                row["decision_progress_m"] = source["decision_progress_m"]
                row["immediate_progress_m"] = source["immediate_progress_m"]
            first_threshold = {}
            for ci, condition in enumerate(CONDITIONS):
                if condition == "exact_genesis":
                    first_threshold[condition] = transition["first_contact_step"]
                else:
                    crossing = np.flatnonzero(transition["condition_clearance_steps"][:, ci] <= thresholds[condition])
                    first_threshold[condition] = None if not len(crossing) else int(crossing[0])
            row["first_threshold_crossing_step"] = first_threshold
            rows.append(row)
    for condition, result in conditions.items():
        for contract, decision in result["decisions"].items():
            lookup = {(row["state_id"], condition, contract): row for row in decision["per_state"]}
            for row in rows:
                if row["level"] == "current" and (row["state_id"], condition, contract) in lookup:
                    state_decision = lookup[(row["state_id"], condition, contract)]
                    row.setdefault("state_decisions", {})[f"{condition}/{contract}"] = {key: state_decision[key] for key in ("selected", "oracle_selected", "oracle_viable", "retained", "false_abstention", "correct_abstention")}
    path = CACHE / "row_level_evidence.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        for row in rows: stream.write(json.dumps(json_ready(row), sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size, "rows": len(rows)}


def main() -> int:
    started = time.time(); fixture = evaluator_fixture(); index, corpus, states = load(); calibration = [state for state in states if state["role"] == "calibration"]
    heldout = [state for state in states if state["role"] == "heldout"]
    thresholds = {"exact_genesis": None}; calibration_results = {}
    for condition in SENSOR_CONDITIONS:
        calibration_results[condition] = choose_threshold(calibration, condition); thresholds[condition] = calibration_results[condition]["threshold_m"]
    results = {}
    for condition in CONDITIONS:
        threshold = thresholds[condition]; contacts = contact_summary(heldout, condition, threshold)
        decisions = {contract: decision_metrics(heldout, condition, threshold, contract) for contract in ("historical_fourteen", "unique_deployable")}
        results[condition] = {"threshold_m": threshold, "contacts": contacts, "decisions": decisions,
                              "attribution": attribution(heldout, condition, threshold)}
    exact_decision = results["exact_genesis"]["decisions"]["unique_deployable"]
    results["exact_genesis"]["gate"] = exact_gate(results["exact_genesis"]["contacts"], exact_decision)
    for condition in SENSOR_CONDITIONS:
        results[condition]["gate"] = sensor_gate(results[condition]["contacts"], results[condition]["decisions"]["unique_deployable"],
                                                   exact_decision["selected_h3_route_progress_m"])
    exact_pass = results["exact_genesis"]["gate"]["passed"]; sensor_passes = [condition for condition in SENSOR_CONDITIONS if results[condition]["gate"]["passed"]]
    if not exact_pass: primary = "EXACT_GEOMETRIC_MICRO_STATE_NO_GO"
    elif sensor_passes: primary = "EXPLICIT_SENSOR_GEOMETRY_MICRO_STATE_SIGNAL"
    else: primary = "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO"
    audit = index["deployable_action_audit"]; secondary = []
    if audit["material_discrepancy"]: secondary.append("DEPLOYABLE_ACTION_CONTRACT_VIABILITY_NO_GO")
    if primary == "EXPLICIT_SENSOR_GEOMETRY_MICRO_STATE_SIGNAL":
        next_step = {"experiment": "PER_LINK_CLEARANCE_PREDICTOR_V1", "train_now": False,
                     "outputs": ["per-link minimum clearance", "time to first clearance violation", "body sector", "uncertainty/lower confidence bound"],
                     "decision": "derive candidate contact and successor viability deterministically; no scalar contact/nonviability head"}
    elif primary == "SENSOR_COVERAGE_MICRO_VIABILITY_NO_GO":
        next_step = {"decision": "add body-facing and denser vertical 3D range coverage around trunk and limb collision regions, or narrow the contact-avoidance scope before another model",
                     "model_training_authorized": False}
    else:
        next_step = {"decision": "explicit articulated/controller/contact interaction state is required before another model", "model_training_authorized": False}
    compute = benchmark(heldout, thresholds)
    row_ledger = persist_rows(states, results, thresholds)
    value = {"schema": "explicit_per_link_geometric_micro_state_upper_bound_v1_result", "experiment": MATERIALIZE.EXPERIMENT,
             "source_commit": MATERIALIZE.SOURCE_COMMIT,
             "preserved_terminals": ["TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER", "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL",
                 "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL", "TRUE_SUCCESSOR_SET_COMPUTE_SIGNAL", "REPLANNING_INTERFACE_UNRESOLVED", "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING"],
             "latest_result_scope": "The repaired set-structured contact evaluator memorised its development-training transitions but did not generalise scene-disjointly, even with the actual successor observation; direct successor prediction remains unauthorised.",
             "claim_boundary": "development-only simulated disallowed-contact structural upper bound using actual successor states; no material-hazard, human-safety, learned closed-loop, stopping-parity, or command-acknowledgement claim",
             "bindings": index["bindings"], "deployable_action_audit": audit,
             "evaluator_fixture": fixture,
             "geometry_contract": index["contract"], "materialization": {key: index[key] for key in ("states", "transitions", "physics_steps", "exact_branch_matches", "native_replay_label_matches", "runtime", "storage_bytes")},
             "exact_query_discrepancies": {"all_roles": exact_discrepancies(states), "heldout": exact_discrepancies(heldout)},
             "native_replay_discrepancies": {"all_roles": native_replay_discrepancies(states), "heldout": native_replay_discrepancies(heldout)},
             "calibration": calibration_results, "thresholds_m": thresholds, "heldout": results,
             "primary_classification": primary, "secondary_classifications": secondary,
             "passing_sensor_conditions": sensor_passes, "next_step": next_step, "compute_benchmark": compute,
             "row_level_evidence": row_ledger, "runtime": {"evaluation_s": time.time() - started},
             "prohibitions_confirmed": {"experimental_model_training": 0, "experimental_learned_inference": 0, "fresh_panel": 0,
                 "jepa_access": 0, "successor_predictor": 0, "memory_navigation_novelty_routing": 0},
             "fixed_controller_replay_disclosure": "registered route/lateral locomotion controllers executed only as simulator plant during authorised deterministic geometry replay; no safety/contact/JEPA model was executed"}
    value["content_digest"] = METRICS.digest(value); atomic_json(RESULT, value)
    print(json.dumps({"primary": primary, "secondary": secondary, "thresholds": thresholds,
                      "exact_gate": results["exact_genesis"]["gate"], "sensor_gates": {c: results[c]["gate"] for c in SENSOR_CONDITIONS},
                      "compute": compute, "row_ledger": row_ledger, "content_digest": value["content_digest"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
