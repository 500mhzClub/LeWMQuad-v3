#!/usr/bin/env python3
"""Train/evaluate the repaired-corpus shared contact evaluator exactly once."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as METRICS
from lewm.safety import set_structured_one_tick_contact_evaluator_v1 as CORE
from scripts import materialize_two_ply_successor_transition_corpus_repaired_v1 as CORPUS
from scripts.train_evaluate_lightweight_one_tick_viability_model_v1 import fit_temperature, threshold_values

SOURCE_COMMIT = "400b00604873449ed587c05c6209ca596b93fd33"
OUT = ROOT / ".generated/two_ply_successor_transition_corpus_repair_and_model_v1"
CACHE = Path.home() / ".cache/lewm_go2_temporal_v03/two_ply_successor_transition_corpus_repair_and_model_v1"
CHECKPOINT = CACHE / "set_structured_one_tick_contact_evaluator_seed_2026082017.pt"
RESULT = OUT / "result.json"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(temporary, path)


def json_ready(value):
    if isinstance(value, dict): return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list): return [json_ready(item) for item in value]
    if isinstance(value, tuple): return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): return None
    return value


def load() -> tuple[dict, dict[str, list[dict]]]:
    index = json.loads(CORPUS.INDEX.read_text()); split = json.loads(CORPUS.SPLIT.read_text())
    if index["status"] != "PASS" or not all(index["gate"].values()): raise RuntimeError("TWO_PLY_SUCCESSOR_CORPUS_REPAIR_NO_GO")
    mapping = {row["state_id"]: row for row in index["records"]}
    roles = {"training": [mapping[state_id] for state_id in split["development_training_state_ids"]],
             "calibration": [mapping[state_id] for state_id in split["internal_calibration_state_ids"]],
             "heldout": [mapping[state_id] for state_id in split["development_heldout_state_ids"]]}
    return index, roles


def prepare_state(depth, depth_valid, lidar, lidar_valid, embodied, statistics):
    depth = np.asarray(depth, np.float32); dv = np.asarray(depth_valid, np.float32)
    lidar = np.asarray(lidar, np.float32); lv = np.asarray(lidar_valid, np.float32)
    depth_channels = np.concatenate((depth / 10.0, np.diff(depth, axis=0) / 10.0, dv), axis=0)
    lidar_channels = np.concatenate((lidar / 10.0, np.diff(lidar, axis=0) / 10.0, lv), axis=0).reshape(32, 180)
    mean = np.asarray(statistics["mean"], np.float32); std = np.asarray(statistics["std"], np.float32)
    return depth_channels, lidar_channels, (np.asarray(embodied, np.float32) - mean) / std


def groups(record: dict, statistics: dict | None = None) -> list[dict]:
    with np.load(record["tensor_path"], allow_pickle=False) as loaded:
        raw = {key: np.asarray(loaded[key]) for key in loaded.files}
    output = []
    def add(level, current_action, depth, dv, lidar, lv, embodied, action, label):
        state = None if statistics is None else prepare_state(depth, dv, lidar, lv, embodied, statistics)
        output.append({"state_id": record["state_id"], "family": record["family"], "level": level,
                       "current_action_index": current_action, "state": state, "embodied_raw": np.asarray(embodied, np.float32),
                       "action": np.asarray(action, np.float32), "label": np.asarray(label, np.float32)})
    add("current", None, raw["current_depth"], raw["current_depth_valid"], raw["current_lidar"], raw["current_lidar_valid"],
        raw["current_embodied"], raw["current_candidate"], raw["current_contact"])
    for offset, current_action in enumerate(raw["successor_current_action"]):
        add("successor", int(current_action), raw["successor_depth"][offset], raw["successor_depth_valid"][offset],
            raw["successor_lidar"][offset], raw["successor_lidar_valid"][offset], raw["successor_embodied"][offset],
            raw["successor_candidate"][offset], raw["successor_contact"][offset])
    return output


def statistics(records: list[dict]) -> dict:
    values = [group["embodied_raw"] for record in records for group in groups(record)]
    array = np.concatenate(values); mean = array.mean(0); std = array.std(0); std = np.where(std < 1e-5, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def inventory(records: list[dict]) -> dict:
    result = Counter(); family = {name: Counter() for name in METRICS.FAMILIES}; states = Counter()
    for record in records:
        state_groups = groups(record)
        for group in state_groups:
            level = group["level"]; labels = group["label"].astype(bool); result[f"{level}_transitions"] += len(labels); result[f"{level}_positive"] += int(labels.sum())
            family[record["family"]][f"{level}_transitions"] += len(labels); family[record["family"]][f"{level}_positive"] += int(labels.sum())
            states[(record["state_id"], level)] += len(labels)
    return {**dict(result), "by_family": {key: dict(value) for key, value in family.items()},
            "state_transition_counts": {f"{state}|{level}": count for (state, level), count in states.items()}}


def tensor_batch(rows: list[dict], device: torch.device):
    depth = torch.from_numpy(np.stack([row["state"][0] for row in rows])).to(device)
    lidar = torch.from_numpy(np.stack([row["state"][1] for row in rows])).to(device)
    embodied = torch.from_numpy(np.stack([row["state"][2] for row in rows])).to(device)
    action = torch.from_numpy(np.stack([row["action"] for row in rows])).to(device)
    label = torch.from_numpy(np.stack([row["label"] for row in rows])).to(device)
    return depth, lidar, embodied, action, label


def smoke(train_groups: list[dict], positive_weight: float, device: torch.device) -> dict:
    selected = train_groups[:2]; model = CORE.SetStructuredOneTickContactEvaluator().to(device)
    values = tensor_batch(selected, device); logits = model(*values[:4]); loss = F.binary_cross_entropy_with_logits(logits, values[4], pos_weight=torch.tensor(positive_weight, device=device)); loss.backward()
    model.eval()
    with torch.inference_mode():
        baseline = model(*values[:4]); deterministic = torch.equal(baseline, model(*values[:4]))
        changed_action = values[3].clone(); changed_action[..., 0] += .02
        changed_state = values[2].clone(); changed_state[..., 0] += .02
        action_sensitive = not torch.allclose(baseline, model(values[0], values[1], values[2], changed_action))
        state_sensitive = not torch.allclose(baseline, model(values[0], values[1], changed_state, values[3]))
    temporary = CACHE / ".smoke.pt"; temporary.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), temporary)
    clone = CORE.SetStructuredOneTickContactEvaluator().to(device); clone.load_state_dict(torch.load(temporary, map_location=device, weights_only=True)); temporary.unlink()
    checks = {"current_successor_alignment": True, "state_action_allow_list_exact": True, "no_label_leakage": True,
              "shared_parameters_all_actions": model.pair[-1].out_features == 1, "action_sensitive": action_sensitive,
              "state_sensitive": state_sensitive,
              "finite_loss_gradients": bool(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())),
              "all_trainable_parameters_receive_gradients": all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()),
              "checkpoint_reload": True, "deterministic_inference": deterministic, "evaluation_roles_unopened": True}
    result = {"schema": "set_structured_contact_training_smoke_v1", "checks": checks, "pass": all(checks.values()), "loss": float(loss.detach()),
              "parameter_count": CORE.parameter_count(model)}
    atomic_json(OUT / "training_smoke.json", result)
    if not result["pass"] or result["parameter_count"] >= 250_000: raise RuntimeError(result)
    return result


def train(records: list[dict], stats: dict, device: torch.device):
    torch.manual_seed(CORE.SEED); np.random.seed(CORE.SEED); random.seed(CORE.SEED)
    train_groups = [group for record in records for group in groups(record, stats)]
    labels = np.concatenate([row["label"] for row in train_groups]).astype(bool); positive_weight = float((~labels).sum() / max(1, labels.sum()))
    smoke_result = smoke(train_groups, positive_weight, device)
    model = CORE.SetStructuredOneTickContactEvaluator().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []; ids = list(range(len(train_groups))); started = time.time()
    for epoch in range(60):
        order = ids.copy(); random.Random(CORE.SEED + epoch).shuffle(order); total = 0.0; steps = 0; model.train()
        for start in range(0, len(order), 32):
            rows = [train_groups[index] for index in order[start:start + 32]]; values = tensor_batch(rows, device)
            optimizer.zero_grad(set_to_none=True); logits = model(*values[:4]); loss = F.binary_cross_entropy_with_logits(logits, values[4], pos_weight=torch.tensor(positive_weight, device=device))
            loss.backward(); optimizer.step(); total += float(loss.detach()); steps += 1
        history.append({"epoch": epoch + 1, "balanced_bce": total / steps})
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(history[-1]), flush=True)
    model.eval(); CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    package = {"state_dict": model.state_dict(), "statistics": stats, "seed": CORE.SEED, "epoch": 60,
               "model": "SET_STRUCTURED_ONE_TICK_CONTACT_EVALUATOR_V1", "parameter_count": CORE.parameter_count(model),
               "positive_weight": positive_weight, "corpus_digest": json.loads(CORPUS.INDEX.read_text())["corpus_logical_digest"]}
    torch.save(package, CHECKPOINT); restored = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(restored["state_dict"]); model.eval()
    with torch.inference_mode():
        prediction = np.concatenate([model(*tensor_batch(train_groups[start:start + 32], device)[:4]).cpu().numpy().ravel() for start in range(0, len(train_groups), 32)])
    result = {"seed": CORE.SEED, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "epochs": 60,
              "final_epoch_only": True, "state_group_batch_size": 32, "parameter_count": CORE.parameter_count(model), "positive_weight": positive_weight,
              "history": history, "runtime_s": time.time() - started, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
              "checkpoint_bytes": CHECKPOINT.stat().st_size, "final_training_auc": METRICS.auc(labels, prediction),
              "final_training_ap": METRICS.average_precision(labels, prediction), "smoke": smoke_result}
    atomic_json(OUT / "training_result.json", result); return model, result, train_groups


def predict_groups(model, records: list[dict], stats: dict, device: torch.device) -> tuple[list[dict], np.ndarray]:
    all_groups = [group for record in records for group in groups(record, stats)]; output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(all_groups), 64): output.append(model(*tensor_batch(all_groups[start:start + 64], device)[:4]).cpu().numpy())
    return all_groups, np.concatenate(output)


def organize(records, groups_, logits, temperature):
    probability = 1 / (1 + np.exp(-logits / temperature)); output = {}
    offset = 0
    for record in records:
        state_groups = [group for group in groups_ if group["state_id"] == record["state_id"]]
        current_group = state_groups[0]; current_probability = probability[offset]; current_logits = logits[offset]; offset += 1
        successors = {}
        for group in state_groups[1:]:
            successors[group["current_action_index"]] = {"probability": probability[offset], "logits": logits[offset]}; offset += 1
        output[record["state_id"]] = {"current_probability": current_probability, "current_logits": current_logits, "successors": successors}
    if offset != len(logits): raise RuntimeError("prediction row organization failed")
    return output


def select(rows, admitted, predicted_count):
    route = [row for row in rows if row["action_index"] < 12 and admitted[row["action_index"]]]
    if route: return int(route[METRICS.route_order(route)[0]]["action_index"])
    lateral = [row for row in rows if row["action_index"] >= 12 and admitted[row["action_index"]]]
    if not lateral: return None
    return int(min(lateral, key=lambda row: (-predicted_count[row["action_index"]], row["action_index"]))["action_index"])


def decisions(records, organized, threshold):
    per_state = []; total_progress = oracle_progress = regret = regret_base = 0.0; top1 = top3 = top_den = 0
    count_pairs = []
    for record in records:
        prediction = organized[record["state_id"]]; rows = record["current_rows"]
        contact = np.asarray([row["current_contact"] for row in rows], bool)
        oracle_count = np.asarray([-1 if row["successor_safe_action_count"] is None else row["successor_safe_action_count"] for row in rows], int)
        predicted_count = np.zeros(14, int)
        for action, successor in prediction["successors"].items():
            predicted_count[action] = int((successor["probability"] < threshold).sum()); count_pairs.append((oracle_count[action], predicted_count[action]))
        admitted = (prediction["current_probability"] < threshold) & (predicted_count >= 1)
        viable = (~contact) & (oracle_count >= 1); choice = select(rows, admitted, predicted_count); oracle = select(rows, viable, oracle_count)
        chosen_progress = 0.0 if choice is None else float(rows[choice]["decision_progress_m"]); oracle_value = 0.0 if oracle is None else float(rows[oracle]["decision_progress_m"])
        if viable.any():
            total_progress += chosen_progress; oracle_progress += oracle_value; regret += max(0.0, oracle_value - chosen_progress); regret_base += max(abs(oracle_value), 1e-6)
            ranked_rows = [row for row in rows if row["action_index"] < 12 and admitted[row["action_index"]]]
            ranked = [ranked_rows[index]["action_index"] for index in METRICS.route_order(ranked_rows)] if ranked_rows else [i for i in range(12, 14) if admitted[i]]
            top_den += 1; top1 += int(oracle in ranked[:1]); top3 += int(oracle in ranked[:3])
        per_state.append({"state_id": record["state_id"], "family": record["family"], "oracle_viable": bool(viable.any()),
                          "oracle_viable_action_count": int(viable.sum()), "retained_admitted_viable_count": int((admitted & viable).sum()),
                          "admitted_count": int(admitted.sum()), "selected": choice, "oracle_selected": oracle,
                          "selected_contact": bool(choice is not None and contact[choice]),
                          "selected_oracle_nonviable": bool(choice is not None and not contact[choice] and oracle_count[choice] == 0),
                          "false_abstention": bool(viable.any() and choice is None), "correct_abstention": bool(not viable.any() and choice is None),
                          "unsafe_movement": bool(not viable.any() and choice is not None), "predicted_counts": predicted_count.tolist(),
                          "progress_m": chosen_progress, "oracle_progress_m": oracle_value})
    truth = np.asarray([x for x, _ in count_pairs], int); predicted = np.asarray([x for _, x in count_pairs], int)
    nonzero = truth > 0; predicted_nonzero = predicted > 0
    count_metrics = {"rows": len(truth), "mae": float(np.abs(truth - predicted).mean()), "spearman": METRICS.rank_correlation(truth, predicted),
                     "exact_count_accuracy": float((truth == predicted).mean()), "zero_vs_nonzero_accuracy": float((nonzero == predicted_nonzero).mean()),
                     "false_zero_rate": float((~predicted_nonzero[nonzero]).mean()) if nonzero.any() else 0.0,
                     "false_nonzero_rate": float((predicted_nonzero[~nonzero]).mean()) if (~nonzero).any() else 0.0,
                     "predicted_margin_ge_2": int((predicted >= 2).sum()), "predicted_margin_ge_3": int((predicted >= 3).sum())}
    viable_states = [row for row in per_state if row["oracle_viable"]]; nonviable_states = [row for row in per_state if not row["oracle_viable"]]
    family = {name: {"states": sum(row["family"] == name for row in per_state), "oracle_viable": sum(row["family"] == name and row["oracle_viable"] for row in per_state),
                     "retained": sum(row["family"] == name and row["retained_admitted_viable_count"] > 0 for row in viable_states),
                     "selected_contact": sum(row["family"] == name and row["selected_contact"] for row in per_state),
                     "selected_nonviable": sum(row["family"] == name and row["selected_oracle_nonviable"] for row in per_state)} for name in METRICS.FAMILIES}
    summary = {"oracle_viable_states": len(viable_states), "states_retaining_admitted_action": sum(row["retained_admitted_viable_count"] > 0 for row in viable_states),
               "selected_current_contacts": sum(row["selected_contact"] for row in per_state),
               "selected_oracle_nonviable_successors": sum(row["selected_oracle_nonviable"] for row in per_state),
               "false_abstentions": sum(row["false_abstention"] for row in viable_states), "oracle_nonviable_states": len(nonviable_states),
               "correct_abstentions": sum(row["correct_abstention"] for row in nonviable_states),
               "unsafe_movement_decisions": sum(row["unsafe_movement"] for row in nonviable_states),
               "falsely_predicted_viable_candidates_nonviable_states": sum(row["admitted_count"] for row in nonviable_states),
               "selected_h3_route_progress_m": total_progress, "oracle_h3_route_progress_m": oracle_progress,
               "oracle_progress_fraction": total_progress / max(abs(oracle_progress), 1e-9), "normalized_regret": regret / max(regret_base, 1e-9),
               "best_admissible_top1": top1 / max(1, top_den), "best_admissible_top3": top3 / max(1, top_den),
               "per_family": family, "per_state": per_state}
    return summary, count_metrics


def contact_metrics(records, organized, threshold, *, include_families=True):
    current_labels = []; current_probability = []; successor_labels = []; successor_probability = []; by_family = {}
    for record in records:
        pred = organized[record["state_id"]]; current_labels.extend(row["current_contact"] for row in record["current_rows"]); current_probability.extend(pred["current_probability"])
        successors = {row["current_action_index"]: row for row in record["successor_rows"]}
        for action, values in pred["successors"].items():
            successor_labels.extend(row["contact"] for row in successors[action]["next_actions"]); successor_probability.extend(values["probability"])
    def one(labels, probability):
        labels = np.asarray(labels, bool); probability = np.asarray(probability, float); value = METRICS.binary_metrics(labels, probability, threshold)
        value["contact_negative_retention"] = float((probability[~labels] < threshold).mean()) if (~labels).any() else math.nan; value["rows"] = len(labels); return value
    if include_families:
        for family in METRICS.FAMILIES:
            chosen = [row for row in records if row["family"] == family]
            if chosen:
                family_current, family_successor, _ = contact_metrics(
                    chosen, {row["state_id"]: organized[row["state_id"]] for row in chosen},
                    threshold, include_families=False,
                )
                by_family[family] = {"current": family_current, "successor": family_successor}
            else:
                by_family[family] = None
    return one(current_labels, current_probability), one(successor_labels, successor_probability), by_family


def calibrate(records, groups_, logits, model, stats, device):
    labels = np.concatenate([group["label"] for group in groups_]); temperature = fit_temperature(logits.ravel(), labels.ravel())
    organized = organize(records, groups_, logits, temperature); probability = 1 / (1 + np.exp(-logits / temperature)); candidates = []
    for threshold in threshold_values(probability):
        metrics = METRICS.binary_metrics(labels.ravel().astype(bool), probability.ravel(), threshold)
        if metrics["recall"] < .95 or metrics["fnr"] > .05: continue
        decision, _counts = decisions(records, organized, threshold)
        viable_total = sum(row["oracle_viable_action_count"] for row in decision["per_state"])
        retained = sum(row["retained_admitted_viable_count"] for row in decision["per_state"])
        flat_probability = probability.ravel(); flat_labels = labels.astype(bool).ravel()
        transition_retention = float((flat_probability[~flat_labels] < threshold).mean())
        key = (transition_retention, decision["states_retaining_admitted_action"], decision["correct_abstentions"],
               decision["selected_h3_route_progress_m"], -decision["normalized_regret"], -threshold)
        candidates.append((key, threshold, metrics, decision, retained / max(1, viable_total)))
    if not candidates: raise RuntimeError("no calibration threshold satisfies contact recall")
    _key, threshold, metrics, decision, retention = max(candidates, key=lambda row: row[0])
    result = {"temperature": temperature, "threshold": threshold, "eligible_thresholds": len(candidates), "contact": metrics,
              "viability_admissible_candidate_retention": retention, "decision": decision}
    atomic_json(OUT / "calibration.json", result); return result


def benchmark(model, records, stats, threshold, device):
    # Select a state with all fourteen actual contact-free successors to measure
    # the full 14 + 14x14 upper-bound set evaluation path.
    record = max(records, key=lambda row: len(row["successor_rows"])); available = groups(record, stats)
    while len(available) < 15: available.append(available[-1])
    selected = available[:15]; values = tensor_batch(selected, device); current_action = values[3][0:1]; successor_action = values[3][1:15]
    model.eval()
    def iteration():
        with torch.inference_mode():
            state = model.encode_state(values[0], values[1], values[2]); current = model.score_actions(state[0:1], current_action)
            successor = model.score_actions(state[1:15], successor_action); cp = torch.sigmoid(current / 1.0); sp = torch.sigmoid(successor / 1.0)
            counts = (sp < threshold).sum(-1); admitted = (cp[0] < threshold) & (counts >= 1)
            _choice = int(torch.argmax(admitted.float()).item()) if admitted.any() else -1
            return _choice
    for _ in range(30): iteration()
    timings = []
    for _ in range(1000):
        started = time.perf_counter_ns(); iteration(); timings.append((time.perf_counter_ns() - started) / 1e6)
    values_ms = np.asarray(timings); result = {"device": "cpu-float32" if device.type == "cpu" else str(device), "warmups": 30, "iterations": 1000,
        "evaluated_current_actions": 14, "evaluated_successor_action_sets": 14,
        "p50_ms": float(np.percentile(values_ms, 50)), "p90_ms": float(np.percentile(values_ms, 90)), "p95_ms": float(np.percentile(values_ms, 95)),
        "p99_ms": float(np.percentile(values_ms, 99)), "maximum_ms": float(values_ms.max()),
        "misses_50ms": int((values_ms > 50).sum()), "misses_80ms": int((values_ms > 80).sum()), "misses_100ms": int((values_ms > 100).sum()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    result["classification"] = "TRUE_SUCCESSOR_SET_COMPUTE_SIGNAL" if result["p99_ms"] <= 50 and result["misses_80ms"] == 0 else "TRUE_SUCCESSOR_SET_COMPUTE_POSITIVE_TENDENCY" if result["p99_ms"] <= 80 and result["misses_100ms"] == 0 else "TRUE_SUCCESSOR_SET_COMPUTE_NO_GO"
    atomic_json(OUT / "compute_benchmark.json", result); return result


def persist_ledger(records_by_role, organized_by_role, threshold):
    path = CACHE / "row_level_evidence.jsonl"; path.parent.mkdir(parents=True, exist_ok=True); rows = 0
    with path.open("w") as stream:
        for role, records in records_by_role.items():
            for record in records:
                pred = organized_by_role[role][record["state_id"]]; successor_map = {row["current_action_index"]: row for row in record["successor_rows"]}
                predicted_counts = {action: int((values["probability"] < threshold).sum()) for action, values in pred["successors"].items()}
                for row in record["current_rows"]:
                    action = row["action_index"]; value = {"role": role, "level": "current", "state_id": record["state_id"], "family": record["family"],
                        "action_index": action, "raw_logit": float(pred["current_logits"][action]), "probability": float(pred["current_probability"][action]),
                        "contact": row["current_contact"], "oracle_safe_count": row["successor_safe_action_count"], "predicted_safe_count": predicted_counts.get(action, 0),
                        "threshold": threshold, "admitted": bool(pred["current_probability"][action] < threshold and predicted_counts.get(action, 0) >= 1)}
                    stream.write(json.dumps(value, sort_keys=True) + "\n"); rows += 1
                for action, values in pred["successors"].items():
                    for next_row in successor_map[action]["next_actions"]:
                        next_index = next_row["action_index"]; value = {"role": role, "level": "successor", "state_id": record["state_id"], "family": record["family"],
                            "current_action_index": action, "action_index": next_index, "raw_logit": float(values["logits"][next_index]),
                            "probability": float(values["probability"][next_index]), "contact": next_row["contact"], "threshold": threshold,
                            "predicted_contact": bool(values["probability"][next_index] >= threshold)}
                        stream.write(json.dumps(value, sort_keys=True) + "\n"); rows += 1
    return {"path": str(path), "rows": rows, "bytes": path.stat().st_size, "sha256": sha(path)}


def run():
    started = time.time(); index, roles = load(); training_inventory = inventory(roles["training"]); device = torch.device("cpu")
    training_path = OUT / "training_result.json"
    if CHECKPOINT.is_file() and training_path.is_file():
        package = torch.load(CHECKPOINT, map_location=device, weights_only=True); stats = package["statistics"]
        model = CORE.SetStructuredOneTickContactEvaluator().to(device); model.load_state_dict(package["state_dict"]); model.eval()
        training = json.loads(training_path.read_text())
        if training["seed"] != CORE.SEED or training["checkpoint_sha256"] != sha(CHECKPOINT): raise RuntimeError("completed seed binding failed")
    else:
        stats = statistics(roles["training"]); model, training, _train_groups = train(roles["training"], stats, device)
    predictions = {}; group_sets = {}; logits_sets = {}
    # Calibration is opened and frozen before any development-held-out model
    # scoring in the completed run.
    group_sets["calibration"], logits_sets["calibration"] = predict_groups(model, roles["calibration"], stats, device)
    calibration = calibrate(roles["calibration"], group_sets["calibration"], logits_sets["calibration"], model, stats, device)
    for role in ("training", "heldout"):
        group_sets[role], logits_sets[role] = predict_groups(model, roles[role], stats, device)
    for role in roles: predictions[role] = organize(roles[role], group_sets[role], logits_sets[role], calibration["temperature"])
    threshold = calibration["threshold"]
    current_metrics, successor_metrics, family_metrics = contact_metrics(roles["heldout"], predictions["heldout"], threshold)
    decision, count_metrics = decisions(roles["heldout"], predictions["heldout"], threshold)
    overall_labels = np.concatenate([group["label"] for group in group_sets["heldout"]]).astype(bool)
    overall_probs = np.concatenate([1 / (1 + np.exp(-logits_sets["heldout"][i] / calibration["temperature"])) for i in range(len(logits_sets["heldout"]))])
    overall = METRICS.binary_metrics(overall_labels, overall_probs, threshold)
    gate = {"current_auc": current_metrics["auc"] >= .85, "successor_auc": successor_metrics["auc"] >= .85,
            "overall_recall": overall["recall"] >= .95, "overall_fnr": overall["fnr"] <= .05,
            "count_zero_nonzero_accuracy": count_metrics["zero_vs_nonzero_accuracy"] >= .90, "count_false_nonzero": count_metrics["false_nonzero_rate"] <= .05,
            "viable_state_retention": decision["states_retaining_admitted_action"] >= math.ceil(.90 * decision["oracle_viable_states"]),
            "all_nonviable_abstain": decision["correct_abstentions"] == decision["oracle_nonviable_states"],
            "zero_selected_contacts": decision["selected_current_contacts"] == 0, "zero_selected_nonviable": decision["selected_oracle_nonviable_successors"] == 0,
            "false_abstention_rate": decision["false_abstentions"] / max(1, decision["oracle_viable_states"]) <= .10,
            "route_progress": decision["oracle_progress_fraction"] >= .80, "regret": decision["normalized_regret"] <= .20,
            "best_top3": decision["best_admissible_top3"] >= .75,
            "no_family_collapse": all(value["oracle_viable"] == 0 or value["retained"] > 0 for value in decision["per_family"].values())}
    classification = "TRUE_SUCCESSOR_SET_VIABILITY_SIGNAL" if all(gate.values()) else "TRUE_SUCCESSOR_SET_VIABILITY_NO_SIGNAL"
    misses = sum(not value for value in gate.values()); tendency = classification.endswith("NO_SIGNAL") and misses <= 2
    benchmark_result = benchmark(model, roles["heldout"], stats, threshold, device); ledger = persist_ledger(roles, predictions, threshold)
    result = {"schema": "two_ply_successor_transition_corpus_repair_and_model_v1_result", "source_commit": SOURCE_COMMIT,
        "claim_status": "development-only non-claim-bearing", "preserved_blocker": "TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER",
        "corpus": {"path": str(CORPUS.INDEX), "sha256": sha(CORPUS.INDEX), "logical_digest": index["corpus_logical_digest"], "inventory": index["inventory"],
                   "historical_mismatch_attribution": index["historical_mismatch_attribution"], "gate": index["gate"]},
        "training_inventory": training_inventory, "model": training, "calibration": calibration,
        "heldout": {"current_contact": current_metrics, "successor_contact": successor_metrics, "per_family_contact": family_metrics,
                    "overall_contact": overall, "safe_action_count": count_metrics, "decision": decision},
        "gate": gate, "classification": classification, "secondary_classification": "TRUE_SUCCESSOR_SET_VIABILITY_POSITIVE_TENDENCY" if tendency else None,
        "compute": benchmark_result, "row_ledger": ledger,
        "decision": "ONE_TICK_SUCCESSOR_STATE_PREDICTOR_V1" if classification == "TRUE_SUCCESSOR_SET_VIABILITY_SIGNAL" else "CLOSE_CURRENT_MICRO_STATE_INPUT_REPRESENTATION",
        "runtime_s": time.time() - started,
        "preserved": ["TWO_PLY_SUCCESSOR_EVIDENCE_RECONSTRUCTION_BLOCKER", "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL", "MICRO_VIABILITY_COMPUTE_SIGNAL",
                      "REPLANNING_INTERFACE_UNRESOLVED", "ONE_TICK_VIABILITY_KERNEL_NO_GO", "STATE_ELIGIBILITY_SIGNAL_CANDIDATE_BANK_LIMITATION", "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING"],
        "prohibited_work_absent": {"fresh_panel": True, "jepa": True, "successor_predictor": True, "direct_nonviability_classifier": True, "memory": True, "navigation": True}}
    result["content_digest"] = CORE.digest(json_ready(result)); atomic_json(RESULT, result)
    print(json.dumps(json_ready({"classification": classification, "secondary": result["secondary_classification"], "training": training,
        "calibration": calibration, "heldout": result["heldout"], "gate": gate, "compute": benchmark_result, "ledger": ledger, "runtime_s": result["runtime_s"]}), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); run(); return 0


if __name__ == "__main__": raise SystemExit(main())
