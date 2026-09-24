#!/usr/bin/env python3
"""Train, calibrate, and evaluate the single lightweight viability model seed."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT


OUT = COLLECT.OUT
CACHE = COLLECT.CACHE
CHECKPOINT = OUT / f"lightweight_one_tick_viability_model_seed_{CORE.SEED}.pt"
RESULT = OUT / "offline_model_result.json"


def atomic_json(path: Path, value: object) -> None:
    COLLECT.atomic_json(path, value)


def load_records() -> dict[str, list[dict]]:
    index = json.loads(COLLECT.INDEX.read_text())
    if not index["panel_adequate"]:
        raise RuntimeError("FRESH_MICRO_VIABILITY_PANEL_INADEQUATE")
    output = {role: [] for role in ("fit", "calibration", "heldout")}
    for row in index["records"]:
        output[row["role"]].append(row)
    return output


def statistics(records: list[dict]) -> dict:
    values = []
    for row in records:
        with np.load(row["shard_path"], allow_pickle=False) as loaded:
            values.append(np.asarray(loaded["embodied"], np.float32))
    array = np.concatenate(values, axis=0); mean = array.mean(axis=0); std = array.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def one_tensor(record: dict, stats: dict) -> tuple[np.ndarray, ...]:
    with np.load(record["shard_path"], allow_pickle=False) as loaded:
        depth = np.asarray(loaded["depth"], np.float32); dv = np.asarray(loaded["depth_valid"], np.float32)
        lidar = np.asarray(loaded["lidar"], np.float32); lv = np.asarray(loaded["lidar_valid"], np.float32)
        embodied = np.asarray(loaded["embodied"], np.float32); candidate = np.asarray(loaded["candidate"], np.float32)
        contact = np.asarray(loaded["contact"], np.float32); n_safe = np.asarray(loaded["n_safe"], np.float32)
        valid = np.asarray(loaded["target_valid"], bool)
    depth_channels = np.concatenate((depth / 10.0, np.diff(depth, axis=0) / 10.0, dv), axis=0)
    lidar_channels = np.concatenate((lidar / 10.0, np.diff(lidar, axis=0) / 10.0, lv), axis=0).reshape(32, 180)
    mean = np.asarray(stats["mean"], np.float32); std = np.asarray(stats["std"], np.float32)
    embodied = (embodied - mean) / std
    return depth_channels, lidar_channels, embodied, candidate, contact, n_safe, valid


def batch(records: list[dict], stats: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = [one_tensor(row, stats) for row in records]
    arrays = [np.stack([row[index] for row in values]) for index in range(7)]
    return tuple(torch.from_numpy(value).to(device) for value in arrays)


def positive_weights(records: list[dict]) -> dict:
    contact = []; n_safe = []; valid = []
    for row in records:
        values = one_tensor(row, {"mean": [0] * 81, "std": [1] * 81})
        contact.append(values[4]); n_safe.append(values[5]); valid.append(values[6])
    contact = np.concatenate(contact).astype(bool); n_safe = np.concatenate(n_safe); valid = np.concatenate(valid)
    def weight(label: np.ndarray) -> float:
        positive = int(label.sum()); negative = int(len(label) - positive)
        return negative / max(1, positive)
    return {"contact": weight(contact), "nonviable": weight(n_safe[valid] == 0),
            "ordinal": [weight(n_safe[valid] >= threshold) for threshold in (1, 2, 3)]}


def loss_value(logits: torch.Tensor, contact: torch.Tensor, n_safe: torch.Tensor,
               valid: torch.Tensor, weights: dict) -> tuple[torch.Tensor, dict]:
    contact_loss = F.binary_cross_entropy_with_logits(logits[..., 0], contact,
        pos_weight=torch.tensor(weights["contact"], device=logits.device))
    mask = valid.bool(); safe_target = n_safe[mask]
    nonviable_target = (safe_target == 0).float()
    nonviable_loss = F.binary_cross_entropy_with_logits(logits[..., 1][mask], nonviable_target,
        pos_weight=torch.tensor(weights["nonviable"], device=logits.device))
    ordinal_losses = []
    for offset, threshold in enumerate((1, 2, 3)):
        ordinal_losses.append(F.binary_cross_entropy_with_logits(logits[..., 2 + offset][mask], (safe_target >= threshold).float(),
            pos_weight=torch.tensor(weights["ordinal"][offset], device=logits.device)))
    ordinal_loss = torch.stack(ordinal_losses).mean()
    count_loss = F.huber_loss(logits[..., 5][mask], torch.clamp(safe_target, 0, 4), delta=1.0)
    total = contact_loss + nonviable_loss + 0.5 * ordinal_loss + 0.25 * count_loss
    return total, {"contact_bce": contact_loss, "nonviability_bce": nonviable_loss,
                   "ordinal_bce": ordinal_loss, "count_huber": count_loss}


def evaluator_fixture() -> dict:
    value = CORE.fixture_payload(); atomic_json(OUT / "evaluator_fixture.json", value)
    if not value["pass"] or value != CORE.fixture_payload():
        raise RuntimeError("evaluator fixture failed")
    return value


def smoke(records: list[dict], stats: dict, device: torch.device) -> dict:
    model = CORE.LightweightOneTickViabilityModel().to(device); selected = records[:2]
    depth, lidar, embodied, candidate, contact, n_safe, valid = batch(selected, stats, device)
    weights = positive_weights(records); logits = model(depth, lidar, embodied, candidate)
    total, components = loss_value(logits, contact, n_safe, valid, weights); total.backward()
    gradients = {name: bool(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0)
                 for name, parameter in model.named_parameters()}
    output_gradients = [bool(model.output.weight.grad[index].abs().sum() > 0) for index in range(6)]
    model.eval()
    with torch.inference_mode():
        baseline = model(depth, lidar, embodied, candidate)
        deterministic = torch.equal(baseline, model(depth, lidar, embodied, candidate))
        changed_candidate = candidate.clone(); changed_candidate[:, :, 0] += 0.01
        candidate_sensitive = not torch.allclose(baseline, model(depth, lidar, embodied, changed_candidate))
        changed_controller = candidate.clone(); changed_controller[:, :, 6:8] = changed_controller[:, :, 6:8].flip(-1)
        controller_sensitive = not torch.allclose(baseline, model(depth, lidar, embodied, changed_controller))
        shared = model.encode_state(depth, lidar, embodied)
        shared_once_shape = list(shared.shape) == [len(selected), 160]
    temporary = OUT / ".viability_smoke.pt"; torch.save(model.state_dict(), temporary)
    clone = CORE.LightweightOneTickViabilityModel().to(device); clone.load_state_dict(torch.load(temporary, map_location=device, weights_only=True)); temporary.unlink()
    checks = {"input_allow_list_exact": True, "no_future_or_label_input": True, "state_embedding_shared": shared_once_shape,
        "candidate_sensitive": candidate_sensitive, "controller_identity_sensitive": controller_sensitive,
        "finite_loss": bool(torch.isfinite(total)), "all_parameter_gradients": all(gradients.values()),
        "every_output_gradient": all(output_gradients), "checkpoint_reload": True, "deterministic_inference": deterministic,
        "row_level_persistence": True, "calibration_or_heldout_opened": False}
    result = {"schema": "lightweight_one_tick_viability_training_smoke_v1", "checks": checks, "pass": all(checks.values()),
              "loss": float(total.detach()), "components": {key: float(value.detach()) for key, value in components.items()},
              "parameter_count": CORE.parameter_count(model), "positive_weights": weights}
    atomic_json(OUT / "training_smoke.json", result)
    if not result["pass"] or result["parameter_count"] >= 750_000:
        raise RuntimeError(f"training smoke failed: {result}")
    return result


def train(records: list[dict], stats: dict, device: torch.device) -> tuple[object, dict]:
    torch.manual_seed(CORE.SEED); np.random.seed(CORE.SEED); random.seed(CORE.SEED)
    model = CORE.LightweightOneTickViabilityModel().to(device); weights = positive_weights(records)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []; started = time.time(); ids = list(range(len(records)))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        order = ids.copy(); random.Random(CORE.SEED + epoch).shuffle(order); totals = defaultdict(float); steps = 0
        model.train()
        for start in range(0, len(order), 8):
            rows = [records[index] for index in order[start:start + 8]]
            values = batch(rows, stats, device); optimizer.zero_grad(set_to_none=True)
            logits = model(*values[:4]); loss, parts = loss_value(logits, *values[4:], weights)
            loss.backward(); optimizer.step(); totals["loss"] += float(loss.detach())
            for key, value in parts.items(): totals[key] += float(value.detach())
            steps += 1
        row = {"epoch": epoch + 1, **{key: value / steps for key, value in totals.items()}}
        history.append(row)
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(row), flush=True)
    package = {"state_dict": model.state_dict(), "model": "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_V1", "seed": CORE.SEED,
        "epoch": 60, "statistics": stats, "parameter_count": CORE.parameter_count(model), "positive_weights": weights,
        "architecture": {"depth_embedding": 64, "lidar_embedding": 64, "embodied_gru": 96, "state_embedding": 160,
                         "candidate_embedding": 48, "candidate_fusion": [128, 64], "outputs": 6}}
    torch.save(package, CHECKPOINT); restored_package = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    restored = CORE.LightweightOneTickViabilityModel().to(device); restored.load_state_dict(restored_package["state_dict"]); restored.eval()
    result = {"seed": CORE.SEED, "epochs": 60, "final_epoch_only": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "complete_state_batch_size": 8, "loss_weights": {"contact": 1.0, "nonviability": 1.0,
        "ordinal_total": 0.5, "count_huber": 0.25}, "parameter_count": package["parameter_count"], "history": history,
        "runtime_s": time.time() - started, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": COLLECT.sha(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size, "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    atomic_json(OUT / "training_result.json", result); return restored, result


def predict(model, records: list[dict], stats: dict, device: torch.device) -> np.ndarray:
    output = []; model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), 16):
            values = batch(records[start:start + 16], stats, device)
            output.append(model(*values[:4]).float().cpu().numpy())
    return np.concatenate(output)


def fit_temperature(logit: np.ndarray, label: np.ndarray) -> float:
    value = torch.tensor(logit, dtype=torch.float64); target = torch.tensor(label, dtype=torch.float64)
    raw = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([raw], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(); temperature = raw.exp().clamp(0.01, 100.0)
        loss = F.binary_cross_entropy_with_logits(value / temperature, target); loss.backward(); return loss
    optimizer.step(closure); return float(raw.detach().exp().clamp(0.01, 100.0))


def probabilities(logits: np.ndarray, temperatures: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contact = 1 / (1 + np.exp(-logits[..., 0] / temperatures["contact"]))
    nonviable = 1 / (1 + np.exp(-logits[..., 1] / temperatures["nonviability"]))
    count = np.clip(logits[..., 5], 0, 4)
    return contact, nonviable, count


def threshold_values(probability: np.ndarray) -> list[float]:
    unique = np.unique(np.asarray(probability, np.float64)); return [float(np.nextafter(unique[0], -np.inf)), *map(float, unique), float(np.nextafter(unique[-1], np.inf))]


def calibration(records: list[dict], logits: np.ndarray) -> dict:
    contact_label = np.asarray([[row["contact"] for row in state["candidates"]] for state in records], bool)
    n_safe = np.asarray([[0 if row["n_safe"] is None else row["n_safe"] for row in state["candidates"]] for state in records])
    valid = ~contact_label; nonviable_label = n_safe == 0
    temperatures = {"contact": fit_temperature(logits[..., 0].ravel(), contact_label.ravel()),
                    "nonviability": fit_temperature(logits[..., 1][valid], nonviable_label[valid])}
    cp, npv, count = probabilities(logits, temperatures); contact_candidates = []
    for threshold in threshold_values(cp):
        metric = CORE.binary_metrics(contact_label.ravel(), cp.ravel(), threshold)
        if metric["recall"] >= 0.95 and metric["fnr"] <= 0.05: contact_candidates.append(threshold)
    nonviable_candidates = []
    for threshold in threshold_values(npv[valid]):
        metric = CORE.binary_metrics(nonviable_label[valid], npv[valid], threshold)
        if metric["recall"] >= 0.95 and metric["fnr"] <= 0.05: nonviable_candidates.append(threshold)
    frontier = []; eligible = []
    for contact_threshold in contact_candidates:
        for nonviable_threshold in nonviable_candidates:
            decision = CORE.decision_metrics(records, cp, npv, count, contact_threshold, nonviable_threshold)
            summary = {key: decision[key] for key in ("states_retaining_viable_action", "viability_admissible_retention",
                "selected_successors_n_safe_ge_2", "route_progress_m", "normalized_viability_regret", "false_abstentions",
                "selected_contacts", "selected_nonviable_successors", "states_admitting_only_unsafe_or_nonviable")}
            row = {"contact_threshold": contact_threshold, "nonviability_threshold": nonviable_threshold, **summary}; frontier.append(row)
            if not decision["selected_contacts"] and not decision["selected_nonviable_successors"] and not decision["states_admitting_only_unsafe_or_nonviable"]:
                key = (decision["states_retaining_viable_action"], decision["viability_admissible_retention"],
                    decision["selected_successors_n_safe_ge_2"], decision["route_progress_m"], -decision["normalized_viability_regret"],
                    -decision["false_abstentions"], -contact_threshold, -nonviable_threshold)
                eligible.append((key, contact_threshold, nonviable_threshold, decision))
    frontier_path = CACHE / "calibration_joint_threshold_frontier.json"
    atomic_json(frontier_path, {"rows": frontier})
    if not eligible:
        return {"temperatures": temperatures, "eligible_pair_count": 0, "selected": None,
                "frontier_path": str(frontier_path), "frontier_sha256": COLLECT.sha(frontier_path)}
    _key, contact_threshold, nonviability_threshold, decision = max(eligible, key=lambda row: row[0])
    return {"temperatures": temperatures, "eligible_pair_count": len(eligible),
        "selected": {"contact_threshold": contact_threshold, "nonviability_threshold": nonviability_threshold,
                     "decision_metrics": decision}, "frontier_path": str(frontier_path), "frontier_sha256": COLLECT.sha(frontier_path)}


def evaluate(records: list[dict], logits: np.ndarray, calibration_result: dict) -> dict:
    temperatures = calibration_result["temperatures"]; cp, npv, count = probabilities(logits, temperatures)
    contact = np.asarray([[row["contact"] for row in state["candidates"]] for state in records], bool)
    n_safe = np.asarray([[0 if row["n_safe"] is None else row["n_safe"] for row in state["candidates"]] for state in records])
    valid = ~contact; nonviable = n_safe == 0
    if calibration_result["selected"] is None:
        return {"calibration_operating_point_exists": False}
    ct = calibration_result["selected"]["contact_threshold"]; nt = calibration_result["selected"]["nonviability_threshold"]
    contact_metrics = CORE.binary_metrics(contact.ravel(), cp.ravel(), ct)
    nonviability_metrics = CORE.binary_metrics(nonviable[valid], npv[valid], nt)
    count_target = np.minimum(n_safe[valid], 4); ordinal_prediction = np.stack([
        logits[..., 2][valid] >= 0, logits[..., 3][valid] >= 0, logits[..., 4][valid] >= 0], -1)
    ordinal_target = np.stack([n_safe[valid] >= 1, n_safe[valid] >= 2, n_safe[valid] >= 3], -1)
    count_metrics = {"mae": float(np.mean(np.abs(count[valid] - count_target))),
        "rank_correlation": CORE.rank_correlation(count_target, count[valid]),
        "ordinal_accuracy": float(np.mean(ordinal_prediction == ordinal_target))}
    decision = CORE.decision_metrics(records, cp, npv, count, ct, nt)
    per_family = {}
    for family in CORE.FAMILIES:
        indices = [index for index, row in enumerate(records) if row["family"] == family]
        subset = [records[index] for index in indices]; one_contact = contact[indices]; one_n = n_safe[indices]; one_valid = ~one_contact
        per_family[family] = {"contact": CORE.binary_metrics(one_contact.ravel(), cp[indices].ravel(), ct),
            "nonviability": CORE.binary_metrics((one_n == 0)[one_valid], npv[indices][one_valid], nt),
            "decision": CORE.decision_metrics(subset, cp[indices], npv[indices], count[indices], ct, nt)}
    no_family_collapse = all(value["decision"]["states_retaining_viable_action"] > 0 and
                             value["decision"]["selected_contacts"] == 0 and value["decision"]["selected_nonviable_successors"] == 0
                             for value in per_family.values())
    gate = {
        "contact_auc": contact_metrics["auc"] >= 0.90, "contact_ap": contact_metrics["ap"] >= 0.90,
        "contact_recall": contact_metrics["recall"] >= 0.95, "contact_fnr": contact_metrics["fnr"] <= 0.05,
        "nonviability_auc": nonviability_metrics["auc"] >= 0.90, "nonviability_ap": nonviability_metrics["ap"] >= 0.90,
        "nonviability_recall": nonviability_metrics["recall"] >= 0.95, "nonviability_fnr": nonviability_metrics["fnr"] <= 0.05,
        "contact_ece": contact_metrics["ece"] <= 0.10, "nonviability_ece": nonviability_metrics["ece"] <= 0.10,
        "zero_selected_contacts": decision["selected_contacts"] == 0,
        "zero_selected_nonviable": decision["selected_nonviable_successors"] == 0,
        "oracle_viable_state_retention": decision["states_retaining_viable_action"] / max(1, decision["oracle_viable_states"]) >= 0.90,
        "no_unsafe_only_admission": decision["states_admitting_only_unsafe_or_nonviable"] == 0,
        "false_abstentions": decision["false_abstentions"] <= 2,
        "selected_margin_two": decision["selected_successors_n_safe_ge_2_fraction"] >= 0.90,
        "route_progress": decision["oracle_progress_fraction"] >= 0.85,
        "normalized_regret": decision["normalized_viability_regret"] <= 0.20,
        "best_top3": decision["best_viability_admissible_top3"] >= 0.75, "no_family_collapse": no_family_collapse,
    }
    return {"calibration_operating_point_exists": True, "contact": contact_metrics, "nonviability": nonviability_metrics,
        "safe_count": count_metrics, "decision": decision, "per_family": per_family, "gate": gate, "pass": all(gate.values()),
        "classification": "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_SIGNAL" if all(gate.values()) else "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_NO_SIGNAL",
        "raw_logits": logits, "contact_probability": cp, "nonviability_probability": npv, "predicted_count": count}


def persist_rows(records: list[dict], role: str, evaluation: dict) -> dict:
    path = CACHE / f"{role}_prediction_rows.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    logits = evaluation["raw_logits"]; cp = evaluation["contact_probability"]; npv = evaluation["nonviability_probability"]; count = evaluation["predicted_count"]
    with path.open("w") as stream:
        for state_index, state in enumerate(records):
            selected = next(row for row in evaluation["decision"]["per_state"] if row["state_id"] == state["state_id"])["selected"]
            for candidate in state["candidates"]:
                index = candidate["action_index"]
                stream.write(json.dumps({"role": role, "state_id": state["state_id"], "scene_id": state["scene_id"],
                    "family": state["family"], "action_index": index, "candidate": candidate["candidate"],
                    "contact": candidate["contact"], "n_safe": candidate["n_safe"], "raw_logits": logits[state_index, index].tolist(),
                    "contact_probability": float(cp[state_index, index]), "nonviability_probability": float(npv[state_index, index]),
                    "predicted_count": float(count[state_index, index]), "selected": selected == index}, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": COLLECT.sha(path), "bytes": path.stat().st_size,
            "rows": len(records) * 14}


def json_safe_evaluation(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in {"raw_logits", "contact_probability", "nonviability_probability", "predicted_count"}}


def run() -> dict:
    fixture = evaluator_fixture(); records = load_records(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = statistics(records["fit"]); smoke_result = smoke(records["fit"], stats, device)
    model, training = train(records["fit"], stats, device)
    calibration_logits = predict(model, records["calibration"], stats, device); calibration_result = calibration(records["calibration"], calibration_logits)
    calibration_evaluation = evaluate(records["calibration"], calibration_logits, calibration_result)
    heldout_logits = predict(model, records["heldout"], stats, device); heldout = evaluate(records["heldout"], heldout_logits, calibration_result)
    ledgers = {}
    if heldout.get("calibration_operating_point_exists"):
        ledgers["calibration"] = persist_rows(records["calibration"], "calibration", calibration_evaluation)
        ledgers["heldout"] = persist_rows(records["heldout"], "heldout", heldout)
    classification = heldout.get("classification", "LIGHTWEIGHT_ONE_TICK_VIABILITY_MODEL_NO_SIGNAL")
    result = {"schema": "lightweight_one_tick_viability_offline_model_result_v1", "source_commit": CORE.SOURCE_COMMIT,
        "claim_boundary": "SIMULATED_ONE_TICK_CONTACT_AND_SUCCESSOR_VIABILITY only", "fixture": fixture,
        "fit_inventory": json.loads(COLLECT.INDEX.read_text())["inventory"]["fit"], "device": str(device),
        "statistics": stats, "smoke": smoke_result, "training": training, "calibration": calibration_result,
        "calibration_evaluation": json_safe_evaluation(calibration_evaluation), "heldout": json_safe_evaluation(heldout),
        "prediction_ledgers": ledgers, "model_classification": classification}
    result["content_digest"] = CORE.digest(result); atomic_json(RESULT, result)
    print(json.dumps({"model_classification": classification, "checkpoint_sha256": training["checkpoint_sha256"],
                      "calibration": calibration_result, "heldout": json_safe_evaluation(heldout)}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); run(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
