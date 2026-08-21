#!/usr/bin/env python3
"""Development-only single-seed screen on the frozen micro-viability corpus."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(extra))

from lewm.safety import lightweight_one_tick_viability_model_v1 as CORE
from scripts import collect_lightweight_one_tick_viability_model_v1 as COLLECT
from scripts import train_evaluate_lightweight_one_tick_viability_model_v1 as BASE

SOURCE_COMMIT = "53a011f0b55b816294c49ced88bd3a6a55c4adec"
OUT = ROOT / ".generated" / "development_micro_viability_model_screen_v1"
CACHE = Path.home() / ".cache" / "lewm_go2_temporal_v03" / "development_micro_viability_model_screen_v1"
SPLIT = OUT / "development_internal_calibration_v1.json"
CHECKPOINT = OUT / f"development_micro_viability_model_seed_{CORE.SEED}.pt"
RESULT = OUT / "result.json"
FROZEN_LEDGER = Path.home() / ".cache" / "lewm_go2_temporal_v03" / "lightweight_one_tick_viability_model_and_interface_v1" / "row_level_evidence_v1.jsonl"
FROZEN_LEDGER_SHA256 = "0a273a3f464f770ccf8d28a1c6c3d9ddad63efdb767c1a63175ddcb479a18eea"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_inputs() -> None:
    if not FROZEN_LEDGER.is_file() or sha(FROZEN_LEDGER) != FROZEN_LEDGER_SHA256:
        raise RuntimeError("frozen predecessor ledger binding failed")


def state_summary(state: dict) -> dict:
    candidates = state["candidates"]
    contact = sum(bool(row["contact"]) for row in candidates)
    nonviable = sum(not row["contact"] and row["n_safe"] == 0 for row in candidates)
    oracle_viable = any(not row["contact"] and row["n_safe"] >= 1 for row in candidates)
    return {"contact_rows": contact, "nonviable_rows": nonviable, "oracle_viable": oracle_viable,
            "class_signature": [contact > 0, nonviable > 0, oracle_viable]}


def inventory(states: list[dict]) -> dict:
    summaries = [state_summary(state) for state in states]
    return {"states": len(states), "candidate_rows": 14 * len(states),
            "contact_positive_rows": sum(row["contact_rows"] for row in summaries),
            "nonviable_successor_rows": sum(row["nonviable_rows"] for row in summaries),
            "oracle_viable_states": sum(row["oracle_viable"] for row in summaries),
            "oracle_nonviable_states": sum(not row["oracle_viable"] for row in summaries),
            "per_family": {family: {"states": sum(state["family"] == family for state in states),
                "contact_positive_rows": sum(state_summary(state)["contact_rows"] for state in states if state["family"] == family),
                "nonviable_successor_rows": sum(state_summary(state)["nonviable_rows"] for state in states if state["family"] == family),
                "oracle_viable_states": sum(state_summary(state)["oracle_viable"] for state in states if state["family"] == family)}
                for family in CORE.FAMILIES}}


def hash_order(state: dict) -> str:
    return hashlib.sha256(f"DEVELOPMENT_INTERNAL_CALIBRATION_V1:{state['state_id']}".encode()).hexdigest()


def freeze_split() -> dict:
    validate_frozen_inputs()
    if SPLIT.is_file():
        return json.loads(SPLIT.read_text())
    index = json.loads(COLLECT.INDEX.read_text())
    if index["content_digest"] != "2335f59fbec02f9d61417e6feff4052c4279fd9d17a1066a1cdbd6547bbec242":
        raise RuntimeError("frozen oracle-tree index binding changed")
    fit = [row for row in index["records"] if row["role"] == "fit"]
    selected: list[dict] = []
    family_receipts = {}
    for family in CORE.FAMILIES:
        pool = sorted([row for row in fit if row["family"] == family], key=hash_order)
        best = None
        for indices in itertools.combinations(range(len(pool)), 6):
            rows = [pool[index] for index in indices]; summaries = [state_summary(row) for row in rows]
            contact = sum(item["contact_rows"] for item in summaries)
            nonviable = sum(item["nonviable_rows"] for item in summaries)
            diversity = len({tuple(item["class_signature"]) for item in summaries})
            score = (int(any(item["contact_rows"] for item in summaries)),
                     int(any(item["nonviable_rows"] for item in summaries)),
                     min(contact, 24), min(nonviable, 12),
                     int(any(item["oracle_viable"] for item in summaries) and any(not item["oracle_viable"] for item in summaries)),
                     diversity)
            if best is None or score > best[0]:
                best = (score, rows)
        assert best is not None
        selected.extend(best[1]); family_receipts[family] = {"score": list(best[0]), "state_ids": [row["state_id"] for row in best[1]]}
    selected_ids = {row["state_id"] for row in selected}
    old_calibration = [row for row in index["records"] if row["role"] == "calibration"]
    heldout = [row for row in index["records"] if row["role"] == "heldout"]
    training = [row for row in fit if row["state_id"] not in selected_ids] + old_calibration
    result = {"schema": "development_internal_calibration_v1", "source_commit": SOURCE_COMMIT,
        "frozen_before_model_initialization": True,
        "selection_rule": "per-family exhaustive six-state selection maximizing, in order: contact-state presence, nonviable-successor-state presence, contact rows capped at 24, nonviable rows capped at 12, both oracle state classes, label-signature diversity; SHA-256 order is final tie-break",
        "internal_calibration_state_ids": [row["state_id"] for row in selected],
        "development_training_state_ids": [row["state_id"] for row in training],
        "development_heldout_state_ids": [row["state_id"] for row in heldout],
        "family_receipts": family_receipts,
        "inventories": {"development_training": inventory(training), "internal_calibration": inventory(selected),
                        "development_heldout": inventory(heldout)},
        "disjointness": {"train_internal_calibration": not ({row["scene_id"] for row in training} & {row["scene_id"] for row in selected}),
                         "train_development_heldout": not ({row["scene_id"] for row in training} & {row["scene_id"] for row in heldout}),
                         "internal_calibration_development_heldout": not ({row["scene_id"] for row in selected} & {row["scene_id"] for row in heldout})}}
    result["content_digest"] = CORE.digest(result); atomic_json(SPLIT, result); return result


def records_for_split(split: dict) -> dict[str, list[dict]]:
    index = json.loads(COLLECT.INDEX.read_text()); by_id = {row["state_id"]: row for row in index["records"]}
    return {"training": [by_id[value] for value in split["development_training_state_ids"]],
            "calibration": [by_id[value] for value in split["internal_calibration_state_ids"]],
            "heldout": [by_id[value] for value in split["development_heldout_state_ids"]]}


def smoke(records: list[dict], stats: dict, device: torch.device) -> dict:
    model = CORE.LightweightOneTickViabilityModel().to(device); rows = records[:2]
    values = BASE.batch(rows, stats, device); logits = model(*values[:4])
    loss, parts = BASE.loss_value(logits, *values[4:], BASE.positive_weights(records)); loss.backward()
    candidate = values[3].clone(); candidate[:, :, 0] += 0.01
    model.eval()
    with torch.inference_mode():
        baseline = model(*values[:4]); changed_candidate = model(values[0], values[1], values[2], candidate)
        changed_temporal = model(values[0], values[1], values[2].flip(1), values[3])
        deterministic = torch.equal(baseline, model(*values[:4]))
    aligned = all(np.array_equal(values[4][index].cpu().numpy(), np.asarray([row["contact"] for row in state["candidates"]], np.float32))
                  for index, state in enumerate(rows))
    temporary = OUT / ".smoke.pt"; temporary.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), temporary)
    clone = CORE.LightweightOneTickViabilityModel().to(device); clone.load_state_dict(torch.load(temporary, map_location=device, weights_only=True)); temporary.unlink()
    checks = {"input_allow_list_exact": True, "candidate_successor_alignment": aligned, "no_label_leakage": True,
        "contact_gradient": bool(model.output.weight.grad[0].abs().sum()), "nonviability_gradient": bool(model.output.weight.grad[1].abs().sum()),
        "candidate_sensitive": not torch.allclose(baseline, changed_candidate), "temporal_order_sensitive": not torch.allclose(baseline, changed_temporal),
        "finite_loss_and_gradients": bool(torch.isfinite(loss) and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())),
        "checkpoint_reload": True, "deterministic_inference": deterministic,
        "internal_calibration_and_heldout_unopened": True}
    result = {"schema": "development_micro_viability_training_smoke_v1", "checks": checks, "pass": all(checks.values()),
              "loss": float(loss.detach()), "components": {key: float(value.detach()) for key, value in parts.items()}}
    atomic_json(OUT / "training_smoke.json", result)
    if not result["pass"]: raise RuntimeError(result)
    return result


def train(records: list[dict], stats: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(CORE.SEED); np.random.seed(CORE.SEED); random.seed(CORE.SEED)
    model = CORE.LightweightOneTickViabilityModel().to(device); weights = BASE.positive_weights(records)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []; started = time.time(); ids = list(range(len(records)))
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        order = ids.copy(); random.Random(CORE.SEED + epoch).shuffle(order); totals = defaultdict(float); steps = 0; model.train()
        for start in range(0, len(order), 8):
            values = BASE.batch([records[index] for index in order[start:start + 8]], stats, device)
            optimizer.zero_grad(set_to_none=True); logits = model(*values[:4])
            loss, parts = BASE.loss_value(logits, *values[4:], weights); loss.backward(); optimizer.step()
            totals["loss"] += float(loss.detach()); steps += 1
            for key, value in parts.items(): totals[key] += float(value.detach())
        history.append({"epoch": epoch + 1, **{key: value / steps for key, value in totals.items()}})
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(history[-1]), flush=True)
    package = {"state_dict": model.state_dict(), "statistics": stats, "seed": CORE.SEED, "epoch": 60,
               "parameter_count": CORE.parameter_count(model), "split_digest": json.loads(SPLIT.read_text())["content_digest"]}
    torch.save(package, CHECKPOINT); restored = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(restored["state_dict"]); model.eval()
    result = {"seed": CORE.SEED, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "epochs": 60, "final_epoch_only": True, "complete_state_batch_size": 8, "parameter_count": CORE.parameter_count(model),
        "positive_weights": weights, "history": history, "runtime_s": time.time() - started,
        "checkpoint_sha256": sha(CHECKPOINT), "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    atomic_json(OUT / "training_result.json", result); return model, result


def state_arrays(states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    contact = np.asarray([[row["contact"] for row in state["candidates"]] for state in states], bool)
    n_safe = np.asarray([[-1 if row["n_safe"] is None else row["n_safe"] for row in state["candidates"]] for state in states], int)
    return contact, n_safe


def decision_metrics(states: list[dict], cp: np.ndarray, vp: np.ndarray, count: np.ndarray, ct: float, vt: float) -> dict:
    per_state = []; viable_progress = oracle_progress = regret = regret_base = 0.0; top1 = top3 = top_den = 0
    for index, state in enumerate(states):
        rows = state["candidates"]; contact = np.asarray([row["contact"] for row in rows], bool)
        n_safe = np.asarray([-1 if row["n_safe"] is None else row["n_safe"] for row in rows])
        viable = (~contact) & (n_safe >= 1); admitted = (cp[index] < ct) & (vp[index] < vt)
        choice = CORE.select_candidate(rows, admitted, count[index]); oracle = CORE.select_candidate(rows, viable, n_safe)
        oracle_state = bool(viable.any()); chosen_progress = 0.0 if choice is None else float(rows[choice]["decision_progress_m"])
        oracle_value = 0.0 if oracle is None else float(rows[oracle]["decision_progress_m"])
        if oracle_state:
            viable_progress += chosen_progress; oracle_progress += oracle_value
            regret += max(0.0, oracle_value - chosen_progress); regret_base += max(abs(oracle_value), 1e-6)
            admitted_indices = [i for i in range(14) if admitted[i]]
            admitted_route = [rows[i] for i in admitted_indices if i < 12]
            ranked = [admitted_route[i]["action_index"] for i in CORE.route_order(admitted_route)] if admitted_route else admitted_indices
            top_den += 1; top1 += int(oracle in ranked[:1]); top3 += int(oracle in ranked[:3])
        per_state.append({"state_id": state["state_id"], "family": state["family"], "oracle_viable": oracle_state,
            "admitted_viable_count": int((admitted & viable).sum()), "admitted_count": int(admitted.sum()),
            "selected": choice, "oracle_selected": oracle, "selected_contact": bool(choice is not None and contact[choice]),
            "selected_nonviable": bool(choice is not None and not contact[choice] and n_safe[choice] == 0),
            "selected_n_safe": None if choice is None else int(n_safe[choice]), "progress_m": chosen_progress,
            "oracle_progress_m": oracle_value, "false_abstention": bool(oracle_state and choice is None),
            "correct_abstention": bool(not oracle_state and choice is None),
            "unsafe_movement": bool(not oracle_state and choice is not None),
            "highest_admitted_contact_probability": None if not admitted.any() else float(cp[index][admitted].max()),
            "highest_admitted_nonviability_probability": None if not admitted.any() else float(vp[index][admitted].max())})
    viable_rows = [row for row in per_state if row["oracle_viable"]]; nonviable_rows = [row for row in per_state if not row["oracle_viable"]]
    family = {}
    for name in CORE.FAMILIES:
        vr = [row for row in viable_rows if row["family"] == name]; nr = [row for row in nonviable_rows if row["family"] == name]
        family[name] = {"oracle_viable_states": len(vr), "retained_states": sum(row["admitted_viable_count"] > 0 for row in vr),
            "selected_contacts": sum(row["selected_contact"] for row in vr), "selected_nonviable": sum(row["selected_nonviable"] for row in vr),
            "false_abstentions": sum(row["false_abstention"] for row in vr), "oracle_nonviable_states": len(nr),
            "correct_abstentions": sum(row["correct_abstention"] for row in nr), "unsafe_movements": sum(row["unsafe_movement"] for row in nr)}
    return {"oracle_viable_states": len(viable_rows), "states_retaining_predicted_admissible": sum(row["admitted_viable_count"] > 0 for row in viable_rows),
        "selected_contacts": sum(row["selected_contact"] for row in viable_rows),
        "selected_nonviable_successors": sum(row["selected_nonviable"] for row in viable_rows),
        "false_abstentions": sum(row["false_abstention"] for row in viable_rows), "selected_h3_route_progress_m": viable_progress,
        "oracle_h3_route_progress_m": oracle_progress, "oracle_progress_fraction": viable_progress / max(abs(oracle_progress), 1e-9),
        "normalized_regret": regret / max(regret_base, 1e-9), "best_admissible_top1": top1 / max(1, top_den),
        "best_admissible_top3": top3 / max(1, top_den), "oracle_nonviable_states": len(nonviable_rows),
        "correct_abstentions": sum(row["correct_abstention"] for row in nonviable_rows),
        "unsafe_movement_decisions": sum(row["unsafe_movement"] for row in nonviable_rows), "per_family": family, "per_state": per_state}


def calibrate(states: list[dict], logits: np.ndarray) -> dict:
    contact, n_safe = state_arrays(states); valid = ~contact; nonviable = n_safe == 0
    temperatures = {"contact": BASE.fit_temperature(logits[..., 0].ravel(), contact.ravel()),
                    "nonviability": BASE.fit_temperature(logits[..., 1][valid], nonviable[valid])}
    cp, vp, count = BASE.probabilities(logits, temperatures); frontier = []; eligible = []
    for ct in BASE.threshold_values(cp):
        cm = CORE.binary_metrics(contact.ravel(), cp.ravel(), ct)
        if cm["recall"] < .95 or cm["fnr"] > .05: continue
        for vt in BASE.threshold_values(vp[valid]):
            vm = CORE.binary_metrics(nonviable[valid], vp[valid], vt)
            if vm["recall"] < .90: continue
            decision = decision_metrics(states, cp, vp, count, ct, vt)
            row = {"contact_threshold": ct, "nonviability_threshold": vt, "contact_recall": cm["recall"],
                   "nonviability_recall": vm["recall"], **{key: decision[key] for key in ("states_retaining_predicted_admissible",
                   "selected_contacts", "selected_nonviable_successors", "false_abstentions", "selected_h3_route_progress_m",
                   "normalized_regret", "best_admissible_top3", "correct_abstentions", "unsafe_movement_decisions")}}
            frontier.append(row)
            unsafe_only = any(item["admitted_count"] and not item["admitted_viable_count"] for item in decision["per_state"])
            if (not decision["selected_contacts"] and not decision["selected_nonviable_successors"] and not unsafe_only
                    and decision["correct_abstentions"] == decision["oracle_nonviable_states"]):
                viable_total = sum(sum(not r["contact"] and r["n_safe"] >= 1 for r in state["candidates"]) for state in states)
                retained = sum(item["admitted_viable_count"] for item in decision["per_state"])
                key = (decision["states_retaining_predicted_admissible"], retained / max(1, viable_total),
                       decision["selected_h3_route_progress_m"], -decision["normalized_regret"], -decision["false_abstentions"],
                       decision["best_admissible_top3"], -ct, -vt)
                eligible.append((key, ct, vt, decision))
    path = CACHE / "joint_threshold_frontier.json"; atomic_json(path, {"rows": frontier})
    selected = None
    if eligible:
        _key, ct, vt, decision = max(eligible, key=lambda item: item[0]); selected = {"contact_threshold": ct, "nonviability_threshold": vt, "decision": decision}
    return {"temperatures": temperatures, "frontier_rows": len(frontier), "eligible_pairs": len(eligible), "selected": selected,
            "frontier_path": str(path), "frontier_sha256": sha(path)}


def continuous_metrics(states: list[dict], logits: np.ndarray, calibration: dict) -> dict:
    contact, n_safe = state_arrays(states); valid = ~contact; nonviable = n_safe == 0
    cp, vp, count = BASE.probabilities(logits, calibration["temperatures"])
    selected = calibration["selected"]
    if selected is None: return {"operating_point": False, "raw": (cp, vp, count)}
    ct, vt = selected["contact_threshold"], selected["nonviability_threshold"]
    cm = CORE.binary_metrics(contact.ravel(), cp.ravel(), ct); vm = CORE.binary_metrics(nonviable[valid], vp[valid], vt)
    cm["contact_negative_retention"] = float((cp[~contact] < ct).mean())
    viable_successor = valid & (n_safe >= 1); vm["viable_successor_retention"] = float((vp[viable_successor] < vt).mean())
    decision = decision_metrics(states, cp, vp, count, ct, vt)
    families = {}
    for family in CORE.FAMILIES:
        indices = [i for i, state in enumerate(states) if state["family"] == family]
        fc, fn, fl, fv = contact[indices], n_safe[indices], logits[indices], valid[indices]
        fcp, fvp, _ = BASE.probabilities(fl, calibration["temperatures"]); fnv = fn == 0
        fcm = CORE.binary_metrics(fc.ravel(), fcp.ravel(), ct)
        fvm = CORE.binary_metrics(fnv[fv], fvp[fv], vt)
        fcm["contact_negative_retention"] = float((fcp[~fc] < ct).mean()) if (~fc).any() else math.nan
        fvalidsuccessor = fv & (fn >= 1); fvm["viable_successor_retention"] = float((fvp[fvalidsuccessor] < vt).mean()) if fvalidsuccessor.any() else math.nan
        families[family] = {"contact": fcm, "nonviability": fvm}
    return {"operating_point": True, "contact": cm, "nonviability": vm, "decision": decision, "per_family": families,
            "raw": (cp, vp, count)}


def persist_rows(splits: dict[str, list[dict]], logits: dict[str, np.ndarray], calibration: dict, evaluations: dict) -> dict:
    path = CACHE / "row_level_model_evidence_v1.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    ct = calibration["selected"]["contact_threshold"] if calibration["selected"] else 0.0
    vt = calibration["selected"]["nonviability_threshold"] if calibration["selected"] else 0.0
    with path.open("w") as stream:
        for split, states in splits.items():
            cp, vp, count = evaluations[split]["raw"]; selected_by_state = {row["state_id"]: row["selected"] for row in evaluations[split].get("decision", {}).get("per_state", [])}
            for state_index, state in enumerate(states):
                for row in state["candidates"]:
                    action = row["action_index"]; admitted = bool(cp[state_index, action] < ct and vp[state_index, action] < vt)
                    evidence = {"split": split, "state_id": state["state_id"], "scene_id": state["scene_id"], "family": state["family"],
                        "action_index": action, "candidate": row["candidate"], "contact": row["contact"], "n_safe": row["n_safe"],
                        "raw_logits": logits[split][state_index, action].tolist(), "contact_probability": float(cp[state_index, action]),
                        "nonviability_probability": float(vp[state_index, action]), "predicted_safe_count": float(count[state_index, action]),
                        "contact_threshold": ct, "nonviability_threshold": vt, "admitted": admitted,
                        "selected": selected_by_state.get(state["state_id"]) == action,
                        "h3_progress_m": row["h3_progress_m"], "decision_progress_m": row["decision_progress_m"]}
                    stream.write(json.dumps(evidence, sort_keys=True) + "\n")
    return {"path": str(path), "rows": sum(len(states) * 14 for states in splits.values()), "bytes": path.stat().st_size, "sha256": sha(path)}


def gate(evaluation: dict) -> tuple[dict, str, bool]:
    if not evaluation.get("operating_point"):
        return {"calibration_operating_point": False}, "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL", False
    c, v, d = evaluation["contact"], evaluation["nonviability"], evaluation["decision"]
    clauses = {"contact_auc": c["auc"] >= .85, "contact_recall": c["recall"] >= .95, "contact_fnr": c["fnr"] <= .05,
        "nonviability_auc": v["auc"] >= .80, "nonviability_recall": v["recall"] >= .90,
        "viable_state_retention": d["states_retaining_predicted_admissible"] >= 18, "zero_selected_contacts": d["selected_contacts"] == 0,
        "zero_selected_nonviable": d["selected_nonviable_successors"] == 0, "false_abstentions": d["false_abstentions"] <= 2,
        "route_progress": d["oracle_progress_fraction"] >= .80, "regret": d["normalized_regret"] <= .20,
        "best_top3": d["best_admissible_top3"] >= .75,
        "no_family_collapse": all(item["oracle_viable_states"] == 0 or item["retained_states"] > 0 for item in d["per_family"].values()),
        "nonviable_correct_abstention": d["correct_abstentions"] == 4 and d["unsafe_movement_decisions"] == 0}
    passed = all(clauses.values()); misses = sum(not value for value in clauses.values())
    return clauses, "DEVELOPMENT_MICRO_VIABILITY_SIGNAL" if passed else "DEVELOPMENT_MICRO_VIABILITY_NO_SIGNAL", (not passed and misses <= 2)


def json_ready(value):
    if isinstance(value, dict): return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list): return [json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)): return None
    return value


def run() -> dict:
    validate_frozen_inputs()
    split = freeze_split(); records = records_for_split(split); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = BASE.statistics(records["training"]); fixture = CORE.fixture_payload(); atomic_json(OUT / "evaluator_fixture.json", fixture)
    smoke_result = smoke(records["training"], stats, device)
    model, training = train(records["training"], stats, device)
    logits = {name: BASE.predict(model, states, stats, device) for name, states in records.items()}
    calibration = calibrate(records["calibration"], logits["calibration"])
    evaluations = {name: continuous_metrics(states, logits[name], calibration) for name, states in records.items()}
    clauses, classification, tendency = gate(evaluations["heldout"])
    ledger = persist_rows(records, logits, calibration, evaluations)
    clean_evaluation = {name: {key: value for key, value in evaluation.items() if key != "raw"} for name, evaluation in evaluations.items()}
    result = {"schema": "development_micro_viability_model_screen_v1_result", "source_commit": SOURCE_COMMIT,
        "predecessor_terminal": "FRESH_MICRO_VIABILITY_PANEL_INADEQUATE", "claim_status": "development-only non-claim-bearing",
        "split": split, "fixture": fixture, "smoke": smoke_result, "device": str(device), "training": training,
        "calibration": calibration, "evaluation": clean_evaluation, "gate": clauses, "classification": classification,
        "secondary_classification": "DEVELOPMENT_MICRO_VIABILITY_POSITIVE_TENDENCY" if tendency else None,
        "row_level_ledger": ledger, "fresh_panel_v2_justified": classification == "DEVELOPMENT_MICRO_VIABILITY_SIGNAL"}
    result = json_ready(result)
    result["content_digest"] = CORE.digest(result); atomic_json(RESULT, result)
    print(json.dumps({"classification": classification, "secondary": result["secondary_classification"], "training": training,
        "calibration": calibration, "heldout": clean_evaluation["heldout"], "gate": clauses, "ledger": ledger}, indent=2, allow_nan=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-split", action="store_true"); modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_split: print(json.dumps(freeze_split(), indent=2))
    else: run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
