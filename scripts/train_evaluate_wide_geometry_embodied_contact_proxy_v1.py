#!/usr/bin/env python3
"""Train and evaluate the single Stage-1 wide-geometry contact-proxy model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts", ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(extra) not in sys.path: sys.path.insert(0, str(extra))

from scripts import collect_wide_geometry_embodied_contact_proxy_v1 as COLLECT
from scripts import evaluate_wide_geometry_score_composition_v1 as STAGE0
from scripts import materialize_wide_geometry_embodied_contact_proxy_v1 as FRESH_GEO
from scripts import materialize_geometry_modality_safety_sufficiency_v1 as GEO
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE
from scripts import train_evaluate_factorised_micro_safety_data_scaling_v2 as SCALE
from scripts import train_evaluate_geometry_modality_safety_sufficiency_v1 as OLD

OUT = COLLECT.OUT
CACHE = COLLECT.CACHE / "stage1"
SEED = 2026082013
MODEL_NAME = "WIDE_GEOMETRY_EMBODIED_CONTACT_HEAD_V1"
FAMILIES = STAGE0.FAMILIES
RECOVERED_TRAINING_MILESTONES = [
    {"epoch": 1, "loss": 1.186064320554336, "bce": 1.045254045476516, "ranking": 0.563241102732718},
    {"epoch": 10, "loss": 0.2879028081273039, "bce": 0.2774242169378946, "ranking": 0.04191436629053745},
    {"epoch": 20, "loss": 0.1059471998460746, "bce": 0.104269294780291, "ranking": 0.006711620448161189},
    {"epoch": 30, "loss": 0.03713338886943044, "bce": 0.03686559892691245, "ranking": 0.0010711595370834289},
    {"epoch": 40, "loss": 0.003993198412111572, "bce": 0.003993037245345476, "ranking": 6.446670981143561e-07},
    {"epoch": 50, "loss": 0.019868473941702783, "bce": 0.01986219757551832, "ranking": 2.5105226455165073e-05},
    {"epoch": 60, "loss": 0.0014942368417981318, "bce": 0.0014942299979566088, "ranking": 2.733081555771529e-08},
]


def sha(path: Path) -> str: return COLLECT.sha(path)
def atomic_json(path: Path, payload: dict) -> None: COLLECT.atomic_json(path, payload)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle: np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def fit_rows():
    rows, index = OLD.load_rows()
    for row in rows: row["split"] = "fit"
    if len(rows) != 2880 or len({row["state_id"] for row in rows}) != 240: raise RuntimeError("fit-240 cardinality mismatch")
    return rows, index


def fresh_rows():
    manifest = json.loads(COLLECT.PANEL.read_text()); sensor_index = json.loads(COLLECT.SENSOR_INDEX.read_text())
    geometry_index = json.loads(FRESH_GEO.INDEX.read_text()); states = {row["state_id"]: row for row in manifest["states"]}
    sensor_records = {row["state_id"]: row for row in sensor_index["state_records"]}; geometry_records = {row["state_id"]: row for row in geometry_index["state_records"]}
    if set(states) != set(sensor_records) or set(states) != set(geometry_records): raise RuntimeError("fresh sensor/geometry identity mismatch")
    rows = []
    for state_id in [row["state_id"] for row in manifest["states"]]:
        state, sensor, geometry = states[state_id], sensor_records[state_id], geometry_records[state_id]
        if sha(Path(sensor["shard_path"])) != sensor["shard_sha256"] or sha(Path(geometry["shard_path"])) != geometry["shard_sha256"]:
            raise RuntimeError(f"fresh shard mismatch {state_id}")
        with np.load(sensor["shard_path"], allow_pickle=False) as loaded:
            current = np.asarray(loaded["current"], np.float32); future = np.asarray(loaded["future"], np.float32)
            action = np.asarray(loaded["action_control"], np.float32); labels = np.asarray(loaded["labels"], np.float32)
        with np.load(geometry["shard_path"], allow_pickle=False) as loaded:
            geo = {key: np.asarray(loaded[key], np.float32) for key in loaded.files}
        heading_body = float(sensor["branches"][0]["route_heading_world_rad"]) - float(state["start_pose"][1])
        waypoint = [*map(float, state["waypoint_body_xy"]), np.sin(heading_body), np.cos(heading_body)]
        for branch in sensor["branches"]:
            candidate = int(branch["candidate_index"])
            rows.append({**branch, "split": state["split"], "family": state["family"], "scene_id": state["scene_id"],
                "current_enhanced": current[candidate], "future_enhanced": future[candidate], "action_control": action[candidate],
                "target": labels[candidate], "kinematic": SCALE.BASE.integrate(branch["post_slew"], waypoint),
                "current_depth": geo["current_depth"], "future_depth": geo["future_depth"][candidate],
                "current_depth_valid": geo["current_depth_valid"], "future_depth_valid": geo["future_depth_valid"][candidate],
                "current_lidar": geo["current_lidar"], "future_lidar": geo["future_lidar"][candidate],
                "current_lidar_valid": geo["current_lidar_valid"], "future_lidar_valid": geo["future_lidar_valid"][candidate],
                "hard_contact": bool(labels[candidate, -1, 2] > .5), "stuck": bool(labels[candidate, -1, 3] > .5)})
    if len(rows) != 576 or len({row["state_id"] for row in rows}) != 48: raise RuntimeError("fresh row cardinality mismatch")
    return rows, {"manifest": manifest, "sensor_index": sensor_index, "geometry_index": geometry_index}


def one_prevalence(rows):
    contact = np.asarray([row["target"][-1, 2] > .5 for row in rows]); stuck = np.asarray([row["target"][-1, 3] > .5 for row in rows])
    unsafe = np.asarray([row["target"][-1, 4] > .5 for row in rows]); grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    return {"states": len(grouped), "branches": len(rows), "contact_negative": int((~contact).sum()), "contact_positive": int(contact.sum()),
        "stuck_positive": int(stuck.sum()), "contact_stuck_overlap": int((contact & stuck).sum()), "aggregate_unsafe": int(unsafe.sum()),
        "contact_event_ticks": int(sum(row["target"][:, 0].sum() for row in rows)), "stuck_event_ticks": int(sum(row["target"][:, 1].sum() for row in rows)),
        "states_with_contact_negative": int(sum(any(not bool(row["hard_contact"]) for row in state_rows) for state_rows in grouped.values())),
        "states_no_contact_negative": int(sum(all(bool(row["hard_contact"]) for row in state_rows) for state_rows in grouped.values()))}


def prevalence(fit, calibration, heldout):
    output = {name: one_prevalence(rows) for name, rows in (("fit240", fit), ("calibration", calibration), ("heldout", heldout))}
    output["per_family"] = {name: {family: one_prevalence([row for row in rows if row["family"] == family]) for family in FAMILIES}
        for name, rows in (("fit240", fit), ("calibration", calibration), ("heldout", heldout))}
    return output


class DepthEncoder(OLD.DepthEncoder): pass
class LidarEncoder(OLD.LidarEncoder): pass


class WideGeometryContactHead(nn.Module):
    def __init__(self):
        super().__init__(); self.depth = DepthEncoder(); self.lidar = LidarEncoder()
        self.embodied = nn.Sequential(nn.Linear(73 * 3, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU())
        self.action = nn.Sequential(nn.Linear(6, 48), nn.GELU())
        self.temporal = nn.GRU(96 * 3 + 48, 160, batch_first=True); self.output = nn.Linear(160, 2)

    def forward(self, depth, lidar, embodied, action):
        features = torch.cat((self.depth(depth), self.lidar(lidar), self.embodied(embodied), self.action(action)), -1)
        hidden, _ = self.temporal(features); return self.output(hidden)


def embodied_stats(rows): return OLD.embodied_stats(rows)


def tensors(rows, statistics, device):
    def geometry(prefix, far):
        current = np.stack([row[f"current_{prefix}"] for row in rows]); future = np.stack([row[f"future_{prefix}"] for row in rows])
        current_valid = np.stack([row[f"current_{prefix}_valid"] for row in rows]); future_valid = np.stack([row[f"future_{prefix}_valid"] for row in rows])
        current = np.repeat(current[:, None], 15, 1); current_valid = np.repeat(current_valid[:, None], 15, 1)
        return np.stack((current / far, future / far, (future - current) / far, current_valid, future_valid, current_valid * future_valid), 2).astype(np.float32)
    depth = geometry("depth", GEO.DEPTH_FAR_M); lidar = geometry("lidar", GEO.LIDAR_FAR_M)
    mean, std = np.asarray(statistics["mean"], np.float32), np.asarray(statistics["std"], np.float32)
    current = np.stack([(row["current_enhanced"] - mean) / std for row in rows]); future = np.stack([(row["future_enhanced"] - mean) / std for row in rows])
    current = np.repeat(current[:, None], 15, 1); embodied = np.concatenate((current, future, future - current), -1).astype(np.float32)
    action = np.stack([row["action_control"] for row in rows]).astype(np.float32); target = np.stack([row["target"][:, [0, 2]] for row in rows]).astype(np.float32)
    return tuple(torch.from_numpy(value).to(device) for value in (depth, lidar, embodied, action, target))


def positive_weights(rows): return OLD.positive_weights(rows)


def ranking_loss(logits, target):
    total = []; cumulative = logits[:, -1, 1]; label = target[:, -1, 1] > .5
    for start in range(0, len(logits), 12):
        positive = cumulative[start:start + 12][label[start:start + 12]]; negative = cumulative[start:start + 12][~label[start:start + 12]]
        if len(positive) and len(negative): total.append(F.softplus(-(positive[:, None] - negative[None, :])).mean())
    return torch.stack(total).mean() if total else cumulative.sum() * 0


def loss_value(logits, target, weight):
    bce = sum(F.binary_cross_entropy_with_logits(logits[..., index], target[..., index], pos_weight=weight[index]) for index in range(2)) / 2
    rank = ranking_loss(logits, target); return bce + .25 * rank, bce, rank


def evaluator_fixture():
    labels = np.asarray([0, 1, 0, 1], bool); perfect = np.asarray([.01, .99, .02, .98]); reverse = 1 - perfect
    cases = {"perfect_auc": bool(STAGE0.auc(labels, perfect) == 1.), "reversed_auc": bool(STAGE0.auc(labels, reverse) == 0.),
        "transient_contact": bool(np.asarray([0, 1, 0]).sum() == 1), "persistent_contact": bool(np.asarray([0, 1, 1]).sum() == 2),
        "threshold_tie_rejected": not (.5 < .5), "no_admission": not bool((np.asarray([.8, .9]) < .5).any()),
        "row_serialization": json.dumps({"x": [1, 2]}, sort_keys=True) == '{"x": [1, 2]}'}
    payload = {"schema": "wide_geometry_contact_proxy_evaluator_fixture_v1", "cases": cases, "pass": all(cases.values())}
    atomic_json(OUT / "stage1_evaluator_fixture.json", payload)
    if not payload["pass"]: raise RuntimeError("Stage-1 evaluator fixture failure")
    return payload


def train(fit, device):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED); statistics = embodied_stats(fit)
    model = WideGeometryContactHead().to(device); count = sum(parameter.numel() for parameter in model.parameters())
    if count >= 750000: raise RuntimeError(f"parameter cap exceeded: {count}")
    weight = torch.from_numpy(positive_weights(fit)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    grouped = defaultdict(list)
    for row in fit: grouped[row["state_id"]].append(row)
    state_ids = sorted(grouped); smoke = grouped[state_ids[0]]; depth, lidar, embodied, action, target = tensors(smoke, statistics, device)
    logits = model(depth, lidar, embodied, action); loss, _, _ = loss_value(logits, target, weight); loss.backward()
    if not torch.isfinite(loss) or any(parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0 for parameter in model.parameters()):
        raise RuntimeError("training smoke gradient failure")
    model.eval()
    with torch.inference_mode():
        baseline = model(depth, lidar, embodied, action)
        if not torch.equal(baseline, model(depth, lidar, embodied, action)): raise RuntimeError("nondeterministic inference")
        variants = ((depth.roll(1, 1), lidar, embodied, action), (depth, lidar.roll(1, 1), embodied, action),
                    (depth, lidar, embodied.roll(1, 1), action), (depth, lidar, embodied, action.roll(1, 1)))
        if any(torch.allclose(baseline, model(*variant)) for variant in variants): raise RuntimeError("modality sensitivity failure")
        if torch.allclose(baseline, model(depth.flip(1), lidar.flip(1), embodied.flip(1), action.flip(1))): raise RuntimeError("temporal sensitivity failure")
    model.train(); model.zero_grad(set_to_none=True)
    smoke_path = OUT / ".wide_geometry_smoke.pt"; torch.save(model.state_dict(), smoke_path); clone = WideGeometryContactHead().to(device)
    clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    history = []; started = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        order = list(state_ids); random.Random(SEED + epoch).shuffle(order); totals = [0., 0., 0.]; batches = 0
        for start in range(0, len(order), 2):
            batch = [row for state_id in order[start:start + 2] for row in grouped[state_id]]
            depth, lidar, embodied, action, target = tensors(batch, statistics, device); optimizer.zero_grad(set_to_none=True)
            logits = model(depth, lidar, embodied, action); loss, bce, rank = loss_value(logits, target, weight); loss.backward(); optimizer.step()
            totals[0] += float(loss.detach()); totals[1] += float(bce.detach()); totals[2] += float(rank.detach()); batches += 1
        history.append({"epoch": epoch + 1, "loss": totals[0] / batches, "bce": totals[1] / batches, "ranking": totals[2] / batches})
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(history[-1]), flush=True)
    checkpoint = OUT / f"wide_geometry_embodied_contact_head_seed_{SEED}.pt"
    torch.save({"state_dict": model.state_dict(), "model": MODEL_NAME, "seed": SEED, "epoch": 60, "embodied_stats": statistics,
                "parameter_count": count, "positive_weights": weight.cpu().tolist(), "ranking_weight": .25}, checkpoint)
    package = torch.load(checkpoint, map_location=device, weights_only=True); restored = WideGeometryContactHead().to(device)
    restored.load_state_dict(package["state_dict"]); restored.eval()
    return restored, statistics, {"model": MODEL_NAME, "seed": SEED, "parameter_count": count, "epochs": 60, "final_epoch_only": True,
        "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001, "ranking_weight": .25, "positive_weights": weight.cpu().tolist(),
        "history": history, "runtime_s": time.time() - started, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size, "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "smoke_passed": True, "input_allow_list": ["front_depth", "lidar", "enhanced_embodied", "candidate_action", "control_history"]}


def load_completed_checkpoint(device):
    checkpoint = OUT / f"wide_geometry_embodied_contact_head_seed_{SEED}.pt"; package = torch.load(checkpoint, map_location=device, weights_only=True)
    if package["model"] != MODEL_NAME or package["seed"] != SEED or package["epoch"] != 60: raise RuntimeError("completed checkpoint contract mismatch")
    model = WideGeometryContactHead().to(device); model.load_state_dict(package["state_dict"]); model.eval()
    geometry_index = FRESH_GEO.INDEX
    elapsed_upper_bound = max(0.0, checkpoint.stat().st_mtime - geometry_index.stat().st_mtime)
    return model, package["embodied_stats"], {"model": MODEL_NAME, "seed": SEED, "parameter_count": package["parameter_count"], "epochs": 60,
        "final_epoch_only": True, "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001, "ranking_weight": package["ranking_weight"],
        "positive_weights": package["positive_weights"], "history_milestones": RECOVERED_TRAINING_MILESTONES, "runtime_s": None,
        "training_plus_smoke_runtime_upper_bound_s": elapsed_upper_bound,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "smoke_passed": True, "input_allow_list": ["front_depth", "lidar", "enhanced_embodied", "candidate_action", "control_history"],
        "recovered_after_post_training_persistence_fault": True, "retrained": False,
        "recovery_note": "The sole 60-epoch run and final checkpoint completed; evaluation resumed from that checkpoint after fixing only an FP32 fit-ledger route-field persistence error."}


def predict(model, rows, statistics, device):
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), 36):
            depth, lidar, embodied, action, _ = tensors(rows[start:start + 36], statistics, device)
            values.append(model(depth, lidar, embodied, action).float().cpu().numpy())
    return np.concatenate(values)


def calibrate(rows, logits):
    labels = np.asarray([row["hard_contact"] for row in rows]); temperature = DENSE.fit_temperature(logits[:, -1, 1], labels)
    probability = 1 / (1 + np.exp(-logits[:, -1, 1] / temperature)); candidates = []; frontier = defaultdict(list)
    states = STAGE0.prepare_states(rows)
    for threshold in STAGE0.threshold_values(probability):
        metric = STAGE0.decision_metrics(rows, states, probability < threshold)
        for key in ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "states_retaining_contact_negative",
                    "mean_selected_route_progress_m", "normalized_route_progress_regret", "false_abstentions", "best_contact_negative_top3"):
            frontier[key].append(np.nan if metric[key] is None else metric[key])
        frontier["threshold"].append(float(threshold))
        if metric["contact_recall"] >= .95:
            key = (metric["states_retaining_contact_negative"], metric["contact_negative_retention"], metric["mean_selected_route_progress_m"],
                   -(metric["normalized_route_progress_regret"] if metric["normalized_route_progress_regret"] is not None else 1e9),
                   -metric["false_abstentions"], metric["best_contact_negative_top3"] or -1., -float(threshold))
            candidates.append((key, float(threshold), metric))
    arrays = {key: np.asarray(value) for key, value in frontier.items()}; path = CACHE / "calibration_threshold_frontier_v1.npz"; atomic_npz(path, **arrays)
    if not candidates: raise RuntimeError("fresh calibration threshold feasibility failure")
    _, threshold, metrics = max(candidates, key=lambda value: value[0])
    return {"temperature": temperature, "threshold": threshold, "metrics": metrics, "thresholds_evaluated": len(arrays["threshold"]),
            "eligible_thresholds": len(candidates), "frontier": {"file": str(path), "sha256": sha(path)},
            "tie_rule": "more conservative after state availability, negative retention, progress, regret, abstention, and top-3"}


def contact_metrics(rows, logits, calibration): return OLD.contact_metrics(rows, logits, calibration["temperature"], calibration["threshold"])


def per_family(rows, logits, calibration):
    output = {}; probability = 1 / (1 + np.exp(-logits[:, -1, 1] / calibration["temperature"]))
    for family in FAMILIES:
        ids = [index for index, row in enumerate(rows) if row["family"] == family]; sub = [rows[index] for index in ids]
        states = STAGE0.prepare_states(sub); output[family] = {"contact": contact_metrics(sub, logits[ids], calibration),
            "filtering_and_planning": STAGE0.decision_metrics(sub, states, probability[ids] < calibration["threshold"], True)}
    return output


def heldout_gate(contact, planning, families):
    no_collapse = all(value["filtering_and_planning"]["states_retaining_contact_negative"] > 0 and value["filtering_and_planning"]["selected_contact_count"] == 0 for value in families.values())
    checks = {"auc_ge_0_90": contact["auc"] >= .90, "ap_ge_0_90": contact["average_precision"] >= .90, "recall_ge_0_95": contact["recall"] >= .95,
        "fnr_le_0_05": contact["false_negative_rate"] <= .05, "negative_retention_ge_0_50": contact["contact_negative_retention"] >= .50,
        "ece_le_0_10": contact["ece"] <= .10, "event_recall_ge_0_85": contact["event_tick_recall"] >= .85,
        "delay_le_1": contact["median_detection_delay_ticks"] is not None and contact["median_detection_delay_ticks"] <= 1,
        "states_retaining_ge_18": planning["states_retaining_contact_negative"] >= 18, "no_only_positive_state": planning["states_only_contact_positive_admitted"] == 0,
        "zero_selected_contact": planning["selected_contact_count"] == 0, "false_abstentions_le_3": planning["false_abstentions"] <= 3,
        "progress_fraction_ge_0_80": planning["oracle_progress_fraction"] is not None and planning["oracle_progress_fraction"] >= .80,
        "regret_le_0_20": planning["normalized_route_progress_regret"] is not None and planning["normalized_route_progress_regret"] <= .20,
        "top3_ge_0_75": planning["best_contact_negative_top3"] is not None and planning["best_contact_negative_top3"] >= .75,
        "no_family_collapse": no_collapse}
    return {"checks": checks, "passed": all(checks.values())}


def persist_ledger(fit, fit_logits, calibration_rows, calibration_logits, heldout, heldout_logits, calibration):
    rows = fit + calibration_rows + heldout; logits = np.concatenate((fit_logits, calibration_logits, heldout_logits)); probability = 1 / (1 + np.exp(-logits / calibration["temperature"]))
    admitted = probability[:, -1, 1] < calibration["threshold"]; fresh_start = len(fit); fresh_rows = rows[fresh_start:]; fresh_admitted = admitted[fresh_start:]
    planning = STAGE0.decision_metrics(fresh_rows, STAGE0.prepare_states(fresh_rows), fresh_admitted, True)
    selected = {(row["state_id"], row["selected_candidate"]) for row in planning["per_state"] if row["selected_candidate"] is not None}
    arrays = {"branch_id": np.asarray([row["branch_id"] for row in rows]), "state_id": np.asarray([row["state_id"] for row in rows]),
        "candidate_index": np.asarray([row["candidate_index"] for row in rows], np.int16), "split": np.asarray([row["split"] for row in rows]),
        "family": np.asarray([row["family"] for row in rows]), "raw_logits": logits.astype(np.float32), "calibrated_probability": probability.astype(np.float64),
        "contact_labels": np.stack([row["target"][:, [0, 2]] for row in rows]).astype(np.uint8),
        "stuck_labels": np.stack([row["target"][:, [1, 3]] for row in rows]).astype(np.uint8), "threshold_decision_admitted": admitted.astype(np.uint8),
        "selected": np.asarray([(row["state_id"], row["candidate_index"]) in selected for row in rows], np.uint8),
        "p_d": np.asarray([row.get("p_d", np.nan) for row in rows], np.float64), "p_theta": np.asarray([row.get("p_theta", np.nan) for row in rows], np.float64),
        "kinematic": np.stack([row.get("kinematic", np.full(6, np.nan, np.float32)) for row in rows]).astype(np.float32)}
    path = CACHE / "row_level_evidence_v1.npz"; atomic_npz(path, **arrays)
    return {"file": str(path), "sha256": sha(path), "content_digest": hashlib.sha256(b"".join(np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays))).hexdigest(),
            "rows": len(rows), "fit_rows": len(fit), "calibration_rows": len(calibration_rows), "heldout_rows": len(heldout), "fields": sorted(arrays)}


def comparators():
    geometry = json.loads((ROOT / ".generated/geometry_modality_safety_sufficiency_v1/result.json").read_text())
    scaling = json.loads((ROOT / ".generated/factorised_micro_safety_data_scaling_v2/result.json").read_text())
    guard = json.loads((ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1/result.json").read_text())
    return {name: {"panel": "previous geometry heldout-24", "contact": geometry["conditions"][name]["heldout"]["contact"],
                   "planning": geometry["conditions"][name]["heldout"]["filtering_and_planning"]}
            for name in ("DEPTH_ONLY", "LIDAR_ONLY", "DEPTH_PLUS_EMBODIED")} | {
        "FIT_192_CONTACT": {"panel": "previous geometry heldout-24", "contact": scaling["conditions"]["fit192"]["heldout"]["components"]["contact"]},
        "STATIC_GRID_GUARD": {"panel": "historical heldout-8", "metrics": guard["guard_discrimination"]["by_split"]["heldout"]},
        "ORACLE_CONTACT": {"panel": "new fresh heldout-24", "auc": 1.0, "recall": 1.0, "retention": 1.0, "selected_contact_count": 0}}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", required=True); parser.parse_args()
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "stage0_result.json").read_text())
    if stage0["classification"] != "WIDE_GEOMETRY_SCORE_FRONTIER_NO_GO" or not stage0["stage1_authorized"]: raise RuntimeError("Stage 1 is not authorized")
    fixture = evaluator_fixture(); fit, old_geometry_index = fit_rows(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = OUT / f"wide_geometry_embodied_contact_head_seed_{SEED}.pt"
    if checkpoint.is_file():
        model, statistics, training = load_completed_checkpoint(device)
    else:
        model, statistics, training = train(fit, device)  # no fresh calibration/heldout row loaded before final fit update
    fresh, bindings = fresh_rows(); calibration_rows = [row for row in fresh if row["split"] == "calibration"]; heldout = [row for row in fresh if row["split"] == "heldout"]
    prevalence_result = prevalence(fit, calibration_rows, heldout)
    calibration_logits = predict(model, calibration_rows, statistics, device); calibration = calibrate(calibration_rows, calibration_logits)
    heldout_logits = predict(model, heldout, statistics, device); contact = contact_metrics(heldout, heldout_logits, calibration)
    heldout_probability = 1 / (1 + np.exp(-heldout_logits[:, -1, 1] / calibration["temperature"])); states = STAGE0.prepare_states(heldout)
    planning = STAGE0.decision_metrics(heldout, states, heldout_probability < calibration["threshold"], True); families = per_family(heldout, heldout_logits, calibration)
    gate = heldout_gate(contact, planning, families); fit_logits = predict(model, fit, statistics, device); ledger = persist_ledger(fit, fit_logits, calibration_rows, calibration_logits, heldout, heldout_logits, calibration)
    if gate["passed"]: classification = "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_SIGNAL"
    else:
        historical = comparators()["DEPTH_PLUS_EMBODIED"]["contact"]
        improved = contact["auc"] > historical["auc"] or contact["contact_negative_retention"] > historical["contact_negative_retention"]
        classification = "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_POSITIVE_TENDENCY" if improved else "WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_NO_SIGNAL"
    result = {"schema": "wide_geometry_embodied_contact_proxy_v1_result", "source_commit": "58d91deb37d41a129e64a6a0c17ae8b9b6f135d2",
        "claim_boundary": "SIMULATED_DISALLOWED_CONTACT_PROXY only; ordinary foot-ground and self-contact excluded; no material-hazard, injury, property-damage, people, or fragile-infrastructure claim",
        "preserved_ontology_result": "CONTACT_HAZARD_ONTOLOGY_OR_INSTRUMENTATION_INSUFFICIENT", "stage0": stage0,
        "stage1_ran": True, "bindings": {"fit_states": 240, "fit_branches": 2880, "fit_geometry_index_digest": old_geometry_index["content_digest"],
            "fresh_panel_digest": bindings["manifest"]["content_digest"], "fresh_sensor_index_digest": bindings["sensor_index"]["content_digest"],
            "fresh_geometry_index_digest": bindings["geometry_index"]["content_digest"]}, "prevalence": prevalence_result, "evaluator_fixture": fixture,
        "sensor_contracts": {"depth": bindings["geometry_index"]["depth_contract"], "lidar": bindings["geometry_index"]["lidar_contract"],
                             "enhanced_embodied_channels": bindings["sensor_index"]["channels"]},
        "training": training, "calibration": calibration, "heldout": {"contact": contact, "filtering_and_planning": planning, "per_family": families, "gate": gate},
        "comparators": comparators(), "row_level_evidence": ledger, "classification": classification,
        "stuck_claim_boundary": "descriptive recoverability/task-performance outcome only; never used for contact admission",
        "next_experiment": "CANDIDATE_CONDITIONED_WIDE_GEOMETRY_CONTACT_PROXY_PREDICTOR_V1" if gate["passed"] else "Return for decision; no automatic additional data, architecture, or predictor",
        "runtime": {"total_script_s": time.time() - started, "branch_generation_compute_s": bindings["sensor_index"]["runtime_compute_s"],
            "branch_generation_wall_s": bindings["sensor_index"]["parallel_wall_runtime_s"], "geometry_materialization_compute_s": bindings["geometry_index"]["runtime_compute_s"],
            "new_sensor_storage_bytes": bindings["sensor_index"]["storage_bytes"], "new_geometry_storage_bytes": bindings["geometry_index"]["storage_bytes"]},
        "custody": {"one_new_model_seed": True, "jepa_predictor_opened": False, "memory_novelty_navigation_trained": False, "material_safety_claimed": False}}
    result["content_digest"] = COLLECT.canonical_digest(result); atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "checkpoint_sha256": training["checkpoint_sha256"], "parameter_count": training["parameter_count"],
        "calibration": {"temperature": calibration["temperature"], "threshold": calibration["threshold"]}, "heldout_contact": contact,
        "heldout_planning": {key: value for key, value in planning.items() if key != "per_state"}, "gate": gate, "ledger": ledger,
        "runtime_s": result["runtime"]["total_script_s"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
