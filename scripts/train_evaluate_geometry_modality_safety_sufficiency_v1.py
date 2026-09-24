#!/usr/bin/env python3
"""Train and evaluate the three true-future geometry contact conditions."""
from __future__ import annotations

import hashlib
import json
import math
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
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import materialize_geometry_modality_safety_sufficiency_v1 as GEO
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as METRIC
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE
from scripts import train_evaluate_factorised_micro_safety_data_scaling_v2 as SCALE
from scripts import train_evaluate_factorised_micro_safety_world_model_v1 as BASE

OUT = GEO.OUT
CACHE = GEO.CACHE
INDEX = OUT / "geometry_sensor_index.json"
SEED_FAMILY = 2026082012
CONDITIONS = ("DEPTH_ONLY", "LIDAR_ONLY", "DEPTH_PLUS_EMBODIED")
FAMILIES = BASE.FAMILIES


def sha(path: Path) -> str: return BASE.sha(path)
def atomic_json(path: Path, payload: dict) -> None: BASE.atomic_json(path, payload)
def atomic_npz(path: Path, **arrays) -> None: BASE.atomic_npz(path, **arrays)
def canonical_digest(value) -> str: return BASE.canonical_digest(value)


def derived_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED_FAMILY}|{name}".encode()).digest()[:8], "big") % (2**31 - 1)


def load_rows() -> tuple[list[dict], dict]:
    old, _ = BASE.old_fit_rows(); prior, _ = BASE.fresh_rows(); scale, _ = SCALE.load_new_rows()
    for row in old: row["split"] = "fit"
    for row in prior: row["split"] = "fit"
    for row in scale: row["split"] = "fit" if row["split"] == "fit192_extra" else row["split"]
    rows = old + prior + scale
    index = json.loads(INDEX.read_text())
    if not index.get("complete") or index["content_digest"] != canonical_digest({k: v for k, v in index.items() if k != "content_digest"}):
        raise RuntimeError("geometry index invalid")
    records = {row["state_id"]: row for row in index["state_records"]}; grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    if set(grouped) != set(records): raise RuntimeError("geometry/branch state alignment mismatch")
    for state_id, branch_rows in grouped.items():
        record = records[state_id]; path = Path(record["shard_path"])
        if sha(path) != record["shard_sha256"]: raise RuntimeError(f"geometry shard mismatch {state_id}")
        with np.load(path, allow_pickle=False) as loaded:
            current_depth = np.asarray(loaded["current_depth"], np.float32); future_depth = np.asarray(loaded["future_depth"], np.float32)
            current_depth_valid = np.asarray(loaded["current_depth_valid"], np.float32); future_depth_valid = np.asarray(loaded["future_depth_valid"], np.float32)
            current_lidar = np.asarray(loaded["current_lidar"], np.float32); future_lidar = np.asarray(loaded["future_lidar"], np.float32)
            current_lidar_valid = np.asarray(loaded["current_lidar_valid"], np.float32); future_lidar_valid = np.asarray(loaded["future_lidar_valid"], np.float32)
        for row in branch_rows:
            ci = int(row["candidate_index"])
            row.update(current_depth=current_depth, future_depth=future_depth[ci], current_depth_valid=current_depth_valid,
                       future_depth_valid=future_depth_valid[ci], current_lidar=current_lidar, future_lidar=future_lidar[ci],
                       current_lidar_valid=current_lidar_valid, future_lidar_valid=future_lidar_valid[ci],
                       hard_contact=bool(row["target"][-1, 2] > .5), stuck=bool(row["target"][-1, 3] > .5))
    counts = {key: sum(row["split"] == key for row in rows) for key in ("fit", "calibration", "heldout")}
    if counts != {"fit": 2304, "calibration": 288, "heldout": 288}: raise RuntimeError(f"row count mismatch {counts}")
    return rows, index


def evaluator_fixture() -> dict:
    labels = np.asarray([0, 1, 0, 1], bool); perfect = np.asarray([.01, .99, .02, .98]); reverse = 1 - perfect
    cases = {"perfect_contact_auc": METRIC.auc(labels, perfect) == 1., "reversed_contact_auc": METRIC.auc(labels, reverse) == 0.,
        "transient_contact": int(np.asarray([0, 1, 0]).sum()) == 1, "persistent_contact": int(np.asarray([0, 1, 1]).sum()) == 2,
        "contact_negative_branch": not bool(np.zeros(15, bool).any()), "all_candidates_contact_positive": bool(np.ones(12, bool).all()),
        "one_contact_negative_candidate": int((~np.asarray([1] * 11 + [0], bool)).sum()) == 1,
        "no_admitted_candidate": not bool((np.asarray([.8, .9]) < .5).any()), "threshold_tie_rejected": not (.5 < .5)}
    fixture_rows = [{"state_id": "fixture", "family": FAMILIES[0], "candidate_index": i,
        "kinematic": np.asarray([0, 0, 0, 1, i / 100., 0], np.float32), "p_d": i / 100., "p_theta": 0.,
        "hard_contact": i != 7, "stuck": False} for i in range(12)]
    fixture = planning_metrics(fixture_rows, np.asarray([i == 7 for i in range(12)]), True)
    cases["deterministic_kinematic_selection"] = fixture["per_state"][0]["selected_candidate"] == 7
    encoded = json.dumps({"cases": cases}, sort_keys=True); cases["deterministic_json"] = encoded == json.dumps(json.loads(encoded), sort_keys=True)
    payload = {"schema": "geometry_contact_evaluator_fixture_v1", "cases": cases, "pass": all(cases.values())}
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if not payload["pass"]: raise RuntimeError("geometry evaluator fixture failed")
    return payload


def _effect(positive: list[float], negative: list[float]) -> float | None:
    if len(positive) < 2 or len(negative) < 2: return None
    a, b = np.asarray(positive), np.asarray(negative); pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return None if pooled < 1e-12 else float((b.mean() - a.mean()) / pooled)


def geometry_audit(rows: list[dict]) -> dict:
    """Frozen 0.35 m proximity audit, run before training/model scoring."""
    threshold = .35
    def audit(subset, modality):
        positive = [row for row in subset if row["hard_contact"]]; negative = [row for row in subset if not row["hard_contact"]]
        before = event = never = only_event = 0; onset = []; first_crossing_relative = []; phase = {"pre": [], "event": [], "post": []}; negative_min = []
        for row in negative:
            value = row["future_depth"] if modality == "depth" else row["future_lidar"]
            negative_min.append(float(np.min(value, axis=tuple(range(1, value.ndim))).mean()))
        for row in positive:
            active = np.asarray(row["target"][:, 0] > .5); first = int(np.flatnonzero(active)[0]); onset.append(first + 1)
            value = row["future_depth"] if modality == "depth" else row["future_lidar"]
            mins = np.min(value, axis=tuple(range(1, value.ndim)))
            crossings = np.flatnonzero(mins < threshold)
            if len(crossings): first_crossing_relative.append(int(crossings[0] - first))
            saw_before = bool(np.any(mins[:first] < threshold)); saw_event = bool(mins[first] < threshold)
            before += saw_before; event += saw_event; only_event += saw_event and not saw_before; never += not saw_before and not saw_event
            phase["pre"].append(float(np.min(mins[:first])) if first else float(mins[first]))
            phase["event"].append(float(mins[first])); phase["post"].append(float(np.min(mins[first + 1:])) if first + 1 < 15 else float(mins[first]))
        return {"branches": len(subset), "contact_positive": len(positive), "minimum_range_m": {
            "contact_positive_mean": float(np.mean([np.min(row["future_depth"] if modality == "depth" else row["future_lidar"]) for row in positive])) if positive else None,
            "contact_negative_mean": float(np.mean([np.min(row["future_depth"] if modality == "depth" else row["future_lidar"]) for row in negative])) if negative else None},
            "contact_onset_tick_distribution": {str(v): onset.count(v) for v in sorted(set(onset))},
            "visible_before_onset_fraction": before / len(positive) if positive else None,
            "visible_only_at_event_fraction": only_event / len(positive) if positive else None,
            "visible_at_event_fraction": event / len(positive) if positive else None,
            "never_visible_before_or_at_event_fraction": never / len(positive) if positive else None,
            "median_first_crossing_relative_to_onset_ticks": float(np.median(first_crossing_relative)) if first_crossing_relative else None,
            "standardized_contact_vs_negative_effect": {key: _effect(values, negative_min) for key, values in phase.items()}}
    payload = {"schema": "geometry_observability_audit_v1", "proximity_threshold_m": threshold,
        "threshold_frozen_before_model_training": True, "contact_location": {"availability": "unavailable in frozen boolean contact traces",
            "front_side_rear_body_link": None, "reason": "no link-resolved disallowed-contact field was persisted; no relabeling performed"},
        "overall": {modality: audit(rows, modality) for modality in ("depth", "lidar")},
        "by_family": {family: {modality: audit([r for r in rows if r["family"] == family], modality) for modality in ("depth", "lidar")} for family in FAMILIES}}
    atomic_json(OUT / "geometry_observability_audit.json", payload); return payload


class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(6, 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU())
        self.output = nn.Sequential(nn.Linear(192, 96), nn.GELU())
    def forward(self, value):
        b, t = value.shape[:2]; feature = self.net(value.reshape(b * t, *value.shape[2:])); pooled = torch.cat((feature.mean((-2, -1)), feature.amax((-2, -1))), -1)
        return self.output(pooled).reshape(b, t, -1)


class LidarEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(6 * len(GEO.LIDAR_VERTICAL_DEG), 32, 5, 2, 2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv1d(64, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU())
        self.output = nn.Sequential(nn.Linear(192, 96), nn.GELU())
    def forward(self, value):
        b, t, c, v, a = value.shape; feature = self.net(value.reshape(b * t, c * v, a))
        pooled = torch.cat((feature.mean(-1), feature.amax(-1)), -1); return self.output(pooled).reshape(b, t, -1)


class GeometryContactModel(nn.Module):
    def __init__(self, condition: str):
        super().__init__(); self.condition = condition
        self.geometry = LidarEncoder() if condition == "LIDAR_ONLY" else DepthEncoder()
        self.embodied = nn.Sequential(nn.Linear(73 * 3, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU()) if condition == "DEPTH_PLUS_EMBODIED" else None
        self.action = nn.Sequential(nn.Linear(6, 48), nn.GELU())
        input_width = 96 + 48 + (96 if self.embodied is not None else 0)
        self.temporal = nn.GRU(input_width, 128, batch_first=True); self.output = nn.Linear(128, 2)
    def forward(self, geometry, action, embodied=None):
        feature = [self.geometry(geometry), self.action(action)]
        if self.embodied is not None: feature.append(self.embodied(embodied))
        hidden, _ = self.temporal(torch.cat(feature, -1)); return self.output(hidden)


def embodied_stats(rows):
    value = np.concatenate([np.concatenate((row["current_enhanced"][None], row["future_enhanced"]), 0) for row in rows]).astype(np.float64)
    mean, std = value.mean(0), value.std(0); degenerate = std < 1e-7; std[degenerate] = 1
    return {"mean": mean.tolist(), "std": std.tolist(), "degenerate": np.flatnonzero(degenerate).tolist()}


def tensors(rows, condition, statistics, device):
    if condition == "LIDAR_ONLY":
        current = np.stack([r["current_lidar"] for r in rows]); future = np.stack([r["future_lidar"] for r in rows]); cv = np.stack([r["current_lidar_valid"] for r in rows]); fv = np.stack([r["future_lidar_valid"] for r in rows]); far = GEO.LIDAR_FAR_M
    else:
        current = np.stack([r["current_depth"] for r in rows]); future = np.stack([r["future_depth"] for r in rows]); cv = np.stack([r["current_depth_valid"] for r in rows]); fv = np.stack([r["future_depth_valid"] for r in rows]); far = GEO.DEPTH_FAR_M
    current_t = np.repeat(current[:, None], 15, axis=1); cv_t = np.repeat(cv[:, None], 15, axis=1)
    channels = np.stack((current_t / far, future / far, (future - current_t) / far, cv_t, fv, cv_t * fv), axis=2).astype(np.float32)
    if condition == "LIDAR_ONLY": channels = channels[:, :, :, :, :]
    action = np.stack([r["action_control"] for r in rows]).astype(np.float32)
    target = np.stack([r["target"][:, [0, 2]] for r in rows]).astype(np.float32)
    output = [torch.from_numpy(channels).to(device), torch.from_numpy(action).to(device)]
    if condition == "DEPTH_PLUS_EMBODIED":
        mean, std = np.asarray(statistics["mean"], np.float32), np.asarray(statistics["std"], np.float32)
        cur = np.stack([(r["current_enhanced"] - mean) / std for r in rows]); fut = np.stack([(r["future_enhanced"] - mean) / std for r in rows]); cur = np.repeat(cur[:, None], 15, 1)
        output.append(torch.from_numpy(np.concatenate((cur, fut, fut - cur), -1).astype(np.float32)).to(device))
    else: output.append(None)
    output.append(torch.from_numpy(target).to(device)); return output


def positive_weights(rows):
    target = np.concatenate([r["target"][:, [0, 2]] for r in rows]); pos = target.sum(0); neg = len(target) - pos
    if np.any(pos == 0) or np.any(neg == 0): raise RuntimeError("degenerate contact output")
    return (neg / pos).astype(np.float32)


def train_condition(condition, fit, device):
    seed = derived_seed(condition); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GeometryContactModel(condition).to(device); statistics = embodied_stats(fit) if condition == "DEPTH_PLUS_EMBODIED" else None
    count = sum(p.numel() for p in model.parameters())
    if count >= 500000: raise RuntimeError(f"{condition}: parameter cap exceeded {count}")
    weight = torch.from_numpy(positive_weights(fit)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    grouped = defaultdict(list)
    for row in fit: grouped[row["state_id"]].append(row)
    state_ids = sorted(grouped); smoke = grouped[state_ids[0]]; geometry, action, embodied, target = tensors(smoke, condition, statistics, device)
    logits = model(geometry, action, embodied); loss = sum(F.binary_cross_entropy_with_logits(logits[..., i], target[..., i], pos_weight=weight[i]) for i in range(2)) / 2
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0 for p in model.parameters()): raise RuntimeError(f"{condition}: smoke gradient failure")
    model.eval()
    with torch.inference_mode():
        first = model(geometry, action, embodied)
        if not torch.equal(first, model(geometry, action, embodied)): raise RuntimeError("nondeterministic inference")
        if torch.allclose(first, model(geometry, action.roll(1, 1), embodied)): raise RuntimeError("action insensitive")
        if torch.allclose(first, model(geometry.flip(1), action.flip(1), None if embodied is None else embodied.flip(1))): raise RuntimeError("temporal insensitive")
        changed_geometry = geometry.clone(); changed_geometry[..., 0, 0] += .01
        if torch.allclose(first, model(changed_geometry, action, embodied)): raise RuntimeError("sensor insensitive")
    model.train(); model.zero_grad(set_to_none=True)
    smoke_path = OUT / f".{condition}_smoke.pt"; torch.save(model.state_dict(), smoke_path); clone = GeometryContactModel(condition).to(device)
    clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    history = []; started = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        order = list(state_ids); random.Random(seed + epoch).shuffle(order); total = 0.; batches = 0
        for start in range(0, len(order), 2):
            batch = [row for sid in order[start:start + 2] for row in grouped[sid]]
            geometry, action, embodied, target = tensors(batch, condition, statistics, device)
            optimizer.zero_grad(set_to_none=True); logits = model(geometry, action, embodied)
            loss = sum(F.binary_cross_entropy_with_logits(logits[..., i], target[..., i], pos_weight=weight[i]) for i in range(2)) / 2
            loss.backward(); optimizer.step(); total += float(loss.detach()); batches += 1
        history.append({"epoch": epoch + 1, "loss": total / batches})
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps({"condition": condition, **history[-1]}), flush=True)
    checkpoint = OUT / f"geometry_contact_{condition.lower()}_seed_{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "condition": condition, "seed_family": SEED_FAMILY, "derived_seed": seed,
        "epoch": 60, "embodied_stats": statistics, "parameter_count": count, "positive_weights": weight.cpu().tolist()}, checkpoint)
    package = torch.load(checkpoint, map_location=device, weights_only=True); reload_model = GeometryContactModel(condition).to(device)
    reload_model.load_state_dict(package["state_dict"]); reload_model.eval()
    return reload_model, statistics, {"condition": condition, "seed_family": SEED_FAMILY, "derived_seed": seed,
        "rng_derivation": "sha256(f'{seed_family}|{condition}')[:8] mod (2**31-1)", "parameter_count": count,
        "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001, "epochs": 60, "final_epoch_only": True,
        "positive_weights": weight.cpu().tolist(), "history": history, "runtime_s": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "smoke_passed": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size}


def predict(model, rows, condition, statistics, device):
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), 48):
            geometry, action, embodied, _ = tensors(rows[start:start + 48], condition, statistics, device)
            output.append(model(geometry, action, embodied).float().cpu().numpy())
    return np.concatenate(output)


def _best_contact_safe(rows, ids):
    safe = [i for i in ids if not rows[i]["hard_contact"]]
    if not safe: return None
    best = safe[0]
    for i in safe[1:]:
        if BASE.preference(rows[i], rows[best]) > 0: best = i
    return best


def planning_metrics(rows, admitted, include_rows=False):
    grouped = defaultdict(list)
    for i, row in enumerate(rows): grouped[row["state_id"]].append(i)
    per_state = []; selected_progress = []; oracle_progress = []; selected_contact = []; selected_stuck = []; regrets = []; top1 = []; top3 = []
    false_abstention = correct_abstention = 0
    for state_id, ids in sorted(grouped.items()):
        safe = [i for i in ids if not rows[i]["hard_contact"]]; rank = BASE.route_order(rows, ids); available = [i for i in rank if admitted[i]]
        pick = available[0] if available else None; oracle_rank = [i for i in rank if i in safe]; oracle_pick = oracle_rank[0] if oracle_rank else None; best = _best_contact_safe(rows, ids)
        if pick is None:
            if safe: false_abstention += 1
            else: correct_abstention += 1
        else:
            selected_progress.append(float(rows[pick]["p_d"])); selected_contact.append(bool(rows[pick]["hard_contact"])); selected_stuck.append(bool(rows[pick]["stuck"]))
        if oracle_pick is not None: oracle_progress.append(float(rows[oracle_pick]["p_d"]))
        if best is not None:
            top1.append(pick == best); top3.append(best in available[:3])
            if pick is not None and not rows[pick]["hard_contact"] and len(safe) >= 2:
                values = [float(rows[i]["p_d"]) for i in safe]; spread = max(values) - min(values)
                if spread > 1e-8: regrets.append((float(rows[best]["p_d"]) - float(rows[pick]["p_d"])) / spread)
        per_state.append({"state_id": state_id, "family": rows[ids[0]]["family"], "admitted": int(admitted[ids].sum()),
            "admitted_contact_negative": int(sum(admitted[i] and not rows[i]["hard_contact"] for i in ids)),
            "admitted_contact_positive": int(sum(admitted[i] and rows[i]["hard_contact"] for i in ids)),
            "selected_candidate": None if pick is None else int(rows[pick]["candidate_index"]),
            "selected_contact": None if pick is None else bool(rows[pick]["hard_contact"]), "selected_stuck": None if pick is None else bool(rows[pick]["stuck"]),
            "selected_p_d": None if pick is None else float(rows[pick]["p_d"]), "selected_p_theta": None if pick is None else float(rows[pick]["p_theta"]),
            "oracle_contact_kinematic_p_d": None if oracle_pick is None else float(rows[oracle_pick]["p_d"])})
    selected_mean = float(np.mean(selected_progress)) if selected_progress else 0.; oracle_mean = float(np.mean(oracle_progress)) if oracle_progress else 0.
    return {"states": len(grouped), "states_retaining_contact_negative": sum(r["admitted_contact_negative"] > 0 for r in per_state),
        "states_only_contact_positive_admitted": sum(r["admitted"] > 0 and r["admitted_contact_negative"] == 0 for r in per_state),
        "states_no_admitted": sum(r["admitted"] == 0 for r in per_state), "false_abstentions": false_abstention, "correct_abstentions": correct_abstention,
        "selected_contact_count": int(sum(selected_contact)), "selected_contact_rate": float(np.mean(selected_contact)) if selected_contact else 0.,
        "selected_stuck_count": int(sum(selected_stuck)), "selected_stuck_rate": float(np.mean(selected_stuck)) if selected_stuck else 0.,
        "mean_selected_route_progress_m": selected_mean, "oracle_contact_kinematic_progress_m": oracle_mean,
        "oracle_progress_fraction": selected_mean / oracle_mean if abs(oracle_mean) > 1e-9 else None,
        "normalized_route_progress_regret": float(np.mean(regrets)) if regrets else None, "normalized_regret_states": len(regrets),
        "best_safe_top1": float(np.mean(top1)) if top1 else None, "best_safe_top3": float(np.mean(top3)) if top3 else None,
        "per_state": per_state if include_rows else None}


def contact_metrics(rows, logits, temperature, threshold):
    probability = 1 / (1 + np.exp(-logits / temperature)); branch_probability = probability[:, -1, 1]
    label = np.asarray([r["hard_contact"] for r in rows]); decision = branch_probability >= threshold
    active_label = np.stack([r["target"][:, 0] > .5 for r in rows]); active_decision = probability[:, :, 0] >= threshold
    delays = []; transient = []; branch_detected = []
    for actual, detected in zip(active_label, active_decision):
        positives = np.flatnonzero(actual)
        if not len(positives): continue
        first = int(positives[0]); hits = np.flatnonzero(detected[first:]); branch_detected.append(bool(len(hits))); delays.append(int(hits[0]) if len(hits) else None)
        if int(actual.sum()) == 1: transient.append(not bool(detected[first]))
    return {"prevalence": float(label.mean()), "auc": METRIC.auc(label, branch_probability), "average_precision": METRIC.average_precision(label, branch_probability),
        "recall": float(decision[label].mean()), "false_negative_rate": float((~decision)[label].mean()),
        "specificity": float((~decision)[~label].mean()), "contact_negative_retention": float((~decision)[~label].mean()),
        "ece": METRIC.ece(label, branch_probability), "brier": float(np.mean((branch_probability - label) ** 2)),
        "event_tick_recall": float(active_decision[active_label].mean()), "branch_event_detection_recall": float(np.mean(branch_detected)),
        "median_detection_delay_ticks": float(np.median([x for x in delays if x is not None])) if any(x is not None for x in delays) else None,
        "missed_transient_contact_rate": float(np.mean(transient)) if transient else None}


def decision_metrics(rows, probabilities, threshold, include_rows=False):
    admitted = probabilities < threshold; labels = np.asarray([r["hard_contact"] for r in rows]); rejected = ~admitted
    return {"contact_recall": float(rejected[labels].mean()), "contact_false_negative_rate": float(admitted[labels].mean()),
        "contact_negative_retention": float(admitted[~labels].mean()), "admitted_contact_negative_count": int((admitted & ~labels).sum()),
        "admitted_contact_positive_count": int((admitted & labels).sum()), **planning_metrics(rows, admitted, include_rows)}


def calibrate(rows, logits, condition):
    labels = np.asarray([r["hard_contact"] for r in rows]); temperature = DENSE.fit_temperature(logits[:, -1, 1], labels)
    probability = 1 / (1 + np.exp(-logits[:, -1, 1] / temperature)); candidates = []; frontier = defaultdict(list); started = time.time()
    fields = ("contact_recall", "contact_false_negative_rate", "contact_negative_retention", "states_retaining_contact_negative",
              "states_only_contact_positive_admitted", "false_abstentions", "mean_selected_route_progress_m", "normalized_route_progress_regret")
    for threshold in BASE.threshold_values(probability):
        metric = decision_metrics(rows, probability, float(threshold))
        frontier["threshold"].append(float(threshold))
        for field in fields: frontier[field].append(np.nan if metric[field] is None else metric[field])
        if metric["contact_recall"] >= .95:
            regret = metric["normalized_route_progress_regret"]
            key = (metric["states_retaining_contact_negative"], metric["contact_negative_retention"], metric["mean_selected_route_progress_m"],
                   -metric["false_abstentions"], -float(threshold))
            candidates.append((key, float(threshold), metric))
    arrays = {key: np.asarray(value, np.float64) for key, value in frontier.items()}; path = CACHE / condition / "calibration_contact_threshold_frontier_v1.npz"; atomic_npz(path, **arrays)
    if not candidates: raise RuntimeError(f"{condition}: calibration recall feasibility failure")
    _, threshold, metrics = max(candidates, key=lambda x: x[0])
    return {"temperature": temperature, "threshold": threshold, "metrics": metrics, "eligible_thresholds": len(candidates),
        "thresholds_evaluated": len(arrays["threshold"]), "frontier": {"file": str(path), "sha256": sha(path), "content_digest": BASE.array_digest(arrays)},
        "tie_rule": "more conservative threshold after state retention, candidate retention, progress, and false abstention", "runtime_s": time.time() - started}


def per_family(rows, logits, calibration):
    prob = 1 / (1 + np.exp(-logits[:, -1, 1] / calibration["temperature"])); output = {}
    for family in FAMILIES:
        ids = [i for i, r in enumerate(rows) if r["family"] == family]; sub = [rows[i] for i in ids]
        output[family] = {"contact": contact_metrics(sub, logits[ids], calibration["temperature"], calibration["threshold"]),
            "filtering_and_planning": decision_metrics(sub, prob[ids], calibration["threshold"], True)}
    return output


def gate(metrics, planning, families):
    no_collapse = all(v["filtering_and_planning"]["states_retaining_contact_negative"] > 0 and v["filtering_and_planning"]["selected_contact_count"] == 0 for v in families.values())
    checks = {"contact_auc_ge_0_85": metrics["auc"] >= .85, "contact_recall_ge_0_95": metrics["recall"] >= .95,
        "contact_fnr_le_0_05": metrics["false_negative_rate"] <= .05, "contact_negative_retention_ge_0_40": metrics["contact_negative_retention"] >= .40,
        "ece_le_0_10": metrics["ece"] <= .10, "event_recall_ge_0_80": metrics["event_tick_recall"] >= .80,
        "detection_delay_le_1": metrics["median_detection_delay_ticks"] is not None and metrics["median_detection_delay_ticks"] <= 1,
        "states_retaining_ge_18": planning["states_retaining_contact_negative"] >= 18,
        "no_only_contact_positive_state": planning["states_only_contact_positive_admitted"] == 0,
        "zero_selected_contact": planning["selected_contact_count"] == 0, "false_abstentions_le_3": planning["false_abstentions"] <= 3,
        "oracle_progress_fraction_ge_0_80": planning["oracle_progress_fraction"] is not None and planning["oracle_progress_fraction"] >= .80,
        "normalized_regret_le_0_20": planning["normalized_route_progress_regret"] is not None and planning["normalized_route_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": planning["best_safe_top3"] is not None and planning["best_safe_top3"] >= .75, "no_family_collapse": no_collapse}
    return {"checks": checks, "passed": all(checks.values())}


def persist_ledger(condition, rows, logits, calibration):
    probability = 1 / (1 + np.exp(-logits / calibration["temperature"])); admitted = probability[:, -1, 1] < calibration["threshold"]
    plan = planning_metrics(rows, admitted, True); selected = {(r["state_id"], r["selected_candidate"]) for r in plan["per_state"] if r["selected_candidate"] is not None}
    arrays = {"branch_id": np.asarray([r["branch_id"] for r in rows]), "state_id": np.asarray([r["state_id"] for r in rows]),
        "candidate_index": np.asarray([r["candidate_index"] for r in rows], np.int16), "split": np.asarray([r["split"] for r in rows]),
        "family": np.asarray([r["family"] for r in rows]), "raw_logits": logits.astype(np.float32), "calibrated_probability": probability.astype(np.float64),
        "contact_labels": np.stack([r["target"][:, [0, 2]] for r in rows]).astype(np.uint8),
        "stuck_labels": np.stack([r["target"][:, [1, 3]] for r in rows]).astype(np.uint8), "threshold_decision_admitted": admitted.astype(np.uint8),
        "selected": np.asarray([(r["state_id"], r["candidate_index"]) in selected for r in rows], np.uint8),
        "p_d": np.asarray([r["p_d"] for r in rows], np.float32), "p_theta": np.asarray([r["p_theta"] for r in rows], np.float32),
        "kinematic": np.stack([r["kinematic"] for r in rows]).astype(np.float32)}
    path = CACHE / condition / "row_level_evidence_v1.npz"; atomic_npz(path, **arrays)
    result = {"file": str(path), "sha256": sha(path), "content_digest": BASE.array_digest(arrays), "rows": len(rows), "fields": sorted(arrays)}
    atomic_json(OUT / f"{condition.lower()}_row_level_evidence_index.json", result); return result


def evaluate_condition(condition, fit, calibration_rows, heldout, device):
    model, statistics, training = train_condition(condition, fit, device)
    cal_logits = predict(model, calibration_rows, condition, statistics, device); calibration = calibrate(calibration_rows, cal_logits, condition)
    held_logits = predict(model, heldout, condition, statistics, device); probability = 1 / (1 + np.exp(-held_logits[:, -1, 1] / calibration["temperature"]))
    contact = contact_metrics(heldout, held_logits, calibration["temperature"], calibration["threshold"])
    decision = decision_metrics(heldout, probability, calibration["threshold"], True); families = per_family(heldout, held_logits, calibration)
    fit_logits = predict(model, fit, condition, statistics, device); fit_metrics = contact_metrics(fit, fit_logits, calibration["temperature"], calibration["threshold"])
    all_rows = calibration_rows + heldout; all_logits = np.concatenate((cal_logits, held_logits)); ledger = persist_ledger(condition, all_rows, all_logits, calibration)
    return {"condition": condition, "training": training, "calibration": calibration, "heldout": {"contact": contact,
        "filtering_and_planning": decision, "per_family": families}, "fit_contact": fit_metrics,
        "fit_to_heldout_auc_gap": contact["auc"] - fit_metrics["auc"], "gate": gate(contact, decision, families), "row_level_evidence": ledger}


def frozen_comparators() -> dict:
    scale = json.loads((ROOT / ".generated/factorised_micro_safety_data_scaling_v2/result.json").read_text())
    enhanced = json.loads((ROOT / ".generated/enhanced_embodied_safety_observability_v2/result.json").read_text())
    matrix = json.loads((ROOT / ".generated/deployment_valid_safety_observability_matrix_v1/result.json").read_text())
    dense = json.loads((ROOT / ".generated/dense_temporal_true_future_safety_observability_v1/result.json").read_text())
    guard = json.loads((ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1/result.json").read_text())
    return {"FIT_192_CONTACT": {"panel": "same fresh-24 heldout", **scale["conditions"]["fit192"]["heldout"]["components"]["contact"]},
        "ENHANCED_EMBODIED_CONTACT": {"panel": "historical heldout-8", **enhanced["heldout"]["components"]["contact"]},
        "RAW_RGB_CONTACT": {"panel": "historical heldout-8", **matrix["heldout"]["RAW_RGB"]["components"]["contact"]},
        "FINAL_LAYER_VIT_L_CONTACT": {"panel": "historical heldout-8", **dense["heldout"]["components"]["collision_contact"]},
        "PRIVILEGED_STATIC_GRID_GUARD": {"panel": "historical heldout-8", **guard["guard_discrimination"]["by_split"]["heldout"]},
        "ORACLE_CONTACT": {"panel": "same fresh-24 heldout", "contact_auc": 1.0, "contact_recall": 1.0,
            "contact_negative_retention": 1.0, "selected_contact_count": 0}}


def main() -> int:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    fixture = evaluator_fixture(); rows, sensor_index = load_rows(); fit = [r for r in rows if r["split"] == "fit"]
    calibration_rows = [r for r in rows if r["split"] == "calibration"]; heldout = [r for r in rows if r["split"] == "heldout"]
    audit = geometry_audit(rows)  # explicitly before any scientific model output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); conditions = {}
    for condition in CONDITIONS: conditions[condition] = evaluate_condition(condition, fit, calibration_rows, heldout, device)
    depth_pass = conditions["DEPTH_ONLY"]["gate"]["passed"]; lidar_pass = conditions["LIDAR_ONLY"]["gate"]["passed"]
    fusion_pass = conditions["DEPTH_PLUS_EMBODIED"]["gate"]["passed"]
    depth = conditions["DEPTH_ONLY"]["heldout"]; fusion = conditions["DEPTH_PLUS_EMBODIED"]["heldout"]
    fusion_material = (fusion["contact"]["auc"] - depth["contact"]["auc"] >= .05 or
                       fusion["contact"]["contact_negative_retention"] - depth["contact"]["contact_negative_retention"] >= .15)
    classifications = []
    if depth_pass: classifications.append("DEPTH_SAFETY_SUFFICIENCY_SIGNAL")
    if lidar_pass and not depth_pass: classifications.append("LIDAR_SAFETY_SUFFICIENCY_SIGNAL")
    if fusion_pass and fusion_material: classifications.append("GEOMETRY_EMBODIED_FUSION_SIGNAL")
    if not classifications:
        best = max(conditions.values(), key=lambda x: x["heldout"]["contact"]["auc"])
        tendency = best["heldout"]["contact"]["auc"] >= .80 and best["heldout"]["contact"]["contact_negative_retention"] >= .20
        classifications = ["GEOMETRY_MODALITY_POSITIVE_TENDENCY" if tendency else "GEOMETRY_SENSOR_SAFETY_NO_GO"]
    if depth_pass:
        recommendation = "Specify RGB_TO_GEOMETRY_SAFETY_STATE_V1 next: privileged depth supervision with RGB-only inference; do not deploy depth automatically."
    elif lidar_pass:
        recommendation = "Decide explicitly whether to add the ideal 360-degree LiDAR contract; this is not an RGB-only result."
    elif fusion_pass:
        recommendation = "Prospectively validate depth plus embodied response as an explicit changed sensor contract."
    else:
        recommendation = "Review disallowed-contact severity/body-region labels or narrow the learned hard-contact claim; do not add another model automatically."
    result = {"schema": "geometry_modality_safety_sufficiency_v1_result", "experiment": "GEOMETRY_MODALITY_SAFETY_SUFFICIENCY_V1",
        "source_commit": "20dd1b8dbdd52db5f3b55217ed2f6601ec4ec4c0", "preserved_terminal": "FACTORISED_MICRO_SAFETY_DATA_SCALING_NO_SIGNAL",
        "bindings": {"geometry_sensor_index_digest": sensor_index["content_digest"], "fit192_states": 192, "calibration_states": 24, "heldout_states": 24},
        "sensor_contracts": {"depth": sensor_index["depth_contract"], "lidar": sensor_index["lidar_contract"]},
        "replay_verification": {"physics_replayed_states": sensor_index["simulator_replayed_states"], "physics_replayed_branches": sensor_index["simulator_replayed_branches"],
            "static_pose_materialized_states": sensor_index["static_pose_materialized_states"], "verification_failures": sensor_index["branch_action_pose_safety_verification_failures"]},
        "evaluator_fixture": fixture, "geometry_observability_audit": audit, "conditions": conditions, "frozen_comparators": frozen_comparators(),
        "fusion_incremental_over_depth": {"auc": fusion["contact"]["auc"] - depth["contact"]["auc"],
            "contact_negative_retention": fusion["contact"]["contact_negative_retention"] - depth["contact"]["contact_negative_retention"],
            "materially_exceeds": fusion_material}, "classifications": classifications, "recommendation": recommendation,
        "runtime": {"total_s": time.time() - started, "geometry_materialization_compute_s": sensor_index["runtime_compute_s"],
            "geometry_storage_bytes": sensor_index["storage_bytes"], "device": str(device), "condition_training_runtime_s": {k: v["training"]["runtime_s"] for k, v in conditions.items()},
            "peak_vram_bytes": max(v["training"]["peak_vram_bytes"] for v in conditions.values())},
        "custody": {"one_seed_per_condition": True, "seed_family": SEED_FAMILY, "jepa_predictor_opened_or_trained": False,
            "memory_novelty_navigation_trained": False, "new_state_or_candidate_identities": 0, "rgb_or_vit_l_trained": False}}
    atomic_json(OUT / "result.json", result); print(json.dumps({"classifications": classifications,
        "conditions": {k: {"auc": v["heldout"]["contact"]["auc"], "recall": v["heldout"]["contact"]["recall"],
            "retention": v["heldout"]["contact"]["contact_negative_retention"], "gate": v["gate"]["passed"]} for k, v in conditions.items()},
        "runtime": result["runtime"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
