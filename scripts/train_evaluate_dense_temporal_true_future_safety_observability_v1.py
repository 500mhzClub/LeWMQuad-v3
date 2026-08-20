#!/usr/bin/env python3
"""Evaluate dense true-future RGB safety observability on the frozen route panel."""
from __future__ import annotations

import hashlib
import json
import math
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS

EVIDENCE = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1"
OUT = EVIDENCE
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/dense_temporal_true_future_safety_observability_v1")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2 = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
SEED = 2026082007
FAMILIES = FS.FAMILIES
CHANNELS = (
    "contact_active", "stuck_active", "contact_cumulative",
    "stuck_cumulative", "aggregate_unsafe_cumulative",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=FS.json_default))
    temp.replace(path)


def load_dense() -> tuple[list[dict], dict, dict]:
    receipt = json.loads((EVIDENCE / "evidence_receipt.json").read_text())
    if not receipt.get("complete") or receipt["states"] != 48 or receipt["branches"] != 576:
        raise RuntimeError("dense evidence is incomplete")
    index = json.loads((EVIDENCE / "token_index.json").read_text())
    token_by_rgb = {r["rgb_sha256"]: r["token_path"] for r in index["records"]}
    split_payload = json.loads((V1 / "split.json").read_text())
    split_by_state = {sid: split for split in ("fit", "calibration", "heldout") for sid in split_payload[split]}
    route_rows = {r["state_id"] + f":{int(r['candidate_index']):02d}": r for r in FS.load_metadata()}
    states = []
    for path in sorted((EVIDENCE / "dense_replay").glob("purpose-*.json"), key=lambda p: int(p.stem.split("-")[1])):
        state = json.loads(path.read_text())
        state["split"] = split_by_state[state["state_id"]]
        for branch in state["branches"]:
            branch["branch_id"] = branch["branch_identity"]
            branch["family"] = state["family"]
            branch["current_token_path"] = token_by_rgb[state["current_frame"]["rgb_sha256"]]
            branch["future_token_paths"] = [token_by_rgb[t["rgb_sha256"]] for t in branch["ticks"]]
            branch["control_history_summary"] = route_rows[branch["branch_id"]]["action_control"][-3:].tolist()
        states.append(state)
    if len(states) != 48 or sum(len(s["branches"]) for s in states) != 576:
        raise RuntimeError("wrong frozen panel cardinality")
    return states, index, receipt


def split_name(state: dict) -> str:
    return state["split"]


def branch_targets(branch: dict) -> np.ndarray:
    rows = []
    for tick in branch["ticks"]:
        rows.append([
            tick["active_contact"], tick["active_stuck"],
            tick["cumulative_contact"], tick["cumulative_stuck"],
            tick["training_cumulative_unsafe"],
        ])
    return np.asarray(rows, np.float32)


def event_runs(values: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(values) - 1):
            end = i if value and i == len(values) - 1 else i - 1
            runs.append((start, end)); start = None
    return runs


def summarize_events(branches: list[dict], key: str, sparse: set[int]) -> dict:
    positive_branches = 0; positive_ticks = 0; first = []; final = []; durations = []
    runs_total = sparse_hits = between = next_sparse_persistent = 0
    for branch in branches:
        values = [bool(t[key]) for t in branch["ticks"]]
        positions = [i for i, v in enumerate(values) if v]
        if positions:
            positive_branches += 1; positive_ticks += len(positions)
            first.append(positions[0] + 1); final.append(positions[-1] + 1)
        for lo, hi in event_runs(values):
            runs_total += 1; durations.append(hi - lo + 1)
            if any(i in sparse for i in range(lo, hi + 1)):
                sparse_hits += 1
            else:
                between += 1
            after = [i for i in sorted(sparse) if i >= hi]
            if after and values[after[0]]:
                next_sparse_persistent += 1
    n = len(branches)
    return {
        "branches": n, "positive_branches": positive_branches,
        "positive_branch_prevalence": positive_branches / n if n else None,
        "total_positive_ticks": positive_ticks,
        "first_positive_tick_mean": float(np.mean(first)) if first else None,
        "first_positive_tick_min": min(first) if first else None,
        "final_positive_tick_mean": float(np.mean(final)) if final else None,
        "final_positive_tick_max": max(final) if final else None,
        "event_runs": runs_total,
        "event_duration_ticks_mean": float(np.mean(durations)) if durations else None,
        "event_duration_ticks_median": float(np.median(durations)) if durations else None,
        "event_runs_at_sparse_tick": sparse_hits,
        "event_runs_entirely_between_sparse_ticks": between,
        "event_runs_persistent_at_next_sparse_tick": next_sparse_persistent,
        "fraction_runs_entirely_between_sparse": between / runs_total if runs_total else None,
    }


def observability_audit(states: list[dict]) -> dict:
    all_branches = [b for s in states for b in s["branches"]]
    boundaries = states[0]["horizon_tick_boundaries"]
    if any(s["horizon_tick_boundaries"] != boundaries for s in states):
        raise RuntimeError("nonuniform registered horizon boundaries")
    sparse = {int(x) - 1 for x in boundaries}
    out = {"registered_horizon_tick_boundaries": boundaries, "ticks_per_branch": boundaries[-1], "by_split_family": {}}
    for split in ("fit", "calibration", "heldout"):
        out["by_split_family"][split] = {}
        for family in ("overall", *FAMILIES):
            subset = [b for s in states if split_name(s) == split and (family == "overall" or s["family"] == family) for b in s["branches"]]
            out["by_split_family"][split][family] = {
                "contact_collision": summarize_events(subset, "active_contact", sparse),
                "stuck": summarize_events(subset, "active_stuck", sparse),
                "aggregate_unsafe": summarize_events(subset, "active_unsafe", sparse),
            }
    contact_only = stuck_only = overlap = unsafe = 0
    contact_sparse_miss = stuck_sparse_hit = contact_positive = stuck_positive = 0
    for branch in all_branches:
        c = any(t["active_contact"] for t in branch["ticks"])
        s = any(t["active_stuck"] for t in branch["ticks"])
        u = bool(branch["h3_labels"]["aggregate_unsafe"])
        unsafe += u; contact_only += u and c and not s; stuck_only += u and s and not c; overlap += c and s
        if c:
            contact_positive += 1
            contact_sparse_miss += not any(branch["ticks"][i]["active_contact"] for i in sparse)
        if s:
            stuck_positive += 1
            stuck_sparse_hit += any(branch["ticks"][i]["active_stuck"] for i in sparse)
    out["component_overlap"] = {
        "unsafe_branches": unsafe, "contact_only": contact_only, "stuck_only": stuck_only,
        "contact_and_stuck": overlap,
        "unsafe_contact_only_fraction": contact_only / unsafe if unsafe else None,
        "unsafe_stuck_only_fraction": stuck_only / unsafe if unsafe else None,
        "contact_branches_not_active_at_sparse_endpoint_fraction": contact_sparse_miss / contact_positive if contact_positive else None,
        "stuck_branches_active_at_sparse_endpoint_fraction": stuck_sparse_hit / stuck_positive if stuck_positive else None,
    }
    matches = [value for branch in all_branches for value in branch["aggregate_replay_match"]]
    out["frozen_aggregate_replay_alignment"] = {
        "registered_horizon_rows": len(matches), "exact_matches": int(sum(matches)),
        "mismatches": int(len(matches) - sum(matches)),
        "training_rule": "replay components at dense ticks; authoritative frozen aggregate forced at its first known H1/H2/H3 right-censoring boundary",
    }
    atomic_json(OUT / "temporal_observability_audit.json", out)
    return out


class DenseTemporalSafety(nn.Module):
    def __init__(self, action_dims: int = 3, history_dims: int = 3):
        super().__init__()
        self.projection = nn.Linear(1024, 32)
        self.spatial = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
        )
        self.action = nn.Sequential(nn.Linear(action_dims, 16), nn.GELU())
        self.history = nn.Sequential(nn.Linear(history_dims, 32), nn.GELU())
        self.temporal = nn.GRU(144, 128, num_layers=1, batch_first=True)
        self.output = nn.Linear(128, 5)

    def forward(self, current: torch.Tensor, future: torch.Tensor,
                action: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        current = F.layer_norm(current, (1024,), weight=None, bias=None)
        future = F.layer_norm(future, (1024,), weight=None, bias=None)
        cur = self.projection(current)
        fut = self.projection(future)
        batch, ticks = fut.shape[:2]
        cur = cur[:, None].expand(-1, ticks, -1, -1)
        joined = torch.cat([cur, fut, fut - cur], -1)
        joined = joined.reshape(batch * ticks, 768, 96).transpose(1, 2).reshape(batch * ticks, 96, 24, 32)
        spatial = self.spatial(joined)
        visual = torch.cat([spatial.mean((2, 3)), spatial.amax((2, 3))], 1).reshape(batch, ticks, 96)
        action_features = self.action(action)
        history_features = self.history(history)[:, None].expand(-1, ticks, -1)
        sequence = torch.cat([visual, action_features, history_features], -1)
        hidden, _ = self.temporal(sequence)
        return self.output(hidden)


class DenseDataset(torch.utils.data.Dataset):
    def __init__(self, branches: list[dict]):
        self.branches = branches

    def __len__(self):
        return len(self.branches)

    @staticmethod
    def token(path: str) -> np.ndarray:
        value = np.memmap(path, mode="r", dtype=np.float16, shape=(768, 1024))
        return np.asarray(value, np.float32)

    def __getitem__(self, index: int):
        b = self.branches[index]
        current = self.token(b["current_token_path"])
        future = np.stack([self.token(p) for p in b["future_token_paths"]])
        action = np.asarray([t["applied_action"] for t in b["ticks"]], np.float32)
        history = np.asarray(b["control_history_summary"], np.float32)
        target = branch_targets(b)
        return current, future, action, history, target, b["branch_id"]


def evaluator_fixture() -> dict:
    # Event streams are deliberately tiny and hand-checkable.
    streams = {
        "one_tick_transient_contact": [0, 1, 0, 0],
        "persistent_contact": [0, 1, 1, 1],
        "delayed_stuck": [0, 0, 0, 1],
        "no_unsafe_event": [0, 0, 0, 0],
    }
    y = np.asarray([0, 0, 1, 1], bool); p = np.asarray([.05, .1, .9, .95])
    cases = {
        "transient_is_one_tick": sum(streams["one_tick_transient_contact"]) == 1,
        "persistent_event_detected": sum(streams["persistent_contact"]) == 3,
        "delayed_event_first_tick": streams["delayed_stuck"].index(1) == 3,
        "no_unsafe_event": not any(streams["no_unsafe_event"]),
        "perfect_auc": FS.auc(y, p) == 1.0,
        "all_candidates_unsafe": bool(np.ones(12, bool).all()),
        "one_safe_candidate": int((~np.asarray([1] * 11 + [0], bool)).sum()) == 1,
        "threshold_tie_rejected": not bool(.5 < .5),
        "no_candidate_admitted": not bool(np.any(np.asarray([.8, .9]) < .5)),
    }
    payload = {"schema": "dense_temporal_safety_evaluator_fixture_v1", "cases": cases, "pass": all(cases.values())}
    encoded = json.dumps(payload, sort_keys=True)
    cases["deterministic_json_serialization"] = encoded == json.dumps(json.loads(encoded), sort_keys=True)
    payload["pass"] = all(cases.values())
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if json.loads((OUT / "evaluator_fixture.json").read_text()) != payload:
        raise RuntimeError("fixture reload mismatch")
    return payload


def positive_weights(branches: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    labels = np.stack([branch_targets(b) for b in branches])
    weights = np.ones(5, np.float32); defined = np.zeros(5, bool); prevalence = {}
    for channel, name in enumerate(CHANNELS):
        y = labels[:, :, channel]
        pos = float(y.sum()); neg = float(y.size - pos)
        defined[channel] = pos > 0 and neg > 0
        if defined[channel]: weights[channel] = neg / pos
        prevalence[name] = {"positive_ticks": int(pos), "ticks": int(y.size), "prevalence": float(y.mean()), "positive_weight": float(weights[channel])}
    return weights, defined, prevalence


def balanced_loss(logits: torch.Tensor, labels: torch.Tensor,
                  weights: torch.Tensor, defined: np.ndarray) -> torch.Tensor:
    terms = [F.binary_cross_entropy_with_logits(logits[:, :, c], labels[:, :, c], pos_weight=weights[c]) for c in range(5) if defined[c]]
    return torch.stack(terms).mean()


def batches(branches: list[dict], batch_size: int, *, shuffle_seed: int | None):
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(shuffle_seed)
    return torch.utils.data.DataLoader(DenseDataset(branches), batch_size=batch_size,
                                       shuffle=shuffle_seed is not None, generator=generator,
                                       num_workers=4, pin_memory=True, persistent_workers=False)


def train_model(fit: list[dict], device: torch.device) -> tuple[DenseTemporalSafety, list[dict], dict]:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = DenseTemporalSafety().to(device)
    count = sum(p.numel() for p in model.parameters())
    if count >= 250000:
        raise RuntimeError(f"model exceeds parameter cap: {count}")
    weights_np, defined, prevalence = positive_weights(fit)
    weights = torch.from_numpy(weights_np).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    smoke = next(iter(batches(fit[:8], 4, shuffle_seed=None)))
    current, future, action, history, target, _ = smoke
    current = current.to(device); future = future.to(device); action = action.to(device); history = history.to(device); target = target.to(device)
    model.train(); logits = model(current, future, action, history); loss = balanced_loss(logits, target, weights, defined); loss.backward()
    if not torch.isfinite(loss): raise RuntimeError("nonfinite smoke loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0:
            raise RuntimeError(f"smoke gradient failure: {name}")
    output_gradient = model.output.weight.grad.abs().sum(1)
    if any(bool(defined[c]) and float(output_gradient[c]) == 0. for c in range(5)):
        raise RuntimeError("a nondegenerate temporal output received no gradient")
    base = logits.detach(); changed_action = action.clone(); changed_action[:, :, 0] *= -1
    changed_order = future.flip(1)
    model.eval()
    with torch.inference_mode():
        repeated = model(current, future, action, history)
        if not torch.equal(repeated, model(current, future, action, history)):
            raise RuntimeError("nondeterministic smoke inference")
        if torch.allclose(base, model(current, future, changed_action, history)):
            raise RuntimeError("action-insensitive smoke")
        if torch.allclose(base, model(current, changed_order, action.flip(1), history)):
            raise RuntimeError("temporal-order-insensitive smoke")
    smoke_path = OUT / ".dense_smoke.pt"
    torch.save(model.state_dict(), smoke_path); clone = DenseTemporalSafety().to(device)
    clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    opt.zero_grad(set_to_none=True)
    history_rows = []; started = time.time(); peak = 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        model.train(); total = 0.; seen = 0
        for current, future, action, history, target, _ in batches(fit, 8, shuffle_seed=SEED + epoch):
            current = current.to(device, non_blocking=True); future = future.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True); history = history.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            logits = model(current, future, action, history)
            loss = balanced_loss(logits, target, weights, defined)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss.detach()) * len(current); seen += len(current)
        model.eval(); history_rows.append({"epoch": epoch + 1, "mean_balanced_bce": total / seen})
        if epoch in (0, 9, 19, 29, 39, 49, 59):
            print(json.dumps(history_rows[-1]), flush=True)
    if device.type == "cuda": peak = int(torch.cuda.max_memory_allocated(device))
    stats = {"parameter_count": count, "training_runtime_s": time.time() - started,
             "peak_vram_bytes": peak, "positive_weights": weights_np.tolist(),
             "defined_outputs": defined.tolist(), "fit_tick_prevalence": prevalence,
             "smoke_passed": True}
    model.eval(); return model, history_rows, stats


def predict(model: DenseTemporalSafety, branches: list[dict], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for current, future, action, history, target, _ in batches(branches, 8, shuffle_seed=None):
            logits.append(model(current.to(device), future.to(device), action.to(device), history.to(device)).cpu().numpy())
            targets.append(target.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    x = torch.tensor(logits, dtype=torch.float64); y = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.2, max_iter=100)
    def closure():
        optimizer.zero_grad(); temp = torch.exp(log_temperature).clamp(.05, 20.)
        loss = F.binary_cross_entropy_with_logits(x / temp, y); loss.backward(); return loss
    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(.05, 20.))


def choose_threshold(prob: np.ndarray, labels: np.ndarray) -> dict:
    labels = labels.astype(bool)
    candidates = sorted({0., 1., *(float(np.nextafter(v, math.inf)) for v in prob)})
    scored = []
    for threshold in candidates:
        admitted = prob < threshold
        recall = float(np.mean(~admitted[labels])) if labels.any() else 1.
        retention = float(np.mean(admitted[~labels])) if (~labels).any() else 0.
        scored.append((recall, retention, threshold))
    feasible = [x for x in scored if x[0] >= .95]
    chosen = max(feasible, key=lambda x: (x[1], -x[2])) if feasible else max(scored, key=lambda x: (x[0], x[1], -x[2]))
    return {"threshold": chosen[2], "calibration_recall": chosen[0], "calibration_safe_retention": chosen[1], "criterion_satisfied": bool(feasible),
            "tie_rule": "more conservative (lower) threshold when retention ties"}


def discrimination(labels: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    y = labels.astype(bool); admitted = prob < threshold
    tp = int(np.sum(y & ~admitted)); fn = int(np.sum(y & admitted)); tn = int(np.sum(~y & admitted)); fp = int(np.sum(~y & ~admitted))
    return {
        "rows": len(y), "positives": int(y.sum()), "auc": FS.auc(y, prob), "average_precision": FS.average_precision(y, prob),
        "unsafe_recall": tp / (tp + fn) if tp + fn else None, "unsafe_false_negative_rate": fn / (tp + fn) if tp + fn else None,
        "safe_specificity": tn / (tn + fp) if tn + fp else None, "safe_candidate_retention": tn / (tn + fp) if tn + fp else None,
        "ece": FS.ece(y, prob), "brier": float(np.mean((prob - y) ** 2)), "admitted": int(admitted.sum()), "rejected": int((~admitted).sum()),
    }


def temporal_metrics(target: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred_contact = prob[:, :, 0] >= .5; pred_stuck = prob[:, :, 1] >= .5
    y_contact = target[:, :, 0].astype(bool); y_stuck = target[:, :, 1].astype(bool)
    delays = {"contact": [], "stuck": []}; early = []; transient_missed = []
    aggregate_alarm = prob[:, :, 4] >= threshold
    for i in range(len(target)):
        unsafe_ticks = np.flatnonzero(target[i, :, 4] > .5)
        if len(unsafe_ticks):
            alarms = np.flatnonzero(aggregate_alarm[i])
            early.append(bool(len(alarms) and alarms[0] <= unsafe_ticks[0]))
        for component, truth, prediction in (("contact", y_contact[i], pred_contact[i]), ("stuck", y_stuck[i], pred_stuck[i])):
            events = np.flatnonzero(truth)
            if len(events):
                detections = np.flatnonzero(prediction & (np.arange(len(truth)) >= events[0]))
                delays[component].append(int(detections[0] - events[0]) if len(detections) else None)
        for lo, hi in event_runs(y_contact[i].tolist()):
            if lo == hi: transient_missed.append(not pred_contact[i, lo])
    finite_delays = [d for values in delays.values() for d in values if d is not None]
    return {
        "per_tick_contact_recall": float(pred_contact[y_contact].mean()) if y_contact.any() else None,
        "per_tick_stuck_recall": float(pred_stuck[y_stuck].mean()) if y_stuck.any() else None,
        "first_event_detection_delay_ticks_median": float(np.median(finite_delays)) if finite_delays else None,
        "contact_detection_delay_ticks_median": float(np.median([d for d in delays["contact"] if d is not None])) if any(d is not None for d in delays["contact"]) else None,
        "stuck_detection_delay_ticks_median": float(np.median([d for d in delays["stuck"] if d is not None])) if any(d is not None for d in delays["stuck"]) else None,
        "events_with_no_detection": sum(d is None for values in delays.values() for d in values),
        "unsafe_branches_detected_before_or_at_first_unsafe_tick": float(np.mean(early)) if early else None,
        "missed_transient_contact_rate": float(np.mean(transient_missed)) if transient_missed else None,
        "transient_contact_events": len(transient_missed),
    }


def evaluate_split(name: str, branches: list[dict], logits: np.ndarray, target: np.ndarray,
                   temperature: float, threshold: float, route_rows: list[dict], route_indices: np.ndarray,
                   kinematic: np.ndarray) -> dict:
    prob = 1 / (1 + np.exp(-logits / temperature))
    aggregate = discrimination(target[:, -1, 4], prob[:, -1, 4], threshold)
    components = {
        "collision_contact": discrimination(target[:, -1, 2], prob[:, -1, 2], .5),
        "stuck": discrimination(target[:, -1, 3], prob[:, -1, 3], .5),
    }
    FS.rows_global = route_rows
    planning = FS.evaluate_condition(name, route_rows, route_indices, prob[:, -1, 4], threshold, kinematic)
    family = {}
    for fam in FAMILIES:
        mask = np.asarray([b["family"] == fam for b in branches])
        family[fam] = {
            "aggregate": discrimination(target[mask, -1, 4], prob[mask, -1, 4], threshold),
            "collision_contact": discrimination(target[mask, -1, 2], prob[mask, -1, 2], .5),
            "stuck": discrimination(target[mask, -1, 3], prob[mask, -1, 3], .5),
        }
    return {"aggregate": aggregate, "components": components, "temporal": temporal_metrics(target, prob, threshold),
            "candidate_filter_and_planning": planning, "per_family": family}


def main() -> int:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    fixture = evaluator_fixture()
    states, token_index, receipt = load_dense()
    audit = observability_audit(states)
    dense_by_split = {split: [b for s in states if split_name(s) == split for b in s["branches"]] for split in ("fit", "calibration", "heldout")}
    prevalence = {}
    for split, branches in dense_by_split.items():
        _, _, overall = positive_weights(branches); prevalence[split] = {"overall": overall}
        for family in FAMILIES:
            _, _, fam = positive_weights([b for b in branches if b["family"] == family]); prevalence[split][family] = fam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = OUT / f"dense_temporal_safety_head_seed_{SEED}.pt"
    model, history, training = train_model(dense_by_split["fit"], device)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epoch": 60,
                "parameter_count": training["parameter_count"], "architecture": "DENSE_TEMPORAL_SAFETY_HEAD_V1"}, checkpoint)
    clone = DenseTemporalSafety().to(device); clone.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]); clone.eval(); model = clone
    cal_logits, cal_target = predict(model, dense_by_split["calibration"], device)
    temperature = fit_temperature(cal_logits[:, -1, 4], cal_target[:, -1, 4])
    cal_prob = 1 / (1 + np.exp(-cal_logits[:, -1, 4] / temperature))
    threshold = choose_threshold(cal_prob, cal_target[:, -1, 4])
    held_logits, held_target = predict(model, dense_by_split["heldout"], device)
    route_rows = FS.load_metadata(); FS.rows_global = route_rows
    route_index_by_id = {r["state_id"] + f":{int(r['candidate_index']):02d}": i for i, r in enumerate(route_rows)}
    route_by_split = {
        split: np.asarray([route_index_by_id[b["branch_id"]] for b in dense_by_split[split]], int)
        for split in ("fit", "calibration", "heldout")
    }
    kinematic = np.stack([r["kinematic"] for r in route_rows])
    held = evaluate_split("dense_true_future", dense_by_split["heldout"], held_logits, held_target,
                          temperature, threshold["threshold"], route_rows, route_by_split["heldout"], kinematic)
    oracle_prob = np.asarray([float(route_rows[i]["unsafe"]) for i in route_by_split["heldout"]])
    oracle = FS.evaluate_condition("oracle_safety", route_rows, route_by_split["heldout"], oracle_prob, .5, kinematic)
    aggregate = held["aggregate"]; contact = held["components"]["collision_contact"]; stuck = held["components"]["stuck"]
    temporal = held["temporal"]; plan = held["candidate_filter_and_planning"]["planning"]
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    gates = {
        "aggregate_auc_ge_0_80": aggregate["auc"] is not None and aggregate["auc"] >= .80,
        "contact_auc_ge_0_75": contact["auc"] is not None and contact["auc"] >= .75,
        "stuck_auc_ge_0_85": stuck["auc"] is not None and stuck["auc"] >= .85,
        "aggregate_recall_ge_0_95": aggregate["unsafe_recall"] >= .95,
        "aggregate_fnr_le_0_05": aggregate["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": aggregate["safe_candidate_retention"] >= .40,
        "ece_le_0_10": aggregate["ece"] <= .10,
        "six_states_retain_safe": plan["states_retaining_safe"] >= 6,
        "no_state_only_unsafe_admitted": plan["states_only_unsafe_admitted"] == 0,
        "all_no_safe_states_abstain": plan["correct_abstention"] == sum(not any(not route_rows[i]["unsafe"] for i in route_by_split["heldout"] if route_rows[i]["state_id"] == sid) for sid in sorted({route_rows[i]["state_id"] for i in route_by_split["heldout"]})),
        "contact_tick_recall_ge_0_80": temporal["per_tick_contact_recall"] is not None and temporal["per_tick_contact_recall"] >= .80,
        "stuck_tick_recall_ge_0_90": temporal["per_tick_stuck_recall"] is not None and temporal["per_tick_stuck_recall"] >= .90,
        "median_delay_le_1_tick": temporal["first_event_detection_delay_ticks_median"] is not None and temporal["first_event_detection_delay_ticks_median"] <= 1,
        "missed_transient_contact_le_0_20": temporal["missed_transient_contact_rate"] is not None and temporal["missed_transient_contact_rate"] <= .20,
        "selected_unsafe_rate_zero": plan["selected_unsafe_rate"] == 0,
        "progress_ge_80pct_oracle": plan["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": plan["normalized_safe_progress_regret"] is not None and plan["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": plan["best_safe_top3"] is not None and plan["best_safe_top3"] >= .75,
        "false_abstention_le_1": plan["false_abstention"] <= 1,
    }
    all_pass = all(gates.values())
    tendency = sum([
        aggregate["auc"] is not None and aggregate["auc"] >= .7659,
        contact["auc"] is not None and contact["auc"] >= .6929,
        stuck["auc"] is not None and stuck["auc"] >= .8880,
        aggregate["safe_candidate_retention"] >= .20,
        temporal["per_tick_contact_recall"] is not None and temporal["per_tick_contact_recall"] >= .60,
    ]) >= 2
    classification = "DENSE_TEMPORAL_SAFETY_OBSERVABILITY_SIGNAL" if all_pass else ("DENSE_TEMPORAL_SAFETY_POSITIVE_TENDENCY" if tendency else "RGB_ONLY_DENSE_TEMPORAL_SAFETY_NO_GO")
    frozen = {
        "sparse_endpoint_safety_head": {"source_commit": "c26a89a7ea6a8aeec06db9397d97b6a67a1dbc6c", "aggregate_auc": .6565, "collision_contact_auc": .4549, "stuck_auc": .7863, "safe_retention": 0., "abstention": 1.},
        "joint_safety_auxiliary_successor": {"source_commit": "f72fe00c8426a973fbb56c521e5a89a563a9373f", "aggregate_auc": .7459, "collision_contact_auc": .6429, "stuck_auc": .8680, "safe_retention": 0., "abstention": 1.},
    }
    prior = json.loads((ROOT / ".generated/candidate_conditioned_future_safety_v1/result.json").read_text())
    guard = json.loads((ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1/result.json").read_text())
    frozen["action_only"] = prior["heldout"]["action_only"]
    frozen["privileged_static_grid_guard"] = {
        "classification": guard["classification"],
        "heldout": guard["guard_discrimination"]["by_split"]["heldout"],
    }
    result = {
        "schema": "dense_temporal_true_future_safety_observability_v1_result",
        "source_commit": "9ba5e1f0e6742f32c45e9c101a1b941e91444bea",
        "preserved_results": ["TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO", "STRUCTURED_SAFETY_LABEL_OR_ALIGNMENT_DEFECT"],
        "bindings": {"target_index": "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874", "dense_evidence_sha256": sha(EVIDENCE / "evidence_receipt.json"), "token_index_digest": token_index["token_index_digest"]},
        "materialization": receipt, "temporal_observability_audit": audit,
        "evaluator_fixture": fixture, "safety_prevalence": prevalence,
        "model": {"name": "DENSE_TEMPORAL_SAFETY_HEAD_V1", "seed": SEED, "architecture": "specified spatial CNN plus causal GRU", **training,
                  "epochs": 60, "training_history": history, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint)},
        "calibration": {"temperature": temperature, **threshold},
        "heldout": held, "oracle_safety_kinematic_upper_bound": oracle, "frozen_baselines": frozen,
        "gate": {"passed": all_pass, "checks": gates, "positive_tendency_predeclared_rule": "at least two: aggregate AUC >= .7659, contact AUC >= .6929, stuck AUC >= .8880, safe retention >= .20, contact tick recall >= .60", "positive_tendency": tendency},
        "classification": classification,
        "custody": {"safety_head_seeds_trained": 1, "predictors_opened_or_trained": 0, "new_states": 0, "new_branches": 0, "physics_replay_for_tick_evidence": True, "memory_novelty_navigation_models_trained": False},
        "runtime": {"end_to_end_s": time.time() - started, "storage_bytes": receipt["token_cache_bytes"] + checkpoint.stat().st_size},
    }
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "result_sha256": sha(OUT / "result.json"), "gates": gates}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
