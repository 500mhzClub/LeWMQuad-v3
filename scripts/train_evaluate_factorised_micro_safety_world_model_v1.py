#!/usr/bin/env python3
"""Train and prospectively evaluate the factorised micro-safety model."""
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

from scripts import materialize_enhanced_embodied_safety_observability_v2 as SENSOR
from scripts import train_evaluate_enhanced_embodied_safety_observability_v2 as OLD
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as METRIC
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE

OUT = ROOT / ".generated/factorised_micro_safety_world_model_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorised_micro_safety_world_model_v1")
FRESH_INDEX = OUT / "fresh_sensor_index.json"
SEED = 2026082010
FAMILIES = ("large_enclosed_maze", "medium_enclosed_maze", "small_enclosed_maze", "loop_alias_stress")
DELTA_D = .03
DELTA_THETA = math.radians(5.)
CONTACT_SENSOR = np.asarray([*range(0, 6), *range(33, 61), *range(61, 73)], np.int64)
STUCK_SENSOR = np.asarray([*range(61, 73), *range(9, 33), *range(6, 9), *range(3, 6)], np.int64)
_PLANNING_CACHE: dict[int, tuple[list[dict], list[dict]]] = {}


def json_default(value):
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                                        default=json_default).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_default) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle: np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name]); digest.update(name.encode()); digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def integrate(post_slew: list, waypoint: list[float]) -> np.ndarray:
    x = y = yaw = 0.
    for vx, vy, wz in [tick for block in post_slew[:3] for tick in block]:
        x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * .1
        y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * .1
        yaw = wrap(yaw + float(wz) * .1)
    wx, wy, sg, cg = waypoint; goal_heading = math.atan2(sg, cg)
    p_d = math.hypot(wx, wy) - math.hypot(wx - x, wy - y)
    p_theta = abs(wrap(goal_heading)) - abs(wrap(goal_heading - yaw))
    return np.asarray([x, y, math.sin(yaw), math.cos(yaw), p_d, p_theta], np.float32)


def old_fit_rows() -> tuple[list[dict], dict]:
    rows, index = OLD.load_branches()
    for row in rows:
        row["target"] = DENSE.branch_targets(row).astype(np.float32)
    return rows, index


def fresh_rows() -> tuple[list[dict], dict]:
    index = json.loads(FRESH_INDEX.read_text())
    if not index.get("complete") or index.get("content_digest") != canonical_digest({k: v for k, v in index.items() if k != "content_digest"}):
        raise RuntimeError("fresh sensor index is not complete or digest-valid")
    manifest = json.loads((OUT / "fresh_panel_manifest.json").read_text())
    states = {row["state_id"]: row for row in manifest["states"]}
    rows = []
    for record in index["state_records"]:
        state = states[record["state_id"]]
        if sha(Path(record["shard_path"])) != record["shard_sha256"]: raise RuntimeError(f"shard mismatch {record['state_id']}")
        with np.load(record["shard_path"], allow_pickle=False) as loaded:
            current = np.asarray(loaded["current"], np.float32); future = np.asarray(loaded["future"], np.float32)
            action = np.asarray(loaded["action_control"], np.float32); labels = np.asarray(loaded["labels"], np.float32)
        heading_body = float(record["branches"][0]["route_heading_world_rad"]) - float(state["start_pose"][1])
        waypoint = [*map(float, state["waypoint_body_xy"]), math.sin(heading_body), math.cos(heading_body)]
        for branch in record["branches"]:
            ci = int(branch["candidate_index"])
            rows.append({**branch, "split": record["split"], "family": record["family"], "scene_id": record["scene_id"],
                         "current_enhanced": current[ci], "future_enhanced": future[ci], "action_control": action[ci],
                         "target": labels[ci], "kinematic": integrate(branch["post_slew"], waypoint)})
    if len(rows) != 576 or len({row["state_id"] for row in rows}) != 48: raise RuntimeError("fresh cardinality mismatch")
    return rows, index


def prevalence(rows: list[dict]) -> dict:
    def one(subset):
        unsafe = np.asarray([row["target"][-1, 4] > .5 for row in subset]); contact = np.asarray([row["target"][-1, 2] > .5 for row in subset])
        stuck = np.asarray([row["target"][-1, 3] > .5 for row in subset]); states = defaultdict(list)
        for row in subset: states[row["state_id"]].append(row)
        safe_counts = {state: sum(not bool(row["target"][-1, 4]) for row in values) for state, values in states.items()}
        return {"branches": len(subset), "safe": int((~unsafe).sum()), "unsafe": int(unsafe.sum()),
                "contact_positive": int(contact.sum()), "stuck_positive": int(stuck.sum()),
                "contact_stuck_overlap": int((contact & stuck).sum()),
                "contact_positive_event_ticks": int(sum(row["target"][:, 0].sum() for row in subset)),
                "stuck_positive_event_ticks": int(sum(row["target"][:, 1].sum() for row in subset)),
                "states": len(states), "states_no_safe_candidate": sum(value == 0 for value in safe_counts.values()),
                "safe_candidates_per_state": safe_counts}
    report = {"by_split": {}, "by_split_family": {}}
    for split in ("calibration", "heldout"):
        subset = [row for row in rows if row["split"] == split]; report["by_split"][split] = one(subset)
        report["by_split_family"][split] = {family: one([row for row in subset if row["family"] == family]) for family in FAMILIES}
    checks = {}
    for split in ("calibration", "heldout"):
        item = report["by_split"][split]
        checks[split] = {"safe_and_unsafe": item["safe"] > 0 and item["unsafe"] > 0,
                         "contact_at_least_24": item["contact_positive"] >= 24,
                         "stuck_at_least_24": item["stuck_positive"] >= 24,
                         "four_safe_states_per_family": all(6 - report["by_split_family"][split][f]["states_no_safe_candidate"] >= 4 for f in FAMILIES),
                         "each_family_contact_and_stuck": all(report["by_split_family"][split][f]["contact_positive"] > 0 and report["by_split_family"][split][f]["stuck_positive"] > 0 for f in FAMILIES)}
    report["checks"] = checks; report["passed"] = all(all(value.values()) for value in checks.values())
    return report


def evaluator_fixture() -> dict:
    contact = np.asarray([[0, 1, 0], [0, 1, 1], [0, 0, 0]], np.float32)
    stuck = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]], np.float32)
    admission = ~(np.asarray([True, True, False]) | np.asarray([False, False, True]))
    tie_rejected = not (.5 < .5)
    cases = {"transient_contact": int(contact[0].sum()) == 1, "persistent_contact": int(contact[1].sum()) == 2,
             "delayed_stuck": int(np.flatnonzero(stuck[2])[0]) == 2, "safe_branch": not bool(contact[2].any()),
             "all_candidates_unsafe": bool(np.ones(12, bool).all()), "one_safe_candidate": int((~np.asarray([1] * 11 + [0], bool)).sum()) == 1,
             "separate_contact_stuck": bool(contact[0].any()) and bool(stuck[2].any()), "or_composition": not bool(admission.any()),
             "threshold_tie_rejected": tie_rejected, "abstention": not bool(np.any(np.asarray([.8, .9]) < .5))}
    candidate = [{"candidate_index": index, "kinematic": np.asarray([0, 0, 0, 1, index / 100., 0], np.float32), "unsafe": index != 7,
                  "p_d": index / 100., "p_theta": 0.} for index in range(12)]
    order = route_order(candidate, list(range(12))); cases["deterministic_kinematic_selection"] = order[0] == 9
    encoded = json.dumps({"cases": cases}, sort_keys=True); cases["deterministic_row_ledger_serialization"] = encoded == json.dumps(json.loads(encoded), sort_keys=True)
    payload = {"schema": "factorised_micro_safety_evaluator_fixture_v1", "cases": cases, "pass": all(cases.values())}
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if not payload["pass"]: raise RuntimeError("evaluator fixture failed")
    return payload


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__(); self.conv = nn.Conv1d(width, width, 3, dilation=dilation); self.dilation = dilation
    def forward(self, x): return x + F.gelu(self.conv(F.pad(x, (2 * self.dilation, 0))))


class ContactSpecialist(nn.Module):
    def __init__(self, input_width: int):
        super().__init__(); self.projection = nn.Sequential(nn.Linear(input_width * 3 + 6, 96), nn.GELU())
        self.temporal = nn.Sequential(CausalBlock(96, 1), CausalBlock(96, 2), CausalBlock(96, 4)); self.output = nn.Linear(96, 2)
    def forward(self, current, future, action):
        current = current[:, None].expand(-1, future.shape[1], -1)
        value = self.projection(torch.cat((current, future, future - current, action), -1)).transpose(1, 2)
        return self.output(self.temporal(value).transpose(1, 2))


class StuckSpecialist(nn.Module):
    def __init__(self, input_width: int):
        super().__init__(); self.embodied = nn.Sequential(nn.Linear(input_width * 3, 64), nn.GELU())
        self.action = nn.Sequential(nn.Linear(6, 64), nn.GELU()); self.temporal = nn.GRU(128, 128, batch_first=True); self.output = nn.Linear(128, 2)
    def forward(self, current, future, action):
        current = current[:, None].expand(-1, future.shape[1], -1)
        state = self.embodied(torch.cat((current, future, future - current), -1)); hidden, _ = self.temporal(torch.cat((state, self.action(action)), -1))
        return self.output(hidden)


class FactorisedModel(nn.Module):
    def __init__(self):
        super().__init__(); self.contact = ContactSpecialist(len(CONTACT_SENSOR)); self.stuck = StuckSpecialist(len(STUCK_SENSOR))


def stats(rows: list[dict], indices: np.ndarray, name: str) -> dict:
    value = np.concatenate([np.concatenate((row["current_enhanced"][None, indices], row["future_enhanced"][:, indices]), 0) for row in rows]).astype(np.float64)
    mean = value.mean(0); std = value.std(0); degenerate = std < 1e-7; std[degenerate] = 1.
    payload = {"specialist": name, "channel_indices": indices.tolist(), "channel_names": [SENSOR.CHANNELS[i] for i in indices],
               "mean": mean.tolist(), "std": std.tolist(), "degenerate": np.flatnonzero(degenerate).tolist()}
    payload["digest"] = canonical_digest(payload); atomic_json(OUT / f"{name}_standardization.json", payload); return payload


def tensors(rows: list[dict], specialist: str, statistical: dict, device):
    indices = CONTACT_SENSOR if specialist == "contact" else STUCK_SENSOR
    mean = np.asarray(statistical["mean"], np.float32); std = np.asarray(statistical["std"], np.float32)
    current = np.stack([(row["current_enhanced"][indices] - mean) / std for row in rows])
    future = np.stack([(row["future_enhanced"][:, indices] - mean) / std for row in rows])
    action = np.stack([row["action_control"] for row in rows]); target_columns = (0, 2) if specialist == "contact" else (1, 3)
    target = np.stack([row["target"][:, target_columns] for row in rows])
    return tuple(torch.from_numpy(value.astype(np.float32)).to(device) for value in (current, future, action, target))


def weights(rows: list[dict], specialist: str) -> np.ndarray:
    columns = (0, 2) if specialist == "contact" else (1, 3); value = np.concatenate([row["target"][:, columns] for row in rows], 0)
    positive = value.sum(0); negative = len(value) - positive
    if np.any(positive == 0) or np.any(negative == 0): raise RuntimeError(f"degenerate fit target for {specialist}")
    return (negative / positive).astype(np.float32)


def component_loss(logits, target, positive_weight):
    bce = sum(F.binary_cross_entropy_with_logits(logits[..., i], target[..., i], pos_weight=positive_weight[i]) for i in range(2)) / 2
    positive = target[:, -1, 1] > .5; negative = ~positive
    if positive.any() and negative.any():
        differences = logits[positive, -1, 1][:, None] - logits[negative, -1, 1][None, :]
        rank = F.softplus(-differences).mean()
    else: rank = logits.sum() * 0
    return bce + .25 * rank, bce, rank


def train(fit: list[dict], device) -> tuple[FactorisedModel, dict, dict, dict]:
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = FactorisedModel().to(device); contact_stats = stats(fit, CONTACT_SENSOR, "contact"); stuck_stats = stats(fit, STUCK_SENSOR, "stuck")
    parameters = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("contact", "stuck")}
    parameters["total"] = sum(parameters.values())
    if parameters["total"] >= 500000: raise RuntimeError("factorised parameter cap exceeded")
    contact_ids = {id(p) for p in model.contact.parameters()}; stuck_ids = {id(p) for p in model.stuck.parameters()}
    if contact_ids & stuck_ids: raise RuntimeError("specialists share parameters")
    positive_weights = {"contact": weights(fit, "contact"), "stuck": weights(fit, "stuck")}
    optimizers = {name: torch.optim.AdamW(getattr(model, name).parameters(), lr=1e-3, weight_decay=1e-4) for name in ("contact", "stuck")}
    grouped = defaultdict(list)
    for row in fit: grouped[row["state_id"]].append(row)
    state_ids = sorted(grouped)
    # Fit-only smoke.
    smoke = grouped[state_ids[0]]
    for name, statistical in (("contact", contact_stats), ("stuck", stuck_stats)):
        current, future, action, target = tensors(smoke, name, statistical, device); module = getattr(model, name)
        logits = module(current, future, action); loss, _, _ = component_loss(logits, target, torch.from_numpy(positive_weights[name]).to(device)); loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0 for p in module.parameters()):
            raise RuntimeError(f"{name} smoke gradient failure")
        module.eval()
        with torch.inference_mode():
            first = module(current, future, action)
            if not torch.equal(first, module(current, future, action)): raise RuntimeError(f"{name} nondeterministic")
            if torch.allclose(first, module(current, future, action.roll(1, 1))): raise RuntimeError(f"{name} action insensitive")
            if torch.allclose(first, module(current, future.flip(1), action.flip(1))): raise RuntimeError(f"{name} temporal-order insensitive")
        module.train(); module.zero_grad(set_to_none=True)
    smoke_path = OUT / ".factorised_smoke.pt"; torch.save(model.state_dict(), smoke_path)
    clone = FactorisedModel().to(device); clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    history = []; started = time.time()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        order = list(state_ids); random.Random(SEED + epoch).shuffle(order); totals = defaultdict(float)
        for state_id in order:
            batch = grouped[state_id]
            for name, statistical in (("contact", contact_stats), ("stuck", stuck_stats)):
                module = getattr(model, name); optimizer = optimizers[name]; current, future, action, target = tensors(batch, name, statistical, device)
                optimizer.zero_grad(set_to_none=True); logits = module(current, future, action)
                loss, bce, rank = component_loss(logits, target, torch.from_numpy(positive_weights[name]).to(device)); loss.backward(); optimizer.step()
                totals[f"{name}_loss"] += float(loss.detach()); totals[f"{name}_bce"] += float(bce.detach()); totals[f"{name}_rank"] += float(rank.detach())
        record = {"epoch": epoch + 1, **{key: value / len(order) for key, value in totals.items()}}; history.append(record)
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps(record), flush=True)
    training = {"seed": SEED, "epochs": 60, "parameter_count": parameters, "optimizer": "AdamW", "learning_rate": .001,
                "weight_decay": .0001, "ranking_weight": .25, "smoke_passed": True, "no_shared_parameters": True,
                "positive_weights": {key: value.tolist() for key, value in positive_weights.items()},
                "runtime_s": time.time() - started, "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    return model, contact_stats, stuck_stats, {"history": history, "training": training}


def predict(model: FactorisedModel, rows: list[dict], contact_stats: dict, stuck_stats: dict, device):
    output = {}
    with torch.inference_mode():
        for name, statistical in (("contact", contact_stats), ("stuck", stuck_stats)):
            values = []
            for start in range(0, len(rows), 96):
                current, future, action, _ = tensors(rows[start:start + 96], name, statistical, device)
                values.append(getattr(model, name)(current, future, action).float().cpu().numpy())
            output[name] = np.concatenate(values)
    return output


def route_order(rows: list[dict], ids: list[int]) -> list[int]:
    remaining = list(ids); order = []
    while remaining:
        best_distance = max(float(rows[index]["kinematic"][4]) for index in remaining)
        near = [index for index in remaining if best_distance - float(rows[index]["kinematic"][4]) <= DELTA_D]
        pick = min(near, key=lambda index: (-float(rows[index]["kinematic"][5]), int(rows[index]["candidate_index"])))
        order.append(pick); remaining.remove(pick)
    return order


def preference(a, b) -> int:
    distance = float(a["p_d"]) - float(b["p_d"])
    if abs(distance) > DELTA_D: return 1 if distance > 0 else -1
    heading = float(a["p_theta"]) - float(b["p_theta"])
    if abs(heading) > DELTA_THETA: return 1 if heading > 0 else -1
    return 0


def best_safe(rows, ids):
    safe = [index for index in ids if not bool(rows[index]["unsafe"])]
    if not safe: return None
    best = safe[0]
    for index in safe[1:]:
        if preference(rows[index], rows[best]) > 0: best = index
    return best


def planning_metrics(rows: list[dict], admitted: np.ndarray, include_rows=True) -> dict:
    cache_key = id(rows)
    if cache_key not in _PLANNING_CACHE or _PLANNING_CACHE[cache_key][0] is not rows:
        grouped = defaultdict(list)
        for index, row in enumerate(rows): grouped[row["state_id"]].append(index)
        prepared = [{"state_id": state_id, "ids": ids,
                                      "safe": [i for i in ids if not rows[i]["unsafe"]],
                                      "rank": route_order(rows, ids), "best": best_safe(rows, ids)}
                                     for state_id, ids in sorted(grouped.items())]
        _PLANNING_CACHE[cache_key] = (rows, prepared)
    state_rows = []; selected_progress = []; oracle_progress = []; regrets = []; top1 = []; top3 = []; selected_unsafe = []
    false_abstention = correct_abstention = 0
    for cached in _PLANNING_CACHE[cache_key][1]:
        state_id, ids, safe, rank, best = cached["state_id"], cached["ids"], cached["safe"], cached["rank"], cached["best"]
        admitted_rank = [i for i in rank if admitted[i]]; pick = admitted_rank[0] if admitted_rank else None
        oracle_rank = [i for i in rank if i in safe]; oracle_pick = oracle_rank[0] if oracle_rank else None
        if pick is None:
            if safe: false_abstention += 1
            else: correct_abstention += 1
        else:
            selected_progress.append(float(rows[pick]["p_d"])); selected_unsafe.append(bool(rows[pick]["unsafe"]))
        if oracle_pick is not None: oracle_progress.append(float(rows[oracle_pick]["p_d"]))
        if best is not None:
            top1.append(pick == best); top3.append(best in admitted_rank[:3])
            if pick is not None and not rows[pick]["unsafe"] and len(safe) >= 2:
                values = [float(rows[i]["p_d"]) for i in safe]; spread = max(values) - min(values)
                if spread > 1e-8: regrets.append((float(rows[best]["p_d"]) - float(rows[pick]["p_d"])) / spread)
        state_rows.append({"state_id": state_id, "family": rows[ids[0]]["family"], "admitted": int(admitted[ids].sum()),
                           "admitted_safe": int(sum(admitted[i] and not rows[i]["unsafe"] for i in ids)),
                           "admitted_unsafe": int(sum(admitted[i] and rows[i]["unsafe"] for i in ids)),
                           "selected_candidate": None if pick is None else int(rows[pick]["candidate_index"]),
                           "selected_unsafe": None if pick is None else bool(rows[pick]["unsafe"]),
                           "selected_p_d": None if pick is None else float(rows[pick]["p_d"]),
                           "selected_p_theta": None if pick is None else float(rows[pick]["p_theta"]),
                           "oracle_kinematic_p_d": None if oracle_pick is None else float(rows[oracle_pick]["p_d"])})
    selected_mean = float(np.mean(selected_progress)) if selected_progress else 0.; oracle_mean = float(np.mean(oracle_progress)) if oracle_progress else 0.
    return {"states": len(_PLANNING_CACHE[cache_key][1]), "states_retaining_safe": sum(row["admitted_safe"] > 0 for row in state_rows),
            "states_only_unsafe_admitted": sum(row["admitted"] > 0 and row["admitted_safe"] == 0 for row in state_rows),
            "states_no_admitted": sum(row["admitted"] == 0 for row in state_rows), "false_abstentions": false_abstention,
            "correct_abstentions": correct_abstention, "selected_unsafe_count": int(sum(selected_unsafe)),
            "selected_unsafe_rate": float(np.mean(selected_unsafe)) if selected_unsafe else 0.,
            "mean_selected_route_progress_m": selected_mean, "oracle_safety_kinematic_progress_m": oracle_mean,
            "oracle_progress_fraction": selected_mean / oracle_mean if abs(oracle_mean) > 1e-9 else None,
            "normalized_safe_progress_regret": float(np.mean(regrets)) if regrets else None, "normalized_regret_states": len(regrets),
            "best_safe_top1": float(np.mean(top1)) if top1 else None, "best_safe_top3": float(np.mean(top3)) if top3 else None,
            "per_state": state_rows if include_rows else None}


def decision_metrics(rows, pc, ps, tc, ts, include_rows=False):
    admitted = (pc < tc) & (ps < ts); contact = np.asarray([row["target"][-1, 2] > .5 for row in rows]); stuck = np.asarray([row["target"][-1, 3] > .5 for row in rows])
    unsafe = np.asarray([row["target"][-1, 4] > .5 for row in rows]); rejected = ~admitted; plan = planning_metrics(rows, admitted, include_rows)
    return {"contact_recall": float((pc[contact] >= tc).mean()) if contact.any() else None,
            "stuck_recall": float((ps[stuck] >= ts).mean()) if stuck.any() else None,
            "aggregate_unsafe_recall": float(rejected[unsafe].mean()), "aggregate_false_negative_rate": float(admitted[unsafe].mean()),
            "safe_candidate_retention": float(admitted[~unsafe].mean()), "admitted_safe_count": int((admitted & ~unsafe).sum()),
            "admitted_unsafe_count": int((admitted & unsafe).sum()), **plan}


def threshold_values(prob):
    return np.concatenate(([np.nextafter(0., -np.inf)], np.unique(prob), [np.nextafter(1., np.inf)]))


def calibrate(rows, logits):
    temperatures = {name: DENSE.fit_temperature(logits[name][:, -1, 1], np.asarray([row["target"][-1, 2 if name == "contact" else 3] for row in rows])) for name in ("contact", "stuck")}
    probabilities = {name: 1 / (1 + np.exp(-logits[name][:, -1, 1] / temperatures[name])) for name in ("contact", "stuck")}
    candidates = []; started = time.time()
    frontier_fields = ("contact_recall", "stuck_recall", "aggregate_unsafe_recall", "aggregate_false_negative_rate",
                       "safe_candidate_retention", "admitted_safe_count", "admitted_unsafe_count", "states_retaining_safe",
                       "states_only_unsafe_admitted", "states_no_admitted", "selected_unsafe_count", "false_abstentions",
                       "mean_selected_route_progress_m", "normalized_safe_progress_regret", "best_safe_top1", "best_safe_top3")
    frontier = {name: [] for name in ("contact_threshold", "stuck_threshold", *frontier_fields)}
    for tc in threshold_values(probabilities["contact"]):
        for ts in threshold_values(probabilities["stuck"]):
            metrics = decision_metrics(rows, probabilities["contact"], probabilities["stuck"], tc, ts)
            frontier["contact_threshold"].append(float(tc)); frontier["stuck_threshold"].append(float(ts))
            for field in frontier_fields:
                value = metrics[field]; frontier[field].append(np.nan if value is None else value)
            eligible = metrics["aggregate_unsafe_recall"] >= .95 and metrics["aggregate_false_negative_rate"] <= .05 and metrics["contact_recall"] >= .90 and metrics["stuck_recall"] >= .90 and metrics["states_only_unsafe_admitted"] == 0 and metrics["selected_unsafe_count"] == 0
            if eligible:
                regret = metrics["normalized_safe_progress_regret"]
                key = (metrics["states_retaining_safe"], metrics["safe_candidate_retention"], metrics["mean_selected_route_progress_m"],
                       -float("inf") if regret is None else -regret, -metrics["false_abstentions"],
                       -float("inf") if metrics["best_safe_top3"] is None else metrics["best_safe_top3"], -tc, -ts)
                candidates.append((key, float(tc), float(ts), metrics))
    frontier_arrays = {name: np.asarray(values, np.float64) for name, values in frontier.items()}
    frontier_path = CACHE / "fresh_calibration_joint_threshold_frontier_v1.npz"; atomic_npz(frontier_path, **frontier_arrays)
    frontier_index = {"file": str(frontier_path), "sha256": sha(frontier_path), "content_digest": array_digest(frontier_arrays),
                      "pairs": len(frontier_arrays["contact_threshold"]), "fields": sorted(frontier_arrays)}
    atomic_json(OUT / "calibration_joint_threshold_frontier_index.json", frontier_index)
    if not candidates: return temperatures, probabilities, {"feasible": False, "pairs_evaluated": frontier_index["pairs"],
                                                              "frontier": frontier_index, "runtime_s": time.time() - started}
    _, tc, ts, metrics = max(candidates, key=lambda value: value[0])
    return temperatures, probabilities, {"feasible": True, "contact_threshold": tc, "stuck_threshold": ts, "metrics": metrics,
                                         "eligible_pairs": len(candidates), "pairs_evaluated": frontier_index["pairs"],
                                         "frontier": frontier_index, "runtime_s": time.time() - started}


def component_metrics(rows, logits, temperature, threshold, component):
    active_column, cumulative_column = (0, 2) if component == "contact" else (1, 3)
    probability = 1 / (1 + np.exp(-logits / temperature)); branch_probability = probability[:, -1, 1]
    branch_label = np.asarray([row["target"][-1, cumulative_column] > .5 for row in rows]); decision = branch_probability >= threshold
    active_label = np.stack([row["target"][:, active_column] > .5 for row in rows]); active_decision = probability[:, :, 0] >= threshold
    delays = []; transient_missed = []; event_detected = []
    for label, detected in zip(active_label, active_decision):
        positive = np.flatnonzero(label)
        if not len(positive): continue
        first = int(positive[0]); hits = np.flatnonzero(detected[first:])
        event_detected.append(bool(len(hits))); delays.append(int(hits[0]) if len(hits) else None)
        if int(label.sum()) == 1: transient_missed.append(not bool(detected[first]))
    return {"prevalence": float(branch_label.mean()), "auc": METRIC.auc(branch_label, branch_probability),
            "average_precision": METRIC.average_precision(branch_label, branch_probability), "recall": float(decision[branch_label].mean()),
            "false_negative_rate": float((~decision)[branch_label].mean()), "specificity": float((~decision)[~branch_label].mean()),
            "ece": METRIC.ece(branch_label, branch_probability), "brier": float(np.mean((branch_probability - branch_label) ** 2)),
            "event_tick_recall": float(active_decision[active_label].mean()), "branch_event_detection_recall": float(np.mean(event_detected)),
            "median_detection_delay_ticks": float(np.median([value for value in delays if value is not None])) if any(value is not None for value in delays) else None,
            "missed_transient_event_rate": float(np.mean(transient_missed)) if transient_missed else None}


def family_results(rows, pc, ps, tc, ts, contact_logits, stuck_logits, temperatures):
    output = {}
    for family in FAMILIES:
        ids = [index for index, row in enumerate(rows) if row["family"] == family]; sub = [rows[i] for i in ids]
        decision = decision_metrics(sub, pc[ids], ps[ids], tc, ts, True)
        decision["contact"] = component_metrics(sub, contact_logits[ids], temperatures["contact"], tc, "contact")
        decision["stuck"] = component_metrics(sub, stuck_logits[ids], temperatures["stuck"], ts, "stuck")
        output[family] = decision
    return output


def persist_ledger(rows, logits, probabilities, tc, ts) -> dict:
    admitted = (probabilities["contact"] < tc) & (probabilities["stuck"] < ts)
    planning = planning_metrics(rows, admitted, True); selected = {(row["state_id"], row["selected_candidate"]) for row in planning["per_state"] if row["selected_candidate"] is not None}
    arrays = {"branch_id": np.asarray([row["branch_id"] for row in rows]), "state_id": np.asarray([row["state_id"] for row in rows]),
              "candidate_index": np.asarray([row["candidate_index"] for row in rows], np.int16), "split": np.asarray([row["split"] for row in rows]),
              "family": np.asarray([row["family"] for row in rows]), "contact_logits": logits["contact"].astype(np.float32),
              "stuck_logits": logits["stuck"].astype(np.float32), "contact_probability": probabilities["contact"].astype(np.float64),
              "stuck_probability": probabilities["stuck"].astype(np.float64), "labels": np.stack([row["target"] for row in rows]).astype(np.uint8),
              "admitted": admitted.astype(np.uint8), "selected": np.asarray([(row["state_id"], row["candidate_index"]) in selected for row in rows], np.uint8),
              "candidate_name": np.asarray([row["candidate"] for row in rows]),
              "candidate_action_control": np.stack([row["action_control"] for row in rows]).astype(np.float32),
              "p_d": np.asarray([row["p_d"] for row in rows], np.float32), "p_theta": np.asarray([row["p_theta"] for row in rows], np.float32),
              "kinematic": np.stack([row["kinematic"] for row in rows]).astype(np.float32)}
    path = CACHE / "fresh_heldout_row_level_evidence_v1.npz"; atomic_npz(path, **arrays); content = array_digest(arrays)
    index = {"schema": "factorised_micro_safety_row_level_evidence_v1", "rows": len(rows), "states": len(set(arrays["state_id"])),
             "file": str(path), "sha256": sha(path), "content_digest": content, "contact_threshold": tc, "stuck_threshold": ts,
             "row_level_evidence_persistence": True, "fields": sorted(arrays)}
    atomic_json(OUT / "row_level_evidence_index.json", index); return index


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True); started = time.time(); fixture = evaluator_fixture()
    fit, old_index = old_fit_rows(); fresh, fresh_index = fresh_rows(); audit = prevalence(fresh); atomic_json(OUT / "fresh_panel_adequacy.json", audit)
    if not audit["passed"]:
        result = {"schema": "factorised_micro_safety_world_model_v1_result", "classification": "FRESH_FACTORISED_SAFETY_PANEL_INADEQUATE",
                  "fresh_panel_adequacy": audit, "evaluator_fixture": fixture, "training_performed": False,
                  "bindings": fresh_index["bindings"], "runtime": {"total_s": time.time() - started, "fresh_collection_compute_s": fresh_index["runtime_compute_s"],
                  "fresh_collection_wall_s": fresh_index["parallel_wall_runtime_s"], "fresh_storage_bytes": fresh_index["storage_bytes"]}}
        atomic_json(OUT / "result.json", result); print(json.dumps({"classification": result["classification"]}, indent=2)); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, contact_stats, stuck_stats, training_bundle = train(fit, device)
    checkpoint = OUT / f"factorised_micro_safety_world_model_v1_seed_{SEED}.pt"
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epoch": 60, "contact_stats": contact_stats, "stuck_stats": stuck_stats,
                "parameter_count": training_bundle["training"]["parameter_count"]}, checkpoint)
    clone = FactorisedModel().to(device); clone.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"]); clone.eval()
    calibration_rows = [row for row in fresh if row["split"] == "calibration"]; heldout_rows = [row for row in fresh if row["split"] == "heldout"]
    calibration_logits = predict(clone, calibration_rows, contact_stats, stuck_stats, device)
    temperatures, calibration_probabilities, calibration = calibrate(calibration_rows, calibration_logits)
    if not calibration["feasible"]: raise RuntimeError("fresh calibration has no eligible threshold pair")
    heldout_logits = predict(clone, heldout_rows, contact_stats, stuck_stats, device)
    heldout_probabilities = {name: 1 / (1 + np.exp(-heldout_logits[name][:, -1, 1] / temperatures[name])) for name in ("contact", "stuck")}
    tc, ts = calibration["contact_threshold"], calibration["stuck_threshold"]
    heldout_decision = decision_metrics(heldout_rows, heldout_probabilities["contact"], heldout_probabilities["stuck"], tc, ts, True)
    heldout_components = {name: component_metrics(heldout_rows, heldout_logits[name], temperatures[name], tc if name == "contact" else ts, name) for name in ("contact", "stuck")}
    per_family = family_results(heldout_rows, heldout_probabilities["contact"], heldout_probabilities["stuck"], tc, ts,
                                heldout_logits["contact"], heldout_logits["stuck"], temperatures)
    no_family_collapse = all(value["states_retaining_safe"] > 0 and value["selected_unsafe_count"] == 0 for value in per_family.values())
    gate_checks = {"unsafe_recall": heldout_decision["aggregate_unsafe_recall"] >= .95,
                   "unsafe_fnr": heldout_decision["aggregate_false_negative_rate"] <= .05,
                   "safe_retention": heldout_decision["safe_candidate_retention"] >= .40,
                   "contact_recall": heldout_decision["contact_recall"] >= .90, "stuck_recall": heldout_decision["stuck_recall"] >= .90,
                   "contact_auc": heldout_components["contact"]["auc"] >= .80, "stuck_auc": heldout_components["stuck"]["auc"] >= .85,
                   "states_retaining_safe": heldout_decision["states_retaining_safe"] >= 18,
                   "no_only_unsafe_state": heldout_decision["states_only_unsafe_admitted"] == 0,
                   "zero_selected_unsafe": heldout_decision["selected_unsafe_rate"] == 0,
                   "false_abstentions": heldout_decision["false_abstentions"] <= 3,
                   "oracle_progress_fraction": heldout_decision["oracle_progress_fraction"] is not None and heldout_decision["oracle_progress_fraction"] >= .80,
                   "normalized_regret": heldout_decision["normalized_safe_progress_regret"] is not None and heldout_decision["normalized_safe_progress_regret"] <= .20,
                   "best_safe_top3": heldout_decision["best_safe_top3"] is not None and heldout_decision["best_safe_top3"] >= .75,
                   "no_family_collapse": no_family_collapse}
    classification = "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_SIGNAL" if all(gate_checks.values()) else "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_NO_SIGNAL"
    ledger = persist_ledger(heldout_rows, heldout_logits, heldout_probabilities, tc, ts)
    result = {"schema": "factorised_micro_safety_world_model_v1_result", "source_commit": "4bb63ff19a3972aa594fa9d14ea39f55a1401ccb",
              "experiment": "FACTORISED_MICRO_SAFETY_WORLD_MODEL_V1", "fit": {"states": 48, "branches": 576,
              "enhanced_sensor_index_digest": "d8b9721a2397961912e604b41b9b4eaea49ee34fc2c4735eba6f6e1edbe0933d",
              "specialist_ledger_digest": "e4e7ae1b494b171dd8a623a5368045a07f315e4ff05a85921b7e004c7d55e9de",
              "specialist_ledger_sha256": "a28be7a1254a77b553730c3024fb6ef24ed914a64ebf8bae3458142e3b0f8a08"},
              "fresh_panel": {"calibration_states": 24, "heldout_states": 24, "branches_each": 288,
              "manifest_digest": fresh_index["fresh_panel_manifest_digest"], "sensor_index_digest": fresh_index["content_digest"]},
              "fresh_panel_adequacy": audit, "evaluator_fixture": fixture, "specialists": {"contact_input_channels": [SENSOR.CHANNELS[i] for i in CONTACT_SENSOR],
              "stuck_input_channels": [SENSOR.CHANNELS[i] for i in STUCK_SENSOR], "parameter_count": training_bundle["training"]["parameter_count"],
              "independent_parameters_and_normalization": True}, "training": {**training_bundle["training"], "history": training_bundle["history"],
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint)}, "calibration": {**calibration, "temperatures": temperatures},
              "heldout": {"components": heldout_components, "combined_admissibility_and_planning": heldout_decision, "per_family": per_family},
              "prospective_gate": {"checks": gate_checks, "passed": all(gate_checks.values())}, "classification": classification,
              "next_experiment": "CANDIDATE_CONDITIONED_MICRO_SAFETY_PREDICTOR_V1" if classification == "FACTORISED_MICRO_SAFETY_TRUE_FUTURE_SIGNAL" else "CHANGED_ENVIRONMENTAL_SENSOR_CONTRACT_OR_NARROWER_SAFETY_CLAIM",
              "row_level_evidence": ledger, "runtime": {"total_s": time.time() - started, "fresh_collection_compute_s": fresh_index["runtime_compute_s"],
              "fresh_collection_wall_s": fresh_index["parallel_wall_runtime_s"],
              "training_s": training_bundle["training"]["runtime_s"], "peak_vram_bytes": training_bundle["training"]["peak_vram_bytes"],
              "fresh_storage_bytes": fresh_index["storage_bytes"], "checkpoint_bytes": checkpoint.stat().st_size, "ledger_bytes": Path(ledger["file"]).stat().st_size},
              "custody": {"one_factorised_seed_trained": True, "seed": SEED, "jepa_predictor_opened_or_trained": False,
              "global_memory_novelty_navigation_opened_or_trained": False, "nothing_left_running_at_commit": True}}
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification, "checkpoint_sha256": sha(checkpoint), "row_ledger_digest": ledger["content_digest"],
                      "gate": gate_checks, "heldout": heldout_decision}, indent=2, default=json_default)); return 0


if __name__ == "__main__": raise SystemExit(main())
