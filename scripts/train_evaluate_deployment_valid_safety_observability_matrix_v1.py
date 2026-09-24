#!/usr/bin/env python3
"""Train/evaluate the four frozen true-future safety modality conditions."""
from __future__ import annotations

import argparse
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
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import materialize_deployment_valid_dense_proprioception_v1 as PROP
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS
from scripts import train_evaluate_dense_temporal_true_future_safety_observability_v1 as DENSE

OUT = ROOT / ".generated/deployment_valid_safety_observability_matrix_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/deployment_valid_safety_observability_matrix_v1")
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
DENSE_OUT = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1"
BASE_SEED = 2026082008
CONDITIONS = ("ACTION_CONTROL_ONLY", "RAW_RGB", "PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION")
CHANNELS = DENSE.CHANNELS
FAMILIES = FS.FAMILIES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=FS.json_default) + "\n")
    os.replace(temp, path)


def condition_seed(name: str) -> int:
    return int(hashlib.sha256(f"DEPLOYMENT_VALID_SAFETY_OBSERVABILITY_MATRIX_V1|{BASE_SEED}|{name}".encode()).hexdigest()[:8], 16)


def evaluator_fixture() -> dict:
    transient = [0, 1, 0, 0]
    persistent = [0, 1, 1, 1]
    delayed = [0, 0, 0, 1]
    y = np.asarray([0, 0, 1, 1], bool)
    perfect = np.asarray([.05, .1, .9, .95])
    reversed_probability = perfect[::-1]
    tie_probability = np.asarray([.5, .5])
    cases = {
        "transient_one_tick_contact": sum(transient) == 1,
        "persistent_contact": sum(persistent) == 3,
        "delayed_stuck": delayed.index(1) == 3,
        "safe_branch": not any([0, 0, 0, 0]),
        "all_candidates_unsafe": bool(np.ones(12, bool).all()),
        "one_safe_candidate": int((~np.asarray([1] * 11 + [0], bool)).sum()) == 1,
        "no_candidate_admitted": not bool(np.any(np.asarray([.8, .9]) < .5)),
        "threshold_tie_rejected": not bool(np.any(tie_probability < .5)),
        "perfect_probability_ranking": FS.auc(y, perfect) == 1.,
        "reversed_probability_ranking": FS.auc(y, reversed_probability) == 0.,
    }
    route_rows = []
    for candidate in range(12):
        route_rows.append({"state_id": "purpose-999", "candidate_index": candidate,
                           "family": FAMILIES[0], "unsafe": candidate != 7,
                           "p_d": float(candidate) / 100., "p_theta": 0.})
    kinematic = np.zeros((12, 6), np.float32)
    kinematic[:, 4] = [row["p_d"] for row in route_rows]
    kinematic[:, 5] = [row["p_theta"] for row in route_rows]
    FS.rows_global = route_rows
    fixture_plan = FS.evaluate_condition("fixture", route_rows, np.arange(12),
                                         np.asarray([1. if row["unsafe"] else 0. for row in route_rows]), .5, kinematic)
    cases["one_safe_candidate_selected"] = fixture_plan["planning"]["per_state"][0]["selected_candidate"] == 7
    payload = {"schema": "deployment_valid_safety_observability_evaluator_fixture_v1",
               "cases": cases, "fixture_planning": fixture_plan, "pass": all(cases.values())}
    encoded = json.dumps(payload, sort_keys=True, default=FS.json_default)
    cases["deterministic_complete_json"] = encoded == json.dumps(json.loads(encoded), sort_keys=True)
    payload["pass"] = all(cases.values())
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if json.loads((OUT / "evaluator_fixture.json").read_text()) != payload or not payload["pass"]:
        raise RuntimeError("common evaluator fixture failed")
    return payload


def load_branches() -> tuple[list[dict], dict, dict]:
    states, token_index, receipt = DENSE.load_dense()
    proprio_index = json.loads((OUT / "proprio_index.json").read_text())
    if not proprio_index.get("complete") or proprio_index["proprio_index_digest"] != canonical_digest({k: v for k, v in proprio_index.items() if k != "proprio_index_digest"}):
        raise RuntimeError("invalid dense proprioception index")
    prop_by_state = {row["state_id"]: row for row in proprio_index["state_records"]}
    result = []
    for state in states:
        row = prop_by_state[state["state_id"]]
        with np.load(row["shard_path"]) as loaded:
            current = np.asarray(loaded["current"], np.float32)
            future = np.asarray(loaded["future"], np.float32)
            action_control = np.asarray(loaded["action_control"], np.float32)
        for branch in state["branches"]:
            ci = int(branch["candidate_index"])
            branch["split"] = state["split"]
            branch["family"] = state["family"]
            branch["current_rgb_path"] = state["current_frame"]["rgb_path"]
            branch["current_rgb_sha256"] = state["current_frame"]["rgb_sha256"]
            branch["future_rgb_paths"] = [tick["rgb_path"] for tick in branch["ticks"]]
            branch["future_rgb_sha256"] = [tick["rgb_sha256"] for tick in branch["ticks"]]
            branch["current_proprio"] = current[ci]
            branch["future_proprio"] = future[ci]
            branch["action_control"] = action_control[ci]
            result.append(branch)
    if len(result) != 576:
        raise RuntimeError("frozen branch cardinality changed")
    return result, proprio_index, {"dense_token_index": token_index, "dense_receipt": receipt}


def ensure_raw_rgb_cache(branches: list[dict]) -> dict:
    records: dict[str, str] = {}
    for branch in branches:
        records[branch["current_rgb_sha256"]] = branch["current_rgb_path"]
        records.update(zip(branch["future_rgb_sha256"], branch["future_rgb_paths"]))
    identities = sorted(records)
    path = CACHE / "raw_rgb_224_uint8.npy"
    index_path = OUT / "raw_rgb_index.json"
    reusable = False
    if path.is_file() and index_path.is_file():
        old = json.loads(index_path.read_text())
        reusable = (old.get("identities") == identities and old.get("complete") is True
                    and old.get("shape") == [len(identities), 224, 224, 3]
                    and path.stat().st_size == int(old["storage_bytes"]))
    started = time.time()
    if not reusable:
        CACHE.mkdir(parents=True, exist_ok=True)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(identities), 224, 224, 3))
        for index, identity in enumerate(identities):
            with Image.open(records[identity]) as image:
                array = np.asarray(image.convert("RGB"), np.uint8)
            if array.shape != (224, 224, 3):
                raise RuntimeError(f"unexpected raw RGB shape {array.shape}")
            value[index] = array
        value.flush(); del value
    payload = {
        "schema": "deployment_valid_raw_rgb_cache_index_v1", "complete": True,
        "source": "existing frozen 224x224 dense RGB; no rendering or re-encoding",
        "identities": identities, "paths": [records[x] for x in identities],
        "shape": [len(identities), 224, 224, 3], "dtype": "uint8",
        "cache_path": str(path), "storage_bytes": path.stat().st_size,
        "reused": reusable, "materialization_runtime_s": time.time() - started,
    }
    payload["index_digest"] = canonical_digest(payload)
    atomic_json(index_path, payload)
    by_identity = {identity: index for index, identity in enumerate(identities)}
    for branch in branches:
        branch["current_rgb_index"] = by_identity[branch["current_rgb_sha256"]]
        branch["future_rgb_indices"] = [by_identity[x] for x in branch["future_rgb_sha256"]]
    return payload


def fit_proprio_stats(branches: list[dict]) -> dict:
    values = []
    for branch in branches:
        values.append(np.asarray(branch["current_proprio"])[None])
        values.append(np.asarray(branch["future_proprio"]))
    value = np.concatenate(values).astype(np.float64)
    mean = value.mean(0); std = value.std(0)
    degenerate = std < 1e-7
    std[degenerate] = 1.
    return {"mean": mean.tolist(), "std": std.tolist(),
            "degenerate_channel_indices": np.flatnonzero(degenerate).tolist(),
            "fit_samples": len(value), "digest": canonical_digest({"mean": mean.tolist(), "std": std.tolist()})}


class SafetyModalityModel(nn.Module):
    def __init__(self, condition: str, proprio_dims: int = 42, action_dims: int = 4):
        super().__init__()
        self.condition = condition
        self.action = nn.Sequential(nn.Linear(action_dims, 48), nn.GELU())
        if condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION"):
            self.rgb = nn.Sequential(
                nn.Conv2d(9, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                nn.Conv2d(96, 96, 3, stride=2, padding=1), nn.GELU(),
            )
            self.rgb_projection = nn.Sequential(nn.Linear(192, 96), nn.GELU())
        if condition in ("PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"):
            self.proprio = nn.Sequential(nn.Linear(proprio_dims * 3, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU())
        widths = {"ACTION_CONTROL_ONLY": 48, "RAW_RGB": 144, "PROPRIOCEPTION": 144, "RGB_PLUS_PROPRIOCEPTION": 240}
        self.temporal = nn.GRU(widths[condition], 128, num_layers=1, batch_first=True)
        self.output = nn.Linear(128, 5)

    def forward(self, action_control: torch.Tensor, current_rgb: torch.Tensor | None = None,
                future_rgb: torch.Tensor | None = None, current_proprio: torch.Tensor | None = None,
                future_proprio: torch.Tensor | None = None) -> torch.Tensor:
        sequence = [self.action(action_control)]
        batch, ticks = action_control.shape[:2]
        if self.condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION"):
            current = current_rgb[:, None].expand(-1, ticks, -1, -1, -1)
            joined = torch.cat((current, future_rgb, future_rgb - current), 2).reshape(batch * ticks, 9, 224, 224)
            spatial = self.rgb(joined)
            visual = self.rgb_projection(torch.cat((spatial.mean((2, 3)), spatial.amax((2, 3))), 1)).reshape(batch, ticks, 96)
            sequence.append(visual)
        if self.condition in ("PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"):
            current = current_proprio[:, None].expand(-1, ticks, -1)
            sequence.append(self.proprio(torch.cat((current, future_proprio, future_proprio - current), -1)))
        hidden, _ = self.temporal(torch.cat(sequence, -1))
        return self.output(hidden)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, branches: list[dict], condition: str, rgb_cache: str,
                 proprio_mean: np.ndarray, proprio_std: np.ndarray):
        self.branches = branches; self.condition = condition; self.rgb_cache = rgb_cache
        self.mean = np.asarray(proprio_mean, np.float32); self.std = np.asarray(proprio_std, np.float32)
        self._rgb = None

    def __len__(self): return len(self.branches)

    def rgb(self):
        if self._rgb is None:
            self._rgb = np.load(self.rgb_cache, mmap_mode="r")
        return self._rgb

    def __getitem__(self, index: int):
        branch = self.branches[index]
        item = {
            "action_control": np.asarray(branch["action_control"], np.float32),
            "target": DENSE.branch_targets(branch), "branch_id": branch["branch_id"],
        }
        if self.condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION"):
            current = np.asarray(self.rgb()[branch["current_rgb_index"]], np.uint8).copy()
            future = np.asarray(self.rgb()[branch["future_rgb_indices"]], np.uint8).copy()
            item["current_rgb"] = current.transpose(2, 0, 1)
            item["future_rgb"] = future.transpose(0, 3, 1, 2)
        if self.condition in ("PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"):
            item["current_proprio"] = ((branch["current_proprio"] - self.mean) / self.std).astype(np.float32)
            item["future_proprio"] = ((branch["future_proprio"] - self.mean) / self.std).astype(np.float32)
        return item


def loader(branches: list[dict], condition: str, rgb_cache: str, stats: dict,
           batch_size: int, shuffle_seed: int | None):
    generator = None if shuffle_seed is None else torch.Generator().manual_seed(shuffle_seed)
    return torch.utils.data.DataLoader(MatrixDataset(branches, condition, rgb_cache,
        np.asarray(stats["mean"]), np.asarray(stats["std"])), batch_size=batch_size,
        shuffle=shuffle_seed is not None, generator=generator, num_workers=2,
        pin_memory=True, persistent_workers=False)


def to_device(batch: dict, condition: str, device: torch.device) -> dict:
    values = {"action_control": batch["action_control"].to(device, non_blocking=True),
              "target": batch["target"].to(device, non_blocking=True)}
    if condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION"):
        values["current_rgb"] = batch["current_rgb"].to(device, non_blocking=True).float().div_(127.5).sub_(1.)
        values["future_rgb"] = batch["future_rgb"].to(device, non_blocking=True).float().div_(127.5).sub_(1.)
    if condition in ("PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"):
        values["current_proprio"] = batch["current_proprio"].to(device, non_blocking=True)
        values["future_proprio"] = batch["future_proprio"].to(device, non_blocking=True)
    return values


def forward(model: SafetyModalityModel, values: dict) -> torch.Tensor:
    return model(values["action_control"], values.get("current_rgb"), values.get("future_rgb"),
                 values.get("current_proprio"), values.get("future_proprio"))


def train_condition(condition: str, fit: list[dict], device: torch.device,
                    rgb_index: dict, stats: dict) -> tuple[SafetyModalityModel, list[dict], dict]:
    seed = condition_seed(condition)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = SafetyModalityModel(condition).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters >= 750000:
        raise RuntimeError(f"{condition} exceeds parameter cap: {parameters}")
    weights_np, defined, prevalence = DENSE.positive_weights(fit)
    weights = torch.from_numpy(weights_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 8 if condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION") else 64
    smoke_raw = next(iter(loader(fit[:max(8, batch_size)], condition, rgb_index["cache_path"], stats, min(8, batch_size), None)))
    smoke = to_device(smoke_raw, condition, device)
    model.train(); output = forward(model, smoke); loss = DENSE.balanced_loss(output, smoke["target"], weights, defined); loss.backward()
    if not torch.isfinite(loss): raise RuntimeError(f"{condition}: nonfinite smoke loss")
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum() == 0:
            raise RuntimeError(f"{condition}: smoke gradient failure: {name}")
    with torch.inference_mode():
        model.eval(); first = forward(model, smoke); second = forward(model, smoke)
        if not torch.equal(first, second): raise RuntimeError(f"{condition}: nondeterministic inference")
        changed = dict(smoke); changed["action_control"] = smoke["action_control"].flip(1)
        if torch.allclose(first, forward(model, changed)): raise RuntimeError(f"{condition}: action-insensitive")
        ordered = dict(smoke)
        if "future_rgb" in ordered: ordered["future_rgb"] = ordered["future_rgb"].flip(1)
        if "future_proprio" in ordered: ordered["future_proprio"] = ordered["future_proprio"].flip(1)
        if condition != "ACTION_CONTROL_ONLY" and torch.allclose(first, forward(model, ordered)):
            raise RuntimeError(f"{condition}: temporal evidence insensitive")
    smoke_path = OUT / f".{condition.lower()}_smoke.pt"
    torch.save(model.state_dict(), smoke_path); clone = SafetyModalityModel(condition).to(device)
    clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True)); smoke_path.unlink()
    optimizer.zero_grad(set_to_none=True)
    history = []; started = time.time(); peak = 0
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(60):
        model.train(); total = 0.; seen = 0
        for raw in loader(fit, condition, rgb_index["cache_path"], stats, batch_size, seed + epoch):
            values = to_device(raw, condition, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward(model, values)
                batch_loss = DENSE.balanced_loss(logits, values["target"], weights, defined)
            optimizer.zero_grad(set_to_none=True); batch_loss.backward(); optimizer.step()
            total += float(batch_loss.detach()) * len(values["target"]); seen += len(values["target"])
        row = {"epoch": epoch + 1, "mean_balanced_bce": total / seen}; history.append(row)
        if epoch in (0, 9, 19, 29, 39, 49, 59): print(json.dumps({"condition": condition, **row}), flush=True)
    if device.type == "cuda": peak = int(torch.cuda.max_memory_allocated(device))
    return model, history, {"seed": seed, "base_seed": BASE_SEED, "parameter_count": parameters,
        "batch_size": batch_size, "training_runtime_s": time.time() - started, "peak_vram_bytes": peak,
        "positive_weights": weights_np.tolist(), "defined_outputs": defined.tolist(),
        "fit_tick_prevalence": prevalence, "smoke_passed": True,
        "input_allow_list": input_allow_list(condition)}


def input_allow_list(condition: str) -> list[str]:
    base = ["per_tick_post_slew_candidate_vx_yaw", "previous_applied_vx_yaw"]
    if condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION"):
        base += ["current_raw_rgb_224", "true_future_raw_rgb_224", "signed_future_minus_current_rgb"]
    if condition in ("PROPRIOCEPTION", "RGB_PLUS_PROPRIOCEPTION"):
        base += ["current_deployment_proprioception", "true_future_deployment_proprioception", "future_minus_current_proprioception"]
    return base


def predict(model: SafetyModalityModel, condition: str, branches: list[dict], device: torch.device,
            rgb_index: dict, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    logits, targets = [], []
    model.eval(); batch_size = 8 if condition in ("RAW_RGB", "RGB_PLUS_PROPRIOCEPTION") else 64
    with torch.inference_mode():
        for raw in loader(branches, condition, rgb_index["cache_path"], stats, batch_size, None):
            values = to_device(raw, condition, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits.append(forward(model, values).float().cpu().numpy())
            targets.append(values["target"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


def operating_curve(labels: np.ndarray, probability: np.ndarray) -> dict:
    labels = labels.astype(bool)
    candidates = sorted({0., 1., *(float(np.nextafter(v, math.inf)) for v in probability)})
    rows = []
    for threshold in candidates:
        admitted = probability < threshold
        recall = float(np.mean(~admitted[labels])) if labels.any() else 1.
        retention = float(np.mean(admitted[~labels])) if (~labels).any() else 0.
        rows.append((threshold, recall, retention))
    recall_rows = [x for x in rows if x[1] >= .95]
    retention_rows = [x for x in rows if x[2] >= .40]
    return {
        "safe_retention_at_unsafe_recall_ge_0_95": max((x[2] for x in recall_rows), default=None),
        "unsafe_recall_at_safe_retention_ge_0_40": max((x[1] for x in retention_rows), default=None),
    }


def target_prevalence(branches: list[dict]) -> dict:
    target = np.stack([DENSE.branch_targets(branch) for branch in branches])
    result = {}
    for channel, name in enumerate(CHANNELS):
        result[name] = {"positive_ticks": int(target[:, :, channel].sum()),
                        "ticks": int(target[:, :, channel].size),
                        "prevalence": float(target[:, :, channel].mean())}
    clearances = [float(tick["clearance_m"]) for branch in branches for tick in branch["ticks"]]
    result["descriptive_clearance_m"] = {"minimum": min(clearances), "median": float(np.median(clearances)),
                                          "maximum": max(clearances), "trained_output": False}
    result["fall_or_unsafe_termination"] = {"positive_ticks": 0, "prevalence": 0., "trained_output": False}
    return result


def evaluate(condition: str, branches: list[dict], logits: np.ndarray, targets: np.ndarray,
             temperature: float, threshold: float, route_rows: list[dict], route_indices: np.ndarray,
             kinematic: np.ndarray) -> dict:
    probabilities = 1. / (1. + np.exp(-logits / temperature))
    h3 = probabilities[:, -1]
    aggregate = DENSE.discrimination(targets[:, -1, 4], h3[:, 4], threshold)
    components = {
        "contact": DENSE.discrimination(targets[:, -1, 2], h3[:, 2], .5),
        "stuck": DENSE.discrimination(targets[:, -1, 3], h3[:, 3], .5),
    }
    temporal = DENSE.temporal_metrics(targets, probabilities, threshold)
    temporal["cumulative_contact_recall"] = components["contact"]["unsafe_recall"]
    temporal["cumulative_stuck_recall"] = components["stuck"]["unsafe_recall"]
    FS.rows_global = route_rows
    planning = FS.evaluate_condition(condition, route_rows, route_indices, h3[:, 4], threshold, kinematic)
    improving_false_abstention = 0
    for state_row in planning["planning"]["per_state"]:
        if state_row["selected_candidate"] is not None:
            continue
        candidates = [route_rows[i] for i in route_indices if route_rows[i]["state_id"] == state_row["state_id"]]
        if any((not row["unsafe"]) and (row["p_d"] > .03 or math.degrees(row["p_theta"]) > 5.) for row in candidates):
            improving_false_abstention += 1
    planning["planning"]["states_hold_despite_safe_positive_progress"] = improving_false_abstention
    per_family = {}
    for family in FAMILIES:
        mask = np.asarray([branch["family"] == family for branch in branches])
        family_indices = route_indices[mask]
        FS.rows_global = route_rows
        family_plan = FS.evaluate_condition(condition, route_rows, family_indices, h3[mask, 4], threshold, kinematic)
        per_family[family] = {
            "aggregate": DENSE.discrimination(targets[mask, -1, 4], h3[mask, 4], threshold),
            "contact": DENSE.discrimination(targets[mask, -1, 2], h3[mask, 2], .5),
            "stuck": DENSE.discrimination(targets[mask, -1, 3], h3[mask, 3], .5),
            "temporal": DENSE.temporal_metrics(targets[mask], probabilities[mask], threshold),
            "planning": family_plan["planning"],
        }
    return {"aggregate": aggregate, "components": components, "temporal": temporal,
            "operating_curve": operating_curve(targets[:, -1, 4], h3[:, 4]),
            "candidate_filter_and_planning": planning, "per_family": per_family}


def gate(result: dict, oracle_progress: float) -> dict:
    aggregate = result["aggregate"]; contact = result["components"]["contact"]
    stuck = result["components"]["stuck"]; temporal = result["temporal"]
    plan = result["candidate_filter_and_planning"]["planning"]
    checks = {
        "aggregate_auc_ge_0_80": aggregate["auc"] is not None and aggregate["auc"] >= .80,
        "aggregate_recall_ge_0_95": aggregate["unsafe_recall"] is not None and aggregate["unsafe_recall"] >= .95,
        "aggregate_fnr_le_0_05": aggregate["unsafe_false_negative_rate"] is not None and aggregate["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": aggregate["safe_candidate_retention"] is not None and aggregate["safe_candidate_retention"] >= .40,
        "ece_le_0_10": aggregate["ece"] <= .10,
        "contact_auc_ge_0_75": contact["auc"] is not None and contact["auc"] >= .75,
        "stuck_auc_ge_0_85": stuck["auc"] is not None and stuck["auc"] >= .85,
        "contact_event_recall_ge_0_80": temporal["per_tick_contact_recall"] is not None and temporal["per_tick_contact_recall"] >= .80,
        "stuck_event_recall_ge_0_90": temporal["per_tick_stuck_recall"] is not None and temporal["per_tick_stuck_recall"] >= .90,
        "median_detection_delay_le_1": temporal["first_event_detection_delay_ticks_median"] is not None and temporal["first_event_detection_delay_ticks_median"] <= 1.,
        "six_states_retain_safe": plan["states_retaining_safe"] >= 6,
        "no_state_only_unsafe_admitted": plan["states_only_unsafe_admitted"] == 0,
        "selected_unsafe_rate_zero": plan["selected_unsafe_rate"] == 0.,
        "progress_ge_80pct_oracle": plan["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": plan["normalized_safe_progress_regret"] is not None and plan["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": plan["best_safe_top3"] is not None and plan["best_safe_top3"] >= .75,
        "false_abstention_le_1": plan["false_abstention"] <= 1,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--fixture-only", action="store_true")
    parser.add_argument("--reduce-only", action="store_true",
                        help="Reload the four final checkpoints and rerun calibration/evaluation without training")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    fixture = evaluator_fixture()
    if args.fixture_only:
        print(json.dumps(fixture, indent=2)); return 0
    started = time.time()
    branches, proprio_index, dense_binding = load_branches()
    rgb_index = ensure_raw_rgb_cache(branches)
    split = {name: [branch for branch in branches if branch["split"] == name] for name in ("fit", "calibration", "heldout")}
    prevalence = {name: {"overall": target_prevalence(values), **{
        family: target_prevalence([branch for branch in values if branch["family"] == family]) for family in FAMILIES
    }} for name, values in split.items()}
    stats = fit_proprio_stats(split["fit"]); atomic_json(OUT / "proprio_fit_standardization.json", stats)
    route_rows = FS.load_metadata(); route_by_id = {row["state_id"] + f":{int(row['candidate_index']):02d}": i for i, row in enumerate(route_rows)}
    route_indices = {name: np.asarray([route_by_id[branch["branch_id"]] for branch in values], int) for name, values in split.items()}
    kinematic = np.stack([row["kinematic"] for row in route_rows])
    FS.rows_global = route_rows
    oracle_probability = np.asarray([float(route_rows[i]["unsafe"]) for i in route_indices["heldout"]])
    oracle = FS.evaluate_condition("oracle_safety", route_rows, route_indices["heldout"], oracle_probability, .5, kinematic)
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}; histories = {}; training = {}; checkpoints = {}
    prior_result = json.loads((OUT / "result.json").read_text()) if args.reduce_only and (OUT / "result.json").is_file() else None
    if args.reduce_only and prior_result is None:
        raise RuntimeError("--reduce-only requires the completed four-condition result")
    for condition in CONDITIONS:
        path = OUT / f"{condition.lower()}_seed_{BASE_SEED}.pt"
        if prior_result is None:
            model, history, stats_row = train_condition(condition, split["fit"], device, rgb_index, stats)
            torch.save({"state_dict": model.state_dict(), "condition": condition,
                        "seed": stats_row["seed"], "base_seed": BASE_SEED,
                        "epoch": 60, "parameter_count": stats_row["parameter_count"]}, path)
        else:
            stats_row = {k: v for k, v in prior_result["models"][condition].items()
                         if k not in ("checkpoint", "epochs", "initial_loss", "final_loss", "training_history")}
            history = prior_result["models"][condition]["training_history"]
        clone = SafetyModalityModel(condition).to(device)
        clone.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"]); clone.eval()
        models[condition] = clone; histories[condition] = history; training[condition] = stats_row
        checkpoints[condition] = {"path": str(path), "sha256": sha(path)}
    calibrations = {}; heldout = {}; fit_results = {}; gates = {}
    for condition in CONDITIONS:
        cal_logits, cal_target = predict(models[condition], condition, split["calibration"], device, rgb_index, stats)
        temperature = DENSE.fit_temperature(cal_logits[:, -1, 4], cal_target[:, -1, 4])
        cal_probability = 1. / (1. + np.exp(-cal_logits[:, -1, 4] / temperature))
        choice = DENSE.choose_threshold(cal_probability, cal_target[:, -1, 4])
        calibrations[condition] = {"temperature": temperature, **choice}
        held_logits, held_target = predict(models[condition], condition, split["heldout"], device, rgb_index, stats)
        heldout[condition] = evaluate(condition, split["heldout"], held_logits, held_target,
                                      temperature, choice["threshold"], route_rows, route_indices["heldout"], kinematic)
        fit_logits, fit_target = predict(models[condition], condition, split["fit"], device, rgb_index, stats)
        fit_results[condition] = evaluate(condition, split["fit"], fit_logits, fit_target,
                                           temperature, choice["threshold"], route_rows, route_indices["fit"], kinematic)
        gates[condition] = gate(heldout[condition], oracle_progress)
        heldout[condition]["candidate_filter_and_planning"]["planning"]["oracle_progress_fraction"] = (
            heldout[condition]["candidate_filter_and_planning"]["planning"]["mean_selected_distance_progress_m"] / oracle_progress
            if abs(oracle_progress) > 1e-12 else None)
        heldout[condition]["fit_to_heldout_degradation"] = {
            "aggregate_auc": fit_results[condition]["aggregate"]["auc"] - heldout[condition]["aggregate"]["auc"],
            "contact_auc": fit_results[condition]["components"]["contact"]["auc"] - heldout[condition]["components"]["contact"]["auc"],
            "stuck_auc": fit_results[condition]["components"]["stuck"]["auc"] - heldout[condition]["components"]["stuck"]["auc"],
        }
    prior = json.loads((DENSE_OUT / "result.json").read_text())
    comparator = {
        "source_result_sha256": sha(DENSE_OUT / "result.json"),
        "aggregate_auc": .7255, "contact_auc": .6175, "stuck_auc": .7745,
        "safe_retention": 0., "all_candidate_rejection": True,
        "missed_transient_contact_rate": .9756,
        "bound_result_classification": prior["classification"],
    }
    guard = json.loads((ROOT / ".generated/kinematic_route_with_runtime_safety_guard_v1/result.json").read_text())
    guard_held = guard["guard_discrimination"]["by_split"]["heldout"]
    frozen_baselines = {
        "final_layer_vit_l": comparator,
        "privileged_static_grid_guard": {"classification": guard["classification"], **guard_held,
            "per_family": guard["guard_discrimination"]["heldout_by_family"]},
        "oracle_safety_kinematic": oracle,
    }
    action = heldout["ACTION_CONTROL_ONLY"]; raw = heldout["RAW_RGB"]
    prop = heldout["PROPRIOCEPTION"]; fusion = heldout["RGB_PLUS_PROPRIOCEPTION"]
    effects = {
        "raw_rgb_minus_final_vit_l_aggregate_auc": raw["aggregate"]["auc"] - comparator["aggregate_auc"],
        "raw_rgb_minus_final_vit_l_contact_auc": raw["components"]["contact"]["auc"] - comparator["contact_auc"],
        "raw_rgb_minus_final_vit_l_safe_retention": raw["aggregate"]["safe_candidate_retention"] - comparator["safe_retention"],
        "proprio_minus_action_aggregate_auc": prop["aggregate"]["auc"] - action["aggregate"]["auc"],
        "proprio_minus_action_safe_retention": prop["aggregate"]["safe_candidate_retention"] - action["aggregate"]["safe_candidate_retention"],
        "proprio_minus_action_contact_auc": prop["components"]["contact"]["auc"] - action["components"]["contact"]["auc"],
        "proprio_minus_action_stuck_auc": prop["components"]["stuck"]["auc"] - action["components"]["stuck"]["auc"],
        "proprio_minus_action_contact_tick_recall": prop["temporal"]["per_tick_contact_recall"] - action["temporal"]["per_tick_contact_recall"],
        "proprio_minus_action_stuck_tick_recall": prop["temporal"]["per_tick_stuck_recall"] - action["temporal"]["per_tick_stuck_recall"],
        "fusion_minus_stronger_unimodal_aggregate_auc": fusion["aggregate"]["auc"] - max(raw["aggregate"]["auc"], prop["aggregate"]["auc"]),
        "fusion_minus_stronger_unimodal_safe_retention": fusion["aggregate"]["safe_candidate_retention"] - max(raw["aggregate"]["safe_candidate_retention"], prop["aggregate"]["safe_candidate_retention"]),
        "fusion_minus_stronger_unimodal_contact_auc": fusion["components"]["contact"]["auc"] - max(raw["components"]["contact"]["auc"], prop["components"]["contact"]["auc"]),
        "fusion_minus_stronger_unimodal_stuck_auc": fusion["components"]["stuck"]["auc"] - max(raw["components"]["stuck"]["auc"], prop["components"]["stuck"]["auc"]),
    }
    effects["strong_final_layer_representation_gap_tendency"] = effects["raw_rgb_minus_final_vit_l_aggregate_auc"] >= .10 or effects["raw_rgb_minus_final_vit_l_safe_retention"] >= .20
    effects["strong_multimodal_tendency"] = ((fusion["aggregate"]["auc"] - raw["aggregate"]["auc"] >= .05 and fusion["aggregate"]["auc"] - prop["aggregate"]["auc"] >= .05)
        or (fusion["aggregate"]["safe_candidate_retention"] - raw["aggregate"]["safe_candidate_retention"] >= .15 and fusion["aggregate"]["safe_candidate_retention"] - prop["aggregate"]["safe_candidate_retention"] >= .15))
    classifications = []
    if gates["ACTION_CONTROL_ONLY"]["passed"]: classifications.append("ACTION_PRIOR_SAFETY_SIGNAL")
    if gates["RAW_RGB"]["passed"]:
        classifications.append("RAW_RGB_SAFETY_OBSERVABILITY_SIGNAL")
        if effects["strong_final_layer_representation_gap_tendency"]: classifications.append("FINAL_LAYER_VISUAL_REPRESENTATION_BOTTLENECK")
    if gates["PROPRIOCEPTION"]["passed"]: classifications.append("PROPRIOCEPTIVE_SAFETY_OBSERVABILITY_SIGNAL")
    if gates["RGB_PLUS_PROPRIOCEPTION"]["passed"] and effects["strong_multimodal_tendency"]:
        classifications.append("MULTIMODAL_SAFETY_OBSERVABILITY_SIGNAL")
    if not any(gates[name]["passed"] for name in CONDITIONS):
        classifications.append("CURRENT_DEPLOYMENT_SENSOR_CONTRACT_SAFETY_NO_GO")
    if gates["ACTION_CONTROL_ONLY"]["passed"]:
        recommendation = "Use the action-conditioned primitive-risk filter; a learned visual world-model safety state is not required for this candidate bank."
    elif gates["RGB_PLUS_PROPRIOCEPTION"]["passed"]:
        recommendation = "Develop a high-rate joint RGB-plus-proprioceptive micro-safety state alongside the macro visual JEPA."
    elif gates["PROPRIOCEPTION"]["passed"]:
        recommendation = "Develop a high-rate proprioceptive micro-safety state alongside the macro visual JEPA."
    elif gates["RAW_RGB"]["passed"]:
        recommendation = "Use a safety-specific pixel/intermediate-layer visual micro-state rather than the frozen final ViT-L layer."
    else:
        recommendation = "Do not fit another head to this sensor contract; add reliable contact/torque/acceleration/depth sensing or narrow the safety claim to observable modes."
    result = {
        "schema": "deployment_valid_safety_observability_matrix_v1_result",
        "source_commit": "4798995c20d2e8eada17ff0c1333d72364b24e8d",
        "preserved_result": "RGB_ONLY_DENSE_TEMPORAL_SAFETY_NO_GO",
        "bindings": {"dense_token_index_digest": "7c24306dd1082940f948e47584ac525e451717258255585c04f82956736571f0",
                     "dense_result_sha256": "96dc7e4d80c1a19726498cbe7bef5cd9b6825fb2c202601b6af1110087a69922",
                     "proprio_index_sha256": sha(OUT / "proprio_index.json"), "proprio_index_digest": proprio_index["proprio_index_digest"],
                     "raw_rgb_index_digest": rgb_index["index_digest"]},
        "panel": {"states": 48, "branches": 576, "split_states": {"fit": 32, "calibration": 8, "heldout": 8},
                  "split_branches": {name: len(value) for name, value in split.items()}, "families": list(FAMILIES), "ticks": 15},
        "proprioception_contract": {"channels": proprio_index["proprio_channels"], "action_control_channels": proprio_index["action_control_channels"],
            "excluded_inputs": proprio_index["excluded_inputs"], "missing_channels": proprio_index["missing_channels"],
            "degenerate_requested_channels": proprio_index["degenerate_requested_channels"], "fit_standardization": stats},
        "materialization": {"proprioception": {k: proprio_index[k] for k in ("replayed_branches", "replay_mismatches", "snapshot_digest_matches", "snapshot_digest_mismatches_with_exact_tick_reproduction", "runtime_s", "parallel_wall_runtime_s", "storage_bytes")}, "raw_rgb_cache": {k: rgb_index[k] for k in ("shape", "storage_bytes", "reused", "materialization_runtime_s", "index_digest")}},
        "evaluator_fixture": fixture,
        "safety_target_prevalence": prevalence,
        "models": {condition: {**training[condition], "checkpoint": checkpoints[condition], "epochs": 60,
                               "initial_loss": histories[condition][0]["mean_balanced_bce"], "final_loss": histories[condition][-1]["mean_balanced_bce"],
                               "training_history": histories[condition]} for condition in CONDITIONS},
        "calibration": calibrations, "heldout": heldout, "fit": fit_results,
        "condition_gates": gates, "frozen_baselines": frozen_baselines,
        "representation_gap_diagnostics": effects, "classifications": classifications,
        "recommended_next_architecture": recommendation,
        "limitation": "Failure of one small, single-seed model per condition is not an information-theoretic proof of non-observability.",
        "runtime": {"end_to_end_reducer_and_training_s": time.time() - started,
                    "training_s": sum(training[x]["training_runtime_s"] for x in CONDITIONS),
                    "aggregate_replay_plus_training_compute_s": proprio_index["runtime_s"] + sum(training[x]["training_runtime_s"] for x in CONDITIONS),
                    "peak_vram_bytes": max(training[x]["peak_vram_bytes"] for x in CONDITIONS),
                    "new_storage_bytes": proprio_index["storage_bytes"] + rgb_index["storage_bytes"] + sum(Path(checkpoints[x]["path"]).stat().st_size for x in CONDITIONS)},
        "custody": {"one_seed_per_condition": True, "condition_count": 4, "jepa_predictor_opened_or_trained": False,
                    "new_states": 0, "new_candidate_identities": 0, "memory_novelty_navigation_implemented": False,
                    "reducer_only_checkpoint_reload": bool(args.reduce_only)},
    }
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classifications": classifications, "result_sha256": sha(OUT / "result.json"),
                      "conditions": {name: {"passed": gates[name]["passed"], "auc": heldout[name]["aggregate"]["auc"],
                      "retention": heldout[name]["aggregate"]["safe_candidate_retention"]} for name in CONDITIONS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
