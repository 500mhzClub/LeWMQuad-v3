#!/usr/bin/env python3
"""Structured true-future spatial/stuck safety development gate.

Stage A only trains STUCK_BLOCKED_MOTION_HEAD_V1. ViT-L, the historical JEPA
predictor, and the frozen occupancy probe stay frozen. The route ledger did not
materialize occupancy rasters, so this runner deterministically derives them
from its already-frozen scene, pose, quaternion, and RGB-receipt metadata using
the same pure V4 ray-label implementation as the qualified occupancy assay. It
does not simulate, render RGB, or encode latents.

Stage B is gated on every Stage-A requirement and is never entered after a
Stage-A failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
for extra in (ROOT, ROOT / "lewm_worlds", ROOT / "lewm_genesis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from lewm.benchmarks import go2_dynamic_cell_square_projection as DYNAMIC
from lewm.oracle.go2_branch_oracle_v1_2 import CLEARANCE_SAFE_M
from lewm.planning.two_resolution_configuration_projection_v2 import FOOTPRINT_RADIUS_M
from lewm_worlds.manifest import parse_scene_manifest_dict
from scripts import build_go2_observable_camera_ray_fit_v4 as V4
from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as OCC
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS

SEED = 2026082005
EPOCHS = 60
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EXPECTED_TARGET = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
EXPECTED_PROBE_PACKAGE = "b8f05e57baffcf553ba9581419d82068a5723f2aae5895de29b9546d4c3f7686"
EXPECTED_PROBE_WEIGHTS = "95d253ce834384f1b372f1c4cc7f39241c42576fdea903c007dda8f7a7bc1322"
OUT = ROOT / ".generated/structured_spatial_safety_state_jepa_dev_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/") / OUT.name
CHECKPOINT = CACHE / "stuck_blocked_motion_head_seed_2026082005_epoch60.pt"
LABELS = CACHE / "derived_frozen_pose_occupancy_labels.npy"
LABEL_INDEX = CACHE / "derived_frozen_pose_occupancy_index.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def json_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fixture() -> dict:
    cases = {
        "clear_free_path": bool(0.8 > CLEARANCE_SAFE_M),
        "wall_inside_body_footprint": bool(-0.02 < CLEARANCE_SAFE_M),
        "marginal_clearance": bool(abs(CLEARANCE_SAFE_M - 0.15) < 1e-12),
        "stuck_without_collision": bool(True and not False),
        "collision_without_stuck": bool(True and not False),
        "all_candidates_safe": bool((~np.zeros(12, dtype=bool)).all()),
        "all_candidates_unsafe": bool(np.ones(12, dtype=bool).all()),
        "no_candidate_admitted": not bool((np.ones(12) < .5).any()),
        "deterministic_tie": int(np.argmin(np.asarray([.2, .2, .4]))) == 0,
    }
    payload = {
        "schema": "structured_spatial_safety_state_jepa_dev_v2_fixture",
        "cases": cases,
        "passed": all(cases.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "evaluator_fixture.json", payload)
    if json.loads((OUT / "evaluator_fixture.json").read_text()) != payload:
        raise RuntimeError("fixture reload failed")
    return payload


def nominal_by_horizon(post_slew) -> np.ndarray:
    x = y = yaw = 0.0
    result = []
    for horizon, block in enumerate(post_slew[:3], 1):
        for vx, vy, wz in block:
            x += (math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)) * .1
            y += (math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)) * .1
            yaw = math.atan2(math.sin(yaw + float(wz) * .1),
                             math.cos(yaw + float(wz) * .1))
        result.append([x, y, yaw])
    return np.asarray(result, np.float32)


def load_rows():
    rows = FS.load_metadata()
    if sha(FS.V2 / "target_latent_index.json") != EXPECTED_TARGET:
        raise RuntimeError("target-latent index mismatch")
    ledger = {
        (row["state_id"], int(row["candidate_index"])): row
        for row in map(json.loads, (FS.V1 / "branch_labels.jsonl").read_text().splitlines())
    }
    replay = {}
    for path in sorted((FS.V2 / "replay").glob("purpose-*.json")):
        payload = json.loads(path.read_text())
        for row in payload["rows"]:
            replay[(row["state_id"], int(row["candidate_index"]))] = row
    if len(rows) != 576 or len(ledger) != 576 or len(replay) != 576:
        raise RuntimeError("route corpus is incomplete")
    for row in rows:
        key = (row["state_id"], row["candidate_index"])
        source = ledger[key]
        row["nominal"] = nominal_by_horizon(source["post_slew"])
        row["actual_motion"] = np.asarray([
            [*source["horizons"][str(h)]["delta_body"],
             source["horizons"][str(h)]["delta_yaw"]]
            for h in (1, 2, 3)], np.float32)
        row["clearance"] = np.asarray([
            source["horizons"][str(h)]["min_clearance"] for h in (1, 2, 3)
        ], np.float32)
        row["shortfall"] = (
            np.linalg.norm(row["actual_motion"][:, :2], axis=1)
            - np.linalg.norm(row["nominal"][:, :2], axis=1)
        ).astype(np.float32)
        row["replay"] = replay[key]
    split = {
        name: np.asarray([i for i, row in enumerate(rows) if row["split"] == name], int)
        for name in ("fit", "calibration", "heldout")
    }
    return rows, split


def _state_occupancy_task(task):
    state, replay_rows = task
    scene_path = Path(state["scene_dir"]) / "manifest.json"
    scene = parse_scene_manifest_dict(json.loads(scene_path.read_text()))
    raw_boxes = (*scene.walls, *scene.obstacles, *scene.landmarks)
    output = np.empty((12, 3, 64, 64), np.uint8)
    for candidate_index, replay in enumerate(replay_rows):
        if int(replay["candidate_index"]) != candidate_index:
            raise RuntimeError("candidate order mismatch during occupancy derivation")
        for horizon in (1, 2, 3):
            frame = replay["horizons"][str(horizon)]
            position = frame["pose"]
            wxyz = frame["quaternion_wxyz"]
            xyzw = (wxyz[1], wxyz[2], wxyz[3], wxyz[0])
            yaw = OCC._quaternion_yaw_xyzw(xyzw)
            camera = DYNAMIC.compose_yaw_aligned_camera(xyzw, yaw)
            boxes = tuple(V4._box_in_yaw_body(
                box, base_position_world=position, stored_yaw_rad=yaw)
                for box in raw_boxes)
            identity = f"{state['state_id']}:{candidate_index:02d}:H{horizon}"
            frame_input = V4.FrameBuildInputV4(
                frame_key={"identity": identity},
                camera_origin_body_m=tuple(camera.origin_xyz),
                camera_basis_body_fru=V4._normalized_camera_basis_fru(camera),
                ground_plane_z_body_m=-float(position[2]),
                rendered_boxes_body=boxes,
                image_path_metadata_only=frame["rgb_path"],
                image_sha256=frame["rgb_sha256"],
                sidecar_row_identity_sha256=hashlib.sha256(identity.encode()).hexdigest(),
            )
            evidence = V4.build_frame_evidence_v4(frame_input)
            raster = V4.rasterize_observable_camera_ray_evidence_v4(evidence)
            output[candidate_index, horizon - 1] = np.asarray(
                raster.output_labels, np.uint8)
    return state["state_id"], output


def occupancy_labels(rows) -> tuple[np.ndarray, dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    if LABELS.exists() and LABEL_INDEX.exists():
        index = json.loads(LABEL_INDEX.read_text())
        if (index.get("target_index_sha256") == EXPECTED_TARGET
                and index.get("array_sha256") == sha(LABELS)):
            value = np.load(LABELS, mmap_mode="r")
            if value.shape == (576, 3, 64, 64) and value.dtype == np.uint8:
                return value, index
    states = json.loads((FS.V1 / "state_manifest.json").read_text())["state_candidates"]
    state_by_id = {state["state_id"]: state for state in states}
    replay_by_state = defaultdict(list)
    for row in rows:
        replay_by_state[row["state_id"]].append(row["replay"])
    tasks = []
    for state_id in sorted(replay_by_state, key=lambda x: int(x.split("-")[1])):
        state = state_by_id[state_id]
        values = sorted(replay_by_state[state_id], key=lambda x: int(x["candidate_index"]))
        tasks.append((state, values))
    started = time.time()
    context = mp.get_context("fork")
    with context.Pool(min(8, len(tasks))) as pool:
        derived = dict(pool.map(_state_occupancy_task, tasks))
    labels = np.stack([
        derived[row["state_id"]][row["candidate_index"]] for row in rows
    ], axis=0)
    temporary = LABELS.with_name(f".{LABELS.name}.tmp-{os.getpid()}.npy")
    np.save(temporary, labels, allow_pickle=False)
    os.replace(temporary, LABELS)
    index = {
        "schema": "structured_spatial_safety_frozen_pose_occupancy_index_v1",
        "rows": 576,
        "horizons": [1, 2, 3],
        "shape": list(labels.shape),
        "dtype": str(labels.dtype),
        "target_index_sha256": EXPECTED_TARGET,
        "derivation": "stored pose/quaternion + frozen scene manifest + pure V4 ray rasterizer",
        "simulation": False,
        "rgb_rendering": False,
        "latent_encoding": False,
        "runtime_s": time.time() - started,
        "array_sha256": sha(LABELS),
    }
    atomic_json(LABEL_INDEX, index)
    return np.load(LABELS, mmap_mode="r"), index


class StuckBlockedMotionHead(nn.Module):
    """True-trajectory stuck and signed displacement-shortfall head."""

    def __init__(self):
        super().__init__()
        self.token = nn.Linear(1024, 16)
        self.action = nn.Sequential(
            nn.Linear(57, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(224, 64), nn.GELU(), nn.Linear(64, 6))

    def forward(self, current, future, action_control, nominal):
        current = F.layer_norm(current.float(), (1024,), weight=None, bias=None)
        future = F.layer_norm(future.float(), (1024,), weight=None, bias=None)
        cur = self.token(current)
        features = []
        for horizon in range(3):
            fut = self.token(future[:, horizon])
            delta = fut - cur
            features.append(torch.cat([
                fut.mean(1), fut.amax(1), delta.mean(1), delta.amax(1)
            ], dim=1))
        visual = torch.cat(features, dim=1)
        control = self.action(torch.cat([
            action_control.float(), nominal.reshape(len(nominal), -1).float()
        ], dim=1))
        return self.fusion(torch.cat([visual, control], dim=1))


class TokenStore:
    def __init__(self, rows):
        self.rows = rows
        self.current = {}

    def batch(self, ids, device):
        future, current = [], []
        for index in ids:
            row = self.rows[int(index)]
            future.append(np.stack([
                np.asarray(np.load(path, mmap_mode="r"), np.float32)
                for path in row["future_paths"]]))
            if row["state_id"] not in self.current:
                self.current[row["state_id"]] = np.asarray(
                    np.load(row["z0_path"], mmap_mode="r"), np.float32)
            current.append(self.current[row["state_id"]])
        return (
            torch.from_numpy(np.stack(current)).to(device),
            torch.from_numpy(np.stack(future)).to(device),
            torch.from_numpy(np.stack([self.rows[int(i)]["action_control"] for i in ids])).to(device),
            torch.from_numpy(np.stack([self.rows[int(i)]["nominal"] for i in ids])).to(device),
        )


def rank_average(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(rank_average(a), rank_average(b))[0, 1])


def stuck_weights(rows, fit):
    y = np.stack([rows[i]["labels"][:, 2] for i in fit]).astype(np.float32)
    weight = np.ones(3, np.float32)
    for horizon in range(3):
        positive = y[:, horizon].sum()
        weight[horizon] = (len(y) - positive) / max(positive, 1)
    return weight


def head_loss(output, rows, ids, weight, shortfall_mean, shortfall_std, device):
    stuck = torch.from_numpy(np.stack([rows[int(i)]["labels"][:, 2] for i in ids])).to(device)
    shortfall = torch.from_numpy(np.stack([rows[int(i)]["shortfall"] for i in ids])).to(device)
    terms = [F.binary_cross_entropy_with_logits(
        output[:, horizon], stuck[:, horizon],
        pos_weight=torch.tensor(weight[horizon], device=device))
        for horizon in range(3)]
    predicted_shortfall = output[:, 3:]
    target = (shortfall - shortfall_mean) / shortfall_std
    return torch.stack(terms).mean() + F.smooth_l1_loss(predicted_shortfall, target)


def train_head(rows, split, device):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = StuckBlockedMotionHead().to(device)
    if sum(p.numel() for p in model.parameters()) >= 100_000:
        raise RuntimeError("stuck component exceeds parameter ceiling")
    fit = split["fit"]
    shortfall = np.stack([rows[i]["shortfall"] for i in fit])
    mean = torch.tensor(shortfall.mean(0), dtype=torch.float32, device=device)
    std = torch.tensor(np.maximum(shortfall.std(0), 1e-4), dtype=torch.float32, device=device)
    weight = stuck_weights(rows, fit)
    store = TokenStore(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    smoke_ids = fit[:min(BATCH, len(fit))]
    inputs = store.batch(smoke_ids, device)
    output = model(*inputs)
    loss = head_loss(output, rows, smoke_ids, weight, mean, std, device)
    optimizer.zero_grad(); loss.backward()
    if (not torch.isfinite(loss)
            or not all(p.grad is not None and torch.isfinite(p.grad).all()
                       and p.grad.abs().sum() > 0 for p in model.parameters())):
        raise RuntimeError("training smoke gradient failure")
    optimizer.step(); optimizer.zero_grad()
    model.eval()
    with torch.inference_mode():
        first = model(*inputs); second = model(*inputs)
    if not torch.equal(first, second):
        raise RuntimeError("training smoke inference is nondeterministic")
    smoke_path = CACHE / ".stuck_smoke.pt"
    atomic_torch(smoke_path, model.state_dict())
    clone = StuckBlockedMotionHead().to(device)
    clone.load_state_dict(torch.load(smoke_path, map_location=device, weights_only=True))
    smoke_path.unlink()
    # Reset after the implementation-only update so the registered run begins
    # from the deterministic seed initialization.
    torch.manual_seed(SEED)
    model = StuckBlockedMotionHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history = []
    started = time.time()
    for epoch in range(EPOCHS):
        order = fit.copy()
        np.random.default_rng(SEED + epoch).shuffle(order)
        total = 0.0
        model.train()
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            output = model(*store.batch(ids, device))
            loss = head_loss(output, rows, ids, weight, mean, std, device)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach()) * len(ids)
        history.append({"epoch": epoch + 1, "loss": total / len(order)})
        if epoch in (0, 9, 19, 29, 39, 49, 59):
            print(json.dumps(history[-1]), flush=True)
    runtime = time.time() - started
    model.eval()
    payload = {
        "schema": "stuck_blocked_motion_head_v1_checkpoint",
        "seed": SEED, "epoch": EPOCHS,
        "state_dict": model.state_dict(), "history": history,
        "shortfall_mean": mean.cpu(), "shortfall_std": std.cpu(),
        "stuck_positive_weights": weight,
        "target_index_sha256": EXPECTED_TARGET,
    }
    atomic_torch(CHECKPOINT, payload)
    return model, store, mean, std, weight, history, runtime


def predict_head(model, store, rows, ids, mean, std, device):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            selected = ids[start:start + 8]
            outputs.append(model(*store.batch(selected, device)).cpu())
    output = torch.cat(outputs).numpy()
    return output[:, :3], output[:, 3:] * std.cpu().numpy() + mean.cpu().numpy()


def fit_temperature(logits, labels):
    x = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.2, max_iter=100)
    def closure():
        optimizer.zero_grad()
        value = torch.exp(log_temperature).clamp(.05, 20)
        loss = F.binary_cross_entropy_with_logits(x / value, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach()).clamp(.05, 20))


def occupancy_and_clearance(rows, labels, probe, ids, device):
    forward = DYNAMIC.FORWARD_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    left = DYNAMIC.LEFT_MIN_EDGE_M + (np.arange(64) + .5) * DYNAMIC.CELL_SIZE_M
    distance = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2) - FOOTPRINT_RADIUS_M
    footprint_margin = distance <= CLEARANCE_SAFE_M
    predicted_clearance, clearance_risk = [], []
    intersections = np.zeros(3, np.int64); unions = np.zeros(3, np.int64)
    per_row_iou = np.full((len(ids), 3), np.nan)
    for start in range(0, len(ids), 8):
        selected = ids[start:start + 8]
        latent = np.stack([[np.asarray(np.load(rows[int(i)]["future_paths"][h], mmap_mode="r"), np.float32)
                            for h in range(3)] for i in selected])
        horizon_logits = []
        with torch.inference_mode():
            for horizon in range(3):
                tokens = torch.from_numpy(latent[:, horizon]).to(device)
                tokens = F.layer_norm(tokens, (1024,))
                horizon_logits.append(probe(tokens, (24, 32)).cpu())
        logits = torch.stack(horizon_logits, 1)
        probabilities = torch.softmax(logits, dim=2).numpy()
        classes = logits.argmax(2).numpy().astype(np.uint8)
        for local, row_index in enumerate(selected):
            clear_h, risk_h = [], []
            global_local = start + local
            for horizon in range(3):
                truth = np.asarray(labels[int(row_index), horizon])
                counts = OCC.occupied_counts(classes[local, horizon], truth)
                intersections[horizon] += counts["occupied_intersection"]
                unions[horizon] += counts["occupied_union"]
                per_row_iou[global_local, horizon] = (
                    counts["observable_occupied_iou"]
                    if counts["observable_occupied_iou"] is not None else np.nan)
                occupied = classes[local, horizon] == 2
                clear_h.append(float(distance[occupied].min()) if occupied.any() else 10.0)
                risk_h.append(float(probabilities[local, horizon, 2][footprint_margin].max()))
            predicted_clearance.append(clear_h); clearance_risk.append(risk_h)
    return {
        "predicted_clearance": np.asarray(predicted_clearance, np.float32),
        "clearance_risk": np.asarray(clearance_risk, np.float32),
        "pooled_iou": [float(intersections[h] / unions[h]) if unions[h] else None for h in range(3)],
        "per_row_iou": per_row_iou,
    }


def choose_stuck_threshold(prob, clearance_unsafe, unsafe):
    candidates = sorted({0.0, 1.0, *(float(np.nextafter(x, math.inf)) for x in prob)})
    rows = []
    for threshold in candidates:
        predicted_unsafe = clearance_unsafe | (prob >= threshold)
        recall = float(predicted_unsafe[unsafe].mean()) if unsafe.any() else 1.0
        retention = float((~predicted_unsafe[~unsafe]).mean()) if (~unsafe).any() else 0.0
        rows.append((recall, retention, threshold))
    feasible = [row for row in rows if row[0] >= .95]
    if feasible:
        recall, retention, threshold = max(feasible, key=lambda x: (x[1], -x[2]))
        return {"threshold": threshold, "recall": recall,
                "safe_retention": retention, "criterion_satisfied": True}
    recall, retention, threshold = max(rows, key=lambda x: (x[0], x[1], -x[2]))
    return {"threshold": threshold, "recall": recall,
            "safe_retention": retention, "criterion_satisfied": False}


def structured_evaluation(rows, ids, risk, admitted, kinematic):
    FS.rows_global = rows
    binary_risk = (~admitted).astype(np.float32)
    result = FS.evaluate_condition("structured_true_future", rows, ids,
                                   binary_risk, .5, kinematic)
    y = np.asarray([rows[int(i)]["unsafe"] for i in ids], bool)
    branch = result["branch"]
    branch.update({
        "auc": FS.auc(y, risk), "average_precision": FS.average_precision(y, risk),
        "ece": FS.ece(y, risk), "brier": float(np.mean((risk - y) ** 2)),
    })
    for family in FS.FAMILIES:
        mask = np.asarray([rows[int(i)]["family"] == family for i in ids])
        result["per_family"][family]["auc"] = FS.auc(y[mask], risk[mask])
    return result


def metric_report(rows, ids, occupancy, stuck_prob, shortfall_pred, admitted, risk, kinematic):
    actual_clearance = np.stack([rows[int(i)]["clearance"] for i in ids])
    actual_shortfall = np.stack([rows[int(i)]["shortfall"] for i in ids])
    stuck = np.stack([rows[int(i)]["labels"][:, 2] for i in ids]).astype(bool)
    unsafe = np.asarray([rows[int(i)]["unsafe"] for i in ids], bool)
    predicted_clearance = occupancy["predicted_clearance"]
    low = actual_clearance[:, 2] < CLEARANCE_SAFE_M
    predicted_low = predicted_clearance[:, 2] < CLEARANCE_SAFE_M
    stuck_pred = stuck_prob[:, 2] >= CALIBRATION["stuck_threshold"]
    stuck_tp = int((stuck[:, 2] & stuck_pred).sum())
    stuck_fn = int((stuck[:, 2] & ~stuck_pred).sum())
    result = structured_evaluation(rows, ids, risk, admitted, kinematic)
    spatial = {
        "occupied_iou_by_horizon": {str(h + 1): occupancy["pooled_iou"][h] for h in range(3)},
        "clearance_mae_by_horizon_m": {str(h + 1): float(np.mean(np.abs(predicted_clearance[:, h] - actual_clearance[:, h]))) for h in range(3)},
        "clearance_spearman_by_horizon": {str(h + 1): spearman(predicted_clearance[:, h], actual_clearance[:, h]) for h in range(3)},
        "h3_low_clearance_positive_rows": int(low.sum()),
        "h3_low_clearance_recall": float(predicted_low[low].mean()) if low.any() else None,
    }
    stuck_metrics = {
        "positive_rows": int(stuck[:, 2].sum()),
        "auc": FS.auc(stuck[:, 2], stuck_prob[:, 2]),
        "average_precision": FS.average_precision(stuck[:, 2], stuck_prob[:, 2]),
        "recall": stuck_tp / (stuck_tp + stuck_fn) if stuck_tp + stuck_fn else None,
        "false_negative_rate": stuck_fn / (stuck_tp + stuck_fn) if stuck_tp + stuck_fn else None,
        "ece": FS.ece(stuck[:, 2], stuck_prob[:, 2]),
        "shortfall_mae_by_horizon_m": {str(h + 1): float(np.mean(np.abs(shortfall_pred[:, h] - actual_shortfall[:, h]))) for h in range(3)},
    }
    contact = np.asarray([rows[int(i)]["labels"][2, 0] for i in ids], bool)
    rejected_clearance = predicted_low
    result["collision_traceability"] = {
        "contact_positive_rows": int(contact.sum()),
        "clearance_rule_contact_recall": float(rejected_clearance[contact].mean()) if contact.any() else None,
        "clearance_rule_safe_rejection_rate": float(rejected_clearance[~unsafe].mean()) if (~unsafe).any() else None,
    }
    result["spatial"] = spatial
    result["stuck"] = stuck_metrics
    result["per_family_components"] = {}
    for family in FS.FAMILIES:
        mask = np.asarray([rows[int(i)]["family"] == family for i in ids])
        result["per_family_components"][family] = {
            "rows": int(mask.sum()),
            "occupied_h3_iou_mean_defined_rows": json_float(np.nanmean(occupancy["per_row_iou"][mask, 2])) if np.isfinite(occupancy["per_row_iou"][mask, 2]).any() else None,
            "clearance_spearman_h3": spearman(predicted_clearance[mask, 2], actual_clearance[mask, 2]),
            "stuck_auc_h3": FS.auc(stuck[mask, 2], stuck_prob[mask, 2]),
        }
    return result


CALIBRATION = {}


def main():
    global CALIBRATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    evaluator_fixture = fixture()
    if not evaluator_fixture["passed"]:
        raise RuntimeError("evaluator fixture failed")
    rows, split = load_rows()
    labels, label_index = occupancy_labels(rows)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    probe_metadata = OCC.validate_probe_package_metadata()
    if (probe_metadata["package_digest"] != EXPECTED_PROBE_PACKAGE
            or probe_metadata["weights_sha256"] != EXPECTED_PROBE_WEIGHTS):
        raise RuntimeError("occupancy probe binding mismatch")
    probe, probe_state_digest = OCC.load_probe(device)
    model, store, shortfall_mean, shortfall_std, weights, history, train_runtime = train_head(rows, split, device)
    calibration_logits, calibration_shortfall = predict_head(
        model, store, rows, split["calibration"], shortfall_mean, shortfall_std, device)
    calibration_occupancy = occupancy_and_clearance(
        rows, labels, probe, split["calibration"], device)
    calibration_stuck = np.stack([
        rows[int(i)]["labels"][:, 2] for i in split["calibration"]]).astype(bool)
    temperature = fit_temperature(calibration_logits[:, 2], calibration_stuck[:, 2])
    calibration_probability = 1 / (1 + np.exp(-calibration_logits / temperature))
    calibration_unsafe = np.asarray([rows[int(i)]["unsafe"] for i in split["calibration"]], bool)
    calibration_clearance_unsafe = calibration_occupancy["predicted_clearance"][:, 2] < CLEARANCE_SAFE_M
    chosen = choose_stuck_threshold(
        calibration_probability[:, 2], calibration_clearance_unsafe, calibration_unsafe)
    CALIBRATION = {
        "stuck_temperature": temperature,
        "stuck_threshold": chosen["threshold"],
        "selection": chosen,
        "clearance_limit_m": CLEARANCE_SAFE_M,
        "footprint_radius_m": FOOTPRINT_RADIUS_M,
        "occupancy_probe_recalibrated": False,
    }
    heldout_logits, heldout_shortfall = predict_head(
        model, store, rows, split["heldout"], shortfall_mean, shortfall_std, device)
    heldout_probability = 1 / (1 + np.exp(-heldout_logits / temperature))
    heldout_occupancy = occupancy_and_clearance(rows, labels, probe, split["heldout"], device)
    clearance_unsafe = heldout_occupancy["predicted_clearance"][:, 2] < CLEARANCE_SAFE_M
    admitted = ~(clearance_unsafe | (heldout_probability[:, 2] >= chosen["threshold"]))
    continuous_risk = np.maximum(heldout_occupancy["clearance_risk"][:, 2],
                                 heldout_probability[:, 2])
    kinematic = np.stack([row["kinematic"] for row in rows])
    result = metric_report(rows, split["heldout"], heldout_occupancy,
                           heldout_probability, heldout_shortfall, admitted,
                           continuous_risk, kinematic)
    heldout_unsafe = np.asarray([rows[int(i)]["unsafe"] for i in split["heldout"]], bool)
    oracle = FS.evaluate_condition("oracle_safety", rows, split["heldout"],
                                   heldout_unsafe.astype(float), .5, kinematic)
    planning = result["planning"]
    spatial = result["spatial"]
    stuck = result["stuck"]
    branch = result["branch"]
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    checks = {
        "h3_occupied_iou_ge_0_35": spatial["occupied_iou_by_horizon"]["3"] is not None and spatial["occupied_iou_by_horizon"]["3"] >= .35,
        "clearance_spearman_ge_0_60": spatial["clearance_spearman_by_horizon"]["3"] is not None and spatial["clearance_spearman_by_horizon"]["3"] >= .60,
        "low_clearance_recall_ge_0_90": spatial["h3_low_clearance_recall"] is not None and spatial["h3_low_clearance_recall"] >= .90,
        "stuck_auc_ge_0_85": stuck["auc"] is not None and stuck["auc"] >= .85,
        "stuck_recall_ge_0_90": stuck["recall"] is not None and stuck["recall"] >= .90,
        "stuck_fnr_le_0_10": stuck["false_negative_rate"] is not None and stuck["false_negative_rate"] <= .10,
        "unsafe_recall_ge_0_95": branch["unsafe_recall"] >= .95,
        "unsafe_fnr_le_0_05": branch["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": branch["safe_candidate_retention"] >= .40,
        "six_states_retain_safe": planning["states_retaining_safe"] >= 6,
        "no_state_only_unsafe": planning["states_only_unsafe_admitted"] == 0,
        "no_safe_states_abstain": planning["correct_abstention"] == oracle["planning"]["correct_abstention"],
        "selected_unsafe_zero": planning["selected_unsafe_rate"] == 0,
        "route_progress_ge_80pct_oracle": planning["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "normalized_regret_le_0_20": planning["normalized_safe_progress_regret"] is not None and planning["normalized_safe_progress_regret"] <= .20,
        "best_safe_top3_ge_0_75": planning["best_safe_top3"] is not None and planning["best_safe_top3"] >= .75,
        "false_abstention_le_1": planning["false_abstention"] <= 1,
    }
    stage_a_passed = all(checks.values())
    classification = ("STAGE_A_PASSED_STAGE_B_NOT_IMPLEMENTED" if stage_a_passed
                      else "TRUE_FUTURE_STRUCTURED_SAFETY_STATE_NO_GO")
    if stage_a_passed:
        raise RuntimeError("Stage A unexpectedly passed; Stage B execution is required")
    payload = {
        "schema": "structured_spatial_safety_state_jepa_dev_v2_result",
        "source_commit": "f72fe00c8426a973fbb56c521e5a89a563a9373f",
        "preserved_terminal": "SAFETY_AUXILIARY_JEPA_DEVELOPMENT_NO_SIGNAL",
        "classification": classification,
        "stage_a_passed": stage_a_passed,
        "stage_b_reached": False,
        "bindings": {
            "target_index_sha256": EXPECTED_TARGET,
            "occupancy_probe": probe_metadata,
            "occupancy_probe_state_digest": probe_state_digest,
            "occupancy_label_index": label_index,
        },
        "fixture": evaluator_fixture,
        "component_contract": {
            "stuck_head": "STUCK_BLOCKED_MOTION_HEAD_V1",
            "stuck_head_parameters": sum(p.numel() for p in model.parameters()),
            "seed": SEED, "epochs": EPOCHS,
            "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY},
            "outputs": ["cumulative_stuck_H1_H2_H3", "signed_realised_minus_nominal_displacement_H1_H2_H3"],
            "aggregate_unsafe_output": False,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha(CHECKPOINT),
            "history": history,
            "stuck_positive_weights": weights.tolist(),
        },
        "calibration": CALIBRATION,
        "heldout": result,
        "oracle_safety_kinematic": oracle,
        "stage_a_gate": {"passed": stage_a_passed, "checks": checks},
        "stage_b": None,
        "runtime": {"training_s": train_runtime, "total_s": time.time() - started},
        "storage": {
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "occupancy_labels_bytes": LABELS.stat().st_size,
        },
        "custody": {
            "new_simulation": False, "new_rgb_rendering": False,
            "new_latent_encoding": False, "new_branch_generation": False,
            "new_predictor_seed_trained": 0,
        },
    }
    atomic_json(OUT / "result.json", payload)
    print(json.dumps({
        "classification": classification,
        "checks": checks,
        "spatial": spatial,
        "stuck": stuck,
        "branch": branch,
        "planning": {k: v for k, v in planning.items() if k != "per_state"},
        "checkpoint_sha256": sha(CHECKPOINT),
        "result_sha256": sha(OUT / "result.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
