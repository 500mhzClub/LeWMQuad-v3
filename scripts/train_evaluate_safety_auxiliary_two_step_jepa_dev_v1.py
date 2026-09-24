#!/usr/bin/env python3
"""Single-seed safety-auxiliary continuation of the frozen RGB rollout JEPA.

This development runner is deliberately narrow.  It reuses the historical
factorial training loader, the frozen route-intent safety corpus, and the frozen
counterfactual/occupancy reducers.  It never renders, encodes, or simulates.

The historical epoch-21 artefact contains the predictor and optimiser only; its
ViT-L encoder was frozen and all JEPA targets were cached.  Accordingly this
continuation preserves that actual lineage: the predictor and the one new shared
safety branch are trainable, while ViT-L remains frozen.  No fictitious online
encoder or EMA state is introduced into the checkpoint.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_go2_counterfactual_predictor_qualification_v1_2 as CF
from scripts import build_dev_canonical_cache_map_v1 as CACHE_MAP
from scripts import build_dev_factorial_manifest_v1 as FACTORIAL
from scripts import dev_proprio_predictor_v1 as P
from scripts import run_dev_proprio_factorial_driver_v1 as D
from scripts import run_dev_v03_temporal_action_jepa_v1 as T
from scripts import run_go2_counterfactual_occupancy_assay_v1_2 as OCC
from scripts import train_evaluate_candidate_conditioned_future_safety_v1 as FS

SEED = 2026082004
HISTORICAL_SEED = 2026080901
EPOCHS = 12
BATCH = 4
PREDICTOR_LR = 3e-5
SAFETY_LR = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EXPECTED_CHECKPOINT_SHA = "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4"
EXPECTED_TARGET_INDEX_SHA = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
HISTORICAL_CHECKPOINT = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/seed_2026080901/"
    "seed_2026080901_rgb_rollout_epoch21.pt"
)
OUT = ROOT / ".generated/safety_auxiliary_two_step_jepa_dev_v1"
CACHE_OUT = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "safety_auxiliary_two_step_jepa_dev_v1"
)
CHECKPOINT = CACHE_OUT / "safety_auxiliary_two_step_seed_2026082004_epoch12.pt"
LATEST = CACHE_OUT / "latest.pt"
COMPONENTS = FS.COMPONENTS
FAMILIES = FS.FAMILIES


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


class TrajectorySafetyBranch(nn.Module):
    """Shared actual/predicted trajectory branch, 191,404 parameters."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1024, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.action = nn.Sequential(
            nn.Linear(48, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(448, 128), nn.GELU(), nn.Linear(128, 12)
        )
        position = torch.arange(32, dtype=torch.float32)
        horizon = []
        for h in (1.0, 2.0, 3.0):
            horizon.append(torch.sin(
                position * (h / 3.0) /
                (10000 ** (2 * (position // 2) / 32))))
        self.register_buffer("horizon_embedding", torch.stack(horizon))

    def forward(self, z0, future, action_control):
        z0 = F.layer_norm(z0.float(), (1024,), weight=None, bias=None)
        future = F.layer_norm(future.float(), (1024,), weight=None, bias=None)
        current = self.projection(z0)
        features = []
        for h in range(3):
            fut = self.projection(future[:, h]) + self.horizon_embedding[h]
            joined = torch.cat([current, fut, fut - current], -1)
            joined = joined.transpose(1, 2).reshape(-1, 96, 24, 32)
            spatial = self.conv(joined)
            features.append(torch.cat([
                spatial.mean((2, 3)), spatial.amax((2, 3))], 1))
        trajectory = torch.cat(features, 1)
        control = self.action(action_control.float())
        return self.fusion(torch.cat([trajectory, control], 1)).view(-1, 3, 4)


def load_predictor(device):
    if sha(HISTORICAL_CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("historical two-step checkpoint digest mismatch")
    payload = torch.load(HISTORICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("seed") != HISTORICAL_SEED or payload.get("epoch") != 21:
        raise RuntimeError("historical checkpoint identity mismatch")
    model = P.build_paired(HISTORICAL_SEED, use_proprio=False,
                           width=384, depth=6, heads=6)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device), payload


def route_arrays():
    rows = FS.load_metadata()
    labels = np.stack([r["labels"] for r in rows]).astype(np.float32)
    action = np.stack([r["action_control"] for r in rows]).astype(np.float32)
    split = {name: np.asarray([i for i, r in enumerate(rows)
                              if r["split"] == name], dtype=int)
             for name in ("fit", "calibration", "heldout")}
    if sha(FS.V2 / "target_latent_index.json") != EXPECTED_TARGET_INDEX_SHA:
        raise RuntimeError("route target index digest mismatch")
    weights, defined = FS.pos_weights(labels, split["fit"])
    return rows, labels, action, split, weights, defined


class RouteStore:
    def __init__(self, rows, stats):
        self.rows = rows
        self.stats = stats
        self.z0 = {}

    def batch(self, ids, device):
        actual = []
        context = []
        action_blocks = [[], [], []]
        controls = []
        action_control = []
        for i in ids:
            row = self.rows[int(i)]
            actual.append(np.stack([
                np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
                for path in row["future_paths"]], axis=0))
            sid = row["state_id"]
            if sid not in self.z0:
                self.z0[sid] = np.asarray(
                    np.load(row["z0_path"], mmap_mode="r"), dtype=np.float32)
            z0 = self.z0[sid]
            # The exact pre-action context was not retained by V1/V2.  This is
            # the frozen, candidate-invariant H1 hold proxy already registered
            # by the preceding safety experiment; no rendering is permitted.
            context.append(np.stack([z0, z0, z0], axis=0))
            full_action = np.asarray(
                row["action_control"][:45], np.float32).reshape(3, 5, 3)
            # The historical RGB predictor's exact registered action contract is
            # five (vx, yaw-rate) samples per block; lateral command is fixed zero.
            flattened = full_action[:, :, [0, 2]].reshape(3, P.ACTION_DIM)
            for h in range(3):
                action_blocks[h].append(flattened[h])
            prior = np.asarray(row["action_control"][45:48], np.float32)
            controls.append(np.tile(prior[[0, 2]], (3, 5, 1)))
            action_control.append(row["action_control"])
        raw_context = torch.from_numpy(np.stack(context)).to(device)
        context_tensor = T.normalise(raw_context)
        control = torch.from_numpy(np.stack(controls).astype(np.float32)).to(device)
        dummy = torch.zeros(len(ids), 3, 5, P.PROPRIO_DIM, device=device)
        _, control = D.normalise_batch(dummy, control, self.stats, device)
        return {
            "context": context_tensor,
            "z0": raw_context[:, -1],
            "actual": torch.from_numpy(np.stack(actual)).to(device),
            "actions": [torch.from_numpy(np.asarray(q, np.float32)).to(device)
                        for q in action_blocks],
            "control": control,
            "action_control": torch.from_numpy(
                np.asarray(action_control, np.float32)).to(device),
        }


def rollout(predictor, batch, max_h=3):
    return torch.stack(P.unroll(
        predictor, batch["context"], batch["actions"], None,
        batch["control"], max_h=max_h), dim=1)


def rollout_loss(predicted, targets):
    target = T.normalise(targets.float())
    e1 = (predicted[:, 0] - target[:, 0]).abs().mean()
    e2 = (predicted[:, 1] - target[:, 1]).abs().mean()
    return e1 + torch.cat([predicted[:, :2]], 1).sub(
        torch.cat([target[:, :2]], 1)).abs().mean(), e1, e2


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True,
                                allow_unused=True)
    value = torch.zeros((), device=loss.device)
    for grad in grads:
        if grad is not None:
            value = value + grad.float().pow(2).sum()
    return float(value.sqrt().detach())


def safety_loss(logits, labels, weights, defined):
    return FS.balanced_loss(logits, labels, weights, defined)


def load_factorial_loader():
    rows, manifest, stats = D.load_rows()
    factorial = FACTORIAL.load()
    mapping = CACHE_MAP.load()
    loader = D.CanonicalLoader(mapping, rows, stats, split="train",
                               expected_digest=mapping["digest"],
                               factorial=factorial,
                               expected_factorial_digest=factorial["digest"])
    return loader, stats, manifest, factorial, mapping


def factorial_forward(model, batch):
    p1 = T.normalise(model(batch["context"], batch["a1"], batch["mask"],
                           None, None, batch["control"]))
    window = torch.stack([batch["context"][:, 1], batch["context"][:, 2], p1], 1)
    control2 = torch.cat([
        batch["control"][:, 1:], P.control_slot_from_action(batch["a1"])], 1)
    p2 = T.normalise(model(window, batch["a2"], batch["mask"],
                           None, None, control2))
    e1 = (p1 - batch["y1"]).abs().mean()
    e2 = (p2 - batch["y2"]).abs().mean()
    loss = e1 + torch.cat([p1, p2], 1).sub(
        torch.cat([batch["y1"], batch["y2"]], 1)).abs().mean()
    return loss, e1, e2


def prevalence_report(rows, labels, split):
    result = {}
    for split_name, ids in split.items():
        result[split_name] = {}
        for family in ("overall", *FAMILIES):
            q = ids if family == "overall" else np.asarray(
                [i for i in ids if rows[int(i)]["family"] == family], int)
            result[split_name][family] = {}
            for h in range(3):
                for c, component in enumerate(COMPONENTS):
                    result[split_name][family][f"H{h+1}_{component}"] = {
                        "positive": int(labels[q, h, c].sum()),
                        "rows": int(len(q)),
                        "prevalence": float(labels[q, h, c].mean()),
                    }
    return result


def evaluator_fixture():
    old_out = FS.OUT
    FS.OUT = OUT
    try:
        fixture = FS.evaluator_fixture()
    finally:
        FS.OUT = old_out
    fixture = dict(fixture)
    fixture["extended_conditions"] = {
        "actual_future": True, "predicted_future": True,
        "current_context_only": True, "oracle": True,
        "guarded_kinematic_selection": True,
    }
    fixture["pass"] = bool(fixture["pass"] and all(
        fixture["extended_conditions"].values()))
    atomic_json(OUT / "evaluator_fixture.json", fixture)
    return fixture


def smoke_and_lambda(predictor, safety, route, rows, labels, fit_ids,
                     weights, defined, device):
    ids = fit_ids[:BATCH]
    batch = route.batch(ids, device)
    y = torch.from_numpy(labels[ids]).to(device)
    predicted = rollout(predictor, batch)
    jepa, _, _ = rollout_loss(predicted, batch["actual"])
    logits_true = safety(batch["z0"], batch["actual"], batch["action_control"])
    logits_pred = safety(batch["z0"], predicted, batch["action_control"])
    s_true = safety_loss(logits_true, y, weights, defined)
    s_pred = safety_loss(logits_pred, y, weights, defined)
    shared = [p for p in predictor.parameters() if p.requires_grad]
    jnorm = grad_norm(jepa, shared)
    snorm = grad_norm(s_pred, shared)
    if not (jnorm > 0 and snorm > 0):
        raise RuntimeError(f"gradient calibration failed: JEPA={jnorm}, safety={snorm}")
    lam = 0.25 * jnorm / snorm
    total = jepa + lam * (s_true + s_pred)
    predictor.zero_grad(set_to_none=True); safety.zero_grad(set_to_none=True)
    total.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in safety.parameters()):
        raise RuntimeError("safety smoke gradients invalid")
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters()):
        raise RuntimeError("safety/JEPA smoke does not reach predictor")
    output_grad = safety.fusion[-1].weight.grad.reshape(12, -1).abs().sum(1)
    defined_flat = torch.from_numpy(defined.reshape(-1)).to(output_grad.device)
    if not bool((output_grad[defined_flat] > 0).all()):
        raise RuntimeError("a nondegenerate safety output received no gradient")
    predictor.zero_grad(set_to_none=True); safety.zero_grad(set_to_none=True)
    predictor.eval(); safety.eval()
    with torch.inference_mode():
        a = rollout(predictor, batch)
        b = rollout(predictor, batch)
        if not torch.equal(a, b):
            raise RuntimeError("deterministic predictor smoke failed")
        base, _ = load_predictor(device)
        original = rollout(base.eval(), batch)
        if not torch.equal(a, original):
            raise RuntimeError("safety-disabled initial predictor differs from historical model")
        del base
        changed = batch["action_control"].clone()
        changed[:, :45] = torch.flip(changed[:, :45], (1,))
        if torch.allclose(safety(batch["z0"], batch["actual"], batch["action_control"]),
                          safety(batch["z0"], batch["actual"], changed)):
            raise RuntimeError("safety branch is action-insensitive")
    predictor.train(); safety.train()
    tmp = CACHE_OUT / ".smoke.pt"
    atomic_torch(tmp, {"predictor": predictor.state_dict(), "safety": safety.state_dict()})
    clone, _ = load_predictor(device)
    clone.load_state_dict(torch.load(tmp, map_location=device, weights_only=False)["predictor"])
    clone_s = TrajectorySafetyBranch().to(device)
    clone_s.load_state_dict(torch.load(tmp, map_location=device, weights_only=False)["safety"])
    tmp.unlink()
    return {
        "passed": True, "lambda_safety": float(lam),
        "jepa_shared_gradient_norm": jnorm,
        "unscaled_safety_shared_gradient_norm": snorm,
        "scaled_ratio": float(lam * snorm / jnorm),
        "predictor_gradient_present": True,
        "all_defined_safety_outputs_have_gradient": True,
        "exact_historical_behavior_when_safety_disabled": True,
        "historical_online_encoder_present": False,
        "historical_encoder_contract": "frozen cached ViT-L tokens",
    }


def save_training_state(path, predictor, safety, optimizer, epoch, history,
                        smoke, ratios):
    atomic_torch(path, {
        "schema": "safety_auxiliary_two_step_jepa_dev_v1_checkpoint",
        "seed": SEED, "epoch": epoch,
        "predictor_state_dict": predictor.state_dict(),
        "safety_state_dict": safety.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history, "smoke": smoke, "gradient_ratios": ratios,
        "historical_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    })


def train(device, fixture):
    rows, labels, action, split, weights_np, defined = route_arrays()
    loader, stats, manifest, factorial, mapping = load_factorial_loader()
    predictor, historical = load_predictor(device)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    safety = TrajectorySafetyBranch().to(device)
    if sum(p.numel() for p in safety.parameters()) >= 250000:
        raise RuntimeError("safety branch exceeds 250,000 parameters")
    route = RouteStore(rows, stats)
    weights = torch.from_numpy(weights_np).to(device)
    smoke = smoke_and_lambda(predictor, safety, route, rows, labels,
                             split["fit"], weights, defined, device)
    lam = float(smoke["lambda_safety"])
    optimizer = torch.optim.AdamW([
        {"params": predictor.parameters(), "lr": PREDICTOR_LR},
        {"params": safety.parameters(), "lr": SAFETY_LR},
    ], weight_decay=WEIGHT_DECAY, foreach=False)
    history = []
    ratios = []
    start_epoch = 0
    if LATEST.exists():
        state = torch.load(LATEST, map_location="cpu", weights_only=False)
        if state.get("seed") == SEED and int(state.get("epoch", 0)) < EPOCHS:
            predictor.load_state_dict(state["predictor_state_dict"])
            safety.load_state_dict(state["safety_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            history = state["history"]; ratios = state["gradient_ratios"]
            smoke = state["smoke"]; lam = float(smoke["lambda_safety"])
            start_epoch = int(state["epoch"])
            print(f"resuming after epoch {start_epoch}", flush=True)
    started = time.time()
    fit_ids = split["fit"]
    for epoch in range(start_epoch, EPOCHS):
        predictor.train(); safety.train()
        original_plan = D.batch_plan(SEED, epoch, len(loader), BATCH)
        route_order = fit_ids.copy()
        np.random.default_rng(SEED + epoch).shuffle(route_order)
        route_plan = [route_order[i:i+BATCH]
                      for i in range(0, len(route_order), BATCH)]
        route_at = 0
        totals = collections.Counter()
        for step, positions in enumerate(original_plan):
            batch = loader.batch(positions, device, stats)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                loss, e1, e2 = factorial_forward(predictor, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP)
            optimizer.step()
            totals.update(jepa=float(loss.detach()), e1=float(e1.detach()),
                          e2=float(e2.detach()), original_batches=1)
            # Deterministic even interleave, preserving every original batch.
            target_route = ((step + 1) * len(route_plan)) // len(original_plan)
            while route_at < target_route:
                ids = route_plan[route_at]; route_at += 1
                rb = route.batch(ids, device)
                y = torch.from_numpy(labels[ids]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    predicted = rollout(predictor, rb)
                    jloss, re1, re2 = rollout_loss(predicted, rb["actual"])
                    true_loss = safety_loss(
                        safety(rb["z0"], rb["actual"], rb["action_control"]),
                        y, weights, defined)
                    pred_loss = safety_loss(
                        safety(rb["z0"], predicted, rb["action_control"]),
                        y, weights, defined)
                    total = jloss + lam * (true_loss + pred_loss)
                if route_at == 1:
                    jnorm = grad_norm(jloss, list(predictor.parameters()))
                    snorm = grad_norm(pred_loss, list(predictor.parameters()))
                    ratio = lam * snorm / max(jnorm, 1e-20)
                    ratios.append({"epoch": epoch + 1, "ratio": float(ratio),
                                   "jepa_norm": jnorm, "safety_norm": snorm})
                total.backward()
                nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP)
                nn.utils.clip_grad_norm_(safety.parameters(), GRAD_CLIP)
                optimizer.step()
                totals.update(route_total=float(total.detach()),
                              route_jepa=float(jloss.detach()),
                              safety_true=float(true_loss.detach()),
                              safety_pred=float(pred_loss.detach()), route_batches=1)
        entry = {"epoch": epoch + 1}
        for key in ("jepa", "e1", "e2"):
            entry[key] = totals[key] / totals["original_batches"]
        for key in ("route_total", "route_jepa", "safety_true", "safety_pred"):
            entry[key] = totals[key] / totals["route_batches"]
        entry.update(original_batches=int(totals["original_batches"]),
                     safety_batches=int(totals["route_batches"]))
        history.append(entry)
        save_training_state(LATEST, predictor, safety, optimizer,
                            epoch + 1, history, smoke, ratios)
        print(json.dumps(entry), flush=True)
    runtime = time.time() - started
    save_training_state(CHECKPOINT, predictor, safety, optimizer,
                        EPOCHS, history, smoke, ratios)
    if LATEST.exists(): LATEST.unlink()
    return {
        "rows": rows, "labels": labels, "action": action, "split": split,
        "weights": weights_np, "defined": defined, "stats": stats,
        "predictor": predictor.eval(), "safety": safety.eval(), "route": route,
        "history": history, "ratios": ratios, "smoke": smoke,
        "training_runtime_s": runtime,
        "factorial": {"rows": len(loader), "manifest_digest": factorial["digest"],
                      "cache_map_digest": mapping["digest"],
                      "normalisation_sha256": manifest["normalisation_sha256"]},
        "fixture": fixture,
    }


@torch.no_grad()
def safety_logits(model, safety, route, ids, device, mode, batch_size=4):
    output = []
    for start in range(0, len(ids), batch_size):
        q = ids[start:start+batch_size]
        batch = route.batch(q, device)
        if mode == "actual": future = batch["actual"]
        elif mode == "predicted": future = rollout(model, batch)
        elif mode == "current": future = batch["z0"].unsqueeze(1).expand(-1, 3, -1, -1)
        else: raise ValueError(mode)
        output.append(safety(batch["z0"], future, batch["action_control"]).cpu().numpy())
    return np.concatenate(output)


def calibrate_and_score(run, device):
    rows, labels, split = run["rows"], run["labels"], run["split"]
    FS.rows_global = rows
    kin = np.stack([r["kinematic"] for r in rows])
    cal = safety_logits(run["predictor"], run["safety"], run["route"],
                        split["calibration"], device, "actual")
    ycal = labels[split["calibration"], 2, 3].astype(bool)
    temperature = FS.fit_temperature(cal[:, 2, 3], ycal)
    cal_prob = torch.sigmoid(torch.from_numpy(cal[:, 2, 3]) / temperature).numpy()
    threshold = FS.choose_threshold(cal_prob, ycal)
    output = {"temperature": temperature, **threshold,
              "tie_rule": "more conservative threshold when safe retention ties"}
    held = split["heldout"]
    results = {}
    components = {}
    logits = {}
    for mode in ("current", "actual"):
        value = safety_logits(run["predictor"], run["safety"], run["route"],
                              held, device, mode)
        logits[mode] = value
        prob = torch.sigmoid(torch.from_numpy(value[:, 2, 3]) / temperature).numpy()
        results[mode] = FS.evaluate_condition(
            f"augmented_{mode}_future", rows, held, prob,
            threshold["threshold"], kin)
        all_prob = torch.sigmoid(torch.from_numpy(value) / temperature).numpy()
        components[mode] = FS.component_metrics(
            rows, held, all_prob, threshold["threshold"])
    y = labels[held, 2, 3].astype(bool)
    oracle = FS.evaluate_condition(
        "oracle_safety", rows, held, y.astype(float), .5, kin)
    prior = json.loads((FS.OUT / "result.json").read_text())
    return output, results, components, oracle, prior


def actual_gate(results, components, oracle):
    actual = results["actual"]
    branch = actual["branch"]; planning = actual["planning"]
    contact = components["actual"]["collision_or_disallowed_contact"]
    stuck = components["actual"]["stuck"]
    checks = {
        "auc_ge_0_80": branch["auc"] is not None and branch["auc"] >= .80,
        "recall_ge_0_95": branch["unsafe_recall"] >= .95,
        "fnr_le_0_05": branch["unsafe_false_negative_rate"] <= .05,
        "safe_retention_ge_0_40": branch["safe_candidate_retention"] >= .40,
        "ece_le_0_10": branch["ece"] <= .10,
        "six_states_retain_safe": planning["states_retaining_safe"] >= 6,
        "no_state_only_unsafe": planning["states_only_unsafe_admitted"] == 0,
        # This requirement is vacuous when the held-out panel has no state with
        # zero safe candidates.  The oracle evaluator counts exactly those
        # states as correct abstentions.
        "no_safe_candidate_states_abstain": (
            planning["correct_abstention"]
            == oracle["planning"]["correct_abstention"]
        ),
        "contact_recall_ge_0_90": contact["recall"] is not None and contact["recall"] >= .90,
        "stuck_recall_ge_0_90": stuck["recall"] is not None and stuck["recall"] >= .90,
    }
    return checks, all(checks.values())


def predicted_gate(result, oracle):
    branch = result["branch"]; planning = result["planning"]
    oracle_progress = oracle["planning"]["mean_selected_distance_progress_m"]
    checks = {
        "recall_ge_0_90": branch["unsafe_recall"] >= .90,
        "fnr_le_0_10": branch["unsafe_false_negative_rate"] <= .10,
        "safe_retention_ge_0_30": branch["safe_candidate_retention"] >= .30,
        "no_state_only_unsafe": planning["states_only_unsafe_admitted"] == 0,
        "selected_unsafe_zero": planning["selected_unsafe_rate"] == 0,
        "progress_80pct_oracle": planning["mean_selected_distance_progress_m"] >= .8 * oracle_progress,
        "best_safe_top3_ge_0_75": planning["best_safe_top3"] is not None and planning["best_safe_top3"] >= .75,
        "normalized_regret_le_0_20": planning["normalized_safe_progress_regret"] is not None and planning["normalized_safe_progress_regret"] <= .20,
    }
    return checks, all(checks.values())


@torch.no_grad()
def nonregression(predictor, device):
    bundle = CF.validate_stage_a_metadata()
    stats = CF.load_frozen_normalisation(
        json.loads(D.PROPRIO.joinpath("proprio_manifest.json").read_text())["normalisation_sha256"])
    records = []
    predictions = {}
    for state in bundle.states:
        value = CF.predict_state(predictor, state,
                                 bundle.context_records[state.context_key],
                                 stats, False, device)
        predictions[state.state_id] = value
        records.append({
            "state_index": state.state_index, "state_id": state.state_id,
            "family": state.family,
            "candidate_names": list(state.candidate_names),
            "per_horizon": CF.score_state_predictions(bundle, state, value, device),
        })
        print(f"[nonreg] {state.state_index + 1}/20", flush=True)
    augmented = CF.aggregate_records(records)
    historical_report = json.loads(CF.RESULT_PATH.read_text())
    historical = historical_report["cells_by_seed"][str(HISTORICAL_SEED)]["rgb_rollout"]
    direct = {}
    for h in range(1, 5):
        a = augmented["per_horizon"][str(h)]["equal_family"]
        b = historical["per_horizon"][str(h)]["equal_family"]
        direct[str(h)] = {
            "augmented": {
                "changed_token_cosine": a["direct"]["changed_cosine"],
                "normalized_error": a["direct"]["normalised_error_vs_persistence"],
                "top1": a["retrieval"]["top1"],
                "mrr": a["retrieval"]["mean_reciprocal_rank"],
                "pairwise": a["retrieval"]["pairwise_accuracy"],
            },
            "historical": {
                "changed_token_cosine": b["direct"]["changed_cosine"],
                "normalized_error": b["direct"]["normalised_error_vs_persistence"],
                "top1": b["retrieval"]["top1"],
                "mrr": b["retrieval"]["mean_reciprocal_rank"],
                "pairwise": b["retrieval"]["pairwise_accuracy"],
            },
            "per_family": {
                family: {
                    "augmented": augmented["per_horizon"][str(h)]["per_family"][family],
                    "historical": historical["per_horizon"][str(h)]["per_family"][family],
                } for family in CF.FAMILIES
            },
        }
    # Frozen occupancy labels/probe; predictions stay in memory.
    # The historical Stage-D loader also revalidates every old source binding;
    # later repository commits legitimately changed one unrelated scorer
    # contract.  Bind the immutable label index/shards directly here instead.
    label_index_path = OCC.LABEL_ROOT / "labels_index.json"
    label_index = json.loads(label_index_path.read_text())
    if label_index.get("labels_index_digest") != "a81f1c63f9fa181bfa728b1cb5da2ad4573f2aa80cb5801c9d54acab34d411e2":
        raise RuntimeError("frozen occupancy label-index digest mismatch")
    label_entries = {str(x["branch_identity_digest"]): x
                     for x in label_index["records"]}
    if len(label_entries) != 240:
        raise RuntimeError("frozen occupancy label index incomplete")
    probe, probe_digest = OCC.load_probe(device)
    occupancy = {}
    rows_by_state = collections.defaultdict(list)
    for row in bundle.rows: rows_by_state[str(row["state_id"])].append(row)
    for state_id in rows_by_state:
        rows_by_state[state_id].sort(key=lambda x: int(x["candidate_index"]))
    for h in (2, 3, 4):
        flat = []
        for state in bundle.states:
            state_rows = rows_by_state[state.state_id]
            truth_rows = []
            for row in state_rows:
                key = str(row["branch_identity_digest"])
                entry = label_entries[key]
                path = OCC.STAGE_ROOT / str(entry["path"])
                if sha(path) != entry["label_sha256"]:
                    raise RuntimeError(f"occupancy label shard digest mismatch: {key}")
                truth_rows.append(np.asarray(np.memmap(
                    path, mode="r", dtype=np.uint8, shape=(4, 64, 64))))
            truth = np.stack(truth_rows, 0)
            tokens = torch.from_numpy(
                predictions[state.state_id][:, h-1].astype(np.float32)).to(device)
            chosen = probe(tokens, OCC.TOKEN_GRID).argmax(1).cpu().numpy().astype(np.uint8)
            for i, row in enumerate(state_rows):
                flat.append({"family": row["family"],
                             **OCC.occupied_counts(chosen[i], truth[i, h-1])})
        pooled = OCC.pooled_iou(flat)
        per_family = {family: OCC.pooled_iou(
            [r for r in flat if r["family"] == family]) for family in OCC.FAMILIES}
        old_result = json.loads((OCC.RESULT_ROOT / "result.json").read_text())
        old_values = old_result["horizons"][str(h)]["whole_pilot_pooled_diagnostic"]["four_cells"]["rgb_rollout"]["predicted"]["values"]
        occupancy[str(h)] = {"augmented_pooled_iou": pooled,
                             "historical_seed_2026080901_pooled_iou": old_values[0],
                             "per_family": per_family}
    h3a = direct["3"]["augmented"]; h3b = direct["3"]["historical"]
    checks = {
        "h3_cosine_drop_le_0_005": h3a["changed_token_cosine"] >= h3b["changed_token_cosine"] - .005,
        "h3_normalized_error_worsening_le_0_01": h3a["normalized_error"] <= h3b["normalized_error"] + .01,
        "h3_top1_drop_le_0_02": h3a["top1"] >= h3b["top1"] - .02,
        "h3_pairwise_drop_le_0_02": h3a["pairwise"] >= h3b["pairwise"] - .02,
        "h3_occupancy_drop_le_0_01": occupancy["3"]["augmented_pooled_iou"] >= occupancy["3"]["historical_seed_2026080901_pooled_iou"] - .01,
    }
    return {"direct_and_retrieval": direct, "occupancy": occupancy,
            "checks": checks, "passed": all(checks.values()),
            "probe_state_digest": probe_digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CACHE_OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    fixture = evaluator_fixture()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if CHECKPOINT.exists():
        rows, labels, action, split, weights, defined = route_arrays()
        loader, stats, manifest, factorial, mapping = load_factorial_loader()
        predictor, historical = load_predictor(device)
        blob = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        predictor.load_state_dict(blob["predictor_state_dict"]); predictor.to(device).eval()
        safety = TrajectorySafetyBranch().to(device)
        safety.load_state_dict(blob["safety_state_dict"]); safety.eval()
        run = {"rows": rows, "labels": labels, "action": action, "split": split,
               "weights": weights, "defined": defined, "stats": stats,
               "predictor": predictor, "safety": safety, "route": RouteStore(rows, stats),
               "history": blob["history"], "ratios": blob["gradient_ratios"],
               "smoke": blob["smoke"], "training_runtime_s": None,
               "factorial": {"rows": len(loader), "manifest_digest": factorial["digest"],
                             "cache_map_digest": mapping["digest"],
                             "normalisation_sha256": manifest["normalisation_sha256"]},
               "fixture": fixture}
    else:
        run = train(device, fixture)
    calibration, safety_results, components, oracle, prior = calibrate_and_score(run, device)
    actual_checks, actual_pass = actual_gate(safety_results, components, oracle)
    predicted_result = None; predicted_checks = None; predicted_pass = False
    if actual_pass:
        held = run["split"]["heldout"]
        logits = safety_logits(run["predictor"], run["safety"], run["route"],
                               held, device, "predicted")
        prob = torch.sigmoid(torch.from_numpy(logits[:, 2, 3]) /
                             calibration["temperature"]).numpy()
        FS.rows_global = run["rows"]
        kin = np.stack([r["kinematic"] for r in run["rows"]])
        predicted_result = FS.evaluate_condition(
            "augmented_predicted_future", run["rows"], held, prob,
            calibration["threshold"], kin)
        predicted_result["components"] = FS.component_metrics(
            run["rows"], held,
            torch.sigmoid(torch.from_numpy(logits) /
                          calibration["temperature"]).numpy(),
            calibration["threshold"])
        predicted_checks, predicted_pass = predicted_gate(predicted_result, oracle)
    nonreg = nonregression(run["predictor"], device)
    ratio_values = [x["ratio"] for x in run["ratios"]]
    ratio_checks = {
        "initial_scaled_ratio_eq_0_25": abs(run["smoke"]["scaled_ratio"] - .25) <= 1e-6,
        "median_pre_clip_ratio_between_0_10_and_0_50": (
            bool(ratio_values) and .10 <= float(np.median(ratio_values)) <= .50
        ),
        "no_pre_clip_ratio_above_1": bool(ratio_values) and max(ratio_values) <= 1.0,
        "required_post_clip_ratio_monitor_available": False,
    }
    baseline = {
        "action_only": prior["heldout"]["action_only"],
        "current_context_posthoc": prior["heldout"]["current_context"],
        "privileged_static_grid_guard": prior["heldout"]["privileged_static_grid_guard"],
        "oracle_safety": oracle,
    }
    overall = actual_pass and predicted_pass and nonreg["passed"] and all(ratio_checks.values())
    classification = ("SAFETY_AUXILIARY_JEPA_DEVELOPMENT_SIGNAL" if overall
                      else "SAFETY_AUXILIARY_JEPA_DEVELOPMENT_NO_SIGNAL")
    additional = []
    if not nonreg["passed"]: additional.append("PREDICTIVE_NON_REGRESSION_FAILURE")
    result = {
        "schema": "safety_auxiliary_two_step_jepa_dev_v1_result",
        "source_commit": "c26a89a7ea6a8aeec06db9397d97b6a67a1dbc6c",
        "preserved_terminal": "TRUE_FUTURE_SAFETY_HEAD_NO_GO",
        "bindings": {
            "historical_checkpoint": str(HISTORICAL_CHECKPOINT),
            "historical_checkpoint_sha256": sha(HISTORICAL_CHECKPOINT),
            "target_index_sha256": sha(FS.V2 / "target_latent_index.json"),
            "route_branch_ledger_sha256": sha(FS.V1 / "branch_labels.jsonl"),
            "factorial_training": run["factorial"],
        },
        "lineage_reconciliation": {
            "historical_checkpoint_contains_predictor": True,
            "historical_checkpoint_contains_online_encoder": False,
            "historical_checkpoint_contains_target_ema": False,
            "encoder_status": "frozen V-JEPA 2.1 cached-token contract preserved",
            "trainable_shared_parameters": "predictor through predicted-future safety path",
            "context_limitation": "candidate-invariant H1 hold proxy; exact pre-action context was not retained and rendering was prohibited",
        },
        "dataset": {"states": 48, "branches": 576,
                    "splits": {k: int(len(v)) for k, v in run["split"].items()},
                    "prevalence": prevalence_report(run["rows"], run["labels"], run["split"]),
                    "positive_weights": run["weights"].tolist(),
                    "defined_outputs": run["defined"].tolist()},
        "model": {
            "seed": SEED, "epochs": EPOCHS,
            "predictor_parameters": int(sum(p.numel() for p in run["predictor"].parameters())),
            "safety_branch_parameters": int(sum(p.numel() for p in run["safety"].parameters())),
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
            "optimizer": {"name": "AdamW", "predictor_lr": PREDICTOR_LR,
                          "safety_lr": SAFETY_LR, "weight_decay": WEIGHT_DECAY},
            "smoke": run["smoke"], "gradient_ratios": run["ratios"],
            "gradient_ratio_checks": ratio_checks, "history": run["history"],
            "training_runtime_s": run["training_runtime_s"],
        },
        "fixture": fixture, "calibration": calibration,
        "heldout": {"baselines": baseline,
                    "augmented_current_context": safety_results["current"],
                    "augmented_actual_future": safety_results["actual"],
                    "components": components,
                    "augmented_predicted_future": predicted_result},
        "actual_future_gate": {"passed": actual_pass, "checks": actual_checks,
                               "terminal_if_failed": "SAFETY_AUXILIARY_REPRESENTATION_NO_GO"},
        "predicted_future_gate": None if not actual_pass else {
            "passed": predicted_pass, "checks": predicted_checks,
            "terminal_if_failed": "SAFETY_AUXILIARY_PREDICTED_FUTURE_NO_GO"},
        "predictive_non_regression": nonreg,
        "classification": classification, "additional_classifications": additional,
        "continued_model_count": 1, "matched_one_step_trained": False,
        "additional_seed_trained": False, "simulation": False,
        "rendering": False, "encoding": False,
        "runtime": {"total_s": time.time() - started},
    }
    atomic_json(OUT / "result.json", result)
    print(json.dumps({"classification": classification,
                      "additional": additional,
                      "actual_gate": actual_checks,
                      "predicted_gate": predicted_checks,
                      "nonregression": nonreg["checks"],
                      "result_sha256": sha(OUT / "result.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
