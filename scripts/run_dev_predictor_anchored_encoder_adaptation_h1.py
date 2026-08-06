#!/usr/bin/env python3
"""DEVELOPMENT-ONLY predictor-anchored encoder adaptation (h=1).

NOT CLAIM BEARING.

Every previous encoder-moving recipe let the predictor move too, and each one
improved generic predictability while weakening the correct-versus-shuffled
margin.  Here the action-conditioned predictor is **completely frozen** -- action
embedder, FiLM transformer and delta head -- but its forward pass stays inside
the autograd graph, so the only way to reduce the JEPA loss is to change the
encoder into a representation the *existing* action model already explains.

Trainable: ViT blocks 4-5 and the final encoder norm.  Nothing else.

Retained unchanged from the WP-E normalised-state recipe: EMA target encoder,
per-token normalised predictive state, MSE, h=1, the BEV auxiliary, the
designated WP-E roles, the schedule, the seed, the preprocessing and the encoder
learning rate.  All WP-F counterfactual matching and ranking terms are removed.

Selection is by correct-minus-shuffled changed-token margin, subject to hard
non-regression of raw variance, effective rank, temporal delta and spatial
geometry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import lewm.models.direct_egocentric_bev_state_jepa_v1 as _preload  # noqa: F401,E402

from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402
import scripts.run_dev_token_counterfactual_matching_h1 as PRIOR  # noqa: E402

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_predictor_anchored_encoder_adaptation_h1"
CKPT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_predictor_normalised_h1/predictor_normalised_epoch40.pt"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

EPOCHS = 40
BATCH = 16
PRED_LR = 3.0e-4
ENC_LR_SCALE = 0.1
ENCODER_LR = PRED_LR * ENC_LR_SCALE          # 3e-5, unchanged
EMA_MOMENTUM = 0.996
SEED = 2_026_080_591
TRAINABLE_ENCODER_PREFIXES = ("blocks.4.", "blocks.5.", "norm.")
DERANGEMENT_SEEDS = (11, 23, 37)

# Registered reference values from the original encoder (WP-E RAW_GATES).
RAW_GATES = {"raw_token_variance": 0.5427520275115967,
             "effective_rank": 15.370167305091424,
             "raw_temporal_delta": 0.297139972448349}
REFERENCE_MARGIN = 0.0496


def LN(t):
    return F.layer_norm(t, (TOKEN_DIM,))


def raw_health(tokens: torch.Tensor) -> dict:
    flat = tokens.reshape(-1, TOKEN_DIM).double()
    flat = flat - flat.mean(0)
    cov = flat.T.mm(flat) / (flat.shape[0] - 1)
    ev = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0.0)
    p = ev / ev.sum()
    return {"raw_token_variance": float(flat.square().mean()),
            "effective_rank": float((-(p * p.clamp_min(1e-12).log()).sum()).exp())}


def encode_raw(paths, encoder, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([P.native_preprocess(p) for p in paths[i:i + bs]])
        with torch.no_grad():
            out.append(encoder.forward_tokens(batch.to(device))[:, 1:, :].cpu())
    return torch.cat(out, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", action="store_true",
                    help="read-only gradient-flow check; trains nothing")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Original encoder state, exactly as accepted.
    encoder, decoder, head = P.load_stack(device)
    import lewm.models.direct_egocentric_bev_signed_boundary_distance_state_v1 as msb

    target_encoder = copy.deepcopy(encoder).to(device).eval()
    for q in target_encoder.parameters():
        q.requires_grad_(False)
    for module in (decoder, head):
        module.eval()
        for q in module.parameters():
            q.requires_grad_(False)

    for name, q in encoder.named_parameters():
        q.requires_grad_(name.startswith(TRAINABLE_ENCODER_PREFIXES))
    encoder.eval()          # frozen blocks must not run in training mode

    # Frozen predictor: no parameter gradients, but its forward stays in the graph.
    predictor = initialize_token_predictor_v1(SEED).to(device)
    predictor.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    for q in predictor.parameters():
        q.requires_grad_(False)
    predictor.eval()

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not val:
        raise RuntimeError(f"designated roles must be non-empty: "
                           f"train={len(fit)} checkpoint_selection={len(val)}")

    one_hot_fit, command_fit = R.action_tensors([r["primitive"] for r in fit])
    one_hot_val, command_val = R.action_tensors([r["primitive"] for r in val])
    trainable = [(n, q) for n, q in encoder.named_parameters() if q.requires_grad]

    def jepa_and_bev(indices):
        cur_img = torch.stack([P.native_preprocess(fit[i]["cur"]) for i in indices]).to(device)
        nxt_img = torch.stack([P.native_preprocess(fit[i]["nxt"]) for i in indices]).to(device)
        cur_tok = encoder.forward_tokens(cur_img)[:, 1:, :]
        with torch.no_grad():
            target_state = LN(target_encoder.forward_tokens(nxt_img)[:, 1:, :])
        predicted = LN(predictor(LN(cur_tok), one_hot_fit[indices].to(device),
                                 command_fit[indices].to(device)))
        jepa = F.mse_loss(predicted, target_state)
        fields = head(decoder(cur_tok))
        labels = PRIOR.raster_batch(fit, indices, "cur_sha").to(device)
        targets, _m = msb.signed_boundary_distance_targets_v1(labels)
        bev = msb._boundary_huber_per_row_v1(fields, targets, labels).mean()
        return jepa, bev

    # ---- schedule: identical to the WP-E normalised recipe ----
    generator = torch.Generator().manual_seed(SEED)
    schedule = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(fit), generator=generator).tolist()
        batches = [order[s:s + BATCH] for s in range(0, len(order), BATCH)]
        schedule.append([b for b in batches if len(b) >= 2])

    if args.verify:
        indices = schedule[0][0]
        predictor_params = [(n, q) for n, q in predictor.named_parameters()]
        for _n, q in predictor_params:
            q.grad = None
        for _n, q in trainable:
            q.grad = None

        # Rebuild the forward here so the predictor's input can be watched.
        cur_img = torch.stack([P.native_preprocess(fit[i]["cur"]) for i in indices]).to(device)
        nxt_img = torch.stack([P.native_preprocess(fit[i]["nxt"]) for i in indices]).to(device)
        cur_tok = encoder.forward_tokens(cur_img)[:, 1:, :]
        with torch.no_grad():
            target_state = LN(target_encoder.forward_tokens(nxt_img)[:, 1:, :])
        predictor_input = LN(cur_tok)
        predictor_input.retain_grad()
        predicted = LN(predictor(predictor_input, one_hot_fit[indices].to(device),
                                 command_fit[indices].to(device)))
        jepa = F.mse_loss(predicted, target_state)
        fields = head(decoder(cur_tok))
        labels = PRIOR.raster_batch(fit, indices, "cur_sha").to(device)
        targets, _m = msb.signed_boundary_distance_targets_v1(labels)
        bev = msb._boundary_huber_per_row_v1(fields, targets, labels).mean()

        # A real backward: gradients accumulate only where they are permitted.
        jepa.backward(retain_graph=True)
        enc_grads = [q.grad for _n, q in trainable]
        groups = {}
        for name, _q in predictor_params:
            groups.setdefault(name.split(".")[0], 0)
            groups[name.split(".")[0]] += 1
        record = {
            "status": STATUS, "claim_bearing": False, "mode": "read_only_verification",
            "predictor_parameter_tensors": len(predictor_params),
            "predictor_parameter_groups": groups,
            "predictor_params_requiring_grad": sum(
                1 for _n, q in predictor_params if q.requires_grad),
            "predictor_grad_attribute_set_after_backward": sum(
                1 for _n, q in predictor_params if q.grad is not None),
            "gradient_reaches_predictor_input": bool(
                predictor_input.grad is not None
                and float(predictor_input.grad.abs().max()) > 0.0),
            "predictor_input_grad_absmax": float(
                predictor_input.grad.abs().max()) if predictor_input.grad is not None else None,
            "trainable_encoder_tensors": len(trainable),
            "encoder_tensors_receiving_gradient": sum(
                1 for g in enc_grads if g is not None),
            "encoder_tensors_with_nonzero_gradient": sum(
                1 for g in enc_grads if g is not None and float(g.abs().max()) > 0.0),
            "min_abs_max_encoder_gradient": float(min(
                float(g.abs().max()) for g in enc_grads if g is not None)),
            "jepa_loss": float(jepa), "bev_loss": float(bev),
            "jepa_grad_norm_on_encoder": float(torch.cat(
                [g.reshape(-1) for g in enc_grads if g is not None]).norm()),
            "predictor_forward_inside_autograd_graph": bool(jepa.requires_grad),
            "encoder_lr": ENCODER_LR,
        }
        record["verification_passed"] = (
            record["predictor_params_requiring_grad"] == 0
            and record["predictor_grad_attribute_set_after_backward"] == 0
            and record["gradient_reaches_predictor_input"]
            and record["encoder_tensors_receiving_gradient"] == len(trainable)
            and record["encoder_tensors_with_nonzero_gradient"] == len(trainable)
            and record["predictor_forward_inside_autograd_graph"])
        (OUT / "gradient_flow_verification.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
        return 0

    optimiser = torch.optim.AdamW([q for _n, q in trainable], lr=ENCODER_LR,
                                  weight_decay=1e-4)
    before_predictor = {n: q.detach().clone() for n, q in predictor.named_parameters()}
    before_frozen = {n: q.detach().clone() for n, q in encoder.named_parameters()
                     if not n.startswith(TRAINABLE_ENCODER_PREFIXES)}
    before_trainable = {n: q.detach().clone() for n, q in trainable}
    before_bev = {f"{m}.{n}": q.detach().clone()
                  for m, mod in (("dec", decoder), ("head", head))
                  for n, q in mod.named_parameters()}

    # Frozen changed-token mask from the ORIGINAL encoder, fixed for every epoch.
    reference_encoder = P.load_stack(device)[0]
    change = (LN(encode_raw([r["nxt"] for r in fit], reference_encoder, device))
              - LN(encode_raw([r["cur"] for r in fit], reference_encoder, device))
              ).pow(2).mean(-1)
    threshold = float(torch.quantile(change.flatten().float(), 0.75))
    del change, reference_encoder
    torch.cuda.empty_cache()

    # Fixed spatial probe, trained once on the ORIGINAL encoder's true-future
    # tokens and applied unchanged at every epoch.  A freshly retrained probe is
    # run separately on the selected checkpoint.
    y_fit = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in fit])).long()
    y_val = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in val])).long()
    original_encoder = P.load_stack(device)[0]
    nxt_fit_original = encode_raw([r["nxt"] for r in fit], original_encoder, device)
    probe = R.SpatialProbe().to(device)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(y_fit.numpy().reshape(-1), minlength=3).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0),
                          dtype=torch.float32, device=device)
    weight = weight / weight.mean()
    gp = torch.Generator().manual_seed(SEED + 1)
    probe.train()
    for _ in range(30):
        order = torch.randperm(len(fit), generator=gp)
        for s in range(0, len(order), BATCH):
            sel = order[s:s + BATCH]
            popt.zero_grad(set_to_none=True)
            F.cross_entropy(probe(nxt_fit_original[sel].to(device)),
                            y_fit[sel].to(device), weight=weight).backward()
            popt.step()
    probe.eval()
    del nxt_fit_original, original_encoder
    torch.cuda.empty_cache()

    @torch.no_grad()
    def probe_occupied(tokens):
        preds = []
        for i in range(0, len(tokens), 64):
            preds.append(probe(tokens[i:i + 64].to(device)).argmax(1).cpu().numpy())
        pred = np.concatenate(preds, 0); truth = y_val.numpy()
        a, c = truth == 2, pred == 2
        union = int((a | c).sum())
        return int((a & c).sum()) / union if union else 0.0

    scenes = [r["scene"] for r in val]

    def evaluate():
        cur_raw = encode_raw([r["cur"] for r in val], encoder, device)
        nxt_ema = encode_raw([r["nxt"] for r in val], target_encoder, device)
        nxt_online = encode_raw([r["nxt"] for r in val], encoder, device)
        current, target_state = LN(cur_raw), LN(nxt_ema)
        mask = ((target_state - current).pow(2).mean(-1) > threshold)

        def predict(one_hot, command):
            out = []
            for i in range(0, len(current), 64):
                with torch.no_grad():
                    out.append(LN(predictor(current[i:i + 64].to(device),
                                            one_hot[i:i + 64].to(device),
                                            command[i:i + 64].to(device))).cpu())
            return torch.cat(out, 0)

        correct_pred = predict(one_hot_val, command_val)
        correct = R.metrics(correct_pred, target_state, current, mask)
        shuffled_preds, shuffled = [], []
        for seed in DERANGEMENT_SEEDS:
            perm = torch.randperm(len(val), generator=torch.Generator().manual_seed(seed))
            sp = predict(one_hot_val[perm], command_val[perm])
            shuffled_preds.append(sp)
            shuffled.append(R.metrics(sp, target_state, current, mask)["cosine"])
        margin = correct["cosine"] - float(np.mean(shuffled))
        per_scene = {}
        for scene in sorted(set(scenes)):
            ii = torch.tensor([i for i, s in enumerate(scenes) if s == scene])
            m = mask[ii]
            c = R.metrics(correct_pred[ii], target_state[ii], current[ii], m)["cosine"]
            per_scene[scene] = float(np.mean(
                [c - R.metrics(sp[ii], target_state[ii], current[ii], m)["cosine"]
                 for sp in shuffled_preds]))
        health = raw_health(nxt_online)
        health["raw_temporal_delta"] = float((nxt_online - cur_raw).abs().mean())
        return {
            "correct_minus_shuffled": margin,
            "correct_changed_cosine": correct["cosine"],
            "shuffled_changed_cosine_mean": float(np.mean(shuffled)),
            "persistence_changed_cosine": R.metrics(
                current, target_state, current, mask)["cosine"],
            **health,
            "fixed_probe_occupied_iou_predicted": probe_occupied(correct_pred),
            "fixed_probe_occupied_iou_persistence": probe_occupied(cur_raw),
            "fixed_probe_occupied_iou_true_future": probe_occupied(nxt_ema),
            "per_scene_correct_minus_shuffled": per_scene,
        }

    record = {
        "status": STATUS, "claim_bearing": False,
        "design": {
            "predictor": "FULLY FROZEN (action embedder, FiLM transformer, delta head); "
                         "forward remains inside the autograd graph",
            "trainable": list(TRAINABLE_ENCODER_PREFIXES),
            "removed": "all WP-F counterfactual matching and ranking terms",
            "objective": "normalised-state JEPA MSE + unchanged BEV auxiliary",
        },
        "predictor_init": {"path": str(CKPT),
                           "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest()},
        "encoder_lr": ENCODER_LR, "epochs": EPOCHS, "batch": BATCH,
        "ema_momentum": EMA_MOMENTUM, "seed": SEED,
        "changed_token_threshold": threshold,
        "reference": {"margin": REFERENCE_MARGIN, **RAW_GATES},
        "split": {"train_pairs": len(fit), "selection_pairs": len(val)},
        "epochs_log": [],
    }
    (OUT / "epoch_0_baseline.json").write_text(json.dumps(evaluate(), indent=2, default=str))
    record["epoch_0_baseline"] = json.loads((OUT / "epoch_0_baseline.json").read_text())

    for epoch, batches in enumerate(schedule, start=1):
        encoder.eval()
        for indices in batches:
            jepa, bev = jepa_and_bev(indices)
            loss = jepa + bev
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([q for _n, q in trainable], 1.0)
            optimiser.step()
            with torch.no_grad():
                for (_n, tq), (_n2, oq) in zip(target_encoder.named_parameters(),
                                               encoder.named_parameters(), strict=True):
                    tq.mul_(EMA_MOMENTUM).add_(oq.detach(), alpha=1.0 - EMA_MOMENTUM)
        metrics = evaluate()
        metrics["epoch"] = epoch
        metrics["jepa_loss_last_batch"] = float(jepa)
        metrics["bev_loss_last_batch"] = float(bev)
        record["epochs_log"].append(metrics)
        torch.save({"encoder": encoder.state_dict(),
                    "target_encoder": target_encoder.state_dict(),
                    "epoch": epoch, "metrics": metrics},
                   OUT / f"epoch_{epoch:02d}.pt")
        (OUT / "progress.json").write_text(json.dumps(record, indent=2, default=str))

    record["assertions"] = {
        "predictor_drift": max(float((q.detach() - before_predictor[n]).abs().max())
                               for n, q in predictor.named_parameters()),
        "predictor_params_with_grad": sum(1 for q in predictor.parameters()
                                          if q.grad is not None),
        "frozen_encoder_drift": max(
            float((q.detach() - before_frozen[n]).abs().max())
            for n, q in encoder.named_parameters()
            if not n.startswith(TRAINABLE_ENCODER_PREFIXES)),
        "trainable_encoder_drift": max(
            float((q.detach() - before_trainable[n]).abs().max()) for n, q in trainable),
        "bev_branch_drift": max(
            float((q.detach() - before_bev[f"{m}.{n}"]).abs().max())
            for m, mod in (("dec", decoder), ("head", head))
            for n, q in mod.named_parameters()),
        "encoder_module_training_mode": encoder.training,
    }
    record["wall_seconds"] = round(time.time() - started, 2)
    (OUT / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"epochs": len(record["epochs_log"]),
                      "assertions": record["assertions"],
                      "wall_seconds": record["wall_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
