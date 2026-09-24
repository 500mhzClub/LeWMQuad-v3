#!/usr/bin/env python3
"""DEVELOPMENT-ONLY normalised-state token predictor (h=1), frozen encoder.

NOT CLAIM BEARING.

This is **V-JEPA-inspired target normalisation adapted into a consistent
autoregressive normalised token state** -- not a copy of the masked V-JEPA
training architecture.  There is no masking, no context/target spatial
asymmetry, no momentum schedule and no predictor-output variance hinge.  What is
borrowed is the per-token ``F.layer_norm`` on the target path, extended to the
predictor's input and output so that all three live in one canonical space:

    current_state   = F.layer_norm(raw_online_current, (D,))
    target_state    = F.layer_norm(raw_ema_future,     (D,))
    predicted_state = F.layer_norm(predictor(current_state, action), (D,))

``predicted_state`` is fed directly back as the next rollout state, so input,
output and target must share a space.  Measured motivation: raw tokens have
per-token mean ~0 but std 0.885 with a learned final-norm affine (scale
0.76-1.09), so ``max|tokens - layer_norm(tokens)| = 1.223`` -- normalising only
the target would have silently changed the task.

Raw online tokens are retained separately for the unchanged BEV auxiliary and
for the collapse gates (variance, effective rank, temporal delta) and raw-token
spatial qualification, so contraction remains visible in the space where it
would occur.
"""
from __future__ import annotations

import argparse
import copy
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

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_predictor_normalised_h1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
D = TOKEN_DIM
EPOCHS, BATCH, LR, SEED = 40, 32, 3.0e-4, 2_026_080_591
DERANGEMENT_SEEDS = (11, 23, 37)

LN = lambda t: F.layer_norm(t, (D,))


@torch.no_grad()
def encode_raw(paths, enc, device, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([P.native_preprocess(p) for p in paths[i:i + bs]])
        out.append(enc.forward_tokens(batch.to(device))[:, 1:, :].cpu())
    return torch.cat(out, 0)


def eff_rank(t):
    x = t.reshape(-1, D).double()
    x = x - x.mean(0)
    ev = torch.linalg.svdvals(x) ** 2
    p = ev / ev.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    enc, dec, head = P.load_stack(device)
    for m in (enc, dec, head):
        m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
    # Frozen stage: the EMA target encoder is an exact copy of the frozen online
    # encoder, so EMA-current and online-current coincide here by construction.
    target_enc = copy.deepcopy(enc).to(device).eval()
    for q in target_enc.parameters():
        q.requires_grad_(False)

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not val:
        raise RuntimeError("designated roles must be non-empty")

    raw_cur_f = encode_raw([r["cur"] for r in fit], enc, device)
    raw_nxt_f = encode_raw([r["nxt"] for r in fit], target_enc, device)
    raw_cur_v = encode_raw([r["cur"] for r in val], enc, device)
    raw_nxt_v = encode_raw([r["nxt"] for r in val], target_enc, device)
    ema_cur_f = encode_raw([r["cur"] for r in fit], target_enc, device)
    ema_cur_v = encode_raw([r["cur"] for r in val], target_enc, device)

    cur_f, tgt_f = LN(raw_cur_f), LN(raw_nxt_f)
    cur_v, tgt_v = LN(raw_cur_v), LN(raw_nxt_v)
    emac_f, emac_v = LN(ema_cur_f), LN(ema_cur_v)

    # changed-token mask from normalised EMA-current vs EMA-future
    change_f = (LN(raw_nxt_f) - emac_f).pow(2).mean(-1)
    thr = float(torch.quantile(change_f.flatten().float(), 0.75))
    mask_v = (tgt_v - emac_v).pow(2).mean(-1) >= thr

    record = {
        "status": STATUS, "claim_bearing": False,
        "description": ("V-JEPA-inspired target normalisation adapted into a consistent "
                        "autoregressive normalised token state; NOT a copy of the masked "
                        "V-JEPA training architecture (no masking, no context/target spatial "
                        "asymmetry, no momentum schedule, no predictor-output variance hinge)"),
        "state_space": {"current": "F.layer_norm(raw_online_current,(D,))",
                        "target": "F.layer_norm(raw_ema_future,(D,))",
                        "predicted": "F.layer_norm(predictor(current_state,action),(D,))",
                        "rollout_feedback": "predicted_state used directly"},
        "split": {"train_pairs": len(fit), "selection_pairs": len(val),
                  "train_scenes": len({r["scene"] for r in fit}),
                  "selection_scenes": len({r["scene"] for r in val})},
        "changed_token_threshold": {"value": thr, "source": "train LN(EMA-current) vs LN(EMA-future) 75th pct",
                                    "selection_changed_fraction": float(mask_v.float().mean())},
        "frozen_stage_note": ("encoder frozen, so LN(EMA-current) and current_state coincide; "
                              "the two persistence baselines only diverge once the encoder moves"),
    }

    predictor = initialize_token_predictor_v1(SEED).to(device)
    with torch.no_grad():
        ident = LN(predictor(cur_v[:8].to(device),
                             *[t.to(device) for t in R.action_tensors([r["primitive"] for r in val[:8]])])).cpu()
    record["identity_at_init"] = {
        "max_abs_diff_from_current_state": float((ident - cur_v[:8]).abs().max()),
        "note": "LN is idempotent on already-normalised input, so zero-delta init stays identity",
    }

    opt = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=1e-4)
    oh_f, cmd_f = R.action_tensors([r["primitive"] for r in fit])
    oh_v, cmd_v = R.action_tensors([r["primitive"] for r in val])
    g = torch.Generator().manual_seed(SEED)

    @torch.no_grad()
    def predict(states, oh, cmd):
        predictor.eval()
        out = []
        for i in range(0, len(states), 64):
            out.append(LN(predictor(states[i:i + 64].to(device),
                                    oh[i:i + 64].to(device), cmd[i:i + 64].to(device))).cpu())
        predictor.train()
        return torch.cat(out, 0)

    curve = []
    predictor.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(fit), generator=g)
        for s in range(0, len(order), BATCH):
            sel = order[s:s + BATCH]
            opt.zero_grad(set_to_none=True)
            pred = LN(predictor(cur_f[sel].to(device), oh_f[sel].to(device), cmd_f[sel].to(device)))
            loss = F.mse_loss(pred, tgt_f[sel].to(device).detach())
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
        if (epoch + 1) in (1, 5, 10, 20, 30, 40):
            p = predict(cur_v, oh_v, cmd_v)
            curve.append({"epoch": epoch + 1,
                          "changed_cosine": float(F.cosine_similarity(p, tgt_v, dim=-1)[mask_v].mean())})
    predictor.eval()
    record["selection_curve"] = curve

    # ---- arms, all in normalised space ----
    arms = {"correct_action": predict(cur_v, oh_v, cmd_v),
            "persistence_identity": cur_v.clone(),
            "persistence_ema_current": emac_v.clone(),
            "neutral_action": predict(cur_v, *R.action_tensors(["hold"] * len(val)))}
    for k, s in enumerate(DERANGEMENT_SEEDS):
        rg = np.random.default_rng(s)
        perm = np.arange(len(val))
        while True:
            rg.shuffle(perm)
            if not (perm == np.arange(len(perm))).any():
                break
        arms[f"shuffled_action_{k}"] = predict(cur_v, *R.action_tensors([val[i]["primitive"] for i in perm]))

    def met(t, base):
        cos = F.cosine_similarity(t, tgt_v, dim=-1)[mask_v]
        err = (t - tgt_v).pow(2).mean(-1)[mask_v]
        b = (base - tgt_v).pow(2).mean(-1)[mask_v]
        return {"changed_cosine": float(cos.mean()), "changed_mse": float(err.mean()),
                "normalised_error_vs_identity": float(err.mean() / b.mean().clamp_min(1e-12))}
    record["prediction"] = {n: met(t, cur_v) for n, t in arms.items()}

    # ---- frozen probe trained on NORMALISED true EMA target states ----
    def y_of(rws):
        out = []
        for r in rws:
            e = r["nxt_sha"]
            a = np.fromfile(Path(e["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
            out.append(a.reshape(-1, 64, 64)[int(e["shard_row"])])
        return torch.from_numpy(np.stack(out)).long()
    y_fit, y_val = y_of(fit), y_of(val)
    probe = R.SpatialProbe().to(device)
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    c = np.bincount(y_fit.numpy().reshape(-1), minlength=3).astype(float)
    w = torch.tensor(c.sum() / np.maximum(c, 1.0), dtype=torch.float32, device=device)
    w = w / w.mean()
    gp = torch.Generator().manual_seed(SEED + 1)
    probe.train()
    for _ in range(30):
        idx = torch.randperm(len(fit), generator=gp)
        for s in range(0, len(idx), 32):
            sel = idx[s:s + 32]
            popt.zero_grad(set_to_none=True)
            F.cross_entropy(probe(tgt_f[sel].to(device)), y_fit[sel].to(device), weight=w).backward()
            popt.step()
    probe.eval()

    @torch.no_grad()
    def probe_eval(tok):
        ps = []
        for i in range(0, len(tok), 64):
            ps.append(probe(tok[i:i + 64].to(device)).argmax(1).cpu().numpy())
        pred_r, truth = np.concatenate(ps, 0), y_val.numpy()
        o = {}
        for k, nm in ((0, "unknown"), (1, "free"), (2, "occupied")):
            a, cc = truth == k, pred_r == k
            sup, inter, uni = int(a.sum()), int((a & cc).sum()), int((a | cc).sum())
            o[nm + "_iou"] = inter / uni if uni else None
            o[nm + "_recall"] = inter / sup if sup else None
            o[nm + "_precision"] = inter / int(cc.sum()) if int(cc.sum()) else None
        return o
    record["spatial_probe_normalised"] = {"trained_on": "normalised true EMA target states",
                                          "true_future": probe_eval(tgt_v)}
    for n, t in arms.items():
        record["spatial_probe_normalised"][n] = probe_eval(t)

    # ---- collapse gates on RAW online tokens ----
    record["raw_token_health"] = {
        "true_token_variance": float(raw_nxt_v.reshape(-1, D).var(0).mean()),
        "effective_rank": eff_rank(raw_nxt_v),
        "raw_temporal_delta": float((raw_nxt_v - raw_cur_v).abs().mean()),
        "measured_on": "raw online/EMA tokens before normalisation",
    }
    torch.save({"state_dict": predictor.state_dict(), "epochs": EPOCHS, "seed": SEED,
                "state_space": "normalised"}, OUT / "predictor_normalised_epoch40.pt")
    record["wall_seconds"] = time.time() - started
    (OUT / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({k: record[k] for k in
                      ("identity_at_init", "changed_token_threshold", "selection_curve",
                       "prediction", "raw_token_health")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
