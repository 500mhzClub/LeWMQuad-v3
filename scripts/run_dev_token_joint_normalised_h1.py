#!/usr/bin/env python3
"""DEVELOPMENT-ONLY normalised-state staged partial-unfreeze joint run (h=1).

Canonical predictive state: current, target and predicted all per-token
LayerNorm-ed.  The BEV auxiliary stays on RAW online tokens, and the collapse
gates are measured on raw tokens, so contraction remains visible.

NOT CLAIM BEARING.

Two arms from identical predictor weights and identical freshly initialised
optimiser state, because the predictor-only checkpoint carries weights only:

``frozen``   encoder fully frozen -- the matched continuation control
``partial``  ViT blocks 4-5 and final norm trainable at 0.1x the predictor LR

Representations are recomputed **every batch**.  No token cache survives an
optimiser step, because both the online encoder and the EMA target move during
the epoch.  Batch order is: online forward, target forward, loss, backward,
optimiser step, EMA update.

The BEV decoder and K/O head have frozen parameters but a live, differentiable
forward: the unchanged baseline BEV objective acts as a fixed spatial
regulariser on the encoder tokens.  Nothing is detached and nothing is wrapped
in ``no_grad`` on that path.
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

D = TOKEN_DIM
LN = lambda t: F.layer_norm(t, (D,))
RAW_GATES = {"true_token_variance": 0.5427520275115967,
             "effective_rank": 15.370167305091424,
             "raw_temporal_delta": 0.297139972448349}
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_joint_normalised_h1"
CKPT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_predictor_normalised_h1/predictor_normalised_epoch40.pt"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

EPOCHS = 40
BATCH = 16
PRED_LR = 3.0e-4
ENC_LR_SCALE = 0.1
EMA_MOMENTUM = 0.996
SEED = 2_026_080_591

TRAINABLE_ENCODER_PREFIXES = ("blocks.4.", "blocks.5.", "norm.")


def frozen_encoder_prefixes(enc):
    return [n for n, _ in enc.named_parameters()
            if not n.startswith(TRAINABLE_ENCODER_PREFIXES)]


def raster_batch(rows, idx, key):
    out = []
    for i in idx:
        e = rows[i][key]
        arr = np.fromfile(Path(e["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        out.append(arr.reshape(-1, 64, 64)[int(e["shard_row"])])
    return torch.from_numpy(np.stack(out)).long()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", choices=("frozen", "partial"), required=True)
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    arm_out = OUT / f"arm_{args.arm}"
    arm_out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    enc, dec, head = P.load_stack(device)
    import lewm.models.direct_egocentric_bev_signed_boundary_distance_state_v1 as msb

    # EMA target encoder: a detached copy, never receives gradients.
    target_enc = copy.deepcopy(enc).to(device).eval()
    for q in target_enc.parameters():
        q.requires_grad_(False)

    # BEV branch: parameters frozen, forward live and differentiable.
    for module in (dec, head):
        module.eval()
        for q in module.parameters():
            q.requires_grad_(False)

    # Encoder trainability by arm.
    for n, q in enc.named_parameters():
        q.requires_grad_(args.arm == "partial" and n.startswith(TRAINABLE_ENCODER_PREFIXES))
    enc.train(args.arm == "partial")

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not val:
        raise RuntimeError("designated roles must be non-empty")

    predictor = initialize_token_predictor_v1(SEED).to(device)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    predictor.load_state_dict(ck["state_dict"], strict=True)
    record = {
        "status": STATUS, "claim_bearing": False, "arm": args.arm,
        "predictor_init": {"path": str(CKPT), "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
                           "contains_optimizer_state": False, "contains_scheduler_state": False,
                           "note": "weights only, so both arms start from identical fresh optimiser state"},
        "optimiser": {"predictor_lr": PRED_LR, "encoder_lr": PRED_LR * ENC_LR_SCALE if args.arm == "partial" else None,
                      "encoder_lr_scale": ENC_LR_SCALE, "epochs": EPOCHS, "batch": BATCH},
        "trainable_encoder_prefixes": list(TRAINABLE_ENCODER_PREFIXES) if args.arm == "partial" else [],
        "representation_cache_across_steps": False,
        "state_space": {"current": "LN(raw_online)", "target": "LN(raw_ema_future)",
                        "predicted": "LN(predictor(current,action))"},
        "raw_token_gates": RAW_GATES,
        "bev_branch": {"parameters_frozen": True, "forward_active": True, "input": "raw online tokens",
                       "differentiable_wrt_encoder": True, "objective": "baseline, unchanged"},
        "split": {"train_pairs": len(fit), "selection_pairs": len(val)},
    }

    groups = [{"params": list(predictor.parameters()), "lr": PRED_LR}]
    if args.arm == "partial":
        enc_params = [q for n, q in enc.named_parameters() if q.requires_grad]
        if not enc_params:
            raise RuntimeError("partial arm has no trainable encoder parameters")
        groups.append({"params": enc_params, "lr": PRED_LR * ENC_LR_SCALE})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    before = {
        "encoder": {n: q.detach().clone() for n, q in enc.named_parameters()},
        "bev": {f"{m}.{n}": q.detach().clone() for m, mod in (("dec", dec), ("head", head))
                for n, q in mod.named_parameters()},
        "target": {n: q.detach().clone() for n, q in target_enc.named_parameters()},
        "predictor": {n: q.detach().clone() for n, q in predictor.named_parameters()},
    }

    if args.checks_only:
        record["checks_only"] = True
        record["wall_seconds"] = time.time() - started
        (arm_out / "checks.json").write_text(json.dumps(record, indent=2, default=str))
        print(json.dumps(record, indent=2, default=str))
        return 0

    oh_all, cmd_all = R.action_tensors([r["primitive"] for r in fit])
    g = torch.Generator().manual_seed(SEED)
    predictor.train()
    losses = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(fit), generator=g)
        for s in range(0, len(order), BATCH):
            sel = order[s:s + BATCH].tolist()
            if len(sel) < 2:
                continue
            cur_img = torch.stack([P.native_preprocess(fit[i]["cur"]) for i in sel]).to(device)
            nxt_img = torch.stack([P.native_preprocess(fit[i]["nxt"]) for i in sel]).to(device)

            # 1) raw online forward, recomputed every batch (grad in partial arm)
            cur_tok = enc.forward_tokens(cur_img)[:, 1:, :]
            # 2) raw EMA target forward, recomputed every batch, never grads
            with torch.no_grad():
                tgt_tok = target_enc.forward_tokens(nxt_img)[:, 1:, :]

            # canonical normalised predictive states
            cur_state = LN(cur_tok)
            tgt_state = LN(tgt_tok)
            pred = LN(predictor(cur_state, oh_all[sel].to(device), cmd_all[sel].to(device)))
            jepa = F.mse_loss(pred, tgt_state.detach())

            # BEV auxiliary on RAW online tokens: frozen params, live forward
            fields = head(dec(cur_tok))
            labels = raster_batch(fit, sel, "cur_sha").to(device)
            targets, _masks = msb.signed_boundary_distance_targets_v1(labels)
            bev = msb._boundary_huber_per_row_v1(fields, targets, labels).mean()

            loss = jepa + bev
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([q for grp in groups for q in grp["params"]], 1.0)
            # 3) optimiser step, then 4) EMA update -- in that order
            opt.step()
            with torch.no_grad():
                for (n, tq), (_n2, oq) in zip(target_enc.named_parameters(),
                                              enc.named_parameters(), strict=True):
                    tq.mul_(EMA_MOMENTUM).add_(oq.detach(), alpha=1.0 - EMA_MOMENTUM)
            losses.append({"epoch": epoch, "jepa": float(jepa), "bev": float(bev)})

    # ---- assertions ----
    def drift(group, mod_params):
        return max((float((q.detach() - group[n]).abs().max()) for n, q in mod_params), default=0.0)

    frozen_enc = [(n, q) for n, q in enc.named_parameters()
                  if not n.startswith(TRAINABLE_ENCODER_PREFIXES)]
    train_enc = [(n, q) for n, q in enc.named_parameters()
                 if n.startswith(TRAINABLE_ENCODER_PREFIXES)]
    record["assertions"] = {
        "frozen_encoder_drift": drift(before["encoder"], frozen_enc),
        "trainable_encoder_drift": drift(before["encoder"], train_enc),
        "bev_branch_drift": drift(before["bev"], [(f"{m}.{n}", q) for m, mod in (("dec", dec), ("head", head))
                                                  for n, q in mod.named_parameters()]),
        "predictor_drift": drift(before["predictor"], list(predictor.named_parameters())),
        "target_received_gradients": any(q.grad is not None for q in target_enc.parameters()),
        "target_moved": drift(before["target"], list(target_enc.named_parameters())),
        "frozen_encoder_params_with_grad": sum(1 for _n, q in frozen_enc if q.grad is not None),
    }
    a = record["assertions"]
    a["passed"] = (
        a["frozen_encoder_drift"] == 0.0
        and a["bev_branch_drift"] == 0.0
        and a["predictor_drift"] > 0.0
        and not a["target_received_gradients"]
        and a["target_moved"] > 0.0
        and a["frozen_encoder_params_with_grad"] == 0
        and (a["trainable_encoder_drift"] > 0.0 if args.arm == "partial" else a["trainable_encoder_drift"] == 0.0)
    )
    record["loss_trace_tail"] = losses[-5:]
    torch.save({"predictor": predictor.state_dict(), "encoder": enc.state_dict(),
                "target_encoder": target_enc.state_dict(), "arm": args.arm},
               arm_out / "joint_checkpoint.pt")
    record["wall_seconds"] = time.time() - started
    (arm_out / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"arm": args.arm, "assertions": record["assertions"],
                      "loss_tail": record["loss_trace_tail"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
