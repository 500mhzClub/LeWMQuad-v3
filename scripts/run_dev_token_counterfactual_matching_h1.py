#!/usr/bin/env python3
"""DEVELOPMENT-ONLY counterfactual-matching joint run (h=1).

NOT CLAIM BEARING.

Two arms from identical predictor weights, identical fresh optimiser state and
an identical arm-independent presentation order:

``frozen``   encoder fully frozen -- the non-JEPA predictor control
``partial``  ViT blocks 4-5 and final norm trainable at 0.1x the predictor LR

Objective, three independently meaned terms plus the unchanged BEV auxiliary::

    L = L_wpe_jepa + L_branch_jepa + lambda_match * L_match + L_bev

``L_branch_jepa`` is meaned over V3 *branches* and ``L_match`` over V3 *groups*,
so the nine branches per group do not create an implicit ninefold V3 weight
relative to the WP-E pair term.

The matching term is anchored entirely to frozen-reference embeddings
``t_bar`` -- both the target relation ``Q`` and the target columns of ``S``::

    Q_ij = softmax_j( cos(t_bar_i, t_bar_j) / tau_t )
    S_ij = cos(p_i, t_bar_j) / tau_p
    L_match = -(1/9) sum_ij Q_ij log softmax_j(S_ij)

Ordinary JEPA prediction is retained on every branch against its own **live**
normalised EMA successor, so the matching term supplements direct prediction
rather than replacing it.

Frozen encoder blocks are held in evaluation mode.  Parameter freezing alone
does not disable dropout or mutable buffers -- the defect recorded in the WP-E
retrospective validity note.
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

from lewm.datasets import go2_v3_counterfactual_branches_v1 as V3  # noqa: E402
from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_counterfactual_matching_h1"
CKPT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_predictor_normalised_h1/predictor_normalised_epoch40.pt"
CACHE = ROOT / ".generated/dev/DEVELOPMENT_ONLY_v3_frozen_reference_embeddings_v1/frozen_reference_train.pt"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

EPOCHS = 40
BATCH = 16
GROUPS_PER_STEP = 2
PRED_LR = 3.0e-4
ENC_LR_SCALE = 0.1
EMA_MOMENTUM = 0.996
SEED = 2_026_080_591
TAU_T = 0.05
TAU_P = 0.05
# Calibrated once on the fixed train-only batch after the frozen-column
# correction: ||grad match|| is 9.674x ||grad ordinary JEPA|| unscaled, so this
# places the matching term at ~25% of the ordinary JEPA encoder-gradient norm.
# Frozen; no sweep.
LAMBDA_MATCH = 0.0258
TRAINABLE_ENCODER_PREFIXES = ("blocks.4.", "blocks.5.", "norm.")


def LN(tokens: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(tokens, (TOKEN_DIM,))


def raster_batch(rows, idx, key):
    out = []
    for i in idx:
        entry = rows[i][key]
        arr = np.fromfile(Path(entry["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        out.append(arr.reshape(-1, 64, 64)[int(entry["shard_row"])])
    return torch.from_numpy(np.stack(out)).long()


def group_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine over the flattened per-group token vector: (G,9,D') x (G,9,D')."""
    return torch.einsum("gid,gjd->gij", F.normalize(a, dim=-1), F.normalize(b, dim=-1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", choices=("frozen", "partial"), required=True)
    ap.add_argument("--calibrate", action="store_true",
                    help="report encoder-gradient norms on the fixed train-only "
                         "batch and exit without training")
    args = ap.parse_args()
    device = torch.device(args.device)
    arm_out = OUT / f"arm_{args.arm}"
    arm_out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    encoder, decoder, head = P.load_stack(device)
    import lewm.models.direct_egocentric_bev_signed_boundary_distance_state_v1 as msb

    target_encoder = copy.deepcopy(encoder).to(device).eval()
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)
    for module in (decoder, head):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(
            args.arm == "partial" and name.startswith(TRAINABLE_ENCODER_PREFIXES)
        )
    # Frozen blocks stay in evaluation mode in BOTH arms.
    encoder.eval()

    predictor = initialize_token_predictor_v1(SEED).to(device)
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    predictor.load_state_dict(checkpoint["state_dict"], strict=True)
    predictor.train()

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    validation = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not validation:
        raise RuntimeError(
            f"designated roles are required and must be non-empty: "
            f"train={len(fit)} checkpoint_selection={len(validation)}"
        )

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    train_ids = set(cache["train_state_ids"])
    order_ids = cache["state_ids"]
    keep = [i for i, s in enumerate(order_ids) if s in train_ids]
    if len(keep) != len(train_ids):
        raise RuntimeError("frozen-reference cache does not cover the train split")
    tbar_flat = cache["successor_flat_unit"][keep].to(device)      # (96,9,D')
    v3_commands = cache["commands"][keep].to(device)               # (96,9,3)
    with torch.no_grad():
        Q = torch.softmax(group_cosine(tbar_flat, tbar_flat) / TAU_T, dim=-1)

    ledger_groups = None
    v3_paths = None
    if not args.calibrate or True:
        from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
        ledger = ceiling.AccessLedgerV1()
        all_groups = V3.load_branch_groups_v1("train", ceiling, ledger)
        by_id = {g.state_id: g for g in all_groups}
        ledger_groups = [by_id[order_ids[i]] for i in keep]
        v3_paths = [(g.current_path, g.successor_paths) for g in ledger_groups]

    one_hot_all = torch.eye(V3.BRANCHES_PER_GROUP, device=device)

    def v3_images(group_indices):
        current = torch.stack([
            V3.preprocess_v3_frame_v1(v3_paths[i][0]) for i in group_indices
        ]).to(device)
        successor = torch.stack([
            V3.preprocess_v3_frame_v1(p)
            for i in group_indices for p in v3_paths[i][1]
        ]).to(device)
        return current, successor

    def compute_losses(pair_indices, group_indices):
        n_groups = len(group_indices)
        # --- WP-E pair JEPA (meaned over pairs) ---
        cur_img = torch.stack([P.native_preprocess(fit[i]["cur"]) for i in pair_indices]).to(device)
        nxt_img = torch.stack([P.native_preprocess(fit[i]["nxt"]) for i in pair_indices]).to(device)
        cur_tok = encoder.forward_tokens(cur_img)[:, 1:, :]
        with torch.no_grad():
            wpe_target = LN(target_encoder.forward_tokens(nxt_img)[:, 1:, :])
        wpe_pred = LN(predictor(LN(cur_tok), one_hot_w[pair_indices].to(device),
                                command_w[pair_indices].to(device)))
        wpe_jepa = F.mse_loss(wpe_pred, wpe_target)

        # --- V3 branches: ordinary JEPA (meaned over branches) + matching (over groups) ---
        v3_cur_img, v3_suc_img = v3_images(group_indices)
        v3_cur_tok = LN(encoder.forward_tokens(v3_cur_img)[:, 1:, :])
        with torch.no_grad():
            v3_ema = LN(target_encoder.forward_tokens(v3_suc_img)[:, 1:, :])
        state = v3_cur_tok.repeat_interleave(V3.BRANCHES_PER_GROUP, dim=0)
        actions = one_hot_all.repeat(n_groups, 1)
        commands = v3_commands[group_indices].reshape(-1, 3)
        v3_pred = LN(predictor(state, actions, commands))
        branch_jepa = F.mse_loss(v3_pred, v3_ema)

        pred_groups = v3_pred.reshape(n_groups, V3.BRANCHES_PER_GROUP, -1)
        logits = torch.einsum(
            "gid,gjd->gij",
            F.normalize(pred_groups, dim=-1),
            tbar_flat[group_indices],
        ) / TAU_P
        target_q = Q[group_indices]
        match = -(target_q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        # --- BEV auxiliary, unchanged, on WP-E raw tokens ---
        fields = head(decoder(cur_tok))
        labels = raster_batch(fit, pair_indices, "cur_sha").to(device)
        targets, _masks = msb.signed_boundary_distance_targets_v1(labels)
        bev = msb._boundary_huber_per_row_v1(fields, targets, labels).mean()
        return wpe_jepa, branch_jepa, match, bev

    one_hot_w, command_w = R.action_tensors([r["primitive"] for r in fit])

    # ---- arm-independent presentation order ----
    schedule_generator = torch.Generator().manual_seed(SEED)
    schedule = []
    n_train_groups = len(keep)
    group_cursor = 0
    group_order = torch.randperm(n_train_groups, generator=schedule_generator).tolist()
    for epoch in range(EPOCHS):
        pair_order = torch.randperm(len(fit), generator=schedule_generator).tolist()
        for start in range(0, len(pair_order), BATCH):
            pairs = pair_order[start:start + BATCH]
            if len(pairs) < 2:
                continue
            picked = []
            for _ in range(GROUPS_PER_STEP):
                if group_cursor >= n_train_groups:
                    group_order = torch.randperm(n_train_groups,
                                                 generator=schedule_generator).tolist()
                    group_cursor = 0
                picked.append(group_order[group_cursor])
                group_cursor += 1
            schedule.append((epoch, pairs, picked))

    record = {
        "status": STATUS, "claim_bearing": False, "arm": args.arm,
        "predictor_init": {"path": str(CKPT),
                           "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest()},
        "frozen_reference_cache": {"path": str(CACHE),
                                   "sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest()},
        "objective": {
            "terms": ["wpe_jepa", "branch_jepa", "match", "bev"],
            "means": {"wpe_jepa": "over WP-E pairs",
                      "branch_jepa": "over V3 branches",
                      "match": "over V3 groups"},
            "tau_t": TAU_T, "tau_p": TAU_P, "lambda_match": LAMBDA_MATCH,
            "match_targets": "frozen reference t_bar for BOTH Q and S columns",
            "branch_jepa_target": "live normalised EMA successor",
        },
        "optimiser": {"predictor_lr": PRED_LR, "encoder_lr_scale": ENC_LR_SCALE,
                      "epochs": EPOCHS, "batch": BATCH,
                      "groups_per_step": GROUPS_PER_STEP},
        "split": {"wpe_train_pairs": len(fit), "wpe_selection_pairs": len(validation),
                  "v3_train_groups": n_train_groups,
                  "v3_selection_groups": len(cache["selection_state_ids"])},
        "crop_ratio": V3.V3_CENTRE_CROP_RATIO,
        "encoder_module_training_mode": encoder.training,
        "steps_per_epoch": len(schedule) // EPOCHS,
    }

    if args.calibrate:
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        if not encoder_parameters:
            raise RuntimeError("--calibrate requires the partial arm")
        pairs, groups = schedule[0][1], schedule[0][2]
        wpe_jepa, branch_jepa, match, _bev = compute_losses(pairs, groups)
        ordinary = wpe_jepa + branch_jepa

        def flat_grad(loss):
            grads = torch.autograd.grad(loss, encoder_parameters,
                                        retain_graph=True, allow_unused=True)
            return torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1)
                for g, p in zip(grads, encoder_parameters)
            ])

        g_ordinary, g_match = flat_grad(ordinary), flat_grad(match)
        n_ordinary, n_match = float(g_ordinary.norm()), float(g_match.norm())
        calibration = {
            "status": STATUS,
            "fixed_batch": {"wpe_pairs": len(pairs), "v3_groups": len(groups),
                            "v3_branches": len(groups) * V3.BRANCHES_PER_GROUP},
            "trainable_encoder_tensors": len(encoder_parameters),
            "trainable_encoder_parameters": sum(p.numel() for p in encoder_parameters),
            "losses": {"wpe_jepa": float(wpe_jepa), "branch_jepa": float(branch_jepa),
                       "ordinary_jepa_total": float(ordinary), "match": float(match)},
            "grad_norm_ordinary_jepa": f"{n_ordinary:.6e}",
            "grad_norm_match_unscaled": f"{n_match:.6e}",
            "raw_ratio_match_over_ordinary": round(n_match / n_ordinary, 4),
            "cos_ordinary_match": round(
                float(F.cosine_similarity(g_ordinary[None], g_match[None])), 6),
            "lambda_for_25pct": round(0.25 * n_ordinary / n_match, 4),
            "lambda_in_use": LAMBDA_MATCH,
            "achieved_fraction_at_lambda_in_use": round(
                LAMBDA_MATCH * n_match / n_ordinary, 4),
        }
        (OUT / "lambda_calibration.json").write_text(json.dumps(calibration, indent=2))
        print(json.dumps(calibration, indent=2))
        return 0

    parameter_groups = [{"params": list(predictor.parameters()), "lr": PRED_LR}]
    if args.arm == "partial":
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        if not encoder_parameters:
            raise RuntimeError("partial arm has no trainable encoder parameters")
        parameter_groups.append({"params": encoder_parameters,
                                 "lr": PRED_LR * ENC_LR_SCALE})
    optimiser = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)

    before = {
        "encoder": {n: p.detach().clone() for n, p in encoder.named_parameters()},
        "bev": {f"{m}.{n}": p.detach().clone()
                for m, mod in (("dec", decoder), ("head", head))
                for n, p in mod.named_parameters()},
        "target": {n: p.detach().clone() for n, p in target_encoder.named_parameters()},
        "predictor": {n: p.detach().clone() for n, p in predictor.named_parameters()},
    }

    trace = []
    for step, (epoch, pairs, groups) in enumerate(schedule):
        wpe_jepa, branch_jepa, match, bev = compute_losses(pairs, groups)
        loss = wpe_jepa + branch_jepa + LAMBDA_MATCH * match + bev
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in parameter_groups for p in g["params"]], 1.0)
        optimiser.step()
        with torch.no_grad():
            for (_n, tp), (_n2, op) in zip(target_encoder.named_parameters(),
                                           encoder.named_parameters(), strict=True):
                tp.mul_(EMA_MOMENTUM).add_(op.detach(), alpha=1.0 - EMA_MOMENTUM)
        if step % 50 == 0 or step == len(schedule) - 1:
            trace.append({"step": step, "epoch": epoch,
                          "wpe_jepa": float(wpe_jepa), "branch_jepa": float(branch_jepa),
                          "match": float(match), "bev": float(bev)})

    def drift(reference, named):
        return max((float((p.detach() - reference[n]).abs().max()) for n, p in named),
                   default=0.0)

    frozen_named = [(n, p) for n, p in encoder.named_parameters()
                    if not n.startswith(TRAINABLE_ENCODER_PREFIXES)]
    trainable_named = [(n, p) for n, p in encoder.named_parameters()
                       if n.startswith(TRAINABLE_ENCODER_PREFIXES)]
    assertions = {
        "frozen_encoder_drift": drift(before["encoder"], frozen_named),
        "trainable_encoder_drift": drift(before["encoder"], trainable_named),
        "bev_branch_drift": drift(before["bev"],
                                  [(f"{m}.{n}", p)
                                   for m, mod in (("dec", decoder), ("head", head))
                                   for n, p in mod.named_parameters()]),
        "predictor_drift": drift(before["predictor"], list(predictor.named_parameters())),
        "target_received_gradients": any(p.grad is not None
                                         for p in target_encoder.parameters()),
        "target_moved": drift(before["target"], list(target_encoder.named_parameters())),
        "frozen_encoder_params_with_grad": sum(1 for _n, p in frozen_named
                                               if p.grad is not None),
        "encoder_module_training_mode": encoder.training,
    }
    assertions["passed"] = (
        assertions["frozen_encoder_drift"] == 0.0
        and assertions["bev_branch_drift"] == 0.0
        and assertions["predictor_drift"] > 0.0
        and not assertions["target_received_gradients"]
        and assertions["target_moved"] > 0.0
        and assertions["frozen_encoder_params_with_grad"] == 0
        and assertions["encoder_module_training_mode"] is False
        and (assertions["trainable_encoder_drift"] > 0.0 if args.arm == "partial"
             else assertions["trainable_encoder_drift"] == 0.0)
    )
    record["assertions"] = assertions
    record["loss_trace"] = trace
    record["wall_seconds"] = round(time.time() - started, 2)
    torch.save({"predictor": predictor.state_dict(),
                "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "arm": args.arm}, arm_out / "joint_checkpoint.pt")
    (arm_out / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"arm": args.arm, "assertions": assertions,
                      "loss_tail": trace[-3:],
                      "wall_seconds": record["wall_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
