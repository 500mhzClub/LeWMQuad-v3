#!/usr/bin/env python3
"""DEVELOPMENT-ONLY masked pairwise margin joint run (h=1).

NOT CLAIM BEARING.

Identical to ``run_dev_token_counterfactual_matching_h1`` except that the
nine-way soft-label matching cross-entropy is removed and replaced by the
target-grounded masked pairwise margin::

    L = L_wpe_jepa + L_branch_jepa + lambda_rank * L_rank + L_bev

Retained unchanged: WP-E normalised-state JEPA, ordinary V3 own-successor branch
JEPA against live EMA targets, the frozen-reference successor embeddings,
partial unfreezing of ViT blocks 4-5 and the final norm, EMA, the BEV auxiliary,
the crop, the horizon, the data roles, the seed, the presentation order and the
learning rates.

Checkpoints are **not** selected by soft nCE.  Selection is by held-out masked
pairwise ranking accuracy on the 32 V3 selection groups, with masked MRR and
mean achieved margin as secondary metrics.
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
from lewm.models.go2_masked_pairwise_margin_v1 import (  # noqa: E402
    masked_pairwise_margin_v1, separation_mask_v1,
)
from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402
import scripts.run_dev_token_counterfactual_matching_h1 as PRIOR  # noqa: E402

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_token_pairwise_margin_h1"
CKPT = PRIOR.CKPT
CACHE = PRIOR.CACHE
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

EPOCHS = PRIOR.EPOCHS
BATCH = PRIOR.BATCH
GROUPS_PER_STEP = PRIOR.GROUPS_PER_STEP
PRED_LR = PRIOR.PRED_LR
ENC_LR_SCALE = PRIOR.ENC_LR_SCALE
EMA_MOMENTUM = PRIOR.EMA_MOMENTUM
SEED = PRIOR.SEED
TRAINABLE_ENCODER_PREFIXES = PRIOR.TRAINABLE_ENCODER_PREFIXES
EVAL_POINTS = 40

# Calibrated once on the fixed train-only batch (16 WP-E pairs, 2 V3 groups,
# 26 valid ordered pairs): ||grad rank|| is 0.4757x ||grad ordinary JEPA||
# unscaled with cosine -0.2046, so this places the ranking term at 25% of the
# ordinary JEPA encoder-gradient norm.  Frozen; no sweep.
LAMBDA_RANK = 0.5255


def LN(t):
    return F.layer_norm(t, (TOKEN_DIM,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", choices=("frozen", "partial"), required=True)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--lambda-rank", type=float, default=LAMBDA_RANK)
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
            args.arm == "partial" and name.startswith(TRAINABLE_ENCODER_PREFIXES))
    encoder.eval()          # frozen blocks stay in eval mode in BOTH arms

    predictor = initialize_token_predictor_v1(SEED).to(device)
    predictor.load_state_dict(
        torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    predictor.train()

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    validation = [r for r in rows if r["role"] == "checkpoint_selection"]
    if not fit or not validation:
        raise RuntimeError(
            f"designated roles are required and must be non-empty: "
            f"train={len(fit)} checkpoint_selection={len(validation)}")

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    train_ids = set(cache["train_state_ids"])
    selection_ids = set(cache["selection_state_ids"])
    keep = [i for i, s in enumerate(order) if s in train_ids]
    sel_rows = [i for i, s in enumerate(order) if s in selection_ids]

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)

    def endpoints_for(indices):
        out = torch.empty(len(indices), V3.BRANCHES_PER_GROUP, 2)
        for row, i in enumerate(indices):
            branches = sorted(records[order[i]]["branches"],
                              key=lambda b: int(b["action_id"]))
            out[row] = torch.tensor(
                [b["endpoint_state"]["base_pos_world"][:2] for b in branches])
        return out.to(device)

    tbar_flat = F.normalize(cache["successor_flat_unit"][keep].to(device), dim=-1)
    frozen_cos = PRIOR.group_cosine(tbar_flat, tbar_flat)
    train_mask = separation_mask_v1(frozen_cos, endpoints_for(keep))
    v3_commands = cache["commands"][keep].to(device)
    v3_paths = [(by_id[order[i]].current_path, by_id[order[i]].successor_paths)
                for i in keep]

    sel_tbar = F.normalize(cache["successor_flat_unit"][sel_rows].to(device), dim=-1)
    sel_cos = PRIOR.group_cosine(sel_tbar, sel_tbar)
    sel_mask = separation_mask_v1(sel_cos, endpoints_for(sel_rows))
    sel_commands = cache["commands"][sel_rows].to(device)
    sel_current = torch.stack([
        V3.preprocess_v3_frame_v1(by_id[order[i]].current_path) for i in sel_rows]).to(device)
    sel_scenes = [by_id[order[i]].scene_id for i in sel_rows]

    one_hot_all = torch.eye(V3.BRANCHES_PER_GROUP, device=device)
    one_hot_w, command_w = R.action_tensors([r["primitive"] for r in fit])

    def v3_images(indices):
        current = torch.stack([V3.preprocess_v3_frame_v1(v3_paths[i][0])
                               for i in indices]).to(device)
        successor = torch.stack([V3.preprocess_v3_frame_v1(p)
                                 for i in indices for p in v3_paths[i][1]]).to(device)
        return current, successor

    def compute_losses(pair_indices, group_indices):
        n_groups = len(group_indices)
        cur_img = torch.stack([P.native_preprocess(fit[i]["cur"])
                               for i in pair_indices]).to(device)
        nxt_img = torch.stack([P.native_preprocess(fit[i]["nxt"])
                               for i in pair_indices]).to(device)
        cur_tok = encoder.forward_tokens(cur_img)[:, 1:, :]
        with torch.no_grad():
            wpe_target = LN(target_encoder.forward_tokens(nxt_img)[:, 1:, :])
        wpe_pred = LN(predictor(LN(cur_tok), one_hot_w[pair_indices].to(device),
                                command_w[pair_indices].to(device)))
        wpe_jepa = F.mse_loss(wpe_pred, wpe_target)

        v3_cur_img, v3_suc_img = v3_images(group_indices)
        v3_cur = LN(encoder.forward_tokens(v3_cur_img)[:, 1:, :])
        with torch.no_grad():
            v3_ema = LN(target_encoder.forward_tokens(v3_suc_img)[:, 1:, :])
        state = v3_cur.repeat_interleave(V3.BRANCHES_PER_GROUP, dim=0)
        v3_pred = LN(predictor(state, one_hot_all.repeat(n_groups, 1),
                               v3_commands[group_indices].reshape(-1, 3)))
        branch_jepa = F.mse_loss(v3_pred, v3_ema)

        stats = masked_pairwise_margin_v1(
            v3_pred.reshape(n_groups, V3.BRANCHES_PER_GROUP, -1),
            tbar_flat[group_indices], train_mask[group_indices],
            frozen_cos[group_indices])

        fields = head(decoder(cur_tok))
        labels = PRIOR.raster_batch(fit, pair_indices, "cur_sha").to(device)
        targets, _masks = msb.signed_boundary_distance_targets_v1(labels)
        bev = msb._boundary_huber_per_row_v1(fields, targets, labels).mean()
        return wpe_jepa, branch_jepa, stats, bev

    @torch.no_grad()
    def selection_ranking():
        was_training = predictor.training
        predictor.eval()
        n = len(sel_rows)
        state = LN(encoder.forward_tokens(sel_current)[:, 1:, :])
        state = state.repeat_interleave(V3.BRANCHES_PER_GROUP, dim=0)
        pred = LN(predictor(state, one_hot_all.repeat(n, 1),
                            sel_commands.reshape(-1, 3)))
        stats = masked_pairwise_margin_v1(
            pred.reshape(n, V3.BRANCHES_PER_GROUP, -1), sel_tbar, sel_mask, sel_cos)
        predictor.train(was_training)
        return {"masked_pairwise_accuracy": stats.pairwise_accuracy,
                "masked_mrr": stats.masked_mrr,
                "mean_achieved_margin": stats.mean_achieved_margin,
                "hinge": float(stats.loss)}

    # ---- arm-independent presentation order, identical to the prior runner ----
    schedule_generator = torch.Generator().manual_seed(SEED)
    schedule = []
    n_train_groups = len(keep)
    cursor = 0
    group_order = torch.randperm(n_train_groups, generator=schedule_generator).tolist()
    for epoch in range(EPOCHS):
        pair_order = torch.randperm(len(fit), generator=schedule_generator).tolist()
        for start in range(0, len(pair_order), BATCH):
            pairs = pair_order[start:start + BATCH]
            if len(pairs) < 2:
                continue
            picked = []
            for _ in range(GROUPS_PER_STEP):
                if cursor >= n_train_groups:
                    group_order = torch.randperm(
                        n_train_groups, generator=schedule_generator).tolist()
                    cursor = 0
                picked.append(group_order[cursor]); cursor += 1
            schedule.append((epoch, pairs, picked))

    record = {
        "status": STATUS, "claim_bearing": False, "arm": args.arm,
        "objective": {
            "terms": ["wpe_jepa", "branch_jepa", "masked_pairwise_margin", "bev"],
            "removed": "nine-way soft-label matching cross-entropy",
            "margin": "m_ij = 1 - cos(t_bar_i, t_bar_j) (target gap)",
            "mask": "displacement > 0.05 m AND frozen cosine < 0.90",
            "averaging": "negatives within anchor, anchors within group",
            "lambda_rank": args.lambda_rank,
        },
        "predictor_init": {"path": str(CKPT),
                           "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest()},
        "frozen_reference_cache": {
            "path": str(CACHE),
            "sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest()},
        "mask_coverage": {
            "train_groups": n_train_groups,
            "train_ordered_pairs": int(train_mask.sum()),
            "train_valid_anchors": int((train_mask.sum(-1) > 0).sum()),
            "selection_groups": len(sel_rows),
            "selection_ordered_pairs": int(sel_mask.sum()),
            "selection_valid_anchors": int((sel_mask.sum(-1) > 0).sum()),
        },
        "checkpoint_selection_metric": "held-out masked pairwise ranking accuracy",
        "split": {"wpe_train_pairs": len(fit), "wpe_selection_pairs": len(validation)},
        "steps_per_epoch": len(schedule) // EPOCHS,
    }

    if args.calibrate:
        encoder_parameters = [p for p in encoder.parameters() if p.requires_grad]
        if not encoder_parameters:
            raise RuntimeError("--calibrate requires the partial arm")
        pairs, groups = schedule[0][1], schedule[0][2]
        wpe_jepa, branch_jepa, stats, _bev = compute_losses(pairs, groups)
        ordinary = wpe_jepa + branch_jepa

        def flat_grad(loss):
            grads = torch.autograd.grad(loss, encoder_parameters,
                                        retain_graph=True, allow_unused=True)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for g, p in zip(grads, encoder_parameters)])

        g_ordinary, g_rank = flat_grad(ordinary), flat_grad(stats.loss)
        n_ordinary, n_rank = float(g_ordinary.norm()), float(g_rank.norm())
        calibration = {
            "status": STATUS,
            "fixed_batch": {"wpe_pairs": len(pairs), "v3_groups": len(groups),
                            "valid_pairs_in_batch": stats.valid_pairs,
                            "valid_anchors_in_batch": stats.valid_anchors},
            "losses": {"wpe_jepa": float(wpe_jepa), "branch_jepa": float(branch_jepa),
                       "ordinary_jepa_total": float(ordinary),
                       "masked_pairwise_margin": float(stats.loss)},
            "batch_ranking": {"pairwise_accuracy": stats.pairwise_accuracy,
                              "masked_mrr": stats.masked_mrr,
                              "mean_achieved_margin": stats.mean_achieved_margin},
            "grad_norm_ordinary_jepa": f"{n_ordinary:.6e}",
            "grad_norm_rank_unscaled": f"{n_rank:.6e}",
            "raw_ratio_rank_over_ordinary": round(n_rank / n_ordinary, 4),
            "cos_ordinary_rank": round(
                float(F.cosine_similarity(g_ordinary[None], g_rank[None])), 6),
            "lambda_for_25pct": round(0.25 * n_ordinary / n_rank, 6),
            "lambda_in_use": args.lambda_rank,
            "achieved_fraction_at_lambda_in_use": round(
                args.lambda_rank * n_rank / n_ordinary, 4),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "lambda_calibration.json").write_text(json.dumps(calibration, indent=2))
        print(json.dumps(calibration, indent=2))
        return 0

    if args.lambda_rank <= 0.0:
        raise RuntimeError("--lambda-rank must be set from the calibration before training")

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

    eval_every = max(1, len(schedule) // EVAL_POINTS)
    curve, best = [], None
    for step, (epoch, pairs, groups) in enumerate(schedule, start=1):
        wpe_jepa, branch_jepa, stats, bev = compute_losses(pairs, groups)
        loss = wpe_jepa + branch_jepa + args.lambda_rank * stats.loss + bev
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in parameter_groups for p in g["params"]], 1.0)
        optimiser.step()
        with torch.no_grad():
            for (_n, tp), (_n2, op) in zip(target_encoder.named_parameters(),
                                           encoder.named_parameters(), strict=True):
                tp.mul_(EMA_MOMENTUM).add_(op.detach(), alpha=1.0 - EMA_MOMENTUM)
        if step % eval_every == 0 or step == len(schedule):
            ranking = selection_ranking()
            curve.append({"step": step, "epoch": epoch,
                          "wpe_jepa": float(wpe_jepa),
                          "branch_jepa": float(branch_jepa),
                          "hinge": float(stats.loss), "bev": float(bev),
                          "train_batch_pairwise_accuracy": stats.pairwise_accuracy,
                          **{f"selection_{k}": v for k, v in ranking.items()}})
            if best is None or ranking["masked_pairwise_accuracy"] > best["masked_pairwise_accuracy"]:
                best = {"step": step, **ranking}
                torch.save({"predictor": predictor.state_dict(),
                            "encoder": encoder.state_dict(),
                            "target_encoder": target_encoder.state_dict(),
                            "arm": args.arm, "step": step, "selection": ranking},
                           arm_out / "best_masked_accuracy.pt")

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
             else assertions["trainable_encoder_drift"] == 0.0))

    final_ranking = selection_ranking()
    torch.save({"predictor": predictor.state_dict(), "encoder": encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(), "arm": args.arm,
                "step": len(schedule), "selection": final_ranking},
               arm_out / "final.pt")
    record["assertions"] = assertions
    record["curve"] = curve
    record["best_selection"] = best
    record["final_selection"] = final_ranking
    record["wall_seconds"] = round(time.time() - started, 2)
    (arm_out / "result.json").write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"arm": args.arm, "assertions": assertions,
                      "best": best, "final": final_ranking,
                      "wall_seconds": record["wall_seconds"]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
