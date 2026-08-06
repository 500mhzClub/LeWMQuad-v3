#!/usr/bin/env python3
"""Audit: is the frozen-column matching loss confounded by latent-basis drift?

DEVELOPMENT ONLY.  NOT CLAIM BEARING.  No training is launched and no
hyper-parameter, architecture or loss is changed.

``S_ij = cos(p_i, t_bar_j)`` compares predictions produced in the **moving**
encoder's coordinate system against targets frozen in the **original** encoder's
coordinate system.  If the partial arm rotated its latent basis, matching would
degrade for a reason that has nothing to do with action discrimination.

The test is an orthogonal Procrustes alignment fitted on the 96 V3 *train*
groups only and applied unchanged to the 32 held-out selection groups.  A
rotation cannot manufacture action information -- it can only undo a change of
basis -- so a large repair implicates basis drift and a small one exonerates it.

Reference values, so the matching numbers are interpretable:
    CE floor  (perfect predictor p_i = t_bar_i) = H(Q)
    CE with uniform S (no information)          = log 9 = 2.1972
    top-1 chance                                = 1/9   = 0.1111
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
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
import scripts.run_dev_token_counterfactual_matching_h1 as RUN  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DERANGEMENT_SEEDS = (11, 23, 37)
NINE = V3.BRANCHES_PER_GROUP


def LN(t):
    return F.layer_norm(t, (TOKEN_DIM,))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two representations over matched rows."""
    x = x.double() - x.double().mean(0, keepdim=True)
    y = y.double() - y.double().mean(0, keepdim=True)
    cross = float((y.T @ x).pow(2).sum())
    xx = float((x.T @ x).pow(2).sum()) ** 0.5
    yy = float((y.T @ y).pow(2).sum()) ** 0.5
    return cross / (xx * yy) if xx > 0 and yy > 0 else float("nan")


def orthogonal_procrustes(live: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
    """Least-squares R in O(d) minimising ||live @ R - frozen||_F."""
    m = live.double().T @ frozen.double()
    u, _s, vh = torch.linalg.svd(m, full_matrices=False)
    return (u @ vh).float()


def matching_report(pred_flat, tbar_flat, q_matrix, tau):
    """CE, normalised error, top-1 and diagonal probability for one panel."""
    pn = F.normalize(pred_flat, dim=-1)
    logits = torch.einsum("gid,gjd->gij", pn, tbar_flat) / tau
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    groups, n, _ = pred_flat.shape
    eye = torch.eye(n, dtype=torch.bool, device=pred_flat.device)
    own = (pred_flat - tbar_flat).pow(2).sum(-1)
    other = (pred_flat[:, :, None, :] - tbar_flat[:, None, :, :]).pow(2).sum(-1)
    other_mean = other.masked_fill(eye[None], 0).sum(-1) / (n - 1)
    return {
        "ce": float(-(q_matrix * log_probs).sum(-1).mean()),
        "normalised_matching_error": float((own.mean() / other_mean.mean())),
        "top1": float((logits.argmax(-1)
                       == torch.arange(n, device=pred_flat.device)[None]).float().mean()),
        "diagonal_probability": float(torch.diagonal(probs, dim1=1, dim2=2).mean()),
        "groups": int(groups),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    train_ids, selection_ids = set(cache["train_state_ids"]), set(cache["selection_state_ids"])
    idx = {"train": [i for i, s in enumerate(order) if s in train_ids],
           "selection": [i for i, s in enumerate(order) if s in selection_ids]}
    tbar_tokens = {k: cache["successor_normalised_tokens"][v].to(device) for k, v in idx.items()}
    tbar_flat = {k: F.normalize(cache["successor_flat_unit"][v].to(device), dim=-1)
                 for k, v in idx.items()}
    commands = {k: cache["commands"][v].to(device) for k, v in idx.items()}

    q_matrix, entropy_floor = {}, {}
    for k in idx:
        c = RUN.group_cosine(tbar_flat[k], tbar_flat[k])
        q = torch.softmax(c / RUN.TAU_T, dim=-1)
        q_matrix[k] = q
        entropy_floor[k] = float(-(q * q.clamp_min(1e-12).log()).sum(-1).mean())

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    groups = {k: [by_id[order[i]] for i in v] for k, v in idx.items()}

    one_hot_all = torch.eye(NINE, device=device)
    report = {"status": STATUS, "claim_bearing": False,
              "reference": {"ce_floor_entropy_of_Q": entropy_floor,
                            "ce_uniform_no_information": math.log(9),
                            "top1_chance": 1.0 / 9.0,
                            "tau_t": RUN.TAU_T, "tau_p": RUN.TAU_P},
              "arms": {}}

    # WP-E selection result carried across for the transfer comparison.
    paired = json.loads((RUN.OUT / "paired_evaluation.json").read_text())

    for arm in ("frozen", "partial"):
        state = torch.load(RUN.OUT / f"arm_{arm}" / "joint_checkpoint.pt",
                           map_location="cpu", weights_only=False)
        encoder = P.load_stack(device)[0]
        encoder.load_state_dict(state["encoder"], strict=True)
        target_encoder = P.load_stack(device)[0]
        target_encoder.load_state_dict(state["target_encoder"], strict=True)
        predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
        predictor.load_state_dict(state["predictor"], strict=True)
        for module in (encoder, target_encoder, predictor):
            module.to(device).eval().requires_grad_(False)

        live_online, live_ema, predictions = {}, {}, {}
        for split, panel in groups.items():
            online, ema, preds = [], [], []
            for gi, group in enumerate(panel):
                batch = torch.stack([V3.preprocess_v3_frame_v1(p)
                                     for p in group.successor_paths]).to(device)
                with torch.no_grad():
                    online.append(LN(encoder.forward_tokens(batch)[:, 1:, :]))
                    ema.append(LN(target_encoder.forward_tokens(batch)[:, 1:, :]))
                    current = LN(encoder.forward_tokens(
                        V3.preprocess_v3_frame_v1(group.current_path)[None].to(device)
                    )[:, 1:, :]).repeat(NINE, 1, 1)
                    preds.append(LN(predictor(current, one_hot_all, commands[split][gi])))
            live_online[split] = torch.stack(online)
            live_ema[split] = torch.stack(ema)
            predictions[split] = torch.stack(preds)

        entry = {}

        # --- 1. same-branch cosine, live vs frozen reference ---
        entry["same_branch_cosine_to_frozen_reference"] = {}
        for split in idx:
            row = {}
            for name, live in (("online", live_online[split]), ("ema", live_ema[split])):
                a = F.normalize(live.reshape(len(groups[split]), NINE, -1), dim=-1)
                cos = (a * tbar_flat[split]).sum(-1)
                row[name] = {"mean": float(cos.mean()), "min": float(cos.min()),
                             "p10": float(cos.flatten().quantile(0.10))}
            entry["same_branch_cosine_to_frozen_reference"][split] = row

        # --- 2. linear CKA, complete representations ---
        entry["linear_cka_frozen_vs_live"] = {}
        for split in idx:
            n_groups = len(groups[split])
            token_live = live_online[split].reshape(-1, TOKEN_DIM)
            token_frozen = tbar_tokens[split].reshape(-1, TOKEN_DIM)
            example_live = live_online[split].reshape(n_groups * NINE, -1)
            example_frozen = tbar_tokens[split].reshape(n_groups * NINE, -1)
            entry["linear_cka_frozen_vs_live"][split] = {
                "token_level_192d": round(linear_cka(token_live, token_frozen), 6),
                "example_level_flattened": round(
                    linear_cka(example_live[:, :4096], example_frozen[:, :4096]), 6),
            }

        # --- 3. train-only orthogonal Procrustes ---
        rotation = orthogonal_procrustes(
            live_online["train"].reshape(-1, TOKEN_DIM),
            tbar_tokens["train"].reshape(-1, TOKEN_DIM),
        ).to(device)
        deviation = float((rotation @ rotation.T
                           - torch.eye(TOKEN_DIM, device=device)).abs().max())
        entry["procrustes"] = {
            "fitted_on": "96 V3 train groups only",
            "applied_unchanged_to": "32 V3 selection groups",
            "orthogonality_max_deviation": deviation,
            "identity_distance_max": float((rotation
                                            - torch.eye(TOKEN_DIM, device=device)).abs().max()),
            "mean_abs_offdiagonal": float(
                rotation.masked_select(~torch.eye(TOKEN_DIM, dtype=torch.bool,
                                                  device=device)).abs().mean()),
        }
        for split in idx:
            n_groups = len(groups[split])
            aligned = (live_online[split].reshape(-1, TOKEN_DIM) @ rotation
                       ).reshape(n_groups, NINE, -1)
            a = F.normalize(aligned, dim=-1)
            entry["procrustes"][f"same_branch_cosine_after_{split}"] = float(
                (a * tbar_flat[split]).sum(-1).mean())

        # --- 4 & 5. matching before/after alignment, train and selection ---
        entry["matching"] = {}
        for split in idx:
            n_groups = len(groups[split])
            before = predictions[split].reshape(n_groups, NINE, -1)
            after = (predictions[split].reshape(-1, TOKEN_DIM) @ rotation
                     ).reshape(n_groups, NINE, -1)
            entry["matching"][split] = {
                "before_alignment": matching_report(before, tbar_flat[split],
                                                    q_matrix[split], RUN.TAU_P),
                "after_train_only_alignment": matching_report(after, tbar_flat[split],
                                                              q_matrix[split], RUN.TAU_P),
            }

        # --- 6. correct-versus-shuffled directly on V3 branch groups ---
        change = (live_ema["train"] - live_online["train"]).pow(2).mean(-1)
        threshold = float(torch.quantile(change.flatten().float(), 0.75))
        entry["v3_correct_vs_shuffled"] = {"changed_token_threshold": threshold}
        for split in idx:
            n_groups = len(groups[split])
            target = live_ema[split]
            predicted = predictions[split]
            current = torch.stack([
                LN(encoder.forward_tokens(
                    V3.preprocess_v3_frame_v1(g.current_path)[None].to(device))[:, 1:, :])[0]
                for g in groups[split]
            ]).unsqueeze(1).repeat(1, NINE, 1, 1)
            mask = ((target - current).pow(2).mean(-1) > threshold)
            flat = lambda t: t.reshape(n_groups * NINE, 256, -1)  # noqa: E731
            correct = R.metrics(flat(predicted), flat(target), flat(current),
                                mask.reshape(n_groups * NINE, 256))
            shuffled_scores = []
            for seed in DERANGEMENT_SEEDS:
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(NINE, generator=g).to(device)
                shuffled = predicted[:, perm, :, :]
                shuffled_scores.append(R.metrics(
                    flat(shuffled), flat(target), flat(current),
                    mask.reshape(n_groups * NINE, 256))["cosine"])
            entry["v3_correct_vs_shuffled"][split] = {
                "correct_changed_cosine": correct["cosine"],
                "shuffled_changed_cosine_mean": float(np.mean(shuffled_scores)),
                "correct_minus_shuffled": correct["cosine"] - float(np.mean(shuffled_scores)),
                "persistence_changed_cosine": R.metrics(
                    flat(current), flat(target), flat(current),
                    mask.reshape(n_groups * NINE, 256))["cosine"],
            }
        entry["wpe_selection_correct_minus_shuffled"] = (
            paired["arms"][arm]["prediction"]["correct_minus_shuffled"])
        report["arms"][arm] = entry
        del encoder, target_encoder, predictor, live_online, live_ema, predictions
        torch.cuda.empty_cache()

    out = RUN.OUT / "basis_drift_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
