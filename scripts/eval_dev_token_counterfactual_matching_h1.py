#!/usr/bin/env python3
"""Paired evaluation of the counterfactual-matching arms.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.

Five registered criteria, all evaluated on held-out roles:

1. correct-versus-shuffled changed-token cosine, WP-E ``checkpoint_selection``;
2. whether the **live online** successor-similarity matrix still matches frozen
   ``Q``, on the 32 held-out V3 selection groups;
3. raw-token variance, effective rank and temporal change;
4. fresh-probe spatial geometry (a probe retrained on each arm's own encoder);
5. positivity on the ``open_obstacle_field`` selection scene.

Criterion 2 carries the decisive distinction.  ``S`` is scored against frozen
reference columns, so a predictor can lower ``L_match`` without the encoder
representing anything new.  If predictor-to-reference matching improves while
the live successor relationships do not, the result is a predictor-side
solution, not an action-discriminative JEPA representation.

Note recorded rather than glossed: in the frozen arm the online encoder never
moves, so its live successor matrix **equals** the frozen reference by
construction and criterion 2 is degenerate against it.  The meaningful reading
for the partial arm is absolute preservation, not a contest it cannot win.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
import scripts.run_dev_token_counterfactual_matching_h1 as RUN  # noqa: E402

OUT = RUN.OUT
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
SEED = RUN.SEED
DERANGEMENT_SEEDS = (11, 23, 37)
PROBE_EPOCHS = 30
BATCH = 16


def LN(tokens):
    return F.layer_norm(tokens, (TOKEN_DIM,))


def raw_health(tokens: torch.Tensor) -> dict:
    """WP-E convention: flatten to (N*256,192), centre over all rows."""
    flat = tokens.reshape(-1, TOKEN_DIM).double()
    flat = flat - flat.mean(0)
    covariance = flat.T.mm(flat) / (flat.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (covariance + covariance.T)).clamp_min(0.0)
    p = eigenvalues / eigenvalues.sum()
    return {
        "raw_token_variance": float(flat.square().mean()),
        "effective_rank": float((-(p * p.clamp_min(1e-12).log()).sum()).exp()),
    }


def encode_raw(paths, encoder, device, preprocess=None, bs=32):
    preprocess = preprocess or P.native_preprocess
    out = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([preprocess(p) for p in paths[i:i + bs]])
        with torch.no_grad():
            out.append(encoder.forward_tokens(batch.to(device))[:, 1:, :].cpu())
    return torch.cat(out, 0)


def load_arm(arm, device):
    encoder, decoder, head = P.load_stack(device)
    predictor = initialize_token_predictor_v1(SEED).to(device)
    state = torch.load(OUT / f"arm_{arm}" / "joint_checkpoint.pt",
                       map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["encoder"], strict=True)
    predictor.load_state_dict(state["predictor"], strict=True)
    target = P.load_stack(device)[0]
    target.load_state_dict(state["target_encoder"], strict=True)
    for module in (encoder, predictor, target):
        module.to(device).eval().requires_grad_(False)
    return encoder, predictor, target, decoder, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]

    # ---- frozen changed-token mask, identical for every arm ----
    reference_encoder = P.load_stack(device)[0]
    ref_cur_fit = encode_raw([r["cur"] for r in fit], reference_encoder, device)
    ref_nxt_fit = encode_raw([r["nxt"] for r in fit], reference_encoder, device)
    ref_cur_val = encode_raw([r["cur"] for r in val], reference_encoder, device)
    ref_nxt_val = encode_raw([r["nxt"] for r in val], reference_encoder, device)
    change_fit = (LN(ref_nxt_fit) - LN(ref_cur_fit)).pow(2).mean(-1)
    threshold = float(torch.quantile(change_fit.flatten().float(), 0.75))
    mask_val = ((LN(ref_nxt_val) - LN(ref_cur_val)).pow(2).mean(-1) > threshold)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    selection_ids = set(cache["selection_state_ids"])
    order = cache["state_ids"]
    keep = [i for i, s in enumerate(order) if s in selection_ids]
    tbar = cache["successor_flat_unit"][keep].to(device)          # (32,9,D')
    v3_commands = cache["commands"][keep].to(device)
    with torch.no_grad():
        Q_ref = torch.softmax(RUN.group_cosine(tbar, tbar) / RUN.TAU_T, dim=-1)
        C_ref = RUN.group_cosine(tbar, tbar)

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    sel_groups = [by_id[order[i]] for i in keep]

    one_hot_w, command_w = R.action_tensors([r["primitive"] for r in val])
    one_hot_all = torch.eye(V3.BRANCHES_PER_GROUP, device=device)
    y_fit = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in fit])).long()
    y_val = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in val])).long()

    report = {"status": STATUS, "claim_bearing": False,
              "changed_token_threshold": {"value": threshold,
                                          "source": "frozen reference encoder, fit 75th pct",
                                          "identical_across_arms": True},
              "wpe_selection_pairs": len(val),
              "v3_selection_groups": len(sel_groups),
              "arms": {}}

    for arm in ("frozen", "partial"):
        encoder, predictor, target, _dec, _head = load_arm(arm, device)
        cur_raw = encode_raw([r["cur"] for r in val], encoder, device)
        nxt_raw_ema = encode_raw([r["nxt"] for r in val], target, device)
        nxt_raw_online = encode_raw([r["nxt"] for r in val], encoder, device)
        current, target_state = LN(cur_raw), LN(nxt_raw_ema)

        def predict(one_hot, command):
            out = []
            for i in range(0, len(current), 64):
                with torch.no_grad():
                    out.append(LN(predictor(current[i:i + 64].to(device),
                                            one_hot[i:i + 64].to(device),
                                            command[i:i + 64].to(device))).cpu())
            return torch.cat(out, 0)

        arms_pred = {"correct_action": predict(one_hot_w, command_w)}
        generator_states = []
        for k, seed in enumerate(DERANGEMENT_SEEDS):
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(len(val), generator=g)
            generator_states.append(int((perm == torch.arange(len(val))).sum()))
            arms_pred[f"shuffled_action_{k}"] = predict(one_hot_w[perm], command_w[perm])

        correct = R.metrics(arms_pred["correct_action"], target_state, current, mask_val)
        shuffled = [R.metrics(arms_pred[f"shuffled_action_{k}"], target_state, current,
                              mask_val) for k in range(len(DERANGEMENT_SEEDS))]
        persistence = R.metrics(current, target_state, current, mask_val)
        margin = correct["cosine"] - float(np.mean([s["cosine"] for s in shuffled]))

        # ---- per-scene ----
        scenes = [r["scene"] for r in val]
        per_scene = {}
        for scene in sorted(set(scenes)):
            idx = torch.tensor([i for i, s in enumerate(scenes) if s == scene])
            m = mask_val[idx]
            c = R.metrics(arms_pred["correct_action"][idx], target_state[idx],
                          current[idx], m)["cosine"]
            d = [c - R.metrics(arms_pred[f"shuffled_action_{k}"][idx], target_state[idx],
                               current[idx], m)["cosine"]
                 for k in range(len(DERANGEMENT_SEEDS))]
            per_scene[scene] = {"pairs": int(len(idx)),
                                "correct_minus_shuffled": float(np.mean(d))}

        # ---- raw health ----
        health = raw_health(nxt_raw_online)
        health["raw_temporal_delta"] = float((nxt_raw_online - cur_raw).abs().mean())

        # ---- fresh spatial probe, retrained on THIS arm's encoder ----
        nxt_fit_arm = encode_raw([r["nxt"] for r in fit], encoder, device)
        probe = R.SpatialProbe().to(device)
        optimiser = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        counts = np.bincount(y_fit.numpy().reshape(-1), minlength=3).astype(np.float64)
        weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0),
                              dtype=torch.float32, device=device)
        weight = weight / weight.mean()
        gp = torch.Generator().manual_seed(SEED + 1)
        probe.train()
        for _ in range(PROBE_EPOCHS):
            perm = torch.randperm(len(fit), generator=gp)
            for s in range(0, len(perm), BATCH):
                sel = perm[s:s + BATCH]
                optimiser.zero_grad(set_to_none=True)
                F.cross_entropy(probe(nxt_fit_arm[sel].to(device)),
                                y_fit[sel].to(device), weight=weight).backward()
                optimiser.step()
        probe.eval()

        @torch.no_grad()
        def probe_eval(tokens):
            preds = []
            for i in range(0, len(tokens), 64):
                preds.append(probe(tokens[i:i + 64].to(device)).argmax(1).cpu().numpy())
            pred = np.concatenate(preds, 0)
            truth = y_val.numpy()
            out = {}
            for k, name in ((1, "free"), (2, "occupied")):
                a, c = truth == k, pred == k
                union = int((a | c).sum())
                out[f"{name}_iou"] = int((a & c).sum()) / union if union else None
            obs = truth != 0
            out["observable_accuracy"] = float((pred[obs] == truth[obs]).mean())
            return out

        # Probe consumes RAW tokens, so map the normalised prediction back by the
        # same statistics the encoder produced -- evaluate on the raw scale.
        spatial = {"fresh_probe_trained_on": f"arm_{arm}_encoder_true_future_tokens",
                   "true_future_reference": probe_eval(nxt_raw_ema),
                   "persistence": probe_eval(cur_raw),
                   "predicted": probe_eval(arms_pred["correct_action"])}

        # ---- criterion 2: live successor relationships vs frozen Q ----
        live = []
        for group in sel_groups:
            batch = torch.stack([V3.preprocess_v3_frame_v1(p)
                                 for p in group.successor_paths]).to(device)
            with torch.no_grad():
                live.append(LN(encoder.forward_tokens(batch)[:, 1:, :]).reshape(9, -1).cpu())
        live = F.normalize(torch.stack(live), dim=-1).to(device)
        C_live = RUN.group_cosine(live, live)
        Q_live = torch.softmax(C_live / RUN.TAU_T, dim=-1)
        off = ~torch.eye(9, dtype=torch.bool, device=device)
        a = C_ref[:, off].flatten().double()
        b = C_live[:, off].flatten().double()
        pearson = float(((a - a.mean()) * (b - b.mean())).mean()
                        / (a.std(unbiased=False) * b.std(unbiased=False)))
        rank = lambda v: v.argsort().argsort().double()  # noqa: E731
        ra, rb = rank(a), rank(b)
        spearman = float(((ra - ra.mean()) * (rb - rb.mean())).mean()
                         / (ra.std(unbiased=False) * rb.std(unbiased=False)))
        kl = float((Q_ref * (Q_ref.clamp_min(1e-12).log()
                             - Q_live.clamp_min(1e-12).log())).sum(-1).mean())

        # predictor-to-reference matching on the same held-out groups
        pred_groups = []
        for gi, group in enumerate(sel_groups):
            image = V3.preprocess_v3_frame_v1(group.current_path)[None].to(device)
            with torch.no_grad():
                state = LN(encoder.forward_tokens(image)[:, 1:, :]).repeat(9, 1, 1)
                p = LN(predictor(state, one_hot_all, v3_commands[gi]))
            pred_groups.append(p.reshape(9, -1).cpu())
        pred_groups = F.normalize(torch.stack(pred_groups), dim=-1).to(device)
        S = torch.einsum("gid,gjd->gij", pred_groups, tbar) / RUN.TAU_P
        match_ce = float(-(Q_ref * F.log_softmax(S, dim=-1)).sum(-1).mean())
        top1 = float((S.argmax(-1) == torch.arange(9, device=device)[None]).float().mean())

        report["arms"][arm] = {
            "prediction": {
                "correct_changed_cosine": correct["cosine"],
                "shuffled_changed_cosine_mean": float(np.mean([s["cosine"] for s in shuffled])),
                "persistence_changed_cosine": persistence["cosine"],
                "correct_minus_shuffled": margin,
                "correct_normalised_error_vs_persistence":
                    correct["normalised_error_vs_persistence"],
                "derangement_fixed_points": generator_states,
            },
            "raw_health": health,
            "spatial_fresh_probe": spatial,
            "counterfactual_structure": {
                "live_vs_frozen_pearson_offdiag": round(pearson, 6),
                "live_vs_frozen_spearman_offdiag": round(spearman, 6),
                "mean_abs_delta_cosine": float((C_ref - C_live)[:, off].abs().mean()),
                "kl_Qref_to_Qlive": round(kl, 6),
                "predictor_to_reference_match_ce": round(match_ce, 6),
                "predictor_to_reference_top1": round(top1, 6),
            },
            "per_scene_correct_minus_shuffled": per_scene,
            "open_obstacle_field_positive": bool(
                min(v["correct_minus_shuffled"] for k, v in per_scene.items()
                    if "open_obstacle_field" in k) > 0),
        }
        del encoder, predictor, target, nxt_fit_arm, probe
        torch.cuda.empty_cache()

    frozen, partial = report["arms"]["frozen"], report["arms"]["partial"]
    gates = {
        "1_improves_correct_vs_shuffled":
            partial["prediction"]["correct_minus_shuffled"]
            > frozen["prediction"]["correct_minus_shuffled"],
        "2_preserves_frozen_counterfactual_relationships":
            partial["counterfactual_structure"]["live_vs_frozen_pearson_offdiag"]
            >= frozen["counterfactual_structure"]["live_vs_frozen_pearson_offdiag"],
        "3_preserves_raw_health": all([
            partial["raw_health"]["raw_token_variance"]
            >= 0.95 * frozen["raw_health"]["raw_token_variance"],
            partial["raw_health"]["effective_rank"]
            >= 0.95 * frozen["raw_health"]["effective_rank"],
            partial["raw_health"]["raw_temporal_delta"]
            >= 0.95 * frozen["raw_health"]["raw_temporal_delta"],
        ]),
        "4_no_fresh_probe_spatial_regression":
            partial["spatial_fresh_probe"]["predicted"]["occupied_iou"]
            >= frozen["spatial_fresh_probe"]["predicted"]["occupied_iou"],
        "5_open_obstacle_field_positive": partial["open_obstacle_field_positive"],
    }
    gates["all_passed"] = all(gates.values())
    gates["predictor_side_only"] = bool(
        partial["counterfactual_structure"]["predictor_to_reference_match_ce"]
        < frozen["counterfactual_structure"]["predictor_to_reference_match_ce"]
        and not gates["2_preserves_frozen_counterfactual_relationships"]
    )
    report["gates"] = gates
    report["note_criterion_2_degenerate_for_frozen_arm"] = (
        "the frozen arm's online encoder never moves, so its live successor "
        "matrix equals the frozen reference by construction"
    )
    (OUT / "paired_evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"gates": gates,
                      "frozen": {k: frozen[k] for k in
                                 ("prediction", "raw_health", "counterfactual_structure")},
                      "partial": {k: partial[k] for k in
                                  ("prediction", "raw_health", "counterfactual_structure")}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
