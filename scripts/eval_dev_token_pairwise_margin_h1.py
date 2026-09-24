#!/usr/bin/env python3
"""Paired acceptance evaluation for the masked pairwise margin arms.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.

Checkpoints are selected by held-out masked pairwise ranking accuracy, never by
soft nCE.  The encoder-moving arm is accepted only if, relative to the frozen
control, it:

1. improves masked pairwise accuracy or MRR on V3 selection;
2. preserves or improves WP-E correct-minus-shuffled changed-token prediction;
3. preserves at least 95% of raw-token variance, effective rank and temporal delta;
4. does not regress fresh-probe occupied geometry;
5. remains non-regressive on the open-obstacle-field selection scene.
"""
from __future__ import annotations

import argparse
import json
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
from lewm.models.go2_masked_pairwise_margin_v1 import (  # noqa: E402
    masked_pairwise_margin_v1, separation_mask_v1,
)
from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402
import scripts.run_dev_token_counterfactual_matching_h1 as PRIOR  # noqa: E402
import scripts.run_dev_token_pairwise_margin_h1 as RUN  # noqa: E402
from scripts.eval_dev_token_counterfactual_matching_h1 import (  # noqa: E402
    LN, encode_raw, raw_health,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DERANGEMENT_SEEDS = (11, 23, 37)
PROBE_EPOCHS = 30
BATCH = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", default="best_masked_accuracy",
                    choices=("best_masked_accuracy", "final"))
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = R.load_pairs()
    fit = [r for r in rows if r["role"] == "train"]
    val = [r for r in rows if r["role"] == "checkpoint_selection"]

    reference = P.load_stack(device)[0]
    # The fit-set 75th-percentile threshold is the same quantity the matching
    # evaluation already computed: same reference checkpoint, same fit rows, same
    # preprocessing, same statistic.  Reuse it rather than re-encoding 8,524
    # frames, and record the provenance.
    prior = PRIOR.OUT / "paired_evaluation.json"
    threshold_source = "recomputed"
    threshold = None
    if prior.exists():
        stored = json.loads(prior.read_text()).get("changed_token_threshold", {})
        if stored.get("identical_across_arms") and isinstance(stored.get("value"), float):
            threshold = float(stored["value"])
            threshold_source = f"reused from {prior.name} (same reference encoder and fit rows)"
    if threshold is None:
        change = (LN(encode_raw([r["nxt"] for r in fit], reference, device))
                  - LN(encode_raw([r["cur"] for r in fit], reference, device))).pow(2).mean(-1)
        threshold = float(torch.quantile(change.flatten().float(), 0.75))
    mask_val = ((LN(encode_raw([r["nxt"] for r in val], reference, device))
                 - LN(encode_raw([r["cur"] for r in val], reference, device)))
                .pow(2).mean(-1) > threshold)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    sel_rows = [i for i, s in enumerate(order) if s in set(cache["selection_state_ids"])]
    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)
    endpoints = torch.stack([
        torch.tensor([b["endpoint_state"]["base_pos_world"][:2] for b in
                      sorted(records[order[i]]["branches"], key=lambda x: int(x["action_id"]))])
        for i in sel_rows]).to(device)
    sel_tbar = F.normalize(cache["successor_flat_unit"][sel_rows].to(device), dim=-1)
    sel_cos = PRIOR.group_cosine(sel_tbar, sel_tbar)
    sel_mask = separation_mask_v1(sel_cos, endpoints)
    sel_commands = cache["commands"][sel_rows].to(device)
    sel_images = torch.stack([V3.preprocess_v3_frame_v1(by_id[order[i]].current_path)
                              for i in sel_rows]).to(device)
    sel_scenes = [by_id[order[i]].scene_id for i in sel_rows]
    one_hot_all = torch.eye(V3.BRANCHES_PER_GROUP, device=device)

    y_fit = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in fit])).long()
    y_val = torch.from_numpy(np.stack([
        np.fromfile(Path(r["nxt_sha"]["shard_dir"]) / "raster_labels.u1", dtype=np.uint8)
        .reshape(-1, 64, 64)[int(r["nxt_sha"]["shard_row"])] for r in val])).long()

    report = {"status": STATUS, "claim_bearing": False,
              "changed_token_threshold_source": threshold_source,
              "checkpoint_rule": "held-out masked pairwise ranking accuracy (never soft nCE)",
              "checkpoint_used": args.which,
              "changed_token_threshold": threshold, "arms": {}}
    one_hot_w, command_w = R.action_tensors([r["primitive"] for r in val])

    for arm in ("frozen", "partial"):
        state = torch.load(RUN.OUT / f"arm_{arm}" / f"{args.which}.pt",
                           map_location="cpu", weights_only=False)
        encoder = P.load_stack(device)[0]
        encoder.load_state_dict(state["encoder"], strict=True)
        target = P.load_stack(device)[0]
        target.load_state_dict(state["target_encoder"], strict=True)
        predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
        predictor.load_state_dict(state["predictor"], strict=True)
        for module in (encoder, target, predictor):
            module.to(device).eval().requires_grad_(False)

        # ---- 1. masked ranking on V3 selection ----
        with torch.no_grad():
            st = LN(encoder.forward_tokens(sel_images)[:, 1:, :])
            st = st.repeat_interleave(V3.BRANCHES_PER_GROUP, dim=0)
            pred = LN(predictor(st, one_hot_all.repeat(len(sel_rows), 1),
                                sel_commands.reshape(-1, 3)))
        stats = masked_pairwise_margin_v1(
            pred.reshape(len(sel_rows), V3.BRANCHES_PER_GROUP, -1),
            sel_tbar, sel_mask, sel_cos)
        per_scene_v3 = {}
        for scene in sorted(set(sel_scenes)):
            rws = [i for i, s in enumerate(sel_scenes) if s == scene]
            s2 = masked_pairwise_margin_v1(
                pred.reshape(len(sel_rows), V3.BRANCHES_PER_GROUP, -1)[rws],
                sel_tbar[rws], sel_mask[rws], sel_cos[rws])
            per_scene_v3[scene] = {"pairwise_accuracy": s2.pairwise_accuracy,
                                   "masked_mrr": s2.masked_mrr,
                                   "valid_pairs": s2.valid_pairs}

        # ---- 2. WP-E correct-minus-shuffled ----
        cur_raw = encode_raw([r["cur"] for r in val], encoder, device)
        nxt_ema = encode_raw([r["nxt"] for r in val], target, device)
        nxt_online = encode_raw([r["nxt"] for r in val], encoder, device)
        current, target_state = LN(cur_raw), LN(nxt_ema)

        def predict(one_hot, command):
            out = []
            for i in range(0, len(current), 64):
                with torch.no_grad():
                    out.append(LN(predictor(current[i:i + 64].to(device),
                                            one_hot[i:i + 64].to(device),
                                            command[i:i + 64].to(device))).cpu())
            return torch.cat(out, 0)

        correct_pred = predict(one_hot_w, command_w)
        correct = R.metrics(correct_pred, target_state, current, mask_val)
        # Shuffled predictions computed once and reused for the overall margin and
        # every per-scene margin, so both use identical derangements.
        shuffled_pred = []
        for seed in DERANGEMENT_SEEDS:
            perm = torch.randperm(len(val), generator=torch.Generator().manual_seed(seed))
            shuffled_pred.append(predict(one_hot_w[perm], command_w[perm]))
        shuffled = [R.metrics(sp, target_state, current, mask_val)["cosine"]
                    for sp in shuffled_pred]
        margin = correct["cosine"] - float(np.mean(shuffled))
        scenes = [r["scene"] for r in val]
        per_scene_wpe = {}
        for scene in sorted(set(scenes)):
            ii = torch.tensor([i for i, s in enumerate(scenes) if s == scene])
            m = mask_val[ii]
            c = R.metrics(correct_pred[ii], target_state[ii], current[ii], m)["cosine"]
            d = [c - R.metrics(sp[ii], target_state[ii], current[ii], m)["cosine"]
                 for sp in shuffled_pred]
            per_scene_wpe[scene] = float(np.mean(d))

        # ---- 3. raw health ----
        health = raw_health(nxt_online)
        health["raw_temporal_delta"] = float((nxt_online - cur_raw).abs().mean())

        # ---- 4. fresh spatial probe ----
        nxt_fit_arm = encode_raw([r["nxt"] for r in fit], encoder, device)
        probe = R.SpatialProbe().to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
        counts = np.bincount(y_fit.numpy().reshape(-1), minlength=3).astype(np.float64)
        w = torch.tensor(counts.sum() / np.maximum(counts, 1.0),
                         dtype=torch.float32, device=device)
        w = w / w.mean()
        gp = torch.Generator().manual_seed(RUN.SEED + 1)
        probe.train()
        for _ in range(PROBE_EPOCHS):
            perm = torch.randperm(len(fit), generator=gp)
            for s in range(0, len(perm), BATCH):
                sel = perm[s:s + BATCH]
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(probe(nxt_fit_arm[sel].to(device)),
                                y_fit[sel].to(device), weight=w).backward()
                opt.step()
        probe.eval()

        @torch.no_grad()
        def probe_eval(tokens):
            preds = []
            for i in range(0, len(tokens), 64):
                preds.append(probe(tokens[i:i + 64].to(device)).argmax(1).cpu().numpy())
            pred_np = np.concatenate(preds, 0); truth = y_val.numpy()
            out = {}
            for k, name in ((1, "free"), (2, "occupied")):
                a, c = truth == k, pred_np == k
                union = int((a | c).sum())
                out[f"{name}_iou"] = int((a & c).sum()) / union if union else None
            obs = truth != 0
            out["observable_accuracy"] = float((pred_np[obs] == truth[obs]).mean())
            return out

        report["arms"][arm] = {
            "step": state["step"],
            "v3_masked_ranking": {"pairwise_accuracy": stats.pairwise_accuracy,
                                  "masked_mrr": stats.masked_mrr,
                                  "mean_achieved_margin": stats.mean_achieved_margin,
                                  "valid_pairs": stats.valid_pairs,
                                  "valid_anchors": stats.valid_anchors,
                                  "chance_pairwise_accuracy": 0.5},
            "v3_per_scene": per_scene_v3,
            "wpe_prediction": {"correct_changed_cosine": correct["cosine"],
                               "shuffled_changed_cosine_mean": float(np.mean(shuffled)),
                               "correct_minus_shuffled": margin},
            "wpe_per_scene_correct_minus_shuffled": per_scene_wpe,
            "raw_health": health,
            "spatial_fresh_probe": {"true_future_reference": probe_eval(nxt_ema),
                                    "persistence": probe_eval(cur_raw),
                                    "predicted": probe_eval(correct_pred)},
        }
        del encoder, target, predictor, nxt_fit_arm, probe
        torch.cuda.empty_cache()

    f, p = report["arms"]["frozen"], report["arms"]["partial"]
    oof_v3 = [k for k in f["v3_per_scene"] if "open_obstacle_field" in k]
    oof_wpe = [k for k in f["wpe_per_scene_correct_minus_shuffled"] if "open_obstacle_field" in k]
    gates = {
        "1_improves_masked_accuracy_or_mrr": bool(
            p["v3_masked_ranking"]["pairwise_accuracy"] > f["v3_masked_ranking"]["pairwise_accuracy"]
            or p["v3_masked_ranking"]["masked_mrr"] > f["v3_masked_ranking"]["masked_mrr"]),
        "2_preserves_wpe_correct_minus_shuffled": bool(
            p["wpe_prediction"]["correct_minus_shuffled"]
            >= f["wpe_prediction"]["correct_minus_shuffled"]),
        "3_preserves_95pct_raw_health": bool(all(
            p["raw_health"][k] >= 0.95 * f["raw_health"][k]
            for k in ("raw_token_variance", "effective_rank", "raw_temporal_delta"))),
        "4_no_fresh_probe_occupied_regression": bool(
            p["spatial_fresh_probe"]["predicted"]["occupied_iou"]
            >= f["spatial_fresh_probe"]["predicted"]["occupied_iou"]),
        "5_open_obstacle_field_non_regressive": bool(
            all(p["v3_per_scene"][k]["pairwise_accuracy"]
                >= f["v3_per_scene"][k]["pairwise_accuracy"] for k in oof_v3)
            and all(p["wpe_per_scene_correct_minus_shuffled"][k]
                    >= f["wpe_per_scene_correct_minus_shuffled"][k] for k in oof_wpe)),
    }
    gates["all_passed"] = all(gates.values())
    report["gates"] = gates
    out = RUN.OUT / f"paired_evaluation_{args.which}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(gates, indent=2))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
