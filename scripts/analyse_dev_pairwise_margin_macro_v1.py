#!/usr/bin/env python3
"""Group- and scene-level masked-ranking analysis for the pairwise margin arms.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.  No training, no checkpoint selection.

The 1,092 ordered selection pairs are **not** independent observations: they are
nine-way comparisons repeated within 32 groups nested in 8 scenes, and groups
with more separated branches contribute more pairs.  Micro accuracy therefore
weights groups unequally.  This recomputes the ranking metrics as macro averages
over groups and over scenes, and reports the paired per-scene
partial-minus-frozen differences, so a small micro difference can be checked for
breadth rather than being carried by weighting.
"""
from __future__ import annotations

import argparse
import collections
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
import scripts.run_dev_token_counterfactual_matching_h1 as PRIOR  # noqa: E402
import scripts.run_dev_token_pairwise_margin_h1 as RUN  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
NINE = V3.BRANCHES_PER_GROUP


def LN(t):
    return F.layer_norm(t, (TOKEN_DIM,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    sel = [i for i, s in enumerate(order) if s in set(cache["selection_state_ids"])]

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)
    endpoints = torch.stack([
        torch.tensor([b["endpoint_state"]["base_pos_world"][:2] for b in
                      sorted(records[order[i]]["branches"], key=lambda x: int(x["action_id"]))])
        for i in sel]).to(device)
    tbar = F.normalize(cache["successor_flat_unit"][sel].to(device), dim=-1)
    frozen_cos = PRIOR.group_cosine(tbar, tbar)
    mask = separation_mask_v1(frozen_cos, endpoints)
    commands = cache["commands"][sel].to(device)
    images = torch.stack([V3.preprocess_v3_frame_v1(by_id[order[i]].current_path)
                          for i in sel]).to(device)
    scenes = [by_id[order[i]].scene_id for i in sel]
    one_hot = torch.eye(NINE, device=device)

    coverage = {
        "per_group_valid_pairs": {order[i]: int(mask[k].sum())
                                  for k, i in enumerate(sel)},
        "per_scene_valid_pairs": dict(collections.Counter()),
    }
    scene_pairs = collections.defaultdict(int)
    scene_groups = collections.defaultdict(int)
    for k, s in enumerate(scenes):
        scene_pairs[s] += int(mask[k].sum())
        scene_groups[s] += 1
    coverage["per_scene_valid_pairs"] = dict(scene_pairs)
    coverage["per_scene_groups"] = dict(scene_groups)
    coverage["total_ordered_pairs"] = int(mask.sum())
    coverage["groups"] = len(sel)
    coverage["scenes"] = len(set(scenes))

    report = {"status": STATUS, "claim_bearing": False,
              "independence_note": "1,092 ordered pairs are nested within 32 groups "
                                   "within 8 scenes; micro accuracy weights groups by "
                                   "their separated-pair count",
              "coverage": coverage, "checkpoints": {}}

    def per_unit(pred_groups):
        """Micro, per-group and per-scene accuracy / MRR."""
        overall = masked_pairwise_margin_v1(pred_groups, tbar, mask, frozen_cos)
        groups = {}
        for k in range(len(sel)):
            s = masked_pairwise_margin_v1(pred_groups[k:k + 1], tbar[k:k + 1],
                                          mask[k:k + 1], frozen_cos[k:k + 1])
            groups[order[sel[k]]] = {"accuracy": s.pairwise_accuracy,
                                     "mrr": s.masked_mrr, "pairs": s.valid_pairs,
                                     "scene": scenes[k]}
        scene_rows = collections.defaultdict(list)
        for k in range(len(sel)):
            scene_rows[scenes[k]].append(k)
        scene_stats = {}
        for scene, rws in scene_rows.items():
            s = masked_pairwise_margin_v1(pred_groups[rws], tbar[rws], mask[rws],
                                          frozen_cos[rws])
            scene_stats[scene] = {"micro_accuracy": s.pairwise_accuracy,
                                  "micro_mrr": s.masked_mrr, "pairs": s.valid_pairs,
                                  "macro_over_groups_accuracy": float(np.mean(
                                      [groups[order[sel[k]]]["accuracy"] for k in rws])),
                                  "macro_over_groups_mrr": float(np.mean(
                                      [groups[order[sel[k]]]["mrr"] for k in rws]))}
        return {
            "micro_accuracy": overall.pairwise_accuracy,
            "micro_mrr": overall.masked_mrr,
            "mean_achieved_margin": overall.mean_achieved_margin,
            "macro_over_32_groups_accuracy": float(np.mean(
                [v["accuracy"] for v in groups.values()])),
            "macro_over_32_groups_mrr": float(np.mean(
                [v["mrr"] for v in groups.values()])),
            "macro_over_8_scenes_accuracy": float(np.mean(
                [v["micro_accuracy"] for v in scene_stats.values()])),
            "macro_over_8_scenes_mrr": float(np.mean(
                [v["micro_mrr"] for v in scene_stats.values()])),
            "per_group": groups, "per_scene": scene_stats,
        }

    for arm in ("frozen", "partial"):
        for which in ("best_masked_accuracy", "final"):
            state = torch.load(RUN.OUT / f"arm_{arm}" / f"{which}.pt",
                               map_location="cpu", weights_only=False)
            encoder = P.load_stack(device)[0]
            encoder.load_state_dict(state["encoder"], strict=True)
            predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
            predictor.load_state_dict(state["predictor"], strict=True)
            encoder.eval().requires_grad_(False)
            predictor.eval().requires_grad_(False)
            with torch.no_grad():
                st = LN(encoder.forward_tokens(images)[:, 1:, :])
                st = st.repeat_interleave(NINE, dim=0)
                pred = LN(predictor(st, one_hot.repeat(len(sel), 1),
                                    commands.reshape(-1, 3)))
            report["checkpoints"][f"{arm}/{which}"] = {
                "step": state["step"], **per_unit(pred.reshape(len(sel), NINE, -1))}
            del encoder, predictor
            torch.cuda.empty_cache()

    # paired per-scene differences, at matched step 10680 and at selected checkpoints
    def paired(a_key, b_key, label):
        a = report["checkpoints"][a_key]; b = report["checkpoints"][b_key]
        rows = {}
        for scene in sorted(a["per_scene"]):
            rows[scene] = {
                "frozen_accuracy": a["per_scene"][scene]["micro_accuracy"],
                "partial_accuracy": b["per_scene"][scene]["micro_accuracy"],
                "delta_accuracy": (b["per_scene"][scene]["micro_accuracy"]
                                   - a["per_scene"][scene]["micro_accuracy"]),
                "frozen_mrr": a["per_scene"][scene]["micro_mrr"],
                "partial_mrr": b["per_scene"][scene]["micro_mrr"],
                "delta_mrr": (b["per_scene"][scene]["micro_mrr"]
                              - a["per_scene"][scene]["micro_mrr"]),
                "pairs": a["per_scene"][scene]["pairs"],
            }
        deltas = [v["delta_accuracy"] for v in rows.values()]
        dm = [v["delta_mrr"] for v in rows.values()]
        group_delta = [b["per_group"][g]["accuracy"] - a["per_group"][g]["accuracy"]
                       for g in a["per_group"]]
        return {"label": label, "per_scene": rows,
                "scenes_partial_ahead_accuracy": int(sum(1 for d in deltas if d > 0)),
                "scenes_tied_accuracy": int(sum(1 for d in deltas if d == 0)),
                "scenes_partial_ahead_mrr": int(sum(1 for d in dm if d > 0)),
                "mean_scene_delta_accuracy": float(np.mean(deltas)),
                "mean_scene_delta_mrr": float(np.mean(dm)),
                "groups_partial_ahead_accuracy": int(sum(1 for d in group_delta if d > 0)),
                "groups_tied_accuracy": int(sum(1 for d in group_delta if d == 0)),
                "groups_partial_behind_accuracy": int(sum(1 for d in group_delta if d < 0)),
                "mean_group_delta_accuracy": float(np.mean(group_delta))}

    report["paired"] = {
        "matched_step_10680_final_vs_final": paired(
            "frozen/final", "partial/final", "matched step 10680 (both finals)"),
        "selected_checkpoints_unmatched_steps": paired(
            "frozen/best_masked_accuracy", "partial/best_masked_accuracy",
            "frozen@5073 vs partial@10413 -- DIFFERENT steps, not matched"),
    }
    out = RUN.OUT / "macro_ranking_analysis.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
