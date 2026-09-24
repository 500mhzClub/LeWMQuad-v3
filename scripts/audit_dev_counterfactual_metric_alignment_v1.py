#!/usr/bin/env python3
"""Metric-alignment audit: calibration, branch ranking, or objective misalignment?

DEVELOPMENT ONLY.  NOT CLAIM BEARING.  No training, no checkpoint selection, no
collection, no objective change.

The learning curve showed selection soft-label nCE improving with scene count
yet never crossing uniform, and the checkpoint chosen by nCE was not the one
with the strongest correct-versus-shuffled margin.  Three candidate
explanations: probability calibration, branch ranking, or the matching
objective being misaligned with the thesis endpoint.

**Recoverability limits, stated rather than worked around.**  The stored curve
logged only ``matching_ce``, ``normalised_ce``, ``top1``,
``diagonal_probability``, ``own_vs_other_l2_ratio`` and
``correct_minus_shuffled`` at each of its 120 evaluation points.  Row entropy,
mean rank, MRR, rank correlation against Q and the separated-pair split were
never logged, and only six checkpoints were saved.  Those quantities are
therefore computed at the six stored checkpoints; recovering them at all 120
points would require retraining, which this audit does not do.

Separated-pair criteria are frozen here before any result is inspected:
    * endpoint displacement between the two branches > 0.05 m
    * frozen-successor cosine between the two branches < 0.90
"""
from __future__ import annotations

import argparse
import collections
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
from scripts.diagnose_dev_counterfactual_input_sufficiency_v1 import (  # noqa: E402
    InputAdapterV1, LN, normalised_ce,
)
from scripts.diagnose_dev_counterfactual_learning_curve_v1 import (  # noqa: E402
    scene_rank_within_family,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
NINE = V3.BRANCHES_PER_GROUP
ARM = "temporal_context"
CURVE = RUN.OUT / "learning_curve_diagnostic.json"
CKPT_DIR = RUN.OUT / "learning_curve"
DERANGEMENT_SEEDS = (11, 23, 37)

# Frozen before inspecting any result.
DISPLACEMENT_M = 0.05
FROZEN_COSINE = 0.90


def spearman(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ra = a.argsort(dim=dim).argsort(dim=dim).double()
    rb = b.argsort(dim=dim).argsort(dim=dim).double()
    ra = ra - ra.mean(dim=dim, keepdim=True)
    rb = rb - rb.mean(dim=dim, keepdim=True)
    num = (ra * rb).sum(dim=dim)
    den = ra.pow(2).sum(dim=dim).sqrt() * rb.pow(2).sum(dim=dim).sqrt()
    return num / den.clamp_min(1e-12)


def pearson(x, y) -> float:
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12))


def rank_metrics(logits: torch.Tensor) -> dict:
    """Own-successor top-1, mean rank (1 = best) and MRR."""
    n = logits.shape[-1]
    own = torch.diagonal(logits, dim1=1, dim2=2)
    rank = (logits > own[:, :, None]).sum(-1) + 1
    return {
        "top1": float((rank == 1).double().mean()),
        "mean_rank": float(rank.double().mean()),
        "mrr": float((1.0 / rank.double()).mean()),
        "chance_mean_rank": (n + 1) / 2,
        "chance_mrr": float(np.mean([1.0 / r for r in range(1, n + 1)])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    curve = json.loads(CURVE.read_text())
    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    train_ids = set(cache["train_state_ids"])
    selection_ids = set(cache["selection_state_ids"])
    idx = {"train": [i for i, s in enumerate(order) if s in train_ids],
           "selection": [i for i, s in enumerate(order) if s in selection_ids]}

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)
    encoder = P.load_stack(device)[0]
    encoder.eval().requires_grad_(False)

    panels = {}
    for split, rows in idx.items():
        groups = [by_id[order[i]] for i in rows]
        frames = torch.empty(len(groups), 3, 256, TOKEN_DIM)
        endpoints = torch.empty(len(groups), NINE, 2)
        for gi, group in enumerate(groups):
            record = records[group.state_id]
            batch = torch.stack([
                V3.preprocess_v3_frame_v1(ceiling.rgb_path_v1("train", a))
                for a in record["context"]["rgb_artifact_ids"]]).to(device)
            with torch.no_grad():
                frames[gi] = LN(encoder.forward_tokens(batch)[:, 1:, :]).cpu()
            branches = sorted(record["branches"], key=lambda b: int(b["action_id"]))
            endpoints[gi] = torch.tensor(
                [b["endpoint_state"]["base_pos_world"][:2] for b in branches])
        tbar_flat = F.normalize(cache["successor_flat_unit"][rows].to(device), dim=-1)
        q = torch.softmax(RUN.group_cosine(tbar_flat, tbar_flat) / RUN.TAU_T, dim=-1)
        panels[split] = {
            "groups": groups, "frames": frames.to(device),
            "proprio": torch.zeros(len(groups), 30, device=device),
            "commands": cache["commands"][rows].to(device),
            "tbar_tokens": cache["successor_normalised_tokens"][rows].to(device),
            "tbar_flat": tbar_flat, "q": q,
            "entropy": float(-(q * q.clamp_min(1e-12).log()).sum(-1).mean()),
            "frozen_cos": RUN.group_cosine(tbar_flat, tbar_flat),
            "endpoints": endpoints.to(device),
            "scenes": [g.scene_id for g in groups],
        }

    # ---- frozen separated-pair masks, plus their coverage ----
    coverage = {}
    for split, panel in panels.items():
        e = panel["endpoints"]
        displacement = (e[:, :, None, :] - e[:, None, :, :]).norm(dim=-1)
        off = ~torch.eye(NINE, dtype=torch.bool, device=device)
        by_displacement = (displacement > DISPLACEMENT_M) & off
        by_cosine = (panel["frozen_cos"] < FROZEN_COSINE) & off
        either = by_displacement | by_cosine
        both = by_displacement & by_cosine
        panel["separated"] = {"displacement": by_displacement, "cosine": by_cosine,
                              "both": both, "either": either}
        n_groups = e.shape[0]
        coverage[split] = {
            "groups": n_groups,
            "ordered_branch_comparisons": int(n_groups * NINE * (NINE - 1)),
            "unordered_pairs_total": int(n_groups * NINE * (NINE - 1) // 2),
            "pairs_displacement_gt_5cm": int(by_displacement.sum() // 2),
            "pairs_frozen_cosine_lt_0.90": int(by_cosine.sum() // 2),
            "pairs_both": int(both.sum() // 2),
            "pairs_either": int(either.sum() // 2),
            "groups_with_any_displacement_pair": int(
                by_displacement.any(-1).any(-1).sum()),
            "groups_with_any_cosine_pair": int(by_cosine.any(-1).any(-1).sum()),
            "rows_with_any_separated_partner": int(either.any(-1).sum()),
            "rows_total": int(n_groups * NINE),
        }

    report = {"status": STATUS, "claim_bearing": False, "arm": ARM,
              "frozen_separation_criteria": {
                  "endpoint_displacement_m": DISPLACEMENT_M,
                  "frozen_successor_cosine": FROZEN_COSINE,
                  "declared_before_inspecting_results": True},
              "separated_pair_coverage": coverage,
              "recoverability": {
                  "logged_at_all_120_evaluation_points": [
                      "matching_ce", "normalised_ce", "top1",
                      "diagonal_probability", "own_vs_other_l2_ratio",
                      "correct_minus_shuffled"],
                  "recoverable_only_at_6_stored_checkpoints": [
                      "predicted_row_entropy", "mean_rank", "mrr",
                      "row_spearman_vs_Q", "separated_pair_metrics",
                      "temperature_diagnostics"],
                  "reason": "the learning-curve runner logged a reduced metric set "
                            "and saved only best/final checkpoints; recovering the "
                            "rest at every point would require retraining"},
              "checkpoints": {}, "correlations": {}, "temperature": {}}

    one_hot_all = torch.eye(NINE, device=device)

    def logits_for(adapter, predictor, panel):
        n = panel["frames"].shape[0]
        with torch.no_grad():
            state = adapter(panel["frames"], panel["proprio"])
            expanded = state.repeat_interleave(NINE, dim=0)
            pred = LN(predictor(expanded, one_hot_all.repeat(n, 1),
                                panel["commands"].reshape(-1, 3)))
        groups_pred = F.normalize(pred.reshape(n, NINE, -1), dim=-1)
        return (torch.einsum("gid,gjd->gij", groups_pred, panel["tbar_flat"])
                / RUN.TAU_P), pred, expanded

    def full_metrics(logits, panel, pred, expanded, threshold):
        n = logits.shape[0]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        q = panel["q"]
        ce = float(-(q * log_probs).sum(-1).mean())
        out = {
            "matching_ce": ce,
            "normalised_ce": normalised_ce(ce, panel["entropy"]),
            "predicted_row_entropy": float(
                -(probs * log_probs).sum(-1).mean()),
            "target_q_row_entropy": panel["entropy"],
            "diagonal_probability": float(torch.diagonal(probs, dim1=1, dim2=2).mean()),
            **rank_metrics(logits),
            "row_spearman_vs_Q_all9": float(spearman(logits, q).mean()),
        }
        off = ~torch.eye(NINE, dtype=torch.bool, device=logits.device)
        so = spearman(logits.masked_select(off).reshape(n, NINE, NINE - 1),
                      q.masked_select(off).reshape(n, NINE, NINE - 1))
        out["row_spearman_vs_Q_offdiagonal"] = float(so.mean())

        # correct-minus-shuffled on token predictions (independent of logits)
        target = panel["tbar_tokens"].reshape(n * NINE, 256, TOKEN_DIM)
        mask = ((target - expanded).pow(2).mean(-1) > threshold)
        correct = R.metrics(pred, target, expanded, mask)
        shuffled = []
        for seed in DERANGEMENT_SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(NINE, generator=g).to(logits.device)
            sh = pred.reshape(n, NINE, 256, TOKEN_DIM)[:, perm].reshape(
                n * NINE, 256, TOKEN_DIM)
            shuffled.append(R.metrics(sh, target, expanded, mask)["cosine"])
        out["correct_minus_shuffled"] = correct["cosine"] - float(np.mean(shuffled))

        # restricted to well-separated candidate sets
        for name in ("displacement", "cosine", "both"):
            keep = panel["separated"][name].clone()
            eye = torch.eye(NINE, dtype=torch.bool, device=logits.device)
            keep = keep | eye[None]
            restricted = logits.masked_fill(~keep, float("-inf"))
            own = torch.diagonal(restricted, dim1=1, dim2=2)
            rank = (restricted > own[:, :, None]).sum(-1) + 1
            candidates = keep.sum(-1)
            valid = candidates > 1
            q_r = panel["q"].masked_fill(~keep, 0.0)
            q_r = q_r / q_r.sum(-1, keepdim=True).clamp_min(1e-12)
            lp = F.log_softmax(restricted, dim=-1)
            ce_r = -(q_r * lp.masked_fill(~keep, 0.0)).sum(-1)
            ent_r = -(q_r * q_r.clamp_min(1e-12).log()).sum(-1)
            uniform_r = candidates.double().clamp_min(1).log()
            span = (uniform_r - ent_r).clamp_min(1e-9)
            out[f"separated_{name}"] = {
                "rows_used": int(valid.sum()),
                "mean_candidates": float(candidates[valid].double().mean()),
                "top1": float((rank[valid] == 1).double().mean()),
                "mean_rank": float(rank[valid].double().mean()),
                "mrr": float((1.0 / rank[valid].double()).mean()),
                "normalised_ce": float(
                    ((ce_r[valid] - ent_r[valid]) / span[valid]).mean()),
            }
        return out

    # threshold reused from the learning curve
    threshold = curve["changed_token_threshold"]

    for size in ("8_scenes", "16_scenes", "24_scenes"):
        subset_scenes = set(curve["subsets"][size]["scenes"])
        rows = [i for i, sc in enumerate(panels["train"]["scenes"]) if sc in subset_scenes]
        train_view = {k: (panels["train"][k][rows] if torch.is_tensor(panels["train"][k])
                          else [panels["train"][k][i] for i in rows])
                      for k in ("frames", "proprio", "commands", "tbar_tokens",
                                "tbar_flat", "q", "endpoints", "frozen_cos", "scenes",
                                "groups")}
        train_view["entropy"] = float(-(train_view["q"]
                                        * train_view["q"].clamp_min(1e-12).log()).sum(-1).mean())
        train_view["separated"] = {k: v[rows] for k, v
                                   in panels["train"]["separated"].items()}
        for which in ("best_selection", "final"):
            path = CKPT_DIR / f"{which}_{size}.pt"
            state = torch.load(path, map_location=device, weights_only=False)
            adapter = InputAdapterV1(ARM, proprio_dim=30).to(device)
            adapter.load_state_dict(state["adapter"])
            predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
            predictor.load_state_dict(state["predictor"])
            adapter.eval(); predictor.eval()

            sel_logits, sel_pred, sel_exp = logits_for(adapter, predictor, panels["selection"])
            tr_logits, _tp, _te = logits_for(adapter, predictor, train_view)
            key = f"{size}/{which}"
            report["checkpoints"][key] = {
                "step": state["step"],
                "selection": full_metrics(sel_logits, panels["selection"],
                                          sel_pred, sel_exp, threshold),
            }

            # ---- post-hoc scalar temperature, no retraining ----
            grid = torch.tensor(np.geomspace(0.02, 50.0, 400), device=device)
            def ce_at(logits, panel, scale):
                lp = F.log_softmax(logits / scale, dim=-1)
                ce = float(-(panel["q"] * lp).sum(-1).mean())
                return normalised_ce(ce, panel["entropy"])
            sel_curve = [ce_at(sel_logits, panels["selection"], float(s)) for s in grid]
            tr_curve = [ce_at(tr_logits, train_view, float(s)) for s in grid]
            oracle_i = int(np.argmin(sel_curve))
            train_i = int(np.argmin(tr_curve))
            applied = float(grid[train_i])
            scaled = sel_logits / applied
            invariance = {
                "top1_unchanged": bool(
                    torch.equal(scaled.argmax(-1), sel_logits.argmax(-1))),
                "ranking_unchanged": bool(torch.equal(
                    scaled.argsort(-1), sel_logits.argsort(-1))),
                "correct_minus_shuffled_unchanged": True,  # computed on tokens, not logits
            }
            report["temperature"][key] = {
                "oracle_min_selection_nce": min(sel_curve),
                "oracle_scale": float(grid[oracle_i]),
                "oracle_label": "ORACLE DIAGNOSTIC ONLY -- fitted on selection, "
                                "not a selectable result",
                "train_fitted_scale": applied,
                "selection_nce_before": sel_curve[
                    int(np.argmin(np.abs(grid.cpu().numpy() - 1.0)))],
                "selection_nce_after_train_fitted": ce_at(
                    sel_logits, panels["selection"], applied),
                "effective_tau_p_after": RUN.TAU_P * applied,
                "invariance_checks": invariance,
                "diagonal_probability_after": float(torch.diagonal(
                    F.softmax(scaled, dim=-1), dim1=1, dim2=2).mean()),
                "predicted_row_entropy_after": float(
                    -(F.softmax(scaled, -1) * F.log_softmax(scaled, -1)).sum(-1).mean()),
            }
            del adapter, predictor
            torch.cuda.empty_cache()

    # ---- correlations across the 120 stored evaluation points ----
    pooled = collections.defaultdict(list)
    per_arm = {}
    for size, sub in curve["subsets"].items():
        c = sub["curve"]
        nce = [p["selection_normalised_ce"] for p in c]
        cms = [p["selection_correct_minus_shuffled"] for p in c]
        top1 = [p["selection_top1"] for p in c]
        per_arm[size] = {
            "points": len(c),
            "pearson_nce_vs_correct_minus_shuffled": round(pearson(nce, cms), 4),
            "spearman_nce_vs_correct_minus_shuffled": round(
                float(spearman(torch.tensor(nce), torch.tensor(cms), dim=0)), 4),
            "pearson_nce_vs_top1": round(pearson(nce, top1), 4),
            "spearman_nce_vs_top1": round(
                float(spearman(torch.tensor(nce), torch.tensor(top1), dim=0)), 4),
        }
        pooled["nce"] += nce; pooled["cms"] += cms; pooled["top1"] += top1
    report["correlations"]["per_arm_120_evaluation_points"] = per_arm
    report["correlations"]["pooled"] = {
        "points": len(pooled["nce"]),
        "pearson_nce_vs_correct_minus_shuffled": round(
            pearson(pooled["nce"], pooled["cms"]), 4),
        "spearman_nce_vs_correct_minus_shuffled": round(
            float(spearman(torch.tensor(pooled["nce"]),
                           torch.tensor(pooled["cms"]), dim=0)), 4),
        "pearson_nce_vs_top1": round(pearson(pooled["nce"], pooled["top1"]), 4),
    }
    # MRR / rank-correlation correlations available only at the 6 checkpoints
    keys = list(report["checkpoints"])
    report["correlations"]["six_checkpoints_only"] = {
        "n": len(keys),
        "caveat": "n=6; MRR and rank correlation were not logged during training",
        "pearson_nce_vs_mrr": round(pearson(
            [report["checkpoints"][k]["selection"]["normalised_ce"] for k in keys],
            [report["checkpoints"][k]["selection"]["mrr"] for k in keys]), 4),
        "pearson_spearmanQ_vs_correct_minus_shuffled": round(pearson(
            [report["checkpoints"][k]["selection"]["row_spearman_vs_Q_offdiagonal"]
             for k in keys],
            [report["checkpoints"][k]["selection"]["correct_minus_shuffled"]
             for k in keys]), 4),
        "pearson_nce_vs_correct_minus_shuffled": round(pearson(
            [report["checkpoints"][k]["selection"]["normalised_ce"] for k in keys],
            [report["checkpoints"][k]["selection"]["correct_minus_shuffled"]
             for k in keys]), 4),
    }

    out = RUN.OUT / "metric_alignment_audit.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str)[:400])
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
