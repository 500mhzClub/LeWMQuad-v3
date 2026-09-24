#!/usr/bin/env python3
"""Paired, family-stratified episode-cluster bootstrap for the frozen H = 1-4 result.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only over the frozen H=1-4 artifacts and
the cached per-row predictions; no model is run, trained or selected.

Resampling unit
---------------
The **episode cluster** ``(scene, env_index, episode_id, reset_count)``.  Rows
from one episode share overlapping frames and are not independent, so row-level
resampling would overstate the sample size.  Maze-seed level was considered and
rejected as the resampling unit: the selection split holds only eight scenes,
too few clusters to bootstrap -- episode level is the finer of the two units the
authorisation permits.

Scope of the intervals
----------------------
Clusters are drawn with replacement **within** each family, so every resample
preserves the family composition of the observed corpus.  The intervals
therefore quantify **variation across episodes within the present eight
families**.  They are NOT intervals for generalisation across independently
sampled maze populations: the eight families and the eight scenes are fixed, not
resampled, and no interval here speaks to a ninth family or an unseen maze
population.

Two weightings, reported side by side
-------------------------------------
* **corpus-weighted** (primary) -- masked cosines pooled over all tokens in the
  sampled clusters.  This is the estimator the frozen FINAL result uses, so its
  observed value reproduces the frozen point estimates exactly; the bootstrap
  only adds uncertainty around them.
* **equal-family-weighted** (robustness) -- the mean of the eight family-level
  pooled scores, so a family contributing more rows does not dominate.  This is
  reported as a separate robustness analysis and does **not** replace the frozen
  point estimates.

Reported separately at every horizon: correct-future score, shuffled-sequence
score, and the correct-minus-shuffled margin.  The shuffled-sequence assay is an
action-conditioning and discrimination diagnostic only; it is not candidate
ranking and not planning regret.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

import numpy as np

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
HOR = CACHE / "horizons"
FINAL = HOR / "FINAL"
SEED = 2_026_080_901
RESAMPLES = 10_000
MAX_H = 4
ALPHA = 0.05


def _ci(arr: np.ndarray, point: float) -> dict:
    arr = arr[~np.isnan(arr)]
    lo, hi = np.percentile(arr, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"point": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "draws": int(arr.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=RESAMPLES)
    ap.add_argument("--rows", default=str(FINAL / "FINAL_horizon_rows_479.jsonl"))
    ap.add_argument("--result", default=str(FINAL / "FINAL_horizon_result.json"))
    ap.add_argument("--pred-dir", default=str(HOR / "predictions"))
    ap.add_argument("--out", default=str(HOR / "bootstrap" / "result.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["max_horizon"] >= MAX_H]
    result = json.loads(Path(args.result).read_text())
    names = list(result["models"].keys())
    if len(names) != 2:
        raise SystemExit("expected exactly two models")
    a, b = names

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_dev_v03_two_step_rollout_v1 as R
    from scripts import run_dev_v03_temporal_action_jepa_v1 as T

    pred_dir = Path(args.pred_dir)
    missing = [p for p in (pred_dir / f"{n}_h{h}_{k}.f16"
                           for n in names for h in range(1, MAX_H + 1)
                           for k in ("correct", "shuffled")) if not p.is_file()]
    if missing:
        raise SystemExit(
            "per-row predictions were not cached by the H=1-4 evaluator; rerun it with "
            f"--cache-predictions (missing {len(missing)}, e.g. {missing[0].name})")

    n = len(rows)
    targets = {h: T.normalise(torch.from_numpy(np.ascontiguousarray(
        np.memmap(HOR / f"target_h{h}.f16", dtype=np.float16, mode="r",
                  shape=(n, R.TOKENS, R.DIM)))).float()) for h in range(1, MAX_H + 1)}

    # Reproduce the evaluator's frozen mask policy exactly: H=1 uses the frozen
    # step-1 threshold, H>=2 the frozen step-2 threshold.  Nothing is refitted.
    matched = json.loads((CACHE / "two_step" / "evaluation"
                          / "MATCHED_24_EPOCH_result_epochs_0_23.json").read_text())
    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    base_sel = [r for r in base if r["role"] == "checkpoint_selection"]
    pos = {r["pair_sha256"]: i for i, r in enumerate(base_sel)}
    idx = np.array([pos[r["pair_sha256"]] for r in rows])
    now = T.normalise(R.load_cache(
        CACHE / "temporal_action_jepa_v1" / "evaluation" / "frozen_current.f16",
        len(base_train) + len(base_sel))[len(base_train):][idx].float())
    masks, tok_counts = {}, {}
    for h in range(1, MAX_H + 1):
        thr = matched["masks"]["step1_threshold"] if h == 1 else matched["masks"]["step2_threshold"]
        masks[h] = (targets[h] - now).pow(2).mean(-1) >= thr
        tok_counts[h] = masks[h].sum(-1).numpy().astype(np.float64)   # per row

    def per_row_sum(name, h, kind) -> np.ndarray:
        """Sum of masked cosines per row (pairs with tok_counts[h] to give means)."""
        p = torch.from_numpy(np.ascontiguousarray(
            np.memmap(pred_dir / f"{name}_h{h}_{kind}.f16", dtype=np.float16, mode="r",
                      shape=(n, R.TOKENS, R.DIM)))).float()
        cos = F.cosine_similarity(p, targets[h], dim=-1)
        return (cos * masks[h]).sum(-1).numpy().astype(np.float64)

    sums = {(name, h, kind): per_row_sum(name, h, kind)
            for name in names for h in range(1, MAX_H + 1)
            for kind in ("correct", "shuffled")}

    # ---- episode clusters -------------------------------------------------
    clusters = [(r["scene"], r["env_index"], r["episode_id"], r["reset_count"]) for r in rows]
    families = [r["family"] for r in rows]
    cluster_rows = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_rows[c].append(i)
    by_family = collections.defaultdict(list)
    for c, members in cluster_rows.items():
        by_family[families[members[0]]].append(c)
    fam_names = sorted(by_family)
    fam_clusters = {f: sorted(by_family[f]) for f in fam_names}

    # cluster-level aggregates (sum of masked cosines, count of masked tokens)
    CS, CC = {}, {}
    for h in range(1, MAX_H + 1):
        for f in fam_names:
            CC[(h, f)] = np.array([tok_counts[h][cluster_rows[c]].sum()
                                   for c in fam_clusters[f]], dtype=np.float64)
        for name in names:
            for kind in ("correct", "shuffled"):
                for f in fam_names:
                    CS[(name, h, kind, f)] = np.array(
                        [sums[(name, h, kind)][cluster_rows[c]].sum()
                         for c in fam_clusters[f]], dtype=np.float64)

    rng = np.random.default_rng(SEED)
    B = args.resamples

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "read_only": True, "used_for_checkpoint_selection": False,
        "resampling_unit": "episode cluster (scene, env_index, episode_id, reset_count)",
        "why_not_row_level": ("rows from one episode share overlapping frames and are not "
                              "independent; row resampling would overstate the sample size"),
        "why_not_maze_seed": ("the selection split holds only 8 scenes -- too few clusters to "
                              "bootstrap; episode level is the finer permitted unit"),
        "paired": True, "family_stratified": True,
        "weightings": {
            "corpus_weighted": ("primary; masked cosines pooled over all tokens -- the estimator "
                                "used by the frozen FINAL result, so observed values reproduce "
                                "the frozen point estimates"),
            "equal_family_weighted": ("separate robustness analysis; mean of the eight "
                                      "family-level pooled scores. Does NOT replace the frozen "
                                      "point estimates"),
        },
        "interval_scope": ("variation across episodes WITHIN the present eight families; NOT "
                           "generalisation across independently sampled maze populations -- the "
                           "eight families and eight scenes are fixed, not resampled"),
        "shuffled_assay_scope": ("action-conditioning and discrimination diagnostic only; not "
                                 "candidate ranking, not planning regret"),
        "clusters_total": int(sum(len(fam_clusters[f]) for f in fam_names)),
        "clusters_per_family": {f: len(fam_clusters[f]) for f in fam_names},
        "rows": n, "resamples": B, "seed": SEED, "alpha": ALPHA,
        "masked_tokens_per_horizon": {str(h): int(tok_counts[h].sum()) for h in range(1, MAX_H + 1)},
        "models": {"a": a, "b": b, "difference": f"{a} minus {b}"},
        "pooled": {}, "per_family": {},
    }

    for h in range(1, MAX_H + 1):
        # one shared set of draws per horizon -> both models, both kinds, and the
        # per-family analyses all see the SAME resampled clusters (paired).
        picks = {f: rng.integers(0, len(fam_clusters[f]), size=(B, len(fam_clusters[f])))
                 for f in fam_names}
        Cb = {f: CC[(h, f)][picks[f]].sum(1) for f in fam_names}          # (B,)
        Cobs = {f: CC[(h, f)].sum() for f in fam_names}

        fam_boot, fam_obs = {}, {}
        for name in names:
            for kind in ("correct", "shuffled"):
                for f in fam_names:
                    s = CS[(name, h, kind, f)]
                    fam_boot[(name, kind, f)] = s[picks[f]].sum(1) / Cb[f]
                    fam_obs[(name, kind, f)] = s.sum() / Cobs[f]

        def corpus(name, kind, boot=True):
            if boot:
                num = sum(CS[(name, h, kind, f)][picks[f]].sum(1) for f in fam_names)
                den = sum(Cb[f] for f in fam_names)
            else:
                num = sum(CS[(name, h, kind, f)].sum() for f in fam_names)
                den = sum(Cobs[f] for f in fam_names)
            return num / den

        def eqfam(name, kind, boot=True):
            src = fam_boot if boot else fam_obs
            return sum(src[(name, kind, f)] for f in fam_names) / len(fam_names)

        entry = {}
        for wname, fn in (("corpus_weighted", corpus), ("equal_family_weighted", eqfam)):
            o = {f"{nm}_{k}": float(fn(nm, k, boot=False))
                 for nm in names for k in ("correct", "shuffled")}
            o[f"{a}_margin"] = o[f"{a}_correct"] - o[f"{a}_shuffled"]
            o[f"{b}_margin"] = o[f"{b}_correct"] - o[f"{b}_shuffled"]
            o["diff_correct"] = o[f"{a}_correct"] - o[f"{b}_correct"]
            o["diff_shuffled"] = o[f"{a}_shuffled"] - o[f"{b}_shuffled"]
            o["diff_margin"] = o[f"{a}_margin"] - o[f"{b}_margin"]

            d = {nm + "_" + k: np.asarray(fn(nm, k)) for nm in names
                 for k in ("correct", "shuffled")}
            d[f"{a}_margin"] = d[f"{a}_correct"] - d[f"{a}_shuffled"]
            d[f"{b}_margin"] = d[f"{b}_correct"] - d[f"{b}_shuffled"]
            d["diff_correct"] = d[f"{a}_correct"] - d[f"{b}_correct"]
            d["diff_shuffled"] = d[f"{a}_shuffled"] - d[f"{b}_shuffled"]
            d["diff_margin"] = d[f"{a}_margin"] - d[f"{b}_margin"]

            entry[wname] = {"observed": o,
                            "intervals": {k: _ci(d[k], o[k]) for k in d}}
        record["pooled"][str(h)] = entry

        fam_entry = {}
        for f in fam_names:
            o_c = fam_obs[(a, "correct", f)] - fam_obs[(b, "correct", f)]
            o_s = fam_obs[(a, "shuffled", f)] - fam_obs[(b, "shuffled", f)]
            o_m = ((fam_obs[(a, "correct", f)] - fam_obs[(a, "shuffled", f)])
                   - (fam_obs[(b, "correct", f)] - fam_obs[(b, "shuffled", f)]))
            d_c = fam_boot[(a, "correct", f)] - fam_boot[(b, "correct", f)]
            d_s = fam_boot[(a, "shuffled", f)] - fam_boot[(b, "shuffled", f)]
            d_m = ((fam_boot[(a, "correct", f)] - fam_boot[(a, "shuffled", f)])
                   - (fam_boot[(b, "correct", f)] - fam_boot[(b, "shuffled", f)]))
            fam_entry[f] = {
                "clusters": len(fam_clusters[f]),
                "rows": int(sum(len(cluster_rows[c]) for c in fam_clusters[f])),
                f"{a}_correct": float(fam_obs[(a, "correct", f)]),
                f"{b}_correct": float(fam_obs[(b, "correct", f)]),
                "diff_correct": _ci(d_c, float(o_c)),
                "diff_shuffled": _ci(d_s, float(o_s)),
                "diff_margin": _ci(d_m, float(o_m)),
            }
        record["per_family"][str(h)] = fam_entry

    out_path.write_text(json.dumps(record, indent=2))

    print(f"clusters {record['clusters_total']} across {len(fam_names)} families")
    print("  " + json.dumps(record["clusters_per_family"]))
    for wname in ("corpus_weighted", "equal_family_weighted"):
        print(f"\n[{wname}]  ({record['clusters_total']} episode clusters)")
        for h in range(1, MAX_H + 1):
            i = record["pooled"][str(h)][wname]["intervals"]
            o = record["pooled"][str(h)][wname]["observed"]
            print(f"  H={h}  {a} {o[a+'_correct']:.4f}  {b} {o[b+'_correct']:.4f}  "
                  f"dcorrect {i['diff_correct']['point']:+.4f}"
                  f"[{i['diff_correct']['ci95_low']:+.4f},{i['diff_correct']['ci95_high']:+.4f}]"
                  f"{'*' if i['diff_correct']['excludes_zero'] else ' '}  "
                  f"dshuffled {i['diff_shuffled']['point']:+.4f}"
                  f"[{i['diff_shuffled']['ci95_low']:+.4f},{i['diff_shuffled']['ci95_high']:+.4f}]"
                  f"{'*' if i['diff_shuffled']['excludes_zero'] else ' '}  "
                  f"dmargin {i['diff_margin']['point']:+.4f}"
                  f"[{i['diff_margin']['ci95_low']:+.4f},{i['diff_margin']['ci95_high']:+.4f}]"
                  f"{'*' if i['diff_margin']['excludes_zero'] else ' '}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
