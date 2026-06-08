#!/usr/bin/env python3
"""Phase A3 — frozen-latent reachability head probe + history-disambiguability.

Completes the v3 plan §4.3-§4.5 decision gate that the A2 aliasing audit
(``probe_lewm_latent_aliasing.py``) left open. A2 showed *distance-based* latent
geometry barely tracks topology; A3 asks whether a *trained/geometric readout*
can recover place/reachability from the frozen latents — i.e. whether the
information is present-but-not-L2-usable (build the belief stack) or
destroyed-at-encode-time (deeper representational problem, plan §2).

Analyses (all on the frozen encoder, raw backbone and projected spaces):

  1. Same-cell retrieval (training-free): for each observation, is its nearest
     latent neighbour in the same scene at the same true cell? Retrieval@1/@5 vs
     the same-cell chance rate. This is exactly the memory graph's place-
     recognition job and is robust to latent dimensionality.

  2. Localization probe (within-scene PCA+ridge: latent -> cell-center xy),
     "recognition" split (held-out frames of seen cells) and "metric" split
     (held-out cells). Median error (m) vs predict-the-mean baseline, and R^2.

  3. Reachability bucket head (cross-scene): linear (|z_a-z_b|) and MLP (concat)
     classifiers predicting the BFS-distance bucket of an observation pair.
     Trained on ``train``-split scenes, evaluated on ``test_id``. top-1 vs
     majority baseline, per-bucket recall, confusion, near/far confusion, and
     the mandatory bucket-distribution audit (§4.4).

  4. History-disambiguability (§4.2 item 4): among single-frame-aliased pairs at
     *different* true cells, can a short history window separate them?
     Mann-Whitney AUC of history-window distance vs single-frame distance.

CPU by default so it does not contend with a live GPU training run.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from probe_lewm_latent_aliasing import (  # noqa: E402
    BUCKETS,
    _bucket_name,
    _encode_frames,
    _find_manifest,
    _iter_label_files,
    _load_observations,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BUCKET_INDEX = {name: i for i, (name, _, _) in enumerate(BUCKETS)}
N_BUCKETS = len(BUCKETS)


# ---------------------------------------------------------------------------
# Per-scene latent bank
# ---------------------------------------------------------------------------


def build_scene_bank(model, *, label_file, family, split, render_root, corpus_root,
                     device, frames_per_scene, max_per_cell, batch_size, min_cells, rng):
    scene_id = label_file.parent.name
    render_dir = render_root / scene_id
    if not (render_dir / "summary.json").exists():
        return None
    manifest_path = _find_manifest(corpus_root, split, family, scene_id)
    if manifest_path is None:
        return None
    manifest = parse_scene_manifest_dict(json.loads(manifest_path.read_text()))
    graph = SceneGraph(manifest)
    graph_cells = {n.node_id for n in manifest.graph_nodes}

    chunk_dir = label_file.parents[1]
    rsum_path = chunk_dir / "rollout" / scene_id / "summary.json"
    by_env = _load_observations(label_file)
    n_envs = (int(json.loads(rsum_path.read_text()).get("n_envs", len(by_env)))
              if rsum_path.exists() else max(by_env) + 1)

    by_cell = defaultdict(list)
    for env_idx, obs in by_env.items():
        for step, (cell, _yaw) in enumerate(obs):
            if cell not in graph_cells:
                continue
            gi = step * n_envs + env_idx
            by_cell[cell].append(render_dir / "rgb" / f"frame_{gi:06d}_env_{env_idx:02d}.png")
    chosen = []
    for cell, items in by_cell.items():
        rng.shuffle(items)
        chosen.extend((p, cell) for p in items[:max_per_cell])
    rng.shuffle(chosen)
    chosen = [c for c in chosen if c[0].exists()][:frames_per_scene]
    if len({c for _, c in chosen}) < min_cells:
        return None

    paths = [p for p, _ in chosen]
    cells = np.array([c for _, c in chosen], dtype=np.int64)
    z_raw, z_proj = _encode_frames(model, paths, device, batch_size)
    xy = np.array([graph.cell_center(int(c)) for c in cells], dtype=np.float64)
    return {"scene_id": scene_id, "family": family, "graph": graph,
            "z": z_proj.astype(np.float64), "z_raw": z_raw.astype(np.float64),
            "cells": cells, "xy": xy, "n_envs": n_envs, "by_env": by_env,
            "render_dir": render_dir, "paths": paths}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _standardize(X, ref):
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-6] = 1.0
    return (X - mu) / sd


def _pca_fit_transform(Xtr, Xte, k):
    mu = Xtr.mean(0)
    _U, _S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    comp = Vt[:k]
    return (Xtr - mu) @ comp.T, (Xte - mu) @ comp.T


def _pairwise_sq(zs):
    G = zs @ zs.T
    sq = np.diag(G)
    D = sq[:, None] + sq[None, :] - 2 * G
    return D


# ---------------------------------------------------------------------------
# 1. Same-cell retrieval (training-free)
# ---------------------------------------------------------------------------


def same_cell_retrieval(z, cells, ks=(1, 5)):
    n = len(z)
    cells = np.asarray(cells)
    cnt = Counter(cells.tolist())
    if n < 6 or max(cnt.values()) < 2:
        return None
    D = _pairwise_sq(_standardize(z, z))
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    out = {}
    for k in ks:
        topk = order[:, :k]
        out[f"retrieval_at_{k}"] = float(np.mean([cells[i] in cells[topk[i]] for i in range(n)]))
    out["chance_same_cell"] = float(sum(c * (c - 1) for c in cnt.values()) / (n * (n - 1)))
    out["lift_at_1"] = out["retrieval_at_1"] / (out["chance_same_cell"] or 1e-9)
    return out


# ---------------------------------------------------------------------------
# 2. Localization probe (PCA + ridge)
# ---------------------------------------------------------------------------


def _ridge_fit(X, Y, alpha):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xb.T @ Y)


def _ridge_pred(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


def _loc_metrics(W, Xte, Yte, Ytr_mean):
    pred = _ridge_pred(W, Xte)
    err = np.linalg.norm(pred - Yte, axis=1)
    base = np.linalg.norm(Ytr_mean[None, :] - Yte, axis=1)
    ss_res = float(((pred - Yte) ** 2).sum())
    ss_tot = float(((Yte - Yte.mean(0)) ** 2).sum()) or 1.0
    return {"median_err_m": float(np.median(err)),
            "baseline_median_err_m": float(np.median(base)),
            "err_ratio_vs_baseline": float(np.median(err) / (np.median(base) or 1.0)),
            "r2": float(1.0 - ss_res / ss_tot)}


def localization_probe(z, xy, cells, *, alpha, k_pca, rng):
    n = len(z)
    out = {}
    idx = list(range(n)); rng.shuffle(idx); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    if len(te) >= 5 and len(tr) > 12:
        Ztr, Zte = _pca_fit_transform(z[tr], z[te], min(k_pca, len(tr) - 2))
        out["recognition"] = _loc_metrics(_ridge_fit(Ztr, xy[tr], alpha), Zte, xy[te], xy[tr].mean(0))
    uniq = list({int(c) for c in cells}); rng.shuffle(uniq); cc = int(0.7 * len(uniq))
    trc = set(uniq[:cc])
    tr = [i for i in range(n) if int(cells[i]) in trc]
    te = [i for i in range(n) if int(cells[i]) not in trc]
    if len(te) >= 5 and len(tr) > 12:
        Ztr, Zte = _pca_fit_transform(z[tr], z[te], min(k_pca, len(tr) - 2))
        out["metric"] = _loc_metrics(_ridge_fit(Ztr, xy[tr], alpha), Zte, xy[te], xy[tr].mean(0))
    return out


# ---------------------------------------------------------------------------
# 3. Reachability bucket head
# ---------------------------------------------------------------------------


def sample_pairs(bank, *, per_bucket_per_scene, rng):
    z, cells, graph = bank["z"], bank["cells"], bank["graph"]
    n = len(z)
    by_bucket = defaultdict(list)
    bfs_cache = {}
    for _ in range(per_bucket_per_scene * N_BUCKETS * 12):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        ci, cj = int(cells[i]), int(cells[j])
        key = (ci, cj) if ci <= cj else (cj, ci)
        if key not in bfs_cache:
            bfs_cache[key] = graph.bfs_distance(ci, cj)
        d = bfs_cache[key]
        if d is None:
            continue
        b = BUCKET_INDEX[_bucket_name(d)]
        if len(by_bucket[b]) < per_bucket_per_scene:
            by_bucket[b].append((i, j))
    A, C, Y = [], [], []
    for b, pairs in by_bucket.items():
        for i, j in pairs:
            za, zb = z[i], z[j]
            A.append(np.abs(za - zb))
            C.append(np.concatenate([za, zb, np.abs(za - zb)]))
            Y.append(b)
    return (np.asarray(A), np.asarray(C), np.asarray(Y, dtype=np.int64)) if Y else None


class _MLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=0):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, n_cls)) if hidden else nn.Linear(in_dim, n_cls))

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, Xte, yte, *, hidden, epochs, device, present):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-6] = 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    counts = np.bincount(ytr, minlength=N_BUCKETS).astype(np.float64)
    w = np.zeros(N_BUCKETS); nz = counts > 0
    w[nz] = counts[nz].sum() / (counts[nz] * nz.sum())
    model = _MLP(Xtr.shape[1], N_BUCKETS, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 2048):
            b = perm[k:k + 2048]
            opt.zero_grad(); lossf(model(Xt[b]), yt[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ptr = model(Xt).argmax(1).cpu().numpy()
        pte = model(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    maj = int(counts.argmax())
    conf = np.zeros((N_BUCKETS, N_BUCKETS), dtype=int)
    for t, p in zip(yte, pte):
        conf[t, p] += 1
    recall = {BUCKETS[b][0]: float((pte[yte == b] == b).mean()) for b in present if (yte == b).any()}
    far, near = N_BUCKETS - 1, 0
    nf = ((yte == far) & (pte == near)) | ((yte == near) & (pte == far))
    denom = int(((yte == far) | (yte == near)).sum())
    return {"top1_train": float((ptr == ytr).mean()),
            "top1_eval": float((pte == yte).mean()),
            "majority_baseline_eval": float((yte == maj).mean()),
            "eval_minus_baseline": float((pte == yte).mean() - (yte == maj).mean()),
            "per_bucket_recall_eval": recall,
            "near_far_confusion_frac": float(nf.sum() / denom) if denom else None,
            "spearman_pred_vs_true": (None if pte.std() == 0
                                      else float(np.corrcoef(yte.astype(float), pte.astype(float))[0, 1])),
            "confusion_matrix": conf.tolist()}


# ---------------------------------------------------------------------------
# 4. History-disambiguability (§4.2 item 4)
# ---------------------------------------------------------------------------


def _auc(pos, neg):
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def history_disambiguation(model, bank, *, device, H_values, seg_len, n_envs_sample,
                           max_obs, batch_size, rng):
    by_env, n_envs, render_dir = bank["by_env"], bank["n_envs"], bank["render_dir"]
    graph_cells = {n.node_id for n in bank["graph"].manifest.graph_nodes}
    envs = [e for e in by_env if len(by_env[e]) >= seg_len]
    if not envs:
        return None
    rng.shuffle(envs)
    segments = []  # (zp[seg], cells[seg])
    for e in envs[:n_envs_sample]:
        obs = by_env[e]
        start = rng.randrange(0, len(obs) - seg_len + 1)
        paths, cells = [], []
        for s in range(start, start + seg_len):
            cell, _ = obs[s]
            gi = s * n_envs + e
            paths.append(render_dir / "rgb" / f"frame_{gi:06d}_env_{e:02d}.png")
            cells.append(cell if cell in graph_cells else -1)
        if not all(p.exists() for p in paths):
            continue
        _zr, zp = _encode_frames(model, paths, device, batch_size)
        segments.append((zp.astype(np.float64), np.asarray(cells)))
    if not segments:
        return None

    report = {}
    for H in H_values:
        singles, hists, cellv = [], [], []
        for zp, cells in segments:
            for s in range(H - 1, len(zp)):
                if cells[s] < 0:
                    continue
                singles.append(zp[s]); hists.append(zp[s - H + 1:s + 1].mean(0)); cellv.append(int(cells[s]))
        if len(singles) < 12:
            report[f"H{H}"] = {"n_aliased_diff_cell": 0, "n_aliased_same_cell": 0,
                               "auc_single_frame": None, "auc_history_window": None}
            continue
        singles = np.asarray(singles); hists = np.asarray(hists); cellv = np.asarray(cellv)
        m = len(singles)
        idx = list(range(m))
        pairs = [(a, b) for ii, a in enumerate(idx) for b in idx[ii + 1:]]
        if len(pairs) > max_obs:
            pairs = rng.sample(pairs, max_obs)
        ssd = _standardize(singles, singles)
        sd = np.array([np.linalg.norm(ssd[a] - ssd[b]) for a, b in pairs])
        thresh = np.quantile(sd, 0.10)
        same_s, diff_s, same_h, diff_h = [], [], [], []
        for (a, b), d in zip(pairs, sd):
            if d > thresh:
                continue
            hd = float(np.linalg.norm(hists[a] - hists[b]))
            (same_s if cellv[a] == cellv[b] else diff_s).append(d)
            (same_h if cellv[a] == cellv[b] else diff_h).append(hd)
        report[f"H{H}"] = {"n_aliased_diff_cell": len(diff_h), "n_aliased_same_cell": len(same_h),
                           "auc_single_frame": _auc(np.array(diff_s), np.array(same_s)),
                           "auc_history_window": _auc(np.array(diff_h), np.array(same_h))}
    return report


# ---------------------------------------------------------------------------


def _agg(values):
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max())}


def _select(rollout_root, split, per_family, seed):
    per = defaultdict(list)
    for fam, lf in _iter_label_files(rollout_root, split, None):
        per[fam].append(lf)
    rng = random.Random(seed)
    sel = []
    for fam in sorted(per):
        files = sorted(per[fam]); rng.shuffle(files)
        sel.extend((fam, lf) for lf in files[:per_family])
    return sel


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollout-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/rollout")
    p.add_argument("--render-root", type=Path, default=REPO_ROOT / ".generated/datagen_full/render_textured_v03")
    p.add_argument("--manifest-corpus", type=Path,
                   default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test_id")
    p.add_argument("--train-scenes-per-family", type=int, default=4)
    p.add_argument("--eval-scenes-per-family", type=int, default=4)
    p.add_argument("--eval-frames-per-scene", type=int, default=240)
    p.add_argument("--train-frames-per-scene", type=int, default=100)
    p.add_argument("--eval-max-per-cell", type=int, default=8)
    p.add_argument("--train-max-per-cell", type=int, default=3)
    p.add_argument("--per-bucket-per-scene", type=int, default=150)
    p.add_argument("--min-cells", type=int, default=6)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--pca-k", type=int, default=48)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--mlp-epochs", type=int, default=50)
    p.add_argument("--history-scenes", type=int, default=8)
    p.add_argument("--history-H", default="4,8")
    p.add_argument("--history-seg-len", type=int, default=200)
    p.add_argument("--history-envs", type=int, default=4)
    p.add_argument("--history-max-pairs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--sigreg-lambda", type=float, default=None)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=20260604)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Only run the same-cell retrieval readout. This is the fast ablation "
            "screening mode; it skips localization, reachability-head training, "
            "and history disambiguation."
        ),
    )
    p.add_argument("--skip-localization", action="store_true")
    p.add_argument("--skip-reachability", action="store_true")
    p.add_argument("--skip-history", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    logger.info("A3 reachability probe on %s", device)
    model, _cfg = load_model(args, device)
    run_localization = not (args.retrieval_only or args.skip_localization)
    run_reachability = not (args.retrieval_only or args.skip_reachability)
    run_history = not (args.retrieval_only or args.skip_history)

    def banks(split, per_family, frames, max_per_cell):
        out = []
        for idx, (fam, lf) in enumerate(_select(args.rollout_root, split, per_family, args.seed)):
            b = build_scene_bank(model, label_file=lf, family=fam, split=split,
                                 render_root=args.render_root, corpus_root=args.manifest_corpus,
                                 device=device, frames_per_scene=frames, max_per_cell=max_per_cell,
                                 batch_size=args.batch_size, min_cells=args.min_cells,
                                 rng=random.Random(args.seed + idx * 7919))
            if b is not None:
                out.append(b)
                logger.info("  bank %s %s frames=%d cells=%d", split, b["scene_id"],
                            len(b["z"]), len(set(b["cells"].tolist())))
        return out

    logger.info("Building eval banks (%s)...", args.eval_split)
    eval_banks = banks(args.eval_split, args.eval_scenes_per_family, args.eval_frames_per_scene, args.eval_max_per_cell)
    train_banks = []
    if run_reachability:
        logger.info("Building train banks (%s)...", args.train_split)
        train_banks = banks(args.train_split, args.train_scenes_per_family, args.train_frames_per_scene, args.train_max_per_cell)

    # --- 1+2 retrieval & localization (per latent space) ---
    place = {}
    for space, key in (("proj", "z"), ("raw", "z_raw")):
        ret1, ret5, lift, rec_r2, met_r2, rec_ratio = [], [], [], [], [], []
        for b in eval_banks:
            r = same_cell_retrieval(b[key], b["cells"])
            if r:
                ret1.append(r["retrieval_at_1"]); ret5.append(r["retrieval_at_5"]); lift.append(r["lift_at_1"])
            if run_localization:
                loc = localization_probe(b[key], b["xy"], b["cells"], alpha=args.ridge_alpha,
                                         k_pca=args.pca_k, rng=random.Random(args.seed))
                if "recognition" in loc:
                    rec_r2.append(loc["recognition"]["r2"]); rec_ratio.append(loc["recognition"]["err_ratio_vs_baseline"])
                if "metric" in loc:
                    met_r2.append(loc["metric"]["r2"])
        place[space] = {
            "retrieval_at_1": _agg(ret1), "retrieval_at_5": _agg(ret5), "lift_at_1": _agg(lift),
            "localization_recognition_r2": _agg(rec_r2),
            "localization_recognition_err_ratio": _agg(rec_ratio),
            "localization_metric_r2": _agg(met_r2),
        }

    # --- 3 reachability head ---
    def pool(bs):
        A, C, Y = [], [], []
        for b in bs:
            s = sample_pairs(b, per_bucket_per_scene=args.per_bucket_per_scene, rng=random.Random(args.seed))
            if s:
                A.append(s[0]); C.append(s[1]); Y.append(s[2])
        return (np.concatenate(A), np.concatenate(C), np.concatenate(Y)) if A else None
    reachability = {}
    if run_reachability:
        tr, te = pool(train_banks), pool(eval_banks)
        if tr and te:
            present = sorted(set(tr[2].tolist()) | set(te[2].tolist()))
            reachability = {
                "bucket_audit": {
                    "train_counts": {BUCKETS[i][0]: int((tr[2] == i).sum()) for i in range(N_BUCKETS)},
                    "eval_counts": {BUCKETS[i][0]: int((te[2] == i).sum()) for i in range(N_BUCKETS)}},
                "linear_abs_diff": train_head(tr[0], tr[2], te[0], te[2], hidden=0,
                                              epochs=args.mlp_epochs, device=device, present=present),
                "mlp_concat": train_head(tr[1], tr[2], te[1], te[2], hidden=args.mlp_hidden,
                                         epochs=args.mlp_epochs, device=device, present=present)}

    # --- 4 history ---
    hist = defaultdict(lambda: defaultdict(list))
    if run_history:
        H_values = [int(x) for x in str(args.history_H).split(",") if x.strip()]
        for b in eval_banks[: args.history_scenes]:
            rep = history_disambiguation(model, b, device=device, H_values=H_values,
                                         seg_len=args.history_seg_len, n_envs_sample=args.history_envs,
                                         max_obs=args.history_max_pairs, batch_size=args.batch_size,
                                         rng=random.Random(args.seed))
            if rep:
                for hk, v in rep.items():
                    hist[hk]["single"].append(v["auc_single_frame"])
                    hist[hk]["hist"].append(v["auc_history_window"])
                    hist[hk]["n"].append(v["n_aliased_diff_cell"])
    history = {hk: {"auc_single_frame": _agg(d["single"]), "auc_history_window": _agg(d["hist"]),
                    "n_aliased_diff_cell_total": int(sum(d["n"]))} for hk, d in hist.items()}

    # --- A4 verdict ---
    mlp = reachability.get("mlp_concat", {})
    ret1 = place["proj"]["retrieval_at_1"].get("median", 0.0)
    if not run_reachability:
        verdict = "not evaluated: retrieval-only/readout-only mode skipped the reachability head"
    elif mlp.get("top1_eval", 0) >= 0.70 and mlp.get("eval_minus_baseline", 0) >= 0.15:
        verdict = "strong frozen-latent regime (reachability head >=70% and beats baseline)"
    elif mlp.get("eval_minus_baseline", 1) <= 0.15:
        verdict = ("insufficient: cross-scene reachability head <= majority baseline + 15pp "
                   "-> reachability not transferably recoverable from frozen latents")
    else:
        verdict = "ambiguous: head beats baseline but below 70% top-1"

    record = {
        "schema": "lewm_reachability_a3_probe_v0",
        "checkpoint": str(args.checkpoint),
        "eval_split": args.eval_split, "train_split": args.train_split,
        "eval_scenes": len(eval_banks), "train_scenes": len(train_banks),
        "place_recognition": place,
        "reachability": reachability,
        "history_disambiguation": history,
        "a4_interpretation": verdict,
        "notes": {
            "retrieval": "training-free same-cell nearest-neighbour. retrieval_at_1 >> chance_same_cell (lift_at_1 >> 1) = place is recognisable from a single frame; this is the memory graph's core job.",
            "localization_recognition": "PCA+ridge latent->cell-center on held-out FRAMES (R^2>0, ratio<1 = decodable).",
            "localization_metric": "same on held-out CELLS (metric generalisation; harder).",
            "reachability": "cross-scene head trained on train-split pairs, eval on eval-split; eval_minus_baseline is the A4 number. top1_train>>top1_eval=baseline means signal is scene-specific / not transferable.",
            "history": "AUC>0.5 means a history window separates aliased different-cell pairs single frames cannot (BeliefEncoder viability)."},
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    print(json.dumps({k: v for k, v in record.items() if k != "config"}, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
