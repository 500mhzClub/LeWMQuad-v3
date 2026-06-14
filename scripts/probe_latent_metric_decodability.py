#!/usr/bin/env python3
"""Decodability-ceiling probe: can a supervised value head regress TRUE distance
from frozen latent pairs?

Planning-by-prediction needs a goal-conditioned VALUE function V(z_state, z_goal)
that approximates distance-to-goal (the smooth far-field gradient an MPC cost
needs). The energy head we shipped was a *recognition* ranker (goal-place vs other
place), never a distance regressor. This probe trains V(z_a, z_b) -> ||pose_a -
pose_b|| on within-window latent pairs (true geometry from cmd integration) using
the FROZEN encoded latents (z_proj), so it measures the ceiling of the latent
SPACE, independent of predictor error.

Reports, on held-out pairs:
  - overall Pearson/Spearman(predicted, true)         -> is distance decodable at all?
  - WITHIN-band Spearman + MAE per distance band      -> at what RESOLUTION?
  - bare-L2(z_a, z_b) correlation                      -> the naive cost baseline
A high within-band correlation at the ~0.1-0.3 m scale = the latent can support a
metric value function (planning rescued, no backbone change). Near-zero fine
resolution = the metric isn't in the latent (-> backbone change or subgoal chaining).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.energy_head import GoalEnergyHead  # noqa: E402  (reuse arch as value net)
from train_lewm_energy_head import _window_xy  # noqa: E402  (cmd -> within-window xy)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def build_pairs(cache: dict, dt: float, latent_key: str = "z_proj"):
    """All within-window latent pairs (a<b) with true kinematic distance labels."""
    zp = cache[latent_key].float()               # (N, T, D)
    cmd = cache["cmd"].cpu().numpy()             # (N, T, A)
    n, t, d = zp.shape
    xy = np.stack([_window_xy(cmd[i], dt) for i in range(n)])  # (N, T, 2)
    a_idx, b_idx = np.triu_indices(t, k=1)
    za = zp[:, a_idx, :].reshape(-1, d).contiguous()
    zb = zp[:, b_idx, :].reshape(-1, d).contiguous()
    dist = np.linalg.norm(xy[:, a_idx] - xy[:, b_idx], axis=-1).reshape(-1)
    return za, zb, torch.from_numpy(dist).float()


@torch.no_grad()
def _predict(head, za, zb, batch=8192):
    out = []
    for i in range(0, za.shape[0], batch):
        out.append(head(za[i:i + batch], zb[i:i + batch]).cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-cache", type=Path, required=True)
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--command-dt-s", type=float, default=0.10)
    ap.add_argument("--latent-key", default="z_proj", choices=("z_proj", "z_raw"),
                    help="Which cached latent to decode from (z_raw = pre-projection encoder features).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    tr = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    ev = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
    za_tr, zb_tr, d_tr = build_pairs(tr, args.command_dt_s, args.latent_key)
    za_ev, zb_ev, d_ev = build_pairs(ev, args.command_dt_s, args.latent_key)
    print(f"latent={args.latent_key}", flush=True)
    D = za_tr.shape[-1]
    print(f"pairs: train {za_tr.shape[0]} | eval {za_ev.shape[0]} | D={D} | "
          f"dist range [{d_tr.min():.2f}, {d_tr.max():.2f}] m", flush=True)

    head = GoalEnergyHead(latent_dim=D, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    n = za_tr.shape[0]
    for ep in range(args.epochs):
        head.train()
        idx = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            b = idx[i:i + args.batch_size]
            pred = head(za_tr[b].to(device), zb_tr[b].to(device))
            loss = F.smooth_l1_loss(pred, d_tr[b].to(device))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 10 == 0 or ep == args.epochs - 1:
            head.eval()
            pe = _predict(head, za_ev.to(device), zb_ev.to(device))
            print(f"ep {ep:2d} | train Huber {tot / n:.4f} | "
                  f"eval pearson {_pearson(pe, d_ev.numpy()):+.3f} spearman {_spearman(pe, d_ev.numpy()):+.3f}",
                  flush=True)

    head.eval()
    pred = _predict(head, za_ev.to(device), zb_ev.to(device))
    true = d_ev.numpy()
    l2 = torch.sqrt(((za_ev - zb_ev) ** 2).sum(-1)).numpy()  # bare L2 in z_proj space

    # Resolution: WITHIN-band correlation (can it order pairs of similar true dist?)
    bands = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 3.0)]
    band_out = []
    for lo, hi in bands:
        m = (true >= lo) & (true < hi)
        if m.sum() < 10:
            band_out.append({"band_m": [lo, hi], "n": int(m.sum())}); continue
        band_out.append({
            "band_m": [lo, hi], "n": int(m.sum()),
            "within_spearman": _spearman(pred[m], true[m]),
            "mae_m": float(np.abs(pred[m] - true[m]).mean()),
            "mean_pred_m": float(pred[m].mean()), "mean_true_m": float(true[m].mean()),
        })

    summary = {
        "schema": "latent_metric_decodability_v0",
        "train_cache": str(args.train_cache), "eval_cache": str(args.eval_cache),
        "n_eval_pairs": int(true.shape[0]),
        "value_head": {"pearson": _pearson(pred, true), "spearman": _spearman(pred, true),
                       "mae_m": float(np.abs(pred - true).mean())},
        "bare_l2_baseline": {"pearson": _pearson(l2, true), "spearman": _spearman(l2, true)},
        "by_distance_band": band_out,
    }

    print("\n=== DECODABILITY CEILING (eval pairs) ===")
    v = summary["value_head"]; bl = summary["bare_l2_baseline"]
    print(f"value head : pearson {v['pearson']:+.3f}  spearman {v['spearman']:+.3f}  MAE {v['mae_m']:.3f} m")
    print(f"bare L2    : pearson {bl['pearson']:+.3f}  spearman {bl['spearman']:+.3f}")
    print(f"  {'band (m)':>12s} {'n':>7s} {'within_rho':>11s} {'MAE_m':>7s} {'pred|true':>14s}")
    for bo in band_out:
        if "within_spearman" not in bo:
            print(f"  {str(bo['band_m']):>12s} {bo['n']:>7d}   (too few)"); continue
        print(f"  {str(bo['band_m']):>12s} {bo['n']:>7d} {bo['within_spearman']:>+11.3f} "
              f"{bo['mae_m']:>7.3f} {bo['mean_pred_m']:>6.2f}|{bo['mean_true_m']:<6.2f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
