#!/usr/bin/env python
"""Train a GoalEnergyHead on cached LeWM latents for image-goal planning.

Recipe (ported from TinyQuadJEPA-v2 script 4): form (z_pred, z_goal) positive
pairs by rolling the FROZEN predictor H steps from a window start under the
recorded commands -> z_pred (projected planning space); the goal is the window's
actual observation latent H steps ahead (cached z_proj[:, H]). Negatives are
other windows' goals (in-batch). The head learns to rank the true goal below
wrong goals, giving a learned planning cost that — unlike bare latent L2 — does
not need the LeJEPA latent to be metric.

Caches come from scripts/cache_lewm_latents.py (keys z_raw, z_proj, cmd), so the
head trains in seconds-to-minutes with no image I/O.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.actions import ACTIVE_BLOCK_DIM, active_block_to_matrix  # noqa: E402
from lewm.models.energy_head import (  # noqa: E402
    GoalEnergyHead,
    energy_ranking_loss,
    sample_negative_goals,
)
from probe_lewm_checkpoint import load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_energy_head")

# Platform timing.command_dt_s. Only sets the absolute scale of the cmd-integrated
# within-window displacement (meters), which --dist-margin-scale then maps to a
# margin; the monotone ORDERING is dt-independent.
COMMAND_DT_S = 0.10


def _window_xy(cmd_window: np.ndarray, dt: float = COMMAND_DT_S) -> np.ndarray:
    """Kinematic xy at each step boundary from a window's active-block commands.

    cmd_window: (seq_len, ACTIVE_BLOCK_DIM). Returns (seq_len, 2) in the window's
    start frame, where row t is the pose after executing steps 0..t-1 (row 0 =
    origin). Mirrors the kinematic integration the nav benchmark uses, so the
    head's distance target matches the cost it will be evaluated under.
    """
    t = cmd_window.shape[0]
    xy = np.zeros((t, 2), dtype=np.float64)
    x = y = yaw = 0.0
    for step in range(t - 1):
        for vx, vy, yaw_rate in active_block_to_matrix(cmd_window[step]):
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            x += (float(vx) * cos_y - float(vy) * sin_y) * dt
            y += (float(vx) * sin_y + float(vy) * cos_y) * dt
            yaw += float(yaw_rate) * dt
        xy[step + 1] = (x, y)
    return xy


@torch.no_grad()
def build_pairs(model, z_raw, cmd, z_proj, horizons, device, batch=512, hard_offsets=(),
                command_dt_s=COMMAND_DT_S):
    """Return (z_pred, z_goal, z_hard, z_hard_disp) over all horizons.

    z_hard is (M, K, D) of same-trajectory nearby-step negatives (None when
    hard_offsets is empty). z_hard_disp is (M, K) of the TRUE kinematic distance
    (meters) between the goal step h and each hard-negative step, derived by
    integrating the cached commands. It lets the loss scale the ranking margin by
    true distance -> energy monotone in distance (a metric cost, not a classifier).
    """
    preds, goals, hards, hdisp = [], [], [], []
    n, t, _ = z_raw.shape
    win_xy = None
    if hard_offsets:
        if cmd.shape[-1] != ACTIVE_BLOCK_DIM:
            logger.warning("cmd_dim=%d != ACTIVE_BLOCK_DIM=%d; disabling distance margins",
                           cmd.shape[-1], ACTIVE_BLOCK_DIM)
        else:
            cmd_np = cmd.cpu().numpy()
            win_xy = np.stack([_window_xy(cmd_np[i], command_dt_s) for i in range(n)])  # (n, t, 2)
    for h in horizons:
        if h < 1 or h >= t:
            logger.warning("skipping horizon %d (needs 1 <= h < seq_len=%d)", h, t)
            continue
        for i in range(0, n, batch):
            zr0 = z_raw[i:i + batch, 0].to(device)
            acts = cmd[i:i + batch, :h].to(device)
            z_pred = model.plan_rollout(zr0, acts)[:, -1, :]   # (b, D) pred_projector space
            preds.append(z_pred.float().cpu())
            goals.append(z_proj[i:i + batch, h].float().clone())  # (b, D) enc_projector space
            if hard_offsets:
                # Hard negatives: SAME trajectory at nearby steps (nearby places).
                # Forces "the goal step" to rank below "a step before/after" =
                # the fine closer-vs-farther gradient navigation needs.
                idxs = [min(max(h + off, 1), t - 1) for off in hard_offsets]
                hs = [z_proj[i:i + batch, j].float() for j in idxs]
                hards.append(torch.stack(hs, dim=1))  # (b, K, D)
                if win_xy is not None:
                    bxy = win_xy[i:i + batch]  # (b, t, 2)
                    d = np.stack([np.linalg.norm(bxy[:, h] - bxy[:, j], axis=1) for j in idxs], axis=1)
                    hdisp.append(torch.from_numpy(d).float())  # (b, K) meters
    z_hard = torch.cat(hards) if hards else None
    z_hard_disp = torch.cat(hdisp) if hdisp else None
    return torch.cat(preds), torch.cat(goals), z_hard, z_hard_disp


def run_epoch(head, opt, z_pred, z_goal, z_hard, z_hard_disp, batch, num_neg, margin,
              device, train, dist_scale=0.0):
    head.train(train)
    n = z_pred.shape[0]
    idx = torch.randperm(n) if train else torch.arange(n)
    tot = {"loss": 0.0, "acc": 0.0, "hard_acc": 0.0, "gap": 0.0, "n": 0}
    for i in range(0, n, batch):
        b = idx[i:i + batch]
        zpb = z_pred[b].to(device)
        zgb = z_goal[b].to(device)
        rand_neg = sample_negative_goals(zgb, num_neg)
        if z_hard is not None:
            hard_neg = z_hard[b].to(device)                    # (B, K, D)
            zneg = torch.cat([hard_neg, rand_neg], dim=1)
            # Per-negative margin: distance-scaled for hard negs (the goal step must
            # sit below a nearby step by ~ their true separation -> energy monotone
            # in distance), constant for cross-scene random negs.
            bsz = zpb.shape[0]
            if dist_scale > 0.0 and z_hard_disp is not None:
                hard_m = float(margin) + float(dist_scale) * z_hard_disp[b].to(device)  # (B, K)
            else:
                hard_m = torch.full((bsz, hard_neg.shape[1]), float(margin), device=device)
            rand_m = torch.full((bsz, rand_neg.shape[1]), float(margin), device=device)
            margins = torch.cat([hard_m, rand_m], dim=1)        # (B, K+k)
        else:
            hard_neg = None
            zneg = rand_neg
            margins = margin
        loss, stats = energy_ranking_loss(head, zpb, zgb, zneg, margins)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # Hard-negative-only ranking accuracy (the decisive diagnostic).
        hard_acc = 0.0
        if hard_neg is not None:
            with torch.no_grad():
                pe = head(zpb, zgb)                            # (B,)
                k = hard_neg.shape[1]
                d = zpb.shape[-1]
                he = head(zpb.unsqueeze(1).expand(-1, k, -1).reshape(-1, d),
                          hard_neg.reshape(-1, d)).view(zpb.shape[0], k)
                hard_acc = float((pe[:, None] < he).float().mean())
        w = zpb.shape[0]
        tot["loss"] += stats["loss"] * w
        tot["acc"] += stats["ranking_acc"] * w
        tot["hard_acc"] += hard_acc * w
        tot["gap"] += stats["gap"] * w
        tot["n"] += w
    return {k: tot[k] / max(tot["n"], 1) for k in ("loss", "acc", "hard_acc", "gap")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--eval-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--horizons", default="3,5,8,10")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--num-negatives", type=int, default=16)
    p.add_argument("--hard-neg-offsets", default="",
                   help="Comma step-offsets for same-trajectory hard negatives, e.g. '-2,-1,1,2' "
                        "(empty = easy random negatives only).")
    p.add_argument("--dist-margin-scale", type=float, default=0.0,
                   help="If >0 (with --hard-neg-offsets), scale the hard-negative ranking "
                        "margin by the true cmd-integrated distance (m) between the goal step "
                        "and the negative step, making energy monotone in distance. "
                        "0 = constant margin (legacy).")
    p.add_argument("--command-dt-s", type=float, default=COMMAND_DT_S,
                   help="Tick dt for the cmd-integrated within-window distance (affects only "
                        "the absolute margin scale).")
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--sigreg-lambda", type=float, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    logger.info("device=%s horizons=%s", device, horizons)

    margs = SimpleNamespace(
        checkpoint=args.checkpoint,
        max_seq_len=args.max_seq_len,
        sigreg_lambda=args.sigreg_lambda,
    )
    model, _ = load_model(margs, device)
    model.eval()
    latent_dim = int(model.latent_dim)

    tr = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    ev = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
    logger.info("caches: train z_raw %s | eval z_raw %s",
                tuple(tr["z_raw"].shape), tuple(ev["z_raw"].shape))

    hard_offsets = tuple(int(x) for x in args.hard_neg_offsets.split(",") if x.strip())
    logger.info("hard-negative offsets: %s | dist-margin-scale=%.3g",
                hard_offsets or "(none / easy only)", args.dist_margin_scale)
    zp_tr, zg_tr, zh_tr, zhd_tr = build_pairs(model, tr["z_raw"], tr["cmd"], tr["z_proj"],
                                              horizons, device, args.batch_size, hard_offsets, args.command_dt_s)
    zp_ev, zg_ev, zh_ev, zhd_ev = build_pairs(model, ev["z_raw"], ev["cmd"], ev["z_proj"],
                                              horizons, device, args.batch_size, hard_offsets, args.command_dt_s)
    logger.info("pairs: train %d | eval %d | D=%d", zp_tr.shape[0], zp_ev.shape[0], latent_dim)

    head = GoalEnergyHead(latent_dim=latent_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    # When hard negatives are on, select the checkpoint by HARD ranking accuracy.
    sel = "hard_acc" if hard_offsets else "acc"

    best_acc, best_state = 0.0, None
    for ep in range(args.epochs):
        tr_m = run_epoch(head, opt, zp_tr, zg_tr, zh_tr, zhd_tr, args.batch_size,
                         args.num_negatives, args.margin, device, True, args.dist_margin_scale)
        with torch.no_grad():
            ev_m = run_epoch(head, opt, zp_ev, zg_ev, zh_ev, zhd_ev, args.batch_size,
                             args.num_negatives, args.margin, device, False, args.dist_margin_scale)
        if ev_m[sel] > best_acc:
            best_acc = ev_m[sel]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            logger.info("ep %2d | train: easy %.3f hard %.3f | eval: easy %.3f hard %.3f gap %.2f",
                        ep, tr_m["acc"], tr_m["hard_acc"], ev_m["acc"], ev_m["hard_acc"], ev_m["gap"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": best_state,
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "horizons": horizons,
        "margin": args.margin,
        "num_negatives": args.num_negatives,
        "hard_neg_offsets": list(hard_offsets),
        "dist_margin_scale": args.dist_margin_scale,
        "command_dt_s": args.command_dt_s,
        "source_checkpoint": str(args.checkpoint),
        "best_eval_ranking_acc": best_acc,
    }, args.output)
    logger.info("BEST eval ranking acc %.4f  ->  %s", best_acc, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
