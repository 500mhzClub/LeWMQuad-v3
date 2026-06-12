#!/usr/bin/env python3
"""Stage 0 of the rollout-objective fast-validation ladder (docs §12).

Encodes a subset of dataset windows at a long seq_len with the FROZEN checkpoint
encoder and saves (z_raw, z_proj, cmd) tensors, so the predictor can be
trained/ablated without images or the encoder. Run once per holdout role.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from probe_lewm_checkpoint import load_model  # noqa: E402
from train_lewm import GenesisWMDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--render-root", type=Path, default=None)
    p.add_argument("--seq-len", type=int, default=11)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--holdout-role", choices=("train", "eval"), default="train")
    p.add_argument("--holdout-fraction", type=float, default=0.02)
    p.add_argument("--holdout-seed", type=int, default=20260524)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--max-windows", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=20260605)
    p.add_argument("--allow-material-color-render", action="store_true")
    p.add_argument("--include-pose-labels", action="store_true")
    p.add_argument("--max-seq-len", type=int, default=None)  # load_model compat
    p.add_argument("--sigreg-lambda", type=float, default=None)  # load_model compat
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for latent caching, but torch.cuda.is_available() is false")
    model, cfg = load_model(args, torch.device("cpu") if device.type == "cuda" else device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA became unavailable after checkpoint load; refusing CPU fallback")
        model.to(device)
    model.eval()
    render_root = args.render_root or (
        Path(cfg["render_root"]) if "render_root" in cfg else None
    )
    allow_material = bool(
        args.allow_material_color_render
        or cfg.get("allow_material_color_render", False)
    )

    ds = GenesisWMDataset(
        root_dir=args.data_root, render_root=render_root, seq_len=args.seq_len,
        stride=args.stride, max_sessions=args.max_sessions,
        allow_material_color_render=allow_material,
        holdout_fraction=args.holdout_fraction, holdout_role=args.holdout_role,
        holdout_seed=args.holdout_seed,
        include_pose_labels=args.include_pose_labels,
    )
    if len(ds) == 0:
        raise SystemExit("cache dataset is empty")
    n = min(len(ds), args.max_windows)
    idx = random.Random(args.sample_seed).sample(range(len(ds)), n)
    loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    zr, zp, cm, ps = [], [], [], []
    done = 0
    for b in loader:
        vis = b["vis_seq"].to(device)
        z_raw, z_proj = model.encode_seq(vis, None)
        zr.append(z_raw.cpu().float())
        zp.append(z_proj.cpu().float())
        cm.append(b["cmd_seq"].cpu().float())
        if args.include_pose_labels:
            ps.append(b["pose_seq"].cpu().float())
        done += vis.shape[0]
        if done % 512 < args.batch_size:
            logger.info("encoded %d/%d windows", done, n)

    out = {
        "z_raw": torch.cat(zr), "z_proj": torch.cat(zp), "cmd": torch.cat(cm),
        "seq_len": args.seq_len, "stride": args.stride,
        "checkpoint": str(args.checkpoint), "holdout_role": args.holdout_role,
        "holdout_fraction": float(args.holdout_fraction),
        "holdout_seed": int(args.holdout_seed),
        "max_sessions": args.max_sessions,
        "sample_seed": int(args.sample_seed),
        "render_root": str(ds.render_root),
        "allow_material_color_render": allow_material,
        "include_pose_labels": bool(args.include_pose_labels),
        "latent_dim": int(zr[0].shape[-1]), "cmd_dim": int(cm[0].shape[-1]),
    }
    if ps:
        out["pose"] = torch.cat(ps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    logger.info("Cached %d windows seq_len=%d -> %s  (z_raw %s, cmd %s)",
                out["z_raw"].shape[0], args.seq_len, args.output,
                tuple(out["z_raw"].shape), tuple(out["cmd"].shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
