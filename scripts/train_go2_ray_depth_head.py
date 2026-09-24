#!/usr/bin/env python3
"""Train the visual ray-depth head: frozen-JEPA latent → K free-depths.

A learned 1D-lidar from RGB. Gate: held-out-scene median absolute ray error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class RayDepthHead(nn.Module):
    def __init__(self, latent_dim: int, k_rays: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, k_rays),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    meta = json.loads((args.dataset_dir / "meta.json").read_text())
    shards = sorted(args.dataset_dir.glob("*.npz"))
    rng = np.random.RandomState(int(args.seed)); rng.shuffle(shards)
    n_val = max(1, int(len(shards) * float(args.val_fraction)))
    val_shards, train_shards = shards[:n_val], shards[n_val:]

    def _load(shard_list):
        lats, ys = [], []
        for p in shard_list:
            d = np.load(p)
            lats.append(d["latents"]); ys.append(d["depths"])
        return np.concatenate(lats), np.concatenate(ys)

    tl, ty = _load(train_shards)
    vl, vy = _load(val_shards)
    lat_mean, lat_std = tl.mean(0), tl.std(0) + 1e-6
    tl = (tl - lat_mean) / lat_std; vl = (vl - lat_mean) / lat_std
    print(f"train {len(ty)} rows ({len(train_shards)} scenes) | val {len(vy)} ({len(val_shards)} scenes)", flush=True)

    model = RayDepthHead(tl.shape[-1], ty.shape[-1], hidden_dim=int(args.hidden_dim)).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    TL = torch.from_numpy(tl).float(); TY = torch.from_numpy(ty).float()
    VL = torch.from_numpy(vl).float().to(device)
    best_mae, best_state = None, None
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(len(TY)); total = 0.0
        for s in range(0, len(perm), int(args.batch_size)):
            idx = perm[s : s + int(args.batch_size)]
            optimizer.zero_grad()
            loss = criterion(model(TL[idx].to(device)), TY[idx].to(device))
            loss.backward(); optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(VL), 4096):
                preds.append(model(VL[s:s+4096]).cpu().numpy())
            pv = np.concatenate(preds)
        err = np.abs(pv - vy)
        mae = float(np.median(err))
        near = vy < 1.5
        mae_near = float(np.median(err[near])) if near.any() else float("nan")
        print(f"epoch {epoch:3d} loss {total/len(TY):.4f} val_median_err {mae:.3f}m near(<1.5m) {mae_near:.3f}m", flush=True)
        if best_mae is None or mae < best_mae:
            best_mae, best_state = mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "go2_ray_depth_head_v0",
        "model_state_dict": best_state,
        "latent_dim": int(tl.shape[-1]),
        "k_rays": int(ty.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "fov_deg": float(meta["fov_deg"]),
        "depth_cap_m": float(meta["depth_cap_m"]),
        "latent_mean": lat_mean.astype(np.float32), "latent_std": lat_std.astype(np.float32),
        "frozen_jepa_checkpoint": str(json.loads((args.dataset_dir.parent / 'bc_sequences' / 'meta.json').read_text())["frozen_jepa_checkpoint"]),
        "image_size": 128,
        "best_val_median_err_m": best_mae,
    }, args.output)
    print(json.dumps({"best_val_median_err_m": best_mae, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
