#!/usr/bin/env python3
"""Train the action-conditioned beacon route-distance head.

(latent_t ⊕ proprio_t) → Δbfs_distance(t→t+1) per beacon color. The executed
primitive one-hot lives inside proprio, so at runtime candidate primitives are
scored by swapping the one-hot and taking the predicted descent for the active
color. The BFS labels give the head global route structure no local signal has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

COLORS = ("red", "yellow", "blue", "green")


class BeaconDistanceHead(nn.Module):
    def __init__(self, latent_dim: int, proprio_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + proprio_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, len(COLORS)),
        )

    def forward(self, latents: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latents, proprio], dim=-1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
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
        lats, pros, ys = [], [], []
        for p in shard_list:
            d = np.load(p)
            lats.append(d["latents"]); pros.append(d["proprio"]); ys.append(d["delta_bfs"])
        return np.concatenate(lats), np.concatenate(pros), np.concatenate(ys)

    tl, tp, ty = _load(train_shards)
    vl, vp, vy = _load(val_shards)
    lat_mean, lat_std = tl.mean(0), tl.std(0) + 1e-6
    pro_mean, pro_std = tp.mean(0), tp.std(0) + 1e-6
    tl = (tl - lat_mean) / lat_std; vl = (vl - lat_mean) / lat_std
    tp = (tp - pro_mean) / pro_std; vp = (vp - pro_mean) / pro_std
    print(f"train {len(ty)} rows ({len(train_shards)} scenes) | val {len(vy)} ({len(val_shards)} scenes)", flush=True)

    model = BeaconDistanceHead(tl.shape[-1], tp.shape[-1], hidden_dim=int(args.hidden_dim)).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    TL = torch.from_numpy(tl).float(); TP = torch.from_numpy(tp).float(); TY = torch.from_numpy(ty).float()
    VL = torch.from_numpy(vl).float().to(device); VP = torch.from_numpy(vp).float().to(device)
    best_sign, best_state = -1.0, None
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(len(TY)); total = 0.0
        for s in range(0, len(perm), int(args.batch_size)):
            idx = perm[s : s + int(args.batch_size)]
            optimizer.zero_grad()
            loss = criterion(model(TL[idx].to(device), TP[idx].to(device)), TY[idx].to(device))
            loss.backward(); optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(VL), 4096):
                preds.append(model(VL[s:s+4096], VP[s:s+4096]).cpu().numpy())
            pv = np.concatenate(preds)
        moving = np.abs(vy) > 0.01
        sign = float(((np.sign(pv) == np.sign(vy)) & moving).sum() / max(1, moving.sum()))
        mae = float(np.abs(pv - vy).mean())
        print(f"epoch {epoch:3d} loss {total/len(TY):.4f} val_sign_acc(moving) {sign:.4f} mae {mae:.4f}", flush=True)
        if sign > best_sign:
            best_sign, best_state = sign, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "go2_beacon_distance_head_v0",
        "model_state_dict": best_state,
        "colors": list(COLORS),
        "latent_dim": int(tl.shape[-1]),
        "proprio_dim": int(tp.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "latent_mean": lat_mean.astype(np.float32), "latent_std": lat_std.astype(np.float32),
        "proprio_mean": pro_mean.astype(np.float32), "proprio_std": pro_std.astype(np.float32),
        "primitives": list(meta["primitives"]),
        "frozen_jepa_checkpoint": str(meta["frozen_jepa_checkpoint"]),
        "image_size": int(meta["image_size"]),
        "best_val_sign_acc": best_sign,
    }, args.output)
    print(json.dumps({"best_val_sign_acc": best_sign, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
