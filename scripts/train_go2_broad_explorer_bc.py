#!/usr/bin/env python3
"""Train the broad explorer BC head on corpus teacher/frontier sequences.

GRU over W ticks of [frozen-JEPA latent ⊕ proprio] → next-primitive logits.
Inputs match the runtime history buffers exactly (same feature contract as
the history-risk model); the label is the primitive executed on the NEXT
tick, so the head learns movement selection, not input copying.
Validation is scene-held-out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class BroadExplorerBC(nn.Module):
    def __init__(self, latent_dim: int, proprio_dim: int, primitive_count: int,
                 hidden_dim: int = 192, latent_proj_dim: int = 96) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, latent_proj_dim)
        self.gru = nn.GRU(latent_proj_dim + proprio_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, primitive_count)
        )

    def forward(self, latents: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(torch.cat([self.latent_proj(latents), proprio], dim=-1))
        return self.head(out[:, -1])


def _windows(data: dict, window: int, sources: set[str]) -> list[int]:
    # frame_index strides by the env count within one env's stream
    # (frame_index = step * n_envs + env), so infer the stride per shard.
    ticks = data["ticks"]; srcs = data["sources"]
    diffs = np.diff(ticks)
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return []
    stride = int(np.bincount(positive.astype(np.int64)).argmax())
    if stride <= 0:
        return []
    out = []
    for end in range(window - 1, len(ticks) - 1):
        if ticks[end + 1] - ticks[end] != stride:
            continue
        if ticks[end] - ticks[end - window + 1] != (window - 1) * stride:
            continue
        if str(srcs[end + 1]) not in sources:
            continue
        out.append(end)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--sources", default="route_teacher,frontier")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--latent-proj-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-windows-per-scene", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    meta = json.loads((args.dataset_dir / "meta.json").read_text())
    vocab = list(meta["primitives"])
    sources = {s.strip() for s in str(args.sources).split(",") if s.strip()}
    shards = sorted(args.dataset_dir.glob("*.npz"))
    rng = np.random.RandomState(int(args.seed))
    rng.shuffle(shards)
    n_val = max(1, int(len(shards) * float(args.val_fraction)))
    val_shards, train_shards = shards[:n_val], shards[n_val:]

    def _load(shard_list):
        lat_w, pro_w, ys = [], [], []
        for path in shard_list:
            d = np.load(path, allow_pickle=True)
            data = {k: d[k] for k in ("latents", "proprio", "labels", "sources", "ticks")}
            ends = _windows(data, int(args.window), sources)
            if not ends:
                continue
            if len(ends) > int(args.max_windows_per_scene):
                ends = list(rng.choice(ends, int(args.max_windows_per_scene), replace=False))
            W = int(args.window)
            lat_w.append(np.stack([data["latents"][e - W + 1 : e + 1] for e in ends]))
            pro_w.append(np.stack([data["proprio"][e - W + 1 : e + 1] for e in ends]))
            ys.append(np.asarray([data["labels"][e + 1] for e in ends]))
        return (np.concatenate(lat_w), np.concatenate(pro_w), np.concatenate(ys))

    train_lat, train_pro, train_y = _load(train_shards)
    val_lat, val_pro, val_y = _load(val_shards)
    lat_mean = train_lat.reshape(-1, train_lat.shape[-1]).mean(0)
    lat_std = train_lat.reshape(-1, train_lat.shape[-1]).std(0) + 1e-6
    pro_mean = train_pro.reshape(-1, train_pro.shape[-1]).mean(0)
    pro_std = train_pro.reshape(-1, train_pro.shape[-1]).std(0) + 1e-6
    train_lat = (train_lat - lat_mean) / lat_std; val_lat = (val_lat - lat_mean) / lat_std
    train_pro = (train_pro - pro_mean) / pro_std; val_pro = (val_pro - pro_mean) / pro_std
    print(f"train windows {len(train_y)} ({len(train_shards)} scenes) | val {len(val_y)} ({len(val_shards)} scenes)", flush=True)

    counts = np.bincount(train_y, minlength=len(vocab)).astype(np.float64)
    weights = (counts.sum() / np.maximum(counts, 1.0)) ** 0.5
    model = BroadExplorerBC(train_lat.shape[-1], train_pro.shape[-1], len(vocab),
                            hidden_dim=int(args.hidden_dim), latent_proj_dim=int(args.latent_proj_dim)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    tl = torch.from_numpy(train_lat).float(); tp = torch.from_numpy(train_pro).float()
    ty = torch.from_numpy(train_y).long()
    vl = torch.from_numpy(val_lat).float().to(device); vp = torch.from_numpy(val_pro).float().to(device)
    vy = torch.from_numpy(val_y).long()
    best_acc, best_state = -1.0, None
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(len(ty))
        total = 0.0
        for s in range(0, len(perm), int(args.batch_size)):
            idx = perm[s : s + int(args.batch_size)]
            optimizer.zero_grad()
            loss = criterion(model(tl[idx].to(device), tp[idx].to(device)), ty[idx].to(device))
            loss.backward(); optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, len(vl), 1024):
                preds.append(model(vl[s : s + 1024], vp[s : s + 1024]).argmax(-1).cpu())
        preds = torch.cat(preds)
        acc = float((preds == vy).float().mean())
        fwd_share = float((preds <= 2).float().mean())
        print(f"epoch {epoch:3d} loss {total/len(ty):.4f} val_acc {acc:.4f} fwd_share {fwd_share:.3f}", flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "go2_broad_explorer_bc_v0",
        "model_state_dict": best_state,
        "primitive_vocab": vocab,
        "latent_dim": int(train_lat.shape[-1]),
        "proprio_dim": int(train_pro.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "latent_proj_dim": int(args.latent_proj_dim),
        "window": int(args.window),
        "latent_mean": lat_mean.astype(np.float32), "latent_std": lat_std.astype(np.float32),
        "proprio_mean": pro_mean.astype(np.float32), "proprio_std": pro_std.astype(np.float32),
        "frozen_jepa_checkpoint": str(meta["frozen_jepa_checkpoint"]),
        "image_size": int(meta["image_size"]),
        "best_val_acc": best_acc,
    }, args.output)
    print(json.dumps({"best_val_acc": best_acc, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
