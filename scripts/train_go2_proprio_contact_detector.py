#!/usr/bin/env python3
"""Train a proprioceptive contact_now detector for Go2 closed-loop escapes.

Consumes the NPZ built by build_go2_proprio_contact_dataset.py. Inputs are
nonprivileged proprioceptive windows; labels were privileged offline occupancy
checks. Reports per-tick AUC/recall@FPR plus a run-level trigger evaluation
(rolling-mean score over 3 ticks) on the validation attempts, which is the
metric the escape wiring actually depends on.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ProprioContactDetector(nn.Module):
    def __init__(self, window: int, feature_dim: int, hidden_dim: int = 128, arch: str = "mlp") -> None:
        super().__init__()
        self.arch = arch
        self.window = window
        self.feature_dim = feature_dim
        if arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(window * feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        elif arch == "gru":
            self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"unknown arch {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)
        output, _ = self.gru(x)
        return self.head(output[:, -1]).squeeze(-1)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> dict:
    neg_scores = np.sort(scores[labels == 0])[::-1]
    if len(neg_scores) == 0:
        return {"threshold": None, "recall": None}
    k = max(0, min(len(neg_scores) - 1, int(np.floor(fpr * len(neg_scores)))))
    threshold = float(neg_scores[k])
    recall = float(np.mean(scores[labels == 1] > threshold)) if (labels == 1).any() else float("nan")
    return {"threshold": threshold, "recall": recall}


def _run_level_eval(
    scores: np.ndarray,
    labels: np.ndarray,
    meta: list[dict],
    thresholds: list[float],
) -> dict:
    by_file: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(meta):
        by_file[row["file"]].append(idx)
    report: dict = {}
    for fname, idxs in by_file.items():
        idxs.sort(key=lambda i: meta[i]["tick"])
        ticks = np.asarray([meta[i]["tick"] for i in idxs])
        yy = labels[idxs]
        rolling = np.convolve(scores[idxs], np.ones(3) / 3, mode="same")
        runs: list[list[int]] = []
        current: list[int] = []
        for j in range(len(yy)):
            if yy[j]:
                if current and ticks[j] != ticks[current[-1]] + 1:
                    runs.append(current)
                    current = []
                current.append(j)
            else:
                if current:
                    runs.append(current)
                    current = []
        if current:
            runs.append(current)
        contact_neighborhood = np.zeros(len(yy), dtype=bool)
        for j in range(len(yy)):
            lo, hi = max(0, j - 2), min(len(yy), j + 3)
            contact_neighborhood[j] = bool(yy[lo:hi].any())
        per_threshold = {}
        for threshold in thresholds:
            long_runs = [run for run in runs if len(run) >= 3]
            detected = sum(1 for run in long_runs if (rolling[run] > threshold).any())
            false_ticks = int(((rolling > threshold) & ~contact_neighborhood).sum())
            per_threshold[f"{threshold:.3f}"] = {
                "long_runs_detected": detected,
                "long_runs_total": len(long_runs),
                "all_runs_detected": sum(1 for run in runs if (rolling[run] > threshold).any()),
                "all_runs_total": len(runs),
                "false_trigger_ticks": false_ticks,
                "clear_ticks": int((~contact_neighborhood).sum()),
            }
        report[fname] = per_threshold
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=("mlp", "gru"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device("cpu" if args.device == "auto" else args.device)
    )

    data = np.load(args.dataset, allow_pickle=True)
    train_X = data["train_X"].astype(np.float32)
    train_y = data["train_y"].astype(np.float32)
    val_X = data["val_X"].astype(np.float32)
    val_y = data["val_y"].astype(np.float32)
    val_meta = [json.loads(str(item)) for item in data["val_meta"]]
    window, feature_dim = train_X.shape[1], train_X.shape[2]

    mean = train_X.reshape(-1, feature_dim).mean(axis=0)
    std = train_X.reshape(-1, feature_dim).std(axis=0) + 1e-6
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std

    model = ProprioContactDetector(window, feature_dim, hidden_dim=int(args.hidden_dim), arch=args.arch).to(device)
    pos_weight = torch.tensor([(train_y == 0).sum() / max(1.0, (train_y == 1).sum())], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    train_Xt = torch.from_numpy(train_X).float()
    train_yt = torch.from_numpy(train_y).float()
    val_Xt = torch.from_numpy(val_X).float().to(device)
    best_auc, best_state = -1.0, None
    history = []
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(len(train_Xt))
        total_loss = 0.0
        for start in range(0, len(perm), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            xb = train_Xt[idx].to(device)
            yb = train_yt[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_Xt)).cpu().numpy()
        auc = _auc(val_scores, val_y.astype(np.int64))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_Xt), "val_auc": auc})
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} loss {total_loss/len(train_Xt):.4f} val_auc {auc:.4f}", flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(model(val_Xt)).cpu().numpy()

    labels = val_y.astype(np.int64)
    recall_at_fpr = {
        f"{fpr:.3f}": _recall_at_fpr(val_scores, labels, fpr) for fpr in (0.01, 0.02, 0.05, 0.10)
    }
    thresholds = sorted(
        {
            round(entry["threshold"], 3)
            for entry in recall_at_fpr.values()
            if entry["threshold"] is not None
        }
        | {0.5, 0.7, 0.9}
    )
    report: dict = {
        "schema": "go2_proprio_contact_detector_report_v1",
        "arch": args.arch,
        "val_auc": _auc(val_scores, labels),
        "recall_at_fpr": recall_at_fpr,
        "run_level": _run_level_eval(val_scores, labels, val_meta, thresholds),
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "go2_proprio_contact_detector_v1",
            "arch": args.arch,
            "model_state_dict": best_state,
            "window": int(window),
            "feature_dim": int(feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "threshold_fpr05": recall_at_fpr["0.050"]["threshold"],
        },
        args.output,
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("arch", "val_auc", "recall_at_fpr", "run_level")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
