#!/usr/bin/env python3
"""Train a history+action-conditioned per-primitive contact-risk head for Go2.

Inputs per tick: frozen-JEPA latent of the ego frame plus nonprivileged
proprioceptive features, over an H-tick window ending at the current pose.
Output: per-primitive blocked probability (counterfactual after-start swept
body clearance below margin). This is the history-aware replacement for the
single-frame current-body-risk head, which is blind to flank walls.

Offline gate: recall at fixed FPR on the executed-primitive channel of the
held-out validation runs' contact ticks (the ticks the single-frame model
missed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class HistoryRiskHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        proprio_dim: int,
        primitive_count: int,
        hidden_dim: int = 192,
        latent_proj_dim: int = 96,
    ) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, latent_proj_dim)
        self.gru = nn.GRU(latent_proj_dim + proprio_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, primitive_count),
        )

    def forward(self, latents: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        projected = self.latent_proj(latents)
        output, _ = self.gru(torch.cat([projected, proprio], dim=-1))
        return self.head(output[:, -1])


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
    if len(neg_scores) == 0 or (labels == 1).sum() == 0:
        return {"threshold": None, "recall": None}
    k = max(0, min(len(neg_scores) - 1, int(np.floor(fpr * len(neg_scores)))))
    threshold = float(neg_scores[k])
    return {"threshold": threshold, "recall": float(np.mean(scores[labels == 1] > threshold))}


class RunData:
    def __init__(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.ticks = data["ticks"]
        self.latents = data["latents"]
        self.proprio = data["proprio"]
        self.clearance = data["clearance"]
        self.progress = data["progress"]
        self.executed = [str(p) for p in data["executed_primitive"]]
        self.contact = data["contact_label"]
        self.tag = path.stem

    def windows(self, window: int) -> list[int]:
        # window end indices with contiguous tick history
        out = []
        for end in range(window - 1, len(self.ticks)):
            if self.ticks[end] - self.ticks[end - window + 1] == window - 1:
                out.append(end)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--val-tags", default="val_attempt407,val_attempt445")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--blocked-margin-m", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--latent-proj-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
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
    summary = json.loads((args.dataset_dir / "summary.json").read_text())
    vocab = list(summary["primitive_vocab"])
    latent_dim = int(summary["latent_dim"])
    proprio_dim = int(summary["proprio_feature_dim"])
    val_tags = {t.strip() for t in str(args.val_tags).split(",") if t.strip()}

    runs = [RunData(p) for p in sorted(args.dataset_dir.glob("*.npz"))]
    train_runs = [r for r in runs if r.tag not in val_tags]
    val_runs = [r for r in runs if r.tag in val_tags]
    if not train_runs or not val_runs:
        raise SystemExit(f"missing splits: train={len(train_runs)} val={len(val_runs)}")

    window = int(args.window)
    margin = float(args.blocked_margin_m)

    # proprio normalization from train runs
    proprio_all = np.concatenate([r.proprio for r in train_runs])
    proprio_mean = proprio_all.mean(axis=0)
    proprio_std = proprio_all.std(axis=0) + 1e-6
    latent_all = np.concatenate([r.latents[:512] for r in train_runs])
    latent_mean = latent_all.mean(axis=0)
    latent_std = latent_all.std(axis=0) + 1e-6

    def _assemble(run: RunData, ends: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lat = (run.latents - latent_mean) / latent_std
        pro = (run.proprio - proprio_mean) / proprio_std
        lat_w = np.stack([lat[e - window + 1 : e + 1] for e in ends])
        pro_w = np.stack([pro[e - window + 1 : e + 1] for e in ends])
        labels = (run.clearance[ends] < margin).astype(np.float32)
        return (
            torch.from_numpy(lat_w).float(),
            torch.from_numpy(pro_w).float(),
            torch.from_numpy(labels).float(),
        )

    train_items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for run in train_runs:
        ends = run.windows(window)
        if ends:
            train_items.append(_assemble(run, ends))
    train_lat = torch.cat([x[0] for x in train_items])
    train_pro = torch.cat([x[1] for x in train_items])
    train_lab = torch.cat([x[2] for x in train_items])

    val_items = []
    for run in val_runs:
        ends = run.windows(window)
        if not ends:
            continue
        lat, pro, lab = _assemble(run, ends)
        executed = [run.executed[e] for e in ends]
        contact = run.contact[ends]
        val_items.append((run.tag, lat, pro, lab, executed, contact))

    model = HistoryRiskHead(
        latent_dim,
        proprio_dim,
        len(vocab),
        hidden_dim=int(args.hidden_dim),
        latent_proj_dim=int(args.latent_proj_dim),
    ).to(device)
    positives = float(train_lab.sum())
    pos_weight = torch.full(
        (len(vocab),),
        max(1.0, (train_lab.numel() - positives) / max(1.0, positives)),
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    best_gate, best_state = -1.0, None
    history = []
    prim_index = {name: idx for idx, name in enumerate(vocab)}
    for epoch in range(int(args.epochs)):
        model.train()
        perm = torch.randperm(len(train_lat))
        total = 0.0
        for start in range(0, len(perm), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            optimizer.zero_grad()
            logits = model(train_lat[idx].to(device), train_pro[idx].to(device))
            loss = criterion(logits, train_lab[idx].to(device))
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        model.eval()
        # gate: executed-primitive channel AUC on val contact ticks
        exec_scores, exec_labels = [], []
        with torch.no_grad():
            for _tag, lat, pro, lab, executed, contact in val_items:
                for start in range(0, len(lat), 512):
                    logits = model(lat[start : start + 512].to(device), pro[start : start + 512].to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    for row_offset, prim in enumerate(executed[start : start + 512]):
                        prim_idx = prim_index.get(prim)
                        if prim_idx is None:
                            continue
                        exec_scores.append(float(probs[row_offset, prim_idx]))
                        exec_labels.append(int(contact[start + row_offset]))
        exec_scores_arr = np.asarray(exec_scores)
        exec_labels_arr = np.asarray(exec_labels)
        gate_auc = _auc(exec_scores_arr, exec_labels_arr)
        history.append({"epoch": epoch, "loss": total / len(train_lat), "val_exec_contact_auc": gate_auc})
        print(f"epoch {epoch:3d} loss {total/len(train_lat):.4f} val_exec_contact_auc {gate_auc:.4f}", flush=True)
        if gate_auc > best_gate:
            best_gate = gate_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    report: dict = {
        "schema": "go2_history_risk_head_report_v0",
        "primitive_vocab": vocab,
        "window": window,
        "blocked_margin_m": margin,
        "best_val_exec_contact_auc": best_gate,
        "history": history,
        "per_run": {},
    }
    with torch.no_grad():
        for tag, lat, pro, lab, executed, contact in val_items:
            probs_all = []
            for start in range(0, len(lat), 512):
                logits = model(lat[start : start + 512].to(device), pro[start : start + 512].to(device))
                probs_all.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate(probs_all)
            exec_probs = np.asarray(
                [
                    probs[i, prim_index[p]] if p in prim_index else 0.0
                    for i, p in enumerate(executed)
                ]
            )
            contact_arr = np.asarray(contact)
            entry = {
                "rows": int(len(exec_probs)),
                "contact_ticks": int(contact_arr.sum()),
                "exec_contact_auc": _auc(exec_probs, contact_arr),
                "exec_contact_recall_at_fpr": {
                    f"{fpr:.3f}": _recall_at_fpr(exec_probs, contact_arr, fpr)
                    for fpr in (0.02, 0.05, 0.10)
                },
                "label_auc_per_primitive": {},
            }
            lab_arr = lab.numpy()
            for prim, prim_idx in prim_index.items():
                entry["label_auc_per_primitive"][prim] = _auc(
                    probs[:, prim_idx], lab_arr[:, prim_idx].astype(np.int64)
                )
            report["per_run"][tag] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "go2_history_risk_head_v0",
            "model_state_dict": best_state,
            "primitive_vocab": vocab,
            "latent_dim": latent_dim,
            "proprio_dim": proprio_dim,
            "hidden_dim": int(args.hidden_dim),
            "latent_proj_dim": int(args.latent_proj_dim),
            "window": window,
            "blocked_margin_m": margin,
            "latent_mean": latent_mean.astype(np.float32),
            "latent_std": latent_std.astype(np.float32),
            "proprio_mean": proprio_mean.astype(np.float32),
            "proprio_std": proprio_std.astype(np.float32),
            "frozen_jepa_checkpoint": str(summary["frozen_jepa_checkpoint"]),
            "image_size": int(summary["image_size"]),
        },
        args.output,
    )
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("best_val_exec_contact_auc", "per_run")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
