#!/usr/bin/env python3
"""Train a runtime-safe claim-success classifier from closed-loop result logs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_SCHEMA = "lewm_go2_claim_success_head_features_v0"
CHECKPOINT_SCHEMA = "lewm_go2_claim_success_head_v0"
DEFAULT_COLORS = ("green", "yellow", "blue", "red")


class ClaimSuccessHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--success-dist-m", type=float, default=1.2)
    parser.add_argument("--colors", default=",".join(DEFAULT_COLORS))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--positive-weight", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    colors = tuple(item.strip().lower() for item in args.colors.split(",") if item.strip())
    if not colors:
        raise SystemExit("--colors must not be empty")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _device(str(args.device))

    features, labels, meta = _load_examples(
        args.results,
        colors=colors,
        success_dist_m=float(args.success_dist_m),
    )
    if features.shape[0] < 2:
        raise SystemExit("not enough claim-success examples")
    train_idx, val_idx = _split_indices(
        features.shape[0],
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    x_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_train = torch.from_numpy(labels[train_idx]).float().to(device)
    x_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(labels[val_idx]).float().to(device)

    model = ClaimSuccessHead(features.shape[1], hidden_dim=int(args.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    if float(args.positive_weight) > 0:
        pos_weight = torch.tensor(float(args.positive_weight), device=device)
    else:
        positives = max(1, int(y_train.sum().item()))
        negatives = max(1, int(y_train.numel() - positives))
        pos_weight = torch.tensor(float(negatives) / float(positives), device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    report_history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        logits = model(x_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(args.epochs) or epoch % 25 == 0:
            metrics = _eval(model, x_val, y_val, threshold=float(args.threshold))
            metrics["epoch"] = int(epoch)
            metrics["loss"] = float(loss.detach().cpu())
            report_history.append(metrics)
            if metrics["f1"] > best_f1:
                best_f1 = float(metrics["f1"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"epoch={epoch} loss={float(loss.detach().cpu()):.4f} "
                f"val_acc={metrics['accuracy']:.3f} val_f1={metrics['f1']:.3f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = _eval(model, x_val, y_val, threshold=float(args.threshold))
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "color_vocab": list(colors),
        "threshold": float(args.threshold),
        "success_dist_m": float(args.success_dist_m),
        "model_state_dict": model.state_dict(),
        "feature_names": _feature_names(colors),
        "source_results": [str(path) for path in args.results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_claim_success_head_report_v0",
        "output": str(args.output),
        "results": [str(path) for path in args.results],
        "examples": int(features.shape[0]),
        "label_counts": dict(Counter(int(v) for v in labels.tolist())),
        "colors": list(colors),
        "success_dist_m": float(args.success_dist_m),
        "threshold": float(args.threshold),
        "validation": final_metrics,
        "history": report_history,
        "sample_meta": meta[:8],
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"claim_success_head: output={args.output} examples={features.shape[0]} "
        f"val_f1={final_metrics['f1']:.3f}",
        flush=True,
    )
    return 0


def _load_examples(
    paths: list[Path],
    *,
    colors: tuple[str, ...],
    success_dist_m: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", payload)
        log = payload.get("log", [])
        target_xy = result.get("target_xy", {})
        if not isinstance(log, list) or not isinstance(target_xy, dict):
            continue
        claimed_so_far = 0
        for row in log:
            if not isinstance(row, dict):
                continue
            color = str(row.get("target_color", "")).lower()
            if color not in colors:
                continue
            target = target_xy.get(color)
            if not isinstance(target, list) or len(target) < 2:
                continue
            gate = row.get("claim_gate")
            if not isinstance(gate, dict):
                gate = {}
            xy = row.get("post_xy") if isinstance(row.get("post_xy"), list) else row.get("xy")
            dist = _float(row.get("dist_to_target_m"), float("nan"))
            if not np.isfinite(dist):
                if not isinstance(xy, list) or len(xy) < 2:
                    continue
                dist = float(
                    np.hypot(float(xy[0]) - float(target[0]), float(xy[1]) - float(target[1]))
                )
            area = _float(row.get("area", gate.get("area")), -99.0)
            bearing = _float(row.get("bearing", gate.get("bearing")), 0.0)
            mem_conf = _float(row.get("mem_conf", gate.get("mem_conf")), 0.0)
            read_score = _float(row.get("read_score", gate.get("read_score")), 0.0)
            seen_age = _float(row.get("seen_age_ticks"), 0.0)
            feature = _feature(
                color=color,
                colors=colors,
                area=area,
                bearing=bearing,
                mem_conf=mem_conf,
                read_score=read_score,
                seen_age_ticks=seen_age,
                seen=bool(row.get("seen", gate.get("seen", False))),
                in_cone=bool(row.get("in_cone", gate.get("in_cone", False))),
                claimed_count=claimed_so_far,
                tick=_float(row.get("tick"), 0.0),
                max_tick=max(1.0, float(result.get("ticks_used") or len(log) or 1)),
            )
            xs.append(feature)
            ys.append(1 if dist <= float(success_dist_m) else 0)
            if len(meta) < 64:
                meta.append(
                    {
                        "path": str(path),
                        "tick": int(row.get("tick", 0)),
                        "color": color,
                        "dist_m": round(dist, 4),
                        "area": round(area, 4),
                        "bearing": round(bearing, 4),
                        "label": int(dist <= float(success_dist_m)),
                    }
                )
            if _accepted_claim(row, gate):
                claimed_so_far = min(int(claimed_so_far) + 1, len(colors))
    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), meta
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), meta


def _accepted_claim(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    if bool(gate.get("accepted", False)):
        return True
    if row.get("claim_reason") is not None:
        return True
    return str(row.get("state", "")).upper() == "CLAIM" and row.get("dist_to_target_m") is not None


def _feature(
    *,
    color: str,
    colors: tuple[str, ...],
    area: float,
    bearing: float,
    mem_conf: float,
    read_score: float,
    seen_age_ticks: float,
    seen: bool,
    in_cone: bool,
    claimed_count: int,
    tick: float,
    max_tick: float,
) -> np.ndarray:
    onehot = [1.0 if color == item else 0.0 for item in colors]
    return np.asarray(
        [
            *onehot,
            float(area),
            float(bearing),
            abs(float(bearing)),
            float(mem_conf),
            float(read_score),
            min(1.0, max(0.0, float(seen_age_ticks) / 64.0)),
            1.0 if bool(seen) else 0.0,
            1.0 if bool(in_cone) else 0.0,
            min(1.0, max(0.0, float(claimed_count) / max(1.0, float(len(colors))))),
            min(1.0, max(0.0, float(tick) / max(1.0, float(max_tick)))),
        ],
        dtype=np.float32,
    )


def _feature_names(colors: tuple[str, ...]) -> list[str]:
    return [
        *(f"color:{color}" for color in colors),
        "area",
        "bearing",
        "abs_bearing",
        "mem_conf",
        "read_score",
        "seen_age_64",
        "seen",
        "in_cone",
        "claimed_fraction",
        "tick_fraction",
    ]


def _split_indices(n: int, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(n))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    val_n = max(1, min(int(n) - 1, int(round(float(validation_fraction) * int(n)))))
    return indices[val_n:], indices[:val_n]


def _eval(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    pred = prob >= float(threshold)
    truth = y >= 0.5
    tp = int((pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    raise SystemExit(main())
