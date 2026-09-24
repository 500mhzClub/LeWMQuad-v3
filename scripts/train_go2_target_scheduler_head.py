#!/usr/bin/env python3
"""Train a learned target scheduler from closed-loop Go2 result logs."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FEATURE_SCHEMA = "lewm_go2_target_scheduler_features_v1"
CHECKPOINT_SCHEMA = "lewm_go2_target_scheduler_head_v0"
DEFAULT_COLORS = ("red", "yellow", "blue", "green")


class TargetSchedulerHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, color_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(color_count)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--colors", default=",".join(DEFAULT_COLORS))
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--source-weight", action="append", default=[],
                        help="Optional path_suffix=weight multiplier.")
    parser.add_argument(
        "--relabel-stale-yellow-to-blue-after",
        type=int,
        default=0,
        help=(
            "When >0, relabel stale yellow rows to blue after this many active "
            "target ticks if red/green are already claimed and blue has strong memory."
        ),
    )
    parser.add_argument("--relabel-stale-blue-min-conf", type=float, default=0.25)
    parser.add_argument("--relabel-stale-blue-min-read-score", type=float, default=0.80)
    parser.add_argument("--relabel-stale-weight-multiplier", type=float, default=3.0)
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

    source_weights = _parse_source_weights(args.source_weight)
    features, labels, weights, meta = _load_examples(
        args.results,
        colors=colors,
        source_weights=source_weights,
        relabel_stale_yellow_to_blue_after=int(args.relabel_stale_yellow_to_blue_after),
        relabel_stale_blue_min_conf=float(args.relabel_stale_blue_min_conf),
        relabel_stale_blue_min_read_score=float(args.relabel_stale_blue_min_read_score),
        relabel_stale_weight_multiplier=float(args.relabel_stale_weight_multiplier),
    )
    if features.shape[0] < 2:
        raise SystemExit("not enough scheduler examples")

    train_idx, val_idx = _split_indices(
        features.shape[0],
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    x_train = torch.from_numpy(features[train_idx]).float().to(device)
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)
    w_train = torch.from_numpy(weights[train_idx]).float().to(device)
    x_val = torch.from_numpy(features[val_idx]).float().to(device)
    y_val = torch.from_numpy(labels[val_idx]).long().to(device)

    model = TargetSchedulerHead(
        input_dim=int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        color_count=len(colors),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    class_weights = _class_weights(labels[train_idx], color_count=len(colors)).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        logits = model(x_train)
        loss_rows = F.cross_entropy(logits, y_train, weight=class_weights, reduction="none")
        loss = (loss_rows * w_train).sum() / w_train.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(args.epochs) or epoch % 25 == 0:
            metrics = _eval(model, x_val, y_val, colors=colors)
            metrics["epoch"] = int(epoch)
            metrics["loss"] = float(loss.detach().cpu())
            history.append(metrics)
            if metrics["macro_f1"] >= best_score:
                best_score = float(metrics["macro_f1"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"epoch={epoch} loss={float(loss.detach().cpu()):.4f} "
                f"val_acc={metrics['accuracy']:.3f} val_macro_f1={metrics['macro_f1']:.3f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = _eval(model, x_val, y_val, colors=colors)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "color_vocab": list(colors),
        "model_state_dict": model.state_dict(),
        "feature_names": _feature_names(colors),
        "source_results": [str(path) for path in args.results],
        "relabel_stale_yellow_to_blue_after": int(args.relabel_stale_yellow_to_blue_after),
        "relabel_stale_blue_min_conf": float(args.relabel_stale_blue_min_conf),
        "relabel_stale_blue_min_read_score": float(args.relabel_stale_blue_min_read_score),
        "relabel_stale_weight_multiplier": float(args.relabel_stale_weight_multiplier),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_target_scheduler_head_report_v0",
        "output": str(args.output),
        "results": [str(path) for path in args.results],
        "examples": int(features.shape[0]),
        "label_counts": {
            colors[int(k)]: int(v)
            for k, v in sorted(Counter(int(v) for v in labels.tolist()).items())
        },
        "colors": list(colors),
        "validation": final_metrics,
        "history": history,
        "source_weights": source_weights,
        "relabel_stale_yellow_to_blue_after": int(args.relabel_stale_yellow_to_blue_after),
        "relabel_stale_blue_min_conf": float(args.relabel_stale_blue_min_conf),
        "relabel_stale_blue_min_read_score": float(args.relabel_stale_blue_min_read_score),
        "relabel_stale_weight_multiplier": float(args.relabel_stale_weight_multiplier),
        "sample_meta": meta[:12],
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"target_scheduler_head: output={args.output} examples={features.shape[0]} "
        f"val_macro_f1={final_metrics['macro_f1']:.3f}",
        flush=True,
    )
    return 0


def _load_examples(
    paths: list[Path],
    *,
    colors: tuple[str, ...],
    source_weights: dict[str, float],
    relabel_stale_yellow_to_blue_after: int,
    relabel_stale_blue_min_conf: float,
    relabel_stale_blue_min_read_score: float,
    relabel_stale_weight_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    ws: list[float] = []
    meta: list[dict[str, Any]] = []
    color_to_idx = {color: idx for idx, color in enumerate(colors)}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload.get("result", payload)
        log = payload.get("log", [])
        if not isinstance(log, list):
            continue
        max_ticks = int(result.get("ticks_used") or len(log) or 1)
        claimed_so_far: set[str] = set()
        previous_active = colors[0]
        active_since_tick = 0
        source_weight = _source_weight(path, source_weights)
        for row in log:
            if not isinstance(row, dict):
                continue
            color = str(row.get("target_color", "")).lower()
            if color not in color_to_idx:
                continue
            tick = int(row.get("tick", 0))
            if str(row.get("state", "")).upper() == "CLAIM":
                claimed_so_far.add(color)
                next_active = _next_unclaimed(colors, claimed_so_far, current=color)
                if next_active != previous_active:
                    active_since_tick = int(tick) + 1
                previous_active = next_active
                continue
            readouts = _normalise_readouts(row.get("color_readouts"))
            if not readouts:
                continue
            switch = row.get("target_switch")
            current_color = previous_active
            if isinstance(switch, dict) and str(switch.get("from", "")).lower() in color_to_idx:
                current_color = str(switch.get("from")).lower()
            current_target_age_ticks = max(0, int(tick) - int(active_since_tick))
            feature = _feature(
                colors=colors,
                current_color=current_color,
                current_target_age_ticks=current_target_age_ticks,
                claimed_colors=claimed_so_far,
                color_readouts=readouts,
                tick=tick,
                max_ticks=max_ticks,
            )
            label_color = color
            example_weight = float(source_weight)
            relabeled = False
            if (
                int(relabel_stale_yellow_to_blue_after) > 0
                and current_color in {"yellow", "blue"}
                and "blue" in color_to_idx
                and "red" in claimed_so_far
                and "green" in claimed_so_far
                and "blue" not in claimed_so_far
                and (
                    current_color == "blue"
                    or current_target_age_ticks >= int(relabel_stale_yellow_to_blue_after)
                )
            ):
                blue_readout = readouts.get("blue", {})
                blue_conf = _float(blue_readout.get("mem_conf"), 0.0)
                blue_read_score = _float(blue_readout.get("read_score"), 0.0)
                if (
                    blue_conf >= float(relabel_stale_blue_min_conf)
                    and blue_read_score >= float(relabel_stale_blue_min_read_score)
                ):
                    label_color = "blue"
                    example_weight *= float(relabel_stale_weight_multiplier)
                    relabeled = True
            xs.append(feature)
            ys.append(color_to_idx[label_color])
            ws.append(example_weight)
            if len(meta) < 96:
                meta.append(
                    {
                        "path": str(path),
                        "tick": tick,
                        "current_color": current_color,
                        "current_target_age_ticks": current_target_age_ticks,
                        "label_color": label_color,
                        "original_label_color": color,
                        "relabeled": relabeled,
                        "claimed": sorted(claimed_so_far),
                        "state": row.get("state"),
                    }
                )
            if relabeled:
                continue
            if label_color != previous_active:
                active_since_tick = int(tick)
            previous_active = label_color
    if not xs:
        return (
            np.zeros((0, len(_feature_names(colors))), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            meta,
        )
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ws, dtype=np.float32),
        meta,
    )


def _feature(
    *,
    colors: tuple[str, ...],
    current_color: str,
    current_target_age_ticks: int,
    claimed_colors: set[str],
    color_readouts: dict[str, dict[str, Any]],
    tick: int,
    max_ticks: int,
) -> np.ndarray:
    current = str(current_color).lower()
    claimed = {str(item).lower() for item in claimed_colors}
    out: list[float] = [
        min(1.0, max(0.0, float(tick) / max(1.0, float(max_ticks)))),
        min(1.0, max(0.0, float(len(claimed)) / max(1.0, float(len(colors))))),
        min(1.0, max(0.0, float(current_target_age_ticks) / 256.0)),
    ]
    for color in colors:
        readout = color_readouts.get(color, {})
        first_seen_tick = readout.get("first_seen_tick")
        last_seen_tick = readout.get("last_seen_tick")
        first_age = (
            9999.0
            if first_seen_tick is None
            else max(0.0, float(tick) - _float(first_seen_tick, float(tick)))
        )
        last_age = (
            9999.0
            if last_seen_tick is None
            else max(0.0, float(tick) - _float(last_seen_tick, float(tick)))
        )
        out.extend(
            [
                1.0 if color == current else 0.0,
                1.0 if color in claimed or bool(readout.get("claimed", False)) else 0.0,
                _float(readout.get("mem_conf"), 0.0),
                _float(readout.get("area"), -99.0),
                _float(readout.get("read_score"), 0.0),
                1.0 if bool(readout.get("read_gate_pass", False)) else 0.0,
                min(1.0, first_age / 128.0),
                min(1.0, last_age / 128.0),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def _feature_names(colors: tuple[str, ...]) -> list[str]:
    names = ["tick_frac", "claimed_count_frac", "current_target_age_frac"]
    for color in colors:
        names.extend(
            [
                f"{color}_is_current",
                f"{color}_claimed",
                f"{color}_mem_conf",
                f"{color}_area",
                f"{color}_read_score",
                f"{color}_read_gate_pass",
                f"{color}_first_seen_age",
                f"{color}_last_seen_age",
            ]
        )
    return names


def _normalise_readouts(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, row in value.items():
        if isinstance(row, dict):
            out[str(key).lower()] = row
    return out


def _next_unclaimed(colors: tuple[str, ...], claimed: set[str], *, current: str) -> str:
    try:
        start = colors.index(str(current).lower()) + 1
    except ValueError:
        start = 0
    for idx in range(start, len(colors)):
        if colors[idx] not in claimed:
            return colors[idx]
    for color in colors:
        if color not in claimed:
            return color
    return colors[0]


def _split_indices(n: int, *, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(n))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    val_n = max(1, int(round(float(validation_fraction) * int(n))))
    val_n = min(max(1, val_n), max(1, int(n) - 1))
    return idx[val_n:], idx[:val_n]


def _class_weights(labels: np.ndarray, *, color_count: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=int(color_count)).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (counts * float(color_count))
    return torch.from_numpy(weights.astype(np.float32))


def _eval(
    model: TargetSchedulerHead,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    colors: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)
    pred_np = pred.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    accuracy = float((pred_np == y_np).mean()) if y_np.size else 0.0
    per_color: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for idx, color in enumerate(colors):
        tp = int(((pred_np == idx) & (y_np == idx)).sum())
        fp = int(((pred_np == idx) & (y_np != idx)).sum())
        fn = int(((pred_np != idx) & (y_np == idx)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        f1s.append(float(f1))
        per_color[color] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((y_np == idx).sum()),
        }
    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1s) / max(1, len(f1s))),
        "per_color": per_color,
    }


def _parse_source_weights(specs: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit("--source-weight entries must be path_suffix=weight")
        key, value = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--source-weight path suffix must not be empty")
        out[key] = float(value)
    return out


def _source_weight(path: Path, source_weights: dict[str, float]) -> float:
    text = str(path)
    weight = 1.0
    for suffix, value in source_weights.items():
        if text.endswith(suffix):
            weight = float(value)
    return weight


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


if __name__ == "__main__":
    raise SystemExit(main())
