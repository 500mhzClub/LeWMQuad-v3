#!/usr/bin/env python3
"""Retrain the Go2 JEPA encoder with geometric auxiliary supervision.

The frozen contrastive JEPA latent does not transfer local wall geometry
across scenes (heads reach ~0.9 train / ~0.6 unseen-scene val even with
on-distribution data and exact labels — see
docs/lewm_go2_fully_learned_tier_execution_2026-07-01.md). This trainer keeps
the Go2JepaEncoder architecture and checkpoint format (drop-in for
load_go2_jepa_encoder) but fine-tunes it end-to-end against counterfactual
per-primitive blocked/progress/clearance targets computed from scene manifest
grids. Grid labels are privileged training-time supervision only; the runtime
input stays egocentric RGB through the (re-)frozen encoder.

The auxiliary geometric head trained here is discarded; deployable heads are
trained afterwards with train_go2_jepa_primitive_outcome_predictor.py on the
frozen retrained encoder so gate metrics stay apples-to-apples with v219.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_genesis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_worlds"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.models.go2_jepa import Go2JepaEncoder, load_go2_jepa_encoder  # noqa: E402
from train_go2_hidden_target_memory_probe import _load_image, _resolve_device  # noqa: E402
from train_go2_jepa_primitive_outcome_predictor import (  # noqa: E402
    _load_counterfactual_body_clearance_rows,
)

PRIMITIVE_VOCAB = [
    "forward_medium",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
]


class GeometricAuxHead(nn.Module):
    """Per-primitive blocked/progress/clearance predictions from one latent."""

    def __init__(self, latent_dim: int, hidden_dim: int, primitive_count: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.blocked = nn.Linear(int(hidden_dim), int(primitive_count))
        self.progress = nn.Linear(int(hidden_dim), int(primitive_count))
        self.clearance = nn.Linear(int(hidden_dim), int(primitive_count))

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(latent)
        return self.blocked(features), self.progress(features), self.clearance(features)


def _frame_targets(paths: list[Path], examples) -> tuple[list[Path], torch.Tensor, torch.Tensor, torch.Tensor]:
    by_frame: dict[Path, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    for item in examples:
        by_frame[item.rgb_path][item.primitive] = (
            float(item.blocked),
            float(item.progress_m),
            float(min(item.clearance_m, 0.5)),
        )
    frames: list[Path] = []
    blocked_rows: list[list[float]] = []
    progress_rows: list[list[float]] = []
    clearance_rows: list[list[float]] = []
    for path in paths:
        per_primitive = by_frame.get(path)
        if per_primitive is None or any(name not in per_primitive for name in PRIMITIVE_VOCAB):
            continue
        frames.append(path)
        blocked_rows.append([per_primitive[name][0] for name in PRIMITIVE_VOCAB])
        progress_rows.append([per_primitive[name][1] for name in PRIMITIVE_VOCAB])
        clearance_rows.append([per_primitive[name][2] for name in PRIMITIVE_VOCAB])
    return (
        frames,
        torch.tensor(blocked_rows, dtype=torch.float32),
        torch.tensor(progress_rows, dtype=torch.float32),
        torch.tensor(clearance_rows, dtype=torch.float32),
    )


def _load_split(paths: list[Path], args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    examples = _load_counterfactual_body_clearance_rows(
        paths,
        primitive_vocab=PRIMITIVE_VOCAB,
        primitive_registry=args.primitive_registry,
        body_forward_m=float(args.body_forward_m),
        body_half_width_m=float(args.body_half_width_m),
        body_clearance_margin_m=float(args.body_clearance_margin_m),
        body_clearance_label_target=str(args.body_clearance_label_target),
        body_clearance_source=str(args.body_clearance_source),
        cell_size_m=float(args.cell_size_m),
        inflation_m=float(args.inflation_m),
        max_source_rows=int(args.max_source_rows),
    )
    unique_paths = list(dict.fromkeys(item.rgb_path for item in examples))
    frames, blocked, progress, clearance = _frame_targets(unique_paths, examples)
    scenes = sorted({item.scene_id for item in examples})
    images = torch.stack(
        [_load_image(path, image_size=int(args.image_size)) for path in frames]
    )
    return images, blocked, progress, clearance, scenes


def _metrics(blocked_logits: torch.Tensor, blocked: torch.Tensor, *, threshold: float) -> dict:
    probs = torch.sigmoid(blocked_logits)
    pred = probs >= float(threshold)
    target = blocked >= 0.5
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    tn = int((~pred & ~target).sum())
    fn = int((~pred & target).sum())
    total = max(1, int(target.numel()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    pos = probs[target].reshape(-1)
    neg = probs[~target].reshape(-1)
    if len(pos) and len(neg):
        order = torch.argsort(torch.cat([pos, neg]))
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float32)
        auc = float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
    else:
        auc = None
    balanced = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    return {
        "count": total,
        "positive_rate": float(target.float().mean()),
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall),
        "auc": auc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional Go2 JEPA encoder checkpoint to initialize from.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--head-hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--progress-loss-weight", type=float, default=2.0)
    parser.add_argument("--clearance-loss-weight", type=float, default=1.0)
    parser.add_argument("--augment-brightness", type=float, default=0.15,
                        help="Uniform brightness jitter amplitude (train only).")
    parser.add_argument("--augment-noise", type=float, default=0.02,
                        help="Gaussian pixel noise sigma (train only).")
    parser.add_argument("--primitive-registry", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/go2_primitive_registry.yaml")
    parser.add_argument("--body-forward-m", type=float, default=0.40)
    parser.add_argument("--body-half-width-m", type=float, default=0.24)
    parser.add_argument("--body-clearance-margin-m", type=float, default=0.02)
    parser.add_argument("--body-clearance-source", choices=("configuration", "obstacle"), default="obstacle")
    parser.add_argument("--body-clearance-label-target",
                        choices=("swept_min", "after_start_min", "final"), default="after_start_min")
    parser.add_argument("--cell-size-m", type=float, default=0.05)
    parser.add_argument("--inflation-m", type=float, default=0.12)
    parser.add_argument("--max-source-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = _resolve_device(str(args.device))

    train_images, train_blocked, train_progress, train_clearance, train_scenes = _load_split(
        list(args.datasets), args
    )
    val_images, val_blocked, val_progress, val_clearance, val_scenes = _load_split(
        list(args.validation_datasets), args
    )
    overlap = set(train_scenes) & set(val_scenes)
    if overlap:
        raise SystemExit(f"scene leakage between train and validation: {sorted(overlap)}")
    print(
        f"train frames={len(train_images)} scenes={len(train_scenes)} | "
        f"val frames={len(val_images)} scenes={len(val_scenes)}",
        flush=True,
    )

    encoder = Go2JepaEncoder(latent_dim=int(args.latent_dim))
    if args.init_checkpoint is not None:
        init_encoder, _ = load_go2_jepa_encoder(args.init_checkpoint, device="cpu", freeze=False)
        encoder.load_state_dict(init_encoder.state_dict())
    encoder = encoder.to(device)
    head = GeometricAuxHead(
        latent_dim=int(args.latent_dim),
        hidden_dim=int(args.head_hidden_dim),
        primitive_count=len(PRIMITIVE_VOCAB),
    ).to(device)

    encoder_lr = float(args.encoder_lr) if args.encoder_lr is not None else float(args.lr)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": head.parameters(), "lr": float(args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    val_images_dev = val_images.to(device)
    val_blocked_dev = val_blocked.to(device)

    best_val = None
    best_encoder_state = None
    best_epoch = None
    history = []
    n = len(train_images)
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        head.train()
        order = torch.randperm(n)
        total_loss = 0.0
        batches = 0
        for start in range(0, n, int(args.batch_size)):
            idx = order[start:start + int(args.batch_size)]
            images = train_images[idx].to(device)
            if float(args.augment_brightness) > 0:
                scale = 1.0 + (torch.rand(len(idx), 1, 1, 1, device=device) * 2 - 1) * float(args.augment_brightness)
                images = (images * scale).clamp(0, 1)
            if float(args.augment_noise) > 0:
                images = (images + torch.randn_like(images) * float(args.augment_noise)).clamp(0, 1)
            blocked = train_blocked[idx].to(device)
            progress = train_progress[idx].to(device)
            clearance = train_clearance[idx].to(device)
            blocked_logits, progress_pred, clearance_pred = head(encoder(images))
            loss = (
                F.binary_cross_entropy_with_logits(blocked_logits, blocked)
                + float(args.progress_loss_weight) * F.mse_loss(progress_pred, progress)
                + float(args.clearance_loss_weight) * F.mse_loss(clearance_pred, clearance)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_logits, _, _ = head(encoder(val_images_dev))
            train_logits, _, _ = head(encoder(train_images[: min(n, 4096)].to(device)))
        val_metrics = _metrics(val_logits.cpu(), val_blocked, threshold=float(args.threshold))
        train_metrics = _metrics(
            train_logits.cpu(), train_blocked[: min(n, 4096)], threshold=float(args.threshold)
        )
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(1, batches),
            "train": train_metrics,
            "validation": val_metrics,
        })
        score = val_metrics["balanced_accuracy"]
        if best_val is None or score > best_val:
            best_val = score
            best_epoch = epoch
            best_encoder_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        print(
            f"epoch={epoch} loss={total_loss / max(1, batches):.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_bal={val_metrics['balanced_accuracy']:.3f} "
            f"val_auc={val_metrics['auc']:.3f}" if val_metrics["auc"] is not None else "",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "go2_jepa_geometric_encoder_v0",
            "encoder_state_dict": best_encoder_state,
            "latent_dim": int(args.latent_dim),
            "image_size": int(args.image_size),
            "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
            "best_epoch": best_epoch,
            "best_val_balanced_accuracy": best_val,
            "label_config": {
                "body_clearance_source": str(args.body_clearance_source),
                "body_clearance_label_target": str(args.body_clearance_label_target),
                "body_clearance_margin_m": float(args.body_clearance_margin_m),
            },
            "train_scenes": train_scenes,
            "validation_scenes": val_scenes,
        },
        args.output,
    )
    print(f"saved encoder (best epoch {best_epoch}, val balanced acc {best_val:.3f}) -> {args.output}", flush=True)
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(json.dumps({
            "schema": "go2_jepa_geometric_encoder_report_v0",
            "checkpoint": str(args.output),
            "best_epoch": best_epoch,
            "history": history,
            "train_scenes": train_scenes,
            "validation_scenes": val_scenes,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
