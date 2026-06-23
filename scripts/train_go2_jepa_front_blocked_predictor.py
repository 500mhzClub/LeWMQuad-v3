#!/usr/bin/env python3
"""Train a frozen-JEPA front-blocked classifier for Go2 wall-aware planning."""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.models.go2_jepa import Go2FrontBlockedHead, load_go2_jepa_encoder  # noqa: E402
from train_go2_hidden_target_memory_probe import _load_image, _resolve_device  # noqa: E402


@dataclass(frozen=True)
class Example:
    rgb_path: Path
    scene_id: str
    traversability_forward_m: float
    label: float


def _load_rows(paths: list[Path], *, block_distance_m: float) -> list[Example]:
    seen = set()
    examples: list[Example] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not bool(row.get("camera_valid", True)):
                continue
            rgb_path = Path(str(row.get("rgb_path", "")))
            trav = row.get("traversability_forward_m")
            if trav is None or not rgb_path.exists():
                continue
            key = str(rgb_path)
            if key in seen:
                continue
            seen.add(key)
            trav_f = float(trav)
            examples.append(
                Example(
                    rgb_path=rgb_path,
                    scene_id=str(row.get("scene_id", "")),
                    traversability_forward_m=trav_f,
                    label=1.0 if trav_f < float(block_distance_m) else 0.0,
                )
            )
    return examples


def _split_by_scene(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    scenes = sorted({item.scene_id for item in examples})
    if len(scenes) <= 1:
        cut = max(1, int(len(examples) * 0.8))
        return examples[:cut], examples[cut:]
    val_scenes = set(scenes[::4] or scenes[-1:])
    train = [item for item in examples if item.scene_id not in val_scenes]
    val = [item for item in examples if item.scene_id in val_scenes]
    if not train or not val:
        cut = max(1, int(len(examples) * 0.8))
        train, val = examples[:cut], examples[cut:]
    return train, val


def _precompute_latents(
    encoder: torch.nn.Module,
    examples: list[Example],
    *,
    image_size: int,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = []
    labels = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            images = torch.stack([
                _load_image(item.rgb_path, image_size=image_size) for item in batch
            ]).to(device)
            latents.append(encoder(images).cpu())
            labels.extend(float(item.label) for item in batch)
    return torch.cat(latents, dim=0), torch.tensor(labels, dtype=torch.float32)


def _metrics(logits: torch.Tensor, labels: torch.Tensor, *, threshold: float) -> dict[str, Any]:
    probs = torch.sigmoid(logits)
    pred = probs >= float(threshold)
    target = labels >= 0.5
    tp = int((pred & target).sum().item())
    fp = int((pred & ~target).sum().item())
    tn = int((~pred & ~target).sum().item())
    fn = int((~pred & target).sum().item())
    total = max(1, int(labels.numel()))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall / max(1e-9, precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "count": total,
        "positive_count": int(target.sum().item()),
        "positive_rate": float(target.float().mean().item()) if total else 0.0,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prob_mean": float(probs.mean().item()),
        "prob_pos_mean": float(probs[target].mean().item()) if int(target.sum().item()) else None,
        "prob_neg_mean": float(probs[~target].mean().item()) if int((~target).sum().item()) else None,
    }


def _train(
    model: Go2FrontBlockedHead,
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    val_latents: torch.Tensor,
    val_labels: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    threshold: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    model.to(device)
    train_latents = train_latents.to(device)
    train_labels = train_labels.to(device)
    val_latents = val_latents.to(device)
    val_labels = val_labels.to(device)
    pos = float(train_labels.sum().item())
    neg = float(train_labels.numel() - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = -1.0
    history = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(12345)
    for epoch in range(1, int(epochs) + 1):
        order = torch.randperm(train_latents.shape[0], generator=gen)
        losses = []
        model.train()
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)].to(device)
            logits = model(train_latents[idx])
            loss = F.binary_cross_entropy_with_logits(
                logits,
                train_labels[idx],
                pos_weight=pos_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            train_logits = model(train_latents).detach().cpu()
            val_logits = model(val_latents).detach().cpu()
        train_metrics = _metrics(train_logits, train_labels.detach().cpu(), threshold=threshold)
        val_metrics = _metrics(val_logits, val_labels.detach().cpu(), threshold=threshold)
        score = float(val_metrics["f1"]) + 0.05 * float(val_metrics["recall"])
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "train": train_metrics,
            "validation": val_metrics,
        })
        if score >= best_score:
            best_score = score
            best_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == int(epochs) or epoch % 10 == 0:
            print(
                f"epoch={epoch} loss={history[-1]['loss']:.4f}"
                f" val_f1={val_metrics['f1']:.3f}"
                f" val_precision={val_metrics['precision']:.3f}"
                f" val_recall={val_metrics['recall']:.3f}",
                flush=True,
            )
    assert best_state is not None and best_metrics is not None
    return best_state, best_metrics, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--block-distance-m", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_examples = _load_rows(args.datasets, block_distance_m=float(args.block_distance_m))
    if args.validation_datasets:
        val_examples = _load_rows(args.validation_datasets, block_distance_m=float(args.block_distance_m))
    else:
        random.shuffle(train_examples)
        train_examples, val_examples = _split_by_scene(train_examples)
    if not train_examples:
        raise SystemExit("no train examples")
    if not val_examples:
        raise SystemExit("no validation examples")

    device = _resolve_device(str(args.device))
    encoder, encoder_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=True,
    )
    train_latents, train_labels = _precompute_latents(
        encoder,
        train_examples,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )
    val_latents, val_labels = _precompute_latents(
        encoder,
        val_examples,
        image_size=int(args.image_size),
        device=device,
        batch_size=int(args.batch_size),
    )

    latent_dim = int(encoder_checkpoint.get("latent_dim", train_latents.shape[-1]))
    model = Go2FrontBlockedHead(latent_dim=latent_dim, hidden_dim=int(args.hidden_dim))
    best_state, best_metrics, history = _train(
        model,
        train_latents,
        train_labels,
        val_latents,
        val_labels,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        threshold=float(args.threshold),
    )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(train_latents).cpu()
        val_logits = model(val_latents).cpu()
    train_metrics = _metrics(train_logits, train_labels.cpu(), threshold=float(args.threshold))
    val_metrics = _metrics(val_logits, val_labels.cpu(), threshold=float(args.threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "go2_jepa_front_blocked_predictor_v0",
            "model_state_dict": best_state,
            "latent_dim": latent_dim,
            "hidden_dim": int(args.hidden_dim),
            "image_size": int(args.image_size),
            "block_distance_m": float(args.block_distance_m),
            "threshold": float(args.threshold),
            "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        },
        args.output,
    )
    report = {
        "schema": "go2_jepa_front_blocked_predictor_report_v0",
        "checkpoint": str(args.output),
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "train_scenes": sorted({item.scene_id for item in train_examples}),
        "validation_scenes": sorted({item.scene_id for item in val_examples}),
        "block_distance_m": float(args.block_distance_m),
        "threshold": float(args.threshold),
        "train": train_metrics,
        "validation": val_metrics,
        "best_validation": best_metrics,
        "history": history,
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("schema", "checkpoint", "train_examples", "validation_examples", "train", "validation")}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
