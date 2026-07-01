#!/usr/bin/env python3
"""Train a learned extractor-mode head for Phase 3A value-field planning."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import read_jsonl  # noqa: E402
from lewm.models.phase3a_latent_map import (  # noqa: E402
    Phase3AValueFieldExtractorHead,
)
from scripts.export_jepa_phase3a_closed_loop_demo_mp4 import _infer_scene_seed  # noqa: E402
from scripts.report_jepa_phase3a_explore_claim import load_model  # noqa: E402
from scripts.train_jepa_phase3a_latent_memory import _load_latent_map_head  # noqa: E402
from scripts.train_jepa_phase3a_value_field import (  # noqa: E402
    _build_examples,
    _load_latent_memory_updater,
)


@torch.no_grad()
def _evaluate(
    extractor: Phase3AValueFieldExtractorHead,
    memories: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    positive_weight: float,
    threshold: float,
    device: torch.device,
) -> dict:
    extractor.eval()
    dataset = TensorDataset(memories, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    loss_total = 0.0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    pos_weight = torch.tensor([positive_weight], dtype=torch.float32, device=device)
    for memory, label in loader:
        memory = memory.to(device)
        label = label.to(device=device, dtype=torch.float32)
        logits = extractor(memory)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            label,
            pos_weight=pos_weight,
        )
        prob = logits.sigmoid()
        pred = prob >= threshold
        truth = label >= 0.5
        total += int(memory.shape[0])
        loss_total += float(loss.item()) * int(memory.shape[0])
        true_positive += int((pred & truth).sum().item())
        false_positive += int((pred & ~truth).sum().item())
        true_negative += int((~pred & ~truth).sum().item())
        false_negative += int((~pred & truth).sum().item())
    correct = true_positive + true_negative
    positive_count = true_positive + false_negative
    predicted_positive = true_positive + false_positive
    return {
        "examples": total,
        "loss": loss_total / max(total, 1),
        "accuracy": correct / max(total, 1),
        "sparse_recall": true_positive / max(positive_count, 1),
        "sparse_precision": true_positive / max(predicted_positive, 1),
        "sparse_examples": positive_count,
        "predicted_sparse_examples": predicted_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-map-head", type=Path, required=True)
    parser.add_argument("--latent-memory-updater", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-cells", type=int, default=17)
    parser.add_argument("--height-cells", type=int, default=17)
    parser.add_argument("--view-size", type=int, default=7)
    parser.add_argument("--memory-size", type=int, default=31)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--max-validation-episodes", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=68)
    parser.add_argument("--include-marker-start-train-groups", action="store_true")
    parser.add_argument("--optimization-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--latent-memory-blocked-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-free-threshold", type=float, default=0.5)
    parser.add_argument("--latent-memory-marker-threshold", type=float, default=0.9)
    parser.add_argument("--extractor-threshold", type=float, default=0.5)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--seed", type=int, default=20260654)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--log-every", type=int, default=128)
    args = parser.parse_args()

    if args.memory_size < args.view_size:
        raise SystemExit("--memory-size must be >= --view-size")
    if args.memory_size % 2 == 0:
        raise SystemExit("--memory-size must be odd")
    if args.optimization_steps < 1:
        raise SystemExit("--optimization-steps must be positive")
    if args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be positive")
    if args.positive_weight <= 0.0:
        raise SystemExit("--positive-weight must be positive")
    if not 0.0 <= args.extractor_threshold <= 1.0:
        raise SystemExit("--extractor-threshold must be in [0, 1]")

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_seed = _infer_scene_seed(args.train_data)
    validation_seed = _infer_scene_seed(args.validation_data)
    if train_seed is None or validation_seed is None:
        raise SystemExit("could not infer train/validation scene seeds")
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)

    base_model, base_report = load_model(args.base_checkpoint, device=device)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    latent_map_head, latent_map_report = _load_latent_map_head(
        args.latent_map_head,
        base_model=base_model,
        device=device,
    )
    latent_memory_updater, latent_memory_report = _load_latent_memory_updater(
        args.latent_memory_updater,
        model=base_model,
        device=device,
    )
    if int(latent_memory_updater.memory_size) != int(args.memory_size):
        raise SystemExit(
            "--memory-size must match latent memory updater size "
            f"({latent_memory_updater.memory_size})"
        )

    train_examples = _build_examples(
        train_rows,
        scene_seed=train_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_train_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        output_channels=1,
        include_marker_start_groups=args.include_marker_start_train_groups,
        rollout_value_field_head=None,
        rollout_target_threshold=0.5,
        rollout_target_top_k=16,
        rollout_fixed_marker_target=False,
        device=device,
    )
    validation_examples = _build_examples(
        validation_rows,
        scene_seed=validation_seed,
        width=args.width_cells,
        height=args.height_cells,
        view_size=args.view_size,
        memory_size=args.memory_size,
        max_episodes=args.max_validation_episodes,
        max_steps=args.max_steps,
        base_model=base_model,
        latent_map_head=latent_map_head,
        latent_memory_updater=latent_memory_updater,
        blocked_threshold=args.latent_memory_blocked_threshold,
        free_threshold=args.latent_memory_free_threshold,
        marker_threshold=args.latent_memory_marker_threshold,
        output_channels=1,
        include_marker_start_groups=False,
        rollout_value_field_head=None,
        rollout_target_threshold=0.5,
        rollout_target_top_k=16,
        rollout_fixed_marker_target=False,
        device=device,
    )

    train_labels = train_examples.marker_targets.to(dtype=torch.float32)
    validation_labels = validation_examples.marker_targets.to(dtype=torch.float32)
    train_dataset = TensorDataset(train_examples.memories, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    extractor = Phase3AValueFieldExtractorHead(
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        memory_channels=3,
    ).to(device)
    optimizer = torch.optim.AdamW(
        extractor.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pos_weight = torch.tensor([args.positive_weight], dtype=torch.float32, device=device)
    iterator = iter(train_loader)
    logs = []
    best_state = None
    best_step = None
    best_metrics = None
    best_score = (-1.0, -1.0, float("inf"))
    for step in range(1, args.optimization_steps + 1):
        try:
            memory, label = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            memory, label = next(iterator)
        memory = memory.to(device)
        label = label.to(device=device, dtype=torch.float32)
        logits = extractor(memory)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            label,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.optimization_steps:
            metrics = _evaluate(
                extractor,
                validation_examples.memories,
                validation_labels,
                batch_size=args.batch_size,
                positive_weight=args.positive_weight,
                threshold=args.extractor_threshold,
                device=device,
            )
            entry = {"step": step, "train_loss": float(loss.item()), **metrics}
            logs.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            score = (
                float(metrics["accuracy"]),
                float(metrics["sparse_recall"]),
                -float(metrics["loss"]),
            )
            if args.save_best and score > best_score:
                best_score = score
                best_step = step
                best_metrics = dict(metrics)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in extractor.state_dict().items()
                }

    final_validation = _evaluate(
        extractor,
        validation_examples.memories,
        validation_labels,
        batch_size=args.batch_size,
        positive_weight=args.positive_weight,
        threshold=args.extractor_threshold,
        device=device,
    )
    selected_step = args.optimization_steps
    selected_validation = final_validation
    if args.save_best and best_state is not None:
        extractor.load_state_dict(best_state)
        selected_step = int(best_step)
        selected_validation = best_metrics or final_validation

    report = {
        "schema": "jepa_phase3a_value_extractor_training_report_v0",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_completed_steps": base_report.get("completed_steps"),
        "latent_map_head": str(args.latent_map_head.resolve()),
        "latent_map_completed_steps": latent_map_report.get("completed_steps"),
        "latent_memory_updater": str(args.latent_memory_updater.resolve()),
        "latent_memory_completed_steps": latent_memory_report.get("completed_steps"),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "train_seed": train_seed,
        "validation_seed": validation_seed,
        "train_examples": int(len(train_labels)),
        "train_sparse_examples": int(train_labels.sum().item()),
        "validation_examples": int(len(validation_labels)),
        "validation_sparse_examples": int(validation_labels.sum().item()),
        "completed_steps": args.optimization_steps,
        "final_validation": final_validation,
        "selected_step": selected_step,
        "selected_validation": selected_validation,
        "logs": logs,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": {
            "memory_size": args.memory_size,
            "hidden_dim": args.hidden_dim,
            "memory_channels": 3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "extractor_state_dict": extractor.state_dict(),
            "report": report,
        },
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
