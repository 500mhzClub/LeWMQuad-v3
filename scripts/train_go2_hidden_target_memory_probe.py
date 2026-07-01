#!/usr/bin/env python3
"""Train a tiny recurrent RGB memory probe for Go2 hidden-target slices.

This is a bridge artifact, not a controller. It checks that the rendered Go2
dataset can drive a learned recurrent state that preserves which landmark has
been seen after it is no longer visible.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Example:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    visible_target: torch.Tensor
    memory_target: torch.Tensor
    hidden_memory_mask: torch.Tensor
    causal_memory_target: torch.Tensor
    causal_memory_mask: torch.Tensor


class TinyGo2MemoryProbe(nn.Module):
    def __init__(self, landmark_count: int, aux_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim + aux_dim, hidden_dim)
        self.visible_head = nn.Linear(hidden_dim, landmark_count)
        self.memory_head = nn.Linear(hidden_dim, landmark_count)

    def forward_sequence(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        reset_each_step: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(images)
        h = torch.zeros(
            encoded.shape[1] if encoded.ndim == 3 else encoded.shape[-1],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        visible_logits = []
        memory_logits = []
        for idx in range(encoded.shape[0]):
            if reset_each_step:
                h = torch.zeros_like(h)
            h = self.gru(torch.cat([encoded[idx], aux[idx]], dim=-1), h)
            visible_logits.append(self.visible_head(h))
            memory_logits.append(self.memory_head(h))
        return torch.stack(visible_logits), torch.stack(memory_logits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows = _load_rows(args.datasets)
    validation_rows = (
        _load_rows(args.validation_datasets) if args.validation_datasets else None
    )
    landmark_ids = _landmark_ids(train_rows, validation_rows or [])
    primitive_vocab = _primitive_vocab(train_rows, validation_rows or [])

    if validation_rows is None:
        train_rows, validation_rows = _split_rows_by_sequence(train_rows)

    feature_stats = _feature_stats(train_rows, primitive_vocab=primitive_vocab)
    train_sequences = _build_examples(
        train_rows,
        landmark_ids=landmark_ids,
        primitive_vocab=primitive_vocab,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
    )
    validation_sequences = _build_examples(
        validation_rows,
        landmark_ids=landmark_ids,
        primitive_vocab=primitive_vocab,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    device = _resolve_device(args.device)
    aux_dim = next(iter(train_sequences.values()))[0].aux.numel()
    model = TinyGo2MemoryProbe(
        landmark_count=len(landmark_ids),
        aux_dim=aux_dim,
        hidden_dim=int(args.hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    history = []
    best_validation_score = -1.0
    best_validation_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_epoch(model, optimizer, train_sequences, device=device)
        train_metrics = _evaluate(model, train_sequences, device=device)
        validation_metrics = _evaluate(model, validation_sequences, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        selection_score = _selection_score(validation_metrics)
        if selection_score >= best_validation_score:
            best_validation_score = selection_score
            best_validation_metrics = dict(validation_metrics)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" train_loss={train_loss:.4f}"
                f" val_memory_acc={validation_metrics['memory_frame_accuracy']:.3f}"
                f" val_hidden_recall={validation_metrics['hidden_memory_recall']:.3f}"
                f" val_hidden_precision={validation_metrics['hidden_memory_precision']:.3f}"
                f" val_hidden_f1={validation_metrics['hidden_memory_f1']:.3f}"
                f" val_causal_bal={validation_metrics['causal_current_balanced_accuracy']:.3f}"
                f" memoryless_hidden_recall={validation_metrics['memoryless_hidden_recall']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate(model, train_sequences, device=device)
    final_validation = _evaluate(model, validation_sequences, device=device)
    final_validation_reset = _evaluate(
        model,
        validation_sequences,
        device=device,
        ablation="reset_recurrent_state",
    )
    final_validation_reversed = _evaluate(
        model,
        validation_sequences,
        device=device,
        ablation="reverse_input_history",
    )

    checkpoint = {
        "schema": "lewm_go2_hidden_target_memory_probe_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "landmark_ids": landmark_ids,
        "primitive_vocab": primitive_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "aux_dim": int(aux_dim),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_hidden_target_memory_probe_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": (
            [str(path) for path in args.validation_datasets]
            if args.validation_datasets
            else []
        ),
        "split_note": (
            "explicit validation datasets"
            if args.validation_datasets
            else "sequence holdout inside the provided dataset; not scene-disjoint"
        ),
        "output": str(args.output),
        "device": str(device),
        "landmark_ids": landmark_ids,
        "primitive_vocab": primitive_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(seq) for seq in train_sequences.values()),
        "validation_row_count": sum(len(seq) for seq in validation_sequences.values()),
        "final_train": final_train,
        "final_validation": final_validation,
        "final_validation_reset_recurrent_state": final_validation_reset,
        "final_validation_reverse_input_history": final_validation_reversed,
        "best_validation_selection_score": best_validation_score,
        "best_validation_selected_metrics": best_validation_metrics or {},
        "history": history,
        "claim_boundary": (
            "This is a supervised recurrent RGB memory probe over rendered "
            "route-teacher event slices. It is not yet a closed-loop Go2 controller."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_hidden_target_memory_probe:"
        f" output={args.output}"
        f" report={report_path}"
        f" val_hidden_recall={final_validation['hidden_memory_recall']:.3f}"
        f" reset_hidden_recall={final_validation_reset['hidden_memory_recall']:.3f}"
        f" reversed_hidden_recall={final_validation_reversed['hidden_memory_recall']:.3f}"
        f" memoryless_hidden_recall={final_validation['memoryless_hidden_recall']:.3f}"
    )
    return 0


def _load_rows(paths: list[Path] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths or []:
        path = path.resolve()
        if path.is_dir():
            path = path / "dataset.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _landmark_ids(*row_groups: list[dict[str, Any]]) -> list[str]:
    ids = {
        str(landmark.get("object_id", ""))
        for rows in row_groups
        for row in rows
        for landmark in row.get("landmarks", ())
    }
    ids.discard("")
    if not ids:
        raise SystemExit("dataset contains no landmark ids")
    return sorted(ids)


def _primitive_vocab(*row_groups: list[dict[str, Any]]) -> list[str]:
    vocab = {
        str((row.get("command") or {}).get("primitive_name", ""))
        for rows in row_groups
        for row in rows
    }
    vocab.discard("")
    return sorted(vocab)


def _split_rows_by_sequence(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_sequence: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_sequence.setdefault(_seq_key(row), []).append(row)
    keys = sorted(by_sequence)
    if len(keys) < 2:
        return rows, rows
    validation_count = max(1, len(keys) // 3)
    validation_keys = set(keys[-validation_count:])
    train_rows = [
        row
        for key, sequence_rows in by_sequence.items()
        for row in sequence_rows
        if key not in validation_keys
    ]
    validation_rows = [
        row
        for key, sequence_rows in by_sequence.items()
        for row in sequence_rows
        if key in validation_keys
    ]
    return train_rows, validation_rows


def _feature_stats(
    rows: list[dict[str, Any]],
    *,
    primitive_vocab: list[str],
) -> dict[str, np.ndarray]:
    features = np.stack([_aux_features(row, primitive_vocab=primitive_vocab) for row in rows])
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), 1e-6)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _build_examples(
    rows: list[dict[str, Any]],
    *,
    landmark_ids: list[str],
    primitive_vocab: list[str],
    feature_stats: dict[str, np.ndarray],
    image_size: int,
) -> dict[tuple[str, int, int], list[Example]]:
    first_visible = _first_visible_steps(rows)
    landmark_index = {landmark_id: idx for idx, landmark_id in enumerate(landmark_ids)}
    sequences: dict[tuple[str, int, int], list[Example]] = {}
    for row in rows:
        seq_key = _seq_key(row)
        step = int(row.get("episode_step", 0))
        visible_ids = set(str(item) for item in row.get("visible_landmark_ids", ()))
        visible = torch.tensor(
            [1.0 if landmark_id in visible_ids else 0.0 for landmark_id in landmark_ids],
            dtype=torch.float32,
        )
        memory = torch.tensor(
            [
                1.0
                if (
                    first_visible.get((seq_key, landmark_id)) is not None
                    and step >= int(first_visible[(seq_key, landmark_id)])
                )
                else 0.0
                for landmark_id in landmark_ids
            ],
            dtype=torch.float32,
        )
        hidden_mask = (memory > 0.5) & (visible < 0.5)
        causal_target = torch.zeros(len(landmark_ids), dtype=torch.float32)
        causal_mask = torch.zeros(len(landmark_ids), dtype=torch.float32)
        for event in row.get("go2_causal_memory_pair_selection", ()):
            role = str(event.get("pair_role", ""))
            if not role.startswith("current_"):
                continue
            object_id = str(event.get("object_id", ""))
            if object_id not in landmark_index:
                continue
            index = landmark_index[object_id]
            causal_mask[index] = 1.0
            causal_target[index] = 1.0 if bool(event.get("seen_before", False)) else 0.0
        aux = _aux_features(row, primitive_vocab=primitive_vocab)
        aux = (aux - feature_stats["mean"]) / feature_stats["std"]
        example = Example(
            seq_key=seq_key,
            episode_step=step,
            image=_load_image(Path(row["rgb_path"]), image_size=image_size),
            aux=torch.tensor(aux, dtype=torch.float32),
            visible_target=visible,
            memory_target=memory,
            hidden_memory_mask=hidden_mask.float(),
            causal_memory_target=causal_target,
            causal_memory_mask=causal_mask,
        )
        sequences.setdefault(seq_key, []).append(example)
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item.episode_step)
    return sequences


def _first_visible_steps(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, int, int], str], int]:
    steps: dict[tuple[tuple[str, int, int], str], int] = {}
    for row in rows:
        seq_key = _seq_key(row)
        for event in row.get("go2_hidden_target_memory_selection", ()):
            if str(event.get("event_field", "")) != "first_visible_step":
                continue
            key = (seq_key, str(event.get("object_id", "")))
            step = int(event.get("event_step", row.get("episode_step", 0)))
            steps[key] = min(step, steps.get(key, step))
        for landmark_id in row.get("visible_landmark_ids", ()):
            key = (seq_key, str(landmark_id))
            step = int(row.get("episode_step", 0))
            steps[key] = min(step, steps.get(key, step))
    return steps


def _load_image(path: Path, *, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _aux_features(row: dict[str, Any], *, primitive_vocab: list[str]) -> np.ndarray:
    command = row.get("command") or {}
    block = [float(v) for v in row.get("integrated_body_motion_block", ())[:3]]
    window = [float(v) for v in row.get("integrated_body_motion_window", ())[:3]]
    while len(block) < 3:
        block.append(0.0)
    while len(window) < 3:
        window.append(0.0)
    command_values = []
    for field in ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"):
        values = [float(v) for v in command.get(field, ())]
        command_values.append(float(np.mean(values)) if values else 0.0)
    primitive = str(command.get("primitive_name", ""))
    primitive_one_hot = [1.0 if primitive == name else 0.0 for name in primitive_vocab]
    return np.asarray(
        block
        + window
        + command_values
        + [
            float(row.get("clearance_m", 0.0)),
            float(row.get("traversability_forward_m", 0.0)),
        ]
        + primitive_one_hot,
        dtype=np.float32,
    )


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _train_epoch(
    model: TinyGo2MemoryProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Example]],
    *,
    device: torch.device,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    for key in keys:
        sequence = sequences[key]
        images, aux, visible, memory, hidden_mask, causal_target, causal_mask = (
            _sequence_tensors(sequence, device=device)
        )
        visible_logits, memory_logits = model.forward_sequence(images, aux)
        visible_loss = F.binary_cross_entropy_with_logits(visible_logits, visible)
        memory_loss = F.binary_cross_entropy_with_logits(memory_logits, memory)
        hidden_loss = _masked_bce(memory_logits, memory, hidden_mask)
        causal_loss = _masked_bce(memory_logits, causal_target, causal_mask)
        loss = visible_loss + memory_loss + hidden_loss + 2.0 * causal_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(keys))


def _evaluate(
    model: TinyGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Example]],
    *,
    device: torch.device,
    ablation: str = "normal",
) -> dict[str, float]:
    model.eval()
    visible_correct = 0.0
    memory_correct = 0.0
    total = 0.0
    hidden_correct = 0.0
    hidden_total = 0.0
    hidden_true_positive = 0.0
    hidden_false_positive = 0.0
    hidden_false_negative = 0.0
    hidden_negative_total = 0.0
    causal_total = 0.0
    causal_correct = 0.0
    causal_seen_total = 0.0
    causal_seen_correct = 0.0
    causal_unseen_total = 0.0
    causal_unseen_correct = 0.0
    memoryless_hidden_correct = 0.0
    losses = []
    with torch.no_grad():
        for sequence in sequences.values():
            images, aux, visible, memory, hidden_mask, causal_target, causal_mask = (
                _sequence_tensors(sequence, device=device)
            )
            if ablation == "normal":
                visible_logits, memory_logits = model.forward_sequence(images, aux)
            elif ablation == "reset_recurrent_state":
                visible_logits, memory_logits = model.forward_sequence(
                    images,
                    aux,
                    reset_each_step=True,
                )
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                visible_logits, memory_logits = model.forward_sequence(images[order], aux[order])
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            losses.append(
                float(
                    (
                        F.binary_cross_entropy_with_logits(visible_logits, visible)
                        + F.binary_cross_entropy_with_logits(memory_logits, memory)
                        + _masked_bce(memory_logits, memory, hidden_mask)
                        + 2.0 * _masked_bce(memory_logits, causal_target, causal_mask)
                    ).cpu()
                )
            )
            visible_pred = (torch.sigmoid(visible_logits) >= 0.5).float()
            memory_pred = (torch.sigmoid(memory_logits) >= 0.5).float()
            visible_correct += float((visible_pred == visible).sum().cpu())
            memory_correct += float((memory_pred == memory).sum().cpu())
            total += float(visible.numel())
            hidden_total += float(hidden_mask.sum().cpu())
            hidden_correct += float(((memory_pred == memory).float() * hidden_mask).sum().cpu())
            hidden_negative_mask = ((memory < 0.5) & (visible < 0.5)).float()
            hidden_true_positive += float((memory_pred * hidden_mask).sum().cpu())
            hidden_false_positive += float((memory_pred * hidden_negative_mask).sum().cpu())
            hidden_false_negative += float(((1.0 - memory_pred) * hidden_mask).sum().cpu())
            hidden_negative_total += float(hidden_negative_mask.sum().cpu())
            causal_total += float(causal_mask.sum().cpu())
            causal_correct += float(
                ((memory_pred == causal_target).float() * causal_mask).sum().cpu()
            )
            causal_seen_mask = causal_mask * (causal_target > 0.5).float()
            causal_unseen_mask = causal_mask * (causal_target < 0.5).float()
            causal_seen_total += float(causal_seen_mask.sum().cpu())
            causal_seen_correct += float((memory_pred * causal_seen_mask).sum().cpu())
            causal_unseen_total += float(causal_unseen_mask.sum().cpu())
            causal_unseen_correct += float(
                ((1.0 - memory_pred) * causal_unseen_mask).sum().cpu()
            )
            memoryless_hidden_correct += float(
                ((visible == memory).float() * hidden_mask).sum().cpu()
            )
    denom = max(1.0, total)
    hidden_denom = max(1.0, hidden_total)
    hidden_precision_denom = max(1.0, hidden_true_positive + hidden_false_positive)
    hidden_f1_denom = max(
        1.0,
        2.0 * hidden_true_positive + hidden_false_positive + hidden_false_negative,
    )
    all_memory_precision_denom = max(1.0, hidden_total + hidden_negative_total)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "visible_frame_accuracy": visible_correct / denom,
        "memory_frame_accuracy": memory_correct / denom,
        "hidden_memory_recall": hidden_correct / hidden_denom,
        "hidden_memory_precision": hidden_true_positive / hidden_precision_denom,
        "hidden_memory_f1": (2.0 * hidden_true_positive) / hidden_f1_denom,
        "hidden_memory_false_positive_rate": hidden_false_positive
        / max(1.0, hidden_negative_total),
        "memoryless_hidden_recall": memoryless_hidden_correct / hidden_denom,
        "always_memory_hidden_recall": 1.0 if hidden_total > 0.0 else 0.0,
        "always_memory_hidden_precision": hidden_total / all_memory_precision_denom,
        "hidden_memory_true_positive_count": hidden_true_positive,
        "hidden_memory_false_positive_count": hidden_false_positive,
        "hidden_memory_false_negative_count": hidden_false_negative,
        "hidden_memory_target_count": hidden_total,
        "hidden_memory_negative_count": hidden_negative_total,
        "causal_current_accuracy": causal_correct / max(1.0, causal_total),
        "causal_seen_before_recall": causal_seen_correct / max(1.0, causal_seen_total),
        "causal_unseen_before_specificity": causal_unseen_correct
        / max(1.0, causal_unseen_total),
        "causal_current_balanced_accuracy": 0.5
        * (
            causal_seen_correct / max(1.0, causal_seen_total)
            + causal_unseen_correct / max(1.0, causal_unseen_total)
        ),
        "causal_current_target_count": causal_total,
        "causal_seen_before_count": causal_seen_total,
        "causal_unseen_before_count": causal_unseen_total,
    }


def _selection_score(metrics: dict[str, float]) -> float:
    if float(metrics.get("causal_current_target_count", 0.0)) > 0.0:
        return float(metrics["causal_current_balanced_accuracy"])
    return float(metrics["hidden_memory_f1"])


def _sequence_tensors(
    sequence: list[Example],
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.stack([item.image for item in sequence]).to(device),
        torch.stack([item.aux for item in sequence]).to(device),
        torch.stack([item.visible_target for item in sequence]).to(device),
        torch.stack([item.memory_target for item in sequence]).to(device),
        torch.stack([item.hidden_memory_mask for item in sequence]).to(device),
        torch.stack([item.causal_memory_target for item in sequence]).to(device),
        torch.stack([item.causal_memory_mask for item in sequence]).to(device),
    )


def _masked_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if float(mask.sum().detach().cpu()) <= 0.0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("requested CUDA/ROCm device is unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


if __name__ == "__main__":
    raise SystemExit(main())
