#!/usr/bin/env python3
"""Train a query-free Go2 RGB/JEPA latent-memory action controller.

The inference path is deliberately stricter than the object-query controllers:
RGB frames are encoded by a Go2 JEPA encoder, a learned recurrent state carries
memory, and the controller emits abstain/right/forward/left. Landmark ids,
object slots, current bearings, ranges, and runtime detector geometry are used
only to build offline labels and optional auxiliary training targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _load_image,
    _load_rows,
    _resolve_device,
)


ACTION_CLASSES = ("abstain", "right", "forward", "left")
STEERING_CLASSES = ("right", "forward", "left")


@dataclass(frozen=True)
class ActionFrame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    target_action: int
    has_query: bool
    is_positive: bool
    positive_objects: tuple[str, ...]
    target_steerings: tuple[str, ...]
    visible_slots: tuple[int, ...]


class JepaLatentMemoryActionController(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_output_dim: int,
        aux_dim: int,
        hidden_dim: int,
        freeze_encoder: bool,
        memory_slot_count: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.memory_slot_count = int(memory_slot_count)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.encoder_projection = (
            nn.Identity()
            if int(encoder_output_dim) == int(hidden_dim)
            else nn.Sequential(
                nn.Linear(int(encoder_output_dim), int(hidden_dim)),
                nn.GELU(),
            )
        )
        self.gru = nn.GRUCell(int(hidden_dim) + int(aux_dim), int(hidden_dim))
        self.action_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), len(ACTION_CLASSES)),
        )
        self.memory_head = (
            nn.Linear(int(hidden_dim), self.memory_slot_count)
            if self.memory_slot_count > 0
            else None
        )

    def forward_hidden(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        reset_each_step: bool = False,
        include_current: bool = True,
    ) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.encoder(images)
        else:
            encoded = self.encoder(images)
        encoded = self.encoder_projection(encoded)
        h = torch.zeros(encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
        hidden_states = []
        for idx in range(encoded.shape[0]):
            if reset_each_step:
                h = torch.zeros_like(h)
            if not include_current:
                hidden_states.append(h)
            h = self.gru(torch.cat([encoded[idx], aux[idx]], dim=-1), h)
            if include_current:
                hidden_states.append(h)
        return torch.stack(hidden_states)

    def action_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.action_head(hidden)

    def memory_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.memory_head is None:
            raise RuntimeError("memory head is disabled")
        return self.memory_head(hidden)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--positive-frame-weight", type=float, default=1.0)
    parser.add_argument("--negative-frame-weight", type=float, default=1.0)
    parser.add_argument("--memory-state-loss-weight", type=float, default=0.5)
    parser.add_argument("--reset-abstain-loss-weight", type=float, default=0.25)
    parser.add_argument("--shuffle-abstain-loss-weight", type=float, default=0.25)
    parser.add_argument("--finetune-jepa-encoder", action="store_true")
    parser.add_argument(
        "--aux-mode",
        choices=("none", "odom", "action", "odom_action"),
        default="odom",
        help=(
            "Non-landmark auxiliary input. 'odom' uses integrated body motion; "
            "'action' uses executed command values; neither includes landmark "
            "ids, slots, range, bearing, clearance, or traversability."
        ),
    )
    parser.add_argument("--exclusive-memory-state", action="store_true")
    parser.add_argument("--arc-threshold-rad", type=float, default=0.1)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-target-steering-success", type=float, default=0.90)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.12)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.30)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows = _load_rows(args.datasets)
    validation_rows = _load_rows(args.validation_datasets)
    if not train_rows:
        raise SystemExit("no train rows")
    if not validation_rows:
        raise SystemExit("no validation rows")

    aux_stats = _aux_stats(train_rows, mode=str(args.aux_mode))
    slot_count = max(_max_slot(train_rows), _max_slot(validation_rows)) + 1
    train_sequences = _build_sequences(
        train_rows,
        image_size=int(args.image_size),
        aux_mode=str(args.aux_mode),
        aux_stats=aux_stats,
        arc_threshold_rad=float(args.arc_threshold_rad),
        yaw_threshold_rad=float(args.yaw_threshold_rad),
        hold_range_m=float(args.hold_range_m),
    )
    validation_sequences = _build_sequences(
        validation_rows,
        image_size=int(args.image_size),
        aux_mode=str(args.aux_mode),
        aux_stats=aux_stats,
        arc_threshold_rad=float(args.arc_threshold_rad),
        yaw_threshold_rad=float(args.yaw_threshold_rad),
        hold_range_m=float(args.hold_range_m),
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    device = _resolve_device(str(args.device))
    encoder, jepa_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=not bool(args.finetune_jepa_encoder),
    )
    aux_dim = next(iter(train_sequences.values()))[0].aux.numel()
    model = JepaLatentMemoryActionController(
        encoder=encoder,
        encoder_output_dim=int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
        aux_dim=int(aux_dim),
        hidden_dim=int(args.hidden_dim),
        freeze_encoder=not bool(args.finetune_jepa_encoder),
        memory_slot_count=slot_count if float(args.memory_state_loss_weight) > 0.0 else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )

    history = []
    best_score = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            positive_frame_weight=float(args.positive_frame_weight),
            negative_frame_weight=float(args.negative_frame_weight),
            memory_state_loss_weight=float(args.memory_state_loss_weight),
            reset_abstain_loss_weight=float(args.reset_abstain_loss_weight),
            shuffle_abstain_loss_weight=float(args.shuffle_abstain_loss_weight),
            include_current=not bool(args.exclusive_memory_state),
        )
        validation_ablations = _evaluate_ablations(
            model,
            validation_sequences,
            device=device,
            include_current=not bool(args.exclusive_memory_state),
        )
        normal = validation_ablations["normal"]
        gap = _target_success_gap(validation_ablations)
        score = _selection_score(validation_ablations)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss),
                "validation": normal,
                "normal_minus_best_corrupted_target_steering_pipeline_success": float(gap),
            }
        )
        if score >= best_score:
            best_score = float(score)
            best_metrics = validation_ablations
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={loss:.4f}"
                f" target_steer={normal['target_steering_pipeline_success']:.3f}"
                f" false_claim={normal['false_claim_rate']:.3f}"
                f" recall={normal['target_recall']:.3f}"
                f" gap={gap:.3f}"
                f" score={score:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate_ablations(
        model,
        train_sequences,
        device=device,
        include_current=not bool(args.exclusive_memory_state),
    )
    validation_ablations = _evaluate_ablations(
        model,
        validation_sequences,
        device=device,
        include_current=not bool(args.exclusive_memory_state),
    )
    normal = validation_ablations["normal"]
    gap = _target_success_gap(validation_ablations)
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and float(gap) >= float(args.min_corrupted_gap)
    )

    checkpoint = {
        "schema": "lewm_go2_jepa_latent_memory_action_controller_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "frozen_jepa_report": {
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
        },
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "aux_dim": int(aux_dim),
        "aux_mode": str(args.aux_mode),
        "aux_mean": aux_stats["mean"].tolist(),
        "aux_std": aux_stats["std"].tolist(),
        "action_classes": list(ACTION_CLASSES),
        "exclusive_memory_state": bool(args.exclusive_memory_state),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_jepa_latent_memory_action_controller_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "frozen_jepa_report": checkpoint["frozen_jepa_report"],
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "aux_mode": str(args.aux_mode),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_frame_label_counts": _label_counts(train_sequences),
        "validation_frame_label_counts": _label_counts(validation_sequences),
        "final_train_ablations": final_train,
        "validation_ablations": validation_ablations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": float(gap),
        "controller_gate_pass": bool(gate_pass),
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Pure Go2 RGB/JEPA latent-memory action controller. At inference it "
            "uses rendered RGB encoded by the JEPA visual encoder, a learned "
            "recurrent memory state, and optional non-landmark egomotion/action "
            "aux selected by aux_mode. It does not consume runtime landmark ids, "
            "object slots, ranges, bearings, detector visibility, or map/geodesic "
            "geometry."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_jepa_latent_memory_action_controller:"
        f" output={args.output}"
        f" report={report_path}"
        f" target_steer={normal['target_steering_pipeline_success']:.3f}"
        f" false_claim={normal['false_claim_rate']:.3f}"
        f" gap={gap:.3f}"
        f" pass={bool(gate_pass)}",
        flush=True,
    )
    return 0


def _build_sequences(
    rows: list[dict[str, Any]],
    *,
    image_size: int,
    aux_mode: str,
    aux_stats: dict[str, np.ndarray],
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> dict[tuple[str, int, int], list[ActionFrame]]:
    sequences: dict[tuple[str, int, int], list[ActionFrame]] = defaultdict(list)
    for row in rows:
        positive_objects = _positive_current_objects(row)
        target_steerings = tuple(
            _target_steering(
                row,
                object_id,
                arc_threshold_rad=arc_threshold_rad,
                yaw_threshold_rad=yaw_threshold_rad,
                hold_range_m=hold_range_m,
            )
            for object_id in positive_objects
        )
        has_query = bool(_current_objects(row))
        if positive_objects:
            target_action = _positive_action(target_steerings)
        else:
            target_action = 0
        aux = _aux_features(row, mode=aux_mode)
        aux = (aux - aux_stats["mean"]) / aux_stats["std"]
        frame = ActionFrame(
            seq_key=_seq_key(row),
            episode_step=int(row.get("episode_step", 0)),
            image=_load_image(Path(row["rgb_path"]), image_size=image_size),
            aux=torch.tensor(aux, dtype=torch.float32),
            target_action=int(target_action),
            has_query=has_query,
            is_positive=bool(positive_objects),
            positive_objects=positive_objects,
            target_steerings=target_steerings,
            visible_slots=tuple(_visible_slots(row)),
        )
        sequences[frame.seq_key].append(frame)
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item.episode_step)
    return dict(sequences)


def _train_epoch(
    model: JepaLatentMemoryActionController,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[ActionFrame]],
    *,
    device: torch.device,
    positive_frame_weight: float,
    negative_frame_weight: float,
    memory_state_loss_weight: float,
    reset_abstain_loss_weight: float,
    shuffle_abstain_loss_weight: float,
    include_current: bool,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    trained = 0
    for key in keys:
        sequence = sequences[key]
        images, aux = _sequence_tensors(sequence, device=device)
        hidden = model.forward_hidden(images, aux, include_current=include_current)
        reset_hidden = (
            model.forward_hidden(
                images,
                aux,
                reset_each_step=True,
                include_current=include_current,
            )
            if float(reset_abstain_loss_weight) > 0.0
            else None
        )
        shuffled_hidden = None
        if float(shuffle_abstain_loss_weight) > 0.0 and hidden.shape[0] > 1:
            shuffled_hidden = torch.roll(hidden, shifts=max(1, hidden.shape[0] // 2), dims=0)
        losses = []
        seen_slots: set[int] = set()
        for step_idx, frame in enumerate(sequence):
            if model.memory_head is not None and float(memory_state_loss_weight) > 0.0:
                seen_slots.update(frame.visible_slots)
                memory_target = torch.zeros(
                    model.memory_slot_count,
                    dtype=hidden.dtype,
                    device=device,
                )
                for slot in seen_slots:
                    if 0 <= slot < model.memory_slot_count:
                        memory_target[slot] = 1.0
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        model.memory_logits(hidden[step_idx]),
                        memory_target,
                    )
                    * float(memory_state_loss_weight)
                )
            if not frame.has_query:
                continue
            target = torch.tensor([frame.target_action], dtype=torch.long, device=device)
            class_weight = torch.ones(len(ACTION_CLASSES), dtype=hidden.dtype, device=device)
            class_weight[0] = float(negative_frame_weight)
            class_weight[1:] = float(positive_frame_weight)
            losses.append(
                F.cross_entropy(
                    model.action_logits(hidden[step_idx]).reshape(1, -1),
                    target,
                    weight=class_weight,
                )
            )
            if reset_hidden is not None:
                losses.append(
                    _abstain_loss(model.action_logits(reset_hidden[step_idx]))
                    * float(reset_abstain_loss_weight)
                )
            if shuffled_hidden is not None:
                losses.append(
                    _abstain_loss(model.action_logits(shuffled_hidden[step_idx]))
                    * float(shuffle_abstain_loss_weight)
                )
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        trained += 1
    return total_loss / max(1, trained)


def _evaluate_ablations(
    model: JepaLatentMemoryActionController,
    sequences: dict[tuple[str, int, int], list[ActionFrame]],
    *,
    device: torch.device,
    include_current: bool,
) -> dict[str, dict[str, Any]]:
    return {
        ablation: _evaluate(
            model,
            sequences,
            device=device,
            ablation=ablation,
            include_current=include_current,
        )
        for ablation in (
            "normal",
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }


def _evaluate(
    model: JepaLatentMemoryActionController,
    sequences: dict[tuple[str, int, int], list[ActionFrame]],
    *,
    device: torch.device,
    ablation: str,
    include_current: bool,
) -> dict[str, Any]:
    hidden_by_key = _hidden_states_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
        include_current=include_current,
    )
    metrics = _ActionMetrics()
    by_target_steering: dict[str, _ActionMetrics] = defaultdict(_ActionMetrics)
    prediction_counts: Counter[str] = Counter()
    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                if not frame.has_query:
                    continue
                if ablation == "memory_off_abstain":
                    predicted_action = 0
                else:
                    logits = model.action_logits(hidden[step_idx])
                    predicted_action = int(torch.argmax(logits).detach().cpu())
                predicted_label = ACTION_CLASSES[
                    max(0, min(predicted_action, len(ACTION_CLASSES) - 1))
                ]
                prediction_counts[predicted_label] += 1
                metrics.add(frame=frame, predicted_action=predicted_action)
                steering_key = frame.target_steerings[0] if frame.target_steerings else "none"
                by_target_steering[steering_key].add(
                    frame=frame,
                    predicted_action=predicted_action,
                )
    result = metrics.to_dict()
    result["ablation"] = ablation
    result["predicted_action_counts"] = dict(sorted(prediction_counts.items()))
    result["by_target_steering"] = {
        key: item.to_dict() for key, item in sorted(by_target_steering.items())
    }
    return result


def _hidden_states_by_sequence(
    model: JepaLatentMemoryActionController,
    sequences: dict[tuple[str, int, int], list[ActionFrame]],
    *,
    device: torch.device,
    ablation: str,
    include_current: bool,
) -> dict[tuple[str, int, int], torch.Tensor]:
    model.eval()
    hidden_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            images, aux = _sequence_tensors(sequence, device=device)
            if ablation in {"normal", "memory_off_abstain", "shuffle_hidden_states"}:
                hidden = model.forward_hidden(images, aux, include_current=include_current)
            elif ablation == "reset_recurrent_state":
                hidden = model.forward_hidden(
                    images,
                    aux,
                    reset_each_step=True,
                    include_current=include_current,
                )
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                hidden = model.forward_hidden(
                    images[order],
                    aux[order],
                    include_current=include_current,
                ).flip(0)
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            hidden_by_key[key] = hidden
    if ablation != "shuffle_hidden_states":
        return hidden_by_key
    flat_hidden = []
    spans: dict[tuple[str, int, int], tuple[int, int]] = {}
    cursor = 0
    for key in sequences:
        hidden = hidden_by_key[key]
        spans[key] = (cursor, cursor + int(hidden.shape[0]))
        flat_hidden.append(hidden)
        cursor += int(hidden.shape[0])
    if cursor <= 1:
        return hidden_by_key
    flat = torch.cat(flat_hidden, dim=0)
    shuffled = torch.roll(flat, shifts=max(1, cursor // 2), dims=0)
    return {key: shuffled[start:end] for key, (start, end) in spans.items()}


class _ActionMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.false_claim = 0
        self.missed_positive = 0
        self.correct_target = 0
        self.target_steer_success = 0
        self.classifications: Counter[str] = Counter()

    def add(self, *, frame: ActionFrame, predicted_action: int) -> None:
        predicted_steering = (
            ACTION_CLASSES[predicted_action]
            if 0 <= predicted_action < len(ACTION_CLASSES)
            else "abstain"
        )
        if frame.is_positive:
            self.positive_frames += 1
            if predicted_steering == "abstain":
                self.missed_positive += 1
                self.classifications["missed_positive"] += 1
                return
            if predicted_steering in frame.target_steerings:
                self.correct_target += 1
                self.target_steer_success += 1
                self.classifications["correct_target"] += 1
            else:
                self.classifications["wrong_steering"] += 1
            return
        self.negative_frames += 1
        if predicted_steering == "abstain":
            self.classifications["abstain"] += 1
        else:
            self.false_claim += 1
            self.classifications["false_claim"] += 1

    def to_dict(self) -> dict[str, Any]:
        positive = max(1, self.positive_frames)
        negative = max(1, self.negative_frames)
        selected = self.correct_target + self.false_claim + self.classifications["wrong_steering"]
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "correct_target_count": float(self.correct_target),
            "target_steering_success_count": float(self.target_steer_success),
            "missed_positive_count": float(self.missed_positive),
            "false_claim_count": float(self.false_claim),
            "target_recall": float(self.correct_target) / positive,
            "target_steering_pipeline_success": float(self.target_steer_success) / positive,
            "false_claim_rate": float(self.false_claim) / negative,
            "target_selection_precision": float(self.correct_target) / max(1, selected),
            "selected_frame_count": float(selected),
            "classification_counts": dict(sorted(self.classifications.items())),
        }


def _sequence_tensors(
    sequence: list[ActionFrame],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([frame.image for frame in sequence]).to(device),
        torch.stack([frame.aux for frame in sequence]).to(device),
    )


def _abstain_loss(logits: torch.Tensor) -> torch.Tensor:
    target = torch.zeros(1, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.reshape(1, -1), target)


def _selection_score(evaluations: dict[str, dict[str, Any]]) -> float:
    normal = evaluations["normal"]
    gap = _target_success_gap(evaluations)
    return (
        2.0 * float(normal["target_steering_pipeline_success"])
        + 0.5 * float(normal["target_recall"])
        + 0.25 * float(normal["target_selection_precision"])
        - 0.75 * float(normal["false_claim_rate"])
        + 0.75 * float(gap)
    )


def _target_success_gap(evaluations: dict[str, dict[str, Any]]) -> float:
    normal = float(evaluations["normal"]["target_steering_pipeline_success"])
    corrupted_best = max(
        float(evaluations[name]["target_steering_pipeline_success"])
        for name in (
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    )
    return normal - corrupted_best


def _label_counts(sequences: dict[tuple[str, int, int], list[ActionFrame]]) -> dict[str, int]:
    counts = Counter()
    for sequence in sequences.values():
        for frame in sequence:
            if not frame.has_query:
                counts["non_query"] += 1
            elif frame.is_positive:
                counts[ACTION_CLASSES[frame.target_action]] += 1
                counts["positive"] += 1
            else:
                counts["abstain"] += 1
                counts["negative"] += 1
    return dict(sorted(counts.items()))


def _positive_action(target_steerings: tuple[str, ...]) -> int:
    counts = Counter(target_steerings)
    if not counts:
        return 0
    steering = counts.most_common(1)[0][0]
    return ACTION_CLASSES.index(steering) if steering in ACTION_CLASSES else 0


def _current_objects(row: dict[str, Any]) -> tuple[str, ...]:
    objects = []
    for event in row.get("go2_causal_memory_pair_selection", ()):
        role = str(event.get("pair_role", ""))
        if not role.startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        if object_id:
            objects.append(object_id)
    return tuple(sorted(set(objects)))


def _positive_current_objects(row: dict[str, Any]) -> tuple[str, ...]:
    objects = []
    for event in row.get("go2_causal_memory_pair_selection", ()):
        role = str(event.get("pair_role", ""))
        if not role.startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        if object_id and bool(event.get("seen_before", False)):
            objects.append(object_id)
    return tuple(sorted(set(objects)))


def _target_steering(
    row: dict[str, Any],
    object_id: str,
    *,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
) -> str:
    for landmark in row.get("landmarks", ()):
        if str(landmark.get("object_id", "")) != str(object_id):
            continue
        bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
        range_m = _finite_float(landmark.get("range_m"), 1.0)
        if range_m <= hold_range_m:
            return "forward"
        if bearing >= arc_threshold_rad or bearing >= yaw_threshold_rad:
            return "left"
        if bearing <= -arc_threshold_rad or bearing <= -yaw_threshold_rad:
            return "right"
        return "forward"
    return "forward"


def _aux_stats(rows: list[dict[str, Any]], *, mode: str) -> dict[str, np.ndarray]:
    features = np.stack([_aux_features(row, mode=mode) for row in rows])
    return {
        "mean": features.mean(axis=0).astype(np.float32),
        "std": np.maximum(features.std(axis=0), 1e-6).astype(np.float32),
    }


def _aux_features(row: dict[str, Any], *, mode: str) -> np.ndarray:
    features: list[float] = []
    if mode in {"odom", "odom_action"}:
        block = [float(v) for v in row.get("integrated_body_motion_block", ())[:3]]
        window = [float(v) for v in row.get("integrated_body_motion_window", ())[:3]]
        while len(block) < 3:
            block.append(0.0)
        while len(window) < 3:
            window.append(0.0)
        features.extend(block + window)
    if mode in {"action", "odom_action"}:
        command = row.get("command") or {}
        for field in ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"):
            values = [float(v) for v in command.get(field, ())]
            features.append(float(np.mean(values)) if values else 0.0)
    if not features:
        features.append(0.0)
    return np.asarray(features, dtype=np.float32)


def _visible_slots(row: dict[str, Any]) -> list[int]:
    slots = []
    for object_id in row.get("visible_landmark_ids", ()):
        slot = _landmark_slot(str(object_id))
        if slot is not None:
            slots.append(slot)
    return slots


def _max_slot(*row_groups: list[dict[str, Any]]) -> int:
    slots = []
    for rows in row_groups:
        for row in rows:
            for landmark in row.get("landmarks", ()):
                slot = _landmark_slot(str(landmark.get("object_id", "")))
                if slot is not None:
                    slots.append(slot)
    return max(slots) if slots else -1


def _landmark_slot(object_id: str) -> int | None:
    for part in str(object_id).split("_"):
        if part.isdigit():
            return int(part)
    return None


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", 0)),
        int(row.get("episode_id", 0)),
    )


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


if __name__ == "__main__":
    raise SystemExit(main())
