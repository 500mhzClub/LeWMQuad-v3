#!/usr/bin/env python3
"""Train a direct frozen-JEPA Go2 target-selection gate.

The earlier query probe trains independent "seen before?" decisions and then
converts them into a frame-level target gate. This script trains the frame-level
decision directly: select one remembered target object, or abstain.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402
from train_go2_causal_memory_query_probe import (  # noqa: E402
    Frame,
    _build_frames,
    _color_vocab,
    _current_role,
    _max_landmark_slot,
    _scrub_command_aux,
    _scrub_runtime_aux,
    _sequence_tensors,
)
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _feature_stats,
    _load_rows,
    _primitive_vocab,
    _resolve_device,
)


class DirectGo2TargetGate(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_output_dim: int,
        aux_dim: int,
        query_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder_projection = (
            nn.Identity()
            if int(encoder_output_dim) == int(hidden_dim)
            else nn.Sequential(
                nn.Linear(int(encoder_output_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
            )
        )
        self.gru = nn.GRUCell(int(hidden_dim) + int(aux_dim), int(hidden_dim))
        self.candidate_head = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(query_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.abstain_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim) // 2, 1),
        )

    def forward_hidden(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        reset_each_step: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.encoder(images)
        encoded = self.encoder_projection(encoded)
        h = torch.zeros(encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
        hidden_states = []
        for idx in range(encoded.shape[0]):
            if reset_each_step:
                h = torch.zeros_like(h)
            h = self.gru(torch.cat([encoded[idx], aux[idx]], dim=-1), h)
            hidden_states.append(h)
        return torch.stack(hidden_states)

    def score_candidates(
        self,
        hidden: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_rows = hidden.repeat(query_features.shape[0], 1)
        candidate_logits = self.candidate_head(
            torch.cat([hidden_rows, query_features], dim=-1)
        ).squeeze(-1)
        abstain_logit = self.abstain_head(hidden).squeeze(-1)
        return abstain_logit, candidate_logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--positive-frame-weight", type=float, default=1.0)
    parser.add_argument("--negative-frame-weight", type=float, default=1.0)
    parser.add_argument(
        "--reset-abstain-loss-weight",
        type=float,
        default=0.0,
        help="Train reset-recurrent hidden states to abstain on current-query frames.",
    )
    parser.add_argument(
        "--shuffle-abstain-loss-weight",
        type=float,
        default=0.0,
        help="Train rolled hidden states to abstain on current-query frames.",
    )
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--include-object-slot",
        action="store_true",
        help="Add parsed landmark ordinal features to the query.",
    )
    parser.add_argument(
        "--scrub-command-aux",
        action="store_true",
        help="Zero current command fields before building aux features.",
    )
    parser.add_argument(
        "--scrub-runtime-aux",
        action="store_true",
        help="Zero command plus scene-derived clearance/traversability aux fields.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows_raw = _load_rows(args.datasets)
    validation_rows_raw = _load_rows(args.validation_datasets)
    if not train_rows_raw:
        raise SystemExit("no train rows")
    if not validation_rows_raw:
        raise SystemExit("no validation rows")
    if args.scrub_runtime_aux:
        train_rows = _scrub_runtime_aux(train_rows_raw)
        validation_rows = _scrub_runtime_aux(validation_rows_raw)
    elif args.scrub_command_aux:
        train_rows = _scrub_command_aux(train_rows_raw)
        validation_rows = _scrub_command_aux(validation_rows_raw)
    else:
        train_rows = train_rows_raw
        validation_rows = validation_rows_raw

    primitive_vocab = _primitive_vocab(train_rows_raw, validation_rows_raw)
    color_vocab = _color_vocab(train_rows, validation_rows)
    max_slot = _max_landmark_slot(train_rows, validation_rows)
    feature_stats = _feature_stats(train_rows, primitive_vocab=primitive_vocab)
    train_sequences = _build_frames(
        train_rows,
        primitive_vocab=primitive_vocab,
        color_vocab=color_vocab,
        max_slot=max_slot,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
        include_object_slot=bool(args.include_object_slot),
        include_privileged_landmark_geometry=False,
    )
    validation_sequences = _build_frames(
        validation_rows,
        primitive_vocab=primitive_vocab,
        color_vocab=color_vocab,
        max_slot=max_slot,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
        include_object_slot=bool(args.include_object_slot),
        include_privileged_landmark_geometry=False,
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    train_counts = _frame_label_counts(train_sequences)
    validation_counts = _frame_label_counts(validation_sequences)
    if train_counts["positive"] <= 0 or train_counts["negative"] <= 0:
        raise SystemExit(f"train split lacks positive/negative frames: {train_counts}")

    device = _resolve_device(str(args.device))
    encoder, jepa_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=True,
    )
    aux_dim = next(iter(train_sequences.values()))[0].aux.numel()
    query_dim = next(
        query.features.numel()
        for sequence in train_sequences.values()
        for frame in sequence
        for query in frame.queries
    )
    model = DirectGo2TargetGate(
        encoder=encoder,
        encoder_output_dim=int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
        aux_dim=int(aux_dim),
        query_dim=int(query_dim),
        hidden_dim=int(args.hidden_dim),
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
        train_loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            positive_frame_weight=float(args.positive_frame_weight),
            negative_frame_weight=float(args.negative_frame_weight),
            reset_abstain_loss_weight=float(args.reset_abstain_loss_weight),
            shuffle_abstain_loss_weight=float(args.shuffle_abstain_loss_weight),
        )
        train_metrics = _evaluate(
            model,
            train_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation="normal",
        )
        normal_validation = _evaluate(
            model,
            validation_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation="normal",
        )
        reset_validation = _evaluate(
            model,
            validation_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation="reset_recurrent_state",
        )
        reversed_validation = _evaluate(
            model,
            validation_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation="reverse_input_history",
        )
        shuffle_validation = _evaluate(
            model,
            validation_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation="shuffle_hidden_states",
        )
        corrupted_best = max(
            reset_validation["overall"]["balanced_frame_accuracy"],
            reversed_validation["overall"]["balanced_frame_accuracy"],
            shuffle_validation["overall"]["balanced_frame_accuracy"],
        )
        gap = normal_validation["overall"]["balanced_frame_accuracy"] - corrupted_best
        score = (
            normal_validation["overall"]["balanced_frame_accuracy"]
            + 0.5 * gap
            + 0.1 * normal_validation["overall"]["target_selection_precision"]
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train": train_metrics["overall"],
                "validation": normal_validation["overall"],
                "reset_validation": reset_validation["overall"],
                "reversed_validation": reversed_validation["overall"],
                "shuffle_validation": shuffle_validation["overall"],
                "normal_minus_best_corrupted_balanced_frame_accuracy": float(gap),
            }
        )
        if score >= best_score:
            best_score = float(score)
            best_metrics = {
                "normal": normal_validation,
                "reset_recurrent_state": reset_validation,
                "reverse_input_history": reversed_validation,
                "shuffle_hidden_states": shuffle_validation,
            }
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            overall = normal_validation["overall"]
            print(
                f"epoch={epoch}"
                f" loss={train_loss:.4f}"
                f" val_bal={overall['balanced_frame_accuracy']:.3f}"
                f" val_recall={overall['positive_frame_recall']:.3f}"
                f" val_abstain={overall['negative_frame_abstain_specificity']:.3f}"
                f" val_precision={overall['target_selection_precision']:.3f}"
                f" gap={gap:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_ablations = {
        ablation: _evaluate(
            model,
            validation_sequences,
            device=device,
            selection_margin=float(args.selection_margin),
            ablation=ablation,
        )
        for ablation in (
            "normal",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }
    normal = validation_ablations["normal"]["overall"]
    corrupted_best = max(
        validation_ablations["reset_recurrent_state"]["overall"][
            "balanced_frame_accuracy"
        ],
        validation_ablations["reverse_input_history"]["overall"][
            "balanced_frame_accuracy"
        ],
        validation_ablations["shuffle_hidden_states"]["overall"][
            "balanced_frame_accuracy"
        ],
    )
    final_train = _evaluate(
        model,
        train_sequences,
        device=device,
        selection_margin=float(args.selection_margin),
        ablation="normal",
    )

    checkpoint = {
        "schema": "lewm_go2_frozen_jepa_target_gate_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "frozen_jepa_report": {
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
            "scrubbed_command_aux": bool(jepa_checkpoint.get("scrubbed_command_aux", False)),
        },
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "aux_dim": int(aux_dim),
        "query_dim": int(query_dim),
        "selection_margin": float(args.selection_margin),
        "scrubbed_command_aux": bool(args.scrub_command_aux),
        "scrubbed_runtime_aux": bool(args.scrub_runtime_aux),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_frozen_jepa_target_gate_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "selection_margin": float(args.selection_margin),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(sequence) for sequence in train_sequences.values()),
        "validation_row_count": sum(
            len(sequence) for sequence in validation_sequences.values()
        ),
        "train_frame_label_counts": train_counts,
        "validation_frame_label_counts": validation_counts,
        "final_train": final_train,
        "validation_ablations": validation_ablations,
        "normal_minus_best_corrupted_balanced_frame_accuracy": (
            float(normal["balanced_frame_accuracy"]) - float(corrupted_best)
        ),
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Offline direct target-selection gate over rendered Go2 "
            "matched-current-view event slices. The visual encoder is a frozen "
            "Go2 JEPA-style latent substrate. This is not closed-loop Go2 "
            "navigation."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_frozen_jepa_target_gate:"
        f" output={args.output}"
        f" report={report_path}"
        f" frame_bal={normal['balanced_frame_accuracy']:.3f}"
        f" pos_recall={normal['positive_frame_recall']:.3f}"
        f" neg_abstain={normal['negative_frame_abstain_specificity']:.3f}"
        f" precision={normal['target_selection_precision']:.3f}"
        f" delta={report['normal_minus_best_corrupted_balanced_frame_accuracy']:.3f}"
    )
    return 0


def _train_epoch(
    model: DirectGo2TargetGate,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    positive_frame_weight: float,
    negative_frame_weight: float,
    reset_abstain_loss_weight: float,
    shuffle_abstain_loss_weight: float,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    trained = 0
    for key in keys:
        sequence = sequences[key]
        images, aux = _sequence_tensors(sequence, device=device)
        hidden = model.forward_hidden(images, aux)
        reset_hidden = (
            model.forward_hidden(images, aux, reset_each_step=True)
            if float(reset_abstain_loss_weight) > 0.0
            else None
        )
        shuffled_hidden = None
        if float(shuffle_abstain_loss_weight) > 0.0 and hidden.shape[0] > 1:
            shuffled_hidden = torch.roll(hidden, shifts=max(1, hidden.shape[0] // 2), dims=0)
        losses = []
        for step_idx, frame in enumerate(sequence):
            candidate_batch = _candidate_batch(frame, device=device)
            if candidate_batch is None:
                continue
            query_features, _object_ids, positive_mask = candidate_batch
            abstain_logit, candidate_logits = model.score_candidates(
                hidden[step_idx],
                query_features,
            )
            loss = _frame_gate_loss(
                abstain_logit,
                candidate_logits,
                positive_mask,
                positive_frame_weight=positive_frame_weight,
                negative_frame_weight=negative_frame_weight,
            )
            losses.append(loss)
            if reset_hidden is not None:
                reset_abstain_logit, reset_candidate_logits = model.score_candidates(
                    reset_hidden[step_idx],
                    query_features,
                )
                losses.append(
                    _abstain_loss(reset_abstain_logit, reset_candidate_logits)
                    * float(reset_abstain_loss_weight)
                )
            if shuffled_hidden is not None:
                shuffle_abstain_logit, shuffle_candidate_logits = model.score_candidates(
                    shuffled_hidden[step_idx],
                    query_features,
                )
                losses.append(
                    _abstain_loss(shuffle_abstain_logit, shuffle_candidate_logits)
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


def _frame_gate_loss(
    abstain_logit: torch.Tensor,
    candidate_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    positive_frame_weight: float,
    negative_frame_weight: float,
) -> torch.Tensor:
    all_logits = torch.cat([abstain_logit.reshape(1), candidate_logits], dim=0)
    if bool(positive_mask.any().detach().cpu()):
        positive_logits = candidate_logits[positive_mask]
        loss = torch.logsumexp(all_logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
        return loss * float(positive_frame_weight)
    target = torch.zeros((), dtype=torch.long, device=all_logits.device)
    return F.cross_entropy(all_logits.reshape(1, -1), target.reshape(1)) * float(
        negative_frame_weight
    )


def _abstain_loss(
    abstain_logit: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    all_logits = torch.cat([abstain_logit.reshape(1), candidate_logits], dim=0)
    target = torch.zeros((), dtype=torch.long, device=all_logits.device)
    return F.cross_entropy(all_logits.reshape(1, -1), target.reshape(1))


def _evaluate(
    model: DirectGo2TargetGate,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    selection_margin: float,
    ablation: str,
) -> dict[str, Any]:
    model.eval()
    metrics = _FrameMetrics()
    by_positive_color: dict[str, _FrameMetrics] = defaultdict(_FrameMetrics)
    by_selected_color: dict[str, _SelectionCounter] = defaultdict(_SelectionCounter)
    with torch.no_grad():
        hidden_by_key = _hidden_states_by_sequence(
            model,
            sequences,
            device=device,
            ablation=ablation,
        )
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                candidate_batch = _candidate_batch(frame, device=device)
                if candidate_batch is None:
                    continue
                query_features, object_ids, positive_mask = candidate_batch
                abstain_logit, candidate_logits = model.score_candidates(
                    hidden[step_idx],
                    query_features,
                )
                selected_index = _select_index(
                    abstain_logit,
                    candidate_logits,
                    selection_margin=selection_margin,
                )
                selected_object = None if selected_index is None else object_ids[selected_index]
                positive_objects = {
                    object_id
                    for object_id, is_positive in zip(
                        object_ids,
                        positive_mask.detach().cpu().numpy(),
                    )
                    if bool(is_positive)
                }
                frame_result = metrics.add(
                    selected_object=selected_object,
                    positive_objects=positive_objects,
                )
                selected_color = (
                    "abstain" if selected_object is None else _object_color(selected_object)
                )
                by_selected_color[selected_color].add(frame_result)
                positive_colors = {
                    _object_color(object_id) for object_id in positive_objects
                }
                for color in sorted(positive_colors) or ["none"]:
                    by_positive_color[color].add_result(frame_result)
    return {
        "ablation": ablation,
        "overall": metrics.to_dict(),
        "by_positive_target_color": {
            key: value.to_dict() for key, value in sorted(by_positive_color.items())
        },
        "by_selected_color": {
            key: value.to_dict() for key, value in sorted(by_selected_color.items())
        },
    }


def _hidden_states_by_sequence(
    model: DirectGo2TargetGate,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    ablation: str,
) -> dict[tuple[str, int, int], torch.Tensor]:
    hidden_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            images, aux = _sequence_tensors(sequence, device=device)
            if ablation == "normal":
                hidden = model.forward_hidden(images, aux)
            elif ablation == "reset_recurrent_state":
                hidden = model.forward_hidden(images, aux, reset_each_step=True)
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                hidden = model.forward_hidden(images[order], aux[order]).flip(0)
            elif ablation == "shuffle_hidden_states":
                hidden = model.forward_hidden(images, aux)
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
        start = cursor
        flat_hidden.append(hidden)
        cursor += int(hidden.shape[0])
        spans[key] = (start, cursor)
    if cursor <= 1:
        return hidden_by_key
    flat = torch.cat(flat_hidden, dim=0)
    shuffled = torch.roll(flat, shifts=max(1, cursor // 2), dims=0)
    return {key: shuffled[start:end] for key, (start, end) in spans.items()}


def _candidate_batch(
    frame: Frame,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], torch.Tensor] | None:
    by_object: dict[str, dict[str, Any]] = {}
    for query in frame.queries:
        if not _current_role(query):
            continue
        item = by_object.setdefault(
            query.object_id,
            {"features": query.features, "target": 0.0},
        )
        item["target"] = max(float(item["target"]), float(query.target))
    if not by_object:
        return None
    object_ids = sorted(by_object)
    query_features = torch.stack([by_object[object_id]["features"] for object_id in object_ids])
    positive_mask = torch.tensor(
        [float(by_object[object_id]["target"]) >= 0.5 for object_id in object_ids],
        dtype=torch.bool,
    )
    return query_features.to(device), object_ids, positive_mask.to(device)


def _select_index(
    abstain_logit: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    selection_margin: float,
) -> int | None:
    best_score, best_index = candidate_logits.max(dim=0)
    if float(best_score.detach().cpu()) <= float(abstain_logit.detach().cpu()) + float(
        selection_margin
    ):
        return None
    return int(best_index.detach().cpu())


def _frame_label_counts(
    sequences: dict[tuple[str, int, int], list[Frame]],
) -> dict[str, int]:
    positive = 0
    negative = 0
    by_color: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    for sequence in sequences.values():
        for frame in sequence:
            candidate_batch = _candidate_batch(frame, device=torch.device("cpu"))
            if candidate_batch is None:
                continue
            _features, object_ids, positive_mask = candidate_batch
            if bool(positive_mask.any()):
                positive += 1
                colors = {
                    _object_color(object_id)
                    for object_id, is_positive in zip(object_ids, positive_mask.numpy())
                    if bool(is_positive)
                }
                for color in colors:
                    by_color[color]["positive"] += 1
            else:
                negative += 1
                for object_id in object_ids:
                    by_color[_object_color(object_id)]["negative"] += 1
    return {
        "positive": positive,
        "negative": negative,
        "total": positive + negative,
        "by_color": {key: dict(value) for key, value in sorted(by_color.items())},
    }


def _object_color(object_id: str) -> str:
    for color in ("red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"):
        if object_id.endswith(f"_{color}") or f"_{color}_" in object_id:
            return color
    return "unknown"


class _FrameMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.correct_positive = 0
        self.correct_negative = 0
        self.wrong_object = 0
        self.false_claim = 0
        self.missed_positive = 0
        self.selected = 0

    def add(self, *, selected_object: str | None, positive_objects: set[str]) -> str:
        selected = selected_object is not None
        if selected:
            self.selected += 1
        if positive_objects:
            self.positive_frames += 1
            if selected_object in positive_objects:
                self.correct_positive += 1
                return "correct_positive"
            if selected_object is None:
                self.missed_positive += 1
                return "missed_positive"
            self.wrong_object += 1
            return "wrong_object"
        self.negative_frames += 1
        if selected:
            self.false_claim += 1
            return "false_claim"
        self.correct_negative += 1
        return "correct_negative"

    def add_result(self, result: str) -> None:
        if result == "correct_positive":
            self.positive_frames += 1
            self.correct_positive += 1
            self.selected += 1
        elif result == "missed_positive":
            self.positive_frames += 1
            self.missed_positive += 1
        elif result == "wrong_object":
            self.positive_frames += 1
            self.wrong_object += 1
            self.selected += 1
        elif result == "false_claim":
            self.negative_frames += 1
            self.false_claim += 1
            self.selected += 1
        elif result == "correct_negative":
            self.negative_frames += 1
            self.correct_negative += 1
        else:
            raise ValueError(f"unknown frame result: {result}")

    def to_dict(self) -> dict[str, float]:
        frame_count = self.positive_frames + self.negative_frames
        positive_recall = self.correct_positive / max(1, self.positive_frames)
        negative_specificity = self.correct_negative / max(1, self.negative_frames)
        precision = self.correct_positive / max(1, self.selected)
        return {
            "frame_count": float(frame_count),
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "positive_frame_recall": positive_recall,
            "negative_frame_abstain_specificity": negative_specificity,
            "balanced_frame_accuracy": 0.5 * (positive_recall + negative_specificity),
            "frame_accuracy": (self.correct_positive + self.correct_negative)
            / max(1, frame_count),
            "target_selection_precision": precision,
            "selected_frame_count": float(self.selected),
            "correct_positive_count": float(self.correct_positive),
            "correct_negative_count": float(self.correct_negative),
            "wrong_object_count": float(self.wrong_object),
            "false_claim_count": float(self.false_claim),
            "missed_positive_count": float(self.missed_positive),
        }


class _SelectionCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, result: str) -> None:
        self.counts[result] += 1

    def to_dict(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
