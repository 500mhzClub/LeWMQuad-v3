#!/usr/bin/env python3
"""Train a query-conditioned Go2 causal memory probe.

The fixed-slot probe answers "what landmarks are in memory?" with one output
per landmark id. This variant answers "has this queried landmark been seen
before this current hidden view?" so object identity transfer can be tested
separately from recurrent memory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_go2_hidden_target_memory_probe import (
    _aux_features,
    _feature_stats,
    _load_image,
    _load_rows,
    _primitive_vocab,
    _resolve_device,
    _seq_key,
)
from lewm.models.go2_jepa import load_go2_jepa_encoder


_COLOR_RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "purple": (0.5, 0.0, 1.0),
    "unknown": (0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class Query:
    object_id: str
    color: str
    role: str
    target: float
    features: torch.Tensor


@dataclass(frozen=True)
class Frame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    queries: tuple[Query, ...]


class QueryConditionedGo2MemoryProbe(nn.Module):
    def __init__(
        self,
        aux_dim: int,
        query_dim: int,
        hidden_dim: int,
        *,
        encoder: nn.Module | None = None,
        encoder_output_dim: int | None = None,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        if encoder is None:
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
            encoder_output_dim = int(hidden_dim)
        else:
            self.encoder = encoder
            encoder_output_dim = int(encoder_output_dim or hidden_dim)
        self.encoder_projection = (
            nn.Identity()
            if int(encoder_output_dim) == int(hidden_dim)
            else nn.Sequential(
                nn.Linear(int(encoder_output_dim), int(hidden_dim)),
                nn.ReLU(inplace=True),
            )
        )
        self.gru = nn.GRUCell(hidden_dim + aux_dim, hidden_dim)
        self.query_head = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_hidden(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        reset_each_step: bool = False,
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
            h = self.gru(torch.cat([encoded[idx], aux[idx]], dim=-1), h)
            hidden_states.append(h)
        return torch.stack(hidden_states)

    def score_queries(self, hidden: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        return self.query_head(torch.cat([hidden, query_features], dim=-1)).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="*", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--train-query-roles",
        choices=("current", "current_and_evidence"),
        default="current_and_evidence",
    )
    parser.add_argument(
        "--positive-weight",
        default="auto",
        help="Positive BCE weight, or 'auto' for capped negative/positive ratio.",
    )
    parser.add_argument("--max-auto-positive-weight", type=float, default=8.0)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-object-slot",
        action="store_true",
        help="Add parsed landmark ordinal features to the query.",
    )
    parser.add_argument(
        "--include-privileged-landmark-geometry",
        action="store_true",
        help="Diagnostic only: add label-derived range/bearing/distance to the query.",
    )
    parser.add_argument(
        "--scrub-command-aux",
        action="store_true",
        help="Zero current command fields before building aux features.",
    )
    parser.add_argument(
        "--scrub-scene-aux",
        action="store_true",
        help=(
            "Zero scene-derived clearance/traversability aux while keeping "
            "executed command/action history."
        ),
    )
    parser.add_argument(
        "--frozen-jepa-checkpoint",
        type=Path,
        default=None,
        help="Use a frozen Go2 JEPA encoder checkpoint instead of training a CNN encoder.",
    )
    parser.add_argument(
        "--finetune-jepa-encoder",
        action="store_true",
        help="Initialize from --frozen-jepa-checkpoint but train the JEPA encoder.",
    )
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows_raw = _load_rows(args.datasets)
    validation_rows_raw = (
        _load_rows(args.validation_datasets) if args.validation_datasets else None
    )
    train_rows = _rows_for_aux(args, train_rows_raw)
    validation_rows = (
        _rows_for_aux(args, validation_rows_raw)
        if validation_rows_raw is not None
        else validation_rows_raw
    )
    if validation_rows is None:
        train_rows, validation_rows = _split_rows_by_sequence(train_rows)

    primitive_vocab = _primitive_vocab(train_rows_raw, validation_rows_raw or [])
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
        include_privileged_landmark_geometry=bool(args.include_privileged_landmark_geometry),
    )
    validation_sequences = _build_frames(
        validation_rows,
        primitive_vocab=primitive_vocab,
        color_vocab=color_vocab,
        max_slot=max_slot,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
        include_object_slot=bool(args.include_object_slot),
        include_privileged_landmark_geometry=bool(args.include_privileged_landmark_geometry),
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    train_role_filter = _role_filter(str(args.train_query_roles))
    train_label_counts = _query_label_counts(train_sequences, train_role_filter)
    if train_label_counts["positive"] <= 0 or train_label_counts["negative"] <= 0:
        raise SystemExit(f"train split lacks both labels for selected roles: {train_label_counts}")
    positive_weight = _resolve_positive_weight(
        str(args.positive_weight),
        train_label_counts=train_label_counts,
        max_auto_positive_weight=float(args.max_auto_positive_weight),
    )

    device = _resolve_device(str(args.device))
    aux_dim = next(iter(train_sequences.values()))[0].aux.numel()
    query_dim = next(
        query.features.numel()
        for sequence in train_sequences.values()
        for frame in sequence
        for query in frame.queries
    )
    frozen_jepa_report: dict[str, Any] | None = None
    jepa_encoder: nn.Module | None = None
    jepa_encoder_dim: int | None = None
    if args.frozen_jepa_checkpoint is not None:
        jepa_encoder, jepa_checkpoint = load_go2_jepa_encoder(
            args.frozen_jepa_checkpoint,
            device=device,
            freeze=not bool(args.finetune_jepa_encoder),
        )
        jepa_encoder_dim = int(jepa_checkpoint.get("latent_dim", 0))
        frozen_jepa_report = {
            "checkpoint": str(args.frozen_jepa_checkpoint),
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": jepa_encoder_dim,
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
            "scrubbed_command_aux": bool(jepa_checkpoint.get("scrubbed_command_aux", False)),
        }

    model = QueryConditionedGo2MemoryProbe(
        aux_dim=aux_dim,
        query_dim=query_dim,
        hidden_dim=int(args.hidden_dim),
        encoder=jepa_encoder,
        encoder_output_dim=jepa_encoder_dim,
        freeze_encoder=(
            args.frozen_jepa_checkpoint is not None and not bool(args.finetune_jepa_encoder)
        ),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )

    history = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            role_filter=train_role_filter,
            positive_weight=positive_weight,
            evidence_loss_weight=float(args.evidence_loss_weight),
        )
        train_metrics = _evaluate(
            model,
            train_sequences,
            device=device,
            threshold=float(args.threshold),
        )
        validation_metrics = _evaluate(
            model,
            validation_sequences,
            device=device,
            threshold=float(args.threshold),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train": train_metrics["overall"],
                "validation": validation_metrics["overall"],
            }
        )
        score = float(validation_metrics["overall"]["balanced_accuracy"])
        if score >= best_score:
            best_score = score
            best_metrics = validation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            overall = validation_metrics["overall"]
            print(
                f"epoch={epoch}"
                f" train_loss={train_loss:.4f}"
                f" val_bal={overall['balanced_accuracy']:.3f}"
                f" val_recall={overall['positive_recall']:.3f}"
                f" val_spec={overall['negative_specificity']:.3f}"
                f" val_f1={overall['f1']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = _evaluate(
        model,
        train_sequences,
        device=device,
        threshold=float(args.threshold),
    )
    final_validation = _evaluate(
        model,
        validation_sequences,
        device=device,
        threshold=float(args.threshold),
    )
    final_validation_reset = _evaluate(
        model,
        validation_sequences,
        device=device,
        threshold=float(args.threshold),
        ablation="reset_recurrent_state",
    )
    final_validation_reversed = _evaluate(
        model,
        validation_sequences,
        device=device,
        threshold=float(args.threshold),
        ablation="reverse_input_history",
    )

    checkpoint = {
        "schema": "lewm_go2_causal_memory_query_probe_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "aux_dim": int(aux_dim),
        "query_dim": int(query_dim),
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "scrubbed_command_aux": bool(args.scrub_command_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    normal_bal = float(final_validation["overall"]["balanced_accuracy"])
    reset_bal = float(final_validation_reset["overall"]["balanced_accuracy"])
    reversed_bal = float(final_validation_reversed["overall"]["balanced_accuracy"])
    report = {
        "schema": "lewm_go2_causal_memory_query_probe_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": (
            [str(path) for path in args.validation_datasets]
            if args.validation_datasets
            else []
        ),
        "output": str(args.output),
        "device": str(device),
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(seq) for seq in train_sequences.values()),
        "validation_row_count": sum(len(seq) for seq in validation_sequences.values()),
        "train_query_label_counts": train_label_counts,
        "validation_current_query_label_counts": _query_label_counts(
            validation_sequences,
            _current_role,
        ),
        "train_query_counts_by_object": _query_counts_by(
            train_sequences,
            train_role_filter,
            key_fn=lambda query: query.object_id,
        ),
        "validation_current_query_counts_by_object": _query_counts_by(
            validation_sequences,
            _current_role,
            key_fn=lambda query: query.object_id,
        ),
        "train_query_counts_by_color": _query_counts_by(
            train_sequences,
            train_role_filter,
            key_fn=lambda query: query.color,
        ),
        "validation_current_query_counts_by_color": _query_counts_by(
            validation_sequences,
            _current_role,
            key_fn=lambda query: query.color,
        ),
        "positive_weight": positive_weight,
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "scrubbed_command_aux": bool(args.scrub_command_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "final_train": final_train,
        "final_validation": final_validation,
        "final_validation_reset_recurrent_state": final_validation_reset,
        "final_validation_reverse_input_history": final_validation_reversed,
        "normal_minus_best_ablation_balanced_accuracy": normal_bal - max(reset_bal, reversed_bal),
        "best_validation_selection_score": best_score,
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "This is still a supervised probe over matched-current-view Go2 "
            "event slices. If frozen_jepa_checkpoint is set, the visual encoder "
            "is a fixed JEPA-style latent substrate and only the recurrent memory "
            "query readout is trained. Passing it supports Go2 memory "
            "translatability; it does not prove closed-loop robot navigation."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_causal_memory_query_probe:"
        f" output={args.output}"
        f" report={report_path}"
        f" val_bal={normal_bal:.3f}"
        f" reset_bal={reset_bal:.3f}"
        f" reversed_bal={reversed_bal:.3f}"
        f" delta={report['normal_minus_best_ablation_balanced_accuracy']:.3f}"
    )
    return 0


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


def _scrub_command_aux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed = []
    for row in rows:
        item = dict(row)
        command = dict(item.get("command") or {})
        command["primitive_name"] = ""
        command["vx_body_mps"] = []
        command["vy_body_mps"] = []
        command["yaw_rate_radps"] = []
        item["command"] = command
        scrubbed.append(item)
    return scrubbed


def _rows_for_aux(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if bool(args.scrub_command_aux):
        return _scrub_command_aux(rows)
    if bool(args.scrub_scene_aux):
        return _scrub_scene_aux(rows)
    return rows


def _scrub_scene_aux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed = []
    for row in rows:
        item = dict(row)
        item["clearance_m"] = 0.0
        item["traversability_forward_m"] = 0.0
        scrubbed.append(item)
    return scrubbed


def _scrub_runtime_aux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove aux fields unavailable to a deployed RGB+egomotion controller."""

    scrubbed = _scrub_command_aux(rows)
    for item in scrubbed:
        item["clearance_m"] = 0.0
        item["traversability_forward_m"] = 0.0
    return scrubbed


def _build_frames(
    rows: list[dict[str, Any]],
    *,
    primitive_vocab: list[str],
    color_vocab: list[str],
    max_slot: int,
    feature_stats: dict[str, np.ndarray],
    image_size: int,
    include_object_slot: bool,
    include_privileged_landmark_geometry: bool,
) -> dict[tuple[str, int, int], list[Frame]]:
    sequences: dict[tuple[str, int, int], list[Frame]] = {}
    for row in rows:
        aux = _aux_features(row, primitive_vocab=primitive_vocab)
        aux = (aux - feature_stats["mean"]) / feature_stats["std"]
        seen_events: set[tuple[str, str, float]] = set()
        queries = []
        for event in row.get("go2_causal_memory_pair_selection", ()):
            role = str(event.get("pair_role", ""))
            if not (role.startswith("current_") or role == "first_visible_evidence"):
                continue
            object_id = str(event.get("object_id", ""))
            if not object_id:
                continue
            target = 1.0 if bool(event.get("seen_before", False)) else 0.0
            dedup_key = (role, object_id, target)
            if dedup_key in seen_events:
                continue
            seen_events.add(dedup_key)
            color = _object_color(object_id)
            queries.append(
                Query(
                    object_id=object_id,
                    color=color,
                    role=role,
                    target=target,
                    features=torch.tensor(
                        _query_features(
                            row,
                            object_id=object_id,
                            color_vocab=color_vocab,
                            max_slot=max_slot,
                            include_object_slot=include_object_slot,
                            include_privileged_landmark_geometry=(
                                include_privileged_landmark_geometry
                            ),
                        ),
                        dtype=torch.float32,
                    ),
                )
            )
        frame = Frame(
            seq_key=_seq_key(row),
            episode_step=int(row.get("episode_step", 0)),
            image=_load_image(Path(row["rgb_path"]), image_size=image_size),
            aux=torch.tensor(aux, dtype=torch.float32),
            queries=tuple(queries),
        )
        sequences.setdefault(frame.seq_key, []).append(frame)
    for sequence in sequences.values():
        sequence.sort(key=lambda item: item.episode_step)
    return sequences


def _query_features(
    row: dict[str, Any],
    *,
    object_id: str,
    color_vocab: list[str],
    max_slot: int,
    include_object_slot: bool,
    include_privileged_landmark_geometry: bool,
) -> list[float]:
    color = _object_color(object_id)
    features = list(_COLOR_RGB.get(color, _COLOR_RGB["unknown"]))
    features.extend(1.0 if color == item else 0.0 for item in color_vocab)
    if include_object_slot:
        slot = _landmark_slot(object_id)
        slot_norm = float(slot) / float(max(1, max_slot))
        features.extend([slot_norm, math.sin(slot), math.cos(slot)])
    if include_privileged_landmark_geometry:
        landmark = _landmark_by_id(row).get(object_id, {})
        bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
        bfs_distance = _finite_float(landmark.get("bfs_distance_cells"), -1.0)
        range_m = _finite_float(landmark.get("range_m"), 0.0)
        features.extend(
            [
                range_m / 12.0,
                math.sin(bearing),
                math.cos(bearing),
                bfs_distance / 32.0,
                1.0 if bool(landmark.get("visible", False)) else 0.0,
            ]
        )
    return features


def _train_epoch(
    model: QueryConditionedGo2MemoryProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    role_filter: Any,
    positive_weight: float,
    evidence_loss_weight: float,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    trained_sequences = 0
    for key in keys:
        sequence = sequences[key]
        images, aux = _sequence_tensors(sequence, device=device)
        hidden = model.forward_hidden(images, aux)
        batch = _query_batch(
            model,
            sequence,
            hidden,
            device=device,
            role_filter=role_filter,
            evidence_loss_weight=evidence_loss_weight,
        )
        if batch is None:
            continue
        logits, targets, weights = batch
        loss = _weighted_bce(logits, targets, weights, positive_weight=positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        trained_sequences += 1
    return total_loss / max(1, trained_sequences)


def _evaluate(
    model: QueryConditionedGo2MemoryProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    threshold: float,
    ablation: str = "normal",
) -> dict[str, Any]:
    model.eval()
    overall = _MetricAccumulator()
    by_object: dict[str, _MetricAccumulator] = defaultdict(_MetricAccumulator)
    by_color: dict[str, _MetricAccumulator] = defaultdict(_MetricAccumulator)
    losses = []
    with torch.no_grad():
        for sequence in sequences.values():
            images, aux = _sequence_tensors(sequence, device=device)
            if ablation == "normal":
                hidden = model.forward_hidden(images, aux)
            elif ablation == "reset_recurrent_state":
                hidden = model.forward_hidden(images, aux, reset_each_step=True)
            elif ablation == "reverse_input_history":
                order = torch.arange(images.shape[0] - 1, -1, -1, device=device)
                hidden = model.forward_hidden(images[order], aux[order]).flip(0)
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            batch = _query_batch(
                model,
                sequence,
                hidden,
                device=device,
                role_filter=_current_role,
                evidence_loss_weight=1.0,
            )
            if batch is None:
                continue
            logits, targets, weights = batch
            losses.append(float(_weighted_bce(logits, targets, weights, positive_weight=1.0).cpu()))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            target_values = targets.detach().cpu().numpy()
            query_index = 0
            for frame in sequence:
                for query in frame.queries:
                    if not _current_role(query):
                        continue
                    prediction = 1.0 if float(probs[query_index]) >= threshold else 0.0
                    target = float(target_values[query_index])
                    overall.add(prediction=prediction, target=target)
                    by_object[query.object_id].add(prediction=prediction, target=target)
                    by_color[query.color].add(prediction=prediction, target=target)
                    query_index += 1
    return {
        "overall": overall.to_dict(loss=float(np.mean(losses)) if losses else 0.0),
        "by_object": {
            key: accumulator.to_dict()
            for key, accumulator in sorted(by_object.items())
        },
        "by_color": {
            key: accumulator.to_dict()
            for key, accumulator in sorted(by_color.items())
        },
        "ablation": ablation,
    }


def _query_batch(
    model: QueryConditionedGo2MemoryProbe,
    sequence: list[Frame],
    hidden: torch.Tensor,
    *,
    device: torch.device,
    role_filter: Any,
    evidence_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    hidden_rows = []
    query_features = []
    targets = []
    weights = []
    for step_idx, frame in enumerate(sequence):
        for query in frame.queries:
            if not role_filter(query):
                continue
            hidden_rows.append(hidden[step_idx])
            query_features.append(query.features)
            targets.append(float(query.target))
            weights.append(
                float(evidence_loss_weight)
                if query.role == "first_visible_evidence"
                else 1.0
            )
    if not hidden_rows:
        return None
    hidden_tensor = torch.stack(hidden_rows).to(device)
    query_tensor = torch.stack(query_features).to(device)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    logits = model.score_queries(hidden_tensor, query_tensor)
    return logits, target_tensor, weight_tensor


def _weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    positive_weight: float,
) -> torch.Tensor:
    pos_weight = torch.tensor(float(positive_weight), dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _sequence_tensors(
    sequence: list[Frame],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([item.image for item in sequence]).to(device),
        torch.stack([item.aux for item in sequence]).to(device),
    )


def _role_filter(name: str) -> Any:
    if name == "current":
        return _current_role
    if name == "current_and_evidence":
        return lambda query: _current_role(query) or query.role == "first_visible_evidence"
    raise ValueError(f"unknown role filter: {name}")


def _current_role(query: Query) -> bool:
    return query.role.startswith("current_")


def _resolve_positive_weight(
    value: str,
    *,
    train_label_counts: dict[str, int],
    max_auto_positive_weight: float,
) -> float:
    if value != "auto":
        return float(value)
    ratio = float(train_label_counts["negative"]) / max(1.0, float(train_label_counts["positive"]))
    return float(min(max_auto_positive_weight, max(1.0, ratio)))


def _query_label_counts(
    sequences: dict[tuple[str, int, int], list[Frame]],
    role_filter: Any,
) -> dict[str, int]:
    positive = 0
    negative = 0
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                if not role_filter(query):
                    continue
                if query.target >= 0.5:
                    positive += 1
                else:
                    negative += 1
    return {"positive": positive, "negative": negative, "total": positive + negative}


def _query_counts_by(
    sequences: dict[tuple[str, int, int], list[Frame]],
    role_filter: Any,
    *,
    key_fn: Any,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                if not role_filter(query):
                    continue
                key = str(key_fn(query))
                label = "positive" if query.target >= 0.5 else "negative"
                counts[key][label] += 1
                counts[key]["total"] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


class _MetricAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, *, prediction: float, target: float) -> None:
        pred = prediction >= 0.5
        truth = target >= 0.5
        if pred and truth:
            self.tp += 1
        elif pred and not truth:
            self.fp += 1
        elif not pred and truth:
            self.fn += 1
        else:
            self.tn += 1

    def to_dict(self, *, loss: float | None = None) -> dict[str, float]:
        positive_total = self.tp + self.fn
        negative_total = self.tn + self.fp
        total = positive_total + negative_total
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, positive_total)
        specificity = self.tn / max(1, negative_total)
        f1 = (2.0 * self.tp) / max(1, 2 * self.tp + self.fp + self.fn)
        result = {
            "accuracy": (self.tp + self.tn) / max(1, total),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "positive_recall": recall,
            "negative_specificity": specificity,
            "precision": precision,
            "f1": f1,
            "true_positive_count": float(self.tp),
            "true_negative_count": float(self.tn),
            "false_positive_count": float(self.fp),
            "false_negative_count": float(self.fn),
            "positive_count": float(positive_total),
            "negative_count": float(negative_total),
            "target_count": float(total),
        }
        if loss is not None:
            result["loss"] = float(loss)
        return result


def _color_vocab(*row_groups: list[dict[str, Any]]) -> list[str]:
    colors = {
        _object_color(str(landmark.get("object_id", "")))
        for rows in row_groups
        for row in rows
        for landmark in row.get("landmarks", ())
    }
    colors.discard("")
    return sorted(colors)


def _object_color(object_id: str) -> str:
    for color in _COLOR_RGB:
        if object_id.endswith(f"_{color}") or f"_{color}_" in object_id:
            return color
    return "unknown"


def _landmark_slot(object_id: str) -> int:
    parts = object_id.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def _max_landmark_slot(*row_groups: list[dict[str, Any]]) -> int:
    slots = [
        _landmark_slot(str(landmark.get("object_id", "")))
        for rows in row_groups
        for row in rows
        for landmark in row.get("landmarks", ())
    ]
    return max(slots) if slots else 0


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): dict(landmark)
        for landmark in row.get("landmarks", ())
    }


def _finite_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    result = float(value)
    if not math.isfinite(result):
        return float(default)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
