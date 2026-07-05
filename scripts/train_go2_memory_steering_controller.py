#!/usr/bin/env python3
"""Train a Go2 memory controller for target selection plus steering.

This is a stricter controller-facing probe than target geometry regression: it
optimizes the exact replay metric used for the hidden-target return bridge.
Given recurrent RGB memory and current candidate queries, the model either
abstains or selects one remembered target and predicts left/forward/right
target steering for that selected object.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
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
    Query,
    _build_frames,
    _color_vocab,
    _max_landmark_slot,
    _scrub_command_aux,
    _scrub_runtime_aux,
    _sequence_tensors,
)
from train_go2_frozen_jepa_target_gate import _candidate_batch, _select_index  # noqa: E402
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _feature_stats,
    _load_rows,
    _primitive_vocab,
    _resolve_device,
)


STEERING_CLASSES = ["right", "forward", "left"]


class Go2MemorySteeringController(nn.Module):
    def __init__(
        self,
        *,
        aux_dim: int,
        query_dim: int,
        hidden_dim: int,
        encoder: nn.Module | None = None,
        encoder_output_dim: int | None = None,
        freeze_encoder: bool = False,
        memory_slot_count: int = 0,
        temporal_memory_layers: int = 0,
        temporal_memory_heads: int = 4,
    ) -> None:
        super().__init__()
        self.freeze_encoder = bool(freeze_encoder)
        self.memory_slot_count = int(memory_slot_count)
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
            if self.freeze_encoder:
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
        self.temporal_memory = None
        if int(temporal_memory_layers) > 0:
            self.temporal_memory = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=int(hidden_dim),
                    nhead=max(1, int(temporal_memory_heads)),
                    dim_feedforward=int(hidden_dim) * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=int(temporal_memory_layers),
            )
        self.candidate_trunk = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(query_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
        )
        self.candidate_head = nn.Linear(int(hidden_dim) // 2, 1)
        self.steering_head = nn.Linear(int(hidden_dim) // 2, len(STEERING_CLASSES))
        self.abstain_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.memory_state_head = (
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
        stacked = torch.stack(hidden_states)
        if self.temporal_memory is None or stacked.shape[0] <= 1:
            return stacked
        causal_mask = torch.triu(
            torch.ones(
                stacked.shape[0],
                stacked.shape[0],
                dtype=torch.bool,
                device=stacked.device,
            ),
            diagonal=1,
        )
        return self.temporal_memory(stacked.unsqueeze(0), mask=causal_mask).squeeze(0)

    def score_candidates(
        self,
        hidden: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_rows = hidden.repeat(query_features.shape[0], 1)
        trunk = self.candidate_trunk(torch.cat([hidden_rows, query_features], dim=-1))
        candidate_logits = self.candidate_head(trunk).squeeze(-1)
        steering_logits = self.steering_head(trunk)
        abstain_logit = self.abstain_head(hidden).squeeze(-1)
        return abstain_logit, candidate_logits, steering_logits

    def score_memory_state(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.memory_state_head is None:
            raise RuntimeError("memory_state_head is disabled")
        return self.memory_state_head(hidden)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--finetune-jepa-encoder",
        action="store_true",
        help=(
            "Initialize the visual encoder from --frozen-jepa-checkpoint but "
            "train it under the memory-controller objective."
        ),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temporal-memory-layers", type=int, default=0)
    parser.add_argument("--temporal-memory-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--positive-frame-weight", type=float, default=1.0)
    parser.add_argument("--negative-frame-weight", type=float, default=1.0)
    parser.add_argument("--steering-loss-weight", type=float, default=1.0)
    parser.add_argument("--candidate-bce-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--memory-state-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary BCE loss that decodes the cumulative seen-landmark set "
            "from the post-frame recurrent state."
        ),
    )
    parser.add_argument(
        "--evidence-write-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary loss on first_visible_evidence queries using the hidden "
            "state after the current frame has been ingested."
        ),
    )
    parser.add_argument("--reset-abstain-loss-weight", type=float, default=0.0)
    parser.add_argument("--shuffle-abstain-loss-weight", type=float, default=0.0)
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--arc-threshold-rad", type=float, default=0.1)
    parser.add_argument("--yaw-threshold-rad", type=float, default=0.75)
    parser.add_argument("--hold-range-m", type=float, default=0.0)
    parser.add_argument("--include-object-slot", action="store_true")
    parser.add_argument(
        "--include-runtime-query-geometry",
        action="store_true",
        help=(
            "Append current object-relative range/bearing/visibility query features. "
            "This is a runtime-perception proxy, not a geodesic/privileged map feature."
        ),
    )
    parser.add_argument(
        "--include-runtime-memory-geometry",
        action="store_true",
        help=(
            "Append per-landmark runtime visible/range/bearing observations to "
            "the recurrent memory input. This assumes an object-detector front "
            "end, not a global map or future label."
        ),
    )
    parser.add_argument(
        "--exclusive-memory-state",
        action="store_true",
        help="Score queries from the recurrent state before the current frame is ingested.",
    )
    parser.add_argument("--scrub-command-aux", action="store_true")
    parser.add_argument("--scrub-runtime-aux", action="store_true")
    parser.add_argument(
        "--scrub-scene-aux",
        action="store_true",
        help=(
            "Zero scene-derived clearance/traversability aux while keeping "
            "executed command/action history. This preserves deployable action "
            "context without using map-like scene geometry."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260649)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--min-target-steering-success", type=float, default=0.90)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.12)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.30)
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
    train_rows = _rows_for_runtime(args, train_rows_raw)
    validation_rows = _rows_for_runtime(args, validation_rows_raw)

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
    train_index = _row_index(train_rows_raw)
    validation_index = _row_index(validation_rows_raw)
    if bool(args.include_runtime_memory_geometry):
        train_sequences = _append_runtime_memory_geometry(
            train_sequences,
            train_index,
            max_slot=max_slot,
        )
        validation_sequences = _append_runtime_memory_geometry(
            validation_sequences,
            validation_index,
            max_slot=max_slot,
        )
    if bool(args.include_runtime_query_geometry):
        train_sequences = _append_runtime_query_geometry(train_sequences, train_index)
        validation_sequences = _append_runtime_query_geometry(
            validation_sequences,
            validation_index,
        )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    device = _resolve_device(str(args.device))
    encoder = None
    encoder_dim = None
    frozen_jepa_report = None
    if args.frozen_jepa_checkpoint is not None:
        encoder, jepa_checkpoint = load_go2_jepa_encoder(
            args.frozen_jepa_checkpoint,
            device=device,
            freeze=not bool(args.finetune_jepa_encoder),
        )
        encoder_dim = int(jepa_checkpoint.get("latent_dim", args.hidden_dim))
        frozen_jepa_report = {
            "checkpoint": str(args.frozen_jepa_checkpoint),
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(encoder_dim),
            "finetuned": bool(args.finetune_jepa_encoder),
        }

    aux_dim = next(iter(train_sequences.values()))[0].aux.numel()
    query_dim = next(
        query.features.numel()
        for sequence in train_sequences.values()
        for frame in sequence
        for query in frame.queries
    )
    model = Go2MemorySteeringController(
        aux_dim=int(aux_dim),
        query_dim=int(query_dim),
        hidden_dim=int(args.hidden_dim),
        encoder=encoder,
        encoder_output_dim=encoder_dim,
        freeze_encoder=encoder is not None and not bool(args.finetune_jepa_encoder),
        memory_slot_count=(max_slot + 1 if float(args.memory_state_loss_weight) > 0.0 else 0),
        temporal_memory_layers=int(args.temporal_memory_layers),
        temporal_memory_heads=int(args.temporal_memory_heads),
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
            train_index,
            device=device,
            positive_frame_weight=float(args.positive_frame_weight),
            negative_frame_weight=float(args.negative_frame_weight),
            steering_loss_weight=float(args.steering_loss_weight),
            candidate_bce_loss_weight=float(args.candidate_bce_loss_weight),
            memory_state_loss_weight=float(args.memory_state_loss_weight),
            evidence_write_loss_weight=float(args.evidence_write_loss_weight),
            reset_abstain_loss_weight=float(args.reset_abstain_loss_weight),
            shuffle_abstain_loss_weight=float(args.shuffle_abstain_loss_weight),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
            include_current=not bool(args.exclusive_memory_state),
        )
        validation_ablations = {
            ablation: _evaluate(
                model,
                validation_sequences,
                validation_index,
                device=device,
                selection_margin=float(args.selection_margin),
                arc_threshold_rad=float(args.arc_threshold_rad),
                yaw_threshold_rad=float(args.yaw_threshold_rad),
                hold_range_m=float(args.hold_range_m),
                ablation=ablation,
                include_current=not bool(args.exclusive_memory_state),
            )
            for ablation in (
                "normal",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_hidden_states",
            )
        }
        score = _selection_score(validation_ablations)
        normal = validation_ablations["normal"]
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss),
                "validation": normal,
                "normal_minus_best_corrupted_target_steering_success": (
                    float(normal["target_steering_pipeline_success"])
                    - max(
                        float(validation_ablations[name]["target_steering_pipeline_success"])
                        for name in (
                            "reset_recurrent_state",
                            "reverse_input_history",
                            "shuffle_hidden_states",
                        )
                    )
                ),
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
                f" recall={normal['target_recall']:.3f}"
                f" false_claim={normal['false_claim_rate']:.3f}"
                f" target_steer={normal['target_steering_pipeline_success']:.3f}"
                f" precision={normal['target_selection_precision']:.3f}"
                f" score={score:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = _evaluate(
        model,
        train_sequences,
        train_index,
        device=device,
        selection_margin=float(args.selection_margin),
        arc_threshold_rad=float(args.arc_threshold_rad),
        yaw_threshold_rad=float(args.yaw_threshold_rad),
        hold_range_m=float(args.hold_range_m),
        ablation="normal",
        include_current=not bool(args.exclusive_memory_state),
    )
    validation_ablations = {
        ablation: _evaluate(
            model,
            validation_sequences,
            validation_index,
            device=device,
            selection_margin=float(args.selection_margin),
            arc_threshold_rad=float(args.arc_threshold_rad),
            yaw_threshold_rad=float(args.yaw_threshold_rad),
            hold_range_m=float(args.hold_range_m),
            ablation=ablation,
            include_current=not bool(args.exclusive_memory_state),
        )
        for ablation in (
            "normal",
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }
    normal = validation_ablations["normal"]
    corrupted_best = max(
        float(validation_ablations[name]["target_steering_pipeline_success"])
        for name in (
            "memory_off_abstain",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    )
    gap = float(normal["target_steering_pipeline_success"]) - corrupted_best
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and gap >= float(args.min_corrupted_gap)
    )

    checkpoint = {
        "schema": "lewm_go2_memory_steering_controller_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "temporal_memory_layers": int(args.temporal_memory_layers),
        "temporal_memory_heads": int(args.temporal_memory_heads),
        "aux_dim": int(aux_dim),
        "query_dim": int(query_dim),
        "memory_slot_count": (
            max_slot + 1 if float(args.memory_state_loss_weight) > 0.0 else 0
        ),
        "steering_classes": STEERING_CLASSES,
        "scrubbed_command_aux": bool(args.scrub_command_aux or args.scrub_runtime_aux),
        "scrubbed_runtime_aux": bool(args.scrub_runtime_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_memory_steering_controller_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "scrubbed_runtime_aux": bool(args.scrub_runtime_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "include_object_slot": bool(args.include_object_slot),
        "include_runtime_query_geometry": bool(args.include_runtime_query_geometry),
        "include_runtime_memory_geometry": bool(args.include_runtime_memory_geometry),
        "exclusive_memory_state": bool(args.exclusive_memory_state),
        "selection_margin": float(args.selection_margin),
        "temporal_memory_layers": int(args.temporal_memory_layers),
        "temporal_memory_heads": int(args.temporal_memory_heads),
        "config": {
            "arc_threshold_rad": float(args.arc_threshold_rad),
            "yaw_threshold_rad": float(args.yaw_threshold_rad),
            "hold_range_m": float(args.hold_range_m),
            "memory_state_loss_weight": float(args.memory_state_loss_weight),
            "evidence_write_loss_weight": float(args.evidence_write_loss_weight),
            "min_target_steering_success": float(args.min_target_steering_success),
            "max_false_claim_rate": float(args.max_false_claim_rate),
            "min_corrupted_gap": float(args.min_corrupted_gap),
        },
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(sequence) for sequence in train_sequences.values()),
        "validation_row_count": sum(len(sequence) for sequence in validation_sequences.values()),
        "final_train": final_train,
        "validation_ablations": validation_ablations,
        "normal_minus_best_corrupted_target_steering_pipeline_success": gap,
        "controller_gate_pass": bool(gate_pass),
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Offline strict-runtime Go2 memory steering controller. This directly "
            "optimizes selected-target steering over rendered event slices; it is "
            "not live Genesis physics execution."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_memory_steering_controller:"
        f" output={args.output}"
        f" report={report_path}"
        f" recall={normal['target_recall']:.3f}"
        f" false_claim={normal['false_claim_rate']:.3f}"
        f" target_steer={normal['target_steering_pipeline_success']:.3f}"
        f" delta={gap:.3f}"
        f" pass={bool(gate_pass)}"
    )
    return 0


def _rows_for_runtime(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if bool(args.scrub_runtime_aux):
        return _scrub_runtime_aux(rows)
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


def _append_runtime_query_geometry(
    sequences: dict[tuple[str, int, int], list[Frame]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
) -> dict[tuple[str, int, int], list[Frame]]:
    updated: dict[tuple[str, int, int], list[Frame]] = {}
    for key, sequence in sequences.items():
        updated_sequence = []
        for frame in sequence:
            row = row_index.get((frame.seq_key, int(frame.episode_step)), {})
            landmarks = _landmark_by_id(row)
            queries = []
            for query in frame.queries:
                landmark = landmarks.get(str(query.object_id), {})
                query_geometry = torch.tensor(
                    _runtime_query_geometry_features(landmark),
                    dtype=query.features.dtype,
                )
                queries.append(
                    Query(
                        object_id=query.object_id,
                        color=query.color,
                        role=query.role,
                        target=query.target,
                        features=torch.cat([query.features, query_geometry], dim=0),
                    )
                )
            updated_sequence.append(
                Frame(
                    seq_key=frame.seq_key,
                    episode_step=frame.episode_step,
                    image=frame.image,
                    aux=frame.aux,
                    queries=tuple(queries),
                )
            )
        updated[key] = updated_sequence
    return updated


def _append_runtime_memory_geometry(
    sequences: dict[tuple[str, int, int], list[Frame]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    max_slot: int,
) -> dict[tuple[str, int, int], list[Frame]]:
    updated: dict[tuple[str, int, int], list[Frame]] = {}
    for key, sequence in sequences.items():
        updated_sequence = []
        for frame in sequence:
            row = row_index.get((frame.seq_key, int(frame.episode_step)), {})
            memory_geometry = torch.tensor(
                _runtime_memory_geometry_features(row, max_slot=max_slot),
                dtype=frame.aux.dtype,
            )
            updated_sequence.append(
                Frame(
                    seq_key=frame.seq_key,
                    episode_step=frame.episode_step,
                    image=frame.image,
                    aux=torch.cat([frame.aux, memory_geometry], dim=0),
                    queries=frame.queries,
                )
            )
        updated[key] = updated_sequence
    return updated


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): landmark
        for landmark in row.get("landmarks", ())
        if str(landmark.get("object_id", ""))
    }


def _runtime_query_geometry_features(landmark: dict[str, Any]) -> list[float]:
    bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
    range_m = _finite_float(landmark.get("range_m"), 0.0)
    visible = 1.0 if bool(landmark.get("visible", False)) else 0.0
    return [
        range_m / 12.0,
        math.sin(bearing),
        math.cos(bearing),
        visible,
    ]


def _runtime_memory_geometry_features(row: dict[str, Any], *, max_slot: int) -> list[float]:
    features = [0.0] * (max(0, int(max_slot) + 1) * 4)
    for landmark in row.get("landmarks", ()):
        slot = _landmark_slot(str(landmark.get("object_id", "")))
        if slot is None or slot < 0 or slot > int(max_slot):
            continue
        bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
        range_m = _finite_float(landmark.get("range_m"), 0.0)
        offset = int(slot) * 4
        features[offset] = 1.0 if bool(landmark.get("visible", False)) else 0.0
        features[offset + 1] = range_m / 12.0
        features[offset + 2] = math.sin(bearing)
        features[offset + 3] = math.cos(bearing)
    return features


def _landmark_slot(object_id: str) -> int | None:
    parts = str(object_id).split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def _row_index(rows: list[dict[str, Any]]) -> dict[tuple[tuple[str, int, int], int], dict[str, Any]]:
    return {
        (
            (
                str(row.get("scene_id", "")),
                int(row.get("env_idx", 0)),
                int(row.get("episode_id", 0)),
            ),
            int(row.get("episode_step", 0)),
        ): row
        for row in rows
    }


def _train_epoch(
    model: Go2MemorySteeringController,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    device: torch.device,
    positive_frame_weight: float,
    negative_frame_weight: float,
    steering_loss_weight: float,
    candidate_bce_loss_weight: float,
    memory_state_loss_weight: float,
    evidence_write_loss_weight: float,
    reset_abstain_loss_weight: float,
    shuffle_abstain_loss_weight: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
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
        write_hidden = (
            model.forward_hidden(images, aux, include_current=True)
            if float(evidence_write_loss_weight) > 0.0 and not bool(include_current)
            else hidden
        )
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
            row = row_index.get((frame.seq_key, int(frame.episode_step)), {})
            if float(memory_state_loss_weight) > 0.0 and model.memory_state_head is not None:
                for object_id in row.get("visible_landmark_ids", ()):
                    slot = _landmark_slot(str(object_id))
                    if slot is not None and 0 <= slot < model.memory_slot_count:
                        seen_slots.add(int(slot))
                target = torch.zeros(
                    model.memory_slot_count,
                    dtype=write_hidden.dtype,
                    device=device,
                )
                for slot in seen_slots:
                    target[slot] = 1.0
                losses.append(
                    F.binary_cross_entropy_with_logits(
                        model.score_memory_state(write_hidden[step_idx]),
                        target,
                    )
                    * float(memory_state_loss_weight)
                )
            batch = _candidate_batch(frame, device=device)
            if batch is None:
                continue
            query_features, object_ids, positive_mask = batch
            steering_targets = _steering_targets(
                row,
                object_ids,
                arc_threshold_rad=arc_threshold_rad,
                yaw_threshold_rad=yaw_threshold_rad,
                hold_range_m=hold_range_m,
                device=device,
            )
            abstain_logit, candidate_logits, steering_logits = model.score_candidates(
                hidden[step_idx],
                query_features,
            )
            losses.append(
                _frame_loss(
                    abstain_logit,
                    candidate_logits,
                    steering_logits,
                    positive_mask,
                    steering_targets,
                    positive_frame_weight=positive_frame_weight,
                    negative_frame_weight=negative_frame_weight,
                    steering_loss_weight=steering_loss_weight,
                    candidate_bce_loss_weight=candidate_bce_loss_weight,
                )
            )
            if reset_hidden is not None:
                reset_abstain, reset_candidates, _reset_steering = model.score_candidates(
                    reset_hidden[step_idx],
                    query_features,
                )
                losses.append(
                    _abstain_loss(reset_abstain, reset_candidates)
                    * float(reset_abstain_loss_weight)
                )
            if shuffled_hidden is not None:
                shuffle_abstain, shuffle_candidates, _shuffle_steering = model.score_candidates(
                    shuffled_hidden[step_idx],
                    query_features,
                )
                losses.append(
                    _abstain_loss(shuffle_abstain, shuffle_candidates)
                    * float(shuffle_abstain_loss_weight)
                )
            if float(evidence_write_loss_weight) > 0.0:
                evidence_batch = _evidence_batch(frame, device=device)
                if evidence_batch is not None:
                    evidence_features, evidence_object_ids, evidence_mask = evidence_batch
                    evidence_targets = _steering_targets(
                        row,
                        evidence_object_ids,
                        arc_threshold_rad=arc_threshold_rad,
                        yaw_threshold_rad=yaw_threshold_rad,
                        hold_range_m=hold_range_m,
                        device=device,
                    )
                    evidence_abstain, evidence_logits, evidence_steering = (
                        model.score_candidates(
                            write_hidden[step_idx],
                            evidence_features,
                        )
                    )
                    losses.append(
                        _frame_loss(
                            evidence_abstain,
                            evidence_logits,
                            evidence_steering,
                            evidence_mask,
                            evidence_targets,
                            positive_frame_weight=1.0,
                            negative_frame_weight=1.0,
                            steering_loss_weight=0.0,
                            candidate_bce_loss_weight=max(
                                1.0,
                                float(candidate_bce_loss_weight),
                            ),
                        )
                        * float(evidence_write_loss_weight)
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


def _evidence_batch(
    frame: Frame,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, list[str], torch.Tensor] | None:
    by_object: dict[str, torch.Tensor] = {}
    for query in frame.queries:
        if query.role != "first_visible_evidence":
            continue
        by_object.setdefault(query.object_id, query.features)
    if not by_object:
        return None
    object_ids = sorted(by_object)
    query_features = torch.stack([by_object[object_id] for object_id in object_ids])
    positive_mask = torch.ones(len(object_ids), dtype=torch.bool)
    return query_features.to(device), object_ids, positive_mask.to(device)


def _frame_loss(
    abstain_logit: torch.Tensor,
    candidate_logits: torch.Tensor,
    steering_logits: torch.Tensor,
    positive_mask: torch.Tensor,
    steering_targets: torch.Tensor,
    *,
    positive_frame_weight: float,
    negative_frame_weight: float,
    steering_loss_weight: float,
    candidate_bce_loss_weight: float,
) -> torch.Tensor:
    all_logits = torch.cat([abstain_logit.reshape(1), candidate_logits], dim=0)
    candidate_bce = F.binary_cross_entropy_with_logits(
        candidate_logits,
        positive_mask.to(dtype=candidate_logits.dtype),
    ) * float(candidate_bce_loss_weight)
    if bool(positive_mask.any().detach().cpu()):
        positive_logits = candidate_logits[positive_mask]
        selection_loss = torch.logsumexp(all_logits, dim=0) - torch.logsumexp(
            positive_logits,
            dim=0,
        )
        steering_loss = F.cross_entropy(
            steering_logits[positive_mask],
            steering_targets[positive_mask],
        )
        return float(positive_frame_weight) * (
            selection_loss + float(steering_loss_weight) * steering_loss
        ) + candidate_bce
    target = torch.zeros((), dtype=torch.long, device=all_logits.device)
    selection_loss = F.cross_entropy(all_logits.reshape(1, -1), target.reshape(1)) * float(
        negative_frame_weight
    )
    return selection_loss + candidate_bce


def _abstain_loss(abstain_logit: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    all_logits = torch.cat([abstain_logit.reshape(1), candidate_logits], dim=0)
    target = torch.zeros((), dtype=torch.long, device=all_logits.device)
    return F.cross_entropy(all_logits.reshape(1, -1), target.reshape(1))


def _evaluate(
    model: Go2MemorySteeringController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    row_index: dict[tuple[tuple[str, int, int], int], dict[str, Any]],
    *,
    device: torch.device,
    selection_margin: float,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
    ablation: str,
    include_current: bool = True,
) -> dict[str, Any]:
    hidden_by_key = _hidden_states_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
        include_current=include_current,
    )
    metrics = _ControllerMetrics()
    with torch.no_grad():
        for key, sequence in sequences.items():
            hidden = hidden_by_key[key]
            for step_idx, frame in enumerate(sequence):
                batch = _candidate_batch(frame, device=device)
                if batch is None:
                    continue
                query_features, object_ids, positive_mask = batch
                positive_objects = {
                    object_id
                    for object_id, is_positive in zip(
                        object_ids,
                        positive_mask.detach().cpu().numpy(),
                    )
                    if bool(is_positive)
                }
                selected_object = None
                predicted_steering = None
                target_steering = None
                if ablation != "memory_off_abstain":
                    abstain_logit, candidate_logits, steering_logits = model.score_candidates(
                        hidden[step_idx],
                        query_features,
                    )
                    selected_index = _select_index(
                        abstain_logit,
                        candidate_logits,
                        selection_margin=selection_margin,
                    )
                    if selected_index is not None:
                        selected_object = object_ids[selected_index]
                        steering_index = int(torch.argmax(steering_logits[selected_index]).detach().cpu())
                        predicted_steering = STEERING_CLASSES[
                            max(0, min(steering_index, len(STEERING_CLASSES) - 1))
                        ]
                        row = row_index.get((frame.seq_key, int(frame.episode_step)), {})
                        target_steering = _target_steering(
                            row,
                            selected_object,
                            arc_threshold_rad=arc_threshold_rad,
                            yaw_threshold_rad=yaw_threshold_rad,
                            hold_range_m=hold_range_m,
                        )
                metrics.add(
                    positive_objects=positive_objects,
                    selected_object=selected_object,
                    predicted_steering=predicted_steering,
                    target_steering=target_steering,
                )
    return metrics.to_dict()


def _hidden_states_by_sequence(
    model: Go2MemorySteeringController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    ablation: str,
    include_current: bool = True,
) -> dict[tuple[str, int, int], torch.Tensor]:
    hidden_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            images, aux = _sequence_tensors(sequence, device=device)
            if ablation in {"normal", "memory_off_abstain"}:
                hidden = model.forward_hidden(
                    images,
                    aux,
                    include_current=include_current,
                )
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
            elif ablation == "shuffle_hidden_states":
                hidden = model.forward_hidden(
                    images,
                    aux,
                    include_current=include_current,
                )
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


def _selection_score(evaluations: dict[str, dict[str, Any]]) -> float:
    normal = evaluations["normal"]
    corrupted_best = max(
        float(evaluations[name]["target_steering_pipeline_success"])
        for name in ("reset_recurrent_state", "reverse_input_history", "shuffle_hidden_states")
    )
    return (
        2.0 * float(normal["target_steering_pipeline_success"])
        + 0.5 * float(normal["target_recall"])
        + 0.25 * float(normal["target_selection_precision"])
        - 0.75 * float(normal["false_claim_rate"])
        + 0.75 * (float(normal["target_steering_pipeline_success"]) - corrupted_best)
    )


def _steering_targets(
    row: dict[str, Any],
    object_ids: list[str],
    *,
    arc_threshold_rad: float,
    yaw_threshold_rad: float,
    hold_range_m: float,
    device: torch.device,
) -> torch.Tensor:
    targets = []
    for object_id in object_ids:
        steering = _target_steering(
            row,
            object_id,
            arc_threshold_rad=arc_threshold_rad,
            yaw_threshold_rad=yaw_threshold_rad,
            hold_range_m=hold_range_m,
        )
        targets.append(STEERING_CLASSES.index(steering) if steering in STEERING_CLASSES else 1)
    return torch.tensor(targets, dtype=torch.long, device=device)


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


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


class _ControllerMetrics:
    def __init__(self) -> None:
        self.positive_frames = 0
        self.negative_frames = 0
        self.selected_frames = 0
        self.correct_target = 0
        self.missed_positive = 0
        self.false_claim = 0
        self.wrong_object = 0
        self.target_steer_success = 0
        self.predicted_steering: Counter[str] = Counter()
        self.target_steering: Counter[str] = Counter()
        self.classifications: Counter[str] = Counter()

    def add(
        self,
        *,
        positive_objects: set[str],
        selected_object: str | None,
        predicted_steering: str | None,
        target_steering: str | None,
    ) -> None:
        has_positive = bool(positive_objects)
        if has_positive:
            self.positive_frames += 1
        else:
            self.negative_frames += 1
        if selected_object is None:
            if has_positive:
                self.missed_positive += 1
                self.classifications["missed_positive"] += 1
            else:
                self.classifications["abstain"] += 1
            return
        self.selected_frames += 1
        if selected_object not in positive_objects:
            if has_positive:
                self.wrong_object += 1
                self.classifications["wrong_object"] += 1
            else:
                self.false_claim += 1
                self.classifications["false_claim"] += 1
            return
        self.correct_target += 1
        self.classifications["correct_target"] += 1
        if predicted_steering is not None:
            self.predicted_steering[predicted_steering] += 1
        if target_steering is not None:
            self.target_steering[target_steering] += 1
        if predicted_steering == target_steering and predicted_steering is not None:
            self.target_steer_success += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_frame_count": float(self.positive_frames),
            "negative_frame_count": float(self.negative_frames),
            "selected_frame_count": float(self.selected_frames),
            "correct_target_count": float(self.correct_target),
            "missed_positive_count": float(self.missed_positive),
            "false_claim_count": float(self.false_claim),
            "wrong_object_count": float(self.wrong_object),
            "target_recall": self.correct_target / max(1, self.positive_frames),
            "false_claim_rate": self.false_claim / max(1, self.negative_frames),
            "target_selection_precision": self.correct_target / max(1, self.selected_frames),
            "target_steering_success_count": float(self.target_steer_success),
            "target_steering_pipeline_success": self.target_steer_success
            / max(1, self.positive_frames),
            "predicted_steering_counts": dict(sorted(self.predicted_steering.items())),
            "target_steering_counts": dict(sorted(self.target_steering.items())),
            "classification_counts": dict(sorted(self.classifications.items())),
        }


if __name__ == "__main__":
    raise SystemExit(main())
