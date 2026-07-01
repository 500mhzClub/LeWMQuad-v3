#!/usr/bin/env python3
"""Train a pure RGB/JEPA recurrent color-query memory probe for Go2 rows.

The probe uses rendered RGB through a frozen JEPA encoder, odometry/action
history, and a target color query. Landmark bearings, ranges, visibility, and
seen-before labels are used only as offline supervision and evaluation labels.
Runtime inputs do not include object ids, landmark slots, map geometry, or
explicit bearing/range observations.
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
from train_go2_hidden_target_memory_probe import _load_image, _load_rows, _resolve_device  # noqa: E402
from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    STEERING_CLASSES,
    SpatialGo2JepaFeatureEncoder,
    _aux_features,
    _color_vocab,
    _finite_float,
    _landmark_by_id,
    _object_color,
    _seq_key,
    _steering_index,
    _steering_name,
    _vector_target,
)


COLOR_RGB = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


@dataclass(frozen=True)
class Query:
    color_index: int
    target: float
    target_steering: int
    target_vec: torch.Tensor
    bucket: str
    group_key: tuple[str, int, int, str]


@dataclass(frozen=True)
class HardQueryExample:
    seq_key: tuple[str, int, int]
    step_idx: int
    color_index: int


@dataclass(frozen=True)
class Frame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    all_vec: torch.Tensor
    all_steering: torch.Tensor
    visible_mask: torch.Tensor
    memory_mask: torch.Tensor
    queries: tuple[Query, ...]


class RelativeLandmarkMemoryProbe(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_output_dim: int,
        aux_dim: int,
        hidden_dim: int,
        color_count: int,
        color_embedding_dim: int,
        freeze_encoder: bool,
        episodic_attention: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        self.encoder_projection = (
            nn.Identity()
            if int(encoder_output_dim) == int(hidden_dim)
            else nn.Sequential(
                nn.Linear(int(encoder_output_dim), int(hidden_dim)),
                nn.GELU(),
                nn.LayerNorm(int(hidden_dim)),
            )
        )
        self.recurrent = nn.GRUCell(int(hidden_dim) + int(aux_dim), int(hidden_dim))
        self.color_embedding = nn.Embedding(int(color_count), int(color_embedding_dim))
        self.episodic_attention = bool(episodic_attention)
        base_feature_dim = int(hidden_dim) + int(color_embedding_dim)
        if self.episodic_attention:
            self.attn_query = nn.Linear(base_feature_dim, int(hidden_dim))
            self.attn_key = nn.Linear(int(hidden_dim), int(hidden_dim))
            self.attn_value = nn.Linear(int(hidden_dim), int(hidden_dim))
        feature_dim = base_feature_dim + (int(hidden_dim) if self.episodic_attention else 0)
        self.query_trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.memory_head = nn.Linear(int(hidden_dim), 1)
        self.steering_head = nn.Linear(int(hidden_dim), len(STEERING_CLASSES))
        self.vector_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 2),
            nn.Tanh(),
        )

    def forward_sequence(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        reset_each_step: bool = False,
        reverse: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.encoder(images)
        else:
            encoded = self.encoder(images)
        hidden = self.encoder_projection(encoded)
        order = list(range(int(images.shape[0])))
        if reverse:
            order = list(reversed(order))
        state = torch.zeros(hidden.shape[-1], device=images.device, dtype=hidden.dtype)
        states_by_time: list[torch.Tensor | None] = [None for _ in order]
        for idx in order:
            if reset_each_step:
                state = torch.zeros_like(state)
            state = self.recurrent(torch.cat([hidden[idx], aux[idx]], dim=-1), state)
            states_by_time[idx] = state
        states = torch.stack([state for state in states_by_time if state is not None])
        colors = torch.arange(
            self.color_embedding.num_embeddings,
            device=images.device,
            dtype=torch.long,
        )
        color_embedding = self.color_embedding(colors)
        expanded_states = states.unsqueeze(1).expand(-1, color_embedding.shape[0], -1)
        expanded_colors = color_embedding.unsqueeze(0).expand(states.shape[0], -1, -1)
        query_input = torch.cat([expanded_states, expanded_colors], dim=-1)
        if self.episodic_attention:
            query = self.attn_query(query_input)
            keys = self.attn_key(states)
            values = self.attn_value(states)
            scale = math.sqrt(float(max(1, keys.shape[-1])))
            contexts = []
            for step_idx in range(int(states.shape[0])):
                scores = torch.einsum(
                    "ch,sh->cs",
                    query[step_idx],
                    keys[: step_idx + 1],
                ) / scale
                weights = torch.softmax(scores, dim=-1)
                contexts.append(weights @ values[: step_idx + 1])
            query_input = torch.cat([query_input, torch.stack(contexts)], dim=-1)
        features = self.query_trunk(query_input)
        return {
            "memory_logits": self.memory_head(features).squeeze(-1),
            "steering_logits": self.steering_head(features),
            "vectors": self.vector_head(features),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--color-embedding-dim", type=int, default=32)
    parser.add_argument("--spatial-output-dim", type=int, default=512)
    parser.add_argument("--spatial-feature-stride", type=int, choices=(8, 16), default=8)
    parser.add_argument("--episodic-attention", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--query-loss-weight", type=float, default=4.0)
    parser.add_argument("--query-positive-loss-scale", type=float, default=1.0)
    parser.add_argument("--query-negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-group-balanced-query-loss-weight", type=float, default=2.0)
    parser.add_argument("--memory-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=2.0)
    parser.add_argument("--query-direction-loss-weight", type=float, default=3.0)
    parser.add_argument("--vector-loss-weight", type=float, default=0.5)
    parser.add_argument("--query-vector-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-pair-updates", type=int, default=32)
    parser.add_argument("--hard-pair-margin", type=float, default=2.0)
    parser.add_argument(
        "--steering-source",
        choices=("head", "vector", "vector_flip"),
        default="head",
    )
    parser.add_argument("--steering-class-balanced-loss", action="store_true")
    parser.add_argument("--finetune-jepa-encoder", action="store_true")
    parser.add_argument("--min-target-steering-success", type=float, default=0.90)
    parser.add_argument("--max-false-claim-rate", type=float, default=0.12)
    parser.add_argument("--min-corrupted-gap", type=float, default=0.30)
    parser.add_argument(
        "--thresholds",
        default="0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
        help="Comma-separated memory-score thresholds to evaluate.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("balanced", "gate"),
        default="balanced",
        help="balanced preserves the original score; gate ranks by strict-gate shortfall.",
    )
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()
    thresholds = _parse_thresholds(str(args.thresholds))

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    train_rows = _load_rows(args.datasets)
    validation_rows = _load_rows(args.validation_datasets)
    if not train_rows:
        raise SystemExit("no train rows")
    if not validation_rows:
        raise SystemExit("no validation rows")
    color_vocab = _color_vocab(train_rows, validation_rows)
    color_index = {color: idx for idx, color in enumerate(color_vocab)}
    train_sequences = _build_sequences(
        train_rows,
        color_index=color_index,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
    )
    validation_sequences = _build_sequences(
        validation_rows,
        color_index=color_index,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
    )
    aux_stats = _aux_stats(train_sequences)
    _normalize_aux(train_sequences, aux_stats)
    _normalize_aux(validation_sequences, aux_stats)
    query_counts = _query_counts(train_sequences)
    query_pos_weight = float(query_counts["negative"]) / float(max(1, query_counts["positive"]))
    hard_group_counts = _hard_group_counts(train_sequences)
    hard_pair_groups = _hard_pair_groups(train_sequences)

    device = _resolve_device(str(args.device))
    base_encoder, jepa_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=not bool(args.finetune_jepa_encoder),
    )
    encoder = SpatialGo2JepaFeatureEncoder(
        base_encoder,
        image_size=int(args.image_size),
        output_dim=int(args.spatial_output_dim),
        feature_stride=int(args.spatial_feature_stride),
    ).to(device)
    model = RelativeLandmarkMemoryProbe(
        encoder=encoder,
        encoder_output_dim=int(args.spatial_output_dim),
        aux_dim=next(iter(train_sequences.values()))[0].aux.numel(),
        hidden_dim=int(args.hidden_dim),
        color_count=len(color_vocab),
        color_embedding_dim=int(args.color_embedding_dim),
        freeze_encoder=not bool(args.finetune_jepa_encoder),
        episodic_attention=bool(args.episodic_attention),
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
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
            query_pos_weight=query_pos_weight,
            hard_group_counts=hard_group_counts,
            query_loss_weight=float(args.query_loss_weight),
            query_positive_loss_scale=float(args.query_positive_loss_scale),
            query_negative_loss_weight=float(args.query_negative_loss_weight),
            hard_group_balanced_query_loss_weight=float(
                args.hard_group_balanced_query_loss_weight
            ),
            memory_state_loss_weight=float(args.memory_state_loss_weight),
            direction_loss_weight=float(args.direction_loss_weight),
            query_direction_loss_weight=float(args.query_direction_loss_weight),
            vector_loss_weight=float(args.vector_loss_weight),
            query_vector_loss_weight=float(args.query_vector_loss_weight),
            hard_pair_groups=hard_pair_groups,
            hard_pair_loss_weight=float(args.hard_pair_loss_weight),
            hard_pair_updates=int(args.hard_pair_updates),
            hard_pair_margin=float(args.hard_pair_margin),
            steering_class_balanced_loss=bool(args.steering_class_balanced_loss),
        )
        threshold_sweep = _threshold_sweep(
            model,
            validation_sequences,
            device=device,
            steering_source=str(args.steering_source),
            thresholds=thresholds,
        )
        threshold_key, threshold_value = _select_threshold(
            threshold_sweep,
            mode=str(args.selection_mode),
            min_target_steering_success=float(args.min_target_steering_success),
            max_false_claim_rate=float(args.max_false_claim_rate),
            min_corrupted_gap=float(args.min_corrupted_gap),
        )
        normal = threshold_value["normal"]
        score = _threshold_value_selection_score(
            threshold_value,
            mode=str(args.selection_mode),
            min_target_steering_success=float(args.min_target_steering_success),
            max_false_claim_rate=float(args.max_false_claim_rate),
            min_corrupted_gap=float(args.min_corrupted_gap),
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "selected_threshold": threshold_key,
                "validation": threshold_value,
            }
        )
        if score >= best_score:
            best_score = score
            best_metrics = threshold_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={train_loss:.4f}"
                f" threshold={threshold_key}"
                f" target_steer={normal['target_steering_pipeline_success']:.3f}"
                f" recall={normal['target_recall']:.3f}"
                f" false_claim={normal['false_claim_rate']:.3f}"
                f" gap={threshold_value['normal_minus_best_corrupted_target_steering_pipeline_success']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train_sweep = _threshold_sweep(
        model,
        train_sequences,
        device=device,
        steering_source=str(args.steering_source),
        thresholds=thresholds,
    )
    final_validation_sweep = _threshold_sweep(
        model,
        validation_sequences,
        device=device,
        steering_source=str(args.steering_source),
        thresholds=thresholds,
    )
    best_threshold_key, best_threshold_value = _select_threshold(
        final_validation_sweep,
        mode=str(args.selection_mode),
        min_target_steering_success=float(args.min_target_steering_success),
        max_false_claim_rate=float(args.max_false_claim_rate),
        min_corrupted_gap=float(args.min_corrupted_gap),
    )
    normal = best_threshold_value["normal"]
    gap = float(best_threshold_value["normal_minus_best_corrupted_target_steering_pipeline_success"])
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and gap >= float(args.min_corrupted_gap)
    )

    checkpoint = {
        "schema": "lewm_go2_jepa_relative_landmark_memory_probe_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "frozen_jepa_report": {
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
        },
        "color_vocab": color_vocab,
        "aux_mean": aux_stats["mean"].tolist(),
        "aux_std": aux_stats["std"].tolist(),
        "range_scale_m": float(args.range_scale_m),
        "steering_source": str(args.steering_source),
        "episodic_attention": bool(args.episodic_attention),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_jepa_relative_landmark_memory_probe_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "color_vocab": color_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(sequence) for sequence in train_sequences.values()),
        "validation_row_count": sum(len(sequence) for sequence in validation_sequences.values()),
        "query_label_counts": query_counts,
        "steering_source": str(args.steering_source),
        "episodic_attention": bool(args.episodic_attention),
        "selection_mode": str(args.selection_mode),
        "thresholds": list(thresholds),
        "hard_group_balanced_count": len(hard_group_counts),
        "hard_pair_group_count": len(hard_pair_groups),
        "hard_pair_example_count": sum(
            len(bucket["positive"]) + len(bucket["negative"])
            for bucket in hard_pair_groups.values()
        ),
        "best_validation_during_training": best_metrics or {},
        "final_train_threshold_sweep": final_train_sweep,
        "threshold_sweep": final_validation_sweep,
        "selected_threshold": best_threshold_key,
        "validation_ablations": best_threshold_value["ablations"],
        "normal_minus_best_corrupted_target_steering_pipeline_success": gap,
        "strict_gate_pass": bool(gate_pass),
        "gate_requirements": {
            "min_target_steering_success": float(args.min_target_steering_success),
            "max_false_claim_rate": float(args.max_false_claim_rate),
            "min_corrupted_gap": float(args.min_corrupted_gap),
        },
        "history": history,
        "claim_boundary": (
            "Pure RGB/JEPA recurrent color-query memory probe. Inference uses rendered "
            "RGB through the JEPA encoder, recurrent odometry/action history, and target "
            "color. Landmark bearings/ranges/visibility/seen-before labels are offline "
            "supervision and evaluation only."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_jepa_relative_landmark_memory_probe:"
        f" output={args.output}"
        f" report={report_path}"
        f" threshold={best_threshold_key}"
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
    color_index: dict[str, int],
    image_size: int,
    range_scale_m: float,
) -> dict[tuple[str, int, int], list[Frame]]:
    color_count = len(color_index)
    sequences_raw: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences_raw[_seq_key(row)].append(row)
    result: dict[tuple[str, int, int], list[Frame]] = {}
    for seq_key, sequence_rows in sequences_raw.items():
        sequence_rows.sort(key=lambda row: int(row.get("episode_step", 0)))
        memory_mask = np.zeros(color_count, dtype=np.float32)
        prior_rgb_objects: set[str] = set()
        prior_rgb_colors: set[str] = set()
        frames = []
        for row in sequence_rows:
            image = _load_image(Path(row["rgb_path"]), image_size=image_size)
            landmark_by_id = _landmark_by_id(row)
            visible_mask = np.zeros(color_count, dtype=np.float32)
            all_vec = np.zeros((color_count, 2), dtype=np.float32)
            all_steering = np.zeros(color_count, dtype=np.int64)
            for landmark in row.get("landmarks", ()):
                color = _object_color(str(landmark.get("object_id", "")))
                if color not in color_index:
                    continue
                idx = int(color_index[color])
                all_vec[idx] = _vector_target(landmark, range_scale_m=range_scale_m)
                all_steering[idx] = _steering_index(
                    _finite_float(landmark.get("bearing_body_rad"), 0.0)
                )
                if bool(landmark.get("visible", False)):
                    visible_mask[idx] = 1.0
            memory_mask = np.maximum(memory_mask, visible_mask)
            queries = []
            seen_query_keys: set[tuple[int, float]] = set()
            for event in row.get("go2_causal_memory_pair_selection", ()):
                role = str(event.get("pair_role", ""))
                if not role.startswith("current_"):
                    continue
                object_id = str(event.get("object_id", ""))
                landmark = landmark_by_id.get(object_id)
                color = _object_color(object_id)
                if landmark is None or color not in color_index:
                    continue
                color_idx = int(color_index[color])
                target = 1.0 if bool(event.get("seen_before", False)) else 0.0
                dedup_key = (color_idx, target)
                if dedup_key in seen_query_keys:
                    continue
                seen_query_keys.add(dedup_key)
                bucket = _bucket_name(
                    target=bool(event.get("seen_before", False)),
                    prior_object=object_id in prior_rgb_objects,
                    prior_color=color in prior_rgb_colors,
                    current_visible=bool(landmark.get("visible", False)),
                )
                bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
                queries.append(
                    Query(
                        color_index=color_idx,
                        target=target,
                        target_steering=_steering_index(bearing),
                        target_vec=torch.tensor(
                            _vector_target(landmark, range_scale_m=range_scale_m),
                            dtype=torch.float32,
                        ),
                        bucket=bucket,
                        group_key=(
                            str(row.get("scene_id", "")),
                            int(row.get("cell_id", -1)),
                            int(row.get("yaw_bin", -1)),
                            color,
                        ),
                    )
                )
            frames.append(
                Frame(
                    seq_key=seq_key,
                    episode_step=int(row.get("episode_step", 0)),
                    image=image,
                    aux=torch.tensor(_aux_features(row), dtype=torch.float32),
                    all_vec=torch.tensor(all_vec, dtype=torch.float32),
                    all_steering=torch.tensor(all_steering, dtype=torch.long),
                    visible_mask=torch.tensor(visible_mask, dtype=torch.float32),
                    memory_mask=torch.tensor(memory_mask.copy(), dtype=torch.float32),
                    queries=tuple(queries),
                )
            )
            for landmark in row.get("landmarks", ()):
                if not bool(landmark.get("visible", False)):
                    continue
                object_id = str(landmark.get("object_id", ""))
                color = _object_color(object_id)
                if color not in color_index or color not in COLOR_RGB:
                    continue
                if _rgb_color_area_from_tensor(image, color=color) >= 0.001:
                    prior_rgb_objects.add(object_id)
                    prior_rgb_colors.add(color)
        result[seq_key] = frames
    return result


def _train_epoch(
    model: RelativeLandmarkMemoryProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    query_pos_weight: float,
    hard_group_counts: dict[tuple[str, int, int, str], dict[str, int]],
    query_loss_weight: float,
    query_positive_loss_scale: float,
    query_negative_loss_weight: float,
    hard_group_balanced_query_loss_weight: float,
    memory_state_loss_weight: float,
    direction_loss_weight: float,
    query_direction_loss_weight: float,
    vector_loss_weight: float,
    query_vector_loss_weight: float,
    hard_pair_groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]],
    hard_pair_loss_weight: float,
    hard_pair_updates: int,
    hard_pair_margin: float,
    steering_class_balanced_loss: bool,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total_loss = 0.0
    trained = 0
    for key in keys:
        batch = _sequence_tensors(sequences[key], device=device)
        outputs = model.forward_sequence(batch["images"], batch["aux"])
        loss = _weighted_bce(
            outputs["memory_logits"],
            batch["memory_mask"],
            positive_scale=1.0,
            negative_weight=1.0,
        ) * float(memory_state_loss_weight)
        loss = loss + _direction_loss(
            outputs["steering_logits"],
            batch["all_steering"],
            balanced=bool(steering_class_balanced_loss),
        ) * float(direction_loss_weight)
        loss = loss + F.smooth_l1_loss(outputs["vectors"], batch["all_vec"]) * float(
            vector_loss_weight
        )
        query_loss = _query_loss(
            sequences[key],
            outputs,
            device=device,
            query_pos_weight=float(query_pos_weight),
            hard_group_counts=hard_group_counts,
            query_positive_loss_scale=float(query_positive_loss_scale),
            query_negative_loss_weight=float(query_negative_loss_weight),
            hard_group_balanced_query_loss_weight=float(hard_group_balanced_query_loss_weight),
            query_direction_loss_weight=float(query_direction_loss_weight),
            query_vector_loss_weight=float(query_vector_loss_weight),
            steering_class_balanced_loss=bool(steering_class_balanced_loss),
        )
        if query_loss is not None:
            loss = loss + query_loss * float(query_loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        trained += 1
        if float(hard_pair_loss_weight) > 0.0 and hard_pair_groups:
            pair_total = _hard_pair_updates(
                model,
                optimizer,
                sequences,
                hard_pair_groups=hard_pair_groups,
                device=device,
                update_count=int(hard_pair_updates),
                margin=float(hard_pair_margin),
                loss_weight=float(hard_pair_loss_weight),
            )
            total_loss += float(pair_total)
    return total_loss / float(max(1, trained))


def _query_loss(
    sequence: list[Frame],
    outputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    query_pos_weight: float,
    hard_group_counts: dict[tuple[str, int, int, str], dict[str, int]],
    query_positive_loss_scale: float,
    query_negative_loss_weight: float,
    hard_group_balanced_query_loss_weight: float,
    query_direction_loss_weight: float,
    query_vector_loss_weight: float,
    steering_class_balanced_loss: bool,
) -> torch.Tensor | None:
    logits = []
    targets = []
    group_keys = []
    steering_logits = []
    steering_targets = []
    vector_preds = []
    vector_targets = []
    for step_idx, frame in enumerate(sequence):
        for query in frame.queries:
            color_idx = int(query.color_index)
            logits.append(outputs["memory_logits"][step_idx, color_idx])
            targets.append(float(query.target))
            group_keys.append(query.group_key)
            if query.target >= 0.5:
                steering_logits.append(outputs["steering_logits"][step_idx, color_idx])
                steering_targets.append(int(query.target_steering))
                vector_preds.append(outputs["vectors"][step_idx, color_idx])
                vector_targets.append(query.target_vec)
    if not logits:
        return None
    stacked_logits = torch.stack(logits)
    stacked_targets = torch.tensor(targets, dtype=stacked_logits.dtype, device=device)
    raw_seen_loss = F.binary_cross_entropy_with_logits(
        stacked_logits,
        stacked_targets,
        reduction="none",
    )
    weights = torch.where(
        stacked_targets >= 0.5,
        torch.full_like(
            stacked_targets,
            float(query_pos_weight) * float(query_positive_loss_scale),
        ),
        torch.full_like(stacked_targets, float(query_negative_loss_weight)),
    )
    loss = (raw_seen_loss * weights).mean()
    if float(hard_group_balanced_query_loss_weight) > 0.0:
        balanced_weights = []
        for group_key, target in zip(group_keys, targets):
            counts = hard_group_counts.get(group_key)
            if not counts:
                balanced_weights.append(0.0)
                continue
            label = "positive" if float(target) >= 0.5 else "negative"
            balanced_weights.append(1.0 / float(max(1, counts[label])))
        balanced = torch.tensor(balanced_weights, dtype=stacked_logits.dtype, device=device)
        if float(balanced.sum().detach().cpu()) > 0.0:
            loss = loss + (
                (raw_seen_loss * balanced).sum() / balanced.sum().clamp_min(1e-6)
            ) * float(hard_group_balanced_query_loss_weight)
    if steering_logits:
        steering_targets_tensor = torch.tensor(
            steering_targets,
            dtype=torch.long,
            device=device,
        )
        steering_raw = F.cross_entropy(
            torch.stack(steering_logits),
            steering_targets_tensor,
            reduction="none",
        )
        if bool(steering_class_balanced_loss):
            counts = torch.bincount(
                steering_targets_tensor,
                minlength=len(STEERING_CLASSES),
            ).to(dtype=steering_raw.dtype)
            class_weights = steering_targets_tensor.numel() / (
                float(len(STEERING_CLASSES)) * counts.clamp_min(1.0)
            )
            steering_raw = steering_raw * class_weights[steering_targets_tensor]
        loss = loss + steering_raw.mean() * float(query_direction_loss_weight)
        loss = loss + F.smooth_l1_loss(
            torch.stack(vector_preds),
            torch.stack(vector_targets).to(device),
        ) * float(query_vector_loss_weight)
    return loss


def _evaluate(
    model: RelativeLandmarkMemoryProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    threshold: float,
    ablation: str,
    steering_source: str,
) -> dict[str, Any]:
    outputs_by_key = _outputs_by_sequence(model, sequences, device=device, ablation=ablation)
    metrics = _Metrics()
    with torch.no_grad():
        for key, sequence in sequences.items():
            outputs = outputs_by_key[key]
            for step_idx, frame in enumerate(sequence):
                for query in frame.queries:
                    color_idx = int(query.color_index)
                    if ablation == "memory_off_abstain":
                        score_value = 0.0
                        selected = False
                    else:
                        score = torch.sigmoid(outputs["memory_logits"][step_idx, color_idx])
                        score_value = float(score.detach().cpu())
                        selected = score_value >= float(threshold)
                    steering_index = None
                    if selected:
                        steering_index = _select_steering_index(
                            outputs,
                            step_idx=step_idx,
                            color_idx=color_idx,
                            steering_source=steering_source,
                        )
                    metrics.add(
                        query=query,
                        selected=selected,
                        steering_index=steering_index,
                        score=score_value,
                    )
    return metrics.to_dict()


def _outputs_by_sequence(
    model: RelativeLandmarkMemoryProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    ablation: str,
) -> dict[tuple[str, int, int], dict[str, torch.Tensor]]:
    model.eval()
    outputs_by_key = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            batch = _sequence_tensors(sequence, device=device)
            if ablation in {"normal", "memory_off_abstain", "shuffle_memory_states"}:
                outputs = model.forward_sequence(batch["images"], batch["aux"])
            elif ablation == "reset_recurrent_state":
                outputs = model.forward_sequence(
                    batch["images"],
                    batch["aux"],
                    reset_each_step=True,
                )
            elif ablation == "reverse_input_history":
                outputs = model.forward_sequence(batch["images"], batch["aux"], reverse=True)
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            outputs_by_key[key] = outputs
    if ablation != "shuffle_memory_states":
        return outputs_by_key
    for name in ("memory_logits", "steering_logits", "vectors"):
        flat = []
        spans = {}
        cursor = 0
        for key in sequences:
            value = outputs_by_key[key][name]
            flat.append(value)
            spans[key] = (cursor, cursor + int(value.shape[0]))
            cursor += int(value.shape[0])
        if cursor <= 1:
            continue
        shuffled = torch.roll(torch.cat(flat, dim=0), shifts=max(1, cursor // 2), dims=0)
        for key, (start, end) in spans.items():
            outputs_by_key[key][name] = shuffled[start:end]
    return outputs_by_key


def _threshold_sweep(
    model: RelativeLandmarkMemoryProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    steering_source: str,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    result = {}
    for threshold in thresholds:
        ablations = {
            ablation: _evaluate(
                model,
                sequences,
                device=device,
                threshold=float(threshold),
                ablation=ablation,
                steering_source=steering_source,
            )
            for ablation in (
                "normal",
                "memory_off_abstain",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_memory_states",
            )
        }
        normal = float(ablations["normal"]["target_steering_pipeline_success"])
        corrupted = max(
            float(ablations[name]["target_steering_pipeline_success"])
            for name in (
                "memory_off_abstain",
                "reset_recurrent_state",
                "reverse_input_history",
                "shuffle_memory_states",
            )
        )
        result[str(float(threshold))] = {
            "threshold": float(threshold),
            "normal": ablations["normal"],
            "ablations": ablations,
            "normal_minus_best_corrupted_target_steering_pipeline_success": (
                normal - corrupted
            ),
        }
    return result


def _select_steering_index(
    outputs: dict[str, torch.Tensor],
    *,
    step_idx: int,
    color_idx: int,
    steering_source: str,
) -> int:
    if steering_source == "head":
        return int(
            torch.argmax(outputs["steering_logits"][step_idx, color_idx]).detach().cpu()
        )
    vector_index = _vector_steering_index(outputs["vectors"][step_idx, color_idx])
    if steering_source == "vector":
        return vector_index
    if steering_source == "vector_flip":
        return _flip_steering_index(vector_index)
    raise ValueError(f"unknown steering source: {steering_source}")


def _vector_steering_index(vector: torch.Tensor) -> int:
    x = float(vector[0].detach().cpu())
    y = float(vector[1].detach().cpu())
    return _steering_index(math.atan2(y, x))


def _flip_steering_index(index: int) -> int:
    if int(index) == 0:
        return 2
    if int(index) == 2:
        return 0
    return 1


def _hard_pair_updates(
    model: RelativeLandmarkMemoryProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    hard_pair_groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]],
    device: torch.device,
    update_count: int,
    margin: float,
    loss_weight: float,
) -> float:
    model.train()
    keys = list(hard_pair_groups)
    if not keys or int(update_count) <= 0:
        return 0.0
    total = 0.0
    for _ in range(int(update_count)):
        bucket = hard_pair_groups[random.choice(keys)]
        positive = random.choice(bucket["positive"])
        negative = random.choice(bucket["negative"])
        pos_logit = _example_memory_logit(model, sequences, positive, device=device)
        neg_logit = _example_memory_logit(model, sequences, negative, device=device)
        loss = F.softplus(float(margin) - (pos_logit - neg_logit)) * float(loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
    return total / float(max(1, update_count))


def _example_memory_logit(
    model: RelativeLandmarkMemoryProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    example: HardQueryExample,
    *,
    device: torch.device,
) -> torch.Tensor:
    batch = _sequence_tensors(sequences[example.seq_key], device=device)
    outputs = model.forward_sequence(batch["images"], batch["aux"])
    return outputs["memory_logits"][int(example.step_idx), int(example.color_index)]


class _Metrics:
    def __init__(self) -> None:
        self.positive = 0
        self.negative = 0
        self.correct_target = 0
        self.false_claim = 0
        self.missed_positive = 0
        self.target_steer = 0
        self.selected = 0
        self.classifications: Counter[str] = Counter()
        self.predicted_steering: Counter[str] = Counter()
        self.target_steering: Counter[str] = Counter()
        self.bucket_counts: Counter[str] = Counter()
        self.bucket_target_steer: Counter[str] = Counter()
        self.bucket_false_claim: Counter[str] = Counter()
        self.bucket_selected: Counter[str] = Counter()
        self.query_records: list[dict[str, Any]] = []

    def add(
        self,
        *,
        query: Query,
        selected: bool,
        steering_index: int | None,
        score: float,
    ) -> None:
        bucket = str(query.bucket)
        self.bucket_counts[bucket] += 1
        predicted_name = (
            None
            if steering_index is None
            else _steering_name(int(steering_index))
        )
        target_name = _steering_name(int(query.target_steering))
        self.query_records.append(
            {
                "bucket": bucket,
                "target": float(query.target),
                "score": float(score),
                "selected": bool(selected),
                "target_steering": target_name,
                "predicted_steering": predicted_name,
                "group_key": list(query.group_key),
            }
        )
        if query.target >= 0.5:
            self.positive += 1
            if not selected:
                self.missed_positive += 1
                self.classifications["missed_positive"] += 1
                return
            self.selected += 1
            self.bucket_selected[bucket] += 1
            self.correct_target += 1
            self.classifications["correct_target"] += 1
            pred = _steering_name(int(steering_index if steering_index is not None else 1))
            target = target_name
            self.predicted_steering[pred] += 1
            self.target_steering[target] += 1
            if pred == target:
                self.target_steer += 1
                self.bucket_target_steer[bucket] += 1
            return
        self.negative += 1
        if selected:
            self.selected += 1
            self.bucket_selected[bucket] += 1
            self.false_claim += 1
            self.bucket_false_claim[bucket] += 1
            self.classifications["false_claim"] += 1
        else:
            self.classifications["abstain"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_frame_count": float(self.positive),
            "negative_frame_count": float(self.negative),
            "selected_frame_count": float(self.selected),
            "correct_target_count": float(self.correct_target),
            "target_steering_success_count": float(self.target_steer),
            "missed_positive_count": float(self.missed_positive),
            "false_claim_count": float(self.false_claim),
            "target_recall": self.correct_target / max(1, self.positive),
            "target_steering_pipeline_success": self.target_steer / max(1, self.positive),
            "false_claim_rate": self.false_claim / max(1, self.negative),
            "target_selection_precision": self.correct_target / max(1, self.selected),
            "classification_counts": dict(sorted(self.classifications.items())),
            "predicted_steering_counts": dict(sorted(self.predicted_steering.items())),
            "target_steering_counts": dict(sorted(self.target_steering.items())),
            "bucket_counts": dict(sorted(self.bucket_counts.items())),
            "bucket_selected_rate": {
                key: float(self.bucket_selected[key]) / float(max(1, count))
                for key, count in sorted(self.bucket_counts.items())
            },
            "bucket_target_steering_pipeline_success": {
                key: float(self.bucket_target_steer[key]) / float(max(1, count))
                for key, count in sorted(self.bucket_counts.items())
                if key.startswith("positive_")
            },
            "bucket_false_claim_rate": {
                key: float(self.bucket_false_claim[key]) / float(max(1, count))
                for key, count in sorted(self.bucket_counts.items())
                if key.startswith("negative_")
            },
            "query_records": self.query_records,
        }


def _sequence_tensors(sequence: list[Frame], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([frame.image for frame in sequence]).to(device),
        "aux": torch.stack([frame.aux for frame in sequence]).to(device),
        "all_vec": torch.stack([frame.all_vec for frame in sequence]).to(device),
        "all_steering": torch.stack([frame.all_steering for frame in sequence]).to(device),
        "visible_mask": torch.stack([frame.visible_mask for frame in sequence]).to(device),
        "memory_mask": torch.stack([frame.memory_mask for frame in sequence]).to(device),
    }


def _direction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    balanced: bool,
) -> torch.Tensor:
    raw_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    if not bool(balanced):
        return raw_loss.mean()
    flat_targets = targets.reshape(-1)
    counts = torch.bincount(flat_targets, minlength=len(STEERING_CLASSES)).to(
        dtype=raw_loss.dtype
    )
    class_weights = flat_targets.numel() / (
        float(len(STEERING_CLASSES)) * counts.clamp_min(1.0)
    )
    return (raw_loss * class_weights[flat_targets]).mean()


def _weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_scale: float,
    negative_weight: float,
) -> torch.Tensor:
    raw_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets >= 0.5,
        torch.full_like(targets, float(positive_scale)),
        torch.full_like(targets, float(negative_weight)),
    )
    return (raw_loss * weights).mean()


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    thresholds = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"threshold outside [0, 1]: {value}")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    return tuple(sorted(set(thresholds)))


def _select_threshold(
    threshold_sweep: dict[str, Any],
    *,
    mode: str,
    min_target_steering_success: float,
    max_false_claim_rate: float,
    min_corrupted_gap: float,
) -> tuple[str, dict[str, Any]]:
    return max(
        threshold_sweep.items(),
        key=lambda item: (
            _threshold_value_selection_score(
                item[1],
                mode=mode,
                min_target_steering_success=min_target_steering_success,
                max_false_claim_rate=max_false_claim_rate,
                min_corrupted_gap=min_corrupted_gap,
            ),
            float(item[1]["normal"]["target_steering_pipeline_success"]),
            -float(item[1]["normal"]["false_claim_rate"]),
            float(item[1]["normal_minus_best_corrupted_target_steering_pipeline_success"]),
        ),
    )


def _threshold_value_selection_score(
    value: dict[str, Any],
    *,
    mode: str,
    min_target_steering_success: float,
    max_false_claim_rate: float,
    min_corrupted_gap: float,
) -> float:
    normal = value["normal"]
    gap = float(value["normal_minus_best_corrupted_target_steering_pipeline_success"])
    if mode == "balanced":
        return _selection_score(normal) + 0.5 * gap
    if mode != "gate":
        raise ValueError(f"unknown selection mode: {mode}")
    steer = float(normal["target_steering_pipeline_success"])
    false_claim = float(normal["false_claim_rate"])
    steer_shortfall = max(0.0, float(min_target_steering_success) - steer) / max(
        1e-6, float(min_target_steering_success)
    )
    false_shortfall = max(0.0, false_claim - float(max_false_claim_rate)) / max(
        1e-6, 1.0 - float(max_false_claim_rate)
    )
    gap_shortfall = max(0.0, float(min_corrupted_gap) - gap) / max(
        1e-6, float(min_corrupted_gap)
    )
    gate_pass = (
        steer >= float(min_target_steering_success)
        and false_claim <= float(max_false_claim_rate)
        and gap >= float(min_corrupted_gap)
    )
    if gate_pass:
        return 1000.0 + _selection_score(normal) + 0.5 * gap
    return -(
        2.0 * steer_shortfall
        + 2.0 * false_shortfall
        + gap_shortfall
    )


def _selection_score(metrics: dict[str, Any]) -> float:
    return (
        2.0 * float(metrics["target_steering_pipeline_success"])
        + 0.5 * float(metrics["target_recall"])
        + 0.25 * float(metrics["target_selection_precision"])
        - 0.75 * float(metrics["false_claim_rate"])
    )


def _aux_stats(sequences: dict[tuple[str, int, int], list[Frame]]) -> dict[str, np.ndarray]:
    features = np.stack([frame.aux.numpy() for sequence in sequences.values() for frame in sequence])
    return {
        "mean": features.mean(axis=0).astype(np.float32),
        "std": np.maximum(features.std(axis=0), 1e-6).astype(np.float32),
    }


def _normalize_aux(
    sequences: dict[tuple[str, int, int], list[Frame]],
    stats: dict[str, np.ndarray],
) -> None:
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    for key, sequence in list(sequences.items()):
        sequences[key] = [
            Frame(
                seq_key=frame.seq_key,
                episode_step=frame.episode_step,
                image=frame.image,
                aux=(frame.aux - mean) / std,
                all_vec=frame.all_vec,
                all_steering=frame.all_steering,
                visible_mask=frame.visible_mask,
                memory_mask=frame.memory_mask,
                queries=frame.queries,
            )
            for frame in sequence
        ]


def _query_counts(sequences: dict[tuple[str, int, int], list[Frame]]) -> dict[str, int]:
    positive = 0
    total = 0
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                positive += int(query.target >= 0.5)
                total += 1
    return {"positive": positive, "negative": total - positive, "total": total}


def _hard_group_counts(
    sequences: dict[tuple[str, int, int], list[Frame]],
) -> dict[tuple[str, int, int, str], dict[str, int]]:
    counts: dict[tuple[str, int, int, str], dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0}
    )
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                label = "positive" if query.target >= 0.5 else "negative"
                counts[query.group_key][label] += 1
    return {
        key: dict(value)
        for key, value in counts.items()
        if value["positive"] > 0 and value["negative"] > 0
    }


def _hard_pair_groups(
    sequences: dict[tuple[str, int, int], list[Frame]],
) -> dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]]:
    groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    for seq_key, sequence in sequences.items():
        for step_idx, frame in enumerate(sequence):
            for query in frame.queries:
                example = HardQueryExample(
                    seq_key=seq_key,
                    step_idx=int(step_idx),
                    color_index=int(query.color_index),
                )
                label = "positive" if query.target >= 0.5 else "negative"
                groups[query.group_key][label].append(example)
    return {
        key: bucket
        for key, bucket in groups.items()
        if bucket["positive"] and bucket["negative"]
    }


def _bucket_name(
    *,
    target: bool,
    prior_object: bool,
    prior_color: bool,
    current_visible: bool,
) -> str:
    prefix = "positive" if target else "negative"
    if prior_object:
        support = "prior_object_rgb"
    elif prior_color:
        support = "prior_color_rgb"
    elif current_visible:
        support = "current_visible_no_prior_rgb"
    else:
        support = "no_prior_current_hidden"
    return f"{prefix}_{support}"


def _rgb_color_area_from_tensor(image: torch.Tensor, *, color: str) -> float:
    rgb = COLOR_RGB.get(color)
    if rgb is None:
        return 0.0
    pixels = image.permute(1, 2, 0).numpy()
    target = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    distance = ((pixels - target) ** 2).mean(axis=2)
    similarity = np.exp(-distance / (2.0 * 0.20**2))
    soft_mask = 1.0 / (1.0 + np.exp(-(similarity - 0.65) / 0.08))
    return float(soft_mask.mean())


if __name__ == "__main__":
    raise SystemExit(main())
