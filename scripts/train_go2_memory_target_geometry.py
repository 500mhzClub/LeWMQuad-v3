#!/usr/bin/env python3
"""Train a scrubbed Go2 memory probe for hidden target geometry.

The target-selection gate answers whether an object was seen before. A Go2
return controller also needs target-relative geometry. This probe tests whether
recurrent RGB memory can recover the hidden remembered target's egocentric
bearing and range without using the current command label as an aux feature.
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

from train_go2_causal_memory_query_probe import (  # noqa: E402
    _scrub_command_aux,
    _scrub_runtime_aux,
)
from train_go2_hidden_target_memory_probe import (  # noqa: E402
    _aux_features,
    _feature_stats,
    _load_image,
    _load_rows,
    _primitive_vocab,
    _resolve_device,
    _seq_key,
)
from lewm.models.go2_jepa import load_go2_jepa_encoder  # noqa: E402


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
class GeometryQuery:
    object_id: str
    color: str
    role: str
    seen_before: float
    features: torch.Tensor
    bearing_rad: float
    range_m: float


@dataclass(frozen=True)
class GeometryFrame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    queries: tuple[GeometryQuery, ...]


class QueryGeometryMemoryProbe(nn.Module):
    def __init__(
        self,
        aux_dim: int,
        query_dim: int,
        hidden_dim: int,
        *,
        encoder: nn.Module | None = None,
        encoder_output_dim: int | None = None,
        freeze_encoder: bool = False,
        predict_steering: bool = False,
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
        self.query_trunk = nn.Sequential(
            nn.Linear(hidden_dim + query_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.seen_head = nn.Linear(hidden_dim // 2, 1)
        self.geometry_head = nn.Linear(hidden_dim // 2, 3)
        self.steering_head = (
            nn.Linear(hidden_dim // 2, 3) if bool(predict_steering) else None
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

    def score_queries(
        self,
        hidden: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen_logits, geom_pred, _steering_logits = self.score_queries_with_steering(
            hidden,
            query_features,
        )
        return seen_logits, geom_pred

    def score_queries_with_steering(
        self,
        hidden: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        trunk = self.query_trunk(torch.cat([hidden, query_features], dim=-1))
        steering_logits = self.steering_head(trunk) if self.steering_head is not None else None
        return self.seen_head(trunk).squeeze(-1), self.geometry_head(trunk), steering_logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--range-scale-m", type=float, default=12.0)
    parser.add_argument("--geometry-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--steering-loss-weight",
        type=float,
        default=0.0,
        help="Add a direct left/forward/right target-steering classification loss.",
    )
    parser.add_argument("--seen-loss-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=0.5)
    parser.add_argument(
        "--balanced-geometry-loss",
        action="store_true",
        help="Weight positive geometry loss by inverse target steering bucket frequency.",
    )
    parser.add_argument("--max-geometry-bucket-weight", type=float, default=4.0)
    parser.add_argument(
        "--include-object-slot",
        action="store_true",
        help="Add parsed landmark ordinal features to the query.",
    )
    parser.add_argument(
        "--allow-command-aux-leak",
        action="store_true",
        help="Diagnostic only: keep current command fields in aux features.",
    )
    parser.add_argument(
        "--scrub-runtime-aux",
        action="store_true",
        help="Zero command plus scene-derived clearance/traversability aux fields.",
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
    validation_rows_raw = _load_rows(args.validation_datasets)
    if not train_rows_raw:
        raise SystemExit("no train rows")
    if not validation_rows_raw:
        raise SystemExit("no validation rows")
    if args.allow_command_aux_leak:
        train_rows = train_rows_raw
        validation_rows = validation_rows_raw
    elif args.scrub_runtime_aux:
        train_rows = _scrub_runtime_aux(train_rows_raw)
        validation_rows = _scrub_runtime_aux(validation_rows_raw)
    elif args.scrub_scene_aux:
        train_rows = _scrub_scene_aux(train_rows_raw)
        validation_rows = _scrub_scene_aux(validation_rows_raw)
    else:
        train_rows = _scrub_command_aux(train_rows_raw)
        validation_rows = _scrub_command_aux(validation_rows_raw)

    primitive_vocab = _primitive_vocab(train_rows_raw, validation_rows_raw)
    color_vocab = _color_vocab(train_rows_raw, validation_rows_raw)
    max_slot = _max_landmark_slot(train_rows_raw, validation_rows_raw)
    feature_stats = _feature_stats(train_rows, primitive_vocab=primitive_vocab)
    train_sequences = _build_frames(
        train_rows,
        primitive_vocab=primitive_vocab,
        color_vocab=color_vocab,
        max_slot=max_slot,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
        include_object_slot=bool(args.include_object_slot),
    )
    validation_sequences = _build_frames(
        validation_rows,
        primitive_vocab=primitive_vocab,
        color_vocab=color_vocab,
        max_slot=max_slot,
        feature_stats=feature_stats,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
        include_object_slot=bool(args.include_object_slot),
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    label_counts = _label_counts(train_sequences)
    if label_counts["positive"] <= 0 or label_counts["negative"] <= 0:
        raise SystemExit(f"train split lacks both seen labels: {label_counts}")
    geometry_bucket_weights = (
        _geometry_bucket_weights(
            train_sequences,
            max_weight=float(args.max_geometry_bucket_weight),
        )
        if bool(args.balanced_geometry_loss)
        else {"forward": 1.0, "left": 1.0, "right": 1.0}
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

    model = QueryGeometryMemoryProbe(
        aux_dim=aux_dim,
        query_dim=query_dim,
        hidden_dim=int(args.hidden_dim),
        encoder=jepa_encoder,
        encoder_output_dim=jepa_encoder_dim,
        freeze_encoder=(
            args.frozen_jepa_checkpoint is not None and not bool(args.finetune_jepa_encoder)
        ),
        predict_steering=float(args.steering_loss_weight) > 0.0,
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
        loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            range_scale_m=float(args.range_scale_m),
            seen_loss_weight=float(args.seen_loss_weight),
            geometry_loss_weight=float(args.geometry_loss_weight),
            steering_loss_weight=float(args.steering_loss_weight),
            positive_weight=float(args.positive_weight),
            geometry_bucket_weights=geometry_bucket_weights,
        )
        train_metrics = _evaluate(
            model,
            train_sequences,
            device=device,
            range_scale_m=float(args.range_scale_m),
        )
        validation_metrics = _evaluate(
            model,
            validation_sequences,
            device=device,
            range_scale_m=float(args.range_scale_m),
        )
        if float(args.steering_loss_weight) > 0.0:
            score = float(
                validation_metrics["positive_steering_head"]["steering_bucket_accuracy"]
            )
        else:
            score = float(validation_metrics["positive_geometry"]["steering_bucket_accuracy"])
            score -= 0.01 * float(validation_metrics["positive_geometry"]["mean_angle_error_deg"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(loss),
                "train": train_metrics["summary"],
                "validation": validation_metrics["summary"],
            }
        )
        if score >= best_score:
            best_score = score
            best_metrics = validation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            geom = validation_metrics["positive_geometry"]
            steering = validation_metrics["positive_steering_head"]
            seen = validation_metrics["seen_classifier"]
            print(
                f"epoch={epoch}"
                f" loss={loss:.4f}"
                f" seen_bal={seen['balanced_accuracy']:.3f}"
                f" angle_deg={geom['mean_angle_error_deg']:.1f}"
                f" range_mae={geom['mean_range_abs_error_m']:.2f}"
                f" steer_acc={geom['steering_bucket_accuracy']:.3f}"
                f" steering_head_acc={steering['steering_bucket_accuracy']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train = _evaluate(
        model,
        train_sequences,
        device=device,
        range_scale_m=float(args.range_scale_m),
    )
    validation_ablations = {
        ablation: _evaluate(
            model,
            validation_sequences,
            device=device,
            range_scale_m=float(args.range_scale_m),
            ablation=ablation,
        )
        for ablation in (
            "normal",
            "reset_recurrent_state",
            "reverse_input_history",
            "shuffle_hidden_states",
        )
    }
    normal_angle = float(
        validation_ablations["normal"]["positive_geometry"]["mean_angle_error_deg"]
    )
    corrupted_best_angle = min(
        float(validation_ablations[name]["positive_geometry"]["mean_angle_error_deg"])
        for name in ("reset_recurrent_state", "reverse_input_history", "shuffle_hidden_states")
    )
    normal_steer = float(
        validation_ablations["normal"]["positive_geometry"]["steering_bucket_accuracy"]
    )
    corrupted_best_steer = max(
        float(validation_ablations[name]["positive_geometry"]["steering_bucket_accuracy"])
        for name in ("reset_recurrent_state", "reverse_input_history", "shuffle_hidden_states")
    )

    checkpoint = {
        "schema": "lewm_go2_memory_target_geometry_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "feature_mean": feature_stats["mean"].tolist(),
        "feature_std": feature_stats["std"].tolist(),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "aux_dim": int(aux_dim),
        "query_dim": int(query_dim),
        "range_scale_m": float(args.range_scale_m),
        "scrubbed_command_aux": not bool(args.allow_command_aux_leak),
        "scrubbed_runtime_aux": bool(args.scrub_runtime_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "has_steering_head": float(args.steering_loss_weight) > 0.0,
        "steering_classes": _STEERING_CLASSES,
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "geometry_bucket_weights": dict(sorted(geometry_bucket_weights.items())),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_memory_target_geometry_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "scrubbed_command_aux": not bool(args.allow_command_aux_leak),
        "scrubbed_runtime_aux": bool(args.scrub_runtime_aux),
        "scrubbed_scene_aux": bool(args.scrub_scene_aux),
        "has_steering_head": float(args.steering_loss_weight) > 0.0,
        "steering_loss_weight": float(args.steering_loss_weight),
        "steering_classes": _STEERING_CLASSES,
        "include_object_slot": bool(args.include_object_slot),
        "frozen_jepa_checkpoint": (
            str(args.frozen_jepa_checkpoint) if args.frozen_jepa_checkpoint else None
        ),
        "frozen_jepa_report": frozen_jepa_report,
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "range_scale_m": float(args.range_scale_m),
        "geometry_bucket_weights": dict(sorted(geometry_bucket_weights.items())),
        "primitive_vocab": primitive_vocab,
        "color_vocab": color_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(seq) for seq in train_sequences.values()),
        "validation_row_count": sum(len(seq) for seq in validation_sequences.values()),
        "train_label_counts": label_counts,
        "validation_label_counts": _label_counts(validation_sequences),
        "final_train": final_train,
        "validation_ablations": validation_ablations,
        "normal_minus_best_corrupted_steering_bucket_accuracy": (
            normal_steer - corrupted_best_steer
        ),
        "best_corrupted_minus_normal_angle_error_deg": (
            corrupted_best_angle - normal_angle
        ),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Offline target-relative geometry memory probe over strict Go2 "
            "hidden-return rows. If frozen_jepa_checkpoint is set, the visual "
            "encoder is a fixed JEPA-style latent substrate and only the memory "
            "readout is trained. This is still not closed-loop robot navigation."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    normal_geom = validation_ablations["normal"]["positive_geometry"]
    print(
        "go2_memory_target_geometry:"
        f" output={args.output}"
        f" report={report_path}"
        f" angle_deg={normal_geom['mean_angle_error_deg']:.1f}"
        f" range_mae={normal_geom['mean_range_abs_error_m']:.2f}"
        f" steer_acc={normal_geom['steering_bucket_accuracy']:.3f}"
        f" steer_delta={report['normal_minus_best_corrupted_steering_bucket_accuracy']:.3f}"
    )
    return 0


def _scrub_scene_aux(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scrubbed = []
    for row in rows:
        item = dict(row)
        item["clearance_m"] = 0.0
        item["traversability_forward_m"] = 0.0
        scrubbed.append(item)
    return scrubbed


def _build_frames(
    rows: list[dict[str, Any]],
    *,
    primitive_vocab: list[str],
    color_vocab: list[str],
    max_slot: int,
    feature_stats: dict[str, np.ndarray],
    image_size: int,
    range_scale_m: float,
    include_object_slot: bool,
) -> dict[tuple[str, int, int], list[GeometryFrame]]:
    sequences: dict[tuple[str, int, int], list[GeometryFrame]] = {}
    for row in rows:
        landmark_index = _landmark_by_id(row)
        queries = []
        seen_keys: set[tuple[str, str, float]] = set()
        for event in row.get("go2_causal_memory_pair_selection", ()):
            role = str(event.get("pair_role", ""))
            if not (role.startswith("current_") or role == "first_visible_evidence"):
                continue
            object_id = str(event.get("object_id", ""))
            landmark = landmark_index.get(object_id)
            if landmark is None:
                continue
            bearing = _finite_float(landmark.get("bearing_body_rad"), float("nan"))
            range_m = _finite_float(landmark.get("range_m"), float("nan"))
            if not math.isfinite(bearing) or not math.isfinite(range_m):
                continue
            seen_before = 1.0 if bool(event.get("seen_before", False)) else 0.0
            dedup_key = (role, object_id, seen_before)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            color = _object_color(object_id)
            queries.append(
                GeometryQuery(
                    object_id=object_id,
                    color=color,
                    role=role,
                    seen_before=seen_before,
                    features=torch.tensor(
                        _query_features(
                            object_id,
                            color_vocab=color_vocab,
                            max_slot=max_slot,
                            include_object_slot=include_object_slot,
                        ),
                        dtype=torch.float32,
                    ),
                    bearing_rad=bearing,
                    range_m=max(0.0, min(float(range_scale_m), range_m)),
                )
            )
        aux = _aux_features(row, primitive_vocab=primitive_vocab)
        aux = (aux - feature_stats["mean"]) / feature_stats["std"]
        frame = GeometryFrame(
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


def _train_epoch(
    model: QueryGeometryMemoryProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[GeometryFrame]],
    *,
    device: torch.device,
    range_scale_m: float,
    seen_loss_weight: float,
    geometry_loss_weight: float,
    steering_loss_weight: float,
    positive_weight: float,
    geometry_bucket_weights: dict[str, float],
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
        batch = _query_batch(
            sequence,
            hidden,
            device=device,
            range_scale_m=range_scale_m,
            geometry_bucket_weights=geometry_bucket_weights,
        )
        if batch is None:
            continue
        (
            hidden_rows,
            query_features,
            seen_targets,
            geom_targets,
            geom_mask,
            geom_weights,
            steering_targets,
        ) = batch
        seen_logits, geom_pred, steering_logits = model.score_queries_with_steering(
            hidden_rows,
            query_features,
        )
        pos_weight = torch.tensor(float(positive_weight), dtype=torch.float32, device=device)
        seen_loss = F.binary_cross_entropy_with_logits(
            seen_logits,
            seen_targets,
            pos_weight=pos_weight,
        )
        geometry_loss = _geometry_loss(geom_pred, geom_targets, geom_mask, geom_weights)
        steering_loss = _steering_loss(
            steering_logits,
            steering_targets,
            geom_mask,
            geom_weights,
        )
        loss = (
            float(seen_loss_weight) * seen_loss
            + float(geometry_loss_weight) * geometry_loss
            + float(steering_loss_weight) * steering_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        trained += 1
    return total_loss / max(1, trained)


def _evaluate(
    model: QueryGeometryMemoryProbe,
    sequences: dict[tuple[str, int, int], list[GeometryFrame]],
    *,
    device: torch.device,
    range_scale_m: float,
    ablation: str = "normal",
) -> dict[str, Any]:
    model.eval()
    seen_metrics = _SeenMetrics()
    geom = _GeometryMetrics()
    steering_head = _SteeringHeadMetrics()
    geom_by_color: dict[str, _GeometryMetrics] = defaultdict(_GeometryMetrics)
    steering_head_by_color: dict[str, _SteeringHeadMetrics] = defaultdict(_SteeringHeadMetrics)
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
                queries = [query for query in frame.queries if query.role.startswith("current_")]
                if not queries:
                    continue
                query_features = torch.stack([query.features for query in queries]).to(device)
                hidden_rows = hidden[step_idx].repeat(len(queries), 1)
                seen_logits, geom_pred, steering_logits = model.score_queries_with_steering(
                    hidden_rows,
                    query_features,
                )
                seen_probs = torch.sigmoid(seen_logits).detach().cpu().numpy()
                geom_np = geom_pred.detach().cpu().numpy()
                steering_np = (
                    steering_logits.detach().cpu().numpy()
                    if steering_logits is not None
                    else None
                )
                for query_idx, (query, prob, pred) in enumerate(zip(queries, seen_probs, geom_np)):
                    seen_metrics.add(prediction=float(prob) >= 0.5, target=query.seen_before >= 0.5)
                    if query.seen_before < 0.5:
                        continue
                    geom.add(pred, query, range_scale_m=range_scale_m)
                    geom_by_color[query.color].add(pred, query, range_scale_m=range_scale_m)
                    if steering_np is not None:
                        steering_head.add(steering_np[query_idx], query)
                        steering_head_by_color[query.color].add(steering_np[query_idx], query)
    result = {
        "ablation": ablation,
        "seen_classifier": seen_metrics.to_dict(),
        "positive_geometry": geom.to_dict(),
        "positive_steering_head": steering_head.to_dict(),
        "positive_geometry_by_color": {
            key: value.to_dict() for key, value in sorted(geom_by_color.items())
        },
        "positive_steering_head_by_color": {
            key: value.to_dict() for key, value in sorted(steering_head_by_color.items())
        },
    }
    result["summary"] = {
        "seen_balanced_accuracy": result["seen_classifier"]["balanced_accuracy"],
        "mean_angle_error_deg": result["positive_geometry"]["mean_angle_error_deg"],
        "mean_range_abs_error_m": result["positive_geometry"]["mean_range_abs_error_m"],
        "steering_bucket_accuracy": result["positive_geometry"][
            "steering_bucket_accuracy"
        ],
        "steering_head_accuracy": result["positive_steering_head"][
            "steering_bucket_accuracy"
        ],
    }
    return result


def _hidden_states_by_sequence(
    model: QueryGeometryMemoryProbe,
    sequences: dict[tuple[str, int, int], list[GeometryFrame]],
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


def _query_batch(
    sequence: list[GeometryFrame],
    hidden: torch.Tensor,
    *,
    device: torch.device,
    range_scale_m: float,
    geometry_bucket_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    hidden_rows = []
    query_features = []
    seen_targets = []
    geom_targets = []
    geom_mask = []
    geom_weights = []
    steering_targets = []
    for step_idx, frame in enumerate(sequence):
        for query in frame.queries:
            hidden_rows.append(hidden[step_idx])
            query_features.append(query.features)
            seen_targets.append(float(query.seen_before))
            geom_targets.append(_geometry_target(query, range_scale_m=range_scale_m))
            geom_mask.append(1.0 if query.seen_before >= 0.5 else 0.0)
            bucket = _steering_bucket(query.bearing_rad)
            geom_weights.append(float((geometry_bucket_weights or {}).get(bucket, 1.0)))
            steering_targets.append(_steering_class_index(bucket))
    if not hidden_rows:
        return None
    return (
        torch.stack(hidden_rows).to(device),
        torch.stack(query_features).to(device),
        torch.tensor(seen_targets, dtype=torch.float32, device=device),
        torch.tensor(geom_targets, dtype=torch.float32, device=device),
        torch.tensor(geom_mask, dtype=torch.float32, device=device),
        torch.tensor(geom_weights, dtype=torch.float32, device=device),
        torch.tensor(steering_targets, dtype=torch.long, device=device),
    )


def _geometry_target(query: GeometryQuery, *, range_scale_m: float) -> list[float]:
    return [
        math.sin(query.bearing_rad),
        math.cos(query.bearing_rad),
        max(0.0, min(float(range_scale_m), query.range_m)) / float(range_scale_m),
    ]


def _geometry_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if float(mask.sum().detach().cpu()) <= 0.0:
        return pred.sum() * 0.0
    loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    effective = mask * weights
    return (loss * effective).sum() / effective.sum().clamp_min(1.0)


def _steering_loss(
    steering_logits: torch.Tensor | None,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if steering_logits is None or float(mask.sum().detach().cpu()) <= 0.0:
        return mask.sum() * 0.0
    loss = F.cross_entropy(steering_logits, targets, reduction="none")
    effective = mask * weights
    return (loss * effective).sum() / effective.sum().clamp_min(1.0)


def _sequence_tensors(
    sequence: list[GeometryFrame],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.stack([item.image for item in sequence]).to(device),
        torch.stack([item.aux for item in sequence]).to(device),
    )


def _query_features(
    object_id: str,
    *,
    color_vocab: list[str],
    max_slot: int,
    include_object_slot: bool,
) -> list[float]:
    color = _object_color(object_id)
    features = list(_COLOR_RGB.get(color, _COLOR_RGB["unknown"]))
    features.extend(1.0 if color == item else 0.0 for item in color_vocab)
    if include_object_slot:
        slot = _landmark_slot(object_id)
        features.extend([slot / float(max(1, max_slot)), math.sin(slot), math.cos(slot)])
    return features


class _SeenMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def add(self, *, prediction: bool, target: bool) -> None:
        if prediction and target:
            self.tp += 1
        elif prediction and not target:
            self.fp += 1
        elif not prediction and target:
            self.fn += 1
        else:
            self.tn += 1

    def to_dict(self) -> dict[str, float]:
        positive = self.tp + self.fn
        negative = self.tn + self.fp
        total = positive + negative
        recall = self.tp / max(1, positive)
        specificity = self.tn / max(1, negative)
        precision = self.tp / max(1, self.tp + self.fp)
        f1 = (2.0 * self.tp) / max(1, 2 * self.tp + self.fp + self.fn)
        return {
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
            "positive_count": float(positive),
            "negative_count": float(negative),
            "target_count": float(total),
        }


class _GeometryMetrics:
    def __init__(self) -> None:
        self.angle_errors: list[float] = []
        self.range_errors: list[float] = []
        self.steering_correct = 0
        self.count = 0
        self.true_buckets: Counter[str] = Counter()
        self.pred_buckets: Counter[str] = Counter()

    def add(self, pred: np.ndarray, query: GeometryQuery, *, range_scale_m: float) -> None:
        pred_bearing = math.atan2(float(pred[0]), float(pred[1]))
        angle_error = abs(_wrap_angle(pred_bearing - query.bearing_rad))
        pred_range = max(0.0, float(pred[2]) * float(range_scale_m))
        range_error = abs(pred_range - query.range_m)
        true_bucket = _steering_bucket(query.bearing_rad)
        pred_bucket = _steering_bucket(pred_bearing)
        self.angle_errors.append(math.degrees(angle_error))
        self.range_errors.append(range_error)
        self.steering_correct += 1 if true_bucket == pred_bucket else 0
        self.count += 1
        self.true_buckets[true_bucket] += 1
        self.pred_buckets[pred_bucket] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": float(self.count),
            "mean_angle_error_deg": float(np.mean(self.angle_errors)) if self.angle_errors else 0.0,
            "median_angle_error_deg": (
                float(np.median(self.angle_errors)) if self.angle_errors else 0.0
            ),
            "mean_range_abs_error_m": float(np.mean(self.range_errors)) if self.range_errors else 0.0,
            "median_range_abs_error_m": (
                float(np.median(self.range_errors)) if self.range_errors else 0.0
            ),
            "steering_bucket_accuracy": self.steering_correct / max(1, self.count),
            "true_steering_bucket_counts": dict(sorted(self.true_buckets.items())),
            "predicted_steering_bucket_counts": dict(sorted(self.pred_buckets.items())),
        }


class _SteeringHeadMetrics:
    def __init__(self) -> None:
        self.correct = 0
        self.count = 0
        self.true_buckets: Counter[str] = Counter()
        self.pred_buckets: Counter[str] = Counter()

    def add(self, logits: np.ndarray, query: GeometryQuery) -> None:
        true_bucket = _steering_bucket(query.bearing_rad)
        pred_index = int(np.argmax(logits))
        pred_bucket = _STEERING_CLASSES[max(0, min(pred_index, len(_STEERING_CLASSES) - 1))]
        self.correct += 1 if true_bucket == pred_bucket else 0
        self.count += 1
        self.true_buckets[true_bucket] += 1
        self.pred_buckets[pred_bucket] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_count": float(self.count),
            "steering_bucket_accuracy": self.correct / max(1, self.count),
            "true_steering_bucket_counts": dict(sorted(self.true_buckets.items())),
            "predicted_steering_bucket_counts": dict(sorted(self.pred_buckets.items())),
        }


def _steering_bucket(bearing_rad: float) -> str:
    if bearing_rad > 0.35:
        return "left"
    if bearing_rad < -0.35:
        return "right"
    return "forward"


_STEERING_CLASSES = ["right", "forward", "left"]


def _steering_class_index(bucket: str) -> int:
    return _STEERING_CLASSES.index(bucket)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _label_counts(
    sequences: dict[tuple[str, int, int], list[GeometryFrame]],
) -> dict[str, int]:
    positive = 0
    negative = 0
    by_color: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                if query.seen_before >= 0.5:
                    positive += 1
                    by_color[query.color]["positive"] += 1
                else:
                    negative += 1
                    by_color[query.color]["negative"] += 1
    return {
        "positive": positive,
        "negative": negative,
        "total": positive + negative,
        "by_color": {key: dict(value) for key, value in sorted(by_color.items())},
    }


def _geometry_bucket_weights(
    sequences: dict[tuple[str, int, int], list[GeometryFrame]],
    *,
    max_weight: float,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                if query.seen_before >= 0.5:
                    counts[_steering_bucket(query.bearing_rad)] += 1
    if not counts:
        return {"forward": 1.0, "left": 1.0, "right": 1.0}
    mean_count = sum(counts.values()) / max(1, len(counts))
    weights = {
        bucket: min(float(max_weight), max(1.0, mean_count / max(1, count)))
        for bucket, count in counts.items()
    }
    for bucket in ("forward", "left", "right"):
        weights.setdefault(bucket, 1.0)
    return weights


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
    for part in object_id.split("_"):
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
