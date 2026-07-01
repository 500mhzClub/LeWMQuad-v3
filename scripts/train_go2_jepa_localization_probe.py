#!/usr/bin/env python3
"""Train a pure RGB/JEPA recurrent localization probe for Go2 strict memory rows.

The probe uses rendered RGB through a frozen JEPA encoder plus odometry/action
aux history. Geometry labels are used only as offline supervision for
``cell_id`` and ``yaw_bin``. Runtime inputs do not include landmark ids, slots,
bearings, ranges, object geometry, or map state.
"""

from __future__ import annotations

import argparse
import json
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
    SpatialGo2JepaFeatureEncoder,
    _aux_features,
)


@dataclass(frozen=True)
class Frame:
    seq_key: tuple[str, int, int]
    episode_step: int
    image: torch.Tensor
    aux: torch.Tensor
    cell_index: int
    yaw_index: int
    query_buckets: tuple[str, ...]


class RecurrentLocalizationProbe(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_output_dim: int,
        aux_dim: int,
        hidden_dim: int,
        cell_count: int,
        yaw_count: int,
        freeze_encoder: bool,
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
            else nn.Sequential(nn.Linear(int(encoder_output_dim), int(hidden_dim)), nn.GELU())
        )
        self.recurrent = nn.GRUCell(int(hidden_dim) + int(aux_dim), int(hidden_dim))
        self.cell_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(cell_count)),
        )
        self.yaw_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, int(yaw_count)),
        )

    def forward_sequence(self, images: torch.Tensor, aux: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.encoder(images)
        else:
            encoded = self.encoder(images)
        hidden = self.encoder_projection(encoded)
        state = torch.zeros(hidden.shape[-1], device=images.device, dtype=hidden.dtype)
        states = []
        for idx in range(images.shape[0]):
            state = self.recurrent(torch.cat([hidden[idx], aux[idx]], dim=-1), state)
            states.append(state)
        stacked = torch.stack(states)
        return {
            "cell_logits": self.cell_head(stacked),
            "yaw_logits": self.yaw_head(stacked),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--spatial-output-dim", type=int, default=512)
    parser.add_argument("--spatial-feature-stride", type=int, choices=(8, 16), default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune-jepa-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
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
    cell_vocab = sorted(
        {
            int(row.get("cell_id", -1))
            for row in train_rows + validation_rows
            if int(row.get("cell_id", -1)) >= 0
        }
    )
    yaw_vocab = sorted(
        {
            int(row.get("yaw_bin", -1))
            for row in train_rows + validation_rows
            if int(row.get("yaw_bin", -1)) >= 0
        }
    )
    cell_index = {value: idx for idx, value in enumerate(cell_vocab)}
    yaw_index = {value: idx for idx, value in enumerate(yaw_vocab)}
    train_sequences = _build_sequences(
        train_rows,
        cell_index=cell_index,
        yaw_index=yaw_index,
        image_size=int(args.image_size),
    )
    validation_sequences = _build_sequences(
        validation_rows,
        cell_index=cell_index,
        yaw_index=yaw_index,
        image_size=int(args.image_size),
    )
    aux_stats = _aux_stats(train_sequences)
    _normalize_aux(train_sequences, aux_stats)
    _normalize_aux(validation_sequences, aux_stats)

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
    model = RecurrentLocalizationProbe(
        encoder=encoder,
        encoder_output_dim=int(args.spatial_output_dim),
        aux_dim=next(iter(train_sequences.values()))[0].aux.numel(),
        hidden_dim=int(args.hidden_dim),
        cell_count=len(cell_vocab),
        yaw_count=len(yaw_vocab),
        freeze_encoder=not bool(args.finetune_jepa_encoder),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = _train_epoch(model, optimizer, train_sequences, device=device)
        validation_metrics = _evaluate(model, validation_sequences, device=device)
        score = float(validation_metrics["cell_yaw_accuracy"])
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation": validation_metrics,
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
            print(
                f"epoch={epoch}"
                f" loss={train_loss:.4f}"
                f" val_cell={validation_metrics['cell_accuracy']:.3f}"
                f" val_yaw={validation_metrics['yaw_accuracy']:.3f}"
                f" val_cell_yaw={validation_metrics['cell_yaw_accuracy']:.3f}"
                f" val_positive_hidden={validation_metrics['bucket_cell_yaw_accuracy'].get('positive_current_hidden', 0.0):.3f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate(model, train_sequences, device=device)
    final_validation = _evaluate(model, validation_sequences, device=device)
    checkpoint = {
        "schema": "lewm_go2_jepa_localization_probe_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "frozen_jepa_report": {
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
        },
        "cell_vocab": cell_vocab,
        "yaw_vocab": yaw_vocab,
        "aux_mean": aux_stats["mean"].tolist(),
        "aux_std": aux_stats["std"].tolist(),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    report = {
        "schema": "lewm_go2_jepa_localization_probe_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "cell_vocab": cell_vocab,
        "yaw_vocab": yaw_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(sequence) for sequence in train_sequences.values()),
        "validation_row_count": sum(len(sequence) for sequence in validation_sequences.values()),
        "final_train": final_train,
        "final_validation": final_validation,
        "best_validation": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Pure RGB/JEPA recurrent localization probe. Inference uses rendered RGB, "
            "JEPA latents, and odometry/action aux history. Geometry labels supervise "
            "cell/yaw only offline."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_jepa_localization_probe:"
        f" output={args.output}"
        f" report={report_path}"
        f" cell={final_validation['cell_accuracy']:.3f}"
        f" yaw={final_validation['yaw_accuracy']:.3f}"
        f" cell_yaw={final_validation['cell_yaw_accuracy']:.3f}"
        f" positive_hidden={final_validation['bucket_cell_yaw_accuracy'].get('positive_current_hidden', 0.0):.3f}",
        flush=True,
    )
    return 0


def _build_sequences(
    rows: list[dict[str, Any]],
    *,
    cell_index: dict[int, int],
    yaw_index: dict[int, int],
    image_size: int,
) -> dict[tuple[str, int, int], list[Frame]]:
    sequences: dict[tuple[str, int, int], list[Frame]] = defaultdict(list)
    for row in rows:
        cell_id = int(row.get("cell_id", -1))
        yaw_bin = int(row.get("yaw_bin", -1))
        if cell_id not in cell_index or yaw_bin not in yaw_index:
            continue
        sequences[_seq_key(row)].append(
            Frame(
                seq_key=_seq_key(row),
                episode_step=int(row.get("episode_step", 0)),
                image=_load_image(Path(row["rgb_path"]), image_size=image_size),
                aux=torch.tensor(_aux_features(row), dtype=torch.float32),
                cell_index=int(cell_index[cell_id]),
                yaw_index=int(yaw_index[yaw_bin]),
                query_buckets=tuple(_query_buckets(row)),
            )
        )
    result = {}
    for key, sequence in sequences.items():
        result[key] = sorted(sequence, key=lambda frame: frame.episode_step)
    return result


def _train_epoch(
    model: RecurrentLocalizationProbe,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total = 0.0
    trained = 0
    for key in keys:
        batch = _sequence_tensors(sequences[key], device=device)
        outputs = model.forward_sequence(batch["images"], batch["aux"])
        loss = F.cross_entropy(outputs["cell_logits"], batch["cell"])
        loss = loss + F.cross_entropy(outputs["yaw_logits"], batch["yaw"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        trained += 1
    return total / max(1, trained)


def _evaluate(
    model: RecurrentLocalizationProbe,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    cell_ok = 0
    yaw_ok = 0
    both_ok = 0
    bucket_total: Counter[str] = Counter()
    bucket_both_ok: Counter[str] = Counter()
    with torch.no_grad():
        for sequence in sequences.values():
            batch = _sequence_tensors(sequence, device=device)
            outputs = model.forward_sequence(batch["images"], batch["aux"])
            cell_pred = torch.argmax(outputs["cell_logits"], dim=-1).cpu()
            yaw_pred = torch.argmax(outputs["yaw_logits"], dim=-1).cpu()
            cell = batch["cell"].cpu()
            yaw = batch["yaw"].cpu()
            for idx, frame in enumerate(sequence):
                total += 1
                c_ok = int(cell_pred[idx]) == int(cell[idx])
                y_ok = int(yaw_pred[idx]) == int(yaw[idx])
                cell_ok += int(c_ok)
                yaw_ok += int(y_ok)
                both = c_ok and y_ok
                both_ok += int(both)
                for bucket in frame.query_buckets:
                    bucket_total[bucket] += 1
                    bucket_both_ok[bucket] += int(both)
    return {
        "frame_count": int(total),
        "cell_accuracy": float(cell_ok) / float(max(1, total)),
        "yaw_accuracy": float(yaw_ok) / float(max(1, total)),
        "cell_yaw_accuracy": float(both_ok) / float(max(1, total)),
        "bucket_counts": dict(sorted(bucket_total.items())),
        "bucket_cell_yaw_accuracy": {
            key: float(bucket_both_ok[key]) / float(max(1, count))
            for key, count in sorted(bucket_total.items())
        },
    }


def _sequence_tensors(sequence: list[Frame], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([frame.image for frame in sequence]).to(device),
        "aux": torch.stack([frame.aux for frame in sequence]).to(device),
        "cell": torch.tensor([frame.cell_index for frame in sequence], dtype=torch.long, device=device),
        "yaw": torch.tensor([frame.yaw_index for frame in sequence], dtype=torch.long, device=device),
    }


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
                cell_index=frame.cell_index,
                yaw_index=frame.yaw_index,
                query_buckets=frame.query_buckets,
            )
            for frame in sequence
        ]


def _query_buckets(row: dict[str, Any]) -> list[str]:
    result = []
    landmark_by_id = {
        str(landmark.get("object_id", "")): landmark
        for landmark in row.get("landmarks", ())
        if str(landmark.get("object_id", ""))
    }
    seen_keys: set[tuple[str, bool]] = set()
    for event in row.get("go2_causal_memory_pair_selection", ()):
        role = str(event.get("pair_role", ""))
        if not role.startswith("current_"):
            continue
        object_id = str(event.get("object_id", ""))
        color = _object_color(object_id)
        target = bool(event.get("seen_before", False))
        dedup_key = (color, target)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        landmark = landmark_by_id.get(object_id)
        if landmark is None:
            continue
        current_visible = bool(landmark.get("visible", False))
        prefix = "positive" if target else "negative"
        if current_visible:
            result.append(f"{prefix}_current_visible")
        else:
            result.append(f"{prefix}_current_hidden")
    return result


def _object_color(object_id: str) -> str:
    lowered = str(object_id).lower()
    for color in ("blue", "green", "red", "yellow"):
        if color in lowered:
            return color
    return "unknown"


def _seq_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("scene_id", "")),
        int(row.get("env_idx", -1)),
        int(row.get("episode_id", -1)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
