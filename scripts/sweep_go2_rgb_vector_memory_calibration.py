#!/usr/bin/env python3
"""Sweep deterministic RGB vector-memory calibration for Go2 hidden-target rows."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    STEERING_CLASSES,
    _COLOR_RGB,
    _build_sequences,
    _color_vocab,
    _load_rows,
    _propagate_vectors,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--motion-translation-scales", nargs="+", type=float, default=[6, 8, 10, 12])
    parser.add_argument("--forward-values", nargs="+", type=float, default=[0.05, 0.15, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--lateral-scales", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--vector-scales", nargs="+", type=float, default=[0.75, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--write-logit-scales", nargs="+", type=float, default=[4.0, 6.0, 8.0, 10.0])
    parser.add_argument("--write-biases", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--rgb-evidence-sigma", type=float, default=0.20)
    parser.add_argument("--rgb-evidence-threshold", type=float, default=0.55)
    parser.add_argument("--rgb-evidence-temperature", type=float, default=0.08)
    parser.add_argument("--rgb-evidence-area-threshold", type=float, default=0.006)
    args = parser.parse_args()

    rows = _load_rows(args.datasets)
    color_vocab = _color_vocab(rows)
    color_index = {color: idx for idx, color in enumerate(color_vocab)}
    color_rgb = torch.tensor([_COLOR_RGB[color] for color in color_vocab], dtype=torch.float32)
    results: list[dict[str, Any]] = []
    for motion_scale in args.motion_translation_scales:
        sequences = _build_sequences(
            rows,
            color_index=color_index,
            image_size=int(args.image_size),
            range_scale_m=float(args.range_scale_m),
            motion_translation_scale_m=float(motion_scale),
        )
        rgb_cache = _precompute_rgb_readouts(
            sequences,
            color_rgb=color_rgb,
            sigma=float(args.rgb_evidence_sigma),
            evidence_threshold=float(args.rgb_evidence_threshold),
            temperature=float(args.rgb_evidence_temperature),
            area_threshold=float(args.rgb_evidence_area_threshold),
        )
        for forward in args.forward_values:
            for lateral_scale in args.lateral_scales:
                for vector_scale in args.vector_scales:
                    for write_logit_scale in args.write_logit_scales:
                        for write_bias in args.write_biases:
                            for threshold in args.thresholds:
                                metrics = _evaluate(
                                    sequences,
                                    rgb_cache=rgb_cache,
                                    color_rgb=color_rgb,
                                    forward=float(forward),
                                    lateral_scale=float(lateral_scale),
                                    vector_scale=float(vector_scale),
                                    write_logit_scale=float(write_logit_scale),
                                    write_bias=float(write_bias),
                                    threshold=float(threshold),
                                )
                                results.append(
                                    {
                                        **metrics,
                                        "motion_translation_scale_m": float(motion_scale),
                                        "forward": float(forward),
                                        "lateral_scale": float(lateral_scale),
                                        "vector_scale": float(vector_scale),
                                        "write_logit_scale": float(write_logit_scale),
                                        "write_bias": float(write_bias),
                                        "threshold": float(threshold),
                                    }
                                )
    results.sort(
        key=lambda item: (
            item["target_steering_pipeline_success"],
            item["target_recall"],
            -item["false_claim_rate"],
            item["target_selection_precision"],
        ),
        reverse=True,
    )
    print(json.dumps(results[: int(args.top_k)], indent=2, sort_keys=True))
    return 0


def _evaluate(
    sequences: dict[tuple[str, int, int], list[Any]],
    *,
    rgb_cache: dict[tuple[tuple[str, int, int], int], tuple[torch.Tensor, torch.Tensor]],
    color_rgb: torch.Tensor,
    forward: float,
    lateral_scale: float,
    vector_scale: float,
    write_logit_scale: float,
    write_bias: float,
    threshold: float,
) -> dict[str, Any]:
    positive = negative = selected = correct = false_claim = steer_ok = missed = 0
    pred_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for seq_key, sequence in sequences.items():
        color_count = int(color_rgb.shape[0])
        memory_vec = torch.zeros(color_count, 2)
        memory_conf = torch.zeros(color_count)
        for step_idx, frame in enumerate(sequence):
            if step_idx > 0:
                memory_vec = _propagate_vectors(memory_vec, frame.motion_block)
            logits, x_centroid = rgb_cache[(seq_key, step_idx)]
            vec = _rgb_vector(
                x_centroid,
                forward=float(forward),
                lateral_scale=float(lateral_scale),
                vector_scale=float(vector_scale),
            )
            write = torch.sigmoid(float(write_logit_scale) * logits - float(write_bias))
            propagated_weight = memory_conf * (1.0 - write)
            new_conf = 1.0 - (1.0 - memory_conf) * (1.0 - write)
            memory_vec = (
                propagated_weight.unsqueeze(-1) * memory_vec
                + write.unsqueeze(-1) * vec
            ) / new_conf.clamp_min(1e-4).unsqueeze(-1)
            memory_conf = new_conf.clamp(0.0, 1.0)
            for query in frame.queries:
                color_idx = int(query.color_index)
                is_positive = query.target >= 0.5
                score = float(memory_conf[color_idx])
                is_selected = score >= float(threshold)
                if is_positive:
                    positive += 1
                    if not is_selected:
                        missed += 1
                        continue
                    selected += 1
                    correct += 1
                    pred = _vector_steering_index(memory_vec[color_idx])
                    target = int(query.target_steering)
                    pred_counts[STEERING_CLASSES[pred]] += 1
                    target_counts[STEERING_CLASSES[target]] += 1
                    steer_ok += int(pred == target)
                else:
                    negative += 1
                    if is_selected:
                        selected += 1
                        false_claim += 1
    return {
        "positive_frame_count": positive,
        "negative_frame_count": negative,
        "selected_frame_count": selected,
        "correct_target_count": correct,
        "target_steering_success_count": steer_ok,
        "missed_positive_count": missed,
        "false_claim_count": false_claim,
        "target_recall": correct / max(1, positive),
        "target_steering_pipeline_success": steer_ok / max(1, positive),
        "false_claim_rate": false_claim / max(1, negative),
        "target_selection_precision": correct / max(1, selected),
        "predicted_steering_counts": dict(sorted(pred_counts.items())),
        "target_steering_counts": dict(sorted(target_counts.items())),
    }


def _precompute_rgb_readouts(
    sequences: dict[tuple[str, int, int], list[Any]],
    *,
    color_rgb: torch.Tensor,
    sigma: float,
    evidence_threshold: float,
    temperature: float,
    area_threshold: float,
) -> dict[tuple[tuple[str, int, int], int], tuple[torch.Tensor, torch.Tensor]]:
    cache: dict[tuple[tuple[str, int, int], int], tuple[torch.Tensor, torch.Tensor]] = {}
    for seq_key, sequence in sequences.items():
        for step_idx, frame in enumerate(sequence):
            cache[(seq_key, step_idx)] = _rgb_readout(
                frame.image.unsqueeze(0),
                color_rgb=color_rgb,
                sigma=float(sigma),
                evidence_threshold=float(evidence_threshold),
                temperature=float(temperature),
                area_threshold=float(area_threshold),
            )
    return cache


def _rgb_readout(
    images: torch.Tensor,
    *,
    color_rgb: torch.Tensor,
    sigma: float,
    evidence_threshold: float,
    temperature: float,
    area_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = ((images[:, None] - color_rgb[None, :, :, None, None]) ** 2).mean(dim=2)
    similarity = torch.exp(-distance / (2.0 * float(sigma) ** 2))
    soft_mask = torch.sigmoid(
        (similarity - float(evidence_threshold)) / float(temperature)
    )
    area = soft_mask.mean(dim=(-1, -2)).clamp_min(1e-8)
    area_logits = torch.log(area) - math.log(float(area_threshold))
    _, _, _, width = soft_mask.shape
    x_coords = torch.linspace(-1.0, 1.0, width, dtype=images.dtype)
    x_centroid = (
        soft_mask * x_coords.reshape(1, 1, 1, width)
    ).sum(dim=(-1, -2)) / soft_mask.sum(dim=(-1, -2)).clamp_min(1e-6)
    return area_logits[0], x_centroid[0]


def _rgb_vector(
    x_centroid: torch.Tensor,
    *,
    forward: float,
    lateral_scale: float,
    vector_scale: float,
) -> torch.Tensor:
    raw = torch.stack(
        [
            torch.full_like(x_centroid, float(forward)),
            -float(lateral_scale) * x_centroid.clamp(-1.0, 1.0),
        ],
        dim=-1,
    )
    return torch.tanh(float(vector_scale) * raw)


def _vector_steering_index(vector: torch.Tensor) -> int:
    bearing = math.atan2(float(vector[1]), float(vector[0]))
    if bearing <= -0.1:
        return 0
    if bearing >= 0.1:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
