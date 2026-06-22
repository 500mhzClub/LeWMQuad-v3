#!/usr/bin/env python3
"""Train a pure RGB/JEPA color-query vector-memory Go2 controller.

Inference inputs are rendered RGB, a JEPA-initialized visual encoder,
odometry/action history, and a target color query. Runtime landmark ids, object
slots, detector visibility, bearing/range, and scene/map geometry are not fed
to the controller. Label geometry is used only as offline supervision.
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


STEERING_CLASSES = ("right", "forward", "left")
PRIMITIVE_NAMES = (
    "forward_medium",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
    "hold",
)
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


class SpatialGo2JepaFeatureEncoder(nn.Module):
    """JEPA-initialized spatial feature readout that preserves image layout."""

    def __init__(
        self,
        base_encoder: nn.Module,
        *,
        image_size: int,
        output_dim: int,
        feature_stride: int,
    ) -> None:
        super().__init__()
        if int(feature_stride) == 8:
            layer_count = 6
            channel_count = 96
        elif int(feature_stride) == 16:
            layer_count = 8
            channel_count = 128
        else:
            raise ValueError(f"unsupported spatial feature stride: {feature_stride}")
        self.feature_stride = int(feature_stride)
        self.features = nn.Sequential(*list(base_encoder.net.children())[:layer_count])
        grid_size = max(1, int(image_size) // int(feature_stride))
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm((channel_count + 2) * grid_size * grid_size),
            nn.Linear((channel_count + 2) * grid_size * grid_size, int(output_dim)),
            nn.GELU(),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        batch, _, height, width = features.shape
        y_coords = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=features.dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=features.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)
        return self.projection(torch.cat([features, coords], dim=1).flatten(1))


@dataclass(frozen=True)
class Query:
    color_index: int
    target: float
    target_vec: torch.Tensor
    target_steering: int
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
    motion_block: torch.Tensor
    motion_window: torch.Tensor
    exact_motion: torch.Tensor
    visible_mask: torch.Tensor
    visible_vec: torch.Tensor
    all_vec: torch.Tensor
    memory_mask: torch.Tensor
    queries: tuple[Query, ...]


class ColorVectorMemoryController(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        encoder_output_dim: int,
        color_count: int,
        aux_dim: int,
        hidden_dim: int,
        color_embedding_dim: int,
        freeze_encoder: bool,
        color_rgb: torch.Tensor,
        rgb_color_evidence: bool,
        rgb_evidence_replaces_learned: bool,
        rgb_evidence_sigma: float,
        rgb_evidence_threshold: float,
        rgb_evidence_temperature: float,
        rgb_evidence_area_threshold: float,
        rgb_evidence_logit_scale: float,
        rgb_evidence_replaces_learned_logits_only: bool,
        rgb_vector_scale: float,
        rgb_vector_calibrated: bool,
        rgb_vector_bearing_a: float,
        rgb_vector_bearing_b: float,
        rgb_vector_range_loglog_m: float,
        rgb_vector_range_loglog_c: float,
        range_scale_m: float,
        evidence_write_logit_bias: float,
        evidence_write_temperature: float,
        read_head_scale: float,
        read_confidence_prior_scale: float,
        latent_memory_features: bool,
        world_belief_features: bool,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.color_count = int(color_count)
        self.range_limit = 1.0
        self.rgb_color_evidence = bool(rgb_color_evidence)
        self.rgb_evidence_replaces_learned = bool(rgb_evidence_replaces_learned)
        self.rgb_evidence_sigma = max(1e-4, float(rgb_evidence_sigma))
        self.rgb_evidence_threshold = float(rgb_evidence_threshold)
        self.rgb_evidence_temperature = max(1e-4, float(rgb_evidence_temperature))
        self.rgb_evidence_area_threshold = max(1e-6, float(rgb_evidence_area_threshold))
        self.rgb_evidence_logit_scale = float(rgb_evidence_logit_scale)
        self.rgb_evidence_replaces_learned_logits_only = bool(
            rgb_evidence_replaces_learned_logits_only
        )
        self.rgb_vector_scale = float(rgb_vector_scale)
        # Calibrated ranged write: emit a true relative position
        # [r*cos(bearing), r*sin(bearing)] (range_scale-normalized) from the
        # color-mask centroid (bearing) and area (range), so _propagate_vectors'
        # rigid-body update preserves bearing as the body moves. Replaces the
        # fixed-range proxy [0.75, -x_centroid] that cannot carry out-of-frame
        # steering through odometry. Coefficients are fit on training data by
        # scripts/audit_go2_rgb_bearing_range_calibration.py.
        self.rgb_vector_calibrated = bool(rgb_vector_calibrated)
        self.rgb_vector_bearing_a = float(rgb_vector_bearing_a)
        self.rgb_vector_bearing_b = float(rgb_vector_bearing_b)
        self.rgb_vector_range_loglog_m = float(rgb_vector_range_loglog_m)
        self.rgb_vector_range_loglog_c = float(rgb_vector_range_loglog_c)
        self.range_scale_m = max(1e-6, float(range_scale_m))
        self.register_buffer("color_rgb", color_rgb.float().clamp(0.0, 1.0), persistent=False)
        self.evidence_write_logit_bias = float(evidence_write_logit_bias)
        self.evidence_write_temperature = max(1e-3, float(evidence_write_temperature))
        self.read_head_scale = float(read_head_scale)
        self.read_confidence_prior_scale = float(read_confidence_prior_scale)
        self.latent_memory_features = bool(latent_memory_features)
        self.world_belief_features = bool(world_belief_features)
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
        self.evidence_head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.color_count * 3),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(int(aux_dim), int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 3),
        )
        self.recurrent_memory = nn.GRUCell(int(hidden_dim) + int(aux_dim), int(hidden_dim))
        self.color_embedding = nn.Embedding(self.color_count, int(color_embedding_dim))
        if self.world_belief_features:
            self.world_belief_head = nn.Sequential(
                nn.LayerNorm(int(hidden_dim) + int(color_embedding_dim)),
                nn.Linear(int(hidden_dim) + int(color_embedding_dim), int(hidden_dim) // 2),
                nn.GELU(),
                nn.Linear(int(hidden_dim) // 2, 3),
            )
        read_dim = int(color_embedding_dim) + int(hidden_dim) + 3
        if self.latent_memory_features:
            read_dim += int(hidden_dim)
        if self.world_belief_features:
            read_dim += 3
        self.read_head = nn.Sequential(
            nn.Linear(read_dim, int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        self.steering_head = nn.Sequential(
            nn.Linear(read_dim, int(hidden_dim) // 2),
            nn.GELU(),
            nn.Linear(int(hidden_dim) // 2, len(STEERING_CLASSES)),
        )

    def forward_sequence(
        self,
        images: torch.Tensor,
        aux: torch.Tensor,
        *,
        motion_delta: torch.Tensor | None = None,
        reset_each_step: bool = False,
        reverse: bool = False,
    ) -> dict[str, torch.Tensor]:
        if reverse:
            order = torch.arange(images.shape[0] - 1, -1, -1, device=images.device)
            ordered_motion = motion_delta[order] if motion_delta is not None else None
            outputs = self.forward_sequence(
                images[order],
                aux[order],
                motion_delta=ordered_motion,
                reset_each_step=reset_each_step,
                reverse=False,
            )
            return {key: value.flip(0) for key, value in outputs.items()}
        if self.freeze_encoder:
            with torch.no_grad():
                encoded = self.encoder(images)
        else:
            encoded = self.encoder(images)
        hidden = self.encoder_projection(encoded)
        raw = self.evidence_head(hidden).reshape(images.shape[0], self.color_count, 3)
        evidence_logits = raw[..., 0]
        evidence_vec = torch.tanh(raw[..., 1:3])
        rgb_logits: torch.Tensor | None = None
        rgb_vec: torch.Tensor | None = None
        if self.rgb_color_evidence:
            rgb_logits, rgb_vec = self._rgb_color_readout(images)
            if self.rgb_evidence_replaces_learned:
                evidence_logits = float(self.rgb_evidence_logit_scale) * rgb_logits
                if self.rgb_evidence_replaces_learned_logits_only:
                    evidence_vec = torch.tanh(
                        raw[..., 1:3] + float(self.rgb_vector_scale) * rgb_vec
                    )
                elif self.rgb_vector_calibrated:
                    # Already a metric range_scale-normalized position; do not
                    # squash it through tanh (that would destroy the range term).
                    evidence_vec = rgb_vec
                else:
                    evidence_vec = torch.tanh(float(self.rgb_vector_scale) * rgb_vec)
            else:
                evidence_logits = (
                    evidence_logits + float(self.rgb_evidence_logit_scale) * rgb_logits
                )
                evidence_vec = torch.tanh(
                    raw[..., 1:3] + float(self.rgb_vector_scale) * rgb_vec
                )

        memory_vec = torch.zeros(
            self.color_count,
            2,
            device=images.device,
            dtype=hidden.dtype,
        )
        memory_conf = torch.zeros(self.color_count, device=images.device, dtype=hidden.dtype)
        memory_latent = torch.zeros(
            self.color_count,
            hidden.shape[-1],
            device=images.device,
            dtype=hidden.dtype,
        )
        memory_vecs = []
        memory_confs = []
        belief_logits = []
        belief_vecs = []
        read_logits = []
        steering_logits = []
        recurrent_state = torch.zeros(hidden.shape[-1], device=images.device, dtype=hidden.dtype)
        color_index = torch.arange(self.color_count, device=images.device)
        color_emb = self.color_embedding(color_index)
        for idx in range(images.shape[0]):
            if reset_each_step:
                memory_vec = torch.zeros_like(memory_vec)
                memory_conf = torch.zeros_like(memory_conf)
                memory_latent = torch.zeros_like(memory_latent)
                recurrent_state = torch.zeros_like(recurrent_state)
            elif idx > 0:
                if motion_delta is None:
                    delta = self._motion_delta(aux[idx])
                else:
                    delta = motion_delta[idx].to(device=images.device, dtype=hidden.dtype)
                memory_vec = _propagate_vectors(memory_vec, delta)
            recurrent_state = self.recurrent_memory(
                torch.cat([hidden[idx], aux[idx]], dim=-1),
                recurrent_state,
            )
            write = torch.sigmoid(
                (evidence_logits[idx] - float(self.evidence_write_logit_bias))
                / float(self.evidence_write_temperature)
            )
            propagated_weight = memory_conf * (1.0 - write)
            new_conf = 1.0 - (1.0 - memory_conf) * (1.0 - write)
            numerator = (
                propagated_weight.unsqueeze(-1) * memory_vec
                + write.unsqueeze(-1) * evidence_vec[idx]
            )
            memory_vec = numerator / new_conf.clamp_min(1e-4).unsqueeze(-1)
            latent_numerator = (
                propagated_weight.unsqueeze(-1) * memory_latent
                + write.unsqueeze(-1) * hidden[idx].unsqueeze(0)
            )
            memory_latent = latent_numerator / new_conf.clamp_min(1e-4).unsqueeze(-1)
            memory_conf = new_conf.clamp(0.0, 1.0)
            recurrent_features = recurrent_state.unsqueeze(0).expand(self.color_count, -1)
            current_belief_logits = None
            current_belief_vec = None
            if self.world_belief_features:
                belief_raw = self.world_belief_head(
                    torch.cat([color_emb, recurrent_features], dim=-1)
                )
                current_belief_logits = belief_raw[:, 0]
                current_belief_vec = torch.tanh(belief_raw[:, 1:3])
            feature_parts = [
                memory_vec,
                memory_conf.unsqueeze(-1),
                color_emb,
                recurrent_features,
            ]
            if self.latent_memory_features:
                feature_parts.append(memory_latent)
            if self.world_belief_features:
                feature_parts.extend(
                    [
                        current_belief_vec,
                        current_belief_logits.unsqueeze(-1),
                    ]
                )
            features = torch.cat(feature_parts, dim=-1)
            memory_vecs.append(memory_vec)
            memory_confs.append(memory_conf)
            if self.world_belief_features:
                belief_logits.append(current_belief_logits)
                belief_vecs.append(current_belief_vec)
            confidence_logit = torch.logit(memory_conf.clamp(1e-4, 1.0 - 1e-4))
            read_logits.append(
                float(self.read_head_scale) * self.read_head(features).squeeze(-1)
                + float(self.read_confidence_prior_scale) * confidence_logit
            )
            steering_logits.append(self.steering_head(features))
        result = {
            "evidence_logits": evidence_logits,
            "evidence_vec": evidence_vec,
            "memory_vec": torch.stack(memory_vecs),
            "memory_conf": torch.stack(memory_confs),
            "read_logits": torch.stack(read_logits),
            "steering_logits": torch.stack(steering_logits),
        }
        if self.world_belief_features:
            result["belief_logits"] = torch.stack(belief_logits)
            result["belief_vec"] = torch.stack(belief_vecs)
        if rgb_logits is not None and rgb_vec is not None:
            result["rgb_evidence_logits"] = rgb_logits
            result["rgb_evidence_vec"] = rgb_vec
        return result

    def _motion_delta(self, aux: torch.Tensor) -> torch.Tensor:
        # Bound single-step motion to keep early training stable. Units are the
        # normalized vector space used by target_vec.
        raw = self.motion_head(aux)
        dx_dy = 0.35 * torch.tanh(raw[:2])
        dyaw = 1.25 * torch.tanh(raw[2])
        return torch.cat([dx_dy, dyaw.reshape(1)])

    def _rgb_color_readout(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        colors = self.color_rgb.to(device=images.device, dtype=images.dtype)
        distance = ((images[:, None] - colors[None, :, :, None, None]) ** 2).mean(dim=2)
        similarity = torch.exp(
            -distance / (2.0 * float(self.rgb_evidence_sigma) ** 2)
        )
        soft_mask = torch.sigmoid(
            (similarity - float(self.rgb_evidence_threshold))
            / float(self.rgb_evidence_temperature)
        )
        area = soft_mask.mean(dim=(-1, -2)).clamp_min(1e-8)
        area_logits = torch.log(area) - math.log(float(self.rgb_evidence_area_threshold))

        _, _, height, width = soft_mask.shape
        x_coords = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
        x_centroid = (
            soft_mask * x_coords.reshape(1, 1, 1, width)
        ).sum(dim=(-1, -2)) / soft_mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        if self.rgb_vector_calibrated:
            # Calibrated metric write: centroid -> body bearing, area -> range.
            bearing = self.rgb_vector_bearing_a * x_centroid + self.rgb_vector_bearing_b
            range_m = torch.exp(
                self.rgb_vector_range_loglog_m * torch.log(area)
                + self.rgb_vector_range_loglog_c
            ).clamp(0.0, self.range_scale_m)
            r = range_m / self.range_scale_m
            vec = torch.stack([r * torch.cos(bearing), r * torch.sin(bearing)], dim=-1)
            return area_logits, vec
        forward = torch.full_like(x_centroid, 0.75)
        lateral_left = -x_centroid.clamp(-1.0, 1.0)
        return area_logits, torch.stack([forward, lateral_left], dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--validation-datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--frozen-jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--init-controller-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--freeze-except-read-head",
        action="store_true",
        help="After optional init checkpoint load, train only the query read/gate head.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--color-embedding-dim", type=int, default=16)
    parser.add_argument(
        "--rgb-color-evidence",
        action="store_true",
        help="Add fixed RGB color-mask evidence to the learned JEPA evidence head.",
    )
    parser.add_argument(
        "--rgb-evidence-replaces-learned",
        action="store_true",
        help="Use RGB color-mask evidence directly instead of adding learned evidence logits.",
    )
    parser.add_argument("--rgb-evidence-sigma", type=float, default=0.20)
    parser.add_argument("--rgb-evidence-threshold", type=float, default=0.55)
    parser.add_argument("--rgb-evidence-temperature", type=float, default=0.08)
    parser.add_argument("--rgb-evidence-area-threshold", type=float, default=0.006)
    parser.add_argument("--rgb-evidence-logit-scale", type=float, default=1.0)
    parser.add_argument(
        "--rgb-evidence-replaces-learned-logits-only",
        action="store_true",
        help=(
            "When replacing learned RGB evidence, replace only write/read logits; "
            "keep the JEPA learned vector head, optionally biased by --rgb-vector-scale."
        ),
    )
    parser.add_argument("--rgb-vector-scale", type=float, default=1.0)
    parser.add_argument(
        "--rgb-vector-calibrated",
        action="store_true",
        help=(
            "Write a calibrated metric position [r*cos(bearing), r*sin(bearing)] from "
            "the color-mask centroid (bearing) and area (range) instead of the fixed-range "
            "[0.75, -x_centroid] proxy. Requires --rgb-evidence-replaces-learned (not "
            "logits-only). Fit coefficients with audit_go2_rgb_bearing_range_calibration.py."
        ),
    )
    parser.add_argument("--rgb-vector-bearing-a", type=float, default=-0.7412162764485124)
    parser.add_argument("--rgb-vector-bearing-b", type=float, default=0.01266205992118909)
    parser.add_argument("--rgb-vector-range-loglog-m", type=float, default=-0.25799125815180496)
    parser.add_argument("--rgb-vector-range-loglog-c", type=float, default=-0.7229343594424763)
    parser.add_argument("--evidence-write-logit-bias", type=float, default=0.0)
    parser.add_argument("--evidence-write-temperature", type=float, default=1.0)
    parser.add_argument("--read-head-scale", type=float, default=1.0)
    parser.add_argument("--read-confidence-prior-scale", type=float, default=0.0)
    parser.add_argument(
        "--latent-memory-features",
        action="store_true",
        help="Store JEPA hidden features in per-color memory and expose them to read/steering heads.",
    )
    parser.add_argument(
        "--world-belief-features",
        action="store_true",
        help=(
            "Predict per-color relative landmark vectors from RGB/JEPA recurrent "
            "history and expose them to read/steering heads."
        ),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--range-scale-m", type=float, default=6.0)
    parser.add_argument(
        "--motion-propagation",
        choices=("learned", "direct_block", "direct_window", "direct_exact"),
        default="learned",
        help=(
            "Use learned aux->delta propagation or direct odometry fields. "
            "direct_exact uses the ground-truth per-frame egomotion "
            "(exact_body_motion) added by add_exact_odometry_to_go2_dataset.py."
        ),
    )
    parser.add_argument(
        "--motion-translation-scale-m",
        type=float,
        default=None,
        help=(
            "Scale dx/dy into normalized target-vector units for direct odometry. "
            "Defaults to --range-scale-m."
        ),
    )
    parser.add_argument("--visible-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--rgb-supervision-from-evidence",
        action="store_true",
        help=(
            "For visible/vector/memory supervision, use the fixed RGB evidence mask "
            "instead of geometry-visible labels."
        ),
    )
    parser.add_argument(
        "--visible-pos-weight-scale",
        type=float,
        default=1.0,
        help="Multiplier on the auto class-balance positive weight for visible-color BCE.",
    )
    parser.add_argument(
        "--write-visible-loss-weight",
        type=float,
        default=0.0,
        help="Extra BCE on the biased/tempered write logits used by the memory ledger.",
    )
    parser.add_argument("--evidence-vector-loss-weight", type=float, default=2.0)
    parser.add_argument("--motion-supervision-loss-weight", type=float, default=0.0)
    parser.add_argument("--memory-state-loss-weight", type=float, default=1.0)
    parser.add_argument("--memory-state-positive-loss-scale", type=float, default=1.0)
    parser.add_argument("--memory-state-negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--memory-vector-loss-weight", type=float, default=0.0)
    parser.add_argument("--belief-vector-loss-weight", type=float, default=0.0)
    parser.add_argument("--belief-query-vector-loss-weight", type=float, default=0.0)
    parser.add_argument("--query-seen-loss-weight", type=float, default=2.0)
    parser.add_argument("--query-positive-loss-scale", type=float, default=1.0)
    parser.add_argument("--query-negative-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-group-balanced-query-loss-weight", type=float, default=0.0)
    parser.add_argument("--query-vector-loss-weight", type=float, default=4.0)
    parser.add_argument("--query-direction-loss-weight", type=float, default=0.0)
    parser.add_argument("--steering-loss-weight", type=float, default=6.0)
    parser.add_argument(
        "--steering-class-balanced-loss",
        action="store_true",
        help="Balance positive-query steering cross entropy by steering class in each epoch batch.",
    )
    parser.add_argument("--hard-pair-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-pair-updates", type=int, default=0)
    parser.add_argument("--hard-pair-margin", type=float, default=1.0)
    parser.add_argument(
        "--steering-source",
        choices=("head", "vector", "vector_flip", "belief_vector", "belief_vector_flip"),
        default="head",
        help="Use learned head logits or deterministic steering from the learned memory vector.",
    )
    parser.add_argument(
        "--spatial-jepa-readout",
        action="store_true",
        help="Use JEPA conv features plus coordinate channels instead of global pooled latent.",
    )
    parser.add_argument("--spatial-output-dim", type=int, default=256)
    parser.add_argument("--spatial-feature-stride", type=int, choices=(8, 16), default=16)
    parser.add_argument("--finetune-jepa-encoder", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260720)
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

    color_vocab = _color_vocab(train_rows, validation_rows)
    color_index = {color: idx for idx, color in enumerate(color_vocab)}
    motion_translation_scale_m = (
        float(args.motion_translation_scale_m)
        if args.motion_translation_scale_m is not None
        else float(args.range_scale_m)
    )
    train_sequences = _build_sequences(
        train_rows,
        color_index=color_index,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
        motion_translation_scale_m=motion_translation_scale_m,
    )
    validation_sequences = _build_sequences(
        validation_rows,
        color_index=color_index,
        image_size=int(args.image_size),
        range_scale_m=float(args.range_scale_m),
        motion_translation_scale_m=motion_translation_scale_m,
    )
    if not train_sequences:
        raise SystemExit("no train sequences")
    if not validation_sequences:
        raise SystemExit("no validation sequences")

    aux_stats = _aux_stats(train_sequences)
    _normalize_aux(train_sequences, aux_stats)
    _normalize_aux(validation_sequences, aux_stats)
    visible_counts = _visible_counts(train_sequences)
    query_counts = _query_counts(train_sequences)
    hard_pair_groups = _hard_query_groups(train_sequences)
    hard_group_label_counts = _hard_group_label_counts(train_sequences)

    device = _resolve_device(str(args.device))
    base_encoder, jepa_checkpoint = load_go2_jepa_encoder(
        args.frozen_jepa_checkpoint,
        device=device,
        freeze=False if bool(args.spatial_jepa_readout) else not bool(args.finetune_jepa_encoder),
    )
    if bool(args.spatial_jepa_readout):
        encoder = SpatialGo2JepaFeatureEncoder(
            base_encoder,
            image_size=int(args.image_size),
            output_dim=int(args.spatial_output_dim),
            feature_stride=int(args.spatial_feature_stride),
        ).to(device)
        encoder_output_dim = int(args.spatial_output_dim)
    else:
        encoder = base_encoder
        encoder_output_dim = int(jepa_checkpoint.get("latent_dim", args.hidden_dim))
    model = ColorVectorMemoryController(
        encoder=encoder,
        encoder_output_dim=encoder_output_dim,
        color_count=len(color_vocab),
        aux_dim=next(iter(train_sequences.values()))[0].aux.numel(),
        hidden_dim=int(args.hidden_dim),
        color_embedding_dim=int(args.color_embedding_dim),
        freeze_encoder=not bool(args.finetune_jepa_encoder),
        color_rgb=torch.tensor([_COLOR_RGB[color] for color in color_vocab], dtype=torch.float32),
        rgb_color_evidence=bool(args.rgb_color_evidence),
        rgb_evidence_replaces_learned=bool(args.rgb_evidence_replaces_learned),
        rgb_evidence_sigma=float(args.rgb_evidence_sigma),
        rgb_evidence_threshold=float(args.rgb_evidence_threshold),
        rgb_evidence_temperature=float(args.rgb_evidence_temperature),
        rgb_evidence_area_threshold=float(args.rgb_evidence_area_threshold),
        rgb_evidence_logit_scale=float(args.rgb_evidence_logit_scale),
        rgb_evidence_replaces_learned_logits_only=bool(
            args.rgb_evidence_replaces_learned_logits_only
        ),
        rgb_vector_scale=float(args.rgb_vector_scale),
        rgb_vector_calibrated=bool(args.rgb_vector_calibrated),
        rgb_vector_bearing_a=float(args.rgb_vector_bearing_a),
        rgb_vector_bearing_b=float(args.rgb_vector_bearing_b),
        rgb_vector_range_loglog_m=float(args.rgb_vector_range_loglog_m),
        rgb_vector_range_loglog_c=float(args.rgb_vector_range_loglog_c),
        range_scale_m=float(args.range_scale_m),
        evidence_write_logit_bias=float(args.evidence_write_logit_bias),
        evidence_write_temperature=float(args.evidence_write_temperature),
        read_head_scale=float(args.read_head_scale),
        read_confidence_prior_scale=float(args.read_confidence_prior_scale),
        latent_memory_features=bool(args.latent_memory_features),
        world_belief_features=bool(args.world_belief_features),
    ).to(device)
    if args.init_controller_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_controller_checkpoint,
            map_location=device,
            weights_only=False,
        )
        missing, unexpected = model.load_state_dict(
            init_checkpoint["model_state_dict"],
            strict=False,
        )
        if missing or unexpected:
            print(
                "init_controller_checkpoint_load:"
                f" missing={list(missing)} unexpected={list(unexpected)}",
                flush=True,
            )
    if bool(args.freeze_except_read_head):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.read_head.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=1e-4,
    )
    best_score = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        loss = _train_epoch(
            model,
            optimizer,
            train_sequences,
            device=device,
            visible_loss_weight=float(args.visible_loss_weight),
            rgb_supervision_from_evidence=bool(args.rgb_supervision_from_evidence),
            evidence_vector_loss_weight=float(args.evidence_vector_loss_weight),
            motion_supervision_loss_weight=float(args.motion_supervision_loss_weight),
            memory_state_loss_weight=float(args.memory_state_loss_weight),
            memory_vector_loss_weight=float(args.memory_vector_loss_weight),
            belief_vector_loss_weight=float(args.belief_vector_loss_weight),
            query_seen_loss_weight=float(args.query_seen_loss_weight),
            query_positive_loss_scale=float(args.query_positive_loss_scale),
            query_negative_loss_weight=float(args.query_negative_loss_weight),
            hard_group_label_counts=hard_group_label_counts,
            hard_group_balanced_query_loss_weight=float(
                args.hard_group_balanced_query_loss_weight
            ),
            query_vector_loss_weight=float(args.query_vector_loss_weight),
            belief_query_vector_loss_weight=float(args.belief_query_vector_loss_weight),
            query_direction_loss_weight=float(args.query_direction_loss_weight),
            steering_loss_weight=float(args.steering_loss_weight),
            steering_class_balanced_loss=bool(args.steering_class_balanced_loss),
            hard_pair_groups=hard_pair_groups,
            hard_pair_loss_weight=float(args.hard_pair_loss_weight),
            hard_pair_updates=int(args.hard_pair_updates),
            hard_pair_margin=float(args.hard_pair_margin),
            motion_propagation=str(args.motion_propagation),
            visible_pos_weight=_pos_weight(visible_counts),
            visible_pos_weight_scale=float(args.visible_pos_weight_scale),
            write_visible_loss_weight=float(args.write_visible_loss_weight),
            query_pos_weight=_pos_weight(query_counts),
            memory_state_positive_loss_scale=float(args.memory_state_positive_loss_scale),
            memory_state_negative_loss_weight=float(args.memory_state_negative_loss_weight),
        )
        normal = _evaluate(
            model,
            validation_sequences,
            device=device,
            threshold=float(args.threshold),
            ablation="normal",
            motion_propagation=str(args.motion_propagation),
            steering_source=str(args.steering_source),
        )
        score = _selection_score(normal)
        history.append({"epoch": int(epoch), "train_loss": float(loss), "validation": normal})
        if score >= best_score:
            best_score = float(score)
            best_metrics = normal
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if int(args.log_every) > 0 and (epoch == 1 or epoch % int(args.log_every) == 0):
            print(
                f"epoch={epoch}"
                f" loss={loss:.4f}"
                f" target_steer={normal['target_steering_pipeline_success']:.3f}"
                f" recall={normal['target_recall']:.3f}"
                f" false_claim={normal['false_claim_rate']:.3f}"
                f" precision={normal['target_selection_precision']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = _evaluate(
        model,
        train_sequences,
        device=device,
        threshold=float(args.threshold),
        ablation="normal",
        motion_propagation=str(args.motion_propagation),
        steering_source=str(args.steering_source),
    )
    threshold_sweep = _threshold_sweep(
        model,
        validation_sequences,
        device=device,
        motion_propagation=str(args.motion_propagation),
        steering_source=str(args.steering_source),
    )
    best_threshold_key, best_threshold_value = max(
        threshold_sweep.items(),
        key=lambda item: (
            item[1]["normal"]["target_steering_pipeline_success"],
            -item[1]["normal"]["false_claim_rate"],
            item[1]["normal_minus_best_corrupted_target_steering_pipeline_success"],
        ),
    )
    validation_ablations = best_threshold_value["ablations"]
    normal = validation_ablations["normal"]
    gap = float(best_threshold_value["normal_minus_best_corrupted_target_steering_pipeline_success"])
    gate_pass = (
        float(normal["target_steering_pipeline_success"])
        >= float(args.min_target_steering_success)
        and float(normal["false_claim_rate"]) <= float(args.max_false_claim_rate)
        and gap >= float(args.min_corrupted_gap)
    )

    checkpoint = {
        "schema": "lewm_go2_rgb_jepa_vector_memory_controller_checkpoint_v0",
        "model_state_dict": model.state_dict(),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "frozen_jepa_report": {
            "schema": str(jepa_checkpoint.get("schema", "")),
            "latent_dim": int(jepa_checkpoint.get("latent_dim", args.hidden_dim)),
            "image_size": int(jepa_checkpoint.get("image_size", args.image_size)),
        },
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "spatial_jepa_readout": bool(args.spatial_jepa_readout),
        "spatial_output_dim": int(args.spatial_output_dim),
        "spatial_feature_stride": int(args.spatial_feature_stride),
        "color_vocab": color_vocab,
        "aux_mean": aux_stats["mean"].tolist(),
        "aux_std": aux_stats["std"].tolist(),
        "range_scale_m": float(args.range_scale_m),
        "motion_propagation": str(args.motion_propagation),
        "motion_translation_scale_m": float(motion_translation_scale_m),
        "steering_source": str(args.steering_source),
        "steering_class_balanced_loss": bool(args.steering_class_balanced_loss),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "color_embedding_dim": int(args.color_embedding_dim),
        "rgb_color_evidence": bool(args.rgb_color_evidence),
        "rgb_evidence_replaces_learned": bool(args.rgb_evidence_replaces_learned),
        "rgb_evidence_sigma": float(args.rgb_evidence_sigma),
        "rgb_evidence_threshold": float(args.rgb_evidence_threshold),
        "rgb_evidence_temperature": float(args.rgb_evidence_temperature),
        "rgb_evidence_area_threshold": float(args.rgb_evidence_area_threshold),
        "rgb_evidence_logit_scale": float(args.rgb_evidence_logit_scale),
        "rgb_evidence_replaces_learned_logits_only": bool(
            args.rgb_evidence_replaces_learned_logits_only
        ),
        "rgb_supervision_from_evidence": bool(args.rgb_supervision_from_evidence),
        "rgb_vector_scale": float(args.rgb_vector_scale),
        "rgb_vector_calibrated": bool(args.rgb_vector_calibrated),
        "evidence_write_logit_bias": float(args.evidence_write_logit_bias),
        "evidence_write_temperature": float(args.evidence_write_temperature),
        "read_head_scale": float(args.read_head_scale),
        "read_confidence_prior_scale": float(args.read_confidence_prior_scale),
        "latent_memory_features": bool(args.latent_memory_features),
        "world_belief_features": bool(args.world_belief_features),
        "steering_classes": list(STEERING_CLASSES),
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    report = {
        "schema": "lewm_go2_rgb_jepa_vector_memory_controller_report_v0",
        "datasets": [str(path) for path in args.datasets],
        "validation_datasets": [str(path) for path in args.validation_datasets],
        "output": str(args.output),
        "device": str(device),
        "frozen_jepa_checkpoint": str(args.frozen_jepa_checkpoint),
        "finetuned_jepa_encoder": bool(args.finetune_jepa_encoder),
        "spatial_jepa_readout": bool(args.spatial_jepa_readout),
        "spatial_output_dim": int(args.spatial_output_dim),
        "spatial_feature_stride": int(args.spatial_feature_stride),
        "rgb_color_evidence": bool(args.rgb_color_evidence),
        "rgb_evidence_replaces_learned": bool(args.rgb_evidence_replaces_learned),
        "rgb_evidence_sigma": float(args.rgb_evidence_sigma),
        "rgb_evidence_threshold": float(args.rgb_evidence_threshold),
        "rgb_evidence_temperature": float(args.rgb_evidence_temperature),
        "rgb_evidence_area_threshold": float(args.rgb_evidence_area_threshold),
        "rgb_evidence_logit_scale": float(args.rgb_evidence_logit_scale),
        "rgb_evidence_replaces_learned_logits_only": bool(
            args.rgb_evidence_replaces_learned_logits_only
        ),
        "rgb_supervision_from_evidence": bool(args.rgb_supervision_from_evidence),
        "rgb_vector_scale": float(args.rgb_vector_scale),
        "latent_memory_features": bool(args.latent_memory_features),
        "world_belief_features": bool(args.world_belief_features),
        "color_vocab": color_vocab,
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_row_count": sum(len(sequence) for sequence in train_sequences.values()),
        "validation_row_count": sum(len(sequence) for sequence in validation_sequences.values()),
        "visible_label_counts": visible_counts,
        "query_label_counts": query_counts,
        "hard_pair_group_count": len(hard_pair_groups),
        "hard_pair_example_count": sum(
            len(bucket["positive"]) + len(bucket["negative"])
            for bucket in hard_pair_groups.values()
        ),
        "hard_group_balanced_count": len(hard_group_label_counts),
        "final_train": final_train,
        "validation_ablations": validation_ablations,
        "threshold_sweep": threshold_sweep,
        "best_threshold_by_target_steering": best_threshold_key,
        "steering_diagnostics": _steering_diagnostics(
            model,
            validation_sequences,
            device=device,
            threshold=float(best_threshold_key),
            ablation="normal",
            motion_propagation=str(args.motion_propagation),
        ),
        "normal_minus_best_corrupted_target_steering_pipeline_success": gap,
        "controller_gate_pass": bool(gate_pass),
        "best_validation_selection_score": float(best_score),
        "best_validation_selected_metrics": best_metrics or {},
        "history": history,
        "claim_boundary": (
            "Pure RGB/JEPA vector-memory controller. Inference uses RGB encoded "
            "by the JEPA visual encoder, odometry/action history, learned color "
            "evidence heads, learned vector memory, and a target color query. "
            "Runtime landmark ids, object slots, detector visibility, range, "
            "bearing, and map/geodesic geometry are not inputs."
        ),
    }
    report_path = args.report_output or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "go2_rgb_jepa_vector_memory_controller:"
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
    motion_translation_scale_m: float,
) -> dict[tuple[str, int, int], list[Frame]]:
    color_count = len(color_index)
    sequences_raw: dict[tuple[str, int, int], list[tuple[dict[str, Any], torch.Tensor]]] = (
        defaultdict(list)
    )
    for row in rows:
        sequences_raw[_seq_key(row)].append(
            (row, _load_image(Path(row["rgb_path"]), image_size=image_size))
        )
    sequences: dict[tuple[str, int, int], list[Frame]] = {}
    for seq_key, items in sequences_raw.items():
        items.sort(key=lambda item: int(item[0].get("episode_step", 0)))
        seen = np.zeros(color_count, dtype=np.float32)
        sequence = []
        for row, image in items:
            visible_mask = np.zeros(color_count, dtype=np.float32)
            visible_vec = np.zeros((color_count, 2), dtype=np.float32)
            all_vec = np.zeros((color_count, 2), dtype=np.float32)
            landmark_by_id = _landmark_by_id(row)
            for landmark in row.get("landmarks", ()):
                color = _object_color(str(landmark.get("object_id", "")))
                if color not in color_index:
                    continue
                idx = color_index[color]
                vector = _vector_target(landmark, range_scale_m=range_scale_m)
                all_vec[idx] = vector
                if bool(landmark.get("visible", False)):
                    visible_mask[idx] = 1.0
                    visible_vec[idx] = vector
            seen = np.maximum(seen, visible_mask)
            queries = []
            query_seen: set[tuple[int, float]] = set()
            for event in row.get("go2_causal_memory_pair_selection", ()):
                role = str(event.get("pair_role", ""))
                if not role.startswith("current_"):
                    continue
                object_id = str(event.get("object_id", ""))
                landmark = landmark_by_id.get(object_id)
                color = _object_color(object_id)
                if landmark is None or color not in color_index:
                    continue
                target = 1.0 if bool(event.get("seen_before", False)) else 0.0
                color_idx = int(color_index[color])
                key = (color_idx, target)
                if key in query_seen:
                    continue
                query_seen.add(key)
                bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
                queries.append(
                    Query(
                        color_index=color_idx,
                        target=target,
                        target_vec=torch.tensor(
                            _vector_target(landmark, range_scale_m=range_scale_m),
                            dtype=torch.float32,
                        ),
                        target_steering=_steering_index(bearing),
                        group_key=(
                            str(row.get("scene_id", "")),
                            int(row.get("cell_id", -1)),
                            int(row.get("yaw_bin", -1)),
                            color,
                        ),
                    )
                )
            sequence.append(
                Frame(
                    seq_key=seq_key,
                    episode_step=int(row.get("episode_step", 0)),
                    image=image,
                    aux=torch.tensor(_aux_features(row), dtype=torch.float32),
                    motion_block=torch.tensor(
                        _motion_delta_from_row(
                            row,
                            field="integrated_body_motion_block",
                            translation_scale_m=motion_translation_scale_m,
                        ),
                        dtype=torch.float32,
                    ),
                    motion_window=torch.tensor(
                        _motion_delta_from_row(
                            row,
                            field="integrated_body_motion_window",
                            translation_scale_m=motion_translation_scale_m,
                        ),
                        dtype=torch.float32,
                    ),
                    exact_motion=torch.tensor(
                        _motion_delta_from_row(
                            row,
                            field="exact_body_motion",
                            translation_scale_m=motion_translation_scale_m,
                        ),
                        dtype=torch.float32,
                    ),
                    visible_mask=torch.tensor(visible_mask, dtype=torch.float32),
                    visible_vec=torch.tensor(visible_vec, dtype=torch.float32),
                    all_vec=torch.tensor(all_vec, dtype=torch.float32),
                    memory_mask=torch.tensor(seen.copy(), dtype=torch.float32),
                    queries=tuple(queries),
                )
            )
        sequences[seq_key] = sequence
    return dict(sequences)


def _train_epoch(
    model: ColorVectorMemoryController,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    visible_loss_weight: float,
    rgb_supervision_from_evidence: bool,
    evidence_vector_loss_weight: float,
    motion_supervision_loss_weight: float,
    memory_state_loss_weight: float,
    memory_vector_loss_weight: float,
    belief_vector_loss_weight: float,
    query_seen_loss_weight: float,
    query_positive_loss_scale: float,
    query_negative_loss_weight: float,
    hard_group_label_counts: dict[tuple[str, int, int, str], dict[str, int]],
    hard_group_balanced_query_loss_weight: float,
    query_vector_loss_weight: float,
    belief_query_vector_loss_weight: float,
    query_direction_loss_weight: float,
    steering_loss_weight: float,
    steering_class_balanced_loss: bool,
    hard_pair_groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]],
    hard_pair_loss_weight: float,
    hard_pair_updates: int,
    hard_pair_margin: float,
    motion_propagation: str,
    visible_pos_weight: float,
    visible_pos_weight_scale: float,
    write_visible_loss_weight: float,
    query_pos_weight: float,
    memory_state_positive_loss_scale: float,
    memory_state_negative_loss_weight: float,
) -> float:
    model.train()
    keys = list(sequences)
    random.shuffle(keys)
    total = 0.0
    trained = 0
    for key in keys:
        sequence = sequences[key]
        batch = _sequence_tensors(sequence, device=device)
        motion_delta = _select_motion_delta(batch, motion_propagation=motion_propagation)
        outputs = model.forward_sequence(
            batch["images"],
            batch["aux"],
            motion_delta=motion_delta,
        )
        supervision_visible_mask = batch["visible_mask"]
        supervision_visible_vec = batch["visible_vec"]
        supervision_memory_mask = batch["memory_mask"]
        if bool(rgb_supervision_from_evidence):
            if "rgb_evidence_logits" not in outputs:
                raise ValueError("--rgb-supervision-from-evidence requires --rgb-color-evidence")
            supervision_visible_mask = (
                outputs["rgb_evidence_logits"].detach() > 0.0
            ).to(dtype=batch["visible_mask"].dtype)
            supervision_visible_vec = (
                batch["all_vec"] * supervision_visible_mask.unsqueeze(-1)
            )
            supervision_memory_mask = torch.cummax(supervision_visible_mask, dim=0).values
        losses = []
        visible_pos = torch.full(
            (model.color_count,),
            float(visible_pos_weight) * float(visible_pos_weight_scale),
            dtype=supervision_visible_mask.dtype,
            device=device,
        )
        losses.append(
            F.binary_cross_entropy_with_logits(
                outputs["evidence_logits"],
                supervision_visible_mask,
                pos_weight=visible_pos,
            )
            * float(visible_loss_weight)
        )
        if float(write_visible_loss_weight) > 0.0:
            write_logits = (
                outputs["evidence_logits"] - float(model.evidence_write_logit_bias)
            ) / float(model.evidence_write_temperature)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    write_logits,
                    supervision_visible_mask,
                    pos_weight=visible_pos,
                )
                * float(write_visible_loss_weight)
            )
        losses.append(
            _masked_vector_loss(
                outputs["evidence_vec"],
                supervision_visible_vec,
                supervision_visible_mask,
            )
            * float(evidence_vector_loss_weight)
        )
        if float(motion_supervision_loss_weight) > 0.0 and motion_delta is None:
            losses.append(
                _motion_supervision_loss(model, batch["aux"], batch["all_vec"])
                * float(motion_supervision_loss_weight)
            )
        losses.append(
            _weighted_binary_loss(
                outputs["read_logits"],
                supervision_memory_mask,
                positive_scale=float(memory_state_positive_loss_scale),
                negative_weight=float(memory_state_negative_loss_weight),
            )
            * float(memory_state_loss_weight)
        )
        if float(memory_vector_loss_weight) > 0.0:
            losses.append(
                _masked_vector_loss(
                    outputs["memory_vec"],
                    batch["all_vec"],
                    supervision_memory_mask,
                )
                * float(memory_vector_loss_weight)
            )
        if float(belief_vector_loss_weight) > 0.0:
            if "belief_vec" not in outputs:
                raise ValueError("--belief-vector-loss-weight requires --world-belief-features")
            losses.append(
                F.smooth_l1_loss(outputs["belief_vec"], batch["all_vec"])
                * float(belief_vector_loss_weight)
            )
        query_loss = _query_losses(
            sequence,
            outputs,
            device=device,
            query_pos_weight=float(query_pos_weight),
            seen_weight=float(query_seen_loss_weight),
            positive_loss_scale=float(query_positive_loss_scale),
            negative_loss_weight=float(query_negative_loss_weight),
            hard_group_label_counts=hard_group_label_counts,
            hard_group_balanced_weight=float(hard_group_balanced_query_loss_weight),
            vector_weight=float(query_vector_loss_weight),
            belief_vector_weight=float(belief_query_vector_loss_weight),
            direction_weight=float(query_direction_loss_weight),
            steering_weight=float(steering_loss_weight),
            steering_class_balanced_loss=bool(steering_class_balanced_loss),
        )
        if query_loss is not None:
            losses.append(query_loss)
        loss = torch.stack(losses).sum()
        if not loss.requires_grad:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        trained += 1
    if float(hard_pair_loss_weight) > 0.0 and int(hard_pair_updates) > 0:
        hard_loss = _train_hard_pair_updates(
            model,
            optimizer,
            sequences,
            hard_pair_groups,
            device=device,
            updates=int(hard_pair_updates),
            loss_weight=float(hard_pair_loss_weight),
            margin=float(hard_pair_margin),
            motion_propagation=motion_propagation,
        )
        total += hard_loss * max(1, trained)
    return total / max(1, trained)


def _train_hard_pair_updates(
    model: ColorVectorMemoryController,
    optimizer: torch.optim.Optimizer,
    sequences: dict[tuple[str, int, int], list[Frame]],
    hard_pair_groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]],
    *,
    device: torch.device,
    updates: int,
    loss_weight: float,
    margin: float,
    motion_propagation: str,
) -> float:
    valid_groups = [
        bucket
        for bucket in hard_pair_groups.values()
        if bucket["positive"] and bucket["negative"]
    ]
    if not valid_groups:
        return 0.0
    model.train()
    total = 0.0
    completed = 0
    for _ in range(max(0, int(updates))):
        bucket = random.choice(valid_groups)
        positive = random.choice(bucket["positive"])
        negative = random.choice(bucket["negative"])
        if positive.seq_key == negative.seq_key:
            outputs_by_key = {
                positive.seq_key: _forward_train_sequence(
                    model,
                    sequences[positive.seq_key],
                    device=device,
                    motion_propagation=motion_propagation,
                )
            }
        else:
            outputs_by_key = {
                positive.seq_key: _forward_train_sequence(
                    model,
                    sequences[positive.seq_key],
                    device=device,
                    motion_propagation=motion_propagation,
                ),
                negative.seq_key: _forward_train_sequence(
                    model,
                    sequences[negative.seq_key],
                    device=device,
                    motion_propagation=motion_propagation,
                ),
            }
        positive_logit = outputs_by_key[positive.seq_key]["read_logits"][
            positive.step_idx,
            positive.color_index,
        ]
        negative_logit = outputs_by_key[negative.seq_key]["read_logits"][
            negative.step_idx,
            negative.color_index,
        ]
        loss = F.softplus(float(margin) - positive_logit + negative_logit) * float(
            loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu())
        completed += 1
    return total / max(1, completed)


def _forward_train_sequence(
    model: ColorVectorMemoryController,
    sequence: list[Frame],
    *,
    device: torch.device,
    motion_propagation: str,
) -> dict[str, torch.Tensor]:
    batch = _sequence_tensors(sequence, device=device)
    motion_delta = _select_motion_delta(batch, motion_propagation=motion_propagation)
    return model.forward_sequence(
        batch["images"],
        batch["aux"],
        motion_delta=motion_delta,
    )


def _query_losses(
    sequence: list[Frame],
    outputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    query_pos_weight: float,
    seen_weight: float,
    positive_loss_scale: float,
    negative_loss_weight: float,
    hard_group_label_counts: dict[tuple[str, int, int, str], dict[str, int]],
    hard_group_balanced_weight: float,
    vector_weight: float,
    belief_vector_weight: float,
    direction_weight: float,
    steering_weight: float,
    steering_class_balanced_loss: bool,
) -> torch.Tensor | None:
    seen_logits = []
    seen_targets = []
    seen_group_keys = []
    vector_preds = []
    vector_targets = []
    belief_vector_preds = []
    steering_logits = []
    steering_targets = []
    for step_idx, frame in enumerate(sequence):
        for query in frame.queries:
            color_idx = int(query.color_index)
            seen_logits.append(outputs["read_logits"][step_idx, color_idx])
            seen_targets.append(float(query.target))
            seen_group_keys.append(query.group_key)
            if query.target >= 0.5:
                vector_preds.append(outputs["memory_vec"][step_idx, color_idx])
                vector_targets.append(query.target_vec)
                if "belief_vec" in outputs:
                    belief_vector_preds.append(outputs["belief_vec"][step_idx, color_idx])
                steering_logits.append(outputs["steering_logits"][step_idx, color_idx])
                steering_targets.append(int(query.target_steering))
    if not seen_logits:
        return None
    logits = torch.stack(seen_logits)
    targets = torch.tensor(seen_targets, dtype=logits.dtype, device=device)
    raw_seen_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    seen_weights = torch.where(
        targets >= 0.5,
        torch.full_like(targets, float(query_pos_weight) * float(positive_loss_scale)),
        torch.full_like(targets, float(negative_loss_weight)),
    )
    loss = (raw_seen_loss * seen_weights).mean() * float(seen_weight)
    if float(hard_group_balanced_weight) > 0.0:
        balanced_weights = []
        for group_key, target_value in zip(seen_group_keys, seen_targets):
            counts = hard_group_label_counts.get(group_key)
            if not counts:
                balanced_weights.append(0.0)
                continue
            label_key = "positive" if float(target_value) >= 0.5 else "negative"
            balanced_weights.append(1.0 / float(max(1, counts[label_key])))
        balanced = torch.tensor(balanced_weights, dtype=logits.dtype, device=device)
        if float(balanced.sum().detach().cpu()) > 0.0:
            loss = loss + (
                (raw_seen_loss * balanced).sum() / balanced.sum().clamp_min(1e-6)
            ) * float(hard_group_balanced_weight)
    if vector_preds:
        pred_vectors = torch.stack(vector_preds)
        target_vectors = torch.stack(vector_targets).to(device)
        loss = loss + F.smooth_l1_loss(
            pred_vectors,
            target_vectors,
        ) * float(vector_weight)
        if float(belief_vector_weight) > 0.0:
            if not belief_vector_preds:
                raise ValueError(
                    "--belief-query-vector-loss-weight requires --world-belief-features"
                )
            loss = loss + F.smooth_l1_loss(
                torch.stack(belief_vector_preds),
                target_vectors,
            ) * float(belief_vector_weight)
        if float(direction_weight) > 0.0:
            loss = loss + _signed_direction_loss(
                pred_vectors,
                torch.tensor(steering_targets, dtype=torch.long, device=device),
            ) * float(direction_weight)
        steering_targets_tensor = torch.tensor(steering_targets, dtype=torch.long, device=device)
        steering_raw_loss = F.cross_entropy(
            torch.stack(steering_logits),
            steering_targets_tensor,
            reduction="none",
        )
        if bool(steering_class_balanced_loss):
            counts = torch.bincount(
                steering_targets_tensor,
                minlength=len(STEERING_CLASSES),
            ).to(dtype=steering_raw_loss.dtype)
            class_weights = steering_targets_tensor.numel() / (
                float(len(STEERING_CLASSES)) * counts.clamp_min(1.0)
            )
            steering_raw_loss = steering_raw_loss * class_weights[steering_targets_tensor]
        loss = loss + steering_raw_loss.mean() * float(steering_weight)
    return loss


def _evaluate(
    model: ColorVectorMemoryController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    threshold: float,
    ablation: str,
    motion_propagation: str,
    steering_source: str,
) -> dict[str, Any]:
    model.eval()
    outputs_by_key = _outputs_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
        motion_propagation=motion_propagation,
    )
    metrics = _Metrics()
    with torch.no_grad():
        for key, sequence in sequences.items():
            outputs = outputs_by_key[key]
            for step_idx, frame in enumerate(sequence):
                for query in frame.queries:
                    color_idx = int(query.color_index)
                    if ablation == "memory_off_abstain":
                        selected = False
                    else:
                        score = torch.sigmoid(outputs["read_logits"][step_idx, color_idx])
                        selected = float(score.detach().cpu()) >= float(threshold)
                    steering_index = None
                    if selected:
                        steering_index = _select_steering_index(
                            outputs,
                            step_idx=step_idx,
                            color_idx=color_idx,
                            steering_source=steering_source,
                        )
                    metrics.add(query=query, selected=selected, steering_index=steering_index)
    return metrics.to_dict()


def _outputs_by_sequence(
    model: ColorVectorMemoryController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    ablation: str,
    motion_propagation: str,
) -> dict[tuple[str, int, int], dict[str, torch.Tensor]]:
    outputs_by_key = {}
    with torch.no_grad():
        for key, sequence in sequences.items():
            batch = _sequence_tensors(sequence, device=device)
            motion_delta = _select_motion_delta(batch, motion_propagation=motion_propagation)
            if ablation in {"normal", "memory_off_abstain", "shuffle_memory_states"}:
                outputs = model.forward_sequence(
                    batch["images"],
                    batch["aux"],
                    motion_delta=motion_delta,
                )
            elif ablation == "reset_recurrent_state":
                outputs = model.forward_sequence(
                    batch["images"],
                    batch["aux"],
                    motion_delta=motion_delta,
                    reset_each_step=True,
                )
            elif ablation == "reverse_input_history":
                outputs = model.forward_sequence(
                    batch["images"],
                    batch["aux"],
                    motion_delta=motion_delta,
                    reverse=True,
                )
            else:
                raise ValueError(f"unknown ablation: {ablation}")
            outputs_by_key[key] = outputs
    if ablation != "shuffle_memory_states":
        return outputs_by_key
    for name in ("memory_vec", "memory_conf", "read_logits", "steering_logits"):
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
    model: ColorVectorMemoryController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    motion_propagation: str,
    steering_source: str,
) -> dict[str, Any]:
    result = {}
    for threshold in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        ablations = {
            ablation: _evaluate(
                model,
                sequences,
                device=device,
                threshold=float(threshold),
                ablation=ablation,
                motion_propagation=motion_propagation,
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

    def add(self, *, query: Query, selected: bool, steering_index: int | None) -> None:
        if query.target >= 0.5:
            self.positive += 1
            if not selected:
                self.missed_positive += 1
                self.classifications["missed_positive"] += 1
                return
            self.selected += 1
            self.correct_target += 1
            self.classifications["correct_target"] += 1
            pred = _steering_name(int(steering_index if steering_index is not None else 1))
            target = _steering_name(int(query.target_steering))
            self.predicted_steering[pred] += 1
            self.target_steering[target] += 1
            if pred == target:
                self.target_steer += 1
            return
        self.negative += 1
        if selected:
            self.selected += 1
            self.false_claim += 1
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
        }


def _steering_diagnostics(
    model: ColorVectorMemoryController,
    sequences: dict[tuple[str, int, int], list[Frame]],
    *,
    device: torch.device,
    threshold: float,
    ablation: str,
    motion_propagation: str,
) -> dict[str, Any]:
    outputs_by_key = _outputs_by_sequence(
        model,
        sequences,
        device=device,
        ablation=ablation,
        motion_propagation=motion_propagation,
    )
    totals: dict[str, Counter[str]] = {
        "head": Counter(),
        "head_flip": Counter(),
        "vector": Counter(),
        "vector_flip": Counter(),
        "belief_vector": Counter(),
        "belief_vector_flip": Counter(),
    }
    positive = 0
    selected_positive = 0
    # Steering accuracy stratified by how many steps since the queried color was
    # last in the camera cone (its RGB mask fired). Out-of-frame steering must be
    # recovered by propagating that last in-cone write through odometry, so error
    # grows with the gap. gap=0 means in-frame now (trivial).
    gap_buckets = (0, 2, 4, 8, 16, 10**9)
    gap_totals = {
        src: {b: Counter() for b in gap_buckets} for src in ("head", "vector")
    }
    with torch.no_grad():
        for key, sequence in sequences.items():
            outputs = outputs_by_key[key]
            has_rgb = "rgb_evidence_logits" in outputs
            last_fire: dict[int, int] = {}
            for step_idx, frame in enumerate(sequence):
                if has_rgb:
                    fired = outputs["rgb_evidence_logits"][step_idx] > 0
                    for c in range(int(fired.shape[0])):
                        if bool(fired[c]):
                            last_fire[c] = step_idx
                for query in frame.queries:
                    if query.target < 0.5:
                        continue
                    positive += 1
                    color_idx = int(query.color_index)
                    score = torch.sigmoid(outputs["read_logits"][step_idx, color_idx])
                    if float(score.detach().cpu()) < float(threshold):
                        continue
                    selected_positive += 1
                    target = int(query.target_steering)
                    head = int(torch.argmax(outputs["steering_logits"][step_idx, color_idx]).cpu())
                    vector = _vector_steering_index(outputs["memory_vec"][step_idx, color_idx])
                    if has_rgb and color_idx in last_fire:
                        gap = step_idx - last_fire[color_idx]
                        bucket = next(bk for bk in gap_buckets if gap <= bk)
                        for src, pred in (("head", head), ("vector", vector)):
                            gap_totals[src][bucket]["correct"] += int(pred == target)
                            gap_totals[src][bucket]["total"] += 1
                    predictions = [
                        ("head", head),
                        ("head_flip", _flip_steering_index(head)),
                        ("vector", vector),
                        ("vector_flip", _flip_steering_index(vector)),
                    ]
                    if "belief_vec" in outputs:
                        belief_vector = _vector_steering_index(
                            outputs["belief_vec"][step_idx, color_idx]
                        )
                        predictions.extend(
                            [
                                ("belief_vector", belief_vector),
                                ("belief_vector_flip", _flip_steering_index(belief_vector)),
                            ]
                        )
                    for name, pred in predictions:
                        totals[name]["correct"] += int(pred == target)
                        totals[name]["total"] += 1
                        totals[name][f"pred_{_steering_name(pred)}"] += 1
                        totals[name][f"target_{_steering_name(target)}"] += 1
    def _gap_label(b: int) -> str:
        return "gt16" if b >= 10**9 else f"le{b}"

    steering_by_incone_gap = {
        src: {
            _gap_label(b): {
                "acc": (
                    float(c["correct"]) / float(c["total"]) if c["total"] else float("nan")
                ),
                "n": int(c["total"]),
            }
            for b, c in sorted(buckets.items())
        }
        for src, buckets in gap_totals.items()
    }
    return {
        name: {
            "selected_positive_accuracy": (
                float(counter["correct"]) / float(max(1, counter["total"]))
            ),
            "pipeline_success": float(counter["correct"]) / float(max(1, positive)),
            "selected_positive_count": int(counter["total"]),
            "positive_count": int(positive),
            "counts": dict(sorted(counter.items())),
        }
        for name, counter in totals.items()
    } | {
        "selected_positive_count": int(selected_positive),
        "positive_count": int(positive),
        "steering_by_incone_gap": steering_by_incone_gap,
    }


def _sequence_tensors(sequence: list[Frame], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([frame.image for frame in sequence]).to(device),
        "aux": torch.stack([frame.aux for frame in sequence]).to(device),
        "motion_block": torch.stack([frame.motion_block for frame in sequence]).to(device),
        "motion_window": torch.stack([frame.motion_window for frame in sequence]).to(device),
        "exact_motion": torch.stack([frame.exact_motion for frame in sequence]).to(device),
        "visible_mask": torch.stack([frame.visible_mask for frame in sequence]).to(device),
        "visible_vec": torch.stack([frame.visible_vec for frame in sequence]).to(device),
        "all_vec": torch.stack([frame.all_vec for frame in sequence]).to(device),
        "memory_mask": torch.stack([frame.memory_mask for frame in sequence]).to(device),
    }


def _select_motion_delta(
    batch: dict[str, torch.Tensor],
    *,
    motion_propagation: str,
) -> torch.Tensor | None:
    if motion_propagation == "learned":
        return None
    if motion_propagation == "direct_block":
        return batch["motion_block"]
    if motion_propagation == "direct_window":
        return batch["motion_window"]
    if motion_propagation == "direct_exact":
        return batch["exact_motion"]
    raise ValueError(f"unknown motion propagation: {motion_propagation}")


def _masked_vector_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if float(mask.sum().detach().cpu()) <= 0.0:
        return pred.sum() * 0.0
    loss = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def _weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_scale: float,
    negative_weight: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if float(positive_scale) == 1.0 and float(negative_weight) == 1.0:
        return loss.mean()
    weights = torch.where(
        targets >= 0.5,
        torch.full_like(targets, float(positive_scale)),
        torch.full_like(targets, float(negative_weight)),
    )
    return (loss * weights).mean()


def _motion_supervision_loss(
    model: ColorVectorMemoryController,
    aux: torch.Tensor,
    all_vec: torch.Tensor,
) -> torch.Tensor:
    if all_vec.shape[0] <= 1:
        return all_vec.sum() * 0.0
    losses = []
    for idx in range(1, int(all_vec.shape[0])):
        delta = model._motion_delta(aux[idx])
        predicted = _propagate_vectors(all_vec[idx - 1], delta)
        losses.append(F.smooth_l1_loss(predicted, all_vec[idx]))
    return torch.stack(losses).mean() if losses else all_vec.sum() * 0.0


def _propagate_vectors(vectors: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    dx, dy, dyaw = delta[0], delta[1], delta[2]
    shifted = vectors - torch.stack([dx, dy]).reshape(1, 2)
    c = torch.cos(-dyaw)
    s = torch.sin(-dyaw)
    x = c * shifted[:, 0] - s * shifted[:, 1]
    y = s * shifted[:, 0] + c * shifted[:, 1]
    return torch.stack([x, y], dim=-1)


def _vector_steering_index(vector: torch.Tensor) -> int:
    x = float(vector[0].detach().cpu())
    y = float(vector[1].detach().cpu())
    return _steering_index(math.atan2(y, x))


def _select_steering_index(
    outputs: dict[str, torch.Tensor],
    *,
    step_idx: int,
    color_idx: int,
    steering_source: str,
) -> int:
    if steering_source == "head":
        return int(torch.argmax(outputs["steering_logits"][step_idx, color_idx]).detach().cpu())
    vector_index = _vector_steering_index(outputs["memory_vec"][step_idx, color_idx])
    if steering_source == "vector":
        return vector_index
    if steering_source == "vector_flip":
        return _flip_steering_index(vector_index)
    if steering_source in {"belief_vector", "belief_vector_flip"}:
        if "belief_vec" not in outputs:
            raise ValueError("--steering-source belief_vector requires --world-belief-features")
        belief_index = _vector_steering_index(outputs["belief_vec"][step_idx, color_idx])
        if steering_source == "belief_vector":
            return belief_index
        return _flip_steering_index(belief_index)
    raise ValueError(f"unknown steering source: {steering_source}")


def _flip_steering_index(index: int) -> int:
    if int(index) == 0:
        return 2
    if int(index) == 2:
        return 0
    return 1


def _signed_direction_loss(pred_vectors: torch.Tensor, steering_targets: torch.Tensor) -> torch.Tensor:
    signs = torch.zeros_like(steering_targets, dtype=pred_vectors.dtype)
    signs = torch.where(
        steering_targets == 2,
        torch.ones_like(signs),
        signs,
    )
    signs = torch.where(
        steering_targets == 0,
        -torch.ones_like(signs),
        signs,
    )
    mask = signs.abs() > 0.0
    if int(mask.sum().detach().cpu()) <= 0:
        return pred_vectors.sum() * 0.0
    return F.softplus(-8.0 * signs[mask] * pred_vectors[mask, 1]).mean()


def _selection_score(metrics: dict[str, Any]) -> float:
    return (
        2.0 * float(metrics["target_steering_pipeline_success"])
        + 0.5 * float(metrics["target_recall"])
        + 0.25 * float(metrics["target_selection_precision"])
        - 0.75 * float(metrics["false_claim_rate"])
    )


def _aux_features(row: dict[str, Any]) -> np.ndarray:
    block = [float(v) for v in row.get("integrated_body_motion_block", ())[:3]]
    window = [float(v) for v in row.get("integrated_body_motion_window", ())[:3]]
    while len(block) < 3:
        block.append(0.0)
    while len(window) < 3:
        window.append(0.0)
    command = row.get("command") or {}
    command_values = []
    for field in ("vx_body_mps", "vy_body_mps", "yaw_rate_radps"):
        values = [float(v) for v in command.get(field, ())]
        command_values.append(float(np.mean(values)) if values else 0.0)
    primitive = str(command.get("primitive_name", ""))
    primitive_one_hot = [1.0 if primitive == name else 0.0 for name in PRIMITIVE_NAMES]
    return np.asarray(block + window + command_values + primitive_one_hot, dtype=np.float32)


def _motion_delta_from_row(
    row: dict[str, Any],
    *,
    field: str,
    translation_scale_m: float,
) -> np.ndarray:
    values = [float(v) for v in row.get(field, ())[:3]]
    while len(values) < 3:
        values.append(0.0)
    scale = max(1e-6, float(translation_scale_m))
    return np.asarray([values[0] / scale, values[1] / scale, values[2]], dtype=np.float32)


def _aux_stats(sequences: dict[tuple[str, int, int], list[Frame]]) -> dict[str, np.ndarray]:
    features = np.stack([frame.aux.numpy() for seq in sequences.values() for frame in seq])
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
                motion_block=frame.motion_block,
                motion_window=frame.motion_window,
                exact_motion=frame.exact_motion,
                visible_mask=frame.visible_mask,
                visible_vec=frame.visible_vec,
                all_vec=frame.all_vec,
                memory_mask=frame.memory_mask,
                queries=frame.queries,
            )
            for frame in sequence
        ]


def _visible_counts(sequences: dict[tuple[str, int, int], list[Frame]]) -> dict[str, int]:
    positive = 0
    total = 0
    for sequence in sequences.values():
        for frame in sequence:
            positive += int(frame.visible_mask.sum().item())
            total += int(frame.visible_mask.numel())
    return {"positive": positive, "negative": total - positive, "total": total}


def _query_counts(sequences: dict[tuple[str, int, int], list[Frame]]) -> dict[str, int]:
    positive = 0
    total = 0
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                positive += 1 if query.target >= 0.5 else 0
                total += 1
    return {"positive": positive, "negative": total - positive, "total": total}


def _hard_query_groups(
    sequences: dict[tuple[str, int, int], list[Frame]],
) -> dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]]:
    groups: dict[tuple[str, int, int, str], dict[str, list[HardQueryExample]]] = (
        defaultdict(lambda: {"positive": [], "negative": []})
    )
    for seq_key, sequence in sequences.items():
        for step_idx, frame in enumerate(sequence):
            for query in frame.queries:
                example = HardQueryExample(
                    seq_key=seq_key,
                    step_idx=int(step_idx),
                    color_index=int(query.color_index),
                )
                bucket = "positive" if query.target >= 0.5 else "negative"
                groups[query.group_key][bucket].append(example)
    return {
        key: bucket
        for key, bucket in groups.items()
        if bucket["positive"] and bucket["negative"]
    }


def _hard_group_label_counts(
    sequences: dict[tuple[str, int, int], list[Frame]],
) -> dict[tuple[str, int, int, str], dict[str, int]]:
    counts: dict[tuple[str, int, int, str], dict[str, int]] = defaultdict(
        lambda: {"positive": 0, "negative": 0}
    )
    for sequence in sequences.values():
        for frame in sequence:
            for query in frame.queries:
                bucket = "positive" if query.target >= 0.5 else "negative"
                counts[query.group_key][bucket] += 1
    return {
        key: dict(value)
        for key, value in counts.items()
        if value["positive"] > 0 and value["negative"] > 0
    }


def _pos_weight(counts: dict[str, int]) -> float:
    return float(counts.get("negative", 0)) / float(max(1, counts.get("positive", 0)))


def _vector_target(landmark: dict[str, Any], *, range_scale_m: float) -> np.ndarray:
    bearing = _finite_float(landmark.get("bearing_body_rad"), 0.0)
    range_m = max(0.0, min(float(range_scale_m), _finite_float(landmark.get("range_m"), 0.0)))
    return np.asarray(
        [
            (range_m * math.cos(bearing)) / float(range_scale_m),
            (range_m * math.sin(bearing)) / float(range_scale_m),
        ],
        dtype=np.float32,
    )


def _steering_index(bearing_rad: float) -> int:
    if bearing_rad <= -0.1:
        return 0
    if bearing_rad >= 0.1:
        return 2
    return 1


def _steering_name(index: int) -> str:
    return STEERING_CLASSES[max(0, min(index, len(STEERING_CLASSES) - 1))]


def _color_vocab(*row_groups: list[dict[str, Any]]) -> list[str]:
    colors = {
        _object_color(str(landmark.get("object_id", "")))
        for rows in row_groups
        for row in rows
        for landmark in row.get("landmarks", ())
    }
    colors.discard("unknown")
    if not colors:
        raise SystemExit("no colors in dataset")
    return sorted(colors)


def _object_color(object_id: str) -> str:
    lowered = str(object_id).lower()
    for color in _COLOR_RGB:
        if color != "unknown" and color in lowered:
            return color
    return "unknown"


def _landmark_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(landmark.get("object_id", "")): landmark
        for landmark in row.get("landmarks", ())
        if str(landmark.get("object_id", ""))
    }


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
