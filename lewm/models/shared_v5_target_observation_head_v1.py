"""Detached-feature four-colour target observation head (source only).

This head has no image encoder, frame runner, decoder, preprocessor, path, or
checkpoint loader.  One call scores all four canonical semantic colours from
already-detached Shared V5 patch and BEV features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CANONICAL_TARGET_COLORS_V1 = ("red", "yellow", "blue", "green")
TARGET_OBSERVATION_HEAD_CONFIG_SCHEMA_V1 = (
    "lewm_go2_shared_v5_target_observation_head_config_v1"
)
PRODUCTION_SHARED_V5_TARGET_OBSERVATION_HEAD_V1 = None
PRODUCTION_TARGET_HEAD_CONFIG_SHA256_V1 = None
PRODUCTION_TARGET_HEAD_CHECKPOINT_SHA256_V1 = None
PRODUCTION_TARGET_HEAD_CALIBRATION_SHA256_V1 = None


class SharedV5TargetObservationHeadV1Error(ValueError):
    """The detached feature or four-colour output contract was violated."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be a positive exact integer")
    return value


def _positive_finite(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be positive and finite")
    return result


def _require_detached_finite_tensor(
    value: object,
    *,
    name: str,
    rank: int,
    last_dim: int | None = None,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be an exact torch.Tensor")
    if value.ndim != rank:
        raise SharedV5TargetObservationHeadV1Error(f"{name} rank is invalid")
    if not value.dtype.is_floating_point:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must have floating dtype")
    if value.requires_grad or value.grad_fn is not None:
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be detached cached features")
    if last_dim is not None and value.shape[-1] != last_dim:
        raise SharedV5TargetObservationHeadV1Error(f"{name} feature dimension changed")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise SharedV5TargetObservationHeadV1Error(f"{name} must be nonempty and finite")
    return value


@dataclass(frozen=True)
class SharedV5TargetObservationHeadConfigV1:
    patch_feature_dim: int
    bev_feature_dim: int
    hidden_dim: int = 128
    color_embedding_dim: int = 16
    minimum_scale: float = 1.0e-4
    maximum_range_m: float = 12.0
    schema: str = TARGET_OBSERVATION_HEAD_CONFIG_SCHEMA_V1

    def __post_init__(self) -> None:
        for name in (
            "patch_feature_dim",
            "bev_feature_dim",
            "hidden_dim",
            "color_embedding_dim",
        ):
            _positive_int(getattr(self, name), name=name)
        _positive_finite(self.minimum_scale, name="minimum_scale")
        _positive_finite(self.maximum_range_m, name="maximum_range_m")
        if self.minimum_scale >= self.maximum_range_m:
            raise SharedV5TargetObservationHeadV1Error(
                "minimum_scale must be smaller than maximum_range_m"
            )
        if type(self.schema) is not str or self.schema != TARGET_OBSERVATION_HEAD_CONFIG_SCHEMA_V1:
            raise SharedV5TargetObservationHeadV1Error("target head schema changed")

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class FourColorTargetObservationOutputV1:
    """One batched prediction row for every canonical colour."""

    colors: tuple[str, str, str, str]
    presence_logit: torch.Tensor
    presence_probability: torch.Tensor
    bearing_mean_rad: torch.Tensor
    bearing_scale_rad: torch.Tensor
    range_mean_m: torch.Tensor
    range_scale_m: torch.Tensor
    uncertainty: torch.Tensor
    quality: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.colors) is not tuple or self.colors != CANONICAL_TARGET_COLORS_V1:
            raise SharedV5TargetObservationHeadV1Error("target output colour order changed")
        names = (
            "presence_logit",
            "presence_probability",
            "bearing_mean_rad",
            "bearing_scale_rad",
            "range_mean_m",
            "range_scale_m",
            "uncertainty",
            "quality",
        )
        first_shape: tuple[int, int] | None = None
        for name in names:
            value = getattr(self, name)
            if type(value) is not torch.Tensor or value.ndim != 2 or value.shape[1] != 4:
                raise SharedV5TargetObservationHeadV1Error(
                    f"{name} must have exact [batch, four colours] shape"
                )
            if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all().item()):
                raise SharedV5TargetObservationHeadV1Error(f"{name} must be finite floating output")
            shape = (int(value.shape[0]), int(value.shape[1]))
            if first_shape is None:
                first_shape = shape
            elif shape != first_shape:
                raise SharedV5TargetObservationHeadV1Error("target output shapes differ")
        if bool(((self.presence_probability < 0.0) | (self.presence_probability > 1.0)).any().item()):
            raise SharedV5TargetObservationHeadV1Error("presence probability left [0,1]")
        if bool(((self.quality < 0.0) | (self.quality > 1.0)).any().item()):
            raise SharedV5TargetObservationHeadV1Error("quality left [0,1]")
        for value in (self.bearing_scale_rad, self.range_mean_m, self.range_scale_m, self.uncertainty):
            if bool((value <= 0.0).any().item()):
                raise SharedV5TargetObservationHeadV1Error("scale/range/uncertainty must be positive")

    @property
    def batch_size(self) -> int:
        return int(self.presence_logit.shape[0])


class SharedV5TargetObservationHeadV1(nn.Module):
    """Small batched head over detached patch/BEV feature caches."""

    owned_encoder_count = 0
    owned_rgb_preprocessor_count = 0

    def __init__(self, config: SharedV5TargetObservationHeadConfigV1) -> None:
        super().__init__()
        if type(config) is not SharedV5TargetObservationHeadConfigV1:
            raise TypeError("config must be exact SharedV5TargetObservationHeadConfigV1")
        self.config = config
        context_dim = config.patch_feature_dim + config.bev_feature_dim
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.color_embedding = nn.Parameter(
            torch.empty(4, config.color_embedding_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(config.hidden_dim + config.color_embedding_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.output_projection = nn.Linear(config.hidden_dim, 7)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.color_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def architecture_config_sha256(self) -> str:
        return self.config.content_sha256

    def forward(
        self,
        patch_features: torch.Tensor,
        bev_features: torch.Tensor,
    ) -> FourColorTargetObservationOutputV1:
        patch = _require_detached_finite_tensor(
            patch_features,
            name="patch_features",
            rank=3,
            last_dim=self.config.patch_feature_dim,
        )
        bev = _require_detached_finite_tensor(
            bev_features,
            name="bev_features",
            rank=4,
        )
        if bev.shape[1] != self.config.bev_feature_dim:
            raise SharedV5TargetObservationHeadV1Error("BEV channel dimension changed")
        if patch.shape[0] != bev.shape[0]:
            raise SharedV5TargetObservationHeadV1Error("patch and BEV batch sizes differ")
        patch_version = patch._version
        bev_version = bev._version
        patch_pool = patch.mean(dim=1)
        bev_pool = bev.mean(dim=(-2, -1))
        context = self.context_projection(torch.cat((patch_pool, bev_pool), dim=-1))
        batch_size = int(context.shape[0])
        context = context[:, None, :].expand(batch_size, 4, self.config.hidden_dim)
        colors = self.color_embedding[None, :, :].expand(batch_size, 4, -1)
        raw = self.output_projection(self.trunk(torch.cat((context, colors), dim=-1)))
        if patch._version != patch_version or bev._version != bev_version:
            raise SharedV5TargetObservationHeadV1Error("cached features were mutated")
        presence_logit = raw[..., 0]
        bearing_mean = math.pi * torch.tanh(raw[..., 1])
        bearing_scale = F.softplus(raw[..., 2]) + self.config.minimum_scale
        range_mean = self.config.minimum_scale + (
            self.config.maximum_range_m - self.config.minimum_scale
        ) * torch.sigmoid(raw[..., 3])
        range_scale = F.softplus(raw[..., 4]) + self.config.minimum_scale
        uncertainty = F.softplus(raw[..., 5]) + self.config.minimum_scale
        quality = torch.sigmoid(raw[..., 6])
        return FourColorTargetObservationOutputV1(
            colors=CANONICAL_TARGET_COLORS_V1,
            presence_logit=presence_logit,
            presence_probability=torch.sigmoid(presence_logit),
            bearing_mean_rad=bearing_mean,
            bearing_scale_rad=bearing_scale,
            range_mean_m=range_mean,
            range_scale_m=range_scale,
            uncertainty=uncertainty,
            quality=quality,
        )


def initialize_deterministic_mock_weights_v1(
    head: SharedV5TargetObservationHeadV1,
    *,
    seed: int,
) -> None:
    """Deterministic synthetic-test initialization; it grants no checkpoint identity."""

    if type(head) is not SharedV5TargetObservationHeadV1:
        raise TypeError("head must be exact SharedV5TargetObservationHeadV1")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact nonnegative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in head.parameters():
            values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
            values.uniform_(-0.05, 0.05, generator=generator)
            parameter.copy_(values.to(device=parameter.device))


__all__ = [
    "CANONICAL_TARGET_COLORS_V1",
    "FourColorTargetObservationOutputV1",
    "SharedV5TargetObservationHeadConfigV1",
    "SharedV5TargetObservationHeadV1",
    "SharedV5TargetObservationHeadV1Error",
    "initialize_deterministic_mock_weights_v1",
]
