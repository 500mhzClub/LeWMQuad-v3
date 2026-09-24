"""Planning-oriented additions for the Go2 world-model progression experiment.

This module deliberately leaves the frozen historical temporal model untouched.
It supplies two narrowly scoped mechanisms for a fresh development comparison:

* a dynamic spatial-token prediction route that can emit either the registered
  64-token panel or a complete, re-entrant 256-token grid; and
* an action decoder that predicts the requested primitive from a spatial latent
  displacement.  The decoder never consumes an action label as an input.

Neither mechanism is evidence of causal action fidelity by itself.  They are
intended to be adjudicated on matched simulator branches and planning regret.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DynamicSpatialPredictionV1:
    """Prediction result for a caller-selected spatial target panel."""

    raw: torch.Tensor
    normalized: torch.Tensor
    recurrent_memory: torch.Tensor


def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tokens.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )


def predict_dynamic_spatial_tokens_v1(
    arm: nn.Module,
    encoded_history: torch.Tensor,
    actions: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    candidate_blind: bool = False,
    time_indices: torch.Tensor | None = None,
) -> DynamicSpatialPredictionV1:
    """Run a temporal patch-memory head for an arbitrary target-token count.

    The registered predecessor uses 64 target tokens.  Its predictor blocks,
    positional queries, and output projection are token-count agnostic, so the
    same parameterization can also predict all 256 positions.  The full-grid
    route is what makes an autoregressive rollout interface structurally
    possible; it does not assert that the resulting rollout is accurate.
    """

    config: Any = getattr(arm, "config", None)
    required = (
        "spatial_token_count",
        "feature_dim",
        "action_count",
        "time_embedding_count",
        "temporal_hidden_dim",
        "normalization_epsilon",
    )
    if config is None or any(not hasattr(config, name) for name in required):
        raise TypeError("arm lacks the registered temporal configuration")
    if encoded_history.ndim != 4:
        raise ValueError("encoded_history must have shape (B,S,N,D)")
    batch, steps, token_count, feature_dim = encoded_history.shape
    if (
        batch < 1
        or not 1 <= steps <= 3
        or token_count != int(config.spatial_token_count)
        or feature_dim != int(config.feature_dim)
        or encoded_history.dtype != torch.float32
        or not bool(torch.isfinite(encoded_history).all())
    ):
        raise ValueError("encoded_history shape, dtype, or values changed")
    if actions.shape != (batch, steps) or actions.dtype != torch.long:
        raise TypeError("actions must be long with shape (B,S)")
    if bool((actions < 0).any()) or bool((actions >= int(config.action_count)).any()):
        raise ValueError("action IDs left the configured vocabulary")
    if (
        target_indices.ndim != 2
        or target_indices.shape[0] != batch
        or target_indices.dtype != torch.long
        or not 1 <= target_indices.shape[1] <= token_count
    ):
        raise TypeError("target_indices must be long with shape (B,K), 1 <= K <= N")
    device = encoded_history.device
    if actions.device != device or target_indices.device != device:
        raise TypeError("arm inputs must share one device")
    if (
        bool((target_indices < 0).any())
        or bool((target_indices >= token_count).any())
        or (
            target_indices.shape[1] > 1
            and not bool((target_indices[:, 1:] > target_indices[:, :-1]).all())
        )
    ):
        raise ValueError("target indices must be strictly increasing and in range")

    if time_indices is None:
        times = torch.arange(steps, dtype=torch.long, device=device).unsqueeze(0)
        times = times.expand(batch, -1)
    else:
        times = time_indices
        if times.ndim == 1 and times.shape == (steps,):
            times = times.unsqueeze(0).expand(batch, -1)
        if times.shape != (batch, steps) or times.dtype != torch.long or times.device != device:
            raise TypeError("time_indices must be long with shape (B,S) on the input device")
    if bool((times < 0).any()) or bool((times >= int(config.time_embedding_count)).any()):
        raise ValueError("time indices left the configured range")

    action_conditioning = arm.action_embedding(actions)
    if candidate_blind:
        gate = torch.ones((1, steps, 1), dtype=action_conditioning.dtype, device=device)
        gate[:, -1] = 0.0
        action_conditioning = action_conditioning * gate
    conditioning = action_conditioning + arm.time_embedding(times)
    streams = (encoded_history + conditioning.unsqueeze(2)).permute(0, 2, 1, 3)
    streams = streams.reshape(batch * token_count, steps, feature_dim)
    initial_hidden = torch.zeros(
        1,
        batch * token_count,
        int(config.temporal_hidden_dim),
        dtype=streams.dtype,
        device=device,
    )
    recurrent_streams, _ = arm.temporal_gru(streams, initial_hidden)
    recurrent_memory = recurrent_streams[:, -1].reshape(
        batch,
        token_count,
        int(config.temporal_hidden_dim),
    )

    memory_tokens = recurrent_memory + arm.predictor_position.unsqueeze(0)
    query_positions = _gather_tokens(
        arm.predictor_position.unsqueeze(0).expand(batch, -1, -1),
        target_indices,
    )
    queries = arm.predictor_mask_token.expand(batch, target_indices.shape[1], -1)
    predictor = torch.cat((memory_tokens, queries + query_positions), dim=1)
    for block in arm.predictor_blocks:
        predictor = block(predictor)
    predicted_queries = arm.predictor_norm(
        predictor[:, -target_indices.shape[1] :]
    )
    raw = arm.predictor_output(predicted_queries)
    normalized = F.normalize(
        raw,
        p=2.0,
        dim=-1,
        eps=float(config.normalization_epsilon),
    )
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("dynamic spatial prediction became nonfinite")
    return DynamicSpatialPredictionV1(
        raw=raw,
        normalized=normalized,
        recurrent_memory=recurrent_memory,
    )


def normalized_spatial_energy_v1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-row normalized token energy for any common token count."""

    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape (B,K,D)")
    if prediction.shape[0] < 1 or prediction.shape[1] < 1 or prediction.shape[2] < 1:
        raise ValueError("prediction and target cannot contain an empty axis")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("prediction and target must use float32")
    if prediction.device != target.device:
        raise TypeError("prediction and target must share one device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("prediction or target contains a nonfinite value")
    pred = F.normalize(prediction, p=2.0, dim=-1, eps=epsilon)
    tgt = F.normalize(target.detach(), p=2.0, dim=-1, eps=epsilon)
    return 0.5 * (pred - tgt).square().sum(dim=-1).mean(dim=-1)


class SpatialLatentDisplacementActionDecoderV1(nn.Module):
    """Decode an action class from a spatial latent displacement only."""

    def __init__(
        self,
        *,
        feature_dim: int = 192,
        hidden_dim: int = 192,
        spatial_token_count: int = 256,
        action_count: int = 9,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim, spatial_token_count, action_count) < 1:
            raise ValueError("decoder dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.spatial_token_count = int(spatial_token_count)
        self.action_count = int(action_count)
        self.delta_norm = nn.LayerNorm(self.feature_dim)
        self.delta_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.spatial_token_count, self.hidden_dim)
        )
        self.attention_score = nn.Linear(self.hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_count),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        current_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current_tokens.ndim != 3 or future_tokens.shape != current_tokens.shape:
            raise ValueError("current and future tokens must share shape (B,K,D)")
        batch, selected_tokens, feature_dim = current_tokens.shape
        if batch < 1 or not 1 <= selected_tokens <= self.spatial_token_count:
            raise ValueError("decoder token panel is empty or too large")
        if feature_dim != self.feature_dim:
            raise ValueError("decoder feature dimension changed")
        if current_tokens.dtype != torch.float32 or future_tokens.dtype != torch.float32:
            raise TypeError("decoder tokens must use float32")
        if current_tokens.device != future_tokens.device:
            raise TypeError("decoder tokens must share one device")
        if not bool(torch.isfinite(current_tokens).all()) or not bool(torch.isfinite(future_tokens).all()):
            raise FloatingPointError("decoder tokens contain a nonfinite value")
        if token_indices is None:
            if selected_tokens != self.spatial_token_count:
                raise ValueError("partial token panels require explicit token_indices")
            indices = torch.arange(
                self.spatial_token_count,
                dtype=torch.long,
                device=current_tokens.device,
            ).unsqueeze(0).expand(batch, -1)
        else:
            indices = token_indices
            if indices.shape != (batch, selected_tokens) or indices.dtype != torch.long:
                raise TypeError("token_indices must be long with shape (B,K)")
            if indices.device != current_tokens.device:
                raise TypeError("token_indices and decoder tokens must share one device")
            if bool((indices < 0).any()) or bool((indices >= self.spatial_token_count).any()):
                raise ValueError("token_indices left the spatial range")

        current = F.normalize(current_tokens, p=2.0, dim=-1, eps=1.0e-8)
        future = F.normalize(future_tokens, p=2.0, dim=-1, eps=1.0e-8)
        hidden = self.delta_projection(self.delta_norm(future - current))
        positions = self.position_embedding.expand(batch, -1, -1).gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim),
        )
        hidden = hidden + positions
        weights = torch.softmax(self.attention_score(hidden), dim=1)
        attentive = (weights * hidden).sum(dim=1)
        mean = hidden.mean(dim=1)
        logits = self.classifier(torch.cat((attentive, mean), dim=-1))
        if logits.shape != (batch, self.action_count) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("action-decoder logits are invalid")
        return logits


__all__ = [
    "DynamicSpatialPredictionV1",
    "SpatialLatentDisplacementActionDecoderV1",
    "normalized_spatial_energy_v1",
    "predict_dynamic_spatial_tokens_v1",
]
