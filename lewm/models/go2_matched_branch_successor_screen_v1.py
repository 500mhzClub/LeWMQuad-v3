"""Compact predictors for the matched-branch successor engineering screen.

The models in this module consume frozen spatial features rather than RGB.  They
are deliberately small: the RSSM variant is a latent-dynamics component, not a
Dreamer agent (there is no actor, critic, reward model, or replay system here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


__all__ = (
    "CompactRSSMPredictorV1",
    "CompactRSSMTrainingOutputV1",
    "DenseActionConditionedPredictorV1",
    "DeterministicStateSpacePredictorV1",
    "diagonal_gaussian_kl_v1",
)


_CONTEXT_STEPS = 3
_HISTORY_STEPS = 2
_GRID_SIZE = 16
_TOKEN_COUNT = _GRID_SIZE * _GRID_SIZE


def _validate_dimensions(feature_dim: int, hidden_dim: int, action_count: int) -> None:
    if min(feature_dim, hidden_dim, action_count) < 1:
        raise ValueError("feature_dim, hidden_dim, and action_count must be positive")


def _validate_inputs(
    context: torch.Tensor,
    history_actions: torch.Tensor,
    candidate_action: torch.Tensor,
    *,
    feature_dim: int,
    action_count: int,
) -> int:
    if context.ndim != 4:
        raise ValueError("context must have shape (B,3,256,D)")
    batch, steps, tokens, width = context.shape
    if batch < 1 or steps != _CONTEXT_STEPS or tokens != _TOKEN_COUNT:
        raise ValueError("context must have shape (B,3,256,D) with B >= 1")
    if width != feature_dim:
        raise ValueError("context feature dimension does not match the model")
    if context.dtype != torch.float32:
        raise TypeError("context must use float32")
    if not bool(torch.isfinite(context).all()):
        raise FloatingPointError("context contains a nonfinite value")
    if history_actions.shape != (batch, _HISTORY_STEPS):
        raise ValueError("history_actions must have shape (B,2)")
    if candidate_action.shape != (batch,):
        raise ValueError("candidate_action must have shape (B,)")
    if history_actions.dtype != torch.long or candidate_action.dtype != torch.long:
        raise TypeError("action IDs must use torch.long")
    if history_actions.device != context.device or candidate_action.device != context.device:
        raise TypeError("context and action IDs must share one device")
    if (
        bool((history_actions < 0).any())
        or bool((history_actions >= action_count).any())
        or bool((candidate_action < 0).any())
        or bool((candidate_action >= action_count).any())
    ):
        raise ValueError("action ID is outside the configured vocabulary")
    return batch


def _validate_target(
    target: torch.Tensor,
    *,
    batch: int,
    feature_dim: int,
    device: torch.device,
) -> None:
    if target.shape != (batch, _TOKEN_COUNT, feature_dim):
        raise ValueError("target must have shape (B,256,D)")
    if target.dtype != torch.float32:
        raise TypeError("target must use float32")
    if target.device != device:
        raise TypeError("target and context must share one device")
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("target contains a nonfinite value")


class _FiLMSpatialBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.channel_mixing = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, grid: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        normalized = self.norm(tokens)
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        normalized = normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        update = normalized.transpose(1, 2).reshape(batch, channels, height, width)
        return grid + self.channel_mixing(self.depthwise(update))


class DenseActionConditionedPredictorV1(nn.Module):
    """Two-block action-conditioned predictor over a 16 x 16 token grid."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 128,
        action_count: int = 9,
    ) -> None:
        super().__init__()
        _validate_dimensions(feature_dim, hidden_dim, action_count)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_count = int(action_count)
        self.input_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.temporal_mixing = nn.Conv2d(
            _CONTEXT_STEPS * self.hidden_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.action_embedding = nn.Embedding(self.action_count, self.hidden_dim)
        self.action_slot_embedding = nn.Parameter(
            torch.empty(_CONTEXT_STEPS, self.hidden_dim)
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(_CONTEXT_STEPS * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            (_FiLMSpatialBlock(self.hidden_dim), _FiLMSpatialBlock(self.hidden_dim))
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.feature_dim)
        nn.init.trunc_normal_(self.action_slot_embedding, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        batch = _validate_inputs(
            context,
            history_actions,
            candidate_action,
            feature_dim=self.feature_dim,
            action_count=self.action_count,
        )
        projected = self.input_projection(context)
        grid = projected.permute(0, 1, 3, 2).reshape(
            batch,
            _CONTEXT_STEPS * self.hidden_dim,
            _GRID_SIZE,
            _GRID_SIZE,
        )
        grid = self.temporal_mixing(grid)

        action_ids = torch.cat((history_actions, candidate_action.unsqueeze(1)), dim=1)
        action_tokens = self.action_embedding(action_ids) + self.action_slot_embedding
        conditioning = self.condition_projection(action_tokens.flatten(1))
        for block in self.blocks:
            grid = block(grid, conditioning)

        tokens = grid.flatten(2).transpose(1, 2)
        prediction = context[:, -1] + self.output_projection(self.output_norm(tokens))
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("dense spatial prediction became nonfinite")
        return prediction


class _PooledContextDynamics(nn.Module):
    """Shared compact recurrence for the pooled state-space models."""

    def __init__(self, feature_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_embedding = nn.Embedding(action_count, hidden_dim)
        self.initial_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.history_update = nn.GRUCell(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
    ) -> torch.Tensor:
        observations = self.observation_projection(context.mean(dim=2))
        state = torch.zeros_like(observations[:, 0])
        state = self.initial_update(observations[:, 0], state)
        for step in range(_HISTORY_STEPS):
            action = self.action_embedding(history_actions[:, step])
            recurrent_input = torch.cat((observations[:, step + 1], action), dim=-1)
            state = self.history_update(recurrent_input, state)
        return state


class _PositionalQueryDecoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(_TOKEN_COUNT, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        queries = self.state_projection(state).unsqueeze(1) + self.position_embedding
        hidden = self.activation(self.hidden_projection(self.norm(queries)))
        return self.output_projection(hidden)


class DeterministicStateSpacePredictorV1(nn.Module):
    """Conventional pooled recurrent dynamics with a deterministic transition."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 128,
        action_count: int = 9,
    ) -> None:
        super().__init__()
        _validate_dimensions(feature_dim, hidden_dim, action_count)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_count = int(action_count)
        self.context_dynamics = _PooledContextDynamics(
            self.feature_dim, self.hidden_dim, self.action_count
        )
        self.candidate_transition = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.decoder = _PositionalQueryDecoder(
            self.hidden_dim, self.hidden_dim, self.feature_dim
        )

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(
            context,
            history_actions,
            candidate_action,
            feature_dim=self.feature_dim,
            action_count=self.action_count,
        )
        state = self.context_dynamics(context, history_actions)
        candidate = self.context_dynamics.action_embedding(candidate_action)
        successor_state = self.candidate_transition(candidate, state)
        prediction = context[:, -1] + self.decoder(successor_state)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("deterministic state-space prediction became nonfinite")
        return prediction


@dataclass(frozen=True)
class CompactRSSMTrainingOutputV1:
    prediction: torch.Tensor
    deterministic_state: torch.Tensor
    latent_state: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_std: torch.Tensor


def diagonal_gaussian_kl_v1(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "batchmean", "sum"] = "mean",
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians parameterized by log standard deviation."""

    if (
        posterior_mean.shape != posterior_log_std.shape
        or posterior_mean.shape != prior_mean.shape
        or posterior_mean.shape != prior_log_std.shape
        or posterior_mean.ndim != 2
        or posterior_mean.shape[0] < 1
        or posterior_mean.shape[1] < 1
    ):
        raise ValueError("all Gaussian parameters must share nonempty shape (B,Z)")
    tensors = (posterior_mean, posterior_log_std, prior_mean, prior_log_std)
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise TypeError("Gaussian parameters must use float32")
    if any(tensor.device != posterior_mean.device for tensor in tensors[1:]):
        raise TypeError("Gaussian parameters must share one device")
    if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise FloatingPointError("Gaussian parameter contains a nonfinite value")
    if reduction not in ("none", "mean", "batchmean", "sum"):
        raise ValueError("unsupported KL reduction")

    variance_ratio = torch.exp(2.0 * (posterior_log_std - prior_log_std))
    mean_term = (posterior_mean - prior_mean).square() * torch.exp(
        -2.0 * prior_log_std
    )
    per_dimension = 0.5 * (
        variance_ratio + mean_term - 1.0 + 2.0 * (prior_log_std - posterior_log_std)
    )
    if not bool(torch.isfinite(per_dimension).all()):
        raise FloatingPointError("diagonal Gaussian KL became nonfinite")
    per_row = per_dimension.sum(dim=-1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    if reduction == "batchmean":
        return per_row.mean()
    return per_dimension.mean()


class CompactRSSMPredictorV1(nn.Module):
    """Compact RSSM-style successor model without Dreamer policy/value heads."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int = 128,
        stochastic_dim: int = 32,
        action_count: int = 9,
        minimum_log_std: float = -5.0,
        maximum_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        _validate_dimensions(feature_dim, hidden_dim, action_count)
        if stochastic_dim < 1:
            raise ValueError("stochastic_dim must be positive")
        if minimum_log_std >= maximum_log_std:
            raise ValueError("minimum_log_std must be below maximum_log_std")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.stochastic_dim = int(stochastic_dim)
        self.action_count = int(action_count)
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.context_dynamics = _PooledContextDynamics(
            self.feature_dim, self.hidden_dim, self.action_count
        )
        self.candidate_transition = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.prior = nn.Linear(
            self.hidden_dim * 2,
            self.stochastic_dim * 2,
        )
        self.target_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.posterior = nn.Linear(
            self.hidden_dim * 3,
            self.stochastic_dim * 2,
        )
        self.decoder = _PositionalQueryDecoder(
            self.hidden_dim + self.stochastic_dim,
            self.hidden_dim,
            self.feature_dim,
        )

    def _statistics(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = parameters.chunk(2, dim=-1)
        return mean, log_std.clamp(self.minimum_log_std, self.maximum_log_std)

    def _prior_dynamics(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_inputs(
            context,
            history_actions,
            candidate_action,
            feature_dim=self.feature_dim,
            action_count=self.action_count,
        )
        context_state = self.context_dynamics(context, history_actions)
        candidate = self.context_dynamics.action_embedding(candidate_action)
        deterministic_state = self.candidate_transition(candidate, context_state)
        prior_mean, prior_log_std = self._statistics(
            self.prior(torch.cat((deterministic_state, candidate), dim=-1))
        )
        return deterministic_state, candidate, prior_mean, prior_log_std

    def _decode(
        self,
        context: torch.Tensor,
        deterministic_state: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> torch.Tensor:
        update = self.decoder(torch.cat((deterministic_state, latent_state), dim=-1))
        prediction = context[:, -1] + update
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("RSSM prediction became nonfinite")
        return prediction

    def forward(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict successor tokens using the candidate-conditioned prior mean."""

        deterministic_state, _, prior_mean, _ = self._prior_dynamics(
            context,
            history_actions,
            candidate_action,
        )
        return self._decode(context, deterministic_state, prior_mean)

    def training_output(
        self,
        context: torch.Tensor,
        history_actions: torch.Tensor,
        candidate_action: torch.Tensor,
        target_tokens: torch.Tensor,
        *,
        sample_posterior: bool = True,
    ) -> CompactRSSMTrainingOutputV1:
        """Return posterior reconstruction and prior/posterior KL parameters."""

        if not isinstance(sample_posterior, bool):
            raise TypeError("sample_posterior must be a bool")
        deterministic_state, candidate, prior_mean, prior_log_std = (
            self._prior_dynamics(context, history_actions, candidate_action)
        )
        _validate_target(
            target_tokens,
            batch=context.shape[0],
            feature_dim=self.feature_dim,
            device=context.device,
        )
        target_observation = self.target_projection(target_tokens.mean(dim=1))
        posterior_mean, posterior_log_std = self._statistics(
            self.posterior(
                torch.cat(
                    (deterministic_state, candidate, target_observation), dim=-1
                )
            )
        )
        if sample_posterior:
            noise = torch.randn_like(posterior_mean)
            latent_state = posterior_mean + noise * posterior_log_std.exp()
        else:
            latent_state = posterior_mean
        prediction = self._decode(context, deterministic_state, latent_state)
        outputs = (
            deterministic_state,
            latent_state,
            prior_mean,
            prior_log_std,
            posterior_mean,
            posterior_log_std,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in outputs):
            raise FloatingPointError("RSSM state became nonfinite")
        return CompactRSSMTrainingOutputV1(
            prediction=prediction,
            deterministic_state=deterministic_state,
            latent_state=latent_state,
            prior_mean=prior_mean,
            prior_log_std=prior_log_std,
            posterior_mean=posterior_mean,
            posterior_log_std=posterior_log_std,
        )

    @staticmethod
    def kl_divergence(
        result: CompactRSSMTrainingOutputV1,
        *,
        reduction: Literal["none", "mean", "batchmean", "sum"] = "mean",
    ) -> torch.Tensor:
        return diagonal_gaussian_kl_v1(
            result.posterior_mean,
            result.posterior_log_std,
            result.prior_mean,
            result.prior_log_std,
            reduction=reduction,
        )
