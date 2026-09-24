"""Token-primary action-conditioned JEPA: predict in ViT patch-token space.

The predictive state is the encoder's ``16x16x192`` patch-token grid.  Nothing in
the predictive path is routed through the ``64x64x64`` learned-query BEV
bottleneck, which the paired A/B screen rejected as the primary latent
(``docs/lewm_go2_v4_occupied_weight_paired_screen_result_2026-08-05.md``).

The BEV branch is deliberately retained, unchanged, as an **auxiliary geometric
output / planning readout**.  It is not part of the predictive state and its
objective is untouched here.

Context worth carrying: in the direct-BEV lineage the action-conditioned
predictor was never trained -- the update-400 checkpoint records 0 predictor
forward, backward and optimizer calls against 400 perception updates.  This
module therefore builds the predictive path for the first time in this line
rather than porting a trained one.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_GRID = 16
TOKEN_COUNT = TOKEN_GRID * TOKEN_GRID          # 256
TOKEN_DIM = 192
ACTION_COUNT = 9
COMMAND_DIM = 3                                 # vx, vy, wz


class TokenJepaOutputV1(NamedTuple):
    """Predicted next tokens and the components used to train them."""

    predicted_tokens: torch.Tensor
    jepa_loss: torch.Tensor
    cosine_to_target: torch.Tensor
    prediction_variance: torch.Tensor
    persistence_cosine: torch.Tensor


class _FiLMBlockV1(nn.Module):
    """Pre-norm self-attention + MLP over the token sequence, FiLM-conditioned."""

    def __init__(self, dim: int = TOKEN_DIM, heads: int = 6, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        # FiLM produces a per-channel scale and shift from the action condition.
        self.film = nn.Linear(dim, 2 * dim)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(condition).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        normed = self.norm_attention(tokens) * (1.0 + scale) + shift
        attended, _weights = self.attention(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens


class TokenPrimaryActionConditionedPredictorV1(nn.Module):
    """Residual action-conditioned transition in patch-token space.

    Inputs are the current ``(B, 256, 192)`` token grid, a one-hot action and its
    commanded body-frame velocity.  The output is a residual delta, so the
    identity transition is the initialisation prior and a collapsed predictor is
    visible as a near-zero delta rather than hidden behind a learned mean.
    """

    def __init__(self, dim: int = TOKEN_DIM, depth: int = 2, heads: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.condition = nn.Sequential(
            nn.Linear(ACTION_COUNT + COMMAND_DIM, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.position = nn.Parameter(torch.zeros(1, TOKEN_COUNT, dim))
        self.blocks = nn.ModuleList(_FiLMBlockV1(dim, heads) for _ in range(depth))
        self.norm_out = nn.LayerNorm(dim)
        self.to_delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    def forward(
        self, tokens: torch.Tensor, action_one_hot: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (TOKEN_COUNT, self.dim):
            raise ValueError(f"tokens must have shape (B,{TOKEN_COUNT},{self.dim})")
        if action_one_hot.shape[-1] != ACTION_COUNT or command.shape[-1] != COMMAND_DIM:
            raise ValueError("action/command conditioning shape changed")
        condition = self.condition(torch.cat((action_one_hot, command), dim=-1))
        hidden = tokens + self.position
        for block in self.blocks:
            hidden = block(hidden, condition)
        delta = self.to_delta(self.norm_out(hidden))
        return tokens + delta


def token_jepa_objective_v1(
    predicted: torch.Tensor, target: torch.Tensor, current: torch.Tensor
) -> TokenJepaOutputV1:
    """Stop-gradient JEPA loss in token space, with collapse diagnostics.

    ``target`` must already be detached EMA-target tokens.  ``persistence_cosine``
    is the score an identity predictor would obtain and is the baseline any
    non-trivial predictor has to beat.
    """

    if predicted.shape != target.shape or predicted.shape != current.shape:
        raise ValueError("predicted, target and current token shapes differ")
    target = target.detach()
    loss = F.mse_loss(predicted, target)
    cosine = F.cosine_similarity(predicted, target, dim=-1).mean()
    persistence = F.cosine_similarity(current, target, dim=-1).mean()
    # Per-dimension variance across the batch and token axes: a collapsed
    # predictor drives this toward zero.
    variance = predicted.reshape(-1, predicted.shape[-1]).var(dim=0).mean()
    return TokenJepaOutputV1(
        predicted_tokens=predicted,
        jepa_loss=loss,
        cosine_to_target=cosine,
        prediction_variance=variance,
        persistence_cosine=persistence,
    )


def initialize_token_predictor_v1(
    seed: int, *, dim: int = TOKEN_DIM, depth: int = 2, heads: int = 6
) -> TokenPrimaryActionConditionedPredictorV1:
    """Construct the predictor without consuming global RNG."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    model = TokenPrimaryActionConditionedPredictorV1(dim=dim, depth=depth, heads=heads)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "to_delta" not in name:
                nn.init.xavier_uniform_(parameter, gain=1.0, generator=generator)
    return model


__all__ = [
    "ACTION_COUNT",
    "COMMAND_DIM",
    "TOKEN_COUNT",
    "TOKEN_DIM",
    "TOKEN_GRID",
    "TokenJepaOutputV1",
    "TokenPrimaryActionConditionedPredictorV1",
    "initialize_token_predictor_v1",
    "token_jepa_objective_v1",
]
