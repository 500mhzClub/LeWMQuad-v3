"""Source-conditioned action-utility ranker for Phase 2I diagnostics."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .encoders import VisionEncoder

UtilityInputMode = Literal["source_action", "action_only"]
UtilityFusionMode = Literal["concat", "film_interaction", "interaction_only"]


class SourceActionUtilityRanker(nn.Module):
    """Predict scalar action utility from a source observation and action sequence."""

    def __init__(
        self,
        *,
        cmd_dim: int,
        horizon: int,
        latent_dim: int = 48,
        action_hidden_dim: int = 96,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_depth: int = 2,
        encoder_heads: int = 3,
        encoder_mlp_ratio: int = 2,
        encoder_dropout: float = 0.0,
        input_mode: UtilityInputMode = "source_action",
        fusion_mode: UtilityFusionMode = "concat",
    ):
        super().__init__()
        if cmd_dim < 1:
            raise ValueError("cmd_dim must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if input_mode not in ("source_action", "action_only"):
            raise ValueError(f"unsupported utility input mode: {input_mode}")
        if fusion_mode not in ("concat", "film_interaction", "interaction_only"):
            raise ValueError(f"unsupported utility fusion mode: {fusion_mode}")
        self.cmd_dim = int(cmd_dim)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.input_mode = input_mode
        self.fusion_mode = fusion_mode
        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.horizon * self.cmd_dim),
            nn.Linear(self.horizon * self.cmd_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, latent_dim),
        )
        self.source_conditioner = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 2),
            )
            if fusion_mode == "film_interaction"
            else None
        )
        if fusion_mode == "film_interaction":
            head_input_dim = latent_dim * 3
        elif fusion_mode == "interaction_only":
            head_input_dim = latent_dim
        else:
            head_input_dim = latent_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, 1),
        )

    def forward(self, start_vision: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return one utility logit per source/action candidate row."""

        if start_vision.ndim != 4:
            raise ValueError("start_vision must have shape (B, C, H, W)")
        batch = start_vision.shape[0]
        if tuple(actions.shape[:2]) != (batch, self.horizon):
            raise ValueError(
                f"actions must have shape {(batch, self.horizon, self.cmd_dim)}"
            )
        if actions.shape[2] != self.cmd_dim:
            raise ValueError(f"expected cmd_dim {self.cmd_dim}, got {actions.shape[2]}")
        source = self.encoder.forward_tokens(start_vision)[:, 0]
        if self.input_mode == "action_only":
            source = torch.zeros_like(source)
        action = self.action_encoder(actions.reshape(batch, -1))
        if self.fusion_mode == "film_interaction":
            gamma, beta = self.source_conditioner(source).chunk(2, dim=-1)
            conditioned_action = action * (1.0 + torch.tanh(gamma)) + beta
            features = torch.cat(
                [source, conditioned_action, source * action],
                dim=-1,
            )
        elif self.fusion_mode == "interaction_only":
            features = source * action
        else:
            features = torch.cat([source, action], dim=-1)
        return self.head(features).squeeze(-1)


class SourceActionFactorizedAffordanceModel(nn.Module):
    """Predict factorized consequence labels from a source image and action sequence."""

    def __init__(
        self,
        *,
        cmd_dim: int,
        horizon: int,
        factor_count: int,
        latent_dim: int = 48,
        action_hidden_dim: int = 96,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_depth: int = 2,
        encoder_heads: int = 3,
        encoder_mlp_ratio: int = 2,
        encoder_dropout: float = 0.0,
        input_mode: UtilityInputMode = "source_action",
        fusion_mode: UtilityFusionMode = "concat",
    ):
        super().__init__()
        if cmd_dim < 1:
            raise ValueError("cmd_dim must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if factor_count < 2:
            raise ValueError("factor_count must be at least 2")
        if input_mode not in ("source_action", "action_only"):
            raise ValueError(f"unsupported utility input mode: {input_mode}")
        if fusion_mode not in ("concat", "film_interaction", "interaction_only"):
            raise ValueError(f"unsupported utility fusion mode: {fusion_mode}")
        self.cmd_dim = int(cmd_dim)
        self.horizon = int(horizon)
        self.factor_count = int(factor_count)
        self.latent_dim = int(latent_dim)
        self.input_mode = input_mode
        self.fusion_mode = fusion_mode
        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.horizon * self.cmd_dim),
            nn.Linear(self.horizon * self.cmd_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, latent_dim),
        )
        self.source_conditioner = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim * 2),
            )
            if fusion_mode == "film_interaction"
            else None
        )
        if fusion_mode == "film_interaction":
            head_input_dim = latent_dim * 3
        elif fusion_mode == "interaction_only":
            head_input_dim = latent_dim
        else:
            head_input_dim = latent_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, action_hidden_dim),
            nn.GELU(),
            nn.Linear(action_hidden_dim, factor_count),
        )

    def forward(self, start_vision: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return factor logits per source/action candidate row."""

        if start_vision.ndim != 4:
            raise ValueError("start_vision must have shape (B, C, H, W)")
        batch = start_vision.shape[0]
        if tuple(actions.shape[:2]) != (batch, self.horizon):
            raise ValueError(
                f"actions must have shape {(batch, self.horizon, self.cmd_dim)}"
            )
        if actions.shape[2] != self.cmd_dim:
            raise ValueError(f"expected cmd_dim {self.cmd_dim}, got {actions.shape[2]}")
        source = self.encoder.forward_tokens(start_vision)[:, 0]
        if self.input_mode == "action_only":
            source = torch.zeros_like(source)
        action = self.action_encoder(actions.reshape(batch, -1))
        if self.fusion_mode == "film_interaction":
            gamma, beta = self.source_conditioner(source).chunk(2, dim=-1)
            conditioned_action = action * (1.0 + torch.tanh(gamma)) + beta
            features = torch.cat(
                [source, conditioned_action, source * action],
                dim=-1,
            )
        elif self.fusion_mode == "interaction_only":
            features = source * action
        else:
            features = torch.cat([source, action], dim=-1)
        return self.head(features)
