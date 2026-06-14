"""Action-conditioned autoregressive prediction over spatial vision tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from .predictor import ActionEmbedder, ConditionalBlock


class SpatialTokenPredictor(nn.Module):
    """Predict future patch-token grids while preserving token correspondence.

    This intentionally mirrors :class:`TransformerPredictor`'s action embedder,
    conditional blocks, dimensions, and depth. The controlled architectural
    change is that attention operates across an ordered spatial token grid
    instead of across a temporal sequence of pooled observations.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        cmd_dim: int = 15,
        num_spatial_tokens: int = 256,
        n_layers: int = 6,
        n_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        if num_spatial_tokens < 1:
            raise ValueError("num_spatial_tokens must be positive")
        self.latent_dim = int(latent_dim)
        self.num_spatial_tokens = int(num_spatial_tokens)
        self.action_embed = ActionEmbedder(
            input_dim=cmd_dim,
            smoothed_dim=10,
            emb_dim=latent_dim,
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, num_spatial_tokens, latent_dim)
        )
        self.input_drop = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    latent_dim,
                    n_heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def predict_step(
        self,
        spatial_tokens: torch.Tensor,
        action_block: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one future spatial-token grid.

        Args:
            spatial_tokens: ``(B, N, D)`` current patch-token grid.
            action_block: ``(B, cmd_dim)`` action executed from that grid.
        """

        if spatial_tokens.ndim != 3:
            raise ValueError(
                "spatial_tokens must have shape (B, N, D), got "
                f"{tuple(spatial_tokens.shape)}"
            )
        if spatial_tokens.shape[1] != self.num_spatial_tokens:
            raise ValueError(
                f"Expected {self.num_spatial_tokens} spatial tokens, got "
                f"{spatial_tokens.shape[1]}"
            )
        if spatial_tokens.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected token dimension {self.latent_dim}, got "
                f"{spatial_tokens.shape[2]}"
            )
        if action_block.ndim != 2 or action_block.shape[0] != spatial_tokens.shape[0]:
            raise ValueError(
                "action_block must have shape (B, cmd_dim) aligned with tokens, got "
                f"{tuple(action_block.shape)}"
            )

        x = self.input_drop(spatial_tokens + self.spatial_pos_embed)
        action = self.action_embed(action_block[:, None, :]).expand_as(x)
        for block in self.blocks:
            x = block(x, action, causal=False)
        return self.norm(x)

    def rollout(
        self,
        start_tokens: torch.Tensor,
        action_sequence: torch.Tensor,
        *,
        teacher_tokens: torch.Tensor | None = None,
        teacher_prob: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressively predict a spatial-token grid for each action block."""

        if action_sequence.ndim != 3:
            raise ValueError(
                "action_sequence must have shape (B, H, cmd_dim), got "
                f"{tuple(action_sequence.shape)}"
            )
        if start_tokens.shape[0] != action_sequence.shape[0]:
            raise ValueError("start_tokens and action_sequence batch sizes differ")
        batch, horizon, _ = action_sequence.shape
        if teacher_tokens is not None:
            expected = (
                batch,
                horizon,
                self.num_spatial_tokens,
                self.latent_dim,
            )
            if tuple(teacher_tokens.shape) != expected:
                raise ValueError(
                    f"teacher_tokens must have shape {expected}, got "
                    f"{tuple(teacher_tokens.shape)}"
                )
        if not 0.0 <= teacher_prob <= 1.0:
            raise ValueError("teacher_prob must lie in [0, 1]")

        current = start_tokens
        predictions = []
        for step in range(horizon):
            predicted = self.predict_step(current, action_sequence[:, step])
            predictions.append(predicted)
            if teacher_tokens is not None and teacher_prob > 0.0:
                use_teacher = (
                    torch.rand(batch, 1, 1, device=predicted.device) < teacher_prob
                )
                current = torch.where(
                    use_teacher,
                    teacher_tokens[:, step],
                    predicted,
                )
            else:
                current = predicted
        return torch.stack(predictions, dim=1)


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``module``."""

    return sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
