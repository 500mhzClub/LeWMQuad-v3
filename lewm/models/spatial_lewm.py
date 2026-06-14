"""End-to-end JEPA objective for action-predictable spatial vision tokens."""
from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from .encoders import Projector, VisionEncoder
from .sigreg import sigreg_stepwise
from .spatial_predictor import SpatialTokenPredictor


def spatial_variance_floor_loss(
    tokens: torch.Tensor,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Penalize collapsed feature dimensions independently at each patch position."""

    if tokens.ndim != 4:
        raise ValueError(
            "tokens must have shape (B, T, N, D), got "
            f"{tuple(tokens.shape)}"
        )
    if tokens.shape[0] * tokens.shape[1] < 2:
        raise ValueError("spatial variance requires at least two batch-time samples")
    std = torch.sqrt(tokens.float().var(dim=(0, 1), unbiased=False) + eps)
    return torch.relu(float(target_std) - std).mean().to(tokens.dtype)


class TokenProjector(nn.Module):
    """Apply the existing LeWM projector independently to every spatial token."""

    def __init__(self, latent_dim: int = 192):
        super().__init__()
        self.projector = Projector(latent_dim, latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        shape = tokens.shape
        if tokens.ndim < 2:
            raise ValueError("tokens must have at least two dimensions")
        return self.projector(tokens.reshape(-1, shape[-1])).reshape(shape)


class SpatialLeWorldModel(nn.Module):
    """Minimal Phase 2B model that trains spatial state end to end.

    The controlled change from the pooled LeWM is that prediction loss targets
    every ordered patch token. Anti-collapse is factorized: SIGReg is applied
    to the appearance CLS branch, while the spatial branch can use a local
    variance floor without requiring global isotropy.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 192,
        cmd_dim: int = 15,
        pred_layers: int = 6,
        pred_heads: int = 16,
        pred_dim_head: int = 64,
        pred_mlp_dim: int = 2048,
        pred_dropout: float = 0.1,
        image_size: int = 224,
        patch_size: int = 14,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        encoder_mlp_ratio: int = 4,
        encoder_dropout: float = 0.0,
        appearance_sigreg_lambda: float = 0.09,
        spatial_variance_lambda: float = 1.0,
        spatial_target_std: float = 1.0,
        sigreg_projections: int = 1024,
        sigreg_knots: int = 17,
        target_ema_momentum: float | None = None,
    ):
        super().__init__()
        if target_ema_momentum is not None and not 0.0 <= target_ema_momentum < 1.0:
            raise ValueError("target_ema_momentum must lie in [0, 1)")
        self.latent_dim = int(latent_dim)
        self.appearance_sigreg_lambda = float(appearance_sigreg_lambda)
        self.spatial_variance_lambda = float(spatial_variance_lambda)
        self.spatial_target_std = float(spatial_target_std)
        self.sigreg_projections = int(sigreg_projections)
        self.sigreg_knots = int(sigreg_knots)
        self.target_ema_momentum = target_ema_momentum

        self.encoder = VisionEncoder(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=latent_dim,
            depth=encoder_depth,
            n_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.appearance_projector = Projector(latent_dim, latent_dim)
        self.spatial_projector = TokenProjector(latent_dim)
        self.predictor = SpatialTokenPredictor(
            latent_dim=latent_dim,
            cmd_dim=cmd_dim,
            num_spatial_tokens=self.encoder.num_patches,
            n_layers=pred_layers,
            n_heads=pred_heads,
            dim_head=pred_dim_head,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
        )
        self.target_encoder = (
            copy.deepcopy(self.encoder) if target_ema_momentum is not None else None
        )
        self.target_spatial_projector = (
            copy.deepcopy(self.spatial_projector)
            if target_ema_momentum is not None
            else None
        )
        if self.uses_ema_target:
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.target_spatial_projector.parameters():
                parameter.requires_grad_(False)
            self.target_encoder.eval()
            self.target_spatial_projector.eval()

    @property
    def uses_ema_target(self) -> bool:
        return self.target_encoder is not None

    def train(self, mode: bool = True) -> SpatialLeWorldModel:
        """Keep the stop-gradient target modules in evaluation mode."""

        super().train(mode)
        if self.uses_ema_target:
            self.target_encoder.eval()
            self.target_spatial_projector.eval()
        return self

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update the stop-gradient target encoder and projector by EMA."""

        if not self.uses_ema_target:
            return
        momentum = float(self.target_ema_momentum)
        for target_module, online_module in (
            (self.target_encoder, self.encoder),
            (self.target_spatial_projector, self.spatial_projector),
        ):
            for target, online in zip(
                target_module.parameters(),
                online_module.parameters(),
                strict=True,
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)
            for target, online in zip(
                target_module.buffers(),
                online_module.buffers(),
                strict=True,
            ):
                target.copy_(online)

    def encode_seq(self, vision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw CLS and patch-token sequences."""

        if vision.ndim != 5:
            raise ValueError(
                "vision must have shape (B, T, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        batch, steps = vision.shape[:2]
        tokens = self.encoder.forward_tokens(
            vision.reshape(batch * steps, *vision.shape[2:])
        ).reshape(batch, steps, self.encoder.num_patches + 1, self.latent_dim)
        return tokens[:, :, 0], tokens[:, :, 1:]

    @torch.no_grad()
    def target_spatial_seq(self, vision: torch.Tensor) -> torch.Tensor:
        """Return projected spatial targets from the EMA teacher or online model."""

        if not self.uses_ema_target:
            _appearance, spatial_raw = self.encode_seq(vision)
            return self.spatial_projector(spatial_raw)
        if vision.ndim != 5:
            raise ValueError(
                "vision must have shape (B, T, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        batch, steps = vision.shape[:2]
        tokens = self.target_encoder.forward_tokens(
            vision.reshape(batch * steps, *vision.shape[2:])
        ).reshape(batch, steps, self.encoder.num_patches + 1, self.latent_dim)
        return self.target_spatial_projector(tokens[:, :, 1:])

    @torch.no_grad()
    def target_spatial_image(self, vision: torch.Tensor) -> torch.Tensor:
        """Return projected target patch tokens for a batch of images."""

        if vision.ndim != 4:
            raise ValueError(
                "vision must have shape (B, C, H, W), got "
                f"{tuple(vision.shape)}"
            )
        return self.target_spatial_seq(vision[:, None])[:, 0]

    def rollout_spatial(
        self,
        start_spatial_raw: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Return projected free-running spatial-token futures."""

        predicted_raw = self.predictor.rollout(start_spatial_raw, action_sequence)
        return self.spatial_projector(predicted_raw)

    def forward(
        self,
        vision: torch.Tensor,
        cmd_seq: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_latents: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute one-step spatial JEPA prediction and factorized regularisation."""

        appearance_raw, spatial_raw = self.encode_seq(vision)
        batch, steps, num_tokens, dim = spatial_raw.shape
        if steps < 2:
            raise ValueError("spatial JEPA training requires at least two frames")
        if cmd_seq.ndim != 3 or cmd_seq.shape[0] != batch or cmd_seq.shape[1] < steps - 1:
            raise ValueError(
                "cmd_seq must have shape (B, >=T-1, cmd_dim) aligned with vision"
            )

        current = spatial_raw[:, :-1].reshape(-1, num_tokens, dim)
        actions = cmd_seq[:, : steps - 1].reshape(-1, cmd_seq.shape[-1])
        predicted_raw = self.predictor.predict_step(current, actions).reshape(
            batch,
            steps - 1,
            num_tokens,
            dim,
        )
        predicted_proj = self.spatial_projector(predicted_raw)
        spatial_proj = self.spatial_projector(spatial_raw)
        target_spatial_proj = (
            self.target_spatial_seq(vision)[:, 1:]
            if self.uses_ema_target
            else spatial_proj[:, 1:]
        )
        per_transition = (predicted_proj - target_spatial_proj).square().mean(
            dim=(2, 3)
        )
        if mask is not None:
            if tuple(mask.shape) != (batch, steps - 1):
                raise ValueError(
                    f"mask must have shape {(batch, steps - 1)}, got {tuple(mask.shape)}"
                )
            valid = mask.float()
            prediction_loss = (per_transition * valid).sum() / valid.sum().clamp(
                min=1.0
            )
        else:
            prediction_loss = per_transition.mean()

        appearance_proj = self.appearance_projector(
            appearance_raw.reshape(batch * steps, dim)
        ).reshape(batch, steps, dim)
        appearance_sigreg_loss = sigreg_stepwise(
            appearance_proj,
            n_projections=self.sigreg_projections,
            n_knots=self.sigreg_knots,
        )
        spatial_variance_loss = spatial_variance_floor_loss(
            spatial_proj,
            target_std=self.spatial_target_std,
        )
        total = (
            prediction_loss
            + self.appearance_sigreg_lambda * appearance_sigreg_loss
            + self.spatial_variance_lambda * spatial_variance_loss
        )
        output = {
            "loss": total,
            "prediction_loss": prediction_loss,
            "appearance_sigreg_loss": appearance_sigreg_loss,
            "spatial_variance_loss": spatial_variance_loss,
        }
        if return_latents:
            output.update(
                {
                    "appearance_raw": appearance_raw,
                    "appearance_proj": appearance_proj,
                    "spatial_raw": spatial_raw,
                    "spatial_proj": spatial_proj,
                    "target_spatial_proj": target_spatial_proj,
                    "predicted_spatial_proj": predicted_proj,
                }
            )
        return output
