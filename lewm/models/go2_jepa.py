"""Small JEPA-style visual substrate for Go2 event-slice experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class Go2JepaEncoder(nn.Module):
    """Compact RGB encoder used as the frozen Go2 latent substrate."""

    def __init__(self, latent_dim: int = 96) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class Go2JepaPredictor(nn.Module):
    """Action/aux-conditioned latent predictor."""

    def __init__(self, latent_dim: int, aux_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim) + int(aux_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, latent: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, aux], dim=-1))


class Go2FrontBlockedHead(nn.Module):
    """Binary near-field front-obstacle classifier over frozen Go2 JEPA latents."""

    def __init__(self, latent_dim: int = 96, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).squeeze(-1)


def load_go2_jepa_encoder(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str | None = None,
    freeze: bool = True,
) -> tuple[Go2JepaEncoder, dict[str, Any]]:
    """Load a Go2 JEPA encoder checkpoint and optionally freeze it."""

    map_location = device if device is not None else "cpu"
    try:
        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(Path(checkpoint_path), map_location=map_location)
    latent_dim = int(checkpoint.get("latent_dim", checkpoint.get("args", {}).get("latent_dim", 96)))
    encoder = Go2JepaEncoder(latent_dim=latent_dim)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if device is not None:
        encoder = encoder.to(device)
    if freeze:
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return encoder, checkpoint


def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    """EMA update for target encoder parameters."""

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(float(decay)).add_(
                source_param.data,
                alpha=1.0 - float(decay),
            )
        for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
            target_buffer.copy_(source_buffer)
