"""Full-ray radial-context variant of categorical radial perception."""
from __future__ import annotations

import torch
import torch.nn as nn

from .categorical_radial_perception import CategoricalRadialPerception


RADIAL_DILATIONS = (1, 2, 4, 8, 16, 32)
REGISTERED_RADIAL_BIN_COUNT = 64
REGISTERED_TOKEN_FEATURE_DIM = 24
REGISTERED_CONTEXT_DIM = 64
REGISTERED_PARAMETER_COUNT = 2_887_067


def direct_radial_reachability(
    *,
    radial_bin_count: int = REGISTERED_RADIAL_BIN_COUNT,
    dilations: tuple[int, ...] = RADIAL_DILATIONS,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Return clipped layer adjacencies and their direct transitive closure."""

    radial_bin_count = int(radial_bin_count)
    if radial_bin_count <= 0:
        raise ValueError("radial_bin_count must be positive")
    indices = torch.arange(radial_bin_count, dtype=torch.int64)
    output_index = indices[:, None]
    input_index = indices[None, :]
    reachability = torch.eye(radial_bin_count, dtype=torch.bool)
    adjacencies = []
    for raw_dilation in dilations:
        dilation = int(raw_dilation)
        if dilation <= 0:
            raise ValueError("all radial dilations must be positive")
        offset = (output_index - input_index).abs()
        adjacency = (offset == 0) | (offset == dilation)
        adjacencies.append(adjacency)
        reachability = (
            adjacency.to(torch.int64) @ reachability.to(torch.int64)
        ) > 0
    return tuple(adjacencies), reachability


class DilatedRadialContextBlock(nn.Module):
    """One zero-padded residual radial block at a registered dilation."""

    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        if channels <= 0 or channels % 8:
            raise ValueError("channels must be a positive multiple of eight")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.dilation = dilation
        self.radial_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            dilation=(dilation, 1),
            padding=(dilation, 0),
            padding_mode="zeros",
        )
        self.norm = nn.GroupNorm(8, channels)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.radial_conv(features)
        residual = self.norm(residual)
        residual = self.activation(residual)
        residual = self.pointwise(residual)
        return features + residual


class FullRayRadialContext(nn.Sequential):
    """Six dilated residual blocks spanning a complete 64-bin radial ray."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            *(
                DilatedRadialContextBlock(channels, dilation=dilation)
                for dilation in RADIAL_DILATIONS
            )
        )


class CategoricalRadialPerceptionFullRay(CategoricalRadialPerception):
    """V2 model with only its radial-context block replaced."""

    def __init__(
        self,
        *,
        token_feature_dim: int = 24,
        context_dim: int = REGISTERED_CONTEXT_DIM,
    ) -> None:
        if int(token_feature_dim) != REGISTERED_TOKEN_FEATURE_DIM:
            raise ValueError("full-ray token_feature_dim is frozen at 24")
        if int(context_dim) != REGISTERED_CONTEXT_DIM:
            raise ValueError("full-ray context_dim is frozen at 64")
        super().__init__(
            token_feature_dim=token_feature_dim,
            context_dim=context_dim,
        )
        self.radial_context = FullRayRadialContext(self.context_dim)


__all__ = [
    "CategoricalRadialPerceptionFullRay",
    "DilatedRadialContextBlock",
    "FullRayRadialContext",
    "RADIAL_DILATIONS",
    "REGISTERED_CONTEXT_DIM",
    "REGISTERED_PARAMETER_COUNT",
    "REGISTERED_RADIAL_BIN_COUNT",
    "REGISTERED_TOKEN_FEATURE_DIM",
    "direct_radial_reachability",
]
