"""Trainable reachability heads for Phase 3B memory experiments."""
from __future__ import annotations

import torch
from torch import nn


class Phase3BReachabilityHead(nn.Module):
    """Predict dense reachability structure from recurrent egocentric memory."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 96,
        memory_channels: int = 3,
        architecture: str = "conv",
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        if architecture not in {"conv", "dilated"}:
            raise ValueError("architecture must be 'conv' or 'dilated'")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.architecture = str(architecture)
        input_channels = self.memory_channels + 2
        if self.architecture == "conv":
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 4, kernel_size=1),
            )
        else:
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            for dilation in (1, 2, 4, 2, 1):
                layers.extend(
                    [
                        nn.Conv2d(
                            self.hidden_dim,
                            self.hidden_dim,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Conv2d(self.hidden_dim, 4, kernel_size=1))
            self.net = nn.Sequential(*layers)

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dense Phase 3B reachability predictions."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(
                f"expected {self.memory_channels} channels, got {channels}"
            )
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        logits = self.net(torch.cat([memory, self._coordinate_planes(memory)], dim=1))
        return {
            "reachable_logits": logits[:, 0:1],
            "current_distance": logits[:, 1:2],
            "target_distance": logits[:, 2:3],
            "target_value_logits": logits[:, 3:4],
        }

    def _coordinate_planes(self, memory: torch.Tensor) -> torch.Tensor:
        batch = int(memory.shape[0])
        values = torch.linspace(
            -1.0,
            1.0,
            self.memory_size,
            device=memory.device,
            dtype=memory.dtype,
        )
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


def reachability_feature_tensor(
    predictions: dict[str, torch.Tensor],
    *,
    memory_size: int,
) -> torch.Tensor:
    """Convert reachability-head outputs into bounded planner features."""

    reachable = predictions["reachable_logits"].sigmoid()
    current_distance = torch.tanh(
        torch.nn.functional.softplus(predictions["current_distance"])
        / float(memory_size)
    )
    target_distance = torch.tanh(
        torch.nn.functional.softplus(predictions["target_distance"])
        / float(memory_size)
    )
    target_value = predictions["target_value_logits"].sigmoid()
    return torch.cat(
        [
            reachable,
            current_distance,
            target_distance,
            target_value,
        ],
        dim=1,
    )


class Phase3BReachabilityConditionedValueMapPlannerHead(nn.Module):
    """Predict value maps from memory, target evidence, and reachability maps."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 96,
        memory_channels: int = 3,
        reachability_channels: int = 4,
        architecture: str = "conv",
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        if reachability_channels < 1:
            raise ValueError("reachability_channels must be positive")
        if architecture not in {"conv", "dilated"}:
            raise ValueError("architecture must be 'conv' or 'dilated'")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.reachability_channels = int(reachability_channels)
        self.architecture = str(architecture)
        input_channels = self.memory_channels + self.reachability_channels + 4
        if self.architecture == "conv":
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )
        else:
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            for dilation in (1, 2, 4, 8, 4, 2, 1):
                layers.extend(
                    [
                        nn.Conv2d(
                            self.hidden_dim,
                            self.hidden_dim,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Conv2d(self.hidden_dim, 1, kernel_size=1))
            self.net = nn.Sequential(*layers)

    def forward(
        self,
        memory: torch.Tensor,
        target_field: torch.Tensor,
        sparse_probability: torch.Tensor,
        reachability_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return dense value logits conditioned on Phase 3B reachability."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(
                f"expected {self.memory_channels} memory channels, got {channels}"
            )
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if target_field.shape != (batch, 1, height, width):
            raise ValueError(
                "target_field must have shape "
                f"{(batch, 1, height, width)}, got {tuple(target_field.shape)}"
            )
        if reachability_features.shape != (
            batch,
            self.reachability_channels,
            height,
            width,
        ):
            raise ValueError(
                "reachability_features must have shape "
                f"{(batch, self.reachability_channels, height, width)}, got "
                f"{tuple(reachability_features.shape)}"
            )
        if sparse_probability.ndim == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        elif sparse_probability.ndim == 2 and sparse_probability.shape[1] == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        else:
            raise ValueError(
                "sparse_probability must have shape (B,) or (B, 1), got "
                f"{tuple(sparse_probability.shape)}"
            )
        sparse_plane = sparse.to(dtype=memory.dtype, device=memory.device).expand(
            batch,
            1,
            height,
            width,
        )
        coords = self._coordinate_planes(memory)
        inputs = torch.cat(
            [
                memory,
                target_field,
                sparse_plane,
                coords,
                reachability_features,
            ],
            dim=1,
        )
        return self.net(inputs)

    def _coordinate_planes(self, memory: torch.Tensor) -> torch.Tensor:
        batch = int(memory.shape[0])
        values = torch.linspace(
            -1.0,
            1.0,
            self.memory_size,
            device=memory.device,
            dtype=memory.dtype,
        )
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)
